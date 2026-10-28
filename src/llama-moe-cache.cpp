#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"
#include "llama-moe-host.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <chrono>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <vector>

namespace {

// SLRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_slru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t empty    = 0; // slot of the experts that are not cached

    // [n_layer] data of the slot maps, the experts at slot n_slots or later are not cached
    // the experts computed on the CPU read zero slots after the cached ones on the device
    std::vector<int32_t *> slot_map;
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // probation and protected LRU lists; hits promote slots to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;
    int32_t n_filled    = 0; // slots that hold an expert

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;
    std::vector<int32_t> misses;
    std::vector<uint64_t> token_zero; // [n_tokens] zero slots used by each token

    // asynchronous admission: slots being uploaded, by key of their expert; not mapped and never victims until published
    std::vector<bool> pending;
    std::unordered_map<int32_t, int32_t> pend; // il*n_expert + expert -> slot

    // exclusive slots (LLAMA_MOE_CACHE_EXCLUSIVE): the host slot a victim was downloaded into, -1 if none, and whether
    // that download is in flight; kept: the slot is a victim of the current plan
    std::vector<int32_t> staged;
    std::vector<bool>    staging;
    std::vector<bool>    kept;

    void staging_keep(int32_t s) {
        kept[s] = true;
    }

    // whether slot s was marked by staging_keep since the last call, which clears the mark
    bool take_kept(int32_t s) {
        const bool k = kept[s];
        kept[s] = false;
        return k;
    }

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        this->empty    = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        is_protected.assign(n_slots, false);
        seen.assign(n_expert, 0);
        pending.assign(n_slots, false);
        staged.assign(n_slots, -1);
        staging.assign(n_slots, false);
        kept.assign(n_slots, false);
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    void touch(int32_t s) {
        move(s, true);
        if (n_protected > n_slots/2) {
            move(head[1], false);
        }
    }

    int32_t victim(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                if (pending[s]) {
                    continue;
                }
                const int32_t key = key_of[s];
                // keep every expert selected by this request, including earlier fills
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        GGML_ABORT("no SLRU victim");
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, evicting from probation first
    // n_cpu(n) of the n misses are left to the CPU and read a zero slot on the device, they are not cached
    // a zero slot is used at most once per token, a miss that does not get one is cached instead
    // with skip_rank > 0, a miss that every token ranks skip_rank-th or later (ids are sorted by score, best
    // first) reads a zero slot and is left out of the CPU part too: it is skipped, an approximation
    // returns false if the ids select more distinct experts than there are slots
    template <typename F>
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int32_t n_used, int32_t n_zero, int32_t skip_rank,
            F && n_cpu, uint64_t & rng, std::vector<fill> & fills, std::vector<fill> & cpu, size_t & n_hit, size_t & n_skip,
            size_t max_admit, std::vector<fill> & admits) {
        fills.clear();
        cpu.clear();
        admits.clear();
        n_hit  = 0;
        n_skip = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // promote hits before choosing victims for misses
        misses.clear();
        for (int32_t e : uniq) {
            if (slots[e] < n_slots) {
                touch(slots[e]);
                n_hit++;
            } else {
                misses.push_back(e);
            }
        }

        // the zero slot of a miss: the first one none of its tokens reads yet, or -1 if there is none left
        const size_t n_tokens = n_ids/n_used;
        bool zero_init = false;
        auto take_zero = [&](int32_t e) -> int32_t {
            if (!zero_init) {
                token_zero.assign(n_tokens, 0);
                zero_init = true;
            }
            uint64_t used = 0;
            for (size_t it = 0; it < n_tokens; ++it) {
                for (int32_t k = 0; k < n_used; ++k) {
                    if (ids[it*n_used + k] == e) {
                        used |= token_zero[it];
                    }
                }
            }
            const int32_t z = used == ~0ull ? 64 : __builtin_ctzll(~used);
            if (z >= n_zero) {
                return -1;
            }
            for (size_t it = 0; it < n_tokens; ++it) {
                for (int32_t k = 0; k < n_used; ++k) {
                    if (ids[it*n_used + k] == e) {
                        token_zero[it] |= 1ull << z;
                    }
                }
            }
            return n_slots + z;
        };

        if (skip_rank > 0 && skip_rank < n_used && n_zero > 0) {
            for (size_t i = 0; i < misses.size(); ) {
                const int32_t e = misses[i];
                bool tail = true;
                for (size_t j = 0; j < n_ids && tail; ++j) {
                    tail = ids[j] != e || (int32_t) (j % n_used) >= skip_rank;
                }
                const int32_t s = tail ? take_zero(e) : -1;
                if (s < 0) {
                    i++;
                    continue;
                }
                slots[e] = s;
                misses[i] = misses.back();
                misses.pop_back();
                n_skip++;
            }
        }

        // the misses left to the CPU are picked at random, so that every expert can enter the cache; with asynchronous
        // admission all of them go to the CPU, and the share that would be filled is admitted in the background
        const bool    async   = max_admit > 0;
        const size_t  n_share = std::min(n_cpu(misses.size()), misses.size());
        const size_t  n_c     = async ? misses.size() : n_share;
        size_t        n_admit = async ? std::min(misses.size() - n_share, max_admit) : 0;
        for (size_t i = 0; i < n_c; ++i) {
            rng = rng*6364136223846793005ull + 1442695040888963407ull;
            std::swap(misses[i], misses[i + (rng >> 33) % (misses.size() - i)]);
        }
        if (n_c > 0) {
            size_t n_keep = n_c;
            for (size_t i = 0; i < n_keep; ) {
                const int32_t e = misses[i];
                const int32_t s = take_zero(e);
                if (s < 0) {
                    std::swap(misses[i], misses[--n_keep]);
                    continue;
                }
                slots[e] = s;
                cpu.push_back({ e, s });
                i++;
            }
            misses.erase(misses.begin(), misses.begin() + n_keep);
        }

        // admit some of the experts left to the CPU: each takes a victim slot now, pending until its upload is done
        for (size_t i = cpu.size(); i-- > 0 && n_admit > 0; ) {
            const int32_t e   = cpu[i].expert;
            const int32_t key = il*n_expert + e;
            if (pend.count(key) > 0) {
                continue;
            }
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = empty;
            } else {
                n_filled++;
            }
            key_of[s]  = key;
            pending[s] = true;
            pend[key]  = s;
            move(s, false);
            admits.push_back({ e, s });
            n_admit--;
        }

        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(misses.begin(), misses.end());
        for (int32_t e : misses) {
            // an expert still being admitted is filled into its own slot, with the same bytes
            if (const auto it = pend.find(il*n_expert + e); it != pend.end()) {
                const int32_t s = it->second;
                pending[s] = false;
                pend.erase(it);
                slots[e] = s;
                fills.push_back({ e, s });
                continue;
            }
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = empty;
            } else {
                n_filled++;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            move(s, false);
            fills.push_back({ e, s });
        }
        return true;
    }

    // an admitted expert whose upload is done becomes a hit, unless it was filled since
    void publish(int32_t il, int32_t e, int32_t s) {
        const int32_t key = il*n_expert + e;
        const auto it = pend.find(key);
        if (it == pend.end() || it->second != s || !pending[s] || key_of[s] != key) {
            return;
        }
        pending[s] = false;
        pend.erase(it);
        slot_map[il][e] = s;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl : public llama_moe_host::listener {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots  = 0;
        int32_t n_core   = 0; // with phases, the slots kept while large batches run
        int32_t n_active = 0; // the slots in use in the current phase
        int32_t n_excl   = 0; // exclusive core: the first slots, whose experts have no host copy
        std::vector<int32_t> spares;     // host slots that hold no expert
        std::vector<int32_t> quarantine; // host slots of experts that entered exclusive slots, spares from the next request
        moe_cache_slru slru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * cpu_map  = nullptr;   // I32 [1, n_expert] in host memory, if cpu_share > 0
        std::vector<int32_t> cpu_experts;   // experts set in cpu_map
        uint32_t cpu_acc = 0;               // fraction of a miss carried over to the next plan, in 1/65536
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t cpu    = 0; // misses computed on the CPU
        size_t skip   = 0; // misses skipped by the tail skip, counted in misses
        size_t bytes  = 0;
    };

    // device data read by the CPU part of a layer
    struct stash_entry {
        const ggml_tensor * base; // the tensor that owns the data
        size_t offs;              // offset of the data in base
        size_t size;
        size_t pos;               // position in stash_buf
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches
    size_t n_admitted  = 0; // experts admitted in the background
    size_t async_bytes = 0;

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_slru::fill> fills;
    std::vector<moe_cache_slru::fill> cpu;
    std::vector<moe_cache_slru::fill> admits;

    // asynchronous admission (LLAMA_MOE_CACHE_ASYNC): uploads into pending slots on a second stream of each device
    struct upload {
        int32_t       dev;
        ggml_tensor * bank;
        const void  * src;
        size_t        offs;
        size_t        size;
    };
    struct publication {
        int32_t ig, il, expert, slot;
    };
    struct admit_job {
        int32_t                  dev;
        std::vector<upload>      uploads;
        std::vector<publication> pubs;
        bool                     gate    = true;  // wait for the work queued on the compute stream before the job
        bool                     readmit = false; // fills lent slots again (readmit_lent)
    };
    bool    async_admit = false;
    size_t  async_max   = 96;   // experts in flight
    size_t  in_flight   = 0;
    std::vector<ggml_backend_t>       side;     // per device
    std::vector<ggml_backend_event_t> gate_ev;  // per device, recorded on the compute stream before an admission
    std::vector<ggml_backend_event_t> done_ev;  // per device, recorded on the side stream after one
    std::thread                       worker;
    std::mutex                        mtx;
    std::condition_variable           cv;
    std::deque<admit_job>             jobs;
    std::vector<publication>          done;
    bool                              quit = false;

    // host mode (LLAMA_MOE_HOST): the device posts the experts it does not cache to the host, see llama-moe-host
    // the cache then follows the selection counts the host gathers: every host_period graphs, the experts selected most
    // that are not cached are uploaded in the background, into free slots or over cached experts selected less
    bool     host_mode       = false;
    int32_t  host_max_tokens = 32;
    int32_t  host_period     = 4;     // graphs between two plans
    int32_t  host_max_swaps  = 32;    // cached experts replaced per plan
    int32_t  host_max_fills  = 1024;  // experts admitted into free slots per plan
    float    host_decay      = 0.97f; // of the selection counts, per plan
    float    host_hyst       = 1.25f; // a replaced expert was selected this many times less than its successor
    uint64_t host_n_begin    = 0;
    size_t   host_n_swaps    = 0;
    size_t   host_n_fills    = 0;
    int64_t  host_begin_ns   = 0; // spent before the graphs, publishing and planning
    int64_t  host_n_plans    = 0;
    int32_t  host_n_expert   = 0;
    int32_t  host_n_embd     = 0;
    std::unique_ptr<llama_moe_host> host;
    llama_moe_host::mailbox_layout  host_lay;
    struct host_device {
        ggml_tensor * state   = nullptr;
        void *        mailbox = nullptr;
        void (*mailbox_free)(void *) = nullptr;
        int32_t * table_host = nullptr; // [n_layer*n_expert] in the mailbox, read by the graphs started afterwards
        size_t in_flight = 0;           // experts being uploaded

        // large batches shared with the host (init_batch): their own mailbox, state and table, -1 for the experts the
        // host computes
        ggml_context_ptr        batch_ctx;
        ggml_backend_buffer_ptr batch_buf;
        ggml_tensor *           batch_state   = nullptr;
        void *                  batch_mailbox = nullptr;
        int32_t *               batch_table   = nullptr;
    };
    std::vector<host_device> hdev;

    // the policy of the service thread (host mode, LLAMA_MOE_HOST_POLICY=service, the default): after the last layer
    // of a graph it plans the swaps of the next graph from the selection counts; while the device waits for the answer
    // of a layer it publishes the uploads done and evicts the victims of a few swaps, of other layers; right after the
    // answer it uploads them, while the device computes up to the next request and the CPU has nothing to do
    bool    svc_policy   = false;
    int32_t svc_max_il   = -1;  // the last cached layer, the end of a graph
    int32_t svc_swaps    = 96;  // replacements per graph
    int32_t svc_fills    = 192; // admissions into free slots per graph
    int32_t svc_per_req  = 2;   // swaps started per request
    struct svc_move {
        float   benefit;
        int32_t ig;
        int32_t key;  // the expert admitted
        int32_t slot; // its slot, free or holding the victim
        bool    started;
    };
    std::vector<svc_move>    svc_plan;
    size_t                   svc_pos = 0;
    std::vector<publication> svc_started; // evicted, not yet given to the upload worker
    std::mutex               svc_mtx;
    size_t svc_n_swaps = 0;
    size_t svc_n_fills = 0;

    // phases (LLAMA_MOE_CACHE_PREFILL_MIB, host mode): between large batches each group holds n_slots experts; while
    // large batches run it holds the first n_core and lends the memory of the others, released from the end of each
    // bank, to the compute buffers and the expert prefetch of the large batches
    bool    phases       = false;
    bool    prefill      = true; // the current phase, a context starts with large batches
    std::vector<publication> lent; // the experts of the lent slots, admitted again into them when small batches return
    size_t  n_readmitted = 0;

    // large batches shared with the host (host mode with phases): for the graph of a large batch the host computes the
    // least selected of the experts the core lacks, for all the tokens of the batch, while the device uploads and
    // computes the others. Each layer gives the host its least selected lacking experts for as long as the host's time
    // for them, measured per kind of layer (llama_moe_host::get_batch_cost), stays within the time of the uploads left,
    // at the upload rate the prefetch measures, or within the device's own time for the layer, measured on the batches
    // neither bounds; LLAMA_MOE_BATCH_CPU=<share> fixes the share of the lacking experts instead, 0 turns sharing off
    llama_moe_host::mailbox_layout batch_lay;
    int32_t              batch_max     = 0;
    int64_t              batch_tokens  = 0;     // the batch the host takes part in, 0 if none
    float                batch_force   = -1.0f; // LLAMA_MOE_BATCH_CPU=<share>: a fixed share of the experts the core lacks
    float                batch_cap     = 1.0f;  // LLAMA_MOE_BATCH_CPU_MAX: the largest share
    int                  batch_threads = 1;
    std::vector<uint8_t> batch_cpu;             // [n_layer*n_expert] the host computes the expert in the current batch
    size_t               batch_n_cpu   = 0;
    uint64_t             batch_graphs  = 0;
    double               batch_share   = 0.0;   // sum of the shares, for the log
    double               batch_c_tok   = 0.5e-3; // the device's seconds per token of a large batch
    // the selections of each expert in the last shared batches (llama_moe_host::take_batch_counts), fading with each
    // batch: they predict the entries of a large batch better than the profile, counted in decoding. A layer uses them
    // once they hold a few selections per expert, the profile until then; for each source, the entries the host
    // finds against the ones it expects (the experts least selected in the last batches are selected more in the
    // next one)
    std::vector<float>   batch_pop;
    double               batch_ratio[2] = { 1.0, 1.5 }; // from the profile, from batch_pop
    int                  batch_src     = 0;     // the source of the batch begun, when it uses only one
    // the measured batch begun, for end_batch: its tokens (0 if none), the entries expected for the host's experts,
    // the host's totals before it, and its predicted seconds without and with the host
    int64_t              batch_begun   = 0;
    double               batch_expected = 0.0;
    double               batch_host_s0 = 0.0;
    uint64_t             batch_host_e0 = 0;
    double               batch_host_w0 = 0.0;
    double               batch_pred[2] = {};

    // exclusive core (LLAMA_MOE_CACHE_EXCLUSIVE=<MiB>, host mode with phases and the service policy, GGML_CUDA_HOST_MEMFD
    // expert buffers): the experts held by the first n_excl slots of each group, up to that size, have no host copy, as
    // in a disjoint cache. A host copy has no fixed place: host slot hloc[key] holds it, an expert position of any layer
    // of the expert's group (they all have the same size), and every host reader takes its address from hptr. At start
    // the profile's experts fill those slots, then each layer is compacted, the experts with a host copy taking its first
    // positions, and the last ones are given back to the system but for a few spare host slots per group. A swap whose
    // victim holds an exclusive slot exchanges: the victim is downloaded into a spare first (staged), only then evicted,
    // and the host slot of the expert admitted in its place becomes a spare from the request after its publication. The
    // host memory stays as it is from start on, and no host registration changes.
    bool    excl_on       = false;
    size_t  excl_released = 0;    // bytes given back at start
    int32_t excl_spares   = 16;   // spare host slots per group (LLAMA_MOE_CACHE_EXCLUSIVE_SPARES)
    int32_t hp_nm         = 0;    // expert tensors per layer
    int32_t hp_ne         = 0;    // experts per layer
    std::vector<const char *> hptr; // [n_layer*hp_nm*hp_ne] host address of each expert tensor's expert
    std::vector<int32_t>      hloc; // [n_layer*hp_ne] host slot of each expert, il*n_expert + position, -1 if none
    struct stage {
        int32_t ig, slot, key, loc;
    };
    struct stage_job {
        int32_t            dev;
        std::vector<stage> stages;
    };
    std::vector<ggml_backend_t> excl_side;  // per device, the downloads
    void (*excl_sync)(ggml_backend_t) = nullptr; // waits for a download, sleeping if the backend can
    std::thread                 excl_worker;
    std::deque<stage_job>       excl_jobs;
    std::vector<stage>          excl_done;
    std::condition_variable     excl_cv;
    bool                        excl_quit      = false;
    int32_t                     excl_in_flight = 0; // downloads not collected yet
    size_t excl_n_exchanges = 0; // victims of exclusive slots evicted with their downloaded copies
    size_t excl_n_staged    = 0;
    size_t excl_n_unstaged  = 0; // downloads no longer needed
    size_t excl_n_waited    = 0; // swaps of exclusive victims a plan ended without starting, none downloaded

    // LLAMA_MOE_CACHE_PROFILE=<file>, host mode: the selection counts of the past sessions; the cache starts with the
    // experts they count most, and the counts of this session are added to the file when the cache is destroyed
    std::string        profile_path;
    std::vector<float> profile;     // [n_layer*n_expert] read from the file, empty without one
    std::vector<float> profile_new; // [n_layer*n_expert] counted in this session
    size_t  prefill_size = 0;
    std::vector<ggml_backend_buffer_t> bank_bufs; // one buffer per bank, its physical memory mapped up to the phase's slots
    std::vector<ggml_tensor *>         bank_tensors;
    std::vector<int32_t>               bank_group;
    size_t (*vmm_commit)(ggml_backend_buffer_t, size_t) = nullptr;

    // phases: the banks stay mapped whole, and the memory of the slots they lend is mapped a second time, one range per
    // device, which a lent buffer type gives to the compute buffers and the staging slots of large batches while the
    // cache lends (lend): a change of phase maps and unmaps nothing but the memory those buffers need beyond the
    // range, mapped after it. LLAMA_MOE_CACHE_ALIAS=0 releases and maps the lent slots again at each change instead.
    struct alias_range {
        ggml_backend_t             backend  = nullptr;
        int                        device   = -1; // the index of the device in its backend registry
        void *                     ptr      = nullptr;
        size_t                     size     = 0;
        size_t                     capacity = 0;  // the address space reserved, past which nothing is mapped
        ggml_backend_buffer_type_t buft     = nullptr;
    };
    std::vector<alias_range> aliases;
    void (*alias_free)(int, void *, size_t, size_t) = nullptr;
    void (*lent_set)(ggml_backend_buffer_type_t, bool) = nullptr;
    void (*lent_free)(ggml_backend_buffer_type_t) = nullptr;
    std::vector<float>   host_score; // [n_layer*n_expert] decayed selection counts
    std::vector<int32_t> host_cand;
    std::vector<int32_t> host_vict;
    std::vector<int32_t> host_free;

    float    cpu_share = 0.0f;
    int32_t  skip_rank = 0; // LLAMA_MOE_CACHE_TAIL_SKIP, 0 = off
    uint32_t cpu_share_q = 0; // cpu_share in 1/65536, integer so that a simulator can reproduce the rounding
    int32_t  n_zero    = 0; // zero slots per group
    uint64_t rng       = 0x853c49e6748fea9bull;

    // the CPU part of the layer of a slot lookup
    struct cpu_part {
        int32_t n_used; // experts per token
        std::vector<const ggml_tensor *> inputs; // device tensors read by the CPU part
    };
    std::unordered_map<const ggml_tensor *, cpu_part> cpu_parts; // slot lookup -> CPU part, no inputs if there is no CPU part
    std::unordered_set<const ggml_tensor *> cpu_outputs; // results of the CPU parts, copied to the device asynchronously

    // the device data read with the selected experts of the last slot lookup, in host memory that the device can copy to asynchronously
    std::vector<stash_entry> stash;
    ggml_backend_buffer_ptr stash_buf;

    // LLAMA_MOE_CACHE_TRACE: the selected experts of each slot lookup, for a simulator of the cache
    FILE * trace = nullptr;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()), cpu_share(std::clamp(cpu_share, 0.0f, 1.0f)) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        if (const char * v = getenv("LLAMA_MOE_HOST"); v != nullptr && atoi(v) != 0 && !no_alloc) {
            host_mode = host_supported(model);
        }
        if (const char * v = getenv("LLAMA_MOE_CACHE_PREFILL_MIB"); v != nullptr && *v != '\0' && host_mode) {
            prefill_size = (size_t) std::max(0ll, atoll(v))*1024*1024;
            phases       = prefill_size < size;
        }

        // with misses on the CPU, the n_zero slots after the cached ones stay zero: the device computes these misses as zero
        // one extra slot at the end, CUDA MMQ can read past the last expert
        n_zero = cpu_share > 0.0f || host_mode ? std::min<int32_t>(n_expert_used, 64) : 0;
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + n_zero + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        cpu_share_q = (uint32_t) std::lround(cpu_share*65536.0f);
        if (const char * v = getenv("LLAMA_MOE_CACHE_ASYNC"); v != nullptr && atoi(v) != 0 && !host_mode) {
            if (n_zero == 0) {
                LLAMA_LOG_WARN("%s: LLAMA_MOE_CACHE_ASYNC needs --moe-cache-cpu-share > 0, ignored\n", __func__);
            } else {
                async_admit = true;
                if (const char * m = getenv("LLAMA_MOE_CACHE_ASYNC_MAX"); m != nullptr) {
                    async_max = std::max(1, atoi(m));
                }
                LLAMA_LOG_INFO("%s: asynchronous admission, up to %zu experts in flight\n", __func__, async_max);
            }
        }
        if (async_admit || host_mode) {
            for (const device & d : devices) {
                ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
                side.push_back(ggml_backend_dev_init(dev, nullptr));
                gate_ev.push_back(ggml_backend_event_new(dev));
                done_ev.push_back(ggml_backend_event_new(dev));
                GGML_ASSERT(side.back() != nullptr && gate_ev.back() != nullptr && done_ev.back() != nullptr);
            }
            worker = std::thread([this] { run_admissions(); });
        }
        if (const char * v = getenv("LLAMA_MOE_CACHE_TAIL_SKIP"); v != nullptr && *v != '\0') {
            skip_rank = std::max(0, atoi(v));
            if (skip_rank > 0 && n_zero == 0) {
                LLAMA_LOG_WARN("%s: LLAMA_MOE_CACHE_TAIL_SKIP needs --moe-cache-cpu-share > 0, ignored\n", __func__);
                skip_rank = 0;
            } else if (skip_rank > 0) {
                LLAMA_LOG_INFO("%s: tail skip: misses every token ranks %d-th or later are skipped\n", __func__, skip_rank + 1);
            }
        }
        if (const char * path = getenv("LLAMA_MOE_CACHE_TRACE"); path != nullptr && *path != '\0') {
            trace = fopen(path, "wb");
            if (trace == nullptr) {
                throw std::runtime_error(std::string("failed to open the MoE cache trace ") + path);
            }
            write_trace_header(split_sum);
        }
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.n_core = g.n_slots;
            if (phases) {
                const size_t budget_core = (size_t) ((double) prefill_size*d.split/split_sum*g.host_bytes/d.host_bytes);
                g.n_core = 0;
                while (g.n_core < g.n_slots && alloc_size(g, g.n_core + 1) <= budget_core) {
                    g.n_core++;
                }
            }
            g.n_active = phases ? g.n_core : g.n_slots;
            g.slru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(n_zero > 0 ? 2 : 1); // slot map and CPU map
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }
        if (host_mode) {
            hdev.resize(devices.size());
            for (size_t id = 0; id < devices.size(); ++id) {
                if (n_tensors[id] > 0) {
                    n_tensors[id] += 1; // mailbox state
                }
            }
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_zero + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + n_zero, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (n_zero > 0) {
                    l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }
        host_n_expert = groups[0].ref[0]->ne[2];
        for (size_t id = 0; id < hdev.size(); ++id) {
            if (!devices[id].ctx) {
                continue;
            }
            hdev[id].state = ggml_new_tensor_1d(devices[id].ctx.get(), GGML_TYPE_I32, 8);
            ggml_format_name(hdev[id].state, "moe_cache.host_state");
        }

        // with no_alloc, dummy buffers are attached so the memory use can still be measured, see llama_context::memory_breakdown
        if (!no_alloc && phases) {
            phases_alloc();
        }
        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            // with phases the banks have their own buffers already: this allocates the rest and the views
            d.buf = llama_memory_bufs::alloc_buf(d.ctx.get(), d.buft, no_alloc, "MoE cache");
            if (!no_alloc) {
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
        }
        buf_host = llama_memory_bufs::alloc_buf(ctx_host.get(), buft_host, no_alloc, "MoE cache slot maps");
        if (!no_alloc) {
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // the CPU maps are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        int32_t * slots = (int32_t *) layers[il].slot_map->data;
                        std::fill(slots, slots + layers[il].slot_map->ne[1], g.slru.empty);
                        g.slru.slot_map[il] = slots;
                    }
                }
            }

            if (host_mode) {
                host_init(model);
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts, %.0f%% of the misses on the CPU\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0, 100.0*cpu_share);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
    }

    // one buffer per bank, its physical memory mapped up to the core slots: the phase of large batches
    void phases_alloc() {
        ggml_backend_buffer_t (*vmm_alloc)(int, size_t, size_t) = nullptr;
        const char * alias_env = getenv("LLAMA_MOE_CACHE_ALIAS");
        const bool   alias_on  = alias_env == nullptr || *alias_env == '\0' || atoi(alias_env) != 0;
        std::vector<int> bank_index; // the registry index of each bank's device
        for (size_t ig = 0; ig < groups.size() && phases; ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            const device & d = devices[g.id];
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            vmm_alloc  = (decltype(vmm_alloc))  ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_alloc");
            vmm_commit = (decltype(vmm_commit)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_commit");
            int index = -1;
            for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                if (ggml_backend_reg_dev_get(reg, i) == dev) {
                    index = (int) i;
                }
            }
            if (vmm_alloc == nullptr || vmm_commit == nullptr || index < 0) {
                LLAMA_LOG_WARN("%s: %s cannot release part of a buffer, the MoE cache has no phases\n", __func__, ggml_backend_name(d.backend));
                phases = false;
                break;
            }
            for (ggml_tensor * bank : g.banks) {
                const size_t size = ggml_backend_buft_get_alloc_size(d.buft, bank);
                const size_t core = (size_t) g.n_core*bank->nb[2];
                ggml_backend_buffer_t buf = vmm_alloc(index, size, alias_on ? size : core);
                if (buf == nullptr) {
                    LLAMA_LOG_WARN("%s: failed to allocate a bank of the MoE cache with phases, the MoE cache has no phases\n", __func__);
                    phases = false;
                    break;
                }
                // the padding past the bank is not mapped yet: as a compute buffer, the allocation does not zero it, and
                // nothing reads it (the extra slot at the end takes what MMQ reads past the last expert)
                ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                ggml_backend_tensor_alloc(buf, bank, ggml_backend_buffer_get_base(buf));
                ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                if (core > 0) {
                    ggml_backend_tensor_memset(bank, 0, 0, core);
                }
                bank_bufs.push_back(buf);
                bank_tensors.push_back(bank);
                bank_group.push_back((int32_t) ig);
                bank_index.push_back(index);
            }
        }
        if (phases && alias_on && !alias_banks(bank_index)) {
            // the lent slots go back to the system while large batches run, as without the ranges
            for (size_t i = 0; i < bank_bufs.size(); ++i) {
                vmm_commit(bank_bufs[i], (size_t) groups[bank_group[i]].n_core*bank_tensors[i]->nb[2]);
            }
        }
        if (!phases) {
            // a whole cache instead: the regular allocation takes the banks
            for (size_t i = 0; i < bank_bufs.size(); ++i) {
                bank_tensors[i]->data   = nullptr;
                bank_tensors[i]->buffer = nullptr;
                ggml_backend_buffer_free(bank_bufs[i]);
            }
            bank_bufs.clear();
            bank_tensors.clear();
            bank_group.clear();
            for (group & g : groups) {
                g.n_core   = g.n_slots;
                g.n_active = g.n_slots;
            }
            return;
        }
        prefill = true;
        size_t size_all  = 0;
        size_t size_core = 0;
        for (size_t i = 0; i < bank_bufs.size(); ++i) {
            size_all  += ggml_backend_buffer_get_size(bank_bufs[i]);
            size_core += (size_t) groups[bank_group[i]].n_core*bank_tensors[i]->nb[2];
        }
        LLAMA_LOG_INFO("%s: MoE cache phases: %.2f MiB between large batches, %.2f MiB while they run\n", __func__,
            size_all/1024.0/1024.0, size_core/1024.0/1024.0);
    }

    // the second mapping of the lent slots of each device's banks, and the lent buffer type over it; false if the
    // backend cannot, nothing kept then
    bool alias_banks(const std::vector<int> & bank_index) {
        void * (*vmm_alias)(ggml_backend_buffer_t *, const size_t *, int, size_t *, size_t *) = nullptr;
        ggml_backend_buffer_type_t (*lent_new)(int, void *, size_t, size_t) = nullptr;
        for (size_t i = 0; i < bank_bufs.size(); ++i) {
            const int32_t id = groups[bank_group[i]].id;
            alias_range * ar = nullptr;
            for (alias_range & a : aliases) {
                if (a.backend == devices[id].backend) {
                    ar = &a;
                }
            }
            if (ar != nullptr) {
                continue;
            }
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(devices[id].backend));
            vmm_alias  = (decltype(vmm_alias))  ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_alias");
            alias_free = (decltype(alias_free)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_alias_free");
            lent_new   = (decltype(lent_new))   ggml_backend_reg_get_proc_address(reg, "ggml_backend_lent_buffer_type_new");
            lent_set   = (decltype(lent_set))   ggml_backend_reg_get_proc_address(reg, "ggml_backend_lent_buffer_type_set");
            lent_free  = (decltype(lent_free))  ggml_backend_reg_get_proc_address(reg, "ggml_backend_lent_buffer_type_free");
            if (vmm_alias == nullptr || alias_free == nullptr || lent_new == nullptr || lent_set == nullptr || lent_free == nullptr) {
                LLAMA_LOG_WARN("%s: %s cannot map the lent slots again, they are released at each change of phase\n", __func__, ggml_backend_name(devices[id].backend));
                aliases_free();
                return false;
            }
            std::vector<ggml_backend_buffer_t> bufs;
            std::vector<size_t> from;
            for (size_t j = 0; j < bank_bufs.size(); ++j) {
                if (groups[bank_group[j]].id == id) {
                    bufs.push_back(bank_bufs[j]);
                    from.push_back((size_t) groups[bank_group[j]].n_core*bank_tensors[j]->nb[2]);
                }
            }
            alias_range a;
            a.backend = devices[id].backend;
            a.device  = bank_index[i];
            a.ptr     = vmm_alias(bufs.data(), from.data(), (int) bufs.size(), &a.size, &a.capacity);
            if (a.ptr == nullptr) {
                LLAMA_LOG_WARN("%s: failed to map the lent slots again, they are released at each change of phase\n", __func__);
                aliases_free();
                return false;
            }
            a.buft = lent_new(a.device, a.ptr, a.size, a.capacity);
            // the context starts with large batches
            lent_set(a.buft, true);
            aliases.push_back(a);
            LLAMA_LOG_INFO("%s: %s: the compute buffers of large batches take %.2f MiB the cache lends\n", __func__,
                ggml_backend_name(a.backend), a.size/1024.0/1024.0);
        }
        return true;
    }

    void aliases_free() {
        for (alias_range & a : aliases) {
            if (a.buft != nullptr) {
                lent_free(a.buft);
            }
            alias_free(a.device, a.ptr, a.size, a.capacity);
        }
        aliases.clear();
    }

    ggml_backend_buffer_type_t lent_buft(ggml_backend_t backend) const {
        for (const alias_range & a : aliases) {
            if (a.backend == backend) {
                return a.buft;
            }
        }
        return nullptr;
    }

    void lend(bool on) {
        for (alias_range & a : aliases) {
            lent_set(a.buft, on);
        }
    }

    // the slots [from, to) of group ig no longer hold an expert
    void evict_range(size_t ig, int32_t from, int32_t to) {
        group & g = groups[ig];
        moe_cache_slru & sl = g.slru;
        const int32_t ne = sl.n_expert;
        int32_t * table = g.id < (int32_t) hdev.size() ? hdev[g.id].table_host : nullptr;
        for (int32_t s = from; s < to; ++s) {
            const int32_t key = sl.key_of[s];
            if (sl.pending[s]) {
                sl.pending[s] = false;
                if (key >= 0) {
                    sl.pend.erase(key);
                }
            }
            if (key < 0) {
                continue;
            }
            sl.slot_map[key / ne][key % ne] = sl.empty;
            if (table != nullptr) {
                table[key] = -1;
            }
            sl.key_of[s] = -1;
            sl.n_filled--;
        }
    }

    // large batches run with the core slots, lending the memory of the others; small ones get them back
    void set_phase(bool large) {
        if (!phases || large == prefill) {
            return;
        }
        if (large) {
            // the re-admissions not started yet are dropped rather than waited for: their slots are lent again, and
            // their experts admitted again after these large batches
            {
                std::lock_guard<std::mutex> lk(mtx);
                for (auto it = jobs.begin(); it != jobs.end(); ) {
                    if (!it->readmit) {
                        ++it;
                        continue;
                    }
                    hdev[it->dev].in_flight -= it->pubs.size();
                    n_readmitted -= it->pubs.size();
                    for (const upload & up : it->uploads) {
                        async_bytes -= up.size;
                    }
                    it = jobs.erase(it);
                }
            }
            if (svc_policy) {
                svc_settle();
            }
            // the admissions in flight may upload past the core
            for (;;) {
                size_t n = 0;
                for (const host_device & hd : hdev) {
                    n += hd.in_flight;
                }
                if (n == 0) {
                    break;
                }
                host_collect();
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            // a slot still pending holds a dropped re-admission: its expert is lent again
            lent.clear();
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                const moe_cache_slru & sl = groups[ig].slru;
                for (int32_t s = groups[ig].n_core; s < groups[ig].n_slots; ++s) {
                    const int32_t key = sl.key_of[s];
                    if (key >= 0) {
                        lent.push_back({ (int32_t) ig, key / sl.n_expert, key % sl.n_expert, s });
                    }
                }
            }
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                if (groups[ig].n_slots > 0) {
                    evict_range(ig, groups[ig].n_core, groups[ig].n_slots);
                    groups[ig].n_active = groups[ig].n_core;
                }
            }
            for (const device & d : devices) {
                ggml_backend_synchronize(d.backend);
            }
            for (size_t i = 0; i < bank_bufs.size() && aliases.empty(); ++i) {
                vmm_commit(bank_bufs[i], (size_t) groups[bank_group[i]].n_core*bank_tensors[i]->nb[2]);
            }
        } else {
            for (size_t i = 0; i < bank_bufs.size(); ++i) {
                ggml_tensor * bank = bank_tensors[i];
                const size_t full = ggml_backend_buffer_get_size(bank_bufs[i]);
                const size_t core = (size_t) groups[bank_group[i]].n_core*bank->nb[2];
                if (aliases.empty() && vmm_commit(bank_bufs[i], full) < full) {
                    GGML_ABORT("the MoE cache could not map its banks again");
                }
                // the zero slots after the cached ones must read as zero
                ggml_backend_tensor_memset(bank, 0, core, ggml_nbytes(bank) - core);
            }
            for (group & g : groups) {
                g.n_active = g.n_slots;
            }
            readmit_lent();
        }
        prefill = large;
    }

    // the experts the lent slots held go back into them, uploaded by the admission worker while the next graphs run:
    // until then they are misses, the cache does not start the small batches with a third of its slots empty
    void readmit_lent() {
        if (!host || lent.empty()) {
            lent.clear();
            return;
        }
        std::unique_lock<std::mutex> lk_svc(svc_mtx, std::defer_lock);
        if (svc_policy) {
            lk_svc.lock();
        }
        // one expert per job, so that it becomes a hit as it lands, and the copies of the device queued behind the
        // job's uploads (a checkpoint restored, the inputs of a graph) wait for one expert, not for many
        constexpr size_t per_job = 1;
        admit_job job;
        job.gate    = false; // nothing queued before reads these slots
        job.readmit = true;
        auto flush = [&]() {
            if (job.pubs.empty()) {
                return;
            }
            hdev[job.dev].in_flight += job.pubs.size();
            {
                std::lock_guard<std::mutex> lk(mtx);
                jobs.push_back(std::move(job));
            }
            cv.notify_one();
            job = admit_job();
            job.gate    = false;
            job.readmit = true;
        };
        for (const publication & pb : lent) {
            group & g = groups[pb.ig];
            moe_cache_slru & sl = g.slru;
            const int32_t key = pb.il*sl.n_expert + pb.expert;
            if (sl.key_of[pb.slot] >= 0 || sl.pending[pb.slot] || sl.slot_map[pb.il][pb.expert] < sl.n_slots || sl.pend.count(key) > 0) {
                continue;
            }
            if (!job.pubs.empty() && job.dev != g.id) {
                flush();
            }
            job.dev = g.id;
            sl.key_of[pb.slot]  = key;
            sl.pending[pb.slot] = true;
            sl.pend[key]        = pb.slot;
            sl.n_filled++;
            const layer & l = layers[pb.il];
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const size_t es = l.experts[ip]->nb[2];
                job.uploads.push_back({ g.id, g.banks[ip], hsrc(pb.il, ip, pb.expert), pb.slot*es, es });
                async_bytes += es;
            }
            job.pubs.push_back(pb);
            n_readmitted++;
            if (job.pubs.size() >= per_job) {
                flush();
            }
        }
        flush();
        lent.clear();
    }

    // whether the host can compute the experts of every cached layer
    bool host_supported(const llama_model & model) const {
        const auto & hp = model.hparams;
        // these clamp the activation of their experts in another way
        const bool other_clamp = model.arch == LLM_ARCH_MAPLE || model.arch == LLM_ARCH_DEEPSEEK4 || model.arch == LLM_ARCH_GLM5_NEXT ||
            model.arch == LLM_ARCH_DFLASH || model.arch == LLM_ARCH_HY_V4;
        for (const group & g : groups) {
            for (int32_t il : g.layers) {
                const llama_layer & ly = model.layers[il];
                const bool plain = ly.ffn_gate_up_exps == nullptr && ly.ffn_gate_exps_b == nullptr && ly.ffn_up_exps_b == nullptr &&
                    ly.ffn_down_exps_b == nullptr && ly.ffn_gate_exps_s == nullptr && ly.ffn_up_exps_s == nullptr &&
                    ly.ffn_down_exps_s == nullptr && ly.ffn_gate_exps_in_s == nullptr && ly.ffn_up_exps_in_s == nullptr &&
                    ly.ffn_down_exps_in_s == nullptr;
                const llama_moe_host::layer l = { ly.ffn_gate_exps, ly.ffn_up_exps, ly.ffn_down_exps, 0.0f };
                if (!plain || (other_clamp && hp.swiglu_clamp_exp[il] > 1e-6f) || ly.ffn_gate_exps == nullptr ||
                    !llama_moe_host::supported(l, ly.ffn_gate_exps->ne[0], ly.ffn_gate_exps->ne[1])) {
                    LLAMA_LOG_WARN("%s: the host cannot compute the experts of layer %d, host mode is disabled\n", __func__, il);
                    return false;
                }
            }
        }
        return true;
    }

    // the host address of the expert e of the ip-th expert tensor of layer il
    const uint8_t * hsrc(int32_t il, size_t ip, int32_t e) const {
        if (hptr.empty()) {
            const ggml_tensor * w = layers[il].experts[ip];
            return (const uint8_t *) w->data + (size_t) e*w->nb[2];
        }
        return (const uint8_t *) hptr[((size_t) il*hp_nm + ip)*hp_ne + e];
    }

    // the host copy of expert key is in host slot loc from now on; -1: it has none
    void set_hloc(int32_t key, int32_t loc) {
        hloc[key] = loc;
        if (loc < 0) {
            return;
        }
        const int32_t il = key / hp_ne;
        const int32_t e  = key % hp_ne;
        const layer & at = layers[loc / hp_ne];
        for (size_t ip = 0; ip < layers[il].experts.size(); ++ip) {
            const ggml_tensor * w = at.experts[ip];
            hptr[((size_t) il*hp_nm + ip)*hp_ne + e] = (const char *) w->data + (size_t) (loc % hp_ne)*w->nb[2];
        }
    }

    // every host copy at its own position
    void hptr_init() {
        for (const layer & l : layers) {
            hp_nm = std::max(hp_nm, (int32_t) l.experts.size());
            if (!l.experts.empty()) {
                hp_ne = std::max(hp_ne, (int32_t) l.experts[0]->ne[2]);
            }
        }
        hptr.assign(layers.size()*hp_nm*hp_ne, nullptr);
        hloc.assign(layers.size()*hp_ne, -1);
        for (size_t il = 0; il < layers.size(); ++il) {
            if (!layers[il].experts.empty()) {
                for (int32_t e = 0; e < (int32_t) layers[il].experts[0]->ne[2]; ++e) {
                    set_hloc((int32_t) il*hp_ne + e, (int32_t) il*hp_ne + e);
                }
            }
        }
    }

    // the slot tables, the mailboxes and the host part
    void host_init(const llama_model & model) {
        auto env_i = [](const char * name, int32_t def) {
            const char * v = getenv(name);
            return v != nullptr && *v != '\0' ? atoi(v) : def;
        };
        auto env_f = [](const char * name, float def) {
            const char * v = getenv(name);
            return v != nullptr && *v != '\0' ? (float) atof(v) : def;
        };
        host_max_tokens = std::max(1, env_i("LLAMA_MOE_HOST_MAX_TOKENS", host_max_tokens));
        host_period     = std::max(1, env_i("LLAMA_MOE_HOST_PERIOD",     host_period));
        host_max_swaps  = std::max(0, env_i("LLAMA_MOE_HOST_MAX_SWAPS",  host_max_swaps));
        host_max_fills  = std::max(0, env_i("LLAMA_MOE_HOST_MAX_FILLS",  host_max_fills));
        host_decay      = std::clamp(env_f("LLAMA_MOE_HOST_DECAY", host_decay), 0.0f, 1.0f);
        host_hyst       = std::max(1.0f, env_f("LLAMA_MOE_HOST_HYST", host_hyst));
        const int n_threads = std::max(1, env_i("LLAMA_MOE_HOST_THREADS", std::max(1, (int) std::thread::hardware_concurrency()/2 - 1)));
        batch_threads = n_threads;

        const int32_t n_layer = (int32_t) layers.size();
        int32_t n_embd = 0;
        int32_t n_ff   = 0;
        hptr_init();
        std::vector<llama_moe_host::layer> hl(n_layer);
        for (const group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            for (int32_t il : g.layers) {
                const llama_layer & ly = model.layers[il];
                const float limit = model.hparams.swiglu_clamp_exp[il];
                hl[il] = { ly.ffn_gate_exps, ly.ffn_up_exps, ly.ffn_down_exps, limit > 1e-6f ? limit : 0.0f };
                // the banks of a layer the host computes are gate, up and down
                GGML_ASSERT(layers[il].experts.size() == 3 && layers[il].experts[0] == ly.ffn_gate_exps &&
                    layers[il].experts[1] == ly.ffn_up_exps && layers[il].experts[2] == ly.ffn_down_exps);
                for (int mi = 0; mi < 3; ++mi) {
                    hl[il].ptrs[mi] = &hptr[((size_t) il*hp_nm + mi)*hp_ne];
                }
                n_embd = ly.ffn_gate_exps->ne[0];
                n_ff   = ly.ffn_gate_exps->ne[1];
            }
        }
        host_lay = llama_moe_host::layout(host_max_tokens, n_expert_used, n_embd, n_layer, host_n_expert);
        host_n_embd = n_embd;

        for (size_t id = 0; id < hdev.size(); ++id) {
            host_device & hd = hdev[id];
            if (hd.state == nullptr) {
                continue;
            }

            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(devices[id].backend));
            auto * alloc_fn = (void * (*)(size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_mapped_alloc");
            auto * free_fn  = (void   (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_mapped_free");
            hd.mailbox      = alloc_fn != nullptr ? alloc_fn(host_lay.size) : nullptr;
            hd.mailbox_free = free_fn;
            if (hd.mailbox == nullptr || free_fn == nullptr) {
                LLAMA_LOG_WARN("%s: %s has no mapped host memory, host mode is disabled\n", __func__, ggml_backend_name(devices[id].backend));
                host_mode = false;
                return;
            }
            hd.table_host = (int32_t *) ((char *) hd.mailbox + host_lay.off_table);
            std::fill(hd.table_host, hd.table_host + (size_t) n_layer*host_n_expert, -1);
        }

        host = std::make_unique<llama_moe_host>(std::move(hl), host_n_expert, n_expert_used, n_embd, n_ff, host_max_tokens, n_threads, skip_rank);
        for (const host_device & hd : hdev) {
            if (hd.mailbox != nullptr) {
                host->add_mailbox(hd.mailbox);
            }
        }
        host_score.assign((size_t) n_layer*host_n_expert, 0.0f);

        const char * pol = getenv("LLAMA_MOE_HOST_POLICY");
        svc_policy = pol == nullptr || strcmp(pol, "graph") != 0;
        if (svc_policy) {
            svc_swaps   = std::max(0, env_i("LLAMA_MOE_HOST_SWAPS", svc_swaps));
            svc_fills   = std::max(0, env_i("LLAMA_MOE_HOST_FILLS", svc_fills));
            svc_per_req = std::max(1, env_i("LLAMA_MOE_HOST_PER_REQUEST", svc_per_req));
            host_decay  = std::clamp(env_f("LLAMA_MOE_HOST_DECAY", 0.9f), 0.0f, 1.0f);
            host_hyst   = std::max(1.0f, env_f("LLAMA_MOE_HOST_HYST", 1.0f));
            for (const group & g : groups) {
                if (g.n_slots > 0) {
                    for (int32_t il : g.layers) {
                        svc_max_il = std::max(svc_max_il, il);
                    }
                }
            }
            host->set_listener(this);
            LLAMA_LOG_INFO("%s: host mode cache policy: per graph %d replacements and %d admissions into free slots, %d started per request, decay %.2f, hysteresis %.2f\n",
                __func__, svc_swaps, svc_fills, svc_per_req, host_decay, host_hyst);
        }

        LLAMA_LOG_INFO("%s: host mode: up to %d tokens, %d threads, a plan every %d graphs (%d fills, %d swaps), decay %.3f, hysteresis %.2f\n",
            __func__, host_max_tokens, n_threads, host_period, host_max_fills, host_max_swaps, host_decay, host_hyst);

        if (const char * path = getenv("LLAMA_MOE_CACHE_PROFILE"); path != nullptr && *path != '\0') {
            profile_path = path;
            profile_new.assign(host_score.size(), 0.0f);
            if (profile_read(n_layer)) {
                profile_fill();
                excl_init();
            }
        }
    }

    static constexpr uint32_t profile_magic = 0x50434d4c; // "LMCP"

    // the file: u32 magic, u32 version 1, i32 n_layer, i32 n_expert, then f32 counts [n_layer*n_expert]
    bool profile_read(int32_t n_layer) {
        FILE * f = fopen(profile_path.c_str(), "rb");
        if (f == nullptr) {
            LLAMA_LOG_INFO("%s: no MoE cache profile at %s yet, the cache starts empty\n", __func__, profile_path.c_str());
            return false;
        }
        uint32_t hdr[2] = {};
        int32_t  dims[2] = {};
        std::vector<float> counts(host_score.size());
        const bool ok = fread(hdr, sizeof(hdr), 1, f) == 1 && fread(dims, sizeof(dims), 1, f) == 1 &&
            hdr[0] == profile_magic && hdr[1] == 1 && dims[0] == n_layer && dims[1] == host_n_expert &&
            fread(counts.data(), sizeof(float), counts.size(), f) == counts.size();
        fclose(f);
        if (!ok) {
            LLAMA_LOG_WARN("%s: %s is not a MoE cache profile of this model, ignored\n", __func__, profile_path.c_str());
            return false;
        }
        profile = std::move(counts);
        return true;
    }

    // the past counts plus this session's, written to a new file that then replaces the old one
    void profile_write() {
        if (profile_path.empty()) {
            return;
        }
        double total = 0.0;
        for (float c : profile_new) {
            total += c;
        }
        if (total == 0.0) {
            return;
        }
        std::vector<float> counts = profile_new;
        if (!profile.empty()) {
            for (size_t i = 0; i < counts.size(); ++i) {
                counts[i] += profile[i];
            }
        }
        // bounded, so that the recent sessions keep their weight
        double sum = 0.0;
        for (float c : counts) {
            sum += c;
        }
        constexpr double max_sum = 1e8;
        if (sum > max_sum) {
            for (float & c : counts) {
                c = (float) (c*(max_sum/sum));
            }
        }
        const std::string tmp = profile_path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (f == nullptr) {
            LLAMA_LOG_WARN("%s: cannot write the MoE cache profile %s\n", __func__, tmp.c_str());
            return;
        }
        const uint32_t hdr[2]  = { profile_magic, 1 };
        const int32_t  dims[2] = { (int32_t) (counts.size()/host_n_expert), host_n_expert };
        const bool ok = fwrite(hdr, sizeof(hdr), 1, f) == 1 && fwrite(dims, sizeof(dims), 1, f) == 1 &&
            fwrite(counts.data(), sizeof(float), counts.size(), f) == counts.size();
        if (fclose(f) != 0 || !ok || rename(tmp.c_str(), profile_path.c_str()) != 0) {
            LLAMA_LOG_WARN("%s: cannot write the MoE cache profile %s\n", __func__, profile_path.c_str());
            remove(tmp.c_str());
            return;
        }
        LLAMA_LOG_INFO("%s: MoE cache profile %s updated with %.0f selections\n", __func__, profile_path.c_str(), total);
    }

    // the experts the profile counts most fill the slots: those of the current phase now, the others (lent while large
    // batches run, a context starts with them) when small batches come
    void profile_fill() {
        std::vector<int32_t> keys;
        size_t n_now   = 0;
        size_t n_later = 0;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            moe_cache_slru & sl = g.slru;
            host_device & hd = hdev[g.id];
            const int32_t ne = sl.n_expert;
            keys.clear();
            for (int32_t il : g.layers) {
                for (int32_t e = 0; e < ne; ++e) {
                    if (profile[(size_t) il*ne + e] > 0.0f) {
                        keys.push_back(il*ne + e);
                    }
                }
            }
            const size_t n = std::min(keys.size(), (size_t) g.n_slots);
            std::partial_sort(keys.begin(), keys.begin() + n, keys.end(),
                [&](int32_t a, int32_t b) { return profile[a] > profile[b]; });
            for (size_t i = 0; i < n; ++i) {
                const int32_t s   = (int32_t) i;
                const int32_t key = keys[i];
                const int32_t il  = key / ne;
                const int32_t e   = key % ne;
                if (sl.key_of[s] >= 0 || sl.slot_map[il][e] < sl.n_slots) {
                    continue;
                }
                if (s >= g.n_active) {
                    lent.push_back({ (int32_t) ig, il, e, s });
                    n_later++;
                    continue;
                }
                const layer & l = layers[il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const size_t es = l.experts[ip]->nb[2];
                    ggml_backend_tensor_set_async(side[g.id], g.banks[ip], hsrc(il, ip, e), s*es, es);
                }
                sl.key_of[s]     = key;
                sl.slot_map[il][e] = s;
                hd.table_host[key] = s;
                sl.n_filled++;
                n_now++;
            }
        }
        for (ggml_backend_t b : side) {
            ggml_backend_synchronize(b);
        }
        LLAMA_LOG_INFO("%s: MoE cache profile %s: %zu experts cached now, %zu when small batches come\n", __func__,
            profile_path.c_str(), n_now, n_later);
    }

    // LLAMA_MOE_CACHE_EXCLUSIVE=<MiB>: once the profile filled the slots, the experts of the first slots of each group, a
    // share of its core up to that size, give their host copies back (see excl_on)
    void excl_init() {
        const char * v = getenv("LLAMA_MOE_CACHE_EXCLUSIVE");
        const size_t budget = v != nullptr ? (size_t) std::max(0, atoi(v)) << 20 : 0;
        if (budget == 0) {
            return;
        }
        if (!host || !phases || !svc_policy || groups.empty() || hptr.empty()) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_CACHE_EXCLUSIVE needs host mode, cache phases and the service policy, ignored\n", __func__);
            return;
        }
        if (const char * sp = getenv("LLAMA_MOE_CACHE_EXCLUSIVE_SPARES"); sp != nullptr && *sp != '\0') {
            excl_spares = std::max(1, atoi(sp));
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(devices[groups[0].id].backend));
        auto * release_fn    = (size_t (*)(const void * const *, const size_t *, size_t, void (*)(void *), void *))
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_release");
        auto * releasable_fn = (bool (*)(const void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_releasable");
        bool ok = release_fn != nullptr && releasable_fn != nullptr;
        for (size_t il = 0; il < layers.size() && ok; ++il) {
            for (const ggml_tensor * w : layers[il].experts) {
                ok = ok && (layers[il].ig < 0 || releasable_fn(w->data, ggml_nbytes(w)));
            }
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: the host expert buffers cannot give memory back (GGML_CUDA_HOST_MEMFD), LLAMA_MOE_CACHE_EXCLUSIVE ignored\n", __func__);
            return;
        }
        excl_sync = (void (*)(ggml_backend_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_synchronize_sleep");
        if (excl_sync == nullptr) {
            excl_sync = ggml_backend_synchronize;
        }
        excl_side.assign(devices.size(), nullptr);
        for (size_t id = 0; id < devices.size(); ++id) {
            excl_side[id] = ggml_backend_dev_init(ggml_backend_get_device(devices[id].backend), nullptr);
            if (excl_side[id] == nullptr) {
                LLAMA_LOG_WARN("%s: no download stream on %s, LLAMA_MOE_CACHE_EXCLUSIVE ignored\n", __func__, ggml_backend_name(devices[id].backend));
                return;
            }
        }

        // the same share of the core in every group
        size_t core_bytes = 0;
        for (const group & g : groups) {
            for (const ggml_tensor * t : g.ref) {
                core_bytes += (size_t) g.n_core*t->nb[2];
            }
        }
        const double share = core_bytes > 0 ? std::min(1.0, (double) budget/core_bytes) : 0.0;
        const int32_t ne = hp_ne;
        for (const group & g : groups) {
            GGML_ASSERT(g.slru.n_expert == ne);
        }
        std::vector<std::vector<int32_t>> excl_of(layers.size()); // the experts of each layer the exclusive slots hold
        for (group & g : groups) {
            g.n_excl = (int32_t) (g.n_core*share);
            for (int32_t s = 0; s < g.n_excl; ++s) {
                const int32_t key = g.slru.key_of[s];
                if (key >= 0 && !g.slru.pending[s]) {
                    excl_of[key / ne].push_back(key % ne);
                }
            }
        }

        // each group's spares: the first positions given back of its layers, in turn
        std::vector<int32_t> n_spare(layers.size(), 0);
        for (group & g : groups) {
            int32_t left = g.n_excl > 0 ? excl_spares : 0;
            for (bool any = true; left > 0 && any; ) {
                any = false;
                for (int32_t il : g.layers) {
                    if (left > 0 && n_spare[il] < (int32_t) excl_of[il].size()) {
                        n_spare[il]++;
                        left--;
                        any = true;
                    }
                }
            }
        }

        // compaction: the experts with a host copy in the last positions of a layer move into the positions of its
        // exclusive experts in front of them; nothing reads these buffers meanwhile
        const int64_t t0 = ggml_time_us();
        auto compact = [&](int32_t il) {
            const layer & l = layers[il];
            const int32_t k = (int32_t) excl_of[il].size();
            std::vector<bool> ex(ne, false);
            for (int32_t e : excl_of[il]) {
                ex[e] = true;
            }
            int32_t p = 0;
            for (int32_t e = ne - k; e < ne; ++e) {
                if (ex[e]) {
                    continue;
                }
                while (p < ne - k && !ex[p]) {
                    p++;
                }
                GGML_ASSERT(p < ne - k);
                for (const ggml_tensor * w : l.experts) {
                    memcpy((uint8_t *) w->data + (size_t) p*w->nb[2], (const uint8_t *) w->data + (size_t) e*w->nb[2], w->nb[2]);
                }
                set_hloc(il*ne + e, il*ne + p);
                p++;
            }
            for (int32_t e : excl_of[il]) {
                hloc[il*ne + e] = -1;
            }
        };
        {
            std::vector<int32_t> todo;
            for (size_t il = 0; il < layers.size(); ++il) {
                if (!excl_of[il].empty()) {
                    todo.push_back((int32_t) il);
                }
            }
            std::atomic<size_t> next { 0 };
            std::vector<std::thread> th;
            for (int t = 0; t < std::min(8, (int) todo.size()); ++t) {
                th.emplace_back([&] {
                    for (size_t i; (i = next++) < todo.size(); ) {
                        compact(todo[i]);
                    }
                });
            }
            for (std::thread & t : th) {
                t.join();
            }
        }

        std::vector<const void *> ptrs;
        std::vector<size_t>       sizes;
        size_t n_excl = 0;
        for (group & g : groups) {
            for (int32_t il : g.layers) {
                const int32_t k = (int32_t) excl_of[il].size();
                for (int32_t p = ne - k; p < ne - k + n_spare[il]; ++p) {
                    g.spares.push_back(il*ne + p);
                }
                for (const ggml_tensor * w : layers[il].experts) {
                    if (k > n_spare[il]) {
                        ptrs.push_back((const uint8_t *) w->data + (size_t) (ne - k + n_spare[il])*w->nb[2]);
                        sizes.push_back((size_t) (k - n_spare[il])*w->nb[2]);
                    }
                }
                n_excl += k;
            }
        }
        excl_released = release_fn(ptrs.data(), sizes.data(), ptrs.size(), nullptr, nullptr);
        excl_on = true;
        excl_worker = std::thread([this] { run_excl(); });
        LLAMA_LOG_INFO("%s: exclusive core: %zu experts of %.0f%% of the core keep no host copy, %.2f GiB given back in %.1f s, "
            "%d spare host slots per group\n", __func__, n_excl, 100.0*share, excl_released/1024.0/1024.0/1024.0,
            (ggml_time_us() - t0)/1e6, excl_spares);
    }

    // the swaps of the next graph: admissions into free slots, then replacements of cached experts selected less,
    // the replacements that gain the most first
    void svc_plan_graph() {
        host->take_counts(host_score, host_decay, profile_new.empty() ? nullptr : &profile_new);
        if (excl_on) {
            for (const svc_move & mv : svc_plan) {
                const moe_cache_slru & sl = groups[mv.ig].slru;
                excl_n_waited += !mv.started && mv.slot < groups[mv.ig].n_excl && sl.key_of[mv.slot] >= 0 && sl.staged[mv.slot] < 0;
            }
        }
        svc_plan.clear();
        svc_pos = 0;
        std::vector<svc_move> fills;
        std::vector<svc_move> swaps;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            moe_cache_slru & sl = g.slru;
            const int32_t ne = sl.n_expert;
            host_free.clear();
            host_vict.clear();
            for (int32_t s = 0; s < g.n_active; ++s) {
                if (!sl.pending[s]) {
                    (sl.key_of[s] < 0 ? host_free : host_vict).push_back(s);
                }
            }
            host_cand.clear();
            for (int32_t il : g.layers) {
                const int32_t * slots = sl.slot_map[il];
                for (int32_t e = 0; e < ne; ++e) {
                    const int32_t key = il*ne + e;
                    if (slots[e] >= sl.n_slots && host_score[key] > 0.0f && sl.pend.count(key) == 0) {
                        host_cand.push_back(key);
                    }
                }
            }
            const size_t n_fill = std::min(host_free.size(), (size_t) svc_fills);
            const size_t n_cand = std::min(host_cand.size(), n_fill + svc_swaps);
            const size_t n_vict = std::min(host_vict.size(), (size_t) svc_swaps);
            std::partial_sort(host_cand.begin(), host_cand.begin() + n_cand, host_cand.end(),
                [&](int32_t a, int32_t b) { return host_score[a] > host_score[b]; });
            std::partial_sort(host_vict.begin(), host_vict.begin() + n_vict, host_vict.end(),
                [&](int32_t a, int32_t b) { return host_score[sl.key_of[a]] < host_score[sl.key_of[b]]; });
            size_t i_vict = 0;
            for (size_t c = 0; c < n_cand; ++c) {
                const int32_t key = host_cand[c];
                if (c < n_fill) {
                    fills.push_back({ host_score[key], (int32_t) ig, key, host_free[c], false });
                    continue;
                }
                if (i_vict >= n_vict) {
                    break;
                }
                const int32_t v = host_vict[i_vict];
                const float   s_v = host_score[sl.key_of[v]];
                if (host_score[key] <= s_v*host_hyst + 1.0f) {
                    break;
                }
                swaps.push_back({ host_score[key] - s_v, (int32_t) ig, key, v, false });
                i_vict++;
            }
        }
        auto by_benefit = [](const svc_move & a, const svc_move & b) { return a.benefit > b.benefit; };
        std::sort(fills.begin(), fills.end(), by_benefit);
        std::sort(swaps.begin(), swaps.end(), by_benefit);
        fills.resize(std::min(fills.size(), (size_t) svc_fills));
        swaps.resize(std::min(swaps.size(), (size_t) svc_swaps));
        svc_plan = std::move(fills);
        svc_plan.insert(svc_plan.end(), swaps.begin(), swaps.end());
        if (excl_on) {
            // a download whose slot is no longer a victim gives its host slot back
            for (const svc_move & mv : svc_plan) {
                if (mv.slot < groups[mv.ig].n_excl) {
                    groups[mv.ig].slru.staging_keep(mv.slot);
                }
            }
            for (group & g : groups) {
                moe_cache_slru & sl = g.slru;
                for (int32_t s = 0; s < g.n_excl; ++s) {
                    const bool victim = sl.take_kept(s);
                    if (sl.staged[s] >= 0 && !victim) {
                        excl_unstage(g, s);
                    }
                }
            }
            excl_stage();
        }
    }

    // The victims of exclusive slots in the plan are downloaded into spare host slots, in the order of the plan, as
    // far as the spares go; the others wait for the host slots that the admitted experts free
    void excl_stage() {
        stage_job job;
        job.dev = -1;
        for (size_t i = svc_pos; i < svc_plan.size(); ++i) {
            const svc_move & mv = svc_plan[i];
            group & g = groups[mv.ig];
            moe_cache_slru & sl = g.slru;
            const int32_t vk = mv.slot < g.n_excl ? sl.key_of[mv.slot] : -1;
            if (mv.started || vk < 0 || sl.pending[mv.slot] || sl.staged[mv.slot] >= 0 || sl.staging[mv.slot] || g.spares.empty()) {
                continue;
            }
            const int32_t loc = g.spares.back();
            g.spares.pop_back();
            sl.staging[mv.slot] = true;
            job.stages.push_back({ mv.ig, mv.slot, vk, loc });
            job.dev = g.id;
        }
        if (!job.stages.empty()) {
            excl_in_flight += (int32_t) job.stages.size();
            {
                std::lock_guard<std::mutex> lk(mtx);
                excl_jobs.push_back(std::move(job));
            }
            excl_cv.notify_one();
        }
    }

    // the service thread makes no CUDA call: a graph waiting for its answer may hold the driver while it is launched
    void before_answer(int32_t il) override {
        std::lock_guard<std::mutex> lk(svc_mtx);
        if (excl_on) {
            // the requests that could read them are answered: the CPU work of a request is done before its answer, and
            // the next one reads the slot table after it
            for (group & g : groups) {
                g.spares.insert(g.spares.end(), g.quarantine.begin(), g.quarantine.end());
                g.quarantine.clear();
            }
        }
        host_collect();
        if (excl_on) {
            excl_stage();
        }
        // start a few swaps whose victims are not of layer il, whose slots the device may read now
        int32_t n = 0;
        for (size_t i = svc_pos; i < svc_plan.size() && n < svc_per_req; ++i) {
            svc_move & mv = svc_plan[i];
            if (mv.started) {
                continue;
            }
            group & g = groups[mv.ig];
            moe_cache_slru & sl = g.slru;
            const int32_t ne = sl.n_expert;
            const int32_t vk = sl.key_of[mv.slot];
            if (mv.slot >= g.n_active || sl.pending[mv.slot] || sl.pend.count(mv.key) > 0 ||
                sl.slot_map[mv.key / ne][mv.key % ne] < sl.n_slots) {
                mv.started = true; // no longer valid
                continue;
            }
            if (vk >= 0 && vk / ne == il) {
                continue; // the device may read the victim now: later
            }
            if (vk >= 0 && mv.slot < g.n_excl) {
                // the victim of an exclusive slot leaves once its download is done, with that host copy
                if (sl.staged[mv.slot] < 0) {
                    continue;
                }
                set_hloc(vk, sl.staged[mv.slot]);
                sl.staged[mv.slot] = -1;
                excl_n_exchanges++;
            }
            if (vk >= 0) {
                sl.slot_map[vk / ne][vk % ne] = sl.empty;
                hdev[g.id].table_host[vk] = -1;
                svc_n_swaps++;
            } else {
                sl.n_filled++;
                svc_n_fills++;
            }
            sl.key_of[mv.slot]  = mv.key;
            sl.pending[mv.slot] = true;
            sl.pend[mv.key]     = mv.slot;
            svc_started.push_back({ mv.ig, mv.key / ne, mv.key % ne, mv.slot });
            mv.started = true;
            n++;
        }
        while (svc_pos < svc_plan.size() && svc_plan[svc_pos].started) {
            svc_pos++;
        }
    }

    void after_answer(int32_t il) override {
        std::lock_guard<std::mutex> lk(svc_mtx);
        if (!svc_started.empty()) {
            // the upload worker uploads them, a single device holds the cached layers in host mode
            admit_job job;
            job.dev  = groups[svc_started[0].ig].id;
            job.gate = false;
            for (const publication & pb : svc_started) {
                const group & g = groups[pb.ig];
                const layer & l = layers[pb.il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const size_t es = l.experts[ip]->nb[2];
                    job.uploads.push_back({ g.id, g.banks[ip], hsrc(pb.il, ip, pb.expert), pb.slot*es, es });
                    async_bytes += es;
                }
            }
            job.pubs = std::move(svc_started);
            svc_started.clear();
            hdev[job.dev].in_flight += job.pubs.size();
            {
                std::lock_guard<std::mutex> lk_jobs(mtx);
                jobs.push_back(std::move(job));
            }
            cv.notify_one();
        }
        if (il == svc_max_il) {
            svc_plan_graph();
        }
    }

    // the swaps in flight are done or undone: before the cache changes its phase
    void svc_settle() {
        std::lock_guard<std::mutex> lk(svc_mtx);
        for (;;) {
            size_t n = (size_t) excl_in_flight;
            for (const host_device & hd : hdev) {
                n += hd.in_flight;
            }
            if (n == 0) {
                break;
            }
            host_collect();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        for (const publication & pb : svc_started) {
            moe_cache_slru & sl = groups[pb.ig].slru;
            const int32_t key = pb.il*sl.n_expert + pb.expert;
            sl.pending[pb.slot] = false;
            sl.pend.erase(key);
            sl.key_of[pb.slot] = -1;
            sl.n_filled--;
        }
        svc_started.clear();
        svc_plan.clear();
        svc_pos = 0;
        for (group & g : groups) {
            for (int32_t s = 0; s < g.n_excl; ++s) {
                if (g.slru.staged[s] >= 0) {
                    excl_unstage(g, s);
                }
            }
        }
    }

    // the downloaded host copy of the expert of exclusive slot s is not needed for now: its host slot becomes a spare
    void excl_unstage(group & g, int32_t s) {
        g.spares.push_back(g.slru.staged[s]);
        g.slru.staged[s] = -1;
        excl_n_unstaged++;
    }

    void init_batch(int32_t max_batch) {
        if (!host || !phases || max_batch <= host_max_tokens) {
            return;
        }
        if (const char * v = getenv("LLAMA_MOE_BATCH_CPU"); v != nullptr && *v != '\0') {
            batch_force = std::clamp((float) atof(v), 0.0f, 1.0f);
            if (batch_force == 0.0f) {
                return;
            }
        }
        if (const char * v = getenv("LLAMA_MOE_BATCH_CPU_MAX"); v != nullptr && *v != '\0') {
            batch_cap = std::clamp((float) atof(v), 0.0f, 1.0f);
        }
        const int32_t n_layer = (int32_t) layers.size();
        batch_lay = llama_moe_host::layout(max_batch, n_expert_used, host_n_embd, n_layer, host_n_expert);
        for (size_t id = 0; id < hdev.size(); ++id) {
            host_device & hd = hdev[id];
            if (hd.mailbox == nullptr) {
                continue;
            }
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(devices[id].backend));
            auto * alloc_fn = (void * (*)(size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_mapped_alloc");
            hd.batch_mailbox = alloc_fn != nullptr ? alloc_fn(batch_lay.size) : nullptr;
            if (hd.batch_mailbox == nullptr) {
                LLAMA_LOG_WARN("%s: no mapped host memory for the batches shared with the host\n", __func__);
                return;
            }
            hd.batch_table = (int32_t *) ((char *) hd.batch_mailbox + batch_lay.off_table);
            std::fill(hd.batch_table, hd.batch_table + (size_t) n_layer*host_n_expert, 0);

            ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
            hd.batch_ctx.reset(ggml_init(params));
            hd.batch_state = ggml_new_tensor_1d(hd.batch_ctx.get(), GGML_TYPE_I32, 8);
            ggml_format_name(hd.batch_state, "moe_cache.batch_state");
            hd.batch_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(hd.batch_ctx.get(), devices[id].buft));
            if (!hd.batch_buf) {
                LLAMA_LOG_WARN("%s: no device memory for the state of the batches shared with the host\n", __func__);
                return;
            }
            ggml_backend_buffer_clear(hd.batch_buf.get(), 0);
            host->add_batch_mailbox(hd.batch_mailbox, max_batch);
        }
        batch_max = max_batch;
        batch_cpu.assign((size_t) n_layer*host_n_expert, 0);
        LLAMA_LOG_INFO("%s: large batches of up to %d tokens share the experts the core lacks with the host (%.1f MiB of mapped host memory)\n",
            __func__, max_batch, batch_lay.size/1024.0/1024.0);
    }

    // the lacking experts of layer il the host takes for a batch of n_tokens, the least selected first, as they cost the
    // host the least and an upload costs the same for all. The device's cycle for the layer is the longer of the uploads
    // left and its own work: what it does before it posts (pre, which the host cannot overlap) then the longer of its
    // experts (dev) and the host's answer. The host takes experts while that cycle does not grow. t[0], t[1]: the
    // layer's predicted seconds without and with the host. A layer whose kind the host has not measured yet takes a
    // few of its cheapest experts, so that it is measured.
    size_t batch_take(int32_t il, const std::vector<int32_t> & lacking, const std::vector<double> & k_exp,
            double up_e, double pre, double dev, double io, double & expected, double t[2]) const {
        llama_moe_host::batch_cost bc;
        const bool known = host->get_batch_cost(il, bc);
        size_t cap = std::min(lacking.size(), (size_t) std::lround(batch_cap*lacking.size()));
        if (!known) {
            bc.per_item  = 1e-3;
            bc.per_entry = 0.3e-3;
            bc.overhead  = 1e-3;
            bc.n_threads = batch_threads;
            cap = std::min<size_t>(cap, 4);
        }
        t[0] = std::max(lacking.size()*up_e, pre + dev);
        t[1] = t[0];
        double sum = 0.0, t_max = 0.0;
        size_t n = 0;
        for (; n < cap; ++n) {
            const double k = k_exp[n];
            if (!known && k > 16.0) {
                break;
            }
            // an expert with fewer than 64 entries is one item if it has any, a larger one an item per 64
            const double items = k < 64.0 ? 1.0 - std::exp(-k) : std::ceil(k/64.0);
            const double t_e = bc.per_item*items + bc.per_entry*k;
            const double t_i = std::min(t_e, bc.per_item + bc.per_entry*64.0);
            const double h   = io + bc.overhead + std::max((sum + t_e)/bc.n_threads, std::max(t_max, t_i));
            const double u   = (lacking.size() - n - 1)*up_e;
            const double c   = std::max(u, pre + std::max(dev, h));
            if (known && c > t[1]) {
                break;
            }
            sum  += t_e;
            t_max = std::max(t_max, t_i);
            t[1]  = c;
            expected += k;
        }
        return n;
    }

    // rate: the bytes per second of the uploads, 0 if not measured yet
    void begin_batch(int64_t n_tokens, double rate) {
        batch_tokens = 0;
        batch_begun  = 0;
        if (batch_max == 0 || !(phases && prefill) || n_tokens <= host_max_tokens || n_tokens > batch_max) {
            return;
        }
        const bool measured = batch_force < 0.0f;
        if (rate <= 0.0) {
            rate = 12e9;
        }
        int32_t n_moe = 0;
        for (const layer & l : layers) {
            n_moe += l.ig >= 0;
        }
        // the device's own seconds per layer, and the inputs to the host and its answer, which the host waits for
        const double dev_layer = batch_c_tok*n_tokens/std::max(1, n_moe);
        const double io        = (double) n_tokens*host_n_embd*6/rate;

        const int32_t ne = host_n_expert;
        std::fill(batch_cpu.begin(), batch_cpu.end(), 0);
        batch_n_cpu = 0;
        if (measured) {
            if (batch_pop.empty()) {
                batch_pop.assign((size_t) layers.size()*ne, 0.0f);
            }
            host->take_batch_counts(batch_pop, 0.7f);
        }
        int    n_src[2]  = {};
        std::vector<int32_t> lacking;
        std::vector<double>  k_exp;
        size_t n_lacking = 0;
        double expected  = 0.0;
        double t_layer[2];
        double t_sum[2]  = {};
        bool   probing   = false;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            size_t expert_bytes = 0;
            for (const ggml_tensor * t : g.ref) {
                expert_bytes += t->nb[2];
            }
            for (int32_t il : g.layers) {
                int32_t * table = hdev[g.id].batch_table;
                if (table == nullptr) {
                    continue;
                }
                const int32_t * slots = g.slru.slot_map[il];
                lacking.clear();
                for (int32_t e = 0; e < ne; ++e) {
                    table[(size_t) il*ne + e] = 0;
                    if (slots[e] >= g.slru.n_slots) {
                        lacking.push_back(e);
                    }
                }
                n_lacking += lacking.size();
                const auto pop = [&](int32_t e) {
                    const size_t key = (size_t) il*ne + e;
                    return !profile.empty() ? profile[key] : host_score[key];
                };
                size_t n_cpu = 0;
                if (!measured) {
                    n_cpu = std::min(lacking.size(), (size_t) std::lround(batch_force*lacking.size()));
                    std::partial_sort(lacking.begin(), lacking.begin() + n_cpu, lacking.end(),
                        [&](int32_t a, int32_t b) { return pop(a) < pop(b); });
                } else {
                    // the entries of each expert expected from its share of the layer's selections, in the last shared
                    // batches if they counted enough of them, in the profile otherwise
                    const float * bp = batch_pop.data() + (size_t) il*ne;
                    double bp_sum = 0.0;
                    for (int32_t e = 0; e < ne; ++e) {
                        bp_sum += bp[e];
                    }
                    const int src = bp_sum >= 4.0*ne ? 1 : 0;
                    n_src[src]++;
                    const auto share = [&](int32_t e) {
                        return src == 1 ? (bp[e] + 0.5)/(bp_sum + 0.5*ne) : (double) pop(e);
                    };
                    std::sort(lacking.begin(), lacking.end(), [&](int32_t a, int32_t b) { return share(a) < share(b); });
                    double pop_sum = 1.0;
                    if (src == 0) {
                        pop_sum = 0.0;
                        for (int32_t e = 0; e < ne; ++e) {
                            pop_sum += pop(e);
                        }
                    }
                    const double n_ent = (double) n_tokens*n_expert_used*batch_ratio[src];
                    k_exp.resize(lacking.size());
                    for (size_t i = 0; i < lacking.size(); ++i) {
                        k_exp[i] = n_ent*(pop_sum > 0.0 ? share(lacking[i])/pop_sum : 1.0/ne);
                    }
                    llama_moe_host::batch_cost bc;
                    probing |= !host->get_batch_cost(il, bc);
                    // before its post the device works for a part of its time the host cannot overlap, measured, or
                    // taken as a fifth until then
                    const double pre_tok = host->get_batch_pre(il);
                    const double pre     = pre_tok > 0.0 ? std::min(dev_layer, pre_tok*n_tokens) : 0.2*dev_layer;
                    n_cpu = batch_take(il, lacking, k_exp, expert_bytes/rate, pre, dev_layer - pre, io, expected, t_layer);
                    t_sum[0] += t_layer[0];
                    t_sum[1] += t_layer[1];
                }
                for (size_t i = 0; i < n_cpu; ++i) {
                    const size_t key = (size_t) il*ne + lacking[i];
                    batch_cpu[key] = 1;
                    table[key] = -1;
                }
                batch_n_cpu += n_cpu;
            }
        }
        // a measured batch is shared only if the host saves more than a little of it, or measures a kind of layer
        if (measured && batch_n_cpu > 0 && !probing && t_sum[1] > 0.98*t_sum[0]) {
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                const group & g = groups[ig];
                for (int32_t il : g.layers) {
                    if (int32_t * table = hdev[g.id].batch_table; table != nullptr && g.n_slots > 0) {
                        std::fill(table + (size_t) il*ne, table + (size_t) (il + 1)*ne, 0);
                    }
                }
            }
            std::fill(batch_cpu.begin(), batch_cpu.end(), 0);
            batch_n_cpu = 0;
        }
        if (measured) {
            batch_begun    = n_tokens;
            batch_expected = batch_n_cpu > 0 && (n_src[0] == 0 || n_src[1] == 0) ? expected : 0.0;
            batch_src      = n_src[1] > 0 ? 1 : 0;
            batch_pred[0]  = t_sum[0];
            batch_pred[1]  = batch_n_cpu > 0 ? t_sum[1] : t_sum[0];
            host->get_batch_totals(batch_host_s0, batch_host_e0, batch_host_w0);
            static const bool profile_log = getenv("LLAMA_PROFILE_UBATCH") != nullptr;
            if (profile_log) {
                LLAMA_LOG_INFO("%s: %lld tokens: %zu of %zu lacking experts to the host, predicted %.0f ms instead of %.0f (uploads at %.1f GB/s, device %.0f ms, %d layers from the batches, entries x%.2f/x%.2f)\n",
                    __func__, (long long) n_tokens, batch_n_cpu, n_lacking, 1e3*batch_pred[1], 1e3*batch_pred[0], rate/1e9,
                    1e3*batch_c_tok*n_tokens, n_src[1], batch_ratio[0], batch_ratio[1]);
            }
        }
        // the graph of a batch shares it with the host whenever experts go to the host, so a graph reused for a batch
        // of the same size has the same posts
        if (batch_n_cpu == 0 && (measured || batch_force <= 0.0f)) {
            return;
        }
        batch_tokens = n_tokens;
        batch_graphs++;
        batch_share += n_lacking > 0 ? (double) batch_n_cpu/n_lacking : 0.0;
    }

    // after the graph of a batch begun in the measured mode: seconds from its start to the device's end, the bytes the
    // prefetch uploaded for it at the rate it measures. The host's entries correct the expectation of its source, and a
    // batch the host took no part in and the uploads did not bound gives the device's time per token
    void end_batch(double seconds, int64_t bytes_uploaded, double rate) {
        if (batch_begun == 0) {
            return;
        }
        const int64_t n_tokens = batch_begun;
        batch_begun = 0;
        double   host_s = 0.0;
        uint64_t host_e = 0;
        double   host_w = 0.0;
        host->get_batch_totals(host_s, host_e, host_w);
        host_s -= batch_host_s0;
        host_e -= batch_host_e0;
        host_w -= batch_host_w0;
        if (batch_expected >= 16.0) {
            const double r = (double) host_e/batch_expected;
            double & ratio = batch_ratio[batch_src];
            ratio = std::clamp(ratio*(1.0 + 0.5*(r - 1.0)), 0.25, 8.0);
        }
        // the device's own time from a batch the host took no part in, if the uploads did not bound it
        const double up_s = rate > 0.0 ? bytes_uploaded/rate : 0.0;
        if (batch_tokens == 0 && seconds > 1.25*up_s) {
            batch_c_tok += 0.5*(seconds/n_tokens - batch_c_tok);
        }
        static const bool profile_log = getenv("LLAMA_PROFILE_UBATCH") != nullptr;
        if (profile_log) {
            LLAMA_LOG_INFO("%s: %lld tokens: %.0f ms (predicted %.0f), uploads %.0f ms at %.1f GB/s, host %.0f ms for %llu entries, the device waited %.0f ms\n",
                __func__, (long long) n_tokens, 1e3*seconds, 1e3*batch_pred[1], 1e3*up_s, rate/1e9, 1e3*host_s,
                (unsigned long long) host_e, 1e3*host_w);
        }
    }

    bool end_batch_wanted() const {
        return batch_begun > 0;
    }

    // every cached layer of a shared batch posts to the host, whatever its own tokens: the prefetch does not upload the
    // host's experts of any layer, so a layer that computed them on the device (the last one, cropped to the outputs)
    // would lose them
    bool get_batch_part(int32_t il, int64_t n_tokens, llama_moe_cache::host_part & hp) const {
        if (batch_tokens == 0 || n_tokens <= 0 || n_tokens > batch_max || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return false;
        }
        const group & g = groups[layers[il].ig];
        const host_device & hd = hdev[g.id];
        if (hd.batch_mailbox == nullptr) {
            return false;
        }
        hp.backend   = devices[g.id].backend;
        hp.state     = hd.batch_state;
        hp.mailbox   = hd.batch_mailbox;
        hp.offs[0]   = batch_lay.off_ids;
        hp.offs[1]   = batch_lay.off_slots;
        hp.offs[2]   = batch_lay.off_w;
        hp.offs[3]   = batch_lay.off_x;
        hp.offs[4]   = batch_lay.off_table;
        hp.off_y     = batch_lay.off_y;
        hp.off_yh    = batch_lay.off_yh;
        hp.off_yflag = batch_lay.off_yflag;
        hp.n_expert  = host_n_expert;
        hp.zero_base = 0;
        return true;
    }

    bool host_computes(const ggml_tensor * w, int64_t e) const {
        if (batch_tokens == 0) {
            return false;
        }
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return false;
        }
        return batch_cpu[(size_t) it->second.il*host_n_expert + e] != 0;
    }

    bool get_host_part(int32_t il, int64_t n_tokens, llama_moe_cache::host_part & hp) const {
        if (!host || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0 || n_tokens <= 0 || n_tokens > host_max_tokens) {
            return false;
        }
        // the zero slots are released while large batches run
        if (phases && prefill) {
            return false;
        }
        const group & g = groups[layers[il].ig];
        const host_device & hd = hdev[g.id];
        if (hd.mailbox == nullptr) {
            return false;
        }
        hp.backend   = devices[g.id].backend;
        hp.state     = hd.state;
        hp.mailbox   = hd.mailbox;
        hp.offs[0]   = host_lay.off_ids;
        hp.offs[1]   = host_lay.off_slots;
        hp.offs[2]   = host_lay.off_w;
        hp.offs[3]   = host_lay.off_x;
        hp.offs[4]   = host_lay.off_table;
        hp.off_y     = host_lay.off_y;
        hp.n_expert  = host_n_expert;
        hp.zero_base = g.n_slots;
        return true;
    }

    // admits the experts of group ig selected most that are not cached, into free slots or over cached ones selected less
    void host_plan(size_t ig, admit_job & job) {
        group & g = groups[ig];
        moe_cache_slru & sl = g.slru;
        host_device & hd = hdev[g.id];
        const int32_t ne = sl.n_expert;

        host_free.clear();
        host_vict.clear();
        for (int32_t s = 0; s < g.n_active; ++s) {
            if (sl.pending[s]) {
                continue;
            }
            (sl.key_of[s] < 0 ? host_free : host_vict).push_back(s);
        }
        host_cand.clear();
        for (int32_t il : g.layers) {
            const int32_t * slots = sl.slot_map[il];
            for (int32_t e = 0; e < ne; ++e) {
                const int32_t key = il*ne + e;
                if (slots[e] >= sl.n_slots && host_score[key] > 0.0f && sl.pend.count(key) == 0) {
                    host_cand.push_back(key);
                }
            }
        }

        const size_t n_fill = std::min(host_free.size(), (size_t) host_max_fills);
        const size_t n_cand = std::min(host_cand.size(), n_fill + host_max_swaps);
        const size_t n_vict = std::min(host_vict.size(), (size_t) host_max_swaps);
        std::partial_sort(host_cand.begin(), host_cand.begin() + n_cand, host_cand.end(),
            [&](int32_t a, int32_t b) { return host_score[a] > host_score[b]; });
        std::partial_sort(host_vict.begin(), host_vict.begin() + n_vict, host_vict.end(),
            [&](int32_t a, int32_t b) { return host_score[sl.key_of[a]] < host_score[sl.key_of[b]]; });

        size_t i_fill = 0;
        size_t i_vict = 0;
        for (size_t c = 0; c < n_cand; ++c) {
            const int32_t key = host_cand[c];
            int32_t s;
            if (i_fill < n_fill) {
                s = host_free[i_fill++];
                sl.n_filled++;
                host_n_fills++;
            } else {
                if (i_vict >= n_vict) {
                    break;
                }
                const int32_t v    = host_vict[i_vict];
                const int32_t vkey = sl.key_of[v];
                if (host_score[key] <= host_score[vkey]*host_hyst + 1.0f) {
                    break;
                }
                i_vict++;
                // the device computes the evicted expert as a miss from the next graph on, and its slot is overwritten
                // after the work queued before
                sl.slot_map[vkey / ne][vkey % ne] = sl.empty;
                hd.table_host[vkey] = -1;
                s = v;
                host_n_swaps++;
            }
            sl.key_of[s]  = key;
            sl.pending[s] = true;
            sl.pend[key]  = s;

            const int32_t il = key / ne;
            const int32_t e  = key % ne;
            const layer & l = layers[il];
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const size_t es = l.experts[ip]->nb[2];
                job.uploads.push_back({ g.id, g.banks[ip], hsrc(il, ip, e), s*es, es });
                async_bytes += es;
            }
            job.pubs.push_back({ (int32_t) ig, il, e, s });
        }
    }

    // before each graph: publish the uploads done, plan the cache now and then, update the slot tables of the devices
    void host_begin() {
        const int64_t t0 = ggml_time_us()*1000;
        host->wake();
        if (svc_policy) {
            host_n_plans++;
            return;
        }
        host_collect();
        // while large batches run, the uploads of the expert prefetch have the link
        if (!(phases && prefill)) {
            host_plan_all();
        }
        host_begin_ns += ggml_time_us()*1000 - t0;
        host_n_plans++;
    }

    // the uploads done become hits
    void host_collect() {
        std::vector<publication> ready;
        {
            std::lock_guard<std::mutex> lk(mtx);
            ready.swap(done);
        }
        for (const publication & pb : ready) {
            group & g = groups[pb.ig];
            host_device & hd = hdev[g.id];
            hd.in_flight--;
            const int32_t key = pb.il*g.slru.n_expert + pb.expert;
            if (g.slru.key_of[pb.slot] != key || !g.slru.pending[pb.slot]) {
                continue;
            }
            g.slru.publish(pb.il, pb.expert, pb.slot);
            hd.table_host[key] = pb.slot;
            n_admitted++;
            // an expert in an exclusive slot needs its host copy no more
            if (excl_on && pb.slot < g.n_excl && hloc[key] >= 0) {
                g.quarantine.push_back(hloc[key]);
                hloc[key] = -1;
            }
        }

        if (excl_on) {
            std::vector<stage> st;
            {
                std::lock_guard<std::mutex> lk(mtx);
                st.swap(excl_done);
            }
            for (const stage & x : st) {
                group & g = groups[x.ig];
                moe_cache_slru & sl = g.slru;
                excl_in_flight--;
                sl.staging[x.slot] = false;
                if (sl.key_of[x.slot] == x.key && !sl.pending[x.slot] && sl.staged[x.slot] < 0) {
                    sl.staged[x.slot] = x.loc;
                    excl_n_staged++;
                } else {
                    g.spares.push_back(x.loc);
                    excl_n_unstaged++;
                }
            }
        }
    }

    // the worker of the downloads: the victims of exclusive slots into spare host slots, on a stream of their own
    void run_excl() {
        for (;;) {
            stage_job job;
            {
                std::unique_lock<std::mutex> lk(mtx);
                excl_cv.wait(lk, [&] { return excl_quit || !excl_jobs.empty(); });
                if (excl_jobs.empty()) {
                    return;
                }
                job = std::move(excl_jobs.front());
                excl_jobs.pop_front();
            }
            ggml_backend_t b = excl_side[job.dev];
            for (const stage & x : job.stages) {
                const group & g = groups[x.ig];
                const layer & at = layers[x.loc / hp_ne];
                GGML_ASSERT(at.experts.size() == g.banks.size());
                for (size_t ip = 0; ip < g.banks.size(); ++ip) {
                    const ggml_tensor * w = at.experts[ip];
                    const size_t es = w->nb[2];
                    ggml_backend_tensor_get_async(b, g.banks[ip], (uint8_t *) w->data + (size_t) (x.loc % hp_ne)*es, (size_t) x.slot*es, es);
                }
                // one expert in flight at a time: the downloads of all streams of a device share a copy engine in order,
                // so the small ones of the graphs (results read back, the drafter's input) would wait for all those queued
                excl_sync(b);
            }
            std::lock_guard<std::mutex> lk(mtx);
            excl_done.insert(excl_done.end(), job.stages.begin(), job.stages.end());
        }
    }

    void host_plan_all() {
        if (++host_n_begin % host_period == 0) {
            host->take_counts(host_score, host_decay, profile_new.empty() ? nullptr : &profile_new);
            for (size_t id = 0; id < devices.size(); ++id) {
                if (id >= hdev.size() || hdev[id].state == nullptr || hdev[id].in_flight > 0) {
                    continue;
                }
                admit_job job;
                job.dev = (int32_t) id;
                for (size_t ig = 0; ig < groups.size(); ++ig) {
                    if (groups[ig].id == (int32_t) id && groups[ig].n_slots > 0) {
                        host_plan(ig, job);
                    }
                }
                if (job.pubs.empty()) {
                    continue;
                }
                hdev[id].in_flight += job.pubs.size();
                ggml_backend_event_record(gate_ev[id], devices[id].backend);
                {
                    std::lock_guard<std::mutex> lk(mtx);
                    jobs.push_back(std::move(job));
                }
                cv.notify_one();
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // trace format, little endian: "MCT1", then u32 n_layer, n_expert_used, n_group
    // per group: u32 device, n_expert, n_layers, the layers (u32), f64 split, split_sum, u64 group host bytes, device host bytes, alignment,
    //            u32 n_tensors, the bytes of one expert of each tensor (u64)
    // then one record per slot lookup: i32 layer, experts per token, n_ids, the ids (i32)
    void write_trace_header(double split_sum) {
        auto u32 = [&](uint32_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto u64 = [&](uint64_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto f64 = [&](double   v) { fwrite(&v, sizeof(v), 1, trace); };
        fwrite("MCT1", 1, 4, trace);
        u32(layers.size());
        u32(n_expert_used);
        u32(groups.size());
        for (const group & g : groups) {
            const device & d = devices[g.id];
            u32(g.id);
            u32(g.ref[0]->ne[2]);
            u32(g.layers.size());
            for (int32_t il : g.layers) {
                u32(il);
            }
            f64(d.split);
            f64(split_sum);
            u64(g.host_bytes);
            u64(d.host_bytes);
            u64(ggml_backend_buft_get_alignment(d.buft));
            u32(g.ref.size());
            for (const ggml_tensor * t : g.ref) {
                u64(t->nb[2]);
            }
        }
    }

    void write_trace(int32_t il, int32_t n_used) {
        const int32_t rec[3] = { il, n_used, (int32_t) ids.size() };
        fwrite(rec, sizeof(int32_t), 3, trace);
        fwrite(ids.data(), sizeof(int32_t), ids.size(), trace);
    }

    void run_admissions() {
        for (;;) {
            admit_job job;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&] { return quit || !jobs.empty(); });
                if (jobs.empty()) {
                    return;
                }
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            // the victims may still be read by work queued on the compute stream before the plan
            if (job.gate) {
                ggml_backend_event_wait(side[job.dev], gate_ev[job.dev]);
            }
            for (const upload & u : job.uploads) {
                ggml_backend_tensor_set_async(side[job.dev], u.bank, u.src, u.offs, u.size);
            }
            ggml_backend_event_record(done_ev[job.dev], side[job.dev]);
            ggml_backend_event_synchronize(done_ev[job.dev]);
            std::lock_guard<std::mutex> lk(mtx);
            done.insert(done.end(), job.pubs.begin(), job.pubs.end());
        }
    }

    ~impl() {
        // the admissions still queued are dropped, and the one in flight lands before the banks go
        if (worker.joinable()) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                quit = true;
                jobs.clear();
            }
            cv.notify_all();
            worker.join();
            for (size_t i = 0; i < side.size(); ++i) {
                ggml_backend_synchronize(side[i]);
                ggml_backend_event_free(gate_ev[i]);
                ggml_backend_event_free(done_ev[i]);
                ggml_backend_free(side[i]);
            }
        }
        if (excl_worker.joinable()) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                excl_quit = true;
            }
            excl_cv.notify_all();
            excl_worker.join();
        }
        for (ggml_backend_t b : excl_side) {
            if (b != nullptr) {
                ggml_backend_synchronize(b);
                ggml_backend_free(b);
            }
        }
        if (host) {
            profile_write();
            host->log_stats();
            if (host_n_fills + host_n_swaps + svc_n_fills + svc_n_swaps > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: host mode: %zu experts admitted into free slots, %zu replaced, %zu admitted again after large batches, %lld graphs, %.1f us before each graph\n",
                    host_n_fills + svc_n_fills, host_n_swaps + svc_n_swaps, n_readmitted, (long long) host_n_plans, host_n_plans > 0 ? host_begin_ns/1e3/host_n_plans : 0.0);
            }
            if (excl_on) {
                size_t n_spare = 0;
                for (const group & g : groups) {
                    n_spare += g.spares.size() + g.quarantine.size();
                }
                LLAMA_LOG_INFO("llama_moe_cache: exclusive core: %.2f GiB given back, %zu victims downloaded, %zu of them evicted "
                    "with that copy, %zu downloads unused, %zu swaps not started without a download, %zu spare host slots at the end\n",
                    excl_released/1024.0/1024.0/1024.0, excl_n_staged, excl_n_exchanges, excl_n_unstaged, excl_n_waited, n_spare);
            }
            if (batch_graphs > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: %llu large batches shared with the host, %.0f%% of the experts the core lacks on average\n",
                    (unsigned long long) batch_graphs, 100.0*batch_share/batch_graphs);
            }
            host.reset();
        }
        for (host_device & hd : hdev) {
            if (hd.mailbox != nullptr && hd.mailbox_free != nullptr) {
                hd.mailbox_free(hd.mailbox);
            }
            if (hd.batch_mailbox != nullptr && hd.mailbox_free != nullptr) {
                hd.mailbox_free(hd.batch_mailbox);
            }
        }
        aliases_free();
        for (ggml_backend_buffer_t buf : bank_bufs) {
            ggml_backend_buffer_free(buf);
        }
        log_stats();
        if (trace != nullptr) {
            fclose(trace);
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // in host mode, the host computes the misses of small batches instead
        if (host_mode) {
            return nullptr;
        }

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    const void * host_data(const ggml_tensor * w, int64_t e) const {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return (const uint8_t *) w->data + e*w->nb[2];
        }
        const binding & b = it->second;
        if (excl_on && hloc[(size_t) b.il*hp_ne + e] < 0) {
            return nullptr;
        }
        return hsrc(b.il, b.ip, (int32_t) e);
    }

    bool cached(ggml_backend_t backend, const ggml_tensor * w, int64_t e) const {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return false;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        const int32_t * slots = g.slru.slot_map[b.il];
        return backend == devices[g.id].backend && slots != nullptr && slots[e] < g.n_slots;
    }

    static const ggml_tensor * base_of(const ggml_tensor * t) {
        while (t->view_src != nullptr) {
            t = t->view_src;
        }
        return t;
    }

    // an input of the CPU part of the last layer, it was read from the device with the selected experts
    bool copy_cpu_input(const ggml_tensor * src, ggml_tensor * dst) const {
        if (stash.empty() || dst->buffer == nullptr || !ggml_backend_buffer_is_host(dst->buffer) || src->data == nullptr) {
            return false;
        }
        const ggml_tensor * base = base_of(src);
        const size_t offs = (const uint8_t *) src->data - (const uint8_t *) base->data;
        const size_t size = ggml_nbytes(src);
        for (const stash_entry & e : stash) {
            if (e.base == base && offs >= e.offs && offs + size <= e.offs + e.size) {
                memcpy(dst->data, (const uint8_t *) ggml_backend_buffer_get_base(stash_buf.get()) + e.pos + (offs - e.offs), size);
                return true;
            }
        }
        return false;
    }

    void set_cpu_part(const ggml_tensor * lookup, int32_t n_used, std::vector<const ggml_tensor *> inputs, const ggml_tensor * output) {
        cpu_parts[lookup] = { n_used, std::move(inputs) };
        if (output != nullptr) {
            cpu_outputs.insert(output);
        }
    }

    // the result of a CPU part, the device copies it after the work queued on it, without waiting on the host
    // the next CPU part runs after the next slot lookup, which synchronizes the device, so the result is not overwritten before it is copied
    bool copy_cpu_output(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) const {
        if (cpu_outputs.count(src) == 0 || src->buffer == nullptr || !ggml_backend_buffer_is_host(src->buffer) ||
            dst->buffer == nullptr || ggml_backend_buffer_is_host(dst->buffer)) {
            return false;
        }
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }

    void begin() {
        stash.clear();
        if (host) {
            host_begin();
        }
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return copy_cpu_input(src, dst) || copy_cpu_output(backend, src, dst);
        }
        const int32_t il = it->second;
        layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        // the device inputs of the CPU part are read with the selected experts, so the CPU part does not wait for the device
        // the inputs computed in this split are left to the scheduler
        // the graph has a CPU part for this layer if it registered its inputs
        const auto it_cpu = cpu_parts.find(lookup);
        const bool has_cpu = l.cpu_map != nullptr && it_cpu != cpu_parts.end() && !it_cpu->second.inputs.empty();

        stash.clear();
        size_t stash_size = 0;
        if (has_cpu) {
            for (const ggml_tensor * t : it_cpu->second.inputs) {
                const ggml_tensor * base = base_of(t);
                if (base->buffer == nullptr || ggml_backend_buffer_is_host(base->buffer) || t->data == nullptr) {
                    continue;
                }
                bool in_split = false;
                for (int i = 0; i < n_nodes && !in_split; ++i) {
                    in_split = ggml_graph_node(graph, i) == base;
                }
                if (in_split) {
                    continue;
                }
                const size_t offs = (const uint8_t *) t->data - (const uint8_t *) base->data;
                stash.push_back({ base, offs, ggml_nbytes(t), stash_size });
                stash_size += GGML_PAD(ggml_nbytes(t), 64);
            }
            if (!stash_buf || ggml_backend_buffer_get_size(stash_buf.get()) < stash_size) {
                ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
                stash_buf.reset(ggml_backend_buft_alloc_buffer(buft ? buft : ggml_backend_cpu_buffer_type(), 2*stash_size));
                GGML_ASSERT(stash_buf);
            }
            uint8_t * stash_data = (uint8_t *) ggml_backend_buffer_get_base(stash_buf.get());
            for (const stash_entry & e : stash) {
                ggml_backend_tensor_get_async(backend, e.base, stash_data + e.pos, e.offs, e.size);
            }
        }

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        if (trace != nullptr) {
            write_trace(il, it_cpu != cpu_parts.end() ? it_cpu->second.n_used : n_expert_used);
        }

        // admissions uploaded since the last plan become hits; after the synchronization above, so the slot maps
        // uploaded earlier from host memory are no longer read
        if (async_admit) {
            std::vector<publication> ready;
            {
                std::lock_guard<std::mutex> lk(mtx);
                ready.swap(done);
            }
            for (const publication & pb : ready) {
                groups[pb.ig].slru.publish(pb.il, pb.expert, pb.slot);
            }
            in_flight -= std::min(in_flight, ready.size());
        }

        // part of the misses is computed on the CPU once the cache is full, the carry keeps the share right on average
        const uint32_t share_q = has_cpu && g.slru.n_filled == g.slru.n_slots ? cpu_share_q : 0;
        auto n_cpu = [&](size_t n_miss) {
            const uint64_t acc = (uint64_t) n_miss*share_q + l.cpu_acc;
            l.cpu_acc = acc & 0xffff;
            return (size_t) (acc >> 16);
        };
        size_t n_hit = 0;
        size_t n_skip = 0;
        const size_t max_admit = async_admit && has_cpu ? async_max - std::min(async_max, in_flight) : 0;
        if (!g.slru.plan(il, ids.data(), ids.size(), has_cpu ? it_cpu->second.n_used : 1, n_zero, has_cpu ? skip_rank : 0,
                n_cpu, rng, fills, cpu, n_hit, n_skip, max_admit, admits)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        if (has_cpu) {
            int32_t * cpu_map = (int32_t *) l.cpu_map->data;
            for (int32_t e : l.cpu_experts) {
                cpu_map[e] = -1;
            }
            l.cpu_experts.clear();
            for (const auto & f : cpu) {
                cpu_map[f.expert] = f.expert;
                l.cpu_experts.push_back(f.expert);
            }
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                const uint8_t * src = hsrc(il, ip, fills[i].expert);
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n &&
                        hsrc(il, ip, fills[i + n].expert) == src + n*expert_size) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, src, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        // the admitted experts are uploaded in the background
        if (!admits.empty()) {
            admit_job job;
            job.dev = g.id;
            for (const auto & a : admits) {
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const size_t expert_size = l.experts[ip]->nb[2];
                    job.uploads.push_back({ g.id, g.banks[ip], hsrc(il, ip, a.expert), a.slot*expert_size, expert_size });
                    async_bytes += expert_size;
                }
                job.pubs.push_back({ l.ig, il, a.expert, a.slot });
            }
            in_flight += admits.size();
            n_admitted += admits.size();
            ggml_backend_event_record(gate_ev[g.id], backend);
            {
                std::lock_guard<std::mutex> lk(mtx);
                jobs.push_back(std::move(job));
            }
            cv.notify_one();
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size() + cpu.size() + n_skip;
        st.cpu    += cpu.size();
        st.skip   += n_skip;
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu (%zu on the CPU, %zu skipped), hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, st.cpu, st.skip, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (n_admitted > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: admitted in the background: %zu experts, %.2f MiB\n", n_admitted, async_bytes/1024.0/1024.0);
        }
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
    pimpl(new impl(model, no_alloc, backends, bufts, size, cpu_share)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

float llama_moe_cache::cpu_share() const {
    return pimpl->cpu_share;
}

ggml_tensor * llama_moe_cache::get_cpu_map(int32_t il) const {
    return il >= 0 && il < (int32_t) pimpl->layers.size() ? pimpl->layers[il].cpu_map : nullptr;
}

void llama_moe_cache::set_cpu_part(const ggml_tensor * lookup, int32_t n_used, std::vector<const ggml_tensor *> inputs, const ggml_tensor * output) {
    pimpl->set_cpu_part(lookup, n_used, std::move(inputs), output);
}

void llama_moe_cache::begin() {
    pimpl->begin();
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

const void * llama_moe_cache::host_data(const ggml_tensor * w, int64_t e) const {
    return pimpl->host_data(w, e);
}

bool llama_moe_cache::cached(ggml_backend_t backend, const ggml_tensor * w, int64_t e) const {
    return pimpl->cached(backend, w, e);
}

void llama_moe_cache::init_batch(int32_t max_batch) {
    pimpl->init_batch(max_batch);
}

void llama_moe_cache::begin_batch(int64_t n_tokens, double upload_rate) {
    pimpl->begin_batch(n_tokens, upload_rate);
}

ggml_backend_buffer_type_t llama_moe_cache::lent_buft(ggml_backend_t backend) const {
    return pimpl->lent_buft(backend);
}

void llama_moe_cache::lend(bool on) {
    pimpl->lend(on);
}

bool llama_moe_cache::batch_shared() const {
    return pimpl->batch_tokens > 0;
}

bool llama_moe_cache::end_batch_wanted() const {
    return pimpl->end_batch_wanted();
}

void llama_moe_cache::end_batch(double seconds, int64_t bytes_uploaded, double upload_rate) {
    pimpl->end_batch(seconds, bytes_uploaded, upload_rate);
}

bool llama_moe_cache::get_batch_part(int32_t il, int64_t n_tokens, host_part & hp) const {
    return pimpl->get_batch_part(il, n_tokens, hp);
}

bool llama_moe_cache::host_computes(const ggml_tensor * w, int64_t e) const {
    return pimpl->host_computes(w, e);
}

bool llama_moe_cache::get_host_part(int32_t il, int64_t n_tokens, host_part & hp) const {
    return pimpl->get_host_part(il, n_tokens, hp);
}

bool llama_moe_cache::has_phases() const {
    return pimpl->phases;
}

void llama_moe_cache::set_phase(bool large) {
    pimpl->set_phase(large);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    for (ggml_backend_buffer_t buf : pimpl->bank_bufs) {
        res[ggml_backend_buffer_get_type(buf)] += ggml_backend_buffer_get_size(buf);
    }
    return res;
}
