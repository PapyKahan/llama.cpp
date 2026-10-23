#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-host.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
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
        bool                     gate = true; // wait for the work queued on the compute stream before the job
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
    std::unique_ptr<llama_moe_host> host;
    llama_moe_host::mailbox_layout  host_lay;
    struct host_device {
        ggml_tensor * state   = nullptr;
        void *        mailbox = nullptr;
        void (*mailbox_free)(void *) = nullptr;
        int32_t * table_host = nullptr; // [n_layer*n_expert] in the mailbox, read by the graphs started afterwards
        size_t in_flight = 0;           // experts being uploaded
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
    size_t  prefill_size = 0;
    std::vector<ggml_backend_buffer_t> bank_bufs; // one buffer per bank, its physical memory mapped up to the phase's slots
    std::vector<ggml_tensor *>         bank_tensors;
    std::vector<int32_t>               bank_group;
    size_t (*vmm_commit)(ggml_backend_buffer_t, size_t) = nullptr;
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

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
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

        if (const char * v = getenv("LLAMA_MOE_HOST"); v != nullptr && atoi(v) != 0 && !model.hparams.no_alloc) {
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

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            if (phases) {
                phases_alloc();
            }
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                // with phases the banks have their own buffers already: this allocates the rest and the views
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
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
                ggml_backend_buffer_t buf = vmm_alloc(index, size, core);
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
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                if (groups[ig].n_slots > 0) {
                    evict_range(ig, groups[ig].n_core, groups[ig].n_slots);
                    groups[ig].n_active = groups[ig].n_core;
                }
            }
            for (const device & d : devices) {
                ggml_backend_synchronize(d.backend);
            }
            for (size_t i = 0; i < bank_bufs.size(); ++i) {
                vmm_commit(bank_bufs[i], (size_t) groups[bank_group[i]].n_core*bank_tensors[i]->nb[2]);
            }
        } else {
            for (size_t i = 0; i < bank_bufs.size(); ++i) {
                ggml_tensor * bank = bank_tensors[i];
                const size_t full = ggml_backend_buffer_get_size(bank_bufs[i]);
                const size_t core = (size_t) groups[bank_group[i]].n_core*bank->nb[2];
                if (vmm_commit(bank_bufs[i], full) < full) {
                    GGML_ABORT("the MoE cache could not map its banks again");
                }
                // the zero slots after the cached ones must read as zero
                ggml_backend_tensor_memset(bank, 0, core, ggml_nbytes(bank) - core);
            }
            for (group & g : groups) {
                g.n_active = g.n_slots;
            }
        }
        prefill = large;
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

        const int32_t n_layer = (int32_t) layers.size();
        int32_t n_embd = 0;
        int32_t n_ff   = 0;
        std::vector<llama_moe_host::layer> hl(n_layer);
        for (const group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            for (int32_t il : g.layers) {
                const llama_layer & ly = model.layers[il];
                const float limit = model.hparams.swiglu_clamp_exp[il];
                hl[il] = { ly.ffn_gate_exps, ly.ffn_up_exps, ly.ffn_down_exps, limit > 1e-6f ? limit : 0.0f };
                n_embd = ly.ffn_gate_exps->ne[0];
                n_ff   = ly.ffn_gate_exps->ne[1];
            }
        }
        host_lay = llama_moe_host::layout(host_max_tokens, n_expert_used, n_embd, n_layer, host_n_expert);

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
    }

    // the swaps of the next graph: admissions into free slots, then replacements of cached experts selected less,
    // the replacements that gain the most first
    void svc_plan_graph() {
        host->take_counts(host_score, host_decay);
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
    }

    // the service thread makes no CUDA call: a graph waiting for its answer may hold the driver while it is launched
    void before_answer(int32_t il) override {
        std::lock_guard<std::mutex> lk(svc_mtx);
        host_collect();
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
                    const ggml_tensor * w = l.experts[ip];
                    const size_t es = w->nb[2];
                    job.uploads.push_back({ g.id, g.banks[ip], (const uint8_t *) w->data + pb.expert*es, pb.slot*es, es });
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
                const ggml_tensor * w = l.experts[ip];
                const size_t es = w->nb[2];
                job.uploads.push_back({ g.id, g.banks[ip], (const uint8_t *) w->data + e*es, s*es, es });
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
        }
    }

    void host_plan_all() {
        if (++host_n_begin % host_period == 0) {
            host->take_counts(host_score, host_decay);
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
        if (host) {
            host->log_stats();
            if (host_n_fills + host_n_swaps + svc_n_fills + svc_n_swaps > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: host mode: %zu experts admitted into free slots, %zu replaced, %lld graphs, %.1f us before each graph\n",
                    host_n_fills + svc_n_fills, host_n_swaps + svc_n_swaps, (long long) host_n_plans, host_n_plans > 0 ? host_begin_ns/1e3/host_n_plans : 0.0);
            }
            host.reset();
        }
        for (host_device & hd : hdev) {
            if (hd.mailbox != nullptr && hd.mailbox_free != nullptr) {
                hd.mailbox_free(hd.mailbox);
            }
        }
        for (ggml_backend_buffer_t buf : bank_bufs) {
            ggml_backend_buffer_free(buf);
        }
        if (worker.joinable()) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                quit = true;
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
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
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
                    const ggml_tensor * w = l.experts[ip];
                    const size_t expert_size = w->nb[2];
                    job.uploads.push_back({ g.id, g.banks[ip], (const uint8_t *) w->data + a.expert*expert_size, a.slot*expert_size, expert_size });
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

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
    pimpl(new impl(model, backends, bufts, size, cpu_share)) {
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

bool llama_moe_cache::cached(ggml_backend_t backend, const ggml_tensor * w, int64_t e) const {
    return pimpl->cached(backend, w, e);
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
