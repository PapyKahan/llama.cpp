#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

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

struct llama_moe_cache::impl {
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
        int32_t n_slots = 0;
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

        // with misses on the CPU, the n_zero slots after the cached ones stay zero: the device computes these misses as zero
        // one extra slot at the end, CUDA MMQ can read past the last expert
        n_zero = cpu_share > 0.0f ? std::min<int32_t>(n_expert_used, 64) : 0;
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
        if (const char * v = getenv("LLAMA_MOE_CACHE_ASYNC"); v != nullptr && atoi(v) != 0) {
            if (n_zero == 0) {
                LLAMA_LOG_WARN("%s: LLAMA_MOE_CACHE_ASYNC needs --moe-cache-cpu-share > 0, ignored\n", __func__);
            } else {
                async_admit = true;
                if (const char * m = getenv("LLAMA_MOE_CACHE_ASYNC_MAX"); m != nullptr) {
                    async_max = std::max(1, atoi(m));
                }
                for (const device & d : devices) {
                    ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
                    side.push_back(ggml_backend_dev_init(dev, nullptr));
                    gate_ev.push_back(ggml_backend_event_new(dev));
                    done_ev.push_back(ggml_backend_event_new(dev));
                    GGML_ASSERT(side.back() != nullptr && gate_ev.back() != nullptr && done_ev.back() != nullptr);
                }
                worker = std::thread([this] { run_admissions(); });
                LLAMA_LOG_INFO("%s: asynchronous admission, up to %zu experts in flight\n", __func__, async_max);
            }
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
            g.slru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(n_zero > 0 ? 2 : 1); // slot map and CPU map
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
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
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
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
            ggml_backend_event_wait(side[job.dev], gate_ev[job.dev]);
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
    return res;
}
