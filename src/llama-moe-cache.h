#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // cpu_share is the share of the misses computed on the CPU instead of uploaded
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // the share of the misses that the CPU computes while the device uploads the others, 0 if all misses are uploaded
    float cpu_share() const;

    // the experts of layer il computed on the CPU: I32 [1, n_expert] in host memory, e for these experts, -1 for the others
    ggml_tensor * get_cpu_map(int32_t il) const;

    // the CPU part of the layer of the slot lookup, with n_used experts per token, no inputs if the graph has no CPU part
    // its inputs on the device are read together with the selected experts, so the CPU part does not wait for the device
    // its output is copied to the device without waiting on the host
    void set_cpu_part(const ggml_tensor * lookup, int32_t n_used, std::vector<const ggml_tensor *> inputs, const ggml_tensor * output);

    // called before each graph compute
    void begin();

    // ggml_backend_sched copy callback, returns false if src is not a slot map or an input of the CPU part
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    // whether expert e of the host weight w is in the cache of the device of backend
    bool cached(ggml_backend_t backend, const ggml_tensor * w, int64_t e) const;

    // the host copy of expert e of the host weight w, which the cache may move (LLAMA_MOE_CACHE_EXCLUSIVE); nullptr if
    // the expert has none, it is then in the cache
    const void * host_data(const ggml_tensor * w, int64_t e) const;

    // host mode (LLAMA_MOE_HOST): for small batches the device computes the cached experts of a layer and the host the
    // others (llama-moe-host), a layer posts its request and waits for the answer with these
    struct host_part {
        ggml_backend_t backend   = nullptr;
        ggml_tensor *  state     = nullptr; // I32 on the device, the state of the mailbox
        void *         mailbox   = nullptr;
        int32_t        offs[5]   = {};      // the ids, slots, weights, input and slot table in the mailbox
        int32_t        off_y     = 0;       // the answer in the mailbox
        int32_t        off_yh    = 0;       // batches: the answer in half precision, and the flags of its rows
        int32_t        off_yflag = 0;
        int32_t        n_expert  = 0;
        int32_t        zero_base = 0;       // the first zero slot of the bank of the layer
    };

    // false if layer il does not run in host mode for n_tokens
    bool get_host_part(int32_t il, int64_t n_tokens, host_part & hp) const;

    // host mode with phases: large batches of up to max_batch tokens may share the experts the core lacks with the host
    void init_batch(int32_t max_batch);

    // before the graph of a large batch of n_tokens (0 before any other graph): which experts the host computes for it;
    // upload_rate: the bytes per second the expert uploads get, 0 if not measured
    void begin_batch(int64_t n_tokens, double upload_rate);

    // whether the batch begun wants end_batch, once its graph is done: seconds from the start of its graph to the end of
    // the device's work, the bytes uploaded for it
    bool end_batch_wanted() const;
    void end_batch(double seconds, int64_t bytes_uploaded, double upload_rate);

    // whether the batch begun is shared with the host: its graph posts to the host, so no graph without the posts is
    // reused for it, nor one with them for a batch that is not
    bool batch_shared() const;

    // false if layer il does not share its n_tokens with the host; the part posts with ggml_moe_host_post_batch
    bool get_batch_part(int32_t il, int64_t n_tokens, host_part & hp) const;

    // whether the host computes expert e of the host weight w in the current large batch, so it is not uploaded
    bool host_computes(const ggml_tensor * w, int64_t e) const;

    // phases (LLAMA_MOE_CACHE_PREFILL_MIB): the cache lends part of its memory while large batches run
    bool has_phases() const;

    // before the batches of the other phase: large batches run with the core of the cache and the memory it lends,
    // small ones in host mode with the whole cache; the context then reserves its compute buffers for the phase
    void set_phase(bool large);

    // phases: the buffer type of the backend's compute buffers, which places them in the memory the cache lends while
    // lend(true), from the end of set_phase(true) to the start of the change back; nullptr if none
    ggml_backend_buffer_type_t lent_buft(ggml_backend_t backend) const;
    void lend(bool on);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
