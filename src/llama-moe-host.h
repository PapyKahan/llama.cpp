#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <vector>

// computes the experts that a device does not cache for the requests it posts to mailboxes in mapped host memory
// (ggml_moe_host_post), and answers them (ggml_moe_host_wait): a service thread polls the mailboxes and computes the
// missing experts of each request with a pool of workers, while the device computes its cached experts
class llama_moe_host {
public:
    // the host experts of a layer, gate and up [n_embd, n_ff, n_expert], down [n_ff, n_embd, n_expert]
    // a layer without experts in host mode has null tensors
    struct layer {
        const ggml_tensor * gate  = nullptr;
        const ggml_tensor * up    = nullptr;
        const ggml_tensor * down  = nullptr;
        float               limit = 0.0f; // SwiGLU clamp of the experts, 0 if none
    };

    // byte offsets of the arrays of a mailbox, after its header
    struct mailbox_layout {
        int32_t off_ids;   // i32 [max_tokens*n_used]
        int32_t off_slots; // i32 [max_tokens*n_used], the device slot of each selected expert, -1 if not cached
        int32_t off_w;     // f32 [max_tokens*n_used]
        int32_t off_x;     // f32 [max_tokens*n_embd]
        int32_t off_y;     // f32 [max_tokens*n_embd]
        int32_t off_table; // i32 [n_layer*n_expert], the slot of each expert in the device cache, -1 if not cached
        int32_t off_yh;    // f16 [max_tokens*n_embd], batch requests: the answer of the tokens it flags
        int32_t off_yflag; // i32 [max_tokens], batch requests: 1 if the host computed part of the token, 0 if not
        size_t  size;
    };

    static mailbox_layout layout(int32_t max_tokens, int32_t n_used, int32_t n_embd, int32_t n_layer, int32_t n_expert);

    // whether the host can compute the experts of a layer
    static bool supported(const layer & l, int32_t n_embd, int32_t n_ff);

    // n_threads: threads computing the experts, the service thread included
    // skip_rank: a miss that every token of the request ranks skip_rank-th or later is skipped, 0 = none
    llama_moe_host(std::vector<layer> layers, int32_t n_expert, int32_t n_used, int32_t n_embd, int32_t n_ff,
                   int32_t max_tokens, int n_threads, int32_t skip_rank);
    ~llama_moe_host();

    // serves a mailbox of the given layout from now on; it must stay allocated until the destruction
    void add_mailbox(void * mailbox);

    // serves a mailbox of layout(max_batch, ...) from now on, for whole large batches (ggml_moe_host_post_batch): the
    // host computes the entries the device marks -1, one expert at a time for all its tokens
    void add_batch_mailbox(void * mailbox, int32_t max_batch);

    // makes the service thread poll without sleeping for a while: call before a graph that posts requests
    void wake();

    // called by the service thread for each request: before_answer while the device waits for the answer, so the
    // device reads no slot table and no slot meanwhile except the slots of layer il, then after_answer, while the
    // device runs on to the next request
    struct listener {
        virtual ~listener() = default;
        virtual void before_answer(int32_t il) = 0;
        virtual void after_answer(int32_t il)  = 0;
    };

    // before the first request
    void set_listener(listener * l);

    // adds the selection counts of each expert since the last call to counts [n_layer*n_expert], then resets them
    // total, if not null: also adds them there, undecayed
    void take_counts(std::vector<float> & counts, float decay, std::vector<float> * total = nullptr);

    void log_stats() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
