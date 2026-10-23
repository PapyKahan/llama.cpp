#include "moe-host.cuh"

// how long the device waits for the host before it gives up, the layer then lacks the host's experts
static constexpr uint64_t MOE_HOST_TIMEOUT_NS = 10ull*1000*1000*1000;

static __device__ __forceinline__ uint64_t moe_host_globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __device__ __forceinline__ int32_t moe_host_load_acquire(const int32_t * p) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    int32_t v;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
#else
    const int32_t v = *(const volatile int32_t *) p;
    __threadfence_system();
    return v;
#endif
}

static __device__ __forceinline__ void moe_host_store_release(int32_t * p, int32_t v) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    asm volatile("st.release.sys.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
#else
    __threadfence_system();
    *(volatile int32_t *) p = v;
#endif
}

// one block: the slots of the selected experts, then the request in the mailbox, then its sequence number
static __global__ void k_moe_host_post(
        const char * __restrict__ ids, const int64_t ids_nb0, const int64_t ids_nb1,
        const char * __restrict__ w,   const int64_t w_nb0,   const int64_t w_nb1,
        const char * __restrict__ x,   const int64_t x_nb0,   const int64_t x_nb1,
        int32_t * __restrict__ state, int32_t * __restrict__ slots,
        char * __restrict__ mb, const int off_ids, const int off_slots, const int off_w, const int off_x, const int off_table,
        const int n_expert, const int n_used, const int n_tokens, const int n_embd, const int layer, const int zero_base) {
    const int tid = threadIdx.x;

    // the host writes the table between graphs, a graph reads it from host memory
    const int32_t * table = (const int32_t *) (mb + off_table) + (int64_t) layer*n_expert;

    int32_t * mb_ids   = (int32_t *) (mb + off_ids);
    int32_t * mb_slots = (int32_t *) (mb + off_slots);
    float   * mb_w     = (float   *) (mb + off_w);
    float   * mb_x     = (float   *) (mb + off_x);

    const int n = n_used*n_tokens;
    for (int i = tid; i < n; i += blockDim.x) {
        const int k = i % n_used;
        const int t = i / n_used;
        const int32_t e = *(const int32_t *) (ids + k*ids_nb0 + t*ids_nb1);
        const int32_t s = __ldcv(table + e);
        const int32_t slot = s >= 0 ? s : zero_base + k;
        slots[i]    = slot;
        mb_ids[i]   = e;
        mb_slots[i] = s;
        mb_w[i]     = *(const float *) (w + k*w_nb0 + t*w_nb1);
    }

    if (x_nb0 == (int64_t) sizeof(float) && n_embd % 4 == 0 && x_nb1 % 16 == 0 && (uintptr_t) x % 16 == 0 && off_x % 16 == 0) {
        const int n4 = n_embd/4;
        for (int i = tid; i < n4*n_tokens; i += blockDim.x) {
            const int j = i % n4;
            const int t = i / n4;
            ((float4 *) mb_x)[i] = ((const float4 *) (x + t*x_nb1))[j];
        }
    } else {
        for (int i = tid; i < n_embd*n_tokens; i += blockDim.x) {
            const int j = i % n_embd;
            const int t = i / n_embd;
            mb_x[i] = *(const float *) (x + j*x_nb0 + t*x_nb1);
        }
    }

    __threadfence_system();
    __syncthreads();

    if (tid == 0) {
        const int32_t seq = state[0] + 1;
        state[0] = seq;
        int32_t * hdr = (int32_t *) mb;
        hdr[GGML_MOE_HOST_REQ_LAYER]    = layer;
        hdr[GGML_MOE_HOST_REQ_N_TOKENS] = n_tokens;
        hdr[GGML_MOE_HOST_REQ_N_USED]   = n_used;
        __threadfence_system();
        moe_host_store_release(hdr + GGML_MOE_HOST_REQ_SEQ, seq);
    }
}

// every block waits for the answer to the last request, then adds it
static __global__ void k_moe_host_wait(
        const char * __restrict__ a, const int64_t a_nb0, const int64_t a_nb1,
        float * __restrict__ dst, const int32_t * __restrict__ state, char * __restrict__ mb, const int off_y,
        const int n_embd, const int n_tokens) {
    __shared__ int32_t has_y;

    if (threadIdx.x == 0) {
        int32_t * hdr = (int32_t *) mb;
        const int32_t seq = state[0];
        const uint64_t t0 = moe_host_globaltimer();
        int32_t got;
        while ((got = moe_host_load_acquire(hdr + GGML_MOE_HOST_RESP_SEQ)) != seq) {
            if (moe_host_globaltimer() - t0 > MOE_HOST_TIMEOUT_NS) {
                break;
            }
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(128);
#endif
        }
        if (got == seq) {
            has_y = moe_host_load_acquire(hdr + GGML_MOE_HOST_RESP_HAS_Y);
        } else {
            has_y = 0;
            hdr[GGML_MOE_HOST_RESP_TIMEOUT] = seq;
            __threadfence_system();
        }
    }
    __syncthreads();

    const float * y = (const float *) (mb + off_y);
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n_embd*n_tokens; i += gridDim.x*blockDim.x) {
        const int j = i % n_embd;
        const int t = i / n_embd;
        float v = *(const float *) (a + j*a_nb0 + t*a_nb1);
        if (has_y) {
            v += __ldcv(y + i);
        }
        dst[i] = v;
    }
}

void ggml_cuda_op_moe_host_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ids     = dst->src[0];
    const ggml_tensor * weights = dst->src[1];
    const ggml_tensor * x       = dst->src[2];
    const ggml_tensor * state   = dst->src[3];

    GGML_ASSERT(ggml_is_contiguous(dst));

    void * mb;
    memcpy(&mb, dst->op_params, sizeof(mb));
    const int32_t layer     = ggml_get_op_params_i32(dst, 2);
    const int32_t zero_base = ggml_get_op_params_i32(dst, 3);
    const int32_t off_ids   = ggml_get_op_params_i32(dst, 4);
    const int32_t off_slots = ggml_get_op_params_i32(dst, 5);
    const int32_t off_w     = ggml_get_op_params_i32(dst, 6);
    const int32_t off_x     = ggml_get_op_params_i32(dst, 7);
    const int32_t off_table = ggml_get_op_params_i32(dst, 8);
    const int32_t n_expert  = ggml_get_op_params_i32(dst, 9);

    k_moe_host_post<<<1, 512, 0, ctx.stream()>>>(
        (const char *) ids->data,     ids->nb[0],     ids->nb[1],
        (const char *) weights->data, weights->nb[0], weights->nb[1],
        (const char *) x->data,       x->nb[0],       x->nb[1],
        (int32_t *) state->data, (int32_t *) dst->data,
        (char *) mb, off_ids, off_slots, off_w, off_x, off_table,
        n_expert, (int) ids->ne[0], (int) ids->ne[1], (int) x->ne[0], layer, zero_base);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_moe_host_wait(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a     = dst->src[0];
    const ggml_tensor * state = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(dst));

    void * mb;
    memcpy(&mb, dst->op_params, sizeof(mb));
    const int32_t off_y = ggml_get_op_params_i32(dst, 2);

    const int n = (int) (a->ne[0]*a->ne[1]);
    const int n_blocks = std::min(8, (n + 255)/256);
    k_moe_host_wait<<<n_blocks, 256, 0, ctx.stream()>>>(
        (const char *) a->data, a->nb[0], a->nb[1], (float *) dst->data, (const int32_t *) state->data,
        (char *) mb, off_y, (int) a->ne[0], (int) a->ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
