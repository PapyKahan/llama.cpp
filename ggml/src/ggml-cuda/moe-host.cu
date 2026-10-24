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

// one block: the slots of the selected experts, then the request in the mailbox, then its sequence number; a pad
// token gets the zero slots and id -1, which the host skips
static __global__ void k_moe_host_post(
        const char * __restrict__ ids, const int64_t ids_nb0, const int64_t ids_nb1,
        const char * __restrict__ w,   const int64_t w_nb0,   const int64_t w_nb1,
        const char * __restrict__ x,   const int64_t x_nb0,   const int64_t x_nb1,
        const int32_t * __restrict__ pad, int32_t * __restrict__ state, int32_t * __restrict__ slots,
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
        const bool    p = pad != nullptr && pad[t] != 0;
        const int32_t e = p ? -1 : *(const int32_t *) (ids + k*ids_nb0 + t*ids_nb1);
        const int32_t s = p ? -1 : __ldcv(table + e);
        const int32_t slot = s >= 0 ? s : zero_base + k;
        slots[i]    = slot;
        mb_ids[i]   = e;
        mb_slots[i] = s;
        mb_w[i]     = p ? 0.0f : *(const float *) (w + k*w_nb0 + t*w_nb1);
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

// batch mode: the layer's table row, the experts marked -1 go to the host and read as -1 for the device; the sequence
// number is taken here and published by k_moe_host_signal once x is in the mailbox
static __global__ void k_moe_host_post_batch(
        const char * __restrict__ ids, const int64_t ids_nb0, const int64_t ids_nb1,
        const char * __restrict__ w,   const int64_t w_nb0,   const int64_t w_nb1,
        int32_t * __restrict__ state, int32_t * __restrict__ ids_dev,
        char * __restrict__ mb, const int off_ids, const int off_slots, const int off_w, const int off_table,
        const int n_expert, const int n_used, const int n_tokens, const int layer) {
    extern __shared__ int32_t table_s[];
    const int32_t * table = (const int32_t *) (mb + off_table) + (int64_t) layer*n_expert;
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        table_s[e] = __ldcv(table + e);
    }
    __syncthreads();

    int32_t * mb_ids   = (int32_t *) (mb + off_ids);
    int32_t * mb_slots = (int32_t *) (mb + off_slots);
    float   * mb_w     = (float   *) (mb + off_w);

    const int n = n_used*n_tokens;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        const int k = i % n_used;
        const int t = i / n_used;
        const int32_t e = *(const int32_t *) (ids + k*ids_nb0 + t*ids_nb1);
        const int32_t s = table_s[e];
        ids_dev[i]  = s < 0 ? -1 : e;
        mb_ids[i]   = e;
        mb_slots[i] = s;
        mb_w[i]     = *(const float *) (w + k*w_nb0 + t*w_nb1);
    }
    __threadfence_system();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int32_t seq = state[0] + 1;
        state[0] = seq;
        state[1] = seq;
    }
}

static __global__ void k_moe_host_signal(const int32_t * __restrict__ state, char * __restrict__ mb, const int layer,
        const int n_tokens, const int n_used) {
    int32_t * hdr = (int32_t *) mb;
    hdr[GGML_MOE_HOST_REQ_LAYER]    = layer;
    hdr[GGML_MOE_HOST_REQ_N_TOKENS] = n_tokens;
    hdr[GGML_MOE_HOST_REQ_N_USED]   = n_used;
    __threadfence_system();
    moe_host_store_release(hdr + GGML_MOE_HOST_REQ_SEQ, state[1]);
}

// the stream that copies the inputs of batch requests to the host while the device computes its experts
struct moe_host_side {
    cudaStream_t stream = nullptr;
    cudaEvent_t  ready  = nullptr;
};

static moe_host_side & moe_host_side_get(int device) {
    static moe_host_side sides[GGML_CUDA_MAX_DEVICES];
    moe_host_side & s = sides[device];
    if (s.stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.ready, cudaEventDisableTiming));
    }
    return s;
}

static void ggml_cuda_op_moe_host_post_batch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ids     = dst->src[0];
    const ggml_tensor * weights = dst->src[1];
    const ggml_tensor * x       = dst->src[2];
    const ggml_tensor * state   = dst->src[3];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(x->nb[0] == sizeof(float));

    void * mb;
    memcpy(&mb, dst->op_params, sizeof(mb));
    const int32_t layer     = ggml_get_op_params_i32(dst, 2);
    const int32_t off_ids   = ggml_get_op_params_i32(dst, 4);
    const int32_t off_slots = ggml_get_op_params_i32(dst, 5);
    const int32_t off_w     = ggml_get_op_params_i32(dst, 6);
    const int32_t off_x     = ggml_get_op_params_i32(dst, 7);
    const int32_t off_table = ggml_get_op_params_i32(dst, 8);
    const int32_t n_expert  = ggml_get_op_params_i32(dst, 9);

    const int n_used   = (int) ids->ne[0];
    const int n_tokens = (int) ids->ne[1];
    const int n_embd   = (int) x->ne[0];

    cudaStream_t stream = ctx.stream();
    const int n_blocks = std::min(256, (n_used*n_tokens + 255)/256);
    k_moe_host_post_batch<<<n_blocks, 256, n_expert*sizeof(int32_t), stream>>>(
        (const char *) ids->data,     ids->nb[0],     ids->nb[1],
        (const char *) weights->data, weights->nb[0], weights->nb[1],
        (int32_t *) state->data, (int32_t *) dst->data,
        (char *) mb, off_ids, off_slots, off_w, off_table,
        n_expert, n_used, n_tokens, layer);
    CUDA_CHECK(cudaGetLastError());

    // x goes to the host on the side stream, then the request is published there; the wait keeps x alive until then
    moe_host_side & side = moe_host_side_get(ctx.device);
    CUDA_CHECK(cudaEventRecord(side.ready, stream));
    CUDA_CHECK(cudaStreamWaitEvent(side.stream, side.ready, 0));
    CUDA_CHECK(cudaMemcpy2DAsync((char *) mb + off_x, n_embd*sizeof(float), x->data, x->nb[1], n_embd*sizeof(float), n_tokens,
        cudaMemcpyDeviceToHost, side.stream));
    k_moe_host_signal<<<1, 1, 0, side.stream>>>((const int32_t *) state->data, (char *) mb, layer, n_tokens, n_used);
    CUDA_CHECK(cudaGetLastError());
}

// batch mode: every block waits for the answer, then the blocks add y from the mailbox
static __global__ void k_moe_host_wait_batch(
        const char * __restrict__ a, const int64_t a_nb1, float * __restrict__ dst, const int32_t * __restrict__ state,
        char * __restrict__ mb, const int off_yh, const int off_yflag, const int n_embd, const int n_tokens) {
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
            __nanosleep(256);
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

    // a block per token at a time: its flag read once, then the row the host flags, in half precision, 8 values a load
    __shared__ int32_t flag;
    const int n8 = n_embd/8;
    const int4    * yh    = (const int4 *) (mb + off_yh);
    const int32_t * yflag = (const int32_t *) (mb + off_yflag);
    for (int t = blockIdx.x; t < n_tokens; t += gridDim.x) {
        if (threadIdx.x == 0) {
            flag = has_y ? __ldcv(yflag + t) : 0;
        }
        __syncthreads();
        const bool from_host = flag != 0;
        const float4 * ar = (const float4 *) (a + t*a_nb1);
        float4       * dr = (float4 *) (dst + (int64_t) t*n_embd);
        for (int j = threadIdx.x; j < n8; j += blockDim.x) {
            float4 v0 = ar[2*j + 0];
            float4 v1 = ar[2*j + 1];
            if (from_host) {
                const int4 hv = __ldcv(yh + (int64_t) t*n8 + j);
                const half2 * h = (const half2 *) &hv;
                const float2 f0 = __half22float2(h[0]), f1 = __half22float2(h[1]), f2 = __half22float2(h[2]), f3 = __half22float2(h[3]);
                v0.x += f0.x; v0.y += f0.y; v0.z += f1.x; v0.w += f1.y;
                v1.x += f2.x; v1.y += f2.y; v1.z += f3.x; v1.w += f3.y;
            }
            dr[2*j + 0] = v0;
            dr[2*j + 1] = v1;
        }
        __syncthreads();
    }
}

static void ggml_cuda_op_moe_host_wait_batch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a     = dst->src[0];
    const ggml_tensor * state = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(a->nb[0] == sizeof(float) && a->ne[0] % 4 == 0 && a->nb[1] % 16 == 0 && (uintptr_t) a->data % 16 == 0);

    void * mb;
    memcpy(&mb, dst->op_params, sizeof(mb));
    const int32_t off_yh    = ggml_get_op_params_i32(dst, 4);
    const int32_t off_yflag = ggml_get_op_params_i32(dst, 5);
    GGML_ASSERT(off_yh % 16 == 0 && a->ne[0] % 8 == 0);

    const int n_blocks = (int) std::min<int64_t>(4*ggml_cuda_info().devices[ctx.device].nsm, a->ne[1]);
    k_moe_host_wait_batch<<<n_blocks, 256, 0, ctx.stream()>>>(
        (const char *) a->data, a->nb[1], (float *) dst->data, (const int32_t *) state->data,
        (char *) mb, off_yh, off_yflag, (int) a->ne[0], (int) a->ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_moe_host_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst, 10) != 0) {
        ggml_cuda_op_moe_host_post_batch(ctx, dst);
        return;
    }

    const ggml_tensor * ids     = dst->src[0];
    const ggml_tensor * weights = dst->src[1];
    const ggml_tensor * x       = dst->src[2];
    const ggml_tensor * state   = dst->src[3];
    const ggml_tensor * pad     = dst->src[4];

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
        pad != nullptr ? (const int32_t *) pad->data : nullptr, (int32_t *) state->data, (int32_t *) dst->data,
        (char *) mb, off_ids, off_slots, off_w, off_x, off_table,
        n_expert, (int) ids->ne[0], (int) ids->ne[1], (int) x->ne[0], layer, zero_base);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_moe_host_wait(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst, 3) != 0) {
        ggml_cuda_op_moe_host_wait_batch(ctx, dst);
        return;
    }

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
