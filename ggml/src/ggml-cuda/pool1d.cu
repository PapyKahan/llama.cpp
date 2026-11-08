#include "pool1d.cuh"

static __global__ void pool1d_nchw_kernel(
        const int iw, const int ow,
        const int kw, const int sw, const int pw,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= parallel_elements) {
        return;
    }

    const int nc = idx / ow;
    const int cur_ow = idx % ow;

    const float * i_ptr = src + nc * iw;
    float * o_ptr = dst + nc * ow;

    const int start = cur_ow * sw - pw;
    const int b = max(0, start);
    const int e = min(iw, start + kw);

    float res;
    switch (op) {
        case GGML_OP_POOL_AVG: res = 0.0f;     break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default: return;
    }

    int count = 0;
    for (int i = b; i < e; i++) {
#if __CUDA_ARCH__ >= 350
        float cur = __ldg(i_ptr + i);
#else
        float cur = i_ptr[i];
#endif
        switch (op) {
            case GGML_OP_POOL_AVG: res += cur;                break;
            case GGML_OP_POOL_MAX: res = max(res, cur);       break;
            default: break;
        }
        count++;
    }

    if (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : 0.0f;
    }

    o_ptr[cur_ow] = res;
}

// one output per row (the kernel covers the row, as a max or mean over the row): a block reduces each row
template<int block_size>
static __global__ void pool1d_row_kernel(const int iw, const float * src, float * dst, const enum ggml_op_pool op) {
    const int row = blockIdx.x;
    const float * i_ptr = src + (int64_t) row*iw;
    float res = op == GGML_OP_POOL_MAX ? -FLT_MAX : 0.0f;
    for (int i = threadIdx.x; i < iw; i += block_size) {
        const float cur = __ldg(i_ptr + i);
        res = op == GGML_OP_POOL_MAX ? max(res, cur) : res + cur;
    }
    __shared__ float part[block_size/WARP_SIZE];
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, res, offset, WARP_SIZE);
        res = op == GGML_OP_POOL_MAX ? max(res, other) : res + other;
    }
    if (threadIdx.x % WARP_SIZE == 0) {
        part[threadIdx.x/WARP_SIZE] = res;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        res = part[0];
        for (int w = 1; w < block_size/WARP_SIZE; ++w) {
            res = op == GGML_OP_POOL_MAX ? max(res, part[w]) : res + part[w];
        }
        dst[row] = op == GGML_OP_POOL_AVG ? (iw > 0 ? res/iw : 0.0f) : res;
    }
}

static void pool1d_nchw_kernel_f32_f32_cuda(
        const int iw, const int ow,
        const int kw, const int sw, const int pw,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op,
        cudaStream_t stream) {
    const int num_blocks = (parallel_elements + CUDA_POOL1D_BLOCK_SIZE - 1) / CUDA_POOL1D_BLOCK_SIZE;
    dim3 block_nums(num_blocks);
    pool1d_nchw_kernel<<<block_nums, CUDA_POOL1D_BLOCK_SIZE, 0, stream>>>(iw, ow, kw, sw, pw, parallel_elements, src, dst, op);
}

void ggml_cuda_op_pool1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int s0 = opts[2];
    const int p0 = opts[3];

    const int64_t IW = src0->ne[0];
    const int64_t OW = dst->ne[0];
    const int64_t nr = ggml_nrows(src0);

    if (OW == 1 && k0 >= IW && p0 == 0 && (op == GGML_OP_POOL_MAX || op == GGML_OP_POOL_AVG)) {
        pool1d_row_kernel<256><<<(unsigned) nr, 256, 0, stream>>>((int) IW, src0_d, dst_d, op);
        return;
    }

    const int parallel_elements = (int)(nr * OW);

    pool1d_nchw_kernel_f32_f32_cuda(IW, OW, k0, s0, p0, parallel_elements, src0_d, dst_d, op, stream);
}

// The power-of-two scale of each row from its max. |x| (llm_graph_context::build_amax_src1_f16), the chain
// abs -> pool1d max over the row -> scale -> log -> scale -> ceil -> clamp -> scale -> exp as one kernel:
// dst[row] = exp(s3*clamp(ceil(s2*log(s1*max|x|)), lo, hi)).
template<int block_size>
static __global__ void amax_pow2_row_kernel(const int ne0, const float * src, float * dst,
        const float s1, const float s2, const float lo, const float hi, const float s3) {
    const int row = blockIdx.x;
    const float * x = src + (int64_t) row*ne0;
    float amax = 0.0f;
    for (int i = threadIdx.x; i < ne0; i += block_size) {
        amax = fmaxf(amax, fabsf(__ldg(x + i)));
    }
    __shared__ float part[block_size/WARP_SIZE];
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }
    if (threadIdx.x % WARP_SIZE == 0) {
        part[threadIdx.x/WARP_SIZE] = amax;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        amax = part[0];
        for (int w = 1; w < block_size/WARP_SIZE; ++w) {
            amax = fmaxf(amax, part[w]);
        }
        dst[row] = expf(s3*fminf(fmaxf(ceilf(s2*logf(s1*amax)), lo), hi));
    }
}

// abs: GGML_UNARY_OP_ABS of the rows; the scale and clamp parameters come from the chain's nodes; dst: the exp node
void ggml_cuda_op_amax_pow2(ggml_backend_cuda_context & ctx, const ggml_tensor * abs, const ggml_tensor * scale1,
        const ggml_tensor * scale2, const ggml_tensor * clamp, const ggml_tensor * scale3, ggml_tensor * dst) {
    const ggml_tensor * src0 = abs->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(dst) == ggml_nrows(src0));
    const float s1 = ggml_get_op_params_f32(scale1, 0);
    const float s2 = ggml_get_op_params_f32(scale2, 0);
    const float lo = ggml_get_op_params_f32(clamp, 0);
    const float hi = ggml_get_op_params_f32(clamp, 1);
    const float s3 = ggml_get_op_params_f32(scale3, 0);
    amax_pow2_row_kernel<256><<<(unsigned) ggml_nrows(src0), 256, 0, ctx.stream()>>>(
        (int) src0->ne[0], (const float *) src0->data, (float *) dst->data, s1, s2, lo, hi, s3);
}
