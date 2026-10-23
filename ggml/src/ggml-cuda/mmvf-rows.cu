#include "mmvf-rows.cuh"

// A warp per row and up to 8 rows per block, so the rows of a block read the activations of the tokens once from L2.
// When the rows cannot fill the device, the long dimension is split over grid.y: each block writes its parts and the
// last block of a row block adds them in split order.

static constexpr int MMVF_ROWS_MAX_COLS = 8;

template <typename T>
static __device__ __forceinline__ void mmvf_rows_load8(const T * p, float * w);

template <>
__device__ __forceinline__ void mmvf_rows_load8<float>(const float * p, float * w) {
    const float4 a = *(const float4 *) p;
    const float4 b = *(const float4 *) (p + 4);
    w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w;
    w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
}

template <>
__device__ __forceinline__ void mmvf_rows_load8<half>(const half * p, float * w) {
    const int4 v = *(const int4 *) p;
    const half2 * h = (const half2 *) &v;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h[i]);
        w[2*i + 0] = f.x;
        w[2*i + 1] = f.y;
    }
}

template <>
__device__ __forceinline__ void mmvf_rows_load8<nv_bfloat16>(const nv_bfloat16 * p, float * w) {
    const int4 v = *(const int4 *) p;
    const uint32_t * u = (const uint32_t *) &v;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        w[2*i + 0] = __uint_as_float(u[i] << 16);
        w[2*i + 1] = __uint_as_float(u[i] & 0xFFFF0000u);
    }
}

template <int ncols>
static __device__ __forceinline__ void mmvf_rows_fma(float * sum, const float * w, const float * y, const int64_t stride_col_y, const int k) {
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const float4 a = __ldg((const float4 *) (y + j*stride_col_y + k));
        const float4 b = __ldg((const float4 *) (y + j*stride_col_y + k + 4));
        sum[j] += w[0]*a.x + w[1]*a.y + w[2]*a.z + w[3]*a.w + w[4]*b.x + w[5]*b.y + w[6]*b.z + w[7]*b.w;
    }
}

// grid: (row blocks, splits), block: (32, rows per block)
template <typename T, int ncols>
static __global__ void mul_mat_vec_f_rows(
        const T * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst, float * __restrict__ parts,
        unsigned int * __restrict__ counters, const int nrows, const int64_t stride_row, const int64_t stride_col_y,
        const int64_t stride_col_dst, const int k_per_split, const int ncols_x) {
    constexpr int UNROLL = 4;
    constexpr int STEP   = 8*WARP_SIZE;

    const int  row   = blockIdx.x*blockDim.y + threadIdx.y;
    const bool valid = row < nrows;
    const int  k0    = blockIdx.y*k_per_split;
    const int  k1    = min(k0 + k_per_split, ncols_x);

    float sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        sum[j] = 0.0f;
    }

    if (valid) {
        const T * xr = x + row*stride_row;
        int k = k0 + 8*threadIdx.x;
        // several weight loads in flight before the activations are read
        for (; k + (UNROLL - 1)*STEP < k1; k += UNROLL*STEP) {
            float w[UNROLL][8];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                mmvf_rows_load8<T>(xr + k + u*STEP, w[u]);
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                mmvf_rows_fma<ncols>(sum, w[u], y, stride_col_y, k + u*STEP);
            }
        }
        for (; k < k1; k += STEP) {
            float w[8];
            mmvf_rows_load8<T>(xr + k, w);
            mmvf_rows_fma<ncols>(sum, w, y, stride_col_y, k);
        }
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        sum[j] = warp_reduce_sum(sum[j]);
    }

    if (gridDim.y == 1) {
        if (valid && threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                dst[j*stride_col_dst + row] = sum[j];
            }
        }
        return;
    }

    if (valid && threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            parts[((int64_t) blockIdx.y*ncols + j)*nrows + row] = sum[j];
        }
    }
    __threadfence();
    __syncthreads();

    __shared__ bool last;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        last = atomicAdd(counters + blockIdx.x, 1u) == gridDim.y - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    if (valid && threadIdx.x < ncols) {
        const int j = threadIdx.x;
        float s = 0.0f;
        for (int p = 0; p < (int) gridDim.y; ++p) {
            s += __ldcg(parts + ((int64_t) p*ncols + j)*nrows + row);
        }
        dst[j*stride_col_dst + row] = s;
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        counters[blockIdx.x] = 0; // ready for the next launch
    }
}

bool ggml_cuda_should_use_mmvf_rows(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_CUDA_MMVF_ROWS") != nullptr;
    if (!enabled) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne11 = src1->ne[1];
    if (ne11 < 1 || ne11 > MMVF_ROWS_MAX_COLS || ne01 > 2048 || ne00 < 1024 || ne00 % 8 != 0) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    const size_t ts = ggml_type_size(src0->type);
    return src0->nb[0] == ts && src0->nb[1] % 16 == 0 && (uintptr_t) src0->data % 16 == 0 &&
           src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 && (uintptr_t) src1->data % 16 == 0 &&
           dst->nb[0] == sizeof(float);
}

// zeroed counters of the row blocks, per device: the last block of a row block resets its counter
static constexpr int MMVF_ROWS_MAX_COUNTERS = 4096;

static unsigned int * mmvf_rows_counters(int device) {
    static unsigned int * counters[GGML_CUDA_MAX_DEVICES] = {};
    if (counters[device] == nullptr) {
        CUDA_CHECK(cudaMalloc(&counters[device], MMVF_ROWS_MAX_COUNTERS*sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(counters[device], 0, MMVF_ROWS_MAX_COUNTERS*sizeof(unsigned int)));
    }
    return counters[device];
}

template <typename T>
static void mul_mat_vec_f_rows_launch(ggml_backend_cuda_context & ctx, const T * x, const float * y, float * dst,
        const int nrows, const int ncols, const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int ncols_x) {
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    const int rpb = nrows >= 8 ? 8 : nrows;
    const int nbx = (nrows + rpb - 1)/rpb;
    // enough blocks for two waves, each split at least 512 long and a multiple of 8
    int nsplit = std::max(1, std::min((2*nsm + nbx - 1)/nbx, ncols_x/512));
    if (nbx > MMVF_ROWS_MAX_COUNTERS) {
        nsplit = 1;
    }
    int k_per_split = GGML_PAD((ncols_x + nsplit - 1)/nsplit, 8);
    nsplit = (ncols_x + k_per_split - 1)/k_per_split;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> parts(ctx.pool());
    unsigned int * counters = nullptr;
    if (nsplit > 1) {
        parts.alloc((size_t) nsplit*ncols*nrows);
        counters = mmvf_rows_counters(ctx.device);
    }
    const dim3 grid(nbx, nsplit, 1);
    const dim3 block(WARP_SIZE, rpb, 1);
    switch (ncols) {
#define MMVF_ROWS_CASE(n) case n: mul_mat_vec_f_rows<T, n><<<grid, block, 0, stream>>>(x, y, dst, parts.ptr, counters, nrows, stride_row, stride_col_y, stride_col_dst, k_per_split, ncols_x); break;
        MMVF_ROWS_CASE(1) MMVF_ROWS_CASE(2) MMVF_ROWS_CASE(3) MMVF_ROWS_CASE(4)
        MMVF_ROWS_CASE(5) MMVF_ROWS_CASE(6) MMVF_ROWS_CASE(7) MMVF_ROWS_CASE(8)
#undef MMVF_ROWS_CASE
        default: GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_vec_f_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int     nrows  = (int) src0->ne[1];
    const int     ncols  = (int) src1->ne[1];
    const int     K      = (int) src0->ne[0];
    const int64_t s_row  = src0->nb[1]/ggml_type_size(src0->type);
    const int64_t s_coly = src1->nb[1]/sizeof(float);
    const int64_t s_cold = dst->nb[1]/sizeof(float);
    const float * y = (const float *) src1->data;
    float * d = (float *) dst->data;
    switch (src0->type) {
        case GGML_TYPE_F32:
            mul_mat_vec_f_rows_launch(ctx, (const float *) src0->data, y, d, nrows, ncols, s_row, s_coly, s_cold, K);
            break;
        case GGML_TYPE_F16:
            mul_mat_vec_f_rows_launch(ctx, (const half *) src0->data, y, d, nrows, ncols, s_row, s_coly, s_cold, K);
            break;
        case GGML_TYPE_BF16:
            mul_mat_vec_f_rows_launch(ctx, (const nv_bfloat16 *) src0->data, y, d, nrows, ncols, s_row, s_coly, s_cold, K);
            break;
        default:
            GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}
