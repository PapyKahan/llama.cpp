#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    ggml_cuda_syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}

// Many tokens: a stable counting sort, the same layout as mm_ids_helper. A warp counts the entries of a segment of
// MMID_SEG tokens per expert, one block turns the counts into the expert bounds and the first position of each
// segment's entries of each expert, and each warp then writes its entries in token order.
static constexpr int MMID_SEG   = 32; // tokens per segment, one warp each
static constexpr int MMID_WARPS = 8;  // warps per block

static __global__ void mm_ids_sort_count(
        const int32_t * __restrict__ ids, int32_t * __restrict__ counts, const int n_experts, const int n_tokens,
        const int n_expert_used, const int si1) {
    extern __shared__ int mm_ids_sort_count_data[];
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    int * wc = mm_ids_sort_count_data + warp*n_experts;

    for (int e = lane; e < n_experts; e += WARP_SIZE) {
        wc[e] = 0;
    }
    __syncwarp();

    const int seg = blockIdx.x*MMID_WARPS + warp;
    const int it0 = seg*MMID_SEG;
    if (it0 >= n_tokens) {
        return;
    }
    const int n_entries = min(MMID_SEG, n_tokens - it0)*n_expert_used;
    for (int k = lane; k < n_entries; k += WARP_SIZE) {
        const int e = ids[(it0 + k/n_expert_used)*si1 + k % n_expert_used];
        if (e >= 0 && e < n_experts) {
            atomicAdd(&wc[e], 1);
        }
    }
    __syncwarp();

    for (int e = lane; e < n_experts; e += WARP_SIZE) {
        counts[(size_t) seg*n_experts + e] = wc[e];
    }
}

// one thread per expert: the counts of each segment become the position of its first entry
static __global__ void mm_ids_sort_scan(
        int32_t * __restrict__ counts, int32_t * __restrict__ expert_bounds, const int n_experts, const int n_seg) {
    __shared__ int warp_sums[32];

    const int e    = threadIdx.x;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    int tot = 0;
    if (e < n_experts) {
        for (int s = 0; s < n_seg; ++s) {
            tot += counts[(size_t) s*n_experts + e];
        }
    }

    int incl = tot;
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        const int v = __shfl_up_sync(0xFFFFFFFF, incl, offset, WARP_SIZE);
        if (lane >= offset) {
            incl += v;
        }
    }
    if (lane == WARP_SIZE - 1) {
        warp_sums[warp] = incl;
    }
    __syncthreads();
    if (warp == 0) {
        const int w = lane < (int) (blockDim.x/WARP_SIZE) ? warp_sums[lane] : 0;
        int wincl = w;
#pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
            const int v = __shfl_up_sync(0xFFFFFFFF, wincl, offset, WARP_SIZE);
            if (lane >= offset) {
                wincl += v;
            }
        }
        warp_sums[lane] = wincl - w;
    }
    __syncthreads();

    if (e >= n_experts) {
        return;
    }
    const int base = warp_sums[warp] + incl - tot;
    expert_bounds[e] = base;
    if (e == n_experts - 1) {
        expert_bounds[n_experts] = base + tot;
    }
    int run = base;
    for (int s = 0; s < n_seg; ++s) {
        const int c = counts[(size_t) s*n_experts + e];
        counts[(size_t) s*n_experts + e] = run;
        run += c;
    }
}

static __global__ void mm_ids_sort_scatter(
        const int32_t * __restrict__ ids, const int32_t * __restrict__ offsets, int32_t * __restrict__ ids_src1,
        int32_t * __restrict__ ids_dst, const int n_experts, const int n_tokens, const int n_expert_used,
        const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    extern __shared__ int mm_ids_sort_scatter_data[];
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;

    const int seg = blockIdx.x*MMID_WARPS + warp;
    const int it0 = seg*MMID_SEG;
    if (it0 >= n_tokens) {
        return;
    }
    int * run = mm_ids_sort_scatter_data + warp*n_experts;
    for (int e = lane; e < n_experts; e += WARP_SIZE) {
        run[e] = offsets[(size_t) seg*n_experts + e];
    }
    __syncwarp();

    const int n_entries = min(MMID_SEG, n_tokens - it0)*n_expert_used;
    for (int k0 = 0; k0 < n_entries; k0 += WARP_SIZE) {
        const int k = k0 + lane;
        int it  = 0;
        int iex = 0;
        int e   = -1;
        if (k < n_entries) {
            it  = it0 + k/n_expert_used;
            iex = k % n_expert_used;
            e   = ids[it*si1 + iex];
            if (e < 0 || e >= n_experts) {
                e = -1;
            }
        }
        // the entries of one expert in this step take consecutive positions in lane order
        const unsigned int peers = __match_any_sync(0xFFFFFFFF, e);
        const int pos = e >= 0 ? run[e] + __popc(peers & ((1u << lane) - 1)) : 0;
        __syncwarp();
        if (e >= 0 && (peers >> lane) == 1u) {
            run[e] += __popc(peers);
        }
        __syncwarp();
        if (e >= 0) {
            ids_dst[pos] = it*n_expert_used + iex;
            if (write_inverse) {
                ids_src1[it*n_expert_used + iex] = pos;
            } else {
                ids_src1[pos] = it*sis1 + iex % nchannels_y;
            }
        }
    }
}

size_t ggml_cuda_mm_ids_sort_tmp_size(const int n_experts, const int n_tokens) {
    if (n_tokens < 512 || n_experts > 1024) {
        return 0;
    }
    const int n_seg = (n_tokens + MMID_SEG - 1)/MMID_SEG;
    return (size_t) n_seg*n_experts;
}

void ggml_cuda_launch_mm_ids_sort(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        int32_t * __restrict__ tmp, const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(ggml_cuda_mm_ids_sort_tmp_size(n_experts, n_tokens) > 0);
    const int    n_seg    = (n_tokens + MMID_SEG - 1)/MMID_SEG;
    const int    n_blocks = (n_seg + MMID_WARPS - 1)/MMID_WARPS;
    const dim3   block_dims(WARP_SIZE, MMID_WARPS, 1);
    const size_t nbytes_shared = (size_t) MMID_WARPS*n_experts*sizeof(int);

    mm_ids_sort_count<<<n_blocks, block_dims, nbytes_shared, stream>>>(ids, tmp, n_experts, n_tokens, n_expert_used, si1);
    mm_ids_sort_scan<<<1, GGML_PAD(n_experts, WARP_SIZE), 0, stream>>>(tmp, expert_bounds, n_experts, n_seg);
    mm_ids_sort_scatter<<<n_blocks, block_dims, nbytes_shared, stream>>>
        (ids, tmp, ids_src1, ids_dst, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
}
