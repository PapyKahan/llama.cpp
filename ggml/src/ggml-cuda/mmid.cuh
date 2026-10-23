#pragma once

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);

// many tokens (tmp size above 0): a counting sort with the same result, tmp holding that many ints
size_t ggml_cuda_mm_ids_sort_tmp_size(int n_experts, int n_tokens);

void ggml_cuda_launch_mm_ids_sort(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds, int32_t * tmp,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);
