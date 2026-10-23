#include "common.cuh"

// multi-token GEMV for weights with few long rows (F32/F16/BF16): whether it applies to this matrix multiplication
bool ggml_cuda_should_use_mmvf_rows(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_mul_mat_vec_f_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
