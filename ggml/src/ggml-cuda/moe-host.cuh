#include "common.cuh"

void ggml_cuda_op_moe_host_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_moe_host_wait(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
