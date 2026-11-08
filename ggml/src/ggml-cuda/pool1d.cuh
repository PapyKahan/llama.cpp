#include "common.cuh"

#define CUDA_POOL1D_BLOCK_SIZE 256

void ggml_cuda_op_pool1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// fused abs -> pool1d max over whole rows -> scale -> log -> scale -> ceil -> clamp -> scale -> exp
void ggml_cuda_op_amax_pow2(ggml_backend_cuda_context & ctx, const ggml_tensor * abs, const ggml_tensor * scale1,
        const ggml_tensor * scale2, const ggml_tensor * clamp, const ggml_tensor * scale3, ggml_tensor * dst);
