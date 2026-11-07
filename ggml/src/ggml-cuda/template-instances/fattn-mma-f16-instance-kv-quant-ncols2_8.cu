// 256/256 with more than 4 query heads per K/V head (Ornith-1.5's full attention) and the K/V cache in q5_1 or q8_0,
// dequantized by the tile loader. Written by hand: generate_cu_files.py does not know the type parameter.

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_F16_CASE_KV(256, 256, 1, 8, GGML_TYPE_Q5_1);
DECL_FATTN_MMA_F16_CASE_KV(256, 256, 2, 8, GGML_TYPE_Q5_1);
DECL_FATTN_MMA_F16_CASE_KV(256, 256, 4, 8, GGML_TYPE_Q5_1);
DECL_FATTN_MMA_F16_CASE_KV(256, 256, 8, 8, GGML_TYPE_Q5_1);
DECL_FATTN_MMA_F16_CASE_KV(256, 256, 1, 8, GGML_TYPE_Q8_0);
DECL_FATTN_MMA_F16_CASE_KV(256, 256, 2, 8, GGML_TYPE_Q8_0);
DECL_FATTN_MMA_F16_CASE_KV(256, 256, 4, 8, GGML_TYPE_Q8_0);
DECL_FATTN_MMA_F16_CASE_KV(256, 256, 8, 8, GGML_TYPE_Q8_0);
