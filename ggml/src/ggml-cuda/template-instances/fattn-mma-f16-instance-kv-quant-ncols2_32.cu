// Mistral Small 4 (DKQ=320, DV=256, GQA 32) with the K/V cache in q5_1 or q8_0, dequantized by the tile loader.
// Written by hand: generate_cu_files.py does not know the type parameter.

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_F16_CASE_KV(320, 256, 1, 32, GGML_TYPE_Q5_1);
DECL_FATTN_MMA_F16_CASE_KV(320, 256, 2, 32, GGML_TYPE_Q5_1);
DECL_FATTN_MMA_F16_CASE_KV(320, 256, 1, 32, GGML_TYPE_Q8_0);
DECL_FATTN_MMA_F16_CASE_KV(320, 256, 2, 32, GGML_TYPE_Q8_0);
