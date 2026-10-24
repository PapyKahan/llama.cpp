# QSA attention at long context

`qwen4exp`'s attention layers use QSA: the cache is cut into blocks (pools) of `compress_ratio` cells in sequence order,
a lightning indexer scores each block's pooled key against the query, and a token attends to the top blocks plus the
incomplete tail block (`src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.cpp`, the k-pool cache). At 262K
context with q5_1 K/V the attention cache is 2.3 GiB and the indexer's key cache 0.6 GiB on the 3090.

Before the work below, decoding slowed from 41 to 28.5 tok/s between 84K and 210K tokens of context, and the step
cost grew with the context almost entirely on the host side. After it, the step costs 32.5 / 33.5 / 35.0 ms at
2.3K / 84K / 210K tokens with three drafts, about Strata's at equal draft counts.

## Upstream pull requests in the branch

| PR | Author | State | |
|---|---|---|---|
| #30640 | am17an | merged upstream 2026-10-21 | a fused op for QSA scoring and the top-k mask |
| #30965 | sanmai | merged upstream 2026-10-23 | the QSA selection built without the `kq_mask` |
| #30815 | sanmai | open | the same plus query chunks for large ubatches (`LLAMA_QSA_CHUNK`, default 2048, 0 disables) |

The branch carries #30815's commits (merged before #30965 landed), which include #30640's op. Building the selection
without the `[n_kv, n_tokens]` mask freed about 800 MiB of compute buffer at a given ubatch.

## Compute buffers

- **Key pool per KV stream** (commit `7ba73dd54`, one line): the graph reserved the pools of a full cache per sequence;
  with `--parallel 3 --kv-unified` (three sequences, one stream) that was three times what the cache can hold. Reserved
  per KV stream, the 3090's compute buffer at a 4096-token ubatch went from 7.7 to 4.65 GiB, which is what lets
  8192-token ubatches fit at 262K context.
- **Chunk scratch freed** (commit `c44d385f4`): with #30815's chunking each chunk's attention scratch stayed allocated
  until the end of the layer, so chunking made the buffer larger; each chunk's scratch is now free before the next.
- **K/V converted once** (commit `65dc51115`, `LLAMA_QSA_KV_F16_ONCE`, default on): a ubatch in several chunks converts a
  quantized K/V to F16 once per layer for all of them, where each chunk's attention would convert the whole cache again.
  CPY gained a quantized-to-F16 path for contiguous tensors (commit `e2ff90695`).

## The k-pool layout and tables

- **Incremental layout** (commit `65dc51115`): after every rejected draft the layout of the pools was rebuilt from all
  cells. It now keeps the cells below an edit's stale position and appends the rest.
- **Tables kept across ubatches** (commit `b1351e725`): `set_input_kpool` rebuilt the pool tables of every sequence for
  every ubatch and sorted all the pooled key rows to find the cells of the padded new pools, about 0.5 ms per ubatch at
  210K tokens, five times per step (the target's verification, the drafter's catch-up and three draft steps). With a
  single stream the tables are kept in the layout (`kpool_layout::flat_tables`) and rebuilt from the first pool that
  changed, which `kpool_layout_update` records per sequence; the new pools come from the list `kpool_build_state` marks,
  and the padding cells are the first cells a bitmap of the pooled rows leaves free. The 210K step went from 41–43 to
  about 35 ms.
- **Build state in the memory** (commit `a93fd2f96`): the per-cell marks of that state were allocated and cleared for
  every decode, because a memory context lives for one decode; the memory now keeps them, with a generation counter.

`LLAMA_KPOOL_CHECK=1` compares the incremental layout and the kept tables with a full rebuild at every update.

## The lightning indexer

- **Tensor cores for four heads** (commit `0bfe8cb7a`, `ggml/src/ggml-cuda/lightning-indexer.cu`): a block scores 64
  tokens against 128 keys with m16n8k16 MMAs, the keys staged once as F16, the queries of one head at a time rounded to
  F16, the products accumulated in F32, rectified and weighted into the score. Batches of 64 tokens or more on Ampere
  and newer take it (`GGML_CUDA_LID_MMA=0` keeps the FMA tile kernel).
- **Pooled keys read in place** (commit `d5379bd75`): `ggml_lightning_indexer_ext` takes I32 row indices as `src[4]`, key
  `i` being row `rows[i]` of `k`. The CUDA kernels and the CPU reference address the keys through them, so the pooled
  keys are scored where the k-pool cache stores them, in the cache type, instead of being gathered to F32 per layer and
  step. Batches of two tokens or more use the tile kernel, which reads each key once for all of its tokens. With it the
  decode speed stopped falling with the context, and 84K-token prefill went from 1732 to 1884 tok/s.

## Sparse flash attention

Commit `e2ff90695` (`ggml/src/ggml-cuda/fattn.cu`, `fattn-common.cuh`):

- Over a quantized cache, the sparse path gathers only the listed K/V rows and mask entries into compact buffers when the
  lists hold fewer cells than the cache, instead of converting the whole cache (`GGML_CUDA_FA_COMPACT=0` keeps the
  conversion).
- Single-token steps take that path instead of the dense vector kernel.
- A long mask row is scanned for its listed cells by many blocks instead of one.
- `GGML_CUDA_FA_SPARSE_MIN_KV=<cells>` lowers the threshold from which query tiles gather instead of using the
  worst-case bound, as the queries of a tile mostly select the same cells (8192 in the measured setup).
