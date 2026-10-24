# Prompt processing

With every expert in host memory, a large ubatch is bound by the uploads: in the large phase (see
[moe-offload.md](moe-offload.md#cache-phases)) a ubatch uploads every expert the cache's core lacks, about 41.6 GB in
2.6 to 3 s over PCIe 4.0 x8, whatever its size. Long ubatches therefore move the least per token: the measured setup
runs 8192-token ubatches, the size Strata uses too. On top of that come the GPU compute (about 0.5 ms per token for this
model on a 3090), the drafter's catch-up on the second GPU, and the host's share of the experts.

From the stock path's 320 tok/s (ubatch 1024) to 1304–1361 / 1710–1756 / 1825–1848 tok/s at 11.6K / 34.8K / 92.8K
tokens, the steps were: the 3090 alone for the target (the drafter's QSA layer on the second GPU had a compute buffer
growing like the 3090's), the key pool reserved per stream ([attention.md](attention.md#compute-buffers)), cache phases,
the expert prefetch, 8192-token ubatches, the kernels and server changes below, and the host's share of large batches.

## MMQ for MoE at large batches

Commit `e2ff90695` (`ggml/src/ggml-cuda/mmq.cu`, `mmid.cu`):

- **Active tiles only** (`GGML_CUDA_MMQ_TILE_MAP`, default on): MMQ for `MUL_MAT_ID` launched a grid sized as if one
  expert took every token, about 655K blocks for an 8192-token ubatch of which about 97 % had no work. A tile map built
  on the device lists the active tiles and the grid covers only those.
- **Stable counting sort of the ids** (`GGML_CUDA_MMID_SORT`, default on): the ids of many tokens are grouped by expert
  with a stable counting sort instead of a scan per expert.

## Prompt chunks in the server

`tools/server/server-context.cpp`:

- **Balanced chunks** (commit `acab26f2b`, `LLAMA_SERVER_BALANCED_CHUNKS`, default on): the server splits a long prompt
  into chunks of equal size up to its last checkpoint, instead of chunks of `n_batch` and a short last one, since the
  last chunk uploads as much as a full one.
- **Checkpoint tail** (commit `f222e5947`, `LLAMA_SERVER_CKPT_TAIL=<n>`): the tokens after the checkpoint before the end
  of a prompt run as a batch of their own, `4 + n_ubatch` of them; with this the tail is `4 + n`, so a context whose
  ubatches are large does not process a long tail right after a large chunk (32 in the measured setup).
- Prompts of up to 512 tokens (`LLAMA_MOE_PHASE_TOKENS`) stay in the small phase, in 128-token host-mode batches: a large
  phase costs its full upload whatever the prompt, and 256-token to 1024-token batches did not beat it.

## The n-gram embedding gather

`qwen4exp` reads 16 hashed rows per token from its 26.8 GiB per-layer embedding table, a file mapping far larger than
what the page cache keeps next to the experts:

- `LLAMA_PLE_RAW=1` (commit `65dc51115`): the rows are gathered on the host as stored (quantized) into a graph input and
  converted on the device; gathered as F32 rows on the CPU backend, a large ubatch's rows would fill the host compute
  buffer and the link to the device.
- The gather of a large ubatch runs on several threads (`LLAMA_PLE_THREADS`, default 16) so that the reads of the pages
  not in memory are in flight together, and the table is advised `MADV_RANDOM`: the kernel's read-around pulled about
  128 KiB per 90-byte row (commit `65dc51115`).
- While decoding, the rows of drafted tokens are read ahead (see [speculative.md](speculative.md#n-gram-embedding-rows-read-ahead)).

## The drafter's catch-up

Each prompt chunk also goes through the MTP layer on the second GPU so that its keys are there for drafting. At
`--spec-draft-n-ubatch 64` that took 716 ms per 8192-token chunk (128 graphs); 512 is 6 % faster on a 92.8K-token
prompt, and with `LLAMA_SPEC_ASYNC_PROCESS=1` the catch-up of a chunk overlaps the target's next chunk (see
[speculative.md](speculative.md#drafter-catch-up)).

## Profiling switches

- `LLAMA_PROFILE_UBATCH=1`: the graph inputs of a large ubatch that take more than a millisecond to set; `=2`: every
  ubatch and decode call timed, plus the k-pool layout rebuilds; `=3`: the time of each graph input averaged over small
  ubatches.
- `GGML_SCHED_PROFILE=1`: host time per backend in the input copies and in the compute calls of the splits.

## A bug worth remembering

A first prompt below `LLAMA_MOE_PREFETCH_MIN` let the prefetch learn its order; the next small graph then allocated the
staging slots in the small phase, the allocation failed, and the prefetch stayed off for the rest of the session
(prompts at about 1180 tok/s instead of 1700). The prefetch now learns from large batches only and allocates its slots
in the large phase.
