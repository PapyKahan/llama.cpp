# Reference

Every switch the branch adds, with its default and the value of the measured setup (see [README.md](README.md#running-it)).
Most are environment variables because they started as experiments; [upstreaming.md](upstreaming.md) lists which should
become parameters.

## Expert offload

`src/llama-moe-cache.cpp`, `src/llama-moe-host.cpp`, `src/llama-context.cpp`. See [moe-offload.md](moe-offload.md).

| Variable | Default | Measured | Meaning |
|---|---|---|---|
| `LLAMA_MOE_HOST` | off | 1 | host mode: the host computes the misses of small batches, the forward pass posts to it |
| `LLAMA_MOE_HOST_MAX_TOKENS` | 32 | 128 | largest batch run in host mode |
| `LLAMA_MOE_HOST_THREADS` | hardware threads / 2 − 1 | (11) | threads computing the misses, the service thread included |
| `LLAMA_MOE_HOST_POLICY` | service | | `graph`: the earlier per-graph planner on the main thread |
| `LLAMA_MOE_HOST_SWAPS` | 96 | | service policy: replacements planned per graph |
| `LLAMA_MOE_HOST_FILLS` | 192 | | service policy: admissions into free slots per graph |
| `LLAMA_MOE_HOST_PER_REQUEST` | 2 | | service policy: swaps started per host request |
| `LLAMA_MOE_HOST_DECAY` | 0.9 (graph policy 0.97) | | decay of the selection counts per plan |
| `LLAMA_MOE_HOST_HYST` | 1.0 (graph policy 1.25) | | a victim is selected this many times less than its successor |
| `LLAMA_MOE_HOST_PERIOD` | 4 | | graph policy: graphs between two plans |
| `LLAMA_MOE_HOST_MAX_SWAPS` / `_MAX_FILLS` | 32 / 1024 | | graph policy: replacements / admissions per plan |
| `LLAMA_MOE_HOST_ROW8` | 2 (3 for IQ4_XS, IQ4_NL, Q2_0) | | entries from which an expert's weight rows are unpacked once; 0 = never |
| `LLAMA_MOE_HOST_STATS` | off | | log the miss rate every n host requests |
| `LLAMA_MOE_HOST_TRACE` | off | | file recording the selections of every request |
| `LLAMA_MOE_CACHE_TAIL_SKIP` | 0 | 7 | skip a missing expert that every token ranks this far or further back (0-based); approximation |
| `LLAMA_MOE_CACHE_TAIL_SKIP_W` | off | | also require its routing weight below this share of the token's total |
| `LLAMA_MOE_CACHE_PREFILL_MIB` | off | 8200 | cache phases: the core kept while large batches run, in MiB |
| `LLAMA_MOE_PHASE_TOKENS` | 256 | 512 | batches of more tokens run in the large phase |
| `LLAMA_MOE_PHASE_UBATCH` | 32 | 128 | ubatch of the small phase |
| `LLAMA_MOE_PREFETCH` | 0 | 3 | expert prefetch for large batches: staging slots |
| `LLAMA_MOE_PREFETCH_ROUTED` | 1 | 0 | stage only the experts the routing selects; 0 stages whole tensors |
| `LLAMA_MOE_PREFETCH_MIN` | 1024 | | smallest batch that uses the prefetch |
| `LLAMA_MOE_BATCH_CPU` | balanced | | share of the experts the core lacks that the host computes in large batches; 0 = off |
| `LLAMA_MOE_BATCH_CPU_MAX` | 1 | | cap of that share |
| `LLAMA_MOE_CACHE_PROFILE` | off | file | selection counts of past sessions: start filled, add this session's at exit |
| `LLAMA_MOE_CACHE_EXCLUSIVE` | off | 8200 | exclusive core: up to this many MiB of the core keep no host copy |
| `LLAMA_MOE_CACHE_EXCLUSIVE_SPARES` | 16 | | spare host slots per group for the exclusive core's downloads |
| `LLAMA_MOE_CACHE_ASYNC` | off | | without host mode: admit the uploaded misses in the background |
| `LLAMA_MOE_CACHE_ASYNC_MAX` | 96 | | experts in flight for it |
| `LLAMA_MOE_CACHE_TRACE` | off | | routing trace for `scripts/moe-cache-tune.py` (from PR #30966) |
| `LLAMA_HOST_COMPUTE_PAGEABLE` | off | 1 | CPU compute buffers in pageable memory, for contexts that re-reserve often |

## CUDA backend

`ggml/src/ggml-cuda/`.

| Variable | Default | Measured | Meaning |
|---|---|---|---|
| `GGML_CUDA_HOST_MEMFD` | off | 1 | page-locked host buffers as registered memfd files whose pages can be given back (Linux) |
| `GGML_CUDA_HOST_MEMFD_GUARD` | off | 1 | given-back pages made inaccessible: a stray access faults |
| `GGML_CUDA_HOST_BUFFER_CACHE` | off | | recycle freed page-locked buffers of 64 MiB or more |
| `GGML_CUDA_VMM_CHUNK_MIB` | 32 | | granularity of the VMM buffers of the cache banks |
| `GGML_CUDA_MMVF_ROWS` | off | 1 | multi-token GEMV for few long F32/F16/BF16 rows |
| `GGML_CUDA_MMVF_FEW_ROWS` | off | 1 | MMVF instead of MMF for BF16 matrices of up to 1024 rows |
| `GGML_CUDA_MMQ_TILE_MAP` | on | | MMQ for `MUL_MAT_ID` launches the active tiles only |
| `GGML_CUDA_MMID_SORT` | on | | ids of many tokens grouped by a stable counting sort |
| `GGML_CUDA_FA_COMPACT` | on | | sparse flash attention gathers the listed cells into compact buffers |
| `GGML_CUDA_FA_SPARSE_MIN_KV` | 0 (worst-case bound) | 8192 | cells from which query tiles gather |
| `GGML_CUDA_LID_MMA` | on | | lightning indexer for four heads on tensor cores |

## Speculative decoding and server

`common/speculative.cpp`, `src/llama-context.cpp`, `tools/server/server-context.cpp`. See [speculative.md](speculative.md)
and [prefill.md](prefill.md).

| Variable | Default | Measured | Meaning |
|---|---|---|---|
| `LLAMA_SPEC_P_CUM` | off | 1 | `--spec-draft-p-min` bounds the product of the draft's probabilities |
| `LLAMA_SPEC_P_TRACE` | off | | file: one line per verified draft (size, accepted, each token's probability and id) |
| `LLAMA_SPEC_PAD` | off | | pad verifications to a constant size (MTP drafter only) |
| `LLAMA_SPEC_ASYNC_PROCESS` | off | 1 | the drafter's catch-up of a prompt chunk runs in the background |
| `LLAMA_GRAPH_SLOTS` | 0 | 6 | keep the graphs of small decoding ubatches in this many slots |
| `LLAMA_GRAPH_SLOT_TOKENS` | 16 | | largest ubatch kept in a slot |
| `LLAMA_SERVER_BALANCED_CHUNKS` | on | | split a long prompt into equal chunks |
| `LLAMA_SERVER_CKPT_TAIL` | `n_ubatch` | 32 | the batch after the last checkpoint is 4 + this many tokens |

## qwen4exp

`src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.cpp`. See [attention.md](attention.md).

| Variable | Default | Measured | Meaning |
|---|---|---|---|
| `LLAMA_QSA_CHUNK` | 2048 | 512 | query tokens per QSA chunk, 0 disables the split (from PR #30815) |
| `LLAMA_QSA_KV_F16_ONCE` | on | | a chunked ubatch converts a quantized K/V to F16 once per layer |
| `LLAMA_PLE_RAW` | off | 1 | n-gram embedding rows gathered as stored on the host, converted on the device |
| `LLAMA_PLE_THREADS` | 16 | | threads gathering the rows of a large ubatch |
| `LLAMA_PLE_PREFETCH` | off | 1 | rows of drafted tokens read ahead by a thread |
| `LLAMA_KPOOL_CHECK` | off | | compare the incremental k-pool layout and tables with a rebuild |

## Profiling

| Variable | Meaning |
|---|---|
| `LLAMA_PROFILE_UBATCH` | 1: slow inputs of large ubatches; 2: every ubatch and decode call, k-pool rebuilds; 3: input times averaged over small ubatches |
| `GGML_SCHED_PROFILE` | host time per backend in the scheduler's input copies and compute calls |

## Command-line options

| Option | Default | From | |
|---|---|---|---|
| `--moe-cache-cpu-share F` | 0 | PR #30966 | share of the cache misses computed on the CPU (also gives host mode its zero slots) |
| `--spec-draft-vocab`, `--no-spec-draft-vocab` | off | PR #30380 | reduced draft LM head: the lowest 32768 ids and the end-of-generation ids, plus dynamic rows |
| `--spec-draft-vocab-dynamic-rows N` | 16384 | PR #30380 | other ids a low-rank projection ranks best per draft token; 0 = the lowest ids only |
| `--spec-draft-vocab-dynamic-rank N` | 512 | PR #30380 | rank of that projection |

## New ggml ops and entry points

| Name | Where | |
|---|---|---|
| `GGML_OP_MOE_HOST_POST`, `ggml_moe_host_post` | ggml, CUDA | post a MoE layer's selection and input to a host mailbox |
| `ggml_moe_host_post_set_pad` | ggml | mark the pad tokens of a padded verification |
| `ggml_moe_host_post_batch` | ggml, CUDA | post a whole large batch, the host's experts marked -1 |
| `GGML_OP_MOE_HOST_WAIT`, `ggml_moe_host_wait`, `ggml_moe_host_wait_batch` | ggml, CUDA | wait for the host's answer and add it |
| `ggml_mul_mat_id_set_negative_ids` | ggml, CPU, CUDA | a `MUL_MAT_ID` whose negative ids leave zero rows |
| `ggml_lightning_indexer_ext` | ggml, CPU, CUDA | the lightning indexer reading its keys through row indices |
| `ggml_backend_sched_new_shared`, `ggml_backend_sched_shared_valid` | ggml-backend | a scheduler allocating from another one's compute buffers |
| `ggml_cpu_row8_supported`, `_unpack`, `_dot` | ggml-cpu | unpack a quantized weight row once, dot it with up to four activation rows |
| `ggml_backend_host_mapped_alloc` / `_free` | CUDA proc address | mapped host memory for the mailboxes |
| `ggml_backend_vmm_buffer_alloc` / `_commit` | CUDA proc address | device buffers whose physical memory grows and shrinks from the end |
| `ggml_backend_host_release`, `ggml_backend_host_releasable` | CUDA proc address | give back pages of memfd host buffers |
| `ggml_backend_synchronize_sleep` | CUDA proc address | wait for a stream without spinning |
| `llama_prefetch_token` | `src/llama-ext.h` | read ahead the file-backed rows a token will need |
| `common_context_set_wait`, `common_context_wait`, `common_context_defer` | common | serialize access to a context a background decode uses |
| `llama_batch_ext_set_pad`, `common_batch::token::pad`, `llama_ubatch::pad` | llama, common | pad entries of a batch |

New files: `src/llama-moe-host.{h,cpp}`, `ggml/src/ggml-cuda/moe-host.cu`, `ggml/src/ggml-cuda/mmvf-rows.cu`,
`tests/test-draft-vocab.cpp`, and from PR #30966 `scripts/moe-cache-tune.py` and `docs/moe-cache.md`.

## Commits

Upstream base `f64a49819`. Merged pull requests: #30966 (`175c8336e`), #30815 (`f64d60f16`, includes #30640),
#30448 (`a95a90a0d`), #30380 (`21fda432c`). Local commits, oldest first:

| Commit | |
|---|---|
| `7ba73dd54` | QSA key pool reserved per KV stream |
| `012d9f7ef` | route tail skip in the MoE cache |
| `5682d1c2b` | merge fix: the CPU `MUL_MAT_ID` grouping skips negative ids (#30966 with `--moe-cache-cpu-share`) |
| `0e80359ad`, `925bd9858`, `814f55149`, `bc0bf148a`, `ddd0cfe81` | expert prefetch for large batches |
| `42207ffc9` | asynchronous admission into the MoE cache |
| `c44d385f4` | each QSA chunk's scratch freed before the next |
| `e1cd4e00f` | host mode, MMVF kernels for few long rows |
| `a29262c1e` | cache phases, service-thread cache policy, pageable CPU compute buffers |
| `e2ff90695` | MMQ tile map, counting sort of ids, compact sparse flash attention, CPY quantized to F16 |
| `65dc51115` | prefetch order from the graph, incremental k-pool layout, parallel raw n-gram gather |
| `acab26f2b` | balanced prompt chunks, catch-up in pieces, safe asynchronous catch-up |
| `0bfe8cb7a` | lightning indexer on tensor cores |
| `d5379bd75` | lightning indexer reads pooled keys in place |
| `8060a31c7` | cache profile, re-admission of lent slots |
| `2b37cca0f` | host kernels unpack a weight row once |
| `f222e5947` | server checkpoint tail |
| `d8b10a771`, `70d3d4ba7` | large batches shared with the host |
| `991413bff` | VMM chunk size, one sync before unmapping |
| `2fe4ae266` | padded verifications |
| `ce345f505`, `027048e39` | cumulative draft stop, draft trace |
| `b1351e725` | k-pool tables kept across ubatches |
| `77ad0af04` | backend sampler chain keeps the reservation |
| `8fa30b1ee` | background catch-up for prompt chunks only |
| `df4fbcbf9` | `LLAMA_PROFILE_UBATCH=3` |
| `b669e4293` | MTP layer cropped to its output rows after the attention |
| `a93fd2f96` | k-pool build state kept in the memory |
| `e51d84af8` | graph slots |
| `27adb1e7d` | n-gram rows of drafted tokens read ahead |
| `170656bca` | exclusive core, memfd host buffers |
