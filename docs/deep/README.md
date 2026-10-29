# The `deep` branch: Qwen3.8-Flash-Next at 262K context on one 24 GB GPU

This branch (fork `PapyKahan/llama.cpp`, branch `deep`) runs Qwen3.8-Flash-Next (`qwen4exp`, 125B parameters, 48 MoE
layers of 512 experts with 10 selected per token, QSA sparse attention, an MTP draft layer and a per-layer n-gram
embedding table) with its full 262,144-token context on a desktop PC:

- an RTX 3090 (24 GB) holds the dense weights, the KV and recurrent caches, the compute buffers and a cache of the most
  selected experts;
- a smaller second GPU (RTX 3070 Ti, 8 GB) holds the MTP drafter and the vision projector;
- every other expert stays in host RAM, where the CPU computes it while the GPU computes the cached ones.

The goal set for the work: at least the speed of [Strata](https://github.com/Niko1221/Strata), an engine written for
this model, on the same machine, with a RAM footprint on par with it, at 262K context.

The branch is upstream `master` up to `39febd2ad` (v0.8.0 and three commits, merged in on 2026-10-28) plus four
upstream pull requests and about fifty commits of its own, named `local: ...`. This directory documents them, as the
basis for upstream pull requests once the work listed in [upstreaming.md](upstreaming.md) is done. Nothing here is
specific to the machine except the measurements; several parts are specific to `qwen4exp` and say so. The expert offload
and the drafter also run Ornith-1.5-122B, a `qwen35moe` model (see [measurements.md](measurements.md#ornith-15-122b)).

## Where it stands

All runs on the same machine (see [measurements.md](measurements.md) for the hardware, the harness and the spread),
2026-10-23 to 2026-10-24; `deep`'s prompts and follow-ups 2026-10-28, with the measured host share and the lent memory.

| | upstream + #30965 + #30966 | Strata 0.2.2 | `deep` |
|---|---|---|---|
| decode, short prompts: cold / warm / code (tok/s) | 61.0 / 61.3 / 55.2 | 88.9 / 87.1 / 72.7 | 108.7–114.0 / 105.7–110.4 / 99.5–104.4 |
| prefill 11.6K / 34.8K / 92.8K tokens (tok/s) | 320 / 313 / 298 | 1389 / 1616 / 1709 | 1531 / 1809 / 1838 |
| code task decode at 2.3K / 84K / 210K context (tok/s) | | 68.4 / 76.1 / 76.1 | 77.3–85.3 / 69.7–78.7 / 72.6–85.6 |
| code task prefill 84K / 210K (tok/s) | | 1708 / 1686 | 1870–1894 / 1743–1843 |
| follow-up turn of +100 / +300 / +1000 / +3000 tokens on an 11.6K conversation (s to first token) | | 0.72 / 0.96 / 3.14 / 3.34 | 1.17 / 1.78 / 2.38 / 2.58 |
| lowest MemAvailable while serving (GiB of 62.7) | | ≥ 13.8 | 13.0–13.3 |

Decode throughput follows the drafter's acceptance, which varies with the sampled text from run to run; the cost of a
step at a given number of drafts is the stable figure, and it is on par with Strata's at long context (28.5–29.4 ms per
step at 84K and 32.3 ms at 210K, against 27.6 and 32.9 ms in Strata's own log, at similar draft counts). Since the
measured host share and the lent memory, follow-up turns of 1000 tokens and more are faster than Strata's (2.38 / 2.58 s
for +1000 / +3000, see [measurements.md](measurements.md#measured-host-share-and-lent-memory)); shorter ones remain
slower.

The upstream column is the stock MoE cache path (`--cpu-moe --moe-cache-mib 14600 --moe-cache-cpu-share 0.7`, ubatch
1024) with the two pull requests it needed at the time. After the merge with upstream, the same benches give the same
figures (see [measurements.md](measurements.md#after-the-merge-with-upstream)).

## How a request runs

**Decoding and speculative verification** (batches of up to 128 tokens) run in *host mode*
([moe-offload.md](moe-offload.md#host-mode)): each MoE layer posts its selected experts, their weights and its input to a
mailbox in mapped host memory, the GPU computes the experts it caches while a host thread pool computes the others, and
the GPU adds the host's answer. The whole forward pass is one CUDA graph with no host synchronization. A service thread
on the host moves experts in and out of the cache between requests, from selection counts it gathers, and the cache
starts each session from a profile of past sessions.

**Long prompts** (chunks of more than 256 tokens) run in 8192-token ubatches in the *large phase*
([moe-offload.md](moe-offload.md#cache-phases)): the expert cache shrinks to its core and lends the rest of its memory to
the compute buffers and to staging slots, into which a worker uploads the next layers' experts while the current layer
computes; the memory it lends stays mapped, so a change of phase costs little. The host computes the least selected of
the experts the core lacks for the whole batch meanwhile, as many as it is measured to save time with. Prompts of up to
256 tokens stay in host mode in 128-token batches.

**The drafter** ([speculative.md](speculative.md)) is the model's MTP layer on the second GPU, with a trimmed output
vocabulary; it drafts up to three tokens and stops when the chance that the whole draft is accepted falls below 0.3. The
sampler chain runs on the GPU. The graphs of each verification size stay allocated side by side, so a varying window
rebuilds nothing.

**Long contexts** ([attention.md](attention.md)) keep a per-step cost that grows little with the context: the QSA
k-pool tables are updated incrementally, the lightning indexer reads the pooled keys in place and on tensor cores, and
sparse flash attention gathers only the cells it attends to.

**RAM** ([moe-offload.md](moe-offload.md#exclusive-core)): the experts the cache's core holds have no copy in RAM; an
evicted one is first downloaded into a spare host slot. Host memory then holds only the experts the core lacks.

## Contents

| File | |
|---|---|
| [moe-offload.md](moe-offload.md) | the expert offload: host mode, the service-thread cache policy, cache phases, expert prefetch, large batches shared with the host, the cache profile, the tail skip, host kernels, the exclusive core and memfd host buffers |
| [speculative.md](speculative.md) | the MTP drafter: trimmed draft vocabulary, the cumulative draft stop, graph slots, padded verifications, the asynchronous catch-up, backend sampling, the n-gram row prefetch |
| [attention.md](attention.md) | QSA at long context: k-pool tables, the lightning indexer, sparse flash attention, chunked QSA |
| [prefill.md](prefill.md) | prompt processing: MMQ for MoE at large batches, balanced prompt chunks, the checkpoint tail, the n-gram gather |
| [reference.md](reference.md) | every switch (environment and command line) with its default, the new ggml ops and backend entry points, the commits |
| [measurements.md](measurements.md) | hardware, model files, harness, method, results and the pitfalls of measuring this model |
| [upstreaming.md](upstreaming.md) | the work to do before any upstream pull request, a possible split for later, and what was tried and dropped |

## Running it

The configuration measured above, as a server command (the model is ISTA-DASLab's GSQ-RCO IQ3_S GGUF, the drafter
Unsloth's MTP head, both on Hugging Face):

```bash
LLAMA_MOE_HOST=1 LLAMA_MOE_CACHE_TAIL_SKIP=7 \
LLAMA_MOE_CACHE_PREFILL_MIB=8200 LLAMA_MOE_PHASE_TOKENS=256 LLAMA_MOE_PHASE_UBATCH=128 LLAMA_MOE_HOST_MAX_TOKENS=128 \
LLAMA_MOE_PREFETCH=3 LLAMA_MOE_PREFETCH_ROUTED=0 \
LLAMA_MOE_CACHE_PROFILE=$HOME/.cache/flash-next.moe-profile \
LLAMA_MOE_CACHE_EXCLUSIVE=8200 GGML_CUDA_HOST_MEMFD=1 GGML_CUDA_HOST_MEMFD_GUARD=1 \
LLAMA_SERVER_CKPT_TAIL=32 LLAMA_SERVER_CKPT_TAIL_MAX=256 LLAMA_QSA_CHUNK=512 LLAMA_PLE_RAW=1 LLAMA_PLE_PREFETCH=1 \
LLAMA_HOST_COMPUTE_PAGEABLE=1 LLAMA_SPEC_ASYNC_PROCESS=1 LLAMA_SPEC_P_CUM=1 LLAMA_GRAPH_SLOTS=6 \
GGML_CUDA_MMVF_ROWS=1 GGML_CUDA_MMVF_FEW_ROWS=1 GGML_CUDA_FA_SPARSE_MIN_KV=8192 MALLOC_MMAP_THRESHOLD_=4194304 \
llama-server \
  --model Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --mmproj mmproj-Qwen3.8-Flash-Next-BF16.gguf --mmproj-device CUDA1 \
  --device CUDA0 --n-gpu-layers 999 --fit off --cpu-moe --moe-cache-mib 13000 --moe-cache-cpu-share 0.7 --load-mode dio \
  --ctx-size 262144 --parallel 3 --kv-unified --flash-attn on --cache-type-k q5_1 --cache-type-v q5_1 \
  --batch-size 8192 --ubatch-size 8192 --cache-ram 0 --ctx-checkpoints 4 \
  --spec-draft-model mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf --spec-type draft-mtp --spec-draft-device CUDA1 \
  --spec-draft-ngl all --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 --spec-draft-n-ubatch 512 \
  --spec-draft-n-max 3 --spec-draft-p-min 0.3 --spec-draft-vocab --spec-draft-backend-sampling \
  -td 2 -tbd 2 -bs --jinja --reasoning on
```

The cache profile file is created on the first clean exit; the cache starts empty without it, and the exclusive core
needs it (see [moe-offload.md](moe-offload.md#cache-profile)). Linux only for `GGML_CUDA_HOST_MEMFD`.
