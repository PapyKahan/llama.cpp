# Measurements

## Machine

| | |
|---|---|
| GPU 0 | RTX 3090, 24 GB, sm_86 (also drives the display; no desktop session during the runs) |
| GPU 1 | RTX 3070 Ti, 8 GB |
| CPU | Ryzen 9 5900X, 12 cores / 24 threads |
| RAM | 62.7 GiB DDR4, dual channel: about 35 GB/s shared by the CPU and DMA, 29–31 GB/s of expert reads by the host workers |
| Links | PCIe 4.0 x8 per GPU: 13.1 GB/s host to device in 0.7 MB pieces, 13.4 GB/s in large ones |
| Software | CachyOS, Linux 7.2, NVIDIA driver 615.78, CUDA 13.4 with gcc 15, `-DCMAKE_CUDA_ARCHITECTURES=86` |

## Model files

- Target: ISTA-DASLab `Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_S, two splits. The experts take 47.7 GiB of host memory; the
  per-layer n-gram embedding table (26.8 GiB, IQ4_NL) is read from the second split through the page cache.
- Drafter: Unsloth `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`.
- Projector: `mmproj-Qwen3.8-Flash-Next-BF16.gguf` on the second GPU.

Strata (0.2.2, engine 0.1.47) ran the same IQ3_S files on the 3090 alone, at 262K context with int8 KV, in its low-RAM
mode (RAM holds only the experts the GPU lacks; 5,868 experts on the GPU).

## Harness

The tools live outside the branch (they carry machine-specific paths); they talk to any OpenAI-compatible server.

- **Standard bench** (`bench2.py`): decode of 1200 tokens on three prompts (cold, warm, code); prefill of prompts of
  2.4K, 11.6K, 34.8K and 92.8K tokens, each starting with a fresh nonce so that no prompt cache serves it, timed as prompt
  tokens over the time to the first streamed token; follow-up turns of about 100, 300, 1000 and 3000 new tokens on a
  cached 11.6K-token conversation, timed to the first token.
- **Code task** (`code_ctx.py`): a question about real source code at 2.3K, 84K and 210K tokens of context, nonce first,
  300–600 sampled tokens of answer. It exposes the long-context costs the standard bench hides.
- **Step cost** (`step_ctx.py`): from the server log, per request: steps, ms per step, drafts per step, acceptance.
- **Memory**: `/proc/meminfo` sampled every 2 s during a run, and an inventory of the server's mappings by kind (anonymous,
  heap, memfd, `cudaMallocHost`'s `/dev/zero` mappings, files) after a request.

## Method

- Every configuration starts its own server; the cache profile is copied fresh for each run so that all start from the
  same cache.
- Decode throughput moves 4–5 % from run to run, more at long context, because the sampled text differs: the cache
  state, the split between the GPU and the CPU and the tail skip depend on timing, so even greedy text is not
  reproducible here. Comparisons are alternating A/B runs, two rounds or more, and the figure compared is the cost of a
  step at a given number of drafts per step, not tokens per second.
- Prefill is measured from the client (streamed first token) and agrees with the server's own timing within 0.5 %.
- Lowest MemAvailable is taken while serving (after the first minute): the start itself loads all 47.7 GiB of experts
  before the exclusive core releases its share.

## Results

### Standard bench

| Run | decode cold / warm / code (tok/s) | prefill 2.4K / 11.6K / 34.8K / 92.8K (tok/s) | follow-ups +100 / +300 / +1000 / +3000 (s) |
|---|---|---|---|
| upstream + #30965 + #30966 (ub1024, cache 14.6 GiB, share 0.7) | 61.0 / 61.3 / 55.2 | – / 320 / 313 / 298 | |
| Strata 0.2.2 | 88.9 / 87.1 / 72.7 | – / 1389 / 1616 / 1709 | 0.72 / 0.96 / 3.14 / 3.34 |
| `deep`, all experts page-locked, run 1 | 114.0 / 110.3 / 104.4 | 550 / 1361 / 1756 / 1848 | 0.93 / 1.48 / 3.03 / 4.32 |
| `deep`, all experts page-locked, run 2 | 113.8 / 110.4 / 99.5 | 515 / 1314 / 1751 / 1839 | 0.96 / 1.55 / 3.04 / 4.32 |
| `deep`, exclusive core, run 1 | 112.4 / 108.5 / 99.8 | 560 / 1304 / 1713 / 1825 | 0.92 / 1.51 / 3.08 / 4.31 |
| `deep`, exclusive core, run 2 | 108.7 / 105.7 / 101.6 | 545 / 1323 / 1710 / 1831 | 0.98 / 1.60 / 3.11 / 4.38 |

The `deep` runs are of 2026-10-24 with the configuration of [README.md](README.md#running-it) (the page-locked runs
without the three exclusive-core variables).

### Code task

| Run | 2.3K: prefill / decode | 84K: prefill / decode | 210K: prefill / decode (tok/s) |
|---|---|---|---|
| Strata 0.2.2 | 698 / 68.4 | 1708 / 76.1 | 1686 / 76.1 |
| `deep`, page-locked, cumulative stop 0.3 (two runs) | 534–537 / 82.1–85.3 | 1890–1894 / 77.5–78.7 | 1743 / 83.1–85.6 |
| `deep`, page-locked (three runs) | 480–531 / 77.3–80.3 | 1879–1882 / 69.7–71.6 | 1843 / 77.0 |
| `deep`, exclusive core (three runs) | 530–540 / 78.7–85.0 | 1870–1873 / 72.9–75.1 | 1838 / 72.6 |

The spread of the decode figures comes from the drafter's acceptance (0.50 to 0.74 across these runs). Cost per step:

| | 2.3K | 84K | 210K |
|---|---|---|---|
| Strata (its own log) | 26.4 ms at 1.4 drafts | 27.6 ms at 1.8 | 32.9 ms at 2.4 |
| `deep` | 27.2–29.7 ms at 1.8–2.1 | 28.5–29.4 ms at 1.9–2.2 | 32.3–35.2 ms at 2.3–2.6 |

### Memory

| | page-locked | exclusive core | Strata |
|---|---|---|---|
| host expert memory (Shmem) | 47.7 GiB | 40.2 GiB | about 39.4 GiB page-locked |
| MemAvailable after one request | 7.5 GiB | 15.3 GiB | |
| lowest MemAvailable while serving | 5.4–5.5 GiB | 13.0–13.3 GiB | 13.8 GiB or more |
| unreclaimable slab + vmalloc, whole system | 1.38 GiB | 1.33 GiB | |

Besides the experts, the server's anonymous memory grows from 0.6 GiB at start to about 1.8 GiB after a request and more
over a session: context checkpoints (112.6 MiB each, up to four per slot), the drafter's and the host's buffers.

### Progress over the work

Standard bench, each line the best configuration of its day (the configurations differ, so the lines show the trend,
not single changes):

| Stage | decode cold / warm / code | prefill 11.6K / 34.8K / 92.8K |
|---|---|---|
| stock profile, 2026-10-22 (ub1024, cache 13.4 GiB) | 48 / 46 / 44 | 278 (one size) |
| + #30965, #30966, CPU share 0.7 | 58.6 / 62.7 / 54.9 | 314 (one size) |
| + target on the 3090 alone, key pool per stream, tail skip, prefetch, pinned experts (ub4096) | 70.7 / 70.4 / 67.1 | 923 / 1016 / 1038 |
| + host mode | 84.4 / 89.0 / 71.0 | |
| + cache phases, service policy, 8192-token ubatches | 93.5 / 103.7 / 90.3 | 744 / 1141 / 1346 |
| + prefill kernels, balanced chunks, async catch-up | 102 / 97.5 / 96.6 | 1297 / 1704 / 1788 |
| + cache profile, indexer in place, row8 host kernels, host share of large batches | 99.0 / 96.6 / 93.5 | 1343 / 1721 / 1843 |
| + k-pool tables kept, backend sampling | 109 / 97 / 97 | 1343 / 1734 / 1849 |
| + graph slots, cumulative stop 0.3 | 108–112 / 112 / 100–103 | 1351–1364 / 1747–1760 / 1858 |

### Quality

KL divergence against a GPU-only reference at ubatch 512 (wikitext-2, 6 × 512 tokens):

| Change | KLD | |
|---|---|---|
| host mode, 32-token batches | 0.0230 | GPU-only at the same ubatch: 0.0235 (the gap to ubatch 512 is QSA's top-k selection) |
| tail skip, rank 7 | +0.020 to +0.027 | top-1 agreement −1 point |
| host share of large batches 0.4 / 1.0 | 0.016 / 0.017 | the same as changing the ubatch from 1024 to 2048 |

## Pitfalls of measuring here

- **systemd-oomd**: the user slice is killed at 80 % memory pressure for 30 s, and the victim is the cgroup holding the
  most memory: a server started from an SSH shell took the whole login session down with it (twice, while testing huge
  pages on the host experts). Run servers in a scope of their own (`systemd-run --user --scope`).
- **earlyoom** ignores page-locked memory: it is counted as Shmem, not AnonPages.
- **ananicy** (CachyOS) puts SSH sessions in a background class (SCHED_IDLE, nice 16) that every child inherits; a server
  started from such a shell gets the CPU only when nothing else wants it. Set SCHED_OTHER and nice 0 before starting it.
- **CUDA spins** in synchronizations with two contexts on 24 logical CPUs; a helper thread that waits a lot takes a core
  from the host workers (see the exclusive core in [moe-offload.md](moe-offload.md#exclusive-core)).
- **A KLD that grows smoothly with a knob can still be a bug**: the host share of large batches first showed 0.017 /
  0.035 / 0.074 at shares 0.1 / 0.4 / 1.0, blamed on CPU rounding; bisecting the mechanism found the last layer losing
  the host's experts.
