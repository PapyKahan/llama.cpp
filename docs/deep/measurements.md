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

### After the merge with upstream

2026-10-28: the branch at `3f62edf67` (upstream `39febd2ad` merged in) against `170656bca`, the same day, alternating,
in the configuration of [README.md](README.md#running-it). Numerics first, against the base logits of earlier builds:

| Check | merged | before | |
|---|---|---|---|
| Flash-Next host mode, 32-token batches, against GPU-only ubatch 512 (6 × 512) | 0.0242 ± 0.0020 | 0.0230 ± 0.0019 | |
| Flash-Next large batches, no host share, against the same phases before the merge (2 × 2048) | 0.0180 ± 0.0020 | | the floor of any change in the arithmetic |
| Flash-Next large batches, host share 0.4, against the same base | 0.0169 ± 0.0017 | 0.0159 ± 0.0014 | |
| Ornith-1.5-122B host mode, 4-token batches, tail skip 7, against BF16 (30 × 512) | 0.1279 ± 0.0037 | 0.1284 ± 0.0038 | |
| Ornith-1.5-122B 512-token batches in the large phase, against BF16 | 0.1264 ± 0.0037 | | the quant's own: 0.126 to 0.128 |

Flash-Next, two runs of each build:

| | merged | before |
|---|---|---|
| decode cold / warm / code (tok/s) | 111.7–113.7 / 108.4–112.5 / 98.1–102.1 | 111.9–114.2 / 107.9–110.9 / 93.2–97.0 |
| prefill 2.4K / 11.6K / 34.8K / 92.8K (tok/s) | 544–550 / 1312–1321 / 1708–1713 / 1811–1827 | 550–557 / 1317–1329 / 1701–1716 / 1829–1834 |
| follow-ups +100 / +300 / +1000 / +3000 (s) | 0.98 / 1.51 / 3.09–3.10 / 4.27–4.35 | 0.95–0.99 / 1.49–1.61 / 3.08–3.12 / 4.33–4.44 |
| code task prefill 2.3K / 84K / 210K (tok/s) | 535–536 / 1870–1872 / 1726–1727 | 529–533 / 1867–1871 / 1727–1731 |
| code task ms per step (drafts per step) at 2.3K / 84K / 210K | 28.2–28.4 (2.0) / 28.5–29.8 (2.0–2.1) / 31.8–33.2 (2.4–2.5) | 27.7–28.1 (1.8–1.9) / 29.6–29.8 (2.0–2.1) / 32.2–33.4 (2.5–2.6) |
| lowest MemAvailable (GiB) | 12.3–12.6 | 12.3–12.6 |

Both builds had about 0.6 GiB less memory available than on 2026-10-24: the machine, not the branch.

### Ornith-1.5-122B

An own GSQ-RCO IQ3_S build of Ornith-1.5-122B (`qwen35moe`: 48 layers, 256 experts with 8 selected per token, DeltaNet
with full attention every fourth layer, vision; 53.4 GB), with a Q8_0 MTP drafter on the second GPU. `deep` runs it with
a 13000 MiB decode cache, a 5800 MiB core (also the exclusive core; the full-attention mask of a large ubatch leaves no
room for more), no host share of large batches, the prefetch from 256-token batches, the cumulative draft stop at 0.4
and `--cache-ram 2048`. Stock is llama.cpp v0.8.0 with the same model, drafter, context, slots and KV types and `--fit`
placement, the best of the placements tried. Same harness, 2026-10-28:

| | stock v0.8.0, ub 4096 | stock v0.8.0, ub 8192 | `deep` before the merge | `deep` merged |
|---|---|---|---|---|
| decode cold / warm / code (tok/s) | 19.8 / 20.7 / 20.5 | 18.3 / 18.2 / 18.7 | 44.2 / 40.7 / 42.8 | 41.1 / 40.7 / 41.4 |
| ms per step, short prompts | | | 55.3–61.0 | 56.7–61.4 |
| prefill 2.3K / 11.6K / 34.8K / 92.8K (tok/s) | 620 / 893 / 843 / 750 | 581 / 1007 / 1052 / 886 | 494 / 1345 / 1612 / 1341 | 493 / 1350 / 1610 / 1338 |
| follow-ups +100 / +300 / +1000 / +3000 (s) | 2.45 / 2.86 / 3.47 / 4.42 | 2.69 / 3.12 / 3.73 / 4.65 | 1.66 / 3.26 / 4.60 / 4.77 | 1.65 / 3.30 / 4.62 / 4.77 |
| code task 91K: prefill / decode (tok/s) | 727 / 17.7 | 898 / 16.6 | 1335 / 36.1 | 1337 / 33.6 |
| code task 230K: prefill / decode (tok/s) | 556 / 16.5 | 581 / 15.1 | 805 / 29.6 | 806 / 30.2 |
| peak VRAM GPU0 / GPU1 (MiB) | 23211 / 5884 | 23339 / 5894 | 23979 / 6012 | 23987 / 6012 |
| lowest MemAvailable (GiB) | 15.3 | 10.0 | 7.4 | 7.2 |

Follow-up turns of 300 tokens and more are slower than stock's: up to 512 tokens the misses of each 128-token batch are
computed on the host, and a larger batch uploads every expert the core lacks, about 3 s whatever its size.

### Follow-up turns

How a follow-up spends its time (server log, 2026-10-28): the server cuts it into batches at the start of the last user
message, at 4 + `LLAMA_SERVER_CKPT_TAIL` and at 4 tokens before the end, each with a copy of the recurrent state (149 MiB
on Ornith, about 35 ms). A small-phase batch costs about its tokens (Ornith 11 to 13 ms per token, Flash-Next 5 to
6), so merging the small batches saves nothing; the tail batch also lets the cache adapt to the conversation. A batch of
more than 512 tokens adds two changes of phase (about 200 ms each) and the upload of every expert the core lacks.

- After a large batch, the experts of the lent slots went back in jobs of 16; a checkpoint restored by the next request,
  and the inputs of its graphs, queued behind whole jobs: 350 to 470 ms before the restore. One expert per job
  (`b1a241661`): 17 to 45 ms; Flash-Next follow-ups 0.1 to 0.2 s faster back to back, the same when 2 s apart.
- `LLAMA_SERVER_CKPT_TAIL_MAX=512` (`1e888e4e7`), each turn 2 s after the previous answer, two runs each:

| | +100 | +300 | +1000 | +3000 |
|---|---|---|---|---|
| Ornith-1.5-122B, tail always (s) | 1.73–1.80 | 3.55–3.61 | 4.55–4.66 | 4.68–4.77 |
| Ornith-1.5-122B, tail after small batches only | 1.88–1.93 | 3.73–3.90 | 3.89–3.99 | 3.97–4.05 |
| Flash-Next, tail always | 1.02–1.09 | 1.61–1.68 | 3.03–3.12 | 4.30–4.62 |
| Flash-Next, tail after small batches only | 1.12–1.16 | 1.83–1.84 | 2.75–2.79 | 4.00–4.07 |

Turns sent back to back show the same gains on large turns and lose a little more on the next small one, which then
pays the re-admission the tail batch used to hide.

### Ornith-1.5-122B decode

The host computes the misses: 6 to 6.8 experts per layer and request, about 130 to 140 µs per expert of about 4.4 MB,
the memory bandwidth of the host (30 to 34 GB/s), so about 0.8 to 0.9 ms per layer, then about 280 µs of device work to
the next layer: 52 to 61 ms per step at 2 drafts. A replay of a decode trace against the cache policies (31 % of the
selected experts missing, as measured): the service policy is at its best decay (0.9), no hysteresis helps, and 24 to
192 replacements per graph give the same hits; a cache that knew the next 20 graphs would miss 14.5 %. Fewer
replacements then only remove upload traffic from host memory: `LLAMA_MOE_HOST_SWAPS=48`, two runs each, decode
45.2 tok/s against 42.9 (96), 51 to 54 ms per step against 57 to 58. Flash-Next keeps 96: at 48 it misses 21 % instead
of 17 %.

### Measured host share and lent memory

2026-10-28. Ornith-1.5-122B at fixed host shares first, the large batches of the follow-up and short-prompt suites,
before the batch wait was fixed (see [moe-offload.md](moe-offload.md#large-batches-shared-with-the-host)):

| share of the lacking experts | 1018-token batch (ms) | 3008-token batch | 5806-token batch | host busy, 1018 tokens |
|---|---|---|---|---|
| 0 | 3227 | 3267 | 3440 | 0 % |
| 0.05 | 3132 | 3402 | 4097 | 27 % |
| 0.15 | 2908 | 4909 | 8366 | 47 % |
| 0.3 | 3365 | 9584 | 17747 | 87 % |

Without a share every large batch uploads the same 42.6 GB at 12.4 to 13.2 GB/s, about 3.3 s from 1K to 5.8K tokens. A
fixed share pays only at about 1000 tokens, and only while the host keeps up. With the wait polling from one thread, each
staging slot's uploads ran at 13.1 to 13.3 GB/s whatever the host's load, against 5 to 6 GB/s while the device waited
for a late host before. The host's cost fitted per kind of layer predicted its time within a few percent from the
actual entries (1687 / 2871 / 3729 ms predicted for 1687 / 3053 / 3767 measured); what the profile expected of the
entries did not (the experts least selected while decoding got nearly the average number of tokens of a prompt).

Then the installed `fd5cd5fd5` (before: Flash-Next with the fixed model of the share, Ornith without a share) against the
measured share and the lent memory (after), each model with its profile's settings, the same evening:

| | Flash-Next before | after | Ornith-1.5-122B before | after |
|---|---|---|---|---|
| decode cold / warm / code (tok/s) | 111.5–113.4 / 108.5–108.7 / 99.2–104.6 | 114.3 / 108.2 / 102.6 | 43.6–47.0 / 42.8–43.5 / 42.2–42.4 | 45.0 / 45.3 / 45.0 |
| prefill 2.3K / 11.6K (tok/s) | 555–568 / 1375–1379 | 577 / 1531 | 577–578 / 1463–1467 | 646 / 1654 |
| prefill 34.8K / 92.8K (tok/s) | 1723–1733 / 1835–1846 | 1809 / 1838 | 1656–1668 / 1354–1358 | 1744 / 1357 |
| follow-ups +100 / +300 (s) | 1.16–1.22 / 1.75–1.89 | 1.17 / 1.78 | 2.01–2.02 / 3.98–4.00 | 2.08 / 3.90 |
| follow-ups +1000 / +3000 (s) | 2.79–2.83 / 4.24–4.28 | 2.38 / 2.58 | 3.92 / 4.38–4.39 | 2.28 / 3.11 |
| peak VRAM GPU0 (MiB) | | | 23971–23977 | 23973 |

A change of phase takes 15 to 35 ms each way, against about 200 ms (up to 570 ms with admissions in flight). A first
version of the policy, which let the host take experts while the predicted cycle merely did not grow, gave follow-ups of
2.23 / 2.51 s on Flash-Next but lost 3.5 % on its 34.8K prompt: there the device's own work binds, and the host piled up
work for nothing until the next batch measured the damage. Each expert must now shorten the cycle.

Numerics against the base logits of [the merge](#after-the-merge-with-upstream): Ornith 512-token batches in the large
phase, the lent memory alone 0.1264 ± 0.0037 (the same as before), with the measured share (the host took 53 % of the
experts the core lacks over 30 batches) 0.1271 ± 0.0037; Flash-Next 2 × 2048, measured share 0.0149 ± 0.0013.

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
