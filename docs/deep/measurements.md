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

### Phase threshold

2026-10-29, `ed84b76ad`, each model with its profile's settings, follow-ups sent back to back after the 11.7K-token
conversation, the tail checkpoint limit (`LLAMA_SERVER_CKPT_TAIL_MAX`) equal to the threshold:

| `LLAMA_MOE_PHASE_TOKENS` | Flash-Next +100 / +300 / +1000 / +3000 (s) | Ornith-1.5-122B +100 / +300 / +1000 / +3000 (s) |
|---|---|---|
| 512 | 1.16 / 1.77 / 2.12 / 2.66 | 2.13 / 4.12 / 2.27 / 2.86 |
| 256 | 1.28 / 1.42 / 2.32 / 2.53 | 2.03 / 1.78 / 2.27 / 2.88 |
| 128 | 1.17 / 1.46 / 2.30 / 2.52 | 2.10 / 1.81 / 2.28 / 2.91 |
| 64 | 2.08 / 1.43 / 2.28 / 2.49 | 2.98 / 1.95 / 2.27 / 2.87 |

A +300 turn gains most from the large phase (Ornith 4.1 to 1.8 s), a +100 turn (about 120 tokens up to its tail) loses
in it. Decode did not move (Flash-Next 97 to 114 tok/s, Ornith 44 to 46), nor the peaks (GPU0 about 23.9 GB, lowest
MemAvailable about 11.3 GB); one build's runs spread by about ±0.15 s. The profiles take 256: 128 does as well at +300
but moves the turns of 129 to 256 tokens into the large phase, closer to where it loses.

### IQ2_XS on the host

The row8 unpacker for IQ2_XS (`1f42db592`), timed by a standalone program against `vec_dot` and the type's
dequantization (bit-exact with Q8_K activations), ns per row of 3072 weights and per entry against 176 to 178 for
`vec_dot`:

| entries per row | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| unpacked once (row8) | 255 | 151 | 111 | 88 | 84 | 78 | 72 | 65 |

The K-quants keep their `vec_dot`, among the cheapest of the host's types (Q2_K 62, Q3_K 109, Q4_K 71, Q5_K 104 ns
against 287 for IQ3_S): an unpacker carrying the minimums of Q4_K and Q5_K took 341 / 241 ns at one entry and 87 / 75
at eight, never ahead.

In Ornith-1.5-122B, whose layers 0 to 8 and 15 have IQ2_XS experts, the host's fitted cost per entry of a large batch
(measured share, 30 × 512 tokens) went from 0.66 to 0.24 ms for an all-IQ2_XS layer (0.41 to 0.24 and 0.49 to 0.24 for
the two mixed kinds), that of the IQ2_S and IQ3_S layers; its cost per item, the unpacking, from 0.11 to 1.0 ms.
Numerics against its BF16 logits: 128-token batches in the small phase identical to the build before (KLD 0.1259 ±
0.0038 both), large batches with the measured share 0.1246 ± 0.0038 against 0.1245 ± 0.0036 (two more runs of the new
build: 0.1238, 0.1271, and one of the build before: 0.1283; the share follows timings, so which experts the host
computes varies).

Ornith follow-ups (+100 / +300 / +1000 / +3000, s) without the decode suite before them, the builds alternated: before
1.98 / 1.80 / 2.27 / 2.87 and 1.95 / 1.79 / 2.26 / 2.89, after 1.83 / 1.81 / 2.23 / 2.86 and 1.85 / 1.78 / 2.23 / 2.85;
the small phase's 128-token batches gain most. After the decode suite, one run each: before 2.03 / 1.78 / 2.27 / 2.88,
after 2.04 / 1.92 / 2.40 / 2.99, its large batches about 0.1 s longer. Decode after: 43.1 / 44.7 / 40.9 tok/s (cold /
warm / code, draft acceptance 0.76 / 0.72 / 0.70) against 45.2 / 44.3 / 44.3 (0.73 / 0.74 / 0.74), the host's time per
missed expert unchanged (131 µs).

### One pass per turn

The server cut a turn at its checkpoints (the start of the last user message, 4 + `LLAMA_SERVER_CKPT_TAIL` and 4 tokens
before its end), and each small-phase batch reads its missed experts again. An experiment that skipped the cuts of turns
of up to 4096 tokens, and so saved no checkpoint, gave Ornith-1.5-122B with `1f42db592` 1.43 / 1.64 / 2.04 / 2.72 s for
+100 / +300 / +1000 / +3000 against 1.83 to 1.85 / 1.78 to 1.81 / 2.23 / 2.85 to 2.86 with the cuts.

With the checkpoints taken from state marks (`d14f39714`, `3e6618196`), follow-ups alone, the same session:

| | +100 | +300 | +1000 | +3000 |
|---|---|---|---|---|
| Ornith-1.5-122B, cut (s) | 1.83 | 1.78 | 2.23 | 2.86 |
| Ornith-1.5-122B, one pass | 1.64 | 1.66 | 2.07 | 2.75 |
| Flash-Next, cut | 1.20 | 1.30 | 2.18 | 2.58 |
| Flash-Next, one pass (two runs) | 1.08 | 1.20 to 1.23 | 2.10 to 2.11 | 2.45 to 2.52 |

The three checkpoints of a short turn are copied in about 35 ms each on Ornith (149 MiB) and 30 ms on Flash-Next
(112.6 MiB), most of what separates one pass from the experiment. After a greedy conversation (each turn sent twice, the
second restoring the checkpoint before the end of its prompt and processing 4 tokens), Ornith 1.52 / 2.03 / 2.36 /
2.85 s against 1.77 to 1.89 / 2.13 to 2.14 / 2.46 / 2.91 to 2.92, and the conversation's second to fourth turns 616 /
1539 / 636 ms against 747 to 776 / 1741 to 1755 / 802 to 868 ms. Greedy replies differ now and then between two sends
of a turn, cut or not: which experts the cache holds changes the arithmetic.

### Ornith-1.5-122B decode, where the time goes

2026-10-29, `bd5510406`, the profile's settings. An Nsight Systems trace of 4 s of decoding with the kernels inside the
graphs (`--cuda-graph-trace=node`, which slows the step a little), per verification step (median of 64, about 2.4
tokens): 54.0 ms of graph, of which 34.2 ms the device waits for the host, 18.1 ms of its own kernels and 1.8 ms between
them, then 7.5 ms before the next graph, 3.5 ms of it the three drafts on the second GPU. A DeltaNet layer runs 51
kernels: six Q6_K matrix-vector products over 80 MB of weights take 113 µs (708 GB/s), the cached experts 133 µs while
the host computes the others, and about 40 small kernels 125 µs (a non-contiguous concat 15 µs, the gather of the 4 MiB
recurrent state 9 µs). Between graphs the main thread spends about 0.6 ms of its own (sampling, `seq_rm` over the 262K
cells, graph preparation), the rest waiting on the devices; a draft token reads about 350 MB of Q8_0 weights (the layer,
and the head's 49K rows), about 1.07 ms.

The host's misses are bound by host memory. On this machine plain reads reach 34.5 GB/s from two threads up: the DDR4
runs at 2400 MT/s, as configured, its DIMMs being rated 3200 with their XMP profile. The host's dot products of a
request of 7 IQ3_S experts read 33.8 GB/s at 11 threads, 35.4 to 36.1 from 12 to 23 (IQ4_XS: memory-bound at any count);
the engine's workers about 31 GB/s. The GPU reading host memory adds nothing: with it reading at 13.3 GB/s, by copies or
by a kernel, the CPU's dot products fall from 35.5 to 24.1 to 25.1 GB/s.

Decode with other settings (tok/s cold / warm / code, then the experts missing, the experts the CPU computes per request
and its time per expert):

| | decode | missing | per request | per expert |
|---|---|---|---|---|
| profile (two runs) | 46.25 / 44.27 / 42.54, 46.87 / 43.95 / 42.86 | 31.0 %, 32.1 % | 5.90, 6.11 | 129 µs |
| 24 replacements per graph | 44.14 / 41.94 / 42.77 | 36.2 % | 6.85 | 127 µs |
| 12 replacements per graph | 41.97 / 41.44 / 40.79 | 37.6 % | 6.98 | 123 µs |
| 12 host threads | 42.19 / 43.85 / 41.37 | 33.2 % | 6.22 | 132 µs |
| one slot, a 14100 MiB cache | 47.96 / 46.41 / 45.42 | 28.1 % | 5.41 | 131 µs |

A replay of a decode trace (`simsweep3.py`, the service policy, 48 replacements) gives the hits a larger cache would
have: 69.4 % at the profile's 28.2 % of the expert bytes, 71.6 % at 30.6 %, 73.2 % at 32.5 %, 74.9 % at 34.7 %. A
second, static tier holding the experts the trace selects most among those the first lacks adds 0.64 points at 3 % of
the bytes and 1.11 at 4.5 %: a cache that follows the conversation already holds what matters.

The uploads of the replacements overlapped the host's requests 64 % of their time, as if at random (the host serves a
request 68 % of the time), the downloads of the exclusive core 5 %. With each upload started while the host serves no
request (`8c02bccc0`), alternated with the uploads as queued, two runs each: 121.6 and 122.4 µs per expert against 130.2
and 129.5, 767 and 799 µs per request with CPU work against 827 and 820, the same misses (33.1 / 34.1 % against 33.4 /
32.9 %); decode 47.04 / 46.45 / 41.46 and 46.86 / 42.69 / 43.32 against 45.25 / 42.59 / 42.32 and 42.61 / 44.43 / 45.95.
Flash-Next, whose host phase is shorter, one run each: 73.1 µs per expert against 73.9, decode 112.84 / 107.89 / 103.22
against 113.52 / 109.28 / 100.06.

### Ornith-1.5-122B decode: slots, drafter, the device's own work

Two slots instead of three take 596 MiB of recurrent state and rollback groups off the GPU, given to the expert cache
(13600 MiB): 48.17 / 47.56 / 44.63 and 47.86 / 44.92 / 46.60 tok/s, misses 32.3 and 32.6 %, against 44.98 and 44.29
on average with three slots and 13000 MiB, the same build (`8c02bccc0`); its follow-ups peaked at 24115 MiB on GPU0,
against at most 24027 with three slots, so the profile takes 13500 MiB (23957 to 23959 MiB at peak).

The drafter requantized from Q8_0 (a draft call, up to three tokens, `dur(g)` of the speculative statistics; the share of
the drafts accepted at each position; decode on average over cold, warm and code), two runs each:

| drafter | per draft call | accepted at positions 1 / 2 / 3 | decode |
|---|---|---|---|
| Q8_0 | 3.05, 3.19 ms | 0.78 to 0.84 / 0.46 to 0.51 / 0.24 to 0.27 | 46.8, 46.5 |
| Q6_K | 2.61, 2.66 ms | 0.77 to 0.78 / 0.42 to 0.43 / 0.20 to 0.22 | 46.7, 44.7 |
| Q5_K_M | 2.54, 2.54 ms | 0.79 to 0.81 / 0.40 to 0.43 / 0.19 to 0.21 | 45.9, 45.7 |

The lighter drafters save about half a millisecond a step and lose more in the later drafts: Q8_0 stays.

The device's own work between an answer and the next request (the host statistics' turnaround), with the shared expert
before the wait (`14b13ca2c`) and the concat of a thread per element (`6fa669b10`), alternated with `2ac12a39e`, two
slots and 13500 MiB both: 243.1 and 234.9 µs against 273.1 and 272.3; decode 47.69 / 49.43 / 46.19 and 46.48 / 48.17 /
46.30 against 45.56 / 45.50 / 45.49 and 46.27 / 45.34 / 47.29; follow-ups 2.01 to 2.17 / 1.74 to 1.75 / 2.08 to 2.10 /
2.74 to 2.77 s against 2.03 to 2.12 / 1.72 to 1.79 / 2.08 to 2.18 / 2.77 to 2.87. Small phase KLD 0.1269 ± 0.0038
against 0.1259 (the sums in another order).

### Mistral-Small-4 at long context: the cache read in place

Mistral-Small-4-119B GSQ-RCO IQ3_S (its profile: 16800 MiB cache, 10000 MiB core, 262K context, q5_1 K, dio, no
drafter for these runs) decoded the 91K-token code prompt at 33.7 tok/s and the 225K one at 25.6, against 35.4 on the
short prompts, although the host's share of a step hardly changes with the context: the device's turnaround between an
answer and the next request grew from 232 to 310 µs per layer. The MMA flash-attention kernel converted the layer's
whole q5_1 cache to F16 on every call (see [attention.md](attention.md#a-quantized-cache-read-in-place-by-the-tensor-core-kernel)):
at 64K cells the op took 180 µs for one query against 85 in F16.

With the cache read in place by the kernel and the shared expert computed before the host wait (`moe_host_defer` in
deepseek2), the same profile and prompts: 91K decode 40.0 tok/s (33.7), 225K 36.2 (25.6), short 36.7 (35.4), prefill
802 and 370 tok/s both ways, peak GPU0 23552 against 23566 MiB, turnaround 240 µs. At 131K context with a 15800 MiB
cache: 38.1 against 34.1 at 91K, 35.2 against 34.3 short; an F16 K cache there gave 34.3 and 35.0, so the conversion's
cost was in its traffic, not in the bytes of the cache. The op itself at 64K cells: one query 79 µs (q5_1), three
194; q8_0 82 and 207; with the q5_1 rows staged by `cp.async` a tile ahead, 73 and 166 (Ornith's 4-query shape 214
against 242), and the profile's prompts 39.3 / 37.7 / 37.3 tok/s at 91K / 225K / short.

Two rounds on the kernel's own compute (the fixup, the dequantization, the prefetches, the tile configurations; see
[attention.md](attention.md#a-quantized-cache-read-in-place-by-the-tensor-core-kernel)) take the op at 64K cells to
48 µs for one query and 123 for three (73 and 166), 23 and 44 at 16K, Ornith's 4-query shape to 176 (214). The
profile's prompts, one run each: 41.5 / 39.1 / 36.9 tok/s at 91K / 225K / short (the staged loader's 39.3 / 37.7 /
37.3, and 37.5 to 40.1 at 91K between two runs of the same build), prefill 906 / 424 tok/s (800 / 370), peak GPU0 23488
against 23516 MiB. Decode is bound by the host's expert traffic, so the attention's 40 layers' savings of a few µs each
stay within the run-to-run spread; the gains are the kernel's.

Ornith-1.5-122B (its harness defaults: 13000 MiB cache, three slots, the MTP drafter with three drafts, q5_1 K/V),
whose twelve full-attention layers took the conversion for every verification batch: 91K decode 40.0 tok/s against
38.3, 230K 34.8 against 29.1, the short suite 47.1 / 48.1 / 48.6 against 48.4 / 46.9 / 46.5 (acceptance 0.61 to 0.70
between runs), prefill 1452 / 621 against 1455 / 620, peak GPU0 23758 against 23756 MiB; turnaround 277 against 315 µs.
The op for its shape at 64K cells (256/256, 16 query heads per K/V head, 4 queries): 242 µs against 553 (187 in F16).

### Mistral-Small-4 decode: the step on the device

An Nsight Systems trace of the profile above decoding the 91K prompt (the kernels of the compute stream, split into
steps at the gaps between them; see [moe-offload.md](moe-offload.md#the-decode-step-on-the-device)) put the step at
25 ms: 11 of them in the kernel that waits for the host's experts, 12 of other kernels, 2.2 between the steps. Per layer,
the attention side 219 µs (kernels 154), the device's own MoE work 182 (run while the host computes, critical in the
layers with few misses), the request 26, the wait 311 on average (50 to 435 between runs). The host path itself is at
the memory's ceiling (the IQ3_S dot product streams 4.6 GB/s per core, 11 workers 45 GB/s, the two DDR4-3200 channels
46), so the step's other parts were the levers.

With the request kernel on eight blocks, the activation scale chain fused, the exclusive core's victims downloaded in
512 KB pieces, the small copies made by a kernel, the sampled probabilities copied on demand and the graph chunks
ramping from 64 nodes: the gap between steps 0.85 ms (2.2), the device's MoE work 142 µs per layer (182), the request
18 (26), the attention side 183 (219); the traced steps averaged 21.7 ms against 24.7 with the same host wait (11 ms per
step). The CUDA graph chunk size itself (512, 128, 48 nodes) made no measurable difference. End to end, without the
profiler, two runs per build against the installed `37cdbf98b`: the short suite 37.1 and 37.4 tok/s against 36.9 and
36.6, the 91K prompt 39.9 and 39.5 against 40.2 and 40.4, prefill 920 against 918 tok/s, peak GPU0 23454 against 23496
MiB: within the run-to-run spread. The step is bound by the host path at this context; the trace proves the mechanics,
not a throughput gain.

The fused scale chain is bit-identical to the ten operations it replaces (the same float operations in the same
order), and the kernel copies' limit must stay below the download piece size: a 1 MB limit took the pieces as kernels
on the compute stream and cost the attention side 70 µs per layer.

### Swaps in flight

2026-11-12, from the engine's exit statistics and a replay of the service policy's mechanics; measured in the engine
the next day (below).

The logs: on Mistral-Small-4 (36 layers, 128 experts, 4 per token, 10.3 MiB per expert) the cache replaced 13 to 14
experts per graph where Ornith replaces about 69 and Flash-Next about 76, as an expert is three uploads and each waits
for the host to be idle (154,319 waits of 493 us on average in one run). A build whose device step was 33 us per layer
shorter (181 against 214 us from an answer to the next request) missed more (48.9 % against 46.3 %) and decoded at
the same speed: the swaps start with the requests, two each, and upload with the idle time left.

The replay runs the plan, the starts, the serial uploads and the downloads of exclusive victims as the code does
(`mech2.py` in the bench directory), on Ornith's recorded routing and, for Mistral, on selections drawn from a session's
counts. Hits with the bound on the swaps started and not uploaded yet:

| in flight | Ornith, 48 swaps per graph | Ornith, 96 swaps per graph | Mistral-Small-4, 48 swaps per graph |
|---|---|---|---|
| no bound | 67.35 % | 64.42 %, 188 uploads queued | 44.74 %, 237 uploads queued |
| 1 | 67.16 % | 67.11 % | 50.37 % |
| 2 | 67.37 % | 68.34 % | 50.45 % |
| 4 | 67.36 % | 68.29 % | 50.37 % |
| 8 | 67.35 % | 68.20 % | 50.26 % |
| 16 | 67.35 % | 67.99 % | 50.20 % |
| 32 | 67.35 % | 67.60 % | 49.86 % |

A bound of 1 holds Ornith's swaps below what its uploads can take; from 2 to 8 the result barely moves, and the default
is 4. Ornith at 48 swaps per graph has no backlog, so the bound changes nothing there; with it, 96 swaps per graph hit
more than 48 do.

In the engine (2026-11-13, the build with upstream `8ca5a8008` merged in; Mistral-Small-4 on the spec bench twice per
server, two servers per row, no draft):

| | decode | experts missing | uploads past the timeout |
|---|---|---|---|
| the build before the merge | 36.9 to 37.1 tok/s | 50.2 to 50.4 % | 1,531 to 1,686 |
| merged, no bound | 37.2 to 37.9 tok/s | 49.3 to 49.7 % | 1,414 to 1,450 |
| merged, bound 4 | 44.1 to 44.9 tok/s | 36.3 to 36.7 % | 544 to 650 |

One server per bound: 45.1 and 45.6 tok/s at 1, 44.7 and 45.6 at 2, 44.4 and 44.6 at 8, 44.1 and 44.6 at 16, so the
engine is as flat as the replay from 1 to 16 and gains more than the replay's hits say: the swaps not started for want
of a downloaded victim fall from about 210,000 to 5,000 per run, and a request is answered in 453 us instead of 598.
With the EAGLE draft: 49.0 to 50.0 tok/s against 41.1, and 45.8 to 47.7 against 41.8 after a 91K-token prompt.

Ornith (48 swaps per graph, MTP draft, one server each): 57.0 / 52.4 / 51.8 tok/s on the three decode prompts against
53.1 / 52.2 / 50.2 before, 33.4 % of the experts missing against 35.6 %; 96 swaps per graph give 55.0 / 52.7 / 51.0, no
better than 48, unlike the replay. Flash-Next: 111.8 / 106.7 / 110.5 against 117.8 / 97.0 / 103.7, the bound holding
2,750 requests of 55,488.

### The score window of the service policy

2026-11-13, `LLAMA_MOE_HOST_DECAY` with the bound of 4 swaps in flight, build `a06ee0391`.

Mistral-Small-4, no draft, the spec bench twice per server:

| decay | decode | experts missing | replaced per graph | request to answer / answer to request |
|---|---|---|---|---|
| no swap (`LLAMA_MOE_HOST_SWAPS=0`) | 42.0 to 42.4 tok/s | 50.0 % | 0 | 530 / 141 us |
| 0.9, two servers | 44.8 to 45.3 tok/s | 36.0 to 36.2 % | 12.4 | 448 / 185 us |
| 0.97 | 46.2 tok/s | 35.3 % | 11.8 | 432 / 184 us |
| 0.99 | 48.9 to 49.2 tok/s | 35.4 % | 7.0 | 410 / 172 us |
| 0.995, three servers | 49.7 to 50.1 tok/s | 36.4 to 36.5 % | 4.4 | 406 / 164 us |
| 0.998 | 48.5 to 48.7 tok/s | 39.9 % | 2.5 | 432 / 155 us |
| 0.999 | 46.6 to 47.0 tok/s | 42.5 % | 1.7 | 458 / 150 us |
| 1.0 | 44.6 to 45.4 tok/s | 45.8 % | 0.6 | 486 / 145 us |

The longer window does not hit more: it replaces a third as many experts for the same hits, and both the host's
answer and the device's step are shorter for it. With the hysteresis at 1.25 or 1.5 on top of 0.995 the speed is the
same (49.4 to 50.2 tok/s) for 2.8 and 2.2 experts replaced per graph. With 14 or 18 host threads instead of 11:
44.2 to 44.9 tok/s, no gain.

With the EAGLE draft (2.2 tokens per request): 49.5 to 49.8 tok/s at 0.9, 51.0 to 51.7 at 0.99, 50.1 to 50.9 at
0.995, and 49.7 tok/s after the 91K-token prompt at 0.995 (45.8 to 47.7 at 0.9). The draft adds little once the cache
swaps less: 50 tok/s without it at 0.995.

Ornith (3 tokens per request) is flat from 0.9 to 0.995 (858, 848, 883 and 852 us per layer for the host and the
device together at 0.9, 0.97, 0.99 and 0.995) and loses at 0.998 (971 us, 41.5 % of the experts missing against 34 %).
Flash-Next (4 tokens per request) loses past 0.97: 14.6 to 16.4 % missing at 0.9, 19.9 % at 0.99, 23.2 % at 0.995. The
default stays 0.9; the value is a setting of the Mistral-Small-4 profiles.

### The merge of upstream 8ca5a8008

2026-11-13, the merged build against the build before it (`47e4fcd`), unified q5_1 KV as served.

Numerics, mean KLD on wikitext, no tail skip:

| | Mistral-Small-4 | Ornith | Flash-Next |
|---|---|---|---|
| host mode, 4-token batches: merged against before | 0.078 | 0.036 | 0.028 |
| host mode, 4-token batches: before against itself | 0.072 | 0.030 | 0.026 |
| large phase: merged against before | 0.085 | 0.036 | 0.022 |
| large phase: before against itself | 0 | 0 | 0 |
| large phase against the BF16 logits, before | 0.1913 +- 0.0021 | 0.1270 +- 0.0038 | |
| large phase against the BF16 logits, merged | 0.1925 +- 0.0021 | 0.1290 +- 0.0039 | |

Host mode does not repeat itself (which experts the host computes depends on the timing, and the routing and the
quantized cache amplify the rounding); the large phase does, and the merged build is another rounding of it at the
same distance from the BF16 model.

Prefill: Mistral-Small-4 on the 91K-token prompt 866 to 868 tok/s against 885 to 893 with the draft, 893 against 909
without; Ornith 1,414 against 1,436 tok/s at 92K tokens, 1,790 against 1,806 at 35K, equal at 12K, 679 against 649 at
2.3K; Flash-Next equal within the spread from 12K to 93K tokens.

The loss is the MMA flash-attention kernel. A trace of the same 30 s of Mistral-Small-4's prefill (context 45K to
70K) has that kernel at 86 % of the kernel time (24.7 of 28.6 s), 192 / 221 / 250 / 280 ms per call over four batches
before the merge and 200 / 230 / 260 / 290 ms after; the expert products cost the same on both. Upstream's K/V row
lists (#29510) reach the F16 tile loads as a pointer tested per row; with none given, the test alone costs those 4 %,
and testing it once per tile instead does not give them back (882 tok/s). As a template parameter of the kernel, set
on the host from the operation, the loads without rows compile as before: 906 to 911 tok/s on the 91K-token prompt
without the draft (909 before the merge, 887 to 893 after), Ornith 1,437 tok/s at 92K tokens and 1,811 at 35K. The
F16 kernels are compiled twice for it, about two thirds more CPU time to build the CUDA backend.

`GGML_CUDA_GRAPH_KEEP_S=300`: 320 to 460 MiB more of the device held on Mistral-Small-4 (peak 24,062 MiB with the
draft against 23,602 at the default 10 s), and no decode gain resolved after the 91K-token prompt (47.7 and 41.9 tok/s
on two servers against 45.8). Not set in the profiles.

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
