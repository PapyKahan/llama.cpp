# Expert offload

Upstream's MoE cache (`--cpu-moe --moe-cache-mib N`, [docs/moe-cache.md](../moe-cache.md)) keeps the most used experts
of host-resident MoE layers in device memory for small batches; a miss is uploaded into the cache, or with
`--moe-cache-cpu-share F` (upstream PR #30966, merged into this branch) computed on the CPU for a share of the misses.
The cache is groups of slots, one group per set of layers with the same expert tensor layout, an SLRU per group, and a
slot map per layer that the graph reads to find each selected expert.

For Qwen3.8-Flash-Next IQ3_S the host experts are 47.7 GiB (48 layers × 512 experts × gate, up and down, 0.4 to 0.9 MB
per expert matrix, seven groups by quantization type) and the cache holds about 6,600 of the 24,576 experts in 13 GB.
Decoding then misses 15 to 25 % of the selected experts. The work below changes where misses are computed, how the cache
follows the routing, how prompts use the same memory, and what stays in RAM.

## Host mode

`LLAMA_MOE_HOST=1` (commit `e1cd4e00f`, files `src/llama-moe-host.{h,cpp}`, `ggml/src/ggml-cuda/moe-host.cu`).

For a batch of up to `LLAMA_MOE_HOST_MAX_TOKENS` tokens (default 32, 128 in the measured setup), a MoE layer does not
upload its misses and does not split its graph between the GPU and the CPU backend. Instead:

1. `GGML_OP_MOE_HOST_POST` (one CUDA block) writes the request into a mailbox in mapped host memory: the selected
   expert ids, their routing weights, the layer input and, per selected expert, its cache slot or -1. It then publishes a
   sequence number.
2. The device computes its cached experts and the shared expert as usual; a missing expert reads one of the zero slots
   that follow the cached ones in each bank, so it contributes nothing on the device.
3. On the host, a service thread polls the mailboxes and hands the missing experts of each request to a worker pool
   (`LLAMA_MOE_HOST_THREADS`, default half the hardware threads minus one), which computes them with ggml-cpu's dot
   products on the host copies, and writes the answer `y` into the mailbox.
4. `GGML_OP_MOE_HOST_WAIT` (every block) waits for the answer's sequence number and adds `y`. After a timeout the
   layer goes on without the host's experts rather than hang.

The forward pass therefore has no host synchronization: it is one CUDA graph, the device and the host work on the same
layer at the same time, and the per-layer cost is the longer of the two plus the turnaround (about 0.55 ms per layer
for a 3-to-4-token verification on the test machine). The slot table the graph reads lives in the mailbox, written by
the host between requests.

Host kernels: the workers use the type's `vec_dot`, and for an expert selected by two tokens or more (three for IQ4_XS,
IQ4_NL and Q2_0, whose `vec_dot` is cheap) `ggml_cpu_row8_unpack` / `ggml_cpu_row8_dot` (commits `2b37cca0f`,
`1f42db592`, AVX2): a weight row of IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL or Q2_0 is unpacked once into int8
weights with integer and float scales, then dotted against up to four activation rows per pass; the integer products are
those of `vec_dot`, so the result is bit-exact for the Q8_K types. `LLAMA_MOE_HOST_ROW8=<n>` raises the threshold, 0
turns it off. The host is DRAM-bound with it (29–31 GB/s of expert reads on a dual-channel DDR4 system).

Upstream PR #30448 (AVX2 dot products for IQ3_S and IQ3_XXS, open) is merged into the branch and used by the host path.

Also in that commit, for the decode step of this model:

- `GGML_CUDA_MMVF_ROWS=1`: a multi-token GEMV for few long F32/F16/BF16 rows, a warp per row, split along the long
  dimension when the rows cannot fill the device (the BF16 hyper-connection and router matrices ran on 10 to 16 blocks).
- `GGML_CUDA_MMVF_FEW_ROWS=1`: MMVF instead of MMF for BF16 matrices of up to 1024 rows and several tokens.
- The recurrent-state rollback slots of `qwen4exp` copy the strided views directly, without a contiguous temporary.

Quality: host mode itself changes nothing measurable (wikitext-2, 6 × 512 tokens, KL divergence against a GPU-only
reference: 0.0230 in host mode, 0.0235 GPU-only at the same ubatch; the CPU quantizes the activations to Q8_K).

Stats: `LLAMA_MOE_HOST_STATS=<n>` logs the miss rate every n requests; at exit the host logs its request count, miss
rate, the time from a request to its answer and from an answer to the next request. `LLAMA_MOE_HOST_TRACE=<file>`
records the selections for offline replay of cache policies.

## Cache policy on the service thread

Commit `a29262c1e`. In host mode the cache no longer admits misses as they happen: it follows selection counts the host
gathers (decayed per plan). After the last layer of a graph, the service thread plans the swaps of the next graph:
admissions into free slots, then replacements of cached experts by uncached ones selected more, best gain first.
While the device waits for the answer of a layer, the thread publishes the uploads that are done and evicts the victims
of a few swaps (not of that layer, whose slots the device may be reading); right after the answer it hands their
uploads to an upload worker on a second stream, while the device computes up to the next request and the CPU has
nothing to do. The CPU and the uploads read the same host memory and share its bandwidth, and the uploads of two swaps
outlast the device's work up to the next request: the worker starts each upload while the host serves no request, one
at a time, and one waiting past 2 ms starts anyway (commit `8c02bccc0`, `LLAMA_MOE_UPLOAD_GATE`).

The service thread makes no CUDA call: a graph that waits for an answer may hold the driver while it is being launched,
and a call from the thread that must answer would deadlock until the wait's timeout.

| Variable | Default | |
|---|---|---|
| `LLAMA_MOE_HOST_SWAPS` | 96 | replacements planned per graph |
| `LLAMA_MOE_HOST_FILLS` | 192 | admissions into free slots per graph |
| `LLAMA_MOE_HOST_PER_REQUEST` | 2 | swaps started per request |
| `LLAMA_MOE_UPLOAD_GATE` | 1 | an upload starts while the host serves no request; 0: as soon as queued |
| `LLAMA_MOE_HOST_DECAY` | 0.9 | decay of the selection counts per plan |
| `LLAMA_MOE_HOST_HYST` | 1.0 | a victim must be selected this many times less than its successor |
| `LLAMA_MOE_HOST_POLICY=graph` | | the earlier policy: a plan every `LLAMA_MOE_HOST_PERIOD` (4) graphs on the main thread, `LLAMA_MOE_HOST_MAX_SWAPS` (32), `LLAMA_MOE_HOST_MAX_FILLS` (1024), decay 0.97, hysteresis 1.25 |

Replaying recorded routing (`LLAMA_MOE_HOST_TRACE`) showed strong locality: a clairvoyant cache of the same size over a
20-step window hits 96 %, while the decayed-count policy reaches 80–83 %. Adapting faster needs more swaps per step,
whose uploads then delay the graphs' own transfers.

## Tail skip

`LLAMA_MOE_CACHE_TAIL_SKIP=<rank>` (commit `012d9f7ef`, needs zero slots: host mode or `--moe-cache-cpu-share` > 0).
A missing expert that every token of the batch ranks `rank`-th or later among its selected experts (counted from 0) is
neither computed nor uploaded: it reads a zero slot and the host skips it. `LLAMA_MOE_CACHE_TAIL_SKIP_W=<share>` also
requires the skipped expert's routing weight to be below that share of the token's total. With rank 7 (of 10) about
5–6 % of the selected experts are skipped in decoding; the KL divergence rises by 0.02–0.027 and the top-1 agreement
drops by about one point (wikitext-2). This is an approximation and is opt-in.

## Cache phases

`LLAMA_MOE_CACHE_PREFILL_MIB=<core MiB>` (commit `a29262c1e`, host mode; `--moe-cache-mib` is then the full size).

Decoding wants the largest cache; prompt processing wants large ubatches, whose compute buffers and expert staging need
memory. With phases the cache banks live in VMM buffers (`ggml_backend_vmm_buffer_alloc` / `_commit` proc addresses in
ggml-cuda) whose physical memory can be released from their end:

- a batch of more than `LLAMA_MOE_PHASE_TOKENS` tokens (default 256, as in the measured setup) runs in the *large
  phase*: ubatches of `-ub`, the cache shrunk to its first slots (the core), the released memory used by the compute
  buffers and the expert prefetch. The context re-reserves only the prompt graph on a switch.
- smaller batches run in the *small phase*: ubatches of `LLAMA_MOE_PHASE_UBATCH` (default 32, 128 measured) in host
  mode with the whole cache.

When small batches return, the experts the lent slots held are admitted again into them by the upload worker
(commit `8060a31c7`), instead of the policy refilling a third of the cache over tens of graphs; a large batch that comes
before they are all back drops the admissions not started yet, whose experts are admitted after it.
`LLAMA_HOST_COMPUTE_PAGEABLE=1` puts the CPU compute buffers in pageable memory: page-locked CPU compute buffers cost
about 450 ms of `cudaHostAlloc`/`cudaFreeHost` per switch (`GGML_CUDA_HOST_BUFFER_CACHE=1` recycles freed page-locked
buffers instead, but keeps them locked). `GGML_CUDA_VMM_CHUNK_MIB` (default 32) sets the granularity at which a VMM
buffer grows and shrinks (commit `991413bff`).

### The lent memory mapped twice

Releasing the lent slots and mapping them again cost about 175 ms each way inside the driver, whatever the number of
calls, and up to 570 ms when admissions were still in flight. The banks now stay mapped whole: the chunks of the lent
slots are mapped a second time, one after the other in one range per device (`ggml_backend_vmm_alias`), and a lent
buffer type over that range (`ggml_backend_lent_buffer_type_new`, `_set`, `_free`) places the compute buffers and the
staging slots of large batches in it while the cache lends, from the end of the switch to large batches to the start of
the switch back. What those buffers need beyond the range is mapped after it, in address space reserved for that, and
released when they are freed, so the large phase holds no more memory than when the slots were released. A switch then
maps and unmaps nothing: 15 to 35 ms each way on the test machine, most of it the compute buffers' reservation. The
CUDA backend's asynchronous copies accept any CUDA buffer type of the device for this. `LLAMA_MOE_CACHE_ALIAS=0`
releases and maps the lent slots again at each switch.

With a 13000 MiB cache and an 8200 MiB core, an 8192-token ubatch fits at 262K context on the 24 GB card with a few
hundred MB of headroom; a larger core ran out of memory around 112K tokens of context, where the allocator packs a
mid-size graph worse than the full reservation.

## Expert prefetch

`LLAMA_MOE_PREFETCH=<slots>` (commits `0e80359ad` to `ddd0cfe81`, `65dc51115`; `src/llama-context.cpp`).

A large batch reads every expert of every layer. Uploaded on the compute stream when the scheduler copies a layer's
weights, the transfers and the computation alternate. With the prefetch, a worker uploads the host expert tensors in
the order the graph will use them (learned from the `MUL_MAT_ID` nodes of the first large graph), on a second stream,
into `slots` device staging slots, while the earlier tensors compute; the scheduler's copy callback then copies a
staged tensor into place. Experts the cache holds are copied from the cache instead (device to device).

- `LLAMA_MOE_PREFETCH_ROUTED` (default 1): stage only the experts the layer's routing selects, reading the routing at
  the gate and staging up and down with it. 0 stages whole tensors without waiting for the routing, which wins at
  8192-token ubatches where nearly every expert is selected anyway.
- `LLAMA_MOE_PREFETCH_MIN` (default 1024): the smallest batch that uses it.
- A routed staging slot is zeroed before its uploads: MMQ reads past the last row of a used expert into the next one,
  and stale bytes of another quantization type can decode to NaN scales.

At 8192-token ubatches the prompt is upload-bound: a chunk uploads every expert the core lacks (about 41.6 GB in 2.6 to
3 s over PCIe 4.0 x8), so long chunks move the least per token.

## Large batches shared with the host

Commits `d8b10a771`, `70d3d4ba7`; default on in host mode with phases.

While a large ubatch is upload-bound the CPU is idle. The graph of a large batch therefore posts the whole batch to a
mailbox of its own (`ggml_moe_host_post_batch`): per layer, the cache marks the least selected of the experts its core
lacks for the host (`llama_moe_cache::begin_batch`), the device multiplies the others (their ids read as -1, and
`ggml_mul_mat_id_set_negative_ids` makes a `MUL_MAT_ID` leave zero rows for negative ids), the prefetch does not upload
the host's experts, and `ggml_moe_host_wait_batch` adds the host's answer, returned as F16 rows for the flagged tokens
only. The input goes to the host by an asynchronous copy on a stream of its own. On the host each expert's entries are
computed in chunks of 64 tokens over the worker pool, gate and up unpacked once per chunk against Q8_0 inputs.

The share is measured, per layer and per batch. Each layer gives the host the least selected of the experts its core
lacks for as long as each one shortens the device's predicted cycle for the layer: the longer of the uploads left and
the device's own work, which is what it does before it posts (the combine of the layer before, the attention, the
routing: the host cannot overlap it) followed by the longer of its own experts and the host's answer. Where the
device's own work is the longer already, the uploads are hidden and the host takes nothing. What goes into it:

- the host's cost of an item (an expert's pass over its weights for up to 64 of its tokens) and of an entry (a token of
  an expert), in thread seconds, fitted to every item the workers time, per kind of layer (the types of its gate, up
  and down): the layers of a GSQ-RCO model differ, an item of an IQ4_XS layer costing more than one of an IQ2_S
  layer, and an entry of a layer with K-quants, whose rows are not unpacked, more than one of the others;
- the upload rate, from timed events around each staging slot's uploads (a CUDA proc address,
  `ggml_backend_event_new_timed` with `ggml_backend_event_elapsed_ms`);
- the device's time before each post, read from its clock in the batch mailbox: the post and the wait write it, and the
  host reads the times of the previous layer at each request;
- each lacking expert's expected entries, from its selections in the last shared batches (the host counts all the
  entries of each request), or from the profile until they exist. The experts least selected in past batches get more
  in the next one; a ratio learned per source from the entries the host finds corrects that.

A batch the host would save less than 2% of is not shared. A kind of layer the host has not measured yet gives it a few
of its cheapest experts, so that it is: in any batch until the host has measured one kind, in a shared one after that. The share is part of the graph's parameters: a graph built without the posts is
not reused for a shared batch, nor the other way round. `LLAMA_PROFILE_UBATCH=1` logs each decision and its outcome.
`LLAMA_MOE_BATCH_CPU=<share>` fixes the share instead, 0 turns it off; `LLAMA_MOE_BATCH_CPU_MAX` caps it (default 1).

Every block of the batch wait used to poll the mailbox across the bus while the device waited: when the host was late,
those reads cut the uploads beside them from 13 to 5 or 6 GB/s, which made a late host later. One thread polls now, and
the blocks that add the answer read its outcome from device memory.

Fixes this needed elsewhere: `mm_ids_helper` counted negative ids as a lower expert, and the MMQ forward and inverse
maps had to handle skipped slots. The first version lost the host experts of the last layer, which is cropped to the
output rows and took no host part while the prefetch had skipped its host experts; every cached layer of a shared batch
now posts. After that fix the KL divergence of shares 0.4 and 1.0 is 0.016–0.017, the same as changing the ubatch size.

## Cache profile

`LLAMA_MOE_CACHE_PROFILE=<file>` (commit `8060a31c7`, host mode).

The cache used to start empty: 87 % misses on the first request, about 70 graphs to warm up. The file holds the
selection counts of past sessions (`u32` magic `LMCP`, `u32` version 1, `i32` n_layer, `i32` n_expert, then `f32`
counts, bounded so that recent sessions keep their weight). At start the experts it counts most fill the slots: the core
at once, the slots lent to large batches when small batches first come. At exit the session's counts are added. A
profile built from unrelated prompts lifted a code task's decode from 59 to 69 tok/s and cut the step from 38–40 to
33.5 ms.

## Asynchronous admission (without host mode)

`LLAMA_MOE_CACHE_ASYNC=1` (commit `42207ffc9`, needs `--moe-cache-cpu-share` > 0, not used in host mode): the share of
the misses that would be uploaded is admitted in the background instead, on a second stream, into slots that stay
pending until their upload is done (`LLAMA_MOE_CACHE_ASYNC_MAX` experts in flight, default 96). Superseded by host mode
for this model; kept for the plain cache path.

## Exclusive core

`LLAMA_MOE_CACHE_EXCLUSIVE=<MiB>` with `GGML_CUDA_HOST_MEMFD=1` (commit `170656bca`; host mode, phases and the
service policy; needs the cache profile).

By default every expert stays page-locked in RAM, including the 7.6 GiB the cache's core also holds: the server left
5.4–5.5 GiB of 62.7 GiB available while serving, against 13.8 GiB or more for Strata, whose low-RAM mode keeps in RAM
only the experts the GPU lacks.

With the exclusive core, the experts held by the first slots of each group (a share of the core, up to the given size)
have no host copy:

- **Host copies move.** Each expert's host address comes from a table (`hptr`, one pointer per expert tensor and expert;
  `hloc`, the host slot of each expert or -1). Host mode, the uploads, the expert prefetch and the scheduler's expert
  copies all read it (`llama_moe_cache::host_data`, `llama_moe_host::layer::ptrs`). A host slot is any expert position
  of a layer of the same group, since they all have the same size.
- **At start**, once the profile filled the slots, each layer is compacted: the experts that keep a host copy and sit in
  the last positions move into the positions of the layer's exclusive experts in front of them. The positions left at the
  end of each bank are given back once (`ggml_backend_host_release` punches them out of the memfd), except a few spare
  host slots per group (`LLAMA_MOE_CACHE_EXCLUSIVE_SPARES`, default 16). For this model: 4,121 experts, 7.64 GiB given
  back in about 4 s.
- **A swap whose victim holds an exclusive slot exchanges**: the planner first downloads the victim into a spare host slot
  on a stream of its own (staged); the swap starts only when that download is done, the victim then evicted with that host
  copy. The host slot of the expert admitted in its place becomes a spare from the next request (the CPU work of a request
  is done before its answer, and the next request reads the slot table after it).
- **Downloads go one expert at a time.** All the streams of a device share the copy engine of a direction in order: a
  burst of downloads queued on one stream holds up a small download on another (a 57 MB burst delayed a 4 KB read-back by
  4.3 ms), which here are the graphs' own read-backs and the drafter's input. Paced one expert at a time, the read-back
  waits 30–80 µs and the downloads still run at 12–13 GB/s. The worker waits on a blocking-sync event
  (`ggml_backend_synchronize_sleep`): with two CUDA contexts and 24 logical CPUs the default schedule spins, which took a
  core from the host workers.

After start, host memory stays as it is and no registration changes. Measured against the same build without it:
misses equal, decode step cost within run-to-run noise (+0 to 1.5 %), prompt processing 1–2.4 % slower (the compaction
scattered some experts, so large-batch uploads go in smaller pieces), follow-ups equal, lowest MemAvailable while
serving 13.0–13.3 GiB instead of 5.4–5.5. About 48 downloads per decode graph remain, about 3 GB/s of DMA into RAM.
The full 47.7 GiB is still loaded first: the release happens about 20 s after the model is loaded.

`GGML_CUDA_HOST_MEMFD_GUARD=1` makes the given-back pages inaccessible, so a stray read faults instead of bringing
memory back; it caught nothing in any run.

### memfd host buffers

`GGML_CUDA_HOST_MEMFD=1` (Linux): a page-locked host buffer of the CUDA backend is a `memfd` file, mapped shared and
registered with `cudaHostRegister(..., cudaHostRegisterPortable)`, instead of `cudaMallocHost` memory, so that parts of
it can be given back: `ggml_backend_host_release` (proc address) unregisters the registrations the ranges cut, punches
the pages out of the file (`fallocate(FALLOC_FL_PUNCH_HOLE)`) and registers the remainders again. A single copy across
two registrations fails, so host copies of a buffer whose registration was split go in pieces at the registration bounds
(`ggml_cuda_memcpy_host_async`, used by `set_tensor`/`get_tensor` and their async and 2D variants); a copy of given-back
memory aborts. `ggml_backend_host_releasable` tells whether a range lies in such a buffer. DMA from these buffers runs at
the same speed as from `cudaMallocHost` memory (13.1 GB/s in 0.7 MB pieces, 13.4 GB/s in large ones, PCIe 4.0 x8).

### What did not work

- Giving back each expert's host copy and refilling it on eviction (`pwrite` from a bounce buffer, then
  `cudaHostRegister` of the piece): every expert slice needs a registration of its own, and 73,700 registrations cost about
  2 GiB of unreclaimable kernel slab (about 16 KiB fixed plus 70 bytes per page each) and 85 µs each to create, which ate
  most of the RAM saved. Limiting the refills with a long-window "tier" of experts that never leave raised the misses from
  19 to 26 %.
- Letting an evicted expert take back the host slot it left, when that spare was not reused yet, saved 0.6 % of the
  downloads with 16 spares per group and 5 % with 64.
- Transparent huge pages on the 47 GiB of host experts: the synchronous compaction they trigger counted as memory pressure
  long enough for systemd-oomd to kill the whole login session the server ran in.
