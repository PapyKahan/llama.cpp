# Before upstreaming

No pull request is planned yet. This file lists the work to do first, then a possible split for when it is time, and
what was tried and dropped so that nobody repeats it.

## Work to do first

### Rebase and dependencies

- The branch merges upstream `master` (last on 2026-10-28, `39febd2ad`), so the commit hashes below stay valid. Upstream
  now has #30640, #30965 and #31243 (the k-pool selection built over the cache cells, whatever the kq_mask shape): the
  branch builds its selection the same way and adds the query chunks of #30815's earlier version. #30815 (open) has
  since moved the chunks into code shared with glm5-next: resolve against it once it lands.
- The MoE work builds on #30966 (`--moe-cache-cpu-share`, open): it must land first, or the offload work must be proposed
  together with it in agreement with its author. The fix in `5682d1c2b` (the CPU `MUL_MAT_ID` grouping must skip
  negative ids, or `--moe-cache-cpu-share` crashes) belongs in #30966's review. #30966 was rebased and extended on
  2026-10-26 (the scheduler copies the CPU part's results, a backend sync for the asynchronous transfer queue); the
  branch carries the 2026-10-22 version, whose CPU part host mode bypasses in the measured setup.
- #26631 (merged) makes a -1 id in `MUL_MAT_ID` a skipped slot with a zero result row on every backend but CUDA
  (`[TAG_MUL_MAT_ID_SKIP]`). The branch's CUDA path skips negative ids for a result marked by
  `ggml_mul_mat_id_set_negative_ids`, after the fused gate/up/GLU kernels too: made unconditional, it would cover CUDA
  and the mark could go.
- Upstream's fused gate/up/GLU MMQ (#31188, #29948) runs through the branch's tile map and counting sort like the
  unfused kernel.
- #30448 (AVX2 IQ3_S/IQ3_XXS dot products) is no longer on GitHub (404 on 2026-10-28); the branch keeps its two commits.
- #30380 (trimmed draft vocabulary) was closed upstream: talk to its author before building on it.

### Switches

- Turn the environment variables that stay into parameters (`llama_context_params`, `common_params`, command-line
  options with documented defaults); [reference.md](reference.md) lists them all.
- Drop the switches whose experiment ended: `LLAMA_SPEC_PAD` (superseded by graph slots), `LLAMA_MOE_HOST_POLICY=graph`,
  `LLAMA_MOE_CACHE_ASYNC` (superseded by host mode), `GGML_CUDA_HOST_BUFFER_CACHE`, and decide which profiling switches
  (`LLAMA_PROFILE_UBATCH`, `GGML_SCHED_PROFILE`, `LLAMA_KPOOL_CHECK`) are worth keeping.
- Make the clear wins the default where they are safe (MMQ tile map and counting sort already are).

### Portability

- Host mode needs mapped host memory and device-side waiting on it: CUDA only today (proc addresses). Decide what other
  backends do (fall back to the plain cache path) and check HIP.
- The cache phases need VMM buffers (CUDA). The exclusive core needs memfd and `fallocate` (Linux); elsewhere it must
  stay off cleanly.
- The host kernels (`ggml_cpu_row8_*`) and #30448 are AVX2: add the scalar path at least, and check ARM.
- Host mode supports plain MoE layers (separate gate, up and down, no biases or scales) and a few activation clamps;
  every other layer falls back. Besides `qwen4exp`, the MoE work runs Ornith-1.5-122B (`qwen35moe`: 256 experts, top-8,
  DeltaNet with full attention every fourth layer, a separate MTP drafter); other architectures remain to check.
- Host mode assumes a single device holds the cached layers; multi-GPU expert caches need a decision.

### Tests and validation

- `test-backend-ops` cases for `ggml_lightning_indexer_ext`, `ggml_mul_mat_id_set_negative_ids`, MMQ at many tokens with
  the tile map and the sort, the compact sparse flash attention, and the quantized-to-F16 CPY.
- The host ops (`GGML_OP_MOE_HOST_POST/WAIT`) block on host threads: they need a harness of their own (a host service
  answering in a test).
- Unit tests for the cache bookkeeping: the service policy, the phases, and the exclusive core's invariants (every expert
  either cached in an exclusive slot with no host copy, or with exactly one host copy; spares never referenced).
- Perplexity and KL divergence for every approximation (tail skip) and every path that changes the arithmetic (host mode,
  the host share of large batches), on standard sets and on more than one model.

### Code

- `src/llama-moe-cache.cpp` grew by 2,400 lines: split it (policy, phases, prefetch coupling, host coupling, exclusive
  core) before review.
- Comments and naming to upstream's style; no experiment leftovers.
- Thread safety documented where it rests on protocol rather than locks (the host location table, read by the host
  workers while the service thread moves copies).

### Known issues

- Short follow-up turns are slower than Strata's: a turn of up to 256 tokens runs in the small phase, 128-token batches
  whose misses the host computes, and the server cuts each turn into batches at its checkpoints, each batch reading its
  missed experts again (Flash-Next 1.16 to 1.28 s for +100 tokens against 0.72 s, 1.42 s for +300 against 0.96 s;
  Ornith-1.5-122B about 1.85 / 1.8 s). From 1000 tokens on, the measured host share and the lent memory make the branch
  faster than Strata (see [measurements.md](measurements.md#measured-host-share-and-lent-memory)). A turn in one pass,
  the recurrent state at its checkpoints saved within it, is the next thing to try (Ornith-1.5-122B without the cuts:
  1.43 / 1.64 / 2.04 / 2.72 s for +100 / +300 / +1000 / +3000, see
  [measurements.md](measurements.md#one-pass-per-turn-experiment)).
- The host's row8 kernels have no unpacker for the K-quants: their `vec_dot` is already among the cheapest of the host's
  types, and an unpacker carrying their minimums did not beat it (see
  [measurements.md](measurements.md#iq2_xs-on-the-host)).
- The start loads all 47.7 GiB of experts before the exclusive core releases its share: loading the exclusive experts
  straight into the cache would remove that peak.
- The exclusive core needs the cache profile, so the very first session has none.
- Over a long session, exchanges scatter the host copies, so large-batch uploads go in smaller pieces (at most about
  2.5 % slower, measured for 0.7 MB pieces against large ones).
- Server (upstream, not this branch's changes): with `--kv-unified`, a text batch that finds no room purges the first idle
  slot in index order, which is often the live session, and an image chunk does not purge and retry at all, so a long
  agent session failed on a screenshot while two stale slots held 120K cells. A small server PR: least recently used
  slot first, and make room before an image chunk.

## A possible split, later

Independent of the rest, small:

1. QSA key pool reserved per KV stream (`7ba73dd54`, one line).
2. A backend sampler chain like the one it replaces keeps the scheduler's reservation (`77ad0af04`); every `-bs` user
   pays 30–45 ms per request without it.
3. CUDA MMQ for `MUL_MAT_ID` at many tokens: tile map and counting sort (part of `e2ff90695`).
4. CUDA MMVF for few long rows (part of `e1cd4e00f`).
5. CPY from a contiguous quantized tensor to F16 (part of `e2ff90695`).
6. Server: balanced prompt chunks (part of `acab26f2b`), checkpoint tail as a parameter (`f222e5947`).

`qwen4exp`, after the QSA rebase:

7. QSA chunk scratch freed, K/V converted once per layer (`c44d385f4`, part of `65dc51115`).
8. k-pool layout updated incrementally, tables kept across ubatches, build state in the memory (part of `65dc51115`,
   `b1351e725`, `a93fd2f96`).
9. Lightning indexer on tensor cores and reading pooled keys in place (`0bfe8cb7a`, `d5379bd75`).
10. Compact sparse flash attention and its single-token path (part of `e2ff90695`).
11. N-gram rows: raw gather, threads, `MADV_RANDOM`, read-ahead of drafted tokens (part of `65dc51115`, `27adb1e7d`).
12. MTP layer cropped to its output rows (`b669e4293`).

Speculative decoding, general:

13. Graph slots and `ggml_backend_sched_new_shared` (`e51d84af8`).
14. Cumulative draft stop as a parameter, with the draft trace (`ce345f505`, `027048e39`).
15. Asynchronous catch-up of prompt chunks and the context wait/defer helpers (parts of `acab26f2b`, `8fa30b1ee`).

Expert offload, after #30966, best preceded by a design discussion upstream (graph ops that wait on host threads, mapped
mailboxes, a service thread that must make no CUDA call):

16. Expert prefetch for large batches (`0e80359ad` to `ddd0cfe81`, part of `65dc51115`).
17. Host mode, its ops and the row8 host kernels (`e1cd4e00f`, `2b37cca0f`, `1f42db592`).
18. Service-thread policy, cache phases on VMM buffers, re-admission of lent slots, the large phase's buffers in the
    lent memory (`a29262c1e`, `991413bff`, part of `8060a31c7`, `b83aeb575`).
19. Cache profile (part of `8060a31c7`).
20. Large batches shared with the host, the share measured (`d8b10a771`, `70d3d4ba7`, `3cce2f756`).
21. Tail skip, opt-in (`012d9f7ef`).
22. Exclusive core and memfd host buffers (`170656bca`).

## Tried and dropped

| Idea | Result |
|---|---|
| ik_llama.cpp instead of llama.cpp | slower on this model in every test |
| QSA query chunks without freeing each chunk's scratch | larger compute buffer, not smaller |
| Expert prefetch v2 (cached experts copied by the worker, out-of-order staging) | slower than v1 |
| Four prefetch slots | prefill collapsed to about 800 tok/s |
| Faster cache adaptation (64–128 swaps per window) | the uploads delayed the graphs' own transfers: net loss |
| Small-phase batches of 256 to 1024 tokens for mid-size prompts | no better than the large phase |
| Queued user-input copies with one synchronization per split | decode within noise, prefill −6 % |
| Padded verifications | beaten by graph slots without padding |
| Drafting the next window during the verification | slightly negative overall |
| Folding the catch-up into the first draft step | at most 0.3 ms per step |
| n-gram drafts ahead of the MTP | uncapped windows broke the verification shape; capped, no drafts |
| Five-token verifications (`n_max 4`) | cost more than their extra drafts return |
| Hottest-first re-admission after large batches | no visible change |
| Exclusive core by giving back and refilling each expert's host copy | 2 GiB of kernel memory for 73,700 registrations; a long-window tier raised misses from 19 to 26 % |
| Evicted experts taking back their old host slot | 0.6 % of downloads saved (5 % with 64 spares) |
| Transparent huge pages on the host experts | the compaction got the whole login session killed by systemd-oomd |
| Lower-rank draft vocabulary scorer, fewer dynamic rows | not adopted: needs a proper A/B (single runs: 0.80–0.81 ms per draft step instead of 0.89, rows 0 loses 1–3 % accepted drafts on code) |
