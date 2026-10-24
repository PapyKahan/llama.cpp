# Speculative decoding with the MTP layer

Qwen3.8-Flash-Next ships a multi-token-prediction (MTP) layer: one full-attention QSA layer that, from the target's last
hidden state and the embedding of the token just sampled, predicts the next token. The branch runs it as a draft model
(`--spec-type draft-mtp`, Unsloth's `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`) on the second GPU (`--spec-draft-device CUDA1
--spec-draft-ngl all`) with a q8_0 KV cache of its own. A step is: the drafter catches up on the tokens the target
accepted, drafts up to `--spec-draft-n-max` tokens one by one, then the target verifies the draft in one batch of up to
n_max + 1 tokens (host mode, see [moe-offload.md](moe-offload.md#host-mode)).

On the test machine a step at 84K context costs about 25.8 ms of target verification and 4.6 ms of drafter time
(catch-up 0.9 ms, then about 1.2 ms per draft step); the step cost at short context is about 25 ms plus 3.3 ms per
draft, so a draft pays only when its chance of being accepted, counting the drafts before it, is above about one third.

## Trimmed draft vocabulary

Upstream PR #30380 (Pranesh Gonegandla, closed upstream without merge; merged into this branch). The draft head's output
projection over the 248K-token vocabulary dominates a draft step. `--spec-draft-vocab` replaces it with a reduced head:
the lowest 32K token ids as a dense head, plus *dynamic rows*: a low-rank scorer (`--spec-draft-vocab-dynamic-rank`,
default 512) over the other ids picks `--spec-draft-vocab-dynamic-rows` candidates whose rows are gathered and scored
exactly. If the reduced head cannot be set up, the full head is used. A test is in `tests/test-draft-vocab.cpp`.

On the 3070 Ti a draft step costs about 0.92 ms with it (dense part 124 µs, scorer 209 µs, gathered rows 72 µs, top-k).
Dynamic-row tokens are 3 % of the accepted drafts on prose but 12 % on 84K-token code, so they stay.

## Cumulative draft stop

`LLAMA_SPEC_P_CUM=1` (commit `ce345f505`). With `--spec-draft-p-min p`, drafting stops before the token that would
bring the product of the draft's token probabilities below `p`, instead of the first token whose own probability is
below `p`; one less certain token after very certain ones no longer ends the draft, while several middling ones do.

Measured on the code task (2.3K / 84K / 210K tokens of context, 400 sampled tokens, 2–3 runs each), with graph slots:

| Draft rule | 2.3K | 84K | 210K (tok/s) |
|---|---|---|---|
| always 3 drafts (`p_min 0`) | 68.6 | 70.4 | 78.8 |
| cumulative, `p_min 0.3` | 83.9 | 78.3 | 84.6 |
| cumulative, `p_min 0.2` | 81.4 | 76.0 | 81.3 |
| cumulative, `p_min 0.4` | 79.0 | 76.5 | 83.9 |
| cumulative 0.3, `n_max 4` | 79.7 | 77.8 | 75.6 |

Runs with three fixed drafts got fewer accepted drafts per step than cumulative-0.3 runs on the same prompts, not only
fewer per draft: low-probability drafts route to atypical experts (more misses, more tail skips), which seems to degrade
the target's agreement with the head beyond their own rejection. A replay of recorded drafts against a step cost
(`LLAMA_SPEC_P_TRACE=<file>` appends, per verified draft, its size, the accepted count and each token's probability and
id) had predicted only 1–2 % for any stop rule, because it replays the same text for every policy.

## Graph slots

`LLAMA_GRAPH_SLOTS=<n>` (commits `e51d84af8`, `a93fd2f96`; `src/llama-context.cpp`, `ggml/src/ggml-alloc.c`,
`ggml/src/ggml-backend.cpp`).

A context reuses only the graph it built last, so shapes that alternate rebuild each other: the drafter's catch-up and
its draft steps every step, the target's verification after each prompt, and every window size of an adaptive draft.
A rebuild costs about 0.2 ms plus a slower first launch of the new CUDA graph. With slots, the graphs of decoding ubatches
of up to `LLAMA_GRAPH_SLOT_TOKENS` tokens (default 16) are kept in `n` slots, each with a scheduler of its own that
allocates from the compute buffers of the main one: `ggml_backend_sched_new_shared(src, graph_size)` over a `gallocr`
that borrows its parent's buffers. A slot keeps its split, its allocation and so its CUDA graphs. It is dropped when the
main buffers move (an epoch counted by the parent allocator, `ggml_backend_sched_shared_valid`) or when the scheduler
is created again; a graph the main buffers cannot hold goes the regular way. Each target slot costs about 16 MB of
graph metadata.

Also: the per-cell state that picks what a ubatch re-pools (one entry per cache cell) was allocated and cleared for
every decode, target and drafter alike, because a memory context lives for one decode; the memory now keeps it.

## Padded verifications

`LLAMA_SPEC_PAD=1` (commit `2fe4ae266`; not used in the measured setup). Before graph slots, a draft that stopped early
shortened the verification batch, which rebuilt the target graph and defeated CUDA graphs. With padding the server
fills each generating slot's entries up to the largest draft plus one (bounded by the recurrent rollback depth), also
when nothing was drafted. Pads are outputs at the positions after the slot's entries, kept out of the prompt and the
verification and removed with the rejected drafts. In host mode a pad reads the zero slots and reaches the host with id
-1 (`ggml_moe_host_post_set_pad`, input `inp_pad`), which it neither computes nor counts. Batch API:
`llama_batch_ext_set_pad`, `common_batch::token::pad`, `llama_ubatch::pad`. At `p_min 0.5` padding beat no padding by
6 % at short context and 15 % at 84K; graph slots without padding beat both.

## Drafter catch-up

Commits `acab26f2b`, `8fa30b1ee`, `b669e4293`.

- The catch-up decodes in pieces of the draft ubatch (`--spec-draft-n-ubatch`, 512 measured), so a prompt chunk is not
  copied whole several times, and keeps only the last row of a prompt chunk.
- `LLAMA_SPEC_ASYNC_PROCESS=1`: the catch-up of a prompt chunk (more than 64 entries) runs in the background while the
  target processes the next chunk. Edits of the draft context's memory are queued while it runs and its state is read
  after it (`common_context_set_wait`, `common_context_wait`/`defer`). The catch-up of a verification batch stays on the
  calling thread: the next draft waits for it right away, so a thread only added its start-up to the step.
- The MTP layer's output is masked to the output rows: past the attention, the rows are cropped to the output rows
  before the FFN, so a catch-up, which outputs nothing and only needs the keys it writes, skips the experts and the
  hyper-connection mix; a draft step keeps its single row.

## Sampling on the GPU

`-bs` (backend sampling, upstream) runs the sampler chain on the GPU after the output layer, so a verification reads
back a few candidates per token instead of the whole vocabulary (the CPU sampling of four verified positions cost about
1.4 ms per step). Each request brings its own sampler chain, and setting it asked for a new scheduler reservation,
30–45 ms on this model plus the compute buffer reallocations around the cache phases, at every request. Commit
`77ad0af04` keeps the chain's backend part and its graph node count per sequence, so the same chain keeps the
reservation; the scheduler still reserves by itself if a graph outgrows it. A request whose samplers the GPU lacks
(DRY, a grammar) samples on the CPU as before. `--spec-draft-backend-sampling` does the same for the drafter.

## N-gram embedding rows read ahead

`LLAMA_PLE_PREFETCH=1` (commit `27adb1e7d`, `qwen4exp` only).

The model's per-layer embedding (PLE) table is 26.8 GiB (320M rows of 90 bytes, IQ4_NL, in the second GGUF split),
read through the page cache, 16 hashed rows per token. It cannot stay cached next to the experts, so nearly every row of
a verification was a read from the SSD on the critical path (0.5 ms per verification). The drafter now hands each token
to `llama_prefetch_token()` (`src/llama-ext.h`) as soon as it is known, the sampled token when drafting starts and each
draft as it is drafted; a thread owned by the model reads its rows, and the verification asks the kernel only for the
rows not read yet. The gather then costs 0.043 ms per verification; the step is 1.0 % faster at short context and 1.4 %
at 84K.

## What did not work

- Drafting the next window on the second GPU while the target verifies: a hit (window wholly accepted and the drafter's
  extra guess right) is 36–60 % of steps, but drafts made ahead are deeper in the chain and fewer of them are accepted;
  net slightly negative overall.
- Folding the catch-up into the first draft step: only its launch (about 0.3 ms) is on the critical path.
- n-gram drafting (`ngram-simple`) ahead of the MTP: its drafts are not capped by `--spec-draft-n-max`, so 10–23-token
  windows broke the verification shape; capped at three tokens it found no drafts.
- A fifth verification slot (`n_max 4`): a 5-token verification costs more than its extra drafts return here.
