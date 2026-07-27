# Row-sparse `wte` gradients: de-residenting the embedding table (encoder half)

*2026-07-27 — README roadmap item 1, encoder half. Team L owns the companion
vocab-chunked LM head; this document covers only the encoder side.*

This is written to be readable without prior exposure to distributed training.
If you know Python and roughly what a transformer is, you have enough.

---

## 1. The vocabulary of the problem

**Parameters** are the weights you train. **Gradients** are the per-parameter
derivatives produced by backpropagation. **Optimizer state** is the extra
bookkeeping an optimizer keeps per parameter — AdamW keeps two (a running mean
and a running variance), so optimizer state is typically *twice* the size of
the parameters. **Activations** are the intermediate values saved during the
forward pass so the backward pass can use them. These four are separate memory
costs and they scale differently; conflating them is the single most common
source of confusion here.

**Data parallelism** (DDP) is the simplest way to use N GPUs: put a complete
copy of the model on each, feed each GPU a *different* slice of the batch, then
average the gradients across GPUs before the optimizer step. Each GPU is a
**rank**; N is the **world size**. Averaging gradients requires the GPUs to talk
to each other, and those group operations are **collectives**:

- **all-reduce** — every rank contributes a vector, every rank receives the sum.
- **reduce-scatter** — every rank contributes a vector, but each rank receives
  only *its* slice of the sum. N times less data received than all-reduce.
- **all-gather** — the inverse: each rank holds one slice, and afterwards every
  rank holds the whole thing.

Plain DDP wastes memory: all N ranks store identical copies of the parameters,
gradients, and optimizer state. **ZeRO** ("Zero Redundancy Optimizer",
[Rajbhandari et al. 2019](https://arxiv.org/abs/1910.02054)) removes that
duplication by **sharding** — splitting a tensor across ranks so each rank
stores only 1/N of it. Its Section 5 defines three stages, which "correspond to
the partitioning of optimizer states, gradients, and parameters":

| Stage | What gets sharded | What it buys |
|---|---|---|
| 1 | optimizer state | usually the biggest single win; AdamW state is 2x the parameters |
| 2 | + gradients | gradients stop being fully resident on every rank |
| 3 | + parameters | each rank stores 1/N of the weights and *gathers* the pieces it needs, just in time |

This trainer implements all three. Stages 2 and 3 are where this work lives,
because those are the stages where a gradient is produced into a small reusable
**pool** and immediately reduce-scattered away, rather than accumulated into a
full-size gradient buffer.

## 2. Why `wte` is the floor

`wte` is the **token embedding table**: one row of `C` numbers per vocabulary
entry, so that token id `i` is represented by row `i`. GPT-2's vocabulary is
50257 entries, **padded** to 50304 (`V_p`) purely so the dimension divides
nicely for GPU kernels. With `C = 768` channels in fp32:

```
50304 x 768 x 4 bytes = 147 MiB
```

That single tensor is the largest in the 124M-parameter model, and it is why
stages 2 and 3 could not get below a ~150 MiB floor: the gradient pool had to
be big enough to hold a full `wte` gradient.

`wte` is also **tied** — the same matrix serves twice. It is the input
embedding *and*, transposed, the output projection that turns final hidden
states into per-token scores ("the LM head"). Tying is standard practice
([Press & Wolf 2016](https://arxiv.org/abs/1608.05859); [Inan et al.
2016](https://arxiv.org/abs/1611.01462), which ties them "greatly reducing the
number of trainable variables"). It means `wte`'s gradient is a **sum of two
contributions**, and the crucial observation for this whole campaign is that
those two contributions have completely different structure.

### The asymmetry that splits the work in two

**The encoder contribution is naturally sparse.** The forward pass reads row
`i` only if token `i` appears in the batch. So in the backward pass, only those
rows receive any gradient at all; every other row's gradient is exactly zero.
At the benchmark shape `B=4, T=64`, one micro-step has 256 token positions, so
it can touch **at most 256 of 50304 rows** — 0.5%. The other 99.5% of that
147 MiB is provably zero and yet was being allocated, zeroed, and communicated
every single step.

**The LM-head contribution is dense.** The output projection multiplies against
*every* row to score *every* possible next token, so every row gets gradient.
No sparsity to exploit.

Same tensor, two contributions, two completely different properties — which is
why this roadmap item needed two different techniques and two teams. Team L
attacks the dense side by *tiling the vocabulary*, working on a slice of rows at
a time. This document attacks the sparse side by *only touching rows the batch
actually named*.

Stated more generally: **a tied embedding has two consumers with fundamentally
different access patterns.** The LM head reads it *densely*, as the `(V_p, C)`
weight matrix of a GEMM — which rows it needs is fixed and known before
training starts. The encoder reads it *sparsely*, as a row-indexed lookup —
which rows it needs is a property of the tokens that happen to arrive this
step. A technique that fixes one does not automatically fix the other. In
particular, **no static vocabulary partition helps the encoder at all**: you
cannot decide ahead of time which rows a future batch will name. That is the
clearest single reason why de-residenting a *tied* embedding is harder than
de-residenting either use on its own, and why this campaign needed two
independent workstreams rather than one clever change.

### Prior art, honestly

Neither technique is novel research.

Vocabulary splitting has direct prior art in [Megatron-LM
(arXiv:1909.08053)](https://arxiv.org/abs/1909.08053), whose Section 3 states:
"We parallelize the input embedding weight matrix E_{H×v} along the vocabulary
dimension E=[E_1,E_2] (column-wise)," and which fuses the output GEMM with the
cross-entropy loss because "communicating scalar losses instead of logits is a
huge reduction in communication." (Those quotes are in the full text, not the
abstract; see the [ar5iv
rendering](https://ar5iv.labs.arxiv.org/html/1909.08053). The implementation
lives in
[`megatron/core/tensor_parallel/cross_entropy.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/cross_entropy.py).)
[Liger Kernel (arXiv:2410.10989)](https://arxiv.org/abs/2410.10989) §3.2 gives
the memory framing at modern scale — "a 256k vocabulary size will result in a
16.8 GB logit tensor of bfloat16, causing a huge spike in the peak memory
usage" — and chunks the hidden states to avoid it. ([Cut Your Losses,
arXiv:2411.09009](https://arxiv.org/abs/2411.09009), often mentioned alongside,
uses a different mechanism — on-the-fly log-sum-exp rather than chunking — so
don't file it as a chunking example.)

The *sparse* side has no Megatron analogue, and the reason is instructive.
Megatron partitions the vocabulary **statically**: rank `p` owns a fixed slice
of rows, decided before training starts. This work partitions **by batch
content**: which rows matter is a property of the data that arrives this step.
Static partitioning is rank-invariant by construction. Content-derived
partitioning is not — and that turns out to be the hard part.

The contribution here is applying a known idea inside a ZeRO stage-2/3 sharded
trainer written in Mojo, and measuring what it actually buys.

## 3. The hazard that shaped the design

The two collectives this change relies on,
`ZeroContext.reducescatter_buckets` and `ZeroContext.allgather_ranges`, take
explicit lists of `(destination, pool offset, length)` triples describing which
sub-ranges of the flat parameter vector to move. Both document that **every
rank must pass identical lists**.

That requirement is fatal for a content-derived row set, because ranks are
data-parallel and therefore see *different tokens*. In this trainer each rank's
`DataLoader` is even seeded differently (`MT19937(42 + rank)`), so the token
sets genuinely diverge.

The failure mode is worse than it first appears. Reading
`_reducescatter_buckets_gpu`, each rank pulls from its **peers'** pools at
offsets taken from its **own** list. So if rank 0 and rank 1 disagree, rank 0
reads rank 1's pool at an offset that on rank 1 holds a *different token's*
gradient row. And it does not hang: both barriers sit *outside* the per-range
loop, so barrier counts still match no matter how long the lists are. The
result is **silently wrong gradients** — no crash, no deadlock, and any test
that only checks "did it run" stays green.

This was a documented invariant with **zero enforcement**. Two fixes went in:

1. **Make the row set rank-invariant.** Each rank builds a bitmap of the token
   ids it uses; a bitwise-OR across ranks produces the **union**, and every
   range list is derived from that union by a deterministic pass. All ranks
   therefore compute byte-identical lists without further communication. The
   union itself is cheap and needs no new collective machinery: ranks in this
   trainer are *threads in one process* sharing a `CpuCoordinator`, so
   `allreduce_or_host` ORs peers' host buffers directly — for GPT-2 the bitmap
   is `ceil(50257/32) x 4 B = 6.3 KB` per rank, and no device transfer is
   involved.

2. **Enforce the invariant.** `ZeroContext.assert_ranges_agree` hashes the
   range lists, sums the hashes across ranks with the existing
   `allreduce_scalar`, and checks the total equals `N x` the local hash — which
   holds iff every rank hashed the same lists. It is called at the top of both
   collectives. Cost is two host barriers; benefit is converting a
   silent-corruption class into a loud error for every future caller. (The hash
   is kept to 40 bits so that up to 8 copies sum *exactly* in a `Float64`,
   making the comparison exact rather than approximate.)

Fix 2 is arguably the more durable result. An invariant that is documented but
unchecked is a bug waiting for its author.

## 4. What the collectives actually cost

Before committing to a design, the per-range cost was measured directly on two
RTX PRO 6000 GPUs — because "many small ranges" is exactly the pattern a sparse
row set produces, and the collectives issue **one driver-staged `enqueue_copy`
per range per peer**.

Total bytes held fixed; only the number of ranges `K` varies:

| Total rows | K (ranges) | `allgather_ranges` | `reducescatter_buckets` |
|---:|---:|---:|---:|
| 512 | 1 | 0.068 ms | 0.080 ms |
| 512 | 64 | 0.956 ms | 1.153 ms |
| 512 | 512 | 7.109 ms | 7.300 ms |
| 8192 | 1 | 0.607 ms | 0.659 ms |
| 8192 | 512 | 7.943 ms | 7.387 ms |
| 8192 | 8192 | 116.4 ms | 138.6 ms |
| 32768 | 1 | 1.993 ms | 2.276 ms |
| 32768 | 2048 | 30.8 ms | 30.2 ms |
| 32768 | 32768 | 552.0 ms | 518.9 ms |

Reference point: gathering the **entire** 147 MiB `wte` as a single range costs
**2.561 ms** (60.3 GB/s).

Read the extremes. Moving 32768 rows as one range costs 1.99 ms; moving the
*same rows* one-per-range costs 552 ms — a **250x** penalty for identical
bytes. The marginal cost of an extra range is a flat **~15 µs**, essentially
independent of size, which is the fixed cost of a driver-staged cross-device
copy.

That single number drives two design decisions:

- **15 µs buys ~900 KB of transfer** at 60 GB/s, which is ~290 rows of 768 fp32
  channels. So if two runs of needed rows are separated by fewer than ~290
  unneeded rows, it is *cheaper to gather the unneeded rows too* than to pay
  for a second copy. Hence `ENC_MERGE_GAP_ROWS = 256` — gap-coalescing is not
  an optimization here, it is a correctness-of-performance requirement.
- **The naive design loses.** The obvious "one range per token row" approach at
  the benchmark shape costs 7.1 ms against a 2.56 ms dense gather. It would
  have been a 2.8x *slowdown* shipped as an optimization.

## 5. The negative result: sparsity does not survive scale

The plan of record sized the gradient pool at `min(B*T, V_p) * C`, expecting
0.75 MiB at `B=4,T=64` and ~96 MiB at production shapes.

That bound is wrong in a way that matters, and the reason is the union from
§3. Sizing must cover the worst case, and the worst case is the union across
*all* ranks:

```
worst-case distinct rows = min(WORLD_SIZE * B * T, V)
```

Work it through for GPT-2's `V = 50257`:

| Shape | Global tokens/micro-step | Worst-case rows | Bound vs. dense |
|---|---:|---:|---|
| `B=4, T=64`, world 2 | 512 | 512 | 1.5 MiB — 99% saving |
| `B=8, T=1024`, world 2 | 16384 | 16384 | 48 MiB — 67% saving |
| `B=32, T=1024`, world 2 | 65536 | **50257 (saturated)** | **no saving at all** |
| `B=32, T=1024`, world 8 | 262144 | **50257 (saturated)** | **no saving at all** |

Past roughly 50k tokens per global micro-step the union covers the whole
vocabulary and content-sparsity **degenerates into the static case**. The
sparsity win evaporates exactly where large-scale training lives — and note it
is the *union* that kills it: a single rank at `B=32,T=1024` touches at most
32768 rows, but four ranks between them touch nearly everything.

This is worth stating plainly because it retroactively explains Megatron's
design choice. Megatron partitions the vocabulary statically and gets a
rank-invariant, batch-size-independent bound. Content-derived sparsity buys more
at small scale and *nothing* at large scale, while costing a cross-rank union
to stay correct. That is a real engineering trade-off, not an oversight.

### The fix: chunk by rows, not by batch

Sparsity is not the only lever. The pool does not have to hold every touched
row at once — it only has to hold *enough rows to make the collective
efficient*. So the encoder backward walks its rows in **chunks** of a compile
-time constant, reduce-scattering each chunk before reusing the pool:

```
pool = ENC_ROW_CHUNK_ROWS * C + wpe   (8192 x 768 x 4 B = 24 MiB, + wpe)
```

Now the bound depends on **neither vocabulary size nor batch size**. At
`B=32,T=1024,world=8`, where pure sparsity saves nothing, chunking still takes
the encoder's gradient bucket from 147 MiB to 24 MiB. 8192 was chosen so the
encoder bucket lands *below* the per-transformer-layer bucket (~28 MiB) and
therefore stops being the binding constraint at all — pushing it lower would
shrink a term that no longer determines the floor, while adding collectives.

The two mechanisms compose: at small shapes the row map keeps the row count
tiny and there is one chunk; at large shapes the row count saturates and
chunking carries the saving. Neither alone is sufficient.

## 6. What was built

**`llmm/encoder.mojo`** — the row map, all host-side:
- `build_token_bitmap` — one bit per token id present in this rank's batch.
- `build_row_runs` — turns the union bitmap into ascending, disjoint,
  gap-coalesced runs of global rows, and fills `row_of_token` mapping a global
  row to its compact index. Deterministic and a pure function of the bitmap,
  which is what makes all ranks agree. Merging is bounded by the caller's
  capacity so the compact buffers can never overflow.
- `build_wte_buckets` gained a row map. The change is one line of substance:
  a bucket's destination is now the *compact* row rather than the global token
  id. Buckets stay in ascending row order and still partition disjoint
  `(row, channel-group)` cells, so the backward reduction remains a plain
  load-add-store rather than a hardware atomic — **bit-reproducibility is
  preserved**, which was a hard requirement.
- `encoder_bwd` gained `include_wpe`. The position-embedding gradient does not
  depend on the row chunk, so recomputing it per chunk would accumulate it
  `num_chunks` times. Chunk 0 takes it; the rest skip it.

**`llmm/zero.mojo`** — `allreduce_or_host` (host-side bitwise-OR across ranks)
and `assert_ranges_agree` (the guard), wired into both bucketed collectives.

**`train_gpt2.mojo`** — `_build_encoder_row_map` per micro-step, and
`_encoder_backward_row_sparse` implementing the chunk loop. Bucket 4 dispatches
to it under ZeRO-2/3; ZeRO-0/1 keep the dense path unchanged, since there the
gradient is a real full buffer rather than a pool.

One implementation note: the backward kernels address rows absolutely, so each
chunk passes a **base pointer biased down by the chunk's first row**. Every row
the kernels touch is `>= row_lo`, so every address formed lands inside the
pool. This mirrors the existing negative rebase offsets in `_z3_stream_layer`
— it is address arithmetic, and "simplifying" it to a window-base pointer
requires changing the kernels' indexing too.

## 7. Scope not taken: the indexed forward gather

The plan also called for a compact **forward** gather — under ZeRO-3, gathering
only the batch's rows into a small `[n_rows, C]` table and feeding the encoder
remapped indices. It was implemented and then removed before commit, for two
honest reasons:

1. **It cannot use chunking.** The encoder kernel reads arbitrary rows in one
   pass, so every touched row must be resident *simultaneously*. The escape
   hatch that rescues the backward is unavailable, so the forward window is
   stuck with the `min(WORLD_SIZE*B*T, V)` bound from §5 — the one that
   saturates.
2. **It cannot pay off yet, and cannot be validated end-to-end yet.** `wte` is
   tied, so the LM head still needs the *dense* table in the ZeRO-3 window
   until Team L's vocab-chunked head lands. Until then a compact encoder gather
   is pure overhead — and by §4, at the benchmark shape it is *measurably*
   overhead: ~7.1 ms of extra gather against a 2.56 ms dense one.

Shipping a default-off path that could not be exercised by the gates would have
been worse than not shipping it. The design and its measured cost are recorded
here instead; the follow-up is unblocked the moment the dense window is gone.

## 8. Honest summary

- **Encoder gradient bucket: 147 MiB + wpe → 24 MiB + wpe**, independent of
  vocabulary and batch size. This is real and it is the deliverable.
- **The realized pool floor does not drop yet.** `_grad_pool_elems` takes a
  `max` over buckets, and the LM-head bucket still contributes a dense
  `wte + wpe` term. That term is Team L's and was deliberately left untouched;
  the encoder term was added as a separate `max` entry so the two shrink
  independently and the merge stays clean. **The floor drops when both land**,
  not when one does.
- **Content-sparsity is a small-shape win only.** Documented in §5 with
  numbers. The chunking is what makes the change worth having at scale.
- **The collectives are now guarded** against a silent-corruption class that
  had existed, documented and unenforced, since they were written.

## 9. Verification status

Stated precisely, because "the gates passed" is a claim that should be
auditable.

**Verified green** at commit `7e441ae`:

- `make format` — clean, no diff.
- `make lint` — passed (it is also the pre-commit hook; never bypassed).
- `make check` — passed.
- `tests/test_zero_equivalence.mojo` — 6/6, stages 1/2/3 at world sizes 2 and 8.
- `tests/test_zero.mojo` — 13/13. `test_multi_gpu_collectives` was confirmed to
  have **actually executed** rather than silently skipping: it no-ops and still
  reports PASS on fewer than two visible GPUs, so it was re-run with a single
  GPU exported and dropped to a near-zero duration versus a substantial one
  with two — an on/off discriminator that does not depend on trusting any
  absolute timing.
- `make test`'s `test-mojo` phase — all 17 `tests/test_*.mojo` files exit 0.
- `tests/test_encoder_equivalence.py` — 6/6 (forward and backward, fp32
  small/large, bf16 small), run in the `cuda` pixi env. This is the gate that
  most directly exercises this change: `build_wte_buckets` gained two
  parameters and `encoder_bwd` gained `include_wpe`, so a signature or
  behavioural regression surfaces here.
- `make build WORLD_SIZE=2` — compiles clean; `build/train_gpt2` confirmed
  newer than `encoder.mojo`, `zero.mojo` and `train_gpt2.mojo` at this commit.

**Not verified.** The `test-python-cuda` phase is incomplete: it was aborted
partway under a box-wide hold on full suites (four concurrent suites had driven
load average to 65). Roughly 46 of 248 pytest items ran, all passing, but the
remainder — including several equivalence tests — did not run.

**Timing caveat.** Every wall-clock number gathered during that window is
contended and is not quoted as a measurement anywhere in this document except
the collective micro-benchmark of §4, which was taken earlier and should be
re-confirmed on a quiet box before its *absolute* constant (~15 µs/range) is
relied upon. The *ratio* it establishes is not in doubt: the K=1 vs
K=one-range-per-row spread is 250x, far outside anything contention explains,
so the design conclusion (coalesce aggressively; never emit one range per row)
stands regardless.

## AI use statement

Written with AI assistance (Claude Opus agent via Claude Code), directed by
Evan.
