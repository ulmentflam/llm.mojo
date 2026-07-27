# Vocabulary-parallel LM head under ZeRO data parallelism: a feasibility study (2026-07-27)

Exploratory workstream for README roadmap item 1 ("de-resident the tied `wte`
embedding"). The question: should the LM head be vocabulary-**parallel**
(Megatron-LM style, each rank owning a slice of the vocabulary) rather than
vocabulary-**tiled** (each rank looping over vocabulary slices in sequence)?

Answer: **no, not in this trainer.** This document explains why, derives the
exact condition under which the answer would flip, and records two things
found along the way that matter more than the original question.

## Headline (read this first)

- **Verdict: NO**, on communication cost. Not on effort — effort was
  explicitly excluded as a criterion.
- **The load-bearing result is a pointwise dominance, not a single
  configuration.** Generalise the idea to a free parameter `g` (vocabulary
  split across groups of `g` ranks). For **every** `g`, vocabulary tiling
  reaches the **same** memory residency at **1.00x** communication, while
  vocabulary-parallelism reaches it at **1.15x-5.13x**. No member of the
  family is a good trade at production shape.
- **The break-even is exact and dtype-independent:** vocabulary-parallelism
  wins **iff `N·B·T < V_p`** — iff *global* tokens per micro-step are below
  50,304. We train at 229,376. We are **4.486x** on the wrong side.
- **On this codebase the premise fails independently.** At world size 7,
  **four of seven ranks hold zero vocabulary rows**, and shard boundaries land
  mid-row (20259.75 rows).
- **It is NOT blocked on numerics.** A distributed softmax would meet the 1e-5
  bar; the machinery is already in `llmm/softmax.mojo`. Recording this so the
  right conclusion doesn't get filed under the wrong reason.
- **Two findings that outlive the question** (Check 4): the campaign's ~150 MiB
  parameter win sits beside a **multi-GiB activation term that no workstream
  touches**, and a correction to a claim this document previously made in its
  own favour.
- **Vocabulary-parallelism is not novel.** It is Megatron-LM's, from 2019
  [Megatron-LM]. The contribution here is evaluating it *inside a ZeRO
  data-parallel trainer* — a setting Megatron does not target.

## The one-sentence result

> Under data parallelism, vocabulary-parallelism **minus the activation
> all-gather simply *is* vocabulary tiling**. The two are the two ends of a
> single trade, and the exchange rate is `N·B·T` against `V_p`.

If you decline to spend `N·B·T·C` elements replicating activations across
ranks, then every rank still needs all `V_p` vocabulary rows for its own
tokens, so it must stream them in slices — and that is vocabulary tiling.
Megatron pays nothing for the replication because its ranks already share a
batch. We would pay `N·B·T·C`. That factor is the entire price of the
mismatch, and at our shapes it is too high.

---

## Vocabulary (skip if you know this)

**Rank, world size.** A *rank* is one worker process, here pinned to one GPU.
*World size* (`N`) is how many there are.

**Data parallelism (DP).** Every rank holds a **complete copy of the model**
and processes a **different batch**. Gradients are averaged across ranks.
This is what this trainer does.

**Tensor parallelism (TP), a kind of model parallelism.** Every rank holds a
**different slice of the weights** and they all process the **same batch**. A
single matrix multiply is split across ranks. This is what Megatron does.

*Different data, same weights* vs. *same data, different weights* — that
contrast is the crux of this entire document.

**Sharding.** Splitting one logical array into disjoint per-rank pieces.

**All-gather**: every rank has one shard, afterwards every rank has the whole
array. **Reduce-scatter**: every rank has a full-size array, afterwards each
has the element-wise *sum across ranks* of just its own shard.
**All-reduce** = both: everyone ends up with the full sum.

**The four kinds of training memory** — keeping these apart is essential:
- **Parameters** — the weights, 124,475,904 elements for GPT-2 124M.
- **Gradients** — one per parameter.
- **Optimizer state** — AdamW keeps two more (`m`, `v`), so 2x parameters.
- **Activations** — intermediates saved in forward for use in backward.
  Scales with batch size. **The largest term here by an order of magnitude**,
  which turns out to matter (Check 4).

**ZeRO** [ZeRO] removes redundancy in the first three under DP by sharding
them. Its three stages "correspond to the partitioning of optimizer states,
gradients, and parameters". Stage 3 shards parameters too.

**Logits.** Raw output scores, one per vocabulary entry per token:
`(B, T, V_p)`. Softmax turns them into probabilities.

**Padded vocabulary (`V_p`).** GPT-2's 50,257 entries padded to
`V_p = 50304` (divisible by 128, for matmul efficiency). The padding is real
memory.

**Weight tying.** GPT-2 uses **the same matrix** `wte` both to look up input
embeddings and to project the final hidden state to logits [Press & Wolf;
Inan et al.]. That is why `wte` is read at both ends and written by two
backward paths — and why it resists being de-resident.

**Shapes.** `B` = per-rank batch, `T` = sequence length, `C` = 768,
`V_p` = 50304, `L` = 12 layers, `N` = world size.

---

## The problem being attacked

`wte` is `V_p x C` = **38,633,472 elements** at flat offset 0 of the parameter
vector — **147.4 MiB** fp32. The ZeRO-3 gather window containing it (`wte` +
`wpe` + the two final-LayerNorm vectors) is 39,421,440 elements =
**150.4 MiB**, and it is a persistent buffer. That is the "~150 MiB floor".

Teams L (vocabulary tiling) and E (row-sparse encoder gather) attacked it
within the data-parallel design. This workstream asked whether vocabulary
*parallelism* was the better answer.

## What Megatron does, and why

[Megatron-LM, Section 3]:

> "We parallelize the input embedding weight matrix E_{H×v} along the
> vocabulary dimension E=[E_1,E_2] (column-wise)."

The obvious completion is to all-gather the logits and run a normal softmax.
Megatron rejects that and says exactly why:

> "However, for this case, the all-gather will communicate b×s×v elements
> (b is the batch-size and s is the sequence length) which is huge due to
> vocabulary size being large. To reduce the communication size, we fuse the
> output of the parallel GEMM [Y_1,Y_2] with the cross entropy loss which
> reduces the dimension to b×s."

> "Communicating scalar losses instead of logits is a huge reduction in
> communication that improves the efficiency of our model parallel approach."

The implementation [Megatron impl] uses **three all-reduces in forward**, each
on a `[sequence, batch]` tensor — one value per *token*, not per token per
vocabulary entry: `ReduceOp.MAX` on the per-token logit maximum, `ReduceOp.SUM`
on the target-class logit (which lives on whichever rank owns that token's
vocabulary shard), and `ReduceOp.SUM` on the per-token sum of exponentials.
Its **backward does no collectives at all**.

## Why it is not a drop-in port

Megatron's ranks all process the **same batch**, so the activations feeding
the LM head are already replicated — the input to the vocabulary-parallel GEMM
is free. This trainer's ranks each process a **different batch**.

*On evidence:* the paper contains no sentence stating "all tensor-parallel
ranks process the same batch." It is unambiguous from the mechanics — Megatron
duplicates rather than shards the surrounding ops ("we choose to duplicate the
computation across GPUs" for dropout, LayerNorm and residual connections) and
synchronises dropout by seeding "the random number generators at the beginning
of training with the same seed... identical dropout patterns across all model
parallel workers" (Appendix B.2). Both are only coherent if every rank holds
identical activations. Established by mechanism, not quoted.

So under DP you must **manufacture** the replication Megatron gets free:

1. **All-gather `ln_f`** — `B·T·C` per rank contributed.
2. Each rank computes logits for **all ranks' tokens** against **its own**
   vocabulary slice.
3. **Distributed softmax** — all-reduce per-token max and sum-exp.
4. **`dwte` is local** — no communication. *This is the genuine win.*
5. **Reduce-scatter `d_ln_f`** — each rank computed a partial gradient for
   every token; those partials must be summed and returned.

Step 4 is the appeal. Steps 1 and 5 are why it loses.

---

## Check 1 — the communication cost model

**Both sides are compared at the same cadence**, which was verified rather
than assumed: the ZeRO-2/3 bucketed reduce-scatter runs **per micro-step**
(`train_gpt2.mojo:4020-4024`), and the `wte` all-gather runs once per forward
(`train_gpt2.mojo:2582-2589`, reused in backward per the comment at
`3359-3364`).

Per rank, per micro-step, in elements:

    baseline        = 2 · W · (N-1)/N,        W = 39,421,440
    vocab-parallel  = 2 · (N-1) · B·T·C  +  3 · 2(N-1)/N · N·B·T

Dropping the negligible scalar term, the ratio collapses:

    vocab-parallel     2(N-1)·B·T·C        N·B·T·C     N·B·T
    ──────────────  =  ────────────   =    ───────  ≈  ─────
    baseline           2W(N-1)/N              W          V_p

**Break-even: `N·B·T = V_p`.** Note what is *absent*: bandwidth, dtype, element
size. Both sides move bytes over the same links by the same mechanism, so all
of it cancels. **The verdict does not depend on any measured rate.**

| shape | N | baseline | vocab-parallel | ratio |
|---|---|---|---|---|
| B=4, T=64 (benchmark) | 7 | 257.8 MiB | 9.0 MiB | **0.035x** |
| B=4, T=1024 | 7 | 257.8 MiB | 144.6 MiB | 0.561x |
| B=8, T=1024 | 7 | 257.8 MiB | 289.1 MiB | 1.122x |
| **B=32, T=1024 (production)** | **7** | **257.8 MiB** | **1156.5 MiB** | **4.486x** |
| B=32, T=1024 | 8 | 263.2 MiB | 1349.2 MiB | 5.127x |

**The model is deliberately charitable to the option being rejected.** Two
choices both bend toward vocabulary-parallelism:

1. Per-collective **latency and barrier costs are ignored**, counting
   bandwidth only. That penalises the *small-volume* side — so the
   benchmark-shape win is **overstated**, the production-shape loss is not.
2. The **extra `B·T·C` forward all-reduce that a vocabulary-parallel encoder
   requires is omitted entirely** (Check 7).

It still loses 4.486x. A model tuned to produce its own conclusion would be
worth little; this one was tuned against it.

**No CUDA P2P makes it worse, not better.** These GPUs expose no peer-to-peer
mappings, so collectives are hand-rolled as reduce-scatter + all-gather over
driver-staged device-to-device copies through a device-side scratch buffer
(`llmm/zero.mojo:236-264`). That amplifies whatever volume it is given, and
vocabulary-parallelism increases the volume.

## Check 2 — the steelman: no member of the family pays

Check 1 evaluates the *extreme* form, vocabulary split across all `N` ranks.
Rejecting the extreme version of an idea is weak evidence against the idea, so
here is the general case.

**Grouped vocabulary-parallelism.** Partition the `N` ranks into `G = N/g`
groups of `g`. Split the vocabulary across the `g` members *within* a group;
replicate that arrangement across groups. Now `g` is a free parameter exactly
as tile count is for tiling: `g = 1` is today's design, `g = N` is Check 1.
Each rank permanently holds `wte/g` and never gathers `wte` at all.

    activations (within group)  = 2·(g-1)·B·T·C
    dwte all-reduce (across G)  = 2·(G-1)/G · (wte/g)
    residency                   = wte/g

The middle term is easy to miss and is decisive: with `G > 1` groups, every
vocabulary row is owned by one rank in *every* group, so its gradient must
still be reduced across groups. It vanishes only at `g = N`.

**At N=8** (modelled):

| g | groups | benchmark B=4,T=64 | production B=32,T=1024 | residency |
|---|---|---|---|---|
| 2 | 4 | 112.0 MiB (0.43x) | 303.3 MiB (**1.15x**) | 73.7 MiB |
| 4 | 2 | 41.4 MiB (0.16x) | 615.1 MiB (**2.34x**) | 36.8 MiB |
| 8 | 1 | 10.5 MiB (0.04x) | 1349.2 MiB (**5.13x**) | 18.4 MiB |

Now tiling, which reaches **the same residency** at **1.00x** communication on
every row:

| tiles | 2 | 4 | 8 | 16 | 64 |
|---|---|---|---|---|---|
| residency | 73.7 MiB | 36.8 MiB | 18.4 MiB | 9.2 MiB | 2.3 MiB |
| communication | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |

**Tiling dominates grouped vocabulary-parallelism pointwise at production
shape.** For any memory target, tiling hits it at 1.00x communication;
vocabulary-parallelism hits the *same* target at 1.15x-5.13x. There is no
value of `g` that is a good trade, so the rejection is not an artifact of
picking the extreme.

And the closing argument: tiling extends past `t = N` (16, 64 tiles) into a
region vocabulary-parallelism **structurally cannot reach**, since `g ≤ N` by
construction. The tile count is decoupled from the world size; the group size
is not.

At benchmark shape the ordering reverses for every `g`, consistent with the
break-even. Same technique, same code, opposite conclusion — driven purely by
shape.

## Check 3 — and on this codebase, the premise fails too

Check 2 is the general result. This is the concrete confirmation: the specific
appeal that motivated the workstream — that ZeRO-3 *already* gives each rank a
vocabulary slice, so vocabulary-parallelism would be nearly free — is false as
built.

The shard is a flat, equal, contiguous split
(`train_gpt2.mojo:1738-1744`): `shard = ceil(num_parameters / N)` rounded up to
the AdamW SIMD width, with `rank r` owning `[r·shard, (r+1)·shard)`.

`wte` is 38,633,472 of 124,475,904 elements = **31.04%**, at offset 0. So it
occupies only the first `ceil(0.31·N)` shards:

| N | shard (elements) | ranks holding any `wte` | ranks holding **none** | shard ÷ C (rows) |
|---|---|---|---|---|
| 2 | 62,237,952 | rank 0 only (100% of `wte`) | 1 of 2 | 81039.00 |
| 4 | 31,118,976 | ranks 0-1 (rank 0: 80.5%) | 2 of 4 | 40519.50 |
| 7 | 17,782,272 | ranks 0-2 | **4 of 7** | 23154.00 |
| 8 | 15,559,488 | ranks 0-2 | **5 of 8** | 20259.75 |

1. **`wte` does not span the ranks.** At N=2 rank 0 holds *all* of it. At our
   N=7, four ranks hold no vocabulary rows at all.
2. **Boundaries are not row-aligned.** At N=8, rank 1's shard begins **576
   elements into vocabulary row 20259**. A vocabulary slice must be whole rows.
3. **The collectives cannot express the fix.** `allgather_ranges`
   (`llmm/zero.mojo:963-973`) accepts flexible *ranges*, but shard *ownership*
   is hard-wired to the uniform `rank r owns [r·shard_size, (r+1)·shard_size)`
   partition; likewise `reducescatter_buckets` (owner = `f // opt`). There is
   no way to say "ranks 0-2 hold data, ranks 3-7 hold none."

This could be fixed by re-laying-out the parameter vector, and per the mandate
the size of that change is not a reason to stop. It doesn't need to be — Check
2 already rules out the family the re-layout would enable.

## Check 4 — memory, a correction, and the finding that outlives the question

**Vocabulary-parallelism does not shrink the logits buffer.** Rank `r` computes
logits for all `N·B·T` gathered tokens against its own `V_p/N` slice:

    N·B·T · (V_p/N)  =  B·T·V_p     — byte-for-byte identical to today.

There is exactly one such buffer: the fused classifier writes `dlogits` **in
place** into `acts.logits`, which is why `grad_act_sizes[Activations.logits]`
is 0 on GPU (`train_gpt2.mojo:2282`, `2313-2317`, `3062-3072`).

| | `wte` residency | logits buffer |
|---|---|---|
| baseline (ZeRO-3 embed window) | 150.4 MiB | `B·T·V_p` |
| vocabulary-parallel, N=7 / N=8 | 21.1 / 18.4 MiB (fixed factor N) | **unchanged** |
| tiling, 8 / 16 / 64 tiles | 18.4 / 9.2 / 2.3 MiB (**tunable**) | **unchanged** |

### A correction, recorded rather than quietly fixed

An earlier draft of this document claimed tiling *also* shrinks the logits
buffer by the tile factor. That was asserted without evidence and is **false
as built**. Team L's change tiles the LM-head **GEMM only**:
`llmm/fused_classifier.mojo`, `llmm/softmax.mojo` and `llmm/crossentropy.mojo`
are untouched; each tile is written as a column slice *into the one full-size*
`acts.logits` via a strided leading dimension, and the classifier then runs
once on the whole materialised buffer. The allocation is still
`act_sizes[Activations.logits] = B * T * V_p`.

The correction does not weaken the verdict — Check 2 was always about
communication at *matched* residency. It changes one thing: on the logits axis
**neither approach helps**. They are equal at zero. It is recorded here in
full because the erroneous version flattered the option this document
recommends, and a reader who watches a convenient claim vanish silently learns
nothing.

### The finding that outlives the original question

Chasing that correction surfaced the real memory picture. At B=32, T=1024,
fp32 — with `logits` and `att_probs` **measured** by Team M's allocation
accounting at B=8 and scaled (both reproduce exactly from first principles):

| tensor | size | vs. the 150 MiB floor |
|---|---|---|
| `att_probs` (goes as `T²`) | **18,432 MiB** | **123x** |
| `logits` | **6,288 MiB** | **42x** |
| `wte` ZeRO-3 window ("the floor") | 150.4 MiB | 1x |

**The campaign's headline win reduces the smallest of these three by ~87%,
while the two terms that are 42x and 123x larger are untouched by every
workstream in flight** — Teams L, E and V alike. A reader who takes away only
"we removed the embedding floor" would misjudge the memory picture badly.

Both larger terms are addressable with known recipes:
- **Logits** — carry per-row running `(m, s)` statistics across tiles using the
  online-softmax recurrence already in `llmm/softmax.mojo:57-90` / `167-213`,
  and recompute each tile in backward. That is Liger's fused linear
  cross-entropy [Liger, §3.2]; [CCE] makes the same memory argument. Team L's
  tiled GEMM loop is exactly the structure it runs inside, and they cost the
  recompute at ~+10% step time. The obstacle is that `fused_classifier` writes
  `dlogits` in place into `acts.logits`, and a chunked design has no full
  buffer to write into. *Now commissioned as phase 2 of Team L's workstream.*
- **`att_probs`** — three times larger still, and as far as this study can
  tell, nobody has looked at it. Flagged, not investigated.

## Check 5 — numerics (NOT a blocker)

`llmm/softmax.mojo:167-213` already computes the row maximum and row
sum-of-exponentials as **two separate block-level reductions**
(`block.max[...]` then `block.sum[...]`), using the stable online recurrence
`s = s·exp(m − m_new) + exp(x − m_new)`. Extending a block-level max/sum
reduction to a cross-rank one is the same operation one level up — precisely
what Megatron's three forward all-reduces are.

All classifier math is already fp32-accumulate regardless of storage dtype
(`GPT2_DTYPE` is bf16 by default; every load casts to fp32 first), and losses
are always fp32 (`StatsDType`). Reassociating a sum across ranks perturbs the
last bits, but max-subtraction keeps the exponentials well-scaled and the
accumulation is fp32.

**Ruling this idea out on numerics would have been wrong**, and saying so
matters: a right conclusion filed under a wrong reason misleads the next
reader who has a variant where numerics *are* the question.

## Check 6 — reusable collective machinery

| primitive | reusable here? |
|---|---|
| `allreduce(ptr, size)` | Flat buffer, needs `size % N == 0`. Usable in principle. |
| `reducescatter`, `allgather` | Equal per-rank shards, flat buffer. Usable for `ln_f` / `d_ln_f`. |
| `allgather_ranges` | Flexible ranges, **fixed uniform ownership** — cannot express partial ownership. |
| `reducescatter_buckets` | Same limitation (owner = `f // opt`). |
| `allreduce_scalar` | **One host-side `Float64`**, two barriers per call. Cannot reduce a vector. |

Two things would have to be built: a **vector** all-reduce over `B·T`
per-token values, and a **MAX-op** reduction — every reduction in `zero.mojo`
today is a SUM. Neither is hard; noted for completeness, not as a reason to
stop. Also, no collective has ever been pointed at an activation buffer; they
are only ever called on `params_memory`, `grads_memory`, and gradient buckets.

## Check 7 — the encoder, and Team E

`wte` is tied, so it is also the encoder's lookup table. Under
vocabulary-parallelism rank `r` holds only rows `[r·V_p/N, (r+1)·V_p/N)`, so a
token outside that range has no local embedding. Megatron zeroes out-of-range
rows per shard and **all-reduces the embedding output in the forward pass** —
"an all-reduce (g operator) is required after the input embedding"
[Megatron-LM, Section 3]. That is an *additional* `B·T·C` forward collective
per micro-step, omitted from Check 1, making that model optimistic.

*A correction for the fleet's shared reference notes:* that all-reduce is a
**forward-pass activation** reduction, not a gradient all-reduce arising from
weight tying. The 2019 paper does not assert an embedding-gradient all-reduce
for tied weights; do not cite it for that.

This also collides with Team E's row-sparse encoder work, which exploits the
fact that only rows for tokens present in the batch receive gradient — a
*batch-content-dependent* sparsity. Vocabulary-parallelism imposes a *static*
partition. The two are not complementary: under a static split, the sparsity
Team E exploits would have to be recomputed against the gathered global token
set rather than the local one.

---

## Verdict

**Do not build it.** Grounds, in order of strength:

1. **No member of the family pays** (Check 2). At production shape, tiling
   reaches every achievable residency at 1.00x communication where
   vocabulary-parallelism costs 1.15x-5.13x, and tiling extends into a region
   vocabulary-parallelism structurally cannot reach.
2. **The break-even puts us 4.486x on the wrong side** (Check 1), under a
   model deliberately biased in the option's favour.
3. **The premise fails on this codebase anyway** (Check 3).
4. **It leaves the dominant memory terms untouched** (Check 4) — as, today,
   does everything else.

Numerics (Check 5) and missing collective machinery (Check 6) are **not**
reasons to reject it.

Teams L and E remain the right answer. Team L's vocabulary tiling is the
correct shadow of Megatron's design under data parallelism: the end of the
trade that spends nothing on communication.

## What would change the answer

Stated positively — the technique is sound in its own setting:

- **Train below `N·B·T = V_p`.** At N=7 that is `B·T ≲ 7,186`; e.g. B=4,
  T=1024 gives 0.561x, a genuine win. If a memory-constrained recipe with
  small micro-batches and heavy gradient accumulation becomes the target,
  revisit.
- **A much larger vocabulary.** The break-even scales with `V_p`; at a 256k
  vocabulary the balance moves ~5x in its favour.
- **Real tensor parallelism.** If this trainer grows a TP dimension — ranks
  sharing a batch — the activation all-gather disappears and Megatron's
  analysis applies directly. Vocabulary-parallelism is the right design *for
  TP*; here it is being asked to do a job TP would make easy.
- **Working P2P / NVLink.** Shrinks the constant, not the ratio. Not
  sufficient alone.

## Provenance of numbers

- **Verified by reading code:** all shard arithmetic (Check 3), the
  once-per-micro-step cadence of both collectives, the single in-place logits
  buffer, the collective inventory, the two-reduction softmax structure, and
  Team L's GEMM-only scope.
- **Verified by fetching sources live:** every quotation, including two
  corrections to the fleet's shared reference notes (the ZeRO stage-taxonomy
  quote is in **Section 1**, not Section 5; the Megatron embedding all-reduce
  is a forward-pass activation reduction, not a tied-weight gradient
  reduction).
- **Measured by Team M** (`LLMM_MEM_REPORT=1`, world 2, B=8, T=1024, fp32):
  `logits` = 1572.000 MiB and `att_probs` = 4608.000 MiB. Both reproduce
  exactly from first principles, and both are linear in `B`, giving the
  B=32 figures in Check 4.
- **Modelled, not measured:** every communication byte count. **The ratio —
  and therefore the verdict — is bandwidth-independent**, because both sides
  move bytes over the same links by the same mechanism. Milliseconds are
  deliberately omitted; they would require a bandwidth anchor and would not
  move the crossover.
- **Not built:** no prototype. The verdict is a "no", so a prototype would
  have measured a design we do not want. A collective-timing harness
  (`bench_collectives.mojo`) *was* built, because the repo had none and the
  ~75 GB/s claim in `llmm/zero.mojo` had never been reproducible.

Machine-readable cost model, including the full grouped-family sweep:
`docs/ai/data/vocab_parallel_cost_model_2026-07-27.json`.

## References

- **[ZeRO]** Rajbhandari, Rasley, Ruwase, He (2019). *ZeRO: Memory
  Optimizations Toward Training Trillion Parameter Models.* arXiv:1910.02054.
  Stage taxonomy quoted from **Section 1** (Extended Introduction).
- **[Megatron-LM]** Shoeybi, Patwary, Puri, LeGresley, Casper, Catanzaro
  (2019). *Megatron-LM: Training Multi-Billion Parameter Language Models Using
  Model Parallelism.* arXiv:1909.08053, Section 3 and Appendix B.2. Quotes are
  in the full text, not the abstract page.
- **[Megatron impl]** NVIDIA/Megatron-LM,
  `megatron/core/tensor_parallel/cross_entropy.py`.
- **[Press & Wolf]** Press, Wolf (2016). *Using the Output Embedding to
  Improve Language Models.* arXiv:1608.05859.
- **[Inan et al.]** Inan, Khosravi, Socher (2016). *Tying Word Vectors and
  Word Classifiers.* arXiv:1611.01462.
- **[Liger]** Hsu et al. (2024). *Liger Kernel: Efficient Triton Kernels for
  LLM Training.* arXiv:2410.10989, §3.2.
- **[CCE]** Wijmans et al. (2024). *Cut Your Losses in Large-Vocabulary
  Language Models.* arXiv:2411.09009 — cross-entropy "consumes an order of
  magnitude more memory than the rest of the LLM combined", which Check 4
  independently confirms.

## AI use statement

Written with AI assistance (Claude Opus agent via Claude Code), directed by
Evan.
