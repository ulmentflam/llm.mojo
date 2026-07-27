# Vocabulary-parallel LM head under ZeRO data parallelism: a feasibility study (2026-07-27)

Exploratory workstream for README roadmap item 1 ("de-resident the tied `wte`
embedding"). The question: should the LM head be vocabulary-**parallel**
(Megatron-LM style, each rank owning a slice of the vocabulary) rather than
vocabulary-**tiled** (each rank looping over vocabulary slices in sequence)?

Answer: **no, not in this trainer.** This document explains why, and — more
usefully — derives the exact condition under which the answer would flip.

## Headline (read this first)

- **Verdict: NO.** Two independent technical grounds, neither of which is
  "too much work":
  1. The premise that ZeRO-3 already shards `wte` usefully is **false**. At
     world size 7, **four of seven ranks hold zero vocabulary rows**, and
     shard boundaries land *mid-row* (20259.75 rows).
  2. The communication **does not pay at production shape**. At B=32, T=1024,
     N=7 it is a **4.5x regression** (modelled +99 ms per micro-step).
- **The break-even is exact and worth remembering:**
  vocabulary-parallelism beats the status quo **iff `N·B·T < V_p`** — iff the
  *global* token count per micro-step is below the padded vocabulary size
  (50,304). We train at ~229,000. We are 4.5x on the wrong side.
- **It is not a blocker on numerics.** A distributed softmax would meet the
  1e-5 bar; the pieces are already in `llmm/softmax.mojo`. Ruling this out on
  numerics would have been wrong.
- **Vocabulary-parallelism is not novel.** It is Megatron-LM's, from 2019
  [Megatron-LM]. The only contribution here is evaluating it *inside a ZeRO
  data-parallel trainer*, which is a setting Megatron does not target.

## The one-sentence result

> Under data parallelism, vocabulary-parallelism **minus the activation
> all-gather simply *is* vocabulary tiling**. The two are the two ends of a
> single trade, and the exchange rate is `N·B·T` against `V_p`.

That is the whole study in one line. If you decline to spend `N·B·T·C`
elements of communication replicating activations across ranks, then every
rank still needs all `V_p` vocabulary rows for its own tokens, so it must
stream those rows in slices — and streaming them in slices is exactly Team L's
vocabulary tiling. Megatron pays nothing for the replication because its
parallel ranks already share a batch. We would pay `N·B·T·C`. That factor is
the entire price of the mismatch, and at our shapes it is too high.

---

## Vocabulary (skip if you know this)

This project is educational, so every term is defined once here.

**Rank, world size.** A *rank* is one worker process, here pinned to one GPU.
*World size* (`N`) is how many there are. Ranks are numbered `0..N-1`.

**Data parallelism (DP).** Every rank holds a **complete copy of the model**
and processes a **different batch of data**. Gradients are averaged across
ranks after each step. This is what this trainer does.

**Tensor parallelism (TP), a kind of model parallelism.** Every rank holds a
**different slice of the model's weights** and they all process the **same
batch**. A single matrix multiply is split across ranks. This is what
Megatron-LM does.

That contrast — *different data, same weights* vs. *same data, different
weights* — is the crux of this entire document.

**Sharding.** Splitting one logical array into disjoint pieces, one per rank.
Rank `r`'s piece is its *shard*.

**All-gather.** Every rank has one shard; afterwards every rank has the whole
array. **Reduce-scatter.** Every rank has a full-size array; afterwards each
rank has the element-wise *sum* across ranks of just its own shard.
**All-reduce** = reduce-scatter + all-gather: everyone ends up with the full
sum. These are the three collective operations used below.

**The four kinds of training memory.** Keeping these apart is essential:
- **Parameters** — the weights, `124,475,904` elements for GPT-2 124M.
- **Gradients** — one per parameter, same size.
- **Optimizer state** — AdamW keeps two more (`m`, `v`), so 2x parameters.
- **Activations** — intermediate values saved during forward for use in
  backward. Scales with batch size; **the largest term here, by far.**

**ZeRO** [ZeRO] removes the redundancy in the first three under DP by
sharding them across ranks. Its three stages "correspond to the partitioning
of optimizer states, gradients, and parameters" (Section 1). Stage 3 shards
parameters too, so no rank holds the whole model at rest.

**Logits.** The model's raw output scores, one per vocabulary entry per token:
shape `(B, T, V_p)`. Softmax turns them into probabilities.

**Padded vocabulary (`V_p`).** GPT-2's real vocabulary is 50,257 entries;
it is padded to `V_p = 50304` because that is divisible by 128, which
matters for GPU matmul efficiency. The padding rows are real memory.

**Weight tying.** GPT-2 uses **the same matrix** `wte` for two jobs: looking
up input token embeddings (the encoder), and projecting the final hidden state
to logits (the LM head). Tying was introduced to cut parameters and improve
quality [Press & Wolf; Inan et al.]. It is why `wte` is read at both ends of
the network and written by two different backward paths — and therefore why it
resists being de-resident.

**Shapes used throughout.** `B` = per-rank batch, `T` = sequence length,
`C` = 768 channels, `V_p` = 50304, `L` = 12 layers, `N` = world size.

---

## The problem being attacked

`wte` is `V_p x C` = 50304 x 768 = **38,633,472 elements**, at flat offset 0
of the parameter vector. In fp32 that is **147.4 MiB**; the ZeRO-3 gather
window that contains it (`wte` + `wpe` + the two final-LayerNorm vectors) is
39,421,440 elements = **150.4 MiB**. That is the "~150 MiB floor" the roadmap
item names.

Three teams attacked it. Teams L (vocabulary tiling) and E (row-sparse
encoder gather) worked within the data-parallel design. This workstream asked
whether vocabulary *parallelism* was the better answer.

## What Megatron does, and why

Megatron-LM splits the embedding matrix across ranks by vocabulary
[Megatron-LM, Section 3]:

> "We parallelize the input embedding weight matrix E_{H×v} along the
> vocabulary dimension E=[E_1,E_2] (column-wise)."

The obvious way to finish is to all-gather the logits so every rank can run a
normal softmax. Megatron rejects that, and says exactly why:

> "However, for this case, the all-gather will communicate b×s×v elements
> (b is the batch-size and s is the sequence length) which is huge due to
> vocabulary size being large. To reduce the communication size, we fuse the
> output of the parallel GEMM [Y_1,Y_2] with the cross entropy loss which
> reduces the dimension to b×s."

> "Communicating scalar losses instead of logits is a huge reduction in
> communication that improves the efficiency of our model parallel approach."

The implementation [Megatron impl] does this with **three all-reduces in
forward**, each on a `[sequence, batch]` tensor — one per token, not one per
token per vocabulary entry:

1. `ReduceOp.MAX` on the per-token logit maximum (numerical stability),
2. `ReduceOp.SUM` on the predicted (target-class) logit, which lives on
   whichever rank owns that token's vocabulary shard,
3. `ReduceOp.SUM` on the per-token sum of exponentials — the softmax
   denominator.

Its **backward does no collectives at all**: each rank already owns exactly
the gradient slice it needs.

## Why it is not a drop-in port

Megatron's ranks all process the **same batch**, so the activations feeding
the LM head are already replicated on every rank — the input to the
vocabulary-parallel GEMM is free. This trainer's ranks each process a
**different batch**.

A note on evidence: the paper contains no single sentence stating "all
tensor-parallel ranks process the same batch." It is unambiguous from the
mechanics — Megatron duplicates rather than shards the surrounding ops ("we
choose to duplicate the computation across GPUs" for dropout, LayerNorm and
residual connections) and synchronises dropout by seeding "the random number
generators at the beginning of training with the same seed... identical
dropout patterns across all model parallel workers" (Appendix B.2). Both are
only coherent if every rank holds identical activations. Treat this as
established-by-mechanism, not quoted.

So under DP, rank `r`'s activations correspond only to rank `r`'s data. To
compute vocabulary-parallel logits you must first **manufacture** the
replication Megatron gets for free. The hybrid would be:

1. **All-gather `ln_f`** across ranks — `B·T·C` per rank contributed,
   `N·B·T·C` total.
2. Each rank computes logits for **all ranks' tokens** against **its own**
   vocabulary slice.
3. **Distributed softmax** — all-reduce the per-token max and sum-exp
   (Megatron's trick, three small collectives).
4. **`dwte` is local.** Each rank's vocabulary slice gradient needs no
   communication at all. This is the genuine win.
5. **Reduce-scatter `d_ln_f`** — each rank computed a partial gradient for
   every token; those partials must be summed and returned to owning ranks.

Step 4 is why the idea is attractive. Steps 1 and 5 are why it loses.

---

## Check 1 — does the ZeRO-3 shard boundary align with vocabulary rows?

This was checked first because it is cheap and it gates everything. The
appeal of the whole scheme was that under ZeRO-3 each rank *already* holds a
slice of `wte`, so vocabulary-parallelism would be free.

The shard is a flat, equal, contiguous split
(`train_gpt2.mojo:1738-1744`): `shard = ceil(num_parameters / N)` rounded up
to the AdamW SIMD width, and `rank r` owns `[r·shard, (r+1)·shard)`.

`wte` is 38,633,472 of 124,475,904 elements = **31.04%** of the vector, at
offset 0. So it occupies only the first `ceil(0.31·N)` shards:

| N | shard (elements) | ranks holding any `wte` | ranks holding **none** | shard ÷ C (rows) |
|---|---|---|---|---|
| 2 | 62,237,952 | rank 0 only (100% of `wte`) | 1 of 2 | 81039.00 |
| 4 | 31,118,976 | ranks 0-1 (rank 0: 80.5%) | 2 of 4 | 40519.50 |
| 7 | 17,782,272 | ranks 0-2 | **4 of 7** | 23154.00 |
| 8 | 15,559,488 | ranks 0-2 | **5 of 8** | 20259.75 |

**Two independent failures.**

1. **`wte` does not span the ranks.** At N=2 rank 0 holds *all* of it and
   rank 1 holds none. At our N=7, four ranks hold no vocabulary rows at all.
   "Each rank already owns a vocabulary slice" is false at every world size
   we run.
2. **Boundaries are not row-aligned.** `shard ÷ C` is fractional at N=4 and
   N=8 — at N=8, rank 1's shard begins **576 elements into vocabulary row
   20259**. Even ranks that do hold `wte` hold a fractional number of rows,
   and a vocabulary slice must be whole rows.

There is a third, practical failure: **the collectives cannot express the
fix.** `allgather_ranges` (`llmm/zero.mojo:963-973`) accepts flexible
*ranges*, but shard *ownership* is hard-wired to the uniform
`rank r owns [r·shard_size, (r+1)·shard_size)` partition; the same is true of
`reducescatter_buckets` (owner is computed as `f // opt`). There is no way to
express "ranks 0-2 hold data, ranks 3-7 hold none."

This kills the *stated* appeal — that vocabulary-parallelism would be free
because the sharding already exists. It does **not** by itself kill the idea,
because the parameter layout could be changed to carve `wte` into its own
row-aligned vocabulary partition. Per the mandate, the size of that change is
not a reason to stop. The idea is ruled out on communication instead.

## Check 2 — the communication cost model

**Both sides are compared at the same cadence.** ZeRO-2/3 with bucketing
reduce-scatters gradients **once per micro-step**, not once per optimizer
step (`train_gpt2.mojo:4020-4024`), and the `wte` all-gather also happens once
per forward (`train_gpt2.mojo:2582-2589`, reused in backward per the comment
at `3359-3364`). So this is apples-to-apples.

**What vocabulary-parallelism removes** (per rank, per micro-step): the
all-gather of the 39,421,440-element embed window, plus the reduce-scatter of
its gradient:

    baseline = 2 · W · (N-1)/N,    W = 39,421,440

**What it adds**: the `ln_f` all-gather and the `d_ln_f` reduce-scatter, plus
three small per-token all-reduces:

    vocab-parallel = 2 · (N-1) · B · T · C  +  O(N·B·T)

**The ratio, dropping the negligible scalar term:**

    vocab-parallel     2·(N-1)·B·T·C        N·B·T·C     N·B·T
    ──────────────  =  ─────────────   =    ───────  ≈  ─────
    baseline           2·W·(N-1)/N            W          V_p

**Break-even: `N·B·T = V_p`.** Note what is *not* in that expression: the
bandwidth, the dtype, and the element size. Both sides move bytes over the
same links by the same mechanism, so all of that cancels. **The verdict does
not depend on any measured rate** — measurement only converts it to
milliseconds.

Converted to wall clock at the measured staged-copy rate (see Check 5):

| shape | N | baseline | vocab-parallel | ratio | delta |
|---|---|---|---|---|---|
| B=4, T=64 (bench) | 7 | 257.8 MiB / 28.5 ms | 9.0 MiB / 1.0 ms | **0.04x** | −27.5 ms |
| B=8, T=1024 | 7 | 257.8 MiB / 28.5 ms | 288.8 MiB / 32.0 ms | 1.12x | +3.4 ms |
| **B=32, T=1024 (production)** | **7** | **257.8 MiB / 28.5 ms** | **1155.0 MiB / 127.9 ms** | **4.48x** | **+99.3 ms** |
| B=32, T=1024 | 8 | 263.2 MiB / 29.1 ms | 1347.5 MiB / 149.2 ms | 5.12x | +120.1 ms |

At the shape this project actually trains at — the README's "B≥32, T=1024,
~250-470 ms steps" — vocabulary-parallelism adds ~99 ms per micro-step to a
~250-470 ms step. That is a large regression, not a marginal one.

**The absence of CUDA P2P makes this worse, not better.** These GPUs expose
no peer-to-peer mappings, so collectives are hand-rolled as reduce-scatter +
all-gather over driver-staged device-to-device copies through a device-side
scratch buffer (`llmm/zero.mojo:236-264`). That machinery amplifies whatever
volume you give it. Vocabulary-parallelism *increases* the volume, so the
no-P2P penalty lands on the larger number.

**Where it *does* win.** Below `N·B·T = V_p` the sign flips, and at the
benchmark shape it wins by 25x. This is a real property, not a rounding
artifact: the technique trades communication-proportional-to-vocabulary for
communication-proportional-to-tokens, so it is the right choice whenever you
have more vocabulary than tokens. Megatron's regime (TP, where the token term
is `B·T` rather than `N·B·T`, and huge vocabularies) sits firmly on the
winning side. Ours does not.

## Check 3 — memory: the term nobody is measuring

Vocabulary-parallelism's memory win is bounded and smaller than it looks,
because **the logits buffer does not shrink at all.**

Rank `r` computes logits for all `N·B·T` gathered tokens against its own
`V_p/N` slice:

    N·B·T · (V_p/N)  =  B·T·V_p     — byte-for-byte identical to today.

And that buffer dominates everything. There is exactly one of them: the fused
classifier writes `dlogits` **in place** into `acts.logits`, which is why
`grad_act_sizes[Activations.logits]` is set to 0 on GPU
(`train_gpt2.mojo:2282`, `2313-2317`, `3062-3072`).

At B=32, T=1024: `32768 x 50304` = 1.648G elements = **6288 MiB fp32 /
3144 MiB bf16** — roughly **20-40x the 150 MiB parameter floor the roadmap
item targets.**

| | `wte` parameter residency | logits buffer |
|---|---|---|
| baseline (ZeRO-3 embed window) | 150.4 MiB | `B·T·V_p` |
| vocabulary-parallel, N=7 | 21.1 MiB (fixed factor N) | **unchanged** |
| vocabulary-parallel, N=8 | 18.4 MiB (fixed factor N) | **unchanged** |
| tiling, 8 / 16 / 64 tiles | 18.4 / 9.2 / 2.3 MiB (**tunable**) | see below |

So on the parameter axis tiling already matches vocabulary-parallelism at 8
tiles and beats it beyond that, with a factor decoupled from world size and
**zero** added communication.

> **PENDING — awaiting Team L.** Whether tiling also shrinks the logits
> buffer depends on whether it chunks the **cross-entropy** or only the
> **GEMM**. If only the GEMM, every tile still lands in the one full-size
> `acts.logits` and the peak is unchanged. If the classifier is chunked too —
> via the per-row running `(m, s)` recurrence already in
> `llmm/softmax.mojo:57-90` / `167-213`, plus a second pass or a backward
> recompute — the peak falls to `(B·T, V_tile)` plus `O(B·T)` stats. That is
> mechanically Liger's fused linear cross-entropy [Liger, §3.2]. Query sent;
> this section will be corrected to whatever was actually built.

Either way the conclusion for *this* workstream is unchanged:
vocabulary-parallelism leaves the dominant term untouched.

## Check 4 — numerics (NOT a blocker)

A distributed softmax must reproduce the single-GPU result to 1e-5 per
parameter. It would.

`llmm/softmax.mojo:167-213` already computes the row maximum and the row
sum-of-exponentials as **two separate block-level reductions**
(`block.max[...]`, then `block.sum[...]`), using the numerically-stable
online recurrence `s = s·exp(m − m_new) + exp(x − m_new)`. Extending a
block-level max/sum reduction to a cross-rank max/sum reduction is
structurally the same operation one level up — which is precisely what
Megatron's three forward all-reduces are.

All classifier math is already fp32-accumulate regardless of storage dtype
(`GPT2_DTYPE` is bf16 by default; every load casts to fp32 before use), and
losses are always fp32 (`StatsDType`). Reassociating a sum across ranks does
perturb results at the last bits, but the max-subtraction makes the
exponentials well-scaled, and the accumulation is fp32. **Ruling this idea out
on numerics would have been wrong**, and it is worth saying so: the reason to
reject it is communication, nothing else.

## Check 5 — what collective machinery could be reused

Less than hoped.

| primitive | reusable for this? |
|---|---|
| `allreduce(ptr, size)` | Flat buffer, requires `size % N == 0`. Usable in principle on an activation buffer. |
| `reducescatter`, `allgather` | Equal per-rank shards over a flat buffer. Usable for the `ln_f` / `d_ln_f` traffic. |
| `allgather_ranges` | Flexible ranges but **fixed uniform shard ownership** — cannot express "only ranks 0-2 hold data." |
| `reducescatter_buckets` | Same limitation (owner = `f // opt`). |
| `allreduce_scalar` | **One host-side `Float64`**, two barriers per call. Cannot reduce a vector. |

Two things do not exist and would have to be built: a **vector** all-reduce
over `B·T` per-token values, and a **MAX-op** reduction — every reduction in
`zero.mojo` today is a SUM. Neither is hard; both are noted for completeness,
not as reasons to stop.

Nothing in `zero.mojo` has ever been pointed at an activation buffer; the
collectives are only ever called on `params_memory`, `grads_memory`, and
gradient buckets. Mechanically they take raw pointers and would work, but this
would be the first use of that kind.

## Check 6 — interaction with the encoder and Team E

`wte` is tied, so it is also the encoder's lookup table. Under
vocabulary-parallelism the encoder becomes the mirror-image problem: rank `r`
holds only rows `[r·V_p/N, (r+1)·V_p/N)`, so a token outside that range has no
local embedding. Megatron handles this by zeroing out-of-range rows per shard
and **all-reducing the embedding output in the forward pass** — "an all-reduce
(g operator) is required after the input embedding" [Megatron-LM, Section 3].

That is an *additional* forward collective of `B·T·C` per micro-step, on top
of the two already counted, making the cost model above **optimistic**.

Note a correction to the fleet's shared reference notes: that all-reduce is a
**forward-pass activation** reduction, not a gradient all-reduce arising from
weight tying. The 2019 paper does not assert an embedding-gradient all-reduce
for tied weights; do not cite it for that.

This also collides with Team E's row-sparse encoder work, which exploits the
fact that only rows for tokens actually present in the batch receive gradient.
That is a *batch-content-dependent* sparsity; vocabulary-parallelism imposes a
*static* partition. The two are not complementary — under a static vocabulary
split, each rank's locally-present rows are whatever its slice happens to
contain, and the sparsity Team E exploits would have to be recomputed against
the gathered global token set rather than the local one.

---

## Verdict

**Do not build it.** The grounds are technical and neither is effort:

1. **Shard alignment fails** (Check 1). The premise that ZeRO-3 already gives
   each rank a vocabulary slice is false at every world size we run, and the
   boundaries are not even row-aligned. Fixable only by re-laying-out the
   parameter vector.
2. **Communication does not pay** (Check 2). 4.5x regression at B=32, T=1024,
   N=7; ~+99 ms per micro-step against a 250-470 ms step. The break-even
   `N·B·T < V_p` puts us 4.5x on the wrong side, and Check 6 shows the model
   is optimistic.
3. **It leaves the dominant memory term untouched** (Check 3). The logits
   buffer is `B·T·V_p` before and after — 20-40x larger than the parameter
   floor being targeted.

Numerics (Check 4) are *not* a reason to reject it, and neither is missing
collective machinery (Check 5).

Teams L and E remain the right answer. Team L's vocabulary tiling is the
correct shadow of Megatron's design under data parallelism: it is the end of
the trade that spends nothing on communication.

## What would change the answer

Stated positively, because the technique is sound in its own setting:

- **Train below `N·B·T = V_p`.** ~50,304 global tokens per micro-step at
  N=7 means B·T ≲ 7,186 — e.g. B=4, T=1024 gives 0.56x, a genuine win. If a
  memory-constrained recipe with small micro-batches and heavy gradient
  accumulation ever becomes the target, revisit this.
- **A much larger vocabulary.** The break-even scales with `V_p`. At a 256k
  vocabulary [Liger cites a 256k case] the balance moves by 5x in
  vocabulary-parallelism's favour.
- **Real tensor parallelism.** If this trainer ever grows a TP dimension —
  ranks sharing a batch — the activation all-gather disappears and Megatron's
  analysis applies directly and favourably. Vocabulary-parallelism is the
  right design *for TP*; it is being asked to do a job here that TP would make
  easy.
- **Working P2P / NVLink.** Would shrink the constant, not the 4.5x ratio.
  Not sufficient on its own.

## Provenance of numbers

Per the documentation bar, what was measured vs. modelled:

- **Verified by reading code:** all shard arithmetic (Check 1), the
  once-per-micro-step cadence of both collectives, the single in-place logits
  buffer, the collective inventory, the two-reduction softmax structure.
- **Verified by fetching sources live:** every quotation below, including two
  corrections to the fleet's shared reference notes (the ZeRO quote is in
  Section 1, not Section 5; the Megatron embedding all-reduce is a
  forward-pass activation reduction, not a tied-weight gradient reduction).
- **Modelled, not measured:** every byte count and every millisecond.
  Milliseconds are anchored on the staged-copy bandwidth measured at N=2.
  The **ratio** — and therefore the verdict — is independent of that anchor,
  because both sides move bytes over the same links by the same mechanism.
- **Not built:** no prototype was written. The verdict is a "no", so a
  prototype would have measured a design we do not want.

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
  Language Models.* arXiv:2411.09009 — for the framing that cross-entropy
  "consumes an order of magnitude more memory than the rest of the LLM
  combined", which Check 3 independently confirms here.

## AI use statement

Written with AI assistance (Claude Opus agent via Claude Code), directed by
Evan.
