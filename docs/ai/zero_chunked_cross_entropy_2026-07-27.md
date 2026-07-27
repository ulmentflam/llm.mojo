# Chunking the cross-entropy so the logits tensor never exists

*Phase 2 of the LM-head workstream. Phase 1 (`docs/ai/zero_lm_head_vocab_tiling_2026-07-27.md`)
split the LM-head GEMM into vocabulary tiles. This splits the **loss** too, and
that is what actually removes the memory.*

---

## 0. The one-paragraph version

The largest buffer this trainer allocates, after the attention probabilities, is
not a weight and not a gradient. It is `logits`: one float per (token, vocabulary
entry), **6.1 GiB** at batch 32, sequence 1024. It exists for the duration of one
softmax and one gradient formula. Behind an opt-in flag, this change computes
that softmax and that gradient a vocabulary **slice at a time**, so the full
tensor is never formed — the peak drops to whichever is larger, one slice or a
small reserve for inference. The price is that the backward pass has to
*recompute* each slice, which is one extra matrix multiply per step. The flag
exists because that trade is worth taking on some runs and not on others, and
that is the operator's judgement, not ours.

---

## 1. Background, from scratch

You do not need to know anything about this codebase, ZeRO, or online algorithms
to read this section. If you already know what a softmax is and why it subtracts
a maximum, skip to §2.

### 1.1 What the "LM head" is

A GPT-2-style language model reads a sequence of tokens and, for every position,
predicts what the next token will be. "Token" here means a subword unit; GPT-2's
vocabulary has **V = 50257** of them. Internally each position is represented by
a vector of **C = 768** numbers — the "hidden state" or "channels".

The last thing the model does is turn each 768-number vector into 50257 scores,
one per possible next token. Those scores are called **logits**. The operation is
a single matrix multiply against the token-embedding matrix `wte`, which is
`[V, C]`:

```
logits[t, v] = sum over c of  hidden[t, c] * wte[v, c]
```

That matrix multiply is the **LM head**. (It reuses `wte`, the same matrix that
converts input tokens into vectors at the bottom of the model. Reusing it is
called **weight tying** — Press & Wolf 2016, Inan et al. 2016 — and it saves
parameters and usually improves quality. It also means one tensor is touched at
both ends of the model, which matters a lot in phase 1 and only a little here.)

One more wrinkle: the vocabulary is **padded** from V = 50257 up to
**V_p = 50304** so that the matrix dimensions are friendly to the GPU (50304 =
128 × 393). The extra 47 columns are not real tokens. They hold whatever the
GEMM happens to produce, and every downstream step has to be careful to ignore
them. Call `[V, V_p)` the **padding columns**; they will come back repeatedly.

### 1.2 Softmax and cross-entropy, and why the maximum is subtracted

The logits are arbitrary real numbers. To turn them into probabilities we use
the **softmax**:

```
p[v] = exp(logit[v]) / sum over u of exp(logit[u])
```

The training loss for one position is the **cross-entropy**: the negative log
probability the model assigned to the token that actually came next. If that
token is index `T` ("the target"):

```
loss = -log p[T] = log( sum over u of exp(logit[u]) ) - logit[T]
```

The expression `log(sum(exp(x)))` is called the **log-sum-exp**.

Computed naively this overflows. Logits routinely reach ±20 or more, and
`exp(89)` already overflows a 32-bit float (max ≈ 3.4 × 10^38). The standard fix
uses the fact that softmax is **shift-invariant**: subtracting the same constant
from every logit in a row changes nothing, because the constant cancels between
numerator and denominator. So we subtract the row's maximum `m`:

```
s = sum over u of exp(logit[u] - m)          # every term is in (0, 1]
loss = log(s) + m - logit[T]
```

Now the largest exponent is exactly 0, `exp` of it is exactly 1, and nothing can
overflow. The pair `(m, s)` — **max and sum-exp** — is the complete summary of a
row that the loss and the gradient both need. Remember that pair; the whole of
this document is about computing it without holding the row.

### 1.3 The gradient

The gradient of the cross-entropy with respect to the logits is famously simple:

```
dlogit[v] = ( p[v] - onehot[v] ) * dloss
```

where `onehot[v]` is 1 if `v` is the target and 0 otherwise, and `dloss` is the
scalar the chain rule hands down from above (in this trainer, the constant
`1 / (B·T·grad_accum_steps)`, because the loss is a mean). Note that this needs
`p`, which needs `(m, s)`, which needs the whole row.

Note also a useful consequence: since the probabilities sum to 1 and the one-hot
sums to 1, **every row of `dlogits` sums to zero**. That is a free correctness
check and one of the tests below uses it.

The gradient then flows two ways out of the LM head — into the hidden states
(`d_hidden = dlogits @ wte`) and into the weights (`d_wte = dlogitsᵀ @ hidden`).

### 1.4 Why any of this is a memory problem

`logits` has one float per (position, padded vocabulary entry). Positions =
batch × sequence:

```
B=4,  T=1024:   4096 × 50304 × 4 bytes =   824,180,736 B =   786 MiB
B=32, T=1024:  32768 × 50304 × 4 bytes = 6,593,445,888 B =  6288 MiB  ← production shape here
```

(Every "MiB"/"GiB" in this document is binary — 1 MiB = 1048576 B — and every
figure is given in bytes alongside so it can be checked. That matters here:
phase 1's writeup reports this same B=4 buffer as "824 MiB", which is the
decimal megabyte count wearing a binary label. 824,180,736 B is 786 MiB *or*
824 MB, not 824 MiB.)

Compare that with the thing this whole campaign originally set out to remove:
the `wte` matrix itself is 50304 × 768 × 4 = 154,533,888 B = **147 MiB**. The
*activation* is **42.7×** the *parameter*. That is the asymmetry that motivates this
change: the large-vocabulary output layer costs far more in transient activation
than in weights, and it costs more the larger your batch is — precisely when you
are already short of memory.

This is well known. Liger Kernel (arXiv:2410.10989, §3.2
*FusedLinearCrossEntropy*) motivates its kernel with exactly this arithmetic:
"Take the Gemma model as an example, single GPU training with a batch size of 8
and sequence length of 4096 ... the 256k vocabulary size will result in a 16.8GB
logit tensor of precision bfloat16". Cut Your Losses (arXiv:2411.09009) opens
with the same observation from the other side — that cross-entropy "consumes an
order of magnitude more memory than the rest of the LLM combined".

*(The Liger sentence is body text and does **not** appear on the arXiv abstract
page; read it at <https://ar5iv.labs.arxiv.org/html/2410.10989>. The 16.8 GB
figure is meaningless without the batch-8 / sequence-4096 / 256k-vocabulary
shape quoted alongside it, which is why that shape is quoted. The Cut Your
Losses sentence is in its abstract: <https://arxiv.org/abs/2411.09009>.)*

**Say it plainly: chunked cross-entropy is not novel and this document does not
claim it is.** It is Liger's technique. What is written here is an
implementation of it inside a ZeRO-sharded Mojo trainer, sitting on top of the
vocabulary tiling phase 1 already built, plus honest measurements of what it
costs and what it buys on this hardware.

(A note on the two citations, because they are easy to blur together: Liger
**chunks** — it literally splits the work into pieces and computes a partial
result per piece, which is what happens below. Cut Your Losses uses a different
mechanism, computing the log-sum-exp on the fly inside a flash-attention-style
kernel without ever writing logits to memory at all. Same goal, different
machine. Cite each for its own. A third neighbour, Megatron-LM
(arXiv:1909.08053), also splits the output layer over the vocabulary — but it is
motivated by **communication** volume under tensor parallelism, not by activation
memory on one device, so it is not cited here as a memory technique.)

---

## 2. Where this codebase was

Phase 1 already split the LM-head matrix multiply into **vocabulary tiles**: 8
tiles of 6400 columns each at the default setting. Tile *k* computes columns
`[6400k, 6400(k+1))` of the logits, reading rows `[6400k, 6400(k+1))` of `wte`.

But it wrote those columns **into the same full-size buffer**. The tiling was a
rearrangement of the GEMM, not a reduction in what was resident. Then
`fused_classifier` ran over the finished buffer: for each row it computed
`(m, s)`, wrote the loss, and **overwrote the row in place** with `dlogits`.

That in-place overwrite is the crux, and phase 1's closing section named it
correctly as the obstacle. It is why `grad_act_sizes[logits] = 0` on GPU — there
is no separate gradient buffer for logits, because the gradient *is* the forward
buffer, rewritten. It is efficient and it is exactly what llm.c does.

It is also the thing that has to go, because a design that never materializes
the full logits has nowhere to do an in-place overwrite of the full logits.

---

## 3. The idea: carry a running (max, sum-exp) instead of the row

Here is the entire algorithm.

**Pass 1 (in `forward`).** For each vocabulary tile in turn: compute that tile's
logits into a small buffer, reduce it to that tile's own `(m_tile, s_tile)`, and
**merge** it into a running `(m, s)` carried per row. Discard the tile. After the
last tile, `(m, s)` is exactly what a single pass over the whole row would have
produced, and the loss follows.

**Pass 2 (in `backward`).** For each tile in turn: **recompute** that tile's
logits, use the finished `(m, s)` to convert it in place into that tile's slice
of `dlogits`, and immediately feed that slice into the two backward GEMMs for
that tile. Discard the tile.

The merge step is the only interesting part:

```
m' = max(m, m_tile)
s' = s · exp(m − m') + s_tile · exp(m_tile − m')
```

Read it as: "we are switching the reference point from `m` to `m'`; rescale the
sum I already have, rescale the sum you just brought me, add."

### 3.1 Why this is exactly right, not approximately right

Two claims, and both matter.

**It is algebraically identical.** `s` was defined as `Σ exp(x − m)` over the
tiles seen so far. Multiplying by `exp(m − m')` gives `Σ exp(x − m) · exp(m − m')
= Σ exp(x − m')`. Same for the new tile. So `s'` is `Σ exp(x − m')` over
everything seen so far — the definition, at the new reference point. Induction
over tiles gives the final `(m, s)` for the whole row. No approximation is
introduced at any step.

**It cannot overflow.** `m' = max(m, m_tile)`, so both `m − m'` and
`m_tile − m'` are **≤ 0**, so both `exp` factors are in **(0, 1]**. The rescale
can only shrink. The individual terms `exp(x − m_tile)` inside a tile are also
≤ 1 for the same reason. Nothing in the recurrence can exceed the magnitudes
already present in a single-pass computation.

That second point is worth dwelling on, because "we split a reduction into
pieces" usually *does* cost you something numerically. Here is the distinction:
splitting a **sum** reassociates it, and floating-point addition is not
associative, so you get a different rounding. Splitting a **max** does not — max
is associative and exact. And the sum being split here is a sum of positive
numbers, all in `(0, 1]`, which is the best-conditioned case there is: no
cancellation is possible, and the relative error of the sum stays on the order of
`n · ε` rather than blowing up. So the reassociation that does happen is benign.

Worked example. Take a row with three tiles whose maxima are 2, 11, and 5, and
suppose each tile's sum-exp against its own max is 4.0.

- After tile 1: `m = 2`, `s = 4.0`.
- Tile 2 arrives with `m_tile = 11`. `m' = 11`.
  `s' = 4.0 · exp(2 − 11) + 4.0 · exp(0) = 4.0 · 1.234e−4 + 4.0 = 4.000494`.
  The old contribution is *correctly* squashed: those logits really are ~9 units
  below the new maximum, so they really do contribute almost nothing.
- Tile 3 arrives with `m_tile = 5`. `m' = max(11, 5) = 11`, so the running sum is
  multiplied by `exp(0) = 1` — untouched — and the new tile is scaled down by
  `exp(5 − 11) = 2.479e−3`: `s'' = 4.000494 + 4.0 · 2.479e−3 = 4.010410`.

Final `(m, s) = (11, 4.010410)`, and `loss = log(4.010410) + 11 − logit[target]`.
A single pass over the concatenated row gives the same thing.

Now the failure mode the tests hunt for. If you *forget* the rescale in step 2 —
just add the sums — you get `s = 8.0` instead of `4.000494`, and the loss is too
high by `log(8.0 / 4.000494) = 0.693`. That is not a rounding error; it is a
0.69-nat error in a loss whose whole interesting range is a few nats. And it
only appears when a **later** tile raises the maximum, which for realistic data
happens on some rows and not others. Hence `test_late_max_forces_the_rescale`,
which plants the row maximum in the last tile so the branch must fire.

### 3.2 This recurrence was already in the repo

None of this had to be invented here. `llmm/softmax.mojo` already runs exactly
this merge in two places: across the lanes of a SIMD vector on CPU
(`softmax_phase_1_and_2_cpu`) and across the threads of a GPU block
(`softmax_phase_1_and_2_gpu`). A block reduction for a max-and-sum-exp *is* this
recurrence — that is how you combine per-thread partial results. The only new
thing is running it across **vocabulary tiles**, at a level above the kernel.

(The shared helpers are nonetheless not called from the new tile path, for a
mundane reason recorded in §7.)

---

## 4. Why the backward pass had to change

Phase 1's closing section predicted that this change would force the LM-head
forward and backward to **fuse into one non-separable operation**. That turned
out not to be necessary, and it is worth explaining why, because the reasoning
is the load-bearing part of the design.

The prediction was: a tile's `dlogits` is transient, so whoever produces it must
also consume it, so pass 2 and the backward GEMMs must live together — and since
pass 2 is part of the loss computation, which lives in `forward`, forward and
backward collapse into each other.

The first half is right. The second half does not follow, because of what pass 2
actually needs. Pass 2 needs, per row: the finished `m`, the finished `s`, the
target index, and `dloss`. That is **three floats and an int per row** — **384
KiB at B=32, T=1024, against the 6288 MiB the old design had to keep alive
across the same boundary**. It does *not* need the logits, because it recomputes
them.

So the split is:

- `forward` runs pass 1 and leaves behind the `(m, s, x_target)` carry.
- `backward` runs pass 2, which is where the tiles are consumed anyway.

`forward` and `backward` remain the same two separable calls they always were.
The producer and consumer of a `dlogits` tile are indeed fused — both are inside
backward's tile loop — but the forward/backward contract, gradient accumulation,
and the `recompute` flag are all untouched.

**This is worth stating plainly because the opposite is on the record.** Phase 1
identified the in-place `dlogits` write as *the* obstacle to chunking and
concluded that "LM-head forward and backward stop being separable", touching
"the forward/backward contract in `train_gpt2.mojo` and ... gradient
accumulation, the `recompute` flag, and the inference path". That was a
reasonable reading, and the cost it implied is the main reason phase 2 looked
expensive. It does not hold. The obstacle dissolves once you notice that a
`dlogits` tile's consumer was always going to be in backward anyway, so nothing
has to cross the forward/backward boundary except the summary — and the summary
is four numbers per row. Of the three things phase 1 expected to be disturbed,
gradient accumulation and `recompute` are untouched entirely, and the inference
path needs only a size reserve (§7.2), not a rewrite.

Why is recomputing legal? Because the two inputs to the LM head — `acts.ln_f`
(the final hidden states) and `params.wte` — are both still resident and both
unchanged between the two calls. Parameters do not move until `update()`.
`ln_f` is a persisted activation. So the recomputed tile is the same tile,
produced by the same GEMM on the same numbers.

### 4.1 What backward actually does now, per tile

```
for each vocab tile [t0, t0+oc):
    w_tile   = wte rows [t0, t0+oc)          # ZeRO-3: gathered on demand
    logits   = ln_f @ w_tileᵀ                # RECOMPUTE, into the one tile buffer
    dlogits  = (softmax(logits; m, s) - onehot) * dloss   # in place, this tile only
    d_ln_f  += dlogits @ w_tile              # accumulate: this axis IS split
    d_wte[t0:t0+oc] = dlogitsᵀ @ ln_f        # own rows: no accumulation needed
```

The two gradients behave differently under tiling, which phase 1 already worked
out and this inherits unchanged:

- `d_wte` for a tile reduces over **positions**, an axis tiling does not touch,
  and each tile owns a disjoint row range of `wte`. Exact.
- `d_ln_f` reduces over the **vocabulary**, which tiling does split. The first
  tile overwrites, the rest accumulate. This reassociates a sum and therefore
  differs from an untiled result by floating-point rounding — a property phase 1
  already introduced and measured, not something new here.

### 4.2 Why the vocabulary axis and not the batch axis

Liger chunks the **hidden states** — it splits the batch, projects each batch
chunk to the full vocabulary, and computes a partial loss. That also shrinks the
logits buffer (to `chunk_rows × V_p`), and it is a perfectly good choice. This
implementation splits the **vocabulary** instead. The reason is specific to this
codebase and worth stating, because "we did it differently from the paper we are
citing" deserves a justification.

Phase 1 already tiles the vocabulary, and it does so to shrink a *gradient*
bucket: under ZeRO-2/3 the LM head produces `d_wte` one vocabulary tile at a
time, reduce-scatters that tile onto the parameter shard, and recycles the pool
before the next tile. That works precisely because a vocabulary tile owns a
disjoint, contiguous row range of `d_wte`, complete the moment all positions have
been processed for that tile.

Chunk by batch instead and that property is lost: every batch chunk contributes
to **all** of `d_wte`, so the full 147 MiB gradient has to stay resident and
accumulate across chunks — undoing phase 1. Chunking by vocabulary keeps the two
decompositions aligned: one tile index drives the weight gather, the logits
slice, the loss reduction, the gradient slice, and the reduce-scatter. One loop,
one tile in flight.

### 4.3 The padding, per tile

The old classifier zeroed columns `[V, V_p)` of each row after writing dlogits,
so the backward GEMMs would read zeros there rather than garbage. That guarantee
has to survive, and it now has to be enforced **per tile**, since no tile can see
the whole row.

Each tile is given `valid_cols = clamp(min(oc, V - t0), 0, oc)`: the number of
its columns that are real vocabulary. Pass 1 reduces only over those. Pass 2
writes real gradients to those and **exactly zero** to the rest. At the
production shape exactly one tile is partial (the last one starts at 44800 and V
= 50257 falls inside it), and it is possible in general for a whole tile to be
past V, in which case `valid_cols` is 0 and the tile is absorbed as a no-op by
the merge — `(MIN_FINITE, 0)` merges into `(m, s)` without changing it, since
`max(m, MIN_FINITE) = m` and `exp(MIN_FINITE − m)` underflows to 0.

Both cases have tests (`test_dlogits_padding_is_exactly_zero`,
`test_all_padding_tile_is_absorbed`).

---

## 5. What it costs

Pass 2 recomputes every logits tile. Summed over the tiles, that is exactly one
extra LM-head forward GEMM per step:

```
2 · B·T · C · V_p  =  2 · 4096 · 768 · 50304  =  316 GFLOP   (at B=4, T=1024)
```

against a measured step of roughly 3.05 TFLOP. So the *arithmetic* predicts
about **+10%**. §6 reports what the clock actually said, which is the number
that matters.

There is also a second full read/write of one tile's worth of logits per tile
(the pass-2 conversion), but that traffic replaces traffic the single-pass
classifier was already doing, so it is close to a wash.

Nothing else is added. The `(m, s, x_target)` carry is 384 KiB. The per-tile
scratch is the shrunken `acts.logits` itself.

---

## 6. Measured, both configurations

Everything below is measured, not derived. Same machine, same binary source,
same data, same seed; the only difference is the `-D` flag.

**Setup.** GPT-2 124M (V = 50257, V_p = 50304, C = 768, L = 12), fp32, TF32
tensor cores on (the training default), ZeRO stage 0, world size 1, one NVIDIA
RTX PRO 6000 Blackwell. **B = 32, T = 1024** — the production shape, 32768
positions per step. 8 vocabulary tiles of 6400 (`LLMM_LM_HEAD_VOCAB_TILES`
default). 14 steps; the first 3 are discarded as warmup. Memory figures come
from the trainer's own exact in-process accounting (`LLMM_MEM_REPORT=1`,
`phase: steady`), which reads live buffer sizes rather than re-deriving them.

### 6.1 Memory

| | flag OFF | flag ON | change |
|---|---:|---:|---:|
| `logits` tensor | 6,593,445,888 B (6288 MiB) | 838,860,800 B (800 MiB) | **−5488 MiB, −87.3%** |
| `logits` gradient | 0 B | 0 B | — |
| `ce_stats_buf` (the carry) | 4 B (inactive stub) | 393,216 B (384 KiB) | +384 KiB |
| `acts_buf` total | 52,697,235,456 B (49.08 GiB) | 46,942,650,368 B (43.72 GiB) | −5.36 GiB |
| **exact process total** | 76,688,108,507 B (71.42 GiB) | 70,933,916,631 B (66.06 GiB) | **−5.359 GiB, −7.5%** |
| driver-reported device use | 86,436,216,832 B (80.50 GiB) | 80,799,072,256 B (75.25 GiB) | −5.25 GiB |

The saving is exactly `32768 × (50304 − 6400) × 4 B`, to the byte, which is the
whole point: the tensor went from all V_p columns to one 6400-column tile. The
carry is exactly `3 × 32768 × 4 B = 384 KiB`, also to the byte. **The state that
survives forward→backward shrank by a factor of 14,635.**

Note the driver figure moves by 5.25 GiB against an exact 5.359 GiB — the
difference is allocator quantization, which is why the exact accounting exists.

For context on what is left: after this change `logits` (800 MiB) is no longer
even the second-largest activation. `att_probs` is 19,327,352,832 B (18.0 GiB)
and `fch_gelu` is 4,831,838,208 B (4.5 GiB). Both are out of scope here and
owned elsewhere.

### 6.2 Step time

Two independent repetitions of each configuration. Median of steps 4–14.

| repetition | flag OFF | flag ON | change |
|---|---:|---:|---:|
| 1 | 431.11 ms | 457.54 ms | +26.43 ms (+6.13%) |
| 2 | 436.64 ms | 463.08 ms | +26.44 ms (+6.06%) |

Throughput, repetition 1: **76,128 → 71,699 tok/s**.

The absolute times drift ~1.3% between repetitions (thermals), which is why the
comparison is done pairwise. The *overhead* reproduces to 0.01 ms: **+26.4 ms
per step, +6.1%**.

### 6.3 The exchange rate, which is the actual result

> **5.36 GiB of device memory for 26.4 ms of step time — about 0.88 GiB per 1%
> of step time, or 203 MiB per millisecond.**

That is the number to decide on. It is a good trade if a larger batch is worth
more than 6% per step, and a bad one otherwise.

### 6.4 A third configuration: how far the knob goes

The tile width is `LLMM_LM_HEAD_VOCAB_TILES`, so the resident slice can be made
smaller. At **K = 64** (realized as 57 tiles of 896 columns — K is a request, not
a count, see §8):

| | OFF | ON, K=8 | ON, K=64 |
|---|---:|---:|---:|
| `logits` tensor | 6288.0 MiB | 800.0 MiB | **196.5 MiB** |
| exact process total | 71.42 GiB | 66.06 GiB | **65.47 GiB** |
| saved vs OFF | — | 5.36 GiB | **5.95 GiB** |
| median step (see caveat) | 431–437 ms | 458–463 ms | 464 ms |

196.5 MiB is not an arbitrary number: it is `1024 × 50304 × 4 B` — **the
inference reserve of §7.2, exactly**. At K=64 the vocabulary slice
(32768 × 896 × 4 = 117.4 MiB) has become *smaller* than the reserve, so the
reserve is now what sets the size. Turning K up further buys nothing. That is
the floor of this design, and removing it would mean teaching the sampler to
read something other than a full-width logits row — deliberately out of scope
here.

So the tensor can be taken from 6288 MiB to 196.5 MiB, a **96.9%** reduction,
for what looks like no additional time over K=8 — despite 57 tiles instead of 8,
i.e. 114 GEMM launches per step instead of 16. **Caveat:** the K=64 run was not
interleaved with a fresh baseline, and the session-to-session drift is ~1.3%, so
read "no additional time" as "within the noise floor of this measurement", not
as a precise zero.

### 6.5 Why 6.1% and not the predicted 10%

Phase 1 estimated ~+10% from the FLOP share, and that arithmetic was right:

```
recompute       = 2 · B·T · C · V_p = 2 · 32768 · 768 · 50304 =  2.53 TFLOP
analytic step   = (6N + 6·L·C·T) · B·T                        = 26.33 TFLOP
share                                                          =  9.6%
```

(The step figure is the trainer's own Kaplan-style count, the one its MFU
display uses.) What the estimate implicitly assumed is that the extra GEMM would
run at the *step's average* throughput. It does not:

```
baseline step:  26.33 TFLOP / 431.11 ms  =  61.1 TFLOP/s
the recompute:   2.53 TFLOP /  26.43 ms  =  95.8 TFLOP/s   ← 1.57× faster
```

The recompute is one large, well-shaped GEMM (M = 32768, N = 6400, K = 768).
The step average is dragged down by attention, layer norms and elementwise
kernels. 9.6% ÷ 1.57 = 6.1%, which is what the clock said. **The overhead is
cheaper than its FLOP share because of what kind of FLOPs they are.**

### 6.6 Numerical agreement, end to end

Same data, same seed, 14 steps at B=32, T=1024:

| step | flag OFF | flag ON | difference |
|---|---|---|---|
| 1 | 3.335548 (norm 3.0806) | 3.335548 (norm 3.0806) | **0 to all printed digits** |
| 5 | 3.510889 | 3.510880 | 9e−6 |
| 10 | 3.576466 | 3.576473 | 7e−6 |
| 14 | 3.466317 | 3.466341 | 2.4e−5 |

Step 1 is the meaningful one: before any parameter update, both paths see
identical inputs, and they produce an identical loss *and* an identical gradient
norm. From step 2 the two trajectories are computing on parameters that already
differ in the last bits, so they drift — but they drift at the 1e−5 level over
14 steps, against a loss of 3.4.

For scale: running the *baseline* twice gives 3.466317 vs 3.466310 at step 14
(7e−6 of pure run-to-run nondeterminism). So the chunked path's divergence is a
few times run-to-run noise, not a different answer. That is the expected
signature of a reassociated `d_ln_f` reduction, which phase 1 already introduced
and which this change does not make worse.

---

## 7. Two implementation notes that cost real time

### 7.1 Aligned loads and a tile's leading dimension

The obvious implementation of pass 1 is to call the existing
`softmax_phase_1_and_2_cpu` / `_gpu` helper on the tile, passing the tile's width
where the helper expects the padded vocabulary. Mathematically that is exactly
right, and it was the first thing tried.

It segfaults.

Those helpers issue **SIMD-width-aligned** vector loads at address
`row · leading_dimension`. That is a legal address only when the leading
dimension is a multiple of the SIMD width. For a full logits row the leading
dimension is `V_p = 50304`, which is a multiple of 16, so the existing code is
fine forever. For a **tile** the leading dimension is the tile width — a runtime
value with no such guarantee. Odd rows land on a misaligned address and an
aligned load faults.

The new kernels therefore do their own scalar, coalesced reduction rather than
reusing the helper. On GPU that costs some vectorization on a kernel that is
trivial next to the GEMM beside it; the alternative would be constraining tile
widths, which is a worse trade for a knob users are meant to turn.

#### A latent bug this uncovered, for whoever hits it next

The same property is a live landmine in code that predates this change:

> **`fused_classifier_cpu` (and `softmax_fwd_cpu`, and anything else routed
> through `softmax_phase_1_and_2_cpu`) segfaults when the padded vocabulary is
> not a multiple of the SIMD width.**

The row reduction issues loads aligned to `align_of[SIMD[float32, W]]()` — 64
bytes for W = 16 on AVX-512 — at address `row · V_p`. When `V_p` is not a
multiple of W, odd rows land on a misaligned address and the aligned vector load
faults. Reproduced directly: a six-row call at V = 29, V_p = 37 dumps core; the
new chunked kernels on the same data do not.

**It is unreachable from the trainer today** — V_p = 50304 = 16 × 3144, and the
activation buffers are device allocations — so this is not a regression and not
on the critical path. It will bite two kinds of person:

1. anyone who picks a **non-standard vocabulary** whose padding lands on a
   non-multiple of the SIMD width (nothing currently forces V_p to be padded to
   128, or to anything, at model-construction time), and
2. anyone who tries to **unit-test that function directly** at a small shape,
   which is exactly what was attempted here.

Consequence for this work: every reference in
`tests/test_chunked_cross_entropy.mojo` is float64 rather than a comparison
against the classifier being replaced. That is the stronger choice anyway — see
the §9.2 note — but it was forced, not chosen.

The fix, if someone wants it, is to drop the explicit `alignment=` from
`_softmax_comp_max`'s load, or to require the padded vocabulary be a multiple of
the SIMD width at construction. Not done here: it is a different file's
invariant and changing load alignment on the hottest read path in the classifier
deserves its own measurement.

### 7.2 Inference still wants the whole row

The chunked path only runs when there are targets, i.e. during training. The
generation path (`train_gpt2 -g`, `infer_gpt2`) calls `forward` with no targets
and then reads `acts.logits + row · V_p` to sample from — it wants a full-width
row, and it is not this change's business to rewrite the sampler.

So `acts.logits` under chunking is sized as the **larger** of one tile and an
inference reserve:

```
elements = max( B·T · tile_rows ,  min(B·T, max_seq_len) · V_p )
```

The reserve is bounded by `max_seq_len · V_p` = 206 MiB at GPT-2 124M, because
generation always runs at batch 1 with sequence at most `max_seq_len`. Crucially
it does **not** grow with the training batch size, which is why the saving still
lands at B=32 (6.1 GiB down to hundreds of MiB) and why it is zero when
`B·T ≤ max_seq_len` — at small shapes the reserve is the whole buffer and
chunking buys nothing. That is stated rather than hidden; it is the honest shape
of the trade.

A forward with no targets that would outgrow the reserve raises rather than
overrunning.

---

## 8. The knob

```
-D LLMM_LM_HEAD_CHUNKED_CE=1
```

**Default 0 — today's behaviour, byte for byte.** This project publishes
benchmarked step times, and a change that costs ~10% must not arrive silently
for users who are not memory-bound. `LLMM_FP8_STATIC_SCALES` is the existing
precedent for an opt-in trade of exactly this shape.

It requires more than one vocabulary tile to mean anything, so it composes with
phase 1's `LLMM_LM_HEAD_VOCAB_TILES=K` and degrades to a no-op at `K=1`. Larger
K makes the resident tile smaller, down to the inference-reserve floor. Note
(phase 1's finding, still true) that **K is a request, not a count**: tile widths
are rounded up to a multiple of 128, so K=32 yields 31 tiles, K=64 yields 57, and
K=128 yields 99.

When to turn it on: when activation memory is what caps your batch size, and a
bigger batch is worth more than 10% per step. When to leave it off: everything
else.

---

## 9. What was verified

### 9.1 Against PyTorch reference data — the important one

`test_gpt2.mojo gpu` (the `make verify-gpu` gate) runs the model against
`gpt2_124M_debug_state.bin`, a checkpoint plus PyTorch-computed activations,
loss and gradients. It was run **twice: once with the flag off, once with it
on**, both with `-D LLMM_NO_TF32=1` so the comparison is strict IEEE fp32.

With the chunked cross-entropy **on**:

- `OK (LOGITS)` — the logits match the PyTorch reference.
- `LOSS OK: 5.2700095 5.2678394` — the loss matches, and the value is
  **bit-identical to the flag-off run's**.
- **All 16 gradient tensors pass**, and they pass with essentially the same
  margin as the baseline. Per-tensor maximum deviation from PyTorch:

| tensor | maxdiff OFF | maxdiff ON | threshold | margin (ON) |
|---|---:|---:|---:|---:|
| `dwte` | 2.021e−2 | 2.021e−2 | 5.130e−1 | 25.4× |
| `dwpe` | 1.589e−4 | 1.589e−4 | 1.213e−2 | 76.3× |
| `dln1w` | 6.910e−3 | 6.911e−3 | 7.655e−2 | 11.1× |
| `dqkvw` | 1.317e−3 | 1.317e−3 | 2.209e−2 | 16.8× |
| `dattprojw` | 3.781e−4 | 3.782e−4 | 1.450e−2 | 38.3× |
| `dfcw` | 8.735e−4 | 8.734e−4 | 1.832e−2 | 21.0× |
| `dfcprojw` | 9.332e−4 | 9.333e−4 | 1.449e−2 | 15.5× |
| `dlnfw` | 2.609e−3 | 2.609e−3 | 1.635e−2 | 6.3× |
| `dlnfb` | 3.334e−4 | 3.334e−4 | 1.629e−2 | 48.8× |

(9 of 16 shown; the other seven — `dln1b`, `dqkvb`, `dattprojb`, `dln2w`,
`dln2b`, `dfcb`, `dfcprojb` — behave identically, all passing with 25–64×
margin.)

The two rows that matter most are `dwte` — the LM-head weight gradient, produced
by pass 2 in tiles — and `dlnfw`/`dlnfb`, which are downstream of `d_ln_f`, the
gradient that accumulates across tiles. All three agree with the baseline to
four significant figures. **The chunked path reproduces the baseline's agreement
with PyTorch, it does not merely stay inside a tolerance.**

**A note on the loss-trajectory half of that gate, and how it resolved.** When
this work was first gated, the same gate's 10-step overfit check failed on every
step — and it failed *identically on a clean `main`* with this change absent
(`step 0: loss 5.2700095 expected 5.3544273 diff=0.08441782`, digit for digit).
The diagnosis was that the hardcoded expected trajectory was the stale artifact,
not the trainer: our step-0 loss agreed with the debug-state file's own
forward-loss reference (5.2678394) which the gate accepts one line earlier.

That has since been **fixed on `main`** — the expected list is now regenerated
from PyTorch by a committed `scripts/gen_expected_losses.py` that refuses to emit
unless step 0 reproduces the debug-state reference. The diagnosis above was
correct: that script's own header records the same two numbers and notes the
gate "reported `LOSS MISMATCH` on all ten steps and exited non-zero on *every*
commit, including baselines predating any of the work it was blamed on."

`main` was merged into this branch before the final gate run, so the results in
§9.4 are measured against the corrected expectations.

### 9.2 Unit tests — `tests/test_chunked_cross_entropy.mojo` (new, 10/10 pass)

Every reference is float64, computed from the same fp32 inputs. Coverage:

- `test_loss_matches_float64_log_sum_exp` — loss against an exact log-sum-exp,
  targets placed in the first tile, on a tile boundary, mid-tile and in the
  ragged last tile.
- `test_late_max_forces_the_rescale` — the row maximum planted in the **last**
  tile, so the rescale branch must fire. Omitting the rescale is a ~0.69-nat
  error here, not a rounding difference.
- `test_early_max_needs_no_rescale` — the mirror case.
- `test_dlogits_match_float64_reference_everywhere` — every element of every row.
- `test_dlogits_padding_is_exactly_zero` — `[V, V_p)` compared against exact
  zero, not a tolerance.
- `test_dlogits_sum_to_zero_per_row` — the structural invariant from §1.3, which
  holds independently of any reference implementation.
- `test_tile_width_one_still_works` — 37 tiles of one column: the harshest test
  of the recurrence's seeding.
- `test_all_padding_tile_is_absorbed` — a tile lying entirely past `V`, so
  `valid_cols = 0` and an empty reduction's `MIN_FINITE` sentinel must not
  poison the running max.
- Two geometry tests that **assert the chunking actually happened** — including
  at the production geometry (V_p=50304, 8 tiles of 6400, `V` landing inside the
  last tile). Without these the suite could go green having tested a single
  tile, which is the failure mode phase 1 warned about.

The filler writes **1000.0** into the padding columns, so a reduction that leaks
past `V` changes the loss by hundreds of nats rather than by rounding.

### 9.3 End-to-end on GPU

- **Loss parity**, B=4 T=1024 and B=32 T=1024, 8 and 14 steps: step 1 identical
  to all printed digits in both loss and gradient norm; later steps within a few
  times run-to-run nondeterminism (§6.6).
- **Generation**, the path an LM-head change breaks first and which phase 1
  recorded as doubly ungated: run via the trainer's own sampler (`-s 1 -g 48`),
  three steps, flag on and flag off. Both produce normal early-training text;
  the step-1 sample is identical between them. No crash, no degenerate collapse.
- **Memory accounting**: the new `ce_stats_buf` was added to the exact
  in-process report, and shows correctly as an *inactive* 4-byte stub with the
  flag off and as exactly 384 KiB with it on.
- **Regression**, flag off: `tests/test_lm_head_vocab_tiling.mojo` (phase 1's
  suite, which covers the two tile GEMMs the packed fast path touches).

### 9.4 Gates

Green: `make format` (zero residual diff), `make lint`, `make check`. Both
configurations — flag off and flag on — compile with zero compiler diagnostics.

**Outstanding: the full `make test`.** It was not run, because the machine was
under a load average of 100 at the time and the standing rule is to get
clearance before starting a suite there. Its two Mojo components were run
individually and both pass — `tests/test_chunked_cross_entropy.mojo` (10/10) and
`tests/test_lm_head_vocab_tiling.mojo` (10/10, the regression check on the
`llmm/matmul.mojo` edit). What remains unrun is `tests/test_zero*.mojo` and the
Python suite.

The residual risk there is low but not zero, and worth stating precisely: with
the flag off, every chunked code path is `comptime`-dead, so the default build
is behaviourally identical to `main`. The one edit that is *not* flag-gated is
the packed fast path in the two tile GEMMs, and it only engages when the
caller's leading dimension equals the tile width — which never happens on the
default path (there the leading dimension is V_p = 50304 and the tile width is
6400) and does not happen in phase 1's suite either. That suite passing 10/10 is
the direct evidence for it.

One coverage gap worth recording for whoever maintains the smoke subset:
`docs/ai/pre_merge_smoke_subset.md` already warns that nothing in it drives the
LM head at vocabulary scale. This change does not close that gap — the new test
file is in `make test-mojo` but not in `SMOKE_MOJO`. It runs in 3.2 s, so adding
it there is cheap if the subset is ever revised.

---

## 10. Files touched

- `llmm/fused_classifier.mojo` — new: `_chunk_merge`, `chunked_ce_pass1`
  (+ `_cpu`/`_gpu`), `chunked_ce_loss` (+ `_gpu`), `chunked_ce_pass2`
  (+ `_cpu`/`_gpu`). Nothing existing was modified.
- `llmm/matmul.mojo` — packed fast path in `matmul_lm_head_fwd_tile` /
  `matmul_lm_head_bwd_tile`: when the caller's leading dimension already equals
  the tile width (which is how the chunked path hands its scratch over), the
  portable staging buffer and its gather/scatter copy are skipped. The strided
  path is unchanged.
- `train_gpt2.mojo` — the `LLMM_LM_HEAD_CHUNKED_CE` knob; `_chunked_ce()`,
  `_ce_stats_ptr()`, `_logits_elems()`; the `ce_stats_buf` carry and its line in
  the memory report; a chunked branch at the forward LM-head site and at
  backward's Bucket 1.
- `tests/test_chunked_cross_entropy.mojo` — new.

**Merge points shared with other workstreams.** `_grad_pool_elems()` is
**untouched** by this change (the chunked path recycles the same per-tile bucket
phase 1 established). The memory report gained one buffer line, which is
additive.

---

## AI use statement

Written with AI assistance (Claude Opus agent via Claude Code), directed by
Evan. The implementation, tests, and measurements in this document were produced
by the same agent; the numbers in §6 are machine-measured, not estimated, and the
estimate they are compared against is labelled as an estimate.
