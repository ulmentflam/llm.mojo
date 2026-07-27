# De-residenting the tied `wte` embedding: campaign synthesis

*2026-07-27. Six workstreams, README roadmap item 1. This is the overview; §10
maps every workstream to its own document, and this one links to them rather
than repeating them. (The row-sparse encoder work of §5 is the one that has no
standalone write-up, so it is covered here in more detail than the rest.)*

The roadmap item reads:

> ZeRO: de-resident the tied `wte` embedding (vocab-chunked LM head + indexed
> encoder gather) to push stages 2/3 below the current ~150 MiB floor.

This document is written for someone who knows Python and basic neural
networks and nothing about distributed training. Every term is defined at first
use. It assumes you have read none of the other documents.

**The short version.** One tensor — GPT-2's token embedding table — was pinning
two ~150 MiB buffers that no amount of extra GPUs could shrink, because the
tensor is used twice, in two incompatible ways, and the two uses needed
different fixes. Both fixes shipped, and the two floors fall to roughly a sixth
and a seventh of their previous size — though half of that is still behind a
default-off flag (§10), which this document flags rather than rounds away. A
third approach, the obvious one from the literature, was modelled carefully and
rejected. And another workstream built the measuring instrument that made any of
the above provable, because the instrument we started with could not see a
150 MiB change **at all**.

**Read this section too, or you will over-read the rest.** At the batch shape
this project actually trains at, the buffers this campaign shrank are dwarfed by
two *activation* tensors, 42× and 123× bigger, that no workstream here touched.
Section 8 gives the numbers. A
reader who finishes this document believing the memory problem is solved has
been misled.

---

## 1. The words you need

Skip ahead if you know them.

**The four kinds of training memory.** Fitting a model on a GPU means fitting
four different things, and they scale differently:

| Kind | What it is | GPT-2 124M, fp32 |
| --- | --- | --- |
| **Parameters** | the weights | ~475 MiB |
| **Gradients** | one number per parameter, produced by the backward pass | ~475 MiB |
| **Optimizer state** | AdamW's two running averages per parameter | ~950 MiB |
| **Activations** | intermediate values saved in forward for use in backward | depends on batch and sequence length |

The first three are *model-proportional* — they depend only on the parameter
count. The fourth is *batch-proportional*. Hold onto that distinction; §8 turns
on it.

Two number formats appear throughout. **fp32** is the ordinary 32-bit float, 4
bytes. **bf16** is a 16-bit float, 2 bytes, used for the weights and activations
while the optimizer keeps fp32 copies — so "half precision" halves some buffers
and *adds* others, and buys considerably less than half. Both appear in the
tables below because the campaign measured both.

**Data parallelism.** The simplest way to use N GPUs: put a complete copy of the
model on each, give each a different slice of the batch, and average everyone's
gradients before the weight update so the copies stay identical. Each
participating process is a **rank** (0..N−1); N is the **world size**. In this
repo the world size is a compile-time constant and the ranks are host threads in
one process, one GPU each.

Data parallelism's weakness is obvious once stated: every GPU stores the same
parameters, the same gradients and the same optimizer state. With 7 ranks you
are storing 7 identical copies of a 950 MiB optimizer state.

**Sharding and ZeRO.** **Sharding** means splitting a buffer into N pieces and
giving rank *r* only piece *r*. **ZeRO** ("Zero Redundancy Optimizer") applies
that to the redundant categories above, in three cumulative stages:

| Stage | Sharded | Each rank still holds in full |
| --- | --- | --- |
| **1** | optimizer state | parameters, gradients |
| **2** | + gradients | parameters |
| **3** | + parameters | nothing — gathered on demand |

The ZeRO paper describes "three main optimization stages ... which correspond to
the partitioning of optimizer states, gradients, and parameters"
([Rajbhandari et al. 2019, §1](https://arxiv.org/abs/1910.02054); the sentence
is in the [full text](https://ar5iv.labs.arxiv.org/html/1910.02054), not the
abstract page). Each stage trades communication for memory.

**Collectives.** An operation every rank calls together. Three matter here:

- **all-reduce** — sum a buffer across ranks; everyone ends with the total.
- **reduce-scatter** — sum across ranks, but rank *r* keeps only slice *r*. This
  is how a sharded gradient is produced.
- **all-gather** — the inverse: each rank contributes its slice, everyone ends
  with the concatenation. This is how ZeRO-3 briefly materializes a parameter.

Because every rank must call them in the same order with the same arguments,
collectives are where distributed bugs hide. §5 contains one that does not
crash, does not hang, and silently returns wrong numbers.

**The symbols**, used throughout and in every formula below:

| | |
| --- | --- |
| `B` | batch size — sequences processed per step |
| `T` | sequence length — tokens per sequence. So `B·T` is tokens per step |
| `C` | channels: the width of the model's internal vector, 768 here |
| `L` | number of transformer layers, 12 here |
| `NH` | number of attention heads per layer, 12 here |
| `V_p` | padded vocabulary size, 50304 here |
| `N` | world size — the number of ranks |

A trailing `· 4` in a size formula is bytes per fp32 number.

**GEMM** stands for *general matrix multiply* — the BLAS name for the operation
`D = A·B`, and by extension for the highly tuned library routines that implement
it. Nearly all of a transformer's arithmetic is GEMMs.

**Resident** means "currently occupying GPU memory". To **de-resident** a tensor
— the word in this campaign's title — is to arrange that it is not all in GPU
memory at once, either by holding one piece at a time or by never gathering the
parts you do not need. It is not the same as making a tensor smaller; the tensor
is unchanged.

**ZeRO-3's key trick** is that a parameter only needs to be resident *while a
kernel is using it*. So the trainer all-gathers one transformer layer's weights
into a small reusable **window** buffer, runs that layer, and reuses the window
for the next. Peak parameter memory becomes `shard + one window` instead of the
whole model — provided every tensor can be gathered in pieces small enough to
make the window small. That proviso is this campaign's entire subject.

---

## 2. One tensor, two access patterns

GPT-2 has an embedding table `wte` ("weight token embedding") with one row per
vocabulary entry and one column per model channel:

```
wte : [V_p, C] = [50304, 768]
```

`C = 768` is the model's internal vector width. `V_p` is the **padded**
vocabulary size — GPT-2's real vocabulary is 50257 tokens, rounded up to
50304 = 128 × 393 because GPU matrix kernels are faster on dimensions that are
multiples of 128. The 47 extra rows exist purely for alignment.

```
50304 × 768 = 38,633,472 elements = 154,533,888 bytes = 147.4 MiB in fp32
```

It is the largest single tensor in the model, and it sits at flat offset 0 of
the parameter vector.

**It is used twice, and the two uses are the same numbers.** This is called
**weight tying** (Press & Wolf 2016, [arXiv:1608.05859](https://arxiv.org/abs/1608.05859);
Inan et al. 2016, [arXiv:1611.01462](https://arxiv.org/abs/1611.01462)), and it
is standard for GPT-2 because it saves 147 MiB of parameters and improves
quality:

1. **The encoder** reads it **sparsely**. Given token id 8721, fetch row 8721 —
   a 768-vector — as that token's input representation. A batch of `B×T` tokens
   touches at most `B×T` distinct rows out of 50304.
2. **The LM head** reads it **densely**. After the transformer stack produces a
   768-vector per position, multiply by `wte` transposed to score every
   vocabulary entry: `logits[B*T, V_p] = ln_f[B*T, C] @ wteᵀ[C, V_p]`. As
   written, this was one matrix multiply spanning the entire vocabulary, so it
   needed every row at once.

**This is the intellectual core of the campaign, so it is worth stating
flatly: any technique that fixes one access pattern does nothing for the
other.** Chunking the dense matrix multiply over the vocabulary makes the LM
head need only one slice of `wte` at a time — and leaves the encoder still
asking
for arbitrary rows spread across the whole table. Gathering only the batch's
token rows shrinks the encoder to the rows the batch actually touches — and
leaves the LM head still needing all 50304 rows to produce all 50304 logits, no
matter which tokens are in the batch. Two problems wearing one tensor's
clothing. That is why this needed two independent solutions rather than one, and
it is why the two had to land in a particular order (§10).

**Why *tied* is harder than either use alone.** An untied model has two separate
matrices, and each can be given whatever layout, gather schedule and gradient
treatment suits its own access pattern. Tying removes that freedom three times
over:

- **One layout must serve both.** You cannot lay one copy out to suit the
  matrix multiply and another to suit row lookups. It is one buffer.
- **It is live from the first op of forward to the last op of backward.** The
  encoder runs first in forward; the LM head runs last in forward and *first* in
  backward; `encoder_bwd` runs last in backward. Under ZeRO-3, whatever window
  holds `wte` must stay alive across essentially the entire step, which is
  exactly the residency that streaming exists to avoid.
- **The two gradient contributions land in the same rows.** Backward writes
  `d_wte` at both ends and the results must be summed, so the gradient is not
  final until the very end. The existing scheme handles this by **bucketing**:
  rather than allocating gradient space for the whole model at once, backward
  keeps one small reusable **pool**, computes one *bucket* of gradients into it,
  reduce-scatters that bucket onto the persistent shard, and recycles the pool
  for the next bucket. `wte`'s two contributions become two buckets — the
  LM-head one early, the encoder one late — which works because reduce-scatter
  is linear: `RS(a) + RS(b) = RS(a+b)` accumulated on the shard. That works,
  but it means both moments need pool space for `wte`.

**The two floors, measured exactly.** The prior work
([gradient bucketing](zero_grad_bucketing_design_2026-07-14.md),
[ZeRO-3 parameter streaming](zero_stage3_param_streaming_2026-07-14.md)) both
bottomed out on the same tensor:

| Buffer | fp32 | What it is |
| --- | --- | --- |
| `grad_pool_buf` | **150.375 MiB** | the gradient bucket pool, sized by its largest bucket — `wte + wpe` |
| `embed_window_buf` | **150.381 MiB** | the ZeRO-3 parameter gather window for `wte`, `wpe` and the final layernorm |

(`wpe` is the 1024×768 position-embedding table, 3.0 MiB; the final layernorm's
two vectors add 6 KiB, which is the entire difference between the two numbers.)

Neither number moves with world size, batch size or extra GPUs. It is a
property of the model, not of the parallelism — which is precisely what makes it
a *floor* rather than a cost.

---

## 3. The instrument came first

**Lead with this, because nothing else in the campaign is trustworthy without
it.** →
[`zero_memory_accounting_2026-07-27.md`](zero_memory_accounting_2026-07-27.md)

The benchmark measured peak GPU memory the obvious way: poll
`nvidia-smi --query-gpu=memory.used` during the run and keep the maximum. That
is a faithful measurement of *what the process costs the machine*, which is the
right answer to "will this job fit on a card". It is the wrong answer to "did
this buffer get smaller", for a reason worth understanding.

Asking the GPU driver for memory is slow — hundreds of microseconds — and a
training step allocates constantly. So essentially every GPU framework puts a
**caching allocator** in between: it takes a large chunk from the driver once,
hands out slices, and when the program frees a slice the allocator keeps the
chunk. `nvidia-smi` sees chunks, not slices. Two consequences: the reading is
chunk-granular, and it only ratchets upward.

The measurement workstream pinned both empirically. Across **all 16** baseline
runs, every in-process `driver_used_bytes` reading was an exact multiple of
**256 MiB** — the observed values were 1792, 2304, 2560, 3072, 11776, 12544,
22016, 22272 and 22784 MiB, i.e. 7, 9, 10, 12, 46, 49, 86, 87 and 89 chunks. Not
one reading fell off the grid. And the `nvidia-smi` figure sat a **near-constant
~696 MiB** above it — exactly 696 in the eight `-b 4 -t 64` rows, 696 or 698 in
the larger shape. ("Near-constant" is the honest word: it is 696 or 698, not
696.) That is the CUDA context's own fixed overhead, which belongs to the
process but to no buffer. Empirically:

```
nvidia-smi peak delta  ≈  roundup_256MiB(bytes actually needed)  +  ~696 MiB
```

Read the formula and the failure stops being bad luck. **The instrument's
resolution is 256 MiB. The buffers this campaign exists to shrink are
150.375 and 150.381 MiB.** Nothing below 256 MiB is resolvable that way *by
construction* — and when a sub-chunk change does happen to straddle a boundary,
the reported step is the size of the chunk, not the size of the change.

The instrument failed in both directions, which is why "it under-reports, so
treat it as a lower bound" would also have been wrong:

| Change | `nvidia-smi` | Exact |
| --- | --- | --- |
| fp32, stage 2 → 3 | **0 MiB** | −60.000 MiB |
| bf16, stage 1 → 2 | **0 MiB** | −43.522 MiB |
| bf16, stage 2 → 3 | **0 MiB** | −30.000 MiB |
| fp32, stage 1 → 2 | −256 MiB | −87.044 MiB (**2.9× over-report**) |

The first three rows are the failure everyone expected: real reductions reported
as exactly zero. The fourth should worry you more. It is not a noisy version of
the truth that averaging would recover; it is a *quantized* version, and
quantized measurements mislead in both directions.

**The fix is to ask the program, not the driver.** Every long-lived GPU
allocation is held in a `DeviceBuffer` field, so `len(buf) * size_of[dtype]()`
is an exact byte count with no quantization at all. `LLMM_MEM_REPORT=1` now
makes each rank emit a JSON line per phase with every buffer's exact size. Three
design choices make it trustworthy, and all three are transferable:

- **It reads the live buffers rather than recomputing sizes from the same
  expressions the allocations used.** A copy of an expression stops matching the
  moment the original moves — and this instrument exists specifically to measure
  *other people's* edits to those expressions.
- **It reports at two phases.** `post_alloc` (before the first forward, so
  activations do not exist yet) isolates the model-proportional footprint;
  `steady` (after the last step) shows the real total. Reporting only one would
  flatter or bury the result depending on which you picked.
- **It distinguishes "eliminated" from "small".** Unused buffers are left at a
  1-element placeholder, so a byte-delta would render an outright elimination as
  a meaningless ~4-byte change. An `inactive_buffers` list names them, and a
  name appearing in the "after" run but not the "before" is a structural win —
  a different and generally better result than a shrink.

It reproduces: the `-b 4 -t 64` sweep was run twice hours apart, once on a quiet
box and once with three other workstreams hammering other GPUs, and all eight
exact totals came back **identical** — not close, identical, to the last of the
three decimal places the report prints in MiB.
That is why this campaign quotes 30 MiB differences without error bars — and
also why it makes no throughput claims from those runs (§10).

The rule the campaign adopted: report the exact accounting for the buffer that
changed, report `nvidia-smi` for whether the job got easier to fit, and **never
quote a sub-256 MiB difference between two `nvidia-smi` readings as a result.**

---

## 4. The dense side: a vocab-tiled LM head

→
[`zero_lm_head_vocab_tiling_2026-07-27.md`](zero_lm_head_vocab_tiling_2026-07-27.md)

The idea is the one you would guess. Instead of one matrix multiply producing
all 50304 logits, do K of them: score the first 6400 vocabulary entries, then
the next 6400, and so on. Each tile needs only its own rows of `wte`, so
backward can hold, reduce-scatter and recycle one tile's gradient rows at a
time.

```
-D LLMM_LM_HEAD_VOCAB_TILES=<K>      # default 8
```

At GPT-2 124M that realizes 8 tiles: seven of 6400 rows and a ragged final one
of 5504 (7 × 6400 + 5504 = 50304). `K = 1` restores the exact previous code
path — the escape hatch.

### The obstruction was purely layout

`logits` is stored **row-major** with shape `[B*T, V_p]`: element `(r, v)` lives
at flat offset `r * V_p + v`. A vocabulary tile is therefore a **column
slice** — all rows, columns `t0 .. t0+w` — and its elements are *not*
contiguous. Each of its rows is `w` numbers long, but the distance from one
row's start to the next is `V_p`.

That distance is the matrix's **leading dimension** (`ld`): the stride, in
elements, from the start of one row to the start of the next. For a standalone
matrix `ld` equals the width. For a slice carved out of a bigger matrix, the
width is the slice's and the `ld` stays the *parent's*:

```
logits (row-major, ld = V_p = 50304)
┌──────────────────────────────────────────┐
│ ....... [  w  ] ......................    │  row 0   ← tile starts here
│ ....... [  w  ] ......................    │  row 1   ← ...50304 elements later
│ ....... [  w  ] ......................    │  row 2
└──────────────────────────────────────────┘
          ↑ t0        width = 6400,  ld = 50304
```

Contrast the *weight* side. A vocabulary tile of `wte[V_p, C]` is a contiguous
**row** range — one unbroken run of memory starting at `wte + t0*C`, leading
dimension still `C`. **A row tile of a row-major matrix needs nothing special; a
column tile does.** That asymmetry is the whole story: `wte` tiles for free,
`logits` does not.

Every matrix-multiply entry point in the repo built its tensors with a
constructor that *assumes* `ld == width`. There was no way to express the
strided slice. On the production path the fix was small — cuBLASLt (NVIDIA's
vendor GEMM library) takes an explicit leading dimension as an argument to every
layout descriptor, so a vocabulary tile is **zero-copy**: a pointer offset plus
one integer. Every other target (CPU, Apple Metal, vendor-neutral GPU)
**stages** instead: copy the tile to or from a contiguous scratch buffer and
call the
existing dense kernels completely unchanged. That is one `O(rows × w)` copy next
to an `O(2 × rows × w × C)` multiply — with `C = 768`, under a thousandth of the
arithmetic — bought in exchange for not touching three working code paths.

### Numerics: which sums get reassociated, and which do not

Splitting a sum and adding the partial sums back is not bit-identical in
floating point, because floating-point addition is not associative. Which
quantities that touches follows mechanically from which axis each one sums over:

- `logits` (forward) sums over `C` — an axis tiling does not touch. Each logit
  is still one full-length dot product over the same values.
- `d_wte` sums over `B*T` — likewise untouched. The vocabulary index is a *free*
  index here, labelling which output row you are computing. So **the tiles
  partition the output**: each tile's rows are complete and final the moment
  they are computed, which is exactly why they can be reduce-scattered and the
  pool recycled immediately.
- `d_input` (the gradient flowing back into the transformer stack) sums over the
  **vocabulary** — the axis tiling splits. Every tile contributes to every
  element, so tiles must accumulate rather than overwrite. This is the one
  quantity tiling perturbs.

One of the three is theoretically at risk and two are not, and a tile-count
sweep matched: **bit-identical to untiled for tile counts 1 through 64**,
diverging only at the largest count tried, 128. (The divergence is consistent
with the GEMM's internal blocking changing once the tile gets narrow enough, but
the sweep recorded the fact, not the cause.)

That sweep is a *bit-identity* check and is separate from the correctness
verification, which was done at **four** tile counts — K = 2, 4, 8, 16 —
producing widths 25216, 12672, 6400 and 3200 — each with a *different* ragged
tail (25088 / 12288 / 5504 / 2304), which is the shape of bug a single divisible
tile count would hide. This
matters more than it sounds: before this work, **no test in the repo drove the
LM-head matmul above `output_channels = 3072`**, while production runs it at
50304 (§9).

**Cost: +0.60% step time** — 56.99 → 57.33 ms/step at `b=4 t=1024`, ZeRO stage
0, single GPU, median of steps 2–12, two runs each.

### What it bought

| | Before | After | Status |
| --- | --- | --- | --- |
| LM-head gradient bucket | 147.4 MiB | **18.8 MiB** | on by default, measured |
| `embed_window_buf` (ZeRO-3) | 150.4 MiB | **21.8 MiB** (window 3.0 + one tile 18.8) | **requires `LLMM_Z3_WTE_ONDEMAND=1`, which is off by default and has never been run end to end** (§10) |

At 18.8 MiB the LM-head bucket is now *below* the 27.0 MiB per-transformer-layer
bucket, so the LM head is no longer a binding constraint on the pool at all. But
the pool is `max()` over all buckets, and on the tiling branch alone the encoder
bucket still asked for 150.4 MiB — so the *allocation* did not move until the
encoder work landed. The tiling document says so explicitly and does not claim
"147 MiB saved". That restraint is the reason the campaign's numbers can be
trusted.

---

## 5. The sparse side: a row-sparse encoder gradient

*No standalone document — this section is the write-up.*

An embedding gradient is naturally sparse. Row *v* of `d_wte` receives gradient
only if token id *v* appears somewhere in the batch. At `B·T = 32768` tokens
against a 50304-row table, most rows get exactly zero — and the old code
reduce-scattered all 147 MiB of them anyway. Gathering and reducing only the
rows that were touched is the obvious win.

Two things made the obvious version wrong.

### 1. Cost: the collectives charge per range, not per byte

The range-based collectives issue **one driver-staged copy per range**, and each
copy costs a flat ~15 µs regardless of how few bytes it moves. So the same bytes
cost wildly different amounts depending on how they are described:

| Description of 32768 rows | Cost |
| --- | --- |
| **one** contiguous range | 1.99 ms |
| **32768** single-row ranges | **552 ms** |

Identical bytes, a **277×** penalty for the phrasing. That is a microbenchmark
of the collective alone. Measured end to end inside the trainer, the naive
one-range-per-row implementation cost **7.1 ms** against the **2.56 ms** dense
gather it
was replacing — *a slowdown shipped as an optimization*, and it would have
passed every correctness test in the repo.

The fix is **coalescing**: merge runs of rows separated by fewer than ~290
rows into one range, accepting some unwanted bytes in exchange for not paying
another fixed 15 µs. It is the same trade a disk scheduler makes.

The general lesson is worth keeping: when a primitive has a large fixed
per-invocation cost, the *representation* of a request is a first-class
performance decision, and "fewer bytes" and "faster" are not the same objective.

### 2. Correctness: divergent range lists do not hang, they lie

This one is nastier. `reducescatter_buckets` takes a list of flat ranges from
the caller and applies those offsets to **every peer's** pool — that is, rank 0
reaches into rank 1's buffer using offsets from *rank 0's* list. The contract
is that every rank passes identical lists.

Ranks in data parallelism see **different tokens**. So a range list derived from
"which rows did my tokens touch" is different on every rank, by construction.

You might hope this deadlocks quickly and loudly. It does not. A **barrier** is
a synchronization point every rank must reach before any may continue, and a
mismatch in how many times each rank hits one is the usual way a distributed bug
announces itself as a hang. But on all four code paths the barriers sit
*outside* the per-range loop, so barrier counts cannot diverge with list
length. The ranks synchronize perfectly and read the wrong
rows. No crash, no hang, no shape error — just wrong gradients, which look like
a slightly worse learning curve.

The fix is to make every rank's list derive from the same data: OR the token
bitmaps across ranks first, so each rank's ranges are computed deterministically
from the **union** of all ranks' tokens. Every rank then walks an identical list
because it is a function of shared input.

And because "documented but unenforced" is how this class of bug survives,
`assert_ranges_agree` now checks it: hash the local range list, sum the hashes
across ranks, and verify the total equals N × the local hash. A cheap collective
that converts a silent corruption into a loud error. (The vocab-tiled LM head is
safe here for a different reason — its tile boundaries are computed from
`padded_vocab_size` and a compile-time constant, never from batch content, so
every rank walks the same sequence by construction.)

### What it bought

The `wte` term in the gradient pool drops from **147 MiB to 24 MiB**. With the
LM-head bucket already at 18.8 MiB and the per-layer bucket at 27.0 MiB, the
pool's `max()` is no longer set by the embedding at all.

Taking the two workstreams together, against the exact baselines in §2 — noting
that the second row needs the encoder work, the tiling work *and* the
default-off
flag, and so is the one number here that has not been observed end to end:

| Floor | Before | After | |
| --- | --- | --- | --- |
| `grad_pool_buf` | 150.375 MiB | ~27 MiB | −82% |
| `embed_window_buf` | 150.381 MiB | 21.8 MiB | −85% |

---

## 6. The road not taken: Megatron-style vocab-parallelism

→
[`vocab_parallel_lm_head_feasibility_2026-07-27.md`](vocab_parallel_lm_head_feasibility_2026-07-27.md)

There is a well-known technique that splits the vocabulary across GPUs, and any
reader who knows the literature will ask why it was not simply used. It was
modelled in detail and **rejected on technical grounds**. The reasoning is the
most reusable part of this campaign, so it gets its own section.

**What it is.** Megatron-LM (Shoeybi et al. 2019,
[arXiv:1909.08053](https://arxiv.org/abs/1909.08053)) is **tensor-parallel**:
different GPUs hold different *parameters* of the same layer. Its §3 says: *"We
parallelize the input embedding weight matrix E_{H×v} along the vocabulary
dimension E=[E_1,E_2] (column-wise)"*, and then *"we fuse the output of the
parallel GEMM [Y_1,Y_2] with the cross entropy loss which reduces the dimension
to b×s"* — send scalar losses between GPUs instead of a full logits matrix.
(Both quotes are in the [full text](https://ar5iv.labs.arxiv.org/html/1909.08053),
not the abstract page. The implementation is
[`megatron/core/tensor_parallel/cross_entropy.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/cross_entropy.py).)

**Why it does not transplant.** Megatron's tensor-parallel ranks all work on the
*same* batch — each holds a different slice of every layer and they cooperate on
one forward pass. (The paper never states this in one sentence; it is implied by
Appendix B.2's "we seed the random number generators at the beginning of
training with the same seed. This results in identical dropout patterns across
all model parallel workers", and by its choice to "duplicate the computation
across GPUs" for dropout, LayerNorm and residuals. Treat it as implied and
cited, not quoted.) So when rank *r* needs the hidden activations to score its
vocabulary slice, **it already has them** — activation replication is free, a
side effect of the parallelism it already chose.

This trainer is **data-parallel**: every rank has a *different* batch. To score
its vocabulary slice against every rank's tokens, a rank would have to
all-gather the `ln_f` activations from all the others. Megatron gets replication
free; we would have to manufacture it, and pay for it, every **micro-step** —
one forward/backward over one slice of the batch, several of which may be
accumulated before a single weight update.

That single difference is the whole argument, and it has a clean structural
form worth stating on its own:

> **Under data parallelism, vocab-parallelism minus the activation all-gather
> simply *is* vocab tiling.** They are two ends of one trade. Splitting the
> vocabulary across *ranks* costs an activation gather and saves nothing else
> that splitting it across *time* does not already save.

The exchange rate is total tokens against the size of the gathered window.

**The premise fails here for a second, more basic reason.** ZeRO shards by
**flat offset** — it cuts the parameter vector, all 124,475,904 elements of it
laid end to end, into N equal pieces, paying no attention to where one tensor
stops and the next starts. `wte` occupies the first 38.6M of those elements. So
at 7 ranks (this box's usable GPU count — see the steelman below) the shard is
17,782,272 elements, and **four of the seven ranks hold zero vocabulary rows at
all**: `wte` is used up two and a bit ranks in. A scheme whose premise is "each
rank owns a vocabulary slice" has no slice to own on most ranks.

At other world sizes it fails a second way — the cut lands *inside* a row. At
N=8 the shard is 15,559,488 elements = **20,259.75 rows**, so rank 1's slice
begins three-quarters of the way through a vocabulary entry. N=7 happens to
escape this (768 divides 17,782,272 exactly, giving a clean 23,154 rows per
rank), which is luck, not design.

### The exchange rate, exactly

The crossover — the point below which vocab-parallelism's cheaper logits
communication outweighs its extra activation gather — is:

```
break-even = W / (C + 3)
           = 39,421,440 / 771
           = 51,130 global tokens per micro-step
```

where `W = C · (V_p + 1026) = 39,421,440` is the size of the gathered embedding
window and the `+3` beside `C` is the three per-token all-reduces the fused
cross-entropy performs.

The natural rule of thumb is `V_p` — "it wins below 50,304 tokens" — and that is
**1.64% low relative to `V_p`**, for two reasons that pull opposite ways: `W` is
slightly larger than `V_p · C` because the window also carries `wpe` and the
final layernorm
(pushing the crossover right), while the `+3` accounts for the per-token
reductions (pushing it left).

**And now the result that actually settles it.** The communication ratio reduces
to

```
ratio = global_tokens · (C + 3) / W
```

**The world size cancels entirely.** N appears only inside `global_tokens =
N·B·T`. The answer depends neither on how many ranks you have nor on how the
tokens divide between batch size and sequence length — only on the *total tokens
processed per micro-step across the whole job*. All 20 modelled shapes, spanning
four different world sizes, fall on a single straight line.

That upgrades the finding from "vocab-parallelism loses on our cluster" to
**"vocab-parallelism loses at our token count, on any cluster"** — and it
forecloses the rescue a reader instinctively reaches for, because no choice of
world size can move the answer. At the production shape (B=32, T=1024, N=7 →
229,376 tokens) the ratio is **4.486×** — a communication regression of nearly
four and a half times.

### The steelman, and the honest limit on it

Rejecting the weakest version of an idea proves nothing. So the analysis
generalized: let **grouped** vocab-parallelism have a free parameter `g`, the
number of ranks sharing a vocabulary split, and compare across all of it.

- At **every** value of `g`, vocab tiling reaches the **same residency at 1.00×
  communication**, where vocab-parallelism costs between **1.15× and 5.13×**.
  There is no crossover to find; it is pointwise dominance.
- Tiling extends past `t = N` into a region vocab-parallelism **structurally
  cannot reach**, because `g ≤ N`. You can always use more tiles; you cannot use
  more ranks than you have.

So no value of `g` is a good trade — not merely the default one. That is the
durable claim, and it holds on any machine.

**But that family was computed at N=8, and this box cannot run N=8.** Physical
GPU index 1 is dead hardware, so there are **7 usable GPUs, not 8** — and 7 is
prime. With equal groups `g` must divide `N`, so at the world size actually in
use the entire family is two points:

| `g` | groups | configuration | communication |
| --- | --- | --- | --- |
| 1 | 7 | today's design (vocab tiling) | **1.00×** |
| 7 | 1 | full vocab-parallelism | **4.486×** |

Nothing in between. This cuts both ways and the document should say so.

It **strengthens** the verdict on this hardware: the only non-trivial
configuration available at N=7 is `g=7`, which is exactly the 4.486× case
already rejected. The tunable middle ground never existed here, so there is no
intermediate setting anyone could tune toward.

It **weakens the steelman as originally presented**: the generality was
demonstrated using configurations this machine cannot produce, and precisely the
rows where grouped vocab-parallelism looked least bad — the intermediate `g` —
are the rows that do not exist at N=7. No parameter was swept on this hardware.
The general argument stands as an argument about the technique; it is not a
sweep of a knob anyone here could turn.

Two more things make the rejection stick. The model was **deliberately
charitable to the option it rejected** — latency was ignored and an extra
encoder all-reduce that vocab-parallelism would owe was omitted from its side of
the ledger — and it still lost by a factor of 4.5. And the cost model was not
merely internally consistent: its **geometry was validated to 0.6%**, predicting
20.87 GB/s for the ZeRO-3 embed-window round trip against an independently
measured ~21.0 GB/s plateau. A model that reproduces a number it was not fitted
to is one you can argue from.

---

## 7. None of these techniques are novel

Worth saying plainly, because the numbers above could otherwise read as a
claim of invention.

Vocabulary partitioning of the output layer is Megatron's, from 2019.
Chunked fused linear-cross-entropy is Liger Kernel's (Hsu et al. 2024,
[arXiv:2410.10989](https://arxiv.org/abs/2410.10989)), whose §3.2 chunks hidden
states, projects each chunk and computes a partial loss per chunk, motivated by
memory: at a batch size of 8 and sequence length 4096, *"the 256k vocabulary
size will result in a 16.8GB logit tensor of precision bfloat16"*. (That
sentence is in the [full text](https://ar5iv.labs.arxiv.org/html/2410.10989),
not the abstract page. Note the shape: 16.8 GB there is a far longer sequence
than our 6.1 GiB at `B=32, T=1024` — the two numbers are not directly
comparable, only the phenomenon is.) Cut Your Losses
(Wijmans et al. 2024, [arXiv:2411.09009](https://arxiv.org/abs/2411.09009))
frames the same cost — *"cross-entropy... consumes an order of magnitude more
memory than the rest of the LLM combined"* — but by a different mechanism
(on-the-fly log-sum-exp inside a flash-style kernel, **not** vocabulary
chunking; the two are routinely conflated and should not be).

Three motivations, one axis, and the distinction is the point:

| | Splits the vocabulary to reduce... |
| --- | --- |
| Megatron-LM | inter-GPU **communication** |
| Liger FLCE | peak **activation** memory (the logits tensor) |
| **This campaign** | resident **parameter and gradient** bytes under ZeRO |

The row-sparse encoder gradient has no Megatron analogue at all, because it
exploits a property Megatron does not use: embedding-gradient sparsity is a
function of *batch content*, whereas Megatron shards the vocabulary statically.

**The contribution here is applying known techniques inside a ZeRO
data-parallel trainer written in Mojo, wiring an explicit leading dimension
through a GEMM stack that did not have one, and measuring what they actually
buy.** That last clause is doing real work: the measured answers repeatedly
differed from the expected ones, in both directions.

---

## 8. Proportion: what this campaign did *not* fix

This section is not a caveat. It is the most important number in the document.

At the shape this project trains at — `B=32, T=1024`, fp32 — the three tensors
under discussion sit like this. These are computed from the shapes, scaled from
the measured `-b 8 -t 1024` run; only that smaller shape was actually measured:

| Tensor | Size at B=32 | Scaling | vs. the **pre-campaign** `wte` window |
| --- | --- | --- | --- |
| `att_probs` | **~18 GiB** | `B · NH · T · T · L · 4` — i.e. **T²** | **123×** |
| `logits` | **~6.1 GiB** | `B · T · V_p · 4` | **42×** |
| the `wte` window | **~150 MiB** | fixed | 1× |

**This campaign reduced the smallest of the three by roughly 85%, while terms
42× and 123× larger were untouched by every workstream in it.**

The percentages, at the shape where they were *measured* (`-b 8 -t 1024`, fp32
stage 0): activations are 17,808 of 20,183 MiB, i.e. **88%** of everything on
the card, and removing 150 MiB is **0.7%** of that total — or **6.3%** of the
2,374 MiB static footprint that exists before the first forward pass. (Those two
denominators are the measured ones; do not mix them with the B=32 column above,
which is four times larger throughout.) Both facts are true at once, and a
report that shows the 150 MiB without showing the 18 GiB next to it is
technically accurate and materially misleading.

That is not an argument against having done the work — the floor was real, it
blocked ZeRO stages 2/3 from doing what they exist to do, and it is now gone.
It is an argument against believing the job is finished.

**On `att_probs` specifically, a correction is owed.** An earlier draft of the
measurement document asserted that nobody had looked at it. That was wrong, and
the correct version is more useful: `train_gpt2.mojo` documents the tradeoff
explicitly above the `attention_fwd` call and ships a working escape hatch —
setting `kv_cache.att_probs_addr = 0` makes attention backward recompute the
`QKᵀ` scores per layer instead of reading the stored ones. That path is
**correctness-verified, 16 of 16 cases** (unrelated to the 16 baseline runs in
§3). The decision to
store rather than recompute was deliberate and measured, not an oversight.

What *is* worth re-examining is narrower and better founded. The cost side was
measured at **+3.5% step time** (fp32 736.6 → 762.9 ms, bf16 587.1 → 606.8 ms)
— **on Metal, at B=4**. There is no CUDA equivalent, and the batch was 8×
smaller than production. Meanwhile the comment names its own trigger condition
— *"if T-scaling makes the store dominate"* — and this campaign's measurements
demonstrate that the condition is now met: `att_probs` is the single largest
tensor on the card, and it grows as T². Flipping one assignment and rerunning is
a cheap experiment against a number larger than everything else in this campaign
combined. Nobody has run it. That is a flag, not a claim.

For `logits`, the path is known: chunk the cross-entropy on top of the existing
vocab tiling, using an online-softmax two-pass (accumulate running max and
sum-exp per row across tiles, then form per-tile probabilities and `dlogits` in
a second pass). It is well-precedented and no numerical blocker is apparent, but
it is a *structural* change rather than a kernel change — a tile's `dlogits`
becomes transient, so LM-head forward and backward stop being separable — and
the second pass must recompute each logits tile, which is one extra full LM-head
forward GEMM per step: `2·B·T·C·V_p` = 316 GFLOP at `B·T = 4096`, against a
measured ~3.05 TFLOP step, so on the order of **+10% step time**.

---

## 9. How we were wrong

The most transferable content in the campaign is the list of things that were
confidently believed and false.

### Five false greens

A test suite that passes without running the code under test is worse than no
test suite, because it produces the same confidence with none of the coverage.
Five instances, all pre-existing, all found during this work (the first four in
[`pre_merge_smoke_subset.md`](pre_merge_smoke_subset.md)):

1. **The only real GPU-collective test opted out with a bare `return`.**
   `test_multi_gpu_collectives` is the sole test in the repo that drives real
   GPU allreduce / reducescatter / allgather, and it is gated on having ≥2 GPUs.
   Below that it returned silently, and the suite printed
   `13 passed, 0 failed, 0 skipped`. The only tell was the per-test timing:
   `PASS [ 0.031 ]` when it opted out versus `PASS [ 330.802 ]` when it actually
   ran across 7 GPUs. Nothing in the summary distinguished them. It now
   prints an explicit skip.
2. **No test drove the LM-head matmul above `output_channels = 3072`**, while
   production runs it at 50304 — a 16× gap. A change that tiles the LM head over
   the vocabulary could therefore pass the entire suite without a single test
   crossing a tile boundary. Green meant "did not run".
3. **The identical-ranges contract was documented and unenforced** — see §5.
   Docstrings said "every rank must pass identical lists"; nothing checked it,
   and violating it does not hang.
4. **Generation is ungated by *two* gates.** `make check` never builds
   `infer_gpt2_bf16`; and even when built, all five generation smoke tests skip
   on a missing checkpoint. Since generation is precisely what an LM-head change
   breaks first, that is two independent reasons a green run means nothing.
5. **The entire gradient-level verification battery was unavailable box-wide,
   and nobody had noticed.** `make verify-gpu` checks every gradient tensor
   against a PyTorch reference — the strongest correctness gate the repo has,
   and the obviously right one for a change that rewrites how `dwte` is
   computed. It could not run: the debug-state file on this box carried llm.c's
   activation-free magic number `20240327` where the check requires `20240520`.
   This is a **setup gap, not a code defect** — a documented one-time step
   (`pixi run python train_gpt2.py`, README:247) had simply never been performed
   here, so the gate had been silently skipped for the life of the box.

That fifth one is the most consequential, and it resolved in the campaign's
favour once the reference was regenerated. **The vocab-tiled LM head passes the
gradient check at its shipped default `K=8`:** `OK (LOGITS)`, and **16 of 16
gradient tensors report `TENSOR OK`, none `NOT OK`** — including `dwte` itself,
the tied-embedding gradient the tiling rewrote (`maxdiff = 0.0202` against a
`threshold = 0.5130`, l2 ratio 1.0013). That is an independent, whole-model
check against PyTorch through the production cuBLASLt strided-descriptor path,
and it is much stronger evidence than the bespoke tests of §4.

The unifying property across all five: each failure produced a *passing* signal,
and the distinguishing evidence (a timing, an argument value, a skip reason, a
magic number) existed but was in a place nobody read.

### The claims that travel furthest are the ones we checked least

The pattern, stated generally: **claims about someone else's work, asserted from
absence of evidence rather than evidence of absence, in the direction that made
one's own narrative tidier.**

"Nobody has looked at `att_probs`" is a claim about an entire codebase, made
without grepping for one. It was also exactly the claim that made the
surrounding analysis look more impressive. Within the hour a second workstream
had adopted it into its own document. Both retracted it (§8), and the retraction
turned out to be more useful than the claim, because it surfaced a working,
16/16-verified escape hatch that the original framing had implied did not exist.

The people who made the error found the sharpest formulation of it:

> The unrigorous claim was the *quotable* one. Three decimal places on a buffer
> size doesn't get repeated; "nobody has looked at this" gets repeated — and it
> got repeated within the hour. **The claims that travel furthest are the ones
> we checked least.**

The asymmetry is the diagnostic, not the error rate. Arithmetic in the same
document was checked to three decimals and reproduced across two runs; the claim
about other people's work was held to no standard at all. If your standard of
evidence varies with how much a finding flatters you, you will be wrong in a
consistent direction — much worse than being wrong at random.

**The second-order version is sharper, and is the part worth operationalizing:
adopting a good correction is not the same as auditing against it.** The
workstream that accepted the correction fastest checked whether the same defect
was present in *its own* work only when explicitly told to audit — and it was,
in its headline. So:

> **Audit the quotable claim first — and specifically, audit it in your own work
> immediately after accepting it in someone else's.**

The least flattering detail belongs here too, because it was volunteered and
because it is the one that generalizes. The "8 GPUs" figure behind the §6
steelman table came from someone who *had run* `nvidia-smi`, *had seen* index 1
reporting `[N/A]`, and was *already* pinning devices by UUID specifically to
avoid this class of problem — and still read "8 GPUs" straight off the row
count. The evidence was on screen and the mitigation was already in place. That
is what an unexamined quotable number does: it survives contact with the
evidence that refutes it.

A counter-example from the same campaign, and the reason the numbers in §6 can
be trusted: when one workstream took another's committed formulas, simplified
them properly, and got a *different* break-even, it flagged the discrepancy
**precisely because the new number made the rejected option look slightly
better**. The original author re-derived it independently, confirmed it, and
corrected their own headline. A correction that helps the option you are arguing
against is the one most worth surfacing, and the one least likely to surface by
accident.

### A named class: unlabelled-basis errors

Worth naming because it recurred three times across two workstreams and is
invisible to every kind of review that checks arithmetic. **An unlabelled-basis
error is a number that is correct under one reading and wrong under another,
with the reading unrecorded.** The value is not wrong. The value is
underdetermined, and both readings look correct in isolation — which is what
makes this dangerous rather than merely sloppy.

| Basis, unrecorded | Instance | Size of the error |
| --- | --- | --- |
| MiB (2²⁰) vs GB (10⁹) | one row of a bandwidth table reads **21.2 GB/s or 19.7 GiB/s** | **7.4%** — larger than several effects under discussion |
| GPUs **installed** vs GPUs **usable** | 8 vs 7, index 1 being dead hardware | collapsed the §6 steelman from a swept family to two points |
| min vs median; per-rank vs aggregate | the same shape, latent in the bandwidth numbers | not yet bitten, caught while fixing the first |

The fix is uniform and cheap: **put the basis in the key, not in the README.**
Name the unit in every field name, add a top-level `units` object, and prefer a
form the reader can re-derive. Concretely, `bytes_moved_per_rank` is now emitted
as an **exact integer** rather than a float product, so the geometry check in §6
is verifiable straight from the file without reconstructing anyone's volume
factors.

### "Negligible" is a claim about size; "droppable" is a claim about structure

The N-cancellation in §6 — the finding that upgrades the verdict from "loses on
our cluster" to "loses on any cluster" — was invisible in the original
derivation. A scalar term had been dropped before the expression was simplified.

The lesson is *not* "don't simplify early", which is too vague to act on. In the
person's own words, the failure was **dropping a term because it was small in
magnitude without checking whether it was structurally load-bearing.** The `+3`
beside `C` is numerically negligible and algebraically essential: keeping it is
what makes `(N−1)` and `N` cancel, and that cancellation is the best result in
the campaign.

> "Negligible" is a claim about size; "droppable" is a claim about structure —
> and I conflated them.

### A good bug report can produce a better fix than the reporter asked for

The units defect above was caught by a workstream other than the one that owned
the file, and the fix ended up better than what was actually requested. The
request was to *record* the volume factors. Removing the float from the
derivation entirely — emitting an exact integer — was the natural next step, but
it was only available because the report explained **why** recording mattered:
that a reader could not otherwise distinguish a binary from a decimal basis, and
so could not check the number at all.

A report that says "this is wrong, please fix" gets the fix you asked for. A
report that says "this is wrong, and here is the property it violates" lets the
owner see a cheaper way to guarantee the property. Explain the why.

### Contention invalidates timing, not arithmetic

Four concurrent test suites drove the box's load average to 65 and made every
duration on it meaningless. A GEMM sweep taken during that window was discarded
outright on an internal-consistency check: a *larger* GEMM measured *faster*
than a smaller one at the same K and N, which is not a thing that happens.

Note the asymmetry that follows. Memory measurements taken under that same load
were kept and are quoted in this document, because `exact_total_bytes` is a sum
of integers read out of live buffer objects and came back bit-identical between
a quiet run and a contended one. Contention corrupts stopwatches, not integers.
Knowing which of your numbers are which tells you which results survive a bad
afternoon on a shared machine.

---

## 10. State of play, and what to read next

**Landed and verified:** the exact memory instrument; the vocab-tiled LM head
(default on, +0.60% step time, bit-identical through 64 tiles, verified at four
tile counts, and — now that `make verify-gpu` can run at all — **16/16 gradient
tensors matching PyTorch at the shipped `K=8`, `dwte` included**); the
row-sparse coalesced encoder gradient with cross-rank range
agreement enforced; the `make smoke` subset (~40 s against ~13 min for the full
gate); the four false-green repairs.

**Landed but not on by default:** `LLMM_Z3_WTE_ONDEMAND=1`, which realizes the
ZeRO-3 window reduction. It compiles and the code path exists; it was held off
by default because removing `wte` from the gather window before the encoder can
fetch its own rows makes `encoder_fwd` read whatever else is in that buffer —
producing wrong numbers **silently**, the worst available failure mode.

**Not measured:** the non-cuBLASLt GPU staging path (this box selects cuBLASLt);
the `att_probs` recompute cost on CUDA at production shape (§8); anything about
throughput from the memory benchmark runs, deliberately.

Where to go for detail:

| Document | What it covers |
| --- | --- |
| [`zero_memory_accounting_2026-07-27.md`](zero_memory_accounting_2026-07-27.md) | The instrument. Why `nvidia-smi` cannot see a 150 MiB change, the 256 MiB / ~696 MiB decomposition, the exact in-process report, the full pre-change baseline by ZeRO stage and precision, and the activation breakdown behind §8. |
| [`zero_lm_head_vocab_tiling_2026-07-27.md`](zero_lm_head_vocab_tiling_2026-07-27.md) | The dense half. Leading dimensions and the strided-GEMM plumbing, the tile-count knob, the float64 test that distinguishes "tiling is wrong" from "fp32 is fp32", the coverage envelope (where tiling buys nothing), and the ordering dependency on the encoder work. |
| [`vocab_parallel_lm_head_feasibility_2026-07-27.md`](vocab_parallel_lm_head_feasibility_2026-07-27.md) | The rejection, in full. The cost model and its geometry validation, the pointwise-dominance family at N=8 and why it degenerates at N=7, the shard-alignment failure, and the two self-corrections §9 draws on. |
| [`wte_campaign_figures_2026-07-27.md`](wte_campaign_figures_2026-07-27.md) | The campaign's figure set: what was measured, what was modelled, and the one figure that was **not** published because the underlying GEMM sweep was discarded (§9). |
| [`pre_merge_smoke_subset.md`](pre_merge_smoke_subset.md) | `make smoke`, what it runs and why each piece earns its place — and, more valuably, the enumerated list of what it does **not** cover, including the first four false greens. |
| [`zero_grad_bucketing_design_2026-07-14.md`](zero_grad_bucketing_design_2026-07-14.md) | Prior state: the bucketed backward that established the `grad_pool_buf` floor, and the reduce-scatter linearity argument that lets `wte`'s two gradient contributions be reduced separately. |
| [`zero_stage3_param_streaming_2026-07-14.md`](zero_stage3_param_streaming_2026-07-14.md) | Prior state: the just-in-time parameter gather that established the `embed_window_buf` floor — including the sweep showing peak memory drops a full 256 MiB *the moment* `wte` residency goes away. |

---

## AI use statement

Written with AI assistance (Claude Opus agent via Claude Code), directed by
Evan.
