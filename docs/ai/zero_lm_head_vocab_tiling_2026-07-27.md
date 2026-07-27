# De-residenting the tied embedding, part 1: the vocab-tiled LM head

README roadmap item 1 reads:

> ZeRO: de-resident the tied `wte` embedding (vocab-chunked LM head + indexed
> encoder gather) to push stages 2/3 below the current ~150 MiB floor.

This document covers the first half — the vocab-chunked LM head. The indexed
encoder gather is a separate workstream and is not described here beyond where
the two meet.

**Scope, stated up front so nothing here is over-read.** This change tiles the
LM-head **GEMM only**. It does not chunk the softmax / cross-entropy. The full
`(B*T, V_p)` logits activation is still materialized exactly as before —
`act_sizes[Activations.logits] = B * T * V_p` is untouched — and
`fused_classifier.mojo` still consumes whole rows. What this buys is a reduction
in resident **parameter and gradient** bytes for `wte`. See §11 for why that
distinction matters more than it first appears.

The document is written for someone who knows Python and basic neural networks
and nothing about distributed training. Every term is defined at first use.

---

## 1. Background, from scratch

### The model, in the two shapes that matter

GPT-2 124M has an embedding table called `wte` ("weight token embedding"). It
has one row per vocabulary entry and one column per model channel:

```
wte : [V_p, C] = [50304, 768]
```

`C = 768` is the *channel* count (the model's internal vector width). `V_p` is
the **padded vocabulary size**: GPT-2's real vocabulary is 50257 tokens, but the
number is rounded up to 50304 because 50304 = 128 × 393 and GPU matrix kernels
run faster on dimensions that are multiples of 128. The extra 47 rows are dead
weight that exists purely for alignment.

In fp32 (4 bytes per number):

```
50304 × 768 × 4 bytes = 154,533,888 bytes = 147.4 MiB
```

That single tensor is the largest object in the model, and everything below is
about it.

### Weight tying

`wte` is used **twice**, and the two uses are the same numbers:

1. **As an embedding table** (the *encoder*). Given a token id like 8721, look
   up row 8721 — a 768-vector — and use it as that token's input representation.
   This is an indexed row gather; a batch of `B×T` tokens touches at most `B×T`
   distinct rows.
2. **As the output projection** (the *LM head*). After the transformer stack
   produces a 768-vector per position, multiply by `wte` transposed to get one
   score ("logit") per vocabulary entry:

```
logits[B*T, V_p] = ln_f[B*T, C] @ wteᵀ[C, V_p]
```

Sharing one tensor for both is called **weight tying**. It saves 147 MiB of
parameters and is standard for GPT-2. It is also the reason this problem is
hard: the encoder needs *a few scattered rows*, but the LM head, as written,
needed *every row at once*, because it was one single matrix multiply spanning
the whole vocabulary.

### Data parallelism, ranks, world size

**Data parallelism** is the simplest way to use N GPUs: put a full copy of the
model on each one, give each a different slice of the batch, and average the
gradients across GPUs before the optimizer step. Each participating process is
a **rank** (numbered 0..N-1) and N is the **world size**. Plain data
parallelism is often called **DDP** (distributed data parallel).

DDP's weakness is that every GPU stores the same everything. For each parameter
you store, in fp32:

| Category | What it is | Bytes per parameter |
| --- | --- | --- |
| **Parameter** | the weight itself | 4 |
| **Gradient** | ∂loss/∂weight, one per parameter | 4 |
| **Optimizer state** | Adam's two running averages (momentum + variance) | 8 |
| **Activation** | intermediate values saved for the backward pass | (depends on batch, not on parameter count) |

So a parameter costs 16 bytes of *persistent* memory under Adam, replicated on
every GPU. The first three categories are pure duplication under DDP.

### ZeRO, and what each stage buys

**ZeRO** ("Zero Redundancy Optimizer") removes that duplication by **sharding**
— splitting a buffer into N pieces and giving rank *r* only piece *r*, instead
of every rank holding the whole thing. It comes in three cumulative stages:

| Stage | What gets sharded | What each rank still holds in full |
| --- | --- | --- |
| **1** | optimizer state | parameters, gradients |
| **2** | optimizer state + gradients | parameters |
| **3** | optimizer state + gradients + parameters | nothing (gathered on demand) |

Each stage cuts the corresponding category from "full size" to "full size / N",
at the cost of extra communication.

Two collective operations do the work. A **collective** is an operation all
ranks call together.

- **Reduce-scatter**: every rank starts with a full-length array; the arrays are
  summed elementwise across ranks; rank *r* keeps only its slice of the result.
  This is how a sharded gradient is produced — used at the end of backward.
- **All-gather**: the inverse. Every rank starts with its own slice; afterwards
  every rank has the concatenation. This is how ZeRO-3 temporarily materializes
  a parameter it needs.

ZeRO-3's trick is that a parameter only needs to be resident *while a kernel is
using it*. So the trainer all-gathers one transformer layer's weights into a
small reusable **window** buffer, runs that layer, and reuses the window for the
next layer. Peak parameter memory becomes `shard + one window` instead of the
whole model.

### Where this repo already was

Two prior pieces of work set the stage (both in `docs/ai/`):
`zero_grad_bucketing_design_2026-07-14.md` and
`zero_stage3_param_streaming_2026-07-14.md`.

- **Gradient bucketing** (ZeRO-2/3): backward does not materialize a full
  gradient buffer. It keeps one small reusable **pool**, points the gradient
  pointers into it, computes one *bucket* of gradients, reduce-scatters that
  bucket onto the persistent shard, and recycles the pool. Peak gradient memory
  is `pool + shard`. The pool is sized by the largest single bucket
  (`_grad_pool_elems`, `train_gpt2.mojo`).
- **Parameter streaming** (ZeRO-3): `_z3_stream_layer` gathers a layer's 12
  tensors into `param_window_buf`; `_z3_stream_embed` gathers the non-layer
  tensors (`wte`, `wpe`, and the two final-layernorm vectors) into
  `embed_window_buf`.

Both mechanisms worked. Both bottomed out at the same number, for the same
reason.

---

## 2. The problem

`wte` is 147.4 MiB and the LM head touched all of it in one operation, so:

**Resident #1 — the gradient pool.** Backward's first bucket was the whole
`wte` gradient. `_grad_pool_elems()` returned
`max(wte + wpe, one transformer layer)`; with `wte + wpe = 150.4 MiB` against a
per-layer bucket of 27.0 MiB, `wte` set the floor. Halving the batch, adding
GPUs, sharding harder — none of it moved that number, because it is a property
of the model, not of the parallelism.

**Resident #2 — the ZeRO-3 embedding window.** `_z3_alloc_windows` sized
`embed_window_buf` as `wte + wpe + ln_f_gamma + ln_f_beta` = 150.4 MiB, and
`_z3_stream_embed` gathered all of it once per forward pass and held it across
the entire forward *and* backward — because the LM head reads `wte` in forward
and again in backward, and the streaming design keeps the window alive between
those two points.

So the "~150 MiB floor" in the roadmap item is the same tensor showing up twice.

The fix in principle is obvious: do the LM head in **vocabulary chunks**. Score
the first 6400 vocabulary entries, then the next 6400, and so on. Each chunk
needs only its own slice of `wte`, so only that slice ever has to be resident or
have gradient space allocated.

The fix in practice ran into one specific obstruction, which is the interesting
part.

---

## 3. Why a column tile needs a leading dimension

`logits` is stored **row-major** with shape `[B*T, V_p]`: element `(r, v)` lives
at flat offset `r * V_p + v`. Row-major means "consecutive elements of a row are
adjacent in memory".

A vocabulary chunk is a **column slice**: rows `0..B*T`, columns `t0..t0+tile`.
Draw it and the problem is immediate — the chunk's elements are *not*
contiguous. Each of its rows is `tile` numbers long, but the distance from one
row's start to the next is `V_p`, not `tile`.

This is the difference between a matrix's **width** and its **leading
dimension** (`ld`) — the stride from one row to the next. For a normal
standalone matrix they are equal. For a slice carved out of a bigger matrix,
`ld` stays the *parent's* width:

```
logits (row-major, ld = V_p = 50304)
┌──────────────────────────────────────────┐
│ ....... [ tile ] ......................  │  row 0   ← chunk row starts here
│ ....... [ tile ] ......................  │  row 1   ← ...and 50304 elements later
│ ....... [ tile ] ......................  │  row 2
└──────────────────────────────────────────┘
          ↑ t0
          width = tile = 6400,  ld = 50304
```

Contrast with the *weight* side. A vocabulary chunk of `wte[V_p, C]` is a
contiguous **row** range — rows `t0..t0+tile` of a row-major matrix. Its
elements are one unbroken run of memory starting at `wte + t0*C`, and its
leading dimension is still `C`. A row tile of a row-major matrix needs nothing
special; a column tile does. That asymmetry is the whole story: `wte` tiles for
free, `logits` does not.

Every matrix-multiply entry point in `llmm/matmul.mojo` built its tensors as
`row_major(rows, out_channels)` — a constructor that *assumes* `ld == width`.
There was no way to express the strided slice. That is what had to change.

### What actually changed

The change is narrower than it sounds, because the production path already
supports strides and simply was not being told about them.

- **cuBLASLt** (the NVIDIA vendor library this repo uses for its fast GEMMs)
  takes a leading dimension natively — it is an argument to every matrix layout
  descriptor. `_matmul_cublaslt` in `llmm/matmul.mojo` already called
  `_lt_make_layout(dt, rows, cols, ld)`; it just always passed the packed
  default. It now takes optional `ld_a` / `ld_b` / `ld_d` overrides where `0`
  means "packed, exactly as before". On this path a vocabulary tile is
  **zero-copy**: a pointer offset plus one integer.
- **Every other target** (CPU, Apple Metal, vendor-neutral GPU) **stages**
  instead. The tile is copied to or from a contiguous scratch buffer, and then
  the existing dense entry points are called completely unchanged. That costs
  one `O(rows × tile)` copy next to an `O(2 × rows × tile × C)` matrix multiply
  — with `C = 768`, the copy is under a thousandth of the arithmetic. It buys
  the guarantee that `linalg.matmul`, the two Metal transpose strategies and the
  CPU accumulate workaround are untouched by this change.

The new entry points are `matmul_lm_head_fwd_tile` and
`matmul_lm_head_bwd_tile`, both in `llmm/matmul.mojo`. Nothing else in the model
calls them: all the transformer layers keep calling `matmul_fwd` / `matmul_bwd`
with byte-identical arguments to before.

---

## 4. Backward: why one gradient tiles freely and the other does not

The LM head's backward computes two things. They behave differently under
tiling, and the difference is not a detail — getting it wrong gives wrong
gradients that no shape check would catch.

Write `dlogits` for the gradient arriving at the logits, `ln_f` for the LM
head's input, and let a tile be vocabulary rows `[t0, t0+w)`.

**Weight gradient — tiles freely.**

```
d_wte[t0:t0+w, :] = dlogits[:, t0:t0+w]ᵀ @ ln_f
```

Note which index is summed over: `B*T` (the token positions). The vocabulary
index is a *free* index — it labels which output row you are computing, not
what you are adding up. So tile `t0` computes exactly the rows `t0..t0+w` of
the answer, using exactly the same dot products the untiled version would have
used, and no other tile contributes to those rows. **The tiles partition the
output.** Each tile's result is complete and final the moment it is computed,
which is precisely why it can be reduce-scattered and the pool recycled
immediately.

Reduce-scatter is linear (it is a sum), so scattering the tiles separately and
scattering one big buffer land the same numbers on the shard. `wte`'s *other*
gradient contribution — from the encoder, since the tensor is tied — already
relied on that same linearity to be applied as a separate bucket.

**Input gradient — does not tile freely.**

```
d_ln_f = dlogits @ wte     (summed over the vocabulary)
```

Here the vocabulary is the **contraction** index — the thing being summed over.
Every tile contributes to *every element* of `d_ln_f`. So the tiles must
accumulate, not overwrite:

```
d_ln_f  = tile_0 contribution     (overwrite)
d_ln_f += tile_1 contribution     (accumulate)
d_ln_f += tile_2 contribution
...
```

Rather than zeroing `d_ln_f` up front and accumulating everything (which needs a
separate zeroing pass over a buffer the trainer does not own as a device
buffer), the first tile overwrites and the rest accumulate. In GEMM terms that
is `beta = 0` on the first tile and `beta = 1` afterwards — `beta` is the
standard BLAS scalar that multiplies the destination before the product is added
in, so `beta=1` means "add to what is already there". cuBLASLt takes it
directly. The staged path fakes it by computing into a temporary and adding.

**One numerical consequence, stated honestly.** Splitting a sum and adding the
partial sums back is not bit-identical in floating point — addition is not
associative. Which quantities that touches follows directly from which axis each
one contracts over:

- `logits` (forward) contracts over `C`, which tiling does not touch. Each logit
  is still one full-length dot product over the same values.
- `d_wte` contracts over `B*T`, likewise untouched.
- `d_ln_f` contracts over the **vocabulary** — the axis tiling splits. This is
  the one quantity tiling actually perturbs.

That prediction is exactly what the tests show, and the third bullet caused a
real (and instructive) test failure worth recording. At production shape
(`V_p = 50304`, 8 tiles) `d_wte` matched the untiled result to `atol = 1e-5` on
values of magnitude ~1e2 — around one unit in the last place, i.e. numerically
indistinguishable. But an initial `rtol = 1e-4` check on `d_ln_f` **failed**, at
`1.004e-4`.

That failure was not a bug, and the way to establish that was not to relax the
tolerance until it passed. The data is mixed-sign, so the dot product cancels
heavily: the sum of the 50304 products' magnitudes is ~20x the final value, and
rounding error scales with the former, not the latter. Comparing two fp32
results to *each other* therefore cannot distinguish "tiling is wrong" from
"fp32 is fp32". The test now computes the answer in **float64** for a sample of
elements and requires *both* the tiled and the untiled fp32 results to sit
within `5e-5 × (accumulated magnitude)` of it. Both do. That tests the claim
that matters — tiling does not degrade accuracy — and it still fails loudly on a
dropped or double-counted tile, which would move the result by roughly the
accumulation scale (~1e3), four orders of magnitude outside the bound.

**Padding rows stay zero.** `fused_classifier` forces `dlogits` to exactly zero
on the padding columns `[V, V_p)`, so the 47 padding rows of `wte` get zero
gradient. Tiling does not disturb this: those columns simply land in whichever
tile contains them and multiply out to zero as before.

**Rank-invariance.** Under ZeRO-2/3 each tile issues its own reduce-scatter.
Collectives here take caller-supplied offset lists and apply them to *peer*
buffers, so if two ranks disagreed about the list they would silently read the
wrong data rather than deadlock. The tile boundaries are computed from
`padded_vocab_size` and a compile-time constant only — never from batch content
— so every rank walks the identical tile sequence by construction. This is why
`_lm_head_tile_rows()` deliberately does not depend on batch size or sequence
length. (It could not anyway: `allocate_gradients()` runs during construction,
before `batch_size` is set.)

---

## 5. Prior art — this technique is not new

Splitting the output layer along the vocabulary is well-established. It is worth
being explicit about that, and about what is and is not a contribution here.

**Megatron-LM** (Shoeybi et al., 2019, [arXiv:1909.08053](https://arxiv.org/abs/1909.08053))
splits along exactly this axis. From its Section 3: *"We parallelize the input
embedding weight matrix E_{H×v} along the vocabulary dimension E=[E_1,E_2]
(column-wise)."* Its motivation is **communication**, not memory: it goes on to
say *"To reduce the communication size, we fuse the output of the parallel GEMM
[Y_1,Y_2] with the cross entropy loss which reduces the dimension to b×s"* —
i.e. send scalar losses between GPUs instead of a full logits matrix. The
concrete implementation lives in
[`megatron/core/tensor_parallel/cross_entropy.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/cross_entropy.py).
(These quotes are in the paper's full text, not on its abstract page; the full
text is at <https://ar5iv.labs.arxiv.org/html/1909.08053>.)

**Liger Kernel** (Hsu et al., 2024, [arXiv:2410.10989](https://arxiv.org/abs/2410.10989))
is mechanically the closest published analogue. Its §3.2 FusedLinearCrossEntropy
chunks the hidden states, projects each chunk, and computes a partial loss per
chunk so the full logits tensor is never materialized — motivated by memory:
*"a 256k vocabulary size will result in a 16.8 GB logit tensor of bfloat16,
causing a huge spike in the peak memory usage."*

**Cut Your Losses** (Wijmans et al., 2024, [arXiv:2411.09009](https://arxiv.org/abs/2411.09009))
frames the same cost — *"cross-entropy... consumes an order of magnitude more
memory than the rest of the LLM combined"* — but reaches it by a different
mechanism (on-the-fly log-sum-exp in a flash-style kernel), **not** vocabulary
chunking. It is cited here for the framing only.

Three motivations, one axis, and the distinction matters:

| | Splits vocabulary to reduce... |
| --- | --- |
| Megatron-LM | inter-GPU **communication** |
| Liger FLCE | peak **activation** memory (the logits tensor) |
| **This change** | resident **parameter and gradient** bytes under ZeRO |

This work targets a third thing. The logits tensor here is *not* the problem —
it is an activation and was already budgeted for. The problem is that a single
whole-vocabulary GEMM forced `wte`'s 147 MiB of *parameters* to be gathered and
its 147 MiB of *gradients* to be allocated, all at once, defeating the sharding
that ZeRO stages 2 and 3 exist to provide.

Weight tying itself comes from Press & Wolf (2016,
[arXiv:1608.05859](https://arxiv.org/abs/1608.05859)) and Inan et al. (2016,
[arXiv:1611.01462](https://arxiv.org/abs/1611.01462)); the ZeRO stage taxonomy
in §1 is from Rajbhandari et al. (2019,
[arXiv:1910.02054](https://arxiv.org/abs/1910.02054)).

One more distinction, because it explains why Megatron's approach was not simply
adopted here. Megatron is **tensor-parallel**: different GPUs hold different
*parameters*, so splitting the vocabulary across GPUs is natural and each GPU
already has the `ln_f` activations it needs. This trainer is **data-parallel**:
different GPUs hold different *data* and (under ZeRO) different *slices* of
every tensor, chosen by flat offset rather than by tensor structure. Making the
LM head vocab-*parallel* here would require all-gathering the `ln_f` activations
so each rank could score its vocabulary slice against every rank's tokens.

That comparison was evaluated separately during this campaign and came back
negative: the ZeRO-3 shard boundary does not align usefully with `wte` rows
(at 7 ranks, four ranks hold zero vocabulary rows and boundaries land mid-row),
and the activation all-gather only pays when `N·B·T < V_p` — a communication
regression at production shape. The useful structural statement from that
analysis is that **under data parallelism, vocab-parallelism minus the
activation all-gather simply is vocab tiling**; the two are ends of one trade,
with `N·B·T` against `V_p` as the exchange rate. Tiling sidesteps the
communication entirely, which is a genuine point in its favour under DP.

So: no novel technique is claimed. The contribution is applying a known
technique inside a ZeRO stage-2/3 sharded trainer written in Mojo, wiring an
explicit leading dimension through a GEMM stack that did not have one, and
measuring what it actually buys.

## 6. The knob

```
-D LLMM_LM_HEAD_VOCAB_TILES=<K>      # default 8
```

`K` is a *requested* tile count. The realized tile row count is
`ceil(V_p / K)`, rounded **up** to a multiple of 128 when `V_p >= 128` so the
GEMM keeps its friendly alignment. At GPT-2 124M:

```
ceil(50304 / 8) = 6288  →  rounded up to 6400  →  8 tiles
                              7 × 6400 = 44800, final tile = 5504 rows
```

The final tile being short is normal and is exercised deliberately in the tests.

`K = 1` restores the exact previous behavior — the trainer branches to the
original single `matmul_fwd` / `matmul_bwd` call, with no staging, no extra
collectives, and no tile entry points involved at all. That branch is the escape
hatch if tiling ever costs more step time than the memory is worth.

Below 128 rows the rounding is skipped, so the tiny test models (`V_p = 64` in
the CPU equivalence harness) tile at 8 rows × 8 tiles rather than collapsing to
a single tile. This matters: a test suite that never took the tiled branch would
prove nothing about it.

The second knob is deliberately **off** by default:

```
-D LLMM_Z3_WTE_ONDEMAND=1            # default off
```

See §7.

---

## 7. Measured and derived numbers

All figures fp32, GPT-2 124M (`V_p = 50304`, `C = 768`, `T_max = 1024`), tile =
6400 rows.

### Gradient pool

| Bucket | Before | After |
| --- | --- | --- |
| LM-head bucket | 38,633,472 elems = **147.4 MiB** | 4,915,200 elems = **18.8 MiB** |
| Encoder bucket (`wte + wpe`) | 150.4 MiB | 150.4 MiB *(other workstream)* |
| One transformer layer | 27.0 MiB | 27.0 MiB |
| **Pool actually allocated** = max | **150.4 MiB** | **150.4 MiB** |

Read that last row carefully. The LM-head bucket shrank 7.9×, and at 18.8 MiB it
is now *below* the 27.0 MiB per-layer bucket — meaning **the LM head is no
longer a binding constraint on the pool at all**. But the allocated pool is
`max()` over all buckets, and the encoder bucket still asks for 150.4 MiB. So
on this branch alone the allocation number does not move. The win is real and
is realized the moment the encoder bucket shrinks; until then it is latent.

Saying "147 MiB saved" here would be false, and it is not claimed.

### ZeRO-3 embedding window

| | Before | After, `LLMM_Z3_WTE_ONDEMAND=1` |
| --- | --- | --- |
| `embed_window_buf` | `wte + wpe + ln_f` = **150.4 MiB** | `wpe + ln_f` = **3.0 MiB** |
| `wte_tile_buf` | — | one tile = **18.8 MiB** |
| **Total** | **150.4 MiB** | **21.8 MiB** |

A 128.6 MiB reduction, 6.9×. With the flag off (the default) the window is
unchanged and the LM head simply addresses tiles *inside* the already-resident
window — pointer arithmetic, no extra gather, no cost.

---

## 8. The dependency on the encoder workstream, stated plainly

`LLMM_Z3_WTE_ONDEMAND` defaults **off**, and the ZeRO-3 saving above is
therefore **not active by default and has not been end-to-end verified.**

The reason is a genuine ordering dependency, not caution for its own sake. Under
ZeRO-3, `params.wte` points into `embed_window_buf`. The LM head can be taught
to want only one tile at a time. The **encoder** cannot — as written it indexes
arbitrary vocabulary rows (whichever tokens the batch happens to contain) out of
`params.wte`. Remove `wte` from the window before the encoder learns to gather
its own rows, and `encoder_fwd` reads whatever else is in that buffer. It would
produce garbage embeddings, and it would do so *silently* — no crash, no shape
error, just wrong numbers. That is the worst possible failure mode, so the flag
is off and documented as not-standalone-safe.

When the indexed encoder gather lands, flipping this one flag realizes the
128.6 MiB. The code path exists, compiles, and is written; it is untested end to
end, and this document says so rather than implying otherwise.

---

## 9. Where this helps, and where it does not

Worth being precise, because the coverage envelope is narrower than "tiling
saves memory" suggests.

| Configuration | What tiling buys |
| --- | --- |
| WORLD_SIZE=1, any stage | **Nothing.** No memory saved. |
| Stages 0 / 1, any world size | **Nothing.** No memory saved. |
| Stages 2 / 3, world size > 1 | LM-head gradient bucket 147.4 → 18.8 MiB |
| Stage 3 + `LLMM_Z3_WTE_ONDEMAND` | additionally, embed window 150.4 → 21.8 MiB |

The reason for the top two rows: the gradient **pool** only exists under
bucketing, which `_use_bucketing()` enables only at `WORLD_SIZE > 1` **and**
`zero_stage >= 2`. Everywhere else the trainer allocates the full monolithic
gradient buffer, because the unsharded optimizer needs every gradient at once —
so there is no per-bucket residency for tiling to reduce. Likewise the ZeRO-3
embedding window only exists at stage 3.

So at WORLD_SIZE=1 this change costs ~0.6% step time and saves nothing. That is
a real if small regression for the single-GPU case and it should be recorded as
one, not glossed.

It is nevertheless left **on** by default rather than gated behind
`_use_bucketing()`, for one specific reason: `tests/test_zero_equivalence.mojo`
validates stages 1/2/3 against a **stage-0** baseline. If stage 0 ran untiled
while stage 2 ran tiled, that comparison would carry a floating-point
reassociation difference in `d_ln_f` that has nothing to do with sharding
correctness — the gate would be measuring the tiling, not the thing it exists to
measure. Keeping every stage on the same code path keeps that comparison
meaningful. `K=1` remains available if the 0.6% ever matters more.

## 10. What was verified

Everything below was run in this worktree and observed to pass. Gates not listed
were not run.

**Gates**

| Gate | Result |
| --- | --- |
| `make format` | clean, no diff left |
| `make lint` | PASS |
| `make check` (lint + 3 builds) | PASS, 1m45s |
| `make build WORLD_SIZE=2` | PASS |

**Multi-tile coverage — which tests actually took the tiled branch.** This
matters because *no pre-existing test in the repo drives `matmul_fwd` /
`matmul_bwd` above `output_channels = 3072`*, so before this work every test
would have run the LM head at exactly one tile and proved nothing about the
strided path. `tests/test_lm_head_vocab_tiling.mojo` (9 tests, all passing)
closes that:

- `test_fwd_tiled_matches_dense_at_vocab_scale` and
  `test_bwd_tiled_matches_dense_at_vocab_scale` run at the **real** `V_p =
  50304` with the real 6400-row default tile. Each asserts at runtime
  `ntiles == 8` and `ragged == 5504`, so they cannot silently degrade into a
  single-tile no-op — the assertion fails first. That is how multi-tile
  execution is confirmed, not inferred.
- The ragged final tile (5504 rows, since 50304 is not a multiple of 6400) is
  therefore exercised on every run, in both directions.
- `test_fwd_tiled_writes_every_column` poisons the logits buffer and asserts
  every element was overwritten — catches a dropped tile or a wrong stride.
- `test_bwd_d_input_accumulates_across_tiles` asserts the accumulated `d_input`
  differs from the last tile's contribution alone, so a stuck-at-overwrite bug
  in the accumulate flag cannot pass.
- The smaller cases (`V_p = 37`, tile 8 — also non-divisible) assert
  `ntiles > 1` and compare every element.
- `tests/test_zero_equivalence.mojo` — 6/6 (stages 1/2/3 at world sizes 2 and
  8). Its `V_p = 64` micro-model tiles at 8 rows × 8 tiles, so the sharded
  stages exercise the tiled path too rather than collapsing to one tile.

**Step time.** K=8 vs K=1, `b=4 t=1024 -z 0`, single GPU, median of steps 2-12,
two runs each: **56.99 → 57.33 ms/step, +0.60%**. Final losses tracked to within
0.0025 across all four runs.

**Not verified — stated plainly:**

- `LLMM_Z3_WTE_ONDEMAND=1` (the 128.6 MiB ZeRO-3 saving). Compiles; never run
  end to end, because it is unsafe until the encoder workstream lands (§8).
- The non-cuBLASLt GPU staging path (Apple Metal, forced-portable). The CPU
  staging path is covered by the tests above; the GPU staging kernel
  `_gpu_col_slice_kernel` is not exercised on this hardware, which selects
  cuBLASLt.
- The gradient-pool *allocation* does not shrink on this branch alone — see the
  §7 table. Only the LM-head bucket term shrank; the encoder term still sets the
  `max()`.

---

## 11. The bigger target this work does not address

Worth recording, because the numbers here could otherwise be read as closing
more than they do.

Roadmap item 1 targets the ~150 MiB `wte` floor, and that is a
**parameter/gradient** memory problem. But the largest single buffer in this
trainer is neither: it is the **logits activation**, `(B*T, V_p)`.

```
B=4,  T=1024 (this repo's benchmark shape):  4096 × 50304 × 4 B =   824 MiB
B=32, T=1024 (production shape):            32768 × 50304 × 4 B =  6288 MiB
```

That is roughly 5x to 40x the floor this campaign set out to remove, and this
change does not touch it. The tiling here writes each tile into that same
full-size buffer at a column offset; `fused_classifier` then consumes whole rows
to compute the row max and sum-exp, and overwrites them in place with `dlogits`.

Chunking the cross-entropy on top of this tiling would address it, and is
well-precedented (Liger's FusedLinearCrossEntropy, §5). The mechanism would be an
online-softmax two-pass: accumulate running `(max, sum-exp)` per row across
tiles, then form per-tile probabilities and `dlogits` in a second pass. The
recurrence already exists in `llmm/softmax.mojo`.

Two honest caveats on that, from a first assessment rather than an
implementation:

1. It is a **structural** change, not a kernel change. A tile's `dlogits` would
   be transient, so the LM-head backward GEMMs must fuse into the second pass —
   LM-head forward and backward stop being separable. That touches the
   forward/backward contract in `train_gpt2.mojo` and interacts with gradient
   accumulation, the `recompute` flag, and the inference path that also reads
   `acts.logits`.
2. The second pass must **recompute** each logits tile (retaining them defeats
   the purpose). That is one extra full LM-head forward GEMM per step:
   `2·B·T·C·V_p` = 316 GFLOP at `B*T = 4096`, against a measured ~3.05 TFLOP
   step — so on the order of **+10% step time**.

No numerical blocker is apparent; online softmax with running-max rescaling
meets the 1e-5 bar comfortably. Credit for identifying the gap goes to the
vocab-parallel workstream's memory analysis.

---

## 12. Files touched

- `llmm/matmul.mojo` — `ld_a`/`ld_b`/`ld_d` on `_matmul_cublaslt`; new
  `matmul_lm_head_fwd_tile`, `matmul_lm_head_bwd_tile`, `_gpu_col_slice_kernel`,
  `_col_slice_copy`.
- `train_gpt2.mojo` — the two knobs; `_lm_head_tile_rows` / `_lm_head_num_tiles`
  / `_lm_head_wte_tile`; the LM-head term in `_grad_pool_elems`; `wte_tile_buf`;
  `_z3_alloc_windows` and `_z3_stream_embed` split; tiled forward and backward
  call sites.
- `tests/test_lm_head_vocab_tiling.mojo` — new.

**Shared merge point:** `_grad_pool_elems()` is edited by the encoder
workstream too. The edit here is one added `lm_head` term and one widened
`max()`; the pre-existing `wte + wpe` term is left exactly as it was, since it
belongs to the encoder bucket.

## AI use statement

Written with AI assistance (Claude Opus agent via Claude Code), directed by
Evan.
