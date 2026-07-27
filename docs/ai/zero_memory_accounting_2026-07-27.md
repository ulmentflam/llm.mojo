# Measuring ZeRO memory: why `nvidia-smi` could not see a 150 MiB saving

*2026-07-27 — measurement workstream (Team M) for README roadmap item 1.*

This document is about an instrument, not an optimization. Two other workstreams
are shrinking large resident GPU buffers. Before they land, somebody has to be
able to *prove* the buffers got smaller — and the measurement we had could not
do that. This explains why, what replaced it, and how to reproduce every number.

It is written to be readable without prior knowledge of ZeRO or of GPU memory
accounting. If you already know the vocabulary, skip to "The quantization trap".

---

## Background: the words you need

**Data parallelism.** Train one model on several GPUs by giving each GPU a
different slice of the batch. Every GPU holds a full copy of the model, computes
gradients on its own slice, and then the GPUs average their gradients so all
copies stay identical. Each participating process is a **rank** (rank 0, rank
1, …) and the number of them is the **world size**. In this repo world size is a
compile-time constant (`make build WORLD_SIZE=2`) and the ranks are host threads
inside one process, one GPU each.

**The four kinds of training memory.** Fitting a model on a GPU means fitting
four quite different things, and they scale differently:

| Kind | What it is | Size for GPT-2 124M in fp32 |
| --- | --- | --- |
| **Parameters** | the weights themselves | ~475 MiB |
| **Gradients** | one number per parameter, produced by the backward pass | ~475 MiB |
| **Optimizer state** | AdamW keeps two running averages (`m`, `v`) per parameter, always in fp32 | ~950 MiB |
| **Activations** | intermediate values saved during the forward pass so the backward pass can use them | depends on batch size and sequence length — from ~0.5 GiB at a toy shape to tens of GiB at production shapes |

The first three are *model-proportional*: they depend only on how many
parameters there are. The fourth is *batch-proportional*. That distinction
matters later, because a saving in the first three can be swamped by the fourth.

**Sharding, and ZeRO.** In plain data parallelism every rank stores all four
things in full, which is enormously redundant: with 2 ranks you are storing two
identical copies of the optimizer state. **ZeRO** ("Zero Redundancy Optimizer")
removes that redundancy by **sharding** — splitting a buffer into `world_size`
pieces and having each rank keep only its own piece, fetching the others over
the interconnect when it briefly needs them. It comes in stages, each shedding
one more copy. The paper describes "three main optimization stages ... which
correspond to the partitioning of optimizer states, gradients, and parameters"
([ZeRO], Rajbhandari, Rasley, Ruwase & He 2019, §1,
<https://arxiv.org/abs/1910.02054>; the sentence is in the full text, not the
abstract — <https://ar5iv.labs.arxiv.org/html/1910.02054>):

- **stage 0** — no sharding; ordinary data parallelism (DDP).
- **stage 1** — shard the **optimizer state**.
- **stage 2** — also shard the **gradients**.
- **stage 3** — also shard the **parameters**, gathering each layer's weights
  just before use and releasing them after.

So peak memory per GPU should *fall* as the stage rises. Demonstrating that fall
is what `scripts/benchmark_zero.py` exists to do.

**A caching allocator.** When a program asks the GPU driver for memory, the
driver call (`cuMemAlloc`) is slow — hundreds of microseconds — and a training
step performs many allocations. So essentially every GPU framework, this one
included (Mojo's `DeviceContext`), puts a **caching allocator** in between: it
requests a large chunk from the driver once, hands out slices of that chunk to
the program, and when the program frees a slice the allocator *keeps* the chunk
and reuses the space for the next request. The chunk is not returned to the
driver.

Two consequences, and the whole of this document follows from them:

1. **"Memory used" is chunk-granular, not byte-granular.** The driver only ever
   sees whole chunks. On this box the observed granularity is ~256 MiB.
2. **It only ratchets upward.** Freeing a buffer does not reduce the driver's
   view. The allocator's high-water mark is permanent for the process lifetime.

---

## The quantization trap

`scripts/benchmark_zero.py` measured peak memory the obvious way: a background
thread polls `nvidia-smi --query-gpu=memory.used` every 0.25 s during the run,
keeps the maximum per GPU, and subtracts a per-GPU reading taken just before
launch so that co-tenant jobs on other GPUs are excluded
(`peak_mem_mib_per_gpu_delta`, `peak_mem_mib_max_delta`).

Nothing about that is wrong. It is a faithful measurement of *what the process
costs the machine* — which is the number you care about when you are deciding
whether a job fits on a card. Keep it.

But it answers a different question from "did this buffer get smaller". What
`nvidia-smi` reports is the CUDA context's **committed** footprint: chunks the
caching allocator has taken from the driver, plus the CUDA context's own
overhead. It is not "bytes the program asked for". Between those two numbers sit
the allocator's uncommitted slack and a large fixed context cost.

So consider removing a 150 MiB buffer. The allocator hands that space back to
its own free list. Unless the removal happens to drop the *total* below a chunk
boundary, the number of chunks held from the driver does not change, and
`nvidia-smi` reports **exactly the same number as before**. The saving is real
and completely invisible.

Both ZeRO design documents in this repo flagged this
(`docs/ai/zero_stage3_param_streaming_2026-07-14.md` lines 108-113, echoed in
`docs/ai/zero_grad_bucketing_design_2026-07-14.md` lines 187-196), but the
benchmark was never given a second instrument. This document is that instrument.

### The evidence that it is not hypothetical

From the pre-change baseline below (world size 2, `-b 4 -t 64`). The middle
column is what `nvidia-smi` reported; the right column is what the program
actually allocated.

| Change | `nvidia-smi` delta | Exact delta |
| --- | --- | --- |
| fp32, stage 2 → 3 | **0 MiB** (3000 → 3000) | **−60.000 MiB** |
| bf16, stage 1 → 2 | **0 MiB** (2488 → 2488) | **−43.522 MiB** |
| bf16, stage 2 → 3 | **0 MiB** (2488 → 2488) | **−30.000 MiB** |
| fp32, stage 1 → 2 | −256 MiB (3256 → 3000) | −87.044 MiB |

The first three rows are the failure everyone expected: real reductions
reported as exactly zero. The fourth is the one that should worry you more —
here `nvidia-smi` reports a change **2.9× larger** than the real one. It is not
under-reporting; it is reporting something else entirely. A chunk boundary
happened to be crossed, and the size of the step is the size of the chunk, not
the size of the buffer.

So the reading is not a noisy version of the truth that you could average away.
It is a *quantized* version, and quantized measurements mislead in both
directions.

### Why, precisely — measured, not assumed

Because the new report also records the driver's own figure from inside the
process, we can decompose the old measurement exactly. Two facts, from all 16
baseline runs:

**1. Every `driver_used_bytes` reading is an exact multiple of 256 MiB.** Across
both shapes and both precisions, the observed values were 1792, 2304, 2560,
3072, 11776, 12544, 22016, 22272 and 22784 MiB — that is 7, 9, 10, 12, 46, 49,
86, 87 and 89 chunks of 256 MiB. Not one reading fell off the grid. That pins
the caching allocator's commit granularity on this box at 256 MiB.

**2. The `nvidia-smi` delta sits a near-constant ~696 MiB above it.** In all
eight `-b 4 -t 64` rows the difference is exactly 696 MiB; in the `-b 8 -t 1024`
rows it is 696 (fp32) or 698 (bf16). That is the fixed cost of the CUDA context
itself, which belongs to the process but not to any buffer.

Putting those together, the old measurement was, empirically:

```
nvidia-smi peak delta  ≈  roundup_256MiB(bytes actually needed)  +  ~696 MiB
```

Read that formula and the failure is not bad luck — it is arithmetic. Any change
smaller than 256 MiB can only move the result if it happens to straddle a
boundary, and when it does move the result, it moves it by 256 MiB regardless of
how big the change was. **The instrument's resolution is 256 MiB. The buffers
being optimized are ~150 MiB.**

This decomposition is an empirical finding on this toolchain and this box, not
something published work covers; the two ZeRO design documents in this repo
noticed the symptom but did not measure the mechanism. Treat the 256 MiB and
~696 MiB figures as properties of *this* setup, to be re-derived rather than
assumed elsewhere. The method of deriving them — compare an in-process exact
total against an in-process driver query against the external tool — transfers
anywhere.

---

## The fix: ask the program, not the driver

The trainer knows exactly what it allocated. Every long-lived GPU allocation is
held in a `DeviceBuffer` field, and `len(buf)` is that buffer's element count,
so `len(buf) * size_of[dtype]()` is an exact byte count with no quantization at
all.

The addition is `GPT2.memory_report_json` / `GPT2.print_memory_report` in
`train_gpt2.mojo`, plus `ZeroContext.comm_bytes` in `llmm/zero.mojo`. Each rank
prints one JSON line, tagged `[mem-report] `, giving every buffer's exact size,
a rollup by class (params / gradients / optimizer / ZeRO-3 windows /
activations / index / scratch / comm), and — for comparison — the driver's own
figure for the same GPU.

Two design choices are worth calling out, because they are what make the
instrument trustworthy:

**It reads the live buffers rather than recomputing sizes.** It would have been
easier to re-derive each size from the same expression the allocation used
(`optimizer_num_parameters * 4`, and so on). That would have been worthless: the
instrument exists to measure *other people's* changes to those expressions, and
a copy of an expression silently stops matching the moment the original moves.
Reading `len(buf)` off the object that was actually allocated cannot drift.

**It is emitted at two phases.**

- `post_alloc` — after parameters, gradients, optimizer state and the ZeRO-3
  gather windows exist, but *before* the first forward pass, so activations have
  not been sized yet. This isolates the model-proportional, ZeRO-shardable
  footprint from the batch-proportional one.
- `steady` — after the final training step, with activations and the ZeRO
  collective scratch buffers sized by real work.

**It distinguishes "gone" from "small".** Every buffer field on the model is
always constructed; a configuration that does not use one leaves it at a
1-element placeholder. So byte counts alone cannot tell "this buffer was
eliminated" from "this buffer is genuinely tiny" — both read as a handful of
bytes, and a before/after diff would render an outright elimination as a
meaningless ~4-byte delta. The report therefore also emits `inactive_buffers`,
naming every buffer sitting at placeholder size. A name that appears in the
"after" run but not the "before" is a structural elimination, and that is a
different — generally better — result than a shrink. This matters concretely
for approaches that stop gathering a tensor at all rather than gathering it in
smaller pieces; without this field, such an approach would score as though it
had merely made a buffer very small.

The split between phases matters because at production batch sizes activations
dominate everything else, and a change to the gradient pool that is obvious at
`post_alloc` is a rounding error in the `steady` total. Reporting only one of
the two would flatter or bury the result depending on which you picked.

The whole path is gated on the `LLMM_MEM_REPORT` environment variable, so
ordinary runs print nothing and pay nothing.

### What it does *not* count

Stated plainly, because an "exact" number with an undisclosed hole is worse than
an approximate one:

- the 32 MiB cuBLASLt workspace and the other process-global allocations made
  through `llmm/memory.mojo`'s `persistent_device_buffer` (layernorm and
  bias-gradient scratch);
- the attention GEMM scratch, which is cached as raw integer addresses inside
  `KVCache` rather than as typed buffer fields;
- pinned host buffers (they are host memory, not device memory);
- the CUDA context's own overhead;
- the caching allocator's uncommitted slack.

Consequently `exact_total_bytes` is always **less** than `driver_used_bytes`,
and the gap between them is not a constant. Every omitted item above is
identical across ZeRO stages and unaffected by the buffer changes being
measured, so they cancel exactly in a before/after delta — which is the only
thing this instrument is used to compute. The natural place to close the largest
remaining gap, if someone wants to, is a byte counter inside
`persistent_device_buffer`, since every process-global allocation already funnels
through that one function.

---

## Method (reproducible)

Box: `workstation-max`, 8× NVIDIA RTX PRO 6000 Blackwell Max-Q **installed, but
only 7 usable** — physical GPU index 1 is faulted hardware (GSP faults at idle;
it survived two reseats and a cold power cycle, and is RMA-grade). It is
independently visible in `nvidia-smi`, which returns `[N/A]` for that card's
utilization while every other card reports a value.

Two consequences worth stating, because neither is obvious from the data files.
Nothing in this document is affected — every measurement here ran on physical
GPUs 5 and 6 — but **a world-size-8 measurement is currently not possible on this
machine at all**, so any scaling claim at that world size cannot be checked here.
And the `gpu_count` field in the benchmark JSON is the number of GPUs `nvidia-smi`
*enumerates*, which is 8; it is not a count of usable devices. Read it as
inventory, not capacity.

Two GPUs pinned
**by UUID**, never by index — CUDA renumbers around a faulted card, so an
ordinal can silently move to different hardware:

```sh
export CUDA_VISIBLE_DEVICES=GPU-1acc88b3-e8ef-5a64-7564-2154c78c10dc,GPU-c68a01d9-4380-2b67-3582-080c2ab34c1f
```

Those are physical GPUs 5 and 6, so world size 2. Then, from the worktree:

```sh
# bench shape
make benchmark-zero BENCH_ZERO_WORLD=2 BENCH_ZERO_STAGES=0,1,2,3 \
     BENCH_ZERO_B=4 BENCH_ZERO_T=64 BENCH_ZERO_STEPS=12

# production-ish shape
make benchmark-zero BENCH_ZERO_WORLD=2 BENCH_ZERO_STAGES=0,1,2,3 \
     BENCH_ZERO_B=8 BENCH_ZERO_T=1024 BENCH_ZERO_STEPS=8 \
     BENCH_ZERO_TIMEOUT=1500 \
     BENCH_ZERO_OUT=zero/bench/bench_zero_world2_b8t1024.json
```

`WORLD_SIZE` is a compile-time constant, so the target rebuilds both the fp32
and bf16 binaries before running; that rebuild is several minutes of Mojo
compilation and is *not* run time. (This matters when reading any timing on this
project: a "slow" invocation is usually a cold compile.) The runtime `-pn 2`
must match the compiled `WORLD_SIZE=2`; the ZeRO stage is a runtime flag `-z`.

`BENCH_ZERO_TIMEOUT` is new. It defaults to 400 s, which earlier production-shape
runs silently exceeded — those entries were recorded as `status="timeout"`, which
is easy to misread as a memory result when it is only a stopwatch.

**Using it without the benchmark harness.** While iterating on a buffer you do
not need the full sweep — set the variable on any run and read the lines:

```sh
LLMM_MEM_REPORT=1 WORLD_SIZE=2 ./build/train_gpt2 \
    -e gpt2_124M.bin \
    -i ./data/.tinyshakespeare/tiny_shakespeare_train.bin \
    -j ./data/.tinyshakespeare/tiny_shakespeare_val.bin \
    -b 4 -t 64 -x 4 -z 2 -pn 2 | grep '^\[mem-report\]'
```

Four lines come back (two phases × two ranks), each a JSON object. Diff the
`buffers` field against the committed baseline to see exactly what your change
moved. Note the binary is not rebuilt automatically — check its mtime against
your sources before trusting a number.

**What is subtracted.** `peak_mem_mib_max_delta` is the maximum `nvidia-smi`
reading during the run minus a per-GPU reading taken immediately before launch,
so co-tenant jobs on other GPUs are excluded. The exact accounting subtracts
nothing — it is an absolute sum of this rank's own buffers.

**Box conditions.** Both baseline runs were taken while the machine was
otherwise idle, before the other workstreams started their GPU work. That
matters far more for the step times than for the memory: memory is baseline-
subtracted and per-GPU, but step times are contended and should not be compared
against numbers taken under load. This document makes no throughput claims.

**Uncertainty.** The memory numbers here are not statistical estimates and carry
no error bars: `exact_total_bytes` is a sum of integers read from live buffer
objects, and it was byte-identical across both ranks and across the two shapes
wherever it should have been (see the consistency check below). The `nvidia-smi`
figures are quantized to 256 MiB, as established above — any difference between
two of them that is smaller than 256 MiB is not a measurement, and this document
does not treat one as such.

---

## Pre-change baseline

`main` @ b3e0a12, world size 2, `-b 4 -t 64`, 12 steps, all `status=ok`.
Steady-phase totals, MiB, maximum across the two ranks. Recorded in
`zero/bench/bench_zero_world2.json`.

| stage | prec | params | grads | optimizer | z3 win | activations | comm | **exact total** | `nvidia-smi` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | fp32 | 474.838 | 474.838 | 949.676 | 0 | 421.506 | 474.838 | **2795.725** | 3768 |
| 1 | fp32 | 474.838 | 474.838 | 474.838 | 0 | 421.506 | 474.838 | **2320.887** | 3256 |
| 2 | fp32 | 474.838 | 387.794 | 474.838 | 0 | 421.506 | 474.838 | **2233.843** | 3000 |
| 3 | fp32 | 237.419 | 387.794 | 474.838 | 177.419 | 421.506 | 474.838 | **2173.843** | 3000 |
| 0 | bf16 | 237.419 | 237.419 | 1424.514 | 0 | 210.943 | 237.419 | **2347.743** | 3256 |
| 1 | bf16 | 237.419 | 237.419 | 712.257 | 0 | 210.943 | 237.419 | **1635.486** | 2488 |
| 2 | bf16 | 237.419 | 193.897 | 712.257 | 0 | 210.943 | 237.419 | **1591.964** | 2488 |
| 3 | bf16 | 118.709 | 193.897 | 712.257 | 88.709 | 210.943 | 237.419 | **1561.964** | 2488 |

(`index` and `scratch` classes are omitted from the table: together they are
under 0.03 MiB at this shape. They are in the JSON.)

The stage curve behaves exactly as ZeRO predicts, and now you can see it happen
per class rather than inferring it: stage 1 halves `optimizer` (949.676 →
474.838 in fp32), stage 2 replaces the full gradient buffer with a pool plus a
shard (474.838 → 387.794), stage 3 halves `params` (474.838 → 237.419) at the
cost of the gather windows (+177.419).

### The two buffers under active work

| buffer | fp32 | bf16 | what it is |
| --- | --- | --- | --- |
| `grad_pool_buf` | 150.375 | 75.188 | largest simultaneous gradient bucket (wte+wpe), ZeRO-2/3 |
| `embed_window_buf` | 150.381 | 75.190 | ZeRO-3 embedding/head gather window |

Both sit **below** the 256 MiB grid. That is the whole problem in one line: the
two buffers this campaign exists to shrink are each smaller than one unit of the
old instrument's resolution, so any improvement to them was going to be
unmeasurable by that instrument no matter how large a fraction of the buffer was
removed.

### Two findings nobody was looking for

**`comm_scratch_total` is 474.838 MiB in fp32** — as large as the entire
parameter buffer, and paid at *every* stage including stage 0, where no
sharding happens at all. It is two staging buffers of one shard each
(`comm_scratch` and `comm_scratch2`, 237.419 MiB apiece). At fp32 stage 0 it is
tied with the parameter and gradient buffers as the largest class after the
optimizer state. Nobody had counted it before, because nothing counted anything
before.

**bf16 stage 0 spends more on optimizer state than fp32 stage 0 does**: 1424.514
MiB vs 949.676 MiB. Mixed-precision training keeps the Adam moments in fp32
*and* adds an fp32 master copy of the weights, so halving the parameter and
gradient buffers costs you a third full-size fp32 buffer. bf16 stage 0's exact
total (2347.743) is only 16% below fp32 stage 0's (2795.725), not the ~50% a
reader might expect from "half precision".

### Production-ish shape, and the honest caveat

Same world size, `-b 8 -t 1024`, 8 steps
(`zero/bench/bench_zero_world2_b8t1024.json`). The static footprint is unchanged
by construction — and measuring it as unchanged is a useful check that the
instrument does what it claims:

| shape | fp32 stage 0 static (`post_alloc`) | fp32 stage 0 steady |
| --- | --- | --- |
| `-b 4 -t 64` | 2374.192 MiB | 2795.725 MiB |
| `-b 8 -t 1024` | 2374.192 MiB | 20183.224 MiB |

Byte-identical static totals across a 64× larger token count, as they must be.

But look at the steady column. At the production shape, activations are 17808.188
MiB of a 20183.224 MiB total — **88%**. The model-proportional buffers that ZeRO
shards, and that this campaign shrinks, are the remaining 12%. Removing 150 MiB
is 6.3% of the static footprint and **0.7%** of the steady footprint at this
shape.

That is not an argument against doing it, but any claim made from these numbers
has to say which denominator it used.

### The terms that actually dominate

Because "activations" as a single 17808.188 MiB number is not actionable, the
report also breaks out the individual tensors that make it up. These are slices
of the one activation buffer, already inside the `activations` class — they are
broken out, not added, and must never be summed into a total.

| tensor | shape | fp32 @ `-b 8 -t 1024` | bf16 @ same |
| --- | --- | --- | --- |
| `att_probs` | (L, B, NH, T, T) | **4608.000 MiB** | 2304.000 MiB |
| `logits` | (B, T, V_p) | **1572.000 MiB** | 786.000 MiB |
| `fch_gelu` | (L, B, T, 4C) | 1152.000 MiB | 576.000 MiB |
| `logits_grad` | — | 0 | 0 |

Two things to take from this.

**The logits tensor is `B × T × padded_vocab_size`, so it scales with the batch
while the ZeRO-shardable state does not.** At `-b 4 -t 64` it is 49.125 MiB —
1.8% of the total. At `-b 8 -t 1024` it is 1572.000 MiB — 7.8%. At the `-b 32
-t 1024` shape used for real training runs it is linear in batch, so 4× again:
**~6.14 GiB in fp32, ~3.07 GiB in bf16.** Against that, a 150 MiB parameter-state
saving is about 2% of one tensor. Both facts are true simultaneously, and a
report that shows the 150 MiB without showing the 6 GiB next to it is
technically accurate and materially misleading. `padded_vocab_size` is emitted
in the JSON metadata so `B × T × V_p` can be checked by hand.

**`att_probs` is larger still** — 4608.000 MiB, three times the logits tensor,
because it goes as `T²` rather than `T`. It is the single largest tensor at this
shape.

I initially wrote that nobody had been looking at it. That was wrong, and the
correction is more useful than the claim was. `train_gpt2.mojo` (the "Store-P"
comment above the `attention_fwd` call) documents the tradeoff explicitly and
provides a working escape hatch: setting `kv_cache.att_probs_addr = 0` disables
the store and takes the true per-layer QKᵀ-recompute backward instead. That path
is correctness-verified (16/16), and the decision to keep the store was
deliberate and measured, not an oversight.

What is worth re-examining is narrower and better-founded. The measurement
behind that decision was **+3.5% step time (fp32 736.6→762.9 ms, bf16
587.1→606.8 ms) at `-b 4 -t 1024`, on Metal**, where the extra backward GEMM
costs more than the store saves with the current tensor-core kernels. Two things
have changed since. The comment names its own trigger condition — "if T-scaling
(`att_probs` grows as T²) ever makes the store dominate" — and this measurement
is evidence that condition is now met: at `-b 8 -t 1024` the store is the largest
single tensor on the card, 4608.000 MiB, and at the `-b 32 -t 1024` training
shape it scales to ~18 GiB in fp32. And the cost side was measured on Metal at a
batch 8× smaller; I have found no equivalent CUDA measurement, so whether the
3.5% figure transfers to this box at production shape is, as far as I can tell,
simply unknown.

So the honest statement is: this is a known, deliberate, documented tradeoff
whose *memory* side has now been quantified for the first time, and whose *cost*
side has not been measured on the hardware and at the shape where the tradeoff
actually matters. That is a cheap experiment — flip one assignment, rerun — and
it is a much larger number than anything else in this campaign. I have not run
it; it is outside this workstream's mandate and I am flagging it, not claiming it.

`logits_grad` is 0 in every row, which is not a bug and is worth stating because
it looks like one: the GPU backward path deliberately sizes the `logits`, `fch`
and `att_probs` gradient tensors to zero (they are dead on that path), which is
what keeps `grad_acts_buf` from being another multi-GiB buffer. That the
instrument reports exactly zero there is a small confirmation it is reading the
real size table rather than assuming symmetry with the forward.

Large-vocabulary logit and activation
memory dominating training footprint is a well-documented phenomenon — Liger
Kernel reports that at a batch of 8 and sequence length 4096, "the 256k
vocabulary size will result in a 16.8 GB logit tensor of precision bfloat16,
causing a huge spike in the peak memory usage" ([Liger], Hsu et al. 2024, §3.2
"FusedLinearCrossEntropy", <https://arxiv.org/abs/2410.10989>) — and this repo's
own numbers tell the same story at smaller scale.

### Does it reproduce?

The `-b 4 -t 64` sweep was run twice, hours apart, with the machine quiet the
first time and three other workstreams active on other GPUs the second. All
eight `exact_total` values came back **bit-identical** — not close, identical, to
the last of three decimal places in MiB. That is the expected result for a sum
of integers read out of live buffer objects, and it is the reason this document
is willing to quote memory differences of 30 MiB without error bars.

Timing is the opposite and is treated accordingly: step times moved between the
two runs with box contention, which is exactly why no throughput claim appears
anywhere here.

### `inactive_buffers` in practice

The distinction between "eliminated" and "small" is not theoretical either. At
`-b 4 -t 64`, fp32:

| stage | buffers not allocated at all |
| --- | --- |
| 0 | `embed_window_buf`, `grad_pool_buf`, `grad_shard_buf`, `master_buf`, `param_window_buf` |
| 3 | `grads_buf`, `master_buf` |

Read across those two rows and the whole ZeRO-3 restructuring is legible as
structure rather than as arithmetic: stage 3 has stopped allocating the
monolithic gradient buffer entirely and started allocating the pool, the shard
and both gather windows. In bf16 the stage-3 row lists only `grads_buf`, because
mixed precision genuinely does allocate `master_buf` — a difference that a
byte-delta view would have shown as two nearly identical small numbers.

### Where the residual goes

`exact_total` is always below `driver_used`, by design (see "What it does not
count"). The gap is 70–276 MiB at `-b 4 -t 64`, consistent with the 32 MiB
cuBLASLt workspace plus up to one 256 MiB chunk of allocator slack. At
`-b 8 -t 1024` it grows to ~2601 MiB (fp32) and ~1496 MiB (bf16), which is
consistent with the attention GEMM scratch: those buffers are sized
`batch × heads × T × T`, so at `T = 1024` each plane is 384 MiB in fp32 and 192
MiB in bf16, and roughly seven of them are cached.

To be clear, this scratch is *not* the `att_probs` tensor tabulated above.
`att_probs` lives inside `acts_buf` and is fully counted; the GEMM scratch is a
separate set of allocations cached as raw integer addresses in `KVCache`, which
is precisely why it has no `DeviceBuffer` field for the accounting to find. The
two happen to share the `B × NH × T × T` shape, which makes them easy to
conflate — they are different memory.

I want to be precise about the strength of that claim: it is an arithmetic
consistency argument, not a measurement. I did not instrument those allocations,
so I cannot attribute the residual to them. What I can say is that the residual
is fully accounted for in order of magnitude, that it is identical across ZeRO
stages at a fixed shape, and therefore that it cancels in the before/after
deltas this instrument is used to compute.

---

## The basis of every number (read before comparing any two)

A number that is correct under one reading and wrong under another, with the
reading unrecorded, is its own category of defect — call it an **unlabelled-basis
error**. Mixing MiB (2²⁰) with GB (10⁹); reporting a count that is inventory
when the reader hears capacity; quoting a min where a median is assumed;
per-rank where aggregate is assumed. All the same bug, and all invisible in the
value itself. Three of the four turned up in this campaign in one day.

So, explicitly, for this JSON:

| field | unit | reduction |
| --- | --- | --- |
| `peak_mem_mib_max_delta` | MiB = 2²⁰ B | max **over time** (0.25 s sampling), per GPU, then max **across GPUs**, minus a pre-launch baseline |
| `exact_total_mib_max` | MiB = 2²⁰ B | **single point in time** (the `steady` phase), max **across ranks** only — no temporal reduction |
| `static_total_mib_max` | MiB = 2²⁰ B | single point in time (`post_alloc`), max across ranks |
| `classes_mib_max`, `buffers_mib_max`, `tensors_mib_max` | MiB = 2²⁰ B | per key, single point in time (`steady`), max across ranks |
| `driver_used_mib_max` | MiB = 2²⁰ B | single point in time (`steady`), max across ranks; whole-device, includes co-tenants |
| `*_bytes*` | bytes | as above, unrounded |
| `gpu_count` | count | GPUs `nvidia-smi` **enumerates** — inventory, not capacity |

The row that matters: **the two headline numbers do not share a basis.** The
`nvidia-smi` figure is a temporal peak; the exact figure is a point-in-time
reading. Putting them on one chart is defensible — both quantities are
monotonically non-decreasing during a run (the caching allocator never releases,
and every buffer counted here is persistent), so the `steady` reading should be
the peak — but it is an argument, not a definition, and I have not independently
confirmed that `steady` is the maximum. Anyone plotting them together should say
so rather than let a reader assume both are peaks.

Per-rank, never aggregate: the ranks are symmetric here, so "max across ranks"
equals either rank's value. Do not multiply by world size.

## What this does and does not license you to claim

**Supported.** Byte-exact resident sizes of every buffer the model and the ZeRO
context own, per rank, at two phases; and therefore exact before/after deltas
for any change to those buffers, at any magnitude, down to single bytes.

**Not supported.** Peak *process* memory — use the `nvidia-smi` numbers, which
remain in the JSON and remain the right answer to "will this job fit". Transient
allocations inside kernels. Any throughput claim from these runs.

**The rule of thumb this campaign should adopt:** report the exact accounting
for the buffer that changed, report the `nvidia-smi` number for whether the job
got easier to fit, and never quote a sub-256 MiB difference between two
`nvidia-smi` readings as a result.

---

## AI use statement

Written with AI assistance (Claude Opus agent via Claude Code), directed by
Evan.
