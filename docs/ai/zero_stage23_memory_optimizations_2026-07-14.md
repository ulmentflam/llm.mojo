# ZeRO stage-2/3 memory optimizations (2026-07-14)

Follow-up to the multi-GPU ZeRO campaign
([`zero_multigpu_rewrite_2026-07-14.md`](zero_multigpu_rewrite_2026-07-14.md)),
which delivered working 8-device data parallelism but noted that ZeRO-2/3
"do NOT save more memory in this implementation" and left the two real
memory wins — bucketed backward gradient reduction and per-layer parameter
streaming — as explicit follow-ups. This is that follow-up.

## Headline (read this first)

- **Landed and verified:** ZeRO-2/3 now reduce-scatter gradients *in place*
  into `grads_memory` instead of into a separate optimizer-sized shard
  buffer. This removes the measured regression where ZeRO-2/3 peak sat
  *above* ZeRO-1 (W4 fp32: z2/z3 2750/2752 MiB vs z1 2494), so the per-stage
  curve is now **monotone** (z0 > z1 = z2 = z3). Numerics unchanged
  (equivalence tests green, atol 1e-5); step time within noise.
- **NOT landed (scoped to a follow-up, with a concrete design below):**
  pushing z2/z3 *below* z1. That requires not materializing the full
  gradient / parameter buffers during backward/forward. In this codebase
  those buffers are a single monolithic **tensor-major** allocation that the
  backward/forward kernels write/read by pointer; capturing the peak win is a
  layout re-architecture (backward gradient bucket buffer + a
  gather-a-flat-range collective + reworking the primary CPU correctness
  gate) that could not be implemented *and verified to equivalence grade* in
  one session on this box. Rather than ship an unverified rewrite of the
  gradient path, this doc lands the correct incremental win and writes the
  full design down. See "Why the deeper win is a re-architecture" below.
- **Production-scale reality:** at `-b 32 -t 1024` activations dominate
  (~83 GB/GPU fp32 for this 124M model at W4), so the entire ZeRO
  optimizer/grad/param curve moves within <1 GB. ZeRO shards
  *model-proportional* state; the visible ZeRO memory win is largest when the
  activation footprint is small. At production batch/seq the lever is
  activation checkpointing (orthogonal to ZeRO), not stage 2/3.

## Optimization 1 — in-place reduce-scatter for the ZeRO-2/3 gradient

### The problem

`allocate_gradients` sized a full `grads_buf` (`padded_num_parameters`) for
every stage, and *additionally* allocated a separate
`sharded_grads_buf` (`optimizer_num_parameters`) for stage >= 2. Backward
reduce-scattered `grads_memory -> sharded_grads_memory`, and the optimizer
read the shard from `sharded_grads_memory`. So ZeRO-2/3 carried the full
gradient buffer **plus** an extra gradient shard that ZeRO-1 does not — their
peak sat above ZeRO-1's rather than at/below it (a small anti-win: the stages
that shard *more* used *more* memory).

### The fix

The reduced gradient a rank actually needs is exactly the slice the optimizer
reads at `grads_memory + rank*opt`. So do the reduce-scatter **in place**:

- New `ZeroContext.reducescatter_inplace(ptr, size)` = the existing in-place
  `allreduce`'s phase 1 without the follow-up all-gather. Rank r's own slice
  `[r*shard, (r+1)*shard)` of `ptr` is overwritten with the cross-rank SUM;
  every other slice keeps this rank's local (unreduced) contribution. In-place
  is safe by slice-disjointness (rank r writes only slice r; peer p reads only
  slice p of r's buffer, untouched by r; r's own read of slice r precedes its
  write per element) — the same argument the in-place allreduce already relies
  on. CPU and NVIDIA-staged-copy paths both implemented.
- ZeRO-2/3 backward calls `reducescatter_inplace(grads_memory,
  padded_num_parameters)`; the optimizer and the grad-norm read the reduced
  shard from `grads_memory + rank*opt` — **identical addressing to ZeRO-1's
  post-allreduce read**. The stage-1 and stage-2/3 optimizer/grad-norm paths
  are now unified.
- `sharded_grads_buf` / `sharded_grads_memory` deleted (struct field, init,
  allocation, and the per-step fill in `zero_gradients`).

### Measured (W4, physical GPUs 3–6; baseline-subtracted peak MiB/GPU)

Small shape, `-b 4 -t 64`, tinyshakespeare, 12 steps:

| prec | stage | before MiB | after MiB | Δ MiB | before ms | after ms |
|------|-------|-----------|-----------|-------|-----------|----------|
| fp32 | 0 | 3264 | 3262 | -2 | 63.3 | 63.3 |
| fp32 | 1 | 2494 | 2494 | 0 | 86.4 | 86.8 |
| fp32 | 2 | 2750 | 2494 | **-256** | 61.3 | 61.4 |
| fp32 | 3 | 2752 | 2494 | **-258** | 61.5 | 63.1 |
| bf16 | 0 | 3008 | 3006 | -2 | 60.5 | 60.5 |
| bf16 | 1 | 1984 | 1982 | -2 | 71.3 | 71.0 |
| bf16 | 2 | 1984 | 1982 | -2 | 58.3 | 58.5 |
| bf16 | 3 | 1984 | 1982 | -2 | 58.5 | 58.3 |

fp32 z2/z3 drop 256/258 MiB onto the z1 level; the curve is now monotone.
(bf16 was already flat before — its gradient shard is half the fp32 size and
fell within allocator rounding.) Step time is unchanged within run-to-run
noise. The measured fp32 saving (~256 MiB) exceeds the shard buffer's nominal
~124 MiB because dropping the separate output also removes a transient peak
where the full gradient buffer and the shard output coexisted during the
reduce.

Big shape, `-b 32 -t 1024`, FineWeb, 8 steps (activations dominate):

| prec | stage | before MiB | after MiB | Δ MiB | after ms | after tok/s |
|------|-------|-----------|-----------|-------|----------|-------------|
| fp32 | 0 | n/a* | 83392 | — | 466.4 | 281024 |
| fp32 | 1 | n/a* | 82624 | — | 491.7 | 266561 |
| fp32 | 2 | 82880 | 82624 | **-256** | 471.1 | 278241 |
| fp32 | 3 | 82880 | 82624 | **-256** | 475.4 | 275736 |
| bf16 | 0 | 43970 | 44795 | +825† | 254.1 | 515751 |
| bf16 | 1 | 42690 | 42690 | 0 | 259.8 | 504456 |
| bf16 | 2 | 42946 | 42690 | **-256** | 251.2 | 521770 |
| bf16 | 3 | 42946 | 42690 | **-256** | 250.3 | 523632 |

*The fp32 z0/z1 *before* runs timed out during the sweep (another team's
FineWeb production run landed on neighbouring physical GPUs mid-measurement,
per the coordinator's allocation note); the z2/z3 rows — the ones this change
touches — completed cleanly. †bf16 z0 is unsharded, so its before/after
should match; the +825 is activation/sampling noise from the same
contention, not a real change.

The signal is clean and consistent with the small shape: **z2/z3 drop exactly
256 MiB** (fp32 and bf16) — the same model-proportional gradient shard buffer,
independent of B/T. Against the ~83 GB (fp32) / ~45 GB (bf16) activation peak
that is a rounding error: the whole z0→z3 curve moves under 1 GB. This is the
honest "production scale" picture — see the reality note above.

## Why the deeper win (z2/z3 below z1) is a re-architecture, not a patch

The mission's headline items — bucketed backward reduction (free full-grad
storage as backward produces buckets) and per-layer param streaming (gather
params just-in-time, free after use) — both require the full gradient /
parameter buffer to **not be resident** during backward / forward. In this
codebase they are one monolithic buffer each, and the layout defeats every
scheme that would free part of it mid-pass:

- **Parameters/gradients are tensor-major, not layer-major.**
  `ParameterTensors.point_parameters` lays the 16 tensors out back to back,
  each spanning *all* layers: `[wte][wpe][ln1w·L][ln1b·L][qkvw·L]…[lnf]`.
  Per-layer access is `base_tensor + layer*stride` — a **strided** slice
  inside a tensor block, not a contiguous model region.
- **Backward completion order vs. flat ranges.** Backward runs layers L-1→0.
  A single (layer, tensor) slice *is* contiguous and completes when that
  layer's kernel runs — but a *whole layer's* gradient is scattered across 12
  far-apart tensor blocks, and any *contiguous flat range* of the buffer
  (which is what a reduce-scatter shard is) receives its last contribution
  only at layer 0, i.e. at the very end of backward. So **no contiguous flat
  range can be reduced-and-freed early** — the property bucketing needs.
- **Forward access is strided too.** The forward reads `wte`, then layer 0's
  12 tensors, then layer 1's 12 tensors… i.e. it jumps across tensor blocks
  per layer. No contiguous half of the flat parameter buffer is "used then
  done," so coarse parameter streaming buys nothing; only per-(layer,tensor)
  JIT gather reduces resident params.
- **The optimizer/reduce-scatter shard is a flat slice** `[r*opt,(r+1)*opt)`
  that cuts across tensor boundaries arbitrarily, so any bucketing must map
  each bucket's flat sub-range onto the (generally 1–2) ranks whose shards it
  overlaps.

### Concrete design for a future campaign

1. **Layer-major gradient bucket (ZeRO-2/3).** Give backward a small reused
   *bucket* buffer (largest bucket ≈ `wte` grad ~154 MiB fp32, or one
   transformer layer ~28 MiB), and repoint the backward's `d_l_*` grad
   pointers into it. As each bucket completes (wte right after the LM-head
   backward; each layer's 12 tensors right after that layer), reduce-scatter
   its flat sub-range into the optimizer shards via a new
   `reducescatter_range(bucket, flat_start, len)` and reuse the buffer. Peak
   grad storage becomes O(bucket + shard) instead of O(model). **Grad-accum
   interaction:** reducing only on the last micro-step (the semantics the
   merged campaign fixed) needs the full accumulated gradient resident, which
   defeats bucketing; so bucket only when `grad_accum_steps == 1` (the common
   / benchmark case) and fall back to the full-buffer path otherwise —
   correctness preserved, peak win only when there is no accumulation.
2. **Per-layer parameter streaming (ZeRO-3).** Persist params as the shard
   only; add `allgather_range(dst, flat_start, len)` (each rank contributes
   the intersection of the requested flat range with its shard). Before layer
   L's forward/backward, gather its 12 tensor slices into a per-layer buffer
   and repoint `self.params.*` reads there; free after the layer.
   `wte`/`wpe`/`ln_f` (needed by the encoder and LM head) stay resident.
   Resident params drop from ~498 MiB to ~max(wte 154, one layer 28) MiB fp32.
   Overlap (double-buffered gather of L+1 during compute of L) is a further
   step. **Cost:** ~12·L range-gathers per forward and per backward — without
   overlap this is well past "a small %" step-time, so overlap is not
   optional at production speed.
3. **Verification.** Both change *where/when* buffers live, not the math, so
   the existing `test_zero_equivalence` (which keeps full buffers resident and
   fakes the collectives with `coordinator=None`) does **not** cover them.
   A faithful test is a true multi-rank (threaded) CPU `GPT2` forward/
   backward/update through a real `CpuCoordinator` — a new harness, since
   `GPT2` is not `Movable` and today's test simulates ranks sequentially.
   Building that gate is part of the campaign, not an afterthought; shipping
   the gradient-path rewrite without it would violate the repo's
   "don't claim success without running it" bar.

## Verification (actually run on this box)

- `tests/test_zero_equivalence.mojo`: **6/6** (stages 1/2/3 at W2 and W8, each
  matching the WORLD_SIZE=1 baseline to atol 1e-5). The stage-2/3 branch was
  updated to read the reduced shard from `grads_memory + rank*shard` (the
  `*N` step already places the summed shard there) instead of copying into
  `sharded_grads_memory`.
- `tests/test_zero.mojo`: **11/11**, including a new
  `test_multi_cpu_reducescatter_inplace` (4-rank CPU) and an in-place
  reduce-scatter leg added to the N=2 multi-GPU collective test (physical
  GPUs 0+3), which drives the real staged-copy in-place reduce across two
  devices.
- `make benchmark-zero BENCH_ZERO_WORLD=4` before/after on physical GPUs
  3–6 (CUDA_VISIBLE_DEVICES=2,3,4,5), both shapes — tables above. GPU 1 is
  hardware-faulted and was never scheduled. W8 was not re-measured (only 7
  GPUs healthy; W4 before/after is the acceptance measurement, per the
  mission).

## Gate

`make format` (no diff), `make lint-mojo` (clean), pre-commit `make lint`
(ruff / mojo format --check / latexindent / pyrefly) passed on commit; the
CPU equivalence + collective tests above are the numerics gate.

## Commits

- `3ce982e` zero: in-place reduce-scatter for ZeRO-2/3 grads (drop redundant
  shard buffer)
- (this doc + W4 before/after benchmark JSONs, both shapes)

## AI use statement

Written with AI assistance (Claude Opus agent via Claude Code), directed by
Evan.
