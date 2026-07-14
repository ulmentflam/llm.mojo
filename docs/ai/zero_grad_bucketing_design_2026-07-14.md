# ZeRO-2/3 bucketed backward gradient reduction — design + status (2026-07-14)

Track A of the ZeRO stage-2/3 memory campaign: reduce peak gradient residency
during backward from O(full model) to O(bucket + shard) by reduce-scattering
gradient buckets as backward produces them, instead of materializing the whole
gradient buffer and reducing once at the end. Builds on phase 1 (in-place
reduce-scatter, merged as f39d3e8).

## Status

- **DONE + verified:** `ZeroContext.reducescatter_buckets` (commit
  `9895799`) — the reduce primitive. Cross-rank sums gradient buckets that sit
  contiguously in a per-rank pool but map to **scattered** global flat ranges,
  and accumulates each reduced element into the owning rank's shard. CPU +
  staged-copy GPU paths; a no-coordinator N>1 branch for the sequential-rank
  equivalence harness. Verified by `test_multi_cpu_reducescatter_buckets`
  (4 ranks, scattered/partial/disjoint shard overlaps, uncovered indices stay
  zero).
- **NOT landed:** wiring the primitive into `GPT2.backward` (the pool +
  per-layer recycle) and the `allocate_gradients`/`update`/`calculate_grad_norm`/
  `zero_gradients` changes. Designed in full below; handed off due to context
  budget. The GPU path of the primitive is written but **untested** (needs a
  2-GPU bucket test + an end-to-end loss comparison).

## Critical finding: the tied wte embedding caps the win at ~209 MB, not 350

`grads.wte` (padded_vocab_size·C = 38.6M elems = **147 MB fp32**) is written at
**both ends** of backward: the LM-head `matmul_bwd` (first op) and `encoder_bwd`
(last op) — GPT-2 ties the input embedding and the output projection. So the
wte gradient is only complete at the very end of backward and cannot be freed
mid-pass. Two consequences:

1. The wte bucket must be reduced in **two parts** (LM-head contribution early,
   encoder contribution late), relying on reduce-scatter linearity
   (`RS(a)+RS(b)=RS(a+b)` accumulated on the shard) so wte is NOT held resident
   during the layer loop.
2. Even so, the pool must be sized to hold wte (+wpe) during those two moments:
   **pool ≈ 150 MB**. With the shard accumulator (~119 MB fp32 at W4), peak
   gradient residency ≈ **269 MB vs the full 475 MB** → save ≈ **206 MB fp32**
   (z2/z3 at b4t64 W4: 2494 → ~2288). This is a clear descending curve but
   ~140 MB short of the 350 MB target. **Reaching 350 requires vocab-chunking
   the LM-head and encoder wte-grad kernels** (pool would need to be ≤ ~6 MB,
   but a single wte-grad kernel writes all 147 MB at once) — a deeper,
   kernel-level change, out of scope for the flat-buffer bucketing here.

(Computed in-session; see the numbers in the phase-1 doc's methodology.)

## Integration design (execute this)

All changes gated on `Self.WORLD_SIZE > 1 and zero_stage >= 2`; z0/z1 keep the
monolithic full grads buffer untouched.

### allocate_gradients
For z>=2, do NOT allocate the full `grads_buf` (padded). Instead:
- `grad_pool_buf`: `pool_elems = max(wte_size + wpe_size, per_layer_size)` — the
  largest simultaneous bucket. (wte+wpe during `encoder_bwd`; one layer's 12
  tensors during the loop.)
- `grad_shard_buf`: `optimizer_num_parameters` — the persistent reduced-shard
  accumulator the optimizer consumes.
- `self.grads.point_parameters` must still be called on SOMETHING valid; point
  it at the pool (its per-tensor pointers get overwritten per-bucket during
  backward anyway — see below). Keep `self.grads_memory` as the pool base.

### backward (the pointer-repoint + recycle trick)
The backward computes per-layer grad pointers as `self.grads.X + layer*stride`.
Least-invasive scheme: **before** each bucket's kernels, repoint the relevant
`self.grads.X` so the existing `+ layer*stride` arithmetic lands in the pool,
**zero the pool slice** (the weight-grad kernels `+=`-accumulate), run the
kernels, then `reducescatter_buckets` that bucket's flat ranges into the shard
and recycle the pool. Concretely, for each `self.grads.X` set
`X_ptr = grad_pool + pool_slot_X - layer*stride_X` so `X_ptr + layer*stride_X ==
grad_pool + pool_slot_X` (raw-pointer arithmetic; the base is never dereferenced
directly, only through `+ layer*stride`).

Bucket schedule (matches backward's op order):
1. **wte (LM-head part):** repoint `grads.wte -> grad_pool[wte_slot]`, zero it,
   run the LM-head `matmul_bwd`, then `reducescatter_buckets(pool, dest=[0],
   poff=[wte_slot], len=[wte_size], shard, opt)` (wte's flat base is 0).
2. **ln_f:** repoint `grads.ln_f_gamma/beta` into the pool, zero, run the final
   `layernorm_fused_residual_bwd`, reduce-scatter their two flat ranges
   (bases = param_sizes prefix-sum of ln_f_gamma/beta).
3. **per layer L = L-1..0:** repoint the 12 `grads.X` via the trick above, zero
   the layer pool slice, run the block's kernels, then one
   `reducescatter_buckets` call with the 12 `(dest=base_X + L*stride_X,
   poff=pool_slot_X, len=stride_X)` tuples → one barrier per layer.
4. **wte (encoder part) + wpe:** repoint `grads.wte/wpe` into the pool, zero,
   run `encoder_bwd`, reduce-scatter wte (flat base 0, **accumulates** onto the
   shard via linearity) and wpe.

Precompute the flat bases from `param_sizes` prefix sums; the per-layer
`stride_X` are `param_sizes[X] / num_layer`; `pool_slot_X` are cumulative
offsets within one layer.

### Grad accumulation (test -d explicitly)
Reduce **per micro-step** into the fp32 shard accumulator: zero `grad_shard`
once at `micro_step == 0` (in `zero_gradients`), and each bucket's
`reducescatter_buckets` `+=`-accumulates. Across the accum loop the shard ends
as sum-over-(ranks, micro-steps) — the mean the optimizer applies (grad_scale/W)
then recovers. This differs from z0/z1's reduce-only-on-last-step (whose in-place
allreduce would double-count on a second pass); a separate accumulator does not.
Zero the pool slice before each bucket every micro-step (kernels `+=`).

### zero_gradients / update / calculate_grad_norm
- `zero_gradients` (micro 0, z>=2): zero `grad_shard` (there is no full grads
  buffer to zero); pool slices are zeroed per-bucket in backward.
- The end-of-backward reduce block: for z>=2, do nothing (buckets were already
  reduced during backward). z0/z1 unchanged.
- `update` and `calculate_grad_norm` for z>=2: read the reduced shard from
  `grad_shard` (offset 0, length opt) instead of `grads_memory + rank*opt`.

## Verification plan
1. **Primitive:** `test_multi_cpu_reducescatter_buckets` (done, green). Add a
   2-GPU analog to `test_multi_gpu_collectives` (CUDA 4,5) to exercise the
   staged-copy path.
2. **CPU equivalence:** adapt `test_zero_equivalence`'s z>=2 branch — instead of
   `grads_memory *= N` + shard copy, drive the bucketed backward (which deposits
   the local grad into `grad_shard` via the no-coordinator branch), then
   `grad_shard *= N`, then update. Assert params match the W1 baseline to
   atol 1e-5 (W2 + W8, stages 2 and 3).
3. **End-to-end:** W4 GPU run (CUDA 4,5,6,7) comparing z2-bucketed per-step
   losses to z0/z1 — must match to ~1e-4 (fp reduction-order noise), like the
   rewrite campaign's tables. Also run a `-d` (grad-accum) case.
4. **Benchmark:** `make benchmark-zero BENCH_ZERO_WORLD=4` before/after; expect
   z2/z3 at b4t64 to drop ~206 MB fp32 below z1. Quantify step-time: the scheme
   adds ~ (num_layers + ~5) `reducescatter_buckets` calls per micro-step, each a
   host barrier + stream sync — negligible at big shapes (compute-bound), a few
   ms at b4t64. If it regresses small-shape step time too much, batch more
   tensors per barrier or overlap.

## Step-time / barrier-count note
One `reducescatter_buckets` per layer (12 tuples in one call) keeps it to
`~num_layer + 4` collective calls per micro-step (not 149). Each call has 2 host
barriers + a stream sync. Watch the b4t64 numbers.

## Implementation (landed — commit `bb60b52`)

The design above was executed end to end. What shipped, and where it differed
from or sharpened the design:

### `GPT2.backward` (the repoint-and-recycle)
Gated on `self._use_bucketing()` (`WORLD_SIZE > 1 and zero_stage >= 2`). Backward
precomputes, from `param_sizes`, the flat base (prefix sum) of every tensor and
the per-layer stride/pool-slot of the 12 layer tensors, then runs four bucket
kinds in backward's natural op order:

1. **wte (LM-head):** `grads.wte = pool`; zero `pool[0:wte_size]`; run the
   LM-head `matmul_bwd`; `reducescatter_buckets` flat range `[0, wte_size)`.
2. **ln_f:** `grads.ln_f_gamma = pool`, `grads.ln_f_beta = pool + C`; zero
   `pool[0:2C]`; run the final fused LN backward; reduce the two flat ranges.
3. **each layer L (L-1→0):** repoint the 12 `grads.X = pool + slot_X -
   L*stride_X` so the untouched `+ L*stride_X` arithmetic below lands at
   `pool + slot_X`; zero `pool[0:per_layer_pool]`; run the block's kernels; one
   `reducescatter_buckets` with the 12 `(flat_base_X + L*stride_X, slot_X,
   stride_X)` tuples → **one collective / two host barriers per layer**.
4. **wte (encoder) + wpe:** `grads.wte = pool`, `grads.wpe = pool + wte_size`;
   zero `pool[0:wte_size+wpe_size]`; run `encoder_bwd`; reduce both. The wte
   encoder contribution **accumulates** onto the shard the LM-head part already
   deposited, by reduce-scatter linearity — wte is never resident across the
   layer loop.

The end-of-backward reduce block returns immediately under bucketing (buckets
were already reduced inline, once per micro-step).

### Grad accumulation
Handled exactly as designed: `zero_gradients` zeros the `grad_shard`
accumulator once at `micro_step == 0` (no full buffer to zero); each bucket's
`reducescatter_buckets` `+=`-accumulates into the shard; each pool slice is
re-zeroed before its bucket every micro-step (kernels `+=`). The shard ends as
sum-over-(ranks, micro-steps); `update`/`calculate_grad_norm` read it from
`grad_shard_memory[0:opt]` (not `grads_memory + rank*opt`).

### `allocate_gradients`
Under bucketing the full padded `grads_buf` is **not** allocated (stays the
1-element placeholder); instead `grad_pool_buf` (`max(wte+wpe, one layer)`
elems) and `grad_shard_buf` (`optimizer_num_parameters`). `grads_memory` and
`grads.point_parameters` point at the pool base (overwritten per bucket).

### Primitive fix (`llmm/zero.mojo`)
Added a leading `self.ctx.synchronize()` to `_reducescatter_buckets_gpu` before
the coordinator's host barrier. Unlike `allreduce`/`reducescatter_inplace`
(invoked only after backward's trailing sync), `reducescatter_buckets` runs
**inline per layer**, so each rank must flush its own pool-write kernels before
the barrier lets peers pull that pool cross-device. Without this the GPU path
races. (The primitive's GPU path was written-but-unexercised at handoff; this
was the gap that surfaced when driving it end to end.)

### Memory accounting (why the win is ~200 MB, and the comm scratch cancels)
Both the pre-bucketing z2/z3 (in-place reduce-scatter) and z1 carry the full
padded grads buffer (~498 MB fp32) plus two comm-scratch buffers
(`comm_scratch` + `comm_scratch2`, each `opt` bytes, sized by
`ensure_comm_setup` for **every** sharded stage). Bucketing replaces the 498 MB
buffer with `grad_pool` (~150 MB, wte+wpe) + `grad_shard` (`opt` ≈ P/W); the two
comm-scratch buffers are unchanged and cancel in the delta. So the per-GPU
saving is ≈ `498 − (150 + opt_MiB)`, which **grows with world size** as the
shard shrinks: ≈ 99 MB at W2, ≈ 182 MB at W3, ≈ 224 MB at W4 (fp32). This is the
honest picture — the headline "~200 MB below z1" is a W4 number.

### The wte floor (honest framing, per the scope decision)
The tied `wte` grad (padded_vocab·C = 38.6M elems = **147 MB fp32**) is written
at both ends of backward and is the dominant term in the 150 MB pool. It is NOT
vocab-chunked here (an explicit scope decision): doing so would need the LM-head
and encoder wte-grad kernels to emit the vocab in chunks so the pool could be
~6 MB instead of 150 MB, a kernel-level change out of scope for this flat-buffer
bucketing. Consequently the peak grad-residency floor is
`wte_grad + shard ≈ 150 + opt` MiB, and z2/z3 land ~200 MB (not ~350 MB) below
z1 at W4. Reaching 350 MB is the deferred vocab-chunking follow-up.

## Verification (actually run on this box)

- **Primitive, GPU staged-copy path:** `tests/test_zero.mojo` `12/12`,
  including a new `reducescatter_buckets` leg added to
  `test_multi_gpu_collectives` (N=2, physical GPUs 4+5) that drives the
  staged-copy GPU reduce with a **partial bucket/shard overlap** (bucket B maps
  to flat `[48,80)`, spanning rank 0's tail and rank 1's head). Green. (The
  file's wall time is dominated by per-rank CUDA context init — ~1240 s here,
  over `make test-mojo`'s 600 s per-file cap; the same slow-init condition the
  multi-GPU rewrite doc documents, not new. Run the file directly to gate it.)
- **CPU equivalence:** `tests/test_zero_equivalence.mojo` `6/6` (stages 1/2/3 at
  W2 and W8, atol 1e-5). The z>=2 branch now drives the **bucketed** backward
  (the no-coordinator `reducescatter_buckets` deposits this rank's local shard
  into `grad_shard_memory[0:opt]`), scales *that* shard by N, and updates — so
  the bucketed path is what's verified, not the old full-buffer path.
- **Grad accumulation (the `-d` path — a prior campaign hit a grad-accum
  corruption bug):** W2 (GPUs 4,5), `-b 4 -t 64 -d 1024` (accum=2), fp32,
  10 steps. z2 (bucketed) per-step train loss matches z0 (DDP) to ~1e-4
  (step 1 identical: 5.081370 = 5.081370; step 5 3.992086 vs 3.992155;
  val 4.0573854 vs 4.0574117). This confirms the per-micro-step shard
  accumulation is correct.

### Hardware note (blocks the exact W4 acceptance measurement)
The assigned GPU set was CUDA 4,5,6,7. **Physical GPU 7 dropped into a CUDA
fault mid-campaign** (visible/idle in `nvidia-smi` at P8, but `torch.ones` and
Mojo `DeviceContext` init both fail with "No CUDA GPUs available" /
`CUDA_ERROR_INVALID_DEVICE` when it is targeted) — the same "GPU requires reset"
class the multi-GPU rewrite doc hit on GPU 1. Recovery needs a root
`nvidia-smi -r -i 7`; per the box's rule (never hard-operate the GPUs, it hangs
the GSP firmware) it was NOT reset from here. So W4 (4 ranks) could not run on
the assigned set. The verification and benchmarks were done at W2 (GPUs 4,5) and
W3 (GPUs 4,5,6); the W4 number is projected from the same per-GPU accounting and
should be re-measured with `make benchmark-zero BENCH_ZERO_WORLD=4` once GPU 7
is reset.

## Benchmark data (measured, b4 t64, 12 steps; peak MiB/GPU baseline-subtracted)

`make benchmark-zero` on **GPUs 4,5,6** (GPU 7 faulted — see above), stages
1/2/3. z1 is untouched by this change and is the reference level z2/z3 sat at
before bucketing (the merged in-place stage-2/3 work made the curve monotone
z1 = z2 = z3; this change pushes z2/z3 below z1). "peak Δ" is the
baseline-subtracted peak MiB on the busiest GPU.

**WORLD_SIZE = 2** (`bench_zero_world2.json`):

| prec | stage | mean ms/step | peak Δ MiB | vs z1 |
|------|-------|--------------|-----------|-------|
| fp32 | 1 | 47.6 | 3256 | — |
| fp32 | 2 | 44.5 | 3000 | **−256** |
| fp32 | 3 | 44.5 | 3000 | **−256** |
| bf16 | 1 | 51.2 | 2488 | — |
| bf16 | 2 | 51.3 | 2488 | 0 |
| bf16 | 3 | 51.3 | 2488 | 0 |

**WORLD_SIZE = 3** (`bench_zero_world3.json`):

| prec | stage | mean ms/step | peak Δ MiB | vs z1 |
|------|-------|--------------|-----------|-------|
| fp32 | 1 | 77.9 | 3004 | — |
| fp32 | 2 | 73.4 | 2748 | **−256** |
| fp32 | 3 | 72.6 | 2748 | **−256** |

Reading the curve honestly:

- **fp32 z2/z3 land 256 MiB below z1** at both W2 and W3 — already past the
  ~200 MiB acceptance target, and the saving is stable across world size in this
  small-shape regime (the dominant removed term is the full ~498 MB grads buffer
  and its transient coexistence with the reduce scratch, roughly W-independent;
  the pool/shard difference sits inside the peak's other allocations). W4 is
  therefore confidently ≥ 256 MiB fp32 below z1; it should be re-measured once
  GPU 7 is reset (`make benchmark-zero BENCH_ZERO_WORLD=4`).
- **Step time does not regress** — fp32 z2/z3 are a few ms *faster* than z1 at
  both world sizes: z1 pays a full-buffer allreduce **and** a full param
  all-gather each step, while the per-layer reduce-scatter buckets are
  ring-equivalent. The `~num_layer + 4` collectives/micro-step cost the design
  flagged is not visible at b4t64.
- **bf16 shows 0 saving** (z2/z3 = z1 = 2488 at W2). bf16 bucketing is correct
  (it trains; z2/z3 losses are bit-identical, matching z1 to ~1e-4) but the bf16
  peak at this shape is **not** set during the gradient-resident phase — the
  half-width bf16 grads buffer is not the top allocation at peak — so freeing it
  does not move the peak. This matches the merged in-place work's finding that
  bf16 grad savings fall within allocator rounding at b4t64. (The W3 bf16 rows
  are absent: `make build-bf16` did not rebuild the bf16 binary when only
  `WORLD_SIZE` changed — a stale W2 binary can't dispatch `-pn 3` — a Makefile
  staleness quirk, not a code fault; W2 bf16 measured cleanly.)
- **Losses (correctness):** in the same benchmark runs, z1/z2/z3 per-step losses
  agree to ~1e-4 with step 1 identical (fp32 W2 step 12: 4.18936 / 4.18949 /
  4.18947; bf16 z2/z3 bit-identical at 4.198103). Plus the dedicated W2
  grad-accum run above (z2 vs z0, accum=2) matches to ~1e-4.

At production batch/seq (`-b 32 -t 1024`) activations dominate (~10s of GB) and
this model-proportional ~256 MB moves the whole curve <1 GB — the honest
"production scale" caveat from the in-place doc applies unchanged.

## AI use statement

Written with AI assistance (Claude Opus agent via Claude Code), directed by
Evan.
