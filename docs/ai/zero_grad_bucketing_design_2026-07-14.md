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

## AI use statement

Written with AI assistance (Claude Opus agent via Claude Code), directed by
Evan.
