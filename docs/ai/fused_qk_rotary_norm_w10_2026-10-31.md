# W10: fused rotary + QK-norm row kernels (2026-10-31)

Branch `mn/fuse` (from main 21a2e4c). Goal: remove the step-time cost of the combined recipe
(`-D LLMM_ROTARY=1 -D LLMM_QK_NORM=1`) by fusing the rotary and QK-norm passes, after profiling
where the cost actually goes.

## Contamination caveat

A 19 h headline run owned the GPU for all of this. `/tmp/llmm-gpu.lock` is held by that run's own
`flock`, so `gpu_slot.sh` would have blocked until the run ended; the W10 GPU jobs used a private lock
(`flock /tmp/llmm-gpu-w10.lock`) and coexisted with the headline run. Every absolute time below is
contaminated (kernel medians are inflated up to 3x; step times drift 2.26-2.60 s between pairs).
What is robust: the per-kernel MIN over the 60 instances (12 layers x 5 steps), and old-vs-new runs
interleaved back to back. A clean timing is still owed after the headline run ends.

## 1. Profile: where the extra time goes (before any code)

Setup: nsys `--trace=cuda`, d12, B=32, T=1024, bf16, `-d 32768` (one micro-batch per step), 5 steps,
main's builds (`build/old_base_bf16`, `build/old_comb_bf16`), two repeats each.

Per-layer per-kernel MIN (ms), idle-equivalent:

| kernel (combined build, before) | min ms/layer | calls/step |
|---|---|---|
| rotary fwd (token layout, whole 3C row grid) | 0.74-0.79 | 12 |
| qk_rows_fwd (norm, head layout) | 0.80-0.86 | 12 |
| qk_rows_bwd (norm bwd, recompute) | 1.39-1.44 | 12 |
| rotary inverse (d_qkv) | 0.73-0.78 | 12 |
| sum | 3.7-3.9 | = 44-46 ms/step |

Findings:

* The four row kernels cost about 45 ms per 32k-token micro-batch at idle (~1.9% of a 2.26 s base step).
  Under contention their medians are 2-4x the minimum (qk_rows_bwd median 4.0 ms vs min 1.4 ms).
* Wall step time, interleaved: combined 2.40-2.43 s vs base 2.26 s, i.e. +6% in this window (earlier
  window 2.60 vs 2.45, also +6%). The +12-14% in the three-seed doc compared against an idle base run;
  the median-of-runs +6% there matches this measurement.
* The rest of the +6% is NOT the row passes: `attention_soft` (forward softmax) median goes from
  ~10 ms to ~13.7 ms per layer in the combined build (~+44 ms/step), both before and after this change.
  Correction from the independent verification: this is NOT data dependent. The softmax kernel has no
  data-dependent control flow. On GPU, QK-norm turns on max-logit tracking
  (`enable_qk_clip(0.0, track_only=True)`), which adds one `Atomic.max` per row. The same flag-off bf16
  binary with `-qk_log 0` vs `1` (3 interleaved 5-step pairs, identical loss) gives a median of 9.6 vs
  13.6 ms/layer, so tracking alone appears to account for it. Same-address serialization is the
  hypothesized mechanism, not measured. Fusion cannot remove it. All GEMMs, layernorm, split/merge, attention_bwd are
  equal base vs combined within noise (the W7 split/merge layout passes exist in the base build too).
* Redirect: the plan (fuse the row passes) stands, but its ceiling is ~1.3 ms/layer = ~15 ms/step
  (0.7% of a step). The max-logit tracking cost is the larger remaining lever and is a separate item.

## 2. Design

* `qk_rows_fwd/bwd[dtype, target, do_norm, do_rot]` (llmm/qk_norm.mojo) take the rotary table and
  `table_rows`; `attention_fwd/bwd` take `rotary: Bool` plus `rot_table`, `rot_rows`.
* Forward (head layout after split): rstd from the raw row, `x * q_mul * rstd` then the half-truncated
  rotation (pairs `(i, i + hd/2)`, `i < hd/4`), one read and one write. Reference order (norm, then
  rotary). The old code rotated first (in place on `acts.qkv`) and normalized second; rotation is
  orthogonal per head so this is a refactor up to fp32 rounding.
* Backward: inverse rotation of the incoming gradient (transpose), then the norm backward with
  `u = x * rstd` from the RAW `acts.qkv`, `q_mul` folded as before. `acts.qkv` is no longer rotated in
  place, so the rotation is not re-done anywhere. The standalone `rotary_qk_rows` launches are removed
  from `train_gpt2.mojo` (the function stays for its test).
* Single-flag paths use the same kernels with comptime switches (`do_rot` False is the old code path
  verbatim). Flag-off: `comptime if qk_norm or rotary` falls through to the old `q_mul != 1` branch,
  no launch.
* `rotate_y1/y2` are shared from llmm/rotary.mojo so both users have the same source arithmetic. Results are not bit-identical: rotary-only CPU runs differ from main by about 1 ulp (34% of model words after 3 steps, median relative difference 1.1e-7) because LLVM contracts a different product into the FMA in each context.

## 3. Gates

Binaries: `build/new_comb_bf16` 16:17:21, `build/new_base_bf16` 16:18:31, fp32 CPU builds in
`$S/w10/f32` 16:15-16:17; sources last modified 16:10:25 (format only), commit ae97fe0.

| gate | result |
|---|---|
| flag-off CPU byte identity vs main build, d6, 3 steps, AdamW: `cmp model_3.bin state_3_0.bin` | identical |
| same, Muon (`-opt muon`) | identical |
| `make verify-cpu` flag-off / with both flags | `result gate=cpu status=pass` / pass |
| combined CPU old vs new, 3 steps | not byte identical (expected: single rounding), losses 10.886160/10.390451/10.203780 vs .../10.390450/10.203780 |
| `make verify-gpu` flag-off | pass |
| `MOJO_DEFINES="-D LLMM_QK_NORM=1"` | pass |
| `make verify-gpu-qknorm-scale` (0.5 scale arm) | pass |
| `-D LLMM_ROTARY=1` | pass |
| `-D LLMM_QK_NORM=1 -D LLMM_ROTARY=1` | pass |
| pytest GPU `tests/test_attention_equivalence.py tests/kernels` (MAX_USE_ACCELERATOR=1) | 42 passed |
| `make test-mojo` flag-off | `tests result=pass files=32 passed=314 failed=0` |
| `make test-mojo` with QK_NORM+ROTARY | 31 of 32 files pass; `test_zero_untied` fails to compile by design (`comptime assert not ROTARY`, its own comment says build it without rotary). It is not a W10 regression. `test_qk_rotary_fused` 11/11 in both. |
| new `tests/test_qk_rotary_fused.mojo` (float64 model, old two-kernel composition, CPU vs GPU, identity dims, q_mul 0.7, hd=64 so lanes hold both halves of a pair) | 11/11 |

### Mutants (separate copies of the tree, run against the new test and the verify arms)

| mutant | caught by |
|---|---|
| (a) fused backward applies the forward rotation instead of its transpose (GPU and CPU) | unit test: 5 fail (bwd cpu/gpu norm+rot, bwd cpu/gpu rot-only, fused-vs-old composition); `verify-gpu` rotary-only FAIL; `verify-gpu` both FAIL |
| (b) norm backward recomputes `u` from the rotated raw row (GPU kernel only) | unit test: 2 fail (GPU bwd norm+rot vs float64 model, GPU-vs-CPU); `verify-gpu` both FAIL; `verify-gpu` rotary-only passes, correct, because that arm has no norm |

## 4. Relative timing, same contention window (5 steps, median of steps 3-5, ms)

| pair | old (main) | new (fused) |
|---|---|---|
| combined, window 1 | 2429, 2397 (two runs) | 2371, 2397 |
| base (flag-off) | 2263 | 2247 |

The per-step difference (about -0.5 to -1.2% combined) is inside the window-to-window noise (~1%);
the expected gain from the kernel numbers is -15 ms/step (-0.6%). Per-kernel, min ms/layer:

| | old | new |
|---|---|---|
| forward row work | rotary 0.74 + qk_fwd 0.81 = 1.55 | fused 0.90 |
| backward row work | qk_bwd 1.40 + rotary inv 0.73-0.78 = 2.15 | fused 1.51 |
| per layer total | 3.7 | 2.4 |

Flag-off new vs old base: 2247 vs 2263 ms, no regression.

## Not verified / open

* A clean (idle GPU) step-time number; the gain is expected ~15 ms per 32k micro-batch.
* Reducing the max-logit tracking cost (about +48 ms/step at 32k tokens): untried ideas are reducing per block before the atomic, or testing against a cached value first.
* Gaps from verification: no rotary+QK-norm reference at a non-default scale (a mutant scaling only rotated pairs passes verify and is caught only by the unit test); only hd=64 is tested; Metal and fp8/fp4 with the flags are not built.
* No convergence run with the fused build (numerics changed only by single vs double rounding).

## AI use statement

Written with AI assistance (Claude Code, Sonnet 5.5 implementation agent), directed by Evan Owen.
