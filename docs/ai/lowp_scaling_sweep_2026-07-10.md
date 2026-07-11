# Low-precision scaling sweep: where does FP8/NVFP4 beat bf16 on GB10?

**Date:** 2026-07-10. **Branch:** `main` (HEAD `d105fc3`, measured tree — no
code changes). **Machine:** NVIDIA GB10 (Grace-Blackwell, aarch64, unified
LPDDR5X ~273-301 GB/s — compute-rich, bandwidth-poor for this box's memory
system). MAMF (maximum achieved matmul FLOPs, task-provided): ~100 TFLOPS
bf16, ~208 TFLOPS fp8, ~416 TFLOPS fp4 (fp8 x2).

Context: `docs/ai/ai_assisted_optimizations_and_benchmarks.md` and the
README's Low-precision section establish the 124M/B=4/T=1024 baseline (bf16
133.9 / fp8 150.5 / fp4 184.2 ms/step — fp8 GEMMs measured only ~13.7%
faster than bf16's own, because at these shapes the M x N GEMM **output**
traffic is bandwidth/latency-bound and isn't reduced by quantizing the
*inputs*). This sweep asks: does scaling batch or model width push FP8/NVFP4
past bf16 anywhere reachable on this box, and how close does that get to the
2x/4x hardware ratios?

**Method:** `build/train_gpt2_{bf16,fp8,fp4}` at HEAD (fresh, mtime-checked
before the sweep), invoked via `scripts/run_train_gpt2_bf16.sh BIN=...`
against a FineWeb shard (`data/.fineweb10B/fineweb_{train_000090,val_000000}.bin`
— tinyshakespeare's stock shards are too small to source `B*T` tokens once
`B*T` exceeds ~32k, which is what first surfaced the loader's shared
`min_required = B*T+1` floor for the val loader). Each config: 20 steps
(`-x 20 -v 0 -s 0 -n 0`, default `total_batch_size` => `grad_accum_steps=1`),
first 5 dropped as warmup, 15 measured steps averaged (mean shown; medians
tracked closely throughout — no outlier-driven means). Precision order was
alternated per config row to cancel thermal/ordering drift; the `d36` B=4
crossover was independently re-run once (order reversed) to confirm it
wasn't noise. All runs serialized under `flock /tmp/llmm-gpu.lock`, one
config at a time.

## Table 1 — Batch sweep (`d12`, C=768, L=12, 124M params, T=1024)

| B | bf16 ms/step | fp8 ms/step | fp4 ms/step | bf16 tok/s | fp8 tok/s | fp4 tok/s | fp8/bf16 | fp4/bf16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4  | 133.6 | 150.6 | 182.5 | 30652 | 27186 | 22464 | 1.127x | 1.365x |
| 16 | 535.0 | 547.0 | 694.1 | 30663 | 29965 | 23617 | 1.022x | 1.297x |
| 32 | 940.4 | 1080.0 | 1356.8 | 34896 | 30335 | 24163 | 1.148x | 1.443x |
| 64 | 1888.3 | 2164.2 | 2691.9 | 34724 | 30427 | 24380 | 1.146x | 1.426x |

fp8 never crosses 1.0x at `d12` in this range. It gets closest at B=16
(1.022x — 2.2% slower than bf16, versus 12.7% slower at B=4), then the ratio
**worsens** at B=32/64 (1.15/1.15x) instead of continuing to improve — batch
scaling at fixed (small) width is not monotonic on this box (see analysis
(c)). fp4/bf16 stays in the 1.30-1.44x band throughout, also non-monotonic
and never close to crossing.

B=32/64 required the FineWeb data (tinyshakespeare's val shard, 32768
tokens, is one token short of what B=32,T=1024 needs); B=64 also briefly
stranded a large chunk of unified memory after the run exited (see Gotcha
below) — `free -g` "available" dropped from ~46 GB to ~33 GB and did not
recover within the session, consistent with the known GB10 exit-strands-
memory behavior on this box (self-recovers over hours; not a step-time
confound since it happened only *after* the last B=64 measurement).

## Table 2 — Width sweep (T=1024; B chosen per-config to fit the ~33 GB
available headroom left after the batch sweep — see Gotcha)

| descriptor | params | B | bf16 ms/step | fp8 ms/step | fp4 ms/step | fp8/bf16 | fp4/bf16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| d12 (C=768,  L=12) | 124M | 16 | 535.0  | 547.0  | 694.1  | 1.022x | 1.297x |
| d24 (C=1024, L=24) | 355M | 4  | 361.7  | 392.7  | 524.2  | 1.086x | 1.449x |
| d24 (C=1024, L=24) | 355M | 16 | 1280.8 | 1411.7 | 1883.2 | 1.102x | 1.470x |
| **d36 (C=1280, L=36)** | **774M** | **4**  | **862.2**  | **792.9**  | **1073.2** | **0.920x** | **1.245x** |
| **d36 (C=1280, L=36)** | **774M** | **8**  | **1582.8** | **1441.1** | **1974.5** | **0.910x** | **1.247x** |
| d36 (C=1280, L=36) | 774M | 16 | ABORTED | — | — | (thrashed, no step-1 in >5 min; killed) | |

**fp8 crosses 1.0x at `d36`, at both B=4 and B=8** — fp8 is ~8-9% *faster*
than bf16 wall-clock, and the ratio, if anything, improves slightly with
batch (0.920x -> 0.910x) rather than degrading the way it did for `d12`.
This is the headline result of the sweep: the crossover is a **width**
effect, not fundamentally a **batch** effect — `d24` at 4x the batch of the
`d36` runs (B=16) still doesn't cross (1.102x), while `d36` crosses at
B=4, the smallest batch tested anywhere in this sweep. fp4/bf16 never
crosses (best: 1.245x at `d36`/B=4-8).

`d36` at B=16 was attempted and aborted: activations 29.57 GB + activation
gradients 13.65 GB (~43 GB) plus ~12.4 GB of fp32 optimizer state (774M
params: bf16 params/grads 1.55+1.55 GB, fp32 master 3.1 GB, fp32 Adam m/v
6.2 GB) sums to ~56 GB against ~33 GB "available" — this did **not** fail
fast like the earlier tinyshakespeare-shard-too-small errors (2-8s); it sat
at 100% CPU for >5 minutes without completing step 1 and was killed
manually. `d48` was dropped from the grid outright (estimated ~85-95 GB
footprint at B>=4, far beyond headroom) per the task's memory-first
trim-the-grid guidance.

**Gotcha (GB10 unified memory):** after the B=64 batch-sweep runs, `free -g`
"available" fell from ~46 GB to ~33 GB and later to ~21 GB after the killed
`d36`/B=16 attempt, with no matching process in `ps aux` — the memory is
stranded, not leaked to a live process. This matches the previously-recorded
GB10 gotcha (killing a GPU tenant strands memory for hours, self-recovering
without a reboot); it did not block this sweep (all remaining configs fit
under the shrinking headroom) but capped how far the width sweep could push
`B` at `d36`/`d48`.

## MFU-derived precision comparison (best-ratio config: `d36`, T=1024)

No ncu profile was taken (skipped per the task's fallback — the shapes here
are non-default and ncu integration at arbitrary `-e`/`-b` is not wired up).
Instead, FLOPs/step is derived analytically (Kaplan et al., matching this
codebase's own `llmm/mfu.mojo`): `flops/token = 6*N + 6*L*C*T`,
`flops/step = flops/token * B*T`, `MFU = achieved_TFLOPS / MAMF_peak`.

| B | precision | ms/step | achieved TFLOPS | MAMF peak | MFU |
|---:|---|---:|---:|---:|---:|
| 4 | bf16 | 862.2 | 23.41 | 100 | 23.4% |
| 4 | fp8  | 792.9 | 25.45 | 208 | 12.2% |
| 4 | fp4  | 1073.2 | 18.81 | 416 | 4.5% |
| 8 | bf16 | 1582.8 | 25.50 | 100 | 25.5% |
| 8 | fp8  | 1441.1 | 28.01 | 208 | 13.5% |
| 8 | fp4  | 1974.5 | 20.44 | 416 | 4.9% |

fp8's achieved FLOPS/s is only ~1.09-1.10x bf16's (25.45/23.41,
28.01/25.50) — nowhere near the 2.08x MAMF ratio — which is exactly why
fp8's *MFU%* is roughly half of bf16's even where it wins on wall-clock:
it's winning by being a little faster in absolute terms while using a
denominator (peak) that grew much more. fp4's achieved FLOPS/s is
*lower* than bf16's (18.81/20.44 vs 23.41/25.50 TFLOPS) at both batch
sizes — fp4 never converts its 4x nominal peak into more delivered FLOPS/s
on this box at these shapes, consistent with it never crossing 1.0x
anywhere in the sweep.

## Analysis

**(a) Does fp8 cross 1.0x (beat bf16) anywhere in the sweep — where?**
Yes, exactly at `d36` (774M params, C=1280, L=36), at both B=4 (0.920x) and
B=8 (0.910x). It does not cross anywhere at `d12` (124M, best 1.022x at
B=16) or `d24` (355M, best 1.086x at B=4). fp4/bf16 never crosses anywhere
tested (best 1.245x at `d36`).

**(b) Best ratios achieved, and how far from 2x/4x hardware ratios?**
Best fp8/bf16 = 1/0.910 = **1.099x** *faster* (at `d36`/B=8) — versus a
2.08x MAMF hardware ratio, i.e. fp8 realizes roughly **9-10%** of its
nominal 2x throughput advantage as wall-clock speedup, leaving the other
~90% on the table to quantize/amax/scale overhead and unreduced GEMM-output
traffic. Best fp4/bf16 = 1/1.245 = **0.803x** — fp4 never beats bf16 at
all in this sweep, so its realized fraction of the nominal 4x ratio is
negative (0% at best, and often costs 25-45% *extra* time). The MFU-derived
achieved-TFLOPS numbers make this concrete: fp8 only delivers ~1.09x bf16's
raw FLOPS/s (not 2.08x), and fp4 delivers *less* raw FLOPS/s than bf16, not
4x more.

**(c) Does the data support "width > batch"?**
Yes, clearly. Per-width-step deltas in the fp8/bf16 ratio, holding B=4
fixed: `d12` 1.127 -> `d24` 1.086 (Delta -0.041) -> `d36` 0.920
(Delta -0.166). Both width steps move the ratio down (toward/through
1.0x), and the `d24`->`d36` step alone (-0.166) is bigger than the *entire*
`d12` B=4->B=64 batch sweep moved the ratio, in either direction. Per-doubling batch
deltas at fixed `d12` width: B4->B16 (2 doublings) -0.105/2 = -0.053/doubling,
then B16->B32 (1 doubling) **+0.126** (regression), B32->B64 (1 doubling)
-0.002 (flat) — batch scaling is small, inconsistent in sign, and net
roughly zero over the full B4->B64 range (1.127 -> 1.146). Width scaling at
fixed B=4 is monotonic and an order of magnitude larger per step. At `d36`
specifically, adding batch (B4->B8) is now a small *additional* win
(0.920->0.910) rather than the wash/regression batch alone produced at
`d12` — width appears to be what makes the GEMMs large enough for batch to
start compounding usefully, not a substitute for it.

**(d) GB10-specific verdict: is a 124M-class model ever compute-bound
enough, or does the crossover need the bigger configs?**
Within this sweep (B up to 64, the largest batch reachable before wall-time
and memory budget considerations), `d12` (124M) never crosses 1.0x for fp8
and the ratio is *non-monotonic* in B — there's no evidence that pushing B
further would reliably continue closing the gap, since B=32/64 already
regressed relative to B=16. The crossover genuinely required moving to the
774M-class `d36` config, not merely a bigger batch at 124M scale. Read
generously for the 124M model actually used in this repo's default configs:
on this box's ~300 GB/s unified-memory system, FP8 training at GPT-2-124M
scale is fundamentally memory/latency-bound territory that batch alone does
not resolve; FP8 becomes a real step-time win only once the model itself is
scaled to roughly 6x this repo's default (124M -> ~774M params).

## AI use statement

Written with AI assistance (Claude Code / Sonnet agent), directed by Evan Owen.
