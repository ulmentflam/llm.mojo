# Muon hyperparameter sweep, GPT-2 124M (d12), 300-step ranking runs

A Muon-specific sweep after [`muon_vs_adamw_124m_2026-10-29.md`](muon_vs_adamw_124m_2026-10-29.md),
which ran Muon with AdamW's recipe unchanged. This is a **ranking, not a convergence** study, in the
spirit of [`hyperparam_tuning_8gpu_2026-07-14.md`](hyperparam_tuning_8gpu_2026-07-14.md).

## Setup

Single GB10, bf16, seed 42, `make train-bf16`. Fixed recipe for every run:

```
-e d12 -i 'data/.fineweb10B/fineweb_train_*.bin' -j 'data/.fineweb10B/fineweb_val_*.bin' \
-b 32 -t 1024 -d 131072 -u 50 -q 0.0 -gc 1.0 -x 300 -v 100 -m 20 -s 0 -qk_clip 0 <varying flags>
```

300 steps x 131072 tokens = 39M tokens, 50 warmup steps, cosine to zero over 300 steps, val (20 batches)
at 100/200/300. Muon uses Moonlight RMS scaling, so `-l` and `-c` (decoupled weight decay) are shared
with AdamW. **Score = val loss at step 300** (primary), mean train loss over steps 251-300 (secondary).
Each run ~20 min (per-run median 3.86-4.02 s/step). Driver: `scripts/sweep_muon.sh` (one run) with an ad-hoc greedy
python driver for stage selection (not committed; stage picks below are reproducible from the table); table via `python3 scripts/compare_runs.py --table sweeps/muon_*.jsonl`.
Logs: `sweeps/muon_<tag>.jsonl`. Divergence rule (NaN or loss > 8 after step 50): no run diverged (max train loss after step 50 was 7.13).
Every run has 300 steps, the same step-1 loss (11.013952, i.e. the same seed-42 init; the seed is a
compile-time constant, not a logged field) and lr ~1e-7 or less at step 300.

What the logs do and do not record: each jsonl `setup` line records `opt`, `lr`, `muon_momentum`,
`grad_clip`, `qk_clip`, B/T/batch and steps, and those match every tag below. It does **not** record
weight decay (`-c`), so the wd values of `wd0`, `wd0.2`, `mu0.9` (0.1) and the AdamW control rest on the
tags and the driver, not on the logs. The three wd runs have identical setup lines but diverge from
step 2, consistent with differing `-c`.

## Results

Greedy, one factor at a time; each stage uses the previous stage's winner. Italic rows in parentheses
repeat the carried-over base config (same run, not a rerun).

| stage | tag | flags (varying) | val@100 | val@200 | val@300 | train 251-300 |
|---|---|---|---:|---:|---:|---:|
| 1 lr | lr3e-4 | `-l 3e-4 -mu 0.95 -c 0.1` | 6.4142 | 6.0498 | 5.9782 | 5.9221 |
| 1 lr | lr6e-4 | `-l 6e-4` | 6.2066 | 5.8344 | 5.7381 | 5.6844 |
| 1 lr | lr1.2e-3 | `-l 1.2e-3` | 6.0643 | 5.6468 | 5.5134 | 5.4630 |
| 1 lr | lr2.4e-3 | `-l 2.4e-3` | 5.9970 | 5.4215 | 5.2113 | 5.1649 |
| 1b lr ladder | **lr4.8e-3** | `-l 4.8e-3` | 5.9976 | 5.2780 | **5.0732** | 5.0307 |
| 1b lr ladder | lr9.6e-3 | `-l 9.6e-3` | 6.0479 | 5.3194 | 5.1029 | 5.0617 |
| 2 mu | *(mu0.95 = lr4.8e-3)* | `-mu 0.95` | 5.9976 | 5.2780 | 5.0732 | 5.0307 |
| 2 mu | **mu0.9** | `-l 4.8e-3 -mu 0.9 -c 0.1` | 5.9985 | 5.2405 | **5.0343** | 4.9933 |
| 2 mu | mu0.98 | `-mu 0.98` | 6.0011 | 5.3548 | 5.1730 | 5.1263 |
| 3 wd | *(wd0.1 = mu0.9)* | `-c 0.1` | 5.9985 | 5.2405 | 5.0343 | 4.9933 |
| 3 wd | **wd0** | `-l 4.8e-3 -mu 0.9 -c 0` | 6.0080 | 5.2289 | **5.0265** | 4.9866 |
| 3 wd | wd0.2 | `-c 0.2` | 6.0030 | 5.2509 | 5.0391 | 4.9978 |
| 4 grad clip | **gc0** | `-l 4.8e-3 -mu 0.9 -c 0 -gc 0` | 5.9712 | 5.2180 | **5.0121** | 4.9709 |
| 5 control | adamw6e-4 | `-opt adamw -l 6e-4 -c 0.1` | 6.5652 | 6.2136 | 6.1181 | 6.0631 |

The lr ladder was extended by doubling after 2.4e-3 because the optimum was not bracketed; it turned
over at 9.6e-3 (the ladder stopped there). Best Muon run 5.0121 vs AdamW control 6.1181 (-1.11).

## Recommended Muon recipe

```
-opt muon -l 4.8e-3 -mu 0.9 -c 0 -gc 0
```

This is the best config **at 300 steps**. For a long run keep regularisation on unless re-tested:
`-opt muon -l 4.8e-3 -mu 0.9 -c 0.1 -gc 1.0` (the wd/gc evidence below is weak, and the best lr may be
lower at a longer horizon).

Evidence, and how much to trust each piece:
- **lr 4.8e-3: well supported from below, marginally from above.** Val@300 improved monotonically
  5.98 -> 5.07 over 3e-4..4.8e-3 (each doubling 0.14-0.30 nats), then got worse at 9.6e-3 (5.10). The
  rise to 9.6e-3 is only 0.030 nats, just above this doc's own ~0.02 single-seed trust threshold, so the
  upper bracket is weak. The optimum lies between ~2.4e-3 and 9.6e-3 (2x grid). Note val@100 favours 2.4e-3/4.8e-3
  equally; the larger lr wins later.
- **mu 0.9: moderately supported.** 5.034 vs 5.073 (0.95) vs 5.173 (0.98); monotone, and 0.9 is the
  edge of the tested range, so lower values were not tried.
- **wd 0 and gc 0: weak.** Gaps are 0.008 (wd 0 vs 0.1) and 0.015 (gc 0 vs 1) nats at one seed, 300 steps;
  wd 0.2 is only 0.005 worse than 0.1. Treat these as "no evidence that weight decay or clipping
  helps over this horizon", not as proof they hurt. Weight decay in particular is expected to matter more
  over longer runs; for a long run keep `-c 0.1` and `-gc 1.0` unless re-tested.
- Greedy one-factor search ignores interactions (e.g. best lr was chosen at mu 0.95, and lower mu may
  shift it).

## Observation: AdamW's 6e-4 is far below Muon's optimum

Under Moonlight RMS scaling Muon shares AdamW's lr scale, yet its optimum here (4.8e-3) is 8x AdamW's
conventional 6e-4, and Muon at 6e-4 (5.74) is already 0.38 nats better than AdamW at 6e-4 (6.12) but
still 0.67 nats short of the best lr at the same mu/wd/gc (4.8e-3, 5.07) and 0.73 short of the best run
(gc0, 5.01). So the first Muon-vs-AdamW comparison (600 steps), which used 6e-4 for both, very likely
under-tuned Muon; how much is not measured, since the lr optimum was found at 300 steps and tends to
shift down at longer horizons. The AdamW control here was **not re-tuned**; its own optimum at this
300-step horizon is likely above 6e-4 too (not tested here)
so the 1.11 gap is an upper bound on the optimizer gap and not a fair tuned-vs-tuned number.

## Caveats

- Single seed (42), no repeats; differences below ~0.02 nats should not be trusted.
- 300 steps with the cosine horizon tied to run length: short horizons favour larger lr and less
  regularisation. Absolute losses are **not comparable** to the 600-step runs in the previous doc
  (different schedule length), and the winner for a long run may differ (lr lower, wd/gc back on).
- Greedy search; AdamW control not tuned; lower mu and finer lr (between 2.4e-3 and 9.6e-3) untested.
- Runs shared the GB10 with sibling agents' short tests; ms/step is indicative only. Contention is
  visible: the lr6e-4, lr1.2e-3, lr2.4e-3 and lr4.8e-3 runs have step-time spikes to 5.4-6.0 s and p95
  of 4.05-4.20 s; the other eight peak at 3.9-4.5 s with p95 <= 4.05 s. Wall time is not an input to training, so rankings are unaffected.
- Two early launch glitches (guard matching sibling processes; a wrongly tagged first launch of the
  4.8e-3 rung) cost time but no results: the affected run was killed and rerun from scratch
  (`sweep_muon.sh` deletes the jsonl before each launch; the committed logs show no trace of the glitch).

## AI use statement

Written with AI assistance (Claude Code, Sonnet agent; numbers re-verified against the logs by an Opus review agent), directed by Evan Owen.
