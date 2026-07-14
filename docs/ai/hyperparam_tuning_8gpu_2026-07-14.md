# Hyperparameter tuning for the 8-GPU GPT-2 124M / FineWeb-10B run

Tuning campaign to produce a flag recipe for an upcoming **8×GPU (WORLD_SIZE=8,
ZeRO) FineWeb-10B** GPT-2 124M (`d12`) pretraining run on an 8× NVIDIA RTX PRO
6000 Blackwell Max-Q box (96 GB/GPU), improving on the prior **single-GB10**
recipe documented in
[`gpt2_124m_fineweb_training_run.md`](gpt2_124m_fineweb_training_run.md).

Prior single-GPU recipe (bf16, GB10):

```
-e d12 -n 1000 -y 1 -b 32 -t 1024 -d 524288 -l 0.0006 -q 0.0 -u 700 -c 0.1 -v 250 -s 20000 -x -1
```

**Scope of this campaign:** GPUs 4–7 only (every run pinned with
`CUDA_VISIBLE_DEVICES`). GPUs 0–3 belonged to a concurrent ZeRO-verification
agent. All experiments are SHORT (ranking, not convergence).

---

## Headline findings

1. **Throughput (measured, bf16, the production precision):** per-GPU
   throughput on this Blackwell part is **saturated and essentially flat** from
   micro-batch `b=8` through `b=64` (~162–173k tok/s, a ~6% spread that is within
   run-to-run / cross-GPU noise). `b=128` **OOMs** (single activation allocation
   requests 98 GB > 96 GB). So **`b=64` is the memory ceiling and ties for the
   best throughput.**
2. **For the 8-GPU run at fixed effective batch 524288, `b=64` is the natural
   choice:** `8 GPUs × b=64 × T=1024 = 524288`, i.e. **grad-accum = 1** (one
   micro-batch per GPU per optimizer step, no local accumulation). It is
   simultaneously the throughput-optimal, the simplest (no accumulation), and the
   largest micro-batch that fits.
3. **bf16 training NaNs immediately on this hardware/commit** (see the blocker
   section). The from-scratch AND checkpoint-loaded bf16 backward produces NaN
   gradients on step 1, which poison AdamW moments and diverge the run even at
   `LR=0`. The **identical fp32 config trains cleanly**, so this is a
   bf16-backward *kernel* bug on this Blackwell part, not a hyperparameter issue.
   Quality/LR tuning was therefore done in **fp32 as a documented proxy**.
4. **Learning rate (measured on real FineWeb-10B val, fp32 proxy, production
   effective batch 524288): `-l 0.0018` with adequate warmup beats the prior
   `-l 0.0006` by 0.083 val loss at a fixed 150-step budget** — ~14× the
   measured run-to-run noise floor (0.0013, from a repeat of the winner).
   0.0022 is stable but worse (the LR curve turns); short warmup at high LR is
   catastrophic (+0.28 val). Full ranking in Part 3.

---

## Environment & harness

- Box: 8× NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96 GB each
  (`nvidia-smi`: 97887 MiB). The MFU table in `llmm/mfu.mojo` does not recognize
  this device (`n/a ... MFU` in every log line) — noted, not load-bearing here.
- Worktree: `/home/evan/workspace/llm.mojo-wt/tuning`, branch `agent/tuning`,
  base `HEAD e8d9222`.
- Build: `make build-bf16 WORLD_SIZE=1` → `build/train_gpt2_bf16`; fp32 proxy
  `make build WORLD_SIZE=1` → `build/train_gpt2`. Run via
  `scripts/run_train_gpt2_bf16.sh` / `scripts/run_train_gpt2.sh`.
- `-t` (sequence length) is **fixed at 1024**: `d12` is a GPT-2 descriptor with
  `max_seq_len=1024` (positional-embedding table sized to 1024), so `T` cannot be
  increased without changing the model. The throughput axis is therefore purely
  micro-batch `b`.
- Data: FineWeb-10B shards were still downloading during the campaign. Throughput
  tuning (timing only) used `data/.tinyshakespeare/tiny_shakespeare_train.bin`
  for both `-i` and `-j` (the stock val shard has exactly 32768 tokens, one short
  of the loader's `B*T+1` floor even with `-v 0`, so it fails to construct; the
  larger train file, 305260 tokens, is used for both). Quality tuning used
  FineWeb shards (val = shard 0).
- Log parse: step line is
  `step N/TOTAL | loss L (Zz)| norm G (Zz)| lr LR | MS ms | MFU | TOK tok/s`;
  val line is `val loss L`. The `(Zz)` fields are z-scores (NaN on the first
  sample — harmless). Timing = the `MS ms` field; steady-state = median of the
  last 10 step lines.

---

## Part 1 — Throughput (bf16, measured)

**Method.** `d12`, `T=1024`, bf16, 25 steps per config, median ms/step over the
last 10 steps, `tok/s = total_batch(-d) / (median_ms/1000)`. Loss is NaN (see
blocker) but timing is unaffected. Two sub-sweeps: (a) an **8-GPU per-GPU
emulation** holding `-d 65536` (= one 8-GPU step's per-GPU token count, so
grad-accum = 64/b), and (b) a **saturation curve** at `-d = b*1024` (grad-accum
= 1, pure micro-batch throughput). Logs: `sweeps/throughput_*.log`; parsed table:
`sweeps/throughput_results.csv`.

### Sub-sweep (a): 8-GPU per-GPU emulation (`-d 65536`)

| b | grad-accum | median ms/step | per-GPU tok/s |
|--:|--:|--:|--:|
| 16 | 4 | 380.81 | 172099 |
| 32 | 2 | 389.37 | 168313 |
| 64 | 1 | 387.90 | 168953 |

### Sub-sweep (b): saturation curve (grad-accum = 1, `-d = b*1024`)

| b | -d | median ms/step | per-GPU tok/s |
|--:|--:|--:|--:|
| 8   | 8192   | 50.53  | 162138 |
| 16  | 16384  | 98.84  | 165763 |
| 32  | 32768  | 198.56 | 165028 |
| 64  | 65536  | 378.80 | 173010 |
| 128 | 131072 | OOM (req 98.16 GB) | — |
| 256 | 262144 | OOM (req 196.31 GB) | — |

**Interpretation.** Per-GPU throughput is flat within ~6% across the whole
feasible range; the b=64 config was measured at 168953 (GPU 4) and 173010
(GPU 6) — a 2.4% cross-GPU spread that is *larger* than the b16→b64 difference,
so the ranking within {16,32,64} is noise. The GPU is saturated by `b=8`
already. Grad-accumulation overhead is negligible (the `-d 65536` sub-sweep with
accum 1/2/4 is flat). The only hard constraint is **memory: `b=64` fits, `b=128`
does not.**

**Throughput decision:** **`b=64`, grad-accum = 1** for the 8-GPU run. It ties
for best measured throughput, is the largest micro-batch that fits, and removes
gradient accumulation entirely (`8 × 64 × 1024 = 524288`). Prior GB10 recipe used
`b=32` because that was a much smaller GPU; here `b=64` is both feasible and
optimal.

Per-GPU steady-state ~170k tok/s bf16 is ~5× the GB10 single-GPU rate (~33.6k
tok/s in the prior run). Aggregate across 8 GPUs (before ZeRO comm overhead,
which the separate ZeRO-verification agent measures) is ~1.36M tok/s; a full 10B
epoch (~19073 steps × 524288) would be ~2 h of compute if comm-free — the ZeRO
agent's numbers are the authority on realized aggregate.

---

## Part 2 — bf16 backward NaN (blocker for direct bf16 quality tuning)

**Symptom.** Every bf16 run — from-scratch `d12` *and* loading the pretrained
`gpt2_124M_bf16.bin` — prints a finite step-1 loss (11.0 from scratch, 4.28 from
the checkpoint) but **`norm nan`** on step 1, then `loss nan` from step 2 onward.
Reproduced across `b ∈ {4,32}`, grad-accum ∈ {1,2}, and `MODULAR_DEBUG=
device-sync-mode` (so it is **not** the async-launch race condition noted in
`gpt2_124m_fineweb_training_run.md`).

**Diagnosis chain.**
1. Step-1 loss is correct → the *forward* pass is fine.
2. Step-1 gradient **norm is NaN** → the *backward* pass emits NaN/Inf gradients.
   Gradient clipping then multiplies every gradient by `clip/norm = 1/nan = nan`.
3. **The run diverges even at `LR=0`:** with a valid checkpoint and `-l 0.0
   -c 0.0`, step-1 loss is 4.28 but step 2 is NaN. AdamW's moment update
   `m = β1·m + (1-β1)·g` turns NaN with a NaN `g`, and the weight update
   `LR · m̂ = 0 · nan = nan` (IEEE) corrupts the weights despite the zero LR. So
   no learning-rate or warmup choice can avoid it — it is upstream of the
   optimizer.
4. **The identical fp32 config trains cleanly** (`make build`; loss
   11.0→8.05 over 8 steps, finite norms 21.6→2.35). This localizes the fault to
   a **bf16 backward kernel on this Blackwell (sm_120-class) part**, not to the
   optimizer, data, or hyperparameters.

This is a kernel-correctness issue outside the hyperparameter mandate (a separate
agent was concurrently building `dump_grads_gpt2_bf16` to debug bf16 gradients).
**The production 8-GPU run is itself blocked on this bf16 fix**; the recipe below
is precision-agnostic on the learning-rate axis and applies once bf16 is
repaired. Quality tuning proceeded in fp32 as a proxy.

---

## Part 3 — Quality / LR tuning (fp32 proxy)

**Why fp32.** Direct bf16 val-loss ranking is impossible (Part 2: bf16 training
NaNs on step 1). The LR/warmup sweep therefore ran in **fp32** (which trains
cleanly). The LR at a *fixed effective batch* (all quality runs use the
production `-d 524288`) is dominated by the optimization landscape, not the
last bit of precision, so the relative LR ordering is expected to transfer to
bf16 once the kernel bug is fixed — but a cheap bf16 re-confirmation of the
winner is recommended before launch (exact command in the recipe section).

**Method (all rounds).** fp32 (`build/train_gpt2` via
`scripts/run_train_gpt2.sh`), `d12`, `b=16`, `T=1024`, `-d 524288`
(= production effective batch, grad-accum 32, so LR transfers directly), cosine
schedule to zero (`-k cosine -q 0.0`), weight decay `-c 0.1`, `-x 150` steps
(the cosine horizon = the step budget: `train_num_batches = max_steps` when
`-x > 0`), held-out val every 25 steps (`-v 25 -m 20`), 4 configs one-per-GPU
on GPUs 4–7, ~7.0–7.2 s/step (~73–76k tok/s fp32). Logs:
`sweeps/quality_*.log`; parsed table `sweeps/quality_fp32_results.csv`.

### Round 1 — tinyshakespeare (stability probe, while FineWeb downloaded)

Held-out val = stock `tiny_shakespeare_val.bin` (32768 tokens, works at
`b=16`). With only 305k train tokens under a 524288-token batch this is
full-batch descent on a memorizable corpus — an **overfitting regime**, so val
*values* rank memorization speed, not pretraining quality. Used as a
**stability-ceiling probe**.

| lr | warmup | min val loss | final train loss | diverged? |
|--:|--:|--:|--:|:--|
| 0.0006 | 15 | 5.5077 (step 100) | 3.9940 | no |
| 0.0010 | 25 | 5.6968 (step 75)  | 4.1584 | no |
| 0.0014 | 30 | 5.7360 (step 100) | 4.2162 | no |
| 0.0018 | 40 | 5.6948 (step 75)  | 4.1240 | no |

**Takeaway:** every LR up to 0.0018 is numerically stable at effective batch
524288 (no NaN/inf anywhere; clean cosine trajectories). Val ordering here is
overfitting-dominated (all runs' val bottoms out then rises) and was not used
for the LR pick.

### Round 2 — FineWeb-10B (the decisive ranking)

FineWeb shards landed mid-campaign (69 train shards + val shard 0 at launch
time). Same method, `-i data/.fineweb10B/fineweb_train_*.bin`,
`-j data/.fineweb10B/fineweb_val_000000.bin` — genuine held-out FineWeb val,
the same metric the production run optimizes. Val loss is still monotonically
falling at step 150 for every config (proper pretraining regime, no overfit).

| lr | warmup | val@50 | val@100 | val@150 (final) | final train loss | diverged? |
|--:|--:|--:|--:|--:|--:|:--|
| 0.0006 | 15 | 7.0573 | 6.6563 | 6.5847 | 6.4745 | no |
| 0.0010 | 25 | 7.0263 | 6.6285 | 6.5587 | 6.4484 | no |
| 0.0014 | 30 | 7.1630 | 6.7544 | 6.6553 | 6.5440 | no |
| **0.0018** | **40** | **7.0032** | **6.6081** | **6.5019** | **6.3911** | no |

**Ranking (final FineWeb val): 0.0018 < 0.0010 < 0.0006 < 0.0014.** The
highest LR tested wins cleanly (-0.083 val vs the prior recipe's 0.0006), the
2nd-highest is 2nd — consistent with the speedrun-literature expectation that
124M tolerates ~2–3× the GPT-3-paper LR at this batch size. The 0.0014/u30 run
is an out-of-order outlier (worse than both neighbors).

### Round 3 — variance, warmup confound, and LR ceiling (FineWeb)

Four follow-ups, same method:

| config | purpose | val@150 | final train loss | diverged? |
|:--|:--|--:|--:|:--|
| 0.0018 / u40 (repeat) | run-to-run variance | 6.5032 | 6.3914 | no |
| 0.0014 / u40 | round-2 outlier confound | 6.5363 | 6.4312 | no |
| 0.0022 / u50 | LR ceiling probe | 6.5246 | 6.4163 | no |
| 0.0018 / u15 | warmup sensitivity | 6.7813 | 6.6843 | no |

**Findings.**
1. **Reproducibility:** the winner repeated at 6.5032 vs 6.5019 —
   **run-to-run delta 0.0013** (fixed init seed; residual GPU nondeterminism
   only). Every cross-config gap in rounds 2–3 (≥0.02) is far above this noise
   floor, so the ranking is real.
2. **The round-2 outlier was a warmup confound:** 0.0014 with u40 scores 6.5363
   (vs 6.6553 with u30), slotting it *between* 0.0010 and 0.0018 — restoring a
   clean, monotone-then-turning LR curve.
3. **LR ceiling:** 0.0022 is still perfectly stable but *worse* than 0.0018
   (6.5246 vs 6.5032) — the optimum at this batch/model is at or near
   **0.0018**; pushing further buys nothing.
4. **Warmup is load-bearing at high LR:** 0.0018 with u15 degrades massively
   (6.7813, worse than every adequately-warmed config including 0.0006). High
   peak LR must be paired with sufficient warmup.

### Combined FineWeb ranking (val loss @ step 150, ~7.1 s/step each)

| rank | lr | warmup | val@150 |
|--:|--:|--:|--:|
| 1 | **0.0018** | 40 | **6.5019 / 6.5032 (repeat)** |
| 2 | 0.0022 | 50 | 6.5246 |
| 3 | 0.0014 | 40 | 6.5363 |
| 4 | 0.0010 | 25 | 6.5587 |
| 5 | 0.0006 (prior recipe) | 15 | 6.5847 |
| — | 0.0014 | 30 (under-warmed) | 6.6553 |
| — | 0.0018 | 15 (under-warmed) | 6.7813 |

**Caveat on horizon:** these are 150-step cosine-to-zero runs; at the full
19552-step horizon the LR advantage typically compresses but does not invert at
this scale (speedrun literature trains 124M at 0.0012–0.0018+ to full
convergence). The 3× LR win over the prior recipe (Δval −0.083 at fixed step
budget, ~14× the noise floor) is the strongest quality lever this campaign
found.

---

## FINAL RECOMMENDED 8-GPU RECIPE

For the WORLD_SIZE=8 / ZeRO FineWeb-10B run (launcher flags, per the multi-GPU
harness the ZeRO agent is verifying; `-z` per its verdict):

```
-i "data/.fineweb10B/fineweb_train_*.bin" \
-j "data/.fineweb10B/fineweb_val_*.bin" \
-e d12 -o log124M_8gpu -n 1000 -y 1 \
-b 64 -t 1024 -d 524288 \
-k cosine -l 0.0018 -u 1000 -q 0.0 -c 0.1 \
-v 250 -s 0 -x -1
```

Change-by-change vs the prior single-GPU recipe, with evidence class:

| flag | prior | new | why | evidence |
|:--|:--|:--|:--|:--|
| `-b` | 32 | **64** | Ties for best measured per-GPU throughput (~169–173k tok/s bf16), max micro-batch that fits in 96 GB (128 OOMs), and makes `8×64×1024 = 524288` → **grad-accum 1** (no accumulation loop at all) | **measured** (Part 1) |
| `-d` | 524288 | 524288 | Keep effective batch: LR results transfer, epoch = 19,552 steps unchanged | design choice |
| `-l` | 0.0006 | **0.0018** | Best FineWeb val at fixed budget in a 5-point LR curve; beats 0.0006 by 0.083 val (~14× the measured 0.0013 noise floor); 0.0022 is worse — this is the turn of the curve | **measured** (fp32 proxy, 150-step horizon; Part 3) |
| `-u` | 700 | **1000** | Round 3 shows high LR is warmup-sensitive (u15 catastrophic, u40 good at the 150-step horizon). At the full horizon, 700 steps (367M tokens) was safe at LR 0.0006; ~1000 steps (~524M tokens, 5% of the run) hedges the 3× higher peak LR | **extrapolated** (direction measured, magnitude not) |
| `-q` | 0.0 | 0.0 | Cosine-to-zero: prior run converged flat well before the end; nothing in the sweep argues for a floor | prior run + unchanged |
| `-c` | 0.1 | 0.1 | Not swept (low expected sensitivity at this scale/horizon); prior-run-proven | unchanged |
| `-t` | 1024 | 1024 | Fixed by `d12` (`max_seq_len=1024`) — not a free axis | structural |
| `-s` | 20000 | **0** | bf16 B=1 generation path caused the prior run's incident #3; per `gpt2_124m_fineweb_training_run.md` the fix status must be re-verified before re-enabling — default safe | prior-run incident |
| `-n`/`-y`/`-v` | 1000/1/250 | 1000/1/250 | Checkpoint/resume cadence proven by the prior run's 4 incidents | unchanged |

**Expected throughput (extrapolated, not measured end-to-end):** ~169k tok/s ×
8 = **~1.35M tok/s aggregate before ZeRO communication overhead**; a 19,552-step
epoch ≈ **2.2 h of pure compute** (vs 88.8 h on the GB10). Realized aggregate
depends on the ZeRO stage/interconnect — the ZeRO verification agent's
measurements are authoritative there; if ZeRO overhead is large, `-b 64` /
grad-accum 1 is the *most* communication-exposed shape (one all-reduce window
per step), and re-testing `-b 32 -d 524288` (accum 2, more overlap opportunity)
is the first knob to try.

**Launch blockers / preconditions:**
1. **The bf16 backward NaN (Part 2) must be fixed and re-verified** — as of
   this campaign the production-precision binary cannot train at all on this
   box. Verification: 20 steps of the recipe on 1 GPU must show finite
   `norm` from step 1.
2. After the bf16 fix, **re-confirm the LR winner in bf16 cheaply** (two
   150-step runs, `-l 0.0006` vs `-l 0.0018`, same flags as the fp32 round-2
   commands in `sweeps/quality_fw_*.log`): expect the same ordering; if bf16
   flips it, trust the bf16 result and fall back toward 0.0010–0.0014.
3. `tests/test_zero_equivalence.mojo` fails at base commit `e8d9222` (all 3
   stages, "Bad magic model file" — the test's own fixture-writer throws before
   writing; reproduced in isolation on an untouched tree, CPU-only, so unrelated
   to this campaign's changes). ZeRO correctness sign-off belongs to the ZeRO
   verification agent; do not launch until its suite is green.

**What was NOT measured here (honest list):** real 8-GPU aggregate throughput
(single-GPU × 8 extrapolation only); bf16 quality at any LR (kernel bug);
full-horizon (19,552-step) LR behavior; weight-decay and schedule-shape (`-k
wsd`, `-q > 0`) sweeps; a second *init* seed (the harness has no seed flag —
the round-3 repeat measures run-to-run nondeterminism, not init sensitivity).

---

## AI use statement

Written with AI assistance (Claude Opus agent via Claude Code), directed by Evan.
