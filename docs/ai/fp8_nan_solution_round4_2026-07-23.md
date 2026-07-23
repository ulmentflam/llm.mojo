> **Superseded by `fp8_multirank_nan_investigation.md`, kept as historical record.**

# fp8 multi-rank NaN — Round 4 CONVERGE: solution selection (2026-07-23)

> Round-4 convergence report. Supersedes the shipping guidance in
> `fp8_nan_hunt_round3_2026-07-22.md` (incl. its 2026-07-23 amendments) where
> they conflict. All round-3 facts cited here are post-amendment.

## 0. Executive verdict

**Winner: S2 — `LLMM_FP8_NO_STASH` (eliminate the forward-written fp8
transpose stashes; re-quantize transposed operands at consumption time).**
It is the only arm of the entire campaign with **zero treatment NaNs across
every conclusive run** (round-3 pooled 0/13 vs interleaved baseline 4/10,
Fisher p≈0.024; round-4 solo-regime decisive battery: SEE §2), it is the only
arm whose target is selected by the mechanism evidence (§3), and it is
**faster than baseline**, not slower. Worktree:
`/home/evan/workspace/llm.mojo-wt/nostash` (single file, `llmm/matmul.mojo`,
+88/−24, flag-gated, default path bit-for-bit unchanged).

Everything else is dead as a cure:

| Arm | Result | Status |
|---|---|---|
| S1 fwd+bwd per-site drains (`FP8_FWD_SYNC_ONLY`+`FP8_BWD_SYNC_ONLY`) | 1/10 T vs 1/5 B; T06 NaN under full drainage | NOT A CURE |
| S3 rank-dependent layout offset (`FP8_RANK_OFFSET`) | 5/10 T vs 1/5 B; both ranks struck | DEAD |
| S4 first-backward prewarm exclusion (`FP8_PREWARM`) | 2/10 T vs 2/5 B, canonical signature | DEAD |
| S5 runtime env knobs (BUSY_WAIT=0, AFFINITY, MAX_CONNECTIONS=1, MLRT_CUDA_DEBUG) | every knob ≥1 treatment NaN; K4 6/6 (possible aggravator) | DEAD (knob space exhausted) |
| S6 stash-lifecycle diagnostic | 0 discriminating events (ambient collapse), +16% cost | NO DATA; probe-lite only if ever needed |
| Round-3 arms (SYNC_ONLY, GEMM/whole-bwd mutex, EXEC mutex) | 3–5/10, indistinguishable from 8/10 baseline | DEAD |
| `CUDA_LAUNCH_BLOCKING=1` | 0/10 | works, env-only fallback |
| `MODULAR_DEBUG=device-sync-mode` | 0/4 (+1000 live steps) | works, excluded by goal |

## 1. The winning solution: S2 `LLMM_FP8_NO_STASH`

### What it does
The fp8 forward (`matmul_fwd_lowp`) previously wrote, per site/layer, two
long-lived e4m3 transposed stashes via `quantize_dual_devscale` into the
persistent `lowp_transpose_cache` buffers ("WT" weight-transposed, "IT"
input-transposed), consumed later by backward (`matmul_d_input_bwd_lowp`
dgrad reads WT; `matmul_d_weight_bwd_lowp` wgrad reads IT). That
forward-write → backward-consume dormancy is the corruption window (§3).
Under `-D LLMM_FP8_NO_STASH=1`:

- forward quantizes natural-layout only (`quantize_devscale`) — **no
  transposed stash is written at all**;
- each backward consumer re-quantizes its transposed operand **fresh,
  immediately before the consuming GEMM** (`quantize_transpose_devscale`,
  same bf16 source, same not-updated-since-forward scale), into the same
  persistent buffer.

The vulnerable window shrinks from an entire forward→backward span (hundreds
of intervening launches, cross-micro-step in wall-time terms) to the
launch-to-launch gap between the fresh quantize and its GEMM — effectively
zero. Note the fix does not require the corrupting writer to stop existing;
it removes the only buffer that dormantly holds consumable fp8 bytes.

### Evidence
- **Round 3 (2026-07-23 batteries, concurrent-suppressed ambient):**
  treatment **0/13** conclusive runs (battery1 0/10, ext 0/2, ext2 0/1; two
  further ext logs empty/killed = inconclusive, excluded) vs interleaved
  same-day baselines **4/10** (base_1a, ext_base_3, ext2_base_1, ext2_base_2 —
  all canonical rank-1-NaN / rank-0 19.617536 signature). Fisher exact
  (0/13 vs 4/10): **p≈0.024**. Ledger:
  `scratchpad/nostash_progress.txt`, logs `scratchpad/nostash_battery/`.
- **Round 4 solo-regime decisive battery (this report):** first true
  solo window of the campaign — all three pair locks held by one script,
  siblings idle, so the concurrency-suppression confound (S5 finding) is
  removed and the baseline runs hot. Interleaved strictly-alternating
  B/T ×12 each on pair0, mandated template (`-u 700`), `timeout INT 600`.
  **RESULTS: PENDING — filled in §2.**
- **Numerics:** forward bit-identical to baseline (train loss 11.097367,
  pre-step val 11.091539 match exactly); backward deterministically shifted
  (r0 pre-reduce 19.559965 vs 19.617536, ~0.3%; post-step val
  10.749122 vs 10.748892, Δ2.3e-4). Benign in magnitude; cause decoded in §3
  (consumption-time quantize reads the live bf16 source, which backward has
  by then legally mutated in place — NOT a bug in S2; it is the same value
  family the fault itself produces, see the 19.559965 identity).
- **Cost:** step-1 **~15% faster** than baseline (~3010–3031 ms vs
  ~3549–3566 ms, MFU 24.3–24.9% vs 20.7%) — removing the strided transposed
  write from the forward dual-quantize outweighs the two extra backward
  transpose-quantize kernels. Steady-state (-x 20): §2.

### Diff location
`/home/evan/workspace/llm.mojo-wt/nostash` (detached at main HEAD b50aa06),
`llmm/matmul.mojo` only, +88/−24:
1. `comptime FP8_NO_STASH = is_defined["LLMM_FP8_NO_STASH"]()` (+ design
   comment) below `FP8_BWD_SYNC_ONLY` (~line 60);
2. `matmul_fwd_lowp` (~line 1897): flag-gated natural-only
   `quantize_devscale` pair replacing the two `quantize_dual_devscale`;
3. `matmul_d_input_bwd_lowp` (~line 3458): `quantize_transpose_devscale`
   into WT immediately before the dgrad GEMM;
4. `matmul_d_weight_bwd_lowp` (~line 3554): same for IT before the wgrad
   GEMM.
Binary: `build/train_gpt2_fp8_nostash_w2_dbg`. Flag undefined ⇒ old path
bit-for-bit.

## 2. Round-4 solo-regime decisive battery (PENDING — results land here)

PENDING

## 3. Mechanism

### The sentence
**The corrupting write lands in the dormant forward-written fp8 transpose
stash (proj IT/WT) during the victim rank's backward, carries the exact
semantics of the victim's own transpose-quantize kernel re-executed late
against the live (backward-mutated) bf16 source, travels outside the victim
stream's ordering (survives full per-context drains), is armed by the
sibling rank's context/launch activity (ws1 clean) and by tight host launch
timing (step-1, fast-mode, solo-regime), and is suppressed only by global
per-launch serialization — so removing the dormant stash (S2) removes the
victim and the failure with it.**

### The decisive datum (found in round-4 cross-arm reconciliation)
S6 itself produced **zero** discriminating events (0 STASHCHECK lines in 12
treatment + 5 baseline runs — ambient collapse; the probe's own +16% timing
perturbation makes it partly self-defeating). The lifecycle selection came
instead from a byte-level identity across arms:

- S5 run `pair0_K4_t4` (stash-path binary): rank 0 pre-reduce norm
  **19.559965** — previously recorded as "first-ever corrupted-but-FINITE
  value";
- S2 no-stash treatment (every clean run): rank 0 pre-reduce norm
  **19.559965**, bit-identical.

A stash-path faulted run producing, to all printed digits, exactly the
deterministic value of the requantize-at-consumption arm cannot be
coincidence. It means: in K4_t4, rank 0's stash bytes at consumption were
those of a **consumption-time requantize of the live source** — i.e. the
stash was rewritten after forward, during backward, by something with the
victim's own kernel semantics. When that late write completes cleanly you
get the finite 19.559965 family ("silent corruption", now decoded as
mistimed-but-well-formed); when it interleaves with consumption or lands
mid-mutation you get NaN bytes. In S6's pre-registered key this is
**good-(a)/bad-(b): struck while dormant** — established without S6 firing.

### Reconciliation with every arm
- **ws1 100% clean** — no sibling context/launch thread, no collision path.
- **d12/d24 clean, d36 strikes, layer-27 preference, always proj wgrad** —
  scale-dependent launch pressure; wgrad is the IT-stash consumer, so a
  struck IT stash surfaces as exactly one corrupted proj wgrad slice; the
  layer preference tracks machinery/allocation order, not data content.
- **Step-1 strike** — cold launch path (first-launch compilation, module
  load) maximizes host-side timing collisions.
- **S1 fwd+bwd per-site drains fail (T06)** — a victim-context
  `ctx.synchronize()` cannot fence a write that does not travel through the
  victim's stream ordering. This is the arm that forces "late write" to
  mean "outside victim stream ordering" rather than simple run-ahead.
- **EXEC/GEMM/whole-bwd mutexes, S4 prewarm fail** — they serialize
  *legitimate* enqueued work; the replayed/late write is not a legitimate
  queue item and ignores cross-rank locks on real kernels.
- **S3 rank-offset fails (both ranks struck under rank-unique layouts)** —
  argues against a fixed-address cross-context scribble and *for* the
  own-kernel-semantics replay (its addresses travel with the kernel args,
  so layout offsets are irrelevant).
- **K3 `CUDA_DEVICE_MAX_CONNECTIONS=1` fails** — HW-channel serialization
  within a context doesn't remove the replay path.
- **K4 `MLRT_CUDA_DEBUG=1` 6/6 (aggravation trend), NaN runs faster,
  dual-rank victims in the low-concurrency window** — tighter/perturbed
  host launch timelines increase collision frequency; at treatment-class
  step times (~3.0 s) stash-path runs still NaN'd 6/6, which also disfavors
  "S2 wins by being 15% faster" (timing-masking).
- **Concurrency suppression (S5: pooled baselines ~1/9 with 3 active pairs
  vs 8/10 historical solo)** — sibling processes on *other* GPUs
  desynchronize the two ranks' launch threads, reducing collisions; this is
  host-timing behavior, not GPU-load behavior.
- **CLB=1 / device-sync-mode cure** — with at most one in-flight launch per
  host thread, the machinery state that emits a late/duplicated write never
  arises.
- **S2 cures** — no forward transpose-quantize launch exists to replay, and
  the fresh backward quantize is consumed within a launch-to-launch gap; any
  late write against it is overwritten before the next consumption.

Writer identity below this level (Modular AsyncRT shared launch machinery
vs guest-driver/GSP replay under QEMU/KVM VFIO) remains **unresolved and is
upstream's problem** (§6); it does not gate shipping S2.

### What remains to prove (mechanism hygiene, not shipping gates)
1. **Sham-timing arm** (only remaining alternative reading): launch the same
   two transpose-quantize kernels per backward site into a scratch buffer
   while still consuming the forward stash — identical timing footprint,
   exposure preserved. Sham NaNs + S2 stays clean ⇒ mechanism confirmed;
   sham also clean ⇒ S2's win is timing-side. K4's fast-mode 6/6 already
   disfavors the latter.
2. **Byte-diff probe** — per-site compare stash bytes vs a fresh requantize
   at consumption on a clean run to pin the in-place backward mutation site
   (suspect: dGELU on the saved input activation) that produces the
   19.559965 family, and to document S2's numeric shift as exactly that.

## 4. Promotion plan

### Phase A — confirmation (this session)
1. Solo-regime interleaved battery (§2). Promotion gate: treatment 0/12 with
   own-baseline ≥4/12 NaN; pooled with round-3 gives Fisher p<0.01.
2. Steady-state cost from the `-x 20` pair (§2).

### Phase B — land in main
1. In `llm.mojo-wt/nostash`: make no-stash the **default fp8 path** — invert
   the gate to `LLMM_FP8_STASH_LEGACY=1` (old path kept for A/B and upstream
   repro), fix the stale "consumed micro-steps later" comment wording
   (per-instance exposure is intra-micro-step), commit on branch
   `agent/fp8-no-stash`.
2. Merge gates (per repo policy, on CUDA): `pixi run format`, `lint`,
   `check`, `test` (30-min per-file timeout), `test_zero` 12/12.
3. Binary batteries on the merged tree (all UUID-pinned, faulted
   GPU-e9c166f6… excluded, `timeout --signal=INT`, never SIGKILL):
   - **ws2 ×10** solo-regime step-1 battery — gate: 0/10 NaN;
   - **ws7 ×6** (all 7 healthy GPUs) step-1 battery — gate: 0/6 NaN
     (S2 is untested >ws2; this is the coverage gate);
   - **3 × 12-step** confirmation runs — gate: no NaN, loss/val trajectory
     within the documented benign shift vs old-fp8 clean references
     (val Δ ≈ 2e-4 class at step 1, grad-norm shift ≈ 0.3%).
4. Merge to main; update `fp8_training_design.md` (stash section) and the
   round-3 doc header pointing here.

### Phase C — resume/finish the 774M run
1. Rebuild the 774M fp8 train binary from post-merge main (no-stash default).
2. Runs/datasets on `/data` (8 TB) per the ENOSPC lesson; resume from the
   latest checkpoint (`-y 1` against the run's `-o` dir); GPUs pinned by
   UUID; pause/resume via INT only.
3. Watch the first 3 steps' `DEBUG rank` gradnorm lines (dbg build for the
   first restart only, then switch to the plain binary for MFU).
4. **Defense-in-depth:** if any NaN appears in the first 100 resumed steps,
   set `CUDA_LAUNCH_BLOCKING=1` (proven 0/10) via env and continue the run
   while filing the datapoint — do not burn GPU-days debugging mid-run.

## 5. If the confirmation gate had failed — shipping stance
(Recorded for completeness; see §2 for whether it fired.) If S2's solo
battery shows any treatment NaN: **`CUDA_LAUNCH_BLOCKING=1` is the shipping
answer** for fp8 WORLD_SIZE>1 (env-only, proven 0/10, cost unmeasured at
scale — measure once on a 20-step run before committing the 774M to it),
with the S2 diff still merged as defense-in-depth (it strictly shrinks the
victim window and is faster). device-sync-mode remains the fallback of last
resort.

## 6. Upstream filing text updates

### Modular (AsyncRT / MAX runtime)
Title: *"fp8 multi-rank NaN on multi-GPU QEMU/KVM VFIO guest: single
corrupted slice in a long-lived device buffer; cured only by global
per-launch serialization; per-site stream drains, cross-rank
submission/execution mutexes, rank-dependent layout offsets, and every
dispatch-adjacent env knob fail; application-level removal of the dormant
buffer eliminates the failure."*
Add to body:
1. the full arm table (round-3 amended + S1–S5 rows above);
2. the **19.559965 identity datum**: a faulted stash-path run reproduced,
   bit-exactly, the deterministic value of the requantize-at-consumption
   build — the dormant buffer was rewritten post-forward with the victim's
   own kernel semantics; corruption is a *mistimed correct write*, not
   random bytes;
3. concurrency suppression: NaN rate collapses when unrelated processes run
   on other GPUs of the same host (host-launch-timing sensitivity);
4. `MLRT_CUDA_DEBUG=1` trend toward aggravation (6/6);
5. env-knob enumeration result: no queue-depth/worker knobs exist in shipped
   libs (disassembly-verified list included).

### NVIDIA (guest driver / GSP, co-primary)
Multi-context fp8 workload in a QEMU/KVM VFIO guest, GSP present. Describe:
late/duplicated effective write carrying a completed kernel's semantics,
not ordered by the issuing context's own stream synchronization
(`ctx.synchronize()` at the write site does not prevent it); armed by a
sibling context's existence (single-context 100% clean); suppressed by
`CUDA_LAUNCH_BLOCKING=1` and by host-timing perturbation; observed as both
NaN bytes and well-formed stale-semantics values. Ask specifically about
launch replay/dedup paths in GSP-RM under VFIO and known multi-context
issues on this driver branch. Attach: arm table, the identity datum, and
the S2 repro pair (flag on/off) as a minimal discriminator.

## 7. Artifacts index
- Winner worktree/diff: `/home/evan/workspace/llm.mojo-wt/nostash`
  (`llmm/matmul.mojo` +88/−24); binary `build/train_gpt2_fp8_nostash_w2_dbg`.
- Round-3 S2 ledger/logs: session scratchpad `nostash_progress.txt`,
  `nostash_battery/`.
- Round-4 solo battery: scratchpad `solo_battery.sh`, `solo_progress.txt`,
  `solo_battery/*.log` (this report §2).
- Dead-arm worktrees (keep until upstream filings land, then prune):
  `s1-fwdsync`, `rankoffset`, `prewarm`, `stashcheck`.
- Prior reports: `fp8_nan_hunt_round3_2026-07-22.md` (+2026-07-23
  amendments), `fp8_nan_hunt_round2_2026-07-22.md`,
  `fp8_multirank_race_2026-07-22.md`.
