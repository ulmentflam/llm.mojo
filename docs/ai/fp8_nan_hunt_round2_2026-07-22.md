# fp8 multi-rank NaN hunt — Round 2 synthesis (2026-07-22)

Status: mechanism narrowed to one family but **not yet proven** — the experiment that
would have proven or killed it (whole-backward host serialization) was invalidated by a
real bug found in the probe's own lock. Verdict, fixes, validation protocol, and the
single next experiment below.

---

## 1. Executive verdict

**Most probable mechanism (single, decisive pick):**

A **first-concurrent-execution race in process-shared host machinery on the fp8 backward
submission path** — the MAX AsyncRT `loadFunction` miss path performing lazy CUDA module
load/link (`cuModuleLoadDataEx`/first-launch finalization, unserialized by any global
lock, on a box with a known-flaky GSP driver), and/or cuBLASLt first backward-shape init
racing other threads' concurrent kernel enqueues. The race is armed by a synchronization
asymmetry that Round 2 surfaced:

> **The val-loss allreduce barrier immediately before step 1 (`train_gpt2.mojo:4965`)
> releases all rank threads into never-before-executed backward code in lockstep.**
> Forward/val first-uses happen under natural multi-second startup skew (checkpoint
> load, dataloader, per-rank init) and are therefore effectively staggered; backward
> first-uses are the only first-use events in the program that execute post-barrier,
> host-code-identical, on all ranks simultaneously.

This reconciles the otherwise-paradoxical `LLMM_FP8_FWD_ONLY`-clean gate with a
first-use mechanism: the val pass does run fp8 forward first-uses "concurrently across 7
threads", but not in lockstep — the backward's first uses are.

Loser threads of the race get a misfired/torn launch of a backward-only kernel. Top
blast target: `_quantize_dual_kernel_devscale[bf16, float8_e5m2, RNE]`
(`llmm/lowp.mojo:722-831`, sole call site `llmm/matmul.mojo:3604-3612`) — its output
buffers (`FP8_BWD_DN_/DT_`, persistent, `zero=False`, `llmm/memory.mojo:73`) feed both
backward GEMMs as the E5M2 operand, and garbage bytes decode to NaN/Inf at 8/256 density
(4x E4M3's), so any misfire NaNs that rank's entire local gradient. Winner ranks compute
bit-identically every run. Victim set = whichever ranks are in the vulnerable phase when
the event fires (0..N-1 victims; matches every observed set including ws2's
strictly-one-victim and ws7's {0,3,4,5}).

**Honest caveats on the pick:**
- The d12-ws7-clean gate weakly contradicts any depth-independent first-use mechanism
  (d12 runs the identical backward-only first-compile/first-load set concurrently).
  Reconciliation: the d12 gate is small-n, and d12's smaller shapes shrink both the
  submission window and per-launch cost; a probabilistic race can pass a few d12 runs
  while firing ~100% at d36. If the corrected serialization probe (§4) still NaNs, this
  family is dead and attention shifts to the shape-dependent residuals: cuBLASLt
  backward-shape init racing *non-Lt* concurrent submissions (never excluded — the
  failed GEMM mutex only serialized Lt-vs-Lt), and read-before-write of a backward-only
  buffer region (padding/tail bytes decoded as E5M2).
- "First-use" vs "every-step race that fires on the first (heaviest) backward" is not
  yet discriminated: step-1 allreduce propagates NaN into all weights, so later-step
  corruption is unobservable without the NANSCAN control (§6).

---

## 2. Round-2 results digest (with adversarial-review amendments folded in)

### 2.1 ws2 minimal repro (CONFIRMED, high)
`build/train_gpt2_fp8_w2_dbg` (-pn 2, -d 131072, -x 1), 5 runs: NaN sets
{}, {0}, {1}, {0}, {1} — **4/5 failing, exactly one victim per failing run**. Clean
norms bit-identical everywhere (rank0 19.617536, rank1 19.492779); loss 11.097367 all
runs; ~53 s/run. **ws2 is the established cheapest iteration config.**
Review amendments: no rank-count-scaling claim is licensed (ws7 rate was never a counted
series before step1-stats); the one-victim pattern has zero discriminating power (ws7's
4-victim set {0,3,4,5} kills a strict pairwise-clobber model — best model is one
corruption event whose victims are all ranks concurrently in a vulnerable phase);
**fix validation needs ≥10 consecutive clean ws2 runs** (5/5 clean occurs by chance ~9%
at the CI-upper clean rate), 5 runs suffice for bug *detection*. Anomaly on file:
failing runs end at val loss ~10.8395 vs 11.0859 clean — nanz-zeroed grads still change
the update.
Logs: `scratchpad/w2_run{1..5}.log`.

### 2.2 Step-1 statistics at ws7 (CONFIRMED, high)
6 runs of `build/train_gpt2_fp8_w7_dbg`: NaN sets {0,5}, {4,5,6}, {3}, {4,5}, {0,5},
{2,4,6}. Pooled with round-2's earlier 3 runs: **9/9 runs dirty**, mean 2.3/7 victims,
never 0, never 7. Pooled per-rank frequency: r0 3/9, r1 1/9, r2 1/9, r3 3/9, r4 5/9,
r5 5/9, r6 3/9. Clean-rank norms bit-identical across all runs (r0 5.487552,
r1 5.6950116, r2 5.672953, r3 5.9431868, r4 5.7389665, r5 5.9086213, r6 5.646068);
loss 11.096148 identical.
Review amendments: "clean step 1 rare-to-nonexistent" overstated — under the per-rank
model P(fully clean run) ≈ 5%, so 0/9 clean is the *expected* outcome; the apparent
rank-dependence (r5 5/9 vs r1 1/9) is exactly what uniform p≈1/3 plus multiple
comparisons predicts — **do not chase per-rank asymmetry**. Independent-per-rank vs
one-global-event models are both consistent; the latter fits the barrier-lockstep
mechanism. Post-step val loss varies across runs (10.839161–10.839872).
Logs: `scratchpad/step1stats/run{1..6}.log`.

### 2.3 CUDA_LAUNCH_BLOCKING (retry completed post-findings; n=1, suggestive only)
Original attempt died with its reaped background task (0-byte `clb_run1.log`); the
review-relaunched retry **completed clean**: all 7 ranks print their bit-identical
known-clean norms, global norm 39.7453 finite (`scratchpad/clb_run1_retry.log`,
20:33 UTC). Against the 9/9-dirty baseline this is p≈0.05 evidence that per-thread
launch-completion blocking suppresses the bug. Per review, **this does not localize**:
CLB is not cross-thread serialization (7 threads still enter the driver concurrently);
it is another "any slowdown/serialization masks it" datapoint — consistent with, but
not probative of, a submission-window race. Both branches of the original
interpretation rule were reviewed unsound; do not spend further runs on CLB.

### 2.4 MAX runtime first-use audit (disassembly; amended, medium-high)
Verified facts (all disassembly-exact after review):
- **No Mojo-side kernel cache**: 67 unmemoized `AsyncRT_DeviceContext_loadFunction`
  call sites; every `ctx.compile_function` call re-enters the runtime.
- **Runtime kernel cache is per-physical-device and mutex-protected**
  (`CUDADeviceContext::loadFunction` 0x680c0; lookup/insert under
  `pthread_mutex_lock(Device+0x78)`). The "process-global cache keyed without device"
  hypothesis is **refuted**.
- **First-call module load is lazy and NOT globally serialized**: on miss,
  `cuModuleLoadDataEx`/`cuModuleGetFunction` run outside the per-device cache mutex;
  at step-1 backward, 7 threads concurrently first-load the backward-only kernels.
  CUDA 13.3 defaults `CUDA_MODULE_LOADING=LAZY`; per review, the real load/link runs
  concurrently in the same step-1 window under **both** LAZY and EAGER (AsyncRT calls
  getFunction immediately after load), so **EAGER is neither a fix nor a clean
  discriminator** — dropped.
- Binary embeds 47 AOT sm_120a cubins, zero PTX; sole runtime-JIT kernel is the bf16
  `_attention_bwd_p_and_ds_gpu` (exonerated by FWD_ONLY... note it is backward — it is
  per-rank cached and low-ranked).
- Backward-path compile-site corrections: `AmaxState.update_scale` → `_update_scale_gpu`
  is `amax.mojo:538` (not ~649); `CUBLASLT_WS` at `matmul.mojo:212`.
Review demotion: candidate ranking is not forced by evidence (d12 gate, val-pass clean
concurrent first-loads); the whole-backward mutex serializes candidates 1–4 *together*;
the warmup is a **fix candidate, not a diagnosis**.

### 2.5 Forward/backward differential (amended, medium-high)
Complete enumeration of what `matmul_bwd_lowp` (matmul.mojo:3494-3647) does that the
proven-clean-under-concurrency forward does not. Survivors after both reviews:
- **[T1] E5M2 quantize first compile+launch** (`lowp.mojo:819`, site matmul.mojo:3604)
  — inside the curing-candidate BWD-mutex scope, outside the refuted GEMM-mutex scope.
- **[T1] Standalone DGELU kernel** `matmul_gelu_backward_scaling_gpu`
  (matmul.mojo:1402-1427, sole live site 3413) — genuinely fp8-backward-only on CUDA
  (`USE_GELU_FUSION=True` makes the bf16 branch comptime-dead); corrupts bf16 d_fch
  in place → the "corrupted bf16" entry route.
- **[T2] `_update_scale_gpu[16]` first compile** (amax.mojo:536-550, site 3594) —
  garbage `scale_inv` would NaN the GEMM's bf16 D via the B_SCALE descale multiplier,
  bypassing the encoder-saturation exoneration.
- **[demoted] Transpose-cache WT/IT lookup-during-insert tear** — structurally the
  only forward-clean-by-construction candidate (forward misses self-heal;
  backward's write-in-call-A/read-in-call-B cannot), and the only depth-scaling one,
  BUT the registry was disassembly-verified safe for reader-during-insert (§2.6):
  fully-initialized nodes published via `lock cmpxchg`, null→non-null slots only, key
  verified by length+bcmp, no resize/deletion. Treat as dead unless everything else
  falls.
- **Exonerated**: e4m3×e5m2 Lt *host* first-use in its Lt-vs-Lt form (whole
  `_matmul_cublaslt_fp8` body, incl. descriptor/heuristics, is inside the refuted
  GEMM mutex); beta=1 accumulate; `compute_amax` instantiations (val-pass-compiled);
  `matmul_bias_bwd` (runs pre-mutex and in clean bf16); host branches.
- **Missing-from-list candidate added by review**: cuBLASLt process-shared host state
  racing *another thread's non-Lt enqueue* (mixed pairwise family) — not excluded by
  any experiment to date; and read-before-write of backward-only buffer bytes.
- Review kill-shot on compile-race ranking: all three T1/T2 kernels are instantiated
  on dtype/width only (shapes runtime), so **d12 runs the identical first-compile set
  concurrently and stays clean** — a depth-independent compile race firing 100% at d36
  and 0% at d12 is a poor fit. The lockstep-barrier asymmetry (§1) is what keeps the
  first-use family alive despite this.

### 2.6 Host allocation/lock audit (CONFIRMED, high) — **one real defect**
**`_fp8_gemm_lock()` (llmm/matmul.mojo:39-57) is broken on first-call races**: on miss
it allocates a private cell, `InsertGlobal`s it (null destructor — loser's cell
survives), and returns **its own pointer, not the registry winner**. Under
`-D LLMM_FP8_BWD_MUTEX=1` the lock's *first-ever* touch is step-1 backward (forward
never takes it in that build), i.e. exactly the suspected corruption window, entered in
lockstep post-barrier: racing ranks acquire *different* private cells → their first
`matmul_bwd_lowp` runs **unserialized**; worse, the release re-lookup
(matmul.mojo:3646-3647) makes each loser's `store(0)` **spuriously unlock the winner's
mutex** while held, and the extra permit persists under contention (blind-store
release), un-serializing essentially all of step-1 backward from a single loser.

**Consequence: `run1_bwdmutex.log`'s step-1 NaN is NOT a valid refutation of the
first-concurrent-backward-race family. It must not enter the refuted list.**
(The FP8_GEMM_MUTEX refutation *stands*: that build takes the lock in
`lowp_gemm_devscale`, shared with forward, so the val pass burns the insert race and
any stray permit dies at the first uncontended release, well before step-1 backward.)

Verified safe and struck from the suspect list: registry concurrent read-vs-insert
(open-addressing, cmpxchg-published fully-built nodes, x86-TSO, bcmp-verified hits, no
deletion); String/StaticString/`alloc[]` (sole route KGEN AlignedAlloc → thread-safe
TCMalloc); exception paths (thread-local); `persistent_device_buffer` keys carry
`ctx.id()` = per-rank device ordinal (no cross-rank collisions); per-rank
`DeviceContext` built fresh on its own thread (`DeviceContext::create`, not the cached
getOrCreate); zero.mojo collectives submit only on `self.ctx` and run post-backward
(not causal for pre-reduce NaN). Note: even a fixed BWD mutex serializes **host
submission only** (no sync inside the region) — a NaN from the corrected probe refutes
host-submission/first-use races, not GPU-execution-overlap hypotheses.

### 2.7 NANSCAN localization instrument (designed, not applied; amended, medium)
Ready diff at `scratchpad/nanscan.patch` (passes `git apply --check` vs blob dad2744):
`-D LLMM_DEBUG_TENSOR_NANSCAN=1` scans grads pre-collective per (class × layer) via
`global_norm_squared_gpu`, 16 launches, prints first-NaN slice in backward execution
order. **Must be amended before burning a run** (both reviews):
1. Tied **wte masking**: wte receives the last-executed encoder contribution, so every
   d_inp-chain contamination prints "first-nan: wte layer -1", concealing the origin.
   Fix: print **all** non-finite slots (or first+last+count); treat wpe as
   chain-reached-encoder sentinel and wte-alone (count 1) as LM-head-wgrad sentinel.
2. Use `not isfinite`, not `isnan` (garbage can decode Inf/huge-finite first); claim is
   "where non-finiteness first appears", not "origin".
3. Dgrad off-by-one: first non-finite wgrad convicts a two-kernel candidate set
   (consumer of corrupted d_inp), not one.
Also: build-check before use (only apply-checked); micro_step in print is constant.

---

## 3. Updated fact table

**Established:** step-1 pre-reduce local-grad NaN on a varying subset of ranks, 9/9 at
ws7, 4/5 at ws2 (one victim/run), 0..N-1 victims, deterministic compute (bit-identical
clean norms, identical loss), corruption strictly in backward, NaN enters via decode of
garbage E5M2/e4m3 operand bytes or corrupted bf16 (encoders saturate — can't emit NaN).
Defect is host-side: both mutex probes serialize host code only, device overlap
untouched, and one of them (GEMM) changes nothing while whole-backward serialization
remains *untested* (§2.6).

**Refuted (unchanged from round 1):** G2 keep-alives; per-call device allocations;
Lt heuristics cache; **Lt-vs-Lt** GEMM host serialization; registry thread-safety
(now including reader-during-insert); handle keying; amax sharing; grads padding;
collective; encoder-NaN.

**Newly refuted (round 2):** process-global device-agnostic kernel cache (per-device +
mutexed); String/heap/exception/DeviceContext-affinity family; per-rank NaN-probability
asymmetry (uniform-p artifact); pairwise-clobber victim model.

**Invalidated (remove from evidence base):** `run1_bwdmutex.log` NaN as a refutation
(broken lock, §2.6); original CLB run (never executed); CLB as a discriminator at all.

**Open:** first-use-race family (lazy module load/link, mixed Lt-init-vs-enqueue,
first-touch of zero=False buffers) vs every-step submission race; d12 reconciliation.

---

## 4. Recommended fix

### 4.1 Proper fix, apply unconditionally (real bug regardless of the NaN):
`_fp8_gemm_lock` (llmm/matmul.mojo:39-57) must converge on the registry winner:
```mojo
# after KGEN_CompilerRT_InsertGlobal(key, p):
var winner = _get_global_or_null(key)   # non-null by now, guaranteed
if winner != p:
    p.free()                            # loser discards private cell
return winner
```
plus (a) **pre-seed on the main thread**: call `_fp8_gemm_lock()` once before
`sync_parallelize` (train_gpt2.mojo:~5395) so no rank ever hits the miss window; and
(b) capture the cell pointer at acquire (matmul.mojo:3570-3576) and pass it to the
release site (3646-3647) instead of re-looking it up. Optional hardening: unlock on the
raise path between 3576 and 3646 (currently a deadlock, not a NaN).

### 4.2 Fix candidate for the NaN itself — honest labeling: **acceptable mitigation,
not yet a proper fix**, because the mechanism is unproven:
**Lock-serialized per-rank backward warmup** (max-init design, comptime-gated on fp8):
in `_run_rank` after ctx/model init and **before** the training loop / rank barrier,
acquire the (fixed, pre-seeded) process-global lock and, per rank: enqueue once with
tiny shapes each backward-only kernel — E5M2 `quantize_dual_devscale`,
`_update_scale_gpu`, standalone DGELU launcher, dbias kernels, `compute_amax` on a
scratch cell — plus one dgrad-shape and one wgrad-shape `lowp_gemm_devscale`
(e4m3×e5m2, burning Lt backward-shape init); touch/insert the backward persistent keys
(`FP8_AMAX_DOUT`, `FP8_BWD_DN_/DT_`); `ctx.synchronize()`; release. This serializes
every first (compile+load+launch+insert+Lt-init) per device and leaves steady state
fully parallel. It covers max-init candidates 1–4 and fwd-bwd-diff T1/T2 in one shot.
It becomes a "proper fix" only in the sense of a durable workaround; the true defect,
if the mechanism is confirmed, is in closed-source MAX AsyncRT / driver lazy-load
machinery → file upstream with the §2.4 disassembly evidence.
- A warmup cure must be bisected afterward (warm one kernel at a time) to name the
  guilty first-use, and must pass the later-step control (§5) before being declared
  durable — "warmup fixed it" without step->1 observability could merely be masking an
  every-step race whose probability peaks at step 1.

### 4.3 Not recommended: `CUDA_MODULE_LOADING=EAGER` (does not serialize the load
window, §2.4); permanent whole-backward mutex (masks, costs steady-state parallelism,
and diagnoses nothing).

---

## 5. Validation protocol (any fix claim)

All runs under the GPU protocol: `flock -x /tmp/llmm-gpu.lock`, inner
`timeout --signal=INT 900`, 7 healthy UUIDs only, never the faulted card, never SIGKILL.
1. **ws2 × 10 consecutive clean** (`-pn 2 -d 131072 -x 1`, ~53 s/run): chance <1% even
   at the pessimistic clean-rate bound. 5 runs are enough only to *detect* the bug.
2. **ws7 × 6 clean** (`-pn 7 -d 458752 -x 1`): against the 9/9-dirty baseline this is
   conclusive at step 1.
3. **Later-step control**: ws7 `-x 12` with the *amended* NANSCAN probe (§2.7) — must
   show no non-finite slice at any step, ruling out the fix merely moving a
   first-use-shaped race to an ongoing one.
4. **Regression canaries**: clean per-rank norms must equal the known bit-exact values
   (ws7: r0 5.487552 … r6 5.646068; ws2: 19.617536/19.492779) and loss 11.096148 /
   11.097367 — any drift means the fix changed numerics, not just timing.
5. Standard merge gates (format/lint/check/test green on CUDA) before main.

---

## 6. Single next most discriminating experiment (mechanism still unproven — run this first)

**Redo the whole-backward serialization probe with the lock actually working.**
Apply §4.1 (winner-convergence + main-thread pre-seed + pointer-passing release),
rebuild `-D WORLD_SIZE=2 -D LLMM_PRECISION=fp8 -D LLMM_FP8_BWD_MUTEX=1
-D LLMM_DEBUG_LOCAL_GRADNORM=1`, run ws2 × 10 (~9 min total GPU time), from a session
that outlives the runs (foreground or monitored — two prior background attempts were
reaped with their sessions).
- **10/10 clean** → host-submission serialization of `matmul_bwd_lowp` cures →
  corruption lives in the per-call backward host-submission sequence
  {amax, update_scale, E5M2 quantize, dgrad+DGELU, wgrad} racing across threads →
  proceed to warmup (§4.2) + bisection to name the site.
- **Any NaN** → the entire host-submission/first-use family (max-init 1–4,
  fwd-bwd-diff T1/T2, mixed Lt-init) is refuted in one stroke → pivot to
  GPU-execution/driver-level hypotheses (GSP-mediated module-link corruption at first
  launch, cross-device driver state) and shape-dependent residuals; the CLB-clean n=1
  and device-sync-clean gates become the primary signal, and the amended NANSCAN run
  becomes the next step to catch the corrupt tensor directly.
Cheap corroborator to piggyback (zero rebuild risk, one run): per-rank staggered sleep
(rank × 100 ms) injected after the val-loss allreduce — if stagger alone cures,
the post-barrier lockstep entry (§1) is confirmed as the arming condition.

---

## 7. Artifact index

Binaries (repo root): `build/train_gpt2_fp8_w7_bwdmutex(_dbg)` — **broken-lock probes,
do not draw conclusions from them**; `build/train_gpt2_fp8_w7_dbg`;
`build/train_gpt2_fp8_w2_dbg`; `build/train_gpt2_fp8_w7_m1`;
`build/train_gpt2_fp8_w1_control`.
Scratchpad `/tmp/claude-1001/-home-evan-workspace-llm-mojo/ece34132-8485-434d-9c52-d15dc2f7f9fb/scratchpad/`:
`w2_run{1..5}.log`, `step1stats/run{1..6}.log`, `clb_run1_retry.log` (clean, n=1),
`run1_bwdmutex.log` (invalidated), `run2_bwdmutex.log` (empty — no data),
`nanscan.patch` (needs §2.7 amendments), disassembly notes (`asyncrt.dis`,
`bwdmutex.dis`, `kgen.dis`, `loadfn.dis`).
Key source anchors: `llmm/matmul.mojo:39-57` (broken lock), `:3494-3647`
(`matmul_bwd_lowp`), `:3604` (E5M2 quantize site), `:1402-1427` (DGELU),
`llmm/amax.mojo:536-550`, `llmm/lowp.mojo:722-831`, `llmm/memory.mojo:73-111`,
`train_gpt2.mojo:4965` (pre-step-1 barrier), `:5395-5404` (rank spawn).
