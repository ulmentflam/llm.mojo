# fp8 multi-rank NaN — consolidated investigation record (2026-07-22..23)

Authoritative consolidation of the four-round fp8 multi-rank NaN hunt. Supersedes
the per-round reports (`fp8_multirank_race_2026-07-22.md`,
`fp8_nan_hunt_fleet_2026-07-22.md`, `fp8_nan_hunt_round2_2026-07-22.md`,
`fp8_nan_hunt_round3_2026-07-22.md` incl. its 2026-07-23 amendments,
`fp8_nan_solution_round4_2026-07-23.md`) — those are kept as historical
artifacts (§8) and every number here is the post-amendment, post-salvage final
value. Round-4/salvage batteries exist only in session scratch logs; their
results are recorded here as the primary record.

Environment (constant throughout): QEMU/KVM guest, Linux 6.8.0-136-generic,
8x RTX PRO 6000 Blackwell Max-Q via VFIO passthrough, driver 610.43.02,
Mojo 1.0.0b3.dev2026071306 (1f3b9e33), one physical card
(GPU-e9c166f6-...8001) excluded for unrelated GSP firmware faults.
Line-number citations are against main at the time of the hunt (b50aa06).

## 1. Executive summary

**Symptom.** `-D LLMM_PRECISION=fp8` GPT-2 d36 (774M), single-process
multi-rank (one host thread + one `DeviceContext` per GPU): at step 1, a
varying subset of ranks (exactly one at ws2) computes a NaN LOCAL gradient
before the allreduce; forward loss is finite; clean ranks are bit-identical
run to run. Never reproduced at ws1 (any config) or at d12/d24. Tensor-level
scanning shows every failure is exactly ONE non-finite slice out of 436 —
always the MLP down-projection weight-grad (`proj_weight`), layer 27 in most
failures.

**Final state.**

- **Cure (pending merge gates): `LLMM_FP8_NO_STASH`** — eliminate the
  forward-written fp8 transpose stashes (WT/IT) and re-quantize transposed
  operands at consumption time. 0/28 NaN under solo protocol vs 14/32
  baseline (Fisher p ≈ 2.7e-5), and step-1 ~15.5% FASTER than baseline
  (steady-state cost pending). Worktree `llm.mojo-wt/nostash`
  (`llmm/matmul.mojo` only, flag-gated, default path bit-for-bit unchanged).
- **Proven env mitigations:** `CUDA_LAUNCH_BLOCKING=1` (0/10, p=0.0007) and
  `MODULAR_DEBUG=device-sync-mode` (0/4 + 1000+ live steps). Both are global
  per-launch synchronizers; every partial serialization failed (§3).
- **Mechanism: partially known.** The strike lands in the dormant
  forward-written fp8 stash (triangulated, §4), but adversarial review of the
  cure found that the 0.3% deterministic numerics delta between the stash and
  requantize paths has NO verified explanation — the active hypothesis is
  that something rewrites stash bytes post-write even in clean ws1 runs
  (aliasing/overlap). A byte-diff probe (worktree `llm.mojo-wt/bytediff`) is
  in flight as a merge gate for NO_STASH.
- **Upstream filing (Modular primary, NVIDIA co-primary): pending the
  byte-diff outcome** — it decides between "upstream launch-machinery fault"
  and "in-repo/toolchain aliasing bug" framings (§7).

## 2. Symptom and evidence gates

Final values for the 13 gates every mechanism proposal had to reconcile:

| # | Gate | Final value |
|---|---|---|
| 1 | ws1 (any device, any accum, all site variants) | CLEAN — zero NaN across the entire campaign |
| 2 | ws2 d36 baseline NaN rate | Intermittent: 8/9 initial solo battery; 14/32 pooled solo-protocol session; exactly one victim per failing run, victim alternates between ranks |
| 3 | Clean-rank determinism | Bit-identical pre-reduce norms every clean run (ws2: r0 19.617536 / r1 19.492779; ws7: r0 5.487552 ... r6 5.646068), loss identical |
| 4 | bf16 d36 ws7 | CLEAN, 18 h (proven NaN-free, not proven corruption-free) |
| 5 | `LLMM_FP8_FWD_ONLY=1` (bf16 backward) ws2/ws7 | CLEAN — fp8 backward is a necessary component |
| 6 | `MODULAR_DEBUG=device-sync-mode` | CLEAN 0/4 + 1000+ live training steps |
| 7 | `CUDA_LAUNCH_BLOCKING=1` | CLEAN 0/10 (Fisher vs 8/10 baseline p=0.0007) |
| 8 | Whole-backward submission mutex | 3/10 NaN — NOT a cure, NOT statistically distinguishable from baseline (p=0.07) |
| 9 | First-backward rank stagger (120 ms x rank) | 4/10 NaN — lockstep entry is not the arming condition |
| 10 | Model scale | d12 0/6, d24 0/5, d36 fails; scale (width/shapes/footprint co-vary), not depth per se; d12 ws7 clean is n=1 (never retested) |
| 11 | Tensor localization (nanscan, 436 slices) | Every failure = exactly 1 non-finite slice, always `proj_weight` wgrad (IT-stash consumer), layer 27 in 3/4 (class concentration p ≈ 1.5e-5); dgrad chain always finite |
| 12 | Failure signature | Always NaN, never Xid, never garbage-loss; no peer access / UVA cross-device mappings exist on this box (`llmm/zero.mojo:242-253,306-334`), so a cross-device stray write would fault noisily |
| 13 | Step-1 only | Non-discriminating (step-1 allreduce poisons weights; later steps unobservable in failing runs) |

Round-4 addition (not numbered in the original table but load-bearing):
**concurrency suppression** — with 3 ws2 batteries running concurrently on
different GPU pairs of the same box, baseline NaN rate collapsed to 1/9 vs
8/9 solo (Fisher p=0.0034). The bug is host-launch-timing sensitive; sibling
processes on OTHER GPUs mask it. All batteries thereafter ran solo-protocol
(§6), and all cross-arm comparisons predating this discovery are confounded.

## 3. Failed attempts ledger

Every attempted fix or diagnosis, in campaign order. Format: what it
predicted / how tested / result / why it failed. All ws2 runs are d36,
`-pn 2 -d 131072 -x 1`, verdicts from `DEBUG rank R local grad norm
pre-reduce:` lines.

### 3.1 G2 keep-alives (round 1) — MISDIAGNOSIS

- **Claim:** two missed G2-pattern keep-alives (`amax_doutput` in
  `matmul_bwd_lowp`, `partial_max`/`partial_bad` in `compute_amax`) let
  Mojo's ASAP destroy release the buffers before their consumer kernels'
  enqueue; multi-rank allocator churn recycles them; garbage amax → Inf
  scale → `0.0 * Inf = NaN` via E5M2's NaN encodings.
- **Test:** keep-alives applied in-tree, failing config re-run, n=3.
- **Result:** NaN persists 3/3.
- **Why wrong (three independent kills):** (a) the quantize encoder
  SATURATES (`llmm/lowp.mojo:285-286`) — it provably cannot emit a NaN/Inf
  fp8 byte, so the round-1 NaN-generation story was impossible as written;
  (b) the runtime allocator/pool is per-device, so the cross-rank recycling
  agent doesn't exist; (c) later probes showed the corruption windows
  contained no allocations at all. Round 1's confident "root cause" write-up
  was refuted empirically within hours — recorded here as the cautionary
  headline of the campaign.

### 3.2 M1 persistent-buffer conversion (fleet round 1) — lifetime family dead

- **Claim:** even WITH keep-alives, per-call `DeviceBuffer` release lands
  while consumer kernels are pending; under 7-thread contention the release
  path degrades and the block is recycled. Fix: eliminate EVERY per-call
  device alloc/free from the fp8 backward (persistent `FP8_AMAX_DOUT`,
  `FP8_BWD_DN_/DT_`, `AMAX_PARTIAL_*` buffers).
- **Test:** conversion implemented (kept in-tree as hygiene/perf), failing
  config re-run.
- **Result:** NaN persists. With 3.1, the ENTIRE buffer-lifetime family is
  dead: the fp8 backward can allocate nothing per call and still NaN.

### 3.3 cuBLASLt heuristics-cache-off — dead

- **Claim:** concurrent heuristics-cache access in the e4m3xe5m2 GEMM
  dispatch corrupts algo selection.
- **Result:** NaN persists with the cache disabled. Also structurally
  disfavored later by nanscan: all 36 layers' proj wgrads share one shape
  (1280x5120x8192) and hence one algo choice — a broken selection would NaN
  all layers or none, observed is exactly one.

### 3.4 GEMM submission mutex (`LLMM_FP8_GEMM_MUTEX`) — dead, plus a probe meta-bug

- **Claim:** cuBLASLt host-side state races concurrent submissions;
  serializing `lowp_gemm_devscale` (descriptor/heuristics/`cublasLtMatmul`)
  behind a process-global lock cures.
- **Meta-bug (document as a lesson):** the first lock implementation
  (`_fp8_gemm_lock`, `llmm/matmul.mojo:39-57` at the time) was itself broken
  on first-call races — on registry-insert miss each loser thread returned
  its OWN private cell instead of the registry winner, so racing ranks
  acquired different locks and ran unserialized; worse, the release-site
  re-lookup made a loser's `store(0)` spuriously unlock the winner's held
  mutex, and the extra permit persisted (blind-store release). Every
  conclusion drawn from a broken-lock binary was struck from the evidence
  base. Fix: converge on the registry winner after `InsertGlobal`, pre-seed
  the cell on the main thread before `sync_parallelize`, pass the cell
  pointer from acquire to release.
- **Result (fixed lock, ws2 x10):** 4/10 NaN (Fisher vs baseline p=0.17).
  Not a cure; Lt-vs-Lt host serialization does not close the window.

### 3.5 Whole-backward submission mutex (`LLMM_FP8_BWD_MUTEX`) — dead

- **Claim:** the corruption lives in the per-call backward host-submission
  sequence {amax, update_scale, E5M2 quantize, dgrad+DGELU, wgrad} racing
  across threads; serializing all of `matmul_bwd_lowp` submission cures.
- **Test:** first run invalidated by the 3.4 broken lock (that log must
  never be cited). Re-run with the verified-sound pre-seeded lock, ws2 x10.
- **Result:** 3/10 NaN. Host submission concurrency is not the mechanism —
  each rank's async queue drains deep after lock release, and the strike
  survives.
- **Statistics lesson (round-3 error):** the rate reduction 8/10 → 3/10 was
  originally reported as p ≈ 0.009 and used to claim "proven host-timing
  sensitivity." That p-value was computed against a FIXED p=0.8 null, not
  the empirical baseline; the correct Fisher exact is p ≈ 0.07. All four
  partial-serialization arms (3.4-3.7) are statistically indistinguishable
  from baseline AND from each other at n=10 (pooled trend p ≈ 0.03 at most).

### 3.6 Per-site backward drain (`LLMM_FP8_BWD_SYNC_ONLY`) — dead; round 3's own prediction failed

- **Claim (round 3's committed mechanism):** per-rank async launch run-ahead
  is the window; bounding queue depth to one fp8-backward site bundle via
  `ctx.synchronize()` at the end of every `matmul_bwd_lowp` — no lock, no
  cross-rank coupling — should cure exactly like CLB/device-sync-mode.
  Pre-registered as the shipping mitigation.
- **Result (ws2 x10):** 5/10 NaN. REFUTED round 3's headline "forcing queue
  depth ~1 closes it" in its per-site-drain limb: the window is NOT the
  fp8-backward launch bundle. (Global per-launch sync still cures; whatever
  is being synchronized away is not site-granular.)

### 3.7 Execution exclusion (`LLMM_FP8_BWD_EXEC_MUTEX`) — dead

- **Claim:** cross-GPU co-execution of fp8 backwards corrupts operands; hard
  mutual exclusion (lock held across enqueue AND a full stream drain, so no
  two ranks' fp8 backwards ever co-execute) cures.
- **Result (ws2 x10):** 3/10 NaN. The strongest single refutation of the
  cross-GPU-overlap family: the strike lands even when sibling fp8 activity
  is excluded for the whole window. (Also operationally nasty: a raise while
  holding the lock livelocks peers — timeouts scored inconclusive, never
  clean.)

### 3.8 First-backward rank stagger (`LLMM_DEBUG_RANK_STAGGER`) — dead

- **Claim (round 2):** the val-loss allreduce barrier immediately before
  step 1 (`train_gpt2.mojo:4965`) releases all ranks into
  never-before-executed backward code in lockstep; breaking the lockstep
  (120 ms x rank sleep post-barrier) removes the arming condition.
- **Result (ws2 x10):** 4/10 NaN. Bit-exact lockstep entry is not the arming
  condition. (The related serialized-warmup fix design was withdrawn before
  ever running: its first-execution premise was unsupported and it had a
  real z>=3 deadlock via `_z3_stream_layer` all-gathers inside backward.)

### 3.9 Rank-dependent layout offset (`LLMM_FP8_RANK_OFFSET`) — dead; refuted spatial scribble

- **Claim:** the rank-independent layer-27 preference means a fixed target
  address in the per-rank-identical allocation layout; giving each rank a
  unique layout offset moves the victim out from under a fixed-address
  cross-context scribble.
- **Result (solo protocol, ws2 x10):** 6/10 NaN, both ranks struck under
  rank-unique layouts. The corruption is not a fixed-address spatial
  scribble — whatever writes travels WITH the victim's own addresses
  (kernel-argument semantics), not to an absolute location.

### 3.10 Prewarm with exclusion (`LLMM_FP8_PREWARM`) — dead; refuted first-use

- **Claim:** first-use machinery (lazy module load/link, first-launch
  compile, Lt backward-shape init) racing across threads at step 1; warming
  every backward kernel/GEMM per rank under exclusion before the training
  loop removes all concurrent first-uses.
- **Result (solo protocol, ws2 x10):** 6/10 NaN with all first-uses
  pre-burned. The first-use family (round 2's top pick) is dead as the
  mechanism.

### 3.11 Runtime env knob sweep — knob space exhausted

Every dispatch-adjacent runtime/driver knob discoverable (including by
disassembly of the shipped MAX libs — no queue-depth/worker-count knobs
exist) was swept at ws2 x N. All failed:

- AsyncRT busy-wait off (`BUSY_WAIT=0` class): >=1 treatment NaN.
- Host thread CPU affinity pinning: >=1 treatment NaN.
- `CUDA_DEVICE_MAX_CONNECTIONS=1` (HW channel serialization within a
  context): >=1 treatment NaN — intra-context channel serialization does not
  remove the path.
- `MLRT_CUDA_DEBUG=1`: 6/6 NaN — a possible AGGRAVATOR (tighter/perturbed
  host launch timeline increases collision frequency); NaN runs were also
  faster, which disfavors "any slowdown masks it" readings.
- `CUDA_MODULE_LOADING=EAGER`: rejected without a run — disassembly showed
  the real load/link runs concurrently in the same step-1 window under both
  LAZY and EAGER, so it is neither a fix nor a discriminator.

### 3.12 Stash-lifecycle checkpoint probe (S6) — broken instrument, no data

- **Goal:** the pre-registered mechanism kill shot — isfinite counts on the
  proj stash (a) after forward write, (b) before wgrad consumption, (c) on
  wgrad output, to directly observe WHERE the strike lands.
- **Result:** zero STASHCHECK output lines across 12 treatment + 5 baseline
  runs (ambient collapse; the probe's own +16% timing perturbation is partly
  self-defeating). **No direct strike observation was ever made in the
  entire campaign** — the dormant-stash attribution (§4) rests on
  triangulation, not on catching the write. The byte-diff probe (§7) is the
  successor instrument.

### 3.13 Hypotheses refuted by audit alone (no fix attempted)

- Registry/`InsertGlobal` insert race: insert set byte-identical between
  clean FWD_ONLY and failing full-fp8; KGEN table is CAS-published,
  bcmp-verified, no deletion/resize (disassembly-verified, including
  reader-during-insert).
- Shared cuBLASLt handle: `_get_global_handle` keys include `ctx.id()` —
  per-device.
- Process-global device-agnostic kernel cache: runtime kernel cache is
  per-physical-device and mutex-protected (disassembly).
- Pure numerics (e5m2 delayed-scale overshoot): rank-count- and
  sync-mode-independent, killed by gates 1/6/7; encoder saturates.
- Collective/barrier ordering: bulk-synchronous with sync-then-barrier at
  every phase edge (`llmm/zero.mojo:455-482`); bf16 18 h clean.
- Peer/UVA silent cross-device writes: no peer mappings exist;
  `can_access == False`; a cross-device dereference would Xid, not silently
  write.
- Electrical/power transients, marginal silicon: FWD_ONLY co-executes
  same-power-class fp8 GEMM bursts cleanly; failures always-NaN-never-Xid;
  survivors bit-identical.
- Quantize kernel itself: full byte coverage both outputs, zero partial
  tiles at all shapes, saturating encoder — cannot produce NaN bytes or
  leave unwritten bytes.
- String/heap/exception/DeviceContext-affinity family: sole allocation route
  is thread-safe; per-rank contexts built fresh on their own threads.

### 3.14 Process lessons (meta-failures)

1. **The round-3 statistics error** (§3.5): a p-value against an assumed
   fixed null instead of the empirical baseline inflated 0.07 into 0.009 and
   propped up a mechanism claim for a full round. Rule: Fisher exact against
   the interleaved same-session baseline, always.
2. **The round-4 concurrency confound** (§2): baseline NaN rate is a
   function of unrelated activity on OTHER GPUs (1/9 concurrent vs 8/9 solo,
   p=0.0034). Every arm comparison run before this discovery used
   potentially suppressed baselines; all batteries thereafter mandated the
   solo protocol. Rule: for timing-sensitive bugs, own-battery interleaved
   baselines under an exclusive-box lock, or the comparison is void.
3. **The broken-lock meta-bug** (§3.4): a serialization probe must have its
   lock verified (pre-seeded, winner-converged) before any conclusion is
   drawn from it; one invalid refutation cost a round of misdirection.
4. **Background-run hygiene:** two runs were lost to reaped background
   sessions (0-byte logs). Runs are launched from sessions that outlive
   them, timeouts use `--signal=INT`, and empty/killed logs are scored
   inconclusive — never clean.

## 4. What cures, and what we know of the mechanism

### 4.1 The cure: `LLMM_FP8_NO_STASH`

The fp8 forward previously wrote, per site/layer, two long-lived e4m3
transposed stashes (`quantize_dual_devscale` into the persistent WT/IT
`lowp_transpose_cache` buffers), consumed by backward dgrad (WT) and wgrad
(IT). Under `-D LLMM_FP8_NO_STASH=1` the forward quantizes natural-layout
only, and each backward consumer re-quantizes its transposed operand fresh
(`quantize_transpose_devscale`, same source, same scale) immediately before
the consuming GEMM. The dormant window shrinks from an entire
forward→backward span to a launch-to-launch gap.

Evidence: round-3 batteries 0/13 conclusive treatment runs (vs interleaved
4/10 baseline, p ≈ 0.024, concurrency-suppressed ambient); round-4/salvage
solo-protocol batteries pooled **0/28 treatment vs 14/32 solo baseline,
Fisher p ≈ 2.7e-5**. Step-1 ~15.5% faster than baseline (removing the
strided transposed write from the forward dual-quantize outweighs the two
extra backward transpose-quantize kernels); steady-state cost pending.
Forward is bit-identical to baseline; backward is deterministically shifted
~0.3% (see 4.3).

### 4.2 Dormant-stash strike triangulation

No probe ever caught the write directly (§3.12). The attribution rests on:

1. **nanscan** (gate 11): the only tensor ever struck is `proj_weight` wgrad
   — the IT-stash consumer; the dgrad chain and every non-fp8-laundered grad
   stay finite. A struck IT stash surfaces as exactly one corrupted wgrad
   slice.
2. **Removing the stash removes the failure** (4.1) while every
   serialization of the legitimate work stream fails (§3.4-3.7) — the fix
   does not require the corrupting writer to stop existing, only the buffer
   that dormantly holds consumable fp8 bytes.
3. **The 19.559965 identity datum:** a faulted stash-path run (`MLRT_CUDA_
   DEBUG` arm) printed rank-0 pre-reduce norm 19.559965 — bit-identical, to
   all printed digits, to the deterministic value of the
   requantize-at-consumption (NO_STASH) build, where the stash-path clean
   value is 19.617536. A stash-path run producing exactly the requantize
   arm's value means the stash bytes at consumption matched a LATE
   re-execution of the victim's own transpose-quantize — corruption as a
   mistimed well-formed write, with NaN when it tears.
4. **Arming:** ws1 is 100% clean (sibling rank thread/context required);
   concurrency suppression and the MLRT aggravator show host-launch-timing
   sensitivity; layer-27 preference is rank-independent (tracks
   machinery/submission order, not data).

### 4.3 The 0.3% clue — and the honest hole in the story

NO_STASH's backward is deterministically ~0.3% shifted vs the stash path
(r0 19.559965 vs 19.617536; post-step val Δ ≈ 2.3e-4). Round 4 initially
explained this as benign: the consumption-time quantize reads the live bf16
source "which backward has by then legally mutated in place." **Adversarial
review REFUTED that explanation:** the same scale is provably used in both
paths, and no legal mutation of the source tensor between the forward write
and the backward read exists in the code. The delta therefore has NO
verified explanation — yet it is real, deterministic, and bit-exactly equal
to what a faulted stash-path run produced.

**Active hypothesis:** something rewrites the stash bytes post-write even in
clean ws1 runs — an aliasing/overlap between the stash and another buffer
(in this repo or the runtime's allocation of the persistent caches) — and
the ws>1 failure is the timing-dependent violent edge of the same rewrite.
This would relocate part of the blame from "upstream launch machinery" to an
addressable aliasing bug. The byte-diff probe (per-site compare of stash
bytes vs a fresh requantize at consumption, ws1 clean runs, worktree
`llm.mojo-wt/bytediff`) is in flight as a NO_STASH merge gate and as the
discriminator (§7).

Additional datum in tension with round 3/4 claims: under the solo protocol,
**fwd+bwd per-site drains went 0/10 clean** (the earlier 1/10-with-NaN
result predates the solo protocol and is confounded). At n=10 vs 14/32 this
is suggestive, not decisive — but it weakens the round-4 assertion that the
write "travels outside the victim stream's ordering," which rested on that
earlier T06 NaN.

### 4.4 Honest unknowns

- Writer identity (shared MAX AsyncRT launch machinery vs guest-driver/GSP
  under VFIO vs in-repo/toolchain aliasing) — undetermined.
- The 0.3% delta's cause — unexplained (byte-diff pending).
- Micro-step of the strike — never localized (probes ran post-accumulation).
- The d36-only scale dial — width/shape/footprint confound (C=1280 cuBLASLt
  kernels) untested in isolation.
- NO_STASH is untested above ws2 (ws7 battery is a Phase-B gate).
- bf16 multi-rank is proven NaN-free, not corruption-free.

## 5. Mitigations and status

| Mitigation | Kind | Evidence | Cost | Caveats |
|---|---|---|---|---|
| `LLMM_FP8_NO_STASH=1` | comptime, cure-class | 0/28 solo (p ≈ 2.7e-5) + 0/13 round 3 | step-1 ~15.5% FASTER; steady-state pending | Pending merge gates: byte-diff probe outcome, ws7 x6 battery, 3x 12-step runs, repo lint/test gates; 0.3% numerics shift unexplained (§4.3) |
| `CUDA_LAUNCH_BLOCKING=1` | env-only | 0/10 ws2 (p=0.0007) | ~0-4% at d36 (big-GEMM amortizes) | Proven at ws2 x1-step only; steady-state cost at scale unmeasured; primary fallback if NO_STASH gate fails |
| `MODULAR_DEBUG=device-sync-mode` | env-only | 0/4 + 1000+ live training steps | ~4-5% | Heaviest; fallback of last resort |

**Standing rule (per `low_precision_gotchas.md` G4):** any fp8 multi-rank
run MUST use one of these until NO_STASH lands as default (Phase B inverts
the gate to `LLMM_FP8_STASH_LEGACY=1`) or an upstream fix arrives.

Explicitly NOT mitigations (kept only as diagnostics): all mutex/drain/
stagger/offset/prewarm flags (§3) — none cure, some cost heavily.

## 6. Reproducer kit

All flags comptime-inert when undefined; they are the reproduction and
validation kit for the upstream reports.

- Core: `-D LLMM_PRECISION=fp8 -D WORLD_SIZE=N`,
  `-D LLMM_DEBUG_LOCAL_GRADNORM=1` (prints the per-rank pre-reduce verdict
  lines).
- Cure/differential: `-D LLMM_FP8_NO_STASH=1` (worktree
  `llm.mojo-wt/nostash` until merged); `-D LLMM_FP8_FWD_ONLY=1`.
- Failed-arm probes (in-tree): `LLMM_FP8_GEMM_MUTEX`, `LLMM_FP8_BWD_MUTEX`,
  `LLMM_FP8_BWD_SYNC_ONLY`, `LLMM_FP8_FWD_SYNC_ONLY`,
  `LLMM_FP8_BWD_EXEC_MUTEX`, `LLMM_DEBUG_RANK_STAGGER`,
  `LLMM_DEBUG_TENSOR_NANSCAN` (436-slice isfinite scan). Worktree-only:
  rankoffset, prewarm, stashcheck, bytediff.

Canonical ws2 repro build and run:

```bash
pixi -q run mojo build -D WORLD_SIZE=2 -D LLMM_PRECISION=fp8 \
  -D LLMM_DEBUG_LOCAL_GRADNORM=1 -I . -Xlinker -lm \
  -o build/train_gpt2_fp8_w2_dbg train_gpt2.mojo
# -e d36 -b 4 -t 1024 -pn 2 -d 131072 -x 1, FineWeb shards from /data,
# first two healthy GPU UUIDs. ~53 s/run. NaN verdict from the
# "DEBUG rank R local grad norm pre-reduce:" lines; clean references
# r0 19.617536 / r1 19.492779 (stash path), r0 19.559965 (NO_STASH).
```

**Solo protocol (mandatory — see §3.14.2):** the bug's rate collapses when
anything else runs on the box's other GPUs. Batteries must hold the
exclusive GPU locks (all pair locks / `flock -x /tmp/llmm-gpu.lock`), run
interleaved baseline/treatment from ONE script, `timeout --signal=INT
600`, never SIGKILL (GSP teardown hang), GPUs pinned by UUID with
GPU-e9c166f6-...8001 excluded, sessions that outlive the runs. Score any
timeout/rc!=0/empty log inconclusive, never clean. Fisher exact against the
interleaved baseline only.

## 7. Upstream reporting status

**Status: filing PENDING the byte-diff probe outcome.** Draft bodies live in
the round-3 (§7, post-amendment titles) and round-4 (§6) documents; the arm
table to attach is §2 + §3 here.

The byte-diff probe (ws1, clean runs: per-site compare of stash bytes at
consumption vs a fresh requantize) decides the framing:

- **Stash bytes differ at ws1** → the post-write rewrite is real and
  environment-independent → primary suspicion moves to aliasing/overlap
  (this repo's persistent-cache layout or the toolchain's allocator), the
  0.3% delta is explained, and the Modular filing becomes "buffer
  aliasing/corruption reproducible single-context" (much stronger repro) —
  or the bug is ours and gets fixed here instead of filed.
- **Stash bytes identical at ws1** → the rewrite exists only under
  multi-context timing → the round-3/4 framing stands: Modular primary
  (multi-DeviceContext single-process launch machinery; cured only by
  global per-launch synchronization; every partial serialization and env
  knob fails; dormant-buffer removal cures), NVIDIA co-primary (GSP-RM
  launch replay/dedup under QEMU/KVM VFIO; armed by a sibling context's
  existence; both NaN bytes and well-formed stale-semantics values
  observed — the 19.559965 identity datum). Include the concurrency-
  suppression finding and the C=1280 shape note in both.

## 8. Historical artifacts (superseded per-round reports)

Kept unedited (plus a superseded banner) as the audit trail; cite THIS
document, not them:

- `docs/ai/fp8_multirank_race_2026-07-22.md` — round 1: the G2 keep-alive
  "root cause" (refuted, §3.1) and the first full audit of the fp8 backward.
- `docs/ai/fp8_nan_hunt_fleet_2026-07-22.md` — fleet round 1: evidence
  gates, M1 persistent-buffer design (§3.2), first refuted-mechanism list.
- `docs/ai/fp8_nan_hunt_round2_2026-07-22.md` — round 2: ws2 minimal repro,
  step-1 statistics, first-use/lockstep hypothesis (refuted §3.8/3.10), the
  broken-lock meta-bug (§3.4), MAX runtime disassembly audit.
- `docs/ai/fp8_nan_hunt_round3_2026-07-22.md` — round 3: CLB 10/10,
  nanscan localization, scale curve, run-ahead mechanism (its per-site-drain
  limb refuted by its own 2026-07-23 amendments, which also correct the
  p=0.009 error and record the final arm table).
- `docs/ai/fp8_nan_solution_round4_2026-07-23.md` — round 4: NO_STASH
  selection, mechanism sentence, promotion plan (its solo-battery section
  was never filled in-doc; final numbers are in §4.1 here, from scratch
  logs).
- `docs/ai/low_precision_gotchas.md` G4 — the operational rule; points here.

Round-4/salvage scratch artifacts (session-scoped, will not survive):
`nostash_progress.txt`, `nostash_battery/`, `solo_battery/`, dead-arm
worktrees `s1-fwdsync`/`rankoffset`/`prewarm`/`stashcheck` (prune after the
upstream filings land).


---

## Final findings (2026-07-23/24, merge-gate phase)

- **Byte-diff probe (ws1, 3-way compare, 3.35 GB verified twice): the stash is
  never touched** between forward write and backward consumption — the
  aliasing hypothesis is refuted, and no in-process corruption exists at ws1.
- **The ~0.3% numerics delta and the step-1 speed difference are a benign
  allocation-layout execution variant** of the backward cuBLASLt GEMMs: the
  ALLOCORDER arm (unmodified stash path, requantize path's allocation order)
  reproduces the requantize path's norms and timing exactly with byte-identical
  operands.
- **The cure is causal, not layout**: ALLOCORDER at ws2 still fails 5/10 while
  the requantize path is 0/48 across ws2 and ws7 (incl. 6/6 ws7 and 3x 12-step
  at the production z1 config). Eliminating the dormant stash window is the
  operative difference.
- **True steady-state cost: +3.1%** (ws2 d36 -x 20 means; the round-4 "15.5%
  faster" was a step-1/layout artifact).
- **verify-fp8-grads / verify-fp8-static-grads fail on this box at every
  runnable sha with byte-identical numbers** — the envelope was calibrated on
  the GB10 (sm_121) and the dump tool could not run here before 82c79f3;
  attribution is hardware-lineage (sm_120 cuBLASLt algorithm selection), not
  any commit. Both fp8 paths produce byte-identical dumps, so the requantize
  default does not move these gates. FOLLOW-UP: recalibrate the envelope for
  sm_120, and separately root-cause the ln_1_gamma layer-0..4 cosine collapse
  (the one pathological-looking signature).
- Shipped: the requantize path is the fp8 DEFAULT; `-D LLMM_FP8_STASH_LEGACY=1`
  keeps the old stash path for reproduction/upstream work. `CUDA_LAUNCH_BLOCKING=1`
  remains the documented env-level defense-in-depth. `uvx ruff` pinned to
  0.15.2 in the Makefile (0.16.0 restyled the tree mid-merge).
