# fp8 multi-rank NaN hunt — Round 3 synthesis (2026-07-22, late)

Status: **mechanism committed** (one level above final attribution), round-3 pivot
hypothesis **refuted by round 3's own data**, strike site localized to a single GEMM
class, mitigation implemented in-tree and awaiting the validation battery below.

Round-3 headline in one paragraph: the "cross-GPU execution overlap corrupts operands"
pivot is dead — both known cures (`CUDA_LAUNCH_BLOCKING=1`, now 10/10 at ws2, and
`MODULAR_DEBUG=device-sync-mode`, 4/4) are **per-rank-only** synchronizers that leave
cross-GPU wall-clock kernel co-execution essentially intact, yet they cure completely.
What they actually remove is per-rank **asynchronous launch run-ahead** (queue depth).
Meanwhile tensor-level scanning localized every failure to **exactly one** non-finite
slice out of 436 — always the **MLP down-projection weight-grad (`proj_weight`)**, layer
27 in 3/4 failures regardless of which rank is the victim — and the code audit proved
there is **no peer access and no UVA cross-device mapping at all** on this box, so no
other GPU can even silently touch the victim's memory. The corruption therefore travels
through the victim rank's *own* deep async launch path, and the second rank is an
*arming condition* (shared in-process runtime and/or shared guest driver instance), not
a direct writer.

---

## 1. Executive verdict — THE mechanism

**Committed mechanism (class level, every gate reconciled in §3):**

> **A per-rank asynchronous run-ahead race in the launch path beneath the application:
> with a sibling rank thread active in the same process (and the same QEMU/KVM VFIO
> guest driver instance), one element of a rank's deep in-flight fp8-backward launch
> stream — operand staging, launch arguments, or an in-flight buffer — is corrupted.
> The strike lands at a fixed site: the `proj_weight` wgrad GEMM (or its fp8 operand
> stash), preferring layer 27. fp8's dense NaN encodings (~3%/garbage byte in e5m2,
> ~0.8% in e4m3) convert the corruption into the observed NaN local grad. The window
> exists only when the per-rank async queue is deep; forcing queue depth ≈ 1
> (`CUDA_LAUNCH_BLOCKING`, device-sync-mode, or a per-site stream drain) closes it.**

Necessary conditions, all established empirically:
1. **fp8 backward path active** (FWD_ONLY ws2/ws7 clean; bf16 NaN-free).
2. **≥ 2 rank threads in one process** (ws1 clean on every device individually).
3. **~774M scale** (d36 ~80% fail; d24 0/5, d12 0/6 — scale, not depth alone: width,
   shapes, and footprint co-vary; see §3.9).
4. **Deep per-rank async run-ahead** (all cures bound it; no failure-mode perturbs it).

**Sub-attribution — the one thing not yet committed.** Two candidates survive, both
cured identically by queue-depth-1:

- **(A) Shared in-process MAX AsyncRT host machinery** (launch staging/arg buffers,
  event/descriptor pools, per-context queues/worker threads — flagged as cross-rank
  shared state in `docs/ai/fp8_multirank_race_2026-07-22.md:95-99`). Favored by: the
  whole-backward *submission* mutex changed the rate (8/10 → 3/10, P(X≤3|n=10,p=.8) ≈
  0.009 — a host-timing-sensitive window, impossible under a pure hardware-overlap
  story); ws1-clean maps to "no sibling host thread"; the scale dial maps to in-flight
  queue depth.
- **(B) NVIDIA guest-driver/GSP deep-queue launch path under VFIO/QEMU** (this box
  already has one card excluded for GSP firmware faults). Not excluded by anything;
  discriminated from (A) by the two-OS-process experiment in §6.5.

The **application and llm.mojo code are exonerated** (round-3 audits, reviewer-verified):
no peer access anywhere (`llmm/zero.mojo:306-334`; `can_access == False`,
`zero.mojo:242-253`), no collective before the step-1 pre-reduce print
(`train_gpt2.mojo:4042-4064` at `-z 0`), all fp8 buffers/handles/streams per-rank
(`llmm/memory.mojo:86` ctx.id()-keyed; `matmul.mojo:440-442`), the KGEN registry is a
CAS-published concurrent table (disassembly-verified), no host-pinned pointer reaches
the fp8 backward, and the quantize kernel **provably cannot emit a NaN/Inf fp8 byte**
(saturating encoder `lowp.mojo:285-286`; SR path is a comptime-error seam) and leaves
zero unwritten bytes at d36/d12 shapes. Whatever writes the bad bytes does so *after*
the producer kernel, through the launch/runtime layer — this is an upstream bug (§7).

---

## 2. Round-3 results digest (adversarial-review amendments folded in)

### 2.1 CUDA_LAUNCH_BLOCKING=1, ws2 × 10 — **10/10 CLEAN** (CONFIRMED)
`build/train_gpt2_fp8_w2_dbg`, standard template. All 20 pre-reduce norms finite and
bit-identical to the known clean values (r0 19.617536 / r1 19.492779), rc=0 × 10.
Same-binary same-day baseline without CLB: 4/5 NaN. p ≈ 0.2^10 ≈ 1e-7 under the 80%
null (≤1e-3 even under a conservative 50% baseline). **Amendment (both reviewers):**
CLB is a *per-rank* host-device synchronizer — it imposes no cross-rank ordering and
the two GPUs still co-execute kernels for most of the wall clock. So this result is
evidence **against** "cross-GPU co-execution corrupts", and **for** "per-rank
run-ahead is the window". Cost note: CLB step times 3.5–3.7 s vs plain 3.547–3.553 s
(≤ ~4%).

### 2.2 First-backward stagger (120 ms × rank), ws2 × 10 — **4/10 NaN** (no cure)
`build/train_gpt2_fp8_w2_stagger_dbg`. One victim per failing run (r1×3, r0×1), clean
ranks bit-identical. Rules out **bit-exact lockstep entry** as the arming condition —
nothing more: 120 ms vs ~443 ms micro-backwards leaves ~73% overlap all step, and the
offset persists (no per-micro-step cross-rank sync at `-z 0`). 4/10 vs 8/10 is not
significant (Fisher p ≈ 0.17). Under §1's mechanism the stagger is expected to fail:
it neither bounds queue depth nor changes shared-machinery contention.

### 2.3 Tensor-level NaN scan, ws2 × 10 — **strike site localized** (CONFIRMED)
`build/train_gpt2_fp8_w2_nanscan` (isfinite probe, 436 slices = 12 classes × 36 layers
+ wte/wpe/ln_f γ/β, pre-reduce). 4/10 failed; **every** failure = exactly **1/436**
non-finite slice, **always `proj_weight`** (MLP down-projection wgrad): L27, L27, L27
(victims r1, r1, r0), L3 (r0). Class concentration p ≈ 1.5e-5 vs uniform; L27 3/4
p ≈ 3e-3 (n=4). Never qkv/attn_proj/fc, never a bias, never ln, never wte/wpe —
i.e. the **dgrad chain never went non-finite in any micro-step** (downstream wgrads,
including non-fp8-laundered bf16/fp32 bias/ln grads, all finite). Isolated corruption
of one site's wgrad output **or** its fp8 activation stash (`proj_input` — written in
forward, consumed micro-steps later in backward: the longest-lived fp8 operand,
i.e. the largest exposure window to a stray in-flight write). Micro-step of corruption
is unlocalized (probe runs once at micro_step 7; wgrads accumulate). Rank-independent
layer preference = fixed target in the per-rank-identical allocation layout **or** a
fixed temporal window in the per-rank-identical submission timeline — not yet
discriminated (§6.5).

### 2.4 Sanity pair — both gates hold (CONFIRMED, with one new signature)
device-sync-mode ws2 4/4 clean *and* bit-identical; FWD_ONLY ws2 4/4 finite. New:
FWD_ONLY grad norms are **not** bit-identical — each rank toggles between exactly two
values (~0.3% apart), a binary timing-dependent kernel/variant selection somewhere in
the bf16-backward stack. Consequence: bit-identity is a valid cleanliness criterion
only for plain-fp8/sync-mode arms. (The "broken-variant-selection" alternative this
spawned is now disfavored — see §4.4.)

### 2.5 Scale curve — **model scale is a sharp dial** (amended from "depth")
d12 (124M) 0/6, d24 (355M) 0/5, d36 (774M) ~8/10. An 80%-like rate is excluded at
both smaller scales (P < 1e-3 each); 95% upper bounds are ~45% (d24) and ~39% (d12),
so low-rate failure there is not excluded. Width/heads/footprint co-vary with depth
(d12=768C, d24=1024C, d36=1280C), so the dial is *scale*, not depth per se. The d12
"reframing trigger" (any d12 ws2 failure) did **not** fire. d12 **ws7** remains an
untested n=1 gate (§6.4).

### 2.6 Execution-level audit — app/pointer layer exonerated, pivot premise killed
See §1. Key kill: **the round-3 pivot's stated premise was wrong** — nothing enables
peer access; there are no cross-device mappings; a cross-device dereference would
fault noisily (Xid 13/31), not silently write. Only host-pinned memory is silently
cross-device-writable and none of it reaches the fp8 backward. The audit's own
"electrical/clock transient" #1 ranking was refuted on review (FWD_ONLY ws7
co-executes same-power-class fp8 GEMM bursts cleanly; sync-mode keeps duty cycle high
yet is clean; failures are always-NaN-never-Xid; survivors bit-identical; both ws2
cards alternate as victim).

### 2.7 Quantize-path audit — kernel exonerated; registry lead raised then demoted
Kernel: full byte coverage both outputs, zero partial tiles at all shapes, symmetric
guards, uniform barriers, saturating encoder ⇒ **cannot** produce NaN bytes or leave
unwritten bytes; holds for both per-call (HEAD) and persistent (uncommitted) DN/DT
allocation — consistent with the bug predating the switch. The "lockless registry
lost-insert" lead was **refuted as argued**: `persistent_device_buffer` keys embed
`ctx.id()` so each key has exactly one inserting thread; the KGEN table is
CAS-published; and a fresh never-written allocation is plausibly zero-scrubbed (finite
zeros, not NaN). Surviving value: the WT/IT and proj_input caches are written in
forward, read in backward, and **never validated by FWD_ONLY** — the natural victim
window for §1's mechanism.

### 2.8 Exec-serialization probe — built, not yet run; interpretation matrix amended
`build/train_gpt2_fp8_w2_execmutex_dbg` (`-D LLMM_FP8_BWD_EXEC_MUTEX=1`: lock held
across enqueue **and** stream drain). Reviewer-critical amendment: the probe conflates
cross-rank exclusion with intra-rank queue-depth-1, and since per-rank-only cures
already exist, "10/10 clean ⇒ cross-GPU execution confirmed" is a non-sequitur.
Required control = the **sync-only** variant (drain without lock), now implemented
(§5). The proposed "serialized warmup micro-step" fix (2a) is **withdrawn**: its
premise (first-execution arming) is unsupported (§2.2), and it had a real z≥3
deadlock gap (`_z3_stream_layer` all-gathers inside backward). The exec-mutex binary
is retained only as the escalation branch in §6.

---

## 3. Gate reconciliation — every fact vs the committed mechanism

| # | Gate / fact | Reconciliation under §1 |
|---|---|---|
| 1 | ws1 always clean (every device) | No sibling rank thread in the process / second context in the guest driver ⇒ arming condition absent. |
| 2 | ws2 ~80%, exactly one victim, victim varies | Per-run timing decides which rank's in-flight element is hit; one discrete corruption event per failure. |
| 3 | Clean ranks bit-identical every run | Corruption is a single discrete event; all other compute is deterministic. Confirmed again in every round-3 arm. |
| 4 | bf16 ws2/ws7 clean | Different backward path and lower launch density; **and** bf16 garbage bytes are usually finite — bf16 is proven NaN-free, *not* corruption-free (honest caveat; see §6.6). |
| 5 | FWD_ONLY ws2/ws7 clean | fp8 backward is a necessary component; forward first-uses run under natural startup skew; forward never *reads* the fwd-written fp8 stashes. |
| 6 | device-sync-mode always clean | Per-enqueue own-stream sync ⇒ queue depth 1 ⇒ window closed. Cross-GPU overlap persists ⇒ overlap not the mechanism. |
| 7 | CUDA_LAUNCH_BLOCKING 10/10 clean | Same equivalence class as #6 (per-rank synchronous launch). Same reading. |
| 8 | Whole-backward submission mutex still 3/10 | App-thread windows serialized but each rank's async queue drains deep after release ⇒ window survives; the significant rate *reduction* (p≈0.009) proves host-timing sensitivity — inconsistent with pure hardware overlap. |
| 9 | Stagger 120 ms 4/10 | Phase offset bounds neither queue depth nor shared-machinery contention. |
| 10 | d12/d24 clean, d36 ~80% | Bigger model ⇒ more in-flight work per queue, longer exposure of in-flight state, plus C=1280-specific cuBLASLt kernels (unresolved confound, §6.6). |
| 11 | nanscan: always `proj_weight`, L27-preferring, rank-independent | Fixed strike site in the per-rank-identical allocation layout / submission timeline; candidates: proj wgrad launch itself or the long-lived fp8 `proj_input` stash. Spatial-vs-temporal open (§6.5). |
| 12 | NaN specifically (never Xid, never garbage-loss) | fp8 NaN encodings are dense (6+2 of 256 in e5m2); saturating encoder means the bytes were corrupted **after** production. With P2P disabled a *cross-device* stray write would Xid — so the writer rides the victim's own path. |
| 13 | Step-1 only | Non-discriminating: step-1 NaN poisons weights via the allreduce, so later steps are unobservable in failing runs; the ~20% surviving runs' steps ≥2 have not been systematically checked (§6.3 covers this with 12-step runs). |

---

## 4. Dead hypotheses (cumulative, rounds 1–3)

1. **Cross-GPU execution overlap / peer-UVA silent writes** — killed twice over: no
   peer mappings exist (audit), and per-rank-only synchronizers cure while overlap
   persists (§2.1, §2.4).
2. **Host submission concurrency as the mechanism** — whole-backward submission mutex
   fails 3/10. (It *is* a contributing timing window: rate dropped significantly.)
3. **Electrical/power-transient, marginal-silicon** — §2.6.
4. **Timing-dependent broken kernel-variant selection** — newly disfavored by §2.3:
   all 36 layers' proj wgrads share one shape (1280×5120×8192) and hence one
   heuristic/algo choice; a broken variant would NaN all layers or none — observed is
   exactly one layer. (Heuristics-cache-off arm was already dead.)
5. **Lockless-registry lost insert** — §2.7.
6. **Quantize kernel, unwritten regions, encoder NaN, amax/scale sharing, handle/
   registry keying, padding, barrier, keep-alives, per-call-alloc lifetime** — dead in
   rounds 1–3 audits/experiments.
7. **First-execution / lockstep-entry arming (warmup-fix premise)** — unsupported by
   §2.2; the serialized-warmup fix design is withdrawn.

---

## 5. The fix — honestly labeled

**There is no root-cause fix available in this repository.** The application layer is
exonerated (§1); the defect lives in the MAX AsyncRT runtime and/or the NVIDIA
driver/GSP under VFIO. What we can ship is a **window-closure mitigation** that bounds
per-rank async run-ahead at the fp8-backward site granularity, plus upstream reports
(§7).

### 5.1 Primary mitigation (implemented this round): `LLMM_FP8_BWD_SYNC_ONLY`

A `ctx.synchronize()` at the end of every `matmul_bwd_lowp` call — **no lock, no
cross-rank coupling**. It is simultaneously (a) the mechanism discriminator the
exec-mutex probe was missing (§2.8) and (b) the shipping mitigation if it validates.
Applied in tree, comptime-gated, zero effect without the define:

`llmm/matmul.mojo:48-59` (new flag, after `FP8_BWD_EXEC_MUTEX`):

```mojo
# -D LLMM_FP8_BWD_SYNC_ONLY=1: the `ctx.synchronize()` at the end of
# `matmul_bwd_lowp` WITHOUT any cross-rank lock. Bounds this rank's async
# launch run-ahead to one fp8-backward site bundle (~7 kernels) while leaving
# cross-rank submission AND cross-GPU execution fully concurrent. Round-3
# discriminator between "cross-GPU execution overlap corrupts" (predicts this
# still fails) and "per-rank deep-async-queue run-ahead race" (predicts this
# cures, like CUDA_LAUNCH_BLOCKING=1 10/10 and MODULAR_DEBUG=device-sync-mode
# 4/4 — both per-rank-only synchronizers that leave cross-GPU overlap intact).
# If clean, this doubles as the shipping MITIGATION for fp8 WORLD_SIZE>1
# (measured cost of the far heavier sync-mode was ~4%; this syncs only
# 144x/micro-step). It is a window-closure mitigation, not a root-cause fix.
comptime FP8_BWD_SYNC_ONLY = is_defined["LLMM_FP8_BWD_SYNC_ONLY"]()
```

`llmm/matmul.mojo:3685` (end of `matmul_bwd_lowp`; drain now gated on either flag):

```mojo
    comptime if FP8_BWD_EXEC_MUTEX or FP8_BWD_SYNC_ONLY:
        ...
        ctx.synchronize()
```

No preseed needed (no lock cell). Expected cost: **≤ ~4%** at ws2 d36 — the strictly
heavier per-enqueue sync-mode measured 3684–3700 ms vs 3547–3553 ms plain, and CLB
~3.5–3.7 s; this variant syncs only 144×/micro-step (1152/step). Measure exactly in
§6.1. If validated, wire `-D LLMM_FP8_BWD_SYNC_ONLY=1` into the fp8 multi-rank build
recipe (and note it in the run script) until an upstream fix lands; remove it when
upstream resolves.

### 5.2 Fallback mitigations (already proven, heavier)

- `CUDA_VISIBLE_DEVICES=... CUDA_LAUNCH_BLOCKING=1` — proven 10/10 at ws2, ~0–4%
  measured cost at d36 (big-GEMM workload amortizes launch sync). Zero code change;
  pin as `EXTRA_ENV` for fp8 `WORLD_SIZE>1` runs if §6.1 fails.
- `MODULAR_DEBUG=device-sync-mode` — proven 4/4, ~4% cost.

### 5.3 Explicitly NOT shipped

- `LLMM_FP8_BWD_EXEC_MUTEX` — diagnostic only (serializes real GPU work × WORLD_SIZE;
  and a raised `synchronize()` while holding the lock livelocks peers ⇒ 900 s
  timeout, which must be scored *inconclusive*, never clean).
- Serialized warmup micro-step (round-3 design 2a) — premise unsupported, z≥3
  deadlock gap; withdrawn.

---

## 6. Validation protocol

Protocols: every GPU run under `flock -x /tmp/llmm-gpu.lock`, `timeout --signal=INT
900`, never SIGKILL, never touch `GPU-e9c166f6-...8001`; builds under `flock -x
/tmp/llmm-build.lock`. Verdicts from `DEBUG rank R local grad norm pre-reduce:` lines
(`_dbg` binaries). Bit-identity is a valid criterion for plain/sync arms only (§2.4).
Score any 900 s timeout or rc≠0 as **inconclusive → rerun**, not clean.

### 6.0 Builds
The ws2 binary is already built and compile-verified this round
(`build/train_gpt2_fp8_w2_synconly_dbg`, BUILD_OK); only the ws7 build remains.
```
pixi -q run mojo build -D WORLD_SIZE=2 -D LLMM_PRECISION=fp8 -D LLMM_FP8_BWD_SYNC_ONLY=1 \
  -D LLMM_DEBUG_LOCAL_GRADNORM=1 -I . -Xlinker -lm -o build/train_gpt2_fp8_w2_synconly_dbg train_gpt2.mojo
pixi -q run mojo build -D WORLD_SIZE=7 -D LLMM_PRECISION=fp8 -D LLMM_FP8_BWD_SYNC_ONLY=1 \
  -D LLMM_DEBUG_LOCAL_GRADNORM=1 -I . -Xlinker -lm -o build/train_gpt2_fp8_w7_synconly_dbg train_gpt2.mojo
```

### 6.1 Stage 1 — ws2 × 10 (mechanism gate + cost measurement)
Standard template, first two healthy UUIDs, `-e d36 -pn 2 -d 131072 -x 1`,
`BIN=build/train_gpt2_fp8_w2_synconly_dbg`.
**Pass:** 10/10 clean (p ≈ 1e-7 vs 80% baseline), norms bit-identical to plain clean
values (r0 19.617536 / r1 19.492779 — the sync changes no arithmetic), step-time delta
vs plain ≤ ~5% recorded.
**Branch on failure:** any NaN ⇒ per-rank queue-depth is *not* sufficient ⇒ run
`build/train_gpt2_fp8_w2_execmutex_dbg` × 10: clean there ⇒ cross-rank execution
exclusion genuinely operative (original exec-level reading revives); dirty there too ⇒
escalate to whole-micro-step serialization and hardware/VFIO attribution rises —
either way rerun §6.5's two-process discriminator before further code work.

### 6.2 Stage 2 — ws7 × 6
All seven healthy UUIDs in rank order, `-e d36 -pn 7 -d 458752 -x 1`,
`BIN=build/train_gpt2_fp8_w7_synconly_dbg`. **Pass:** 6/6 clean, all 42 pre-reduce
lines finite (historical ws7 failures showed multi-victim sets, e.g. {0,3,4,5}).

### 6.3 Stage 3 — standing gate: 3 × ws7 12-step
Same as 6.2 with `-x 12`. **Pass:** all three runs rc=0, zero NaN in every per-step
pre-reduce line, finite losses throughout. This also closes the step-1-only blind spot
(gate #13): 36 rank-steps × 12 with the mitigation active.

### 6.4 Parallel, non-blocking — SUSPECT gate retest
d12 ws7 "clean" is n=1: rerun × 5 with `build/train_gpt2_fp8_w7_dbg`
(`-e d12 -pn 7 -d 458752 -l 0.0006 -x 1`, no mitigation flag). Any failure here
re-opens the scale story (§2.5) but does not block shipping the mitigation.

### 6.5 Attribution follow-ups (for the upstream reports, not for shipping)
1. **Two-OS-process ws2** (one GPU per process, no shared process state): clean ⇒
   shared AsyncRT machinery (A) confirmed, driver largely exonerated; dirty ⇒
   driver/GSP-under-VFIO (B) becomes primary. Needs a small harness (single-rank
   binary × 2 + file/socket barrier + host-side grad reduce, or simply two
   independent ws1 processes running concurrently on both GPUs with the plain binary
   and the NaN check — corruption per §1 does not require the collective).
2. **Pointer + checksum probe** on `proj_input`/WT/IT for L27/L3: log forward-write
   vs backward-read pointer and a device-side NaN-count immediately before wgrad
   consumption in one failing plain-binary run — "same pointer, changed bytes" proves
   post-write corruption and pinpoints operand-stash vs GEMM-output; also separates
   spatial (fixed address) from temporal (fixed window) via per-layer timestamps.

### 6.6 Honest residual risks
- bf16 multi-rank is proven NaN-free, not corruption-free (gate #4). The 12-step ws7
  bf16 loss curves being historically sane is the practical backstop; a one-off bf16
  ws2 bitwise-vs-ws1 grad comparison would close this if desired.
- The C=1280 shape confound (gate #10) is untested; irrelevant to the mitigation but
  worth one line in the NVIDIA report.
- If §6.1 passes, sub-attribution (A vs B) remains open until §6.5.1 runs; the
  mitigation is valid either way.

---

## 7. Upstream filings

File **both**, cross-referenced; lead with Modular. **Correct the round-3 draft
skeleton before filing** — it contains two now-refuted claims: delete "peer access +
UVA mappings enabled across all 7 devices" (nothing enables peer access;
`can_access == False`; no cross-device mappings exist) and reframe the title away from
"concurrent FP8 GEMMs across multiple GPUs corrupt" (disfavored by §2.1/§2.4).

**Modular (primary).** Title: "Intermittent single-rank NaN in fp8 (e4m3×e5m2)
cuBLASLt backward under multi-DeviceContext single-process training; cured by any
per-rank launch synchronization". Body: environment (QEMU/KVM guest, Linux
6.8.0-136-generic, 8× RTX PRO 6000 Blackwell Max-Q via VFIO, driver 610.43.02, Mojo
1.0.0b3.dev2026071306 (1f3b9e33)); one host thread + DeviceContext per GPU in one
process; symptom (step-1, exactly one victim rank at ws2, victim varies, survivors
bit-identical, never at ws1, ~774M scale only); localization (always the same wgrad
GEMM class, layer-27-preferring, single slice, dgrad chain finite); evidence ladder —
CUDA_LAUNCH_BLOCKING=1 10/10 clean, MODULAR_DEBUG=device-sync-mode 4/4 clean, **both
per-rank-only**; app-level whole-backward submission mutex fails but significantly
reduces the rate; 120 ms stagger fails; app layer audited clean (per-ctx buffers/
handles/streams, no peer access, saturating encoder). Questions: are launch
staging/argument buffers, event/descriptor pools, and per-context queue workers
thread-isolated per DeviceContext? Known issues with N contexts in one process under
deep async queues? [Attach §6.1 result and, when available, §6.5 outcomes.]

**NVIDIA (secondary; promote to primary if the two-process run is dirty).** Title:
"Possible deep-async-launch-queue corruption in QEMU/KVM VFIO guest (GSP, sm_120):
intermittent NaN in fp8 cuBLASLt GEMM operands under multi-GPU single-process load".
Add: one physical card in the box already faults in GSP firmware (excluded from runs);
ask about known VFIO/large-BAR/doorbell issues with deep launch queues, GSP
interactions, and recommended isolation given compute-sanitizer racecheck is
intra-device only. Include the shape note (only C=1280 model sizes affected so far).

---

## 8. Working-tree state / bookkeeping

Uncommitted, all comptime-gated no-ops without their defines: persistent-scratch fp8
backward + `FP8_BWD_MUTEX`/`FP8_BWD_EXEC_MUTEX`/`FP8_BWD_SYNC_ONLY` in
`llmm/matmul.mojo`; nanscan probe (+276 lines) and `LLMM_DEBUG_RANK_STAGGER` in
`train_gpt2.mojo`. After §6 passes: commit the persistent-scratch backward and the
three diagnostic flags (they are the reproduction/validation kit for the upstream
reports), keep nanscan gated, and gate-check per the merge rules
(format/lint/check/test green on CUDA) before merging to main. Binaries present:
`_w2_dbg`, `_w2_stagger_dbg`, `_w2_nanscan`, `_w2_fwdonly_dbg`, `_w2_execmutex_dbg`,
`_w7_dbg`, `_w1_control` (+ the two §6.0 builds to make).

Round-3 logs (scratchpad, session ece34132…): `clb_ws2_logs/`, `stagger10/`,
`nanscan/`, `logs/{syncmode,fwdonly,d24,d12}_run*.log`.
