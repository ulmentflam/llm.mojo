> **Superseded by `fp8_multirank_nan_investigation.md`, kept as historical record.**

# FP8 multi-rank NaN hunt — fleet synthesis (2026-07-22)

Bug: `-D LLMM_PRECISION=fp8`, GPT-2 d36, multi-rank single-process (7 host
threads, one `DeviceContext` per rank): grad-norm NaN from step 1; forward
loss finite 1-2 steps, then NaN.

## 0. Evidence gates (established; every mechanism must fit all of them)

| Gate | Config | Result |
|---|---|---|
| G-ws1 | ws=1 d36 fp8, any accum, all site variants | CLEAN |
| G-d12 | ws=7 d12 fp8, full epoch | CLEAN |
| G-sync | failing config + `MODULAR_DEBUG=device-sync-mode` | CLEAN |
| G-fwd | `LLMM_FP8_FWD_ONLY=1` (bf16 backward), ws=7 d36 | CLEAN |
| G-bf16 | bf16 d36 ws=7, 18 h | CLEAN |
| G-z | `-z 0` accum 8 AND `-z 1` accum 1 | BOTH FAIL |

Already refuted before this round: adding G2-style keep-alives for
`amax_doutput` (matmul.mojo ~3534) and `partial_max`/`partial_bad`
(amax.mojo ~287) — keep-alives are IN the tree, NaN persists 3/3.
Note the precise wording of that refutation: it kills "keep-alive missing"
as the mechanism; it does NOT kill "release of those buffers is unsafe even
WITH a keep-alive" (a keep-alive only moves the release point to function
end; the release still happens while the consumer kernels are pending —
see M1).

Scout status at time of writing: "Monitoring the E1 build. Will launch the
ws2 run as soon as the binary lands." — i.e. **no new empirical results
landed this round**; the ranking below is code-derived and pre-registers how
the pending ws2 result updates it (§4).

## 1. What this round verified in the tree (audit trail)

- `train_gpt2.mojo:3966` — `self.ctx.synchronize()` runs after backward,
  before any collective. `zero.mojo` allreduce is fully bulk-synchronous:
  `_register_and_sync` barrier1 → phase-1 copies+adds → `ctx.synchronize()`
  (zero.mojo:455) → barrier2 → phase-2 copies → `ctx.synchronize()` (:481)
  → barrier1 (:482). Every peer read is provably ordered after every peer's
  device is idle; pointer slots cannot be overwritten early (exit barrier).
  Consistent with G-bf16. **Collective ordering is exonerated.**
- `_get_global_handle` (linalg.matmul.vendor.blas): key on current `main` is
  `"LINALG_VENDOR_BLAS_{backend}_{ctx.id()}"` — **per-device**. Each rank
  gets its own cuBLASLt handle; no cross-rank handle sharing. Consistent
  with bf16-clean. (Open question 4: confirm the pinned nightly matches.)
- `persistent_device_buffer` (llmm/memory.mojo:86) keys `name + ctx.id()` —
  per-rank. bf16-clean (which exercises `CUBLASLT_WS`, dbias, LN globals at
  ws=7) says `ctx.id()` really is distinct per rank.
- Registry insert traffic (`KGEN_CompilerRT_InsertGlobal`) is **identical**
  between `FP8_FWD_ONLY` and full fp8: the backward inserts no new names —
  `matmul_fwd_lowp` creates all 288 WT/IT caches either way, dgrad/wgrad
  only look them up. So G-fwd kills any registry-race mechanism outright.
- Per-rank device work (quantize, amax, update_scale, fp8 GEMMs, dbias) is
  all enqueued on the rank's single stream; intra-rank enqueue order is
  identical at ws=1 and ws=7. Therefore the failure needs either
  (a) a cross-rank data channel — all exonerated above — or
  (b) host-side behavior that changes under 7-thread concurrency:
  **the runtime's buffer release/alloc path**. That is where M1 lands.
- `lowp.mojo` quantize entry points allocate no transients; the only
  per-call `DeviceBuffer`s in the fp8 backward are `amax_doutput`,
  `partial_max`/`partial_bad` (inside `compute_amax`), and
  `doutput_fp8_nat`/`doutput_fp8_t` (matmul.mojo:3527, 3547-3551), all with
  keep-alives but all still **freed at scope end while their consumer GEMMs
  /kernels are still pending on the stream**.

## 2. Ranked mechanisms

### M1 (TOP) — non-stream-ordered release + reuse of fp8-backward per-call
### DeviceBuffers under multi-thread driver contention

`DeviceBuffer.__del__` *claims* stream-ordered release, but this toolchain
has already been caught twice not honoring that claim end-to-end:
gotcha G2 (a real, reproduced recycled-while-pending race on exactly the
`doutput_fp8_nat/t` dual-quantize buffers, at ws=1) and G3 (the same
signature reappearing on unmodified code under GPU contention). The
keep-alive "fixes" only delay the host-side release to function return —
the release is still issued while the dgrad/wgrad GEMMs (3 call levels of
enqueues deep) are pending. If, under 7-thread driver/runtime contention,
the release path degrades to a non-stream-ordered free (or the pool hands
the block out again before the pending consumer retires), the next
allocation of the same size class (next site/layer's transients — same
sizes every layer) recycles the block and its producer kernel overwrites
the fp8 operand bytes / the amax scalar before the pending GEMM reads them
→ garbage-scaled GEMM output → Inf/NaN in `d_weight`/`d_input` on some
rank at step 1 → grad-norm NaN; params absorb the NaN at `update` → loss
NaN 1-2 steps later. Matches the observed temporal signature exactly.

Gate fit:
- **G-ws1 clean**: single-threaded release path is well-behaved (this is
  the regime where the G2 keep-alive fix was validated bit-identical 3×);
  the unsafe path needs cross-thread contention in the runtime/driver
  (G3's own lesson: contention widens this exact class of window).
- **G-d12 clean**: 124M model leaves the per-GPU pool uncontended and
  under-committed — freed blocks are rarely recycled before the pending
  consumer retires; d36 (774M: 3× layers, 1280-wide, WT/IT caches, master
  + moments) puts the pool under pressure so freed transients are reused
  promptly. Both the trial count and the reuse probability scale with
  depth.
- **G-sync clean**: forced-synchronous launches retire every consumer
  kernel before the host reaches the release → window closed by
  construction. This gate is the strongest single discriminator and it
  points squarely at an async release/consume race.
- **G-fwd clean**: the racing buffers (`doutput_fp8_nat/t`, `amax_doutput`,
  backward `compute_amax` partials) exist only in the fp8 backward; the
  bf16 backward allocates no per-call device transients. (Forward's
  `a/b_scratch` are created and consumed inside one function — the shallow
  topology G2 explicitly found safe; the failing buffers have G2's exact
  3-deep sibling-consumer topology.)
- **G-bf16 clean**: no fp8 transients at all.
- **G-z both fail**: the race is inside backward compute, upstream of and
  independent from the z0/z1 collective choice; accum 1 still gives
  144 windows/step at d36.
- **Refuted keep-alives**: consistent — keep-alives cannot fix a release
  that is unsafe *wherever* it lands while consumers are pending; only
  eliminating the per-call free can.

### M2 — process-wide (not per-device) thread-unsafety in the runtime's
### buffer pool itself

Variant of M1 where the defect is on the *allocation* side: 7 threads
hammering `enqueue_create_buffer`/release concurrently corrupt or misroute
pool metadata. Same gate fit as M1 (contention-gated, async-gated,
churn-rate-gated), same mitigation (remove per-call alloc/free from the
fp8 path). Ranked below M1 only because M1 needs no new defect — it reuses
a failure mode this repo has already reproduced twice (G2/G3).

### M3 (low) — cuBLASLt e4m3×e5m2 dispatch concurrency

The e4m3×e5m2 dgrad/wgrad GEMMs are the only cuBLASLt shape/type family
that fwd-only never runs. Handles are per-device (verified), cuBLASLt is
documented thread-safe, and bf16 + fp8-forward hammer the same library
concurrently and stay clean — so this needs a driver-level fp8-specific
defect. Keep only as a fallback experiment: serialize
`_lt_pick_algo`+`cublasLtMatmul` behind a process mutex (or pin one algo)
and rerun the failing config; clean ⇒ revisit.

## 3. Refuted this round (with reasons)

- **R1 — registry/`InsertGlobal` insert race**: killed by G-fwd — the
  insert set is byte-identical between clean fwd-only and failing full-fp8
  (backward inserts no new names; verified matmul.mojo:3338/3418 only look
  up caches forward already created).
- **R2 — shared `_get_global_handle` cross-device cuBLASLt handle**: key
  includes `ctx.id()` (main-branch source, quoted in §1) → per-device.
  Also incompatible with G-bf16/G-fwd even if shared.
- **R3 — e5m2 Inf via delayed-scale overshoot (numerics)**: any pure
  numerics mechanism is rank-count- and sync-mode-independent; killed by
  G-ws1 + G-sync + FF2/FF3 bit-stability results. `compute_amax`'s
  NaN-infection + `scale=1.0` fallback additionally blocks NaN entering via
  the scale path — corruption must enter via operand/output buffers, which
  supports M1.
- **R4 — collective ordering / barrier generation defect**: killed by
  G-bf16 plus this round's structural audit (train:3966 sync; zero.mojo
  455/456/481/482 sync-then-barrier at every phase edge; exit barrier
  protects slot reuse).
- **(pre-round) R0 — missing keep-alives on `amax_doutput` /
  `partial_max`/`partial_bad`**: in tree, NaN persists 3/3.

## 4. Integrating the scout + pre-registered updates

No empirical results landed (E1 build still in flight; ws2 run queued).
Pre-registration so the next round can apply results ruthlessly:

- **ws2 d36 fp8 FAILS** → contention threshold is low; M1 unchanged (it
  needs ≥2 threads, not 7). If ws2 fails *deterministically at step 1 every
  run*, that is evidence of something structural rather than a
  probabilistic window — re-examine per-rank keying end-to-end (dump every
  registry name + `ctx.id()` per rank at startup) before trusting M1.
- **ws2 CLEAN, higher ws fails** → contention-scaling confirmed; M1/M2
  strengthened.
- **Patched binary (§5) fails the standing gate** → M1/M2 both dead for
  the patched buffers; escalate to M3's mutex experiment and to
  per-site bisection (`LLMM_FP8_SITE_*=0` one at a time at ws=7 d36) plus
  a step-1 per-tensor NaN dump (`dump_grads`-style) to localize the first
  poisoned tensor.

## 5. Minimal patch for M1 (also mitigates M2)

Principle: **eliminate every per-call device alloc/free from the fp8
backward hot path** — the same persistent-buffer design (never freed, one
per rank, single-stream write-before-read) that the WT/IT caches already
use and that every clean gate exercises. Numerics are bit-identical; only
buffer lifetime changes. Lifetimes are intra-call (d_output quantize and
amax are produced and consumed inside one `matmul_bwd_lowp` invocation, on
one stream), so per-site keys (layer-independent — all layers share
shapes) suffice; memory cost ≈ 2×rows×ΣOC ≈ 190 MB/rank at rows=8192,
far below the per-(site,layer) WT/IT caches already shipped.

### 5.1 `llmm/matmul.mojo` — `matmul_bwd_lowp` (~3526-3561)

Replace the three per-call buffers:

```mojo
    comptime if not FP8_STATIC_SCALES:
        # Persistent (never-freed) amax scratch: was a per-call DeviceBuffer
        # whose release landed while _update_scale_gpu was still pending —
        # the suspected M1 window. Intra-call lifetime, single stream, so
        # one per-rank cell is safe to reuse across all sites/layers.
        var amax_doutput = persistent_device_buffer[DType.float32](
            ctx, "FP8_AMAX_DOUT", 1
        )
        compute_amax[FP8_SPEC, dtype](
            amax_doutput, d_output_ptr, rows * out_channels, ctx
        )
        doutput_state.update_scale[FP8_SPEC.bwd_dtype](
            kernel_ptr_as_immut(amax_doutput), ctx
        )

    # Persistent per-site d_output fp8 staging (was per-call DeviceBuffers —
    # G2's original race site; keep-alives only delayed the release, they
    # did not remove it). Site-keyed: every layer shares the site's shape,
    # and the lifetime is intra-call (quantize -> dgrad/wgrad reads, one
    # stream), so layers may safely reuse the same buffer.
    var doutput_fp8_nat = persistent_device_buffer[DType.uint8](
        ctx, String("FP8_DN_") + String(site), rows * out_channels
    )
    var doutput_fp8_t = persistent_device_buffer[DType.uint8](
        ctx, String("FP8_DT_") + String(site), out_channels * rows
    )
    quantize_dual_devscale[FP8_SPEC, FP8_SPEC.bwd_dtype, dtype, target](
        doutput_fp8_nat,
        doutput_fp8_t,
        d_output_ptr,
        kernel_ptr_as_immut(device_buf_mut_ptr(doutput_state.scale)),
        rows,
        out_channels,
        ctx,
    )
```

Pass `doutput_fp8_nat` / `doutput_fp8_t` (already `MutKernelPtr[uint8]`)
straight into `matmul_d_input_bwd_lowp` / `matmul_d_weight_bwd_lowp` (drop
the `device_buf_mut_ptr(...)` wrappers at :3566/:3582), and delete the
now-moot keep-alives at :3545 and :3606-3607.

### 5.2 `llmm/amax.mojo` — `compute_amax` (~256-257, 287-288)

```mojo
        # Persistent partial buffers, sized once to the device-wide grid
        # ceiling (max_grid is invariant per device); kernels only touch
        # [0:grid_size]. Removes the last per-call alloc/free in the fp8
        # backward (and, harmlessly, the forward).
        var partial_max = persistent_device_buffer[DType.float32](
            ctx, "AMAX_PARTIAL_MAX", max_grid
        )
        var partial_bad = persistent_device_buffer[DType.float32](
            ctx, "AMAX_PARTIAL_BAD", max_grid
        )
```

(Add `persistent_device_buffer` to the `llmm.memory` import at
amax.mojo:55; adapt `device_buf_mut_ptr(partial_max)` call sites to use the
pointer directly; delete the keep-alives at :287-288.)

Rationale check for `count=max_grid`: `persistent_device_buffer` raises if
a later call requests more than the first call allocated; `max_grid` is the
per-device fixed upper bound of `grid_size`, so passing it on every call
satisfies the "count upper-bounds all sites" contract.

### 5.3 Why each gate stays satisfied post-patch

- Per-rank keys (`name + ctx.id()`), same single stream, producer enqueued
  before consumer every call ⇒ no new sharing at any ws.
- Sizes site-keyed and shape-invariant for a fixed training config ⇒ no
  `count >` raise.
- Bit-identical numerics ⇒ ws=1 twin-run bit-identity gate (G2 protocol)
  must still pass — run it as part of validation.
- If NaN persists post-patch, M1 is refuted **for the backward transients**;
  the forward's `a/b_scratch`/`amax_input`/`amax_weight` get the identical
  treatment as the follow-up arm before abandoning M1 entirely (G-fwd only
  proves removing the backward's fp8 work suffices, not that forward
  transients are individually safe in the full-fp8 process).

## 6. Validation protocol (standing gate)

1. Build fp8 d36 pn7 binary with §5 applied (sequential make targets — T5).
2. **Primary**: 3× 12-step runs, `-z 0`, accum 8, d36, ws=7, async (no
   `MODULAR_DEBUG`), GPUs pinned by UUID (per memory: never SIGKILL
   multi-rank runs; new runs write under /data), under
   `flock /tmp/llmm-gpu.lock`. PASS = zero NaN/Inf in loss and grad-norm in
   all 36 step-lines.
3. **d12 control**: 1× 12-step ws=7 d12 fp8 run — must stay clean (guards
   against the patch introducing a regression the big config can't see).
4. **Bit-identity control**: ws=1 d36 fp8 10-step twin run ×2 — must be
   bit-identical to each other (G2 gate; per G3, if it wiggles, run the
   pre-patch binary as the contention control before blaming the patch).
5. Merge gates per memory: format/lint/check/test green on CUDA.

## 7. Open questions

1. **What does `DeviceBuffer.__del__` actually do on this pinned nightly**
   (1.0.0b3.dev2026062706)? Inspect `std.gpu.host` source / disassemble
   `_hal.mojoc`: cuMemFreeAsync-on-owning-stream vs host-side free decides
   between M1 (release side) and M2 (pool side). This is the single
   cheapest way to convert the top mechanism from inference to fact.
2. **ws2 scout result (E1)** — apply §4 pre-registration when it lands.
3. Why the forward's per-call transients never fire at ws=7 d36 under
   fwd-only — call-depth topology (G2's own finding) is the working
   explanation; falsifiable via the §5.3 follow-up arm.
4. Confirm the pinned toolchain's `_get_global_handle` key includes
   `ctx.id()` like current `main` (probe: print handle pointer per rank at
   startup; distinct values expected).
5. If the patch validates, decide whether to also persist the forward
   scratch pair as hardening (removes the last enqueue_create_buffer churn
   from the training hot loop; also a small perf win — allocation overhead
   is nonzero at 4 sites × 36 layers × accum 8).
