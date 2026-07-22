# fp8 multi-rank backward race — root cause analysis (2026-07-22)

**Symptom.** fp8 build (`-D LLMM_PRECISION=fp8`), GPT-2 d36, single-process
multi-rank (one thread + one `DeviceContext` per rank): grad norm NaN from
step 1; loss (forward) finite. Repro matrix (established before this
analysis, not re-derived here): world_size=1 clean at any accum;
world_size=7 NaN for both `-z 0` (accum 8) and `-z 1` (accum 1); d12 fp8
W7 clean; bf16 d36 W7 clean (18 h); `LLMM_FP8_FWD_ONLY=1` W7 clean;
`MODULAR_DEBUG=device-sync-mode` makes the failing config clean (race, not
numerics).

## Audit scope and exonerated machinery

Everything the fp8 backward enqueues was traced to a launch mechanism:

- All quantize / amax / scale-update kernels go through
  `ctx.compile_function` + `ctx.enqueue_function` on the rank's own `ctx`
  (`llmm/lowp.mojo:573,701,821`, `llmm/amax.mojo:261,273,531,642`).
- Both fp8 GEMMs (`_matmul_cublaslt_fp8`, `llmm/matmul.mojo:605-744`) pass
  `CUDA(ctx.stream())` explicitly (line 642/740) — same stream discipline
  as the bf16 `_matmul_cublaslt` (line 382) that the 18 h bf16 W7 run
  proved sound end-to-end, including the collective machinery.
- The collective itself (`ZeroContext.allreduce`, `llmm/zero.mojo:373-487`)
  is ordered by: per-rank `ctx.synchronize()` after backward
  (`train_gpt2.mojo:3965-3966`) → generational `CpuBarrier`
  (`zero.mojo:40-66`, checked: lock-protected counter + generation spin,
  correct under the barrier1/barrier2/barrier1 reuse pattern) → phase-1
  peer reads. The barrier is NOT the bug.
- `persistent_device_buffer` keys by `(name, ctx.id())`
  (`llmm/memory.mojo:69-111`) — per-device; the transpose caches and
  cuBLASLt workspace cannot collide across ranks.

So every kernel/copy the fp8 backward enqueues targets the rank's own
in-order stream, and that stream is drained before any peer's phase-1 read.
What is NOT protected is **`DeviceBuffer` lifetime**: the fp8 backward is
the only per-step CUDA code path that creates and destroys per-call
`DeviceBuffer`s at all — the bf16 CUDA backward allocates nothing per call
(persistent cuBLASLt workspace, persistent dbias scratch/counters,
persistent attention GEMM scratch).

## Root cause: G2-class destroy-before-consumer-enqueue in the fp8 backward

Repo gotcha G2 (`docs/ai/low_precision_gotchas.md:607-663`) documents — from
a real, reproduced, nondeterministic race in this exact function — that
Mojo's ASAP destroy-at-last-use drops a `DeviceBuffer` whose last remaining
reference is a borrow inside a call's argument list *before the consuming
call body runs*, and that the runtime's stream-ordered-release guarantee is
not sufficient at that call depth. The fix pattern is an explicit keep-alive
(`_ = buf.unsafe_ptr()`) after every consumer's enqueue. That fix was
applied to `doutput_fp8_nat`/`doutput_fp8_t`
(`llmm/matmul.mojo:3595-3607`) — but two instances in the same backward
call chain were missed:

1. **`amax_doutput` in `matmul_bwd_lowp`** (`llmm/matmul.mojo:3536`,
   pre-patch). Its last use is the borrow inside
   `device_buf_mut_ptr(amax_doutput)` in `update_scale`'s argument list
   (line 3544). That borrow ends when the helper returns — BEFORE
   `update_scale`'s body (`llmm/amax.mojo:507-543`) enqueues
   `_update_scale_gpu`, the kernel that READS the buffer. The buffer's
   release is therefore issued between the amax-aggregate kernel (its
   writer) and the scale-update kernel (its reader):

   `[aggregate writes A] [release A] [_update_scale_gpu reads A] ...`

2. **`partial_max`/`partial_bad` in `compute_amax`**
   (`llmm/amax.mojo:256-257`, pre-patch). Last use is the borrow in the
   aggregate launch's argument list (lines 276-277) — the release can be
   issued before `ctx.enqueue_function` for the aggregate kernel even runs.
   This is the sharpest possible instance of the G2 mechanism
   ("destroyed before the SECOND call even runs").

### Why this NaNs, and why only at world_size > 1 / d36

- **NaN amplification is backward-specific.** A corrupted `amax_current`
  read feeds `_scale_from_history` (`llmm/amax.mojo:350-372`): during
  warmup (`step < amax_history_len=16`, i.e. from step 1) the scale is
  derived DIRECTLY from `amax_current`. A garbage tiny-positive float gives
  `scale = fmt_max * margin / amax = Inf`; the E5M2 quantize then computes
  `0.0 * Inf = NaN`, and **E5M2 has NaN encodings** (E4M3 saturates), so
  the wgrad GEMM (`beta=1` accumulate straight into `grads_memory`)
  poisons the gradient while the forward stays finite — exactly the
  observed signature (finite loss, NaN grad norm from step 1). Other
  garbage values give finite-but-wrong scales that the symmetric
  `scale * scale_inv` round trip mostly cancels — which is why the same
  hole in the forward (`amax_input`/`amax_weight` in `matmul_fwd_lowp`,
  same borrow shape at lines 1782-1788) can be empirically clean
  (`LLMM_FP8_FWD_ONLY=1` W7 clean): E4M3 has no NaN encoding and forward
  mis-scales self-cancel.
- **Single rank survives by in-order luck.** With one thread per process,
  a prematurely released allocation can only be recycled by the SAME
  stream's later allocations, whose writers are enqueued after the stale
  reader — in-order execution keeps the read value-correct. G2's own
  history shows the failure needs allocator churn + scale to surface.
- **world_size > 1 provides the missing out-of-order agents.** Seven rank
  threads share one process: one runtime allocator and global registry,
  concurrent per-layer allocation/release traffic on seven streams, and —
  on the last micro-step — the collective window, where the post-backward
  `ctx.synchronize()` (`train_gpt2.mojo:3966`) is the first drain point
  for the whole backward's released-too-early allocations while the
  driver-staged cross-device phase-1 copies begin touching every device
  outside any compute-stream ordering. A 4-byte amax cell whose release
  was issued before its reader is exactly the allocation most likely to be
  recycled in that window.
- **d36 vs d12 is a window-width effect** (timing plausibility, per the
  task brief): 3x the layers means 3x the per-micro-step
  create/release pairs (5 per site-layer: amax scalar, 2 amax partials,
  nat, t) and a much deeper pending queue behind each release —
  d36 ≈ 720 premature-release windows per micro-step vs d12 ≈ 240.
- **device-sync-mode closes the race** definitionally: every enqueue
  completes before the host reaches the destroy point, so no release can
  precede its consumer's execution.

**Confidence.** That the two buffers violate the repo's own G2 rule (release
issued before the consumer kernel's enqueue) is code-provable and is the
only lifetime hole in the implicated path — HIGH. That this is the whole
multi-rank NaN story (i.e. the cross-rank allocator/collective interaction
is what converts the ws=1-benign stale read into corruption) is the
best-supported mechanism found but is not directly observable from source —
MEDIUM; the validation plan below is designed to confirm or falsify it.

## Patch (applied to working tree, uncommitted)

Established G2 keep-alive pattern, no syncs, no behavior change on the
happy path:

- `llmm/matmul.mojo` (`matmul_bwd_lowp`): `_ = amax_doutput.unsafe_ptr()`
  after `update_scale` returns, inside the `not FP8_STATIC_SCALES` block —
  moves the release after the `_update_scale_gpu` enqueue in stream order.
- `llmm/amax.mojo` (`compute_amax`): `_ = partial_max.unsafe_ptr()` /
  `_ = partial_bad.unsafe_ptr()` after the aggregate `enqueue_function` —
  moves both releases after the aggregate read.

Known same-shape instances left unpatched (deliberately, to keep the diff
minimal and scoped to the implicated backward path; the FWD_ONLY W7 run
empirically exonerates them for this failure): `matmul_fwd_lowp`'s
`amax_input`/`amax_weight` (consumed via `update_scale_pair` args,
`llmm/matmul.mojo:1782-1788`) and `a_scratch`/`b_scratch` (consumed via
`lowp_gemm_devscale` args, lines 1841-1842). If the validation below is
green, applying the same one-line keep-alives there is recommended hygiene.

## Validation plan (NOT run — GPUs are occupied by production training)

Build once, then 3 runs for NaN absence + 2 additional runs for
bit-identity on the failing config, and the same protocol on the d12
control. Pin GPUs by UUID (GPU 1 is the faulted/renumbered one on this
box — see memory note) and never SIGKILL the multi-rank process.

```bash
make build-fp8            # mojo build -D WORLD_SIZE=... -D LLMM_PRECISION=fp8 -o build/train_gpt2_fp8

# Failing config: pn7, z0, accum 8 (total batch = 8 * 7 * B * T with B=4 T=1024), d36, 12 steps.
# Adjust -i/-j to the FineWeb shards on /data and CUDA_VISIBLE_DEVICES to the 7 good UUIDs.
RUN="CUDA_VISIBLE_DEVICES=<7-good-GPU-UUIDs> ./build/train_gpt2_fp8 \
  -e d36 -b 4 -t 1024 -d 229376 -z 0 -pn 7 -x 12 -v 0 -s 0 \
  -i '/data/fineweb10B/fineweb_train_*.bin' -j '/data/fineweb10B/fineweb_val_*.bin'"

# 3x NaN-absence (async, background; check every step line for finite grad norm):
for i in 1 2 3; do eval "$RUN" 2>&1 | tee /tmp/fp8race_d36_run$i.log; done
grep -il "nan" /tmp/fp8race_d36_run*.log && echo "FAIL: NaN present" || echo "PASS: no NaN"

# 2x bit-identity (per G2/G3 protocol: compare the per-step loss/grad-norm lines;
# run a pre-patch control if they diverge before blaming the diff):
diff <(grep -E "^step" /tmp/fp8race_d36_run1.log) <(grep -E "^step" /tmp/fp8race_d36_run2.log) \
  && echo "PASS: bit-identical" || echo "CHECK: diverged (run 3-way + unpatched control per G3)"

# d12 control (same protocol, accum 8: -d 229376 with the same B/T):
RUN_D12="CUDA_VISIBLE_DEVICES=<7-good-GPU-UUIDs> ./build/train_gpt2_fp8 \
  -e d12 -b 4 -t 1024 -d 229376 -z 0 -pn 7 -x 12 -v 0 -s 0 \
  -i '/data/fineweb10B/fineweb_train_*.bin' -j '/data/fineweb10B/fineweb_val_*.bin'"
for i in 1 2 3; do eval "$RUN_D12" 2>&1 | tee /tmp/fp8race_d12_run$i.log; done
# plus the same 2-run bit-identity diff on the d12 logs.

# Also re-run the z1 accum-1 variant once (-z 1 -d 28672) for NaN absence.
```

Expected: 3/3 finite grad norms on d36 W7 (was 0/3 pre-patch), d12 W7 still
clean, and run-to-run bit-identity (modulo the pre-existing
`ln_dparam_accum` atomic jitter documented in G3 — control against an
unpatched binary before attributing any divergence to this patch).

## noqkv pre-step-1 crash (`-D LLMM_FP8_SITE_QKV=0`, WORLD_SIZE=1)

Static findings — the obvious suspects are clean:

- **Gate coupling is correct.** Forward (`train_gpt2.mojo:2697`) and
  backward (`:3785`) both gate the QKV site on the same `FP8_SITE_QKV`, so
  the "transpose cache + AmaxState only valid if the site's forward ran"
  invariant (`:207-217`) is not violated: with QKV off, nothing ever reads
  `FP8_WT_qkv_*`/`FP8_IT_qkv_*` and the qkv AmaxStates are allocated
  (unconditionally, `Fp8State.__init__` `:867-879`) but never touched.
  No Fp8State indexing shift exists — all four sites' lists are always
  populated to `num_layer`.
- **The bf16 QKV branch's arguments/shapes check out** (`:3806-3819`):
  13 args matching `matmul_bwd`'s signature; `pre_gelu=NULL` is dead under
  `use_gelu=False`; the `scratch` arg (`d_l_fch_gelu`) is never touched on
  the GPU cuBLASLt path (`matmul_d_weight_bwd` `:2963-2980`, and the
  allocation-time comment `:2248-2267` relies on exactly that).
- The only build-specific novelty is that the bf16
  `matmul_d_input_bwd[use_gelu=False]` / `matmul_d_weight_bwd[accumulate=True]`
  instantiations now execute interleaved between fp8 sites in the same
  backward — code identical to every bf16 build, where it is proven.

Diagnosis: no static shape/ordering/gating defect exists in the site-gate
plumbing itself. "Crashes before step 1 with no output" — including the
unconditional allocation banner prints — is the signature of a hard fault
(SIGSEGV/SIGABRT) with *buffered stdout lost*, which places the crash
anywhere inside step 1, not necessarily before it. Given the main finding,
the most plausible mechanism is the SAME G2 lifetime class rather than an
independent bug: disabling one site changes the per-layer allocator churn
pattern (the qkv site's 5 create/release pairs vanish while attn_proj/fc/
proj still churn), which re-times which premature release gets recycled —
`compute_amax`'s partial buffers dying before their aggregate enqueue
(patched here) are exercised identically in that build. Recommended
follow-up when GPUs free up: re-run the noqkv config with the patch,
`stdbuf -o0 -e0`, and stderr captured — if it still crashes, capture the
CUDA error string; it is then an independent bug and the next suspect is
the first-execution path of the bf16-in-fp8-build backward instantiations,
not the gates.

## Files touched

- `llmm/matmul.mojo` — keep-alive for `amax_doutput` in `matmul_bwd_lowp`.
- `llmm/amax.mojo` — keep-alives for `partial_max`/`partial_bad` in
  `compute_amax`.
- This document.
