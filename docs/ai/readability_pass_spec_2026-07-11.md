# Readability Pass Spec — fp8/fp4 Low-Precision Campaign (2026-07-11)

Fix specification for the readability/cleanup pass over the ~15K lines the
2026-07-09..11 low-precision campaign added (`git diff cf40d67..HEAD -- '*.mojo'`).
Produced by a read-only review; **no source file was changed by the review
itself**. Fixers apply this spec.

**Governing rule** (AGENTS.md): a comment states a constraint the code can't
show — never provenance, never PR/campaign narrative. Everything referencing
chunks, agents, sessions, coordinators, sibling ownership, "landed in",
"opt B", or dates-as-explanation gets deleted or rewritten to the timeless
constraint it encodes.

**Tags** used throughout:
- `MECHANICAL` — fixer applies exactly as written (rewrite text is verbatim).
- `JUDGMENT` — fixer decides within the stated bounds.
- `DO-NOT-TOUCH` — hand-tuned kernel bodies / unsafe vendor code; readability
  there means better comments only, never restructuring.

Rewrite text below is the full replacement for the cited comment unless the
item says "trim" (delete only the quoted clause, keep the rest).

---

## 1. Conventions (the decided scheme)

### 1.1 Comment content
- Constraints, contracts, and non-obvious invariants stay. Provenance
  (chunk/agent/session/date/commit/doc-entry citations used as *explanation*)
  goes. A `docs/ai/` pointer may remain only when the doc holds data the
  comment genuinely cannot carry (e.g. `low_precision_gotchas.md G2`'s full
  repro of the destroy-at-last-use race); cite the doc, not the campaign.
- "byte-identical to before this flag existed" phrasing → state the default
  behavior positively ("default unset = full fp8 backward"), not relative to
  a vanished before-state.
- Assert/raise message strings follow the same rule: drop "landmine #1" /
  "per docs/ai/... §N" jargon from user-visible errors; keep the constraint.

### 1.2 Naming scheme
Decided axes (documented, mostly no renames — renames only where confusion is
real; public/test-visible names are NOT churned):

| Axis | Convention |
|---|---|
| fp8 family | `lowp` prefix/suffix == the fp8 delayed-scaling family (`matmul_fwd_lowp`, `lowp_gemm_devscale`, `LOWP_ENABLED`). Historical but consistent; KEEP. Document once (matmul fp8 section header, see item M-2). |
| fp4 family | `fp4` / `nvfp4` (`matmul_fwd_fp4`, `nvfp4_quantize`). KEEP. |
| Device-resident scalar scale | `_devscale` suffix. KEEP. |
| Scale reciprocal | `scale_inv` (never `inv_scale`). Already uniform. |
| Buffer lifetime nouns | `_scratch` = transient per-call; `cache` = process-global persistent (`persistent_device_buffer`); `_buf` = per-call local `DeviceBuffer`. Already the matmul practice; document in matmul near `lowp_transpose_cache` (item M-3). |
| Rounding modes | Two enums stay (`RoundMode.RNE/SR` in lowp for fp8; `ROUND_MODE_RNE/ROUND_MODE_STOCHASTIC` in nvfp4 for fp4) — they parameterize different codecs and unifying churns tests for zero clarity gain. Cross-reference comments added (items L-9, N-2). |
| SR seed/streams | `LLMM_SR_SEED` is the one seed; decorrelation is by *stream id*. Canonical stream registry comment lives in `llmm/rng_device.mojo` (item R-2): 1 = adamw SR-master, 2/3 = nvfp4 fwd A/B, 4/5 = nvfp4 dgrad/wgrad d_output. New SR call sites take the next id and add it there. |
| Comptime flag values | `LLMM_*` names are reserved for actual `-D` flags. Comptime *variables* holding resolved flag values drop the prefix (renames T-N1/T-N2). |

**Minimal rename set** (everything else keeps its name):

| # | Rename | Scope touched | Tag |
|---|---|---|---|
| T-N1 | `LLMM_FP4_FIRST` (comptime var, train_gpt2.mojo:219) → `FP4_FIRST` | train_gpt2.mojo only | MECHANICAL |
| T-N2 | `LLMM_FP4_LAST_OVERRIDE` (train_gpt2.mojo:220) → `FP4_LAST_RAW` | train_gpt2.mojo only | MECHANICAL |
| T-N3 | `LOWP_BWD_ENABLED` → `FP8_BWD_ENABLED` (it is fp8-only by construction; the current name reads as covering fp4) | train_gpt2.mojo only | MECHANICAL |
| T-N4 | `LowpState` → `Fp8State` (+ var `lowp_state` → `fp8_state`). Verified: zero references outside train_gpt2.mojo. It is exclusively fp8 (fp4 never touches it). | train_gpt2.mojo only | JUDGMENT (do it unless a hidden ref surfaces; then document instead) |
| T-N5 | Test names encoding provenance → property: `test_e2m1_tie_breaking_matches_probe` → `test_e2m1_ties_to_lower_magnitude`; `test_e4m3_golden_value_matches_probe_algorithm` → `test_e4m3_golden_value`; `test_quantize_matches_probe_golden_vector` → `test_quantize_golden_vector` (tests/test_nvfp4_quant.mojo); `test_sign_vector_matches_documented_provenance` → `test_sign_vector_matches_reference` (tests/test_hadamard.mojo) | test files only | MECHANICAL |
| T-N6 | Consolidate the four rel-L2/cosine test helpers (`rel_l2` dead in `_lowp_test_common:83`, `cosine_and_rel_l2` `_lowp_test_common:103`, `_rel_l2` `test_lowp_gemm_fp4:94`, `_rel_l2_cosine` `test_matmul_bwd_fp4:75`, plus inline copies in test_lowp_gemm.mojo:522-547 and both test_matmul_fwd_*.mojo) onto **`cosine_and_rel_l2`** in `tests/_lowp_test_common.mojo`; likewise hoist the 3 `_pseudo_gaussian_fill` copies (test_nvfp4_quant:252, test_lowp_gemm_fp4:52, test_matmul_bwd_fp4:512) there. | tests only | JUDGMENT (behavior-neutral consolidation; keep per-file thresholds untouched) |
| — | Explicitly NOT renamed: `matmul_fwd_lowp`, `matmul_bwd_lowp`, `lowp_gemm_devscale`, `lowp_gemm_fp4`, `RoundMode`, `ROUND_MODE_*`, `NVFP4_SR_*`, all `LLMM_*` flags, `nvfp4_quant.encode_e4m3` (the collision with `lowp.encode_e4m3` is handled by cross-ref comments, items L-9/N-2 — renaming would churn test_nvfp4_quant broadly). | — | — |

### 1.3 Test-gate conventions
- Every numeric bound carries the measured value it was derived from
  (`measured 0.097, bound 0.12` style), derived cosine floors via
  `1/sqrt(1+relL2^2)`.
- A metric that is computed must be asserted (or deleted); print-only metrics
  are banned outside explicitly-labeled ablation reports.
- Skip behavior on no-GPU: silent `return` (the majority convention);
  test_rng_sr's two skip prints are removed.

### 1.4 Canonical flag documentation point
**Decision:** one comment block in `train_gpt2.mojo`, inserted immediately
above `_resolve_precision()` (currently line 117), replacing nothing —
it is the single registry; per-file comments keep only their local semantics.
The block is given verbatim in §2 (item T-1, MECHANICAL).

---

## 2. Flag registry (verbatim block for train_gpt2.mojo, item T-1)

```mojo
# ===----------------------------------------------------------------------=== #
# Build-flag registry (comptime -D unless marked env). Single source of truth;
# defining files hold the mechanics.
#
# Precision axis
#   LLMM_PRECISION=fp32|bf16|fp8|fp4   default fp32. Master axis (below).
#   LLMM_BF16=1                        alias for LLMM_PRECISION=bf16;
#                                      comptime-error if both set inconsistently.
# FP8 (all inert unless LLMM_PRECISION=fp8)
#   LLMM_FP8_FWD_ONLY=1        keep fp8 forward, force all 4 backward sites bf16.
#   LLMM_FP8_SITE_QKV=0        per-site fp8 off-switch (default 1=on); a site
#   LLMM_FP8_SITE_ATTN_PROJ=0  disabled here is bf16 in BOTH fwd and bwd (the
#   LLMM_FP8_SITE_FC=0         transpose cache + AmaxState scale are only valid
#   LLMM_FP8_SITE_PROJ=0       if that site's forward ran this step).
#   LLMM_FP8_STATIC_SCALES=1   calibrated constant scales; skips (never
#                              instantiates) the amax/update_scale kernels.
#                              Defined in llmm/lowp.mojo.
#   LLMM_FP8_STATIC_D36=1      select the d36 constant table (default d12).
#                              Only meaningful with LLMM_FP8_STATIC_SCALES.
#   LLMM_FP8_FAST_ACCUM=1      cuBLASLt fast accumulation, FORWARD GEMM only;
#                              dgrad/wgrad always precise. llmm/matmul.mojo.
# FP4 (all inert unless LLMM_PRECISION=fp4)
#   LLMM_FP4_FIRST=<int>       first N blocks stay bf16 (default 2).
#   LLMM_FP4_LAST=<int>        fp4 range end; default -1 = num_layers-2,
#                              resolved at runtime (_layer_in_fp4_range).
#   LLMM_FP4_NO_RHT=1          ablation: disable the Wgrad random Hadamard
#                              transform. llmm/matmul.mojo.
# Stochastic rounding
#   LLMM_SR_MASTER=1           SR on the master->bf16 param store (llmm/adamw.mojo).
#   LLMM_SR_SEED=<int>         shared SR seed, default 1746221221 (adamw +
#                              nvfp4; decorrelated by stream id, see
#                              llmm/rng_device.mojo's stream registry).
# Numerics / dispatch
#   LLMM_NO_TF32=1             true IEEE fp32 GEMMs (llmm/vendor.mojo).
#   LLMM_FORCE_PORTABLE_GPU=1  vendor-neutral GPU path (llmm/vendor.mojo).
#   LLMM_DISABLE_METAL=1       CPU fallback on Apple GPU (llmm/vendor.mojo).
#   WORLD_SIZE=<int>           comptime monomorphization value, default 1;
#                              runtime env WORLD_SIZE must match.
# Runtime env vars (read at startup, not -D)
#   LLMM_USE_CPU=1             force CPU dispatch (raises under bf16-storage
#                              builds, which includes fp8/fp4).
#   LLMM_OUTPUT_DIR=<path>     override output_log_dir.
#   LLMM_SAVE_EVERY=<int>      override checkpoint_every.
#   LLMM_RECOMPUTE=1           test_gpt2 only: activation-recompute build in
#                              run_test. Not read by train_gpt2 (see docs).
# Profiling (out of precision scope): LLMM_ATTN_PROFILE, LLMM_PROFILE_*,
#   LLMM_THREAD_TRACE, LLMM_TRACE.
# ===----------------------------------------------------------------------=== #
```

---

## 3. Per-file fix lists

Line numbers are absolute at HEAD `df89d4f`. Counts per file are summarized in
§7. Items are `<file-prefix>-<n>`.

### 3.1 llmm/matmul.mojo (WP1)

DELETE (MECHANICAL — remove entirely):
- M-D1 L873–876 (`lowp_gemm_devscale` docstring): "Chunk B originally also
  shipped host-Float32-scale twins … DRY pass F3 deleted it" sentence.
- M-D2 L993–1009: "does not expose transpose_a/transpose_b … deferred to the
  training-integration chunk" paragraph (now inaccurate — `lowp_gemm_fp4`
  exposes both).
- M-D3 L1844–1855: "Off-critical-path amax … left as a documented future
  optimization (see this chunk's final report)" paragraph.
- M-D4 L4018–4021: parenthetical "(FP4-closeout gotcha F8, re-shipped
  2026-07-11 after compute-sanitizer racecheck cleared F7 as environmental …)".

REWRITE (MECHANICAL — replacements verbatim):
- M-1 L73–81 (`FP8_FAST_ACCUM`):
  ```
  # -D LLMM_FP8_FAST_ACCUM=1 (default OFF) enables cuBLASLt fast-accumulation
  # on the FORWARD fp8 GEMM only (matmul_fwd_lowp's lowp_gemm_devscale call);
  # dgrad/wgrad always accumulate precisely (fast_accum=False). Independent of
  # FP8_STATIC_SCALES — any on/off combination is valid.
  ```
- M-2 L600–680 (FP8 GEMM section header). Replace the whole block with:
  ```
  # ===------------------------------------------------------------------=== #
  # FP8 GEMM — cuBLASLt native fp8 x fp8 -> bf16, TN-only.
  # ("lowp" in identifiers throughout this file == this fp8 delayed-scaling
  # family; the NVFP4 family uses "fp4"/"nvfp4".)
  #
  # Only fp8 operands with bf16 output work on this toolchain (fp8->fp8-output
  # and MAX's generic linalg fp8 both fail to lower on sm_121); there is
  # exactly one vendor body here and no comptime branch to select an
  # alternative. Forward runs e4m3 x e4m3; dgrad/wgrad run e4m3 x e5m2 (the
  # E5M2 d_output operand). TN-only: transA=OP_T, transB=OP_N; A/B
  # column-major, ld=k.
  #
  # TN operand orientation (zero-copy whenever an operand's row-major storage
  # already has the contraction dim trailing):
  #   FORWARD (contraction=C): weight[OC,C] and input[rows,C] both TN-native —
  #     no transpose (transpose_a=False, transpose_b=False).
  #   DGRAD  (contraction=OC): d_output[rows,OC] native (B role); weight needs
  #     a transposed fp8 copy (transpose_a=True, transpose_b=False).
  #   WGRAD  (contraction=rows): BOTH need a transposed fp8 copy
  #     (transpose_a=True, transpose_b=True); quantize_transpose_devscale
  #     fuses the transpose into the quantize pass.
  #
  # beta=1 accumulate is supported for fp8-operand GEMMs — wgrad passes
  # accumulate straight through to cuBLASLt beta, no bf16-scratch fallback.
  # A_SCALE_POINTER/B_SCALE_POINTER (device fp32) are honored: lowp_gemm_devscale
  # passes each operand's scale_inv so d_bf16 comes out already descaled, no
  # separate dequantize pass.
  # ===------------------------------------------------------------------=== #
  ```
  (Note: this also fixes the stale "e4m3 x e4m3 is the confirmed-working
  combination" claim — backward e4m3 x e5m2 is exercised by
  `make verify-fp8-grads`. Also update the `_lt_pick_algo` raise string at
  L793–795 to: `"no cuBLASLt algorithm for this fp8 GEMM on this
  toolchain/arch"` + the dtype triple it already prints.)
- M-3 L180: `# Persistent 32 MB cuBLASLt workspace (mirrors llm.c's single
  global workspace).` — and append one line documenting the lifetime nouns:
  `# Naming: _scratch = per-call transient; cache = persistent_device_buffer
  (process-global); _buf = per-call local DeviceBuffer.`
- M-4 L194–200 (`_lt_dt` e4m3): `# e4m3 -> R_8F_E4M3: same mapping MAX's
  blas._convert_to_cublas_datatype uses; native fp8 GEMM with bf16 output is
  the confirmed-working path on this box.`
- M-5 L205–213 (`_lt_dt` e2m1): `# e2m1 -> R_4F_E2M1 (vendor-fixed enum 33).
  Packed 2 elements/byte; cublasLtMatrixLayoutCreate rows/cols/ld are ELEMENT
  counts, not bytes — see _matmul_cublaslt_fp4.`
- M-6 L241–251 (FAST_ACCUM setter): `# CUBLASLT_MATMUL_DESC_FAST_ACCUM is an
  int8_t attribute (0=disabled, default), so it needs its own one-byte setter
  rather than reusing _lt_set_op (which sets the int32 cublasOperation_t).`
- M-7 L268–283 (shared sub-helpers header):
  ```
  # ===------------------------------------------------------------------=== #
  # Shared cuBLASLt sub-helpers.
  #
  # The three _matmul_cublaslt* orchestrators below (bf16/fp32, fp8, fp4) are
  # deliberately NOT merged into one generic wrapper: a single body with many
  # comptime switches makes this unsafe-pointer vendor code harder to audit
  # than three explicit descriptor sequences whose per-precision attributes
  # (epilogue/bias vs per-tensor scale pointers vs block-scale mode+pointers;
  # caller-selectable vs TN-fixed transposes) stay visible at each use site.
  # Only the four mechanical, byte-identical sub-steps are factored here:
  # layout creation, preference setup, heuristic-select (+empty raise), destroy.
  # ===------------------------------------------------------------------=== #
  ```
- M-8 L704–722 (`_matmul_cublaslt_fp8` docstring, fast_accum paragraph only):
  `fast_accum (default False): sets CUBLASLT_MATMUL_DESC_FAST_ACCUM
  (lower-precision, periodically-promoted accumulation). Callers enable it for
  the forward GEMM only; dgrad/wgrad keep it False.`
- M-9 L831–923 (`lowp_gemm_devscale` docstring): keep device-scalar-scale
  requirement + orientation table; replace the quantize_a/quantize_b paragraph
  (L905–922) with: `quantize_a/quantize_b (default True): when False the
  operand is NOT read/quantized here — the caller must have already filled
  a_fp8_scratch/b_fp8_scratch (in the layout transpose_a/b implies), e.g. via
  quantize_dual_devscale, because the same tensor at the same scale is also
  needed in another orientation elsewhere this step.`
- M-10 L971–1010 (FP4 GEMM section header):
  ```
  # ===------------------------------------------------------------------=== #
  # FP4 (NVFP4) GEMM — cuBLASLt native e2m1 x e2m1 -> bf16, block-scaled,
  # TN-only (transA=OP_T, transB=OP_N; op(A) reads A's [K,M] col-major buffer,
  # byte-identical to [M,K] row-major with K trailing — the same free TN
  # duality as fp8, preserved by NVFP4 K-major packing).
  # Block-scale mode CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 on both operands;
  # A/B_SCALE_POINTER = swizzled e4m3 block-scale buffers; bf16 D, fp32 compute.
  # k must be a multiple of 16 (NVFP4 block size).
  # ===------------------------------------------------------------------=== #
  ```
- M-11 L1179–1220 (per-tensor-scale correction header):
  ```
  # ===------------------------------------------------------------------=== #
  # Per-tensor-scale correction.
  #
  # NVFP4 is two-level: value ~= e2m1 * decode_e4m3(block_code) * tensor_scale.
  # cuBLASLt's VEC16_UE4M3 mode applies only decode_e4m3(block_code) per block;
  # it has no per-tensor multiplier. So _matmul_cublaslt_fp4's raw output is
  # D_true / (tensor_scale_A * tensor_scale_B). _nvfp4_post_scale_gpu multiplies
  # it back in one elementwise pass (both tensor_scales stay device-resident).
  #
  # accumulate: a cuBLASLt beta=1 accumulate is incorrect here (it would also
  # rescale the pre-existing accumulated value by this call's tensor scales).
  # So _matmul_cublaslt_fp4 always runs beta=0 into a fresh raw buffer, and
  # accumulation happens in the post-scale step, in fp32, on the fresh
  # contribution only. The post-scale kernel takes a separate raw_ptr (GEMM
  # output) and d_ptr (accumulator); callers may alias them when
  # accumulate=False.
  # ===------------------------------------------------------------------=== #
  ```
- M-12 L1318–1379 (`lowp_gemm_fp4` docstring): strip "Chunk T1/T2/T2b" labels.
  Round-mode paragraph → `a_round_mode/b_round_mode (default RNE) select RNE
  vs stochastic per operand independently; set STOCHASTIC on the gradient
  operand only. sr_seed/sr_step are forwarded to both quantize calls (consulted
  only under stochastic rounding); A and B use separate RNG substreams so their
  dither never collides.` extra_scale paragraph → `extra_scale (default 1.0):
  folded into the post-scale multiply. A caller that pre-applied the 16-wide
  RHT to both operands ((H@a)^T@(H@b) == 16 a^T b) passes extra_scale=1/16 to
  recover the untransformed result for free.`
- M-13 L1470–1492 (`bf16_control_gemm` docstring): `Plain bf16 TN GEMM through
  the SAME _matmul_cublaslt vendor call and col-major-D layout convention
  _matmul_cublaslt_fp4 uses. Exists so tests can run a same-code-path bf16
  control arm (not just a host fp32 reference), which isolates GEMM/readback
  harness issues from kernel numerics.`
- M-14 L1822–1856 (`matmul_fwd_lowp` section header):
  ```
  # matmul_fwd_lowp — fp8 forward linear. A separate entry point, not a branch
  # inside matmul_fwd: matmul_fwd's signature has no room for the per-site
  # AmaxStates a delayed-scaling fp8 GEMM needs, and every bf16/fp32 caller
  # (and the LM-head, excluded from fp8) must stay unchanged. GEMM runs
  # E4M3 x E4M3 -> bf16, then bias/GELU as the existing separate bf16 kernel
  # (cuBLASLt fp8 has restricted epilogue support).
  ```
- M-15 L1859–1921 (dual-output cache design block):
  ```
  # Dual-output quantize for the weight/input pair shared across the
  # forward/backward boundary: matmul_fwd_lowp's natural-layout quantize and
  # the backward's transposed-layout quantize of the SAME bf16 tensor at the
  # SAME (not-updated-between-fwd-and-bwd) scale are fused into one
  # quantize_dual_devscale call in forward; the transposed copy is cached in a
  # persistent per-(site,layer) buffer that backward reads read-only.
  #
  # The cache MUST be a persistent_device_buffer (process-global heap cell),
  # NOT a local DeviceBuffer: a local's destroy-at-last-use can drop the buffer
  # before a nested consumer several call-levels deep reads it (a real,
  # reproduced race — docs/ai/low_precision_gotchas.md G2). A persistent
  # buffer has no scope-tracked owner, so there is no "last use" to destroy at.
  #
  # forward-writes-then-backward-reads is safe for ANY grad_accum_steps: the
  # training loop runs forward then backward once per micro-step, sequentially,
  # on one stream — a per-(site,layer) buffer written by this micro-step's
  # forward is consumed by this same micro-step's backward before the next
  # forward can overwrite it. site/layer only build the cache's name.
  ```
- M-16 L1975–2006 (`matmul_fwd_lowp` docstring): keep orientation +
  re-quantize-every-step facts; strip "Chunk D"/"Optimization D" labels
  (JUDGMENT within that bound).
- M-17 L2016–2029 (step-1 comment): `# 1. Per-operand amax -> delayed-scaling
  update (call update_scale once per step BEFORE reading state.scale below).
  Under -D LLMM_FP8_STATIC_SCALES=1 this whole block is comptime-skipped
  (never instantiated); scales were seeded once at LowpState.__init__ and
  never change.`
- M-18 L2042–2048: `# Both amaxes are ready and no call site reads one state's
  scale before the other's, so the two update_scale calls fuse into one kernel
  launch (update_scale_pair).`
- M-19 L2057–2062: drop "Optimization D —"; keep the weight-as-A/input-as-B
  convention note.
- M-20 L2130–2177 (`matmul_fwd_fp4` docstring): strip chunk labels; keep
  2D-16x16-weight / 1D-1x16-activation / RNE-fprop facts (JUDGMENT).
- M-21 L2416–2476 + L2597–2648 (`_dbias_fused_gpu` docstring, grid-geometry
  comment): keep everything; drop the three "2026-07-10 dbias-vectorize
  writeup" citations (MECHANICAL trim).
- M-22 L2764–2770 and L3187–3195 (`matmul_d_input_bwd` / `matmul_d_weight_bwd`
  docstrings): replace the "Chunk E … mirrors Chunk D's split" sentence with:
  `fp8 dgrad/wgrad live in the sibling *_lowp entry points rather than a
  branch here; this function keeps exactly its pre-fp8 behavior.`
- M-23 L2963–3005 (`_rht_transpose_tiled_kernel` docstring): keep
  RHT-fusion/bank-conflict-padding/`rows % 16`/bit-identical facts; strip
  "gotcha F8"/"P14/P15" labels + dated re-ship note (MECHANICAL trim).
- M-24 L3097–3113 (`_gpu_transpose_add_into_kernel` docstring): keep the
  Metal SHARED-address-space / no-early-return-before-barrier constraints;
  drop the P14/P15 doc pointers (MECHANICAL trim).
- M-25 L3507–3554 (fp8 backward section header): keep the orientation table
  (L3519–3534) and the AmaxState-ownership paragraph (L3536–3543) in
  substance; strip "Chunk E", "gate (b)", "UPDATE (Optimization D,
  opt/fp8-kernels follow-on session)" labels. Lead sentence: `# fp8 backward —
  separate sibling entry points (matmul_d_input_bwd_lowp /
  matmul_d_weight_bwd_lowp / matmul_bwd_lowp), mirroring the forward split:
  the fp8 signatures need AmaxStates and cached transposed operands that the
  bf16 entry points must not carry.` (JUDGMENT within these bounds.)
- M-26 L3577–3596, L3660–3677, L3743–3778 (three fp8 bwd docstrings): keep
  "bf16 args accepted-but-unread under quantize_*=False", "doutput_state
  updated exactly once here", "input/weight states already updated in
  forward"; strip Optimization/Chunk labels (JUDGMENT).
- M-27 L3870–3993 (fp4 backward section header): strip all Chunk/T1/T2/T2a/T2b
  labels. MUST survive verbatim-in-substance: (i) L3896–3912 — why RHT is a
  separate transpose+RHT pass, NOT a fused quantize prologue (a fused prologue
  would compute the per-tensor scale from PRE-RHT data while block scales use
  POST-RHT amaxes, ~4x too high → e4m3 block-scale overflow → silent
  misscale); (ii) the operand orientation/round-mode table (L3938–3962);
  (iii) why wgrad always allocates a fresh d_raw_scratch; (iv) the SR
  seed/stream scheme (L3983–3992). (JUDGMENT within these bounds.)
- M-28 L3995: `# Ablation flag: -D LLMM_FP4_NO_RHT=1 disables the Wgrad RHT
  (see the RHT section above).`
- M-29 L4000–4025 (`_rht_transpose_prep` docstring): `Materializes RHT(src^T)
  into scratch_ptr in ONE kernel launch (coalesced 32x32 tiled transpose with
  the 16-wide RHT fused into the write phase). rows (the Wgrad contraction
  dim) must be a multiple of 16.`
- M-30 L4156–4162, L4239–4242: drop "Chunk T2a/T2b" labels; keep
  RHT-into-scratch / non-transposed-GEMM / extra_scale=1/16 rationale and
  "the ablation reproduces the no-RHT path bit-for-bit".
- M-31 Raise/assert strings at L924–927, L1380–1387, L2007–2011, L2178–2182,
  L3597–3600, L3678–3681, L4064–4067, L4148–4151: drop "landmine #1"/doc-index
  jargon; keep "low-precision kernels are GPU-only; never instantiate for the
  cpu target". (MECHANICAL trim.)

DEAD (MECHANICAL):
- M-X1 Delete unused imports: L65 `PrecisionSpec`, L19 `sync_parallelize`,
  L20 `DeviceAttribute`.
- M-X2 `USE_GELU_FUSION` (L123, hardcoded `True`): keep, but add one line:
  `# Hand-edit A/B knob (not a -D flag); the not-fused branches below are
  unreachable unless this is flipped.` The `_launch_gelu_fwd_gpu` helper
  (L1509) is currently dead code kept for that A/B — keep with the same note.

STRUCTURE:
- M-S1 JUDGMENT: extract the duplicated six-buffer NVFP4 scratch allocation in
  `matmul_d_weight_bwd_fp4` (L4184–4204 vs L4244–4263) into a
  `_fp4_wgrad_scratch(...)` helper. Behavior-neutral; do it only if the
  extraction compiles cleanly with no signature contortions.
- M-S2 Do NOT extract the Metal wgrad sub-branches of `matmul_d_weight_bwd`
  (L3260–3393) — low value, moderate risk.

DO-NOT-TOUCH zones (comments only, no restructuring):
- `_matmul_cublaslt` / `_fp8` / `_fp4` descriptor bodies: L378–597, L683–828,
  L1017–1176.
- `_matmul_bias_bwd_gpu` block reduction: L2241–2297.
- `_dbias_fused_gpu` threadfence/atomic last-block reduction: L2416–2527.
- `_gpu_transpose_kernel` L2934–2951; `_rht_transpose_tiled_kernel` (butterfly
  RHT + tiled transpose + warp shuffles) L2954–3070;
  `_gpu_transpose_add_into_kernel` L3087–3169.

### 3.2 llmm/lowp.mojo (WP2)

DELETE (MECHANICAL):
- L-D1 L8–9: "Chunk A (this file's initial state) only stubs … those are
  Chunks B/C" (factually stale).
- L-D2 L557–563: "History (DRY pass F3…): Chunk B originally also shipped
  host-Float32-scale twins … deleted" (narrative about removed code).
- L-D3 L1021–1046: the whole `FP8StaticScaleD12` "RETRY NOTE" (see bug B-3 —
  first reconcile the /4.0-vs-2.0 contradiction, then delete the incident
  retelling).

REWRITE (MECHANICAL):
- L-1 L10–14: `# Every GPU kernel here MUST be comptime
  if is_gpu[target]()-guarded and never instantiated for the "cpu" target:
  AArch64 GPU codegen crashes if any low-precision kernel is instantiated for
  cpu (see train_gpt2.mojo's _dispatch_cpu).`
- L-2 L37–39 (ScalingKind): `# PerTensor is fp8's single-scalar-per-operand
  scheme. Block1D (1x16) and Block2D (16x16) are NVFP4 granularities; the
  PerTensor amax machinery in llmm/amax.mojo rejects the block kinds (FP4
  scales are computed in-kernel by llmm/nvfp4_quant.mojo).`
- L-3 L77–78: drop "(Chunk G)": `…uses stochastic rounding instead of plain
  round-to-nearest-even.`
- L-4 L103–141 (FP4_SPEC block), collapse to:
  ```
  # NVFP4: e2m1 for BOTH fwd and bwd operands, e4m3 per-block scale, Block1D
  # (1x16) default; the weight path uses Block2D (16x16) selected by
  # nvfp4_quantize's BLOCK_ROWS template param, not a second spec.
  #
  # amax_history_len=0 / margin=0 are deliberately inert (NOT stub-pending):
  # NVFP4 computes its two-level scale fresh every nvfp4_quantize call, fully
  # device-resident; there is no delayed/history scaling for FP4 and it never
  # uses AmaxState.
  #
  # stochastic_rounding/hadamard here are precision-level markers, not
  # per-operand switches. The recipe applies SR to gradient operands only and
  # RHT to Wgrad operands only; that per-operand-role selection is made at the
  # GEMM call site, not expressible in one struct-wide bool.
  ```
- L-5 L173: drop "(Chunk B)" from the header; body L174–185 KEEP (toolchain
  constraint), may trim probe filenames (JUDGMENT).
- L-6 L195: `Stochastic rounding — not implemented for fp8 encode (see
  _SR_SEAM_MSG); use RNE.`
- L-7 L199–228 (TieMode/NanPolicy block): keep the full divergence contract
  (two encoders, two oracles, -0.0 and subnormal notes); replace the lead with
  `# TWO e4m3 encoders share _fp8_encode and answer to two DIFFERENT oracles —
  the divergence is load-bearing, do NOT unify:` and drop the "DRY pass F1" /
  audit citations. (MECHANICAL trim.)
- L-8 L443–450 `_SR_SEAM_MSG`: replace the message with `"stochastic-rounding
  fp8 encode is unimplemented. The device RNG it needs exists
  (llmm/rng_device.mojo), but the fp8 narrowing-cast SR body was never wired.
  Use RoundMode.RNE."` (also bug B-2).
- L-9 Add one cross-ref line above `encode_e4m3` (~L230): `# NOTE: a second,
  deliberately different encode_e4m3 lives in llmm/nvfp4_quant.mojo
  (ties-away, NaN-emitting — see the TieMode block above).`
- L-10 L527–537: drop "(Chunk B)"; keep the uint8-byte-view constraint.
- L-11 L542–556: `# _devscale kernels take scale_ptr: ImmutKernelPtr[float32]
  and dereference it inside the kernel, so AmaxState.scale/scale_inv
  (device-resident) are never read back to host. A scale: Float32 parameter
  would force a host readback per operand per step.`
- L-12 L631–647: `# 32x32 shared-memory tile transpose so both global read
  and write are coalesced; _QT_STRIDE = tile+1 pads the shared row to avoid
  bank conflicts on the transposed access.`
- L-13 L773–799 (`quantize_dual_devscale` docstring): `# Reads the source ONCE
  per tile and emits BOTH the natural-layout and transposed-layout fp8 copies
  (same tensor, same scale, two orientations needed by two GEMM call sites).
  Bit-identical to quantize_devscale + quantize_transpose_devscale run
  separately; a memory-traffic optimization only.`
- L-14 L914–930 (`precision_spec` docstring): keep the resolver mapping
  sentence, then one constraint sentence: `"fp4" resolves to FP4_SPEC so
  -D LLMM_PRECISION=fp4 compiles; the fp4 GEMM call sites in train_gpt2.mojo
  dispatch on PRECISION directly and never read SPEC.` Delete the
  chunk-progression paragraph.
- L-15 L944–1007 (A1 static-scales block): keep flag semantics / default-off /
  min-over-layers / 2x-safety / saturates-not-NaN constraints; trim
  "docs/ai/speedrun…A1", "the mission's own suggested margin", "both
  calibration runs printed" narrative (JUDGMENT within that bound).
- L-16 L1013–1019 (`FP8StaticScaleD12` docstring, after B-3 is reconciled):
  `"""Calibrated for GPT-2 124M (d12, 12 layer / 768 channel),
  checkpoint-init, 20 steps B=4 T=1024; running min(scale over layers AND
  steps) / <factor — see B-3>. Valid ONLY for this width/depth; recalibrate
  via calibrate_fp8_scales.mojo for any other config."""`
- L-17 L1064–1077 (`FP8StaticScaleD36`): keep "calibrated for d36 only, NOT
  valid for d12"; drop the from-scratch-40x-spread story.

DEAD (MECHANICAL):
- L-X1 Delete unused imports: L20 `is_cpu`, L23 `simd_width_of`, `size_of`.
- L-X2 The vestigial `rand: UInt32 = 0` SR params on
  `encode_e4m3`/`encode_e5m2`/`encode_fp8`: KEEP (they are the documented SR
  seam), but only after L-8 fixes the stale message.

DO-NOT-TOUCH: `_fp8_encode` L278–415, `_fp8_decode` L418–440,
`_quantize_transpose_kernel_devscale` L654–709,
`_quantize_dual_kernel_devscale` L802–863.

### 3.3 llmm/amax.mojo (WP2)

- A-D1 L4–13 DELETE (MECHANICAL): "Chunk C … DELIBERATE DEVIATION … parallel
  agents … folding at merge time" — replace with one line: `# GPU amax
  reduction + delayed per-tensor (fp8) scaling state.`
- A-1 L15–22 REWRITE (MECHANICAL): `# GPU-only, unconditionally. Callers gate
  on LOWP_ENABLED and is_gpu[target]() (AArch64 codegen crashes if any cpu
  target is instantiated under low precision). Do NOT add a CPU path here —
  the gate is the caller's.`
- A-2 L24–39 KEEP (max-is-associative determinism note).
- A-3 L41–42 REWRITE (MECHANICAL): drop "this is Chunk C's concrete
  interpretation, called out explicitly for chunk D/E/F reviewers"; keep the
  NaN/Inf-infectious description (L43–56).
- A-4 L118–122 REWRITE (MECHANICAL): `# float4_e2m1fn (max 6.0) is included
  though this file's machinery is PerTensor-only — it's a single constant.`
- A-5 L255–270 REWRITE (MECHANICAL): `# FP4 never calls
  compute_amax/AmaxState — nvfp4_quantize computes both scale levels itself,
  fresh every call. Block1D/Block2D per-block reduction is intentionally
  unbuilt (it would be unused machinery).`
- A-6 L334–352 REWRITE (MECHANICAL — the single-update ownership contract):
  `# static_scale > 0 seeds scale/scale_inv from a calibrated constant; under
  LLMM_FP8_STATIC_SCALES this init is the ONLY write scale/scale_inv ever
  receive (matmul skips update_scale entirely). static_scale <= 0 (default
  -1.0) = dynamic path: placeholder 1.0/1.0, not valid until the first
  update_scale.`
- A-7 L367–374 REWRITE (MECHANICAL trim): keep "compute the scale from the
  existing history BEFORE pushing this step's amax"; drop "…which pinned this
  down after an initial reversed-order draft failed them".
- A-8 L439 REWRITE (MECHANICAL): "Calling contract for chunk D/E (the
  training-loop integrators)" → "Calling contract:". Body L440–469 KEEP.
- A-9 L471–476 REWRITE (MECHANICAL): `Block1D/Block2D (NVFP4) raises a
  comptime error — FP4 computes its scales in nvfp4_quantize and never uses
  AmaxState (see compute_amax).`
- A-10 L478–490 REWRITE (MECHANICAL): `# (Movable): List[AmaxState] (the
  per-site storage) requires the element type to conform to Movable; Mojo
  does not infer it. Fields are all movable and there is no custom
  __moveinit__/__del__, so the synthesized move is correct.`
- A-11 L577–627 REWRITE (MECHANICAL): `# Fuses the two adjacent update_scale
  launches at a matmul_fwd_lowp site (input_state + weight_state, both
  already holding fresh amax, both consumed later in the same GEMM) into ONE
  launch (grid=(2,), one block per state). Bit-identical to two separate
  calls. Deliberately a 2-state fusion, not a whole-step batch (every other
  site updates alone, or a batch would defer a scale past where its own GEMM
  needs it).`
- A-12 L645–650 KEEP (trim "byte-for-byte `_update_scale_gpu`'s" phrasing —
  JUDGMENT).
- A-S1 JUDGMENT: extract the byte-identical scale arithmetic duplicated in
  `_update_scale_gpu` (L365–418) and `_update_scale_pair_gpu` (L657–686) into
  an `@always_inline _scale_from_history(...)` helper. Behavior-neutral
  (single-thread device code in both).

### 3.4 llmm/nvfp4_quant.mojo (WP2)

- N-1 L78–91, L94–114, L177–183 REWRITE (MECHANICAL trim): drop "chunk
  T1/T2 (backward)" labels; keep the stream-reservation constraint (distinct
  streams so dither never collides) and the transposed-quantize body.
- N-2 Add one cross-ref line above its `encode_e4m3`: `# NOTE: deliberately
  different from llmm/lowp.mojo's encode_e4m3 (ties-to-even, NaN-saturating) —
  this one is ties-away / NaN-emitting to match the cuBLAS/PyTorch NVFP4
  block-scale convention. See lowp.mojo's TieMode block.` (MECHANICAL)
- N-3 L116–121 DELETE the "that is the later merge's job" sentence; keep
  "standalone utility kernels only". (MECHANICAL)
- N-4 L736–738, L908–917 REWRITE (MECHANICAL trim): keep "a strided transpose
  read is ~3x slower, so TRANSPOSE=True dispatches to the coalesced kernel";
  drop the "2026-07-10/11 FP4-closeout finding" framing.
- N-5 L770–775 REWRITE (MECHANICAL): `# A 32x32 SMEM-tile variant is slower at
  BLOCK_ROWS=16 (only 4/256 threads survive to quantize); this
  register-per-row design keeps all threads active.`
- N-X1 Delete unused import L128 `is_cpu`. (MECHANICAL)
- N-S1 JUDGMENT: extract the duplicated block-scale computation
  (amax → block_scale_raw → sc_code → sc_val; L646–654 and L831–839) into an
  `@always_inline _nvfp4_block_scale(amax, tensor_scale) -> (UInt8, Float32)`.
  Hot kernels — do it only if PTX-diff or a perf spot-check shows no change.
- N-S2 Do NOT try to unify the RNE and STOCHASTIC bracket ladders in
  `encode_e2m1` (midpoint table vs bracket table — different math).

DO-NOT-TOUCH: `encode_e2m1` L199–281, `_nvfp4_quantize_gpu` L597–713,
`_nvfp4_quantize_transpose_coalesced_gpu` L715–864.

### 3.5 llmm/hadamard.mojo (WP2)

- H-D1 L13–16 DELETE (MECHANICAL): "…out of scope here per the coordinator's
  HARD CONSTRAINT (do not touch llmm/matmul.mojo)". Keep: `This module builds
  the transform and its exact inverse only.`
- H-D2 L43–49 DELETE (MECHANICAL): "the coordinator's device-RNG chunk … FP8
  team's design …".
- H-1 L32–42 REWRITE (MECHANICAL): `# The sign vector is a single fixed +/-1
  vector for ALL of training — NOT regenerated per call, NOT a device RNG.
  Hardcoded literal below (generated once from random.Random(1234)); the
  hadamard_sign values must not change or checkpoints/repro diverge.` Also fix
  the `HADAMARD_SIGNS` name in prose → `hadamard_sign()` (bug B-12).
- H-2 L51–60 KEEP (trim "for consistency with the rest of the
  low-precision-adjacent surface" — JUDGMENT).
- H-X1 Delete unused import L65 `is_cpu`. (MECHANICAL)

DO-NOT-TOUCH: `hadamard_sign` L79–115, `_fwht16` L123–143,
`_hadamard16_kernel` L151–188.

### 3.6 llmm/rng_device.mojo (WP2)

- R-1 L5–11 REWRITE (MECHANICAL): `# Consumers: llmm/adamw.mojo's SR
  master-weight store (LLMM_SR_MASTER) and llmm/nvfp4_quant.mojo's
  ROUND_MODE_STOCHASTIC e2m1 encode.` (Current text says the quantize seam is
  "future" and "owned by other agents" — both stale.)
- R-2 L54 REWRITE (MECHANICAL): "## Contract (what other chunks — and
  quantize()'s SR seam — can rely on)" → "## Contract". Append the canonical
  stream registry (convention §1.2): `# Stream registry (one id per SR call
  site, all sharing LLMM_SR_SEED): 1 = adamw SR-master; 2/3 = nvfp4 forward
  A/B operands; 4/5 = nvfp4 dgrad/wgrad d_output. New SR call sites take the
  next id and record it here.`
- R-3 L87 REWRITE (MECHANICAL trim): drop "per the landmine notes above"
  phrasing; keep the AArch64-targets risk statement.
- L12–52, L55–88 KEEP (MT19937 contrast, Squares-vs-Philox rationale,
  contract bullets, single-implementation rationale — all timeless).

DO-NOT-TOUCH: `squares32` L135–158, `sr_round_bits` L206–230.

### 3.7 llmm/adamw.mojo (WP2)

- W-1 L25–33 REWRITE (MECHANICAL):
  ```
  # Opt-in stochastic rounding for the master->low-precision param store,
  # enabled with `-D LLMM_SR_MASTER=1`. Default OFF: the store falls through
  # to the plain `param.cast[dtype]()` RNE path below. Only valid for a bf16
  # param store (asserted in `_adamw_update`); fp8/fp4 encoders are not wired
  # through here.
  ```
- W-2 L36–40 KEEP (`SR_MASTER_SEED`).
- W-3 L43–46 REWRITE (MECHANICAL): `# rng_device stream id reserved for this
  call site so its substream stays independent of any other SR call site
  sharing LLMM_SR_SEED. See llmm/rng_device.mojo's stream registry.`
- W-4 L184–190 REWRITE (MECHANICAL) the assert message to:
  `"LLMM_SR_MASTER=1 requires a bf16 low-precision param store
  (dtype == DType.bfloat16); fp8/fp4 stochastic rounding is not wired into
  this kernel"`.
- L118–122, L170–182, L193–196 KEEP.

### 3.8 llmm/io.mojo, llmm/memory.mojo, llmm/vendor.mojo, llmm/attention.mojo (WP2)

- I-1 io.mojo L18–27 REWRITE (MECHANICAL): `# 1 byte. fp8 is a transient
  GEMM-operand dtype and should not reach a host byte-count call site, but
  give it an explicit case so it never falls through to the wrong 4-byte
  default below.`
- I-2 io.mojo L28–45 REWRITE (MECHANICAL trim): drop the design-doc seam
  citation; keep "sub-byte/packed dtypes (fp4/e2m1) have no whole-byte element
  size and must be special-cased by any future packed path". See bug B-6
  (the debug_assert is a release no-op).
- E-1 memory.mojo L75–91 REWRITE (MECHANICAL):
  ```
  """Allocate-once, heap-held device buffer keyed by (name_suffix, ctx.id())
  via a process global. The first call for a given key allocates `count`
  elements (zeroed iff `zero=True`); every later call with the same
  name/device returns the cached pointer and IGNORES its `count`/`zero`. The
  caller must pass a `count` that upper-bounds every call site sharing a
  name_suffix.
  """
  ```
- V-1 vendor.mojo L38–54 KEEP; JUDGMENT: trim the hardcoded
  `train_gpt2_fp32.cu:1614-1618` line citation to "llm.c's fp32 arm
  auto-enables CUBLAS_COMPUTE_32F_FAST_TF32 on cc>=8.0".
- attention.mojo L4266–4271 KEEP (TF32 tie-in; nothing to do).

### 3.9 llmm/layernorm.mojo (WP1)

- Y-1 L1490–1518 REWRITE (MECHANICAL) the `_layernorm_bwd_fused_gpu`
  docstring:
  ```
  # Fused LN-backward MAIN pass (paired with the _ln_bwd_fused_finalize_gpu
  # launch below). One sweep over [B,T,C] produces d_input, folds the
  # residual-grad seed (HAS_RESID_IN), and reduces dgamma/dbeta.
  #
  # dgamma/dbeta accumulate into per-thread REGISTERS, not shared memory:
  # thread `tid` owns a fixed disjoint set of channel offsets on every row
  # (i = tile_base + tid*width aligned; i = col_base + tid scalar), so no
  # other thread touches its accumulator. The reduction is therefore
  # deterministic and needs only a single scratch flush at the end; the
  # cross-block sum is a separate finalize launch.
  #
  # `aligned` proof: the host dispatch sets aligned=True only when
  # channels % width == 0, making every row*channels+i offset width-aligned
  # and the scalar tail (i < channels < i+width) unreachable.
  ```
- Y-2 L1738–1750 REWRITE (MECHANICAL):
  ```
  # Deterministic cross-block reduction of the per-block dgamma/dbeta
  # partials. ONE BLOCK PER CHANNEL (block_idx.x == channel) so all channels
  # reduce concurrently; each channel's blocks_cap partials are contiguous in
  # the channel-major scratch and read fully coalesced. Fixed block dims ->
  # fixed reduction tree -> bit-reproducible. d_gamma/d_beta ALWAYS accumulate
  # (grad accumulation across micro-steps). No re-zero needed: the main kernel
  # overwrites every [0:num_blocks) slot each launch.
  ```
- Y-3 L2334–2348 REWRITE (MECHANICAL): keep the two-kernel description +
  `# SM_OVERPROVISION=3: at 72 reg/thread the kernel is register-bound to 3
  blocks/SM, so 3*num_sm saturates every SM in one wave. Fewer blocks means
  fewer per-block partials, shrinking both the scratch flush and the finalize
  reduction.` Drop the edd2b67/ncu narrative.
- Y-4 L2441 DELETE (MECHANICAL) the sentence "Replaces the first attempt's
  serial single-block in-kernel finalize."
- Y-5 L2295–2300 REWRITE (MECHANICAL): `# CPU: seed broadcast (if
  HAS_RESID_IN) then layernorm_bwd then broadcast. The seed call lives here
  (not at the train_gpt2.mojo call site) so both targets share one signature;
  arithmetic and kernel order are unchanged.`
- L1442–1452, L1521–1525, L1460–1466, L2264–2273, L2582–2587 KEEP.

DO-NOT-TOUCH: `_layernorm_bwd_fused_gpu` vector path L1529–1651, scalar path
L1652–1724; `_ln_bwd_fused_finalize_gpu` body L1751–1767.

### 3.10 train_gpt2.mojo (WP3)

- T-1 MECHANICAL: insert the §2 flag-registry block immediately above
  `_resolve_precision()` (L117).
- T-2 L111–116 REWRITE (MECHANICAL): `# fp8/fp4 are transient dtypes used
  only inside the GEMM, never the parameter/activation/gradient storage
  dtype — so STORAGE_DTYPE collapses fp8/fp4 onto bfloat16, and USE_BF16
  (fp32-master + bf16 storage) is true for fp8/fp4 too.`
- T-3 L147–152 REWRITE (MECHANICAL): `# -D LLMM_FP8_FWD_ONLY=1 keeps the fp8
  forward linears but forces the four per-block backward sites onto their
  bf16 matmul_bwd branch. Default unset is full fp8 backward.`
- T-4 L153–163 REWRITE (MECHANICAL): `# fp4's backward (matmul_bwd_fp4) is a
  separate branch at the two FP4-eligible MLP sites: it has a different
  signature (no AmaxStates; an sr_step counter) than fp8's matmul_bwd_lowp,
  so it is not folded into FP8_BWD_ENABLED.`
- T-5 L168–184 REWRITE (MECHANICAL): `# Per-site fp8 gates (default all on).
  Setting -D LLMM_FP8_SITE_<SITE>=0 routes that single site to the bf16
  matmul_fwd/matmul_bwd branch in BOTH passes — a site disabled in forward
  MUST be disabled in backward, because matmul_fwd_lowp's transpose cache and
  the site's AmaxState scale are only valid if that site's forward ran this
  step.`
- T-6 L196–200 KEEP (USE_BF16 meaning); L204–218 KEEP but trim the two
  `docs/ai/fp4_training_recipes_research.md §1 "…"` citations (JUDGMENT).
- T-7 L685–733 (`LowpState` module comment) REWRITE (JUDGMENT within bounds):
  collapse to the two constraints — (a) per-layer (not shared) AmaxState
  because delayed scaling needs same-tensor amax history; (b) fields always
  declared, bodies gated `comptime if LOWP_ENABLED` so AmaxState/GPU kernels
  are never instantiated on non-lowp and CPU builds. Drop all
  Chunk/coordinator/parallel-work narrative and the design-doc line-number
  refs.
- T-8 L746, L750, L754, L758 REWRITE (MECHANICAL): each inline
  `# Chunk E (bwd, E5M2)` → `# backward d_output operand (E5M2)`.
- T-9 L774–789 + L301–312 REWRITE (MECHANICAL trim): drop "A1
  (docs/…)"/"modded-nanogpt"; keep "static mode shares one calibrated constant
  per (site, role) across all layers, unlike the per-layer dynamic path".
- T-10 L1043–1046, L1273–1281 REWRITE (MECHANICAL trim): drop "(Chunk D)"
  tags.
- T-11 L2752–2759 REWRITE (MECHANICAL): `# fp4 SR step counter: unique per
  (training step, micro-step) so grad-accum micro-steps draw distinct dither;
  only consulted under fp4.`
- T-12 L2969, L2992, L3052, L3075 REWRITE (MECHANICAL trim): delete
  "(Chunk E)"/"(Chunk T2, …)" tags; keep the operand-mapping constraint text.
- T-13 Dead code (MECHANICAL): delete `comptime SPEC =
  precision_spec[PRECISION]()` (L202) and the `precision_spec` import (L25) —
  verified unreferenced anywhere.
- T-14 Renames T-N1/T-N2/T-N3 (§1.2, MECHANICAL) and T-N4 (JUDGMENT).
- T-15 JUDGMENT (optional, only if the precision surface keeps growing): move
  the precision axis (`_resolve_precision` … `_layer_in_fp4_range`) into a new
  `llmm/precision_config.mojo`. NOT required by this pass; do not do it in the
  same commit as the comment rewrites.
- T-16 JUDGMENT declined: leave `LowpState.__init__`'s 12-fold `.append`
  unrolling as-is (a `@parameter for` table may fight Mojo's typing).

### 3.11 test_gpt2.mojo, calibrate_fp8_scales.mojo, dump_grads_gpt2.mojo (WP3)

- G-1 test_gpt2.mojo L234–247 KEEP; trim the "see docs/ai/…, 2026-07-10
  entry" citation (MECHANICAL trim). L274–275 KEEP.
- C-1 calibrate_fp8_scales.mojo L1–63 REWRITE (JUDGMENT within bounds): KEEP
  the "read every state's scale after EVERY step and keep a running min — an
  end-of-run readback forgets steps older than amax_history_len=16, which for
  a fresh checkpoint can be the largest amax of the run" rationale (genuine
  algorithmic constraint). DELETE "A1", "the mission's own allowance"
  (L54), "first version of this tool … incident" provenance.
- P-1 dump_grads_gpt2.mojo L1–25 REWRITE (JUDGMENT within bounds): keep what
  the tool does (148 per-tensor fp32 grad files: 12×L per-layer + 4 global;
  build fp8 and bf16 into separate dirs; compare with
  tests/compare_grad_dumps.py). Drop "Chunk E Gate C tooling" and MEMORY.md
  references.

### 3.12 tests/*.mojo (WP3)

Debug prints — DELETE (MECHANICAL) unless marked keep:
- test_lowp_gemm_fp4.mojo L271, L521; test_matmul_fwd_fp4.mojo L190;
  test_matmul_fwd_lowp.mojo L216; test_matmul_bwd_fp4.mojo L273–283, L479;
  test_nvfp4_quant.mojo L363, L373; test_rng_sr.mojo L348, L383 (skip
  messages → silent `return`, matching every other file).
- KEEP test_matmul_bwd_fp4.mojo L731–741 (the RHT on/off ablation table is
  the test's stated output) — but see bug B-1: its metrics must also be
  asserted.

Dead helpers / unused imports — DELETE (MECHANICAL):
- _lowp_test_common.mojo L83 `rel_l2` (dead; its own comment admits it) —
  superseded by rename T-N6's consolidation.
- test_lowp_gemm.mojo L32 `assert_false`, L41 `RoundMode`, L47 `MutKernelPtr,
  ImmutKernelPtr`; test_matmul_fwd_lowp.mojo L57 + test_matmul_fwd_fp4.mojo
  L43 `MutKernelPtr, ImmutKernelPtr`; test_lowp_bwd.mojo L53 `MutKernelPtr`.

Narrative headers/comments — REWRITE (MECHANICAL unless noted):
- test_amax.mojo L2 header → `# test_amax.mojo — amax reduction + AmaxState
  delayed-scaling gates.` (keep the coverage list L3–15, dropping the
  "shared-GPU convention" citation at L15). L487–492 →
  `# update_scale_pair (one kernel launch, two states) must be bit-identical
  to calling update_scale on each state separately, across a multi-step
  warmup+steady-state sequence.`
- test_lowp_bwd.mojo L1–30 header: keep the three-function description and
  metric; drop every Chunk/MEMORY reference. L136–165: drop
  "Optimization B/D (docs/… entry)" tags; keep "the test primes the cache
  itself because it calls the sibling directly".
- test_lowp_gemm.mojo L2 → plain title. L12, L248–249, L447–449 →
  `# Host-computed scale uploaded into a 1-element device buffer (the
  devscale contract).` L144 → keep the regression description ("an e4m3
  reference that rounded ties away from even produced 0x2E for 0.09375;
  correct ties-to-even is 0x2D") and drop "found in a sibling (FP4-probe)".
  L286, L429 → drop "Chunk C's"; keep the constraint. L519–521 → keep the
  aggregate-metrics-near-zero-crossings rationale; drop MEMORY/Chunk refs.
- test_lowp_gemm_fp4.mojo L1–24: keep the bf16-control-arm rationale, drop
  probe provenance. L284–286 → `…suspect the comparison harness (D
  layout/readback), not the fp4 kernel.` L308–313, L388–392 → `# Use 512^3:
  much smaller M/N/K can make cublasLtMatmulAlgoGetHeuristic return zero
  NVFP4 candidates, a dispatch artifact unrelated to this test.` L345: drop
  "per the recipe".
- test_matmul_bwd_fp4.mojo L1–31 → plain title. L285–304: keep the
  `cosine ~ 1/sqrt(1+relL2^2)` derivation + measured floors; drop the
  `cac1968` commit hash and T2a/T2b labels. L794–821 KEEP the outlier-physics
  explanation; trim the meta line L797–799. L500–507 KEEP (accurate
  comptime-flag limitation note).
- test_matmul_fwd_fp4.mojo L2 → plain title. L212–219: keep the cosine-floor
  derivation; drop "gate C failure, 2026-07-10".
- test_matmul_fwd_lowp.mojo L2–3 → plain title. L23–41 DEVIATION block: keep
  the reason 0.125 replaced the design doc's 0.02; strip Chunk tags and
  "coordinator's allowance".
- test_hadamard.mojo L62–66: drop "see … this file's task notes"; state the
  golden vector is independently computed in Python. Rename per T-N5.
- test_nvfp4_quant.mojo L139–143 → `# Golden vector for a fixed 16-value
  block, independently computed in Python and checked against this codec.`
  L58–63, L113–124, L211: restate as properties (ties-to-lower-magnitude;
  e4m3 golden 0x39 → 1.125; swizzle offsets) without citing probe_fp4.cu.
- test_rng_sr.mojo L5, L14, L29–33: drop "Chunk G" and landing-commit
  narrative; keep the (a)–(e) coverage list.
- _lowp_test_common.mojo L1–25: keep the reflection constraint (helpers split
  out so `__functions_in_module()` doesn't treat them as test candidates);
  delete the F5-audit citation and the `TODO(post fp4-merge)` story. L35,
  L98–99, L145–152: drop "verbatim" + source-line citations.
- _rng_sr_gpu_kernels.mojo L1–20 KEEP (genuine structural constraint);
  JUDGMENT: may trim the MEMORY.md cross-ref.

Probe dirs (JUDGMENT, maintainer call): `tests/probe_fp8/` and
`tests/probe_fp4/` are throwaway probes and can be deleted — but ONLY after
the rewrites above remove the production-test references to
`probe_fp4.cu` / `probe_*/RESULTS.md`, and after the lowp.mojo L174–185
toolchain-constraint comment is confirmed to stand alone without them. If
kept, add a one-line README in each: "historical probes; nothing imports
them".

Threshold fixes (report → §6 bug B-1; the fix itself is JUDGMENT for the
fixer of WP3): test_matmul_bwd_fp4 L767–774, L822–825; test_nvfp4_quant
L364, L375 (attach measured values to the bounds).

---

## 4. DO-NOT-TOUCH master list

| File | Zone | Lines |
|---|---|---|
| llmm/matmul.mojo | cuBLASLt descriptor bodies (x3) | 378–597, 683–828, 1017–1176 |
| llmm/matmul.mojo | `_matmul_bias_bwd_gpu` reduction | 2241–2297 |
| llmm/matmul.mojo | `_dbias_fused_gpu` threadfence reduction | 2416–2527 |
| llmm/matmul.mojo | transpose/RHT kernels | 2934–2951, 2954–3070, 3087–3169 |
| llmm/lowp.mojo | `_fp8_encode` / `_fp8_decode` | 278–415, 418–440 |
| llmm/lowp.mojo | tiled quantize-transpose kernels | 654–709, 802–863 |
| llmm/nvfp4_quant.mojo | `encode_e2m1`, quantize kernels | 199–281, 597–713, 715–864 |
| llmm/hadamard.mojo | sign/butterfly kernels | 79–115, 123–143, 151–188 |
| llmm/rng_device.mojo | `squares32`, `sr_round_bits` | 135–158, 206–230 |
| llmm/layernorm.mojo | fused-bwd vector/scalar paths + finalize | 1529–1651, 1652–1724, 1751–1767 |

In every zone: comment improvements only; no code motion, no "simplification",
no loop restructuring. Any change there requires its own perf + bit-exactness
gate.

---

## 5. Work packages (3 disjoint fixers)

| WP | Files | Approx. items |
|---|---|---|
| **WP1 — big kernels** | llmm/matmul.mojo, llmm/layernorm.mojo, llmm/attention.mojo, llmm/vendor.mojo | 35 rewrites + 4 deletes + 3 dead imports + M-S1 |
| **WP2 — low-precision library** | llmm/lowp.mojo, llmm/amax.mojo, llmm/nvfp4_quant.mojo, llmm/hadamard.mojo, llmm/rng_device.mojo, llmm/adamw.mojo, llmm/io.mojo, llmm/memory.mojo | 38 rewrites/deletes + 4 dead imports + A-S1/N-S1 |
| **WP3 — trainer, tools, tests** | train_gpt2.mojo, test_gpt2.mojo, calibrate_fp8_scales.mojo, dump_grads_gpt2.mojo, tests/*.mojo (12 campaign files), probe dirs decision | flag-registry insert + 16 trainer items + renames T-N1..T-N6 + ~30 test items + 10 debug prints + 6 unused imports |

Rules for fixers: one WP per fixer, separate worktrees, no cross-WP edits
(the only shared seam is T-N3/T-N4 renames, which are train_gpt2-local, i.e.
WP3's). Every WP commit must pass `make lint` and the affected build targets
(`build`, `build-bf16`, `build-fp8`, `build-fp4` for WP1/WP2; plus
`test-gpt2` and the campaign test targets for WP3). Comment-only changes
still require a rebuild check — binary mtime, per AGENTS.md.

---

## 6. Real bugs / risks found during review (REPORT ONLY — not part of the readability fix; each needs its own owner/gate)

- **B-1 (HIGH, test gate) — weak wgrad-RHT gates.**
  `tests/test_matmul_bwd_fp4.mojo:822-825` asserts `rel_l2_rht < 0.40` against
  a measured 0.0969 (4x headroom — cannot catch a regression) and never
  asserts the computed `cosine_rht` / RHT-off metrics; L767–774 likewise
  computes but never asserts the ablation's RHT-off arm. Exactly the
  weak-gates-overrule-nothing failure class. Tighten to measured+margin and
  assert every computed metric.
- **B-2 (MED, latent trap) — stale `_SR_SEAM_MSG`** (llmm/lowp.mojo:443–450):
  claims the device RNG "does not exist yet"; it does (rng_device.mojo, in
  production use by nvfp4). Latent: `quantize_devscale[spec=FP4_SPEC]` would
  select `RoundMode.SR` and hit the comptime `assert False`. Safe today only
  because fp4 never routes through `quantize_devscale`.
- **B-3 (MED, doc/data) — FP8StaticScaleD12 contradiction** (llmm/lowp.mojo):
  L1017–1018 says the table is `min / 4.0`; the RETRY NOTE (L1042–1044) says
  the factor was kept at 2.0 (and that 4.0 made the gate worse). Verified in
  source. Reconcile against calibrate_fp8_scales.mojo's actual factor before
  applying L-16.
- **B-4 (MED, perf/waste) — LowpState fully allocated under fp4**
  (train_gpt2.mojo:790): 12 × num_layer `AmaxState[FP8_SPEC]` instances +
  their GPU init kernels are created under `LOWP_ENABLED` (which includes
  fp4) but every read is fp8-gated. Gate population on `PRECISION == "fp8"`.
- **B-5 (LOW, dead code) — `comptime SPEC` + `precision_spec` import dead in
  train_gpt2.mojo** (L202, L25). Covered by T-13.
- **B-6 (LOW, silent-failure) — io.mojo `debug_assert(False)` then
  `return 4`:** debug_assert is a no-op in release, so an unrecognized/packed
  dtype silently gets a wrong 4-byte size — the exact landmine class the
  branch was added to close. Consider raising instead.
- **B-7 (LOW, footgun) — `persistent_device_buffer` first-call `count` wins
  permanently** (llmm/memory.mojo); a later larger-count caller silently gets
  the smaller buffer → OOB device writes, no diagnostic. Cheap fix: cache the
  count, `debug_assert(count <= cached)` on reuse (plus B-6's caveat about
  debug_assert in release).
- **B-8 (LOW, confirm intent) — `LLMM_RECOMPUTE` only wired in
  test_gpt2.mojo** (L717); train_gpt2.mojo never reads it, so recompute
  cannot be enabled for real training runs. Confirm test-only is intended.
- **B-9 (LOW, misleading API) — `PrecisionSpec.hadamard` and `.block` are
  never read by any code** (prose only). A reader will assume
  `hadamard=True` auto-applies RHT; it does not. Either wire them or mark
  them descriptive-only in the L-4 rewrite (the given text already does).
- **B-10 (LOW, drift risk) — format-max constants triplicated:** 448 / 57344 /
  6.0 exist as `E4M3_MAX`/`E5M2_MAX` (lowp), `E2M1_MAX` (nvfp4), AND as bare
  literals in `amax.format_max`. Make `format_max` consume the named consts.
- **B-11 (LOW, perf trap) — `_nvfp4_quantize_gpu`'s TRANSPOSE=True branch is
  dead in production** (dispatch routes TRANSPOSE=True to the coalesced
  kernel) and is the known ~3x-slower path; a future direct caller silently
  regresses. Consider dropping the param from that kernel.
- **B-12 (COSMETIC) — hadamard.mojo docstring names `HADAMARD_SIGNS`, which
  does not exist** (the implementation is `hadamard_sign()`); fixed by H-1.
- **B-13 (INFO, stale claim now fixed by M-2) — matmul's fp8 header/error
  string claim only e4m3 x e4m3 is "confirmed-working"** while both fp8
  backward GEMMs run e4m3 x e5m2 in production (`make verify-fp8-grads`
  exercises them). Not a live bug — a stale comment; M-2/M-31 update it. A
  one-time explicit confirmation that the e4m3 x e5m2 heuristic returns
  algorithms on a clean box would close it fully.
- **B-14 (INFO, asymmetry) — `matmul_fwd_lowp` lacks the defensive keep-alive
  `_ = buf.unsafe_ptr()` pattern `matmul_bwd_lowp` uses** (L3866–3867). Safe
  as-is (the consuming GEMM is enqueued before return); worth a one-line
  comment so nobody "fixes" the asymmetry in the wrong direction.
- **B-15 (INFO, perf) — per-call cuBLASLt descriptor create/destroy +
  heuristic on every GEMM** (48+ calls/step across the three orchestrators).
  Profiling candidate, not a correctness issue.

---

## 7. Counts per file (DELETE / REWRITE / KEEP-noted / dead items)

| File | DELETE | REWRITE | KEEP (noted) | Dead/unused |
|---|---|---|---|---|
| llmm/matmul.mojo | 4 | 27 (+8 assert strings) | 4 | 3 imports |
| llmm/lowp.mojo | 3 | 14 | 3 | 3 imports |
| llmm/amax.mojo | 1 | 10 | 3 | 0 |
| llmm/nvfp4_quant.mojo | 1 | 5 | — | 1 import |
| llmm/hadamard.mojo | 2 | 2 | 1 | 1 import |
| llmm/rng_device.mojo | 0 | 3 | 4 | 0 |
| llmm/adamw.mojo | 0 | 3 | 4 | 0 |
| llmm/io.mojo | 0 | 2 | 0 | 0 |
| llmm/memory.mojo | 0 | 1 | 0 | 0 |
| llmm/vendor.mojo | 0 | 1 (trim) | 1 | 0 |
| llmm/attention.mojo | 0 | 0 | 1 | 0 |
| llmm/layernorm.mojo | 1 | 4 | 5 | 0 |
| train_gpt2.mojo | 0 | 12 | 2 | SPEC + import |
| test_gpt2.mojo | 0 | 1 (trim) | 2 | 0 |
| calibrate_fp8_scales.mojo | 0 | 1 (block) | — | 0 |
| dump_grads_gpt2.mojo | 0 | 1 (block) | — | 0 |
| tests/ (12 files) | 10 prints | ~28 | 4 | 6 imports + 1 helper |

---

_Written with AI assistance (Claude Code / Fable agent, five parallel review
subagents with coordinator verification of load-bearing claims), directed by
Evan Owen._
