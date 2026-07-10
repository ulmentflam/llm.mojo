# DRY Consolidation Audit — parallel low-precision campaign

Audit HEAD: `c939e1b` (branch `consolidation/dry-audit`) — FP8 chunks A–G merged +
FP4 kernels (`llmm/nvfp4_quant.mojo`, `llmm/hadamard.mojo`) merged. Date: 2026-07-10.

Scope: duplicate files, duplicated logic, and merge-order for the eventual `main`
merge. This document is **analysis + fix specs only** — nothing outside it was
changed. Fixers should read the per-finding SAFE-NOW / WAIT tag before touching code.

## TL;DR — the one strategic fact

**Almost every consolidation-worthy code duplication lives in exactly three files
that are all in the active FP4 merge path: `llmm/matmul.mojo`, `llmm/lowp.mojo`,
`llmm/nvfp4_quant.mojo`.** `llmm/matmul.mojo` is touched by **all four** merge-bound
branches (fp32-parity +14, dbias-fusion +170, fp8-training +917, fp4-training +1449
lines vs `main`). Consolidating those duplications *now* means a **third** conflicting
rewrite of the hottest merge file while an in-flight T1 agent is still adding
`matmul_fwd_fp4` / train wiring / `build-fp4` on top of `goal3b/fp4-training`
(tip `48cfd73`, already 1 commit ahead of this HEAD).

Therefore the highest-value move is **not** to DRY these files in isolation, but to
**land the FP4 merge first, then do the codec/GEMM consolidation as the immediately
following cleanup pass on a single settled tree.** The genuinely SAFE-NOW items are
the ones that avoid those three files: the test-helper extraction and the docs
cross-referencing. Everything else is WAIT-FOR-fp4-merge, with fix specs pre-written
below so the post-merge pass is mechanical.

---

## Status (consolidation pass executed 2026-07-10, branch `integrate/all-goals`)

Executed post-merge in the recommended order F4 -> F3 -> F1 -> F2, one commit
each, gates re-run after each; full gate results and the trajectory verdict
are in `ai_assisted_optimizations_and_benchmarks.md`'s "DRY consolidation
pass" entry.

| finding | status | note |
|---|---|---|
| F1 e4m3 codec | **DONE** (`8d7626f`) | Decode unified; `_fp8_encode[tie_mode, nan_policy]` core; both public names kept. Staleness correction: the divergence is four-fold, not two — the AWAY mode also preserves the probe-lineage -0.0 sign and float-arithmetic subnormal branch bit-for-bit (documented at `TieMode`). Brute-force 0-mismatch sweep vs the old body. |
| F2 cuBLASLt dance | **DONE, partial per this audit's own verdict** (`5bca47e`) | Only `_lt_make_layout`/`_lt_make_pref`/`_lt_pick_algo`/`_lt_destroy` extracted; the three orchestrators stay distinct, with the deliberately-not-consolidated rationale in a comment block at the helper section. |
| F3 host-scale twins | **DONE** (`2d3038c`) | All three host-scale functions deleted (~172 net LOC); tests ported to `_devscale` + 1-element device scale buffers. Post-audit re-check: still zero production callers; `quantize_dual_devscale` (added by fp8-quant-opt after this audit) has no host twin. |
| F4 persistent-buffer global | **DONE** (`b12691e`) | `persistent_device_buffer` in `llmm/memory.mojo`; **five** sites collapsed (the three inventoried here plus post-audit `_dbias_counters` and `_ln_bwd_dparam_scratch`). |
| F5 test helpers | **DONE (previous pass)** | `tests/_lowp_test_common.mojo` landed with the F5 safe-now spec; composition with quant-opt re-proven as GATE 0 of this pass (10/10, 5/5, 19/19). `test_lowp_gemm_fp4.mojo`'s adoption of `_host_gemm_ref` remains an open one-liner follow-up. |
| F6 Makefile clones | **SKIPPED (as recommended)** | Leave-as-is verdict stands; repetition is make-idiomatic. |
| F7 accum+finalize idiom | **NOT CONSOLIDATED (note stands)** | Post-LN-redesign the dbias and ln-dparam bodies remain substantively different (contention-free partials vs block-sum + different finalize); no shared shape emerged. |
| D1-D4 docs overlap | **OUT OF SCOPE for this pass** | Belongs to the separate docs pass (task 6). D3's fix (orphan `26b45a4`) already landed on this branch. |

## Ranked findings

Ranked by (duplication severity × consolidation safety). "Severity" = how much
duplicated logic and how likely it drifts into a bug; "safety" = can it be done now
without colliding with an in-flight branch.

### F1 — e4m3 codec: two full implementations, decode identical, encode deliberately divergent  (HIGH severity · WAIT-FOR-fp4-merge)

Two independent e4m3 encode/decode pairs exist under the **same names**:

- `llmm/lowp.mojo`: `encode_e4m3` (→ generic `_fp8_encode_rne[4,3,7,E4M3_MAX]`,
  lines 205–311, 349–362) and `decode_e4m3` (→ `_fp8_decode[4,3,7]`, lines 314–336,
  379–382).
- `llmm/nvfp4_quant.mojo`: `encode_e4m3` (lines 285–338) and `decode_e4m3`
  (lines 341–357).

Semantic diff (verified by reading both, bit-level):

- **`decode_e4m3` is bit-for-bit identical.** Both compute subnormal `m/512` and
  normal `(1 + m/8)·2^(e−7)` by direct exponent-field construction. Pure duplication.
- **`encode_e4m3` deliberately differs and the difference is LOAD-BEARING:**
  - lowp's is **round-to-nearest-EVEN** (ties-to-even: `rem > half or (rem==half and
    lsb==1)`), chosen to match Mojo's **host** `.cast[float8_e4m3fn]()` bit-for-bit so
    the GPU quantizer agrees with the host oracle in `tests/test_lowp_gemm.mojo`.
  - nvfp4's is **round-to-nearest, ties-AWAY-from-zero** (`Int(x*8 + 0.5)`), chosen to
    match `tests/probe_fp4/probe_fp4.cu`'s `roundf`/nearest-candidate reference and the
    cuBLAS/PyTorch NVFP4 convention.
  - **NaN policy also differs:** lowp saturates NaN→448 (`not (ax <= max)` catches NaN);
    nvfp4 emits the `0x7F` NaN byte. In nvfp4 the codec only ever encodes **non-negative
    block scales** (`ue4m3`), so its NaN/sign paths are effectively dead but present.

Verdict: the two encoders answer to **two different oracles** (host-cast RNE vs
probe/cuBLAS ties-away), so they cannot simply collapse to one call — but they *can*
share a parameterized core. Both `test_lowp_gemm.mojo::test_encode_near_max_no_nan_pattern`
and `test_nvfp4_quant.mojo::test_e4m3_saturation` independently re-derived the *same*
"probe's `e_biased>=15 → m3=7` saturation bug decodes to 240 not 448" lesson — clear
evidence the drift is real and each side re-litigated it.

Consolidation shape (post-merge): move the identical `decode_e4m3` to a single home
(`llmm/lowp.mojo`'s `_fp8_decode[4,3,7]`) and have `nvfp4_quant.mojo` import it;
generalize `_fp8_encode_rne` to take a comptime `tie_mode ∈ {EVEN, AWAY}` and a
`nan_policy ∈ {SATURATE, EMIT}` so nvfp4's `encode_e4m3` becomes
`_fp8_encode[4,3,7,E4M3_MAX, tie=AWAY, nan=EMIT]`. Keep the two public names as thin
wrappers (they carry meaning). Risk: MODERATE — both e4m3 test suites must stay green
bit-for-bit; the tie-mode/NaN parameters must be exercised by both. **Do NOT force
both encoders to one rounding rule** — that is a correctness regression against one of
the two oracles.

WHY WAIT: both files are in the fp4 merge churn set; `nvfp4_quant.mojo` arrived via the
fp4 lineage and the T1 agent is still editing that area. Doing this now = a third editor
on `lowp.mojo`/`nvfp4_quant.mojo`.

### F2 — the cuBLASLt descriptor dance: three near-identical vendor bodies  (HIGH severity · WAIT-FOR-fp4-merge)

`llmm/matmul.mojo` has (at merged/branch state) three functions that each run the same
create-desc → set-transA/B → create-4-layouts → preference → heuristic → matmul →
destroy-6-handles boilerplate (~90–200 lines each):

- `_matmul_cublaslt` (line 203, bf16/fp32, caller-selectable transA/transB, epilogue).
- `_matmul_cublaslt_fp8` (line 575, TN-fixed, A/B scale pointers, bf16 out).
- `_matmul_cublaslt_fp4` (branch `48cfd73` line 1024 — **pending surface**, not in this
  HEAD; block-scale pointers + `_nvfp4_post_scale` correction).

The descriptor/layout/preference/heuristic/destroy scaffolding is copy-pasted three
ways; only (a) the dtype→`_lt_dt` mapping, (b) which `*_SCALE_POINTER` attrs are set,
(c) TN-fixed vs caller-selectable, and (d) epilogue vs none actually differ.

Verdict: real duplication, but this is **vendor-API boilerplate where over-DRYing hurts
readability** (a single mega-generic cuBLASLt wrapper with 6 comptime switches is harder
to audit than three explicit bodies — and this code is security-adjacent unsafe-pointer
casting). Recommended *partial* extraction only: factor the three genuinely mechanical
sub-steps into helpers — `_lt_make_layout(dt, rows, cols, ld)` (create+error-check one
layout, replacing 4 inline blocks × 3 functions = 12 copies), `_lt_pick_algo(...)`
(preference + heuristic + `cnt==0` raise), and `_lt_destroy(desc, a_l, b_l, c_l, d_l,
pref)`. Leave the three top-level functions as distinct orchestrators. `_lt_set_op`
(line 181) and `_lt_dt` (line 161) already show this factoring is the house style.

WHY WAIT: `matmul.mojo` is the all-four-branches conflict file and `_matmul_cublaslt_fp4`
is still landing. Extract after the fp4 merge settles, in the same pass as F3.

### F3 — fp8 host-scale / device-scale "twin" functions  (MEDIUM-HIGH severity · WAIT-FOR-fp4-merge)

The host-`Float32`-scale vs device-`fp32*`-scale split spawned six near-clone pairs:

- `llmm/lowp.mojo`: `quantize` (450) / `quantize_devscale` (535);
  `quantize_transpose` (589) / `quantize_transpose_devscale` (657) — the `_devscale`
  twin differs only in `scale: Float32` → `scale_ptr[0]` inside the kernel.
- `llmm/matmul.mojo`: `lowp_gemm` (779) / `lowp_gemm_devscale` (881) — identical
  quantize-then-GEMM structure, host vs device scale.

The docstrings are explicit: the host-scale variants exist **only** for the Chunk B/D
unit-test gate (`tests/test_lowp_gemm.mojo` supplies a host-computed scale); every
**production** call site (`matmul_fwd_lowp`, the bwd lowp path) uses the `_devscale`
form. So ~3 of the 6 functions are test-only scaffolding.

Consolidation shape (post-merge): delete the three host-scale variants; have the tests
upload their host scalar into a 1-element device buffer (they already create device
buffers) and call the `_devscale` form. That removes `quantize`, `quantize_transpose`,
and `lowp_gemm` outright (~120 LOC) with zero production impact. Risk: LOW-MODERATE —
only test call sites change; behavior is identical. Note FP4 has **no** twin problem
(NVFP4 computes its scale in-kernel, one `nvfp4_quantize`), so this is purely an fp8
cleanup.

WHY WAIT: touches `lowp.mojo` + `matmul.mojo`, both in the fp4 merge path. Safe on the
fp8 axis (goal2/fp8-training is complete bar the b98ce9f verification commit), so this
can go in the **first** post-fp4-merge cleanup.

### F4 — persistent device-buffer process-global: 3× identical boilerplate  (MEDIUM severity · WAIT-FOR-{dbias-vectorize, ln-bwd-fusion})

The "allocate-once, heap-held via `KGEN_CompilerRT_InsertGlobal`, keyed by
`{ctx.id()}`" idiom is copy-pasted three times, byte-for-byte modulo dtype/size/memset:

- `_cublaslt_workspace` (`matmul.mojo` 141, uint8, 32 MB, no memset).
- `_dbias_scratch` (`matmul.mojo` 1519, fp32, 65536, memset 0).
- `_ln_dparam_scratch` (`layernorm.mojo` 1670, fp32, 2·cap, memset 0).

Each is ~18 lines of the same `_get_global_or_null` / `alloc` / `init_pointee_move` /
`external_call` / `rebind` dance. (Note: the fp8/fp4 GEMM scratch buffers use a
*different*, per-call `enqueue_create_buffer` pattern — not this idiom — so they are
correctly out of scope here.)

Consolidation shape: one generic
`persistent_device_buffer[dtype](ctx, name_suffix, count, *, zero=False)
-> MutKernelPtr[dtype]` in `llmm/memory.mojo` (where the pointer-bridge helpers already
live). All three call sites collapse to one line. Risk: LOW mechanically.

WHY WAIT: `_dbias_scratch`/`_dbias_accum_gpu` are being edited by the in-flight
`goal1/dbias-vectorize`; `_ln_dparam_scratch` is being rewritten by the quarantined
`goal1/ln-bwd-fusion`. Extracting now collides with both. Do it after those two land
(or extract just the helper into `memory.mojo` now and retrofit call sites as each
branch lands — but the clean single-PR version waits).

### F5 — test helpers: `_host_gemm_ref` + comparison utils duplicated across lowp tests  (MEDIUM severity · **SAFE-NOW**, partial)

Confirmed duplication:

- `_host_gemm_ref` — generic `[transpose_a, transpose_b]` version in
  `tests/test_lowp_gemm.mojo` (line 63); a forward-only hardcoded copy in
  `tests/test_lowp_gemm_fp4.mojo` (line 77) whose **own comment admits** "ported here
  rather than imported since it is file-local there." The generic version subsumes the
  fp4 one exactly (fp4 = `transpose_a=False, transpose_b=True`).
- Comparison/error utilities re-implemented per file: `_rel_l2`
  (`test_lowp_gemm_fp4`), `_cosine_and_rel_l2` (`test_lowp_bwd`), plus bf16-fill
  helpers (`_random_bf16`/`_zeros_bf16`/`_clone_bf16` in `test_lowp_bwd`;
  `_pseudo_gaussian_fill` in `test_lowp_gemm_fp4`; `_make_bf16_tensor` in `test_amax`).
- The two e4m3 codecs (F1) are each separately imported and tested — `test_lowp_gemm`
  imports from `lowp`, `test_nvfp4_quant` from `nvfp4_quant` — which is correct as long
  as F1 keeps two encoders, but the *tests* of the shared decode could share a fixture.

Consolidation shape: new `tests/_lowp_test_common.mojo` (precedent:
`tests/_rng_sr_gpu_kernels.mojo`, `tests/_tokenizer_bridge.mojo`) exporting the generic
`_host_gemm_ref`, `rel_l2`, `cosine_and_rel_l2`, and the bf16-fill helpers. Refactor the
HEAD-side tests (`test_lowp_gemm`, `test_lowp_bwd`, `test_amax`) to import them.

SAFE-NOW because: it creates a **new** file and edits only test files (not the hot merge
surface). The fp4 test (`test_lowp_gemm_fp4.mojo`, added by the fp4 branch) adopts the
shared helper as a one-line follow-up **after** the fp4 merge — do not edit that file
now (it is on the in-flight branch). Fix spec below.

### F6 — Makefile `build-{bf16,fp8}` (+ pending `build-fp4`) target clones  (LOW-MEDIUM severity · WAIT-FOR-fp4-merge, then optional)

`build`, `build-bf16`, `build-fp8` and their `build-profile-*` / `build-infer-*` and
`train-*` siblings are near-identical recipes differing only in the `-D` flag
(`` , `-D LLMM_BF16=1`, `-D LLMM_PRECISION=fp8`). The T1 agent will add a fourth column
(`build-fp4`, `-D LLMM_PRECISION=fp4`). That is 3×3 = 9 nearly-identical `mojo build`
recipes plus their `.PHONY`/help lines.

Verdict: **borderline — lean toward LEAVE AS-IS.** GNU Make static/pattern rules *can*
collapse these (e.g. a `build/train_gpt2_%: ` pattern with a per-precision flag lookup),
but Make pattern rules over phony convenience targets with per-target help text and
runner-script wiring are notoriously hard to read and debug, and this Makefile's house
style is explicit-and-commented (every target carries a rationale comment). The
duplication is shallow (one flag) and stable. If consolidated at all, do it *after* the
fp4 column lands so the pattern is designed against the final 4-wide set, not retrofitted.
This is make-idiomatic repetition, not a logic-duplication hazard.

### F7 — block-counter accumulate+finalize idiom (dbias vs ln-dparam)  (NOTE ONLY · WAIT-FOR-ln-bwd-fusion redesign)

`matmul.mojo`'s `_dbias_accum_gpu`/`_dbias_finalize_gpu` (1540/1568) and
`layernorm.mojo`'s `_ln_dparam_accum_gpu`/`_ln_dparam_finalize_gpu` (1691/1727) share
the "one-thread-per-column, grid.y row-blocks → partials in persistent scratch →
finalize kernel reduces + re-zeros" shape. But they diverge substantively: dbias writes
contention-free `[row_blocks, OC]` partials (no atomics); ln uses `Atomic.fetch_add`
into a `[dgamma | dbeta]` split buffer. The reduction/finalize bodies are **not**
mechanically identical.

Per the campaign directive, extraction here **waits for the `goal1/ln-bwd-fusion`
redesign verdict** — that branch is actively reshaping the ln-dparam path, so the target
shape is unknown. Note only; no fix spec.

---

## SAFE-NOW fix specs (sized for a Sonnet fixer, minimal context)

### Fix spec F5 (the only fully safe-now code change)

Goal: kill the `_host_gemm_ref` + comparison-util duplication among the **HEAD-side**
lowp tests. Do NOT touch `tests/test_lowp_gemm_fp4.mojo` (in-flight fp4 branch).

1. Create `tests/_lowp_test_common.mojo` with:
   - `_host_gemm_ref[transpose_a: Bool, transpose_b: Bool](a_host, b_host, out_host,
     m, n, k)` — copy verbatim the generic body from
     `tests/test_lowp_gemm.mojo` lines 63–90 (col-major-D `out[j*m+i]` convention).
   - `rel_l2(got, ref, n) -> Float32` — copy from `test_lowp_gemm_fp4.mojo` `_rel_l2`.
   - `cosine_and_rel_l2(...)` — lift from `test_lowp_bwd.mojo` `_cosine_and_rel_l2`.
   - bf16-fill helpers `random_bf16`/`zeros_bf16`/`clone_bf16` — lift from
     `test_lowp_bwd.mojo`.
   Match the file header/format of `tests/_rng_sr_gpu_kernels.mojo` so `make lint-mojo`
   (`mojo format --check`) passes.
2. In `tests/test_lowp_gemm.mojo`: delete the local `_host_gemm_ref`; add
   `from tests._lowp_test_common import _host_gemm_ref` (verify the import path style
   against how `test_rng_sr.mojo` imports `tests/_rng_sr_gpu_kernels.mojo`).
3. In `tests/test_lowp_bwd.mojo` and `tests/test_amax.mojo`: replace the local
   `_cosine_and_rel_l2`/bf16-fill helpers with imports from the new module.
4. Run `make lint-mojo` and `pixi run mojo run -I . tests/test_lowp_gemm.mojo`,
   `tests/test_lowp_bwd.mojo`, `tests/test_amax.mojo` — all must stay green.
5. Leave a one-line TODO in the new module: "`test_lowp_gemm_fp4.mojo` should adopt
   `_host_gemm_ref` (transpose_a=False, transpose_b=True) after the fp4 merge."

### Fix specs F1–F4, F6 — pre-written for the post-fp4-merge cleanup pass

Run these **only after** `goal3b/fp4-training` (incl. T1's `matmul_fwd_fp4`/wiring) is
merged to `main`, on a single settled tree, in this order (they touch overlapping files):

- **F4 first** (smallest, and unblocks the others' file): add
  `persistent_device_buffer[dtype](ctx, name_suffix, count, *, zero=False)` to
  `llmm/memory.mojo`; rewrite `_cublaslt_workspace`, `_dbias_scratch`,
  `_ln_dparam_scratch` as one-liners. (Also gate on `goal1/dbias-vectorize` +
  `goal1/ln-bwd-fusion` having landed.)
- **F3**: delete host-scale `quantize`/`quantize_transpose`/`lowp_gemm`; point their
  tests at the `_devscale` forms via a 1-element device scale buffer.
- **F1**: unify `decode_e4m3` (identical) into `lowp._fp8_decode`; parameterize
  `_fp8_encode_rne` with `tie_mode`/`nan_policy`; nvfp4 imports both. Keep both public
  encoder names. Bit-for-bit re-run both e4m3 test suites.
- **F2**: extract `_lt_make_layout` / `_lt_pick_algo` / `_lt_destroy`; leave the three
  `_matmul_cublaslt*` orchestrators distinct.
- **F6** (optional): only if a reviewer wants it, against the final 4-wide target set.

---

## Merge-topology recommendation

Merge base for all four merge-bound branches is `cf40d67` (current `main` tip).
`llmm/matmul.mojo` is the universal conflict file (touched by all four); `Makefile` and
`docs/ai/ai_assisted_optimizations_and_benchmarks.md` are touched by the goal1 pair +
fp8. Recommended order — smallest matmul.mojo churn first, so each later merge rebases
onto an already-integrated smaller delta rather than the reverse:

1. **`goal1/fp32-parity`** (matmul.mojo +14/−2 — trivial). Land first; it is the
   smallest and also carries the TF32 correctness fix the perf docs depend on.
2. **`goal1/dbias-fusion`** (+170/−30; already green). Small, self-contained; shares
   only `Makefile`/attention/vendor with #1 (minor conflicts).
3. **`goal2/fp8-training`** (+917/−2) — **but first fast-forward this HEAD's fp8 to its
   tip `b98ce9f`**, which is 1 commit ahead of what HEAD merged (see orphan note). That
   commit is verification/tooling only (grad-gate `compare_grad_dumps.py`,
   `make verify-fp8-grads`, CHANGELOG fix, +210 lines across two docs) — no kernel code,
   low conflict risk beyond `Makefile`/the ai-assisted doc it shares with #1/#2.
4. **`goal3b/fp4-training`** (+1449/−3, the largest) **last, after T1 finishes**
   (`matmul_fwd_fp4` + train call sites + `build-fp4` are not yet on tip `48cfd73`).
   Expect the biggest `matmul.mojo` conflict here — it interleaves `_matmul_cublaslt_fp4`
   / `lowp_gemm_fp4` / `_nvfp4_post_scale` among the fp8 GEMM functions #3 just added.
5. **Then** run the F1–F4 consolidation pass on the settled tree.

Expected conflict hotspots: `llmm/matmul.mojo` (all four — resolve by keeping the
function-block ordering: bf16 cuBLASLt → fp8 → fp4 → fwd/bwd siblings), `Makefile` (the
`build-*`/help/`.PHONY` lists — F6's territory; resolve by concatenating precision
columns), and `docs/ai/ai_assisted_optimizations_and_benchmarks.md` (append-only
sections, low risk).

### Orphan / branch-hygiene notes (from the topology sub-audit)

- **One true code-free orphan**: `26b45a4` on `goal3/fp4-research`
  ("Update fp4_readiness_summary.md: probe CONFIRMED") — docs-only (+16/−12), not
  contained in any surviving branch. Cherry-pick that one doc edit before deleting
  `goal3/fp4-research`, or it is lost.
- All other historical branches (`goal2/fp8-chunk-{a,B,C,D,E,G}`, `goal1/tf32-fix`,
  `goal2-fp8-chunkD-local`) are **fully contained** in the survivor set — safe to delete,
  no orphans.
- `goal2/fp8-training` tip `b98ce9f` is 1 commit ahead of this HEAD's merge of it (the
  verification commit above) — fold it in at step 3.

---

## What is genuinely NOT worth consolidating (over-DRYing is also a failure mode)

- **F2 full merge** — collapsing the three `_matmul_cublaslt*` bodies into one generic
  wrapper with 6 comptime switches would make security-sensitive unsafe-pointer vendor
  code *harder* to audit than three explicit orchestrators. Extract only the mechanical
  sub-helpers.
- **F6 Makefile pattern rules** — the duplication is one `-D` flag per target; Make
  pattern rules over phony targets with per-target help/runner wiring are less readable
  than the current explicit-and-commented recipes. Repetition is the right call.
- **F1 encoder unification into ONE rule** — the RNE-ties-even (host-cast oracle) vs
  ties-away (cuBLAS/probe oracle) split is load-bearing; share the *core*, keep both
  rounding rules. Forcing one rule is a correctness regression.
- **The two amax reductions** — `amax.mojo::compute_amax` (per-tensor, fp8, NaN-infectious
  with delayed-scaling history) and `nvfp4_quant.mojo::nvfp4_compute_tensor_scale`
  (per-tensor scale computed fresh in-kernel, no history) look similar but encode a
  deliberate design decision (FP4 does not use `AmaxState` at all — documented at length
  in both `FP4_SPEC` and `compute_amax`'s docstrings). They are different machinery for a
  reason; do not merge.
- **The fwd/bwd `*_lowp` sibling functions** (`matmul_fwd`/`matmul_fwd_lowp`,
  `matmul_bwd`/`matmul_bwd_lowp`) — the bias/GELU epilogue is **already factored**
  (both call the shared `bias_gelu_fwd` / `matmul_bias_bwd`); the siblings differ only
  in the GEMM core (vendor bf16 vs quantize+fp8). This is the accepted, well-factored
  convention — no action.
- **`rng_device.mojo`** — single RNG, single SR-bit-dither implementation, host==device
  by construction (no host-reference twin). Exemplary; nothing to do.

---

## Docs overlap (for the separate docs pass — task 6; flagged here, not fixed)

All SAFE-NOW (docs edits don't touch the hot code merge surface), **except** where a
doc is also edited by `goal2/fp8-training`'s pending `b98ce9f` (`low_precision_gotchas.md`,
`ai_assisted_optimizations_and_benchmarks.md`) — sequence those after step 3 of the merge.

**Act on these (true redundancy / active contradiction):**

- **D1 — `low_precision_gotchas.md` A1/A2 re-state design-doc content near-verbatim.**
  A1 ("FP8 transient, storage stays bf16") duplicates `fp8_training_design.md` §1.1;
  A2 ("one `LLMM_PRECISION` axis", incl. the `_resolve_precision()` code block)
  duplicates §2. Both AGREE and already cite the source. Fix: collapse A1/A2 to a
  one-line source pointer + the only non-duplicated bit (A2's "awaiting Chunk B" status).
- **D2 — SR + device-RNG gap described three times.** `fp8_training_design.md` §3 Chunk G
  (build spec), `fp4_training_recipes_research.md` "Current state"/§4 (numerics motive),
  `fp4_readiness_summary.md` §2 (missing-infra list) — all AGREE, all point at
  `adamw.mojo:~141` + counter-based RNG. Fix: designate §3 Chunk G the single home;
  the other two link to it.
- **D3 (highest value — active contradiction) — stale FP4 probe status.**
  `fp4_modular_support_research.md` §6 and `low_precision_gotchas.md` C2 both report the
  cuBLASLt sm_120→sm_121 NVFP4 dispatch probe **CONFIRMED** (rel-L2 0.1445); but
  `fp4_readiness_summary.md` TL;DR/§1b still says the probe *"is running now / in flight
  / pending."* A reader would act on the wrong status. **This drift is already fixed by
  the un-merged orphan commit `26b45a4`** ("Update fp4_readiness_summary.md: probe
  CONFIRMED") flagged in the branch-hygiene notes — cherry-pick that commit and D3
  resolves itself.
- **D4 (contradiction, cross-ref not merge) — `pop.cast` fp8→GPU.**
  `low_precision_gotchas.md` T1 states elementwise fp8 cast kernels do **not** compile on
  this GPU (must use raw bit patterns) — but `fp8_training_design.md` §3 still describes
  exactly such `quantize`/`dequantize_accum` cast kernels as the plan (it predates the
  probe). Add a forward-reference from §3 to T1. NOTE: the *code* already resolved this
  (`lowp.mojo`/`nvfp4_quant.mojo` both do manual bit-pattern encode) — only the design
  doc's prose is stale. The "no pop.cast" gotcha itself is **single-sourced** (T1 only),
  so it is not a duplication.

**Leave as-is (healthy different-altitude coverage — do NOT merge):**

- NVFP4 format description across 4 docs (recipe numerics vs support/decode vs impl
  gotcha vs summary) — three are genuinely different lenses; only the
  summary↔recipe pair is redundant → cross-reference the summary, leave the rest.
- e4m3 saturation/rounding: design §1.3 *specifies*, gotchas E5 *measures* — complementary.
- MAX sm_100/tcgen05-only fp4/fp8 kernels (modular#5707) — cited at 3 altitudes; fine.
- FP4 recipe items (selective hi-precision layers, RHT Wgrad-only, late-training BF16
  switch) duplicated summary↔recipe — that IS the summary's stated job; leave separate.
- GB10 throughput ladder / MAMF numbers — single-sourced; no cross-doc conflict.
- Benchmark-hygiene checklist (gotchas TF3) vs the incident log it distills
  (`ai_assisted_optimizations_and_benchmarks.md`) — useful checklist↔log split; cross-ref only.

---

_Written with AI assistance (Claude Code / Opus agent), directed by Evan Owen._
