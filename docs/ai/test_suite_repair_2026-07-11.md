# Test-suite repair campaign, 2026-07-11

A root-cause writeup of the four independent failures that were breaking or
hanging the test suite on 2026-07-11, found while investigating a single
symptom: "`make test` is stuck on `tests/test_hadamard.mojo`". None of the
four turned out to be a slow test. The suite ended the day fully green —
`make test-mojo` 17/17 files in ~78–102 s wall, the CUDA pytest arm 248
passed / 0 failed in ~3 minutes.

Environment for everything below: GB10 (NVIDIA, aarch64), Mojo
`26.5.0.dev2026071006` (the toolchain the auto-reprovision on the pixi
version bump installed).

---

## 1. The "stuck test": a second `DeviceContext()` deadlocks the process

**Symptom.** `pixi run mojo run -I . tests/test_hadamard.mojo` sat 8+ minutes
with 0% CPU, 0% GPU utilization, ~177 MiB resident on the GPU, and every
thread parked in `futex_wait`. Not slow JIT compilation (that burns CPU) — a
runtime deadlock.

**Root cause.** Constructing a *second* fresh `DeviceContext()` in the same
process — after a prior context has launched a kernel and done a
device→host readback (`enqueue_copy_to`) — never returns on this toolchain.
Bisected empirically:

- The hadamard kernel itself is innocent: it has no `barrier()`/shared
  memory, and replacing its body with a trivial passthrough still hangs.
- The minimal trigger is: kernel launch + `copy_to` readback in context A,
  then construct context B. Sharing one context makes everything pass.
- `test_hadamard.mojo` built a fresh `var ctx = DeviceContext()` in each of
  its three GPU tests, so it hung entering the second one.
  `test_nvfp4_quant.mojo` and `test_rng_sr.mojo` had the same pattern and
  deadlocked identically (600 s timeouts). `test_amax.mojo` constructs ~20
  fresh contexts and happens to dodge it — the readback-then-new-context
  sequence is what matters, not the context count.
- Production is unaffected: the training loop uses a single persistent
  `DeviceContext` for its whole life.

**Fix.** `tests/_gpu_test_common.mojo` (new, underscore-prefixed so the
Makefile's `tests/test_*.mojo` glob never executes it) provides
`shared_gpu_ctx()`: one process-global context stored via
`_get_global_or_null` + `KGEN_CompilerRT_InsertGlobal`, mirroring the
`persistent_device_buffer` idiom in `llmm/memory.mojo`. All three affected
test files now use it; each runs in ~2–6 s wall. This is a workaround for a
likely dev-toolchain bug — worth re-testing on the next Mojo bump.

## 2. Toolchain-bump compile breakage (7 test files)

The same Mojo bump changed the cuBLASLt binding: `cublasLtMatmul` now takes
the C-matrix pointer as `Optional[UnsafePointer[NoneType, ImmutAnyOrigin]]`.
The three call sites in `llmm/matmul.mojo` (`_matmul_cublaslt`,
`_matmul_cublaslt_fp8`, `_matmul_cublaslt_fp4`) passed a mutable-origin
pointer and stopped compiling, which took down every test that imports the
matmul path (5 lowp/matmul mojo test files plus `test_zero_equivalence.mojo`
via `train_gpt2.mojo`) *and* the whole pytest bridge (`llmm.mojoc` would not
build, so the CUDA pytest arm was a wall of failures). Fixed by passing the
same pointer `.as_immutable().as_unsafe_any_origin()` — semantics unchanged.

Separately, stricter origin checking broke `tests/test_zero.mojo` (three
`.unsafe_ptr()` assignments into `MutUntrackedOrigin` slots) and, once that
was fixed, unmasked the identical pre-existing pattern in `llmm/zero.mojo`'s
`gather()` CPU path (line ~551, `_register_and_sync` with a hardcoded
`MutUntrackedOrigin` output origin). Both fixed with explicit
`.unsafe_origin_cast[MutUntrackedOrigin]()` at the call sites.

## 3. dbias vectorization regression: odd `out_channels` rejected

**Symptom.** 5 CUDA pytest failures, all `[fp32_odd_sizes]` variants of
`test_matmul_equivalence.py` backward tests: `matmul_bias_bwd: out_channels
(42) must be a multiple of the dbias vector width (4)` raised at
graph-execution time.

**Root cause.** Commit `69d7a15` ("Vectorize matmul dbias accum kernel to
128-bit loads (NVIDIA)", 2026-07-10) gave each thread a contiguous
`width`-wide run of output columns, computed the group count as a truncating
`oc // width`, and guarded the launch with a hard `raise` on `oc % width !=
0`. The `fp32_odd_sizes` case (`output_channels=42`) has existed since the
original matmul commit (`202fa80`, 2026-06-12); the pre-vectorization
two-launch kernels handled arbitrary OC via a per-column bounds guard. A
one-day-old regression, not a toolchain issue — it just couldn't be observed
until the compile breakage above was fixed.

**Fix.** `ceildiv(oc, width)` group count, and `_dbias_fused_gpu` splits on
`col + width <= out_channels`: aligned threads keep the exact branch-free
128-bit path (bit-identical results), the single ragged tail thread takes a
scalar loop over the remaining `oc % width` columns. The tail branch is
required for correctness, not just coverage: a width-wide store there would
run past column `oc` and corrupt the next row-block's region of the shared
`[row_blocks, OC]` scratch.

## 4. fp32 attention "failures": TF32, not a bug

**Symptom.** 7 CUDA pytest failures in `test_attention_equivalence.py`, all
fp32 (bf16 twins passed), drift ~3e-4..1.6e-3 against `atol=2e-5`.

**Root cause.** These are not regressions and not kernel bugs. The failing
outputs are exactly TF32-quantized (e.g. `1.1591796875` — a 10-bit-mantissa
step — vs the fp32 reference `1.1594818830…`), because the fp32 attention
GEMMs run on TF32 tensor cores: `USE_TF32` has been default-on since
`c27a1f9` to match llm.c's fp32 arm. Two proofs:

- The just-pulled Metal commit `aaf0fa1` was the prime suspect and is
  exonerated: its `llmm/attention.mojo` change is inside `comptime if
  HAS_METAL` (compile-time dead on CUDA), and the failures reproduce with
  the pre-pull file.
- With the attention GEMM's compute type temporarily forced to true IEEE
  fp32 (`COMPUTE_32F`), all 7 tests pass at the strict `atol=2e-5`.

These cases were "green before" only in the default suite, where
`prefer_accelerator=False` sends fp32 cases to the CPU; the
`MAX_USE_ACCELERATOR=1` CUDA arm had likely never been green for them.

**Fix.** A TF32-sized tolerance (`atol=3e-3`, sized from the worst measured
divergence, bwd `fp32_gpt2` dq at 1.6e-3) applied *only* when a case
actually executes on an NVIDIA accelerator
(`_runs_on_tf32_accelerator()` in the test). The strict fp32 gate stays
fully in force on the CPU and Metal paths. A blanket widening was
explicitly rejected: it would be a ~100× loosening that masks
small-magnitude backward bugs — the exact weak-gate failure mode that
previously buried three real backward bugs behind a flat `atol=2.0` (see
`compare_grad_dumps.py`'s history).

## Harness hardening and a profiling gotcha

- `make test-mojo` now wraps each per-file `mojo run` in `timeout 600`,
  captures the exit code explicitly, prints a per-file wall-time line
  (`==> tests/foo.mojo done in Ns (exit 0)`), and keeps the fail-aggregation
  semantics. A future deadlock costs one failed file, not a wedged `make
  test`.
- **Do not trust `TestSuite`'s per-test `[ seconds ]` output on this
  toolchain.** It printed 945 s for a test inside a process whose total wall
  time was 2.2 s (`/usr/bin/time`). Several plausible-looking "optimization
  targets" in this campaign were phantoms created by that timer; per-file
  wall clock is the only meaningful signal until a toolchain fix lands.
- After the repairs, no per-file test optimization was warranted: every mojo
  test file runs in 0.5–8 s wall, and the suite total is ~78–102 s.

## Verified end state

- `make test-mojo`: 17/17 files, 0 failing sub-tests, `TOTAL:78.29` s
  (second run 102.34 s), every file exit 0 with its own timing line.
- `MAX_USE_ACCELERATOR=1 pixi run -e cuda pytest tests/ -q`: **248 passed, 0
  failed in 180.60 s**.
- `git diff` scope at commit time: `llmm/matmul.mojo`, `llmm/zero.mojo`,
  `Makefile`, `tests/_gpu_test_common.mojo` (new), `tests/test_hadamard.mojo`,
  `tests/test_nvfp4_quant.mojo`, `tests/test_rng_sr.mojo`,
  `tests/test_zero.mojo`, `tests/test_attention_equivalence.py`, `pixi.lock`.

## Method note

The campaign ran as three multi-agent workflows (~29 subagents total): a
sweep (one deep-diagnosis agent on the hang, 18 read-only per-file/harness
analyzers, one serial GPU-exclusive timing agent), a repair fan-out (five
fixers on disjoint files + a full-suite verifier), and a root-cause pass
(two independent investigators for §3 and §4 + verification). Two findings
came from agents catching other agents' work: the `test_zero.mojo` fixer
refused to touch `llmm/zero.mojo` outside its assignment and reported the
unmasked bug instead, and the deadlock-fix agent found and fixed a
nonexistent-API call (`unsafe_write` on `UnsafePointer`) in the original
diagnosis agent's helper.

## AI use statement

Written with AI assistance (Claude Code / Claude Fable 5, multi-agent
workflows), directed by Evan Owen.
