# MAX CPU custom-op execute SIGSEGV under a concurrent trainer (2026-07-24)

**Host:** workstation-max (192-core x86, 8x RTX PRO 6000 Blackwell Max-Q).
**Toolchain:** MAX/Mojo `26.5.0.dev2026071306` (pixi default and cuda envs —
same build in both).
**Symptom:** every `make test-python` (CPU pytest suite, MAX-bridge custom
ops) dies with SIGSEGV (exit 139); with `pytest -n auto`, 109 of 248 tests
fail as their workers crash.

## Verdict

Upstream MAX runtime bug, not an llm.mojo kernel bug. Executing any
`@compiler.register`'d custom op on the **CPU** device calls a **NULL
function pointer** from JIT-compiled code — on many AsyncRT worker threads
simultaneously — **if and only if another MAX/AsyncRT process (a multi-rank
trainer) is active on the box**. The same models execute correctly when the
box is quiet, and the GPU device path is unaffected throughout (the cuda-env
suite passes while a 7-rank trainer runs).

## Evidence

- Crash site: `model.execute(...)` (`max/engine/api.py::_Model_execute`).
  Compile (`Graph(...)` + `session.load`) succeeds. A `SA_SIGINFO` handler
  captured the fault: `signal 11, addr (nil)`, top frame in anonymous JIT
  memory, hit concurrently by many runtime threads — a call through a null
  op/work pointer, not a data access.
- Trainer-activity correlation (the load-bearing observation): the 774M fp8
  trainer finished 03:17; a fresh fp4 trainer started ~03:30. Every repro
  variant passed in the 03:19–03:29 idle gap (10+ consecutive passes across
  4 scripts) and crashed outside it (30+ crashes) — including reruns of
  byte-identical scripts that had just "proven" a fix. Trainer-active
  crashes reproduce with `CUDA_VISIBLE_DEVICES=` (no GPU access), so the
  interference is not GPU-side.
- Under gdb (ptrace serializes thread startup) the same crashing setup
  passes even while the trainer runs.
- With `MODULAR_THREAD_BUSY_WAIT_US` set, the failure turns into
  `LLVM ERROR: Init::getOrCreateContext() requested an M::Context with
  different Init::Options ...` (abort) — the process double-initializes the
  Modular runtime context with mismatched options, consistent with a
  runtime-init race armed by a sibling process.
- Ruled out by direct experiment (each "fix" below also later crashed once
  the trainer resumed — beware short pass-streaks): prebuilt `.mojoc` vs
  source-dir `custom_extensions`, MEF cache on/off, pytest capture /
  assertion-rewrite / faulthandler / xdist plugins, core count and CPU
  affinity (`taskset -c 0` still crashes), ASLR (`setarch -R`), global
  `LD_LIBRARY_PATH`, `MODULAR_ENABLE_AFFINITY`, isolated
  `MODULAR_DERIVED_PATH`, kernel-side `sync_parallelize` (an inline
  sequential variant still crashed → the null call is in MAX's own
  dispatch, not our fan-out).
- Related prior art: the fp8 multi-rank NaN
  (`docs/ai/fp8_multirank_nan_investigation.md`) was likewise an upstream
  launch-machinery race "armed by a sibling context's existence". This is
  the CPU-side sibling: cross-process rather than cross-context.

## Repro (for the upstream report)

1. Start any multi-rank trainer (`build/train_gpt2* -pn 7 ...`) on the box.
2. `pixi run pytest tests/test_softmax_equivalence.py::test_padding_left_untouched -x`
   → SIGSEGV in `model.execute` (any bridge kernel; adamw shows the same).
3. Stop the trainer, rerun → passes.

## In-repo hardening (shipped)

- `tests/_max_bridge.py`: MEF export moved to after the model's first
  successful execute, so a compile that crashes at execute never persists a
  questionable artifact into the cache.
- `Makefile test-python`: warns when a `train_gpt2` process is running (the
  suite cannot pass until it exits) and retries failures twice via
  `--last-failed` to absorb one-off flakes on a quiet box. `-n auto`
  parallelism is unchanged.
- The CUDA suite (`make test-python-cuda`) remains the reliable gate on this
  box and is unaffected by trainer activity.

## AI use statement

Written with AI assistance (Claude Code / Fable agent), directed by Evan Owen.
