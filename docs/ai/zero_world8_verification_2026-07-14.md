# ZeRO verification & repair at WORLD_SIZE=8 (2026-07-14)

A verification-and-repair campaign for ZeRO stages 0/1/2/3 at world size 8 on
an 8× NVIDIA RTX PRO 6000 Blackwell box (96 GB each). The headline results:

- **The ZeRO sharded-optimizer math is correct** and is now verified end-to-end
  on CPU at both WORLD_SIZE=2 and **WORLD_SIZE=8** for stages 1/2/3
  (`tests/test_zero_equivalence.mojo`, all green).
- Getting there required fixing **two real, previously-hidden bugs** (a test
  read helper that freed its buffer mid-copy, and a segfault in the sharded
  AdamW update from an unaligned shard offset) plus one robustness fix
  (single-GPU `ZeroContext` did needless P2P setup).
- **Multi-GPU ZeRO on the GPU target does not train** — every stage (including
  stage 0 / DDP) crashes at the first collective with
  `CUDA_ERROR_INVALID_VALUE`. This is **root-caused** (below) to a
  calling-convention mismatch between `llmm/zero.mojo` and the Modular `comm`
  collectives on the current toolchain, and it needs a real multi-device
  rewrite — it is **not** fixed in this campaign. The finding is proven with an
  isolated probe, not inferred.

## Per-stage verdict

| Path | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|------|---------|---------|---------|---------|
| CPU equivalence, W2 (`test_zero_equivalence`) | baseline | **PASS (fixed)** | **PASS (fixed)** | **PASS (fixed)** |
| CPU equivalence, W8 (`test_zero_equivalence`) | baseline | **PASS (new)** | **PASS (new)** | **PASS (new)** |
| CPU collectives unit (`test_zero`, W4) | PASS | PASS | PASS | — |
| GPU training, W8 (real `train_gpt2`) | **CRASH** | **CRASH** | **CRASH** | **CRASH** |

"PASS (fixed)" = was red (crashing/erroring) at the start of the campaign and is
green now. "CRASH" = `CUDA_ERROR_INVALID_VALUE` at the first collective; same
crash for fp32 and bf16.

Verified **by actually running** on this box: all CPU test rows; the GPU crash
(fp32 and bf16, world sizes 2/4/8); the isolated `comm.allreduce` probe; the
per-stage memory curve. Everything labelled "root cause" below was reproduced,
not read off the source alone.

---

## Fix 1 — `test_zero_equivalence` read helper freed its buffer mid-copy

**Symptom.** All three stage tests failed deterministically in ~30 ms with
`"Bad magic model file"`, before any ZeRO code ran. (First observed on this
linux-64 x86_64 host; the `linux-64` pixi platform was only added in HEAD.)

**Root cause.** `read_to_dtype_pointer` read the debug-state file into a
`read_bytes` `List`, then `memcpy`'d from `bytes_data.unsafe_ptr()` after
rebinding that pointer to `MutUntrackedOrigin`. Dropping the origin drops the
`List`'s lifetime tracking, so `bytes_data` was freed at its last *tracked* use
— the `.unsafe_ptr()` call — **before** the `memcpy` executed. The allocator
then reused/clobbered the first element, so the magic read back as garbage while
later fields (B, T at offsets 2,3) survived. Reproduced in isolation: byte-wise
`memcpy` returned magic `0x3fc0…` (garbage) while an element-wise loop returned
`0x0134d888` (= 20240520) from the same file.

**Fix.** Copy element-wise like `llmm.io.read_and_copy` and keep `bytes_data`
live across the whole copy (`_ = bytes_data^`). Commit `23aed82`.

## Fix 2 — sharded AdamW update segfaulted on an unaligned shard offset

**Symptom.** With Fix 1 in place the test reached real execution and
**segfaulted** in `adamw_update` (`llmm/adamw.mojo:131`) from
`GPT2.update` for a WORLD_SIZE=2 rank>0 shard.

**Root cause.** The sharded optimizer step indexes params/grads at
`rank * optimizer_num_parameters`, and `adamw_update` issues
`alignment = align_of[SIMD[dtype, width]]` (naturally `width` elements) aligned
vector loads/stores. `optimizer_num_parameters` was `ceil(num_parameters /
WORLD_SIZE)`, which is not generally a multiple of the SIMD width. Example: the
test model has `num_parameters = 4592`, so at W2 the shard is 2296 elements —
only 32-byte aligned, versus the 64-byte AVX-512 load — and rank 1's
`params/grads + 2296` faulted. Rank 0 (offset 0) was fine, which is why the
baseline (W1) and rank 0 ran clean and only rank 1 crashed.

**Fix.** Round the per-rank shard length up to a multiple of the AdamW SIMD
width so every `rank * optimizer_num_parameters` offset stays aligned; the extra
tail elements live in the already-zero-filled padding region of the params/grads
buffers (both sized to `padded_num_parameters`), so numerics are unchanged.
Commit `1f685d5`. (This alignment invariant also matters for the GPU path once
that path works.)

## Fix 3 — single-GPU `ZeroContext` did needless cross-GPU P2P setup

**Symptom.** `tests/test_zero.mojo` was reported hitting the 600 s `make
test-mojo` timeout on this host. On an unloaded box it actually **passes 9/9 in
~1.8 s wall clock** (the per-test TestSuite timer misreports — it printed
"557 s"/"1792 s" for one test while the whole process exited in under 2 s). The
timeout was GPU contention during the coordinator's baseline.

**Contributing cause / fix.** `ZeroContext.__init__` unconditionally called
`enable_p2p()` (which enumerates and cross-enables peer access across **all**
visible GPUs) and allocated+initialized the signal buffer, even for N==1 — where
every collective early-returns without touching the signal buffer. On an 8-GPU
box that is slow and contention-sensitive. Gated the whole setup on `Self.N >=
2`; N==1 now gets the same 1-byte dummy buffer used on non-NVIDIA targets. This
also speeds up default (WORLD_SIZE=1) training/inference startup. Commit
`e46888c`.

## Test extension — WORLD_SIZE=8 equivalence

`run_zero_equivalence_test` is now parameterized over the world size N, with
stage-1/2/3 cases at **N=8** (the mission target) alongside N=2. `GPT2` is not
`Movable`, so N models can't be held in a `List`; instead the ranks are
simulated one at a time. Because every rank here sees the same `gpt2_tiny.bin`
init and the same batch, all ranks compute the identical gradient `g`, the
all-reduce is exactly `N*g` (and `update()` divides `grad_scale` by
`WORLD_SIZE`, recovering `g`), and each rank's owned shard is optimized
independently. Each rank's post-update shard is asserted against the single-GPU
baseline slice; the full parameter vector is the concatenation of the N shards,
so per-shard equivalence *is* whole-vector equivalence after an all-gather. W8
also exercises the shard-length padding (4592 is not divisible by 8) and the
last rank's shard running into the zero padding. Commit `b1ad5d5`.

Result (measured): `6 tests run: 6 passed` — stages 1/2/3 at W2 **and** W8, each
matching the WORLD_SIZE=1 baseline to `atol=1e-5`.

---

## The GPU multi-GPU blocker (root-caused, NOT fixed)

> **UPDATE (same day):** fixed in the follow-up campaign — all four ZeRO
> stages now train at WORLD_SIZE=8 via hand-rolled staged-copy collectives
> (this box turned out to have no CUDA P2P at all, so the design sketch
> below was adapted). See
> [`zero_multigpu_rewrite_2026-07-14.md`](zero_multigpu_rewrite_2026-07-14.md).

Every ZeRO stage crashes on the GPU target at the first collective:

```
val loss 5.22084
Unhandled exception … device_context.mojo:6625:17:
CUDA call failed: CUDA_ERROR_INVALID_VALUE (invalid argument)
```

Confirmed at world size **2, 4, and 8**, for **both fp32 and bf16**, so it is
neither an N=8 nor a precision issue.

**Root cause (proven by probe).** `llmm/zero.mojo`'s GPU collectives call the
Modular `comm.allreduce` / `reducescatter` / `allgather` with a **single**
`DeviceContext` and **N `TileTensor`s that all alias one device-0 buffer** (for
`allreduce`, N copies of the same pointer). On the current toolchain
(`mojo 1.0.0b3.dev2026071306`) the `comm` collectives require **one
`DeviceContext` per GPU with each rank's tensor on its own device** — the same
shape Modular's own `max/_distributed_ops/distributed_ops.mojo` uses
(`DeviceContextList[ngpus]`, per-device output buffers).

An isolated 16-element probe made this unambiguous:

- Single ctx + N aliased device-0 tensors (exactly what `zero.mojo` does) →
  `CUDA_ERROR_INVALID_VALUE`, identical to the training crash.
- N contexts (`DeviceContext(device_id=i)`) + one buffer per device + per-device
  signal buffers → **works**; `allreduce` of `[1.0]*16` and `[2.0]*16` returns
  `3.0`. Note it reduces to the *single* output device (not a full all-reduce to
  all ranks), so a true all-reduce needs one call per output device.

**Why it's not a small fix.** The GPU training path is not just mis-calling the
collectives — it is structurally single-device: `_try_gpu` runs **one** `train()`
with `rank=0` (unlike `_dispatch_cpu`, which `sync_parallelize`s N ranks over a
shared `CpuCoordinator`), and `train()` always builds **one** `GPT2` on
`DeviceContext()` (device 0). So there are no per-rank replicas on devices
1..N-1 for a real collective to reduce across, and with `rank` fixed at 0 the
sharded `update()` would only ever touch shard 0. This was visible in
`nvidia-smi` during a run: GPU 0 holds the whole model while GPUs 1..7 hold only
~566 MiB of `comm` scratch.

**What a real fix needs (design sketch for follow-up).** True N-device data
parallelism: launch N ranks (mirror `_dispatch_cpu`'s `sync_parallelize`), give
rank `r` a `DeviceContext(device_id=r)` and its own full `GPT2`, share each
rank's grad/param **device** pointers + per-device signal pointers through a
GPU coordinator (analogous to `CpuCoordinator`), and rewrite the `ZeroContext`
collectives to build the per-device `TileTensor` array from those shared
pointers. The concurrency model of `comm`'s single-call-reduces-to-one-device
semantics vs. N per-rank calls must be settled first (the probe shows the
single-output behavior but not whether N concurrent collectives sharing signal
buffers are safe). This is a substantial, separate piece of work.

---

## Benchmark data (`zero/bench/bench_zero_world8.json`)

Collected with the new `scripts/benchmark_zero.py` (run-side only; JSON, no
plotting) via `make benchmark-zero`. Flags held identical across stages:
`-b 4 -t 64`, `-pn 8`, 12 steps, `gpt2_124M` (124M params). Because every stage
crashes at the first training step, `mean_step_ms` / `tokens_per_sec` are
`null`; the memory numbers are **allocation-phase peak up to the crash**, not
steady-state training memory. Per-GPU memory is baseline-subtracted (pre-launch
`nvidia-smi` snapshot) so the other team's GPU-4..7 jobs don't inflate it — the
whole model lives on GPU 0 here, so GPU-0 delta is the ZeRO-relevant figure.

Even truncated at the crash, the sharding shows the expected monotonic
memory-per-stage curve (GPU-0 peak delta, MiB):

| Precision | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|---------|
| fp32 | 4275 | 3507 | 2739 | 2353 |
| bf16 | 3763 | 2483 | 1971 | 1841 |

Each higher stage shards more state (stage 1: optimizer moments; stage 2: + grad
comm buffer; stage 3: + param handling), and peak allocation drops accordingly.
These are single-GPU (device-0) allocation peaks; a true per-GPU steady-state
curve across all 8 GPUs is blocked on the multi-GPU fix above.

## Reproduction

- CPU equivalence (the correctness gate): `pixi run mojo run -I .
  tests/test_zero_equivalence.mojo` → 6/6 pass (W2+W8, stages 1/2/3).
- CPU collectives: `pixi run mojo run -I . tests/test_zero.mojo` → 9/9 pass.
- GPU crash: `WORLD_SIZE=8 ./scripts/run_train_gpt2.sh -e gpt2_124M.bin -b 4 -t
  64 -x 10 -z 1 -pn 8` → `CUDA_ERROR_INVALID_VALUE`.
- Benchmark JSON: `make benchmark-zero` (or the `scripts/benchmark_zero.py`
  invocation in its docstring).

## Commits

- `1f685d5` zero: pad optimizer shard length to the AdamW SIMD width
- `23aed82` test_zero_equivalence: fix read helper that freed its buffer mid-copy
- `e46888c` zero: skip P2P/signal-buffer setup for single-GPU (N==1) ZeroContext
- `b1ad5d5` test_zero_equivalence: generalize to any world size; add WORLD_SIZE=8
- (this doc + `scripts/benchmark_zero.py` + `make benchmark-zero` + `zero/bench/bench_zero_world8.json`)

## AI use statement

Written with AI assistance (Claude (Opus agent via Claude Code)), directed by
Evan.
