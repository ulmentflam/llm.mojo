# Mojo 1.2.0 bump and the LayoutTensor to TileTensor migration

A log of the October 2026 move from Mojo 1.1.0 / MAX 26.6 to **Mojo 1.2.0 /
MAX 26.7**, the stale-debug-state incident found while verifying the bump, and
the migration of every `LayoutTensor` use in `llmm/` and `bench_gemm.mojo` to
`TileTensor`. Baseline for all comparisons is commit `6aea140` (the 1.2 bump
plus the verify guard, already on `main`).

The interesting part is not the syntax change but how it was validated: a
passing loss trace is a weak gate for this kind of change, and byte-identical
SASS is a strong one.

---

## 1. The Mojo 1.2.0 / MAX 26.7 bump (`c17e37a`)

Three hard breaks, all found by compiling:

- `std.sys.has_nvidia_gpu_accelerator()` / `has_apple_gpu_accelerator()` were
  removed. `llmm/vendor.mojo` now defines same-named shims over
  `std.sys.info._accelerator_arch()` (a **private** API; the string looks like
  `nvidia:sm_121a`), and every caller imports them from there. The `metal`
  prefix match is untested: there is no Apple host on this machine.
- `linalg.matmul`'s `elementwise_lambda_fn` now receives a `Coord` instead of
  `IndexList[2]`. The three epilogues in `llmm/matmul.mojo` index it with
  `idx[i].value()`.
- `comm.allgather` wants `rank_sigs` sized `N_GPUS`, not `MAX_GPUS`.

Evidence: `git show c17e37a` (its message lists what was verified on the GB10).

## 2. The stale debug-state incident (`6aea140`)

After the bump, `make verify-cpu` / `verify-gpu` printed ten `LOSS MISMATCH`
lines, which read like a trainer regression.

- Cause: this machine's untracked `gpt2_124M_debug_state.bin` stored loss
  `5.3544` for step 0, while the expected trajectory in `test_gpt2.mojo` starts
  at `5.2700`. The file had been written by an older `train_gpt2.py` with a
  different first batch.
- Not a 1.2 regression: the failure was identical on Mojo 1.1.0.
- Fix: regenerate with
  `pixi run python train_gpt2.py --device cpu --num_iterations 0`. The upstream
  starter-pack file stores `5.2700086` and passes.
- Guard: `test_gpt2.mojo` now compares the state's stored loss to the
  trajectory's step 0 (tolerance 1e-3; the upstream file differs by 2.6e-5)
  before running anything, and on mismatch prints a `STALE DEBUG STATE` banner
  with the cause and the regeneration command, then fails.

## 3. The TileTensor migration

Mojo 1.2 deprecates `LayoutTensor`. `main` emitted **101** warnings per build;
after the migration `llmm/` emits **0** (by file before: attention 91,
layernorm 4, matmul 2, lowp 2, nvfp4 1, encoder 1).

### Syntax

| `LayoutTensor` | `TileTensor` |
| --- | --- |
| `Layout.row_major(a, b)` (runtime `IntTuple`) | `row_major[a, b]()` (comptime, `Coord`-based trait layouts) |
| `LayoutTensor[dt, layout, MutAnyOrigin, address_space=SHARED].stack_allocation()` | free function `stack_allocation[dt, address_space=SHARED](row_major[a, b]())` |
| `LayoutTensor[...](ptr)` over a raw pointer | `TileTensor(Span[...](unsafe_ptr=ptr, length=n), row_major[a, b]())` |
| `IndexList[N]` coordinates | `Coord` |
| helper parameters typed as bare `LayoutTensor` | `TileTensor` |
| `.ptr` | `.ptr` (unchanged) |

### Behavioural differences worth knowing

- Bounds are checked at compile time where the layout is static.
- Shared-memory tiles use `linear_idx_type` int32.
- An immutable `TileTensor`'s `.load()` defaults to an invariant
  (read-only-cache) load. This code does not hit it because every access goes
  through raw `.ptr`; the SASS check below confirmed no such loads appeared.
- `stack_allocation`'s origin changes from `MutAnyOrigin` to
  `MutUntrackedOrigin`.
- `layout.tensor_core.TensorCore` (`load_a`, `load_b`, `store_d`) accepts only
  `LayoutTensor`. Research found no TileTensor-native sm80 MMA API, so the
  flash-attention MMA subtiles are bridged with `.to_layout_tensor()` at the
  call boundary, as Modular does for its own sm80/sm90 code.

### Phases

Work was split by file region across agents in separate worktrees (branches
`tt-p1-ln`, `tt-p1-enc`, `tt-p1-lp`, `tt-p2-fwd`, `tt-p2-gemm`, `tt-p3-bwd`,
`tt-p4-flash`), merged into `tiletensor-migration`. `git log --oneline
6aea140..HEAD` lists them.

| Phase | Commits | Scope |
| --- | --- | --- |
| 0 | `2aada91`, `055effe` | layernorm residual-row shared buffer; encoder `accum_shared`; nvfp4 `amax_sh` |
| 1 | `94931e8`, `7fd4842` | layernorm `s_weight/s_bias/s_res` (dead Metal kernel); 2D padded transpose tiles in `lowp.mojo` and `matmul.mojo` |
| 2 | `06ac32c`, `ef3fc70` | attention transpose / batched-gemm / headout4 / scoreout tiles (x10); forward chain allocs and params |
| 3 | `b5211da` | attention dQ/dK/dV tile-kernel chain |
| 4 | `33f8b3d` | flash-forward `_flash_sh` tiles, bridged to `TensorCore` |
| - | `6478552` | `bench_gemm.mojo` shared `a_sh`/`b_sh` tiles |
| - | `4e09f5d` | README / CHANGELOG |

## 4. Validation

### Why loss traces are weak gates here

- fp32 training is **not bit-deterministic** run to run (atomics). Step 1 is
  identical; the spread reaches ~2e-3 by step 30.
- A 5-run envelope of the baseline was too narrow: the baseline itself fell
  outside it in 3 of 3 fresh runs. A trace "inside the noise" therefore proves
  little.
- bf16 / fp8 / fp4 training **is** bit-deterministic, so those were compared
  exactly.
- More fundamentally, a passing loss trace could not have caught the changes
  this migration risks: alignment, stride, shared-memory layout or access-width
  changes that alter performance (or crash on a different shape) but not the
  loss at the tested size. Compare the misaligned-address bug in
  [`bf16_generation_misaligned_address_bug.md`](bf16_generation_misaligned_address_bug.md),
  which passed training and failed only at a particular sequence length.

### The strong gate: SASS identity

Cubins were carved out of the trainer binaries (baseline vs migrated) and
diffed kernel by kernel:

| Trainer | Kernels compared | Diffs |
| --- | --- | --- |
| fp32 | 37 | 0 |
| bf16 | 37 | 0 |
| fp8 | 48 | 0 |
| fp4 | 49 | 0 |

The trainers do not contain every migrated kernel (training uses cuBLAS
attention plus `attention_bwd_gemm`), so identity had to be established by
**forced instantiation** of kernels a trainer never compiles:

- `attention_fwd_gpu` plus six gemm / transpose kernels: 28 variants, all
  identical.
- Backward dq / dkv kernels: **not fully identical.** The Phase 3 implementer
  reported 10 variants identical, but the independent validator forced 12
  (f32 / bf16 / f16, five Br / Bc / BLOCK_SIZE combinations) and found 4
  identical and 8 different. The differences are integer address arithmetic
  only (LEA, IMAD.SHL, LOP3, MOV, NOP): e.g. the baseline computes
  `(tid*4)&0xfe0 + (tid&7)*0x80` where the migrated kernel computes
  `((tid&7)*4 + (tid>>3))*0x20`, both the same
  `reduction_shared[local_row, thread_row_index, c]` offset. Counts and
  modifiers of every memory instruction (LDG, STG, LDS, STS, LDL, STL, BAR, LDC)
  match, register count, frame size and `.nv.shared` size / alignment are
  unchanged, and the dq / dkv outputs are bit-identical (18 runs over head_dim
  64, 36 and 128 with T = 37, hashed on the bit patterns, plus a
  `compute-sanitizer memcheck` run with 0 errors). Judged benign, but this is
  the one place the "identical SASS" gate was relaxed to "same memory ops and
  bit-identical outputs".
- Flash forward, bf16 and f16: 4 variants identical in SASS, with identical
  numerics against the CPU reference.
- `bench_gemm`: identical.

Also compared: `.nv.shared` size and alignment per kernel (identical), and a
scan for read-only-cache loads (`LDG.E.CONSTANT`-style) that an invariant
`.load()` would introduce (none appeared).

### Metal

Cross-compiled with `--target-accelerator=metal:4 --emit asm`: the IR is
byte-identical for 17 kernels in f32/f16. The Metal-only dead layernorm
kernels were compile-checked only. **Later, on Apple hardware** (reported by a
Mac session at `12a180a`/`bf39d73`, not run on the GB10): `make test-mojo`
builds all 23 files and passes (141 tests, 140 passed, 1 skipped), and pytest
on the Metal GPU (`MAX_USE_ACCELERATOR=1`) passes (263 passed, 5 skipped,
3 xfailed). The hadamard, SR, amax, NVFP4-quant, dbias and attention
shape-change tests run on Metal. bf16 generation sampled normally on Metal by
hand; the bf16 generation smoke tests still skip there (no `log124M`
checkpoint on that machine).

### Functional gates (all on the GB10)

- `make test-mojo`: 21 files / 168 tests passed at the time of the migration;
  the later fixes below bring it to 23 files, all passing. (The intermittent
  `test_zero` hang was root-caused and fixed afterwards, see
  `docs/ai/test_zero_hang.md`.)
- `make verify-gpu`: step 0 loss `5.2700114` exactly.
- `make verify-gpu-tf32` passes.
- `test_attention_equivalence`: 32 passed.
- `make lint` passes.

## 5. Open items

- ~~`test_zero` hangs intermittently~~ Fixed: a barrier/runtime-scheduling
  deadlock, not part of the migration (`docs/ai/test_zero_hang.md`).
- Metal on hardware: now run by a Mac session (see the Metal section above).
  Still open: the bf16 generation smoke tests there need the `log124M`
  checkpoint copied over.
- The SASS-identity results in section 4 were taken at `6325a36`. Later
  commits changed host-side code in `attention.mojo` and `train_gpt2.mojo`
  (shape-change zeroing, sampling); they are not expected to change any kernel,
  but the comparison was not repeated.
- A TileTensor-native tensor-core path would let the `.to_layout_tensor()`
  bridge go away.
- The backward dq/dkv DRAM `TileTensor` views (`Span(unsafe_ptr=...,
  length=R*MAX_HEAD_DIM)`, `row_major[R, MAX_HEAD_DIM]()`) overstate the extent
  when `head_dim < MAX_HEAD_DIM` (real row stride is `head_dim`, and the last
  tile can have fewer than R rows). Audited: all 13 such views (4 sites) are
  consumed only by `_attention_copy_tile`, via `.ptr` with the real stride, so
  nothing is out of bounds. Deliberately left as is, since a true-extent fix
  would change the types/layouts of SASS-validated kernels; the invariant
  ("access only via `.ptr`; extent is an upper bound") is now documented in
  comments (comment-only change). The per-head GEMM views (`length=M*K` etc.)
  are exact.
- The flash path's dispatch glue is unexercised: `USE_FLASH_FWD` is hard-coded
  off. Only the kernels were checked, by direct instantiation.

## Benchmarks

**Verdict: still neutral within noise on current `main`.** Re-measured
2026-10-26 at `bf39d73`. No dtype got slower or faster. Every new-to-baseline
median ratio is within 0.3% of 1.000. The round-to-round spread of either
binary is 0.3-1.4%, and the per-round ratios straddle 1.0 for every dtype.
This fits the SASS-identity result in section 4.

The re-run was needed because the first measurement (2026-10-25, at
`03d9b9e`, migrated branch `e5f7cf8`) predates later changes to the trainer:
the forward attention path in `llmm/attention.mojo` (shape-change re-zeroing in
`attention_fwd_gemm`), plus `train_gpt2.mojo`, `llmm/zero.mojo`,
`llmm/memory.mojo` and `llmm/sampler.mojo`. Those first figures are kept below
for reference.

**Trees.** Baseline is a fresh detached worktree `llm.mojo-base` at `6aea140`
(the pre-migration commit). New is an agent worktree at `bf39d73` (current
`main`). Both trees share the main checkout's `.pixi` env (Mojo 1.2.0,
`pixi.lock` identical between the two commits), weights and tinyshakespeare
data via symlinks. Both were built with `make build build-bf16 build-fp8
build-fp4` just before the run. Every `build/train_gpt2*` binary was
timestamped 2026-10-26 11:18-11:20, newer than every `.mojo` source in its
tree, and `make -n` confirmed that neither tree would rebuild during the run.

**Method.** Unchanged from the first run. Each arm ran through its own tree's
make target:
`make -C <tree> train-gpt2-124m-<dtype> STEPS=40 ARGS="-v 1000 -s 1000 -m 1"`.
That is GPT-2 124M checkpoint init on tinyshakespeare, B=4, T=1024, with
40 steps per run. Runs were interleaved base, new, base, new for 7 rounds per
dtype, in the order fp32, bf16, fp8, fp4 (56 runs, 11:20-11:30). Only one GPU
job ran at a time.
The metric is the trainer's own per-step `ms` print, which is synchronized
and excludes val/sampling. Steps 1-5 were dropped as warmup, and the median
of steps 6-40 is taken per run. The table gives the median and the minimum
of those 7 per-run medians, plus the fastest single step seen in any run.
"Spread" is (max - min) / median of the per-run medians. `make benchmark-gpu`
was not used. It times the `profile_gpt2*` harnesses against llm.c/PyTorch
rather than comparing two trees' trainers, and it has no fp8/fp4 arms.

**Load.** The vLLM tenant (`VLLM::EngineCore`, pid 27742, 65,121 MiB) was
resident on the GPU throughout. `nvidia-smi --query-compute-apps` was checked
before the run, after it, and before every run, and nothing else was using
the GPU. The instantaneous utilization sampled before each launch was often
non-zero (1-82%). That sample was taken right after the previous run exited,
so it may be the previous run's tail rather than vLLM traffic; it cannot be
told apart, and it hit both arms alike. The tenant's footprint slows
tensor-core kernels, so these absolute numbers are not comparable to
tenant-free figures elsewhere in `docs/ai/`. Only the same-window ratio is
meaningful.

| dtype | base median | new median | ratio (median) | base min | new min | ratio (min) | fastest step base / new | spread base / new | per-round ratio range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 282.89 | 282.92 | **1.0001** | 280.12 | 281.00 | 1.0031 | 277.31 / 277.30 | 1.4% / 1.0% | 0.994-1.004 |
| bf16 | 138.29 | 138.31 | **1.0001** | 137.75 | 138.00 | 1.0018 | 134.95 / 136.10 | 1.1% / 0.7% | 0.993-1.005 |
| fp8 | 153.72 | 153.82 | **1.0007** | 153.47 | 153.40 | 0.9995 | 151.30 / 151.24 | 0.3% / 0.4% | 0.997-1.002 |
| fp4 | 158.21 | 158.62 | **1.0026** | 157.89 | 158.07 | 1.0011 | 155.91 / 155.80 | 0.8% / 0.6% | 0.997-1.005 |

(ms/step; ratio = new / base, so less than 1 means current `main` is faster.)

fp4's 1.0026 is the largest ratio. It is below fp4's own spread (0.6-0.8%),
its per-round ratios range 0.997-1.005 with three rounds at 1.000, and its
min-run ratio is 1.0011. The first run measured fp4 at 0.9991. It is noise,
not a regression.

**Numerics, as a side check.** The bf16, fp8 and fp4 loss traces were
bit-identical across all 14 runs of each dtype: every round, and baseline
vs new alike (step-40 loss bf16 3.076523, fp8 3.106864, fp4 3.095101).
fp32 is not run-to-run deterministic in *either* tree. Its step-40 loss
ranges over 3.081985-3.082455 across the 7 baseline runs and over
3.081875-3.082516 across the 7 new runs. The two ranges overlap, so fp32
shows no effect either.

### First run (2026-10-25, migrated branch `e5f7cf8`, merged as `03d9b9e`)

Same method, baseline binaries built 2026-10-22, same vLLM tenant resident.
Neutral within noise: every ratio within 0.2% of 1.000, spread 0.2-0.9%,
bf16/fp8/fp4 loss traces bit-identical.

| dtype | base median | tt median | ratio (median) | base min | tt min | ratio (min) | fastest step base / tt | spread base / tt | per-round ratio range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 282.41 | 282.38 | **0.9999** | 280.39 | 280.80 | 1.0015 | 277.83 / 278.21 | 0.9% / 0.8% | 0.994-1.005 |
| bf16 | 137.80 | 138.03 | **1.0017** | 137.57 | 137.50 | 0.9995 | 134.68 / 135.48 | 0.8% / 0.6% | 0.993-1.004 |
| fp8 | 153.42 | 153.34 | **0.9995** | 153.29 | 153.15 | 0.9991 | 151.40 / 150.99 | 0.2% / 0.2% | 0.999-1.001 |
| fp4 | 158.03 | 157.89 | **0.9991** | 157.57 | 157.55 | 0.9999 | 155.20 / 155.50 | 0.4% / 0.5% | 0.998-1.002 |

---

## AI use statement

Written with AI assistance (Claude Code, Opus 5.5 and Sonnet 5.5 agents),
directed by Evan Owen. The syntax mapping, the absence of a TileTensor-native
MMA API, and the equality of the generated code were established by compiling,
probing and comparing SASS / Metal IR, not from documentation.
