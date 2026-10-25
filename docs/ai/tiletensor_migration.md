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
kernels were compile-checked only. **Not done:** bf16 on Metal, and any run on
Apple hardware.

### Functional gates (all on the GB10)

- `make test-mojo`: 21 files / 168 tests pass (the intermittent `test_zero`
  hang aside, see below).
- `make verify-gpu`: step 0 loss `5.2700114` exactly.
- `make verify-gpu-tf32` passes.
- `test_attention_equivalence`: 32 passed.
- `make lint` passes.

## 5. Open items

- `test_zero` hangs intermittently; root cause is being chased in a separate
  worktree (see `docs/ai/test_zero_hang.md` if present).
- Metal is untested on hardware, and bf16-on-Metal was not even cross-compiled.
- A TileTensor-native tensor-core path would let the `.to_layout_tensor()`
  bridge go away.
- Some comments in `llmm/attention.mojo` still mention `LayoutTensor`s.
- The flash path's dispatch glue is unexercised: `USE_FLASH_FWD` is hard-coded
  off. Only the kernels were checked, by direct instantiation.

## Benchmarks

Pending: post-migration benchmark results will be added here by the benchmark pass (benchmark-gpu / profile step times, same-window interleaved against the 6aea140 baseline binaries).

---

## AI use statement

Written with AI assistance (Claude Code, Opus 5.5 and Sonnet 5.5 agents),
directed by Evan Owen. The syntax mapping, the absence of a TileTensor-native
MMA API, and the equality of the generated code were established by compiling,
probing and comparing SASS / Metal IR, not from documentation.
