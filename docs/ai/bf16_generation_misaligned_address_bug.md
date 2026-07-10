# The bf16 generation `CUDA_ERROR_MISALIGNED_ADDRESS` bug

A root-cause writeup of the crash that hit the 124M FineWeb training run's
end-of-run text generation on 2026-07-09, and the fix in commit `f0da883`.
This is the companion bug-report to
[`gpt2_124m_fineweb_training_run.md`](gpt2_124m_fineweb_training_run.md),
which covers that run's full timeline; this document goes deep on just the
kernel bug itself, since it's a real correctness issue independent of that
specific run.

---

## The symptom

Training completed all 19,552 steps cleanly (LR decayed to exactly 0, val
loss 3.2807), then crashed during the optional end-of-run sampling step —
`train_gpt2.mojo`'s generation block, which runs `model.forward(gen_tokens,
null_int32_ptr, 1, t)` for `t` in `1..gen_max_length`. The process died with
`CUDA_ERROR_MISALIGNED_ADDRESS`, and the kernel log independently confirmed a
hardware-level fault:

```
NVRM: Xid (PCI:000f:01:00): 13, Graphics SM Warp Exception on (GPC 0, TPC 0, SM 1): Misaligned Address
NVRM: Xid (PCI:000f:01:00): 13, Graphics Exception: ESR 0x5057b0=0x511000f 0x5057b4=0x20 ...
[... dozens more, one per affected SM/TPC/GPC unit ...]
NVRM: Xid (PCI:000f:01:00): 43, pid=74059, name=train_gpt2_bf16, channel 0x00000002
```

Xid 13 is the GPU reporting the actual misaligned-access exception; Xid 43 is
the driver force-resetting that CUDA channel, which is what actually killed
the training process. This was a big deal in practice: the training loop
writes checkpoints *after* the sampling block, so this crash cost the true
final checkpoint (`model_19552.bin`) on the first attempt, even though
training had already fully converged one step earlier.

## Why generation and not training

`model.forward()` is shared code between training and generation — the only
difference is the shape it's called with. Training always calls it with
`batch_size=32` (or 16), `seq_len=1024` (`T`, the fixed context length), for
all 19,552 steps, flawlessly. Generation calls it with `batch_size=1` and
`seq_len` walking through every value `1, 2, 3, ..., gen_max_length-1` as it
samples one token at a time — a code path with a variable, often-small
`seq_len` that training's fixed-`T=1024` calls never exercise.

Compounding this: `-s` (`sample_every`) was set to `20000` for a 19,552-step
run, and the sampling block triggers unconditionally at `last_step`
regardless of `sample_every`'s value — so this was the *first time this
entire code path had ever run* in this run's ~4.5 days of wall clock. It
crashed on literally its first real exercise.

## Root cause

`_attention_softmax_causal_gpu` (`llmm/attention.mojo`), reached via the
GEMM-attention path (`attention_fwd_gemm`, selected whenever
`USE_GEMM_ATTENTION` — `HAS_CUBLAS or HAS_METAL` — is true, which is every
production GPU build including this one), does vectorized bf16 loads/stores
`softmax_width` elements at a time (8, bf16's SIMD width), at a per-row base
offset of `row * seq_len`. That offset is only guaranteed a multiple of 8 —
and therefore correctly aligned for the hardware's vectorized load
instruction — when `seq_len` itself is a multiple of 8.

Training's `seq_len=1024` (`1024 / 8 = 128`, exact) always satisfied this.
Generation's `seq_len` walks through every integer starting at 1, and the
first value that isn't a multiple of 8 is `T=9` — reproduced deterministically
5/5 times at exactly that step, independent of RNG seed. This is a genuine,
deterministic alignment bug, not a flaky race condition.

## A red herring along the way

The first hypothesis (reasonable, given the existing but disabled
`# TODO: Race condition fix: Enable device-sync-mode` comment already sitting
in `scripts/run_train_gpt2.sh`) was a missing-synchronization race between
async GPU kernel launches. Rerunning the crash-triggering command with
`MODULAR_DEBUG=device-sync-mode` (which forces every kernel launch to
synchronize before the next one is issued) completed successfully twice in a
row — seemingly confirming the race-condition theory.

That test used `-g 8` (`gen_max_length=8`), which walks `seq_len` through
`1..7` — never reaching the actual failure point at `T=9`. The "successful"
run under `device-sync-mode` never actually exercised the buggy input; it
just happened not to reach it. **Lesson:** when a repro "passes" under a
debug/instrumentation flag, verify the repro still reaches the exact failing
input before trusting the flag as a fix or as evidence for a hypothesis.

## The fix

`_attention_softmax_causal_gpu` is now compiled in both its vectorized
(`softmax_width`-wide) and scalar (`width=1`) forms; a runtime check
(`T % softmax_width == 0`) picks which one to launch. Training's fast path
is untouched — `T=1024` always takes the vectorized branch, byte-for-byte
identical to before. Generation's non-aligned `T` values fall back to the
scalar kernel, which has no alignment assumption to violate.

```mojo
if T % softmax_width == 0:
    # existing vectorized kernel, unchanged
    ...
else:
    # scalar (width=1) fallback — no alignment assumption
    ...
```

## A second bug found in the same pass

While isolating the crash, a second, independent bug turned up in the
generation sampling readback: it reinterpreted a `bfloat16` host buffer
directly as `float32` — a raw 2-byte-vs-4-byte pointer reinterpret, not a
cast — reading roughly double the intended byte range per sampled token. This
explains why generated text was garbage even on the rare occasion the
alignment crash didn't fire before it. Fixed by casting element-by-element
into a correctly-sized fp32 scratch buffer instead of reinterpreting the
pointer.

## New tool: `infer_gpt2.mojo`

Root-causing and verifying this fix needed a fast way to exercise just the
checkpoint-load + generation code path, without paying for a full training
harness spin-up (dataloader, optimizer state, multi-GB activation buffers
sized for training batch sizes) on every iteration. `infer_gpt2.mojo` (`make
build-infer` / `make build-infer-bf16`) is a lean, training-harness-free
binary: load a checkpoint, generate N tokens, print the text. Use it for any
future generation-path debugging rather than reaching for the full trainer.

## Verification

- **Pre-fix**: 5/5 crashes, deterministically at `t=9`, across different RNG
  seeds.
- **Post-fix**: 55/55 crash-free generation trials (seeds spanning 1–55,
  `gen_max_length` 64–128), producing plausible (if imperfect, given a 124M
  model) English text fragments.
- **Training-path regression check**: 3 training steps at the real
  `B=32, T=1024` shape, pre-fix vs. post-fix, matched loss (`10.130247`,
  `9.694317`) and throughput (`33.7k tok/s`, `21.6% MFU`) to 5 decimal
  places — confirming zero impact on the path that actually trains the
  model.

## Testing this in CI

The existing Python equivalence suite (`tests/test_attention_equivalence.py`)
defaults to CPU (`tests/_max_bridge.py`'s `pick_device()`) unless
`MAX_USE_ACCELERATOR=1` is set — meaning it exercises `attention_fwd_cpu`, a
different function entirely, not the GPU-only `attention_fwd_gemm` path this
bug lived in. A case with a non-8-aligned `seq_len` added to that suite
without accelerator targeting would silently test the wrong kernel and prove
nothing about this bug.

Regression coverage landed in commits `69859e9`/`ca62e28`:
- `bf16_seq_len_9_unaligned` / `bf16_seq_len_17_unaligned` — bf16, non-8-aligned
  `seq_len`, via a new `Case.prefer_accelerator` field and `_device_for_case()`
  helper that picks the accelerator automatically when one's present (no env
  var needed), independent of the suite's shared CPU-default `pick_device()`.
  Verified fail-then-pass: reverting just the `llmm/attention.mojo` hunk from
  `f0da883` reproduces the exact `CUDA_ERROR_MISALIGNED_ADDRESS` on these
  cases; the fix makes them pass.
- `bf16_seq_len_9/17_unaligned_cpu_forced` — the same shapes deliberately
  pinned to `attention_fwd_cpu` (via a `Case.force_cpu` field), plus a whole-
  module `LLMM_TEST_FORCE_CPU=1` override — so the CPU path (which never had
  this bug) is verified on-demand, not just as an accidental fallback when no
  accelerator happens to be present.
- `tests/test_infer_gpt2_generation_smoke.py` — a coarser, production-path
  smoke test: several real generation trials through `build/infer_gpt2_bf16`,
  asserting no crash/`CUDA_ERROR` in stderr.

---

## AI use statement

This bug was root-caused and fixed with AI assistance via Claude Code, under
the direction of Evan Owen, immediately following the 124M FineWeb training
run documented in
[`gpt2_124m_fineweb_training_run.md`](gpt2_124m_fineweb_training_run.md).
