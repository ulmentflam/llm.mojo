# `wte_backward_cpu` derived its channel span from GPU warp geometry

Reported against the CPU custom-op path on Apple Silicon / macOS in
[PR #3](https://github.com/ulmentflam/llm.mojo/pull/3): `make test-python` failed
in `tests/test_encoder_equivalence.py` because encoder backward left the token
embedding gradients (`dwte`) at zero for token rows that appeared in the batch.

## Root cause

The bucket metadata that drives the sparse `wte` backward is produced against a
module-level contract:

```mojo
comptime WTE_BWD_SIMD_WIDTH = 4
comptime WTE_C_PER_WARP = 32 * WTE_BWD_SIMD_WIDTH        # 128
...
var num_channel_groups = ceildiv(channels, WTE_C_PER_WARP)
```

and the Python reference builder in the test hardcodes the same 128. The CPU
consumer instead re-derived the span from GPU warp geometry:

```mojo
var c_per_warp = WARP_SIZE * width      # width = simd_width_of[dtype]()
var c_base = channel_group * c_per_warp
```

`WARP_SIZE` is a GPU concept and carries no meaning on a CPU target, and `width`
here is the **host** SIMD width rather than `WTE_BWD_SIMD_WIDTH`. The GPU kernel
uses the identical expression and is correct there only because on a GPU target
it evaluates to exactly `32 * 4 == WTE_C_PER_WARP`. The CPU path inherited the
expression without inheriting the property that makes it true.

## Why the failure is platform-dependent

The span is wrong on every platform, but whether it produces wrong *results*
depends on which way it is wrong.

| | `WARP_SIZE * width` | vs 128 | result |
| --- | --- | --- | --- |
| x86-64 AVX-512, fp32 | 32 * 16 = 512 | larger | correct, badly imbalanced |
| reporter's macOS CPU compile | 0 | smaller | `dwte` never written |

When the consumer stride is **larger** than the producer's, group 0 covers
`[0, 512)`, every higher group computes `c_len <= 0` and is skipped, and the
union of the non-empty spans is still exactly `[0, channels)` covered once. The
answer is right; the work distribution is degenerate, with one bucket doing
everything and the rest idling.

When the consumer stride is **smaller**, coverage is truncated: at stride 0
nothing is written at all (the reported symptom), and at any stride below 128 the
channels between the consumer's stride and 128 are never accumulated while the
next group starts at the wrong offset.

Note that the mechanism in the report is not quite the one that fires on Apple
Silicon by itself: NEON fp32 gives `width == 4`, and `32 * 4 == 128` would have
matched. The operative fact is `WARP_SIZE` resolving to 0 on that sandboxed CPU
compile, not the SIMD width.

## Why the suite did not catch it

Two independent gaps, both worth keeping in mind:

1. **`make test` on a GPU box never runs this code.** It auto-detects the GPU and
   runs `test-mojo` + `test-python-cuda`, which takes the GPU device path. The
   CPU custom-op path is only exercised by `make test-python`, which needs a
   quiet box (see `max_cpu_custom_op_crash_2026-07-24.md`).
2. **Every encoder case had `channels <= 128`.** With `ceildiv(channels, 128) == 1`
   there is only ever `channel_group == 0`, so `c_base == 0 * anything == 0` and
   any wrong stride cancels.

## Fix

`wte_backward_cpu` uses `WTE_C_PER_WARP` directly.

A `channels=256` case (`fp32_multi_channel_group`) is added so the multi-group
path is covered at all. Be aware of what that case does and does not buy: on
x86-64 it passes both before and after the fix, because a too-large stride still
tiles `[0, channels)` correctly. It is real coverage for the multi-group path and
it catches this bug on the platforms where the bug is live, but it is not a
regression test that would have failed here. Catching a too-small stride on
x86 would require asserting the per-bucket span against `WTE_C_PER_WARP`
directly rather than checking end-to-end gradients.

## Verification

On workstation-max (Linux, 7x RTX PRO 6000):

- `tests/test_encoder_equivalence.py`, CPU custom-op path: 8 passed.
- Same file, GPU path (`MAX_USE_ACCELERATOR=1`, cuda env): 8 passed.
- Full `make gate`: green.
- `make verify`: 32/32 gradient tensors on both the CPU and GPU arms.

The reporter's own verification covered the macOS side, where the bug actually
reproduced.

## AI use statement

Written with AI assistance (Claude Code, Opus 5), directed by Evan Owen. The
root cause was confirmed by measuring `WARP_SIZE * simd_width_of[float32]` on a
host compile (512) against `WTE_C_PER_WARP` (128), and the coverage gap by
adding a two-channel-group case and observing that it passes on x86 regardless,
which is why that limitation is stated above rather than left implied.
