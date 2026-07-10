"""16x16 Randomized Hadamard Transform (RHT) — GPU kernels.

Per `docs/ai/fp4_training_recipes_research.md` §1 (NVIDIA NVFP4 pretraining
recipe, arXiv:2509.25149): a **16x16 Hadamard transform with a single fixed
random sign vector, shared globally across all layers and all of training**,
applied block-wise along the contraction (K) dimension of **Wgrad GEMM
inputs only** (activation + output-grad operands feeding the weight-gradient
GEMM). It is deliberately *not* applied to Fprop/Dgrad operands (would also
require transforming the weight, breaking forward/backward 2D-scale
consistency). RHT "Gaussianizes" outliers within a 16-element block so NVFP4
quantization (see `llmm/nvfp4_quant.mojo`) has less error to absorb.

This module builds the transform + its exact inverse only. Wiring it into
the Wgrad GEMM call sites (`llmm/matmul.mojo:matmul_d_weight_bwd`) is
integration work for a later merge — out of scope here per the coordinator's
HARD CONSTRAINT (do not touch `llmm/matmul.mojo`).

## Math

For a 16-element block x, forward: `y = H16 @ (s ⊙ x)`, where `H16` is the
(unnormalized) Sylvester-construction Hadamard matrix and `s` is a fixed
+/-1 sign vector. `H16` is symmetric and `H16 @ H16 == 16 * I` (Sylvester
Hadamard matrices are self-inverse up to a scale), so the exact inverse is:

    x = s ⊙ ((1/16) * H16 @ y)

i.e. the *same* butterfly network run again, then a 1/16 scale, then the
same sign multiply again (since `s_i ∈ {+1,-1} => s_i^{-1} == s_i`). Both
directions reuse `_fwht16`.

## Sign vector provenance

`HADAMARD_SIGNS` below is fixed for all of training (per the recipe — NOT
regenerated per call, NOT a device RNG). It was generated once, offline,
deterministically:

```python
import random
rng = random.Random(1234)
signs = [1 if rng.random() < 0.5 else -1 for _ in range(16)]
# -> [-1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1]
```

and hardcoded as a literal below. This is intentionally *not* built on top
of any device RNG — the coordinator's device-RNG chunk (stochastic-rounding
cast, `llmm/lowp.mojo` Chunk G in the FP8 team's design) is separate
infrastructure for per-element stochastic rounding, not for this one-time
global sign vector.

## GPU-only

Per project policy (AArch64 codegen hazard when any "cpu" target is
instantiated under low-precision-adjacent code — see memory note
`bf16-build-needs-gpu-only-dispatch`), the kernel-launching entry points
below are `comptime if is_gpu[target]()`-guarded and raise on any other
target. Nothing here actually uses an fp8/fp4 SIMD dtype (bf16 in/out,
fp32 compute), but the guard is kept for consistency with the rest of the
low-precision-adjacent surface and because these kernels are meant to run
only on the GPU training path.
"""

from std.collections import InlineArray
from std.math import ceildiv
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from llmm.memory import ImmutKernelPtr, MutKernelPtr


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime HADAMARD_BLOCK = 16
comptime HADAMARD_INV_SCALE = Float32(1.0 / 16.0)


@always_inline
def hadamard_sign(i: Int) -> Float32:
    """Fixed +/-1 sign vector, index 0..15. See module docstring for
    provenance (python `random.Random(1234)`, hardcoded — not regenerated).
    """
    if i == 0:
        return -1.0
    elif i == 1:
        return 1.0
    elif i == 2:
        return 1.0
    elif i == 3:
        return -1.0
    elif i == 4:
        return -1.0
    elif i == 5:
        return -1.0
    elif i == 6:
        return -1.0
    elif i == 7:
        return 1.0
    elif i == 8:
        return -1.0
    elif i == 9:
        return 1.0
    elif i == 10:
        return 1.0
    elif i == 11:
        return -1.0
    elif i == 12:
        return 1.0
    elif i == 13:
        return -1.0
    elif i == 14:
        return -1.0
    else:
        return 1.0


# ===----------------------------------------------------------------------=== #
# Fast Walsh-Hadamard Transform, 16-point, unnormalized (H16 @ x)
# ===----------------------------------------------------------------------=== #


@always_inline
def _fwht16(mut buf: InlineArray[Float32, HADAMARD_BLOCK]) -> None:
    """In-place unnormalized 16-point Sylvester-Hadamard butterfly:
    `buf <- H16 @ buf`. Standard iterative FWHT (natural/Hadamard order);
    `H16` is symmetric with `H16 @ H16 == 16 * I`, so this same routine
    serves as both the forward transform and (up to the caller applying a
    1/16 rescale) its own inverse.
    """
    var h = 1
    while h < HADAMARD_BLOCK:
        var i = 0
        while i < HADAMARD_BLOCK:
            var j = i
            while j < i + h:
                var a = buf[j]
                var b = buf[j + h]
                buf[j] = a + b
                buf[j + h] = a - b
                j += 1
            i += 2 * h
        h *= 2


# ===----------------------------------------------------------------------=== #
# GPU kernel: one thread per (row, k-block) 16-element tile
# ===----------------------------------------------------------------------=== #


@always_inline
def _hadamard16_kernel[
    dtype: DType,
    forward: Bool,
](
    out_ptr: MutKernelPtr[dtype],
    x_ptr: ImmutKernelPtr[dtype],
    rows: Int,
    k: Int,
    k_blocks: Int,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = rows * k_blocks
    if idx >= total:
        return
    var r = idx // k_blocks
    var kb = idx % k_blocks
    var k0 = kb * HADAMARD_BLOCK
    var base = r * k + k0

    var buf = InlineArray[Float32, HADAMARD_BLOCK](uninitialized=True)
    for i in range(HADAMARD_BLOCK):
        buf[i] = x_ptr[base + i].cast[DType.float32]()

    comptime if forward:
        # y = H16 @ (s ⊙ x)
        for i in range(HADAMARD_BLOCK):
            buf[i] = buf[i] * hadamard_sign(i)
        _fwht16(buf)
    else:
        # x = s ⊙ ((1/16) * H16 @ y)
        _fwht16(buf)
        for i in range(HADAMARD_BLOCK):
            buf[i] = (buf[i] * HADAMARD_INV_SCALE) * hadamard_sign(i)

    for i in range(HADAMARD_BLOCK):
        out_ptr[base + i] = buf[i].cast[dtype]()


# ===----------------------------------------------------------------------=== #
# Host entry points
# ===----------------------------------------------------------------------=== #


def _hadamard16_dispatch[
    dtype: DType,
    target: StaticString,
    forward: Bool,
](
    out_ptr: MutKernelPtr[dtype],
    x_ptr: ImmutKernelPtr[dtype],
    rows: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> None:
    if k % HADAMARD_BLOCK != 0:
        raise Error(
            "hadamard16: k must be a multiple of " + String(HADAMARD_BLOCK)
        )
    comptime if is_gpu[target]():
        var device_ctx = ctx
        var k_blocks = k // HADAMARD_BLOCK
        var total = rows * k_blocks
        comptime BLOCK_SIZE = 256
        var num_blocks = ceildiv(total, BLOCK_SIZE)

        comptime kernel = _hadamard16_kernel[dtype, forward]
        var compiled = device_ctx.compile_function[kernel]()
        device_ctx.enqueue_function(
            compiled,
            out_ptr,
            x_ptr,
            rows,
            k,
            k_blocks,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("hadamard16 is GPU-only")


def hadamard16_fwd_gpu[
    dtype: DType,
    target: StaticString,
](
    out_ptr: MutKernelPtr[dtype],
    x_ptr: ImmutKernelPtr[dtype],
    rows: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> None:
    """Forward RHT: `out[r, kb*16:(kb+1)*16] = H16 @ (s ⊙ x[r, kb*16:(kb+1)*16])`
    for every row `r` and every 16-wide block along `k`. `x` is `[rows, k]`
    row-major; `k` must be a multiple of 16. Per the recipe this is applied
    to the Wgrad GEMM's activation and output-grad operands only (not
    Fprop/Dgrad) — call-site wiring is out of scope for this module.
    """
    _hadamard16_dispatch[dtype, target, forward=True](
        out_ptr, x_ptr, rows, k, ctx
    )


def hadamard16_inv_gpu[
    dtype: DType,
    target: StaticString,
](
    out_ptr: MutKernelPtr[dtype],
    x_ptr: ImmutKernelPtr[dtype],
    rows: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> None:
    """Exact inverse of `hadamard16_fwd_gpu`: recovers `x` from `y` via
    `x = s ⊙ ((1/16) * H16 @ y)`.
    """
    _hadamard16_dispatch[dtype, target, forward=False](
        out_ptr, x_ptr, rows, k, ctx
    )
