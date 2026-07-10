# ===----------------------------------------------------------------------=== #
# GPU kernels for tests/test_rng_sr.mojo, split into their own module.
#
# `TestSuite.discover_tests[__functions_in_module()]()` (used by
# tests/test_rng_sr.mojo) reflects over *every* function defined in that
# module to find the `test_`-prefixed ones — and, on this toolchain,
# constructing that reflection tuple forces every listed function's
# signature/value to be resolved, which for a `block_idx`/`thread_idx`-using
# raw GPU kernel means attempting to compile its body for the ambient
# (non-GPU) host target and hitting
# "Current compilation target does not support operation: _get_intrinsic_name".
# That's a different flavor of the same class of landmine documented in
# `bf16-build-needs-gpu-only-dispatch` (AArch64 codegen breaks if a GPU-only
# kernel is ever instantiated outside an explicit `compile_function[...]`
# GPU dispatch) — the fix here is the same in spirit: keep raw kernel `def`s
# out of any module that also does `__functions_in_module()` reflection.
# Putting them in this separate, non-`test_*`, TestSuite-free module sidesteps
# it entirely: `tests/test_rng_sr.mojo` only ever *imports* these, so they
# never appear in its own `__functions_in_module()` tuple.
# ===----------------------------------------------------------------------=== #

from std.memory import bitcast, UnsafePointer
from std.gpu import block_dim, block_idx, thread_idx

from llmm.rng_device import rng_u32, sr_cast_bf16


def rng_u32_kernel(
    out_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    seed: UInt64,
    stream: UInt64,
    n: Int,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < n:
        out_ptr[idx] = rng_u32(seed, UInt64(idx), stream)


def sr_cast_kernel(
    out_ptr: UnsafePointer[UInt16, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    seed: UInt64,
    stream: UInt64,
    n: Int,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < n:
        var bf = sr_cast_bf16(x_ptr[idx], seed, UInt64(idx), stream)
        out_ptr[idx] = bitcast[DType.uint16](bf)
