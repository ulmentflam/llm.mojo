# Minimal bisection repro: identical to probe2a_passthrough_ok.mojo except
# the fp32 value derived from an fp8_e4m3fn load has one `+ 1.0` applied
# before being stored (as fp32, so no down-cast is even involved). This
# fails GPU-target LLVM lowering with:
#   error: conversion from 'f8e4m3fn' to 'f32' is not implemented
#   note: see current operation: %N = "pop.cast"(...) :
#       (!kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f32>
# i.e. the compiler defers materializing the actual fp8->fp32 conversion
# until the value is used by a real arithmetic op, and that lowering path is
# unimplemented for the GPU target on this toolchain (1.0.0b3.dev2026062706).
# Reproduces identically for float8_e5m2 and for SIMD width>1 loads (see
# RESULTS.md) — this is a general GPU fp8-elementwise-compute gap, not
# specific to e4m3, scalar-vs-SIMD, or this particular expression.

from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.sys import has_nvidia_gpu_accelerator


def _mini_kernel(
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    x_ptr: UnsafePointer[Scalar[DType.float8_e4m3fn], ImmutAnyOrigin],
    n: Int,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < n:
        var x = x_ptr[idx].cast[DType.float32]()
        var y = x + 1.0
        out_ptr[idx] = y


def main() raises:
    if not has_nvidia_gpu_accelerator():
        print("no GPU")
        return
    var ctx = DeviceContext()
    comptime N = 32
    var dev_in = ctx.enqueue_create_buffer[DType.float8_e4m3fn](N)
    var dev_out = ctx.enqueue_create_buffer[DType.float32](N)
    var compiled = ctx.compile_function[_mini_kernel]()
    ctx.enqueue_function(
        compiled,
        dev_out.unsafe_ptr(),
        dev_in.unsafe_ptr(),
        N,
        grid_dim=(1,),
        block_dim=(32,),
    )
    ctx.synchronize()
    print("mini kernel3 (add-only, fp32 out) compiled+ran OK")
