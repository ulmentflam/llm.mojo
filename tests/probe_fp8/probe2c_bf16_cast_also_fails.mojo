# Answers the literal ask in probe 4 of the FP8 toolchain probe plan: "try a
# manual fp8 load -> cast -> bf16 MAX GEMM fallback" (i.e. a GPU kernel that
# dequantizes fp8 straight to bf16, skipping the fp32 intermediate, then
# feeds a normal bf16 GEMM). This kernel alone (dequant fp8->bf16 + one
# arithmetic op) already fails the same way probe2b did for fp8->fp32:
#   error: conversion from 'f8e4m3fn' to 'bf16' is not implemented
#   note: see current operation: %N = "pop.cast"(...) :
#       (!kgen.scalar<f8e4m3fn>) -> !kgen.scalar<bf16>
# So the "manual dequant kernel" fallback is not viable either — the
# pop.cast lowering gap is general across ANY fp8 -> non-fp8 GPU conversion
# used in arithmetic, not specific to f32. The only working GPU fp8 path
# found by this probe suite is native cuBLASLt fp8 GEMM (raw pointers
# in, vendor library does the math) — see probe4b_cublaslt_fp8_bf16out.mojo.

from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.sys import has_nvidia_gpu_accelerator


def _mini_kernel(
    out_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    x_ptr: UnsafePointer[Scalar[DType.float8_e4m3fn], ImmutAnyOrigin],
    n: Int,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < n:
        var x = x_ptr[idx].cast[DType.bfloat16]()
        var y = x + 1.0
        out_ptr[idx] = y


def main() raises:
    if not has_nvidia_gpu_accelerator():
        print("no GPU")
        return
    var ctx = DeviceContext()
    comptime N = 32
    var dev_in = ctx.enqueue_create_buffer[DType.float8_e4m3fn](N)
    var dev_out = ctx.enqueue_create_buffer[DType.bfloat16](N)
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
    print("fp8->bf16 direct cast + arith compiled+ran OK")
