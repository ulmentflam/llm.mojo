# Minimal bisection repro (see probe2b_arithmetic_fails.mojo and
# tests/probe_fp8/RESULTS.md): a GPU kernel that casts fp8_e4m3fn -> fp32
# and immediately casts back to fp8_e4m3fn with NO intervening arithmetic
# compiles and runs fine. Contrast with probe2b, which is identical except
# for one extra `+ 1.0` on the fp32 value — that one fails GPU codegen.

from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.sys import has_nvidia_gpu_accelerator


def _mini_kernel(
    out_ptr: UnsafePointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin],
    x_ptr: UnsafePointer[Scalar[DType.float8_e4m3fn], ImmutAnyOrigin],
    n: Int,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < n:
        var x = x_ptr[idx].cast[DType.float32]()
        out_ptr[idx] = x.cast[DType.float8_e4m3fn]()


def main() raises:
    if not has_nvidia_gpu_accelerator():
        print("no GPU")
        return
    var ctx = DeviceContext()
    comptime N = 32
    var dev_in = ctx.enqueue_create_buffer[DType.float8_e4m3fn](N)
    var dev_out = ctx.enqueue_create_buffer[DType.float8_e4m3fn](N)
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
    print("mini kernel compiled+ran OK")
