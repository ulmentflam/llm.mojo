# FP8 toolchain probe 2: trivial elementwise GPU kernel reading/writing an
# fp8 (e4m3) buffer — cast to fp32 inside the kernel, compute, cast back.
#
# Follows the llmm/gelu.mojo gpu-kernel pattern (compile_function +
# enqueue_function). GPU-only: guarded by has_nvidia_gpu_accelerator() so it
# is a silent no-op on a CPU-only box, matching the repo's own bf16 GPU-only
# dispatch convention (see llmm/vendor.mojo HAS_CUBLAS/HAS_METAL guards and
# the bf16-build-needs-gpu-only-dispatch incident).
#
# Run with:
#   flock -w 10800 /tmp/llmm-gpu.lock -c \
#     'pixi run mojo run -I . tests/probe_fp8/probe2_gpu_kernel.mojo'
#
# RESULT (toolchain 1.0.0b3.dev2026062706, GB10/sm_121): this FAILS to
# compile. See probe2a_passthrough_ok.mojo / probe2b_arithmetic_fails.mojo
# for the minimal bisection — GPU-target codegen cannot lower an
# fp8_e4m3fn/e5m2 -> fp32 cast when the resulting fp32 value feeds into any
# further arithmetic (only a bare cast-in/cast-out passthrough compiles).
# Kept here (rather than deleted) as the "realistic elementwise kernel"
# probe demonstrating the failure in a shape close to real usage.

from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.sys import has_nvidia_gpu_accelerator
from std.math import ceildiv


def _fp8_axpy_kernel(
    out_ptr: UnsafePointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin],
    x_ptr: UnsafePointer[Scalar[DType.float8_e4m3fn], ImmutAnyOrigin],
    n: Int,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < n:
        var x = x_ptr[idx].cast[DType.float32]()
        var y = x * 2.0 + 1.0
        out_ptr[idx] = y.cast[DType.float8_e4m3fn]()


def main() raises:
    print("=== probe2: fp8 GPU elementwise kernel ===")
    if not has_nvidia_gpu_accelerator():
        print("no NVIDIA GPU accelerator detected — skipping (no-op)")
        return

    var ctx = DeviceContext()
    comptime N = 256
    comptime BLOCK_SIZE = 64

    var host_in = ctx.enqueue_create_host_buffer[DType.float32](N)
    for i in range(N):
        host_in.unsafe_ptr()[i] = Float32(i) * 0.1 - 5.0

    # Host-side fp32 -> fp8 cast, then upload the fp8 buffer.
    var host_in_fp8 = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](N)
    for i in range(N):
        host_in_fp8.unsafe_ptr()[i] = host_in.unsafe_ptr()[i].cast[
            DType.float8_e4m3fn
        ]()
    ctx.synchronize()

    var dev_in = ctx.enqueue_create_buffer[DType.float8_e4m3fn](N)
    var dev_out = ctx.enqueue_create_buffer[DType.float8_e4m3fn](N)
    dev_in.enqueue_copy_from(host_in_fp8)
    ctx.synchronize()

    var num_blocks = ceildiv(N, BLOCK_SIZE)
    var compiled = ctx.compile_function[_fp8_axpy_kernel]()
    ctx.enqueue_function(
        compiled,
        dev_out.unsafe_ptr(),
        dev_in.unsafe_ptr(),
        N,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )
    ctx.synchronize()

    var host_out_fp8 = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](N)
    dev_out.enqueue_copy_to(host_out_fp8)
    ctx.synchronize()

    var max_abs_err = Float32(0.0)
    for i in range(N):
        var x = host_in_fp8.unsafe_ptr()[i].cast[DType.float32]()
        var expect = (
            (x * 2.0 + 1.0).cast[DType.float8_e4m3fn]().cast[DType.float32]()
        )
        var got = host_out_fp8.unsafe_ptr()[i].cast[DType.float32]()
        var err = abs(Float32(got) - Float32(expect))
        if err > max_abs_err:
            max_abs_err = err

    print("N =", N, " max_abs_err (fp8-quantized ref) =", max_abs_err)
    if max_abs_err == 0.0:
        print("probe2 PASSED (GPU fp8 kernel compiled, ran, exact match)")
    else:
        print("probe2 FAILED (numeric mismatch)")
