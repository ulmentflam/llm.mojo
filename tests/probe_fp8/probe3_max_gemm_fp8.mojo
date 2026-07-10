# FP8 toolchain probe 3: FP8 GEMM through MAX's prebuilt `linalg.matmul`
# kernel (the same generic entry point `llmm/matmul.mojo` imports as
# `from linalg.matmul import matmul` for its vendor-neutral/CPU/Metal path).
#
# A[M,K] (e4m3) @ B[N,K]^T (e4m3) -> C[M,N] (float32), M=N=K=256, compared
# against an fp32 reference GEMM computed on the (already fp8-quantized)
# inputs, tolerance scaled for fp8 (~2 decimal digits of precision).
#
# Run with:
#   flock -w 10800 /tmp/llmm-gpu.lock -c \
#     'pixi run mojo run -I . tests/probe_fp8/probe3_max_gemm_fp8.mojo'
#
# RESULT (toolchain 1.0.0b3.dev2026062706, GB10/sm_121): FAILS to compile,
# with the exact same error as probe2 (GPU-target
# "conversion from 'f8e4m3fn' to 'f32' is not implemented" / unlowered
# `pop.cast`). MAX's generic `linalg.matmul.matmul()` GPU path apparently
# performs the same kind of scalar/SIMD fp8->f32 upcast-then-compute
# internally (for the accumulator and/or output store), so it hits the
# identical codegen gap. The vendor-neutral "MAX prebuilt kernel" GEMM path
# is therefore NOT usable for fp8 on this toolchain — see
# probe4b_cublaslt_fp8_bf16out.mojo for the GEMM path that does work
# (raw cuBLASLt FFI, which never asks Mojo to materialize an fp8 SIMD
# value in arithmetic).

from std.gpu.host import DeviceContext
from std.sys import has_nvidia_gpu_accelerator
from layout import Layout, TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul


def main() raises:
    print("=== probe3: fp8 GEMM via linalg.matmul (MAX prebuilt kernel) ===")
    if not has_nvidia_gpu_accelerator():
        print("no NVIDIA GPU accelerator detected — skipping (no-op)")
        return

    var ctx = DeviceContext()
    comptime M = 256
    comptime N = 256
    comptime K = 256
    comptime A_DT = DType.float8_e4m3fn
    comptime B_DT = DType.float8_e4m3fn
    comptime C_DT = DType.float32

    # Host buffers: small-magnitude values so fp8 quantization error stays
    # bounded and a K=256 accumulation doesn't blow up fp8's ~2-digit
    # precision budget.
    var host_a_f32 = ctx.enqueue_create_host_buffer[DType.float32](M * K)
    var host_b_f32 = ctx.enqueue_create_host_buffer[DType.float32](N * K)
    for i in range(M * K):
        host_a_f32.unsafe_ptr()[i] = (Float32(i % 13) - 6.0) * 0.05
    for i in range(N * K):
        host_b_f32.unsafe_ptr()[i] = (Float32((i * 7) % 11) - 5.0) * 0.05

    var host_a = ctx.enqueue_create_host_buffer[A_DT](M * K)
    var host_b = ctx.enqueue_create_host_buffer[B_DT](N * K)
    for i in range(M * K):
        host_a.unsafe_ptr()[i] = host_a_f32.unsafe_ptr()[i].cast[A_DT]()
    for i in range(N * K):
        host_b.unsafe_ptr()[i] = host_b_f32.unsafe_ptr()[i].cast[B_DT]()
    ctx.synchronize()

    # fp32 reference using the *fp8-quantized-then-upcast* values (host-side
    # casts — proven working by probe1) so the only error source under test
    # is the GEMM kernel itself, not fp8 quantization noise.
    var ref_a = ctx.enqueue_create_host_buffer[DType.float32](M * K)
    var ref_b = ctx.enqueue_create_host_buffer[DType.float32](N * K)
    for i in range(M * K):
        ref_a.unsafe_ptr()[i] = host_a.unsafe_ptr()[i].cast[DType.float32]()
    for i in range(N * K):
        ref_b.unsafe_ptr()[i] = host_b.unsafe_ptr()[i].cast[DType.float32]()

    var host_c_ref = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    for m in range(M):
        for n in range(N):
            var acc = Float32(0.0)
            for k in range(K):
                acc += (
                    ref_a.unsafe_ptr()[m * K + k]
                    * ref_b.unsafe_ptr()[n * K + k]
                )
            host_c_ref.unsafe_ptr()[m * N + n] = acc

    var dev_a = ctx.enqueue_create_buffer[A_DT](M * K)
    var dev_b = ctx.enqueue_create_buffer[B_DT](N * K)
    var dev_c = ctx.enqueue_create_buffer[C_DT](M * N)
    dev_a.enqueue_copy_from(host_a)
    dev_b.enqueue_copy_from(host_b)
    ctx.synchronize()

    var c = TileTensor(
        Span[Scalar[C_DT], MutAnyOrigin](ptr=dev_c.unsafe_ptr(), length=M * N),
        row_major(M, N),
    )
    var a = TileTensor(
        Span[Scalar[A_DT], ImmutAnyOrigin](
            ptr=dev_a.unsafe_ptr(), length=M * K
        ),
        row_major(M, K),
    )
    var b = TileTensor(
        Span[Scalar[B_DT], ImmutAnyOrigin](
            ptr=dev_b.unsafe_ptr(), length=N * K
        ),
        row_major(N, K),
    )

    matmul[transpose_b=True, target="gpu"](c, a, b, ctx=ctx)
    ctx.synchronize()

    var host_c = ctx.enqueue_create_host_buffer[C_DT](M * N)
    dev_c.enqueue_copy_to(host_c)
    ctx.synchronize()

    var max_abs_err = Float32(0.0)
    var max_rel_err = Float32(0.0)
    for i in range(M * N):
        var got = host_c.unsafe_ptr()[i]
        var want = host_c_ref.unsafe_ptr()[i]
        var err = abs(got - want)
        if err > max_abs_err:
            max_abs_err = err
        var rel = err / (abs(want) + 1e-3)
        if rel > max_rel_err:
            max_rel_err = rel

    print(
        "M=N=K=", M, " max_abs_err=", max_abs_err, " max_rel_err=", max_rel_err
    )
    if max_abs_err < 0.5 and max_rel_err < 0.1:
        print(
            "probe3 PASSED (MAX linalg.matmul accepts fp8 e4m3 inputs,"
            " numerics OK)"
        )
    else:
        print("probe3 FAILED (numeric mismatch beyond fp8 tolerance)")
