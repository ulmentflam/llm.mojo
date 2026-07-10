# FP8 toolchain probe 4b: raw cuBLASLt FP8 GEMM with a bf16 output — the
# fix-up of probe4_cublaslt_fp8_gemm.mojo's CUBLAS_STATUS_NOT_SUPPORTED
# (e4m3-in/e4m3-out) result. Identical setup, except D (and C) are bf16
# instead of e4m3fn. Also fixes a layout bug found while building this
# variant: A/B are column-major (cuBLASLt convention, matching the ld=k
# layouts below), so element (row=k, col=m) of A is at offset m*K+k, not
# the row-major k*M+m an initial draft of probe4 used (which passed the
# CUBLAS call but produced a wildly wrong reference comparison).
#
# A[K,M]^T (e4m3) @ B[K,N] (e4m3) -> D[M,N] (bf16), M=N=K=256 (K%16==0, the
# fp8 tensor-core alignment requirement).
#
# Run with:
#   flock -w 10800 /tmp/llmm-gpu.lock -c \
#     'pixi run mojo run -I . tests/probe_fp8/probe4b_cublaslt_fp8_bf16out.mojo'
#
# RESULT (toolchain 1.0.0b3.dev2026062706, GB10/sm_121): PASSES.
# max_abs_err ≈ 8.4e-4, max_rel_err ≈ 3.1e-3 against an fp32 reference
# computed on the same fp8-quantized operands — i.e. error is at bf16
# rounding precision, not fp8 quantization noise. This is the recommended
# lowest-risk FP8 GEMM path on this toolchain: native e4m3 x e4m3 tensor
# inputs, bf16 accumulator/output, via the same cuBLASLt FFI bindings
# `llmm/matmul.mojo` already uses for its fast NVIDIA path (just needs
# `_lt_dt` extended with the float8_e4m3fn/e5m2 -> R_8F_E4M3/R_8F_E5M2
# mapping, and the fast path forced into TN format for fp8 operands).

from std.memory import UnsafePointer
from std.sys import size_of
from std.sys import has_nvidia_gpu_accelerator
from std.gpu.host import DeviceContext
from std.gpu.host._nvidia_cuda import CUDA
from _cublas.dtype import DataType
from _cublas.cublas import cublasOperation_t, ComputeType, check_cublas_error
from _cublas.cublaslt import (
    cublasLtMatmul,
    cublasLtMatmulDesc_t,
    cublasLtMatmulDescCreate,
    cublasLtMatmulDescDestroy,
    cublasLtMatmulDescSetAttribute,
    cublasLtMatmulDescAttributes_t,
    cublasLtMatrixLayout_t,
    cublasLtMatrixLayoutCreate,
    cublasLtMatrixLayoutDestroy,
    cublasLtMatmulPreference_t,
    cublasLtMatmulPreferenceCreate,
    cublasLtMatmulPreferenceDestroy,
    cublasLtMatmulPreferenceSetAttribute,
    Preference,
    cublasLtMatmulHeuristicResult_t,
    cublasLtMatmulAlgoGetHeuristic,
)
from linalg.matmul.vendor.blas import _get_global_handle, Backend


@always_inline
def _lt_dt_ext[dtype: DType]() -> DataType:
    comptime if dtype == DType.bfloat16:
        return DataType.R_16BF
    elif dtype == DType.float32:
        return DataType.R_32F
    elif dtype == DType.float8_e4m3fn:
        return DataType.R_8F_E4M3
    elif dtype == DType.float8_e5m2:
        return DataType.R_8F_E5M2
    else:
        return DataType.R_16F


def _fp8_gemm_tn[
    in_dtype: DType,
    out_dtype: DType,
](
    d_ptr: UnsafePointer[Scalar[out_dtype], MutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    # TN format required by cuBLASLt for fp8: A transposed, B not.
    # D[m,n] = A[k,m]^T @ B[k,n] (all column-major, per cuBLASLt convention).
    var handle = _get_global_handle[in_dtype, Backend.CUBLASLT](ctx)
    var lt = handle._get_cublas()
    var cuda_stream = CUDA(ctx.stream())
    comptime dt_in = _lt_dt_ext[in_dtype]()
    comptime dt_out = _lt_dt_ext[out_dtype]()

    var desc = cublasLtMatmulDesc_t()
    check_cublas_error(
        cublasLtMatmulDescCreate(
            UnsafePointer(to=desc).as_unsafe_any_origin(),
            ComputeType.COMPUTE_32F,
            DataType.R_32F,
        )
    )
    var transa = cublasOperation_t.CUBLAS_OP_T
    var transb = cublasOperation_t.CUBLAS_OP_N
    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_TRANSA,
            UnsafePointer(to=transa)
            .bitcast[NoneType]()
            .as_immutable()
            .as_unsafe_any_origin(),
            size_of[cublasOperation_t](),
        )
    )
    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_TRANSB,
            UnsafePointer(to=transb)
            .bitcast[NoneType]()
            .as_immutable()
            .as_unsafe_any_origin(),
            size_of[cublasOperation_t](),
        )
    )

    var a_l = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            UnsafePointer(to=a_l).as_unsafe_any_origin(),
            dt_in,
            UInt64(k),
            UInt64(m),
            Int64(k),
        )
    )
    var b_l = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            UnsafePointer(to=b_l).as_unsafe_any_origin(),
            dt_in,
            UInt64(k),
            UInt64(n),
            Int64(k),
        )
    )
    var c_l = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            UnsafePointer(to=c_l).as_unsafe_any_origin(),
            dt_out,
            UInt64(m),
            UInt64(n),
            Int64(m),
        )
    )
    var d_l = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            UnsafePointer(to=d_l).as_unsafe_any_origin(),
            dt_out,
            UInt64(m),
            UInt64(n),
            Int64(m),
        )
    )

    var pref = cublasLtMatmulPreference_t()
    check_cublas_error(
        cublasLtMatmulPreferenceCreate(
            UnsafePointer(to=pref).as_unsafe_any_origin()
        )
    )
    var ws_size = 32 * 1024 * 1024
    check_cublas_error(
        cublasLtMatmulPreferenceSetAttribute(
            pref,
            Preference.MAX_WORKSPACE_BYTES,
            UnsafePointer(to=ws_size)
            .bitcast[NoneType]()
            .as_immutable()
            .as_unsafe_any_origin(),
            size_of[Int](),
        )
    )

    var heur = cublasLtMatmulHeuristicResult_t()
    var cnt = 0
    check_cublas_error(
        cublasLtMatmulAlgoGetHeuristic(
            lt,
            desc,
            a_l,
            b_l,
            c_l,
            d_l,
            pref,
            1,
            UnsafePointer(to=heur).as_unsafe_any_origin(),
            UnsafePointer(to=cnt).as_unsafe_any_origin(),
        )
    )
    if cnt == 0:
        raise Error(
            "no cuBLASLt algorithm for fp8 TN GEMM (in_dtype="
            + String(in_dtype)
            + " out_dtype="
            + String(out_dtype)
            + ")"
        )

    var ws = ctx.enqueue_create_buffer[DType.uint8](ws_size)
    ctx.synchronize()
    var alpha = Float32(1.0)
    var beta = Float32(0.0)
    check_cublas_error(
        cublasLtMatmul(
            lt,
            desc,
            UnsafePointer(to=alpha)
            .bitcast[NoneType]()
            .as_immutable()
            .as_unsafe_any_origin(),
            a_ptr.bitcast[NoneType]().as_immutable().as_unsafe_any_origin(),
            a_l,
            b_ptr.bitcast[NoneType]().as_immutable().as_unsafe_any_origin(),
            b_l,
            UnsafePointer(to=beta)
            .bitcast[NoneType]()
            .as_immutable()
            .as_unsafe_any_origin(),
            d_ptr.bitcast[NoneType]().as_unsafe_any_origin(),
            c_l,
            d_ptr.bitcast[NoneType]().as_unsafe_any_origin(),
            d_l,
            UnsafePointer(to=heur.algo).as_immutable().as_unsafe_any_origin(),
            ws.unsafe_ptr().bitcast[NoneType]().as_unsafe_any_origin(),
            ws_size,
            cuda_stream.value()[],
        )
    )

    check_cublas_error(cublasLtMatmulPreferenceDestroy(pref))
    check_cublas_error(cublasLtMatmulDescDestroy(desc))
    check_cublas_error(cublasLtMatrixLayoutDestroy(a_l))
    check_cublas_error(cublasLtMatrixLayoutDestroy(b_l))
    check_cublas_error(cublasLtMatrixLayoutDestroy(c_l))
    check_cublas_error(cublasLtMatrixLayoutDestroy(d_l))


def main() raises:
    print("=== probe4b: cuBLASLt fp8 (e4m3, TN) x fp8 -> bf16 GEMM ===")
    if not has_nvidia_gpu_accelerator():
        print("no NVIDIA GPU accelerator detected — skipping (no-op)")
        return

    var ctx = DeviceContext()
    comptime M = 256
    comptime N = 256
    comptime K = 256
    comptime DT = DType.float8_e4m3fn
    comptime OUT_DT = DType.bfloat16

    # A/B are column-major (cuBLASLt convention, ld=k): element (row=k,
    # col=m) of A lives at offset m*K+k, and (row=k, col=n) of B at n*K+k.
    var host_a_f32 = ctx.enqueue_create_host_buffer[DType.float32](K * M)
    var host_b_f32 = ctx.enqueue_create_host_buffer[DType.float32](K * N)
    for i in range(K * M):
        host_a_f32.unsafe_ptr()[i] = (Float32(i % 13) - 6.0) * 0.05
    for i in range(K * N):
        host_b_f32.unsafe_ptr()[i] = (Float32((i * 7) % 11) - 5.0) * 0.05

    var host_a = ctx.enqueue_create_host_buffer[DT](K * M)
    var host_b = ctx.enqueue_create_host_buffer[DT](K * N)
    for i in range(K * M):
        host_a.unsafe_ptr()[i] = host_a_f32.unsafe_ptr()[i].cast[DT]()
    for i in range(K * N):
        host_b.unsafe_ptr()[i] = host_b_f32.unsafe_ptr()[i].cast[DT]()
    ctx.synchronize()

    # fp32 reference: D[m,n] = sum_k A[k,m] * B[k,n], using the
    # fp8-quantized (then upcast) operands so the only remaining error
    # source is cuBLASLt's own GEMM + bf16-output rounding.
    var host_c_ref = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    for m in range(M):
        for n in range(N):
            var acc = Float32(0.0)
            for k in range(K):
                var av = host_a.unsafe_ptr()[m * K + k].cast[DType.float32]()
                var bv = host_b.unsafe_ptr()[n * K + k].cast[DType.float32]()
                acc += av * bv
            host_c_ref.unsafe_ptr()[m * N + n] = acc

    var dev_a = ctx.enqueue_create_buffer[DT](K * M)
    var dev_b = ctx.enqueue_create_buffer[DT](K * N)
    var dev_d = ctx.enqueue_create_buffer[OUT_DT](M * N)
    dev_a.enqueue_copy_from(host_a)
    dev_b.enqueue_copy_from(host_b)
    ctx.synchronize()

    _fp8_gemm_tn[DT, OUT_DT](
        dev_d.unsafe_ptr(),
        dev_a.unsafe_ptr(),
        dev_b.unsafe_ptr(),
        M,
        N,
        K,
        ctx,
    )
    ctx.synchronize()

    var host_d = ctx.enqueue_create_host_buffer[OUT_DT](M * N)
    dev_d.enqueue_copy_to(host_d)
    ctx.synchronize()

    # D is column-major [m,n] with ld=m; host_d[n*M + m] == D[m,n].
    var max_abs_err = Float32(0.0)
    var max_rel_err = Float32(0.0)
    for m in range(M):
        for n in range(N):
            var got = host_d.unsafe_ptr()[n * M + m].cast[DType.float32]()
            var want = host_c_ref.unsafe_ptr()[m * N + n]
            var err = abs(got - want)
            if err > max_abs_err:
                max_abs_err = err
            var rel = err / (abs(want) + 1e-3)
            if rel > max_rel_err:
                max_rel_err = rel

    print(
        "M=N=K=", M, " max_abs_err=", max_abs_err, " max_rel_err=", max_rel_err
    )
    if max_abs_err < 0.05 and max_rel_err < 0.05:
        print(
            "probe4b PASSED (cuBLASLt e4m3 x e4m3 -> bf16 GEMM runs, numerics"
            " at bf16-rounding precision)"
        )
    else:
        print("probe4b FAILED (numeric mismatch beyond bf16 tolerance)")
