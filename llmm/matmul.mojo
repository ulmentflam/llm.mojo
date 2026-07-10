import compiler
from std.memory import alloc, stack_allocation
from std.math import ceildiv
from layout import Layout, TileTensor
from layout.layout_tensor import LayoutTensor
from linalg.matmul import matmul
from std.sys import simd_width_of
from std.gpu.primitives import block
from linalg.matmul.vendor import blas
from std.utils.index import IndexList
from extensibility import InputTensor
from linalg.transpose import transpose
from layout.tile_layout import row_major
from std.gpu.host.info import is_cpu, is_gpu
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from std.runtime.asyncrt import parallelism_level
from std.algorithm import vectorize, sync_parallelize
from std.gpu.host import DeviceContext, DeviceAttribute
from std.gpu.memory import AddressSpace
from std.gpu import barrier, block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import threadfence, Scope
from std.atomic import Atomic
from std.ffi import _get_global_or_null, external_call
from std.sys import size_of
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
    cublasLtMatrixLayoutSetAttribute,
    LayoutAttribute,
    cublasLtMatmulPreference_t,
    cublasLtMatmulPreferenceCreate,
    cublasLtMatmulPreferenceDestroy,
    cublasLtMatmulPreferenceSetAttribute,
    Preference,
    cublasLtMatmulHeuristicResult_t,
    cublasLtMatmulAlgoGetHeuristic,
)
from linalg.matmul.vendor.blas import _get_global_handle, Backend

from llmm.gelu import gelu, gelu_grad, gelu_fwd_gpu, bias_gelu_fwd
from llmm.profiler import traced_parallelize
from llmm.memory import ImmutKernelPtr, MutKernelPtr
from llmm.vendor import HAS_CUBLAS, HAS_METAL, USE_TF32


# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime UNROLL = 4
comptime _CUBLASLT_WS_BYTES = 32 * 1024 * 1024

# llm.c defaults `gelu_fusion=0` (train_gpt2.cu:350, the H100-conditional
# `>= 9 ? 2 : 0` is commented out/dead) with the comment "cuBLAS seems to be
# inefficient for fused GELU on Ada/Ampere" (llmc/matmul.cuh:235) — i.e. by
# default llm.c runs the FC forward GEMM with a plain BIAS epilogue + a
# separate `gelu_forward` kernel, and the d_input backward GEMM with no DGELU
# epilogue + a separate `gelu_backward_inplace` kernel. This flag lets us A/B
# that choice on our own hardware: True = always fuse GELU into the cuBLASLt
# epilogue (GELU_AUX_BIAS forward / DGELU backward, our long-standing
# default); False = llm.c's unfused default (plain GEMM + standalone
# elementwise GELU kernel, both directions). See docs/benchmarks.md item 2
# (GELU fusion A/B) for the measured verdict on this GPU.
comptime USE_GELU_FUSION = True


# ===----------------------------------------------------------------------=== #
# Matmul Forward
# ===----------------------------------------------------------------------=== #


def _matmul_bias_act_gpu[
    dtype: DType,
    width: Int,
    use_gelu: Bool,
    has_bias: Bool,
](
    out_ptr: MutKernelPtr[dtype],
    pre_gelu_ptr: MutKernelPtr[dtype],
    raw_ptr: ImmutKernelPtr[dtype],
    bias_ptr: ImmutKernelPtr[dtype],
    out_channels: Int,
    num_params: Int,
) -> None:
    # Fused post-matmul epilogue: read the plain GEMM output (raw_ptr == out_ptr),
    # add the per-column bias in fp32, optionally store the pre-activation and the
    # gelu. Replaces `linalg.matmul`'s `elementwise_lambda_fn`, which this backend
    # lowers to a *separate, poorly-occupied* `elementwise` sweep (~6× off
    # bandwidth for the small bias adds). width-aligned and out_channels is a
    # multiple of width, so a width-block never straddles a row → bias[col] is a
    # single contiguous load.
    var idx = Int((block_idx.x * block_dim.x + thread_idx.x) * width)
    if idx + width <= num_params:
        var v = (raw_ptr + idx).load[width=width]().cast[DType.float32]()
        comptime if has_bias:
            var col = idx % out_channels
            v += (bias_ptr + col).load[width=width]().cast[DType.float32]()
        comptime if use_gelu:
            (pre_gelu_ptr + idx).store(v.cast[dtype]())
            (out_ptr + idx).store(gelu[DType.float32, width](v).cast[dtype]())
        else:
            (out_ptr + idx).store(v.cast[dtype]())
    elif idx < num_params:
        for i in range(idx, num_params):
            var v = (raw_ptr + i).load[width=1]().cast[DType.float32]()
            comptime if has_bias:
                var col = i % out_channels
                v += (bias_ptr + col).load[width=1]().cast[DType.float32]()
            comptime if use_gelu:
                (pre_gelu_ptr + i).store(v.cast[dtype]())
                (out_ptr + i).store(gelu[DType.float32, 1](v).cast[dtype]())
            else:
                (out_ptr + i).store(v.cast[dtype]())


def _cublaslt_workspace(
    ctx: DeviceContext,
) raises -> OpaquePointer[MutAnyOrigin]:
    # Persistent 32 MB cuBLASLt workspace (allocate-once, heap-held via a
    # device-keyed process global — mirrors llm.c's single global workspace).
    comptime BufType = type_of(ctx.enqueue_create_buffer[DType.uint8](1))
    var name = String(t"LLMM_CUBLASLT_WS_{ctx.id()}")
    if gp := _get_global_or_null(name):
        var p = gp.value().bitcast[BufType]()
        return p[].unsafe_ptr().bitcast[NoneType]().as_unsafe_any_origin()
    var buf = ctx.enqueue_create_buffer[DType.uint8](_CUBLASLT_WS_BYTES)
    var hp = alloc[BufType](1)
    hp.init_pointee_move(buf^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), hp.bitcast[NoneType]()
    )
    return hp[].unsafe_ptr().bitcast[NoneType]().as_unsafe_any_origin()


@always_inline
def _lt_dt[dtype: DType]() -> DataType:
    comptime if dtype == DType.bfloat16:
        return DataType.R_16BF
    elif dtype == DType.float32:
        return DataType.R_32F
    else:
        return DataType.R_16F


@always_inline
def _lt_set_op(
    desc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    trans: Bool,
) raises:
    var op = (
        cublasOperation_t.CUBLAS_OP_T if trans else cublasOperation_t.CUBLAS_OP_N
    )
    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            desc,
            attr,
            UnsafePointer(to=op)
            .bitcast[NoneType]()
            .as_immutable()
            .as_unsafe_any_origin(),
            size_of[cublasOperation_t](),
        )
    )


def _matmul_cublaslt[
    dtype: DType,
    transA: Bool,
    transB: Bool,
](
    d_ptr: MutKernelPtr[dtype],
    a_ptr: ImmutKernelPtr[dtype],
    b_ptr: ImmutKernelPtr[dtype],
    bias_ptr: ImmutKernelPtr[dtype],  # used iff (epilogue & 4)
    aux_ptr: MutKernelPtr[dtype],  # pre_gelu; used iff (epilogue & 128)
    m: Int,
    n: Int,
    k: Int,
    epilogue: Int32,
    accumulate: Bool,
    ctx: DeviceContext,
    batch_count: Int = 1,
    stride_a: Int = 0,
    stride_b: Int = 0,
    stride_d: Int = 0,
) raises:
    # General cuBLASLt matmul with fused epilogue, replicating llm.c's
    # matmul_cublaslt. Column-major D[m,n]=op(A)·op(B); the bias (epilogue&4) is
    # per-m and the aux/gelu (epilogue&128) uses ld=m. Descriptors are created/
    # destroyed per call (algo cached internally); workspace is a persistent
    # global. Epilogue values: BIAS=4, GELU_AUX_BIAS=164, GELU_AUX=160,
    # DGELU=192, DEFAULT=1.
    var handle = _get_global_handle[dtype, Backend.CUBLASLT](ctx)
    var lt = handle._get_cublas()
    var cuda_stream = CUDA(ctx.stream())
    comptime dt = _lt_dt[dtype]()

    # fp32 GEMMs route through TF32 tensor cores by default (matches llm.c's
    # fp32 arm, which is TF32-accelerated on any compute-capability-8.0+ GPU —
    # see llmm/vendor.mojo's USE_TF32 doc comment); bf16/fp16 keep the
    # existing COMPUTE_32F (mixed-precision GEMMs are already tensor-core
    # eligible by input dtype alone). Disable with -D LLMM_NO_TF32=1 for true
    # IEEE fp32 math.
    comptime compute_type = (
        ComputeType.COMPUTE_32F_FAST_TF32 if (
            dtype == DType.float32 and USE_TF32
        ) else ComputeType.COMPUTE_32F
    )

    var desc = cublasLtMatmulDesc_t()
    check_cublas_error(
        cublasLtMatmulDescCreate(
            UnsafePointer(to=desc).as_unsafe_any_origin(),
            compute_type,
            DataType.R_32F,
        )
    )
    _lt_set_op(
        desc, cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_TRANSA, transA
    )
    _lt_set_op(
        desc, cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_TRANSB, transB
    )

    # Layouts (llm.c): transA→A(dt,k,m,k) else A(dt,m,k,m); transB→B(dt,n,k,n)
    # else B(dt,k,n,k); C/D(dt,m,n,m).
    var a_l = cublasLtMatrixLayout_t()
    comptime if transA:
        check_cublas_error(
            cublasLtMatrixLayoutCreate(
                UnsafePointer(to=a_l).as_unsafe_any_origin(),
                dt,
                UInt64(k),
                UInt64(m),
                Int64(k),
            )
        )
    else:
        check_cublas_error(
            cublasLtMatrixLayoutCreate(
                UnsafePointer(to=a_l).as_unsafe_any_origin(),
                dt,
                UInt64(m),
                UInt64(k),
                Int64(m),
            )
        )
    var b_l = cublasLtMatrixLayout_t()
    comptime if transB:
        check_cublas_error(
            cublasLtMatrixLayoutCreate(
                UnsafePointer(to=b_l).as_unsafe_any_origin(),
                dt,
                UInt64(n),
                UInt64(k),
                Int64(n),
            )
        )
    else:
        check_cublas_error(
            cublasLtMatrixLayoutCreate(
                UnsafePointer(to=b_l).as_unsafe_any_origin(),
                dt,
                UInt64(k),
                UInt64(n),
                Int64(k),
            )
        )
    var c_l = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            UnsafePointer(to=c_l).as_unsafe_any_origin(),
            dt,
            UInt64(m),
            UInt64(n),
            Int64(m),
        )
    )
    var d_l = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            UnsafePointer(to=d_l).as_unsafe_any_origin(),
            dt,
            UInt64(m),
            UInt64(n),
            Int64(m),
        )
    )

    # Strided-batched (llm.c's attention path): set BATCH_COUNT + per-matrix
    # STRIDED_BATCH_OFFSET on every layout. cuBLASLt's heuristic picks the kernel
    # (potentially better than cublasGemmStridedBatchedEx's fixed algo for hd=64).
    if batch_count > 1:
        var bc = Int32(batch_count)

        @parameter
        def _set_batch(lay: cublasLtMatrixLayout_t, stride: Int) raises:
            check_cublas_error(
                cublasLtMatrixLayoutSetAttribute(
                    lay,
                    LayoutAttribute.BATCH_COUNT,
                    UnsafePointer(to=bc)
                    .bitcast[NoneType]()
                    .as_immutable()
                    .as_unsafe_any_origin(),
                    size_of[Int32](),
                )
            )
            var so = Int64(stride)
            check_cublas_error(
                cublasLtMatrixLayoutSetAttribute(
                    lay,
                    LayoutAttribute.STRIDED_BATCH_OFFSET,
                    UnsafePointer(to=so)
                    .bitcast[NoneType]()
                    .as_immutable()
                    .as_unsafe_any_origin(),
                    size_of[Int64](),
                )
            )

        _set_batch(a_l, stride_a)
        _set_batch(b_l, stride_b)
        _set_batch(c_l, stride_d)
        _set_batch(d_l, stride_d)

    var pref = cublasLtMatmulPreference_t()
    check_cublas_error(
        cublasLtMatmulPreferenceCreate(
            UnsafePointer(to=pref).as_unsafe_any_origin()
        )
    )
    var ws_size = _CUBLASLT_WS_BYTES
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

    # aux (gelu/dgelu) — epilogue bit 128.
    if (Int(epilogue) & 128) != 0:
        var gelu_ld = Int64(m)
        check_cublas_error(
            cublasLtMatmulDescSetAttribute(
                desc,
                cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                UnsafePointer(to=gelu_ld)
                .bitcast[NoneType]()
                .as_immutable()
                .as_unsafe_any_origin(),
                size_of[Int64](),
            )
        )
        var aux = aux_ptr
        check_cublas_error(
            cublasLtMatmulDescSetAttribute(
                desc,
                cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                UnsafePointer(to=aux)
                .bitcast[NoneType]()
                .as_immutable()
                .as_unsafe_any_origin(),
                size_of[MutKernelPtr[dtype]](),
            )
        )
    var epi = epilogue
    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_EPILOGUE,
            UnsafePointer(to=epi)
            .bitcast[NoneType]()
            .as_immutable()
            .as_unsafe_any_origin(),
            size_of[Int32](),
        )
    )
    # bias — epilogue bit 4.
    if (Int(epilogue) & 4) != 0:
        var bdt = _lt_dt[dtype]()
        check_cublas_error(
            cublasLtMatmulDescSetAttribute(
                desc,
                cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                UnsafePointer(to=bdt)
                .bitcast[NoneType]()
                .as_immutable()
                .as_unsafe_any_origin(),
                size_of[DataType](),
            )
        )
        var bias_dev = bias_ptr
        check_cublas_error(
            cublasLtMatmulDescSetAttribute(
                desc,
                cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                UnsafePointer(to=bias_dev)
                .bitcast[NoneType]()
                .as_immutable()
                .as_unsafe_any_origin(),
                size_of[ImmutKernelPtr[dtype]](),
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
        raise Error("no cuBLASLt algorithm for the fused matmul")

    var ws = _cublaslt_workspace(ctx)
    var alpha = Float32(1.0)
    var beta = Float32(1.0) if accumulate else Float32(0.0)
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
            ws,
            _CUBLASLT_WS_BYTES,
            cuda_stream.value()[],
        )
    )

    check_cublas_error(cublasLtMatmulPreferenceDestroy(pref))
    check_cublas_error(cublasLtMatmulDescDestroy(desc))
    check_cublas_error(cublasLtMatrixLayoutDestroy(a_l))
    check_cublas_error(cublasLtMatrixLayoutDestroy(b_l))
    check_cublas_error(cublasLtMatrixLayoutDestroy(c_l))
    check_cublas_error(cublasLtMatrixLayoutDestroy(d_l))


def _launch_gelu_fwd_gpu[
    dtype: DType,
](
    out_ptr: MutKernelPtr[dtype],
    x_ptr: ImmutKernelPtr[dtype],
    num_params: Int,
    ctx: DeviceContext,
) raises -> None:
    # USE_GELU_FUSION=False forward path: standalone elementwise GELU kernel
    # applied to the pre-activation buffer a plain-BIAS GEMM just wrote —
    # llm.c's `gelu_forward` (gelu_fusion<1 branch of matmul_forward_cublaslt).
    comptime width = simd_width_of[dtype]()
    comptime BLOCK_SIZE = 256
    var num_threads = ceildiv(num_params, width)
    var num_blocks = ceildiv(num_threads, BLOCK_SIZE)
    comptime gpu_kernel = gelu_fwd_gpu[dtype, width]
    var compiled = ctx.compile_function[gpu_kernel]()
    ctx.enqueue_function(
        compiled,
        out_ptr,
        x_ptr,
        num_params,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )


def _launch_matmul_gelu_backward_scaling_gpu[
    dtype: DType,
](
    d_input_ptr: MutKernelPtr[dtype],
    pre_gelu_ptr: ImmutKernelPtr[dtype],
    num_params: Int,
    ctx: DeviceContext,
) raises -> None:
    # USE_GELU_FUSION=False backward path: standalone in-place gelu-grad
    # scaling kernel (d_input *= gelu'(pre_gelu)) applied after a plain
    # (non-DGELU) d_input GEMM — llm.c's `gelu_backward_inplace`
    # (gelu_fusion<2 branch of matmul_backward).
    comptime width = simd_width_of[dtype]()
    comptime BLOCK_SIZE = 256
    var num_threads = ceildiv(num_params, width)
    var num_blocks = ceildiv(num_threads, BLOCK_SIZE)
    comptime gpu_kernel = matmul_gelu_backward_scaling_gpu[dtype, width]
    var compiled = ctx.compile_function[gpu_kernel]()
    ctx.enqueue_function(
        compiled,
        d_input_ptr,
        pre_gelu_ptr,
        num_params,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )


def matmul_fwd[
    dtype: DType,
    target: StaticString,
    use_gelu: Bool = False,
    has_bias: Bool = True,
](
    out_ptr: MutKernelPtr[dtype],
    pre_gelu_ptr: MutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    weight_ptr: ImmutKernelPtr[dtype],
    bias_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    output_channels: Int64,
    ctx: DeviceContext,
) raises -> None:
    var rows = Int(batch_size * seq_len)  # M in matmul
    var in_channels = Int(channels)  # K in matmul
    var out_channels = Int(output_channels)  # N in matmul
    comptime elem_dtype = dtype

    var c = TileTensor(
        Span[Scalar[dtype], MutAnyOrigin](
            ptr=out_ptr, length=rows * out_channels
        ),
        row_major(rows, out_channels),
    )
    var a = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](
            ptr=input_ptr, length=rows * in_channels
        ),
        row_major(rows, in_channels),
    )
    var b = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](
            ptr=weight_ptr, length=out_channels * in_channels
        ),
        row_major(out_channels, in_channels),
    )

    @parameter
    @always_inline
    def epilogue_with_bias[
        dtype: DType, width: SIMDSize, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        var offset = idx[0] * out_channels + idx[1]
        var v = (
            val.cast[DType.float32]()
            + (bias_ptr + idx[1]).load[width=width]().cast[DType.float32]()
        )
        comptime if use_gelu:
            (pre_gelu_ptr + offset).store(v.cast[elem_dtype]())
            (out_ptr + offset).store(
                gelu[DType.float32, width](v).cast[elem_dtype]()
            )
        else:
            (out_ptr + offset).store(v.cast[elem_dtype]())

    @parameter
    @always_inline
    def epilogue_no_bias[
        dtype: DType, width: SIMDSize, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        var offset = idx[0] * out_channels + idx[1]
        var v = val.cast[DType.float32]()
        comptime if use_gelu:
            (pre_gelu_ptr + offset).store(v.cast[elem_dtype]())
            (out_ptr + offset).store(
                gelu[DType.float32, width](v).cast[elem_dtype]()
            )
        else:
            (out_ptr + offset).store(v.cast[elem_dtype]())

    # Vendor-neutral matmul with the fused bias/gelu epilogue lambdas: used
    # on the CPU target only (linalg applies the epilogue inline; there is no
    # separate-kernel penalty on the CPU path). The non-CUBLAS GPU targets use
    # the plain-GEMM + standalone bias_gelu_fwd split below instead, because
    # Metal's fused epilogue (linalg.matmul elementwise_lambda_fn) is broken.
    # The host fences bracket the call (matching matmul_d_input_bwd's portable
    # fallback below): empirically required on the GPU target — without them
    # this raced against neighboring kernels (CUDA_ERROR_ILLEGAL_ADDRESS) even
    # though the fast cuBLASLt path needs none (its epilogue is inside the
    # vendor GEMM itself, not a separately-launched elementwise sweep).
    @parameter
    @always_inline
    def _portable_matmul() raises:
        ctx.synchronize()
        comptime if has_bias:
            matmul[
                transpose_b=True,
                elementwise_lambda_fn=epilogue_with_bias,
                target=target,
            ](c, a, b, ctx=ctx)
        elif use_gelu:
            matmul[
                transpose_b=True,
                elementwise_lambda_fn=epilogue_no_bias,
                target=target,
            ](c, a, b, ctx=ctx)
        else:
            matmul[transpose_b=True, target=target](c, a, b, ctx=ctx)
        ctx.synchronize()

    # Fused epilogue on cuBLASLt; split GEMM+bias_gelu elsewhere — Metal's
    # elementwise_lambda_fn epilogue produces scrambled output on Metal AIR (G2).
    # See docs/ai/metal_port_gotchas_and_optimizations.md G2.
    comptime if is_gpu[target]():
        comptime if HAS_CUBLAS:
            comptime if use_gelu and not USE_GELU_FUSION:
                # llm.c's default (gelu_fusion=0): plain BIAS-only GEMM
                # writing the pre-activation straight into pre_gelu_ptr (no
                # AUX epilogue), then a separate standalone elementwise GELU
                # kernel pre_gelu_ptr -> out_ptr. Same D-matrix layout
                # (m=out_channels, transA(weight)) as the AUX buffer write in
                # the fused path below, so pre_gelu_ptr ends up bit-identical
                # either way.
                comptime epi = Int32(4) if has_bias else Int32(1)
                _matmul_cublaslt[dtype, transA=True, transB=False](
                    pre_gelu_ptr,
                    weight_ptr,
                    input_ptr,
                    bias_ptr,
                    pre_gelu_ptr,
                    out_channels,
                    rows,
                    in_channels,
                    epi,
                    False,
                    ctx,
                )
                _launch_gelu_fwd_gpu[dtype](
                    out_ptr,
                    rebind[ImmutKernelPtr[dtype]](pre_gelu_ptr),
                    rows * out_channels,
                    ctx,
                )
            else:
                # GPU: single fused cuBLASLt GEMM with bias/gelu epilogue (llm.c's
                # technique) — bias + gelu (+ pre_gelu aux for the backward) happen inside
                # the GEMM. out[rows,OC]=input@weightᵀ → m=OC, n=rows, k=C, transA(weight).
                comptime epi = Int32(164) if (use_gelu and has_bias) else (
                    Int32(160) if use_gelu else (
                        Int32(4) if has_bias else Int32(1)
                    )
                )
                _matmul_cublaslt[dtype, transA=True, transB=False](
                    out_ptr,
                    weight_ptr,
                    input_ptr,
                    bias_ptr,
                    pre_gelu_ptr,
                    out_channels,
                    rows,
                    in_channels,
                    epi,
                    False,
                    ctx,
                )
        else:
            # Vendor-neutral GPU: plain GEMM (no epilogue) + standalone
            # bias(+GELU) kernel. NVIDIA without cuBLAS: host fences guard a real
            # race (CUDA_ERROR_ILLEGAL_ADDRESS without them). Metal: fences are
            # removed (comptime if not HAS_METAL) because the in-order command
            # queue sequences linalg.matmul and the epilogue kernel automatically —
            # validated by _attn_gemm_batched_metal running fence-free and test_gpt2
            # passing. Removing Metal fences saves ~90 ms/step (see P13).
            # See docs/ai/metal_port_gotchas_and_optimizations.md P13.
            comptime if not HAS_METAL:
                ctx.synchronize()
            matmul[transpose_b=True, target=target](c, a, b, ctx=ctx)
            comptime if not HAS_METAL:
                ctx.synchronize()
            comptime if has_bias or use_gelu:
                bias_gelu_fwd[
                    dtype, target, has_bias=has_bias, use_gelu=use_gelu
                ](
                    out_ptr,
                    pre_gelu_ptr,
                    bias_ptr,
                    rows,
                    out_channels,
                    ctx,
                )
    else:
        # CPU: keep the fused epilogue lambda (linalg applies it inline; there is
        # no separate-kernel penalty on the CPU path).
        _portable_matmul()
    # No host fence: forward linear GEMM + bias/gelu run on ctx's stream and the
    # consumer (next op) is stream-ordered after them. (The backward weight-grad
    # accumulation DOES still need its fences — removing those races.)


@compiler.register("matmul_fwd")
struct MatmulFwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        use_gelu: Bool = False,
        has_bias: Bool = True,
    ](
        output: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        pre_gelu: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        x: InputTensor[dtype=dtype, rank=2, static_spec=...],
        weight: InputTensor[dtype=dtype, rank=2, static_spec=...],
        bias: InputTensor[dtype=dtype, rank=1, static_spec=...],
        batch_size: Int64,
        seq_len: Int64,
        channels: Int64,
        output_channels: Int64,
        ctx: DeviceContext,
    ) capturing raises:
        if output.size() != Int(batch_size * seq_len * output_channels):
            raise Error(
                "output must have the same size as batch_size * seq_len *"
                " output_channels"
            )
        # The use_gelu=False instantiation contains no stores to pre_gelu
        # (comptime-dead code), so a dummy buffer of any size is sound there.
        comptime if use_gelu:
            if pre_gelu.size() != Int(batch_size * seq_len * output_channels):
                raise Error(
                    "pre_gelu must have the same size as batch_size * seq_len"
                    " * output_channels"
                )
        if x.size() != Int(batch_size * seq_len * channels):
            raise Error(
                "input must have the same size as batch_size * seq_len *"
                " channels"
            )
        if weight.size() != Int(output_channels * channels):
            raise Error(
                "weight must have the same size as output_channels * channels"
            )
        comptime if has_bias:
            if bias.size() != Int(output_channels):
                raise Error("bias must have the same size as output_channels")

        matmul_fwd[
            dtype,
            target,
            use_gelu=use_gelu,
            has_bias=has_bias,
        ](
            output.unsafe_ptr(),
            pre_gelu.unsafe_ptr(),
            x.unsafe_ptr(),
            weight.unsafe_ptr(),
            bias.unsafe_ptr(),
            batch_size,
            seq_len,
            channels,
            output_channels,
            ctx,
        )


# ===----------------------------------------------------------------------=== #
# Matmul Backward
# ===----------------------------------------------------------------------=== #


@always_inline
def _matmul_bias_bwd_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    accumulate: Bool,
    width: Int,
](
    d_bias_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    rows: Int,
    out_channels: Int,
    num_tiles: Int,
    tid: Int,
    stride: Int,
    block_row: Int,
) -> None:
    for tile in range(block_row, num_tiles, stride):
        var base = tile * width
        if base + width <= out_channels:
            var accumulator = SIMD[DType.float32, width](0.0)
            for r in range(tid, rows, BLOCK_SIZE):
                accumulator += (
                    (d_output_ptr + r * out_channels + base)
                    .load[width=width]()
                    .cast[DType.float32]()
                )
            var tile_sum = SIMD[DType.float32, width](0.0)
            comptime for i in range(width):
                tile_sum[i] = block.sum[block_size=BLOCK_SIZE](accumulator[i])
            if tid == 0:
                comptime if accumulate:
                    var previous = (
                        (d_bias_ptr + base)
                        .load[width=width]()
                        .cast[DType.float32]()
                    )
                    (d_bias_ptr + base).store(
                        (previous + tile_sum).cast[dtype]()
                    )
                else:
                    (d_bias_ptr + base).store(tile_sum.cast[dtype]())
        else:
            # Ragged edge of the last tile, scalar steps. Loop bounds depend
            # only on `base`, so every thread reaches block.sum uniformly.
            for c in range(base, out_channels):
                var accumulator = Scalar[DType.float32](0.0)
                for r in range(tid, rows, BLOCK_SIZE):
                    accumulator += d_output_ptr[r * out_channels + c].cast[
                        DType.float32
                    ]()
                var col_sum = block.sum[block_size=BLOCK_SIZE](accumulator)
                if tid == 0:
                    comptime if accumulate:
                        var previous = d_bias_ptr[c].cast[DType.float32]()
                        d_bias_ptr[c] = (previous + col_sum).cast[dtype]()
                    else:
                        d_bias_ptr[c] = col_sum.cast[dtype]()


def matmul_bias_bwd_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    accumulate: Bool,
    width: Int = 4,
](
    d_bias_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    output_channels: Int64,
) -> None:
    var rows = Int(batch_size * seq_len)
    var out_channels = Int(output_channels)
    var num_tiles = ceildiv(out_channels, width)
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x)
    var block_row = Int(block_idx.x)

    _matmul_bias_bwd_gpu[dtype, BLOCK_SIZE, accumulate, width](
        d_bias_ptr,
        d_output_ptr,
        rows,
        out_channels,
        num_tiles,
        tid,
        stride,
        block_row,
    )


def _dbias_scratch(ctx: DeviceContext) raises -> MutKernelPtr[DType.float32]:
    # Persistent fp32 accumulator for the dbias reduction (allocate-once,
    # heap-held via a process global keyed by device id; big enough for any
    # bias width). Zeroed on allocation; the finalize kernel re-zeros it after
    # each use so successive calls start clean.
    #
    # CAP must hold `row_blocks * out_channels` fp32 elements. The
    # vectorized HAS_CUBLAS fused kernel (`matmul_bias_bwd`) scales
    # row_blocks up by the vector width (`ROW_BLOCKS * width`, up to 16*8
    # = 128 for bf16) to keep grid occupancy up after `width`-ing the
    # column dimension — worst case here is 128 row-blocks * 3072 (the 4C
    # MLP fc bias) = 393,216 elements (1.5 MiB). 1<<20 leaves >2x headroom
    # for that plus the portable (Metal) path's unvectorized 16 * 3072.
    comptime CAP = 1 << 20
    comptime BufType = type_of(ctx.enqueue_create_buffer[DType.float32](1))
    var name = String(t"LLMM_DBIAS_SCRATCH_{ctx.id()}")
    if gp := _get_global_or_null(name):
        var p = gp.value().bitcast[BufType]()
        return rebind[MutKernelPtr[DType.float32]](p[].unsafe_ptr())
    var buf = ctx.enqueue_create_buffer[DType.float32](CAP)
    ctx.enqueue_memset(buf, Float32(0))
    var hp = alloc[BufType](1)
    hp.init_pointee_move(buf^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), hp.bitcast[NoneType]()
    )
    return rebind[MutKernelPtr[DType.float32]](hp[].unsafe_ptr())


def _dbias_counters(ctx: DeviceContext) raises -> MutKernelPtr[DType.int32]:
    # Persistent per-column-block "arrival" counters for the fused dbias
    # reduction's last-block-finalizes signal (NVIDIA-only, kernel9-style —
    # see `_dbias_fused_gpu`). One counter per column-block (`bx` in the
    # launch grid); zeroed on allocation, and the block that finalizes a
    # column-block resets its own slot back to 0 immediately after use, so no
    # per-call host-side memset is needed (mirrors `_dbias_scratch` below).
    # CAP=4096 column-blocks is far beyond any OC/(BLOCK_SIZE*width) this
    # model produces (largest bias is the 4C=3072-wide MLP fc, i.e. at most
    # 12 column-blocks pre-vectorization / 3 post-vectorization).
    comptime CAP = 4096
    comptime BufType = type_of(ctx.enqueue_create_buffer[DType.int32](1))
    var name = String(t"LLMM_DBIAS_COUNTERS_{ctx.id()}")
    if gp := _get_global_or_null(name):
        var p = gp.value().bitcast[BufType]()
        return rebind[MutKernelPtr[DType.int32]](p[].unsafe_ptr())
    var buf = ctx.enqueue_create_buffer[DType.int32](CAP)
    ctx.enqueue_memset(buf, Int32(0))
    var hp = alloc[BufType](1)
    hp.init_pointee_move(buf^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), hp.bitcast[NoneType]()
    )
    return rebind[MutKernelPtr[DType.int32]](hp[].unsafe_ptr())


def _dbias_accum_gpu[
    dtype: DType,
](
    scratch: MutKernelPtr[DType.float32],
    d_output_ptr: ImmutKernelPtr[dtype],
    rows: Int,
    out_channels: Int,
    row_tile: Int,
) -> None:
    # dbias[c] += sum over a row-chunk of dOutput[:, c]. One thread per column
    # (adjacent threads → adjacent columns → COALESCED reads), and grid.y splits
    # the row reduction so occupancy stays high even for small OC. Each block's
    # per-column partial is atomically added into the fp32 scratch.
    var col = Int(block_idx.x * block_dim.x + thread_idx.x)
    if col >= out_channels:
        return
    var by = Int(block_idx.y)
    var r0 = by * row_tile
    var r1 = min(r0 + row_tile, rows)
    var acc = Scalar[DType.float32](0.0)
    for r in range(r0, r1):
        acc += d_output_ptr[r * out_channels + col].cast[DType.float32]()
    # Write this row-block's partial (no atomics — a contention-free write to a
    # [row_blocks, OC] scratch, reduced by the finalize pass). Matches llm.c's
    # kernel9 buffer+reduce approach.
    scratch[by * out_channels + col] = acc


def _dbias_finalize_gpu[
    dtype: DType,
    accumulate: Bool,
](
    d_bias_ptr: MutKernelPtr[dtype],
    scratch: MutKernelPtr[DType.float32],
    out_channels: Int,
    row_blocks: Int,
) -> None:
    var col = Int(block_idx.x * block_dim.x + thread_idx.x)
    if col < out_channels:
        var v = Scalar[DType.float32](0.0)
        for by in range(row_blocks):
            v += scratch[by * out_channels + col]
        comptime if accumulate:
            d_bias_ptr[col] = (d_bias_ptr[col].cast[DType.float32]() + v).cast[
                dtype
            ]()
        else:
            d_bias_ptr[col] = v.cast[dtype]()


def _dbias_fused_gpu[
    dtype: DType,
    accumulate: Bool,
    width: Int,
](
    d_bias_ptr: MutKernelPtr[dtype],
    scratch: MutKernelPtr[DType.float32],
    counters: MutKernelPtr[DType.int32],
    d_output_ptr: ImmutKernelPtr[dtype],
    rows: Int,
    out_channels: Int,
    row_tile: Int,
    row_blocks: Int,
) -> None:
    # NVIDIA-only (HAS_CUBLAS-gated): matmul_backward_bias_kernel9-style fused
    # dbias reduction — one launch per call site instead of the accum+finalize
    # pair above. Same partial layout as `_dbias_accum_gpu`
    # (row-block-contention-free [row_blocks, OC] scratch), but the last
    # row-block to finish for a given column-block finalizes in the same
    # launch instead of a second kernel doing it. "Last block" is detected
    # with the classic CUDA global-sync idiom (NVIDIA's `threadFenceReduction`
    # sample / llm.c's own `global_sum_deterministic`): each writer thread
    # __threadfence()s to publish its scratch write GPU-wide, thread 0 then
    # atomically increments a per-column-block arrival counter and broadcasts
    # via shared memory whether it was the last of `row_blocks` arrivals; only
    # that block reads the full column back out of scratch and finalizes.
    #
    # VECTORIZED (kernel9-style x128/f128 access): each thread now owns a
    # contiguous `width`-wide run of output columns (width = one 128-bit
    # transaction: 4 fp32 lanes or 8 bf16 lanes) instead of a single scalar
    # column. Adjacent threads own adjacent runs, so a warp's row-load lands
    # in one contiguous, fully-coalesced stretch of `d_output` — matching
    # llm.c's `x128 packed_dout = load128(dout + global_oc + idx*OC)` in
    # `matmul_backward_bias_kernel9` (llmc/matmul.cuh:41). The scratch
    # writes/reads and final dbias write are vectorized the same way (still
    # the same fp32 `[row_blocks, OC]` scratch layout — just written/read
    # `width` columns at a time).
    #
    # Grid geometry (grid.x column-blocks, grid.y row-blocks, threads/block)
    # is chosen by the caller (`matmul_bias_bwd`) to keep every column-block
    # fully packed (no idle threads from `width`-ing the column dimension)
    # and to keep total block count high enough to fill the GPU's SMs
    # despite grid.x shrinking by ~`width`× — see the comment there. Launch
    # count itself (one launch per `matmul_bias_bwd` call, 48/step) is
    # unchanged from the scalar fused kernel this replaces.
    #
    # REQUIRES out_channels % width == 0 (checked by the caller before
    # launch, not re-checked here — every bias this model produces, C/3C/4C
    # at 768/2304/3072, is a multiple of 8, so this never trips in practice;
    # since `col` only ever takes multiples of `width` and out_channels
    # itself is a multiple of `width`, `col < out_channels` already implies
    # `col + width <= out_channels`, so no separate ragged-edge/tail branch
    # is needed here). A future irregular OC would need a scalar-tail path,
    # deliberately not added here to keep the hot loop branch-free — see the
    # 2026-07-10 dbias-vectorize writeup in
    # docs/ai/ai_assisted_optimizations_and_benchmarks.md.
    #
    # GOTCHA: this relies on `threadfence(Scope.GPU)`, which is NVIDIA-only
    # (comptime-asserts `is_nvidia_gpu()`) — never call this kernel outside a
    # `HAS_CUBLAS` branch (see `matmul_bias_bwd` below); Apple Metal keeps the
    # portable `_dbias_accum_gpu` + `_dbias_finalize_gpu` two-launch path.
    var bx = Int(block_idx.x)
    var by = Int(block_idx.y)
    var col = (bx * Int(block_dim.x) + Int(thread_idx.x)) * width

    if col < out_channels:
        var r0 = by * row_tile
        var r1 = min(r0 + row_tile, rows)
        var acc = SIMD[DType.float32, width](0.0)
        for r in range(r0, r1):
            acc += (
                (d_output_ptr + r * out_channels + col)
                .load[width=width]()
                .cast[DType.float32]()
            )
        (scratch + by * out_channels + col).store(acc)

    # Every thread that may have written above must fence its own write —
    # __threadfence() only orders the calling thread's prior stores, so a
    # single "leader" fence would not publish the other threads' columns.
    threadfence[Scope.GPU]()

    var flag = stack_allocation[
        1, DType.int32, address_space=AddressSpace.SHARED
    ]()
    if Int(thread_idx.x) == 0:
        var arrived = Atomic[DType.int32].fetch_add(counters + bx, 1)
        flag[0] = 1 if arrived == Int32(row_blocks - 1) else 0
    barrier()
    if flag[0] == 0:
        return
    if col >= out_channels:
        return

    var total = SIMD[DType.float32, width](0.0)
    for rb in range(row_blocks):
        total += (scratch + rb * out_channels + col).load[width=width]()
    comptime if accumulate:
        var previous = (
            (d_bias_ptr + col).load[width=width]().cast[DType.float32]()
        )
        (d_bias_ptr + col).store((previous + total).cast[dtype]())
    else:
        (d_bias_ptr + col).store(total.cast[dtype]())

    # Self-reset: the next call reusing this column-block's counter slot
    # (next layer / next grad-accum micro-batch) must see 0 again. No
    # host-side memset between launches (mirrors the scratch buffer, which
    # `_dbias_accum_gpu`'s comment already notes is fully overwritten, not
    # zeroed, every call).
    if Int(thread_idx.x) == 0:
        counters[bx] = 0


@always_inline
def matmul_bias_bwd_cpu[
    dtype: DType,
    width: Int,
    accumulate: Bool,
](
    d_bias_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    rows: Int,  # (B * T) or (batch_size * seq_len)
    out_channels: Int,  # OC
) raises -> None:
    var max_workers = parallelism_level()
    var cols_per_worker = ceildiv(out_channels, max_workers)
    var num_workers = ceildiv(out_channels, cols_per_worker)

    @parameter
    def _worker(w: Int):
        var base = w * cols_per_worker
        var count = min(cols_per_worker, out_channels - base)

        @always_inline
        def _simd[
            w: Int,
        ](local: Int) {d_bias_ptr, d_output_ptr, base, rows, out_channels,}:
            var idx = base + local
            var accumulator = SIMD[DType.float32, w](0.0)
            for r in range(rows):
                var offset = r * out_channels + idx
                accumulator += (
                    (d_output_ptr + offset)
                    .load[width=w]()
                    .cast[DType.float32]()
                )
            comptime if accumulate:
                var previous = (
                    (d_bias_ptr + idx).load[width=w]().cast[DType.float32]()
                )
                (d_bias_ptr + idx).store((previous + accumulator).cast[dtype]())
            else:
                (d_bias_ptr + idx).store(accumulator.cast[dtype]())

        vectorize[width, unroll_factor=UNROLL](count, _simd)

    traced_parallelize["matmul_bias_bwd", _worker](num_workers)


def matmul_bias_bwd[
    dtype: DType,
    target: StaticString,
    accumulate: Bool,
](
    d_bias_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    output_channels: Int64,
    ctx: DeviceContext,
) raises -> None:
    comptime if is_cpu[target]():
        comptime simd_width = simd_width_of[DType.float32]()
        matmul_bias_bwd_cpu[dtype, simd_width, accumulate](
            d_bias_ptr,
            d_output_ptr,
            Int(batch_size * seq_len),
            Int(output_channels),
        )
    elif is_gpu[target]():
        # Coalesced, row-parallel dbias: one thread per column, grid.y row-blocks
        # split the row reduction into per-row-block partials in an fp32
        # scratch. Replaces the uncoalesced row-strided block-reduction (was
        # ~2.6× slower per call than llm.c's).
        comptime BLOCK_SIZE = 256
        comptime ROW_BLOCKS = 16
        var device_ctx = ctx
        var oc = Int(output_channels)
        var rows = Int(batch_size * seq_len)
        var row_tile = ceildiv(rows, ROW_BLOCKS)
        var col_blocks = max(ceildiv(oc, BLOCK_SIZE), 1)
        var scratch = _dbias_scratch(device_ctx)

        comptime if HAS_CUBLAS:
            # matmul_backward_bias_kernel9-style single launch: last row-block
            # to finish a column-block finalizes in-kernel via a threadfence +
            # atomic-counter "last block" flag. NVIDIA-only (threadfence is
            # CUDA-only) — see `_dbias_fused_gpu`. Vectorized 128-bit-wide
            # (kernel9's x128/f128 access pattern): `width` = 4 fp32 lanes /
            # 8 bf16 lanes, same idiom as every other GPU elementwise kernel
            # in this file (`_launch_gelu_fwd_gpu` etc). Two grid-shape
            # adjustments beyond "each thread now loads `width` columns
            # instead of 1", both load-bearing and both found empirically
            # (see the 2026-07-10 dbias-vectorize writeup in
            # docs/ai/ai_assisted_optimizations_and_benchmarks.md for the
            # full sweep — a naive port that just widened per-thread loads
            # 1:1 was 2x SLOWER, not faster):
            #
            # 1. Column-block granularity: widening the per-thread column
            #    count shrinks the natural column-block count (grid.x) by
            #    ~`width`×, and with a FIXED BLOCK_SIZE threads/block the
            #    LAST column-block of a call is often ragged (e.g. oc=2304
            #    bf16 has num_groups=288 width-8 vectors: one full 256-thread
            #    block + one only 32/256 = 12.5% active). Re-deriving
            #    threads/block as `num_groups` evenly divided by the chosen
            #    column-block count keeps every block fully packed instead
            #    (288 groups → 2 blocks of 144, zero idle threads). Halving
            #    BLOCK_SIZE as the block-count divisor (rather than using it
            #    directly) trades a bit more column parallelism for smaller,
            #    still-fully-packed blocks — measured faster than both
            #    BLOCK_SIZE and BLOCK_SIZE//4 on this GPU.
            # 2. Row-block count: the column-side shrink alone starves
            #    occupancy — as few as col_blocks=1 * ROW_BLOCKS=16 = 16
            #    blocks for the 768-wide biases vs this GPU's 48 SMs
            #    (`torch.cuda.get_device_properties`), which measured ~2x
            #    SLOWER than the scalar kernel. Scaling row-blocks by
            #    `width` (`FUSED_ROW_BLOCKS`) restores enough total blocks
            #    to fill the GPU again — trading `width`× fewer, `width`×
            #    shorter-row-tile blocks per column for `width`-wide
            #    per-thread loads, on top of the vectorized-load win.
            #    Over/under-shooting this factor (tried 0.5x, 1.5x, 2x) both
            #    measured worse; 1x was the local optimum.
            comptime width = simd_width_of[dtype]()
            if oc % width != 0:
                raise Error(
                    "matmul_bias_bwd: out_channels ("
                    + String(oc)
                    + ") must be a multiple of the dbias vector width ("
                    + String(width)
                    + ") — the fused GPU kernel has no scalar-tail path"
                )
            # num_groups = number of width-wide column vectors (exact, since
            # oc % width == 0 is checked above).
            var num_groups = oc // width
            var fused_col_blocks = max(ceildiv(num_groups, BLOCK_SIZE // 2), 1)
            var fused_block_threads = ceildiv(num_groups, fused_col_blocks)
            comptime FUSED_ROW_BLOCKS = ROW_BLOCKS * width
            var fused_row_tile = ceildiv(rows, FUSED_ROW_BLOCKS)
            var counters = _dbias_counters(device_ctx)
            comptime fused_k = _dbias_fused_gpu[dtype, accumulate, width]
            var fused_c = device_ctx.compile_function[fused_k]()
            device_ctx.enqueue_function(
                fused_c,
                d_bias_ptr,
                scratch,
                counters,
                d_output_ptr,
                rows,
                oc,
                fused_row_tile,
                FUSED_ROW_BLOCKS,
                grid_dim=(fused_col_blocks, FUSED_ROW_BLOCKS),
                block_dim=(fused_block_threads,),
            )
        else:
            # Portable (Apple Metal) path: two launches — contention-free
            # per-row-block partials, then a separate finalize pass writes
            # d_bias (and re-zeros the scratch for the next call).
            comptime accum_k = _dbias_accum_gpu[dtype]
            var accum_c = device_ctx.compile_function[accum_k]()
            device_ctx.enqueue_function(
                accum_c,
                scratch,
                d_output_ptr,
                rows,
                oc,
                row_tile,
                grid_dim=(col_blocks, ROW_BLOCKS),
                block_dim=(BLOCK_SIZE,),
            )
            comptime fin_k = _dbias_finalize_gpu[dtype, accumulate]
            var fin_c = device_ctx.compile_function[fin_k]()
            device_ctx.enqueue_function(
                fin_c,
                d_bias_ptr,
                scratch,
                oc,
                ROW_BLOCKS,
                grid_dim=(col_blocks,),
                block_dim=(BLOCK_SIZE,),
            )
    else:
        raise Error("Invalid target")


@always_inline
def _matmul_gelu_backward_scaling[
    dtype: DType,
    width: Int,
](
    idx: Int,
    d_input_ptr: MutKernelPtr[dtype],
    pre_gelu_ptr: ImmutKernelPtr[dtype],
) -> None:
    var dy = (d_input_ptr + idx).load[width=width]().cast[DType.float32]()
    var pre_gelu = (
        (pre_gelu_ptr + idx).load[width=width]().cast[DType.float32]()
    )
    var scaled = dy * gelu_grad[DType.float32, width](pre_gelu)
    (d_input_ptr + idx).store(scaled.cast[dtype]())


def matmul_gelu_backward_scaling_gpu[
    dtype: DType,
    width: Int,
](
    d_input_ptr: MutKernelPtr[dtype],
    pre_gelu_ptr: ImmutKernelPtr[dtype],
    num_params: Int,
) -> None:
    var idx = Int((block_idx.x * block_dim.x + thread_idx.x) * width)
    if idx + width <= num_params:
        _matmul_gelu_backward_scaling[dtype, width](
            idx, d_input_ptr, pre_gelu_ptr
        )
    elif idx < num_params:
        for i in range(idx, num_params):
            _matmul_gelu_backward_scaling[dtype, 1](
                i, d_input_ptr, pre_gelu_ptr
            )


def matmul_d_input_bwd[
    dtype: DType,
    target: StaticString,
    use_gelu: Bool,
](
    d_input_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    weight_ptr: ImmutKernelPtr[dtype],
    pre_gelu_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    output_channels: Int64,
    ctx: DeviceContext,
) raises -> None:
    """Computes d_input = d_output @ weight. Overwrites d_input: llm.c overwrites
    activation grads; only weight/bias grads accumulate across micro-steps."""
    var rows = Int(batch_size * seq_len)
    var in_channels = Int(channels)
    var out_channels = Int(output_channels)
    comptime elem_dtype = dtype

    var c_d_input = TileTensor(
        Span[Scalar[dtype], MutAnyOrigin](
            ptr=d_input_ptr, length=rows * in_channels
        ),
        row_major(rows, in_channels),
    )
    var a_d_output = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](
            ptr=d_output_ptr, length=rows * out_channels
        ),
        row_major(rows, out_channels),
    )
    var b_weight = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](
            ptr=weight_ptr, length=out_channels * in_channels
        ),
        row_major(out_channels, in_channels),
    )

    # Vendor-neutral matmul with the gelu-backward epilogue lambda: used
    # unconditionally on CPU, and on GPU whenever HAS_CUBLAS is False.
    @parameter
    @always_inline
    def _portable_matmul_d_input() raises:
        ctx.synchronize()

        @parameter
        @always_inline
        def d_input_epilogue[
            dtype: DType, width: SIMDSize, *, alignment: Int = 1
        ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
            var offset = idx[0] * in_channels + idx[1]
            var v = val.cast[DType.float32]()
            var pre_gelu = (
                (pre_gelu_ptr + offset)
                .load[width=width]()
                .cast[DType.float32]()
            )
            v *= gelu_grad[DType.float32, width](pre_gelu)
            (d_input_ptr + offset).store(v.cast[elem_dtype]())

        comptime if use_gelu:
            matmul[
                transpose_b=False,
                elementwise_lambda_fn=d_input_epilogue,
                target=target,
            ](c_d_input, a_d_output, b_weight, ctx=ctx)
        else:
            matmul[transpose_b=False, target=target](
                c_d_input, a_d_output, b_weight, ctx=ctx
            )
        ctx.synchronize()

    comptime if is_gpu[target]():
        comptime if HAS_CUBLAS:
            comptime if use_gelu and not USE_GELU_FUSION:
                # llm.c's default (gelu_fusion<2): plain (DEFAULT-epilogue)
                # d_input GEMM, then a separate standalone in-place
                # gelu-grad-scaling kernel (llm.c's `gelu_backward_inplace`).
                _matmul_cublaslt[dtype, transA=False, transB=False](
                    d_input_ptr,
                    weight_ptr,
                    d_output_ptr,
                    weight_ptr,  # dummy (no bias bit)
                    rebind[MutKernelPtr[dtype]](pre_gelu_ptr),
                    in_channels,
                    rows,
                    out_channels,
                    Int32(1),  # DEFAULT
                    False,
                    ctx,
                )
                _launch_matmul_gelu_backward_scaling_gpu[dtype](
                    d_input_ptr,
                    pre_gelu_ptr,
                    rows * in_channels,
                    ctx,
                )
            else:
                # cuBLASLt d_input = d_output·W (m=C, n=rows, k=OC, no transpose), with the
                # gelu backward FUSED via the DGELU epilogue (multiplies by gelu'(pre_gelu)
                # inside the GEMM — removes the separate gelu-grad kernel). llm.c's
                # gelu_fusion>=2 path.
                comptime epi_di = Int32(192) if use_gelu else Int32(
                    1
                )  # DGELU / DEFAULT
                _matmul_cublaslt[dtype, transA=False, transB=False](
                    d_input_ptr,
                    weight_ptr,
                    d_output_ptr,
                    weight_ptr,  # dummy (no bias bit)
                    rebind[MutKernelPtr[dtype]](pre_gelu_ptr),
                    in_channels,
                    rows,
                    out_channels,
                    epi_di,
                    False,
                    ctx,
                )
        else:
            comptime if HAS_METAL:
                # Apple Metal path: linalg.matmul's elementwise_lambda_fn epilogue
                # produces wrong results on Metal (same bug class as the forward
                # bias epilogue — see probe_matmul_bias.mojo / probe_matmul_bwd.mojo).
                # Fix: plain GEMM into d_input, then a standalone gelu-grad scaling
                # kernel. Mirrors the USE_GELU_FUSION=False cuBLAS path above.
                # Metal: linalg.matmul is stream-ordered; no host fence needed
                # (same stream-ordering guarantee as _attn_gemm_batched_metal).
                matmul[transpose_b=False, target=target](
                    c_d_input, a_d_output, b_weight, ctx=ctx
                )
                comptime if use_gelu:
                    _launch_matmul_gelu_backward_scaling_gpu[dtype](
                        d_input_ptr,
                        pre_gelu_ptr,
                        rows * in_channels,
                        ctx,
                    )
            else:
                _portable_matmul_d_input()
    else:
        _portable_matmul_d_input()


@always_inline
def _add_into[
    dtype: DType,
    width: Int,
](
    dst_ptr: MutKernelPtr[dtype],
    src_ptr: MutKernelPtr[dtype],
    total: Int,
) raises -> None:
    """dst += src elementwise, f32 math, one rounding at the store."""
    var max_workers = parallelism_level()
    var chunk = ceildiv(total, max_workers)
    var num_workers = ceildiv(total, chunk)

    @parameter
    def _worker(w: Int):
        var base = w * chunk
        var count = min(chunk, total - base)

        @always_inline
        def _simd[
            w_: Int,
        ](local: Int) {dst_ptr, src_ptr, base}:
            var idx = base + local
            var a = (dst_ptr + idx).load[width=w_]().cast[DType.float32]()
            var b = (src_ptr + idx).load[width=w_]().cast[DType.float32]()
            (dst_ptr + idx).store((a + b).cast[dtype]())

        vectorize[width, unroll_factor=UNROLL](count, _simd)

    traced_parallelize["matmul_d_input_bwd", _worker](num_workers)


def _gpu_transpose_kernel[
    dtype: DType
](
    dst: MutKernelPtr[dtype],
    src: ImmutKernelPtr[dtype],
    rows: Int,
    cols: Int,
) -> None:
    """Transpose src[rows, cols] (row-major) into dst[cols, rows] (row-major).
    Used on Apple Metal to materialise d_outputᵀ[OC, rows] into the scratch
    buffer so `linalg.matmul.matmul(transpose_b=False)` can compute
    d_weight[OC, C] = scratch[OC, rows] @ input[rows, C].
    1-D grid: thread i handles element (i/cols, i%cols)."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < rows * cols:
        var r = idx / cols
        var c = idx % cols
        dst[c * rows + r] = src[idx]  # src[idx] == src[r*cols + c]


def _gpu_add_into_kernel[
    dtype: DType
](dst: MutKernelPtr[dtype], src: MutKernelPtr[dtype], total: Int,) -> None:
    """Elementwise dst[i] += src[i] with fp32 accumulation.
    Used on Apple Metal to fold a freshly-computed d_weight GEMM result into
    the running gradient accumulator (beta=1 path, avoiding the epilogue-
    reads-overwritten-C double-count that the CPU branch guards against)."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < total:
        dst[idx] = (
            dst[idx].cast[DType.float32]() + src[idx].cast[DType.float32]()
        ).cast[dtype]()


def _gpu_transpose_add_into_kernel[
    dtype: DType,
    accumulate: Bool,
    BLOCK_SIZE: Int,
](
    dst: MutKernelPtr[dtype],  # d_weight[OC, C] row-major
    src: MutKernelPtr[dtype],  # d_weight_T[C, OC] row-major (read-only here)
    C_: Int,
    OC_: Int,
) -> None:
    """Tiled shared-memory transpose-fold src[C,OC] into dst[OC,C].
    See docs/ai/metal_port_gotchas_and_optimizations.md P14 (choosing the
    smaller operand to pre-transpose) and P15 (32×33 padding eliminates
    bank conflicts in this kernel).

    Each 32×32 tile of src is loaded coalesced into shared memory padded to
    32×33 (one extra column to avoid bank conflicts on the transposed read),
    then written transposed back to dst with coalesced stores.

    Metal safety: no early-returns before barrier() — all threads in the
    threadgroup reach each barrier() call. Out-of-bounds guards are placed
    inside the load/store loops so inactive threads still hit the barrier.
    Metal fix: tile.ptr[i] is used directly to preserve AddressSpace.SHARED;
    GENERIC address-space casts corrupt threadgroup pointers on Metal AIR.

    accumulate=False: dst[oc*C+c]  = src[c*OC+oc]           (overwrite)
    accumulate=True:  dst[oc*C+c] += src[c*OC+oc]  (fp32 accumulation)
    """
    comptime TILE = 32
    comptime STRIDE = TILE + 1  # 33 — avoids shared-memory bank conflicts
    var tile = LayoutTensor[
        dtype,
        Layout.row_major(TILE, STRIDE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var tiles_c = ceildiv(C_, TILE)  # tiles along C  dimension (src rows)
    var tiles_oc = ceildiv(OC_, TILE)  # tiles along OC dimension (src cols)
    var total_tiles = tiles_c * tiles_oc

    # Flatten 1-D block: tx = column-within-tile, ty = row-within-tile.
    var tx = Int(thread_idx.x) % TILE
    var ty = Int(thread_idx.x) // TILE
    comptime ROW_STEP = BLOCK_SIZE // TILE  # = 8 for BLOCK_SIZE=256

    var bt = Int(block_idx.x)
    while bt < total_tiles:
        var tile_c = (bt // tiles_oc) * TILE  # row offset in C  dimension
        var tile_oc = (bt % tiles_oc) * TILE  # col offset in OC dimension

        # Phase 1 — coalesced load from global memory into shared memory.
        # Thread tx reads src column (tile_oc + tx); consecutive tx values map
        # to consecutive src columns → coalesced read.
        # Guard is inside the loop so all threads still reach barrier().
        var r = ty
        while r < TILE:
            var gr = tile_c + r  # global C  index (src row)
            var goc = tile_oc + tx  # global OC index (src col)
            if gr < C_ and goc < OC_:
                tile.ptr[r * STRIDE + tx] = src[gr * OC_ + goc]
            r += ROW_STEP
        barrier()

        # Phase 2 — transposed write from shared memory to global memory.
        # Thread tx writes to dst column (tile_c + tx); consecutive tx values
        # map to consecutive dst columns → coalesced write.
        r = ty
        while r < TILE:
            var goc = tile_oc + r  # global OC index (dst row)
            var gc = tile_c + tx  # global C  index (dst col)
            if goc < OC_ and gc < C_:
                var v = tile.ptr[tx * STRIDE + r].cast[DType.float32]()
                comptime if accumulate:
                    dst[goc * C_ + gc] = (
                        dst[goc * C_ + gc].cast[DType.float32]() + v
                    ).cast[dtype]()
                else:
                    dst[goc * C_ + gc] = v.cast[dtype]()
            r += ROW_STEP
        barrier()

        bt += Int(grid_dim.x)


def matmul_d_weight_bwd[
    dtype: DType,
    target: StaticString,
    accumulate: Bool,
](
    d_weight_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    scratch_ptr: MutKernelPtr[dtype],
    batch_size: Int64,  # B
    seq_len: Int64,  # T
    channels: Int64,  # C
    output_channels: Int64,  # OC
    ctx: DeviceContext,
) raises -> None:
    """Computes d_weight = d_output^T @ input. linalg.matmul rejects transpose_a (#6626), so
    GPU goes through the vendor BLAS (transposed A is native and beta folds
    in accumulation) and CPU materializes d_output^T once into scratch
    (O(rows * OC) traffic next to the GEMM's O(2 * rows * OC * C) flops)."""
    var rows = Int(batch_size * seq_len)
    var in_channels = Int(channels)
    var out_channels = Int(output_channels)

    var c_d_weight = TileTensor(
        Span[Scalar[dtype], MutAnyOrigin](
            ptr=d_weight_ptr, length=out_channels * in_channels
        ),
        row_major(out_channels, in_channels),
    )
    var a_d_output = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](
            ptr=d_output_ptr, length=rows * out_channels
        ),
        row_major(rows, out_channels),
    )
    var b_input = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](
            ptr=input_ptr, length=rows * in_channels
        ),
        row_major(rows, in_channels),
    )

    comptime if is_gpu[target]():
        comptime if HAS_CUBLAS:
            # cuBLASLt d_weight[OC,C] = d_outputᵀ·input, all heads via col-major
            # m=C, n=OC, k=rows with a=input(transA=false), b=d_output(transB=true);
            # beta folds gradient accumulation. Same-stream, no host fence.
            _matmul_cublaslt[dtype, transA=False, transB=True](
                d_weight_ptr,
                input_ptr,
                d_output_ptr,
                input_ptr,  # dummy (no bias)
                d_weight_ptr,  # dummy aux (no gelu)
                in_channels,
                out_channels,
                rows,
                Int32(1),  # DEFAULT
                accumulate,
                ctx,
            )
        else:
            comptime if HAS_METAL:
                # Apple Metal: linalg.matmul rejects transpose_a (issue #6626).
                # To compute d_weight[OC,C] = d_outputᵀ[OC,rows] @ input[rows,C]
                # we choose which operand to pre-transpose based on relative size:
                #
                #   C < OC  → transpose the SMALLER input[rows,C] → input_T[C,rows],
                #             compute d_weight_T[C,OC] = input_T @ d_output
                #             (transpose_b=False, no big-matrix transpose), then
                #             fold into d_weight[OC,C] via _gpu_transpose_add_into_kernel.
                #             For LM-head (C=768, OC=50304, rows=4096) this replaces
                #             the rows×OC transpose (206 M elements) with rows×C
                #             (3.1 M) + C×OC result-fold (38.6 M) — ~5× less work.
                #
                #   C ≥ OC  → transpose d_output (original strategy, smaller matrix
                #             in this case; e.g. proj: C=3072, OC=768).
                #
                # All ops are stream-ordered on Metal; no host fence needed (P13).
                # See docs/ai/metal_port_gotchas_and_optimizations.md G6 (why
                # linalg.matmul rejects transpose_a), P14 (smaller-operand choice),
                # P15 (tiled transpose-add kernel eliminates bank conflicts).
                comptime _TRANS_BLOCK = 256
                if in_channels < out_channels:
                    # ---- small-operand path: transpose input (C × rows << OC × rows) ----
                    var t_in_total = rows * in_channels
                    var input_T_buf = ctx.enqueue_create_buffer[dtype](
                        t_in_total
                    )
                    var input_T_ptr = rebind[MutKernelPtr[dtype]](
                        input_T_buf.unsafe_ptr()
                    )
                    comptime t_k = _gpu_transpose_kernel[dtype]
                    var t_c = ctx.compile_function[t_k]()
                    # Transpose input[rows, C] → input_T[C, rows]
                    ctx.enqueue_function(
                        t_c,
                        input_T_ptr,
                        input_ptr,
                        rows,
                        in_channels,
                        grid_dim=(ceildiv(t_in_total, _TRANS_BLOCK),),
                        block_dim=(_TRANS_BLOCK,),
                    )
                    # input_T_ptr holds inputᵀ [C, rows] row-major.
                    var a_input_T = TileTensor(
                        Span[Scalar[dtype], ImmutAnyOrigin](
                            ptr=rebind[ImmutKernelPtr[dtype]](input_T_ptr),
                            length=t_in_total,
                        ),
                        row_major(in_channels, rows),
                    )
                    # d_weight_T[C, OC] = input_T[C, rows] @ d_output[rows, OC]
                    var dw_t_total = in_channels * out_channels
                    var dw_t_buf = ctx.enqueue_create_buffer[dtype](dw_t_total)
                    var dw_t_ptr = rebind[MutKernelPtr[dtype]](
                        dw_t_buf.unsafe_ptr()
                    )
                    var c_dw_t = TileTensor(
                        Span[Scalar[dtype], MutAnyOrigin](
                            ptr=dw_t_ptr, length=dw_t_total
                        ),
                        row_major(in_channels, out_channels),
                    )
                    matmul[transpose_b=False, target=target](
                        c_dw_t, a_input_T, a_d_output, ctx=ctx
                    )
                    # Fold d_weight_T[C,OC] into d_weight[OC,C] with a tiled
                    # shared-memory transpose-add kernel (32×32 tiles, 32×33
                    # padded smem to avoid bank conflicts, coalesced r/w).
                    comptime tadd_k = _gpu_transpose_add_into_kernel[
                        dtype, accumulate, _TRANS_BLOCK
                    ]
                    var tadd_c = ctx.compile_function[tadd_k]()
                    var tadd_tiles = ceildiv(in_channels, 32) * ceildiv(
                        out_channels, 32
                    )
                    ctx.enqueue_function(
                        tadd_c,
                        d_weight_ptr,
                        dw_t_ptr,
                        in_channels,
                        out_channels,
                        grid_dim=(tadd_tiles,),
                        block_dim=(_TRANS_BLOCK,),
                    )
                    # input_T_buf / dw_t_buf destruct here; the GPU commands hold
                    # the underlying allocations alive until stream completion.
                else:
                    # ---- large-C path: transpose d_output (C ≥ OC, OC×rows ≤ C×rows) ----
                    var t_total = rows * out_channels
                    # Use a local device buffer — never scratch_ptr. On GPU builds
                    # the caller may pass a zero-sized scratch (e.g. grad_acts.logits
                    # / grad_acts.fch on GPU; see allocate_activations).
                    var transpose_buf = ctx.enqueue_create_buffer[dtype](
                        t_total
                    )
                    var transpose_ptr = rebind[MutKernelPtr[dtype]](
                        transpose_buf.unsafe_ptr()
                    )
                    comptime t_k = _gpu_transpose_kernel[dtype]
                    var t_c = ctx.compile_function[t_k]()
                    ctx.enqueue_function(
                        t_c,
                        transpose_ptr,
                        d_output_ptr,
                        rows,
                        out_channels,
                        grid_dim=(ceildiv(t_total, _TRANS_BLOCK),),
                        block_dim=(_TRANS_BLOCK,),
                    )
                    # transpose_ptr holds d_outputᵀ [OC, rows] row-major.
                    var scratch_t_gpu = TileTensor(
                        Span[Scalar[dtype], MutAnyOrigin](
                            ptr=transpose_ptr, length=out_channels * rows
                        ),
                        row_major(out_channels, rows),
                    )
                    comptime if accumulate:
                        # Materialise GEMM into a temp device buffer, then GPU-add
                        # into d_weight. Avoids the epilogue-reads-overwritten-C
                        # double-count (see CPU branch comment above).
                        var temp_buf = ctx.enqueue_create_buffer[dtype](
                            out_channels * in_channels
                        )
                        var temp_ptr = rebind[MutKernelPtr[dtype]](
                            temp_buf.unsafe_ptr()
                        )
                        var c_temp = TileTensor(
                            Span[Scalar[dtype], MutAnyOrigin](
                                ptr=temp_ptr,
                                length=out_channels * in_channels,
                            ),
                            row_major(out_channels, in_channels),
                        )
                        matmul[transpose_b=False, target=target](
                            c_temp, scratch_t_gpu, b_input, ctx=ctx
                        )
                        comptime _ADD_BLOCK = 256
                        var a_total = out_channels * in_channels
                        comptime add_k = _gpu_add_into_kernel[dtype]
                        var add_c = ctx.compile_function[add_k]()
                        ctx.enqueue_function(
                            add_c,
                            d_weight_ptr,
                            temp_ptr,
                            a_total,
                            grid_dim=(ceildiv(a_total, _ADD_BLOCK),),
                            block_dim=(_ADD_BLOCK,),
                        )
                        # temp_buf / transpose_buf destruct here; stream keeps
                        # allocations alive until GPU completion.
                    else:
                        matmul[transpose_b=False, target=target](
                            c_d_weight, scratch_t_gpu, b_input, ctx=ctx
                        )
                    # transpose_buf destructs here.
            else:
                # AMD / other vendor-BLAS GPU: blas.matmul resolves to
                # rocBLAS/hipBLASLt which supports transpose_a natively and
                # folds beta-accumulation in a single GEMM call — no separate
                # transpose or add kernel needed. Behaviour unchanged.
                blas.matmul(
                    ctx,
                    c_d_weight,
                    a_d_output,
                    b_input,
                    c_row_major=True,  # our TileTensors are row-major storage
                    transpose_a=True,
                    transpose_b=False,
                    beta=Float32(1.0) if accumulate else Float32(0.0),
                )
    else:
        ctx.synchronize()
        var scratch_t = TileTensor(
            Span[Scalar[dtype], MutAnyOrigin](
                ptr=scratch_ptr, length=out_channels * rows
            ),
            row_major(out_channels, rows),
        )
        var perms = alloc[Scalar[DType.int]](2)
        perms[0] = 1
        perms[1] = 0
        transpose(scratch_t, a_d_output, perms)
        perms.free()

        comptime if accumulate:
            # NOT an epilogue += : on the f32 Apple Accelerate path the
            # elementwise lambda runs as a sweep AFTER cblas has already
            # overwritten C, so "load previous" there reads the fresh GEMM
            # result and doubles it. Materialize, then add.
            var temp = alloc[Scalar[dtype]](out_channels * in_channels)
            var c_temp = TileTensor(
                Span[Scalar[dtype], MutAnyOrigin](
                    ptr=temp.as_unsafe_any_origin(),
                    length=out_channels * in_channels,
                ),
                row_major(out_channels, in_channels),
            )
            matmul[transpose_b=False, target=target](
                c_temp, scratch_t, b_input, ctx=ctx
            )
            comptime simd_width = simd_width_of[DType.float32]()
            _add_into[dtype, simd_width](
                d_weight_ptr,
                rebind[MutKernelPtr[dtype]](temp.as_unsafe_any_origin()),
                out_channels * in_channels,
            )
            temp.free()
        else:
            matmul[transpose_b=False, target=target](
                c_d_weight, scratch_t, b_input, ctx=ctx
            )
        ctx.synchronize()


def matmul_bwd[
    dtype: DType,
    target: StaticString,
    use_gelu: Bool = False,
    accumulate: Bool = True,
    has_bias: Bool = True,
](
    d_input_ptr: MutKernelPtr[dtype],
    d_weight_ptr: MutKernelPtr[dtype],
    d_bias_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    weight_ptr: ImmutKernelPtr[dtype],
    pre_gelu_ptr: ImmutKernelPtr[dtype],
    scratch_ptr: MutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    output_channels: Int64,
    ctx: DeviceContext,
) raises -> None:
    comptime if has_bias:
        matmul_bias_bwd[dtype, target, accumulate](
            d_bias_ptr,
            d_output_ptr,
            batch_size,
            seq_len,
            output_channels,
            ctx,
        )
    matmul_d_input_bwd[dtype, target, use_gelu](
        d_input_ptr,
        d_output_ptr,
        weight_ptr,
        pre_gelu_ptr,
        batch_size,
        seq_len,
        channels,
        output_channels,
        ctx,
    )
    matmul_d_weight_bwd[dtype, target, accumulate](
        d_weight_ptr,
        d_output_ptr,
        input_ptr,
        scratch_ptr,
        batch_size,
        seq_len,
        channels,
        output_channels,
        ctx,
    )


# ===----------------------------------------------------------------------=== #
# Matmul Backward Compiler Registration
# ===----------------------------------------------------------------------=== #


@always_inline
def _check_bwd_sizes[
    has_bias: Bool = True,
](
    d_input_size: Int,
    d_weight_size: Int,
    d_bias_size: Int,
    d_output_size: Int,
    x_size: Int,
    weight_size: Int,
    scratch_size: Int,
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    output_channels: Int64,
) raises -> None:
    var rows_x_channels = Int(batch_size * seq_len * channels)
    var rows_x_out = Int(batch_size * seq_len * output_channels)
    var weight_elems = Int(output_channels * channels)
    if d_input_size != rows_x_channels:
        raise Error(
            "d_input must have the same size as batch_size * seq_len * channels"
        )
    if d_weight_size != weight_elems:
        raise Error(
            "d_weight must have the same size as output_channels * channels"
        )
    comptime if has_bias:
        if d_bias_size != Int(output_channels):
            raise Error("d_bias must have the same size as output_channels")
    if d_output_size != rows_x_out:
        raise Error(
            "d_output must have the same size as batch_size * seq_len *"
            " output_channels"
        )
    if x_size != rows_x_channels:
        raise Error(
            "input must have the same size as batch_size * seq_len * channels"
        )
    if weight_size != weight_elems:
        raise Error(
            "weight must have the same size as output_channels * channels"
        )
    if scratch_size != rows_x_out:
        raise Error(
            "scratch must have the same size as batch_size * seq_len *"
            " output_channels"
        )


@compiler.register("matmul_bwd")
struct MatmulBwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        use_gelu: Bool = False,
        accumulate: Bool = True,
        has_bias: Bool = True,
    ](
        d_input: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        d_weight: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        d_bias: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        scratch: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        d_output: InputTensor[dtype=dtype, rank=2, static_spec=...],
        x: InputTensor[dtype=dtype, rank=2, static_spec=...],
        weight: InputTensor[dtype=dtype, rank=2, static_spec=...],
        pre_gelu: InputTensor[dtype=dtype, rank=2, static_spec=...],
        batch_size: Int64,
        seq_len: Int64,
        channels: Int64,
        output_channels: Int64,
        ctx: DeviceContext,
    ) capturing raises:
        _check_bwd_sizes[has_bias=has_bias](
            d_input.size(),
            d_weight.size(),
            d_bias.size(),
            d_output.size(),
            x.size(),
            weight.size(),
            scratch.size(),
            batch_size,
            seq_len,
            channels,
            output_channels,
        )
        # The use_gelu=False instantiation contains no loads from pre_gelu
        # (comptime-dead code), so a dummy buffer of any size is sound there.
        # pre_gelu here is the pre-activation of this matmul's INPUT (llm.c
        # composition: d_input = (d_output @ W) * gelu'(pre_gelu)), so it has
        # d_input's shape, not the forward's (B*T, OC).
        comptime if use_gelu:
            if pre_gelu.size() != Int(batch_size * seq_len * channels):
                raise Error(
                    "pre_gelu must have the same size as batch_size * seq_len"
                    " * channels"
                )
        matmul_bwd[
            dtype,
            target,
            use_gelu=use_gelu,
            accumulate=accumulate,
            has_bias=has_bias,
        ](
            d_input.unsafe_ptr(),
            d_weight.unsafe_ptr(),
            d_bias.unsafe_ptr(),
            d_output.unsafe_ptr(),
            x.unsafe_ptr(),
            weight.unsafe_ptr(),
            pre_gelu.unsafe_ptr(),
            scratch.unsafe_ptr(),
            batch_size,
            seq_len,
            channels,
            output_channels,
            ctx,
        )
