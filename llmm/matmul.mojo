import compiler
from std.memory import alloc, stack_allocation
from std.math import ceildiv
from layout import Layout, TileTensor
from layout.layout_tensor import LayoutTensor
from linalg.matmul import matmul
from std.sys import simd_width_of, get_defined_int, is_defined
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
from std.algorithm import vectorize
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu import barrier, block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import threadfence, Scope
from std.atomic import Atomic
from std.ffi import _get_global_or_null, external_call


# -D LLMM_FP8_GEMM_MUTEX=1: serialize fp8 cuBLASLt submission across rank
# threads (probe for a suspected library-level concurrency defect in the
# e4m3xe5m2 path; submission is host-side microseconds, execution stays
# concurrent).
comptime FP8_GEMM_MUTEX = is_defined["LLMM_FP8_GEMM_MUTEX"]()
# -D LLMM_FP8_BWD_MUTEX=1: serialize the ENTIRE fp8 backward host-side
# submission (quantize + amax + both GEMMs) across rank threads — probe for
# a first-use/init race in the backward kernel path.
comptime FP8_BWD_MUTEX = is_defined["LLMM_FP8_BWD_MUTEX"]()
# -D LLMM_FP8_BWD_EXEC_MUTEX=1: like LLMM_FP8_BWD_MUTEX, but the lock is
# held until a `ctx.synchronize()` at the end of `matmul_bwd_lowp`
# COMPLETES — i.e. every fp8 backward kernel this call enqueued has
# finished EXECUTING on the GPU before any other rank may enqueue its own.
# Round-3 probe: submission-only serialization still failed 3/10 at ws2,
# so the remaining hypothesis is execution-level cross-device concurrency
# (fp8 backward kernels physically co-executing on different GPUs, with
# peer access + UVA mappings live). DIAGNOSTIC ONLY — this serializes real
# GPU work across ranks (~WORLD_SIZE x the fp8 backward-bundle time, plus
# a full stream drain per site call); never ship it as a fix.
comptime FP8_BWD_EXEC_MUTEX = is_defined["LLMM_FP8_BWD_EXEC_MUTEX"]()
# -D LLMM_FP8_BWD_SYNC_ONLY=1: the `ctx.synchronize()` at the end of
# `matmul_bwd_lowp` WITHOUT any cross-rank lock. Bounds this rank's async
# launch run-ahead to one fp8-backward site bundle (~7 kernels) while leaving
# cross-rank submission AND cross-GPU execution fully concurrent. Round-3
# discriminator between "cross-GPU execution overlap corrupts" (predicts this
# still fails) and "per-rank deep-async-queue run-ahead race" (predicts this
# cures, like CUDA_LAUNCH_BLOCKING=1 10/10 and MODULAR_DEBUG=device-sync-mode
# 4/4 — both per-rank-only synchronizers that leave cross-GPU overlap intact).
# If clean, this doubles as the shipping MITIGATION for fp8 WORLD_SIZE>1
# (measured cost of the far heavier sync-mode was ~4%; this syncs only
# 144x/micro-step). It is a window-closure mitigation, not a root-cause fix.
comptime FP8_BWD_SYNC_ONLY = is_defined["LLMM_FP8_BWD_SYNC_ONLY"]()
# fp8 transposed-operand policy. DEFAULT (requantize): forward quantizes
# natural-layout only (quantize_devscale); matmul_d_input_bwd_lowp /
# matmul_d_weight_bwd_lowp re-quantize the transposed operand fresh at
# consumption time (quantize_transpose_devscale), immediately before the
# consuming GEMM — no long-lived forward-written WT/IT stash, closing the
# written-in-forward/consumed-micro-step-later exposure at the cost of two
# extra transpose-quantize kernels per fp8 site per backward. The two
# paths' results differ deterministically by ~0.3%: a benign
# allocation-layout execution variant of the backward GEMMs (byte-diff
# verified operand-identical; see docs/ai/fp8_multirank_nan_investigation.md).
# -D LLMM_FP8_STASH_LEGACY=1: the old path — forward's
# quantize_dual_devscale also writes the transposed fp8 stash into
# lowp_transpose_cache, and dgrad/wgrad consume it a forward-to-backward
# window later.
comptime FP8_STASH_LEGACY = is_defined["LLMM_FP8_STASH_LEGACY"]()


def _fp8_gemm_lock() -> UnsafePointer[Atomic[DType.int32], MutUntrackedOrigin]:
    """Process-global lock cell for the FP8_*_MUTEX probes. MUST be seeded
    from a single thread before any concurrency (`fp8_mutex_preseed`, called
    ahead of `sync_parallelize` in train_gpt2.mojo) — a concurrent first
    call races the registry insert and can split ranks across two cells.
    After InsertGlobal, the registry is re-read and its winner adopted so a
    raced loser at least converges instead of keeping a private cell."""
    var name = String("LLMM_FP8_GEMM_LOCK")
    if gp := _get_global_or_null(name):
        return gp.value().bitcast[Atomic[DType.int32]]()
    # Atomic[int32] is layout-compatible with a bare int32 cell; allocate
    # and zero the cell, then hand out Atomic-typed views of it.
    var p = alloc[Scalar[DType.int32]](1)
    p[0] = 0
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), p.bitcast[NoneType]()
    )
    var winner = _get_global_or_null(name)
    if winner:
        var wp = winner.value().bitcast[Scalar[DType.int32]]()
        if wp != p:
            p.free()
        return rebind[UnsafePointer[Atomic[DType.int32], MutUntrackedOrigin]](
            wp.bitcast[Atomic[DType.int32]]().as_unsafe_any_origin()
        )
    return rebind[UnsafePointer[Atomic[DType.int32], MutUntrackedOrigin]](
        p.bitcast[Atomic[DType.int32]]().as_unsafe_any_origin()
    )


def fp8_mutex_preseed():
    """Seed the FP8_*_MUTEX lock cell single-threaded, before rank threads
    exist. No-op beyond the first call."""
    comptime if FP8_GEMM_MUTEX or FP8_BWD_MUTEX or FP8_BWD_EXEC_MUTEX:
        _ = _fp8_gemm_lock()


from std.sys import size_of
from std.gpu.host._nvidia_cuda import CUDA
from _cublas.dtype import DataType
from _cublas.cublas import cublasOperation_t, ComputeType, check_cublas_error
from _cublas.cublaslt import (
    cublasLtHandle_t,
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
    cublasLtMatmulMatrixScale_t,
)
from linalg.matmul.vendor.blas import _get_global_handle, Backend

from llmm.gelu import gelu, gelu_grad, gelu_fwd_gpu, bias_gelu_fwd
from llmm.profiler import traced_parallelize
from llmm.memory import (
    ImmutKernelPtr,
    MutKernelPtr,
    persistent_device_buffer,
)
from llmm.vendor import HAS_CUBLAS, HAS_METAL, USE_TF32
from std.gpu.primitives.warp import shuffle_xor

from llmm.hadamard import hadamard_sign
from llmm.lowp import (
    FP8_SPEC,
    FP8_STATIC_SCALES,
    quantize_devscale,
    quantize_transpose_devscale,
    quantize_dual_devscale,
)

# -D LLMM_FP8_FAST_ACCUM=1 (default OFF) enables cuBLASLt fast-accumulation
# on the FORWARD fp8 GEMM only (matmul_fwd_lowp's lowp_gemm_devscale call);
# dgrad/wgrad always accumulate precisely (fast_accum=False). Independent of
# FP8_STATIC_SCALES — any on/off combination is valid.
comptime FP8_FAST_ACCUM = is_defined["LLMM_FP8_FAST_ACCUM"]()
from llmm.amax import (
    AmaxState,
    compute_amax,
    update_scale_pair,
    kernel_ptr_as_immut,
    device_buf_mut_ptr,
)
from llmm.nvfp4_quant import (
    nvfp4_quantize,
    nvfp4_quantize_transpose,
    nvfp4_packed_size,
    nvfp4_scale_buffer_size,
    ROUND_MODE_RNE,
    ROUND_MODE_STOCHASTIC,
    NVFP4_SR_SEED,
    NVFP4_SR_STREAM,
    NVFP4_SR_STREAM_DGRAD_DOUTPUT,
    NVFP4_SR_STREAM_WGRAD_DOUTPUT,
    NVFP4_BLOCK,
)


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
# Hand-edit A/B knob (not a -D flag); the not-fused branches below are
# unreachable unless this is flipped.


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
    # Persistent 32 MB cuBLASLt workspace (mirrors llm.c's single global
    # workspace).
    # Naming: _scratch = per-call transient; cache = persistent_device_buffer
    # (process-global); _buf = per-call local DeviceBuffer.
    var p = persistent_device_buffer[DType.uint8](
        ctx, "CUBLASLT_WS", _CUBLASLT_WS_BYTES
    )
    return p.bitcast[NoneType]().as_unsafe_any_origin()


@always_inline
def _lt_dt[dtype: DType]() -> DataType:
    comptime if dtype == DType.bfloat16:
        return DataType.R_16BF
    elif dtype == DType.float32:
        return DataType.R_32F
    elif dtype == DType.float8_e4m3fn:
        # e4m3 -> R_8F_E4M3: same mapping MAX's
        # blas._convert_to_cublas_datatype uses; native fp8 GEMM with bf16
        # output is the confirmed-working path on this box.
        return DataType.R_8F_E4M3
    elif dtype == DType.float8_e5m2:
        return DataType.R_8F_E5M2
    elif dtype == DType.float4_e2m1fn:
        # e2m1 -> R_4F_E2M1 (vendor-fixed enum 33). Packed 2 elements/byte;
        # cublasLtMatrixLayoutCreate rows/cols/ld are ELEMENT counts, not
        # bytes — see _matmul_cublaslt_fp4.
        return DataType.R_4F_E2M1
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


# CUBLASLT_MATMUL_DESC_FAST_ACCUM is an int8_t attribute (0=disabled,
# default), so it needs its own one-byte setter rather than reusing
# _lt_set_op (which sets the int32 cublasOperation_t).
@always_inline
def _lt_set_fast_accum(desc: cublasLtMatmulDesc_t, enable: Bool) raises:
    var flag = Int8(1) if enable else Int8(0)
    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_FAST_ACCUM,
            UnsafePointer(to=flag)
            .bitcast[NoneType]()
            .as_immutable()
            .as_unsafe_any_origin(),
            size_of[Int8](),
        )
    )


# ===------------------------------------------------------------------=== #
# Shared cuBLASLt sub-helpers.
#
# The three _matmul_cublaslt* orchestrators below (bf16/fp32, fp8, fp4) are
# deliberately NOT merged into one generic wrapper: a single body with many
# comptime switches makes this unsafe-pointer vendor code harder to audit
# than three explicit descriptor sequences whose per-precision attributes
# (epilogue/bias vs per-tensor scale pointers vs block-scale mode+pointers;
# caller-selectable vs TN-fixed transposes) stay visible at each use site.
# Only the four mechanical, byte-identical sub-steps are factored here:
# layout creation, preference setup, heuristic-select (+empty raise), destroy.
# ===------------------------------------------------------------------=== #


def _lt_make_layout(
    dt: DataType, rows: Int, cols: Int, ld: Int
) raises -> cublasLtMatrixLayout_t:
    """Create one col-major cuBLASLt matrix layout (rows x cols, leading
    dimension ld — element units, even for sub-byte dtypes like e2m1)."""
    var lay = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            UnsafePointer(to=lay).as_unsafe_any_origin(),
            dt,
            UInt64(rows),
            UInt64(cols),
            Int64(ld),
        )
    )
    return lay


def _lt_make_pref() raises -> cublasLtMatmulPreference_t:
    """Create the matmul preference, capped at the shared persistent
    workspace size (`_CUBLASLT_WS_BYTES`)."""
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
    return pref


def _lt_pick_algo(
    lt: cublasLtHandle_t,
    desc: cublasLtMatmulDesc_t,
    a_l: cublasLtMatrixLayout_t,
    b_l: cublasLtMatrixLayout_t,
    c_l: cublasLtMatrixLayout_t,
    d_l: cublasLtMatrixLayout_t,
    pref: cublasLtMatmulPreference_t,
    err_msg: String,
) raises -> cublasLtMatmulHeuristicResult_t:
    """Run the heuristic for one algorithm; raise `err_msg` if none found."""
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
        raise Error(err_msg)
    return heur


def _lt_destroy(
    desc: cublasLtMatmulDesc_t,
    a_l: cublasLtMatrixLayout_t,
    b_l: cublasLtMatrixLayout_t,
    c_l: cublasLtMatrixLayout_t,
    d_l: cublasLtMatrixLayout_t,
    pref: cublasLtMatmulPreference_t,
) raises:
    """Destroy the six per-call cuBLASLt handles (same order as the inline
    tails this replaces: pref, desc, then the four layouts)."""
    check_cublas_error(cublasLtMatmulPreferenceDestroy(pref))
    check_cublas_error(cublasLtMatmulDescDestroy(desc))
    check_cublas_error(cublasLtMatrixLayoutDestroy(a_l))
    check_cublas_error(cublasLtMatrixLayoutDestroy(b_l))
    check_cublas_error(cublasLtMatrixLayoutDestroy(c_l))
    check_cublas_error(cublasLtMatrixLayoutDestroy(d_l))


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
    var a_l: cublasLtMatrixLayout_t
    comptime if transA:
        a_l = _lt_make_layout(dt, k, m, k)
    else:
        a_l = _lt_make_layout(dt, m, k, m)
    var b_l: cublasLtMatrixLayout_t
    comptime if transB:
        b_l = _lt_make_layout(dt, n, k, n)
    else:
        b_l = _lt_make_layout(dt, k, n, k)
    var c_l = _lt_make_layout(dt, m, n, m)
    var d_l = _lt_make_layout(dt, m, n, m)

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

    var pref = _lt_make_pref()

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

    var heur = _lt_pick_algo(
        lt,
        desc,
        a_l,
        b_l,
        c_l,
        d_l,
        pref,
        "no cuBLASLt algorithm for the fused matmul",
    )

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
            d_ptr.bitcast[NoneType]().as_immutable().as_unsafe_any_origin(),
            c_l,
            d_ptr.bitcast[NoneType]().as_unsafe_any_origin(),
            d_l,
            UnsafePointer(to=heur.algo).as_immutable().as_unsafe_any_origin(),
            ws,
            _CUBLASLT_WS_BYTES,
            cuda_stream.value()[],
        )
    )

    _lt_destroy(desc, a_l, b_l, c_l, d_l, pref)


# ===------------------------------------------------------------------=== #
# FP8 GEMM — cuBLASLt native fp8 x fp8 -> bf16, TN-only.
# ("lowp" in identifiers throughout this file == this fp8 delayed-scaling
# family; the NVFP4 family uses "fp4"/"nvfp4".)
#
# Only fp8 operands with bf16 output work on this toolchain (fp8->fp8-output
# and MAX's generic linalg fp8 both fail to lower on sm_121); there is
# exactly one vendor body here and no comptime branch to select an
# alternative. Forward runs e4m3 x e4m3; dgrad/wgrad run e4m3 x e5m2 (the
# E5M2 d_output operand). TN-only: transA=OP_T, transB=OP_N; A/B
# column-major, ld=k.
#
# TN operand orientation (zero-copy whenever an operand's row-major storage
# already has the contraction dim trailing):
#   FORWARD (contraction=C): weight[OC,C] and input[rows,C] both TN-native —
#     no transpose (transpose_a=False, transpose_b=False).
#   DGRAD  (contraction=OC): d_output[rows,OC] native (B role); weight needs
#     a transposed fp8 copy (transpose_a=True, transpose_b=False).
#   WGRAD  (contraction=rows): BOTH need a transposed fp8 copy
#     (transpose_a=True, transpose_b=True); quantize_transpose_devscale
#     fuses the transpose into the quantize pass.
#
# beta=1 accumulate is supported for fp8-operand GEMMs — wgrad passes
# accumulate straight through to cuBLASLt beta, no bf16-scratch fallback.
# A_SCALE_POINTER/B_SCALE_POINTER (device fp32) are honored: lowp_gemm_devscale
# passes each operand's scale_inv so d_bf16 comes out already descaled, no
# separate dequantize pass.
# ===------------------------------------------------------------------=== #


def _matmul_cublaslt_fp8[
    a_dtype: DType,
    b_dtype: DType,
    out_dtype: DType,
    fast_accum: Bool = False,
](
    d_ptr: MutKernelPtr[out_dtype],
    a_ptr: ImmutKernelPtr[
        DType.uint8
    ],  # fp8 `a_dtype`-encoded bytes, [k,m] col-major (ld=k)
    b_ptr: ImmutKernelPtr[
        DType.uint8
    ],  # fp8 `b_dtype`-encoded bytes, [k,n] col-major (ld=k)
    a_scale_inv_ptr: ImmutKernelPtr[DType.float32],  # device scalar, 1/scale_a
    b_scale_inv_ptr: ImmutKernelPtr[DType.float32],  # device scalar, 1/scale_b
    m: Int,
    n: Int,
    k: Int,
    accumulate: Bool,
    ctx: DeviceContext,
) raises -> None:
    """Raw cuBLASLt fp8 GEMM: `D[m,n] = op(A)*op(B)` (fp32 compute, bf16 `D`),
    TN-only (`transA=CUBLAS_OP_T`, `transB=CUBLAS_OP_N` — cuBLASLt's fp8
    requirement; not caller-selectable, unlike `_matmul_cublaslt`'s bf16/fp32
    path). `a_ptr`/`b_ptr` must already be the correctly-oriented fp8 bytes —
    this function does no quantization or transpose; see `lowp_gemm_devscale`
    (which calls this) for that, and the module-level comment above for which
    operands of which GEMM need a transposed quantize. `k` must be a multiple
    of 16 (fp8 tensor-core alignment; unchecked here — caller's
    responsibility, same as `_matmul_cublaslt`'s implicit assumptions).

    fast_accum (default False): sets CUBLASLT_MATMUL_DESC_FAST_ACCUM
    (lower-precision, periodically-promoted accumulation). Callers enable it
    for the forward GEMM only; dgrad/wgrad keep it False.
    """
    var handle = _get_global_handle[a_dtype, Backend.CUBLASLT](ctx)
    var lt = handle._get_cublas()
    var cuda_stream = CUDA(ctx.stream())
    comptime dt_a = _lt_dt[a_dtype]()
    comptime dt_b = _lt_dt[b_dtype]()
    comptime dt_out = _lt_dt[out_dtype]()

    var desc = cublasLtMatmulDesc_t()
    check_cublas_error(
        cublasLtMatmulDescCreate(
            UnsafePointer(to=desc).as_unsafe_any_origin(),
            ComputeType.COMPUTE_32F,
            DataType.R_32F,
        )
    )
    _lt_set_op(
        desc, cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_TRANSA, True
    )
    _lt_set_op(
        desc, cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_TRANSB, False
    )
    comptime if fast_accum:
        _lt_set_fast_accum(desc, True)

    var a_sp = a_scale_inv_ptr
    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            UnsafePointer(to=a_sp)
            .bitcast[NoneType]()
            .as_immutable()
            .as_unsafe_any_origin(),
            size_of[ImmutKernelPtr[DType.float32]](),
        )
    )
    var b_sp = b_scale_inv_ptr
    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            UnsafePointer(to=b_sp)
            .bitcast[NoneType]()
            .as_immutable()
            .as_unsafe_any_origin(),
            size_of[ImmutKernelPtr[DType.float32]](),
        )
    )

    var a_l = _lt_make_layout(dt_a, k, m, k)
    var b_l = _lt_make_layout(dt_b, k, n, k)
    var c_l = _lt_make_layout(dt_out, m, n, m)
    var d_l = _lt_make_layout(dt_out, m, n, m)

    var pref = _lt_make_pref()

    var heur = _lt_pick_algo(
        lt,
        desc,
        a_l,
        b_l,
        c_l,
        d_l,
        pref,
        "no cuBLASLt algorithm for this fp8 GEMM on this toolchain/arch"
        " (a_dtype="
        + String(a_dtype)
        + " b_dtype="
        + String(b_dtype)
        + " out_dtype="
        + String(out_dtype)
        + ")",
    )

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
            d_ptr.bitcast[NoneType]().as_immutable().as_unsafe_any_origin(),
            c_l,
            d_ptr.bitcast[NoneType]().as_unsafe_any_origin(),
            d_l,
            UnsafePointer(to=heur.algo).as_immutable().as_unsafe_any_origin(),
            ws,
            _CUBLASLT_WS_BYTES,
            cuda_stream.value()[],
        )
    )

    _lt_destroy(desc, a_l, b_l, c_l, d_l, pref)


def lowp_gemm_devscale[
    a_out_dtype: DType,
    b_out_dtype: DType,
    in_dtype: DType,
    out_dtype: DType,
    target: StaticString,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    quantize_a: Bool = True,
    quantize_b: Bool = True,
    fast_accum: Bool = False,
](
    d_ptr: MutKernelPtr[out_dtype],
    a_ptr: ImmutKernelPtr[in_dtype],
    b_ptr: ImmutKernelPtr[in_dtype],
    a_fp8_scratch: MutKernelPtr[DType.uint8],
    b_fp8_scratch: MutKernelPtr[DType.uint8],
    a_scale_ptr: ImmutKernelPtr[DType.float32],
    a_scale_inv_ptr: ImmutKernelPtr[DType.float32],
    b_scale_ptr: ImmutKernelPtr[DType.float32],
    b_scale_inv_ptr: ImmutKernelPtr[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    accumulate: Bool,
    ctx: DeviceContext,
) raises -> None:
    """The dtype-generic fp8 GEMM entry point: quantizes `a_ptr`/`b_ptr`
    (bf16 or fp32) into fp8 scratch buffers at the caller-chosen scale, then
    runs `_matmul_cublaslt_fp8` (the only vendor body available on this
    toolchain — see the module comment above), producing bf16 `d_ptr`
    already correctly descaled (no separate dequantize pass needed).

    Both the quantize-time multiplier (`a_scale_ptr`/`b_scale_ptr`) AND the
    GEMM's descale reciprocal (`a_scale_inv_ptr`/`b_scale_inv_ptr`) are
    DEVICE fp32 scalars — the `AmaxState.scale`/`scale_inv` device buffers
    (llmm/amax.mojo), read with no host sync on the training step's critical
    path. See `quantize_devscale`'s docstring in llmm/lowp.mojo for why a
    host-scale parameter would force a host readback here. Keep the scale
    and scale-inv buffers in sync at the call site;
    `tests/test_lowp_gemm.mojo` shows the single-scalar-buffer pattern.

    `m`, `n`, `k` follow this file's existing column-major convention: `m` is
    D's trailing (fastest-varying, "column") dimension in row-major terms,
    `n` its leading dimension, `k` the contraction dimension — identical to
    `_matmul_cublaslt`'s `m`/`n`/`k` at each of its three call sites
    (`matmul_fwd`, `matmul_d_input_bwd`, `matmul_d_weight_bwd`).

    `transpose_a`/`transpose_b` select `quantize_devscale` (operand's
    natural row-major layout already fits the TN "A-role"/"B-role" free
    pattern) vs `quantize_transpose_devscale` (needs a physically transposed
    fp8 copy) per operand — see the module comment above for the derivation
    per GEMM:
      - forward:       transpose_a=False, transpose_b=False
      - dgrad (d_input): transpose_a=True,  transpose_b=False
      - wgrad (d_weight): transpose_a=True,  transpose_b=True

    When `transpose_a`: `a_ptr` is logically `[k, m]` row-major (source
    shape), `a_fp8_scratch` becomes `[m, k]` row-major (transposed) fp8
    bytes. When not: `a_ptr` is `[m, k]` row-major already, copied through
    `quantize_devscale` unchanged (mirrored for `b_ptr`/`[k, n]`/`[n, k]`).

    `a_fp8_scratch`/`b_fp8_scratch` must be at least `m*k`/`k*n` bytes.
    The scales are the quantize-time multipliers (`fp8_max / amax`, computed
    by `AmaxState`/`update_scale`); `a_scale_inv_ptr`/`b_scale_inv_ptr` hold
    their reciprocals (cuBLASLt's `A_SCALE_POINTER`/`B_SCALE_POINTER`
    require a device pointer, not a host value).

    quantize_a/quantize_b (default True): when False the operand is NOT
    read/quantized here — the caller must have already filled
    a_fp8_scratch/b_fp8_scratch (in the layout transpose_a/b implies), e.g.
    via quantize_dual_devscale, because the same tensor at the same scale is
    also needed in another orientation elsewhere this step.
    """
    comptime assert is_gpu[target](), (
        "lowp_gemm_devscale is GPU-only; low-precision kernels are never"
        " instantiated for the cpu target"
    )
    comptime assert HAS_CUBLAS, (
        "lowp_gemm_devscale's only implemented vendor body is cuBLASLt fp8"
        " (tests/probe_fp8/RESULTS.md probe3/probe4) -- no MAX-linalg or"
        " emulated fallback exists on this toolchain (see the module comment"
        " above _matmul_cublaslt_fp8)"
    )

    comptime if quantize_a:
        comptime if transpose_a:
            quantize_transpose_devscale[
                FP8_SPEC, a_out_dtype, in_dtype, target
            ](a_fp8_scratch, a_ptr, a_scale_ptr, k, m, ctx)
        else:
            quantize_devscale[FP8_SPEC, a_out_dtype, in_dtype, target](
                a_fp8_scratch, a_ptr, a_scale_ptr, m * k, ctx
            )

    comptime if quantize_b:
        comptime if transpose_b:
            quantize_transpose_devscale[
                FP8_SPEC, b_out_dtype, in_dtype, target
            ](b_fp8_scratch, b_ptr, b_scale_ptr, k, n, ctx)
        else:
            quantize_devscale[FP8_SPEC, b_out_dtype, in_dtype, target](
                b_fp8_scratch, b_ptr, b_scale_ptr, n * k, ctx
            )

    comptime if FP8_GEMM_MUTEX:
        var lk = _fp8_gemm_lock()
        while True:
            var expected = Scalar[DType.int32](0)
            if lk[].compare_exchange(expected, 1):
                break
            _ = external_call["sched_yield", Int32]()

    _matmul_cublaslt_fp8[
        a_out_dtype, b_out_dtype, out_dtype, fast_accum=fast_accum
    ](
        d_ptr,
        a_fp8_scratch.as_immutable(),
        b_fp8_scratch.as_immutable(),
        a_scale_inv_ptr,
        b_scale_inv_ptr,
        m,
        n,
        k,
        accumulate,
        ctx,
    )

    comptime if FP8_GEMM_MUTEX:
        _fp8_gemm_lock()[].store(0)


# ===------------------------------------------------------------------=== #
# FP4 (NVFP4) GEMM — cuBLASLt native e2m1 x e2m1 -> bf16, block-scaled,
# TN-only (transA=OP_T, transB=OP_N; op(A) reads A's [K,M] col-major buffer,
# byte-identical to [M,K] row-major with K trailing — the same free TN
# duality as fp8, preserved by NVFP4 K-major packing).
# Block-scale mode CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 on both operands;
# A/B_SCALE_POINTER = swizzled e4m3 block-scale buffers; bf16 D, fp32 compute.
# k must be a multiple of 16 (NVFP4 block size).
# ===------------------------------------------------------------------=== #

comptime _NVFP4_LT_SCALE_MODE = (
    cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC16_UE4M3
)


def _matmul_cublaslt_fp4[
    out_dtype: DType,
](
    d_ptr: MutKernelPtr[out_dtype],
    # packed e2m1, [k,m] col-major, ld=k elements (== k/2 bytes).
    a_ptr: ImmutKernelPtr[DType.uint8],
    # packed e2m1, [k,n] col-major, ld=k elements (== k/2 bytes).
    b_ptr: ImmutKernelPtr[DType.uint8],
    a_scale_ptr: ImmutKernelPtr[DType.uint8],  # swizzled e4m3 block scales
    b_scale_ptr: ImmutKernelPtr[DType.uint8],  # swizzled e4m3 block scales
    m: Int,
    n: Int,
    k: Int,
    accumulate: Bool,
    ctx: DeviceContext,
) raises -> None:
    """Raw cuBLASLt NVFP4 GEMM: `D[m,n] = op(A)*op(B)` (fp32 compute, bf16
    `D`), TN-only (`transA=CUBLAS_OP_T`, `transB=CUBLAS_OP_N`, matching
    `_matmul_cublaslt_fp4`'s probe origin and `_matmul_cublaslt_fp8`'s
    convention). `a_ptr`/`b_ptr` are already packed e2m1 bytes (2
    elements/byte) and `a_scale_ptr`/`b_scale_ptr` are already
    cuBLAS-swizzled e4m3 block-scale buffers (`llmm/nvfp4_quant.mojo`'s
    `nvfp4_quantize` output layout) — this function does no quantization;
    see `lowp_gemm_fp4` for that. `k` must be a multiple of 16 (NVFP4 block
    size; unchecked here, `nvfp4_quantize` enforces it upstream).

    `m`/`n`/`k` are *element* counts of the logical (unpacked) e2m1 tensors —
    `cublasLtMatrixLayoutCreate`'s rows/cols/ld for `CUDA_R_4F_E2M1` are in
    element units even though the physical buffer is 2 elements/byte (matches
    tests/probe_fp4/probe_fp4.cu's `cublasLtMatrixLayoutCreate(&Adesc,
    CUDA_R_4F_E2M1, K, M, K)`).
    """
    var handle = _get_global_handle[DType.uint8, Backend.CUBLASLT](ctx)
    var lt = handle._get_cublas()
    var cuda_stream = CUDA(ctx.stream())
    comptime dt_ab = _lt_dt[DType.float4_e2m1fn]()
    comptime dt_out = _lt_dt[out_dtype]()

    var desc = cublasLtMatmulDesc_t()
    check_cublas_error(
        cublasLtMatmulDescCreate(
            UnsafePointer(to=desc).as_unsafe_any_origin(),
            ComputeType.COMPUTE_32F,
            DataType.R_32F,
        )
    )
    _lt_set_op(
        desc, cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_TRANSA, True
    )
    _lt_set_op(
        desc, cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_TRANSB, False
    )

    var scale_mode = _NVFP4_LT_SCALE_MODE
    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
            UnsafePointer(to=scale_mode)
            .bitcast[NoneType]()
            .as_immutable()
            .as_unsafe_any_origin(),
            size_of[cublasLtMatmulMatrixScale_t](),
        )
    )
    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
            UnsafePointer(to=scale_mode)
            .bitcast[NoneType]()
            .as_immutable()
            .as_unsafe_any_origin(),
            size_of[cublasLtMatmulMatrixScale_t](),
        )
    )

    var a_sp = a_scale_ptr
    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            UnsafePointer(to=a_sp)
            .bitcast[NoneType]()
            .as_immutable()
            .as_unsafe_any_origin(),
            size_of[ImmutKernelPtr[DType.uint8]](),
        )
    )
    var b_sp = b_scale_ptr
    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            UnsafePointer(to=b_sp)
            .bitcast[NoneType]()
            .as_immutable()
            .as_unsafe_any_origin(),
            size_of[ImmutKernelPtr[DType.uint8]](),
        )
    )

    var a_l = _lt_make_layout(dt_ab, k, m, k)
    var b_l = _lt_make_layout(dt_ab, k, n, k)
    var c_l = _lt_make_layout(dt_out, m, n, m)
    var d_l = _lt_make_layout(dt_out, m, n, m)

    var pref = _lt_make_pref()

    var heur = _lt_pick_algo(
        lt,
        desc,
        a_l,
        b_l,
        c_l,
        d_l,
        pref,
        "no cuBLASLt algorithm for the NVFP4 TN GEMM (m="
        + String(m)
        + " n="
        + String(n)
        + " k="
        + String(k)
        + ") -- tests/probe_fp4/RESULTS.md's 512^3 probe found 4 viable"
        " algorithms on this box, so a shape returning zero here is"
        " likely too small/misaligned for the vs16 block-scaled kernel"
        " (k must be a multiple of 16)",
    )

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
            d_ptr.bitcast[NoneType]().as_immutable().as_unsafe_any_origin(),
            c_l,
            d_ptr.bitcast[NoneType]().as_unsafe_any_origin(),
            d_l,
            UnsafePointer(to=heur.algo).as_immutable().as_unsafe_any_origin(),
            ws,
            _CUBLASLT_WS_BYTES,
            cuda_stream.value()[],
        )
    )

    _lt_destroy(desc, a_l, b_l, c_l, d_l, pref)


# ===------------------------------------------------------------------=== #
# Per-tensor-scale correction.
#
# NVFP4 is two-level: value ~= e2m1 * decode_e4m3(block_code) * tensor_scale.
# cuBLASLt's VEC16_UE4M3 mode applies only decode_e4m3(block_code) per block;
# it has no per-tensor multiplier. So _matmul_cublaslt_fp4's raw output is
# D_true / (tensor_scale_A * tensor_scale_B). _nvfp4_post_scale_gpu multiplies
# it back in one elementwise pass (both tensor_scales stay device-resident).
#
# accumulate: a cuBLASLt beta=1 accumulate is incorrect here (it would also
# rescale the pre-existing accumulated value by this call's tensor scales).
# So _matmul_cublaslt_fp4 always runs beta=0 into a fresh raw buffer, and
# accumulation happens in the post-scale step, in fp32, on the fresh
# contribution only. The post-scale kernel takes a separate raw_ptr (GEMM
# output) and d_ptr (accumulator); callers may alias them when
# accumulate=False.
# ===------------------------------------------------------------------=== #


def _nvfp4_post_scale_gpu[
    out_dtype: DType,
    accumulate: Bool,
](
    d_ptr: MutKernelPtr[out_dtype],
    raw_ptr: ImmutKernelPtr[out_dtype],
    a_tensor_scale_ptr: ImmutKernelPtr[DType.float32],
    b_tensor_scale_ptr: ImmutKernelPtr[DType.float32],
    n: Int,
    extra_scale: Float32,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < n:
        var s = a_tensor_scale_ptr[0] * b_tensor_scale_ptr[0] * extra_scale
        var v = raw_ptr[idx].cast[DType.float32]() * s
        comptime if accumulate:
            v = v + d_ptr[idx].cast[DType.float32]()
        d_ptr[idx] = v.cast[out_dtype]()


def _nvfp4_post_scale[
    out_dtype: DType,
    target: StaticString,
    accumulate: Bool = False,
](
    d_ptr: MutKernelPtr[out_dtype],
    raw_ptr: ImmutKernelPtr[out_dtype],
    a_tensor_scale_ptr: ImmutKernelPtr[DType.float32],
    b_tensor_scale_ptr: ImmutKernelPtr[DType.float32],
    n: Int,
    ctx: DeviceContext,
    extra_scale: Float32 = Float32(1.0),
) raises -> None:
    comptime assert is_gpu[target](), "_nvfp4_post_scale is GPU-only"
    comptime BLOCK_SIZE = 256
    var num_blocks = ceildiv(n, BLOCK_SIZE)
    comptime kernel = _nvfp4_post_scale_gpu[out_dtype, accumulate]
    var compiled = ctx.compile_function[kernel]()
    ctx.enqueue_function(
        compiled,
        d_ptr,
        raw_ptr,
        a_tensor_scale_ptr,
        b_tensor_scale_ptr,
        n,
        extra_scale,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )


def lowp_gemm_fp4[
    in_dtype: DType,
    out_dtype: DType,
    target: StaticString,
    a_block_rows: Int = 1,
    b_block_rows: Int = 1,
    a_round_mode: Int = ROUND_MODE_RNE,
    b_round_mode: Int = ROUND_MODE_RNE,
    accumulate: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
](
    d_ptr: MutKernelPtr[out_dtype],
    d_raw_scratch: MutKernelPtr[
        out_dtype
    ],  # >= m*n; raw (pre-tensor-scale) GEMM output scratch. MAY alias
    # `d_ptr` when `accumulate=False` (the post-scale kernel reads then
    # writes each index in place, safe to alias); MUST be a buffer distinct
    # from `d_ptr` when `accumulate=True` (post-scale reads `d_ptr`'s
    # PRE-EXISTING value to add to, so it cannot also be where the fresh raw
    # GEMM output landed -- see the module comment above
    # `_nvfp4_post_scale_gpu`).
    a_ptr: ImmutKernelPtr[in_dtype],  # [m, k] row-major, bf16 or fp32
    b_ptr: ImmutKernelPtr[in_dtype],  # [n, k] row-major, bf16 or fp32
    a_q_scratch: MutKernelPtr[DType.uint8],  # >= nvfp4_packed_size(m, k)
    a_scale_scratch: MutKernelPtr[
        DType.uint8
    ],  # >= nvfp4_scale_buffer_size(m, k, a_block_rows)
    a_tensor_scale_scratch: MutKernelPtr[DType.float32],  # 1 element
    b_q_scratch: MutKernelPtr[DType.uint8],  # >= nvfp4_packed_size(n, k)
    b_scale_scratch: MutKernelPtr[
        DType.uint8
    ],  # >= nvfp4_scale_buffer_size(n, k, b_block_rows)
    b_tensor_scale_scratch: MutKernelPtr[DType.float32],  # 1 element
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
    sr_seed: UInt64 = NVFP4_SR_SEED,
    a_sr_stream: UInt64 = NVFP4_SR_STREAM,
    b_sr_stream: UInt64 = NVFP4_SR_STREAM + 1,
    sr_step: Int = 0,
    extra_scale: Float32 = Float32(1.0),
) raises -> None:
    """The dtype-generic NVFP4 GEMM entry point — the FP4 analogue of
    `lowp_gemm_devscale` (fp8, above): quantizes `a_ptr`/`b_ptr` (bf16 or
    fp32, both forward-orientation `[rows, k]` row-major — `a_ptr` is the `[m,k]`
    "input"-role operand, `b_ptr` the `[n,k]` "weight"-role operand, e.g.
    weight `[OC,C]`) into NVFP4 (packed e2m1 + swizzled e4m3 block scales +
    fresh fp32 tensor scale) via `llmm/nvfp4_quant.mojo`'s `nvfp4_quantize`,
    then runs `_matmul_cublaslt_fp4`, producing bf16 `d_ptr`.

    `a_block_rows`/`b_block_rows` select `nvfp4_quantize`'s `BLOCK_ROWS`
    granularity per operand independently (1 -> 1D 1x16, the recipe's
    default for activations/gradients; 16 -> 2D 16x16, for weights) — pass
    `b_block_rows=16` when `b_ptr` is a weight operand.

    a_round_mode/b_round_mode (default RNE) select RNE vs stochastic per
    operand independently; set STOCHASTIC on the gradient operand only.
    sr_seed/sr_step are forwarded to both quantize calls (consulted only
    under stochastic rounding); A and B use separate RNG substreams so their
    dither never collides.

    `accumulate` (comptime, default False): when True, `d_ptr[idx] +=
    raw_gemm[idx] * tensor_scale_A * tensor_scale_B` instead of overwriting
    — see the module comment above `_nvfp4_post_scale_gpu` for why this
    needs the separate `d_raw_scratch` buffer rather than a cuBLASLt-level
    beta=1 accumulate (which NVFP4's two-level scale makes incorrect).

    `transpose_a`/`transpose_b` (comptime): when set, that operand
    is quantized via `nvfp4_quantize_transpose` instead of `nvfp4_quantize`
    — needed when the operand's natural `[rows, cols]` row-major storage
    does NOT already have the contraction dimension `k` trailing (cuBLASLt's
    NVFP4 GEMM is TN-only, mirroring fp8's
    `lowp_gemm_devscale`/`_matmul_cublaslt_fp8`
    — see `llmm/matmul.mojo`'s dgrad/wgrad module comment below
    `matmul_bwd_lowp` for the fp8 derivation and the fp4 sibling comment
    above `matmul_d_input_bwd_fp4` for the fp4-specific wrinkle). When
    `transpose_a`: `a_ptr` is logically `[k, m]` row-major (source shape,
    e.g. weight `[OC,C]` for Dgrad, `k=OC, m=C`); `a_q_scratch`/
    `a_scale_scratch` still size for the LOGICAL `[m,k]` a-role shape
    (unchanged from the non-transposed case). Mirrored for `transpose_b`/
    `b_ptr`/`[k,n]`.
    `m`/`n`/`k` follow this file's existing column-major-`D` convention,
    identical to `lowp_gemm_devscale`'s.

    extra_scale (default 1.0): folded into the post-scale multiply. A caller
    that pre-applied the 16-wide RHT to both operands ((H@a)^T@(H@b) ==
    16 a^T b) passes extra_scale=1/16 to recover the untransformed result
    for free.
    """
    comptime assert is_gpu[target](), (
        "lowp_gemm_fp4 is GPU-only; low-precision kernels are never"
        " instantiated for the cpu target"
    )
    comptime assert HAS_CUBLAS, (
        "lowp_gemm_fp4's only implemented vendor body is cuBLASLt NVFP4"
        " (tests/probe_fp4/RESULTS.md)"
    )

    comptime if transpose_a:
        nvfp4_quantize_transpose[in_dtype, target, a_block_rows, a_round_mode](
            a_q_scratch,
            a_scale_scratch,
            a_tensor_scale_scratch,
            a_ptr,
            k,
            m,
            ctx,
            sr_seed,
            a_sr_stream,
            sr_step,
        )
    else:
        nvfp4_quantize[in_dtype, target, a_block_rows, a_round_mode](
            a_q_scratch,
            a_scale_scratch,
            a_tensor_scale_scratch,
            a_ptr,
            m,
            k,
            ctx,
            sr_seed,
            a_sr_stream,
            sr_step,
        )
    comptime if transpose_b:
        nvfp4_quantize_transpose[in_dtype, target, b_block_rows, b_round_mode](
            b_q_scratch,
            b_scale_scratch,
            b_tensor_scale_scratch,
            b_ptr,
            k,
            n,
            ctx,
            sr_seed,
            b_sr_stream,
            sr_step,
        )
    else:
        nvfp4_quantize[in_dtype, target, b_block_rows, b_round_mode](
            b_q_scratch,
            b_scale_scratch,
            b_tensor_scale_scratch,
            b_ptr,
            n,
            k,
            ctx,
            sr_seed,
            b_sr_stream,
            sr_step,
        )

    # Always a fresh (beta=0) raw GEMM result into `d_raw_scratch` -- NVFP4's
    # two-level scale means cuBLASLt-level accumulation is never correct
    # here (module comment above `_nvfp4_post_scale_gpu`); any accumulation
    # happens in the post-scale step below instead.
    _matmul_cublaslt_fp4[out_dtype](
        d_raw_scratch,
        a_q_scratch.as_immutable(),
        b_q_scratch.as_immutable(),
        a_scale_scratch.as_immutable(),
        b_scale_scratch.as_immutable(),
        m,
        n,
        k,
        False,
        ctx,
    )

    _nvfp4_post_scale[out_dtype, target, accumulate=accumulate](
        d_ptr,
        d_raw_scratch.as_immutable(),
        a_tensor_scale_scratch.as_immutable(),
        b_tensor_scale_scratch.as_immutable(),
        m * n,
        ctx,
        extra_scale,
    )


def bf16_control_gemm[
    target: StaticString,
](
    d_ptr: MutKernelPtr[DType.bfloat16],
    a_ptr: ImmutKernelPtr[DType.bfloat16],
    b_ptr: ImmutKernelPtr[DType.bfloat16],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> None:
    """Plain bf16 TN GEMM through the SAME _matmul_cublaslt vendor call and
    col-major-D layout convention _matmul_cublaslt_fp4 uses. Exists so tests
    can run a same-code-path bf16 control arm (not just a host fp32
    reference), which isolates GEMM/readback harness issues from kernel
    numerics.
    """
    comptime assert is_gpu[target](), "bf16_control_gemm is GPU-only"
    _matmul_cublaslt[DType.bfloat16, True, False](
        d_ptr,
        a_ptr,
        b_ptr,
        a_ptr,  # bias_ptr placeholder — unused (epilogue has no BIAS bit)
        d_ptr,  # aux_ptr placeholder — unused (epilogue has no AUX bit)
        m,
        n,
        k,
        Int32(1),  # epilogue = DEFAULT
        False,
        ctx,
    )


def _launch_gelu_fwd_gpu[
    dtype: DType,
](
    out_ptr: MutKernelPtr[dtype],
    x_ptr: ImmutKernelPtr[dtype],
    num_params: Int,
    ctx: DeviceContext,
) raises -> None:
    # Hand-edit A/B knob (not a -D flag); this helper is currently dead code
    # kept for that A/B, unreachable unless USE_GELU_FUSION is flipped.
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


# matmul_fwd_lowp — fp8 forward linear. A separate entry point, not a branch
# inside matmul_fwd: matmul_fwd's signature has no room for the per-site
# AmaxStates a delayed-scaling fp8 GEMM needs, and every bf16/fp32 caller
# (and the LM-head, excluded from fp8) must stay unchanged. GEMM runs
# E4M3 x E4M3 -> bf16, then bias/GELU as the existing separate bf16 kernel
# (cuBLASLt fp8 has restricted epilogue support).


# Transposed-operand fp8 cache shared across the forward/backward
# boundary. Default: forward quantizes natural-layout only, and dgrad/
# wgrad fill this cache themselves at consumption time
# (quantize_transpose_devscale). -D LLMM_FP8_STASH_LEGACY=1: the
# natural-layout and transposed-layout quantizes of the SAME bf16 tensor
# at the SAME (not-updated-between-fwd-and-bwd) scale are fused into one
# quantize_dual_devscale call in forward, and the transposed copy is
# stashed here for backward to read read-only.
#
# The cache MUST be a persistent_device_buffer (process-global heap cell),
# NOT a local DeviceBuffer: a local's destroy-at-last-use can drop the buffer
# before a nested consumer several call-levels deep reads it (a real,
# reproduced race — docs/ai/low_precision_gotchas.md G2). A persistent
# buffer has no scope-tracked owner, so there is no "last use" to destroy at.
#
# forward-writes-then-backward-reads (legacy path) is safe for ANY
# grad_accum_steps: the training loop runs forward then backward once per
# micro-step, sequentially, on one stream — a per-(site,layer) buffer
# written by this micro-step's forward is consumed by this same
# micro-step's backward before the next forward can overwrite it.
# site/layer only build the cache's name.


@always_inline
def lowp_transpose_cache(
    ctx: DeviceContext,
    tag: StaticString,
    site: StaticString,
    layer: Int,
    count: Int,
) raises -> MutKernelPtr[DType.uint8]:
    """Persistent per-(tag,site,layer) fp8 scratch buffer — `tag`
    distinguishes the weight-transposed ("WT") vs. input-transposed ("IT")
    cache; `site`/`layer` distinguish the up-to-48 (4 sites x num_layer)
    independent GEMM operand identities sharing this one process. First call
    for a given name allocates `count` bytes; every later call with the same
    name just returns the cached pointer (see
    `llmm.memory.persistent_device_buffer`'s docstring) — safe here because
    every `(tag, site, layer)` triple's `count` is invariant across the
    whole run (fixed `batch_size`/`seq_len`/`channels` for the training
    config).
    """
    var name = (
        String("FP8_")
        + String(tag)
        + String("_")
        + String(site)
        + String("_")
        + String(layer)
    )
    return persistent_device_buffer[DType.uint8](ctx, name, count)


def matmul_fwd_lowp[
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
    mut input_state: AmaxState[FP8_SPEC],
    mut weight_state: AmaxState[FP8_SPEC],
    site: StaticString,
    layer: Int,
    ctx: DeviceContext,
) raises -> None:
    """Forward linear (fp8): `out = gelu?(bias? + input @ weightᵀ)`, the same
    geometry as `matmul_fwd`, but the GEMM itself runs E4M3 x E4M3 -> bf16
    via `lowp_gemm_devscale` (forward orientation: `transpose_a=False,
    transpose_b=False` — both `weight` `[out_channels,channels]` and `input`
    `[rows,channels]` are already TN-native, no transposed quantize needed —
    see `lowp_gemm_devscale`'s docstring for the per-GEMM orientation
    derivation).

    `input_state`/`weight_state` are this call site's `AmaxState[FP8_SPEC]`
    (one instance per transformer layer per site, from `train_gpt2.mojo`'s
    `Fp8State` container) — updated in place (`update_scale`) every call, so
    repeated calls (one per training step) build the delayed-scaling
    history. The weight is re-quantized from its bf16 storage on every call
    (no persistent fp8 weight cache) because the optimizer updates it every
    step.

    By default weight AND input are each quantized natural-layout only
    (`quantize_devscale`) — the backward re-quantizes the transposed
    copies fresh at consumption time. Under -D LLMM_FP8_STASH_LEGACY=1
    they are instead quantized via `quantize_dual_devscale` — ONE read of
    the bf16 source producing BOTH the natural-layout fp8 copy (consumed
    immediately below, by THIS function's own GEMM) and the
    transposed-layout copy (cached in a persistent `(site, layer)`-keyed
    buffer via `lowp_transpose_cache`, consumed later this same micro-step
    by `matmul_d_input_bwd_lowp`'s dgrad (weight-transposed) /
    `matmul_d_weight_bwd_lowp`'s wgrad (input-transposed) — see the module
    comment above this function for the full lifetime/ownership design).
    `site`/`layer` identify this call site for that cache; they carry no
    other meaning (not used for dispatch, only for building the persistent
    buffer's name).

    GPU-only (comptime-asserted): fp8 is never instantiated for the `cpu`
    target.
    """
    comptime assert is_gpu[target](), (
        "matmul_fwd_lowp is GPU-only; low-precision kernels must never be"
        " instantiated for the cpu target"
    )
    var rows = Int(batch_size * seq_len)
    var in_channels = Int(channels)
    var out_channels = Int(output_channels)

    # 1. Per-operand amax -> delayed-scaling update (call update_scale once
    #    per step BEFORE reading state.scale below). Under -D
    #    LLMM_FP8_STATIC_SCALES=1 this whole block is comptime-skipped
    #    (never instantiated); scales were seeded once at Fp8State.__init__
    #    and never change.
    comptime if not FP8_STATIC_SCALES:
        var amax_input = ctx.enqueue_create_buffer[DType.float32](1)
        var amax_weight = ctx.enqueue_create_buffer[DType.float32](1)
        compute_amax[FP8_SPEC, dtype](
            device_buf_mut_ptr(amax_input), input_ptr, rows * in_channels, ctx
        )
        compute_amax[FP8_SPEC, dtype](
            device_buf_mut_ptr(amax_weight),
            weight_ptr,
            out_channels * in_channels,
            ctx,
        )
        # Both amaxes are ready and no call site reads one state's scale
        # before the other's, so the two update_scale calls fuse into one
        # kernel launch (update_scale_pair).
        update_scale_pair[FP8_SPEC, FP8_SPEC.fwd_dtype](
            input_state,
            kernel_ptr_as_immut(device_buf_mut_ptr(amax_input)),
            weight_state,
            kernel_ptr_as_immut(device_buf_mut_ptr(amax_weight)),
            ctx,
        )

    # 2. Quantize (dual-output — see the module comment above) + fp8 GEMM
    #    -> raw (bias-free) bf16 output in out_ptr.
    #    `a`=weight (m=out_channels,k=in_channels), `b`=input
    #    (n=rows,k=in_channels) — matches matmul_fwd's own weight-as-A/
    #    input-as-B convention (`_matmul_cublaslt[transA=True]` called with
    #    `weight_ptr` first, `input_ptr` second).
    var a_scratch = ctx.enqueue_create_buffer[DType.uint8](
        out_channels * in_channels
    )
    var b_scratch = ctx.enqueue_create_buffer[DType.uint8](rows * in_channels)
    comptime if not FP8_STASH_LEGACY:
        # Default (see the flag comment near the top of this file):
        # natural-layout quantize ONLY — no transposed stash is written in
        # forward at all; backward re-quantizes the transposed operands
        # fresh at consumption time.
        quantize_devscale[FP8_SPEC, FP8_SPEC.fwd_dtype, dtype, target](
            device_buf_mut_ptr(a_scratch),
            weight_ptr,
            kernel_ptr_as_immut(device_buf_mut_ptr(weight_state.scale)),
            out_channels * in_channels,
            ctx,
        )
        quantize_devscale[FP8_SPEC, FP8_SPEC.fwd_dtype, dtype, target](
            device_buf_mut_ptr(b_scratch),
            input_ptr,
            kernel_ptr_as_immut(device_buf_mut_ptr(input_state.scale)),
            rows * in_channels,
            ctx,
        )
    else:
        # -D LLMM_FP8_STASH_LEGACY=1: dual-output quantize — the transposed
        # fp8 copy is stashed here for dgrad/wgrad to consume a
        # forward-to-backward window later.
        var weight_t = lowp_transpose_cache(
            ctx, "WT", site, layer, in_channels * out_channels
        )
        var input_t = lowp_transpose_cache(
            ctx, "IT", site, layer, in_channels * rows
        )

        quantize_dual_devscale[FP8_SPEC, FP8_SPEC.fwd_dtype, dtype, target](
            device_buf_mut_ptr(a_scratch),
            weight_t,
            weight_ptr,
            kernel_ptr_as_immut(device_buf_mut_ptr(weight_state.scale)),
            out_channels,
            in_channels,
            ctx,
        )
        quantize_dual_devscale[FP8_SPEC, FP8_SPEC.fwd_dtype, dtype, target](
            device_buf_mut_ptr(b_scratch),
            input_t,
            input_ptr,
            kernel_ptr_as_immut(device_buf_mut_ptr(input_state.scale)),
            rows,
            in_channels,
            ctx,
        )

    lowp_gemm_devscale[
        FP8_SPEC.fwd_dtype,
        FP8_SPEC.fwd_dtype,
        dtype,
        dtype,
        target,
        transpose_a=False,
        transpose_b=False,
        quantize_a=False,
        quantize_b=False,
        fast_accum=FP8_FAST_ACCUM,
    ](
        out_ptr,
        weight_ptr,
        input_ptr,
        device_buf_mut_ptr(a_scratch),
        device_buf_mut_ptr(b_scratch),
        kernel_ptr_as_immut(device_buf_mut_ptr(weight_state.scale)),
        kernel_ptr_as_immut(device_buf_mut_ptr(weight_state.scale_inv)),
        kernel_ptr_as_immut(device_buf_mut_ptr(input_state.scale)),
        kernel_ptr_as_immut(device_buf_mut_ptr(input_state.scale_inv)),
        out_channels,
        rows,
        in_channels,
        False,
        ctx,
    )

    # 3. Bias (+GELU) epilogue in bf16, same standalone kernel matmul_fwd's
    #    vendor-neutral GPU branch uses (cuBLASLt fp8 has restricted
    #    epilogue support — design §5 point 1).
    comptime if has_bias or use_gelu:
        bias_gelu_fwd[dtype, target, has_bias=has_bias, use_gelu=use_gelu](
            out_ptr, pre_gelu_ptr, bias_ptr, rows, out_channels, ctx
        )


def matmul_fwd_fp4[
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
    """NVFP4 forward linear: `out = gelu?(bias? + input @ weightᵀ)`, the
    fp4 sibling of `matmul_fwd_lowp` (fp8, above) — same geometry as
    `matmul_fwd`, but the GEMM runs NVFP4 x NVFP4 -> bf16 via
    `lowp_gemm_fp4` (forward orientation, matching `matmul_fwd_lowp`'s
    weight-as-A/input-as-B convention: `a`=`weight_ptr`
    `[out_channels,channels]` -> `m`=`out_channels`, `b`=`input_ptr`
    `[rows,channels]` -> `n`=`rows`, so the column-major `D` cuBLASLt
    writes lands as the `[rows,out_channels]` row-major buffer `out_ptr`
    expects — see `lowp_gemm_devscale`'s docstring for the derivation).

    The weight operand uses 2D 16x16 block scaling (`a_block_rows=
    NVFP4_BLOCK`, so the same quantized weight buffer can serve both Fprop
    row-major and a Dgrad column-major read without requantizing), the
    activation operand uses 1D 1x16 (`b_block_rows=1`). Fprop is always RNE
    (`round_mode` left at its `ROUND_MODE_RNE` default) — stochastic
    rounding is reserved for gradient operands.

    Unlike `matmul_fwd_lowp`, there is no `AmaxState`/delayed-scaling
    argument: `nvfp4_quantize` (inside `lowp_gemm_fp4`) computes both
    scale levels (fp32 per-tensor + e4m3 per-block) fresh, fully
    device-resident, every call — see `llmm/lowp.mojo`'s `FP4_SPEC`
    comment ("decision: FP4 does not use `AmaxState` at all"). Scratch
    buffers (packed e2m1 + swizzled scale buffers + 1-elem tensor scales)
    are freshly enqueued each call, mirroring `matmul_fwd_lowp`'s
    `ctx.enqueue_create_buffer` scratch pattern above.

    GPU-only (comptime-asserted): fp4 is never instantiated for the `cpu`
    target.
    """
    comptime assert is_gpu[target](), (
        "matmul_fwd_fp4 is GPU-only; low-precision kernels must never be"
        " instantiated for the cpu target"
    )
    var rows = Int(batch_size * seq_len)
    var in_channels = Int(channels)
    var out_channels = Int(output_channels)

    # a = weight [out_channels, in_channels], 2D 16x16 block scale.
    # b = input  [rows, in_channels], 1D 1x16 block scale.
    var a_q_scratch = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_packed_size(out_channels, in_channels)
    )
    var a_scale_scratch = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_scale_buffer_size(out_channels, in_channels, NVFP4_BLOCK)
    )
    var a_tensor_scale_scratch = ctx.enqueue_create_buffer[DType.float32](1)
    var b_q_scratch = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_packed_size(rows, in_channels)
    )
    var b_scale_scratch = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_scale_buffer_size(rows, in_channels, 1)
    )
    var b_tensor_scale_scratch = ctx.enqueue_create_buffer[DType.float32](1)

    lowp_gemm_fp4[
        dtype,
        dtype,
        target,
        a_block_rows=NVFP4_BLOCK,
        b_block_rows=1,
    ](
        out_ptr,
        out_ptr,  # d_raw_scratch aliases d_ptr -- accumulate=False (default),
        # safe: the post-scale kernel reads then writes each index in place.
        weight_ptr,
        input_ptr,
        device_buf_mut_ptr(a_q_scratch),
        device_buf_mut_ptr(a_scale_scratch),
        device_buf_mut_ptr(a_tensor_scale_scratch),
        device_buf_mut_ptr(b_q_scratch),
        device_buf_mut_ptr(b_scale_scratch),
        device_buf_mut_ptr(b_tensor_scale_scratch),
        out_channels,
        rows,
        in_channels,
        ctx,
    )

    # Bias (+GELU) epilogue in bf16 — same standalone kernel matmul_fwd_lowp
    # uses (lowp_gemm_fp4 has no fused epilogue of its own).
    comptime if has_bias or use_gelu:
        bias_gelu_fwd[dtype, target, has_bias=has_bias, use_gelu=use_gelu](
            out_ptr, pre_gelu_ptr, bias_ptr, rows, out_channels, ctx
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
    return persistent_device_buffer[DType.float32](
        ctx, "DBIAS_SCRATCH", CAP, zero=True
    )


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
    return persistent_device_buffer[DType.int32](
        ctx, "DBIAS_COUNTERS", CAP, zero=True
    )


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
    # the row reduction so occupancy stays high even for small OC. Each block
    # writes its per-column partial to a [row_blocks, OC] scratch (no atomics;
    # the finalize pass reduces it — see the store below).
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
    # RAGGED OC: out_channels need not be a multiple of `width`. The caller
    # rounds the column-group count up (`ceildiv(oc, width)`), so the single
    # last thread of the last column-block may own fewer than `width`
    # columns; that thread takes the scalar-tail branch below (in both the
    # accum and finalize halves) instead of a width-wide transaction, which
    # would read/write past column oc into the next row-block's scratch
    # region. Every aligned thread — and every bias this model actually
    # produces, C/3C/4C at 768/2304/3072, is a multiple of 8 — stays on the
    # branch-free 128-bit fast path; the tail branch is a predictable,
    # at-most-once-per-block divergence that only the odd-OC tests exercise.
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
        if col + width <= out_channels:
            # Aligned fast path: one 128-bit vector transaction per row.
            var acc = SIMD[DType.float32, width](0.0)
            for r in range(r0, r1):
                acc += (
                    (d_output_ptr + r * out_channels + col)
                    .load[width=width]()
                    .cast[DType.float32]()
                )
            (scratch + by * out_channels + col).store(acc)
        else:
            # Ragged tail (oc % width != 0): this thread owns the final
            # <width columns. Reduce them one at a time — a width-wide
            # load/store here would run past column oc and clobber the next
            # row-block's scratch. Only the last thread of the last
            # column-block ever reaches this branch.
            for c in range(col, out_channels):
                var acc = Scalar[DType.float32](0.0)
                for r in range(r0, r1):
                    acc += d_output_ptr[r * out_channels + c].cast[
                        DType.float32
                    ]()
                scratch[by * out_channels + c] = acc

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

    if col + width <= out_channels:
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
    else:
        # Ragged-tail finalize — scalar, mirroring the scalar accum above.
        for c in range(col, out_channels):
            var total = Scalar[DType.float32](0.0)
            for rb in range(row_blocks):
                total += scratch[rb * out_channels + c]
            comptime if accumulate:
                d_bias_ptr[c] = (
                    d_bias_ptr[c].cast[DType.float32]() + total
                ).cast[dtype]()
            else:
                d_bias_ptr[c] = total.cast[dtype]()

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
            # (a naive port that just widened per-thread loads 1:1 was 2x
            # SLOWER, not faster):
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
            # num_groups = number of width-wide column vectors, rounded UP so
            # a ragged tail (oc % width != 0) still gets a thread; that single
            # last thread runs the scalar-tail branch in `_dbias_fused_gpu`
            # over its <width leftover columns (see the kernel).
            var num_groups = ceildiv(oc, width)
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
    activation grads; only weight/bias grads accumulate across micro-steps.

    fp8 dgrad/wgrad live in the sibling *_lowp entry points rather than a
    branch here; this function keeps exactly its pre-fp8 behavior.
    """
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


def _rht_transpose_tiled_kernel[
    dtype: DType,
    BLOCK_SIZE: Int,
](
    dst: MutKernelPtr[dtype],
    src: ImmutKernelPtr[dtype],
    rows: Int,
    cols: Int,
) -> None:
    """Coalesced 32x32-shared-memory-tile transpose of `src[rows, cols]`
    (row-major) into `dst[cols, rows]` (row-major) with `llmm/hadamard.mojo`'s
    forward 16-wide RHT (`y = H16 @ (s (*) x)`, unnormalized Sylvester
    butterfly) FUSED into the write phase along dst's trailing `rows` axis —
    one kernel replacing `_rht_transpose_prep`'s previous two-pass
    `_gpu_transpose_kernel` + `hadamard16_fwd_gpu` pipeline (the naive
    one-thread-one-element transpose has a maximally-strided WRITE,
    `dst[c*rows+r] = src[idx]`, structurally identical to fp8's original
    `_quantize_transpose_kernel` before its proven 32x32-tile rewrite — see
    `llmm/lowp.mojo`'s `_quantize_transpose_kernel` and this file's own
    `_gpu_transpose_add_into_kernel`; the separate in-place hadamard pass
    additionally cost a full extra read+write of the whole tensor).

    Tile shape: each 32x32 tile of `src` is loaded coalesced into shared
    memory (padded to 32x33 to dodge bank conflicts on the transposed read),
    then written transposed to `dst` with coalesced stores — mirrors
    `_gpu_transpose_add_into_kernel`'s two-phase load/barrier/store shape.

    RHT fusion: in phase 2, the 16 elements of a Hadamard block are 16
    consecutive dst-column (`gr = tile_r + tx`) positions = 16 consecutive
    LANES of the writing warp (`tx = thread_idx.x % 32` is the lane id, so
    lanes 0-15 / 16-31 hold the tile's two Hadamard blocks along the
    contraction axis). The butterfly therefore runs as 4 `warp.shuffle_xor`
    stages (offsets 1/2/4/8 — confined within each 16-lane half by
    construction) on the fp32-cast value, after the sign multiply
    `hadamard_sign(tx % 16)`, before the single bf16 store. Stage h computes
    exactly `_fwht16`'s `buf[j] = a + b; buf[j+h] = a - b` pair (bit-h-clear
    lane: `v + partner`; bit-h-set lane: `partner - v`), on the same
    fp32-cast inputs, with one final cast to `dtype` at the store — the same
    op sequence per output element as the unfused
    `_gpu_transpose_kernel`-then-`_hadamard16_kernel[forward=True]` pipeline
    (which also round-trips the UNTRANSFORMED bf16 value through global
    memory unrounded, casts to fp32 once, butterflies in registers, and
    casts back once), so the fused result is bit-identical.

    REQUIRES `rows % 16 == 0` (raised by `_rht_transpose_prep`, and already
    `hadamard16_fwd_gpu`'s own contract): guarantees a 16-lane Hadamard
    half-warp never straddles the `rows` boundary, so out-of-bounds lanes
    only ever exchange values with other out-of-bounds lanes (which never
    store). Out-of-tile SMEM slots are zero-filled in phase 1 so the
    shuffles stay NaN-free regardless.
    """
    comptime TILE = 32
    comptime STRIDE = TILE + 1  # 33 — avoids shared-memory bank conflicts
    var tile = LayoutTensor[
        dtype,
        Layout.row_major(TILE, STRIDE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var tiles_r = ceildiv(rows, TILE)  # tiles along src's row dimension
    var tiles_c = ceildiv(cols, TILE)  # tiles along src's col dimension
    var total_tiles = tiles_r * tiles_c

    var tx = Int(thread_idx.x) % TILE
    var ty = Int(thread_idx.x) // TILE
    comptime ROW_STEP = BLOCK_SIZE // TILE

    var bt = Int(block_idx.x)
    while bt < total_tiles:
        var tile_r = (bt // tiles_c) * TILE  # row offset in src
        var tile_c = (bt % tiles_c) * TILE  # col offset in src

        # Phase 1 — coalesced load: thread tx reads src column (tile_c+tx);
        # consecutive tx -> consecutive src columns -> coalesced read.
        # Out-of-bounds slots are zero-filled (see docstring: keeps the
        # phase-2 shuffles NaN-free; OOB lanes never store).
        var r = ty
        while r < TILE:
            var gr = tile_r + r
            var gc = tile_c + tx
            var lv: Scalar[dtype]
            if gr < rows and gc < cols:
                lv = src[gr * cols + gc]
            else:
                lv = Scalar[dtype](0)
            tile.ptr[r * STRIDE + tx] = lv
            r += ROW_STEP
        barrier()

        # Phase 2 — coalesced transposed write with the fused RHT butterfly:
        # thread tx writes dst column (tile_r+tx); consecutive tx ->
        # consecutive dst columns -> coalesced write. The shuffle stages run
        # UNGUARDED (every lane participates uniformly — the loop bound is
        # uniform and only the store is bounds-guarded).
        r = ty
        while r < TILE:
            var gc = tile_c + r  # dst row index (src col)
            var gr = tile_r + tx  # dst col index (src row)
            var v = tile.ptr[tx * STRIDE + r].cast[
                DType.float32
            ]() * hadamard_sign(tx & 15)
            var p = shuffle_xor(v, UInt32(1))
            v = (p - v) if (tx & 1) != 0 else (v + p)
            p = shuffle_xor(v, UInt32(2))
            v = (p - v) if (tx & 2) != 0 else (v + p)
            p = shuffle_xor(v, UInt32(4))
            v = (p - v) if (tx & 4) != 0 else (v + p)
            p = shuffle_xor(v, UInt32(8))
            v = (p - v) if (tx & 8) != 0 else (v + p)
            if gc < cols and gr < rows:
                dst[gc * rows + gr] = v.cast[dtype]()
            r += ROW_STEP
        barrier()

        bt += Int(grid_dim.x)


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
    (O(rows * OC) traffic next to the GEMM's O(2 * rows * OC * C) flops).

    fp8 dgrad/wgrad live in the sibling *_lowp entry points rather than a
    branch here; this function keeps exactly its pre-fp8 behavior.
    """
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


# fp8 backward — separate sibling entry points (matmul_d_input_bwd_lowp /
# matmul_d_weight_bwd_lowp / matmul_bwd_lowp), mirroring the forward split:
# the fp8 signatures need AmaxStates and cached transposed operands that the
# bf16 entry points must not carry.
#
# Operand orientations (see the module comment above `_matmul_cublaslt_fp8`
# for the full TN-orientation derivation):
#   dgrad (matmul_d_input_bwd_lowp): A-role=weight (transpose_a=True, needs a
#     transposed fp8 copy, E4M3), B-role=d_output (transpose_b=False, native,
#     E5M2).
#   wgrad (matmul_d_weight_bwd_lowp): A-role=input (transpose_a=True, E4M3),
#     B-role=d_output (transpose_b=True, E5M2) — BOTH operands need a
#     transposed fp8 copy. `accumulate` passes straight through to
#     `lowp_gemm_devscale`'s `accumulate` -> cuBLASLt `beta` (confirmed
#     working for fp8-operand GEMMs — no bf16-scratch-and-add fallback
#     needed).
# DGELU is NOT fused into the fp8 GEMM epilogue (cuBLASLt fp8 has restricted
# epilogue support): it runs as the existing separate bf16
# `_launch_matmul_gelu_backward_scaling_gpu` kernel afterward, same as
# `matmul_d_input_bwd`'s own `USE_GELU_FUSION=False` bf16 path.
#
# `AmaxState` "once per step" contract (llmm/amax.mojo): `weight_state` (for
# dgrad) and `input_state` (for wgrad) are the SAME per-site `AmaxState`
# `matmul_fwd_lowp` already updated during THIS step's forward pass — dgrad/
# wgrad never call `update_scale` on them again (that would double-push this
# step's amax into the ring buffer). `doutput` has no forward counterpart,
# so `matmul_bwd_lowp` updates it exactly once (before either sub-GEMM runs)
# and both dgrad and wgrad read the result — see `matmul_bwd_lowp`'s
# docstring.
#
# By default dgrad/wgrad re-quantize weight's/input's transposed fp8 copy
# themselves, fresh at consumption time, into the `(site, layer)`-keyed
# cache buffer (`lowp_transpose_cache`). Under -D LLMM_FP8_STASH_LEGACY=1
# `matmul_fwd_lowp` already produced it this micro-step (dual-output,
# alongside its own natural-layout quantize) and dgrad/wgrad only read it.
# Either way the GEMM consumes the cache via `lowp_gemm_devscale`'s
# `quantize_a=False`. See the module comment above `matmul_fwd_lowp` for
# the full lifetime/ownership design and docs/ai/low_precision_gotchas.md
# G2 for why that cache is a persistent process-global buffer rather than
# a plain per-call `DeviceBuffer`.


def matmul_d_input_bwd_lowp[
    dtype: DType,
    target: StaticString,
    use_gelu: Bool,
](
    d_input_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    doutput_fp8_nat_ptr: MutKernelPtr[DType.uint8],
    weight_ptr: ImmutKernelPtr[dtype],
    pre_gelu_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    output_channels: Int64,
    weight_state: AmaxState[FP8_SPEC],
    doutput_state: AmaxState[FP8_SPEC],
    site: StaticString,
    layer: Int,
    ctx: DeviceContext,
) raises -> None:
    """Dgrad (fp8): `d_input = d_output @ weight` via `lowp_gemm_devscale`.
    `weight_state`/`doutput_state` are read-only here (not `mut`) — see the
    module comment above for why neither is updated in this function.

    `doutput_fp8_nat_ptr`: the natural-layout fp8 quantization of
    `d_output`, already produced by `matmul_bwd_lowp`'s single
    `quantize_dual_devscale` call (shared with `matmul_d_weight_bwd_lowp`'s
    transposed copy — same tensor, same scale, one read instead of two).
    `d_output_ptr` (bf16) is still accepted only to satisfy
    `lowp_gemm_devscale`'s uniform signature under `quantize_b=False` — it
    is not read.

    `site`/`layer` (see the module comment above `matmul_fwd_lowp`):
    identify this call site's persistent transposed-fp8 cache buffer
    (`lowp_transpose_cache`). By default this function re-quantizes
    `weight` into it fresh (`weight_ptr` IS read); under
    -D LLMM_FP8_STASH_LEGACY=1 the forward already filled it and
    `weight_ptr` is accepted only to satisfy `lowp_gemm_devscale`'s
    uniform signature under `quantize_a=False`.
    """
    comptime assert is_gpu[target](), (
        "matmul_d_input_bwd_lowp is GPU-only; low-precision kernels must"
        " never be instantiated for the cpu target"
    )
    var rows = Int(batch_size * seq_len)
    var in_channels = Int(channels)
    var out_channels = Int(output_channels)

    var weight_t = lowp_transpose_cache(
        ctx, "WT", site, layer, in_channels * out_channels
    )
    comptime if not FP8_STASH_LEGACY:
        # Default (see the flag comment near the top of this file): the
        # forward wrote NO transposed stash — re-quantize weight's
        # transposed fp8 copy fresh, immediately before the consuming GEMM,
        # from the same bf16 source at the same (not-updated-since-forward)
        # scale. `weight_ptr` IS read here on this path.
        quantize_transpose_devscale[
            FP8_SPEC, FP8_SPEC.fwd_dtype, dtype, target
        ](
            weight_t,
            weight_ptr,
            kernel_ptr_as_immut(device_buf_mut_ptr(weight_state.scale)),
            out_channels,
            in_channels,
            ctx,
        )

    lowp_gemm_devscale[
        FP8_SPEC.fwd_dtype,
        FP8_SPEC.bwd_dtype,
        dtype,
        dtype,
        target,
        transpose_a=True,
        transpose_b=False,
        quantize_a=False,
        quantize_b=False,
    ](
        d_input_ptr,
        weight_ptr,
        d_output_ptr,
        weight_t,
        doutput_fp8_nat_ptr,
        kernel_ptr_as_immut(device_buf_mut_ptr(weight_state.scale)),
        kernel_ptr_as_immut(device_buf_mut_ptr(weight_state.scale_inv)),
        kernel_ptr_as_immut(device_buf_mut_ptr(doutput_state.scale)),
        kernel_ptr_as_immut(device_buf_mut_ptr(doutput_state.scale_inv)),
        in_channels,
        rows,
        out_channels,
        False,  # d_input is always overwritten, never accumulated
        ctx,
    )
    comptime if use_gelu:
        _launch_matmul_gelu_backward_scaling_gpu[dtype](
            d_input_ptr, pre_gelu_ptr, rows * in_channels, ctx
        )


def matmul_d_weight_bwd_lowp[
    dtype: DType,
    target: StaticString,
    accumulate: Bool,
](
    d_weight_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    doutput_fp8_t_ptr: MutKernelPtr[DType.uint8],
    input_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    output_channels: Int64,
    input_state: AmaxState[FP8_SPEC],
    doutput_state: AmaxState[FP8_SPEC],
    site: StaticString,
    layer: Int,
    ctx: DeviceContext,
) raises -> None:
    """Wgrad (fp8): `d_weight = d_output^T @ input` via `lowp_gemm_devscale`.
    `input_state`/`doutput_state` are read-only here (not `mut`) — see the
    module comment above for why neither is updated in this function.

    `doutput_fp8_t_ptr` (see `matmul_d_input_bwd_lowp`'s docstring): the
    transposed-layout fp8 quantization of `d_output`, already produced by
    `matmul_bwd_lowp`'s single `quantize_dual_devscale` call. `d_output_ptr`
    (bf16) is still accepted only to satisfy `lowp_gemm_devscale`'s uniform
    signature under `quantize_b=False` — it is not read.

    `site`/`layer` (see the module comment above `matmul_fwd_lowp`):
    identify this call site's persistent transposed-fp8 cache buffer
    (`lowp_transpose_cache`). By default this function re-quantizes
    `input` into it fresh (`input_ptr` IS read); under
    -D LLMM_FP8_STASH_LEGACY=1 the forward already filled it and
    `input_ptr` is accepted only to satisfy `lowp_gemm_devscale`'s
    uniform signature under `quantize_a=False`.
    """
    comptime assert is_gpu[target](), (
        "matmul_d_weight_bwd_lowp is GPU-only; low-precision kernels must"
        " never be instantiated for the cpu target"
    )
    var rows = Int(batch_size * seq_len)
    var in_channels = Int(channels)
    var out_channels = Int(output_channels)

    var input_t = lowp_transpose_cache(
        ctx, "IT", site, layer, in_channels * rows
    )
    comptime if not FP8_STASH_LEGACY:
        # Default (see the flag comment near the top of this file): the
        # forward wrote NO transposed stash — re-quantize input's
        # transposed fp8 copy fresh, immediately before the consuming GEMM,
        # from the same bf16 source at the same (not-updated-since-forward)
        # scale. `input_ptr` IS read here on this path.
        quantize_transpose_devscale[
            FP8_SPEC, FP8_SPEC.fwd_dtype, dtype, target
        ](
            input_t,
            input_ptr,
            kernel_ptr_as_immut(device_buf_mut_ptr(input_state.scale)),
            rows,
            in_channels,
            ctx,
        )

    lowp_gemm_devscale[
        FP8_SPEC.fwd_dtype,
        FP8_SPEC.bwd_dtype,
        dtype,
        dtype,
        target,
        transpose_a=True,
        transpose_b=True,
        quantize_a=False,
        quantize_b=False,
    ](
        d_weight_ptr,
        input_ptr,
        d_output_ptr,
        input_t,
        doutput_fp8_t_ptr,
        kernel_ptr_as_immut(device_buf_mut_ptr(input_state.scale)),
        kernel_ptr_as_immut(device_buf_mut_ptr(input_state.scale_inv)),
        kernel_ptr_as_immut(device_buf_mut_ptr(doutput_state.scale)),
        kernel_ptr_as_immut(device_buf_mut_ptr(doutput_state.scale_inv)),
        in_channels,
        out_channels,
        rows,
        accumulate,
        ctx,
    )


def matmul_bwd_lowp[
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
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    output_channels: Int64,
    input_state: AmaxState[FP8_SPEC],
    weight_state: AmaxState[FP8_SPEC],
    mut doutput_state: AmaxState[FP8_SPEC],
    site: StaticString,
    layer: Int,
    ctx: DeviceContext,
) raises -> None:
    """Backward for one block-linear site (fp8): bias grad (bf16, unchanged
    `matmul_bias_bwd`) + dgrad + wgrad (both fp8). Sibling of `matmul_bwd`,
    mirroring the forward `matmul_fwd`/`matmul_fwd_lowp` split.

    `site`/`layer` (see the module comment above `matmul_fwd_lowp`) MUST be
    the identical `(site, layer)` pair the
    forward pass's `matmul_fwd_lowp` call for this same GEMM site used this
    micro-step — they select the persistent weight/input transposed-fp8
    caches `matmul_d_input_bwd_lowp`/`matmul_d_weight_bwd_lowp` read below.

    `doutput_state` is this function's only `mut` `AmaxState`: `d_output`
    has no forward counterpart (design §1.2 — it's the backward-only E5M2
    gradient operand), so its amax/scale is computed and pushed into the
    delayed-scaling history EXACTLY ONCE here, before either sub-GEMM runs.
    Both `matmul_d_input_bwd_lowp` (dgrad) and `matmul_d_weight_bwd_lowp`
    (wgrad) then read the resulting `doutput_state.scale`/`scale_inv`
    read-only, since they consume the SAME d_output tensor/site — calling
    `update_scale` twice per step would double-push this step's amax into
    the ring buffer, violating the "once per step" contract
    (`AmaxState`'s docstring in llmm/amax.mojo). `input_state`/`weight_state`
    are read-only for the complementary reason: `matmul_fwd_lowp` already
    updated them during this same step's forward pass, before backward ever
    runs (see `matmul_d_input_bwd_lowp`'s docstring / the module comment
    above).

    `d_output` is quantized exactly ONCE here — natural AND transposed fp8
    copies from a single read, via
    `quantize_dual_devscale` — rather than `matmul_d_input_bwd_lowp` and
    `matmul_d_weight_bwd_lowp` each separately re-reading/re-encoding it
    from bf16 (they used to call `quantize_devscale`/
    `quantize_transpose_devscale` on `d_output_ptr` internally; now they
    take the pre-quantized buffers this function fills). Bit-identical to
    the old two-separate-calls behavior (same `encode_fp8` per output
    element at the same scale) — purely a memory-traffic optimization.
    """
    comptime if has_bias:
        matmul_bias_bwd[dtype, target, accumulate](
            d_bias_ptr,
            d_output_ptr,
            batch_size,
            seq_len,
            output_channels,
            ctx,
        )

    var rows = Int(batch_size * seq_len)
    var out_channels = Int(output_channels)
    # See the matching comment in `matmul_fwd_lowp` above: under
    # `-D LLMM_FP8_STATIC_SCALES=1`, `doutput_state.scale` was seeded once
    # at `Fp8State.__init__` time and never updated — skip the amax
    # reduction + scale-update kernel entirely (comptime-gated).
    comptime if FP8_BWD_MUTEX or FP8_BWD_EXEC_MUTEX:
        var lk = _fp8_gemm_lock()
        while True:
            var expected = Scalar[DType.int32](0)
            if lk[].compare_exchange(expected, 1):
                break
            _ = external_call["sched_yield", Int32]()

    # Persistent scratch, NOT per-call DeviceBuffers: a per-call buffer's
    # release is not reliably ordered after consumers enqueued post-borrow
    # (G2/G3, and the 2026-07-22 multi-rank NaN hunt), so the fp8 backward
    # keeps NO per-call device allocations. Per-(site) sizes are fixed for
    # the whole run (B/T/channels invariant); intra-call reuse is safe on
    # the rank's single in-order stream.
    comptime if not FP8_STATIC_SCALES:
        var amax_doutput = persistent_device_buffer[DType.float32](
            ctx, "FP8_AMAX_DOUT", 1
        )
        compute_amax[FP8_SPEC, dtype](
            amax_doutput,
            d_output_ptr,
            rows * out_channels,
            ctx,
        )
        doutput_state.update_scale[FP8_SPEC.bwd_dtype](
            kernel_ptr_as_immut(amax_doutput), ctx
        )

    var doutput_fp8_nat = persistent_device_buffer[DType.uint8](
        ctx, String("FP8_BWD_DN_") + String(site), rows * out_channels
    )
    var doutput_fp8_t = persistent_device_buffer[DType.uint8](
        ctx, String("FP8_BWD_DT_") + String(site), out_channels * rows
    )
    quantize_dual_devscale[FP8_SPEC, FP8_SPEC.bwd_dtype, dtype, target](
        doutput_fp8_nat,
        doutput_fp8_t,
        d_output_ptr,
        kernel_ptr_as_immut(device_buf_mut_ptr(doutput_state.scale)),
        rows,
        out_channels,
        ctx,
    )

    matmul_d_input_bwd_lowp[dtype, target, use_gelu](
        d_input_ptr,
        d_output_ptr,
        doutput_fp8_nat,
        weight_ptr,
        pre_gelu_ptr,
        batch_size,
        seq_len,
        channels,
        output_channels,
        weight_state,
        doutput_state,
        site,
        layer,
        ctx,
    )
    matmul_d_weight_bwd_lowp[dtype, target, accumulate](
        d_weight_ptr,
        d_output_ptr,
        doutput_fp8_t,
        input_ptr,
        batch_size,
        seq_len,
        channels,
        output_channels,
        input_state,
        doutput_state,
        site,
        layer,
        ctx,
    )

    comptime if FP8_BWD_EXEC_MUTEX or FP8_BWD_SYNC_ONLY:
        # Drain this rank's stream BEFORE releasing the lock: every fp8
        # backward kernel enqueued above (quantize_dual + dgrad + wgrad and,
        # under dynamic scaling, amax/scale-update) must have finished
        # executing before another rank may start enqueueing its own —
        # execution-exclusion, not just submission-exclusion. Safe to hold
        # the lock across this sync: nothing inside `matmul_bwd_lowp`
        # requires another rank's progress (no collectives / host barriers
        # in this scope), so the blocked ranks' sched_yield spin cannot
        # deadlock against the sync.
        ctx.synchronize()
    comptime if FP8_BWD_MUTEX or FP8_BWD_EXEC_MUTEX:
        _fp8_gemm_lock()[].store(0)


# ===----------------------------------------------------------------------=== #
# matmul_d_input_bwd_fp4 / matmul_d_weight_bwd_fp4 / matmul_bwd_fp4 — NVFP4
# backward, mirroring the fp8 dgrad/wgrad/bundle split immediately above
# (`matmul_d_input_bwd_lowp`/`matmul_d_weight_bwd_lowp`/`matmul_bwd_lowp`)
# and `matmul_fwd_fp4` (forward). Separate sibling entry points, not a
# branch inside `matmul_d_input_bwd`/`matmul_d_weight_bwd`/`matmul_bwd`
# themselves — same rationale as fp8's split (those functions' signatures
# have no room for fp4's scratch buffers, and bf16/fp32/fp8 callers must
# stay byte-for-byte unchanged). `train_gpt2.mojo` wires these into a THIRD
# `elif PRECISION == "fp4":` branch alongside fp8's `comptime if
# FP8_BWD_ENABLED:`, gated per-layer by `_layer_in_fp4_range` (the same
# middle-block policy the forward pass uses) — outside that range it falls
# through to plain bf16 `matmul_bwd`.
#
# Recipe: Dgrad = SR on the gradient operand, no RHT. Wgrad = RHT on BOTH
# GEMM inputs + SR on the gradient operand.
#
# ## RHT integration
#
# `matmul_d_weight_bwd_fp4` applies `llmm/hadamard.mojo`'s fixed 16-wide RHT
# to BOTH Wgrad operands (`input`, `d_output`) along the contraction (K =
# `rows` = batch*seq_len) dimension, BEFORE NVFP4 quantization, per the
# recipe. **Design: a separate transpose+RHT materialization pass, not a
# fused prologue inside `_nvfp4_quantize_gpu`.** A fused prologue was
# considered (BLOCK_ROWS=1 means each quantize-kernel thread already owns
# exactly one 16-element block — a tempting match for HADAMARD_BLOCK=16) but
# rejected: `_nvfp4_quantize_impl` computes the per-TENSOR fp32 scale
# (`nvfp4_compute_tensor_scale`, a separate flat-amax reduction pass) BEFORE
# the per-block quantize kernel runs, reading `x_ptr` directly. If RHT were
# applied only inside the quantize kernel's per-block prologue, the tensor
# scale would be computed from the PRE-RHT data while the block scales are
# encoded against POST-RHT block amaxes — for Gaussian-ish data RHT scales
# per-block amax by roughly `sqrt(16)=4` (variance-additive sum of 16 signed
# terms), so `block_scale_raw = (rht_block_amax/6)/tensor_scale` would run
# ~4x too high, overflowing e4m3's block-scale encoding range and silently
# clipping/misscaling every block — a real correctness bug, not a rounding
# nuance. Materializing the RHT'd (and transposed, since the contraction
# dimension is the OUTER/non-contiguous axis of `input`/`d_output`'s natural
# `[rows, channels]` storage) tensor into a fresh bf16 scratch buffer BEFORE
# calling the ordinary (unmodified) `nvfp4_quantize` sidesteps this entirely
# — `nvfp4_compute_tensor_scale` then naturally sees the same RHT'd data the
# block-quantize step does. Cost: two extra kernel launches per operand
# (`_gpu_transpose_kernel` + `hadamard16_fwd_gpu`, the latter run in-place)
# and one extra `rows*channels`-sized bf16 scratch buffer per operand, on top
# of the quantize scratch `lowp_gemm_fp4` already needs — see
# `_rht_transpose_prep`'s docstring.
#
# The RHT-composition contract (`(H@a)^T @ (H@b) == 16 * a^T @ b`, verified
# in `tests/test_lowp_gemm_fp4.mojo::test_fp4_rht_quantize_gemm_contract`)
# means the resulting GEMM is 16x the desired Wgrad value; rather than an
# extra O(m*n) elementwise divide-by-16 pass over the (potentially large)
# weight-gradient output, the 1/16 is folded into `lowp_gemm_fp4`'s
# `extra_scale` parameter (multiplied into the existing per-tensor-scale
# post-scale step, `_nvfp4_post_scale_gpu` — effectively free). Both RHT'd
# scratch buffers are already in the `[channels, rows]` orientation
# `lowp_gemm_fp4` wants for a non-transposed read, so `transpose_a`/
# `transpose_b` are both `False` on the RHT path (unlike the no-RHT path's
# fused `nvfp4_quantize_transpose` reads directly off `input_ptr`/
# `d_output_ptr`).
#
# `-D LLMM_FP4_NO_RHT=1` (default 0, i.e. RHT ON) is an ablation flag: when
# set, `matmul_d_weight_bwd_fp4` falls back to EXACTLY the no-RHT code path
# (`nvfp4_quantize_transpose` directly on `input_ptr`/`d_output_ptr`, no RHT,
# no extra scratch) — same streams, same `sr_step` usage, nothing else
# touched, so the ablation reproduces the no-RHT path's numbers bit-for-bit
# (a cheap consistency check of the flag plumbing, not a new code path to
# validate independently).
#
# Operand orientations (same TN-only cuBLASLt constraint fp8's module
# comment above `_matmul_cublaslt_fp8` derives, mirrored here for NVFP4's
# `_matmul_cublaslt_fp4`):
#   dgrad (matmul_d_input_bwd_fp4): A-role=weight (transpose_a=True, needs a
#     transposed NVFP4 copy via `nvfp4_quantize_transpose`, 2D 16x16 block
#     scale per the recipe's weight-scaling rule, RNE — weights are never
#     SR'd), B-role=d_output (transpose_b=False, native orientation, 1D 1x16,
#     **SR** — the recipe's gradient operand). `d_input` is always
#     overwritten (accumulate=False, matching the fp8 sibling: activation
#     grads never accumulate across micro-steps, only weight/bias grads do).
#   wgrad (matmul_d_weight_bwd_fp4): A-role=input (transpose_a=True, needs a
#     transposed NVFP4 copy, 1D 1x16, RNE — an activation, not a gradient),
#     B-role=d_output (transpose_b=True, ALSO needs a transposed copy, 1D
#     1x16, **SR**). `accumulate` is a comptime template parameter threaded
#     straight from the caller (mirrors fp8's `matmul_d_weight_bwd_lowp`) —
#     unlike fp8 (whose cuBLASLt beta=1 accumulate works natively for fp8
#     operands), NVFP4's two-level scale means accumulation happens in the
#     elementwise post-scale-add step (`_nvfp4_post_scale_gpu`'s
#     `accumulate` branch), not cuBLASLt's `beta` — see that kernel's module
#     comment. This function always allocates a FRESH `d_raw_scratch`
#     buffer (distinct from `d_weight_ptr`) regardless of `accumulate`'s
#     value, avoiding a conditional-aliasing footgun, consistent with this
#     function's existing "no persistent state, fresh scratch every call"
#     design (fp4 does not use `AmaxState` at all — see `matmul_fwd_fp4`'s
#     docstring).
#
# DGELU is NOT fused into the fp4 GEMM epilogue, same reasoning as fp8's
# `matmul_d_input_bwd_lowp`: it runs as the existing separate bf16
# `_launch_matmul_gelu_backward_scaling_gpu` kernel afterward.
#
# No `AmaxState`: every quantize call here (like `matmul_fwd_fp4`) computes
# both NVFP4 scale levels fresh from the current bf16 tensor, every call —
# there is no delayed-scaling history to maintain and no "once per step"
# update-ownership contract to reason about (contrast fp8's `AmaxState`
# module comment above `matmul_d_input_bwd_lowp`). This also means Dgrad's
# weight-operand quantization does NOT reuse Fprop's already-quantized 2D
# 16x16 weight buffer — the recipe's "same buffer serves Fprop row-major and
# Dgrad column-major without requantizing" is a PERFORMANCE optimization
# `matmul_fwd_fp4`'s module comment flags as explicitly deferred; freshly
# re-quantizing here is functionally equivalent (RNE is a deterministic pure
# function of the weight tensor) at the cost of redundant compute, left on
# the table for a later performance pass.
#
# SR seed/stream/step scheme: `sr_step` should be the caller's training-step
# counter (thread from `train_gpt2.mojo`'s outer loop through `backward()`,
# NOT always 0 — see the `train_gpt2.mojo` call-site comment for the exact
# `step`/`micro_step` combination used) so repeated backward calls across
# training steps draw fresh dither rather than reusing the same bit pattern
# every step under the fixed default seed. `NVFP4_SR_STREAM_DGRAD_DOUTPUT`/
# `NVFP4_SR_STREAM_WGRAD_DOUTPUT` (llmm/nvfp4_quant.mojo) keep dgrad's and
# wgrad's `d_output` SR draws on disjoint substreams from each other and
# from the forward reservation, even though both consume the same
# underlying `d_output` tensor values in different layouts.
# ===----------------------------------------------------------------------=== #

# Ablation flag: -D LLMM_FP4_NO_RHT=1 disables the Wgrad RHT (see the RHT
# section above).
comptime LLMM_FP4_NO_RHT = get_defined_int["LLMM_FP4_NO_RHT", 0]() != 0
comptime LLMM_FP4_WGRAD_RHT = not LLMM_FP4_NO_RHT


def _rht_transpose_prep[
    dtype: DType,
](
    scratch_ptr: MutKernelPtr[dtype],  # >= rows*cols; becomes [cols, rows]
    src_ptr: ImmutKernelPtr[dtype],  # [rows, cols] row-major source
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises -> None:
    """Materializes RHT(src^T) into scratch_ptr in ONE kernel launch
    (coalesced 32x32 tiled transpose with the 16-wide RHT fused into the
    write phase). rows (the Wgrad contraction dim) must be a multiple of 16.
    """
    if rows % 16 != 0:
        raise Error("_rht_transpose_prep: rows must be a multiple of 16")
    comptime BLOCK_SIZE = 256
    comptime TILE = 32
    var tiles = ceildiv(rows, TILE) * ceildiv(cols, TILE)
    comptime t_kernel = _rht_transpose_tiled_kernel[dtype, BLOCK_SIZE]
    var t_compiled = ctx.compile_function[t_kernel]()
    ctx.enqueue_function(
        t_compiled,
        scratch_ptr,
        src_ptr,
        rows,
        cols,
        grid_dim=(tiles,),
        block_dim=(BLOCK_SIZE,),
    )


def matmul_d_input_bwd_fp4[
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
    sr_seed: UInt64 = NVFP4_SR_SEED,
    sr_step: Int = 0,
) raises -> None:
    """NVFP4 dgrad: `d_input = d_output @ weight` via `lowp_gemm_fp4`. See
    the module comment above for the orientation/rounding derivation.
    """
    comptime assert is_gpu[target](), (
        "matmul_d_input_bwd_fp4 is GPU-only; low-precision kernels must"
        " never be instantiated for the cpu target"
    )
    var rows = Int(batch_size * seq_len)
    var in_channels = Int(channels)
    var out_channels = Int(output_channels)

    # a = weight [out_channels, in_channels] (transposed read), 2D 16x16, RNE.
    var a_q_scratch = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_packed_size(in_channels, out_channels)
    )
    var a_scale_scratch = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_scale_buffer_size(in_channels, out_channels, NVFP4_BLOCK)
    )
    var a_tensor_scale_scratch = ctx.enqueue_create_buffer[DType.float32](1)
    # b = d_output [rows, out_channels] (native), 1D 1x16, SR.
    var b_q_scratch = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_packed_size(rows, out_channels)
    )
    var b_scale_scratch = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_scale_buffer_size(rows, out_channels, 1)
    )
    var b_tensor_scale_scratch = ctx.enqueue_create_buffer[DType.float32](1)

    lowp_gemm_fp4[
        dtype,
        dtype,
        target,
        a_block_rows=NVFP4_BLOCK,
        b_block_rows=1,
        a_round_mode=ROUND_MODE_RNE,
        b_round_mode=ROUND_MODE_STOCHASTIC,
        accumulate=False,  # d_input is always overwritten, never accumulated
        transpose_a=True,
        transpose_b=False,
    ](
        d_input_ptr,
        d_input_ptr,  # d_raw_scratch aliases d_ptr (accumulate=False, safe)
        weight_ptr,
        d_output_ptr,
        device_buf_mut_ptr(a_q_scratch),
        device_buf_mut_ptr(a_scale_scratch),
        device_buf_mut_ptr(a_tensor_scale_scratch),
        device_buf_mut_ptr(b_q_scratch),
        device_buf_mut_ptr(b_scale_scratch),
        device_buf_mut_ptr(b_tensor_scale_scratch),
        in_channels,
        rows,
        out_channels,
        ctx,
        sr_seed,
        NVFP4_SR_STREAM,  # a_sr_stream: unused (a_round_mode is RNE)
        NVFP4_SR_STREAM_DGRAD_DOUTPUT,
        sr_step,
    )
    comptime if use_gelu:
        _launch_matmul_gelu_backward_scaling_gpu[dtype](
            d_input_ptr, pre_gelu_ptr, rows * in_channels, ctx
        )


def matmul_d_weight_bwd_fp4[
    dtype: DType,
    target: StaticString,
    accumulate: Bool,
](
    d_weight_ptr: MutKernelPtr[dtype],
    d_output_ptr: ImmutKernelPtr[dtype],
    input_ptr: ImmutKernelPtr[dtype],
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    output_channels: Int64,
    ctx: DeviceContext,
    sr_seed: UInt64 = NVFP4_SR_SEED,
    sr_step: Int = 0,
) raises -> None:
    """NVFP4 wgrad: `d_weight = d_output^T @ input` via `lowp_gemm_fp4`. See
    the module comment above for the orientation/rounding/accumulate
    derivation and the RHT-integration design (both operands RHT'd
    along the contraction dimension before quantization, `-D
    LLMM_FP4_NO_RHT=1` ablates back to the plain RNE/SR path).
    """
    comptime assert is_gpu[target](), (
        "matmul_d_weight_bwd_fp4 is GPU-only; low-precision kernels must"
        " never be instantiated for the cpu target"
    )
    var rows = Int(batch_size * seq_len)
    var in_channels = Int(channels)
    var out_channels = Int(output_channels)

    # Scratch shapes are identical in the RHT and no-RHT branches below
    # (only the source data written into a_q_scratch/b_q_scratch differs),
    # so the six quantize buffers + raw-GEMM scratch are allocated once here
    # rather than duplicated per branch.
    var a_q_scratch = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_packed_size(in_channels, rows)
    )
    var a_scale_scratch = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_scale_buffer_size(in_channels, rows, 1)
    )
    var a_tensor_scale_scratch = ctx.enqueue_create_buffer[DType.float32](1)
    var b_q_scratch = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_packed_size(out_channels, rows)
    )
    var b_scale_scratch = ctx.enqueue_create_buffer[DType.uint8](
        nvfp4_scale_buffer_size(out_channels, rows, 1)
    )
    var b_tensor_scale_scratch = ctx.enqueue_create_buffer[DType.float32](1)
    # Always a FRESH raw-GEMM scratch buffer -- see the module comment
    # above for why this avoids a conditional-aliasing footgun.
    var d_raw_scratch = ctx.enqueue_create_buffer[dtype](
        in_channels * out_channels
    )

    comptime if LLMM_FP4_WGRAD_RHT:
        # RHT both operands (along the K=rows contraction
        # dimension) into scratch BEFORE quantizing, then run a
        # NON-transposed NVFP4 GEMM (the RHT scratch is already in
        # `[channels, rows]` orientation) with the `(H@a)^T@(H@b) == 16*a^T@b`
        # factor folded into `extra_scale` — see the module comment above
        # `matmul_d_input_bwd_fp4` for the full design rationale.
        var a_rht_scratch = ctx.enqueue_create_buffer[dtype](in_channels * rows)
        var b_rht_scratch = ctx.enqueue_create_buffer[dtype](
            out_channels * rows
        )
        _rht_transpose_prep[dtype](
            device_buf_mut_ptr(a_rht_scratch),
            input_ptr,
            rows,
            in_channels,
            ctx,
        )
        _rht_transpose_prep[dtype](
            device_buf_mut_ptr(b_rht_scratch),
            d_output_ptr,
            rows,
            out_channels,
            ctx,
        )

        # a = RHT(input)^T [in_channels, rows], already correctly oriented
        # (k=rows trailing), 1D 1x16, RNE.
        # b = RHT(d_output)^T [out_channels, rows], already correctly
        # oriented, 1D 1x16, SR.
        lowp_gemm_fp4[
            dtype,
            dtype,
            target,
            a_block_rows=1,
            b_block_rows=1,
            a_round_mode=ROUND_MODE_RNE,
            b_round_mode=ROUND_MODE_STOCHASTIC,
            accumulate=accumulate,
            transpose_a=False,
            transpose_b=False,
        ](
            d_weight_ptr,
            device_buf_mut_ptr(d_raw_scratch),
            kernel_ptr_as_immut(device_buf_mut_ptr(a_rht_scratch)),
            kernel_ptr_as_immut(device_buf_mut_ptr(b_rht_scratch)),
            device_buf_mut_ptr(a_q_scratch),
            device_buf_mut_ptr(a_scale_scratch),
            device_buf_mut_ptr(a_tensor_scale_scratch),
            device_buf_mut_ptr(b_q_scratch),
            device_buf_mut_ptr(b_scale_scratch),
            device_buf_mut_ptr(b_tensor_scale_scratch),
            in_channels,
            out_channels,
            rows,
            ctx,
            sr_seed,
            NVFP4_SR_STREAM,  # a_sr_stream: unused (a_round_mode is RNE)
            NVFP4_SR_STREAM_WGRAD_DOUTPUT,
            sr_step,
            extra_scale=Float32(1.0) / Float32(16.0),
        )
    else:
        # No-RHT ablation path (`-D LLMM_FP4_NO_RHT=1`): plain RNE/SR, same
        # streams, same `sr_step` usage, no RHT scratch -- the ablation
        # reproduces the no-RHT path bit-for-bit (a consistency check of the
        # ablation flag plumbing).
        # a = input [rows, in_channels] (transposed read), 1D 1x16, RNE.
        # b = d_output [rows, out_channels] (transposed read), 1D 1x16, SR.
        lowp_gemm_fp4[
            dtype,
            dtype,
            target,
            a_block_rows=1,
            b_block_rows=1,
            a_round_mode=ROUND_MODE_RNE,
            b_round_mode=ROUND_MODE_STOCHASTIC,
            accumulate=accumulate,
            transpose_a=True,
            transpose_b=True,
        ](
            d_weight_ptr,
            device_buf_mut_ptr(d_raw_scratch),
            input_ptr,
            d_output_ptr,
            device_buf_mut_ptr(a_q_scratch),
            device_buf_mut_ptr(a_scale_scratch),
            device_buf_mut_ptr(a_tensor_scale_scratch),
            device_buf_mut_ptr(b_q_scratch),
            device_buf_mut_ptr(b_scale_scratch),
            device_buf_mut_ptr(b_tensor_scale_scratch),
            in_channels,
            out_channels,
            rows,
            ctx,
            sr_seed,
            NVFP4_SR_STREAM,  # a_sr_stream: unused (a_round_mode is RNE)
            NVFP4_SR_STREAM_WGRAD_DOUTPUT,
            sr_step,
        )


def matmul_bwd_fp4[
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
    batch_size: Int64,
    seq_len: Int64,
    channels: Int64,
    output_channels: Int64,
    ctx: DeviceContext,
    sr_seed: UInt64 = NVFP4_SR_SEED,
    sr_step: Int = 0,
) raises -> None:
    """NVFP4 backward for one block-linear site: bias grad (bf16,
    unchanged `matmul_bias_bwd` — reused exactly like fp8's
    `matmul_bwd_lowp`) + dgrad + wgrad (both NVFP4). Sibling of `matmul_bwd`/
    `matmul_bwd_lowp`, mirroring `matmul_fwd`/`matmul_fwd_lowp`/
    `matmul_fwd_fp4`'s three-way split.
    """
    comptime if has_bias:
        matmul_bias_bwd[dtype, target, accumulate](
            d_bias_ptr,
            d_output_ptr,
            batch_size,
            seq_len,
            output_channels,
            ctx,
        )

    matmul_d_input_bwd_fp4[dtype, target, use_gelu](
        d_input_ptr,
        d_output_ptr,
        weight_ptr,
        pre_gelu_ptr,
        batch_size,
        seq_len,
        channels,
        output_channels,
        ctx,
        sr_seed,
        sr_step,
    )
    matmul_d_weight_bwd_fp4[dtype, target, accumulate](
        d_weight_ptr,
        d_output_ptr,
        input_ptr,
        batch_size,
        seq_len,
        channels,
        output_channels,
        ctx,
        sr_seed,
        sr_step,
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
