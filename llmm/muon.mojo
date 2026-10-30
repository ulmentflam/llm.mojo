# ===----------------------------------------------------------------------=== #
# Muon optimizer (Jordan et al. 2024): Nesterov-momentum SGD on 2-D weight
# matrices whose momentum is orthogonalized with 5 Newton-Schulz iterations in
# bf16, scaled per the Moonlight RMS rule (0.2 * sqrt(max(rows, cols))) and
# decoupled weight decay. Design + phase plan:
# docs/ai/muon_muonclip_implementation_plan.md.
#
# The signatures below are the contract the trainer (train_gpt2.mojo) and the
# pytest bridge (tests/kernels/muon.py) build against; keep the public names
# and argument orders stable.
#
# GPU path: batched cuBLASLt bf16 GEMMs (fp32 accumulate) over contiguous
# scratch + small elementwise kernels. cuBLASLt is column-major, so the
# "transpose when rows > cols" of the reference is free: a row-major
# [rows, cols] matrix with rows > cols is read directly as the column-major
# [R, C] (R = min, C = max) matrix X^T, and the Gram X X^T is then R x R.
# CPU path: naive fp32 reference (correctness only, not a perf target).
# ===----------------------------------------------------------------------=== #

from extensibility import register, InputTensor
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_gpu
from max.gpu import block_idx, thread_idx
from max.gpu.primitives import block
from std.atomic import Atomic
from std.math import sqrt, ceildiv
from std.sys import is_defined, get_defined_int

from llmm.memory import (
    ImmutKernelPtr,
    MutKernelPtr,
    persistent_device_buffer,
)
from llmm.rng_device import rng_step_counter, sr_cast_bf16
from llmm.vendor import HAS_CUBLAS
from llmm.matmul import _matmul_cublaslt

# Quintic Newton-Schulz coefficients (Jordan et al.).
comptime NS_A = Float32(3.4445)
comptime NS_B = Float32(-4.7750)
comptime NS_C = Float32(2.0315)
comptime NS_EPS = Float32(1e-7)

comptime BLOCK = 256
comptime ELEMS = 8  # elements per thread in the per-matrix kernels
comptime CHUNK = BLOCK * ELEMS

# Mirrors llmm/adamw.mojo's opt-in stochastic rounding of the low-precision
# param store (same macros, same seed), but with its own rng_device stream id
# (see the registry at the top of adamw.mojo) so the substreams are independent.
comptime SR_MASTER_ENABLED = is_defined["LLMM_SR_MASTER"]()
comptime SR_MASTER_SEED = UInt64(get_defined_int["LLMM_SR_SEED", 1746221221]())
comptime SR_MUON_STREAM = UInt64(2)


@fieldwise_init
struct MuonConfig(Copyable, Movable):
    """All hyperparameters fp32 regardless of parameter dtype (as AdamWConfig).
    """

    var learning_rate: Scalar[DType.float32]
    var momentum: Scalar[DType.float32]  # mu, 0.95 by default
    var weight_decay: Scalar[DType.float32]
    var grad_scale: Scalar[DType.float32]  # global grad-clip factor
    var ns_steps: Int  # Newton-Schulz iterations, 5
    var nesterov: Bool


@fieldwise_init
struct MuonGroup(Copyable, Movable):
    """`count` equally strided row-major [rows, cols] matrices inside the flat
    parameter buffer: matrix i starts at element `offset + i * stride`.
    `stride >= rows * cols`; for the Q/K/V row blocks of the fused
    [L, 3C, C] qkv weight, rows = cols = C and stride = 3*C*C."""

    var offset: Int
    var rows: Int
    var cols: Int
    var count: Int
    var stride: Int


# ===----------------------------------------------------------------------=== #
# Shared final-update store (mirrors llmm/adamw.mojo::_adamw_update's tail)
# ===----------------------------------------------------------------------=== #


@always_inline
def _muon_store[
    dtype: DType
](
    idx: Int,
    pidx: Int,
    params_ptr: MutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    has_master: Bool,
    o: Float32,
    lr: Float32,
    wd: Float32,
    scale: Float32,
    t: UInt32,
):
    """W -= lr * (scale * o + wd * W) in fp32 against the master when
    `has_master`, else against W itself. `idx` indexes params/master
    (already group-relative pointers), `pidx` is the global flat index used
    as the SR counter."""
    var old_param: Float32
    if has_master:
        old_param = master_ptr[unsafe_offset=idx]
    else:
        old_param = params_ptr[unsafe_offset=idx].cast[DType.float32]()
    var param = old_param - lr * (scale * o + wd * old_param)

    # Mirrors adamw: RNE cast by default, stochastic rounding under
    # -D LLMM_SR_MASTER=1 (bf16 stores only).
    comptime if SR_MASTER_ENABLED:
        comptime assert dtype == DType.bfloat16, (
            "LLMM_SR_MASTER=1 requires a bf16 low-precision param store"
            " (dtype == DType.bfloat16)"
        )
        var counter = rng_step_counter(UInt64(t), UInt64(pidx))
        params_ptr[unsafe_offset=idx] = sr_cast_bf16(
            param, SR_MASTER_SEED, counter, SR_MUON_STREAM
        ).cast[dtype]()
    else:
        params_ptr[unsafe_offset=idx] = param.cast[dtype]()
    if has_master:
        master_ptr[unsafe_offset=idx] = param


# ===----------------------------------------------------------------------=== #
# GPU kernels
# ===----------------------------------------------------------------------=== #


def muon_zero_gpu(ptr: MutKernelPtr[DType.float32], n_arg: Int64):
    var i = Int(block_idx.x * 256 + thread_idx.x)
    if i < Int(n_arg):
        ptr[unsafe_offset=i] = Float32(0)


def muon_momentum_gpu[
    dtype: DType
](
    grads_ptr: ImmutKernelPtr[dtype],
    m_ptr: MutKernelPtr[DType.float32],
    x_ptr: MutKernelPtr[DType.bfloat16],
    norms_ptr: MutKernelPtr[DType.float32],
    n_arg: Int64,
    stride_arg: Int64,
    grad_scale: Scalar[DType.float32],
    mu: Scalar[DType.float32],
    nesterov: UInt8,
) -> None:
    """Grid (ceil(n / CHUNK), count). M = mu*M + g; X = bf16(U); accumulates
    sum(X^2) per matrix into norms (pre-zeroed) with one atomic per block."""
    var n = Int(n_arg)
    var mat = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var base = Int(block_idx.x) * CHUNK
    var gp = grads_ptr.unsafe_offset(mat * Int(stride_arg))
    var mp = m_ptr.unsafe_offset(mat * Int(stride_arg))
    var xp = x_ptr.unsafe_offset(mat * n)
    var acc = Float32(0)
    for j in range(ELEMS):
        var i = base + j * BLOCK + tid
        if i < n:
            var g = grad_scale * gp[unsafe_offset=i].cast[DType.float32]()
            var mm = mu * mp[unsafe_offset=i] + g
            mp[unsafe_offset=i] = mm
            var u = g + mu * mm if nesterov != 0 else mm
            var xb = u.cast[DType.bfloat16]()
            xp[unsafe_offset=i] = xb
            var xf = xb.cast[DType.float32]()
            acc += xf * xf
    var s = block.sum[block_size=BLOCK](acc)
    if tid == 0:
        _ = Atomic[Float32].fetch_add(norms_ptr.unsafe_offset(mat), s)


def muon_normalize_gpu(
    x_ptr: MutKernelPtr[DType.bfloat16],
    norms_ptr: MutKernelPtr[DType.float32],
    n_arg: Int64,
) -> None:
    """X /= (||X||_F + 1e-7), per matrix (grid = (ceil(n / CHUNK), count))."""
    var n = Int(n_arg)
    var mat = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var base = Int(block_idx.x) * CHUNK
    var xp = x_ptr.unsafe_offset(mat * n)
    var inv = Float32(1) / (sqrt(norms_ptr[unsafe_offset=mat]) + NS_EPS)
    for j in range(ELEMS):
        var i = base + j * BLOCK + tid
        if i < n:
            xp[unsafe_offset=i] = (
                xp[unsafe_offset=i].cast[DType.float32]() * inv
            ).cast[DType.bfloat16]()


def muon_axpby_gpu(
    y_ptr: MutKernelPtr[DType.bfloat16],
    x_ptr: MutKernelPtr[DType.bfloat16],
    n_arg: Int64,
    alpha: Scalar[DType.float32],
    beta: Scalar[DType.float32],
) -> None:
    """Flat y = bf16(alpha * y + beta * x)."""
    var i = Int(block_idx.x * 256 + thread_idx.x)
    if i < Int(n_arg):
        var yv = y_ptr[unsafe_offset=i].cast[DType.float32]()
        var xv = x_ptr[unsafe_offset=i].cast[DType.float32]()
        y_ptr[unsafe_offset=i] = (alpha * yv + beta * xv).cast[DType.bfloat16]()


def muon_apply_gpu[
    dtype: DType
](
    params_ptr: MutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    o_ptr: MutKernelPtr[DType.bfloat16],
    has_master: UInt8,
    n_arg: Int64,
    stride_arg: Int64,
    offset_arg: Int64,
    lr: Scalar[DType.float32],
    wd: Scalar[DType.float32],
    scale: Scalar[DType.float32],
    t: UInt32,
) -> None:
    """W update from the orthogonalized O (grid = (ceil(n / CHUNK), count)).
    params_ptr / master_ptr already point at the group's first element."""
    var n = Int(n_arg)
    var mat = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var base = Int(block_idx.x) * CHUNK
    var pp = params_ptr.unsafe_offset(mat * Int(stride_arg))
    var mp = master_ptr.unsafe_offset(mat * Int(stride_arg))
    var op = o_ptr.unsafe_offset(mat * n)
    for j in range(ELEMS):
        var i = base + j * BLOCK + tid
        if i < n:
            _muon_store[dtype](
                i,
                Int(offset_arg) + mat * Int(stride_arg) + i,
                pp,
                mp,
                has_master != 0,
                op[unsafe_offset=i].cast[DType.float32](),
                lr,
                wd,
                scale,
                t,
            )


@always_inline
def _imm_bf16(
    p: MutKernelPtr[DType.bfloat16],
) -> ImmutKernelPtr[DType.bfloat16]:
    return rebind[ImmutKernelPtr[DType.bfloat16]](
        p.as_imm().as_unsafe_any_origin()
    )


def _gemm_bf16(
    d: MutKernelPtr[DType.bfloat16],
    a: MutKernelPtr[DType.bfloat16],
    b: MutKernelPtr[DType.bfloat16],
    m: Int,
    n: Int,
    k: Int,
    trans_a: Bool,
    trans_b: Bool,
    count: Int,
    stride_a: Int,
    stride_b: Int,
    stride_d: Int,
    ctx: DeviceContext,
) raises:
    """Batched column-major D[m,n] = op(A) op(B), bf16 in/out, fp32 accum."""
    comptime bf = DType.bfloat16
    if trans_a:
        _matmul_cublaslt[bf, True, False](
            d,
            _imm_bf16(a),
            _imm_bf16(b),
            _imm_bf16(a),
            d,
            m,
            n,
            k,
            Int32(1),
            False,
            ctx,
            batch_count=count,
            stride_a=stride_a,
            stride_b=stride_b,
            stride_d=stride_d,
        )
    elif trans_b:
        _matmul_cublaslt[bf, False, True](
            d,
            _imm_bf16(a),
            _imm_bf16(b),
            _imm_bf16(a),
            d,
            m,
            n,
            k,
            Int32(1),
            False,
            ctx,
            batch_count=count,
            stride_a=stride_a,
            stride_b=stride_b,
            stride_d=stride_d,
        )
    else:
        _matmul_cublaslt[bf, False, False](
            d,
            _imm_bf16(a),
            _imm_bf16(b),
            _imm_bf16(a),
            d,
            m,
            n,
            k,
            Int32(1),
            False,
            ctx,
            batch_count=count,
            stride_a=stride_a,
            stride_b=stride_b,
            stride_d=stride_d,
        )


def _muon_update_gpu[
    dtype: DType,
](
    groups: List[MuonGroup],
    params_ptr: MutKernelPtr[dtype],
    grads_ptr: ImmutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    has_master: Bool,
    m_ptr: MutKernelPtr[DType.float32],
    t: UInt32,
    config: MuonConfig,
    ctx: DeviceContext,
) raises:
    var max_x = 1
    var max_g = 1
    var max_count = 1
    for g in groups:
        var r = min(g.rows, g.cols)
        max_x = max(max_x, g.count * g.rows * g.cols)
        max_g = max(max_g, g.count * r * r)
        max_count = max(max_count, g.count)
    # Names carry the size: persistent_device_buffer fixes the allocation at
    # the first call, and distinct group sets (tests, trainer) need different
    # sizes. A production run has one group set, hence one allocation.
    var x = persistent_device_buffer[DType.bfloat16](
        ctx, "muon_x_" + String(max_x), max_x
    )
    var y = persistent_device_buffer[DType.bfloat16](
        ctx, "muon_y_" + String(max_x), max_x
    )
    var a = persistent_device_buffer[DType.bfloat16](
        ctx, "muon_a_" + String(max_g), max_g
    )
    var a2 = persistent_device_buffer[DType.bfloat16](
        ctx, "muon_a2_" + String(max_g), max_g
    )
    var norms = persistent_device_buffer[DType.float32](
        ctx, "muon_norms_" + String(max_count), max_count
    )

    var k_zero = ctx.compile_function[muon_zero_gpu]()
    var k_mom = ctx.compile_function[muon_momentum_gpu[dtype]]()
    var k_norm = ctx.compile_function[muon_normalize_gpu]()
    var k_axpby = ctx.compile_function[muon_axpby_gpu]()
    var k_apply = ctx.compile_function[muon_apply_gpu[dtype]]()

    for g in groups:
        var rows = g.rows
        var cols = g.cols
        var count = g.count
        var n = rows * cols
        var tall = rows > cols
        var R = min(rows, cols)
        var C = max(rows, cols)
        var gs = R * R
        var chunks = ceildiv(n, CHUNK)

        ctx.enqueue_function(
            k_zero,
            norms,
            Int64(count),
            grid_dim=(ceildiv(count, 256),),
            block_dim=(256,),
        )
        ctx.enqueue_function(
            k_mom,
            grads_ptr.unsafe_offset(g.offset),
            m_ptr.unsafe_offset(g.offset),
            x,
            norms,
            Int64(n),
            Int64(g.stride),
            config.grad_scale,
            config.momentum,
            UInt8(1) if config.nesterov else UInt8(0),
            grid_dim=(chunks, count),
            block_dim=(BLOCK,),
        )
        ctx.enqueue_function(
            k_norm,
            x,
            norms,
            Int64(n),
            grid_dim=(chunks, count),
            block_dim=(BLOCK,),
        )

        for _ in range(config.ns_steps):
            # A = X X^T  (R x R). Wide/square: memory is the col-major C x R
            # matrix X^T -> A = op(T)(X^T) * X^T. Tall: memory is the
            # col-major R x C matrix X' (= X^T) -> A = X' * op(T)(X').
            _gemm_bf16(a, x, x, R, R, C, not tall, tall, count, n, n, gs, ctx)
            _gemm_bf16(a2, a, a, R, R, R, False, False, count, gs, gs, gs, ctx)
            # A = b*A + c*A2 (this is the "B" polynomial matrix).
            ctx.enqueue_function(
                k_axpby,
                a,
                a2,
                Int64(count * gs),
                NS_B,
                NS_C,
                grid_dim=(ceildiv(count * gs, 256),),
                block_dim=(256,),
            )
            # Y = B X. Wide/square: Xc * B (C x R); tall: B * Xc' (R x C).
            if tall:
                _gemm_bf16(y, a, x, R, C, R, False, False, count, gs, n, n, ctx)
            else:
                _gemm_bf16(y, x, a, C, R, R, False, False, count, n, gs, n, ctx)
            # X = a*X + Y
            ctx.enqueue_function(
                k_axpby,
                x,
                y,
                Int64(count * n),
                NS_A,
                Float32(1),
                grid_dim=(ceildiv(count * n, 256),),
                block_dim=(256,),
            )

        var scale = Float32(0.2) * sqrt(Float32(C))
        ctx.enqueue_function(
            k_apply,
            params_ptr.unsafe_offset(g.offset),
            master_ptr.unsafe_offset(g.offset),
            x,
            UInt8(1) if has_master else UInt8(0),
            Int64(n),
            Int64(g.stride),
            Int64(g.offset),
            config.learning_rate,
            config.weight_decay,
            scale,
            t,
            grid_dim=(chunks, count),
            block_dim=(BLOCK,),
        )


# ===----------------------------------------------------------------------=== #
# CPU reference (naive fp32; correctness only)
# ===----------------------------------------------------------------------=== #


def muon_ns_host(
    u: List[Float32], rows: Int, cols: Int, ns_steps: Int
) -> List[Float32]:
    """Host fp32 reference of the whole orthogonalization: Frobenius
    normalize, optional transpose, `ns_steps` quintic Newton-Schulz
    iterations, transpose back. `u` is row-major [rows, cols]; returns O in
    the same layout."""
    var tall = rows > cols
    var R = min(rows, cols)
    var C = max(rows, cols)
    var x = List[Float32](length=R * C, fill=Float32(0))
    var ss = Float32(0)
    for i in range(rows):
        for j in range(cols):
            var v = u[i * cols + j]
            ss += v * v
            if tall:
                x[j * C + i] = v
            else:
                x[i * C + j] = v
    var inv = Float32(1) / (sqrt(ss) + NS_EPS)
    for i in range(R * C):
        x[i] *= inv
    var a = List[Float32](length=R * R, fill=Float32(0))
    var a2 = List[Float32](length=R * R, fill=Float32(0))
    var y = List[Float32](length=R * C, fill=Float32(0))
    for _ in range(ns_steps):
        for i in range(R):
            for j in range(R):
                var s = Float32(0)
                for k in range(C):
                    s += x[i * C + k] * x[j * C + k]
                a[i * R + j] = s
        for i in range(R):
            for j in range(R):
                var s = Float32(0)
                for k in range(R):
                    s += a[i * R + k] * a[k * R + j]
                a2[i * R + j] = s
        for i in range(R * R):
            a[i] = NS_B * a[i] + NS_C * a2[i]
        for i in range(R):
            for j in range(C):
                var s = Float32(0)
                for k in range(R):
                    s += a[i * R + k] * x[k * C + j]
                y[i * C + j] = s
        for i in range(R * C):
            x[i] = NS_A * x[i] + y[i]
    var out = List[Float32](length=rows * cols, fill=Float32(0))
    for i in range(rows):
        for j in range(cols):
            out[i * cols + j] = x[j * C + i] if tall else x[i * C + j]
    return out^


def _muon_update_cpu[
    dtype: DType,
](
    groups: List[MuonGroup],
    params_ptr: MutKernelPtr[dtype],
    grads_ptr: ImmutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    has_master: Bool,
    m_ptr: MutKernelPtr[DType.float32],
    t: UInt32,
    config: MuonConfig,
) raises:
    for g in groups:
        var n = g.rows * g.cols
        var C = max(g.rows, g.cols)
        var scale = Float32(0.2) * sqrt(Float32(C))
        for mat in range(g.count):
            var base = g.offset + mat * g.stride
            var u = List[Float32](length=n, fill=Float32(0))
            for i in range(n):
                var gr = (
                    config.grad_scale
                    * grads_ptr[unsafe_offset=base + i].cast[DType.float32]()
                )
                var mm = config.momentum * m_ptr[unsafe_offset=base + i] + gr
                m_ptr[unsafe_offset=base + i] = mm
                u[i] = (gr + config.momentum * mm) if config.nesterov else mm
            var o = muon_ns_host(u, g.rows, g.cols, config.ns_steps)
            for i in range(n):
                _muon_store[dtype](
                    base + i,
                    base + i,
                    params_ptr,
                    master_ptr,
                    has_master,
                    o[i],
                    config.learning_rate,
                    config.weight_decay,
                    scale,
                    t,
                )


def muon_update[
    dtype: DType,
    target: StaticString,
](
    groups: List[MuonGroup],
    params_ptr: MutKernelPtr[dtype],
    grads_ptr: ImmutKernelPtr[dtype],
    # fp32 master shadow of `params` (same flat indexing); unused when
    # `has_master` is False (pure fp32 training), exactly as llmm.adamw.
    master_ptr: MutKernelPtr[DType.float32],
    has_master: Bool,
    # fp32 momentum, same flat indexing as params (the trainer's m_memory).
    m_ptr: MutKernelPtr[DType.float32],
    t: UInt32,
    config: MuonConfig,
    ctx: DeviceContext,
) raises:
    """One Muon step over every matrix of every group. Scratch (bf16 X/Y/A/A2
    buffers and per-matrix norms) is allocated once per process through
    llmm.memory.persistent_device_buffer, so callers hold no scratch state.
    """
    comptime if is_cpu[target]():
        _muon_update_cpu[dtype](
            groups,
            params_ptr,
            grads_ptr,
            master_ptr,
            has_master,
            m_ptr,
            t,
            config,
        )
    elif is_gpu[target]():
        comptime if HAS_CUBLAS:
            _muon_update_gpu[dtype](
                groups,
                params_ptr,
                grads_ptr,
                master_ptr,
                has_master,
                m_ptr,
                t,
                config,
                ctx,
            )
        else:
            raise Error("muon_update: GPU path needs cuBLASLt (NVIDIA)")
    else:
        raise Error("Invalid target")


# ===----------------------------------------------------------------------=== #
# Compiler registration (pytest bridge)
# ===----------------------------------------------------------------------=== #


@register("muon_update")
struct MuonUpdate:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        params: MutableInputTensor[dtype=dtype, rank=2, static_spec=...],
        m: MutableInputTensor[dtype=DType.float32, rank=2, static_spec=...],
        t: UInt32,
        grads: InputTensor[dtype=dtype, rank=2, static_spec=...],
        learning_rate: Scalar[dtype],
        momentum: Scalar[dtype],
        weight_decay: Scalar[dtype],
        grad_scale: Scalar[dtype],
        ctx: DeviceContext,
    ) capturing raises:
        var rows = Int(params.shape()[0])
        var cols = Int(params.shape()[1])
        if Int(grads.shape()[0]) != rows or Int(grads.shape()[1]) != cols:
            raise Error("params and grads must have the same shape")
        if Int(m.shape()[0]) != rows or Int(m.shape()[1]) != cols:
            raise Error("m and params must have the same shape")
        var config = MuonConfig(
            learning_rate=Scalar[DType.float32](learning_rate),
            momentum=Scalar[DType.float32](momentum),
            weight_decay=Scalar[DType.float32](weight_decay),
            grad_scale=Scalar[DType.float32](grad_scale),
            ns_steps=5,
            nesterov=True,
        )
        var groups = List[MuonGroup]()
        groups.append(
            MuonGroup(
                offset=0, rows=rows, cols=cols, count=1, stride=rows * cols
            )
        )
        # has_master=False: params are their own master; m's pointer is the
        # unused placeholder (MAX forbids null pointers), as in adamw_update.
        muon_update[dtype, target](
            groups,
            params.unsafe_ptr(),
            grads.unsafe_ptr(),
            m.unsafe_ptr(),
            False,
            m.unsafe_ptr(),
            t,
            config,
            ctx,
        )
