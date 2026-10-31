# ===----------------------------------------------------------------------=== #
# QK-norm and fixed attention scale (modded-nanogpt record 5 / record 20).
#
# `qk_rows_fwd` / `qk_rows_bwd` are the ONE self-contained "apply an op to every
# q and k row" kernel pair (W7). They run on the head-layout q and k buffers
# [B, NH, T, hd] that `attention_fwd` produces by splitting the qkv activation,
# and on the matching gradient buffers d_q / d_k before they are merged back
# into d_qkv. Row r of either buffer is (b * NH + h) * T + t, so a later op that
# needs the token position (rotary, W8) has t = r % T available here.
#
#   forward   q <- q_mul * rmsnorm(q)        k <- rmsnorm(k)        (do_norm)
#             q <- q_mul * q                 k unchanged            (not do_norm)
#   rmsnorm(x) = x * rsqrt(mean(x^2) + QK_NORM_EPS) over head_dim, no affine.
#
# q_mul = attn_scale * sqrt(head_dim). The attention kernels keep applying their
# built-in 1 / sqrt(head_dim) to q.k, so q_mul * q . k / sqrt(hd) = attn_scale *
# q.k exactly: the fixed scale needs no change inside any attention kernel
# (forward, backward, CPU, Metal, the QK-clip max-logit hook all stay correct).
#
# Backward needs the pre-norm row, which stays in acts.qkv (the split copies it),
# so no rstd buffer is stored: rstd and the unit-RMS row are recomputed from it
# with the identical fp32 arithmetic of the forward.
#   dy = q_mul * d_q'           dx = rstd * (dy - u * mean(dy * u)),  u = x * rstd
# ===----------------------------------------------------------------------=== #

from std.math import rsqrt
from extensibility import register, InputTensor
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from max.gpu import block_dim, block_idx, thread_idx, WARP_SIZE
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_gpu
from max.gpu.primitives import warp

from llmm.memory import ImmutKernelPtr, MutKernelPtr
from llmm.head_layout import null_immut_ptr
from llmm.rotary import rotate_y1, rotate_y2

comptime QK_NORM_EPS = Float32(1e-6)
comptime _ROWS_PER_BLOCK = 4  # one warp per row, 128 threads per block


@always_inline
def _row_sumsq[
    dtype: DType
](
    x: ImmutKernelPtr[dtype], base: Int, hd: Int, lane: Int, lanes: Int
) -> Scalar[DType.float32]:
    var ss = Scalar[DType.float32](0)
    for d in range(lane, hd, lanes):
        var v = x[unsafe_offset=base + d].cast[DType.float32]()
        ss += v * v
    return ss


@always_inline
def _qkv_row_base(r: Int, is_k: Int, NH: Int, T: Int, hd: Int) -> Int:
    # Offset of head-layout row r's pre-norm data inside the token-layout qkv
    # activation [B, T, 3C]: section 0 = q, 1 = k.
    var C = NH * hd
    var t = r % T
    var h = (r // T) % NH
    var b = r // (T * NH)
    return (b * T + t) * 3 * C + is_k * C + h * hd


def qk_rows_fwd_gpu[
    dtype: DType, do_norm: Bool, do_rot: Bool
](
    q_ptr: MutKernelPtr[dtype],
    k_ptr: MutKernelPtr[dtype],
    table: ImmutKernelPtr[DType.float32],
    rows: Int64,
    hd_arg: Int64,
    seq_len: Int64,
    table_rows: Int64,
    q_mul: Scalar[DType.float32],
) -> None:
    var hd = Int(hd_arg)
    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var g = Int(block_idx.x) * _ROWS_PER_BLOCK + warp_id  # [0, 2 * rows)
    var n = Int(rows)
    if g >= 2 * n:
        return
    var is_k = g >= n
    var r = g - n if is_k else g
    var base = r * hd
    var ptr = k_ptr if is_k else q_ptr
    var mul = Scalar[DType.float32](1) if is_k else q_mul
    comptime if do_norm:
        var ss = Scalar[DType.float32](0)
        for d in range(lane, hd, WARP_SIZE):
            var v = ptr[unsafe_offset=base + d].cast[DType.float32]()
            ss += v * v
        ss = warp.sum(ss)
        mul = mul * rsqrt(ss / Scalar[DType.float32](hd) + QK_NORM_EPS)
    comptime if do_rot:
        # norm (x mul) first, then the rotary pair (i, i + hd/2), i < hd/4;
        # the other pairs are the identity. One read and one write per element.
        var half = hd // 2
        var nr = hd // 4
        var t = r % Int(seq_len)
        var rt = Int(table_rows)
        for i in range(lane, half, WARP_SIZE):
            var x1 = ptr[unsafe_offset=base + i].cast[DType.float32]() * mul
            var x2 = (
                ptr[unsafe_offset=base + i + half].cast[DType.float32]() * mul
            )
            if i < nr:
                var c = table[unsafe_offset=t * nr + i]
                var sn = table[unsafe_offset=rt * nr + t * nr + i]
                var y1 = rotate_y1[False](x1, x2, c, sn)
                var y2 = rotate_y2[False](x1, x2, c, sn)
                x1 = y1
                x2 = y2
            ptr[unsafe_offset=base + i] = x1.cast[dtype]()
            ptr[unsafe_offset=base + i + half] = x2.cast[dtype]()
    else:
        for d in range(lane, hd, WARP_SIZE):
            ptr[unsafe_offset=base + d] = (
                ptr[unsafe_offset=base + d].cast[DType.float32]() * mul
            ).cast[dtype]()


@always_inline
def _bwd_grad_pair[
    dtype: DType, do_rot: Bool
](
    dptr: MutKernelPtr[dtype],
    table: ImmutKernelPtr[DType.float32],
    base: Int,
    i: Int,
    half: Int,
    nr: Int,
    tc: Int,
    ts: Int,
) -> Tuple[Float32, Float32]:
    # Gradient w.r.t. the pre-rotation (post-norm) pair: the transpose rotation
    # of the stored gradient w.r.t. the rotated output; plain load if no rotary
    # or if the pair is in the identity half.
    var g1 = dptr[unsafe_offset=base + i].cast[DType.float32]()
    var g2 = dptr[unsafe_offset=base + i + half].cast[DType.float32]()
    comptime if do_rot:
        if i < nr:
            var c = table[unsafe_offset=tc]
            var sn = table[unsafe_offset=ts]
            var h1 = rotate_y1[True](g1, g2, c, sn)
            var h2 = rotate_y2[True](g1, g2, c, sn)
            g1 = h1
            g2 = h2
    return (g1, g2)


def qk_rows_bwd_gpu[
    dtype: DType, do_norm: Bool, do_rot: Bool
](
    dq_ptr: MutKernelPtr[dtype],
    dk_ptr: MutKernelPtr[dtype],
    qkv_ptr: ImmutKernelPtr[dtype],
    table: ImmutKernelPtr[DType.float32],
    batch: Int64,
    num_heads: Int64,
    seq_len: Int64,
    hd_arg: Int64,
    table_rows: Int64,
    q_mul: Scalar[DType.float32],
) -> None:
    var hd = Int(hd_arg)
    var NH = Int(num_heads)
    var T = Int(seq_len)
    var n = Int(batch) * NH * T
    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var g = Int(block_idx.x) * _ROWS_PER_BLOCK + warp_id
    if g >= 2 * n:
        return
    var is_k = g >= n
    var r = g - n if is_k else g
    var base = r * hd
    var dptr = dk_ptr if is_k else dq_ptr
    var mul = Scalar[DType.float32](1) if is_k else q_mul
    comptime if do_rot:
        # Fused: inverse rotation of the incoming gradient, then (do_norm) the
        # norm backward against the PRE-norm, PRE-rotation row in acts.qkv.
        var half = hd // 2
        var nr = hd // 4
        var t = r % T
        var rt = Int(table_rows)
        comptime if do_norm:
            var xb = _qkv_row_base(r, 1 if is_k else 0, NH, T, hd)
            var ss = warp.sum(
                _row_sumsq[dtype](qkv_ptr, xb, hd, lane, WARP_SIZE)
            )
            var rstd = rsqrt(ss / Scalar[DType.float32](hd) + QK_NORM_EPS)
            var dot = Scalar[DType.float32](0)
            for i in range(lane, half, WARP_SIZE):
                var gp = _bwd_grad_pair[dtype, True](
                    dptr,
                    table,
                    base,
                    i,
                    half,
                    nr,
                    t * nr + i,
                    rt * nr + t * nr + i,
                )
                var u1 = (
                    qkv_ptr[unsafe_offset=xb + i].cast[DType.float32]() * rstd
                )
                var u2 = (
                    qkv_ptr[unsafe_offset=xb + i + half].cast[DType.float32]()
                    * rstd
                )
                dot += gp[0] * mul * u1 + gp[1] * mul * u2
            dot = warp.sum(dot) / Scalar[DType.float32](hd)
            for i in range(lane, half, WARP_SIZE):
                var gp = _bwd_grad_pair[dtype, True](
                    dptr,
                    table,
                    base,
                    i,
                    half,
                    nr,
                    t * nr + i,
                    rt * nr + t * nr + i,
                )
                var u1 = (
                    qkv_ptr[unsafe_offset=xb + i].cast[DType.float32]() * rstd
                )
                var u2 = (
                    qkv_ptr[unsafe_offset=xb + i + half].cast[DType.float32]()
                    * rstd
                )
                var dy1 = gp[0] * mul
                var dy2 = gp[1] * mul
                dptr[unsafe_offset=base + i] = (rstd * (dy1 - u1 * dot)).cast[
                    dtype
                ]()
                dptr[unsafe_offset=base + i + half] = (
                    rstd * (dy2 - u2 * dot)
                ).cast[dtype]()
        else:
            for i in range(lane, half, WARP_SIZE):
                var gp = _bwd_grad_pair[dtype, True](
                    dptr,
                    table,
                    base,
                    i,
                    half,
                    nr,
                    t * nr + i,
                    rt * nr + t * nr + i,
                )
                dptr[unsafe_offset=base + i] = (gp[0] * mul).cast[dtype]()
                dptr[unsafe_offset=base + i + half] = (gp[1] * mul).cast[
                    dtype
                ]()
    else:
        comptime if do_norm:
            var xb = _qkv_row_base(r, 1 if is_k else 0, NH, T, hd)
            var ss = warp.sum(
                _row_sumsq[dtype](qkv_ptr, xb, hd, lane, WARP_SIZE)
            )
            var rstd = rsqrt(ss / Scalar[DType.float32](hd) + QK_NORM_EPS)
            var dot = Scalar[DType.float32](0)
            for d in range(lane, hd, WARP_SIZE):
                var u = (
                    qkv_ptr[unsafe_offset=xb + d].cast[DType.float32]() * rstd
                )
                dot += (
                    dptr[unsafe_offset=base + d].cast[DType.float32]() * mul * u
                )
            dot = warp.sum(dot) / Scalar[DType.float32](hd)
            for d in range(lane, hd, WARP_SIZE):
                var u = (
                    qkv_ptr[unsafe_offset=xb + d].cast[DType.float32]() * rstd
                )
                var dy = (
                    dptr[unsafe_offset=base + d].cast[DType.float32]() * mul
                )
                dptr[unsafe_offset=base + d] = (rstd * (dy - u * dot)).cast[
                    dtype
                ]()
        else:
            if not is_k:
                for d in range(lane, hd, WARP_SIZE):
                    dptr[unsafe_offset=base + d] = (
                        dptr[unsafe_offset=base + d].cast[DType.float32]() * mul
                    ).cast[dtype]()


def qk_rows_fwd[
    dtype: DType, target: StaticString, do_norm: Bool, do_rot: Bool = False
](
    q_ptr: MutKernelPtr[dtype],
    k_ptr: MutKernelPtr[dtype],
    batch: Int,
    num_heads: Int,
    seq_len: Int,
    head_dim: Int,
    q_mul: Float32,
    ctx: DeviceContext,
    table: ImmutKernelPtr[DType.float32] = null_immut_ptr[DType.float32](),
    table_rows: Int = 0,
) raises:
    """q <- q_mul * rmsnorm(q) (do_norm) then rotary (do_rot), k likewise.

    With `do_rot` the rotary table (`rotary_table_host`, `table_rows` = its
    T_max >= seq_len) is applied after the norm in the same pass.
    """
    var rows = batch * num_heads * seq_len
    comptime if do_rot:
        if head_dim % 4 != 0:
            raise Error("rotary requires head_dim % 4 == 0")
        if seq_len > table_rows:
            raise Error("rotary table shorter than the sequence length")
    comptime if is_cpu[target]():
        var nr = head_dim // 4
        var half = head_dim // 2
        for g in range(2 * rows):
            var is_k = g >= rows
            var r = g - rows if is_k else g
            var base = r * head_dim
            var ptr = k_ptr if is_k else q_ptr
            var mul = Float32(1) if is_k else q_mul
            comptime if do_norm:
                var ss = Float32(0)
                for d in range(head_dim):
                    var v = ptr[unsafe_offset=base + d].cast[DType.float32]()
                    ss += v * v
                mul = mul * rsqrt(ss / Float32(head_dim) + QK_NORM_EPS)
            comptime if do_rot:
                var t = r % seq_len
                for i in range(half):
                    var x1 = (
                        ptr[unsafe_offset=base + i].cast[DType.float32]() * mul
                    )
                    var x2 = (
                        ptr[unsafe_offset=base + i + half].cast[DType.float32]()
                        * mul
                    )
                    if i < nr:
                        var c = table[unsafe_offset=t * nr + i]
                        var sn = table[
                            unsafe_offset=table_rows * nr + t * nr + i
                        ]
                        var y1 = rotate_y1[False](x1, x2, c, sn)
                        var y2 = rotate_y2[False](x1, x2, c, sn)
                        x1 = y1
                        x2 = y2
                    ptr[unsafe_offset=base + i] = x1.cast[dtype]()
                    ptr[unsafe_offset=base + i + half] = x2.cast[dtype]()
            else:
                for d in range(head_dim):
                    ptr[unsafe_offset=base + d] = (
                        ptr[unsafe_offset=base + d].cast[DType.float32]() * mul
                    ).cast[dtype]()
    elif is_gpu[target]():
        comptime gpu_kernel = qk_rows_fwd_gpu[dtype, do_norm, do_rot]
        var compiled = ctx.compile_function[gpu_kernel]()
        ctx.enqueue_function(
            compiled,
            q_ptr,
            k_ptr,
            table,
            Int64(rows),
            Int64(head_dim),
            Int64(seq_len),
            Int64(table_rows),
            q_mul,
            grid_dim=((2 * rows + _ROWS_PER_BLOCK - 1) // _ROWS_PER_BLOCK,),
            block_dim=(_ROWS_PER_BLOCK * WARP_SIZE,),
        )
    else:
        raise Error("Invalid target")


def qk_rows_bwd[
    dtype: DType, target: StaticString, do_norm: Bool, do_rot: Bool = False
](
    dq_ptr: MutKernelPtr[dtype],
    dk_ptr: MutKernelPtr[dtype],
    qkv_ptr: ImmutKernelPtr[dtype],
    batch: Int,
    num_heads: Int,
    seq_len: Int,
    head_dim: Int,
    q_mul: Float32,
    ctx: DeviceContext,
    table: ImmutKernelPtr[DType.float32] = null_immut_ptr[DType.float32](),
    table_rows: Int = 0,
) raises:
    """Backward of `qk_rows_fwd`: with `do_rot` the inverse rotation of the
    incoming gradient comes first, then the norm backward against the raw
    (pre-norm, pre-rotation) row in `qkv_ptr`."""
    var rows = batch * num_heads * seq_len
    comptime if do_rot:
        if head_dim % 4 != 0:
            raise Error("rotary requires head_dim % 4 == 0")
        if seq_len > table_rows:
            raise Error("rotary table shorter than the sequence length")
    comptime if is_cpu[target]():
        var nr = head_dim // 4
        var half = head_dim // 2
        for g in range(2 * rows):
            var is_k = g >= rows
            var r = g - rows if is_k else g
            var base = r * head_dim
            var dptr = dk_ptr if is_k else dq_ptr
            var mul = Float32(1) if is_k else q_mul
            var t = r % seq_len
            # Gradient w.r.t. the pre-rotation row (inverse rotation applied
            # first when do_rot), kept in fp32 scratch.
            var gbuf = List[Float32](length=head_dim, fill=Float32(0))
            for i in range(half):
                var g1 = dptr[unsafe_offset=base + i].cast[DType.float32]()
                var g2 = dptr[unsafe_offset=base + i + half].cast[
                    DType.float32
                ]()
                comptime if do_rot:
                    if i < nr:
                        var c = table[unsafe_offset=t * nr + i]
                        var sn = table[
                            unsafe_offset=table_rows * nr + t * nr + i
                        ]
                        var h1 = rotate_y1[True](g1, g2, c, sn)
                        var h2 = rotate_y2[True](g1, g2, c, sn)
                        g1 = h1
                        g2 = h2
                gbuf[i] = g1
                gbuf[i + half] = g2
            comptime if do_norm:
                var xb = _qkv_row_base(
                    r, 1 if is_k else 0, num_heads, seq_len, head_dim
                )
                var ss = _row_sumsq[dtype](qkv_ptr, xb, head_dim, 0, 1)
                var rstd = rsqrt(ss / Float32(head_dim) + QK_NORM_EPS)
                var dot = Float32(0)
                for d in range(head_dim):
                    var u = (
                        qkv_ptr[unsafe_offset=xb + d].cast[DType.float32]()
                        * rstd
                    )
                    dot += gbuf[d] * mul * u
                dot = dot / Float32(head_dim)
                for d in range(head_dim):
                    var u = (
                        qkv_ptr[unsafe_offset=xb + d].cast[DType.float32]()
                        * rstd
                    )
                    var dy = gbuf[d] * mul
                    dptr[unsafe_offset=base + d] = (rstd * (dy - u * dot)).cast[
                        dtype
                    ]()
            else:
                comptime if do_rot:
                    for d in range(head_dim):
                        dptr[unsafe_offset=base + d] = (gbuf[d] * mul).cast[
                            dtype
                        ]()
                else:
                    if not is_k:
                        for d in range(head_dim):
                            dptr[unsafe_offset=base + d] = (gbuf[d] * mul).cast[
                                dtype
                            ]()
    elif is_gpu[target]():
        comptime gpu_kernel = qk_rows_bwd_gpu[dtype, do_norm, do_rot]
        var compiled = ctx.compile_function[gpu_kernel]()
        ctx.enqueue_function(
            compiled,
            dq_ptr,
            dk_ptr,
            qkv_ptr,
            table,
            Int64(batch),
            Int64(num_heads),
            Int64(seq_len),
            Int64(head_dim),
            Int64(table_rows),
            q_mul,
            grid_dim=((2 * rows + _ROWS_PER_BLOCK - 1) // _ROWS_PER_BLOCK,),
            block_dim=(_ROWS_PER_BLOCK * WARP_SIZE,),
        )
    else:
        raise Error("Invalid target")


# ===----------------------------------------------------------------------=== #
# MAX custom ops (test bridge: tests/kernels/qk_norm.py)
# ===----------------------------------------------------------------------=== #


@register("qk_rows_fwd")
struct QkRowsFwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        do_norm: Bool = True,
    ](
        q: MutableInputTensor[dtype=dtype, rank=4, static_spec=...],
        k: MutableInputTensor[dtype=dtype, rank=4, static_spec=...],
        batch_size: Int64,
        num_heads: Int64,
        seq_len: Int64,
        head_dim: Int64,
        q_mul: Float32,
        ctx: DeviceContext,
    ) capturing raises:
        qk_rows_fwd[dtype, target, do_norm](
            q.unsafe_ptr(),
            k.unsafe_ptr(),
            Int(batch_size),
            Int(num_heads),
            Int(seq_len),
            Int(head_dim),
            q_mul,
            ctx,
        )


@register("qk_rows_bwd")
struct QkRowsBwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        do_norm: Bool = True,
    ](
        dq: MutableInputTensor[dtype=dtype, rank=4, static_spec=...],
        dk: MutableInputTensor[dtype=dtype, rank=4, static_spec=...],
        qkv: InputTensor[dtype=dtype, rank=3, static_spec=...],
        batch_size: Int64,
        num_heads: Int64,
        seq_len: Int64,
        head_dim: Int64,
        q_mul: Float32,
        ctx: DeviceContext,
    ) capturing raises:
        qk_rows_bwd[dtype, target, do_norm](
            dq.unsafe_ptr(),
            dk.unsafe_ptr(),
            qkv.unsafe_ptr(),
            Int(batch_size),
            Int(num_heads),
            Int(seq_len),
            Int(head_dim),
            q_mul,
            ctx,
        )
