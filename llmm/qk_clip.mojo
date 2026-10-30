# ===----------------------------------------------------------------------=== #
# QK-clip (the "Clip" in MuonClip, Kimi K2 2025): after an optimizer step,
# any attention head whose max pre-softmax logit S_max exceeded `tau` gets its
# Q and K projection rows (and the matching bias entries) rescaled by
# sqrt(tau / S_max), so the next step's logits are bounded by tau. The
# per-head maxima come from the causal softmax kernel in llmm/attention.mojo.
# Design: docs/ai/muon_muonclip_implementation_plan.md, section 1.
#
# ===----------------------------------------------------------------------=== #

from std.math import sqrt
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_gpu

from llmm.memory import ImmutKernelPtr, MutKernelPtr


@always_inline
def _clip_factor(
    head_max_ptr: ImmutKernelPtr[DType.float32],
    layer: Int,
    row: Int,
    num_heads: Int,
    channels: Int,
    tau: Scalar[DType.float32],
) -> Scalar[DType.float32]:
    # `row` in [0, 2C): Q block then K block; head = (row % C) // hd.
    var hd = channels // num_heads
    var h = (row % channels) // hd
    var smax = head_max_ptr[unsafe_offset=layer * num_heads + h]
    if smax > tau:
        return sqrt(tau / smax)
    return Scalar[DType.float32](1.0)


@always_inline
def _scale_elem[
    dtype: DType
](
    idx: Int,
    f: Scalar[DType.float32],
    params_ptr: MutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    has_master: Bool,
):
    if has_master:
        var v = master_ptr[unsafe_offset=idx] * f
        master_ptr[unsafe_offset=idx] = v
        params_ptr[unsafe_offset=idx] = v.cast[dtype]()
    else:
        params_ptr[unsafe_offset=idx] = (
            params_ptr[unsafe_offset=idx].cast[DType.float32]() * f
        ).cast[dtype]()


def qk_clip_gpu[
    dtype: DType
](
    params_ptr: MutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    has_master_flag: UInt8,
    qkv_weight_offset: Int64,
    qkv_bias_offset: Int64,
    head_max_ptr: ImmutKernelPtr[DType.float32],
    num_heads_arg: Int64,
    channels_arg: Int64,
    tau: Scalar[DType.float32],
) -> None:
    # grid = (2C rows, L layers); threads stride over the C columns of a row.
    var C = Int(channels_arg)
    var NH = Int(num_heads_arg)
    var row = Int(block_idx.x)
    var layer = Int(block_idx.y)
    var f = _clip_factor(head_max_ptr, layer, row, NH, C, tau)
    if f == Scalar[DType.float32](1.0):
        return
    var has_master = has_master_flag != 0
    var row_base = Int(qkv_weight_offset) + (layer * 3 * C + row) * C
    var tid = Int(thread_idx.x)
    for c in range(tid, C, Int(block_dim.x)):
        _scale_elem[dtype](row_base + c, f, params_ptr, master_ptr, has_master)
    if tid == 0:
        _scale_elem[dtype](
            Int(qkv_bias_offset) + layer * 3 * C + row,
            f,
            params_ptr,
            master_ptr,
            has_master,
        )


def qk_clip_apply[
    dtype: DType,
    target: StaticString,
](
    params_ptr: MutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    has_master: Bool,
    # Element offsets of the [L, 3C, C] qkv weight and the [L, 3C] qkv bias
    # inside the flat parameter buffer (same indexing for master_ptr).
    qkv_weight_offset: Int,
    qkv_bias_offset: Int,
    # [L, NH] fp32 per-(layer, head) max attention logit of the step just
    # taken; entries <= tau are left untouched.
    head_max_ptr: ImmutKernelPtr[DType.float32],
    num_layers: Int,
    num_heads: Int,
    channels: Int,
    tau: Scalar[DType.float32],
    ctx: DeviceContext,
) raises:
    comptime if is_cpu[target]():
        for l in range(num_layers):
            for row in range(2 * channels):
                var f = _clip_factor(
                    head_max_ptr, l, row, num_heads, channels, tau
                )
                if f == Scalar[DType.float32](1.0):
                    continue
                var row_base = (
                    qkv_weight_offset + (l * 3 * channels + row) * channels
                )
                for c in range(channels):
                    _scale_elem[dtype](
                        row_base + c, f, params_ptr, master_ptr, has_master
                    )
                _scale_elem[dtype](
                    qkv_bias_offset + l * 3 * channels + row,
                    f,
                    params_ptr,
                    master_ptr,
                    has_master,
                )
    elif is_gpu[target]():
        comptime gpu_kernel = qk_clip_gpu[dtype]
        var compiled = ctx.compile_function[gpu_kernel]()
        ctx.enqueue_function(
            compiled,
            params_ptr,
            master_ptr,
            UInt8(1) if has_master else UInt8(0),
            Int64(qkv_weight_offset),
            Int64(qkv_bias_offset),
            head_max_ptr,
            Int64(num_heads),
            Int64(channels),
            tau,
            grid_dim=(2 * channels, num_layers),
            block_dim=(128,),
        )
    else:
        raise Error("Invalid target")


# ---------------------------------------------------------------------------
# Sharded variants (WORLD_SIZE > 1): the Q/K matrices live in the owner's
# owner-space buffers and the qkv bias entries in this rank's flat optimizer
# shard, so the clip is applied per piece instead of on the flat buffer. The
# per-element math is `_scale_elem`, i.e. bit-identical to `qk_clip_apply`.
# ---------------------------------------------------------------------------


def qk_clip_matrix_gpu[
    dtype: DType
](
    params_ptr: MutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    has_master_flag: UInt8,
    layer_arg: Int64,
    head_max_ptr: ImmutKernelPtr[DType.float32],
    num_heads_arg: Int64,
    channels_arg: Int64,
    tau: Scalar[DType.float32],
) -> None:
    var C = Int(channels_arg)
    var row = Int(block_idx.x)
    var f = _clip_factor(
        head_max_ptr, Int(layer_arg), row, Int(num_heads_arg), C, tau
    )
    if f == Scalar[DType.float32](1.0):
        return
    var has_master = has_master_flag != 0
    var tid = Int(thread_idx.x)
    for c in range(tid, C, Int(block_dim.x)):
        _scale_elem[dtype](row * C + c, f, params_ptr, master_ptr, has_master)


def qk_clip_matrix[
    dtype: DType,
    target: StaticString,
](
    # Base of ONE row-major [C, C] Q or K block (row r belongs to head
    # r // (C / NH)); master_ptr is its fp32 master at the same indexing.
    params_ptr: MutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    has_master: Bool,
    layer: Int,
    head_max_ptr: ImmutKernelPtr[DType.float32],
    num_heads: Int,
    channels: Int,
    tau: Scalar[DType.float32],
    ctx: DeviceContext,
) raises:
    """Scale the rows of one owned Q or K block of `layer` whose head's max
    exceeded tau by sqrt(tau / max)."""
    comptime if is_cpu[target]():
        for row in range(channels):
            var f = _clip_factor(
                head_max_ptr, layer, row, num_heads, channels, tau
            )
            if f == Scalar[DType.float32](1.0):
                continue
            for c in range(channels):
                _scale_elem[dtype](
                    row * channels + c, f, params_ptr, master_ptr, has_master
                )
    elif is_gpu[target]():
        comptime gpu_kernel = qk_clip_matrix_gpu[dtype]
        var compiled = ctx.compile_function[gpu_kernel]()
        ctx.enqueue_function(
            compiled,
            params_ptr,
            master_ptr,
            UInt8(1) if has_master else UInt8(0),
            Int64(layer),
            head_max_ptr,
            Int64(num_heads),
            Int64(channels),
            tau,
            grid_dim=(channels,),
            block_dim=(128,),
        )
    else:
        raise Error("Invalid target")


def qk_clip_bias_range_gpu[
    dtype: DType
](
    params_ptr: MutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    has_master_flag: UInt8,
    rel_start_arg: Int64,
    n_arg: Int64,
    head_max_ptr: ImmutKernelPtr[DType.float32],
    num_heads_arg: Int64,
    channels_arg: Int64,
    tau: Scalar[DType.float32],
) -> None:
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= Int(n_arg):
        return
    var C = Int(channels_arg)
    var r = Int(rel_start_arg) + i  # index inside the [L, 3C] bias
    var row = r % (3 * C)
    if row >= 2 * C:
        return
    var f = _clip_factor(
        head_max_ptr, r // (3 * C), row, Int(num_heads_arg), C, tau
    )
    if f == Scalar[DType.float32](1.0):
        return
    _scale_elem[dtype](i, f, params_ptr, master_ptr, has_master_flag != 0)


def qk_clip_bias_range[
    dtype: DType,
    target: StaticString,
](
    # `params_ptr` / `master_ptr` point at the first of `n` consecutive
    # elements of the [L, 3C] qkv bias that this rank's flat optimizer shard
    # holds; `rel_start` is the first one's index inside the whole bias.
    params_ptr: MutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    has_master: Bool,
    rel_start: Int,
    n: Int,
    head_max_ptr: ImmutKernelPtr[DType.float32],
    num_heads: Int,
    channels: Int,
    tau: Scalar[DType.float32],
    ctx: DeviceContext,
) raises:
    """Scale the Q/K bias entries (V entries are skipped) of the slice
    [rel_start, rel_start + n) of the qkv bias whose head exceeded tau."""
    comptime if is_cpu[target]():
        for i in range(n):
            var r = rel_start + i
            var row = r % (3 * channels)
            if row >= 2 * channels:
                continue
            var f = _clip_factor(
                head_max_ptr, r // (3 * channels), row, num_heads, channels, tau
            )
            if f == Scalar[DType.float32](1.0):
                continue
            _scale_elem[dtype](i, f, params_ptr, master_ptr, has_master)
    elif is_gpu[target]():
        comptime BLOCK = 128
        comptime gpu_kernel = qk_clip_bias_range_gpu[dtype]
        var compiled = ctx.compile_function[gpu_kernel]()
        ctx.enqueue_function(
            compiled,
            params_ptr,
            master_ptr,
            UInt8(1) if has_master else UInt8(0),
            Int64(rel_start),
            Int64(n),
            head_max_ptr,
            Int64(num_heads),
            Int64(channels),
            tau,
            grid_dim=((n + BLOCK - 1) // BLOCK,),
            block_dim=(BLOCK,),
        )
    else:
        raise Error("Invalid target")


def qk_head_max_fold_gpu(
    step_ptr: MutKernelPtr[DType.float32],
    fwd_ptr: ImmutKernelPtr[DType.float32],
    n_arg: Int64,
    reset: UInt8,
) -> None:
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < Int(n_arg):
        var v = fwd_ptr[unsafe_offset=i]
        if reset == 0:
            v = max(v, step_ptr[unsafe_offset=i])
        step_ptr[unsafe_offset=i] = v


def qk_head_max_fold[
    target: StaticString,
](
    step_ptr: MutKernelPtr[DType.float32],
    fwd_ptr: ImmutKernelPtr[DType.float32],
    n: Int,
    reset: Bool,
    ctx: DeviceContext,
) raises:
    """`step = fwd` if reset else `step = max(step, fwd)`, elementwise over the
    [L*NH] per-head logit maxima (one forward's into the optimizer step's)."""
    comptime if is_cpu[target]():
        for i in range(n):
            var v = fwd_ptr[unsafe_offset=i]
            if not reset:
                v = max(v, step_ptr[unsafe_offset=i])
            step_ptr[unsafe_offset=i] = v
    elif is_gpu[target]():
        comptime BLOCK = 128
        var compiled = ctx.compile_function[qk_head_max_fold_gpu]()
        ctx.enqueue_function(
            compiled,
            step_ptr,
            fwd_ptr,
            Int64(n),
            UInt8(1) if reset else UInt8(0),
            grid_dim=((n + BLOCK - 1) // BLOCK,),
            block_dim=(BLOCK,),
        )
    else:
        raise Error("Invalid target")
