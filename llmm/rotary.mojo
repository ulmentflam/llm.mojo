# ===----------------------------------------------------------------------=== #
# Rotary position embeddings (modded-nanogpt "half-truncated" form), opt-in via
# -D LLMM_ROTARY=1. Design and gates: docs/ai/modded_nanogpt_adoption_plan_
# 2026-10-30.md (W8) and docs/ai/rotary_embeddings_w8_2026-10-30.md.
#
# The speedrun's Rotary module, per head of size hd (hd % 4 == 0):
#     angular_freq = (1/1024) ** linspace(0, 1, hd/4)      # n = hd/4 values
#     angular_freq = cat([angular_freq, zeros(hd/4)])        # hd/2 values
#     theta[t, i]  = t * angular_freq[i]
#     x1, x2 = x[..., :hd/2], x[..., hd/2:]                  # rotate-half pairs
#     y1 = x1 cos + x2 sin;  y2 = -x1 sin + x2 cos
# The second half of `angular_freq` is zero, so pairs i >= hd/4 are the identity
# (cos 1, sin 0) and only the n = hd/4 pairs (i, i + hd/2), i < n, are touched.
#
# Everything here is ONE self-contained entry point that rotates the q and k
# thirds of the [B, T, 3C] qkv activation in place:
#
#     rotary_qk_rows[dtype, target, inverse=False](qkv, table, ...)   # forward
#     rotary_qk_rows[dtype, target, inverse=True](d_qkv, table, ...)  # backward
#
# `inverse=True` applies the transpose (= inverse) rotation, which is exactly
# the gradient of the forward map, so it is run on dq/dk (the q and k thirds of
# d_qkv) right after the attention backward and before the qkv matmul backward.
# The v third is never touched. The table is fp32 [cos | sin], each [T_max, n],
# built once on the host in float64 and rounded, see `rotary_table_host`.
#
# Merging with a per-row q/k norm kernel (W7): the speedrun order is
# norm -> rotary, so a fused kernel would load a (b, t) row's q and k head
# slices, RMS-normalize each head, then apply the rotation pairs below before
# storing; the backward does the inverse rotation on dq/dk then the norm
# backward. The math below is per (row, head, pair) and is a drop-in for the
# store step of a per-row kernel.
# ===----------------------------------------------------------------------=== #

from std.math import cos, sin
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_gpu

from llmm.memory import ImmutKernelPtr, MutKernelPtr
from llmm.profiler import traced_parallelize

comptime ROTARY_BASE = 1024.0
comptime ROTARY_BLOCK = 128


def rotary_pairs(head_dim: Int) -> Int:
    """Number of rotated pairs per head (the non-identity ones): hd / 4."""
    return head_dim // 4


def rotary_table_host(max_seq_len: Int, head_dim: Int) -> List[Float32]:
    """fp32 [cos | sin] table, each [max_seq_len, hd/4] row-major.

    angle[t, i] = t * (1/1024) ** (i / (n - 1)), n = hd/4 (linspace(0, 1, n),
    endpoint included; n == 1 gives frequency 1). Computed in float64 and
    rounded once so the forward and inverse rotations share identical values.
    """
    var n = head_dim // 4
    var table = List[Float32](length=2 * max_seq_len * n, fill=Float32(0.0))
    for t in range(max_seq_len):
        for i in range(n):
            var expo = 0.0
            if n > 1:
                expo = Float64(i) / Float64(n - 1)
            var freq = Float64(1.0) / (Float64(ROTARY_BASE) ** expo)
            var ang = Float64(t) * freq
            table[t * n + i] = Float32(cos(ang))
            table[max_seq_len * n + t * n + i] = Float32(sin(ang))
    return table^


@always_inline
def _rotate_pair[
    dtype: DType, inverse: Bool
](
    qkv: MutKernelPtr[dtype],
    table: ImmutKernelPtr[DType.float32],
    base: Int,  # element offset of x1 (first half of the head slice)
    half: Int,  # hd / 2: distance to the partner x2
    tc: Int,  # table index of cos for (t, i)
    ts: Int,  # table index of sin for (t, i)
):
    var c = table[unsafe_offset=tc]
    var s = table[unsafe_offset=ts]
    var x1 = qkv[unsafe_offset=base].cast[DType.float32]()
    var x2 = qkv[unsafe_offset=base + half].cast[DType.float32]()
    var y1: Float32
    var y2: Float32
    comptime if inverse:
        y1 = x1 * c - x2 * s
        y2 = x1 * s + x2 * c
    else:
        y1 = x1 * c + x2 * s
        y2 = x2 * c - x1 * s
    qkv[unsafe_offset=base] = y1.cast[dtype]()
    qkv[unsafe_offset=base + half] = y2.cast[dtype]()


def rotary_qk_gpu[
    dtype: DType, inverse: Bool
](
    qkv: MutKernelPtr[dtype],
    table: ImmutKernelPtr[DType.float32],
    seq_len_arg: Int64,
    table_rows_arg: Int64,
    num_heads_arg: Int64,
    head_dim_arg: Int64,
):
    # grid = B*T rows (one block per (b, t)); threads stride over the
    # 2 (q,k) * H * n independent pairs of the row.
    var T = Int(seq_len_arg)
    var H = Int(num_heads_arg)
    var hd = Int(head_dim_arg)
    var n = hd // 4
    var C = H * hd
    var bt = Int(block_idx.x)
    var t = bt % T
    var rows = Int(table_rows_arg)
    var row_base = bt * 3 * C
    for j in range(Int(thread_idx.x), 2 * H * n, Int(block_dim.x)):
        var qk = j // (H * n)
        var r = j % (H * n)
        var h = r // n
        var i = r % n
        _rotate_pair[dtype, inverse](
            qkv,
            table,
            row_base + qk * C + h * hd + i,
            hd // 2,
            t * n + i,
            rows * n + t * n + i,
        )


def rotary_qk_rows[
    dtype: DType,
    target: StaticString,
    inverse: Bool = False,
](
    qkv: MutKernelPtr[dtype],
    table: ImmutKernelPtr[DType.float32],
    batch_size: Int,
    seq_len: Int,
    table_rows: Int,
    num_heads: Int,
    head_dim: Int,
    ctx: DeviceContext,
) raises:
    """Rotate (or inverse-rotate) the q and k thirds of [B, T, 3C] qkv rows.

    `table_rows` is the T_max the table was built with (the sin plane starts at
    `table_rows * hd/4`); it must be >= seq_len.
    """
    if head_dim % 4 != 0:
        raise Error("rotary requires head_dim % 4 == 0")
    if seq_len > table_rows:
        raise Error("rotary table shorter than the sequence length")
    var n = head_dim // 4
    var C = num_heads * head_dim
    comptime if is_cpu[target]():
        var total = batch_size * seq_len

        @__parameter
        def _worker(w: Int):
            var bt = w
            var t = bt % seq_len
            var row_base = bt * 3 * C
            for qk in range(2):
                for h in range(num_heads):
                    for i in range(n):
                        _rotate_pair[dtype, inverse](
                            qkv,
                            table,
                            row_base + qk * C + h * head_dim + i,
                            head_dim // 2,
                            t * n + i,
                            table_rows * n + t * n + i,
                        )

        traced_parallelize["rotary_qk", _worker](total)
    elif is_gpu[target]():
        comptime gpu_kernel = rotary_qk_gpu[dtype, inverse]
        var compiled = ctx.compile_function[gpu_kernel]()
        ctx.enqueue_function(
            compiled,
            qkv,
            table,
            Int64(seq_len),
            Int64(table_rows),
            Int64(num_heads),
            Int64(head_dim),
            grid_dim=(batch_size * seq_len,),
            block_dim=(ROTARY_BLOCK,),
        )
    else:
        raise Error("Invalid target")
