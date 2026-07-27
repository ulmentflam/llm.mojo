from extensibility import register
from extensibility import InputTensor
from std.gpu.host import DeviceContext
from std.math import ceildiv, exp, log
from std.gpu.host import DeviceAttribute
from std.sys import simd_width_of, align_of
from std.gpu.host.info import is_cpu, is_gpu
from extensibility.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from std.sys._assembly import inlined_assembly
from std.runtime.asyncrt import parallelism_level
from std.algorithm import vectorize, sync_parallelize
from std.gpu import barrier, block_idx, grid_dim, thread_idx
from std.gpu.primitives import block

from llmm.vendor import HAS_CUBLAS
from llmm.profiler import traced_parallelize
from llmm.memory import ImmutKernelPtr, MutKernelPtr
from llmm.softmax import softmax_phase_1_and_2_cpu, softmax_phase_1_and_2_gpu

# ===----------------------------------------------------------------------=== #
# Constants and Comptime Variables
# ===----------------------------------------------------------------------=== #

comptime UNROLL = 4
comptime LOG2_E = Scalar[DType.float32](1.4426950408889634)


# ===----------------------------------------------------------------------=== #
# Fast Exponential Approximations
# ===----------------------------------------------------------------------=== #


@always_inline
def _fast_exp2[
    width: Int
](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    # Hardware ex2.approx.f32 (what CUDA's __expf/__exp2f compile to) via inline
    # PTX — Mojo's exp/exp2 are accurate polynomials (~10× the ops). Per-lane.
    var out = SIMD[DType.float32, width](0)

    comptime for i in range(width):
        out[i] = inlined_assembly[
            "ex2.approx.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x[i])
    return out


@always_inline
def _fast_exp[
    width: Int
](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    return _fast_exp2(x * LOG2_E)


@always_inline
def _classifier_exp[
    width: Int
](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    # NVIDIA (HAS_CUBLAS=True): hardware ex2.approx.f32 via inline PTX, the
    # same approximation __expf uses in CUDA — about 10× fewer ops than the
    # accurate polynomial and within ~1 ULP for softmax inputs.
    # Metal / portable (HAS_CUBLAS=False): accurate `exp` polynomial, already
    # used on the scalar ragged edge and the CPU path. Mathematically identical
    # (exp2(x*LOG2_E) == exp(x)); slightly slower but fully portable.
    # The inline PTX assembly in _fast_exp2 does NOT compile on Metal —
    # HAS_CUBLAS must be False to exclude it (guaranteed by vendor.mojo).
    comptime if HAS_CUBLAS:
        return _fast_exp(x)
    else:
        return exp(x)


# ===----------------------------------------------------------------------=== #
# Fused Classifier Forward and Backward
# ===----------------------------------------------------------------------=== #


@always_inline
def _fused_classifier_cpu[
    dtype: DType, width: Int, write_d_logits: Bool = True
](
    idx: Int,
    logits_ptr: MutKernelPtr[dtype],
    losses_ptr: MutKernelPtr[DType.float32],
    d_losses_ptr: ImmutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    vocab_size: Int,  # Our V
    vocab_size_padded: Int,  # Our Vp, padding is garbage
) -> None:
    var base = idx * vocab_size_padded
    var target_idx = Int(targets_ptr[idx])

    var stats = softmax_phase_1_and_2_cpu[dtype, width](
        idx, logits_ptr, vocab_size, vocab_size_padded
    )
    var m_row = stats[0]
    var s_row = stats[1]

    # NOTE: Loss in log-softmax form, and BEFORE the in-place overwrite below.
    var x_t = logits_ptr[base + target_idx].cast[DType.float32]()
    losses_ptr[idx] = log(s_row) + m_row - x_t

    comptime if write_d_logits:
        var d_loss = d_losses_ptr[idx]
        var inv_s = Scalar[DType.float32](1) / s_row

        @always_inline
        def _d_logits[
            w: Int,
        ](local: Int) {logits_ptr, m_row, inv_s, d_loss, base,}:
            # Indicator-free body: the one-hot touches exactly one element
            # of the row, fixed up scalarly after the loop. The in-place
            # wide load then store to the same location is safe
            # sequentially.
            var p_idx = base + local
            var x = (logits_ptr + p_idx).load[width=w]().cast[DType.float32]()
            var p = exp(x - SIMD[DType.float32, w](m_row)) * SIMD[
                DType.float32, w
            ](inv_s)
            (logits_ptr + p_idx).store[width=w](
                (p * SIMD[DType.float32, w](d_loss)).cast[dtype]()
            )

        vectorize[width, unroll_factor=UNROLL](vocab_size, _d_logits)

        # x_t is saved above, so we can compute the one-hot fix op here.
        var p_t = exp(x_t - m_row) * inv_s
        logits_ptr[base + target_idx] = ((p_t - 1.0) * d_loss).cast[dtype]()

        # Zero the padded tail so the backward matmul reads zeros.
        for i in range(vocab_size, vocab_size_padded):
            logits_ptr[base + i] = Scalar[dtype](0)


def fused_classifier_cpu[
    dtype: DType,
    width: Int,
    write_d_logits: Bool = True,
](
    logits_ptr: MutKernelPtr[dtype],
    losses_ptr: MutKernelPtr[DType.float32],
    d_losses_ptr: ImmutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size: Int64,  # Our V
    vocab_size_padded: Int64,  # Our Vp
) raises -> None:
    var total = Int(batch_size * seq_len)
    var max_workers = parallelism_level()
    var rows_per_worker = ceildiv(total, max_workers)
    var num_workers = ceildiv(total, rows_per_worker)

    @parameter
    def _worker(w: Int):
        var base = w * rows_per_worker
        var count = min(rows_per_worker, total - base)
        for local in range(count):
            var idx = base + local
            _fused_classifier_cpu[dtype, width, write_d_logits](
                idx,
                logits_ptr,
                losses_ptr,
                d_losses_ptr,
                targets_ptr,
                Int(vocab_size),
                Int(vocab_size_padded),
            )

    traced_parallelize["fused_classifier", _worker](num_workers)


@always_inline
def _fused_classifier_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
    write_d_logits: Bool = True,
](
    num_rows: Int,
    tid: Int,
    stride: Int,
    block_row: Int,
    logits_ptr: MutKernelPtr[dtype],
    losses_ptr: MutKernelPtr[DType.float32],
    d_losses_ptr: ImmutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    vocab_size: Int,
    vocab_size_padded: Int,
) -> None:
    comptime BLOCK_SPAN = BLOCK_SIZE * width

    # Grid strided over the rows, same as the softmax kernels. The phase
    # 1+2 helper and the barrier below synchronize the whole block, so
    # every thread must take the same trip.
    for row in range(block_row, num_rows, stride):
        var base = row * vocab_size_padded
        var target_idx = Int(targets_ptr[row])

        var stats = softmax_phase_1_and_2_gpu[dtype, BLOCK_SIZE, width](
            row, tid, logits_ptr, vocab_size, vocab_size_padded
        )
        var m_row = stats[0]
        var s_row = stats[1]

        # Loss in log-softmax form, single thread. NOTE: assigns, where
        # llm.c accumulates with -=; this matches crossentropy_ohe_fwd.
        if tid == 0:
            var x_t = logits_ptr[base + target_idx].cast[DType.float32]()
            losses_ptr[row] = log(s_row) + m_row - x_t

        comptime if write_d_logits:
            # barrier() matches __syncthreads() in Karpathy's kernel
            barrier()

            var d_loss = d_losses_ptr[row]
            var inv_s = Scalar[DType.float32](1) / s_row
            var m_vec = SIMD[DType.float32, width](m_row)
            var inv_s_vec = SIMD[DType.float32, width](inv_s)
            var d_loss_vec = SIMD[DType.float32, width](d_loss)

            comptime align = align_of[SIMD[dtype, width]]()
            for tile_base in range(0, vocab_size, BLOCK_SPAN):
                var lane_base = tile_base + tid * width
                if lane_base + width <= vocab_size:
                    var x = (
                        (logits_ptr + base + lane_base)
                        .load[width=width, alignment=align]()
                        .cast[DType.float32]()
                    )
                    var p = _classifier_exp(x - m_vec) * inv_s_vec
                    var d = p * d_loss_vec
                    if (
                        lane_base <= target_idx
                        and target_idx < lane_base + width
                    ):
                        var k = target_idx - lane_base
                        d[k] = d[k] - d_loss
                    (logits_ptr + base + lane_base).store[
                        width=width, alignment=align
                    ](d.cast[dtype]())
                elif lane_base < vocab_size:
                    # Ragged edge of the last tile: scalar steps, same
                    # uniform-trip-count rule as the softmax kernels.
                    for i in range(lane_base, vocab_size):
                        var x = logits_ptr[base + i].cast[DType.float32]()
                        var p = exp(x - m_row) * inv_s
                        var ind = Scalar[DType.float32](
                            1.0
                        ) if i == target_idx else Scalar[DType.float32](0.0)
                        logits_ptr[base + i] = ((p - ind) * d_loss).cast[
                            dtype
                        ]()

            # Zero the padded tail so the backward matmul reads zeros.
            for i in range(vocab_size + tid, vocab_size_padded, BLOCK_SIZE):
                logits_ptr[base + i] = Scalar[dtype](0)


def fused_classifier_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
    width: Int = 4,
    write_d_logits: Bool = True,
](
    logits_ptr: MutKernelPtr[dtype],
    losses_ptr: MutKernelPtr[DType.float32],
    d_losses_ptr: ImmutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size: Int64,  # Our V
    vocab_size_padded: Int64,  # Our Vp
) -> None:
    _fused_classifier_gpu[dtype, BLOCK_SIZE, width, write_d_logits](
        Int(batch_size * seq_len),
        Int(thread_idx.x),
        Int(grid_dim.x),
        Int(block_idx.x),
        logits_ptr,
        losses_ptr,
        d_losses_ptr,
        targets_ptr,
        Int(vocab_size),
        Int(vocab_size_padded),
    )


def fused_classifier[
    dtype: DType,
    target: StaticString,
    write_d_logits: Bool = True,
](
    logits_ptr: MutKernelPtr[dtype],
    losses_ptr: MutKernelPtr[DType.float32],
    d_losses_ptr: ImmutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    batch_size: Int64,  # Our B
    seq_len: Int64,  # Our T
    vocab_size: Int64,  # Our V
    vocab_size_padded: Int64,  # Our Vp
    ctx: DeviceContext,
) capturing raises:
    comptime if is_cpu[target]():
        comptime simd_width = simd_width_of[dtype]()
        fused_classifier_cpu[dtype, simd_width, write_d_logits](
            logits_ptr,
            losses_ptr,
            d_losses_ptr,
            targets_ptr,
            batch_size,
            seq_len,
            vocab_size,
            vocab_size_padded,
        )
    elif is_gpu[target]():
        # Duplicated gpu dispatch code from the softmax ops. 1024 threads/block
        # (matching llm.c's fused_classifier_kernel5) — one block reduces the full
        # V=50k row, so more threads = a faster per-row softmax + gradient pass.
        comptime BLOCK_SIZE = 1024
        comptime SM_OVERPROVISION = 32
        var device_ctx = ctx
        var num_rows = Int(batch_size * seq_len)
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var num_blocks = max(min(num_rows, SM_OVERPROVISION * num_sm), 1)

        # 8-wide (128-bit) vectorized loads/stores like llm.c's x128.
        comptime gpu_kernel = fused_classifier_gpu[
            dtype, BLOCK_SIZE, 8, write_d_logits=write_d_logits
        ]
        var compiled = device_ctx.compile_function[gpu_kernel]()
        device_ctx.enqueue_function(
            compiled,
            logits_ptr,
            losses_ptr,
            d_losses_ptr,
            targets_ptr,
            batch_size,
            seq_len,
            vocab_size,
            vocab_size_padded,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


# ===----------------------------------------------------------------------=== #
# Chunked (two-pass) cross-entropy
#
# `fused_classifier` above needs the WHOLE `(B*T, V_p)` logits matrix resident:
# it reads a full row to get that row's max and sum-exp, then overwrites the
# same row in place with dlogits. At B=32, T=1024, V_p=50304, fp32 that buffer
# is 6.1 GiB — the largest single activation in the trainer after `att_probs`.
#
# The three entry points below let a caller that already walks the LM head in
# vocabulary tiles (see `matmul_lm_head_fwd_tile`) do the same job while never
# holding more than ONE tile of logits, `(B*T, tile_oc)`:
#
#   pass 1  `chunked_ce_pass1`  — fold one tile into the running per-row
#                                 (max, sum-exp) and capture the target logit.
#   loss    `chunked_ce_loss`   — losses[row] = log(s) + m - x_target, once the
#                                 running pair has seen every tile.
#   pass 2  `chunked_ce_pass2`  — given the finished (m, s), turn a RECOMPUTED
#                                 tile of logits into that tile's dlogits, in
#                                 place, so the caller can fold it straight into
#                                 its per-tile d_weight / d_input GEMMs.
#
# WHY TWO PASSES ARE NUMERICALLY SAFE. Softmax is shift-invariant: subtracting
# any constant from a row leaves the probabilities unchanged. The standard
# stable form subtracts the row max, which needs the whole row. The online
# recurrence carries a partial max `m` and the sum-exp `s` measured against it,
# and rescales when a later tile raises the max:
#
#     m' = max(m, m_tile)
#     s' = s * exp(m - m') + s_tile * exp(m_tile - m')
#
# Both exponents are <= 0, so every factor is in (0, 1] — the rescale can only
# shrink, never overflow, and the result is algebraically identical to a
# single-pass reduction. This is the same recurrence `llmm/softmax.mojo` already
# runs across SIMD lanes and thread blocks; here it runs across vocab tiles.
#
# PRIOR ART. Chunking the classifier so the full logits tensor never
# materializes is not novel: it is Liger Kernel's FusedLinearCrossEntropy
# (arXiv:2410.10989 §3.2), which chunks the hidden states, projects each chunk,
# and computes a partial loss per chunk. Cut Your Losses (arXiv:2411.09009)
# attacks the same memory problem by a different mechanism (an on-the-fly
# log-sum-exp in the kernel). The tiling axis here is the vocabulary rather
# than the batch, because that is the axis the LM-head weight tiling already
# uses.
#
# COST. Pass 2 must RECOMPUTE each logits tile — retaining them is exactly what
# we are refusing to do — so a step pays one extra full LM-head forward GEMM.
# ===----------------------------------------------------------------------=== #


# ---------------------------------------------------------------------------
# Pass 1 — fold one vocab tile into the running (max, sum-exp)
# ---------------------------------------------------------------------------


@always_inline
def _chunk_merge(
    m_run: Scalar[DType.float32],
    s_run: Scalar[DType.float32],
    m_tile: Scalar[DType.float32],
    s_tile: Scalar[DType.float32],
) -> Tuple[Scalar[DType.float32], Scalar[DType.float32]]:
    """Online-softmax merge of a running (max, sum-exp) pair with one tile's
    pair. Both `exp` arguments are <= 0 by construction, so neither factor can
    overflow; an empty tile arrives as (MIN_FINITE, 0) and is absorbed as a
    no-op."""
    var m_new = max(m_run, m_tile)
    var s_new = s_run * exp(m_run - m_new) + s_tile * exp(m_tile - m_new)
    return (m_new, s_new)


def chunked_ce_pass1_cpu[
    dtype: DType,
](
    tile_ptr: ImmutKernelPtr[dtype],
    m_ptr: MutKernelPtr[DType.float32],
    s_ptr: MutKernelPtr[DType.float32],
    xt_ptr: MutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    num_rows: Int,
    tile_ld: Int,
    tile_start: Int,
    valid_cols: Int,
    first: Bool,
) raises -> None:
    var max_workers = parallelism_level()
    var rows_per_worker = ceildiv(num_rows, max_workers)
    var num_workers = ceildiv(num_rows, rows_per_worker)

    @parameter
    def _worker(w: Int):
        var lo = w * rows_per_worker
        var count = min(rows_per_worker, num_rows - lo)
        for local in range(count):
            var row = lo + local
            # Scalar. NOT `softmax_phase_1_and_2_cpu`, tempting as that is:
            # its wide loads are issued with SIMD-width alignment, which holds
            # for a full row of the [rows, V_p] logits matrix (V_p is a multiple
            # of the width in every real config) but NOT for a tile, whose
            # leading dimension is an arbitrary runtime value. An aligned load
            # off a misaligned base faults.
            var base = row * tile_ld
            var m_tile = Scalar[DType.float32].MIN_FINITE
            var s_tile = Scalar[DType.float32](0)
            for c in range(valid_cols):
                var x = tile_ptr[base + c].cast[DType.float32]()
                var nm = max(m_tile, x)
                s_tile = s_tile * exp(m_tile - nm) + exp(x - nm)
                m_tile = nm
            if first:
                m_ptr[row] = m_tile
                s_ptr[row] = s_tile
                xt_ptr[row] = Scalar[DType.float32](0)
            else:
                var merged = _chunk_merge(
                    m_ptr[row], s_ptr[row], m_tile, s_tile
                )
                m_ptr[row] = merged[0]
                s_ptr[row] = merged[1]
            var t = Int(targets_ptr[row])
            if t >= tile_start and t < tile_start + valid_cols:
                xt_ptr[row] = tile_ptr[row * tile_ld + (t - tile_start)].cast[
                    DType.float32
                ]()

    traced_parallelize["chunked_ce_pass1", _worker](num_workers)


def chunked_ce_pass1_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
](
    tile_ptr: ImmutKernelPtr[dtype],
    m_ptr: MutKernelPtr[DType.float32],
    s_ptr: MutKernelPtr[DType.float32],
    xt_ptr: MutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    num_rows: Int,
    tile_ld: Int,
    tile_start: Int,
    valid_cols: Int,
    first: Int,
) -> None:
    var tid = Int(thread_idx.x)
    # One block per row, grid-strided. `block.max`/`block.sum` synchronize the
    # whole block, so every thread must take the same number of trips — the
    # grid stride guarantees that.
    for row in range(Int(block_idx.x), num_rows, Int(grid_dim.x)):
        var base = row * tile_ld
        # Scalar, block-strided (hence coalesced) reads. Deliberately NOT the
        # 128-bit vectorized form used on the full-row path: a tile's leading
        # dimension is a runtime value with no alignment guarantee.
        var m_t = Scalar[DType.float32].MIN_FINITE
        var s_t = Scalar[DType.float32](0)
        for c in range(tid, valid_cols, BLOCK_SIZE):
            var x = tile_ptr[base + c].cast[DType.float32]()
            var nm = max(m_t, x)
            s_t = s_t * exp(m_t - nm) + exp(x - nm)
            m_t = nm

        var m_tile = block.max[block_size=BLOCK_SIZE](m_t)
        s_t = s_t * exp(m_t - m_tile)
        var s_tile = block.sum[block_size=BLOCK_SIZE](s_t)

        if tid == 0:
            if first != 0:
                m_ptr[row] = m_tile
                s_ptr[row] = s_tile
                xt_ptr[row] = Scalar[DType.float32](0)
            else:
                var merged = _chunk_merge(
                    m_ptr[row], s_ptr[row], m_tile, s_tile
                )
                m_ptr[row] = merged[0]
                s_ptr[row] = merged[1]
            var t = Int(targets_ptr[row])
            if t >= tile_start and t < tile_start + valid_cols:
                xt_ptr[row] = tile_ptr[base + (t - tile_start)].cast[
                    DType.float32
                ]()


def chunked_ce_pass1[
    dtype: DType,
    target: StaticString,
](
    tile_ptr: ImmutKernelPtr[dtype],
    m_ptr: MutKernelPtr[DType.float32],
    s_ptr: MutKernelPtr[DType.float32],
    xt_ptr: MutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    num_rows: Int,
    tile_ld: Int,
    tile_start: Int,
    valid_cols: Int,
    first: Bool,
    ctx: DeviceContext,
) capturing raises:
    """Fold vocab tile `[tile_start, tile_start + valid_cols)` — held
    contiguously as `[num_rows, tile_ld]` — into the running per-row max
    `m_ptr` and sum-exp `s_ptr`, and capture the target column's logit into
    `xt_ptr` if it falls inside this tile.

    `valid_cols` is the count of REAL vocabulary columns in the tile
    (`min(tile_ld, V - tile_start)`, clamped at 0). Columns past it are the
    `[V, V_p)` padding and never enter the reduction. `first` must be True on
    the first tile only; it seeds the running pair instead of merging."""
    comptime if is_cpu[target]():
        chunked_ce_pass1_cpu[dtype](
            tile_ptr,
            m_ptr,
            s_ptr,
            xt_ptr,
            targets_ptr,
            num_rows,
            tile_ld,
            tile_start,
            valid_cols,
            first,
        )
    elif is_gpu[target]():
        comptime BLOCK_SIZE = 256
        comptime SM_OVERPROVISION = 32
        var device_ctx = ctx
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var num_blocks = max(min(num_rows, SM_OVERPROVISION * num_sm), 1)
        comptime k = chunked_ce_pass1_gpu[dtype, BLOCK_SIZE]
        var compiled = device_ctx.compile_function[k]()
        device_ctx.enqueue_function(
            compiled,
            tile_ptr,
            m_ptr,
            s_ptr,
            xt_ptr,
            targets_ptr,
            num_rows,
            tile_ld,
            tile_start,
            valid_cols,
            1 if first else 0,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


# ---------------------------------------------------------------------------
# Loss epilogue — the same log-softmax form the single-pass classifier uses
# ---------------------------------------------------------------------------


def chunked_ce_loss_gpu[
    BLOCK_SIZE: Int,
](
    losses_ptr: MutKernelPtr[DType.float32],
    m_ptr: ImmutKernelPtr[DType.float32],
    s_ptr: ImmutKernelPtr[DType.float32],
    xt_ptr: ImmutKernelPtr[DType.float32],
    num_rows: Int,
) -> None:
    var idx = Int(block_idx.x) * BLOCK_SIZE + Int(thread_idx.x)
    if idx < num_rows:
        losses_ptr[idx] = log(s_ptr[idx]) + m_ptr[idx] - xt_ptr[idx]


def chunked_ce_loss[
    target: StaticString,
](
    losses_ptr: MutKernelPtr[DType.float32],
    m_ptr: ImmutKernelPtr[DType.float32],
    s_ptr: ImmutKernelPtr[DType.float32],
    xt_ptr: ImmutKernelPtr[DType.float32],
    num_rows: Int,
    ctx: DeviceContext,
) capturing raises:
    """`losses[row] = log(s) + m - x_target`, bit-for-bit the expression
    `_fused_classifier_*` evaluates once it has the row's (max, sum-exp)."""
    comptime if is_cpu[target]():
        for i in range(num_rows):
            losses_ptr[i] = log(s_ptr[i]) + m_ptr[i] - xt_ptr[i]
    elif is_gpu[target]():
        comptime BLOCK_SIZE = 256
        var device_ctx = ctx
        comptime k = chunked_ce_loss_gpu[BLOCK_SIZE]
        var compiled = device_ctx.compile_function[k]()
        device_ctx.enqueue_function(
            compiled,
            losses_ptr,
            m_ptr,
            s_ptr,
            xt_ptr,
            num_rows,
            grid_dim=(ceildiv(num_rows, BLOCK_SIZE),),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


# ---------------------------------------------------------------------------
# Pass 2 — one recomputed logits tile, in place, becomes that tile's dlogits
# ---------------------------------------------------------------------------


def chunked_ce_pass2_cpu[
    dtype: DType,
](
    tile_ptr: MutKernelPtr[dtype],
    m_ptr: ImmutKernelPtr[DType.float32],
    s_ptr: ImmutKernelPtr[DType.float32],
    d_losses_ptr: ImmutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    num_rows: Int,
    tile_ld: Int,
    tile_start: Int,
    valid_cols: Int,
) raises -> None:
    var max_workers = parallelism_level()
    var rows_per_worker = ceildiv(num_rows, max_workers)
    var num_workers = ceildiv(num_rows, rows_per_worker)

    @parameter
    def _worker(w: Int):
        var lo = w * rows_per_worker
        var count = min(rows_per_worker, num_rows - lo)
        for local in range(count):
            var row = lo + local
            var base = row * tile_ld
            var m_row = m_ptr[row]
            var inv_s = Scalar[DType.float32](1) / s_ptr[row]
            var d_loss = d_losses_ptr[row]
            var tgt = Int(targets_ptr[row])
            for c in range(valid_cols):
                var x = tile_ptr[base + c].cast[DType.float32]()
                var p = exp(x - m_row) * inv_s
                var ind = Scalar[DType.float32](
                    1.0
                ) if tile_start + c == tgt else Scalar[DType.float32](0.0)
                tile_ptr[base + c] = ((p - ind) * d_loss).cast[dtype]()
            # Padding columns [V, V_p) must read as exactly zero in the
            # backward GEMMs, same guarantee the single-pass classifier gives.
            for c in range(valid_cols, tile_ld):
                tile_ptr[base + c] = Scalar[dtype](0)

    traced_parallelize["chunked_ce_pass2", _worker](num_workers)


def chunked_ce_pass2_gpu[
    dtype: DType,
    BLOCK_SIZE: Int,
](
    tile_ptr: MutKernelPtr[dtype],
    m_ptr: ImmutKernelPtr[DType.float32],
    s_ptr: ImmutKernelPtr[DType.float32],
    d_losses_ptr: ImmutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    num_rows: Int,
    tile_ld: Int,
    tile_start: Int,
    valid_cols: Int,
) -> None:
    var tid = Int(thread_idx.x)
    for row in range(Int(block_idx.x), num_rows, Int(grid_dim.x)):
        var base = row * tile_ld
        var m_row = m_ptr[row]
        var inv_s = Scalar[DType.float32](1) / s_ptr[row]
        var d_loss = d_losses_ptr[row]
        var tgt = Int(targets_ptr[row])
        for c in range(tid, tile_ld, BLOCK_SIZE):
            if c < valid_cols:
                var x = tile_ptr[base + c].cast[DType.float32]()
                var p = _classifier_exp[1](x - m_row) * inv_s
                var d = p * d_loss
                if tile_start + c == tgt:
                    d = d - d_loss
                tile_ptr[base + c] = d.cast[dtype]()
            else:
                # Padding columns [V, V_p) — exactly zero, as above.
                tile_ptr[base + c] = Scalar[dtype](0)


def chunked_ce_pass2[
    dtype: DType,
    target: StaticString,
](
    tile_ptr: MutKernelPtr[dtype],
    m_ptr: ImmutKernelPtr[DType.float32],
    s_ptr: ImmutKernelPtr[DType.float32],
    d_losses_ptr: ImmutKernelPtr[DType.float32],
    targets_ptr: ImmutKernelPtr[DType.int32],
    num_rows: Int,
    tile_ld: Int,
    tile_start: Int,
    valid_cols: Int,
    ctx: DeviceContext,
) capturing raises:
    """Overwrite one RECOMPUTED logits tile with that tile's dlogits, using the
    finished per-row (max, sum-exp) from pass 1.

    `dlogits = (softmax - onehot) * dloss`, evaluated column by column — the
    indicator only ever fires inside the tile that contains the target. Columns
    in `[valid_cols, tile_ld)` are the `[V, V_p)` padding and are forced to
    exactly zero, so the caller's d_weight / d_input GEMMs read zeros there."""
    comptime if is_cpu[target]():
        chunked_ce_pass2_cpu[dtype](
            tile_ptr,
            m_ptr,
            s_ptr,
            d_losses_ptr,
            targets_ptr,
            num_rows,
            tile_ld,
            tile_start,
            valid_cols,
        )
    elif is_gpu[target]():
        comptime BLOCK_SIZE = 256
        comptime SM_OVERPROVISION = 32
        var device_ctx = ctx
        var num_sm = device_ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var num_blocks = max(min(num_rows, SM_OVERPROVISION * num_sm), 1)
        comptime k = chunked_ce_pass2_gpu[dtype, BLOCK_SIZE]
        var compiled = device_ctx.compile_function[k]()
        device_ctx.enqueue_function(
            compiled,
            tile_ptr,
            m_ptr,
            s_ptr,
            d_losses_ptr,
            targets_ptr,
            num_rows,
            tile_ld,
            tile_start,
            valid_cols,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK_SIZE,),
        )
    else:
        raise Error("Invalid target")


# ===----------------------------------------------------------------------=== #
# Fused Classifier Compiler Registration
# ===----------------------------------------------------------------------=== #


@register("fused_classifier")
struct FusedClassifier:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        logits: MutableInputTensor[dtype=dtype, rank=1, static_spec=...],
        losses: MutableInputTensor[
            dtype=DType.float32, rank=1, static_spec=...
        ],
        d_losses: InputTensor[dtype=DType.float32, rank=1, static_spec=...],
        targets: InputTensor[dtype=DType.int32, rank=1, static_spec=...],
        batch_size: Int64,  # Our B
        seq_len: Int64,  # Our T
        vocab_size: Int64,  # Our V
        vocab_size_padded: Int64,  # Our Vp
        ctx: DeviceContext,
    ) capturing raises:
        if vocab_size > vocab_size_padded:
            raise Error("vocab_size must not exceed vocab_size_padded")
        if logits.size() != Int(batch_size * seq_len * vocab_size_padded):
            raise Error(
                "logits must have the same size as batch_size * seq_len *"
                " vocab_size_padded"
            )
        if losses.size() != Int(batch_size * seq_len):
            raise Error(
                "losses must have the same size as batch_size * seq_len"
            )
        if d_losses.size() != Int(batch_size * seq_len):
            raise Error(
                "d_losses must have the same size as batch_size * seq_len"
            )
        if targets.size() != Int(batch_size * seq_len):
            raise Error(
                "targets must have the same size as batch_size * seq_len"
            )
        fused_classifier[dtype, target, True](
            logits.unsafe_ptr(),
            losses.unsafe_ptr(),
            d_losses.unsafe_ptr(),
            targets.unsafe_ptr(),
            batch_size,
            seq_len,
            vocab_size,
            vocab_size_padded,
            ctx,
        )


@register("fused_classifier_fwd")
struct FusedClassifierFwd:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        losses: MutableInputTensor[
            dtype=DType.float32, rank=1, static_spec=...
        ],
        logits: InputTensor[dtype=dtype, rank=1, static_spec=...],
        targets: InputTensor[dtype=DType.int32, rank=1, static_spec=...],
        batch_size: Int64,  # Our B
        seq_len: Int64,  # Our T
        vocab_size: Int64,  # Our V
        vocab_size_padded: Int64,  # Our Vp
        ctx: DeviceContext,
    ) capturing raises:
        if vocab_size > vocab_size_padded:
            raise Error("vocab_size must not exceed vocab_size_padded")
        if logits.size() != Int(batch_size * seq_len * vocab_size_padded):
            raise Error(
                "logits must have the same size as batch_size * seq_len *"
                " vocab_size_padded"
            )
        if losses.size() != Int(batch_size * seq_len):
            raise Error(
                "losses must have the same size as batch_size * seq_len"
            )
        if targets.size() != Int(batch_size * seq_len):
            raise Error(
                "targets must have the same size as batch_size * seq_len"
            )
        # The write_d_logits=False instantiation contains no stores to
        # logits (comptime-dead code), so handing the shared kernel
        # signature a mutable-origin pointer is sound. The dangling
        # d_losses sentinel is likewise never dereferenced.
        var null_d_losses = ImmutKernelPtr[DType.float32].unsafe_dangling()
        fused_classifier[dtype, target, False](
            logits.unsafe_ptr(),
            losses.unsafe_ptr(),
            null_d_losses,
            targets.unsafe_ptr(),
            batch_size,
            seq_len,
            vocab_size,
            vocab_size_padded,
            ctx,
        )
