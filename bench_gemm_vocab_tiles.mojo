"""What does vocab TILING cost? A GEMM sweep over LM-head tile counts.

The tied-`wte` de-residency work proposes splitting the LM-head projection

    logits[B*T, V_p] = x[B*T, C] @ wte[V_p, C]^T          (transpose_b=True)

into column-blocks, so that only one `[B*T, width]` logits block is resident
at a time instead of the whole `[B*T, V_p]` tensor. That is the memory win.
This file measures the PRICE: one big GEMM becomes several smaller ones, and
smaller GEMMs can fall off peak.

The tile width is NOT an even division of V_p. It comes from the shipped
head's own rule (`lm_head_tile_rows` below, mirroring `_lm_head_tile_rows`):
the width is rounded UP to a multiple of 128, so the final tile is ragged
and the realized tile count can be lower than the knob. Measuring an even
split would time a decomposition the real head never performs.

Methodology is deliberately copied from `bench_gemm.mojo` -- same
`linalg.matmul[transpose_b=True, target="gpu"]` call, same warmup/iters
timing shape, same `2*M*N*K` FLOP count -- so the numbers here are
comparable with that harness rather than being a new methodology.

What is modelled faithfully and what is not:
  * FAITHFUL: the tile widths are the shipped head's own, ragged final tile
    included; and the output block is allocated ONCE at `[M, width]` and
    reused across tiles, which is why logits residency falls with the tile
    count.
  * NOT MODELLED: the softmax/cross-entropy consumption of each block, and
    THE BACKWARD PASS. The shipped head tiles backward too
    (`matmul_lm_head_bwd_tile` handles d_weight and d_input per tile, with
    d_input accumulating across tiles), so a forward-only sweep
    UNDERSTATES the true cost of tiling. Any figure built on this must say
    so on its face.

This is a SYNTHETIC decomposition benchmark. It does not exercise, and
cannot vouch for, any particular vocab-tiled LM-head implementation in the
tree. It does check its own decomposition numerically (`check_tiles` below,
tiled vs untiled at a small M) so that the thing being timed is at least
known to compute the same product.

Run under the shared GPU lock, pinned to one GPU:
  CUDA_VISIBLE_DEVICES=<uuid> flock -w 3600 /tmp/llmm-gpu.lock -c \
      'pixi run -e cuda mojo run -I . bench_gemm_vocab_tiles.mojo'
  (flock, not lockf -- this box is Linux.)

Output is machine-readable lines prefixed `RESULT ` / `CHECK ` for
`scripts/benchmark_vocab_tiles.py` to collect into JSON.
"""

from std.math import ceildiv, sqrt
from std.time import global_perf_counter_ns
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu.host import DeviceContext

from llmm.memory import MutKernelPtr, ImmutKernelPtr


# GPT-2 124M LM-head shape constants.
comptime C_MODEL = 768  # embedding width (the GEMM's K)
comptime V_P = 50304  # padded vocab (the GEMM's full N)


def lm_head_tile_rows(k: Int) -> Int:
    """Tile width for tile-count knob `k`, matching the shipped LM head.

    This mirrors `_lm_head_tile_rows` exactly. It is NOT an even division of
    V_p: the width is rounded UP to a multiple of 128, so the last tile is
    ragged and the REALIZED tile count can be lower than `k`. Measuring an
    even split instead would time a decomposition the real head never
    performs -- a ragged final tile is a differently-shaped GEMM and an
    aligned width may select a different kernel.
    """
    if k <= 1:
        return V_P
    var t = ceildiv(V_P, k)
    comptime if V_P >= 128:
        t = ceildiv(t, 128) * 128
    return min(max(t, 1), V_P)


def linalg_gemm[
    dtype: DType,
](
    c_ptr: MutKernelPtr[dtype],
    a_ptr: ImmutKernelPtr[dtype],
    b_ptr: ImmutKernelPtr[dtype],
    M: Int,
    N: Int,
    K: Int,
    ctx: DeviceContext,
) raises -> None:
    """One `C[M,N] = A[M,K] @ B[N,K]^T`, exactly as `bench_gemm.mojo` calls it.
    """
    var c = TileTensor(
        Span[Scalar[dtype], MutAnyOrigin](unsafe_ptr=c_ptr, length=M * N),
        row_major(M, N),
    )
    var a = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](unsafe_ptr=a_ptr, length=M * K),
        row_major(M, K),
    )
    var b = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](unsafe_ptr=b_ptr, length=N * K),
        row_major(N, K),
    )
    matmul[transpose_b=True, target="gpu"](c, a, b, ctx=ctx)


def _lcg(mut state: UInt64) -> Float32:
    # xorshift64 -> [-0.5, 0.5]; same generator as bench_gemm.mojo.
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    var u = Float32((state >> 40) & 0xFFFFFF) / Float32(0xFFFFFF)
    return u - 0.5


def run_tiles(
    M: Int, tiles: Int, warmup: Int, iters: Int, ctx: DeviceContext
) raises -> None:
    """Time the full LM-head projection split into `tiles` column blocks."""
    comptime K = C_MODEL
    comptime N = V_P
    var tile_n = lm_head_tile_rows(tiles)
    var ntiles = ceildiv(N, tile_n)

    # ---- host operands (fp32 source, cast to bf16 on device) ----
    var a_host = ctx.enqueue_create_host_buffer[DType.float32](M * K)
    var b_host = ctx.enqueue_create_host_buffer[DType.float32](N * K)
    ctx.synchronize()
    var scale = 1.0 / sqrt(Float32(K))
    var st: UInt64 = 0x243F6A8885A308D3 + UInt64(M * 131 + N * 17 + K)
    for i in range(M * K):
        a_host.unsafe_ptr()[unsafe_offset=i] = _lcg(st) * scale
    for i in range(N * K):
        b_host.unsafe_ptr()[unsafe_offset=i] = _lcg(st) * scale

    var a_bf_host = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
    var b_bf_host = ctx.enqueue_create_host_buffer[DType.bfloat16](N * K)
    ctx.synchronize()
    for i in range(M * K):
        a_bf_host.unsafe_ptr()[unsafe_offset=i] = a_host.unsafe_ptr()[unsafe_offset=i].cast[
            DType.bfloat16
        ]()
    for i in range(N * K):
        b_bf_host.unsafe_ptr()[unsafe_offset=i] = b_host.unsafe_ptr()[unsafe_offset=i].cast[
            DType.bfloat16
        ]()

    var a_bf = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var b_bf = ctx.enqueue_create_buffer[DType.bfloat16](N * K)
    # ONE output block, reused across tiles -- this is the memory win.
    var c_bf = ctx.enqueue_create_buffer[DType.bfloat16](M * tile_n)
    a_bf.enqueue_copy_from(a_bf_host)
    b_bf.enqueue_copy_from(b_bf_host)
    ctx.synchronize()

    var a_p = rebind[ImmutKernelPtr[DType.bfloat16]](
        a_bf.unsafe_ptr().as_imm().as_unsafe_any_origin()
    )
    var b_p = rebind[ImmutKernelPtr[DType.bfloat16]](
        b_bf.unsafe_ptr().as_imm().as_unsafe_any_origin()
    )
    var c_p = rebind[MutKernelPtr[DType.bfloat16]](
        c_bf.unsafe_ptr().as_unsafe_any_origin()
    )

    for _ in range(warmup):
        for t in range(ntiles):
            var start = t * tile_n
            var this_n = min(tile_n, N - start)
            linalg_gemm[DType.bfloat16](
                c_p, a_p, b_p.unsafe_offset((start * K)), M, this_n, K, ctx
            )
    ctx.synchronize()

    var t0 = global_perf_counter_ns()
    for _ in range(iters):
        for t in range(ntiles):
            var start = t * tile_n
            var this_n = min(tile_n, N - start)
            linalg_gemm[DType.bfloat16](
                c_p, a_p, b_p.unsafe_offset((start * K)), M, this_n, K, ctx
            )
    ctx.synchronize()
    var ms = Float64(global_perf_counter_ns() - t0) / 1e6 / Float64(iters)

    # Total work is identical for every tile count: 2*M*N*K.
    var flops = 2.0 * Float64(M) * Float64(N) * Float64(K)
    var gflops = flops / (ms * 1e6)

    # Resident logits block, bytes (bf16 = 2 B/elem).
    var logits_bytes = Float64(M) * Float64(tile_n) * 2.0

    print(
        "RESULT m=",
        M,
        " k=",
        K,
        " n=",
        N,
        " tiles=",
        tiles,
        " realized_tiles=",
        ntiles,
        " tile_n=",
        tile_n,
        " ms=",
        ms,
        " gflops=",
        gflops,
        " logits_bytes=",
        logits_bytes,
        " iters=",
        iters,
        " warmup=",
        warmup,
    )


def check_tiles(M: Int, tiles: Int, ctx: DeviceContext) raises -> None:
    """Numerically compare the tiled decomposition against the untiled GEMM.

    Establishes only that the decomposition being timed computes the same
    product as one big GEMM. Says nothing about any LM-head implementation.
    """
    comptime K = C_MODEL
    comptime N = V_P
    var tile_n = lm_head_tile_rows(tiles)
    var ntiles = ceildiv(N, tile_n)

    var a_host = ctx.enqueue_create_host_buffer[DType.float32](M * K)
    var b_host = ctx.enqueue_create_host_buffer[DType.float32](N * K)
    ctx.synchronize()
    var scale = 1.0 / sqrt(Float32(K))
    var st: UInt64 = 0x243F6A8885A308D3 + UInt64(M * 131 + N * 17 + K)
    for i in range(M * K):
        a_host.unsafe_ptr()[unsafe_offset=i] = _lcg(st) * scale
    for i in range(N * K):
        b_host.unsafe_ptr()[unsafe_offset=i] = _lcg(st) * scale

    var a_bf_host = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
    var b_bf_host = ctx.enqueue_create_host_buffer[DType.bfloat16](N * K)
    ctx.synchronize()
    for i in range(M * K):
        a_bf_host.unsafe_ptr()[unsafe_offset=i] = a_host.unsafe_ptr()[unsafe_offset=i].cast[
            DType.bfloat16
        ]()
    for i in range(N * K):
        b_bf_host.unsafe_ptr()[unsafe_offset=i] = b_host.unsafe_ptr()[unsafe_offset=i].cast[
            DType.bfloat16
        ]()

    var a_bf = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var b_bf = ctx.enqueue_create_buffer[DType.bfloat16](N * K)
    var c_full = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    var c_tile = ctx.enqueue_create_buffer[DType.bfloat16](M * tile_n)
    a_bf.enqueue_copy_from(a_bf_host)
    b_bf.enqueue_copy_from(b_bf_host)
    ctx.synchronize()

    var a_p = rebind[ImmutKernelPtr[DType.bfloat16]](
        a_bf.unsafe_ptr().as_imm().as_unsafe_any_origin()
    )
    var b_p = rebind[ImmutKernelPtr[DType.bfloat16]](
        b_bf.unsafe_ptr().as_imm().as_unsafe_any_origin()
    )
    var c_full_p = rebind[MutKernelPtr[DType.bfloat16]](
        c_full.unsafe_ptr().as_unsafe_any_origin()
    )
    var c_tile_p = rebind[MutKernelPtr[DType.bfloat16]](
        c_tile.unsafe_ptr().as_unsafe_any_origin()
    )

    # Untiled reference.
    linalg_gemm[DType.bfloat16](c_full_p, a_p, b_p, M, N, K, ctx)
    ctx.synchronize()
    var full_host = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
    c_full.enqueue_copy_to(full_host)
    ctx.synchronize()

    var tile_host = ctx.enqueue_create_host_buffer[DType.bfloat16](M * tile_n)
    var max_abs = Float32(0)
    var max_rel = Float32(0)
    var ref_mag = Float32(0)

    for t in range(ntiles):
        var start = t * tile_n
        var this_n = min(tile_n, N - start)
        linalg_gemm[DType.bfloat16](
            c_tile_p, a_p, b_p.unsafe_offset((start * K)), M, this_n, K, ctx
        )
        ctx.synchronize()
        c_tile.enqueue_copy_to(tile_host)
        ctx.synchronize()
        for r in range(M):
            for j in range(this_n):
                var want = full_host.unsafe_ptr()[unsafe_offset=r * N + start + j].cast[
                    DType.float32
                ]()
                var got = tile_host.unsafe_ptr()[unsafe_offset=r * tile_n + j].cast[
                    DType.float32
                ]()
                var d = abs(got - want)
                max_abs = max(max_abs, d)
                ref_mag = max(ref_mag, abs(want))
                var denom = abs(want)
                if denom > 1e-4:
                    max_rel = max(max_rel, d / denom)

    print(
        "CHECK m=",
        M,
        " n=",
        N,
        " tiles=",
        tiles,
        " realized_tiles=",
        ntiles,
        " tile_n=",
        tile_n,
        " max_abs=",
        max_abs,
        " max_rel=",
        max_rel,
        " ref_mag=",
        ref_mag,
    )


def main() raises:
    var ctx = DeviceContext()

    print("=== vocab-tiled LM-head GEMM sweep (bf16, K=C=768, N=V_p=50304) ===")
    print("")

    # 50304 = 2^7 * 393, so every tile count below divides it exactly and
    # every tile is the same width -- no ragged remainder tile to explain.

    # Correctness of the decomposition itself, at a small M so the full
    # [M, V_p] reference fits comfortably.
    check_tiles(512, 1, ctx)
    check_tiles(512, 2, ctx)
    check_tiles(512, 4, ctx)
    check_tiles(512, 8, ctx)
    check_tiles(512, 16, ctx)
    check_tiles(512, 32, ctx)
    check_tiles(512, 64, ctx)
    check_tiles(512, 128, ctx)
    print("")

    # M = B*T. 8192 is B=8/T=1024 (the shape Team M measured memory at);
    # 32768 is B=32/T=1024 (the production shape the campaign targets).
    run_tiles(8192, 1, 5, 20, ctx)
    run_tiles(8192, 2, 5, 20, ctx)
    run_tiles(8192, 4, 5, 20, ctx)
    run_tiles(8192, 8, 5, 20, ctx)
    run_tiles(8192, 16, 5, 20, ctx)
    run_tiles(8192, 32, 5, 20, ctx)
    run_tiles(8192, 64, 5, 20, ctx)
    run_tiles(8192, 128, 5, 20, ctx)
    print("")
    run_tiles(32768, 1, 5, 20, ctx)
    run_tiles(32768, 2, 5, 20, ctx)
    run_tiles(32768, 4, 5, 20, ctx)
    run_tiles(32768, 8, 5, 20, ctx)
    run_tiles(32768, 16, 5, 20, ctx)
    run_tiles(32768, 32, 5, 20, ctx)
    run_tiles(32768, 64, 5, 20, ctx)
    run_tiles(32768, 128, 5, 20, ctx)
    print("")
