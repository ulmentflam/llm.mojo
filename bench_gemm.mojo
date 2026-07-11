"""Standalone micro-benchmark: bf16 Metal GEMM candidates vs stdlib linalg.matmul.

Times, in ONE process (so ambient GPU load cancels in the ratios), the REAL
GPT-2 124M linear-layer shapes for three arms:

  * `linalg.matmul[transpose_b=True, target="gpu"]` in bf16  (the CURRENT path
    matmul_fwd's Metal branch takes),
  * `linalg.matmul[transpose_b=True, target="gpu"]` in fp32  (to quantify the
    bf16 headroom the current path is / isn't getting),
  * a hand-written register-tiled SIMD GEMM (fp32 accumulate, no tensor cores)
    — "Plan B", because Mojo's `layout.tensor_core.TensorCore` MMA store_d does
    not instantiate on the Metal target in this nightly (see notes below).

TensorCore probe results on this box (Mojo 1.0.0b3.dev2026062706, Apple M4):
  * `Index(16, 8, 16)` bf16  -> `mma` op itself is unsupported on the target.
  * `Index(8, 8, 8)` / `Index(8, 8, 4)` bf16 -> load_a/load_b/mma_op compile,
    but `store_d` fails ("No valid shape/type to store to LayoutTensor d") for
    every out dtype (fp32, bf16) and address space (SHARED, LOCAL). i.e. you
    can run the simdgroup multiply but cannot get the accumulator back out
    through the public API. So a tiled tensor-core GEMM is not expressible here.
    (The flash-attention kernel that uses TensorCore is gated behind
    USE_FLASH_FWD = False, i.e. it is dead code and was never Metal-validated.)

Run under the shared GPU lock:
  lockf -t 10800 /tmp/llmm-gpu.lock pixi run mojo run -I . bench_gemm.mojo
"""

from std.math import ceildiv, sqrt
from std.time import global_perf_counter_ns
from layout import Layout, TileTensor
from layout.tile_layout import row_major
from layout.layout_tensor import LayoutTensor
from linalg.matmul import matmul
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu import block_idx, thread_idx, barrier
from std.collections import InlineArray

from llmm.memory import MutKernelPtr, ImmutKernelPtr


# ===----------------------------------------------------------------------=== #
# Plan B: register-tiled SIMD GEMM (fp32 accumulate, no tensor cores)
# ===----------------------------------------------------------------------=== #
#
# C[M,N] = A[M,K] @ opB, opB = B[N,K] (transpose_b=True, the forward linear
# orientation) or B[K,N] (transpose_b=False). One threadgroup computes a
# BM x BN output tile; each thread owns a TM x TN micro-tile accumulated in
# fp32 registers. A/B are staged tile-by-tile through threadgroup memory; the
# inner kk loop is a rank-1 outer-product update (classic register blocking).
# Boundary-guarded stores handle non-multiple N (e.g. vocab 50257).


def gemm_simd_kernel[
    in_dtype: DType,
    out_dtype: DType,
    transpose_b: Bool,
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
](
    c_ptr: MutKernelPtr[out_dtype],
    a_ptr: ImmutKernelPtr[in_dtype],
    b_ptr: ImmutKernelPtr[in_dtype],
    M: Int,
    N: Int,
    K: Int,
) -> None:
    comptime TX = BN // TN  # threads spanning N
    comptime TY = BM // TM  # threads spanning M
    comptime NTHREADS = TX * TY

    var t = Int(thread_idx.x)
    var tx = t % TX
    var ty = t // TX
    var block_row = Int(block_idx.y) * BM
    var block_col = Int(block_idx.x) * BN

    var a_sh = LayoutTensor[
        in_dtype,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var b_sh = LayoutTensor[
        in_dtype,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var acc = InlineArray[Float32, TM * TN](fill=0.0)

    var k0 = 0
    while k0 < K:
        # Cooperative global -> shared stage of A[BM,BK] and B[BK,BN].
        var e = t
        while e < BM * BK:
            var i = e // BK
            var j = e % BK
            var gr = block_row + i
            var gk = k0 + j
            a_sh.ptr[e] = a_ptr[gr * K + gk] if (gr < M and gk < K) else Scalar[
                in_dtype
            ](0)
            e += NTHREADS
        e = t
        while e < BK * BN:
            var kj = e // BN
            var ni = e % BN
            var gn = block_col + ni
            var gk = k0 + kj

            comptime if transpose_b:
                b_sh.ptr[e] = b_ptr[gn * K + gk] if (
                    gn < N and gk < K
                ) else Scalar[in_dtype](0)
            else:
                b_sh.ptr[e] = b_ptr[gk * N + gn] if (
                    gn < N and gk < K
                ) else Scalar[in_dtype](0)
            e += NTHREADS
        barrier()

        comptime for kk in range(BK):
            var a_frag = InlineArray[Float32, TM](uninitialized=True)
            var b_frag = InlineArray[Float32, TN](uninitialized=True)
            comptime for i in range(TM):
                a_frag[i] = a_sh.ptr[(ty * TM + i) * BK + kk].cast[
                    DType.float32
                ]()
            comptime for j in range(TN):
                b_frag[j] = b_sh.ptr[kk * BN + (tx * TN + j)].cast[
                    DType.float32
                ]()
            comptime for i in range(TM):
                comptime for j in range(TN):
                    acc[i * TN + j] += a_frag[i] * b_frag[j]
        barrier()
        k0 += BK

    comptime for i in range(TM):
        var gr = block_row + ty * TM + i
        comptime for j in range(TN):
            var gc = block_col + tx * TN + j
            if gr < M and gc < N:
                c_ptr[gr * N + gc] = acc[i * TN + j].cast[out_dtype]()


def launch_gemm_simd[
    in_dtype: DType,
    out_dtype: DType,
    transpose_b: Bool,
](
    c_ptr: MutKernelPtr[out_dtype],
    a_ptr: ImmutKernelPtr[in_dtype],
    b_ptr: ImmutKernelPtr[in_dtype],
    M: Int,
    N: Int,
    K: Int,
    ctx: DeviceContext,
) raises -> None:
    comptime BM = 64
    comptime BN = 64
    comptime BK = 16
    comptime TM = 4
    comptime TN = 4
    comptime k = gemm_simd_kernel[
        in_dtype, out_dtype, transpose_b, BM, BN, BK, TM, TN
    ]
    var compiled = ctx.compile_function[k]()
    ctx.enqueue_function(
        compiled,
        c_ptr,
        a_ptr,
        b_ptr,
        M,
        N,
        K,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=((BM // TM) * (BN // TN),),
    )


# ===----------------------------------------------------------------------=== #
# linalg.matmul baselines (transpose_b=True forward orientation)
# ===----------------------------------------------------------------------=== #


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
    var c = TileTensor(
        Span[Scalar[dtype], MutAnyOrigin](ptr=c_ptr, length=M * N),
        row_major(M, N),
    )
    var a = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](ptr=a_ptr, length=M * K),
        row_major(M, K),
    )
    var b = TileTensor(
        Span[Scalar[dtype], ImmutAnyOrigin](ptr=b_ptr, length=N * K),
        row_major(N, K),
    )
    matmul[transpose_b=True, target="gpu"](c, a, b, ctx=ctx)


# ===----------------------------------------------------------------------=== #
# Harness
# ===----------------------------------------------------------------------=== #


def _lcg(mut state: UInt64) -> Float32:
    # xorshift64 -> [-0.5, 0.5]
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    var u = Float32((state >> 40) & 0xFFFFFF) / Float32(0xFFFFFF)
    return u - 0.5


def run_shape(
    name: String, M: Int, K: Int, N: Int, ctx: DeviceContext
) raises -> None:
    comptime WARMUP = 10
    comptime ITERS = 50

    # ---- host random A[M,K], B[N,K] (transpose_b=True: weight is [N,K]) ----
    var a_host = ctx.enqueue_create_host_buffer[DType.float32](M * K)
    var b_host = ctx.enqueue_create_host_buffer[DType.float32](N * K)
    ctx.synchronize()
    var scale = 1.0 / sqrt(Float32(K))
    var st: UInt64 = 0x243F6A8885A308D3 + UInt64(M * 131 + N * 17 + K)
    for i in range(M * K):
        a_host.unsafe_ptr()[i] = _lcg(st) * scale
    for i in range(N * K):
        b_host.unsafe_ptr()[i] = _lcg(st) * scale

    # ---- device buffers ----
    var a_bf = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var b_bf = ctx.enqueue_create_buffer[DType.bfloat16](N * K)
    var c_lin = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    var c_cand = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    var a_f32 = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_f32 = ctx.enqueue_create_buffer[DType.float32](N * K)
    var c_f32 = ctx.enqueue_create_buffer[DType.float32](M * N)

    var a_bf_host = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
    var b_bf_host = ctx.enqueue_create_host_buffer[DType.bfloat16](N * K)
    ctx.synchronize()
    for i in range(M * K):
        a_bf_host.unsafe_ptr()[i] = a_host.unsafe_ptr()[i].cast[
            DType.bfloat16
        ]()
    for i in range(N * K):
        b_bf_host.unsafe_ptr()[i] = b_host.unsafe_ptr()[i].cast[
            DType.bfloat16
        ]()
    a_bf.enqueue_copy_from(a_bf_host)
    b_bf.enqueue_copy_from(b_bf_host)
    a_f32.enqueue_copy_from(a_host)
    b_f32.enqueue_copy_from(b_host)
    ctx.synchronize()

    var a_bf_p = rebind[ImmutKernelPtr[DType.bfloat16]](
        a_bf.unsafe_ptr().as_immutable().as_unsafe_any_origin()
    )
    var b_bf_p = rebind[ImmutKernelPtr[DType.bfloat16]](
        b_bf.unsafe_ptr().as_immutable().as_unsafe_any_origin()
    )
    var c_lin_p = rebind[MutKernelPtr[DType.bfloat16]](
        c_lin.unsafe_ptr().as_unsafe_any_origin()
    )
    var c_cand_p = rebind[MutKernelPtr[DType.bfloat16]](
        c_cand.unsafe_ptr().as_unsafe_any_origin()
    )
    var a_f32_p = rebind[ImmutKernelPtr[DType.float32]](
        a_f32.unsafe_ptr().as_immutable().as_unsafe_any_origin()
    )
    var b_f32_p = rebind[ImmutKernelPtr[DType.float32]](
        b_f32.unsafe_ptr().as_immutable().as_unsafe_any_origin()
    )
    var c_f32_p = rebind[MutKernelPtr[DType.float32]](
        c_f32.unsafe_ptr().as_unsafe_any_origin()
    )

    var flops = 2.0 * Float64(M) * Float64(N) * Float64(K)

    # ---- current path: linalg bf16 ----
    for _ in range(WARMUP):
        linalg_gemm[DType.bfloat16](c_lin_p, a_bf_p, b_bf_p, M, N, K, ctx)
    ctx.synchronize()
    var t0 = global_perf_counter_ns()
    for _ in range(ITERS):
        linalg_gemm[DType.bfloat16](c_lin_p, a_bf_p, b_bf_p, M, N, K, ctx)
    ctx.synchronize()
    var ms_lin = Float64(global_perf_counter_ns() - t0) / 1e6 / ITERS

    # ---- fp32 linalg (headroom reference) ----
    for _ in range(WARMUP):
        linalg_gemm[DType.float32](c_f32_p, a_f32_p, b_f32_p, M, N, K, ctx)
    ctx.synchronize()
    t0 = global_perf_counter_ns()
    for _ in range(ITERS):
        linalg_gemm[DType.float32](c_f32_p, a_f32_p, b_f32_p, M, N, K, ctx)
    ctx.synchronize()
    var ms_f32 = Float64(global_perf_counter_ns() - t0) / 1e6 / ITERS

    # ---- candidate: Plan-B register-tiled SIMD bf16 ----
    for _ in range(WARMUP):
        launch_gemm_simd[DType.bfloat16, DType.bfloat16, True](
            c_cand_p, a_bf_p, b_bf_p, M, N, K, ctx
        )
    ctx.synchronize()
    t0 = global_perf_counter_ns()
    for _ in range(ITERS):
        launch_gemm_simd[DType.bfloat16, DType.bfloat16, True](
            c_cand_p, a_bf_p, b_bf_p, M, N, K, ctx
        )
    ctx.synchronize()
    var ms_cand = Float64(global_perf_counter_ns() - t0) / 1e6 / ITERS

    # ---- correctness: candidate bf16 vs linalg bf16 ----
    var c_lin_host = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
    var c_cand_host = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
    c_lin.enqueue_copy_to(c_lin_host)
    c_cand.enqueue_copy_to(c_cand_host)
    ctx.synchronize()
    var max_abs = Float32(0)
    var max_rel = Float32(0)
    var ref_mag = Float32(0)
    for i in range(M * N):
        var r = c_lin_host.unsafe_ptr()[i].cast[DType.float32]()
        var g = c_cand_host.unsafe_ptr()[i].cast[DType.float32]()
        var d = abs(g - r)
        max_abs = max(max_abs, d)
        ref_mag = max(ref_mag, abs(r))
        var denom = abs(r)
        if denom > 1e-4:
            max_rel = max(max_rel, d / denom)

    var gf_lin = flops / (ms_lin * 1e6)
    var gf_f32 = flops / (ms_f32 * 1e6)
    var gf_cand = flops / (ms_cand * 1e6)

    print(name, "M=", M, "K=", K, "N=", N)
    print("   linalg-bf16 : ", ms_lin, " ms   ", gf_lin, " GFLOP/s")
    print("   linalg-fp32 : ", ms_f32, " ms   ", gf_f32, " GFLOP/s")
    print(
        "   simd-reg-bf16:",
        ms_cand,
        " ms   ",
        gf_cand,
        " GFLOP/s   x_vs_bf16=",
        ms_lin / ms_cand,
        "  x_vs_fp32=",
        ms_f32 / ms_cand,
    )
    print(
        "   cand vs linalg-bf16:  max_abs=",
        max_abs,
        " max_rel=",
        max_rel,
        " (ref_mag=",
        ref_mag,
        ")",
    )
    print("")


def main() raises:
    var ctx = DeviceContext()
    print("=== bf16 Metal GEMM micro-benchmark (M = B*T = 256) ===")
    print("")
    run_shape("qkv       ", 256, 768, 2304, ctx)
    run_shape("attn_proj ", 256, 768, 768, ctx)
    run_shape("fc        ", 256, 768, 3072, ctx)
    run_shape("fc_proj   ", 256, 3072, 768, ctx)
    run_shape("classifier", 256, 768, 50257, ctx)
