# ===----------------------------------------------------------------------=== #
# Tests for llmm.rng_device — counter-based device RNG (Widynski's Squares)
# + fp32 -> bf16 stochastic-rounding cast.
#
# Run with:  make test-mojo   (equivalent to `pixi run mojo run -I . tests/test_rng_sr.mojo`)
# GPU-launching tests self-skip if no NVIDIA GPU is present (matching
# tests/test_zero.mojo's convention); when a GPU *is* present, wrap the
# invocation in the shared-GPU lock:
#   flock -w 10800 /tmp/llmm-gpu.lock -c \
#     'pixi run mojo run -I . tests/test_rng_sr.mojo'
#
# Coverage:
#   (a) RNG determinism: same (seed, counter, stream) -> identical output,
#       different counters -> different output, and a GPU kernel calling the
#       exact same functions matches the host call bit-for-bit.
#   (b) Uniformity sanity: mean(uniform01) ~= 0.5, a chi-square-ish bucket
#       check on rng_u32.
#   (c) SR unbiasedness: for fp32 values straddling bf16 ULPs, the mean of
#       many independent sr_cast draws converges to the true value.
#   (d) SR determinism: fixed seed -> bit-identical output, on host and
#       across two separate GPU kernel launches.
#   (e) NaN/Inf passthrough + "exact value has zero rounding variance" +
#       "output is always one of the two representable neighbors" — the
#       correctness properties the unbiasedness claim in (c) depends on.
#
# The flag-off RNE bit-identity gate for llmm/adamw.mojo is *not* re-tested
# here — see the module docstring in llmm/adamw.mojo (`SR_MASTER_ENABLED`)
# for why the `-D LLMM_SR_MASTER=1`-off path is provably unchanged code (the
# `else` branch is byte-identical to the original `param.cast[dtype]()`
# line).
# ===----------------------------------------------------------------------=== #

from std.memory import bitcast, UnsafePointer
from std.math import isnan, isinf, nan, inf, ceildiv
from std.sys import has_nvidia_gpu_accelerator
from std.testing import (
    TestSuite,
    assert_true,
    assert_equal,
    assert_almost_equal,
)

from llmm.rng_device import (
    rng_key,
    squares32,
    rng_u32,
    rng_uniform01,
    sr_round_bits,
    sr_cast_bf16,
)
from _rng_sr_gpu_kernels import rng_u32_kernel, sr_cast_kernel
from _gpu_test_common import shared_gpu_ctx


# ===----------------------------------------------------------------------=== #
# Small helpers shared across tests.
# ===----------------------------------------------------------------------=== #


@always_inline
def _bf16_bits(x: Scalar[DType.bfloat16]) -> UInt16:
    return bitcast[DType.uint16](x)


@always_inline
def _f32_bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


@always_inline
def _bf16_lo_bits(x: Float32) -> UInt32:
    """The fp32 bit pattern of `x` truncated toward zero to its bf16
    round-down neighbor (i.e. bf16's bit pattern zero-extended back to 32
    bits) — the "floor" value any SR draw of `x` must bracket.
    """
    return _f32_bits(x) & UInt32(0xFFFF0000)


@always_inline
def _straddle(base_bits: UInt32, frac_num: UInt32) -> Float32:
    """Construct an fp32 value `frac_num/65536` of the way from the bf16
    round-down neighbor of `base_bits` to its round-up neighbor. `base_bits`
    must already have its low 16 bits zeroed (a `_bf16_lo_bits` result).
    """
    return bitcast[DType.float32](base_bits | (frac_num & UInt32(0xFFFF)))


# ===----------------------------------------------------------------------=== #
# (a) RNG determinism — host.
# ===----------------------------------------------------------------------=== #


def test_rng_u32_deterministic_same_input() raises:
    var seed: UInt64 = 12345
    var a = rng_u32(seed, 7, 0)
    var b = rng_u32(seed, 7, 0)
    assert_equal(a, b)


def test_rng_key_deterministic_same_input() raises:
    var k1 = rng_key(999, 3)
    var k2 = rng_key(999, 3)
    assert_equal(k1, k2)
    # Squares requires an odd key; rng_key must always produce one.
    assert_equal(k1 & UInt64(1), UInt64(1))


def test_rng_u32_differs_across_counters() raises:
    var seed: UInt64 = 42
    var key = rng_key(seed, 0)
    var seen = List[UInt32]()
    for c in range(2000):
        var v = squares32(UInt64(c), key)
        seen.append(v)
    # 32-bit output space is ~4.3e9; 2000 draws collide by chance with
    # probability ~2000^2 / (2*4.3e9) ~= 4.6e-4 for a *good* generator (all
    # distinct is expected, not a coincidence of good luck). Sort a copy and
    # scan for adjacent duplicates rather than relying on a Set/Dict type.
    var sorted_seen = seen.copy()
    sort(sorted_seen[:])
    var num_dup = 0
    for i in range(1, len(sorted_seen)):
        if sorted_seen[i] == sorted_seen[i - 1]:
            num_dup += 1
    assert_equal(num_dup, 0)


def test_rng_u32_differs_across_seed_and_stream() raises:
    var a = rng_u32(1, 0, 0)
    var b = rng_u32(2, 0, 0)  # different seed
    var c = rng_u32(1, 0, 1)  # different stream
    assert_true(a != b)
    assert_true(a != c)
    assert_true(b != c)


def test_rng_uniform01_range() raises:
    var seed: UInt64 = 7
    for c in range(5000):
        var u = rng_uniform01(seed, UInt64(c), 0)
        assert_true(u >= Float32(0.0))
        assert_true(u < Float32(1.0))


# ===----------------------------------------------------------------------=== #
# (b) Uniformity sanity: mean ~= 0.5 and a chi-square-ish bucket check.
# ===----------------------------------------------------------------------=== #


def test_rng_uniform01_mean_and_bucket_uniformity() raises:
    comptime N = 200_000
    comptime NUM_BUCKETS = 10
    var seed: UInt64 = 2026
    var sum = Float64(0.0)
    var buckets = List[Int]()
    for _ in range(NUM_BUCKETS):
        buckets.append(0)

    for c in range(N):
        var u = rng_uniform01(seed, UInt64(c), 5)
        sum += Float64(u)
        var b = Int(u * Float32(NUM_BUCKETS))
        if b >= NUM_BUCKETS:  # u can be arbitrarily close to 1.0
            b = NUM_BUCKETS - 1
        buckets[b] += 1

    var mean = sum / Float64(N)
    # Hoeffding bound for N=200000 bounded-in-[0,1] draws: P(|mean-0.5|>0.01)
    # <= 2*exp(-2*N*0.01^2) = 2*exp(-40) ~= 8e-18 — effectively deterministic
    # for a correctly-uniform generator, not a tuned-to-barely-pass tolerance.
    assert_true(abs(mean - 0.5) < 0.01)

    # Chi-square-ish bucket check (not a formal p-value lookup; a full
    # TestU01-style suite is out of scope for this codebase). Expected count
    # per bucket is N/NUM_BUCKETS = 20000; chi-square(9 dof) has mean 9, std
    # sqrt(18)~=4.24, so a true-uniform generator overwhelmingly lands well
    # under 60 (~12 std above the mean) and a badly broken one blows past it.
    var expected = Float64(N) / Float64(NUM_BUCKETS)
    var chi2 = Float64(0.0)
    for b in range(NUM_BUCKETS):
        var diff = Float64(buckets[b]) - expected
        chi2 += diff * diff / expected
    assert_true(chi2 < 60.0)


# ===----------------------------------------------------------------------=== #
# (e)-part-1: sr_round_bits correctness properties the unbiasedness test
# depends on: NaN/Inf passthrough, exact values have zero variance, and the
# result is always one of exactly the two bracketing bf16 neighbors.
# ===----------------------------------------------------------------------=== #


def test_sr_round_bits_nan_inf_passthrough() raises:
    var nan_f = nan[DType.float32]()
    var pos_inf = inf[DType.float32]()
    var neg_inf = -inf[DType.float32]()
    var rand_bits: List[UInt32] = [0, 1, 0x7FFF, 0x8000, 0xFFFF]
    for rb in rand_bits:
        var r_nan = sr_round_bits(nan_f, rb)
        assert_true(isnan(Float32(r_nan)))
        assert_equal(
            _bf16_bits(sr_round_bits(pos_inf, rb)),
            _bf16_bits(pos_inf.cast[DType.bfloat16]()),
        )
        assert_equal(
            _bf16_bits(sr_round_bits(neg_inf, rb)),
            _bf16_bits(neg_inf.cast[DType.bfloat16]()),
        )


def test_sr_round_bits_exact_value_zero_variance() raises:
    # Values whose fp32 bit pattern already has zero low-16 bits are exactly
    # bf16-representable; SR must return that exact value regardless of the
    # random draw (frac = 0, so P(round up) = 0).
    var bases_exact: List[Float32] = [0.0, 1.0, -1.0, 2.0, 0.5, -8.0, 100.0]
    for base in bases_exact:
        var bits = _bf16_lo_bits(base)
        var exact = bitcast[DType.float32](bits)
        var want = exact.cast[DType.bfloat16]()
        var rand_bits: List[UInt32] = [
            0,
            1,
            100,
            0x7FFF,
            0x8000,
            0xFFFE,
            0xFFFF,
        ]
        for rb in rand_bits:
            assert_equal(_bf16_bits(sr_round_bits(exact, rb)), _bf16_bits(want))


def test_sr_round_bits_only_two_neighbors() raises:
    var bases: List[Float32] = [
        1.0,
        -1.0,
        3.14159,
        -2.71828,
        12345.6789,
        0.0001234,
    ]
    for base in bases:
        var lo_bits = _bf16_lo_bits(base)
        var hi_bits = lo_bits + UInt32(0x10000)
        var lo_bf = bitcast[DType.float32](lo_bits).cast[DType.bfloat16]()
        var hi_bf = bitcast[DType.float32](hi_bits).cast[DType.bfloat16]()
        var x = _straddle(lo_bits, UInt32(30000))  # frac ~= 0.458
        var rand_bits: List[UInt32] = [0, 1, 12345, 32767, 32768, 65000, 65535]
        for rb in rand_bits:
            var got = _bf16_bits(sr_round_bits(x, rb))
            var is_lo = got == _bf16_bits(lo_bf)
            var is_hi = got == _bf16_bits(hi_bf)
            assert_true(is_lo or is_hi)


def test_sr_cast_bf16_deterministic_same_input() raises:
    var x = Float32(3.14159)
    var seed: UInt64 = 555
    var a = sr_cast_bf16(x, seed, 42, 1)
    var b = sr_cast_bf16(x, seed, 42, 1)
    assert_equal(_bf16_bits(a), _bf16_bits(b))


# ===----------------------------------------------------------------------=== #
# (c) SR unbiasedness — the load-bearing statistical property.
# ===----------------------------------------------------------------------=== #


def test_sr_cast_bf16_unbiased() raises:
    comptime M = 20_000  # draws per (base, frac) test point
    var seed: UInt64 = 31337
    var bases: List[Float32] = [
        1.0,
        3.14159,
        -2.71828,
        100.25,
        0.0001234,
        -0.5,
        12345.6789,
    ]
    # Fractions of the way from the bf16 round-down to round-up neighbor.
    var fracs: List[UInt32] = [9830, 32768, 55705]  # ~0.15, 0.5, ~0.85

    var stream: UInt64 = 0
    for base in bases:
        var lo_bits = _bf16_lo_bits(base)
        var hi_bits = lo_bits + UInt32(0x10000)
        var lo_val = Float64(bitcast[DType.float32](lo_bits))
        var hi_val = Float64(bitcast[DType.float32](hi_bits))
        # For negative `base`, increasing the bit pattern moves the value
        # *away* from zero (more negative) — see llmm/rng_device.mojo's
        # sr_round_bits docstring on same-sign bit-pattern monotonicity — so
        # `hi_val` can be algebraically less than `lo_val`. `step`'s only use
        # below is as a positive-width tolerance scale, so take the
        # magnitude.
        var step = abs(hi_val - lo_val)

        for frac in fracs:
            var x = _straddle(lo_bits, frac)
            stream += 1  # distinct substream per (base, frac) test point

            var sum = Float64(0.0)
            for c in range(M):
                var got = sr_cast_bf16(x, seed, UInt64(c), stream)
                sum += Float64(got.cast[DType.float32]())
            var mean = sum / Float64(M)

            # Hoeffding bound (draws bounded in [lo_val, hi_val], width
            # `step`): P(|mean - E[X]| > 0.05*step) <=
            # 2*exp(-2*M*(0.05)^2) = 2*exp(-100) — not a tuned tolerance,
            # the true failure probability here is astronomically small for
            # a correctly-unbiased rounder and would fail hard (mean pinned
            # near lo_val or hi_val) for a biased one.
            var tol = step * 0.05
            if tol == 0.0:  # base already exactly bf16-representable (frac
                tol = 1e-12  # rounds to itself every draw; guard div-by-0 UX)
            var msg = (
                "SR mean diverges from true value: base="
                + String(base)
                + " frac="
                + String(frac)
                + " x="
                + String(x)
                + " mean="
                + String(mean)
                + " tol="
                + String(tol)
            )
            assert_true(abs(mean - Float64(x)) < tol, msg)


# ===----------------------------------------------------------------------=== #
# GPU kernels + (a)/(d) device-vs-host + device-determinism tests.
#
# These launch an actual GPU kernel calling the *same* rng_device functions
# used above on host — see llmm/rng_device.mojo's module docstring for why a
# single shared implementation is the meaningful "host reference" on this
# toolchain (the risk is GPU-codegen divergence from host behavior, not a
# second independently-written algorithm disagreeing with the first).
# ===----------------------------------------------------------------------=== #


def test_rng_u32_device_matches_host() raises:
    if not has_nvidia_gpu_accelerator():
        return

    var ctx = shared_gpu_ctx()
    comptime N = 4096
    comptime BLOCK_SIZE = 128
    var seed: UInt64 = 8675309
    var stream: UInt64 = 3

    var dev_out = ctx.enqueue_create_buffer[DType.uint32](N)
    var num_blocks = ceildiv(N, BLOCK_SIZE)
    var compiled = ctx.compile_function[rng_u32_kernel]()
    ctx.enqueue_function(
        compiled,
        dev_out.unsafe_ptr(),
        seed,
        stream,
        N,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )
    var host_out = ctx.enqueue_create_host_buffer[DType.uint32](N)
    dev_out.enqueue_copy_to(host_out)
    ctx.synchronize()

    var mismatches = 0
    for i in range(N):
        var want = rng_u32(seed, UInt64(i), stream)
        if host_out.unsafe_ptr()[i] != want:
            mismatches += 1
    assert_equal(mismatches, 0)


def test_sr_cast_device_matches_host_and_is_deterministic() raises:
    if not has_nvidia_gpu_accelerator():
        return

    var ctx = shared_gpu_ctx()
    comptime N = 4096
    comptime BLOCK_SIZE = 128
    var seed: UInt64 = 20260710
    var stream: UInt64 = 4

    var host_x = ctx.enqueue_create_host_buffer[DType.float32](N)
    for i in range(N):
        # A mix of magnitudes/signs so this exercises more than one exponent
        # bucket (denormal-adjacent through large values).
        host_x.unsafe_ptr()[i] = (Float32(i) - Float32(N // 2)) * 0.0173

    var dev_x = ctx.enqueue_create_buffer[DType.float32](N)
    dev_x.enqueue_copy_from(host_x)
    ctx.synchronize()

    var dev_out_a = ctx.enqueue_create_buffer[DType.uint16](N)
    var dev_out_b = ctx.enqueue_create_buffer[DType.uint16](N)
    var num_blocks = ceildiv(N, BLOCK_SIZE)
    var compiled = ctx.compile_function[sr_cast_kernel]()

    # Two independent launches with identical inputs -> (d) SR determinism.
    ctx.enqueue_function(
        compiled,
        dev_out_a.unsafe_ptr(),
        dev_x.unsafe_ptr(),
        seed,
        stream,
        N,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )
    ctx.enqueue_function(
        compiled,
        dev_out_b.unsafe_ptr(),
        dev_x.unsafe_ptr(),
        seed,
        stream,
        N,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )

    var host_out_a = ctx.enqueue_create_host_buffer[DType.uint16](N)
    var host_out_b = ctx.enqueue_create_host_buffer[DType.uint16](N)
    dev_out_a.enqueue_copy_to(host_out_a)
    dev_out_b.enqueue_copy_to(host_out_b)
    ctx.synchronize()

    var det_mismatches = 0
    var host_ref_mismatches = 0
    for i in range(N):
        var a = host_out_a.unsafe_ptr()[i]
        var b = host_out_b.unsafe_ptr()[i]
        if a != b:
            det_mismatches += 1
        var want = _bf16_bits(
            sr_cast_bf16(host_x.unsafe_ptr()[i], seed, UInt64(i), stream)
        )
        if a != want:
            host_ref_mismatches += 1

    assert_equal(det_mismatches, 0)  # (d) two device runs, bit-identical
    assert_equal(host_ref_mismatches, 0)  # (a) device output == host reference


# ===----------------------------------------------------------------------=== #
# Entry point
# ===----------------------------------------------------------------------=== #


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
