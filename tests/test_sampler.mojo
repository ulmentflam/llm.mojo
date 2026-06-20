# ===----------------------------------------------------------------------=== #
# Pure-Mojo tests for llmm.sampler.sample_softmax vs llmc/sampler.h reference.
#
# Run with:  make test-mojo
#
# The reference below is an independent transcription of Karpathy's C:
#   expf logits, double norm accumulator, float coin/cdf, coin *= norm in f64.
# ===----------------------------------------------------------------------=== #

from std.math import exp
from std.testing import assert_equal, TestSuite
from std.memory import alloc, UnsafePointer

from llmm.sampler import sample_softmax


def sample_softmax_llmc_reference(
    logits_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    n: Int,
    mut coin: Float32,
) -> Int:
    var norm: Float64 = 0.0
    for i in range(n):
        norm += Float64(exp(logits_ptr[i].cast[DType.float32]()))

    coin = Float32(Float64(coin) * norm)

    var cdf: Float32 = 0.0
    for i in range(n):
        cdf += exp(logits_ptr[i].cast[DType.float32]())
        if coin < cdf:
            return i
    return n - 1


def _alloc_logits(n: Int) -> UnsafePointer[Scalar[DType.float32], MutAnyOrigin]:
    return alloc[Scalar[DType.float32]](n).as_unsafe_any_origin()


def _assert_matches_reference(
    logits_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    n: Int,
    coin: Float32,
) raises:
    var logits_immut = rebind[
        UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]
    ](logits_ptr)
    var coin_ref = coin
    var coin_got = coin
    var expected = sample_softmax_llmc_reference(logits_immut, n, coin_ref)
    var got = sample_softmax(logits_immut, n, coin_got)
    assert_equal(got, expected)


def test_single_element() raises:
    var logits = _alloc_logits(1)
    logits[0] = Scalar[DType.float32](3.5)
    _assert_matches_reference(logits, 1, Float32(0.0))
    _assert_matches_reference(logits, 1, Float32(0.5))
    _assert_matches_reference(logits, 1, Float32(0.999))


def test_uniform_logits() raises:
    var n = 4
    var logits = _alloc_logits(n)
    for i in range(n):
        logits[i] = Scalar[DType.float32](0.0)
    for k in range(20):
        var coin = Float32(Float64(k) / 20.0 * 0.99)
        _assert_matches_reference(logits, n, coin)


def test_peaked_distribution() raises:
    var n = 5
    var logits = _alloc_logits(n)
    logits[0] = Scalar[DType.float32](-1.0)
    logits[1] = Scalar[DType.float32](0.0)
    logits[2] = Scalar[DType.float32](10.0)
    logits[3] = Scalar[DType.float32](-2.0)
    logits[4] = Scalar[DType.float32](-3.0)
    for k in range(50):
        var coin = Float32(Float64(k) / 50.0 * 0.999)
        _assert_matches_reference(logits, n, coin)


def test_arithmetic_progression() raises:
    var n = 8
    var logits = _alloc_logits(n)
    for i in range(n):
        logits[i] = Scalar[DType.float32](Float32(i) * 0.25 - 1.0)
    for k in range(100):
        var coin = Float32(Float64(k) / 100.0 * 0.9999)
        _assert_matches_reference(logits, n, coin)


def test_seeded_logits_sweep() raises:
    var sizes = List[Int]()
    sizes.append(16)
    sizes.append(64)
    sizes.append(256)
    sizes.append(1024)
    for s_idx in range(len(sizes)):
        var n = sizes[s_idx]
        var logits = _alloc_logits(n)
        var state = UInt64(0x123456789ABCDEF0 + UInt64(s_idx))
        for i in range(n):
            state ^= state >> 12
            state ^= state << 25
            state ^= state >> 27
            var u = (state * UInt64(0x2545F4914F6CDD1D)) >> 32
            var x = Float32(Int(u % UInt64(10000))) / 1000.0 - 5.0
            logits[i] = Scalar[DType.float32](x)
        for k in range(200):
            var coin = Float32(Float64(k) / 200.0 * 0.9999)
            _assert_matches_reference(logits, n, coin)


def test_large_vocab_like() raises:
    var n = 50304
    var logits = _alloc_logits(n)
    var state = UInt64(42)
    for i in range(n):
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        var u = (state * UInt64(0x2545F4914F6CDD1D)) >> 32
        var x = Float32(Int(u % UInt64(10000))) / 500.0 - 10.0
        logits[i] = Scalar[DType.float32](x)
    for k in range(50):
        var coin = Float32(Float64(k) / 50.0 * 0.9999)
        _assert_matches_reference(logits, n, coin)


def test_extreme_logits() raises:
    var n = 6
    var logits = _alloc_logits(n)
    logits[0] = Scalar[DType.float32](-100.0)
    logits[1] = Scalar[DType.float32](-50.0)
    logits[2] = Scalar[DType.float32](0.0)
    logits[3] = Scalar[DType.float32](50.0)
    logits[4] = Scalar[DType.float32](100.0)
    logits[5] = Scalar[DType.float32](-100.0)
    for k in range(30):
        var coin = Float32(Float64(k) / 30.0 * 0.999)
        _assert_matches_reference(logits, n, coin)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
