from std.math import exp
from llmm.memory import ImmutMemPtr
from llmm.rand import FLOAT32_SIGNIFICAND_BITS, U32_BITS


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #


# xorshift64* (Marsaglia's xorshift, Vigna's multiplicative scrambler), with
# llm.c's constants (llmc/sampler.h). max_gpt2_common.py mirrors these under
# the same names; changing any of them breaks sample parity between the two.
comptime XORSHIFT_RIGHT_1 = 12
comptime XORSHIFT_LEFT = 25
comptime XORSHIFT_RIGHT_2 = 27
comptime XORSHIFT_STAR_MULTIPLIER = 0x2545F4914F6CDD1D
# The scrambled state's low bits are weak, so the output keeps the high half.
comptime U32_OUTPUT_SHIFT = U32_BITS
# A float32 carries 24 significant bits: keep the top 24 of the u32 and scale
# by 2^24, so every value in [0, 1) is exactly representable.
comptime F32_DISCARD_BITS = U32_BITS - FLOAT32_SIGNIFICAND_BITS
comptime F32_UNIT_SCALE = Float32(1 << FLOAT32_SIGNIFICAND_BITS)


# ===----------------------------------------------------------------------=== #
# Random Functions
# ===----------------------------------------------------------------------=== #


def random_u32(mut state: UInt64) -> UInt32:
    """One xorshift* step; UInt64 arithmetic wraps like llm.c's uint64_t."""
    state ^= state >> XORSHIFT_RIGHT_1
    state ^= state << XORSHIFT_LEFT
    state ^= state >> XORSHIFT_RIGHT_2
    var scrambled = state * XORSHIFT_STAR_MULTIPLIER
    return (scrambled >> U32_OUTPUT_SHIFT).cast[DType.uint32]()


def random_f32(mut state: UInt64) -> Float32:
    """Uniform float32 in [0, 1)."""
    return Float32(random_u32(state) >> F32_DISCARD_BITS) / F32_UNIT_SCALE


def random_permutation(mut arr: List[Int], mut state: UInt64):
    var n = len(arr)
    for i in range(n - 1, 0, -1):
        var r = Int(random_u32(state) % UInt32(i + 1))
        var tmp = arr[i]
        arr[i] = arr[r]
        arr[r] = tmp


# ===----------------------------------------------------------------------=== #
# Sampling Functions
# ===----------------------------------------------------------------------=== #


def sample_softmax(
    logits_ptr: ImmutMemPtr[DType.float32],
    n: Int,
    mut coin: Float32,
) -> Int:
    var norm: Float64 = 0.0
    for i in range(n):
        norm += Float64(exp(logits_ptr[unsafe_offset=i].cast[DType.float32]()))

    coin = Float32(Float64(coin) * norm)

    var cdf: Float32 = 0.0
    for i in range(n):
        cdf += exp(logits_ptr[unsafe_offset=i].cast[DType.float32]())
        if coin < cdf:
            return i
    return n - 1  # in case of rounding errors
