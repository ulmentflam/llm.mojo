from std.math import exp
from std.memory import UnsafePointer


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #


comptime RU32_HEX = 0x2545F4914F6CDD1D
comptime FLOAT_CONST = 16777216.0


# ===----------------------------------------------------------------------=== #
# Random Functions
# ===----------------------------------------------------------------------=== #


def random_u32(mut state: UInt64) -> UInt32:
    state ^= (
        state >> 12
    )  # Magic numbers from the Karpathy's llm.c implementation.
    state ^= (
        state << 25
    )  # Magic numbers from the Karpathy's llm.c implementation.
    state ^= (
        state >> 27
    )  # Magic numbers from the Karpathy's llm.c implementation.
    return ((state * RU32_HEX) >> 32).cast[DType.uint32]()


def random_f32(mut state: UInt64) -> Float32:
    return Float32(random_u32(state) >> 8) / FLOAT_CONST


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
    return n - 1  # in case of rounding errors
