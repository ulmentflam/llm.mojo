"""Mersenne Twister RNG, numerically identical to torch / llm.c's rand.h.

This is a direct port of `llmc/rand.h` from Karpathy's llm.c. It exists so that
from-scratch weight initialization (see `gpt_build_from_descriptor` in llm.c)
draws the exact same random numbers as PyTorch's `torch.manual_seed` +
`torch.normal`, giving bit-identical initial conditions for correctness tests.

Only the subset needed for `normal_` is exercised by weight init, but the whole
generator is ported for completeness (and matches the reference output in the
header comment of rand.h).
"""

from std.math import sqrt, log, cos, sin, pi

from llmm.memory import MutMemPtr


# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #


comptime MERSENNE_STATE_M = 397
comptime MERSENNE_STATE_N = 624

comptime LMASK = UInt32(0x7FFFFFFF)
comptime UMASK = UInt32(0x80000000)

comptime MATRIX_A_1 = UInt32(0x9908B0DF)

# 1e-12, added inside the Box-Muller log to avoid log(0).
comptime BOX_MULLER_EPSILON = Float32(1e-12)


# ===----------------------------------------------------------------------=== #
# Mersenne Twister State
# ===----------------------------------------------------------------------=== #


struct MT19937(Copyable, Movable):
    """PyTorch-compatible Mersenne Twister (MT19937).

    UInt32 arithmetic wraps modulo 2**32 in Mojo just like C `unsigned int`, so
    the multiply/add in `seed` and the tempering shifts below match the C
    reference without explicit masking.
    """

    var state: List[UInt32]
    var left: Int
    var next: Int

    def __init__(out self, seed: UInt32):
        self.state = List[UInt32]()
        for _ in range(MERSENNE_STATE_N):
            self.state.append(UInt32(0))
        self.left = 1
        self.next = 0
        self.seed(seed)

    def seed(mut self, seed: UInt32):
        """Equivalent to `manual_seed`."""
        self.state[0] = seed
        for j in range(1, MERSENNE_STATE_N):
            var prev = self.state[j - 1]
            self.state[j] = UInt32(1812433253) * (prev ^ (prev >> 30)) + UInt32(
                j
            )
        self.left = 1
        self.next = 0

    def _next_state(mut self):
        self.left = MERSENNE_STATE_N
        self.next = 0
        var y: UInt32
        for j in range(MERSENNE_STATE_N - MERSENNE_STATE_M):
            y = (self.state[j] & UMASK) | (self.state[j + 1] & LMASK)
            self.state[j] = (
                self.state[j + MERSENNE_STATE_M]
                ^ (y >> 1)
                ^ (MATRIX_A_1 if (y & 1) else UInt32(0))
            )
        for j in range(
            MERSENNE_STATE_N - MERSENNE_STATE_M, MERSENNE_STATE_N - 1
        ):
            y = (self.state[j] & UMASK) | (self.state[j + 1] & LMASK)
            self.state[j] = (
                self.state[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)]
                ^ (y >> 1)
                ^ (MATRIX_A_1 if (y & 1) else UInt32(0))
            )
        y = (self.state[MERSENNE_STATE_N - 1] & UMASK) | (self.state[0] & LMASK)
        self.state[MERSENNE_STATE_N - 1] = (
            self.state[MERSENNE_STATE_M - 1]
            ^ (y >> 1)
            ^ (MATRIX_A_1 if (y & 1) else UInt32(0))
        )

    def randint32(mut self) -> UInt32:
        self.left -= 1
        if self.left <= 0:
            self._next_state()
        var y = self.state[self.next]
        self.next += 1
        y ^= y >> 11
        y ^= (y << 7) & UInt32(0x9D2C5680)
        y ^= (y << 15) & UInt32(0xEFC60000)
        y ^= y >> 18
        return y

    def randint64(mut self) -> UInt64:
        # First draw supplies the high 32 bits (matches llm.c's evaluation).
        var hi = UInt64(self.randint32())
        var lo = UInt64(self.randint32())
        return (hi << 32) | lo

    def randfloat32(mut self) -> Float32:
        return Float32(Int(self.randint32() & UInt32(0xFFFFFF))) * (
            Float32(1.0) / Float32(1 << 24)
        )

    def randfloat64(mut self) -> Float64:
        return Float64(Int(self.randint64() & UInt64((1 << 53) - 1))) * (
            Float64(1.0) / Float64(1 << 53)
        )


# ===----------------------------------------------------------------------=== #
# Gaussian sampling (Box-Muller), matching torch.normal_
# ===----------------------------------------------------------------------=== #


def _normal_fill_16(
    data: MutMemPtr[DType.float32], mean: Float32, std: Float32
):
    """In-place Box-Muller over a window of 16 uniforms -> 16 gaussians."""
    for t in range(8):
        var u1 = Float32(1.0) - data[t]
        var u2 = data[t + 8]
        var radius = sqrt(Float32(-2.0) * log(u1 + BOX_MULLER_EPSILON))
        var theta = Float32(Float64(2.0) * Float64(pi) * Float64(u2))
        data[t] = radius * cos(theta) * std + mean
        data[t + 8] = radius * sin(theta) * std + mean


def normal_(
    mut rng: MT19937,
    data: MutMemPtr[DType.float32],
    numel: Int,
    mean: Float32,
    std: Float32,
):
    """Fill `data[0:numel]` with N(mean, std**2), matching torch's `normal_`."""
    if numel >= 16:
        for t in range(numel):
            data[t] = rng.randfloat32()
        var i = 0
        while i < numel - 15:
            _normal_fill_16(data + i, mean, std)
            i += 16
        if numel % 16 != 0:
            # Recompute the final 16 values (they overlap the last full block).
            var tail = data + (numel - 16)
            for j in range(16):
                tail[j] = rng.randfloat32()
            _normal_fill_16(tail, mean, std)
    else:
        # numel < 16 draws float64 uniforms two-at-a-time (one cos, one sin).
        var has_next = False
        var next_sample = Float64(0.0)
        for t in range(numel):
            if has_next:
                data[t] = Float32(next_sample * Float64(std) + Float64(mean))
                has_next = False
                continue
            var u1 = Float32(rng.randfloat64())
            var u2 = Float32(rng.randfloat64())
            var radius = sqrt(
                Float32(-2.0) * log(Float32(1.0) - u2 + BOX_MULLER_EPSILON)
            )
            var theta = Float32(Float64(2.0) * Float64(pi) * Float64(u1))
            next_sample = Float64(radius * sin(theta))
            has_next = True
            data[t] = radius * cos(theta) * std + mean
