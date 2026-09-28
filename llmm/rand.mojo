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


# MT19937 parameters, named as in Matsumoto & Nishimura (1998) and llm.c's
# rand.h. Any change breaks bit-parity with torch.manual_seed / llm.c.

# State size in 32-bit words (n), and the middle-word offset of the twist (m).
comptime MT19937_N = 624
comptime MT19937_M = 397
# The twist joins the top bit of one word with the low 31 bits of the next.
comptime MT19937_UPPER_MASK = UInt32(0x80000000)
comptime MT19937_LOWER_MASK = UInt32(0x7FFFFFFF)
# Twist matrix A's last row, XORed in when the joined word is odd.
comptime MT19937_MATRIX_A = UInt32(0x9908B0DF)

# Seeding: state[j] = f * (state[j-1] ^ (state[j-1] >> 30)) + j (Knuth's f).
comptime MT19937_INIT_MULTIPLIER = UInt32(1812433253)
comptime MT19937_INIT_SHIFT = 30

# Output tempering: shifts u, s, t, l and masks b, c.
comptime MT19937_TEMPER_U = 11
comptime MT19937_TEMPER_S = 7
comptime MT19937_TEMPER_B = UInt32(0x9D2C5680)
comptime MT19937_TEMPER_T = 15
comptime MT19937_TEMPER_C = UInt32(0xEFC60000)
comptime MT19937_TEMPER_L = 18

# Width of one draw. randint64 packs two, the first into the high word;
# rng_device and the sampler reuse it for their own 64/32-bit splits.
comptime U32_BITS = 32

# Uniform floats keep as many bits of a draw as the float's significand holds
# (implicit bit included), then scale by 2^-bits: every value in [0, 1) is
# exact. MT19937 and llmm/rng_device.mojo keep the LOW bits (torch/llm.c
# rand.h); llmm/sampler.mojo keeps the HIGH bits (llm.c sampler.h). Both
# derive from FLOAT32_SIGNIFICAND_BITS.
comptime FLOAT32_SIGNIFICAND_BITS = 24
comptime FLOAT64_SIGNIFICAND_BITS = 53
comptime FLOAT32_UNIFORM_MASK = UInt32((1 << FLOAT32_SIGNIFICAND_BITS) - 1)
comptime FLOAT64_UNIFORM_MASK = UInt64((1 << FLOAT64_SIGNIFICAND_BITS) - 1)
comptime FLOAT32_UNIFORM_STEP = Float32(1.0) / Float32(
    1 << FLOAT32_SIGNIFICAND_BITS
)
comptime FLOAT64_UNIFORM_STEP = Float64(1.0) / Float64(
    1 << FLOAT64_SIGNIFICAND_BITS
)

# torch's normal_ transforms uniforms in windows of 16: Box-Muller pairs
# element t with t + 8 inside each window. Tensors shorter than a window take
# a separate float64 path.
comptime NORMAL_WINDOW = 16
comptime NORMAL_HALF_WINDOW = NORMAL_WINDOW // 2

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
        for _ in range(MT19937_N):
            self.state.append(UInt32(0))
        self.left = 1
        self.next = 0
        self.seed(seed)

    def seed(mut self, seed: UInt32):
        """Equivalent to `manual_seed`."""
        self.state[0] = seed
        for j in range(1, MT19937_N):
            var prev = self.state[j - 1]
            self.state[j] = MT19937_INIT_MULTIPLIER * (
                prev ^ (prev >> MT19937_INIT_SHIFT)
            ) + UInt32(j)
        self.left = 1
        self.next = 0

    def _next_state(mut self):
        self.left = MT19937_N
        self.next = 0
        var y: UInt32
        for j in range(MT19937_N - MT19937_M):
            y = (self.state[j] & MT19937_UPPER_MASK) | (
                self.state[j + 1] & MT19937_LOWER_MASK
            )
            self.state[j] = (
                self.state[j + MT19937_M]
                ^ (y >> 1)
                ^ (MT19937_MATRIX_A if (y & 1) else UInt32(0))
            )
        for j in range(MT19937_N - MT19937_M, MT19937_N - 1):
            y = (self.state[j] & MT19937_UPPER_MASK) | (
                self.state[j + 1] & MT19937_LOWER_MASK
            )
            self.state[j] = (
                self.state[j + (MT19937_M - MT19937_N)]
                ^ (y >> 1)
                ^ (MT19937_MATRIX_A if (y & 1) else UInt32(0))
            )
        y = (self.state[MT19937_N - 1] & MT19937_UPPER_MASK) | (
            self.state[0] & MT19937_LOWER_MASK
        )
        self.state[MT19937_N - 1] = (
            self.state[MT19937_M - 1]
            ^ (y >> 1)
            ^ (MT19937_MATRIX_A if (y & 1) else UInt32(0))
        )

    def randint32(mut self) -> UInt32:
        self.left -= 1
        if self.left <= 0:
            self._next_state()
        var y = self.state[self.next]
        self.next += 1
        y ^= y >> MT19937_TEMPER_U
        y ^= (y << MT19937_TEMPER_S) & MT19937_TEMPER_B
        y ^= (y << MT19937_TEMPER_T) & MT19937_TEMPER_C
        y ^= y >> MT19937_TEMPER_L
        return y

    def randint64(mut self) -> UInt64:
        # First draw supplies the high 32 bits (matches llm.c's evaluation).
        var hi = UInt64(self.randint32())
        var lo = UInt64(self.randint32())
        return (hi << U32_BITS) | lo

    def randfloat32(mut self) -> Float32:
        return (
            Float32(Int(self.randint32() & FLOAT32_UNIFORM_MASK))
            * FLOAT32_UNIFORM_STEP
        )

    def randfloat64(mut self) -> Float64:
        return (
            Float64(Int(self.randint64() & FLOAT64_UNIFORM_MASK))
            * FLOAT64_UNIFORM_STEP
        )


# ===----------------------------------------------------------------------=== #
# Permutation, matching llm.c's random_permutation in rand.h
# ===----------------------------------------------------------------------=== #


def random_permutation(mut arr: List[Int], mut rng: MT19937):
    """Fisher-Yates over the mt19937 stream, draw-for-draw identical to
    llm.c's random_permutation(); with the same seed, a shuffled dataloader
    visits batches in llm.c's exact order."""
    var n = len(arr)
    for i in range(n - 1, 0, -1):
        var j = Int(rng.randint32() % UInt32(i + 1))
        var tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp


# ===----------------------------------------------------------------------=== #
# Gaussian sampling (Box-Muller), matching torch.normal_
# ===----------------------------------------------------------------------=== #


def _normal_fill_window(
    data: MutMemPtr[DType.float32], mean: Float32, std: Float32
):
    """In-place Box-Muller over one window of uniforms -> gaussians."""
    for t in range(NORMAL_HALF_WINDOW):
        var u1 = Float32(1.0) - data[unsafe_offset=t]
        var u2 = data[unsafe_offset=t + NORMAL_HALF_WINDOW]
        var radius = sqrt(Float32(-2.0) * log(u1 + BOX_MULLER_EPSILON))
        var theta = Float32(Float64(2.0) * Float64(pi) * Float64(u2))
        data[unsafe_offset=t] = radius * cos(theta) * std + mean
        data[unsafe_offset=t + NORMAL_HALF_WINDOW] = (
            radius * sin(theta) * std + mean
        )


def normal_(
    mut rng: MT19937,
    data: MutMemPtr[DType.float32],
    numel: Int,
    mean: Float32,
    std: Float32,
):
    """Fill `data[0:numel]` with N(mean, std**2), matching torch's `normal_`."""
    if numel >= NORMAL_WINDOW:
        for t in range(numel):
            data[unsafe_offset=t] = rng.randfloat32()
        var i = 0
        while i <= numel - NORMAL_WINDOW:
            _normal_fill_window(data.unsafe_offset(i), mean, std)
            i += NORMAL_WINDOW
        if numel % NORMAL_WINDOW != 0:
            # Recompute the final window (it overlaps the last full block).
            var tail = data.unsafe_offset((numel - NORMAL_WINDOW))
            for j in range(NORMAL_WINDOW):
                tail[unsafe_offset=j] = rng.randfloat32()
            _normal_fill_window(tail, mean, std)
    else:
        # Below one window: float64 uniforms two-at-a-time (one cos, one sin).
        var has_next = False
        var next_sample = Float64(0.0)
        for t in range(numel):
            if has_next:
                data[unsafe_offset=t] = Float32(
                    next_sample * Float64(std) + Float64(mean)
                )
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
            data[unsafe_offset=t] = radius * cos(theta) * std + mean
