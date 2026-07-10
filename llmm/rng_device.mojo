# ===----------------------------------------------------------------------=== #
# rng_device.mojo — counter-based, stateless device RNG + a stochastic-rounding
# fp32 -> bf16 cast built on top of it.
#
# See docs/ai/fp8_training_design.md §3/§6 "Chunk G". This is the reusable
# primitive consumed by:
#   - llmm/adamw.mojo's master-weight store (SR instead of plain RNE,
#     behind `-D LLMM_SR_MASTER=1`; closes the `~line 141` TODO).
#   - the future `quantize()` seam in llmm/lowp.mojo (Chunks B/C, owned by
#     other agents), for fp8/fp4 narrowing casts.
#
# `llmm/rand.mojo` (MT19937) is the *host-only*, stateful generator used for
# from-scratch weight init (bit-matches torch/llm.c). This file is a
# different animal on purpose: a *counter-based* generator with **no shared
# state** — same (seed, counter, stream) always produces the same bits, on
# host or device, and two threads never contend over a state pointer. That
# statelessness is exactly the property a GPU kernel launched over millions
# of elements/steps needs (deterministic replay, no cross-thread races).
#
# ## RNG choice: Widynski's Squares (squares32, 4 rounds) over Philox4x32-10
#
# Squares (B. Widynski, "Squares: A Fast Counter-Based RNG",
# arXiv:2004.06278) is chosen over Philox4x32-10 for three reasons:
#
#   1. **Toolchain-risk minimization.** Squares is ~10 lines of 64-bit
#      multiply/add/"swap-halves" — the same class of wraparound integer
#      arithmetic `llmm/rand.mojo`'s MT19937 already exercises correctly on
#      this box (see its "UInt32 arithmetic wraps modulo 2**32" comment; the
#      same holds for UInt64). Philox needs a 32x32->64 "multiply-high" split
#      into two 32-bit lanes, replicated across 4 lanes x 10 rounds — more
#      surface area for a GPU-codegen surprise on an aarch64/sm_121 toolchain
#      that has already shown narrow-dtype GPU codegen gaps
#      (`tests/probe_fp8/probe2_gpu_kernel.mojo`: fp8 arithmetic *casts*
#      compile fine standalone but fail once fed into further arithmetic on
#      this GPU target). Squares avoids that whole class of risk by using
#      only widely-exercised 64-bit integer ops.
#   2. **Cost.** One draw is 4 rounds of (square + add + swap) on a single
#      64-bit register, versus Philox's 10 rounds x 4 lanes. This matters
#      because the AdamW SR seam draws once per parameter element per
#      optimizer step — a bandwidth-bound kernel (see the
#      llmmojo-slow-at-long-seqlen parity work; alignment/extra-op cost shows
#      up directly in step time there).
#   3. **Sufficient statistical quality for SR.** Stochastic rounding only
#      needs an approximately-uniform, decorrelated-across-counter bit
#      stream — it is diffusing quantization error over many steps, not
#      running a Monte-Carlo integral that needs TestU01/BigCrush-grade
#      equidistribution. Widynski reports Squares passes BigCrush, which is
#      already well beyond what SR needs; either generator would work, and
#      Squares is the simpler, cheaper, lower-risk one for this toolchain.
#
# Both generators are counter-based/stateless; the choice is implementation
# risk and cost, not the determinism/race-freedom property (both have it).
#
# ## Contract (what other chunks — and quantize()'s SR seam — can rely on)
#
#   - `rng_key(seed, stream=0) -> UInt64`: expand a 64-bit seed (+ a stream id
#     that separates independent random substreams, e.g. "adamw master store"
#     vs. a future "fp8 grad quantize") into a Squares key. Call once per
#     kernel launch / call site, reuse across all elements.
#   - `squares32(counter, key) -> UInt32`: the core generator. `counter` is
#     normally a per-element index (or a mix of element index and step).
#   - `rng_u32(seed, counter, stream=0) -> UInt32`: one-call convenience
#     (`squares32(counter, rng_key(seed, stream))`) for call sites that don't
#     need to amortize the key derivation across many elements.
#   - `rng_uniform01(seed, counter, stream=0) -> Float32`: uniform draw in
#     [0, 1), 24-bit precision (same construction as `llmm/rand.mojo`'s
#     `randfloat32`).
#   - `sr_round_bits(x, rand_bits) -> BFloat16`: pure, RNG-agnostic
#     stochastic-rounding narrowing of an already-drawn 32-bit random value
#     (testable independent of the RNG; also the fast path for hot loops that
#     already have a `UInt32` in hand, e.g. from `squares32` directly).
#   - `sr_cast_bf16(x, seed, counter, stream=0) -> BFloat16`: one-call
#     convenience combining `rng_u32` + `sr_round_bits`.
#
# All functions here are plain `def`s over scalar `UInt64`/`UInt32`/`Float32`
# math — no `is_gpu[target]()`/`is_cpu[target]()` branching, unlike the
# fp8/bf16 *kernel* dispatch elsewhere in this tree. There is nothing
# dtype-exotic here (no fp8, no "cpu" instantiation of a GPU-only kernel
# body), so the AArch64-codegen-crash landmine
# (`bf16-build-needs-gpu-only-dispatch`) does not apply: the exact same
# function bodies compile and run correctly as ordinary host code *and*
# inline correctly into a `compile_function`/`enqueue_function` GPU kernel.
# That is also why there is a single implementation rather than a duplicated
# "host reference" — `tests/test_rng_sr.mojo` calls these same functions
# directly on host as the reference, and separately launches a GPU kernel
# that calls them, to prove the toolchain lowers them identically on both
# targets (the real risk on this box, per the landmine notes above).
# ===----------------------------------------------------------------------=== #

from std.memory import bitcast


# ===----------------------------------------------------------------------=== #
# Key derivation (seed, stream) -> Squares key, via splitmix64.
# ===----------------------------------------------------------------------=== #


@always_inline
def _splitmix64_step(state: UInt64) -> UInt64:
    """One splitmix64 output step (Vigna's public-domain construction). Used
    only to mix a `(seed, stream)` pair into a well-distributed 64-bit key —
    not used as a general-purpose RNG here.
    """
    var z: UInt64 = state
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


@always_inline
def rng_key(seed: UInt64, stream: UInt64 = 0) -> UInt64:
    """Derive a Squares "key" from a `(seed, stream)` pair.

    `stream` separates independent random substreams under the same `seed`
    (e.g. one stream per SR call site) without needing a second counter
    dimension. The key is forced odd, matching Squares' requirement that the
    key behave as an odd Weyl-sequence-like increment; splitmix64's output is
    already well-mixed (passes standard empirical RNG batteries as a mixer),
    so parity-forcing its output is sufficient for SR's accuracy needs (see
    the module docstring's point 3 — this is not claimed to match the
    exhaustively-vetted key tables in Widynski's paper, which matter for
    exact-equidistribution guarantees SR does not need).
    """
    var state = seed ^ (
        stream * UInt64(0x9E3779B97F4A7C15) + UInt64(0xD1B54A32D192ED03)
    )
    return _splitmix64_step(state) | UInt64(1)


# ===----------------------------------------------------------------------=== #
# Squares core (Widynski, squares32 / 4-round variant).
# ===----------------------------------------------------------------------=== #


@always_inline
def squares32(counter: UInt64, key: UInt64) -> UInt32:
    """Widynski's Squares RNG, 4-round 32-bit-output variant.

    Deterministic, stateless: the same `(counter, key)` always produces the
    same output, on host or device, with no shared/global state — safe to
    call from arbitrarily many concurrent GPU threads keyed by their own
    `counter` (typically a per-element index).
    """
    var x: UInt64 = counter * key
    var y: UInt64 = x
    var z: UInt64 = y + key

    x = x * x + y
    x = (x >> 32) | (x << 32)  # round 1: swap 32-bit halves

    x = x * x + z
    x = (x >> 32) | (x << 32)  # round 2

    x = x * x + y
    x = (x >> 32) | (x << 32)  # round 3

    x = x * x + z  # round 4 — take the high 32 bits directly, no swap needed
    return UInt32(x >> 32)


@always_inline
def rng_u32(seed: UInt64, counter: UInt64, stream: UInt64 = 0) -> UInt32:
    """One-call convenience: `squares32(counter, rng_key(seed, stream))`.

    Hot loops that draw many values under the same `(seed, stream)` should
    instead call `rng_key` once and `squares32` per element, to avoid
    re-deriving the key every draw.
    """
    return squares32(counter, rng_key(seed, stream))


@always_inline
def rng_uniform01(seed: UInt64, counter: UInt64, stream: UInt64 = 0) -> Float32:
    """Uniform draw in [0, 1), 24-bit precision — same construction as
    `llmm/rand.mojo`'s `MT19937.randfloat32` (top 24 bits of a u32, scaled by
    2^-24), for consistency with the rest of the tree.
    """
    var bits = rng_u32(seed, counter, stream)
    return Float32(Int(bits & UInt32(0xFFFFFF))) * (
        Float32(1.0) / Float32(1 << 24)
    )


# ===----------------------------------------------------------------------=== #
# Stochastic-rounding cast: fp32 -> bf16.
#
# bf16's bit pattern is exactly the top 16 bits of the fp32 bit pattern (same
# sign+exponent field, mantissa truncated 23 -> 7 bits). Adding a uniform
# random value drawn from [0, 2^16) to the low 16 (discarded) bits before
# truncating is the standard stochastic-rounding-via-integer-dither trick:
# for a fixed sign, the fp32 bit pattern is monotonic in magnitude, so
# "carries into bit 16" <-> "rounds away from zero" and "no carry" <-> "rounds
# toward zero", each with probability proportional to how close `x` sits to
# that neighbor. This is unbiased in the bit-pattern domain by construction
# (see tests/test_rng_sr.mojo's derivation + unbiasedness test), and holds
# symmetrically for negative `x` because same-sign IEEE-754 bit patterns are
# magnitude-monotonic (only cross-sign comparisons need the "flip sign bit"
# trick — irrelevant here since we never cross the sign boundary within one
# rounding decision).
# ===----------------------------------------------------------------------=== #

comptime _F32_EXP_MASK = UInt32(0x7F800000)
comptime _LOW16_MASK = UInt32(0xFFFF)


@always_inline
def sr_round_bits(x: Float32, rand_bits: UInt32) -> BFloat16:
    """Stochastically round `x` to bf16 using an already-drawn `rand_bits`
    (only its low 16 bits are consumed — the ULP gap being rounded into is
    exactly 16 bits wide for fp32 -> bf16).

    NaN/Inf (exponent field all-ones) bypass dithering and go through a
    plain narrowing cast: Inf has a zero mantissa, and dithering would
    fabricate a nonzero mantissa with the exponent still all-ones — i.e.
    silently turn Inf into NaN some fraction of the time. NaN payload bits
    should likewise not be perturbed.
    """
    var bits: UInt32 = bitcast[DType.uint32](x)
    if (bits & _F32_EXP_MASK) == _F32_EXP_MASK:
        return x.cast[DType.bfloat16]()

    var dither = rand_bits & _LOW16_MASK
    # No overflow risk: `bits` for any finite x is <= 0xFF7FFFFF and
    # `dither` <= 0xFFFF, so `rounded` never wraps UInt32 — the carry we
    # want (into bit 16, and rarely from there into the exponent field on a
    # power-of-two boundary, which is correct rounding-up behavior) is the
    # only carry that can occur.
    var rounded: UInt32 = bits + dither
    var bf_bits = UInt16(rounded >> 16)
    return bitcast[DType.bfloat16](bf_bits)


@always_inline
def sr_cast_bf16(
    x: Float32, seed: UInt64, counter: UInt64, stream: UInt64 = 0
) -> BFloat16:
    """One-call convenience: draw one `u32` from `(seed, counter, stream)`
    and stochastically round `x` to bf16 with it.
    """
    return sr_round_bits(x, rng_u32(seed, counter, stream))
