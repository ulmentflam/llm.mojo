"""Model FLOPs Utilization (MFU) estimation, ported from llm.c's `llmc/mfu.h`.

We estimate the GPU's promised peak TFLOPs by looking the device up in a small
hand-maintained database (NVIDIA cards keyed by exact driver name; Apple Silicon
keyed by substring of Metal MTLDevice.name), scaled from a per-architecture
tensor-core archetype by the card's core count and clock. MFU is then
achieved-FLOPs / promised-FLOPs for the step.

Devices not in the table (or running on CPU) return -1 => "unknown", exactly as
llm.c does; add an entry below for a new card.
"""


# ===----------------------------------------------------------------------=== #
# Per-architecture tensor-core archetypes (TFLOPs).
#   tf32      = TF32 tensor-core (used for the fp32 build's MFU)
#   bf16_32   = bf16 with fp32 accumulate (used for the bf16 build's MFU)
#   cores     = spec-sheet tensor-core count for the archetype
#   clock_mhz = spec-sheet boost clock for the archetype
# A specific card scales these by (its cores / archetype cores) and
# (its clock / archetype clock). Matches llm.c's PerfData + GPUEntry.
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct GPUEntry(Copyable, Movable):
    var name: String
    var tf32: Float32
    var bf16_32: Float32
    var cores: Float32
    var clock_mhz: Float32
    var new_cores: Float32
    var new_mhz: Float32


# Archetype constants: (tf32, bf16_32, cores, clock_mhz).
comptime VOLTA_TF32 = Float32(125.0)
comptime VOLTA_BF16 = Float32(-1.0)
comptime VOLTA_CORES = Float32(640.0)
comptime VOLTA_CLOCK = Float32(1530.0)

comptime AMPERE_DC_TF32 = Float32(156.0)
comptime AMPERE_DC_BF16 = Float32(312.0)
comptime AMPERE_DC_CORES = Float32(432.0)
comptime AMPERE_DC_CLOCK = Float32(1410.0)

comptime AMPERE_CONS_TF32 = Float32(40.0)
comptime AMPERE_CONS_BF16 = Float32(80.0)
comptime AMPERE_CONS_CORES = Float32(336.0)
comptime AMPERE_CONS_CLOCK = Float32(1860.0)

comptime HOPPER_TF32 = Float32(378.0)
comptime HOPPER_BF16 = Float32(756.0)
comptime HOPPER_CORES = Float32(456.0)
comptime HOPPER_CLOCK = Float32(1620.0)

comptime ADA_TF32 = Float32(82.6)
comptime ADA_BF16 = Float32(165.2)
comptime ADA_CORES = Float32(512.0)
comptime ADA_CLOCK = Float32(2520.0)

# NVIDIA GB10 (Grace Blackwell, in DGX Spark). The datasheet headlines
# "1 petaFLOP of AI performance at FP4 with sparsity" => 1000 TFLOPS FP4 sparse
# => 500 TFLOPS FP4 dense. NVIDIA does not publish the per-precision dense rates,
# so we derive them from Blackwell's standard tensor-core ratios
# (FP4 : FP8 : BF16 : TF32 = 8 : 4 : 2 : 1, dense):
#   BF16 with fp32 accumulate = 500 / 4 = 125 TFLOPS
#   TF32                      = 500 / 8 = 62.5 TFLOPS
# Unlike the other archetypes (which scale a reference card by cores/clock), GB10
# is a single known part, so its db entry sets cores/clock == new_cores/new_mhz
# (identity scaling) and these peaks are used directly. Update if NVIDIA
# publishes exact dense figures.
comptime GB10_TF32 = Float32(62.5)
comptime GB10_BF16 = Float32(125.0)


def _volta(name: String, nc: Float32, mhz: Float32) -> GPUEntry:
    return GPUEntry(
        name, VOLTA_TF32, VOLTA_BF16, VOLTA_CORES, VOLTA_CLOCK, nc, mhz
    )


def _ampere_dc(name: String, nc: Float32, mhz: Float32) -> GPUEntry:
    return GPUEntry(
        name,
        AMPERE_DC_TF32,
        AMPERE_DC_BF16,
        AMPERE_DC_CORES,
        AMPERE_DC_CLOCK,
        nc,
        mhz,
    )


def _ampere_cons(name: String, nc: Float32, mhz: Float32) -> GPUEntry:
    return GPUEntry(
        name,
        AMPERE_CONS_TF32,
        AMPERE_CONS_BF16,
        AMPERE_CONS_CORES,
        AMPERE_CONS_CLOCK,
        nc,
        mhz,
    )


def _hopper(name: String, nc: Float32, mhz: Float32) -> GPUEntry:
    return GPUEntry(
        name, HOPPER_TF32, HOPPER_BF16, HOPPER_CORES, HOPPER_CLOCK, nc, mhz
    )


def _ada(name: String, nc: Float32, mhz: Float32) -> GPUEntry:
    return GPUEntry(name, ADA_TF32, ADA_BF16, ADA_CORES, ADA_CLOCK, nc, mhz)


# ===----------------------------------------------------------------------=== #
# Apple Silicon GPU helpers
#
# Apple GPUs have no dedicated tensor cores; all numeric precisions (fp32,
# fp16, bf16) share the same ALU throughput.  We therefore set tf32 == bf16_32
# to the measured fp32 peak so that both the fp32 and bf16 builds get the same
# denominator — which is conservative (slightly understates bf16 MFU) but
# correct absent published contrary data.
#
# M5 added Neural Accelerators that support mma-style FP16 ops via Metal 4,
# but those are NOT used by this codebase's compute kernels, so we keep the
# same parity assumption for M5.
#
# Peak values: Apple does not publish official TFLOPS figures. Values below
# are measured shader FP32 throughput reported by independent benchmarks:
#   Notebookcheck GPU database: https://www.notebookcheck.net/
#   Nanoreview GPU benchmarks:  https://nanoreview.net/en/gpu/
#   cpu-monkey GPU specs:       https://www.cpu-monkey.com/en/
#   flopper.io GPU specs:       https://flopper.io/gpu/
#   check-mac.com FP32 list:    https://www.check-mac.com/en/benchmark-fp32-3
#
# Derivation formula (for cross-checking):
#   peak_fp32 = gpu_cores × 128 ALUs/core × 2 FLOP/cycle × clock_GHz
# e.g. M4 Max 40-core at ~1.58 GHz → 40×128×2×1.58e9 ≈ 16.2 TFLOPS.
# Theoretical boost-clock estimates (≈1.8 GHz) give ~18.4 TFLOPS; measured
# sustained values from benchmark sites converge around 16.2 TFLOPS, which
# is the figure used here.
#
# Where a chip ships in two GPU-core counts (e.g. M4 Max 32- or 40-core) the
# Metal MTLDevice.name string is identical ("Apple M4 Max") for both variants.
# We use the higher-core-count TFLOPS as the peak, which understates MFU for
# the lower-core variant by a proportional amount — acceptable for a training
# throughput indicator.
#
# Name matching: entries whose name starts with "Apple " use substring matching
# in get_flops_promised (Pass 2).  Entries are ordered Ultra → Max → Pro →
# base within each generation so that a more-specific entry always wins before
# a shorter prefix like "Apple M4" could match "Apple M4 Max".
# ===----------------------------------------------------------------------=== #


def _apple(name: String, fp32_tflops: Float32) -> GPUEntry:
    """Apple Silicon GPU entry with a directly-known FP32 peak TFLOPS.
    Identity scaling (cores == new_cores, clock == new_mhz) means the tf32 /
    bf16_32 values are used as-is.  fp32 == bf16 (no tensor cores)."""
    return GPUEntry(
        name,
        fp32_tflops,
        fp32_tflops,
        Float32(1.0),
        Float32(1.0),
        Float32(1.0),
        Float32(1.0),
    )


def _gpu_db() -> List[GPUEntry]:
    """GPU database: NVIDIA cards keyed by exact driver name (matches llm.c),
    followed by Apple Silicon entries keyed by MTLDevice.name substring."""
    var db = List[GPUEntry]()
    db.append(_volta("Tesla V100-SXM2-16GB", 640, 1530))
    db.append(_volta("Tesla V100-PCIE-32GB", 640, 1530))
    db.append(_ampere_dc("NVIDIA A100-PCIE-40GB", 432, 1410))
    db.append(_ampere_dc("NVIDIA A100-PCIE-80GB", 432, 1410))
    db.append(_ampere_dc("NVIDIA A100-SXM4-40GB", 432, 1410))
    db.append(_ampere_dc("NVIDIA A100-SXM4-80GB", 432, 1410))
    db.append(_ampere_cons("NVIDIA RTX A2000", 104, 1200))
    db.append(_ampere_cons("NVIDIA RTX A4000", 192, 1560))
    db.append(_ampere_cons("NVIDIA RTX A4500", 224, 1650))
    db.append(_ampere_cons("NVIDIA RTX A5000", 256, 1695))
    db.append(_ampere_cons("NVIDIA RTX A5500", 320, 1770))
    db.append(_ampere_cons("NVIDIA RTX A6000", 336, 1800))
    db.append(_ampere_cons("NVIDIA GeForce RTX 3090 Ti", 336, 1860))
    db.append(_ampere_cons("NVIDIA GeForce RTX 3090", 328, 1695))
    db.append(_ampere_cons("NVIDIA GeForce RTX 3080 Ti", 320, 1665))
    db.append(_ampere_cons("NVIDIA GeForce RTX 3080", 272, 1710))
    db.append(_ampere_cons("NVIDIA GeForce RTX 3070 Ti", 192, 1770))
    db.append(_ampere_cons("NVIDIA GeForce RTX 3070", 184, 1725))
    db.append(_ampere_cons("NVIDIA GeForce RTX 3060 Ti", 152, 1665))
    db.append(_ampere_cons("NVIDIA GeForce RTX 3060", 112, 1777))
    db.append(_ada("NVIDIA RTX A2000 ADA", 88, 2130))
    db.append(_ada("NVIDIA RTX A4000 ADA", 192, 2175))
    db.append(_ada("NVIDIA RTX A4500 ADA", 224, 2580))
    db.append(_ada("NVIDIA RTX A5000 ADA", 400, 2550))
    db.append(_ada("NVIDIA RTX A5880 ADA", 440, 2460))
    db.append(_ada("NVIDIA RTX A6000 ADA", 568, 2505))
    db.append(_ada("NVIDIA GeForce RTX 4090", 512, 2520))
    db.append(_ada("NVIDIA GeForce RTX 4080 SUPER", 320, 2550))
    db.append(_ada("NVIDIA GeForce RTX 4080", 304, 2505))
    db.append(_ada("NVIDIA GeForce RTX 4070 Ti SUPER", 264, 2610))
    db.append(_ada("NVIDIA GeForce RTX 4070 Ti", 240, 2610))
    db.append(_ada("NVIDIA GeForce RTX 4070 SUPER", 224, 2475))
    db.append(_ada("NVIDIA GeForce RTX 4070", 184, 2475))
    db.append(_ada("NVIDIA GeForce RTX 4060 Ti", 136, 2535))
    db.append(_ada("NVIDIA GeForce RTX 4060", 96, 2460))
    db.append(_hopper("NVIDIA H100 PCIe", 456, 1620))
    db.append(_hopper("NVIDIA H100 80GB HBM3", 528, 1830))
    # Identity scaling (cores == new_cores, clock == new_mhz): use the GB10 peaks
    # directly. The driver reports this device as "NVIDIA GB10".
    db.append(GPUEntry("NVIDIA GB10", GB10_TF32, GB10_BF16, 1.0, 1.0, 1.0, 1.0))

    # ===--- Apple Silicon GPU entries (substring-matched in Pass 2) ---===
    # Ordered Ultra → Max → Pro → base so that the most-specific name is
    # found before any shorter prefix (e.g. "Apple M4" must not match
    # "Apple M4 Max" before "Apple M4 Max" is checked).
    #
    # M-Ultra: two Max dies fused via die-to-die interconnect.
    # Sources: cpu-monkey / notebookcheck / nanoreview; ≈ 2× the Max 38/40-core values.
    db.append(_apple("Apple M1 Ultra", Float32(21.0)))  # 64 GPU cores (2×32)
    db.append(_apple("Apple M2 Ultra", Float32(27.2)))  # 76 GPU cores (2×38)
    db.append(_apple("Apple M3 Ultra", Float32(28.4)))  # 80 GPU cores (2×40)
    # M-Max
    db.append(
        _apple("Apple M1 Max", Float32(10.4))
    )  # 32 GPU cores; 24-core ≈ 7.8 T
    db.append(
        _apple("Apple M2 Max", Float32(13.6))
    )  # 38 GPU cores; 30-core ≈ 10.7 T
    db.append(
        _apple("Apple M3 Max", Float32(14.2))
    )  # 40 GPU cores; 30-core ≈ 10.6 T
    db.append(
        _apple("Apple M4 Max", Float32(16.2))
    )  # 40 GPU cores; 32-core ≈ 14.75 T
    db.append(
        _apple("Apple M5 Max", Float32(16.59))
    )  # 40 GPU cores (flopper.io)
    # M-Pro
    db.append(
        _apple("Apple M1 Pro", Float32(5.2))
    )  # 16 GPU cores; 14-core ≈ 4.5 T
    db.append(
        _apple("Apple M2 Pro", Float32(6.8))
    )  # 19 GPU cores; 16-core ≈ 5.7 T
    db.append(
        _apple("Apple M3 Pro", Float32(7.4))
    )  # 18 GPU cores; 14-core ≈ 5.8 T
    db.append(
        _apple("Apple M4 Pro", Float32(9.2))
    )  # 20 GPU cores (check-mac.com)
    # M-base
    db.append(_apple("Apple M1", Float32(2.6)))  # 8 GPU cores; 7-core ≈ 1.8 T
    db.append(_apple("Apple M2", Float32(3.6)))  # 10 GPU cores; 8-core ≈ 2.9 T
    db.append(_apple("Apple M3", Float32(3.5)))  # 10 GPU cores (cpu-monkey)
    db.append(_apple("Apple M4", Float32(4.26)))  # 10 GPU cores (cpu-monkey)
    db.append(_apple("Apple M5", Float32(4.15)))  # 10 GPU cores (flopper.io)
    return db^


def get_flops_promised(device: String, use_bf16: Bool) -> Float32:
    """Promised peak TFLOPs (units of 1e12) for `device` at the given precision,
    or -1 if the device is unknown / lacks data for the precision. The bf16 build
    uses the bf16-with-fp32-accumulate column; the fp32 build uses TF32.

    Two lookup passes:
      Pass 1 — exact match: used for NVIDIA cards whose driver names are stable
               and fully known (matches llm.c behaviour).
      Pass 2 — substring match for Apple Silicon: Metal MTLDevice.name returns
               e.g. "Apple M4 Max" and our table key IS that string, but we use
               find() rather than == so a future driver suffix (e.g. "Apple M4
               Max GPU") still hits the right entry.  Entries are ordered
               most-specific-first (Ultra > Max > Pro > base) in _gpu_db() so
               the first substring hit is always the correct tier.
    """
    var db = _gpu_db()
    # Pass 1: exact match (NVIDIA and other fully-known names).
    for ref entry in db:
        if entry.name == device:
            var value = entry.bf16_32 if use_bf16 else entry.tf32
            if value < 0.0:
                return Float32(-1.0)
            return (
                value
                * (entry.new_cores / entry.cores)
                * (entry.new_mhz / entry.clock_mhz)
            )
    # Pass 2: substring match for Apple Silicon.
    for ref entry in db:
        if entry.name.startswith("Apple ") and entry.name in device:
            var value = entry.bf16_32 if use_bf16 else entry.tf32
            if value < 0.0:
                return Float32(-1.0)
            return (
                value
                * (entry.new_cores / entry.cores)
                * (entry.new_mhz / entry.clock_mhz)
            )
    return Float32(-1.0)


def estimate_mfu(
    num_parameters: Int,
    num_layers: Int,
    channels: Int,
    seq_len: Int,
    num_tokens: Int,
    dt_seconds: Float64,
    device: String,
    use_bf16: Bool,
) -> Float64:
    """Estimate model FLOPs utilization for one step. Returns -1 if the device's
    peak is unknown. Mirrors llm.c's gpt2_estimate_mfu (ref: Kaplan et al. 2.1):
    flops_per_token = 6*N + 6*L*C*T (weight matmuls + attention)."""
    if dt_seconds <= 0.0:
        return -1.0
    var flops_promised = Float64(get_flops_promised(device, use_bf16))
    if flops_promised < 0.0:
        return -1.0
    var N = Float64(num_parameters)
    var flops_per_token = 6.0 * N + 6.0 * Float64(num_layers) * Float64(
        channels
    ) * Float64(seq_len)
    var flops_per_step = flops_per_token * Float64(num_tokens)
    var flops_achieved = flops_per_step / dt_seconds  # per second
    return flops_achieved / (flops_promised * 1e12)
