"""Model FLOPs Utilization (MFU) estimation, ported from llm.c's `llmc/mfu.h`.

We estimate the GPU's promised peak TFLOPs by looking the device up in a small
hand-maintained database of NVIDIA cards (keyed by the exact name reported by the
driver), scaled from a per-architecture tensor-core archetype by the card's core
count and clock. MFU is then achieved-FLOPs / promised-FLOPs for the step.

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


def _gpu_db() -> List[GPUEntry]:
    """The NVIDIA card database, keyed by exact driver name (matches llm.c)."""
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
    return db^


def get_flops_promised(device: String, use_bf16: Bool) -> Float32:
    """Promised peak TFLOPs (units of 1e12) for `device` at the given precision,
    or -1 if the device is unknown / lacks data for the precision. The bf16 build
    uses the bf16-with-fp32-accumulate column; the fp32 build uses TF32."""
    var db = _gpu_db()
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
