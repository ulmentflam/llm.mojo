# ===----------------------------------------------------------------------=== #
# Muon optimizer (Jordan et al. 2024): Nesterov-momentum SGD on 2-D weight
# matrices whose momentum is orthogonalized with 5 Newton-Schulz iterations in
# bf16, scaled per the Moonlight RMS rule (0.2 * sqrt(max(rows, cols))) and
# decoupled weight decay. Design + phase plan:
# docs/ai/muon_muonclip_implementation_plan.md.
#
# INTERFACE STUB. The signatures below are the contract the trainer
# (train_gpt2.mojo) and the pytest bridge (tests/kernels/muon.py) build
# against; the kernel track fills in the bodies. Keep the public names and
# argument orders stable.
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceContext

from llmm.memory import ImmutKernelPtr, MutKernelPtr


@fieldwise_init
struct MuonConfig(Copyable, Movable):
    """All hyperparameters fp32 regardless of parameter dtype (as AdamWConfig).
    """

    var learning_rate: Scalar[DType.float32]
    var momentum: Scalar[DType.float32]  # mu, 0.95 by default
    var weight_decay: Scalar[DType.float32]
    var grad_scale: Scalar[DType.float32]  # global grad-clip factor
    var ns_steps: Int  # Newton-Schulz iterations, 5
    var nesterov: Bool


@fieldwise_init
struct MuonGroup(Copyable, Movable):
    """`count` equally strided row-major [rows, cols] matrices inside the flat
    parameter buffer: matrix i starts at element `offset + i * stride`.
    `stride >= rows * cols`; for the Q/K/V row blocks of the fused
    [L, 3C, C] qkv weight, rows = cols = C and stride = 3*C*C."""

    var offset: Int
    var rows: Int
    var cols: Int
    var count: Int
    var stride: Int


def muon_update[
    dtype: DType,
    target: StaticString,
](
    groups: List[MuonGroup],
    params_ptr: MutKernelPtr[dtype],
    grads_ptr: ImmutKernelPtr[dtype],
    # fp32 master shadow of `params` (same flat indexing); unused when
    # `has_master` is False (pure fp32 training), exactly as llmm.adamw.
    master_ptr: MutKernelPtr[DType.float32],
    has_master: Bool,
    # fp32 momentum, same flat indexing as params (the trainer's m_memory).
    m_ptr: MutKernelPtr[DType.float32],
    t: UInt32,
    config: MuonConfig,
    ctx: DeviceContext,
) raises:
    """One Muon step over every matrix of every group. Scratch (bf16 X/A/B
    buffers and per-matrix norms) is allocated once per process through
    llmm.memory.persistent_device_buffer, so callers hold no scratch state.
    """
    raise Error("muon_update: not implemented yet (llmm/muon.mojo stub)")
