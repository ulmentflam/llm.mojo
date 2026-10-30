# ===----------------------------------------------------------------------=== #
# QK-clip (the "Clip" in MuonClip, Kimi K2 2025): after an optimizer step,
# any attention head whose max pre-softmax logit S_max exceeded `tau` gets its
# Q and K projection rows (and the matching bias entries) rescaled by
# sqrt(tau / S_max), so the next step's logits are bounded by tau. The
# per-head maxima come from the causal softmax kernel in llmm/attention.mojo.
# Design: docs/ai/muon_muonclip_implementation_plan.md, section 1.
#
# INTERFACE STUB: the qk-clip track fills in the body; keep the signature.
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceContext

from llmm.memory import ImmutKernelPtr, MutKernelPtr


def qk_clip_apply[
    dtype: DType,
    target: StaticString,
](
    params_ptr: MutKernelPtr[dtype],
    master_ptr: MutKernelPtr[DType.float32],
    has_master: Bool,
    # Element offsets of the [L, 3C, C] qkv weight and the [L, 3C] qkv bias
    # inside the flat parameter buffer (same indexing for master_ptr).
    qkv_weight_offset: Int,
    qkv_bias_offset: Int,
    # [L, NH] fp32 per-(layer, head) max attention logit of the step just
    # taken; entries <= tau are left untouched.
    head_max_ptr: ImmutKernelPtr[DType.float32],
    num_layers: Int,
    num_heads: Int,
    channels: Int,
    tau: Scalar[DType.float32],
    ctx: DeviceContext,
) raises:
    raise Error("qk_clip_apply: not implemented yet (llmm/qk_clip.mojo stub)")
