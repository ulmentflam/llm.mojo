# ===----------------------------------------------------------------------=== #
# calibrate_fp8_scales.mojo — A1 calibration tool (speedrun A1: static/
# calibrated FP8 scales, docs/ai/speedrun_techniques_research.md).
#
# Runs N steps (default 20) of ordinary fp8 training with the EXISTING
# delayed-scaling (`AmaxState`) path, reading back every per-(site,role,layer)
# `AmaxState.scale` (device-resident fp32 scalars, `llmm/amax.mojo`) AFTER
# EVERY step and tracking a running MIN across the whole run (not just a
# single readback at the end — see the "why every-step, not just final"
# note below). All reads are host syncs off the training hot path (this
# tool, not the trainer, pays them) — fine for a one-shot offline
# calibration utility.
#
# Why every-step running-min, not "read once at the end" (first version of
# this tool did the latter and produced an UNSAFE table — see docs/ai/
# ai_assisted_optimizations_and_benchmarks.md's A1 calibration entry for the
# full incident): `AmaxState`'s delayed-scaling ring buffer only remembers
# the last `amax_history_len=16` steps by construction (`llmm/amax.mojo`) —
# reading `.scale` only after step 20 sees the max amax over steps ~4-19
# and has ALREADY FORGOTTEN steps 0-3. The single most extreme amax in a
# short run is very often the FIRST step or two off a fresh checkpoint
# (loss/gradients have not yet settled onto the tiny fine-tuning dataset),
# so an end-of-run-only readback can silently miss exactly the outlier a
# safety margin is supposed to cover. Reading every step and keeping a
# running min-over-time (in addition to the min-over-layers the design
# always wanted) fixes this: the reported constant is now the smallest
# scale ANY layer needed at ANY point during the whole calibration run,
# which is the correct "worst case observed" input to the safety-factor
# margin.
#
# For each of the 12 (site, role) pairs — site in {qkv, attn_proj, fc, proj},
# role in {input, weight, doutput} — this tool has `num_layer` independent
# `AmaxState`s (one per transformer layer). The static-scale design (A1)
# wants ONE constant per (site, role), shared across all layers (mirroring
# modded-nanogpt's per-tensor-role constants, not per-layer-instance
# constants — see the research doc's "hardcoded constant per-tensor scales"
# finding). This tool picks the MIN scale across (layers x steps) for each
# (site, role) — i.e. the scale belonging to whichever (layer, step) saw
# the LARGEST amax — so that using this one constant everywhere is safe
# for every layer at every point in training, not just the "typical" one.
# A configurable safety factor (default 2.0, i.e. "amax*2" headroom) is
# then divided into that min-scale before it is printed, so the emitted
# table already has margin baked in.
#
# Usage:
#   pixi run -e cuda mojo run -I . calibrate_fp8_scales.mojo \
#       [checkpoint_or_descriptor] [steps] [batch_size] [seq_len] [safety_factor]
#
#   checkpoint_or_descriptor  default "gpt2_124M_bf16.bin" (checkpoint-init,
#                             the campaign's standard invocation). Pass a
#                             model descriptor like "d36" for a from-scratch
#                             (random-init) calibration run at another width
#                             — scales only need the right order of
#                             magnitude, per the mission's own allowance.
#   steps                     default 20.
#   batch_size / seq_len      default 4 / 1024 (the campaign's B=4 T=1024).
#   safety_factor             default 2.0 (scale_static = min_scale / factor,
#                             i.e. treats amax as if it were `factor`x larger).
#
# Must be built with `-D LLMM_PRECISION=fp8` (same as dump_grads_gpt2.mojo /
# train_gpt2.mojo's fp8 build) — this tool has no bf16/fp32 mode, it exists
# to calibrate the fp8 delayed-scaling states.
# ===----------------------------------------------------------------------=== #

from std.sys import argv
from std.memory import alloc, UnsafePointer
from std.gpu.host import DeviceContext

from llmm.memory import MutMemPtr
from llmm.lowp import FP8_SPEC
from llmm.amax import AmaxState

from train_gpt2 import GPT2, GPT2_DTYPE, LOWP_ENABLED
from llmm.dataloader import DataLoader


def _read_scale(
    ctx: DeviceContext, state: AmaxState[FP8_SPEC]
) raises -> Float32:
    """One-element device->host readback of `state.scale`."""
    var host_buf = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.enqueue_copy(
        dst_ptr=rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](
            host_buf.unsafe_ptr().as_unsafe_any_origin()
        ),
        src_ptr=rebind[UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]](
            state.scale.unsafe_ptr().as_unsafe_any_origin()
        ),
        size=1,
    )
    ctx.synchronize()
    var v = host_buf[0]
    _ = host_buf^
    return v


def _init_running_min(num_layer: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(num_layer):
        out.append(Float32(1e30))
    return out^


def _update_running_min(
    ctx: DeviceContext,
    states: List[AmaxState[FP8_SPEC]],
    mut running_min: List[Float32],
) raises -> None:
    for i in range(len(states)):
        var s = _read_scale(ctx, states[i])
        if s < running_min[i]:
            running_min[i] = s


def _finalize_site_role(
    name: String,
    running_min: List[Float32],
    safety_factor: Float32,
) -> Float32:
    """Prints every layer's worst-case-over-the-whole-run scale for one
    (site, role), returns the calibrated (min-over-layers-and-steps /
    safety_factor) constant."""
    var min_scale = Float32(1e30)
    var line = String("  ") + name + ": ["
    for i in range(len(running_min)):
        if running_min[i] < min_scale:
            min_scale = running_min[i]
        if i > 0:
            line += ", "
        line += String(running_min[i])
    line += "]"
    print(line)
    var calibrated = min_scale / safety_factor
    print(
        "    -> min(over layers & steps)="
        + String(min_scale)
        + "  calibrated(min/"
        + String(safety_factor)
        + ")="
        + String(calibrated)
    )
    return calibrated


def main() raises:
    var args = argv()
    var checkpoint = String("gpt2_124M_bf16.bin")
    var steps = 20
    var batch_size = 4
    var seq_len = 1024
    var safety_factor = Float32(2.0)
    if len(args) > 1:
        checkpoint = String(args[1])
    if len(args) > 2:
        steps = atol(String(args[2]))
    if len(args) > 3:
        batch_size = atol(String(args[3]))
    if len(args) > 4:
        seq_len = atol(String(args[4]))
    if len(args) > 5:
        safety_factor = Float32(atof(String(args[5])))

    comptime assert LOWP_ENABLED, (
        "calibrate_fp8_scales must be built with -D LLMM_PRECISION=fp8 (or"
        " fp4, though this tool's AmaxState reads are fp8-specific)"
    )

    var ctx = DeviceContext()
    var model = GPT2["gpu", 1, False](
        checkpoint,
        rank=0,
        zero_stage=0,
        ctx=ctx,
    )

    print(
        "calibrating fp8 static scales: checkpoint/descriptor="
        + checkpoint
        + " steps="
        + String(steps)
        + " B="
        + String(batch_size)
        + " T="
        + String(seq_len)
        + " safety_factor="
        + String(safety_factor)
        + " (running min-over-steps, not just final step)"
    )

    var train_tokens = "./data/.tinyshakespeare/tiny_shakespeare_train.bin"
    var train_loader = DataLoader(train_tokens, batch_size, seq_len, 0, 1)

    var num_layer = model.config.num_layer
    var qkv_input_min = _init_running_min(num_layer)
    var qkv_weight_min = _init_running_min(num_layer)
    var qkv_doutput_min = _init_running_min(num_layer)
    var attn_proj_input_min = _init_running_min(num_layer)
    var attn_proj_weight_min = _init_running_min(num_layer)
    var attn_proj_doutput_min = _init_running_min(num_layer)
    var fc_input_min = _init_running_min(num_layer)
    var fc_weight_min = _init_running_min(num_layer)
    var fc_doutput_min = _init_running_min(num_layer)
    var proj_input_min = _init_running_min(num_layer)
    var proj_weight_min = _init_running_min(num_layer)
    var proj_doutput_min = _init_running_min(num_layer)

    for step in range(steps):
        train_loader.next_batch()
        model.forward(
            train_loader.inputs, train_loader.targets, batch_size, seq_len
        )
        model.backward(1, 0, step)
        _ = model.calculate_grad_norm()
        model.update(UInt32(step + 1), Float32(3e-4))

        # Running-min readback (see the module comment above for why this
        # must happen every step, not just once at the end).
        _update_running_min(ctx, model.lowp_state.qkv_input, qkv_input_min)
        _update_running_min(ctx, model.lowp_state.qkv_weight, qkv_weight_min)
        _update_running_min(ctx, model.lowp_state.qkv_doutput, qkv_doutput_min)
        _update_running_min(
            ctx, model.lowp_state.attn_proj_input, attn_proj_input_min
        )
        _update_running_min(
            ctx, model.lowp_state.attn_proj_weight, attn_proj_weight_min
        )
        _update_running_min(
            ctx, model.lowp_state.attn_proj_doutput, attn_proj_doutput_min
        )
        _update_running_min(ctx, model.lowp_state.fc_input, fc_input_min)
        _update_running_min(ctx, model.lowp_state.fc_weight, fc_weight_min)
        _update_running_min(ctx, model.lowp_state.fc_doutput, fc_doutput_min)
        _update_running_min(ctx, model.lowp_state.proj_input, proj_input_min)
        _update_running_min(ctx, model.lowp_state.proj_weight, proj_weight_min)
        _update_running_min(
            ctx, model.lowp_state.proj_doutput, proj_doutput_min
        )
    ctx.synchronize()
    print("done training; loss=" + String(model.mean_loss))

    print(
        "\n=== per-(site,role) worst-case scale (min over layers AND steps) ==="
    )
    var qkv_input = _finalize_site_role(
        "qkv_input", qkv_input_min, safety_factor
    )
    var qkv_weight = _finalize_site_role(
        "qkv_weight", qkv_weight_min, safety_factor
    )
    var qkv_doutput = _finalize_site_role(
        "qkv_doutput", qkv_doutput_min, safety_factor
    )
    var attn_proj_input = _finalize_site_role(
        "attn_proj_input", attn_proj_input_min, safety_factor
    )
    var attn_proj_weight = _finalize_site_role(
        "attn_proj_weight", attn_proj_weight_min, safety_factor
    )
    var attn_proj_doutput = _finalize_site_role(
        "attn_proj_doutput", attn_proj_doutput_min, safety_factor
    )
    var fc_input = _finalize_site_role("fc_input", fc_input_min, safety_factor)
    var fc_weight = _finalize_site_role(
        "fc_weight", fc_weight_min, safety_factor
    )
    var fc_doutput = _finalize_site_role(
        "fc_doutput", fc_doutput_min, safety_factor
    )
    var proj_input = _finalize_site_role(
        "proj_input", proj_input_min, safety_factor
    )
    var proj_weight = _finalize_site_role(
        "proj_weight", proj_weight_min, safety_factor
    )
    var proj_doutput = _finalize_site_role(
        "proj_doutput", proj_doutput_min, safety_factor
    )

    print(
        "\n=== paste-ready Mojo comptime table (calibrated "
        + checkpoint
        + ", "
        + String(steps)
        + " steps, safety_factor="
        + String(safety_factor)
        + ") ==="
    )
    print("comptime QKV_INPUT = Float32(" + String(qkv_input) + ")")
    print("comptime QKV_WEIGHT = Float32(" + String(qkv_weight) + ")")
    print("comptime QKV_DOUTPUT = Float32(" + String(qkv_doutput) + ")")
    print("comptime ATTN_PROJ_INPUT = Float32(" + String(attn_proj_input) + ")")
    print(
        "comptime ATTN_PROJ_WEIGHT = Float32(" + String(attn_proj_weight) + ")"
    )
    print(
        "comptime ATTN_PROJ_DOUTPUT = Float32("
        + String(attn_proj_doutput)
        + ")"
    )
    print("comptime FC_INPUT = Float32(" + String(fc_input) + ")")
    print("comptime FC_WEIGHT = Float32(" + String(fc_weight) + ")")
    print("comptime FC_DOUTPUT = Float32(" + String(fc_doutput) + ")")
    print("comptime PROJ_INPUT = Float32(" + String(proj_input) + ")")
    print("comptime PROJ_WEIGHT = Float32(" + String(proj_weight) + ")")
    print("comptime PROJ_DOUTPUT = Float32(" + String(proj_doutput) + ")")
