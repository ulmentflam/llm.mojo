# ===----------------------------------------------------------------------=== #
# dump_grads_gpt2.mojo — Chunk E Gate C tooling (docs/ai/fp8_training_design.md
# §6, "Gate E": per-tensor gradient comparison over ALL ~148 parameter
# gradient tensors, NOT a flat atol — see MEMORY.md
# `weak-gates-overrule-nothing`).
#
# Loads the SAME fixed reference batch `test_gpt2.mojo` uses
# (gpt2_124M_debug_state.bin's B=4,T=64 x/y), runs exactly one forward+
# backward step, then dumps every one of the 16 named gradient tensors to
# `<out_dir>/`, split per transformer layer for the 12 per-layer tensor
# classes (ln_1_gamma/beta, qkv_weight/bias, attn_proj_weight/bias,
# ln_2_gamma/beta, fc_weight/bias, proj_weight/bias -- L files each) and as a
# single file for the 4 global tensors (wte, wpe, ln_f_gamma, ln_f_beta) --
# L=12 for the 124M config, so 12*12 + 4 = 148 files, matching the design's
# count exactly. Always dumps as raw fp32 (bf16 host-cast to fp32) so the
# comparison script needs no per-build dtype awareness.
#
# Build this SAME file into both train configurations (`-D LLMM_PRECISION=fp8`
# and `-D LLMM_BF16=1`) and run each against a different `<out_dir>` -- with
# Chunk D's `matmul_fwd_lowp` wiring merged, the fp8 build's forward pass is
# ALSO fp8 (design §1.2's four per-block linears), so this compares the full
# fp8-fwd+fp8-bwd training step against the bf16 reference (the gate's
# intended shape: "fwd will also be fp8 ... gate vs bf16 reference"), not an
# isolated backward-only delta. Compare the two directories with
# `tests/compare_grad_dumps.py`.
#
# Usage: pixi run -e cuda mojo run -I . dump_grads_gpt2.mojo <out_dir>
# ===----------------------------------------------------------------------=== #

from std.sys import argv
from std.os import makedirs
from std.os.path import exists
from std.memory import alloc, UnsafePointer
from std.gpu.host import DeviceContext

from llmm.memory import MutMemPtr
from llmm.io import write_buffer

from train_gpt2 import GPT2, Parameters
from test_gpt2 import read_to_dtype_pointer


def dump_tensor[
    dtype: DType
](
    ctx: DeviceContext, ptr: MutMemPtr[dtype], n: Int, path: String
) raises -> None:
    var host_buf = ctx.enqueue_create_host_buffer[dtype](n)
    ctx.enqueue_copy(
        dst_ptr=rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            host_buf.unsafe_ptr().as_unsafe_any_origin()
        ),
        src_ptr=rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
            ptr.as_unsafe_any_origin()
        ),
        size=n,
    )
    ctx.synchronize()

    var f32 = alloc[Scalar[DType.float32]](n)
    for i in range(n):
        f32[i] = host_buf[i].cast[DType.float32]()

    var file = open(path, "w")
    write_buffer[DType.float32](
        file, rebind[MutMemPtr[DType.float32]](f32.as_unsafe_any_origin()), n
    )
    file.close()
    f32.free()
    _ = host_buf^


def dump_global[
    dtype: DType
](
    ctx: DeviceContext,
    ptr: MutMemPtr[dtype],
    n: Int,
    out_dir: String,
    name: String,
) raises -> None:
    dump_tensor(ctx, ptr, n, out_dir + "/" + name + ".bin")


def _zero_pad2(n: Int) -> String:
    if n < 10:
        return "0" + String(n)
    return String(n)


def dump_per_layer[
    dtype: DType
](
    ctx: DeviceContext,
    ptr: MutMemPtr[dtype],
    num_layer: Int,
    per_layer_n: Int,
    out_dir: String,
    name: String,
) raises -> None:
    for layer in range(num_layer):
        var layer_ptr = ptr + layer * per_layer_n
        var path = out_dir + "/" + name + "_layer" + _zero_pad2(layer) + ".bin"
        dump_tensor(ctx, layer_ptr, per_layer_n, path)


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: dump_grads_gpt2.mojo <out_dir>")
        return
    var out_dir = String(args[1])
    if not exists(out_dir):
        makedirs(out_dir)

    var ctx = DeviceContext()
    var model = GPT2["gpu", 1, False](
        "gpt2_124M.bin",
        rank=0,
        zero_stage=0,
        ctx=ctx,
    )

    var C: Int = model.config.channels
    var L: Int = model.config.num_layer

    # Same fixed reference batch test_gpt2.mojo uses -- "identical step-1
    # data" across the fp8 and bf16 builds this tool is run against.
    var state_file = open("gpt2_124M_debug_state.bin", "r")
    var state_header = alloc[Int32](256)
    read_to_dtype_pointer[DType.int32](state_header, state_file, 256)
    var B: Int = Int(state_header[2])
    var T: Int = Int(state_header[3])
    var x = alloc[SIMD[DType.int32, 1]](B * T)
    var y = alloc[SIMD[DType.int32, 1]](B * T)
    read_to_dtype_pointer[DType.int32](x, state_file, B * T)
    read_to_dtype_pointer[DType.int32](y, state_file, B * T)
    state_file.close()
    state_header.free()

    model.forward(
        rebind[MutMemPtr[DType.int32]](x),
        rebind[MutMemPtr[DType.int32]](y),
        B,
        T,
    )
    model.zero_gradients()
    model.backward()
    ctx.synchronize()

    print("dumping 148 gradient tensors to " + out_dir)

    # Global tensors (4).
    dump_global(
        ctx,
        model.grads.wte,
        model.param_sizes[Parameters.wte],
        out_dir,
        "wte",
    )
    dump_global(
        ctx,
        model.grads.wpe,
        model.param_sizes[Parameters.wpe],
        out_dir,
        "wpe",
    )
    dump_global(ctx, model.grads.ln_f_gamma, C, out_dir, "ln_f_gamma")
    dump_global(ctx, model.grads.ln_f_beta, C, out_dir, "ln_f_beta")

    # Per-layer tensors (12 * L).
    dump_per_layer(ctx, model.grads.ln_1_gamma, L, C, out_dir, "ln_1_gamma")
    dump_per_layer(ctx, model.grads.ln_1_beta, L, C, out_dir, "ln_1_beta")
    dump_per_layer(
        ctx, model.grads.qkv_weight, L, (3 * C) * C, out_dir, "qkv_weight"
    )
    dump_per_layer(ctx, model.grads.qkv_bias, L, 3 * C, out_dir, "qkv_bias")
    dump_per_layer(
        ctx,
        model.grads.attn_proj_weight,
        L,
        C * C,
        out_dir,
        "attn_proj_weight",
    )
    dump_per_layer(
        ctx, model.grads.attn_proj_bias, L, C, out_dir, "attn_proj_bias"
    )
    dump_per_layer(ctx, model.grads.ln_2_gamma, L, C, out_dir, "ln_2_gamma")
    dump_per_layer(ctx, model.grads.ln_2_beta, L, C, out_dir, "ln_2_beta")
    dump_per_layer(
        ctx, model.grads.fc_weight, L, (4 * C) * C, out_dir, "fc_weight"
    )
    dump_per_layer(ctx, model.grads.fc_bias, L, 4 * C, out_dir, "fc_bias")
    dump_per_layer(
        ctx, model.grads.proj_weight, L, C * (4 * C), out_dir, "proj_weight"
    )
    dump_per_layer(ctx, model.grads.proj_bias, L, C, out_dir, "proj_bias")

    print("done: loss=" + String(model.mean_loss))
