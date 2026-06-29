from std.os import getenv
from std.sys import argv, exit
from std.gpu.host.info import is_cpu
from std.gpu.host import DeviceContext
from std.time import global_perf_counter_ns
from std.memory import alloc, UnsafePointer

from llmm.memory import MutMemPtr
from llmm.profiler import TraceProfiler

from train_gpt2 import GPT2

# Convenience harness for profiling the CUDA/CPU kernels in the train_gpt2
# training step. This is the llm.mojo analogue of llm.c's profile_gpt2.cu: it
# builds the GPT-2 model, runs a single forward / backward / update iteration on
# synthetic data, and synchronizes — a tight, deterministic target for an
# external profiler.
#
# Build (see the Makefile `build-profile` target):
#
#     make build-profile
#
# Profile the GPU kernels with NVIDIA Nsight Compute (per-kernel metrics):
#
#     make profile-ncu          # ncu --set full + profile_gpt2.py table
#
# Profile the timeline with NVIDIA Nsight Systems:
#
#     make profile-nsys         # nsys timeline -> build/profile_gpt2.nsys-rep
#
# Emit a Perfetto trace of the high-level phases (forward/backward/update),
# loadable directly at https://ui.perfetto.dev:
#
#     make profile-trace        # writes build/profile_gpt2.perfetto-trace.json
#
# Tunables (env, all optional):
#   LLMM_PROFILE_LAYERS=N   transformer layers to run (default 1, like llm.c —
#                           one layer is representative and keeps the profile
#                           small; set to 12 for the full model)
#   LLMM_PROFILE_B=N        batch size (default 4)
#   LLMM_PROFILE_T=N        sequence length (default 1024)
#   LLMM_PROFILE_STEPS=N    iterations to run (default 1)
#   LLMM_PROFILE_TRACE=path Perfetto trace output (empty = no trace)


# Read an integer environment variable, falling back to `default` when unset.
def _env_int(name: String, default: Int) raises -> Int:
    var v = getenv(name)
    if v == "":
        return default
    return atol(v)


def run_profile[
    target: StaticString,
](
    B: Int, T: Int, num_layers: Int, num_steps: Int, trace_path: String
) raises -> None:
    var ctx: DeviceContext
    comptime if is_cpu[target]():
        ctx = DeviceContext(api="cpu")
    else:
        ctx = DeviceContext()

    # Build the GPT-2 124M model from the checkpoint.
    var model = GPT2[target, 1](
        "gpt2_124M.bin",
        rank=0,
        zero_stage=0,
        ctx=ctx,
    )

    # Trim the layer count for a fast, representative profile (mirrors llm.c's
    # `model.config.num_layers = 1`). Parameters for all layers stay resident —
    # only the forward/backward layer loop is shortened — so the optimizer step
    # still profiles over the full parameter set.
    if 0 < num_layers and num_layers < model.config.num_layer:
        model.config.num_layer = num_layers

    print("[profile] target:", target)
    print("[profile] batch size:", B)
    print("[profile] sequence length:", T)
    print("[profile] layers:", model.config.num_layer)
    print("[profile] steps:", num_steps)

    # Synthetic inputs/targets, exactly like profile_gpt2.cu: token i = i % V.
    var vocab_size = model.config.vocab_size
    var x = alloc[Scalar[DType.int32]](B * T)
    var y = alloc[Scalar[DType.int32]](B * T)
    for i in range(B * T):
        x[i] = Int32(i % vocab_size)
        y[i] = Int32(i % vocab_size)
    var x_ptr = rebind[MutMemPtr[DType.int32]](x)
    var y_ptr = rebind[MutMemPtr[DType.int32]](y)

    var prof = TraceProfiler(trace_path)
    prof.process_name(0, String("llm.mojo profile"))
    prof.thread_name(0, 0, String(target) + " step")

    for step in range(num_steps):
        var t0 = global_perf_counter_ns()
        model.forward(x_ptr, y_ptr, B, T)
        ctx.synchronize()
        var t1 = global_perf_counter_ns()

        model.backward()
        ctx.synchronize()
        var t2 = global_perf_counter_ns()

        model.update(UInt32(step + 1), Scalar[DType.float32](1e-4))
        ctx.synchronize()
        var t3 = global_perf_counter_ns()

        var args = (
            '{"step":'
            + String(step)
            + ',"loss":'
            + String(model.mean_loss)
            + "}"
        )
        prof.complete("forward", "fwd", t0, t1, 0, 0, args)
        prof.complete("backward", "bwd", t1, t2, 0, 0, args)
        prof.complete("update", "opt", t2, t3, 0, 0, args)
        prof.complete("step", "step", t0, t3, 0, 0, args)

        print(
            "step "
            + String(step)
            + ": loss "
            + String(model.mean_loss)
            + " | forward: "
            + String(Float64(t1 - t0) / 1e9)
            + "s | backward: "
            + String(Float64(t2 - t1) / 1e9)
            + "s | update: "
            + String(Float64(t3 - t2) / 1e9)
            + "s"
        )

    prof.close()
    x.free()
    y.free()


def main() raises -> None:
    var args = argv()
    var target = String("gpu")
    if len(args) > 1:
        target = args[1]

    var B = _env_int("LLMM_PROFILE_B", 4)
    var T = _env_int("LLMM_PROFILE_T", 1024)
    var layers = _env_int("LLMM_PROFILE_LAYERS", 1)
    var steps = _env_int("LLMM_PROFILE_STEPS", 1)
    var trace_path = getenv("LLMM_PROFILE_TRACE")

    if target == "gpu":
        run_profile["gpu"](B, T, layers, steps, trace_path)
    elif target == "cpu":
        run_profile["cpu"](B, T, layers, steps, trace_path)
    else:
        print("Unknown target:", target, "(expected 'cpu' or 'gpu')")
        exit(1)
