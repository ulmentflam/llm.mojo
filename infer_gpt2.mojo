from std.sys import argv, exit, has_accelerator
from std.os import getenv
from max.gpu.host import DeviceContext


from llmm.memory import MutMemPtr, ImmutMemPtr, heap_alloc
from llmm.sampler import random_f32
from llmm.tokenizer import Tokenizer, safe_print
from llmm.eval_dataloader import EvalDataLoader, eval_stat_correct
from llmm.hf_download import fetch_hf_checkpoint

from llmm.safetensors import read_hf_arch, HF_MODEL_TYPE_STOCK
from train_gpt2 import GPT2, GPT2_DTYPE, ARCH_FLAGS

# Lean inference-only harness: load a checkpoint, and either (a) run
# autoregressive B=1 generation and print the decoded text, or (b) run a
# multiple-choice completion eval (HellaSwag-style) and print the accuracy.
# No backward pass, no optimizer step, no training-shard dataloading — just
# the forward-pass code paths, isolated so they can be iterated on without
# spinning up the full training harness.
#
# Build (see the Makefile `build-infer` / `build-infer-bf16` targets):
#
#     make build-infer-bf16
#
# Run (generation mode):
#
#     MOJO_PYTHON_LIBRARY=... ./build/infer_gpt2_bf16 log124M/model_19552.bin 64
#
#   arg1: checkpoint path — accepts our own model_*.bin, OR a HuggingFace
#         export's model.safetensors (config.json must sit alongside it —
#         see scripts/export_to_hf.py). Format is auto-detected by extension.
#   arg2: number of tokens to generate (default 64)
#   arg3: RNG seed (default 1337)
#
# Run (generation mode, fetching straight from a HuggingFace repo):
#
#     MOJO_PYTHON_LIBRARY=... ./build/infer_gpt2_bf16 --hf ulmentflam/gpt2-124m-fineweb-mojo 64
#
#   arg1: --hf
#   arg2: HuggingFace repo id (downloads config.json + model.safetensors)
#   arg3: number of tokens to generate (default 64)
#   arg4: RNG seed (default 1337)
#
# Run (loss mode: mean loss of one real batch, to compare a model.safetensors
# export against its raw .bin, or against the PyTorch reference):
#
#     ./build/infer_gpt2_bf16 --loss model_5700.bin data/.fineweb10B/<val>.bin 4 1024
#
#   arg1: --loss
#   arg2: checkpoint path (.bin or .safetensors)
#   arg3: token shard (llm.c layout: 256-int32 header, then uint16 tokens)
#   arg4: B (default 4); arg5: T (default 1024); the batch is the first B*T+1
#         tokens of the shard (inputs tokens[:-1], targets tokens[1:]).
#   Prints loss=<mean> and the first per-position losses of row 0.
#
# Run (eval mode — e.g. HellaSwag; see data/hellaswag.py to produce the .bin):
#
#     MOJO_PYTHON_LIBRARY=... ./build/infer_gpt2_bf16 --eval log124M/model_19552.bin data/.hellaswag/hellaswag_val.bin
#
#   arg1: --eval
#   arg2: checkpoint path (model_*.bin)
#   arg3: eval file path (hellaswag_val.bin, from data/hellaswag.py)
#   arg4: micro batch size B, must be a multiple of 4 (default 64)
#   arg5: sequence length T (default 512 — comfortably above any single
#         HellaSwag row's context+completion length, well under the model's
#         max_seq_len=1024, to avoid needlessly forwarding padding)


def refuse_stock_safetensors_in_flagged_build(checkpoint_path: String) raises:
    """A stock `model_type: gpt2` .safetensors in a rotary or QK-norm build.

    Unlike a .bin (where an unflagged file in a flagged build is allowed with a
    note: the trainer's `-e` and the verify gates start flagged runs from the
    unflagged pretrained gpt2_124M*.bin on purpose, and the GPT2 constructor
    prints that note), inference of a stock export in a flagged build is
    always the wrong model for the weights: the stock export is plain GPT-2 by
    construction (a rotary/QK-norm checkpoint exports as llm_mojo_gpt2). Refuse.
    The trainer's `-e` from such a file stays allowed (it is an init).
    """
    if ARCH_FLAGS == 0 or not checkpoint_path.endswith(".safetensors"):
        return
    var slash_idx = checkpoint_path.rfind("/")
    var config_dir = String(
        checkpoint_path[byte=0:slash_idx]
    ) if slash_idx >= 0 else String(".")
    var cfg = config_dir + "/config.json"
    var arch = read_hf_arch(cfg)
    if arch.model_type == HF_MODEL_TYPE_STOCK and ARCH_FLAGS != 0:
        raise Error(
            "Cannot run "
            + checkpoint_path
            + " for inference: its config ("
            + cfg
            + ") records architecture flags "
            + String(arch.arch_flags)
            + " but this build has "
            + String(ARCH_FLAGS)
            + " (1=rotary, 4=QK-norm). A stock gpt2 export is plain GPT-2, so"
            " this build would compute a different model from the weights. Use"
            " a build without the flags (or the export of a file trained with"
            " them)."
        )


def run_infer[
    target: StaticString,
](checkpoint_path: String, gen_max_length: Int, seed: UInt64) raises -> None:
    var ctx = DeviceContext()
    refuse_stock_safetensors_in_flagged_build(checkpoint_path)

    var model = GPT2[target, 1](
        checkpoint_path,
        rank=0,
        zero_stage=0,
        ctx=ctx,
    )
    # W7: a model trained with -attn_scale needs the same scale here
    # (LLMM_ATTN_SCALE=<f>); with -D LLMM_QK_NORM=1 the norm itself is baked in.
    if getenv("LLMM_ATTN_SCALE") != "":
        model.set_attn_scale(Float32(atof(getenv("LLMM_ATTN_SCALE"))))
    print("[infer] target:", target)
    print("[infer] checkpoint:", checkpoint_path)
    print("[infer] gen_max_length:", gen_max_length)
    print(
        "[infer] vocab_size:",
        model.config.vocab_size,
        "padded_vocab_size:",
        model.config.padded_vocab_size,
    )

    var tokenizer = Tokenizer("gpt2_tokenizer.bin")

    var gen_tokens = heap_alloc[Scalar[DType.int32]](gen_max_length)
    var zero = 0
    var null_int32_ptr = MutMemPtr[DType.int32](unsafe_from_address=zero)
    var rng_state = seed

    # Warm-up: forward() lazily allocates activations sized to its FIRST call's
    # (batch_size, seq_len), then rejects any later call with a larger shape.
    # In the real training->generate flow, training's (B=32, T=1024) forward
    # calls size this generously before generation's tiny (1, t) calls ever
    # run. A standalone inference-only tool's first call IS the tiny one, so
    # replicate training's sizing here or every (1, t>1) call raises "Sequence
    # length or batch size is larger than the previous allocations".
    var warmup_tokens = heap_alloc[Scalar[DType.int32]](model.config.max_seq_len)
    for i in range(model.config.max_seq_len):
        warmup_tokens[unsafe_offset=i] = Scalar[DType.int32](tokenizer.eot_token)
    model.forward(warmup_tokens, null_int32_ptr, 1, model.config.max_seq_len)
    warmup_tokens.unsafe_free()

    gen_tokens[unsafe_offset=0] = Scalar[DType.int32](tokenizer.eot_token)

    print("generating:\n---")
    for t in range(1, gen_max_length):
        model.forward(gen_tokens, null_int32_ptr, 1, t)
        # Same readback + sampling code as train_gpt2's periodic sample
        # (GPT2.sample_next_token), so the two paths cannot drift apart again
        # (docs/ai/attention_stale_upper_triangle_generation_bug.md).
        var coin = random_f32(rng_state)
        var next_token = model.sample_next_token(t, coin)
        gen_tokens[unsafe_offset=t] = Scalar[DType.int32](next_token)
        var token_str = tokenizer.decode(next_token)
        safe_print(token_str)
    print("\n---")

    gen_tokens.unsafe_free()


def run_eval[
    target: StaticString,
](
    checkpoint_path: String, eval_path: String, batch_size: Int, seq_len: Int
) raises -> None:
    var ctx = DeviceContext()
    refuse_stock_safetensors_in_flagged_build(checkpoint_path)

    var model = GPT2[target, 1](
        checkpoint_path,
        rank=0,
        zero_stage=0,
        ctx=ctx,
    )
    if getenv("LLMM_ATTN_SCALE") != "":
        model.set_attn_scale(Float32(atof(getenv("LLMM_ATTN_SCALE"))))
    print("[eval] target:", target)
    print("[eval] checkpoint:", checkpoint_path)
    print("[eval] eval file:", eval_path)
    print("[eval] batch_size:", batch_size, "seq_len:", seq_len)

    var loader = EvalDataLoader(eval_path, batch_size, seq_len)
    print("[eval] num_examples:", loader.num_examples)
    print("[eval] num_batches:", loader.num_batches)

    var num_correct = 0
    for i in range(loader.num_batches):
        loader.next_batch()
        model.forward(loader.inputs, loader.targets, batch_size, seq_len)
        num_correct += eval_stat_correct(
            loader,
            rebind[Pointer[Scalar[DType.float32], MutAnyOrigin]](
                model.losses_host_buf.unsafe_ptr().as_unsafe_any_origin()
            ),
        )
        if i % 10 == 0 or i == loader.num_batches - 1:
            print(
                "evaluating:",
                i + 1,
                "/",
                loader.num_batches,
                "| running correct:",
                num_correct,
            )

    var acc = Float32(num_correct) / Float32(loader.num_examples)
    print("HellaSwag:", num_correct, "/", loader.num_examples, "=", acc)


def run_loss[
    target: StaticString,
](
    checkpoint_path: String, data_path: String, batch_size: Int, seq_len: Int
) raises -> None:
    var ctx = DeviceContext()
    refuse_stock_safetensors_in_flagged_build(checkpoint_path)
    var model = GPT2[target, 1](
        checkpoint_path,
        rank=0,
        zero_stage=0,
        ctx=ctx,
    )
    if getenv("LLMM_ATTN_SCALE") != "":
        model.set_attn_scale(Float32(atof(getenv("LLMM_ATTN_SCALE"))))
    print("[loss] target:", target)
    print("[loss] checkpoint:", checkpoint_path)
    print("[loss] data:", data_path, "B:", batch_size, "T:", seq_len)

    var n = batch_size * seq_len
    var file = open(data_path, "r")
    _ = file.read_bytes(256 * 4)  # llm.c data header
    var raw = file.read_bytes((n + 1) * 2)
    file.close()
    if len(raw) < (n + 1) * 2:
        raise Error("data shard shorter than B*T+1 tokens")
    var inputs = heap_alloc[Scalar[DType.int32]](n)
    var targets = heap_alloc[Scalar[DType.int32]](n)
    var tok = raw.unsafe_ptr()
    for i in range(n):
        var a = Int(tok[unsafe_offset=2 * i]) | (
            Int(tok[unsafe_offset=2 * i + 1]) << 8
        )
        var b = Int(tok[unsafe_offset=2 * i + 2]) | (
            Int(tok[unsafe_offset=2 * i + 3]) << 8
        )
        inputs[unsafe_offset=i] = Scalar[DType.int32](a)
        targets[unsafe_offset=i] = Scalar[DType.int32](b)

    # Same warm-up sizing as run_infer: the first forward sizes the activations.
    model.forward(inputs, targets, batch_size, seq_len)
    print("loss=", model.mean_loss)
    var line = String("row0_losses=")
    for i in range(min(8, seq_len)):
        line += String(Float32(model.losses_host_buf[i])) + " "
    print(line)
    inputs.unsafe_free()
    targets.unsafe_free()


def main() raises -> None:
    var args = argv()

    if len(args) > 1 and args[1] == "--loss":
        if len(args) < 4:
            print("Usage: --loss <checkpoint> <data.bin> [B] [T]")
            exit(1)
        var batch_size = 4
        if len(args) > 4:
            batch_size = atol(args[4])
        var seq_len = 1024
        if len(args) > 5:
            seq_len = atol(args[5])
        comptime if GPT2_DTYPE == DType.bfloat16:
            if not has_accelerator():
                print(
                    "bf16 build supports only the GPU target (CPU stays fp32)."
                )
                exit(1)
            comptime if has_accelerator():
                run_loss["gpu"](args[2], args[3], batch_size, seq_len)
        else:
            comptime if has_accelerator():
                run_loss["gpu"](args[2], args[3], batch_size, seq_len)
            else:
                run_loss["cpu"](args[2], args[3], batch_size, seq_len)
        return

    # --hf <repo_id>: fetch config.json + model.safetensors from a public
    # HuggingFace model repo (e.g. ulmentflam/gpt2-124m-fineweb-mojo), then
    # generate from it exactly like a local checkpoint path. The fetch is the
    # one Python-bridged step (llmm/hf_download.mojo); loading the fetched
    # .safetensors into GPT2's parameter buffers is pure Mojo either way (see
    # GPT2.__init__'s .safetensors branch, train_gpt2.mojo).
    if len(args) > 1 and args[1] == "--hf":
        if len(args) <= 2:
            print("Usage: --hf <repo_id> [gen_max_length] [seed]")
            exit(1)
        var checkpoint_path = fetch_hf_checkpoint(args[2])
        var gen_max_length = 64
        if len(args) > 3:
            gen_max_length = atol(args[3])
        var seed: UInt64 = 1337
        if len(args) > 4:
            seed = UInt64(atol(args[4]))

        comptime if GPT2_DTYPE == DType.bfloat16:
            if not has_accelerator():
                print(
                    "bf16 build supports only the GPU target (CPU stays fp32)."
                )
                exit(1)
            comptime if has_accelerator():
                run_infer["gpu"](checkpoint_path, gen_max_length, seed)
        else:
            comptime if has_accelerator():
                run_infer["gpu"](checkpoint_path, gen_max_length, seed)
            else:
                run_infer["cpu"](checkpoint_path, gen_max_length, seed)
        return

    if len(args) > 1 and args[1] == "--eval":
        var checkpoint_path = String("log124M/model_19552.bin")
        if len(args) > 2:
            checkpoint_path = args[2]
        var eval_path = String("data/.hellaswag/hellaswag_val.bin")
        if len(args) > 3:
            eval_path = args[3]
        var batch_size = 64
        if len(args) > 4:
            batch_size = atol(args[4])
        var seq_len = 512
        if len(args) > 5:
            seq_len = atol(args[5])

        comptime if GPT2_DTYPE == DType.bfloat16:
            if not has_accelerator():
                print(
                    "bf16 build supports only the GPU target (CPU stays fp32)."
                )
                exit(1)
            comptime if has_accelerator():
                run_eval["gpu"](checkpoint_path, eval_path, batch_size, seq_len)
        else:
            comptime if has_accelerator():
                run_eval["gpu"](checkpoint_path, eval_path, batch_size, seq_len)
            else:
                run_eval["cpu"](checkpoint_path, eval_path, batch_size, seq_len)
        return

    var checkpoint_path = String("log124M/model_19552.bin")
    if len(args) > 1:
        checkpoint_path = args[1]
    var gen_max_length = 64
    if len(args) > 2:
        gen_max_length = atol(args[2])
    var seed: UInt64 = 1337
    if len(args) > 3:
        seed = UInt64(atol(args[3]))

    comptime if GPT2_DTYPE == DType.bfloat16:
        if not has_accelerator():
            print("bf16 build supports only the GPU target (CPU stays fp32).")
            exit(1)
        comptime if has_accelerator():
            run_infer["gpu"](checkpoint_path, gen_max_length, seed)
    else:
        comptime if has_accelerator():
            run_infer["gpu"](checkpoint_path, gen_max_length, seed)
        else:
            run_infer["cpu"](checkpoint_path, gen_max_length, seed)
