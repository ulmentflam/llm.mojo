from std.os import getenv
from std.sys import argv, exit, has_accelerator
from std.gpu.host import DeviceContext
from std.memory import alloc

from llmm.memory import MutMemPtr, ImmutMemPtr
from llmm.sampler import random_f32, sample_softmax
from llmm.tokenizer import Tokenizer, safe_print
from llmm.eval_dataloader import EvalDataLoader, eval_stat_correct
from llmm.hf_download import fetch_hf_checkpoint

from train_gpt2 import GPT2, GPT2_DTYPE

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


def run_infer[
    target: StaticString,
](checkpoint_path: String, gen_max_length: Int, seed: UInt64) raises -> None:
    var ctx = DeviceContext()

    var model = GPT2[target, 1](
        checkpoint_path,
        rank=0,
        zero_stage=0,
        ctx=ctx,
    )
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

    var gen_tokens = alloc[Scalar[DType.int32]](gen_max_length)
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
    var warmup_tokens = alloc[Scalar[DType.int32]](model.config.max_seq_len)
    for i in range(model.config.max_seq_len):
        warmup_tokens[i] = Scalar[DType.int32](tokenizer.eot_token)
    model.forward(warmup_tokens, null_int32_ptr, 1, model.config.max_seq_len)
    warmup_tokens.free()

    gen_tokens[0] = Scalar[DType.int32](tokenizer.eot_token)

    # fp32 scratch buffer for the logits actually consumed by sample_softmax
    # (see the dtype-mismatch fix below): sized to vocab_size, not
    # padded_vocab_size, matching what sample_softmax iterates over.
    var vocab_size = model.config.vocab_size
    var logits_fp32 = alloc[Float32](vocab_size)

    print("generating:\n---")
    for t in range(1, gen_max_length):
        model.forward(gen_tokens, null_int32_ptr, 1, t)
        var dev_logits_ptr = (
            model.acts.logits + (t - 1) * model.config.padded_vocab_size
        )
        model.ctx.enqueue_copy(
            dst_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                model.logits_host_buf.unsafe_ptr().as_unsafe_any_origin()
            ),
            src_ptr=rebind[UnsafePointer[Scalar[GPT2_DTYPE], ImmutAnyOrigin]](
                dev_logits_ptr.as_unsafe_any_origin()
            ),
            size=vocab_size,
        )
        model.ctx.synchronize()

        # BUGFIX (was a raw rebind of a GPT2_DTYPE=bfloat16 host buffer to
        # DType.float32 — a 2-byte-vs-4-byte reinterpret, not a cast. That read
        # vocab_size*4 bytes from a padded_vocab_size*2-byte buffer: a ~100KB
        # out-of-bounds host read every sampled token, producing garbage
        # logits regardless of whether the GPU-side crash also fires.
        # Cast element-by-element into a properly-sized fp32 scratch buffer.
        var logits_bf16 = rebind[ImmutMemPtr[GPT2_DTYPE]](
            model.logits_host_buf.unsafe_ptr().as_unsafe_any_origin()
        )
        for i in range(vocab_size):
            logits_fp32[i] = logits_bf16[i].cast[DType.float32]()

        var coin = random_f32(rng_state)
        var next_token = sample_softmax(
            rebind[ImmutMemPtr[DType.float32]](logits_fp32), vocab_size, coin
        )
        gen_tokens[t] = Scalar[DType.int32](next_token)
        var token_str = tokenizer.decode(next_token)
        safe_print(token_str)
    print("\n---")

    gen_tokens.free()
    logits_fp32.free()


def run_eval[
    target: StaticString,
](
    checkpoint_path: String, eval_path: String, batch_size: Int, seq_len: Int
) raises -> None:
    var ctx = DeviceContext()

    var model = GPT2[target, 1](
        checkpoint_path,
        rank=0,
        zero_stage=0,
        ctx=ctx,
    )
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
            rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](
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


def main() raises -> None:
    var args = argv()

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
