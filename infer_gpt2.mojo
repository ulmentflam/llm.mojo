from std.os import getenv
from std.sys import argv, exit, has_accelerator
from std.gpu.host import DeviceContext
from std.memory import alloc

from llmm.memory import MutMemPtr, ImmutMemPtr
from llmm.sampler import random_f32, sample_softmax
from llmm.tokenizer import Tokenizer, safe_print

from train_gpt2 import GPT2, GPT2_DTYPE

# Lean inference-only harness: load a checkpoint, run autoregressive B=1
# generation, print the decoded text. No backward pass, no optimizer step, no
# training-shard dataloading — just the forward-pass + sampling code path that
# train_gpt2.mojo's end-of-run generation block exercises, isolated so it can
# be iterated on without spinning up the full training harness.
#
# Build (see the Makefile `build-infer` / `build-infer-bf16` targets):
#
#     make build-infer-bf16
#
# Run:
#
#     MOJO_PYTHON_LIBRARY=... ./build/infer_gpt2_bf16 log124M/model_19552.bin 64
#
#   arg1: checkpoint path (model_*.bin)
#   arg2: number of tokens to generate (default 64)
#   arg3: RNG seed (default 1337)


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


def main() raises -> None:
    var args = argv()
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
