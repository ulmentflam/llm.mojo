"""Shared plumbing for the MAX GPT-2 inference scaffolds.

Everything here is model-API agnostic, so the stable graph scaffold
(`infer_gpt2_max_graph.py`) and the experimental eager scaffold
(`infer_gpt2_max_eager.py`) share one checkpoint reader, one tokenizer, one
sampler and one generation loop. The two scaffolds differ only in how they
build and run the forward pass, which is the part left to implement.

The checkpoint reader, sampler and generation loop mirror the Mojo path
(`llmm/checkpointing.mojo`, `llmm/sampler.mojo`, `infer_gpt2.mojo`), so a MAX
implementation seeded the same way samples the same tokens as
`build/infer_gpt2` until float differences between the two forwards flip a
draw.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llmm_pkg import ensure_llmm_package

# Model checkpoint magic written by train_gpt2.mojo / train_gpt2.py. llm.c's
# own starter-pack magic (20240326) uses the identical layout, so accept both.
MODEL_MAGICS = (20240520, 20240326)
VERSION_FP32 = 3
VERSION_BF16 = 5
HEADER_INTS = 256

TOKENIZER_MAGIC = 20240520
TOKENIZER_VERSION = 2

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPO_ROOT / "gpt2_124M.bin"
DEFAULT_TOKENIZER = REPO_ROOT / "gpt2_tokenizer.bin"

# --- llmm's Mojo kernels as MAX custom ops.
#
# Both scaffolds can call the @register'd kernels in llmm/ through their
# `llmm_attention` / `llmm_gelu` helpers, loading the package llmm_pkg builds
# and caches by source content (`make build-mojo`; shared with the tests).
#
# Two rules shape both helpers. The kernels declare their outputs as
# MutableInputTensor and write them in place, so MAX must hand them buffers
# rather than the immutable tensors ordinary ops return; each helper allocates
# those buffers and reads the result back out. And a kernel's Int64 arguments
# (sizes such as seq_len) travel as 0-d int64 tensors on the host CPU,
# whatever device the data lives on: the kernel reads them on the host to
# size-check its buffers and pick a launch grid, and MAX rejects scalar
# operands anywhere else ("Scalars should always be on the host CPU").
ATTENTION_OP = "attention_fwd"
# Exact exp and unconditional rescaling: the variants tests/ checks against
# PyTorch. The Mojo defaults (True) select approximate fast paths.
ATTENTION_PARAMETERS: dict[str, bool | int | str] = {
    "use_soft_exp": False,
    "use_conditional_rescale": False,
}
GELU_OP = "gelu_fwd"
# gelu_fwd works on [rows, cols], so the helpers flatten leading dims. MAX
# 26.6 cannot buffer_load a buffer sized by an algebraic dim (e.g. rows =
# 2 * seq_len): the graph fails to compile with "'mo.mutable.load' op failed
# to verify that all of {inBuffer, outTensor} have same shape". Leading dims
# must therefore flatten to a single static or symbolic dim, which the
# scaffolds' [1, seq_len, 4C] activations do.
GELU_ALGEBRAIC_ROWS = (
    "llmm_gelu: leading dims {dims} flatten to the algebraic dim {rows}, which"
    " MAX 26.6 cannot size a kernel output buffer with (mo.mutable.load fails"
    " to verify). Use a static batch of 1 or a single symbolic leading dim."
)


def llmm_package() -> Path:
    """The precompiled llmm package for the current sources.

    Content-addressed (see llmm_pkg), so it can never be stale; built on
    first use if `make build-mojo` has not already.
    """
    return ensure_llmm_package()


@dataclass(frozen=True)
class GPT2Config:
    """GPT-2 hyperparameters, as stored in the checkpoint header."""

    max_seq_len: int = 1024
    vocab_size: int = 50257
    padded_vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    channels: int = 768

    @property
    def head_dim(self) -> int:
        return self.channels // self.num_heads


def param_shapes(config: GPT2Config) -> list[tuple[str, tuple[int, ...]]]:
    """Canonical parameter names and shapes, in checkpoint file order.

    Names follow HF GPT-2 minus the `transformer.` prefix, and every scaffold
    module tree is laid out so its state dict uses exactly these keys. Linear
    weights are `[out, in]` (PyTorch convention; forward is `x @ W.T + b`).
    `wte.weight` keeps the padded vocab rows and doubles as the tied LM head.
    """
    c, vp, t = config.channels, config.padded_vocab_size, config.max_seq_len
    per_layer = [
        ("ln_1.weight", (c,)),
        ("ln_1.bias", (c,)),
        ("attn.c_attn.weight", (3 * c, c)),
        ("attn.c_attn.bias", (3 * c,)),
        ("attn.c_proj.weight", (c, c)),
        ("attn.c_proj.bias", (c,)),
        ("ln_2.weight", (c,)),
        ("ln_2.bias", (c,)),
        ("mlp.c_fc.weight", (4 * c, c)),
        ("mlp.c_fc.bias", (4 * c,)),
        ("mlp.c_proj.weight", (c, 4 * c)),
        ("mlp.c_proj.bias", (c,)),
    ]
    shapes: list[tuple[str, tuple[int, ...]]] = [
        ("wte.weight", (vp, c)),
        ("wpe.weight", (t, c)),
    ]
    # The file stores each per-layer tensor for ALL layers contiguously
    # ([L, ...]) before moving to the next tensor kind.
    for suffix, shape in per_layer:
        shapes.extend((f"h.{i}.{suffix}", shape) for i in range(config.num_layers))
    shapes += [("ln_f.weight", (c,)), ("ln_f.bias", (c,))]
    return shapes


def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Widen raw bf16 bit patterns (uint16) to float32, exactly."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def load_checkpoint(path: Path) -> tuple[GPT2Config, dict[str, np.ndarray]]:
    """Read a model_*.bin / gpt2_124M*.bin checkpoint into float32 arrays.

    bf16 checkpoints (version 5) are widened to float32 losslessly; casting
    down to the compute dtype is the scaffold's job.
    """
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(HEADER_INTS * 4), dtype=np.int32)
        blob = f.read()

    magic, version = int(header[0]), int(header[1])
    if magic not in MODEL_MAGICS:
        raise ValueError(f"{path}: bad magic {magic}, want one of {MODEL_MAGICS}")
    if version not in (VERSION_FP32, VERSION_BF16):
        raise ValueError(
            f"{path}: unsupported version {version} (want {VERSION_FP32} fp32"
            f" or {VERSION_BF16} bf16)"
        )
    config = GPT2Config(
        max_seq_len=int(header[2]),
        vocab_size=int(header[3]),
        num_layers=int(header[4]),
        num_heads=int(header[5]),
        channels=int(header[6]),
        padded_vocab_size=int(header[7]),
    )

    if version == VERSION_FP32:
        flat = np.frombuffer(blob, dtype=np.float32)
    else:
        flat = _bf16_to_f32(np.frombuffer(blob, dtype=np.uint16))

    weights: dict[str, np.ndarray] = {}
    offset = 0
    for name, shape in param_shapes(config):
        n = int(np.prod(shape))
        weights[name] = flat[offset : offset + n].reshape(shape)
        offset += n
    if offset != flat.size:
        raise ValueError(
            f"{path}: header describes {offset} params but the file holds"
            f" {flat.size}; header and payload disagree"
        )
    return config, weights


class Tokenizer:
    """Decode-only GPT-2 tokenizer read from llm.c's gpt2_tokenizer.bin.

    Same file and semantics as llmm/tokenizer.mojo, so generation output is
    comparable byte-for-byte with the Mojo binary. Prompts are encoded with
    tiktoken, which is only imported when a prompt is given.
    """

    def __init__(self, path: Path = DEFAULT_TOKENIZER) -> None:
        with open(path, "rb") as f:
            header = np.frombuffer(f.read(HEADER_INTS * 4), dtype=np.int32)
            if int(header[0]) != TOKENIZER_MAGIC:
                raise ValueError(f"{path}: bad tokenizer magic {int(header[0])}")
            if int(header[1]) != TOKENIZER_VERSION:
                raise ValueError(f"{path}: unsupported tokenizer version")
            self.vocab_size = int(header[2])
            self.eot_token = int(header[3])
            self._table: list[bytes] = []
            for _ in range(self.vocab_size):
                (length,) = f.read(1)
                self._table.append(f.read(length))

    def decode(self, token: int) -> bytes:
        return self._table[token] if 0 <= token < self.vocab_size else b""

    def encode(self, text: str) -> list[int]:
        import tiktoken

        return tiktoken.get_encoding("gpt2").encode(
            text, allowed_special={"<|endoftext|>"}
        )


# --- Sampling: llm.c's xorshift RNG + inverse-CDF sampling (llmc/sampler.h).

# xorshift64* (Marsaglia's xorshift, Vigna's multiplicative scrambler), with
# llm.c's constants. llmm/sampler.mojo defines them under the same names;
# changing any of them breaks sample parity with the Mojo binary.
U64_MASK = (1 << 64) - 1  # Python ints are unbounded; wrap like a C uint64_t
XORSHIFT_RIGHT_1 = 12
XORSHIFT_LEFT = 25
XORSHIFT_RIGHT_2 = 27
XORSHIFT_STAR_MULTIPLIER = 0x2545F4914F6CDD1D
U32_BITS = 32
FLOAT32_SIGNIFICAND_BITS = 24
# The scrambled state's low bits are weak, so the output keeps the high half.
U32_OUTPUT_SHIFT = U32_BITS
# A float32 carries 24 significant bits: keep the top 24 of the u32 and scale
# by 2^24, so every value in [0, 1) is exactly representable.
F32_DISCARD_BITS = U32_BITS - FLOAT32_SIGNIFICAND_BITS
F32_UNIT_SCALE = float(1 << FLOAT32_SIGNIFICAND_BITS)


def random_u32(state: int) -> tuple[int, int]:
    """One xorshift* step. Returns (value, new_state); state is a u64."""
    state ^= state >> XORSHIFT_RIGHT_1
    state ^= (state << XORSHIFT_LEFT) & U64_MASK
    state ^= state >> XORSHIFT_RIGHT_2
    scrambled = (state * XORSHIFT_STAR_MULTIPLIER) & U64_MASK
    return scrambled >> U32_OUTPUT_SHIFT, state


def random_f32(state: int) -> tuple[float, int]:
    """Uniform float32 in [0, 1). Returns (value, new_state)."""
    u, state = random_u32(state)
    return float(np.float32((u >> F32_DISCARD_BITS) / F32_UNIT_SCALE)), state


def sample_softmax(logits: np.ndarray, coin: float) -> int:
    """Sample from softmax(logits) with the C reference's float semantics.

    Matches llmm.sampler.sample_softmax and tests/_sampler_reference.py:
    exp in float32, normalizer accumulated sequentially in float64, CDF
    accumulated sequentially in float32. It skips the usual max-subtraction
    because llm.c does and the draw must round exactly as llm.c's; exp only
    overflows float32 for logits above ~88.
    """
    with np.errstate(over="ignore"):
        e = np.exp(np.asarray(logits, dtype=np.float32))
    norm = np.cumsum(e.astype(np.float64))[-1]
    scaled_coin = np.float32(np.float64(np.float32(coin)) * norm)
    cdf = np.cumsum(e, dtype=np.float32)
    hits = np.nonzero(scaled_coin < cdf)[0]
    return int(hits[0]) if hits.size else int(e.shape[0] - 1)


# --- Generation loop, parameterized by a forward callable.

# tokens int64 [1, T] -> logits float32 [1, T, padded_vocab_size]
ForwardFn = Callable[[np.ndarray], np.ndarray]


def generate(
    forward: ForwardFn,
    config: GPT2Config,
    eot_token: int,
    max_tokens: int,
    seed: int,
    prompt: list[int] | None = None,
) -> Iterator[int]:
    """Autoregressive B=1 sampling; yields each new token as it is sampled.

    Mirrors infer_gpt2.mojo: start from <|endoftext|> (plus an optional
    prompt), recompute the whole window every step (no KV cache), sample the
    last position's logits over the real (unpadded) vocab. `max_tokens`
    counts the whole sequence including the start token, as in the Mojo tool.
    Without a KV cache every step reruns the forward over the whole window,
    so a T-token sequence costs O(T^2) positions; a cache belongs in the
    forward, not in this loop.
    """
    tokens = [eot_token] + list(prompt or [])
    rng_state = seed
    while len(tokens) < max_tokens:
        window = tokens[-config.max_seq_len :]
        logits = forward(np.asarray([window], dtype=np.int64))
        last = np.asarray(logits, dtype=np.float32)[0, -1, : config.vocab_size]
        coin, rng_state = random_f32(rng_state)
        token = sample_softmax(last, coin)
        tokens.append(token)
        yield token


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    """CLI flags common to both scaffolds."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="model .bin checkpoint (fp32 or bf16), default gpt2_124M.bin",
    )
    p.add_argument(
        "-n",
        "--max-tokens",
        type=int,
        default=64,
        help="total sequence length to generate, including the start token",
    )
    p.add_argument("--seed", type=int, default=1337, help="sampler RNG seed")
    p.add_argument("--prompt", type=str, default=None, help="optional text prompt")
    p.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="gpu = first MAX accelerator (Metal on Apple Silicon, CUDA elsewhere)",
    )
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    return p


def run_generation(
    forward: ForwardFn,
    config: GPT2Config,
    tokenizer: Tokenizer,
    max_tokens: int,
    seed: int,
    prompt: str | None,
) -> None:
    """Stream sampled text to stdout, framed like infer_gpt2.mojo."""
    prompt_tokens = tokenizer.encode(prompt) if prompt else None
    out = sys.stdout.buffer
    print("generating:\n---", flush=True)
    if prompt:
        out.write(prompt.encode())
    for token in generate(
        forward, config, tokenizer.eot_token, max_tokens, seed, prompt_tokens
    ):
        out.write(tokenizer.decode(token))
        out.flush()
    print("\n---", flush=True)
