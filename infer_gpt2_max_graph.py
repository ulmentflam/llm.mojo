"""GPT-2 inference on MAX's stable graph API (max.nn + max.graph).

Scaffold: the module tree, checkpoint loading, graph construction,
compilation and the sampling loop are wired up; the forward math (every
`__call__` marked TODO) is left to implement. The eager-API twin is
`infer_gpt2_max_eager.py`; both share `max_gpt2_common.py`.

How the pieces fit:

    load_checkpoint()  -> float32 numpy weights keyed by canonical names
    GPT2(...)          -> max.nn.Module tree whose state-dict keys match them
    build_graph()      -> traces GPT2.__call__ over a symbolic [1, seq_len]
                          int64 input into a max.graph.Graph
    session.load()     -> compiles the graph, binding weights by name
    generate()         -> sampling loop calling the compiled model

Run:

    make infer-max-graph                         # CPU, gpt2_124M.bin
    make infer-max-graph ARGS='--device gpu -n 128 --prompt "Hello"'

Acceptance test: tests/test_max_gpt2.py compares logits against the PyTorch
reference model in train_gpt2.py. It is marked xfail while the forward
raises NotImplementedError and turns into a hard failure (strict xfail) the
moment the forward runs, so drop the marker once it passes.
"""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops  # noqa: F401
from max.nn import Embedding, LayerList, LayerNorm, Linear, Module

from max_gpt2_common import (
    GPT2Config,
    Tokenizer,
    base_arg_parser,
    load_checkpoint,
    run_generation,
)

# Compute dtype for weights and activations. The loader hands back float32;
# a bf16 path would cast the weights in `load_weights` and the logits back
# to float32 before sampling.
DTYPE = DType.float32
LAYERNORM_EPS = 1e-5


class CausalSelfAttention(Module):
    """Multi-head causal self-attention with fused QKV projection."""

    def __init__(self, config: GPT2Config, device: DeviceRef) -> None:
        super().__init__()
        c = config.channels
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.c_attn = Linear(c, 3 * c, DTYPE, device, has_bias=True)
        self.c_proj = Linear(c, c, DTYPE, device, has_bias=True)

    def __call__(self, x: TensorValue) -> TensorValue:
        """x: [B, T, C] -> [B, T, C].

        TODO: qkv = c_attn(x) split into q, k, v; reshape each to
        [B, num_heads, T, head_dim]; scores = q @ k^T / sqrt(head_dim);
        mask j > i to -inf (ops.band_part or a constant upper-triangular
        mask); softmax; @ v; merge heads back to [B, T, C]; c_proj.
        """
        raise NotImplementedError("CausalSelfAttention.__call__")


class MLP(Module):
    """Position-wise feed-forward: c_fc -> GELU (tanh approx) -> c_proj."""

    def __init__(self, config: GPT2Config, device: DeviceRef) -> None:
        super().__init__()
        c = config.channels
        self.c_fc = Linear(c, 4 * c, DTYPE, device, has_bias=True)
        self.c_proj = Linear(4 * c, c, DTYPE, device, has_bias=True)

    def __call__(self, x: TensorValue) -> TensorValue:
        """x: [B, T, C] -> [B, T, C].

        TODO: c_proj(gelu(c_fc(x))). GPT-2 uses the tanh approximation
        (ops.gelu(..., approximate="tanh")), same as llmm/gelu.mojo.
        """
        raise NotImplementedError("MLP.__call__")


class Block(Module):
    """Pre-LN transformer block."""

    def __init__(self, config: GPT2Config, device: DeviceRef) -> None:
        super().__init__()
        c = config.channels
        self.ln_1 = LayerNorm(c, [device], DTYPE, eps=LAYERNORM_EPS)
        self.attn = CausalSelfAttention(config, device)
        self.ln_2 = LayerNorm(c, [device], DTYPE, eps=LAYERNORM_EPS)
        self.mlp = MLP(config, device)

    def __call__(self, x: TensorValue) -> TensorValue:
        """x: [B, T, C] -> [B, T, C].

        TODO: x = x + attn(ln_1(x)); x = x + mlp(ln_2(x)).
        """
        raise NotImplementedError("Block.__call__")


class GPT2(Module):
    """GPT-2 with the LM head tied to the token embedding."""

    def __init__(self, config: GPT2Config, device: DeviceRef) -> None:
        super().__init__()
        self.config = config
        c = config.channels
        # Padded vocab rows are real rows in the checkpoint (zeros); keeping
        # them means the weights load without reshaping.
        self.wte = Embedding(config.padded_vocab_size, c, DTYPE, device)
        self.wpe = Embedding(config.max_seq_len, c, DTYPE, device)
        self.h = LayerList([Block(config, device) for _ in range(config.num_layers)])
        self.ln_f = LayerNorm(c, [device], DTYPE, eps=LAYERNORM_EPS)

    def __call__(self, tokens: TensorValue) -> TensorValue:
        """tokens: int64 [B, T] -> logits: float32 [B, T, padded_vocab_size].

        TODO: x = wte(tokens) + wpe(arange(T)); run every block in self.h;
        ln_f; logits = x @ wte.weight^T (tied head, no bias). Returning the
        padded width is fine: the sampler slices to vocab_size.
        """
        raise NotImplementedError("GPT2.__call__")


def pick_device(name: str) -> Device:
    if name == "gpu":
        if accelerator_count() == 0:
            raise SystemExit("--device gpu: MAX sees no accelerator on this host")
        return Accelerator()
    return CPU()


def build_graph(model: GPT2, device: DeviceRef) -> Graph:
    """Trace the model over a dynamic sequence length into a MAX graph."""
    tokens_type = TensorType(DType.int64, [1, "seq_len"], device=device)
    with Graph("gpt2", input_types=[tokens_type]) as graph:
        graph.output(model(graph.inputs[0].tensor))
    return graph


def load_weights(model: GPT2, weights: dict[str, np.ndarray]) -> None:
    """Bind checkpoint arrays to the module's Weights (strict: names must match)."""
    model.load_state_dict(weights)


def compile_model(
    config: GPT2Config, weights: dict[str, np.ndarray], device: Device
) -> Model:
    device_ref = DeviceRef.from_device(device)
    model = GPT2(config, device_ref)
    load_weights(model, weights)
    graph = build_graph(model, device_ref)
    session = InferenceSession(devices=[device])
    return session.load(graph, weights_registry=model.state_dict())


def make_forward(compiled: Model, device: Device):
    """Adapt the compiled model to the ForwardFn the sampling loop expects."""

    def forward(tokens: np.ndarray) -> np.ndarray:
        (logits,) = compiled.execute(Buffer.from_numpy(tokens).to(device))
        assert isinstance(logits, Buffer)
        return logits.to(CPU()).to_numpy()

    return forward


def main() -> None:
    args = base_arg_parser(__doc__.splitlines()[0]).parse_args()
    config, weights = load_checkpoint(args.checkpoint)
    device = pick_device(args.device)
    print(f"[max-graph] checkpoint: {args.checkpoint}")
    print(f"[max-graph] device: {args.device}, config: {config}")
    compiled = compile_model(config, weights, device)
    run_generation(
        make_forward(compiled, device),
        config,
        Tokenizer(args.tokenizer),
        args.max_tokens,
        args.seed,
        args.prompt,
    )


if __name__ == "__main__":
    main()
