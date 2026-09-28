"""GPT-2 inference on MAX's experimental eager API (max.experimental).

Scaffold: the module tree, checkpoint loading, device placement, the
eager/compiled execution switch and the sampling loop are wired up; the
forward math (every `forward` marked TODO) is left to implement. The
stable-graph-API twin is `infer_gpt2_max_graph.py`; both share
`max_gpt2_common.py`.

The same `forward` body serves both modes:

    --mode eager     runs forward op-by-op on real Tensors (PyTorch-style),
                     the fast edit/debug loop
    --mode compiled  Module.compile() traces forward once over a symbolic
                     [1, seq_len] input into a graph and compiles it

max.experimental is explicitly unstable; expect renames on MAX upgrades.

Run:

    make infer-max-eager                          # CPU, compiled mode
    make infer-max-eager ARGS='--mode eager -n 32 --prompt "Hello"'

Acceptance test: tests/test_max_gpt2.py (see infer_gpt2_max_graph.py).
"""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Accelerator, Device, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Embedding, LayerNorm, Linear, Module, ModuleList
from max.experimental.tensor import Tensor, default_dtype
from max.graph import AlgebraicDim, DeviceRef, Dim, TensorType, ops

from max_gpt2_common import (
    ATTENTION_OP,
    ATTENTION_PARAMETERS,
    GELU_ALGEBRAIC_ROWS,
    GELU_OP,
    GPT2Config,
    Tokenizer,
    base_arg_parser,
    llmm_package,
    load_checkpoint,
    run_generation,
)

# Compute dtype. max.experimental defaults new tensors to bfloat16, so the
# model is built under an explicit default to match the float32 loader (and
# the graph scaffold's DTYPE).
DTYPE = DType.float32
LAYERNORM_EPS = 1e-5


# --- llmm's Mojo kernels as custom ops. Each helper runs eagerly and under
# Module.compile (symbolic seq_len). Kernel outputs are written in place, so
# they are zeros tensors handed over with __buffervalue__(), which needs the
# realization context; hence the explicit F.ensure_context(). Called eagerly,
# MAX warns that its interpreter cannot run custom ops and compiles the op
# instead; --mode compiled builds the whole forward once and never hits that.


def _host_scalar(dim: Dim) -> Tensor:
    """A dimension as the 0-d int64 tensor a kernel's Int64 argument takes.

    Always on the CPU, whatever device the tensors are on: MAX hands scalar
    operands to the kernel as host values and rejects them elsewhere
    ("Scalars should always be on the host CPU"). Works for symbolic dims.
    """
    return Tensor.from_graph_value(
        ops.reshape(ops.shape_to_tensor([dim]), []).to(DeviceRef.CPU())
    )


def llmm_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Causal self-attention on llmm's `attention_fwd` kernel.

    q, k, v: [B, NH, T, HS] on one device, in the kernel dtype. The kernel
    applies the causal mask and the 1/sqrt(HS) scale. Returns [B, NH, T, HS];
    the log-sum-exp it also writes (for the backward) is discarded.
    """
    with F.ensure_context():
        dims = list(q.shape)
        out = Tensor.zeros(dims, dtype=q.dtype, device=q.device)
        lse = Tensor.zeros(dims[:3], dtype=DType.float32, device=q.device)
        F.inplace_custom(
            name=ATTENTION_OP,
            device=DeviceRef.from_device(q.device),
            values=[
                out.__buffervalue__(),
                q,
                k,
                v,
                lse.__buffervalue__(),
                *(_host_scalar(d) for d in dims),
            ],
            parameters=dict(ATTENTION_PARAMETERS),
            custom_extensions=llmm_package(),
        )
    return out


def llmm_gelu(x: Tensor) -> Tensor:
    """GELU (tanh approximation, as GPT-2) on llmm's `gelu_fwd` kernel.

    The kernel works on [rows, cols], so leading dims are flattened into rows
    and the input's shape is restored on the way out. They must flatten to
    one static or symbolic dim ([T, C], [1, T, C], [B, C] with B static);
    see GELU_ALGEBRAIC_ROWS for why [2, T, C] cannot under compile.
    """
    with F.ensure_context():
        shape = list(x.shape)
        x2d = F.reshape(x, [-1, shape[-1]])
        rows = x2d.shape[0]
        if isinstance(rows, AlgebraicDim):
            raise NotImplementedError(
                GELU_ALGEBRAIC_ROWS.format(dims=shape[:-1], rows=rows)
            )
        out = Tensor.zeros(list(x2d.shape), dtype=x.dtype, device=x.device)
        F.inplace_custom(
            name=GELU_OP,
            device=DeviceRef.from_device(x.device),
            values=[out.__buffervalue__(), x2d],
            custom_extensions=llmm_package(),
        )
    return F.reshape(out, shape)


class CausalSelfAttention(Module[[Tensor], Tensor]):
    """Multi-head causal self-attention with fused QKV projection."""

    def __init__(self, config: GPT2Config) -> None:
        c = config.channels
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.c_attn = Linear(c, 3 * c)
        self.c_proj = Linear(c, c)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, C] -> [B, T, C].

        TODO: qkv = c_attn(x) split into q, k, v; reshape each to
        [B, num_heads, T, head_dim]; scores = q @ k^T / sqrt(head_dim);
        causal mask (F.band_part, or F.where against an upper-triangular
        mask); F.softmax; @ v; merge heads; c_proj.
        Or replace the scores-through-@-v steps with llmm_attention(q, k, v).
        """
        raise NotImplementedError("CausalSelfAttention.forward")


class MLP(Module[[Tensor], Tensor]):
    """Position-wise feed-forward: c_fc -> GELU (tanh approx) -> c_proj."""

    def __init__(self, config: GPT2Config) -> None:
        c = config.channels
        self.c_fc = Linear(c, 4 * c)
        self.c_proj = Linear(4 * c, c)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, C] -> [B, T, C].

        TODO: c_proj(F.gelu(c_fc(x), approximate="tanh")), or
        c_proj(llmm_gelu(c_fc(x))) for llmm's kernel.
        """
        raise NotImplementedError("MLP.forward")


class Block(Module[[Tensor], Tensor]):
    """Pre-LN transformer block."""

    def __init__(self, config: GPT2Config) -> None:
        c = config.channels
        self.ln_1 = LayerNorm(c, eps=LAYERNORM_EPS)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(c, eps=LAYERNORM_EPS)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, C] -> [B, T, C].

        TODO: x = x + attn(ln_1(x)); x = x + mlp(ln_2(x)).
        """
        raise NotImplementedError("Block.forward")


class GPT2(Module[[Tensor], Tensor]):
    """GPT-2 with the LM head tied to the token embedding."""

    def __init__(self, config: GPT2Config) -> None:
        self.config = config
        c = config.channels
        # Padded vocab rows are kept so the checkpoint loads without reshaping.
        self.wte = Embedding(config.padded_vocab_size, dim=c)
        self.wpe = Embedding(config.max_seq_len, dim=c)
        self.h = ModuleList(Block(config) for _ in range(config.num_layers))
        self.ln_f = LayerNorm(c, eps=LAYERNORM_EPS)

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: int64 [B, T] -> logits: float32 [B, T, padded_vocab_size].

        TODO: x = wte(tokens) + wpe(F.arange(T)); run every block in self.h;
        ln_f; logits = x @ wte.weight.T (tied head, no bias).
        """
        raise NotImplementedError("GPT2.forward")


def pick_device(name: str) -> Device:
    if name == "gpu":
        if accelerator_count() == 0:
            raise SystemExit("--device gpu: MAX sees no accelerator on this host")
        return Accelerator()
    return CPU()


def build_model(
    config: GPT2Config, weights: dict[str, np.ndarray], device: Device
) -> GPT2:
    """Construct without materialising the random init, then load real weights."""
    with F.lazy(), default_dtype(DTYPE):
        model = GPT2(config)
    model.load_state_dict(weights)
    model.to(device)
    return model


def make_forward(model: GPT2, mode: str):
    """Adapt eager or compiled execution to the ForwardFn the sampler expects."""
    if mode == "compiled":
        tokens_type = TensorType(DType.int64, [1, "seq_len"], device=model.device)
        run = model.compile(tokens_type)
    else:
        run = model

    def forward(tokens: np.ndarray) -> np.ndarray:
        logits = run(Tensor.from_dlpack(tokens).to(model.device))
        return np.from_dlpack(logits.to(CPU()))

    return forward


def main() -> None:
    parser = base_arg_parser(__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("compiled", "eager"), default="compiled")
    args = parser.parse_args()
    config, weights = load_checkpoint(args.checkpoint)
    device = pick_device(args.device)
    print(f"[max-eager] checkpoint: {args.checkpoint}")
    print(f"[max-eager] device: {args.device}, mode: {args.mode}, config: {config}")
    model = build_model(config, weights, device)
    run_generation(
        make_forward(model, args.mode),
        config,
        Tokenizer(args.tokenizer),
        args.max_tokens,
        args.seed,
        args.prompt,
    )


if __name__ == "__main__":
    main()
