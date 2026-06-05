import math
import inspect
from typing import Optional
from dataclasses import dataclass

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.nn.functional.init as init
from torch.distributed.optim import ZeroRedundancyOptimizer


"""
Functions (F)
"""


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    # The equation for softmax is \frac{\exp{x_i}}{\sum_{j}\exp{x_j}}
    max_val, _ = torch.max(
        input, dim=dim, keepdim=True
    )  # Asymptotically O(N) but is parallelized
    stabilized_x = (
        input - max_val
    )  # This prevents us from exponentiating a very large value.
    x_i = torch.exp(stabilized_x)
    sum_x_j = torch.sum(x_i, dim=dim, keepdim=True)
    return x_i / sum_x_j


def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    # Q, K, V of shapes [B = batch_size, num_heads, T = seq_len, head_dims]
    d_k = K.size(-1)
    # The Standard Attention formula is \softmax{\frac{QK^T}{\sqrt{d_k}}}V

    # Calculate the attention scores
    # Q @ K^T
    # [B, num_heads, T, head_dims] @ [B, num_heads, head_dims, T] -> [B, num_heads, T, T]
    QK = Q @ K.transpose(
        -2, -1
    )  # We can't use K.T because the batch_size and num_heads need to stay in place.
    # For single head attention the tensors would be in the shape [B, C, T] so we could use K.T

    scores = QK / torch.sqrt(d_k)

    # Apply a mask (most commonly causal) if provided
    if mask is not None:
        # Mask out future tokens by setting them to negative infinity before softmax.
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Apply softmax on the sequence dimension (-1)
    attn_weights = softmax(scores, dim=-1)

    # Dropout can optionally be added here (I prefer not to)

    # Multiply Values
    # [B, num_heads, T, T] @ [B, num_heads, T, head_dims] -> [B, num_heads, T, head_dims]
    output = attn_weights @ V
    # output in the original shape [B, num_heads, T, head_dims], attn_weights in shape [B, num_heds, T, T]
    return output, attn_weights


"""
Basic Class Definitions
"""


class Linear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight is stored in the form (out, in) to speed up backpropogation.
        self.weight = nn.Paramater(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        # Init the weights with a kaiming uniform
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # NOTE: this will need to be hand implemented in the target language

        # The forward pass can be made slightly faster if you store the weights in (in, out)
        # However since pytorch only saves the view transpose itself is essentally free O(1)
        # The tensor is becomes incontigous after the tranpose op.

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            # Init the bias (single dim) with a variation of the kaiming uniform
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / torch.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            # NOTE: This will need to be hand implemented in the target language
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias


class LayerNorm(nn.Module):
    __constants__ = ["in_features", "epislon"]
    in_features: int
    epislon: torch.float
    gamma: Tensor
    beta: Tensor

    def __init__(
        self,
        in_features: int,
        epislon: torch.float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.epislon = epislon

        self.gamma = nn.Paramater(torch.ones((in_features), **factory_kwargs))
        self.beta = nn.Paramater(torch.zeros((in_features), **factory_kwargs))

    def forward(self, x: Tensor) -> Tensor:
        # Compute the mean across the feature dimension
        sigma, u = torch.var_mean(x, dim=-1, keepdim=True)
        # Stabilize x
        stabilized_x = x - u
        # Square root of variance
        sqrt_sigma = torch.sqrt(sigma + self.epislon)

        return (stabilized_x / sqrt_sigma) * self.gamma + self.beta


class ReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, min=0)  # Clamp is element wise max


# NOTE: Find out how OpenAI derived this number.
GELU_CONSTANT: torch.float = 0.044715


class GeLU(nn.Module):
    # This is based on OpanAI's GeLU for their GPT2 model.

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    torch.sqrt(2.0 / math.pi) * (x + GELU_CONSTANT * torch.pow(x, 3))
                )
            )
        )


class Sequential(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self.children():
            x = module(x)
        return x


class Dropout(nn.Module):
    __constants__ = ["probability"]
    probability: torch.float

    def __init__(self, probability: torch.float = 0.5) -> None:
        super().__init__()
        assert 0 <= probability < 1, "Dropout must be between [0, 1)"
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.probability > 0:
            # Create a binary mask where the keep probability is (1-p)
            mask = (torch.rand_like(x) >= self.probability).float()

            # Zero out the masked units and scale up
            return (x * mask) / (1 - self.probability)
        return x


class Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, x_hat: Tensor) -> Tensor:
        return x + x_hat


"""
Transformer Class Definitions
"""


class MLP(nn.Module):
    __constants__ = ["input_dim", "hidden_dim", "out_dim"]
    input_dim: int
    hidden_dim: int
    out_dim: int

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: Optional[torch.float] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        ordered_layers = [
            Linear(input_dim, hidden_dim),
            GeLU(),
            Linear(hidden_dim, input_dim),
        ]
        ordered_layers[-1].LLMC_RESIDUAL_SCALE_FLAG = 1  # Compatibility with Karpathy.
        if dropout:
            ordered_layers.append(Dropout(dropout))
        self.layers = Sequential(*ordered_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    __constants__ = ["d_model", "num_heads", "d_head"]
    d_model: int
    num_heads: int
    d_head: int

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # Integer division (floor)

        self.qkv_proj = Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, "Tensor Shape Mismatch"

        # Project Q, K, V
        qkv = self.qkv_proj(x)  # Produces [batch_size, seq_len, 3 * d_model]
        # Reshape and split into num_heads to [batch_size, seq_len, num_heads, 3 * self.d_head]
        qkv = qkv.view(batch_size, seq_len, self.num_heads, 3 * self.d_head)

        # Transpose to [batch_size, num_heads, seq_len, 3 * d_head]
        qkv = qkv.transpose(1, 2)

        # Split the combined 3 * self.d_head into independent Q, K, V
        Q, K, V = qkv.chunk(3, dim=-1)

        # Scaled Dotproduct Attention
        # Karpathy adds support for FLASH mode, so I may need to do the same
        # Creates a causal mask in the shape of [seq_len, seq_len] that fills
        # The lower triangle for causal attention.
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        out, _ = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Concatinate the Heads back together back to [batch_size, seq_len, num_heads, d_head]
        out = out.transpose(1, 2).contigious()
        # Tensor must be modified in memory and made contigious

        # Flatten the num_heads dimension to [batch_size, seq_len, d_model]
        out = out.view(batch_size, seq_len, d_model)

        # Return the final out projection
        return self.out_proj(out)


# GPT2Block, no additional linear, normalization before mlp
# updated with my special class for residual given the new world of mHC we live in.
class Block(nn.Module):
    __constants__ = ["d_model", "num_heads", "hidden_dim"]
    d_model: int
    num_heads: int
    hidden_dim: int

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        dropout: Optional[torch.float] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.mlp = MLP(d_model, hidden_dim, d_model, dropout=dropout)
        self.norm_attn = LayerNorm(d_model)
        self.norm_mlp = LayerNorm(d_model)
        self.residual = Residual()

    def forward(self, x: Tensor) -> Tensor:
        x = self.residual(x, self.attention(self.norm_attn(x)))
        x = self.residual(x, self.mlp(self.norm_mlp(x)))
        return x


"""
GPT For Causal LM
"""
# TODO: GPT


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.tranformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList(
                    [
                        Block(
                            d_model=config.n_embd,
                            num_heads=config.num_heads,
                            hidden_dim=4 * config.n_embd,
                        )
                    ]
                    for _ in range(config.n_layer)
                ),
                ln_f=LayerNorm(config.n_embd),
            )
        )

        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1  # Don't init this one, we tie weights.
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # paperswith code weight tieing

        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(43)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, Linear):
            # Apply special scaled init to residual projections via the GPT2 paper.
            std = (
                0.02
                if not hasattr(module, "LLMC_RESIDUAL_SCALE_FLAG")
                else 0.02 / torch.sqrt(2 * self.config.n_layer)
            )
            # We skip initilizing lm_head that shares params with wte
            if not hasattr(module, "LLMC_SKIP_INIT"):
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=std, generator=self.init_rng
                )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02, generator=self.init_rng
            )

    def forward(
        self, idx: Tensor, targets: Tensor | None = None, return_logits: bool = True
    ) -> tuple[Tensor, Tensor | None]:
        device = idx.device
        b, t = idx.shape
        assert t <= self.config.block_size, (
            f"Cannot find forward seq_len {t}, block size is only {self.config.block_size}"
        )

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # Shape [t]

        token_emb = self.transformer.wte(
            idx
        )  # Token embeddings of shape [B, T, n_embd]
        possition_emb = self.tranformer.wpe(
            pos
        )  # Positioning embeddings pf shape [t, n_embd]

        x = token_emb + possition_emb

        for block in self.tranformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            # NOTE: I decided not to hand roll cross_entropy loss of sake of time. Since the python implementation is not the main priority.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )  # We want the last row of logits against the last row of the targets.
        else:
            # Inference optimization to only forward the last logits of the head.
            logits = self.lm_head(x[:, [-1], :])  # Uses [-1] to preserve the time dim
            loss = None

        if not return_logits:
            loss = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str) -> "GPT":
        """Loades pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel  # Inline imports are not prefered.

        # n_layer, n_head, and n_embed are determined by model model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2=medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # Static for all GPT model checkpoints
        config_args["block_size"] = 1024  # Static for all GPT model checkpoints

        # Create a from-scratch initilized GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask/buffer not a params

        # Init a huggingface transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy and ensure all of the paramaters are aligned and match in name and shape
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask buffer

        # Transposed values
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        # OpenAI checkpoints use a "Conv1D" module, but we are only wanting to use a vanilla linear.
        # This means we have to transpose these weights.
        assert len(sd_keys_hf) == len(sd_keys), (
            f"Key Mismatch: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights for transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape, (
                    f"Transpose Shape Mismatch {k}"
                )
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Copy over the other paramaters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(
        self,
        weight_decay: torch.float,
        learning_rate: torch.float,
        betas: torch.float,
        device: torch.device,
        zero_stage: int,
    ) -> torch.optim.Optimizer:

        # Candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optimizer groups. Any parameter that is is 2D will be weight decay, otherwise none.
        # e.g. all weight tensors in matmuls + embedding decay, all biases and layernorms
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters"
        )

        # Create the AdamW optimizer and use fused version if it is avaliable
        # NOTE: I haven't implemented AdamW from scratch in this project so I can get to the mojo code faster.

        fused_avaliable = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avaliable and device == "cuda"

        print(f"Using fused AdamW: {use_fused}")

        if zero_stage == 1:
            print("Using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(
                **optim_groups[0],
                optimizer_class=torch.optim.AdamW,
                lr=learning_rate,
                betas=betas,
                fused=use_fused,
            )
            optimizer.add_param_group(optim_groups[1])
        else:
            print("Using regular AdamW")
            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, fused=use_fused
            )

        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temprature: torch.float = 1.0,
        top_k: int | None = None,
    ):
        for _ in range(max_new_tokens):
            # Crop the seq_len to the batch_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            # Forward the model for the next logits
            logits, _ = self(idx_cond)

            # Pluck logits and scale by the desired temprature
            logits = logits[:, -1, :] / temprature

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))

            # Apply softmax to convert logits to probabilities
            probs = softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the sample back to the sequence and start against
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx


"""
Our own, simple, Distributed Data Loader
"""

MAGIC_NUMBER = 20240520


def _read_header(f) -> int:
    # Reads the header from the buffer, 256 int32 integers (4 bytes each)
    header = np.frombuffer(f.read(256 * 4), dtype=np.int32)

    if header[0] != MAGIC_NUMBER:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)

    assert header[1] == 1, "Unsupported Version"
    return header[2]  # Number of tokens (claimed)


def _peek_data_shard(filename: str) -> int:
    # Reads the header, returns the data

    with open(filename, "rb") as f:
        ntok = _read_header(f)
    return ntok


def _load_data_shard(filename: str) -> bytes:
    with open(filename, "rb") as f:
        # Read the header for ntok first
        ntok = _read_header(f)
        # The remainder are tokens, stored in uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "Number of tokens read does not match header"
    return tokens
