#!/usr/bin/env python3
import os
import math
import glob
import struct
import inspect
import contextlib
from dataclasses import dataclass
from typing import BinaryIO, Optional, Any, cast

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn.init as init
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, init_process_group


"""
Print Helper
"""


def print_zero_rank(*args, **kwargs) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


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

    scores = QK / math.sqrt(d_k)

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
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        # Init the weights with a kaiming uniform
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # NOTE: this will need to be hand implemented in the target language

        # The forward pass can be made slightly faster if you store the weights in (in, out)
        # However since pytorch only saves the view transpose itself is essentally free O(1)
        # The tensor is becomes noncontiguous after the transpose op.

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            # Init the bias (single dim) with a variation of the kaiming uniform
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            # NOTE: This will need to be hand implemented in the target language
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(nn.Module):
    __constants__ = ["in_features", "epsilon"]
    in_features: int
    epsilon: float
    gamma: Tensor
    beta: Tensor

    def __init__(
        self,
        in_features: int,
        epsilon: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones((in_features), **factory_kwargs))
        self.beta = nn.Parameter(torch.zeros((in_features), **factory_kwargs))

    def forward(self, x: Tensor) -> Tensor:
        # Compute the mean across the feature dimension
        sigma, u = torch.var_mean(x, dim=-1, keepdim=True)
        # Stabilize x
        stabilized_x = x - u
        # Square root of variance
        sqrt_sigma = torch.sqrt(sigma + self.epsilon)

        return (stabilized_x / sqrt_sigma) * self.gamma + self.beta


class ReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, min=0)  # Clamp is element wise max


# NOTE: Find out how OpenAI derived this number.
GELU_CONSTANT: float = 0.044715


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
                    math.sqrt(2.0 / math.pi) * (x + GELU_CONSTANT * torch.pow(x, 3))
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
    probability: float

    def __init__(self, probability: float = 0.5) -> None:
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
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        ordered_layers: list[nn.Module] = [
            Linear(input_dim, hidden_dim),
            GeLU(),
            Linear(hidden_dim, input_dim),
        ]
        if dropout:
            ordered_layers.append(Dropout(dropout))
        self.layers = Sequential(*ordered_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class KarpathyMLP(nn.Module):
    __constants__ = ["input_dim", "hidden_dim", "out_dim"]
    input_dim: int
    hidden_dim: int
    out_dim: int

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.c_fc = Linear(input_dim, hidden_dim)
        self.gelu = GeLU()
        self.c_proj = Linear(hidden_dim, out_dim)
        setattr(
            self.c_proj, "LLMC_RESIDUAL_SCALE_FLAG", 1
        )  # Compatibility with Karpathy.

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


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
        out = out.transpose(1, 2).contiguous()
        # Tensor must be modified in memory and made contiguous

        # Flatten the num_heads dimension to [batch_size, seq_len, d_model]
        out = out.view(batch_size, seq_len, d_model)

        # Return the final out projection
        return self.out_proj(out)


# Using a global to toggle flash-attention
FLASH = 0


# The purpose if this class is to name match the attention class vars for GPT-2
class CausalSelfAttention(nn.Module):
    __constants__ = ["d_model", "num_heads", "d_head", "block_size"]
    d_model: int
    num_heads: int
    d_head: int
    block_size: int
    bias: Tensor

    def __init__(self, d_model: int, num_heads: int, block_size: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.block_size = block_size
        # K, Q, V projections for all heads in batch
        self.c_attn = Linear(d_model, 3 * d_model)
        # Output projection
        self.c_proj = Linear(d_model, d_model)
        setattr(
            self.c_proj, "LLMC_RESIDUAL_SCALE_FLAG", 1
        )  # Compatibility with Karpathy.
        # Causal mask for self-attention: lower-triangular [1, 1, block_size, block_size]
        # so it broadcasts against attention scores of shape [B, num_heads, T, T] and
        # masks out positions j > i for each query i.
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        if FLASH:
            # Flash Attention via PyTorch
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(
                self.bias[:, :, :T, :T] == 0, float("-inf")
            )  # Karpathy's Causal Masking, not-preferred but will work fine.
            att = F.softmax(att, dim=-1)
            y = att @ v  # [B, nh, T, T] x [B, nh, T, hs] -> [B, nh, T, hs]
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # Re-assemble all head outputs side by side.
        # Output projection
        y = self.c_proj(y)
        return y


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
        dropout: Optional[float] = None,
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


class KarpathyBlock(nn.Module):
    __constants__ = ["d_model", "num_heads", "hidden_dim", "block_size"]
    d_model: int
    num_heads: int
    hidden_dim: int
    block_size: int

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        block_size: int,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.block_size = block_size

        self.ln_1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, block_size)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = KarpathyMLP(d_model, hidden_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
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


class Transformer(nn.Module):
    wte: nn.Embedding
    wpe: nn.Embedding
    h: nn.ModuleList
    ln_f: LayerNorm

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList(
            [
                KarpathyBlock(
                    d_model=config.n_embd,
                    num_heads=config.n_head,
                    hidden_dim=4 * config.n_embd,
                    block_size=config.block_size,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = LayerNorm(config.n_embd)


class GPT(nn.Module):
    transformer: Transformer
    lm_head: Linear

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = Transformer(config)

        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        setattr(
            self.lm_head, "LLMC_SKIP_INIT", 1
        )  # Don't init this one, we tie weights.
        self.transformer.wte.weight = self.lm_head.weight  # paperswithcode weight tying

        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(43)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, Linear):
            # Apply special scaled init to residual projections via the GPT2 paper.
            std = (
                0.02
                if not hasattr(module, "LLMC_RESIDUAL_SCALE_FLAG")
                else 0.02 / math.sqrt(2 * self.config.n_layer)
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
        position_emb = self.transformer.wpe(
            pos
        )  # Positioning embeddings pf shape [t, n_embd]

        x = token_emb + position_emb

        for block in self.transformer.h:
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
            logits = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str) -> "GPT":
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel  # Inline imports are not preferred.

        # n_layer, n_head, and n_embed are determined by model model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
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

        # HF's nn.LayerNorm uses weight/bias; our LayerNorm uses gamma/beta.
        # Map HF param names ending in .weight/.bias on LayerNorms to local gamma/beta.
        ln_suffixes = ("ln_1", "ln_2", "ln_f")

        def hf_to_local(key: str) -> str:
            for ln in ln_suffixes:
                if key.endswith(f".{ln}.weight"):
                    return key[: -len(".weight")] + ".gamma"
                if key.endswith(f".{ln}.bias"):
                    return key[: -len(".bias")] + ".beta"
            return key

        # OpenAI checkpoints use a "Conv1D" module, but we are only wanting to use a vanilla linear.
        # This means we have to transpose these weights.
        assert len(sd_keys_hf) == len(sd_keys), (
            f"Key Mismatch: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            local_k = hf_to_local(k)
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights for transpose
                assert sd_hf[k].shape[::-1] == sd[local_k].shape, (
                    f"Transpose Shape Mismatch {k}"
                )
                with torch.no_grad():
                    sd[local_k].copy_(sd_hf[k].t())
            else:
                # Copy over the other paramaters
                assert sd_hf[k].shape == sd[local_k].shape
                with torch.no_grad():
                    sd[local_k].copy_(sd_hf[k])
        return model

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
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
        print_zero_rank(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print_zero_rank(
            f"num non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters"
        )

        # Create the AdamW optimizer and use fused version if it is avaliable
        # NOTE: I haven't implemented AdamW from scratch in this project so I can get to the mojo code faster.

        fused_avaliable = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avaliable and device.type == "cuda"

        print_zero_rank(f"Using fused AdamW: {use_fused}")

        if zero_stage == 1:
            print_zero_rank("Using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(
                decay_params,
                optimizer_class=torch.optim.AdamW,
                weight_decay=weight_decay,
                lr=learning_rate,
                betas=betas,
                fused=use_fused,
            )
            optimizer.add_param_group(optim_groups[1])
        else:
            print_zero_rank("Using regular AdamW")
            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, fused=use_fused
            )

        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
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

            # Pluck logits and scale by the desired temperature
            logits = logits[:, -1, :] / temperature

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
Low-precision (FP8 / NVFP4) Linear layers — CUDA-only `--precision` support.

Both paths below are NATIVE (not emulated) tensor-core GEMMs, per
`tests/probe_torch_precisions.py`'s capability probe: `torch._scaled_mm`
(fp8, e4m3 x e4m3 -> bf16 forward, e5m2 x e4m3 -> bf16 grad-path) and
`torch.nn.functional.scaled_mm` (NVFP4, `ScalingType.BlockWise1x16`) both
dispatch real cuBLASLt/cutlass tensor-core kernels on this box (NVIDIA GB10,
sm_121). See `docs/ai/pytorch_precision_support.md` for the full writeup.

Master weights stay fp32 (`nn.Parameter`, untouched by the optimizer scheme);
fp8/fp4 exist only as transient, per-forward-call quantized copies of the
GEMM operands — mirroring `docs/ai/fp8_training_design.md`'s "storage stays
high precision; low precision is a transient inside the GEMM" scheme used by
the Mojo trainer.
"""

_FP8_E4M3_MAX = 448.0
_FP8_E5M2_MAX = 57344.0
_NVFP4_BLOCK = 16
_NVFP4_MAX = 6.0
_NVFP4_LADDER = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _amax_scale(x: Tensor, fp8_max: float) -> Tensor:
    """Quantization multiplier `s` such that `(x * s).clamp(-max,max)` fills
    the target fp8 dtype's dynamic range. `torch._scaled_mm`'s `scale_a`/
    `scale_b` want the DEQUANT factor `1/s`, not `s` itself — verified in
    `tests/probe_torch_precisions.py::probe_scaled_mm_fp8` (passing `s`
    silently produces >1e5x relative error rather than raising)."""
    amax = x.detach().abs().amax().clamp_min(1e-12).float()
    return fp8_max / amax


def _quantize_e4m3(x: Tensor, scale: Tensor) -> Tensor:
    return (
        (x.float() * scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    )


def _quantize_e5m2(x: Tensor, scale: Tensor) -> Tensor:
    return (
        (x.float() * scale).clamp(-_FP8_E5M2_MAX, _FP8_E5M2_MAX).to(torch.float8_e5m2)
    )


def _fp8_eligible(in_features: int, out_features: int) -> bool:
    # cuBLASLt fp8 TN GEMM requires K (in_features) a multiple of 16 (probed
    # in tests/probe_fp8/RESULTS.md); GPT-2's own dims (768/3072/...) always
    # satisfy this. Guarded defensively for arbitrary custom configs.
    return in_features % 16 == 0 and out_features % 16 == 0


class _Float8MatmulFn(torch.autograd.Function):
    """`y = x @ weight.T (+ bias)` with the GEMM run in fp8 (forward: E4M3 x
    E4M3 -> bf16; backward: E5M2 `d_output` x E4M3 weight/activation -> bf16,
    the Transformer-Engine HYBRID pairing also used by the Mojo trainer, see
    docs/ai/fp8_training_design.md §1.2). Per-tensor "current" (just-in-time)
    amax scaling — simpler than the Mojo build's delayed/history scaling,
    which exists purely for a performance reason (skip a max-reduction on
    the critical path) that doesn't apply to this reference script.

    cuBLASLt requires `mat_a` row-major and `mat_b` COL-major. The forward
    GEMM gets col-major "for free" (`weight.t()` on a contiguous `[N,K]`
    `nn.Parameter` is naturally `[K,N]` with stride `(1,K)`). The backward
    GEMMs contract along a DIFFERENT axis than the forward GEMM did, so the
    weight/activation/grad operand each needs re-quantizing from a freshly
    `.t().contiguous()`-ed copy in the orientation that GEMM needs, then
    `.t()`-ed back to a col-major VIEW — a pure memory-layout trick (the
    quantized VALUES are unaffected by transposition, since quantization is
    elementwise), not a numerics change. See
    `tests/probe_torch_precisions.py`'s `probe_scaled_mm_fp8` for the same
    pattern in isolation.
    """

    @staticmethod
    def forward(ctx, x2d: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        s_x = _amax_scale(x2d, _FP8_E4M3_MAX)
        s_w = _amax_scale(weight, _FP8_E4M3_MAX)
        xq = _quantize_e4m3(x2d, s_x)
        wq = _quantize_e4m3(weight, s_w)
        y = torch._scaled_mm(
            xq,
            wq.t(),
            scale_a=(1.0 / s_x).reshape(1),
            scale_b=(1.0 / s_w).reshape(1),
            out_dtype=torch.bfloat16,
        )
        if bias is not None:
            y = y + bias.to(y.dtype)
        ctx.save_for_backward(x2d, weight, bias if bias is not None else torch.empty(0))
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor):
        (grad_out,) = grad_outputs
        x2d, weight, bias = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        s_g = _amax_scale(grad_out, _FP8_E5M2_MAX)
        gq = _quantize_e5m2(grad_out, s_g)

        # dgrad: dX[M,K] = G[M,N] @ W[N,K]. mat_b must be [N,K] col-major;
        # requantize a [K,N]-row-major copy of W and take a `.t()` view.
        s_w = _amax_scale(weight, _FP8_E4M3_MAX)
        wtq = _quantize_e4m3(weight.t().contiguous(), s_w)  # [K, N] row-major
        dx = torch._scaled_mm(
            gq,
            wtq.t(),
            scale_a=(1.0 / s_g).reshape(1),
            scale_b=(1.0 / s_w).reshape(1),
            out_dtype=torch.bfloat16,
        )

        # wgrad: dW[N,K] = G^T[N,M] @ X[M,K]. Both operands need requantizing
        # in the M-contracting orientation for the same col-major reason.
        gtq = _quantize_e5m2(grad_out.t().contiguous(), s_g)  # [N, M] row-major
        s_x = _amax_scale(x2d, _FP8_E4M3_MAX)
        xtq = _quantize_e4m3(x2d.t().contiguous(), s_x)  # [K, M] row-major
        dw = torch._scaled_mm(
            gtq,
            xtq.t(),
            scale_a=(1.0 / s_g).reshape(1),
            scale_b=(1.0 / s_x).reshape(1),
            out_dtype=torch.bfloat16,
        )

        if ctx.has_bias:
            dbias = grad_out.sum(dim=0).to(bias.dtype)
        else:
            dbias = None
        return dx.to(x2d.dtype), dw.to(weight.dtype), dbias


class Float8Linear(Linear):
    """Drop-in replacement for `Linear` (a real subclass, so it type-checks
    when swapped into an attribute typed `Linear`, e.g. `CausalSelfAttention.
    c_attn`), swapped onto the model's QKV/attn-out and MLP fc/proj
    projections under `--precision fp8` (every block, every layer — matches
    `docs/ai/fp8_training_design.md`'s "the four per-block linear layers"
    scope; LM head and embeddings stay bf16/fp32, same as the Mojo build).
    Falls back to a plain matmul when `in_features`/`out_features` aren't
    fp8-GEMM-eligible (see `_fp8_eligible`). `weight`/`bias` are inherited
    from `Linear` unchanged (fp32 master weights); only `forward` differs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self._eligible = _fp8_eligible(in_features, out_features)

    @classmethod
    def from_linear(cls, linear: "Linear") -> "Float8Linear":
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            mod.weight.copy_(linear.weight)
            if linear.bias is not None and mod.bias is not None:
                mod.bias.copy_(linear.bias)
        if hasattr(linear, "LLMC_RESIDUAL_SCALE_FLAG"):
            setattr(mod, "LLMC_RESIDUAL_SCALE_FLAG", linear.LLMC_RESIDUAL_SCALE_FLAG)
        return mod

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        if self._eligible:
            y = _Float8MatmulFn.apply(x2d, self.weight, self.bias)
        else:
            y = F.linear(x2d.to(self.weight.dtype), self.weight, self.bias)
        return y.reshape(*orig_shape[:-1], self.out_features)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _to_blocked(input_matrix: Tensor) -> Tensor:
    """cuBLASLt's 128x4-tile / 32x4x4-internal block-scale-factor swizzle
    (cuBLASLt docs §3.1.4.3.2). Re-derived independently (not imported from
    `torch.testing._internal.common_quantized`, which needs the
    not-installed `expecttest` package) — cross-checked against
    `tests/probe_fp4/probe_fp4.cu`'s independent C++ derivation of the same
    formula (both agree, see `tests/probe_fp4/RESULTS.md`)."""
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows, padded_cols = n_row_blocks * 128, n_col_blocks * 4
    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _pack_uint4(uint8_data: Tensor) -> Tensor:
    shape = uint8_data.shape
    flat = uint8_data.contiguous().view(-1)
    return (flat[1::2] << 4 | flat[::2]).view(*shape[:-1], shape[-1] // 2)


def _nvfp4_eligible(in_features: int, out_features: int) -> bool:
    # NVFP4 packs 2 elements/byte along the contraction (in_features) dim,
    # and cuBLASLt needs the packed dim a multiple of 16 -> in_features a
    # multiple of 32; out_features (unpacked) needs a multiple of 16. GPT-2's
    # dims (768/3072) satisfy both; guarded for arbitrary custom configs.
    return in_features % 32 == 0 and out_features % 16 == 0


def _nvfp4_quantize(x2d: Tensor, block: int = _NVFP4_BLOCK) -> tuple[Tensor, Tensor]:
    """RNE (no stochastic rounding) NVFP4 quantizer: E2M1 elements, E4M3
    block scale (blocks along the last/contraction dim). Matches
    `tests/probe_fp4/probe_fp4.cu`'s conventions (`scale = block_amax / 6.0`,
    nearest-value E2M1 encode) — see `_quantize_nvfp4` in
    `tests/probe_torch_precisions.py` for the isolated probe version this is
    copied from."""
    rows, cols = x2d.shape
    ladder = torch.tensor(_NVFP4_LADDER, device=x2d.device)
    xb = x2d.reshape(rows, cols // block, block).float()
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / _NVFP4_MAX).to(torch.float8_e4m3fn)
    scale_f32 = scale.float()
    xq = (xb / scale_f32).clamp(-_NVFP4_MAX, _NVFP4_MAX)
    sign = torch.sign(xq)
    absval = xq.abs()
    idx = torch.searchsorted(ladder, absval).clamp(max=len(ladder) - 1)
    lo = ladder[(idx - 1).clamp(min=0)]
    hi = ladder[idx]
    nearest = torch.where((hi - absval).abs() < (absval - lo).abs(), hi, lo)
    code_idx = torch.searchsorted(ladder, nearest).clamp(max=len(ladder) - 1)
    sign_bit = (sign < 0).to(torch.uint8) << 3
    code = (sign_bit | code_idx.to(torch.uint8)).reshape(rows, cols)
    packed = _pack_uint4(code).view(torch.float4_e2m1fn_x2)
    return packed, scale.reshape(rows, cols // block)


def _nvfp4_scaled_mm(
    a_packed: Tensor, a_scale: Tensor, b_packed_t: Tensor, b_scale: Tensor
) -> Tensor:
    from torch.nn.functional import ScalingType, SwizzleType

    return F.scaled_mm(
        a_packed,
        b_packed_t,
        scale_a=_to_blocked(a_scale),
        scale_recipe_a=ScalingType.BlockWise1x16,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        scale_b=_to_blocked(b_scale),
        scale_recipe_b=ScalingType.BlockWise1x16,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16,
    )


class _NVFP4MatmulFwdFn(torch.autograd.Function):
    """Forward-only NVFP4: `y = x @ weight.T (+ bias)` with the forward GEMM
    run as a NATIVE NVFP4 block-scaled cuBLASLt/cutlass tensor-core GEMM
    (`torch.nn.functional.scaled_mm`, `ScalingType.BlockWise1x16` — probed
    WORKING on this GB10/sm_121 box, dispatching the same
    `cutlass3x_sm120_..._ue4m3xe2m1_..._vs16` kernel
    `tests/probe_fp4/RESULTS.md` found via raw cuBLASLt). Backward
    (dgrad/wgrad) is plain bf16 matmul on the ORIGINAL (unquantized)
    activation/weight — deliberately NOT NVFP4.

    Why backward stays bf16: a real NVFP4 dgrad/wgrad needs each operand
    re-blocked-and-requantized along whichever axis THAT GEMM contracts over
    (forward contracts over in_features; dgrad contracts over out_features;
    wgrad contracts over the batch*seq axis — three different block
    orientations per tensor), and per
    docs/ai/fp4_training_recipes_research.md §1, a numerically STABLE fwd+bwd
    NVFP4 recipe additionally wants stochastic rounding on gradient operands
    and a 16x16 Hadamard transform on the Wgrad input — machinery this
    reference script has no other use for and doesn't implement anywhere
    else. The Mojo trainer's `matmul_bwd_fp4` does implement all of this;
    this class intentionally scopes down to "real NVFP4 forward, honest bf16
    backward" so `--precision nvfp4` still demonstrates genuine end-to-end
    NVFP4-tensor-core training (loss decreases through a real NVFP4 GEMM)
    without silently mislabeling an under-built backward as the full recipe.
    """

    @staticmethod
    def forward(ctx, x2d: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        xq, xs = _nvfp4_quantize(x2d)
        wq, ws = _nvfp4_quantize(weight)
        y = _nvfp4_scaled_mm(xq, xs, wq.t(), ws)
        if bias is not None:
            y = y + bias.to(y.dtype)
        ctx.save_for_backward(x2d, weight, bias if bias is not None else torch.empty(0))
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor):
        (grad_out,) = grad_outputs
        x2d, weight, bias = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        dx = (grad_out.float() @ weight.float()).to(x2d.dtype)
        dw = (grad_out.float().t() @ x2d.float()).to(weight.dtype)
        if ctx.has_bias:
            dbias = grad_out.sum(dim=0).to(bias.dtype)
        else:
            dbias = None
        return dx, dw, dbias


class NVFP4Linear(Linear):
    """Drop-in replacement for `Linear` (a real subclass — see `Float8Linear`'s
    docstring for why), swapped onto the MLP `c_fc`/`c_proj` projections of
    MIDDLE transformer blocks only under `--precision nvfp4` (see
    `_layer_in_fp4_range`, mirroring `train_gpt2.mojo`'s
    `LLMM_FP4_FIRST`/`LLMM_FP4_LAST` policy —
    docs/ai/fp4_training_recipes_research.md §1 "Selective high-precision
    layers": keep the first ~2 and final ~2 blocks, plus all attention/
    LayerNorm/embeddings/LM-head, in BF16). Forward is native NVFP4; backward
    is bf16 — see `_NVFP4MatmulFwdFn`. Falls back to a plain matmul when
    `in_features`/`out_features` aren't NVFP4-GEMM-eligible (`_nvfp4_eligible`).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self._eligible = _nvfp4_eligible(in_features, out_features)

    @classmethod
    def from_linear(cls, linear: "Linear") -> "NVFP4Linear":
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            mod.weight.copy_(linear.weight)
            if linear.bias is not None and mod.bias is not None:
                mod.bias.copy_(linear.bias)
        if hasattr(linear, "LLMC_RESIDUAL_SCALE_FLAG"):
            setattr(mod, "LLMC_RESIDUAL_SCALE_FLAG", linear.LLMC_RESIDUAL_SCALE_FLAG)
        return mod

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        if self._eligible:
            y = _NVFP4MatmulFwdFn.apply(x2d, self.weight, self.bias)
        else:
            y = F.linear(x2d.to(self.weight.dtype), self.weight, self.bias)
        return y.reshape(*orig_shape[:-1], self.out_features)


def _layer_in_fp4_range(layer: int, num_layers: int) -> bool:
    """True if `layer` falls in the NVFP4-eligible middle-block range —
    mirrors `train_gpt2.mojo`'s `_layer_in_fp4_range` (`LLMM_FP4_FIRST`
    defaults to 2, `LLMM_FP4_LAST` defaults to `num_layers - 2`), overridable
    via the same env vars for parity: `LLMM_FP4_FIRST`/`LLMM_FP4_LAST`."""
    fp4_first = int(os.environ.get("LLMM_FP4_FIRST", 2))
    fp4_last_override = int(os.environ.get("LLMM_FP4_LAST", -1))
    fp4_last = fp4_last_override if fp4_last_override >= 0 else num_layers - 2
    return fp4_first <= layer < fp4_last


def swap_precision_layers(model: "GPT", precision: str) -> dict:
    """Post-hoc, in-place swap of selected `Linear` submodules to
    `Float8Linear`/`NVFP4Linear`. Must run AFTER model construction (i.e.
    after `_init_weights`/pretrained-weight loading has already populated
    plain `Linear`s) and BEFORE `.to(device)`/DDP-wrapping. Returns counters
    for the startup banner."""
    assert precision in ("fp8", "nvfp4"), precision
    n_layer = model.config.n_layer
    counts = {"fp8_linears": 0, "nvfp4_linears": 0, "nvfp4_layers": 0}
    for i, block in enumerate(model.transformer.h):
        kblock = cast(KarpathyBlock, block)
        if precision == "fp8":
            kblock.attn.c_attn = Float8Linear.from_linear(kblock.attn.c_attn)
            kblock.attn.c_proj = Float8Linear.from_linear(kblock.attn.c_proj)
            kblock.mlp.c_fc = Float8Linear.from_linear(kblock.mlp.c_fc)
            kblock.mlp.c_proj = Float8Linear.from_linear(kblock.mlp.c_proj)
            counts["fp8_linears"] += 4
        elif precision == "nvfp4" and _layer_in_fp4_range(i, n_layer):
            kblock.mlp.c_fc = NVFP4Linear.from_linear(kblock.mlp.c_fc)
            kblock.mlp.c_proj = NVFP4Linear.from_linear(kblock.mlp.c_proj)
            counts["nvfp4_linears"] += 2
            counts["nvfp4_layers"] += 1
    return counts


def _torch_precision_capabilities(device_type: str) -> str:
    """Short, dependency-free summary of what this torch build supports on
    this device, for the startup banner. Presence-checks are dynamic; the
    "empirically probed WORKING" annotation summarizes a one-time run of
    `tests/probe_torch_precisions.py` on this box (torch/driver/GPU-specific
    findings, not worth re-running every training step)."""
    if device_type != "cuda":
        return "torch fp8/nvfp4 tensor-core paths are CUDA-only; not applicable on this device."
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    has_fp8 = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "_scaled_mm")
    has_fp4 = hasattr(torch, "float4_e2m1fn_x2") and hasattr(torch, "_scaled_mm_v2")
    return (
        f"{name} (sm_{cap[0]}{cap[1]}), torch {torch.__version__}: "
        f"float8_e4m3fn/_scaled_mm present={has_fp8}, "
        f"float4_e2m1fn_x2/_scaled_mm_v2 present={has_fp4} "
        f"(empirically probed WORKING on this box: e4m3xe4m3->bf16, "
        f"e5m2xe4m3->bf16 grad-path, NVFP4 BlockWise1x16 fwd — see "
        f"tests/probe_torch_precisions.py)"
    )


def precision_banner(
    precision: str, device_type: str, swap_counts: Optional[dict] = None
) -> str:
    """Startup banner text: requested precision, what's actually active
    (always native on this box — both probes passed, see
    `tests/probe_torch_precisions.py`), and the torch capability findings."""
    lines = [f"=== --precision={precision} ==="]
    if precision == "fp32":
        lines.append(
            "Active: fp32 strict (allow_tf32=False, matmul_precision='highest'), no autocast."
        )
    elif precision == "tf32":
        lines.append(
            "Active: fp32 storage, TF32 tensor-core matmuls "
            "(allow_tf32=True, matmul_precision='high'), no autocast."
        )
    elif precision == "bf16":
        lines.append(
            "Active: torch.amp.autocast(bfloat16) over fwd+loss; fp32 params/optimizer."
        )
    elif precision == "fp16":
        lines.append(
            "Active: torch.amp.autocast(float16) + GradScaler over "
            "fwd+loss/backward/step; fp32 params/optimizer."
        )
    elif precision == "fp8":
        lines.append(
            "Active: NATIVE fp8 (Float8Linear on all 4 per-block projections: "
            "qkv/attn_proj/mlp_fc/mlp_proj, every layer) — forward E4M3 x E4M3 "
            "-> bf16, backward d_output in E5M2 (dgrad+wgrad), per-tensor "
            "current/JIT amax scaling (not delayed-history). Everything else "
            "(LayerNorm/softmax/embeddings/LM head) in bf16 autocast, fp32 "
            "master weights."
        )
        if swap_counts:
            lines.append(
                f"  Swapped {swap_counts['fp8_linears']} Linear -> Float8Linear."
            )
    elif precision == "nvfp4":
        lines.append(
            "Active: NATIVE NVFP4 forward (E2M1 4-bit elements, 16-elem E4M3 "
            "block scale, single-level BlockWise1x16 cuBLASLt/cutlass "
            "tensor-core GEMM) on MLP fc/proj of MIDDLE blocks only "
            "(LLMM_FP4_FIRST/LLMM_FP4_LAST-style policy, mirroring "
            "train_gpt2.mojo); backward is bf16 matmul, NOT NVFP4 (see "
            "NVFP4Linear docstring). Everything else bf16 autocast, fp32 "
            "master weights."
        )
        if swap_counts:
            lines.append(
                f"  Swapped {swap_counts['nvfp4_linears']} Linear -> NVFP4Linear "
                f"across {swap_counts['nvfp4_layers']} middle block(s)."
            )
    lines.append(f"  {_torch_precision_capabilities(device_type)}")
    return "\n".join(lines)


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
            "---> HINT: For example re-run: `python data/tinyshakespeare.py`, then re-try"
        )
        exit(1)

    assert header[1] == 1, "Unsupported Version"
    return header[2]  # Number of tokens (claimed)


def _peek_data_shard(filename: str) -> int:
    # Reads the header, returns the data

    with open(filename, "rb") as f:
        ntok = _read_header(f)
    return ntok


def _load_data_shard(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        # Read the header for ntok first
        ntok = _read_header(f)
        # The remainder are tokens, stored in uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "Number of tokens read does not match header"
    return tokens


class DistributedDataLoader:
    def __init__(
        self,
        filename_pattern: str,
        B: int,
        T: int,
        process_rank: int,
        num_processes: int,
    ) -> None:
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, (
            f"did not find any files that match the pattern {filename_pattern}"
        )

        # Load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for f_name in self.files:
            shard_ntok = _peek_data_shard(f_name)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print_zero_rank(
            f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files"
        )

        # Kick off
        self.current_shard = None
        self.reset()

    def reset(self) -> None:
        # If shard zero is loaded, then don't need to reload it, just reset the pointer.
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self) -> None:
        self.current_shard = ((self.current_shard or 0) + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self) -> tuple[Tensor, Tensor]:
        B = self.B
        T = self.T
        buffer = self.tokens[self.current_position : self.current_position + B * T + 1]
        buffer = torch.tensor(buffer.astype(np.int32), dtype=torch.long)
        x = (buffer[:-1]).view(B, T)  # Inputs
        y = (buffer[1:]).view(B, T)  # Targets
        # Advance the start pointer in the current shard
        self.current_position += B * T * self.num_processes
        # If loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y


"""
Python to C bridge utilities for saving params/grads/activations to .bin files.

This might need to be adapted to mojo.
"""


def _uint32(value: int) -> bytes:
    LITTLE_ENDIAN_UNSIGNED_INT_32 = "<I"
    return struct.pack(
        LITTLE_ENDIAN_UNSIGNED_INT_32, value
    )  # little-endian unsigned int 32


def _int32(value: int) -> bytes:
    LITTLE_ENDIAN_SIGNED_INT_32 = "<i"
    return struct.pack(
        LITTLE_ENDIAN_SIGNED_INT_32, value
    )  # little-endian signed int 32


def _int32_array(ndim: int, values: list[int]) -> bytes:
    LITTLE_ENDIAN_SIGNED_INT_32_ARRAY = "<%di"
    return struct.pack(
        LITTLE_ENDIAN_SIGNED_INT_32_ARRAY % ndim, *values
    )  # little-endian signed int 32 * ndim


def write_fp32(tensor: Tensor, file: BinaryIO) -> None:
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.write(b)


def write_bf16(tensor: Tensor, file: BinaryIO) -> None:
    t = tensor.detach().cpu().to(torch.bfloat16)
    # Numpy doesn't have bf16 datatype so we have to trick it into int16
    t = t.view(torch.int16)
    b = t.numpy().tobytes()
    file.write(b)


def write_fp16(tensor: Tensor, file: BinaryIO) -> None:
    t = tensor.detach().cpu().to(torch.float16)
    b = t.numpy().tobytes()
    file.write(b)


def write_fp8(tensor: Tensor, file: BinaryIO) -> None:
    t = tensor.detach().cpu().to(torch.float8_e4m3fn)
    # Numpy doesn't have an fp8 datatype, so view as int8 (both are 1 byte).
    t = t.view(torch.int8)
    b = t.numpy().tobytes()
    file.write(b)


# TODO: This is a copy of Karpathy's code, it needs to be adapted to my classes.
def write_tensor(
    model_tensors: dict[str, Tensor], L: int, file: BinaryIO, dtype: str
) -> None:
    # Writes the GPT-2 model weights to a binary file.
    assert dtype in {"float32", "bfloat16", "float16", "float8"}
    write_fns = {
        "float32": write_fp32,
        "bfloat16": write_bf16,
        "float16": write_fp16,
        "float8": write_fp8,
    }
    write_fn = write_fns[dtype]
    write_fn(model_tensors["transformer.wte.weight"], file)  # [V, C]
    write_fn(model_tensors["transformer.wpe.weight"], file)  # [T, C]
    for i in range(L):  # [L, C]
        write_fn(model_tensors[f"transformer.h.{i}.ln_1.gamma"], file)
    for i in range(L):  # [L, C]
        write_fn(model_tensors[f"transformer.h.{i}.ln_1.beta"], file)
    for i in range(L):  # [L, 3C, C]
        write_fn(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L):  # [L, 3C]
        write_fn(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    for i in range(L):  # [L, C, C]
        write_fn(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L):  # [L, C]
        write_fn(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    for i in range(L):  # [L, C]
        write_fn(model_tensors[f"transformer.h.{i}.ln_2.gamma"], file)
    for i in range(L):  # [L, C]
        write_fn(model_tensors[f"transformer.h.{i}.ln_2.beta"], file)
    for i in range(L):  # [L, 4C, C]
        write_fn(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L):  # [L, 4C]
        write_fn(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    for i in range(L):  # [L, C, 4C]
        write_fn(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    for i in range(L):  # [L, C]
        write_fn(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fn(model_tensors["transformer.ln_f.gamma"], file)  # [C, ]
    write_fn(model_tensors["transformer.ln_f.beta"], file)  # [C, ]


def write_activations(
    activations: dict[str, Tensor], file: BinaryIO, dtype: str = "float32"
) -> None:
    """
    Writes captured forward activations and their grads. Two blobs per entry
    (activation, then grad). Entries are sorted by name.
    """
    assert dtype in {"float32", "bfloat16", "float16", "float8"}
    write_fns = {
        "float32": write_fp32,
        "bfloat16": write_bf16,
        "float16": write_fp16,
        "float8": write_fp8,
    }
    write_fn = write_fns[dtype]
    names = sorted(activations.keys())
    file.write(_uint32(len(names)))  # uint32 count
    for name in names:
        act = activations[name]
        grad = act.grad
        assert grad is not None, (
            f"activation {name!r} has no grad — call capture_activations() before "
            f"forward and backward() before write_activations()"
        )
        name_bytes = name.encode("utf-8")
        file.write(_uint32(len(name_bytes)))  # uint32 name length
        file.write(name_bytes)
        file.write(_int32(act.ndim))  # int32 ndim
        file.write(_int32_array(act.ndim, list(act.shape)))  # int32 * ndim shape
        write_fn(act.detach().cpu(), file)  # activation blob
        write_fn(grad.cpu(), file)  # grad blob


@torch.no_grad()
def pad_vocab(tensor: Tensor, multiple: int = 128, value: int = 0) -> Tensor:
    """
    The dimension of the vocab size in GPT-2 is 50,257 which is unfortunatly a
    very unfriendly number for many matrix operations on the GPU. So we pad to
    the nearest friendlier multiple, e.g. 50,304 if the multiple is 128 when we
    export to .bin files. This is a NOOP algorithmically and is only done to make
    tensor operations more efficient.
    """
    assert tensor.ndim == 2, "Tensor must be 2D"
    V, C = tensor.shape
    assert V == 50257, "Vocab size must be 50257"
    # Calculate the padded vocab size by rounding up to the nearest multiple
    V_p = ((V + multiple - 1) // multiple) * multiple
    # Pad the Tensor
    pad_rows = V_p - V
    padded = (
        tensor if pad_rows == 0 else F.pad(tensor, (0, 0, 0, pad_rows), value=value)
    )
    assert padded.shape == (V_p, C), "Padded shape mismatch"
    return padded


def write_model(model: Any, file: str, dtype: str) -> None:
    # Everything we need to instantiate the model.
    # 1) Header is: version int, GPTConfig ints, padding to 1024 bytes
    assert dtype in {"float32", "bfloat16", "float16", "float8"}
    version = {
        "float32": 3,  # 3: all tensors are fp32, padded vocab
        "bfloat16": 5,  # 5: all tensors are bf16, padded vocab
        "float16": 7,  # 7: all tensors are fp16, padded vocab
        "float8": 9,  # 9: all tensors are fp8 (e4m3fn), padded vocab
    }[dtype]
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = MAGIC_NUMBER
    header[1] = version
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_embd
    # 2) the parameters follow the header
    params = {name: param.cpu() for name, param in model.named_parameters()}
    # Pad the vocab to a multiple of 128 here at export, for efficiency in C.
    wte = params["transformer.wte.weight"]  # [V, C]
    wte_padded = pad_vocab(wte)  # [V_p, C]
    params["transformer.wte.weight"] = wte_padded  # [V_p, C]
    print(f"Padded vocab size from {wte.shape[0]} to {wte_padded.shape[0]}")
    header[7] = wte_padded.shape[0]  # Padded vocabsize stored in the header

    # 3) Write the parameters to the file
    with open(file, "wb") as f:
        f.write(header.numpy().tobytes())
        write_tensor(params, model.config.n_layer, f, dtype)  # Params
    print(f"Wrote {file}")


def write_state(
    model: GPT,
    x: Tensor,
    y: Tensor,
    logits: Tensor,
    loss: Tensor,
    activations: dict[str, Tensor],
    file: str,
) -> None:
    """
    Write state is used to debug. It contains information about the input, logits, loss, and the param gradients.
    This can be used for checking the computation correctness in the target language.
    """
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = MAGIC_NUMBER
    header[1] = (
        3  # Run state version = 3 (1 -> 2 for the padded vocab, 2 -> 3 for the activations)
    )
    header[2] = x.shape[0]  # Batch size
    header[3] = x.shape[1]  # Temporal extent of the batch (seq_len)
    grads = {
        name: param.grad.cpu()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    # Pad the vocab grads here as well, to mirror write_model
    wte_grad = grads["transformer.wte.weight"]  # [V, C]
    wte_grad_padded = pad_vocab(wte_grad, value=0)  # [V_p, C]
    grads["transformer.wte.weight"] = wte_grad_padded  # [V_p, C]
    print(
        f"Padded vocab size in reference grads from {wte_grad.shape[0]} to {wte_grad_padded.shape[0]}"
    )

    # Write the file
    with open(file, "wb") as f:
        # Header
        f.write(header.numpy().tobytes())
        # Input X
        f.write(x.cpu().numpy().astype(np.int32).tobytes())  # [B, T]
        # Target Y
        f.write(y.cpu().numpy().astype(np.int32).tobytes())  # [B, T]
        # Logits
        write_fp32(logits.cpu(), f)
        # Loss
        write_fp32(loss.cpu(), f)
        # Gradients
        write_tensor(grads, model.config.n_layer, f, "float32")
        # Activations + activation grads
        write_activations(activations, f, "float32")

    print(f"Wrote {file}")


def write_tokenizer(encoder, file: str) -> None:
    n = encoder.max_token_value + 1
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = MAGIC_NUMBER
    header[1] = 2  # Tokenizer version = 2 (1 -> 2: includes EOT token)
    header[2] = n  # Number of tokens
    header[3] = encoder.eot_token  # EOT token

    # Write the file
    with open(file, "wb") as f:
        f.write(header.numpy().tobytes())
        for i in range(n):
            b = encoder.decode_bytes([i])
            length = len(b)
            assert length < 256, f"Token length {length} exceeds 255"
            f.write(
                struct.pack("<B", length)
            )  # Writes the lenght as a 1-byte unsigned int (c++ struct utils)
            f.write(b)
        print(f"Wrote {file}")


def capture_activations(model: GPT) -> tuple[dict[str, Tensor], list]:
    """
    Registers forward hooks on a curated set of submodules so each module's output
    tensor and its activation grad survive the backward pass.

    lm_head is intentionally skipped — its output is the `logits` tensor, already
    written explicitly in write_state. The GPT root and the Transformer wrapper
    are skipped too (GPT returns a (logits, loss) tuple; Transformer has no own
    forward), so we hand-pick the modules that produce a single Tensor.
    """
    activations: dict[str, Tensor] = {}
    handles: list = []

    def capture_forward(name: str):
        def hook(
            _module: nn.Module, _input: tuple[Tensor, ...], output: Tensor
        ) -> None:
            output.retain_grad()
            activations[name] = output

        return hook

    targets: dict[str, nn.Module] = {
        "transformer.wte": model.transformer.wte,
        "transformer.wpe": model.transformer.wpe,
        "transformer.ln_f": model.transformer.ln_f,
    }
    for i, block in enumerate(model.transformer.h):
        kblock = cast(KarpathyBlock, block)
        targets[f"transformer.h.{i}.ln_1"] = kblock.ln_1
        targets[f"transformer.h.{i}.attn"] = kblock.attn
        targets[f"transformer.h.{i}.ln_2"] = kblock.ln_2
        targets[f"transformer.h.{i}.mlp"] = kblock.mlp
        targets[f"transformer.h.{i}"] = kblock  # post-residual stream output

    for name, module in targets.items():
        handles.append(module.register_forward_hook(capture_forward(name)))

    return activations, handles


if __name__ == "__main__":
    import time
    import argparse
    from data.utils import get_gpt2_encoding

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"

    print_zero_rank(f"Running pytorch {torch.__version__} on {device_name}")

    # GROSS! This should be handled by a real config parser.....
    def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
        # default settings will overfit a tiny batch of data
        # and save model weights and debug state to disk on the first iteration
        # file system input / output
        parser.add_argument(
            "--input_bin",
            type=str,
            default="data/.tinyshakespeare/tiny_shakespeare_val.bin",
            help="input .bin to train on",
        )
        parser.add_argument(
            "--input_val_bin",
            type=str,
            default="",
            help="input .bin to eval validation loss on",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="",
            help="output directory to which to write logs and checkpoints",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="gpt2",
            help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48",
        )
        # token layout for each step of the optimization
        parser.add_argument(
            "--batch_size",
            type=int,
            default=4,
            help="batch size, in units of #batch dimensions",
        )
        parser.add_argument(
            "--sequence_length", type=int, default=64, help="sequence length"
        )
        parser.add_argument(
            "--total_batch_size",
            type=int,
            default=256,
            help="total desired batch size, in units of #tokens",
        )
        # workload (number of steps)
        parser.add_argument(
            "--num_iterations", type=int, default=10, help="number of iterations to run"
        )
        parser.add_argument(
            "--inference_only", type=int, default=0, help="only run inference"
        )
        # optimization
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-4,
            help="learning rate warmup iterations",
        )
        parser.add_argument(
            "--warmup_iters",
            type=int,
            default=0,
            help="learning rate warmup iterations",
        )
        parser.add_argument(
            "--learning_rate_decay_frac",
            type=float,
            default=1.0,
            help="learning rate warmup iterations",
        )
        parser.add_argument(
            "--weight_decay", type=float, default=0.0, help="weight decay"
        )
        parser.add_argument(
            "--grad_clip", type=float, default=1.0, help="maximum gradient magnitude"
        )
        # evaluation
        parser.add_argument(
            "--val_loss_every",
            type=int,
            default=0,
            help="every how mant steps to evaluate val loss?",
        )
        parser.add_argument(
            "--val_max_steps",
            type=int,
            default=20,
            help="how many batches of val to average?",
        )
        parser.add_argument(
            "--sample_every",
            type=int,
            default=0,
            help="how often to sample from the model?",
        )
        # debugging
        parser.add_argument(
            "--overfit_single_batch",
            type=int,
            default=1,
            help="overfit just one batch of data",
        )
        # numerics
        parser.add_argument(
            "--tensorcores", type=int, default=0, help="use tensorcores"
        )
        # memory management
        parser.add_argument(
            "--device",
            type=str,
            default="",
            help="by default we autodetect, or set it here",
        )
        parser.add_argument(
            "--compile", type=int, default=0, help="torch.compile the model"
        )
        parser.add_argument("--flash", type=int, default=0, help="use flash attention")
        parser.add_argument(
            "--dtype", type=str, default="float32", help="float32|float16|bfloat16"
        )
        parser.add_argument(
            "--precision",
            type=str,
            default="",
            help=(
                "unified precision axis: fp32|tf32|bf16|fp16|fp8|nvfp4 "
                "(fp8/nvfp4 are CUDA-only). Empty (default) preserves the "
                "legacy --dtype/--tensorcores-driven behavior byte-for-byte, "
                "for backward compatibility with scripts/benchmark_train.py's "
                "MPS arm and any other existing caller. See "
                "docs/ai/pytorch_precision_support.md."
            ),
        )
        parser.add_argument(
            "--zero_stage",
            type=int,
            default=0,
            help="zero redundancy optimizer stage (0/1/2/3)",
        )
        # python -> C bridge
        parser.add_argument(
            "--write_tensors", type=int, default=1, help="write tensors to disk"
        )
        args = parser.parse_args()
        return args

    args = parse_args(argparse.ArgumentParser())

    # Args Error Handling and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024, "sequence length must be between 1 and 1024"
    assert args.dtype in {"float32", "float16", "bfloat16", "float8"}, "invalid dtype"
    _PRECISION_MODES = ("fp32", "tf32", "bf16", "fp16", "fp8", "nvfp4")
    assert args.precision == "" or args.precision in _PRECISION_MODES, (
        f"invalid --precision {args.precision!r}, expected '' (legacy "
        f"--dtype/--tensorcores behavior) or one of {_PRECISION_MODES}"
    )
    precision_active = args.precision  # "" => legacy path, byte-identical to before
    assert args.model in {
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "d12",
        "d24",
        "d36",
        "d48",
    }, "invalid model"

    # Setup Distributed Data Parallel. Torch run sets this environment variable.
    ddp = int(os.environ.get("RANK", -1) != -1)  # Is this a ddp run?

    if ddp:
        # Use of DDP at the moment demands CUDA, set the device appropriately. (This might now be not true)
        assert torch.cuda.is_available(), "DDP requires CUDA for now"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ.get("RANK", 0))
        ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # ddp_local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        # ddp_master_addr = os.environ.get("MASTER_ADDR", "localhost")
        # ddp_master_port = os.environ.get("MASTER_PORT", "29500")
        # ddp_device = f"cuda:{ddp_local_rank}"
        # ddp_num_nodes = int(os.environ.get("NODES", 1))
        # ddp_num_gpus = int(os.environ.get("GPUS", 1))
        # ddp_num_cpus = int(os.environ.get("CPUS", 1))
        # ddp_num_mems = int(os.environ.get("MEMS", 1))
        # ddp_num_disks = int(os.environ.get("DISKS", 1))
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (
            ddp_rank == 0
        )  # This process is for logging, checkpointing, and so on.
        seed_offset = 0  # All processes get the exact same seed
        zero_stage = args.zero_stage  # Zero redundancy optimizer stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        # Select the device
        if args.device:
            device = args.device
        else:
            # attempt to autodetect the device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    print(f"Using device: {device}")
    device_type = "cuda" if "cuda" in device else ("mps" if device == "mps" else "cpu")

    # Calculate the gradient accumulation from the desired total batch size and the current run configuration.
    tokens_per_fwd_bwd = B * T * ddp_world_size
    assert args.total_batch_size % tokens_per_fwd_bwd == 0, (
        "total batch size must be divisible by the number of tokens per forward-backward pass"
    )
    grad_accum_steps = args.total_batch_size // tokens_per_fwd_bwd
    print_zero_rank(f"Total desired batch size: {args.total_batch_size}")
    print_zero_rank(f"=> Gradient accumulation steps: {grad_accum_steps}")
    print_zero_rank(f"=> Tokens per forward-backward pass: {tokens_per_fwd_bwd}")
    print_zero_rank(f"=> Batch size: {B}")
    print_zero_rank(f"=> Sequence length: {T}")
    print_zero_rank(f"=> World size: {ddp_world_size}")
    print_zero_rank(f"=> Device: {device}")
    print_zero_rank(f"=> Device type: {device_type}")
    print_zero_rank(f"=> Zero stage: {zero_stage}")
    print_zero_rank(f"=> DDP rank: {ddp_rank}")
    print_zero_rank(f"=> DDP local rank: {ddp_local_rank}")
    print_zero_rank(f"=> Master process: {master_process}")
    print_zero_rank(f"=> Seed offset: {seed_offset}")

    # Set up a context manager following the desired dtype and device.
    #
    # `precision_active == ""` (the default, i.e. `--precision` not passed) runs
    # the ORIGINAL --dtype/--tensorcores-driven codepath below UNCHANGED — this
    # is load-bearing for scripts/benchmark_train.py's MPS arm and any other
    # existing caller, which never pass --precision (see
    # docs/ai/pytorch_precision_support.md "compatibility" section).
    scaler: Optional[torch.amp.GradScaler] = None
    if precision_active:
        # --- unified --precision path -------------------------------------
        if precision_active in ("fp8", "nvfp4"):
            assert device_type == "cuda", (
                f"--precision {precision_active} requires CUDA "
                f"(torch._scaled_mm's fp8/fp4 tensor-core dispatch); "
                f"device_type={device_type!r}"
            )
        use_tf32 = precision_active == "tf32"
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = use_tf32
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")

        if precision_active in ("bf16", "fp8", "nvfp4"):
            ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        elif precision_active == "fp16":
            ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float16)
        else:  # "fp32" / "tf32": strict or TF32-internal fp32 matmuls, no autocast
            ctx = contextlib.nullcontext()

        scaler = torch.amp.GradScaler(
            device=device_type if device_type in ("cuda", "cpu") else "cpu",
            enabled=(precision_active == "fp16" and device_type == "cuda"),
        )
        print_zero_rank(f"Precision: --precision={precision_active!r} (unified axis)")
    else:
        # --- legacy path (default) ------------------------------------------
        ptdtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            # "float8": torch.float8,
        }[args.dtype]
        # On MPS, torch.amp.autocast does not support float32 (warns and disables itself).
        # Use nullcontext for float32 on MPS to avoid noise; use proper MPS autocast for fp16/bf16.
        # On CPU, autocast with float32 is also a no-op so nullcontext is cleaner.
        if device_type in ("mps", "cpu") and ptdtype == torch.float32:
            ctx = contextlib.nullcontext()
            print_zero_rank(f"Precision: float32 (no autocast on {device_type})")
        else:
            ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
            print_zero_rank(
                f"Precision: {args.dtype} via torch.amp.autocast(device_type={device_type!r})"
            )

    # RNG Setup
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Set the torch percision mode to use TensorFloat32 for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    # (legacy knob; the unified --precision path above already set this itself
    # when active, so this is a no-op unless a caller passes BOTH flags.)
    if args.tensorcores:
        torch.set_float32_matmul_precision("high")

    # Toggle Flash Attention
    assert args.flash in {0, 1}, "flash must be 0 or 1"
    FLASH = args.flash
    print_zero_rank(f"Using Flash Attention: {FLASH}")

    # Init and write the tokenizer
    enc = get_gpt2_encoding()
    if master_process and args.write_tensors:
        write_tokenizer(enc, "gpt2_tokenizer.bin")

    # Init the model, from scratch or from OpenAI pretrained checkpoint
    model: nn.Module
    if args.model[0] == "d":
        # From scratch (random weights)
        model_config = {
            "d12": GPTConfig(
                block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768
            ),
            "d24": GPTConfig(
                block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024
            ),
            "d36": GPTConfig(
                block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280
            ),
            "d48": GPTConfig(
                block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600
            ),
        }[args.model]
        model = GPT(model_config)
    else:
        # Load the GPT-2 model weights
        model = GPT.from_pretrained(args.model)

    # Low-precision layer swap (fp8/nvfp4 only): must happen AFTER weight
    # init/loading above and BEFORE .to(device)/DDP-wrapping below. No-op
    # (swap_counts stays None) for every other --precision value and for the
    # legacy (precision_active == "") path.
    swap_counts = None
    if precision_active in ("fp8", "nvfp4"):
        swap_counts = swap_precision_layers(cast(GPT, model), precision_active)
    if precision_active:
        print_zero_rank(precision_banner(precision_active, device_type, swap_counts))

    # Set Model to train mode/device
    model.train()
    model.to(device)
    if args.compile:
        if device_type == "mps":
            print_zero_rank(
                "[NOTE] torch.compile is not supported on MPS (AssertionError: duplicate template name) — skipping compilation."
            )
        else:
            # Lazy import: pulling torch._inductor in at module load drags in the
            # dynamo/inductor stack, which hangs `import train_gpt2` and every test
            # that imports it. Only compiled non-MPS training needs this flag.
            import torch._inductor.config as inductor_config

            if hasattr(inductor_config, "coordinate_descent_tuning"):
                inductor_config.coordinate_descent_tuning = True  # @Chillee
            print_zero_rank("Compiling the model...")
            model = cast(nn.Module, torch.compile(model))

    """
    Our own version of simple DistributedDataLoader
    """

    # Load Tokens
    train_loader = DistributedDataLoader(
        args.input_bin,
        B,
        T,
        ddp_rank,
        ddp_world_size,
    )
    print_zero_rank(
        f"Loaded {train_loader.ntok_total} tokens from {len(train_loader.files)} files"
    )
    print_zero_rank(f"=> Batch size: {B}")
    print_zero_rank(f"=> Sequence length: {T}")
    print_zero_rank(f"=> World size: {ddp_world_size}")

    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(
            args.input_val_bin,
            B,
            T,
            ddp_rank,
            ddp_world_size,
        )
        print_zero_rank(
            f"Loaded {val_loader.ntok_total} tokens from {len(val_loader.files)} files"
        )
        print_zero_rank(f"=> Batch size: {B}")
        print_zero_rank(f"=> Sequence length: {T}")
        print_zero_rank(f"=> World size: {ddp_world_size}")

    """
    Pytorch -> C bridge: save some weights and sate for C to load later as referenced.
    """

    # Do a single forward pass to generate ground truth for our language test.
    if master_process and args.write_tensors and (not args.inference_only):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        activations, handles = capture_activations(cast(GPT, model))
        logits, loss = model(x, y)
        loss.backward()
        for h in handles:
            h.remove()
        # Save model params
        model_to_size = {
            "gpt2": "124M",
            "gpt2-medium": "355M",
            "gpt2-large": "774M",
            "gpt2-xl": "1558M",
        }
        model_to_size.update({f"d{d}": f"d{d}" for d in [12, 24, 36, 48]})
        model_size_str = model_to_size[args.model]  # e.g. "124M", or "d12"
        write_model(model, f"gpt2_{model_size_str}.bin", dtype="float32")
        write_model(model, f"gpt2_{model_size_str}_bf16.bin", dtype="bfloat16")
        write_model(model, f"gpt2_{model_size_str}_fp16.bin", dtype="float16")
        write_model(model, f"gpt2_{model_size_str}_fp8.bin", dtype="float8")
        # Save x, y, logits, loss, and parameter gradients, for debugging C
        # Always store these in fp32 to have an accurate reference (?)
        write_state(
            cast(GPT, model),
            x,
            y,
            logits,
            loss,
            activations,
            f"gpt2_{model_size_str}_debug_state.bin",
        )
        # Reset the train_loader for the optimization below
        train_loader.reset()

    """
    Training Loop
    """
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model: GPT = cast(
        GPT, model.module if ddp else model
    )  # Always holds the raw, unwrapped model

    # Init the optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device=torch.device(device),
        zero_stage=zero_stage,
    )
    print_zero_rank(f"Using optimizer: {optimizer.__class__.__name__}")
    print_zero_rank(f"=> Weight decay: {args.weight_decay}")
    print_zero_rank(f"=> Learning rate: {args.learning_rate}")

    # Learning Rate Decay Scheduler (cosine with warmup)
    def get_lr(
        iteration: int, lr: float, decay: float, warmup: int, num_iters: int
    ) -> float:
        # Minimum learning rate
        min_lr = lr * decay
        # Linear warmup
        if iteration < warmup:
            return lr * (iteration + 1) / warmup
        # If iteration is after total_iters, return min_lr
        if iteration > num_iters:
            return min_lr
        # If iteration is after warmup, use cosine decay
        decay_ratio = (iteration - warmup) / (num_iters - warmup)
        assert 0 <= decay_ratio <= 1, "decay_ration must be between 0 and 1"
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (lr - min_lr)

    # Create the logging directory if it doesn't exist
    logfile = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "log.txt")
        with open(logfile, "w") as f:
            pass

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if device == "mps":
        getattr(torch.mps, "reset_peak_memory_stats", lambda: None)()

    timings = []
    norm = -1.0
    num_iters = args.num_iterations
    val_loss_every = args.val_loss_every
    val_max_steps = args.val_max_steps
    learning_rate = args.learning_rate
    warmup = args.warmup_iters
    decay = args.learning_rate_decay_frac
    sample_every = args.sample_every

    for step in range(num_iters + 1):
        t_0 = time.time()
        last_step = step == num_iters

        # Occasionally evaluate the validation loss
        if val_loss_every > 0 and step % val_loss_every == 0 and val_loader is not None:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=False)
                    val_loss += loss.item()
                val_loss /= val_max_steps
            print_zero_rank(f"Validation loss: {val_loss:.4f}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))

        # Occasionally perform model inference on the master process
        if sample_every > 0 and step % sample_every == 0 and master_process:
            model.eval()
            start_ids = [enc.eot_token]
            xg = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            max_new_tokens = 32
            temperature = 1.0
            top_k = 40
            yg = raw_model.generate(xg, max_new_tokens, temperature, top_k)
            print_zero_rank("--------------------------------")
            print_zero_rank(enc.decode(yg[0].tolist()))
            print_zero_rank("--------------------------------")

        # BREAK so se don't train on the last step, just run evaluations or inference
        if last_step:
            break

        """
        Training Step
        """
        model.train()
        optimizer.zero_grad(
            set_to_none=True
        )  # Always zero the gradients before doing a forward pass

        # If we want to overfit a single batch, do so here
        if args.overfit_single_batch:
            train_loader.reset()

        # Micro-batch loop where we do gradient accumulation to reach the desired total batch size
        lossf: Tensor = torch.tensor(0.0, device=device)
        for micro_step in range(grad_accum_steps):
            # Fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            if ddp:
                # We only want the final micro-step to sync grads in DDP model.
                # The library way to do this is with model.no_sync() but the context manager bloats the code.
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

            # Forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                # Karpathy's NOTE:
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach()  # track the mean loss
            if not args.inference_only:
                if scaler is not None:
                    cast(Tensor, scaler.scale(loss)).backward()
                else:
                    loss.backward()

        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.SUM)

        # `scaler` is only non-None on the unified --precision path (see its
        # construction above); it is a real GradScaler there, but only
        # ACTUALLY enabled for --precision fp16 (its .scale()/.unscale_()/
        # .step()/.update() are documented no-ops-that-fall-through-to-the-
        # plain-optimizer-call when disabled — see torch.amp.GradScaler's
        # docstring). The legacy (precision_active == "") path never
        # constructs a scaler, so it always takes the `else` branch below,
        # byte-identical to before this flag existed.
        if scaler is not None:
            scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # Determine and set the learning rate for this iteration
        lr = get_lr(step, learning_rate, decay, warmup, num_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # Step the optimizer
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        """
        Training step complete
        """

        # Wait on the CPU for all device work to end so we get an accurate per-iteration timing
        if device == "cuda":
            torch.cuda.synchronize()
        if device == "mps":
            torch.mps.synchronize()

        # Time and Print
        t_1 = time.time()

        # The 0th interation is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t_1 - t_0)
        print_zero_rank(
            f"step {step + 1:4d}/{args.num_iterations} | train loss {lossf.item():.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t_1 - t_0) * 1000:.2f} ms | {tokens_per_second:.0f} tok/s)"
        )

        # Log to the logfile
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf.item()))

        # Keep track of smooth timings last 20 iterations
        if step > 0 and step > num_iters - 20:
            timings.append(t_1 - t_0)

    # Print the average of the last 20 timings to get somting smooth-esque
    timings = timings[-20:]
    if timings:
        print_zero_rank(
            f"Average of last 20 timings: {sum(timings) / len(timings):.2f} ms"
        )
    if device_type == "cuda":
        print_zero_rank(
            f"Peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
        )
    elif device_type == "mps":
        peak_mem = torch.mps.driver_allocated_memory()
        print_zero_rank(f"Peak MPS memory (driver): {peak_mem // 1024 // 1024} MiB")

    # Cleanup
    if ddp:
        destroy_process_group()
