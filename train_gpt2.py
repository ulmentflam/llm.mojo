#!/usr/bin/env python3
import os
import math
import glob
import struct
import inspect
from dataclasses import dataclass
from typing import BinaryIO, Optional, Any, cast

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
import torch.nn.init as init
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, init_process_group


"""
Print Helper
"""


def print_zero_rank(*args, **kwargs) -> None:
    if torch.distributed.get_rank() == 0:
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
        return x @ self.weight.T + self.bias


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
        self.c_fc = Linear(input_dim, hidden_dim, bias=False)
        self.gelu = GeLU()
        self.c_proj = Linear(hidden_dim, out_dim, bias=False)
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
        self.c_attn = Linear(d_model, 3 * d_model, bias=False)
        # Output projection
        self.c_proj = Linear(d_model, d_model, bias=False)
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
            loss = None

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


# def write_fp8(tensor: Tensor, file: BinaryIO) -> None:
#     t = tensor.detach().cpu().to(torch.float8)
#     b = t.numpy().tobytes()
#     file.write(b)


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
        # "float8": write_fp8,
    }
    write_fn = write_fns[dtype]
    write_fn(model_tensors["transformer.wte.weight"], file)  # [V, C]
    write_fn(model_tensors["transformer.wpe.weight"], file)  # [T, C]
    for i in range(L):  # [L, C]
        write_fn(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
        write_fn(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    for i in range(L):  # [L, 3C, C]
        write_fn(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L):  # [L, 3C]
        write_fn(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    for i in range(L):  # [L, C, C]
        write_fn(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L):  # [L, C]
        write_fn(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
        write_fn(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
        write_fn(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    for i in range(L):  # [L, 4C, C]
        write_fn(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L):  # [L, 4C]
        write_fn(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    for i in range(L):  # [L, C, 4C]
        write_fn(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    for i in range(L):  # [L, C]
        write_fn(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fn(model_tensors["transformer.ln_f.weight"], file)  # [C, ]
    write_fn(model_tensors["transformer.ln_f.bias"], file)  # [C, ]


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
    model: GPT, x: Tensor, y: Tensor, logits: Tensor, loss: Tensor, file: str
) -> None:
    """
    Write state is used to debug. It contains information about the input, logits, loss, and the param gradients.
    This can be used for checking the computation correctness in the target language.
    """
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = MAGIC_NUMBER
    header[1] = 2  # Run state version = 2 (1 -> 2 for the padded vocab)
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


if __name__ == "__main__":
    import time
    import argparse
    import tiktoken

    print_zero_rank(
        f"Running pytorch {torch.__version__} on {torch.cuda.get_device_name()}"
    )

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
    device_type = "cuda" if "cuda" in device else "cpu"

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
    ptdtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        # "float8": torch.float8,
    }[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # RNG Setup
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Set the torch percision mode to use TensorFloat32 for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision("high")

    # Toggle Flash Attention
    assert args.flash in {0, 1}, "flash must be 0 or 1"
    FLASH = args.flash
    print_zero_rank(f"Using Flash Attention: {FLASH}")

    # Init and write the tokenizer
    enc = tiktoken.get_encoding("gpt2")
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

    # Set Model to train mode/device
    model.train()
    model.to(device)
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True  # suggested by @Chillee
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
        logits, loss = model(x, y)
        loss.backward()
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
                loss.backward()

        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.SUM)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # Determine and set the learning rate for this iteration
        lr = get_lr(step, learning_rate, decay, warmup, num_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # Step the optimizer
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
    print_zero_rank(f"Average of last 20 timings: {sum(timings) / len(timings):.2f} ms")
    print_zero_rank(
        f"Peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )

    # Cleanup
    if ddp:
        destroy_process_group()
