import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F, init
from torch import Tensor

"""
Functions (F)
"""

def softmax(input: Tensor, dim: int = -1) -> Tensor:
    # The equation for softmax is \frac{\exp{x_i}}{\sum_{j}\exp{x_j}}
    max_val, _ = torch.max(input, dim=dim, keepdim=True) # Asymptotically O(N) but is parallelized
    stabilized_x = input - max_val  # This prevents us from exponentiating a very large value.
    x_i = torch.exp(stabilized_x) 
    sum_x_j = torch.sum(x_i, dim=dim, keepdim=True)
    return x_i/sum_x_j


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    # Q, K, V of shapes [B = batch_size, num_heads, T = seq_len, head_dims]
    d_k = K.size(-1)
    # The Standard Attention formula is \softmax{\frac{QK^T}{\sqrt{d_k}}}V

    # Calculate the attention scores
    # Q @ K^T
    # [B, num_heads, T, head_dims] @ [B, num_heads, head_dims, T] -> [B, num_heads, T, T]
    QK = Q @ K.transpose(-2, -1) # We can't use K.T because the batch_size and num_heads need to stay in place.
    # For single head attention the tensors would be in the shape [B, C, T] so we could use K.T

    score = QK/torch.sqrt(d_k)

    # Apply a mask (most commonly causal) if provided
    if mask is not None:
        # Mask out future tokens by setting them to negative infinity before softmax.
        scores = scores.masked_fill(mask == 0, float('-inf'))

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
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None
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
        dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.epislon = epislon

        self.gamma = nn.Paramater(torch.ones((in_features)))
        self.beta = nn.Paramater(torch.zeros((in_features)))

    def forward(self, x: Tensor) -> Tensor:
        # Compute the mean across the feature dimension
        sigma, u = torch.var_mean(x, dim=-1, keepdim=True)
        # Stabilize x 
        stablized_x = x - u
        # Square root of variance
        sqrt_sigma = torch.sqrt(sigma + self.epislon)
        
        return (stabilized_x/sqrt_sigma) * self.gamma + self.beta 


class ReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, min=0) # Clamp is element wise max

class GeLU(nn.Module):
    CONSTANT: torch.float = 0.044715
    TWO_OVER_PI: torch.float = (2.0 / math.pi)

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(torch.sqrt(TWO_OVER_PI) * (x + CONSTANT * torch.pow(x, 3))))


class Sequential(nn.Module):
    def __init__(self, *args) -> None
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self.children():
            x = module(x)
        return x

class Dropout(nn.Module):
    __constants__ = ['probability']
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
    __constants__ = ['input_dim', 'hidden_dim', 'out_dim']
    input_dim: int
    hidden_dim: int
    out_dim: int

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: Optional[torch.float] = None 
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        ordered_layers = [
            Linear(input_dim, hidden_dim),
            GeLU(),
            Linear(hidden_dim, input_dim)
        ]
        if dropout:
            ordered_layers.append(Dropout(dropout))
        self.layers = Sequential(
            *ordered_layers
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

# TODO: MultiHead and Causal Attention
# TODO: GPT2Block


"""
GPT For Causal LM
"""
# TODO: GPT




