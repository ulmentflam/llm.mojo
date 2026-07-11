#!/usr/bin/env python
"""MLX GPT-2 124M training-throughput benchmark (Apple Silicon / Metal).

The Metal benchmark's third framework, alongside llm.mojo and PyTorch MPS. MLX
(Apple's array framework) runs natively on the Metal GPU with lazy evaluation,
so it is the closest "native Apple" point of comparison. This script trains a
from-scratch GPT-2 124M (d12) for a fixed number of steps on one reused batch
(the overfit-single-batch setup the other arms use) and prints one timing line
per step in the SAME format the PyTorch arm uses, so benchmark_train.py's
`_TORCH_RE` (`(<ms> ms | <tok/s> tok/s)`) parses it unchanged:

    step N: loss L.LLLL (MM.MM ms | TTTT tok/s)

One step = forward + backward + AdamW update, timed end to end with mx.eval()
forcing the whole lazy graph (params, optimizer state, loss) to materialize on
the GPU before the clock stops — the MLX analogue of torch.mps.synchronize().

fp32 keeps parameters/compute in float32. bf16 casts the model (and therefore
optimizer state) to bfloat16; MLX has no autocast, so this is bf16 *parameters*
rather than the fp32-master/bf16-compute autocast the PyTorch arm uses — noted
here because it is the one axis on which the two bf16 bars are not apples-to-
apples. Throughput, not convergence, is the metric, so bf16 masters are moot.
"""

from __future__ import annotations

import argparse
import math
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def __call__(self, x, mask):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        hd = C // self.n_head
        # (B, T, C) -> (B, n_head, T, head_dim) so attention batches over heads.
        q = q.reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(hd), mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(out)


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def __call__(self, x):
        return self.c_proj(nn.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def __call__(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab: int, n_layer: int, n_head: int, n_embd: int, block: int):
        super().__init__()
        self.wte = nn.Embedding(vocab, n_embd)
        self.wpe = nn.Embedding(block, n_embd)
        self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab, bias=False)

    def __call__(self, idx):
        _, T = idx.shape
        x = self.wte(idx) + self.wpe(mx.arange(T))
        # Additive causal mask, built in the model's dtype so bf16 runs stay bf16.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(x.dtype)
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.ln_f(x))


def build_model(dtype):
    # GPT-2 124M ("d12"): matches the shape the llm.mojo and PyTorch MPS arms run.
    model = GPT(vocab=50257, n_layer=12, n_head=12, n_embd=768, block=1024)
    if dtype != mx.float32:
        model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--num-iterations", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3, help="untimed graph-warmup steps")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = ap.parse_args()

    dtype = mx.float32 if args.dtype == "float32" else mx.bfloat16
    B, T = args.batch_size, args.seq_len
    vocab = 50257

    model = build_model(dtype)
    optimizer = optim.AdamW(learning_rate=3e-4, betas=[0.9, 0.95], weight_decay=0.0)

    # One fixed batch, reused every step (overfit-single-batch: isolates compute
    # throughput from any data pipeline, matching the other arms' --overfit).
    mx.random.seed(1337)
    idx = mx.random.randint(0, vocab, (B, T))
    targets = mx.random.randint(0, vocab, (B, T))
    mx.eval(idx, targets)

    def loss_fn(model, idx, targets):
        logits = model(idx).astype(mx.float32)
        return mx.mean(
            nn.losses.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step():
        loss, grads = loss_and_grad(model, idx, targets)
        optimizer.update(model, grads)
        return loss

    for _ in range(args.warmup):
        loss = step()
        mx.eval(model.parameters(), optimizer.state, loss)

    tokens = B * T
    print(
        f"[mlx] GPT-2 124M B={B} T={T} dtype={args.dtype} "
        f"warmup={args.warmup} steps={args.num_iterations}",
        flush=True,
    )
    for i in range(args.num_iterations):
        t0 = time.perf_counter()
        loss = step()
        mx.eval(model.parameters(), optimizer.state, loss)
        dt = time.perf_counter() - t0
        print(
            f"step {i}: loss {float(loss):.4f} "
            f"({dt * 1000.0:.2f} ms | {tokens / dt:.0f} tok/s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
