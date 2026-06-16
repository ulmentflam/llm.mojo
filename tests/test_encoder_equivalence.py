"""Equivalence: Mojo encoder_{fwd,bwd} vs PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tests._dtypes import DTYPE_TOLERANCES, TORCH_DTYPES, from_storage, to_storage
from tests.kernels import encoder


@dataclass(frozen=True)
class Case:
    """Fixed seed + shape + dtype uniquely determine the inputs."""

    name: str
    batch_size: int
    seq_len: int
    vocab_size: int
    channels: int
    dtype: str  # "float32" | "bfloat16" | "float16"
    seed: int


CASES: tuple[Case, ...] = (
    # Small size, odd channels to hit vectorization tails
    Case(
        "fp32_small",
        batch_size=2,
        seq_len=5,
        vocab_size=37,
        channels=33,
        dtype="float32",
        seed=42,
    ),
    # Power of 2 channels, normal shapes
    Case(
        "fp32_large",
        batch_size=4,
        seq_len=64,
        vocab_size=256,
        channels=128,
        dtype="float32",
        seed=43,
    ),
    # BF16 small test case
    Case(
        "bf16_small",
        batch_size=2,
        seq_len=8,
        vocab_size=64,
        channels=64,
        dtype="bfloat16",
        seed=44,
    ),
)


def _ids(case: Case) -> str:
    return case.name


def _make_inputs(case: Case) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate mock inputs for the encoder."""
    g = torch.Generator().manual_seed(case.seed)
    td = TORCH_DTYPES[case.dtype]

    # Input token indices (0 to vocab_size - 1)
    inp = torch.randint(
        0, case.vocab_size, (case.batch_size, case.seq_len), generator=g
    ).to(torch.int32)

    # Embedding weights
    wte = torch.randn(case.vocab_size, case.channels, generator=g).to(td)
    wpe = torch.randn(case.seq_len, case.channels, generator=g).to(td)

    # Output gradient (dout)
    dout = torch.randn(case.batch_size, case.seq_len, case.channels, generator=g).to(td)

    # Cast to float32 numpy to perform clean PyTorch references and then
    # storage convert to the target dtype
    return (
        inp.numpy(),
        wte.to(torch.float32).numpy(),
        wpe.to(torch.float32).numpy(),
        dout.to(torch.float32).numpy(),
    )


def _build_wte_buckets(
    inp: np.ndarray, vocab_size: int, channels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Construct deterministic workload buckets and flat list of indices.

    Replicates the logic of llm.c's token gradient bucket accumulation.
    """
    B, T = inp.shape
    # Find positions for each token
    token_positions = [[] for _ in range(vocab_size)]
    for b in range(B):
        for t in range(T):
            token = int(inp[b, t])
            token_positions[token].append(b * T + t)

    workload_indices = []
    bucket_info = []

    # Channel groups: each warp processes 32 * width channels (width is 4)
    width = 4
    c_per_warp = 32 * width  # 128
    num_channel_groups = (channels + c_per_warp - 1) // c_per_warp

    for token in range(vocab_size):
        positions = token_positions[token]
        size = len(positions)
        if size == 0:
            continue
        start_idx = len(workload_indices)
        workload_indices.extend(positions)

        for g in range(num_channel_groups):
            # bucket_info: start_idx, size, token_idx, channel_group
            bucket_info.append([start_idx, size, token, g])

    # If no buckets/indices were created, return empty int32 arrays
    if not bucket_info:
        return np.empty((0, 4), dtype=np.int32), np.empty((0,), dtype=np.int32)

    return np.array(bucket_info, dtype=np.int32), np.array(
        workload_indices, dtype=np.int32
    )


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_forward_matches_torch(case: Case) -> None:
    inp, wte, wpe, _ = _make_inputs(case)

    # Convert to storage format
    wte_storage = to_storage(wte, case.dtype)
    wpe_storage = to_storage(wpe, case.dtype)

    out = encoder.forward(
        inp=inp, wte=wte_storage, wpe=wpe_storage, dtype_name=case.dtype
    )

    # Torch reference
    wte_t = torch.from_numpy(wte)
    wpe_t = torch.from_numpy(wpe)
    inp_t = torch.from_numpy(inp).long()
    expected = wte_t[inp_t] + wpe_t[torch.arange(case.seq_len, device=inp_t.device)]

    tol = DTYPE_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        from_storage(out, case.dtype),
        expected.numpy(),
        atol=tol["atol"],
        rtol=tol["rtol"],
    )


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_backward_matches_torch(case: Case) -> None:
    inp, wte, wpe, dout = _make_inputs(case)

    # Setup PyTorch reference
    wte_t = torch.from_numpy(wte).requires_grad_(True)
    wpe_t = torch.from_numpy(wpe).requires_grad_(True)
    inp_t = torch.from_numpy(inp).long()
    dout_t = torch.from_numpy(dout)

    out_t = wte_t[inp_t] + wpe_t[torch.arange(case.seq_len, device=inp_t.device)]
    out_t.backward(gradient=dout_t)

    assert wte_t.grad is not None
    assert wpe_t.grad is not None
    expected_dwte = wte_t.grad.numpy()
    expected_dwpe = wpe_t.grad.numpy()

    # Setup workload buckets for Mojo backward
    bucket_info, workload_indices = _build_wte_buckets(
        inp, case.vocab_size, case.channels
    )

    # Allocate clean zeroed outputs for Mojo
    dwte = np.zeros_like(wte)
    dwpe = np.zeros_like(wpe)

    dwte_storage = to_storage(dwte, case.dtype)
    dwpe_storage = to_storage(dwpe, case.dtype)
    dout_storage = to_storage(dout, case.dtype)

    out_dwte_storage, out_dwpe_storage = encoder.backward(
        dwte=dwte_storage,
        dwpe=dwpe_storage,
        bucket_info=bucket_info,
        workload_indices=workload_indices,
        dout=dout_storage,
        dtype_name=case.dtype,
    )

    out_dwte = from_storage(out_dwte_storage, case.dtype)
    out_dwpe = from_storage(out_dwpe_storage, case.dtype)

    tol = DTYPE_TOLERANCES[case.dtype]

    np.testing.assert_allclose(
        out_dwte,
        expected_dwte,
        atol=tol["atol"],
        rtol=tol["rtol"],
    )
    np.testing.assert_allclose(
        out_dwpe,
        expected_dwpe,
        atol=tol["atol"],
        rtol=tol["rtol"],
    )
