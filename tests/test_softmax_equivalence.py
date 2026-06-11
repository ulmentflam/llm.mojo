"""Equivalence: Mojo softmax_fwd vs PyTorch and vs Modular's own kernel.

Forward-only for now — these pin the numerics the backward will
differentiate against. Three properties matter beyond plain closeness:

  *  V vs Vp: the kernel must reduce over the real vocab columns only.
     Inputs fill the padded tail with huge garbage so a kernel that reads
     past V fails loudly (the garbage would become the row max and zero
     every real probability).
  *  The padded tail of the output buffer is never written (llm.c zeroes
     it once at init and relies on it staying zero).
  *  Parity with Modular's production softmax (`max.graph.ops.softmax`,
     the nn.softmax kernels this implementation was reviewed against),
     not just with PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tests._dtypes import DTYPE_TOLERANCES, TORCH_DTYPES, from_storage, to_storage
from tests.kernels import softmax


@dataclass(frozen=True)
class Case:
    """Fixed seed + shapes + dtype uniquely determine the logits."""

    name: str
    batch_size: int  # B
    seq_len: int  # T
    vocab_size: int  # V — the reduction runs over [0, V)
    vocab_size_padded: int  # Vp — row stride; the tail is garbage by design
    dtype: str  # "float32" | "bfloat16" | "float16"
    seed: int


CASES: tuple[Case, ...] = (
    Case(
        "fp32_small",
        batch_size=4,
        seq_len=8,
        vocab_size=50,
        vocab_size_padded=64,
        dtype="float32",
        seed=0,
    ),
    Case(
        "fp32_no_padding",
        batch_size=2,
        seq_len=16,
        vocab_size=32,
        vocab_size_padded=32,
        dtype="float32",
        seed=1,
    ),
    # V % simd_width != 0 exercises the scalar tail of the online pass.
    Case(
        "fp32_odd_sizes",
        batch_size=3,
        seq_len=7,
        vocab_size=13,
        vocab_size_padded=16,
        dtype="float32",
        seed=2,
    ),
    # V smaller than the simd width: the vector loop never runs, the whole
    # row goes through the tail path.
    Case(
        "fp32_tiny_vocab",
        batch_size=2,
        seq_len=3,
        vocab_size=3,
        vocab_size_padded=8,
        dtype="float32",
        seed=3,
    ),
    # Real GPT-2 row shape: V=50257 (odd), Vp=50304.
    Case(
        "fp32_gpt2_shape",
        batch_size=1,
        seq_len=4,
        vocab_size=50257,
        vocab_size_padded=50304,
        dtype="float32",
        seed=4,
    ),
    Case(
        "bf16_small",
        batch_size=4,
        seq_len=8,
        vocab_size=50,
        vocab_size_padded=64,
        dtype="bfloat16",
        seed=5,
    ),
)

# The kernel must never read past V, so every padded input slot holds a
# value that would dominate the row max and zero out the real probs.
PAD_POISON = 1.0e30


def _make_logits(case: Case) -> np.ndarray:
    """(BT, Vp) fp32 logits, dtype-rounded, padded tail poisoned.

    Round-tripped through the kernel dtype so the kernel and both
    references consume bit-identical values; any mismatch is then the
    kernel's own arithmetic, not input quantization.
    """
    g = torch.Generator().manual_seed(case.seed)
    bt = case.batch_size * case.seq_len
    logits = torch.randn(bt, case.vocab_size_padded, generator=g) * 4.0
    if case.vocab_size_padded > case.vocab_size:
        logits[:, case.vocab_size :] = PAD_POISON
    logits = logits.to(TORCH_DTYPES[case.dtype]).to(torch.float32)
    return logits.numpy()


def _reference(logits_f32: np.ndarray, vocab_size: int) -> np.ndarray:
    """fp32 torch softmax over the real vocab columns only."""
    x = torch.from_numpy(logits_f32[:, :vocab_size])
    return torch.softmax(x, dim=1).numpy()


def _run_kernel(case: Case, logits_f32: np.ndarray, probs: np.ndarray | None = None):
    return softmax.forward(
        logits=to_storage(logits_f32.reshape(-1), case.dtype),
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        vocab_size=case.vocab_size,
        vocab_size_padded=case.vocab_size_padded,
        dtype_name=case.dtype,
        probs=probs,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_forward_matches_torch(case: Case):
    logits_f32 = _make_logits(case)
    expected = _reference(logits_f32, case.vocab_size)

    got = _run_kernel(case, logits_f32)

    bt = case.batch_size * case.seq_len
    got_v = from_storage(got, case.dtype).reshape(bt, case.vocab_size_padded)[
        :, : case.vocab_size
    ]

    # The kernel accumulates in fp32 and rounds to the kernel dtype only on
    # store, so the comparison tolerance follows the case dtype.
    tol = DTYPE_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        got_v,
        expected,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: forward probs diverged from torch",
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_forward_matches_modular(case: Case):
    """Cross-check against Modular's production softmax kernel.

    ops.softmax has no V/Vp split, so it sees only the real vocab columns.
    Their CPU path accumulates in the input dtype while ours accumulates
    in fp32, so the bf16 comparison leans on the bf16 tolerance; in fp32
    the two should agree to rounding.
    """
    logits_f32 = _make_logits(case)
    bt = case.batch_size * case.seq_len

    theirs = softmax.modular_forward(
        to_storage(np.ascontiguousarray(logits_f32[:, : case.vocab_size]), case.dtype),
        case.dtype,
    )

    got = _run_kernel(case, logits_f32)
    got_v = from_storage(got, case.dtype).reshape(bt, case.vocab_size_padded)[
        :, : case.vocab_size
    ]

    tol = DTYPE_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        got_v,
        from_storage(theirs, case.dtype),
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: forward probs diverged from Modular's softmax",
    )


def test_padding_left_untouched():
    """The kernel writes [0, V) of each row and must leave the padded tail
    byte-identical — llm.c zeroes the tail once and relies on it staying
    zero. Sentinel-fill the output buffer to catch stray writes."""
    case = next(c for c in CASES if c.name == "fp32_small")
    logits_f32 = _make_logits(case)
    bt = case.batch_size * case.seq_len

    sentinel = np.float32(7.0)
    probs_in = np.full(bt * case.vocab_size_padded, sentinel, dtype=np.float32)

    got = _run_kernel(case, logits_f32, probs=probs_in).reshape(
        bt, case.vocab_size_padded
    )

    np.testing.assert_array_equal(
        got[:, case.vocab_size :],
        np.full((bt, case.vocab_size_padded - case.vocab_size), sentinel),
        err_msg="kernel wrote into the padded vocab tail",
    )


def test_rows_sum_to_one():
    """Each row of probs must be a distribution over V — this is the
    invariant the crossentropy forward consumes."""
    case = next(c for c in CASES if c.name == "fp32_gpt2_shape")
    logits_f32 = _make_logits(case)
    bt = case.batch_size * case.seq_len

    got = _run_kernel(case, logits_f32).reshape(bt, case.vocab_size_padded)

    # Each stored prob carries ~1e-7 relative rounding error; summed over
    # V≈50k that legitimately reaches a few 1e-5, so the bound is 1e-4 —
    # tight enough to catch a wrong normalizer, loose enough for fp32.
    sums = got[:, : case.vocab_size].astype(np.float64).sum(axis=1)
    np.testing.assert_allclose(
        sums, np.ones(bt), atol=1e-4, rtol=0, err_msg="rows do not sum to 1"
    )


def test_uniform_logits_is_uniform():
    """Analytic anchor: constant logits ⇒ every prob = 1/V, independent of
    the constant (the max subtraction must cancel it exactly)."""
    case = Case(
        "uniform",
        batch_size=2,
        seq_len=4,
        vocab_size=64,
        vocab_size_padded=64,
        dtype="float32",
        seed=42,
    )
    bt = case.batch_size * case.seq_len
    logits = np.full((bt, case.vocab_size_padded), 123.456, dtype=np.float32)

    got = _run_kernel(case, logits).reshape(bt, case.vocab_size_padded)

    tol = DTYPE_TOLERANCES["float32"]
    np.testing.assert_allclose(
        got,
        np.full_like(logits, 1.0 / case.vocab_size),
        atol=tol["atol"],
        rtol=tol["rtol"],
    )
