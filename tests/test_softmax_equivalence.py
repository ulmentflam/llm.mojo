"""Equivalence: Mojo softmax_{fwd,bwd} vs PyTorch and vs Modular's kernel.

The forward is checked against torch and Modular's production softmax;
the backward against the analytic formula and end-to-end torch autograd
(the inference graph API exposes no softmax backward to compare with).
Three properties matter beyond plain closeness:

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


# ---------------------------------------------------------------------------
# Backward: d_logits = probs * (d_probs - <d_probs, probs>)
#
# No Modular cross-check here — the inference graph API exposes softmax
# forward only, so the references are the analytic formula (on the same
# dtype-rounded inputs the kernel reads) and end-to-end torch autograd.
# ---------------------------------------------------------------------------


def _make_bwd_inputs(case: Case) -> tuple[np.ndarray, np.ndarray]:
    """(probs, d_probs) as (BT, Vp) fp32, dtype-rounded, padding poisoned.

    Probs are a real softmax over V so the row is a valid distribution;
    d_probs is an arbitrary dense upstream gradient. Both paddings hold
    PAD_POISON so a dot pass that strays past V fails loudly.
    """
    g = torch.Generator().manual_seed(case.seed + 100)
    bt = case.batch_size * case.seq_len

    logits = torch.randn(bt, case.vocab_size, generator=g) * 4.0
    probs = torch.full((bt, case.vocab_size_padded), PAD_POISON)
    probs[:, : case.vocab_size] = torch.softmax(logits, dim=1)

    d_probs = torch.randn(bt, case.vocab_size_padded, generator=g)
    if case.vocab_size_padded > case.vocab_size:
        d_probs[:, case.vocab_size :] = PAD_POISON

    probs = probs.to(TORCH_DTYPES[case.dtype]).to(torch.float32)
    d_probs = d_probs.to(TORCH_DTYPES[case.dtype]).to(torch.float32)
    return probs.numpy(), d_probs.numpy()


def _reference_bwd(
    probs_f32: np.ndarray, d_probs_f32: np.ndarray, vocab_size: int
) -> np.ndarray:
    """Analytic softmax backward over the real vocab columns, fp32."""
    p = torch.from_numpy(probs_f32[:, :vocab_size])
    g = torch.from_numpy(d_probs_f32[:, :vocab_size])
    dot = (p * g).sum(dim=1, keepdim=True)
    return (p * (g - dot)).numpy()


def _run_bwd(
    case: Case,
    probs_f32: np.ndarray,
    d_probs_f32: np.ndarray,
    d_logits: np.ndarray | None = None,
):
    return softmax.backward(
        d_probs=to_storage(d_probs_f32.reshape(-1), case.dtype),
        probs=to_storage(probs_f32.reshape(-1), case.dtype),
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        vocab_size=case.vocab_size,
        vocab_size_padded=case.vocab_size_padded,
        dtype_name=case.dtype,
        d_logits=d_logits,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_backward_matches_formula(case: Case):
    probs_f32, d_probs_f32 = _make_bwd_inputs(case)
    expected = _reference_bwd(probs_f32, d_probs_f32, case.vocab_size)

    got = _run_bwd(case, probs_f32, d_probs_f32)

    bt = case.batch_size * case.seq_len
    got_v = from_storage(got, case.dtype).reshape(bt, case.vocab_size_padded)[
        :, : case.vocab_size
    ]

    tol = DTYPE_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        got_v,
        expected,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: backward d_logits diverged from the formula",
    )


def test_backward_matches_autograd():
    """End-to-end gradient check: torch autograd through torch.softmax must
    agree with the kernel applied to torch's own probs. fp32 only, where
    the dtype round-trip is the identity and the comparison is exact up to
    arithmetic order."""
    case = next(c for c in CASES if c.name == "fp32_small")
    g = torch.Generator().manual_seed(case.seed + 200)
    bt = case.batch_size * case.seq_len

    x = (torch.randn(bt, case.vocab_size, generator=g) * 4.0).requires_grad_(True)
    p = torch.softmax(x, dim=1)
    upstream = torch.randn(bt, case.vocab_size, generator=g)
    p.backward(upstream)
    assert x.grad is not None

    probs_f32 = np.full((bt, case.vocab_size_padded), PAD_POISON, dtype=np.float32)
    probs_f32[:, : case.vocab_size] = p.detach().numpy()
    d_probs_f32 = np.full((bt, case.vocab_size_padded), PAD_POISON, dtype=np.float32)
    d_probs_f32[:, : case.vocab_size] = upstream.numpy()

    got = _run_bwd(case, probs_f32, d_probs_f32).reshape(bt, case.vocab_size_padded)

    tol = DTYPE_TOLERANCES["float32"]
    np.testing.assert_allclose(
        got[:, : case.vocab_size],
        x.grad.numpy(),
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg="backward diverged from torch autograd",
    )


def test_backward_one_hot_collapse():
    """With a one-hot upstream gradient (crossentropy-shaped: d_probs =
    val at the target, 0 elsewhere) the backward collapses to
    d_logits = val * p_t * (onehot - p) — the identity behind the fused
    classifier. Pin it before fusing."""
    case = next(c for c in CASES if c.name == "fp32_small")
    probs_f32, _ = _make_bwd_inputs(case)
    bt = case.batch_size * case.seq_len

    rng = np.random.default_rng(case.seed + 300)
    targets = rng.integers(0, case.vocab_size, size=bt)
    vals = rng.standard_normal(bt).astype(np.float32)

    d_probs_f32 = np.zeros((bt, case.vocab_size_padded), dtype=np.float32)
    d_probs_f32[np.arange(bt), targets] = vals

    got = _run_bwd(case, probs_f32, d_probs_f32).reshape(bt, case.vocab_size_padded)

    p = probs_f32[:, : case.vocab_size]
    p_t = p[np.arange(bt), targets]
    onehot = np.zeros_like(p)
    onehot[np.arange(bt), targets] = 1.0
    expected = (vals * p_t)[:, None] * (onehot - p)

    tol = DTYPE_TOLERANCES["float32"]
    np.testing.assert_allclose(
        got[:, : case.vocab_size],
        expected,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg="one-hot upstream did not collapse to val*p_t*(onehot - p)",
    )


def test_backward_padding_left_untouched():
    """d_logits' padded tail must stay byte-identical — the backward matmul
    reads the full Vp width, so stray writes there poison wte gradients."""
    case = next(c for c in CASES if c.name == "fp32_small")
    probs_f32, d_probs_f32 = _make_bwd_inputs(case)
    bt = case.batch_size * case.seq_len

    sentinel = np.float32(7.0)
    d_logits_in = np.full(bt * case.vocab_size_padded, sentinel, dtype=np.float32)

    got = _run_bwd(case, probs_f32, d_probs_f32, d_logits=d_logits_in).reshape(
        bt, case.vocab_size_padded
    )

    np.testing.assert_array_equal(
        got[:, case.vocab_size :],
        np.full((bt, case.vocab_size_padded - case.vocab_size), sentinel),
        err_msg="kernel wrote into the padded tail of d_logits",
    )


def test_backward_rows_sum_to_zero():
    """Analytic anchor: sum_i p_i*(g_i - dot) = dot - dot = 0 exactly, so
    each d_logits row must sum to ~0 — softmax gradients live in the
    zero-sum plane (shifting all logits equally never changes probs)."""
    case = next(c for c in CASES if c.name == "fp32_gpt2_shape")
    probs_f32, d_probs_f32 = _make_bwd_inputs(case)
    bt = case.batch_size * case.seq_len

    got = _run_bwd(case, probs_f32, d_probs_f32).reshape(bt, case.vocab_size_padded)

    sums = got[:, : case.vocab_size].astype(np.float64).sum(axis=1)
    np.testing.assert_allclose(
        sums,
        np.zeros(bt),
        atol=1e-5,
        rtol=0,
        err_msg="d_logits rows do not sum to zero",
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
