"""Equivalence: Mojo fused_classifier{,_fwd} vs torch and the unfused ops.

The fused op is the composition of softmax and crossentropy with the
Jacobian collapsed (docs/backprop.tex eq. fused-bwd): the forward emits
losses in log-softmax form without materializing probs, and the training
variant overwrites logits in place with
d_logits = (p - onehot(target)) * d_loss, zeroing the padded tail.

References: torch cross_entropy (losses) and torch autograd through it
(d_logits), both on the same dtype-rounded logits the kernel reads, plus
a composition cross-check against this repo's own softmax and
crossentropy ops. Shapes and the poisoned-padding generator are shared
with the softmax suite.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests._dtypes import DTYPE_TOLERANCES, from_storage, to_storage
from tests.kernels import crossentropy, fused_classifier, softmax
from tests.test_softmax_equivalence import CASES, Case, _make_logits


def _make_targets_d_losses(case: Case) -> tuple[np.ndarray, np.ndarray]:
    g = torch.Generator().manual_seed(case.seed + 1000)
    bt = case.batch_size * case.seq_len
    targets = torch.randint(0, case.vocab_size, (bt,), generator=g, dtype=torch.int32)
    d_losses = torch.randn(bt, generator=g, dtype=torch.float32)
    return targets.numpy(), d_losses.numpy()


def _reference_losses(
    logits_f32: np.ndarray, targets: np.ndarray, vocab_size: int
) -> np.ndarray:
    x = torch.from_numpy(logits_f32[:, :vocab_size])
    t = torch.from_numpy(targets).long()
    return torch.nn.functional.cross_entropy(x, t, reduction="none").numpy()


def _reference_d_logits(
    logits_f32: np.ndarray,
    targets: np.ndarray,
    d_losses: np.ndarray,
    vocab_size: int,
) -> np.ndarray:
    x = torch.from_numpy(logits_f32[:, :vocab_size]).clone().requires_grad_(True)
    t = torch.from_numpy(targets).long()
    losses = torch.nn.functional.cross_entropy(x, t, reduction="none")
    losses.backward(torch.from_numpy(d_losses))
    assert x.grad is not None
    return x.grad.numpy()


def _run_eval(case: Case, logits_f32: np.ndarray, targets: np.ndarray):
    return fused_classifier.forward(
        logits=to_storage(logits_f32.reshape(-1), case.dtype),
        targets=targets,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        vocab_size=case.vocab_size,
        vocab_size_padded=case.vocab_size_padded,
        dtype_name=case.dtype,
    )


def _run_training(
    case: Case,
    logits_f32: np.ndarray,
    targets: np.ndarray,
    d_losses: np.ndarray,
):
    return fused_classifier.forward_backward(
        logits=to_storage(logits_f32.reshape(-1), case.dtype),
        d_losses=d_losses,
        targets=targets,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        vocab_size=case.vocab_size,
        vocab_size_padded=case.vocab_size_padded,
        dtype_name=case.dtype,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_eval_losses_match_torch(case: Case):
    logits_f32 = _make_logits(case)
    targets, _ = _make_targets_d_losses(case)
    expected = _reference_losses(logits_f32, targets, case.vocab_size)

    got = _run_eval(case, logits_f32, targets)

    # Losses are computed and stored in fp32 regardless of logits dtype,
    # and the reference reads the same dtype-rounded logits, so fp32
    # tolerances apply even for the bf16 case.
    tol = DTYPE_TOLERANCES["float32"]
    np.testing.assert_allclose(
        got,
        expected,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: eval losses diverged from torch",
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_training_losses_match_eval(case: Case):
    """The training op runs the same loss code path as eval; the gradient
    epilogue must not perturb it. Identical inputs, identical losses."""
    logits_f32 = _make_logits(case)
    targets, d_losses = _make_targets_d_losses(case)

    eval_losses = _run_eval(case, logits_f32, targets)
    _, train_losses = _run_training(case, logits_f32, targets, d_losses)

    np.testing.assert_array_equal(
        train_losses,
        eval_losses,
        err_msg=f"{case.name}: training losses differ from eval losses",
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_training_d_logits_match_autograd(case: Case):
    logits_f32 = _make_logits(case)
    targets, d_losses = _make_targets_d_losses(case)
    expected = _reference_d_logits(logits_f32, targets, d_losses, case.vocab_size)

    got_d_logits, _ = _run_training(case, logits_f32, targets, d_losses)

    bt = case.batch_size * case.seq_len
    got_v = from_storage(got_d_logits, case.dtype).reshape(bt, case.vocab_size_padded)[
        :, : case.vocab_size
    ]

    tol = DTYPE_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        got_v,
        expected,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: d_logits diverged from torch autograd",
    )


def test_training_matches_unfused_composition():
    """The fused op must agree with this repo's own unfused pipeline:
    softmax_fwd -> crossentropy_ohe_fwd for the losses, and
    crossentropy_ohe_bwd -> softmax_bwd for d_logits. This pins fusion as
    a pure optimization, not a semantic change."""
    case = next(c for c in CASES if c.name == "fp32_small")
    logits_f32 = _make_logits(case)
    targets, d_losses = _make_targets_d_losses(case)
    bt = case.batch_size * case.seq_len

    # Unfused pipeline.
    probs = softmax.forward(
        logits=logits_f32.reshape(-1).copy(),
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        vocab_size=case.vocab_size,
        vocab_size_padded=case.vocab_size_padded,
        dtype_name="float32",
    )
    unfused_losses = crossentropy.forward(
        probs=probs,
        targets=targets,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        vocab_size_padded=case.vocab_size_padded,
        dtype_name="float32",
    )
    d_probs = crossentropy.backward(
        d_losses=d_losses,
        probs=probs,
        targets=targets,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        vocab_size_padded=case.vocab_size_padded,
        dtype_name="float32",
        d_probs=np.zeros_like(probs),
    )
    unfused_d_logits = softmax.backward(
        d_probs=d_probs,
        probs=probs,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        vocab_size=case.vocab_size,
        vocab_size_padded=case.vocab_size_padded,
        dtype_name="float32",
    )

    # Fused op on a fresh copy (the training op consumes its logits).
    fused_d_logits, fused_losses = _run_training(case, logits_f32, targets, d_losses)

    tol = DTYPE_TOLERANCES["float32"]
    np.testing.assert_allclose(
        fused_losses,
        unfused_losses,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg="fused losses diverged from the unfused composition",
    )
    np.testing.assert_allclose(
        fused_d_logits.reshape(bt, case.vocab_size_padded)[:, : case.vocab_size],
        unfused_d_logits.reshape(bt, case.vocab_size_padded)[:, : case.vocab_size],
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg="fused d_logits diverged from the unfused composition",
    )


def test_training_zeroes_padding():
    """In-place reuse means the buffer becomes d_logits, and the backward
    matmul reads its full Vp width; the kernel must zero the padded tail
    it inherited (the input padding is poisoned by _make_logits)."""
    case = next(c for c in CASES if c.name == "fp32_small")
    logits_f32 = _make_logits(case)
    targets, d_losses = _make_targets_d_losses(case)
    bt = case.batch_size * case.seq_len

    got_d_logits, _ = _run_training(case, logits_f32, targets, d_losses)
    got = got_d_logits.reshape(bt, case.vocab_size_padded)

    np.testing.assert_array_equal(
        got[:, case.vocab_size :],
        np.zeros((bt, case.vocab_size_padded - case.vocab_size)),
        err_msg="padded tail of d_logits was not zeroed",
    )


def test_eval_leaves_logits_unchanged():
    """write_d_logits=False must not store to logits. The bridge may copy
    host arrays to device buffers, so this is a best-effort guard: it
    catches stray writes whenever the CPU buffer shares memory."""
    case = next(c for c in CASES if c.name == "fp32_small")
    logits_f32 = _make_logits(case)
    targets, _ = _make_targets_d_losses(case)

    logits_storage = to_storage(logits_f32.reshape(-1), case.dtype)
    snapshot = logits_storage.copy()
    fused_classifier.forward(
        logits=logits_storage,
        targets=targets,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        vocab_size=case.vocab_size,
        vocab_size_padded=case.vocab_size_padded,
        dtype_name=case.dtype,
    )

    np.testing.assert_array_equal(
        logits_storage,
        snapshot,
        err_msg="eval op modified the logits buffer",
    )


def test_uniform_logits_loss_is_log_vocab():
    """Analytic anchor: constant logits => p_target = 1/V, so every loss
    equals log(V) (and the max subtraction must cancel the constant)."""
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
    g = np.random.default_rng(case.seed)
    targets = g.integers(0, case.vocab_size, size=bt).astype(np.int32)

    got = _run_eval(case, logits, targets)

    tol = DTYPE_TOLERANCES["float32"]
    np.testing.assert_allclose(
        got,
        np.full(bt, np.log(np.float64(case.vocab_size)), dtype=np.float32),
        atol=tol["atol"],
        rtol=tol["rtol"],
    )
