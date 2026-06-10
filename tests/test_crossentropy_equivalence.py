"""End-to-end equivalence: Mojo crossentropy_ohe_{fwd,bwd} vs PyTorch autograd.

The kernel exploits the one-hot structure of the loss: forward reads only
the target column of each (b, t) row of `probs`, and backward writes only
that column of `d_probs`. The reference is plain torch autograd through
`-log(probs.gather(targets))`, computed on the same dtype-rounded probs
the kernel reads, so any mismatch is the kernel's own arithmetic.

Note on backward semantics: this kernel *assigns* the target-column
gradient (`d_probs[target] = -d_loss/p`); llm.c's crossentropy_backward
accumulates (`+=`). The off-target-preservation test below pins the
assign-and-leave-the-rest behavior so a future switch to accumulation is
a deliberate, test-visible change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tests._dtypes import DTYPE_TOLERANCES, TORCH_DTYPES, from_storage, to_storage
from tests.kernels import crossentropy


@dataclass(frozen=True)
class Case:
    """Fixed seed + shapes + dtype uniquely determine probs/targets/d_losses."""

    name: str
    batch_size: int  # B
    seq_len: int  # T
    vocab_size: int  # V — targets are drawn from [0, V)
    vocab_size_padded: int  # Vp — row stride; padded tail is never indexed
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
    Case(
        "fp32_odd_sizes",
        batch_size=3,
        seq_len=7,
        vocab_size=13,
        vocab_size_padded=16,
        dtype="float32",
        seed=2,
    ),
    Case(
        "bf16_small",
        batch_size=4,
        seq_len=8,
        vocab_size=50,
        vocab_size_padded=64,
        dtype="bfloat16",
        seed=3,
    ),
)


def _make_inputs(case: Case) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (probs_f32 (BT, Vp), targets (BT,) int32, d_losses (BT,) f32).

    Probs are a real softmax over V logits, zero-padded to Vp (mirroring
    llm.c, where the padded vocab tail of probs is zeroed and never
    targeted). They are round-tripped through the kernel dtype so the
    reference and the kernel consume bit-identical values.
    """
    g = torch.Generator().manual_seed(case.seed)
    bt = case.batch_size * case.seq_len
    logits = torch.randn(bt, case.vocab_size, generator=g, dtype=torch.float32)
    probs = torch.softmax(logits, dim=1)
    if case.vocab_size_padded > case.vocab_size:
        pad = torch.zeros(bt, case.vocab_size_padded - case.vocab_size)
        probs = torch.cat([probs, pad], dim=1)
    probs = probs.to(TORCH_DTYPES[case.dtype]).to(torch.float32)

    targets = torch.randint(0, case.vocab_size, (bt,), generator=g, dtype=torch.int32)
    d_losses = torch.randn(bt, generator=g, dtype=torch.float32)
    return probs.numpy(), targets.numpy(), d_losses.numpy()


def _reference(
    probs_f32: np.ndarray, targets: np.ndarray, d_losses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Torch autograd through -log(p_target): returns (losses, d_probs)."""
    p = torch.from_numpy(probs_f32).clone().requires_grad_(True)
    gathered = p.gather(1, torch.from_numpy(targets).long().unsqueeze(1)).squeeze(1)
    losses = -torch.log(gathered)
    losses.backward(torch.from_numpy(d_losses))
    assert p.grad is not None
    return losses.detach().numpy(), p.grad.numpy()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_forward_matches_torch(case: Case):
    probs_f32, targets, _ = _make_inputs(case)
    expected_losses, _ = _reference(
        probs_f32, targets, np.ones_like(targets, dtype=np.float32)
    )

    got = crossentropy.forward(
        probs=to_storage(probs_f32.reshape(-1), case.dtype),
        targets=targets,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        vocab_size_padded=case.vocab_size_padded,
        dtype_name=case.dtype,
    )

    # Losses are computed and stored in fp32 regardless of probs dtype, and
    # the reference reads the same dtype-rounded probs — so fp32 tolerances
    # apply even for the bf16 case (the only difference is the log impl).
    tol = DTYPE_TOLERANCES["float32"]
    np.testing.assert_allclose(
        got,
        expected_losses,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: forward losses diverged from torch",
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_backward_matches_torch(case: Case):
    probs_f32, targets, d_losses = _make_inputs(case)
    _, expected_d_probs = _reference(probs_f32, targets, d_losses)

    got = crossentropy.backward(
        d_losses=d_losses,
        probs=to_storage(probs_f32.reshape(-1), case.dtype),
        targets=targets,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        vocab_size_padded=case.vocab_size_padded,
        dtype_name=case.dtype,
    )

    # d_probs is cast back to the kernel dtype on store, so the comparison
    # tolerance follows the case dtype. d_prob = -d_loss/p can be large when
    # p_target is small; rtol carries the comparison there.
    tol = DTYPE_TOLERANCES[case.dtype]
    np.testing.assert_allclose(
        from_storage(got, case.dtype).reshape(expected_d_probs.shape),
        expected_d_probs,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"{case.name}: backward d_probs diverged from torch",
    )


def test_backward_only_writes_target_entries():
    """The kernel assigns d_probs[target] and must leave every other entry
    byte-identical — including the padded vocab tail. Sentinel-fill the
    buffer to catch stray writes."""
    case = next(c for c in CASES if c.name == "fp32_small")
    probs_f32, targets, d_losses = _make_inputs(case)
    bt = case.batch_size * case.seq_len

    sentinel = np.float32(7.0)
    d_probs_in = np.full(bt * case.vocab_size_padded, sentinel, dtype=np.float32)

    got = crossentropy.backward(
        d_losses=d_losses,
        probs=probs_f32.reshape(-1).astype(np.float32),
        targets=targets,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        vocab_size_padded=case.vocab_size_padded,
        dtype_name="float32",
        d_probs=d_probs_in,
    ).reshape(bt, case.vocab_size_padded)

    rows = np.arange(bt)
    target_mask = np.zeros((bt, case.vocab_size_padded), dtype=bool)
    target_mask[rows, targets] = True

    np.testing.assert_array_equal(
        got[~target_mask],
        np.full((~target_mask).sum(), sentinel),
        err_msg="kernel wrote outside the target column",
    )

    expected_at_targets = -d_losses / probs_f32[rows, targets]
    tol = DTYPE_TOLERANCES["float32"]
    np.testing.assert_allclose(
        got[target_mask], expected_at_targets, atol=tol["atol"], rtol=tol["rtol"]
    )


def test_forward_uniform_probs_is_log_vocab():
    """Analytic anchor: p_target = 1/V everywhere ⇒ every loss = log(V)."""
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
    probs = np.full(
        (bt, case.vocab_size_padded), 1.0 / case.vocab_size, dtype=np.float32
    )
    g = np.random.default_rng(case.seed)
    targets = g.integers(0, case.vocab_size, size=bt).astype(np.int32)

    got = crossentropy.forward(
        probs=probs.reshape(-1),
        targets=targets,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        vocab_size_padded=case.vocab_size_padded,
        dtype_name="float32",
    )

    tol = DTYPE_TOLERANCES["float32"]
    np.testing.assert_allclose(
        got,
        np.full(bt, np.log(np.float64(case.vocab_size)), dtype=np.float32),
        atol=tol["atol"],
        rtol=tol["rtol"],
    )
