"""Equivalence: Mojo matmul_{fwd,bwd} vs PyTorch.

The forward must match x @ W^T + bias (llm.c weight layout (OC, C)); the
backward must match torch autograd's grads of that expression contracted
with d_output. Properties beyond plain closeness:

  *  accumulate=True (llm.c beta=1 contract): calling backward twice must
     exactly double d_weight/d_bias while d_input is overwritten, and
     accumulate=False must overwrite garbage-filled grad buffers.
  *  use_gelu forward: pre_gelu must equal the bias-added matmul. The GELU
     application itself is still a kernel TODO, so `out` is deliberately
     not asserted on that path yet.
  *  use_gelu backward: d_weight/d_bias consume d_output directly and are
     gelu-independent; they must match the plain backward even before
     gelu_grad lands. d_input on that path is pending the same TODO.

Inputs are round-tripped through the kernel dtype so the kernel and the
reference consume bit-identical values (mismatches are then kernel
arithmetic, not input quantization).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tests._dtypes import TORCH_DTYPES, from_storage, to_storage
from tests.kernels import matmul


@dataclass(frozen=True)
class Case:
    """Fixed seed + shapes + dtype uniquely determine all operands."""

    name: str
    batch_size: int  # B
    seq_len: int  # T
    channels: int  # C
    output_channels: int  # OC
    dtype: str  # "float32" | "bfloat16" | "float16"
    seed: int


CASES: tuple[Case, ...] = (
    # Odd sizes: rows and OC exercise the kernels' tail paths (OC % 4 != 0
    # hits the GPU ragged tile and the CPU vectorize tail).
    Case(
        "fp32_odd_sizes",
        batch_size=2,
        seq_len=7,
        channels=24,
        output_channels=42,
        dtype="float32",
        seed=0,
    ),
    # Real GPT-2 qkv shape (scaled-down batch): C=768 -> OC=2304 routes the
    # f32 CPU path through Apple Accelerate rather than the tiled fallback.
    Case(
        "fp32_gpt2_qkv",
        batch_size=1,
        seq_len=16,
        channels=768,
        output_channels=2304,
        dtype="float32",
        seed=1,
    ),
    Case(
        "bf16_small",
        batch_size=2,
        seq_len=8,
        channels=32,
        output_channels=48,
        dtype="bfloat16",
        seed=2,
    ),
)

# fp32 reduction-order noise (torch and Accelerate sum in different block
# orders) grows with the reduction length and the operand scale. Operands
# below use GPT-2-realistic scales (weights/grads ~0.02 std, llm.c's init)
# precisely so that noise stays under a tight fp32 atol even at K=2304;
# unit-scale operands would push outputs to ~|200|, where one ulp alone is
# 1.5e-5 and a tight atol stops meaning anything. bf16 reflects one storage
# rounding after f32 accumulation.
TOLERANCES = {
    "float32": {"atol": 1e-5, "rtol": 1e-4},
    "bfloat16": {"atol": 1e-2, "rtol": 3e-2},
    "float16": {"atol": 5e-3, "rtol": 1e-2},
}

# GPT-2 initialization scale (llm.c: 0.02 std for weights); activations
# stay near unit scale, upstream grads near loss scale.
WEIGHT_SCALE = 0.02
GRAD_SCALE = 0.02


def _ids(case: Case) -> str:
    return case.name


def _make_operands(case: Case):
    """Returns (x, w, b, d_out) as f32 numpy arrays already rounded to the
    kernel dtype, llm.c layouts: x (B*T, C), w (OC, C), b (OC,), d_out
    (B*T, OC)."""
    g = torch.Generator().manual_seed(case.seed)
    rows = case.batch_size * case.seq_len
    td = TORCH_DTYPES[case.dtype]

    def rounded(*shape, scale: float = 1.0):
        t = torch.randn(*shape, generator=g) * scale
        return t.to(td).to(torch.float32).numpy()

    x = rounded(rows, case.channels)
    w = rounded(case.output_channels, case.channels, scale=WEIGHT_SCALE)
    b = rounded(case.output_channels, scale=WEIGHT_SCALE)
    d_out = rounded(rows, case.output_channels, scale=GRAD_SCALE)
    return x, w, b, d_out


def _torch_grads(x, w, b, d_out):
    """Autograd reference grads of y = x @ w^T + b contracted with d_out."""
    xt = torch.from_numpy(x).requires_grad_(True)
    wt = torch.from_numpy(w).requires_grad_(True)
    bt = torch.from_numpy(b).requires_grad_(True)
    y = xt @ wt.T + bt
    y.backward(gradient=torch.from_numpy(d_out))
    assert xt.grad is not None and wt.grad is not None and bt.grad is not None
    return xt.grad.numpy(), wt.grad.numpy(), bt.grad.numpy()


def _run_forward(case: Case, x, w, b, *, use_gelu: bool = False):
    out, pre_gelu = matmul.forward(
        x=to_storage(x, case.dtype),
        weight=to_storage(w, case.dtype),
        bias=to_storage(b, case.dtype),
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        channels=case.channels,
        output_channels=case.output_channels,
        dtype_name=case.dtype,
        use_gelu=use_gelu,
    )
    return from_storage(out, case.dtype), from_storage(pre_gelu, case.dtype)


def _run_backward(case: Case, d_out, x, w, **kwargs):
    d_input, d_weight, d_bias = matmul.backward(
        d_output=to_storage(d_out, case.dtype),
        x=to_storage(x, case.dtype),
        weight=to_storage(w, case.dtype),
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        channels=case.channels,
        output_channels=case.output_channels,
        dtype_name=case.dtype,
        **kwargs,
    )
    return (
        from_storage(d_input, case.dtype),
        from_storage(d_weight, case.dtype),
        from_storage(d_bias, case.dtype),
    )


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_forward_matches_torch(case: Case) -> None:
    x, w, b, _ = _make_operands(case)
    out, _ = _run_forward(case, x, w, b)
    expected = torch.from_numpy(x) @ torch.from_numpy(w).T + torch.from_numpy(b)
    np.testing.assert_allclose(out, expected.numpy(), **TOLERANCES[case.dtype])


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_forward_gelu_pre_gelu_matches_linear(case: Case) -> None:
    """pre_gelu must hold the bias-added matmul. `out` is not asserted:
    the gelu application in the kernel epilogue is still a TODO."""
    x, w, b, _ = _make_operands(case)
    _, pre_gelu = _run_forward(case, x, w, b, use_gelu=True)
    expected = torch.from_numpy(x) @ torch.from_numpy(w).T + torch.from_numpy(b)
    np.testing.assert_allclose(pre_gelu, expected.numpy(), **TOLERANCES[case.dtype])


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_backward_matches_torch(case: Case) -> None:
    x, w, b, d_out = _make_operands(case)
    ref_dx, ref_dw, ref_db = _torch_grads(x, w, b, d_out)
    d_input, d_weight, d_bias = _run_backward(case, d_out, x, w)
    tol = TOLERANCES[case.dtype]
    np.testing.assert_allclose(d_input, ref_dx, **tol)
    np.testing.assert_allclose(d_weight, ref_dw, **tol)
    np.testing.assert_allclose(d_bias, ref_db, **tol)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_backward_accumulates(case: Case) -> None:
    """Second accumulate=True pass exactly doubles d_weight/d_bias while
    d_input (overwrite semantics) stays at one gradient's worth."""
    x, w, b, d_out = _make_operands(case)
    d_input, d_weight, d_bias = _run_backward(case, d_out, x, w)
    d_input2, d_weight2, d_bias2 = _run_backward(
        case,
        d_out,
        x,
        w,
        d_weight=to_storage(d_weight, case.dtype),
        d_bias=to_storage(d_bias, case.dtype),
    )
    tol = TOLERANCES[case.dtype]
    np.testing.assert_allclose(d_input2, d_input, atol=0, rtol=0)
    np.testing.assert_allclose(d_weight2, 2.0 * d_weight, **tol)
    np.testing.assert_allclose(d_bias2, 2.0 * d_bias, **tol)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_backward_no_accumulate_overwrites(case: Case) -> None:
    """accumulate=False must ignore (overwrite) pre-existing grad values."""
    x, w, b, d_out = _make_operands(case)
    ref_dx, ref_dw, ref_db = _torch_grads(x, w, b, d_out)
    rows = case.batch_size * case.seq_len
    poison_dw = to_storage(
        np.full((case.output_channels, case.channels), 7.0, dtype=np.float32),
        case.dtype,
    )
    poison_db = to_storage(
        np.full(case.output_channels, 3.0, dtype=np.float32), case.dtype
    )
    d_input, d_weight, d_bias = _run_backward(
        case,
        d_out,
        x,
        w,
        accumulate=False,
        d_weight=poison_dw,
        d_bias=poison_db,
    )
    tol = TOLERANCES[case.dtype]
    np.testing.assert_allclose(d_input, ref_dx, **tol)
    np.testing.assert_allclose(d_weight, ref_dw, **tol)
    np.testing.assert_allclose(d_bias, ref_db, **tol)
    assert d_input.shape == (rows, case.channels)


GELU_FWD_TODO = (
    "kernel gelu TODO: the use_gelu forward epilogue stores only pre_gelu;"
    " enable once gelu(v) lands in matmul_fwd"
)
GELU_BWD_TODO = (
    "kernel gelu_grad TODO: the use_gelu d_input epilogue stores the raw"
    " matmul; enable once gelu_grad lands in matmul_d_input_bwd"
)


@pytest.mark.skip(reason=GELU_FWD_TODO)
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_forward_gelu_out_matches_torch(case: Case) -> None:
    """out must be gelu(x @ W^T + b), llm.c tanh approximation."""
    x, w, b, _ = _make_operands(case)
    out, _ = _run_forward(case, x, w, b, use_gelu=True)
    linear = torch.from_numpy(x) @ torch.from_numpy(w).T + torch.from_numpy(b)
    expected = torch.nn.functional.gelu(linear, approximate="tanh")
    np.testing.assert_allclose(out, expected.numpy(), **TOLERANCES[case.dtype])


@pytest.mark.skip(reason=GELU_BWD_TODO)
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_backward_gelu_d_input_matches_torch(case: Case) -> None:
    """llm.c composition: the op's x input is the POST-gelu activation
    (gelu(pre_gelu)) and use_gelu backward must return
    d_input = (d_output @ W) * gelu'(pre_gelu), i.e. the gradient at the
    pre-activation, replacing a separate gelu_backward pass."""
    g = torch.Generator().manual_seed(CASES.index(case) + 100)
    rows = case.batch_size * case.seq_len
    td = TORCH_DTYPES[case.dtype]
    h = torch.randn(rows, case.channels, generator=g).to(td).to(torch.float32)
    _, w, b, d_out = _make_operands(case)

    ht = h.clone().requires_grad_(True)
    a = torch.nn.functional.gelu(ht, approximate="tanh")
    y = a @ torch.from_numpy(w).T + torch.from_numpy(b)
    y.backward(gradient=torch.from_numpy(d_out))

    d_input, _, _ = _run_backward(
        case,
        d_out,
        a.detach().numpy(),
        w,
        use_gelu=True,
        pre_gelu=to_storage(h.numpy(), case.dtype),
    )
    assert ht.grad is not None
    np.testing.assert_allclose(d_input, ht.grad.numpy(), **TOLERANCES[case.dtype])


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_backward_gelu_weight_and_bias_grads(case: Case) -> None:
    """d_weight/d_bias are gelu-independent (they consume d_output
    directly), so the use_gelu path must already match torch. d_input on
    this path is pending the kernel's gelu_grad TODO and is not asserted."""
    x, w, b, d_out = _make_operands(case)
    _, ref_dw, ref_db = _torch_grads(x, w, b, d_out)
    _, pre_gelu = _run_forward(case, x, w, b, use_gelu=True)
    _, d_weight, d_bias = _run_backward(
        case,
        d_out,
        x,
        w,
        use_gelu=True,
        pre_gelu=to_storage(pre_gelu, case.dtype),
    )
    tol = TOLERANCES[case.dtype]
    np.testing.assert_allclose(d_weight, ref_dw, **tol)
    np.testing.assert_allclose(d_bias, ref_db, **tol)
