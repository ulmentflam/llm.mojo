"""PyTorch reference for the untied LM head (GPT2_UNTIED=1).

Slice S1 of docs/ai/untied_lm_head_design_2026-10-30.md. These tests guard the
three things the Mojo side (S0/S2) will be compared against:

* RNG alignment: the untied model's other 16 tensors are bitwise the tied
  model's init, and the head is exactly zero.
* The gradient split: tied dwte == untied dwte + dlm_head; rows absent from the
  batch satisfy dwte_untied == 0 and dlm_head == dwte_tied exactly.
* The dump layout: model-file versions 13/15, the 17th tensor at the end, the
  state header[4] flag; the tied dump layout is unchanged.

The model is tiny (random init) except the pretrained test, which needs the
cached Hugging Face gpt2 weights and is skipped when they are unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import train_gpt2 as tg


def _tiny_config() -> tg.GPTConfig:
    return tg.GPTConfig(block_size=16, vocab_size=50257, n_layer=2, n_head=2, n_embd=16)


def _build(monkeypatch: pytest.MonkeyPatch, untied: bool, config=None) -> tg.GPT:
    if untied:
        monkeypatch.setenv("GPT2_UNTIED", "1")
    else:
        monkeypatch.delenv("GPT2_UNTIED", raising=False)
    return tg.GPT(config or _tiny_config())


def _batch(vocab: int, B: int = 2, T: int = 8):
    g = torch.Generator().manual_seed(7)
    x = torch.randint(0, vocab, (B, T), generator=g)
    y = torch.randint(0, vocab, (B, T), generator=g)
    return x, y


def test_default_is_tied(monkeypatch):
    m = _build(monkeypatch, untied=False)
    assert not tg.is_untied(m)
    assert m.lm_head.weight is m.transformer.wte.weight
    assert not any(n == "lm_head.weight" for n, _ in m.named_parameters())


def test_untied_scratch_init_aligned_with_tied(monkeypatch):
    tied = _build(monkeypatch, untied=False)
    untied = _build(monkeypatch, untied=True)
    assert tg.is_untied(untied)
    tp = dict(tied.named_parameters())
    up = dict(untied.named_parameters())
    assert set(up) == set(tp) | {"lm_head.weight"}
    for name, t in tp.items():
        assert torch.equal(t, up[name]), f"{name} differs: init RNG stream moved"
    head = up["lm_head.weight"]
    assert head.shape == (50257, 16)
    assert torch.count_nonzero(head) == 0
    # The head must be a distinct tensor (planted check: mutate it, wte stays).
    with torch.no_grad():
        head.add_(1.0)
    assert torch.count_nonzero(up["transformer.wte.weight"] == 1.0) == 0


def _grads_by_name(model: tg.GPT, x, y):
    model.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    loss.backward()
    grads = {}
    for n, p in model.named_parameters():
        assert p.grad is not None, n
        grads[n] = p.grad.clone()
    return logits.detach(), loss.detach(), grads


def test_untied_copy_of_wte_matches_tied_forward_and_grad_split(monkeypatch):
    """Head := copy of wte (the design's nonzero-head fixture recipe)."""
    tied = _build(monkeypatch, untied=False)
    untied = _build(monkeypatch, untied=True)
    with torch.no_grad():
        for (n, p), (_, q) in zip(tied.named_parameters(), untied.named_parameters()):
            if n != "lm_head.weight":
                q.copy_(p)
        untied.lm_head.weight.copy_(tied.transformer.wte.weight)
    # Absent rows: pick a batch that leaves most vocab rows unused.
    x, y = _batch(50257)
    lt, losst, gt = _grads_by_name(tied, x, y)
    lu, lossu, gu = _grads_by_name(untied, x, y)
    assert torch.equal(lt, lu) and torch.equal(losst, lossu)
    dwte_t = gt["transformer.wte.weight"]
    dwte_u = gu["transformer.wte.weight"]
    dhead = gu["lm_head.weight"]
    present = torch.zeros(50257, dtype=torch.bool)
    present[x.flatten()] = True
    assert not present.all() and present.any()
    assert torch.count_nonzero(dwte_u[~present]) == 0
    assert torch.equal(dhead[~present], dwte_t[~present])
    scale = dwte_t.abs().max()
    assert torch.allclose(dwte_u + dhead, dwte_t, rtol=1e-5, atol=1e-6 * float(scale))
    # The encoder part must be nonzero for present rows (not a blind test).
    assert torch.count_nonzero(dwte_u[present]) > 0
    # Body gradients agree (the head is a copy, so the body sees the same dlogits path).
    for n, g in gt.items():
        if n != "transformer.wte.weight":
            assert torch.allclose(
                g, gu[n], rtol=1e-5, atol=1e-6 * float(g.abs().max())
            ), n


def test_zero_head_blocks_body_gradient(monkeypatch):
    """Finding 4: with W_head = 0 only the head gets a gradient at step 0."""
    m = _build(monkeypatch, untied=True)
    x, y = _batch(50257)
    _, _, g = _grads_by_name(m, x, y)
    assert torch.count_nonzero(g["lm_head.weight"]) > 0
    for n, grad in g.items():
        if n not in (
            "lm_head.weight",
            "transformer.ln_f.gamma",
            "transformer.ln_f.beta",
        ):
            assert torch.count_nonzero(grad) == 0, n


def _read_header(path) -> np.ndarray:
    return np.fromfile(path, dtype=np.int32, count=256)


def test_dump_layout(monkeypatch, tmp_path):
    cfg = _tiny_config()
    tied = _build(monkeypatch, untied=False, config=cfg)
    untied = _build(monkeypatch, untied=True, config=cfg)
    with torch.no_grad():
        untied.lm_head.weight.copy_(torch.randn_like(untied.lm_head.weight))
    Vp, C = 50304, 16
    for dtype, tied_v, nbytes in (("float32", 3, 4), ("bfloat16", 5, 2)):
        tp = tmp_path / f"tied_{dtype}.bin"
        up = tmp_path / f"untied_{dtype}.bin"
        tg.write_model(tied, str(tp), dtype=dtype)
        tg.write_model(untied, str(up), dtype=dtype)
        th, uh = _read_header(tp), _read_header(up)
        assert th[1] == tied_v and uh[1] == tied_v + 10
        assert np.array_equal(np.delete(th, 1), np.delete(uh, 1))
        assert uh[7] == Vp
        tb, ub = tp.read_bytes()[1024:], up.read_bytes()[1024:]
        head_bytes = Vp * C * nbytes
        assert len(ub) == len(tb) + head_bytes
        # The 16-tensor prefix of the untied blob is the tied layout (same init).
        assert ub[: len(tb)] == tb
        # The tail is the padded head: V rows of data, then zero pad rows.
        if dtype == "float32":
            tail = np.frombuffer(ub[len(tb) :], dtype=np.float32).reshape(Vp, C)
            assert np.array_equal(tail[:50257], untied.lm_head.weight.detach().numpy())
            assert not tail[50257:].any()


def test_state_header_flag_and_head_grad(monkeypatch, tmp_path):
    cfg = _tiny_config()
    x, y = _batch(50257)
    sizes = {}
    for untied in (False, True):
        m = _build(monkeypatch, untied=untied, config=cfg)
        if untied:
            with torch.no_grad():
                m.lm_head.weight.copy_(torch.randn_like(m.lm_head.weight))
        logits, loss = m(x, y)
        loss.backward()
        out = tmp_path / f"state_{untied}.bin"
        tg.write_state(m, x, y, logits, loss, {}, str(out))
        h = _read_header(out)
        assert h[1] == 3 and h[4] == (1 if untied else 0)
        sizes[untied] = out.stat().st_size
    assert sizes[True] - sizes[False] == 50304 * 16 * 4


def test_pretrained_untied_head_is_wte_copy_and_forward_bitwise(monkeypatch):
    """The fixture recipe: from_pretrained untied gives head == wte (nonzero)."""
    pytest.importorskip("transformers")
    monkeypatch.delenv("GPT2_UNTIED", raising=False)
    try:
        tied = tg.GPT.from_pretrained("gpt2")
        monkeypatch.setenv("GPT2_UNTIED", "1")
        untied = tg.GPT.from_pretrained("gpt2")
    except Exception as e:  # offline / weights not cached
        pytest.skip(f"gpt2 weights unavailable: {e}")
    assert tg.is_untied(untied) and not tg.is_untied(tied)
    head = untied.lm_head.weight
    assert torch.count_nonzero(head) > 0
    assert torch.equal(head, untied.transformer.wte.weight)
    assert torch.equal(head, tied.transformer.wte.weight)
    x, y = _batch(50257, B=2, T=16)
    lt, losst, gt = _grads_by_name(tied, x, y)
    lu, lossu, gu = _grads_by_name(untied, x, y)
    assert torch.equal(lt, lu) and torch.equal(losst, lossu)
    present = torch.zeros(50257, dtype=torch.bool)
    present[x.flatten()] = True
    dwte_t = gt["transformer.wte.weight"]
    assert torch.count_nonzero(gu["transformer.wte.weight"][~present]) == 0
    assert torch.equal(gu["lm_head.weight"][~present], dwte_t[~present])
    assert torch.allclose(
        gu["transformer.wte.weight"] + gu["lm_head.weight"],
        dwte_t,
        rtol=1e-5,
        atol=1e-6 * float(dwte_t.abs().max()),
    )
