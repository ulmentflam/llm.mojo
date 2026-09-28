"""Tests for the MAX GPT-2 inference scaffolds.

Two layers:

* Plumbing that works today: the checkpoint reader, the sampler, and the
  mapping from checkpoint names onto each scaffold's module tree.
* Acceptance tests for the forward pass (`test_*_logits_match_hf`): a tiny
  random Hugging Face GPT2LMHeadModel is the reference, and both scaffolds
  must reproduce its logits. They are strict xfails while the forward raises
  NotImplementedError; once a forward runs, a pass reports XPASS(strict) as a
  failure, which is the cue to delete that marker.

The reference is Hugging Face's GPT-2 rather than train_gpt2.py so the MAX
forward is checked against an implementation this repo did not write.
train_gpt2.py's LayerNorm used the unbiased (N-1) variance until 2026-09-28,
a deviation an in-repo reference would have passed straight through.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2Config as HFConfig
from transformers import GPT2LMHeadModel

import infer_gpt2_max_eager as eager
import infer_gpt2_max_graph as graph_api
from max_gpt2_common import (
    F32_DISCARD_BITS,
    F32_UNIT_SCALE,
    REPO_ROOT,
    U32_OUTPUT_SHIFT,
    XORSHIFT_LEFT,
    XORSHIFT_RIGHT_1,
    XORSHIFT_RIGHT_2,
    XORSHIFT_STAR_MULTIPLIER,
    GPT2Config,
    llmm_package,
    load_checkpoint,
    param_shapes,
    random_f32,
    sample_softmax,
)
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, Graph, TensorType
from tests._sampler_reference import coin_sweep, make_logits, sample_softmax_c

TINY = GPT2Config(
    max_seq_len=16,
    vocab_size=100,
    padded_vocab_size=128,
    num_layers=2,
    num_heads=4,
    channels=32,
)
ATOL = 1e-4


def _reference_model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        HFConfig(
            n_positions=TINY.max_seq_len,
            vocab_size=TINY.vocab_size,
            n_layer=TINY.num_layers,
            n_head=TINY.num_heads,
            n_embd=TINY.channels,
            activation_function="gelu_new",  # tanh approximation, as GPT-2
            layer_norm_epsilon=1e-5,
            bos_token_id=0,  # the 50256 default lies outside the tiny vocab
            eos_token_id=0,
        )
    )
    # GPT-2 initialises biases and LN shifts to zero; perturb every parameter
    # so a forward that drops a bias or swaps a LN term cannot still match.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    return model.eval()


# HF stores these as Conv1D, i.e. [in, out]; the checkpoint layout is [out, in].
_CONV1D = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def _canonical_weights(model: GPT2LMHeadModel) -> dict[str, np.ndarray]:
    """HF parameter names/layouts -> the scaffolds' canonical ones."""
    weights = {}
    for name, p in model.transformer.named_parameters():
        w = p.detach().numpy().astype(np.float32)
        weights[name] = np.ascontiguousarray(w.T if name.endswith(_CONV1D) else w)
    # lm_head is tied to wte; pad the vocab rows with zeros the way
    # write_model does for the .bin files.
    wte = weights["wte.weight"]
    pad = TINY.padded_vocab_size - wte.shape[0]
    weights["wte.weight"] = np.pad(wte, ((0, pad), (0, 0)))
    return weights


def _reference_logits(model: GPT2LMHeadModel, tokens: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(tokens)).logits.numpy()


@pytest.fixture(scope="module")
def reference() -> tuple[GPT2LMHeadModel, dict[str, np.ndarray], np.ndarray]:
    model = _reference_model()
    tokens = np.random.default_rng(0).integers(
        0, TINY.vocab_size, size=(1, 11), dtype=np.int64
    )
    return model, _canonical_weights(model), tokens


# --- checkpoint reader


def test_param_shapes_cover_reference_model(reference) -> None:
    _, weights, _ = reference
    assert {k: v.shape for k, v in weights.items()} == dict(param_shapes(TINY))


@pytest.mark.parametrize("name", ["gpt2_124M.bin", "gpt2_124M_bf16.bin"])
def test_load_checkpoint_header_and_size(name: str) -> None:
    path = REPO_ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not downloaded (make data)")
    config, weights = load_checkpoint(path)
    assert config == GPT2Config()  # the 124M defaults
    assert len(weights) == 2 + 12 * 12 + 2
    assert all(w.dtype == np.float32 for w in weights.values())


def test_bf16_checkpoint_is_rounded_fp32() -> None:
    fp32_path, bf16_path = REPO_ROOT / "gpt2_124M.bin", REPO_ROOT / "gpt2_124M_bf16.bin"
    if not (fp32_path.exists() and bf16_path.exists()):
        pytest.skip("starter-pack checkpoints not downloaded (make data)")
    _, w32 = load_checkpoint(fp32_path)
    _, w16 = load_checkpoint(bf16_path)
    for name in ("wte.weight", "h.5.attn.c_attn.weight", "ln_f.bias"):
        a, b = w32[name], w16[name]
        # bf16 keeps 8 significant bits: relative error <= 2^-8.
        assert np.all(np.abs(a - b) <= np.abs(a) * 2.0**-8 + 1e-30), name


def test_load_checkpoint_rejects_bad_magic(tmp_path: Path) -> None:
    bad = tmp_path / "bad.bin"
    bad.write_bytes(np.zeros(256, dtype=np.int32).tobytes())
    with pytest.raises(ValueError, match="bad magic"):
        load_checkpoint(bad)


# --- sampler


def test_sample_softmax_matches_c_reference() -> None:
    for seed in range(4):
        logits = make_logits(257, seed)
        for coin in coin_sweep(64):
            assert sample_softmax(logits, float(coin)) == sample_softmax_c(
                logits, float(coin)
            )


def test_random_f32_matches_u64_arithmetic() -> None:
    """Python-int masking must agree with native uint64 wraparound (C/Mojo)."""
    state_py, state_np = 1337, np.uint64(1337)
    for _ in range(100):
        got, state_py = random_f32(state_py)
        with np.errstate(over="ignore"):
            state_np ^= state_np >> np.uint64(XORSHIFT_RIGHT_1)
            state_np ^= state_np << np.uint64(XORSHIFT_LEFT)
            state_np ^= state_np >> np.uint64(XORSHIFT_RIGHT_2)
            u = (state_np * np.uint64(XORSHIFT_STAR_MULTIPLIER)) >> np.uint64(
                U32_OUTPUT_SHIFT
            )
        want = np.float32(np.uint32(u) >> np.uint32(F32_DISCARD_BITS)) / np.float32(
            F32_UNIT_SCALE
        )
        assert got == float(want)
        assert 0.0 <= got < 1.0


# --- llmm kernel helpers (the llmm_pkg package; `make test` builds it)

HELPER_T = (1, 9)  # one compiled graph must serve several sequence lengths


def _causal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(q.shape[-1])
    t = q.shape[2]
    s = np.where(np.triu(np.ones((t, t), bool), 1), -np.inf, s)
    s = np.exp(s - s.max(-1, keepdims=True))
    return (s / s.sum(-1, keepdims=True)) @ v


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def _qkv(t: int) -> list[np.ndarray]:
    rng = np.random.default_rng(t)
    return [rng.standard_normal((1, 4, t, 16), dtype=np.float32) for _ in range(3)]


def _activations(t: int) -> np.ndarray:
    # Rank 3, like the MLP's [1, T, 4C]: exercises the flatten/restore.
    return np.random.default_rng(t).standard_normal((1, t, 32), dtype=np.float32)


def _graph_model(fn, input_types: list[TensorType]):
    graph = Graph(
        "llmm_helper",
        forward=fn,
        input_types=input_types,
        custom_extensions=[llmm_package()],
    )
    return InferenceSession(devices=[CPU()]).load(graph)


def _run_graph(model, *arrays: np.ndarray) -> np.ndarray:
    (out,) = model.execute(*(Buffer.from_numpy(a) for a in arrays))
    assert isinstance(out, Buffer)
    return out.to_numpy()


def test_graph_llmm_attention() -> None:
    qkv_type = TensorType(DType.float32, [1, 4, "seq_len", 16], DeviceRef.CPU())
    model = _graph_model(graph_api.llmm_attention, [qkv_type] * 3)
    for t in HELPER_T:
        q, k, v = _qkv(t)
        got = _run_graph(model, q, k, v)
        np.testing.assert_allclose(got, _causal_attention(q, k, v), atol=1e-5)


def test_graph_llmm_gelu() -> None:
    x_type = TensorType(DType.float32, [1, "seq_len", 32], DeviceRef.CPU())
    model = _graph_model(graph_api.llmm_gelu, [x_type])
    for t in HELPER_T:
        x = _activations(t)
        np.testing.assert_allclose(_run_graph(model, x), _gelu_tanh(x), atol=1e-5)


class _EagerAttention(Module[[Tensor, Tensor, Tensor], Tensor]):
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return eager.llmm_attention(q, k, v)


class _EagerGelu(Module[[Tensor], Tensor]):
    def forward(self, x: Tensor) -> Tensor:
        return eager.llmm_gelu(x)


def _to_numpy(t: Tensor) -> np.ndarray:
    return np.from_dlpack(t.to(CPU()))


@pytest.mark.filterwarnings("ignore:The eager interpreter failed")
@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_eager_llmm_attention(mode: str) -> None:
    fn = eager.llmm_attention
    if mode == "compiled":
        qkv_type = TensorType(DType.float32, [1, 4, "seq_len", 16], device=CPU())
        fn = _EagerAttention().compile(qkv_type, qkv_type, qkv_type)
    for t in HELPER_T:
        q, k, v = _qkv(t)
        got = _to_numpy(fn(*(Tensor.from_dlpack(a) for a in (q, k, v))))
        np.testing.assert_allclose(got, _causal_attention(q, k, v), atol=1e-5)


@pytest.mark.filterwarnings("ignore:The eager interpreter failed")
@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_eager_llmm_gelu(mode: str) -> None:
    fn = eager.llmm_gelu
    if mode == "compiled":
        x_type = TensorType(DType.float32, [1, "seq_len", 32], device=CPU())
        fn = _EagerGelu().compile(x_type)
    for t in HELPER_T:
        x = _activations(t)
        got = _to_numpy(fn(Tensor.from_dlpack(x)))
        np.testing.assert_allclose(got, _gelu_tanh(x), atol=1e-5)


def test_graph_llmm_gelu_rejects_algebraic_rows() -> None:
    """[2, T, C] flattens to rows = 2 * T, which MAX 26.6 cannot size a
    kernel buffer with. The helper must say so, not let MAX fail opaquely
    at compile time. If a MAX upgrade lifts the limit, relax the helper."""
    x_type = TensorType(DType.float32, [2, "seq_len", 32], DeviceRef.CPU())
    with pytest.raises(NotImplementedError, match="algebraic dim"):
        Graph("gelu_2xT", forward=graph_api.llmm_gelu, input_types=[x_type])


def test_llmm_package_is_content_addressed(tmp_path, monkeypatch) -> None:
    """A source edit must resolve to a different package path (never a stale
    package), and reverting it must find the original path again."""
    import llmm_pkg

    src = tmp_path / "llmm"
    src.mkdir()
    kernel = src / "gelu.mojo"
    kernel.write_text("v1")
    monkeypatch.setattr(llmm_pkg, "SOURCE_DIR", src)
    monkeypatch.setattr(llmm_pkg, "CACHE_ROOT", tmp_path / "cache")

    def path_now():
        llmm_pkg.package_fingerprint.cache_clear()
        return llmm_pkg.package_path()

    try:
        original = path_now()
        kernel.write_text("v2")
        edited = path_now()
        kernel.write_text("v1")
        reverted = path_now()
    finally:
        # Drop the tmp-tree fingerprint so later tests hash the real llmm/.
        llmm_pkg.package_fingerprint.cache_clear()
    assert original != edited
    assert reverted == original
    # MAX resolves kernels by the package's file name.
    assert original.name == edited.name == "llmm.mojoc"
    assert original.parent.parent == tmp_path / "cache"


# --- scaffold module trees accept the checkpoint names (strict)


def test_graph_scaffold_loads_weights(reference) -> None:
    _, weights, _ = reference
    model = graph_api.GPT2(TINY, DeviceRef.CPU())
    graph_api.load_weights(model, weights)  # strict: raises on any mismatch
    assert set(model.raw_state_dict()) == set(weights)


def test_eager_scaffold_loads_weights(reference) -> None:
    _, weights, _ = reference
    model = eager.build_model(TINY, weights, CPU())
    loaded = dict(model.parameters)
    assert set(loaded) == set(weights)
    np.testing.assert_array_equal(
        np.from_dlpack(loaded["h.1.mlp.c_fc.weight"].to(CPU())),
        weights["h.1.mlp.c_fc.weight"],
    )


# --- forward acceptance (xfail until implemented)

not_implemented = pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="forward pass is a TODO"
)


def _check_logits(got: np.ndarray, want: np.ndarray) -> None:
    assert got.shape == (1, want.shape[1], TINY.padded_vocab_size)
    np.testing.assert_allclose(got[..., : TINY.vocab_size], want, atol=ATOL)


@not_implemented
def test_graph_logits_match_hf(reference) -> None:
    model, weights, tokens = reference
    device = CPU()
    compiled = graph_api.compile_model(TINY, weights, device)
    got = graph_api.make_forward(compiled, device)(tokens)
    _check_logits(got, _reference_logits(model, tokens))


@not_implemented
@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_eager_logits_match_hf(reference, mode: str) -> None:
    model, weights, tokens = reference
    max_model = eager.build_model(TINY, weights, CPU())
    got = eager.make_forward(max_model, mode)(tokens)
    _check_logits(got, _reference_logits(model, tokens))
