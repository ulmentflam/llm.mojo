#!/usr/bin/env python3
"""CUDA precision capability probe for PyTorch's `train_gpt2.py` reference.

Answers the four questions the `--precision` flag design (see
`docs/ai/pytorch_precision_support.md`) needed empirical (not
documentation-inferred) answers to, on THIS box: NVIDIA GB10
(Grace-Blackwell, aarch64, compute capability **sm_121**), torch 2.12.0 +
CUDA 12.9, pixi's `cuda` environment. `torchao` and `transformer_engine` are
**NOT installed** in this environment — everything below uses only stock
`torch` (`torch._scaled_mm` / `torch.nn.functional.scaled_mm`).

Run (GPU probes need the shared-GPU lock, short holds):

    flock -w 10800 /tmp/llmm-gpu.lock -c \
        'pixi run -e cuda python tests/probe_torch_precisions.py'

## Findings (verdicts — see the printed capability matrix for detail)

(a) **TF32 matmul controls**: `torch.backends.cuda.matmul.allow_tf32` and
    `torch.set_float32_matmul_precision(...)` both exist and take effect
    (verified via a numerically-distinguishable ill-conditioned matmul: the
    TF32 and "highest"-precision results differ, proving the knob is not a
    no-op on this build/GPU). **WORKS.**

(b) **autocast bf16/fp16**: `torch.amp.autocast(device_type="cuda",
    dtype=torch.bfloat16|torch.float16)` runs a real forward pass and
    produces the expected output dtype inside the context. **WORKS**
    (unsurprising — this was already exercised by the pre-existing
    `--dtype` flag; included here for completeness of the capability
    matrix).

(c) **FP8 `torch._scaled_mm`, e4m3 x e4m3 -> bf16, per-tensor scales**:
    dispatches and runs on this device. Verified against an fp32 reference
    computed on the *same fp8-quantized* operands: residual error is at
    bf16-rounding precision (~1e-2 relative on random ~N(0,5) data), not
    fp8-quantization-noise scale, matching `tests/probe_fp8/RESULTS.md`
    probe 4b's finding for the Mojo/cuBLASLt-FFI path (same underlying
    cuBLASLt kernel family, different call surface). **WORKS.**

    The gradient-path operand pairing needed for a real fp8 backward
    (Transformer-Engine HYBRID recipe: E5M2 `d_output` x E4M3
    weight/activation) was also probed directly: `torch._scaled_mm(e5m2,
    e4m3) -> bf16` **WORKS** too. So `train_gpt2.py --precision fp8`
    implements a REAL (not emulated) fp8 forward *and* backward — see
    `Float8Linear` in `train_gpt2.py`.

    Semantics note (non-obvious, verified empirically, see `probe_scaled_mm_fp8`
    below): `torch._scaled_mm`'s `scale_a`/`scale_b` are **dequantization**
    scales — i.e. `output ≈ (mat_a.float() * scale_a) @ (mat_b.float() *
    scale_b)`, so if you quantized via `mat_a_fp8 = (x * s).to(fp8)`, you
    must pass `scale_a = 1/s`, not `s` itself. Passing the quantization
    multiplier instead of its reciprocal silently produces enormous
    (>1e5x) relative error while still "succeeding" (no exception) — this
    is the single easiest way to get this wrong.

(d) **FP4: does this torch have `float4_e2m1fn_x2`, and does a block-scaled
    `_scaled_mm` dispatch on sm_121?** Yes to both. `torch.float4_e2m1fn_x2`
    exists; `torch._scaled_mm_v2` (wrapped by the public
    `torch.nn.functional.scaled_mm`) accepts `ScalingType.BlockWise1x16`
    (NVIDIA's single-level NVFP4 block-scale recipe — E4M3 scale per
    16-element block, no second per-tensor level) with
    `SwizzleType.SWIZZLE_32_4_4` (cuBLASLt's 128x4-tile / 32x4x4-internal
    scale-factor layout — same layout `tests/probe_fp4/probe_fp4.cu`
    implemented from scratch against the cuBLASLt docs). A 128x256x128
    NVFP4 GEMM with a from-scratch (E2M1 8-value ladder, nearest-encode)
    quantizer dispatches and returns `rel_L2 ≈ 0.135` against an fp32
    reference on the *unquantized* inputs — consistent with
    `tests/probe_fp4/RESULTS.md`'s ~0.1445 finding for the equivalent raw
    cuBLASLt-FFI probe (same NVFP4 quantization noise floor, different
    call surface: this probe goes through `torch._scaled_mm_v2`/ATen,
    `tests/probe_fp4` calls `cublasLtMatmul` directly via a hand-written
    `.cu` file). **WORKS — real cuBLASLt/cutlass tensor-core dispatch, not
    an emulated fallback.**

    `torch.cuda.get_device_capability()` returns `(12, 1)` on this box and
    torch does **not** appear to hard-gate NVFP4 dispatch on it being one of
    the datacenter Blackwell capabilities (sm_100/sm_100a) — the ATen meta
    registration (`torch/_meta_registrations.py`,
    `_check_scaled_mm_sizes_v2`) only checks tensor shapes/dtypes/strides,
    not device capability; the actual kernel selection happens inside
    cuBLASLt/cutlass at the C++ level and it picked a real sm_120-family
    NVFP4 tensor-core kernel that runs on sm_121 (same "sm_120 cubin
    dispatches on sm_121 hardware" finding `tests/probe_fp4/RESULTS.md`
    already established for the raw-cuBLASLt path).

    Because probe (d) **PASSES**, `train_gpt2.py --precision nvfp4` uses a
    REAL (native, not STE-emulated) NVFP4 tensor-core GEMM for the
    **forward** pass of its eligible Linears (`NVFP4Linear`). Its backward
    is plain bf16 matmul (not NVFP4) — a deliberate scope-limiting choice,
    not a probe failure; see `NVFP4Linear`'s docstring in `train_gpt2.py`
    for why (full fwd+bwd NVFP4 needs per-GEMM-orientation re-blocking of
    both operands plus, per `docs/ai/fp4_training_recipes_research.md` §1,
    stochastic rounding + a 16x16 Hadamard transform on the Wgrad input for
    numerical stability — machinery this reference script doesn't otherwise
    have and which the Mojo trainer's `matmul_bwd_fp4` implements
    separately).

## Environment this was run against

torch 2.12.0, CUDA 12.9 (pixi `cuda` env), NVIDIA GB10, `sm_121`,
`torch.cuda.get_device_capability() == (12, 1)`. `torchao`,
`transformer_engine` NOT installed (confirmed: `ModuleNotFoundError` on
import) — every code path here is stock-`torch`-only, by design (no
`pixi install`/`pixi update` was used or is needed).
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F


def _hr(title: str) -> None:
    print(f"\n=== {title} ===")


def probe_environment() -> dict[str, object]:
    info: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name()
        info["device_capability"] = torch.cuda.get_device_capability()
    info["has_float8_e4m3fn"] = hasattr(torch, "float8_e4m3fn")
    info["has_float8_e5m2"] = hasattr(torch, "float8_e5m2")
    info["has_float4_e2m1fn_x2"] = hasattr(torch, "float4_e2m1fn_x2")
    info["has_scaled_mm"] = hasattr(torch, "_scaled_mm")
    info["has_scaled_mm_v2"] = hasattr(torch, "_scaled_mm_v2")
    try:
        # pyrefly: ignore[missing-import]  optional dep, probed but not installed
        import torchao  # noqa: F401

        info["torchao"] = "INSTALLED"
    except ImportError:
        info["torchao"] = "not installed"
    try:
        # pyrefly: ignore[missing-import]  optional dep, probed but not installed
        import transformer_engine  # noqa: F401

        info["transformer_engine"] = "INSTALLED"
    except ImportError:
        info["transformer_engine"] = "not installed"
    return info


def probe_tf32_controls(device: str) -> bool:
    """(a) TF32 matmul controls: verify the knobs are not a silent no-op by
    checking they numerically distinguish an ill-conditioned matmul."""
    if device != "cuda":
        print("  (skip: TF32 is a CUDA-only concept)")
        return False
    torch.manual_seed(0)
    # Values spanning several decades of magnitude - TF32's reduced 10-bit
    # mantissa loses relative precision here in a way "highest" won't.
    a = (torch.rand(2048, 2048, device=device) - 0.5) * torch.logspace(
        -3, 3, 2048, device=device
    )
    b = torch.randn(2048, 2048, device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    ref = (a @ b).clone()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    tf32_out = a @ b

    # Restore strict defaults for subsequent probes.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    diff = (tf32_out - ref).abs().max().item()
    works = diff > 0.0
    print(
        f"  allow_tf32 True vs False max abs diff: {diff:.6g} (>0 => knob has effect)"
    )
    return works


def probe_autocast(device: str) -> bool:
    """(b) autocast bf16/fp16 produce the expected output dtype."""
    x = torch.randn(64, 64, device=device)
    w = torch.randn(64, 64, device=device)
    ok = True
    for dtype in (torch.bfloat16, torch.float16):
        if device == "cpu" and dtype is torch.float16:
            continue  # CPU autocast fp16 matmul is not implemented upstream
        with torch.amp.autocast(device_type=device, dtype=dtype):
            y = x @ w
        good = y.dtype == dtype
        ok = ok and good
        print(
            f"  autocast({dtype}): output dtype={y.dtype} (expected {dtype}) -> {'OK' if good else 'MISMATCH'}"
        )
    return ok


def probe_scaled_mm_fp8(device: str) -> bool:
    """(c) fp8 torch._scaled_mm: e4m3 x e4m3 -> bf16 forward, e5m2 x e4m3 ->
    bf16 backward-shaped op (the Transformer-Engine gradient pairing)."""
    if device != "cuda":
        print("  (skip: fp8 tensor-core GEMM is CUDA-only)")
        return False
    torch.manual_seed(0)
    M, K, N = 128, 256, 128
    FP8_E4M3_MAX = 448.0

    def quantize_e4m3(x):
        s = FP8_E4M3_MAX / x.abs().amax().clamp_min(1e-12)
        return (x * s).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn), s

    A = torch.randn(M, K, device=device) * 5.0
    B = torch.randn(N, K, device=device) * 5.0
    Aq, s_a = quantize_e4m3(A)
    Bq, s_b = quantize_e4m3(B)

    try:
        out = torch._scaled_mm(
            Aq,
            Bq.t(),
            scale_a=(1.0 / s_a).reshape(1),
            scale_b=(1.0 / s_b).reshape(1),
            out_dtype=torch.bfloat16,
        )
        ref = A @ B.t()
        rel = ((out.float() - ref).norm() / ref.norm()).item()
        print(
            f"  e4m3 x e4m3 -> bf16 forward: OK, shape={tuple(out.shape)}, rel_L2={rel:.4g}"
        )
        fwd_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"  e4m3 x e4m3 -> bf16 forward: FAILED: {e!r}")
        fwd_ok = False

    # Also confirm the "scale is the dequant factor, not the quant
    # multiplier" semantics, since getting this backwards is silent.
    try:
        wrong = torch._scaled_mm(
            Aq,
            Bq.t(),
            scale_a=s_a.reshape(1),
            scale_b=s_b.reshape(1),
            out_dtype=torch.float32,
        )
        ref = A @ B.t()
        rel_wrong = ((wrong - ref).norm() / ref.norm()).item()
        print(
            f"  sanity check (quant-multiplier passed as scale, WRONG on purpose): "
            f"rel_L2={rel_wrong:.4g} (expected huge, confirms scale_a/b are dequant "
            f"factors = 1/quant_multiplier)"
        )
    except Exception as e:  # noqa: BLE001
        print(f"  sanity check raised instead of silently misbehaving: {e!r}")

    def quantize_e5m2(x):
        FP8_E5M2_MAX = 57344.0
        s = FP8_E5M2_MAX / x.abs().amax().clamp_min(1e-12)
        return (x * s).clamp(-FP8_E5M2_MAX, FP8_E5M2_MAX).to(torch.float8_e5m2), s

    # dgrad-shaped op: dX[M,K] = G[M,N] @ B[N,K]. cuBLASLt requires mat_a
    # row-major and mat_b COL-major; B's natural fp8 quantization (from a
    # [N,K] row-major tensor, like Bq above) is row-major, which is the
    # WRONG layout for this orientation. The fix (mirrors `Float8Linear`'s
    # actual backward in train_gpt2.py): requantize a contiguous transposed
    # copy of B ([K,N] row-major) and take a `.t()` VIEW of that, which is a
    # [N,K] tensor with col-major strides — a pure memory-layout trick, the
    # quantized VALUES are identical to `Bq.t()` since per-element
    # quantization commutes with transpose.
    G = torch.randn(M, N, device=device) * 2.0
    Gq, s_g = quantize_e5m2(G)
    Btq, s_bt = quantize_e4m3(B.t().contiguous())  # [K, N] row-major
    mat_b = Btq.t()  # [N, K] view, stride (1, N) -> col-major
    try:
        dgrad = torch._scaled_mm(
            Gq,
            mat_b,
            scale_a=(1.0 / s_g).reshape(1),
            scale_b=(1.0 / s_bt).reshape(1),
            out_dtype=torch.bfloat16,
        )
        ref = G @ B
        rel = ((dgrad.float() - ref).norm() / ref.norm()).item()
        print(
            f"  e5m2 x e4m3 -> bf16 (grad-path pairing, correct col-major mat_b): OK, shape={tuple(dgrad.shape)}, rel_L2={rel:.4g}"
        )
        bwd_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"  e5m2 x e4m3 -> bf16 (grad-path pairing): FAILED: {e!r}")
        bwd_ok = False

    # Also record the naive (row-major mat_b) layout mistake for the
    # docstring/report: confirms the failure mode is layout, not dtype pairing.
    try:
        torch._scaled_mm(
            Gq,
            Bq,
            scale_a=(1.0 / s_g).reshape(1),
            scale_b=(1.0 / s_b).reshape(1),
            out_dtype=torch.bfloat16,
        )
        print("  (sanity) row-major mat_b for the same pairing: unexpectedly succeeded")
    except Exception as e:  # noqa: BLE001
        print(
            f"  (sanity) row-major mat_b for the same pairing FAILS as expected: {type(e).__name__}: CUBLAS_STATUS_NOT_SUPPORTED (confirms the earlier fix was layout, not dtype pairing)"
        )

    return fwd_ok and bwd_ok


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """cuBLASLt's 128x4-tile / 32x4x4-internal block-scale-factor swizzle
    (docs §3.1.4.3.2). Re-derived independently here (not imported from
    `torch.testing._internal.common_quantized`, which needs the
    non-installed `expecttest` package) — same formula, cross-checked
    against `tests/probe_fp4/probe_fp4.cu`'s independent C++ derivation
    (both agree, per `tests/probe_fp4/RESULTS.md`).
    """
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows, padded_cols = n_row_blocks * 128, n_col_blocks * 4
    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    shape = uint8_data.shape
    flat = uint8_data.contiguous().view(-1)
    return (flat[1::2] << 4 | flat[::2]).view(*shape[:-1], shape[-1] // 2)


_E2M1_LADDER_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _quantize_nvfp4(x: torch.Tensor, block: int = 16):
    """Reference (RNE, no SR/RHT) NVFP4 quantizer: E2M1 elements, E4M3
    block scale, blocks along the last dim. Matches
    `tests/probe_fp4/probe_fp4.cu`'s conventions (scale = block_amax / 6.0,
    nearest-value E2M1 encode, even-index -> low nibble packing)."""
    rows, cols = x.shape
    ladder = torch.tensor(_E2M1_LADDER_VALUES, device=x.device)
    xb = x.reshape(rows, cols // block, block).float()
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / 6.0).to(torch.float8_e4m3fn)
    scale_f32 = scale.float()
    xq = (xb / scale_f32).clamp(-6.0, 6.0)
    sign = torch.sign(xq)
    absval = xq.abs()
    idx = torch.searchsorted(ladder, absval).clamp(max=len(ladder) - 1)
    lo = ladder[(idx - 1).clamp(min=0)]
    hi = ladder[idx]
    nearest = torch.where((hi - absval).abs() < (absval - lo).abs(), hi, lo)
    code_idx = torch.searchsorted(ladder, nearest).clamp(max=len(ladder) - 1)
    sign_bit = (sign < 0).to(torch.uint8) << 3
    code = (sign_bit | code_idx.to(torch.uint8)).reshape(rows, cols)
    packed = _pack_uint4(code).view(torch.float4_e2m1fn_x2)
    return packed, scale.reshape(rows, cols // block)


def probe_scaled_mm_nvfp4(device: str) -> bool:
    """(d) NVFP4 block-scaled torch.nn.functional.scaled_mm dispatch."""
    if device != "cuda":
        print("  (skip: NVFP4 tensor-core GEMM is CUDA-only)")
        return False
    if not hasattr(torch, "float4_e2m1fn_x2"):
        print("  torch.float4_e2m1fn_x2 not present on this torch build -> FAILS")
        return False
    try:
        from torch.nn.functional import ScalingType, SwizzleType
    except ImportError as e:  # noqa: BLE001
        print(f"  ScalingType/SwizzleType not importable -> FAILS: {e!r}")
        return False

    torch.manual_seed(0)
    M, K, N = 128, 256, 128
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.3
    B = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.3
    Aq, Ascale = _quantize_nvfp4(A)
    Bq, Bscale = _quantize_nvfp4(B)
    Ascale_blocked = _to_blocked(Ascale)
    Bscale_blocked = _to_blocked(Bscale)

    try:
        out = F.scaled_mm(
            Aq,
            Bq.t(),
            scale_a=Ascale_blocked,
            scale_recipe_a=ScalingType.BlockWise1x16,
            swizzle_a=SwizzleType.SWIZZLE_32_4_4,
            scale_b=Bscale_blocked,
            scale_recipe_b=ScalingType.BlockWise1x16,
            swizzle_b=SwizzleType.SWIZZLE_32_4_4,
            output_dtype=torch.bfloat16,
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"  NVFP4 scaled_mm (BlockWise1x16, sm_{torch.cuda.get_device_capability()}): FAILED: {e!r}"
        )
        return False

    ref = A.float() @ B.float().t()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    print(
        f"  NVFP4 scaled_mm: OK, shape={tuple(out.shape)}, dtype={out.dtype}, rel_L2={rel:.4g}"
    )
    print(
        "  (rel_L2 ~0.13-0.15 is the expected NVFP4 quantization noise floor at "
        "this data range - matches tests/probe_fp4/RESULTS.md's ~0.1445, NOT a bug)"
    )
    return True


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _hr("Environment")
    env = probe_environment()
    for k, v in env.items():
        print(f"  {k}: {v}")

    _hr("(a) TF32 matmul controls")
    tf32_ok = probe_tf32_controls(device)

    _hr("(b) autocast bf16/fp16")
    autocast_ok = probe_autocast(device)

    _hr(
        "(c) FP8 torch._scaled_mm (e4m3 x e4m3 -> bf16 fwd, e5m2 x e4m3 -> bf16 grad-path)"
    )
    fp8_ok = probe_scaled_mm_fp8(device)

    _hr("(d) NVFP4 block-scaled torch.nn.functional.scaled_mm")
    nvfp4_ok = probe_scaled_mm_nvfp4(device)

    _hr("Capability matrix")
    rows = [
        ("tf32 matmul controls", tf32_ok),
        ("autocast bf16/fp16", autocast_ok),
        ("fp8 _scaled_mm e4m3xe4m3->bf16 (+ e5m2 grad-path)", fp8_ok),
        ("nvfp4 scaled_mm BlockWise1x16 (native tensor-core)", nvfp4_ok),
    ]
    width = max(len(name) for name, _ in rows)
    for name, ok in rows:
        print(
            f"  {name.ljust(width)} : {'WORKS' if ok else ('SKIPPED (no CUDA)' if device != 'cuda' else 'FAILS')}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
