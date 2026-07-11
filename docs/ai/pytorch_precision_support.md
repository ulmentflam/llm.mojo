# PyTorch reference trainer: `--precision` support

`train_gpt2.py` (the PyTorch GPT-2 reference at the repo root) gained a
unified `--precision {fp32,tf32,bf16,fp16,fp8,nvfp4}` flag covering every
precision regime the installed torch/CUDA/GPU combination supports, mirroring
the precision axis the Mojo trainer (`train_gpt2.mojo`, `LLMM_PRECISION`)
already has. This doc is the capability matrix, what's native vs. what's
scoped-down, and usage examples.

Env this was built/validated against: pixi `cuda` environment, torch 2.12.0 +
CUDA 12.9, NVIDIA GB10 (Grace-Blackwell, aarch64, `sm_121` — a **consumer**
Blackwell part, not the datacenter sm_100 family most fp8/fp4 library support
matrices target). `torchao` and `transformer_engine` are **NOT installed** —
everything below is stock `torch` only (`torch._scaled_mm` /
`torch.nn.functional.scaled_mm`), and nothing here required `pixi
install`/`pixi update`.

## Capability matrix (this box)

Probed empirically in `tests/probe_torch_precisions.py` (run it yourself:
`flock -w 10800 /tmp/llmm-gpu.lock -c 'pixi run -e cuda python
tests/probe_torch_precisions.py'`). Full narrative + gotchas are in that
file's module docstring; summary:

| Capability | Verdict | Notes |
| --- | --- | --- |
| `torch.backends.cuda.matmul.allow_tf32` / `set_float32_matmul_precision` | **WORKS** | Numerically verified to change output (not a silent no-op) |
| `torch.amp.autocast` bf16 / fp16 | **WORKS** | Unsurprising — pre-existing `--dtype` flag already exercised this |
| `torch._scaled_mm` fp8 E4M3×E4M3→bf16 (forward) | **WORKS** | Real cuBLASLt tensor-core dispatch; same kernel family as `tests/probe_fp8/RESULTS.md` probe 4b (raw cuBLASLt FFI via Mojo) |
| `torch._scaled_mm` fp8 E5M2×E4M3→bf16 (grad-path pairing) | **WORKS** | The Transformer-Engine HYBRID backward pairing — same call, different dtypes |
| `torch.nn.functional.scaled_mm` NVFP4 `BlockWise1x16` (block-scaled) | **WORKS** | Real cuBLASLt/cutlass tensor-core dispatch — matches `tests/probe_fp4/RESULTS.md`'s finding (~same quant-noise floor, rel_L2 ≈ 0.13–0.15) that the sm_120 NVFP4 cubin dispatches on sm_121 hardware |

**Both fp8 and NVFP4 probes PASSED on this box.** Neither `--precision fp8`
nor `--precision nvfp4` needed the emulated (quantize-dequantize STE)
fallback the original task brief anticipated for the case where dispatch
failed — that fallback was written into the design as a contingency, not
implemented, because it wasn't needed. Both precisions get a **real**
tensor-core GEMM for at least their forward pass (see below for fp8's
backward, which is also real).

One non-obvious semantics finding (see the probe's docstring for detail):
`torch._scaled_mm`'s `scale_a`/`scale_b` are **dequantization** factors
(`output ≈ (mat_a.float()*scale_a) @ (mat_b.float()*scale_b)`), not the
quantization multiplier — passing the multiplier instead of its reciprocal
silently produces >1e5x relative error rather than raising. And cuBLASLt's
row-major/col-major operand-layout requirement means any GEMM operand
reused across *differently-oriented* GEMMs (e.g. a weight used in both the
forward GEMM and the backward dgrad GEMM, which contract along different
axes) needs re-quantizing from a freshly transposed-and-contiguous copy,
then a `.t()` view taken of *that* — a pure memory-layout trick that keeps
the quantized values identical (quantization is elementwise, so it commutes
with transposition) while satisfying cuBLASLt's stride requirements. Getting
this backwards manifests as `CUBLAS_STATUS_NOT_SUPPORTED`, not silently
wrong numbers — the probe script demonstrates the working AND the naive
failing version side by side.

## What each `--precision` value actually runs

| `--precision` | Native or scoped-down | What runs |
| --- | --- | --- |
| `fp32` | native | strict fp32 (`allow_tf32=False`, `matmul_precision='highest'`), no autocast |
| `tf32` | native | fp32 storage, TF32 tensor-core matmuls (`allow_tf32=True`, `matmul_precision='high'`), no autocast |
| `bf16` | native | `torch.amp.autocast(bfloat16)` over forward+loss; fp32 params/optimizer (same shape as the pre-existing `--dtype bfloat16` path) |
| `fp16` | native | `torch.amp.autocast(float16)` + `torch.amp.GradScaler` over forward/backward/step; fp32 params/optimizer |
| `fp8` | **native, forward AND backward** | `Float8Linear` swapped onto all 4 per-block projections (QKV, attn-out, MLP fc, MLP proj) of **every** transformer block. Forward: E4M3×E4M3→bf16. Backward: E5M2 `d_output` × E4M3 weight/activation → bf16 (dgrad and wgrad both use `torch._scaled_mm`, not a bf16 fallback). Per-tensor **current/JIT** amax scaling (computed fresh each call), not the Mojo build's delayed/history scaling — a reference-script simplification, not a numerics gap. LayerNorm/softmax/embeddings/LM head stay in bf16 autocast; master weights stay fp32. |
| `nvfp4` | **native forward, bf16 backward** | `NVFP4Linear` swapped onto the MLP `c_fc`/`c_proj` projections of **middle transformer blocks only** (mirrors `train_gpt2.mojo`'s `LLMM_FP4_FIRST`/`LLMM_FP4_LAST` layer policy — first ~2 and final ~2 blocks, plus all attention/LayerNorm/embeddings/LM-head, stay bf16). Forward: real NVFP4 (E2M1 4-bit elements, 16-element E4M3 block scale, single-level `BlockWise1x16` cuBLASLt/cutlass tensor-core GEMM). Backward (dgrad/wgrad): plain bf16 matmul on the *unquantized* activation/weight — see "Why NVFP4 backward stays bf16" below. |

fp8/nvfp4 require CUDA (`torch._scaled_mm`'s fp8/fp4 tensor-core dispatch
doesn't exist on CPU/MPS); the script asserts this explicitly rather than
silently falling back.

### Why NVFP4 backward stays bf16 (not a probe failure)

A real NVFP4 dgrad/wgrad needs each operand re-blocked-and-requantized along
whichever axis *that* GEMM contracts over — forward contracts over
`in_features`, dgrad over `out_features`, wgrad over the batch×seq axis, so
naively that's three distinct block orientations per tensor (the fp8 path
above does this exact re-orientation trick, but only needs a *scalar*
per-tensor scale each time; NVFP4's block-scale + swizzle bookkeeping makes
the equivalent three-way version substantially more code). Per
`docs/ai/fp4_training_recipes_research.md` §1, a numerically **stable**
fwd+bwd NVFP4 recipe additionally wants stochastic rounding on gradient
operands and a 16×16 Hadamard transform on the Wgrad input — machinery this
reference script doesn't otherwise implement anywhere (there is no
stochastic rounding path here at all, unlike the Mojo trainer's
`matmul_bwd_fp4`). Given the probe passed, scope was deliberately capped at
"real NVFP4 tensor-core **forward**, honest bf16 backward" — this still
demonstrates genuine end-to-end NVFP4 training (loss decreases through a
real NVFP4 GEMM every step) without mislabeling an intentionally-simplified
backward as the full recipe. `NVFP4Linear`'s docstring in `train_gpt2.py`
carries the same explanation next to the code.

## Compatibility (no behavior change without `--precision`)

`--precision` defaults to `""` (unset). Every existing caller
(`scripts/benchmark_train.py`'s CUDA and MPS arms, which drive this file via
`--dtype`/`--tensorcores` and never pass `--precision`) takes the *exact*
original code path — the `ptdtype`/`ctx` setup block, the TF32 knob, and the
training loop's `loss.backward()`/`optimizer.step()` calls are byte-for-byte
unchanged in that branch. Verified: running the pre-change `train_gpt2.py`
and the post-change one with identical `--dtype float32 --tensorcores 1` and
`--dtype bfloat16` arguments (`--model d12`, 15 steps, `tinyshakespeare`
val split) produces **identical per-step losses/norms/lr to the printed
digits** in both cases (see the validation table below — "fp32 (legacy)" and
"bf16 (legacy)" rows exactly match the unmodified-script run).

## Validation

10–20 steps each, `--model d12` (random-init GPT-2 124M, no HF download),
`tinyshakespeare` val split (the default `--input_bin`), `--batch_size 4
--sequence_length 64 --total_batch_size 256` (grad_accum=1), NVIDIA GB10.
"legacy" rows ran the unmodified (pre-this-change) `train_gpt2.py` for a
byte-identical-behavior cross-check; all other rows ran the new
`--precision` flag.

| Precision | Native/Emulated | Steps | First loss | Last loss | ~step time (post-warmup) | NaN? |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 (legacy, `--dtype float32 --tensorcores 1`) | native | 15 | 11.003708 | 1.317628 | ~47 ms | no |
| bf16 (legacy, `--dtype bfloat16`) | native | 15 | 11.003174 | 1.317638 | ~50 ms | no |
| fp32 (`--precision fp32`) | native | 15 | 11.003649 | 1.318913 | ~56 ms | no |
| tf32 (`--precision tf32`) | native | 15 | 11.003708 | 1.317628 | ~47 ms | no |
| bf16 (`--precision bf16`) | native | 15 | 11.003174 | 1.317638 | ~50 ms | no |
| fp16 (`--precision fp16`) | native, GradScaler | 15 | 11.003906 | 1.320246 | ~54 ms | no |
| fp8 (`--precision fp8`) | native fwd+bwd | 20 | 11.002930 | 0.280753 | ~73 ms | no |
| nvfp4 (`--precision nvfp4`) | native fwd, bf16 bwd | 20 | 11.005615 | 0.291298 | ~83 ms | no |

(`--precision fp32`/`tf32` losses match the legacy fp32/bf16 rows to 3-4
significant figures — small residual differences vs. the *legacy* fp32 row
come from `allow_tf32`/`matmul_precision` being set explicitly rather than
left at whatever the process-global default was; `tf32` matches the legacy
fp32 row almost exactly since both end up TF32-enabled with `tensorcores=1`
in the legacy case. fp8/nvfp4 reach a visibly *lower* loss than the
bf16/fp32 rows at step 20 vs 15 — different step counts, not evidence of a
numerics advantage; both decrease monotonically-ish with no divergence or
NaN, which is the actual bar for this validation pass.)

Step times are **not** a claimed benchmark result — this reference script
quantizes fp8/nvfp4 operands from scratch every forward/backward call (no
delayed scaling, no cached quantized copies across the fwd/bwd boundary
beyond what a given `torch.autograd.Function.forward`/`backward` pair
naturally shares), so fp8/nvfp4 being slower than bf16 here is expected and
not informative about the underlying tensor-core throughput (see
`tests/probe_fp4/RESULTS.md`'s §Timing for real throughput context).

## Usage

```sh
# Native TF32 matmuls, otherwise fp32 storage
pixi run -e cuda python train_gpt2.py --precision tf32 --model d12 --device cuda \
    --num_iterations 20 --batch_size 4 --sequence_length 64 --total_batch_size 256

# fp16 + GradScaler
pixi run -e cuda python train_gpt2.py --precision fp16 --model d12 --device cuda \
    --num_iterations 20 --batch_size 4 --sequence_length 64 --total_batch_size 256

# Native fp8 (all 4 per-block projections, every layer, fwd+bwd)
pixi run -e cuda python train_gpt2.py --precision fp8 --model d12 --device cuda \
    --num_iterations 20 --batch_size 4 --sequence_length 64 --total_batch_size 256

# Native NVFP4 forward / bf16 backward on MLP linears of middle blocks
# (override the layer range the same way as the Mojo build, via env vars):
LLMM_FP4_FIRST=2 LLMM_FP4_LAST=10 \
pixi run -e cuda python train_gpt2.py --precision nvfp4 --model d12 --device cuda \
    --num_iterations 20 --batch_size 4 --sequence_length 64 --total_batch_size 256

# Legacy behavior (--precision omitted) — unchanged, this is what
# scripts/benchmark_train.py still drives:
pixi run -e cuda python train_gpt2.py --dtype bfloat16 --tensorcores 1 --model d12 \
    --device cuda --num_iterations 20 --batch_size 4 --sequence_length 64 \
    --total_batch_size 256
```

GPU runs must go through the shared-GPU lock: `flock -w 10800
/tmp/llmm-gpu.lock -c '<cmd>'`.

## Files

- `train_gpt2.py` — `--precision` flag, `Float8Linear`/`NVFP4Linear`
  (subclasses of the file's own `Linear`), `swap_precision_layers`,
  `precision_banner`, and the training-loop `GradScaler` wiring.
- `tests/probe_torch_precisions.py` — the standalone capability probe this
  doc summarizes (run it directly for the full narrative + printed matrix).
- `docs/ai/fp8_training_design.md`, `docs/ai/fp4_training_recipes_research.md`
  — the Mojo-side design docs this PyTorch path's layer policy and fp8
  tensor-format choices mirror.
- `tests/probe_fp8/RESULTS.md`, `tests/probe_fp4/RESULTS.md` — the
  Mojo/raw-cuBLASLt-FFI probes that first established fp8/NVFP4 dispatch on
  this exact GPU; this doc's torch-level probe corroborates both findings
  through a different call surface (`torch._scaled_mm`/`_scaled_mm_v2`
  instead of hand-written cuBLASLt FFI).

---

Written with AI assistance (Claude Code / Sonnet agent), directed by Evan Owen.
