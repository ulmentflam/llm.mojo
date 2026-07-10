# Stable FP4 Training of Transformer LMs — Recipes & What llm.mojo Should Adopt

Research snapshot: mid-2026. Scope: **training recipes and numerical stability
only.** Toolchain / MAX / Blackwell-hardware FP4 support is covered by a sibling
agent and is deliberately out of scope here.

Target context: llm.mojo trains GPT-2 124M in pure Mojo on an NVIDIA GB10
(Blackwell, sm_121). bf16 training is solid (fp32 master weights in the AdamW
optimizer; **no stochastic rounding yet** — see "Current state" below). An FP8
mixed-precision path (e4m3 fwd / e5m2 grad, Transformer-Engine-style scaling,
fp32 master weights) is being built by a sibling team.

Every load-bearing claim is tagged **[E]** established (paper/vendor-doc backed)
or **[S]** speculative (my inference / extrapolation). Citations are inline.

---

## TL;DR — the three deliverables

### A. Actionable recipe checklist, ranked

**MANDATORY for FP4 stability (do not train FP4 without these):**

1. **NVFP4 format, not MXFP4** — E2M1 elements, **16-element blocks**, **E4M3**
   per-block scale, plus a **per-tensor FP32** second-level scale. **[E]**
2. **Stochastic rounding (SR) on gradient tensors** (both Dgrad and Wgrad
   inputs). Round-to-nearest-even (RNE) for weights and activations. **[E]**
3. **Random Hadamard Transform (RHT), 16×16, on Wgrad inputs only.** **[E]**
4. **2D 16×16 block scaling for weights**; 1D 1×16 for activations & gradients.
   **[E]**
5. **fp32 master weights** in the optimizer (already present in llm.mojo). **[E]**
6. **Selective high-precision layers**: keep ~15% of linear layers in BF16 —
   the **first ~2 and final several blocks**; **always** keep embeddings, all
   LayerNorms, the LM head, and the attention (QKV/softmax path) out of FP4.
   **[E]**

**NICE-TO-HAVE (close the last of the gap / de-risk):**

7. **Switch forward-pass tensors to BF16 for the final ~1–18% of training**
   (at LR-decay onset). Forward-only switch recovers almost all the gap. **[E]**
8. **Loss-spike / divergence monitoring** vs a BF16 shadow run; watch relative
   validation-loss gap and gradient-norm-to-quant-noise ratio. **[E]/[S]**
9. **Oscillation mitigation** (OsciReset-style bin re-centering) and
   **outlier-channel retention** for the last-mile gap on smaller models. **[E]**

### B. Recommended tensor-format map for our GPT-2 124M pipeline

| Tensor / op | Recommended precision | Why |
|---|---|---|
| Token + positional **embeddings** | BF16 (fp32 master) | Sensitive; tiny FLOP share **[E]** |
| **All LayerNorms** (weights + stats) | fp32 stats / BF16 params | Norms never quantized **[E]** |
| **Attention** QKV proj, out proj, **softmax** | BF16 (at 124M) | Softmax amplifies quant noise **[E]** |
| **MLP** fc / fc_proj GEMMs (middle blocks) | **NVFP4** fwd+bwd | Bulk of FLOPs, FP4-tolerant **[E]** |
| First ~2 blocks' MLP GEMMs | BF16 | Early-layer sensitivity **[E]** |
| Final few blocks' MLP GEMMs | BF16 | Late-layer sensitivity dominates gap **[E]** |
| **LM head / classifier** GEMM | BF16 | Output layer; folds into loss **[E]** |
| Fprop GEMM operands (weight, act) | NVFP4 (RNE) | **[E]** |
| Dgrad GEMM operands (weight, grad) | NVFP4, **SR on grad** | **[E]** |
| Wgrad GEMM operands (act, grad) | NVFP4 + **RHT**, **SR on grad** | **[E]** |
| Adam moments m, v | fp32 | Already fp32 in llm.mojo **[E]** |
| Master weights | fp32 | Already present **[E]** |

Note: at 124M the honest recommendation is to keep attention entirely in
BF16/FP8 and apply NVFP4 **only to the MLP GEMMs of the middle blocks** — a
minority of the network. See the small-scale caveat (§3); the expected win is
small and may be negative.

### C. Top citations

- **NVFP4 pretraining recipe (the anchor paper):** Alvarez et al., NVIDIA,
  "Pretraining Large Language Models with NVFP4," arXiv:2509.25149.
  <https://arxiv.org/abs/2509.25149>
- **Fully-quantized FP4 / the √3 noise threshold:** "FP4 All the Way: Fully
  Quantized Training of LLMs," arXiv:2505.19115 (NeurIPS 2025).
  <https://arxiv.org/abs/2505.19115>
- **FP4 scaling law / compute-optimality:** "Quartet: Native FP4 Training Can Be
  Optimal for Large Language Models," arXiv:2505.14669.
  <https://arxiv.org/abs/2505.14669>
- **Oscillation + outlier control (small models):** "TetraJet-v2: Accurate NVFP4
  Training … with Oscillation Suppression and Outlier Control,"
  arXiv:2510.27527. <https://arxiv.org/abs/2510.27527>
- **Transformer-Engine NVFP4 reference:**
  <https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html>

---

## Current state of llm.mojo (audited)

- `llmm/adamw.mojo`: fp32 master-weight path exists (`has_master`), all Adam math
  in fp32. **Stochastic rounding is NOT implemented** — line ~141 is an explicit
  `TODO: Karpathy adds a stochastic rounding function here for low-precision
  params`. The low-precision store is a plain `param.cast[dtype]()` (RNE-ish
  truncation). **[E, from source]**
- `llmm/rand.mojo`: host-side PyTorch-compatible MT19937 used only for weight
  init. There is **no device-side / per-element RNG** suitable for in-kernel SR.
  A GPU SR primitive would be new infrastructure. **[E, from source]**

Implication: SR (mandatory item #2) is genuinely missing and is the single
biggest build item beyond the FP8 work. It is needed in two places for FP4:
(a) quantizing **gradient GEMM operands** to E2M1, and (b) optionally the
weight-update store (the existing `TODO`). NVFP4's recipe requires SR on
**gradients**, not on the weight update per se, but a cheap counter-based device
RNG (e.g. Philox/`squares`) serves both. **[E]/[S]**

---

## 1. NVFP4 — NVIDIA's recipe (arXiv:2509.25149)

The anchor result: a **12B hybrid Mamba-Transformer trained to 10T tokens**
in NVFP4 lands within **~1%** validation-loss of an FP8 baseline; benchmark
accuracy is on par (MMLU-Pro 62.58 NVFP4 vs 62.62 FP8), with a small deficit on
coding tasks. **[E]**
<https://arxiv.org/abs/2509.25149>,
<https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/>

**Format.** E2M1 (1-sign/2-exp/1-mantissa) elements. Blocks of **16** share an
**E4M3** (FP8) scale; a **per-tensor FP32** scale sits on top ("two-level"
scaling). E4M3 scales are the decisive advantage over MXFP4's power-of-two E8M0
(§2). **[E]**

**Which GEMMs / which operands are FP4.** All **three** per-layer GEMMs take
NVFP4 inputs and emit BF16/FP32 outputs: **[E]**
- **Fprop:** weight ⊗ activation, both NVFP4, **RNE**.
- **Dgrad:** weight ⊗ output-grad, NVFP4, **SR on the gradient operand**.
- **Wgrad:** activation ⊗ output-grad, NVFP4, **RHT + SR on the gradient**.

**Random Hadamard Transform.** 16×16 Hadamard with a **single fixed random sign
vector shared across all layers and all of training**, applied **only to Wgrad
inputs.** It is deliberately **not** applied to Fprop/Dgrad: transforming those
paths would also require transforming the weight, which breaks the 2D-scale
consistency between forward and backward weight representations. RHT
"Gaussianizes" outliers so a 16-value block quantizes with less error. **[E]**
<https://arxiv.org/html/2509.25149v1>

**Stochastic rounding.** SR on **gradient tensors only** (unbiased quant of the
noisy signal that must accumulate correctly); RNE for weights and activations
(lower variance where the value is "real"). NVIDIA reports SR is **essential at
12B/10T** — RNE-only gradients drift and lose convergence. **[E]**

**2D vs 1D scaling.** Weights use **2D 16×16** block scales so the *same* weight
block quantizes identically whether read row-major (Fprop) or column-major
(Dgrad) — forward/backward consistency. Activations and gradients use **1D
1×16** blocks. **[E]**
<https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html>

**Selective high-precision layers.** ~**15–16%** of linear layers stay BF16,
concentrated in the **final blocks** plus the **first ~2**. For the 12B: first 2
+ final 8 of 62 blocks. Embeddings, all norms, and the output head are **always**
higher precision. Attention blocks are kept out of FP4 to avoid softmax
amplifying quant noise. The **final** layers matter most. **[E]**

**Switch-to-BF16 late-training trick.** Switching (forward-pass tensors suffice)
from NVFP4 to BF16 for the tail of training closes most of the residual gap.
Switching at **8.2T of 10T (~18% tail)** cut relative loss error from ~1.5% to
**~0.5%**; switching only in the final <1% (at 10T) had little effect because the
LR was already tiny. Recommended trigger: **LR-decay onset**. **Forward-only**
switch ≈ full switch. **[E]**

**2026 follow-ups.**
- *Nemotron-3* and later NVIDIA models productionize this recipe at scale. **[E]**
  <https://arxiv.org/abs/2512.20856>
- *Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient
  Estimation* (arXiv:2601.22813) — better unbiased gradient estimators on top of
  NVFP4. **[E]** <https://arxiv.org/abs/2601.22813>
- *Metis* (arXiv:2509.00404) argues **spectral decomposition + random embedding**
  beats RHT (RHT "only smooths a few outliers without reducing overall spread");
  claims to **surpass BF16**, vs NVIDIA's recipe's reported ~1–2% higher test
  loss. Heavier to implement. **[E]** <https://arxiv.org/abs/2509.00404>

---

## 2. MXFP4 and why NVFP4 wins for training

MXFP4 (OCP microscaling): E2M1 elements, **32-element** blocks, **E8M0**
(power-of-two, exponent-only) shared scale. **[E]**

**Training-stability verdict:** NVFP4 > MXFP4. NVIDIA's 8B/1T comparison:
NVFP4 ~**1.5%** relative loss error vs MXFP4 ~**2.5%**; **MXFP4 needs ~36% more
tokens (1.36T vs 1T) to match NVFP4's loss.** Two causes: (a) E8M0's power-of-two
scale has no mantissa, so per-block scaling is coarse; (b) 32-element blocks let
a single outlier wreck more values. **[E]**
<https://arxiv.org/abs/2509.25149>,
<https://www.spheron.network/blog/nvfp4-vs-mxfp4-gpu-cloud-4bit-quantization-guide/>

MXFP4's gap is partly closable with **block-wise Hadamard rotation before
quantization** (e.g. MR-GPTQ / AdaHOP-style, ICLR 2026), but for *training* on
Blackwell (which supports NVFP4 natively) there is no reason to pick MXFP4.
**[E]/[S]**

Other relevant 2025–26 training work:
- **"FP4 All the Way"** (arXiv:2505.19115): fully-quantized FP4 (weights, acts,
  grads) to 200B tokens; endorses NVFP4 + **SR on backward/update, RNE on
  forward**; contributes the **√3 threshold** (below, §3). **[E]**
- **TetraJet-v2** (arXiv:2510.27527): **OsciReset** (reset oscillating weights to
  bin center late in training) + **OutControl** (RHT on backward + keep 5–10% of
  large-magnitude activation channels in FP8/BF16). Tested on **OLMo2 70M/150M/
  370M** — directly relevant to our scale — and cuts the FP4↔full-precision gap
  by an average **51.3%** (370M/200B: 25.11 vs 26.20 PPL). **[E]**
- **Dissecting Outlier Dynamics in NVFP4 Pretraining** (arXiv:2602.02047) and
  **"Curse and Blessing of Mean Bias in FP4"** (arXiv:2603.10444) — mechanism
  papers on why SR/RHT help. **[E]**

---

## 3. Small-model caveat (this is the honest part)

**Nearly all FP4 *training* wins are demonstrated at ≥1B params.** The anchor
NVFP4 result is 8B–12B; "FP4 All the Way" is billion-scale; Quartet's
compute-optimality crossover is explicitly in the **multi-billion-parameter
regime**. **[E]** GPT-2 **124M is roughly an order of magnitude below** where any
paper claims FP4 *pays off*.

Why small scale is hostile to FP4:
- **The √3 gradient-noise threshold** (arXiv:2505.19115): once the **gradient
  norm falls below ≈√3 × the quantization noise**, quantized training stalls —
  SR removes *bias* but not *variance*, and the variance floor dominates. Small
  models have smaller gradient norms, so they hit this floor **sooner** and
  spend more of training near it. **[E]**
- **Quartet's scaling law** predicts FP4 is compute-optimal only at large N/large
  data; at ~100–300M the quantization-noise penalty typically **outweighs** the
  compute saving. **[E for direction; S for exact crossover]**
- The **fixed overheads don't shrink**: embeddings, norms, LM head, attention all
  stay BF16 and are a **larger FLOP fraction** at 124M than at 12B, so the FP4-
  eligible MLP GEMMs are a smaller slice — the achievable speedup is smaller
  precisely where the accuracy risk is largest. **[S]**
- **Batch size / gradient signal:** small batches worsen the noise ratio. If FP4
  is attempted at 124M, use the **largest stable batch** and expect SR to be
  load-bearing. **[S, consistent with 2505.19115]**

**Encouraging counter-evidence at our scale:** TetraJet-v2 trains **OLMo2
70M–370M** in NVFP4 and reaches near-full-precision perplexity *with* oscillation
+ outlier handling. So FP4 *can* converge at 124M — the open question is whether
the loss gap is small enough and the *speedup* real enough to be worth it. **[E]**

**Extrapolation risk: HIGH.** Do not assume the 12B recipe transfers to 124M.
The right framing for this project is a **research/parity exercise** ("can we
train GPT-2 124M in FP4 within X% of BF16?"), **not** a throughput win. The
honest expectation is a **measurable loss gap (order ~1–3% relative, possibly
worse)** and **little or no wall-clock benefit at this size** on a single GB10.
**[S]**

---

## 4. Practical delta: FP8 pipeline → FP4

Given a working FP8 mixed-precision path (e4m3 fwd / e5m2 grad, TE-style scaling,
fp32 master), the incremental build for NVFP4:

**Formats.** Everything FP4-eligible becomes **E2M1** with **16-elem blocks +
E4M3 block scale + FP32 per-tensor scale**. Weights, activations, and gradient
GEMM operands all move to NVFP4 for the *quantized* linears (MLP middle blocks).
**[E]**

**Which of the 3 GEMMs go FP4.** All three (Fprop/Dgrad/Wgrad) *can* be NVFP4 on
Blackwell. Conservative staging for a small model: **[E]/[S]**
1. Fprop NVFP4 first (RNE) — easiest, lowest risk.
2. Then Dgrad + Wgrad NVFP4 with **SR on gradients** and **RHT on Wgrad**.
3. Keep attention + first/last blocks + embeddings + norms + head in BF16/FP8.

**What must stay higher precision.** Embeddings, all LayerNorms, LM head,
attention (QKV/out proj + softmax at 124M), first ~2 and final several MLP
blocks, Adam moments, master weights. **[E]**

**New infrastructure vs FP8 (the real delta):**
- **Stochastic-rounding quantizer** for gradient operands + a **device RNG**
  (counter-based Philox/`squares`) — *not present today* (`rand.mojo` is host
  MT19937). **Biggest item.** **[E]**
- **16×16 2D block scaling for weights** (FP8 typically uses per-tensor or 1×N);
  plus 1×16 for acts/grads. New quantizer + scale-factor layout. **[E]**
- **16×16 Random Hadamard Transform** on Wgrad inputs (fixed random sign vector,
  shared globally). Fuse into the pre-GEMM cast. **[E]**
- **Two-level scaling** (E4M3 per-block under FP32 per-tensor) vs FP8's single
  per-tensor scale. **[E]**
- **Per-layer precision routing** (BF16 vs FP4 selection for first/last blocks).
  **[E]**
- **Late-training forward→BF16 switch** hook at LR-decay onset. **[E]**

**Loss-spike / divergence monitoring.**
- Run a **BF16 shadow** (or reuse the known-good BF16 run) and track the
  **relative validation-loss gap**; alarm if it widens past a set band. **[E/S]**
- Watch **grad-norm vs quant-noise**; approaching the **√3** floor predicts a
  stall — that is the cue to trigger the BF16 switch or stop. **[E]**
- Watch **weight-oscillation rate** (fraction of weights flipping FP4 bins per
  step); rising oscillation late in training is the OsciReset trigger. **[E]**
- Standard: monitor global grad-norm spikes (already have `global_norm.mojo`)
  and NaN/Inf guards on the loss readback. **[S]**

**Expected convergence gap.** At large scale, **~1% (≤0.5% with the BF16 tail
switch)**. At **124M, expect worse and be honest about it** — plausibly a few
percent relative loss, possibly a stall near the noise floor; the tail-switch and
TetraJet-v2 tricks are the mitigations most likely to matter at our size. **[E
for large scale; S for 124M]**

---

## Recommendation for llm.mojo, concretely

1. **Land SR + a device RNG first** — it is mandatory for FP4, it also finishes
   the existing `adamw.mojo` `TODO`, and it benefits the FP8 path (SR on e5m2
   grads). Highest-leverage, reusable. **[E]**
2. **Adopt NVFP4, never MXFP4.** **[E]**
3. **Quantize only middle-block MLP GEMMs** initially; keep attention, embeddings,
   norms, head, and first/last blocks BF16. **[E]**
4. **Wgrad = RHT(16×16) + SR; Dgrad = SR; Fprop = RNE.** 2D 16×16 weight scales,
   1×16 act/grad scales, E4M3 + FP32 two-level. **[E]**
5. **Wire the late-training forward→BF16 switch** and the monitoring hooks. **[E]**
6. **Frame it as a parity study, not a speed win.** Set the success bar as
   "within X% of BF16 loss at 124M," measure honestly, and be prepared for the
   answer that FP4 doesn't pay off at this size on one GB10. Consider validating
   the *machinery* at 124M but reserving the *claim* for larger runs. **[S]**

---

## Sources

- NVFP4 pretraining recipe — arXiv:2509.25149 — <https://arxiv.org/abs/2509.25149>
  (HTML: <https://arxiv.org/html/2509.25149v1>)
- NVIDIA blog, NVFP4 training — <https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/>
- Transformer Engine NVFP4 docs — <https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html>
- FP4 All the Way (√3 threshold, FQT) — arXiv:2505.19115 — <https://arxiv.org/abs/2505.19115>
- Quartet (FP4 scaling law / optimality) — arXiv:2505.14669 — <https://arxiv.org/abs/2505.14669>
- Quartet II (unbiased gradient estimation) — arXiv:2601.22813 — <https://arxiv.org/abs/2601.22813>
- TetraJet-v2 (oscillation + outlier, 70M–370M) — arXiv:2510.27527 — <https://arxiv.org/abs/2510.27527>
- Metis (spectral decomposition vs RHT) — arXiv:2509.00404 — <https://arxiv.org/abs/2509.00404>
- Dissecting Outlier Dynamics in NVFP4 Pretraining — arXiv:2602.02047 — <https://arxiv.org/abs/2602.02047>
- Curse and Blessing of Mean Bias in FP4 — arXiv:2603.10444 — <https://arxiv.org/abs/2603.10444>
- NVFP4 vs MXFP4 guide — <https://www.spheron.network/blog/nvfp4-vs-mxfp4-gpu-cloud-4bit-quantization-guide/>
- Nemotron 3 (productionized recipe) — arXiv:2512.20856 — <https://arxiv.org/abs/2512.20856>

---

Written with AI assistance (Claude Code / Opus agent), directed by Evan Owen.
