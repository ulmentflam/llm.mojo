# NanoGPT Speedrun Techniques — Mined for llm.mojo FP8/NVFP4 Speedups

Research pass mining Keller Jordan's `modded-nanogpt` speedrun ecosystem (the
Prime Intellect "NanoGPT Speedrun" leaderboard) for techniques that could make
llm.mojo's FP8 and NVFP4 training paths **faster**, triaged against our actual
stack. Every claim is tagged **VERIFIED-WEB (URL)** or **INFERRED**.

## Context: what the speedrun is, and why its evidence is relevant to us

- The speedrun trains a 124M-active-param GPT-2-class model to **3.28 FineWeb val
  loss** on **8×H100** in minimum wallclock. Current record ~1.32 min (record 84,
  05/21/26), down from llm.c's 45 min. VERIFIED-WEB
  (<https://github.com/KellerJordan/modded-nanogpt>,
  <https://app.primeintellect.ai/speedrun/nanogpt>).
- **Directly relevant to us:** their models are *the same size class as ours*
  (124M), so their FP8 stability experience is real evidence for a 124M model,
  not extrapolation from 70B pretraining. Their hardware (8×H100, HBM3
  ~3.3 TB/s/GPU, NVLink) is the opposite of our single bandwidth-poor GB10
  (unified LPDDR5x ~300 GB/s), so *systems/distributed* tricks are mostly Tier C
  while *precision/kernel-fusion* tricks are mostly Tier A.

The single most important finding up front, because it reframes our whole FP8
scaling investment:

> **modded-nanogpt does FP8 training with hardcoded constant per-tensor scales —
> no amax, no history, no dynamic range tracking, no synchronization at all.**
> Their LM-head FP8 GEMM uses literal constants `x_s = 100/448`, `w_s = 1.6/448`,
> `grad_s = grad_scale * 0.75/448`, cast to `torch.float8_e4m3fn` (fwd) /
> `torch.float8_e5m2` (grad) and fed to `torch._scaled_mm`. VERIFIED-WEB
> (<https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/train_gpt.py>,
> <https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/triton_kernels.py>).

Our stack (`llmm/amax.mojo`) implements full TE-style delayed per-tensor scaling
with a 16-deep amax history ring buffer, a two-pass GPU amax reduction, and a
per-operand `AmaxState.update_scale` kernel — *every step, every operand, every
layer*. The speedrun evidence says a 124M model reaches target loss with **none
of that**. This is the backbone of the Tier A recommendations below.

---

## TIER A — step-time-neutral kernel/precision techniques (drop-in)

### A1. Constant (static) FP8 scales — delete the amax machinery from the hot path
- **What:** Replace per-step dynamic/delayed amax scaling with fixed per-tensor
  scale constants chosen once (offline calibration or a short warmup), applied as
  `x/scale → e4m3`. No amax reduction, no history, no scale-update kernel, no host
  readback/sync. VERIFIED-WEB (train_gpt.py: `x_s=100/448`, `w_s=1.6/448`,
  `grad_s=grad_scale*0.75/448`; triton_kernels.py "entirely avoids amax
  computation and synchronization").
- **Why it's fast:** Kills two kernel families per operand per layer per step —
  the two-pass amax reduction (`compute_amax` → `_amax_partial_gpu` +
  `_amax_aggregate_gpu`) and `AmaxState.update_scale` (`_update_scale_gpu`) — plus
  the ring-buffer state traffic. On our bandwidth-poor GB10 these memory-bound
  reduction kernels are pure overhead; the speedrun proves they're unnecessary for
  loss at 124M.
- **Source:** Record 19 "FP8 head, offset logits" (@YouJiacheng, 01/13/25) and
  the mechanism in current `train_gpt.py`. VERIFIED-WEB
  (<https://github.com/KellerJordan/modded-nanogpt/blob/master/README.md> line 129;
  <https://x.com/YouJiacheng/status/1878827972519772241>).
- **Maps to our code:** `llmm/amax.mojo` (bypass `compute_amax`/`AmaxState` for a
  new `ScalingKind`-like "static" path), `llmm/lowp.mojo` (`quantize_devscale`
  already takes a device scale pointer — just point it at a constant instead of
  `AmaxState.scale_inv`), `llmm/matmul.mojo` (`matmul_fwd_lowp`/`matmul_bwd_lowp`
  drop the `AmaxState.update_scale` calls), `train_gpt2.mojo`
  (`LowpState`/`AmaxState` lists become scalar constants; a calibration knob).
- **Expected win against our breakdown:** Our "fp8 quantize family ~17% of step"
  includes the amax reduction + scale-update kernels. Removing them should recover
  a large fraction of that 17% (INFERRED — exact split needs a profile; the
  reduction+update are the memory-bound part). Also removes an fp32 host-buffer
  sync that our own memory notes flag as a past NaN source
  (`bf16-loss-buffer-dtype-mismatch`).
- **Risk to validate:** Static scales need per-tensor calibration; the speedrun
  retuned them twice (records 51 "retune fp8 scales" and 19). Keep a fallback amax
  path behind a comptime flag. VERIFIED-WEB (README line 163, PR 175).

### A2. Quantize weights once per optimizer step, activations once per forward
- **What:** Don't re-quantize the FP8 weight operand inside every GEMM. Quantize
  each weight to e4m3 **once per optimizer step** and each activation **once per
  forward**, then reuse the persistent quantized copy across fwd/bwd. VERIFIED-WEB
  (PR 306 / record 84: "quantize weights once per optimizer step, and activations
  once per forward pass, using per-tensor scales to FP8" —
  <https://github.com/KellerJordan/modded-nanogpt/pull/306>).
- **Why it's fast:** A weight used in fwd (dgrad reuses w, wgrad reuses x) is
  currently quantized multiple times. One quantize per step amortizes the
  bandwidth cost — decisive on the GB10 where quantize is memory-bound.
- **Maps to our code:** `llmm/matmul.mojo` (`matmul_fwd_lowp` currently quantizes
  the weight each call via `lowp_gemm_devscale`; hoist weight quantization to the
  optimizer step), `train_gpt2.mojo` (store a persistent e4m3 weight buffer next
  to bf16/master weights; the AdamW step writes bf16 *and* the e4m3 copy),
  `llmm/lowp.mojo` (`quantize_dual_devscale` already emits natural+transposed — do
  it once at weight-update time so dgrad/wgrad reuse both layouts).
- **Expected win:** Directly attacks the "fp8 quantize family ~17%" and the fp4
  "quantize 14.4%" line items by removing redundant weight re-quantization
  (INFERRED — magnitude depends on how many times each weight is currently
  re-encoded per step; with dual-output transpose reuse this compounds with A3).

### A3. Fuse the FP8 quantize into adjacent kernels; fused tiled transpose
- **What:** (a) Fuse FP8 quantization into the producing/consuming kernel rather
  than a standalone pass — the speedrun landed "fuse fp8 quantization in LM head"
  (record 67, PR 207) and fuses the whole softcapped-CE + fp8 path. (b) Use custom
  **tiled transpose_copy** kernels (64×128 / 32×32 blocks, coalesced) to produce
  the TN-only operand layout cuBLASLt/`_scaled_mm` demands, instead of a strided
  transpose. VERIFIED-WEB (README line 179, PR 207; triton_kernels.py
  "transpose_copy ... 64×128 blocks with coalesced reads and writes").
- **Why it's fast:** Removes a full read+write round-trip of the tensor per
  quantize; coalesced transpose avoids non-coalesced global traffic — again the
  dominant cost on a 300 GB/s part.
- **Maps to our code:** `llmm/lowp.mojo` already has `quantize_transpose_devscale`
  (fused transpose+quantize, 32×32 tile) and `quantize_dual_devscale`
  (Optimization B, one read → natural + transposed) — this is exactly the
  speedrun pattern; **the work-in-flight "tiled transposes / dual-output quantize"
  is confirmed correct-direction.** For NVFP4 the analogue is
  `nvfp4_quant.nvfp4_quantize_transpose`. Push these to be the default path in
  `matmul.mojo`'s lowp wrappers.
- **Expected win:** Attacks both "fp8 quantize ~17%" and the fp4 "RHT-prep 18.8%"
  (the RHT transpose-prep `_rht_transpose_prep` is a transpose+quantize that can
  fuse) (INFERRED).

### A4. Asymmetric rescale + logit softcap so FP8 head stays in e4m3 range
- **What:** The FP8 head pairs the constant scales with a **softcap / asymmetric
  logit rescale** to keep activations inside e4m3's ~448 dynamic range:
  train-time fused softcapped CE, inference `logits = 23*sigmoid((logits+5)/7.5)`.
  This is *how they get away with static scales* — they bound the range by
  construction instead of tracking it. VERIFIED-WEB (README line 16 "Use FP8 for
  head, and asymmetric rescale and softcap logits"; train_gpt.py softcap formula;
  record 18/19).
- **Why it's fast:** It's what makes A1 safe on the head — no amax needed because
  the softcap guarantees the range. Also fuses into the CE kernel (records 60, 67,
  79 progressively fused softcapped CE fwd/bwd). VERIFIED-WEB (README lines 172,
  179, 191).
- **Maps to our code:** **Note our LM head is bf16 and tied to `wte` — we do NOT
  fp8 the head** (confirmed: `train_gpt2.mojo` output matmul uses `params.wte`,
  plain `matmul_fwd`, bf16). So A4 is only relevant *if* we later fp8 the head.
  Softcap would touch the loss/CE path in `train_gpt2.mojo`. Lower priority for us
  than A1–A3, which apply to the 4 per-block linears we already fp8.
- **Expected win:** Enables a future fp8 head (a layer we currently leave bf16),
  but changes the training recipe (softcap) → this straddles Tier A/B. Flagged
  honestly.

### A5. FP8 forward on the MLP up-projection specifically (and *not* down-proj)
- **What:** The most recent FP8 win (record 84) is FP8 on **MLP up-projection
  (c_fc) forward only**; extending to the down-projection (c_proj) **failed**
  because its wider input activations are bandwidth-bound. VERIFIED-WEB (PR 306:
  "attempts to extend to MLP down-projection were unsuccessful due to bandwidth
  constraints from its wider input activations").
- **Why it matters to us:** We fp8 *all four* per-block linears (fp8 mode) and
  fp4 *both* MLP linears of middle blocks. The speedrun's finding that
  **down-projection FP8 is a bandwidth loss even on H100** is a warning for our
  much-more-bandwidth-starved GB10: the c_proj quantize overhead may not be paying
  for itself. Worth an ablation.
- **Maps to our code:** `train_gpt2.mojo` layer-selection (`matmul_fwd_lowp` on
  fc vs proj; `_layer_in_fp4_range` already gates fp4 to middle-block MLP).
  Consider gating fp8 similarly per-linear, measuring c_proj on/off.
- **Expected win:** Possible net speedup by *removing* fp8 from a linear where
  quantize cost > tensor-core savings (INFERRED — needs our own per-linear
  profile; the speedrun result makes it a high-value ablation).

### A6. `use_fast_accum` for the forward FP8 GEMM, precise-accum for backward
- **What:** Forward `_scaled_mm(..., use_fast_accum=True)`; backward
  (dgrad/wgrad) uses `use_fast_accum=False`. VERIFIED-WEB (train_gpt.py: fwd
  `use_fast_accum=True`, bwd-x `use_fast_accum=False`).
- **Why it's fast:** Fast-accum uses lower-precision tensor-core accumulation for
  the fwd where it's tolerable; keeps precise accumulation where gradients need it.
- **Maps to our code:** `llmm/matmul.mojo` `_matmul_cublaslt_fp8` / cuBLASLt algo
  selection (`_lt_pick_algo`) — check whether we expose the fast-accum flag and
  whether we've split fwd (fast) vs bwd (precise). Cheap to verify/enable.
- **Expected win:** Small but free if cuBLASLt on sm_121 honors it (INFERRED).

---

## TIER B — techniques that change the training recipe/architecture (need sign-off)

These break the strict llm.c-parity story (our benchmark is *matching llm.c's
exact GPT-2 arch/optimizer*), so they're evaluated honestly, not recommended
blindly.

### B1. Muon / NorMuon optimizer (the single biggest speedrun win)
- **Speedrun benefit:** Muon (orthogonalized-momentum SGD via Newton–Schulz in
  bf16) on the Linear layers, AdamW on embeddings/head, was the foundational jump
  (7.51h → 4.53h in one worklog). Current records use NorMuon + Polar Express
  orthogonalization. VERIFIED-WEB
  (<https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/>;
  <https://kellerjordan.github.io/posts/muon/>; README records 41/48/57).
- **Transfer to our setting:** Muon changes *convergence per step*, not per-step
  FP8 kernel cost. It would **break our llm.c-parity benchmark** (llm.c uses
  AdamW; our whole "134.6 vs 134.7 ms parity" story is same-arch-same-optimizer).
  Muon's Newton–Schulz runs in bf16 and would interact with our fp32
  master-weight + SR store infra (`llmm/adamw.mojo`, `master_ptr`) — the
  orthogonalization step assumes a matrix param and its own momentum buffer, not a
  drop-in for AdamW's per-element moments.
- **Verdict:** Out of scope for the *parity* benchmark. Only relevant if we ever
  add a "fastest-to-loss" track. Note honestly: it would not speed up a *fixed*
  number of steps; it reduces *steps to target*, a different axis than our
  per-step-latency FP8/FP4 work.

### B2. FlexAttention / FlashAttention-3
- **Speedrun benefit:** FlexAttention (record 12) enabled long-context sparse
  masks (5.03 min jump); FA3 (record 29) further sped attention. VERIFIED-WEB
  (README lines 122, 148).
- **Transfer:** We use `cublasGemmStridedBatchedEx` + custom online-softmax, **no
  flash attention**, T=1024 dense causal. A real flash/FA kernel would cut
  attention memory traffic — plausibly a win even at T=1024 on the GB10. But it's
  a large kernel-engineering effort and orthogonal to FP8/FP4 (attention stays
  bf16 in both stacks). Not a precision-path speedup; a separate project.
- **Verdict:** Legitimate future work but not in the FP8/FP4 mandate; doesn't
  break parity (llm.c also does dense attention) but is a big lift.

### B3. Architecture deltas — rotary, QK-norm, RMSNorm, ReLU², untied/value
  embeddings, U-Net skips, logit softcap
- **Speedrun benefit:** Collectively huge (records 11, 14, 15, 17, and the
  ModernArch baseline). VERIFIED-WEB (README lines 121–127; Tyler Romero worklog).
- **Transfer:** **All break llm.c-parity** — llm.c is stock GPT-2 (learned pos
  emb, GELU, LayerNorm, tied head). Keller Jordan himself notes some of these
  "impose additional structure ... unlikely to scale" (softcap). VERIFIED-WEB
  (README line 315). For our fixed-arch parity story these are non-starters.
- **Verdict:** Tier B → effectively Tier C for *us* because our benchmark's whole
  point is matching llm.c's architecture. Listed so we don't relitigate.

### B4. BF16 attn/mlp weights + mixed-precision Muon; BF16 cross-entropy
- **Speedrun benefit:** Record 57 "Bfloat16 attn/mlp weights, mixed precision
  Muon"; record 37 "Compute cross entropy in BF16 during training". VERIFIED-WEB
  (README lines 169, 149).
- **Transfer:** We already store bf16 weights + fp32 master. BF16 CE is the one
  cheap idea here that *doesn't* need Muon: computing the loss/CE in bf16 rather
  than fp32. Could shave the CE/softmax path. Mild recipe change (numerics of the
  loss), so Tier B not Tier A, but low-risk and worth a look.
- **Maps to our code:** loss/CE + `llmm/softmax.mojo` in `train_gpt2.mojo`.

---

## TIER C — NOT transferable (8×H100 / distributed / contradicted by GB10)

Listed briefly so we don't chase them:

- **All gradient all-reduce / communication-overlap records** (22 "Faster
  gradient all-reduce", 23 "Overlap computation and gradient communication", 30
  "noallreduce", 31 async grad hook). Single-GB10 has no NCCL/multi-GPU comm to
  optimize. VERIFIED-WEB (README lines 132–134, 148, 155).
- **Distributed/sharded Muon, sharded mixed-precision Muon.** Distribution-only.
  VERIFIED-WEB (README).
- **Batch-size / batch-size-schedule records tuned for 8×H100 aggregate batch**
  (records 21, 84-adjacent). Our B=4,T=1024 single-device economics differ.
- **PyTorch-version-upgrade records** (18-series re-timings, 24, 25 "PyTorch
  2.9.0"). We're pure Mojo/MAX, not PyTorch. VERIFIED-WEB (README lines 129-fwd).
- **`use_fast_accum`-style H100 tensor-core-count-specific tuning** where it
  depends on H100 SM behavior — verify on sm_121 before trusting (see A6 caveat).

---

## FP8 training stability at 124M scale — the evidence (this is the headline)

**Claim: a 124M GPT-2-class model trains to target loss with static, hardcoded
FP8 per-tensor scales and NO amax history / delayed scaling. VERIFIED-WEB.**

Evidence:
1. Current `train_gpt.py` uses literal constants `x_s=100/448`, `w_s=1.6/448`,
   `grad_s=grad_scale*0.75/448`; e4m3 fwd operands, e5m2 grad operand;
   `torch._scaled_mm`. No amax anywhere. VERIFIED-WEB
   (<https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/train_gpt.py>).
2. `triton_kernels.py` "entirely avoids amax computation and synchronization ...
   scales are externally provided as fixed values." VERIFIED-WEB
   (<https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/triton_kernels.py>).
3. Stability is bought with **range control by construction** (logit softcap +
   asymmetric rescale on the head; per-tensor scales chosen so activations sit
   inside e4m3's ~448), not with dynamic tracking. VERIFIED-WEB (README line 16).
4. The scales *do* matter and were retuned by hand across records (record 19
   introduced them, record 51 "retune fp8 scales", leloykun found "fp8 scales can
   not only affect validation loss, but also the wallclock time" and set them from
   traced weight/grad max-abs). So the cheap approach is **calibrate-once, hold
   constant**, not fully dynamic. VERIFIED-WEB
   (<https://x.com/leloykun/status/1885640350368420160>; README line 163, PR 175).
5. Scope caveat: the speedrun only FP8s the **LM head** and (recently) the **MLP
   up-projection forward** — a *narrower* FP8 surface than our all-4-linears fp8
   mode. Their static-scale success is proven on those layers; our QKV/attn-proj
   fp8 usage is beyond what they've validated. INFERRED (extrapolation).

**Implication for our TE-style delayed scaling (`llmm/amax.mojo`):** the 16-deep
amax history + two-pass reduction + per-step scale-update we built is *more
machinery than a 124M model demonstrably needs*. The speedrun is direct evidence
that we can replace it with calibrated constants on at least the head and MLP
up-proj, and probably more, recovering the memory-bound overhead it costs. We
should keep the amax path behind a flag as a safety net and for the layers the
speedrun never validated (QKV/attn-proj).

## NVFP4 / FP4 in the speedrun lineage — essentially none

**Claim: FP4/NVFP4 was never adopted in the speedrun. VERIFIED-WEB.**
- There is a `records/track_1_short/2024-11-14_QuantizedFP4/` **directory**, but
  it is **not in the README records table** (the table goes record 11 →
  11/10/24 U-Net, straight to record 12 → 11/19/24 FlexAttention; record 14 on
  12/04/24 is "Value Embeddings"). The QuantizedFP4 dir shares its log file hash
  (`a833bed8-...txt`) with record 10 "Bfloat16 activations" (2024-11-08_CastBf16)
  and was only touched by a 2025-10-15 "reorganize records" chore commit — i.e.
  it is an **orphan/experimental artifact that never became a record.** VERIFIED-WEB
  (README lines 120–124; GitHub contents API for the two dirs; commit
  `chore(records): reorganize into track_1_short and track_2_medium`).
  (The 1.4 MB log exceeds GitHub's inline-content API limit so its embedded script
  couldn't be fully decoded here — INFERRED that no FP4 GEMM shipped, from its
  absence in the records table.)
- Every precision win in the actual leaderboard is **FP8 (e4m3/e5m2)** or **bf16
  casts**, never fp4. NVFP4/e2m1/block-scaling never appears. VERIFIED-WEB
  (README precision-related lines: 16–17, 129, 163, 169, 196).

**So our NVFP4 path has no speedrun precedent to borrow from** — the transferable
speedrun ideas (static scales, quantize-once, fused transpose-quantize) apply to
FP8; for FP4 we're on our own recipe (which our `docs/ai/fp4_training_recipes_research.md`
already sources from NVIDIA/TE/Quartet literature instead). The Tier A *kernel*
patterns (A2 quantize-once, A3 fused tiled transpose-quantize) still map onto
`nvfp4_quant.mojo`/`matmul.mojo` even though the *scaling scheme* doesn't (NVFP4's
block scales are inherently dynamic, so A1's static-scale idea does NOT transfer to
FP4 — the e4m3 block scale already adapts within the tensor, which our
`amax.mojo` docstrings already note).

---

## Top-3 recommended implementations (expected-value order)

1. **A1 — static/calibrated FP8 scales, bypassing `amax.mojo` in the hot path.**
   Highest EV: directly deletes memory-bound amax-reduction + scale-update kernels
   that the speedrun proves a 124M model doesn't need, attacking the ~17% fp8
   quantize overhead at its root. Low code surface (`lowp.quantize_devscale`
   already takes a device scale pointer). Keep amax behind a comptime flag.
   Validate loss on our 4-linear fp8 surface (broader than the speedrun's, so
   measure QKV/attn-proj carefully).

2. **A2 — quantize weights once per optimizer step, activations once per forward.**
   Removes redundant weight re-quantization across fwd/dgrad/wgrad; compounds with
   A1 (constant scale means the persistent e4m3 weight copy never needs
   re-scaling) and A3 (dual-output transpose written once at weight-update time).
   Touches `matmul.mojo` lowp wrappers + `train_gpt2.mojo` optimizer step +
   persistent e4m3 weight buffers.

3. **A5 ablation — is FP8 on the MLP down-projection (and QKV/attn-proj) actually
   paying off on the GB10?** The speedrun found down-proj FP8 is a *net loss* even
   on H100 due to wide-activation bandwidth. On our 300 GB/s part the quantize
   cost is relatively larger, so some of our four fp8 linears may be net-negative.
   Cheapest to try (it's a layer-gating flag in `train_gpt2.mojo`), and could
   *reduce* step time by *removing* fp8 where it doesn't pay — a rare "delete code,
   get faster" lever.

(Honorable mention: A3 is already the direction of our in-flight "tiled
transposes / dual-output quantize" work — this research **confirms** that's the
right path and matches the speedrun's `transpose_copy` + fused-quantize pattern.)

---

## Source index (all VERIFIED-WEB unless noted)

- modded-nanogpt repo & README records table —
  <https://github.com/KellerJordan/modded-nanogpt>,
  <https://github.com/KellerJordan/modded-nanogpt/blob/master/README.md>
- FP8 mechanism (constant scales, e4m3/e5m2, `_scaled_mm`, transpose_copy) —
  <https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/train_gpt.py>,
  <https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/triton_kernels.py>
- Record 19 FP8 head — <https://x.com/YouJiacheng/status/1878827972519772241>
- Record 51 retie/retune fp8 scales — PR
  <https://github.com/KellerJordan/modded-nanogpt/pull/175>
- Record 84 FP8 MLP up-proj — PR
  <https://github.com/KellerJordan/modded-nanogpt/pull/306>
- Record 67 fuse fp8 quantization in LM head — PR
  <https://github.com/KellerJordan/modded-nanogpt/pull/207>
- leloykun "fp8 scales affect loss AND wallclock" —
  <https://x.com/leloykun/status/1885640350368420160>
- Muon writeup — <https://kellerjordan.github.io/posts/muon/>
- Tyler Romero speedrun worklog (Muon/FlexAttention/softcap/arch) —
  <https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/>
- Prime Intellect speedrun leaderboard/rules —
  <https://app.primeintellect.ai/speedrun/nanogpt>
- DeepWiki speedrun records/hardware —
  <https://deepwiki.com/KellerJordan/modded-nanogpt/8.1-speedrun-records>

---

Written with AI assistance (Claude Code / Opus agent), directed by Evan Owen.
