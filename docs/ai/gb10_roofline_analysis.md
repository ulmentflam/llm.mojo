# GB10 roofline analysis: which GPT-2 GEMMs can *ever* be compute-bound, per precision — and what that means for FP8/FP4 training ceilings

**Scope.** GPT-2 training in Mojo on **one NVIDIA GB10** (Grace-Blackwell, DGX
Spark, aarch64, sm_121, unified LPDDR5x). This note answers, rigorously and
per-GEMM: on *this* box, which matmul sites can **ever** be compute-bound, in
bf16 / fp8 / fp4; what that implies for the best-possible ("ceiling") fp8 and
fp4 training step times at 124M and 774M scale; how much of that ceiling the
shipped code has reached; where the crossover model size is; and what the
"1 PetaFLOP FP4" marketing number would actually require. No new GPU work — all
inputs are prior measured docs (cited inline).

Methodology follows the JAX *Scaling Book* GPU/roofline chapter
(<https://jax-ml.github.io/scaling-book/gpus/>): arithmetic intensity
`AI = FLOPs / bytes-moved`; the **ridge point** (critical intensity) is
`peak-FLOP/s ÷ memory-bandwidth`; an op is **compute-bound** iff its `AI` exceeds
the ridge point, otherwise **memory-bound**. The book's own caveat that measured
(MAMF) throughput sits well below nominal peak, and that small shapes are
latency/bandwidth-bound "despite high theoretical FLOPs," is central here — it is
essentially the whole story on GB10.

---

## 1. Hardware numbers (marketing vs measured) and the ridge points

| precision | nominal dense TFLOPS | **MAMF (measured/derived) TFLOPS** | source |
|---|---:|---:|---|
| FP4 | ~500 (marketing: "1 PFLOP FP4" is the **sparse** peak → ÷2 dense) | **~415** (est: 2× measured fp8; not directly benchmarked) | NVIDIA datasheet; `fp4_modular_support_research.md` §3 |
| FP8 (e4m3/e5m2) | ~250 | **207.7** (measured MAMF) | DGX-Spark forum MAMF; `fp4_modular_support_research.md` |
| BF16 (fp32-accum) | ~125 | **99.8** (measured MAMF) | same |
| TF32 | ~62.5 | **~49.9** (≈ half bf16) | `llmm/mfu.mojo` ladder |

Memory: unified LPDDR5x, spec **~273 GB/s** (DGX Spark); some listings quote up
to ~301 GB/s. The repo's empirical step times are consistent with an *effective*
high-200s GB/s. All ridge points below use **BW = 273 GB/s** (spec anchor); the
273→301 sensitivity is small and shown where it matters. MAMF < nominal is normal
and expected (Scaling Book); on this consumer-Blackwell part it is compounded by
sm_121 having only warp-level `mma.sync` (no `tcgen05`/Tensor Memory), so the
vendor kernels that *do* run here (cuBLASLt sm_120 cubins) are the thinner tier —
see `fp4_modular_support_research.md`.

### Ridge points (critical arithmetic intensity, FLOP/byte)

| precision | MAMF ÷ BW → **ridge (FLOP/byte)** | at 301 GB/s |
|---|---:|---:|
| TF32 | **183** | 166 |
| BF16 | **366** | 332 |
| FP8 | **761** | 690 |
| FP4 | **1520** | 1379 |

**The ridge point doubles every time you halve the operand precision** (because
MAMF doubles while bandwidth is fixed). This single fact drives every conclusion
that follows: to *use* fp8's 2× tensor throughput a GEMM must clear a 2× higher
intensity bar (761 vs 366); to use fp4's 4× it must clear a 4× bar (1520). On a
bandwidth-poor box, most GPT-2-124M GEMMs cannot.

---

## 2. Per-GEMM intensity and the compute-bound verdict

GPT-2-124M, **d12**: C=768, L=12, nh=12, head dim=64, B=4, T=1024 → **M=B·T=4096**.
For a forward GEMM `Y = X·W` with X:[M,K], W:[K,N], Y:[M,N]:
`FLOPs = 2·M·K·N`. Bytes moved depend on the operand-dtype scenario:

- **all-bf16**: `2·(MK + KN + MN)` (every operand 2 B).
- **fp8 in / bf16 out**: `1·(MK + KN) + 2·MN` — inputs halved, **output NOT reduced**.
- **fp4 in / bf16 out**: `(0.5 + 1/16)·(MK + KN) + 2·MN` — packed e2m1 (0.5 B) +
  e4m3 block scale (1 B per 16 elems = 0.0625 B/elem); **output still bf16 (2 B)**.

The bf16 **output** term `2·MN` is the crux: it is the largest operand for these
tall-skinny shapes and **is not shrunk by quantizing the inputs**. Low precision
attacks only the `MK + KN` (input) half of traffic.

### Per-site table (forward orientation, d12, M=4096) — AI and verdict

| site | K | N | GFLOP | AI bf16 | vs 366 | AI fp8 | vs 761 | AI fp4 | vs 1520 |
|---|---:|---:|---:|---:|:--:|---:|:--:|---:|:--:|
| qkv        | 768  | 2304  | 14.5  | 505 | **C** | 609 | mem | 670 | mem |
| attn_proj  | 768  | 768   | 4.8   | 351 | mem | 482 | mem | 576 | mem |
| fc         | 768  | 3072  | 19.3  | 534 | **C** | 630 | mem | 684 | mem |
| fc_proj    | 3072 | 768   | 19.3  | 534 | **C** | 910 | **C** | 1315 | mem |
| lm_head    | 768  | 50257 | 316.2 | 639 | **C** | 697 | mem | 727 | mem |
| attn QKᵀ / AV (per head) | 64 | 1024 | 0.13 | **57** | mem | 78 | mem | 91 | mem |

**C = compute-bound, mem = memory-bound.** (Attention QKᵀ/AV are never quantized
in this codebase; shown to make the point that the per-head 64-deep contractions
are *deeply* memory-bound — AI 57, ~6× under even the bf16 ridge — the least
low-precision-friendly matmuls in the model.)

### The output-orientation subtlety (dgrad / wgrad)

Backward produces two GEMMs per site with the **same FLOPs** but **different
output shapes**, which changes AI and can change the verdict. For qkv:

| orientation | (M,K,N) | AI bf16 | AI fp8 | AI fp4 |
|---|---|---:|---:|---:|
| fwd  (out M×N) | (4096, 768, 2304) | 505 / **C** | 609 / mem | 670 / mem |
| dgrad (out M×K) | (4096, 2304, 768) | 505 / **C** | 828 / **C** | 1151 / mem |
| wgrad (out K×N) | (768, 4096, 2304) | 505 / **C** | 899 / **C** | 1365 / mem |

The dgrad/wgrad orientations have a **smaller output relative to the reduction
depth**, so they clear the fp8 ridge even where the forward does not. This is why
fp8 is not uniformly hopeless at 124M — the *backward* GEMMs are compute-bound in
fp8 — but the output-heavy **forward** GEMMs are not, and they gate the win.

### Verdict summary — "can this site EVER be compute-bound?" (d12, 124M)

| site | bf16 | fp8 | fp4 |
|---|:--:|:--:|:--:|
| qkv (fwd) | ✅ | ❌ | ❌ |
| qkv (dgrad/wgrad) | ✅ | ✅ | ❌ |
| attn_proj | ❌ | ❌ | ❌ |
| fc (fwd) | ✅ | ❌ | ❌ |
| fc (dgrad/wgrad) | ✅ | ✅ | ❌ |
| fc_proj (all) | ✅ | ✅ | ❌ |
| lm_head | ✅ | ❌ | ❌ |
| attention QKᵀ/AV | ❌ | ❌ | ❌ |

**Headline of the section:** at 124M, **no GEMM in the model is compute-bound in
fp4** — the deepest one (wgrad, AI 1365) still falls short of the 1520 ridge. In
fp8, only the backward and fc_proj orientations clear the bar; every
output-dominated forward GEMM (qkv, fc, **lm_head**) is memory-bound. In bf16 most
linear GEMMs are already compute-bound. **Precision reduction moves the goalposts
faster than it moves the ball.**

---

## 3. Why fp8 wasn't 2× — modeling the realized speedup, and the quantization tax

Two independent effects cap the win, both visible in the measured `ncu` buckets
(`ai_assisted_optimizations_and_benchmarks.md`, Chunk F + quant-opt closeout).

**(a) Unreduced output + sub-peak utilization.** The *ideal* roofline (every GEMM
at `max(compute, memory)` using MAMF and BW) predicts the four fp8 linear-GEMM
sites should run at **0.529×** bf16's time. The **measured** GEMM-kernel ratio is
**0.832× wall (0.863× ncu, i.e. only 13.7% faster)**. The gap is the
*realization* gap: at M=4096 the shapes do not fill the 2×-denser fp8 tensor
pipeline, so the realized-vs-roofline efficiency is **worse** for fp8 (η≈0.26)
than for bf16 (η≈0.38). The output-traffic floor plus poor fp8 utilization
together explain the "only 13.7%."

**(b) The quantization tax.** fp8/fp4 training must *read* each bf16 operand and
*write* low-precision copies (natural + transposed for the cuBLASLt orientations
the backward needs). That traffic is pure overhead at bandwidth-bound rates. Post
optimization (coalesced transpose + dual-output d_output fusion), the quantize
family is **25.1 ms wall / step** at d12/B4 — **16.8% of fp8 GPU time** and the
single reason fp8 is *slower* than bf16 there.

### Measured wall-clock bucket decomposition (d12, B=4, T=1024)

| bucket | bf16 | fp8 (shipped) |
|---|---:|---:|
| all GEMMs (linear + lm_head + attention) | 86.0 ms | 71.6 ms |
| quantize/amax/scale family | 0 | 25.1 ms |
| everything else (LN, GELU, softmax, optimizer, encoder, split/merge) | 48.0 ms | 53.3 ms |
| **total (wall)** | **134 ms** | **150 ms** |

fp8 shaves 14.4 ms off GEMMs and spends 25.1 ms quantizing to do it — a net **+16
ms loss**. That is the entire story at 124M.

### Postdiction-vs-measured calibration

| config | metric | model | measured | notes |
|---|---|---:|---:|---|
| d12/B4 | fp8/bf16 GEMM-kernel ratio | 0.53 (roofline ideal) → **0.83 (with realization η)** | **0.83** (−13.7% ncu) | model reproduces the measured GEMM ratio once η is applied |
| d12/B4 | fp8 quant tax (wall) | ~24 ms (traffic ÷ BW, ×transpose factor) | **25.1 ms** | within ~5% |
| d12/B4 | fp8 step | ~150 ms | **150.5 ms** | anchored |
| d12/B4 | fp4 step | ~1.16–1.30× bf16 (fp4 GEMMs *all* memory-bound; extra pack/scale traffic) | **1.155×** (post-tiling) … 1.375× (pre) | model postdicts fp4 loses; magnitude tracks |
| d36/B4 | ideal fp8/bf16 linear-GEMM ratio | **0.481** | fp8 step **0.881–0.920×** bf16 | GEMMs now dominate the step → the deep-GEMM fp8 win surfaces end-to-end |
| d36/B4 | fp4 step | ~1.0× (fwd/dgrad still memory-bound; only wgrad compute-bound) | **1.004–1.245×** | model postdicts fp4 ≈ breakeven, never a win |

The model postdicts the **sign and rough magnitude** everywhere: fp8 loses at
d12, wins modestly at d36; fp4 loses at d12, breaks even at d36. It is not a
sub-1% wall-clock predictor (the realization η and per-config kernel-quality
differences are lumped), but it correctly explains *why* each number lands where
it does, and the fp8-GEMM ratio and quant tax match the measured buckets within
~5%.

---

## 4. Ceilings: best-possible fp8/fp4 step times if all overhead vanished

"Ceiling" = quantization made **free**, **all** linear sites *and* the lm_head in
low precision, GEMMs at today's realized kernel efficiency, irreducible non-GEMM
work (LN/GELU/softmax/optimizer) unchanged.

### d12 (124M), B=4

| scenario | step time | vs bf16 | 
|---|---:|---:|
| bf16 (measured) | 134 ms | 1.00× |
| **fp8 ceiling**, lm_head bf16, quant-free | **~125 ms** | **0.93×** |
| **fp8 ceiling**, lm_head fp8, quant-free | **~115 ms** | **0.85×** |
| **fp4 ceiling**, lm_head bf16, quant-free | **~104 ms** | **0.78×** |
| fp8 measured (shipped) | 150 ms | 1.12× |
| fp4 measured (post-tiling) | ~155 ms | 1.155× |

**Fraction of ceiling reached at 124M: zero — and the ceiling itself is
shallow.** Even with *free* quantization the fp8 step only falls to ~125 ms
(0.93×): a **7% best-case win** with lm_head left in bf16. The shipped fp8 (150 ms)
is *above* bf16, so we have realized **none** of that 7% — the 25 ms quant tax
alone exceeds the entire available headroom. Covering the lm_head too pushes the
ceiling to ~0.85× and fp4's to ~0.78×, but those require *both* free quant *and*
fp8/fp4 kernels that hit their roofline — neither is true today. **At 124M there
is almost nothing to win, so overhead trivially swamps it.**

### d36 (774M), B=4 — where the ceiling opens up

At d36 the GEMMs dominate the step and the ideal fp8/bf16 linear-GEMM ratio falls
to **0.481** (deeper K → forwards clear the fp8 ridge too). The measured fp8 step
already **crosses to 0.881–0.920×** with the quant tax *still present*, so the
quant-free fp8 ceiling is meaningfully below that — on the order of **~0.80–0.85×**
bf16. This is the regime where fp8 is a real, if modest, end-to-end win; the
shipped code has reached a large fraction of it because quant is now a smaller
share of a GEMM-dominated step. fp4 at d36 sits at ~1.0–1.25× (breakeven at best):
its wgrad orientation is compute-bound but the output-heavy forward/dgrad are
still memory-bound, capping it.

---

## 5. Full-low-precision scenarios (including the lm_head)

The **lm_head is the single biggest FLOPs block**: M×768×50257 = **316 GFLOP
forward** (≈ the four per-layer linear GEMMs of *one* block, ×~16), and it recurs
in dgrad+wgrad. In bf16 it is ~9.5 ms of ideal GEMM time — a full section of the
step. But its forward is **output-dominated** (N=50257 → the M×V logits at 2 B
each are enormous) and therefore **memory-bound in both fp8 (AI 697 < 761) and
fp4 (727 < 1520)**.

Consequence per the traffic model: quantizing the lm_head **does not** unlock a 2×
here — the M×50257 bf16 logit write dominates and is untouched. The modeled
benefit of moving lm_head to low precision at d12/B4:

| move | modeled step-time reduction |
|---|---:|
| lm_head bf16 → fp8 (quant-free) | ~10 ms (0.93×→0.85×) |
| lm_head bf16 → fp4 (quant-free) | ~17 ms |

"Everything in fp8 incl. lm_head" → **~115 ms ceiling (0.85×)**; "everything in
fp4" → **~104 ms ceiling (0.78×)** — *and only if quantization is free*. With the
real quant tax added back, full-fp8 at 124M lands near **~140 ms** (still worse
than or roughly equal to bf16) and full-fp4 higher still, because the lm_head's
own operands must now also be quantized every step against an unreduced 2 B logit
output. **The biggest-FLOPs block is also one of the most memory-bound, so it is
the *least* rewarding to quantize per byte of quant traffic spent.**

---

## 6. The "1 PetaFLOP FP4" question, answered plainly

NVIDIA's "1 PFLOP FP4" is the **sparse** peak; **dense** is ~500 TFLOPS, and the
*measured* MAMF ceiling on this box is ~415. To actually *sustain* 500 dense-fp4
TFLOPS a workload must be compute-bound at the fp4 ridge:

**required AI > 500e12 ÷ 273e9 ≈ 1832 FLOP/byte** (≈1520 against the measured 415
MAMF).

GPT-2-124M's fp4 arithmetic intensities span **576–1365 FLOP/byte** across all
sites and orientations. The deepest single GEMM (wgrad, 1365) is **~1.3× under**
even the MAMF ridge and **~1.5× under** the marketing ridge; the output-dominated
forwards (576–727) are **3× under**. So GPT-2-124M is **not marginally short — it
is categorically in the wrong regime**: every one of its fp4 GEMMs is
memory-bound, and the effective fp4 throughput it can ever deliver is set by
bandwidth, not by the 500-TFLOP tensor cores, which sit **mostly idle**. Delivered
fp4 FLOP/s at 124M is **an order of magnitude** below the 500-TFLOP headline
(measured MFU in the sweep was **4.5%** of the 415 MAMF ≈ 18.8 TFLOPS delivered —
i.e. ~3.8% of the 500 marketing number).

**What it would take to see 500 dense-fp4 TFLOPS on GB10:** GEMMs with
`AI > ~1800` — large square-ish contractions where the output is small relative to
`M·K·N`. In transformer terms that means **wide** models (large C, large batch, so
the M×K and K×N terms dominate the M×N output), specifically the fp4 forward GEMMs
need width **C ≳ 1800** to clear the ridge:

| width C | fc-fwd AI (fp4) | fc-wgrad AI (fp4) | fp4 forward compute-bound? |
|---:|---:|---:|:--:|
| 768 (124M) | 684 | 1365 | ❌ |
| 1280 (774M) | 1105 | 1928 | wgrad only |
| 1536 (~1.5B) | 1306 | 2114 | wgrad only |
| **1792 (~2B)** | **1502** | 2271 | ✅ forward crosses |
| 2048 (~2.6B) | 1691 | 2405 | ✅ |
| 3072 | 2398 | 2789 | ✅ |

---

## 7. Conclusions (plain language)

1. **On GB10, precision reduction raises the compute-bound bar faster than GPT-2's
   GEMMs can meet it.** The ridge point doubles from bf16 (366) to fp8 (761) to
   fp4 (1520 FLOP/byte); the bf16 GEMM output (M×N×2 B) is not reduced by
   quantizing inputs, so most 124M forward GEMMs that are compute-bound in bf16
   become **memory-bound** in fp8, and **all** of them are memory-bound in fp4.

2. **Can full-fp4 training ever pay at 124M on this box? No — not even in
   principle.** No GEMM in the 124M model is compute-bound in fp4 (deepest AI 1365
   < 1520 ridge), so fp4 cannot convert its 4× nominal tensor throughput into
   delivered FLOP/s; the quant/pack/scale traffic makes it a **guaranteed net
   loss** (measured 1.155–1.375× *slower*). The fp4 tensor cores sit idle.

3. **fp4 starts to pay only at width C ≳ 1800 (~2B params),** where the
   output-dominated *forward* GEMMs finally clear the fp4 ridge and fp4's 2×-over-
   fp8 GEMM advantage becomes visible end-to-end. Below that (including d36/774M,
   where only the wgrad orientation is fp4-compute-bound), fp4 is breakeven at
   best.

4. **The realistic fp8 ceiling.** At 124M the *quant-free* fp8 ceiling is only
   ~0.93× bf16 (lm_head bf16) or ~0.85× (full) — a 7–15% win that today's 25 ms
   quant tax entirely erases (shipped fp8 is 1.12× *slower*). fp8 becomes a
   genuine, ~0.88–0.92× **measured** win at **d36/774M**, where GEMMs dominate the
   step; its quant-free ceiling there is ~0.80–0.85×. **fp8 is worth it from
   roughly 6× this repo's default width upward; fp4 needs ~16× and is a research
   curiosity at 124M.** The lm_head — the biggest FLOPs block — is also memory-
   bound in both low precisions, so it is the least rewarding place to spend quant
   traffic, not the jackpot its FLOP count suggests.

---

## Appendix: reproducibility

All figures derive from: `docs/ai/lowp_scaling_sweep_2026-07-10.md` (batch/width
sweep, MFU table), `docs/ai/ai_assisted_optimizations_and_benchmarks.md`
(2026-07-10/11 ncu buckets: Chunk F, quant-opt closeout, shipped-tree 6-arm),
`docs/ai/fp4_modular_support_research.md` (MAMF, sm_121 kernel path, 1-PFLOP
framing), and `llmm/mfu.mojo` (FLOP/token formula, tensor-core ladder). The
roofline arithmetic (ridge points, per-site AI, ceilings) is reproduced by the
scratch scripts described in this session; every table entry is `2·M·K·N` FLOPs
over the scenario byte model of §2, divided against the §1 MAMF/BW constants.

---

Written with AI assistance (Claude Code / Opus agent), directed by Evan Owen.
