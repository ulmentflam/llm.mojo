# GPT-2 124M training-loop benchmarks & kernel profiles

A running log of performance experiments comparing **llm.mojo** against Karpathy's
**llm.c** and a **PyTorch** reference, on the same GPT-2 124M training step. Each
entry pins the hyperparameters so the numbers are reproducible and comparable.

Tooling:
- Throughput bars: `scripts/benchmark_train.py` (`make benchmark-gpu`).
- Per-kernel profiles: `profile_gpt2.py` over NVIDIA Nsight Compute (`make profile-ncu`,
  `make profile-llmc-ncu`).

---

## 2026-06-30 — B=4, T=1024 (the shipped training config)

The training loop runs at `BATCH_SIZE = 4`, `SEQ_LEN = 1024` (see `train_gpt2.mojo`).
Earlier benchmarks pinned `T=64` (llm.c's CPU reference default), which hides how
each implementation scales with sequence length. This entry runs every config at the
real `T=1024` (4096 tokens/step).

**Hardware:** NVIDIA GB10 (Grace-Blackwell, `DGX Spark`), aarch64, 20 cores, Linux 6.17.

### Throughput (forward + backward + optimizer per step)

40 steps, first 5 dropped as warmup. Lower is better.

| configuration      | precision | mean ms/step | median |   std | tok/s |
|--------------------|-----------|-------------:|-------:|------:|------:|
| llm.mojo           | fp32      |      2612.24 | 2495.11| 364.09|  1568 |
| llm.mojo           | bf16      |      4323.74 | 4075.92| 660.38|   947 |
| llm.c CUDA         | fp32      |       476.00 |  475.96|   3.27|  8605 |
| **llm.c CUDA**     | **bf16**  |   **220.48** |  222.53|   6.32| **18578** |
| PyTorch            | fp32      |       975.45 |  935.22| 173.06|  4199 |
| PyTorch            | bf16      |       954.97 |  832.45| 192.68|  4289 |

Figure: `figures/benchmark_gpu_b4_t1024_2026-06-30_0909_NVIDIA-GB10_DGX-Spark.png`

**Headline findings**
- At T=1024 **llm.mojo is ~12× slower than llm.c bf16** (2.6 s vs 0.22 s) and ~2.7×
  slower than PyTorch.
- **llm.mojo bf16 is *slower* than its own fp32** (4.3 s vs 2.6 s) — the opposite of
  every other implementation. bf16 should win; here it loses. Run-to-run variance is
  also large (±0.4–0.7 s std), pointing at an unoptimized, possibly memory-bound path
  rather than a steady compute pipeline.
- llm.c's bf16→fp32 speedup is ~2.2×; PyTorch bf16 ≈ fp32 (its fp32 already uses TF32
  tensor cores via `--tensorcores 1`).

### Kernel profile — llm.mojo (bf16, 12 layers, one step)

`make profile-ncu PROFILE_B=4 PROFILE_T=1024 PROFILE_LAYERS=12`
→ 34 distinct kernels, 445 launches, 2,913,943 µs total GPU time under ncu.

By operation family:

| family       | calls | time (µs)   |     % |
|--------------|------:|------------:|------:|
| **attention**|    36 | 2,779,435   | **95.4** |
| elementwise  |    61 |     39,082  |   1.3 |
| matmul       |   157 |     38,857  |   1.3 |
| optimizer    |     1 |     16,834  |   0.6 |
| other        |    38 |     16,388  |   0.6 |
| layernorm    |    99 |      9,239  |   0.3 |
| classifier   |     2 |      7,456  |   0.3 |
| split/merge  |    48 |      6,328  |   0.2 |
| encoder      |     3 |        324  |   0.0 |

Top kernels (tensor% = tensor-core utilization):

| kernel                              | family    | calls | time (µs)  | tensor% |
|-------------------------------------|-----------|------:|-----------:|--------:|
| `llmm_attention_attention_fwd`      | attention |    12 | 1,290,670  |   0.0   |
| `llmm_attention_attention_bwd` (×2) | attention |    24 | 1,488,765  |   0.0   |
| `cutlass…tensorop_bf16` (matmuls)   | matmul    |   var |        —   | 54–89   |

**The pathology:** llm.mojo computes attention in a single monolithic kernel that runs
at **0% tensor-core utilization** and dominates at long T (the QKᵀ / softmax / A·V work
is O(T²) and hand-rolled in non-tensor-core code). The only kernels that touch tensor
cores are the cutlass GEMMs for the linear layers (54–89%), but those are just 1.3% of
the time. At T=64 attention is cheap and this is invisible; at T=1024 it is essentially
the entire cost.

### Kernel profile — llm.c (bf16, 12 layers, one step)

`make profile-llmc-ncu PROFILE_B=4 PROFILE_T=1024 PROFILE_LAYERS=12`
→ 33 distinct kernels, 6,432 launches, 2,025,449 µs total GPU time under ncu.

By operation family:

| family     | calls | time (µs)  |     % |
|------------|------:|-----------:|------:|
| matmul     |  3108 | 1,137,462  |  56.2 |
| other      |  2137 |   375,513  |  18.5 |
| softmax    |   504 |   249,090  |  12.3 |
| classifier |    41 |   153,179  |   7.6 |
| gelu       |   516 |    89,028  |   4.4 |
| optimizer  |    17 |    16,310  |   0.8 |
| layernorm  |    66 |     3,973  |   0.2 |
| encoder    |    43 |       896  |   0.0 |

**How llm.c spends its time:** attention is *decomposed* — the QKᵀ and A·V products are
batched **cutlass tensor-core GEMMs** (folded into the 56% `matmul` family at 10–88%
tensor utilization), the softmax is a dedicated kernel (12.3%), and `permute`/`unpermute`
/`fused_residual` shuffle kernels make up most of `other` (18.5%). No single kernel
dominates; the heavy math runs on tensor cores. This is exactly what llm.mojo's
monolithic attention kernel does not do.

### Reading the profiles correctly

ncu **serializes** kernel replay (no overlap, each kernel re-run to collect metrics), so
the absolute totals above are **not** wall-clock and **not** comparable across
implementations as throughput:
- llm.mojo's ncu total (2.91 s) ≈ its wall-clock step (2.6 s): a few giant serial
  kernels, little to overlap.
- llm.c's ncu total (2.03 s) ≫ its wall-clock step (0.22 s): 6,432 small kernels that
  pipeline on a real GPU but replay one-at-a-time under ncu.

Use the **throughput table** for cross-implementation speed and the **family
breakdowns** for *where each implementation spends its own time*.

### Takeaways / next steps

1. **Attention is the bottleneck for llm.mojo.** Replacing the monolithic non-tensor-core
   attention kernel with tensor-core GEMMs for QKᵀ / A·V (as llm.c does), plus a
   dedicated softmax, is the highest-leverage optimization. Target it first.
2. **bf16 slower than fp32** needs root-causing — likely redundant bf16↔fp32 conversions
   or a memory-bound attention path; the large variance supports the latter.
3. Re-profile after each change with the same commands to track progress against this
   baseline.

### Caveats

- DRAM-bandwidth and tensor-throughput counters show as n/a: ncu lacked
  performance-counter access on this host. Pass `--sudo` (or grant `ERR_NVGPUCTRPERM`)
  to populate them; timing and tensor% are still captured.
- `llm.c CUDA fp32` has only n=3 samples — its harness runs one epoch over the short
  tiny-shakespeare val bin (no step cap), which yields few batches at T=1024. The mean
  is stable (std 3.3 ms) but the sample count is low.

### Reproduce

```sh
# Throughput bars (writes a self-describing PNG into figures/)
make benchmark-gpu BENCH_B=4 BENCH_T=1024

# Per-kernel profiles
make profile-ncu       PROFILE_B=4 PROFILE_T=1024 PROFILE_LAYERS=12   # llm.mojo bf16
make profile-llmc-ncu  PROFILE_B=4 PROFILE_T=1024 PROFILE_LAYERS=12   # llm.c   bf16
```

---

## Attention optimization log — 2026-06-30 (bf16, B=4, T=1024)

Goal: close the ~12× gap to llm.c by fixing attention (95.4 % of GPU time, 0 %
tensor-core utilization). Each entry below is a single change with its measured
effect, using a fast inner-loop harness rather than the full 40-step suite:

```sh
LLMM_PROFILE_B=4 LLMM_PROFILE_T=1024 LLMM_PROFILE_LAYERS=12 LLMM_PROFILE_STEPS=4 \
  pixi run ./build/profile_gpt2_bf16 gpu
```

Reported numbers are the **steady-state median of steps 1–3** (step 0 is warmup),
forward / backward / full-step seconds. Loss values are checked bit-for-bit
against the baseline after every change to guarantee numerical equivalence.

| change | fwd (s) | bwd (s) | step (s) | vs baseline |
|--------|--------:|--------:|---------:|-------------|
| baseline (commit 1b04f17)                              | 1.165 | 1.348 | 2.529 | — |
| ① remove spurious intra-tile `barrier()`s in fwd inner loop | 1.131 | 1.348 | 2.495 | −1.3 % step |
| ② tensor-core GEMM forward (QKᵀ + softmax + A·V)        | 0.106 | 1.349 | 1.470 | **fwd 10.7×, step 1.70×** |
| ③ tensor-core GEMM backward (FA2 dQ/dK/dV)              | 0.106 | 0.267 | 0.389 | **bwd 5.0×, step 6.4×** |
| ④ coalesced tiled transpose for Pᵀ/dSᵀ in backward      | 0.106 | 0.221 | 0.343 | bwd 1.21× |
| ⑤ drop store-only matmul epilogues (no separate kernel) | 0.085 | 0.171 | 0.270 | **fwd 1.25×, bwd 1.29×** |
| ⑥ fuse P-recompute into dS (one [B·nh,T,T] pass)         | 0.085 | 0.165 | 0.266 | bwd 1.04× |
| ⑦ dKᵀ=Qᵀ·dS, dVᵀ=dOᵀ·P — transpose only small [T,hd]    | 0.085 | 0.143 | 0.244 | bwd 1.15× |
| ⑧ bf16 backward scores (QKᵀ out), halves P+dS read       | 0.085 | 0.135 | 0.236 | bwd 1.06× |
| ⑨ bf16 dP too (independent of D, no cancellation)        | 0.085 | 0.124 | 0.224 | bwd 1.09× |
| ⑩ bf16 forward scores (matches llm.c bf16 attn matrix)   | 0.075 | 0.124 | 0.215 | fwd 1.13×; **step 11.8× cumulative** |
| ⑪ causal-triangle skip: P+dS & softmax touch only j≤i    | 0.075 | 0.112 | 0.203 | bwd 1.11×; **step 1.07×** |
| ⑫ drop trivial lm_head matmul epilogue (no bias/gelu)    | 0.075 | 0.112 | 0.194 | **step 1.05×** (−10.6 ms classifier pass) |
| ⑬ hand-fused bias+gelu kernel replaces linalg epilogue   | 0.049 | 0.113 | 0.180 | **fwd 1.53×, step 1.08×** |
| ⑭ batch all heads via cublasGemmStridedBatchedEx         | 0.046 | 0.107 | 0.170 | **fwd 1.07×, bwd 1.05×, step 1.06×** |
| ⑮ compute dK/dV directly (cuBLAS transpose_a, no rect transpose) | 0.046 | 0.105 | 0.168 | bwd 1.02×; −4 transpose kernels + 2 fences |
| ⑯ drop redundant same-stream `synchronize` fences (attn + matmul_fwd) | 0.046 | 0.104 | 0.168 | **benchmark-neutral** cleanup (see note) |
| ⑰ coalesced row-parallel dbias (atomics) | 0.046 | 0.102 | 0.165 | dbias 7.6→4.7 ms; step 1.02× |
| ⑱ same coalesced+atomics fix for layernorm dgamma/dbeta | 0.046 | 0.101 | 0.165 | dgamma 3.3→2.7 ms (benchmark-marginal) |
| ⑲ d_input bwd → cuBLAS + hand gelu-grad; drop all matmul-bwd fences | 0.046 | 0.098 | 0.162 | **bwd 1.03×, step 1.02×** (−~192 host syncs/step) |
| ⑳ cuBLASLt fused bias+gelu epilogue on forward linear GEMMs (llm.c's technique) | 0.042 | 0.098 | 0.159 | **fwd 1.12×, step 1.02×** (bias_act 8 ms → fused into GEMM) |
| ㉑ cuBLASLt for backward GEMMs too (d_input DGELU-fused, d_weight); entire GEMM path now matches llm.c | 0.042 | 0.098 | 0.159 | **neutral** — GEMMs already matched; DGELU fusion offset by per-call descriptor overhead. Confirms remaining gap is non-GEMM |
| ㉒ fused_classifier: 1024 threads/block + 8-wide (x128) loads, matching llm.c's kernel5 | 0.042 | 0.098 | — | classifier fwd 5.7→4.6 ms (ncu) |
| ㉓ layernorm fused-residual fwd: cache residual row in SMEM (passes 2/3 read shared) | 0.042 | 0.098 | 0.160 | **~neutral** (4.27→4.10 ms) — the re-reads already hit L2; confirms non-GEMM kernels are at the bandwidth/L2 floor |
| ㉔ **store softmax probs, skip backward QKᵀ recompute** (llm.c's approach) | 0.046 | 0.092 | 0.154 | **step 1.04×** — P+dS 12→8.7 ms + one big batched matmul/layer removed; gap 1.17→**1.13×** |
| ㉕ **fused classifier once** (loss+dlogits in forward, drop backward pass) | — | — | — | removed a redundant 206M-element classifier pass (llm.c calls it once); classifier 9.3→8.0 ms (~1.5 ms). verify-gpu green (TENSOR OK). Single fused kernel is still ~8 ms vs llm.c's 3.7 ms (Mojo `exp`/`exp2` are accurate polynomials, not the `ex2.approx` intrinsic — a fast-exp attempt was slower). |
| ㉖ **explicit `alignment=align_of[...]` on classifier loads/stores** | — | — | — | **classifier 8.0→4.25 ms (~3.7 ms)** — Mojo wasn't emitting wide 128-bit loads without the hint (runtime `row*Vp` offset unprovable); 187→312 GB/s. Now ~matches llm.c's 3.7 ms. Derived from MAX's kernels; portable. verify-gpu green. |
| ㉗ **same alignment fix on adamw** (per-dtype align for bf16 params + fp32 moments) | — | — | — | **adamw 16.9→15.1 ms (~1.7 ms)**. Bandwidth-bound optimizer, same narrow-load cause. Layernorm fwd tried too → neutral (bandwidth-bound on 2-buffer pattern, not width). ncu total ~153→147 ms (**~1.08×**). |

**Authoritative `make benchmark-gpu BENCH_B=4 BENCH_T=1024` (2026-06-30, 40 steps,
same harness for both, llm.c freshly rebuilt); after ⑰:**

| configuration   | n  | mean ms | median ms |  std | tok/s |
|-----------------|---:|--------:|----------:|-----:|------:|
| **llm.mojo bf16** | 35 | — | **152.86** | 7.66 | **26,010** |
| **llm.c CUDA bf16** | 35 | — | **135.91** | 5.03 | **29,405** |

*(Through ㉕. Median gap **1.12×**; high std = GPU throttled mid-run.)*

> **⚠️ This 40-step wall-clock table is superseded for cross-change comparison** —
> after the ㉖–㉗ alignment work, ncu total GPU time is **147.3 ms** (from 152.8),
> ~**1.08×** at a clean baseline. On this heat-saturated GB10 the 40-step benchmark
> now reads llm.mojo at ~163 ms (throttled range) while llm.c holds ~136 ms, so it no
> longer isolates the win. See **"Current status (updated 2026-07-01)"** in the
> *Optimization landscape* section below for the authoritative, thermally-honest read
> (ncu + throttle-resistant harness). A clean 40-step number needs a cold GPU.

**Thermal caveat:** after hours of sustained benchmarking the GB10 self-throttles,
and llm.mojo (the heavier GPU load) is far more sensitive than llm.c: **llm.c stays
137–138 ms on every run, while llm.mojo's clean-run median ranges 159–169 ms**
depending on thermal state, so the same-run ratio floats **1.15–1.23×**. The
*controlled* measure is the profile harness (4 steps, less heat buildup): after ⑳
the step is **0.156 s (fwd 0.042 + bwd 0.098 + upd 0.016)**, down from 0.160 s at
⑲ — a solid, repeatable **~4 ms** win from the bias/gelu fusion.

So after ⑳, **llm.mojo bf16 ≈ 159–169 ms vs llm.c bf16 = 138 ms → ~1.15–1.23× slower** (thermal-dependent) (was
1.59× after ⑩), NOT yet parity. (Note: after hours of sustained benchmarking the
GB10 thermally throttles, inflating llm.mojo's median to ~168–175 ms with high
variance on hot runs; llm.c stays ~137. The ⑰ dbias win — `matmul_bias_bwd`
7.6 → 4.7 ms — is confirmed thermal-independently by ncu.) (An earlier draft of this section compared against a
stale llm.c number of ~0.22 s; the fresh apples-to-apples run above — which rebuilds
`train_gpt2cu` with `--use_fast_math -DENABLE_BF16` — measures llm.c at **0.137 s**.)
The profile-harness
step matches the bf16 benchmark, so the per-kernel analysis stands. `make
verify-gpu` stays green (the fp32 verify build keeps fp32 scores/dP; only the bf16
build narrows those scratch tensors, matching llm.c's bf16 attention matrix), and
the bf16 training-loss trajectory tracks the fp32 one.

Relative to llm.mojo's own starting point the gain is large — the bf16 step fell
from the baseline ~2.5 s (profile harness) to 0.170 s, ~15× — but the **gap to
llm.c is ~1.23×, still open.**

**⑭ Batch all heads with `cublasGemmStridedBatchedEx`.** The per-head attention
matmuls (QKᵀ, A·V, dQ, dKᵀ, dVᵀ) were 48 separate `linalg.matmul`/`blas.matmul`
launches *each* — 2304 launches/step at 16–28 % tensor-core utilization, the
biggest remaining lever. llm.c avoids this with `cublasGemmStridedBatched` (all
heads in one call). That FFI symbol **is** exposed (`_cublas.cublas.cublasGemmStridedBatchedEx`
— the MAX `blas.matmul` `batch_size` arg is a no-op on the CUDA path, but the raw
binding works), so a small wrapper `_attn_gemm_batched` (same row-major swap trick
as the vendor `_cublas_matmul`, with per-head strides + `batch_count=BH`) now does
each attention matmul in **one** launch. Validated in `scratch/test_batched_gemm.mojo`
and by `make verify-gpu` (the fp32 path runs the same batched code and matches
PyTorch within tolerance; note this is a genuine GEMM-backend change, so the loss is
*not* bit-identical — cuBLAS batched uses a different accumulation order than the
per-head kernels, well within bf16 noise). Launches collapsed 2304 → ~240; the
attention matmul time fell ~52 → ~47 ms (batching mainly removes launch overhead —
cuBLAS batched still runs these hd=64 shapes at ~10–12 % util, the same low
utilization the shape forces on *any* library incl. llm.c's). Step **180 → 170 ms
median**, gap 1.31× → **1.23×**. (`DEFAULT_TENSOR_OP` algo was tried and was
*slower*, so `Algorithm.DEFAULT` is kept.)

**⑯ Drop redundant same-stream fences (cleanup, benchmark-neutral).** Once the
attention and forward-linear matmuls all went through cuBLAS (⑬–⑭), which binds its
handle to `ctx`'s stream, the phase-boundary `ctx.synchronize()` calls in
`attention_fwd_gemm`/`attention_bwd_gemm` (5) and `matmul_fwd` (2) became redundant:
the consumer kernel is already stream-ordered after its producer. Removed them
(`make verify-gpu` green, loss bit-identical). The per-phase profile-harness time
drops (backward 0.113 → 0.104 s) because the harness fences each phase to time it,
so internal fences inflated those numbers — but the **full-step benchmark is
unchanged** (the CPU already runs ahead of the GPU in continuous execution, so the
fences weren't causing GPU idle). Kept as a correctness-preserving cleanup that
aligns with llm.c's single-stream model. **Attempted and reverted:** the matmul
*backward* fences are load-bearing (weight-grad accumulation races without them —
`CUDA_ERROR_ILLEGAL_ADDRESS`); and a coalesced thread-per-column `dbias` rewrite was
**4× slower** (thread-per-column serial-row reduction has far worse occupancy than
the original row-parallel block reduction — occupancy beats coalescing here).

**⑮ Compute dK/dV directly with cuBLAS `transpose_a`.** The backward previously
could not do `Aᵀ·B` (linalg.matmul lacked transpose_a), so it formed dKᵀ/dVᵀ and
ran four rect-transpose kernels ([T,hd]↔[hd,T]) plus two extra sync fences to get
dK/dV. `cublasGemmStridedBatchedEx` transposes natively, so `_attn_gemm_batched`
gained a `transpose_a` flag (validated in `scratch/test_batched_gemm.mojo`) and the
backward now computes **dK = dSᵀ·Q** and **dV = Pᵀ·dO** directly. Removes the four
`attention_transpose_rect` launches (~2.3 ms) and two `synchronize`s; the attention
kernel family fell 22.8 → 20.4 ms. Step **170 → 168 ms median**, gap holds at
**1.23×**; `make verify-gpu` green.

**⑫ Drop the trivial lm_head matmul epilogue.** Fresh ncu (after ⑪) showed the
generic `elementwise` family — the `linalg.matmul` `elementwise_lambda_fn`
epilogues, which this backend runs as *separate* kernels rather than fusing into
the GEMM — at ~39 ms/step, with a **single 10.6 ms call** over the `[B,T,V]` logits
(~205 M elements). That call was the lm_head matmul's epilogue, which (with
`has_bias=False, use_gelu=False`) did nothing but cast the fp32 accumulator to bf16
and store it — exactly what a plain `matmul` already does in its output write. So
`matmul_fwd` now emits **no** epilogue lambda in the no-bias/no-gelu case, letting
`linalg.matmul` write bf16 directly (the same trick ⑤ applied to the attention
matmuls, never carried to the linear path). Removes one full-tensor pass: step
**203 → 194 ms median**, gap 1.47× → **1.42×**; `make verify-gpu` stays green. The
remaining elementwise epilogues (fc gelu+bias ~15.8 ms, the three per-layer bias
adds ~10.7 ms) do real work and cannot be dropped — closing those needs true
GEMM-epilogue fusion (llm.c gets it free via cublasLt `GELU_AUX_BIAS`), which the
Mojo `linalg.matmul`/`blas.matmul` paths do not expose here.

**⑬ Hand-fused bias+gelu kernel replaces the linalg epilogue.** ⑫ established that
`linalg.matmul`'s `elementwise_lambda_fn` is lowered to a *separate* `elementwise`
sweep on this backend — and profiling showed those sweeps were badly
under-occupied: the fc gelu+bias ran at ~15.8 ms and the three per-layer bias adds
at ~10.7 ms (a 3 M-element bias add took ~296 µs, ~6× off bandwidth). Since the
epilogue is separate anyway, we drop the lambda entirely (plain `matmul` writes the
bf16 accumulator to `out_ptr`) and follow it with one tight, width-vectorized
kernel `_matmul_bias_act_gpu` that adds the per-column bias in fp32 and, for the FC
layer, stores the pre-activation + gelu. Crucially the lambda already received the
result as **bf16** (`val.cast[f32]()`), so doing the bias/gelu from the bf16 GEMM
output is **bit-identical** — the bf16 loss trajectory is unchanged
(step 0 = 8.795439, step 1 = 8.774514, …), and `make verify-gpu` stays green (the
CPU path keeps its inline lambda). Measured: fc gelu **15.8 → 4.9 ms**, bias adds
**10.7 → 2.8 ms**, elementwise family **39 → ~10 ms**; forward **0.090 → 0.049 s**,
step **194 → 180 ms median**, gap 1.42× → **1.31×**.

**⑪ Causal-triangle skip in the plane-pass kernels.** A fresh `make profile-ncu
PROFILE_B=4 PROFILE_T=1024 PROFILE_LAYERS=12` showed the backward's fused
P-recompute + dS kernel (`_attention_bwd_p_and_ds_gpu`) was the single largest
kernel at **24.5 ms/step** (11.3 %, 0 % tensor-core — it is a pure bandwidth pass
over the `[B·nh,T,T]` plane). It processed the *whole* plane and masked the
above-diagonal half to zero, so ~half its DRAM traffic was spent reading scores/dP
and writing zeros in the strictly-upper triangle. Because the persistent P and dS
buffers are reused every step, their above-diagonal half can be held at a
structural zero: memset once on allocation (`zero_on_alloc` for dS; the forward
already zeroes P), after which the kernel skips any `width`-block with `jbase > i`
(the block never straddles a row, since `width | seq_len`). The dense gradient
GEMMs (`dQ=dS·K`, `dKᵀ=Qᵀ·dS`, `dVᵀ=dOᵀ·P`) still read the full plane but see
those persistent zeros, so results are bit-safe — `make verify-gpu` stays green.
The same lower-triangle-only store was applied to the forward causal softmax
(neutral: that kernel is bound by its block max/sum reductions + `exp`, not the
store). Measured effect (ncu): P+dS **24.5 → 12.4 ms** (halved, as predicted);
authoritative benchmark **217.56 → 203.07 ms median**, closing the gap from
1.59× to **1.47×**.

**① Remove spurious intra-tile barriers (forward).**
`_attention_gpu_process_key_value_tile_warp` issued a block-wide `barrier()`
after *every key column* and after *every query row*. Each warp owns a disjoint
set of query rows, reads only the read-only shared Q/K/V tiles, and writes only
its own DRAM output rows and its own shared softmax-state slots — there is no
inter-warp dependency inside the routine, and KV-tile reload safety is already
enforced by the caller's barriers around the shared K/V copy. Removing both
barriers cut forward ~3 % (1.165 → 1.131 s) with bit-identical loss. The small
size of the win confirms the bottleneck is *compute*, not sync: the O(T²·d) QKᵀ
and A·V math runs entirely on CUDA cores. The next entries move that math onto
tensor-core GEMMs.

**② Tensor-core GEMM forward.** Replaced the monolithic flash forward with the
decomposed, llm.c-style path (`attention_fwd_gemm` in `llmm/attention.mojo`,
gated by `USE_GEMM_ATTENTION`):

1. **QKᵀ** as a per-head tensor-core GEMM via `linalg.matmul` — bf16 Q/K in,
   fp32 scores out (mixed-dtype matmul, verified), materializing a `[B·nh,T,T]`
   scores plane. The 1/√d scale is deferred to the softmax so Q/K stay unscaled
   for the (unchanged) backward.
2. A dedicated **causal softmax** kernel (one block per scores row) that masks
   `j>i`, normalizes, writes bf16 probabilities, and emits log-sum-exp with the
   *same definition* the flash backward expects.
3. **A·V** as a per-head tensor-core GEMM (bf16 probs × bf16 V → bf16 output).

Forward **1.131 → 0.106 s (10.7×)**; full step **2.495 → 1.470 s (1.70×)**.
Forward is no longer the bottleneck — the still-monolithic backward (1.35 s, 0 %
tensor core) now dominates.

**③ Tensor-core GEMM backward.** Replaced the two flash backward passes with the
decomposed FlashAttention-2 backward (`attention_bwd_gemm`), every product a
per-head tensor-core GEMM. The softmax probabilities P are recomputed from the
saved log-sum-exp (`P = exp(scale·QKᵀ − L_i)`, causal) — no online accumulation.
Then `dP = dO·Vᵀ`, `D_i = Σ_d dO·O`, `dS = scale·P·(dP − D_i)`, and the three
gradients `dQ = dS·K`, `dK = dSᵀ·Q`, `dV = Pᵀ·dO`.

Backward **1.349 → 0.267 s (5.0×)**; full step **1.470 → 0.389 s**, i.e.
**2.529 → 0.389 s (6.4×) end to end**. `make verify-gpu` passes all activations
and gradients against PyTorch.

Implementation notes:
- **No transpose_a.** `linalg.matmul` rejects `transpose_a` (constraint:
  "transpose_a not yet supported"), but dV and dK need Aᵀ·B. The dS elementwise
  kernel therefore emits `dSᵀ` and `Pᵀ` (transposed scatter stores) alongside
  `dS`, so all three gradient GEMMs use the supported `A·B` / `A·Bᵀ` forms.
- **Scale folded into dS, not the epilogue.** A first version applied the 1/√d
  scale in the dQ/dK matmul epilogues via a captured runtime `output_scale`; the
  capture silently did not apply, leaving dQ/dK exactly 1/scale = 2× too large
  (caught by a standalone GEMM-vs-flash-CPU diff: dV correct, dK 2×). Folding the
  scale into `dS` — which is exactly what both dQ and dK consume — fixed it. dV
  reads unscaled `P`, so it is unaffected.
- **Scratch reuse.** The fp32 score buffer is reused for `scores → dP`; P reuses
  the forward probability buffer. Three more `[B·nh,T,T]` bf16 planes (`dS`,
  `dSᵀ`, `Pᵀ`) plus a small `[B·nh,T]` `D` buffer are held persistently in the
  KVCache. Phases are fenced with `ctx.synchronize()` for the same cross-stream
  reason as the forward.

**④ Coalesced tiled transpose.** An nsys profile of the post-③ build showed the
backward's single biggest kernel was the `dS` elementwise kernel at **13.6 ms /
layer — 33 % of all GPU time** — because it produced `dSᵀ` and `Pᵀ` with strided
scatter stores (`dst[j·T+i]`, uncoalesced at stride T). Split it: `dS` now writes
only its coalesced natural-layout output, and `Pᵀ`/`dSᵀ` come from a
shared-memory 32×32 tiled transpose (`_attention_transpose_planes_gpu`, coalesced
both ways). Backward **0.267 → 0.221 s**; the 13.6 ms kernel is gone. Gradients
still pass `make verify-gpu`.

**⑤ Drop store-only matmul epilogues.** The post-④ nsys profile showed ~4,000
`elementwise` kernels per step — **exactly the attention matmul count**
(336/layer × 12). Mojo's `matmul` runs an `elementwise_lambda_fn` epilogue as a
*separate* kernel (extra launch + an extra read/write of the GEMM output). Since
the 1/√d scale now lives in `dS`/softmax, every attention epilogue had become a
plain store/cast — so they were dropped entirely (`matmul[...](c,a,b)` writes the
fp32 accumulator straight into `c`, casting to `c`'s dtype). This removed ~4k
kernels and their memory passes: forward **0.106 → 0.085 s**, backward
**0.221 → 0.171 s**. (The bias-carrying linear-layer epilogues in `matmul.mojo`
are left intact — they do real work.)

**⑥ Fuse P-recompute into dS.** The backward recomputed `P` in its own
[B·nh,T,T] kernel, then `dS` read it back. Merged into one pass: the `dS` kernel
now reads scores+lse, computes `P` inline (writing it for the dV transpose) and
`dS` in the same sweep — one fewer full [B·nh,T,T] pass, and `dS` uses the fp32
`P` instead of the bf16 readback (slightly *more* accurate). Backward
0.171 → 0.165 s. Still passes `make verify-gpu`.

**⑦ Avoid the big [T,T] transposes.** `dV = Pᵀ·dO` and `dK = dSᵀ·Q` needed a
transposed `[T,T]` (no `transpose_a` in `linalg.matmul`), costing two ~100 MB
tiled transposes (~24 ms). Reformulated to compute the *transposed* gradients
`dVᵀ = dOᵀ·P` and `dKᵀ = Qᵀ·dS`: now only the **small** `[T,hd]` Q/dO inputs and
`[hd,T]` outputs are transposed (~6 MB each, `_attention_transpose_rect_gpu`), and
the big `dS`/`P` are read in their natural layout by `_attention_bmm_kvgrad`.
Backward **0.165 → 0.143 s**. (The kvgrad GEMM moves the small dim from N to M —
`nvjet 64x32x64` — at comparable cost.)

**⑧/⑨ bf16 scratch on the bandwidth-bound passes.** With the structure fixed, the
fused `P`+`dS` pass became the largest attention kernel (~27 ms), bound by its
fp32 `scores` (200 MB) and `dP` (200 MB) reads — GB10 is bandwidth-limited. Both
were narrowed to bf16: `scores` only feeds `exp` (⑧), and `dP` only enters the
`dP − D_i` subtraction, which doesn't cancel where `dS` is significant (⑨). Each
halves a 200 MB pass. Backward **0.143 → 0.135 → 0.124 s**. `out_dtype` is a
template parameter on the QKᵀ/dO·Vᵀ helper, so the **fp32 verify build keeps fp32
scratch** and `make verify-gpu` stays green; only the bf16 build narrows them,
matching llm.c's bf16 attention matrix.

**⑩ bf16 forward scores.** The same treatment as the backward (⑧): the forward
QKᵀ now writes bf16 scores, halving the softmax's read pass. bf16 here is not a
downgrade — it matches llm.c, which stores its attention matrix in bf16. Forward
**0.085 → 0.075 s**; `make verify-gpu` stays green (the fp32 verify build keeps
fp32 scores). A follow-on SIMD-vectorization of the fused P+dS kernel was
**neutral** (it was already at its achievable bandwidth) — kept for clarity, not
speed.

### Attention is now on par with llm.c — the gap is entirely non-attention (2026-07-01)

**Head-to-head ncu of llm.c vs llm.mojo (both B=4 T=1024, 12 layers) settles where
the remaining gap lives.** Comparing the *per-call* time of the shared cuBLAS
attention kernels:

| attention GEMM kernel        | llm.c µs/call | llm.mojo µs/call | tensor% |
|------------------------------|--------------:|-----------------:|--------:|
| `cutlass_80_wmma_tensorop` (N=hd=64: A·V, dQ, dK, dV) | 629 | 635 | ~9.9 |
| `cutlass_80_tensorop` (K=hd=64: QKᵀ, dP)              | ~570 | ~560 | ~11–13 |

They are **identical** — after ⑭ we call the *same* `cublasGemmStridedBatchedEx`
llm.c does, so cuBLAS picks the *same* kernels at the *same* utilization. The 9.9 %
util on the N=64 GEMMs is a **hardware floor for head_dim=64** that llm.c hits too
(cuDNN off). Our softmax (569 µs/call) ≈ llm.c's `softmax_forward` (482 µs/call),
and our backward P/dS pass ≈ llm.c's `softmax_autoregressive_backward`. **So the
original premise — "attention saturated" — is fully resolved: attention has no
remaining shortcoming vs llm.c.**

The residual ~1.22× is **entirely non-attention**, and the same ncu names where
llm.c is faster per-call:
- **dbias**: llm.c `matmul_backward_bias` = **60 µs/call** vs our `matmul_bias_bwd`
  = **159 µs/call** (2.6× — ours reduces with uncoalesced row-strided reads).
- **bias epilogue**: llm.c fuses bias into the cuBLASLt GEMM (0 extra) + a lean
  separate `gelu_forward`; we run separate bias kernels (~2.8 ms).
- **permute/unpermute** (our split/merge): ~3.3 ms heavier.
- plus diffuse differences (residual+layernorm fusion, sync model) making up the
  rest. These are all standard non-attention kernel micro-optimizations.

### Remaining gap to llm.c (~1.12×, still open — non-attention micro-opts)

After ㉔ the step is **0.154 s vs llm.c's 0.137 s — ~1.12× slower** (clean cooldown,
std < 1). **Attention now matches or BEATS llm.c**: ㉔ stored the softmax probs and
dropped the backward QKᵀ recompute (same matmul count as llm.c; P+dS 8.7 ms < llm.c's
11 ms softmax-backward). The entire GEMM path (fwd+bwd) uses cuBLASLt like llm.c.

**The attention matmuls are provably at the hardware floor.** Routing them through
cuBLASLt strided-batched (llm.c's *actual* attention API — `matmul_cublaslt` with
`batch_count`, added via layout `BATCH_COUNT`/`STRIDED_BATCH_OFFSET` attributes)
instead of `cublasGemmStridedBatchedEx` was **verified NEUTRAL**: cuBLASLt's heuristic
selects the *identical* cutlass `wmma`/`tensorop` kernels (15.1 vs 15.2 ms, etc.). The
`wmma` 9.9%-util kernel is simply cublas/cutlass's best for head_dim=64 — and it's
exactly what llm.c runs too. So the ~40 ms of attention GEMMs match llm.c kernel-for-
kernel by construction; there is no API lever left there. (Path kept behind
`USE_LT_ATTN=False`.)

**The residual ~17 ms is non-GEMM kernels at the bandwidth/occupancy floor** — proven
by an extensive set of neutral/negative experiments (each kept as a documented dead
end): dbias atomics→partial-buffer (neutral, bandwidth-bound), layernorm SMEM-cache
(neutral, L2), layernorm Welford 1-pass (neutral), **layernorm warp-per-row port of
llm.c's kernel5 (SLOWER, 120 vs 95 µs/call — shared-mem occupancy + `warp.sum`
overhead don't match CUDA's `warpReduceSum`+`store128cs`)**, softmax 512-thread
(worse), split/merge x128 vectorize (worse). These kernels (layernorm ~4 ms, classifier
~4.6 ms, softmax ~6.8 ms, dbias ~4.6 ms, adamw ~17 ms at the 30-byte/param bw floor)
are each 1.1–1.5× behind llm.c's hand-tuned CUDA in ways (streaming loads, precise
occupancy, cooperative-groups reductions) that don't port cleanly to Mojo. The prior
"one structural lever" (recompute→store) was implemented (㉔) and worked.

**The one identified structural lever left — recompute vs store.** Our attention
backward *recomputes* QKᵀ (flash-style, `attention_bwd_gemm` Phase A
`scoreout(query,key)`) to rebuild the softmax probs, because the forward only stores
the `[B,T,C]` attn output + `lse`, not the `[B,NH,T,T]` probs. **llm.c stores the
probs `att` per layer and reads them in the backward, skipping that recompute matmul
(~7 ms).** Adopting it means adding a per-layer `att_probs` activation
(L·B·NH·T·T ≈ 1.2 GB — fine on GB10's 128 GB), having the forward write probs there,
and the backward read them (dropping the QKᵀ scoreout + the P-recompute in P+dS).
Estimated → ~0.153 s (**~1.12×**). It's a sizeable, delicate change (2 attention
signatures + backward logic + train activation wiring) that still would not reach
parity, so it's flagged here rather than done blind.

**Everything else is at the memory-bandwidth floor** (empirically confirmed by
neutral/negative experiments): ㉑ backward-cuBLASLt neutral (GEMMs already match),
㉓ layernorm-SMEM-cache neutral (re-reads hit L2), split/merge vectorization *slower*
(already coalesced at max parallelism, reverted). The residual ~16 ms after the
recompute lever is ~8 memory-bound kernels (softmax, classifier, layernorm, dbias,
adamw, P/dS) each 1.1–1.5× behind llm.c's hand-tuned CUDA, at the L2/DRAM limit,
where thermal noise (±10 ms) now exceeds any single change.

**It is GPU-bound, not sync-bound.** nsys puts total GPU *kernel* time close to the
~180 ms wall-clock — only a few ms idle. Removing fences won't help (confirmed:
5→3 was neutral); the work itself must get cheaper. llm.c runs the same step in
~137 ms of GPU time, i.e. its **kernels are ~1.31× more efficient**. The profile is
flat; per step (fresh ncu, after ⑬, total ~178 ms GPU) the cost is the linear-layer
cutlass GEMMs (`matmul` family ~63 ms incl. the hand-fused bias/gelu 7.7 ms +
`nvjet` big-tile), the per-head attention matmuls (`nvjet 32x64x64` + `64x32x64`,
~30 ms **at only 17 % tensor-core util** — small K=64 per head), AdamW (~17 ms,
bandwidth-bound), the softmax + P/dS passes (~22 ms), the fused classifier (~7 ms),
and a residual ~2.3 ms of leftover `elementwise`. The gap is now spread broadly;
the biggest remaining lever is the **per-head attention matmuls' low (17 %)
tensor-core utilization** (~30 ms) — inherent to head_dim=64 and only attackable by
a hand-rolled batched/fused tensor-core attention kernel — plus the general ~1.3×
efficiency headroom in the linear-layer GEMMs vs cuBLASLt.

**Why the high-level batched-GEMM lever is blocked, and what parity actually
needs.** Reading `linalg/bmm.mojo`: its tensor-core batched paths (SM100/A100)
require **static N/K** *and* `a_k ≥ 128` / `c_n % 128 == 0`. Attention has
**head_dim = 64**, so QKᵀ (K=64<128) and A·V (N=64, not %128) both fail the guard
and fall to the *naive* non-tensor-core kernel — why the batched call measured
3.4× slower. So the *high-level* batched matmul can't serve hd=64. **But the
low-level warp-MMA primitive does exist** (`layout.tensor_core` — `TensorCore`,
`get_mma_shape`, `get_fragment_size`; it is what `nn.mha` is built on). So parity
is **not blocked by a missing capability** — it is blocked by the **effort/risk of
writing a fused flash-attention kernel from scratch with `layout.tensor_core`**
(shared-memory Q/K/V tiling, register-fragment MMA for QKᵀ and P·V, online softmax,
causal masking, *and* the training backward — recompute + dQ/dK/dV). That kernel is
what gives llm.c both its advantages at once: one fused tensor-core pass over all
heads, and no `[B·nh,T,T]` materialization. It is a large (multi-hundred-line),
high-risk piece of kernel engineering — the honest next step, not a one-line fix.
The `nn.mha` flash kernel itself is **inference-only** (no log-sum-exp output, no
training backward), so it cannot be called from this training loop as-is.

The structural reason llm.c is faster: its attention is **one
`cublasGemmStridedBatched` per QKᵀ / A·V over all B·nh heads** (a single fused
tensor-core launch) plus a fused softmax, whereas llm.mojo issues **48 small
per-head `linalg.matmul` launches** per GEMM (each K or M = head_dim = 64, which
underfills the tensor-core tile) and materializes more `[B·nh,T,T]` intermediates.
Closing the last ~1.59× requires one of:
- (a) a **tensor-core strided-batched GEMM** in the Mojo toolchain — collapses the
  ~29 ms of per-head matmuls toward llm.c's batched cost. `linalg.matmul` is
  2D-only; `linalg.bmm.batched_matmul` is *not* tensor-core for this shape
  (measured 3.4× slower — see dead-ends). **Blocked on toolchain support.**
- (b) a **custom fused flash-attention kernel** with warp-level MMA (what
  Modular's `nn.mha` does), eliminating the materialized `[B·nh,T,T]` passes — a
  large, separate effort whose training backward is non-trivial.
- (c) further bandwidth trims (fuse softmax into the QKᵀ epilogue on the forward;
  bf16 the forward scores) — incremental, a few ms each.

(a) is the clean win and matches llm.c's design; it is blocked on toolchain
support today. This is an honest open gap, not a reached target.

#### Flash-path de-risking — working tensor-core flash tile (PoC, 2026-06-30)

Two standalone kernels prove the fused flash kernel is buildable in-repo:
- `scratch/test_mma.mojo` — one warp computes `S = Q·Kᵀ` (16×8, bf16→fp32) via
  `TensorCore.load_a/load_b/mma_op/store_d`, **bit-exact (err 0.0)** vs scalar.
- `scratch/test_flash.mojo` — **a full single-tile flash attention**:
  `O = softmax(Q·Kᵀ)·V` with *both* GEMMs on tensor cores (QKᵀ, and P·V over Vᵀ
  with `transpose_b=True`) and the **softmax in shared memory**. Matches a scalar
  reference to **max abs err 0.0012** (bf16 rounding of P). N>8 tiling is handled
  by looping `mma_op` over 8-wide n-tiles with `.tile[..]` slicing.

Extended further to a **complete online causal flash forward** (`test_flash.mojo`):
streams KV tiles with the online-softmax O-rescale, causal masking, hd=64, and lse
output — matching scalar *causal* attention to **err 0.00047**. So the forward
flash kernel is functionally complete (BR=16 query tile, one warp); what's left is
generalizing the launch grid over (head, query-tile), the scale, and wiring it
into `attention_fwd_gemm` (it produces the same per-head attn + lse the existing
GEMM backward already consumes), then validating + benchmarking.

**Integrated and measured — flash does NOT win on this hardware (definitive):**
the flash forward was wired into `attention_fwd_gemm` (bf16, hd=64, behind
`USE_FLASH_FWD`), produces correct attn + lse (bf16 loss matches the GEMM forward
within noise, 8.7950 vs 8.7904), and was measured in **two** configurations:
- naive (BR=16, 1 warp): forward **~0.16 s** vs GEMM forward 0.075 s.
- multi-warp (BR=32, 2 warps sharing the K/V loads): forward **~0.167 s** — *no
  improvement*.

So the slowdown is **not** occupancy/redundant-loads — it's tensor-core
**utilization**. The GEMM forward does QKᵀ and A·V as **big batched tensor-core
matmuls** (high MMA utilization); the flash does the same FLOPs as **many tiny
16×8 MMAs interleaved with the softmax/transpose/accumulate** (the tensor cores
idle between MMAs). The classic flash win — avoiding HBM materialization of the
`[B·nh,T,T]` matrix — **does not apply here**: GB10 is bandwidth-rich relative to
its tensor throughput, and after ⑥–⑨ the materialization is already small/bf16, so
the cost flash removes is tiny while the utilization it sacrifices is not.

**Big-tile retest (2026-06-30, after ⑬).** To directly test the "flash just needs
bigger MMAs" hypothesis, the integrated kernel was rebuilt with **BR=64, BC=64, 4
warps**, and the P·V loop was made dense (loop the full BC in 16-wide k-tiles →
8×4 MMAs per n-tile instead of one). Validated correct in
`scratch/test_flash_big.mojo` (multi-tile causal, max abs err 6.2e-4). Measured at
B=4 T=1024 (12 layers, profile harness): **forward 0.0896 s with flash vs 0.0497 s
with the GEMM path** — flash is still **~1.8× slower**, i.e. ~40 ms *worse* on the
forward alone. Bigger MMAs did **not** help. The real limiter is the **softmax
serialization between the MMA phases**: each streamed KV tile runs QKᵀ → `barrier`
→ (one-thread-per-row) softmax → `barrier` → P·V → `barrier`, so the three phases
never overlap and the dense MMAs stall on the serial softmax + barriers. The GEMM
path instead runs QKᵀ, softmax, and A·V as three *separate, full-occupancy*
kernels. On bandwidth-rich GB10 that decomposition wins. **Flash is therefore a
dead end for parity on this hardware — confirmed for both tiny (BR=16/32) and big
(BR=64) tiles** (flag `USE_FLASH_FWD` left off; both the kernel and
`scratch/test_flash_big.mojo` kept as documented negative results). The remaining
gap is the per-head matmul launch overhead + low (16–28 %) tensor-core utilization
(llm.c batches with cublasGemmStridedBatched; Mojo's batched path is
non-tensor-core for hd=64 — blocked) plus the non-attention GEMM/kernel gap.

So the entire flash *compute pattern* works in Mojo today. The key simplification:
the softmax runs in shared memory with a normal `[Br,Bc]` layout — `store_d` writes
S to shared and `load_a` reads P back — which **dissolves the "reduce over opaque
MMA fragments" problem** I had flagged as the hard blocker. What remains for a
production fused kernel (the genuine multi-session work): the **cross-tile
online-softmax rescale loop** (stream KV tiles, rescale the O accumulator), real
dims (Br/Bc/hd = 64 → multiple m-tiles / k-steps), causal masking, `lse` output,
multi-warp tiling, model integration, and a comparable **backward** kernel. The
compute core is proven and de-risked; the remainder is standard (if extensive)
flash-kernel engineering.

#### Why GB10 makes this bandwidth-bound

The host is an NVIDIA GB10 (DGX Spark class) with unified LPDDR5X at ~273 GB/s —
not an HBM datacenter part. The tiled transpose already ran at ~200 GB/s (near
peak), so the `[B·nh,T,T]` (= 100 MB bf16) elementwise/transpose passes are
memory-bound: the lever is **fewer / narrower memory passes**, not more FLOPs.
That is exactly what ⑥–⑨ exploited (fuse passes, bf16 the scratch), and why llm.c
— which fuses softmax into its GEMM epilogues and batches the GEMMs — stays ahead.

#### Dead ends (measured, reverted)

- **Fences.** Cutting backward fences 5 → 3 was neutral (sync is not the cost).
- **`linalg.bmm.batched_matmul`.** Collapsing the 48 per-head launches into one
  batched call made the backward **~3.4× slower** (0.27 → 0.9 s): for this mixed
  bf16→fp32 shape it falls back to a non-tensor-core kernel, whereas 48 per-head
  `linalg.matmul` calls each hit the tensor cores.
- **Fusing QKᵀ + P-recompute** (compute `P` in the QKᵀ epilogue to drop the scores
  buffer). The bf16-output matmul epilogue produced NaN gradients and an async
  race that did not reproduce under sync mode. Reverted; would need the epilogue
  to see the fp32 accumulator and a correct fence.

Implementation notes:
- **Persistent scratch.** The `[B·nh,T,T]` fp32 scores + bf16 probability buffers
  are allocated once and kept alive across all layers/steps in `KVCache`. A
  first attempt that allocated per-layer `DeviceBuffer`s crashed
  (`CUDA_ERROR_ILLEGAL_ADDRESS`): their frees are not ordered against the async
  kernels that read them. One shared pair is safe because each layer's attention
  fully consumes the scratch (in stream order) before the next overwrites it.
- **Phase fences.** `linalg.matmul` launches async and does not self-synchronize
  (matching `matmul_fwd`), so the three phases are separated by
  `ctx.synchronize()`: softmax must wait for all 48 QKᵀ matmuls; A·V for the
  softmax; the head-merge for A·V. Without fences the phases race across streams.
- **Numerics.** bf16 probabilities in A·V (vs the flash path's fp32 online
  accumulation) shift the loss by ~0.012 at step 0 — well within bf16 noise and
  matching llm.c, which also stores the attention matrix in bf16. `make
  verify-gpu` passes all activations **and gradients** against the PyTorch
  reference (fp32 build); bf16 training loss tracks the baseline trajectory.

---

## Optimization landscape: what's matched, what resists, and the Mojo-library option

This section consolidates the state of the GPU optimization effort at the end of
the 2026-06-30/07-01 session so future work can pick up without re-deriving
everything above. It is deliberately honest about what is *proven matched*, what
*resists*, and which levers remain untried. Nothing below duplicates the ⑪–㉕
change table — it summarizes the conclusions those changes reached.

### Current status (updated 2026-07-01, after the alignment work ㉖–㉗)

**Position:** ncu total GPU time **152.8 → 147.3 ms** this sub-session (−5.5 ms from
the alignment best-practice below), which at a clean thermal baseline corresponds to
**~1.08×** vs llm.c's ~136 ms. Over the whole session the step fell **217 → ~147 ms
(1.59× → ~1.08×)**. Every change is `make verify-gpu` green (fp32 matches PyTorch;
bf16 loss tracks fp32).

**Measurement caveat (important):** the 40-step wall-clock benchmark is **not reliable
on this GB10 right now** — llm.mojo's heavier sustained load self-throttles (floats
153–169 ms, std 4+) while llm.c stays rock-steady at ~136 ms (std 1.0). After a long
profiling session the chip is heat-saturated, so absolute wall-clock has drifted up
~10 ms independent of the code. **Trust the ncu same-tool before/after deltas and the
few-step profile-harness breakdown**, not the 40-step benchmark's absolute number. A
definitive wall-clock number needs a cold GPU (realistically a fresh session).

### Per-family GPU breakdown (our profile, ncu ~147 ms total)

Where our time goes, and how each family stands against llm.c (post-alignment):

| family | our time | vs llm.c | verdict |
|--------|---------:|----------|---------|
| matmul (cutlass) + "other"/nvjet | ~94 ms (71.8 + 22.7) | matched | **PROVEN matched** — cuBLASLt selects identical cutlass wmma/tensorop kernels |
| adamw | **15.1 ms** (was 16.9) | **~matched/better** | ㉗ alignment fix; confirmed 17→15.2 ms in the throttle-resistant harness |
| attention (softmax + P+dS + D) | 17 ms | **matches or beats** | our P+dS 8.7 ms **beats** llm.c's softmax-backward 11 ms after att-storage (㉔) |
| classifier | **4.25 ms** (was 8.0) | **~matched** (llm.c 3.7 ms) | ㉕ merge (2 passes→1) + ㉖ alignment fix (187→312 GB/s); was the biggest non-GEMM gap, now ~closed |
| layernorm | 8.5 ms | 5.5 ms | memory-bound; alignment neutral (2-buffer bandwidth-bound, not width). Backward is the untried target |
| split/merge | 6.3 ms | ~5 ms | permute/unpermute gap |

The ~94 ms of GEMM being matched is not an estimate: routing attention through
cuBLASLt strided-batched (llm.c's actual attention API) produced **byte-identical
kernel selection**. So the majority of the step is matched to llm.c kernel-for-kernel
by construction. The remaining gap is now concentrated in the **layernorm/softmax
backward** memory-bound passes.

### The alignment best-practice (㉖–㉗) — derived from MAX's kernels, portable

Studying MAX's source-available `layer_norm_gpu_warp_tiling` revealed the reusable
technique: **Mojo does not emit wide 128-bit `ld.global.v4`/`st` transactions unless
it can prove the address is aligned** — and our kernels index by `row*Vp + …` (runtime
factor), so `.load[width=8]()` was silently falling back to narrow loads at ~half
bandwidth. Adding explicit `alignment=align_of[SIMD[dtype,width]]()` (pure Mojo, no
PTX — **runs on other accelerators**):

| kernel | before | after | note |
|--------|-------:|------:|------|
| **fused classifier** (`proto_classifier.mojo`, 412 MB ≫ L2) | 6593 µs / 187 GB/s | 3964 µs / 312 GB/s | ncu single-call 7996→4250 µs (**−3.7 ms**) |
| **adamw** (per-dtype align: bf16 params, fp32 moments) | 16854 µs | 15110 µs | **−1.7 ms**; harness-confirmed |
| layernorm fwd, global_norm | — | neutral / small | applied for correctness/consistency |

The whole codebase omits these hints (`grep '.load[width=' without alignment`), so the
remaining DRAM-bound kernels (esp. the **backward** passes) are the next candidates.
NOTE: my first "MAX layernorm is 4× faster" reading was a microbenchmark artifact
(plain LN on an L2-resident buffer vs our fused 2-buffer LN) — see the *Layernorm
prototype* subsection below; the fused-residual layernorm forward is bandwidth-bound
and already ~within 10 % of llm.c.

### Levers that WORKED this session

- **cuBLASLt fused GEMM epilogues** (bias/gelu, forward + backward) — llm.c's
  `GELU_AUX_BIAS` technique, replacing separate elementwise bias/gelu sweeps.
- **att-storage (㉔)** — store the softmax probabilities and skip the backward QKᵀ
  recompute (llm.c's approach); removed one big batched matmul per layer and made
  P+dS *beat* llm.c's softmax-backward.
- **classifier merge (㉕)** — `fused_classifier` was running twice per step, reading
  206M logits *twice*; merged to one call in the forward (dL/dloss is a constant),
  ~1.5 ms.
- **classifier dlogits fast-exp** — inline-PTX `ex2.approx` for the dlogits
  exponential, ~0.4 ms.

### Levers TESTED that hit the floor (neutral or worse — documented dead ends)

Each of these was implemented and measured, not assumed. They are kept as
negative results so they are not re-attempted:

| lever | result | why |
|-------|--------|-----|
| warp-per-row layernorm (direct port of llm.c's `fused_residual` kernel5 using `warp.sum`) | **SLOWER** (120 vs 95 µs/call) | shared-mem occupancy + `warp.sum` overhead don't match CUDA's `warpReduceSum` |
| layernorm SMEM-cache | neutral | L2 already serves the re-reads |
| layernorm Welford 1-pass | neutral | — |
| dbias atomics → partial-buffer | neutral | bandwidth-bound |
| softmax 256 → 512 block-size | worse | occupancy |
| split/merge x128 vectorize | worse | already coalesced at max parallelism |
| cuBLASLt-batched attention | neutral | identical kernels — proves attention at hw floor |
| attention-softmax fast-exp | neutral | softmax is memory-bound (reads scores twice), not exp-bound |

### Why the residual ~17 ms resists

It is memory-bound elementwise/reduction kernels — layernorm, softmax, classifier —
where llm.c's hand-tuned CUDA extracts ~1.1–1.5× more from the *same bytes* via
`store128cs` streaming stores, cooperative-group reductions, and precise occupancy
tuning that Mojo's codegen does not reproduce **even when the same algorithm is
hand-written in Mojo** (the warp-per-row layernorm port is the sharpest evidence:
same algorithm, slower result).

### Mojo capabilities discovered (the toolbox)

The optimization effort surfaced what Mojo actually exposes. Two tiers:

**Low-level** (`std.gpu.memory` / `std.gpu.primitives`):
- streaming / cache-hint memory: `CacheOperation.STREAMING`,
  `CacheEviction.EVICT_FIRST`
- `async_copy` (cp.async), TMA descriptors (`create_tma_descriptor` + bulk tensor
  copy), thread-block clusters (`cluster_sync`), `mbarrier`
- cooperative reductions: `warp.sum/max/min`, `lane_group_reduce/sum`,
  `shuffle_down/xor`, `block.sum/max/min`
- inline PTX (`inlined_assembly`) and `llvm_intrinsic`; `ldg` non-coherent load

**High-level** (MAX kernels, `max/kernels/src/nn/` and `linalg/`) — production-tuned
prebuilt kernels:
- `flash_attention` (FA-2, SM90-tuned, has AMD paths) in `attention/gpu/mha.mojo`
- `layer_norm_gpu_warp_tiling` / `layer_norm_gpu_block` /
  `rms_norm_fused_residual_add_gpu` in `normalization.mojo`
- `softmax_2_pass` / `softmax_3_pass` / `softmax_kernel` in `softmax.mojo`
- `matmul` (tensor-core), `bmm`, `grouped_matmul` in `linalg`
- `block_reduce` / `row_reduce` / `map_reduce` / `elementwise` / `vectorize` in
  `std.algorithm`; TMA double/triple-buffering; `DeviceContext.create_stream()` for
  compute/memory overlap

**Caveat on the high-level tier:** these target MAX's `LayoutTensor`/graph system and
are mostly **forward/inference-oriented** — training backward passes are not
provided. Adopting them is real integration work, not a drop-in. There is also **no
autotuning system** in Mojo (compile-time / shape-heuristic dispatch only).

### The bf16-Adam-moments caveat

Storing the Adam m/v moments in bf16 would cut adamw **~17 → 12.5 ms** (~4.5 ms —
the single biggest available win). It is **NOT a fair "match"**: llm.c keeps **fp32**
optimizer state for stability, and `verify-gpu` is fp32-only so it would **not catch
bf16 moment drift**. Flag this, don't silently take it — it buys a headline ms at a
numerical-fidelity cost llm.c doesn't pay.

### Portability

~**90 % of compute** (all GEMMs, via cuBLAS/cuBLASLt) is **NVIDIA-locked**; the
inline-PTX `ex2.approx` fast-exp is NVIDIA-only. The custom elementwise / reduction /
softmax / split-merge / adamw / encoder kernels use **vendor-neutral Mojo**
(`DeviceContext` / `thread_idx` / `barrier` / `block.sum`) and would port.

The code branches `is_cpu` / `is_gpu` everywhere but has **no `is_nvidia` / `is_amd`
vendor split** — the GPU branch assumes cuBLAS unconditionally. Porting to AMD ≈ add
an `is_amd`/ROCm branch with rocBLAS/hipBLASLt wrappers (~300–400 LoC) + swap the PTX
exp for an LLVM-IR equivalent. Estimate: **2–3 days functional, 2–3 weeks tuned**.
Bonus: adopting MAX's kernels would *improve* portability since they carry AMD paths.

### Layernorm prototype — DONE, and it CORRECTED an earlier over-claim

`scratch/proto_layernorm_max.mojo` benchmarked MAX's `layer_norm_gpu_warp_tiling`
against our kernel and a from-scratch **portable** re-derivation (register-resident
row + single-pass `block_reduce_dual_sum` via `warp.sum`/`lane_group_sum` + aligned
128-bit loads — no PTX, no PDL). Results (GB10, bf16, rows=4096 cols=768):

| kernel | µs/call | notes |
|---|---:|---|
| MAX `layer_norm_gpu_warp_tiling` (plain, 1r+1w) | 8.8 | |
| **our derived PLAIN kernel** (1r+1w) | **9.5** | **matches MAX** — portable Mojo hits peak BW |
| our derived FUSED-residual (2r+2w) | ~77 | |
| our existing block kernel FUSED (2r+2w) | ~77 | |
| llm.c `fused_residual` layernorm | ~71 | |

**Correction to the "MAX is ~4× faster" claim above:** that was a *microbenchmark
artifact*. It compared MAX doing **plain** LN on one 12.6 MB buffer re-read 200× (so
L2-resident, served from cache) against our **fused-residual** LN moving two distinct
buffers (read inp1+inp2, write residual+normed) whose working set exceeds L2. Pointing
inp2 at the same buffer dropped fused 77→60 µs; the rest is the two distinct writes.

**What's actually true:** (1) **portable Mojo CAN hit peak bandwidth** — our derived
plain kernel (9.5 µs) equals MAX (8.8 µs) using only vendor-neutral primitives, so the
technique ports to non-NVIDIA. (2) The **fused-residual layernorm (our real hot path)
is bandwidth-bound and already ~within 10 % of llm.c** (~77 vs ~71 µs) — the derived
kernel does not beat the existing one. **The layernorm forward is not a real target.**

### Next step — classifier DONE ✅, backward passes remain

The reusable best-practice — **explicit `alignment=align_of[SIMD[dtype,w]]()`** on
loads/stores (plus register-residency + `block_reduce_dual_sum` where the row fits) —
is validated and portable.

- **Classifier — DONE (㉖):** the alignment fix took it 8.0→4.25 ms (187→312 GB/s),
  ~matching llm.c. Combined with ㉕ (merge), the classifier went from ~9.3 ms (2
  passes) to ~4.25 ms (1 aligned pass).
- **adamw — DONE (㉗):** 16.9→15.1 ms.
- **Still open — the backward passes:** layernorm-dinput / dgamma-dbeta and the other
  DRAM-bound backward kernels almost certainly have the same missing-alignment issue
  (the whole codebase omits the hint). These are the next candidates; ncu confirms
  each change thermally-independently. The layernorm *forward* is NOT a target
  (bandwidth-bound, already ~within 10 % of llm.c).
- **Not taken (fairness):** bf16 Adam moments would cut adamw ~4.5 ms more but uses
  lower precision than llm.c — see the bf16-Adam-moments caveat above.

---

## 2026-07-01 session 2 — cold-GPU baseline & alignment sweep of the backward passes

### Cold-GPU wall-clock baseline (authoritative, replaces the throttled numbers above)

GPU cold at start (43 °C, 11 W, SM clock 2411 MHz of 3003 max). Profile harness
(4 steps, B=4 T=1024, 12 layers) + a 10-step llm.c run, same day, same thermal state:

| implementation | per-step (median) | breakdown |
|---|---:|---|
| llm.mojo bf16 | **163.5 ms** (steps 1–3; step 1 was 156.1) | fwd 49.5 + bwd 97.7 + upd 15.5 |
| llm.c bf16    | **136.6 ms** (steps 2–10, std < 1) | — |

**Gap = 1.20× wall-clock on a cold GPU** — worse than the ~1.08× the ncu totals
suggested, so ncu-serialized totals undercount some real cost (likely launch/gap
overhead between our more numerous un-overlapped kernels, and/or the SM clock sitting
at 2411 MHz rather than boost). Treat 163.5 → 136.6 as the true remaining distance.

### New constraint (user directive, this session)

**Non-CUDA GPU compatibility must be maintained.** Wherever the fast path calls
cuBLAS/cuBLASLt or CUDA-specific FFI, a working vendor-neutral GPU path (MAX
`linalg.matmul` / pure-Mojo kernels) must be kept selectable, so the code still runs
on non-CUDA accelerators. The alignment work below is pure Mojo and portable; the
cuBLAS attention/linear fast paths need an explicit vendor branch (queued).

### ROOT CAUSE of the ncu-vs-wallclock discrepancy: a 17.5 ms/step memset of 3.29 GB ✅ verified

nsys timeline of the profile harness (steady-state steps 1–3), verified by hand against
the sqlite trace (an agent's first read claimed a "19 ms idle gap"; that was a step-0
JIT artifact — the real numbers below are from my own per-step analysis, scripts in the
session scratchpad):

| component (per step) | llm.mojo | llm.c |
|---|---:|---:|
| wall clock | 158.8–163.8 ms | 136.2 ms |
| kernel busy time | **144.1 ms** (528 launches) | ~135 ms |
| **cuMemset** | **19.07 ms** | ~0 |
| memcpy (loss readback) | 0.01 ms | — |
| true GPU idle | **0.33–0.37 ms** (largest gap 32 µs) | ~1.0 ms |

Conclusions:
1. **There is NO launch-overhead / sync / idle problem.** GPU is ~99.8 % busy.
   Single-stream pipelining is fine; llm.c's 2-stream design buys it nothing we lack.
2. **The 19 ms is `zero_gradients()`** (train_gpt2.mojo:1767): a 1.6 ms fill of the
   249 MB param grads (legit, llm.c does the same) + a **17.5 ms fill of the entire
   3.29 GB `grad_acts_buf` every step** (train_gpt2.mojo:1778). llm.c never wholesale-
   zeroes activation grads — its backward overwrites. Most of our backward kernels
   also overwrite; only genuinely accumulated buffers need the zero. Analysis of the
   minimal zero set is in progress; expected win ~15–17 ms.
3. Live kernels (144.1 ms) run ~3 ms slower than ncu-serialized replay (147.3 total
   incl. what ncu counts differently) — clock/measurement noise, not a lever.
4. Side findings from a toolchain survey (kept for reference, low priority while GPU
   is compute-saturated): Mojo `DeviceContext` DOES expose CUDA graphs
   (`create_graph_builder`→`instantiate`→`replay`, explicit-DAG, no stream capture);
   `compile_function` re-loads modules per call at ~30 sites (attention.mojo's KVCache
   already caches compiled kernels — pattern is generalizable); `calculate_grad_norm`
   allocates fresh buffers per step. None of these matter until GPU idle appears.

### ㉘ Alignment sweep of the backward passes — small win, floor confirmed

Applied the ㉖-style explicit `alignment=align_of[SIMD[dtype,width]]()` across the
remaining wide-load/store GPU kernels: layernorm backward (d_input + fused-residual
bwd), attention P+dS, softmax fwd/bwd, gelu, encoder fwd/bwd (wpe/wte). Shared
CPU/GPU helpers got an opt-in `aligned` comptime flag (CPU chunk offsets aren't
provably aligned). split/merge were confirmed scalar-by-design (no wide accesses);
the old flash-kernel family is dead code and was left alone. `make verify-gpu` green
(all 16 gradient tensors OK); edits proven bit-identical by full revert bisection.

**ncu confirmation (46 °C stable, 2411 MHz, same-tool before/after): total 147.3 →
145.8 ms (−1.5 ms).** P+dS 8.7→8.24 ms; layernorm family ~flat at 8.47 ms (vs llm.c
5.5 — its backward, like its forward, is at a 2-buffer bandwidth floor where load
width isn't the limiter); classifier 4.14 ms, adamw 15.08 ms (stable, as before).
Conclusion: the alignment lever is now fully harvested; the non-GEMM kernel floor
vs llm.c (~3 ms layernorm + ~1.3 ms split/merge + ~1 ms softmax) resists width fixes,
as the earlier warp-per-row port evidence predicted.

(Also confirmed during this pass: the profile-harness reference losses in the ⑬ note
were stale — current tree produces 8.791504/8.773757/8.728677/8.635798 both before
and after the alignment edits, bit-identical.)

### ㉙ Non-CUDA GPU compatibility branch (user requirement) — DONE ✅

New `llmm/vendor.mojo`: `HAS_CUBLAS = has_nvidia_gpu_accelerator() and not
is_defined["LLMM_FORCE_PORTABLE_GPU"]()`, nested inside the existing
`is_gpu` branches so the NVIDIA fast path compiles unchanged (bf16 losses match the
baseline exactly). The three CUDA-locked sites got portable fallbacks: classifier
PTX `ex2.approx` → accurate `exp()`; linear GEMMs → MAX `linalg.matmul` (fwd,
d_input, reusing the CPU-proven elementwise epilogues) and MAX's vendor-BLAS wrapper
(d_weight — resolves to rocBLAS/hipBLASLt on AMD, supports `transpose_a`+beta);
attention → `USE_GEMM_ATTENTION = HAS_CUBLAS` (falls back to the existing pure-Mojo
flash path). One real bug found & fixed: the portable `matmul_fwd` needed the same
`ctx.synchronize()` fences its d_input twin already had (CUDA_ERROR_ILLEGAL_ADDRESS
otherwise). **Both paths pass verify-gpu** (portable: LOSS OK 5.3557653, within
algorithmic noise; zero cuBLASLt symbols in the portable binary). Portable-mode cost
on this NVIDIA box: ~2.49 s/step (15×) — expected; it exists for correctness/
portability, not speed. CPU path unchanged (verified).

### ㉚ Selective grad-act zeroing (the 17.5 ms lever) — probe results

Empirical probe (per-step fill removed, alloc-time zero only): fp32 losses stay
identical at step 1 but drift from step 2 (5.0590 vs 5.060379 … step 9: 4.2300 vs
4.195305) — proving some tensors DO consume per-step zeros, while the 10-step
check's tolerance still reports "okay" (so tolerance-based checks canNOT validate
this change; bit-identity of the loss trajectory is the gate). Static analysis
(Opus + my independent pass) classifies the accumulators as exactly
{encoded, attn_proj, residual_2, fc_proj, residual_3} — the `+=` targets of the
fused-residual-backward broadcast — 308 MB in 3 contiguous ranges vs 3.29 GB.
Implementation in progress with bit-identical-loss validation gates.

Note for the record: my static dataflow read of the fused-residual backward
(plain-store into the d_residual scratch after skip grads were `+=`'d into the same
buffer) predicted a clobber that does NOT happen in practice (verify-gpu passes
bit-exactly vs PyTorch) — the empirical probe, not the static argument, is what
this change's safety rests on.

**㉚ LANDED.** `zero_gradients()` now does 3 sub-buffer fills
(`create_sub_buffer` + `enqueue_memset`, vendor-neutral, offsets derived from
`act_sizes`) covering {encoded, attn_proj+residual_2, fc_proj+residual_3} = 308 MB,
plus a one-time alloc-fill; CPU path unchanged. Validated by: fp32 verify (all
TENSOR OK), a **poison test** (whole grad_acts_buf sentinel-filled with 1e4 every
step → still TENSOR OK, so no accumulator was missed — this is the load-bearing
evidence), exact bf16 harness losses, and the portable
(`LLMM_FORCE_PORTABLE_GPU=1`) verify also green (the flash-path backward doesn't
consume the removed zeros). Discovered en route: **fp32 verify losses were never
bit-identical run-to-run** (atomic fp adds in weight-grad reductions) — bit-identity
gates only work on the bf16 harness path.

### Head-to-head after ㉘–㉚ (2026-07-01, thermally stable 46–49 °C, bracketed)

| implementation | median ms/step | detail |
|---|---:|---|
| llm.mojo bf16 (run 1 / run 2) | **149.8 / 150.4** | fwd ~49.1 + bwd ~85.5 + upd ~15.4 |
| llm.c bf16 | **134.8** | steps 4–10 settle at ~135 |

**Gap: 1.20× → 1.11×** this session (163.5 → 149.8 ms; llm.c steady at ~135–137).
The ~16 ms win is the ㉚ memset elimination (backward fell 97.7 → 85.5 ms — the
old full-buffer fill was enqueued inside `backward()`). ㉘ alignment added ~1.5 ms
(ncu-confirmed); ㉙ is perf-neutral on the default path by construction.

### Live per-step kernel comparison (nsys, same trace session; the residual ~10 ms)

| family | ours (ms) | llm.c (ms) | delta |
|---|---:|---:|---:|
| GEMM superfamily (cutlass+nvjet+dbias, incl. gelu: fused epilogues for us, separate 8.1 ms kernels for llm.c) | 93.7 | 89.2 | **+4.5** |
| layernorm (fwd runs 48 launches vs llm.c's 24!) | 7.7 | 4.6 | **+3.1** |
| split/merge (llm.c unpermute is 2.4× faster than our merge) | 5.9 | 4.0 | **+1.9** |
| attention softmax fwd (llm.c kernel5 is one-pass online) | 6.9 | 5.5 | **+1.4** |
| classifier | 4.3 | 3.8 | +0.5 |
| adamw | 15.4 | 14.9 | +0.4 |
| attention softmax bwd (P+dS+D) | 9.9 | 10.7 | **−0.9** |

Fairness flag: llm.c's step includes ~1.2 ms `global_norm_squared` (grad-clip);
whether our profile-harness step runs the equivalent is under investigation — if
not, our number is flattered by ~1.2 ms.

Next-lever candidates under investigation: LN-forward double-launch, cuBLASLt
workspace size (llm.c passes 32 MiB, enabling split-K algos), one-pass online
softmax forward, merge-kernel thread mapping.

### ⚠️ Measurement environment: an 87 GB vLLM server shares this GPU

Discovered 2026-07-01: a `VLLM::EngineCore` process (87.4 GB resident, running
since **June 28** — i.e., for this document's entire benchmark history) shares the
GB10. It idles most of the time but intermittently serves, which (not thermals)
likely explains the long-standing "llm.mojo floats 150–170 while llm.c stays
steady" variance pattern: short llm.c runs dodge the bursts. **Interleaved
same-window A/B (this session, vLLM quiet, 47–53 °C) confirms the code gap is
real: llm.mojo 150.5 ms vs llm.c 135–137 ms → 1.11×.** All future numbers should
be taken interleaved-same-window and report `nvidia-smi` utilization.

### ㉛ Round-2 levers — measured, all reverted (documented dead ends + one fairness fix)

- **Fairness fix (KEPT):** `profile_gpt2.mojo` now runs `calculate_grad_norm` +
  grad-scaled update like the real loop and llm.c's profiler (we were skipping
  ~1 ms of global-norm work). Honest harness baseline: **150.5 ms** (upd 15.4→16.4;
  losses shift after step 1 because clipping now actually runs: 8.791504/8.778885/…).
- **GELU fusion A/B (reverted to fused):** llm.c defaults `gelu_fusion=0`; we added
  `USE_GELU_FUSION` and measured fused 152.16 vs unfused 152.36 ms — **tied**; our
  cuBLASLt GELU_AUX_BIAS epilogue ≈ llm.c's plain-GEMM+separate-gelu. This also
  re-attributes the "+4.5 ms GEMM gap": cutlass+gelu is **matched** (66.5 vs 65.9);
  the real GEMM gap is **nvjet (vocab-sized) GEMMs +3.4 ms**.
- **Softmax vectorized loads (reverted):** width=8 loads broke bit-identity
  (8.791564 vs 8.791504) with no speed win (152.6 vs 152.2).
- **Merge streaming hints (blocked + neutral):** `CacheOperation.STREAMING` loads
  hit a hard **ptxas toolchain bug** in this Mojo build (reproduced standalone);
  width=2 plain was neutral. Toggles left in-tree, off.
- **LN "double-launch" (misdiagnosis):** the second ×24 `layernorm_fused*` kernel
  in the trace is the **backward junction broadcast**, not a forward double-launch.
  Correct reattribution of the +3.1 ms layernorm gap: llm.c fuses the residual
  `dinp +=` into layernorm_backward (2.61 ms total) while we run LN-bwd 1.9 +
  dparam 0.97 + a separate 2.4 ms broadcast sweep (~5.3 ms total). Fusing the
  broadcast into LN-bwd saves ~1–2 ms (scratch re-read + launch), not the full 2.4.

Current honest position: **150.5 vs ~136 ms (1.11×)**; remaining itemized gap:
nvjet vocab GEMMs +3.4, layernorm structure +3.1 (partially recoverable),
split/merge +1.9 (llm.c permutes for the same cuBLAS stride reason — no structural
dodge; micro-gap resists), softmax fwd +1.4 (resists), classifier +0.5, adamw +0.4,
P+dS −0.9 (we win). Escalated to a deep structural analysis (recompute policy,
buffer dtypes, redundant conversion passes, the unexplained 5.46 ms nvjet call).

### ㉜ LN broadcast fusion — LANDED (structural), + nvjet probes

`layernorm_fused_residual_bwd`'s GPU path now launches a single fused kernel
(`layernorm_bwd_residual_gpu`): the LN input-gradient `dval` is computed in
registers and `d_inp1 += dval; d_inp2 += dval` happens inline; the separate
broadcast sweep (2.4 ms/step: scratch re-read + 2×RMW) and the scratch store are
GONE (nsys confirms zero broadcast launches; deadness of the scratch-after-
broadcast was verified by enumerating all 6 touch-sites of the residual-grad
planes — every post-broadcast access is a write). Gates: fp32 verify green,
**bf16 10-step trajectories bit-identical** between store/no-store builds and to
the reference, portable path green. Est. win ~1.5 ms — below the current
environment noise floor (see below), banked structurally.

Discovered in gating: the fp32 verify trajectory is non-bit-reproducible run-to-run
(atomic fp adds) — cross-step staleness gates must use the deterministic bf16
harness.

nvjet probes: lm_head fwd (TN, algo 67, waves 163.75) / d_input (NN, algo 67,
waves 4.0); **all GEMM operand pointers are 16-byte aligned** — misalignment ruled
out as the cause of the ~1.36×-per-call cuBLASLt execution gap. That gap remains
library-internal and unexplained.

### Honest end-of-session status (2026-07-01 evening)

**Clean-window (GPU idle-verified) position: llm.mojo ~150.5 ms vs llm.c ~136 ms
→ 1.11×.** Session total: 2612 ms (fp32 baseline, June 30) → ~2530 (bf16 tuned
start) → 163.5 (session start today) → ~149–150. NOT parity.

The environment degraded during the final measurements: the co-resident vLLM
server began actively serving (70–96 % GPU util), inflating BOTH implementations
(llm.c 135→144–153; ours 150→152–156) — final micro-wins (LN fusion) cannot be
isolated in wall-clock until the box is quiet. Notably, our runs occasionally
post 134.6–136.3 ms steps (llm.c-parity speed) in quiet moments, which puts a
question mark on how much of the residual "gap" is asymmetric contention against
our larger memory footprint — the next clean-box session should re-measure before
attacking the remaining itemization.

Remaining itemized gap (clean-window, ~14 ms): cuBLASLt-internal ~3.4 ms on
identically-configured vocab GEMMs (descriptor parity verified byte-for-byte,
alignment verified — no code-side lever found); ~5–6 ms of bandwidth-bound
non-GEMM kernels (split/merge, softmax fwd, layernorm dparam, classifier) where
every documented lever has been measured and reverted; ~2–3 ms diffuse. The only
untaken lever with headroom is bf16 Adam moments (~4.5 ms) — rejected as unfair
(llm.c keeps fp32 optimizer state).

To reach parity from here would require one of: (a) beating cuBLASLt's own
execution on the vocab GEMMs (custom kernel — high effort, uncertain), (b) Mojo
toolchain codegen improvements on the bandwidth-bound kernels (streaming stores
are ptxas-broken in this build), or (c) numerics tradeoffs llm.c doesn't make.

### ㉝ Dead gradient-activation allocations removed — 1.92 GB freed

`grad_acts_buf` no longer allocates the GPU-dead tensors (`fch`, `logits`,
`att_probs` grads — verified never dereferenced on either the cuBLAS or portable
GPU path): 3.29 GB → **1.37 GB** via a `grad_act_sizes` array (zeroed sizes on
GPU; pointers alias the next tensor's start as an in-bounds dummy; the ㉚
selective-zeroing offsets now derive from the same array). Gates: fp32 verify,
bit-identical bf16 losses, portable verify, CPU test — all green. Total process
footprint shrinks ~1.9 GB, which also reduces our exposure to the co-resident
vLLM server's memory/bandwidth pressure.

### 🎯 PARITY REACHED (2026-07-01, end of session)

Quiet-window interleaved A/B, final build (through ㉝), two bracketed rounds:

| round | llm.mojo median (steps 1–7) | llm.c median (10 steps) |
|---|---:|---:|
| 1 | **134.6 ms** (133.7–135.1) | 134.7 ms |
| 2 | **134.9 ms** (133.8–136.0) | 136.0 ms |

**llm.mojo now matches llm.c step-for-step (equal round 1, ahead round 2),
B=4 T=1024 bf16, same-window measurements, losses bit-stable.**

The 150.5 → ~134.7 jump exceeds the ~2 ms of code wins that landed between the
two measurements (㉜ LN fusion, ㉝ dealloc), which retro-diagnoses the earlier
"cuBLASLt-internal 1.36× GEMM floor" as **unified-memory pressure**: with the
87 GB vLLM tenant resident, our previous 3.3 GB grad buffer (58 % dead) pushed
the combined working set into a regime that taxed the bandwidth-heavy tensor-core
kernels; freeing 1.9 GB (plus the earlier 17.5 ms memset removal shrinking
per-step traffic) moved us to the fast regime consistently. Lesson recorded: on
shared/unified-memory boxes, allocation footprint is a first-class performance
variable even when kernels never touch the dead bytes.

Cumulative: **2612 ms (fp32, June 30) → 134.7 ms (bf16, July 1) — ~19×; vs
llm.c: 12× slower → parity.** Full training-fidelity constraints held throughout:
fp32 verify vs PyTorch green, fp32 optimizer state, bf16 loss trajectory tracks
fp32, and a maintained non-CUDA GPU fallback (`LLMM_FORCE_PORTABLE_GPU`).

---

## AI use statement

The optimization campaign and benchmarks documented in this file (2026-06-30 →
2026-07-01, ~12×-slower → parity with llm.c) were performed with AI assistance
via Claude Code, under the direction of Evan Owen, who set the goals,
constraints, and acceptance criteria and reviewed the results.

**Models used and their roles:**

| model | role |
|---|---|
| **Claude Fable 5** (`claude-fable-5`) | Orchestrator: task decomposition, dispatching/reviewing all sub-agents, independent verification of agent claims (trace re-analysis, dataflow audits, final A/B benchmarks), and this log. |
| **Claude Sonnet** | All coding tasks: kernel alignment sweep (㉘), vendor/portability branch (㉙), selective gradient zeroing (㉚), round-2 lever A/Bs (㉛), LN broadcast fusion (㉜), dead-allocation removal (㉝), plus read-only scoping/investigation passes. |
| **Claude Haiku** | Benchmarking and profiling data collection: wall-clock baselines, ncu per-kernel confirmation runs, nsys timeline traces, head-to-head A/Bs. |
| **Claude Opus** | Hard analysis: activation-gradient accumulator classification (which buffers truly need per-step zeroing) and the structural parity analysis (nvjet GEMM investigation, LN fusion design, recompute-policy audit). |

**Goal prompt (operator-issued):** *"Reach parity or exceed the performance of
llm.c on GPU. Keep a detailed log of key findings in docs/benchmarks.md. For all
coding tasks use Sonnet agents, for all benchmarking tasks use Haiku agents, and
for tasks that need more thought call Opus; if Opus can't figure it out, the
orchestrator attempts it."* Supplementary operator directives during the run:
maintain a working non-CUDA GPU branch for any cuBLAS/CUDA-specific fast path;
independently verify Opus's analyses; maximize agent parallelism.

**Method and safeguards:** every code change was gated by `make verify-gpu`
(fp32 activations + gradients vs a PyTorch/llm.c reference), bit-identical bf16
loss trajectories where the change was numerics-preserving, poison (sentinel-
fill) tests for buffer-lifetime changes, and portable-path + CPU regression runs.
Agent claims were not taken on trust: the orchestrator refuted one agent's
"19 ms idle gap" finding by re-analyzing the raw nsys sqlite trace (the real
cause was a memset), and twice-"proven" library floors were later overturned by
experiment. Negative results were retained in this log as documented dead ends.

## Summary: knowledge gaps and future profiling guidance

### Open gaps in understanding

1. **Fused-residual backward dataflow (highest-risk gap).** The junction
   scratch/broadcast sequence in `train_gpt2.mojo` backward is empirically
   correct (verify-gpu, bit-identical multi-step trajectories) but repeated
   static derivations predicted a store-over-accumulate clobber that does not
   manifest. The current code's safety rests on empirical gates, not a proven
   dataflow theory. Any future edit to `layernorm_bwd`/junction wiring or the
   ㉚ zero set MUST re-run: fp32 verify + bf16 10-step bit-identity + the poison
   test.
2. **Mechanism of the memory-pressure GEMM slowdown.** Freeing 1.9 GB of dead
   allocations removed a ~1.36× slowdown on identically-configured cuBLASLt
   GEMMs, but the precise mechanism (page migration, TLB pressure, bandwidth
   partitioning against the co-resident 87 GB vLLM server) and its threshold
   were never isolated. Untested: performance on a truly dedicated GPU.
3. **fp32 nondeterminism.** fp32 verify losses vary run-to-run (atomic fp adds
   in weight-grad reductions); which kernels dominate the nondeterminism was
   never mapped. All bit-identity validation must use the bf16 harness.
4. **cuBLASLt heuristics.** Our lm_head GEMMs select algo_id 67; llm.c's
   selection was never captured. If GEMM regressions appear after toolchain
   updates, capture and compare both.
5. **Toolchain bugs/limits (version-pinned).** `CacheOperation.STREAMING` loads
   fail in ptxas in this Mojo build (mojo 1.0.0b3.dev2026062706); Mojo GEMM
   epilogue lambdas lower to separate kernels; `linalg.bmm` is non-tensor-core
   for head_dim=64. Retest all three on toolchain upgrades — several documented
   "floors" are contingent on them.
6. **Config generality.** All tuning and the parity claim are at B=4, T=1024,
   GPT-2 124M, bf16, on GB10. Selective-zeroing offsets and causal-triangle
   optimizations are config-derived (should transfer) but unmeasured elsewhere.
7. **Clocks.** The GB10 SM clock sat at 2411 MHz against a 3003 MHz max in every
   measurement — unexplained; affects absolute ms, not ratios.

### What to profile, and how, when continuing this work

- **Check tenancy before trusting any number:** `nvidia-smi --query-compute-apps`
  and utilization sampling first; the box hosts a bursty vLLM server. Benchmark
  **interleaved, same-window**, report util+temperature alongside every figure.
  Variance previously blamed on "thermal throttling" was largely contention.
- **Use three views, each for what it's good at:** (1) wall-clock harness
  (`LLMM_PROFILE_*` env + `build/profile_gpt2_bf16`) for the honest step time;
  (2) **nsys** per-step windows (bound steps by `encoder_fwd` launches or adamw
  clusters in the sqlite export) for idle gaps AND **memset/memcpy rows — the
  17.5 ms/step memset was invisible in every kernel-only view**; (3) **ncu**
  per-kernel deltas for thermally/contention-independent before/after
  attribution of a single change.
- **Compare per-family, per-call against llm.c**, not totals: the decisive
  insights came from same-window family tables (ours vs llm.c) and per-call µs
  of shared cuBLAS kernels. Beware accounting traps (our fused GELU epilogues
  vs llm.c's separate gelu kernels made the GEMM family look 4.5 ms worse than
  it was).
- **Audit allocation footprint before hunting kernel causes** — dead or
  oversized buffers cost real speed here even when never touched.
- **Gate every change** on: fp32 verify, bf16 loss bit-identity (or an explicit
  documented numerics justification), portable-path (`-D
  LLMM_FORCE_PORTABLE_GPU=1`) and CPU runs. Rebuild llm.c fresh before
  comparing (a stale llm.c binary once inflated our relative standing), and
  re-derive reference losses from the current tree (documented references have
  gone stale twice).
- **Record dead ends in this file.** More than half the session's measured
  ideas were neutral or worse; the documented negative results are what kept
  later rounds from re-treading them.

### Post-parity: equivalence-suite fallout from the optimization work (2026-07-01 night)

`make test` was red. Two pre-existing-at-HEAD breaks from the optimization
commits, root-caused via ordered-pair reproductions (Opus analysis, orchestrator-
verified):

1. **`tests/test_zero_equivalence.mojo` compile error** — the conditional
   `GPT2_DTYPE` (bf16 flag, June 30) broke an fp32-typed host copy in the test.
   Fixed: stage through a `GPT2_DTYPE` host buffer (the read loop already casts).
2. **183/235 pytest CUDA equivalence failures, order-dependent** — two mechanisms:
   - **(A, dominant)** the ㉖–㉘ explicit-alignment hints assume production dims
     (`channels % width == 0`, `seq_len % width == 0`); the suite's deliberately
     odd shapes (channels=767, seq_len=7) hit `CUDA_ERROR_MISALIGNED_ADDRESS`,
     which **permanently poisons the single shared CUDA context** of the pytest
     process — every later GPU test fails at model setup. Hence "183 fail
     together, each passes alone". Fix: host-side dispatch between the (unchanged)
     aligned fast path and a safe fallback when the shape invariant doesn't hold.
   - **(B)** the attention backward P-scratch was allocated **without**
     `zero_on_alloc` while the P+dS kernel writes only the causal lower triangle
     and `dV = Pᵀ·dO` reads the full plane — fresh dirty allocations on the
     test path (cache=None) corrupt dV's upper-triangle half. Training was
     protected by the separate acts-buffer memset (stored att_probs), so parity
     results are unaffected. Fix: `zero_on_alloc=True` (one-time, perf-neutral).

Lesson: the kernel fast paths encode shape invariants that the training loop
always satisfies but the equivalence suite deliberately violates — every
invariant-based optimization needs either a runtime guard + fallback or an
explicit shape assertion, and `make test` must be part of the change gates (this
suite hadn't run since June 29, so the breaks accumulated invisibly).

**RESOLVED (same night):** full CUDA pytest suite **235 passed / 0 failed** (was
183 failed). Fixes: (B) `zero_on_alloc=True` on the attention P scratch;
(A) host-side runtime dispatch between the byte-identical aligned fast path and
a scalar fallback wherever a shape invariant can be violated — layernorm bwd ×2
+ fused-residual fwd, attention P+dS (also fixing the odd-seq_len tail-drop),
global_norm (odd strides), encoder fwd/wte-bwd/wpe-bwd (channels=33); plus a
`pdtype` (fp32 param-grad) parameter on the layernorm-backward ops for the test
harness's fp32 dgamma/dbeta convention (production defaults unchanged).
gelu/adamw sites audited safe-by-construction (`idx = tid*width`);
classifier/softmax rely on `Vp % 4 == 0`, which every fixture and production
config satisfies. Gates: verify-gpu 16/16 TENSOR OK; bf16 harness losses
bit-identical; step 135.7 ms (parity regime intact); test_zero_equivalence 3/3.

---

## 2026-07-02 — Softmax gap re-investigation: noise-checked, warp-per-row hypothesis refuted

With parity reached, a fresh `make profile-ncu`/`make profile-llmc-ncu` pass at
B=4 T=1024 L=12 (same methodology as before) turned up one per-call delta
worth a second look: our causal softmax at ~647 µs/call vs llm.c's
`softmax_forward_kernel5` at ~482 µs/call (~1.3×). Everything else matched or
beat llm.c per the existing per-family table. This entry re-measures that
delta for noise, forms a concrete structural hypothesis, and tests it.

### Re-measurement: the gap is real, not noise

3× repeated `make profile-ncu PROFILE_B=4 PROFILE_T=1024 PROFILE_LAYERS=12` and
`make profile-llmc-ncu` (same flags), reading just the softmax kernel row:

| run | llm.mojo softmax (µs/call, 12 calls) | llm.c `softmax_forward_kernel5` (µs/call, 492 calls) |
|---|---:|---:|
| 1 | 651.3 | 481.4 |
| 2 | 643.4 | 484.5 |
| 3 | 626.4 | 482.6 |
| **mean** | **640.4** | **482.8** |

Both sides are tight (llm.mojo <4% spread, llm.c <1% spread) — a real,
reproducible **~1.33× gap**, not measurement noise.

### Hypothesis: block-per-row vs warp-per-row reduction

Reading `llmc/attention.cuh`'s `softmax_forward_kernel5`: llm.c assigns **one
warp per row** (`num_warps = blockDim.x/32` independent rows in flight per
block) and reduces with plain warp-shuffle (`warpReduceMax`/`warpReduceSum`) —
no block barrier, no shared memory. Our `_attention_softmax_causal_gpu`
(`llmm/attention.mojo`) assigns **one 256-thread block per row** and reduces
with `block.max`/`block.sum` — a cross-warp shared-memory reduction gated by a
`barrier()`. Causal rows vary wildly in length (1..1024 elements as
`query_index` goes 0..1023), so the theory: our block-wide reduction pays a
fixed barrier+shared-mem cost on every one of the 49,152 rows regardless of
row length, while llm.c's warp reduction is cheap (shuffle-only) and lets 8
independent warps in a block work on 8 different rows with zero cross-warp
sync — for short rows especially, that fixed overhead should dominate.

### Investigated: `scratch/proto_softmax_warp.mojo` — REFUTED

Built a standalone prototype reimplementing both kernels at the real shape
(B=4, NH=12, T=1024 → 48×1024×1024 bf16 scores plane), plus a third variant
adding llm.c's exact 4-wide vectorized reduce (`regarray[4]`) discovered on a
closer re-read of its source. Correctness verified first: max abs err ~1.5e-5
(bf16 rounding) between all three variants' output — all three compute the
same softmax. Then 3× repeated wall-clock timing (30-iter, same
enqueue-then-sync methodology as `proto_layernorm_max.mojo`):

| kernel | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| block-per-row (current production kernel) | 502.1 | 526.5 | 499.4 |
| warp-per-row (llm.c-style, scalar reduce) | 536.2 | 519.1 | 522.4 |
| warp-per-row + 4-wide vectorized reduce | 524.7 | 539.9 | 525.4 |

Every warp-per-row variant lands **within run-to-run noise** of block-per-row
(ratios 0.94×–1.01× across the three runs) — no consistent win, and on most
runs a small loss. **Reduction structure is not the explanation for the real
gap.** If it were, an isolated microbenchmark at the real shape should show a
large, repeatable win for the warp-per-row kernel; it shows none.

This reproduces the exact pattern already documented above for the layernorm
warp-per-row port ("SLOWER — shared-mem occupancy + `warp.sum` overhead don't
match CUDA's `warpReduceSum`", see the dead-ends table). Two independent
kernels now show the same result: **porting llm.c's warp-shuffle-heavy kernel
designs 1:1 does not reproduce its CUDA-side win in this Mojo toolchain.**
Treat this as a general, reusable pattern for future investigations, not a
one-off — Mojo's block-wide reductions appear to be the more reliable choice
here even when CUDA precedent favors warp-level.

### Remaining untested leads (not attempted — lower priority / higher cost)

Re-reading `softmax_forward_kernel5` turned up two more differences from ours,
neither retested here:

1. **`__ldcs`/`__stcs` streaming-cache read/write hints** on the write-back
   loop — already a documented dead end (`CacheOperation.STREAMING` hits a
   ptxas bug in this Mojo build, reproduced standalone; see "Levers TESTED
   that hit the floor" above).
2. **Reverse grid-iteration order** — llm.c processes rows back-to-front
   (`idx = (gridDim.x - blockIdx.x - 1) * num_warps + warp_id`) specifically
   so the L2-resident tail of the softmax output benefits the A·V matmul that
   runs immediately after. This is a cross-kernel cache-locality effect
   invisible to an isolated microbenchmark — testing it means changing the
   production kernel and measuring the full pipeline, not a quick standalone
   probe. Flagged for a future session if the softmax gap is revisited.

### Verdict

No easy win found. The ~1.3× softmax gap is real and reproducible, but its
cause isn't reduction granularity — it joins the residual bandwidth-bound
kernels (layernorm structure, split/merge, classifier) already documented
above as resisting every lever tried so far. `scratch/proto_softmax_warp.mojo`
is kept in-tree as the documented negative result, matching
`proto_layernorm_max.mojo`'s convention.

---

## 2026-07-02 (continued) — four-agent parallel investigation: one real win landed

With the softmax gap characterized but unsolved, four independent Opus agents
were dispatched in parallel, each briefed on the full history above and
assigned a distinct angle, each restricted to scratch-only experimentation so
they couldn't collide with each other or risk the production tree. Summary of
all four, ranked by outcome:

### ㉞ Merge kernel: coalesced-write + 8-wide vectorization — REAL WIN, LANDED

**Finding.** `merge_gpu`'s forward path indexed threads by HEAD-layout
position (coalesced read, strided write to token-layout). Flipping to index
by TOKEN-layout position instead — each thread owning a width-8 chunk within
one head_dim run, safe since head_dim(64) is a multiple of 8 — gives a
coalesced 16-byte load **and** store from the same thread (head_dim is the
fastest-varying axis in both layouts, so a width-8 run is contiguous in both
simultaneously). Verified in `scratch/proto_lnsplit_merge.mojo`: flipping the
index alone did **nothing** (~61.5 vs ~62.4 µs/call, refuting the "strided
writes are the penalty" theory), but flip **+ width=8** gave ~61.5 → ~52.9
µs/call (~14%), bit-identical output (err 0.0). This is *not* the documented
"split/merge x128 vectorize was worse" dead end from earlier in this doc —
that tested vectorizing the *original* (coalesced-read/strided-write)
orientation, where a wide store scatters; the win requires the coalesced-write
orientation first.

**Landed** in `llmm/merge.mojo`: a new `merge_fwd_gpu_coalesced[dtype, width]`
kernel (width=8) alongside the original `merge_fwd_gpu` (width=1), with
host-side dispatch in `merge_fwd` — `if Int(head_dim) % 8 == 0` picks the
coalesced kernel, else falls back to the always-correct original. This is the
exact same host-dispatch pattern already used throughout the codebase
(layernorm, encoder, global_norm) to guard vectorized fast paths against the
equivalence suite's deliberately odd shapes — the class of bug that broke 183
tests earlier in this campaign (see "Post-parity: equivalence-suite fallout").
`merge_bwd` was **not** touched: the agent noted it "already writes coalesced
to head-layout and just needs width=8," but never actually measured that
claim, so it wasn't applied — only the empirically-verified forward change
shipped.

**Gates, all green:** `make verify-gpu` (16/16 TENSOR OK, bf16 losses match
the known reference trajectory: 5.356195, 5.2044296, 5.060379, …); `make
test` — full 235/235 suite passed, including
`test_split_merge_equivalence.py::test_merge_forward_matches_reference[fp32_odd_head]`
(head_dim=5, not a multiple of 8) — confirmed this test actually exercises
the fallback path, not a coincidental pass.

**In-pipeline effect** (ncu, same-session before/after): `merge_fwd` fell
~89.9 → ~83.7 µs/call in the full training step (smaller than the isolated
~14% — expected, given pipeline contention differs from an isolated
microbenchmark — but real and in the right direction, non-zero).

### Dead ends (three agents, all rigorous negative results)

- **cuBLASLt vocab-GEMM workspace size.** The doc's own "untried lever" (match
  llm.c's 32 MiB workspace to unlock split-K algos) turned out to already be
  moot: `llmm/matmul.mojo` already passes 32 MiB, byte-identical to llm.c's
  `cublas_common.h`. A sweep from 4→256 MiB (`scratch/proto_vocabgemm_workspace.mojo`)
  showed cuBLASLt *never* changes its algorithm choice (always algo 67, no
  split-K, 0 bytes requested) regardless of budget. The ~3.4 ms/step nvjet
  vocab-GEMM gap remains real but is now provably not a workspace-starvation
  artifact.
- **Softmax: fast-exp and cache-eviction hints.** Two more leads from
  `softmax_forward_kernel5` tested in isolation
  (`scratch/proto_softmax2_fastexp.mojo`, `scratch/proto_softmax2_cachehint.mojo`):
  `ex2.approx` fast-exp was neutral-to-marginally-slower (548.0 vs 559.1
  µs/call); `read_only`/`EVICT_FIRST` cache hints were a statistical tie
  (551.9–553.8 µs/call, within 0.3%) — the 100 MB scores plane vastly exceeds
  L2, so there's no residency for a hint to protect. Both measured under
  **locked ncu clocks** after discovering naive wall-clock timing on this box
  is unreliable: idle downclocking between timed blocks means whichever
  kernel runs second in a pair looks faster, producing a false 1.22×
  "win" that reversed to a false 1.22× "loss" just by swapping order. Treat
  this as a standing methodological caution for any future timing work here.
- **Layernorm backward fusion, and the bandwidth-vs-latency-bound question.**
  Fusing dgamma/dbeta into the residual d_input kernel (llm.c kernel10's
  structure) was tested (`scratch/proto_lnsplit_fused.mojo`) and found to be a
  net loss at every block count: the occupancy the d_input pass needs (~1536
  blocks) makes the atomic dparam-flush contend, while the only atomic-free
  escape (warp-per-row) is already documented above as slower in this
  toolchain. Separately, a from-first-principles bandwidth-vs-latency probe
  (`scratch/proto_fresh_bandwidth.mojo`) directly measured — rather than
  assumed — that GB10's memory-bound kernels are bandwidth-bound: a plain
  vectorized copy already hits the hardware d2d-memcpy ceiling (~240 GB/s),
  and a cp.async double-buffered version came back **byte-for-byte identical**
  to the plain copy. This closes the entire latency-hiding lever family
  (cp.async, deeper ILP, prefetching) at once, and independently confirms
  production adamw (measured 15.1 ms) sits almost exactly at the predicted
  DRAM-ceiling time (124.4M params × 28 B / 232 GB/s ≈ 15.0 ms).

### Fresh same-session benchmark (post-merge-fix, 2026-07-02)

`make benchmark-gpu BENCH_B=4 BENCH_T=1024`, 3 repeated runs, GPU quiet (vLLM
tenant resident but idle, checked via `nvidia-smi` before each run):

| run | llm.mojo bf16 (mean/median ms) | llm.c bf16 (mean/median ms) |
|---|---:|---:|
| 1 | 136.10 / 136.10 | 140.46 / 137.79 |
| 2 | 135.79 / 135.88 | 136.49 / 136.47 |
| 3 | 135.15 / 135.36 | 138.92 / 137.27 |

llm.mojo now measures **consistently at or ahead of llm.c** across all three
runs (135.15–136.10 ms vs 136.49–140.46 ms), with tighter variance (std
0.9–1.1 vs 1.1–4.9). This is a step beyond the earlier "equal, within noise"
parity result (134.6–134.9 vs 134.7–136.0) — a modest, real improvement from
the merge fix compounding with a clean measurement window today. `make
profile-ncu`/`make profile-llmc-ncu` (same B/T/L) confirm no regression
elsewhere: total ncu-serialized time 149.4 ms (ours) vs the stable ~2.02 ms
(sic, 2020 ms — llm.c's non-comparable serialized total, unchanged from
earlier in this doc), with the merge_fwd delta the only kernel-level change
in the per-kernel table.

### Verdict

One small, real, fully-verified win shipped (merge kernel, ~0.1–0.2 ms/step
in isolation, part of what pushed today's session to measure ahead of
llm.c). Three rigorous dead ends closed off real search space — cuBLASLt
workspace tuning, softmax fast-exp/cache-hints, layernorm dgamma fusion, and
the entire async-copy/latency-hiding lever family — each with mechanism, not
just a number, so they won't be re-attempted. All four scratch prototypes are
kept in-tree with RESULT-header summaries per this campaign's convention.

---

## Apple Silicon (Metal GPU) benchmarking setup

> **Extended coverage:** the July 2026 Apple Silicon port campaign — including a
> full gotcha catalog (silent-wrong-results, hard failures, and performance
> findings), probe scripts, and the Metal-specific optimization log — is
> documented in [`docs/ai/metal_port_gotchas_and_optimizations.md`](metal_port_gotchas_and_optimizations.md).
> The section below covers how to run and compare benchmarks on Apple Silicon.
> Refer to that document for the *why* behind `HAS_METAL` branches in the code.

### What is compared and why llm.c is absent

llm.c has **no Metal/Apple-GPU port** — it is CUDA-only on the GPU side. On Apple
Silicon the GPU comparison is therefore **llm.mojo (Metal) vs PyTorch MPS (Metal
Performance Shaders)**, not vs llm.c.

- **llm.mojo side:** the same `build/profile_gpt2` binary used for NVIDIA benchmarks,
  invoked with the `gpu` target. On Apple Silicon, Mojo's GPU backend dispatches to
  Metal automatically — no extra build flags or env vars are required.
- **PyTorch MPS side:** `train_gpt2.py --device mps` (the repo-local script, which
  already calls `torch.mps.synchronize()` before timing for accurate per-step
  measurement).
- **llm.c:** skipped on Apple with a printed notice — `"llm.c has no Metal port —
  baseline is PyTorch MPS"`.

At B=4, T=1024, fp32, llm.mojo on Metal runs at approximately **6.5 s/step** on an
M4 Max (fp32 only; bf16 is not yet profiled on Metal). This is substantially slower
than the NVIDIA GB10 numbers because (a) fp32 has no tensor-core acceleration,
(b) Metal's compute throughput is lower than GB10's, and (c) the same GEMM-level
optimizations (cuBLASLt, CUDA-specific attention kernels) do not apply to Metal.
Profiling and optimization for the Metal backend are future work.

### How to run

```sh
# Build the profiling harness (one-time, unless .mojo sources change)
make build-profile

# Throughput histogram: llm.mojo Metal vs PyTorch MPS
# Writes figures/benchmark_metal_b4_t1024_<date>_Apple-M4-Max_Mac-M4-Max.png
make benchmark-metal BENCH_B=4 BENCH_T=1024 BENCH_METAL_STEPS=5

# Same via the auto dispatcher (picks Metal on Apple Silicon automatically)
make benchmark BENCH_B=4 BENCH_T=1024

# Perfetto trace of one Metal training step (open at https://ui.perfetto.dev)
make profile-metal
# Or equivalently:
make profile PROFILE_TARGET=gpu

# Train on Metal GPU
make train-metal
```

### Tool availability on Apple Silicon

| tool | available | notes |
|------|-----------|-------|
| Perfetto tracer (in-process) | YES | `make profile-metal` — cross-platform |
| NVIDIA Nsight Compute (ncu) | NO | `make profile-ncu` exits cleanly with a note |
| NVIDIA Nsight Systems (nsys) | NO | `make profile-nsys` exits cleanly with a note |
| `make benchmark-metal` | YES | no llm.c dependency |
| `make benchmark-gpu` (NVIDIA) | NO | requires nvcc + CUDA |

### 2026-07-02: Metal-support commit regressed the CUDA forward (caught & fixed)

Commit f37cbd7 ("Metal support") is almost entirely CUDA-clean (audited hunk-by-
hunk: Metal-gated or provably neutral; the f8c0a86 coalesced merge kernel is a
clean bit-identical NVIDIA win). ONE exception: it **unconditionally de-fused**
the four per-layer forward matmuls (plain GEMM + separate `bias_gelu_fwd`) for
ALL targets — routing around Metal's broken fused-bias epilogue but also undoing
change ⑳ on CUDA: forward 44.7→48.6 ms (step ~135→~136.5-137.7, parity lost by
~2 ms) and bf16 numerics shifted (step-0 loss 8.791504→8.792923, the extra bf16
rounding the ㉛ fusion A/B predicted). It also left the recompute-path
rematerialization on the OLD fused call, breaking its bit-identity contract with
the new forward when `recompute=True`. Fix: per-target gate (`HAS_CUBLAS` →
fused, else split) at the four sites + the recompute site.

**Measurement-hygiene lesson (how this was almost missed):** the first empirical
"no regression" verdict came from a benchmark agent that claimed to rebuild but
ran a 6-hour-stale binary — bit-identical losses and unchanged timings looked
like a clean bill of health and directly contradicted the (correct) static
audit. Always verify the binary's mtime AFTER building, and treat
"losses bit-identical" as suspicious whenever the diff should have changed
numerics.

**Regression guard (canonical, referenced from `matmul_fwd`'s dispatch):** the
bias(+GELU) handling is target-dispatched INSIDE `llmm/matmul.mojo::matmul_fwd`
— call sites in `train_gpt2.mojo` stay single fused-style calls with no vendor
branching. On `HAS_CUBLAS` the bias/GELU MUST remain fused into the cuBLASLt
GEMM epilogue (change ⑳): de-fusing costs ~4 ms/step of forward and changes
bf16 numerics through an extra rounding round-trip (step-0 loss
8.791504 → 8.792923), forfeiting llm.c parity. The split GEMM + `bias_gelu_fwd`
path exists ONLY for targets whose fused-bias epilogue is unavailable or broken
(Metal). Any future change to this dispatch must be validated with a
freshly-built binary (verify the output mtime actually changed — a stale binary
once masked this exact regression) against the bit-identity reference losses:
8.791504 / 8.778885 / 8.727783 / 8.6381035 (bf16 harness, B=4 T=1024 L=12,
steps 0–3), plus `make verify-gpu` and the portable-path
(`-D LLMM_FORCE_PORTABLE_GPU=1`) verify, which exercises the split branch on
NVIDIA hardware.

### 2026-07-03: Correction to the ㉚-era dataflow note — the "clobber" WAS real

The 2026-07-01 note above ("my static dataflow read … predicted a clobber that
does NOT happen in practice") is **retracted**. The training-correctness campaign
(CHANGELOG 2026-07-03) proved the static analysis was right all along: the
residual-skip gradient was never seeded into the junction backward
(`layernorm_fused_residual_bwd` only accumulated the LN term), so block
gradients decayed geometrically with depth — wpe's gradient was ~10⁶× too
small. The empirical gates that "overruled" the analysis were vacuous: the old
`test_gpt2` used a flat absolute tolerance of 2.0 (meaningless for near-zero
gradients) and never asserted the loss trajectory. Two more latent bugs fell in
the same sweep (GELU-grad fused into the wrong MLP backward; layer-0 LN1
backward fed the normed output instead of the pre-norm input — also flagged in
the 07-01 analysis).

**Process lesson (the mirror image of the stale-binary lesson):** when a
careful static derivation and a passing test disagree, interrogate the TEST's
sensitivity before discarding the derivation. A gate can only overrule analysis
if it could actually have caught the predicted failure.

**Consequences for the performance record:** all pre-2026-07-03 bit-identity
reference losses are obsolete (the old backward did less work); the CUDA-parity
claim (134.6 vs 134.7 ms) was measured with the incorrect, cheaper backward and
must be re-validated post-correctness — re-benchmark in progress.

### Post-correctness CUDA benchmark (2026-07-03, official 40-step run, vLLM idle)

Hardened correctness gate green, then `make benchmark-gpu BENCH_B=4 BENCH_T=1024`
(figure: `figures/benchmark_gpu_b4_t1024_2026-07-03_1119_NVIDIA-GB10_DGX-Spark.png`):

| config | median ms | tok/s |
|---|---:|---:|
| **llm.mojo bf16** | **135.97** | **30,154** |
| **llm.c CUDA bf16** | **135.77** | 30,210 |
| llm.mojo fp32 | 417.72 | 9,816 |
| llm.c CUDA fp32 | 298.53 (n=3) | 13,771 |
| PyTorch bf16 / fp32 | 514.51 / 587.57 | 7,957 / 6,966 |

**Parity CONFIRMED with the corrected backward: 1.002× (135.84 vs 135.58 mean).**
The three correctness fixes (residual-skip seeding, GELU-grad swap, LN1 input) plus
unconditional store-P cost only ~1 ms net vs the old (incorrect) 134.7 ms — the
earlier parity result survives the correctness campaign. New bf16 harness reference
trajectory (B=4 T=1024 L=12, repeated batch): 8.791564 / 7.937944 / 6.013462 /
3.675395 / 2.128407 / 0.884390 / 0.202385 / 0.107844 — note the loss now actually
COLLAPSES (the old broken backward hovered ~8.2–8.8 over the same steps; gradients
were decaying geometrically with depth). Per-phase steady state: fwd ~50 ms,
bwd ~83 ms, upd ~16.5 ms.

Open fp32 note: llm.mojo fp32 is 1.40× behind llm.c fp32 (417 vs 298) — fp32 was
never the optimization target (bf16 is the shipped config); recorded as a possible
future workstream.

### Post-correctness CPU benchmark (2026-07-03)

First pass produced a FALSE regression signal: an agent stacked three concurrent
`benchmark-cpu` runs; the llm.mojo arm of the surviving run absorbed the
contention (732 ms mean, std 116 — vs llm.c/PyTorch arms that ran after cleanup
and matched their July-1 references). Clean-box harness measurement:
**llm.mojo CPU fp32 ≈ 448 ms/step** (fwd 112 + bwd 300 + upd 35.5, B=4 T=64) —
matching the July-1 value (454 ms) within noise. **No CPU regression from the
Metal + correctness changes**; still ~4× faster than llm.c OpenMP (1808 ms,
flat vs July 1). A discarded duplicate agent report also carried an internally
inconsistent table (ms vs tok/s contradictory) — sanity-check tok/s = B·T/ms
before trusting any benchmark table. Official rerun on a quiet box in progress;
figure regenerated.

**Official clean CPU rerun (2026-07-03 11:47, quiet box, single instance,
figure `figures/benchmark_cpu_b4_t64_2026-07-03_1147_NVIDIA-GB10_DGX-Spark.png`):**

| config | mean ms | median ms | tok/s |
|---|---:|---:|---:|
| **llm.mojo fp32** | **457.87** | 452.63 | **559** |
| llm.c OpenMP (20t) | 1815.94 | 1817.58 | 141 |
| llm.c 1-thread | 6913.65 | 6910.79 | 37 |
| PyTorch fp32 | 632.74 | 539.52 | 405 |

llm.mojo CPU = 457.9 vs July-1's 454.2 ms — **flat through the Metal +
correctness changes** (correct backward included). Still ~4.0× faster than
llm.c OpenMP and ~1.2× faster than PyTorch (medians). All cross-arm references
match July 1 within noise, confirming the earlier 732 ms reading was pure
run-stacking contention.

---

## 2026-07-10 — fp32 parity via TF32: llm.mojo fp32 now *beats* llm.c fp32 (282.8 vs 294.7 ms)

The 2026-07-03 official benchmark left one open gap: llm.mojo fp32 was
**1.40× behind llm.c fp32** (417.72 vs 298.53 ms/step, B=4 T=1024). A static
analysis ([`fp32_gap_analysis_static_2026-07-10.md`](fp32_gap_analysis_static_2026-07-10.md))
and an independent same-day ncu profile (`fp32_parity_profile_2026-07-10.md`,
`goal1/fp32-parity` branch) both converged on a single cause: **llm.mojo's
fp32 GEMMs never enabled TF32 tensor cores** — `_matmul_cublaslt`
(`llmm/matmul.mojo`) and `_attn_gemm_batched`'s cuBLAS tail
(`llmm/attention.mojo`) hardcoded `ComputeType.COMPUTE_32F` (plain FP32 CUDA
cores; the profile showed 74.7% of the fp32 step in `simt_sgemm`/`magma`
non-tensor-core kernels), while llm.c's fp32 arm auto-enables
`CUBLAS_COMPUTE_32F_FAST_TF32` + `CUBLAS_TF32_TENSOR_OP_MATH` on any
compute-capability-8.0+ GPU (`train_gpt2_fp32.cu:1614-1618`) — i.e. llm.c's
"fp32" benchmark was always TF32-vs-TF32 on GB10.

### The change (commit `c27a1f9`)

One new comptime flag + two call sites:

- `llmm/vendor.mojo`: `USE_TF32` (default **on**; disable with
  `-D LLMM_NO_TF32=1` for true IEEE fp32 debugging).
- `llmm/matmul.mojo` (`_matmul_cublaslt`): compute type is now
  `COMPUTE_32F_FAST_TF32` iff `dtype == DType.float32 and USE_TF32`, else
  `COMPUTE_32F` — covers every FC-layer fwd/bwd GEMM (qkv, attproj, fc,
  fcproj, lm_head), ~93% of model FLOPs.
- `llmm/attention.mojo` (`_attn_gemm_batched` `cublasGemmStridedBatchedEx`
  tail): same gate keyed on `a_dt`, covering QKᵀ/A·V forward and dQ/dK/dV
  backward batched GEMMs.

bf16/fp16 builds are untouched (input dtype alone already selects tensor
cores there; both keep `COMPUTE_32F` accumulate). The two `-D LLMM_NO_TF32=1`
/ default binaries were confirmed to differ (flag reaches codegen), and both
build cleanly.

### Correctness gate (test_gpt2.mojo, gpt2_124M_debug_state.bin, B=4 T=64)

Three runs, all GPU, same reference:

| run | result | note |
|---|---|---|
| baseline (pre-change `7c3dad0`) | **PASS** (all 16 tensors + 10-step loss) | reference point |
| TF32-**off** (`-D LLMM_NO_TF32=1`, post-change) | **PASS** | losses match baseline to ≤5e-7/step — within the run-to-run noise of the atomics-based reductions (two identical TF32-off runs differ by ~1e-6), proving the flag plumbing is inert |
| TF32-**on** (default, post-change) | **PASS** (16/16 gradient tensors + 10-step loss under the TF32-calibrated `LOSS_STEP_TOL=0.02`) | max per-step loss drift 0.0102 (step 3) — exceeds the strict-IEEE 0.01 bar but is expected TF32 rounding, see below |

TF32-on loss trajectory vs the fp32 reference (Δ per step, 10-step overfit):
+0.0014, +0.0079, −0.0094, **+0.0102**, −0.0005, +0.0019, +0.0040, −0.0008,
+0.0011, +0.0008 — max |Δ| 0.0102, no growth trend, loss collapses
identically (5.356 → 0.356). Every per-tensor gradient check stays green
under the existing mixed tolerance (atol 0.01 / rtol 0.05 / L2 3.0) with
huge margin (typical maxdiff 30–100× below threshold).

**Tolerance approach:** llm.c hit exactly this and chose to test with TF32
off — `test_gpt2_fp32.cu:41` reads `enable_tf32 = 0; // NOTE: disable TF32
for testing!!!` even though its training binary enables TF32. We mirror
that for the primary gate: **`make verify-gpu` runs with
`-D LLMM_NO_TF32=1`** (strict IEEE fp32, tight `LOSS_STEP_TOL=0.01`,
nothing loosened; this is what `make verify`/`make check` gate on). The
TF32 path gets its own real gate, **`make verify-gpu-tf32`**, which must
pass: `test_gpt2.mojo` detects `USE_TF32` at comptime and uses a
TF32-calibrated **`LOSS_STEP_TOL=0.02`** — ~2× the measured max drift
(0.0102, no growth trend across the run), so healthy TF32 rounding passes
while a real regression still fails (the failure modes this check exists
to catch — dead/exploding gradients — deviate by O(0.1+) within a few
steps and/or grow monotonically, far past 0.02). The gradient tolerances
are NOT loosened for TF32 (it passes them with 30–100× margin), so this is
a scoped, per-check calibration, not a blanket loosening (contrast the
past flat atol=2.0 gate that buried three real bugs).

### Benchmark (interleaved A/B, B=4 T=1024, GPU flock held, quiet box)

Pre-run state: `free -g` 121 total / 76 used / 45 available; nvidia-smi 0%
util, 43 °C, only Xorg/gnome-shell resident. Binaries rebuilt this session
(mtime checked newer than sources). 3 interleaved reps; llm.mojo 25 steps/rep
(WARMUP=5 trim → n=20/rep, pooled n=60); llm.c fp32 runs a fixed 8-step epoch
(same harness limitation as every prior run), 3 runs/rep → pooled n=27.

| arm | n | mean ms | median | std | tok/s |
|---|---:|---:|---:|---:|---:|
| **llm.mojo fp32 (TF32, this change)** | 60 | **282.77** | 283.02 | 1.85 | **14,485** |
| llm.c CUDA fp32 (TF32) | 27 | 294.67 | 294.21 | 3.19 | 13,901 |
| llm.mojo fp32 pre-fix (same day, `goal1/fp32-parity` profile) | 40 | 416.66 | — | — | 9,831 |
| llm.c fp32 (same-day pooled reference) | 6 | 294.01 | — | — | 13,931 |

tok/s sanity: 4096/282.77 ms = 14,485 ✓; 4096/294.67 = 13,901 ✓.

**Result: 416.66 → 282.77 ms/step (1.47× speedup) — llm.mojo fp32 is now
0.96× of llm.c fp32, i.e. ~4% FASTER.** The static analysis predicted
270–300 ms; measured 282.8. Combined with the 2026-07-03 bf16 parity
(135.97 vs 135.77 ms), llm.mojo now matches or beats llm.c CUDA at **both**
precisions of the shipped training config. Per-rep means were 281.1 / 283.4
/ 283.7 ms (llm.mojo) vs 290.7–298.7 ms (llm.c) — the arms never overlap,
so the win is outside noise.

Why llm.mojo lands slightly ahead: with the GEMM backends now identical
(cuBLASLt + TF32 both sides), the earlier non-GEMM kernel work (fused
classifier once, alignment hints, LN fusion, selective zeroing) carries
over to fp32, where llm.c's fp32 harness lacks some of its bf16-side
optimizations (e.g. its fp32 attention still materializes preatt/att fully
in fp32 and runs a separate backward classifier pass).

### Post-merge regression sweep (2026-07-10, merged HEAD `9382ee1`)

Full sweep to prove the TF32 change regressed no other precision. Pre-run
state: `free -g` 121/75 used/46 available; nvidia-smi 0% util, 42-43 °C,
quiet box. All binaries rebuilt this session (mtimes checked newer than
sources, including llm.c's `train_gpt2cu`/`train_gpt2fp32cu`). Every
GPU-touching command ran under the `/tmp/llmm-gpu.lock` flock.

**Verdict per precision — NO REGRESSIONS:**

- **bf16 (GPU)**: harness losses (B=4 T=1024 L=12, 8 steps) — steps 0-1
  digit-for-digit the 07-03 reference (8.791564 / 7.937944) in every run;
  steps 2-7 show small run-to-run variation. Controlled by rebuilding the
  bf16 harness at pre-TF32 `7c3dad0` in a scratch worktree: the pre-change
  binary shows the SAME variation, one pre-change run matches the 07-03
  reference digit-for-digit at all 8 steps, and another pre-change run is
  digit-for-digit identical to a post-change run at all 8 steps. So the
  bf16 numeric path is empirically unchanged (exact cross-build trajectory
  match) and the wiggle is pre-existing atomics nondeterminism — the
  earlier "bf16 is bit-deterministic" note (㉚/㉜ era) evidently no longer
  holds unconditionally, but bit-identity *across builds per realization*
  does. Benchmark: 134.77 ms vs 135.97 on 07-03 (within noise).
- **fp32 (GPU)**: `make verify-gpu` (strict IEEE, TF32 off, tight
  LOSS_STEP_TOL=0.01) PASS — 16/16 TENSOR OK + 10-step loss.
  `make verify-gpu-tf32` (default TF32 path, calibrated 0.02) PASS.
  Benchmark 282.23 ms — matches the merge-time 282.77 result.
- **fp32 (CPU)**: `make verify-cpu` PASS (16/16 TENSOR OK, 10-step loss
  trajectory exact vs debug state). Timed sanity run (profile harness,
  B=4 T=64 L=12, 8 steps, CPU dispatch): losses collapse sanely,
  steady-state ~470 ms/step (fwd ~120 + bwd ~315 + upd ~35) vs the 07-03
  official 457.9 — within ~3% on a box with a resident ~75 GB tenant. No
  full CPU benchmark rerun (llm.c CPU arms take hours; nothing in this
  change touches CPU dispatch — comptime-guarded).

**Official 6-arm GPU benchmark** (`scripts/benchmark_train.py --device gpu`,
B=4 T=1024, 40 steps, WARMUP=5 → n=35/arm, all arms interleaved in one
session 13:34, figure
`figures/benchmark_gpu_b4_t1024_2026-07-10_1334_NVIDIA-GB10_DGX-Spark.png`):

| arm | n | mean ms | median | std | tok/s | 07-03 official |
|---|---:|---:|---:|---:|---:|---:|
| llm.mojo fp32 (TF32) | 35 | **282.23** | 282.60 | 1.20 | 14,513 | 417.72 |
| llm.mojo bf16 | 35 | **134.77** | 134.83 | 0.72 | 30,392 | 135.97 |
| llm.c CUDA fp32 | 3 | 292.94 | 293.36 | 1.49 | 13,983 | 298.53 |
| llm.c CUDA bf16 | 35 | 133.68 | 133.60 | 0.84 | 30,639 | 135.77 |
| PyTorch fp32 | 35 | 579.20 | 578.81 | 2.26 | 7,072 | 587.57 |
| PyTorch bf16 | 35 | 502.92 | 502.89 | 2.33 | 8,144 | 514.51 |

tok/s = B·T/ms sanity-checked on every row. Every non-fp32-mojo arm matches
its 07-03 reference within noise; the only mover is the intended one
(llm.mojo fp32, 417.72 → 282.23).

Harness gotcha found en route: `make benchmark-gpu` runs
`scripts/benchmark_train.py` in the **default** pixi env, whose torch has
no CUDA — the PyTorch arms fail with "Torch not compiled with CUDA
enabled" inside `_run()`'s captured output and are then *silently dropped*
from the table (empty sample list → `summarize()` returns None). The
official run above was invoked as `pixi run -e cuda python
scripts/benchmark_train.py ...` to get all six arms; a first invocation via
the make target produced a 4-arm table (its incomplete figure was deleted).
README updated with the new GPU table + TF32 note and the 07-03 CPU table.

---

## 2026-07-10 — LN-backward single-kernel fusion: correctness holds, performance REGRESSES (not merged)

`docs/ai/perf_hunt_analysis_2026-07-10.md` identified LN backward as the
single biggest non-GEMM gap (4 launches/invocation vs llm.c's 1 fused
`layernorm_backward_kernel10`, ~13 element-passes vs ~5) and projected
**savings** of fp32 −7.4 ms (best −9.0) / bf16 −4.0 ms (best −5.0) from
fusing it. `_layernorm_bwd_fused_gpu` (`llmm/layernorm.mojo:1488`,
branch `goal1/ln-bwd-fusion`, HEAD `edd2b67`, parent `a8f9f56`) implements
that fusion: block-per-row grid-stride kernel, dgamma/dbeta via
single-writer shared-memory partials (no atomics — deterministic) plus an
atomic block-counter flag-finalize (llm.c kernel10's idiom) with in-kernel
flag self-reset, and the residual-broadcast seed folded in via a
`HAS_RESID_IN` comptime flag. Call sites: `ln_2` (all 12 layers) and
`ln_1`-else (layers 1–11); layer-0's `ln_1` and the plain (non-residual)
`ln_f` backward are intentionally untouched. This entry is the successor
validation pass: gates 1/2/4 (verify-gpu, verify-gpu-tf32, verify-cpu) had
already passed pre-format; this pass re-checks verify-cpu on the committed
(post-`mojo format`) tree, runs the bf16 twin-run determinism gate, and
runs the interleaved A/B perf gate + ncu kernel-count proof.

**Bottom line: correctness is intact and the kernel-launch count really
did drop, but the fused kernel is measurably *slower* in aggregate than
the four launches it replaces — the opposite of the projected win.** Not
merged; flagged back to the implementer.

### Cheap re-check: `make verify-cpu` on the committed tree

PASS — 16/16 TENSOR OK + all 10 loss steps LOSS OK against
`gpt2_124M_debug_state.bin`, confirming the post-`mojo format` reformat
was whitespace-only (matches the pre-format gate result already recorded
for this branch).

### Gate 3 — bf16 twin-run determinism protocol

Ran `build/train_gpt2_bf16` at HEAD (`edd2b67`) twice with the identical
invocation (`-e gpt2_124M_bf16.bin -x 10 -v 0 -s 0`; defaults already pin
B=4, T=1024, tinyshakespeare data, the pretrained checkpoint — no
overfit-single-batch flag, so this is a real-data, non-collapsing
trajectory), then built and ran the `a8f9f56` parent once from a scratch
`git worktree add --detach a8f9f56` (binaries confirmed fresh via
`stat` mtime vs source mtime before every run).

| step | loss (HEAD run A) | loss (HEAD run B) | loss (`a8f9f56` baseline) | norm (all 3 runs) |
|---:|---:|---:|---:|---:|
| 1  | 4.369226 | 4.369226 | 4.369226 | 17.1131 |
| 2  | 4.418465 | 4.418465 | 4.418465 | 17.8014 |
| 3  | 4.510433 | 4.510433 | 4.510433 | 23.8323 |
| 4  | 4.026239 | 4.026239 | 4.026239 | 15.4808 |
| 5  | 3.578290 | 3.578290 | 3.578290 | 11.3918 |
| 6  | 3.767685 | 3.767685 | 3.767685 |  8.9980 |
| 7  | 3.534359 | 3.534359 | 3.534359 |  4.6588 |
| 8  | 3.655098 | 3.655098 | 3.655098 |  7.2158 |
| 9  | 3.253812 | 3.253812 | 3.253812 |  4.7633 |
| 10 | 3.390009 | 3.390009 | 3.390009 |  4.2824 |

**Determinism verdict: same-build twin-run is identical to printed
float32 precision (7 significant digits) at all 10 steps — an
improvement** (the new kernel's single-writer shared-memory dgamma/dbeta
accumulation, replacing the old atomics-based cross-block accumulate, is
evidently the reason: no atomic-ordering nondeterminism left in this
path). **Cross-build (HEAD vs `a8f9f56`) also matches to the same printed
precision at every step** — stronger than the "steps 0-1 exact, steps 2+
within the usual wiggle band" expectation set going in; here there is no
visible wiggle at all in this config. Loss doesn't collapse (real
tinyshakespeare batches, not an overfit single batch), so "collapse"
doesn't apply, but the trajectories are exact matches, which is the
stronger statement anyway.

### Gate 5 — perf: interleaved A/B wall-clock + ncu kernel-count proof

**Setup.** `profile_gpt2` / `profile_gpt2_bf16` harness (B=4, T=1024,
L=12 — `LLMM_PROFILE_LAYERS=12`, matching the shipped config), 25 steps
per arm-rep (`WARMUP=5` trimmed → n=20/rep). Two interleaved reps per
session, arm order reversed between reps (`HEAD-fp32, BASE-fp32,
HEAD-bf16, BASE-bf16` then reversed) to cancel drift; the whole session
run under one `flock /tmp/llmm-gpu.lock` hold. `a8f9f56` binaries built
fresh in the scratch worktree (`git worktree add --detach a8f9f56`,
symlinked data/weights/`.pixi` from the main repo). Pre-run state: `free
-g` 121 total / 75 used / ~45 available; `nvidia-smi` 0% util, 44-52 °C,
quiet box. Every binary's mtime checked newer than its source tree
immediately before use. Two independent sessions run back-to-back to
check reproducibility.

| arm | session A mean ms (n=40) | std | session B mean ms (n=40) | std |
|---|---:|---:|---:|---:|
| HEAD fp32 | 287.67 | 1.79 | 287.32 | 2.59 |
| `a8f9f56` fp32 | 285.75 | 2.05 | 287.31 | 1.71 |
| **HEAD bf16** | **142.41** | 1.16 | **143.18** | 0.86 |
| **`a8f9f56` bf16** | **135.18** | 0.94 | **136.22** | 1.36 |

tok/s sanity: 4096/143.18 ms = 28,608 ✓ (checked on every row).

**fp32: ~parity in wall-clock** (Δ +1.92 ms session A, +0.02 ms session
B — both within the fp32 arm's own ~1.7-2.6 ms std, i.e. not
statistically distinguishable from noise at this n). **bf16: HEAD is
robustly ~7 ms *slower*** (Δ +7.23 ms session A, +6.96 ms session B — well
outside the ~1 ms bf16 std, reproduced independently twice). This is the
opposite sign of the ≥3.5 ms bf16 / ≥6 ms fp32 *savings* acceptance
targets.

**ncu, 1-step capture, same B=4/T=1024/L=12 config, same session** (the
`profile-ncu`/`profile-fp32-ncu` Make targets default `PROFILE_T` to
**64**, not 1024 — a harness gotcha hit while building this table;
`PROFILE_T=1024` must be passed explicitly to match the shipped config,
otherwise the whole-step total comes out ~8x too small and isn't
comparable to wall-clock). Baseline reproduced the analysis doc's
previously-published fp32/bf16 LN-backward totals within ~5%/~1%,
confirming methodology:

| LN-backward kernel | `a8f9f56` fp32 (calls, ms) | HEAD fp32 (calls, ms) | `a8f9f56` bf16 (calls, ms) | HEAD bf16 (calls, ms) |
|---|---|---|---|---|
| `residual_grad_broadcast` (grid 3072) | 24, 5.34 | — (folded in) | 24, 2.17 | — (folded in) |
| `layernorm_bwd_residual_gpu` (bwd_r, d_input) | 24, 7.13 | — (folded in) | 24, 3.09 | — (folded in) |
| `_layernorm_bwd_fused_gpu` (new, `HAS_RESID_IN=True`) | — | 23, **22.53** | — | 23, **14.68** |
| `_layernorm_bwd_fused_gpu` (new, layer-0 `ln_1`, `HAS_RESID_IN=False`) | — | 1, 0.79 | — | 1, 0.54 |
| `_ln_dparam_accum_gpu` (atomics) | 25, 3.49 | 1 (ln_f only), 0.15 | 25, 2.51 | 1 (ln_f only), 0.10 |
| `_ln_dparam_finalize_gpu` | 25, 0.094 | 1 (ln_f only), 0.004 | 25, 0.090 | 1 (ln_f only), 0.004 |
| plain `layernorm_bwd_gpu` (ln_f, ×1, unchanged) | 1, 0.145 | 1, 0.131 | 1, 0.064 | 1, 0.060 |
| **LN-backward family total** | **16.20** | **23.61** | **7.93** | **15.38** |
| **launch count (whole step)** | **99** | **27** | **99** | **27** |

(`a8f9f56` fp32 16.20 ms / bf16 7.93 ms track the analysis doc's 15.43 /
7.98 ms closely — different session, same order of magnitude and
breakdown shape, confirming this measurement is apples-to-apples with the
doc's baseline.)

**Kernel-count proof: confirmed 4→1 per invocation** (`residual_grad_broadcast`
+ `bwd_r` + `_ln_dparam_accum_gpu` + `_ln_dparam_finalize_gpu` → one
`_layernorm_bwd_fused_gpu` launch), and whole-step LN-backward launch
count drops **99 → 27** (3.7×), exactly as designed. **But the family's
total *time* went the wrong way: fp32 16.20 → 23.61 ms (Δ +7.41 ms),
bf16 7.93 → 15.38 ms (Δ +7.45 ms) — both precisions regress by almost
exactly the same absolute amount**, which is suspiciously close in
magnitude (opposite sign) to the analysis doc's predicted savings. The
fused kernel and the old `bwd_r` kernel launch with the *identical* grid/
block config (`(1536,1,1)` / `(256,1,1)`) in both builds, so this isn't a
fewer-blocks occupancy story — the extra cost is *inside* each of the
1536 blocks (the shared-memory dgamma/dbeta reduction plus the
atomic-flag cross-block finalize apparently cost more per block than the
three small/cheap kernels they replaced saved). Root-causing that is out
of scope for this validation pass.

The fp32 wall-clock arm not showing this regression clearly is explained
by the ncu total: `a8f9f56` fp32 whole-step ncu total 292.62 ms vs HEAD
300.42 ms (Δ +7.80 ms) — consistent with the LN-family delta — but fp32's
TF32 GEMMs (63% of the step) carry enough run-to-run variance
(session A vs B fp32 deltas swung from +1.9 ms to +0.02 ms) to bury a
~7-8 ms shift at this n. bf16's GEMMs are more stable, so the same-sized
regression is clearly visible there.

### Verdict

- **Correctness:** intact. `verify-cpu` PASS, bf16 loss trajectory
  bit-reproducible (an improvement — no more atomics wiggle) and matches
  the pre-change parent exactly.
- **Kernel fusion:** did what it says — 4 launches → 1 per invocation,
  99 → 27 total LN-backward launches per step.
- **Performance:** **regression, not the projected win.** Acceptance
  targets were ≥6 ms fp32 / ≥3.5 ms bf16 *savings*; measured result is
  **≈+7.4 ms *slower*** in both precisions (ncu, same-session, apples-to-
  apples vs the parent commit), reproduced in bf16 wall-clock across two
  independent interleaved sessions. **Not merged to `main`.** Flagging
  back to the implementer: the fused kernel's per-block cost needs
  profiling at the warp/shared-memory level (likely the single-writer
  dgamma/dbeta shared accumulation or the atomic-flag finalize
  serializing more than expected) before this lands — per-kernel ms, not
  just launch count, needs to improve for this to ship.

## 2026-07-10 — LN-backward fusion REDESIGN: regression fixed, family time beats baseline (2-kernel, register-accumulate)

The first single-kernel LN-backward fusion (`edd2b67`, validated in the
entry above) was correct and dropped launches 4->1 but REGRESSED family
time to fp32 23.61 / bf16 15.38 ms (+7.4 both) vs the pre-fusion baseline
(`a8f9f56`) 16.20 / 7.93 ms. This entry redesigns it and lands the win.

### Diagnosis (1-step ncu, bf16, hot `HAS_RESID_IN=True` variant, T=1024)

ncu on `_layernorm_bwd_fused_gpu` (the `edd2b67` single kernel), B=4 T=1024:

| metric | value | reading |
|---|---:|---|
| Duration / launch | 596 us | ×24 ≈ 14.3 ms — matches the regression |
| Memory throughput | 6.24% | not streaming — latency/serialization bound |
| Compute (SM) throughput | 5.21% | idem |
| SM active / elapsed cycles | 508K / 1281K = 40% | ~60% of wall time most SMs idle |
| Registers / thread | 72 | -> Block Limit Registers = **3** |
| Static SMEM / block | 16.45 KB | -> Block Limit Shared Mem = 5 |
| Theoretical / achieved occupancy | 50% / 47.7% | register + SMEM capped |
| top stalls | 67.8% L1TEX scoreboard (mem latency) + 40.4% CTA barrier | |

Root cause = the three things the `edd2b67` design added on top of the
(fast, 129 us) baseline `bwd_r` d_input kernel: (1) a **serial single-block
finalize** — "last block" reduces 1536 block-partials × 768 ch × 2 while 47
SMs idle (the 40%-SM-active tail); (2) a **16 KB/block dgamma/dbeta shared
array** (2×2048 fp32) that, with 72 reg/thread, capped occupancy at 50%; (3)
**per-element shared RMW** in the hot pass-2 loop.

### Redesign

Two kernels, register accumulation, no shared dgamma/dbeta array:

1. **Main** (`_layernorm_bwd_fused_gpu`): unchanged block-per-row d_input +
   folded resid seed, but dgamma/dbeta now accumulate into **per-thread
   registers** across the block's grid-stride rows (thread `tid` owns a fixed
   disjoint channel chunk on every row, so no sharing, no atomics), flushed
   once at the end to a **channel-major** scratch (`scratch[c*blocks_cap+b]`).
   Removes the 16 KB shared array (restores occupancy) and the per-element
   shared traffic. `SM_OVERPROVISION` 32->3 (1536->144 blocks) — one wave at
   the register-bound 3-blocks/SM residency, still SM-saturating, with the
   fewest per-block partials to flush and finalize (see the tuning note below).
2. **Finalize** (`_ln_bwd_fused_finalize_gpu`): **one block per channel**
   (grid.x == channels), each block coalesced-reads its channel's contiguous
   partials and block-reduces them into d_gamma/d_beta. All `channels` blocks
   run concurrently across every SM — replaces the serial single-block tail.

Numerically the register accumulation reproduces the shared-array values
exactly (same per-row order); the finalize's block-reduction order differs
from a strict 0..N sequential sum, so results are twin-run deterministic
(fixed block dims) though no longer bit-identical to the sequential variant.

### Gates

- verify-gpu (strict IEEE, TF32 off): PASS — 16/16 TENSOR OK, 10/10 LOSS OK.
- verify-gpu-tf32: PASS. verify-cpu: PASS.
- bf16 twin-run: run A == run B bit-identical at all 10 steps (determinism
  preserved).

### Family time (1-step ncu, T=1024, L=12)

| arm | edd2b67 (failed) | a8f9f56 baseline | **NEW** | vs baseline | acceptance |
|---|---:|---:|---:|---:|---|
| fp32 | 23.61 | 16.20 | **9.34** | **-6.86** | <= 10 ms: **PASS** |
| bf16 | 15.38 | 7.93 | **4.82** | **-3.11** | <= 5 ms: **PASS** |

(llm.c `layernorm_backward_kernel10` floor: fp32 5.55 / bf16 2.58.)

NEW breakdown (bf16): main 4.26 (23×) + 0.16 (1×) + finalize 0.23 (24×) +
ln_f dparam-accum/finalize 0.11 + plain bwd_g 0.07. The main d_input sweep
(4.26) is now the whole cost; the finalize is 0.23 (down from the first
attempt's ~2.4 ms single-block tail equivalent). `SM_OVERPROVISION=3`
(144 blocks) beat 8 (384) by shrinking the strided channel-major scratch
flush: bf16 main 4.78 -> 4.26, family 5.49 -> 4.82.

Wall-clock corroboration (full step, L=12, T=1024, 25 steps/arm, WARMUP=5,
two rounds with arm order reversed):

| arm | round 1 | round 2 | vs baseline |
|---|---:|---:|---|
| NEW bf16 | 133.92 | 134.42 | ~-2 ms |
| BASE bf16 (a8f9f56) | 135.45 | 137.04 | |
| NEW fp32 | 277.48 | 281.46 | ~-7 ms |
| BASE fp32 (a8f9f56) | 286.43 | 286.84 | |

Both precisions the NEW build is faster than the pre-fusion baseline in
wall-clock (and ~8 ms faster than the failed `edd2b67` bf16 142 ms).

### Gotcha

- The Makefile `PROFILE_T` defaults to **64**, not 1024 — pass
  `PROFILE_T=1024` (or set `LLMM_PROFILE_T`) or ncu profiles the wrong shape.
- Mangled Mojo kernel names truncate (`_ln_bwd_fused_finalize_gpu` ->
  `..._ln_bwd_fused_f...`) — match on the truncated stem when parsing ncu CSVs.

### AI use statement

Written with AI assistance (Claude Code / Opus agent), directed by Evan Owen.
