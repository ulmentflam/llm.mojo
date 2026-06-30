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

**Hardware:** NVIDIA GB10 (Grace-Blackwell, `spark-c265`), aarch64, 20 cores, Linux 6.17.

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

Figure: `figures/benchmark_gpu_b4_t1024_2026-06-30_0909_NVIDIA-GB10_spark-c265.png`

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
