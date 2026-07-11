# The Metal `headout4` unaligned-`seq_len` garbage bug

A root-cause writeup of a bf16 attention correctness bug that only manifests on
the Apple Metal GPU, found on 2026-07-11 when Metal testing was restored (the
box had been missing Xcode 26.6's separately-downloadable Metal Toolchain, so
nothing had compiled or run for a while). This is a companion to
[`bf16_generation_misaligned_address_bug.md`](bf16_generation_misaligned_address_bug.md):
same family (bf16 attention at a non-8-aligned `seq_len`), but a different
kernel, a different failure mode, and Metal-only.

---

## The symptom

Four cases in `tests/test_attention_equivalence.py` failed, all on the Metal
accelerator path (`prefer_accelerator=True`):

```
test_forward_matches_torch[bf16_seq_len_9_unaligned]
test_forward_matches_torch[bf16_seq_len_17_unaligned]
test_backward_matches_torch[bf16_seq_len_9_unaligned]
test_backward_matches_torch[bf16_seq_len_17_unaligned]
```

Not a small tolerance miss: `np.testing.assert_allclose` reported ~94% of
elements wrong with a max absolute difference around `8.5e37` (near the fp32
ceiling), i.e. the output was garbage, not slightly off. The forward's softmax
normalizer `l_vec` was correct in the same runs; only the attention output
(and the gradients) were corrupt. That split is the whole clue.

## Why this is not the earlier bug

The earlier `CUDA_ERROR_MISALIGNED_ADDRESS` bug lived in the forward softmax
kernel (`_attention_softmax_causal_gpu`) and was fixed (commit `f0da883`) by
compiling both a vectorized and a scalar variant and dispatching on
`T % softmax_width == 0`. That fix is present and correct, and it is why
`l_vec` is fine here. It was verified on CUDA, where a misaligned vector access
hard-faults. This bug is in a different place and never faults, so the CUDA
regression suite that guards the earlier fix could not have caught it.

## Root cause

The attention output `A·V` and the three backward products `dQ = dS·K`,
`dK = dSᵀ·Q`, `dV = Pᵀ·dO` are batched matmuls whose contracted (K) dimension
is `T`, the sequence length. On CUDA these go through cuBLAS, which handles any
shape. Metal has no cuBLAS, so they run two hand-written batched-matmul
kernels: `_attn_headout4_gpu` (for `A·V` and `dQ`) and `_attn_headout4_transA_gpu`
(for `dK` and `dV`).

Both kernels tile the K axis in steps of `BK = 16` and accumulate with an
inner loop written as `comptime for kk in range(BK)`: a compile-time unroll
that always executes exactly `BK` iterations. Their shared-memory tile loads
had no bounds guard. When `T` is a multiple of `BK` (training's `T = 1024`,
and the aligned test cases) every K-tile is full and the unrolled loop reads
exactly the valid elements. When `T` is not a multiple of `BK` (generation's
`T = 1, 2, 3, ...`, and the `seq_len = 9, 17` test cases) the final K-tile is
short, and the loop's tail iterations read past the valid K range. On Metal an
out-of-bounds read returns garbage rather than faulting, and those garbage
`fma` terms land in every valid output row's accumulator, which is why the
whole output is corrupt while `l_vec` (produced by the separate softmax, which
was already guarded) stays clean.

The scoreout products (`QKᵀ` in the forward, `dP = dO·Vᵀ` in the backward)
were unaffected because their K dimension is `head_dim` (32 or 64 here), which
is aligned to `BK`; only the K = `T` products break.

## The fix

Zero-pad the K-remainder (and, for hygiene, any past-`M`/past-`N` reads) in
both kernels' shared-tile loads: read the real element only when its row and
column are in range, otherwise store `0`. A padded `0` contributes `0` to the
`fma`, so the unrolled tail is harmless. For aligned production shapes
(`T = 1024`, `head_dim = 64`) every guard is always true, so the training path
is byte-for-byte unchanged.

```mojo
a_sh.ptr[ra * (BK + 1) + ca] = (
    a_ptr[a_base + ra * K + k + ca] if (m_off + ra < M and k + ca < K)
    else Scalar[dt](0)
)
```

This is a genuine kernel-logic gap, not a toolchain codegen problem: no
`mojo`/`max` update was needed, and the same source compiles correct output on
CUDA (which never runs these kernels) and, after the guard, on Metal.

## Verification

- `tests/test_attention_equivalence.py`: 32/32 pass on the Metal accelerator
  (was 4 failing, the four cases above). The aligned bf16, fp32, and
  `*_cpu_forced` cases still pass, confirming no regression.
- The existing `bf16_seq_len_9/17_unaligned` cases are the regression guard:
  they exercise `head_dim = 32`, `seq_len = 9` and `17`, which is exactly the
  short-final-K-tile condition, and they route to the accelerator kernel.

## Why it stayed hidden

CUDA CI exercises cuBLAS for these products, never the Metal `headout4`
kernels, so the alignment assumption in those kernels was never tested there.
Locally the build had been broken (missing Metal Toolchain), so the Metal
accelerator path had not run at all. Restoring the Metal Toolchain and the test
suite surfaced it immediately.

---

## AI use statement

Root-caused and fixed with AI assistance (Claude Code), directed by Evan Owen,
while restoring Metal testing after the Metal Toolchain reinstall.
