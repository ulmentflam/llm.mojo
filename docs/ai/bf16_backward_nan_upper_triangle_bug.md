# The bf16 backward NaN bug (Blackwell sm_120 upper-triangle scratch)

A root-cause writeup of the bug that made **every bf16 training run NaN from
step 1** on the 8× RTX PRO 6000 Blackwell Max-Q (sm_120) box, and the fix on
branch `agent/bf16-nan`. This is a companion to
[`bf16_generation_misaligned_address_bug.md`](bf16_generation_misaligned_address_bug.md)
— another Blackwell-class bf16 attention-kernel bug — and to the tuning-campaign
report [`hyperparam_tuning_8gpu_2026-07-14.md`](hyperparam_tuning_8gpu_2026-07-14.md),
whose "Part 2 — bf16 backward NaN" section is the blocker this fixes.

---

## The symptom

`make build-bf16 WORLD_SIZE=1` → `build/train_gpt2_bf16`, run on
tiny_shakespeare, printed a **finite step-1 loss but `norm nan`** from step 1
onward, then `loss nan` from step 2:

```
val loss 4.5120053
step 1/10 | loss 4.365238 | norm nan | lr 0.0 | ...
step 2/10 | loss nan     | norm nan | lr 0.0 | ...
```

Reproduced from-scratch (`-e d12`) **and** from the pretrained
`gpt2_124M_bf16.bin` checkpoint; persisted at `LR=0` (`-l 0.0 -c 0.0`); not
fixed by `MODULAR_DEBUG=device-sync-mode` (so not the async-launch race from the
prior run report). The identical **fp32** config trained cleanly. bf16 training
had previously been verified green on GB10 (aarch64), so this looked
Blackwell-specific — and it was.

Because the loss was finite but the gradient norm was NaN, the fault was in the
**backward** pass. Gradient clipping then multiplies every gradient by
`clip/norm = 1/nan = nan`, and AdamW's `m = β1·m + (1-β1)·nan` corrupts the
moments, so no learning-rate or warmup choice can dodge it — the NaN is upstream
of the optimizer.

## Localizing it: the grad-dump machinery

`dump_grads_gpt2.mojo` runs exactly one forward+backward on a fixed batch and
dumps all 148 parameter-gradient tensors as fp32. Two incidental fixes were
needed to use it here:

1. Its token reader (`test_gpt2.read_to_dtype_pointer`, a raw `memcpy` off a
   `read_bytes` List) corrupted the first one or two tokens of every read on
   this box — `x[0]` came back as `0x7FC01000` (a NaN bit pattern) instead of
   the file's `50256`. Switched the tool to `llmm.io.read_and_copy` (a
   bounds-checked per-element copy), which reads correctly.
2. The fixed reference batch is `B=4, T=64`, and the bug **does not reproduce
   at T=64** (all 148 tensors clean). It needs the production `T=1024`. Added
   an optional `<out_dir> [T] [B]` override that tiles the debug tokens to fill
   a larger batch with valid in-range tokens.

At `T=1024` the dump showed a textbook backward-propagation pattern: layer 11
(last, processed first in backward) fully clean; **layer 10 the origin** — its
`qkv_weight` grad only *partially* NaN (128 of 2304 rows) and `ln_1` NaN, while
its `proj`/`fc`/`ln_2`/`attn_proj` grads were clean; layers 0–9 fully NaN. The
NaN is born between layer 10's `attn_proj` backward (clean) and its `qkv`
backward (NaN) — i.e. inside **`attention_bwd`**.

Finer: the NaN in `qkv_weight` fell exactly in **head 0's Q and K rows, with V
untouched**. In attention backward `dV = Pᵀ·dO` (clean → `P` and `dO` are fine)
but `dQ = dS·K` and `dK = dSᵀ·Q` (NaN). Both dQ and dK read `dS`; dV does not.
So **`dS` is the poisoned tensor.**

## Root cause

Env-gated instrumentation (`-D LLMM_ATTN_BWD_DEBUG*`) scanned the internal
attention-backward scratch per head. On the very first backward call (layer 11)
head 0 alone already had **238 bad `dS` values**, while `dP`, `P`, and `D` were
all finite. Every bad `dS` sat at a position with **`j > i` — above the causal
diagonal — where `P = 0`.** Two more scans nailed the timing: the `dS` buffer
was **verified all-zero the instant before the P+dS kernel launched** (the
one-time `zero_on_alloc` memset works), and came back with the 238 NaN
**immediately after** it.

The P+dS kernel (`_attention_bwd_p_and_ds_gpu`) processes the `[B·nh,T,T]` score
plane in `width=8` blocks. As a bandwidth optimization it **skipped** any block
lying entirely above the causal diagonal (`jbase > i`) with a bare `continue`,
relying on "the persistent scratch was zeroed once at allocation and nothing
ever writes the upper triangle, so it stays zero." That invariant held on GB10.

On **Blackwell sm_120 it did not**: the upper-triangle of the `dS` scratch —
provably zero the moment before the kernel, and never written by the kernel's
own source logic — came back holding NaN/Inf bit patterns after the launch.
The dense gradient GEMMs `dQ = dS·K` and `dK = dSᵀ·Q` sum `dS` over the **full**
`T` range (they rely on the masked half being a numeric zero, not a skipped
region), so those NaNs contaminated every query and key gradient, and the
residual stream carried the poison down through every earlier layer — the whole
backward went NaN from step 1. The forward never touches this buffer (finite
loss), and the fp32 build's plane is large enough / its kernels structured such
that it stayed clean — which is exactly why the bug looked LR-independent and
precision-specific.

## The fix

Stop relying on "skip == stays zero." In `_attention_bwd_p_and_ds_gpu`'s aligned
path, the above-diagonal blocks now **explicitly store a vectorized zero** every
step instead of `continue`-ing over them:

```mojo
if jbase > i:
    var zero_vec = SIMD[dtype, width](0)
    (ds_ptr + e0).store[width=width, alignment=align](zero_vec)
    comptime if not stored_p:
        (p_ptr + e0).store[width=width, alignment=align](zero_vec)
    b += grid_stride
    continue
```

An explicit zero-store is a cheap streaming write (no load, no compute) and
makes correctness independent of whatever the scratch happens to carry —
defensive against exactly the class of "untouched device memory is not
zero" surprise this was. The on/below-diagonal math is byte-for-byte unchanged,
so healthy gradients (and the T=64 reference / fp8 gate) are unaffected.

## Verification

- **Backward buffers (T=1024, `-D LLMM_ATTN_BWD_DEBUG`):** pre-fix, layer 11
  head 0 had 238 bad `dS`, cascading to `dP_bad≈993k` per head by layer 10;
  post-fix, **zero bad values across all layers and heads.**
- **Grad dump (T=1024, 148 tensors):** pre-fix 85/148 tensors NaN; post-fix
  **0/148.**
- **bf16 training, from checkpoint, `-l 0 -c 0`:** pre-fix `norm nan` step 1;
  post-fix finite norms (`17.07, 13.96, 11.26, …`) and finite loss for all 10
  steps.
- **bf16 training, from scratch (`-e d12`, `-l 3e-4`):** loss `11.006 → 7.94`
  over 10 steps, finite norms throughout — matching the fp32 reference
  trajectory (`11.0 → 8.05`) from the tuning report.
- **fp32 unaffected:** identical config trains as before (unchanged code path).

## Lesson

"Nothing writes here, so it stays at its initial value" is not a safe invariant
for GPU scratch across hardware generations, even when the buffer is verified
zero microseconds earlier. When a kernel *skips* a region that a downstream
dense op will still read, prefer an explicit write over relying on persistence —
the bandwidth saved is not worth a silent NaN that only shows up on one GPU
family. Compare the sibling bug in
[`bf16_generation_misaligned_address_bug.md`](bf16_generation_misaligned_address_bug.md):
both were Blackwell-only bf16 attention-kernel faults invisible on GB10, one an
alignment assumption, this one a persistence assumption.

---

## AI use statement

Written with AI assistance (Claude Opus agent via Claude Code), directed by Evan.
