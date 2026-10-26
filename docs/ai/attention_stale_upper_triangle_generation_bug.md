# Word-salad generation: stale causal-attention planes (+ a bf16 sample reinterpret)

A root-cause writeup of why a 40-step bf16 fine-tune on tinyshakespeare
(`make train-gpt2-124m-bf16 STEPS=40 ARGS="-v 1000 -s 1000 -m 1"`) printed a
healthy val loss (3.685) followed by a gibberish sample, and of the two bugs
behind it. Neither came from the TileTensor migration: both reproduce on the
pre-migration baseline. The more serious one also **silently corrupted every
training step after a mid-run sample** (`-s N` with N smaller than the run
length).

---

## The symptom

```
val loss 3.6853046
generating:
---
 avatar Lehsports stancesauRONT own underscores No ne involve �S policemen stockp copol ...
```

A GPT-2 124M with val loss 3.69 on Shakespeare should produce mostly
well-formed, Shakespeare-flavoured English. Val loss being right while the
sample is wrong points at code that only the sampling block runs: logits
readback, the sampler, the tokenizer, or `forward()` at the shapes only
generation uses (B=1, T = 1, 2, 3, ...).

## Bug 1 (the cosmetic one): the training loop reinterprets bf16 logits as fp32

`train_gpt2.mojo`'s sampling block copied `vocab_size` logits into
`logits_host_buf` (a `HostBuffer[GPT2_DTYPE]`, so bf16 in this build) and then
did

```mojo
var logits = rebind[ImmutMemPtr[DType.float32]](model.logits_host_buf...)
sample_softmax(logits, model.config.vocab_size, coin)
```

That is a reinterpretation of 2-byte storage as 4-byte floats, not a cast. It
reads twice the buffer and samples from garbage, which explains the
non-English byte soup (`avatar Lehsports stancesauRONT`). This is the
**same bug** that commit `f0da883` fixed in `infer_gpt2.mojo`
([`bf16_generation_misaligned_address_bug.md`](bf16_generation_misaligned_address_bug.md),
"A second bug found in the same pass"). Only the copy in `infer_gpt2.mojo`
was fixed. The training loop's copy has had the rebind since `666806e`
(2026-06-20), before bf16 storage existed, and became a bug when bf16 landed.

Fixing only this made the training sample English words, but still word
salad:

```
Wearwonderoto beot beot living bestod Red color are all for most like anything like3 ...
```

Even the **fp32** `infer_gpt2` on the **pretrained** `gpt2_124M.bin` produced
the same kind of salad ("ToughTough things all of any other just about them
don't and all and..."), so a second, precision-independent bug remained.

## Bug 2 (the real one): the causal probability plane keeps stale values across shapes

### Isolating it

Two throwaway diagnostics, built against `train_gpt2.GPT2` like
`infer_gpt2.mojo`:

1. **Logits readback is fine.** On the first 1024 tinyshakespeare val tokens,
   the host cross-entropy computed from the no-targets logits matched the
   GPU's targets-path loss to 7 digits at every T (1024, 64, 9, 1). So
   readback, the row offset (`(t-1) * V_p`) and padded-vocab handling are
   correct.
2. **`forward()` is neither causal nor idempotent across shapes.** Per-row CE
   for rows 0–2 should not depend on T, since row i only sees tokens 0..i.
   Running the same tokens at T = 1024, 1024, 64, 64, 64, 1024, 1024 gave:

   | call | T | mean CE | row-0 CE |
   |---|---|---|---|
   | 1, 2 | 1024 | 4.6753526 | 7.3099 |
   | 3, 4, 5 | 64 | 6.058106 | 6.5483 |
   | 6, 7 | 1024 | **6.5056634** | **9.1638** |

   Results are deterministic for a given call sequence. But one T=64 call
   *permanently* changed every later T=1024 result, so some persistent device
   state depended on the previous call's shape.

Hugging Face `gpt2` (transformers, CPU fp32) on the same tokens:
T=1024 → 4.675241, T=64 → 5.151654, T=9 → 5.145991, row 0 → 7.307611. The
fresh first T=1024 call was right. Every call after a shape change was wrong.

### Root cause

GPU attention runs through `attention_fwd_gemm` (`llmm/attention.mojo`):
QKᵀ → `_attention_softmax_causal_gpu` → A·V, through a `[B·nh, T, T]`
probability plane. The `775eeab` perf pass (2026-07-01) halved the softmax's
store traffic by having it **write only the lower triangle**
`[0, query_index]` of each row. It relies on the upper triangle staying zero
from a **one-time memset**:

- the GPT2 store-P path writes P into `acts.att_probs[layer]` (per-layer
  stride `B·nh·T·T` *for the current call's shape*), zeroed once when
  `allocate_activations` memsets `acts_buf`;
- the shared `gemm_att` scratch (used when store-P is off) is memset once
  when allocated.

"Zero once" is only valid while every call uses the same `(B·nh, T)`.
Training does, so it never noticed. Token-by-token generation calls
`forward(1, t)` for t = 1, 2, 3, ..., and each call reinterprets the same
memory with a new row stride T and new per-layer slice offsets. The values the
previous call wrote in *its* lower triangle now sit in *this* call's upper
triangle, and A·V multiplies them into V as if they were attention weights on
future positions. T=1 has no upper triangle, so it was correct, and a freshly
zeroed first call was correct. Everything after a shape change was wrong, which
matches the table above exactly.

### The consequence for training

The training loop runs generation forwards at `-s` intervals. They write
generation-shaped planes into `acts.att_probs`, and the next training forward
at (B=4, T=1024) writes only the lower triangle, so the stale values in its
upper triangle **persist for the rest of the run**. Measured with
`STEPS=20 -v 1000 -m 1`:

| binary | `-s` | step 11 loss | final val loss |
|---|---|---|---|
| HEAD `15d0968` | 1000 (no mid-run sample) | 3.954165 | 3.7883291 |
| HEAD `15d0968` | 10 | **4.522503** | **4.271024** |
| fixed | 10 | 3.954165 | 3.7883291 |
| fixed | 1000 | 3.954165 | 3.7883291 |

At HEAD, one mid-run sample at step 10 makes the loss jump by 0.57 and it
never recovers. The fixed binary's `-s 10` run is bit-identical to the
no-sample run. The 124M FineWeb run
([`gpt2_124m_fineweb_training_run.md`](gpt2_124m_fineweb_training_run.md))
used `-s 20000` for 19,552 steps, so it never sampled mid-run and its training
is unaffected. Only its end-of-run sample (after the last step) was garbage.
HellaSwag eval (`infer_gpt2 --eval`) uses one fixed shape and is unaffected.

## The fixes

**Attention (`llmm/attention.mojo`).** `KVCache` now records the `(B·nh, T)`
for which each probability destination's upper triangle is known to be zero:
`gemm_att_zero_{bh,t}` for the shared scratch, and
`att_probs_zero_{bh,t}` plus a per-layer bitmask for the store-P slices.
When a call's shape differs, `attention_fwd_gemm` memsets that plane (or that
layer's slice) once before the softmax and records the new shape. Training
runs one shape, so this fires once per layer on the first forward and never
again. The fixed binary's 40-step training losses and val loss are
bit-identical to HEAD's when no mid-run sample happens, and step time is
unchanged (~137 ms). Generation pays a memset of `L·nh·t²` elements per
token, which is negligible next to its full-recompute forward.

**Sampling (`train_gpt2.mojo`, `infer_gpt2.mojo`, `llmm/sampler.mojo`).**
`sample_softmax` is now generic over the logits dtype and widens each element
to fp32 on read. The duplicated readback-and-sample code moved into one method,
`GPT2.sample_next_token(t, coin)`, which both the training loop and
`infer_gpt2` call. There is no rebind left to get wrong, and the two paths can
no longer drift apart (the drift is how f0da883's fix missed the training
loop).

## Verification

- **Reported command** (`make train-gpt2-124m-bf16 STEPS=40 ARGS="-v 1000 -s 1000 -m 1"`),
  fixed binary (mtime checked after rebuild): val loss 3.6853046 (unchanged),
  sample:
  ```
  Weary servant:
  It's your sharpness; which you have all in store.
  BUCKINGHAM:
  Make it look as unnatural.
  ```
  `build/infer_gpt2_bf16` on the `model_40.bin` saved by the same run
  prints the identical text, so the training-loop and inference paths now
  agree.
- **Teacher-forced vs HF fp32** (first 1024 val tokens), fixed `forward()`:
  T=1024 4.67535 (HF 4.67524), T=64 5.15021 (HF 5.15165), T=1 7.30760
  (HF 7.30761), and T=1024 re-run after the small-T calls is bit-identical.
  Var-T vs full-T logits for row t-1 agree to ≤0.18 at |logit|≈100 for
  t = 1..39, which is consistent with TF32 at different GEMM shapes.
- **Sampled tokens vs HF fp32**, same xorshift* RNG (seed 1337) and llm.c
  sampling semantics: the first 8 tokens match exactly
  (`51 615 396 735 5451 198 50256 3260`, "Tavistock Island\n<|endoftext|>After").
  Token 9 differs because its coin lands near a CDF boundary, and after that
  the two trajectories legitimately diverge. The pretrained model emitting
  `<|endoftext|>` often for this seed is genuine GPT-2 behaviour; HF's own
  sample does the same.
- `make verify-gpu` (model-level parity with the PyTorch debug state): all
  tensors OK, 10/10 loss steps OK.
- `make lint` clean.

## Regression tests

- `tests/test_attention_shape_change.mojo` (GPU, runs under `make
  test-mojo`). It replays the generation pattern T = 64, 9, 8, 33, 1, 17, 64
  through **one** `KVCache`, in both store-P and scratch modes, fp32 and
  bf16, checking each call against an fp64 host causal-attention reference
  and requiring the last T=64 output to be bit-identical to the first.
  Every pre-existing attention test built a fresh cache per call, which is why
  none could see this. **Shown to catch the bug:** built against the
  pre-fix `attention.mojo` (`15d0968`), all 4 cases fail at the T=9 call with
  max_err 0.31–0.97 against tolerances of 1e-2 (fp32) / 3e-2 (bf16). Post-fix
  max_err is ≤4.7e-4 (fp32) / ≤6.2e-3 (bf16).
- `tests/test_infer_gpt2_generation_smoke.py::test_fp32_generation_matches_hf_reference`
  is an end-to-end check that `build/infer_gpt2 gpt2_124M.bin 9 1337` starts
  with the HF reference prefix above. The pre-fix binary printed
  "ToughTough things all...". It skips if the binary or checkpoint is absent,
  like the crash smoke test next to it, which only checks the exit code and so
  could never have flagged word salad.
- `tests/test_sampler.mojo::test_bf16_logits_sample_like_widened_fp32` checks
  that the bf16 sampler instantiation samples exactly like the fp32 sampler on
  the widened values.

## Lessons

- A "zero once, then write only the lower triangle" optimisation turns a
  buffer's history into part of its contract. Whoever owns the buffer has to
  re-establish the invariant whenever the layout changes. The shape it was
  zeroed for is now stored next to the buffer.
- Loss being correct only validates the shapes training uses. Generation was
  the only caller of `forward()` at varying T, and it had no correctness gate,
  only a "did not crash" gate.
- A fix applied to one of two copies of the same code is half a fix.
  Deduplicating (`sample_next_token`) is what keeps it fixed.

---

## AI use statement

This bug was root-caused and fixed with AI assistance via Claude Code (Claude
Opus agent in an isolated git worktree), under the direction of Evan Owen.
