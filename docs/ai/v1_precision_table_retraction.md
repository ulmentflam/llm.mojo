# Retraction: the v1 six-arm precision comparison

This is the retraction notice that used to sit inline under the first six-arm
precision table in [`README.md`](../../README.md), moved here to keep the README
readable. The table it retracts is the pre-v2 one (124M/774M x bf16/fp8/nvfp4,
val loss and HellaSwag, published 2026-07-25). The README keeps a one-line
pointer to this file so those numbers are never read as sound.

The replacement results are generated into the README's `v2-precision-table`
block by `scripts/update_readme_results.py`, from the re-trained arms tracked in
`docs/ai/v2_arm_results/*.json`.

---

**Retracted (2026-07-27): most of these runs trained with some biases frozen, and not the same ones, so the comparison is confounded.** A scratch-buffer overrun in the fused dbias kernel (`llmm/matmul.mojo`, fixed in `e747faf`) wrote past its scratch allocation and, on this box, landed on the kernel's own arrival counters — so no block ever observed the last-arrival condition and `d_bias` was never written. The repo's fixtures could not see it: at their B=4/T=64 shape the out-of-bounds writes deposit `0.0`, which is exactly what the counters should hold. See [`docs/ai/dbias_scratch_overrun_silent_zero_bug.md`](docs/ai/dbias_scratch_overrun_silent_zero_bug.md).

The damage is measurable directly from the published checkpoints, with no re-run and no instrumentation: GPT-2 initialises every bias to exactly `0` (`train_gpt2.mojo`), so a bias still bit-exactly zero after 22,345 optimizer steps never received a gradient. Counting non-zero entries per bias tensor:

| Arm | `qkvb` | `attprojb` | `fcb` | `fcprojb` | layernorm biases |
|---|---|---|---|---|---|
| 124M bf16 | **0** / 27,648 | 768 / 9,216 | **0** / 36,864 | 2,304 / 9,216 | all trained |
| 124M fp8 | 27,648 / 27,648 | 9,216 / 9,216 | 36,864 / 36,864 | 9,216 / 9,216 | all trained |
| 124M nvfp4 | **0** / 27,648 | **0** / 9,216 | **0** / 36,864 | 3,072 / 9,216 | all trained |
| 774M bf16 | **0** / 138,240 | 46,080 / 46,080 | **0** / 184,320 | 46,080 / 46,080 | all trained |
| 774M fp8 | 138,240 / 138,240 | 46,080 / 46,080 | 184,320 / 184,320 | 46,080 / 46,080 | all trained |
| 774M nvfp4 | **0** / 138,240 | 46,080 / 46,080 | **0** / 184,320 | 46,080 / 46,080 | all trained |

Three things follow, and the third is why the table above is retracted rather than merely annotated.

First, the layernorm biases trained normally everywhere. They have their own backward and never touch this kernel, which is the control that rules out a dead optimizer or a checkpoint-writer fault.

Second, the wide biases are the ones that died. Scratch demand is `FUSED_ROW_BLOCKS × out_channels`, and only `qkvb` (`3C`) and `fcb` (`4C`) exceed the old cap; the `C`-wide `attprojb`/`fcprojb` fit. They still get caught at 124M, and only partially — 768 of 9,216 is exactly one layer of twelve — because they share the counter array that the wide calls had already poisoned. Which layers survive drifts across a run (`fcprojb` in the 124M bf16 arm: 1,536 non-zero at step 1,000, 2,304 by step 22,345), so this was a race, not a clean freeze.

Third — the part that invalidates the comparison — **the two fp8 arms are undamaged.** All three precisions call the same bf16 `matmul_bias_bwd`, so this is not a code-path difference; an out-of-bounds write hits whatever the allocator happened to place next, and in those two builds it evidently was not the counters. The earlier claim here, that all six arms shared the defect and the precision comparison therefore stood, was wrong: fp8 trained with working biases while bf16 and nvfp4 largely did not. That confound points the same way as the headline result — it is a candidate explanation for fp8 appearing to edge out bf16 at 774M — so that finding cannot be trusted either. All six arms are being re-trained; this section will be replaced with results that are actually comparable.

## AI use statement

Written with AI assistance (Claude Code, Opus 5), directed by Evan Owen. The
text below the header is the original retraction verbatim from the README; only
its blockquote markers were removed in the move.
