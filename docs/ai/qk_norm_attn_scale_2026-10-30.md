# W7: QK-norm and fixed attention scale (2026-10-30)

Work item W7 of [`modded_nanogpt_adoption_plan_2026-10-30.md`](modded_nanogpt_adoption_plan_2026-10-30.md).
Branch `mn/w7`. Opt-in: `-D LLMM_QK_NORM=1` (build) and `-attn_scale <f>` (run;
env `LLMM_ATTN_SCALE`; 0 = `1/sqrt(head_dim)`). Defaults leave every path unchanged.

## Design

- **One kernel pair, `llmm/qk_norm.mojo`.** `qk_rows_fwd` / `qk_rows_bwd` apply an
  op to every q row and k row of the head-layout buffers `[B, NH, T, hd]` that
  `attention_fwd` produces by splitting the qkv activation (row r = (b*NH+h)*T+t,
  so a rotary op (W8) has t = r % T). Called from the outer `attention_fwd` /
  `attention_bwd` in `llmm/attention.mojo`, between split and core attention
  (forward) and between core backward and the merge into d_qkv (backward).
- **Forward:** `q <- q_mul * rmsnorm(q)`, `k <- rmsnorm(k)`, `rmsnorm(x) = x *
  rsqrt(mean(x^2) + 1e-6)` over head_dim, no affine, fp32 math, one warp per row.
- **Fixed scale without touching the attention kernels.** The kernels keep
  dividing q.k by `sqrt(hd)`; `q_mul = attn_scale * sqrt(hd)` is folded into q,
  so the effective scale is `attn_scale`. This deviates from the item text
  ("thread the scale into every kernel"), deliberately: seven kernels (CPU,
  flash, gemm, fwd and bwd, Metal portable) stay byte-identical, and the QK-clip
  max-logit hook still sees the true post-scale logit. Without QK-norm,
  `-attn_scale` alone just scales q the same way.
- **Backward** needs no stored statistics: the pre-norm rows stay in `acts.qkv`,
  so rstd and the unit row u are recomputed with the forward's arithmetic.
  `dy = q_mul * dq'`, `dx = rstd * (dy - u * mean(dy * u))`, in place on d_q/d_k.
- **Measurement.** `-qk_log 1` (automatic with QK-norm on a GPU target) tracks the
  per-head max attention logit without applying QK-clip; the step event gets
  `qk_max` (verbose). `model.enable_qk_clip(0.0, track_only=True)`.
- `infer_gpt2` and the infer/dump builds take `MOJO_DEFINES` and
  `LLMM_ATTN_SCALE`.

## Reference and fixtures

`train_gpt2.py`: `GPT2_QK_NORM=1`, `GPT2_ATTN_SCALE=<f>`, same eps. The debug
state is written from a scratch directory (it also writes `gpt2_124M*.bin`, which
are symlinks to shared weights in a worktree):
`GPT2_QK_NORM=1 [GPT2_ATTN_SCALE=0.12] python train_gpt2.py --device cpu
--num_iterations 0 --input_bin <tiny_shakespeare_val.bin>`, renamed to
`gpt2_124M_debug_state_qknorm[_s012].bin` (gitignored, 647 MB). Trajectories from
`scripts/gen_expected_losses.py --state-file ...` with the same env. `test_gpt2.mojo`
under `-D LLMM_QK_NORM=1` reads the qknorm state and an embedded trajectory;
`LLMM_DEBUG_STATE`, `LLMM_ATTN_SCALE`, `LLMM_EXPECTED_LOSSES` select the 0.12 arm.
The stock `make test-fixtures` is the optimizer fixture set and is not involved.
Caveat: the pretrained weights were not trained with QK-norm, so every logit is
< -10 and the logits check compares 0 elements; loss, all 16 gradient tensors and
the 10-step trajectory are the real gates.

## Verification

All on branch `mn/w7`, fp32 gates with `-D LLMM_NO_TF32=1`:

| check | result |
|---|---|
| `make verify-gpu`, flag off, untouched debug state | pass: loss 5.2700114 vs 5.2699833, 16/16 gradient tensors, 10-step trajectory within 0.01 (same numbers as before the change) |
| `make verify-gpu MOJO_DEFINES="-D LLMM_QK_NORM=1"` (qknorm state, PyTorch trajectory) | pass: loss 5.7229743 vs 5.722967, 16/16 gradient tensors (incl. qkv weight/bias, both attn weights), steps within 0.01 |
| same, `LLMM_ATTN_SCALE=0.12` + the 0.12 state | pass: loss 5.7112312 vs 5.7112207, 16/16, steps within 0.01 |
| `tests/test_attention_equivalence.py` + `test_split_merge_equivalence.py` on GPU (`MAX_USE_ACCELERATOR=1`) | 66 passed, includes 10 new `qkn_*`/`scale_only` chain cases (fwd out, dq, dk, dv vs torch autograd; fp32 and bf16; T=7, 9, 17, 37, 64) |
| mutant: backward without the `-u*mean(dy*u)` term | the 3 fp32 `qkn_*` cases at hd 32/64 fail, so the gate can see a wrong norm backward |
| `make test-mojo` | `tests result=pass files=26 passed=223 failed=0` |
| bf16 generation, `-D LLMM_QK_NORM=1` infer build, `gpt2_124M_bf16.bin`, seeds 1-8 and `LLMM_ATTN_SCALE=0.12`, every T = 1..39 (the f0da883 crash was at t=9) | all exit 0, no CUDA_ERROR |
| bf16 5-step smoke, qknorm build, `-attn_scale 0.12` | runs; `qk_max` logged (about 5.0 at init) |

Not verified: Metal path compile (flag-off code is unchanged except new default
arguments), multi-rank, fp8/fp4 builds with the flag on, and any interaction with
W8 rotary (not yet on main when this was written).

## Ranking jobs

`w7_qknorm.sh` (QK-norm, default scale) and `w7_qknorm_s012.sh` (QK-norm,
`-attn_scale 0.12`): base recipe plus `-qk_log 1`; logs `sweeps/w7_qknorm*.jsonl`,
per-step `qk_max` in the verbose log. Results to be added after the runs.

## AI use statement

Written with AI assistance (Claude Code, Sonnet 5.5 implementation agent), directed by Evan Owen.
