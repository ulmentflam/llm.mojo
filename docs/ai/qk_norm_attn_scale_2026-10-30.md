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

See the report section appended below when results are in.

## AI use statement

Written with AI assistance (Claude Code, Sonnet 5.5 implementation agent), directed by Evan Owen.
