# modded-nanogpt adoption plan: work items for the Muon track (2026-10-30)

Status at writing: **plan only, nothing implemented.** This is the post-Muon
follow-up to [`speedrun_techniques_research.md`](speedrun_techniques_research.md),
which triaged Keller Jordan's
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) from the
FP8/FP4 kernel angle and marked every training-recipe and architecture item as
out of scope for the llm.c parity benchmark. Muon has since landed as an
opt-in optimizer ([`muon_muonclip_implementation_2026-10-29.md`](muon_muonclip_implementation_2026-10-29.md)),
so there is now a second track, "fastest to loss", on which those items are
in scope. This doc turns them into work items an agent can pick up and run
without re-deriving the research.

Every speedrun fact below was read from the README record table, the embedded
`train_gpt.py` in the 2025-01-26 record log
(`records/track_1_short/2025-01-26_BatchSize/0bdd5ee9-....txt`, the last record
whose script is still a single readable file) and the Muon write-up
(<https://kellerjordan.github.io/posts/muon/>). Record numbers refer to the
README table; the "gain" is the wall-clock delta of that record on 8xH100 and
is evidence of relative value only, not a prediction for the GB10.

## 1. Ground rules (apply to every item)

- **Opt-in only.** Every change ships behind a runtime flag or a `-D` build
  constant that defaults to today's behaviour. The AdamW/llm.c parity path
  must stay bit-identical: `make verify` (cpu and gpu gates) and the
  deterministic 3-step CPU run used in the Muon campaign are the proof.
- **One `git worktree` per item**, branch named `mn/<item>` (e.g. `mn/w2-momentum-warmup`).
  Commit on the branch; do not push or merge to `main`. The coordinator
  verifies load-bearing claims (rebuild, rerun) before merging. `make lint`
  gates every commit.
- **Ranking protocol** is the one from
  [`muon_hparam_sweep_124m_2026-10-30.md`](muon_hparam_sweep_124m_2026-10-30.md):
  single GB10, `make train-bf16`, seed 42, 300 steps at 131k tokens/step,
  `sweeps/<item>_<tag>.jsonl` via `LLMM_LOG_FILE`, table via
  `python3 scripts/compare_runs.py --table sweeps/<item>_*.jsonl`. Score is val
  loss at step 300. Single-seed differences under about 0.02 nats are noise;
  anything that will go into a recommended recipe needs the three init seeds
  (`-D LLMM_INIT_SEED=42|1|2`) as in the README table. Base recipe:

  ```
  -e d12 -i 'data/.fineweb10B/fineweb_train_*.bin' -j 'data/.fineweb10B/fineweb_val_*.bin' \
  -b 32 -t 1024 -d 131072 -x 300 -v 100 -m 20 -s 0 -qk_clip 0 \
  -opt muon -l 4.8e-3 -mu 0.9 -c 0.1 -gc 1.0 -u 50 -q 0.0
  ```

  (Muon tuned lr and momentum; weight decay and clip kept on because the sweep
  found no evidence either way at 300 steps and they matter at length.)
- **Check binary mtime** after every build before trusting a run. Use the `make`
  targets, not raw binaries.
- **Architecture items (W5 to W9) need the PyTorch reference to match.**
  `train_gpt2.py` is the reference for `make verify` and for the equivalence
  fixtures (`make test-fixtures`). Each architecture flag gets a matching flag
  in the reference, the fixture set is regenerated under both settings, and the
  gradient-tensor comparison in `test_gpt2.mojo` must pass with the flag on
  and off. A passing loss curve is not a substitute (see the weak-gates
  incident in [`v1_precision_table_retraction.md`](v1_precision_table_retraction.md)).
- **Write the result into this doc's section 6** (table row per item: flag,
  val@300 at seed 42, three-seed mean if run, verdict) and into the item's own
  `docs/ai/` entry if the work was non-trivial.

## 2. Already adopted (do not redo)

| speedrun element | where it lives here |
|---|---|
| Muon on hidden matrices, 5-step Newton-Schulz, coefficients (3.4445, -4.7750, 2.0315), bf16 iteration, Nesterov | `llmm/muon.mojo` |
| Q, K, V orthogonalized as three separate C×C matrices (the Muon post's "apply to Q, K, V separately") | `MuonGroup` row blocks of `qkv_weight`, `llmm/muon.mojo` |
| AdamW on embeddings, LayerNorm, biases | `_update_hybrid` in `train_gpt2.mojo` |
| Vocab padded to a multiple of 128 (50304) | `V_p` everywhere |
| Attention logit control (QK-clip, the reactive form) | `llmm/qk_clip.mojo`, `-qk_clip` |
| Learning rate in the speedrun's range: our Moonlight scale turns `-l 4.8e-3` into 0.027 on square matrices and 0.053 on fc; the speedrun uses 0.05 and 0.1 (`lr * max(1, rows/cols)^0.5`); our sweep turned over at 9.6e-3, which is their value | sweep doc |

## 3. Work items

Each item: what, evidence, code anchors, flag, gate, deliverable.

### W1. Separate learning rates for the Adam-managed parameters

**What.** Under `-opt muon`, `-l` is shared by the Muon matrices and by the
AdamW ranges (wte, wpe, LayerNorm, biases). The sweep's win from raising `-l`
from 6e-4 to 4.8e-3 therefore also raised the AdamW rate 8x, which is an
unresolved confound. Add a multiplier for the AdamW ranges so the two can be
set independently, then ablate.

**Evidence.** The speedrun runs three Adam groups at very different rates and
one Muon group (2025-01-26 script):

| group | lr | betas | eps | weight decay |
|---|---:|---|---|---|
| Muon matrices | 0.05 | momentum 0.95 (see W2) | | none |
| embeddings (`embed`, value embeds) | 0.6 | (0.8, 0.95) | 1e-10 | none |
| lm_head | 0.008 | (0.8, 0.95) | 1e-10 | none |
| scalars (ndim < 2) | 0.04 | (0.8, 0.95) | 1e-10 | none |

The intuition: Adam-managed parameters in a Muon run want a rate far above the
AdamW baseline's 6e-4, embeddings most of all, because their rows get sparse
gradients. Record 20 includes "lower Adam eps".

**Code anchors.** `_update_hybrid` and `_update_sharded_muon` in
`train_gpt2.mojo` build one `AdamWConfig` and one `MuonConfig` from the same
`config.learning_rate`; the AdamW ranges come from `_adamw_ranges()`. The CLI
parser is the `elif flag ==` chain near `-mu` / `-qk_clip`; args struct near
`learning_rate: Float32  # -l`. The scheduler (`llmm/scheduler.mojo`) returns a
single lr per step; apply the multiplier after it.

**Flag.** `-l_adam <mult>` (env `LLMM_LR_ADAM_MULT`), default 1.0 so today's
runs are unchanged. Optional second knob for Adam betas/eps on the AdamW
ranges (`-b1 -b2 -eps`), default to current AdamW values. With the tied `wte`
the split is "matrices vs everything else"; the 0.6 vs 0.008 embedding/head
split needs W9.

**Gate.** Multiplier 1.0 must reproduce the current tuned run bit-identically
(same step losses, same log). `tests/test_muon*.py` unchanged.

**Deliverable.** Ablation table at 300 steps:
Muon 4.8e-3 with AdamW ranges at 6e-4 (mult 0.125), 4.8e-3 (mult 1, today),
and a ladder above (mult 2, 4, 8). Then, if the embedding direction wins,
rerun with W2 and W3 applied. State plainly how much of the sweep's win was
the AdamW side effect.

### W2. Momentum warmup

**What.** Ramp Muon momentum linearly from 0.85 to 0.95 over the first 300
optimizer steps, then hold.

**Evidence.** Record 9 ("momentum warmup", with value and embedding skips and
softcap, 10.8 to 8.2 min). Script:
`frac = min(step/300, 1); momentum = (1-frac)*0.85 + frac*0.95`, Muon group
only. Our sweep found 0.9 beat 0.95 at 300 steps, at the edge of the tested
range; that is the horizon where low momentum wins, and a long run will want
0.95. Warmup reconciles both.

**Code anchors.** `self.muon_momentum` is read once per step in
`_update_hybrid` / `_muon_config`. Add `muon_momentum_start` and
`muon_momentum_warmup_steps` to the args struct and compute the step's value
next to the lr-scheduler call.

**Flag.** `-mu_start <f>` and `-mu_warmup <steps>` (env
`LLMM_MUON_MOMENTUM_START`, `LLMM_MUON_MOMENTUM_WARMUP`). Defaults: start equal
to `-mu`, warmup 0, which reproduces today. Log the effective momentum in the
per-step JSON line at `verbose`.

**Gate.** Warmup 0 bit-identical to today. Checkpoint resume must reproduce
the ramp (the step counter is in the state file; no new state is needed).

**Deliverable.** 300-step ranking of `-mu 0.9` (today), `-mu_start 0.85 -mu 0.95 -mu_warmup 300`,
`-mu_start 0.85 -mu 0.9 -mu_warmup 300`. Carry the winner into the 2000-step
run in [`muon_vs_adamw_124m_2026-10-29.md`](muon_vs_adamw_124m_2026-10-29.md) section "2000 steps".

### W3. Schedule: no warmup, flat, linear cooldown to a nonzero floor

**What.** Add a cooldown-fraction and a linear decay shape to the WSD
scheduler, and test zero lr warmup under Muon.

**Evidence.** The speedrun schedule is `w = min(t_remaining / cooldown_frac, 1); lr_mult = w + (1-w)*0.1`
with no warmup, `cooldown_frac` 0.4 in Jan 2025, 0.45 at record 26. Record 19
moved the floor from 0 to 0.1; record 72 raised the minimum lr again. The
same multiplier applies to every param group. Zero warmup is tolerable because
an orthogonalized update has bounded norm per step, so there is nothing to
guard against early.

**Code anchors.** `get_learning_rate_wsd` in `llmm/scheduler.mojo` hard-codes
the decay point at 4/5 of training and the `1 - sqrt` shape. `-k` selects the
scheduler, `-u` warmup, `-q` final fraction.

**Flag.** `-k wsd` gains `-cooldown <frac>` (default 0.2, today's value) and
`-decay_shape sqrt|linear` (default sqrt). `-q 0.1` already gives the floor.

**Gate.** Defaults reproduce the current `wsd` curve exactly (unit test on the
scheduler: a table of step to lr for both shapes).

**Deliverable.** 300-step ranking of: cosine `-u 50 -q 0` (today);
`-k wsd -u 0 -cooldown 0.4 -decay_shape linear -q 0.1`; same with `-u 50`;
same with `-cooldown 0.45`. Note that at 300 steps the cooldown is 120 steps,
so also report the 600-step result for the winner against the README table.

### W4. Weight decay policy for long Muon runs

**What.** No code. A decision and a measurement: keep `-c 0.1` for any run
past the 300-step ranking horizon, and measure it.

**Evidence.** The Jan 2025 speedrun Muon had no weight decay at all; decay
only became a record once runs were long enough for weight norms to drift
(records 43 "cautious weight decay with schedule" and 50, extending it to the
Adam params). Our sweep's "wd 0 wins by 0.008 nats" is inside single-seed
noise at 300 steps. The speedrun ties decay to the lr multiplier, which our
decoupled form `lr * (scale*O + wd*W)` already does.

**Deliverable.** In the 2000-step runs, add a `-c 0` arm next to `-c 0.1` and
log the per-matrix RMS weight norm at `verbose` (one line per 100 steps) so
the drift is visible, not inferred. Cautious weight decay (decay only where
update and weight agree in sign) is a follow-up only if `-c 0.1` loses.

### W5. Zero-init of the residual projections

**What.** Initialize `attn_proj_weight` and `proj_weight` to zero instead of
N(0, 0.02/sqrt(2L)). Every block then starts as the identity on the residual
stream.

**Evidence.** Record 5 (pad embeddings, ReLU², zero-init projections, QK-norm:
22.3 to 15.2 min, the largest non-optimizer jump in the table), README bullet
"zero-init projections (muP-like)". Script: `c_proj.weight.detach().zero_()`
for both attention and MLP, plus `lm_head`. Record 64 later moved to "tuned
nonzero attn V and O init", so zero and small both beat the GPT-2 default. The
head cannot be zeroed here while `wte` is tied (it would zero the embedding);
that part waits for W9.

**Code anchors.** `gpt_build_from_descriptor` init loop in `train_gpt2.mojo`,
the branch `if i == Parameters.attn_proj_weight or i == Parameters.proj_weight:
scale = INIT_WEIGHT_STD * residual_scale`. `INIT_WEIGHT_STD`, `INIT_RNG_SEED`
are comptime constants next to `LLMM_INIT_SEED`. The reference init is in
`train_gpt2.py` (`_init_weights`, the `LLMC_RESIDUAL_SCALE_FLAG` path).

**Flag.** `-D LLMM_ZERO_INIT_PROJ=1` build constant (init is comptime-seeded,
so a build flag is the honest form). Default 0. Note in the log setup table.

**Gate.** With the flag on, the from-scratch weights must match the reference
with the same flag (`make verify-gpu` after fixture regeneration with the
reference's matching flag). Checkpoint format unchanged (no new tensors).

**Deliverable.** 300-step ranking on and off, three seeds if it wins by more
than 0.02. Optional second arm: small nonzero (`INIT_WEIGHT_STD * 0.1`) per
record 64.

### W6. ReLU² in place of GELU

**What.** MLP activation `relu(x)^2`, forward and backward. No parameter or
shape change.

**Evidence.** Record 5 bundle; README bullet "Rotary embeddings, QK-Norm, and
ReLU²". Script: `x = F.relu(x).square()`, hidden dim stays 4C.

**Code anchors.** `llmm/gelu.mojo` (`gelu_fwd`, `bias_gelu_fwd`, `gelu_bwd`
and their cpu/gpu/Metal kernels). The trainer calls the fused `bias_gelu_fwd`
on the fc output; the backward reads the pre-activation. Add `relu2_*`
siblings with the same signatures and select by a comptime constant at the
two call sites, so the fp8/fp4 lowp paths and the Metal port are untouched
when the flag is off. The reference gets `GPT2_ACT=relu2` in `train_gpt2.py`
and the equivalence test (`tests/test_gelu_equivalence.py`) gains the relu2
case.

**Flag.** `-D LLMM_MLP_ACT=gelu|relu2`, default gelu.

**Gate.** `make verify-gpu` with regenerated fixtures under relu2; the gelu
fixtures must be byte-identical to today's. `make test-mojo` and the
equivalence test pass under both.

**Deliverable.** 300-step ranking; note any change in step time (the
activation is memory-bound, so none is expected) and whether the fp8 path's
amax statistics move (ReLU² activations are half zeros and unbounded above).

### W7. QK-norm with a fixed attention scale

**What.** RMS-normalize q and k per head (over head_dim, no affine) before
the attention scores, and replace the `1/sqrt(head_dim)` scale by a constant.
The proactive alternative to QK-clip.

**Evidence.** Record 5 bundle (QK-norm); record 20 "attention scale" (0.12 in
the Jan 2025 script, passed to flex_attention in place of `head_dim**-0.5`).
Script: `q, k = norm(q), norm(k)` with `F.rms_norm(x, (x.size(-1),))`,
applied before rotary. Our sweep ran with `-qk_clip 0` and nothing diverged at
lr 4.8e-3, so there is no current stability emergency; the value is the
freedom to raise lr and the removal of the max-logit tracking kernel.

**Code anchors.** The qkv activation is `[B, T, 3C]` straight from the qkv
matmul; the attention kernels in `llmm/attention.mojo` (`attention_fwd_gemm`,
`_attention_flash_fwd_gpu`, `attention_bwd_*`) read q and k rows from it and
apply the scale inside. QK-norm is a per-row kernel over the q and k thirds
(forward in place or into a copy; backward is the RMSNorm backward with no
gamma, fused with the existing dqkv write). `llmm/layernorm.mojo` already has
the row-norm building blocks. The max-logit hook from `llmm/qk_clip.mojo`
stays valid and can double as the measurement that QK-norm bounds the logits.

**Flag.** `-D LLMM_QK_NORM=1` and `-attn_scale <f>` (default 0, meaning
`1/sqrt(head_dim)`). Reference: `GPT2_QK_NORM=1` in `train_gpt2.py`.

**Gate.** `make verify-gpu` under the flag with regenerated fixtures: the
qkv gradient and both attention-weight gradients must match the reference.
`tests/test_attention_equivalence.py` gains the normalized case. Also run the
bf16 misaligned-T regression (T not a multiple of 8) from
[`bf16_generation_misaligned_address_bug.md`](bf16_generation_misaligned_address_bug.md)
because the new kernel touches the same rows.

**Deliverable.** 300-step ranking: off; on with default scale; on with
`-attn_scale 0.12`. Report the max attention logit per 100 steps from the
QK-clip hook for each arm.

### W8. Rotary position embeddings

**What.** Apply rotary to q and k inside each attention layer and drop `wpe`
from the forward (keep the tensor so the parameter layout, checkpoint and
ZeRO shards do not change; it is simply not added and gets zero gradient).
Half-truncated rotary, base 1024: `angular_freq = (1/1024)^linspace(0, 1, head_dim/4)`
on the first half of the head dims, zero frequency on the second half.

**Evidence.** Record 2 ("tuned LR and rotary embeddings", 45 to 31.4 min);
record 17 "improve rotary" is the half-truncation. The README lists rotary
first among the techniques.

**Code anchors.** Encoder: `llmm/encoder.mojo` adds `wte + wpe` rows; the
trainer's encoder forward/backward calls (`_encoder_backward_row_sparse` and
the dense path) need a flag that skips the `wpe` add and the `dwpe` write.
Attention: a rotate-half kernel over the q and k thirds of the `[B, T, 3C]`
qkv activation, run after W7's norm if both are on, with a precomputed
`[T, head_dim/2]` cos/sin table in the activation arena; backward applies the
inverse rotation to dq and dk before the qkv matmul backward. Generation
(`infer_gpt2.mojo`) uses the same forward and needs the table sized to the
max generated position. Reference: `GPT2_ROTARY=1` in `train_gpt2.py`, with
the same half-truncation.

**Flag.** `-D LLMM_ROTARY=1`, default 0. Under the flag `wpe` is excluded
from the AdamW ranges (zero gradient, but excluding it keeps the optimizer
state honest) and from the init draw. Document in the README that rotary
checkpoints are not loadable by a non-rotary build (store the flag in the
checkpoint header: bump `STATE_VERSION` in `llmm/checkpointing.mojo` with a
flags word, refuse on mismatch as the optimizer-kind check already does).

**Gate.** `make verify-gpu` under the flag with regenerated fixtures. The
T-not-multiple-of-8 bf16 regression. `make test-mojo` for the checkpoint
round trip with the new header word under both settings. ZeRO-1/2/3 CPU
multi-rank tests (`tests/test_zero_muon.mojo`) unchanged because the layout is
unchanged.

**Deliverable.** 300-step ranking off and on; then on with W7 (the speedrun
never runs rotary without QK-norm). Also a generation smoke
(`tests/test_infer_gpt2_generation_smoke.py`) from a rotary checkpoint.

### W9. Untied LM head, with "tie early, untie late" as a variant

**What.** A separate `lm_head` weight `[V_p, C]` so the head and the
embedding can have different learning rates (W1's 0.6 vs 0.008) and the head
can be zero-initialized (W5). The variant: train tied for the first two
thirds of the run, then copy `wte` into a fresh head and untie.

**Evidence.** Record 8 ("untied embedding and head", 12.0 to 10.8 min);
record 51 retied them (with other changes); record 53 "untie embed/lm_head
at 2/3 of training" is the current form. The Jan 2025 script has separate
`self.embed` and `self.lm_head`, head zero-initialized, head FP8.

**Why this is the largest lift.** It is the only item that adds a parameter
tensor. `NUM_PARAMETER_TENSORS = 16` is baked into the descriptor, the
gradient buckets, the ZeRO shard maps, the Muon owner-space layout, the
checkpoint format and the 16-tensor gradient comparison in `test_gpt2.mojo`.
The LM-head vocab tiling (`_lm_head_wte_tile`, `LLMM_LM_HEAD_VOCAB_TILES`) and
the row-sparse `wte` gradient were built precisely because the tied tensor
had two consumers; untying makes the head a plain dense GEMM weight again
(tiling still applies, the row-sparse path then serves only the encoder).
Memory: +147 MiB fp32 master plus AdamW `m`/`v` at 124M, which is the floor
the ZeRO wte campaign ([`zero_wte_deresidency_campaign_2026-07-27.md`](zero_wte_deresidency_campaign_2026-07-27.md))
spent weeks lowering; this must be measured, not assumed acceptable.

**Code anchors.** `Parameters` enum and `NUM_PARAMETER_TENSORS` in
`train_gpt2.mojo`; `_lm_head_wte_tile` and the "Bucket 1, wte (LM-head
contribution)" backward block; `_adamw_ranges()`; the descriptor /
`param_sizes` list; `llmm/checkpointing.mojo` (`STATE_VERSION`);
`llmm/safetensors.mojo` for HF import (HF GPT-2 ties, so import copies `wte`
into the head). ZeRO: `llmm/zero.mojo` shard segments and the owner-space
groups in [`zero_sharded_muon_design_2026-10-30.md`](zero_sharded_muon_design_2026-10-30.md).
Reference: `GPT2_UNTIED=1` in `train_gpt2.py` (remove the
`self.transformer.wte.weight = self.lm_head.weight` line under the flag).

**Flag.** `-D LLMM_UNTIED_HEAD=1`, default 0, with `NUM_PARAMETER_TENSORS`
becoming 17 under it. Runtime `-untie_at <frac>` for the variant (default 0,
meaning untied from step 0; 1.0 meaning never, which is the tied build's
behaviour expressed in the untied layout). Under the flag the head joins W1's
Adam groups at its own multiplier (`-l_head <mult>`).

**Gate.** Tied build byte-identical to today (verify, test-mojo, ZeRO tests,
checkpoint round trip, HF import). Untied build: `make verify-gpu` with a
17-tensor fixture set; ZeRO-1/2/3 CPU multi-rank bit-exactness tests extended
to the 17th tensor; checkpoint resume across the untie point reproduces the
continuous run to the AdamW resume gap (about 3e-5). Memory accounting line
in the log for both builds.

**Deliverable.** 300-step ranking: tied (today); untied from step 0 with head
zero-init; untied from step 0 with the W1 split (embedding high, head low).
Then a 2000-step comparison of untied-from-0 against untie-at-2/3. Report
peak memory and step time for each.

## 4. Dependency order and tracks

```
W1 (Adam lr groups) ──┐
W2 (momentum warmup) ─┼── recipe track, independent of each other, merge first
W3 (schedule) ────────┤
W4 (wd policy) ───────┘   (measurement only, rides on the 2000-step runs)

W5 (zero-init) ───┐
W6 (ReLU²) ───────┼── architecture track A: independent, each needs reference flag + fixtures
W7 (QK-norm) ─────┘

W8 (rotary) ──── architecture track B: after W7 (shares the q/k row kernel and the T-alignment regression)
W9 (untied head) ─ architecture track C: after W1 (needs the per-group lr) and W5 (head zero-init)
```

Three agents can start today on W1, W2 and W3 in separate worktrees; a
fourth on W5 and W6 together (same init-loop and reference files, small
diffs); a fifth on W7. W8 and W9 start once W7 and W1 have landed on `main`.
The coordinator re-runs the base recipe on each merged tree and keeps the
section 6 table current.

Merge order: recipe track first (no reference changes, no fixture churn),
then architecture A, then B, then C. After every merge, rerun
`make verify` and the 5-step W=1 Muon smoke
(`make train ARGS="-opt muon -x 5 -v 0 -s 0"`) and record the losses next to
the ones in the Muon campaign report.

## 5. Headline metric for the track

The speedrun's unit of evidence is tokens to a fixed target, judged across
seeds, never loss at a fixed step count. The README's Muon table already
carries "first eval at or below AdamW's step-600 val". For this track the
headline is **tokens to val 3.28 on FineWeb**, which makes llm.mojo comparable
to both llm.c (10B tokens) and modded-nanogpt (under 0.7B tokens with Muon
plus the record 5 architecture in January 2025; under 330M today with
everything). The existing 524k-token AdamW run that reached 3.28 is the
baseline. Once W1 to W3 are in, schedule a single full run of the best recipe
at the full batch and record tokens-to-3.28 in the README table.

## 6. Results

To be filled by the item owners. One row per arm.

| item | flag(s) | val@300 seed 42 | three-seed mean | step time | verdict |
|---|---|---:|---:|---:|---|
| base | `-opt muon -l 4.8e-3 -mu 0.9 -c 0.1 -gc 1.0` (sweep doc, wd/gc on) | | | | reference |

## 7. Not in scope (so nobody re-litigates)

FlexAttention, sliding and long-short windows, window warmup with YaRN, the
48k-token sequence regime, document-aligned batches; every distributed and
communications record; the batch-size and sequence-length schedules tuned for
the 8xH100 aggregate batch; FP8 head and static FP8 scales (covered by
[`speedrun_techniques_research.md`](speedrun_techniques_research.md) Tier A);
and everything from record 62 onward (hashed n-gram tables, MUDD skips, paired
heads, XSA, ANVIL, sampled softmax, multi-token prediction, tail averaging).
Those are speedrun-specific structure that Keller Jordan's own README flags as
unlikely to scale, or they presuppose kernels we do not have. Value
embeddings, U-net skips and logit softcap (records 9, 11, 14, 15, 18) are
plausible later items but each adds parameters or changes the loss; they are
deferred until W5 to W9 have results.

## AI use statement

Written with AI assistance (Claude Code, Fable 5.1 agent), directed by Evan
Owen.
