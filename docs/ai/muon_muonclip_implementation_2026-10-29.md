# Muon + MuonClip implementation campaign (2026-10-29)

Campaign report for phases 1 and 2 of
[`muon_muonclip_implementation_plan.md`](muon_muonclip_implementation_plan.md):
the opt-in Muon optimizer (`-opt muon`) and QK-clip (`-qk_clip tau`), built by
parallel agent tracks and merged on branch `muon-int` (head `f646bbe`). The
default optimizer is still AdamW; the llm.c parity track is untouched. The
convergence comparison lives in the forthcoming
`muon_vs_adamw_124m_2026-10-29.md`; step-time optimization is a separate
follow-up track.

## 1. What was built

| Piece | Where | Notes |
|---|---|---|
| Muon update | `llmm/muon.mojo` | `muon_update` custom op: momentum + Nesterov kernel, per-matrix Frobenius normalize, 5-step Newton-Schulz, apply kernel |
| Newton-Schulz | `llmm/muon.mojo` | coefficients (3.4445, -4.7750, 2.0315), bf16, strided-batched cuBLASLt, one batch per tensor group (batch count = L) |
| QK-clip | `llmm/qk_clip.mojo`, `llmm/attention.mojo` | softmax kernel tracks per-(layer, head) max post-scale logit; `qk_clip_apply` rescales hot heads |
| Trainer wiring | `train_gpt2.mojo` | flags, `enable_muon` / `enable_qk_clip`, hybrid `update()`, logging |
| Checkpoint | `llmm/checkpointing.mojo` | `optimizer_kind` in state-file header slot 40 |
| RNG registry | `llmm/rng_device.mojo` | `SR_MUON_STREAM = 6` |
| Tests | `tests/test_muon.mojo`, `tests/test_muon_equivalence.py`, `tests/test_muon_crosscheck.py`, `tests/test_qk_clip.mojo`, `tests/reference.py`, `tests/kernels/muon.py` | see section 4 |

### Muon

- Update rule is the plan's: fp32 momentum `M` (reuses `m_memory`; `v_memory`
  stays allocated and unused for Muon ranges), Nesterov `U`, normalize by the
  Frobenius norm, 5 NS iterations in bf16, then
  `W -= lr * (0.2*sqrt(max(rows, cols)) * O + wd * W)`. The Moonlight RMS
  scale means `-l` and `-c` are shared with AdamW; there is no second LR.
- Tensor groups: Q, K, V (the `qkv_weight` row blocks, each orthogonalized as
  its own C x C matrix), attn-proj, fc, proj. Tall matrices (fc) are read as
  column-major X^T so the transpose is free (a layout choice, no copy).
  Scratch comes from `persistent_device_buffer`, whose allocation is fixed at
  first call, so the scratch names carry their sizes.
- Everything else (wte, wpe, biases, LN gamma/beta) runs the unchanged AdamW
  kernel over the five non-matrix flat ranges, plus one `muon_update`.
- Targets: GPU is the fast path (cuBLASLt). CPU has a naive fp32 host
  reference (124 s/step at 30M params, so tests only). Metal raises.
- Under `LLMM_SR_MASTER` the param-store stochastic rounding draws from rng
  stream 6.
- Phase 1 scope: WORLD_SIZE 1 / ZeRO-0 only; `enable_muon` raises otherwise.

### Trainer surface

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `-opt adamw\|muon` | `LLMM_OPTIMIZER` | `adamw` | optimizer |
| `-mu <float>` | `LLMM_MUON_MOMENTUM` | 0.95 | Muon momentum |
| `-qk_clip <tau>` | `LLMM_QK_CLIP` | 0 (off) | QK-clip threshold; independent of `-opt` |

Logging: setup table `opt=`; verbose setup `muon_momentum=` and `qk_clip=`;
per-step `qk_max=`. State checkpoints record `optimizer_kind` (0 adamw, 1
muon) in header slot 40; resuming with the other optimizer raises, because
Muon's momentum lives in the `m` buffer.

### QK-clip

The causal softmax kernel already computes each row's post-scale max; thread 0
atomic-maxes it into a `[L*NH]` buffer (non-negative fp32 as int32 bits;
negatives clamp to 0, harmless against tau). After the optimizer update,
`qk_clip_step` scales the Q and K rows of each head with max > tau by
`sqrt(tau / S_max)`, and the matching `qkv_bias` entries too, on the fp32
master (bf16 builds) and its rounded param copy. It raises on CPU (no logit-max
hook, it would silently never fire) and WORLD_SIZE > 1 (ranks would diverge
without an allreduce-max; ZeRO >= 1 also shards the master it indexes in full).

Max lifetime: every forward zeroes and fills a scratch `[L*NH]` buffer;
`backward()` folds it into the step max (overwrite on micro-step 0,
elementwise max afterwards) via `qk_head_max_fold`. Eval and generation
forwards never reach `backward()`, so they cannot leak into the clip.
`qk_max()` is valid from the step's last backward to the next step's first.

## 2. Process

Five parallel tracks, each in its own `git worktree`, merged into `muon-int`
by the orchestrator: reference + pytest (`muon/pytest`), kernel
(`muon/kernel`), trainer (`muon/trainer`), QK-clip (`muon/qkclip`) and
wiring (`muon/wire`), plus a torch cross-check track (`muon/crosscheck`) and
this docs track. Sonnet agents implemented; Opus agents reviewed adversarially
(reading the code and trying to construct a wrong kernel the gates would
accept). Every defect below was found by a reviewer, not by the author.

## 3. Gotchas (reviewer findings fixed before merge)

1. **bf16 hyperparameters must be rounded to the param dtype.** The op passes
   lr/momentum/wd/grad_scale as `Scalar[dtype]`, so a bf16 case hands the
   kernel momentum 0.94921875, not 0.95. The reference used fp32 values and a
   correct kernel failed the 1e-5 momentum check on every bf16 case. Fixed
   with `effective_hp()` in the reference.
2. **The W-only bf16 gate could not see Newton-Schulz.** One step at lr 0.02
   moves W less than a bf16 ulp at |W| ~ 2; simulated wrong kernels (no NS,
   wrong scale rule, no Nesterov, no weight decay) all passed the trajectory
   atol. Added `test_update_direction`: one step from W = 0, recover
   `O = -W1/(lr*scale)`, gate cosine >= 0.995 and relative Frobenius
   <= max(0.10, 5 x spread vs fp32 NS). Correct kernels measure 0.03-0.04,
   mutants >= 0.278. (Same lesson as the weak-gate memory note: a passing test
   overrules nothing unless it could have failed.)
3. **Reference rounding points were not kernel-faithful.** The reference
   rounded the norm to bf16 and each elementwise op separately; now it applies
   fp32 `1/(norm+eps)` once with bf16 operands, fp32 accumulate and fp32
   GEMM-beta epilogues, one bf16 rounding per GEMM.
4. **SR stream collision.** `SR_MUON_STREAM` was 2, already
   `NVFP4_SR_STREAM` (forward A-operand quantization). Moved to 6 and
   recorded in the `rng_device.mojo` registry (the old comment pointed at
   `adamw.mojo`, wrong file).
5. **QK-clip could never fire.** The step max was reset inside
   `zero_gradients(zero_param_grads=True)`, which the trainer calls from
   `backward()` of micro-step 0, after that micro-step's forward; the buffer
   was always zero at `qk_clip_step`. The forward also wrote straight into the
   step buffer, so val and sampling forwards leaked in. Fixed with the scratch
   plus fold design above. The new end-to-end test fails on the previous
   commit (head max 0.0 after backward).
6. **QK-clip silently inert on CPU / wrong on multi-rank**: now raises (see
   above).
7. **Crosscheck misattribution.** The first explanation of the ~1e-2 NS gap
   to stock `torch.optim.Muon` blamed torch's bf16 `addmm` epilogues.
   Isolation on torch 2.13.0 CPU showed torch rounds once with an fp32
   epilogue like ours (a double-rounded model doubles the gap); the gap is
   torch's bf16 norm plus 1-ulp accumulation-order flips amplified by NS. Our
   NS with a bf16 norm and exactly-rounded GEMMs matches torch bit-for-bit at
   48x96 / 96x48 and to 1.1e-3 / 3.7e-3 at 64x64 / 256x768.
8. **Widened gate must still reject mutants.** Under the `torch_muon` source
   the trajectory atol is widened to the measured torch-vs-ours gap;
   `test_tolerance_not_absurd` is now parametrized over both sources.
9. **Accidental absolute symlinks.** `data/.fineweb10B` and
   `data/.hellaswag` (pointing into the main checkout) were committed by
   accident and untracked in the QK-clip fix.

Also noted: torch's Muon uses lerp momentum (buffer and Nesterov update are
(1-mu) x ours, normalized away by NS) and `match_rms_adamw` equals
`0.2*sqrt(max)`; the fp32 param keeps torch's momentum buffer in fp32 like the
kernel's `M`. The crosscheck rescales by (1-mu) before comparing.

## 4. Verification matrix

| Check | Covers | Result |
|---|---|---|
| `tests/test_muon.mojo` (GPU, shared ctx) | NS vs host reference; batched and strided groups | pass |
| `tests/test_muon_equivalence.py` | Mojo op vs torch reference and vs stock `torch.optim.Muon` 2.13; trajectory and update-direction gates | 33/33 CPU path; 33/33 GPU path with `MAX_USE_ACCELERATOR=1` |
| `tests/test_muon_crosscheck.py` | our reference vs `torch.optim.Muon`: NS rel-Frobenius 0.9-1.5e-2 (torch's bf16 norm); trajectories within 1 bf16 ulp; momentum after the (1-mu) rescale agrees to 1e-6; mutants (skip-NS, 4 steps, no-Nesterov, Keller scale, sqrt(min), transpose-not-undone, no-wd) all rejected | pass |
| `tests/test_qk_clip.mojo` | softmax head max incl. scalar (T=13) and vector (T=16) kernels, B=2 planted logits, write-past-NH sentinel, above-diagonal trap, hook on/off bit-identical; `qk_clip_apply` fp32 and bf16+master; end-to-end tiny GPT2 (eval forward does not leak, micro-step max/reset, only hot heads' Q/K rows and biases scale, post-clip head max == tau) | pass |
| Default path unchanged | AdamW run, deterministic CPU: bit-identical to before; GPU: within run-to-run noise | pass |

First runs (10 steps, tinyshakespeare, pretrained 124M): Muon finishes within
~0.03 loss of AdamW on fp32 and bf16; checkpoint resume works; step time is
+18% (fp32) / +29% (bf16), unoptimized. QK-clip with tau = 5 on the pretrained
124M: `qk_max` goes 110.6 -> 6.3 -> 5.9 -> ... (fp32) and 110.5 -> 6.2 -> ...
(bf16). With tau = 100 it clips mainly step 1: the pretrained checkpoint's max
logit is ~110, so K2's default tau is not a no-op on it. (The plan predicted
it would not trigger on 124M; that holds for from-scratch training at normal
LR, not for this checkpoint.)

## 5. Known limits and follow-ups

- **Performance**: the +18% / +29% step time is unoptimized; see the
  follow-up perf track (the plan's ~5% estimate has not been met).
- **ZeRO sharding (plan phase 3, open)**: matrix-granular ownership plus
  allgather of updated rows. Needed before WORLD_SIZE > 1 or ZeRO >= 1; QK-clip
  also needs an allreduce-max there.
- **`v_memory`** is allocated but unused under Muon (~0.3 GB at 124M).
- **Metal** raises; **CPU** is test-only speed.
- **NorMuon** (per-row normalization) and **Polar Express** coefficients are
  not implemented.
- **Convergence evidence** at scale: see the forthcoming
  `muon_vs_adamw_124m_2026-10-29.md`.

## AI use statement

Written with AI assistance (Claude Code: Sonnet coding agents, Opus verification agents, Fable 5.1 orchestrator), directed by Evan Owen.
