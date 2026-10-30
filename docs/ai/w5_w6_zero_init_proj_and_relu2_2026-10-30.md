# W5 zero-init residual projections and W6 ReLU² (modded-nanogpt adoption)

Branch `mn/w56`. Both items are opt-in build constants; defaults are unchanged.

| Flag | Values | Effect |
|---|---|---|
| `-D LLMM_ZERO_INIT_PROJ` | `0` (default), `1`, `2` | residual projections (`attn_proj_weight`, `proj_weight`) at init: `0` N(0, 0.02/sqrt(2L)); `1` zeros; `2` N(0, 0.002). `wte` (tied head) untouched. |
| `-D LLMM_MLP_ACT` | `gelu` (default), `relu2` | MLP activation. |

Reference-side counterparts: `GPT2_ZERO_INIT_PROJ=0|1|2` and `GPT2_ACT=gelu|relu2`
environment variables in `train_gpt2.py`. Both are recorded in the trainer's
setup event / table only when non-default (default log output is unchanged).

## What the existing gates actually cover (read this before trusting "verify passes")

* `make test-fixtures` regenerates only the AdamW `.npz` fixtures. It has nothing
  to do with init or activations.
* `make verify-cpu/-gpu` (`test_gpt2.mojo`) load the **pretrained** `gpt2_124M.bin`
  and compare against `gpt2_124M_debug_state.bin`. Initialization code never runs
  there, so W5 cannot be gated by it. New gate: `make verify-init
  [ZERO_INIT_PROJ=0|1|2]` (`test_init_gpt2.mojo`): builds the d12 model from
  scratch in Mojo, writes the reference d12 checkpoint with `train_gpt2.py`, and
  compares all 16 tensors (`maxdiff <= 1e-6`) plus planted structure (proj tensors
  exactly zero in mode 1, rms 0.002 in mode 2, other weights rms 0.02) so two
  wrongly-all-zero sides cannot pass.
* Reference seed is 43 (`GPT.__init__`), trainer default `LLMM_INIT_SEED` is 42,
  so values do not match at the default seed. The gate pins `-D LLMM_INIT_SEED=43`.
  With seed 43 all 16 tensors match to <= 4.5e-8. (Pre-existing; not changed.)

## RNG stream decision (W5)

The `normal_` draw is performed in every mode and only the value written changes
(zeroed afterwards in mode 1, different std in mode 2), so every other tensor and
every later layer is bit-identical across modes. The reference does the same
(`normal_` then `zeros_`).

## W6 design

`llmm/gelu.mojo`: `relu2`, `relu2_grad`, build-constant dispatch `mlp_act` /
`mlp_act_grad` (identical code to `gelu` / `gelu_grad` by default), `act_relu2`
parameter on the standalone fwd/bwd kernels, always-compiled ops `relu2_fwd` /
`relu2_bwd` (tested in `tests/test_gelu_equivalence.py` in the default build).
`llmm/matmul.mojo`: the trainer's call sites are inside `matmul_fwd` /
`matmul_d_input_bwd` (and the lowp backward scaling helper), so dispatch is there,
not in `train_gpt2.mojo`. `USE_GELU_FUSION = not MLP_RELU2`: the cuBLASLt
GELU/DGELU epilogues are GELU-specific, so relu2 uses the existing unfused branch
(plain GEMM + elementwise kernel). Metal: the Metal path already uses the
unfused route and goes through the same `mlp_act`; not compiled here (Linux/CUDA
box), so the Metal build with the flag on is **untested**. fp8/fp4 forward paths
reach `bias_gelu_fwd` and the backward scaling kernel, so they inherit relu2, but
amax behaviour under relu2 is **not examined**.

## Evidence

Flag-off fixtures byte-identical (CPU torch is deterministic; sha256):

```
debug_state, existing main checkout   0cb7240803fe3372e0d485dd7bf5cff0ed6b554e35b52b88d01c308251228321
debug_state, base train_gpt2.py (HEAD) 0cb7240803fe3372e0d485dd7bf5cff0ed6b554e35b52b88d01c308251228321
debug_state, modified, flags off       0cb7240803fe3372e0d485dd7bf5cff0ed6b554e35b52b88d01c308251228321
gpt2_124M.bin (all three + relu2 run)  7e55b0603e37af0059186a065b551584b14823b14889507e6c668ec351710574
d12 init ckpt base script / modified   893cc193367302e7a2fd9f6c6194598f9de2722b81342deef0b15e4b64669f3a (both)
relu2 debug state (new file)           7110809b07a8532cab3d83c354806cfacf3839d0baf4bcc3ddf39c05318a95ac
```

* `make verify-init ZERO_INIT_PROJ=0|1|2`: `result gate=init ... status=pass` all three.
* `make verify-gpu` flags off: logits/loss/grads(16/16)/steps pass, `result gate=gpu status=pass`
  (loss 5.2700114 vs ref 5.2699833).
* `make verify-cpu` flags off: `result gate=cpu status=pass`.
* `make verify-gpu MOJO_DEFINES="-D LLMM_MLP_ACT=relu2"`: forward loss 7.1847344 vs
  ref 7.184724; all 16 gradient tensors pass with l2_ratio 0.9996-1.0000. The
  10-step trajectory is checked at the default `LOSS_STEP_TOL=0.01` against a
  **float64** PyTorch trajectory (section "Root cause" below explains why the
  fp32 one is the wrong yardstick here). An earlier revision of this branch
  loosened the tolerance to 0.02 instead; that is reverted.
* `pytest tests/test_gelu_equivalence.py`: 18 passed (gelu + relu2 fwd/bwd/input
  survival; relu2 forward is also asserted to differ from gelu).

## Root cause: `make verify-cpu` under relu2 failed (fixed)

Symptom (first revision of this branch): `make verify-cpu MOJO_DEFINES="-D
LLMM_MLP_ACT=relu2"` gave loss 7.1832 vs ref 7.1847 and 13/16 gradient tensors
at l2_ratio ~0.8, while the GPU build passed. The first hypothesis (a wrong relu2
backward on CPU) was wrong: per-layer gradient error was 0.5-1% at layer 11 and
grew smoothly to ~22% at layers 0-1, i.e. an error amplified by depth, not a
local defect.

Method: dump the step-0 activations (residual_2/3, ln_2, attn_proj, fc_proj,
fch, fch_gelu) from the Mojo CPU and GPU builds and compare each layer against
an independent float64 numpy forward of the same weights and batch (relative L2
error per layer):

| build | layer-0 attn_proj | worst layer, any tensor |
|---|---|---|
| CPU, gelu (default) | 3.1e-4 | 1.9e-3 |
| CPU, relu2 | 3.1e-4 | 3.5e-3 |
| GPU fp32 (no TF32), relu2 | 1.9e-7 | 4.2e-6 |
| CPU, relu2, libm exp in attention | 4.5e-7 | 1.2e-5 |

The CPU error is present at layer 0 before any MLP and under gelu too, so it is
not the activation. The CPU attention (`attention_fwd_cpu` / `attention_bwd_cpu`,
called from `train_gpt2.mojo` with `use_soft_exp=True`) uses the FA4 degree-4
polynomial `software_emulated_exp` (`llmm/attention.mojo:161`, coefficients
`C0..C4` at `:141-147`), whose maximum relative error over [0,1) is **7.8e-4**
(measured in numpy). The cuBLAS GEMM attention on GPU uses libm `exp`, which is
why only the CPU build showed it. Under gelu this costs 3e-4 and passes. Under
relu2 on GELU-pretrained weights the model is badly conditioned: the residual
stream reaches 1.2e5 by layer 3 (gelu: 3e3), and the small attention error is
amplified into a 2e-4 loss error and 20% gradient error in the lowest layers.

Fix: `ATTN_SOFT_EXP = not MLP_RELU2` in `train_gpt2.mojo`; relu2 builds pass
`use_soft_exp=False` to both attention calls. Default builds are unchanged (the
constant is True, the same instantiation as before); on CUDA the GEMM attention
path ignores the parameter, so relu2 GPU training binaries are unaffected. With
the fix, verify-cpu under relu2: loss 7.1847253 vs 7.184724, 16/16 gradients pass.

Mutant/planted checks that this gate now has teeth:
* soft exp restored under relu2 (the pre-fix build): verify-cpu fails, 13/16 tensors.
* relu2 forward with the gelu derivative in the backward (`mlp_act_grad` mutated
  in a scratch copy): verify-gpu must fail at the gradient check (result in the
  final report of the verifying agent, logs in the campaign scratchpad).

## Why the relu2 trajectory reference is float64

Before the attention fix the GPU relu2 trajectory "drifted" monotonically to
0.0118 above the PyTorch fp32 trajectory by step 9, and the CPU build with exact
exp drifts to 0.0126. Both Mojo builds agree with each other and disagree with
torch. Torch fp32 vs torch fp64 of the same 10 steps:

| step | torch fp32 | torch fp64 | Mojo GPU fp32 | Mojo CPU fp32 (exact exp) |
|---|---|---|---|---|
| 0 | 7.184724 | 7.184734 | 7.184734 | 7.184725 |
| 8 | 3.748043 | 3.760766 | 3.759350 | |
| 9 | 3.367974 | 3.380598 | 3.379655 | 3.380547 |

So the drift was torch fp32's own rounding on an ill-conditioned model, not a Mojo
backward error (a wrong activation derivative fails the gradient check instead,
see the mutant above). The relu2 expected list in `test_gpt2.mojo` is now
`GPT2_ACT=relu2 scripts/gen_expected_losses.py --dtype float64` and the
tolerance stays at 0.01. For gelu, torch fp32 and fp64 agree closely and the
existing fp32 list is kept.

## Not verified

Metal build (relu2 there now uses libm exp in attention: correct by the same argument as CPU, untested); fp8/fp4 under relu2; step-time change
(ranking jobs record it); bf16 training quality (ranking jobs below).

## Ranking jobs

`gpu_queue/pending/w56_{base,zeroinit,relu2,both,smallinit}.sh`: each rebuilds
`build/train_gpt2_bf16` with its own `MOJO_DEFINES` (recorded with `ls -l
--time-style=full-iso` in `sweeps/w56_<tag>.build.txt`) and runs the section-1
Muon recipe, 300 steps, seed 42, logging to `sweeps/w56_<tag>.jsonl`.

## AI use statement

Written with AI assistance (Claude Code, Sonnet; root cause and gate fixes by Claude Code, Opus), directed by Evan Owen.
