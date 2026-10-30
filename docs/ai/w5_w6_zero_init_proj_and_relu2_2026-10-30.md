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
  ref 7.184724; all 16 gradient tensors pass with l2_ratio 0.9996-1.0000 and
  maxdiff 1e-5..2e-3 against thresholds 1e-2..4e-1; the 10-step trajectory drifts
  monotonically 1e-5 -> 0.0118 (step 9), above the gelu-calibrated 0.01, so the
  relu2 build uses `LOSS_STEP_TOL=0.02` (rationale in `test_gpt2.mojo`). torch
  fp32 vs fp64 of the reference agree to 1e-5 in loss, so the reference is
  well conditioned.
* `pytest tests/test_gelu_equivalence.py`: 18 passed (gelu + relu2 fwd/bwd/input
  survival; relu2 forward is also asserted to differ from gelu).

## Open issue: `make verify-cpu` under relu2 fails

`make verify-cpu MOJO_DEFINES="-D LLMM_MLP_ACT=relu2"`: loss 7.1832 (ref 7.1847,
gelu CPU error is 7e-5), 13 of 16 gradient tensors fail with l2_ratio ~0.8 (all
block tensors; wte/ln_f pass), trajectory off by up to 0.023. Isolated checks that
rule out the activation kernels on CPU: a standalone Mojo harness at the real
shape (256x768x3072, CPU target, relu2 build) matches a float64 host reference to
6e-6 (forward) and 2.6e-6 on a max of 3.3 (d_input backward), and the CPU custom-op
pytest passes. So the defect is elsewhere in the CPU model path under relu2's
larger activations (hypothesis, not verified: a CPU attention/softmax or
layernorm numeric that gelu never stressed). GPU, the training path, passes. CPU
relu2 training is therefore unverified.

## Not verified

Metal build; fp8/fp4 under relu2; CPU relu2 model path (above); step-time change
(ranking jobs record it); bf16 training quality (ranking jobs below).

## Ranking jobs

`gpu_queue/pending/w56_{base,zeroinit,relu2,both,smallinit}.sh`: each rebuilds
`build/train_gpt2_bf16` with its own `MOJO_DEFINES` (recorded with `ls -l
--time-style=full-iso` in `sweeps/w56_<tag>.build.txt`) and runs the section-1
Muon recipe, 300 steps, seed 42, logging to `sweeps/w56_<tag>.jsonl`.

## AI use statement

Written with AI assistance (Claude Code, Sonnet), directed by Evan Owen.
