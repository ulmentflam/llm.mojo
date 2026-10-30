# Muon vs AdamW, GPT-2 124M from random init (600 steps, FineWeb)

First controlled AdamW-vs-Muon comparison in `llm.mojo`, run on the `muon-int` branch
(commit `f646bbe`). It is an early-training comparison only: 600 steps at 131,072
tokens/step = 78.6M tokens, one seed per optimizer. The Muon implementation itself was
verified against torch on other tracks; this document reports only the convergence run.
Format follows [`gpt2_124m_fineweb_training_run.md`](gpt2_124m_fineweb_training_run.md).

**Validation gate:** the runs' own val-loss curves. Both runs share the same init (step-0
val loss is identical, 11.009493, and step-1 train loss is identical, 11.013952), data order
and LR schedule, so the optimizer is the only intended difference.

## Setup

- **Model:** GPT-2 124M (`-e d12`), random init, bf16 (fp32 master weights/moments).
- **Data:** FineWeb-10B shards (`data/.fineweb10B/fineweb_train_*.bin`, `fineweb_val_*.bin`).
- **Batch:** `-b 32 -t 1024 -d 131072` (4 grad-accum micro-steps), 600 steps.
- **Recipe (both):** peak lr `6e-4`, 50 warmup steps, cosine to 0 (`-q 0.0`, schedule spans the 600 steps), weight decay 0.1, grad-norm clip 1.0, QK-clip off (`-qk_clip 0`), val every 100 steps over 20 batches, no sampling, no checkpoints.
- **Muon:** `-opt muon -mu 0.95`, same peak lr and weight decay as AdamW (no per-optimizer lr tuning was done).
- **Binary:** `build/train_gpt2_bf16` built by `make build-bf16` at 2026-10-29 20:58:28 -0400 from commit `f646bbe` (sources last modified 20:56:30, i.e. the binary is newer than all sources). Mojo toolchain as provisioned in the worktree pixi env.
- **Hardware:** single GB10 (DGX Spark class), Linux 6.17.0-1026-nvidia. A ~37 GB vLLM process (another user's, pid 862977, `VLLM::EngineCore`, 37426 MiB) was resident on the GPU before both runs (`nvidia-smi` captured before each run). It slows tensor-core kernels, so absolute step times are contended.

Commands (sequential, never concurrent; logs under the session scratchpad `jobs/`):

```
LLMM_LOG_LEVEL=verbose LLMM_LOG_FILE=conv_adamw.jsonl make train-bf16 ARGS="-e d12 -i 'data/.fineweb10B/fineweb_train_*.bin' -j 'data/.fineweb10B/fineweb_val_*.bin' -b 32 -t 1024 -d 131072 -l 6e-4 -u 50 -q 0.0 -c 0.1 -gc 1.0 -x 600 -v 100 -m 20 -s 0 -opt adamw -mu 0.95 -qk_clip 0"
LLMM_LOG_LEVEL=verbose LLMM_LOG_FILE=conv_muon.jsonl  make train-bf16 ARGS="... same ... -opt muon -mu 0.95 -qk_clip 0"
```

(The glob patterns must be single-quoted inside `ARGS`, otherwise the shell expands them and the trainer prints usage.) Each command was dry-run with `-x 3 -v 0` first. JSON-lines logs are committed in `docs/ai/data/muon_vs_adamw_{adamw,muon}.jsonl`; tables below come from `scripts/compare_runs.py` (built on `scripts/train_log.py`):

`pixi run python scripts/compare_runs.py docs/ai/data/muon_vs_adamw_adamw.jsonl docs/ai/data/muon_vs_adamw_muon.jsonl --names AdamW Muon`

## Results

Both runs exited 0, no NaN/divergence. Final val loss: **AdamW 5.4690, Muon 5.1115** (Muon 0.358 lower). Muon first reaches AdamW's final val loss at the step-400 eval (5.2565).

| step | val AdamW | val Muon | delta (Muon-AdamW) |
|---:|---:|---:|---:|
| 0 | 11.0095 | 11.0095 | +0.0000 |
| 100 | 6.5674 | 6.2045 | -0.3628 |
| 200 | 6.1798 | 5.7955 | -0.3843 |
| 300 | 5.9016 | 5.4905 | -0.4111 |
| 400 | 5.6607 | 5.2565 | -0.4041 |
| 500 | 5.5044 | 5.1361 | -0.3682 |
| 600 | 5.4690 | 5.1115 | -0.3575 |

| steps | train AdamW | train Muon | delta (Muon-AdamW) |
|---|---:|---:|---:|
| 1-50 | 8.1260 | 8.3476 | +0.2216 |
| 51-100 | 6.6972 | 6.3856 | -0.3116 |
| 101-150 | 6.4179 | 6.0381 | -0.3798 |
| 151-200 | 6.2066 | 5.8253 | -0.3813 |
| 201-250 | 6.0675 | 5.6694 | -0.3981 |
| 251-300 | 5.9161 | 5.5068 | -0.4092 |
| 301-350 | 5.7816 | 5.3700 | -0.4116 |
| 351-400 | 5.6727 | 5.2613 | -0.4114 |
| 401-450 | 5.5575 | 5.1626 | -0.3949 |
| 451-500 | 5.4737 | 5.0976 | -0.3761 |
| 501-550 | 5.4421 | 5.0758 | -0.3664 |
| 551-600 | 5.4133 | 5.0573 | -0.3560 |

| run | mean ms/step | tok/s | steps timed |
|---|---:|---:|---:|
| AdamW | 4171.4 | 31782 | 598 |
| Muon | 3972.0 | 32986 | 598 |

First Muon val <= AdamW final val (5.4690, step 600): step 400 (val 5.2565)

Timing: step 1 excluded plus step 2 (warmup). The AdamW mean is inflated by 40 steps above 5 s (max 13.0 s), all absent in the Muon run; this looks like interference from the co-resident GPU tenant, not the optimizer. Medians are the fairer comparison: **AdamW 4008 ms, Muon 3955 ms** (min 3920 vs 3889 ms). So at this batch size Muon was within noise of AdamW (or marginally faster). This is expected, not a contradiction of the +29% figure measured at the 4k-token smoke config: the Muon optimizer step is a fixed cost of roughly 20-40 ms per optimizer step (Newton-Schulz GEMMs plus elementwise passes; see the performance section of [`muon_muonclip_implementation_2026-10-29.md`](muon_muonclip_implementation_2026-10-29.md)), and here each optimizer step follows ~3.9 s of forward/backward over 4 accumulated micro-batches, so the overhead is about 1%. At the full run's 524k tokens/step it would be about 0.25%. The +29% applies only when a whole step is a single B=4 micro-batch.

## Caveats

- Single seed per optimizer; no variance estimate. A 0.36 val-loss gap this large and consistent across all evals and 50-step train windows is unlikely to be seed noise, but it is not quantified.
- 600 steps / 79M tokens is early training with a cosine schedule compressed to that length. The gap peaked (-0.41 at step 300) and was narrowing (-0.36 at step 600); this says nothing about the converged gap (the AdamW 524k-token run reached val 3.28).
- Muon used AdamW's lr and weight decay without tuning; AdamW used the recipe from the earlier full run, scaled down. Neither was tuned for this short horizon.
- Muon's first 50 steps had higher train loss (+0.22) than AdamW, then crossed over.
- Step times were measured under GPU contention (vLLM tenant), with bursty slowdowns in the AdamW run.
- Muon kernel is unoptimized (a separate track covers that).

## AI use statement

Written with AI assistance (Claude Code, Sonnet agent), directed by Evan Owen.
