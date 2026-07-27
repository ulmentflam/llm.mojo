#!/usr/bin/env python3
"""Generate the HuggingFace model card for one re-trained precision arm.

These cards REPLACE published ones, so they have to say what changed and why.
The 124M bf16 repo in particular previously documented a different run
entirely — a single GB10, 19,552 steps at 524,288 tokens/step — while its
replacement is a 7-GPU run of 22,345 steps at 458,752. Every number on that
card changes, so the card is generated from the run's own recorded results
rather than edited.

Numbers come from docs/ai/v2_arm_results/<arm>.json, written by
record_arm_result.py from the run's train.log and make eval output.
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "ai" / "v2_arm_results"

PRECISION_BLURB = {
    "bf16": "bf16 mixed precision — parameters, activations and gradients in bf16, fp32 AdamW master weights and moments.",
    "fp8": "fp8 (e4m3) for every per-block linear GEMM, with delayed scaling from a 16-step amax history; parameters stay bf16 and the optimizer stays fp32.",
    "nvfp4": "NVFP4 (e2m1, block-scaled) on the middle blocks' MLP forward GEMMs with stochastic rounding and a randomized Hadamard transform; backward stays bf16, parameters stay bf16, optimizer stays fp32.",
}

LR = {"124M": "6e-4", "774M": "2.5e-4"}
LAYERS = {
    "124M": "12 layers, 12 heads, 768 channels (d12)",
    "774M": "36 layers, 20 heads, 1280 channels (d36)",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    r = json.loads((RESULTS_DIR / f"{args.arm}.json").read_text())
    scale, precision = r["scale"], r["precision"]
    acc = 100 * r["hellaswag_acc"]
    lo, hi = (100 * x for x in r["hellaswag_ci95"])

    card = f"""---
license: odc-by
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- gpt2
- llm.c
- llm.mojo
- mojo
- fineweb
datasets:
- HuggingFaceFW/fineweb
---

# GPT-2 {scale} ({precision}), trained from scratch on FineWeb with llm.mojo

A GPT-2 {scale} ({LAYERS[scale]}) language model trained **from scratch** — random
initialisation, no GPT-2 warm-start — for one epoch of the
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) classic
10B-token sample, using [llm.mojo](https://github.com/ulmentflam/llm.mojo), a
Mojo/MAX port of Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c).

This is a training-run artifact and reproducibility reference, not a
state-of-the-art model. At this scale and data budget GPT-2 produces
locally-coherent but not highly capable text.

## Why these weights were replaced

An earlier version of this repo held a checkpoint trained with a defect: a
scratch-buffer overrun in the fused bias-gradient kernel meant several bias
tensors received **no gradient at any step**. It was measurable directly from
the published weights — GPT-2 initialises biases to exactly zero, so a bias
still bit-exactly zero after 22,345 optimizer steps was never updated.

The kernel is fixed and this checkpoint is a complete re-run. The export now
refuses to publish any checkpoint whose matmul biases are all zero. Full
write-up:
[`docs/ai/dbias_scratch_overrun_silent_zero_bug.md`](https://github.com/ulmentflam/llm.mojo/blob/main/docs/ai/dbias_scratch_overrun_silent_zero_bug.md).

The previous weights remain reachable in this repo's commit history.

## Results

| | |
|---|---|
| Final validation loss | **{r["final_val_loss"]:.4f}** |
| HellaSwag (acc_norm) | **{r["hellaswag_k"]}/{r["hellaswag_n"]} = {acc:.2f}%** (Wilson 95% CI [{lo:.1f}%, {hi:.1f}%]) |
| Final train loss | {r["final_train_loss"]:.4f} |
| Throughput | ~{r["median_tok_per_s"]:,} tok/s ({r["median_ms_per_step"]:.1f} ms/step, median over the last 500 steps) |

## Training

- **Precision:** {PRECISION_BLURB[precision]}
- **Data:** FineWeb classic `sample-10BT`, GPT-2 BPE, one epoch = {r["total_steps"]:,} steps
- **Batch:** 458,752 tokens/step (7 ranks x gradient accumulation)
- **LR:** cosine, peak {LR[scale]}, 700-step warmup, decayed to 0, weight decay 0.1
- **Parallelism:** 7x RTX PRO 6000 Blackwell Max-Q, ZeRO-1 data parallel
- **Code:** [github.com/ulmentflam/llm.mojo](https://github.com/ulmentflam/llm.mojo)

## Files

- `model.safetensors` — `GPT2LMHeadModel`-compatible export, loadable with `transformers`
- `model_{r["final_step"]}.bin` — the raw llm.mojo/llm.c-format checkpoint

`infer_gpt2.mojo` loads any of a local `.bin`, a local `.safetensors`, or this
repo directly via `--hf {r["hf_repo"]}`.

## Reproducing

```sh
git clone --recurse-submodules https://github.com/ulmentflam/llm.mojo.git
cd llm.mojo && make install-cuda
pixi run python data/fineweb.py -t classic -v 10B -m gpt-2
make build-bf16 WORLD_SIZE=7
```

See the launcher for this arm's exact flags.
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(card)
    print(f"[write_model_card] wrote {args.out} for {args.arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
