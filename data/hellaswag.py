#!/usr/bin/env python3
"""
Downloads and tokenizes the HellaSwag validation split for multiple-choice
completion-style evaluation.
- HellaSwag: https://github.com/rowanz/hellaswag
- The tokenization is the GPT-2 tokenizer via tiktoken.

Each example is a context plus 4 candidate endings; the correct ending is
whichever completion the model assigns the lowest average per-token loss.
GPT-2 124M scores roughly 29-30% completion-style accuracy on this split
(see Karpathy's llm.c reference implementation, dev/data/hellaswag.py).

The output is written to a newly created .hellaswag/ folder as
hellaswag_val.bin, in the eval-file format (data/utils.py's write_evalfile) —
a 256-int32 header followed by one uint16 stream per example (see
write_evalfile's docstring for the exact layout). Read by llmm's
EvalDataLoader for evaluation in Mojo.
"""

import os
import json
import argparse

from utils import download_file, get_gpt2_encoding, write_evalfile

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".hellaswag")

HELLASWAGS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}


def download(split: str) -> None:
    """Downloads the HellaSwag jsonl for the given split to DATA_CACHE_DIR."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = HELLASWAGS[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")


def render_example(example: dict) -> dict:
    """Tokenize one HellaSwag example into write_evalfile's expected shape.

    Each of the 4 endings is tokenized with a leading space prepended (GPT-2's
    BPE treats " word" and "word" as different tokens, and the context never
    ends with a trailing space, so this matches how the model would actually
    see a continuation).
    """
    enc = get_gpt2_encoding()
    ctx_tokens = enc.encode(example["ctx"])
    ending_tokens = [enc.encode(" " + ending) for ending in example["endings"]]
    return {
        "label": example["label"],
        "ctx_tokens": ctx_tokens,
        "ending_tokens": ending_tokens,
    }


def tokenize(split: str = "val") -> None:
    download(split)
    jsonl_path = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    datas = []
    with open(jsonl_path, "r") as f:
        for line in f:
            example = json.loads(line)
            datas.append(render_example(example))
    filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.bin")
    write_evalfile(filename, datas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HellaSwag dataset preprocessing")
    parser.add_argument(
        "-s",
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="HellaSwag split, train|val|test",
    )
    args = parser.parse_args()
    tokenize(args.split)
