#!/usr/bin/env python3
"""
Downloads and tokenizes the TinyStories dataset.
- The download is from HuggingFace datasets.
- The tokenization is GPT-2 tokenizer via tiktoken.

The output is written to a newly created .tinystories/ folder.
"""

import os
import glob
import json
import argparse
import tarfile

from transformers import AutoTokenizer

from utils import download_file, get_gpt2_encoding, write_datafile

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".tinystories")

TINY_STORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
ARCHIVE_NAME = "TinyStories_all_data.tar.gz"
DATA_DIR_NAME = "TinyStories_all_data"
VAL_FILE_NAME = "TinyStories_val.bin"
TRAIN_FILE_NAME = "TinyStories_train.bin"


def download() -> None:
    """
    Downloads the TinyStories dataset from HuggingFace and extracts it.
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    archive_path = os.path.join(DATA_CACHE_DIR, ARCHIVE_NAME)
    if not os.path.exists(archive_path):
        download_file(TINY_STORIES_URL, archive_path)
        print(f"Downloaded TinyStories dataset to {archive_path}")
    else:
        print(f"TinyStories dataset already downloaded to {archive_path}")

    data_dir = os.path.join(DATA_CACHE_DIR, DATA_DIR_NAME)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Extracting {archive_path} ...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(data_dir)
        print(f"Extracted TinyStories dataset to {data_dir}")
    else:
        print(f"TinyStories dataset already extracted to {data_dir}")


def tokenize(model_desc: str = "gpt-2") -> None:
    if model_desc == "gpt-2":
        encoder = get_gpt2_encoding()

        def encode(s: str) -> list[int]:
            return encoder.encode(s)

        eot = encoder._special_tokens["<|endoftext|>"]  # End of text token
    elif model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        assert tokenizer is not None, "Failed to load Llama-3 tokenizer"

        def encode(s: str) -> list[int]:
            return tokenizer.encode(
                s, add_special_tokens=False, verbose=False, split_special_tokens=True
            )

        eot = tokenizer.eos_token_id
    else:
        raise ValueError(f"Invalid model descriptor: {model_desc}")

    data_dir = os.path.join(DATA_CACHE_DIR, DATA_DIR_NAME)
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    # The first shard is used for validation, the rest for training, following
    # the convention from Karpathy's llm.c.
    val_shards = shard_filenames[:1]
    train_shards = shard_filenames[1:]

    for split_name, shards, out_name in [
        ("val", val_shards, VAL_FILE_NAME),
        ("train", train_shards, TRAIN_FILE_NAME),
    ]:
        tokens = []
        for shard in shards:
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                text = example["story"]
                text = text.strip()  # get rid of leading/trailing whitespace
                tokens.append(eot)
                tokens.extend(encode(text))
        write_datafile(os.path.join(DATA_CACHE_DIR, out_name), tokens, model_desc)
        print(
            f"Tokenized TinyStories {split_name} split and wrote to "
            f"{os.path.join(DATA_CACHE_DIR, out_name)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyStories dataset preprocessing")
    parser.add_argument(
        "-m",
        "--model_desc",
        type=str,
        default="gpt-2",
        choices=["gpt-2", "llama-3"],
        help="Model type, gpt-2|llama-3",
    )
    args = parser.parse_args()
    download()
    tokenize(args.model_desc)
