#!/usr/bin/env python3
"""
Downloads and tokenizes the FineWeb / FineWeb-Edu datasets (for pretraining).
- FineWeb:     https://huggingface.co/datasets/HuggingFaceFW/fineweb
- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- The tokenization is GPT-2 tokenizer via tiktoken (or Llama-3 via transformers).

The output is written as sharded .bin files to a newly created
.fineweb10B/ (or .edu_fineweb10B/, etc.) folder. Shard 0 of each split is the
validation set, the remaining shards are training data.

Example of downloading the 100B sample of FineWeb-Edu, from the data directory:
    python fineweb.py -t edu -v 100B
100B runs for a few hours, depending on your internet and computer.
"""

import os
import argparse
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from utils import get_gpt2_encoding, write_datafile

# FineWeb has a few possible subsamples available. Each (type, version) maps to
# a local cache directory name and the remote dataset config name.
DIRECTORIES: dict[tuple[str, str], tuple[str, str]] = {
    ("classic", "10B"): ("fineweb10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B", "sample-100BT"),
    ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B", "sample-100BT"),
}

TOKEN_DTYPE = {
    "gpt-2": np.uint16,
    "llama-3": np.uint32,
}


def tokenize_gpt2(doc: dict) -> np.ndarray:
    """Tokenize a single document and return a numpy array of uint16 tokens."""
    enc = get_gpt2_encoding()
    eot = enc._special_tokens["<|endoftext|>"]  # end of text token
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), (
        "token dictionary too large for uint16"
    )
    return tokens_np.astype(np.uint16)


def tokenize_llama(doc: dict) -> np.ndarray:
    """Tokenize a single document and return a numpy array of uint32 tokens."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    assert tokenizer is not None, "Failed to load Llama-3 tokenizer"
    eot = tokenizer.encode("")[0]  # the tokenizer adds the EOT token (128000)
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(
        tokenizer.encode(
            doc["text"],
            add_special_tokens=False,
            verbose=False,
            split_special_tokens=True,
        )
    )
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), (
        "token dictionary too large for uint32"
    )
    return tokens_np.astype(np.uint32)


def main(
    fineweb_type: str = "classic",
    version: str = "10B",
    model_desc: str = "gpt-2",
    shard_size: int = 10**8,
) -> None:
    assert version in {"10B", "100B"}, "version must be one of: 10B, 100B"
    assert fineweb_type in {"edu", "classic"}, "type must be one of: edu, classic"

    local_dir, remote_name = DIRECTORIES[(fineweb_type, version)]

    # create the local cache directory if it doesn't exist yet
    data_cache_dir = os.path.join(os.path.dirname(__file__), f".{local_dir}")
    os.makedirs(data_cache_dir, exist_ok=True)

    # download the dataset
    if fineweb_type == "classic":
        fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")
        name = "fineweb"
    else:  # edu
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
        name = "edu_fineweb"

    if model_desc == "gpt-2":
        tokenize = tokenize_gpt2
    elif model_desc == "llama-3":
        tokenize = tokenize_llama
    else:
        raise ValueError(f"unknown model {model_desc}")

    token_dtype = TOKEN_DTYPE[model_desc]

    # tokenize all documents and write output shards, each of shard_size tokens
    # (the last shard holds the remainder)
    nprocs = max(1, (os.cpu_count() or 2) - 2)  # don't hog the entire system
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=token_dtype)
        token_count = 0
        progress_bar = None

        # pyrefly: ignore[bad-argument-type]  datasets' Dataset.__iter__ yields a
        # broader union than Iterable[dict], but each element is a doc mapping.
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    data_cache_dir, f"{name}_{split}_{shard_index:06d}.bin"
                )
                # split the doc into whatever fits; the remainder goes to next shard
                remainder = shard_size - token_count
                assert progress_bar is not None
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[
                    :remainder
                ]
                write_datafile(filename, all_tokens_np.tolist(), model_desc)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                data_cache_dir, f"{name}_{split}_{shard_index:06d}.bin"
            )
            write_datafile(filename, all_tokens_np[:token_count].tolist(), model_desc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FineWeb and Edu-FineWeb dataset preprocessing"
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="classic",
        choices=["classic", "edu"],
        help="Fineweb type, classic|edu",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="10B",
        choices=["10B", "100B"],
        help="Fineweb data sample size, 10B|100B",
    )
    parser.add_argument(
        "-m",
        "--model_desc",
        type=str,
        default="gpt-2",
        choices=["gpt-2", "llama-3"],
        help="Model descriptor, gpt-2|llama-3",
    )
    parser.add_argument(
        "-s",
        "--shard_size",
        type=int,
        default=10**8,
        help="Size of each data shard in the output .bin files, in tokens",
    )
    args = parser.parse_args()
    main(args.type, args.version, args.model_desc, args.shard_size)
