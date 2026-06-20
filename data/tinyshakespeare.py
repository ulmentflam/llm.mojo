#!/usr/bin/env python3
"""
Downloads and tokenizes the Tiny Shakespeare dataset.
- The download is from Karpathy's github repository.
- The toknization is GPT-2 tokenizer via tiktoken.

The output is written to a newly created tinyshakespear/ folder.


"""

import os
import argparse

from transformers import AutoTokenizer

from utils import download_file, get_gpt2_encoding, write_datafile

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".tinyshakespeare")

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
FILE_NAME = "tiny_shakespeare.txt"
VAL_FILE_NAME = "tiny_shakespeare_val.bin"
TRAIN_FILE_NAME = "tiny_shakespeare_train.bin"
VAL_SIZE = 32768


def download() -> None:
    """
    Downloads the Tiny Shakespeare dataset from Karpathy's github repository.
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    input_txt_path = os.path.join(DATA_CACHE_DIR, FILE_NAME)
    if not os.path.exists(input_txt_path):
        download_file(TINY_SHAKESPEARE_URL, input_txt_path)
        print(f"Downloaded Tiny Shakespeare dataset to {input_txt_path}")
        return
    print(f"Tiny Shakespeare dataset already downloaded to {input_txt_path}")


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

    filename = os.path.join(DATA_CACHE_DIR, FILE_NAME)
    text = open(filename, "r").read()
    sections = text.split("\n\n")
    tokens = []
    for i, s in enumerate(sections):
        tokens.append(eot)
        # there was a mild bug where I originally intended to remove \n\n, but instead just added
        # the EOT right after each \n\n, so I'm keeping that behavior for backwards compatibility
        # therefore we have to here add an extra \n\n at the end of each section, except the last
        s_pad = s + "\n\n" if i != len(sections) - 1 else s
        tokens.extend(encode(s_pad))
    # Let's take the first 10% of the tokens for validation
    val_tokens = tokens[:VAL_SIZE]
    train_tokens = tokens[VAL_SIZE:]
    write_datafile(
        os.path.join(DATA_CACHE_DIR, TRAIN_FILE_NAME), train_tokens, model_desc
    )
    write_datafile(os.path.join(DATA_CACHE_DIR, VAL_FILE_NAME), val_tokens, model_desc)
    print(
        f"Tokenized Tiny Shakespeare dataset and wrote to {os.path.join(DATA_CACHE_DIR, TRAIN_FILE_NAME)} and {os.path.join(DATA_CACHE_DIR, VAL_FILE_NAME)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tiny Shakespeare dataset preprocessing"
    )
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
