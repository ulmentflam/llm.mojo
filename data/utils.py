"""
Common utilities for data processing.
"""

from typing import Any

# pyrefly: ignore[untyped-import]  requests ships no stubs; types-requests
# is not in the pixi env.
import requests
import numpy as np

# pyrefly: ignore[untyped-import]  same for types-tqdm.
from tqdm import tqdm


def download_file(url: str, output_path: str) -> None:
    """Helper to download a file from a URL to a local path."""
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with (
        open(output_path, "wb") as file,
        tqdm(
            desc=output_path,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


HEADERS_INFO: dict = {
    "gpt-2": {
        "magic": 20240520,
        "version": 1,
        "token_dtype": np.uint16,
    },
    "llama-3": {
        "magic": 20240801,
        "version": 7,
        "token_dtype": np.uint32,
    },
}

MAGIC_NUMBER = 20240522


def write_datafile(filename: str, tokens: list[int], model_desc: str = "gpt-2") -> None:
    """
    Saves token data as a binary file, for reading in the target language.
    - First comes a header with 256 int32s
    - The tokens follow, each as uint16 or uint32, depending on the model.
    """
    assert len(tokens) < 2**31, "Too many tokens"  # ~2.1 billion tokens
    assert model_desc in HEADERS_INFO, f"Invalid model description: {model_desc}"
    header_info = HEADERS_INFO[model_desc]
    magic = header_info["magic"]
    version = header_info["version"]
    token_dtype = header_info["token_dtype"]
    header = np.zeros(256, dtype=np.int32)
    header[0] = magic
    header[1] = version
    header[2] = len(tokens)
    tokens_np = np.array(tokens, dtype=token_dtype)
    num_bytes = (256 * 4) + (len(tokens) * tokens_np.itemsize)
    print(
        f"Writing {len(tokens):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format"
    )
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_np.tobytes())
    print(f"Wrote {filename}")


def write_evalfile(filename: str, datum: list[dict[str, Any]]) -> None:
    """
    Saves eval data as a .bin file, for reading in the target language.
    Used for multiple-choice style evals, e.g. HellaSwag and MMLU
    - First comes a header with 256 int32s
    - The examples follow, each example is a stream of uint16_t:
        - <START_EXAMPLE> delimiter of 2**16-1, i.e. 65,535
        - <EXAMPLE_BYTES>, bytes encoding this example, allowing efficient skip to next
        - <EXAMPLE_INDEX>, the index of the example in the dataset
        - <LABEL>, the index of the correct completion
        - <NUM_COMPLETIONS>, indicating the number of completions (usually 4)
        - <NUM><CONTEXT_TOKENS>, where <NUM> is the number of tokens in the context
        - <NUM><COMPLETION_TOKENS>, repeated NUM_COMPLETIONS times
    """
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC_NUMBER  # magic number
    header[1] = 1  # version
    header[2] = len(datum)  # number of examples
    header[3] = 0  # longest_example_bytes
    longest_example_bytes = 0
    full_stream = []  # The stream of uint16s, we write a single uint16 at a time at the end
    assert len(datum) < 2**16, "Too many examples"
    for idx, data in enumerate(datum):
        stream = []
        stream.append(2**16 - 1)  # <START_EXAMPLE> delimiter
        stream.append(
            0
        )  # <EXAMPLE_BYTES>, bytes encoding this example, allowing efficient skip to next
        stream.append(idx)  # <EXAMPLE_INDEX>, the index of the example in the dataset
        stream.append(data["label"])  # <LABEL>, the index of the correct completion
        ending_tokens = data["ending_tokens"]
        assert len(ending_tokens) == 4, "Only 4 completions are supported"
        stream.append(
            len(ending_tokens)
        )  # <NUM_COMPLETIONS>, indicating the number of completions (usually 4)
        ctx_tokens = data["ctx_tokens"]
        assert all(0 <= tok < 2**16 for tok in ctx_tokens), (
            "Context tokens must be uint16"
        )
        stream.append(len(ctx_tokens))
        stream.extend(ctx_tokens)
        for end_tokens in ending_tokens:
            assert all(0 <= tok < 2**16 for tok in end_tokens), (
                "Ending tokens must be uint16"
            )
            stream.append(len(end_tokens))
            stream.extend(end_tokens)
        n_bytes = len(stream) * 2  # 2 bytes per uint16
        assert n_bytes < 2**16, "Example too long"
        stream[1] = (
            n_bytes  # <EXAMPLE_BYTES>, bytes encoding this example, allowing efficient skip to next
        )
        longest_example_bytes = max(longest_example_bytes, n_bytes)
        full_stream.extend(stream)
    stream_np = np.array(full_stream, dtype=np.uint16)
    assert 0 < longest_example_bytes < 2**16, "Longest example bytes must be uint16"
    header[3] = longest_example_bytes
    num_bytes = (256 * 4) + (len(full_stream) * 2)
    print(
        f"Writing {len(datum):,} examples to {filename} ({num_bytes:,} bytes) in the eval format"
    )
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(stream_np.tobytes())
    print(f"Wrote {filename}")
