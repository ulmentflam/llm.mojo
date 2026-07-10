from std.python import Python

# ===----------------------------------------------------------------------=== #
# HuggingFace Hub fetch — the one deliberate Python bridge in the safetensors
# loading path.
#
# Everything downstream of "bytes on local disk" (safetensors parsing,
# config.json parsing, populating GPT2's parameter buffers) is pure Mojo —
# see llmm/safetensors.mojo. Mojo has no HTTP client in this SDK release, so
# fetching a remote repo's files needs *some* network layer; this mirrors the
# existing, narrowly-scoped Python bridge in llmm/dataloader.mojo (which
# reaches for Python's `glob` for one specific need, not as a general
# crutch). `huggingface_hub` is already a dependency of this project
# (pixi.toml, via `transformers`) and is the correct client for HF's auth,
# revision pinning, and local caching — reimplementing that in raw HTTP would
# be strictly worse, not more "pure".
# ===----------------------------------------------------------------------=== #


def fetch_hf_checkpoint(repo_id: String) raises -> String:
    """Download `config.json` + `model.safetensors` from a public HuggingFace
    model repo (e.g. "ulmentflam/gpt2-124m-fineweb-mojo") into the local HF
    cache, and return the local path to `model.safetensors` (its sibling
    `config.json` lands in the same cached snapshot directory, which
    `GPT2.__init__`'s `.safetensors` branch expects).

    Two single-file `hf_hub_download` calls rather than one `snapshot_download`
    with `allow_patterns`: passing a Python list through Mojo's Python bridge
    as a keyword argument didn't round-trip correctly (surfaced downstream as
    a confusing `'list' object has no attribute 'endswith'`); plain string
    kwargs are unambiguous and this repo only ever needs these two files.
    """
    var hub = Python.import_module("huggingface_hub")
    var config_path = hub.hf_hub_download(
        repo_id=repo_id, filename="config.json"
    )
    var weights_path = hub.hf_hub_download(
        repo_id=repo_id, filename="model.safetensors"
    )
    _ = config_path  # Downloaded as a side effect: lands next to weights_path.
    return String(weights_path)
