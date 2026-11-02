#!/usr/bin/env python3
"""Convert an llm.mojo model checkpoint to a Hugging Face GPT2LMHeadModel export.

Why this script exists
-----------------------
llm.mojo uses its own checkpoint magic number (20240520, see MODEL_MAGIC in
llmm/checkpointing.mojo) to distinguish its checkpoint format from upstream
llm.c's own checkpoints (magic 20240326), even though the 256-int32 header
layout and parameter blob order are otherwise byte-identical (same fields:
maxT, V, L, H, C, Vp; same parameter order: wte, wpe, ln1w/b, qkvw/b,
attprojw/b, ln2w/b, fcw/b, fcprojw/b, lnfw/b). This is intentional and
permanent -- llmm/checkpointing.mojo's MODEL_MAGIC must not be changed to
match llm.c's.

This script makes a *patched copy* of an llm.mojo checkpoint with llm.c's
original magic number written in, solely so upstream llm.c tooling
(third_party/llm.c/dev/eval/export_hf.py) can read it, and then runs that
converter to produce a standard Hugging Face `transformers` export
(safetensors, GPT2LMHeadModel-compatible). It NEVER modifies the source
checkpoint -- it operates on a temporary copy only.

Usage
-----
  python scripts/export_to_hf.py --input log124M/model_19552.bin --output /tmp/hf_export

Rotary / QK-norm checkpoints
----------------------------
A checkpoint whose header flags word (slot 20) records rotary and/or QK-norm is
NOT a stock GPT-2: a stock `GPT2LMHeadModel` would ignore both and produce wrong
output with no error. Such files take a separate path (`export_llm_mojo`, picked
automatically from the flags; llm.c's exporter is not used) that writes
`model_type: llm_mojo_gpt2`, which `transformers` refuses to load (no such
architecture) instead of loading it as GPT-2. Layout notes:
  - Tensor names and Conv1D transposes are the same as the stock export; the
    tied lm_head is not stored (config `tie_word_embeddings: true`), as there.
  - Weights keep the file's dtype (bf16 file -> bf16, fp32 file -> fp32): the
    export is then bit-exact to the .bin. Converting fp32 -> bf16 is refused;
    `--dtype float32` on a bf16 file widens exactly (for fp32 builds/tools).
  - `transformer.wpe.weight` is omitted when it is all zero and the file is
    rotary (it is never read; 1.5 MB of zeros that suggest a position table that
    does not exist). Nonzero wpe is kept. infer_gpt2 and the trainer's
    .safetensors loader accept the omission only for rotary configs.
  - config.json records rotary (base 1024, half-truncated), qk_norm, eps, the
    attention scale and the llm.mojo flags word `llm_mojo_arch_flags`.
Files written before QK-norm had a flag bit record rotary only; correct a copy
first with scripts/set_checkpoint_arch_flags.py (the exporter only believes the
header).

This is a thin wrapper; the real conversion logic (header parsing, tensor
reshaping, bf16/fp32 handling) lives in
third_party/llm.c/dev/eval/export_hf.py and is unchanged/untouched here.
"""

import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# See llmm/checkpointing.mojo: MODEL_MAGIC (llm.mojo) vs. llm.c's own magic,
# which third_party/llm.c/dev/eval/export_hf.py hardcodes as the only magic
# it accepts.
LLM_MOJO_MAGIC = 20240520
LLM_C_MAGIC = 20240326

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_HF_PY = REPO_ROOT / "third_party" / "llm.c" / "dev" / "eval" / "export_hf.py"

# The four bias tensors produced by `matmul_bias_bwd`, in HF naming. These are
# checked for non-zeroness on export because a scratch overrun in that kernel
# once shipped six published checkpoints whose `c_attn`/`c_fc` biases were
# bit-exactly zero -- never updated by a single optimizer step -- and nothing
# between the trainer and the Hub noticed
# (docs/ai/dbias_scratch_overrun_silent_zero_bug.md). GPT-2 initialises biases
# to exactly 0, so "still zero after training" is decisive rather than
# suggestive, and upload is the last point at which it is cheap to catch.
MATMUL_BIAS_SUFFIXES = (
    "attn.c_attn.bias",
    "attn.c_proj.bias",
    "mlp.c_fc.bias",
    "mlp.c_proj.bias",
)


def verify_export(output: Path) -> bool:
    """Check an export is loadable, finite, and not silently missing gradients.

    Deliberately CPU-only and dependency-light: it must be runnable while the
    GPUs are busy training, which is exactly when exports happen.
    """
    from safetensors import safe_open

    path = output / "model.safetensors"
    if not path.is_file():
        print(f"[verify] FAIL: {path} does not exist")
        return False

    ok = True
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        nonfinite = []
        dead = []
        for k in keys:
            t = f.get_tensor(k).float()
            if not t.isfinite().all():
                nonfinite.append(k)
            if k.endswith(MATMUL_BIAS_SUFFIXES) and not t.any():
                dead.append(k)

    print(f"[verify] {len(keys)} tensors in {path}")

    if nonfinite:
        print(f"[verify] FAIL: {len(nonfinite)} tensor(s) contain NaN/Inf:")
        for k in nonfinite[:8]:
            print(f"           {k}")
        ok = False

    if dead:
        print(
            f"[verify] FAIL: {len(dead)} matmul bias tensor(s) are entirely zero,"
            " i.e. never received a gradient:"
        )
        for k in dead[:8]:
            print(f"           {k}")
        print(
            "         See docs/ai/dbias_scratch_overrun_silent_zero_bug.md."
            " Do NOT publish this checkpoint."
        )
        ok = False

    if ok:
        print("[verify] OK: all tensors finite, all matmul biases trained")
    return ok


# Version + 10 marks an untied head (docs/ai/untied_lm_head_design_2026-10-30.md).
UNTIED_VERSIONS = (13, 15)


def make_llmc_compatible_copy(src: Path, dst: Path) -> None:
    """Copy `src` to `dst`, then patch only the first int32 (the magic
    number) from LLM_MOJO_MAGIC to LLM_C_MAGIC. The source file is opened
    read-only and is never written to; only `dst` is modified.
    """
    with open(src, "rb") as f:
        head = f.read(8)
    if len(head) == 8:
        _, version = struct.unpack("<ii", head)
        if version in UNTIED_VERSIONS:
            raise SystemExit(
                f"[export_to_hf] refusing {src}: untied-head checkpoint "
                f"(version {version}: a separate lm_head tensor follows ln_f). "
                "The llm.c/HF 16-tensor layout has no lm_head, so the export "
                "would drop the head and misread the blob. Untied export is "
                "not supported; use a tied (version 3/5/7/9) checkpoint."
            )
    shutil.copyfile(src, dst)

    with open(dst, "r+b") as f:
        (magic,) = struct.unpack("<i", f.read(4))
        if magic != LLM_MOJO_MAGIC:
            raise ValueError(
                f"expected llm.mojo magic {LLM_MOJO_MAGIC} at the start of "
                f"{src}, found {magic}. Refusing to patch a file that "
                "doesn't look like an llm.mojo checkpoint."
            )
        f.seek(0)
        f.write(struct.pack("<i", LLM_C_MAGIC))

    print(
        f"[export_to_hf] patched copy magic {LLM_MOJO_MAGIC} -> {LLM_C_MAGIC} in {dst}"
    )


# --------------------------------------------------------------------------
# Rotary / QK-norm export (model_type llm_mojo_gpt2)
# --------------------------------------------------------------------------

MODEL_FLAGS_SLOT = 20
ARCH_FLAG_ROTARY = 1
ARCH_FLAG_QK_NORM = 4
KNOWN_MODEL_FLAGS = ARCH_FLAG_ROTARY | ARCH_FLAG_QK_NORM
LLM_MOJO_MODEL_TYPE = "llm_mojo_gpt2"
ROTARY_BASE = 1024
QK_NORM_EPS = 1e-6
LAYER_NORM_EPS = 1e-5  # train_gpt2.mojo EPSILON


def read_header(path: Path) -> list[int]:
    import numpy as np

    with open(path, "rb") as f:
        return [int(x) for x in np.frombuffer(f.read(256 * 4), dtype=np.int32)]


def conv1d(arr, name: str = ""):
    """Our [out, in] matmul weight -> HF Conv1D [in, out] (a transposed copy).
    `name` is only for scripts/check_hf_export.py's mutants."""
    import numpy as np

    return np.ascontiguousarray(arr.T)


def read_bin_tensors(path: Path, header: list[int]):
    """The model blob as {name: raw ndarray} in OUR layout (padded wte, stacked
    layers), plus its dtype kind ("bf16" raw uint16 bits, or "fp32")."""
    import numpy as np

    magic, version, maxT, V, L, H, C, Vp = header[:8]
    if magic != LLM_MOJO_MAGIC:
        raise SystemExit(f"{path}: bad magic {magic} (want {LLM_MOJO_MAGIC})")
    if version in UNTIED_VERSIONS:
        raise SystemExit(
            f"{path}: untied-head checkpoint (version {version}); the HF layout "
            "here has a tied head. Untied export is not supported."
        )
    if version not in (3, 5):
        raise SystemExit(f"{path}: unsupported model version {version}")
    kind = "fp32" if version == 3 else "bf16"
    dtype = np.float32 if version == 3 else np.uint16
    shapes = [
        ("wte", (Vp, C)),
        ("wpe", (maxT, C)),
        ("ln1w", (L, C)),
        ("ln1b", (L, C)),
        ("qkvw", (L, 3 * C, C)),
        ("qkvb", (L, 3 * C)),
        ("attprojw", (L, C, C)),
        ("attprojb", (L, C)),
        ("ln2w", (L, C)),
        ("ln2b", (L, C)),
        ("fcw", (L, 4 * C, C)),
        ("fcb", (L, 4 * C)),
        ("fcprojw", (L, C, 4 * C)),
        ("fcprojb", (L, C)),
        ("lnfw", (C,)),
        ("lnfb", (C,)),
    ]
    out = {}
    with open(path, "rb") as f:
        f.seek(256 * 4)
        for name, shape in shapes:
            n = int(np.prod(shape))
            data = np.fromfile(f, dtype=dtype, count=n)
            if data.size != n:
                raise SystemExit(f"{path}: truncated at tensor {name}")
            out[name] = data.reshape(shape)
        if f.read(1):
            raise SystemExit(f"{path}: trailing bytes after the tied 16-tensor blob")
    return out, kind


def llm_mojo_state_dict(w: dict, header: list[int], kind: str, widen: bool = False):
    """HF-named safetensors dict from the raw blob, bit-exact (no dtype cast).
    `widen` stores a bf16 file as fp32 (bf16 -> fp32 is exact: the bf16 bits
    shifted into the high half), for builds and tools that compute in fp32.

    Returns (tensors, wpe_omitted)."""
    import numpy as np
    import torch

    _, _, maxT, V, L, H, C, Vp = header[:8]
    flags = header[MODEL_FLAGS_SLOT]

    def mk(arr):
        if kind == "bf16" and widen:
            wide = (np.ascontiguousarray(arr).astype(np.uint32) << 16).view(np.float32)
            return torch.from_numpy(wide)
        if kind == "bf16":
            return torch.from_numpy(np.ascontiguousarray(arr).view(np.int16)).view(
                torch.bfloat16
            )
        return torch.from_numpy(np.ascontiguousarray(arr))

    t = {}
    t["transformer.wte.weight"] = mk(np.ascontiguousarray(w["wte"][:V]))
    wpe_omitted = False
    if (flags & ARCH_FLAG_ROTARY) and not np.any(w["wpe"]):
        wpe_omitted = True  # all zero and never read under rotary
    else:
        t["transformer.wpe.weight"] = mk(w["wpe"])
    for i in range(L):
        h = f"transformer.h.{i}"
        t[f"{h}.ln_1.weight"] = mk(w["ln1w"][i])
        t[f"{h}.ln_1.bias"] = mk(w["ln1b"][i])
        t[f"{h}.attn.c_attn.weight"] = mk(conv1d(w["qkvw"][i], "c_attn"))
        t[f"{h}.attn.c_attn.bias"] = mk(w["qkvb"][i])
        t[f"{h}.attn.c_proj.weight"] = mk(conv1d(w["attprojw"][i], "attn_c_proj"))
        t[f"{h}.attn.c_proj.bias"] = mk(w["attprojb"][i])
        t[f"{h}.ln_2.weight"] = mk(w["ln2w"][i])
        t[f"{h}.ln_2.bias"] = mk(w["ln2b"][i])
        t[f"{h}.mlp.c_fc.weight"] = mk(conv1d(w["fcw"][i], "c_fc"))
        t[f"{h}.mlp.c_fc.bias"] = mk(w["fcb"][i])
        t[f"{h}.mlp.c_proj.weight"] = mk(conv1d(w["fcprojw"][i], "mlp_c_proj"))
        t[f"{h}.mlp.c_proj.bias"] = mk(w["fcprojb"][i])
    t["transformer.ln_f.weight"] = mk(w["lnfw"])
    t["transformer.ln_f.bias"] = mk(w["lnfb"])
    return t, wpe_omitted


def llm_mojo_config(header: list[int], kind: str, wpe_omitted: bool) -> dict:
    """config.json for an llm_mojo_gpt2 export: a GPT2Config's fields (so the
    shape keys the Mojo loader reads are the same) with the stock-attention keys
    that would mislead removed and the architecture spelled out."""
    from transformers import GPT2Config

    _, _, maxT, V, L, H, C, Vp = header[:8]
    flags = header[MODEL_FLAGS_SLOT]
    cfg = GPT2Config(
        vocab_size=V,
        n_positions=maxT,
        n_embd=C,
        n_layer=L,
        n_head=H,
        layer_norm_epsilon=LAYER_NORM_EPS,
        bos_token_id=50256,
        eos_token_id=50256,
    ).to_dict()
    # Keys the existing gpt2-124m-fineweb-mojo config.json does not carry
    # (newer transformers serialise them), plus the stock-attention keys that
    # would mislead on a model that is not stock attention.
    for k in (
        "_name_or_path",
        "chunk_size_feed_forward",
        "id2label",
        "label2id",
        "is_encoder_decoder",
        "output_attentions",
        "output_hidden_states",
        "problem_type",
        "return_dict",
        "scale_attn_weights",
        "scale_attn_by_inverse_layer_idx",
        "reorder_and_upcast_attn",
        "add_cross_attention",
    ) + tuple(k for k in cfg if k.startswith("summary_")):
        cfg.pop(k, None)
    hd = C // H
    cfg["n_ctx"] = maxT  # the stock export carries it; GPT2Config no longer takes it
    cfg["model_type"] = LLM_MOJO_MODEL_TYPE
    cfg["architectures"] = ["LlmMojoGPT2LMHeadModel"]
    cfg["dtype"] = "bfloat16" if kind == "bf16" else "float32"
    cfg["activation_function"] = "gelu_new"
    cfg["tie_word_embeddings"] = True
    cfg["llm_mojo_arch_flags"] = flags
    cfg["rotary"] = bool(flags & ARCH_FLAG_ROTARY)
    if flags & ARCH_FLAG_ROTARY:
        cfg["rotary_type"] = "half_truncated"
        cfg["rotary_base"] = ROTARY_BASE
        cfg["rotary_note"] = (
            f"per head (head_dim {hd}, n = head_dim/4 = {hd // 4}): freq[i] = "
            "rotary_base ** -(i/(n-1)) for i < n, rotating the pair "
            "(x[i], x[i + head_dim/2]); every other dim is unrotated; applied "
            "to q and k after QK-norm, no learned position embedding"
        )
        cfg["wpe_omitted"] = wpe_omitted
    cfg["qk_norm"] = bool(flags & ARCH_FLAG_QK_NORM)
    if flags & ARCH_FLAG_QK_NORM:
        cfg["qk_norm_eps"] = QK_NORM_EPS
        cfg["qk_norm_affine"] = False
    # Effective scale of q.k (the default 1/sqrt(head_dim); llm.mojo takes it at
    # run time and the headline run used the default).
    cfg["attn_scale"] = 1.0 / (hd**0.5)
    return cfg


def export_llm_mojo(src: Path, output: Path, dtype: str, tokenizer_from: Path | None):
    from safetensors.torch import save_file

    header = read_header(src)
    flags = header[MODEL_FLAGS_SLOT]
    if flags & ~KNOWN_MODEL_FLAGS or flags == 0:
        raise SystemExit(f"{src}: flags {flags} are not rotary/QK-norm model flags")
    w, kind = read_bin_tensors(src, header)
    want = {"auto": kind, "bfloat16": "bf16", "float32": "fp32"}[dtype]
    if want == "bf16" and kind == "fp32":
        raise SystemExit(
            f"{src} holds fp32 weights; exporting as bf16 would not be "
            "bit-exact to the file (it rounds). Use --dtype auto or float32."
        )
    # bf16 file -> fp32 export is exact (widening); the values are unchanged.
    widen = want == "fp32" and kind == "bf16"
    tensors, wpe_omitted = llm_mojo_state_dict(w, header, kind, widen)
    if widen:
        kind = "fp32"
    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output / "model.safetensors"), metadata={"format": "pt"})
    cfg = llm_mojo_config(header, kind, wpe_omitted)
    (output / "config.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True) + "\n"
    )

    import transformers

    # Same keys the stock export's save_pretrained wrote for this repo.
    gen = {
        "_from_model_config": True,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "output_attentions": False,
        "output_hidden_states": False,
        "transformers_version": transformers.__version__,
        "use_cache": True,
    }
    (output / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n")
    if tokenizer_from is not None:
        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(tokenizer_from / name, output / name)
    else:
        from transformers import GPT2Tokenizer

        GPT2Tokenizer.from_pretrained("gpt2").save_pretrained(output)
    print(
        f"[export_to_hf] llm_mojo_gpt2 export: flags={flags} dtype={kind} "
        f"tensors={len(tensors)} wpe_omitted={wpe_omitted} -> {output}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="llm.mojo model_*.bin checkpoint"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="output directory for the HF export"
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["auto", "bfloat16", "float32"],
        help="dtype to export weights as (passed through to export_hf.py; "
        "default bfloat16). Rotary/QK-norm files: only the file's own dtype "
        "(default auto) keeps the export bit-exact",
    )
    parser.add_argument(
        "--spin",
        action="store_true",
        help="run upstream's post-export generation smoke test (needs a free "
        "CUDA device, accelerate, and flash-attention; off by default)",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="skip the CPU-side check that the export is finite and its matmul "
        "biases actually trained",
    )
    parser.add_argument(
        "--tokenizer-from",
        type=Path,
        default=None,
        help="rotary/QK-norm export: copy tokenizer.json and tokenizer_config.json "
        "from this directory instead of building them from the `gpt2` tokenizer",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input checkpoint not found: {args.input}")
    flags = read_header(args.input)[MODEL_FLAGS_SLOT]
    if flags:
        if flags & ~KNOWN_MODEL_FLAGS:
            raise SystemExit(f"{args.input}: unknown architecture flag bits {flags}")
        export_llm_mojo(
            args.input, args.output, args.dtype or "auto", args.tokenizer_from
        )
        return 0 if (not args.verify or verify_export(args.output)) else 1
    if args.dtype == "auto":
        args.dtype = None
    args.dtype = args.dtype or "bfloat16"
    if not EXPORT_HF_PY.is_file():
        parser.error(
            f"could not find llm.c's export_hf.py at {EXPORT_HF_PY} "
            "(is the third_party/llm.c submodule checked out?)"
        )

    with tempfile.TemporaryDirectory(prefix="llm_mojo_hf_export_") as tmpdir:
        patched = Path(tmpdir) / args.input.name
        make_llmc_compatible_copy(args.input, patched)

        cmd = [
            sys.executable,
            str(EXPORT_HF_PY),
            "--input",
            str(patched),
            "--output",
            str(args.output),
            "--dtype",
            args.dtype,
            # `spin` is upstream's post-export generation smoke test. It is off
            # by default here because it needs a free CUDA device, `accelerate`,
            # and a flash-attention build -- none of which hold while the box is
            # training, and its absence used to surface as a traceback AFTER a
            # perfectly good export had already been written. argparse declares
            # it `type=bool`, so any non-empty string is True and "" is False.
            "--spin",
            "1" if args.spin else "",
        ]
        print(f"[export_to_hf] running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return result.returncode

    if args.verify:
        return 0 if verify_export(args.output) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
