#!/usr/bin/env python3
"""Prove a rotary / QK-norm safetensors export (scripts/export_to_hf.py) is the
checkpoint it came from.

  (a) round trip: every tensor of <dir>/model.safetensors, with the Conv1D
      transposes undone, equals the raw .bin element for element (bit patterns,
      so bf16 is compared as uint16); wpe is either present and equal or
      omitted with the .bin's wpe all zero.
  (b) independent forward: load the safetensors into the PyTorch reference
      (train_gpt2.py, GPT2_ROTARY / GPT2_QK_NORM set from the config) and print
      the loss and first per-position losses of a real batch, for comparison
      with `infer_gpt2 --loss` on the same file and tokens.

  python scripts/check_hf_export.py --bin model_5700.bin --dir out/ \
      --data data/.fineweb10B/fineweb_val_000000.bin [--mutant c_attn|attn_c_proj|c_fc|mlp_c_proj]

Run it from a scratch directory, not the repo root: train_gpt2.py's gpt2_*.bin
are symlinks into the shared checkout in a worktree (nothing here writes them,
but the habit is the rule). `--mutant NAME` re-exports with that one Conv1D
transpose forgotten into a temp dir and checks THAT; the run must FAIL.
Prints key=value lines; exit status 0 only when every requested check passes.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def load_st(path: Path) -> dict:
    from safetensors import safe_open

    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            out[k] = t
    return out


def bits(t) -> np.ndarray:
    """Raw bit pattern of a torch tensor as uint16 (bf16) or uint32 (fp32)."""
    import torch

    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).numpy().view(np.uint16)
    assert t.dtype == torch.float32, t.dtype
    return t.contiguous().numpy().view(np.uint32)


def widen_bits(a: np.ndarray) -> np.ndarray:
    """bf16 bit patterns -> the fp32 bit patterns of the same values."""
    return a.astype(np.uint32) << 16


def raw_bin(path: Path):
    """Independent read of the .bin: header, then {name: bit-pattern array}."""
    with open(path, "rb") as f:
        h = np.frombuffer(f.read(1024), dtype=np.int32)
        version = int(h[1])
        dt = np.uint32 if version == 3 else np.uint16
        blob = np.frombuffer(f.read(), dtype=dt)
    maxT, V, L, H, C, Vp = (int(x) for x in h[2:8])
    sizes = [
        ("wte", Vp * C),
        ("wpe", maxT * C),
        ("ln1w", L * C),
        ("ln1b", L * C),
        ("qkvw", L * 3 * C * C),
        ("qkvb", L * 3 * C),
        ("attprojw", L * C * C),
        ("attprojb", L * C),
        ("ln2w", L * C),
        ("ln2b", L * C),
        ("fcw", L * 4 * C * C),
        ("fcb", L * 4 * C),
        ("fcprojw", L * C * 4 * C),
        ("fcprojb", L * C),
        ("lnfw", C),
        ("lnfb", C),
    ]
    assert sum(n for _, n in sizes) == blob.size, "blob size != header-derived size"
    out, off = {}, 0
    for name, n in sizes:
        out[name] = blob[off : off + n]
        off += n
    return h, out


def round_trip(bin_path: Path, st: dict) -> bool:
    h, raw = raw_bin(bin_path)
    maxT, V, L, H, C, Vp = (int(x) for x in h[2:8])
    flags = int(h[20])
    ok = True
    n_cmp = 0

    def eq(name, got, want):
        nonlocal ok, n_cmp
        got = bits(got)
        want = want.reshape(got.shape)
        if got.dtype == np.uint32 and want.dtype == np.uint16:
            want = widen_bits(want)  # fp32 export of a bf16 file (exact)
        same = np.array_equal(got, want)
        n_cmp += got.size
        if not same:
            ok = False
            print(
                f"mismatch tensor={name} differing={int((got != want).sum())}/{got.size}"
            )

    wte = raw["wte"].reshape(Vp, C)
    eq("transformer.wte.weight", st["transformer.wte.weight"], wte[:V])
    pad_zero = not wte[V:].any()
    print(
        f"wte_padding_rows_zero={pad_zero} (rows {V}..{Vp - 1} are dropped by the export)"
    )
    if "transformer.wpe.weight" in st:
        eq("transformer.wpe.weight", st["transformer.wpe.weight"], raw["wpe"])
        print("wpe=present")
    else:
        wpe_zero = not raw["wpe"].any()
        print(f"wpe=omitted bin_wpe_all_zero={wpe_zero} rotary={bool(flags & 1)}")
        ok &= wpe_zero and bool(flags & 1)
    per_layer = [
        ("ln_1.weight", "ln1w", (C,), False),
        ("ln_1.bias", "ln1b", (C,), False),
        ("attn.c_attn.weight", "qkvw", (3 * C, C), True),
        ("attn.c_attn.bias", "qkvb", (3 * C,), False),
        ("attn.c_proj.weight", "attprojw", (C, C), True),
        ("attn.c_proj.bias", "attprojb", (C,), False),
        ("ln_2.weight", "ln2w", (C,), False),
        ("ln_2.bias", "ln2b", (C,), False),
        ("mlp.c_fc.weight", "fcw", (4 * C, C), True),
        ("mlp.c_fc.bias", "fcb", (4 * C,), False),
        ("mlp.c_proj.weight", "fcprojw", (C, 4 * C), True),
        ("mlp.c_proj.bias", "fcprojb", (C,), False),
    ]
    for suffix, key, shape, transposed in per_layer:
        arr = raw[key].reshape((L,) + shape)
        for i in range(L):
            name = f"transformer.h.{i}.{suffix}"
            got = st[name]
            if transposed:
                # HF Conv1D is [in, out]; our .bin is [out, in].
                want_shape = shape[::-1]
                if tuple(got.shape) != want_shape:
                    print(
                        f"mismatch tensor={name} shape={tuple(got.shape)} want={want_shape}"
                    )
                    ok = False
                    continue
                eq(name, got.t().contiguous(), arr[i])
            else:
                eq(name, got, arr[i])
    eq("transformer.ln_f.weight", st["transformer.ln_f.weight"], raw["lnfw"])
    eq("transformer.ln_f.bias", st["transformer.ln_f.bias"], raw["lnfb"])
    expected = {
        "transformer.wte.weight",
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
    }
    expected |= {f"transformer.h.{i}.{sfx}" for i in range(L) for sfx, *_ in per_layer}
    if "transformer.wpe.weight" in st:
        expected.add("transformer.wpe.weight")
    extra = set(st) ^ expected
    if extra:
        print(f"tensor_name_set_differs={sorted(extra)}")
        ok = False
    print(
        f"round_trip tensors={len(st)} elements_compared={n_cmp} status={'pass' if ok else 'fail'}"
    )
    return ok


def reference_forward(st: dict, cfg: dict, data: Path, B: int, T: int) -> None:
    flags = int(cfg.get("llm_mojo_arch_flags", 0))
    os.environ["GPT2_ROTARY"] = "1" if flags & 1 else "0"
    os.environ["GPT2_QK_NORM"] = "1" if flags & 4 else "0"
    os.environ["GPT2_ATTN_SCALE"] = "0"  # default 1/sqrt(hd) == config attn_scale
    import torch

    sys.path.insert(0, str(REPO))
    import train_gpt2 as ref

    assert ref.ROTARY == bool(flags & 1) and ref.QK_NORM == bool(flags & 4)
    model = ref.GPT(
        ref.GPTConfig(
            block_size=cfg["n_positions"],
            vocab_size=cfg["vocab_size"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
        )
    )
    sd = model.state_dict()
    transposed = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )
    loaded = set()
    for k, v in st.items():
        lk = k
        for ln in ("ln_1", "ln_2", "ln_f"):
            if k.endswith(f".{ln}.weight"):
                lk = k[: -len(".weight")] + ".gamma"
            if k.endswith(f".{ln}.bias"):
                lk = k[: -len(".bias")] + ".beta"
        v = v.float()
        if any(k.endswith(t) for t in transposed):
            v = v.t()
        assert sd[lk].shape == v.shape, (lk, sd[lk].shape, v.shape)
        with torch.no_grad():
            sd[lk].copy_(v)
        loaded.add(lk)
    if "transformer.wpe.weight" not in st:
        with torch.no_grad():
            sd["transformer.wpe.weight"].zero_()  # omitted: all zero, unused
        loaded.add("transformer.wpe.weight")
    # lm_head is tied to wte in the reference.
    missing = [
        k
        for k in sd
        if not k.endswith(".attn.bias") and k not in loaded and k != "lm_head.weight"
    ]
    assert not missing, missing
    assert model.lm_head.weight is model.transformer.wte.weight
    model.eval()
    with open(data, "rb") as f:
        f.seek(1024)
        toks = np.frombuffer(f.read((B * T + 1) * 2), dtype=np.uint16).astype(np.int64)
    x = torch.from_numpy(toks[:-1]).view(B, T)
    y = torch.from_numpy(toks[1:]).view(B, T)
    with torch.no_grad():
        logits, _ = model(x, y, True)
        lp = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
        )
    print(
        f"reference_loss={lp.mean().item():.8f} B={B} T={T} rotary={ref.ROTARY} qk_norm={ref.QK_NORM}"
    )
    print("reference_row0_losses=" + " ".join(f"{v:.6f}" for v in lp[:8].tolist()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bin", required=True, type=Path)
    ap.add_argument("--dir", required=True, type=Path, help="export directory")
    ap.add_argument("--data", type=Path, help="token shard for the forward check")
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--mutant", choices=["c_attn", "attn_c_proj", "c_fc", "mlp_c_proj"])
    ap.add_argument("--no-forward", action="store_true")
    args = ap.parse_args()

    directory = args.dir
    if args.mutant:
        import importlib

        ex = importlib.import_module("export_to_hf")  # scripts/ is on sys.path

        real = getattr(ex, "conv1d")

        def broken(arr, name=""):
            return np.ascontiguousarray(arr) if name == args.mutant else real(arr, name)

        setattr(ex, "conv1d", broken)
        tmp = tempfile.mkdtemp(prefix="hf_mutant_")
        ex.export_llm_mojo(args.bin, Path(tmp), "auto", args.dir)
        directory = Path(tmp)
        print(f"mutant={args.mutant} (that Conv1D transpose forgotten) dir={tmp}")

    st = load_st(directory / "model.safetensors")
    cfg = json.loads((directory / "config.json").read_text())
    ok = round_trip(args.bin, st)
    if args.data and not args.no_forward:
        try:
            reference_forward(st, cfg, args.data, args.B, args.T)
        except AssertionError as e:
            print(f"reference_forward status=fail ({e!r})")
            ok = False
    print(f"result check=hf_export status={'pass' if ok else 'fail'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
