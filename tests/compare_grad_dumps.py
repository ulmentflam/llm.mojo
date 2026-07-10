#!/usr/bin/env python3
"""Compare two grad-dump dirs (raw fp32 .bin per tensor). Reference = dir B.
Usage: compare_grads.py <dir_test> <dir_ref> [--csv out.csv]
Prints per-tensor cosine + relL2, pass/fail (cos>0.99 && relL2<0.1), summary.
"""

import glob
import os
import sys

import numpy as np


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.bin"))):
        name = os.path.basename(p)[:-4]
        out[name] = np.fromfile(p, dtype=np.float32)
    return out


def main():
    dt, dr = sys.argv[1], sys.argv[2]
    csv = None
    if "--csv" in sys.argv:
        csv = sys.argv[sys.argv.index("--csv") + 1]
    a, b = load(dt), load(dr)
    keys = sorted(set(a) & set(b))
    missing = sorted(set(a) ^ set(b))
    if missing:
        print("WARN unmatched:", missing[:10], "...")
    rows = []
    for k in keys:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            print(f"SHAPE MISMATCH {k}: {x.shape} vs {y.shape}")
            continue
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        cos = (
            float(np.dot(x, y) / (nx * ny))
            if nx > 0 and ny > 0
            else (1.0 if nx == ny else 0.0)
        )
        rel = float(np.linalg.norm(x - y) / ny) if ny > 0 else float(nx)
        nnan = int(np.sum(~np.isfinite(x)))
        rows.append((k, cos, rel, nnan, y.size))
    npass = sum(1 for _, c, r, _, _ in rows if c > 0.99 and r < 0.1)
    print(f"{'tensor':32s} {'cosine':>9s} {'relL2':>9s} {'pass':>5s} {'nonfin':>7s}")
    for k, c, r, nn, sz in rows:
        ok = "OK" if (c > 0.99 and r < 0.1) else "FAIL"
        flag = f" NONFIN={nn}" if nn else ""
        print(f"{k:32s} {c:9.4f} {r:9.4f} {ok:>5s}{flag}")
    cosv = np.array([c for _, c, _, _, _ in rows])
    relv = np.array([r for _, _, r, _, _ in rows])
    print("\n=== SUMMARY ===")
    print(f"tensors: {len(rows)}  pass(cos>0.99&&relL2<0.1): {npass}/{len(rows)}")
    print(
        f"cosine : min {cosv.min():.4f}  median {np.median(cosv):.4f}  max {cosv.max():.4f}"
    )
    print(
        f"relL2  : min {relv.min():.4f}  median {np.median(relv):.4f}  max {relv.max():.4f}"
    )
    worst = sorted(rows, key=lambda t: t[1])[:8]
    print("worst by cosine:")
    for k, c, r, _, _ in worst:
        print(f"  {k:32s} cos={c:.4f} relL2={r:.4f}")
    if csv:
        with open(csv, "w") as f:
            f.write("tensor,cosine,relL2,nonfinite,size\n")
            for k, c, r, nn, sz in rows:
                f.write(f"{k},{c:.6f},{r:.6f},{nn},{sz}\n")
        print("wrote", csv)


if __name__ == "__main__":
    main()
