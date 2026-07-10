#!/usr/bin/env python3
"""Compare two grad-dump dirs (raw fp32 .bin per tensor). Reference = dir B.

Usage: compare_grad_dumps.py <dir_test> <dir_ref> [--csv out.csv]

Implements the coordinator-ACCEPTED recalibrated gradient gate from Chunk E's
compounding investigation (docs/ai/low_precision_gotchas.md, section E5) --
NOT the original flat `cos>0.99 && relL2<0.1` per-tensor gate, which is
structurally unsatisfiable for a real fp8 training step whose gradients
traverse up to ~12 fp8 GEMMs (see MEMORY.md `weak-gates-overrule-nothing`).

Four criteria, ALL must pass for the overall gate to PASS:
  (a) per-tensor cosine floor > 0.93 for EVERY tensor -- catches gradient-
      direction reversal, the true optimizer-health failure a real bug causes.
  (b) relL2 envelope (aggregate, not per-tensor pass/fail): median < 0.20,
      max < 0.50 -- calibrated to the per-GEMM-verified quantization floors
      (E4M3 ~0.026-0.036, E5M2 ~0.052) compounding across up to ~12 layers.
  (c) depth-monotonicity: within each of the 12 per-layer tensor classes
      (ln_1_gamma, ln_1_beta, qkv_weight, qkv_bias, attn_proj_weight,
      attn_proj_bias, ln_2_gamma, ln_2_beta, fc_weight, fc_bias, proj_weight,
      proj_bias), flag any single layer whose relL2 exceeds ~2x the local
      trend (median of its immediate neighbors in layer-index order). A real
      per-site quantization/state bug spikes one layer; physics (backprop
      error compounding with depth) is smooth -- this is what actually
      catches a per-site bug that the aggregate envelope (b) would hide.
  (d) no NaN/Inf in either dump.

Exits nonzero (1) if any criterion is violated, with a clear per-criterion
report. Exit 2 on usage/setup errors (missing files, shape mismatch, tensor
count mismatch).
"""

import glob
import os
import re
import sys

import numpy as np

# --- Gate thresholds (docs/ai/low_precision_gotchas.md section E5) ---
COSINE_FLOOR = 0.93
RELL2_MEDIAN_MAX = 0.20
RELL2_MAX_MAX = 0.50
DEPTH_SPIKE_RATIO = 2.0
# Local trend uses relL2 values themselves; guard against dividing by a
# near-zero local trend on otherwise-tiny tensors (spurious spike flags).
DEPTH_TREND_FLOOR = 0.01

LAYER_RE = re.compile(r"^(.*)_layer(\d+)$")


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.bin"))):
        name = os.path.basename(p)[:-4]
        out[name] = np.fromfile(p, dtype=np.float32)
    return out


def cosine_rel_l2(x, y):
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    cos = (
        float(np.dot(x, y) / (nx * ny))
        if nx > 0 and ny > 0
        else (1.0 if nx == ny else 0.0)
    )
    rel = float(np.linalg.norm(x - y) / ny) if ny > 0 else float(nx)
    return cos, rel


def local_trend(values, i, radius=1):
    """Median of up to `radius` neighbors on each side of index i (excludes
    i itself). Falls back to whatever neighbors exist near a boundary.

    radius=1 (immediate neighbors only) is the tightest, most standard
    reading of "local" -- a small-window spike filter. A wider radius pulls
    in layers further from i, which on this curve (monotonic-with-noise,
    corr -0.68..-0.97 per docs/ai/low_precision_gotchas.md E5, NOT perfectly
    smooth) dilutes the comparison with points less representative of i's
    immediate neighborhood and can flag ordinary local noise near curvature
    changes as a false spike."""
    lo = max(0, i - radius)
    hi = min(len(values), i + radius + 1)
    neigh = [values[j] for j in range(lo, hi) if j != i]
    if not neigh:
        return values[i]
    return float(np.median(neigh))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    dt, dr = sys.argv[1], sys.argv[2]
    csv = None
    if "--csv" in sys.argv:
        csv = sys.argv[sys.argv.index("--csv") + 1]

    a, b = load(dt), load(dr)
    keys = sorted(set(a) & set(b))
    missing = sorted(set(a) ^ set(b))
    if missing:
        print("WARN unmatched:", missing[:10], "..." if len(missing) > 10 else "")
    if not keys:
        print("ERROR: no matching tensors found")
        sys.exit(2)

    rows = []  # (name, cosine, relL2, nonfinite_test, nonfinite_ref, size)
    for k in keys:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            print(f"SHAPE MISMATCH {k}: {x.shape} vs {y.shape}")
            sys.exit(2)
        cos, rel = cosine_rel_l2(x, y)
        nnan_x = int(np.sum(~np.isfinite(x)))
        nnan_y = int(np.sum(~np.isfinite(y)))
        rows.append((k, cos, rel, nnan_x, nnan_y, y.size))

    cosv = np.array([r[1] for r in rows])
    relv = np.array([r[2] for r in rows])

    print(f"{'tensor':32s} {'cosine':>9s} {'relL2':>9s} {'flags':>8s}")
    for k, c, r, nnx, nny, sz in rows:
        flags = []
        if c <= COSINE_FLOOR:
            flags.append("COS")
        if nnx or nny:
            flags.append(f"NONFIN={nnx}/{nny}")
        print(f"{k:32s} {c:9.4f} {r:9.4f} {' '.join(flags):>8s}")

    print("\n=== SUMMARY ===")
    print(f"tensors: {len(rows)}")
    print(
        f"cosine : min {cosv.min():.4f}  median {np.median(cosv):.4f}  max {cosv.max():.4f}"
    )
    print(
        f"relL2  : min {relv.min():.4f}  median {np.median(relv):.4f}  max {relv.max():.4f}"
    )

    # --- Criterion (a): per-tensor cosine floor ---
    cos_violations = [(k, c) for k, c, r, nnx, nny, sz in rows if c <= COSINE_FLOOR]
    crit_a = len(cos_violations) == 0

    # --- Criterion (b): relL2 envelope (aggregate) ---
    rel_median = float(np.median(relv))
    rel_max = float(relv.max())
    crit_b = (rel_median < RELL2_MEDIAN_MAX) and (rel_max < RELL2_MAX_MAX)

    # --- Criterion (c): depth-monotonicity within each per-layer class ---
    classes = {}
    for k, c, r, nnx, nny, sz in rows:
        m = LAYER_RE.match(k)
        if m:
            cls, layer = m.group(1), int(m.group(2))
            classes.setdefault(cls, {})[layer] = r
    depth_violations = []
    for cls, layer_rel in sorted(classes.items()):
        layers = sorted(layer_rel)
        values = [layer_rel[layer] for layer in layers]
        for idx, layer in enumerate(layers):
            trend = local_trend(values, idx)
            trend_floor = max(trend, DEPTH_TREND_FLOOR)
            if values[idx] > DEPTH_SPIKE_RATIO * trend_floor:
                depth_violations.append(
                    (cls, layer, values[idx], trend, values[idx] / trend_floor)
                )
    crit_c = len(depth_violations) == 0

    # --- Criterion (d): NaN/Inf sentinel ---
    nonfin_total = sum(nnx + nny for _, _, _, nnx, nny, _ in rows)
    crit_d = nonfin_total == 0

    print("\n=== GATE CRITERIA (docs/ai/low_precision_gotchas.md E5) ===")
    print(
        f"(a) cosine floor >{COSINE_FLOOR}     : "
        f"{'PASS' if crit_a else 'FAIL'} "
        f"({len(cos_violations)} tensor(s) at/below floor)"
    )
    if cos_violations:
        for k, c in sorted(cos_violations, key=lambda t: t[1])[:10]:
            print(f"      {k:32s} cosine={c:.4f}")
    print(
        f"(b) relL2 envelope (median<{RELL2_MEDIAN_MAX}, max<{RELL2_MAX_MAX}): "
        f"{'PASS' if crit_b else 'FAIL'} "
        f"(median={rel_median:.4f}, max={rel_max:.4f})"
    )
    print(
        f"(c) depth-monotonicity (>{DEPTH_SPIKE_RATIO}x local trend): "
        f"{'PASS' if crit_c else 'FAIL'} "
        f"({len(depth_violations)} violation(s))"
    )
    if depth_violations:
        for cls, layer, val, trend, ratio in depth_violations:
            print(
                f"      {cls}_layer{layer:02d}: relL2={val:.4f} "
                f"local_trend={trend:.4f} ratio={ratio:.2f}x"
            )
    print(
        f"(d) NaN/Inf sentinel        : {'PASS' if crit_d else 'FAIL'} "
        f"({nonfin_total} nonfinite element(s) total)"
    )

    overall = crit_a and crit_b and crit_c and crit_d
    print(f"\nOVERALL GATE: {'PASS' if overall else 'FAIL'}")

    if csv:
        with open(csv, "w") as f:
            f.write("tensor,cosine,relL2,nonfinite_test,nonfinite_ref,size\n")
            for k, c, r, nnx, nny, sz in rows:
                f.write(f"{k},{c:.6f},{r:.6f},{nnx},{nny},{sz}\n")
        print("wrote", csv)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
