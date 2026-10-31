#!/usr/bin/env python3
"""Compare dump_grads_gpt2's `lm_head` (17th gradient) against the untied
reference debug state, with test_gpt2.mojo's per-tensor criterion
(maxdiff <= atol + rtol*ref_maxabs, and the L2 norm within a factor of 3)
plus a relative L2 error bound, global and per vocab row.

Usage: check_dump_lm_head.py <dump_dir> [debug_state.bin]
Pure stdlib (no numpy): the system python is enough.
"""

import array
import math
import struct
import sys

ATOL, RTOL, L2_FACTOR = 0.01, 0.05, 3.0
REL_L2_MAX, ROW_REL_L2_MAX = 1e-3, 0.25


def read_f32(f, n):
    a = array.array("f")
    a.fromfile(f, n)
    return a


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    dump_dir = sys.argv[1]
    state = sys.argv[2] if len(sys.argv) > 2 else "gpt2_124M_untied_debug_state.bin"
    with open(state, "rb") as f:
        hdr = struct.unpack("256i", f.read(1024))
        if hdr[4] != 1:
            print(f"error: {state} header[4]={hdr[4]}, not an untied state")
            return 2
        B, T = hdr[2], hdr[3]
        f.seek(B * T * 4 * 2, 1)  # x, y
        V, V_p, C = 50257, 50304, 768
        f.seek(B * T * V * 4 + 4, 1)  # logits + loss
        # grads follow the loss: the 16 tied tensors (padded wte) then lm_head.
        # The file carries more after them, so read exactly N floats.
        n_total = 163109376 if len(sys.argv) < 4 else int(sys.argv[3])
        ref_all = read_f32(f, n_total)
    n_head = V_p * C
    ref = ref_all[len(ref_all) - n_head :]
    ours = read_f32(open(f"{dump_dir}/lm_head.bin", "rb"), n_head)
    # rows >= V are padding; test_gpt2 checks V*C elements
    n = V * C
    maxdiff = max(abs(a - b) for a, b in zip(ours[:n], ref[:n]))
    ref_max = max(abs(b) for b in ref[:n])
    ref_l2 = math.sqrt(sum(b * b for b in ref[:n]))
    our_l2 = math.sqrt(sum(a * a for a in ours[:n]))
    thr = ATOL + RTOL * ref_max
    # test_gpt2's criterion alone is weak for this tensor: 50145 of the 50257
    # rows have every |g| below the 0.51 threshold, so zeroing a whole vocab
    # tile (rows 32000-38399) or any single row passes it. Also require the
    # relative L2 error, globally and per row (measured on the GB10, fp32
    # no-TF32: 2.2e-5 global, 0.025 worst row; a zeroed tile gives >= 3.2e-3
    # globally and a zeroed row 1.0 for that row).
    err2 = 0.0
    worst_row = 0.0
    for r in range(V):
        a = ours[r * C : (r + 1) * C]
        b = ref[r * C : (r + 1) * C]
        e2 = sum((x - y) * (x - y) for x, y in zip(a, b))
        b2 = sum(y * y for y in b)
        err2 += e2
        if b2 > 0.0:
            worst_row = max(worst_row, math.sqrt(e2 / b2))
        elif e2 > 0.0:
            worst_row = math.inf
    rel_l2 = math.sqrt(err2) / ref_l2
    ok = (
        maxdiff <= thr
        and our_l2 <= L2_FACTOR * ref_l2
        and our_l2 * L2_FACTOR >= ref_l2
        and rel_l2 <= REL_L2_MAX
        and worst_row <= ROW_REL_L2_MAX
    )
    print(
        f"dlm_head maxdiff={maxdiff:.3e} threshold={thr:.3e} "
        f"l2_ratio={our_l2 / ref_l2:.4f} ref_maxabs={ref_max:.3e} "
        f"rel_l2_err={rel_l2:.3e} (max {REL_L2_MAX:g}) "
        f"worst_row_rel_l2_err={worst_row:.3e} (max {ROW_REL_L2_MAX:g}) "
        f"status={'ok' if ok else 'fail'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
