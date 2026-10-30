"""Sharded Muon under ZeRO-1/2 (design: docs/ai/zero_sharded_muon_design_*.md).

T2: threaded multi-rank CPU run through the real collectives. Every rank gets
a DIFFERENT planted batch. ZeRO-1 sharded Muon (matrix-granular owner space)
must be BIT-EXACT against ZeRO-0 Muon (every rank updates every matrix) at the
same world size after 3 steps: the gradients are all-reduced identically, Muon
runs on identical values on the owner as on a replica, and AdamW is
elementwise. Plus the ownership/segment bookkeeping (A_r, O_r, S) including
the tiny configs where a rank owns nothing.

ZeRO-2 (S2, bucketed reduce-scatter straight to the owners): bit-exact against
ZeRO-0 at step 1 for every tensor except wte, and within rtol 1e-4 over 3
steps for all params. wte differs because ZeRO-0 forms lm_k + enc_k locally
before the all-reduce, while the bucketed path reduces the two parts
separately, (lm_0+lm_1) + (enc_0+enc_1): fp32 addition is not associative.
"""

from std.testing import assert_equal, assert_true, TestSuite
from std.python import Python

from max.algorithm import sync_parallelize
from max.gpu.host import DeviceContext

from llmm.memory import MutMemPtr, heap_alloc
from llmm.zero import CpuCoordinator

from train_gpt2 import GPT2, GPT2_DTYPE, UNTIED_HEAD

comptime TT = 16
comptime BB = 1
comptime VOCAB = 64
comptime STEPS = 3


def _write_checkpoint(path: String, L: Int, NH: Int, C: Int) raises -> Int:
    """Synthetic fp32 checkpoint; returns the parameter count."""
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    var n = (
        VOCAB * C
        + TT * C
        + L * (C + C + 3 * C * C + 3 * C + C * C + C + C + C)
        + L * (4 * C * C + 4 * C + 4 * C * C + C)
        + 2 * C
    )
    var header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 3
    header[2] = TT
    header[3] = VOCAB
    header[4] = L
    header[5] = NH
    header[6] = C
    header[7] = VOCAB
    np.random.seed(7 + L)
    var weights = np.random.normal(0, 0.08, n).astype(np.float32)
    var f = builtins.open(path, "wb")
    _ = f.write(header.tobytes())
    _ = f.write(weights.tobytes())
    f.close()
    return n


def _remove(path: String) raises:
    var os = Python.import_module("os")
    try:
        _ = os.remove(path)
    except:
        pass


def _train[
    W: Int
](
    path: String,
    n: Int,
    stage: Int,
    steps: Int,
    result: MutMemPtr[DType.float32],
    accum: Int = 1,
    same_batch: Bool = False,
) raises:
    """Run `steps` steps on W threaded ranks at `stage`; rank r's final params
    go to result[r*n : (r+1)*n]. `accum` micro-steps per step (gradient
    accumulation); `same_batch` gives every rank the same batch."""
    var coord = heap_alloc[CpuCoordinator](1)
    coord[] = CpuCoordinator(W)
    var errors = heap_alloc[Int](W)
    for r in range(W):
        errors[unsafe_offset=r] = 0

    def _run_rank(rank: Int) {imm}:
        try:
            var ctx = DeviceContext(api="cpu")
            var model = GPT2["cpu", W](
                path,
                rank,
                stage,
                ctx,
                cpu_coordinator_ptr=Optional[
                    Pointer[CpuCoordinator, MutUntrackedOrigin]
                ](coord),
            )
            model.enable_muon(Scalar[DType.float32](0.95))
            var x = heap_alloc[SIMD[DType.int32, 1]](BB * TT)
            var y = heap_alloc[SIMD[DType.int32, 1]](BB * TT)
            var rk = 0 if same_batch else rank
            for step in range(steps):
                for ms in range(accum):
                    # A different planted batch per rank (unless same_batch),
                    # step and micro-step.
                    for i in range(BB * TT):
                        x[unsafe_offset=i] = Int32(
                            (rk * 7919 + i * 31 + step * 13 + ms * 11 + 5)
                            % VOCAB
                        )
                        y[unsafe_offset=i] = Int32(
                            (rk * 17 + i * 5 + step * 3 + ms * 7 + 1) % VOCAB
                        )
                    model.forward(
                        rebind[MutMemPtr[DType.int32]](x),
                        rebind[MutMemPtr[DType.int32]](y),
                        BB,
                        TT,
                        accum,
                    )
                    model.ctx.synchronize()
                    if ms == 0:
                        model.zero_gradients()
                    model.backward(accum, ms, step)
                    model.ctx.synchronize()
                model.update(
                    t=UInt32(step + 1),
                    learning_rate=Scalar[DType.float32](1e-3),
                    beta1=Scalar[DType.float32](0.9),
                    beta2=Scalar[DType.float32](0.999),
                    eps=Scalar[DType.float32](1e-8),
                    weight_decay=Scalar[DType.float32](0.01),
                )
                model.ctx.synchronize()
            if stage == 3:
                # Streaming: only the flat shard is resident. Gather the full
                # vector (collective: every rank) and check that the owner
                # space and the next forward's gathered windows agree
                # bit-exactly for the Muon matrices.
                # Sized to the live model: an untied build gathers N16 + head.
                var full = model.ctx.enqueue_create_buffer[GPT2_DTYPE](
                    model.num_parameters
                )
                model.ctx.synchronize()
                var fp = rebind[Pointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                    full.unsafe_ptr().as_unsafe_any_origin()
                )
                model._z3_gather_full(fp)
                model.ctx.synchronize()
                for i in range(len(model.muon_slots)):
                    ref sl = model.muon_slots[i]
                    if sl.owner != rank:
                        continue
                    for j in range(sl.len):
                        if (
                            full.unsafe_ptr()[unsafe_offset=sl.flat_off + j]
                            != model.muon_p_memory[
                                unsafe_offset=sl.local_off + j
                            ]
                        ):
                            raise Error("gather_full != owner shard")
                var Lc = model.config.num_layer
                var Cc = model.config.channels
                for layer in range(Lc):
                    model._z3_stream_layer(layer)
                    model.ctx.synchronize()
                    for i in range(len(model.muon_slots)):
                        ref sl = model.muon_slots[i]
                        if sl.l != layer:
                            continue
                        var wp = model.params.qkv_weight.unsafe_offset(
                            layer * 3 * Cc * Cc + sl.k * Cc * Cc
                        )
                        if sl.k == 3:
                            wp = model.params.attn_proj_weight.unsafe_offset(
                                layer * Cc * Cc
                            )
                        elif sl.k == 4:
                            wp = model.params.fc_weight.unsafe_offset(
                                layer * 4 * Cc * Cc
                            )
                        elif sl.k == 5:
                            wp = model.params.proj_weight.unsafe_offset(
                                layer * 4 * Cc * Cc
                            )
                        for j in range(sl.len):
                            if (
                                wp[unsafe_offset=j]
                                != full.unsafe_ptr()[
                                    unsafe_offset=sl.flat_off + j
                                ]
                            ):
                                raise Error(
                                    "streamed window != owner-space copy"
                                )
                for i in range(n):
                    result[unsafe_offset=rank * n + i] = full.unsafe_ptr()[
                        unsafe_offset=i
                    ].cast[DType.float32]()
            else:
                for i in range(n):
                    result[unsafe_offset=rank * n + i] = model.params_memory[
                        unsafe_offset=i
                    ].cast[DType.float32]()
            x.unsafe_free()
            y.unsafe_free()
        except e:
            print("rank", rank, "failed:", e)
            errors[unsafe_offset=rank] = 1
            coord[].abort()

    sync_parallelize(_run_rank, W)
    coord[].free()
    coord.unsafe_free()
    var bad = 0
    for r in range(W):
        bad += errors[unsafe_offset=r]
    errors.unsafe_free()
    assert_equal(bad, 0)


def _t2[W: Int](L: Int, NH: Int, C: Int) raises:
    var path = String("gpt2_zm_w") + String(W) + "_l" + String(L) + ".bin"
    var n = _write_checkpoint(path, L, NH, C)
    var ref_p = heap_alloc[Float32](W * n)
    var got = heap_alloc[Float32](W * n)
    try:
        _train[W](path, n, 0, STEPS, ref_p)
        _train[W](path, n, 1, STEPS, got)
    except e:
        _remove(path)
        ref_p.unsafe_free()
        got.unsafe_free()
        raise e^
    _remove(path)
    for r in range(W):
        for i in range(n):
            var a = ref_p[unsafe_offset=r * n + i]
            var b = got[unsafe_offset=r * n + i]
            if a != b:
                raise Error(
                    "ZeRO-1 sharded Muon != ZeRO-0 Muon at W="
                    + String(W)
                    + " rank "
                    + String(r)
                    + " idx "
                    + String(i)
                    + ": "
                    + String(a)
                    + " vs "
                    + String(b)
                )
    # Ranks agree with each other (the propagation reconstructed one model).
    for r in range(1, W):
        for i in range(n):
            if got[unsafe_offset=r * n + i] != got[unsafe_offset=i]:
                raise Error(
                    "ZeRO-1 ranks disagree at W="
                    + String(W)
                    + " rank "
                    + String(r)
                    + " idx "
                    + String(i)
                )
    ref_p.unsafe_free()
    got.unsafe_free()


def _t2_z2[W: Int](L: Int, NH: Int, C: Int, stage: Int = 2) raises:
    var path = (
        String("gpt2_zm")
        + String(stage)
        + "_w"
        + String(W)
        + "_l"
        + String(L)
        + ".bin"
    )
    var n = _write_checkpoint(path, L, NH, C)
    var wte_n = VOCAB * C
    var ref1 = heap_alloc[Float32](W * n)
    var got1 = heap_alloc[Float32](W * n)
    var ref3 = heap_alloc[Float32](W * n)
    var got3 = heap_alloc[Float32](W * n)
    try:
        _train[W](path, n, 0, 1, ref1)
        _train[W](path, n, stage, 1, got1)
        _train[W](path, n, 0, STEPS, ref3)
        _train[W](path, n, stage, STEPS, got3)
    except e:
        _remove(path)
        raise e^
    _remove(path)
    # Step 1: bit-exact everywhere except wte.
    var max_wte1 = Float32(0.0)
    for r in range(W):
        for i in range(n):
            var a = ref1[unsafe_offset=r * n + i]
            var b = got1[unsafe_offset=r * n + i]
            if i < wte_n:
                max_wte1 = max(max_wte1, abs(a - b) / max(abs(a), 1e-6))
                continue
            if a != b:
                raise Error(
                    "ZeRO-"
                    + String(stage)
                    + " step 1 != ZeRO-0 at W="
                    + String(W)
                    + " rank "
                    + String(r)
                    + " idx "
                    + String(i)
                    + ": "
                    + String(a)
                    + " vs "
                    + String(b)
                )
    # 3 steps: |a-b| <= 1e-4*|a| + 1e-6 everywhere. A pure rtol is ill-posed
    # for near-zero parameters (init std 0.08, so an element that lands near 0
    # carries ~1e-8 absolute fp32 noise from the cancellation). This is a
    # coarse multi-step check only: it catches gross errors (a skipped muon_g
    # zero lands at ~700x the tolerance, a dropped call B ~800x, swapped Q/K
    # ~1100x) but NOT small gradient errors, because AdamW and Muon are nearly
    # scale-invariant: a 1.001 scale on a routed Q block (all of it, one row
    # or one element) lands at 0.19x to 1.65x the tolerance, mostly passing,
    # while the W=4 baseline noise is 0.18x. Routing is pinned by the
    # step-1 bit-exact check above and by `_t2_z2_same` (S2 review). The
    # reported rel number is over elements with |a| >= 1e-2 (a true rtol).
    var max_rel = Float32(0.0)
    var max_abs = Float32(0.0)
    for r in range(W):
        for i in range(n):
            var a = ref3[unsafe_offset=r * n + i]
            var b = got3[unsafe_offset=r * n + i]
            max_abs = max(max_abs, abs(a - b))
            if abs(a) >= 1e-2:
                max_rel = max(max_rel, abs(a - b) / abs(a))
            if abs(a - b) > 1e-4 * abs(a) + 1e-6:
                raise Error(
                    "ZeRO-"
                    + String(stage)
                    + " 3-step beyond rtol 1e-4 + atol 1e-6 at W="
                    + String(W)
                    + " rank "
                    + String(r)
                    + " idx "
                    + String(i)
                    + ": "
                    + String(a)
                    + " vs "
                    + String(b)
                )
    for r in range(1, W):
        for i in range(n):
            if got3[unsafe_offset=r * n + i] != got3[unsafe_offset=i]:
                raise Error("ranks disagree at idx " + String(i))
    print(
        "t2_z" + String(stage) + " W=" + String(W) + " L=" + String(L),
        "step1_wte_max_rel=" + String(max_wte1),
        "step3_max_rel(|p|>=1e-2)=" + String(max_rel),
        "step3_max_abs=" + String(max_abs),
    )
    ref1.unsafe_free()
    got1.unsafe_free()
    ref3.unsafe_free()
    got3.unsafe_free()


def _t2_z2_same[W: Int](L: Int, NH: Int, C: Int) raises:
    """ZeRO-2 vs ZeRO-0, every rank on the SAME batch, 2 micro-steps per step,
    3 steps: bit-exact for every parameter.

    With identical per-rank gradients x, both reduction orders are exact
    multiples of x (ZeRO-0's W*(lm+enc) vs ZeRO-2's W*lm + W*enc, and the
    per-micro-step += vs the local accumulate-then-reduce), so the only source
    of the different-batch test's 3-step drift is gone and any routing,
    accumulation or zeroing error shows up as a hard mismatch. Measured during
    the S2 review: a 1.001 scale on ONE element of one routed Q gradient fails
    this, while `_t2_z2`'s 3-step tolerance passes it at 0.65x of the
    tolerance (only its step-1 bit-exact check catches it there).
    """
    var path = String("gpt2_zm2s_w") + String(W) + "_l" + String(L) + ".bin"
    var n = _write_checkpoint(path, L, NH, C)
    var ref_p = heap_alloc[Float32](W * n)
    var got = heap_alloc[Float32](W * n)
    try:
        _train[W](path, n, 0, STEPS, ref_p, accum=2, same_batch=True)
        _train[W](path, n, 2, STEPS, got, accum=2, same_batch=True)
    except e:
        _remove(path)
        ref_p.unsafe_free()
        got.unsafe_free()
        raise e^
    _remove(path)
    for r in range(W):
        for i in range(n):
            var a = ref_p[unsafe_offset=r * n + i]
            var b = got[unsafe_offset=r * n + i]
            if a != b:
                raise Error(
                    "ZeRO-2 same-batch accum != ZeRO-0 at W="
                    + String(W)
                    + " rank "
                    + String(r)
                    + " idx "
                    + String(i)
                    + ": "
                    + String(a)
                    + " vs "
                    + String(b)
                )
    ref_p.unsafe_free()
    got.unsafe_free()


def test_t2_zero2_same_batch_accum_bit_exact_w2() raises:
    _t2_z2_same[2](2, 2, 16)


def test_t2_zero2_same_batch_accum_bit_exact_w4() raises:
    _t2_z2_same[4](2, 2, 16)


def test_t2_zero2_w2() raises:
    _t2_z2[2](2, 2, 16)


def test_t2_zero2_w4() raises:
    _t2_z2[4](2, 2, 16)


def test_t2_zero2_w8_rank_owns_nothing() raises:
    _t2_z2[8](1, 1, 16)


def test_t2_zero3_w2() raises:
    _t2_z2[2](2, 2, 16, 3)


def test_t2_zero3_w4() raises:
    _t2_z2[4](2, 2, 16, 3)


def test_t2_zero3_w8_rank_owns_nothing() raises:
    _t2_z2[8](1, 1, 16, 3)


def test_t2_zero1_bit_exact_w2() raises:
    _t2[2](2, 2, 16)


def test_t2_zero1_bit_exact_w4() raises:
    _t2[4](2, 2, 16)


def test_t2_zero1_bit_exact_w8_rank_owns_nothing() raises:
    # L=1 at W=8: ranks 6 and 7 own no Muon matrix, S > opt.
    _t2[8](1, 1, 16)


def test_ownership_table() raises:
    """Ownership bookkeeping at W=4, L=2 (threaded, so the coordinator-backed
    constructor works): O_r sums to 6*L matrices' elements, every slot has
    owner (k*L+l)%W, and owner_idx = owner*S + local_off with S >= max O_r."""
    comptime W = 4
    var path = String("gpt2_zm_own.bin")
    var C = 16
    var L = 2
    _ = _write_checkpoint(path, L, 2, C)
    var coord = heap_alloc[CpuCoordinator](1)
    coord[] = CpuCoordinator(W)
    var errors = heap_alloc[Int](W)
    for r in range(W):
        errors[unsafe_offset=r] = 0

    def _run_rank(rank: Int) {imm}:
        try:
            var ctx = DeviceContext(api="cpu")
            var model = GPT2["cpu", W](
                path,
                rank,
                1,
                ctx,
                cpu_coordinator_ptr=Optional[
                    Pointer[CpuCoordinator, MutUntrackedOrigin]
                ](coord),
            )
            model.enable_muon(Scalar[DType.float32](0.95))
            var S = model.muon_shard_elems
            var total = 0
            var mine = 0
            for i in range(len(model.muon_slots)):
                ref sl = model.muon_slots[i]
                if sl.owner != (sl.k * L + sl.l) % W:
                    raise Error("wrong owner")
                if sl.owner_idx != sl.owner * S + sl.local_off:
                    raise Error("wrong owner_idx")
                if sl.local_off + sl.len > S:
                    raise Error("slot beyond S")
                total += sl.len
                if sl.owner == rank:
                    mine += sl.len
            if len(model.muon_slots) != 6 * L:
                raise Error("slot count")
            if mine != model.muon_owned_elems:
                raise Error("O_r mismatch")
            var groups_elems = 0
            for i in range(len(model.muon_owned_groups)):
                ref g = model.muon_owned_groups[i]
                groups_elems += g.count * g.rows * g.cols
            if groups_elems != mine:
                raise Error("owned groups do not cover O_r")
            var segs = model._adamw_shard_segments()
            var a_r = 0
            for i in range(len(segs)):
                a_r += segs[i].len
            if a_r != model.v_num_parameters:
                raise Error("A_r != v_num_parameters")
            # S7: flat m (and master, bf16 builds) are compact like v.
            if len(model.m_buf) != max(a_r, 1):
                raise Error("m_buf is not compact (A_r)")
            if model._m_count() != a_r:
                raise Error("_m_count != A_r")
        except e:
            print("rank", rank, "failed:", e)
            errors[unsafe_offset=rank] = 1
            coord[].abort()

    sync_parallelize(_run_rank, W)
    coord[].free()
    coord.unsafe_free()
    _remove(path)
    var bad = 0
    for r in range(W):
        bad += errors[unsafe_offset=r]
    errors.unsafe_free()
    assert_equal(bad, 0)


def _sizes_w8(
    L: Int, exp_s: Int, exp_opt: Int, exp_o: List[Int], exp_a: List[Int]
) raises:
    """Run enable_muon on 8 threaded ZeRO-1 ranks of an L-layer C=16 model and
    compare S, opt, O_r and A_r against values computed independently of the
    trainer (a Python model of _compute_param_sizes / owner (k*L+l) mod W /
    [r*opt, (r+1)*opt) intersected with the AdamW ranges). Also checks the
    segment invariants: nonempty, ascending, shard_rel = flat_start - lo,
    compact_off = prefix sum, all inside [lo, lo+opt)."""
    comptime W = 8
    var path = String("gpt2_zm_sz_l") + String(L) + ".bin"
    _ = _write_checkpoint(path, L, 1, 16)
    var coord = heap_alloc[CpuCoordinator](1)
    coord[] = CpuCoordinator(W)
    var errors = heap_alloc[Int](W)
    var got = heap_alloc[Int](4 * W)
    for r in range(W):
        errors[unsafe_offset=r] = 0

    def _run_rank(rank: Int) {imm}:
        try:
            var ctx = DeviceContext(api="cpu")
            var model = GPT2["cpu", W](
                path,
                rank,
                1,
                ctx,
                cpu_coordinator_ptr=Optional[
                    Pointer[CpuCoordinator, MutUntrackedOrigin]
                ](coord),
            )
            model.enable_muon(Scalar[DType.float32](0.95))
            var opt = model.optimizer_num_parameters
            var lo = rank * opt
            var segs = model._adamw_shard_segments()
            var acc = 0
            var prev_end = lo
            for i in range(len(segs)):
                ref sg = segs[i]
                if sg.len <= 0:
                    raise Error("empty segment")
                if sg.flat_start < prev_end:
                    raise Error("segments not ascending/disjoint")
                if sg.shard_rel != sg.flat_start - lo:
                    raise Error("shard_rel != flat_start - lo")
                if sg.compact_off != acc:
                    raise Error("compact_off is not the prefix sum")
                if sg.flat_start + sg.len > lo + opt:
                    raise Error("segment beyond the shard")
                acc += sg.len
                prev_end = sg.flat_start + sg.len
            got[unsafe_offset=4 * rank] = model.muon_shard_elems
            got[unsafe_offset=4 * rank + 1] = opt
            got[unsafe_offset=4 * rank + 2] = model.muon_owned_elems
            got[unsafe_offset=4 * rank + 3] = model.v_num_parameters
            if acc != model.v_num_parameters:
                raise Error("sum of segment lengths != v_num_parameters")
        except e:
            print("rank", rank, "failed:", e)
            errors[unsafe_offset=rank] = 1
            coord[].abort()

    sync_parallelize(_run_rank, W)
    coord[].free()
    coord.unsafe_free()
    _remove(path)
    var bad = 0
    for r in range(W):
        bad += errors[unsafe_offset=r]
    errors.unsafe_free()
    assert_equal(bad, 0)
    for r in range(W):
        assert_equal(got[unsafe_offset=4 * r], exp_s)
        assert_equal(got[unsafe_offset=4 * r + 1], exp_opt)
        assert_equal(got[unsafe_offset=4 * r + 2], exp_o[r])
        assert_equal(got[unsafe_offset=4 * r + 3], exp_a[r])
    got.unsafe_free()


def test_ownership_sizes_tiny_w8() raises:
    """The review-flagged tiny W=8 configs: L=1 has ranks owning no matrix
    (O_r = 0) and S > opt; both L=1 and L=2 have a rank with A_r = 0."""
    # The expected per-rank sizes are for the 16-tensor layout. The untied
    # build appends a head block, which shifts opt and every A_r; the generic
    # invariants (ascending disjoint segments, prefix-sum compact_off, sum ==
    # v_num_parameters) are still checked by the other ownership tests there.
    comptime if UNTIED_HEAD:
        print("SKIP test_ownership_sizes_tiny_w8: 16-tensor expectations")
        return
    _sizes_w8(
        1,
        1024,
        576,
        [256, 256, 256, 256, 1024, 1024, 0, 0],
        [576, 576, 160, 48, 48, 0, 64, 48],
    )
    _sizes_w8(
        2,
        1280,
        984,
        [1280, 1280, 1280, 1280, 256, 256, 256, 256],
        [984, 360, 72, 120, 0, 128, 0, 64],
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
