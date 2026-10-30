"""Sharded Muon under ZeRO-1 (design: docs/ai/zero_sharded_muon_design_*.md).

T2: threaded multi-rank CPU run through the real collectives. Every rank gets
a DIFFERENT planted batch. ZeRO-1 sharded Muon (matrix-granular owner space)
must be BIT-EXACT against ZeRO-0 Muon (every rank updates every matrix) at the
same world size after 3 steps: the gradients are all-reduced identically, Muon
runs on identical values on the owner as on a replica, and AdamW is
elementwise. Plus the ownership/segment bookkeeping (A_r, O_r, S) including
the tiny configs where a rank owns nothing.
"""

from std.testing import assert_equal, assert_true, TestSuite
from std.python import Python

from max.algorithm import sync_parallelize
from max.gpu.host import DeviceContext

from llmm.memory import MutMemPtr, heap_alloc
from llmm.zero import CpuCoordinator

from train_gpt2 import GPT2, GPT2_DTYPE

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
) raises:
    """Run STEPS steps on W threaded ranks at `stage`; rank r's final params go
    to result[r*n : (r+1)*n]."""
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
            for step in range(steps):
                # A different planted batch per rank and per step.
                for i in range(BB * TT):
                    x[unsafe_offset=i] = Int32(
                        (rank * 7919 + i * 31 + step * 13 + 5) % VOCAB
                    )
                    y[unsafe_offset=i] = Int32(
                        (rank * 17 + i * 5 + step * 3 + 1) % VOCAB
                    )
                model.forward(
                    rebind[MutMemPtr[DType.int32]](x),
                    rebind[MutMemPtr[DType.int32]](y),
                    BB,
                    TT,
                )
                model.ctx.synchronize()
                model.zero_gradients()
                model.backward()
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
