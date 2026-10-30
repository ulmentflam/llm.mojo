"""W1: `adam_lr_mult` (the -l_adam flag) under Muon, planted-value style.

One Muon step on a tiny CPU model with identical weights and batch, at
multiplier 1 and multiplier 4 (and 0.5 for a non-integer case, 0 for the
frozen case). Expected:
  * every Muon-matrix element is BIT-IDENTICAL across multipliers (a mutant
    that applies the multiplier to the Muon lr fails here);
  * every AdamW-range element that moved moves by exactly mult x its mult-1
    displacement (wd included: AdamW decay is lr-scaled too), so a mutant that
    forgets the multiplier (delta4 == delta1) fails here;
  * at multiplier 0 every AdamW element is BIT-IDENTICAL to its initial
    value (frozen: lr-scaled decay included; the moments still advance);
  * the Muon matrices did move and many AdamW elements moved (the checks are
    not vacuous).
Covers the unsharded path (`_update_hybrid`, ZeRO-0) and the sharded path
(`_update_sharded_muon`, ZeRO-1 W=2, threaded CPU ranks).
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
comptime L = 2
comptime NH = 2
comptime C = 16


def _write_checkpoint(path: String) raises -> Int:
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
    np.random.seed(11)
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


def _run[
    W: Int
](
    path: String,
    n: Int,
    stage: Int,
    mult: Float32,
    init: MutMemPtr[DType.float32],
    after: MutMemPtr[DType.float32],
    is_adam: MutMemPtr[DType.float32],
) raises:
    """One Muon step; rank 0's pre/post params go to init/after, and is_adam[i]
    is 1.0 where element i belongs to an AdamW range."""
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
            model.enable_muon(Scalar[DType.float32](0.95), mult)
            var x = heap_alloc[SIMD[DType.int32, 1]](BB * TT)
            var y = heap_alloc[SIMD[DType.int32, 1]](BB * TT)
            for i in range(BB * TT):
                x[unsafe_offset=i] = Int32((i * 31 + 5) % VOCAB)
                y[unsafe_offset=i] = Int32((i * 5 + 1) % VOCAB)
            if rank == 0:
                for i in range(n):
                    init[unsafe_offset=i] = model.params_memory[
                        unsafe_offset=i
                    ].cast[DType.float32]()
                    is_adam[unsafe_offset=i] = 0.0
                var ranges = model._adamw_ranges()
                for k in range(len(ranges) // 2):
                    for i in range(ranges[2 * k], ranges[2 * k + 1]):
                        is_adam[unsafe_offset=i] = 1.0
            model.forward(
                rebind[MutMemPtr[DType.int32]](x),
                rebind[MutMemPtr[DType.int32]](y),
                BB,
                TT,
                1,
            )
            model.ctx.synchronize()
            model.zero_gradients()
            model.backward(1, 0, 0)
            model.ctx.synchronize()
            model.update(
                t=UInt32(1),
                learning_rate=Scalar[DType.float32](1e-3),
                beta1=Scalar[DType.float32](0.9),
                beta2=Scalar[DType.float32](0.999),
                eps=Scalar[DType.float32](1e-8),
                weight_decay=Scalar[DType.float32](0.01),
            )
            model.ctx.synchronize()
            if rank == 0:
                for i in range(n):
                    after[unsafe_offset=i] = model.params_memory[
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


def _check[W: Int](stage: Int, mult: Float32) raises:
    var path = String("gpt2_alr_w") + String(W) + "_s" + String(stage) + ".bin"
    var n = _write_checkpoint(path)
    var init1 = heap_alloc[Float32](n)
    var a1 = heap_alloc[Float32](n)
    var m1 = heap_alloc[Float32](n)
    var init2 = heap_alloc[Float32](n)
    var a2 = heap_alloc[Float32](n)
    var m2 = heap_alloc[Float32](n)
    try:
        _run[W](path, n, stage, Float32(1.0), init1, a1, m1)
        _run[W](path, n, stage, mult, init2, a2, m2)
    except e:
        _remove(path)
        raise e^
    _remove(path)
    var muon_moved = 0
    var adam_moved = 0
    for i in range(n):
        if init1[unsafe_offset=i] != init2[unsafe_offset=i]:
            raise Error("runs started from different weights")
        var d1 = a1[unsafe_offset=i] - init1[unsafe_offset=i]
        var d2 = a2[unsafe_offset=i] - init1[unsafe_offset=i]
        if m1[unsafe_offset=i] == 0.0:
            # Muon matrix element: independent of the multiplier, exactly.
            if a1[unsafe_offset=i] != a2[unsafe_offset=i]:
                raise Error(
                    "Muon matrix element " + String(i) + " changed with mult"
                )
            if d1 != 0.0:
                muon_moved += 1
        else:
            if mult == 0.0 and a2[unsafe_offset=i] != init1[unsafe_offset=i]:
                # lr 0 must freeze exactly: the decoupled decay is lr-scaled
                # too, so not even wd may move an AdamW element.
                raise Error(
                    "AdamW element " + String(i) + " moved under mult 0"
                )
            var ad1 = d1 if d1 >= 0 else -d1
            if ad1 > 1e-5:
                adam_moved += 1
                var want = d1 * mult
                var err = d2 - want
                var aerr = err if err >= 0 else -err
                var aw = want if want >= 0 else -want
                if aerr > 2e-3 * aw + 1e-7:
                    raise Error(
                        "AdamW element "
                        + String(i)
                        + ": delta "
                        + String(d2)
                        + " != mult * mult1 delta "
                        + String(want)
                    )
    assert_true(muon_moved > 500, "Muon matrices did not move")
    assert_true(adam_moved > 100, "AdamW ranges barely moved")
    init1.unsafe_free()
    a1.unsafe_free()
    m1.unsafe_free()
    init2.unsafe_free()
    a2.unsafe_free()
    m2.unsafe_free()


def test_adam_lr_mult_unsharded_x4() raises:
    _check[1](0, Float32(4.0))


def test_adam_lr_mult_unsharded_x0p5() raises:
    _check[1](0, Float32(0.5))


def test_adam_lr_mult_unsharded_x0_frozen() raises:
    _check[1](0, Float32(0.0))


def test_adam_lr_mult_sharded_zero1_w2_x4() raises:
    _check[2](1, Float32(4.0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
