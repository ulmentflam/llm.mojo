"""Untied LM head (W9) under ZeRO: threaded multi-rank CPU bit-exactness (slice S3).

Design: docs/ai/untied_lm_head_design_2026-10-30.md (critical findings 6 and
8, section 3.6, slice S3). Modelled on tests/test_zero_muon.mojo. Built with
`make test-mojo TEST_DEFINES="-D LLMM_UNTIED_HEAD=1"`; in the tied build every
test prints SKIP (the trainer is compiled once per flag value).

The head is ACTIVE and planted: the checkpoint is a version-13 (untied) file
whose head block is an independent random draw (std 0.08, seed 1000 + L), so
head != wte everywhere and a zero head (which blocks the body gradient at step
0, finding 4) never hides a body bug. Every "bit-exact" below is `==` per
element, never a tolerance.

Coverage:
  1. ZeRO-1 == ZeRO-0 for all 17 tensors at W=2, 4, 8; 1 and 3 steps; AdamW and
     Muon; DIFFERENT batch per rank (this includes step-1 wte, finding 6: the
     untied wte carries only the encoder part). Every head element and every
     wte element must also have MOVED (a shard boundary that drops a row
     cannot pass).
  2. Ownership with the 17th tensor: opt, the AdamW segments per rank and the
     head's overlap with each shard against values computed here from the
     layout, independently of the trainer. The tests assert where the head
     lands: L=2 (N16 > H) puts the whole head on rank W-1 at W=2, 4, 8; W=8
     with L=1 straddles ranks 6 and 7.
  3. ZeRO-2 (bucketed reduce-scatter, head as its own Bucket-1 segment) and
     ZeRO-3 (streaming; the head is gathered one vocab tile at a time) with the
     head ACTIVE equal ZeRO-0 on all 17 tensors, `==` per element, different
     batch per rank, 1 and 3 steps, AdamW and Muon, W=2/4/8 (L=2) and W=8
     (L=1, head straddles ranks 6/7). Step 1 includes wte with no exemption:
     finding 6's prediction. After the last ZeRO-3 update `_z3_gather_full`
     must equal each rank's flat shard, and every gathered head tile
     (`_lm_head_tile`) must equal the ZeRO-0 head rows, bit-exactly.
  4. ZeRO-2/3 (W=2, 4) with the idle head (`head_tied = True`) equal ZeRO-0 with the idle
     head bit-exactly (same batch, accumulation, 3 steps), and the idle head's
     planted values do not move. W=8 is not run here: with 8 identical
     per-rank gradients the reduction orders differ by an ulp in wte AND wpe
     (measured: wte[18], wpe[9]) even on the idle head, so the "same batch is
     exact" argument holds for W=2, 4 only; test_zero_muon covers W=8 at its
     tolerance.

  5. The -untie_at transition (`untie_head()`) at ZeRO-1/2/3 (W=2; ZeRO-2/3 also
     W=4), AdamW and Muon: 1 tied step, untie, 2 trained steps == ZeRO-0 on all
     17 tensors, and right after `untie_head()` the head is a bitwise copy of wte
     on every rank (ZeRO-3 via the full gather). The direct check matters:
     ZeRO-0/1/2 share the local-copy branch, so a broken copy there is equal in
     reference and candidate (measured: a mutant skipping that copy passed every
     ZeRO-2 transition test before the check existed).

Finding 6 as measured: see the S6 section of the design doc (the ZeRO-2/3
step-1 different-batch check now runs here).
"""

from std.testing import assert_equal, assert_true, TestSuite
from std.python import Python
from std.sys import simd_width_of, is_defined

from max.algorithm import sync_parallelize
from max.gpu.host import DeviceContext

from llmm.memory import MutMemPtr, heap_alloc
from llmm.zero import CpuCoordinator

from train_gpt2 import GPT2, GPT2_DTYPE, UNTIED_HEAD

comptime TT = 16
comptime BB = 1
comptime VOCAB = 64
comptime STEPS = 3
# Tensor indices of the six per-layer Muon matrices' parents and the tail.
comptime N_TENSORS = 17


def _sizes(L: Int, C: Int) -> List[Int]:
    """Per-tensor element counts, in flat-buffer order (16 tensors + head)."""
    var s = List[Int]()
    s.append(VOCAB * C)  # 0 wte
    s.append(TT * C)  # 1 wpe
    s.append(L * C)  # 2 ln_1_gamma
    s.append(L * C)  # 3 ln_1_beta
    s.append(L * 3 * C * C)  # 4 qkv_weight (Muon)
    s.append(L * 3 * C)  # 5 qkv_bias
    s.append(L * C * C)  # 6 attn_proj_weight (Muon)
    s.append(L * C)  # 7 attn_proj_bias
    s.append(L * C)  # 8 ln_2_gamma
    s.append(L * C)  # 9 ln_2_beta
    s.append(L * 4 * C * C)  # 10 fc_weight (Muon)
    s.append(L * 4 * C)  # 11 fc_bias
    s.append(L * 4 * C * C)  # 12 proj_weight (Muon)
    s.append(L * C)  # 13 proj_bias
    s.append(C)  # 14 ln_f_gamma
    s.append(C)  # 15 ln_f_beta
    s.append(VOCAB * C)  # 16 lm_head
    return s^


def _write_untied_checkpoint(
    path: String, L: Int, NH: Int, C: Int
) raises -> Int:
    """Version-13 (untied) fp32 file: 16 tensors then an independent planted
    head. Returns the parameter count N17."""
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    var sz = _sizes(L, C)
    var n16 = 0
    for i in range(N_TENSORS - 1):
        n16 += sz[i]
    var header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 13
    header[2] = TT
    header[3] = VOCAB
    header[4] = L
    header[5] = NH
    header[6] = C
    header[7] = VOCAB
    np.random.seed(7 + L)
    var weights = np.random.normal(0, 0.08, n16).astype(np.float32)
    np.random.seed(1000 + L)
    var head = np.random.normal(0, 0.08, sz[16]).astype(np.float32)
    var f = builtins.open(path, "wb")
    _ = f.write(header.tobytes())
    _ = f.write(weights.tobytes())
    _ = f.write(head.tobytes())
    f.close()
    return n16 + sz[16]


def _remove(path: String) raises:
    var os = Python.import_module("os")
    try:
        _ = os.remove(path)
    except:
        pass


def _nullp() -> MutMemPtr[DType.float32]:
    var zero = 0
    return MutMemPtr[DType.float32](unsafe_from_address=zero)


def _train[
    W: Int
](
    path: String,
    n: Int,
    stage: Int,
    steps: Int,
    result: MutMemPtr[DType.float32],
    init: MutMemPtr[DType.float32],
    muon: Bool = True,
    accum: Int = 1,
    same_batch: Bool = False,
    idle_head: Bool = False,
    tiles: MutMemPtr[DType.float32] = _nullp(),
    check_tiles: Bool = False,
    untie_after: Int = 0,
) raises:
    """`steps` steps on W threaded ranks at `stage`; rank r's final 17-tensor
    params go to result[r*n:(r+1)*n]; rank 0's initial params to init[0:n].
    `idle_head` sets head_tied. At stage 3 with the head active, `tiles` (W*H
    floats) receives each rank's gathered head tiles, and the full gather is
    checked against the rank's own flat shard."""
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
            comptime if UNTIED_HEAD:
                model.head_tied = idle_head or untie_after > 0
                if untie_after > 0:
                    model.untie_step = untie_after
            if model.num_parameters != n:
                raise Error("model has the wrong tensor count")
            if rank == 0:
                for i in range(n):
                    init[unsafe_offset=i] = model.params_memory[
                        unsafe_offset=i
                    ].cast[DType.float32]()
            if muon:
                model.enable_muon(Scalar[DType.float32](0.95))
            var x = heap_alloc[SIMD[DType.int32, 1]](BB * TT)
            var y = heap_alloc[SIMD[DType.int32, 1]](BB * TT)
            var rk = 0 if same_batch else rank
            for step in range(steps):
                comptime if UNTIED_HEAD:
                    if untie_after > 0 and step == untie_after:
                        model.untie_head()
                        model.ctx.synchronize()
                        # Right after the transition the head must be a
                        # bitwise copy of wte on every rank and stage (S4
                        # spec). Checked here, not only via ZeRO-k == ZeRO-0,
                        # because ZeRO-0/1/2 share the local-copy branch: a
                        # broken copy there is identical in reference and
                        # candidate and only this check sees it.
                        var hn = model.param_sizes[16]
                        var ho = model.param_offset(16)
                        if model.z3_streaming:
                            var hfull = model.ctx.enqueue_create_buffer[
                                GPT2_DTYPE
                            ](model.num_parameters)
                            model.ctx.synchronize()
                            model._z3_gather_full(
                                rebind[
                                    Pointer[Scalar[GPT2_DTYPE], MutAnyOrigin]
                                ](hfull.unsafe_ptr().as_unsafe_any_origin())
                            )
                            model.ctx.synchronize()
                            for i in range(hn):
                                if hfull.unsafe_ptr()[unsafe_offset=ho + i] != (
                                    hfull.unsafe_ptr()[unsafe_offset=i]
                                ):
                                    raise Error(
                                        "after untie_head (ZeRO-3) head != wte"
                                        " at "
                                        + String(i)
                                    )
                        else:
                            for i in range(hn):
                                if (
                                    model.params_memory[unsafe_offset=ho + i]
                                    != model.params_memory[unsafe_offset=i]
                                ):
                                    raise Error(
                                        "after untie_head head != wte at "
                                        + String(i)
                                    )
                for ms in range(accum):
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
                # Streaming: only the flat shard is resident; gather the full
                # vector (collective: every rank).
                var full = model.ctx.enqueue_create_buffer[GPT2_DTYPE](
                    model.num_parameters
                )
                model.ctx.synchronize()
                var fp = rebind[Pointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                    full.unsafe_ptr().as_unsafe_any_origin()
                )
                model._z3_gather_full(fp)
                model.ctx.synchronize()
                if not model.z3_streaming:
                    raise Error("ZeRO-3 test is not streaming")
                # The gather must reproduce this rank's own flat shard.
                var opt = model.optimizer_num_parameters
                var lo = rank * opt
                for j in range(min(opt, n - lo)):
                    if muon:
                        # Sharded Muon keeps the four matrix tensors in the
                        # owner space; their flat-shard slots are not live.
                        var live = True
                        for t in range(N_TENSORS):
                            if t == 4 or t == 6 or t == 10 or t == 12:
                                var o = model._z3_param_offset(t)
                                if (
                                    lo + j >= o
                                    and lo + j < o + model.param_sizes[t]
                                ):
                                    live = False
                        if not live:
                            continue
                    if full.unsafe_ptr()[unsafe_offset=lo + j] != (
                        model.params_memory[unsafe_offset=j]
                    ):
                        raise Error(
                            "gather_full != flat shard at rank "
                            + String(rank)
                            + " elem "
                            + String(j)
                        )
                comptime if UNTIED_HEAD:
                    if (
                        check_tiles
                        and not idle_head
                        and not is_defined["LLMM_Z3_HEAD_RESIDENT"]()
                    ):
                        var c = model.config.channels
                        var vp = model.config.padded_vocab_size
                        var tr = model._lm_head_tile_rows()
                        var t0 = 0
                        while t0 < vp:
                            var oc = min(tr, vp - t0)
                            var wt = model._lm_head_tile(t0, oc)
                            model.ctx.synchronize()
                            for j in range(oc * c):
                                tiles[
                                    unsafe_offset=rank * vp * c + t0 * c + j
                                ] = wt[unsafe_offset=j].cast[DType.float32]()
                            t0 += tr
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


def _first_diff(
    a: MutMemPtr[DType.float32], b: MutMemPtr[DType.float32], W: Int, n: Int
) -> Int:
    """Index (r*n + i) of the first element with a != b, or -1."""
    for k in range(W * n):
        if a[unsafe_offset=k] != b[unsafe_offset=k]:
            return k
    return -1


def _tensor_of(i: Int, sz: List[Int]) -> String:
    var names = List[String]()
    names.append("wte")
    names.append("wpe")
    names.append("ln1g")
    names.append("ln1b")
    names.append("qkvw")
    names.append("qkvb")
    names.append("aprojw")
    names.append("aprojb")
    names.append("ln2g")
    names.append("ln2b")
    names.append("fcw")
    names.append("fcb")
    names.append("projw")
    names.append("projb")
    names.append("lnfg")
    names.append("lnfb")
    names.append("lm_head")
    var acc = 0
    for t in range(N_TENSORS):
        if i < acc + sz[t]:
            return names[t] + "[" + String(i - acc) + "]"
        acc += sz[t]
    return "?"


def _zero1_vs_zero0[
    W: Int
](L: Int, NH: Int, C: Int, muon: Bool, stage: Int = 1) raises:
    """Different batch per rank; ZeRO-`stage` == ZeRO-0 element-for-element for
    all 17 tensors after 1 step and after 3 steps (step 1 includes wte: finding
    6); ranks agree; the head and wte have moved from their planted init (every
    element). At stage 3 the gathered head tiles equal the ZeRO-0 head."""
    var tag = String("m") if muon else String("a")
    var path = (
        String("gpt2_zu_")
        + tag
        + "_s"
        + String(stage)
        + "_w"
        + String(W)
        + "_l"
        + String(L)
        + ".bin"
    )
    var n = _write_untied_checkpoint(path, L, NH, C)
    var sz = _sizes(L, C)
    var r0a = heap_alloc[Float32](W * n)
    var r1a = heap_alloc[Float32](W * n)
    var r0b = heap_alloc[Float32](W * n)
    var r1b = heap_alloc[Float32](W * n)
    var init = heap_alloc[Float32](n)
    var scratch = heap_alloc[Float32](n)
    var tiles = heap_alloc[Float32](W * sz[16])
    try:
        _train[W](path, n, 0, 1, r0a, init, muon=muon)
        _train[W](path, n, stage, 1, r1a, scratch, muon=muon)
        _train[W](path, n, 0, STEPS, r0b, scratch, muon=muon)
        _train[W](
            path,
            n,
            stage,
            STEPS,
            r1b,
            scratch,
            muon=muon,
            tiles=tiles,
            check_tiles=True,
        )
    except e:
        _remove(path)
        raise e^
    _remove(path)
    var d1 = _first_diff(r0a, r1a, W, n)
    if d1 >= 0:
        raise Error(
            "step 1: ZeRO-"
            + String(stage)
            + " != ZeRO-0 at W="
            + String(W)
            + " rank "
            + String(d1 // n)
            + " "
            + _tensor_of(d1 % n, sz)
            + ": "
            + String(r0a[unsafe_offset=d1])
            + " vs "
            + String(r1a[unsafe_offset=d1])
        )
    var d3 = _first_diff(r0b, r1b, W, n)
    if d3 >= 0:
        raise Error(
            "step 3: ZeRO-"
            + String(stage)
            + " != ZeRO-0 at W="
            + String(W)
            + " rank "
            + String(d3 // n)
            + " "
            + _tensor_of(d3 % n, sz)
            + ": "
            + String(r0b[unsafe_offset=d3])
            + " vs "
            + String(r1b[unsafe_offset=d3])
        )
    for r in range(1, W):
        for i in range(n):
            if r1b[unsafe_offset=r * n + i] != r1b[unsafe_offset=i]:
                raise Error(
                    "ZeRO-"
                    + String(stage)
                    + " ranks disagree at W="
                    + String(W)
                    + " rank "
                    + String(r)
                    + " "
                    + _tensor_of(i, sz)
                )
    # The planted head is nonzero and distinct from wte, and every head and
    # wte element moved in step 1 (softmax gives every vocab row a gradient;
    # weight decay moves the rest): a shard boundary that dropped a row would
    # leave elements at their init value.
    var head_off = n - sz[16]
    for i in range(sz[16]):
        assert_true(init[unsafe_offset=head_off + i] != 0.0, "head init is 0")
        assert_true(
            init[unsafe_offset=head_off + i] != init[unsafe_offset=i],
            "head init == wte init",
        )
    for r in range(W):
        for i in range(sz[0]):
            if r1a[unsafe_offset=r * n + i] == init[unsafe_offset=i]:
                raise Error("wte element did not move: " + _tensor_of(i, sz))
        for i in range(sz[16]):
            if (
                r1a[unsafe_offset=r * n + head_off + i]
                == init[unsafe_offset=head_off + i]
            ):
                raise Error(
                    "head element did not move at rank "
                    + String(r)
                    + ": "
                    + _tensor_of(head_off + i, sz)
                )
    if stage == 3 and not is_defined["LLMM_Z3_HEAD_RESIDENT"]():
        # (With the head resident the window is refreshed by the next forward,
        # so after the last update it is legitimately stale: not checked.)
        # Every gathered head tile (rank-local copy) is the ZeRO-0 head.
        var hn = sz[16]
        for r in range(W):
            for i in range(hn):
                if (
                    tiles[unsafe_offset=r * hn + i]
                    != r0b[unsafe_offset=head_off + i]
                ):
                    raise Error(
                        "gathered head tile != ZeRO-0 head at rank "
                        + String(r)
                        + " elem "
                        + String(i)
                    )
    tiles.unsafe_free()
    r0a.unsafe_free()
    r1a.unsafe_free()
    r0b.unsafe_free()
    r1b.unsafe_free()
    init.unsafe_free()
    scratch.unsafe_free()


def _skip_tied(name: String) -> Bool:
    comptime if UNTIED_HEAD:
        return False
    else:
        print("SKIP " + name + ": tied build")
        return True


def test_zero1_equals_zero0_muon_w2() raises:
    if _skip_tied("test_zero1_equals_zero0_muon_w2"):
        return
    _zero1_vs_zero0[2](2, 2, 16, True)


def test_zero1_equals_zero0_muon_w4() raises:
    if _skip_tied("test_zero1_equals_zero0_muon_w4"):
        return
    _zero1_vs_zero0[4](2, 2, 16, True)


def test_zero1_equals_zero0_muon_w8() raises:
    if _skip_tied("test_zero1_equals_zero0_muon_w8"):
        return
    _zero1_vs_zero0[8](2, 2, 16, True)


def test_zero1_equals_zero0_muon_w8_head_straddles_l1() raises:
    # L=1: the head straddles ranks 6 and 7 (asserted in the ownership test).
    if _skip_tied("test_zero1_equals_zero0_muon_w8_head_straddles_l1"):
        return
    _zero1_vs_zero0[8](1, 1, 16, True)


def test_zero1_equals_zero0_adamw_w2() raises:
    if _skip_tied("test_zero1_equals_zero0_adamw_w2"):
        return
    _zero1_vs_zero0[2](2, 2, 16, False)


def test_zero1_equals_zero0_adamw_w4() raises:
    if _skip_tied("test_zero1_equals_zero0_adamw_w4"):
        return
    _zero1_vs_zero0[4](2, 2, 16, False)


def test_zero1_equals_zero0_adamw_w8() raises:
    if _skip_tied("test_zero1_equals_zero0_adamw_w8"):
        return
    _zero1_vs_zero0[8](2, 2, 16, False)


def test_zero1_equals_zero0_adamw_w8_head_straddles_l1() raises:
    if _skip_tied("test_zero1_equals_zero0_adamw_w8_head_straddles_l1"):
        return
    _zero1_vs_zero0[8](1, 1, 16, False)


# --------------------------------------------------------------------------
# Ownership and shard sizes with the 17th tensor.
# --------------------------------------------------------------------------


def _ownership[W: Int](L: Int, NH: Int, C: Int) raises:
    """Run enable_muon on W ZeRO-1 ranks, record opt and the AdamW segments, and
    compare them with values computed here from the layout. Then assert where
    the head lands (see the callers)."""
    var path = String("gpt2_zu_own_w") + String(W) + "_l" + String(L) + ".bin"
    var n = _write_untied_checkpoint(path, L, NH, C)
    var sz = _sizes(L, C)
    var off = List[Int]()
    var acc = 0
    for t in range(N_TENSORS + 1):
        off.append(acc)
        if t < N_TENSORS:
            acc += sz[t]
    var n16 = off[16]
    var head_n = sz[16]
    assert_equal(off[N_TENSORS], n)

    # Expected optimizer shard length: ceil(N/W) rounded up to the SIMD width.
    comptime SA = simd_width_of[GPT2_DTYPE]()
    var base = (n + W - 1) // W
    var opt = ((base + SA - 1) // SA) * SA

    # AdamW ranges: everything but the four Muon matrix tensors (4, 6, 10, 12),
    # with the tail split at the head (so the head is its own range).
    var ranges = List[Int]()
    ranges.append(0)
    ranges.append(off[4])
    ranges.append(off[5])
    ranges.append(off[6])
    ranges.append(off[7])
    ranges.append(off[10])
    ranges.append(off[11])
    ranges.append(off[12])
    ranges.append(off[13])
    ranges.append(n16)
    ranges.append(n16)
    ranges.append(n)

    # Per-rank expectations, independent of the trainer.
    var exp_nseg = List[Int]()
    var exp_seg = List[Int]()  # flat 8 ranks x 8 segs x (start, len)
    var exp_head = List[Int]()
    for _ in range(W * 8 * 2):
        exp_seg.append(0)
    for r in range(W):
        var lo = r * opt
        var hi = min(lo + opt, n)
        var ns = 0
        var hl = 0
        for k in range(len(ranges) // 2):
            var a = max(ranges[2 * k], lo)
            var b = min(ranges[2 * k + 1], hi)
            if b <= a:
                continue
            exp_seg[(r * 8 + ns) * 2] = a
            exp_seg[(r * 8 + ns) * 2 + 1] = b - a
            ns += 1
            if a >= n16:
                hl += b - a
        exp_nseg.append(ns)
        exp_head.append(hl)

    # The straddle / landing claims, asserted from the computed layout so a
    # config change cannot silently drop the coverage.
    var head_ranks = List[Int]()
    var total_head = 0
    for r in range(W):
        if exp_head[r] > 0:
            head_ranks.append(r)
            total_head += exp_head[r]
    assert_equal(total_head, head_n)
    if L == 2:
        # N16 > H: rank W-1 starts below N16, so the whole head is on rank W-1.
        assert_equal(len(head_ranks), 1)
        assert_equal(head_ranks[0], W - 1)
        assert_equal(exp_head[W - 1], head_n)
        assert_true((W - 1) * opt <= n16, "rank W-1 must start before the head")
    else:
        # L=1, W=8: r*opt falls strictly inside (head_off, head_off + H).
        assert_equal(W, 8)
        assert_equal(len(head_ranks), 2)
        assert_equal(head_ranks[0], 6)
        assert_equal(head_ranks[1], 7)
        assert_true(7 * opt > n16 and 7 * opt < n16 + head_n, "no straddle")
        assert_true(exp_head[6] > 0 and exp_head[7] > 0, "straddle empty side")
        assert_equal(exp_head[6] + exp_head[7], head_n)

    var coord = heap_alloc[CpuCoordinator](1)
    coord[] = CpuCoordinator(W)
    var errors = heap_alloc[Int](W)
    var got = heap_alloc[Int](W * (3 + 16))
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
            comptime if UNTIED_HEAD:
                model.head_tied = False
            model.enable_muon(Scalar[DType.float32](0.95))
            var segs = model._adamw_shard_segments()
            var b = rank * (3 + 16)
            got[unsafe_offset=b] = model.optimizer_num_parameters
            got[unsafe_offset=b + 1] = model.param_offset(16)
            got[unsafe_offset=b + 2] = len(segs)
            for i in range(min(len(segs), 8)):
                got[unsafe_offset=b + 3 + 2 * i] = segs[i].flat_start
                got[unsafe_offset=b + 3 + 2 * i + 1] = segs[i].len
            # Planted: the head block of the loaded params is the file's head
            # (nonzero, != wte) at the offset the layout says.
            var seen = 0
            for i in range(head_n):
                if (
                    model.params_memory[unsafe_offset=n16 + i]
                    != model.params_memory[unsafe_offset=i]
                    and model.params_memory[unsafe_offset=n16 + i] != 0
                ):
                    seen += 1
            if seen != head_n:
                raise Error("loaded head is not the planted head block")
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
        var b = r * (3 + 16)
        assert_equal(got[unsafe_offset=b], opt)
        assert_equal(got[unsafe_offset=b + 1], n16)
        assert_equal(got[unsafe_offset=b + 2], exp_nseg[r])
        var head_got = 0
        for i in range(exp_nseg[r]):
            assert_equal(
                got[unsafe_offset=b + 3 + 2 * i], exp_seg[(r * 8 + i) * 2]
            )
            assert_equal(
                got[unsafe_offset=b + 3 + 2 * i + 1],
                exp_seg[(r * 8 + i) * 2 + 1],
            )
            if got[unsafe_offset=b + 3 + 2 * i] >= n16:
                head_got += got[unsafe_offset=b + 3 + 2 * i + 1]
        assert_equal(head_got, exp_head[r])
    got.unsafe_free()


def test_head_ownership_w2_whole_head_on_last_rank() raises:
    if _skip_tied("test_head_ownership_w2_whole_head_on_last_rank"):
        return
    _ownership[2](2, 2, 16)


def test_head_ownership_w4_whole_head_on_last_rank() raises:
    if _skip_tied("test_head_ownership_w4_whole_head_on_last_rank"):
        return
    _ownership[4](2, 2, 16)


def test_head_ownership_w8_whole_head_on_last_rank() raises:
    if _skip_tied("test_head_ownership_w8_whole_head_on_last_rank"):
        return
    _ownership[8](2, 2, 16)


def test_head_ownership_w8_l1_straddles_ranks_6_and_7() raises:
    if _skip_tied("test_head_ownership_w8_l1_straddles_ranks_6_and_7"):
        return
    _ownership[8](1, 1, 16)


# --------------------------------------------------------------------------
# ZeRO-2/3 with the head ACTIVE (S6): bit-exact against ZeRO-0.
# --------------------------------------------------------------------------


def test_zero2_equals_zero0_adamw_w2() raises:
    if _skip_tied("test_zero2_equals_zero0_adamw_w2"):
        return
    _zero1_vs_zero0[2](2, 2, 16, False, 2)


def test_zero2_equals_zero0_adamw_w4() raises:
    if _skip_tied("test_zero2_equals_zero0_adamw_w4"):
        return
    _zero1_vs_zero0[4](2, 2, 16, False, 2)


def test_zero2_equals_zero0_adamw_w8() raises:
    if _skip_tied("test_zero2_equals_zero0_adamw_w8"):
        return
    _zero1_vs_zero0[8](2, 2, 16, False, 2)


def test_zero2_equals_zero0_adamw_w8_head_straddles_l1() raises:
    if _skip_tied("test_zero2_equals_zero0_adamw_w8_head_straddles_l1"):
        return
    _zero1_vs_zero0[8](1, 1, 16, False, 2)


def test_zero2_equals_zero0_muon_w2() raises:
    if _skip_tied("test_zero2_equals_zero0_muon_w2"):
        return
    _zero1_vs_zero0[2](2, 2, 16, True, 2)


def test_zero2_equals_zero0_muon_w4() raises:
    if _skip_tied("test_zero2_equals_zero0_muon_w4"):
        return
    _zero1_vs_zero0[4](2, 2, 16, True, 2)


def test_zero2_equals_zero0_muon_w8() raises:
    if _skip_tied("test_zero2_equals_zero0_muon_w8"):
        return
    _zero1_vs_zero0[8](2, 2, 16, True, 2)


def test_zero2_equals_zero0_muon_w8_head_straddles_l1() raises:
    if _skip_tied("test_zero2_equals_zero0_muon_w8_head_straddles_l1"):
        return
    _zero1_vs_zero0[8](1, 1, 16, True, 2)


def test_zero3_equals_zero0_adamw_w2() raises:
    if _skip_tied("test_zero3_equals_zero0_adamw_w2"):
        return
    _zero1_vs_zero0[2](2, 2, 16, False, 3)


def test_zero3_equals_zero0_adamw_w4() raises:
    if _skip_tied("test_zero3_equals_zero0_adamw_w4"):
        return
    _zero1_vs_zero0[4](2, 2, 16, False, 3)


def test_zero3_equals_zero0_adamw_w8() raises:
    if _skip_tied("test_zero3_equals_zero0_adamw_w8"):
        return
    _zero1_vs_zero0[8](2, 2, 16, False, 3)


def test_zero3_equals_zero0_adamw_w8_head_straddles_l1() raises:
    if _skip_tied("test_zero3_equals_zero0_adamw_w8_head_straddles_l1"):
        return
    _zero1_vs_zero0[8](1, 1, 16, False, 3)


def test_zero3_equals_zero0_muon_w2() raises:
    if _skip_tied("test_zero3_equals_zero0_muon_w2"):
        return
    _zero1_vs_zero0[2](2, 2, 16, True, 3)


def test_zero3_equals_zero0_muon_w4() raises:
    if _skip_tied("test_zero3_equals_zero0_muon_w4"):
        return
    _zero1_vs_zero0[4](2, 2, 16, True, 3)


def test_zero3_equals_zero0_muon_w8() raises:
    if _skip_tied("test_zero3_equals_zero0_muon_w8"):
        return
    _zero1_vs_zero0[8](2, 2, 16, True, 3)


def test_zero3_equals_zero0_muon_w8_head_straddles_l1() raises:
    if _skip_tied("test_zero3_equals_zero0_muon_w8_head_straddles_l1"):
        return
    _zero1_vs_zero0[8](1, 1, 16, True, 3)


def _untie_transition[
    W: Int
](L: Int, NH: Int, C: Int, stage: Int, muon: Bool) raises:
    """-untie_at transition at ZeRO-`stage`: 1 tied (idle-head) step, then
    `untie_head()` (head := wte, a collective under ZeRO-3 streaming), then 2
    trained steps; same batch, accum 2. Must equal the same schedule at ZeRO-0
    on all 17 tensors, bit-exactly (same-batch + W<=4 is the regime where the
    tied phase itself is exact, see the idle-head tests)."""
    var path = (
        String("gpt2_zu_untie")
        + String(stage)
        + ("m" if muon else "a")
        + "_w"
        + String(W)
        + "_l"
        + String(L)
        + ".bin"
    )
    var n = _write_untied_checkpoint(path, L, NH, C)
    var sz = _sizes(L, C)
    var ref_p = heap_alloc[Float32](W * n)
    var got = heap_alloc[Float32](W * n)
    var init = heap_alloc[Float32](n)
    var scratch = heap_alloc[Float32](n)
    try:
        _train[W](
            path,
            n,
            0,
            STEPS,
            ref_p,
            init,
            muon=muon,
            accum=2,
            same_batch=True,
            untie_after=1,
        )
        _train[W](
            path,
            n,
            stage,
            STEPS,
            got,
            scratch,
            muon=muon,
            accum=2,
            same_batch=True,
            untie_after=1,
        )
    except e:
        _remove(path)
        raise e^
    _remove(path)
    var d = _first_diff(ref_p, got, W, n)
    if d >= 0:
        raise Error(
            "untie transition: ZeRO-"
            + String(stage)
            + " != ZeRO-0 at W="
            + String(W)
            + " rank "
            + String(d // n)
            + " "
            + _tensor_of(d % n, sz)
            + ": "
            + String(ref_p[unsafe_offset=d])
            + " vs "
            + String(got[unsafe_offset=d])
        )
    # The head really was replaced by wte at the transition and then trained:
    # it differs from its planted init everywhere.
    var head_off = n - sz[16]
    for i in range(sz[16]):
        if got[unsafe_offset=head_off + i] == init[unsafe_offset=head_off + i]:
            raise Error("head still holds its planted init: " + String(i))
    ref_p.unsafe_free()
    got.unsafe_free()
    init.unsafe_free()
    scratch.unsafe_free()


def test_untie_transition_zero1_adamw_w2() raises:
    if _skip_tied("test_untie_transition_zero1_adamw_w2"):
        return
    _untie_transition[2](2, 2, 16, 1, False)


def test_untie_transition_zero1_muon_w2() raises:
    if _skip_tied("test_untie_transition_zero1_muon_w2"):
        return
    _untie_transition[2](2, 2, 16, 1, True)


def test_untie_transition_zero2_adamw_w2() raises:
    if _skip_tied("test_untie_transition_zero2_adamw_w2"):
        return
    _untie_transition[2](2, 2, 16, 2, False)


def test_untie_transition_zero2_adamw_w4() raises:
    if _skip_tied("test_untie_transition_zero2_adamw_w4"):
        return
    _untie_transition[4](2, 2, 16, 2, False)


def test_untie_transition_zero2_muon_w2() raises:
    if _skip_tied("test_untie_transition_zero2_muon_w2"):
        return
    _untie_transition[2](2, 2, 16, 2, True)


def test_untie_transition_zero2_muon_w4() raises:
    if _skip_tied("test_untie_transition_zero2_muon_w4"):
        return
    _untie_transition[4](2, 2, 16, 2, True)


def test_untie_transition_zero3_adamw_w2() raises:
    if _skip_tied("test_untie_transition_zero3_adamw_w2"):
        return
    _untie_transition[2](2, 2, 16, 3, False)


def test_untie_transition_zero3_adamw_w4() raises:
    if _skip_tied("test_untie_transition_zero3_adamw_w4"):
        return
    _untie_transition[4](2, 2, 16, 3, False)


def test_untie_transition_zero3_muon_w2() raises:
    if _skip_tied("test_untie_transition_zero3_muon_w2"):
        return
    _untie_transition[2](2, 2, 16, 3, True)


def test_untie_transition_zero3_muon_w4() raises:
    if _skip_tied("test_untie_transition_zero3_muon_w4"):
        return
    _untie_transition[4](2, 2, 16, 3, True)


def _idle_head_same_batch[W: Int](L: Int, NH: Int, C: Int, stage: Int) raises:
    """Idle head at ZeRO-2/3 == idle head at ZeRO-0, same batch, accum 2, 3
    steps, Muon: every element bit-exact, and the idle head still holds its
    planted values (nothing reduced, updated or read it)."""
    var path = (
        String("gpt2_zu_idle")
        + String(stage)
        + "_w"
        + String(W)
        + "_l"
        + String(L)
        + ".bin"
    )
    var n = _write_untied_checkpoint(path, L, NH, C)
    var sz = _sizes(L, C)
    var ref_p = heap_alloc[Float32](W * n)
    var got = heap_alloc[Float32](W * n)
    var init = heap_alloc[Float32](n)
    var scratch = heap_alloc[Float32](n)
    try:
        _train[W](
            path,
            n,
            0,
            STEPS,
            ref_p,
            init,
            accum=2,
            same_batch=True,
            idle_head=True,
        )
        _train[W](
            path,
            n,
            stage,
            STEPS,
            got,
            scratch,
            accum=2,
            same_batch=True,
            idle_head=True,
        )
    except e:
        _remove(path)
        raise e^
    _remove(path)
    var d = _first_diff(ref_p, got, W, n)
    if d >= 0:
        raise Error(
            "idle head: ZeRO-"
            + String(stage)
            + " != ZeRO-0 at W="
            + String(W)
            + " rank "
            + String(d // n)
            + " "
            + _tensor_of(d % n, sz)
            + ": "
            + String(ref_p[unsafe_offset=d])
            + " vs "
            + String(got[unsafe_offset=d])
        )
    var head_off = n - sz[16]
    for r in range(W):
        for i in range(sz[16]):
            if (
                got[unsafe_offset=r * n + head_off + i]
                != init[unsafe_offset=head_off + i]
            ):
                raise Error("idle head moved at rank " + String(r))
        # ...while the body did train.
        var moved = 0
        for i in range(sz[1], head_off):
            if got[unsafe_offset=r * n + i] != init[unsafe_offset=i]:
                moved += 1
        assert_true(moved > (head_off - sz[1]) // 2, "body did not train")
    ref_p.unsafe_free()
    got.unsafe_free()
    init.unsafe_free()
    scratch.unsafe_free()


def test_idle_head_zero2_equals_zero0_w2() raises:
    if _skip_tied("test_idle_head_zero2_equals_zero0_w2"):
        return
    _idle_head_same_batch[2](2, 2, 16, 2)


def test_idle_head_zero2_equals_zero0_w4() raises:
    if _skip_tied("test_idle_head_zero2_equals_zero0_w4"):
        return
    _idle_head_same_batch[4](2, 2, 16, 2)


def test_idle_head_zero3_equals_zero0_w2() raises:
    if _skip_tied("test_idle_head_zero3_equals_zero0_w2"):
        return
    _idle_head_same_batch[2](2, 2, 16, 3)


def test_idle_head_zero3_equals_zero0_w4() raises:
    if _skip_tied("test_idle_head_zero3_equals_zero0_w4"):
        return
    _idle_head_same_batch[4](2, 2, 16, 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
