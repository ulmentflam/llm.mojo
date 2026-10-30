"""GPU tests for QK-clip (llmm/qk_clip.mojo, the per-head logit max in the
causal softmax kernel of llmm/attention.mojo, and the GPT2 wiring in
train_gpt2.mojo: enable_qk_clip / backward fold / qk_clip_step / qk_max).

Run: `make test-mojo` (GPU required; skips cleanly without one).
"""

from std.math import exp, sin, sqrt
from std.python import Python
from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from max.algorithm import sync_parallelize
from max.gpu.host import DeviceContext

from llmm.vendor import (
    has_nvidia_gpu_accelerator,
    has_apple_gpu_accelerator,
)
from llmm.memory import MutKernelPtr, ImmutKernelPtr, MutMemPtr, heap_alloc
from llmm.attention import attention_fwd
from llmm.qk_clip import qk_clip_apply
from llmm.zero import CpuCoordinator
from _gpu_test_common import shared_gpu_ctx
from train_gpt2 import GPT2, GPT2_DTYPE, UNTIED_HEAD


def has_gpu_accelerator() -> Bool:
    return has_nvidia_gpu_accelerator() or has_apple_gpu_accelerator()


def _apply_case[dtype: DType, has_master: Bool]() raises:
    """Runs qk_clip_apply on an [L,3C,C] weight + [L,3C] bias block embedded
    at odd offsets in a larger buffer, with three kinds of head: over tau
    (two different factors), exactly at tau (strict `>`: untouched) and
    under tau. Every element of the buffer is checked: scaled Q/K rows and
    their bias entries; V rows, other heads, other layers and the padding
    around/between the regions bit-identical."""
    comptime L = 2
    comptime NH = 4
    comptime C = 32
    comptime HD = C // NH
    comptime W_OFF = 5
    comptime B_OFF = W_OFF + L * 3 * C * C + 7
    comptime N = B_OFF + L * 3 * C + 3
    comptime TAU = Float32(100.0)
    var ctx = shared_gpu_ctx()
    var p_dev = ctx.enqueue_create_buffer[dtype](N)
    var m_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var hm_dev = ctx.enqueue_create_buffer[DType.float32](L * NH)
    var p_host = ctx.enqueue_create_host_buffer[dtype](N)
    var m_host = ctx.enqueue_create_host_buffer[DType.float32](N)
    var hm_host = ctx.enqueue_create_host_buffer[DType.float32](L * NH)
    var orig = List[Float32](capacity=N)
    ctx.synchronize()
    for i in range(N):
        # Distinct, bf16-exact values so a misindexed write cannot pass.
        var v = Float32(1.0) + Float32(i % 7) * 0.125
        orig.append(v)
        p_host[i] = v.cast[dtype]()
        m_host[i] = v
    for i in range(L * NH):
        hm_host[i] = 50.0
    hm_host[1 * NH + 2] = 200.0  # factor sqrt(0.5)
    hm_host[0 * NH + 3] = 400.0  # factor 0.5
    hm_host[1 * NH + 0] = TAU  # == tau: not clipped
    p_dev.enqueue_copy_from(p_host)
    m_dev.enqueue_copy_from(m_host)
    hm_dev.enqueue_copy_from(hm_host)
    ctx.synchronize()

    qk_clip_apply[dtype, "gpu"](
        rebind[MutKernelPtr[dtype]](p_dev.unsafe_ptr().as_unsafe_any_origin()),
        rebind[MutKernelPtr[DType.float32]](
            m_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        has_master,
        W_OFF,
        B_OFF,
        rebind[ImmutKernelPtr[DType.float32]](
            hm_dev.unsafe_ptr().as_imm().as_unsafe_any_origin()
        ),
        L,
        NH,
        C,
        TAU,
        ctx,
    )
    ctx.synchronize()
    p_dev.enqueue_copy_to(p_host)
    m_dev.enqueue_copy_to(m_host)
    ctx.synchronize()

    var expect_f = List[Float32](capacity=N)
    for _ in range(N):
        expect_f.append(1.0)
    var n_scaled = 0
    for l in range(L):
        for r in range(2 * C):  # Q and K rows only; V rows stay 1.0
            var smax = hm_host[l * NH + (r % C) // HD]
            if smax > TAU:
                var f = sqrt(TAU / smax)
                for c in range(C):
                    expect_f[W_OFF + (l * 3 * C + r) * C + c] = f
                expect_f[B_OFF + l * 3 * C + r] = f
                n_scaled += 1
    # Q + K rows of two hot heads.
    assert_equal(n_scaled, 2 * 2 * HD)

    for i in range(N):
        var f = expect_f[i]
        var m_got = m_host[i]
        var p_got = p_host[i]
        if f == 1.0:
            assert_true(
                p_got == orig[i].cast[dtype](), "touched param " + String(i)
            )
            assert_true(m_got == orig[i], "touched master " + String(i))
            continue
        var want = orig[i] * f
        comptime if has_master:
            # Master scaled in fp32, the param is that master rounded.
            assert_true(abs(m_got - want) <= 1e-6 * want, "master " + String(i))
            assert_true(p_got == m_got.cast[dtype](), "param != round(master)")
        else:
            assert_true(m_got == orig[i], "master touched without master")
            assert_true(
                abs(p_got.cast[DType.float32]() - want) <= 1e-6 * want,
                "param " + String(i),
            )


def test_qk_clip_apply_fp32_in_place() raises:
    if not has_gpu_accelerator():
        print("SKIP test_qk_clip_apply_fp32_in_place: needs a GPU")
        return
    _apply_case[DType.float32, False]()


def test_qk_clip_apply_bf16_master() raises:
    if not has_gpu_accelerator():
        print("SKIP test_qk_clip_apply_bf16_master: needs a GPU")
        return
    _apply_case[DType.bfloat16, True]()


def _qv(which: Int, b: Int, h: Int, t: Int, d: Int) -> Float32:
    # Planted logits (rows are (b, h, t)):
    #   b=0 h=1: q[t=10]=3 . k[t=7]=4 -> 96/sqrt(8) ~ 33.9 (head 1)
    #   b=1 h=2: q[t=5]=2.5 . k[t=3]=2.5 -> 50/sqrt(8) ~ 17.7 (head 2, batch 1)
    #   b=0 h=0: q[t=2]=5 . k[t=12]=5 is ABOVE the diagonal (key after query):
    #            ~70.7 if the causal mask were ignored; must not count.
    if b == 0 and h == 1:
        if which == 0 and t == 10:
            return 3.0
        if which == 1 and t == 7:
            return 4.0
    if b == 1 and h == 2:
        if which == 0 and t == 5:
            return 2.5
        if which == 1 and t == 3:
            return 2.5
    if b == 0 and h == 0:
        if which == 0 and t == 2:
            return 5.0
        if which == 1 and t == 12:
            return 5.0
    return (
        Float32(sin(Float64(which * 53 + b * 19 + h * 31 + t * 7 + d) * 0.37))
        * 0.5
    )


def _softmax_head_max_case[T: Int]() raises:
    comptime dtype = DType.bfloat16
    comptime B = 2
    comptime NH = 3
    comptime HD = 8
    comptime n = B * NH * T * HD
    var ctx = shared_gpu_ctx()
    var q_dev = ctx.enqueue_create_buffer[dtype](n)
    var k_dev = ctx.enqueue_create_buffer[dtype](n)
    var v_dev = ctx.enqueue_create_buffer[dtype](n)
    var o_dev = ctx.enqueue_create_buffer[dtype](n)
    var o2_dev = ctx.enqueue_create_buffer[dtype](n)
    var lse_dev = ctx.enqueue_create_buffer[DType.float32](B * NH * T)
    var lse2_dev = ctx.enqueue_create_buffer[DType.float32](B * NH * T)
    # [NH] slots plus B*NH sentinels: an index missing the `% NH` (e.g. row //
    # T, which is b*NH + h) writes past NH and is caught below.
    var hm_dev = ctx.enqueue_create_buffer[DType.float32](NH + B * NH)
    var hq = ctx.enqueue_create_host_buffer[dtype](n)
    var hk = ctx.enqueue_create_host_buffer[dtype](n)
    var hv = ctx.enqueue_create_host_buffer[dtype](n)
    var hm = ctx.enqueue_create_host_buffer[DType.float32](NH + B * NH)
    var ho = ctx.enqueue_create_host_buffer[dtype](n)
    var ho2 = ctx.enqueue_create_host_buffer[dtype](n)
    var hl = ctx.enqueue_create_host_buffer[DType.float32](B * NH * T)
    var hl2 = ctx.enqueue_create_host_buffer[DType.float32](B * NH * T)
    ctx.synchronize()
    for b in range(B):
        for h in range(NH):
            for t in range(T):
                for d in range(HD):
                    var o = ((b * NH + h) * T + t) * HD + d
                    hq[o] = _qv(0, b, h, t, d).cast[dtype]()
                    hk[o] = _qv(1, b, h, t, d).cast[dtype]()
                    hv[o] = _qv(2, b, h, t, d).cast[dtype]()
    q_dev.enqueue_copy_from(hq)
    k_dev.enqueue_copy_from(hk)
    v_dev.enqueue_copy_from(hv)
    hm_dev.enqueue_fill(Scalar[DType.float32](0))
    ctx.synchronize()

    var qp = rebind[ImmutKernelPtr[dtype]](
        q_dev.unsafe_ptr().as_imm().as_unsafe_any_origin()
    )
    var kp = rebind[ImmutKernelPtr[dtype]](
        k_dev.unsafe_ptr().as_imm().as_unsafe_any_origin()
    )
    var vp = rebind[ImmutKernelPtr[dtype]](
        v_dev.unsafe_ptr().as_imm().as_unsafe_any_origin()
    )
    attention_fwd[dtype, "gpu", use_soft_exp=True](
        rebind[MutKernelPtr[dtype]](o_dev.unsafe_ptr().as_unsafe_any_origin()),
        qp,
        kp,
        vp,
        rebind[MutKernelPtr[DType.float32]](
            lse_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        Int64(B),
        Int64(NH),
        Int64(T),
        Int64(HD),
        ctx,
        head_max=Optional[MutKernelPtr[DType.float32]](
            rebind[MutKernelPtr[DType.float32]](
                hm_dev.unsafe_ptr().as_unsafe_any_origin()
            )
        ),
    )
    # Same call without the hook: outputs and lse must be bit-identical.
    attention_fwd[dtype, "gpu", use_soft_exp=True](
        rebind[MutKernelPtr[dtype]](o2_dev.unsafe_ptr().as_unsafe_any_origin()),
        qp,
        kp,
        vp,
        rebind[MutKernelPtr[DType.float32]](
            lse2_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        Int64(B),
        Int64(NH),
        Int64(T),
        Int64(HD),
        ctx,
    )
    ctx.synchronize()
    hm_dev.enqueue_copy_to(hm)
    o_dev.enqueue_copy_to(ho)
    o2_dev.enqueue_copy_to(ho2)
    lse_dev.enqueue_copy_to(hl)
    lse2_dev.enqueue_copy_to(hl2)
    ctx.synchronize()
    for i in range(n):
        assert_true(ho[i] == ho2[i], "hook changed attention output")
    for i in range(B * NH * T):
        assert_true(hl[i] == hl2[i], "hook changed lse")
    for i in range(NH, NH + B * NH):
        assert_true(hm[i] == 0.0, "write past [NH] slice at " + String(i))

    for h in range(NH):
        var ref_max = Float64(0.0)  # buffer is zero-initialised: clamps at 0
        for b in range(B):
            for i in range(T):
                for j in range(i + 1):
                    var s = Float64(0.0)
                    for d in range(HD):
                        var o_i = ((b * NH + h) * T + i) * HD + d
                        var o_j = ((b * NH + h) * T + j) * HD + d
                        s += (
                            hq[o_i].cast[DType.float64]()
                            * hk[o_j].cast[DType.float64]()
                        )
                    s /= sqrt(Float64(HD))
                    if s > ref_max:
                        ref_max = s
        var got = Float64(hm[h])
        print("  T", T, "head", h, "got", got, "ref", ref_max)
        assert_true(abs(got - ref_max) <= 5e-2 * abs(ref_max) + 1e-6)
    assert_true(Float64(hm[1]) > 30.0, "planted (b0,h1) logit")
    assert_true(
        Float64(hm[2]) > 15.0 and Float64(hm[2]) < 25.0, "planted (b1,h2)"
    )
    assert_true(Float64(hm[0]) < 15.0, "above-diagonal logit counted")


def test_softmax_head_max_vector_kernel() raises:
    if not has_gpu_accelerator():
        print("SKIP test_softmax_head_max_vector_kernel: needs a GPU")
        return
    _softmax_head_max_case[16]()  # T multiple of the bf16 SIMD width


def test_softmax_head_max_scalar_kernel() raises:
    if not has_gpu_accelerator():
        print("SKIP test_softmax_head_max_scalar_kernel: needs a GPU")
        return
    _softmax_head_max_case[13]()  # T not width-aligned: width=1 kernel


# ---------------------------------------------------------------------------
# GPT2 wiring: tiny model (L=2, NH=2, C=16, V=64, maxT=16) whose layer-1
# head-1 Q/K rows (and biases) are planted large, so only that head's logit
# max exceeds tau.
# ---------------------------------------------------------------------------

comptime CKPT = "gpt2_tiny_qk_clip.bin"
comptime GL = 2
comptime GNH = 2
comptime GC = 16
comptime GV = 64
comptime GMAXT = 16
comptime GB = 2
comptime GT = 8
comptime GHD = GC // GNH
# Parameters order: wte, wpe, ln_1_gamma, ln_1_beta, qkv_weight, qkv_bias, ...
comptime LN1G_OFF = GV * GC + GMAXT * GC
comptime QKVW_OFF = LN1G_OFF + 2 * GL * GC
comptime QKVB_OFF = QKVW_OFF + GL * 3 * GC * GC
comptime NUM_PARAMS = (
    GV * GC
    + GMAXT * GC
    + 4 * GL * GC
    + GL * 3 * GC * GC
    + GL * 3 * GC
    + GL * GC * GC
    + GL * GC
    + GL * 4 * GC * GC
    + GL * 4 * GC
    + GL * GC * 4 * GC
    + GL * GC
    + 2 * GC
)
# Parameters in the live model: the tied .bin above plus, in an
# LLMM_UNTIED_HEAD build, the head block the loader copies from wte.
comptime MODEL_PARAMS = NUM_PARAMS + (GV * GC if UNTIED_HEAD else 0)
comptime HOT_L = 1
comptime HOT_H = 1


def _write_tiny_checkpoint() raises:
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    var header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 3  # version
    header[2] = GMAXT
    header[3] = GV
    header[4] = GL
    header[5] = GNH
    header[6] = GC
    header[7] = GV
    np.random.seed(11)
    var w = np.random.normal(0, 0.02, NUM_PARAMS).astype(np.float32)
    for i in range(GL * GC):
        w[LN1G_OFF + i] = 1.0  # unit LN gain: normalized attention inputs
    # Planted rows are i.i.d. (a structured pattern like sin(a*row + b*c) is
    # rank-2 and can make every logit of the head negative).
    var pw = np.random.normal(0, 0.6, 2 * GHD * GC).astype(np.float32)
    var k = 0
    for qk in range(2):
        for r in range(GHD):
            var row = qk * GC + HOT_H * GHD + r
            for c in range(GC):
                w[QKVW_OFF + (HOT_L * 3 * GC + row) * GC + c] = pw[k]
                k += 1
            w[QKVB_OFF + HOT_L * 3 * GC + row] = 0.3
    var f = builtins.open(CKPT, "wb")
    _ = f.write(header.tobytes())
    _ = f.write(w.tobytes())
    f.close()


def _remove_tiny_checkpoint() raises:
    var os = Python.import_module("os")
    try:
        _ = os.remove(CKPT)
    except:
        pass


def _fill_batch(
    x: MutMemPtr[DType.int32], y: MutMemPtr[DType.int32], seed: Int
):
    for i in range(GB * GT):
        x[unsafe_offset=i] = Int32((i * 13 + 5 + seed * 7) % GV)
        y[unsafe_offset=i] = Int32((i * 29 + 11 + seed * 3) % GV)


def test_gpt2_qk_clip_end_to_end() raises:
    """Enable -> forward/backward (max survives backward; eval forwards do not
    leak into it; micro-steps max-accumulate and micro-step 0 resets) ->
    qk_clip_step (only the hot head's Q/K rows + biases scale, by
    sqrt(tau/S)) -> the next forward's max for that head is tau."""
    comptime if has_accelerator():
        if not has_gpu_accelerator():
            print("SKIP test_gpt2_qk_clip_end_to_end: needs a GPU")
            return
        comptime TAU = Float32(1.0)
        _write_tiny_checkpoint()
        var xa = heap_alloc[Scalar[DType.int32]](GB * GT)
        var ya = heap_alloc[Scalar[DType.int32]](GB * GT)
        var xb = heap_alloc[Scalar[DType.int32]](GB * GT)
        var yb = heap_alloc[Scalar[DType.int32]](GB * GT)
        _fill_batch(xa, ya, 0)
        _fill_batch(xb, yb, 1)
        var ctx = shared_gpu_ctx()
        var model = GPT2["gpu", 1](CKPT, rank=0, zero_stage=0, ctx=ctx)
        _remove_tiny_checkpoint()
        assert_equal(model.num_parameters, MODEL_PARAMS)
        model.enable_qk_clip(TAU)
        comptime NHM = GL * GNH
        var hm = ctx.enqueue_create_host_buffer[DType.float32](NHM)
        var params = ctx.enqueue_create_host_buffer[GPT2_DTYPE](MODEL_PARAMS)
        ctx.synchronize()

        @__parameter
        def read_hm() raises -> List[Float32]:
            model.ctx.synchronize()
            model.head_max_buf.enqueue_copy_to(hm)
            ctx.synchronize()
            var out = List[Float32]()
            for i in range(NHM):
                out.append(hm[i])
            return out^

        # Single step on A.
        model.forward(xa, ya, GB, GT)
        model.backward()
        var hm_a = read_hm()
        # An eval-style forward on B (no backward) must not change the step max.
        model.forward(xb, yb, GB, GT)
        var hm_a2 = read_hm()
        for i in range(NHM):
            print("  head_max[", i, "] A =", hm_a[i])
            assert_true(hm_a2[i] == hm_a[i], "eval forward leaked into max")
        var hot = HOT_L * GNH + HOT_H
        assert_true(hm_a[hot] > 2.0 * TAU, "planted head below 2*tau")
        for i in range(NHM):
            if i != hot:
                assert_true(hm_a[i] < TAU, "cold head over tau")
        var best = Float32(0.0)
        for i in range(NHM):
            best = max(best, hm_a[i])
        assert_true(model.qk_max() == best, "qk_max != max over heads")

        # Single step on B, then a two-micro-step step A+B: elementwise max.
        model.forward(xb, yb, GB, GT)
        model.backward()
        var hm_b = read_hm()
        model.forward(xa, ya, GB, GT, 2)
        model.backward(2, 0, 0)
        model.forward(xb, yb, GB, GT, 2)
        model.backward(2, 1, 0)
        var hm_ab = read_hm()
        var differs = False
        for i in range(NHM):
            assert_true(hm_ab[i] == max(hm_a[i], hm_b[i]), "micro-step max")
            if hm_a[i] != hm_b[i]:
                differs = True
        assert_true(differs, "batches A and B give identical maxima")
        # Micro-step 0 of the next step resets (no carry-over from A+B).
        model.forward(xa, ya, GB, GT)
        model.backward()
        var hm_a3 = read_hm()
        for i in range(NHM):
            assert_true(hm_a3[i] == hm_a[i], "micro-step 0 did not reset")

        # Clip. lr=0 / wd=0 AdamW leaves params unchanged, so the only change
        # comes from qk_clip_step.
        model.update(UInt32(1), Float32(0.0))
        model.ctx.synchronize()
        model.params_buf.enqueue_copy_to(params)
        ctx.synchronize()
        var before = List[Float32](capacity=NUM_PARAMS)
        for i in range(NUM_PARAMS):
            before.append(params[i].cast[DType.float32]())
        model.qk_clip_step()
        model.ctx.synchronize()
        model.params_buf.enqueue_copy_to(params)
        ctx.synchronize()
        var f = sqrt(TAU / hm_a3[hot])
        var n_changed = 0
        for i in range(NUM_PARAMS):
            var want = before[i]
            var scaled = False
            if i >= QKVW_OFF and i < QKVB_OFF:
                var rel = i - QKVW_OFF
                var l = rel // (3 * GC * GC)
                var r = (rel // GC) % (3 * GC)
                scaled = l == HOT_L and r < 2 * GC and (r % GC) // GHD == HOT_H
            elif i >= QKVB_OFF and i < QKVB_OFF + GL * 3 * GC:
                var rel = i - QKVB_OFF
                var l = rel // (3 * GC)
                var r = rel % (3 * GC)
                scaled = l == HOT_L and r < 2 * GC and (r % GC) // GHD == HOT_H
            if scaled:
                want = before[i] * f
                n_changed += 1
                assert_true(
                    abs(params[i].cast[DType.float32]() - want)
                    <= 1e-6 * abs(want) + 1e-12,
                    "scaled param " + String(i),
                )
            else:
                assert_true(
                    params[i].cast[DType.float32]() == want,
                    "untouched param " + String(i),
                )
        assert_equal(n_changed, 2 * GHD * GC + 2 * GHD)

        # Post-clip: the hot head's logits all scale by tau/S, so its max is
        # now tau; cold heads unchanged.
        model.forward(xa, ya, GB, GT)
        model.backward()
        var hm_post = read_hm()
        print("  hot head max after clip", hm_post[hot], "tau", TAU)
        assert_true(abs(hm_post[hot] - TAU) <= 0.02 * TAU, "post-clip max")
        for i in range(NHM):
            if i != hot:
                assert_true(hm_post[i] == hm_a[i], "cold head changed")
        xa.unsafe_free()
        ya.unsafe_free()
        xb.unsafe_free()
        yb.unsafe_free()
    else:
        print("SKIP test_gpt2_qk_clip_end_to_end: needs a GPU")


# ---------------------------------------------------------------------------
# S5: QK-clip at WORLD_SIZE > 1 (threaded CPU ranks; CPU attention records no
# logits, so every rank PLANTS its own per-head maxima after backward).
# ---------------------------------------------------------------------------

comptime MT = 16
# MV = 192 puts a rank boundary of the flat optimizer shard (n / 2 at W=2,
# n / 4 * 2 at W=4, n = 9920) INSIDE the [L, 3C] qkv bias, so one rank's bias
# slice starts mid-bias (nonzero rel_start, beginning on layer 0's V rows):
# the sharded bias clip's layer/row arithmetic is exercised, not just a
# whole-bias slice at rel_start 0.
comptime MV = 192
comptime ML = 2
comptime MNH = 2
comptime MC = 16
comptime MTAU = Float32(5.0)
comptime NOT_OWNED = Float32(-1.0e30)


def _planted(rank: Int, i: Int) -> Float32:
    """Rank `rank`'s planted max for head slot i = layer * NH + head. Differs
    across ranks; the global max is [5, 20, 9, 4.5] at W=4 and [3, 20, 6, 4.5]
    at W=2, so (0,1) and (1,0) are hot only because of a PEER's value and the
    cold heads (0,0) (== tau at W=4: strict `>`) and (1,1) stay untouched."""
    if rank == 0:
        return SIMD[DType.float32, 4](3.0, 8.0, 2.0, 4.0)[i]
    if rank == 1:
        return SIMD[DType.float32, 4](1.0, 20.0, 6.0, 4.5)[i]
    if rank == 2:
        return SIMD[DType.float32, 4](5.0, 7.0, 9.0, 1.0)[i]
    return Float32(2.0)


def _write_mckpt(path: String) raises -> Int:
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    var L = ML
    var C = MC
    var n = (
        MV * C
        + MT * C
        + L * (C + C + 3 * C * C + 3 * C + C * C + C + C + C)
        + L * (4 * C * C + 4 * C + 4 * C * C + C)
        + 2 * C
    )
    var header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 3
    header[2] = MT
    header[3] = MV
    header[4] = L
    header[5] = MNH
    header[6] = C
    header[7] = MV
    np.random.seed(11)
    var weights = np.random.normal(0, 0.08, n).astype(np.float32)
    var f = builtins.open(path, "wb")
    _ = f.write(header.tobytes())
    _ = f.write(weights.tobytes())
    f.close()
    return n


def _mclip_train[
    W: Int
](
    path: String,
    n: Int,
    stage: Int,
    clip: Bool,
    result: MutMemPtr[DType.float32],
    qmax: MutMemPtr[DType.float32],
    owner: MutMemPtr[DType.float32],
) raises:
    """One Muon step on W threaded CPU ranks at `stage`. With `clip`, QK-clip
    is enabled and each rank plants `_planted(rank, .)` after backward; the
    trainer's `qk_max()` + `qk_clip_step()` sequence follows `update()`.
    result[r*n:(r+1)*n] = rank r's params (ZeRO-3: the full vector gathered by
    `_z3_gather_full`); qmax[r] = rank r's qk_max(). At ZeRO-3 owner[r*n:..]
    holds rank r's owner-space Muon matrices at their natural flat offsets
    (`NOT_OWNED` elsewhere)."""
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
                # S2 refuses ZeRO-2/3 while the head trains (S6 adds the head
                # bucketing/streaming). These ZeRO-2/3 checks exercise the
                # tied-phase layout: head block present in the 17-tensor
                # buffer but idle (neither reduced, updated nor read).
                model.head_tied = stage >= 2
                # The model really carries the 17-tensor layout (head block
                # after the 16 tied tensors), so the shard map differs from the
                # tied build's even with the head idle.
                assert_equal(model.num_parameters, n + model.param_sizes[0])
                assert_equal(model.param_sizes[16], model.param_sizes[0])
            model.enable_muon(Scalar[DType.float32](0.95))
            if clip:
                model.enable_qk_clip(MTAU, True)
            var x = heap_alloc[SIMD[DType.int32, 1]](MT)
            var y = heap_alloc[SIMD[DType.int32, 1]](MT)
            for i in range(MT):
                x[unsafe_offset=i] = Int32((rank * 7919 + i * 31 + 5) % MV)
                y[unsafe_offset=i] = Int32((rank * 17 + i * 5 + 1) % MV)
            model.forward(
                rebind[MutMemPtr[DType.int32]](x),
                rebind[MutMemPtr[DType.int32]](y),
                1,
                MT,
            )
            model.ctx.synchronize()
            model.zero_gradients()
            model.backward()
            model.ctx.synchronize()
            if clip:
                for i in range(ML * MNH):
                    model.head_max_memory[unsafe_offset=i] = _planted(rank, i)
            model.update(
                t=UInt32(1),
                learning_rate=Scalar[DType.float32](1e-3),
                beta1=Scalar[DType.float32](0.9),
                beta2=Scalar[DType.float32](0.999),
                eps=Scalar[DType.float32](1e-8),
                weight_decay=Scalar[DType.float32](0.01),
            )
            model.ctx.synchronize()
            qmax[unsafe_offset=rank] = model.qk_max()
            model.qk_clip_step()
            model.ctx.synchronize()
            if stage == 3:
                # Streaming: params_memory is only the flat shard. Gather the
                # full vector (collective) and export the owner-space copies.
                # Sized to the live model: an untied build gathers N16 + head.
                var full = model.ctx.enqueue_create_buffer[GPT2_DTYPE](
                    model.num_parameters
                )
                model.ctx.synchronize()
                model._z3_gather_full(
                    rebind[Pointer[Scalar[GPT2_DTYPE], MutAnyOrigin]](
                        full.unsafe_ptr().as_unsafe_any_origin()
                    )
                )
                model.ctx.synchronize()
                for i in range(n):
                    result[unsafe_offset=rank * n + i] = full.unsafe_ptr()[
                        unsafe_offset=i
                    ].cast[DType.float32]()
                    owner[unsafe_offset=rank * n + i] = NOT_OWNED
                for i in range(len(model.muon_slots)):
                    ref sl = model.muon_slots[i]
                    if sl.owner != rank:
                        continue
                    for j in range(sl.len):
                        owner[
                            unsafe_offset=rank * n + sl.flat_off + j
                        ] = model.muon_p_memory[
                            unsafe_offset=sl.local_off + j
                        ].cast[
                            DType.float32
                        ]()
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


def _mclip_case[W: Int](stage: Int) raises:
    """Clipped multi-rank step == unclipped step + W=1-style `qk_clip_apply`
    with the GLOBAL maxima, bit-exact, identically on every rank."""
    var path = String("gpt2_qkclip_w") + String(W) + "_z" + String(stage)
    path += ".bin"
    var n = _write_mckpt(path)
    var unclipped = heap_alloc[Float32](W * n)
    var got = heap_alloc[Float32](W * n)
    var qmax = heap_alloc[Float32](W)
    var qmax_ref = heap_alloc[Float32](W)
    var owner = heap_alloc[Float32](W * n)
    var owner_ref = heap_alloc[Float32](W * n)
    try:
        _mclip_train[W](path, n, stage, False, unclipped, qmax_ref, owner_ref)
        _mclip_train[W](path, n, stage, True, got, qmax, owner)
    except e:
        _remove(path)
        raise e^
    _remove(path)

    # Reference: the unclipped step at the SAME stage (reduce-scatter vs
    # allreduce differ in the last ulp elsewhere) clipped by the
    # single-process kernel.
    var gmax = heap_alloc[Float32](ML * MNH)
    for i in range(ML * MNH):
        var m = _planted(0, i)
        for r in range(1, W):
            m = max(m, _planted(r, i))
        gmax[unsafe_offset=i] = m
    var C = MC
    var w_off = MV * C + MT * C + 2 * ML * C
    var b_off = w_off + ML * 3 * C * C
    # Keep the straddle described at MV (shards are n / W rounded up to the
    # SIMD width; n / W is a multiple of 16 here so the rounding is a no-op).
    var boundary = (W // 2) * ((n + W - 1) // W)
    assert_true(
        b_off < boundary and boundary < b_off + ML * 3 * C,
        "qkv bias no longer straddles a shard boundary",
    )
    var want = heap_alloc[Float32](n)
    for i in range(n):
        want[unsafe_offset=i] = unclipped[unsafe_offset=i]
    var cctx = DeviceContext(api="cpu")
    qk_clip_apply[DType.float32, "cpu"](
        rebind[MutKernelPtr[DType.float32]](want.as_unsafe_any_origin()),
        rebind[MutKernelPtr[DType.float32]](want.as_unsafe_any_origin()),
        False,
        w_off,
        b_off,
        rebind[ImmutKernelPtr[DType.float32]](gmax.as_unsafe_any_origin()),
        ML,
        MNH,
        MC,
        MTAU,
        cctx,
    )
    var n_changed = 0
    var hd = MC // MNH
    for i in range(n):
        if want[unsafe_offset=i] != unclipped[unsafe_offset=i]:
            n_changed += 1
    # Hot heads (0,1) and (1,0): Q and K rows (hd rows x (C weights + bias)).
    assert_equal(n_changed, 2 * 2 * hd * (C + 1))
    for r in range(W):
        for i in range(n):
            var a = want[unsafe_offset=i]
            var b = got[unsafe_offset=r * n + i]
            if a != b:
                raise Error(
                    "multi-rank QK-clip != reference at W="
                    + String(W)
                    + " stage "
                    + String(stage)
                    + " rank "
                    + String(r)
                    + " idx "
                    + String(i)
                    + ": "
                    + String(a)
                    + " vs "
                    + String(b)
                )
    if stage == 3:
        # Owner-space copies carry the clip too (the next forward gathers the
        # Muon matrices from there): every owned element equals the reference
        # (bit-exact), each Muon element is owned exactly once, and exactly
        # the hot heads' Q and K matrix rows differ from the unclipped step.
        var owned = 0
        var owned_changed = 0
        for r in range(W):
            for i in range(n):
                var o = owner[unsafe_offset=r * n + i]
                if o == NOT_OWNED:
                    continue
                owned += 1
                if o != want[unsafe_offset=i]:
                    raise Error(
                        "owner-space != reference at rank "
                        + String(r)
                        + " idx "
                        + String(i)
                    )
                if o != unclipped[unsafe_offset=i]:
                    owned_changed += 1
        assert_equal(owned, ML * 12 * C * C)
        assert_equal(owned_changed, 2 * 2 * hd * C)
    # After the bit-exact params check, so a local-max collective is reported
    # by the params mismatch first; qk_max() must also be the GLOBAL max.
    for r in range(W):
        assert_equal(qmax[unsafe_offset=r], Float32(20.0))
    want.unsafe_free()
    gmax.unsafe_free()
    unclipped.unsafe_free()
    got.unsafe_free()
    qmax.unsafe_free()
    qmax_ref.unsafe_free()
    owner.unsafe_free()
    owner_ref.unsafe_free()


def test_qk_clip_multirank_zero1_w2() raises:
    _mclip_case[2](1)


def test_qk_clip_multirank_zero1_w4() raises:
    _mclip_case[4](1)


def test_qk_clip_multirank_zero0_w2() raises:
    _mclip_case[2](0)


def test_qk_clip_multirank_zero2_w2() raises:
    _mclip_case[2](2)


def test_qk_clip_multirank_zero2_w4() raises:
    _mclip_case[4](2)


def test_qk_clip_multirank_zero3_w2() raises:
    _mclip_case[2](3)


def test_qk_clip_multirank_zero3_w4() raises:
    _mclip_case[4](3)


def _remove(path: String) raises:
    var os = Python.import_module("os")
    try:
        _ = os.remove(path)
    except:
        pass


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
