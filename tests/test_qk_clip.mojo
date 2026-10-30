"""GPU tests for QK-clip (llmm/qk_clip.mojo + the per-head logit max in the
causal softmax kernel of llmm/attention.mojo).

Run: `make test-mojo` (GPU required; skips cleanly without one).
"""

from std.math import exp, sin, sqrt
from std.testing import assert_equal, assert_true, TestSuite

from llmm.vendor import (
    has_nvidia_gpu_accelerator,
    has_apple_gpu_accelerator,
)
from llmm.memory import MutKernelPtr, ImmutKernelPtr
from llmm.attention import attention_fwd
from llmm.qk_clip import qk_clip_apply
from _gpu_test_common import shared_gpu_ctx


def has_gpu_accelerator() -> Bool:
    return has_nvidia_gpu_accelerator() or has_apple_gpu_accelerator()


def test_qk_clip_apply_scales_only_hot_heads() raises:
    if not has_gpu_accelerator():
        print("SKIP test_qk_clip_apply_scales_only_hot_heads: needs a GPU")
        return
    comptime L = 2
    comptime NH = 4
    comptime C = 32
    comptime HD = C // NH
    comptime W_OFF = 5
    comptime B_OFF = W_OFF + L * 3 * C * C + 7
    comptime N = B_OFF + L * 3 * C + 3
    var ctx = shared_gpu_ctx()
    var p_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var m_dev = ctx.enqueue_create_buffer[DType.float32](N)  # unused stand-in
    var hm_dev = ctx.enqueue_create_buffer[DType.float32](L * NH)
    var p_host = ctx.enqueue_create_host_buffer[DType.float32](N)
    var hm_host = ctx.enqueue_create_host_buffer[DType.float32](L * NH)
    ctx.synchronize()
    for i in range(N):
        p_host[i] = 1.0
    for l in range(L):
        for r in range(3 * C):
            p_host[B_OFF + l * 3 * C + r] = 2.0
    for i in range(L * NH):
        hm_host[i] = 50.0
    var hot_layer = 1
    var hot_head = 2
    hm_host[hot_layer * NH + hot_head] = 200.0
    p_dev.enqueue_copy_from(p_host)
    hm_dev.enqueue_copy_from(hm_host)
    ctx.synchronize()

    qk_clip_apply[DType.float32, "gpu"](
        rebind[MutKernelPtr[DType.float32]](
            p_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.float32]](
            m_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        False,
        W_OFF,
        B_OFF,
        rebind[ImmutKernelPtr[DType.float32]](
            hm_dev.unsafe_ptr().as_imm().as_unsafe_any_origin()
        ),
        L,
        NH,
        C,
        Float32(100.0),
        ctx,
    )
    ctx.synchronize()
    p_dev.enqueue_copy_to(p_host)
    ctx.synchronize()

    var f = sqrt(Float32(100.0) / Float32(200.0))
    var n_scaled = 0
    for l in range(L):
        for r in range(3 * C):
            var hot = False
            if l == hot_layer and r < 2 * C:
                hot = ((r % C) // HD) == hot_head
            var wv = f if hot else Float32(1.0)
            var bv = Float32(2.0) * f if hot else Float32(2.0)
            for c in range(C):
                var got = p_host[W_OFF + (l * 3 * C + r) * C + c]
                assert_true(
                    got == wv,
                    "weight mismatch at l/r/c "
                    + String(l)
                    + "/"
                    + String(r)
                    + "/"
                    + String(c),
                )
            assert_true(p_host[B_OFF + l * 3 * C + r] == bv, "bias mismatch")
            if hot:
                n_scaled += 1
    # Q rows + K rows of one head.
    assert_equal(n_scaled, 2 * HD)
    # Untouched padding around the regions.
    for i in range(W_OFF):
        assert_true(p_host[i] == 1.0)
    assert_true(p_host[N - 1] == 1.0)


def _qv(which: Int, h: Int, t: Int, d: Int) -> Float32:
    if which == 0 and h == 1 and t == 10:
        return 3.0  # planted large logit against key (h=1, t=7)
    if which == 1 and h == 1 and t == 7:
        return 4.0
    return Float32(sin(Float64(which * 53 + h * 31 + t * 7 + d) * 0.37)) * 0.5


def test_softmax_head_max() raises:
    if not has_gpu_accelerator():
        print("SKIP test_softmax_head_max: needs a GPU")
        return
    comptime dtype = DType.bfloat16
    comptime NH = 2
    comptime T = 16
    comptime HD = 8
    comptime n = NH * T * HD
    var ctx = shared_gpu_ctx()
    var q_dev = ctx.enqueue_create_buffer[dtype](n)
    var k_dev = ctx.enqueue_create_buffer[dtype](n)
    var v_dev = ctx.enqueue_create_buffer[dtype](n)
    var o_dev = ctx.enqueue_create_buffer[dtype](n)
    var lse_dev = ctx.enqueue_create_buffer[DType.float32](NH * T)
    var hm_dev = ctx.enqueue_create_buffer[DType.float32](NH)
    var hq = ctx.enqueue_create_host_buffer[dtype](n)
    var hk = ctx.enqueue_create_host_buffer[dtype](n)
    var hv = ctx.enqueue_create_host_buffer[dtype](n)
    var hm = ctx.enqueue_create_host_buffer[DType.float32](NH)
    ctx.synchronize()
    for h in range(NH):
        for t in range(T):
            for d in range(HD):
                var o = (h * T + t) * HD + d
                hq[o] = _qv(0, h, t, d).cast[dtype]()
                hk[o] = _qv(1, h, t, d).cast[dtype]()
                hv[o] = _qv(2, h, t, d).cast[dtype]()
    q_dev.enqueue_copy_from(hq)
    k_dev.enqueue_copy_from(hk)
    v_dev.enqueue_copy_from(hv)
    hm_dev.enqueue_fill(Scalar[DType.float32](0))
    ctx.synchronize()

    attention_fwd[dtype, "gpu", use_soft_exp=True](
        rebind[MutKernelPtr[dtype]](o_dev.unsafe_ptr().as_unsafe_any_origin()),
        rebind[ImmutKernelPtr[dtype]](
            q_dev.unsafe_ptr().as_imm().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[dtype]](
            k_dev.unsafe_ptr().as_imm().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[dtype]](
            v_dev.unsafe_ptr().as_imm().as_unsafe_any_origin()
        ),
        rebind[MutKernelPtr[DType.float32]](
            lse_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        Int64(1),
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
    ctx.synchronize()
    hm_dev.enqueue_copy_to(hm)
    ctx.synchronize()

    for h in range(NH):
        var ref_max = Float64(0.0)  # buffer is zero-initialised: clamps at 0
        for i in range(T):
            for j in range(i + 1):
                var s = Float64(0.0)
                for d in range(HD):
                    var o_i = (h * T + i) * HD + d
                    var o_j = (h * T + j) * HD + d
                    s += (
                        hq[o_i].cast[DType.float64]()
                        * hk[o_j].cast[DType.float64]()
                    )
                s /= sqrt(Float64(HD))
                if s > ref_max:
                    ref_max = s
        var got = Float64(hm[h])
        print("  head", h, "got", got, "ref", ref_max)
        assert_true(abs(got - ref_max) <= 5e-2 * abs(ref_max) + 1e-6)
    assert_true(Float64(hm[1]) > 20.0)  # the planted logit dominates head 1


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
