"""GPU regression test: attention_fwd with a persistent KVCache across calls at
different sequence lengths (docs/ai/attention_stale_upper_triangle_generation_bug.md).

The GEMM attention's causal softmax writes only the lower triangle of the
[B·nh, T, T] probability plane and relies on the upper triangle staying zero
from a one-time memset. GPT2 keeps that plane alive across forwards (the
KVCache scratch, or the per-layer `acts.att_probs` store-P slices), so a call
at a smaller T reinterprets it with a new row stride and A·V used to read the
previous call's lower-triangle values as live probabilities. Training (one
fixed T) never saw it; token-by-token generation (T = 1, 2, 3, ...) and every
training step after a mid-run sample did.

Every existing attention test builds a fresh cache per call, so none could
catch it. This test replays the generation pattern -- big T, then smaller and
unaligned T, then big T again -- through ONE cache, in both the store-P mode
GPT2 uses and the shared-scratch mode, and checks each output against an
independent fp64 causal-attention reference on the host. Pre-fix, the T=9
call's error is O(0.1-1) against a tolerance of 1e-2 (fp32) / 3e-2 (bf16).

Run: `make test-mojo` (GPU required; skips cleanly without an NVIDIA GPU).
"""

from std.math import exp, sin, sqrt
from std.testing import assert_true, TestSuite

from llmm.vendor import has_nvidia_gpu_accelerator
from llmm.memory import MutKernelPtr, ImmutKernelPtr
from llmm.attention import attention_fwd, KVCache, KVCachePtr
from _gpu_test_common import shared_gpu_ctx

comptime NH = 12
comptime HD = 64
comptime T_MAX = 64
comptime NUM_LAYERS = 2


def _val(which: Int, h: Int, t: Int, d: Int) -> Float64:
    """Deterministic input in [-1, 1], independent of the call's T so every
    call sees the same causal prefix."""
    return sin(
        Float64(which * 7919 + h * 104729 + t * 131 + d * 17) * 0.0137 + 0.3
    )


def _reference(T: Int, h: Int, i: Int, d: Int) -> Float64:
    """fp64 causal softmax(QKᵀ/sqrt(hd))·V for row i of head h."""
    var scale = 1.0 / sqrt(Float64(HD))
    var m = Float64(-1e300)
    var scores = List[Float64]()
    for j in range(i + 1):
        var s: Float64 = 0.0
        for c in range(HD):
            s += _val(0, h, i, c) * _val(1, h, j, c)
        s *= scale
        scores.append(s)
        if s > m:
            m = s
    var den: Float64 = 0.0
    var num: Float64 = 0.0
    for j in range(i + 1):
        var p = exp(scores[j] - m)
        den += p
        num += p * _val(2, h, j, d)
    return num / den


def _run_sequence[dtype: DType](store_p: Bool, atol: Float64) raises:
    var ctx = shared_gpu_ctx()
    var plane_max = NH * T_MAX * T_MAX
    var qkv_elems = NH * T_MAX * HD
    var q_dev = ctx.enqueue_create_buffer[dtype](qkv_elems)
    var k_dev = ctx.enqueue_create_buffer[dtype](qkv_elems)
    var v_dev = ctx.enqueue_create_buffer[dtype](qkv_elems)
    var o_dev = ctx.enqueue_create_buffer[dtype](qkv_elems)
    var lse_dev = ctx.enqueue_create_buffer[DType.float32](NH * T_MAX)
    # Stand-in for GPT2's acts.att_probs: zeroed once at allocation, exactly
    # like train_gpt2.mojo's acts_buf memset.
    var probs_dev = ctx.enqueue_create_buffer[dtype](NUM_LAYERS * plane_max)
    ctx.enqueue_memset(probs_dev, Scalar[dtype](0))
    var host_q = ctx.enqueue_create_host_buffer[dtype](qkv_elems)
    var host_k = ctx.enqueue_create_host_buffer[dtype](qkv_elems)
    var host_v = ctx.enqueue_create_host_buffer[dtype](qkv_elems)
    var host_o = ctx.enqueue_create_host_buffer[dtype](qkv_elems)
    ctx.synchronize()

    var cache = KVCache()
    # Generation-shaped sequence: full T, then smaller (aligned and unaligned)
    # T, then full T again through the same cache.
    var seq_lens: List[Int] = [T_MAX, 9, 8, 33, 1, 17, T_MAX]
    var first_full = List[Float32]()
    for call in range(len(seq_lens)):
        var T = seq_lens[call]
        for h in range(NH):
            for t in range(T):
                for d in range(HD):
                    var o = (h * T + t) * HD + d
                    host_q.unsafe_ptr()[unsafe_offset=o] = Scalar[
                        DType.float64
                    ](_val(0, h, t, d)).cast[dtype]()
                    host_k.unsafe_ptr()[unsafe_offset=o] = Scalar[
                        DType.float64
                    ](_val(1, h, t, d)).cast[dtype]()
                    host_v.unsafe_ptr()[unsafe_offset=o] = Scalar[
                        DType.float64
                    ](_val(2, h, t, d)).cast[dtype]()
        q_dev.enqueue_copy_from(host_q)
        k_dev.enqueue_copy_from(host_k)
        v_dev.enqueue_copy_from(host_v)
        ctx.synchronize()

        var layer = call % NUM_LAYERS
        if store_p:
            cache.att_probs_addr = Int(probs_dev.unsafe_ptr())
            cache.att_probs_layer = layer
            cache.att_probs_stride = NH * T * T
        attention_fwd[dtype, "gpu", use_soft_exp=True](
            rebind[MutKernelPtr[dtype]](
                o_dev.unsafe_ptr().as_unsafe_any_origin()
            ),
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
            cache=rebind[KVCachePtr](Pointer(to=cache)),
        )
        ctx.synchronize()
        o_dev.enqueue_copy_to(host_o)
        ctx.synchronize()

        var max_err: Float64 = 0.0
        for h in range(NH):
            for t in range(T):
                for d in range(HD):
                    var o = (h * T + t) * HD + d
                    var got = Float64(
                        host_o.unsafe_ptr()[unsafe_offset=o].cast[
                            DType.float64
                        ]()
                    )
                    var err = abs(got - _reference(T, h, t, d))
                    if err > max_err:
                        max_err = err
        print(
            "  store_p=",
            store_p,
            "dtype=",
            dtype,
            "call",
            call,
            "T=",
            T,
            "max_err=",
            max_err,
        )
        assert_true(
            max_err <= atol,
            String("attention output drifted after a shape change: T=")
            + String(T)
            + " call "
            + String(call)
            + " max_err "
            + String(max_err),
        )

        # The final full-T call must be bit-identical to the first one: the
        # plane has to be back in exactly its fresh state.
        if call == 0 or call == len(seq_lens) - 1:
            for i in range(NH * T * HD):
                var x = host_o.unsafe_ptr()[unsafe_offset=i].cast[
                    DType.float32
                ]()
                if call == 0:
                    first_full.append(x)
                else:
                    assert_true(
                        x == first_full[i],
                        (
                            "full-T attention not bit-identical after smaller-T"
                            " calls through the same cache"
                        ),
                    )


def test_store_p_fp32() raises:
    if not has_nvidia_gpu_accelerator():
        return
    _run_sequence[DType.float32](True, 1e-2)


def test_scratch_fp32() raises:
    if not has_nvidia_gpu_accelerator():
        return
    _run_sequence[DType.float32](False, 1e-2)


def test_store_p_bf16() raises:
    if not has_nvidia_gpu_accelerator():
        return
    _run_sequence[DType.bfloat16](True, 3e-2)


def test_scratch_bf16() raises:
    if not has_nvidia_gpu_accelerator():
        return
    _run_sequence[DType.bfloat16](False, 3e-2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
