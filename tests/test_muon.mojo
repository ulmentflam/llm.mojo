"""GPU unit tests for llmm/muon.mojo (Muon: Nesterov momentum + bf16
Newton-Schulz orthogonalization + Moonlight RMS scale + decoupled decay).

All tests need an NVIDIA GPU (cuBLASLt batched GEMMs) and share the one
process-wide DeviceContext from tests/_gpu_test_common.mojo.
"""

from std.math import sqrt
from std.testing import assert_true, assert_almost_equal, TestSuite

from llmm.memory import MutKernelPtr, ImmutKernelPtr
from llmm.muon import MuonConfig, MuonGroup, muon_update, muon_ns_host
from llmm.vendor import has_nvidia_gpu_accelerator
from _gpu_test_common import shared_gpu_ctx


def randn(n: Int, seed: UInt64) -> List[Float32]:
    """Approximately normal (sum of 12 uniforms) from an LCG; deterministic."""
    var out = List[Float32](length=n, fill=Float32(0))
    var s = seed * 6364136223846793005 + 1442695040888963407
    for i in range(n):
        var acc = Float32(0)
        for _ in range(12):
            s = s * 6364136223846793005 + 1442695040888963407
            acc += Float32(Int((s >> 40) & 0xFFFFFF)) / Float32(16777216.0)
        out[i] = acc - 6.0
    return out^


def run_muon(
    groups: List[MuonGroup],
    mut params: List[Float32],
    grads: List[Float32],
    mut m: List[Float32],
    lr: Float32,
    mu: Float32,
    wd: Float32,
    grad_scale: Float32,
    t: UInt32,
) raises:
    """One GPU muon_update over fp32 host vectors (updated in place)."""
    var ctx = shared_gpu_ctx()
    var n = len(params)
    var hp = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hg = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hm = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        hp.unsafe_ptr()[unsafe_offset=i] = params[i]
        hg.unsafe_ptr()[unsafe_offset=i] = grads[i]
        hm.unsafe_ptr()[unsafe_offset=i] = m[i]
    var dp = ctx.enqueue_create_buffer[DType.float32](n)
    var dg = ctx.enqueue_create_buffer[DType.float32](n)
    var dm = ctx.enqueue_create_buffer[DType.float32](n)
    dp.enqueue_copy_from(hp)
    dg.enqueue_copy_from(hg)
    dm.enqueue_copy_from(hm)
    ctx.synchronize()
    var config = MuonConfig(
        learning_rate=lr,
        momentum=mu,
        weight_decay=wd,
        grad_scale=grad_scale,
        ns_steps=5,
        nesterov=True,
    )
    var pp = rebind[MutKernelPtr[DType.float32]](
        dp.unsafe_ptr().as_unsafe_any_origin()
    )
    muon_update[DType.float32, "gpu"](
        groups,
        pp,
        rebind[ImmutKernelPtr[DType.float32]](
            dg.unsafe_ptr().as_imm().as_unsafe_any_origin()
        ),
        pp,  # master placeholder (has_master=False)
        False,
        rebind[MutKernelPtr[DType.float32]](
            dm.unsafe_ptr().as_unsafe_any_origin()
        ),
        t,
        config,
        ctx,
    )
    ctx.synchronize()
    dp.enqueue_copy_to(hp)
    dm.enqueue_copy_to(hm)
    ctx.synchronize()
    for i in range(n):
        params[i] = hp.unsafe_ptr()[unsafe_offset=i]
        m[i] = hm.unsafe_ptr()[unsafe_offset=i]


def check_orthogonal(rows: Int, cols: Int) raises:
    var params = List[Float32](length=rows * cols, fill=Float32(0))
    var m = List[Float32](length=rows * cols, fill=Float32(0))
    var grads = randn(rows * cols, 1)
    var groups = List[MuonGroup]()
    groups.append(MuonGroup(0, rows, cols, 1, rows * cols))
    run_muon(groups, params, grads, m, 1.0, 0.0, 0.0, 1.0, 1)
    var scale = Float32(0.2) * sqrt(Float32(max(rows, cols)))
    var R = min(rows, cols)
    var tall = rows > cols
    var K = max(rows, cols)
    var diag_err = Float32(0)
    var off = Float32(0)
    for i in range(R):
        for j in range(R):
            var s = Float32(0)
            for k in range(K):
                # Gram on the min-sided dimension: O O^T (wide) / O^T O (tall).
                if tall:
                    s += params[k * cols + i] * params[k * cols + j]
                else:
                    s += params[i * cols + k] * params[j * cols + k]
            s /= scale * scale
            if i == j:
                diag_err += abs(s - 1.0)
            else:
                off += abs(s)
    diag_err /= Float32(R)
    off /= Float32(R * R - R)
    print(
        "ns ortho ",
        rows,
        "x",
        cols,
        ": mean|diag-1|=",
        diag_err,
        " mean|offdiag|=",
        off,
    )
    assert_true(diag_err < 0.25)
    assert_true(off < 0.05)


def test_ns_orthogonalizes_rect() raises:
    if not has_nvidia_gpu_accelerator():
        print("SKIP test_ns_orthogonalizes_rect: needs NVIDIA GPU")
        return
    check_orthogonal(192, 256)
    check_orthogonal(256, 192)


def check_group_vs_host(
    rows: Int, cols: Int, count: Int, stride: Int, offset: Int
) raises:
    var total = offset + (count - 1) * stride + rows * cols + 64
    var params = List[Float32](length=total, fill=Float32(0))
    var m = List[Float32](length=total, fill=Float32(0))
    var grads = randn(total, UInt64(rows * 1000 + cols))
    var groups = List[MuonGroup]()
    groups.append(MuonGroup(offset, rows, cols, count, stride))
    # lr=1, wd=0, mu=0 -> W = -scale * O on zero params.
    run_muon(groups, params, grads, m, 1.0, 0.0, 0.0, 1.0, 1)
    var scale = Float32(0.2) * sqrt(Float32(max(rows, cols)))
    var n = rows * cols
    for mat in range(count):
        var base = offset + mat * stride
        var u = List[Float32](length=n, fill=Float32(0))
        for i in range(n):
            u[i] = grads[base + i]
        var o_ref = muon_ns_host(u, rows, cols, 5)
        var ss = Float32(0)
        for i in range(n):
            ss += o_ref[i] * o_ref[i]
        var rms = sqrt(ss / Float32(n))
        var worst = Float32(0)
        for i in range(n):
            worst = max(worst, abs(-params[base + i] / scale - o_ref[i]))
        print(
            "ns ",
            rows,
            "x",
            cols,
            " mat",
            mat,
            " max-abs=",
            worst,
            " rms=",
            rms,
        )
        assert_true(worst <= 0.05 * rms + 2e-2)
    # Padding between matrices / outside the group is untouched.
    for i in range(offset):
        assert_true(params[i] == 0.0)
    for mat in range(count - 1):
        var pad = offset + mat * stride + n
        for i in range(pad, offset + (mat + 1) * stride):
            assert_true(params[i] == 0.0)


def test_ns_matches_host_reference() raises:
    if not has_nvidia_gpu_accelerator():
        print("SKIP test_ns_matches_host_reference: needs NVIDIA GPU")
        return
    check_group_vs_host(64, 96, 1, 64 * 96, 0)
    check_group_vs_host(96, 64, 1, 96 * 64, 0)
    check_group_vs_host(64, 96, 3, 64 * 96 + 128, 32)
    check_group_vs_host(96, 64, 3, 96 * 64 + 128, 32)


def test_momentum_and_update() raises:
    if not has_nvidia_gpu_accelerator():
        print("SKIP test_momentum_and_update: needs NVIDIA GPU")
        return
    var rows = 8
    var cols = 16
    var n = rows * cols
    var lr = Float32(0.02)
    var mu = Float32(0.95)
    var wd = Float32(0.1)
    var gsc = Float32(0.5)
    var w = randn(n, 7)
    var m = List[Float32](length=n, fill=Float32(0))
    var wh = w.copy()
    var mh = m.copy()
    var groups = List[MuonGroup]()
    groups.append(MuonGroup(0, rows, cols, 1, n))
    var scale = Float32(0.2) * sqrt(Float32(cols))
    for step in range(2):
        var grads = randn(n, UInt64(100 + step))
        run_muon(groups, w, grads, m, lr, mu, wd, gsc, UInt32(step + 1))
        var u = List[Float32](length=n, fill=Float32(0))
        for i in range(n):
            var g = gsc * grads[i]
            mh[i] = mu * mh[i] + g
            u[i] = g + mu * mh[i]
        var o = muon_ns_host(u, rows, cols, 5)
        for i in range(n):
            wh[i] = wh[i] - lr * (scale * o[i] + wd * wh[i])
        for i in range(n):
            assert_almost_equal[DType.float32](
                m[i], mh[i], atol=1e-5, rtol=1e-5
            )
            assert_almost_equal[DType.float32](
                w[i], wh[i], atol=5e-3, rtol=1e-2
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
