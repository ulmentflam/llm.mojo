"""Unit tests for llmm/hadamard.mojo (16x16 Randomized Hadamard Transform).

Run: `pixi run mojo run -I . tests/test_hadamard.mojo` (all tests here need
the GPU -- flock per project policy).
"""

from std.sys import has_nvidia_gpu_accelerator
from std.gpu.host import DeviceContext
from std.testing import (
    assert_equal,
    assert_true,
    assert_almost_equal,
    TestSuite,
)

from llmm.memory import MutKernelPtr, ImmutKernelPtr
from llmm.rand import MT19937
from llmm.hadamard import (
    HADAMARD_BLOCK,
    hadamard_sign,
    hadamard16_fwd_gpu,
    hadamard16_inv_gpu,
)


# ===----------------------------------------------------------------------=== #
# Sign vector sanity (host-only, no GPU)
# ===----------------------------------------------------------------------=== #


def test_sign_vector_is_plus_minus_one() raises:
    for i in range(HADAMARD_BLOCK):
        var s = hadamard_sign(i)
        assert_true(s == Float32(1.0) or s == Float32(-1.0))


def test_sign_vector_matches_documented_provenance() raises:
    # python random.Random(1234): 1 if rng.random() < 0.5 else -1, 16 draws.
    var expected: List[Float32] = [
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
    ]
    for i in range(HADAMARD_BLOCK):
        assert_equal(hadamard_sign(i), expected[i])


# ===----------------------------------------------------------------------=== #
# GPU: known golden vector (independently computed in Python, see
# llmm/hadamard.mojo module docstring / this file's task notes: x=[1..16],
# fixed sign vector above -> y = H16 @ (s ⊙ x); inverse recovers x exactly).
# All values here are small integers, exactly representable in bf16, so this
# test uses tight tolerances despite running through a bf16 buffer.
# ===----------------------------------------------------------------------=== #


def test_hadamard_known_vector_gpu() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var rows = 1
    var k = HADAMARD_BLOCK

    var x: List[Float32] = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]
    var expected_y: List[Float32] = [
        -10.0,
        -10.0,
        -10.0,
        22.0,
        10.0,
        26.0,
        18.0,
        -126.0,
        -10.0,
        -10.0,
        -10.0,
        -10.0,
        10.0,
        10.0,
        10.0,
        74.0,
    ]

    var host_x = ctx.enqueue_create_host_buffer[DType.bfloat16](k)
    for i in range(k):
        host_x.unsafe_ptr()[i] = x[i].cast[DType.bfloat16]()
    var x_dev = ctx.enqueue_create_buffer[DType.bfloat16](k)
    x_dev.enqueue_copy_from(host_x)
    var y_dev = ctx.enqueue_create_buffer[DType.bfloat16](k)
    ctx.synchronize()

    hadamard16_fwd_gpu[DType.bfloat16, "gpu"](
        rebind[MutKernelPtr[DType.bfloat16]](
            y_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DType.bfloat16]](
            x_dev.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rows,
        k,
        ctx,
    )
    ctx.synchronize()

    var host_y = ctx.enqueue_create_host_buffer[DType.bfloat16](k)
    y_dev.enqueue_copy_to(host_y)
    ctx.synchronize()

    for i in range(k):
        var got = host_y.unsafe_ptr()[i].cast[DType.float32]()
        assert_almost_equal[DType.float32](got, expected_y[i], atol=1e-2)

    # Inverse recovers x exactly (all values fit bf16 exactly).
    var xr_dev = ctx.enqueue_create_buffer[DType.bfloat16](k)
    ctx.synchronize()
    hadamard16_inv_gpu[DType.bfloat16, "gpu"](
        rebind[MutKernelPtr[DType.bfloat16]](
            xr_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DType.bfloat16]](
            y_dev.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rows,
        k,
        ctx,
    )
    ctx.synchronize()

    var host_xr = ctx.enqueue_create_host_buffer[DType.bfloat16](k)
    xr_dev.enqueue_copy_to(host_xr)
    ctx.synchronize()

    for i in range(k):
        var got = host_xr.unsafe_ptr()[i].cast[DType.float32]()
        assert_almost_equal[DType.float32](got, x[i], atol=1e-2)


# ===----------------------------------------------------------------------=== #
# GPU: orthogonality (inverse(fwd(x)) == x) and energy preservation
# (||fwd(x)||^2 == 16 * ||x||^2) on random multi-block/multi-row data.
# ===----------------------------------------------------------------------=== #


def test_hadamard_orthogonality_random_gpu() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var rows = 5
    var k = HADAMARD_BLOCK * 4  # 4 blocks per row
    var n = rows * k

    var rng = MT19937(UInt32(7))
    var host_x = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    for i in range(n):
        var u = rng.randfloat32()  # [0, 1)
        var v = (u - 0.5) * 6.0  # roughly [-3, 3)
        host_x.unsafe_ptr()[i] = v.cast[DType.bfloat16]()

    var x_dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    x_dev.enqueue_copy_from(host_x)
    var y_dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    var xr_dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    ctx.synchronize()

    hadamard16_fwd_gpu[DType.bfloat16, "gpu"](
        rebind[MutKernelPtr[DType.bfloat16]](
            y_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DType.bfloat16]](
            x_dev.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rows,
        k,
        ctx,
    )
    ctx.synchronize()
    hadamard16_inv_gpu[DType.bfloat16, "gpu"](
        rebind[MutKernelPtr[DType.bfloat16]](
            xr_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DType.bfloat16]](
            y_dev.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rows,
        k,
        ctx,
    )
    ctx.synchronize()

    var host_y = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    var host_xr = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    y_dev.enqueue_copy_to(host_y)
    xr_dev.enqueue_copy_to(host_xr)
    ctx.synchronize()

    # Orthogonality: inverse(fwd(x)) == x to bf16 precision.
    for i in range(n):
        var orig = host_x.unsafe_ptr()[i].cast[DType.float32]()
        var back = host_xr.unsafe_ptr()[i].cast[DType.float32]()
        assert_almost_equal[DType.float32](back, orig, atol=0.05, rtol=0.02)

    # Energy preservation per 16-block: sum(y_block^2) == 16*sum(x_block^2).
    var k_blocks = k // HADAMARD_BLOCK
    for r in range(rows):
        for kb in range(k_blocks):
            var ex = Float32(0.0)
            var ey = Float32(0.0)
            for kk in range(HADAMARD_BLOCK):
                var idx = r * k + kb * HADAMARD_BLOCK + kk
                var xv = host_x.unsafe_ptr()[idx].cast[DType.float32]()
                var yv = host_y.unsafe_ptr()[idx].cast[DType.float32]()
                ex += xv * xv
                ey += yv * yv
            assert_almost_equal[DType.float32](
                ey, ex * 16.0, atol=0.5, rtol=0.03
            )


def test_hadamard_multi_row_grid_indexing_gpu() raises:
    # Regression test for the flattened (row, k-block) grid index math: use
    # a row count that is not a multiple of the block-per-launch tiling to
    # make sure every row is touched exactly once.
    if not has_nvidia_gpu_accelerator():
        return
    var ctx = DeviceContext()
    var rows = 13
    var k = HADAMARD_BLOCK * 3
    var n = rows * k

    var host_x = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    for i in range(n):
        host_x.unsafe_ptr()[i] = Float32(i % 7 - 3).cast[DType.bfloat16]()
    var x_dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    x_dev.enqueue_copy_from(host_x)
    var y_dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    var xr_dev = ctx.enqueue_create_buffer[DType.bfloat16](n)
    ctx.synchronize()

    hadamard16_fwd_gpu[DType.bfloat16, "gpu"](
        rebind[MutKernelPtr[DType.bfloat16]](
            y_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DType.bfloat16]](
            x_dev.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rows,
        k,
        ctx,
    )
    hadamard16_inv_gpu[DType.bfloat16, "gpu"](
        rebind[MutKernelPtr[DType.bfloat16]](
            xr_dev.unsafe_ptr().as_unsafe_any_origin()
        ),
        rebind[ImmutKernelPtr[DType.bfloat16]](
            y_dev.unsafe_ptr().as_immutable().as_unsafe_any_origin()
        ),
        rows,
        k,
        ctx,
    )
    ctx.synchronize()

    var host_xr = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    xr_dev.enqueue_copy_to(host_xr)
    ctx.synchronize()

    for i in range(n):
        var orig = host_x.unsafe_ptr()[i].cast[DType.float32]()
        var back = host_xr.unsafe_ptr()[i].cast[DType.float32]()
        assert_almost_equal[DType.float32](back, orig, atol=0.05, rtol=0.02)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
