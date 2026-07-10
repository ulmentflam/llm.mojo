# Follow-up to probe5: exercises the literal `target: StaticString` comptime
# dispatch mechanism (is_cpu[target]()/is_gpu[target]()) that the bf16 CPU-
# target hazard was actually about — llmm/gelu.mojo's `gelu_fwd[dtype,
# target]` pattern, instantiated with target="cpu" and dtype=float8_e4m3fn,
# via a real DeviceContext(api="cpu") (matching tests/test_zero.mojo's own
# CPU DeviceContext pattern) rather than plain host `vectorize()`.
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
from std.algorithm import vectorize
from std.memory import alloc


def _fp8_op[
    dtype: DType, target: StaticString
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    count: Int,
    ctx: DeviceContext,
) raises:
    comptime if is_cpu[target]():

        @always_inline
        def _simd[width: Int](i: Int) {out_ptr, x_ptr}:
            var x = (x_ptr + i).load[width=width]().cast[DType.float32]()
            var y = x * 2.0 + 1.0
            (out_ptr + i).store(y.cast[dtype]())

        vectorize[4](count, _simd)
    elif is_gpu[target]():
        raise Error("gpu path not exercised by this probe")
    else:
        raise Error("invalid target")


def main() raises:
    print("=== probe5b: fp8 target='cpu' comptime-dispatch hazard check ===")
    var ctx = DeviceContext(api="cpu")
    comptime N = 64
    comptime DT = DType.float8_e4m3fn
    var x = alloc[Scalar[DT]](N)
    var y = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Float32(i).cast[DT]()

    _fp8_op[DT, "cpu"](
        y.as_unsafe_any_origin(),
        x.as_immutable().as_unsafe_any_origin(),
        N,
        ctx,
    )

    var ok = True
    for i in range(N):
        var got = y[i].cast[DType.float32]()
        var want = (
            (x[i].cast[DType.float32]() * 2.0 + 1.0)
            .cast[DT]()
            .cast[DType.float32]()
        )
        if got != want:
            ok = False

    x.free()
    y.free()
    if ok:
        print("probe5b PASSED (target='cpu' fp8 dispatch compiled+ran OK)")
    else:
        print("probe5b FAILED (numeric mismatch)")
