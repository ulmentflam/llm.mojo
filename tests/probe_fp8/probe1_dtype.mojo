# FP8 toolchain probe 1: dtype existence + scalar/SIMD casts + arithmetic.
#
# Host-only (no GPU kernel launch, no "cpu" *target* dispatch) — just checks
# that the Mojo compiler's DType enum has float8_e4m3fn / float8_e5m2 members
# and that Scalar/SIMD casts and elementwise arithmetic work on them at the
# language level. Run with: pixi run mojo run -I . tests/probe_fp8/probe1_dtype.mojo
#
# See tests/probe_fp8/RESULTS.md for the recorded outcome.


def main() raises:
    print("=== probe1: fp8 dtype existence ===")
    print("float8_e4m3fn =", DType.float8_e4m3fn)
    print("float8_e5m2 =", DType.float8_e5m2)
    print("float8_e4m3fnuz =", DType.float8_e4m3fnuz)
    print("float8_e5m2fnuz =", DType.float8_e5m2fnuz)

    print("=== probe1: scalar cast fp32 -> fp8 -> fp32 ===")
    var s32 = Float32(3.25)
    var s_e4m3 = s32.cast[DType.float8_e4m3fn]()
    var s_back = s_e4m3.cast[DType.float32]()
    print("3.25 -> e4m3 -> f32 =", s_back)

    var s_e5m2 = s32.cast[DType.float8_e5m2]()
    var s_back2 = s_e5m2.cast[DType.float32]()
    print("3.25 -> e5m2 -> f32 =", s_back2)

    print("=== probe1: SIMD cast fp32 -> fp8 -> fp32 ===")
    var v32 = SIMD[DType.float32, 8](
        0.5, 1.0, 1.5, 2.0, -1.0, 100.0, 0.0, -0.25
    )
    var v_e4m3 = v32.cast[DType.float8_e4m3fn]()
    var v_back = v_e4m3.cast[DType.float32]()
    print("v32       =", v32)
    print("e4m3 rt   =", v_back)

    var v_e5m2 = v32.cast[DType.float8_e5m2]()
    var v_back2 = v_e5m2.cast[DType.float32]()
    print("e5m2 rt   =", v_back2)

    print("=== probe1: direct fp8 arithmetic (SIMD add/mul in fp8 dtype) ===")
    var a = SIMD[DType.float8_e4m3fn, 4](1.0, 2.0, 3.0, 4.0)
    var b = SIMD[DType.float8_e4m3fn, 4](0.5, 0.5, 0.5, 0.5)
    var c = a + b
    var d = a * b
    print("a+b (e4m3, cast to f32 to print) =", c.cast[DType.float32]())
    print("a*b (e4m3, cast to f32 to print) =", d.cast[DType.float32]())

    print("=== probe1: out-of-range saturation/overflow behavior ===")
    var big = Float32(1000.0).cast[DType.float8_e4m3fn]().cast[DType.float32]()
    print("1000.0 -> e4m3 -> f32 =", big)
    var big2 = Float32(1000.0).cast[DType.float8_e5m2]().cast[DType.float32]()
    print("1000.0 -> e5m2 -> f32 =", big2)

    print("probe1 PASSED (reached end without compiler/runtime error)")
