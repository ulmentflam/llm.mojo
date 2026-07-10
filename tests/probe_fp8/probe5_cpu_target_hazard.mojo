# FP8 toolchain probe 5: CPU-target hazard check.
#
# Prior incident (see MEMORY note bf16-build-needs-gpu-only-dispatch):
# LLMM_BF16 builds CRASH AArch64 codegen if any "cpu" *target* kernel gets
# instantiated with the low-precision dtype — the fix was to keep all bf16
# kernel dispatch comptime-guarded GPU-only. This probe checks whether
# float8_e4m3fn has the same hazard on this AArch64 (GB10) box, by
# deliberately instantiating one tiny CPU-target vectorized fp8 kernel —
# the same `vectorize[width, unroll_factor=UNROLL]` pattern
# `llmm/gelu.mojo`'s `gelu_fwd_cpu` uses — and recording what happens.
#
# Per the probe plan: a crash here is an EXPECTED, USEFUL result (it just
# means fp8 kernel dispatch must stay comptime-guarded GPU-only, exactly
# like bf16 already is). Do not retry this probe more than once if it
# crashes the compiler/process — one data point is sufficient to establish
# the hazard exists and to recommend the same GPU-only guard used for bf16.
#
# Run with (CPU-only build, no GPU/lock needed — but per the bf16 incident
# this is exactly the kind of "innocuous CPU build" that crashed codegen):
#   pixi run mojo build -I . -o /tmp/probe5_cpu_hazard \
#     tests/probe_fp8/probe5_cpu_target_hazard.mojo

from std.algorithm import vectorize
from std.memory import alloc

comptime UNROLL = 4


def _fp8_axpy_cpu(
    out_ptr: UnsafePointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin],
    x_ptr: UnsafePointer[Scalar[DType.float8_e4m3fn], ImmutAnyOrigin],
    count: Int,
) -> None:
    @always_inline
    def _simd[width: Int](i: Int) {out_ptr, x_ptr}:
        var x = (x_ptr + i).load[width=width]().cast[DType.float32]()
        var y = x * 2.0 + 1.0
        (out_ptr + i).store(y.cast[DType.float8_e4m3fn]())

    vectorize[4, unroll_factor=UNROLL](count, _simd)


def main() raises:
    print("=== probe5: fp8 CPU-target (AArch64) codegen hazard check ===")
    comptime N = 64
    var x = alloc[Scalar[DType.float8_e4m3fn]](N)
    var y = alloc[Scalar[DType.float8_e4m3fn]](N)
    for i in range(N):
        x[i] = Float32(i).cast[DType.float8_e4m3fn]()

    _fp8_axpy_cpu(
        y.as_unsafe_any_origin(),
        x.as_immutable().as_unsafe_any_origin(),
        N,
    )

    var ok = True
    for i in range(N):
        var got = y[i].cast[DType.float32]()
        var want = (
            (x[i].cast[DType.float32]() * 2.0 + 1.0)
            .cast[DType.float8_e4m3fn]()
            .cast[DType.float32]()
        )
        if got != want:
            ok = False

    x.free()
    y.free()

    if ok:
        print(
            "probe5 PASSED (CPU-target fp8 vectorized kernel compiled+ran OK)"
        )
    else:
        print("probe5 FAILED (numeric mismatch)")
