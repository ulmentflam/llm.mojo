"""Standalone micro-benchmark: TIMES the ZeRO staged-copy GPU collectives in
isolation (allreduce, reducescatter, allgather — see llmm/zero.mojo's
ZeroContext) at a fixed WORLD_SIZE across a sweep of fp32 buffer sizes.

Unlike tests/test_zero.mojo (correctness only, silently no-ops below 2 GPUs)
and scripts/benchmark_zero.py (times whole training steps), this harness
isolates the three collectives themselves and reports per-op MIN/MEDIAN wall
time plus per-rank effective bandwidth, so a collective's real cost can be
read off directly instead of inferred from a step-time delta.

Mirrors the known-good multi-GPU pattern from
tests/test_zero.mojo:test_multi_gpu_collectives: a CpuCoordinator shared by
one host thread per rank (sync_parallelize), one DeviceContext(device_id=rank)
per rank, one ZeroContext[target, WORLD_SIZE] per rank.

Unlike that test, this harness FAILS LOUDLY (nonzero exit) when fewer than
WORLD_SIZE GPUs are visible instead of silently returning.

Build (WORLD_SIZE is a comptime monomorphization parameter, default 2):
  pixi run -e cuda mojo build -D WORLD_SIZE=2 -I . -Xlinker -lm \
      -o build/bench_collectives bench_collectives.mojo

Run (pin GPUs by UUID -- see MEMORY.md workstation-max-gpu1-gsp-hang):
  CUDA_VISIBLE_DEVICES=<uuid0>,<uuid1> ./build/bench_collectives
"""

from max.algorithm import sync_parallelize
from max.gpu.host import DeviceContext
from std.math import isnan

from std.os import getenv
from std.sys import get_defined_int, has_nvidia_gpu_accelerator, simd_width_of
from std.sys.info import size_of
from std.time import global_perf_counter_ns

from llmm.zero import CpuCoordinator, ZeroContext
from llmm.memory import heap_alloc

comptime WORLD_SIZE = get_defined_int["WORLD_SIZE", 2]()
comptime DTYPE = DType.float32

# Bump when the emitted JSON shape changes, so consumers can refuse to compare
# results across incompatible schema versions rather than silently mixing them.
comptime BENCH_SCHEMA_VERSION = 1

comptime WARMUP_ITERS = 3
comptime TIMED_ITERS = 10

# Buffer-size sweep, in ELEMENTS (fp32), bracketing both sides of the ZeRO
# cost model:
#   196_608       B*T*C at B=4,  T=64
#   1_572_864
#   6_291_456
#   25_165_824    B*T*C at B=32, T=1024
#   39_421_440    ZeRO-3 embed window (wte + wpe + ln_f)
#   62_914_560
#   124_475_904   full GPT-2 124M parameter vector (reproduces the
#                 ZeroContext docstring's ~92 ms aggregate-allreduce claim)
comptime NUM_SIZES = 7


def _sweep_sizes() -> List[Int]:
    var sizes = List[Int]()
    sizes.append(196_608)
    sizes.append(1_572_864)
    sizes.append(6_291_456)
    sizes.append(25_165_824)
    sizes.append(39_421_440)
    sizes.append(62_914_560)
    sizes.append(124_475_904)
    return sizes^


def _round_up_to_multiple(n: Int, multiple: Int) -> Int:
    return ((n + multiple - 1) // multiple) * multiple


def _fmt_fixed(x: Float64, decimals: Int) -> String:
    """Format `x` with a fixed number of decimal places (no printf in Mojo)."""
    if isnan(x):
        return "nan"
    var neg = x < 0.0
    var v = -x if neg else x
    var scale = 1
    for _ in range(decimals):
        scale *= 10
    var scaled = Int(v * Float64(scale) + 0.5)  # round half up
    var int_part = scaled // scale
    var frac_part = scaled % scale
    var out = String(int_part)
    if decimals > 0:
        var frac = String(frac_part)
        while frac.byte_length() < decimals:
            frac = "0" + frac
        out += "." + frac
    return ("-" + out) if neg else out


def _sort_ascending(mut xs: List[Float64]):
    """Insertion sort -- xs is always TIMED_ITERS (>= 10) elements, so an
    O(n^2) sort is instant and avoids depending on a stdlib List sort API.
    """
    var n = len(xs)
    for i in range(1, n):
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > key:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = key


def _min_and_median(mut times: List[Float64]) -> Tuple[Float64, Float64]:
    _sort_ascending(times)
    var n = len(times)
    var min_ms = times[0]
    var median_ms: Float64
    if n % 2 == 1:
        median_ms = times[n // 2]
    else:
        median_ms = (times[n // 2 - 1] + times[n // 2]) / 2.0
    return (min_ms, median_ms)


def _report(
    mut json_rows: List[String],
    op: String,
    size_elems: Int,
    mut times: List[Float64],
    bytes_moved_per_rank: Int,
) raises:
    """Print one greppable result line and append one JSON row fragment.

    UNITS -- stated explicitly because mixing them is a real hazard:
      * `buffer_bytes` and `bytes_moved_per_rank` are EXACT integers.
      * MiB means 2^20 bytes (display convenience only).
      * GB/s means 10^9 bytes/second (DECIMAL), not GiB/s.
    The same measurement differs by 7.4% between the two bandwidth
    conventions, so the unit is named in the key, not left to a README.

    `bytes_moved_per_rank` is the exact per-rank wire volume for this op and
    is recorded in the output so the derivation is checkable rather than
    assumed: allreduce moves 2*(N-1)*shard, reducescatter and allgather move
    (N-1)*shard, where shard = buffer/N (see llmm/zero.mojo's ZeroContext).
    Bandwidth is computed from the MIN time (best-achieved, the standard
    peak-bandwidth microbenchmark convention).
    """
    var min_median = _min_and_median(times)
    var min_ms = min_median[0]
    var median_ms = min_median[1]

    var buffer_bytes = size_elems * size_of[Scalar[DTYPE]]()
    var size_mib = Float64(buffer_bytes) / (1024.0 * 1024.0)
    var bw_gbs = 0.0
    if min_ms > 0.0:
        bw_gbs = (Float64(bytes_moved_per_rank) / (min_ms / 1000.0)) / 1e9

    print(
        "BENCH_ROW op="
        + op
        + " world_size="
        + String(WORLD_SIZE)
        + " size_elems="
        + String(size_elems)
        + " buffer_bytes="
        + String(buffer_bytes)
        + " bytes_moved_per_rank="
        + String(bytes_moved_per_rank)
        + " size_mib="
        + _fmt_fixed(size_mib, 4)
        + " min_ms="
        + _fmt_fixed(min_ms, 4)
        + " median_ms="
        + _fmt_fixed(median_ms, 4)
        + " per_rank_bw_gb_s="
        + _fmt_fixed(bw_gbs, 4)
    )

    var samples_blob = String("")
    for i in range(len(times)):
        if i > 0:
            samples_blob += ","
        samples_blob += _fmt_fixed(times[i], 6)

    json_rows.append(
        '{"op":"'
        + op
        + '","world_size":'
        + String(WORLD_SIZE)
        + ',"status":"ok","size_elems":'
        + String(size_elems)
        + ',"buffer_bytes":'
        + String(buffer_bytes)
        + ',"bytes_moved_per_rank":'
        + String(bytes_moved_per_rank)
        + ',"size_mib":'
        + _fmt_fixed(size_mib, 6)
        + ',"min_ms":'
        + _fmt_fixed(min_ms, 6)
        + ',"median_ms":'
        + _fmt_fixed(median_ms, 6)
        + ',"per_rank_bw_gb_s":'
        + _fmt_fixed(bw_gbs, 6)
        + ',"samples_ms":['
        + samples_blob
        + "]}"
    )


def _check_gpu_availability(world_size: Int) raises:
    """Fail loudly (print + raise, nonzero exit) instead of the
    test-harness convention of silently returning when GPUs are short.
    """
    comptime banner = (
        "======================================================================"
    )
    if not has_nvidia_gpu_accelerator():
        print(banner)
        print("FATAL: bench_collectives requires NVIDIA GPUs; none detected.")
        print(banner)
        raise Error("bench_collectives: no NVIDIA GPU accelerator available")

    var visible = DeviceContext.number_of_devices()
    if visible < world_size:
        print(banner)
        print(
            "FATAL: bench_collectives built for WORLD_SIZE="
            + String(world_size)
            + " but only "
            + String(visible)
            + " GPU(s) are visible. Set CUDA_VISIBLE_DEVICES to at least "
            + String(world_size)
            + " devices."
        )
        print(banner)
        raise Error(
            "bench_collectives: insufficient GPUs (need "
            + String(world_size)
            + ", have "
            + String(visible)
            + ")"
        )

    # A driver-faulted GPU ("GPU requires reset") can shrink the usable
    # ordinal range below number_of_devices() -- probe every rank ordinal for
    # real instead of trusting the device count alone (mirrors
    # tests/test_zero.mojo's _gpu_multirank_available).
    for i in range(world_size):
        try:
            var probe_ctx = DeviceContext(device_id=i)
            var probe_buf = probe_ctx.enqueue_create_buffer[DType.float32](4)
            probe_buf.enqueue_fill(Float32(0.0))
            probe_ctx.synchronize()
        except e:
            print(banner)
            print(
                "FATAL: bench_collectives could not initialize GPU device_id="
                + String(i)
                + ": "
                + String(e)
            )
            print(banner)
            raise Error(
                "bench_collectives: GPU device_id="
                + String(i)
                + " failed to initialize"
            )


def main() raises:
    print(
        "=== ZeRO collective bandwidth microbenchmark (world_size="
        + String(WORLD_SIZE)
        + ", dtype=fp32, warmup="
        + String(WARMUP_ITERS)
        + ", iters="
        + String(TIMED_ITERS)
        + ") ==="
    )

    _check_gpu_availability(WORLD_SIZE)

    var cpu_coord_ptr = heap_alloc[CpuCoordinator](1)
    cpu_coord_ptr[] = CpuCoordinator(WORLD_SIZE)

    var rank_ok = heap_alloc[Int](WORLD_SIZE)
    for i in range(WORLD_SIZE):
        rank_ok[unsafe_offset=i] = 1

    comptime simd_align = WORLD_SIZE * simd_width_of[DTYPE]()

    @parameter
    def _run_rank(rank: Int):
        try:
            var ctx = DeviceContext(device_id=rank)
            var z_ctx = ZeroContext["gpu", WORLD_SIZE](
                rank=rank,
                zero_stage=1,
                ctx=ctx,
                cpu_coord=cpu_coord_ptr,
            )

            var raw_sizes = _sweep_sizes()
            var padded_sizes = List[Int]()
            var max_shard = 0
            for i in range(len(raw_sizes)):
                var padded = _round_up_to_multiple(raw_sizes[i], simd_align)
                padded_sizes.append(padded)
                var shard = padded // WORLD_SIZE
                if shard > max_shard:
                    max_shard = shard

            # Size the staged-copy scratch once, for the largest per-rank
            # shard in the whole sweep.
            z_ctx.ensure_comm_setup(max_shard * size_of[Scalar[DTYPE]]())

            var json_rows = List[String]()

            for i in range(len(raw_sizes)):
                var padded = padded_sizes[i]
                var shard = padded // WORLD_SIZE
                # Exact per-rank wire volumes (integers, no float rounding):
                # allreduce moves 2*(N-1)*shard, reducescatter and allgather
                # each move (N-1)*shard. shard divides `padded` exactly.
                var shard_bytes = shard * size_of[Scalar[DTYPE]]()
                var ar_bytes = 2 * (WORLD_SIZE - 1) * shard_bytes
                var rs_ag_bytes = (WORLD_SIZE - 1) * shard_bytes

                # ---------------- allreduce ----------------
                var ar_buf = ctx.enqueue_create_buffer[DTYPE](padded)
                ar_buf.enqueue_fill(Float32(rank + 1))
                ctx.synchronize()
                var ar_ptr = rebind[Pointer[Scalar[DTYPE], MutAnyOrigin]](
                    ar_buf.unsafe_ptr().as_unsafe_any_origin()
                )

                for _ in range(WARMUP_ITERS):
                    z_ctx.allreduce[DTYPE](ar_ptr, padded)
                ctx.synchronize()

                var ar_times = List[Float64]()
                for _ in range(TIMED_ITERS):
                    cpu_coord_ptr[].barrier1[].wait()
                    ctx.synchronize()
                    var t0 = global_perf_counter_ns()
                    z_ctx.allreduce[DTYPE](ar_ptr, padded)
                    ctx.synchronize()
                    var t1 = global_perf_counter_ns()
                    ar_times.append(Float64(t1 - t0) / 1e6)

                if rank == 0:
                    _report(
                        json_rows,
                        "allreduce",
                        padded,
                        ar_times,
                        ar_bytes,
                    )

                # ---------------- reducescatter ----------------
                var rs_in = ctx.enqueue_create_buffer[DTYPE](padded)
                var rs_out = ctx.enqueue_create_buffer[DTYPE](shard)
                rs_in.enqueue_fill(Float32(rank + 1))
                rs_out.enqueue_fill(Float32(0.0))
                ctx.synchronize()
                var rs_in_ptr = rebind[
                    Pointer[Scalar[DTYPE], MutAnyOrigin]
                ](rs_in.unsafe_ptr().as_unsafe_any_origin())
                var rs_out_ptr = rebind[
                    Pointer[Scalar[DTYPE], MutAnyOrigin]
                ](rs_out.unsafe_ptr().as_unsafe_any_origin())

                for _ in range(WARMUP_ITERS):
                    z_ctx.reducescatter[DTYPE](rs_in_ptr, rs_out_ptr, shard)
                ctx.synchronize()

                var rs_times = List[Float64]()
                for _ in range(TIMED_ITERS):
                    cpu_coord_ptr[].barrier1[].wait()
                    ctx.synchronize()
                    var t0 = global_perf_counter_ns()
                    z_ctx.reducescatter[DTYPE](rs_in_ptr, rs_out_ptr, shard)
                    ctx.synchronize()
                    var t1 = global_perf_counter_ns()
                    rs_times.append(Float64(t1 - t0) / 1e6)

                if rank == 0:
                    _report(
                        json_rows,
                        "reducescatter",
                        padded,
                        rs_times,
                        rs_ag_bytes,
                    )

                # ---------------- allgather ----------------
                # No correctness check here (that's test_zero.mojo's job) --
                # any deterministic fill is fine for a bandwidth timing.
                var ag_buf = ctx.enqueue_create_buffer[DTYPE](padded)
                ag_buf.enqueue_fill(Float32(rank + 1))
                ctx.synchronize()
                var ag_ptr = rebind[Pointer[Scalar[DTYPE], MutAnyOrigin]](
                    ag_buf.unsafe_ptr().as_unsafe_any_origin()
                )

                for _ in range(WARMUP_ITERS):
                    z_ctx.allgather[DTYPE](ag_ptr, shard)
                ctx.synchronize()

                var ag_times = List[Float64]()
                for _ in range(TIMED_ITERS):
                    cpu_coord_ptr[].barrier1[].wait()
                    ctx.synchronize()
                    var t0 = global_perf_counter_ns()
                    z_ctx.allgather[DTYPE](ag_ptr, shard)
                    ctx.synchronize()
                    var t1 = global_perf_counter_ns()
                    ag_times.append(Float64(t1 - t0) / 1e6)

                if rank == 0:
                    _report(
                        json_rows,
                        "allgather",
                        padded,
                        ag_times,
                        rs_ag_bytes,
                    )

            if rank == 0:
                var results_blob = String("")
                for i in range(len(json_rows)):
                    if i > 0:
                        results_blob += ","
                    results_blob += json_rows[i]
                # Provenance stamp. GPU UUIDs / git SHA / host are supplied by
                # the caller (the `benchmark-collectives` make target sets
                # them) because they are not obtainable from inside Mojo --
                # record ordinals here would be actively misleading, since CUDA
                # renumbers around a faulted card.
                print(
                    "BENCH_JSON: {"
                    + '"schema":'
                    + String(BENCH_SCHEMA_VERSION)
                    + ',"units":{"buffer_bytes":"bytes (exact int)",'
                    + '"bytes_moved_per_rank":"bytes (exact int)",'
                    + '"size_mib":"MiB = 2^20 bytes (display only)",'
                    + '"per_rank_bw_gb_s":"GB/s = 10^9 bytes/s (DECIMAL, not'
                    ' GiB/s)",'
                    + '"samples_ms":"milliseconds"},'
                    + '"generated":"'
                    + getenv("LLMM_BENCH_GENERATED")
                    + '","gpu_name":"'
                    + getenv("LLMM_BENCH_GPU_NAME")
                    + '","driver_version":"'
                    + getenv("LLMM_BENCH_DRIVER")
                    + '","cotenancy_mib_at_start":"'
                    + getenv("LLMM_BENCH_COTENANCY")
                    + '","world_size":'
                    + String(WORLD_SIZE)
                    + ',"dtype":"fp32","warmup_iters":'
                    + String(WARMUP_ITERS)
                    + ',"timed_iters":'
                    + String(TIMED_ITERS)
                    + ',"git_sha":"'
                    + getenv("LLMM_BENCH_GIT_SHA")
                    + '","host":"'
                    + getenv("LLMM_BENCH_HOST")
                    + '","gpu_uuids":"'
                    + getenv("CUDA_VISIBLE_DEVICES")
                    + '","results":['
                    + results_blob
                    + "]}"
                )
        except e:
            print("bench_collectives rank", rank, "error:", e)
            rank_ok[unsafe_offset=rank] = 0

    sync_parallelize[_run_rank](WORLD_SIZE)

    var all_ok = True
    for i in range(WORLD_SIZE):
        if rank_ok[unsafe_offset=i] == 0:
            all_ok = False

    rank_ok.unsafe_free()
    cpu_coord_ptr[].free()
    cpu_coord_ptr.unsafe_free()

    if not all_ok:
        print("FATAL: bench_collectives: one or more ranks raised (see above)")
        raise Error("bench_collectives: rank failure")
