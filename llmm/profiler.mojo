from std.os import getenv
from std.memory import alloc
from std.sys import is_defined
from std.ffi import external_call
from std.time import global_perf_counter_ns
from std.algorithm import sync_parallelize


# Per-thread CPU trace, written directly as Perfetto/Chrome-trace JSON.
#
# We tried Mojo's native std.runtime.tracing (the LLVM TimeTraceProfiler that
# Modular uses internally): it captures per-thread spans correctly, but the MAX
# runtime only flushes them when its last DeviceContext is destroyed, and the
# GPT2 model leaks a context reference, so the trace never gets written. This
# mechanism sidesteps that entirely — it has no dependency on the MAX profiler,
# works in a compiled binary (no `mojo run` needed), and is cross-platform
# (pthread_self + plain file writes, so macOS and Linux alike, unlike nsys).
#
# The output file is an array of "X" (complete) events that the harness brackets
# with `[` ... `trace_end ]` (see thread_trace_begin/end). Each event carries the
# worker's real OS thread id (pthread_self) as `tid`, so the ~20 CPU worker
# threads land on their own lanes in https://ui.perfetto.dev.


@always_inline
def thread_trace_path() -> String:
    """The per-thread trace output path, or "" when tracing is disabled."""
    return getenv("LLMM_THREAD_TRACE")


def thread_trace_begin(path: String) raises:
    """Truncate the trace file and open the JSON event array. No-op if path="".
    """
    if path == "":
        return
    var f = open(path, "w")
    f.write("[\n")
    f.close()


def thread_trace_end(path: String) raises:
    """Close the JSON array. The final no-comma metadata event terminates the
    comma-separated span entries cleanly, yielding a valid document."""
    if path == "":
        return
    var f = open(path, "a")
    f.write('{"name":"trace_end","ph":"M","pid":0,"tid":0,"args":{}}\n]\n')
    f.close()


@always_inline
def _trace_event(
    name: String, start_ns: UInt64, end_ns: UInt64, tid: UInt64
) -> String:
    return (
        '{"name":"'
        + _json_escape(name)
        + '","ph":"X","ts":'
        + String(Float64(start_ns) / 1000.0)
        + ',"dur":'
        + String(Float64(end_ns - start_ns) / 1000.0)
        + ',"pid":0,"tid":'
        + String(tid)
        + "},\n"
    )


def thread_trace_span(
    path: String, name: String, start_ns: UInt64, end_ns: UInt64, tid: UInt64
) raises:
    """Append a single span (e.g. a harness phase on the main thread)."""
    if path == "":
        return
    var f = open(path, "a")
    f.write(_trace_event(name, start_ns, end_ns, tid))
    f.close()


@always_inline
def current_thread_id() -> UInt64:
    return UInt64(external_call["pthread_self", UInt]())


# Drop-in replacement for `sync_parallelize` that records, per worker, the OS
# thread it ran on and its wall-clock span, then appends one Perfetto event per
# worker to LLMM_THREAD_TRACE.
#
# The instrumentation is gated at COMPILE TIME on the `LLMM_TRACE` define: unless
# the binary is built with `-D LLMM_TRACE=1`, the whole tracing path (the getenv,
# the allocations, the timing, the file I/O) is comptime-eliminated and this is
# *exactly* `sync_parallelize[work_fn]` — provably zero overhead. Regular
# train/test builds therefore pay nothing; only the profiling binary opts in.
# Within a tracing build it is additionally runtime-gated by LLMM_THREAD_TRACE,
# so a profiling binary run without that env var still just does the work.
@always_inline
def traced_parallelize[
    origins: OriginSet,
    //,
    label: StaticString,
    work_fn: def(Int) raises capturing[origins] -> None,
](num_workers: Int) raises -> None:
    comptime if not is_defined["LLMM_TRACE"]():
        sync_parallelize[work_fn](num_workers)
    else:
        var path = getenv("LLMM_THREAD_TRACE")
        if path == "":
            sync_parallelize[work_fn](num_workers)
            return

        var starts = alloc[UInt64](num_workers)
        var ends = alloc[UInt64](num_workers)
        var tids = alloc[UInt64](num_workers)

        @parameter
        def _timed_worker(i: Int) raises:
            tids[i] = current_thread_id()
            starts[i] = global_perf_counter_ns()
            work_fn(i)
            ends[i] = global_perf_counter_ns()

        sync_parallelize[_timed_worker](num_workers)

        # Appended from the calling (main) thread after the barrier, so the file
        # is written sequentially — no cross-thread contention on the handle.
        var f = open(path, "a")
        for i in range(num_workers):
            f.write(_trace_event(String(label), starts[i], ends[i], tids[i]))
        f.close()

        starts.free()
        ends.free()
        tids.free()


# Escape the characters that are not legal inside a JSON string literal. Trace
# names/categories are normally plain identifiers, but args values may contain
# arbitrary text, so guard the few characters that would corrupt the document.
def _json_escape(s: String) -> String:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# A single in-process tracer that emits the Chrome Trace Event Format
# (https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU).
# The resulting JSON loads directly in the Perfetto UI (https://ui.perfetto.dev)
# or chrome://tracing — no conversion step — which is what makes it a portable,
# "modular" profiling artifact: anything that can call `complete()` produces a
# timeline Perfetto can render.
#
# Constructed with an empty path the tracer is a no-op (`enabled == False`), so
# callers can wrap hot loops unconditionally and pay nothing unless a trace path
# is requested (e.g. via LLMM_PROFILE_TRACE).
struct TraceProfiler(Movable):
    var enabled: Bool
    var path: String
    var body: String  # accumulated "traceEvents" entries, comma-separated
    var n_events: Int
    var has_t0: Bool
    var t0_ns: UInt64

    def __init__(out self, path: String):
        self.path = path
        self.enabled = path != ""
        self.body = String("")
        self.n_events = 0
        self.has_t0 = False
        self.t0_ns = 0

    # Build a tracer from an environment variable holding the output path. An
    # unset/empty variable yields a disabled tracer.
    @staticmethod
    def from_env(name: String = "LLMM_PROFILE_TRACE") -> Self:
        return Self(getenv(name))

    def _append(mut self, frag: String):
        if self.n_events > 0:
            self.body += ","
        self.body += frag
        self.n_events += 1

    # Record a complete ("X") event spanning [start_ns, end_ns]. Timestamps are
    # raw nanoseconds from `global_perf_counter_ns()`; they are rebased to the
    # first event and emitted in microseconds (the trace format's unit) so the
    # values stay small and keep full float precision.
    def complete(
        mut self,
        name: String,
        cat: String,
        start_ns: UInt64,
        end_ns: UInt64,
        pid: Int = 0,
        tid: Int = 0,
        args: String = "",
    ):
        if not self.enabled:
            return
        if not self.has_t0:
            self.t0_ns = start_ns
            self.has_t0 = True
        var ts_us = Float64(start_ns - self.t0_ns) / 1000.0
        var dur_us = Float64(end_ns - start_ns) / 1000.0
        var frag = (
            '{"name":"'
            + _json_escape(name)
            + '","cat":"'
            + _json_escape(cat)
            + '","ph":"X","ts":'
            + String(ts_us)
            + ',"dur":'
            + String(dur_us)
            + ',"pid":'
            + String(pid)
            + ',"tid":'
            + String(tid)
        )
        if args != "":
            frag += ',"args":' + args
        frag += "}"
        self._append(frag)

    # Name a process lane in the Perfetto timeline (metadata "M" event).
    def process_name(mut self, pid: Int, name: String):
        if not self.enabled:
            return
        self._append(
            '{"name":"process_name","ph":"M","pid":'
            + String(pid)
            + ',"args":{"name":"'
            + _json_escape(name)
            + '"}}'
        )

    # Name a thread/track lane in the Perfetto timeline (metadata "M" event).
    def thread_name(mut self, pid: Int, tid: Int, name: String):
        if not self.enabled:
            return
        self._append(
            '{"name":"thread_name","ph":"M","pid":'
            + String(pid)
            + ',"tid":'
            + String(tid)
            + ',"args":{"name":"'
            + _json_escape(name)
            + '"}}'
        )

    # Serialize the accumulated events to the output path. A no-op when disabled.
    def close(mut self) raises:
        if not self.enabled:
            return
        var doc = (
            '{"traceEvents":['
            + self.body
            + '],"displayTimeUnit":"ns",'
            + '"metadata":{"tool":"llm.mojo"}}'
        )
        var f = open(self.path, "w")
        f.write(doc)
        f.close()
        print(
            "[profiler] wrote "
            + String(self.n_events)
            + " events to "
            + self.path
            + " (open in https://ui.perfetto.dev)"
        )
