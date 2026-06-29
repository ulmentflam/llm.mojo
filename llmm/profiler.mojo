from std.os import getenv
from std.time import global_perf_counter_ns


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
