# Mojo 1.0.0 migration: the three breaks that mattered

A log of the August 2026 migration of `llm.mojo` from Mojo `1.0.0b3` (nightly,
`max-nightly` channel) to **Mojo 1.0.0 stable** (released 2026-08-09, `max`
channel). Mojo 1.0 is the first release with a stability commitment, so this is
the last time the project needs to track a nightly.

The compiler also emitted **1803 deprecation warnings**, none of which are what
broke the build. All three hard breaks below were silent until they were compile
errors, and two of them could not be found by reading the changelog alone. The
warnings were cleared afterwards in a separate pass (see "Clearing the
deprecation warnings"), leaving only the one category that cannot be fixed by a
rename.

---

## Break 1: accelerator APIs moved from `std` to `max`, per symbol, not per module

The changelog sentence is "most standard library APIs related to accelerator
programming have moved to a new `max` Mojo package", which reads like a
module-level move. It is not. The split runs *through* modules, so
`from std.gpu import barrier, block_idx` becomes two imports from two packages:

| Stayed in `std` | Moved to `max` |
| --- | --- |
| `std.gpu`: `block_dim`, `block_idx`, `grid_dim`, `thread_idx`, `WARP_SIZE` | `max.gpu`: `barrier` |
| `std.gpu.primitives`: `warp` (and `primitives.warp.shuffle_xor`) | `max.gpu.primitives`: `block` |
| `std.gpu.intrinsics`: `threadfence`, `Scope` | `max.gpu.memory`: `AddressSpace`, `CacheOperation`, `load` |
| `std.gpu.host.info`: `is_cpu`, `is_gpu` | `max.gpu.host`: `DeviceContext`, `DeviceBuffer`, `HostBuffer`, `DeviceAttribute`, `_nvidia_cuda.CUDA` |
| `std.algorithm`: `vectorize` | `max.algorithm`: `sync_parallelize` |

`std.gpu.host` is the sharpest edge: the package still exists and still exports
`info`, so `from std.gpu.host.info import is_cpu` keeps compiling while
`from std.gpu.host import DeviceContext` right above it does not.

The stdlib ships as compiled `.mojoc` (an `MPKG` container; `strings` finds
nothing in it), and the online API docs did not yet list the new homes, so the
mapping was recovered **from the compiler**: one probe file importing every
candidate path at once, since a failed import reports per-symbol errors while a
successful one is silent. That turns a guessing game into a single compile per
round. The table above is the probe that finally compiled clean.

52 files, rewritten mechanically and then `mojo format`ed.

## Break 2: `Int`/`UInt` are no longer `DevicePassable`

> `constraint failed: Int and UInt do not conform to DevicePassable; use a
> fixed-width type such as Int32 or Int64 instead`

Every GPU kernel here took `Int` extents (`num_params: Int`, `rows: Int`,
`channels: Int`): 61 kernels across 19 files, 142 parameters, and 88 of the
tree's 102 `enqueue_function` sites.

Two facts shaped the fix, both established by probe rather than assumption:

1. **Marshalling keys off the argument type, not the parameter type.** A kernel
   declared `(n: Int64, ...)` still fails to launch when handed a bare `Int`.
   Both sides have to change; widening only the signature is not enough.
2. **There is no implicit `Int` <-> `Int64` conversion.** `a * b` with
   `a: Int64, b: Int` is a hard error, as are `<`, `+`, `range()` and indexing.
   So widening a parameter *in place* would cascade type errors through every
   line of every kernel body that touches it.

Hence the shape actually used, widening at the boundary and restoring immediately:

```mojo
def some_kernel(n_arg: Int64, p: MutKernelPtr[dtype]):
    var n = Int(n_arg)      # body below is byte-for-byte unchanged
    ...
```

with launch sites wrapping the matching positional argument in `Int64(...)`.
This keeps every kernel body's index arithmetic at 64-bit `Int`, exactly as
before, so the change carries **no arithmetic or overflow semantics**. The only
difference on the wire is an 8-byte argument that was already 8 bytes. `Int32`
would have been the faster wire type and the more idiomatic GPU choice, but it
would have narrowed extents that the old code allowed to be 64-bit; that is a
performance decision to make deliberately with a benchmark, not a side effect of
a toolchain bump.

Both halves were applied by script (`resolve` walks each `enqueue_function` call
back through the nearest *preceding* `comptime k = kernel[...]` /
`var compiled = ctx.compile_function[k]()` binding, because alias names like
`gpu_kernel` are reused 22 times across the tree and several times within a
single file, and a global name->kernel dict silently resolves most of them to the
wrong kernel and quietly skips the rest). The insertion point skips a leading
docstring, so no kernel lost its docstring to a displaced `var`.

**The bug in the script, worth knowing before writing the next one.** Splitting a
parameter list on top-level commas is wrong when a *comment* contains a comma:

```mojo
    tile_rows_arg: Int64,  # ceildiv(rows, BLOCK_ROWS) -- row-TILE count, not the
    # physical scale-buffer row count (see nvfp4_scale_buffer_size).
    k_blocks: Int,
```

`ceildiv(rows, BLOCK_ROWS)`'s comma is safely inside parentheses, but
`row-TILE count, not the` is at depth 0, so the splitter cut the list there.
`k_blocks` landed in a chunk whose first line was prose, was not recognised as a
parameter, and was **silently left as `Int`**, while every index after it
shifted by one, so the launch-site rewriter wrapped the wrong arguments
(`Int64(sr_seed)` around a value that was already `UInt64`). Strip comments
*before* splitting.

This cost four FP4/NVFP4 test files. It was caught by the compiler, but the
useful move was auditing the whole tree for the pattern rather than fixing the
one file it reported: re-parsing every kernel both ways and diffing found 10
kernels where the two parses disagreed, of which only `_nvfp4_quantize_gpu` had
a *mid-list* shift that moves argument indices. The other nine were trailing
comments after the last parameter: a spurious final entry, no index impact,
which is why they compiled and passed. A mechanical rewrite deserves a
mechanical audit; "the compiler didn't complain" only covers the code paths that
got instantiated.

## Break 3: `InlineArray` lost `ImplicitlyCopyable`

> `value of type 'Array[Pointer[Signal, MutAnyOrigin], Int(8)]' cannot be
> implicitly copied, it does not conform to 'ImplicitlyCopyable'`

One site: `ZeroContext.get_rank_sigs_any` returned its local array by implicit
copy. Now returns by transfer (`rank_sigs_any^`). (`InlineArray` is itself
deprecated in favour of `Array`, though the compiler emits no warning for it
here, so the spelling is left alone.)

## Smaller edge: list literals infer as `Array`, not `List`

`var dst_offsets = [0]` now infers `Array[Int, 1]`, which does not convert to
the `List[Int]` that `ZeroContext.allgather_ranges` takes. Annotating the
binding (`var dst_offsets: List[Int] = [0]`) restores the old meaning.
`List[Int](7)`, the obvious alternative, no longer constructs a one-element
list at all.

---

## Clearing the deprecation warnings

1803 warnings across the tree, down to **zero**. The rewrite is
driven by **the compiler's own coordinates**, not by regex: `p + i` and `a + b`
are indistinguishable textually, but every warning carries `file:line:col`
pointing at the exact token, and for operators the underline art delimits both
operands:

```
train_gpt2.mojo:6301:39: warning: '__add__' is deprecated, use 'unsafe_offset' instead
                    model.acts.logits + (t - 1) * model.config.padded_vocab_size
                    ~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

Leading tildes are the left operand, `^` the operator, trailing tildes the
right, so the expression extent is known exactly rather than guessed. Column
conventions differ per warning kind and were calibrated against real
diagnostics before any edit: method renames point at the `.`, name renames and
`alloc` point at the token start, positional `__getitem__` points at the `[`.

Cleared: `__getitem__` -> `[unsafe_offset=i]`, `load`/`store`/`bitcast`/`free`
-> `unsafe_*`, `UnsafePointer` -> `Pointer`, `__del__` -> `__deinit__`, and
`p + i` / `p - i` / `p += i` -> `unsafe_offset`.

Three things are worth knowing:

- **Nested pointer arithmetic needs iteration, not cleverness.** `p + a + b` is
  two warnings whose ranges overlap; rewriting both in one pass corrupts the
  inner one. Overlapping sites are deferred and picked up on the next
  build-rewrite round, which converges in three passes.
- **Operators that open a continuation line have no left operand on their own
  line**, so the underline gives nothing to anchor to. These (33 sites, mostly
  a 12-fold repeated `pool + pool_slot[...] - layer * stride[...]` block) were
  rewritten as whole multi-line chains instead.
- **Relative vs absolute paths.** `make build` reports absolute paths but a
  direct `mojo build -I .` reports `./llmm/...`, and a path regex anchored on
  `/` silently skips the latter. That left 24 warnings apparently "stuck" until
  the log was normalised. If a mechanical pass reports zero progress, suspect
  the parser before the code.
- **A warning census is only as complete as the compilations you ran.** Warnings
  are emitted per *instantiation*, so building `train_gpt2` plus every
  `tests/test_*.mojo` still missed ~115 sites in `llmm/layernorm.mojo`,
  `llmm/matmul.mojo`, `llmm/attention.mojo` and `llmm/gelu.mojo`. Those live in
  functions only instantiated by `make build-mojo`, which compiles the whole
  `llmm` package through the MAX bridge rather than only the code paths one
  binary reaches. The census has to cover `build-mojo`, `build`,
  `build-profile`, `compile-rest` and the test files, or it silently
  under-reports. This surfaced only because the gate log showed warning kinds a
  "final" census had already declared clear.

**One suggested fix is actively harmful: `if var gp := ...`.** Mojo 1.0
deprecates implicit walrus bindings and suggests declaring them with `var`.
Applying that to the three `_get_global_or_null` call sites (the process-global
device-buffer cache in `llmm/memory.mojo`, `llmm/matmul.mojo`, and
`tests/_gpu_test_common.mojo`) **SIGSEGVs the MAX compiler** when a custom op is
built against the module. The crash lands inside `max/engine/api.py`'s
`compile`, nowhere near the source change.

It cost a full gate to find and is worth describing because of how it presented:
`make build` was clean, `mojo format`/`lint` were clean, all 21 mojo test files
passed, and `test_zero` passed. Only `test-python-cuda` died, on
`test_attention_equivalence.py`, because that suite is the only thing that
compiles kernels through the custom-op path. Bisecting a three-line diff
(`UnsafePointer` -> `Pointer` aliases, `bitcast` -> `unsafe_bitcast`, and the
walrus) found the walrus was the only guilty one; the other two are kept.

**The fix is to hoist rather than to annotate.** Binding first and testing
separately is warning-free AND does not crash:

```mojo
var cached = _get_global_or_null(name)      # not `if var gp := ...`
if cached:
    ...
```

So the bug is specific to the `if var x := ...` form, not to owning the
Optional: `var cached = ...` is equally owned and compiles fine. All three sites
(`llmm/memory.mojo`, `llmm/matmul.mojo`, `tests/_gpu_test_common.mojo`) use the
hoisted spelling, which is why this category is now at zero warnings. **Do not
"simplify" them back into a walrus**, and do not apply the compiler's suggested
`if var gp := ...`: that is the exact edit that crashes.

Two lessons: **a compiler's suggested fix is a suggestion, not a proof**, and a
warning cleanup needs the same gate as a semantic change, because "it still
compiles" covers none of the paths that build kernels at runtime.

**The 185 `alloc` sites, and two wrong turns worth recording.** The warning
suggests `unsafe_alloc` "as a temporary migration step". Importing that from
`std.memory` fails with "package 'memory' does not contain 'unsafe_alloc'",
which is easy to read as "the symbol does not exist". It does exist, in the
**module** `std.memory.alloc` rather than the package that re-exports `alloc`:

```mojo
from std.memory.alloc import unsafe_alloc     # count-based, not deprecated
```

The second wrong turn was concluding that the real replacement was unusable.
`alloc(Layout[T](count=n))` returns an owning `Allocation[T]`, and probing it
for `unsafe_ptr`, `span`, `data`, indexing and `release` made every probe fail,
which looked like an API with no accessors. The probes were failing for an
unrelated reason: `Allocation` is an *explicitly destroyed* type, so a probe
that allocates without calling `dealloc(a^)` does not compile no matter which
method it tests. Written correctly it is perfectly usable, and it does NOT free
on destruction, so the double-free risk that argument rested on was imaginary:

```mojo
var a = alloc(Layout[Scalar[DType.float32]](count=4))
var p = a.unsafe_ptr()      # also unsafe_span(), unsafe_leak(), into_managed()
dealloc(a^)
```

Lesson: a probe that reports "absent" for every name is evidence about the
probe, not about the API. Check one name you are *sure* exists before trusting
a sweep of negatives.

All 156 call sites route through one `heap_alloc` wrapper in `llmm/memory.mojo`,
which calls `unsafe_alloc`. That is what takes the tree to **zero** warnings.
The wrapper is also the whole migration surface: adopting the `Layout`/
`Allocation` ownership model properly is a change to that one function plus the
lifetimes of whoever holds the result, rather than a 185-site refactor. Nothing
blocks it technically; it is deferred because this codebase frees by hand at
sites far from the allocation (struct fields, process globals), so the value of
the owning handle only appears once those lifetimes move with it.

Separately, `fn` has been **removed** in 1.0 (`'fn' has been removed; use 'def'
instead`). Exactly one survived here, in `bench_gemm_vocab_tiles.mojo`, because
nothing in the gate compiled that file.

That hole is now closed by **`make compile-rest`**, wired into `check`: it
compiles the seven first-party entrypoints that neither `check` nor `test-mojo`
touched (the two benches, the vocab-tile bench, the gradient dumper, the
inference binary, the reference checker, and the calibration tool, the last of
which needs `-D LLMM_PRECISION=fp8` for its `comptime assert`). The target was
validated against the bug it exists for: re-introducing the `fn` makes it exit
non-zero with the same diagnostic. `tests/probe_fp8/` is deliberately excluded,
since several of those probes record what the toolchain could NOT do and fail
to compile on purpose; gating on them would mean a red board by design.

## Validation

`make gate` (the box-wide-serialized `make format lint check test`) on
workstation-max, 7x RTX PRO 6000 Blackwell, all seven CUDA-probed healthy first
per the standing rule that a faulted card turns a multi-rank test hang into a
45-minute discovery. Green, exit 0:

- **21** mojo test files, 0 compile failures (`test_zero.mojo` in 4 s).
- **243 passed, 5 skipped** in `test-python-cuda` (594 s). The 5 skips are the
  pre-existing generation gap: `log124M/model_19552.bin` is not present, so
  `test_infer_gpt2_generation_smoke.py` has no checkpoint to load. Unrelated to
  this migration, and still an untested path.
- `test_multi_gpu_collectives` bracket **429.7 ms**, a real run. This suite has
  historically reported `N passed` while silently returning early, and the
  bracket is how you tell; a ~0.03 bracket means it did not run.
- `make verify` (CPU **and** GPU arms, GPU with `-D LLMM_NO_TF32=1`):
  **32/32 gradient tensors OK, 0 mismatched**, and the 10-step loss trajectory
  matching the PyTorch reference on both: step 0 `5.2700095` vs expected
  `5.2678394` through step 9 `0.37649563` vs `0.3759496`. `overall okay: True`.
  This is the load-bearing evidence for Break 2: widening 142 kernel parameters
  did not perturb the arithmetic.
- All eight binaries build clean: train fp32/bf16/fp8/fp4, infer fp32/bf16,
  profile bf16/fp8/fp4. `make check` builds only the fp32 pair, so the
  low-precision arms need building explicitly, and a green `check` does not cover
  them.

**One false alarm worth recording.** The first gate run timed out
`test_zero.mojo` at the 2700 s cap (exit 124). It was not a regression: the same
binary then passed 14/14 in 2-4 s in five separate configurations: cards
{0,1,2}, {3,4,5}, a trio including card 6, and all seven visible. A timeout
under gate contention is the documented case for re-running before
root-causing; that rule does not extend to numeric mismatches, which never
"pass on rerun".

Note when reading `test-mojo` output that the bracketed per-suite timings are
**milliseconds**.

## What was deliberately NOT done

- **The `Layout`/`Allocation` ownership model.** The tree is warning-free via
  `unsafe_alloc`, which is the sanctioned migration step, not the destination.
  Adopting the owning handle properly means moving allocation lifetimes to
  match, and it is confined to `heap_alloc`. See above.
- **`uvx ruff@0.15.2` bump.** 0.16.2 is current, but the pin is load-bearing:
  unpinned ruff has dirtied the whole tree mid-merge before (0.16.0 did).
  Bumping it is a formatting decision, not an environment fix.
- **`Int32` kernel extents.** See Break 2. A performance change wearing a
  migration's clothes.

## AI use statement

Written with AI assistance (Claude Code, Opus 5), directed by Evan Owen. The
`std`/`max` symbol table and the two `DevicePassable` facts in Break 2 were
each established by compiling probe files against the 1.0.0 toolchain rather
than taken from documentation; the import and kernel-signature rewrites were
applied by script and reviewed as a diff.
