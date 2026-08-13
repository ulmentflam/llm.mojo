# CPU custom ops SIGSEGV on AVX-512: an alignment we promise but do not have

`make test-python` segfaults on workstation-max (Linux x86-64, AVX-512) in
`test_adamw_equivalence.py`, `test_fused_classifier_equivalence.py` and
`test_softmax_equivalence.py`. The same suite passes on macOS/arm64
(235 passed, 15 skipped). This is **our** bug, not the upstream MAX CPU
custom-op crash documented in `max_cpu_custom_op_crash_2026-07-24.md`.

## The faulting instruction

Captured by running the crashing test under gdb. Note that the upstream crash
doc reports gdb *masking* its failure (ptrace serializes thread startup); this
one reproduces under gdb, which is the first sign the two are different bugs.

```
Thread 3 "🔥 Thread1" received signal SIGSEGV, Segmentation fault.
0x00007ffc34007d50 in ?? ()
=> 0x7ffc34007d50:      vmovaps -0xc0(%rbp,%r15,1),%zmm4
```

Two things identify the bug outright:

- **`vmovaps` is the alignment-REQUIRED move.** Into a `%zmm` (512-bit)
  register it demands 64-byte alignment and faults otherwise. The unaligned
  form is `vmovups`, which would not fault.
- **The address is stack-relative** (`-0xc0(%rbp,%r15,1)`), so the unaligned
  thing is a stack temporary, not a caller-supplied tensor.

This is a *data access* at a plausible address, not a call through a null
pointer at `addr (nil)` in JIT memory, which is the signature of the upstream
bug. Different fault, different cause.

## Cause

The CPU kernels assert an alignment to the compiler:

```mojo
comptime align_d = align_of[SIMD[dtype, width]]()
comptime align_f = align_of[SIMD[DType.float32, width]]()
...
.unsafe_load[width=width, alignment=align_d]()
```

with this justification in `llmm/adamw.mojo`:

> Explicit alignment (idx = global_tid\*width is naturally aligned but the
> compiler can't prove it) so the wide **128-bit** loads/stores are emitted

"128-bit" dates the assumption: it was written when `width` was 4 for fp32, so
`align_of[SIMD[float32, 4]]` is 16 bytes, which stack slots and tensor bases
satisfy anyway. `width` is not a constant though, it is
`simd_width_of[dtype]()`, which resolves against the **host**:

| host | fp32 width | asserted alignment | emitted | result |
| --- | --- | --- | --- | --- |
| arm64 NEON | 4 | 16 B | `vmovaps %xmm` | fine |
| x86-64 AVX-512 | 16 | **64 B** | `vmovaps %zmm` | **SIGSEGV** |

`idx = global_tid * width` is also *GPU* addressing. The CPU path reaches the
same helper through `vectorize` over chunk-relative offsets, so even the stated
proof does not hold there.

This is the third bug in this codebase from `simd_width_of` resolving against
the host rather than a fixed contract, after the dbias scratch overrun
(`dbias_scratch_overrun_silent_zero_bug.md`) and the encoder channel span
(`wte_backward_cpu_channel_span_bug.md`).

## What was ruled out first

Each of these was tested, not assumed:

| Hypothesis | Test | Result |
| --- | --- | --- |
| Stale MEF cache from the old toolchain | removed `tests/.mef_cache` + `__pycache__`, fresh `build-mojo` | still crashes |
| The Mojo 1.0 pointer rewrites | ran at `6c39f07`, before them | still crashes |
| AsyncRT thread race (the upstream bug) | `MODULAR_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MODULAR_THREAD_BUSY_WAIT_US=0` | still crashes |
| xdist worker count | crashes serially with `-p no:xdist` | still crashes |
| Platform-independent | macOS arm64 full suite | 235 passed |

`-n auto` is a real aggravator but not the cause: on this 192-core box it turns
3 broken files into 13 failures, 10 of them collateral from dead workers.

## Fix

Alignment is asserted only where it is actually proven. The helpers take a
comptime `aligned: Bool`, mirroring `_encoder_fwd_vector_slice`, and the
alignment constant degrades to the element alignment when the caller cannot
prove more:

```mojo
comptime align_d = align_of[SIMD[dtype, width]]() if aligned else align_of[Scalar[dtype]]()
```

GPU callers keep `aligned=True`, where `idx = global_tid*width` and device
allocations do prove it. CPU callers pass `aligned=False`.

`llmm/softmax.mojo` is the instructive one. `_softmax_dot` in that same file
**already** had this exact fix, with a comment reading:

> `aligned` is opt-in: the CPU caller's unroll_width chunking isn't provably
> width-aligned, but the GPU caller's ... IS ... same proof as the classifier's
> `_softmax_comp_max`, which already carries this hint.

So the hazard was understood, fixed in one helper, and the neighbouring helper
was cited as precedent for keeping the unconditional hint. `_softmax_comp_max`
is the one that crashes. If you are adding an alignment hint, the question is
never "does a nearby function do this" but "is it proven **for this caller**".

`llmm/fused_classifier.mojo` needed no change at all: its own `align_of` is
inside `_fused_classifier_gpu`, and its CPU crash came entirely through the
shared `_softmax_comp_max`.

## Verification

CPU custom-op path (`-p no:xdist`), all previously segfaulting:

| file | before | after |
| --- | --- | --- |
| `test_softmax_equivalence.py` | SIGSEGV | 25 passed |
| `test_fused_classifier_equivalence.py` | SIGSEGV | 22 passed |
| `test_adamw_equivalence.py` | SIGSEGV | 6 passed |

An earlier partial experiment is worth recording as a warning: forcing the CPU
width to 4 made `test_matches_torch_trajectory[fp32_small]` pass while the rest
of the file still crashed, which briefly looked like evidence of a second
unrelated defect. It was not. It was the same bug in cases the single forced
width did not cover. One passing test is not a diagnosis; validate against the
whole file.

## Rule for the future

`align_of[SIMD[dtype, width]]` is **not a constant**. When `width` comes from
`simd_width_of[dtype]()` it tracks the host: 16 bytes on NEON, 64 bytes on
AVX-512. Any comment that says "16-byte" or "128-bit" about such an expression
was written for one host and is wrong on another. Assert alignment only behind
a comptime `aligned` flag that the caller opts into, and only where the index
arithmetic proves it.

## AI use statement

Written with AI assistance (Claude Code, Opus 5), directed by Evan Owen. The
faulting instruction was captured under gdb rather than inferred; every
hypothesis in the ruled-out table was tested on the box before being discarded.
