# The intermittent `make verify-cpu` wrong-values failure

This is a root-cause writeup of the `make verify-cpu` flake: under load, some
runs reported most gradient tensors NOT OK and a loss trajectory that drifted
off from step 1, and the next run passed. The memory note
`open-flakes-f7-verifycpu` filed it as "suspected lost worker partition in a
CPU parallelize reduction", and `docs/ai/test_zero_hang.md` left it open. It
is not a reduction bug. It is the **dL/dloss seed race** that
`docs/ai/dloss_seed_race_cpu_backward_bug.md` (commit `25fff32`, 2026-07-27)
found through a different test and fixed. Both recorded sightings date from
before that fix. This doc shows the two are the same bug and puts a numeric
rate on the fix. It also adds the deterministic regression test that
`25fff32` could not build.

---

## The symptom, as recorded

There are two recorded sightings, both from 2026-07-10, in
`docs/ai/ai_assisted_optimizations_and_benchmarks.md`. Both came right after
builds that saturated every core:

- 12/16 tensors NOT OK, `our_l2/ref_l2 ≈ 0.65`, step-1 loss 3.9165 against
  3.8650.
- 7/16 NOT OK, ratios 0.595 to 0.70, with the exception that **`dlnfw` was
  1.324**. Step losses were off by 0.04 to 0.18 from step 1 on.

In both, the step-0 logits and loss passed. The next run passed cleanly.

## Ruling out the stale debug state

`gpt2_124M_debug_state.bin` is untracked, so a stale copy could produce
LOSS MISMATCH lines. It cannot produce this failure:

- The stale-state guard in `test_gpt2.mojo` compares the stored loss with
  step 0 of the PyTorch trajectory and refuses to run when they differ by
  more than 1e-3. Every run in this investigation, failing or passing,
  printed `LOSS OK: 5.2700706 5.2699833`. The stored value 5.2699833 matches
  the expected 5.2699823, so the state is current. The file in the main
  checkout (symlinked into this worktree) is dated Oct 22.
- A stale state also gives the same answer on every run. This failure comes
  and goes on identical inputs, which no fixed input file can do.

## Reproduction

The scratch harness is `stress.sh` in
`/tmp/claude-1001/-home-evan-workspace-llm-mojo/ac59f053-24bd-4cce-9ba6-964c9123a916/scratchpad/vcpu/`.
It runs a built `test_gpt2` binary (`pixi run mojo build test_gpt2.mojo`, the
same source `make verify-cpu` JIT-runs) with the `cpu` argument from the
worktree. It runs C copies at once next to N `while [ : ]` busy-loop burners,
optionally under `taskset`. Memory sets the concurrency limit: each run holds
about 5.3 GB RSS, and the box had about 33 GB free next to the vLLM tenant.
The GPU was not used.

"no-barrier" is HEAD `15d0968` with exactly one change: the CPU
`self.ctx.synchronize()` after the dL/dloss seed copy in `GPT2.forward`
(the `25fff32` fix) is replaced by `pass`.

| Binary | Load | Runs | Fails |
|---|---|---|---|
| HEAD `15d0968` | 3 concurrent + 24 burners, 20 cores | 40 | **0** |
| no-barrier | same | 40 | **11 (27.5%)** |
| final (`7317761`) | 3 concurrent + 24 burners | 80 | **0** |
| final (`7317761`) | `taskset -c 0-5`, 2 concurrent + 8 burners | 30 | **0** |
| no-barrier | `taskset -c 0-5`, 2 concurrent + 8 burners | 30 | **20 (67%)** |
| `make verify-cpu` (JIT), final tree | 2 concurrent + 24 burners | 12 | **0** |

At the unpinned no-barrier rate of 27.5%, the chance that the 120 unpinned
HEAD/final runs come up clean by luck is 0.725^120 ≈ 1.6e-17. At the pinned
rate of 67%, the chance for the 30 pinned final runs is 0.33^30 ≈ 4e-15.
Other agents' jobs were running on the box at the same time (load average 20
to 88), which only adds stress.

## The failing values

Every one of the 11 no-barrier failures has the recorded signature:

```
run  NOT OK  l2 ratio range     step-0 loss  logits  loss mismatches
10   9/16   [0.895, 0.954]      OK           OK      7/10
13   15/16  [0.749, 1.568]      OK           OK      8/10
1    11/16  [0.730, 0.913]      OK           OK      5/10
20   14/16  [0.777, 1.118]      OK           OK      5/10
21   12/16  [0.721, 0.913]      OK           OK      7/10
27   13/16  [0.779, 0.963]      OK           OK      8/10
31   8/16   [0.725, 0.957]      OK           OK      6/10
33   10/16  [0.862, 1.041]      OK           OK      4/10
35   9/16   [0.723, 0.939]      OK           OK      7/10
38   14/16  [0.743, 1.218]      OK           OK      9/10
3    8/16   [0.874, 0.918]      OK           OK      6/10
```

Pinning to 6 cores makes it worse, and the damage grows with how far
behind the copy falls. 20 of 30 pinned no-barrier runs failed. 13 of those
had all 16 tensors NOT OK, with L2 ratios as low as 0.05 and the step-1
loss off by up to 1.06. Every pinned failure still had step-0 logits and
loss OK.

Run 13 (unpinned) in full: every tensor's L2 is about 0.75 to 0.82 of the reference
except `dlnfw`, which is **1.568**. That matches the recorded sighting
(`dlnfw` 1.324, everything else 0.6 to 0.7). The step-1 loss is 4.0999
against 4.0596. The forward is correct (logits and loss OK) and the gradients
are scaled down overall. That is the fingerprint of dlogits rows being
zeroed, not of a numerical drift.

## Mechanism

This is a short version; `dloss_seed_race_cpu_backward_bug.md` has the full
account.

1. `GPT2.forward` writes dL/dloss = 1/(B·T) into `losses_host_buf` and
   seeds `grad_acts.losses` with an **async** `ctx.enqueue_copy`. On the CPU
   target that copy runs later on an AsyncRT worker.
2. `fused_classifier_cpu` runs **synchronously on the host**, under
   `traced_parallelize`. That is `sync_parallelize` with no context, which
   makes a fresh CPU `DeviceContext` for each call. So it is not ordered
   after the model context's queue. For every row it reads
   `d_losses_ptr[row]`, scales the softmax gradient by it, and overwrites
   the logits with dlogits in place.
3. Any row whose read comes before the copy lands sees the old contents.
   Those are zeros, because the allocation and `zero_gradients` both zero
   `grad_acts`. That row's dlogits are then zero, and its contribution is
   missing from every parameter gradient. `25fff32` measured torn, partly
   landed copies directly, so how many rows are lost varies from run to
   run. Losing some of the 256 rows shrinks most gradients' L2: the L2
   ratio went down to about 0.05 when the copy fell far behind (pinned runs
   below). A tensor whose gradient is a cancelling sum, like `dlnfw`, can
   come out larger instead. The lost-row fraction was not measured
   directly here; the dose-response below is the evidence.
4. The loss is computed from the logits **before** the in-place overwrite,
   so step 0 always passes. The damaged gradients feed AdamW, and the loss
   drifts from step 1.

Load matters because it delays the AsyncRT worker that runs the copy,
relative to the main thread's walk into the classifier. On an idle box the
copy nearly always wins. `25fff32` measured about 1 lost race in 4000 on an
idle box. The flake showed up "right after core-saturating builds" for that
reason.

So the flake was neither a lost worker partition in a reduction nor
legitimate reduction-order nondeterminism. Run to run, the passing runs are
**bit-identical**. All 120 unpinned HEAD and final runs printed identical
step losses, per-tensor maxdiffs and L2 norms (one md5 over those lines,
120/120). No tolerance needed loosening.

## Auditing for siblings

`25fff32` left an open lead: other CPU call sites might have the same
async-write / synchronous-host-read pattern. This pass went through each
`enqueue_copy` / `enqueue_fill` / `enqueue_memset` reachable on the CPU
target:

- `train_gpt2.mojo`: parameter load (`params_buf` fill and copy), grads and
  moments and master allocation, activation allocation (`acts_buf` and
  `grad_acts_buf` memsets), and `zero_gradients`. Each is followed by
  `ctx.synchronize()` before any host read. The GPU-only input and bucket
  copies are inside `comptime if is_gpu`. The loss readback synchronizes
  before the host sum. `backward()` itself makes no async writes. `update()`
  calls `adamw_update` and then synchronizes.
- `llmm/`: the `enqueue_memset`s in `attention.mojo` (GEMM-path scratch),
  `persistent_device_buffer(zero=True)` (`LN_DPARAM_SCRATCH`,
  `DBIAS_SCRATCH`, `DBIAS_COUNTERS`) and `_gemm_scratch_buffer` sit only on
  GPU paths.
- The CPU `matmul_*` paths go through `linalg.matmul(..., ctx=ctx)`, and
  the CPU kernel runs `sync_parallelize(..., ctx)`. That enqueues on the
  model context and synchronizes it, so they are ordered. They are also
  bracketed by explicit `ctx.synchronize()` calls.

No second instance was found. The regression test below gives a mechanical
check for the two hooked sites, so this does not rest on the audit alone.

## The regression test

`25fff32` shipped without a test. Losing the race required the queue to be
backed up at one instant inside `forward()`, and that could not be reached
from outside. The fix here is to inject the backlog at that instant:

- `llmm.memory.stall_cpu_queue(ctx)` enqueues a 50 ms busy-wait
  (`CPU_QUEUE_STALL_NS`) on the context's CPU queue with `enqueue_cpu_range`
  and returns at once. The queue runs in order, so nothing enqueued after it
  can land before it finishes.
- `GPT2.debug_stall_cpu_queue` (default `False`, CPU target only) calls it
  just before the seed `enqueue_copy` and at the top of `zero_gradients`. In
  production it costs one branch.
- `tests/test_cpu_queue_ordering.mojo` runs on a tiny model (L=1, C=16,
  V=64, B=2, T=8):
  - `test_stall_holds_back_queued_copy` checks the premise. A copy queued
    behind the stall has not landed when control returns, and has landed
    after `synchronize()`. If a toolchain change made the stall a no-op,
    this test fails, and the gradient test cannot pass vacuously.
  - `test_stalled_cpu_queue_does_not_change_gradients` runs two
    forward/zero/backward steps with and without the stall and requires the
    gradients to match (|Δ| ≤ 1e-6 + 1e-4·|ref|). It also requires the
    reference gradients to be mostly nonzero, so an all-zero result cannot
    match.

| Tree | Result |
|---|---|
| fixed (`7317761`) | 2/2 pass, 0.46 s wall |
| seed barrier removed | **FAIL**: 3107 of 9184 gradient elements differ, the first being ref=-9.12e-05, got=0.0 |
| only `zero_gradients`' trailing barrier removed | passes: **not caught** |

The last row is deliberate and recorded in the test. Before any host kernel
touches the zeroed buffers, the backward runs a CPU GEMM. That GEMM goes
through `linalg.matmul(..., ctx=ctx)`, whose `sync_parallelize(..., ctx)`
enqueues on the model's in-order queue and synchronizes it, and
`matmul_d_weight_bwd` also calls `ctx.synchronize()` first. So the fill is
ordered even without `zero_gradients`' own barrier. The `zero_gradients` stall would catch a future
change that removes both. It is not evidence that the barrier is needed
today.

### A trap in the hook itself

The first version of `stall_cpu_queue` took a runtime `stall_ns` and
captured it in the `enqueue_cpu_range` closure. The test then hung forever.
gdb showed AsyncRT `Thread0` spinning inside `stall_cpu_queue` and the main
thread in `AsyncRT_DeviceContext_synchronize`.
`DeviceContext.enqueue_cpu_range` wraps `func` in an `async def ... capturing`
coroutine that runs **after the enqueuing function has returned**. The
captured value was then read from a dead stack frame, and the spin bound was
garbage. `sync_parallelize` never hits this because it synchronizes before
returning. Fix: make the duration a comptime constant and capture nothing.
Any future `enqueue_cpu_range` caller that does not synchronize before
returning has the same lifetime hazard.

## Verification

- Stress tables above. The rows cover HEAD, no-barrier and the final
  commit, as built binaries and through `make verify-cpu`.
- `make lint`: clean.
- The relevant `make test-mojo` files, built and run exactly as the target
  does (`mojo build -I . -Xlinker -lm`, then run with `$CONDA_PREFIX/lib`
  on `LD_LIBRARY_PATH`). All exited 0. `test_cpu_queue_ordering` 2/2,
  `test_zero_equivalence` 6/6, `test_lm_head_vocab_tiling` 13/13,
  `test_zero` 14/15 with 1 skipped (it needs 3 or more GPUs),
  `test_chunked_cross_entropy` 10/10, `test_encoder_row_sparse` passed. The
  full `make test-mojo` was not run, because its GPU files would use the
  GPU.

## AI use statement

Written with AI assistance (Claude Code, Opus), directed by Evan Owen.
