# The intermittent `tests/test_zero.mojo` hang

A root-cause writeup of the hang where `make test-mojo` killed
`tests/test_zero.mojo` at the per-file timeout (exit 124) while about two
threads spun at 100% CPU in `sched_yield`. The bug is a deadlock between
`CpuBarrier.wait` (llmm/zero.mojo) and the AsyncRT worker pool that
`sync_parallelize` runs ranks on. It is not a race in the barrier's own
protocol. The fix is in `CpuBarrier.wait`, and a new looped regression test
turns a few-percent flake into a hang every time.

---

## The symptom

Run alone, `test_zero` takes about 1.3 s of wall clock and passes. During a
full `make test-mojo`, with GPU training runs and other agents' builds on the
same 20-core GB10, it once hung until `TEST_FILE_TIMEOUT` killed it. Its
output is buffered, so the killed run did not show which test was running.

Related history: `docs/ai/zero_row_sparse_wte_2026-07-27.md` §11 diagnosed a
separate hang through `/proc/*/wchan`. That one had one barrier spinner and
191 workers parked in `futex_wait_queue`, and it was put down to nested
`traced_parallelize` work that never got picked up. It turned out to be the
same bug, and the same fix cures it; see "What this does and does not
explain" below.

## Reproduction

The scratch harness is
`/tmp/claude-1001/-home-evan-workspace-llm-mojo/ac59f053-24bd-4cce-9ba6-964c9123a916/scratchpad/zero_hunt/stress.py`.
It builds the test exactly as `make test-mojo` does (`mojo build -I .
-Xlinker -lm`, with `$CONDA_PREFIX/lib` on `LD_LIBRARY_PATH`). It runs a copy
of the test with a flushed `print("BEGIN <test>")` added at the top of each
test function, so a hang names its test. It runs several copies at once,
next to Python busy-loop burners that oversubscribe the cores. Any run still
alive at the cap gets a `/proc/<pid>/task/*` dump (state, wchan) and
a `gdb -batch -p <pid> -ex 'thread apply all bt'`.

`ptrace_scope` is 1 on this box, so gdb cannot attach to a sibling process.
The harness therefore starts each test through a small exec wrapper that
calls `prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)` and then `execv`s the
binary. With that, attaching works and the stacks are symbolized for Mojo
frames.

| Binary | Stress | Runs | Hangs |
|---|---|---|---|
| pre-fix (main `6aea140`) | 6 concurrent copies + 24 burners, 20 cores, 20-30 s cap (a loaded run takes a few seconds) | 600 (in batches of 120, 120, 360) | **34 (5.7%)**: 4, 8, 22 |
| fix v1 (single `_run` per pump), original test file | same | 600 | **0** |
| final fix + 500-iteration regression test | same, 120 s cap | 300 | **0** |
| final fix + 500-iteration regression test | `taskset -c 0-5` (6 cores), 3 copies + 8 burners, 180 s cap | 120 | **0** |

Assuming the 5.7% pre-fix rate, the chance of 600 clean runs by luck is
about 0.943^600, or roughly 1e-15.

Hangs by test: `test_multi_cpu_reducescatter_buckets` 26,
`test_multi_cpu_reducescatter` 2, `test_multi_cpu_reducescatter_inplace` 2,
`test_multi_cpu_allreduce` 2, `test_multi_sharded_parameter_gather_cpu` 2.
Every multi-rank CPU test that synchronizes a context before a barrier can
hang. The buckets test hangs most often because it makes the most rendezvous
(`assert_ranges_agree` adds four barriers).

## What the hung process looks like

Every capture had the same shape (gdb, trimmed):

```
"Thread0": sched_yield <- CpuBarrier::wait <- ZeroContext::assert_ranges_agree <- enqueue_cpu_range wrapper
"Thread1": sem_wait <- ... <- AsyncRT_DeviceContext_synchronize <- enqueue_cpu_range wrapper
"Thread2": sched_yield <- CpuBarrier::wait <- ZeroContext::assert_ranges_agree <- ...
"Thread3": sched_yield <- CpuBarrier::wait <- ZeroContext::assert_ranges_agree <- ...
"Thread4".."Thread19": sem_wait (idle workers, asleep)
main: sem_wait <- AsyncRT_DeviceContext_synchronize   (sync_parallelize waiting for the ranks)
```

- Workers 0 to 3 always carry ranks 0 to 3. Ranks are AsyncRT tasks, not
  dedicated threads.
- Some ranks (1 to 3 of them, depending on the capture) spin in the barrier.
  The rest are **not missing and have not aborted**. They are asleep in
  `ctx.synchronize()` on the shared CPU `DeviceContext`, the call just before
  the collective in every `test_multi_cpu_*` rank body.
- Every other worker is idle and asleep, so no runnable work is visible
  anywhere.

The rank in `synchronize` is waiting for work that has finished, or that
could run, but whose completion notice never reaches it.

## Root cause

The AsyncRT runtime source (`modular/AsyncRT`, `ThreadPoolWorkQueue.cpp`,
`AsyncValue.h`, `docs/WorkQueue.md`) gives the mechanism:

1. When a worker completes an async value, the value's waiters are queued
   with `AsyncValue::runWaiterLater` and then `WorkQueue::addLocalTask`, on
   **that worker's private `localTaskList`**. The list is unsynchronized and
   no other worker can steal from it. Its owner drains it only when it next
   enters its `runItems` loop.
2. `ctx.synchronize()` called from a worker is an `await`. The worker donates
   itself to `runItems`, which may run *other* ranks' queued stream work
   while it waits. `runItems` checks its early-stop predicate ("my values are
   ready") at the top of every iteration. So it returns as soon as this
   rank's own work is done, **with whatever waiters it just queued still on
   its local list**.
3. AsyncRT assumes tasks never block (`docs/WorkQueueNonblocking.md`).
   `CpuBarrier.wait` blocks: it spins on `sched_yield`, which yields the OS
   thread but never re-enters the runtime.

Together: rank A's `synchronize` runs a stream op whose completion is rank
B's wakeup. The wakeup goes onto A's local list. A's own value is ready, so A
returns and spins in the barrier. B sleeps in `synchronize` for a wakeup only
A can deliver, and A waits in the barrier for B. Load does not create the bug.
It only changes which worker ends up running which stream op, and so how
often a parked wakeup lands on a worker that is about to spin.

That explains why the ruled-out hypotheses fail:

- **The barrier protocol is fine.** The generation counter is read and bumped
  under the spinlock, and Mojo atomics default to sequentially consistent, so
  AArch64 ordering is not a factor. `variant d` below (barriers only) never
  hung.
- **No rank leaves or aborts.** Every rank is present in the stacks.
- **The shared-ctx `synchronize` traffic alone does not hang.** `variant b`
  (the same traffic, no barrier) ran 9,000 iterations clean. The hang needs
  the spin.

## Isolating it

A scratch harness (`tests_scratch/variants.mojo` in the scratch dir) loops a
4-rank `sync_parallelize`. On an idle box:

| Variant | Rank body | Result |
|---|---|---|
| a | shared-ctx copy/fill + `synchronize`, then `barrier1`/`barrier2` (`CpuCoordinator`) | hung 6/6 runs, between iteration 0 and ~100 |
| b | same ctx traffic, no barrier | 3 x 3,000 iterations, clean |
| d | barriers only, no ctx traffic | clean |
| e0 | as a, with a scratch barrier that never pumps | hung 3/3 |
| e1 | as a, scratch barrier pumping the runtime every 32 spins | 3 x 5,000 iterations clean, ~0.6 s each |
| e1' | as e1, but pumping only once, on entry | **hung 5/5** |

(The scratch e1 pump did not have the thread-identity retry described under
"The fix".)

e1' matters. A single drain on entry is not enough, because the pump can
itself run a global task whose completion parks a new wakeup locally (the
same early-stop exit as in step 2). Pumping has to continue while the rank
waits.

## The fix

`llmm/zero.mojo`, `CpuBarrier.wait`: on the first spin iteration and every
`_BARRIER_PUMP_PERIOD` (32) iterations after it, the waiting rank calls
`_pump_asyncrt_worker()`. That function uses `std.runtime._asyncrt._run` to
run a tiny `async def` that returns `pthread_self()`:

- `_run` schedules the coroutine on the **global** queue
  (`desired_worker_id = -1`) and awaits it. If the value is not ready yet,
  the worker enters `runItems`, which drains its local list first, then its
  affinity list, then the global queue.
- The wait always finishes. Any worker, including the awaiting one, can run
  a global-queue task.
- **A single await is not proof of a drain.** An idle worker can steal the
  task and finish it before this thread reaches `await`. `await` then
  returns at once ("all values ready") without touching the local list. So
  the coroutine reports which thread ran it, and the pump retries (up to
  `_BARRIER_PUMP_MAX_TRIES` = 8) until it ran on the calling thread. Running
  on the calling thread proves the run loop was entered, and the run loop
  drains the local list before it runs any global task.
- On a non-worker thread the coroutine never runs locally. The pump then
  just pays its 8 bounded round trips, which is harmless.

The thread-identity retry is **reasoned hardening, not measured
necessity**. The first version of the fix (a single `_run` per pump) is
what produced the 600/600 result in the table. At `taskset -c 0-5` with
burners, that version did hit the stress harness's 60 s cap in 3/150 runs.
Each time it was in the regression test, with what looked like the original
signature. But the same oversubscription made an average run take about
27 s, so the cap was not a clean hang detector. A follow-up with progress
timestamps settled it. Both versions ran 6 x 2,000 loop iterations under
that load (plus 24 stray unpinned burners), and neither stalled for more
than 19 s per 100 iterations. So those 3 were most likely slow runs, not
hangs. The retry stays because the stolen-task window is real by
construction and closing it costs nothing measurable.

**Lesson for stress harnesses:** a fixed wall-clock cap on a looped liveness
test cannot tell "hung" from "slow under 3x oversubscription". Log progress
with timestamps, and call it a hang only when progress stops.

Cost: about one tiny task per 32 `sched_yield`s, paid only while a rank actually
waits. The barrier's arrival path and the rank that arrives last are
unchanged. Full `test_zero` wall clock with the new 500-iteration test is
about 1.4 s idle, against about 1.3 s before.

Caveat, written into the code: `std.runtime._asyncrt` is private API, and
Mojo exposes no public way to yield to the runtime. If it moves, the
regression test below is what catches it.

## Regression test

`tests/test_zero.mojo::test_barrier_does_not_strand_parked_asyncrt_work`
runs the exact rank-body shape (shared CPU ctx traffic, `synchronize`, two
coordinator barriers) 500 times.

- Test file built against the **pre-fix** `llmm/zero.mojo`: hung in that
  test in 10/10 runs (20 s cap) at 500 iterations, and in 5/5 runs
  (30 s cap) at an earlier 1,000.
- Fixed: 15 tests, 14 passed, 1 skipped (the multi-GPU test needs 3 or more
  GPUs). About 1.4 s.

Like `test_rank_failure_aborts_peers_instead_of_hanging`, this test fails by
hanging, so `make test-mojo` reports a regression as a per-file timeout.

## What this does and does not explain

- **Production exposure.** Both multi-rank dispatchers in `train_gpt2.mojo`
  run ranks with `sync_parallelize`, and every rank both synchronizes a
  context and waits in `CpuBarrier`. The CPU path has exactly the same shape
  as the test. On the GPU path, completions come from CUDA's event thread (a
  foreign thread, whose waiters go to the global queue), so the window is
  much narrower but not provably closed. The fix covers both, because it is
  in the barrier.
- **The row-sparse nested-parallelize hang** (zero_row_sparse doc §11) fits
  the same model. Nested `parallelize` items sit on per-worker *affinity*
  lists. If they land on a worker that is spinning in the barrier, nothing
  runs them. The pump drains the affinity list too. **Confirmed:** the disabled
  `test_row_sparse_matches_dense_with_divergent_tokens`, built against the
  pre-fix `llmm/zero.mojo`, hung 3/3 at a 60 s cap. Built against the fix, it
  passed 100/100 runs in under 0.1 s each. It is now enabled in
  `tests/test_encoder_row_sparse.mojo`'s `main()`, and the two campaign docs
  that list it as ungated carry an update note.
- **The verify-cpu "lost worker partition" flake** (memory note
  `open-flakes-f7-verifycpu`) produces wrong values, not a hang. Nothing
  here explains it. **Update:** it was the dL/dloss seed race, already
  fixed by `25fff32`. See `docs/ai/verify_cpu_wrong_values_flake.md`.

## AI use statement

Written with AI assistance (Claude Code, Opus), directed by Evan Owen.
