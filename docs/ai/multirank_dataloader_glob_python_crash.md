# Multi-rank startup crashes: Python interop on rank threads

A root-cause writeup of the startup segfaults that hit multi-rank bf16
training, and the fixes on branch `agent/dataloader-fix`. Three sites shared
one failure class (CPython interop on per-rank host threads); chasing them
also surfaced an unrelated pre-existing bf16 multi-rank resume bug (part 2
below).

## Part 1a: the DataLoader glob segfault (commit 4712628)

---

### The symptom

`make build-bf16 WORLD_SIZE=8`, then
`scripts/run_train_gpt2_bf16.sh -i "data/.fineweb10B/fineweb_train_*.bin"
-j "data/.fineweb10B/fineweb_val_*.bin" -e d12 -b 8 -t 1024 -z 2 -pn 8`
segfaults during startup — after the model-init prints, before step 1. The
stack is unambiguous:

```
libpython3.12.so   PyImport_ImportModule + 8
train_gpt2_bf16    ...
libAsyncRTMojoBindings.so
libKGENCompilerRTShared.so   (thread-pool worker entry)
libc.so.6          (clone/thread start)
```

i.e. libpython crashing inside `PyImport_ImportModule`, called from a worker
thread of the Mojo async runtime's thread pool — not the main thread.

Reproduced at the smallest crashing world size, **WORLD_SIZE=2**
(`CUDA_VISIBLE_DEVICES=2,3`), so it is not specific to the 8-GPU config.

### The three clues

1. The **identical** invocation *without* `-i/-j` (default Tiny Shakespeare
   paths) runs fine at WORLD_SIZE=8.
2. The **same** FineWeb glob patterns run fine at WORLD_SIZE=1.
3. Only multi-rank + a glob pattern crashes.

### Root cause

Multi-rank training (`_try_gpu` / `_dispatch_cpu` in `train_gpt2.mojo`) runs
one host thread per rank via `sync_parallelize[_run_rank](world_size)`. Each
rank's thread independently constructs its own `DataLoader`s inside `train()`.

`DataLoader.__init__` resolved its glob pattern *itself*, by calling into
CPython:

```mojo
var glob = Python.import_module("glob")
var py_files = glob.glob(filename_pattern)
```

That is the **first** Python interop in the process, and it happens on a bare
runtime worker thread. When N ranks hit it near-simultaneously, N threads race
to initialize/enter the CPython interpreter and import a module with no GIL
coordination between them. Concurrent interpreter init + import from threads
CPython never handed a thread-state to corrupts interpreter state and crashes
in `PyImport_ImportModule`. The crash is timing-dependent (it often surfaces on
a *later* import once state is already corrupt), which is why the rank-0 prints
appear before the dump.

This exactly explains all three clues:

- **Default paths survive** because they are single literal files (no `*`/`?`).
  `DataLoader` already had a wildcard check that appended the literal path and
  *skipped Python entirely* — so no rank ever touches the interpreter.
- **WORLD_SIZE=1 survives** because there is only one thread; the interpreter
  is initialized and used single-threaded, exactly as CPython requires.
- **Multi-rank + glob** is the only combination that does concurrent
  first-touch Python from multiple bare threads.

### The fix

Resolve the file list **once, on the main thread, before any rank thread
spawns**, and hand each rank a plain `List[String]`:

- `llmm/dataloader.mojo`: extracted the glob/literal resolution into a
  module-level `resolve_data_files(pattern) -> List[String]`. `DataLoader` now
  has two constructors: the pre-resolved `List[String]` overload (no Python
  interop — safe on rank threads), and a convenience string/glob overload that
  just calls `resolve_data_files` then delegates (for single-threaded callers
  like tests and `calibrate_fp8_scales.mojo`).
- `train_gpt2.mojo`: `main()` now calls `resolve_data_files` for the train and
  val patterns immediately after argument parsing (single-threaded, before
  `_try_gpu`/`_dispatch_cpu`), stashing the results in new `TrainArgs.train_files`
  / `TrainArgs.val_files`. `train()` passes those lists straight into the
  per-rank `DataLoader`s.

Net effect: the main thread does all globbing and is the only thread that ever
initializes the Python interpreter; rank threads do zero Python interop. This
also covers the multi-rank CPU path, which spawns rank threads the same way.

Locks were deliberately avoided — the init flow was fixed instead of guarding
every Python call.

## Part 1b: the checkpointing-flag sites (`-o`/`-n`/`-y`)

The commit above claimed the remaining pre-step Python touch points were
rank-0-only and never concurrent. That was wrong for one of them, and
insufficient for the other — the production launch (which adds
`-o log124M_fineweb -n 1000 -y 1`) still segfaulted in
`PyImport_ImportModule` after the DataLoader fix:

- `_find_max_step` (the `-y 1` resume scan) ran `Python.import_module("os")`
  + `os.listdir` from **every rank's** thread — it was never rank-gated.
- The `os.makedirs(output_dir)` call *was* rank-0-gated, but in multi-rank
  mode rank 0 **is a worker thread** (all ranks live in one process under
  `sync_parallelize`), so its Python first-touch still races the other ranks'
  `_find_max_step` imports.

The earlier verification runs never passed `-o`/`-n`/`-y`, which is why both
sites survived round 1.

Fix, same design plus one step further — these two sites don't even need
Python:

- `_find_max_step` and the makedirs call now use **pure Mojo** `std.os`
  (`listdir`/`isdir`/`makedirs`), eliminating the interop entirely.
- Both are also **hoisted to main()** before rank threads spawn: the output
  dir is created once, and the resume step is resolved once into a new
  `TrainArgs.resume_from_step` that `train()` consumes. Hoisting
  `_find_max_step` is load-bearing beyond thread safety: per-rank resolution
  could disagree on the resume step if a checkpoint landed mid-spawn.

After this, `train_gpt2.mojo` contains **zero** Python interop; the only
remaining `Python.import_module` calls in the training path live in
`resolve_data_files` (main thread only) and `llmm/hf_download.mojo`
(single-threaded `infer_gpt2.mojo` only). Checkpoint WRITE
(`write_checkpoint` → `llmm/checkpointing.mojo`) and the generation/tokenizer
path were audited: pure Mojo, no Python — a long run with `-n 1000` does not
touch Python at step N.

## Part 2: pre-existing bf16 multi-rank resume buffer overrun

With startup fixed, `-y 1` resume at WORLD_SIZE=2/ZeRO-2 got further and
died differently: `CUDA_ERROR_INVALID_VALUE` from `device_context.mojo` on
**every** rank, inside `GPT2.load_checkpoint`. WORLD_SIZE=1 resume worked.

Root cause: for sharded stages (WORLD_SIZE > 1, ZeRO >= 1) the fp32 master
weight buffer (`master_buf`) holds `optimizer_num_parameters` elements — the
per-rank comm shard (~`num_parameters / WORLD_SIZE`), not `num_parameters`.
`load_checkpoint`'s bf16 master re-seed block copied `num_parameters`
elements into it — a 2x (at WS2) device-buffer overrun that CUDA rejects with
`CUDA_ERROR_INVALID_VALUE` — and seeded from offset 0, which would have given
every rank rank-0's weight shard as its master even if it had fit.

This bug predates the Python-interop work (the code was introduced with the
master-weights resume re-seed) but was unreachable in production: any
multi-rank glob launch died at startup first, and bf16 multi-rank `-y 1`
resume had evidently never been exercised.

Fix: mirror `allocate_optimizer_moments`' shard logic — seed
`optimizer_num_parameters` elements from the loaded params at
`rank * optimizer_num_parameters`, zero-filling the final rank's alignment
padding tail (the host params buffer has no padding to read from).

Correctness evidence: after resuming from a step-5 checkpoint, WS2 and WS4
losses for steps 6-8 match a continuous WORLD_SIZE=1 run to 4-5 decimals
(e.g. step 6: 10.209476 WS1-continuous vs 10.209473 WS2-resumed vs
10.209488 WS4-resumed).

## Verification (round 2)

All with the FineWeb globs and the production hyperparameters
(`-e d12 -o <dir> -n ... -y 1 -b 64 -t 1024 -d 524288 -k cosine -l 0.0018
-u 1000 -q 0.0 -c 0.1 -v 250 -s 0 -z 2`), GPU 1 unavailable:

- **Exact production invocation, WORLD_SIZE=2** (`-pn 2`,
  `CUDA_VISIBLE_DEVICES=2,3`): 224 steps before the observation timeout
  killed it (was: startup segfault). ~295k tok/s.
- **Exact production invocation, WORLD_SIZE=4** (`-pn 4`,
  `CUDA_VISIBLE_DEVICES=2,3,4,5`): 303 steps before timeout. ~570k tok/s.
- **Checkpoint WRITE** (`-n 2 -x 5`), WS2 and WS4: checkpoints written at
  steps 2, 4, 5 (model_N.bin + per-rank state_N_r.bin).
- **Checkpoint RESUME** (`-y 1 -x 8`), WS2 and WS4: "Resumed from checkpoint
  at step 5", steps 6-8 train and write further checkpoints
  (CUDA_ERROR_INVALID_VALUE before the part-2 fix).
- **WORLD_SIZE=1** resume cycle (write `-x 5`, resume `-x 8`): unchanged.
- Regressions: default Tiny Shakespeare paths at WS2, FineWeb globs at WS1 —
  both train 3 steps.

## Verification (round 1)

Rebuilt bf16 and ran the exact failing invocation shape (`-e d12 -b 8 -t 1024
-z 2 -x 3`, FineWeb globs). GPU 1 was hardware-faulted (8 healthy GPUs
unavailable), so world size 8 was not run; the dispatch instantiates world
sizes 1/2/4/8 into every binary, so the same binary was exercised at each:

- **WORLD_SIZE=2**, FineWeb globs (`CUDA_VISIBLE_DEVICES=2,3`): 3 steps, loss
  11.02 → 9.69. (Was a segfault before the fix — reproduced twice.)
- **WORLD_SIZE=4**, FineWeb globs (`CUDA_VISIBLE_DEVICES=2,3,4,5`): 3 steps.
- **WORLD_SIZE=1**, FineWeb globs: 3 steps (regression check — unchanged).
- **WORLD_SIZE=2**, default Tiny Shakespeare paths: 3 steps (single-file path
  unchanged).
- `make format` (clean), `make lint`, `make check`, `make test` all pass —
  including `test_dataloader.mojo`'s `test_dataloader_shards` (glob pattern via
  the string constructor) and `test_dataloader_distributed`.

## AI use statement

This bug was root-caused and fixed with AI assistance via Claude Code, under
the direction of Evan Owen.
