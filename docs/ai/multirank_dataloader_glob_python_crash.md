# The multi-rank DataLoader glob segfault

A root-cause writeup of the startup segfault that hit multi-rank bf16 training
whenever the data paths were globs (e.g. FineWeb's
`fineweb_train_*.bin`), and the fix on branch `agent/dataloader-fix`.

---

## The symptom

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

## The three clues

1. The **identical** invocation *without* `-i/-j` (default Tiny Shakespeare
   paths) runs fine at WORLD_SIZE=8.
2. The **same** FineWeb glob patterns run fine at WORLD_SIZE=1.
3. Only multi-rank + a glob pattern crashes.

## Root cause

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

## The fix

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
every Python call. (The other pre-step Python touch points, `os.makedirs` and
`_find_max_step`, are rank-0-only and thus never concurrent.)

## Verification

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
