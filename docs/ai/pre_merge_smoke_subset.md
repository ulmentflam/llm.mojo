# Pre-merge smoke subset (`make smoke`)

**Status:** measured on workstation-max (8x RTX PRO 6000 Blackwell Max-Q), MAX
`26.5.0.dev2026072606`, against `main` @ b3e0a12. Every number below is a
stopwatch reading from an actual run, not an estimate.

## What this is for

The full merge gate takes about 13 minutes. That is fine to run once before a
merge and miserable to run on every edit. `make smoke` runs the parts that
would actually catch a break in the code this campaign is touching — the LM
head, the encoder, and ZeRO — in about **40 seconds** warm.

```
make smoke
```

**`make smoke` is not a merge gate.** Merging still requires the full
`make format lint check test`, all green, on the CUDA box. Passing smoke is
necessary, never sufficient. The "What smoke does not cover" section below is
the important half of this document.

## Terms, briefly

If you already know these, skip ahead.

- **Data parallelism** — every GPU holds a complete copy of the model and
  processes a different slice of the batch. The copies are kept identical by
  summing everyone's gradients before each weight update.
- **Collective** — an operation every participating GPU (each one a *rank*)
  enters together, e.g. *allreduce* (sum a buffer across ranks, everyone gets
  the total), *reducescatter* (sum, but each rank keeps only its slice),
  *allgather* (the inverse: each rank contributes its slice, everyone ends up
  with the whole). Because every rank must call them in the same order,
  collectives are where distributed bugs hide.
- **Sharding** — splitting one logical tensor across ranks so each stores only
  a piece, instead of every rank storing the whole thing.
- **ZeRO** — a family of memory optimizations for data parallelism. Plain data
  parallelism is wasteful: N GPUs store N identical copies of the optimizer
  state, the gradients, and the weights. ZeRO shards those instead, in three
  cumulative stages — **stage 1** shards optimizer state, **stage 2** adds
  gradients, **stage 3** adds the parameters themselves. Later stages save more
  memory and require more communication to reassemble what a rank needs.
- **`wte`** — the token embedding table, which in GPT-2 is *tied*: the same
  matrix serves both as the input embedding lookup and, transposed, as the
  output ("LM head") projection that produces one logit per vocabulary entry.
  Its size is `vocab_size x channels` (50257 x 768 for GPT-2 124M), which is
  why de-residenting it is worth the trouble.

## What it runs, and why each piece earns its place

| Component | Time | Why |
|---|---|---|
| `lint` | 19s | Also the pre-commit hook, so you pay it anyway. |
| `build-mojo` | ~0s warm | **Load-bearing** — see below. |
| `tests/test_zero_equivalence.mojo` | ~8s | ZeRO stages 1/2/3 at world sizes 2 and 8, CPU-only. |
| `tests/test_zero.mojo` | ~3s | The collectives, incl. the ZeRO-2/3 bucketing path. |
| 7 pytest equivalence files (126 tests) | 7s | Encoder fwd+bwd, matmul, logits→loss, grad-norm, AdamW. |

`build-mojo` is a genuine prerequisite, not a nicety. The pytest suite does not
compile Mojo kernels on the fly; it loads pre-compiled ones from
`tests/.mef_cache/<source-fingerprint>/`. Skip the rebuild and you will happily
test the *previous* version of a kernel you just edited and get a green run that
means nothing. This is exactly the failure mode this document exists to prevent,
so the target depends on `build-mojo` rather than trusting you to remember.

The highest-value single item is `tests/test_zero_equivalence.mojo`: it drives
all three ZeRO stages at two world sizes by simulating ranks sequentially on the
CPU, so it needs no GPUs at all and still catches most sharding-math errors. It
uses a deliberately awkward `NUM_PARAMS = 4592` so the world-size-8 cases
exercise shard-length padding rather than the easy path.

Note *why* it pads, because the obvious reading is wrong: 4592 **is** divisible
by 8 (8 x 574). The padding comes from the AdamW SIMD width, not from the world
size. `train_gpt2.mojo:1908-1913` computes `base_shard = ceil(4592/8) = 574` and
then rounds it up to a multiple of `simd_width_of[GPT2_DTYPE]()`, giving
`comm_shard = 576` and `padded_num_parameters = 4608`. That rounding is
load-bearing: every rank indexes params/grads at `rank * optimizer_num_parameters`
with naturally-aligned vector loads, so an unrounded shard length misaligns every
`rank > 0` offset and the aligned CPU load segfaults. The consequence for the
test is that ranks 0-6 hold 576 real elements each (4032 total) and **rank 7
holds only 560 of its 576** — the last 16 run into the zero-filled padding
region. That last-rank short shard is the path being exercised.

## What smoke does **not** cover

`make smoke` prints these caveats itself at the end of every run, because a
caveat in a document nobody opens is not a control.

**1. GPU collectives silently vanish below 2 GPUs.** `test_multi_gpu_collectives`
(`tests/test_zero.mojo`) is the *only* test in the repo that drives real GPU
allreduce / reducescatter / allgather / `reducescatter_buckets`. It is gated on
`DeviceContext.number_of_devices() >= 2`. Until commit `df6b7bf` it opted out
with a bare `return`, so a one-GPU box printed `13 tests run: 13 passed, 0
failed, 0 skipped` while the collectives went completely untested. The
tell-tale, if you look for it, is the per-test timing: it reports
`PASS [ 0.031 ]` when it opts out versus `PASS [ 330.802 ]` when it genuinely
runs across 7 GPUs. Nothing in the summary line distinguished those. It now
prints an explicit `SKIP ... GPU collectives NOT exercised`.

That banner alone was not enough, and the gap is worth recording. It printed
while the test still counted as a pass, so the summary line — the part most
people actually read, and the only part a script scrapes — still said
`13 tests run: 13 passed , 0 failed , 0 skipped`. `0 skipped` was false, and a
truncated log tail gave you exactly the false green the banner was meant to
kill.

**The summary now tells the truth**, because `main()` marks the test skipped
through the framework instead of letting it return early:

```
SKIP tests/test_zero.mojo::test_multi_gpu_collectives: requires >=2 GPUs, found 1 - GPU collectives NOT exercised
    SKIP [ 0.001 ] test_multi_gpu_collectives
Summary [ 900.813 ] 13 tests run: 12 passed , 0 failed , 1 skipped
```

Two things to know if you touch this. `TestSuite.skip[f]()` takes the test
function as a **compile-time parameter** and must be called on the suite
instance before `run()`, so the decision lives in `main()`, not in the test
body. And the availability probe must happen *before* the suite is
constructed: it can raise, a `TestSuite` must be consumed by `run()` on every
path, and a raising call with the suite live fails to compile with
`'suite' abandoned without being explicitly destroyed`.

The bracket timing remains a useful cross-check (~300-400 ms means the
collectives really ran, ~0.03 means they did not), but it is no longer the
*only* signal. **If you are changing anything that touches cross-rank
behaviour, you need ≥2 GPUs; ask the coordinator for a window.**

**2. Nothing tests the LM head at vocabulary scale.** The LM head is a single
dense matmul with `output_channels = vocab_size_padded` (50304), called from
`train_gpt2.mojo:3025`. The largest `output_channels` any existing test passes
to that kernel is **3072** (the MLP `fc` site). `test_matmul_equivalence.py`
tops out at 2304, `test_lowp_gemm.mojo` at 128. Real vocab dimensions
(V=50257 / padded 50304) appear only in `test_softmax_equivalence.py`, which
exercises softmax and cross-entropy on pre-made logits and never calls the
matmul. **Consequence: a change that chunks or tiles the LM head over the vocab
dimension can pass this entire suite without a single test ever crossing a tile
boundary.** Anyone making that change must add a case driving the matmul at
OC≈50304 — otherwise green means "did not run", not "works".

**3. Encoder coverage is small-scale.** `tests/test_encoder_equivalence.py` does
cover both forward and the backward gradient scatter into `dwte`, checked
against torch autograd — but its largest case is B=4, T=64, V=256, C=128. That
is far from production shapes.

**4. The identical-ranges contract is documented, not enforced.**
`llmm/zero.mojo:693` (`reducescatter_buckets`) and `:963` (`allgather_ranges`)
both require every rank to pass *identical* range lists; the docstrings at
`:720-723` and `:980-982` say so plainly. **No assertion, `debug_assert`, or
runtime check enforces it.** On all four code paths the barriers sit outside the
per-range loop (barrier1 at `:819` / loop `:836` / barrier2 `:896` for the GPU
reducescatter; `:1030` / `:1038` / `:1084` for the GPU allgather), so barrier
counts cannot diverge with list length. This matters for how the bug presents:
mismatched lists **will not deadlock**. They will read and write mismatched
offsets and corrupt data silently, which is strictly worse than a hang because
there is no symptom to wait for. Any change deriving ranges from rank-local
token content is in this blast radius.

**5. Generation is never exercised by the default gate.** The five
`test_infer_gpt2_generation_smoke.py` cases skip with `build/infer_gpt2_bf16 not
built`. `make check` builds only `train_gpt2` and `profile_gpt2`. Since
generation is precisely what an LM-head change affects, run
`make build-infer-bf16` and re-run that file by hand if you touch it.

**6. LaTeX linting is a no-op here.** `lint-latex` / `format-latex` skip
entirely because `latexindent` is not installed on this box. They exit 0 without
checking anything; do not read that as coverage.

## Running the full gate

```
make format          # 2s   — must leave no residual diff
make lint            # 19s
make check           # 108s — lint + build-mojo + build + build-profile
make test            # auto-detects GPU -> test-mojo + test-python-cuda
```

Baseline on `main` @ b3e0a12, warm, quiet box: `test-mojo` 17/17 in 65s (with
7 GPUs visible), `test-python-cuda` 243 passed / 5 skipped in 617s. The
617s pytest run is the whole reason this smoke subset exists.

### Take the lock — `make gate`

**Run the full gate through the box-wide lock, always.**

```
make gate            # format + lint + check + test, serialized box-wide
```

`make gate` is exactly `format lint check test` wrapped in
`scripts/llm-mojo-gate.sh`. If you would rather not trust a script, the
zero-install equivalent locks identically and needs nothing installed:

```
flock /tmp/llm-mojo-gate.lock make test
```

This is not bureaucracy. MAX compiles are already multi-threaded, so concurrent
suites oversubscribe the cores rather than pipeline — four teams each run about
4x slower and *every duration any of them measures is garbage*. Pass/fail
survives a loaded box; timings do not. One suite at a time is strictly faster in
wall-clock terms for everyone. Measured on this box: five concurrent worktrees
drove load average to **105**. Two runners were verified to serialize correctly,
the second blocking exactly 9s behind a 12s holder.

`scripts/llm-mojo-gate.sh status` reports who holds it. **Do not lock short
targeted runs** — `make smoke`, a single test file, a GEMM sweep. A lock that is
inconvenient is a lock nobody takes.

**`make -n gate` is not a safe dry run.** GNU make executes recipe lines
containing `$(MAKE)` even under `-n`, so that command really launches the gate
script and really blocks on the lock. Use `scripts/llm-mojo-gate.sh status` to
inspect the lock instead.

**The one non-obvious property, which you must not "simplify" away.** `flock`
holds the lock on an open file descriptor, and **child processes inherit that
descriptor**. Killing the runner therefore does *not* free the lock while any
descendant lives: an orphaned `make test` keeps holding it. This is the correct
semantics — the work really is still running and really should still exclude
others — but it means the intuition *"the holder died, so the lock is free"* is
false in precisely the case where you go looking. Confirmed by SIGKILLing a
holder and watching its child hold the lock for a further 54 seconds. So
`status` scans `/proc/*/fd` for processes that actually hold the descriptor
rather than trusting the recorded PID, and an orphan shows up by name and
working directory instead of looking like a wedged lock.

To reclaim from an orphan, stop it **gracefully** — `kill -INT`, wait, then
`-TERM`. Never SIGKILL a multi-rank run; a hard kill mid-collective hangs GPU
firmware on this box. The lock frees when the last holder exits.

### The `verify` battery needs regenerated reference files

`make verify` (`verify-cpu` + `verify-gpu`) and `verify-gpu-tf32` are **not**
part of `make check` or `make test`, and on a fresh box they do not run at all:

```
gpt2_124M_debug_state.bin is llm.c's downloaded debug state, which lacks the
activation tensors this test checks. Regenerate the reference files with:
pixi run python train_gpt2.py
```

`test_gpt2.mojo:302` requires header magic **20240520** (version 3), written by
`train_gpt2.py`. The starter-pack file that `make data` downloads is llm.c's
activation-free debug state, magic **20240327**. README.md:247 documents the
regeneration as a one-time step on a fresh clone; it is easy to skip, and every
worktree symlinks the same file, so skipping it disables the verify battery
box-wide rather than in one tree. This matters more than it sounds: `verify-gpu`
is the *only* gate that checks every gradient tensor against PyTorch, so while
it is unrunnable there is no gradient-level verification of the GPU path at all.
Check the magic before trusting a green board:

```
od -An -tu4 -N4 gpt2_124M_debug_state.bin     # want 20240520, not 20240327
```

Once the reference is in place the whole battery passes — measured on the
merged LM-head work at the shipped default `LLMM_LM_HEAD_VOCAB_TILES=8`:
`verify-gpu` 39s, `verify-gpu-tf32` 35s, `verify-cpu` 40s, all exit 0, 16 of 16
gradient tensors `TENSOR OK` on both GPU paths.

**The 10-step loss trajectory is generated, not hand-maintained.** Regenerate
it with `pixi run python scripts/gen_expected_losses.py`, never by pasting this
trainer's own output — that would compare the implementation against itself and
produce a check that can never fail. The generator asserts that its step 0
reproduces the loss stored in the debug state exactly, and refuses to emit a
list otherwise. This is not hypothetical: the previous list came from a script
at a `/Users/...` path that was never in this repo, claimed step 0 was 5.354
when the reference file said 5.2678394, and consequently made `make verify-gpu`
exit 1 on *every* commit for everyone. A gate that fails for a bogus reason is
at least as harmful as one that passes without testing anything — a permanently
red gate teaches people to ignore it, and then it cannot warn them when
something real breaks.

Two practical notes. Cold runs are dominated by Mojo compilation, not by the
tests: `test_zero.mojo` runs in ~3s warm but was still compiling after 128s from
cold, which is worth remembering before you conclude something has hung. And do
**not** run the CPU `make test-python` while any trainer is alive on the box —
CPU custom-op `model.execute` segfaults under a concurrent MAX process
(`docs/ai/max_cpu_custom_op_crash_2026-07-24.md`). Use `make test-python-cuda`;
the GPU path is unaffected.

## Fresh worktree setup

A new git worktree has no pixi environment, no weights, and no dataset. Both
install steps are required — `make install` alone does *not* create the `cuda`
environment, because the provisioning rule only installs it `if [ -d
.pixi/envs/cuda ]`, which is false on a fresh tree:

```
make install        # .pixi/envs/default
make install-cuda   # .pixi/envs/cuda  <- easy to miss; test-python-cuda needs it
ln -sf /home/evan/workspace/llm.mojo/{gpt2_tokenizer.bin,gpt2_124M.bin,gpt2_124M_bf16.bin,gpt2_124M_debug_state.bin} .
mkdir -p data/.tinyshakespeare && ln -sf /home/evan/workspace/llm.mojo/data/.tinyshakespeare/* data/.tinyshakespeare/
```

Provisioning takes ~2s and 13G, hardlinked from the shared package cache;
symlinking the 1.5G of weights avoids re-downloading them per worktree.

## A note on `TEST_FILE_TIMEOUT`

The comment above `TEST_FILE_TIMEOUT ?= 2700` says `test_zero.mojo`
"legitimately needs ~30 min (13/13 passed at 1773s)". That does not reproduce
here: 13/13 in ~3-4s warm.

**Settled: the framework's bracketed figures are MILLISECONDS.** Measured
directly, by building the file ahead-of-time and timing the binary
(`mojo build -I . -Xlinker -lm tests/test_zero.mojo`; the `-Xlinker -lm` is
required or the link fails on `sincosf`):

```
Summary [ 714.635 ] 13 tests run: 13 passed , 0 failed , 0 skipped
Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.99
```

714.6ms of test time plus ~275ms of process startup is 0.99s of wall clock. As
seconds the bracket would claim 11.9 minutes for a binary that demonstrably
exits in under one second. An earlier independent run on a quiet box read
`Summary [ 376.328 ]` against 0.591s wall — the same ~200-280ms startup offset.
Two measurements, taken at different box loads, both reconcile as milliseconds
and neither reconciles as seconds.

That also explains the lore: `test_sharded_parameter_gather_gpu` reports
`[ 1737.998 ]`, i.e. 1.7 seconds, against the original comment's "1773s". The
"30 minutes" was a units misread of a number that was always a millisecond
figure. The same comment's aside about "multi-context teardown overran an 1800s
cap" is a behavioural claim that no longer has any surviving evidence behind it
and does not reproduce; treat it as unverified.

The recommendation is unchanged: **do not lower the timeout.** A generous cap
costs nothing and a wrong-but-safe number is not worth a gate failure to fix.
But do not budget 30 minutes for this file either — if it ever genuinely takes
30 minutes on this box, that is a signal something is wrong, not business as
usual.
