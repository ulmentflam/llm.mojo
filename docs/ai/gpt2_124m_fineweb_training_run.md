# GPT-2 124M from-scratch training run on FineWeb-10B

A record of the first full from-scratch GPT-2 124M pretraining run completed with
`llm.mojo`, on a single NVIDIA GB10 ("DGX Spark"). This document is the training-run
companion to the two kernel/port campaigns already logged in this directory:
[`ai_assisted_optimizations_and_benchmarks.md`](ai_assisted_optimizations_and_benchmarks.md)
(the NVIDIA optimization campaign that reached bf16 parity with `llm.c`) and
[`metal_port_gotchas_and_optimizations.md`](metal_port_gotchas_and_optimizations.md)
(the Apple Silicon port). Those documents cover *making the kernels fast and
correct*; this one covers *what happened when we pointed the resulting binary at a
real 10B-token dataset for four and a half days* — the incidents, the software bugs
they exposed, and the numbers that came out the other end.

**Validation gate for the run itself:** the training loop's own loss/val-loss curve
and `make verify-gpu`/`make test` (the correctness suite from the campaigns above,
unchanged by this run) are the ground truth that the model trained correctly. No new
correctness gate was introduced for this document; it reports on operating a
long-running job on top of an already-validated kernel stack.

---

## Run overview

- **Model:** GPT-2 124M, `d12` config (12 layers, 12 heads, 768 channels), trained
  **from scratch** — random init, no pretrained weights loaded.
- **Dataset:** FineWeb classic, 10B-token sample (`HuggingFaceFW/fineweb`,
  `sample-10BT` config via `data/fineweb.py`), GPT-2 BPE tokenizer (`tiktoken`
  `gpt2` encoding). One full epoch, 19,552 steps.
- **Precision:** bf16 mixed precision — params, activations, and gradients in
  bf16; fp32 AdamW master weights and moments (`comptime USE_BF16` /
  `MASTER_DTYPE` in `train_gpt2.mojo`).
- **Hardware:** single NVIDIA GB10 (Grace-Blackwell devkit, "DGX Spark"), driver
  595.71.05, CUDA 13.2, Linux 6.17.0-1021-nvidia aarch64. Mojo
  1.0.0b3.dev2026062706.
- **Effective batch size:** 524,288 tokens/step (`-d 524288`) — micro-batch
  `B=32` (`-b 32`), `T=1024` (`-t 1024`), 16 gradient-accumulation steps.
- **Learning-rate schedule:** cosine, peak `6e-4` (`-l 0.0006`), 700-step warmup
  (`-u 700`), decayed to exactly `0.0` at step 19,552 (`-q 0.0`), weight decay
  `0.1` (`-c 0.1`).
- **Launch script:** `scratch/train_fineweb_124M.sh`, supervised by autosentry
  (a self-healing process supervisor) via `.autosentry/autosentry.yaml` in this
  repo for crash-resilient checkpoint-resume on restart.

All of the flags above were re-checked against the current
`scratch/train_fineweb_124M.sh` and `.autosentry/autosentry.yaml` (both still
present and unchanged in content from what the run used) — see
[Replication instructions](#replication-instructions).

## Final results

- **Final checkpoint:** `log124M/model_19552.bin` (248,952,832 bytes ≈ 249 MB,
  bf16 params) + `log124M/state_19552_0.bin` (995,808,256 bytes ≈ 996 MB, fp32
  AdamW moments — only needed to resume further training, not for inference).
- **Final train loss** (step 19552, the last logged step): **3.345018**. **Final
  val loss: 3.2806964** — essentially flat vs. step 19000's val loss of
  **3.2820423** (and the checkpoints in between: 3.2808414, 3.2808414, 3.2806778,
  3.2807465), i.e. the model had converged well before the LR schedule finished
  decaying to zero.
- **Steady-state throughput:** ~33.6k tok/s (~15.5–15.6 s/step at B=32,
  T=1024), **~21.6% bf16 MFU**, confirmed directly from the training log
  (`step 19552/19552 | loss 3.345018 | ... | 15572.90 ms | 21.6% bf16 MFU | 33659
  tok/s`).

## Timing

All times UTC.

| | |
|---|---|
| Real run launch (T0)* | 2026-07-05 17:43:35 |
| Completion (final checkpoint written, clean exit) | 2026-07-10 04:51:13 |
| Total wall clock | 4 days, 11:07:38 (107.13 h) |
| Total downtime (4 incidents, see below) | 11:03:17 (11.05 h) |
| GPU-busy time (wall clock minus downtime) | 4 days, 0:04:21 (96.07 h) |
| — of which net forward progress (19,552 unique steps) | 88.81 h |
| — of which wasted redo from crash-triggered replays | 7.60 h |

*T0 excludes an earlier ~35-minute setup/false-start window (17:07–17:43 UTC) spent
fixing an autosentry stall-detector false-positive bug before the real,
correctly-configured run began. `.autosentry/autosentry.yaml` still documents why:
its `detectors` list omits a `stall` detector on purpose, with a comment explaining
that in the autosentry version used, the stall detector's tracked metric did not
re-attach to a restarted child's log stream (it kept the pre-restart step number),
so it fired spuriously every backoff interval and kill-looped the trainer during
this exact false-start window (incidents timestamped 2026-07-05T17-40/17-41 in
`.autosentry/incidents/`). Hang detection was moved to an external log-mtime
watchdog instead. That window is not counted as training time above.

**Bottom line:** if the two unrelated hardware crashes (incidents #1 and #4 below)
hadn't happened, this run would have taken **~88.8 hours (3.7 days)** of clean
compute. The actual 4.46-day wall clock is attributable to those two unexplained
hardware/power events, not to the training code, autosentry, or the bf16 kernel.

## Incident timeline

Four events, chronological.

### Incident 1 — Unclean reboot #1

**2026-07-07 16:28:13 → 20:06:40 UTC (3h 38m 27s).**

The whole machine went down with **zero kernel-level trace of any kind** — no
panic, no OOM, no thermal shutdown, no MCE, nothing. Confirmed via
`journalctl --list-boots` boot-boundary timestamps and `last -x`, which labeled
the session "crash" (the wtmp convention for "the next boot appeared with no
matching shutdown record"). Root cause unknown/external (most likely power loss
or a hard hang beneath what Linux can log) — **not related to the training code**.
Recovered via manual relaunch once discovered.

### Incident 2 — Deliberate restart (not a crash)

**2026-07-07 20:42:25 → 20:47:58 UTC (5m 33s).**

An intentional kill to rebuild the bf16 binary picking up commit `fcddc74`
("Fixing training seed for resuming the run" — the master-weight reseed fix, see
[Software fixes](#software-fixes-made-duringbecause-of-this-run) below).
autosentry's Claude-healer subprocess auto-escalated (its unverified-restart
budget was exhausted) during this restart and additionally:

- Committed the long-untracked `scratch/*.sh` operational scripts onto a new
  side branch, `autosentry/fix-2026-07-07T20-47-47Z-error-exit_code`.
- Misdiagnosed the cause as OOM-like and halved the batch-size env override to
  `B=16` for a while. Harmless: gradient accumulation keeps the effective batch
  size constant at 524,288 tokens regardless of the micro-batch size, so only
  throughput was affected, not the training trajectory (this exact invariant is
  called out in the `B` sweep comment at the top of
  `scratch/train_fineweb_124M.sh`).

### Incident 3 — Xid 13/43 GPU exception (end-of-run sampling crash)

**2026-07-09 18:51:29 → 18:55:39 UTC (4m 10s).**

Confirmed via the kernel log:
```
NVRM: Xid (PCI:000f:01:00): 13, Graphics SM Warp Exception ... Misaligned Address
```
firing across dozens of SM/TPC/GPC units, immediately followed by:
```
NVRM: Xid (PCI:000f:01:00): 43, pid=74059, name=train_gpt2_bf16, channel 0x00000002
```
— the driver detecting the fault and force-resetting that CUDA channel, killing
the training process. This surfaced to the process as `CUDA_ERROR_MISALIGNED_ADDRESS`
at `train_gpt2.mojo:1905`.

**Root cause:** the bf16 B=1 end-of-run text-generation/sampling code path
(`model.forward(gen_tokens, null_targets, batch_size=1, seq_len=t)`), exercised
only **once** in this entire run — at the very last training step. The run's
`sample_every` was 20000 (confirmed in the training log's printed argument
table), which never evenly divides the 19,552-step run, so the periodic
mid-run sampling check never fires — but `train_gpt2.mojo`'s last-step logic
forces one sample at the final step regardless of `sample_every` (see
`... or last_step` in the sampling condition), so this B=1 inference-shape
code path was hit for the first time ever at the worst possible moment: the
very last step of a 4.5-day run.

This did **not** crash the whole machine — the OS kept logging normally for
~20 more minutes afterward, so this is a distinct, contained GPU-driver-level
event, separate from the two full "unclean reboot" incidents.

**Compounding factor:** the training loop writes checkpoints *after* the sampling
block, so this crash cost the true final checkpoint (`model_19552.bin`) on the
first attempt — training had actually already fully converged (LR at exactly
`0.0`) one step earlier, but that state was never persisted to disk.

**Recovery:** disabled sampling (`-s 0`) and resumed from the step-19000
checkpoint. The recovery was further compounded by the restart script
`scratch/train_fineweb_124M.sh` having been deleted from disk — an earlier
`git checkout main` deleted it because it had only become git-tracked on the
side branch created during incident #2 — so autosentry's automatic restart
initially failed outright with "No such file or directory" until the script was
manually restored (see commit `66c0ff5`'s message, which explains this
sequence directly).

### Incident 4 — Unclean reboot #2

**2026-07-09 19:11:57 → 2026-07-10 02:27:04 UTC (7h 15m 07s).**

A separate full-machine crash, ~20 minutes after the Xid event in incident #3 —
again zero kernel-level trace (no panic/thermal/MCE/OOM/additional Xid); `last -x`
again shows "crash". A cron watchdog installed after incident #1 specifically to
auto-relaunch the training supervisor on reboot (`@reboot sleep 30 &&
scratch/ensure_autosentry.sh`, plus a `*/10 * * * *` liveness check) fired
exactly on schedule twice (confirmed in `scratch/ensure_autosentry.log`) but
silently failed both times: the script invoked bare `autosentry`, resolvable only
via `PATH` in an interactive shell — cron's minimal environment doesn't include
`~/.local/bin` (a `uv tool install` location) — so
`nohup: failed to run command 'autosentry': No such file or directory`, and no
relaunch happened. Fixed by hardcoding the absolute path in commit `cb331e4`,
whose message documents the same root cause: *"cron's minimal environment
doesn't include ~/.local/bin, so `autosentry` ... resolved to nothing, and
training sat idle for ~13 [hours]."*

---

## Software fixes made during/because of this run

All six commits below were re-verified against current `git log` — all still
exist at the hashes cited and their diffs match the descriptions given.

- **`a0782b0`** — "Fixies cpu branching" (2026-07-03, i.e. *before* this run
  started on 2026-07-05). This is the commit that actually added the bf16
  CPU-dispatch compile guard: `-D LLMM_BF16=1` builds must not instantiate the
  `"cpu"` dispatch target, because the CPU bf16 GEMM packing path crashes
  AArch64 instruction selection at compile time. The guard lives in
  `_dispatch_cpu` in `train_gpt2.mojo` (current line ~3709):
  `comptime if USE_BF16: raise Error("bf16 build supports only the GPU target
  (CPU stays fp32)...")`, matching the existing fp32-only-CPU policy in
  `profile_gpt2.mojo`. **Correction to an earlier draft of this history:** this
  fix was originally misattributed to `17ae22a`; that commit is a *different*,
  later (2026-07-06, mid-run) dispatch-logic cleanup — it reorganizes
  `_dispatch_gpu` into `_try_gpu`, adds an explicit compile-time Metal
  opt-out and a clearer error for `world_size > 1` on Apple GPU, and simplifies
  `main()`'s CPU/GPU branch — it does not touch the bf16/AArch64 guard at all.
  Both are real commits in this repo; only the attribution of *which fix* is
  which has been corrected here.
- **`aa922a4`** ("fixing batch size", 2026-07-05, pre-launch) — epoch/LR-schedule
  step-count fix in `train_gpt2.mojo`. `train_num_batches` was computed by
  dividing total train tokens by the *micro-batch* token count
  (`tokens_per_fwdbwd`) instead of the *total* (grad-accumulated) batch size
  (`total_batch_size`), inflating a run's total step count — and therefore its
  cosine LR decay horizon — by `grad_accum_steps`× (16× in this run: would have
  been ~312,841 steps and a badly-stretched LR schedule instead of the correct
  19,552).
- **`633032c`** ("Fixing tokenization speed", 2026-07-05, pre-launch) —
  tokenizer LRU-cache fix in `data/utils.py` (`get_gpt2_encoding()`) and
  `data/fineweb.py` (`tokenize_llama`'s tokenizer, plus an unrelated
  `.tolist()` removal in the shard-writer). `get_gpt2_encoding()` was rebuilding
  the entire 50k-merge BPE table from scratch on every single document instead
  of once — a measured ~385× slowdown — turning FineWeb-10B tokenization from
  well under an hour into a multi-day ordeal. Fixed with
  `@functools.lru_cache(maxsize=1)`.
- **`fcddc74`** ("Fixing training seed for resuming the run", 2026-07-06,
  mid-run) — bf16 checkpoint-resume master-weight reseed in `train_gpt2.mojo`.
  Resuming a bf16 run without re-seeding the fp32 "master" weight copy from the
  just-loaded (already-trained) bf16 params meant the next optimizer step wrote
  `bf16(stale_initial_master - delta)` back into params, silently discarding all
  trained progress (loss briefly jumping back to ~11.0, i.e. random-init level)
  for 100+ steps until it re-converged. This exact bug recurred once more during
  incident #3's recovery because the *binary* running at the time predated this
  source fix (a stale-binary problem, not a regression in the fix itself);
  rebuilding resolved it.
- **`66c0ff5`** ("Commit training operation scripts and disable crashing
  end-of-run sampling", 2026-07-09) — committed the operational scripts
  (`scratch/*.sh`) that had been untracked the whole run, and disabled
  end-of-run sampling (`-s 0` in `scratch/train_fineweb_124M.sh`) as a stopgap
  for incident #3's GPU crash. The commit message directly documents the
  git-checkout-deletes-side-branch-only-files sequence described in incident 3.
- **`cb331e4`** ("Fix cron PATH failure in ensure_autosentry.sh: use absolute
  autosentry path", 2026-07-09) — see incident #4.

## Known open issue

The bf16 B=1 text-generation code path that caused incident #3 is suspected to
be a **race condition** (missing synchronization between async GPU kernel
launches), not a fundamental shape/alignment bug: rerunning the identical
crash-triggering code path with `MODULAR_DEBUG=device-sync-mode` (forces
synchronous kernel launches) completed successfully twice in a row.

As of this writing, **the fix has not landed** — `scripts/run_train_gpt2.sh`
still carries it only as a commented-out TODO:
```sh
# TODO: Race condition fix: Enable device-sync-mode
# export MODULAR_DEBUG="${MODULAR_DEBUG:-device-sync-mode}"
```
and `git log` shows no commits after `cb331e4` (the current `HEAD`) touching
`llmm/matmul.mojo`, `llmm/attention.mojo`, or introducing a new inference-only
binary. Sampling stays disabled (`-s 0`) in the shipped training script in the
meantime — this is a workaround, not a fix. Check the state of that TODO and
`git log` before relying on end-of-run sampling in any future run.

---

## Replication instructions

Verified against the current repo state (2026-07-10); flags and commands below
match what's actually in the Makefile/README/scripts today.

1. **Clone and bootstrap the environment.** Per `README.md`:
   ```sh
   git clone --recurse-submodules https://github.com/ulmentflam/llm.mojo.git
   cd llm.mojo
   curl -fsSL https://pixi.sh/install.sh | sh   # install pixi, if needed
   make install-cuda        # CUDA-enabled deps (use `make install` for CPU-only)
   ```
   `make install`/`make install-cuda` do *not* fetch the FineWeb dataset — only
   the small Tiny Shakespeare + starter-weights bundle via `make data`, which
   isn't needed for this run.

2. **Download and tokenize FineWeb classic 10B:**
   ```sh
   pixi run python data/fineweb.py -t classic -v 10B -m gpt-2
   ```
   Writes ~104 shards (shard 0 is validation, the rest train) to
   `data/.fineweb10B/`, ~20 GB total. **This requires the tokenizer-caching fix
   (commit `633032c`) to be present** — confirmed still in `data/utils.py`'s
   `get_gpt2_encoding()` as of this writing — or tokenization takes days instead
   of under an hour.

3. **Build the bf16 trainer.** As of this writing there is still no dedicated
   Makefile target for a bf16 build of the full trainer (`build`/`build-train`
   builds the fp32 `build/train_gpt2`; `build-profile-bf16` only builds the
   *profiling harness*, `build/profile_gpt2_bf16`, not the trainer). The manual
   command used for this run, confirmed against the Makefile's own
   `MOJO_INCLUDES`/`MOJO_LINK_FLAGS`/`WORLD_SIZE` conventions, is:
   ```sh
   pixi run mojo build -D WORLD_SIZE=1 -D LLMM_BF16=1 -I . -Xlinker -lm \
     -o build/train_gpt2_bf16 train_gpt2.mojo
   ```
   If a `make build-bf16`-style target has since been added, prefer it — check
   `make help` / the Makefile's `train`/`build` section first.

4. **Launch the run**, ideally under autosentry for crash resilience:
   ```sh
   scratch/train_fineweb_124M.sh
   # or, per .autosentry/autosentry.yaml's process.command:
   autosentry run   # from the repo root, using this repo's .autosentry/autosentry.yaml
   ```
   Confirmed current contents of `scratch/train_fineweb_124M.sh` match what
   this run used: `B=32` default (overridable via env), `-e d12`, checkpoints
   every 1000 steps (`-n 1000`) to `-o log124M`, resume-enabled (`-y 1`),
   `-t 1024 -d 524288 -l 0.0006 -q 0.0 -u 700 -c 0.1 -v 250`, and **sampling
   now disabled by default (`-s 0`)** — a change made *because of* incident #3,
   so a from-fresh-clone replication today will not by default reproduce that
   crash.

5. **Budget time.** Expect **~88.8 hours (3.7 days)** of GPU-busy time on a
   single idle GB10 assuming zero interruptions. Budget more wall-clock margin:
   this box has an unexplained history of unclean reboots — `last -x` shows this
   pattern predates this run, back to at least June 22 — so plan for autosentry
   plus the cron watchdog (`scratch/ensure_autosentry.sh`, now fixed to use an
   absolute path per commit `cb331e4`) to actually be exercised, not just present.

---

## AI use statement

This document, and the diagnosis of the four incidents above, were compiled with
AI assistance via Claude Code, under the direction of Evan Owen, from
autosentry logs, `journalctl`, kernel Xid logs, and git history that had already
been independently re-derived and verified by the operator before this writeup;
this document's job was to verify every file path, commit hash, and flag cited
against current repo state (see the correction to the `17ae22a`/`a0782b0`
attribution above) and present the findings in the style of the two campaign
documents it accompanies.
