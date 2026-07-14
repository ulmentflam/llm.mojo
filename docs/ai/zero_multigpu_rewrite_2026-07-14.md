# ZeRO multi-GPU rewrite: true 8-device data parallelism (2026-07-14)

Follow-up to [`zero_world8_verification_2026-07-14.md`](zero_world8_verification_2026-07-14.md),
which root-caused why every ZeRO stage crashed on the GPU target and scoped
the fix. This campaign implements that fix. Headline:

- **fp32 `-z 0/1/2/3` all train at WORLD_SIZE=8**, per-step losses matching
  across stages to ~1e-4 (step 1 bit-identical). This is the first working
  multi-GPU training in this repo.
- **bf16 smokes pass on all four stages** at WORLD_SIZE=8.
- The GPUs on this box have **no CUDA P2P** (details below), so the
  collectives are hand-rolled as reduce-scatter + all-gather over
  driver-staged cross-device copies rather than Modular's `comm` kernels.
- Six additional real bugs/design gaps were found and fixed on the way
  (micro-step reduction, grad-norm clipping consistency, bf16 master shard
  seeding, loss averaging, stage-3 generation deadlock, grad-norm scale).
- During the final benchmark sweep, **physical GPU 1 hard-faulted** ("GPU
  requires reset") — a hardware/driver incident that blocks re-measuring at
  W8 until a root reset; all W8 verification runs had already completed and
  a full measured curve exists at W4.

## What was wrong (recap) and what the probes established

The old code ran ONE model on device 0 with `rank` pinned to 0 and called
`comm.allreduce` with a single `DeviceContext` plus N `TileTensor`s aliasing
one device-0 buffer. Fetching Modular's kernel sources and probing on this
box established:

1. `comm`'s convention is **one call per GPU, concurrently**, inputs living
   on N different devices, output = the calling rank's buffer, rank =
   `ctx.id()`; signal buffers need `size_of(Signal) + payload` for the
   2-stage path.
2. **This box has no CUDA P2P**: `DeviceContext.can_access(0,1) == False`
   and `enable_all_peer_access()` raises "hardware does not support P2P
   access". (`nvidia-smi topo -p2p r` prints "OK" and PyTorch's
   `can_device_access_peer` returns True — those report a different,
   weaker capability than what the Mojo stdlib/`comm` signal protocol
   needs; the GPUs are PCIe PHB peers with no NVLink.) Consequently
   `comm.reducescatter`/`allgather` raise ("requires P2P access"), and
   `comm.allreduce` falls into a naive fallback that dies in a
   cross-device raw-pointer `enqueue_copy` — the very
   `CUDA_ERROR_INVALID_VALUE` (device_context.mojo:6625) from the original
   crash. An isolated probe reproduced that exact failure with a plain
   `ctx.enqueue_copy(dst_ptr, src_ptr)` across devices.
3. What DOES work without P2P: **DeviceBuffer-level copies**. Non-owning
   views — `DeviceBuffer[T](ctx_handle, ptr, n, owning=False)` with a fresh
   `DeviceContext(device_id=peer)` handle — passed to
   `ctx.enqueue_copy(dst_buf=..., src_buf=...)` copy across devices
   (driver-staged), including slice views. Probe-verified, then scaled to
   a full 8-rank staged reduce-scatter + all-gather at the exact GPT-2
   size (124475904 fp32): **correct on all ranks, ~92 ms allreduce-
   equivalent, ~75 GB/s aggregate**.

## The design

One process, one host thread per rank (mirroring the CPU multi-rank
dispatch), rank r owning `DeviceContext(device_id=r)` and a full model
replica. The existing `CpuCoordinator` (host barriers + pointer-exchange
slots, plus new per-rank Float64 scalar slots) is shared by the rank
threads; for GPU ranks the exchanged pointers are *device* addresses.

`ZeroContext` GPU collectives (N >= 2), in `llmm/zero.mojo`:

- **reducescatter**: rank r seeds its shard output with its own slice
  (same-device copy), then for each peer p pulls p's slice
  `[r*shard, (r+1)*shard)` into a scratch shard buffer and accumulates via
  a small fp32-accumulate add kernel. No input buffer is ever written, so
  concurrent peer reads are safe.
- **allgather**: rank r pulls each peer p's slice `[p*shard, (p+1)*shard)`
  into its own full buffer. r never writes its own slice r (that is
  exactly what peers read) — concurrent pulls are safe.
- **allreduce** = in-place reduce-scatter into my own slice + all-gather,
  with a coordinator barrier between phases. In-place is safe by
  slice-disjointness (phase 1 writes only slice r of r's buffer; peers
  read only slice p).
- `ensure_comm_setup(max_shard_bytes)` (called from
  `_compute_param_sizes` once shard sizes are known) sizes the scratch
  shard and builds peer `DeviceContext` handles.
- `allreduce_scalar(v)` sums a host Float64 across ranks through the
  coordinator (used for loss averaging and the sharded grad-norm).

Per-rank traffic is `2*(N-1)/N` of the buffer per allreduce — ring-
equivalent — vs `N*(N-1)` total for comm's naive all-pull fallback.

`padded_num_parameters` is now shard-aligned (`WORLD_SIZE *
round_up(ceil(P/W), simd_width)`) for **every** stage including 0, because
the staged allreduce also slices the gradient buffer into equal SIMD-aligned
shards. The padding tail is zero-filled and never written by backward.

## Additional bugs fixed en route (each bites even with perfect collectives)

1. **Gradient reduction ran every micro-step.** With grad accumulation the
   stage-0/1 allreduce is in place, so the second micro-step would
   accumulate local grads on top of already-summed global grads and re-sum
   — multiply-counting earlier contributions. Now reduced only on the last
   micro-step (llm.c semantics). Verified with `-d 4096` (accum=2).
2. **Grad-norm/clipping inconsistency (stages 2/3).** `calculate_grad_norm`
   reduced over the full `grads_memory`, which after a reduce-scatter holds
   the *local unreduced* gradient — per-rank-different and meaningless for
   clipping. Now stages 2/3 take the sumsq over the rank's REDUCED shard
   and combine across ranks via `allreduce_scalar`, giving bit-comparable
   clipping decisions to stages 0/1.
3. **Grad-norm scale vs llm.c.** Post-reduce grads hold the SUM over ranks
   while the optimizer applies `grad_scale / WORLD_SIZE` (the mean);
   returning `||sum||` made the clip threshold engage WORLD_SIZE times
   earlier than single-GPU/llm.c. The norm is now divided by WORLD_SIZE, so
   clipping rescales the effective mean gradient to unit norm exactly like
   llm.c.
4. **bf16 master weights seeded from the wrong shard.** The fp32 master
   copy (shard-sized) was seeded from `params_memory[0:n]` on every rank —
   every rank > 0 would train its shard from rank-0's weights. Now seeded
   from `rank * optimizer_num_parameters`.
5. **Stage-3 generation deadlock.** Generation ran on rank 0 only, but at
   stage >= 3 every `forward()` begins with a param all-gather whose
   coordinator barriers need all ranks. Generation is triggered on the
   *last step* of every run (`... or last_step`), so any stage-3 run that
   sampled would hang. All ranks now run the generation forwards (like
   llm.c); only rank 0 decodes/prints.
6. **Losses were per-rank.** Train and val losses are now averaged across
   ranks via the coordinator (llm.c parity), so the printed loss covers the
   whole global batch.

## Verification (all actually run on this box)

fp32, WORLD_SIZE=8 (`-e gpt2_124M.bin -b 4 -t 64 -x 10 -pn 8`, tiny
Shakespeare), per-step train loss:

| step | z0 | z1 | z2 | z3 |
|------|----|----|----|----|
| 1 | 5.092038 | 5.092038 | 5.092038 | 5.092038 |
| 5 | 4.340118 | 4.340195 | 4.340106 | 4.340180 |
| 10 | 3.671936 | 3.672031 | 3.671990 | 3.671965 |
| val@10 | 3.9067504 | 3.9067586 | 3.906742 | 3.9067216 |

Step time ~103-106 ms for z0/z2/z3, ~149 ms for z1 (grad allreduce + full
param all-gather each step). ZeRO is an optimizer-sharding *equivalence*
and the numbers agree to ~1e-4 across stages (fp reduction-order noise);
step 1 is identical because clipping/Adam state have not yet diverged.

bf16, WORLD_SIZE=8, same flags (`gpt2_124M_bf16.bin`): step-1 loss
5.084249 on all four stages; step-10 3.676855 (z0/z1, bit-identical pair)
vs 3.674875 (z2/z3, bit-identical pair) — the allreduce-based and
reduce-scatter-based paths round bf16 sums differently; val ~3.919 both.
~80 ms/step (z1 ~103 ms).

Grad accumulation: `-d 4096` (accum=2) at z1/W8 trains (25.7k tok/s).

Tests: `tests/test_zero_equivalence.mojo` 6/6 (W2+W8 CPU equivalence,
unchanged semantics); `tests/test_zero.mojo` 12/12 including three NEW
multi-GPU tests (N=2) driving the staged allreduce / reducescatter /
allgather end to end.

## Benchmark data

`bench_zero_world4.json` (complete, measured post-rewrite on 4 healthy
GPUs; B=4 T=64, 12 steps):

| Precision | Stage | mean ms/step | tok/s | peak MiB/GPU (delta) |
|-----------|-------|--------------|-------|----------------------|
| fp32 | 0 | 51.1 | 20.0k | 3266 |
| fp32 | 1 | 68.9 | 14.9k | 2498 |
| fp32 | 2 | 49.9 | 20.5k | 2754 |
| fp32 | 3 | 50.4 | 20.3k | 2754 |
| bf16 | 0 | 54.6 | 18.8k | 3010 |
| bf16 | 1 | 62.0 | 16.5k | 1986 |
| bf16 | 2 | 52.6 | 19.5k | 1986 |
| bf16 | 3 | 53.0 | 19.3k | 1986 |

Reading the curve honestly:

- The big saving is stage 0 -> 1 (~770 MiB fp32): Adam m/v shrink from
  2 x 498 MiB to 2 x 124 MiB per rank. That matches theory.
- Stages 2/3 do NOT save more memory in this implementation: the full
  gradient buffer must exist during backward anyway (grads are produced
  densely, then reduce-scattered), and stage 3 re-gathers the full
  parameter buffer before forward. True ZeRO-2/3 *memory* wins require
  bucketed grad reduction during backward and per-layer param streaming —
  both explicitly out of scope here (the forward all-gather is
  coarse-grained by design, see the comment in `forward()`). What stages
  2/3 do deliver is the equivalence + the comm pattern, at z0-level speed.
- z1 is the slowest stage: it pays a full-size grad allreduce AND a
  full-size param all-gather; z2/z3 replace the allreduce with a
  reduce-scatter and end up ring-equivalent overall.

`bench_zero_world8.json` (partial): fp32 stage 0 measured at 105.1
ms/step, 19.5k tok/s, ~3278 MiB/GPU across all 8 GPUs — then the sweep was
cut short by the GPU fault below. Per-step timings for the other W8
stages are in the verification table above (from the 10-step runs).

## Hardware incident: GPU 1 requires reset

Between the first and second runs of the W8 benchmark sweep (~07:00 local),
physical GPU 1 dropped into a driver fault state: `nvidia-smi` shows
`ERR!`/`[GPU requires reset]` for it, CUDA exposes only 7 usable ordinals
(`torch.ones` on ordinal 7 fails "invalid device ordinal"), and any
`-pn 8` run now fails with `CUDA_ERROR_INVALID_DEVICE`. Kernel Xid logs
are not readable without root, so the trigger cannot be pinned; note that
~15 heavy 8-rank runs (including all verification tables above) completed
on the same GPU earlier in the day. Recovery needs root:
`sudo nvidia-smi -r -i 1` (or reboot); after that, re-measure with
`make benchmark-zero BENCH_ZERO_WORLD=8`.

## Known limitations / follow-ups

- ZeRO-2/3 provide equivalence and comm sharding, not yet their full
  memory savings (bucketed backward reduction, per-layer param streaming).
- `ShardedParameter.gather`'s multi-GPU path still uses the P2P-only
  `comm.allgather` and is unusable on this box (it is exercised only by
  single-rank/CPU tests; training does not call it).
- Multi-rank checkpoint writing was already rank-aware
  (`write_checkpoint`: rank 0 writes the model file, every rank its own
  optimizer-state shard) — untested at W8 in this campaign.
- The staged collectives synchronize via host barriers + per-rank stream
  syncs; overlap (double-buffered scratch, copy/add pipelining) is an easy
  future win if comm time starts to matter at larger models.

## Commits

- `d28fb75` zero: true N-device GPU data parallelism via staged-copy collectives
- `5db51e7` tests+bench: GPU multi-rank collective tests; measured ZeRO benchmark data
- (this doc)

## AI use statement

Written with AI assistance (Claude (Opus agent via Claude Code)), directed
by Evan.
