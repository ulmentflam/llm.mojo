# ZeRO-3 parameter streaming (2026-07-14)

Follow-up to [`zero_stage23_memory_optimizations_2026-07-14.md`](zero_stage23_memory_optimizations_2026-07-14.md),
which landed the in-place gradient reduce-scatter and left **per-layer
parameter streaming** as the remaining ZeRO-3 memory win. This campaign
implements that streaming (Track B; a parallel Track A owns bucketed backward
*gradient* reduction — untouched here).

## Headline (read this first)

- **Landed and verified:** ZeRO-3 now holds parameters as **shard-only at
  rest** and gathers each transformer layer's tensors **just-in-time** into a
  small window before that layer's forward/backward, plus the embedding/head
  tensors once per forward. Resident parameter *allocations* drop from the full
  padded buffer (~498 MiB fp32) to **shard 124.5 + one-layer window 28.4 +
  embedding/head window 157.7 ≈ 310 MiB** per rank. Numerics are unchanged:
  the streamed run's per-step losses are **bit-identical to the coarse-gather
  ZeRO-3 baseline** for the first three steps and match a ZeRO-0 run to ~2e-4
  over 10 steps (GPU fp reduction-order noise, same grade the earlier ZeRO
  campaigns report). The CPU equivalence suite stays 6/6.
- **Measured peak did NOT move at W4 b4t64** (2498 MiB/GPU streamed vs 2496
  baseline). The reason is characterised precisely below and is **not** a bug:
  the runtime's device allocator commits memory in ~256 MiB chunks, and the
  157.7 MiB **weight-tied `wte`** window keeps params-side resident at ~310 MiB
  — inside the same chunk as the 498 MiB baseline. A controlled window-size
  sweep shows the peak *does* fall by a full 256 MiB the moment params-side
  drops below ~256 MiB (i.e. once the `wte` residency is removed). So the
  streaming mechanism is sound; the remaining blocker at this world size is
  solely the resident tied embedding, whose removal needs a vocab-tiled LM head
  + indexed encoder gather (kernel work, scoped as the follow-up).
- **Step time cost is real and measured:** streamed ZeRO-3 runs at ~87 ms/step
  vs ~49–61 ms/step for coarse ZeRO-3 (≈ +40–75 %), from ~26 range-gather
  collectives per step (12 layers × forward+backward + the embedding gather).
  This is the expected streaming trade; it pays off where the memory win is
  visible (see below).

## Design

Parameters/gradients are **tensor-major**: `ParameterTensors.point_parameters`
lays the 16 tensors back-to-back, each spanning all layers
(`[wte][wpe][ln1w·L]…[lnf]`), so a single layer's params are scattered across
12 far-apart tensor blocks and the reduce-scatter shard is a flat cut across
tensor boundaries. Streaming therefore gathers **per (layer, tensor) flat
ranges**, not contiguous model regions.

New collective `ZeroContext.allgather_ranges(shard_base, shard_size, dst_base,
dst_offsets, flat_starts, lengths)` (`llmm/zero.mojo`): every rank owns
full-vector indices `[r*shard, (r+1)*shard)` at local offset 0 of its shard
buffer; each rank reconstructs a set of arbitrary flat sub-ranges into its own
window by copying, for each range and each rank p, the intersection with p's
shard (a same-device copy for its own piece, a driver-staged cross-device pull
for peers — the same no-P2P staging the other collectives use). One barrier
pair amortises a whole layer's 12 ranges. No shard is written, so concurrent
peer reads are safe. CPU (threaded) and NVIDIA staged-copy paths both
implemented.

`GPT2` changes (all gated on `z3_streaming`, set true only when
`zero_stage >= 3`, `WORLD_SIZE > 1`, **and** a real coordinator is attached):

- **Shard-only params.** `_z3_finalize_params` (shared tail of all three
  allocation paths) allocates `params_buf` at **shard size** and seeds it with
  this rank's slice from the full host buffer. The full parameter buffer is
  **never made resident on the device** — this matters because the caching
  allocator's high-water mark does not shrink if you allocate full then free
  (an earlier compaction-after-load version freed the full buffer and saw *no*
  change; allocating small from the start is what actually lowers the arena).
- **Per-layer gather.** At the top of each forward/backward layer loop,
  `_z3_stream_layer(l)` gathers layer `l`'s 12 tensor slices into
  `param_window_buf` and re-points the 12 `self.params.*` bases by
  `window_offset − l*stride`, so the existing `self.params.<t> + l*stride`
  addressing in the kernels lands on the window with **no call-site changes**.
- **Embedding/head gather.** `_z3_stream_embed` gathers `wte, wpe, ln_f_γ,
  ln_f_β` into `embed_window_buf` once per forward and re-points those four
  bases. They bracket the step (encoder at the start; LM head + final
  layernorm at the end; LM-head backward at the start of backward) and params
  don't change until `update()`, so the window stays valid across the whole
  forward+backward.
- **Optimizer / master / checkpoint.** The optimizer reads the shard from
  `params_memory + 0` (offset 0, since the buffer *is* the shard) instead of
  `+ rank*opt`; the bf16 master seed reads offset 0 too. `write_checkpoint`
  reconstructs the full vector into a temporary buffer via
  `_z3_gather_full` (rank 0 only writes); `load_checkpoint` copies just this
  rank's shard slice back. Gradients stay a full buffer (Track A's domain).

Non-streaming ZeRO-3 (the CPU equivalence harness passes no coordinator) is
untouched: params stay fully resident and the old coarse all-gather runs as a
no-op, so that path — and the existing equivalence proof — is unchanged.

## Why the measured peak is flat at W4 (fully characterised)

Method: run 40 steps, sample `nvidia-smi` used-MiB on GPU 0 every 0.3 s, take
the max; GPUs 0–3 idle otherwise. Calibrated against the known curve —
**z0 = 3266, z1 = 2498 MiB** — reproducing the published table exactly, so the
sampler *does* track allocation changes (the 770 MiB z0→z1 Adam-shard drop
shows).

Controlled sweep of resident params-side (shard 124.5 + windows), fp32 W4
b4t64, everything else fixed:

| params-side resident (MiB) | what | peak GPU0 (MiB) |
|---|---|---|
| 498 (full buf, no windows) | coarse ZeRO-3 baseline | 2498 |
| 683 (full buf 498 + windows 185) | forced-full probe | 2754 |
| **310 (shard 124 + windows 186)** | **shipped streaming** | **2498** |
| 230 (shard 124 + layer 28 + ½·wte 78) | probe | **2242** |
| 153 (shard 124 + layer 28, no wte) | probe | **2242** |

The peaks land on a **~256 MiB grid** (2242 / 2498 / 2754): the allocator
commits device memory in ~256 MiB chunks. The shipped 310 MiB and the 498 MiB
baseline sit in the **same chunk** (→ 2498), so the real 188 MiB allocation
saving is invisible. Dropping params-side below ~256 MiB — which requires the
157.7 MiB `wte` window gone (or ≤ ~78 MiB) — moves down one chunk to **2242
(−256 MiB)**.

`wte` is `padded_vocab_size × C` = 38.6 M elems = 157.7 MiB and is **weight-tied
to the LM head**, so it must be a contiguous `[V,C]` tensor for the logits
matmul; the encoder also indexes arbitrary rows of it. Holding it resident is
what pins params-side in the upper chunk at W4. At **W8** the shard halves to
62 MiB, so params-side = 62 + 157.7 + 28.4 = **248 MiB < 256** → the *same code*
crosses the boundary and shows the ~256 MiB win — but W8 needs all 8 GPUs
(Track A owns 4–7) so it was not measured here.

Honest bottom line: the structural streaming is implemented and numerically
exact, and the sweep proves it lowers peak by a full chunk once `wte` residency
is addressed; at W4 b4t64 specifically the shipped configuration does not yet
cross the allocator chunk boundary.

## Follow-up to capture the W4 win

Remove the resident `wte` window (target params-side ≤ ~230 MiB → 2242):

1. **Vocab-tile the LM head** (`matmul_fwd`/`matmul_bwd` for logits): loop over
   ~2 vocab tiles, gathering each `wte` slice (~78 MiB) into a reused window and
   writing the matching logit columns / accumulating `d_ln_f`. Straightforward.
2. **Indexed encoder gather**: gather only the batch's token rows
   (`allgather_ranges` already supports per-row ranges — ≤ B·T·C ≈ 0.8 MiB) and
   feed the encoder a compact `[B·T,C]` buffer with remapped indices. This
   touches the encoder's forward path and its wte-grad bucket machinery, so it
   needs its own equivalence check (interacts with weight tying and Track A's
   grad buffer) — kept out of this surgical diff.

## Verification (actually run on this box, GPUs 0–3)

- **CPU equivalence** `tests/test_zero_equivalence.mojo`: **6/6** (stages 1/2/3
  at W2 and W8; streaming is off with no coordinator, so this proves the
  optimizer-sharding math is unchanged).
- **New collective unit test** `tests/test_zero.mojo::test_multi_cpu_allgather_ranges`:
  4-rank threaded CPU gather of two boundary-straddling flat ranges,
  reconstructing the exact full-vector values on every rank.
- **Numerics** fp32 W4 b4t64: streamed ZeRO-3 step 1–3 losses bit-identical to
  the coarse-gather ZeRO-3 baseline; streamed-z3 vs z0 over 10 steps agree to
  ~2e-4 (step 1 identical; step 10 4.217648 vs 4.217410; val 3.98718 vs
  3.98707).
- **Memory / step time** measured with the sampler above (tables in this doc).

## Gate

`make format` (clean), `make lint` (ruff / mojo format --check / pyrefly, exit
0), `make build` / `build-bf16` / `build-profile` at WORLD_SIZE=4 (the compile
legs of `make check`), CPU equivalence (6/6) + the new collective test as the
numerics gate. Note: a full `mojo run tests/test_zero.mojo` pass crossed the
600 s per-file budget on the first attempt on this box (compile-dominated, the
same wall-time fluctuation the multi-GPU rewrite doc records for this file);
it was relaunched with a 1500 s window — see HANDOFF.

## HANDOFF (for the coordinator)

**Done and verified on this box (GPUs 0–3, W4 fp32 b4t64):**

- Streaming implementation complete: `allgather_ranges` collective
  (`llmm/zero.mojo`), shard-only `params_buf`, per-layer + embed/head JIT
  gather windows, optimizer/master/checkpoint offset handling
  (`train_gpt2.mojo`), new 4-rank CPU unit test (`tests/test_zero.mojo`).
- Numerics: streamed z3 losses bit-identical to coarse z3 (steps 1–3), z3 vs
  z0 ~2e-4 over 10 steps (fp reduction-order grade), equivalence suite 6/6.
- Memory: peak flat at W4 (2498 both) because of the ~256 MiB allocator chunk
  + resident 157.7 MiB tied `wte` window; probes (table above) prove a
  −256 MiB drop (2242) once params-side < ~256 MiB. bf16 W4 builds but was
  not benchmarked.
- Step time: streamed z3 ~87 ms vs coarse z3 ~49–61 ms vs z0 ~51 ms.

**Remaining (exact next steps):**

1. Confirm the relaunched `tests/test_zero.mojo` run (12 tests incl. the new
   `test_multi_cpu_allgather_ranges` and the N=2 GPU collective test) exits 0
   — it was still inside its 1500 s window at handoff; log at the scratchpad's
   `test_zero2.log`. The first attempt died only on the 600 s timeout
   (rc=124), with zero test failures printed.
2. To surface the W4 win: vocab-tiled LM head + indexed encoder gather (design
   in "Follow-up" above) so the `wte` window is not resident. Alternatively
   measure at W8 (shard halves → params-side 248 MiB < 256) once Track A frees
   GPUs 4–7.
3. Merge with Track A: this diff touches `forward`/`backward` only at the
   layer-loop heads and the post-backward reduce block is untouched; grads
   remain a full buffer, so the in-place reduce-scatter path is exactly as
   Track A found it. `_z3_finalize_params` replaced the identical tails of the
   three `allocate_parameters*` variants — watch for conflicts if Track A
   touched those.

**Gotchas:** (a) never allocate the full param buffer then free — the caching
allocator keeps the high-water mark; allocate shard-sized from the start.
(b) `z3_streaming` must stay false without a coordinator or the CPU
equivalence harness (sequential rank simulation) breaks. (c) The negative
rebase offsets in `_z3_stream_layer` are pure address arithmetic — do not
"simplify" them to window-base pointers without also changing every
`+ layer*stride` call site.

## AI use statement

Written with AI assistance (Claude Opus agent via Claude Code), directed by
Evan.
