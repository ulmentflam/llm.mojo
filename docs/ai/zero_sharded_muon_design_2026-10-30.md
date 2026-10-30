# Sharded Muon under ZeRO-1/2/3: design + test plan (2026-10-30)

Phase 3 of [`muon_muonclip_implementation_plan.md`](muon_muonclip_implementation_plan.md).
Muon is on `main` (`2e1913d`) for WORLD_SIZE=1 and ZeRO-0 only:
`GPT2.enable_muon` raises `"Muon phase 1 supports WORLD_SIZE=1 / ZeRO-0 only"`
when `zero_stage >= 1` or `z3_streaming` (train_gpt2.mojo:`enable_muon`).
This document designs Muon for ZeRO stages 1, 2 and 3 (streaming included). It
is written so a coding agent can implement it in slices (section 9) without
re-deriving anything. Nothing here has been implemented yet.

All line references are to `fu/zdesign` at `2e1913d`. File:function citations
come from reading the code. Facts I could not confirm by reading or running are
marked **UNVERIFIED**.

## Status

- **DESIGN ONLY.** No code changed.
- **Run on this box (fp32 CPU build, `make build WORLD_SIZE=2`, binary mtime
  checked after the build):**
  - `make train-cpu WORLD_SIZE=2 ARGS="-e d6 -i <tinyshakespeare> -b 1 -t 64 -x 3 -z {0,3} -pn 2 -v 0 -s 0"`
    and `-x 2 -z {1,2}` all run, at about 0.25 to 0.4 s/step. Steps 1 and 2
    print identical losses at every stage (10.865850, 10.369871), and step 3
    matches for `-z 0` and `-z 3` (10.128281). So multi-rank CPU training,
    including ZeRO-3 streaming (a coordinator is present, so
    `_z3_want_streaming` is true), works here end to end. It is the
    integration gate for every slice.
  - `-z 1 -opt muon` at WORLD_SIZE=2 fails on both ranks with the phase-1
    error, as expected.
- **Cannot run here:** any GPU collective at N >= 2. `ensure_comm_setup` needs
  a coordinator and builds `DeviceContext(device_id=i)` for every rank
  (llmm/zero.mojo:`ensure_comm_setup`), and this box has one GPU. See section 8.

### Review notes, generation 1 (2026-10-30, independent review, PARTIAL: interrupted by a reboot; finished by generation 2 below)

Reviewed against the code at `7d68ac1`. Items 7 (fu/vfree) and 8 (prior-art
spot check) were not finished. Verdicts so far:

- **Ownership rule (section 2): CONFIRMED.** I recomputed it with my own model
  of `_compute_param_sizes`/`enable_muon`. At 124M, W in {2, 4, 8} gives
  max/ideal NS cost 1.000, W=5 gives 1.042, W=7 gives 1.108. Each rank's share
  of a group is the arithmetic progression `l0, l0+W, ...` (asserted for every
  rank and group). Finding 1's A_r values match. "d6" is C=384 (gpt2-30M).
  Also note that tiny L=2 at W=8 has S=1280 > opt=984, not only L=1.
- **Traffic claim (sections 1 and 6): WRONG per rank, correct in aggregate.**
  The total bytes moved equal plain ZeRO, and are slightly fewer because there
  is no padding. Per rank they are not equal, because rank r handles
  `A_r + O_r` elements instead of `opt`. At 124M W=4, rank 0 reduce-scatters
  52.35M elements (plain: 31.12M, +68%) and sends 314 MB in the gathers (plain:
  187 MB). Ranks 2 and 3 receive 206 MB (plain: 187 MB). At W=8, ranks 0 and 1
  are +68%. The slowest rank sets the collective time, so the "+0 extra" column
  in section 6 should read "+0 total; the critical-path rank carries up to
  +68%". This does not block S0 to S3, but S6 must measure it, and S7's
  ownership work should weigh `A_r + O_r`, not NS alone.
- **S0 (a) allgather_ranges no-coordinator OOB: CONFIRMED by a run.** I
  compiled a standalone repro with `ZeroContext["cpu", 4]`, rank 2, shard 16,
  range [32, 40). Today's code copies `shard_base[32:40]`, which is out of
  bounds for a 16-element shard. The correct source is `shard_base[0:8]`.
- **S0 (b) self-copy alias: CONFIRMED by reading.** With `shard_base =
  params_memory + r*opt` and `dst = params_memory`, the GPU `p == self.rank`
  arm copies from and to the same address. No current caller does this; it is
  a hazard for the new callers.
- **S0 (c) missing scratch check: CONFIRMED.** `_reducescatter_buckets_gpu`
  never calls `_check_scratch`. Today a segment is at most opt, which equals
  the scratch size, so it is a missing guard, not a live bug.
- **S0 (d) resume world-size check: CONFIRMED, but finding 4 has the
  direction backwards.** Resuming at a *larger* W (smaller opt), or with a
  ZeRO-0 state at ZeRO >= 1, reads a prefix silently. The `v` read then starts
  inside the file's `m` block, so `v` is loaded from `m` data. Resuming at a
  smaller W raises "Failed to read enough bytes". S4's exact-length check
  covers both cases.
- **Section 4 SR change: CONFIRMED necessary, with two corrections.**
  (1) `muon_apply_gpu` uses `stride_arg` both for indexing and for `pidx`, so
  it needs a new `sr_stride` kernel argument. `offset_arg` is used only for
  `pidx`, so it can carry `sr_offset`. (2) It is **not sufficient** for W=1 vs
  W>1 bit-equality on GPU, for two reasons. `muon_momentum_gpu` accumulates
  per-matrix norms with float atomics across blocks, and the block order is
  nondeterministic once a matrix has more than BLOCK*VEC = 2048 elements. The
  cuBLASLt batched GEMMs also run with a different `batch_count`. T4 is
  bit-exact only for matrices of at most 2048 elements and equal group counts.
- **Section 8 T2 bit-exactness: WRONG for ZeRO-2/3.** ZeRO-0 forms
  `wte_grad = lm_k + enc_k` locally and then allreduces. Bucketed ZeRO-2/3
  reduce-scatters the LM-head part and the encoder part separately:
  `(lm_0+lm_1) + (enc_0+enc_1)`. fp32 addition is not associative, so wte
  (an AdamW tensor) differs by ulps after step 1, and from step 2 every tensor
  differs. zero_grad_bucketing_design already states about 1e-4 vs z0/z1.
  Expected results:
  - ZeRO-1 vs ZeRO-0 (grad_scale=1, no accumulation): bit-exact.
  - ZeRO-2/3 step 1: bit-exact for all Muon matrices and every non-wte tensor.
  - ZeRO-2/3 multi-step: tolerance-based: keep the step-1 exact check and
    use rtol ~1e-4 over 3 steps.
  Grad accumulation also breaks exactness: `sum_k(g_k1 + g_k2)` vs
  `sum_k g_k1 + sum_k g_k2`.
- **T5 / trainer runs: CONFIRMED, with a caveat.** I rebuilt
  `make build-train WORLD_SIZE=2` (binary mtime 23:05, fresh) and ran d6 for
  3 steps at -z 2 and -z 0. Losses were 10.865850, 10.369871, 10.128281 at
  both stages, matching the doc. The step-1 grad norm already differs
  (14.0609 vs 14.0608, from the sharded sumsq order), and the clip is active,
  so "identical" in T5 means identical to printed precision only.
- **Section 5 data flow: CONFIRMED by reading.** This covers the
  `_update_hybrid` guard for the CPU W>1 branch, ZeRO-1 pack from allreduced
  grads with the same grad_scale as AdamW, ZeRO-2 Call A/B splits with Q/K/V
  at `pool_slot[qkv] + k*C²`, and the wte two-part reduce, which stays in
  Call A and is untouched. It also covers the `_z3_stream_layer` split: the
  window `dst_offsets` are unchanged, so the `self.params.*` re-basing is
  unchanged.
- **Items 7 and 8:** not reached by generation 1; see generation 2 below.

### Review notes, generation 2 (2026-10-30, completes the review)

Reviewed at `58dea3e`, plus fu/vfree `684de0c` and fu/zs0 `e2975e9`
(read-only). Generation 1's corrections are now folded into the body,
not just the notes above:

- **Finding 4:** direction fixed. A larger W, or ZeRO-0 state at ZeRO >= 1,
  reads silently with `v` taken from `m` bytes; a smaller W raises.
- **Sections 1 and 6:** "zero extra bytes" now means total over ranks. The
  per-rank `A_r + O_r` imbalance (+68% on rank 0 at 124M W=4) is stated, and
  S6 and S7 act on it.
- **Section 4:** the new `sr_stride_arg` kernel argument is required. The SR
  fields do not give GPU W=1/W>1 bit-equality.
- **Section 8:** T2 is exact for ZeRO-1 and for ZeRO-2/3 step 1, and rtol
  ~1e-4 over 3 steps at ZeRO-2/3. T4 uses matrices of at most 2048 elements
  with equal counts. T5 compares to printed precision. T6 checks the exact
  length in both directions.
- **Item 7 (fu/vfree): DONE.** The contract is in section 6. In short: one
  segment list, per-rank compact `v` of `A_r` elements (0 on some tiny-config
  ranks), copy-down only at ZeRO-0, `_update_hybrid` never on the sharded
  path, and a positional-argument merge hazard with fu/zs0 in
  `read_state_checkpoint`.
- **Item 8 (prior art):** the Moonlight, DeepSpeed ZeRO-1/2 and DeepSpeed
  ZeRO-3 rows are CONFIRMED against primary sources (section 1, with URLs).
  The other rows were not spot-checked.
- **S0 (fu/zs0):** conforms to the design. Two CPU-testable test gaps are
  moved to S1 (section 9).

## Critical findings (read these first)

1. **The flat shards and Muon matrices are badly misaligned.** At 124M and W=4,
   `optimizer_num_parameters` (opt) = 31,118,976. Rank 0's flat shard is
   entirely `wte` (38.6M elements), so it owns **zero** Muon elements. Ranks 2
   and 3 are about 99.9% Muon matrices (computed with the `_compute_param_sizes`
   formulas; script in section 10).

   | rank | AdamW elements in flat shard (A_r) | Muon elements in flat shard |
   |---|---:|---:|
   | 0 | 31,118,976 | 0 |
   | 1 | 8,347,008 | 22,771,968 |
   | 2 | 27,648 | 31,091,328 |
   | 3 | 47,616 | 31,071,360 |

   Any scheme that derives Muon work from the flat shards inherits this skew
   (Moonlight/DeepSpeed style: rank 0 runs no Newton-Schulz, ranks 2 and 3 each
   run about 37% of it). That is the main reason this design uses a separate,
   matrix-granular **owner space** instead.

2. **`allgather_ranges` has a no-coordinator N>1 branch that would read out
   of bounds** for any shard space it was not written for. On CPU with
   `N == 1 or not self.cpu_coordinator_ptr` it copies
   `shard_base[flat_starts[i] + j]`, which treats the local shard as the whole
   vector (llmm/zero.mojo:`allgather_ranges`). No current caller reaches it at
   N>1 (streaming requires a coordinator). The sequential equivalence harness
   (`tests/test_zero_equivalence.mojo`, `cpu_coordinator_ptr=None`) will reach
   it as soon as `update()` uses it. Fix in S0: at N>1 without a coordinator,
   copy only the intersection with `[rank*shard_size, (rank+1)*shard_size)`,
   the same rule `reducescatter_buckets` already uses in that branch.

3. **`allgather_ranges` self-copy aliases when the destination is the buffer
   that holds the shard.** ZeRO-1/2 propagation (section 5) gathers the AdamW
   ranges with `shard_base = params_memory + rank*opt` and
   `dst_base = params_memory`. The `p == self.rank` arm then copies a region
   onto itself. On CPU that is a harmless element loop. On GPU it is an
   `enqueue_copy` with `src == dst`; whether that is safe on this driver is
   **UNVERIFIED**, and a memcpy with overlapping ranges is formally undefined.
   Fix in S0: skip the self copy when the computed source pointer equals the
   destination pointer.

4. **Resume does not check world size today.** `load_checkpoint` reads
   `optimizer_num_parameters` (opt) elements per blob. `read_and_copy` raises
   only when the file is short. So a resume at a **larger** W (smaller opt),
   or a ZeRO-0 state file loaded at ZeRO >= 1, reads a prefix of the `m` blob
   without error, and because `v` is read straight after an opt-length `m`,
   `v` is filled from the file's `m` bytes. A resume at a **smaller** W
   (larger opt) raises "Failed to read enough bytes". Header slot 2
   (`num_processes`) is written but never compared
   (llmm/checkpointing.mojo:`read_state_header`,
   train_gpt2.mojo:`load_checkpoint`). S4 adds the check and an exact
   file-length check, which also changes AdamW resume behaviour.
   *(Direction corrected by review gen 2; the first version said a smaller W
   reads a silent prefix. S0 on fu/zs0 already compares `num_processes` and
   `process_rank`. That does not catch a ZeRO-0 state loaded at ZeRO >= 1 at
   the same W, so the exact length check is still needed.)*

5. **QK-clip cannot be exercised end to end on CPU.** The logit-max hook exists
   only in the GPU softmax, and `enable_qk_clip` raises on the CPU target. The
   multi-rank clip is therefore testable here only with planted `head_max`
   values (section 7).

6. **The reduce-scatter-in-place branch for ZeRO-2/3 is dead at W>1.**
   `_use_bucketing()` is `zero_stage >= 2` whenever WORLD_SIZE > 1, so
   `backward`'s `reducescatter_inplace` branch never runs at W>1. The sharded
   Muon path supports only bucketed ZeRO-2/3, and `enable_muon` should raise
   if `zero_stage >= 2 and not _use_bucketing()`, to make the assumption
   explicit.

## 1. Prior art (researched 2026-10-29 from papers and repo source)

| System | Where Newton-Schulz runs | What is communicated | Momentum | Ownership / balance | Overhead claimed | Stages | Source | Spot check |
|---|---|---|---|---|---|---|---|---|
| Moonshot "Distributed Muon" (Moonlight, Alg. 1) | On the full matrix, on every DP rank whose flat partition intersects it | reduce-scatter G (fp32) → momentum on local shard → **DP gather** of momentum-applied shard (bf16) → NS → keep local partition → all-gather params | partitioned (flat shard), applied before the gather | inherits Megatron flat partition; no balancing | comms "(1, 1.25] of Distributed AdamW"; memory "half of Distributed AdamW" (1 buffer vs 2) | ZeRO-1 | arXiv 2502.16982; github.com/MoonshotAI/Moonlight | **CONFIRMED** (gen 2; see below) |
| Megatron PoC PR #1428 (linked from Moonlight; closed) | every rank holding a slice; a matrix split across k ranks is computed k times | momentum on 1-D shard, `all_gather_into_tensor` of whole bucket (bf16), slice local range | partitioned | flat buffer | none stated | ZeRO-1 + TP/PP | github.com/NVIDIA/Megatron-LM/pull/1428 | not checked |
| Kimi K2 (MuonClip) | as Moonlight | not described | "optimizer states are distributed" | n/a | ~30 GB state/GPU | ZeRO-1 | arXiv 2507.20534. Cross-rank aggregation of the QK-clip max is **not stated (UNVERIFIED)** | not checked |
| DeepSpeed ZeRO-1/2 (`stage_1_and_2.py`, `runtime/zero/muon/`) | full gradient of each param intersecting the partition; boundary-straddlers redundantly | normal reduce; straddling Muon params switch the bucket to all-reduce | flat buffer, full size of each intersecting param (duplicated for straddlers) | contiguous partition | Muon state 4 B/param vs Adam 8 B; 31.4 vs 34.5 GiB on Qwen2.5-3B | 0/1/2 | github.com/deepspeedai/DeepSpeed (blogs/muon-optimizer) | **CONFIRMED** (gen 2; see below) |
| DeepSpeed ZeRO-3 (`stage3.py`) | each rank a round-robin share of params, once per optimizer step | grads reduce-scattered, then **all-gathered back** with momentum; update + momentum all-gathered again | partitioned at rest, gathered transiently | round-robin in param order | none stated | 3 (rejects reduce_scatter / all2all modes) | same repo | **CONFIRMED** (gen 2; see below) |
| KellerJordan/Muon `Muon` | owner only | none for grads (assumes DDP all-reduce); padded `all_gather` per block of `world_size` params | owner only | sort by size desc, round-robin | n/a | DDP | github.com/KellerJordan/Muon/blob/master/muon.py | not checked |
| modded-nanogpt NorMuon / ANVIL (current) | owner of a chunk of a stacked same-shape group, one batched NS | `reduce_scatter_tensor` of the stack straight into owners, then `all_gather_into_tensor` of updated params | owner-local chunk | same-shape stacks padded to a multiple of W | NorMuon paper: distributed NS +2.9% step vs +8.1% redundant | ZeRO-2-like | github.com/KellerJordan/modded-nanogpt (records 2025-12-11, `optim/anvil.py`); arXiv 2510.05491 | not checked |
| Megatron `LayerWiseDistributedOptimizer` | whole params per DP rank | DDP all-reduce grads, all_gatherv updated params; newer layout makes buckets shard-aligned for reduce-scatter | owner | LPT by NS cost `m·n·min(m,n)`; legacy snake order | n/a | ZeRO-1-like | NVIDIA Megatron `layer_wise_optimizer.py`; NeMo emerging-optimizers docs | not checked |

**Spot checks (review gen 2, 2026-10-30, against primary sources):**

- **Moonlight, CONFIRMED.** Source: arXiv 2502.16982 section 2.3
  (https://arxiv.org/html/2502.16982), Algorithm 1. The order is
  reduce-scatter G, momentum on the local partition, DP **gather of the
  momentum-applied partition** (`G = gather(g', dp_group)`), Newton-Schulz on
  the full matrix, keep the local slice, all-gather params. The paper says
  "Distributed Muon follows ZeRO-1" and "the communication workload of
  Distributed Muon is (1,1.25] of that of Distributed AdamW". The 1.25 is
  (fp32 reduce-scatter 4 + bf16 Muon gather 2 + fp32 param gather 4) /
  (4 + 4). Memory is "half of Distributed AdamW" (one buffer instead of two).
  Two caveats from the paper itself: the Method prose says the gather
  collects "partitioned gradients", which contradicts Algorithm 1 (this
  table follows Algorithm 1). The empirical-overhead sentence ("closer to the
  lower-bound") renders with a stray "1.44" in the HTML version. Treat the
  empirical figure as unquantified.
- **DeepSpeed ZeRO-1/2, CONFIRMED.** Source: deepspeedai/DeepSpeed at
  9148ab2, `deepspeed/runtime/zero/stage_1_and_2.py` and
  `blogs/muon-optimizer/README.md`
  (https://github.com/deepspeedai/DeepSpeed/blob/9148ab22be812b563c345c20959ac399fdc6f7a5/deepspeed/runtime/zero/stage_1_and_2.py).
  `muon_update` runs per whole (unflattened) parameter inside
  `get_flat_partition`, after reduction and before flattening into the
  partition. A Muon param whose `partition_ids` has more than one entry gets
  `copy_ranks` > 1, which forces its bucket to `("allreduce", ...)` ("Split
  Muon matrices require all-reduce"), so every rank that owns a slice runs NS
  on the full matrix. The blog states support for stages 1 and 2 (not with
  optimizer offload plus `reduce_scatter`), plus Muon 31.4 vs AdamW 34.5 GiB
  on Qwen2.5-3B at ZeRO-2. Stage 0 uses `MuonWithAuxAdam.step` on full
  `p.grad`.
- **DeepSpeed ZeRO-3, CONFIRMED.** Source: `deepspeed/runtime/zero/stage3.py`
  at the same commit. The constructor raises "Muon and reduce scatter cannot
  be used together" and the same for all2all. Once per step (after the
  overflow check; issue #8443 moved it off the micro-batch path), the
  partitioned averaged gradients are all-gathered back to full
  (`_partitioned_buffers_all_gather`). `_muon_update_sub_group` gathers
  momentum, gives each rank `round_params[rank]` from each round of
  `world_sz` params (round-robin in param order), and then all-gathers the
  updates and the momentum.

Two patterns appear. In **gather-to-all, compute redundantly** (Moonlight,
DeepSpeed ZeRO-1/2), Newton-Schulz runs on every rank that touches a matrix.
This keeps the flat partition and the state format, and costs up to 1.25x
traffic plus duplicated NS. In **owner computes** (KellerJordan,
modded-nanogpt, NorMuon, Megatron layer-wise), each matrix has one owner,
gradients reduce-scatter straight to that owner, and the params are
all-gathered. Total traffic is the same as AdamW ZeRO and NS is divided by
W. It needs a balance rule and its own state layout, and the per-rank traffic
follows the ownership, not the flat partition.

**Recommendation for llm.mojo: owner computes, with a separate matrix-granular
owner space** (sections 2 to 5). Four reasons:

1. Finding 1: in our flat layout, deriving work from the flat shards gives
   rank 0 no NS and ranks 2 and 3 about 37% each. Owner-space round-robin is
   balanced exactly at 124M (section 2).
2. The existing primitives already express it. `reducescatter_buckets` and
   `allgather_ranges` take a generic `(shard_ptr, shard_size)` and a list of
   global indices. Pointing them at an owner space of W equal shards of size S
   routes gradients to owners and params back to everyone, with **zero extra
   bytes in total** over plain ZeRO. Per rank it is not even: rank r handles
   `A_r + O_r` elements instead of opt, so the most-loaded rank carries up to
   +68% at 124M W=4 (section 6). No new collective is needed except a host
   max-reduce for QK-clip.
3. `muon_update` stays whole-matrix. A Moonlight-style split would have to cut
   `muon_update` into momentum (on shard), a collective, NS, then a partial
   apply over a sub-range of the matrix. The owner-space design needs only
   separate stochastic-rounding index fields on `MuonGroup` (section 4).
4. It is the shape modded-nanogpt and Megatron converged on, and the NorMuon
   paper measured the NS saving (+2.9% vs +8.1% step time).

What we give up compared with Moonlight: a new persistent per-rank state (owner
param, momentum, and master copies), a checkpoint payload addition (S4), and
split ZeRO-3 layer gathers (S3). One DeepSpeed lesson is already satisfied:
Muon runs once per optimizer step inside `update()`, never per micro-batch. We
have no loss scaling, so DeepSpeed's momentum stage/commit on overflow does not
apply.

## 2. Ownership: the owner space

### The rule

`self.muon_groups` keeps its current order from `enable_muon`: k = 0..5 is
Q, K, V (row blocks of the fused `[L,3C,C]` qkv weight), attn_proj, fc, proj.
Each has `count = L` and `stride` = 3C², 3C², 3C², C², 4C², 4C².

Enumerate the matrices group-major, layer-minor: `j = k*L + l`. Then:

```
owner(k, l) = (k*L + l) mod W
```

Properties (all checked by `calc.py`, section 10):

- **W=1**: owner is always 0, which is exactly today's behaviour.
- **Batching survives.** Within group k, rank r owns layers
  `l0, l0+W, l0+2W, ...` where `l0 = (r - k*L) mod W` (empty if `l0 >= L`).
  That is an arithmetic progression, so each rank's share of a group is one
  `MuonGroup` with `count_rk = ceil((L - l0)/W)` (0 if `l0 >= L`). Its flat
  stride is `W*stride_k`, so the strided-batched cuBLASLt path in
  `_muon_group_gpu` is kept.
- **Balance.** Because the counter carries across groups, group k's leftover
  matrices start where group k-1's ended. The cost model is NS FLOPs per
  iteration, `4·R²·Cmax + 2·R³` with `R = min(rows, cols)`: 6C³ for a C×C
  matrix and 18C³ for fc/proj, a 3x ratio. At 124M, W ∈ {2, 4, 8} gives a
  max/ideal cost ratio of exactly 1.000. W=5 gives 1.042, W=7 gives 1.108.
- **Some ranks own nothing in tiny tests.** For L=1, W=8: O_r =
  [256, 256, 256, 256, 1024, 1024, 0, 0]. Every code path must tolerate
  `count_rk == 0` and `O_r == 0`. Skip empty groups; a rank that owns nothing
  still joins every collective.

LPT bin-packing (Megatron) or memory-aware assignment were rejected for this
phase. Both break the arithmetic progression, which costs batching. LPT is
also no better at 124M, where round-robin is already exact. Memory-aware
ownership (keep Muon off rank 0, which already holds all of wte's AdamW state)
pays off only after the flat state is trimmed (S7), so it is deferred there.
The rule is one pure function. Its identity is recorded in the checkpoint as
`muon_ownership_version = 1` (S4), so changing it later is local.

### Owner-space layout

Rank r's owner shard is the concatenation, in (k ascending, l ascending)
order, of the matrices it owns, each stored contiguous row-major `rows*cols`:

- `O_r` = owned element count.
- `S = round_up(max_r O_r, simd_width_of[GPT2_DTYPE]())`, the same for every
  rank. Index space size is `W*S`.
- The global owner-space index of matrix (k, l) is `owner*S + local_off(k, l)`.

Build one table in `enable_muon`, identical on every rank (the collectives
require identical range lists, checked by `assert_ranges_agree`):

```
struct MuonSlot: k, l, flat_off, len, owner, owner_idx   # owner_idx = owner*S + local_off
self.muon_slots: List[MuonSlot]       # all 6L matrices, order (k, l)
self.muon_owned_groups: List[MuonGroup]  # this rank's <= 6 owner-space groups (section 4)
self.muon_shard_elems: Int            # S
```

`flat_off(k, l) = muon_groups[k].offset + l*muon_groups[k].stride`. For Q, K
and V that is `qkv_base + l*3C² + k*C²`.

### Owner-local state (new buffers, allocated in `enable_muon`)

| buffer | dtype | size | role |
|---|---|---|---|
| `muon_p_buf` / `muon_p_shard` | GPT2_DTYPE | S | **source of truth** for the owned matrices' params (all stages) |
| `muon_m_buf` / `muon_m` | fp32 | S | momentum (replaces flat `m` for Muon matrices) |
| `muon_master_buf` / `muon_master` | fp32 | S under bf16, 1 under fp32 | fp32 master (bf16 builds), as `master_buf` |
| `muon_g_buf` / `muon_grad_shard` | GPT2_DTYPE | S | reduced owned gradients (ZeRO-2/3 accumulator; ZeRO-1 pack target) |

All four are sized deterministically from (rank, W, config) and are zero-filled.

**Seeding**, in `enable_muon` after the owner table exists:

- ZeRO-1 and ZeRO-2, and non-streaming ZeRO-3 (the harness): `params_memory`
  is full and consistent. For each owned slot, copy
  `params_memory[flat_off : +len]` into `muon_p_shard[local_off]`, one
  `ctx.enqueue_copy` per slot (works on both CPU and GPU contexts, as
  `_z3_finalize_params` does).
- ZeRO-3 streaming: only the flat shard is resident. Run W rounds. In round o
  every rank calls `allgather_ranges(_z3_shard_ptr(), opt, scratch, ...)` with
  owner o's slots (dst offsets = local_off, flat_starts = flat_off). Only rank
  o copies `scratch[0:O_o]` into `muon_p_shard`. Each rank receives M elements
  in total, once at startup. The scratch is S elements, freed afterwards.
- bf16: `muon_master[i] = fp32(muon_p_shard[i])`, the same promotion
  `allocate_optimizer_moments` does for `master_buf`.

**What happens to the Muon portion of the flat `m` / `v` / `master` shards:
it is left allocated and unused (dead) in S1 to S6.** Trimming is S7, and must
be done together with fu/vfree (section 6). The cost is quantified in section 6.

## 3. Unchanged pieces (confirmed by reading)

- **AdamW tensors** keep the flat shards. Rank r updates the intersection of
  `[lo, hi) = [r*opt, min(r*opt + opt, num_parameters))` with each of the 5
  `_adamw_ranges()` pairs. Add one helper used by every stage, and by
  fu/vfree's compact `v`:

  ```
  _adamw_shard_segments() -> List[(flat_start, len, shard_rel, compact_off)]
  ```

  `shard_rel = flat_start - lo` indexes `m`, `v`, `master` and the flat grad
  shard. `compact_off` is the prefix sum over segments (fu/vfree's compact
  base). At W=1 or ZeRO-0, `[lo, hi) = [0, N)` and the segments are exactly
  `_adamw_ranges()`. Segment starts are either tensor boundaries (multiples of
  C) or `r*opt`, both of which existing GPU AdamW calls already use. The CPU
  AdamW path uses `aligned=False` loads (llmm/adamw.mojo:`adamw_update_cpu`),
  so unaligned starts are safe there.
- **Grad clipping / global norm** (`calculate_grad_norm`):
  - ZeRO-1: grads are fully allreduced; unchanged.
  - ZeRO-2/3: the norm becomes `sumsq(grad_shard[0:opt]) + sumsq(muon_grad_shard[0:S])`
    before `allreduce_scalar`. This is a two-term change, not a no-op. The flat
    shard's Muon positions stay zero because Muon buckets no longer target it
    (section 5) and `zero_gradients` fills it. Padding is zero, and the tiling
    is exact: every Muon element is reduced into exactly one owner slot, and
    every AdamW element into exactly one flat slot.
- **`update()`'s `grad_scale / WORLD_SIZE`**: the Muon config receives the
  same scaled `grad_scale` as ZeRO-0 Muon today (train_gpt2.mojo:`update`,
  `_update_hybrid`). Unchanged.
- **Embedding / LM-head buckets** (`wte`, `wpe`, `ln_f`) are all AdamW tensors.
  The encoder row-sparse path and the vocab-tiled LM head are untouched.

## 4. `llmm/muon.mojo` change: separate stochastic-rounding indexing

`muon_update` indexes params, grads, master and momentum all with
`g.offset + mat*g.stride + i`. The stochastic-rounding counter `pidx` uses the
same base: `Int(offset_arg) + mat*stride + i` in `muon_apply_gpu`, and
`base + i` in `_muon_update_cpu` (llmm/muon.mojo). In the owner space all four
buffers share one layout (offset = local_off, stride = rows*cols), but the SR
counter must stay the **global flat index**. Otherwise `-D LLMM_SR_MASTER=1`
runs stop matching W=1.

The change:

- Add `sr_offset: Int, sr_stride: Int` to `MuonGroup`.
- Add a convenience constructor `MuonGroup.flat(offset, rows, cols, count, stride)`
  that sets `sr_* = offset/stride`. Update the existing call sites:
  `enable_muon`, the `@register("muon_update")` bridge, and `tests/test_muon.mojo`.
- `muon_apply_gpu` needs a **new `sr_stride_arg` kernel argument**. Today
  `stride_arg` does two jobs: it indexes params and master
  (`mat*stride_arg + i`) and it feeds `pidx = offset_arg + mat*stride_arg +
  i`. In the owner space those two strides differ (`rows*cols` vs
  `W*stride_k`), so they cannot share an argument. `offset_arg` is used only
  in `pidx`, so it can carry `g.sr_offset` (rename it `sr_offset_arg`). The
  result is `pidx = sr_offset_arg + mat*sr_stride_arg + i`. Do not pass
  `sr_stride` as `stride_arg`.
- In `_muon_update_cpu`, pass `pidx = g.sr_offset + mat*g.sr_stride + i`
  separately from `idx = base + i`.

Rank r's owned group k is:

```
MuonGroup(offset = local_off(k, l0), rows, cols, count = count_rk,
          stride = rows*cols, sr_offset = flat_off(k, l0), sr_stride = W*stride_k)
```

The SR fields make the rounding counter match W=1. They do **not** make GPU
W>1 bit-identical to W=1, for two reasons. `muon_momentum_gpu` accumulates
per-matrix norms with float atomics across blocks, so the order is
nondeterministic once a matrix has more than BLOCK*VEC = 2048 elements. The
cuBLASLt strided-batched GEMMs also run with a different `batch_count`.
Expect bf16-noise agreement on GPU, not equality (T4, S6).

The vector-path predicate (`n`, `stride`, `offset`, `r*r` multiple of 8) holds
whenever C is a multiple of 8. The persistent scratch name already encodes its
size (`muon_x_<n>`), so a rank's smaller group set allocates less, and nothing
else changes.

## 5. Data flow per stage

All sharded stages go through one target-generic method,
`GPT2._update_sharded_muon(t, config)`. Route to it from **both** the CPU and
the GPU branches of `update()` when
`optimizer_kind == 1 and WORLD_SIZE > 1 and zero_stage >= 1`, and **skip**
those branches' own trailing flat `allgather` (it would overwrite fresh Muon
matrices with stale flat copies). Today the CPU W>1 branch calls
`_update_hybrid` whenever `optimizer_kind == 1`; that call must be guarded by
`zero_stage == 0`. Notation: `lo = rank*opt`, `seg` =
`_adamw_shard_segments()`, `slots` = `muon_slots`.

### ZeRO-1 (grads allreduced, params fully resident)

1. *(backward, unchanged)* On the last micro-step,
   `allreduce(grads_memory, padded_num_parameters)`.
2. **Pack**: for each owned slot, local `enqueue_copy`
   `grads_memory[flat_off:+len] → muon_grad_shard[local_off]` (S elements,
   local only).
3. **AdamW**: for each segment, `adamw_update(len, params_memory + flat_start,
   grads_memory + flat_start, master_memory + shard_rel, m + shard_rel,
   v + compact_off)` when fu/vfree's compact `v` is on main, or
   `v + shard_rel` while `v` is still opt-sized (the contract is in section 6).
4. **Muon**: `muon_update(muon_owned_groups, muon_p_shard, muon_grad_shard,
   muon_master, USE_BF16, muon_m, t, cfg)`.
5. *(S5)* QK-clip on owner/flat pieces (section 7).
6. `ctx.synchronize()`.
7. **Propagate AdamW**: `allgather_ranges(shard_base = params_memory + lo,
   shard_size = opt, dst = params_memory, dst_offsets = flat_starts = the 5
   _adamw_ranges() starts, lengths)`. This needs finding 3's self-alias skip.
   It is safe by disjointness: rank r writes only AdamW regions outside its own
   slice, and peers read only rank r's slice.
8. **Propagate Muon**: `allgather_ranges(shard_base = muon_p_shard,
   shard_size = S, dst = params_memory, dst_offsets = [s.flat_off],
   flat_starts = [s.owner_idx], lengths = [s.len] for s in slots)`.
9. `ctx.synchronize()`.

Steps 7 and 8 replace the existing `allgather(params_memory, opt)`. Do **not**
keep the flat allgather and add 8 on top. That costs (W-1)/W·M·2 B =
127 MB/rank/step more at 124M, W=4.

### ZeRO-2 (bucketed; params fully resident)

Backward change, in each per-layer bucket
(train_gpt2.mojo:`backward`, "Reduce-scatter this layer's 12 weight-grad
tensors as one bucket"), when `optimizer_kind == 1`:

- **Call A (flat)**: the 8 non-Muon tensors (ln_1 γ/β, qkv_bias,
  attn_proj_bias, ln_2 γ/β, fc_bias, proj_bias) go through
  `_reduce_grad_bucket(d, po, ln)` into `grad_shard` / opt, unchanged.
- **Call B (owner)**: the 6 Muon matrices of this layer, with
  `dest = owner_idx`. qkv_weight splits into Q, K and V at pool offsets
  `pool_slot[qkv] + k*C²`. Call
  `zero_ctx.reducescatter_buckets(pool, [owner_idx...], [pool_off...], [len...], muon_grad_shard, S)`.
  Every rank passes identical lists, because they come from the static table.
  `reducescatter_buckets` already keeps the elements with
  `f // shard_size == rank` and `+=` accumulates across micro-steps.
- `zero_gradients`: also `muon_g_buf.enqueue_fill(0)` whenever it fills
  `grad_shard_buf`.

Update:

1. AdamW per segment with `grads = grad_shard + shard_rel` and
   `params = params_memory + flat_start`.
2. Muon: same call as ZeRO-1 step 4. There is no pack, because
   `muon_grad_shard` is already the reduced accumulator.
3. Propagate: ZeRO-1 steps 7 to 9.

### ZeRO-3 streaming (the only ZeRO-3 mode with a coordinator)

Params at rest are the flat shard (`params_memory[0:opt]`) plus `muon_p_shard`.
After the first update, the flat shard's Muon positions are stale and never
read.

- **Grads**: same as ZeRO-2.
- **Update**: AdamW per segment with `params = params_memory + shard_rel`;
  Muon as above; **no propagation** (ZeRO-3 semantics).
- **`_z3_stream_layer`**: split the single `allgather_ranges` into two:
  - (a) the 8 non-Muon tensors from `(_z3_shard_ptr(), opt)`;
  - (b) the 6 matrices from `(muon_p_shard, S)`, with `flat_starts = owner_idx`
    and `dst_offsets` = the same window slots as today. Q, K and V land at
    `dst_offsets[qkv] + k*C²`.

  Both run on every rank with identical lists. The re-basing of
  `self.params.*` is unchanged. This applies to every caller: forward's layer
  loop and backward's re-stream (train_gpt2.mojo:`backward`, the
  `if self.z3_streaming: self._z3_stream_layer(layer)` site).
- **`_z3_stream_embed`, `_lm_head_wte_tile`**: unchanged (AdamW tensors only).
- **`_z3_gather_full`** (checkpoint writer): split the same way. Non-Muon
  tensors come from the flat shard; the 6L matrices come from the owner space.
- **Seeding**: the W-round gather in section 2.

### ZeRO-3 non-streaming (sequential harness only: no coordinator)

`params_memory` is full. AdamW uses `params = params_memory + flat_start`
(today's `p_off = offset`). Muon writes `muon_p_shard`. There is no
propagation: forward's `allgather` is a no-op without a coordinator. The
harness checks `muon_p_shard` directly (section 8).

### Order constraints that are easy to get wrong

- Steps 7 and 8 must both come **after** every local write: AdamW, Muon, and
  the S5 clip. The clip must therefore move **inside** `update()` for W>1
  (section 7). Today the trainer calls `qk_clip_step()` after `update()` has
  already all-gathered.
- At W>1, `update()` must never call `_update_hybrid` for `zero_stage >= 1`.
  It addresses params, grads, `m_memory` and `master_memory` at full-buffer
  flat offsets (it assumes no sharding), which overruns opt-sized buffers.
  Under fu/vfree it also indexes `v` by range number k (`v_compact_bases[k]`),
  not by shard segment. Do not reuse or generalize it for the sharded path.
  `_update_sharded_muon` gets its own loop over `_adamw_shard_segments()`.
- `ensure_comm_setup(max(comm_shard, S) * size_of[GPT2_DTYPE]())` in
  `enable_muon`. The GPU `_reducescatter_buckets_gpu` stages each
  bucket/shard overlap (up to one matrix, at most S) through `comm_scratch`
  sized for opt. With tiny configs S can exceed opt (L=1, W=8: S=1024 >
  opt=576). Add a `_check_scratch` call inside `_reducescatter_buckets_gpu`
  per segment, which it lacks today (S0).

## 6. Memory and traffic at 124M, W=4, bf16

Per-rank optimizer state in MB. fp32 m, v and master take 12 B per flat
element; the owner space takes 10 B per element (m 4 + master 4 + param 2).

| rank | AdamW ZeRO today | Muon ZeRO-0 today (only multi-rank option) | sharded Muon S1-S6 (flat state untrimmed) | + fu/vfree compact v | S7 fully trimmed |
|---|---:|---:|---:|---:|---:|
| 0 | 373.4 | 1493.7 | 585.8 | 585.8 | 585.8 |
| 1 | 373.4 | 1493.7 | 585.8 | 494.7 | 312.5 |
| 2 | 373.4 | 1493.7 | 585.8 | 461.4 | 212.7 |
| 3 | 373.4 | 1493.7 | 585.8 | 461.5 | 212.9 |

How to read the table:

- **Peak is rank 0, whatever is trimmed**, because its flat shard is all wte
  (finding 1). Bringing the peak to AdamW's 373 MB needs both S7 trimming and
  memory-aware ownership (no Muon on rank 0). That would put NS on W-1 ranks,
  4/3 of the balanced time.
- Compared with today's only multi-rank Muon (ZeRO-0), sharding cuts state
  from 1494 to 586 MB per rank.
- The other new buffers:
  - `muon_grad_shard`: S·2 B = 42.5 MB.
  - ZeRO-3 streaming params grow by S·2 B = 42.5 MB, because dead Muon
    positions remain in the flat param shard (22.8M, 31.1M, 31.1M elements on
    ranks 1 to 3).

**fu/vfree composition (contract for S1; checked against fu/vfree at
`684de0c` by review gen 2).** What fu/vfree does (commits b43bfe5 and
684de0c), at ZeRO-0 only:

- `enable_muon` drops the constructor's full-size `v` and allocates a compact
  one. `v_compact_bases[k]` is the prefix sum over the 5 `_adamw_ranges()`
  pairs, one entry per range k. `v_num_parameters` is the total.
- `_update_hybrid` indexes `v` at `v_compact_bases[k]`, and params, grads,
  `m` and `master` at the full-buffer `start`.
- `_v_count()` returns `v_num_parameters` under Muon and opt otherwise.
  `write_state_checkpoint` and `read_state_checkpoint` take an optional
  trailing `v_num_parameters` (-1 means the same as m).
- `load_checkpoint` peeks the header and refuses a different
  `optimizer_kind`. It then requires `getsize == 256*4 + (opt + _v_count())*4`
  for every optimizer. The one exception is a Muon file of exactly
  `256*4 + 2*opt*4` (written before v went compact), which it reads at full
  size and copies down in place, range by range, in ascending order
  (`bases[k] <= start_k`).

How S1 must adapt it (if fu/vfree is on main when S1 starts; otherwise S1
keeps `v` opt-sized and indexed by `shard_rel`, and whichever branch merges
second applies this):

1. **One segment list.** `_adamw_shard_segments()` is the only source of
   AdamW ranges for the update, the size of `v`, `_v_count()`, the
   state-file length and the copy-down. Entries are
   `(flat_start, len, shard_rel, compact_off)`: nonempty intersections of
   `[lo, hi)` with the 5 ranges, in ascending order, with
   `compact_off = sum of the earlier lens`. At W=1 or ZeRO-0 they are the 5
   `_adamw_ranges()` (all nonempty for L >= 1), and `compact_off` equals
   fu/vfree's `v_compact_bases[k]`.
2. **Retire `v_compact_bases`**, or derive it from the segments. Indexing by
   range number k is wrong at ZeRO >= 1. A rank's segment i is not range i:
   empty intersections are dropped, and the first segment can start
   mid-range at `lo`.
3. **Per-rank compact `v`.** In `enable_muon`, set
   `v_num_parameters = sum(seg.len) = A_r`. At 124M W=4 that is 31,118,976,
   8,347,008, 27,648 and 47,616. Allocate `max(A_r, 1)` elements, because
   A_r is **0** on some ranks in the tiny test configs (L=1 W=8 rank 5;
   L=2 W=8 ranks 4 and 6). Whether a 0-element `enqueue_create_buffer` works
   is UNVERIFIED, so don't depend on it.
4. **AdamW per segment:** params and grads as in section 5,
   `m + shard_rel`, `master + shard_rel`, `v + compact_off`.
5. **`_v_count()`** returns `A_r` under Muon at every stage. The state file
   is `256*4 + (opt + A_r)*4` bytes in S1 to S3, and
   `256*4 + (opt + A_r + S)*4` once S4 appends `muon_m`.
6. **`_update_hybrid` stays ZeRO-0 only** (section 5, order constraints).
7. **The legacy full-`v` copy-down applies only at `zero_stage == 0`** (at
   W=1 or ZeRO-0 W>1), the only layouts phase-1 Muon could write. Guard it
   explicitly. At ZeRO >= 1 any length mismatch raises. Write the copy as a
   loop over segments, `host[compact_off + i] = host[shard_rel + i]` in
   ascending order. That is safe in place because
   `compact_off <= shard_rel` for disjoint ascending segments.
8. **Merge hazard with fu/zs0.** Both branches add trailing optional
   arguments to `read_state_checkpoint` after `shard_num_parameters`
   (fu/vfree: `v_num_parameters`; fu/zs0: `expect_num_processes`,
   `expect_process_rank`), and both call sites in `load_checkpoint` pass
   them positionally. Whoever merges second must pass them by keyword.
   Otherwise `WORLD_SIZE` silently becomes the v length.

**S7's trim helper should be the same function.** S7 moves `m` and `master`
to `compact_off` using the same `_adamw_shard_segments()`. The copy-down
becomes one helper, `_compact_copy_down(host_buf, segs)` (shard_rel to
compact_off, ascending), used now for a legacy full `v` and in S7 for
untrimmed `m`/`v` files. After S7, `A_r` is the length of `m`, `v` and
`master` alike.

**Traffic per step** (bytes received, bf16, **averaged over ranks**; plain-ZeRO
baselines use the padded N). The last column is extra bytes **summed over
all ranks**. It is not a per-rank figure; see the imbalance note below the
table.

| stage | plain ZeRO | sharded Muon (this design), rank average | extra (total over ranks) |
|---|---:|---:|---:|
| ZeRO-1 | allreduce 373.4 MB + allgather 186.7 MB | same allreduce; AdamW-range gather 59.3 MB + owner gather 127.4 MB = 186.7 MB | **0 total**; per-rank gather sends up to +68% |
| ZeRO-2 | reduce-scatter 186.7 + allgather 186.7 | reduce-scatter split flat/owner (same total) + same gather | **0 total**; per rank up to +68% |
| ZeRO-3 | per-layer gathers + reduce-scatter | same bytes, split across two calls | **0 total**; per rank up to +68% |
| rejected: flat allgather + owner re-gather | | | +127.4 MB |
| rejected: Moonlight-style gather of U to intersecting ranks (gather-all) | | | +127.4 MB (their 1.25x) |
| rejected: owner broadcast via allreduce-with-zeros | | | +254.8 MB |

**Per-rank imbalance (review gen 2 correction; the first version said "+0
extra" per rank).** The total is unchanged, but rank r handles `A_r + O_r`
elements instead of opt: its AdamW intersection plus its owned matrices. At
124M W=4, `A_r + O_r` is 52.35M, 29.58M, 21.26M and 21.28M, against
opt = 31.12M for every rank. So rank 0 reduce-scatters 52.35M elements
instead of 31.12M (+68%), and sends `52.35M x 2 B x 3 = 314 MB` in the
gathers instead of 187 MB. Ranks 2 and 3 receive 206 MB instead of 187 MB.
At W=8, ranks 0 and 1 are +68%. The slowest rank sets the collective time.
This does not block S0 to S3. S6 must measure it per rank, and S7's
ownership work must weigh `A_r + O_r`, not NS cost alone.

The extra **collective calls** (each one adds barriers and, on GPU, a
synchronize):

- ZeRO-1/2: +1 per step.
- ZeRO-2/3: +L `reducescatter_buckets` per micro-step.
- ZeRO-3: +L `allgather_ranges` per forward and +L per backward re-stream.

GPU `allgather_ranges` issues one copy per (range, peer) intersection, so the
owner gather is 72 copies of 1.2 to 4.7 MB. The zero.mojo docstring measures
per-rank bandwidth falling off below about 6 MiB, so this may be
latency-bound (**UNVERIFIED** until a multi-GPU run). The mitigation, deferred
to S7, is to gather into owner-ordered staging (adjacent owner ranges coalesce
into W-1 copies) and then scatter locally.

NS compute per rank drops from all 72 matrices (ZeRO-0 today: about 30 to
40 ms/step on one GB10, per the Muon campaign report §6) to 18.

## 7. QK-clip at WORLD_SIZE > 1 (slice S5)

Per-rank head maxima differ, because each rank sees different data. Q and K
rows are Muon matrices (owner space), and `qkv_bias` is AdamW (flat shards).
`qk_clip_step` today indexes full `params_memory` and `master_memory`
(train_gpt2.mojo:`qk_clip_step`), which is wrong for any sharded master.

1. **Global max.** The smallest addition is
   `ZeroContext.allreduce_max_host[dtype](src, dst, size)`, a copy of
   `allreduce_or_host` with `max` in place of `|`. It is host-side on every
   target, and without a coordinator it degenerates to a copy. Calling the
   existing `allreduce_scalar` 144 times would work but costs 288 barriers.
   At the start of `_update_sharded_muon`: copy `head_max_buf` (L·NH fp32) to
   host, reduce, copy back. `qk_max()` then also reports the global value.
2. **Apply inside `update()`**, between the Muon step and propagation (step 5
   in section 5):
   - (a) For each (l, h) with `max > tau`, the owners of slots (Q, l) and
     (K, l) scale rows `[h*hs, (h+1)*hs)` of that C×C block in
     `muon_p_shard` (and `muon_master` under bf16) by `sqrt(tau/max)`. This is
     a new small kernel, or `qk_clip_apply` generalized to take one matrix
     base.
   - (b) Each rank scales its flat-segment intersection with the `qkv_bias`
     Q/K entries of that head (params + master shard).
   - The decision is rank-invariant (identical global maxima), so steps 7 and
     8 then carry the result. Under ZeRO-3 the next forward's gathers do.
3. The trainer's call to `qk_clip_step()` after `update()` becomes a no-op
   when `_update_sharded_muon` already clipped (flag set in `update()`).
   ZeRO-0 at W>1 needs only step 1, then the existing full-buffer
   `qk_clip_step`.
4. **Scope**: lift the `WORLD_SIZE > 1` raise in `enable_qk_clip` for
   `optimizer_kind == 1` (any stage) and for AdamW at ZeRO-0. AdamW with
   ZeRO >= 1 plus QK-clip keeps raising: its Q/K rows sit in flat shards, a
   third layout that this phase does not need. The CPU-target raise stays,
   since there is no logit hook.

S5 is optional for "Muon works under ZeRO". If it slips, the current
`enable_qk_clip` raise already covers it explicitly.

## 8. Test plan (one GPU, CPU multi-rank)

The tiny model is a synthetic checkpoint written by the test, in the style of
`setup_test_files`: L=2, NH=2, C=16, V=V_p=64, T=16, giving N=7,872 and
M=6,144. A second checkpoint with L=1 covers the "rank owns nothing" and
"S > opt" cases at W=8. NS on 16×16 and 64×16 matrices is instant in the host
reference. The d6 trainer run (124 s/step at W=1 Muon) is used only for the
manual smoke.

**T1: sequential-rank harness**, extending `tests/test_zero_equivalence.mojo`
(no coordinator, same batch on every rank). Add
`run_muon_equivalence_test[N](stage)`:

- Baseline: `GPT2["cpu", 1]` with `enable_muon`, one step.
- Per rank r:
  - Build `GPT2["cpu", N](cpu_coordinator_ptr=None)` and call `enable_muon`.
  - Run forward, backward, and the existing grad scaling. Also scale
    `muon_grad_shard[0:S]` by N for stages >= 2, since the no-coordinator
    `reducescatter_buckets` deposits local contributions into owner slots too.
  - Call `update`.
  - Assert (a) every AdamW segment of `params_memory` (stage 3: the flat shard
    copy) matches the baseline, and (b) for each owned slot,
    `muon_p_shard[local_off:+len]` matches `baseline[flat_off:+len]`.
- atol 1e-6. The fp32 math is identical; only multiplying the grad by N and
  then by 1/N separates the two paths.
- Cases: W ∈ {2, 4, 8} × stage ∈ {1, 2, 3}, plus L=1, W=8.
- Note: this harness cannot see propagation bugs, because it never runs a
  real collective.

**T2: threaded multi-rank** (new `tests/test_zero_muon.mojo`; the real
collectives, so this is the main gate). Pattern: `sync_parallelize` +
`CpuCoordinator`, as in tests/test_zero.mojo and the CPU branch of the
trainer's `main`. Give each rank its own model and its own
`DeviceContext(api="cpu")`, as the trainer does. A shared CPU ctx, as
tests/test_encoder_row_sparse.mojo uses for `ZeroContext`, is
**UNVERIFIED** for whole `GPT2` instances. Each rank gets a **different** batch.

- Reference: W ranks at ZeRO-0 with `-opt muon`, which is supported today, for
  3 steps. Copy full params to host arrays.
- Run: the same W, the same batches, ZeRO ∈ {1, 2, 3} (3 = streaming), 3 steps.
  Compare full params: `params_memory` for ZeRO-1/2, `_z3_gather_full` for
  ZeRO-3.
- Expected (corrected by review gen 2; the first version said bit-exact at
  every stage):
  - **ZeRO-1 vs ZeRO-0: bit-exact** (atol 0) for fp32 CPU. Grads are
    allreduced the same way, every collective sums ranks in the fixed order
    k = 0..N-1, Muon runs on the same values whether owner or replica, and
    AdamW is elementwise.
  - **ZeRO-2/3, step 1: bit-exact** for every Muon matrix and every non-wte
    tensor.
  - **ZeRO-2/3, 3 steps: rtol ~1e-4.** ZeRO-0 forms `wte_grad = lm_k + enc_k`
    locally, then allreduces. Bucketed ZeRO-2/3 reduce-scatters the LM-head
    and encoder parts separately, giving `(lm_0+lm_1) + (enc_0+enc_1)`. fp32
    addition is not associative, so wte differs by ulps after step 1, and
    every tensor differs from step 2. zero_grad_bucketing_design already
    states about 1e-4 vs z0/z1. Keep the step-1 exact check as the strong
    gate.
  - **Grad accumulation is not exact** at ZeRO-2/3 either:
    `sum_k(g_k1 + g_k2)` vs `sum_k g_k1 + sum_k g_k2`.
  - If an exact case fails, find out why before loosening anything (see the
    weak-gates memory note).
- Pass `grad_scale = 1`. The sharded `calculate_grad_norm` sums partials in a
  different order, so a clip-active comparison can differ by an ulp.
- Also check W=1 with batch B·W (the concatenated batches) at atol 1e-5,
  which shows DP equivalence and not just stage equivalence.
- Sizes: W ∈ {2, 4} with L=2, and W=8 with L=1.

**T3: collective units** (tests/test_zero.mojo, S0):

- `allgather_ranges` at N=4 without a coordinator, sequential: copies only
  the own-shard intersection and leaves the rest untouched. This test fails
  on today's code; it is the finding 2 regression.
- `allgather_ranges` threaded with `dst` aliasing the shard (finding 3).
  Results are correct, and the self copy is skipped (instrument with a
  counter or check by pointer equality).
- `reducescatter_buckets` into an owner space with `S != opt` and ranges that
  each lie wholly in one owner (extending `test_multi_cpu_reducescatter_buckets`).
- `allreduce_max_host` (S5): threaded W=4 plus the no-coordinator copy.

**T4: GPU single-rank kernel test** (tests/test_muon.mojo, S1). On a flat
layout of 3 matrices, `muon_update` with `MuonGroup.flat` must equal
`muon_update` on compact owner-layout copies with `sr_offset`/`sr_stride` set,
with params, master and m **bit-identical**. Bit-identity holds only when
every matrix has at most 2048 elements (BLOCK*VEC; above that the float-atomic
norm order is nondeterministic) and both sides use the same group `count`
(cuBLASLt `batch_count`). Use e.g. 32x32 and 32x64 matrices, and give the
owner side the same count as the flat side. Repeat under a
`-D LLMM_SR_MASTER=1` bf16 build if the test target supports it (whether
test-mojo builds an SR variant is **UNVERIFIED**). This is the only runtime
coverage the owner-layout GPU kernels get on this box.

**T5: trainer smoke** (manual gate for S1 to S3). Run
`make build WORLD_SIZE=2`, check the binary mtime, then
`make train-cpu WORLD_SIZE=2 ARGS="-e d6 -i <tinyshakespeare train> -j <val> -b 1 -t 64 -x 2 -pn 2 -v 0 -s 0 -opt muon -z {0,1,2,3}"`.
The step losses must agree to printed precision (6 decimals) across all
four stages. The step-1 grad norm already differs in the last digit (14.0609
vs 14.0608, sharded sumsq order) and the clip is active, so this is not
bit-equality. The fp32 CPU NS at
d6 makes this about 1 to 2 min/step. `TRAIN_BIN` (`build/train_gpt2`) does
not encode WORLD_SIZE, so rebuild after switching W.

**T6: checkpoint** (S4, threaded CPU W=2, ZeRO-2 and ZeRO-3):

- Run 2 steps, `write_checkpoint`, then 1 more step to get P.
- Build fresh models, `enable_muon`, `load_checkpoint`, run 1 step. The result
  must equal P bit-exactly (fp32).
- Loading a W=2 state at W=4 must raise, and so must W=4 at W=2. So must
  loading a ZeRO-0 Muon state at ZeRO >= 1 at the same W, and the reverse.
  So must a state file one element longer or shorter than
  `256*4 + (opt + A_r + S)*4`.

**GPU at W > 1: what stays untested here.** All of the following run only
with N >= 2 GPUs:

- `_reducescatter_buckets_gpu` with owner-space dests, including the
  scratch-size check;
- GPU `allgather_ranges` for the owner and AdamW-range propagation, including
  the self-alias skip;
- `ensure_comm_setup` with `max(opt, S)`;
- the GPU-branch routing in `update()`;
- the GPU ZeRO-3 split streaming;
- the W-round streaming seed on GPU;
- bf16 owner master seeding and re-seeding on resume;
- every performance claim.

On this box they are covered only by compile gates (`make build WORLD_SIZE=2`
and `make build-bf16 WORLD_SIZE=4` must compile the GPU N>=2 instantiations)
and by T4.

**2-GPU smoke for another machine.** Add configs `zero/muon_z{1,2,3}_w2.json`
(bf16, `train_flags: {b: 4, t: 1024, opt: muon}`), then run:

1. `make test-mojo` (includes the GPU multi-rank collective tests in
   test_zero.mojo).
2. `make train-zero ZERO_CONFIG=zero/muon_z{1,2,3}_w2.json ARGS="-x 30 -e d12"`.
   Compare against the same flags with `-z 0`: losses should agree to bf16
   noise (the fp32 build is expected to be exact), and no NaNs.
3. A 10-step write / resume / continue at ZeRO-2, against an uninterrupted run.
4. `LLMM_MEM_REPORT` at ZeRO-1/2/3, checked against section 6.
5. Step time at ZeRO-0 vs ZeRO-2 Muon.

## 9. Implementation slices

Each slice must build and pass `make lint` and `make test-mojo` on this box.
Check binary mtimes. Run the trainer only through `make`.

**S0: collective fixes + tests** (llmm/zero.mojo, tests/test_zero.mojo).

- `allgather_ranges`: N>1 without a coordinator copies only the own-shard
  intersection (finding 2), and the self copy is skipped when `src == dst`
  (finding 3, both CPU and GPU arms).
- `_reducescatter_buckets_gpu`: `_check_scratch[dtype](seg)` per segment.
- Gate: T3 (minus `allreduce_max_host`); existing test_zero and
  test_zero_equivalence green.
- **Status (review gen 2, design conformance only; gates are run
  separately):** fu/zs0 at `e2975e9` (7255440, 89a176c) **conforms**, with
  two test gaps for S1 to close.
  - (a) The no-coordinator intersection rule is implemented as specified.
  - (b) The self-copy skip is present in all five arms: CPU N=1, CPU
    no-coordinator, CPU coordinator, GPU N>1 `p == rank`, GPU N=1. It
    compares exact addresses, so callers must pass `dst_base`/`shard_base`
    that alias exactly.
  - (c) The scratch check is a host-side pre-pass (`_bucket_overlap_max`)
    over the largest bucket/shard overlap, run before any barrier. That is
    stronger than the per-segment check the design asked for. It also adds
    a CPU+GPU bounds check (`dest + len <= N*shard_size`), which suits the
    owner space (`shard_size = S`).
  - Beyond the design, 89a176c pulls S4's `num_processes` check forward and
    also checks `process_rank`. It does not cover ZeRO-0 state loaded at
    ZeRO >= 1 at the same W, so S4's exact length check is still required.
  - Test gaps (all runnable on CPU):
    - The alias test runs only at N=1 and in the no-coordinator arm. A CPU
      self-copy gives the same bytes whether or not it is skipped, so the
      test cannot see the skip. Add the threaded (coordinator) alias case.
    - T3's `reducescatter_buckets` into an owner space with `S != opt` is
      missing.
- **Verification (fu/zs0, Opus reviewer):** two defects fixed on top. (a) The
  `_bucket_overlap_max` scratch bound was the overlap with THIS rank's shard,
  so with an undersized scratch one rank could raise while a peer walked into
  the barrier and hung; it is now the max over every rank's shard, so the
  verdict is rank-invariant. (b) The world-size/rank check does not catch
  review item 4d's ZeRO-0 state resumed at ZeRO>=1 (same world size, same
  rank, larger shard): it still read a silent prefix (probe on the 7d68ac1
  sources: half-size read succeeded, `v[0]` held m's tail).
  `read_state_checkpoint` now raises if any byte follows v. That composes with
  fu/vfree's file-size check in `load_checkpoint` (its legacy-Muon path reads
  the full v, so no trailing bytes remain), and the new arguments are passed
  by keyword, since fu/vfree adds a positional `v_num_parameters` in the same
  slot. End to end at WORLD_SIZE=2: a `-z 0` checkpoint resumed with `-z 1`
  is refused on both ranks; `-z 1` written then resumed with `-z 1` gives step
  3 = 10.128281, the same as a straight run.

**S1: ownership + owner buffers + ZeRO-1** (llmm/muon.mojo `MuonGroup`
sr fields, `muon_apply_gpu`/`_muon_update_cpu`, the register bridge; train_gpt2.mojo).

- `enable_muon`: the raise is relaxed to only "stage >= 2 not bucketed"; it
  builds `muon_slots`, `muon_owned_groups` and S, allocates the four owner
  buffers, seeds them for the non-streaming case, and calls
  `ensure_comm_setup`. ZeRO-2/3 keep raising until S2/S3.
- New `_adamw_shard_segments` and `_update_sharded_muon` (the ZeRO-1 path).
  Route the CPU and GPU `update()` branches to it; skip their flat allgather.
- `memory_report_json`: account for the owner buffers.
- `muon_apply_gpu`: new `sr_stride_arg` kernel argument (section 4).
  `stride_arg` keeps indexing and must not carry `sr_stride`.
- fu/vfree: follow the 8-point contract in section 6 (one segment list,
  per-rank `A_r` compact `v`, copy-down only at ZeRO-0, keyword arguments for
  `read_state_checkpoint`).
- Tolerate `A_r == 0`, `O_r == 0` and `count_rk == 0` (tiny W=8 configs).
- Close S0's test gaps: threaded `allgather_ranges` alias, and
  `reducescatter_buckets` into an `S != opt` owner space.
- Gate: T1 at stage 1, T2 at ZeRO-1 (**bit-exact**), T4 (matrices <= 2048
  elements, equal counts), T5 at `-z 1`, compile gates.
- **Status (fu/zs1, implemented):** ownership table, owner buffers, the ZeRO-1
  `_update_sharded_muon` and the S0 test gaps are in. Details and deviations:
  - `MuonGroup` keeps its 5-argument constructor (flat: `sr_* = offset/stride`),
    so no W=1 call site changed; the 7-argument form and `MuonGroup.flat` are
    new. `muon_apply_gpu` takes `sr_offset_arg` and `sr_stride_arg`; `stride_arg`
    only indexes.
  - Owner buffers are allocated for `WORLD_SIZE > 1` and `zero_stage >= 1`
    (ZeRO-0 W>1 keeps the replicated path and `v_compact_bases`). At ZeRO >= 1
    `v_compact_bases` is empty and `v` is indexed by `AdamSeg.compact_off`;
    `_v_count()` is `A_r`. The legacy full-`v` copy-down runs only at ZeRO-0
    and loops over `_adamw_shard_segments()`.
  - `update()` routes to `_update_sharded_muon` once, ahead of both target
    branches, so neither flat allgather runs. AdamW propagation is
    `allgather_ranges(params+lo, opt -> params)`; the self copy is skipped. The
    owner gather uses `shard_size = S`. `ensure_comm_setup` gets
    `max(opt, S)` elements.
  - **Gotcha found by T2:** the CPU Muon reference reads `muon_g` on the host
    right after the pack's queued `enqueue_copy`; without a `synchronize()` in
    between, W=4 was nondeterministic (1-ulp differences in the proj matrices,
    and then in everything). The fix is one synchronize before the optimizer
    kernels. Any future pack/scatter into an owner buffer needs the same.
  - `write_checkpoint` / `load_checkpoint` raise for sharded Muon until S4
    (owner momentum is not in the state file yet).
  - `memory_report_json` counts the four owner buffers under `optimizer`.
  - Verified: T2 (`tests/test_zero_muon.mojo`, threaded CPU, different batch
    per rank, 3 steps, atol 0) ZeRO-1 == ZeRO-0 at W=2, W=4 and W=8 with L=1
    (ranks 6 and 7 own nothing). T4 (`test_owner_layout_matches_flat`, GPU) is
    bit-identical for 32x32, 32x64, 64x32. T5 at `-z 1` (W=2, d6): losses
    10.865850, 10.716245, 10.648577, equal to `-z 0` to all printed digits, and
    the ZeRO-1 steps take about half the time (NS runs on one rank's half of the
    matrices). `-opt adamw -z 1` is unchanged against main's binary. T1 (the
    sequential harness) was not extended; T2 covers the same ground with real
    collectives, and the harness cannot see propagation bugs. The multi-GPU
    arms stay compile-checked only.
- **Verification (fu/zs1, Opus reviewer):** the implementation conforms to
  sections 2 to 6. Two defects outside the update path are fixed on top, and
  one test was added.
  - **Checks that held.** Ownership, S, `O_r` and `A_r` match an independent
    Python model at 124M W=2/4/8, d6 and the tiny configs (L=1 W=8 rank 5,
    L=2 W=8 ranks 4 and 6 have `A_r = 0`). A new test,
    `test_ownership_sizes_tiny_w8`, pins those values. It also checks the
    segment invariants: nonempty, ascending, `shard_rel = flat_start - lo`,
    `compact_off` is the prefix sum. Neither `_update_hybrid` nor
    `v_compact_bases` is on the sharded path. The SR counter is
    `sr_offset + mat*sr_stride + i` on both CPU and GPU, and
    `sr_offset + j*sr_stride` is the flat offset of the owner's j-th matrix.
    This is confirmed by reading and by a compile of
    `-D LLMM_BF16=1 -D LLMM_SR_MASTER=1 -D WORLD_SIZE=2`. A bf16 SR
    W=1-vs-W>1 run is still UNVERIFIED. The ZeRO-1 owners pack the
    all-reduced `grads_memory` with the same scaled `grad_scale` as AdamW.
  - **The synchronize after the pack is a real fence.** CPU `adamw_update`
    and the CPU Muon path run directly on the calling thread, but CPU
    `enqueue_copy` is queued. Removing the synchronize failed T2 in 8 of 10
    runs, at W=2, 4 and 8. With it in place, T2 passed 20 of 20 runs.
  - **T2 mutants.** These were caught: owner gather skipped (all three
    sizes); `compact_off` not advanced (W=2 and W=4); a wrong `l0` in the
    owned groups (W=4, plus the ownership-table test). The W=8 L=1 case has
    at most one AdamW segment per rank, so it cannot see segment-offset bugs.
    Indexing `m` by `compact_off` instead of `shard_rel` passes, but that is
    an equivalent mutant under fp32 with no checkpoint: it is a consistent,
    disjoint relayout. It would break bf16 (`master` seeding is by
    `shard_rel`) and S4/S7 state files.
  - **Fixed: sharded Muon failed only at the first save.** With `-n`, the
    run trained up to the first save step and then raised. With `-y`, it
    raised at load. `train()` now refuses `-opt muon -z >= 1` at W>1 with
    `-o` plus `-n > 0` or `-y 1` up front. The method-level raises in
    `write_checkpoint` and `load_checkpoint` stay.
  - **Fixed: a failed threaded multi-rank run exited 0.** Both
    `sync_parallelize` launchers in `main` (CPU and GPU) printed
    `Rank r failed:`, aborted the coordinator and returned normally, so
    every exit-code gate passed on a dead W>1 run. This predates S1.
    `_raise_if_rank_failed` now raises after the join (exit 2).
  - Gates, rerun at `ce9f12c` with the binary mtime checked after each
    `make -B build-train`: `make test-mojo` gave `tests result=pass files=26
    passed=204 skipped=1 failed=0`, with test_zero_muon at 5 passed.
    `make verify` gave `result gate=cpu status=pass` and
    `result gate=gpu status=pass`. `make lint` passed, and
    `tests/test_muon_equivalence.py` gave 89 passed on the GPU. T5 losses
    reproduced exactly (`-opt muon -z 1` and `-z 0`: 10.865850, 10.716245,
    10.648577; `-opt adamw -z 1`: 10.865850, 10.369871, 10.128281; AdamW
    `-z 1` save at step 2 then `-y 1` gives step 3 = 10.128281). W=1 GPU
    Muon (d6, B=4, T=256, 5 steps) prints the same losses as main's
    `e4d91d3` binary.

**S2: ZeRO-2** (`backward` per-layer bucket split; `zero_gradients`;
`calculate_grad_norm` two-term norm; the ZeRO-2 arm of `_update_sharded_muon`).

- Gate: T1 and T2 at ZeRO-2 (step 1 bit-exact for Muon and non-wte
  tensors; 3 steps at rtol ~1e-4, see T2); T5 at `-z 2`.
- Also run with grad accumulation (`-d` = 2·B·T·W) in T5 or T2. The owner
  accumulator must `+=` across micro-steps and be zeroed once per step.
- **Status (fu/zs2, implemented):** `backward`'s per-layer bucket now makes
  two reduce-scatters when `optimizer_kind == 1`: Call A (the 8 non-Muon
  tensors, flat shard) and Call B (the 6 matrices, `dest = owner_idx` into
  `muon_g`, Q/K/V at `pool_slot[qkv] + k*C^2`). The wte two-part reduce is
  untouched. Details and deviations:
  - `zero_gradients` zeroes `muon_g_buf` next to `grad_shard_buf` (bucketed
    path). `enable_muon` now raises only for ZeRO-3 / streaming.
  - `_update_sharded_muon`: with bucketing, no pack; AdamW reads
    `grad_shard + shard_rel` (the flat shard's Muon positions stay zero, since
    Call A skips them); Muon reads `muon_g` as at ZeRO-1. Propagation is the
    ZeRO-1 pair of `allgather_ranges` calls.
  - **The grad norm needed the "two-term" fix** (`calculate_grad_norm`): at
    ZeRO-2 Muon the clip norm is `sumsq(grad_shard) + sumsq(muon_g)` before the
    scalar allreduce (`_grad_sumsq` helper). Without it the Muon matrices
    would be missing from the norm.
  - T2 (`tests/test_zero_muon.mojo`, threaded CPU, different batch per rank):
    step 1 bit-exact vs ZeRO-0 for every tensor except wte (wte max rel diff
    1.0e-7 to 4.1e-7); 3 steps pass `|a-b| <= 1e-4|a| + 1e-6`. Measured:
    W=2: max rel (|p| >= 1e-2) 1.8e-7, max abs 1.5e-8; W=4: 3.4e-5, 6.6e-7;
    W=8 L=1 (ranks 6, 7 own nothing): 4.1e-6, 1.3e-7. **A pure rtol 1e-4 does
    not hold**: at W=4 element 1884 (a qkv weight that happens to sit near 0,
    4.7887e-5 vs 4.7831e-5) is off by 5.6e-8 absolute, 1.2e-3 relative. That
    is fp32 cancellation noise after the wte difference feeds step 2/3
    gradients (step 1 is exact elsewhere, and ZeRO-1 stays bit-exact); the
    absolute gate of 1e-6 is ~100x below one lr=1e-3 step, so a wrong gradient
    still fails it. *(Reviewer: the noise claim holds, but the last clause
    does not. Small gradient errors pass this gate; see Verification below.)*
  - T5 at `-z 2 -opt muon` (W=2, d6, 3 steps): 10.865850, 10.716245,
    10.648577, identical to `-z 0`/`-z 1`. `-opt adamw -z 2` unchanged
    (10.865850, 10.369871, 10.128281).
  - Not done: the grad-accumulation case (`-d`); `+=` across micro-steps is
    the existing reduce-scatter semantics, but no test covers it with Muon.
    *(Reviewer: now covered, see Verification below.)*
  - **For S3 (streaming):** during `update`, params live in `params_memory`
    (fully resident at ZeRO-1/2; the owner's authoritative copy of its
    matrices is `muon_p_shard`, and `params_memory` is refreshed from it by the
    owner-space gather). At ZeRO-3 streaming only the flat shard
    `params_memory[0:opt]` is resident, so the owner must hold current params
    (and the bf16 fp32 master) in `muon_p_shard` / `muon_master` seeded by the
    W-round gather; there is no propagation after the update.

- **Verification (fu/zs2, Opus reviewer):** the routing, zeroing, grad norm
  and update path conform to section 5. No code defect was found. The test
  overclaimed what its 3-step tolerance catches and had no
  grad-accumulation coverage. Both are fixed in 9852e0d.
  - **Routing (read and mutated).** Call A plus call B cover exactly the old
    single call: the 4 skipped tensors are qkv (3C², split into 3 C² blocks
    at `pool_slot[qkv] + k*C²`), attn_proj (C²), fc (4C²) and proj (4C²).
    `muon_slots[k*L + l]` is the slot for (k, l) because pass 1 of
    `_build_muon_ownership` appends k-major. Each slot's `len` equals the
    pool length, and `owner_idx = owner*S + local_off` with
    `local_off + len <= S`, so every element reaches exactly one owner. The
    wte/wpe bucket is untouched. `muon_g` is zeroed only on the bucketed
    branch of `zero_gradients`, which runs on micro-step 0 before any bucket.
    ZeRO-1 still packs, which overwrites `muon_g`.
  - **Grad norm.** Call A never writes the flat shard's Muon positions
    (zero-filled), so `sumsq(grad_shard) + sumsq(muon_g)` counts every
    element once. At ZeRO-1 the second term is off (`_use_bucketing()` is
    false) and the full all-reduced buffer is summed once. The GPU kernel
    is grid-strided over a fixed grid, so reusing it with `n = S` is fine.
    The step-1 `norm=` at W=2, d6: `-z 0` and `-z 1` print 14.0608,
    `-z 2` Muon 14.0610, and `-z 2` AdamW 14.0609. That is 1.4e-5 relative,
    so not equal to printed precision. The CPU kernel accumulates fp32 per
    worker chunk, and the chunks differ by stage (one buffer vs. shard +
    `muon_g`, plus the cross-rank sum). In the threaded harness (L=2 C=16),
    `calculate_grad_norm` at step 1 prints 0.08311849 (W=2) and 0.06502338
    (W=4) for ZeRO-0, -1 and -2 alike. A missing or double-counted tensor
    would show there.
  - **Update path.** With bucketing the pack is skipped and AdamW reads
    `grad_shard + shard_rel`. The synchronize before the CPU AdamW/Muon
    calls is still there. Muon reads `muon_g`. The device-to-host copy in
    `_grad_sumsq` is followed by `ctx.synchronize()`.
  - **The tolerance claim is half right.** With every rank on the same batch,
    ZeRO-2 vs ZeRO-0 is bit-exact for all 3 steps at W=2 and W=4, with and
    without 2 micro-steps. So the different-batch 3-step drift comes only
    from the reduction order (the wte two-part sum, plus the per-micro-step
    reduce under accumulation). It is not a routing error, and `1e-4|a| +
    1e-6` is a fair bound for that drift. But the gate does not catch "any
    wrong gradient". These are the mutants (all ranks, applied to the pool
    before call B), as max `|a-b| / (1e-4|a| + 1e-6)` over 3 steps, at W=2 /
    W=4:

    | mutant | 3-step ratio | caught by step-1 exact? |
    |---|---|---|
    | none (baseline noise) | 0.0011 / 0.18 | n/a |
    | x1.001 on layer-0 Q block | 0.19 / 0.36 (pass) | yes (448 / 960 elems) |
    | x1.001 on one row of it | 1.65 (fail) / 0.49 (pass) | yes |
    | x1.001 on one element | 0.39 / 0.32 (pass) | yes |
    | x1.001 on ln_1 gamma (call A) | 1.16 (fail) / 0.35 (pass) | yes (4 elems) |
    | Q/K pool offsets swapped | 1090 / 1277 | yes |
    | last layer's call B dropped | 820 / 1126 | yes |
    | `muon_g` never zeroed | 693 (W=2) | **no** (step 1 is unaffected) |

    AdamW and Muon are nearly invariant to gradient scale, so a 1e-3 relative
    gradient error moves params by about as much as the reduction-order
    noise. No tolerance separates the two. Routing is pinned by the step-1
    exact check, and the per-step zero by the 3-step tolerance (693x). The
    margin is thin with accumulation: different-batch, 2 micro-steps, W=2
    reaches 0.84x, unmutated.
  - **Added:** `test_t2_zero2_same_batch_accum_bit_exact_w{2,4}`. Same
    batch on every rank, 2 micro-steps, 3 steps, ZeRO-2 == ZeRO-0
    bit-exactly. The one-element scale, never-zeroed and dropped-call-B
    mutants all fail it. It is the strong gate for accumulation and
    zeroing. The 3-step tolerance comment now says what that check can and
    cannot catch.
  - **Gates.** These ran on the tree with main merged in (20c4769, so S1 and
    S5 are included). Since ce9f12c a failed rank exits 2, so the rc=0 below
    is a real pass.
    - `make test-mojo`: `tests result=pass files=26 files_failed=0
      passed=214 skipped=1 failed=0`, exit 0. test_zero_muon passed 10/10.
    - `make verify`: `result gate=cpu status=pass` and
      `result gate=gpu status=pass`, exit 0.
    - `make lint`: pass.
    - `make build-bf16`: rc=0.
    - W=2 CPU, d6, 3 steps (`make -B build-train WORLD_SIZE=2`, mtime
      checked): `-opt muon -z {0,1,2}` all give 10.865850, 10.716245,
      10.648577, rc=0. `-opt adamw -z 2` gives 10.865850, 10.369871,
      10.128281.
    - Grad accumulation `-d 256` (2 micro-steps), `-opt muon`: `-z 2` and
      `-z 0` both give 10.851870, 10.717384.
    - W=1 GPU (`make -B build-train`, then
      `make train ARGS="-opt muon -x 5 -v 0 -s 0"`): 4.287241, 3.985736,
      4.111536, 3.619405, 3.700305.
  - **For S3:** call B's lists come from the static `muon_slots` table, so
    they are rank-invariant by construction. Keep that true for the ZeRO-3
    backward, which goes through the same bucket code. `muon_g` is zeroed
    only by `zero_gradients(zero_param_grads=True)`. A ZeRO-3 path that
    zeroes `grad_shard_buf` anywhere else must zero `muon_g` there too, or
    `_t2_z2_same`-style tests catch it only at step >= 2. QK-clip at
    ZeRO-2 still raises (`enable_qk_clip`). `_qk_clip_sharded` touches only
    params, so lifting that raise for ZeRO-2 should need only a T2-style
    test.

**S3: ZeRO-3** (`_z3_stream_layer` split, `_z3_gather_full` split, W-round
streaming seed, the ZeRO-3 arms of `_update_sharded_muon` for streaming and
non-streaming).

- Gate: T1 and T2 at ZeRO-3 (streaming, via the coordinator; same
  tolerances as ZeRO-2); T5 at `-z 3`.
- **Status (fu/zs3, implemented):**
  - `enable_muon` no longer raises for ZeRO-3 / streaming. Under streaming
    `_build_muon_ownership` seeds `muon_p_shard` (and the bf16 `muon_master`)
    with the W-round gather of section 2 (scratch of S elements, freed
    afterwards); `ensure_comm_setup` now runs before the seed, not after.
  - `_z3_stream_layer` and `_z3_gather_full` make two `allgather_ranges` when
    Muon is on: the 8 non-Muon tensors from the flat shard, the 6 matrices
    from `(muon_p_memory, S)` with `flat_starts = owner_idx` (Q/K/V land at
    `dst_offsets[qkv] + k*C^2`). `self.params.*` re-pointing is unchanged.
  - `_update_sharded_muon` at ZeRO-3: AdamW writes `params_memory + shard_rel`
    under streaming (`flat_start` otherwise), then returns before any
    propagation. Non-streaming ZeRO-3 only updates `muon_p_shard`; it does not
    write the matrices back into `params_memory` (no peers in the harness).
  - T2 (`tests/test_zero_muon.mojo`, `_t2_z2[W](..., stage=3)`, threaded CPU):
    step 1 bit-exact vs ZeRO-0 except wte (rel 4.1e-7 / 1.1e-7 / 1.1e-7 at
    W=2/4/8); 3 steps pass `|a-b| <= 1e-4|a| + 1e-6`: step3 max rel
    (|p|>=1e-2) 1.8e-7, 3.4e-5, 4.1e-6 and max abs 1.5e-8, 6.6e-7, 1.3e-7 at
    W=2, W=4, W=8 L=1 (identical to ZeRO-2). The streaming assertion: after
    the last update, `_z3_gather_full` agrees bit-exactly with the owner's
    `muon_p_shard` on its slots, and each layer's `_z3_stream_layer` window
    agrees bit-exactly with the gathered vector for all 6 matrices.
  - T5 `-z 3 -opt muon` (W=2 d6, 3 steps): 10.865850, 10.716245, 10.648577
    (= `-z 0`/`-z 2`); `-opt adamw -z 3`: 10.865850, 10.369871, 10.128281
    unchanged. `LLMM_MEM_REPORT=1` shows `muon_p_buf`/`muon_g_buf`/`muon_m_buf`
    at 21233664 B each on both ranks. (CPU Muon at `-z 0` takes ~250 s/step,
    ~66 s/step sharded; the `-z 0` losses were not re-run here.)
  - **For S4 (checkpoint):** under streaming the authoritative Muon params are
    `muon_p_shard`; `params_memory[0:opt]` holds stale values at the Muon
    positions. `write_checkpoint` must go through `_z3_gather_full` (already
    split) and a resume must re-seed `muon_p_shard` from the loaded host vector,
    not from `params_memory`.
  - **For S5 (QK-clip):** under streaming the flat shard holds no bias at
    `params_memory.unsafe_offset(a)` for a global offset `a`; use the
    shard-relative offset (`a - rank*opt`, only if it lies in this rank's
    slice) or gather the bias. Clipping Q/K matrices must scale `muon_p_shard`
    (and `muon_master` under bf16), because that is the next forward's source.
  - **Verification (fu/zs3, Opus reviewer, merged tree cc7d3b3):** no code
    defects found; S3's delta (7578837, e09f725) and the merge resolution
    were reviewed against 15f5064 (S2 itself was verified separately on
    fu/zs2). Read: the streaming seed computes each round's range list from
    the static slot table, so every rank passes identical lists (checked by
    `assert_ranges_agree` inside `allgather_ranges`) and a round whose owner
    holds no slots (W=8 L=1, ranks 6 and 7) is skipped by all ranks alike;
    `ensure_comm_setup(max(opt, S))` runs before the seed; each round's gather
    and the owner's copy out of the scratch are each followed by
    `ctx.synchronize()`, and the bf16 master promotion runs after the final
    synchronize. A slot never straddles an owner shard (`local_off + len <=
    O_owner <= S` by construction). `_z3_stream_layer` maps k=0..5 to window
    slots 2/4/8/10 with Q/K/V at `+k*C^2` and indexes `muon_slots[k*L +
    layer]` ((k, l) order); `_z3_stream_embed`, `_lm_head_wte_tile` and the
    `Z3_WTE_ONDEMAND` paths are untouched by the diff. The merge keeps the S5
    clip block before the ZeRO-3 early return; `enable_qk_clip` still raises
    for ZeRO >= 2 or streaming, so `_qk_clip_sharded`'s full-buffer bias
    offset `params_memory.unsafe_offset(a)` is unreachable at ZeRO-3 (a
    comment there now names the shard-relative fix needed when the raise is
    lifted). Non-streaming ZeRO-3 (`params_memory` full, no propagation):
    after an update the owned Muon matrices in `params_memory` are stale, and
    forward's `allgather` is a no-op there and in any case reads the flat
    layout, not the owner space. This is reachable only without a coordinator
    (`_z3_want_streaming` is true whenever one is attached, so the trainer and
    T2, which always pass one, take the streaming path); no test runs Muon in
    that mode, so a multi-step non-streaming ZeRO-3 Muon harness would train
    on stale matrices. That matches section 5's contract (the harness checks
    `muon_p_shard` directly), but it is a trap for anyone adding such a test.
    Mutation (in a scratch copy, not the worktree): `_z3_stream_layer` forced
    to its pre-S3 single flat-shard gather -> all three ZeRO-3 T2 tests fail
    with "streamed window != owner-space copy" on every rank (W=2, W=4, W=8).
    Gates on the merged tree (binary mtimes checked after each `-B` build):
    `make build-bf16` exit 0; `make -B build-train` + `make train ARGS="-opt
    muon -x 5 -v 0 -s 0"` -> 4.287241, 3.985713, 4.111485, 3.619386, 3.700239,
    `done steps=5`, exit 0; `make test-mojo` -> `tests result=pass files=26
    files_skipped=0 files_failed=0 passed=215 skipped=1 failed=0` (exit 0;
    test_zero_muon 11/11, test_qk_clip 8/8 incl. the W=2/W=4 ZeRO-1 and W=2
    ZeRO-0 planted-clip cases); `make verify` -> `result gate=cpu
    status=pass`, `result gate=gpu status=pass` (exit 0); `make -B
    build-train WORLD_SIZE=2` + `make train-cpu WORLD_SIZE=2` (d6,
    tinyshakespeare, -b 1 -t 64 -x 3 -pn 2): `-opt muon` at `-z 3`, `-z 2`,
    `-z 1` -> 10.865850, 10.716245, 10.648577 each, exit 0; `-opt adamw -z 3`
    -> 10.865850, 10.369871, 10.128281, exit 0. The commit adding this note
    changes only comments/docstrings in train_gpt2.mojo.

**S4: checkpoint** (llmm/checkpointing.mojo, `write_checkpoint`/`load_checkpoint`).

- New state header slots: 41 zero_stage, 42 muon_ownership_version
  (0 = none, 1 = round-robin `(k*L+l) mod W`), 43 muon_shard_elems S.
  Slot 2 (`num_processes`) already holds W.
- Payload appended after `m` and `v`, only when slot 42 != 0: `muon_m[S]`
  (fp32). Masters are not saved, matching the existing convention: re-seed
  `muon_master` from the loaded params, and re-seed `muon_p_shard` from the
  loaded full host vector (load_checkpoint already has it).
- `load_checkpoint` refuses on any of these:
  - `num_processes != WORLD_SIZE` (all optimizers; this is a behaviour change
    for AdamW, so say so in the commit);
  - a different ownership version or S;
  - Muon with `(zero_stage >= 1) != (saved >= 1)`. The layouts differ: ZeRO-0
    keeps Muon momentum in the full flat `m`.

  ZeRO-1, 2 and 3 share a layout, so a resume across those stages is allowed.
- Also check the file length exactly against
  `256*4 + (opt + _v_count() + S)*4` (S term only when slot 42 != 0), and
  raise on any mismatch in either direction. A longer file must not be read
  as a prefix, and a shorter one must fail with a clear message rather than
  "Failed to read enough bytes". The only exception is fu/vfree's legacy
  full-`v` file, at `zero_stage == 0` only (section 6, contract point 7). If
  fu/zs0's `num_processes`/`process_rank` check is already on main, keep it
  and add only the remaining refusals.
- Gate: T6.
- **Status (fu/zs4): implemented.** Layout of a sharded-Muon rank file:
  header (slots 2 W, 3 rank, 40 kind=1, **41 zero_stage, 42
  muon_ownership_version=1, 43 S**) + `m` (opt, fp32) + `v` (A_r, compact) +
  `muon_m` (S, fp32). Length `256*4 + (opt + A_r + S)*4`. AdamW and W=1 /
  ZeRO-0 Muon files write zeros in 41-43 and are byte-identical to before.
  **Master policy: `muon_master` is not saved** (as `master_buf` is not); on
  resume `muon_p_shard` is re-seeded from the loaded full params and
  `muon_master` is rebuilt from it, so bf16 resume is not bit-exact and fp32 is.
  `read_state_checkpoint` takes `muon_m_out` + `expect_muon_shard_elems`,
  refuses owner/replicated layout mismatch (so ZeRO-0 file at ZeRO>=1 and the
  reverse), ownership version or S mismatch, and checks the file length exactly
  (header + m + v [+ muon_m]) in both directions with actual vs expected bytes;
  this replaced the "Failed to read enough bytes" and trailing-byte messages.
  `load_checkpoint` keeps the legacy full-v copy-down at ZeRO-0 only. The stage
  recorded in slot 41 is informational; ZeRO-1/2/3 share the layout.
  Verified: unit T6 `test_state_checkpoint_sharded_muon_t6` (stage list
  `[1, 2, 3]`; S2/S3 need no change here), and end to end at threaded CPU W=2
  `-z 1`: `-x 2` then `-y 1 -x 4` gives losses 10.648577, 10.410458 for steps
  3-4, equal to a continuous `-x 4` run, and the step-4 model and both state
  files are byte-identical (`cmp`). Refused: `-z 0` resume of the `-z 1` files,
  W=1 resume of W=2 files, a file 4 bytes short (both byte counts printed).
- **Verification (fu/zs4, Opus reviewer, 2026-10-30).** Rebuilt and reran
  everything below on fu/zs4 (2105d06 + comment fixes); binary mtimes checked
  after each `make -B`.
  - Gates: `make lint` exit 0; `make test-mojo` exit 0, `tests result=pass
    files=26 passed=211 skipped=1 failed=0` (test_checkpointing 14/14; T6 =
    `test_state_checkpoint_sharded_muon_t6`, plus
    `test_state_header_plain_files_unchanged`, each also run alone with
    `--only`); `make verify` exit 0, `result gate=cpu status=pass`,
    `result gate=gpu status=pass`; `make -B build-bf16` exit 0.
  - Byte identity vs main: trainers built from 20c4769 sources (`git archive`,
    same toolchain) and from fu/zs4, CPU d6 `-x 2 -n 2`. `cmp` of
    `model_2.bin` and `state_2_0.bin`: identical for AdamW W=1, Muon W=1 and
    AdamW W=2 `-z 1` (both rank files). Slots 41-43 are 0 in all of them.
  - Sharded layout: W=2 `-z 1` rank files have slots (2,3,40,41,42,43) =
    (2,r,1,1,1,5308416) and sizes 142664704 / 100197376 bytes; rank 1 is
    `1024 + (15178752 + 4561920 + 5308416)*4` exactly.
  - Save/resume, threaded CPU W=2 `-z 1` Muon: `-x 2` then `-y 1 -x 4` gives
    steps 3-4 = 10.648577, 10.410458, exit 0; continuous `-x 4` prints the same
    and `model_4.bin`, `state_4_0.bin`, `state_4_1.bin` are `cmp`-identical.
    AdamW W=2 `-z 1`: resumed steps 3-4 = 10.128281, 9.916376, identical files.
    W=1 Muon: 10.794926, 10.554329, files identical to continuous. A legacy
    full-v file (2e1913d sources, 242861056 bytes, i.e. `1024 + 2*opt*4`)
    resumes with the same two losses and a `model_4.bin` identical to the
    continuous run. Note: `-y 1` re-saves the resume step at loop start, so the
    step-N files in a resumed dir are rewritten (in the current layout).
  - Refusals (make exit 2; direct binary exit 1): `-z 0` resume of `-z 1`
    files ("holds owner-local Muon momentum ... expects the replicated
    layout"); W=1 resume of W=2 files ("written by rank 0 of 2 process(es) but
    this run is rank 0 of 1"); rank 1 file 4 bytes short ("is 100197372 bytes,
    expected 100197376 (m=15178752, v=4561920, muon_m=5308416 elements)").
    Reverse direction: a W=2 `-z 0` Muon file at `-z 1` ("has no owner-local
    Muon momentum"), and a W=2 `-z 0` AdamW file at `-z 1` ("is 242861056
    bytes, expected 121431040"), which is the AdamW behaviour change.
  - Mutant: one `muon_m` element of rank 0's step-2 file set from 4.6e-7 to
    1.0. The resumed run exits 0; steps 3-4 print the same losses (step 3's
    loss is computed before the step-3 update, and one warmup-lr step moves
    the loss below print precision) but `model_4.bin` differs from continuous
    in 19,861,728 of 30,357,504 params (max |diff| 1.8e-3, largest in layer 0's
    qkv), and the step-4 `muon_m[0]` is 0.9029 vs -1.29e-5. The blob is read
    and used.
  - bf16 (W=1, GPU, gpt2_124M_bf16): Muon `-x 4 -n 2` then `-y 1 -x 6` exit 0,
    steps 5-6 = 3.756170, 3.979250, `done steps=6`; AdamW the same, 3.673406,
    3.937004. The sharded bf16 path (`_load_muon_owner_state`'s master rebuild)
    needs W > 1 on GPU and is not run here.
  - Code order in `load_checkpoint`: model file read and copied to device
    first (unchanged from main), then the optimizer_kind peek, then
    `read_state_checkpoint`: world/rank, owner layout, version/S, exact length,
    and only then the m, v, muon_m blobs. No state payload is read before a
    refusal; params are already overwritten when a refusal raises, which only
    matters to in-process callers that catch the error (train() exits).
  - Master policy confirmed: `_load_muon_owner_state` re-seeds `muon_p_shard`
    from the loaded host vector `host_params` (not `params_memory`), runs after
    `enable_muon` (train() calls it before `load_checkpoint`), and rebuilds
    `muon_master` from the same vector under bf16 (padding zeroed).
  - Fixes in this pass: two stale comments (the legacy copy-down pointed at a
    size check that moved into `read_state_checkpoint`; the trailing-byte
    read is now a backstop behind the exact length check). No behaviour
    change.
  - Integration with S2/S3 (not on this branch): S2 needs no checkpoint
    change. S3 makes the owner space authoritative at ZeRO-3 and drops the
    post-update propagation, so the flat shard's Muon ranges go stale.
    `write_checkpoint` under streaming still gets correct full params because
    S3's `_z3_gather_full` gathers the matrices from `muon_p` (second gather).
    `load_checkpoint` under streaming copies only the flat shard into
    `params_memory`, but must keep passing the full host vector `host_params`
    to `_load_muon_owner_state`; never `params_memory`. The merge must also
    keep S3's `enable_muon` seeding (W rounds under streaming) before the
    load, and the T6 stage list stays `[1, 2, 3]`. End to end, the merge must
    add threaded CPU W=2 `-z 2` and `-z 3` Muon save/resume runs (`-x 2`, then
    `-y 1 -x 4`, vs continuous `-x 4`, `cmp` model and both state files), and
    a cross-stage resume (`-z 1` files at `-z 3`), which the layout allows.
- **Integration (fu/zs4int, S4 onto main 537a850 with S2+S3, 2026-10-30).**
  Merge resolutions: `enable_muon` = S3's (W-round seeding, `ensure_comm_setup`
  before the seed, no raise); `write_checkpoint`/`load_checkpoint` = S4's logic
  (streaming `write_checkpoint` goes through S3's split `_z3_gather_full`;
  `_load_muon_owner_state(host_params, host_mu)` gets the full host vector
  after seeding); train() refusal block stays removed; `_update_sharded_muon`
  = S3+S5; `llmm/checkpointing.mojo` and `tests/test_checkpointing.mojo` = S4;
  `tests/test_zero_muon.mojo` = main. Git merged without textual conflicts.
  Gates on the merged tree: `make test-mojo` `tests result=pass files=26
  passed=219 skipped=1 failed=0` (test_checkpointing 14/14, test_zero_muon
  13/13); `make verify` cpu+gpu pass; `make lint` exit 0; `make build-bf16`
  exit 0. Threaded CPU W=2 d6 Muon `-x 2 -n 2` then `-y 1 -x 4` vs continuous
  `-x 4`: `-z 1`, `-z 2` and `-z 3` give step 3 = 10.648577 in resumed and
  continuous runs; step 4 prints 10.410458 resumed vs 10.410459 continuous at
  `-z 2`/`-z 3` (1e-6 print-level difference in the logged loss; `-z 1` is
  10.410458 in both) and `cmp` of model_4.bin, state_4_0.bin, state_4_1.bin is identical
  in all three. Cross-stage: `-z 1` step-2 files resumed at `-z 3` load and
  give 10.648577, 10.410458, the same as the `-z 1` continuous run. Regression:
  AdamW W=2 `-z 2` resume bit-exact (cmp), W=1 Muon resume bit-exact (cmp),
  W=1 GPU Muon 5 steps 4.287241, 3.985745, 4.111457, 3.619328, 3.700196.
  Observed flake (pre-existing, not S4): a resume with `-n k` dividing the
  resume step rewrites that step's `model_k.bin` while the other rank may
  still be reading it; one run under heavy parallel load failed rank 1 with
  "Failed to read enough bytes" and passed on rerun.

**S5: QK-clip multi-rank** (`allreduce_max_host`; the clip inside
`_update_sharded_muon`; `enable_qk_clip` scope).

- Add a test-only enable that skips the CPU-target guard, so tests can plant
  `head_max`.
- Gate: T3 `allreduce_max_host`; threaded CPU W=2 ZeRO-{1,2,3} with planted
  per-rank head maxima (different per rank) vs ZeRO-0 + global max: params
  bit-equal.
- If S5 slips, keep the raise and document it.
- **Status (fu/zs5, ZeRO-0/1 implemented; ZeRO-2/3 done in fu/qklift, below):**
  - `ZeroContext.allreduce_max_host` (llmm/zero.mojo) is a host-side copy of
    `allreduce_or_host` with `max`; one code path for every target (the
    coordinator's shared-inputs registry or, without a coordinator, a copy).
    Tested threaded at W=4 and without a coordinator (tests/test_zero.mojo).
  - `enable_qk_clip(tau, planted_head_max_only=False)`: the WORLD_SIZE > 1 raise
    is gone for ZeRO-0 (either optimizer) and ZeRO-1 Muon. It still raises for
    AdamW + ZeRO >= 1 (flat-shard Q/K rows) and for ZeRO-2/3 (sharded Muon
    itself raises there until S2/S3). The CPU-target raise STAYS: the CPU
    attention path has no logit-max hook, so `planted_head_max_only=True` is the
    test-only escape and the caller plants `head_max_memory`.
  - `_qk_sync_head_max` (all-reduce-max of `head_max_buf`, idempotent per step,
    flag cleared by the backward fold) is called from `qk_max()`,
    `qk_clip_step()` and `_qk_clip_sharded`; so `qk_max()` logs the global max
    and is a collective at W>1 (every rank calls it, as the trainer does).
  - ZeRO-1 Muon: `_qk_clip_sharded()` is invoked from ONE delimited block in
    `_update_sharded_muon`, after the `synchronize()` that follows the AdamW and
    Muon local updates and BEFORE the "Propagate AdamW" gather. It scales (a)
    the owner's owner-space Q/K C x C blocks (`muon_p_memory`, and
    `muon_master_memory` under bf16) with the new `qk_clip_matrix`, and (b) the
    qkv-bias entries inside this rank's flat shard (params + flat master shard)
    with `qk_clip_bias_range`; the two existing gathers then carry the result.
    It sets `qk_clip_applied`, which turns the trainer's later `qk_clip_step()`
    into a no-op. ZeRO-0 at W>1 only runs the sync, then the existing full-buffer
    `qk_clip_step`. W=1 is untouched.
  - Merge note for S2/S3: the block is the 3 lines
    `if self.qk_clip_tau > 0.0: self._qk_clip_sharded()` in
    `_update_sharded_muon`. For ZeRO-2/3 it must stay after the last local write
    and before the gathers (ZeRO-3: the owner-space Q/K clip is enough, the next
    forward's gather carries it, but the bias clip needs the flat shard under
    the ZeRO-3 params layout, so `_qk_clip_sharded`'s `params_memory.unsafe_offset(a)`
    must be revisited for streaming), and `enable_qk_clip`'s ZeRO-2/3 raise lifted.
  - Verified: tests/test_qk_clip.mojo threaded CPU W=2 ZeRO-1, W=4 ZeRO-1 and
    W=2 ZeRO-0 with per-rank-different planted maxima; every rank's final params
    are bit-equal to (unclipped step + `qk_clip_apply` with the global maxima).
    Not run: a multi-GPU run (one GPU here); the W=2 GPU-target build compiles.
  - **Verification (fu/zs5, Opus reviewer):** one test gap fixed (d076235),
    no code defects found. Read: `allreduce_max_host` is `allreduce_or_host`
    with `max` (same barrier1/barrier2 pair, so a rank that raises first is
    released by the trainer's `CpuCoordinator.abort()`; nothing raises between
    the barriers); on GPU `_qk_sync_head_max` copies `head_max_buf` to a host
    buffer and synchronizes before the reduce, then copies back and
    synchronizes. The clip block sits after the post-update `synchronize()`
    and before both gathers; Q is group 0 and K group 1 (`enable_muon`), so
    `sl.k > 1` skips V; bias layer = r // 3C, Q/K = row < 2C. `qk_max()` is
    called unconditionally on every rank (only the `qk_max` log field is
    verbose-gated); `planted_head_max_only` has no CLI path. The trainer's step
    loop is byte-identical to 6fa563b. Gap: with MV=64 every rank's flat shard
    held all or none of the qkv bias, so `qk_clip_bias_range` only ever ran
    with `rel_start = 0`, and the mutation `r = i` (drop `rel_start`) PASSED
    all three multi-rank tests. MV=192 puts the shard boundary at flat 4960
    inside the bias [4928, 5024) at W=2 and W=4 (asserted in the test). The
    `qk_max == 20` check moved after the params comparison. Mutations (each
    applied, run, reverted): collective returns the local copy -> W=4 ZeRO-1
    and W=2 ZeRO-0 params mismatch (idx 3520), test_multi_cpu_allreduce_max_host
    fails; `_qk_clip_sharded` skips the sync (clips on local max, `qk_max`
    still global) -> mismatch idx 1472; bias clip skipped -> idx 2888 (W=2,
    W=4, pre-MV change); V bias rows scaled -> idx 2920; V matrices scaled
    (`k > 2`) -> idx 1984; clip moved after the gathers -> idx 1472; `r = i`
    -> idx 4968 (W=2, W=4) after the MV change. Not covered by any test: the
    bf16 master offsets in `_qk_clip_sharded` (`sg.shard_rel + (a -
    sg.flat_start)`, `muon_master_memory + local_off`), since CPU builds are
    fp32 (has_master=False) and a multi-rank GPU run needs two GPUs; read and
    consistent with `_update_sharded_muon`'s own AdamW/Muon master indexing.
    Gates at d076235: `tests result=pass files=26 passed=208 skipped=1
    failed=0` (test_qk_clip 8/8 incl. the three multi-rank cases, test_zero
    21 pass incl. the two `allreduce_max_host` tests), `result gate=cpu
    status=pass`, `result gate=gpu status=pass`, `make lint` and `make
    build-bf16` clean, W=1 forced clip qk_max 110.63404 / 6.28879 / 5.8591275 / 5.617808 / 5.2096047,
    W=1 Muon losses 4.287241 / 3.985745 / 4.111457 / 3.619319 / 3.700221, W=2
    CPU ZeRO-1 Muon 10.865850 / 10.716245 / 10.648577, W=2 CPU `-qk_clip 5`
    raises "QK-clip needs a GPU target" on both ranks).
  - **Status (fu/qklift, ZeRO-2/3 lifted):** `enable_qk_clip` no longer
    raises for ZeRO-2 or ZeRO-3/streaming (AdamW + ZeRO >= 1 and the
    CPU-target-without-planted-maxima raise stay). `_qk_clip_sharded`'s qkv-bias
    write is layout-aware: under `z3_streaming` it indexes the resident flat
    shard with `sg.shard_rel + (a - sg.flat_start)`; ZeRO-1/2 keep the
    full-buffer offset `a`; the master offset was already shard-relative. The
    clip block already ran after the post-update synchronize and before the
    ZeRO-3 early return, so the owner-space Q/K clip lands before the next
    forward's gather. Tests: `test_qk_clip_multirank_zero{2,3}_w{2,4}`
    (MV=192 shard boundary inside the bias) compare every rank's params (ZeRO-3:
    `_z3_gather_full`) bit-exactly against an unclipped step AT THE SAME STAGE
    + `qk_clip_apply` with the global maxima (the stage-0 reference differs by
    1 ulp in unrelated rows because reduce-scatter and allreduce sum in a
    different order), assert the exact changed-element count, and at ZeRO-3
    also check the owner-space copies. Mutant (full-buffer bias offset under
    streaming) fails `test_qk_clip_multirank_zero3_w2` with the gathered-params
    bit-exact assertion ("multi-rank QK-clip != reference ... stage 3").
  - **Verification (fu/qklift, Opus reviewer, 2026-10-30):** independent
    read of `git diff 537a850..b44fa52` plus reruns; no defects found.
    - Bias write: `p_off` is shard-relative only under `z3_streaming`, the
      same indexing `adamw_update` uses for `params_memory` in
      `_update_sharded_muon`; ZeRO-1/2 and non-streaming ZeRO-3 keep `a`;
      the master offset is unchanged. Each segment `[a, b)` is clamped to
      this rank's flat shard, so a bias range that straddles a shard boundary
      is split, and each rank clips only its own part with the same global
      maxima (`a - b_lo` gives the kernel the bias-relative row). Note: the
      `(a - sg.flat_start)` term is always 0 in the current layout (the AdamW
      range holding the bias starts exactly at `b_lo`, so `a` is either
      `b_lo` or the shard start, and either one equals `flat_start`). It is
      correct as written, but no test can tell it apart from `+ 0`.
    - Ordering: the clip block still runs after the post-update
      `synchronize` and before the ZeRO-3 early return. `_qk_sync_head_max`
      runs once per step on every rank, because the `qk_clip_tau > 0` gate is
      rank-invariant. The later `qk_max()` hits the `qk_head_max_global` latch
      and makes no second collective, and `qk_clip_step()` returns on
      `qk_clip_applied`.
    - Remaining raises: AdamW + ZeRO >= 1 is unchanged (read).
      `planted_head_max_only` is reachable only from tests: the one trainer
      call site, `enable_qk_clip(args.qk_clip_tau)`, uses the default. The CPU
      raise was confirmed by running `make train-cpu WORLD_SIZE=2 ARGS="... -opt muon
      -z 3 -qk_clip 5"`. Both ranks printed "QK-clip needs a GPU target (no CPU
      logit-max hook)" and the run exited 2.
    - Tests: the changed-element count `2*2*hd*(C+1)` is asserted on the
      reference (Q/K weight rows plus their bias entries), and every rank
      bit-equals the reference, so the count carries over. The ZeRO-3
      owner-space check counts `2*2*hd*C` (matrices only; the bias is not in
      owner space) and requires each Muon element to be owned exactly once.
      A same-stage reference is required. With the reference forced to stage
      0, `zero2_w2` and `zero3_w2` fail at idx 1240 (wte) by 1 ulp
      (-0.15410821 vs -0.1541082), so the author's ulp claim holds.
    - Mutants on `test_qk_clip_multirank_zero3_w2`, built from scratch
      copies: (1) the author's full-buffer offset `p_off = a` under streaming,
      and (2) the bias clip skipped on every rank except 0. Both FAIL with
      "multi-rank QK-clip != reference at W=2 stage 3 rank 0 idx 4976"
      (layer-1 Q bias of hot head (1,0), in rank 1's shard).
    - Gates: `make test-mojo` -> `tests result=pass files=26 files_skipped=0
      files_failed=0 passed=221 skipped=1 failed=0` (test_qk_clip 12/12);
      `make verify` -> `result gate=cpu status=pass`, `result gate=gpu
      status=pass` (exit 0); `make lint` passes; `make -B build-bf16` builds.
      `make -B build-train` plus `LLMM_LOG_LEVEL=verbose make train
      ARGS="-opt muon -qk_clip 5 -x 5 -v 0 -s 0"` gives qk_max 110.63404 /
      6.2860556 / 5.8567953 / 5.619182 / 5.2094717. `make -B build-train
      WORLD_SIZE=2` (mtime checked) plus `make train-cpu WORLD_SIZE=2` (d6,
      -b 1 -t 64 -x 3 -pn 2) with `-opt muon -z 2` and `-z 3` give 10.865850 /
      10.716245 / 10.648577 for each, exit 0.


**S6: GPU multi-rank verification** (configs `zero/muon_z*_w2.json`, the
runbook in section 8). No new code paths are expected: S1 to S3 are
target-generic by construction. The GPU-only differences are inside
ZeroContext, and S0 changes them. Gate here: compile gates. Gate on a 2-GPU
box: the smoke list, plus **per-rank** collective time and bytes for the
reduce-scatter and the gathers (rank 0 carries up to +68%, section 6). Report
the slowest rank, not the average. Losses vs ZeRO-0 agree to bf16 noise,
not bitwise (section 4).

**S7 (follow-up, not this phase)**:

- **Status (fu/zs7, 2026-10-30): flat `m`/`master` trim DONE; rebalancing
  measured and proposed only, not implemented.** Under sharded Muon
  (W>1, ZeRO>=1) `enable_muon` (`_trim_flat_m_master`) replaces the
  shard-sized `m` and bf16 `master` with `A_r`-element buffers in the
  `compact_off` layout of the compact `v`. AdamW reads/writes `m`/`master` at
  `AdamSeg.compact_off` and the S5 qkv-bias QK-clip writes `master` at
  `compact_off + (a - flat_start)`. The state file's `m` blob is `A_r`
  elements (`_m_count()`; header unchanged; `muon_ownership_version` stays 1
  because the ownership rule did not change). A pre-trim sharded file is
  refused by the exact-length check (message names the trim); W=1, ZeRO-0 and
  AdamW buffers/files are untouched (byte-identical to a 3482c54 build).
  Memory at W=2 `-z 1` d6 Muon (fp32, `LLMM_MEM_REPORT=1`): rank 1 `m_buf`
  60,715,008 -> 18,247,680 B and `optimizer` class 142,663,688 -> 100,196,360
  B; rank 0 is unchanged (60,715,008 / 185,131,016) because its whole shard is
  AdamW (`A_0 = opt`: wte/wpe sit in rank 0), so the per-job peak is rank 0's
  and only improves when `A_r < opt` there (W>=4 at 124M: A_0 = 31.12M of
  31.12M still, so rank 0 does not shrink either; ranks 1+ do).
  **Rebalancing finding:** the imbalance is the flat layout's `A_r`, not the
  owner rule (O_r is already even: 21.23M per rank at W=4, 10.62M at W=8).
  124M `A_r + O_r` (M elements): W=4 52.35/29.58/21.26/21.28 and W=8
  26.18/26.18/18.94/10.64/10.64/10.62/10.65/10.63, max/mean 1.68 in both.
  Proposal: owner = greedy longest-processing-time assignment of the 6L
  matrices (largest first) to the rank with the smallest running `A_r + O_r`,
  seeded with `A_r`; offline model gives W=4 31.12/30.76/31.29/31.31 (max/mean
  1.006) and W=8 max/mean 1.024. It needs a new rule, so bump
  `MUON_OWNERSHIP_VERSION` to 2 and refuse version-1 files, and update the
  `exp_o`/`exp_a` tables in test_zero_muon. Muon updates are per-matrix, so
  bit-exactness should hold, but it must be re-proven with T2 before landing.
- Trim flat `m`/`master` (and `v` with fu/vfree) to `_adamw_shard_segments`.
- Optional memory-aware ownership. Bump the ownership version. Weigh each
  rank's `A_r + O_r` (traffic and state), not NS cost alone. Under
  round-robin, rank 0 already carries all of wte's AdamW elements.
- Use the same `_adamw_shard_segments()` and `_compact_copy_down` helper
  that fu/vfree's `v` uses (section 6).
- Coalesced owner gathers via staging.
- Combine the two per-layer bucket calls into one by interleaving the flat and
  owner spaces into a single `(opt + S)` shard space.

**What makes slices harder than they look:**

- The CPU and GPU `update()` branches are separate code with different
  structure. Both must route to `_update_sharded_muon` and both must skip
  their own allgather.
- `_z3_stream_layer` is called from both forward and backward.
- The bucket lists must stay rank-identical: build them only from the static
  slot table.
- The `qkv_weight` pool slot must be split into three ranges.
- `TRAIN_BIN` silently keeps the previous WORLD_SIZE when the sources are
  unchanged. Run `make build WORLD_SIZE=N` and check the mtime.
- The harness's grad scaling must cover `muon_grad_shard`.

## 10. Reproduction of the numbers

The sizes, ownership balance, memory and traffic figures come from a short
Python model of `_compute_param_sizes`, `enable_muon`'s groups and
`_adamw_ranges`. It reproduces opt with the shard rounded to 8, which is
`simd_width_of` for bf16. The fp32 CPU width is 4, which can change opt by a
few elements. To recompute, implement owner `(k*L+l) mod W`, S = max owned
rounded to 8, and `A_r = |[r*opt, (r+1)*opt) ∩ adamw_ranges|`. Selected rows:

| config | N | M | opt | S | O_r | NS cost max/ideal |
|---|---:|---:|---:|---:|---|---:|
| 124M W=4 | 124,475,904 | 84,934,656 | 31,118,976 | 21,233,664 | 4 × 21,233,664 | 1.000 |
| 124M W=8 | | | 15,559,488 | 10,616,832 | 8 × 10,616,832 | 1.000 |
| 124M W=7 | | | 17,782,272 | 13,565,952 | 11.2M to 13.6M | 1.108 |
| d6 W=2 | 30,357,504 | 10,616,832 | 15,178,752 | 5,308,416 | 2 × 5,308,416 | 1.000 |
| tiny L=2 C=16 W=4 | 7,872 | 6,144 | 1,968 | 1,536 | 4 × 1,536 | 1.000 |
| tiny L=1 C=16 W=8 | 4,592 | 3,072 | 576 | 1,024 | 256,256,256,256,1024,1024,0,0 | 2.4 |

## AI use statement

Written with AI assistance (Claude Code, Opus agent), directed by Evan Owen.
The prior-art table was compiled by a research subagent from the cited papers
and repository source.
The independent review (generations 1 and 2) was done with AI assistance
(Claude Code, Opus agents), directed by Evan Owen.
The S1 implementation (Sonnet agent) and its verification (Opus agent) were
done with AI assistance (Claude Code), directed by Evan Owen.
