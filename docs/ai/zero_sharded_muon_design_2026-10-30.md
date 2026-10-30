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
   `optimizer_num_parameters` elements per blob. `read_and_copy` raises only
   when the file is short. A resume at a smaller W reads a prefix of a larger
   shard file without error. Header slot 2 (`num_processes`) is written but
   never compared (llmm/checkpointing.mojo:`read_state_header`,
   train_gpt2.mojo:`load_checkpoint`). S4 adds the check, which also changes
   AdamW resume behaviour.

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

| System | Where Newton-Schulz runs | What is communicated | Momentum | Ownership / balance | Overhead claimed | Stages | Source |
|---|---|---|---|---|---|---|---|
| Moonshot "Distributed Muon" (Moonlight, Alg. 1) | On the full matrix, on every DP rank whose flat partition intersects it | reduce-scatter G (fp32) → momentum on local shard → **DP gather** of momentum-applied shard (bf16) → NS → keep local partition → all-gather params | partitioned (flat shard), applied before the gather | inherits Megatron flat partition; no balancing | comms "(1, 1.25] of Distributed AdamW"; memory "half of Distributed AdamW" (1 buffer vs 2) | ZeRO-1 | arXiv 2502.16982; github.com/MoonshotAI/Moonlight |
| Megatron PoC PR #1428 (linked from Moonlight; closed) | every rank holding a slice; a matrix split across k ranks is computed k times | momentum on 1-D shard, `all_gather_into_tensor` of whole bucket (bf16), slice local range | partitioned | flat buffer | none stated | ZeRO-1 + TP/PP | github.com/NVIDIA/Megatron-LM/pull/1428 |
| Kimi K2 (MuonClip) | as Moonlight | not described | "optimizer states are distributed" | n/a | ~30 GB state/GPU | ZeRO-1 | arXiv 2507.20534. Cross-rank aggregation of the QK-clip max is **not stated (UNVERIFIED)** |
| DeepSpeed ZeRO-1/2 (`stage_1_and_2.py`, `runtime/zero/muon/`) | full gradient of each param intersecting the partition; boundary-straddlers redundantly | normal reduce; straddling Muon params switch the bucket to all-reduce | flat buffer, full size of each intersecting param (duplicated for straddlers) | contiguous partition | Muon state 4 B/param vs Adam 8 B; 31.4 vs 34.5 GiB on Qwen2.5-3B | 0/1/2 | github.com/deepspeedai/DeepSpeed (blogs/muon-optimizer) |
| DeepSpeed ZeRO-3 (`stage3.py`) | each rank a round-robin share of params, once per optimizer step | grads reduce-scattered, then **all-gathered back** with momentum; update + momentum all-gathered again | partitioned at rest, gathered transiently | round-robin in param order | none stated | 3 (rejects reduce_scatter / all2all modes) | same repo |
| KellerJordan/Muon `Muon` | owner only | none for grads (assumes DDP all-reduce); padded `all_gather` per block of `world_size` params | owner only | sort by size desc, round-robin | n/a | DDP | github.com/KellerJordan/Muon/blob/master/muon.py |
| modded-nanogpt NorMuon / ANVIL (current) | owner of a chunk of a stacked same-shape group, one batched NS | `reduce_scatter_tensor` of the stack straight into owners, then `all_gather_into_tensor` of updated params | owner-local chunk | same-shape stacks padded to a multiple of W | NorMuon paper: distributed NS +2.9% step vs +8.1% redundant | ZeRO-2-like | github.com/KellerJordan/modded-nanogpt (records 2025-12-11, `optim/anvil.py`); arXiv 2510.05491 |
| Megatron `LayerWiseDistributedOptimizer` | whole params per DP rank | DDP all-reduce grads, all_gatherv updated params; newer layout makes buckets shard-aligned for reduce-scatter | owner | LPT by NS cost `m·n·min(m,n)`; legacy snake order | n/a | ZeRO-1-like | NVIDIA Megatron `layer_wise_optimizer.py`; NeMo emerging-optimizers docs |

Two patterns appear. In **gather-to-all, compute redundantly** (Moonlight,
DeepSpeed ZeRO-1/2), Newton-Schulz runs on every rank that touches a matrix.
This keeps the flat partition and the state format, and costs up to 1.25x
traffic plus duplicated NS. In **owner computes** (KellerJordan,
modded-nanogpt, NorMuon, Megatron layer-wise), each matrix has one owner,
gradients reduce-scatter straight to that owner, and the params are
all-gathered. Traffic is the same as AdamW ZeRO and NS is divided by W. It
needs a balance rule and its own state layout.

**Recommendation for llm.mojo: owner computes, with a separate matrix-granular
owner space** (sections 2 to 5). Four reasons:

1. Finding 1: in our flat layout, deriving work from the flat shards gives
   rank 0 no NS and ranks 2 and 3 about 37% each. Owner-space round-robin is
   balanced exactly at 124M (section 2).
2. The existing primitives already express it. `reducescatter_buckets` and
   `allgather_ranges` take a generic `(shard_ptr, shard_size)` and a list of
   global indices. Pointing them at an owner space of W equal shards of size S
   routes gradients to owners and params back to everyone, with **zero extra
   bytes** over plain ZeRO (section 6). No new collective is needed except a
   host max-reduce for QK-clip.
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
- Pass `g.sr_offset` and `g.sr_stride` to `muon_apply_gpu` in place of
  `g.offset` and the stride in the `pidx` expression.
- In `_muon_update_cpu`, pass `pidx = g.sr_offset + mat*g.sr_stride + i`
  separately from `idx = base + i`.

Rank r's owned group k is:

```
MuonGroup(offset = local_off(k, l0), rows, cols, count = count_rk,
          stride = rows*cols, sr_offset = flat_off(k, l0), sr_stride = W*stride_k)
```

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
   v + shard_rel /* or compact_off under fu/vfree */)`.
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
  It addresses `m_memory` and `master_memory` at full-buffer flat offsets,
  which overruns opt-sized buffers.
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

**fu/vfree composition.** fu/vfree compacts `v` to the AdamW ranges with bases
from `_adamw_ranges()`. Ask it to express those bases as
`_adamw_shard_segments()[i].compact_off` for the shard `[lo, hi)`; at
W=1/ZeRO-0 the two are identical. The sharded path then indexes `v` with
`compact_off` and `m`/`master` with `shard_rel`. S7 moves `m` and `master` to
`compact_off` too.

**Traffic per rank per step** (bytes received, bf16; plain-ZeRO baselines use
the padded N):

| stage | plain ZeRO | sharded Muon (this design) | extra |
|---|---:|---:|---:|
| ZeRO-1 | allreduce 373.4 MB + allgather 186.7 MB | same allreduce; AdamW-range gather 59.3 MB + owner gather 127.4 MB = 186.7 MB | **0** |
| ZeRO-2 | reduce-scatter 186.7 + allgather 186.7 | reduce-scatter split flat/owner (same total) + same gather | **0** |
| ZeRO-3 | per-layer gathers + reduce-scatter | same bytes, split across two calls | **0** |
| rejected: flat allgather + owner re-gather | | | +127.4 MB |
| rejected: Moonlight-style gather of U to intersecting ranks (gather-all) | | | +127.4 MB (their 1.25x) |
| rejected: owner broadcast via allreduce-with-zeros | | | +254.8 MB |

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
- Expected: **bit-exact** (atol 0) for fp32 CPU. Every collective sums ranks
  in the fixed order k = 0..N-1 (`_reducescatter_cpu`, the coordinator branch
  of `reducescatter_buckets`), Muon runs on the same values whether owner or
  replica, and AdamW is elementwise. The trainer evidence is that d6 losses
  at `-z 0` and `-z 3` agree to 6 decimals over 3 steps. If exactness fails,
  find out why before loosening anything (see the weak-gates memory note).
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
with params, master and m **bit-identical**. Repeat under a
`-D LLMM_SR_MASTER=1` bf16 build if the test target supports it (whether
test-mojo builds an SR variant is **UNVERIFIED**). This is the only runtime
coverage the owner-layout GPU kernels get on this box.

**T5: trainer smoke** (manual gate for S1 to S3). Run
`make build WORLD_SIZE=2`, check the binary mtime, then
`make train-cpu WORLD_SIZE=2 ARGS="-e d6 -i <tinyshakespeare train> -j <val> -b 1 -t 64 -x 2 -pn 2 -v 0 -s 0 -opt muon -z {0,1,2,3}"`.
The step losses must be identical across all four stages. The fp32 CPU NS at
d6 makes this about 1 to 2 min/step. `TRAIN_BIN` (`build/train_gpt2`) does
not encode WORLD_SIZE, so rebuild after switching W.

**T6: checkpoint** (S4, threaded CPU W=2, ZeRO-2 and ZeRO-3):

- Run 2 steps, `write_checkpoint`, then 1 more step to get P.
- Build fresh models, `enable_muon`, `load_checkpoint`, run 1 step. The result
  must equal P bit-exactly (fp32).
- Loading a W=2 state at W=4 must raise. So must loading a ZeRO-0 Muon state
  at ZeRO >= 1, and the reverse.

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
- **Status (fu/zs0):** all of the above reproduced on the pre-fix binary and
  fixed (7255440, 89a176c). Notes for later slices: (1) the scratch check runs
  as a host-side pre-pass (`_bucket_overlap_max`) before any barrier, and also
  raises on CPU for buckets outside `[0, N*opt)`; with `ensure_comm_setup`
  sized to `comm_shard` the overlap cannot exceed scratch, so the check is a
  guard, not a live failure. (2) The self-copy skip compares addresses, so
  callers must pass `dst_base`/`shard_base` that alias exactly. (3) Resume
  now raises on `num_processes`/`process_rank` mismatch (AdamW and Muon alike);
  `read_state_checkpoint` takes optional `expect_num_processes` /
  `expect_process_rank`. (4) The multi-GPU arms are compile-checked only on
  this 1-GPU box.

**S1: ownership + owner buffers + ZeRO-1** (llmm/muon.mojo `MuonGroup`
sr fields, `muon_apply_gpu`/`_muon_update_cpu`, the register bridge; train_gpt2.mojo).

- `enable_muon`: the raise is relaxed to only "stage >= 2 not bucketed"; it
  builds `muon_slots`, `muon_owned_groups` and S, allocates the four owner
  buffers, seeds them for the non-streaming case, and calls
  `ensure_comm_setup`. ZeRO-2/3 keep raising until S2/S3.
- New `_adamw_shard_segments` and `_update_sharded_muon` (the ZeRO-1 path).
  Route the CPU and GPU `update()` branches to it; skip their flat allgather.
- `memory_report_json`: account for the owner buffers.
- Gate: T1 at stage 1, T2 at ZeRO-1, T4, T5 at `-z 1`, compile gates.

**S2: ZeRO-2** (`backward` per-layer bucket split; `zero_gradients`;
`calculate_grad_norm` two-term norm; the ZeRO-2 arm of `_update_sharded_muon`).

- Gate: T1 and T2 at ZeRO-2; T5 at `-z 2`.
- Also run with grad accumulation (`-d` = 2·B·T·W) in T5 or T2. The owner
  accumulator must `+=` across micro-steps and be zeroed once per step.

**S3: ZeRO-3** (`_z3_stream_layer` split, `_z3_gather_full` split, W-round
streaming seed, the ZeRO-3 arms of `_update_sharded_muon` for streaming and
non-streaming).

- Gate: T1 and T2 at ZeRO-3 (streaming, via the coordinator); T5 at `-z 3`.

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
- Also check the file length exactly, so a longer file cannot be silently
  read as a prefix.
- Gate: T6.

**S5: QK-clip multi-rank** (`allreduce_max_host`; the clip inside
`_update_sharded_muon`; `enable_qk_clip` scope).

- Add a test-only enable that skips the CPU-target guard, so tests can plant
  `head_max`.
- Gate: T3 `allreduce_max_host`; threaded CPU W=2 ZeRO-{1,2,3} with planted
  per-rank head maxima (different per rank) vs ZeRO-0 + global max: params
  bit-equal.
- If S5 slips, keep the raise and document it.

**S6: GPU multi-rank verification** (configs `zero/muon_z*_w2.json`, the
runbook in section 8). No new code paths are expected: S1 to S3 are
target-generic by construction. The GPU-only differences are inside
ZeroContext, and S0 changes them. Gate here: compile gates. Gate on a 2-GPU
box: the smoke list.

**S7 (follow-up, not this phase)**:

- Trim flat `m`/`master` (and `v` with fu/vfree) to `_adamw_shard_segments`.
- Optional memory-aware ownership. Bump the ownership version.
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
