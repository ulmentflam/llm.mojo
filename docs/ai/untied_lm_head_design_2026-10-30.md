# Untied LM head (W9): design + slice plan (2026-10-30)

Work item W9 of [`modded_nanogpt_adoption_plan_2026-10-30.md`](modded_nanogpt_adoption_plan_2026-10-30.md):
a separate `lm_head` weight `[V_p, C]` behind `-D LLMM_UNTIED_HEAD=1`, with a
runtime "tie early, untie late" variant (`-untie_at <frac>`). This document is
written so Sonnet implementers can execute it slice by slice (section 6)
without re-deriving anything. Its shape follows
[`zero_sharded_muon_design_2026-10-30.md`](zero_sharded_muon_design_2026-10-30.md).

**Status: design only, nothing implemented.** All line references are to
`main` at `fe3312a` (the base of `mn/w9`). Anchors marked **(read)** were
confirmed by reading the code at that commit; anchors marked **(inferred)**
come from a grep hit or a doc and were not read in full. Facts I could not
confirm by reading or running are marked **UNVERIFIED**. The memory numbers
in section 5 come from a short Python model of `_compute_param_sizes`,
`_adamw_ranges` and the Muon ownership rule (section 8); its tied-build
outputs reproduce the published figures in the accounting and Muon docs.

## Critical findings (read these first)

1. **The memory cost is about 4x what the plan states.** The plan says
   "+147 MiB fp32 master plus AdamW `m`/`v`". The head is 38,633,472
   elements and every training buffer scales with it: at W=1 the untied build
   adds **589.5 MiB** in both fp32 (4 param + 4 grad + 8 m/v bytes per
   element) and bf16 (2 + 2 + 4 master + 8 m/v). That is +31% of the static
   model state (1899.4 to 2488.9 MiB). At production shape it is still small
   next to activations (the campaign doc's section 8 puts attention
   probabilities at ~18 GiB and logits at ~6.1 GiB at B=32, T=1024 fp32,
   about half that in bf16), so it does not block the ranking runs on the
   GB10. Section 5.
2. **Put the head at the end (index 16).** Every existing prefix offset is
   then unchanged: Muon groups, QK-clip offsets, `_z3_alloc_windows`'
   per-layer range, `_per_layer_pool_elems`, the init loop's RNG stream, the
   16-tensor checkpoint prefix. `_adamw_ranges()`' last pair
   `[proj_bias, num_parameters)` already absorbs the head; splitting it at
   the head offset leaves the compact-`v` concatenation order unchanged.
   Defining `Parameters.lm_head = 16 if UNTIED_HEAD else 0` (an alias of
   `wte` in the tied build) lets most call sites be written once. Section 3.
3. **Zero-init needs no init-loop change.** `allocate_parameters_random`
   zero-fills the host buffer and draws only for wte, wpe and the four layer
   matrices (train_gpt2.mojo:2313-2357 (read); its comment at :2310 says "the 16 parameter tensors" and needs a wording update only). A 17th tensor is skipped by
   both predicates, so the head stays zero and the RNG stream for the other
   16 tensors is unchanged. The PyTorch reference stays aligned if
   `lm_head` keeps `LLMC_SKIP_INIT` and is zeroed explicitly (no draws).
4. **A zero head blocks the body gradient at step 0.** With `W_head = 0`
   the logits are 0 and `d ln_f = dlogits @ W_head = 0`, so the first step
   updates only the head (wte, wpe and every block get zero gradient). This is
   the speedrun's behaviour and not a bug, but any gradient test that uses a
   zero head is blind to the body. Every planted test below uses a nonzero
   head.
5. **A tied model file silently loads as the wrong thing in two places
   today.** `read_model_checkpoint` reads `header.config.num_parameters()`
   elements and ignores trailing bytes (llmm/checkpointing.mojo:278-310
   (read)), and llm.c's `export_hf.py`, which `scripts/export_to_hf.py` wraps,
   reads the 16-tensor blob (inferred). An untied file with the tied version
   number would be read as tied with the head dropped, with no error. The
   untied model file therefore gets new version numbers (13 fp32, 15 bf16) so
   every existing reader refuses it loudly (`read_model_header` accepts only
   3 and 5, llmm/checkpointing.mojo:247 (read); `max_gpt2_common.py:155`
   (read) the same; `GPT2.__init__` compares against `EXPECTED_VERSION`,
   train_gpt2.mojo:1670-1677 (read)).
6. **Untying makes ZeRO tests stricter, not looser.** Today the ZeRO-2/3
   multi-rank tests exempt `wte` from step-1 bit-exactness because ZeRO-0
   forms `lm_k + enc_k` locally before the all-reduce while the bucketed path
   reduces the two parts separately (tests/test_zero_muon.mojo:11-16,
   297-305 (read)). Untied, `wte` carries only the encoder part and the head
   only the LM part, so the predicted outcome is step-1 bit-exactness for all
   17 tensors. If the measurement disagrees, the residual difference is in the
   row-sparse versus dense encoder reduction order and must be explained, not
   tolerated. Section 6, S3.
7. **ZeRO-3 on-demand gathering becomes safe for the head.**
   `LLMM_Z3_WTE_ONDEMAND` is off by default and has never run end to end,
   because the encoder still indexes arbitrary `wte` rows (train_gpt2.mojo:163-174
   (read)). The untied head has one consumer, the LM-head GEMM, which already
   walks vocab tiles through `_lm_head_wte_tile`. The head can be gathered per
   tile with no encoder dependency, so the ZeRO-3 window grows by one tile
   (18.75 MiB fp32) instead of the whole head (147.4 MiB). `wte` stays
   resident. `LLMM_Z3_WTE_ONDEMAND` must be comptime-refused in the untied
   build. Section 3.6.
8. **Under sharded Muon the head lands on the last rank(s).** The flat shard
   puts `wte` on rank 0 and the head on rank W-1, so S7's per-rank AdamW state
   `A_r` now has two heavy ranks. At W=4 bf16 the peak per-rank optimizer
   state rises from 585.8 MB (rank 0) to 685.6 MB (rank 0; rank 3 676.1 MB).
   Memory-aware ownership is already the open S7 follow-up. Section 5.3.
9. **The flag is not available in `make test-mojo`, `make verify` or the
   infer builds.** `test-mojo` builds every test with no defines
   (Makefile:1355-1381 (read)); `build-infer*` omits `$(MOJO_DEFINES)`
   (Makefile:605, 624 (read)). The untied gates need new targets (S0, S7).
10. **W1 has not started.** `mn/w1` has no commits and a clean worktree at the
    time of writing (`git rev-list --count fe3312a..mn/w1` = 0). The `-l_head`
    interface in section 4.3 is therefore designed against the plan's
    description of `-l_adam`, and it lists what W1 must expose.

## 1. Inventory: every consumer of the tied-`wte` / 16-tensor assumption

Grouped by file. "Change" says what the untied build needs; "tied build"
says whether the tied build's code path is touched at all.

### train_gpt2.mojo

| # | anchor | what it assumes | change under `LLMM_UNTIED_HEAD` | status |
|---|---|---|---|---|
| 1 | `comptime NUM_PARAMETER_TENSORS = 16` :180 | 16 tensors | `17 if UNTIED_HEAD else 16` | read |
| 2 | build-flag registry comment :190-253 | single source of truth for `-D` flags | add `LLMM_UNTIED_HEAD`, and `LLMM_Z3_HEAD_RESIDENT` if adopted (Q5) | read |
| 3 | `struct Parameters` :465-481 | indices 0..15 | `comptime lm_head = 16 if UNTIED_HEAD else 0` (alias of `wte` when tied) | read |
| 4 | `_parameter_name` :484-518 | 16 names | `"lm_head"` for 16, guarded | read |
| 5 | `ParameterTensors` fields :522-574, `point_parameters` :576-608 | 16 field pointers; the loop indexes `ptrs[i]` for `i < len(param_sizes)`, so a 17th size would index past the list | add `var lm_head` field; append its pointer under the flag; in the tied build set `self.lm_head = self.wte` after the loop | read |
| 6 | `param_sizes` init :1641-1643 | sized by the constant | none (constant changes) | read |
| 7 | `.bin` loader in `GPT2.__init__` :1660-1690 | version 3/5 only; config has no layout field | untied build accepts tied version (copy `wte` into head) and untied version 13/15 | read |
| 8 | `enc_row_sparse = self._use_bucketing()` :1721 | encoder and LM head both write `wte` | unchanged; row-sparse now serves only the encoder | read |
| 9 | `_z3_param_offset` :1785 | prefix sums | none | read |
| 10 | `_z3_finalize_params` :1815 | full vector of `num_parameters` | none (sizes from `num_parameters`) | read |
| 11 | `_z3_alloc_windows` :1871-1900 | embed window = wte + wpe + ln_f (wte resident unless ONDEMAND) | add one head tile buffer (on-demand) or the whole head (resident) | read |
| 12 | `_z3_stream_embed` :1902-1944 | gathers wte/wpe/ln_f | untied: `wte` always gathered; head not gathered when on-demand | read |
| 13 | `_lm_head_wte_tile` :1946-1979 | LM-head tile is a `wte` row range; ONDEMAND gathers `_z3_param_offset(wte)` | rename `_lm_head_tile`; source tensor is `self._head_t()`; on-demand gathers `_z3_param_offset(head_t)` | read |
| 14 | `_z3_gather_full` :2121-2169 | loops `NUM_PARAMETER_TENSORS`, skipping Muon matrices | none (17th tensor is an AdamW tensor, gathered in the flat loop) | read |
| 15 | `_compute_param_sizes` :2171-2242 | 16 sizes; shard = ceil(N/W) aligned | append `param_sizes[16] = V_p*C` under the flag; `opt`, `padded` follow | read |
| 16 | `allocate_parameters` :2244-2264 | reads `num_parameters` from the model file | tied file in untied build: read N16, then copy the wte block into the head block on the host | read |
| 17 | `allocate_parameters_random` :2266-2366 | init loop over 16 tensors in PyTorch order | none (finding 3); head stays zero | read |
| 18 | `allocate_parameters_from_safetensors` :2368-2530 | reads 16 HF tensors | read `lm_head.weight` if `"lm_head.weight" in st.tensors`, else copy wte; head padding rows stay zero | read |
| 19 | `_lm_head_tile_rows` / `_lm_head_num_tiles` / `_chunked_ce` :2539-2572 | tiles over `V_p` | none (applies to the dense head unchanged) | read |
| 20 | `_enc_grad_pool_elems`, `_grad_pool_elems` :2632-2668 | the docstring argues the pool bound for the tied tensor | bound unchanged (head bucket = one tile, encoder = chunk); docstring update only | read |
| 21 | `_encoder_backward_row_sparse` :2702 | `flat_wte` dest | unchanged | read (signature) |
| 22 | encoder forward `params.wte` :3603 | encoder reads `wte` | unchanged | read (grep) |
| 23 | forward LM head: chunked :4092, K=1 `matmul_fwd(..., self.params.wte, ...)` :4132, tiled :4144 | head weight is `wte` | use `_lm_head_tile(...)` and the head base pointer | read |
| 24 | backward Bucket 1 :4497-4662 (chunked :4527-4581, K=1 :4589-4614, tiled :4631-4658) | writes `self.grads.wte` (pool or real), dest `flat_base[Parameters.wte]` | write the head's grad pointer and `flat_base[head_t]` | read |
| 25 | backward Bucket 4, encoder :5444-5495 | `wte` grad accumulates onto the LM-head part | unchanged code; in the untied build `grads.wte` holds only the encoder part | read |
| 26 | `param_offset` :5571 | prefix sums | none | read |
| 27 | `enable_muon` :5578-5660, `v_compact_bases` :5637-5647 | one compact base per `_adamw_ranges()` pair (5 pairs) | 6 pairs (head split off); concatenation order unchanged | read |
| 28 | `_build_muon_ownership` :5732 | owner space over Muon matrices only | none (head is AdamW); `opt` grows, `ensure_comm_setup(max(opt, S))` follows | inferred |
| 29 | `_adamw_shard_segments` :5909-5932 | intersects shard with `_adamw_ranges()` | none in code; gains a head segment | read |
| 30 | `_adamw_ranges` :5940-5954 | 5 pairs, last `[proj_bias, N)` | split the last pair at `param_offset(lm_head)` under the flag | read |
| 31 | `_update_hybrid` :5956-6006 | one `AdamWConfig` for all ranges | head range gets its own config (lr mult) and step counter; skipped while tied | read |
| 32 | `_update_sharded_muon` :6019-6155 | one config for all segments | same per-segment switch on `sg.flat_start >= head_off` | read |
| 33 | `update()` AdamW sites :6232, :6270, :6292, :6337, :6365 | one `adamw_update` over the whole shard | one helper that splits `[lo, hi)` at `head_off` (body config, head config) | read |
| 34 | `calculate_grad_norm` :6378, `_grad_sumsq` :6433 | one reduction over `n` | while the head is tied, reduce over the body only (keeps the CPU chunk partition identical, section 3.4) | read |
| 35 | debug probes `LLMM_DEBUG_LOCAL_GRADNORM` / `TENSOR_NANSCAN` :6557-6700 | whole-tensor classes wte, wpe, ln_f (slots 0-3) | add the head as a fifth whole-tensor class | read |
| 36 | `write_checkpoint` :6753, model write :6806 | `num_parameters` blob, tied version | untied version via `CheckpointConfig` | read |
| 37 | `load_checkpoint` :6866-7050 | model file of `num_parameters`; state header kind check | refuse a tied model or state file in the untied build and the reverse; read untie fields (section 4.4) | read |
| 38 | `memory_report_json` :7097 | schema 3 keys | add `untied_head` key, emitted only in the untied build (Q11) | read |
| 39 | `TrainArgs` :7340-7400, CLI chain :8355-8370, setup table :7590-7612, config event :7820-7830 | no head flags | `-l_head`, `-untie_at` (+ env), refused in the tied build unless default | read |
| 40 | resume :7755-7790, checkpoint in loop :7949-7983, `model.update` :8039 | | untie transition goes after the checkpoint write and before the training section | read |

### llmm/

| file:anchor | assumption | change | status |
|---|---|---|---|
| llmm/checkpointing.mojo:34-47 `STATE_VERSION = 1`, `VERSION_FP32 = 3`, `VERSION_BF16 = 5` | | add `VERSION_FP32_UNTIED = 13`, `VERSION_BF16_UNTIED = 15`; keep `STATE_VERSION = 1` (Q10) | read |
| llmm/checkpointing.mojo:55-110 `CheckpointConfig`, `num_parameters()` :69-97, `__eq__` | 16-tensor sum | add `untied_head: Int` (default 0); `num_parameters()` adds `V_p*C` when set; `__eq__` compares it | read |
| llmm/checkpointing.mojo:195-231 `write_model_checkpoint` | version from dtype | untied version when `config.untied_head` | read |
| llmm/checkpointing.mojo:234-263 `read_model_header` | accepts 3/5 | accept 13/15 and set `untied_head = 1` | read |
| llmm/checkpointing.mojo:278-310 `read_model_checkpoint` | ignores trailing bytes | add an exact file-length check (finding 5) | read |
| llmm/checkpointing.mojo:121-181 `TrainingState`, :313-374 writer, :377-414 header reader, slots 40-43 at :347-350 / :408-411 | | new slots 44 (`untied_head`), 45 (`untie_step`, -1 = never), zero in tied files | read |
| llmm/checkpointing.mojo:417-552 `read_state_checkpoint` | exact length check | unchanged; the m/v lengths grow with `opt` and the length check already rejects a tied file | read |
| llmm/safetensors.mojo:287-325 `SafetensorsFile`, `tensors` dict, `get_info` raises on a missing key | | presence test via `name in st.tensors` | read |
| llmm/zero.mojo | collectives take caller range lists; no tensor knowledge (the only `wte` hit is a comment at :654) | none | read (grep) |
| llmm/encoder.mojo, llmm/matmul.mojo (`matmul_lm_head_fwd_tile`/`_bwd_tile`), llmm/fused_classifier.mojo | kernels take pointers | none | read (grep) |

### Reference, tests, tools

| file:anchor | assumption | change | status |
|---|---|---|---|
| train_gpt2.py:546-550 | `self.transformer.wte.weight = self.lm_head.weight`; `LLMC_SKIP_INIT` on the head | under `GPT2_UNTIED=1`: no tie, keep `LLMC_SKIP_INIT`, zero the head after `self.apply` (from-scratch only) | read |
| train_gpt2.py:556-575 `_init_weights` | head skipped | unchanged | read |
| train_gpt2.py:618 `from_pretrained` | copies HF keys by name; HF has `lm_head.weight` (= wte) | expected to work untied (both keys already in the tied `state_dict`) | inferred |
| train_gpt2.py:1451-1490 `write_tensor` | 16 `write_fn(model_tensors[...])` calls ending at `transformer.ln_f.beta` :1490 | append `lm_head.weight` (padded to V_p) when present | read |
| train_gpt2.py:1549-1580 `write_model` | pads only wte; version 3/5/7/9 | pad `lm_head.weight` too; version +10 when untied | read |
| train_gpt2.py:1583-1630 `write_state` | pads only the wte grad (:1609-1611); state version 3 | pad the head grad; set header[4] = 1 when untied | read |
| train_gpt2.py:2134-2147 | writes `gpt2_124M{,_bf16,_fp16,_fp8}.bin` and `gpt2_124M_debug_state.bin` | untied must write `_untied` names, never overwrite the starter pack | read |
| scripts/gen_expected_losses.py | 10-step loss table for test_gpt2 | untied table | inferred |
| test_gpt2.mojo:359 model from `gpt2_124M.bin`; :374-395 debug state magic/version; :406-431 expected grads sized by `model.num_parameters`; :652 `Array[Bool, 16]`; :654-862 sixteen `check_tensor` calls; :864-882 totals of 16 | | read `gpt2_124M_untied_debug_state.bin` under the flag, assert header[4] == 1, 17th check `dlm_head`, counts from `NUM_PARAMETER_TENSORS`, untied loss table | read |
| dump_grads_gpt2.mojo:8-14, :188 | dumps 16 named gradients | dump `lm_head` under the flag | read (grep) |
| tests/test_zero_muon.mojo:35-61 `_write_checkpoint` (16-tensor `n` formula :39-45) | | unchanged: a tied synthetic file loads into the untied build by copy (S0) | read |
| tests/test_zero_equivalence.mojo:62, tests/test_qk_clip.mojo:359/590, tests/test_cpu_queue_ordering.mojo:45, tests/test_lm_head_vocab_tiling.mojo:193 | synthetic tied `.bin` writers | unchanged, same reason | read (grep) |
| tests/test_checkpointing.mojo:57 `test_num_parameters`, :77 round trip, :633 `test_state_header_plain_files_unchanged` | 16-tensor count; plain headers | add untied cases (S0, S4) | read (grep) |
| Makefile:1355 `test-mojo`, :1504 `verify-cpu`, :1516 `verify-gpu`, :605/:624 `build-infer*` | no `-D` plumbing | `TEST_DEFINES` for test-mojo, `verify-untied-{cpu,gpu}`, `$(MOJO_DEFINES)` on infer builds | read |
| max_gpt2_common.py:152-157 | version 3/5 only | refuses untied files already; supporting them is optional (S7) | read |
| scripts/export_to_hf.py:10, :115-200 | 16-tensor llm.c layout via upstream `export_hf.py` | refuse version 13/15 explicitly (S7); full support optional | read |
| scripts/check_checkpoint_biases.py:56 | parses header[:7] and derives the 16-tensor total | refuse or extend (S7) | read (grep) |
| infer_gpt2.mojo | imports `GPT2` from train_gpt2 (:11) | inherits; needs the flag in its build | read |
| Metal | no LM-head-specific code: `HAS_METAL` only gates syncs (:4160 et al.) and the GEMMs go through the shared entry points | none expected; **UNVERIFIED** (no Mac on this box) | inferred |

## 2. What the speedrun does (from the plan, not re-read)

Per the plan's W1 and W9 sections: separate `embed` and `lm_head` since
record 8; the head zero-initialized; Adam groups at lr 0.6 (embeddings) and
0.008 (head), betas (0.8, 0.95), eps 1e-10, no weight decay; record 53 unties
at 2/3 of training. How record 53 initializes the head's **optimizer state**
at the untie point is **UNVERIFIED**; section 4.4 picks a rule and Q3 flags
it.

## 3. Design: the untied layout

### 3.1 Flag and constants

```
comptime UNTIED_HEAD = get_defined_int["LLMM_UNTIED_HEAD", 0]() != 0
comptime NUM_PARAMETER_TENSORS = 17 if UNTIED_HEAD else 16
struct Parameters: ... comptime lm_head = 16 if UNTIED_HEAD else 0
comptime if UNTIED_HEAD: comptime assert not Z3_WTE_ONDEMAND, "..."
```

`Parameters.lm_head == Parameters.wte` in the tied build is deliberate: code
written as `flat_base[head_t]`, `param_sizes[head_t]` resolves to the same
integers the tied build uses today, so the tied build executes the same
operations in the same order. Every place that would index `param_sizes[16]`
in the tied build is impossible by construction.

### 3.2 Which tensor the LM head reads: `_head_t()`

```
def _head_t(self) -> Int:
    comptime if UNTIED_HEAD:
        if not self.head_tied:
            return Parameters.lm_head
    return Parameters.wte
```

`self.head_tied: Bool` exists in both builds (always True in the tied build).
In the untied build it is False from construction unless `-untie_at` asks for
a tied phase (section 4.4). Forward and backward use `_head_t()` and two
helpers: `_head_param_ptr()` (`params.wte` or `params.lm_head`) and the
renamed `_lm_head_tile(t0, oc)`; Bucket 1 writes `self._head_grad_ptr()` and
reduces to `flat_base[self._head_t()] + t0*C`. The encoder never changes.

### 3.3 Why index 16

- Prefix offsets of tensors 0..15 are unchanged, so Muon groups
  (`param_offset(qkv_weight)` etc., :5594-5620), QK-clip offsets
  (:2926-2931, :3025), the ZeRO-3 per-layer window range
  (`range(ln_1_gamma, ln_f_gamma)`, :1877) and `_per_layer_pool_elems`
  (:2603) need no edit.
- The model file is the tied blob plus a trailing `[V_p, C]` block. A tied
  file can be loaded by reading the prefix and copying `wte`.
- `_adamw_ranges()`' last pair already covers it; splitting the pair keeps
  the compact `v` concatenation (prefix sums) identical, so `compact_off`
  for every tied segment is unchanged.
- The cost is placement under flat ZeRO sharding: the head lands on rank W-1
  (finding 8). Putting it at index 1 would put both 147 MiB tensors on rank
  0 and shift every offset, the fixture order and the init mapping. Not worth
  it.

### 3.4 Tied build stays byte-identical: what is comptime-guarded

Guarded by `comptime if UNTIED_HEAD` (the tied build compiles none of it):
the 17th `param_sizes` entry; the 17th `point_parameters` pointer; the
tied-file copy in the loaders and the safetensors head read; the head split
in `_adamw_ranges`; the split AdamW helper's second call; the body-only grad
norm; the ZeRO-3 head window or tile buffer; the debug-probe fifth class;
the untie transition, its log event, state slots 44/45 and the memory-report
key; the CLI flags' acceptance (the tied build raises if `-l_head` or
`-untie_at` is given a non-default value).

Not guarded, but provably the same operations in the tied build:
`Parameters.lm_head` (= 0), `_head_t()` (constant `wte`), the
`ParameterTensors.lm_head` field (aliases `wte`), `CheckpointConfig.untied_head`
(always 0, so `num_parameters()` and the written version are unchanged).

Proof obligations (S0 gate, not assertions): `make verify` (cpu and gpu),
`make test-mojo`, the deterministic 3-step CPU run, and a model and state
file `cmp` against a binary built from `fe3312a`, all on the tied build.

**The grad-norm subtlety.** `global_norm_squared_cpu` partitions
`num_params` into `ceildiv(n, parallelism_level())` chunks
(llmm/global_norm.mojo:60-68 (read)), so making `n` larger changes the
summation order even if the extra elements are zero. On GPU the grid is
fixed by the SM count (train_gpt2.mojo:3133-3140 (read)) and extra zeros add
exact `+0`. So while the head is tied (S0's idle head, or the tied phase of
`-untie_at`), `calculate_grad_norm` reduces only `[0, head_off)` at
W=1/ZeRO-0/ZeRO-1. Otherwise "untied build with the head idle equals the
tied build" would hold on GPU and fail on CPU in the last bits.

### 3.5 Optimizer split

One helper replaces the five whole-shard `adamw_update` calls in `update()`:

```
def _adamw_flat(mut self, p, g, master, m, v, flat_lo, n, t, cfg_body, cfg_head)
  # [flat_lo, flat_lo+n) is the rank's slice in flat coordinates; p/g/master/m/v
  # are already offset to flat_lo. Split at head_off = param_offset(lm_head):
  #   body: [flat_lo, min(flat_lo+n, head_off))       -> cfg_body, step t
  #   head: [max(flat_lo, head_off), flat_lo+n)       -> cfg_head, step t - untie_step
  #   head is skipped entirely while self.head_tied
```

In the tied build it is the original single call (comptime). `_update_hybrid`
and `_update_sharded_muon` do the same per range/segment. Two caveats:

- `LLMM_SR_MASTER` keys the stochastic rounding by element index within the
  call (llmm/adamw.mojo:178-202 (read)). Splitting the call changes the
  head's indices. Harmless (the untied build has no SR baseline), but the
  head call should pass its flat offset if SR determinism across layouts is
  ever wanted. Off by default.
- `cfg_head.weight_decay = cfg_body.weight_decay` (Q7). AdamW's decay term is
  multiplied by the lr (llmm/adamw.mojo:169 (read)), so `-l_head` scales the
  head's decay too, as in PyTorch.

### 3.6 ZeRO stages

The flat shard machinery is tensor-agnostic: `opt = ceil(N17/W)` aligned,
`padded = opt*W`, and every collective takes caller-built range lists. What
changes per stage:

| stage | params | grads | optimizer | head-specific |
|---|---|---|---|---|
| 0 | full | full `grads_buf` | full m/v(/master) | none |
| 1 | full | full, all-reduced | shard | head may straddle ranks W-2/W-1 at small W; segments handle it |
| 2 | full | Bucket 1 reduce-scatters head tiles to `flat_base[lm_head] + t0*C`; Bucket 4 sends only encoder chunks to `flat_base[wte]` | shard | the pool bound is unchanged: max(head tile, encoder chunk, per-layer) |
| 3 streaming | shard | as stage 2 | shard | embed window = wte + wpe + ln_f (as today); head gathered per tile into `head_tile_buf` before each tile GEMM, forward and backward (Q5) |

Sharded Muon: the head is an AdamW segment; owner space and S are unchanged.
The AdamW propagation gather in `_update_sharded_muon` (:6113-6131) already
uses `_adamw_ranges()` and includes the head.

The on-demand gather calls `allgather_ranges` with
`flat_starts = [_z3_param_offset(lm_head) + t0*C]`, a pure function of the
tile index, so range lists stay rank-invariant (the property
`_lm_head_wte_tile`'s ONDEMAND branch already relies on). This is the first
time that branch will run end to end; the S6 test must cover it.

## 4. Design: head init, optimizer, loaders, and the untie variant

### 4.1 Head init

| source | head contents |
|---|---|
| model descriptor (`-e d12`), from scratch | zero (speedrun); no RNG draws |
| tied `.bin` (version 3/5, e.g. `gpt2_124M.bin`) | copy of `wte`, all V_p rows (pad rows are zero in both) |
| untied `.bin` (version 13/15) | read from the file |
| HF safetensors | `lm_head.weight` if present (V rows, padding zero), else copy of `wte` |

The tied build cannot zero the head: it is `wte`. That is why W5's head part
waits for W9. A nonzero-init option (N(0, 0.02) or copy) is Q8.

### 4.2 Model file

Header slot 1 is the version: 13 (fp32 untied) or 15 (bf16 untied); slots
2-7 unchanged. The blob is the 16-tensor blob followed by `[V_p, C]`.
`CheckpointConfig.untied_head` carries the layout. `read_model_checkpoint`
gains an exact length check (finding 5), which also hardens the tied path.

### 4.3 `-l_head` and the W1 interface

W1 (`-l_adam <mult>`, env `LLMM_LR_ADAM_MULT`, default 1.0) has no commits
yet. W9 needs exactly this from it:

1. A `GPT2` field `adam_lr_mult: Float32` (or an equivalent hook) that
   `_update_hybrid`, `_update_sharded_muon` and the AdamW `update()` path
   read to build the AdamW-range config as `lr_sched(step) * adam_lr_mult`,
   applied after the scheduler (plan W1 anchors).
2. A single place where the AdamW `AdamWConfig` is built per range, so a
   second config can be built for the head range.
3. If W1 adds Adam betas/eps knobs for the AdamW ranges, the head range
   shares them (the speedrun's head uses the same betas/eps as embed).

W9 adds `-l_head <mult>` (env `LLMM_LR_HEAD_MULT`). Default sentinel `-1`
means "same multiplier as the other AdamW ranges" (`adam_lr_mult`, or 1.0
before W1), so the untied build with no flags trains every AdamW tensor at
one rate. The head lr is `lr_sched(step) * l_head`. It applies under both
`-opt adamw` and `-opt muon` (Q6). If W1 lands after W9's S5, S5 implements
items 1 and 2 itself with `adam_lr_mult` fixed at 1.0, and W1 fills it in.

The speedrun ratio for orientation: embed 0.6 and head 0.008 against Muon
0.05, i.e. head/embed = 1/75. In our units, with Muon at `-l 4.8e-3`, the
first ranking arm can try `-l_adam 4 -l_head 0.05` (section 7); W1's ablation
sets the embedding side.

### 4.4 `-untie_at <frac>`: train tied, then untie

Semantics (per the plan, with the step-0 case made explicit):

| `-untie_at` | behaviour |
|---|---|
| 0 (default) | untied from construction; head keeps its init (zero from scratch); no transition |
| 0 < f < 1 | tied until `s_u = max(1, floor(f * train_num_batches))`, then untie |
| >= 1 | never untie: the tied model expressed in the 17-tensor layout |

Only valid in the untied build. The whole head (params, grads, m, v, master)
is allocated from step 0 in every case (Q12); lazy allocation would change
`num_parameters`, the shard layout and every collective's ranges mid-run.

**Tied phase (step < s_u).** `head_tied = True`: the LM head reads `wte`,
Bucket 1 writes `wte`'s gradient, the head range is skipped by the
optimizer (its m and v stay exactly 0), and the grad norm covers the body
only. With `-untie_at 1` this must reproduce the tied build's run (S4 gate).

**Transition at step s_u**, placed after the checkpoint write at the top of
loop iteration `s_u` and before its training section (train_gpt2.mojo:7949-7990
(read)), so a checkpoint at exactly `s_u` is always pre-untie:

1. **params:** `head[:] = wte[:]`. ZeRO-0/1/2: params are replicated, each
   rank copies locally (identical on every rank). ZeRO-3 streaming: per
   LM-head tile, all ranks `allgather_ranges` the tile's `wte` rows into
   `head_tile_buf`, then each rank writes the intersection of that tile's
   head range with its own shard. Collective, rank-invariant ranges.
2. **master (bf16 builds):** for the rank's slice of the head range,
   `master = fp32(params)`, the same promotion `load_checkpoint` uses on
   resume (:6913-6955). Indexing: full at ZeRO-0, shard-relative at
   ZeRO >= 1, `compact_off` of the head segment under sharded Muon (S7 trim).
   Not a copy of `wte`'s fp32 master: that would need a cross-rank fp32
   gather at ZeRO >= 1, and promotion makes a resumed run and a continuous
   run take the same head master (Q4).
3. **m, v:** nothing to do; they are zero because the head was skipped. The
   head's AdamW step counter is `t - s_u`, so bias correction starts fresh,
   exactly like a parameter newly added to a PyTorch optimizer. Copying
   `wte`'s moments instead (Q3) would need the fp32 cross-rank gather and is
   not what a fresh param group does. Zero moments **without** a fresh step
   counter would be wrong: at large t the bias correction is ~1, so the
   first update would be about `lr * (1-b1)/sqrt(1-b2) ~ 3.2 lr` per element.
4. `head_tied = False`; emit the `untie` event.

**Checkpoint and resume.** State header slot 44 = 1 (untied layout), slot 45
= `s_u` (-1 = never, 0 = untied from construction). Tied-build files write
zeros, so they stay byte-identical. On resume, the untied build:

- refuses a state file with slot 44 = 0 (tied layout) or a model file with
  version 3/5 when resuming (the exact length check would also refuse, with
  a worse message);
- recomputes `s_u` from the current `-untie_at` and `-x`, and refuses if it
  differs from slot 45 ("pass the same -untie_at and -x, or ...");
- sets `head_tied = (resume_step < s_u)`.

Cases: resume from `s < s_u` replays the transition at `s_u` from the loaded
state; resume from `s = s_u` likewise (the checkpoint is pre-untie); resume
from `s > s_u` loads the head's params, m, v from the files and uses
`t - s_u`. In fp32 every case is exact. In bf16 the masters are rebuilt from
params on resume as today, so the gap is the existing AdamW resume gap
(about 3e-5).

**Log.** Setup table rows and the config event (llm mode `key=value`):
`untied_head=1 untie_at=<f> untie_step=<s_u> l_head=<mult>`. At the
transition: `untie step=<s_u> head_from=wte master=promoted moments=fresh`.
The `resume` event gains `head_tied=0|1 untie_step=<s_u>`. The memory report
gains `untied_head=1`. Validation at step `s_u` runs before the transition,
so the val loss logged at `s_u` is still the tied model's.

## 5. Memory accounting at 124M

From the model script in section 8, checked against the tied numbers in
[`zero_memory_accounting_2026-07-27.md`](zero_memory_accounting_2026-07-27.md)
(fp32 W=2 ZeRO-0 params 474.838 MiB reproduced exactly) and
[`muon_muonclip_implementation_2026-10-29.md`](muon_muonclip_implementation_2026-10-29.md)
(Muon compact v 158.2 MB and sharded-Muon rank-0 585.8 MB reproduced).
N16 = 124,475,904, N17 = 163,109,376, H = V_p·C = 38,633,472
(147.375 MiB fp32, 73.69 MiB bf16). MiB = 2^20 B. Per rank, at a single
point in time, model-proportional buffers only: params, grads (full buffer
or shard), m, v, master, comm scratch (two shard-sized staging buffers at
W > 1) and the ZeRO-3 embed window. Not included: the gradient pool (it does
not change, 3.6), activations (they do not change), CUDA context.

### 5.1 AdamW, per rank

| prec | W | ZeRO | tied MiB | untied MiB | delta | ZeRO-3 with head on-demand |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 1 | 0 | 1899.4 | 2488.9 | **+589.5** | |
| bf16 | 2 | 0 | 2136.8 | 2800.0 | +663.2 | |
| bf16 | 2 | 1 | 1424.5 | 1866.6 | +442.1 | |
| bf16 | 2 | 2 | 1305.8 | 1711.1 | +405.3 | |
| bf16 | 2 | 3 | 1262.3 | 1704.4 | +442.1 | 1640.1 (+377.8) |
| bf16 | 4 | 1 | 949.7 | 1244.4 | +294.8 | |
| bf16 | 4 | 2 | 771.6 | 1011.1 | +239.5 | |
| bf16 | 4 | 3 | 668.7 | 926.6 | +257.9 | 862.3 (+193.6) |
| bf16 | 8 | 3 | 372.0 | 537.8 | +165.8 | 473.4 (+101.5) |
| fp32 | 1 | 0 | 1899.4 | 2488.9 | **+589.5** | |
| fp32 | 2 | 0 | 2374.2 | 3111.1 | +736.9 | |
| fp32 | 2 | 3 | 1574.9 | 2164.4 | +589.5 | 2035.8 (+460.9) |
| fp32 | 4 | 3 | 862.6 | 1231.1 | +368.4 | 1102.5 (+239.8) |

By class, bf16 W=4 ZeRO-3 (resident head): params +18.4, grads +18.4, m+v
+73.7, master +36.8, comm +36.8, window +73.7. The replicated pieces (params
at ZeRO-0/1/2, grads at ZeRO-0/1, the window at ZeRO-3 unless on-demand) do
not shrink with W; everything else shrinks as 1/W.

### 5.2 Muon, W=1 (the ranking configuration on the GB10)

Muon reuses `m` at full size and keeps `v` compact over the AdamW ranges.
The head is an AdamW range, so the compact `v` grows by the whole head:
bf16 optimizer state 1100.5 MiB (m 474.8, v 150.8, master 474.8) becomes
1542.6 MiB (622.2, 298.2, 622.2). With params and grads the delta is again
+589.5 MiB. At `-b 32 -t 1024` bf16 the steady footprint is dominated by
activations (logits about 3 GiB and attention probabilities about 9 GiB,
scaled from the accounting doc's fp32 figures), so the head is a few percent
of the total and does not limit the ranking runs. The deliverable's "report
peak memory" must still be measured (S8), not taken from this table.

### 5.3 Sharded Muon after the S7 trim, bf16 (A_r·12 B + owner space S·10 B, the Muon doc's convention)

MB here is 10^6 B to match the Muon doc. The script in section 8 prints MiB and
counts 12 B per owner element (it includes `muon_g`), so its absolute figures
are higher; the A_r values and the tied-to-untied deltas are the same.

| W | build | A_r per rank (AdamW elements in the flat shard) | peak rank |
|---|---|---|---|
| 4 | tied | 31,118,976 / 8,347,008 / 27,648 / 47,616 | rank 0, 585.8 MB |
| 4 | untied | 39,438,336 / 55,296 / 36,864 / 38,644,224 | rank 0, 685.6 MB (rank 3 676.1 MB) |
| 8 | tied | rank 0 and 1 15,559,488, rest <= 8.3M | rank 0 |
| 8 | untied | ranks 0, 1, 6, 7 between 18.3M and 20.4M | ranks 0 and 7 |

### 5.4 What the memory campaign's floor becomes

The campaign ([`zero_wte_deresidency_campaign_2026-07-27.md`](zero_wte_deresidency_campaign_2026-07-27.md))
lowered two ~150 MiB floors. Honest status under the untied build:

- **Gradient pool (ZeRO-2/3): unchanged.** The head's Bucket 1 is still one
  vocab tile (18.75 MiB fp32, 6400 rows at K=8), the encoder is still one
  8192-row chunk plus wpe, and the per-layer bucket (27.0 MiB) remains the
  max. The row-sparse path now serves only the encoder; the vocab tiling now
  serves only the head. Both keep working.
- **ZeRO-3 embed window: rises unless the head is on-demand.** Tied default
  today: 150.4 MiB fp32 (wte resident because of the encoder). Untied with a
  resident head: 297.8 MiB. Untied with the head on-demand: 169.1 MiB
  (150.4 + one 18.75 MiB tile). The 21.8 MiB the campaign reports needs the
  indexed encoder gather, which still does not exist; that does not change.
- **Model state: the floor the campaign worked against is not the binding
  cost any more.** The head adds 16·H/W bytes of sharded state per rank plus
  the replicated pieces above, i.e. +589.5 MiB at W=1 and +194 to +258 MiB per
  rank at W=4 ZeRO-3, which is larger than both floors the campaign removed
  (about 123 and 129 MiB). Untying gives back more than the campaign saved, in
  exchange for an optimizer degree of freedom. That is the trade the ranking
  runs must justify.

## 6. Implementation slices

Each slice builds, passes `make lint` and `make test-mojo` on this box, and
checks binary mtimes. Builds go through `$S/build_slot.sh`; short GPU jobs
through `$S/gpu_slot.sh`. No run longer than 10 steps outside the queue.
Every "bit-exact" below means `==` on every element, not a tolerance.

**S0: 17-tensor layout, head idle** (train_gpt2.mojo, llmm/checkpointing.mojo,
Makefile, tests/test_checkpointing.mojo, new tests/test_untied_head.mojo).

- Interface: section 3.1-3.4 constants, `Parameters.lm_head`,
  `ParameterTensors.lm_head`, `_head_t()` returning `wte` always (the head is
  idle: forced `head_tied = True` in this slice), `_compute_param_sizes`,
  `point_parameters`, `_parameter_name`, loaders (tied `.bin` copy,
  untied `.bin` read, safetensors), `CheckpointConfig.untied_head`, versions
  13/15, `read_model_checkpoint` exact length, `_adamw_ranges` split,
  `_adamw_flat` with the head skipped, body-only grad norm. Makefile:
  `TEST_DEFINES ?=` passed to `mojo build` in `test-mojo` (empty default),
  `verify-untied-cpu`/`verify-untied-gpu` stubs.
- Tests:
  - Tied build: `make verify`, `make test-mojo` (`tests result=pass`), 3-step
    deterministic CPU run identical to `fe3312a`'s binary, and `cmp` of a
    3-step model and state file against `fe3312a`'s.
  - Untied build, planted: load a synthetic tied `.bin` with distinct random
    wte; head == wte bitwise, head padding rows zero. Write/read an untied
    file with a head distinct from wte (planted values 1000 + index); round
    trip bit-exact; the file is version 13; a tied-build reader refuses it;
    a truncated or extended file is refused.
  - Untied build with the head idle equals the tied build: the 3-step CPU run
    (W=1, `-gc 1.0`, so clipping is live) prints the same losses and the
    16-tensor prefix of the model file `cmp`s equal to the tied build's file
    (strip the trailing head block). This is what catches a wrong grad-norm
    range or a body offset error.
  - Mutant: make `_grad_sumsq` cover all N17 on CPU: the CPU comparison above
    must fail in the last digits (proves the test can see it).
- Risk: medium. The loaders have four paths; the `point_parameters` loop
  silently mis-points if the 17th pointer is missing (list index past end).
  The `TEST_DEFINES` plumbing must leave the default `test-mojo` command line
  unchanged.

**S1: reference side** (train_gpt2.py, scripts/gen_expected_losses.py,
tests/ pytest). Can run in parallel with S0.

- Interface: `GPT2_UNTIED=1` env in train_gpt2.py: no tie; `LLMC_SKIP_INIT`
  kept on the head; `torch.nn.init.zeros_(self.lm_head.weight)` for
  from-scratch models; `from_pretrained` copies HF's `lm_head.weight`;
  `write_tensor` appends the padded head after `ln_f.beta`; `write_model`
  version +10; `write_state` pads the head grad and sets `header[4] = 1`;
  outputs named `gpt2_124M_untied*.bin` and `gpt2_124M_untied_debug_state.bin`
  (never overwrite the starter pack). Expected 10-step losses for the untied
  run via `gen_expected_losses.py`.
- Tests (pytest, CPU):
  - From scratch, untied with a zero head: every one of the 16 shared tensors
    equals the tied model's init bitwise (RNG alignment).
  - From pretrained: `dwte_tied` versus `dwte_untied + dlm_head` on the same
    batch. Rows with no occurrence in the batch must satisfy
    `dwte_untied == 0` and `dlm_head == dwte_tied` exactly; other rows within
    1e-6 relative.
  - Logits and loss of the untied-from-pretrained model equal the tied
    model's bitwise (the head is a copy).
- Risk: low. The one trap is overwriting `gpt2_124M.bin`.

**S2: untied forward/backward and the head update, W=1** (train_gpt2.mojo,
test_gpt2.mojo, dump_grads_gpt2.mojo, Makefile).

- Interface: `head_tied = False` by default in the untied build; forward
  sites (inventory 23), Bucket 1 (24), `_lm_head_tile`, head AdamW config
  with `l_head` fixed at 1.0 and step counter `t` (no transition yet).
  test_gpt2: under the flag, untied fixtures, 17th check `dlm_head` at
  dwte's tolerance, loops over `NUM_PARAMETER_TENSORS`, untied loss table.
  `verify-untied-cpu`: `mojo -D LLMM_UNTIED_HEAD=1 test_gpt2.mojo cpu`;
  `verify-untied-gpu` with `-D LLMM_NO_TF32=1` as `verify-gpu`.
- Tests:
  - `make verify-untied-cpu` and `verify-untied-gpu`: 17/17 tensors and the
    10-step trajectory against S1's fixtures (`result gate=... status=pass`).
  - Planted, in tests/test_untied_head.mojo (CPU, tiny model, head = wte
    copy): loss bitwise equal to the tied build's printed loss; rows absent
    from the batch: `grads.wte == 0` and `grads.lm_head` equal the tied
    run's dwte rows exactly (the tied values are read from a fixture the
    test writes with the tied build, or recomputed in Python).
  - Mutant: point the K=1 forward at `params.wte` instead of the head with a
    head != wte: the test must fail.
  - Tile coverage: run the planted test at `LLMM_LM_HEAD_VOCAB_TILES` = 1
    and 8 and with `LLMM_LM_HEAD_CHUNKED_CE=1`.
- Risk: medium-high. Bucket 1 has three branches and nine `wte` references;
  missing one gives a run that trains (the encoder grad still flows) and a
  verify that fails only on `dlm_head` or `dwte`. That is why the 17-tensor
  comparison and not the loss curve is the gate.

**S3: ZeRO multi-rank CPU bit-exactness** (new tests/test_zero_untied.mojo
in the style of tests/test_zero_muon.mojo; built with
`TEST_DEFINES="-D LLMM_UNTIED_HEAD=1"`; the file compiles to a skip in the
default build).

- Interface: none new; exercises S2 at W>1. The synthetic checkpoint writer
  writes an untied `.bin` (version 13) with a head distinct from wte.
- Tests, threaded CPU, W=2, 4, 8 (W=8 with L=1 where ranks own nothing):
  - Same batch per rank, 2 micro-steps, 3 steps: ZeRO-1, 2, 3 equal ZeRO-0
    bit-exactly for all 17 tensors, AdamW and Muon.
  - Different batch per rank: step 1 bit-exact for all 17 tensors at ZeRO-2
    and ZeRO-3 (finding 6). If wte is not bit-exact, record the measured
    relative difference and trace it to the row-sparse reduction order before
    adding any tolerance.
  - The head straddles a shard boundary at W=2 and W=8 (check by
    construction: `opt` versus `head_off`), so the segment split is
    exercised.
  - Mutant: route Bucket 1's dest to `flat_base[Parameters.wte]` in the
    untied build: the same-batch test must fail.
- Risk: medium. The threaded harness's nested-parallelism hang
  ([`test_zero_hang.md`](test_zero_hang.md)) applies; reuse
  `tests/_gpu_test_common.mojo` rules for any GPU case.

**S4: `-untie_at` and checkpoint fields** (train_gpt2.mojo,
llmm/checkpointing.mojo, tests/test_checkpointing.mojo,
tests/test_untied_head.mojo, tests/test_zero_untied.mojo).

- Interface: section 4.4: `-untie_at`, env `LLMM_UNTIE_AT`, `s_u`, the
  transition, `head_tied` in the tied phase, step counter `t - s_u`, state
  slots 44/45, resume refusals, log events.
- Tests:
  - `-untie_at 1` in the untied build equals the tied build: 3-step CPU run,
    losses and the 16-tensor model prefix `cmp` equal (as S0, now through the
    runtime path).
  - Transition, planted: 4-step CPU run with `s_u = 2`; dump params right
    after the transition: head == wte bitwise at W=1 and, in S3's harness,
    at W=2/4 for ZeRO-0..3 (ZeRO-3 streaming exercises the tile copy); head
    m and v are zero; the head's first post-untie update equals a
    hand-computed AdamW step-1 update.
  - Resume across the untie point, fp32 CPU, `-x 4`, `s_u = 2`: checkpoints at
    1, 2 and 3; each resume's `model_4.bin` and state files `cmp` equal to the
    continuous run's.
  - Refusals: resume with a different `-untie_at` or `-x`; tied state file into
    the untied build; untied into tied. Each raises before reading payload.
  - `test_state_header_plain_files_unchanged` still passes for tied files
    (slots 44/45 zero).
  - bf16 GPU (`$S/gpu_slot.sh`, `-x 6`, 124M, `s_u = 3`, checkpoint at 2 and
    4): resumed versus continuous step losses within the AdamW resume gap.
- Risk: medium. The transition is a collective at ZeRO-3 and must be called
  on every rank in the same order.

**S5: `-l_head` and W1 integration** (train_gpt2.mojo).

- Interface: section 4.3. If W1 has landed, consume its hook; otherwise add
  `adam_lr_mult` fixed at 1.0 and note it in the commit.
- Tests:
  - Planted lr, head != 0: `-l_head 0` for 3 steps leaves the head
    bit-identical to its initial value. After step 1 every other tensor is
    bit-identical to a `-l_head 1` run's (step 1's gradients do not depend on
    the head's lr); from step 2 on the runs diverge through the forward, so
    only step 1 is compared.
  - `-l_head k`: the head's step-1 update equals k times the `-l_head 1`
    update to within fp32 rounding of the multiply (both with the AdamW
    step-1 closed form, CPU).
  - Default sentinel: no flag equals `-l_head <adam_lr_mult>` bitwise.
  - Tied build: `-l_head 0.5` raises.
- Risk: low.

**S6: ZeRO-3 head on-demand and memory report** (train_gpt2.mojo,
tests/test_zero_untied.mojo).

- Interface: `head_tile_buf` (one tile), `_z3_stream_embed` without the head,
  `_lm_head_tile` gathering under streaming, `untied_head` memory-report key,
  optional `-D LLMM_Z3_HEAD_RESIDENT=1` escape hatch (Q5).
- Tests:
  - S3's ZeRO-3 tests pass with on-demand (they are streaming because the
    harness has a coordinator).
  - After the last update, `_z3_gather_full` and every gathered head tile
    agree with the flat shard bit-exactly.
  - Mutant: gather from `_z3_param_offset(wte)` instead of the head: fails.
  - `LLMM_MEM_REPORT` on a W=2 d6 CPU run, tied and untied, ZeRO-0..3: the
    per-buffer deltas match section 5's model scaled to d6 (exact bytes,
    recompute with the section 8 script).
- Risk: medium (first end-to-end use of the per-tile gather branch).

**S7: tooling and docs** (dump_grads_gpt2.mojo, debug probes,
scripts/export_to_hf.py, scripts/check_checkpoint_biases.py,
max_gpt2_common.py, Makefile infer builds, README).

- Interface: debug probes' fifth whole-tensor class; `dump_grads_gpt2` 17th
  tensor; `$(MOJO_DEFINES)` on `build-infer*`; export_to_hf refuses version
  13/15 with a clear message (full export with `lm_head.weight` and
  `tie_word_embeddings: false` is optional); check_checkpoint_biases handles
  the version.
- Tests: export of an untied file exits nonzero with the message; infer built
  untied generates from an untied 124M checkpoint (`make` target, a few
  tokens, `$S/gpu_slot.sh`).
- Risk: low.

**S8: ranking and measurement** (queue only; `sweeps/w9_*.jsonl`).

- 300-step ranking, seed 42, base recipe from the plan section 1:
  `w9_tied` (today's binary); `w9_untied0` (untied, zero head, no lr split);
  `w9_untied0_split` (with W1: `-l_adam` from W1's best, `-l_head` low).
- 2000-step: untied-from-0 versus `-untie_at 0.667`.
- Every arm reports peak memory (`LLMM_MEM_REPORT` exact total and the
  driver figure) and step time. The untied build needs its own binary; per the
  protocol each job script rebuilds with its `-D` flags first.

## 7. Open questions for the coordinator

1. **Head position.** Index 16 (end), recommended (3.3). Alternative: after
   `wte`. Rejected: shifts every offset and puts both big tensors on rank 0.
2. **Model-file marker.** New versions 13/15, recommended: every existing
   reader refuses loudly (finding 5). Alternative: a flag in header slot 8 with
   versions 3/5; old readers would load the file as tied and drop the head.
3. **Moments at the untie point.** Fresh moments with the head's own step
   counter `t - s_u`, recommended (4.4). Alternative: copy `wte`'s m and v
   (needs a cross-rank fp32 gather at ZeRO >= 1). What record 53 does is
   UNVERIFIED; if someone checks and it copies, S4 can add `-untie_moments copy`.
4. **Master at the untie point (bf16).** Promote the bf16 head params,
   recommended (matches resume, no fp32 collective). Cost: the head starts
   from the bf16-rounded wte, one rounding of wte's fp32 master.
5. **ZeRO-3 head residency.** On-demand per tile by default in the untied
   build, recommended (169 vs 298 MiB window, finding 7), with
   `-D LLMM_Z3_HEAD_RESIDENT=1` as an escape hatch. Its step-time cost is
   unmeasured (two extra gathers per tile per step).
6. **`-l_head` default.** Inherit the AdamW-range multiplier, recommended, so
   "untied" alone changes one thing. Alternative: default to the speedrun
   ratio. That conflates two effects in one arm.
7. **Weight decay on the head.** Inherit `-c`, recommended; W4 owns decay
   policy and the speedrun's "no decay on Adam groups" applies to embed too.
8. **Head init options.** Zero only for from-scratch, recommended for the
   first ranking; a later `-D LLMM_HEAD_INIT=normal` (draws after all other
   tensors, so the shared tensors' stream is unchanged) if someone wants the
   GPT-2-style head as a control.
9. **`-untie_at` encoding.** Fraction of `-x`, stored as the step `s_u`, strict
   equality on resume, recommended. A resumed run that extends `-x` must pass
   an `-untie_at` that maps to the same step; an absolute `-untie_step` flag
   would avoid that and could be added if it bites.
10. **`STATE_VERSION`.** Keep 1, recommended: slots 44/45 are zero in tied
    files (the slots 41-43 precedent), and every layout mismatch is caught by
    the slot check or the exact length check.
11. **Memory-report key in the tied build.** Emit `untied_head` only in the
    untied build, recommended, so tied logs stay byte-identical.
12. **Allocation during the tied phase.** Allocate the head from step 0,
    recommended (4.4). The tied phase of an untie-at-2/3 run pays the full
    section 5 cost from the start.
13. **Sharded-Muon imbalance.** Accept the second heavy rank for now; fold it
    into the S7 memory-aware ownership follow-up of the sharded-Muon design.
14. **Merge order.** The plan puts W9 after W1 and W5. W9 does not need W5's
    code (the head zero-init is intrinsic, 4.1); it needs W1 only for S5. S0
    to S4 can start before W1 lands; recommended.

## 8. Reproduction of the numbers

The model script mirrors `_compute_param_sizes` (sizes, `opt = round_up(ceil(N/W),
align)` with align 8 for bf16 and 4 for fp32, `padded = opt*W`),
`_adamw_ranges` (5 pairs), the Muon ownership `(k*L+l) mod W` with
`S = round_up(max O_r, 8)`, and `A_r = |[r*opt, (r+1)*opt) ∩ adamw_ranges|`.
The script (run with `python3`, no dependencies) is reproduced below. Checks against
published tied values: fp32 W=2 ZeRO-0 params 474.838 MiB (accounting doc
table), Muon compact `v` 150.8 MiB = 158.2 MB and sharded-Muon W=4 rank 0
31,118,976·12 + 21,233,664·10 = 585.8 MB (Muon doc). The ZeRO-3 tile is
`ceil(ceil(50304/8)/128)*128 = 6400` rows = 18.75 MiB fp32.

```python
# W9 memory model: per-rank model-proportional buffers, tied (16) vs untied (17).
# Mirrors train_gpt2.mojo _compute_param_sizes, _adamw_ranges, enable_muon
# ownership (k*L+l) mod W, S7 trim (m/master = A_r), compact v.
import math
L, C, Vp, T = 12, 768, 50304, 1024
MiB = 2**20
def sizes(untied):
    s = [Vp*C, T*C, L*C, L*C, L*3*C*C, L*3*C, L*C*C, L*C, L*C, L*C, L*4*C*C, L*4*C, L*C*4*C, L*C, C, C]
    if untied: s.append(Vp*C)
    return s
def off(s, i): return sum(s[:i])
def adamw_ranges(s):
    N = sum(s)
    r = [(0, off(s,4)), (off(s,5), off(s,6)), (off(s,7), off(s,10)), (off(s,11), off(s,12)), (off(s,13), N)]
    return r
def shard(N, W, align):
    if W == 1: return N, N
    base = -(-N // W); cs = -(-base // align) * align
    return cs, cs * W
def tile_rows():
    t = -(-Vp // 8); t = -(-t // 128) * 128; return min(t, Vp)
def A_r(s, W, r, opt):
    lo, hi = r*opt, min(r*opt+opt, sum(s))
    return sum(max(0, min(b,hi)-max(a,lo)) for a,b in adamw_ranges(s))
def owner_S(W):
    strides = [C*C]*3 + [C*C, 4*C*C, 4*C*C]
    O = [0]*W
    for k in range(6):
        for l in range(L):
            O[(k*L+l) % W] += strides[k]
    return max(O)
def adamw_mem(untied, W, Z, prec, head_ondemand=False):
    s = sizes(untied); N = sum(s); p = 2 if prec == 'bf16' else 4
    align = 8 if prec == 'bf16' else 4
    cs, padded = shard(N, W, align)
    opt = cs if (W > 1 and Z >= 1) else N
    stream = W > 1 and Z >= 3
    d = {}
    d['params'] = (opt if stream else padded) * p
    d['grads'] = (opt * p) if (W > 1 and Z >= 2) else padded * p
    d['m+v'] = 2 * opt * 4
    d['master'] = opt * 4 if prec == 'bf16' else 0
    d['comm'] = 2 * cs * p if W > 1 else 0
    if stream:
        win = s[0] + s[1] + s[14] + s[15]
        if untied: win += (tile_rows()*C if head_ondemand else s[16])
        d['z3win'] = win * p
    else: d['z3win'] = 0
    return d
def muon_mem(untied, W, Z, prec):
    s = sizes(untied); N = sum(s); p = 2 if prec == 'bf16' else 4
    align = 8
    cs, padded = shard(N, W, align)
    sharded = W > 1 and Z >= 1
    opt = cs if sharded else N
    if not sharded:
        Ar = sum(b-a for a,b in adamw_ranges(s))
        return {'m': N*4, 'v': Ar*4, 'master': N*4 if prec=='bf16' else 0}, [Ar]
    Ars = [A_r(s, W, r, opt) for r in range(W)]
    S = -(-owner_S(W)//8)*8
    rows = []
    for Ar in Ars:
        rows.append(Ar*4*(3 if prec=='bf16' else 2) + S*(p+p+4+(4 if prec=='bf16' else 0)))
    return rows, Ars
H = Vp*C
print("N16", sum(sizes(False)), "N17", sum(sizes(True)), "H", H, "H fp32 MiB", H*4/MiB)
print("\nAdamW per-rank model-proportional MiB (excl. grad pool, activations): tied -> untied (delta)")
for prec in ('bf16','fp32'):
    for W in (1,2,4,8):
        for Z in ((0,) if W == 1 else (0,1,2,3)):
            a = adamw_mem(False, W, Z, prec); b = adamw_mem(True, W, Z, prec)
            ta, tb = sum(a.values())/MiB, sum(b.values())/MiB
            extra = ""
            if W > 1 and Z == 3:
                c = adamw_mem(True, W, Z, prec, head_ondemand=True); tc = sum(c.values())/MiB
                extra = f" | head on-demand: {tc:9.1f} (+{tc-ta:6.1f})"
            dd = {k: (b[k]-a[k])/MiB for k in a}
            print(f"{prec} W={W} Z={Z}: {ta:9.1f} -> {tb:9.1f} (+{tb-ta:6.1f})  by class {dict((k, round(v,1)) for k,v in dd.items() if v)}{extra}")
print("\nTied baseline check vs accounting doc (fp32 W=2 Z0 params):", adamw_mem(False,2,0,'fp32')['params']/MiB)
print("\nMuon W=1 bf16 optimizer MiB:")
for u in (False, True):
    d, Ar = muon_mem(u, 1, 0, 'bf16'); print(" untied" if u else " tied", {k: round(v/MiB,1) for k,v in d.items()}, "sum", round(sum(d.values())/MiB,1), "A", Ar)
print("\nSharded Muon (S7) bf16 per-rank optimizer state MiB (A_r*12 + S*12):")
for W in (2,4,8):
    for u in (False, True):
        rows, Ars = muon_mem(u, W, 1, 'bf16')
        print(f" W={W} {'untied' if u else 'tied  '} A_r={Ars} MiB={[round(x/MiB,1) for x in rows]}")
print("tile rows", tile_rows(), "tile MiB fp32", tile_rows()*C*4/MiB)
```

## S2 implementation notes (2026-10-30)

What S2 landed (W=1, ZeRO-0/1 layouts): the LM-head forward (all three
branches) reads `_head_param_ptr()`; Bucket 1 writes `_head_grad_ptr()` and
re-points the gradient pool through `_set_head_grad_ptr()` (never `grads.wte`
once untied, so the encoder stays the only writer of dwte);
`_lm_head_wte_tile` is now `_lm_head_tile` and gathers `_head_t()`. The head
has its own `AdamWConfig` (`_head_config`: lr * `-l_head`, sentinel < 0 means
`adam_lr_mult`, so the default trains every AdamW tensor at one rate) at all
five `update()` sites (`_adamw_flat` splits the slice at the head offset),
`_update_hybrid` and `_update_sharded_muon`. Step counter is `t` (no tie phase
yet). The grad norm covers all 17 tensors once the head trains; body-only only
while `head_tied`. `-l_head 0` (frozen head) is accepted.

Decisions and caveats:
- ZeRO >= 2 with a trained head is refused at the top of `forward()` with a
  clear message (bucketing and the ZeRO-3 window re-point `grads.wte` /
  `params.wte` and know nothing of the head until S6). The idle head
  (`head_tied`) is unaffected, which is why the S0 multi-rank tests still run.
- `LLMM_UNTIED_IDLE_HEAD=1` is a test hook that keeps the head tied and idle
  (`scripts/check_untied_idle.py` uses it; `-untie_at` in S4 supersedes it).
  Note an idle head keeps its *loaded* value in the model file, not the trained
  wte (S4 must write head := wte for tied-phase checkpoints).
- Gates: `make verify-untied-cpu/gpu` (17 tensors; the loss trajectory cannot
  tell tied from untied, see the Makefile comment), `make check-untied-idle`,
  `make check-untied-head` (head moves, m/v nonzero, idle head frozen with
  weight decay > 0). A mutant pointing the K=1 forward at `params.wte` passes
  the 17-tensor check at step 0 (head == wte there) and fails the 10-step
  trajectory (`steps failed=9`), so the trajectory is not useless for the head.

S2 generation 2 (tests and GPU gates):
- `make test-mojo TEST_DEFINES="-D LLMM_UNTIED_HEAD=1"` first failed in four
  files: `test_zero_equivalence`, `test_zero_muon` and `test_qk_clip` train at
  ZeRO-2/3 with the head live (refused until S6), and `test_adam_lr_mult`
  segfaulted because its comparison buffers held 16 tensors while the model has
  17. The ZeRO-2/3 cases now run the idle head (`model.head_tied = True`, for
  reference and candidate alike) under `comptime if UNTIED_HEAD`; ZeRO-0/1
  cases keep the trained head. Result: tied `tests result=pass files=29
  passed=250`, untied `passed=250`, both failed=0.
- Planted test `test_untied_head_gradient_split_matches_tied`
  (tests/test_untied_head.mojo): untied model vs the same build with the head
  idle, from a tied .bin. Loss bitwise equal; absent rows dwte == 0 and
  dlm_head == tied dwte; present rows dwte + dlm_head == tied dwte within 1e-6.
  Mutant (`_head_grad_ptr` returns `grads.wte`) fails it with "absent row has
  encoder gradient in wte".
- 5-step bf16 smokes (B=4, T=1024): tied loss 3.609831 3.668198 3.438826
  3.279619 3.634629; untied 3.609831 3.670668 3.440166 3.283439 3.632877
  (identical at step 1, diverge after the head trains separately).
- S3 inherits: ZeRO-1 at W>1 with the head active is untested; ZeRO >= 2 is
  refused until S6.

S2 independent verification (2026-10-30, separate agent, rebuilt and rerun):
- Code read of `ea49594..e9f4051`: every LM-head forward/backward branch
  (K=1, vocab-tiled, chunked CE) reads `_head_param_ptr()` / `_lm_head_tile`
  and writes `_head_grad_ptr()`; remaining `params.wte`/`grads.wte` uses are
  the encoder, the encoder bucket, the ZeRO-3 embed window (refused while the
  head trains), and the debug non-finite probe (scans `wte` only; debug
  build). The five `update()` sites pass the right `flat_lo`; Muon ranges
  and sharded segments never straddle the head offset. Tied build: helpers
  fall through to `wte` and the 4-config tied 3-step CPU run (AdamW gc 1,
  gc 1e-3, gc 1e-3 + wd 0.1, Muon gc 1e-3) is `cmp`-identical (model and
  state) to the fe3312a binary.
- Gaps found and closed with tests: nothing tested `-l_head` or the
  17-tensor clip norm (new `test_untied_head_lr_mult`,
  `test_untied_grad_norm_covers_head`); the planted split test's inputs were
  exactly vocab tile 0, so a tile-0 misroute into `grads.wte` passed it
  (inputs now leave absent rows in every tile). Mutants: `_head_grad_ptr ->
  grads.wte` fails the split test, lr-mult and norm tests and verify-untied-cpu;
  body lr on the head in `_adamw_flat` fails `test_untied_head_lr_mult` (and a
  2-step `-l_head 0` run moves 19.3M head elements vs 0); tile-0 misroute fails
  the split test and verify-untied-cpu; 16-tensor norm fails
  `test_untied_grad_norm_covers_head` only (verify-untied-cpu passes it).
- Threaded CPU W=2 smoke (d6 from a trained tied start, 3 steps, gc 1e-3,
  wd 0.1): ZeRO-1 == ZeRO-0 bit-exact for all 17 tensors, AdamW and Muon.
  W=2 vs W=1 with the same -d differs by ~1.7e-3 max abs in the tied build
  too (the two runs do not see the same micro-batch order), so that pair is
  not an equivalence check.
- `LLMM_LOG_LEVEL=quiet` prints the `result` verdict of test_gpt2 with no
  fields (gate/status are NORMAL-level fields); exit codes still gate.
## S3 implementation notes (2026-10-30)

tests/test_zero_untied.mojo (untied build; SKIP lines in the tied build), 18
tests, threaded CPU, version-13 file with an independent random head
(head != wte, nonzero). `pixi run` of the built binary: 18 passed.

- ZeRO-1 == ZeRO-0, `==` per element, all 17 tensors, different batch per
  rank, 1 and 3 steps, AdamW and Muon, W=2/4/8 (L=2) and W=8 L=1. Every wte
  and head element must also have moved from the planted init.
- Finding 6, measured: ZeRO-1 vs ZeRO-0 is bit-exact for all 17 tensors at
  step 1, wte and head included (zero residual). That is the all-reduce of
  the full flat gradient, so it does not exercise the finding's actual claim
  (bucketed reduce-scatter keeps lm and encoder parts separate): ZeRO >= 2
  with a trained head is refused until S6, so the ZeRO-2/3 step-1
  different-batch check is NOT run. S6 must add it (tests assert the refusal
  today, so they will fail loudly when S6 lifts it).
- Ownership: opt, per-rank AdamW segments and head overlap vs values computed
  in the test from the layout. L=2: the whole head is on rank W-1 at W=2/4/8
  (asserted: exactly one head rank, `(W-1)*opt <= N16`). W=8 L=1: ranks 6 and
  7 split the head (asserted: `7*opt` strictly inside the head, both sides
  nonempty, sum == H).
- Refusal: ZeRO-2 and ZeRO-3 with the head active raise the S2 error on every
  rank. Idle head at ZeRO-2/3 == ZeRO-0 idle, same batch, accum 2, Muon, W=2
  and 4, bit-exact; the idle head keeps its planted values.
- W=8 same-batch idle-head ZeRO-2/3 is NOT bit-exact (1 ulp, first at wte[18]
  and wpe[9]): 8 identical gradients are not an exact multiple under the
  reduction orders, a property of the harness (tied-phase layout), not of the
  head. Left out of this file; test_zero_muon covers W=8 at its tolerance.
- Mutant (temporary, reverted): second `lm_head` entry of `_adamw_ranges`
  shifted by one row (`+ channels`), i.e. the Muon-path head update skips row
  0. ZeRO-1 == ZeRO-0 still PASSES (both layouts use the same ranges), which
  is why the ownership and moved checks exist: 4 ownership tests fail on the
  segment table and the 4 Muon ZeRO-1 tests fail with "head element did not
  move at rank 0: lm_head[0]". AdamW tests are unaffected (that path uses
  `_adamw_flat`). The design's Bucket-1 mutant is unreachable until S6.
- No trainer change.
## S4 implementation notes (2026-10-30)

What landed (branch `mn/w9s4`, W=1; ZeRO-0/1 layouts): `-untie_at <frac>` (env
`LLMM_UNTIE_AT`, untied build only; the tied build refuses the flag). The
trainer computes `s_u` once from the final step count: `0` (default, also an
unset flag) = untied from construction, i.e. S2 behaviour with no `wte` copy;
`0 < f < 1` = `max(1, floor(f * steps))`; `>= 1` = `-1`, never. `GPT2.untie_step`
holds it and `head_tied = (untie_step != 0)`.
- Transition (`GPT2.untie_head`) runs at the top of loop iteration `s_u`, after
  that step's checkpoint write and before its training, so a checkpoint at
  `s_u` is pre-untie. It copies `wte` into `lm_head` on the device, promotes the
  head's fp32 master from the copied params (bf16 builds; same promotion as
  `load_checkpoint`), clears `head_tied` and emits
  `untie head_from=wte master=promoted moments=fresh step=<s_u>`. m and v are
  untouched (still exactly zero: the head was skipped).
- Head Adam counter `t - s_u` via `_head_adam_step`, applied at the three head
  update sites (`_adamw_flat`, `_update_hybrid`, `_update_sharded_muon`).
- `write_checkpoint` writes `head := wte` into the model file while `head_tied`
  (host copy before `write_model_checkpoint`), so `-o` output and every
  tied-phase checkpoint run as the trained model in infer/eval/export.
- State slots 44 (untied layout) / 45 (`s_u`) are written and checked at the top
  of `load_checkpoint`, before any payload: a different `-untie_at` or `-x`, a
  tied state in the untied build and an untied state in the tied build are all
  refused with a message. Tied-build files keep zeros in both slots.
- After a resume `head_tied = False` only if `resume_step > s_u`; otherwise the
  transition replays from the loaded state. The `resume` event gains
  `head_tied` and `untie_step`; the setup event gains `untie_at`, `untie_step`
  (`l_head` as before). The `LLMM_UNTIED_IDLE_HEAD` hook is removed;
  `-untie_at 1.0` replaces it in `check_untied_idle.py`/`check_untied_head.py`
  (the idle model file now carries head := wte instead of the stale value).
- Refused: a tied phase at ZeRO >= 2 (the transition collective lands with
  S6). bf16 at W > 1 with ZeRO >= 1 raises in `untie_head` (shard-relative
  master promotion is not written; S3/S6).

Gates (all fp32 CPU unless stated; `scripts/check_untie_at.py`, `make
check-untie-at`, d6 from scratch, B=2 T=64, `-c 0.1`, 6 steps, `-untie_at 0.5`
so `s_u = 3`, checkpoint every step; AdamW and Muon):
- Tied build vs the `fe3312a` binary: model and state files of steps 1-3
  identical (`cmp`), both optimizers.
- `-untie_at 0` and the no-flag run equal the S2 binary (built from `e9f4051`):
  model file identical, state file identical except header slots 44/45, step
  losses and norms identical.
- `-untie_at 1.0` model prefix equals the tied build's and the head block equals
  `wte`; `make check-untied-idle` and `make check-untied-head` pass with the new
  contract.
- Tied-phase files: `model_1..3` have head == wte bitwise (version 13);
  `model_4..6` have head != wte; head m/v exactly zero at step 3 and nonzero at
  step 6.
- Bias correction: the step 3 -> 4 head update recomputed in the script from
  the step-3 head and the step-4 m/v matches an AdamW step with `t = 1`
  (max abs error 5.7e-9; `t = 4` would be off by 1.3e-4), and `v == (1-b2)g^2`.
- Resume across the transition (resume at step 2 = before `s_u`, at step 4 =
  after): every later `model_k.bin` and `state_k_0.bin` is byte-identical to the
  continuous run, AdamW and Muon (measured gap: 0 differing files of 8 / 4).
- Refusals (exit 1, message before payload): `-untie_at 0.7`, `-x 8`, no flag on
  an `s_u = 3` state, tied state into the untied build, untied state into the
  tied build, and `-untie_at` in the tied build.

GPU (one `gpu_slot.sh` chain): `make test-mojo` tied `tests result=pass
files=29 passed=251 skipped=1 failed=0`, untied (`TEST_DEFINES="-D
LLMM_UNTIED_HEAD=1"`) the same counts (+1 new case, the slot 44/45 round trip).
5-step bf16 smoke (B=4, T=1024, `-untie_at 0.6`, `s_u = 3`): losses 3.609831
3.668198 3.438826 3.279619 3.624149 (steps 1-4 equal the tied run, step 5
diverges; S2's tied run had 3.634629 there), `untie head_from=wte
master=promoted moments=fresh step=3` printed between the step-3 checkpoint and
step 4; the model files are version 15 with head == wte at steps 1-3 and
head != wte at 4-5.

S5/S7 inherit: `_head_config` still carries no step counter (the offset lives
in `_head_adam_step`, so S5 must not fold it into the config); the sharded
Muon path (`_update_sharded_muon`) has the offset but is untested at W>1;
the memory report does not yet print `untie_step`.

## S6 implementation notes (2026-10-30)

What S6 landed (train_gpt2.mojo, tests/test_zero_untied.mojo, Makefile):
- The S2 refusal of ZeRO >= 2 with a trained head is gone. Bucket 1 already
  wrote `_head_grad_ptr()` (re-pointed to the pool by `_set_head_grad_ptr()`) and
  reduce-scattered to `flat_base[_head_t()] + t0*C`; nothing else was needed for
  ZeRO-2 (the encoder's row-sparse wte bucket is untouched, the pool bound is
  unchanged: `grad_pool_elems` is identical tied/untied at d6).
- ZeRO-3 streaming: the head is not in the embed window. `_lm_head_tile` gathers
  one vocab tile (`flat_starts = [_z3_param_offset(lm_head) + t0*C]`) into
  `wte_tile_buf` (the pre-existing on-demand buffer, now allocated in the untied
  build with `Z3_HEAD_ONDEMAND = UNTIED_HEAD and not Z3_HEAD_RESIDENT`). Used by
  the forward, the chunked-CE recompute and the backward weight-gradient GEMM.
  The K=1 paths (single GEMM) now also go through `_lm_head_tile(0, V_p)`; in the
  tied build that is the same pointer, so the tied binary is unchanged. Only a
  trained head is gathered: with `head_tied` the head reads the resident wte.
  `LLMM_Z3_WTE_ONDEMAND` stays comptime-refused in the untied build.
- `-D LLMM_Z3_HEAD_RESIDENT=1`: head appended to the embed window
  (`params.lm_head` re-pointed in `_z3_stream_embed`), no per-tile collective.
  The whole test file passes in this build too (the post-update tile check is
  skipped there: the window is only refreshed by the next forward).
- `-untie_at` under ZeRO-2/3 (merged from S4, whose startup refusal is lifted):
  `untie_head()` is now `_untie_apply_range`-based. ZeRO-3 streaming gathers wte
  tile by tile (bounded by the tile buffer) and each rank keeps the overlap with
  its shard (params and bf16 master); ZeRO-1/2 keep full params, copy locally and
  set the master slice; the old "bf16 at W>1" error is gone. The bf16 multi-rank
  master path is NOT exercised by any gate (single GPU; bf16 CPU codegen is
  unsupported on this aarch64 host): it reuses the fp32-tested index logic and
  the W=1 bf16 promotion code.
- Memory report: `head_tile_buf` joins `buffers` and `classes.z3_windows` (only
  when a tile gather is compiled in, so the tied report is byte-identical), and an
  `untied_head` object (head_elems, flat offset, elements in this rank's shard,
  params/grads/m/v/(master) bytes, tile rows and bytes, resident-window bytes) is
  emitted in the untied build (it replaces S7's flat `untied_head:1` flag, whose job its presence does; S7's top-level `untie_step` and `head_tied` stay). Like `tensors` it is a breakdown of buffers already
  counted, not additive.

Finding 6, measured: step 1 AND step 3 are bit-exact (`==`, zero residual) on all
17 tensors, wte included, for ZeRO-2 and ZeRO-3 against ZeRO-0, different batch
per rank, AdamW and Muon, W=2/4/8 (L=2) and W=8 L=1 (head straddling ranks 6/7).
The prediction held; there was nothing to explain. (The tied build's step-1 wte
exemption exists because wte there carries lm + encoder parts reduced in a
different order.) After the last ZeRO-3 update `_z3_gather_full` equals each
rank's flat shard (sharded Muon: outside the four matrix tensors, whose owner
space is checked via the ZeRO-0 comparison) and every gathered head tile equals
the ZeRO-0 head, bit-exactly.

Mutants (temporary, reverted): tile gather from `_z3_param_offset(wte)`: the 8
ZeRO-3 tests fail, ZeRO-2 unaffected. Bucket 1 K>1 destination `flat_base[wte]`
instead of the head: all 8 ZeRO-2 and 8 ZeRO-3 tests fail. `-untie_at` tile
gather from the head offset instead of wte: the 4 ZeRO-3 transition tests fail.
Variants (32 tests, before the transition tests): the file also passes with `-D LLMM_Z3_HEAD_RESIDENT=1`,
`-D LLMM_LM_HEAD_VOCAB_TILES=1` (K=1 through `_lm_head_tile`) and
`-D LLMM_LM_HEAD_CHUNKED_CE=1`. The default tests run K=8 (tile = 8 rows of the
64-row toy vocab), so the on-demand branch walks 8 tiles per pass.

Untie transition tests (new): tied step, `untie_head()`, two trained steps at
ZeRO-2/3 == ZeRO-0, same batch per rank, accum 2, W=2/4, AdamW and Muon. W=8 is
not included: same-batch tied phase at W=8 is 1 ulp off in wte (the known harness
property noted under S3), unrelated to the transition.

Memory (d6: L=6, C=384, V_p=50304, T=1024, fp32, W=2, AdamW, `LLMM_MEM_REPORT`
steady phase, rank 0, bytes). Delta = untied minus tied; model = section 8
script rescaled to d6 (tile 6400 rows = 9,830,400 B). Every delta matches exactly:

| ZeRO | params_buf | grads (full+shard) | m+v | z3 window + tile buf |
|---|---|---|---|---|
| 0 | +77,266,944 | +77,266,944 | +154,533,888 | +4 (stub) |
| 1 | +77,266,944 | +77,266,944 | +77,266,944 | +4 (stub) |
| 2 | +77,266,944 | +38,633,472 | +77,266,944 | +4 (stub) |
| 3 | +38,633,472 | +38,633,472 | +77,266,944 | +9,830,400 |

grad pool, param window and comm scratch are identical tied/untied at every stage.
ZeRO-3 window: tied 78,842,880 B (wte+wpe+ln_f), untied 88,673,280 B; growth is
exactly one tile (the whole head would have been 77,266,944 B). The 4-byte
entries are the 1-element placeholder reported as-is.

Gates on the merged tree (mn/w9 at a602274 merged in): see the final report;
`check-untie-at` was run AdamW half only (the Muon half is S4's and ~1 h of CPU):
`result gate=untie-at status=pass`.

Gates (earlier list): tied 3-step CPU d6
`cmp` identical (AdamW, Muon) against the fe3312a binary; `verify-cpu`,
`verify-untied-cpu` at K=8 and K=1, `check-untied-idle`, `check-untied-head`,
`check-untie-at`; GPU: `verify-gpu`, `verify-untied-gpu`, `make test-mojo` tied
and untied.

Not verified: multi-rank on real GPUs (one GB10; threaded CPU is the proof of
the collectives' logic), bf16 multi-rank untie master slice, Metal.

## S7 implementation notes (2026-10-30)

Tooling and docs. Branch mn/w9s7, gated on the tree after merging mn/w9 (S4).

- Label and MFU: `flops_params = num_parameters - V_p*C` in the untied build
  (train_gpt2.mojo main). The model label (`gpt2-124M`) and `estimate_mfu` use
  it; the `params=` field stays the true stored count (163,109,376). Formula:
  flops/token = 6*N16 + 6*L*C*T, because the LM-head GEMM is one V_p*C matmul
  whether its weight is wte or lm_head, and the untied wte is lookup-only.
  Per-step FLOPs are therefore identical tied vs untied. Old formula (6*N17)
  would inflate by 1.288x at 124M/T=1024 (1,035.3M vs 803.5M flops/token).
  5-step bf16 smoke, B=4 T=1024, GB10: tied `mfu_pct` 5.5 18.5 17.8 18.3 17.6,
  untied 5.6 18.2 18.1 18.2 18.1 (step times 142-150 ms both); the old formula
  would have shown ~23.3 for the untied step 5. Label before: `gpt2-163M`,
  after: `gpt2-124M`.
- Memory report: untied builds add `untied_head`, `untie_step`, `head_tied` to
  the schema-3 line (tied line unchanged).
- nanscan probe: slot 4 = `lm_head` (fifth whole-tensor class), per-layer
  slots start at `N_WHOLE`; in the untied build the backward order is
  lm_head first and `wte` (encoder only) beside `wpe` last. The probe only runs
  at WORLD_SIZE > 1 (the call is inside the multi-rank branch), so it was
  compile-verified (`WORLD_SIZE=2 -D LLMM_UNTIED_HEAD=1 -D
  LLMM_DEBUG_TENSOR_NANSCAN=1`) but NOT run: this box has one GPU.
- `dump_grads_gpt2`: under the flag it loads `gpt2_124M_untied{_bf16}.bin` and
  the untied debug state and writes `lm_head.bin` ([V_p, C], 149 files).
  `make verify-untied-dump` builds an fp32 untied dump and
  `scripts/check_dump_lm_head.py` compares dlm_head with the reference using
  test_gpt2's criterion: maxdiff 3.2e-4 against threshold 0.51, l2_ratio
  1.0000. Planted check: copying `wte.bin` (the encoder-only gradient) over
  `lm_head.bin` fails (maxdiff 10.06, l2_ratio 0.027). The checker's first
  version read the grads as the file tail and failed on garbage: the debug
  state carries 98 MB after the gradients, so it now reads exactly N floats.
- Makefile: `$(MOJO_DEFINES)` on all three `build-infer*` (W8 already had it
  on bf16; same text), `build-infer-untied` / `infer-untied` (separate binary
  `build/infer_gpt2_bf16_untied`, because make does not rebuild on a flag
  change). Generation from the untied checkpoint that the 5-step untied smoke
  wrote (`model_5.bin`, version 15): `make infer-untied CHECKPOINT=... ARGS='16
  1337'` printed 16 tokens of text.
- Refusals: `scripts/export_to_hf.py` (SystemExit, rc 1, message names version
  and why), `scripts/check_checkpoint_biases.py`, `max_gpt2_common.py`
  (ValueError) refuse version 13/15. Tied path unchanged: export of
  `gpt2_124M_bf16.bin` before/after produces a byte-identical
  `model.safetensors` and the same log modulo temp dir names;
  `check_checkpoint_biases.py` output on the tied file diffs empty. Full
  untied export (`lm_head.weight`, `tie_word_embeddings: false`) NOT done.
- Log levels (finding of the S2 verifier): `Log._field_level` makes a QUIET
  event show its NORMAL-level fields, so verdict lines keep their keys.
  Before: `result` (test_gpt2 at `LLMM_LOG_LEVEL=quiet`); after: `result
  gate=gpu status=pass`. The trainer's `done steps=.. tok_s=..` line benefits
  the same way. VERBOSE-level fields stay hidden at quiet.
  `scripts/test_log.py` has no per-field levels, so no mirror change.
- Gates (all on the merged tree): tied 3-step CPU d6 AdamW `cmp` of model and
  state against the fe3312a-era base binary: identical. `make verify-cpu` and
  `make verify-untied-cpu`: `result gate=cpu status=pass`. In one gpu_slot
  chain: `make verify-gpu`, `make verify-untied-gpu` (also at quiet), both
  `result gate=gpu status=pass`; `make test-mojo` tied and with
  `TEST_DEFINES="-D LLMM_UNTIED_HEAD=1"`: `tests result=pass files=30 passed=271
  skipped=1 failed=0` both.

## AI use statement

Written with AI assistance (Claude Code, Opus agent), directed by Evan Owen.
The inventory anchors were read from the code at `fe3312a` by the same agent;
the memory figures come from a Python model it wrote and checked against the
published tied-build figures. The S4 slice was implemented by a Claude Code
(Sonnet) agent under the same direction.
