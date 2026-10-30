# Muon + MuonClip implementation campaign (2026-10-29)

Campaign report for phases 1, 2 and 3 of
[`muon_muonclip_implementation_plan.md`](muon_muonclip_implementation_plan.md).
Sections 1 to 7 are the 2026-10-29 report for phases 1 and 2 (one-line notes
mark text that phase 3 superseded); phase 3, the hyperparameter sweep and the
small trainer changes of 2026-10-30 are sections 8 to 10. Phases 1 and 2:
the opt-in Muon optimizer (`-opt muon`) and QK-clip (`-qk_clip tau`), built by
parallel agent tracks and merged on branch `muon-int` (functional merge
`f646bbe`; the perf pass later brought it to `f7a5b5d`). The default optimizer
is still AdamW and the AdamW path is unchanged (section 4). Results of the
convergence comparison are in
[`muon_vs_adamw_124m_2026-10-29.md`](muon_vs_adamw_124m_2026-10-29.md); this
report does not quote them. Step-time work is section 6.

## 1. What was built

| Piece | Where | Notes |
|---|---|---|
| Muon update | `llmm/muon.mojo` | `muon_update` custom op: momentum + Nesterov kernel, per-matrix Frobenius normalize, 5-step Newton-Schulz, apply kernel |
| Newton-Schulz | `llmm/muon.mojo` | coefficients (3.4445, -4.7750, 2.0315), bf16, strided-batched cuBLASLt, one batch per tensor group (batch count = L) |
| QK-clip | `llmm/qk_clip.mojo`, `llmm/attention.mojo` | softmax kernel tracks per-(layer, head) max post-scale logit; `qk_clip_apply` rescales hot heads |
| Trainer wiring | `train_gpt2.mojo` | flags, `enable_muon` / `enable_qk_clip`, hybrid `update()`, logging |
| Checkpoint | `llmm/checkpointing.mojo` | `optimizer_kind` in state-file header slot 40 |
| RNG registry | `llmm/rng_device.mojo` | `SR_MUON_STREAM = 6` |
| Tests | `tests/test_muon.mojo`, `tests/test_muon_equivalence.py`, `tests/test_muon_crosscheck.py`, `tests/test_muon_crosscheck_keller.py`, `tests/test_qk_clip.mojo`, `tests/reference.py`, `tests/kernels/muon.py` | see sections 4 and 5 |

### Muon

- Update rule is the plan's: fp32 momentum `M` (reuses `m_memory`; `v_memory`
  stays allocated and unused for Muon ranges), Nesterov `U`, normalize by the
  Frobenius norm, 5 NS iterations in bf16, then
  `W -= lr * (0.2*sqrt(max(rows, cols)) * O + wd * W)`. The Moonlight RMS
  scale means `-l` and `-c` are shared with AdamW; there is no second LR.
- Tensor groups: Q, K, V (the `qkv_weight` row blocks, each orthogonalized as
  its own C x C matrix), attn-proj, fc, proj. Tall matrices (fc) are read as
  column-major X^T so the transpose is free (a layout choice, no copy).
  Scratch comes from `persistent_device_buffer`, whose allocation is fixed at
  first call, so the scratch names carry their sizes.
- Everything else (wte, wpe, biases, LN gamma/beta) runs the unchanged AdamW
  kernel over the five non-matrix flat ranges, plus one `muon_update`.
- Targets: the GPU path needs cuBLASLt (NVIDIA). On Metal/AMD it raises
  "GPU path needs cuBLASLt (NVIDIA)" at the first `update()`, not at startup.
  CPU has a naive fp32 host Newton-Schulz: correct (tested), but about
  124 s/step at 30M params, so a reference and not for training.
- Under `LLMM_SR_MASTER` the param-store stochastic rounding draws from rng
  stream 6.
- Scope (phases 1 and 2): `enable_muon` raised when WORLD_SIZE > 1 together
  with ZeRO stage >= 1 or ZeRO-3 streaming. **Superseded by phase 3 (section
  8): ZeRO-1/2/3 are supported.** At WORLD_SIZE = 1 the `-z` flag is a no-op
  (losses are identical for `-z 0` and `-z 3`).
- `-mu` defaults to 0.95 and is not range-checked. `-l` and `-c` are shared
  with AdamW.

### Trainer surface

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `-opt adamw\|muon` | `LLMM_OPTIMIZER` | `adamw` | optimizer |
| `-mu <float>` | `LLMM_MUON_MOMENTUM` | 0.95 | Muon momentum |
| `-qk_clip <tau>` | `LLMM_QK_CLIP` | 0 (off) | QK-clip threshold; independent of `-opt` |

Logging: setup table `opt=`; verbose setup `muon_momentum=` and `qk_clip=`;
`qk_max=` is a VERBOSE-level step field, and the setup table has a "qk clip
tau" row. State checkpoints record `optimizer_kind` (0 adamw, 1
muon) in header slot 40; resuming with the other optimizer is refused in both
directions, because Muon's momentum lives in the `m` buffer. A Muon resume
reproduces the continuous run to about 3e-5, the same as AdamW's resume gap.

### QK-clip

The causal softmax kernel already computes each row's post-scale max; thread 0
atomic-maxes it into a `[L*NH]` buffer (non-negative fp32 as int32 bits;
negatives clamp to 0, harmless against tau). After the optimizer update,
`qk_clip_step` scales the Q and K rows of each head with max > tau by
`sqrt(tau / S_max)`, and the matching `qkv_bias` entries too, on the fp32
master (bf16 builds) and its rounded param copy. `enable_qk_clip` raises on the CPU target (no logit-max
hook, it would silently never fire) and, in phases 1 and 2, WORLD_SIZE > 1
(ranks would diverge without an allreduce-max; ZeRO >= 1 also shards the
master it indexes in full). **Phase 3 lifted the WORLD_SIZE > 1 raise** for
Muon at every ZeRO stage and for either optimizer at ZeRO-0; AdamW with
ZeRO >= 1 and the CPU target still raise (section 8).

Reported readings: bf16 builds quantize the max to 0.5 near 100 (bf16 scores).

Max lifetime: every forward zeroes and fills a scratch `[L*NH]` buffer;
`backward()` folds it into the step max (overwrite on micro-step 0,
elementwise max afterwards) via `qk_head_max_fold`, so the per-step max is a
fold over the step's training micro-steps only. Eval and generation forwards
never reach `backward()`, so they cannot leak into the clip.
`qk_max()` is valid from the step's last backward to the next step's first.

## 2. Process

Tracks, each in its own `git worktree` and branch, merged into `muon-int` by
the orchestrator: `muon/kernel`, `muon/trainer`, `muon/pytest` (reference and
tests), `muon/qkclip`, `muon/wire`, `muon/crosscheck` (torch, then Keller),
`muon/integ` (first training smokes), `muon/perf` (two generations) and
`muon/docs`. Sonnet agents implemented; Opus agents verified each track
adversarially (reading the code and trying to construct a wrong kernel the
gates would accept), and a whole-merge Opus verifier audited the merged tree.
The orchestration followed the user's protocol: Sonnet writes code, Opus
verifies, and agents hand off to a fresh agent at 256K context tokens
(mandatory stop at 512K). Every defect below was found by a reviewer, not by
the author.

## 3. Gotchas (reviewer findings fixed before merge)

1. **bf16 hyperparameters must be rounded to the param dtype.** The op passes
   lr/momentum/wd/grad_scale as `Scalar[dtype]`, so a bf16 case hands the
   kernel momentum 0.94921875, not 0.95. The reference used fp32 values and a
   correct kernel failed the 1e-5 momentum check on every bf16 case. Fixed
   with `effective_hp()` in the reference.
2. **The W-only bf16 gate could not see Newton-Schulz.** One step at lr 0.02
   moves W less than a bf16 ulp at |W| ~ 2; simulated wrong kernels (no NS,
   wrong scale rule, no Nesterov, no weight decay) all passed the trajectory
   atol. Added `test_update_direction`: one step from W = 0, recover
   `O = -W1/(lr*scale)`, gate cosine >= 0.995 and relative Frobenius
   <= max(0.10, 5 x spread vs fp32 NS). Correct kernels measure 0.03-0.04,
   mutants >= 0.278. (Same lesson as the weak-gate memory note: a passing test
   overrules nothing unless it could have failed.)
3. **Reference rounding points were not kernel-faithful.** The reference
   rounded the norm to bf16 and each elementwise op separately; now it applies
   fp32 `1/(norm+eps)` once with bf16 operands, fp32 accumulate and fp32
   GEMM-beta epilogues, one bf16 rounding per GEMM.
4. **SR stream collision.** `SR_MUON_STREAM` was 2, already
   `NVFP4_SR_STREAM` (forward A-operand quantization). Moved to 6 and
   recorded in the `rng_device.mojo` registry (the old comment pointed at
   `adamw.mojo`, wrong file).
5. **QK-clip could never fire.** The step max was reset inside
   `zero_gradients(zero_param_grads=True)`, which the trainer calls from
   `backward()` of micro-step 0, after that micro-step's forward; the buffer
   was always zero at `qk_clip_step`. The forward also wrote straight into the
   step buffer, so val and sampling forwards leaked in. Fixed with the scratch
   plus fold design above. The new end-to-end test fails on the previous
   commit (head max 0.0 after backward).
6. **QK-clip silently inert on CPU / wrong on multi-rank**: raised (see
   above); the multi-rank raise was lifted in phase 3 by a max-allreduce.
7. **Crosscheck misattribution.** The first explanation of the ~1e-2 NS gap
   to stock `torch.optim.Muon` blamed torch's bf16 `addmm` epilogues.
   Isolation on torch 2.13.0 CPU showed torch rounds once with an fp32
   epilogue like ours (a double-rounded model doubles the gap); the gap is
   torch's bf16 norm plus 1-ulp accumulation-order flips amplified by NS. Our
   NS with a bf16 norm and exactly-rounded GEMMs matches torch bit-for-bit at
   48x96 / 96x48 and to 1.1e-3 / 3.7e-3 at 64x64 / 256x768.
8. **Widened gate must still reject mutants.** Under the `torch_muon` source
   the trajectory atol is widened to the measured torch-vs-ours gap;
   `test_tolerance_not_absurd` is now parametrized over both sources.
9. **Accidental absolute symlinks.** `data/.fineweb10B` and
   `data/.hellaswag` (pointing into the main checkout) were committed by
   accident and untracked in the QK-clip fix.
10. **Keller harness: "weight decay must be excluded" was wrong.** The first
    Keller cross-check ran wd = 0 on both sides, claiming his decay
    `(1 - lr_keller*wd)` cannot match ours once lr is rescaled. It can: both
    decay the old W before adding the update, so `wd_keller = wd*lr/lr_keller`
    makes the factor identical. The fp32 trajectory gap is unchanged from
    wd = 0 (1.27e-3 to 1.70e-3), while unscaled wd gives 8.6e-3 to 6.2e-2.
    Fixed by rescaling, not zeroing.

Phase 3 gotchas (rank failures exiting 0, missing synchronize, ZeRO-2/3 not
bit-exact against ZeRO-0, positional optional arguments, `pkill -f`, sweep
logs without weight decay) are listed at the end of section 8.

Also noted: torch's Muon uses lerp momentum (buffer and Nesterov update are
(1-mu) x ours, normalized away by NS) and `match_rms_adamw` equals
`0.2*sqrt(max)`; the fp32 param keeps torch's momentum buffer in fp32 like the
kernel's `M`. The crosscheck rescales by (1-mu) before comparing.

## 4. Verification matrix

All GPU runs below used `MAX_USE_ACCELERATOR=1`; without it the pytest bridge
defaults to the CPU op path.

| Check | Covers | Result |
|---|---|---|
| `tests/test_muon.mojo` (GPU, shared ctx) | NS vs host reference; batched and strided groups | pass |
| `tests/test_muon_equivalence.py`, `test_muon_crosscheck.py`, `test_muon_crosscheck_keller.py` | Mojo op vs three references (section 5) | 151 passed (+6 adamw = 157) |
| `tests/test_qk_clip.mojo` | softmax head max incl. scalar (T=13) and vector (T=16) kernels, B=2 planted logits, write-past-NH sentinel, above-diagonal trap, hook on/off bit-identical; `qk_clip_apply` fp32 and bf16+master; end-to-end tiny GPT2 (eval forward does not leak, micro-step max/reset, only hot heads' Q/K rows and biases scale, post-clip head max == tau) | pass |
| Default path unchanged (trainer-track Opus verification, run) | deterministic CPU 3-step AdamW run bit-identical to the pre-Muon tree (4 runs); GPU 5-step losses within run-to-run noise (~1e-4 at step 5); `make verify` cpu and gpu gates | pass |

Gates on the merged tree (`f646bbe`, whole-merge verifier run):

| Gate | Result |
|---|---|
| `make check` | exit 0 (compile-rest built all tools) |
| `make test-mojo` | 190 passed, 1 skipped (the 3-GPU test) |
| `make test-python-cuda` | 351 passed, 6 skipped, 3 xfailed; the skips are generation smoke tests that need the infer binaries, which were then rebuilt and 6 passed |
| `make verify` | cpu and gpu pass |

First training runs (10 steps, tinyshakespeare, pretrained 124M): Muon
finished within about 0.03 loss of AdamW on fp32 and bf16; checkpoint resume
works.

### QK-clip at 124M

The OpenAI-pretrained GPT-2 124M already has max attention logits of about 110
at step 1, so `-qk_clip 100` rescales Q/K on step 1 of a fine-tune. The plan's
"never triggers at 124M" holds only for from-scratch runs. Forced-clip
evidence with tau = 5: `qk_max` 110.6, 6.3, 5.9, 5.6, 5.2... (fp32) and 110.5,
6.2, 5.9, 5.6, 5.2... (bf16).

Phase 3 additions to this matrix (threaded CPU multi-rank harness, sharded
Muon, QK-clip at every stage, state-file layout) are in the gate table of
section 8; the GPU W > 1 rows there are compile-checked only.

## 5. Verification against three independent references

The kernel is gated against (1) our torch reference (`tests/reference.py`
`newton_schulz`, rounding points faithful to the kernel, hyperparameters
dtype-rounded by `effective_hp`), (2) stock `torch.optim.Muon` 2.13
(`adjust_lr_fn="match_rms_adamw"` = 0.2*sqrt(max(A, B)); lerp momentum, so
buffer and update are (1-mu) x ours and the cross-check rescales; same NS
constants; bf16 norm with clamp), and (3) Keller Jordan's `muon-optimizer`
package (`SingleDeviceMuon`, `zeropower_via_newtonschulz5`). Keller applies
the scale `max(1, rows/cols)^0.5` inside `muon_update`; the harness reconciles
with `lr_keller = lr*0.2*sqrt(max)/max(1, r/c)^0.5` and
`wd_keller = wd*lr/lr_keller`, so decay is identical. A copy of our NS
rounding exactly where Keller's does is bit-identical to his function.

Two gate types per source: trajectory (stepwise params and momentum) and
update direction (one step from W = 0; cosine >= 0.995, relative Frobenius
<= 0.10 for the kernel). Counts: 151 passed across the three test files.

Mutant catalogue, every one rejected by every source. The multiple is how many
times over the bound the mutant measured:

| Mutant | Rejected at |
|---|---|
| skip Newton-Schulz | 8.5 to 9.4x |
| 4-step NS | 3.0 to 3.5x |
| no Nesterov | 3.8 to 6.0x |
| transpose not undone | 14x |
| Keller scale | 2.8 to 8.2x |
| sqrt(min) scale | 2.9 to 4.3x |
| scale = 1 | 3.8 to 8.2x |
| no weight decay | rejected by the fp32 trajectory gate |

The remaining NS gaps are explained rounding, not defects. Ours vs
`torch.optim.Muon`: 0.9 to 1.5e-2 relative Frobenius (torch's bf16 norm). Ours
vs Keller: 2.7 to 3.8e-2 (his per-op bf16 rounding); a single GEMM ulp flip
is amplified to about 2e-2 over 5 NS steps, which is the metric's noise floor.
After the perf pass (section 6) the kernel-vs-reference direction gap is
<= 0.0095.

## 6. Performance

Measured with nsys on GPT-2 124M, bf16, B=4 T=1024, a single micro-step, on a
GB10 with a 37 GB vLLM tenant resident (so absolute numbers are contended).

Before tuning, Muon cost +41 ms/step (137 to 178 ms, +30%). The plan's "~5%"
estimate (1.2 TFLOP at 150 to 200 TFLOP/s effective) was wrong twice: the NS
GEMMs are 1.6 TFLOP/step, and the GB10 reaches only about 57 TFLOP/s on these
batched 768-sided GEMMs (about 74 on the model's best GEMM). Attribution:

| Component | ms/step |
|---|---|
| NS GEMMs | 28.3 (1.6 TFLOP) |
| standalone axpby | 13.2 (60 launches, ~2.5 GB traffic) |
| momentum | 4.4 |
| apply | 4.4 |
| normalize | 1.5 |

The kernel-sum delta equals the step delta, so there is no host-side
cuBLASLt setup cost and no hidden sync. cuBLASLt heuristic ranks 1 to 6 were
all slower than the top pick.

- **Change 1 (`b6874a2`)**: width-8 vectorized elementwise kernels (scalar W=1
  fallback for unaligned shapes). Per launch under contention: momentum
  455 to 369 us, apply 455 to 355 us.
- **Change 2 (`6c33f62`)**: `B = b*A + c*A*A` and `X' = a*X + B*X` folded into
  cuBLASLt alpha/beta epilogues (`_matmul_cublaslt` gained optional
  alpha/beta/c_ptr; existing call sites are unchanged). This removes the axpby
  kernel and one bf16 rounding per op, so the kernel now rounds exactly where
  the reference does. Under about 92% GPU contention Muon went 418 to 396
  ms/step (-23 ms); overhead over AdamW there fell 96 to 72 ms. 10-step
  losses differ in the third decimal (rounding change).

**Uncontended A/B** (final tree, both binaries rebuilt after the last
source edit, GPU idle apart from the resident 37 GB vLLM tenant, interleaved
runs, steady-state means; smoke = steps 4-10, realistic = steps 2-4):

| config | AdamW ms/step | Muon ms/step | Muon overhead |
|---|---:|---:|---:|
| bf16 smoke, B=4 T=1024, 1 micro-step (2 runs each) | 138.4 | 167.9 | +29.5 ms, +21.3% |
| bf16 realistic, B=32 T=1024, 131k tokens/step (2 runs each) | 3832.8 | 3872.0 | +39.2 ms, +1.0% |
| fp32 smoke, B=4 T=1024 (1 run each) | 283.6 | 315.3 | +31.7 ms, +11.2% |

The two AdamW realistic runs differ by about 1% between themselves, the same
size as the Muon overhead there. Before the perf pass the bf16 smoke overhead
was +41 ms (+30%).

**Amortization.** The overhead is a fixed cost per optimizer step. In the
600-step FineWeb comparison (B=32, T=1024, 4 grad-accum micro-steps, 131k
tokens/step) both arms run at about 3.93 to 3.95 s/step (Muon 3929 ms vs
AdamW 3949 ms at comparable steps), so the Muon cost is about 1%; at the full
run's 524k tokens/step it is about 0.25%. The +29% figure applies only to the
4k-token smoke config.

**Remaining floor**: about 22 to 28 ms of GEMM per optimizer step plus about
10 ms elementwise. Further cuts need fewer FLOPs (Gram symmetry, which
cuBLASLt cannot exploit for bf16) or tuned batched tiles.

## 7. Known limits and follow-ups

- **Performance**: see section 6. The plan's ~5% step-time estimate was
  wrong; the cost is a fixed per-step overhead, about 1% at 131k tokens/step.
- **ZeRO sharding (plan phase 3)**: landed, see section 8 for what is
  verified and what is still open (GPU W > 1, S6 and S7).
- **`v_memory`** was allocated but unused under Muon (~0.3 GB at 124M): fixed
  by the compact `v` (section 8).
- **Metal/AMD** raise at the first `update()`; **CPU** is a slow reference.
- **NorMuon** (per-row normalization) and **Polar Express** coefficients are
  not implemented.
- **Convergence evidence** at scale:
  [`muon_vs_adamw_124m_2026-10-29.md`](muon_vs_adamw_124m_2026-10-29.md)
  (600 steps from `-e d12` random init on FineWeb, bf16, 131k tokens/step).

## 8. Phase 3: sharded Muon (2026-10-30)

Phase 3 of the plan landed on `main` (merge `3482c54`) as slices S0 to S5 of
[`zero_sharded_muon_design_2026-10-30.md`](zero_sharded_muon_design_2026-10-30.md),
which holds the full design, the per-slice status and every verification note
(section 9 there). Sonnet agents implemented and Opus agents verified each
slice on its own branch (`fu/zs0` ... `fu/zs5`, `fu/qklift`, `fu/zs4int`), each in
its own worktree, under the same protocol as sections 2 and 3. This section
is the summary and the lessons; the doc has the evidence.

### What landed

`-opt muon` now runs under ZeRO-1, ZeRO-2 and ZeRO-3 (streaming and
non-streaming) at WORLD_SIZE > 1, with QK-clip at every stage.

- **Why a new layout.** Flat ZeRO shards misalign with Muon matrices. At 124M
  and W=4 rank 0's flat shard is entirely `wte` (it owns zero Muon elements)
  and ranks 2 and 3 are about 99.9% Muon matrices, so deriving Muon work from
  the flat shard would run no Newton-Schulz on rank 0. The design instead
  gives each matrix to one rank: `owner(k, l) = (k*L + l) mod W` for group k
  and layer l, stored in a per-rank **owner space** of S elements, driven
  through the existing `reducescatter_buckets` / `allgather_ranges`
  primitives. At 124M W in {2, 4, 8} the Newton-Schulz cost is balanced
  exactly (max/ideal 1.000).
- **S0 (`7255440`, `89a176c`, `02bf799`): collective and resume fixes**, found
  by reading the design and reproduced on the pre-fix binary. See "Fixed" in
  the CHANGELOG and the gotchas below.
- **S1 (`1d1c67e`, `4705370`, `ce9f12c`): ownership and ZeRO-1.** Owner table,
  owner-local param/momentum/master/grad buffers, a compact per-rank `v` of
  `A_r` elements from `_adamw_shard_segments()`, and `_update_sharded_muon`
  (pack owned grads, AdamW per segment, Muon over the owner-space groups,
  then the AdamW-range and owner-matrix gathers). `muon_apply_gpu` gained an
  `sr_stride_arg` so stochastic-rounding counters stay global flat indices.
- **S2 (`95a5a2e`, `9852e0d`): ZeRO-2.** The per-layer bucket makes two
  reduce-scatters, flat (non-Muon tensors) and owner-space (the six matrices
  into `muon_g`); the clip norm adds `muon_g`'s sum of squares.
- **S3 (`7578837`): ZeRO-3.** The owner space is authoritative for Muon
  matrices. The streaming seed is a W-round gather, `_z3_stream_layer` and
  `_z3_gather_full` gather the six matrices from the owner space in a second
  `allgather_ranges`, and the update skips propagation.
- **S4 (`021a720`, `2105d06`): checkpoints.** See "Checkpoint layout" below.
- **S5 (`44517a7`, `1a22677`, `d076235`, `a7e1b97`): QK-clip at every stage.**
  A host-side max-allreduce (`ZeroContext.allreduce_max_host`) makes the
  per-head maxima global; the clip then runs inside `_update_sharded_muon`
  after the last local write and before the gathers, scaling the owner-space
  Q/K blocks (and the bf16 master) and the qkv-bias entries inside this
  rank's flat shard. `a7e1b97` made the bias write shard-relative under
  ZeRO-3 streaming, which lifted the ZeRO-2/3 raise.
- **Compact `v` (`b43bfe5`, `684de0c`).** Under Muon the AdamW second moment
  `v` is allocated and checkpointed only for the non-matrix ranges (wte, wpe,
  LayerNorm, biases): 497.9 MB to 158.2 MB at 124M (the "`v_memory` unused"
  follow-up in section 7 is closed). `684de0c` added a state-file size check
  so a pre-compact full-`v` Muon file cannot be read silently (it is read at
  full size and copied down; verified by the S4 reviewer).

### Checkpoint layout

A sharded-Muon rank file is the header, `m` (opt, fp32), `v` (`A_r`, compact)
and `muon_m` (S, fp32): length `256*4 + (opt + A_r + S)*4`. Header slots 41
(zero stage), 42 (ownership version, 1) and 43 (S) are new; AdamW and W=1 /
ZeRO-0 Muon files write zeros there and are byte-identical to before (the S4
reviewer `cmp`'d them against a trainer built from `20c4769`). Masters are not
saved: on resume `muon_p_shard` is re-seeded from the loaded full params and
`muon_master` rebuilt, so fp32 resume is exact and bf16 resume is not
bit-exact. ZeRO-1/2/3 share the layout, so a `-z 1` file resumes at `-z 3`
(verified). `read_state_checkpoint` refuses, before any payload is read: a
different world size or rank, an owner layout versus a replicated one (ZeRO-0
file at ZeRO >= 1 and the reverse), a different ownership version or S, and
any length other than the exact expected size, in both directions.

### Gates and what each proves

All multi-rank evidence here is from the **threaded CPU harness** on one box.
No GPU collective at N >= 2 has run: this machine has one GPU.

| Gate | What it proves | Result (recorded in the design doc) |
|---|---|---|
| `tests/test_zero_muon.mojo` T2 (threaded CPU, a different batch per rank, 3 steps) | ZeRO-1 equals ZeRO-0 **bit-exactly** at W=2, 4 and 8 (L=1: ranks 6 and 7 own nothing). ZeRO-2/3: step 1 bit-exact except `wte` (rel 4.1e-7 / 1.1e-7 / 1.1e-7 at W=2/4/8); 3 steps within `\|a-b\| <= 1e-4\|a\| + 1e-6` | pass |
| Same-batch grad-accum test (`9852e0d`) | with every rank on one batch both reduction orders are exact multiples of one gradient, so ZeRO-2 equals ZeRO-0 bit-for-bit for 3 steps with 2 micro-steps; this is what catches small gradient errors, which the tolerance does not (mutants landed at 0.19x to 1.65x of it) | exact at W=2 and W=4 |
| ZeRO-3 window check | after the last update `_z3_gather_full` and each streamed layer window agree bit-exactly with the owner space; forcing `_z3_stream_layer` back to the flat-shard gather fails all three tests | pass; mutant fails |
| `test_ownership_sizes_tiny_w8` | S, `O_r`, `A_r` and the segment invariants against an independent Python model, including ranks with `A_r = 0` | pass |
| T4 (GPU) | owner layout versus the flat group, bit-identical for matrices up to 2048 elements with equal counts | pass |
| T6 `test_state_checkpoint_sharded_muon_t6`, `test_state_header_plain_files_unchanged` | layout, refusals, plain files unchanged | pass |
| `tests/test_qk_clip.mojo` multi-rank tests | W=2/4 at ZeRO-0/1/2/3 with per-rank-different planted maxima; every rank's params bit-equal an unclipped step **at the same stage** plus `qk_clip_apply` with the global maxima; the qkv bias straddles a shard boundary (`d076235`); mutants (full-buffer bias offset under streaming; bias clip skipped off rank 0) fail | pass |
| `make train-cpu WORLD_SIZE=2` d6, 3 steps | `-opt muon` at `-z 1/2/3`: 10.865850, 10.716245, 10.648577 each, exit 0; AdamW `-z 3` 10.865850, 10.369871, 10.128281 | pass |
| Save/resume, W=2 `-x 2` then `-y 1 -x 4` vs continuous | `-z 1`: `model_4.bin` and both state files `cmp`-identical. `-z 2` and `-z 3`: also identical files, with step 4 printing 10.410458 resumed vs 10.410459 continuous (a 1e-6 print-level difference). Cross-stage `-z 1` to `-z 3` resume matches the `-z 1` run | pass |
| `muon_m` mutant | one momentum element set to 1.0 in a rank file: the resumed run still exits 0 and prints the same losses, but 19,861,728 of 30,357,504 params differ at step 4; the blob is read and used | caught by file comparison only |
| `make test-mojo` on the merged tree | | `passed=221 skipped=1 failed=0` (the skip is the 3-GPU test) |
| `make verify`, `make lint`, `make -B build-bf16` | | pass |
| W=1 GPU Muon, 5 steps (`make train ARGS="-opt muon -x 5 -v 0 -s 0"`) | no regression from the merges | 4.287241, 3.985745, 4.111457, 3.619328, 3.700196 on the merged tree |

**What the CPU harness does not prove.** The GPU arms of
`_reducescatter_buckets_gpu` (owner-space dests and the scratch check), GPU
`allgather_ranges` (owner and AdamW propagation, the self-alias skip),
`ensure_comm_setup` with `max(opt, S)`, GPU-branch routing in `update()`, the
GPU ZeRO-3 split streaming and W-round seed, bf16 owner-master seeding and
re-seeding on resume, and every performance claim are **compile-checked only**.
The design doc has a 2-GPU runbook (section 8, "2-GPU smoke for another
machine": configs `zero/muon_z{1,2,3}_w2.json`, a 30-step comparison against
`-z 0`, a write/resume at ZeRO-2, `LLMM_MEM_REPORT`, and step time) for
whoever has two GPUs. Do not read the CPU result as GPU W>1 verified.

### Per-rank memory and traffic (124M, W=4, bf16, design doc section 6)

Per-rank optimizer state in MB (fp32 m, v and master 12 B per flat element,
owner space 10 B per element):

| rank | AdamW ZeRO | Muon ZeRO-0 (before phase 3, the only multi-rank option) | sharded Muon now (with compact v) | after the open S7 trim |
|---|---:|---:|---:|---:|
| 0 | 373.4 | 1493.7 | 585.8 | 585.8 |
| 1 | 373.4 | 1493.7 | 494.7 | 312.5 |
| 2 | 373.4 | 1493.7 | 461.4 | 212.7 |
| 3 | 373.4 | 1493.7 | 461.5 | 212.9 |

Sharding cuts Muon state from 1494 to at most 586 MB per rank. The peak is
rank 0 whatever is trimmed, because its flat shard is all `wte`; reaching
AdamW's 373 MB needs both the S7 trim and memory-aware ownership. Extra
buffers: `muon_grad_shard` 42.5 MB, and ZeRO-3 streaming params grow by
42.5 MB (dead Muon positions remain in the flat param shard). The "now"
column is the design doc's "sharded Muon S1-S6 + compact v" column; it is a
model of the allocations, and `LLMM_MEM_REPORT` on a real run (W=2 d6 CPU)
showed `muon_p_buf` / `muon_g_buf` / `muon_m_buf` at 21,233,664 B each on both
ranks, not the 124M W=4 figures.

**Per-rank traffic caveat.** Summed over ranks the bytes moved equal plain
ZeRO (slightly fewer: no padding). Per rank they do not: rank r handles
`A_r + O_r` elements instead of opt. At W=4, rank 0 reduce-scatters 52.35M
elements against 31.12M (+68%) and sends 314 MB in the gathers against 187 MB;
the slowest rank sets the collective time. This was a first-draft error
("zero extra bytes") corrected by the independent review. Nobody has measured
it: S6 must report the slowest rank, and the S7 ownership work must weigh
`A_r + O_r`, not Newton-Schulz cost alone. The extra collective calls (+1 per
step at ZeRO-1/2, +L reduce-scatters per micro-step at ZeRO-2/3, +L gathers per
forward and per backward re-stream at ZeRO-3) may be latency-bound on GPU
(**unverified**).

### Phase 3 gotchas

1. **Rank failures exited 0 before `ce9f12c`.** Both threaded multi-rank
   launchers caught a rank's error, printed `Rank r failed:`, aborted the
   coordinator and returned normally, so a W>1 run where every rank died
   passed any exit-code gate. This predates sharded Muon. `_raise_if_rank_failed`
   now raises after the join (exit 2). Any older "W=2 smoke passed" claim
   based on the exit code alone is suspect.
2. **Every enqueue followed by a host read needs a `synchronize`.** The CPU
   Muon reference read `muon_g` on the host right after the pack's queued
   `enqueue_copy`; without one `synchronize()`, W=4 was nondeterministic
   (1-ulp differences in proj weights, then everything). Removing it failed T2
   in 8 of 10 runs; with it 20 of 20 passed. The same applies to the pack
   and scatter into any owner buffer, and to the streaming seed's rounds.
3. **ZeRO-2/3 cannot be bit-exact against ZeRO-0.** ZeRO-0 forms `wte_grad =
   lm + enc` locally and then allreduces; bucketed ZeRO-2/3 reduce-scatters
   the LM-head and encoder parts separately, `(lm_0+lm_1) + (enc_0+enc_1)`.
   fp32 addition is not associative, so `wte` differs by ulps after step 1
   and every tensor from step 2. The design's first draft claimed bit-exact;
   tests compare to a **same-stage** reference where exactness is needed
   (QK-clip, the grad-accum test), and ZeRO-1 only is compared bit-exactly
   against ZeRO-0.
4. **A tolerance gate that "catches any wrong gradient" does not.** AdamW and
   Muon are nearly scale-invariant: a 1.001 scale on a routed Q block lands at
   0.19x to 1.65x of the 3-step tolerance while the W=4 baseline noise is
   0.18x. Only the step-1 exact check and the same-batch accumulation test
   caught them. (Same lesson as gotcha 2 in section 3.)
5. **Positional optional arguments in `read_state_checkpoint`.** Two branches
   added trailing optional arguments after `shard_num_parameters`
   (`v_num_parameters`, and `expect_num_processes` / `expect_process_rank`),
   and the call sites passed them positionally, so a merge could silently
   rebind `WORLD_SIZE` as the `v` length. Callers now pass them by keyword
   (`02bf799`).
6. **The scratch bound must be rank-invariant.** `_bucket_overlap_max`
   originally used the overlap with this rank's shard, so with an undersized
   scratch one rank could raise while a peer entered the barrier and hung. It
   now takes the max over every rank's shard (`02bf799`).
7. **Resume read a silent prefix.** A larger world size, or a ZeRO-0 state at
   ZeRO >= 1, loaded a prefix of `m` and filled `v` from the file's `m` bytes
   (the design's first draft had the direction backwards; a smaller world size
   raised). Fixed by the world-size/rank check (`89a176c`), the trailing-byte
   check (`02bf799`) and finally the exact length check (`021a720`).
8. **`pkill -f` / `pgrep -f` match sibling processes.** The sweep driver's
   guard matched sibling compiles and trainers and blocked runs for hours
   (`b68f028`: match the exact process name and retry; `8a1c330`: guard on
   `nvidia-smi` compute apps, since sibling CPU-target trainers never use the
   GPU).
9. **A fresh `make build-train` is required after changing WORLD_SIZE.**
   `TRAIN_BIN` does not encode it and silently keeps the previous one when the
   sources are unchanged: run `make build WORLD_SIZE=N` and check the mtime.
10. **Sweep logs could not prove which weight decay a run used.** The jsonl
    setup event recorded lr, momentum, clip and QK-clip but not `-c`, so the
    sweep's wd values rested on run tags. `fc68e8c` logs `weight_decay`.
    Sweep logs written before it still lack it.

### Remaining follow-ups

- **S6, 2-GPU verification (open):** the runbook above, plus per-rank
  collective time and bytes. This is the only way to claim GPU W>1 works.
- **S7, trim the flat `m` / `master` to the AdamW segments: landed** (flat
  `m`/bf16 `master` are now `A_r`-sized in the compact-v layout; state files
  shrink accordingly and pre-trim files are refused; Opus-verified, see the
  design doc §9 S7). Still open there: memory-aware ownership (proposed with
  numbers, needs ownership version 2), coalesced owner gathers via staging,
  and merging the two per-layer bucket calls into one.
- **Resume-rewrite race on `-y` with `-n` dividing the resume step:** `-y 1`
  re-saves the resume step at loop start, so a `model_k.bin` the other rank may
  still be reading is rewritten. Observed once, under heavy parallel load (rank
  1 failed with "Failed to read enough bytes", passed on rerun). Pre-existing,
  not caused by S4; fixed by mirroring llm.c (the resumed step is not
  re-saved) plus a cross-rank barrier after loading, verified bit-exact.
- **bf16 master paths under streaming are untested:** bf16 owner-master
  seeding, re-seeding on resume and the streaming QK-clip master offset have
  only been compile-checked (`-D LLMM_BF16=1 -D LLMM_SR_MASTER=1 -D WORLD_SIZE=2`).
  A bf16 SR W=1-versus-W>1 run is unverified, and the GPU is not bit-equal
  between W=1 and W>1 anyway (float-atomic norm accumulation above 2048
  elements, different cuBLASLt batch counts).
- **Non-streaming ZeRO-3 is harness-only and goes stale:** it is reachable
  only without a coordinator; after an update the owned Muon matrices in
  `params_memory` are stale. The trainer and T2 always attach a coordinator and
  stream. A multi-step non-streaming ZeRO-3 Muon test would train on stale
  matrices.
- **Still raising:** AdamW with `-qk_clip` at ZeRO >= 1 (flat-shard Q/K rows)
  and QK-clip on the CPU target (no logit-max hook; tests plant the maxima
  with a test-only flag, so the clip's end-to-end trigger on CPU is untested).
  The Opus S5 reviewer confirmed the CPU raise by running ZeRO-3 with
  `-qk_clip 5` (both ranks print "QK-clip needs a GPU target", exit 2).
- Metal/AMD still raise at the first `update()`; NorMuon and Polar Express
  coefficients remain unimplemented.

## 9. Hyperparameter sweep (2026-10-30)

Full write-up, results table and caveats:
[`muon_hparam_sweep_124m_2026-10-30.md`](muon_hparam_sweep_124m_2026-10-30.md)
(12 run logs under `sweeps/muon_*.jsonl`; numbers re-derived from them by an
independent Opus review). It was prompted by the first comparison running Muon
with AdamW's recipe unchanged.

It is a **ranking study, not a convergence study**: single GB10, bf16, seed 42,
300 steps of 131072 tokens (39M tokens), 50 warmup steps, cosine to zero, val
(20 batches) at 100/200/300, greedy one factor at a time. Score is val loss at
step 300.

Recipe at 300 steps:

```
-opt muon -l 4.8e-3 -mu 0.9 -c 0 -gc 0
```

Val 5.012 against 6.118 for the AdamW control (`-opt adamw -l 6e-4 -c 0.1`,
not re-tuned). **For a long run keep `-c 0.1 -gc 1.0`** (that is,
`-opt muon -l 4.8e-3 -mu 0.9 -c 0.1 -gc 1.0`) unless re-tested.

How much to trust each piece:

- **lr is the robust finding.** Val@300 improved monotonically from 5.98 at
  3e-4 to 5.07 at 4.8e-3, then 5.10 at 9.6e-3. The optimum, 4.8e-3, is 8x
  AdamW's conventional 6e-4, even though Muon shares AdamW's lr scale under
  Moonlight RMS scaling. The upper bracket is weak: the 9.6e-3 turnover is
  0.030 nats, barely above the doc's 0.02 single-seed threshold.
- **mu 0.9: moderate** (5.034 vs 5.073 at 0.95 vs 5.173 at 0.98; 0.9 is the
  edge of the tested range).
- **wd 0 and gc 0: weak.** Gaps of 0.008 and 0.015 nats at one seed. Read as
  "no evidence these help over 300 steps", not "they hurt".

Caveats: one seed, no repeats (differences under about 0.02 nats are noise);
the 300-step cosine horizon favours larger lr and less regularisation, so the
long-run optimum is likely lower lr with wd and clipping on; the AdamW control
was not tuned, so the 1.11 nats gap is an upper bound on the optimizer gap, not
a tuned-versus-tuned number; interactions, lower mu and lr between 2.4e-3 and
9.6e-3 are untested; the earlier 600-step Muon-vs-AdamW comparison used 6e-4
for both and very likely under-tuned Muon (by how much is not measured). Runs
shared the GPU with sibling agents (step-time spikes to 5.4 to 6.0 s in four
runs), which does not affect the ranking. The run logs did not record weight
decay when the sweep ran (see gotcha 10).

## 10. Compact v, seed override, weight-decay logging

- **Compact `v`**: see section 8 (`b43bfe5`, `684de0c`).
- **`-D LLMM_INIT_SEED` (`27975e3`)**: the from-scratch init seed is a
  build-time override, default 42 (the default build still prints step-1 loss
  11.013952; seed 1 gives 10.949014). `make` train and profile builds take
  `MOJO_DEFINES`, e.g. `MOJO_DEFINES="-D LLMM_INIT_SEED=1"`, so multi-seed
  comparisons need no source edits. It is a compile-time constant, not a
  run-time flag, and the seed is not a logged field: check the step-1 loss.
- **`weight_decay` in the setup event (`fc68e8c`)**: sweep logs can now prove
  which `-c` a run used.

## AI use statement

Written with AI assistance (Claude Code: Sonnet coding agents, Opus verification agents, Fable 5.1 orchestrator), directed by Evan Owen.

Sections 8 to 10 (phase 3, sweep, small changes) were summarized from the
design doc, the sweep doc and the git history by a Claude Code docs agent; the
implementation and verification they describe were done by Sonnet coding agents
and Opus review agents, directed by Evan Owen.
