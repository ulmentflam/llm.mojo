# QK-norm checkpoint flag and the rotary + QK-norm safetensors export (2026-11-02)

Branch `mn/hfexport` (from main c714346). Two related changes, motivated by the
published headline checkpoint (`checkpoints/headline_x5700`, rotary + QK-norm).

## 1. QK-norm is bit 2 of the architecture flags

Model header slot 20 and state header slot 44 recorded rotary (bit 0) and, in state
files, untied head (bit 1). QK-norm was not recorded, so a build without
`-D LLMM_QK_NORM=1` loaded a QK-norm checkpoint and computed the wrong model with
no complaint. Bit 2 (`ARCH_FLAG_QK_NORM = 4`) is now written by a QK-norm build to
both file kinds and checked by every reader.

| reader | file has a bit the build lacks | build has a bit the file lacks |
|---|---|---|
| trainer `-e` init (`GPT2.__init__`, also `test_gpt2`, `infer_gpt2`, profile) | refused, names the bit | allowed (the verify gates start flagged builds from pretrained weights) |
| trainer resume (`load_state`), model and state file | refused | refused (strict, both directions) |
| `infer_gpt2` | refused (same constructor) | allowed, prints a `[infer] note` |
| `.safetensors` (config.json `llm_mojo_arch_flags`) | refused | allowed |
| `max_gpt2_common.load_checkpoint` (MAX graphs) | any flag refused: they build stock GPT-2 | n/a |
| `scripts/export_to_hf.py` | reads the header: flagged files take the llm_mojo path, unknown bits refused | n/a |
| `scripts/check_checkpoint_biases.py` | prints the flags; bias statistics do not depend on the architecture | |

`infer_gpt2` with an unflagged file in a QK-norm build: allowed with a note, not
refused. The note now comes from the `GPT2` constructor (`unflagged_init_note`), so
the trainer's `-e`, `infer_gpt2`, `test_gpt2` and profile all print it, for a `.bin`
and for a `.safetensors` (a stock export in a flagged build). `infer_gpt2` alone goes
further for a stock `model_type: gpt2` safetensors in a flagged build: it refuses
(`refuse_stock_safetensors_in_flagged_build`), because nothing in that path is an
intentional init (a stock export is plain GPT-2 by construction, and a rotary/QK-norm
checkpoint exports as `llm_mojo_gpt2`); the trainer's `-e` from the same file stays
allowed, with the note, since fine-tuning a pretrained stock model in a flagged build
is the same init as the `.bin` case. Every checkpoint written before this change (including the published
`model_5700.bin`, flags = rotary only) is "unflagged" for bit 2, and refusing would
make the published file unusable until it is rewritten; the verify gates and bf16
generation smoke tests also run flagged builds on the unflagged pretrained
`gpt2_124M*.bin` on purpose. Silence was the original failure, so the note says
which flags the build has and the file lacks.

Consequence for in-flight work: a QK-norm build can no longer resume a state file
written by an older QK-norm build (it records rotary only). Rewrite a copy with
`scripts/set_checkpoint_arch_flags.py` first.

Flag-off files are unchanged: version and header bytes are written only when a flag
is nonzero (state version 2 only then). `test_gpt2.mojo` and `train_gpt2.mojo` use
the same `GPT2.__init__` `.bin` path, so there is no second loader.

### Corrected copies of the published files

`scripts/set_checkpoint_arch_flags.py <in> <out> --flags rotary,qk-norm` writes a
copy changing only the flag word (refuses unknown magic or version, an existing
output, a version-1 state file, and an untied-bit change). For `model_5700.bin` and
`state_5700_0.bin`, `cmp -l` against the originals shows exactly one differing byte
each (offsets 81 and 177, 1 -> 5 octal).

## 2. Safetensors export for rotary + QK-norm

`scripts/export_to_hf.py` picks the path from the header flags. Flagged files no
longer go through llm.c's exporter (which would have written a stock GPT-2 config for
a model that is not stock GPT-2).

- `config.json`: `model_type: llm_mojo_gpt2`, `architectures: [LlmMojoGPT2LMHeadModel]`,
  `llm_mojo_arch_flags`, `rotary` (type `half_truncated`, `rotary_base` 1024, the
  frequency rule in `rotary_note`), `qk_norm`, `qk_norm_eps` 1e-6,
  `qk_norm_affine` false, `attn_scale` (effective, 1/sqrt(head_dim) = 0.125),
  `activation_function: gelu_new`, `layer_norm_epsilon` 1e-5, `n_layer`/`n_head`/`n_embd`,
  `wpe_omitted`. `AutoModelForCausalLM` and `AutoConfig` raise "does not recognize
  this architecture".
  Not preventable from the export side: an explicit `GPT2LMHeadModel.from_pretrained(dir)`
  still loads, with a model-type warning and a load report that `transformer.wpe.weight`
  is MISSING (that is the reason to omit wpe rather than ship zeros: shipping it would
  make the explicit stock load clean and silently wrong).
- Tensors: HF GPT-2 names, Conv1D transposes for `c_attn`, `attn.c_proj`, `c_fc` and
  `mlp.c_proj`, `wte` cut to the 50257 real rows, tied `lm_head` not stored.
  `transformer.wpe.weight` is omitted when the file is rotary and its wpe is all zero
  (it is never read); a nonzero wpe is kept.
- dtype: the file's own (bf16 file -> bf16, fp32 file -> fp32), so the export is
  bit-exact to the `.bin`. fp32 -> bf16 is refused (it rounds); bf16 -> fp32
  (`--dtype float32`) widens exactly, for fp32 builds.
- `generation_config.json`, `tokenizer.json`, `tokenizer_config.json` are produced the
  way the existing `gpt2-124m-fineweb-mojo` repo's were (`GPT2Tokenizer.from_pretrained("gpt2")`
  `.save_pretrained`); the two tokenizer files are byte-identical (sha256) to that repo's.
- `infer_gpt2` and the trainer load such a `.safetensors` (pure Mojo): config.json
  flags go through the same `check_arch_flags`, a missing wpe is accepted only for
  rotary, and the config's `attn_scale` is applied unless `-attn_scale` or
  `LLMM_ATTN_SCALE` overrides it. `infer_gpt2 --loss <ckpt> <shard> [B] [T]` prints the
  mean loss of one real batch for any checkpoint.

### Checks

`scripts/check_hf_export.py` (a) compares every tensor to the raw `.bin` (bit patterns,
transposes undone) and (b) loads the safetensors into the PyTorch reference
(`train_gpt2.py` with `GPT2_ROTARY=1 GPT2_QK_NORM=1`, fp32, run from a scratch dir).
`--mutant NAME` re-exports with one Conv1D transpose forgotten; all four fail.
Numbers are in the commit message and the session report.

### Review fixes (Opus verifier, 2026-11-02)

- Regression: `note_flag_mismatch` in `infer_gpt2` called `peek_model_header`, which
  accepts only llm.mojo's magic 20240520, while `GPT2.__init__` also accepts llm.c's
  20240326. A legacy-magic copy of `gpt2_124M_bf16.bin` ran on main but exited 1 in
  generate, `--eval` and `--loss` on the default build ("bad model magic number in
  header: 20240326"; reproduced with the pre-fix binary, fixed binary generates and
  reports loss 3.5096061). The pre-constructor peek is gone from infer (the
  constructor does the check and the note); `peek_model_arch_flags` accepts both
  magics for any future pre-constructor look. The other `peek_model_header` call
  sites are the resume path (`load_state`, model file written by this trainer, so
  the strict peek is right there) and tests. Test:
  `test_peek_model_arch_flags_accepts_the_legacy_llm_c_magic` (planted legacy magic
  with flags 5; mutant restoring the strict magic fails it).
- Exporter: a flags = 1 file is refused without `--flags-confirmed`, because pre-QK-bit
  rotary+QK-norm files record the same word. A flags = 0 file takes the stock path and
  that cannot be detected (a QK-norm-only model written before the bit also has flags
  0 and would export as stock GPT-2); the exporter prints that limit, the check script
  with `--expect-loss` is the defence.
- Exporter `--attn-scale` (default 1/sqrt(head_dim)) is written to `config.json`; a
  non-numeric, non-finite or non-positive value is refused, and a non-default value is
  refused on the stock path (a stock config cannot record it). The loader applies the
  config's scale.
- `config.json`: `rotary_note` gives the exact rotation, theta = pos * freq,
  x1' = x1 cos + x2 sin, x2' = x2 cos - x1 sin (sign checked against
  `llmm/rotary.mojo` `rotate_y1`/`rotate_y2`, `inverse=False`).
  `attn_pdrop`/`embd_pdrop`/`resid_pdrop` are 0.0: llm.mojo's Mojo trainer has no
  dropout anywhere (`grep -ril dropout *.mojo llmm/ tests/*.mojo` finds one comment in `llmm/safetensors.mojo` and no code); the PyTorch
  reference `train_gpt2.py` has an optional, default-off one. GPT2Config's 0.1
  defaults would have claimed otherwise.
- Loader cross-checks (`check_hf_arch`): for `llm_mojo_gpt2`, `rotary` and `qk_norm`
  must be present booleans that agree with `llm_mojo_arch_flags` (the mutant
  `qk_norm: false` with flags 5 loaded before); a stock config may not say rotary or
  qk_norm true. `scripts/check_hf_export.py` checks the same against the `.bin` header.
- Shape checks in the Mojo loader: every non-matrix tensor, `wte`, `wpe`, and the
  transposed matrices are checked against config.json dimensions before reading. A
  SQUARE matrix stored the wrong way round (`attn.c_proj`) is invisible to shape, so
  `check_hf_export.py` (bit-exact round trip + forward) is the gate for that, not the
  loader.
- `check_hf_export.py --expect-loss V --loss-tol T` fails a wrong-model export (a
  flags = 1 export run under the rotary-only reference scores 8.5543 against 3.2880
  expected: status=fail), and `--mojo-loss M` compares against `infer_gpt2 --loss`.
- Tests (`tests/test_checkpointing.mojo`): `read_hf_arch` and `check_hf_arch` on planted
  config files: missing flags, flags the build lacks, flags on a non-llm_mojo
  model_type, key disagreement (eight planted variants), a stock config saying
  rotary; and `unflagged_init_note` over all flag pairs. Mutants (each applied alone,
  each fails exactly its test): strict magic in `peek_model_arch_flags`; the
  key-agreement check removed; the missing-flags check removed; the final
  `check_arch_flags` removed; the stock-with-flags check removed.

Gates on the fix commit (bd8f584): flag-off CPU `cmp` against main c714346 (AdamW and
Muon, model and state files, `LLMM_USE_CPU=1`) byte-identical; `make build`,
`make build-bf16`, `make verify-cpu` pass; `make verify-gpu` passes with no flags,
QK_NORM, ROTARY and both; `make verify-untied-gpu` passes; `make test-mojo` passes
325 tests flag-off. Under `-D LLMM_ROTARY=1 -D LLMM_QK_NORM=1` `test_zero_untied` does
not compile (its ownership test deliberately refuses a rotary build, a constraint
that predates this branch and fails the same on the pre-fix tree); the other 31 files
pass, including `test_checkpointing` (32 of 32). Re-export of the published corrected
`model_5700.bin`: `model.safetensors` byte-identical (sha256 540566164813ec7c...);
`config.json` differs only in `attn_pdrop`, `embd_pdrop`, `resid_pdrop` (0.1 to 0.0)
and `rotary_note`. `infer --loss` on the 4 x 1024 check batch: 3.2883191 for both
`.bin` and `.safetensors`; an independent forward (not importing the repo) 3.28799.

## AI use statement

Written with AI assistance (Claude Code, Sonnet 5.5 agent), directed by Evan Owen.
