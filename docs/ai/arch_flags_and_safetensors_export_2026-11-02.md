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
refused. Every checkpoint written before this change (including the published
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

## AI use statement

Written with AI assistance (Claude Code, Sonnet 5.5 agent), directed by Evan Owen.
