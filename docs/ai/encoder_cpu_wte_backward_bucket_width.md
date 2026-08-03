# Encoder CPU `dwte` backward skipped token rows

**Symptom.** `make test-python` failed on all three
`tests/test_encoder_equivalence.py::test_backward_matches_torch` cases. The
position-embedding gradient (`dwpe`) matched PyTorch, but the token-embedding
gradient (`dwte`) stayed entirely zero for rows whose tokens appeared in the
batch.

For the small fp32 case, PyTorch expected non-zero gradients for token rows
`[5, 12, 15, 16, 25, 26, 27, 28]`; the custom op returned no non-zero `dwte`
rows at all.

## Cause

The CPU `wte_backward_cpu` path derived each bucket's channel span with:

```mojo
var c_per_warp = WARP_SIZE * width
```

That is the right shape for GPU terminology, but the CPU path is consuming
bucket metadata produced with the module-level bucket contract:

```mojo
comptime WTE_C_PER_WARP = 32 * WTE_BWD_SIMD_WIDTH
```

On this MAX/Mojo CPU custom-op path, the mismatch made `c_len` evaluate to zero
for the Python equivalence cases, so the kernel never entered the accumulation
loop that writes `dwte`.

## Fix

Use `WTE_C_PER_WARP` directly in the CPU backward path when mapping a bucket's
`channel_group` to `[c_base, c_end)`. This keeps the consumer in lockstep with
`build_wte_buckets` and the Python test helper, both of which emit bucket groups
using the same constant.

## Verification

Commands run on Apple Silicon/macOS with Pixi-installed Mojo
`1.0.0b3.dev2026072606`:

```bash
PATH=/Users/saisandeepkantareddy/.pixi/bin:$PATH \
LLMM_DISABLE_MEF_CACHE=1 \
/Users/saisandeepkantareddy/.pixi/bin/pixi -q run pytest -q \
  tests/test_encoder_equivalence.py
# 6 passed

PATH=/Users/saisandeepkantareddy/.pixi/bin:$PATH \
/Users/saisandeepkantareddy/.pixi/bin/pixi -q run pytest -q \
  tests/test_dataloader.py \
  tests/test_tokenizer_equivalence.py \
  tests/test_encoder_equivalence.py
# 12 passed

PATH=/Users/saisandeepkantareddy/.pixi/bin:$PATH \
make PIXI='/Users/saisandeepkantareddy/.pixi/bin/pixi -q' test-python
# 233 passed, 15 skipped

PATH=/Users/saisandeepkantareddy/.pixi/bin:$PATH \
make PIXI='/Users/saisandeepkantareddy/.pixi/bin/pixi -q' build
# passed
```

The Mojo build printed Crashpad warnings under the sandboxed macOS environment,
but exited successfully and produced `build/train_gpt2`.

## AI use statement

Written with AI assistance (Codex/Agent), directed by Sai Sandeep Kantareddy.
