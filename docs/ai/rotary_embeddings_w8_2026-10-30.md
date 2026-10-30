# Rotary position embeddings (W8), 2026-10-30

Work item W8 of [`modded_nanogpt_adoption_plan_2026-10-30.md`](modded_nanogpt_adoption_plan_2026-10-30.md):
modded-nanogpt's half-truncated rotary on q and k, replacing the learned `wpe`
add, behind `-D LLMM_ROTARY=1` (default 0). Branch `mn/w8`.

## What it does

Per head of size `hd` (`hd % 4 == 0`, 64 at GPT-2 124M), with `n = hd/4`:

```
freq[i]   = (1/1024) ** (i / (n - 1))          i = 0..n-1  (linspace(0, 1, n))
theta[t,i]= t * freq[i]
pair (x1, x2) = (x[i], x[i + hd/2])             i < n only; dims i >= n are identity
y1 =  x1 cos + x2 sin
y2 =  x2 cos - x1 sin
```

The speedrun concatenates `n` zero frequencies, so the second half of the
rotate-half pairs is exactly the identity; the kernel therefore touches only
`n` pairs per head and the other `hd/2 - n` pairs and the v third are
bitwise unchanged.

- **Kernel** (`llmm/rotary.mojo`, self-contained): `rotary_qk_rows[dtype, target, inverse](qkv, table, B, T, table_rows, H, hd, ctx)`
  rotates the q and k thirds of a `[B, T, 3C]` qkv activation in place;
  `inverse=True` applies the transpose rotation. Table: fp32 `[cos | sin]`, each
  `[T_max, hd/4]`, built once on the host in float64 and rounded
  (`rotary_table_host`), uploaded in `GPT2._allocate_rotary_table` and sized to
  the model's `max_seq_len` (the `wpe` row count), so generation up to the
  context length needs nothing more.
- **Forward**: after the qkv matmul (+ bias) and before `attention_fwd`, which
  splits the (now rotated) qkv into q, k, v. The stored `acts.qkv`, `q`, `k`
  are the rotated values; attention backward therefore needs no change.
  QK-clip sees the rotated logits (rotation preserves the norms it rescales).
- **Backward**: after `attention_bwd` has merged dq, dk, dv into `d_qkv`, the
  inverse rotation is applied to the q and k thirds of `d_qkv`, before the qkv
  matmul backward. The weight gradient of that matmul uses its input (ln_1), not
  the rotated output, so nothing else changes.
- **wpe**: stays in the parameter layout, so the descriptor, the checkpoint
  tensor list, ZeRO shards and the Muon owner-space are unchanged. The encoder
  forward gets `use_wpe=False` (a comptime parameter on `encoder_fwd`; wte is
  copied, wpe is not even loaded), the encoder backward gets
  `include_wpe=False` so `dwpe` is never written (the gradient buffer/pool is
  zeroed beforehand, so the wpe gradient range is exactly 0 and the ZeRO
  reduce-scatter still moves a zero range, keeping every rank's range list
  identical), and under `-opt muon` `_adamw_ranges` is split around wpe so it
  is not in the AdamW ranges (m/v/master sizes follow from the ranges).
  Under plain AdamW the update is dense over the flat buffer; wpe has a zero
  gradient and zero moments there, so it just stays where it was (zero from a
  from-scratch init).
- **Init**: `allocate_parameters_random` still *draws* the wpe `N(0, 0.02)`
  values and discards them, so every later tensor gets exactly the values it
  gets in a non-rotary build with the same `-D LLMM_INIT_SEED`; rotary and
  non-rotary arms differ only by the position mechanism, not by a shifted RNG
  stream. wpe is zero-filled. (Verified: the rotary d6 checkpoint has
  `max|wpe| = 0.0` and `wte` std 0.01999, the off one `max|wpe| = 0.089`.)

## Checkpoint flag

An architecture-flags word (`ARCH_FLAG_ROTARY = 1`) is stored in
- the **state** header, slot 44, which bumps the state version to
  `STATE_VERSION_FLAGS = 2`. A state with flags 0 is still written as version 1
  with slot 44 untouched, so default-build state files are byte-identical to
  before (proved below), and the reader accepts both versions;
- the **model** header, slot 20, written only when nonzero (default model files
  byte-identical; llm.c ignores the slot).

`check_arch_flags` (train_gpt2.mojo) refuses:
- resume (`-y`): state file **and** model file flags must equal this build's,
  in both directions, before any tensor is read;
- initialising from a model `.bin` (`-e path.bin`): a flagged file in a build
  without the flag is refused. The reverse (a non-rotary `.bin`, e.g.
  pretrained GPT-2, into a rotary build) is deliberately allowed: that is how
  `make verify` feeds the rotary reference the pretrained weights, and as a
  conversion experiment it is legitimate (loss starts high: 9.36 on the debug
  batch against 5.27, since the weights were trained with `wpe`).

## Reference and gates

`GPT2_ROTARY=1 python train_gpt2.py` is the reference: same `apply_rotary` math
(table in float64 then fp32), no `wpe` add, `wpe` gradient dumped as zeros (the
dump layout is unchanged), `transformer.wpe` activation hook skipped (no graph,
no grad). It writes **only** `gpt2_124M_rotary_debug_state.bin`; the model
`.bin` files (identical weights) are not rewritten, so flag-off fixtures stay
byte-identical. `make rotary-reference` regenerates the rotary state and prints
the 10-step PyTorch loss list pasted into `test_gpt2.mojo` as the rotary branch
of `expected_losses` (generated by PyTorch, not by the trainer). Under the flag
`test_gpt2.mojo` reads the rotary state; `make verify-gpu|verify-cpu` and
`make test-mojo` take `MOJO_DEFINES` (e.g. `MOJO_DEFINES="-D LLMM_ROTARY=1"`).

RESULTS_PLACEHOLDER

## Merge notes for W7 (QK-norm)

W7 adds a per-row kernel over the q and k thirds of `[B, T, 3C]` (norm per head
of the row). The speedrun order is norm then rotary. The two call sites in
`GPT2.forward` / `backward` are adjacent: W7's kernel runs first and rotary
second in forward (`qk_norm_rows(qkv)` then `rotary_qk_rows[..., inverse=False]`),
and in backward the inverse rotation runs first and then W7's norm backward
(`rotary_qk_rows[..., inverse=True]` on `d_qkv`, then `qk_norm_bwd`). Both are
in-place on the qkv-shaped buffers, so the order is just statement order.
A fused kernel would load a `(b, t)` row's q and k head slices, normalise, apply
the `_rotate_pair` math of `llmm/rotary.mojo` before the store; the rotation
needs the table row for `t = bt % T` only, so it fits a per-row kernel
unchanged. The norm backward must see the gradient w.r.t. the *rotated*
q/k un-rotated (i.e. after the inverse rotation), which the order above gives.

## Not verified / limits

- Multi-rank ZeRO with the flag on was not run (the layout is unchanged and the
  default-build ZeRO tests are the gate); the `include_wpe` skip is in the
  row-sparse and dense paths and zeroed pools make the reduced wpe range 0.
- Plain AdamW (not Muon) still updates wpe through the dense flat update; it
  stays at its initial value because gradient and moments are 0, but a
  pretrained non-zero wpe would be weight-decayed (harmless: it is not read).
- Rotary on Metal and fp8/fp4 builds were not built.

## AI use statement

Written with AI assistance (Claude Code, Sonnet agent), directed by Evan Owen.
