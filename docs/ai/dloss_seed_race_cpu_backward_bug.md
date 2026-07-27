# The dL/dloss seed race: silently wrong CPU gradients, 1 run in 4000

**Symptom.** `tests/test_zero_equivalence.mojo` failed roughly 1 run in 120,
with a handful of parameters differing from the single-GPU baseline by about
`2e-4`. The failures looked like a ZeRO sharding bug. They were not. The real
fault was a missing memory barrier in `GPT2.forward` that let the CPU backward
read an uninitialized buffer, corrupting **every** parameter gradient — with no
error, no NaN, and a bit-identical loss.

The fix is one `ctx.synchronize()`.

---

## Terms, defined once

- **ZeRO** (Zero Redundancy Optimizer) is a way to train one model across many
  GPUs without storing a full copy of everything on each one. **Stage 1** splits
  the optimizer state across ranks, **stage 2** also splits gradients, **stage 3**
  also splits the parameters themselves. A **rank** is one participating process;
  **world size** is how many there are. A **shard** is the slice of the parameter
  vector one rank owns.
- **SIMD** ("single instruction, multiple data") lets a CPU apply one operation
  to a block of adjacent floats at once. Code often **pads** array lengths up to
  a multiple of that block width so every block is full and aligned.
- **AdamW** is the optimizer. It keeps two running averages per parameter — the
  first moment `m` (a smoothed gradient) and the second moment `v` (a smoothed
  squared gradient) — and steps each weight by roughly `m / sqrt(v)`.
- **dlogits** is the gradient of the loss with respect to the model's raw output
  scores. It is the first thing the backward pass computes, and everything else
  is derived from it.
- **ULP** ("unit in the last place") is the gap between adjacent representable
  floating-point numbers — the smallest possible difference between two floats.

---

## Why `2e-4` was a red herring

The learning rate in the test is `1e-4`, and every observed difference was close
to `2e-4` — twice the learning rate. That looks like a smoking gun: a parameter
updated twice, or updated with the wrong sign.

It is neither. At the very first optimizer step (`t=1`), AdamW's bias correction
makes the moments collapse to something remarkably simple. With `m` and `v`
starting at zero:

```
m  = (1 - beta1) * g          m_hat = m / (1 - beta1) = g
v  = (1 - beta2) * g^2        v_hat = v / (1 - beta2) = g^2
sqrt(v_hat) = |g|
```

so the step is

```
step = lr * ( g / (|g| + eps) + weight_decay * param )
```

That `g / |g|` is a **sign function**. At `t=1`, AdamW throws away the magnitude
of the gradient entirely and steps by `±lr` no matter how large or small `g` is.
So if two runs disagree only about the *sign* of some gradient element, the
resulting parameters differ by exactly `2 * lr` — regardless of whether the
gradient was `1.0` or `1e-9`.

This makes AdamW a spectacular amplifier. Measurement on this model: **1140 of
4592 gradient elements have `0 < |g| < 1e-6`**, and 128 more are exactly zero. A
quarter of the parameters sit on a sign knife-edge, where an arbitrarily small
perturbation flips the sign and produces a full `2e-4` parameter difference.

The observed diffs were not all exactly `2e-4` — they ranged from `1.76e-4` to
`2.00e-4`. That spread is itself a measurement. Solving
`diff = 2 * lr * |g| / (|g| + eps)` with `eps = 1e-8`:

| observed diff | implied \|g\| |
|---|---|
| 2.00e-4 | much greater than 1e-8 |
| 1.98e-4 | about 1e-6 |
| 1.76e-4 | about 7.3e-8 |

The diffs fall just short of `2e-4` by exactly the amount `eps` softens the
normalization. The failing elements were confirmed to be the near-zero ones.

**Lesson:** a suspiciously round multiple of the learning rate does not identify
a double update. It identifies AdamW's first step normalizing away magnitude.

---

## The reasoning chain: from "intermittent" to "uninitialized read"

The starting hypothesis, inherited from an earlier investigation, was sound in
form: identical inputs plus identical code plus different output across
processes implies something is being read before it is written, since heap
layout and address-space randomization differ per process.

That inference was right. Its *localization* was wrong — the search had been
aimed at the padded tail of the optimizer shard, because `4592 / 8 = 574` pads up
to `576` and the last rank's real data runs out partway through its shard.

Three cheap measurements redirected it.

**1. The padding tail is not read.** On CPU the sharded AdamW is called with
`local_num_params = min(num_parameters - offset, optimizer_num_parameters)` —
560, not 576, for the last rank. It never touches the padding. And
`params_buf` is allocated at `padded_num_parameters` (4608) and zero-filled, so
the tail is in-bounds and zeroed anyway.

**2. World size 8 was not special.** The failures had clustered at world size 8,
which looked like a clue about the `574 → 576` padding. But each test compares
the baseline against `N` ranks, so a w8 test contributes 8 comparisons and a w2
test only 2 — w8 is 80% of all exposure. Three sightings landing on w8 has
probability `0.8^3 ≈ 0.51`. That is the *expected* outcome, not an anomaly.
Running the loop longer duly produced w2 failures, at stages 1, 2 and 3, at
ranks 0, 3 and 7. Nothing was special about any of them.

**3. The failures were never a single element.** Instrumenting the test to print
every mismatching index showed 16 to 228 elements failing at once, spread across
the whole shard — about 14% of it, which is what you would expect if roughly half
of the 25% sign-unstable elements flipped. A single bad index in a padded tail
cannot do that. The entire gradient vector was being perturbed.

At that point the ZeRO framing was clearly wrong, so the next step was to remove
ZeRO from the experiment entirely.

---

## The isolating experiment

Build the *same* single-GPU model (`WORLD_SIZE=1`, `zero_stage=0`) 4000 times in
one process, run forward and backward, and compare every gradient against the
first run:

```
NONDET it=1289  grad_elems_differing=4464  sign_flips=1  loss_differs=False
                ref_loss=4.1590047  loss=4.1590047
ITERS=4000  runs_with_grad_diff=1  runs_with_loss_diff=0
```

Two facts fall out, and together they name the bug.

**The backward is nondeterministic about 1 run in 4000, with no ZeRO involved at
all.** That rate explains the entire observed flake: the test binary performs
6 tests × (1 baseline + N ranks) = 30 model runs, and `30/4000 = 1/133`, against
a measured `1/120` over 600 binary runs. The ZeRO equivalence test was never
testing the thing that was broken. It was just the canary — the only test that
compared two independently computed gradients element by element.

**The loss is bit-identical while 4464 of 4592 gradients change.** The 128
elements that matched are exactly the 128 that are structurally zero. So *every
nonzero gradient* changed, but the forward pass was perfect. Whatever went wrong
happened after the loss was computed and before the first gradient was written.

A per-parameter-tensor breakdown of a captured event:

```
tensor  0 wte          1024/1024 differ   max relative diff  25.1
tensor  4 qkv_weight    768/768  differ   max relative diff 467.9
tensor 10 fc_weight    1024/1024 differ   max relative diff 151.3
tensor 12 proj_weight  1024/1024 differ   max relative diff 676.5
tensor 15 ln_f_beta      16/16   differ   max relative diff   9.6
```

Every tensor 100% corrupted, by factors of up to 676× — this is not rounding
noise. And `ln_f_beta` is the *first* parameter gradient produced after the
classifier, so the corruption enters right at the top of the backward, at
`dlogits`.

---

## The bug

`GPT2.forward`, in `train_gpt2.mojo`:

```mojo
var dloss_mean = Scalar[DType.float32](1.0) / Scalar[DType.float32](
    batch_size * seq_len * grad_accum_steps
)
for i in range(batch_size * seq_len):
    self.losses_host_buf[i] = dloss_mean
self.ctx.enqueue_copy(                      # ASYNC: seeds grad_acts.losses
    dst_ptr = ... self.grad_acts.losses ...,
    src_ptr = ... self.losses_host_buf ...,
    size = batch_size * seq_len,
)
fused_classifier[GPT2_DTYPE, Self.target, write_d_logits=True](
    as_mut_kernel[GPT2_DTYPE](self.acts.logits),
    as_mut_kernel[StatsDType](self.acts.losses),
    as_immut_kernel_from_mut[DType.float32](self.grad_acts.losses),   # READS it
    ...
)
```

`dL/dloss` is the constant `1 / (B*T*grad_accum_steps)`. It is staged into a host
buffer and pushed to the device with `enqueue_copy` — an **asynchronous** operation
that goes onto the device context's queue. `fused_classifier` then reads that
same buffer.

On a GPU target this is fine: the classifier is enqueued as a kernel on the same
stream, so the stream orders it after the copy, for free.

**On CPU it is not fine.** `fused_classifier` dispatches to
`fused_classifier_cpu`, which runs *synchronously on the calling host thread*
under `sync_parallelize`. It never enters the device queue. Nothing orders it
against the copy. The two race.

Direct measurement of the window — stamp a sentinel into `grad_acts.losses`
synchronously, then read it back on the host immediately before the classifier
runs:

```
3570 / 4000 forwards: all 16 d_loss entries still un-copied
  ~14 / 4000 forwards: PARTIALLY copied (1, 2, 4, 6, 10, 11, 12, 13, 14, 15 of 16)
```

The copy has almost never landed at that point. It normally catches up during the
classifier's softmax phase, before any row reaches its
`var d_loss = d_losses_ptr[idx]`. Occasionally — when the device queue is backed
up, which is exactly the state left behind by a model's construction-time
allocation fills — it does not.

A row that loses the race reads the buffer's **pre-copy contents**: zeros on a
zero-filled `grad_acts` buffer, or uninitialized heap on a freshly allocated
model. `d_loss` for that token is then wrong, so that token's `dlogits` are
wrong, and since `dlogits` is the root of the entire backward, every parameter
gradient downstream is wrong.

And the loss survives, because `_fused_classifier_cpu` computes it from the
logits *before* overwriting them in place with `dlogits`:

```mojo
var x_t = logits_ptr[base + target_idx].cast[DType.float32]()
losses_ptr[idx] = log(s_row) + m_row - x_t     # loss: uses logits, not d_loss
...
var d_loss = d_losses_ptr[idx]                 # dlogits: uses d_loss
```

That is why this hid for so long. Every visible health signal — the loss curve,
the absence of NaNs — is computed on the safe side of the race. Only the
gradients are wrong, and only sometimes, and training is robust enough to absorb
one bad step in four thousand without an obvious scar.

This also explains the magnitude signature. A row that reads `d_loss = 0` drops
that token out of the backward entirely, which is why the corrupted gradients
were measured 17× to 64× *smaller* than the reference:

```
flat_idx   g_base       g_rank       ratio
   6       5.47e-05    -3.26e-06     17x smaller, sign flipped
   7       3.14e-04    -8.00e-06     39x smaller, sign flipped
  16       1.32e-03    -2.53e-05     52x smaller, sign flipped
  17      -4.38e-04     6.81e-06     64x smaller, sign flipped
```

---

## The fix

Order the seed before the read, on the target where the queue does not do it for
you:

```mojo
comptime if is_cpu[Self.target]():
    self.ctx.synchronize()
```

Gated on CPU specifically: on GPU targets the copy and the classifier kernel are
both enqueued on this context's stream, so the stream already orders them and an
extra host-side barrier would cost a stream synchronization every step for
nothing.

---

## Evidence

Same binary, same loop, same box, before and after.

| measurement | before fix | after fix |
|---|---|---|
| `test_zero_equivalence` built binary | **9 failures / 1350 runs** | **0 failures / 1350 runs** |
| isolated no-ZeRO backward determinism | **2 nondeterministic / 16000 runs** | **0 nondeterministic / 16000 runs** |

If the failure rate were unchanged, seeing zero failures in 1350 runs has
probability `(1 - 9/1350)^1350 ≈ e^-9 ≈ 1.2e-4`.

Intermediate reproductions during the hunt, all pre-fix: 5/600 and 4/750 on the
test binary; 19/48000 rank-comparisons on an in-process ZeRO harness; 1/4000 and
1/12000 on the isolated single-model probe.

---

## On the absence of a regression test

There is no deterministic regression test in this change, and that is a
deliberate choice rather than an omission.

Losing this race requires the device queue to be backed up at one specific
instant *inside* `GPT2.forward` — between the `enqueue_copy` and the classifier's
`d_losses_ptr[idx]` read. That instant is not reachable from a test: `forward`
drains the queue at an earlier synchronization point, so backpressure applied
from outside is gone before it matters.

Two candidate tests were built and both were rejected for failing to
discriminate:

- **Poisoning `grad_acts.losses` before each forward** (3000 iterations,
  re-poisoned every step). Passes with the fix. Also passes *without* it — the
  copy still lands in time, because a warm reused model leaves the queue idle.
- **Poisoning plus a 128× larger `B*T`** to widen the copy, with a fresh model
  per iteration. Also passes both ways at any iteration count that runs in
  reasonable time.

A test that passes with and without the fix is worse than no test: it converts
absence of evidence into false confidence. The honest gate here is the
statistical one recorded above.

The reproduction harness is preserved in this document rather than in the tree —
the isolated probe is about 200 lines that build the same model in a loop and
compare gradients bitwise, and it is straightforward to reconstruct from the
description in "The isolating experiment" above.

---

## Open lead: the same pattern may exist elsewhere (unaudited)

The hazard class here is general, and this fix addresses exactly one instance of
it. The pattern is:

> a buffer is written by an **asynchronous device-queue operation**
> (`enqueue_copy`, `enqueue_fill`, `enqueue_memset`) and then read by a
> **synchronous host-side CPU kernel** that runs under `sync_parallelize` and
> never enters that queue.

On GPU targets the stream orders these for free, so the bug is invisible there
and only CPU builds are exposed. That asymmetry is what let this one survive:
the GPU path is the one that gets the scrutiny.

This change fixes the `dL/dloss` seed in `GPT2.forward`. The remaining call
sites have **not** been audited. Two things are known:

- `zero_gradients` does end with `self.ctx.synchronize()` (guarded
  `comptime if not HAS_METAL`), so its `enqueue_fill` of the gradient and
  activation-gradient buffers is correctly ordered against the backward that
  follows.
- No other `enqueue_*` call was found inside `backward`'s own body.

What has not been checked is every other `enqueue_fill` / `enqueue_memset` /
`enqueue_copy` in `train_gpt2.mojo` and `llmm/` against the CPU kernel that next
reads the same buffer. A mechanical audit — enumerate the async writes, and for
each one confirm a `synchronize` (or a genuine queue-ordered consumer) before
the next host-side read — would either close the class or find siblings.

Priority: low. Any sibling would share this bug's signature — rare, silent,
CPU-only, and invisible to the loss — so it is worth doing before the next time
someone chases an intermittent numeric failure, but nothing currently points at
a specific second instance.

---

## What to take from this

1. **A rate is a fingerprint.** `1/4000` per model run times 30 model runs per
   test binary equals the observed `1/120`. Matching those two numbers is what
   confirmed the isolated probe was reproducing the same bug and not a second one.
2. **Suspect the amplifier before the source.** AdamW at `t=1` converts a sign
   disagreement into exactly `2 * lr`, which made a gradient-corruption bug
   present as a suspiciously structured optimizer bug.
3. **A healthy loss does not mean a healthy backward.** Here the loss was
   bit-identical in the corrupted runs, by construction — it is computed before
   the in-place overwrite that the race affects.
4. **Mixed execution models need explicit barriers.** Anywhere a synchronous host
   kernel reads a buffer written by an asynchronous queue operation, the ordering
   must be stated. On GPU the stream hides this class of bug; on CPU it does not.
   This codebase already synchronizes at the two neighbouring points in the same
   function — this one seed was simply missed.

---

## AI use statement

This bug was root-caused and fixed with AI assistance via Claude Code, under the
direction of Evan Owen. The investigation inherited a diagnostic hand-off that
had correctly inferred an uninitialized read but had localized it to the ZeRO
optimizer's SIMD padding tail; the work recorded here redirected that search,
isolated the fault to the CPU backward with ZeRO removed entirely, and measured
the race window directly.
