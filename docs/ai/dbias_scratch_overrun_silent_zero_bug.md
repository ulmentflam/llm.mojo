# The dbias scratch overrun: bias gradients silently never written

**Symptom.** `tests/test_matmul_bwd_fp4.mojo` passed. It should not have. On
310 of 320 instrumented runs the bias gradient `d_bias` it was checking came
back **entirely zero** — all 768 entries at one call site, all 3072 at the
other — in *both* arms of the comparison. The test's only `d_bias` assertion
was that the two arms be bit-identical, so it compared `0` against `0` and
reported PASS.

The zero was wrong. Reducing the test's own input on the host gives
`d_bias[0] = -2.897968` at one site and `-0.1594845` at the other.

The cause was not in the finalize protocol everyone suspected. A GPU kernel was
writing **1.5x past the end of its scratch buffer**, and the buffer allocated
immediately after it — with a measured gap of exactly **0 bytes** — held the
synchronization counters that same kernel used to decide whether to write its
result at all. The kernel corrupted its own control state and then silently
skipped its only output.

---

## Terms, defined once

- A **kernel** is a function that runs on the GPU. It is launched as a **grid**
  of **thread blocks**; each block is a group of threads (here 24–128 of them)
  that run on the same physical core and can talk to each other cheaply. Blocks,
  by contrast, cannot generally talk to each other and are not guaranteed to run
  at the same time or in any particular order.
- A **reduction** collapses many values into one — here, summing a
  4096-row × 3072-column matrix down the rows to get 3072 sums. **Block-level
  reduction** means each block reduces a slice, writes its **partial** result to
  memory, and something later adds the partials together.
- **`barrier()`** (CUDA's `__syncthreads()`) makes every thread *in one block*
  wait until all of them arrive. It says nothing about other blocks.
- An **atomic** operation is one that cannot be interleaved with another thread's
  operation on the same address. **`fetch_add(p, 1)`** atomically adds 1 to
  `*p` and returns the value it held *before* the add, so if N threads each call
  it once on a counter starting at 0, they receive the values `0..N-1` in some
  order, each exactly once. That "each exactly once" is the whole basis of the
  protocol below.
- **`threadfence()`** makes the calling thread's earlier writes visible to the
  rest of the GPU. It orders **only that one thread's** writes. It is not a
  barrier and it does not wait for anybody.
- **bf16** ("bfloat16") is a 16-bit float with 8 bits of mantissa — roughly
  0.4% relative precision. **fp32** is the ordinary 32-bit float.
- **SIMD width** is how many values of a given type fit in one vector register.
  On a CPU with 512-bit vectors, 32 bf16 values fit; a 128-bit vector holds 8.
- **`d_bias`** is the gradient of the loss with respect to a linear layer's bias
  vector. For a layer with output `y = xW + b`, it is just the column sums of
  the incoming gradient `d_output` — the simplest quantity in the entire
  backward pass.

---

## The protocol, and why it is fragile

`_dbias_fused_gpu` in `llmm/matmul.mojo` computes `d_bias` in a **single**
kernel launch, using the classic "last block finalizes" idiom (NVIDIA's
`threadFenceReduction` sample; llm.c's `matmul_backward_bias_kernel9`). The
grid is 2-D: `grid.x` splits the columns into **column-blocks**, `grid.y` splits
the rows into **row-blocks**. For one column-block:

1. Each of the `row_blocks` blocks sums its own slice of rows and stores that
   partial into a scratch array at `scratch[by * out_channels + col]`.
2. Every thread calls `threadfence()` to publish its store.
3. Thread 0 of each block calls `arrived = fetch_add(counters + bx, 1)`.
4. The one block that sees `arrived == row_blocks - 1` — i.e. the block that
   arrived last, and therefore the only one that knows all partials are
   written — reads the whole column back out of scratch, adds the partials, and
   writes `d_bias`. Every other block simply returns.
5. That finalizing block resets `counters[bx] = 0` for the next call.

The saving is real: one launch instead of two. But look at what step 4 assumes.
The finalize is guarded by a **single equality test against a single counter
value**. If the counter does not start at exactly 0, the arriving blocks receive
tickets `k, k+1, … k+row_blocks-1` for some `k != 0`, and *none* of them equals
`row_blocks - 1`. No block finalizes. `d_bias` is never written.

Nothing detects this. There is no error, no NaN, no launch failure. The output
buffer simply keeps whatever it held before — and a freshly allocated CUDA
buffer holds zeros. **A correct-looking zero.** The protocol has exactly one
failure mode and it is indistinguishable from a legitimate answer.

That fragility is real and worth fixing on its own. But it was the *mechanism*,
not the cause. Something had to make the counter non-zero in the first place.

---

## The false lead: a dirty counter left over from a previous call

The obvious story is that the counter is stale: some earlier call failed to
reset it, so every later call in the same process is dead. It fits the shape of
the evidence — the kernel is correct in isolation, correct in a fresh replay,
and broken inside a longer-lived process.

It was wrong, and one measurement killed it: the failure happens on the **very
first** `d_bias` call in a fresh process. There is no previous call to have left
anything dirty. The counter buffer is explicitly zeroed at allocation, and the
self-reset in step 5 had been in the code since the kernel landed.

So the counter was being corrupted *by the very launch that then failed to
read it correctly*.

---

## The actual cause: a comptime constant that resolved against the wrong target

The caller, `matmul_bias_bwd`, picks a per-thread vector width:

```mojo
comptime width = simd_width_of[dtype]()
comptime FUSED_ROW_BLOCKS = ROW_BLOCKS * width   # ROW_BLOCKS = 16
```

The kernel's comments describe `width` as "one 128-bit transaction: 4 fp32 lanes
or 8 bf16 lanes", mirroring llm.c's 128-bit `x128` load. Every piece of sizing
arithmetic in the file was done against that assumed 8.

But `matmul_bias_bwd` is **host** code. `simd_width_of[DType.bfloat16]()`
resolves against the machine doing the compiling, and this box has 512-bit
vectors:

```
simd_width_of[bfloat16] (host) = 32      # not 8
```

So `FUSED_ROW_BLOCKS` is `16 * 32 = 512`, not `16 * 8 = 128`. The kernel is
still *functionally* fine at that width — the loads stay in-bounds and
coalesced — but `row_blocks` is 4x larger than anyone's arithmetic assumed, and
`row_blocks` is one of the two factors that size the scratch.

The scratch capacity was a hand-computed constant:

```mojo
# "worst case here is 128 row-blocks * 3072 = 393,216 elements … 1<<20
#  leaves >2x headroom"
comptime CAP = 1 << 20          # 1,048,576 fp32
```

With the real width, the requirement is not 393,216:

| out_channels | what it is | needs `512 * oc` | `CAP = 1<<20` | |
|---|---|---|---|---|
| 768  | `C`, attention-proj / MLP-proj bias | 393,216 | 1,048,576 | fits |
| 2304 | `3C`, QKV bias | 1,179,648 | 1,048,576 | **overruns by 131,072** |
| 3072 | `4C`, MLP fc bias | 1,572,864 | 1,048,576 | **overruns by 524,288** |

The two widest biases in a GPT-2 layer both overrun. And the overrun does not
land somewhere harmless. Printing the two allocations back to back, in the
order the code creates them:

```
scratch base   = 17716740096
counters base  = 17720934400
scratch bytes  = 4194304
gap (counters - scratch_end) bytes = 0
counters inside overrun?  True
```

`DBIAS_COUNTERS` begins at exactly `scratch_end + 0`. The **first**
out-of-bounds element the kernel stores lands precisely on `counters[0]` — and
`counters[0]` is the slot for column-block 0, which *every* call uses, including
the 768-wide calls that never overrun anything themselves.

The full chain:

1. A call with `out_channels` 2304 or 3072 stores partial fp32 sums past the end
   of the scratch.
2. Those stores overwrite the arrival counters with float bit patterns — large,
   arbitrary, essentially never 0.
3. Later in **that same launch**, thread 0 of each block increments the now-
   garbage counter. Nobody sees `row_blocks - 1`.
4. No block finalizes. `d_bias` is left untouched — zero.
5. The reset in step 5 never runs either, because it lives inside the finalize
   branch. The counter stays garbage, so every subsequent call, at *any* width,
   is also dead.

Step 5 is why a 768-wide call — which fits fine — also returned zero: it was
collateral damage from a 3072-wide call that ran earlier and poisoned the
shared counter slot.

This also explains the rare (roughly 2 in 240) variant in which `d_bias` came
back as *asymmetric garbage* rather than zero: whether the corrupted counter
happens to pass through `row_blocks - 1` while the blocks are arriving is a
race, and when it does, some block finalizes early and sums partials that have
not all been written yet.

---

## Was the trainer affected?

This is the question that mattered. `_dbias_fused_gpu` computes the bias
gradients for the transformer's linear layers, and a training run is thousands
of steps inside **one** process with **one** persistent counter buffer. If the
counter is poisoned on step 1, the biases would receive no gradient for the rest
of the run — silently.

**Yes — in bf16, the precision real runs use.** `train_gpt2` built with
`-D LLMM_BF16=1` and run at `-b 4 -t 1024 -x 6`, with the four bias gradients
copied to the host after each step:

| step | `qkv_bias` | `attn_proj_bias` | `fc_bias` | `proj_bias` |
|---|---|---|---|---|
| 0 | 0/27648 | 0/9216 | 0/36864 | **768**/9216 |
| 1 | 0/27648 | 0/9216 | 0/36864 | 0/9216 |
| 2 | 0/27648 | 0/9216 | 0/36864 | 0/9216 |
| 3 | 0/27648 | 0/9216 | 0/36864 | 0/9216 |
| 4 | 0/27648 | 0/9216 | 0/36864 | 0/9216 |
| 5 | 0/27648 | 0/9216 | 0/36864 | 0/9216 |

Every bias in the model received an identically zero gradient, on every step,
for the whole run. The optimizer dutifully applied weight decay to biases that
never got a gradient signal, and nothing anywhere reported a problem.

The single non-zero cell is the tell. It is exactly one layer's worth of
`proj_bias` (`C = 768`), from the **first** `matmul_bias_bwd` call in the very
first backward pass — layer 11's proj, `oc = 768`, which fits under the old cap
and so completed normally. The *second* call is `fc_bias` at `oc = 3072`, which
overran and poisoned `counters[0]`. Everything after it, at every width, was
dead. That is the mechanism visible in a single table cell.

**fp32 was not affected.** `simd_width_of[float32]()` is 16 on this host, so
`FUSED_ROW_BLOCKS` is 256 and the largest requirement is `256 * 3072 = 786,432`
— under the old `1<<20` cap. All four tensors were fully non-zero on all six
steps. So the bug needs bf16 *parameters*, which is precisely what production
runs use — including the fp8 and nvfp4 recipes, whose parameters stay bf16 —
and is the configuration least covered by the fp32-reference test suite.

Note the scope of this measurement: it is one binary, one shape, six steps. It
establishes that the bug breaks a training run. What it did to the runs already
published is a separate question, answered in
[What the published checkpoints actually show](#what-the-published-checkpoints-actually-show)
— and the answer there is not the one this table would lead you to predict.

After the fix, the bf16 run has all four tensors fully non-zero on all six
steps, and its loss trajectory **diverges from the pre-fix run starting at step
2** (4.639164 vs 4.617250; step 1 is identical, because the step-1 loss is
computed before the first update). Restoring the gradients changes training —
which is the last thing needed to rule out "the biases were converged anyway".

### Why the existing tests never caught it

A minimal `forward/backward/update` driver on the pre-fix source fails at
`T = 256`, `512` and `1024` — but **passes at `T = 64`**, which is exactly the
`B=4, T=64` reference batch that `dump_grads_gpt2.mojo` and `test_gpt2.mojo`
use.

That is not luck, it falls out of the code. The scratch store sits *outside* the
row loop:

```mojo
var acc = SIMD[DType.float32, width](0.0)
for r in range(r0, r1):
    acc += ...
(scratch + by * out_channels + col).store(acc)   # always runs
```

so a row-block whose slice is empty (`r0 >= rows`) still stores — it just stores
`0.0`. At `T = 64`, `rows = 256` and `row_tile = ceildiv(256, 512) = 1`, so every
row-block from `by = 256` upward is empty. The overrun for `oc = 3072` begins at
`by = 341`, past that point, so **every out-of-bounds store writes zero** — it
lands on the counters and leaves them at 0, which is the value they were
supposed to have. The bug writes out of bounds and is harmless.

At `T = 1024`, `rows = 4096` and `row_tile = 8`, so `by = 341` covers real rows
`2728..2735` and stores a real sum. The same out-of-bounds write now deposits a
non-zero float bit pattern on the counters, and the kernel dies.

So the reference batch was small enough that the corruption was silent *twice
over*: out of bounds, and writing the correct value anyway.

---

## What the published checkpoints actually show

The six-step probe above answers "does this break a training run?" — it does.
It does **not** answer "what did it do to the runs we already published," and
extrapolating from one configuration to six real runs turned out to be wrong.

There is a way to check the real runs directly, with no re-run and no
instrumentation. GPT-2 initialises every bias to exactly `0`
(`train_gpt2.mojo`: "weights ~ N(0, 0.02), biases 0"), so a bias tensor still
**bit-exactly zero** after 22,345 optimizer steps demonstrably never received a
gradient. `scripts/check_checkpoint_biases.py` counts non-zero entries per bias
tensor in a checkpoint. Run against the six published precision-comparison
checkpoints:

| Arm | `qkvb` (`3C`) | `attprojb` (`C`) | `fcb` (`4C`) | `fcprojb` (`C`) |
|---|---|---|---|---|
| 124M bf16 | **0** / 27,648 | 768 / 9,216 | **0** / 36,864 | 2,304 / 9,216 |
| 124M fp8 | 27,648 / 27,648 | 9,216 / 9,216 | 36,864 / 36,864 | 9,216 / 9,216 |
| 124M nvfp4 | **0** / 27,648 | **0** / 9,216 | **0** / 36,864 | 3,072 / 9,216 |
| 774M bf16 | **0** / 138,240 | 46,080 / 46,080 | **0** / 184,320 | 46,080 / 46,080 |
| 774M fp8 | 138,240 / 138,240 | 46,080 / 46,080 | 184,320 / 184,320 | 46,080 / 46,080 |
| 774M nvfp4 | **0** / 138,240 | 46,080 / 46,080 | **0** / 184,320 | 46,080 / 46,080 |

The three layernorm biases (`ln1b`, `ln2b`, `lnfb`) are fully trained in all six
files. They have their own backward and never reach this kernel, so they are the
control: whatever went wrong is specific to `matmul_bias_bwd`, and is not a dead
optimizer or a checkpoint-writer fault.

Three readings, in increasing order of how much they cost us.

**The width threshold is exactly where the arithmetic says it should be.**
Scratch demand is `FUSED_ROW_BLOCKS × out_channels = 512 × oc`, against the old
`CAP = 1<<20`. Only `qkvb` (`oc = 3C`) and `fcb` (`oc = 4C`) exceed it — at
either scale — and those are precisely the two that are dead in every damaged
arm. The `C`-wide `attprojb`/`fcprojb` fit under the cap and are the two that
survive. The mechanism predicted the pattern before the pattern was measured.

**The partial columns are a race, not a freeze.** `attprojb` and `fcprojb` do not
overrun on their own launches, but they share the counter array the wide calls
have already poisoned, so they die too when the poison happens to reach them.
`768 / 9,216` is exactly one layer of twelve. Which layers survive *drifts within
a single run* — the 124M bf16 arm's `fcprojb` has 1,536 non-zero entries at step
1,000 and 2,304 at step 22,345 — so a handful of layers intermittently got a
real gradient through. "The biases were frozen" is the right summary of the
outcome and the wrong description of the mechanism.

**The two fp8 arms are undamaged, and that is the expensive part.** All three
precisions route their bias gradient through the same bf16 `matmul_bias_bwd`
(`llmm/matmul.mojo` — the fp8 and fp4 backward wrappers call it unchanged), so
this is not a code-path difference. An out-of-bounds write lands on whatever the
allocator placed after the scratch buffer, and in those two builds that was
evidently not the counters. Neither `MODULAR_DEBUG=device-sync-mode` nor build
date separates the arms cleanly: the 774M fp8 run set it and the 124M fp8 run did
not, and both came out clean. The per-arm variation is not currently root-caused
beyond "allocation layout decides the victim," and it does not need to be — the
fix removes the write. But it is worth stating plainly, because a bug whose
damage depends on allocator adjacency can silently corrupt a *different* buffer
in a build where the counters happen to be somewhere else, and no one would see
a row of zeros to tip them off.

The consequence for the published work is not that six runs are uniformly
wrong. It is worse than that: **the arms are damaged differently, so the
comparison between them is confounded** — fp8 trained its biases, bf16 and
nvfp4 largely did not, and the headline finding was that fp8 edged out bf16.
The README's precision comparison has been retracted on that basis and all six
arms are being re-trained.

---

## From "the test passes" to "the test proves nothing"

The test's `d_bias` check was:

```mojo
for i in range(out_channels):
    assert_equal(host_d_bias_ref.unsafe_ptr()[i],
                 host_d_bias_fp4.unsafe_ptr()[i], ...)
```

Both arms deliberately reuse the *same* bf16 `matmul_bias_bwd` kernel, so
bit-identity is the right property to assert — it catches a wiring mistake. The
flaw is that it is a **relative** assertion with no anchor. It compares the code
against itself. A failure mode that takes out both arms equally is invisible to
it, and "the kernel wrote nothing" is exactly such a failure mode: both buffers
are freshly allocated, both are zero, and `0 == 0`.

A relative assertion is only as good as the assumption that the two sides can
fail independently. When both sides run the same code, that assumption is
false — and the test degenerates into a tautology that reports PASS with maximum
confidence.

The repair is to give the comparison an anchor that does not come from the code
under test. `d_bias` is a column sum of `d_output`, and the test already holds
`d_output` on the host as its own input data. So the test now:

1. reduces `d_output` on the host, in fp32, independently of any GPU kernel;
2. asserts every entry of `d_bias` is non-zero — each is a sum of 4096 random
   values, so an exact zero means "never written", not "the gradient is zero";
3. asserts the GPU result matches the host reduction in relative L2 (< 0.02;
   measured ~0.002, which is bf16 rounding).

Rebuilt against the **unfixed** kernel, the new test fails loudly:

```
FAIL test_fc_bwd_site
  left: 0  right: 3072
  reason: fc-bwd: bf16 d_bias has only 0/3072 nonzero entries -- an all-zero
  d_bias means the dbias kernel never wrote ..., not that the gradient is zero
FAIL test_proj_bwd_site
  left: 0  right: 768
```

Note the anchor must be computed from the test's *inputs*. Pasting the kernel's
own output in as the expected value would rebuild the same tautology in a form
that looks rigorous.

---

## The fix

Three changes in `llmm/matmul.mojo`, in decreasing order of importance.

**1. Size the scratch from the quantities that actually drive it, and check it.**
`DBIAS_SCRATCH_CAP` is now derived — `ROW_BLOCKS(16) * 32 lanes * 8192 max
out_channels` — rather than back-computed from an assumed vector width, and
`matmul_bias_bwd` refuses to launch if the real requirement exceeds it:

```mojo
if FUSED_ROW_BLOCKS * oc > DBIAS_SCRATCH_CAP:
    raise Error("matmul_bias_bwd: fused dbias needs … but DBIAS_SCRATCH_CAP is …")
```

Silent memory corruption becomes a loud, specific error naming both numbers.

**2. Make a missed finalize impossible rather than silent.** The arrival test is
now a residue, not an equality:

```mojo
var is_last = arrived % Int32(row_blocks) == Int32(row_blocks - 1)
```

Each launch adds exactly `row_blocks` to the slot, so `row_blocks` consecutive
tickets cover every residue mod `row_blocks` exactly once — **exactly one block
takes the finalize branch regardless of what value the counter started at**. The
reset to 0 is kept (it keeps the counter bounded, far from int32 overflow) but
is no longer load-bearing for correctness, and it was moved above the
`col >= out_channels` early-return so no exit path can skip it.

**3. Add the missing `barrier()`.** Between the per-thread scratch stores and
thread 0's `fetch_add` there was no block-wide synchronization — only
`threadfence()`, which orders one thread's own writes and waits for nobody. So
thread 0 could announce its block's arrival while sibling warps were still
storing, letting the finalizing block read a half-written column. This is a
genuine, independent defect. It could not explain the observed zeros — column 0
belongs to thread 0, which fences its own store before signalling, so the one
column a person is most likely to spot-check is precisely the one that stays
correct.

### Which change actually fixed it

Fixes 2 and 3 are defense in depth, and it would have been easy to ship all
three and claim credit for the wrong one. Building each half separately settles
it:

| variant | fc (oc=3072) | proj (oc=768) |
|---|---|---|
| unfixed | 0/3072 nonzero | 0/768 nonzero |
| **CAP fix only** (equality test retained) | 3072/3072, matches host | 768/768, matches host |
| **residue + barrier only** (old CAP restored, guard disabled) | **0/3072 — still broken** | — |

The capacity fix is the necessary and sufficient one. The residue test alone
cannot help: it guarantees that *somebody* finalizes, but it cannot reconstruct
scratch that was overwritten out of bounds, and forcing a finalize from a block
that is not genuinely last produces garbage instead of zeros.

---

## Before and after

Same driver, same loop, 80 process launches x 4 call sites per arm — the only
difference is which `llmm/matmul.mojo` it was linked against:

| | observations | all-zero `d_bias` | fully non-zero **and** matching the host reference |
|---|---|---|---|
| before (unfixed) | 318 | **317** | 0 |
| after (fixed) | 320 | **0** | **320** |

The before arm records 318 rather than 320 because two runs aborted partway
through, losing their remaining probe lines — the pre-existing bit-identity
assertion fires when one arm gets garbage and the other gets zero.

The one before-arm observation that was not all-zero is the rare flake, and it
is not a near-miss: `d_bias[0] = -0.93359375` against a true value of
`-2.897968`. That is the "some block finalized too early off a corrupted
counter" case, and it reproduces the `-0.934` seen in earlier instrumentation.

After the fix, every observation is not merely non-zero but takes exactly one
value per call site, matching the independent host reduction:

```
     160 ref0=-2.890625      (proj site, host says -2.897968)
     160 ref0=-0.15917969    (fc site,   host says -0.1594845)
```

Both agree to within bf16's ~0.4% relative precision, as they should. These
host references come from reducing the test's input data; they were never read
back out of the kernel.

---

## Lessons

1. **A constant derived from an assumption must be derived in code, not in a
   comment.** `CAP = 1 << 20` was justified by a comment doing correct
   arithmetic on a wrong premise. The premise (`width == 8`) was written down
   three times in nearby comments and never once evaluated. Deriving the bound
   from the same expression the launch uses removes the possibility of the two
   disagreeing.
2. **`comptime` resolves against the target it is compiled for.** A width chosen
   in host code and handed to a device kernel is the *host's* width. That is
   fine as a tiling parameter and dangerous as a sizing parameter.
3. **Bounds you rely on should be enforced, not documented.** The overrun was
   silent because nothing checked it. One comparison converts an
   unbounded-memory-corruption bug into an error message.
4. **A synchronization protocol with a single equality test has a silent failure
   mode.** Prefer a formulation that cannot no-op — a residue test here — and
   treat "no error was reported" as no evidence at all.
5. **Relative assertions need an absolute anchor.** Comparing two arms that
   share a kernel cannot detect a fault in the shared kernel. Anchor at least
   one side to a value computed independently of the code under test, and assert
   that the output is not trivially empty.
6. **"Correct in isolation" is compatible with "badly broken in situ."** This
   kernel passed 200/200 standalone runs, because the standalone harness used
   the one bias width that happens to fit.
7. **A reference configuration that is small in every dimension can be small
   enough to hide the bug.** The `B=4, T=64` fp32 reference batch missed this on
   two independent counts: fp32 picks a vector width that fits, and `T=64`
   leaves the overrunning row-blocks with no rows, so they overwrite the
   counters with the zeros the counters were supposed to hold. Gradient-checking
   fixtures should exercise at least one configuration matching the shape and
   precision of a production step, not only the cheapest one.

---

## AI use statement

This bug was root-caused and fixed with AI assistance via Claude Code, under the
direction of Evan Owen. The investigation inherited a hand-off that had
correctly identified the silent-finalize failure mode and the missing
`barrier()`, but had attributed the cause to a counter left dirty by a previous
call. The work recorded here refuted that attribution — the failure occurs on
the first call in a fresh process — and traced the corruption to a scratch
overrun caused by a host-resolved vector width, confirmed the overrun lands on
the counter buffer by measuring the allocation gap, separated the necessary fix
from the defensive ones by building each independently, and established the
trainer impact.
