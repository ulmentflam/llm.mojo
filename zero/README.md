# zero/ — ZeRO baseline configs and benchmark data

Baseline configs for multi-GPU ZeRO training (one rank per GPU, single
process), in the spirit of the `deepspeed/` config directory other projects
carry. Launch one with:

```bash
make train-zero ZERO_CONFIG=zero/zero2.json           # stage 2, bf16, 8 GPUs
make train-zero ZERO_CONFIG=zero/zero3.json ARGS="-x 50 -o log_z3"
```

`make train-zero` drives `scripts/train_zero.py`, which builds the training
binary at the config's `world_size` and runs it with the config's stage,
precision, and flags. Anything in `ARGS` is appended after the config's flags,
so it overrides them.

## Configs

| config | stage | shards |
|---|---|---|
| `zero1.json` | 1 | optimizer state |
| `zero2.json` | 2 | + gradients (bucketed backward reduction) |
| `zero3.json` | 3 | + parameters (just-in-time streaming) |

Stage 0 (plain DDP) needs no config: `make train ARGS="-z 0 -pn N"` after
`make build WORLD_SIZE=N`.

## Schema

```json
{
  "zero_stage": 2,        // ZeRO stage 0-3 (runtime -z flag)
  "world_size": 8,        // ranks == GPUs (compile-time WORLD_SIZE=N)
  "precision": "bf16",    // fp32 | bf16 | fp8 | fp4 — selects the binary
  "train_flags": {        // llm.c-style trainer flags, minus the dash:
    "b": 4,               //   -b micro-batch size per rank
    "t": 1024             //   -t sequence length
  }
}
```

Any trainer flag (`./build/train_gpt2 -h` for the full list) can go in
`train_flags`; `-z` and `-pn` come from `zero_stage`/`world_size` and should
not be repeated there.

## bench/

`bench/` holds the machine-readable per-stage benchmark results
(`bench_zero_world<N>.json` and before/after snapshots from the stage-2/3
memory pass) written by `make benchmark-zero` / `scripts/benchmark_zero.py`.
The rendered charts land in `figures/`. See the README's
[Multi-GPU (ZeRO stages 0-3)](../README.md#multi-gpu-zero-stages-0-3) section
for the headline numbers.
