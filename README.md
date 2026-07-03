# LLM.🔥

This project is my port of Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) that extends the GPU kernel functionality of @dorjeduck's [llm.🔥](https://github.com/dorjeduck/llm.mojo) in honor of [Mojo's](https://mojolang.org) official v1.0.0 release, currently in beta. Visit [llm.c](https://github.com/karpathy/llm.c) for a detailed explanation of the original project.

> **Note**: This project is based on nightly Mojo 1.0.0b3 release.

## Installation

### Step 1: Clone the repository

This project vendors Karpathy's `llm.c` as a git submodule (used as the CPU/GPU
reference for benchmarking), so clone with `--recurse-submodules`:

```bash
git clone --recurse-submodules git@github.com:ulmentflam/llm.mojo.git
cd llm.mojo
```

If you already cloned without that flag, run `git submodule update --init` from
the repo root (the Makefile also does this automatically the first time a
`benchmark`/`profile-llmc-*` target needs it).

### Step 2: Install Pixi & Make

If you don't have it, install [pixi](https://pixi.sh/latest/):

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### Step 3: Install Dependencies and Git Hooks

Quick setup — pixi environment + git `pre-commit`/`pre-push` hooks (which run
`make lint` and `make check` respectively; see `make install-hooks`; requires
[pre-commit](https://pre-commit.com/) to already be on your `PATH` — skipped,
not fatal, if not found):

```bash
make install
```

If you have CUDA available and installed (beyond the scope of this file), use
the CUDA-enabled equivalent instead:
```bash
make install-cuda
```

`make install`/`make install-cuda` do **not** download the Tiny Shakespeare
dataset or GPT-2 124M starter weights (~1.5 GB) — that's a separate step,
needed before `make train`, `make verify`, or `make benchmark*` will work:

```bash
make data
```

Or combine both in one shot with `make install-with-data` (or
`make install-cuda-with-data` for the CUDA variant).

### Step 4: Train

```bash
make train
```

For additional help, see `make help`.

## Benchmarks

Benchmark Results: (NVIDIA DGX Spark)

- Below are the average training loop times, observed across llm.mojo, llm.c, and PyTorch.
- We run OpenMP-enabled llm.c with 20 threads.
- CPU comparisons are all done in float32. GPU comparisons are done in float32 and mixed-precision bfloat16.
- All benchmarks and metrics are designed to match hyperparameters in an apples-to-apples comparison.
- **Apple Silicon (M1/M2/M3/M4):** use `make benchmark-metal` — llm.mojo (Metal GPU) vs PyTorch MPS; llm.c has no Metal port so it is replaced by the PyTorch MPS baseline. See `docs/ai/ai_assisted_optimizations_and_benchmarks.md` for the full Apple Silicon benchmarking setup.

### Single GPU

!['Best Single GPU Benchmark'](figures/benchmark_gpu_b4_t64_2026-07-01_2137_NVIDIA-GB10_DGX-Spark.png)

### Single CPU

!['Best Single CPU Benchmark'](figures/benchmark_cpu_b4_t64_2026-07-01_2154_NVIDIA-GB10_DGX-Spark.png)

## Test

We ported `test_gpt2.c` from the original repository to validate our port's functionality. We also have a full verification suite available via make.

### Run Tests

```bash
make test
```

### Run Verification

```bash
make verify
```

## Development Roadmap
Future development includes:

1. Full ZeRO-3 verification
2. Mamba1/Mamba2/Mamba3 architecture and MoE

## Motivation

LLMs in Mojo without the need for PyTorch or CPython. Inspired by Karpathy's [llm.c](https://github.com/karpathy/llm.c), with a focus on proving out the viability of autograd in pure Mojo syntax. The focus is to reproduce GPT-2 and GPT-3 alongside a parallel PyTorch reference in `train_gpt*.py`.

A personal goal of this project is to write all kernel and main Python code without using any LSPs or LLMs, writing every algorithm (forward and backpropagation) from scratch. I received feedback recently that my "coding and math expertise" is not strong enough, and building out this framework is how I intend to strengthen those skills. Just like writing a compiler, writing the fundamentals of generative models from scratch sharpens both engineering and mathematics.

As part of that goal, I will be leveraging NVIDIA Nsight and Perfetto for performance analysis and comparison against my PyTorch implementation of GPT-2. As the project evolves, I will include benchmarking results and other insights into the performance comparisons between Mojo, PyTorch, and even Karpathy's C implementation.

In order to speed up testing, I decided to leverage LLMs/AI to help write and accelerate runtime of test cases. All of the code in `llmm/` and the root directory is written by hand, but the tests have been aided by LSPs and LLMs in order to accelerate writing them. I also use the formatter and compiler to typecheck, but that has always been the case.

## Thanks

A special thanks to https://github.com/dorjeduck/llm.mojo and @dorjeduck for writing the original implementation of llm.mojo in Mojo 25.5.

## Agentic Optimizations

After I reached functional success, my kernels were dramatically underperforming Karpathy and PyTorch. I did the initial profiling and caught that attention was the initial bottleneck. After a few attempts at writing a more optimal attention kernel, I decided to leverage AI agents for optimizing the kernel. Originally I leveraged Google Gemini, and after quickly running out of credits, I decided to leverage OpenCode and NVIDIA Nemotron 3 Ultra. After Nemotron 3 struggled for a few days on the optimization, I pivoted to Claude Opus (and more recently Fable) to optimize the kernel, eventually reaching parity in bfloat16. The full exploration is documented in `docs/ai/ai_assisted_optimizations_and_benchmarks.md`. My initial results are documented below:

!['Bad Times'](figures/benchmark_gpu_b4_t1024_2026-06-30_0909_NVIDIA-GB10_DGX-Spark.png)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of notable changes to this project.

## License

This project is licensed under the [MIT License](LICENSE).
