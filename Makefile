# Source roots only — never `find .` (crawls .pixi and hangs on iCloud).
MOJO_PATHS := train_gpt2.mojo profile_gpt2.mojo llmm tests
PYTHON_PATHS := train_gpt2.py profile_gpt2.py scripts tests data
LATEX_SOURCES := docs/backprop.tex
 
# Auto-detect python library for Mojo standard library python interop.
# Keep this a repo-relative path — absolute iCloud paths contain spaces and
# break Mojo's libpython discovery if MOJO_PYTHON_LIBRARY is unset at runtime.
MOJO_PYTHON_LIBRARY ?= $(shell find .pixi/envs/default/lib -maxdepth 1 \( -name 'libpython3*.dylib' -o -name 'libpython3*.so' \) -print -quit 2>/dev/null)
export MOJO_PYTHON_LIBRARY

TRAIN_MOJO_SRC := train_gpt2.mojo
TRAIN_BIN := build/train_gpt2
TRAIN_RUNNER := scripts/run_train_gpt2.sh
MOJO_INCLUDES := -I .
# Link libm explicitly: from-scratch weight init (llmm/rand.mojo) draws Gaussian
# noise via sinf/cosf, which the backend fuses into sincosf — a libm symbol that
# the AOT linker otherwise reports as "DSO missing from command line".
MOJO_LINK_FLAGS := -Xlinker -lm
LLMM_SOURCES := $(shell find llmm -name '*.mojo' 2>/dev/null)
WORLD_SIZE ?= 1

# Inference-only: load a checkpoint and run autoregressive B=1 generation.
# No backward/optimizer/training-dataloader — a fast iteration loop for the
# generation code path alone (encoder/attention/matmul/sampling), separate
# from the training binary.
INFER_MOJO_SRC := infer_gpt2.mojo
INFER_BIN := build/infer_gpt2
INFER_BIN_BF16 := build/infer_gpt2_bf16

# Profiling: a single forward/backward/update step on synthetic data, built as
# its own binary so external profilers (ncu/nsys) and the Perfetto tracer can
# target it without the full training loop.
PROFILE_MOJO_SRC := profile_gpt2.mojo
PROFILE_BIN := build/profile_gpt2
# bf16 mixed-precision build of the same harness (-D LLMM_BF16=1). GPU-only:
# CPU training is fp32 by policy. Used as the llm.mojo bf16 bar in the GPU
# benchmark.
PROFILE_BIN_BF16 := build/profile_gpt2_bf16
# Separate binary built with -D LLMM_TRACE=1 so the per-thread kernel
# instrumentation is compiled in. The default PROFILE_BIN omits it, so its
# kernels are byte-for-byte the training build (zero tracing overhead) — that is
# the binary used for the throughput numbers.
PROFILE_TRACE_BIN := build/profile_gpt2_trace
PROFILE_RUNNER := scripts/run_profile_gpt2.sh
PROFILE_SCRIPT := profile_gpt2.py
PROFILE_TARGET ?= gpu
# Shared profiling config so llm.mojo and llm.c profile the SAME problem by
# default — an apples-to-apples comparison. The llm.c profile binaries run the
# full 12-layer GPT-2 124M at B=4, T=64, one step; the mojo harness defaults
# (T=1024, 1 layer) would otherwise diverge, so we pin it to match here and feed
# the same B/T into the llm.c argument lists below. Override on the command line,
# e.g. `make profile-ncu PROFILE_T=256 PROFILE_LAYERS=1`.
PROFILE_B ?= 4
PROFILE_T ?= 64
PROFILE_LAYERS ?= 12
PROFILE_STEPS ?= 1
# Env prefix that pins the mojo harness to the shared config (the binary reads
# these LLMM_PROFILE_* vars). ncu/nsys launch the binary with the inherited
# environment, so prefixing the recipe command is enough to reach it.
PROFILE_ENV := LLMM_PROFILE_B=$(PROFILE_B) LLMM_PROFILE_T=$(PROFILE_T) \
	LLMM_PROFILE_LAYERS=$(PROFILE_LAYERS) LLMM_PROFILE_STEPS=$(PROFILE_STEPS)
# Suffix the Perfetto trace with the target (…gpu./…cpu.) so the GPU and CPU
# runs write distinct files instead of clobbering each other. Recursive `=`
# (not `:=`) so it re-expands with any per-target PROFILE_TARGET override.
PROFILE_TRACE = $(PROFILE_BIN).$(PROFILE_TARGET).perfetto-trace.json
# Per-thread CPU trace: every sync_parallelize worker (traced_parallelize) emits
# a span tagged with its OS thread id, so the ~20 CPU worker threads show on
# their own lanes in the Perfetto UI. Written directly by the binary (no nsys,
# no MAX profiler), so it works on macOS and Linux alike.
PROFILE_THREAD_TRACE = $(PROFILE_BIN).$(PROFILE_TARGET).threads.perfetto-trace.json
# nsys report is likewise target-suffixed. nsys' --sample=process-tree +
# --trace=osrt capture every OS worker thread (the ~20 sync_parallelize workers
# the kernels fan out to), which is how the per-thread CPU timeline is surfaced
# — our in-process Perfetto tracer only sees the harness-level phases, not the
# threads spawned inside the kernels. PROFILE_NSYS_ENV/FLAGS are overridden per
# target: gpu needs the cuda pixi env and cuda trace; cpu needs neither.
# Binary that profile-ncu / profile-nsys capture. Defaults to the bf16 build —
# bf16 is the precision we ship and profile by default. The profile-fp32-*
# variants override it to the fp32 build, and the CPU nsys target overrides it
# to the fp32 build too (bf16 is GPU-only). Recursive (`=`) so per-target
# overrides re-expand. Output filenames derive from it, so each precision/target
# writes its own report.
PROFILE_PROF_BIN = $(PROFILE_BIN_BF16)
PROFILE_NSYS_REP = $(PROFILE_PROF_BIN).$(PROFILE_TARGET).nsys-rep
PROFILE_NSYS_ENV := -e cuda
PROFILE_NSYS_FLAGS := --force-overwrite true --trace=cuda,osrt,nvtx --sample=process-tree

# Comparative benchmark against Karpathy's llm.c (git submodule). CPU and GPU are
# separated so the CPU path builds/runs with no CUDA toolchain (e.g. on macOS).
LLMC_DIR := third_party/llm.c
LLMC_CPU_BIN := $(LLMC_DIR)/train_gpt2
LLMC_GPU_BIN := $(LLMC_DIR)/train_gpt2cu          # bf16 CUDA build
LLMC_FP32_GPU_BIN := $(LLMC_DIR)/train_gpt2fp32cu  # fp32 CUDA build
BENCH_SCRIPT := scripts/benchmark_train.py
# Benchmark hyperparameters, fed to every config (Mojo, llm.c CPU+CUDA, PyTorch)
# so the bars stay apples-to-apples. Override on the command line, e.g.
# `make benchmark-gpu BENCH_B=4 BENCH_T=1024`. Defaults match the historical
# B=4, T=64 reference config. T must be <= 1024 (GPT-2's max).
BENCH_B ?= 4
BENCH_T ?= 64
BENCH_CPU_STEPS ?= 40
BENCH_GPU_STEPS ?= 40
# Metal fp32 at B=4 T=1024 is ~6.5 s/step; default to 10 measured steps so the
# full run is ~70 s of GPU time (manageable for a local Mac). Override with
# BENCH_METAL_STEPS=N on the command line.
BENCH_METAL_STEPS ?= 10
# Seconds to sleep between Metal benchmark arms so each arm starts comparably cool
# (M4 Max hits the thermal throttle cliff at ~8 s of sustained load — P16).
# Set to 0 to skip cooldown (useful for quick dev runs with small BENCH_METAL_STEPS).
BENCH_COOLDOWN_S ?= 30
BENCH_ARGS := --batch-size $(BENCH_B) --seq-len $(BENCH_T)
HAVE_NVCC := $(shell command -v nvcc 2>/dev/null)
# Detect Apple Silicon so make benchmark auto-selects the Metal path.
IS_DARWIN := $(shell uname -s 2>/dev/null)
IS_APPLE_SILICON := $(shell [ "$$(uname -s)" = "Darwin" ] && [ "$$(uname -m)" = "arm64" ] && echo 1 || echo 0)

SHELL := /bin/bash

.PHONY: help install install-cuda install-with-data install-cuda-with-data install-hooks data update lint lint-python lint-mojo lint-c lint-cuda lint-latex \
        format format-python format-mojo format-c format-cuda format-latex \
        typecheck check clean build         build-mojo build-train build-bf16 train train-cpu train-metal train-bf16 \
        build-profile build-profile-bf16 profile profile-trace profile-cpu profile-threads-cpu profile-ncu \
        profile-nsys profile-nsys-cpu profile-fp32-ncu profile-fp32-nsys \
        profile-metal \
        build-infer build-infer-bf16 data-hellaswag eval eval-cpu benchmark-eval \
        build-llmc build-llmc-cpu build-llmc-gpu benchmark benchmark-cpu benchmark-gpu benchmark-metal \
        stage-llmc profile-llmc-ncu profile-llmc-nsys \
        profile-llmc-fp32-ncu profile-llmc-fp32-nsys \
        test test-cpu test-cuda test-python test-mojo test-fixtures \
        verify verify-cpu verify-gpu \
        docs docs-clean

.DEFAULT_GOAL := help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  install               Quick setup: pixi deps + git hooks (no dataset/weights)"
	@echo "  install-cuda          Same as install, with CUDA/GPU-enabled pixi dependencies"
	@echo "  install-with-data     install + dataset/weights (needed for train/verify/benchmark)"
	@echo "  install-cuda-with-data  install-cuda + dataset/weights"
	@echo "  data                  Download the Tiny Shakespeare dataset + GPT-2 124M starter weights"
	@echo "                        (idempotent — skips files already present; ~1.5 GB total)"
	@echo "  install-hooks         Install git pre-commit/pre-push hooks (make lint / make check)"
	@echo "  update                Update pixi dependencies and refresh pixi.lock"
	@echo ""
	@echo "Quality gates:"
	@echo "  check         Run lint (incl. typecheck), build-mojo, build train_gpt2, and build-profile"
	@echo "  build         Compile train_gpt2.mojo to build/train_gpt2"
	@echo "  build-train   Alias for build"
	@echo "  train         Build and run build/train_gpt2 (sets MOJO_PYTHON_LIBRARY)"
	@echo "  train-cpu     Build and run build/train_gpt2 on CPU (LLMM_USE_CPU=1)"
	@echo "  train-metal   Build (WORLD_SIZE=1) and run on Apple Metal GPU (experimental)"
	@echo "  build-bf16    Compile train_gpt2.mojo (-D LLMM_BF16) to build/train_gpt2_bf16"
	@echo "  train-bf16    Build and run build/train_gpt2_bf16 (ARGS=\"...\" for training flags)"
	@echo ""
	@echo "Profiling:"
	@echo "  build-profile Compile profile_gpt2.mojo to build/profile_gpt2"
	@echo "  build-profile-bf16  Compile the bf16 (-D LLMM_BF16) harness, build/profile_gpt2_bf16"
	@echo "  profile       Run one step and emit a Perfetto trace (alias: profile-trace)"
	@echo "  profile-trace Write build/profile_gpt2.<target>.perfetto-trace.json (ui.perfetto.dev)"
	@echo "  profile-metal Run one step on the Metal GPU and emit a Perfetto trace (Apple Silicon)"
	@echo "  profile-cpu   Run the profile on CPU and emit a Perfetto trace"
	@echo "  profile-threads-cpu  CPU per-thread Perfetto trace (all worker threads)"
	@echo "  profile-ncu   Profile Mojo bf16 GPU kernels with ncu (NVIDIA only)"
	@echo "  profile-nsys  Capture a GPU nsys timeline of Mojo bf16 kernels (NVIDIA only)"
	@echo "  profile-fp32-ncu   Same ncu table for the fp32 build (NVIDIA only)"
	@echo "  profile-fp32-nsys  fp32 GPU nsys timeline (NVIDIA only)"
	@echo "  profile-nsys-cpu  Capture a CPU nsys timeline showing all worker threads"
	@echo "  NOTE: profile-ncu / profile-nsys / profile-llmc-* require NVIDIA Nsight tools"
	@echo "        and are not available on Apple Silicon (darwin)."
	@echo ""
	@echo "Inference & eval:"
	@echo "  build-infer      Compile infer_gpt2.mojo to build/infer_gpt2 (fp32/CPU-capable)"
	@echo "  build-infer-bf16 Compile the bf16 (-D LLMM_BF16) inference binary, build/infer_gpt2_bf16"
	@echo "  data-hellaswag   Download + tokenize the HellaSwag val split (data/hellaswag.py)"
	@echo "  eval             Build (bf16) + score CHECKPOINT on HellaSwag (default"
	@echo "                   CHECKPOINT=log124M/model_19552.bin; override on the command line,"
	@echo "                   e.g. make eval CHECKPOINT=log124M/model_5000.bin EVAL_B=32)"
	@echo "  eval-cpu         Same as eval, but the fp32/CPU inference binary"
	@echo "  benchmark-eval   Score + render the llm.mojo-vs-llm.c HellaSwag comparison"
	@echo "                   chart into figures/ (make eval + Wilson CI + plot)"
	@echo ""
	@echo "Benchmark (vs llm.c submodule on NVIDIA; vs PyTorch MPS on Apple Silicon):"
	@echo "  build-llmc    Build llm.c CPU (train_gpt2) + CUDA (train_gpt2cu, if nvcc)"
	@echo "  build-llmc-cpu  Build only the llm.c CPU reference (portable, macOS-ok)"
	@echo "  build-llmc-gpu  Build only the llm.c CUDA reference (needs nvcc)"
	@echo "  benchmark     Histogram of train-loop time (auto: Metal on Apple, NVIDIA GPU on Linux)"
	@echo "  benchmark-cpu Only the CPU comparison (no CUDA / Metal needed)"
	@echo "  benchmark-gpu Only the NVIDIA GPU comparison"
	@echo "  benchmark-metal  Apple Silicon Metal GPU: llm.mojo fp32+bf16 vs PyTorch MPS fp32+bf16"
	@echo "                   (4 arms in one graph, mirroring benchmark-gpu's fp32+bf16 layout)"
	@echo "                   llm.c has no Metal port — baseline is PyTorch MPS"
	@echo "                   (hyperparams: BENCH_B, BENCH_T, BENCH_METAL_STEPS, BENCH_COOLDOWN_S)"
	@echo "  profile-llmc-ncu  Profile llm.c bf16 CUDA kernels with ncu (NVIDIA only)"
	@echo "  profile-llmc-nsys Capture an nsys timeline of llm.c bf16 CUDA kernels (NVIDIA only)"
	@echo "  profile-llmc-fp32-ncu   Profile llm.c fp32 CUDA kernels with ncu (NVIDIA only)"
	@echo "  profile-llmc-fp32-nsys  nsys timeline of llm.c fp32 CUDA kernels (NVIDIA only)"
	@echo "  lint          Lint Python, Mojo, C, CUDA, LaTeX sources, and typecheck"
	@echo "  lint-python   Lint Python sources with ruff"
	@echo "  lint-mojo     Lint Mojo sources with mojo format --check"
	@echo "  lint-c        Lint C sources with clang-format and clang-tidy"
	@echo "  lint-cuda     Lint CUDA sources with clang-format and clang-tidy"
	@echo "  lint-latex    Lint LaTeX sources with latexindent (check only)"
	@echo "  typecheck     Type-check Python sources with pyrefly"
	@echo "  build-mojo    Precompile llmm.mojoc into the test cache; surfaces Mojo warnings"
	@echo ""
	@echo "Formatting:"
	@echo "  format        Format Python, Mojo, C, CUDA, and LaTeX sources"
	@echo "  format-python Format Python sources with ruff"
	@echo "  format-mojo   Format Mojo sources with mojo format"
	@echo "  format-c      Format C sources with clang-format"
	@echo "  format-cuda   Format CUDA sources with clang-format"
	@echo "  format-latex  Format LaTeX sources with latexindent"
	@echo ""
	@echo "Testing:"
	@echo "  test          Run Mojo + Python test suites"
	@echo "  test-mojo     Run pure-Mojo unit tests with mojo test"
	@echo "  test-python   Run pytest equivalence + property tests"
	@echo "  test-python-cuda Run GPU/accelerator pytest equivalence + property tests"
	@echo "  test-fixtures Regenerate tests/fixtures/*.npz from PyTorch reference"
	@echo "  verify        Verify both CPU and GPU versions against reference state"
	@echo "  verify-cpu    Verify CPU version against reference state"
	@echo "  verify-gpu    Verify GPU version against reference state"
	@echo ""
	@echo "Documents:"
	@echo "  docs          Build docs/backprop.pdf with latexmk"
	@echo "  docs-clean    Remove LaTeX build artifacts (keeps the PDF)"
	@echo ""
	@echo "Housekeeping:"
	@echo "  help          Show this help message"
	@echo "  clean         Remove cache directories"

# `install`/`install-cuda` are the quick path: pixi deps + git hooks only, no
# network-heavy dataset/weights download. Use `install-with-data`/
# `install-cuda-with-data` for that (needed before `make train`/`make verify`/
# `make benchmark*` will work) — or run `make data` standalone at any point.
install:
	pixi install
	@$(MAKE) install-hooks

install-cuda:
	pixi install -e cuda
	@$(MAKE) install-hooks

install-with-data: install
	@$(MAKE) data

install-cuda-with-data: install-cuda
	@$(MAKE) data

update:
	pixi update

# ---------------------------------------------------------------------------
# Dataset + starter-weights download. One-time (~1.5 GB); each file is a real
# Make target so already-downloaded files are skipped on re-run. Needed before
# `make train`, `make verify`, or `make benchmark*` will work.
# ---------------------------------------------------------------------------
STARTER_PACK_URL := https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main
STARTER_PACK_FILES := gpt2_tokenizer.bin gpt2_124M.bin gpt2_124M_bf16.bin gpt2_124M_debug_state.bin

data: data/.tinyshakespeare/tiny_shakespeare_train.bin $(STARTER_PACK_FILES)
	@echo "Data ready: tokenized Tiny Shakespeare + GPT-2 124M starter weights."

data/.tinyshakespeare/tiny_shakespeare_train.bin:
	pixi run python data/tinyshakespeare.py

$(STARTER_PACK_FILES):
	curl -fL -o $@ "$(STARTER_PACK_URL)/$@?download=true"

# Writes git hook shims (respecting core.hooksPath, whatever it's set to)
# that run `pre-commit run --hook-stage ...` per .pre-commit-config.yaml.
# Tolerant: skips (doesn't fail `make install`) if pre-commit isn't on PATH.
install-hooks:
	@if command -v pre-commit >/dev/null 2>&1; then \
		hooks_dir=$$(git rev-parse --git-path hooks); \
		mkdir -p "$$hooks_dir"; \
		printf '#!/usr/bin/env bash\nexec pre-commit run --hook-stage pre-commit\n' > "$$hooks_dir/pre-commit"; \
		chmod +x "$$hooks_dir/pre-commit"; \
		printf '#!/usr/bin/env bash\nexec pre-commit run --hook-stage pre-push\n' > "$$hooks_dir/pre-push"; \
		chmod +x "$$hooks_dir/pre-push"; \
		echo "Installed git hooks: pre-commit -> make lint, pre-push -> make check"; \
	else \
		echo "pre-commit not found on PATH — skipping git hook install."; \
		echo "  Install it (e.g. 'pipx install pre-commit') then run: make install-hooks"; \
	fi

check: lint build-mojo build build-profile

# Compiles the GPT-2 training binary. MOJO_PYTHON_LIBRARY must be set because
# DataLoader uses Python glob; pixi run supplies the Modular std/toolchain env.
build build-train: $(TRAIN_BIN)

$(TRAIN_BIN): $(TRAIN_MOJO_SRC) $(LLMM_SOURCES)
	@mkdir -p build
	pixi run mojo build -D WORLD_SIZE=$(WORLD_SIZE) $(MOJO_INCLUDES) $(MOJO_LINK_FLAGS) -o $(TRAIN_BIN) $(TRAIN_MOJO_SRC)

# bf16 mixed-precision training binary (GPU-only — see the bf16-build-needs-
# gpu-only-dispatch guard in train_gpt2.mojo). All training flags are passed
# through scripts/run_train_gpt2_bf16.sh; there are no baked-in hyperparameters.
TRAIN_BIN_BF16 := build/train_gpt2_bf16
TRAIN_RUNNER_BF16 := scripts/run_train_gpt2_bf16.sh

build-bf16: $(TRAIN_BIN_BF16)

$(TRAIN_BIN_BF16): $(TRAIN_MOJO_SRC) $(LLMM_SOURCES)
	@mkdir -p build
	pixi run mojo build -D WORLD_SIZE=$(WORLD_SIZE) -D LLMM_BF16=1 $(MOJO_INCLUDES) $(MOJO_LINK_FLAGS) -o $(TRAIN_BIN_BF16) $(TRAIN_MOJO_SRC)

train: $(TRAIN_BIN) $(TRAIN_RUNNER)
	@$(TRAIN_RUNNER)

train-cpu: $(TRAIN_BIN) $(TRAIN_RUNNER)
	@LLMM_USE_CPU=1 $(TRAIN_RUNNER)

# Pass training flags via ARGS, e.g. `make train-bf16 ARGS="-i ... -j ... -e d12 ..."`
train-bf16: $(TRAIN_BIN_BF16) $(TRAIN_RUNNER_BF16)
	@$(TRAIN_RUNNER_BF16) $(ARGS)

# Metal (Apple GPU) training: force WORLD_SIZE=1 because multi-GPU collectives
# are NVIDIA-only. Metal is the default device on Apple Silicon (no extra flags
# needed); this target just pins the world size and makes intent explicit.
train-metal: WORLD_SIZE := 1
train-metal: $(TRAIN_BIN) $(TRAIN_RUNNER)
	@$(TRAIN_RUNNER)

# Compiles the single-step profiling harness. Depends on train_gpt2.mojo because
# it imports GPT2 from it (the llm.mojo analogue of profile_gpt2.cu #include'ing
# train_gpt2.cu).
build-profile: $(PROFILE_BIN)

$(PROFILE_BIN): $(PROFILE_MOJO_SRC) $(TRAIN_MOJO_SRC) $(LLMM_SOURCES)
	@mkdir -p build
	pixi run mojo build -D WORLD_SIZE=$(WORLD_SIZE) $(MOJO_INCLUDES) $(MOJO_LINK_FLAGS) -o $(PROFILE_BIN) $(PROFILE_MOJO_SRC)

# bf16 mixed-precision build of the harness (params/acts/grads bf16, fp32 master
# weights + optimizer moments). GPU-only by policy; see profile_gpt2.mojo.
build-profile-bf16: $(PROFILE_BIN_BF16)

$(PROFILE_BIN_BF16): $(PROFILE_MOJO_SRC) $(TRAIN_MOJO_SRC) $(LLMM_SOURCES)
	@mkdir -p build
	pixi run mojo build -D WORLD_SIZE=$(WORLD_SIZE) -D LLMM_BF16=1 $(MOJO_INCLUDES) $(MOJO_LINK_FLAGS) -o $(PROFILE_BIN_BF16) $(PROFILE_MOJO_SRC)

build-infer: $(INFER_BIN)

$(INFER_BIN): $(INFER_MOJO_SRC) $(TRAIN_MOJO_SRC) $(LLMM_SOURCES)
	@mkdir -p build
	pixi run mojo build -D WORLD_SIZE=$(WORLD_SIZE) $(MOJO_INCLUDES) $(MOJO_LINK_FLAGS) -o $(INFER_BIN) $(INFER_MOJO_SRC)

build-infer-bf16: $(INFER_BIN_BF16)

$(INFER_BIN_BF16): $(INFER_MOJO_SRC) $(TRAIN_MOJO_SRC) $(LLMM_SOURCES)
	@mkdir -p build
	pixi run mojo build -D WORLD_SIZE=$(WORLD_SIZE) -D LLMM_BF16=1 $(MOJO_INCLUDES) $(MOJO_LINK_FLAGS) -o $(INFER_BIN_BF16) $(INFER_MOJO_SRC)

# HellaSwag eval: `make eval` builds the bf16 inference binary, ensures the
# eval data is tokenized (see data/hellaswag.py), and scores a checkpoint.
# Override on the command line, e.g. `make eval CHECKPOINT=log124M/model_5000.bin`.
CHECKPOINT ?= log124M/model_19552.bin
EVAL_BIN := data/.hellaswag/hellaswag_val.bin
EVAL_B ?= 64
EVAL_T ?= 512

data-hellaswag: $(EVAL_BIN)

$(EVAL_BIN):
	pixi run python data/hellaswag.py

eval: build-infer-bf16 $(EVAL_BIN)
	./$(INFER_BIN_BF16) --eval $(CHECKPOINT) $(EVAL_BIN) $(EVAL_B) $(EVAL_T)

eval-cpu: build-infer $(EVAL_BIN)
	./$(INFER_BIN) --eval $(CHECKPOINT) $(EVAL_BIN) $(EVAL_B) $(EVAL_T)

# Runs `make eval`, computes a Wilson CI, and renders the llm.mojo-vs-llm.c
# comparison chart into figures/ (see scripts/benchmark_eval.py's docstring).
benchmark-eval:
	pixi run python scripts/benchmark_eval.py

# Tracing build: same harness, with the per-thread kernel instrumentation
# compiled in via -D LLMM_TRACE=1.
$(PROFILE_TRACE_BIN): $(PROFILE_MOJO_SRC) $(TRAIN_MOJO_SRC) $(LLMM_SOURCES)
	@mkdir -p build
	pixi run mojo build -D WORLD_SIZE=$(WORLD_SIZE) -D LLMM_TRACE=1 $(MOJO_INCLUDES) $(MOJO_LINK_FLAGS) -o $(PROFILE_TRACE_BIN) $(PROFILE_MOJO_SRC)

# Run one forward/backward/update step and emit a Chrome-trace-format JSON of the
# high-level phases, loadable directly at https://ui.perfetto.dev. Set
# PROFILE_TARGET=cpu to profile the CPU path.
profile profile-trace: $(PROFILE_BIN) $(PROFILE_RUNNER)
	@LLMM_PROFILE_TRACE=$(PROFILE_TRACE) $(PROFILE_RUNNER) $(PROFILE_TARGET)
	@echo "Perfetto trace: $(PROFILE_TRACE) (open at https://ui.perfetto.dev)"

# Same as profile-trace but forces the CPU path (mirrors train-cpu), so the
# trace can be captured on machines without a GPU. The target-specific
# PROFILE_TARGET also redirects PROFILE_TRACE to the …cpu. filename.
profile-cpu: PROFILE_TARGET := cpu
profile-cpu: $(PROFILE_BIN) $(PROFILE_RUNNER)
	@LLMM_PROFILE_TRACE=$(PROFILE_TRACE) $(PROFILE_RUNNER) $(PROFILE_TARGET)
	@echo "Perfetto trace: $(PROFILE_TRACE) (open at https://ui.perfetto.dev)"

# Per-thread CPU trace: shows every sync_parallelize worker on its own OS-thread
# lane in the Perfetto UI. Cross-platform (no nsys). Uses the -D LLMM_TRACE build.
profile-threads-cpu: PROFILE_TARGET := cpu
profile-threads-cpu: $(PROFILE_TRACE_BIN) $(PROFILE_RUNNER)
	@PROFILE_EXE=$(PROFILE_TRACE_BIN) LLMM_THREAD_TRACE=$(PROFILE_THREAD_TRACE) $(PROFILE_RUNNER) $(PROFILE_TARGET)
	@echo "Per-thread Perfetto trace: $(PROFILE_THREAD_TRACE) (open at https://ui.perfetto.dev)"

# Metal GPU Perfetto trace (Apple Silicon). The Mojo profiling harness runs its
# 'gpu' target on Metal automatically — no extra flags needed. This is a thin
# alias for `make profile PROFILE_TARGET=gpu` that makes the intent explicit and
# is discoverable from `make help`. The Perfetto trace is written to
# build/profile_gpt2.gpu.perfetto-trace.json (same path as the Linux GPU trace).
profile-metal: $(PROFILE_BIN) $(PROFILE_RUNNER)
	@LLMM_PROFILE_TRACE=$(PROFILE_TRACE) $(PROFILE_RUNNER) gpu
	@echo "Metal Perfetto trace: $(PROFILE_BIN).gpu.perfetto-trace.json (open at https://ui.perfetto.dev)"

# Per-kernel GPU profile via NVIDIA Nsight Compute (ncu), printed as a table by
# profile_gpt2.py. Defaults to the bf16 build (PROFILE_PROF_BIN); use
# profile-fp32-ncu for the fp32 build. Add `--full` for the heavy metric set,
# `--sudo` if the GPU performance counters need elevated access (DRAM/tensor
# metrics otherwise show as n/a). The raw CSV is saved alongside the binary.
# NOTE: ncu is an NVIDIA-only tool. On Apple Silicon (darwin) this target prints
# a notice and exits cleanly instead of failing.
profile-ncu: $(PROFILE_PROF_BIN) $(PROFILE_SCRIPT)
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		echo "SKIPPED: profile-ncu requires NVIDIA Nsight Compute (ncu) — not available on Apple Silicon."; \
		echo "  Use 'make profile-metal' for a Perfetto trace on the Metal GPU."; \
	else \
		$(PROFILE_ENV) pixi run -e cuda python $(PROFILE_SCRIPT) \
			--exe $(PROFILE_PROF_BIN) --target $(PROFILE_TARGET) \
			--output $(PROFILE_PROF_BIN).ncu.csv; \
	fi

# fp32 build of the same per-kernel profile (mirrors profile-llmc-fp32-ncu).
# NOTE: NVIDIA-only. See profile-ncu for the darwin guard note.
profile-fp32-ncu: PROFILE_PROF_BIN := $(PROFILE_BIN)
profile-fp32-ncu: $(PROFILE_BIN) $(PROFILE_SCRIPT)
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		echo "SKIPPED: profile-fp32-ncu requires NVIDIA ncu — not available on Apple Silicon."; \
		echo "  Use 'make profile-metal' for a Perfetto trace on the Metal GPU."; \
	else \
		$(PROFILE_ENV) pixi run -e cuda python $(PROFILE_SCRIPT) \
			--exe $(PROFILE_BIN) --target $(PROFILE_TARGET) \
			--output $(PROFILE_BIN).ncu.csv; \
	fi

# Timeline capture via NVIDIA Nsight Systems. The report's thread timeline shows
# every CPU worker thread (the ~20 sync_parallelize workers), which the in-process
# Perfetto tracer cannot see. Open the .nsys-rep in the Nsight Systems UI
# (File > Open). Defaults to the bf16 build; use profile-fp32-nsys for fp32, or
# profile-nsys-cpu for the CPU path.
# NOTE: nsys is an NVIDIA-only tool. On Apple Silicon (darwin) this target prints
# a notice and exits cleanly instead of failing.
profile-nsys: $(PROFILE_PROF_BIN)
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		echo "SKIPPED: profile-nsys requires NVIDIA Nsight Systems (nsys) — not available on Apple Silicon."; \
		echo "  Use 'make profile-metal' for a Perfetto trace on the Metal GPU."; \
	else \
		$(PROFILE_ENV) pixi run $(PROFILE_NSYS_ENV) nsys profile $(PROFILE_NSYS_FLAGS) \
			-o $(PROFILE_PROF_BIN).$(PROFILE_TARGET) $(PROFILE_PROF_BIN) $(PROFILE_TARGET); \
		echo "nsys report: $(PROFILE_NSYS_REP) (per-thread CPU timeline in Nsight Systems)"; \
	fi

# fp32 build of the GPU timeline.
# NOTE: NVIDIA-only. See profile-nsys for the darwin guard note.
profile-fp32-nsys: PROFILE_PROF_BIN := $(PROFILE_BIN)
profile-fp32-nsys: $(PROFILE_BIN)
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		echo "SKIPPED: profile-fp32-nsys requires NVIDIA nsys — not available on Apple Silicon."; \
		echo "  Use 'make profile-metal' for a Perfetto trace on the Metal GPU."; \
	else \
		$(PROFILE_ENV) pixi run $(PROFILE_NSYS_ENV) nsys profile $(PROFILE_NSYS_FLAGS) \
			-o $(PROFILE_PROF_BIN).$(PROFILE_TARGET) $(PROFILE_PROF_BIN) $(PROFILE_TARGET); \
		echo "nsys report: $(PROFILE_NSYS_REP) (per-thread CPU timeline in Nsight Systems)"; \
	fi

# CPU thread timeline — runs in the default pixi env (no GPU required) and traces
# only OS-runtime + CPU samples, so the ~20 worker threads show on the timeline.
# bf16 is GPU-only, so the CPU path profiles the fp32 build.
profile-nsys-cpu: PROFILE_TARGET := cpu
profile-nsys-cpu: PROFILE_PROF_BIN := $(PROFILE_BIN)
profile-nsys-cpu: PROFILE_NSYS_ENV :=
profile-nsys-cpu: PROFILE_NSYS_FLAGS := --force-overwrite true --trace=osrt --sample=process-tree
profile-nsys-cpu: $(PROFILE_BIN)
	$(PROFILE_ENV) pixi run $(PROFILE_NSYS_ENV) nsys profile $(PROFILE_NSYS_FLAGS) \
		-o $(PROFILE_PROF_BIN).$(PROFILE_TARGET) $(PROFILE_PROF_BIN) $(PROFILE_TARGET)
	@echo "nsys report: $(PROFILE_NSYS_REP) (per-thread CPU timeline in Nsight Systems)"

# ---------------------------------------------------------------------------
# llm.c comparison (git submodule third_party/llm.c). CPU and GPU split so the
# CPU half needs no CUDA toolchain — it builds and runs on macOS.
# ---------------------------------------------------------------------------

# Ensure the submodule is checked out (no-op once present).
$(LLMC_DIR)/train_gpt2.c:
	git submodule update --init $(LLMC_DIR)

# CPU reference (OpenMP). Portable: builds on Linux and macOS.
build-llmc-cpu: $(LLMC_DIR)/train_gpt2.c
	$(MAKE) -C $(LLMC_DIR) train_gpt2

# CUDA reference. Skipped with a notice when no nvcc is present (e.g. macOS).
# NO_MULTI_GPU=1 drops the NCCL dependency; the sed makes one CUDA-13 API call
# (cudaMemAdvise, in a dead OOM-fallback path) compile against the newer
# cudaMemLocation signature without touching the committed submodule history.
build-llmc-gpu: $(LLMC_DIR)/train_gpt2.c
ifeq ($(HAVE_NVCC),)
	@echo "nvcc not found — skipping llm.c CUDA build (CPU-only host)."
else
	@if ! grep -q cudaMemLocationTypeHost $(LLMC_DIR)/llmc/cuda_utils.cuh; then \
		sed -i.bak 's/cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId)/cudaMemAdviseSetPreferredLocation, cudaMemLocation{cudaMemLocationTypeHost, 0})/' \
			$(LLMC_DIR)/llmc/cuda_utils.cuh && rm -f $(LLMC_DIR)/llmc/cuda_utils.cuh.bak; \
	fi
	$(MAKE) -C $(LLMC_DIR) train_gpt2cu NO_MULTI_GPU=1
endif

# Build both halves (GPU half self-skips without nvcc).
build-llmc: build-llmc-cpu build-llmc-gpu

# Benchmarks: histogram of per-step train-loop time, llm.mojo vs llm.c.
benchmark-cpu: $(PROFILE_BIN) build-llmc-cpu $(BENCH_SCRIPT)
	pixi run python $(BENCH_SCRIPT) --device cpu $(BENCH_ARGS) \
		--cpu-steps $(BENCH_CPU_STEPS)

benchmark-gpu: $(PROFILE_BIN) $(PROFILE_BIN_BF16) build-llmc-gpu $(BENCH_SCRIPT)
	pixi run python $(BENCH_SCRIPT) --device gpu $(BENCH_ARGS) \
		--gpu-steps $(BENCH_GPU_STEPS)

# Apple Silicon Metal GPU benchmark: llm.mojo fp32+bf16 vs PyTorch MPS fp32+bf16.
# 4 arms in one graph, mirroring how benchmark-gpu combines fp32+bf16 for NVIDIA.
# No llm.c dependency — llm.c has no Metal port; the baseline is PyTorch MPS.
# Hyperparams: BENCH_B, BENCH_T, BENCH_METAL_STEPS (default 10 due to ~6.5 s/step),
#              BENCH_COOLDOWN_S (default 30 s — thermal discipline for M4 Max).
benchmark-metal: $(PROFILE_BIN) $(PROFILE_BIN_BF16) $(BENCH_SCRIPT)
	pixi run python $(BENCH_SCRIPT) --device metal $(BENCH_ARGS) \
		--metal-steps $(BENCH_METAL_STEPS) --cooldown-s $(BENCH_COOLDOWN_S)

# Auto mode: on Apple Silicon run the Metal benchmark; on NVIDIA run CPU + GPU.
ifeq ($(IS_APPLE_SILICON),1)
benchmark: $(PROFILE_BIN) $(PROFILE_BIN_BF16) $(BENCH_SCRIPT)
	@echo "Apple Silicon detected — running Metal benchmark (llm.c CUDA not available)."
	@echo "  llm.c has no Metal port; baseline is PyTorch MPS."
	@echo "  4 arms: llm.mojo fp32+bf16, PyTorch MPS fp32+bf16."
	pixi run python $(BENCH_SCRIPT) --device auto $(BENCH_ARGS) \
		--cpu-steps $(BENCH_CPU_STEPS) --metal-steps $(BENCH_METAL_STEPS) \
		--cooldown-s $(BENCH_COOLDOWN_S)
else
benchmark: $(PROFILE_BIN) $(PROFILE_BIN_BF16) build-llmc $(BENCH_SCRIPT)
	pixi run python $(BENCH_SCRIPT) --device auto $(BENCH_ARGS) \
		--cpu-steps $(BENCH_CPU_STEPS) --gpu-steps $(BENCH_GPU_STEPS)
endif

# One short, deterministic train step of llm.c's CUDA build — the target for
# profiling Karpathy's GPU kernels. Same B/T as our harness (shared PROFILE_B/T)
# for an apples-to-apples comparison; data paths are relative to the staged
# build/llmc cwd. Recursive `=` so PROFILE_B/T overrides flow through.
LLMC_GPU_ARGS = -e gpt2_124M_bf16.bin \
	-i dev/data/tinyshakespeare/tiny_shakespeare_train.bin \
	-j dev/data/tinyshakespeare/tiny_shakespeare_val.bin \
	-b $(PROFILE_B) -t $(PROFILE_T) -x $(PROFILE_STEPS) -v 0 -s 0 -l 0

stage-llmc: $(BENCH_SCRIPT)
	pixi run python $(BENCH_SCRIPT) --stage-only --device gpu

# Profile Karpathy's CUDA kernels with ncu, through the SAME analysis table as
# our Mojo kernels (profile_gpt2.py categorizes both by kernel name). Compare
# against `make profile-ncu` (our kernels). NVIDIA-only — not available on Apple.
profile-llmc-ncu: build-llmc-gpu stage-llmc $(PROFILE_SCRIPT)
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		echo "SKIPPED: profile-llmc-ncu requires NVIDIA ncu — not available on Apple Silicon."; \
	else \
		pixi run -e cuda python $(PROFILE_SCRIPT) \
			--exe $(abspath $(LLMC_GPU_BIN)) \
			--exe-args "$(LLMC_GPU_ARGS)" \
			--cwd build/llmc \
			--output build/profile_llmc.ncu.csv; \
	fi

# nsys timeline of llm.c's CUDA training step. NVIDIA-only — not available on Apple.
profile-llmc-nsys: build-llmc-gpu stage-llmc
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		echo "SKIPPED: profile-llmc-nsys requires NVIDIA nsys — not available on Apple Silicon."; \
	else \
		cd build/llmc && pixi run -e cuda nsys profile --force-overwrite true \
			--trace=cuda,nvtx,osrt -o $(abspath build/profile_llmc) \
			$(abspath $(LLMC_GPU_BIN)) $(LLMC_GPU_ARGS); \
		echo "nsys report: build/profile_llmc.nsys-rep"; \
	fi

# llm.c's *fp32* CUDA build (train_gpt2fp32cu). Unlike the bf16 build it has no
# -e / -x flags: it loads gpt2_124M.bin and runs a full epoch. We point the data
# at the small val bin and cap ncu to one step's worth of kernel launches with
# --launch-count, so the per-kernel table mirrors `make profile-ncu` (our fp32
# GPU kernels) without ncu replaying the whole epoch.
LLMC_FP32_GPU_ARGS = \
	-i dev/data/tinyshakespeare/tiny_shakespeare_val.bin \
	-j dev/data/tinyshakespeare/tiny_shakespeare_val.bin \
	-b $(PROFILE_B) -t $(PROFILE_T) -v 100000 -s 100000
# One 12-layer fwd+bwd+update is a few hundred kernels; 400 covers it and the
# table aggregates by kernel name, so the exact cap is not sensitive.
LLMC_FP32_LAUNCH_COUNT := 400

profile-llmc-fp32-ncu: build-llmc-gpu stage-llmc $(PROFILE_SCRIPT)
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		echo "SKIPPED: profile-llmc-fp32-ncu requires NVIDIA ncu — not available on Apple Silicon."; \
	else \
		pixi run -e cuda python $(PROFILE_SCRIPT) \
			--exe $(abspath $(LLMC_FP32_GPU_BIN)) \
			--exe-args "$(LLMC_FP32_GPU_ARGS)" \
			--launch-count $(LLMC_FP32_LAUNCH_COUNT) \
			--cwd build/llmc \
			--output build/profile_llmc_fp32.ncu.csv; \
	fi

# nsys timeline of llm.c's fp32 CUDA training step. NVIDIA-only — not available on Apple.
profile-llmc-fp32-nsys: build-llmc-gpu stage-llmc
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		echo "SKIPPED: profile-llmc-fp32-nsys requires NVIDIA nsys — not available on Apple Silicon."; \
	else \
		cd build/llmc && pixi run -e cuda nsys profile --force-overwrite true \
			--trace=cuda,nvtx,osrt -o $(abspath build/profile_llmc_fp32) \
			$(abspath $(LLMC_FP32_GPU_BIN)) $(LLMC_FP32_GPU_ARGS); \
		echo "nsys report: build/profile_llmc_fp32.nsys-rep"; \
	fi

# Builds llmm.mojoc into the persistent cache the pytest suite consumes
# (tests/.mef_cache/<source-fingerprint>/llmm.mojoc, via the bridge so
# the fingerprint logic lives in one place). Content-addressed: a no-op
# when sources are unchanged, so chaining it before test-python costs
# nothing on warm runs while keeping the two steps independently runnable.
build-mojo:
	@if [ -d llmm ]; then \
		pixi run python -m tests._max_bridge; \
	else \
		echo "No llmm package found, skipping mojo build."; \
	fi

# C/CUDA passes (lint-c lint-cuda) dropped from the default flow: the only
# C/CUDA sources are the vendored llm.c submodule (third_party/), built solely
# for profiling and benchmark comparison — we don't lint/format it. Run
# `make lint-c` / `make lint-cuda` directly if first-party C/CUDA is ever added.
lint: lint-python lint-mojo lint-latex typecheck

lint-python:
	@uvx ruff check $(PYTHON_PATHS)
	@uvx ruff format --check $(PYTHON_PATHS)

lint-mojo:
	@fail=0; \
	tmpdir=$$(mktemp -d); \
	trap "rm -rf $$tmpdir" EXIT; \
	i=0; \
	while IFS= read -r -d '' f; do \
		i=$$((i+1)); \
		tmp="$$tmpdir/file_$$i.mojo"; \
		cp "$$f" "$$tmp"; \
		pixi run mojo format -q "$$tmp"; \
		if ! diff -q "$$f" "$$tmp" >/dev/null; then \
			echo "needs formatting: $$f"; \
			fail=1; \
		fi; \
	done < <(find $(MOJO_PATHS) -name '*.mojo' -print0); \
	if [ $$i -eq 0 ]; then \
		echo "No .mojo files found, skipping mojo lint."; \
	else \
		exit $$fail; \
	fi

lint-c:
	@files=$$(find llmm tests docs -name '*.c' -o -name '*.h' 2>/dev/null); \
	if [ -n "$$files" ]; then \
		clang-format --dry-run --Werror $$files; \
		clang-tidy $$files --; \
	else \
		echo "No .c/.h files found, skipping c lint."; \
	fi

lint-cuda:
	@files=$$(find llmm tests docs -name '*.cu' -o -name '*.cuh' 2>/dev/null); \
	if [ -n "$$files" ]; then \
		clang-format --dry-run --Werror $$files; \
		clang-tidy $$files -- -x cuda; \
	else \
		echo "No .cu/.cuh files found, skipping cuda lint."; \
	fi

# latexindent insists on writing logs to cwd, so both latex targets run it
# against a scratch copy and diff/copy back (same trick as lint-mojo).
lint-latex:
	@if ! command -v latexindent >/dev/null 2>&1; then \
		echo "latexindent not installed, skipping latex lint."; \
	elif [ -n "$(LATEX_SOURCES)" ]; then \
		fail=0; \
		tmpdir=$$(mktemp -d); \
		trap "rm -rf $$tmpdir" EXIT; \
		for f in $(LATEX_SOURCES); do \
			latexindent -s -l docs/latexindent.yaml -g "$$tmpdir/indent.log" \
				-o "$$tmpdir/out.tex" "$$f"; \
			if ! diff -q "$$f" "$$tmpdir/out.tex" >/dev/null; then \
				echo "needs formatting: $$f"; \
				fail=1; \
			fi; \
		done; \
		exit $$fail; \
	else \
		echo "No .tex files found, skipping latex lint."; \
	fi

# format-c / format-cuda dropped from the default flow (see the `lint:` note).
format: format-python format-mojo format-latex

format-python:
	@uvx ruff check --fix $(PYTHON_PATHS)
	@uvx ruff format $(PYTHON_PATHS)

format-mojo:
	@if find $(MOJO_PATHS) -name '*.mojo' -print -quit 2>/dev/null | grep -q .; then \
		find $(MOJO_PATHS) -name '*.mojo' -print0 | xargs -0 pixi run mojo format; \
	else \
		echo "No .mojo files found, skipping mojo format."; \
	fi

format-c:
	@files=$$(find llmm tests docs -name '*.c' -o -name '*.h' 2>/dev/null); \
	if [ -n "$$files" ]; then \
		clang-format -i $$files; \
	else \
		echo "No .c/.h files found, skipping c format."; \
	fi

format-cuda:
	@files=$$(find llmm tests docs -name '*.cu' -o -name '*.cuh' 2>/dev/null); \
	if [ -n "$$files" ]; then \
		clang-format -i $$files; \
	else \
		echo "No .cu/.cuh files found, skipping cuda format."; \
	fi

format-latex:
	@if ! command -v latexindent >/dev/null 2>&1; then \
		echo "latexindent not installed, skipping latex format."; \
	elif [ -n "$(LATEX_SOURCES)" ]; then \
		tmpdir=$$(mktemp -d); \
		trap "rm -rf $$tmpdir" EXIT; \
		for f in $(LATEX_SOURCES); do \
			latexindent -s -l docs/latexindent.yaml -g "$$tmpdir/indent.log" \
				-o "$$tmpdir/out.tex" "$$f"; \
			if ! diff -q "$$f" "$$tmpdir/out.tex" >/dev/null; then \
				cp "$$tmpdir/out.tex" "$$f"; \
				echo "formatted: $$f"; \
			fi; \
		done; \
	else \
		echo "No .tex files found, skipping latex format."; \
	fi

typecheck:
	@pixi run pyrefly check $(PYTHON_PATHS)

test:
	@if command -v nvidia-smi >/dev/null 2>&1 && timeout 2 nvidia-smi >/dev/null 2>&1; then \
		echo "GPU detected. Running GPU tests..."; \
		$(MAKE) test-mojo test-python-cuda; \
	else \
		echo "GPU not detected (or unresponsive). Running CPU tests..."; \
		$(MAKE) test-mojo test-python; \
	fi

test-cpu: test-mojo test-python

test-cuda: test-mojo test-python-cuda

test-mojo:
	@if ls tests/test_*.mojo >/dev/null 2>&1; then \
		fail=0; \
		for f in tests/test_*.mojo; do \
			echo "==> $$f"; \
			pixi run mojo run -I . "$$f" || fail=1; \
		done; \
		exit $$fail; \
	else \
		echo "No mojo tests found, skipping."; \
	fi

# Sequential: parallel (-n 6 --dist loadfile) measured only ~10% faster at
# best and slower under any other load — MAX compiles are already
# multi-threaded, so workers oversubscribe the cores. Measurements in
# tests/conftest.py. Compiled models persist in tests/.mef_cache (see
# tests/_max_bridge.py): warm runs take seconds, and only a kernel-source
# change pays compiles again.
test-python: build-mojo
	pixi run pytest tests/ -v -n auto

test-python-cuda: build-mojo
	MAX_USE_ACCELERATOR=1 pixi run -e cuda pytest tests/ -v

test-fixtures:
	pixi run python -m tests.reference dump

# Run verification of activations, losses, and gradients against gpt2_124M_debug_state.bin.
# We run them as separate processes because initializing both CPU and GPU standard
# DeviceContexts in the same process leads to a MAX multi-context conflict/crash.
verify: verify-cpu verify-gpu

verify-cpu:
	pixi run mojo test_gpt2.mojo cpu

verify-gpu:
	pixi run -e cuda mojo test_gpt2.mojo gpu

docs:
	latexmk -pdf -quiet -cd docs/backprop.tex

docs-clean:
	latexmk -c -quiet -cd docs/backprop.tex

clean:
	rm -rf .ruff_cache .pyrefly_cache tests/fixtures tests/.mef_cache
	rm -f $(TRAIN_BIN)
	rm -f $(PROFILE_BIN) $(PROFILE_TRACE_BIN) $(PROFILE_BIN).ncu.csv \
		$(PROFILE_BIN).*.perfetto-trace.json \
		$(PROFILE_BIN).*.nsys-rep $(PROFILE_BIN).*.sqlite \
		build/profile_llmc.ncu.csv build/profile_llmc.nsys-rep build/profile_llmc.sqlite
	rm -rf build/llmc
	@# figures/ holds dated, hardware-stamped benchmark PNGs — kept on purpose.
