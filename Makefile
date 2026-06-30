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
LLMM_SOURCES := $(shell find llmm -name '*.mojo' 2>/dev/null)
WORLD_SIZE ?= 1

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
PROFILE_NSYS_REP = $(PROFILE_BIN).$(PROFILE_TARGET).nsys-rep
PROFILE_NSYS_ENV := -e cuda
PROFILE_NSYS_FLAGS := --force-overwrite true --trace=cuda,osrt,nvtx --sample=process-tree

# Comparative benchmark against Karpathy's llm.c (git submodule). CPU and GPU are
# separated so the CPU path builds/runs with no CUDA toolchain (e.g. on macOS).
LLMC_DIR := third_party/llm.c
LLMC_CPU_BIN := $(LLMC_DIR)/train_gpt2
LLMC_GPU_BIN := $(LLMC_DIR)/train_gpt2cu
BENCH_SCRIPT := scripts/benchmark_train.py
HAVE_NVCC := $(shell command -v nvcc 2>/dev/null)

SHELL := /bin/bash

.PHONY: help install update lint lint-python lint-mojo lint-c lint-cuda lint-latex \
        format format-python format-mojo format-c format-cuda format-latex \
        typecheck check clean build         build-mojo build-train train train-cpu \
        build-profile build-profile-bf16 profile profile-trace profile-cpu profile-threads-cpu profile-ncu \
        profile-nsys profile-nsys-cpu \
        build-llmc build-llmc-cpu build-llmc-gpu benchmark benchmark-cpu benchmark-gpu \
        stage-llmc profile-llmc-ncu profile-llmc-nsys \
        test test-cpu test-cuda test-python test-mojo test-fixtures \
        verify verify-cpu verify-gpu \
        docs docs-clean

.DEFAULT_GOAL := help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install pixi dependencies (first-time setup)"
	@echo "  install-cuda  Install CUDA/GPU-enabled pixi dependencies"
	@echo "  update        Update pixi dependencies and refresh pixi.lock"
	@echo ""
	@echo "Quality gates:"
	@echo "  check         Run lint (incl. typecheck), build-mojo, build train_gpt2, and build-profile"
	@echo "  build         Compile train_gpt2.mojo to build/train_gpt2"
	@echo "  build-train   Alias for build"
	@echo "  train         Build and run build/train_gpt2 (sets MOJO_PYTHON_LIBRARY)"
	@echo "  train-cpu     Build and run build/train_gpt2 on CPU (LLMM_USE_CPU=1)"
	@echo ""
	@echo "Profiling:"
	@echo "  build-profile Compile profile_gpt2.mojo to build/profile_gpt2"
	@echo "  build-profile-bf16  Compile the bf16 (-D LLMM_BF16) harness, build/profile_gpt2_bf16"
	@echo "  profile       Run one step and emit a Perfetto trace (alias: profile-trace)"
	@echo "  profile-trace Write build/profile_gpt2.<target>.perfetto-trace.json (ui.perfetto.dev)"
	@echo "  profile-cpu   Run the profile on CPU and emit a Perfetto trace"
	@echo "  profile-threads-cpu  CPU per-thread Perfetto trace (all worker threads)"
	@echo "  profile-ncu   Profile our Mojo GPU kernels with ncu + profile_gpt2.py table"
	@echo "  profile-nsys  Capture a GPU nsys timeline (build/profile_gpt2.gpu.nsys-rep)"
	@echo "  profile-nsys-cpu  Capture a CPU nsys timeline showing all worker threads"
	@echo ""
	@echo "Benchmark (vs llm.c submodule):"
	@echo "  build-llmc    Build llm.c CPU (train_gpt2) + CUDA (train_gpt2cu, if nvcc)"
	@echo "  build-llmc-cpu  Build only the llm.c CPU reference (portable, macOS-ok)"
	@echo "  build-llmc-gpu  Build only the llm.c CUDA reference (needs nvcc)"
	@echo "  benchmark     Histogram of train-loop time: llm.mojo vs llm.c (CPU + GPU)"
	@echo "  benchmark-cpu Only the CPU comparison (no CUDA needed)"
	@echo "  benchmark-gpu Only the GPU comparison (NVIDIA)"
	@echo "  profile-llmc-ncu  Profile llm.c CUDA kernels with ncu (same table)"
	@echo "  profile-llmc-nsys Capture an nsys timeline of llm.c CUDA kernels"
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

install:
	pixi install

install-cuda:
	pixi install -e cuda

update:
	pixi update

check: lint build-mojo build build-profile

# Compiles the GPT-2 training binary. MOJO_PYTHON_LIBRARY must be set because
# DataLoader uses Python glob; pixi run supplies the Modular std/toolchain env.
build build-train: $(TRAIN_BIN)

$(TRAIN_BIN): $(TRAIN_MOJO_SRC) $(LLMM_SOURCES)
	@mkdir -p build
	pixi run mojo build -D WORLD_SIZE=$(WORLD_SIZE) $(MOJO_INCLUDES) -o $(TRAIN_BIN) $(TRAIN_MOJO_SRC)

train: $(TRAIN_BIN) $(TRAIN_RUNNER)
	@$(TRAIN_RUNNER)

train-cpu: $(TRAIN_BIN) $(TRAIN_RUNNER)
	@LLMM_USE_CPU=1 $(TRAIN_RUNNER)

# Compiles the single-step profiling harness. Depends on train_gpt2.mojo because
# it imports GPT2 from it (the llm.mojo analogue of profile_gpt2.cu #include'ing
# train_gpt2.cu).
build-profile: $(PROFILE_BIN)

$(PROFILE_BIN): $(PROFILE_MOJO_SRC) $(TRAIN_MOJO_SRC) $(LLMM_SOURCES)
	@mkdir -p build
	pixi run mojo build -D WORLD_SIZE=$(WORLD_SIZE) $(MOJO_INCLUDES) -o $(PROFILE_BIN) $(PROFILE_MOJO_SRC)

# bf16 mixed-precision build of the harness (params/acts/grads bf16, fp32 master
# weights + optimizer moments). GPU-only by policy; see profile_gpt2.mojo.
build-profile-bf16: $(PROFILE_BIN_BF16)

$(PROFILE_BIN_BF16): $(PROFILE_MOJO_SRC) $(TRAIN_MOJO_SRC) $(LLMM_SOURCES)
	@mkdir -p build
	pixi run mojo build -D WORLD_SIZE=$(WORLD_SIZE) -D LLMM_BF16=1 $(MOJO_INCLUDES) -o $(PROFILE_BIN_BF16) $(PROFILE_MOJO_SRC)

# Tracing build: same harness, with the per-thread kernel instrumentation
# compiled in via -D LLMM_TRACE=1.
$(PROFILE_TRACE_BIN): $(PROFILE_MOJO_SRC) $(TRAIN_MOJO_SRC) $(LLMM_SOURCES)
	@mkdir -p build
	pixi run mojo build -D WORLD_SIZE=$(WORLD_SIZE) -D LLMM_TRACE=1 $(MOJO_INCLUDES) -o $(PROFILE_TRACE_BIN) $(PROFILE_MOJO_SRC)

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

# Per-kernel GPU profile via NVIDIA Nsight Compute (ncu), printed as a table by
# profile_gpt2.py. Add `--full` for the heavy metric set, `--sudo` if the
# GPU performance counters need elevated access (DRAM/tensor metrics otherwise
# show as n/a). The raw CSV is saved alongside the binary.
profile-ncu: $(PROFILE_BIN) $(PROFILE_SCRIPT)
	pixi run -e cuda python $(PROFILE_SCRIPT) \
		--exe $(PROFILE_BIN) --target $(PROFILE_TARGET) \
		--output $(PROFILE_BIN).ncu.csv

# Timeline capture via NVIDIA Nsight Systems. The report's thread timeline shows
# every CPU worker thread (the ~20 sync_parallelize workers), which the in-process
# Perfetto tracer cannot see. Open the .nsys-rep in the Nsight Systems UI
# (File > Open). Default target is gpu; use profile-nsys-cpu for the CPU path.
profile-nsys: $(PROFILE_BIN)
	pixi run $(PROFILE_NSYS_ENV) nsys profile $(PROFILE_NSYS_FLAGS) \
		-o $(PROFILE_BIN).$(PROFILE_TARGET) $(PROFILE_BIN) $(PROFILE_TARGET)
	@echo "nsys report: $(PROFILE_NSYS_REP) (per-thread CPU timeline in Nsight Systems)"

# CPU thread timeline — runs in the default pixi env (no GPU required) and traces
# only OS-runtime + CPU samples, so the ~20 worker threads show on the timeline.
profile-nsys-cpu: PROFILE_TARGET := cpu
profile-nsys-cpu: PROFILE_NSYS_ENV :=
profile-nsys-cpu: PROFILE_NSYS_FLAGS := --force-overwrite true --trace=osrt --sample=process-tree
profile-nsys-cpu: $(PROFILE_BIN)
	pixi run $(PROFILE_NSYS_ENV) nsys profile $(PROFILE_NSYS_FLAGS) \
		-o $(PROFILE_BIN).$(PROFILE_TARGET) $(PROFILE_BIN) $(PROFILE_TARGET)
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
	pixi run python $(BENCH_SCRIPT) --device cpu

benchmark-gpu: $(PROFILE_BIN) $(PROFILE_BIN_BF16) build-llmc-gpu $(BENCH_SCRIPT)
	pixi run python $(BENCH_SCRIPT) --device gpu

# CPU always; GPU too iff an NVIDIA GPU is present.
benchmark: $(PROFILE_BIN) $(PROFILE_BIN_BF16) build-llmc $(BENCH_SCRIPT)
	pixi run python $(BENCH_SCRIPT) --device auto

# One short, deterministic train step of llm.c's CUDA build — the target for
# profiling Karpathy's GPU kernels. Same B/T (4/64) as our harness; data paths
# are relative to the staged build/llmc cwd.
LLMC_GPU_ARGS := -e gpt2_124M_bf16.bin \
	-i dev/data/tinyshakespeare/tiny_shakespeare_train.bin \
	-j dev/data/tinyshakespeare/tiny_shakespeare_val.bin \
	-b 4 -t 64 -x 1 -v 0 -s 0 -l 0

stage-llmc: $(BENCH_SCRIPT)
	pixi run python $(BENCH_SCRIPT) --stage-only --device gpu

# Profile Karpathy's CUDA kernels with ncu, through the SAME analysis table as
# our Mojo kernels (profile_gpt2.py categorizes both by kernel name). Compare
# against `make profile-ncu` (our kernels).
profile-llmc-ncu: build-llmc-gpu stage-llmc $(PROFILE_SCRIPT)
	pixi run -e cuda python $(PROFILE_SCRIPT) \
		--exe $(abspath $(LLMC_GPU_BIN)) \
		--exe-args "$(LLMC_GPU_ARGS)" \
		--cwd build/llmc \
		--output build/profile_llmc.ncu.csv

# nsys timeline of llm.c's CUDA training step.
profile-llmc-nsys: build-llmc-gpu stage-llmc
	cd build/llmc && pixi run -e cuda nsys profile --force-overwrite true \
		--trace=cuda,nvtx,osrt -o $(abspath build/profile_llmc) \
		$(abspath $(LLMC_GPU_BIN)) $(LLMC_GPU_ARGS)
	@echo "nsys report: build/profile_llmc.nsys-rep"

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
