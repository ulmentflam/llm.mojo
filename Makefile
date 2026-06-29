# Source roots only — never `find .` (crawls .pixi and hangs on iCloud).
MOJO_PATHS := train_gpt2.mojo profile_gpt2.mojo llmm tests
PYTHON_PATHS := train_gpt2.py profile_gpt2.py tests data
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
PROFILE_RUNNER := scripts/run_profile_gpt2.sh
PROFILE_SCRIPT := profile_gpt2.py
PROFILE_TARGET ?= gpu
# Suffix the Perfetto trace with the target (…gpu./…cpu.) so the GPU and CPU
# runs write distinct files instead of clobbering each other. Recursive `=`
# (not `:=`) so it re-expands with any per-target PROFILE_TARGET override.
PROFILE_TRACE = $(PROFILE_BIN).$(PROFILE_TARGET).perfetto-trace.json
PROFILE_NSYS_REP := $(PROFILE_BIN).nsys-rep

SHELL := /bin/bash

.PHONY: help install update lint lint-python lint-mojo lint-c lint-cuda lint-latex \
        format format-python format-mojo format-c format-cuda format-latex \
        typecheck check clean build         build-mojo build-train train train-cpu \
        build-profile profile profile-trace profile-cpu profile-ncu profile-nsys \
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
	@echo "  profile       Run one step and emit a Perfetto trace (alias: profile-trace)"
	@echo "  profile-trace Write build/profile_gpt2.<target>.perfetto-trace.json (ui.perfetto.dev)"
	@echo "  profile-cpu   Run the profile on CPU and emit a Perfetto trace"
	@echo "  profile-ncu   Profile GPU kernels with ncu + print profile_gpt2.py table"
	@echo "  profile-nsys  Capture an nsys timeline to build/profile_gpt2.nsys-rep"
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

# Per-kernel GPU profile via NVIDIA Nsight Compute (ncu), printed as a table by
# profile_gpt2.py. Add `--full` for the heavy metric set, `--sudo` if the
# GPU performance counters need elevated access (DRAM/tensor metrics otherwise
# show as n/a). The raw CSV is saved alongside the binary.
profile-ncu: $(PROFILE_BIN) $(PROFILE_SCRIPT)
	pixi run -e cuda python $(PROFILE_SCRIPT) \
		--exe $(PROFILE_BIN) --target $(PROFILE_TARGET) \
		--output $(PROFILE_BIN).ncu.csv

# Timeline capture via NVIDIA Nsight Systems. Open the .nsys-rep in the Nsight
# Systems UI (File > Open).
profile-nsys: $(PROFILE_BIN)
	pixi run -e cuda nsys profile --force-overwrite true \
		-o $(PROFILE_BIN) $(PROFILE_BIN) $(PROFILE_TARGET)
	@echo "nsys report: $(PROFILE_NSYS_REP)"

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

lint: lint-python lint-mojo lint-c lint-cuda lint-latex typecheck

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

format: format-python format-mojo format-c format-cuda format-latex

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
	rm -f $(PROFILE_BIN) $(PROFILE_NSYS_REP) \
		$(PROFILE_BIN).ncu.csv $(PROFILE_BIN).sqlite \
		$(PROFILE_BIN).*.perfetto-trace.json
