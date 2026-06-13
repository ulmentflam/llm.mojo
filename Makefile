PYTHON_SOURCES := $(shell find . -name '*.py' -not -path './.pixi/*' -not -path './data/.*/*' -not -path '*/__pycache__/*' 2>/dev/null)
MOJO_SOURCES := $(shell find . -name '*.mojo' -not -path './.pixi/*' -not -path './data/.*/*' 2>/dev/null)
C_SOURCES := $(shell find . \( -name '*.c' -o -name '*.h' \) -not -path './.pixi/*' -not -path './data/.*/*' 2>/dev/null)
CUDA_SOURCES := $(shell find . \( -name '*.cu' -o -name '*.cuh' \) -not -path './.pixi/*' -not -path './data/.*/*' 2>/dev/null)
LATEX_SOURCES := $(shell find . -name '*.tex' -not -path './.pixi/*' -not -path './data/.*/*' 2>/dev/null)

.PHONY: help lint lint-python lint-mojo lint-c lint-cuda lint-latex \
        format format-python format-mojo format-c format-cuda format-latex \
        typecheck check clean build-mojo \
        test test-python test-mojo test-fixtures \
        docs docs-clean

.DEFAULT_GOAL := help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Quality gates:"
	@echo "  check         Run lint, typecheck, and build-mojo"
	@echo "  lint          Lint Python, Mojo, C, CUDA, and LaTeX sources"
	@echo "  lint-python   Lint Python sources with ruff"
	@echo "  lint-mojo     Lint Mojo sources with mojo format --check"
	@echo "  lint-c        Lint C sources with clang-format and clang-tidy"
	@echo "  lint-cuda     Lint CUDA sources with clang-format and clang-tidy"
	@echo "  lint-latex    Lint LaTeX sources with latexindent (check only)"
	@echo "  typecheck     Type-check Python sources with pyrefly"
	@echo "  build-mojo    Compile llmm.mojopkg into the test cache; surfaces Mojo warnings"
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
	@echo "  test-fixtures Regenerate tests/fixtures/*.npz from PyTorch reference"
	@echo ""
	@echo "Documents:"
	@echo "  docs          Build docs/backprop.pdf with latexmk"
	@echo "  docs-clean    Remove LaTeX build artifacts (keeps the PDF)"
	@echo ""
	@echo "Housekeeping:"
	@echo "  help          Show this help message"
	@echo "  clean         Remove cache directories"

check: lint typecheck build-mojo

# Builds llmm.mojopkg into the persistent cache the pytest suite consumes
# (tests/.mef_cache/<source-fingerprint>/llmm.mojopkg, via the bridge so
# the fingerprint logic lives in one place). Content-addressed: a no-op
# when sources are unchanged, so chaining it before test-python costs
# nothing on warm runs while keeping the two steps independently runnable.
build-mojo:
	@if [ -d llmm ]; then \
		pixi run python -m tests._max_bridge; \
	else \
		echo "No llmm package found, skipping mojo build."; \
	fi

lint: lint-python lint-mojo lint-c lint-cuda lint-latex

lint-python:
	@if [ -n "$(PYTHON_SOURCES)" ]; then \
		uvx ruff check $(PYTHON_SOURCES); \
		uvx ruff format --check $(PYTHON_SOURCES); \
	else \
		echo "No .py files found, skipping python lint."; \
	fi

lint-mojo:
	@if [ -n "$(MOJO_SOURCES)" ]; then \
		fail=0; \
		tmpdir=$$(mktemp -d); \
		trap "rm -rf $$tmpdir" EXIT; \
		i=0; \
		for f in $(MOJO_SOURCES); do \
			i=$$((i+1)); \
			tmp="$$tmpdir/file_$$i.mojo"; \
			cp "$$f" "$$tmp"; \
			pixi run mojo format -q "$$tmp"; \
			if ! diff -q "$$f" "$$tmp" >/dev/null; then \
				echo "needs formatting: $$f"; \
				fail=1; \
			fi; \
		done; \
		exit $$fail; \
	else \
		echo "No .mojo files found, skipping mojo lint."; \
	fi

lint-c:
	@if [ -n "$(C_SOURCES)" ]; then \
		clang-format --dry-run --Werror $(C_SOURCES); \
		clang-tidy $(C_SOURCES) --; \
	else \
		echo "No .c/.h files found, skipping c lint."; \
	fi

lint-cuda:
	@if [ -n "$(CUDA_SOURCES)" ]; then \
		clang-format --dry-run --Werror $(CUDA_SOURCES); \
		clang-tidy $(CUDA_SOURCES) -- -x cuda; \
	else \
		echo "No .cu/.cuh files found, skipping cuda lint."; \
	fi

# latexindent insists on writing logs to cwd, so both latex targets run it
# against a scratch copy and diff/copy back (same trick as lint-mojo).
lint-latex:
	@if [ -n "$(LATEX_SOURCES)" ]; then \
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
	@if [ -n "$(PYTHON_SOURCES)" ]; then \
		uvx ruff check --fix $(PYTHON_SOURCES); \
		uvx ruff format $(PYTHON_SOURCES); \
	else \
		echo "No .py files found, skipping python format."; \
	fi

format-mojo:
	@if [ -n "$(MOJO_SOURCES)" ]; then \
		pixi run mojo format $(MOJO_SOURCES); \
	else \
		echo "No .mojo files found, skipping mojo format."; \
	fi

format-c:
	@if [ -n "$(C_SOURCES)" ]; then \
		clang-format -i $(C_SOURCES); \
	else \
		echo "No .c/.h files found, skipping c format."; \
	fi

format-cuda:
	@if [ -n "$(CUDA_SOURCES)" ]; then \
		clang-format -i $(CUDA_SOURCES); \
	else \
		echo "No .cu/.cuh files found, skipping cuda format."; \
	fi

format-latex:
	@if [ -n "$(LATEX_SOURCES)" ]; then \
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
	@if [ -n "$(PYTHON_SOURCES)" ]; then \
		pixi run pyrefly check $(PYTHON_SOURCES); \
	else \
		echo "No .py files found, skipping python typecheck."; \
	fi

test: test-mojo test-python

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
	pixi run pytest tests/ -v

test-fixtures:
	pixi run python -m tests.reference dump

docs:
	latexmk -pdf -quiet -cd docs/backprop.tex

docs-clean:
	latexmk -c -quiet -cd docs/backprop.tex

clean:
	rm -rf .ruff_cache .pyrefly_cache tests/fixtures tests/.mef_cache
