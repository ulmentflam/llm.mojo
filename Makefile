PYTHON_SOURCES := $(shell find . -name '*.py' -not -path './.pixi/*' -not -path './data/.*/*' -not -path '*/__pycache__/*' 2>/dev/null)
MOJO_SOURCES := $(shell find . -name '*.mojo' -not -path './.pixi/*' -not -path './data/.*/*' 2>/dev/null)
C_SOURCES := $(shell find . \( -name '*.c' -o -name '*.h' \) -not -path './.pixi/*' -not -path './data/.*/*' 2>/dev/null)
CUDA_SOURCES := $(shell find . \( -name '*.cu' -o -name '*.cuh' \) -not -path './.pixi/*' -not -path './data/.*/*' 2>/dev/null)

.PHONY: help lint lint-python lint-mojo lint-c lint-cuda \
        format format-python format-mojo format-c format-cuda \
        typecheck check clean

.DEFAULT_GOAL := help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  help          Show this help message"
	@echo "  check         Run lint and typecheck"
	@echo "  lint          Lint Python, Mojo, C, and CUDA sources"
	@echo "  lint-python   Lint Python sources with ruff"
	@echo "  lint-mojo     Lint Mojo sources with mojo format --check"
	@echo "  lint-c        Lint C sources with clang-format and clang-tidy"
	@echo "  lint-cuda     Lint CUDA sources with clang-format and clang-tidy"
	@echo "  format        Format Python, Mojo, C, and CUDA sources"
	@echo "  format-python Format Python sources with ruff"
	@echo "  format-mojo   Format Mojo sources with mojo format"
	@echo "  format-c      Format C sources with clang-format"
	@echo "  format-cuda   Format CUDA sources with clang-format"
	@echo "  typecheck     Type-check Python sources with pyrefly"
	@echo "  clean         Remove cache directories"

check: lint typecheck

lint: lint-python lint-mojo lint-c lint-cuda

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

format: format-python format-mojo format-c format-cuda

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

typecheck:
	@if [ -n "$(PYTHON_SOURCES)" ]; then \
		pixi run pyrefly check $(PYTHON_SOURCES); \
	else \
		echo "No .py files found, skipping python typecheck."; \
	fi

clean:
	rm -rf .ruff_cache .pyrefly_cache
