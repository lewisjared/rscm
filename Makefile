.DEFAULT_GOAL := all

# using pip install cargo (via maturin via pip) doesn't get the tty handle
# so doesn't render color without some help
export CARGO_TERM_COLOR=$(shell (test -t 0 && echo "always") || echo "auto")

.PHONY: virtual-environment
virtual-environment:
	uv venv
	pre-commit install
	$(MAKE) build-dev


.PHONY: build-dev
build-dev:
	@rm -f python/rscm/*.so
	uv run maturin develop

.PHONY: build-prod
build-prod:
	@rm -f python/rscm/*.so
	uv run maturin develop --release

.PHONY: format
format:
	uv run ruff check --fix

	uv run ruff format
	cargo fmt

.PHONY: lint-python
lint-python:
	uv run ruff check
	uv run ruff format --check
	uv run mypy python/rscm

.PHONY: validate-pyi
validate-pyi: build-dev  ## Validate .pyi stubs match the actual module
	uv run python -m mypy.stubtest rscm._lib --allowlist stubtest-allowlist.txt

.PHONY: lint-rust
lint-rust:
	cargo fmt --version
	cargo fmt --all -- --check
	cargo clippy --version
	cargo clippy --tests

.PHONY: lint
lint: lint-python lint-rust

.PHONY: test-python
test-python: build-dev
	uv run pytest

.PHONY: test-rust
test-rust:
	cargo test --workspace

.PHONY: test
test: test-python test-rust

.PHONY: all
all: format build-dev lint test

.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -rf .cache
	rm -rf htmlcov
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -rf perf.data*
	rm -rf python/rscm/*.so


.PHONY: docs-metadata
docs-metadata: ## Generate component metadata JSON from Rust sources
	cargo run -p rscm-doc-gen -- \
		--crates crates/rscm-components,crates/rscm-two-layer \
		--output docs/component_metadata/

.PHONY: docs
docs: build-dev docs-metadata  ## build the docs
	uv run --group docs mkdocs build

.PHONY: docs-strict
docs-strict: build-dev docs-metadata ## build the docs strictly (e.g. raise an error on warnings, this most closely mirrors what we do in the CI)
	uv run --group docs mkdocs build --strict

.PHONY: docs-serve
docs-serve: build-dev docs-metadata ## serve the docs locally
	uv run --group docs mkdocs serve

.PHONY: sync-katex
sync-katex:  ## Sync katex-header.html to all crates
	cp assets/katex-header.html crates/rscm/katex-header.html
	cp assets/katex-header.html crates/rscm-core/katex-header.html
	cp assets/katex-header.html crates/rscm-components/katex-header.html
	cp assets/katex-header.html crates/rscm-two-layer/katex-header.html
	cp assets/katex-header.html crates/rscm-magicc/katex-header.html

.PHONY: docs-rust
docs-rust:  ## Build Rust documentation
	RUSTDOCFLAGS="--html-in-header ./assets/katex-header.html" cargo doc --no-deps --workspace
