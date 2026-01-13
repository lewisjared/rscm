# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RSCM (Rust Simple Climate Model) is a framework for building reduced-complexity climate models.
It combines a Rust core library with Python bindings via PyO3/maturin.

The purpose of this is to be a testbed for modularise MAGICC and reimplement in rust.

## Build Commands

```bash
# Setup environment (first time)
make virtual-environment

# Rebuild Rust extension after code changes (required after any .rs changes)
make build-dev

# Run all tests
make test

# Run tests separately
cargo test --workspace        # Rust tests only
uv run pytest                 # Python tests only (requires build-dev first)

# Run a single Rust test
cargo test test_name --workspace

# Run a single Python test
uv run pytest tests/test_file.py::test_name

# Linting
make lint                     # Both Python and Rust
cargo clippy --tests          # Rust only
uv run ruff check             # Python only

# Format code
make format
```

## Architecture

### Workspace Structure

- **rscm/** - Root crate: PyO3 Python bindings
- **rscm-core/** - Core traits and abstractions (Component, Model, Timeseries)
- **rscm-components/** - Concrete climate model components (carbon cycle, CO2 ERF, etc.)
- **python/rscm/** - Python package that wraps the Rust extension (`_lib`)

### Key Concepts

**Component trait** (`rscm-core/src/component.rs`): The fundamental building block. Each component:

- Declares input/output requirements via `definitions()`
- Implements `solve(t_current, t_next, input_state) -> OutputState`
- Must use `#[typetag::serde]` macro for serialization support

**Model** (`rscm-core/src/model.rs`): Orchestrates multiple components:

- `ModelBuilder` constructs the dependency graph between components
- Components are solved in dependency order via BFS traversal
- State flows between components through `TimeseriesCollection`

**Timeseries** (`rscm-core/src/timeseries.rs`): Time-indexed data with interpolation strategies (linear, previous, next).

### Adding a New Component

1. Create struct with parameters in `rscm-components/src/components/`
2. Implement `Component` trait with `#[typetag::serde]`
3. Export from `rscm-components/src/components/mod.rs`
4. Add Python bindings in `rscm-components/src/python/mod.rs` if needed

### Serialization

Models and components serialize to JSON/TOML via serde. The `#[typetag::serde(tag = "type")]` pattern enables deserializing trait objects.

## Conventions

- British English spelling
- Conventional commits for commit messages
- Changelog fragments in `changelog/` directory (towncrier)
- Docstrings follow numpy convention (Python) and rustdoc with KaTeX for math (Rust)
