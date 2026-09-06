# AGENTS.md

## Project Overview

RSCM (Rust Simple Climate Model) is a framework for building reduced-complexity climate models.
It combines a Rust core library with Python bindings via PyO3/maturin. The project serves as a
testbed for modularising MAGICC and reimplementing it in Rust.

**Languages:** Rust 1.75+ (2021 edition), Python 3.11+

**Build tooling:** cargo, maturin (PyO3), uv (Python package manager)

## Key Documentation

- [docs/key_concepts.md](docs/key_concepts.md) - Architecture: Components, Models, Timeseries, and how they fit together
- [docs/modules/README.md](docs/modules/README.md) - MAGICC module specs, dependency graph, and implementation status (consult before changing physics)
- [.github/workflows/ci.yml](.github/workflows/ci.yml) - CI pipeline configuration
- [pyproject.toml](pyproject.toml) - Python config, linting rules, test markers, mypy and towncrier settings
- [Makefile](Makefile) - All build, test, lint, and docs targets

## Workspace Structure

```
rscm/
├── Cargo.toml                 # Workspace root (workspace-only, no package)
├── pyproject.toml             # Python build config (manifest-path = crates/rscm/Cargo.toml)
├── python/rscm/               # Python package wrapping Rust extension (_lib)
├── crates/
│   ├── rscm/                  # PyO3 Python bindings (root extension module)
│   ├── rscm-core/             # Core traits: Component, Model, Timeseries, VariableSchema
│   ├── rscm-components/       # Shared/generic climate model components
│   ├── rscm-macros/           # Procedural macros (ComponentIO derive)
│   ├── rscm-two-layer/        # Two-layer ocean model component
│   ├── rscm-magicc/           # MAGICC components (in progress)
│   ├── rscm-calibrate/        # Calibration support
│   └── rscm-doc-gen/          # Documentation metadata generator
├── tests/                     # Python tests (pytest)
│   └── regression/            # Regression/parity tests against reference implementations
├── changelog/                 # Towncrier changelog fragments
└── docs/                      # MkDocs documentation source
```

## Setup and Development

```bash
# First-time setup: creates venv, installs pre-commit hooks, builds extension
make virtual-environment

# Rebuild Rust extension after .rs changes (required before Python tests)
make build-dev

# Run everything: format, build, lint, test
make all
```

## Testing

```bash
make test                                       # All tests
cargo test --workspace --exclude rscm           # Rust tests only
uv run pytest                                   # Python tests only (needs build-dev)
cargo test test_name --workspace                # Single Rust test
uv run pytest tests/test_file.py::test_name     # Single Python test
uv run pytest -m "not slow"                     # Skip slow tests
```

## Linting and Formatting

```bash
make format          # Format all code (run before committing)
make lint            # Lint Python (ruff + mypy) and Rust (fmt + clippy)
make validate-pyi    # Validate .pyi stubs match compiled module
```

## CI Pipeline

CI runs on push to `main` and on all PRs (`.github/workflows/ci.yml`):

- **test**: Python + Rust tests across Python 3.11/3.12/3.13
- **lint**: ruff, mypy, stubtest, cargo fmt, clippy
- **docs**: MkDocs strict build + Rust API docs with KaTeX sync verification

## PR and Commit Guidelines

- Run `make format` before committing
- Commit messages use conventional commit format (`feat:`, `fix:`, `chore:`)
- Add a [changelog fragment](./changelog/README.md) in `changelog/` for user-facing changes
- All CI checks must pass
