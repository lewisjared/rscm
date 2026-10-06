# AGENTS.md

RSCM (Rust Simple Climate Model) is a framework for reduced-complexity climate models:
a Rust workspace in `crates/` with Python bindings via PyO3/maturin (`crates/rscm`, wrapped by `python/rscm/`).
It is a testbed for modularising MAGICC and reimplementing it in Rust.

## Key documentation

- [docs/key_concepts.md](docs/key_concepts.md): Components, Models, Timeseries, and how they fit together.
- [docs/modules/README.md](docs/modules/README.md): MAGICC module specs, dependency graph and status.
  Consult it before changing physics.
- `tests/regression/`: parity tests against reference implementations.

## Commands

```bash
make virtual-environment                      # venv, pre-commit hooks, build extension
make build-dev                                # Rebuild the extension after .rs changes (needed before Python tests)
make all                                      # format, build, lint, test
make test                                     # All tests
cargo test --workspace --exclude rscm         # Rust tests only (rscm is the PyO3 crate)
uv run pytest -m "not slow"                   # Python tests, skipping slow ones
make format                                   # Run before committing
make lint                                     # ruff, mypy, cargo fmt, clippy
make validate-pyi                             # Check .pyi stubs match the compiled module
```

## PRs and commits

- Conventional commit messages (`feat:`, `fix:`, `chore:`).
- Add a [changelog fragment](./changelog/README.md) in `changelog/` for user-facing changes.
