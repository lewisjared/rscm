# Project Context

## Purpose

RSCM (Rust Simple Climate Model) is a framework for building reduced-complexity climate models.
It serves as a testbed for a rust-based implementation of MAGICC (Model for the Assessment of Greenhouse Gas Induced Climate Change).
This rewrite should result in improved performance and maintainability.

The project combines a high-performance Rust core with Python bindings via PyO3/maturin,
enabling climate scientists to leverage Rust's speed while maintaining Python's ease of use for scientific workflows.

## Tech Stack

### Core Languages

- **Rust 1.75+** (2021 edition) - Core library and performance-critical components
- **Python 3.11+** - User-facing API and scientific workflows

### Build & Packaging

- **maturin** - PyO3-based Python extension builder
- **cargo** - Rust package manager and build system
- **uv** - Fast Python package and environment manager

### Key Rust Dependencies

- `pyo3` (0.27.0) - Rust/Python FFI bindings with abi3-py38 stable ABI
- `serde` + `typetag` - Serialization with trait object support
- `ndarray` - N-dimensional arrays (NumPy-compatible)
- `ode_solvers` - ODE numerical integration
- `petgraph` - Graph data structures for dependency resolution
- `thiserror` - Error handling

### Key Python Dependencies

- `numpy` - Numerical computing
- `scmdata` (>=0.17.0) - Climate model data structures

### Development Tools

- `ruff` - Python linting and formatting
- `clippy` - Rust linting
- `pytest` - Python testing
- `rstest` + `approx` - Rust testing
- `towncrier` - Changelog management
- `bump-my-version` - Version management
- `mkdocs` + `mkdocs-material` - Documentation

## Project Conventions

### Code Style

**Rust:**

- Standard rustfmt formatting
- Clippy lints enabled (run with `--tests`)
- KaTeX math notation in rustdoc comments
- All public items documented

**Python:**

- ruff for linting and formatting (line-length: 88)
- numpy docstring convention
- Type hints encouraged

**General:**

- British English spelling throughout
- No emojis unless explicitly requested

### Architecture Patterns

**Workspace Structure:**

```
rscm/           # Root crate: PyO3 Python bindings
rscm-core/      # Core traits: Component, Model, Timeseries
rscm-components/# Concrete climate model components
python/rscm/    # Python package wrapping Rust extension (_lib)
```

**Component Trait Pattern:**

- Components declare inputs/outputs via `definitions()`
- Implement `solve(t_current, t_next, input_state) -> OutputState`
- Use `#[typetag::serde]` macro for serialization support

**Model Orchestration:**

- `ModelBuilder` constructs dependency graph between components
- Components solved in dependency order via BFS traversal
- State flows through `TimeseriesCollection`

**Serialization:**

- JSON/TOML via serde
- `#[typetag::serde(tag = "type")]` for trait object deserialization

### Testing Strategy

- Every function must have tests
- Tests must be accurate, reflect real usage, and reveal flaws
- Tests should be verbose for debugging purposes
- No "cheater" tests that just pass

**Commands:**

```bash
cargo test --workspace        # Rust tests
uv run pytest                 # Python tests (requires build-dev)
```

### Git Workflow

- **Main branch:** `main`
- **Commit style:** Conventional commits (feat:, fix:, chore:, docs:, etc.)
- **Changelog:** Fragments in `changelog/` directory (towncrier)
- **Releases:** Managed via bump-my-version with automatic changelog build

## Domain Context

**Climate Modelling Terms:**

- **SCM (Simple Climate Model):** Reduced-complexity models that approximate full Earth system models
- **ERF (Effective Radiative Forcing):** A measure of energy imbalance in the climate system
- **Carbon Cycle:** The biogeochemical cycle of carbon exchange between atmosphere, ocean, and land
- **MAGICC:** Model for the Assessment of Greenhouse Gas Induced Climate Change - a widely-used reduced-complexity climate model

**Key Abstractions:**

- **Timeseries:** Time-indexed data with interpolation strategies (linear, previous, next)
- **Component:** Building block representing a physical process or calculation
- **Model:** Orchestrator that connects components and manages state flow
- **Grid/Spatial Resolution:** Regional structure for spatially-resolved variables (e.g., four-box: Northern Ocean, Northern Land, Southern Ocean, Southern Land)

## Important Constraints

- **Rust ABI Stability:** Using pyo3 abi3-py38 for stable Python ABI
- **Minimum Python:** 3.10+
- **Minimum Rust:** 1.75+
- **License:** Apache-2.0
- **Computational Efficiency:** Climate models run many timesteps; performance is critical
- **Backwards Compatibility:** Changes must not break existing components or models

## External Dependencies

- **GitHub:** Repository hosting and CI/CD (Actions)
- **Documentation:** Hosted at <https://lewisjared.github.io/rscm/>
- **PyPI/crates.io:** Package distribution (planned)

## Common Development Tasks

### Building

```bash
# Setup environment (first time)
make virtual-environment

# Rebuild Rust extension after code changes
make build-dev

# Run all tests
make test

# Linting and formatting
make lint
make format
```

### Testing

```bash
# Run tests separately
cargo test --workspace        # Rust tests only
uv run pytest                 # Python tests only

# Run a single Rust test
cargo test test_name --workspace

# Run a single Python test
uv run pytest tests/test_file.py::test_name
```

### Documentation

```bash
# Build Rust documentation
cargo doc --no-deps --open

# Build Python documentation (if configured)
cd docs && mkdocs serve
```

## Design Principles

1. **Modularity:** Components should be independent, composable building blocks
2. **Explicitness:** Make coupling between components explicit (e.g., grid transformations must be defined)
3. **Type Safety:** Leverage Rust's type system to catch errors at compile time
4. **Performance:** Zero-cost abstractions where possible; use runtime flexibility only when necessary
5. **Reproducibility:** Models should serialize/deserialize completely for reproducible science
6. **Gradual Adoption:** New features should be backwards compatible; existing code continues to work

## Current State

**Implemented:**

- Component trait and basic model orchestration
- Scalar timeseries with interpolation
- Basic carbon cycle and CO2 ERF components
- Python bindings for core types
- Serialization support (JSON/TOML)

**In Progress:**

- Grid-based timeseries for spatial resolution (four-box)
- Additional MAGICC-equivalent components

**Planned:**

- Full MAGICC parity
- Performance benchmarking and optimization
- Enhanced Python API (integration with scmdata)
