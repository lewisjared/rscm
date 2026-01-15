<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:

- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:

- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

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

Always run make format before committing.
Keep commit messages and plans concise

## Architecture

### Workspace Structure

```
rscm/
├── Cargo.toml                 # Workspace root (workspace-only, no package)
├── pyproject.toml             # Python build config (manifest-path = crates/rscm/Cargo.toml)
├── python/
│   └── rscm/                  # Python package wrapping Rust extension (_lib)
│       ├── __init__.py
│       ├── two_layer/         # Two-layer model namespace
│       ├── magicc/            # MAGICC components namespace (scaffold)
│       └── _lib/              # PyO3 extension module
├── crates/
│   ├── rscm/                  # PyO3 Python bindings (root extension module)
│   ├── rscm-core/             # Core traits and abstractions (Component, Model, Timeseries)
│   ├── rscm-components/       # Shared/generic climate model components
│   ├── rscm-macros/           # Procedural macros for component development
│   ├── rscm-two-layer/        # Two-layer model component (extracted from root)
│   └── rscm-magicc/           # MAGICC components scaffold (future implementations)
```

All Rust crates are in the `crates/` subdirectory. The root `Cargo.toml` is workspace-only.
The pyproject.toml uses `manifest-path = "crates/rscm/Cargo.toml"` to point to the PyO3 crate.

### Key Concepts

**Component trait** (`crates/rscm-core/src/component.rs`): The fundamental building block. Each component:

- Declares input/output requirements via `definitions()`
- Implements `solve(t_current, t_next, input_state) -> OutputState`
- Must use `#[typetag::serde]` macro for serialization support

**Model** (`crates/rscm-core/src/model.rs`): Orchestrates multiple components:

- `ModelBuilder` constructs the dependency graph between components
- Components are solved in dependency order via BFS traversal
- State flows between components through `TimeseriesCollection`

**Timeseries** (`crates/rscm-core/src/timeseries.rs`): Time-indexed data with interpolation strategies (linear, previous, next).

### Adding a New Component

**Recommended: Use the ComponentIO derive macro with struct-level attributes:**

```rust
use rscm_core::{ComponentIO, component::{InputState, OutputState, RequirementDefinition}};

#[derive(Debug, Serialize, Deserialize, ComponentIO)]
#[inputs(
    emissions { name = "Emissions|CO2", unit = "GtCO2" },
)]
#[outputs(
    concentration { name = "Concentration|CO2", unit = "ppm" },
)]
pub struct MyComponent {
    // Only actual parameters - no phantom fields needed
    pub sensitivity: f64,
}

#[typetag::serde]
impl Component for MyComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()  // Uses macro-generated definitions
    }

    fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        // Use typed inputs with TimeseriesWindow
        let inputs = MyComponentInputs::from_input_state(input_state);
        let emissions = inputs.emissions.current();  // Type-safe access

        // Return typed outputs
        let outputs = MyComponentOutputs {
            concentration: emissions * self.sensitivity,
        };
        Ok(outputs.into())
    }
}
```

The macro generates:

- `MyComponentInputs<'a>` with `TimeseriesWindow` fields for each input in `#[inputs(...)]`
- `MyComponentOutputs` with typed fields for each output in `#[outputs(...)]`
- `MyComponent::generated_definitions()` for the Component trait
- `From<MyComponentOutputs> for OutputState` conversion

**For state variables** (read previous, write new):

- Use `#[states(field { name = "...", unit = "..." })]`
- State variables appear in both `Inputs` and `Outputs` structs

**For grid-based variables:**

- Use `grid = "FourBox"` or `grid = "Hemispheric"` in the field declaration
- Access via `inputs.field.current(FourBoxRegion::NorthernOcean)` or `inputs.field.current_all()`
- Output types become `FourBoxSlice` or `HemisphericSlice` for grid outputs

**Manual implementation (legacy):**

1. Create struct with parameters in `rscm-components/src/components/`
2. Implement `Component` trait with `#[typetag::serde]`
3. Export from `rscm-components/src/components/mod.rs`
4. Add Python bindings in `rscm-components/src/python/mod.rs` if needed

### Serialization

Models and components serialize to JSON/TOML via serde. The `#[typetag::serde(tag = "type")]` pattern enables deserializing trait objects.

### Python Type Stubs (.pyi files)

Type stub files in `python/rscm/_lib/*.pyi` provide type hints for the PyO3 bindings. These files must be kept in sync with the Rust Python bindings.

**When to update .pyi files:**

- After adding new PyO3 classes or functions in `crates/*/src/python/`
- After changing method signatures on existing PyO3 classes
- After adding new enum variants or class attributes

**Note:** PyO3 automatically generates accurate Python bindings from Rust code, but type stubs must be manually maintained to provide IDE support and type checking.

## Conventions

- British English spelling
- Conventional commits for commit messages
- Changelog fragments in `changelog/` directory (towncrier)
- Docstrings follow numpy convention (Python) and rustdoc with KaTeX for math (Rust)

## Active Technologies

- Rust 1.75+ (2021 edition), Python 3.10+ + maturin (PyO3 bindings), GitHub Actions, cargo, uv (001-publish-packages)
