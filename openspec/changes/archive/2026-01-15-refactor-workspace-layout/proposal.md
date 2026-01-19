# Change: Refactor workspace layout to crates/ subdirectory with namespace Python API

## Prerequisites

This change MUST be applied after the `dx` branch (improve-component-dx) is merged to main. The dx branch introduces `rscm-macros/` crate which will be moved to `crates/rscm-macros/` as part of this restructure.

## Why

The current flat workspace structure mixes the root PyO3 bindings crate with the workspace configuration and makes it difficult to add new model-specific crates. As MAGICC components are added, a clearer separation between core framework, model-specific implementations, and Python bindings is needed. Moving to a `crates/` subdirectory enables easier cross-crate refactoring now while allowing future extraction of rscm-magicc into its own repository.

## What Changes

**Workspace Structure:**

- Move all Rust crates into `crates/` subdirectory
- Root `Cargo.toml` becomes workspace-only (no `[package]` section)
- Current `src/` PyO3 bindings become `crates/rscm/`
- `rscm-core/` becomes `crates/rscm-core/`
- `rscm-components/` becomes `crates/rscm-components/`
- `rscm-macros/` becomes `crates/rscm-macros/` (from dx branch)
- Extract `two_layer` model into its own `crates/rscm-two-layer/` crate
- Create `crates/rscm-magicc/` scaffold with domain subdirectories

**Module Reorganisation (rscm-core):**

- Split `spatial.rs` (922 lines) into `spatial/` subdirectory:
  - `spatial/mod.rs` - `SpatialGrid` trait and module documentation
  - `spatial/scalar.rs` - `ScalarGrid`, `ScalarRegion`
  - `spatial/four_box.rs` - `FourBoxGrid`, `FourBoxRegion`
  - `spatial/hemispheric.rs` - `HemisphericGrid`, `HemisphericRegion`
- Split `python/spatial.rs` to mirror the Rust structure

**Python Core Module Reorganisation:**

- Split `rscm._lib.core` into logical submodules:
  - `rscm._lib.core` - Essential types (14 classes): Timeseries, TimeAxis, Model, ModelBuilder, Component types, TimeseriesCollection, TestComponentBuilder
  - `rscm._lib.core.spatial` - Spatial types (6 classes): ScalarRegion/Grid, FourBoxRegion/Grid, HemisphericRegion/Grid
  - `rscm._lib.core.state` - State types (5 classes): FourBoxSlice, HemisphericSlice, TimeseriesWindow variants

**Python Namespace Pattern:**

- `rscm` package exports only core data structures (Model, ModelBuilder, Timeseries, TimeseriesCollection)
- Model-specific components accessed via namespaces:
  - `rscm.two_layer.TwoLayerComponentBuilder`
  - `rscm.magicc.CH4ChemistryBuilder` (future)
- Each component crate has its own PyO3 python submodule registered by the main rscm crate

**Build Configuration:**

- `pyproject.toml` uses `manifest-path = "crates/rscm/Cargo.toml"` for maturin
- Workspace dependencies remain centralised in root `Cargo.toml`

## Impact

- **Affected specs:** workspace-structure (new capability)
- **Affected code:**
  - All `Cargo.toml` files (paths)
  - `pyproject.toml` (maturin manifest-path)
  - `src/` directory (move to `crates/rscm/`)
  - `rscm-core/`, `rscm-components/` (move to `crates/`)
  - `rscm-core/src/spatial.rs` (split into `spatial/` subdirectory)
  - `rscm-core/src/python/spatial.rs` (split to mirror Rust)
  - `python/rscm/__init__.py` (update exports)
  - `Makefile` (update paths if needed)
  - `.bumpversion.toml` (update file paths)
- **BREAKING:** Python import paths change for `TwoLayerComponentBuilder` (now under `rscm.two_layer`)
