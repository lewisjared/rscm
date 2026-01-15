# workspace-structure Specification

## Purpose

This specification defines the canonical directory structure and module organisation for the RSCM workspace. It establishes:

- The Rust workspace layout with all crates under `crates/`
- Python namespace patterns that expose model-specific components through dedicated subpackages
- PyO3 submodule registration for component crates
- Build configuration for maturin to work with the workspace structure
- Internal module organisation patterns for spatial types and domain-specific components

## Requirements

### Requirement: Workspace Directory Layout

The workspace SHALL organise all Rust crates under a `crates/` subdirectory with the root `Cargo.toml` containing only workspace configuration.

#### Scenario: Standard workspace structure

- **WHEN** examining the repository root
- **THEN** the structure MUST include:
  - `crates/rscm/` - PyO3 extension module
  - `crates/rscm-core/` - Core traits and abstractions
  - `crates/rscm-components/` - Shared component implementations
  - `crates/rscm-macros/` - Procedural macros (from dx branch)
  - `crates/rscm-two-layer/` - Two-layer model component
  - `crates/rscm-magicc/` - MAGICC component implementations

#### Scenario: Root Cargo.toml is workspace-only

- **WHEN** reading the root `Cargo.toml`
- **THEN** it MUST contain only `[workspace]` configuration
- **AND** it MUST NOT contain a `[package]` section

### Requirement: Python Namespace Pattern

The Python package SHALL expose model-specific components through dedicated namespace subpackages rather than the root `rscm` namespace.

#### Scenario: Core exports from root package

- **WHEN** importing from `rscm`
- **THEN** only core framework types are available:
  - `Model`, `ModelBuilder`
  - `Timeseries`, `TimeseriesCollection`
  - `InterpolationStrategy`, `VariableType`

#### Scenario: Two-layer model namespace

- **WHEN** importing `TwoLayerComponentBuilder`
- **THEN** it MUST be imported via `from rscm.two_layer import TwoLayerComponentBuilder`
- **AND** it MUST NOT be available directly from `rscm`

#### Scenario: MAGICC namespace (scaffold)

- **WHEN** importing from `rscm.magicc`
- **THEN** the subpackage MUST exist
- **AND** it MAY be empty until components are implemented

### Requirement: Component Crate PyO3 Submodules

Each component crate that provides Python bindings SHALL expose a `#[pymodule]` that gets registered as a submodule of the main `_lib` extension.

#### Scenario: Two-layer crate provides submodule

- **WHEN** the `rscm-two-layer` crate is compiled
- **THEN** it MUST export a `two_layer` PyO3 module
- **AND** this module MUST be registered under `rscm._lib.two_layer`

#### Scenario: Submodule path registration

- **WHEN** the `_lib` extension is loaded
- **THEN** submodules MUST be registered in `sys.modules` with their full path
- **AND** `rscm._lib.two_layer` MUST be accessible

### Requirement: Build Configuration

The maturin build configuration SHALL point to the correct crate location within the `crates/` subdirectory.

#### Scenario: pyproject.toml manifest-path

- **WHEN** reading `pyproject.toml`
- **THEN** `[tool.maturin].manifest-path` MUST equal `"crates/rscm/Cargo.toml"`

#### Scenario: Workspace builds correctly

- **WHEN** running `cargo build --workspace`
- **THEN** all crates in `crates/` MUST compile successfully
- **AND** the rscm crate MUST link against all component crates

### Requirement: Core Submodules for Spatial and State Types

The `rscm._lib.core` module SHALL provide submodules for spatial grid types and state access types.

#### Scenario: Spatial submodule

- **WHEN** importing from `rscm._lib.core.spatial`
- **THEN** the following types MUST be available:
  - `ScalarRegion`, `ScalarGrid`
  - `FourBoxRegion`, `FourBoxGrid`
  - `HemisphericRegion`, `HemisphericGrid`

#### Scenario: State submodule

- **WHEN** importing from `rscm._lib.core.state`
- **THEN** the following types MUST be available:
  - `FourBoxSlice`, `HemisphericSlice`
  - `TimeseriesWindow`, `FourBoxTimeseriesWindow`, `HemisphericTimeseriesWindow`

#### Scenario: Submodules registered in sys.modules

- **WHEN** the `_lib` extension is loaded
- **THEN** `rscm._lib.core.spatial` MUST be in `sys.modules`
- **AND** `rscm._lib.core.state` MUST be in `sys.modules`

### Requirement: Spatial Module Organisation

The `rscm-core` crate SHALL organise spatial grid types into a `spatial/` subdirectory with separate files for each grid implementation.

#### Scenario: Spatial subdirectory structure

- **WHEN** examining `crates/rscm-core/src/spatial/`
- **THEN** it MUST include:
  - `mod.rs` - `SpatialGrid` trait definition and module documentation
  - `scalar.rs` - `ScalarRegion` and `ScalarGrid` implementations
  - `four_box.rs` - `FourBoxRegion` and `FourBoxGrid` implementations
  - `hemispheric.rs` - `HemisphericRegion` and `HemisphericGrid` implementations

#### Scenario: Public re-exports maintained

- **WHEN** using `use rscm_core::spatial::*`
- **THEN** all grid types, region enums, and the `SpatialGrid` trait MUST be accessible
- **AND** existing code using these types MUST continue to compile without changes

#### Scenario: Python bindings mirror Rust structure

- **WHEN** examining `crates/rscm-core/src/python/spatial/`
- **THEN** it MUST include separate files for each grid type's PyO3 wrappers
- **AND** the structure MUST mirror the Rust `spatial/` module

### Requirement: MAGICC Crate Domain Structure

The `rscm-magicc` crate SHALL organise components into domain-based subdirectories reflecting MAGICC's physical process categories.

#### Scenario: Domain subdirectories exist

- **WHEN** examining `crates/rscm-magicc/src/`
- **THEN** it MUST include subdirectories:
  - `chemistry/` - Atmospheric chemistry components
  - `forcing/` - Radiative forcing components
  - `climate/` - Climate response components
  - `carbon/` - Carbon cycle components

#### Scenario: Domain modules are declared

- **WHEN** reading `crates/rscm-magicc/src/lib.rs`
- **THEN** it MUST declare modules for each domain
- **AND** each domain module MAY be empty initially
