# Project Context

## Purpose

RSCM (Rust Simple Climate Model) is a framework for building reduced-complexity climate models.
It serves as a testbed for a rust-based implementation of MAGICC (Model for the Assessment of Greenhouse Gas Induced Climate Change).
This rewrite should result in improved performance and maintainability.

The project combines a high-performance Rust core with Python bindings via PyO3/maturin,
enabling climate scientists to leverage Rust's speed while maintaining Python's ease of use for scientific workflows.

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

- **Rust ABI Stability:** Using pyo3 abi3-py311 for stable Python ABI
- **Minimum Python:** 3.11+
- **Minimum Rust:** 1.75+
- **License:** Apache-2.0
- **Computational Efficiency:** Climate models run many timesteps; performance is critical

## External Dependencies

- **GitHub:** Repository hosting and CI/CD (Actions)
- **Documentation:** Hosted at <https://lewisjared.github.io/rscm/>
- **crates.io:** [Rust Package distribution](https://crates.io/crates/rscm)
- **PyPI:** [Python Package distribution](https://pypi.org/project/rscm/)

## Design Principles

1. **Modularity:** Components should be independent, composable building blocks
2. **Explicitness:** Make coupling between components explicit (e.g., grid transformations must be defined)
3. **Type Safety:** Leverage Rust's type system to catch errors at compile time
4. **Performance:** Zero-cost abstractions where possible; use runtime flexibility only when necessary
5. **Reproducibility:** Models should serialize/deserialize completely for reproducible science
6. **Gradual Adoption:** New features should be backwards compatible; existing code continues to work

## Current State

**Implemented:**

- Core framework: Component trait, Model orchestration, ModelBuilder
- Timeseries with interpolation strategies (linear, previous, next)
- Spatial grid types: Scalar, FourBox, Hemispheric with typed slices
- ComponentIO derive macro for type-safe I/O declarations
- TimeseriesWindow for zero-cost temporal data access
- Components: CarbonCycle, CO2 ERF, FourBoxOceanHeatUptake, TwoLayer
- Python bindings for core types via PyO3
- Serialization support (JSON/TOML) with typetag for trait objects
- Workspace structure with crates/ subdirectory organisation

**In Progress:**

- Grid-based timeseries for spatial resolution (four-box)
- Python namespace pattern (rscm.two_layer, rscm.magicc)

## Planned Improvements

The following areas represent substantial work packages suitable for OpenSpec proposals.

### 1. MAGICC Carbon Cycle Components

Implement the carbon cycle components that form the foundation of MAGICC's carbon-climate feedback system.

**Scope:**

- Terrestrial carbon cycle (land biosphere uptake, temperature feedback)
- Ocean carbon cycle (surface-deep exchange, solubility pump, biological pump)
- Permafrost carbon release module
- Land-use change emissions handling

**Dependencies:** Requires grid-based timeseries (in progress)

**Success criteria:** Carbon cycle outputs match MAGICC7 reference within 1% for standard scenarios

### 2. MAGICC Atmospheric Chemistry Components

Implement atmospheric chemistry for non-CO2 greenhouse gases.

**Scope:**

- Methane (CH4) lifetime and concentration
- Nitrous oxide (N2O) chemistry
- Tropospheric ozone from precursors
- Halocarbon (CFC, HCFC, HFC) atmospheric lifetimes
- Stratospheric chemistry interactions

**Dependencies:** Core component framework (complete)

**Success criteria:** Concentration projections match MAGICC7 for SSP scenarios

### 3. MAGICC Radiative Forcing Components

Implement radiative forcing calculations for all forcing agents.

**Scope:**

- Well-mixed GHG forcing (CO2, CH4, N2O, halocarbons)
- Aerosol forcing (direct and indirect effects)
- Tropospheric ozone forcing
- Stratospheric ozone forcing
- Land-use albedo change
- Volcanic forcing
- Solar forcing

**Dependencies:** Atmospheric chemistry components

**Success criteria:** ERF outputs match MAGICC7 for AR6 forcing timeseries

### 4. MAGICC Climate Response Components

Implement the climate response system.

**Scope:**

- Energy balance model (ocean heat uptake)
- Multi-layer ocean temperature model
- Hemispheric temperature patterns
- Sea level rise (thermal expansion, ice sheets, glaciers)
- Climate sensitivity parameterisation

**Dependencies:** Radiative forcing components

**Success criteria:** GMST and sea level match MAGICC7 for calibrated parameter sets

### 5. Python API Enhancement

Improve the Python developer experience and scientific workflow integration.

**Scope:**

- Pandas integration for scenario I/O (read/write pandas dataframes)
- High-level convenience API for common workflows
- Improved error messages with actionable suggestions
- Batch model execution with parallel processing
- Progress reporting for long-running simulations
- Jupyter notebook integration (rich repr, plotting helpers)

**Dependencies:** Core Python bindings (complete)

**Success criteria:** Scientists can run RSCM models with same ergonomics as existing Python SCMs

### 6. Model Validation Framework

Establish infrastructure for validating RSCM against reference implementations.

**Scope:**

- Reference data management (MAGICC7 outputs, RCMIP benchmarks)
- Automated comparison test suite
- Tolerance specifications for numerical comparisons
- Regression testing infrastructure
- Validation report generation
- CI integration for validation on every PR

**Dependencies:** At least one complete component chain (forcing -> climate)

**Success criteria:** Automated validation catches regressions and quantifies MAGICC7 agreement

### 7. Performance Benchmarking Infrastructure

Establish benchmarking to guide optimisation and prevent regressions.

**Scope:**

- Criterion-based microbenchmarks for hot paths
- End-to-end scenario benchmarks
- Memory profiling integration
- Baseline measurements and tracking
- CI integration for benchmark regressions
- Comparison with MAGICC7 and other SCMs

**Dependencies:** Core framework (complete)

**Success criteria:** Performance baselines established; regressions caught in CI

### 8. Configuration and Parameterisation System

Enable model configuration management for reproducible science.

**Scope:**

- Named parameter sets (MAGICC7 defaults, AR6 calibrations)
- Configuration file format (TOML/YAML)
- Parameter validation with range checking
- Import from existing MAGICC .CFG files
- Configuration diff and merge tools
- Documentation of parameter meanings and sources

**Dependencies:** Components to configure

**Success criteria:** Scientists can share and reproduce model configurations
