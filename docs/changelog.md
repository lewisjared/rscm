## rscm 0.5.0 (2026-01-26)

### ⚠️ Breaking Changes ⚠️

- Replaced ambiguous `current()` and `next()` methods on `TimeseriesWindow` with explicit `at_start()` and `at_end()` accessors that clearly communicate timestep semantics. ([#67](https://github.com/lewisjared/rscm/pulls/67))

### Features

- Added VariableSchema for declaring model variables and aggregation relationships. Supports Sum, Mean, and Weighted aggregation operations with hierarchical aggregates, automatic validation, and cycle detection. Virtual aggregator components are inserted into the model graph during build. ([#66](https://github.com/lewisjared/rscm/pulls/66))
- Added schema-driven grid auto-aggregation. Components can now read variables at coarser resolutions than the schema declares (e.g., reading a Scalar when the schema stores FourBox data), and write at finer resolutions. The model automatically aggregates using configurable area weights via `ModelBuilder.with_grid_weights()`. ([#68](https://github.com/lewisjared/rscm/pulls/68))
- Added model calibration and uncertainty quantification framework (`rscm.calibrate`). Supports MCMC sampling via affine-invariant ensemble sampler (Goodman & Weare 2010), point estimation, convergence diagnostics (R-hat, ESS), and chain persistence. Includes Python bindings with tqdm progress bars and pandas DataFrame integration. ([#69](https://github.com/lewisjared/rscm/pulls/69))

### Improvements

- Improved the `/proposal` OpenSpec command with a two-phase workflow that separates investigation from specification, requiring user approval between phases. ([#62](https://github.com/lewisjared/rscm/pulls/62))
- Split CI docs into parallel jobs with path filtering for faster builds. ([#63](https://github.com/lewisjared/rscm/pulls/63))

### Bug Fixes

- Fixed DOT graph output to properly escape quotes in component labels, enabling compatibility with pydot and other DOT parsers. ([#66](https://github.com/lewisjared/rscm/pulls/66))

### Improved Documentation

- Added Variable Schemas tutorial covering aggregation operations (Sum, Mean, Weighted), hierarchical aggregates, NaN handling, and schema validation. ([#66](https://github.com/lewisjared/rscm/pulls/66))

### Trivial/Internal Changes

- [#64](https://github.com/lewisjared/rscm/pulls/64)


## rscm 0.4.1 (2026-01-17)

No significant changes.

## rscm 0.4.0 (2026-01-17)

### ⚠️ Breaking Changes ⚠️

- Renamed components to remove the trailing "Component" suffix (e.g., `TwoLayerComponent` is now `TwoLayer`, `CarbonCycleComponent` is now `CarbonCycle`). ([#61](https://github.com/lewisjared/rscm/pulls/61))

### Improvements

- Added mermaid diagram rendering support in documentation. ([#61](https://github.com/lewisjared/rscm/pulls/61))

### Improved Documentation

- Expanded documentation with new tutorials for spatial grids, debugging, and scenario pipelines. Added cross-references from documentation pages to the Python API. ([#61](https://github.com/lewisjared/rscm/pulls/61))

## rscm 0.3.0 (2026-01-16)

### ⚠️ Breaking Changes ⚠️

- Replaced `RequirementType::InputAndOutput` with `RequirementType::State` for variables that read previous values and write new values. ([#53](https://github.com/lewisjared/rscm/pulls/53))
- Reorganised workspace structure with all Rust crates now under `crates/` directory and improved Python module organisation with dedicated submodules for spatial and state types.
  The public API of the Python package was reorganised. ([#55](https://github.com/lewisjared/rscm/pulls/55))

### Features

- Added grid-based timeseries infrastructure for spatially-resolved climate modeling.

  This release introduces support for regional climate data through a flexible grid system:

  - **Grid Types**: Implemented `ScalarGrid` (single global value), `FourBoxGrid` (MAGICC standard with Northern/Southern Ocean/Land regions), and `HemisphericGrid` (Northern/Southern hemispheres)
  - **GridTimeseries**: Generic timeseries type `GridTimeseries<T, G>` with compile-time grid knowledge, supporting interpolation, aggregation, and transformations between grid types
  - **Type-Safe Access**: Region enums (`FourBoxRegion`, `HemisphericRegion`, `ScalarRegion`) provide compile-time safety for regional data access
  - **State System Integration**: Added `StateValue` enum and grid-aware methods (`get_latest_value()`, `get_global()`, `get_region()`) to `InputState` for seamless scalar/grid handling
  - **Python Bindings**: Exposed all grid types to Python with PyO3 wrappers, including region constants and grid operations
  - **Example Component**: Included `FourBoxOceanHeatUptakeComponent` demonstrating scalar-to-grid disaggregation patterns

  The implementation maintains full backwards compatibility with existing scalar timeseries code. Components can continue using scalar values while new components can leverage regional resolution where appropriate. Grid transformations are explicit and type-checked to prevent silent data loss.

  ([#48](https://github.com/lewisjared/rscm/pulls/48))
- Added `ComponentIO` derive macro for type-safe component development with struct-level `#[inputs(...)]`, `#[outputs(...)]`, and `#[states(...)]` attributes.

  The macro generates typed input/output structs with compile-time field validation, eliminating stringly-typed component APIs. Components can now access inputs through `TimeseriesWindow` for zero-cost timeseries access and return typed output structs. ([#53](https://github.com/lewisjared/rscm/pulls/53))
- Components can now return spatially-resolved grid outputs (FourBox and Hemispheric) natively, enabling regional values to flow between components without aggregation to scalars.
  The `OutputState` type now supports `StateValue` variants for grids, and the ComponentIO macro automatically wraps grid outputs in the appropriate variant. ([#56](https://github.com/lewisjared/rscm/pulls/56))
- Added automatic component documentation generation.
   The new `rscm-doc-gen` tool extracts metadata from `ComponentIO` macro attributes and generates documentation pages for each component, including inputs, outputs, states, and mathematical formulations. ([#59](https://github.com/lewisjared/rscm/pulls/59))

### Improvements

- Added spatial grid support (`GridType::Scalar`, `GridType::FourBox`, `GridType::Hemispheric`) to component requirements with compile-time grid validation and automatic grid transformation in the model coupler. ([#53](https://github.com/lewisjared/rscm/pulls/53))
- Updated legacy components to use the new ComponentIO derive macro. ([#58](https://github.com/lewisjared/rscm/pulls/58))

### Trivial/Internal Changes

- [#58](https://github.com/lewisjared/rscm/pulls/58)

## rscm 0.2.3 (2026-01-14)

No significant changes.

## rscm 0.2.2 (2026-01-14)

No significant changes.

## rscm 0.2.1 (2026-01-14)

No significant changes.

## rscm 0.2.0 (2026-01-14)

### ⚠️ Breaking Changes ⚠️

- Refactored `InputState` to include references to `TimeSeries` instead of scalar values.
  This is requires a change to the `Component` interface. ([#17](https://github.com/lewisjared/rscm/pulls/17))

### Features

- Add Ocean Surface Partial Pressure (OSPP) component to the `rscm-components` crate. ([#10](https://github.com/lewisjared/rscm/pulls/10))
- Added automated release workflow for publishing packages to crates.io and PyPI with cross-platform wheel builds. ([#47](https://github.com/lewisjared/rscm/pulls/47))

### Bug Fixes

- Fixed CI failure on Python 3.13 by upgrading dependencies (pandas 2.2.2 -> 2.3.3 which includes Python 3.13 wheels). ([#39](https://github.com/lewisjared/rscm/pulls/39))

### Improved Documentation

- Add the basic framework for a `mkdocs`-based documentation site in the `docs/` directory. ([#18](https://github.com/lewisjared/rscm/pulls/18))
- Added MAGICC module documentation ([#46](https://github.com/lewisjared/rscm/pulls/46))

### Trivial/Internal Changes

- [#39](https://github.com/lewisjared/rscm/pulls/39)

## rscm 0.1.0 (2024-09-24)

### Improvements

- Add changelog management to the release process ([#9](https://github.com/lewisjared/rscm/pulls/9))
