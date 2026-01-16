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
