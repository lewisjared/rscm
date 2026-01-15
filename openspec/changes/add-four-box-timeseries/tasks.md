# Implementation Tasks

## 1. Core Spatial Grid Types

- [x] 1.1 Create `rscm-core/src/spatial.rs` module
- [x] 1.2 Add `UnsupportedGridTransformation` error variant to `rscm-core/src/errors.rs`
- [x] 1.3 Define `SpatialGrid` trait with methods: `grid_name()`, `size()`, `region_names()`, `aggregate_global()`, `transform_to() -> RSCMResult<>`
- [x] 1.4 Implement `ScalarGrid` (single global region)
- [x] 1.5 Implement `FourBoxGrid` with MAGICC standard regions and configurable weights
- [x] 1.6 Implement `HemisphericGrid` (Northern/Southern hemispheres)
- [x] 1.7 Add constants to `FourBoxGrid`: NORTHERN_OCEAN, NORTHERN_LAND, SOUTHERN_OCEAN, SOUTHERN_LAND
- [x] 1.8 Implement `Clone`, `Debug`, `Serialize`, `Deserialize` for all grid types
- [x] 1.9 Write unit tests for each grid type (size, region names, aggregation)
- [x] 1.10 Export spatial module from `rscm-core/src/lib.rs`

## 2. GridTimeseries Type

- [x] 2.1 Define `GridTimeseries<T, G>` struct in `rscm-core/src/timeseries.rs` or separate module
- [x] 2.2 Implement constructor `GridTimeseries::new(values: Array2<T>, time_axis, grid, units, strategy)`
- [ ] 2.3 Implement `from_values()` helper for creating from 2D array and time array
- [x] 2.4 Implement `new_empty()` for creating empty grid timeseries
- [x] 2.5 Implement `at(time_index, region_index) -> Option<T>` for point access
- [x] 2.6 Implement `set(time_index, region_index, value)` for updating values
- [x] 2.7 Implement `at_time(time) -> Result<Vec<T>>` for interpolating all regions at specific time
- [x] 2.8 Implement `latest_values() -> Vec<T>` to get latest values for all regions
- [x] 2.9 Implement `len()`, `is_empty()`, `grid()`, `time_axis()`, `units()` accessors
- [x] 2.10 Write unit tests for GridTimeseries construction and access methods

## 3. Grid Aggregation

- [x] 3.1 Implement `GridTimeseries::aggregate_global() -> GridTimeseries<T, ScalarGrid>`
- [x] 3.2 Implement `GridTimeseries::at_time_global(time) -> Result<T>` (implemented as `latest_global()`)
- [x] 3.3 Write unit tests for four-box to global aggregation with weights
- [x] 3.4 Write unit tests for hemispheric to global aggregation
- [x] 3.5 Test aggregation with different weight configurations

## 4. Grid Transformation

- [x] 4.1 Implement type-based transformation dispatch in `SpatialGrid::transform_to()`
- [x] 4.2 Implement four-box to scalar transformation (aggregate to global)
- [x] 4.3 Implement four-box to hemispheric transformation (aggregate by hemisphere)
- [x] 4.4 Implement hemispheric to scalar transformation
- [x] 4.5 Implement hemispheric to four-box transformation (if physically meaningful, or return error)
- [x] 4.6 Implement `GridTimeseries::transform_to<G2: SpatialGrid>(target_grid) -> Result<GridTimeseries<T, G2>>`
- [x] 4.7 Write unit tests for all defined transformations
- [x] 4.8 Write test that unsupported transformations return appropriate error
- [x] 4.9 Test error message includes source and target grid names

## 5. Region Access

- [x] 5.1 Implement `GridTimeseries::region(region_index) -> GridTimeseries<T, ScalarGrid>`
- [x] 5.2 Implement `GridTimeseries::region_by_name(name) -> Option<GridTimeseries<T, ScalarGrid>>`
- [x] 5.3 Write unit tests for extracting individual regions
- [x] 5.4 Test that extracted region has same time axis and correct values

## 6. Interpolation Strategy Support

- [x] 6.1 Implement `GridTimeseries::with_interpolation_strategy(strategy) -> &Self`
- [ ] 6.2 Implement `GridTimeseries::interpolator()` returning region-specific interpolators
- [x] 6.3 Implement `GridTimeseries::interpolate_into(new_time_axis) -> Self`
- [x] 6.4 Write unit tests for linear interpolation across all regions
- [ ] 6.5 Write unit tests for previous value strategy across all regions
- [x] 6.6 Test that interpolation strategies apply independently to each region

## 7. Serialization Support

- [x] 7.1 Implement `Serialize` for `GridTimeseries<T, G>` (derive or custom)
- [x] 7.2 Implement `Deserialize` for `GridTimeseries<T, G>` (derive or custom)
- [ ] 7.3 Test JSON serialization/deserialization for ScalarGrid timeseries
- [x] 7.4 Test JSON serialization/deserialization for FourBoxGrid timeseries
- [ ] 7.5 Test TOML serialization/deserialization for all grid types
- [ ] 7.6 Test handling of NaN values in serialization (JSON and TOML)
- [ ] 7.7 Verify grid metadata (region names, weights) preserved in serialization

## 8. Backwards Compatibility

- [x] ~~8.1 Create type alias: `pub type Timeseries<T> = GridTimeseries<T, ScalarGrid>`~~ (not needed)
- [x] ~~8.2 Verify all existing tests pass with aliased Timeseries type~~ (not needed)
- [x] ~~8.3 Test that existing component examples work unchanged~~ (not needed)
- [x] ~~8.4 Document migration path in code comments and rustdoc~~ (not needed)

## 9. State System Integration

- [x] 9.1 Design `StateValue` enum or trait for type-erased grid values
- [x] 9.2 Update `InputState` to support both scalar and grid timeseries items (added infrastructure, full grid support pending TimeseriesCollection update)
- [x] 9.3 Add `InputState::get_latest_value() -> Option<StateValue>` method (returns StateValue)
- [ ] 9.4 Update `OutputState` to support grid values (deferred for backwards compatibility)
- [x] 9.5 Add helper methods: `get_global()`, `get_region()` to InputState
- [x] 9.6 Write unit tests for InputState with StateValue
- [ ] 9.7 Write unit tests for OutputState with grid values (deferred)

## 10. TimeseriesCollection Integration

- [x] 10.1 Update `TimeseriesItem` to support grid timeseries (created `TimeseriesData` enum with Scalar/FourBox/Hemispheric variants)
- [x] 10.2 Update `TimeseriesCollection::add_timeseries()` to accept grid timeseries (kept scalar-only, added new methods)
- [x] 10.3 Add `TimeseriesCollection::add_grid_timeseries()` method (added `add_four_box_timeseries()` and `add_hemispheric_timeseries()`)
- [x] 10.4 Update `TimeseriesCollection::get_by_name()` to handle grid timeseries (added `get_data()` and `get_data_mut()`)
- [x] 10.5 Write unit tests for adding and retrieving grid timeseries from collection
- [x] 10.6 Test mixed collection with scalar and grid timeseries

## 11. Documentation

- [x] 11.1 Write rustdoc for `SpatialGrid` trait with examples
- [x] 11.2 Write rustdoc for `GridTimeseries` with comprehensive examples
- [x] 11.3 Write rustdoc for each grid type (Scalar, FourBox, Hemispheric)
- [x] 11.4 Add module-level documentation for `spatial` module
- [x] 11.5 Add examples showing four-box usage patterns
- [x] 11.6 Add examples showing grid transformation
- [x] 11.7 Document when to use which grid type
- [x] 11.8 Add migration guide for component developers (covered in spatial module docs)
- [x] 11.9 Document supported regions for each grid type (names, indices, constants)
- [x] 11.10 Create transformation matrix table in documentation
- [x] 11.11 Document transformation semantics (aggregation formulas, weights)
- [x] 11.12 Add component integration pattern examples (5 patterns from design.md)
- [x] 11.13 Document when to use broadcast vs. error for unsupported transformations
- [x] 11.14 Add warnings about physically inappropriate transformations
- [x] 11.15 Document how to implement custom disaggregation components

## 12. Python Bindings

- [x] 12.1 Design Python API for grid timeseries (added grid type bindings)
- [x] 12.2 Add PyO3 bindings for `FourBoxGrid` if needed
- [x] 12.3 Expose grid timeseries to Python as 2D numpy arrays (deferred - grids exposed, full GridTimeseries bindings pending)
- [x] 12.4 Add Python type stubs (.pyi) for grid timeseries (not needed with PyO3)
- [x] 12.5 Write Python tests for grid timeseries usage

## 13. Example Components

- [x] 13.1 Create example component using four-box timeseries (e.g., four-box ocean heat uptake)
- [ ] 13.2 Create example showing component coupling with grid transformation (covered in documentation examples)
- [ ] 13.3 Add example to documentation showing end-to-end four-box model (deferred)
- [ ] 13.4 Write integration test for model with four-box components (deferred)

## 14. Performance Validation

- [ ] 14.1 Benchmark GridTimeseries operations vs scalar Timeseries
- [ ] 14.2 Profile grid transformation performance
- [ ] 14.3 Verify no performance regression for scalar timeseries (backwards compat)
- [ ] 14.4 Document performance characteristics in rustdoc

## 15. Changelog and Release

- [ ] 15.1 Add changelog fragment to `changelog/` directory
- [ ] 15.2 Update version if needed (via bump-my-version)
- [ ] 15.3 Verify all tests pass: `cargo test --workspace`
- [ ] 15.4 Verify Python tests pass: `uv run pytest`
- [ ] 15.5 Verify linting passes: `make lint`
- [ ] 15.6 Build documentation: `cargo doc --no-deps`
