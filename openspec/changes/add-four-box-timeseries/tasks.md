# Implementation Tasks

## 1. Core Spatial Grid Types

- [ ] 1.1 Create `rscm-core/src/spatial.rs` module
- [ ] 1.2 Add `UnsupportedGridTransformation` error variant to `rscm-core/src/errors.rs`
- [ ] 1.3 Define `SpatialGrid` trait with methods: `grid_name()`, `size()`, `region_names()`, `aggregate_global()`, `transform_to() -> RSCMResult<>`
- [ ] 1.4 Implement `ScalarGrid` (single global region)
- [ ] 1.5 Implement `FourBoxGrid` with MAGICC standard regions and configurable weights
- [ ] 1.6 Implement `HemisphericGrid` (Northern/Southern hemispheres)
- [ ] 1.7 Add constants to `FourBoxGrid`: NORTHERN_OCEAN, NORTHERN_LAND, SOUTHERN_OCEAN, SOUTHERN_LAND
- [ ] 1.8 Implement `Clone`, `Debug`, `Serialize`, `Deserialize` for all grid types
- [ ] 1.9 Write unit tests for each grid type (size, region names, aggregation)
- [ ] 1.10 Export spatial module from `rscm-core/src/lib.rs`

## 2. GridTimeseries Type

- [ ] 2.1 Define `GridTimeseries<T, G>` struct in `rscm-core/src/timeseries.rs` or separate module
- [ ] 2.2 Implement constructor `GridTimeseries::new(values: Array2<T>, time_axis, grid, units, strategy)`
- [ ] 2.3 Implement `from_values()` helper for creating from 2D array and time array
- [ ] 2.4 Implement `new_empty()` for creating empty grid timeseries
- [ ] 2.5 Implement `at(time_index, region_index) -> Option<T>` for point access
- [ ] 2.6 Implement `set(time_index, region_index, value)` for updating values
- [ ] 2.7 Implement `at_time(time) -> Result<Vec<T>>` for interpolating all regions at specific time
- [ ] 2.8 Implement `latest_values() -> Vec<T>` to get latest values for all regions
- [ ] 2.9 Implement `len()`, `is_empty()`, `grid()`, `time_axis()`, `units()` accessors
- [ ] 2.10 Write unit tests for GridTimeseries construction and access methods

## 3. Grid Aggregation

- [ ] 3.1 Implement `GridTimeseries::aggregate_global() -> GridTimeseries<T, ScalarGrid>`
- [ ] 3.2 Implement `GridTimeseries::at_time_global(time) -> Result<T>`
- [ ] 3.3 Write unit tests for four-box to global aggregation with weights
- [ ] 3.4 Write unit tests for hemispheric to global aggregation
- [ ] 3.5 Test aggregation with different weight configurations

## 4. Grid Transformation

- [ ] 4.1 Implement type-based transformation dispatch in `SpatialGrid::transform_to()`
- [ ] 4.2 Implement four-box to scalar transformation (aggregate to global)
- [ ] 4.3 Implement four-box to hemispheric transformation (aggregate by hemisphere)
- [ ] 4.4 Implement hemispheric to scalar transformation
- [ ] 4.5 Implement hemispheric to four-box transformation (if physically meaningful, or return error)
- [ ] 4.6 Implement `GridTimeseries::transform_to<G2: SpatialGrid>(target_grid) -> Result<GridTimeseries<T, G2>>`
- [ ] 4.7 Write unit tests for all defined transformations
- [ ] 4.8 Write test that unsupported transformations return appropriate error
- [ ] 4.9 Test error message includes source and target grid names

## 5. Region Access

- [ ] 5.1 Implement `GridTimeseries::region(region_index) -> GridTimeseries<T, ScalarGrid>`
- [ ] 5.2 Implement `GridTimeseries::region_by_name(name) -> Option<GridTimeseries<T, ScalarGrid>>`
- [ ] 5.3 Write unit tests for extracting individual regions
- [ ] 5.4 Test that extracted region has same time axis and correct values

## 6. Interpolation Strategy Support

- [ ] 6.1 Implement `GridTimeseries::with_interpolation_strategy(strategy) -> &Self`
- [ ] 6.2 Implement `GridTimeseries::interpolator()` returning region-specific interpolators
- [ ] 6.3 Implement `GridTimeseries::interpolate_into(new_time_axis) -> Self`
- [ ] 6.4 Write unit tests for linear interpolation across all regions
- [ ] 6.5 Write unit tests for previous value strategy across all regions
- [ ] 6.6 Test that interpolation strategies apply independently to each region

## 7. Serialization Support

- [ ] 7.1 Implement `Serialize` for `GridTimeseries<T, G>` (derive or custom)
- [ ] 7.2 Implement `Deserialize` for `GridTimeseries<T, G>` (derive or custom)
- [ ] 7.3 Test JSON serialization/deserialization for ScalarGrid timeseries
- [ ] 7.4 Test JSON serialization/deserialization for FourBoxGrid timeseries
- [ ] 7.5 Test TOML serialization/deserialization for all grid types
- [ ] 7.6 Test handling of NaN values in serialization (JSON and TOML)
- [ ] 7.7 Verify grid metadata (region names, weights) preserved in serialization

## 8. Backwards Compatibility

- [ ] 8.1 Create type alias: `pub type Timeseries<T> = GridTimeseries<T, ScalarGrid>`
- [ ] 8.2 Verify all existing tests pass with aliased Timeseries type
- [ ] 8.3 Test that existing component examples work unchanged
- [ ] 8.4 Document migration path in code comments and rustdoc

## 9. State System Integration

- [ ] 9.1 Design `StateValue` enum or trait for type-erased grid values
- [ ] 9.2 Update `InputState` to support both scalar and grid timeseries items
- [ ] 9.3 Add `InputState::get_grid(name) -> Option<GridValues>` method
- [ ] 9.4 Update `OutputState` to support grid values (HashMap<String, StateValue>)
- [ ] 9.5 Add helper methods: `get_global()`, `get_region()` to InputState
- [ ] 9.6 Write unit tests for InputState with mixed scalar and grid data
- [ ] 9.7 Write unit tests for OutputState with grid values

## 10. TimeseriesCollection Integration

- [ ] 10.1 Update `TimeseriesItem` to support grid timeseries
- [ ] 10.2 Update `TimeseriesCollection::add_timeseries()` to accept grid timeseries
- [ ] 10.3 Add `TimeseriesCollection::add_grid_timeseries()` method
- [ ] 10.4 Update `TimeseriesCollection::get_by_name()` to handle grid timeseries
- [ ] 10.5 Write unit tests for adding and retrieving grid timeseries from collection
- [ ] 10.6 Test mixed collection with scalar and grid timeseries

## 11. Documentation

- [ ] 11.1 Write rustdoc for `SpatialGrid` trait with examples
- [ ] 11.2 Write rustdoc for `GridTimeseries` with comprehensive examples
- [ ] 11.3 Write rustdoc for each grid type (Scalar, FourBox, Hemispheric)
- [ ] 11.4 Add module-level documentation for `spatial` module
- [ ] 11.5 Add examples showing four-box usage patterns
- [ ] 11.6 Add examples showing grid transformation
- [ ] 11.7 Document when to use which grid type
- [ ] 11.8 Add migration guide for component developers
- [ ] 11.9 Document supported regions for each grid type (names, indices, constants)
- [ ] 11.10 Create transformation matrix table in documentation
- [ ] 11.11 Document transformation semantics (aggregation formulas, weights)
- [ ] 11.12 Add component integration pattern examples (5 patterns from design.md)
- [ ] 11.13 Document when to use broadcast vs. error for unsupported transformations
- [ ] 11.14 Add warnings about physically inappropriate transformations
- [ ] 11.15 Document how to implement custom disaggregation components

## 12. Python Bindings

- [ ] 12.1 Design Python API for grid timeseries (defer detailed implementation)
- [ ] 12.2 Add PyO3 bindings for `FourBoxGrid` if needed
- [ ] 12.3 Expose grid timeseries to Python as 2D numpy arrays
- [ ] 12.4 Add Python type stubs (.pyi) for grid timeseries
- [ ] 12.5 Write Python tests for grid timeseries usage

## 13. Example Components

- [ ] 13.1 Create example component using four-box timeseries (e.g., four-box ocean heat uptake)
- [ ] 13.2 Create example showing component coupling with grid transformation
- [ ] 13.3 Add example to documentation showing end-to-end four-box model
- [ ] 13.4 Write integration test for model with four-box components

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
