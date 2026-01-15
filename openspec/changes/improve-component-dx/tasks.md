## 1. Core Infrastructure

- [x] 1.1 Add `TimeseriesWindow` struct to `rscm-core/src/state.rs`
- [x] 1.2 Implement `current()`, `previous()`, `at_offset()` methods
- [x] 1.3 Implement `last_n()` returning `ArrayView1`
- [x] 1.4 Implement `interpolate()` delegating to timeseries strategy
- [x] 1.5 Add `GridTimeseriesWindow` for grid-based timeseries
- [x] 1.6 Add unit tests for `TimeseriesWindow` with scalar timeseries
- [x] 1.7 Add unit tests for `GridTimeseriesWindow` with FourBox timeseries

## 2. Typed Output Slices

- [x] 2.1 Add `FourBoxSlice` zero-cost wrapper with `#[repr(transparent)]`
- [x] 2.2 Implement `new()`, `with()`, `set()`, `get()` for FourBoxSlice
- [x] 2.3 Add `HemisphericSlice` wrapper
- [x] 2.4 Add `Into<Vec<FloatValue>>` for slice types
- [x] 2.5 Add unit tests for typed slices
- [x] 2.6 Add Python bindings for slice types with named kwargs

## 3. RequirementDefinition Updates

- [x] 3.1 Add `RequirementType::State` variant to `component.rs`
- [x] 3.2 Add `GridType` enum (Scalar, FourBox, Hemispheric, Any)
- [x] 3.3 Add `grid: GridType` field to `RequirementDefinition`
- [x] 3.4 Update `inputs()` to include State requirements
- [x] 3.5 Update `outputs()` to include State requirements
- [x] 3.6 Remove `RequirementType::InputAndOutput`
- [x] 3.7 Add tests for new requirement types

## 4. Coupler Grid Validation

- [x] 4.1 Add grid compatibility checking in `ModelBuilder::build()`
- [x] 4.2 Create `GridMismatch` error type with descriptive message
- [x] 4.3 Add `GridTransformComponent` for auto-aggregation
- [x] 4.4 Implement FourBox → Scalar aggregation transform
- [x] 4.5 Implement FourBox → Hemispheric aggregation transform
- [x] 4.6 Implement Hemispheric → Scalar aggregation transform
- [x] 4.7 Add integration tests for grid auto-transform
- [x] 4.8 Add integration tests for grid mismatch errors

## 5. Rust Derive Macro (rscm-macros crate)

- [x] 5.1 Create `rscm-macros` crate with proc-macro setup
- [x] 5.2 Add `rscm-macros` to workspace Cargo.toml
- [x] 5.3 Implement `#[derive(Component)]` basic parsing
- [x] 5.4 Parse `#[component(...)]` attribute for inputs/outputs/state
- [x] 5.5 Generate `{Name}Inputs` struct with TimeseriesWindow fields
- [x] 5.6 Generate `{Name}Outputs` struct with typed slice fields
- [x] 5.7 Handle grid-aware field types (FourBoxSlice for FourBox, FloatValue for Scalar)
- [x] 5.8 Implement `Into<OutputState>` for generated outputs
- [x] 5.9 Generate `definitions()` implementation from attributes
- [x] 5.10 Generate `solve()` wrapper that converts InputState to typed inputs
- [x] 5.11 Re-export macro from `rscm-core`
- [ ] 5.12 Add compile-fail tests for invalid field access

## 6. Rewrite Existing Components

- [x] 6.1 Rewrite `CO2ERF` component using derive macro
- [x] 6.2 Rewrite `CarbonCycleComponent` using derive macro
- [x] 6.3 Update `TestComponent` example component
- [x] 6.4 Remove old `InputState.get_latest()` API
- [x] 6.5 Update all component tests

## 7. Python Typed Inputs

- [ ] 7.1 Create `ComponentMeta` metaclass for Python components
- [ ] 7.2 Implement dataclass generation from `inputs`/`outputs` lists
- [x] 7.3 Create `PyTimeseriesWindow` PyO3 class
- [x] 7.4 Implement `current`, `previous` properties
- [x] 7.5 Implement `last_n()` returning numpy array view
- [x] 7.6 Create `PyGridTimeseriesWindow` for grid variables
- [x] 7.7 Implement `region()` method for single-region access
- [ ] 7.8 Update `PythonComponent` to construct typed inputs
- [x] 7.9 Add Python tests for typed scalar input access
- [x] 7.10 Add Python tests for typed grid input access

## 8. Python Typed Outputs

- [x] 8.1 Create `PyFourBoxSlice` with named kwargs constructor
- [x] 8.2 Create `PyHemisphericSlice` with named kwargs
- [ ] 8.3 Generate output dataclass using typed slices
- [ ] 8.4 Implement `__post_init__` validation for required fields
- [ ] 8.5 Convert typed outputs to dict for Rust interop
- [x] 8.6 Add Python tests for typed slice outputs

## 9. Error Handling Improvements

- [x] 9.1 Add `MissingInitialValue` error variant
- [x] 9.2 Add `GridMismatch` error variant with component names
- [x] 9.3 Add `InvalidGridTransform` error for impossible transforms
- [x] 9.4 Update error messages to include resolution hints

## 10. Documentation

- [x] 10.1 Add docstrings to all new public types
- [x] 10.2 Document grid compatibility rules
- [x] 10.3 Add example component using derive macro in docs
- [ ] 10.4 Add example Python component with typed inputs
- [x] 10.5 Update CLAUDE.md with new component patterns

## 11. Integration Testing

- [x] 11.1 Test model with multiple components using derive macro
- [ ] 11.2 Test model with grid auto-transform between components
- [x] 11.3 Test Python component with typed inputs/outputs
- [x] 11.4 Test serialisation/deserialisation with new components
- [x] 11.5 Test error messages for common mistakes
