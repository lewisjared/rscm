# Tasks: Schema-Driven Grid Auto-Aggregation

## 1. Grid Coarseness Utilities

**File:** `crates/rscm-core/src/component.rs`

- [x] 1.1 Add `GridType::is_coarser_than(&self, other: GridType) -> bool` method
- [x] 1.2 Add `GridType::can_aggregate_to(&self, target: GridType) -> bool` method
- [x] 1.3 Add tests for coarseness comparison

## 2. Error Types

**File:** `crates/rscm-core/src/errors.rs`

- [x] 2.1 Add `GridTransformationNotSupported` error variant with source_grid, target_grid, variable fields
- [x] 2.2 Add helpful error message explaining that broadcast is not supported

## 3. Weight Configuration in ModelBuilder

**File:** `crates/rscm-core/src/model.rs`

- [x] 3.1 Add `grid_weights: HashMap<GridType, Vec<f64>>` field to `ModelBuilder`
- [x] 3.2 Implement `ModelBuilder::with_grid_weights(grid_type, weights)` method
- [x] 3.3 Validate weights sum to 1.0 in `with_grid_weights()`
- [x] 3.4 Pass weights to Model for use during execution
- [x] 3.5 Add `grid_weights` field to `Model` struct

## 4. ~~GridTransformerComponent~~ (SUPERSEDED)

**Status:** Removed in favour of transform-on-read approach.

The virtual component approach was rejected because:

- Timestep complexity (transformers need `at_end()` with fallback)
- Input remapping (components declare "Temperature" but read "Temperature|_to_scalar")
- Graph complexity (extra nodes for each transformation)

See design.md "Alternatives Rejected" section for details.

## 5. Relaxed Validation (Read and Write)

**File:** `crates/rscm-core/src/model.rs`

- [x] 5.1 Modify `validate_component_against_schema()` to allow coarser input grids (read-side)
- [x] 5.2 Modify `validate_component_against_schema()` to allow finer output grids (write-side)
- [x] 5.3 Track required read transformations (variable, component_grid, direction=Read)
- [x] 5.4 Track required write transformations (variable, component_grid, direction=Write)
- [x] 5.5 Return error for unsupported read transformations (finer-from-coarser)
- [x] 5.6 Return error for unsupported write transformations (coarser-to-finer)
- [x] 5.7 Add tests for relaxed input validation
- [x] 5.8 Add tests for relaxed output validation

## 6. Store Transformations for Runtime

**File:** `crates/rscm-core/src/model.rs`

- [x] 6.1 Add `read_transforms: HashMap<String, RequiredTransformation>` to `Model`
- [x] 6.2 Add `write_transforms: HashMap<String, RequiredTransformation>` to `Model`
- [x] 6.3 Populate transforms from `all_transformations` during `ModelBuilder::build()`
- [x] 6.4 Add `Model::required_transformations()` method for introspection
- [x] 6.5 Serialise/deserialise transforms with Model

## 7. Transform-on-Read in InputState

**File:** `crates/rscm-core/src/state.rs`

- [x] 7.1 Add transformation context to `InputState` (reference to read transforms + weights)
- [x] 7.2 Modify `get_scalar_window()` to detect FourBox/Hemispheric source and return aggregating window
- [x] 7.3 Modify `get_hemispheric_window()` to detect FourBox source and return aggregating window
- [x] 7.4 Create `AggregatingTimeseriesWindow` wrapper that aggregates on `at_start()`/`at_end()` calls
- [x] 7.5 Use Model's grid weights for aggregation calculations
- [x] 7.6 Add tests for aggregating window behaviour

## 8. Transform-on-Write in Model.step()

**File:** `crates/rscm-core/src/model.rs`

- [ ] 8.1 After component solve, check output variables against write_transforms
- [ ] 8.2 If transform needed, aggregate StateValue before writing to collection
- [ ] 8.3 Add helper function `aggregate_state_value(value, source_grid, target_grid, weights)`
- [ ] 8.4 Add tests for write-side aggregation

## 9. Integration Tests

**File:** `crates/rscm-core/src/model.rs` (tests module)

**Read-side tests:**

- [ ] 9.1 Test: FourBox schema variable with scalar consumer - auto-aggregates on read
- [ ] 9.2 Test: FourBox schema variable with hemispheric consumer - auto-aggregates on read
- [ ] 9.3 Test: Hemispheric schema variable with scalar consumer - auto-aggregates on read
- [ ] 9.4 Test: Multiple consumers at different resolutions - each gets correct aggregation
- [ ] 9.5 Test: Scalar schema variable with FourBox consumer - errors (no broadcast)

**Write-side tests:**

- [ ] 9.6 Test: Scalar schema variable with FourBox producer - auto-aggregates on write
- [ ] 9.7 Test: Scalar schema variable with Hemispheric producer - auto-aggregates on write
- [ ] 9.8 Test: Hemispheric schema variable with FourBox producer - auto-aggregates on write
- [ ] 9.9 Test: FourBox schema variable with Scalar producer - errors (no broadcast)

**Combined tests:**

- [ ] 9.10 Test: Chain with write-aggregate then read-aggregate
- [ ] 9.11 Test: Custom weights affect aggregation results
- [ ] 9.12 Test: Model without schema - unchanged behaviour (no auto-aggregation)

## 10. Python Bindings

**File:** `crates/rscm/src/python/mod.rs`

- [ ] 10.1 Expose `with_grid_weights()` on Python `ModelBuilder`
- [ ] 10.2 Add Python test for grid auto-aggregation

## 11. Documentation

**File:** `docs/grids.md`

- [ ] 11.1 Document schema-driven auto-aggregation feature
- [ ] 11.2 Add example showing FourBox→Scalar auto-aggregation
- [ ] 11.3 Document weight configuration via ModelBuilder
- [ ] 11.4 Update troubleshooting section for new error type
