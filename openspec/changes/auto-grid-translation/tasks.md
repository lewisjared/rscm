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

- [ ] 3.1 Add `grid_weights: HashMap<GridType, Vec<f64>>` field to `ModelBuilder`
- [ ] 3.2 Implement `ModelBuilder::with_grid_weights(grid_type, weights)` method
- [ ] 3.3 Validate weights sum to 1.0 in `with_grid_weights()`
- [ ] 3.4 Pass weights to Model for use during execution
- [ ] 3.5 Add `grid_weights` field to `Model` struct

## 4. GridTransformerComponent

**File:** `crates/rscm-core/src/schema.rs`

- [ ] 4.1 Create `GridTransformerComponent` struct with fields: source_var, target_grid, weights
- [ ] 4.2 Implement `Component` trait for `GridTransformerComponent`
- [ ] 4.3 Implement `solve()` to read source variable and apply `transform_to()`
- [ ] 4.4 Add `#[typetag::serde]` for serialization support
- [ ] 4.5 Add tests for GridTransformerComponent solve logic

## 5. Relaxed Validation (Read and Write)

**File:** `crates/rscm-core/src/model.rs`

- [ ] 5.1 Modify `validate_component_against_schema()` to allow coarser input grids (read-side)
- [ ] 5.2 Modify `validate_component_against_schema()` to allow finer output grids (write-side)
- [ ] 5.3 Track required read transformations (variable, component_grid, direction=Read)
- [ ] 5.4 Track required write transformations (variable, component_grid, direction=Write)
- [ ] 5.5 Return error for unsupported read transformations (finer-from-coarser)
- [ ] 5.6 Return error for unsupported write transformations (coarser-to-finer)
- [ ] 5.7 Add tests for relaxed input validation
- [ ] 5.8 Add tests for relaxed output validation

## 6. Transformer Insertion in Build

**File:** `crates/rscm-core/src/model.rs`

- [ ] 6.1 Collect all required transformations after component validation
- [ ] 6.2 Deduplicate transformations (one per variable+target_grid pair)
- [ ] 6.3 Create `GridTransformerComponent` for each unique transformation
- [ ] 6.4 Insert transformer nodes into component graph
- [ ] 6.5 Create edges: producer → transformer, transformer → consumers
- [ ] 6.6 Create intermediate timeseries for transformed outputs (e.g., "Var|_to_scalar")
- [ ] 6.7 Add tests for transformer insertion

## 7. Timeseries Creation for Transforms

**File:** `crates/rscm-core/src/model.rs`

- [ ] 7.1 Add transformed variable definitions to `definitions` map
- [ ] 7.2 Ensure TimeseriesCollection creates timeseries at target grid type
- [ ] 7.3 Wire consumer components to read from transformed variable name

## 8. Graph Visualisation

**File:** `crates/rscm-core/src/model.rs`

- [ ] 8.1 Add distinguishing style for transformer nodes in `to_dot()`
- [ ] 8.2 Label transformer nodes with transformation (e.g., "Temperature: FourBox→Scalar")

## 9. Integration Tests

**File:** `crates/rscm-core/src/model.rs` (tests module)

**Read-side tests:**
- [ ] 9.1 Test: FourBox schema variable with scalar consumer - auto-aggregates on read
- [ ] 9.2 Test: FourBox schema variable with hemispheric consumer - auto-aggregates on read
- [ ] 9.3 Test: Hemispheric schema variable with scalar consumer - auto-aggregates on read
- [ ] 9.4 Test: Multiple consumers at different resolutions - shares read transformers
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
