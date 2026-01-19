# Tasks: Variable Schema with Timeseries Aggregation

## 1. Core Types

- [x] 1.1 Create `crates/rscm-core/src/schema.rs` with `AggregateOp` enum
- [x] 1.2 Add `VariableDefinition` struct (named `SchemaVariableDefinition` to avoid conflict with model.rs)
- [x] 1.3 Add `AggregateDefinition` struct
- [x] 1.4 Add `VariableSchema` struct with `variables` and `aggregates` HashMaps
- [x] 1.5 Export from `crates/rscm-core/src/lib.rs`

## 2. Builder API

- [x] 2.1 Implement `VariableSchema::new()`
- [x] 2.2 Implement `VariableSchema::variable(name, unit)` for scalar variables
- [x] 2.3 Implement `VariableSchema::variable_with_grid(name, unit, grid_type)`
- [x] 2.4 Create `AggregateBuilder` struct
- [x] 2.5 Implement `VariableSchema::aggregate(name, unit, op)` returning `AggregateBuilder`
- [x] 2.6 Implement `AggregateBuilder::from(contributor)` for adding contributors
- [x] 2.7 Implement `AggregateBuilder::build()` returning `VariableSchema`

## 3. Schema Validation

- [x] 3.1 Add `VariableSchema::validate()` method
- [x] 3.2 Validate contributor references exist in schema (as variable or aggregate)
- [x] 3.3 Validate unit consistency between contributors and aggregates
- [x] 3.4 Validate grid type consistency between contributors and aggregates
- [x] 3.5 Validate weighted aggregate weights count matches contributors count
- [x] 3.6 Detect circular aggregate dependencies
- [x] 3.7 Add error variants to `RSCMError`: `UndefinedContributor`, `SchemaUnitMismatch`, `SchemaGridTypeMismatch`, `WeightCountMismatch`, `AggregateCircularDependency`, `SchemaUndefinedOutput`, `SchemaUndefinedInput`, `ComponentSchemaUnitMismatch`, `ComponentSchemaGridMismatch`

## 4. ModelBuilder Integration

- [x] 4.1 Add `schema: Option<VariableSchema>` field to `ModelBuilder`
- [x] 4.2 Implement `ModelBuilder::with_schema(schema)`
- [x] 4.3 Validate component outputs against schema in `build()`
- [x] 4.4 Validate component inputs against schema in `build()`
- [x] 4.5 Create timeseries for schema variables not written by components (NaN)

## 5. Aggregate Execution

- [x] 5.1 Create virtual aggregator component type (internal)
- [x] 5.2 Insert aggregator nodes into component graph during `build()`
- [x] 5.3 Add dependency edges from contributor-producing components to aggregator
- [x] 5.4 Implement `compute_aggregate(contributors, op)` function
- [x] 5.5 Handle NaN contributors (exclude from computation, filter corresponding weights for Weighted)
- [x] 5.6 Ensure Mean uses count of valid values as divisor
- [x] 5.7 Write aggregate values to timeseries collection after contributors solve

## 6. Serialization

- [x] 6.1 Add `#[derive(Serialize, Deserialize)]` to all schema types
- [ ] 6.2 Add validation on deserialization
- [x] 6.3 Test round-trip JSON serialization

## 7. Python Bindings

- [x] 7.1 Add `#[pyclass]` to `AggregateOp` (Note: complex enum - using string-based API instead)
- [x] 7.2 Add `#[pyclass]` to `VariableSchema`
- [x] 7.3 Implement `#[pymethods]` for builder pattern
- [x] 7.4 Add `with_schema()` to Python `ModelBuilder`
- [x] 7.5 Export from `crates/rscm/src/python/mod.rs`

## 8. Testing

- [x] 8.1 Unit tests for `VariableSchema` builder
- [x] 8.2 Unit tests for schema validation (happy path)
- [x] 8.3 Unit tests for schema validation (error cases)
- [x] 8.4 Integration test: model with sum aggregate
- [x] 8.5 Integration test: model with mean aggregate
- [x] 8.6 Integration test: model with weighted aggregate
- [x] 8.7 Integration test: aggregate with NaN contributors
- [x] 8.8 Integration test: schema variable with no writer (NaN)
- [x] 8.9 Python integration test for schema API

## 9. Documentation

- [ ] 9.1 Rustdoc for all public types
- [ ] 9.2 Update CLAUDE.md with schema usage example
- [x] 9.3 Update Python type stubs (.pyi files)

## Dependencies

- Tasks 1.x must complete before 2.x
- Tasks 2.x must complete before 3.x
- Tasks 3.x and 4.x can run in parallel
- Tasks 5.x depend on 4.x
- Tasks 6.x can run in parallel with 5.x
- Tasks 7.x depend on 1.x, 2.x, and 3.x (Python bindings need validation logic)
- Tasks 8.x depend on all implementation tasks
- Tasks 9.x can start after 1.x

## Parallelizable Work

- 3.x and 4.x (validation logic is independent)
- 6.x and 5.x (serialization is independent of execution)
- 7.x and 5.x (Python bindings can start after 3.x, don't need execution logic)
