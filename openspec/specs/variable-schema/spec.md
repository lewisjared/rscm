# variable-schema Specification

## Purpose

TBD - created by archiving change aggregate-ts. Update Purpose after archive.

## Requirements

### Requirement: VariableSchema Type

The system SHALL provide a `VariableSchema` type for declaring model variables and aggregates.

#### Scenario: Create empty schema

- **WHEN** creating `VariableSchema::new()`
- **THEN** it MUST return an empty schema with no variables or aggregates

#### Scenario: Add variable to schema

- **WHEN** calling `schema.variable(name, unit)`
- **THEN** it MUST add a scalar variable definition
- **AND** return self for method chaining
- **AND** use `GridType::Scalar` as default

#### Scenario: Add variable with grid type

- **WHEN** calling `schema.variable_with_grid(name, unit, grid_type)`
- **THEN** it MUST add a variable definition with the specified grid type

#### Scenario: Duplicate variable name

- **WHEN** adding a variable with a name that already exists
- **THEN** it MUST return an error

### Requirement: AggregateOp Enum

The system SHALL provide an `AggregateOp` enum for specifying aggregation operations.

#### Scenario: Sum operation

- **WHEN** using `AggregateOp::Sum`
- **THEN** aggregation MUST compute the sum of all contributor values

#### Scenario: Mean operation

- **WHEN** using `AggregateOp::Mean`
- **THEN** aggregation MUST compute the arithmetic mean of all contributor values

#### Scenario: Weighted operation

- **WHEN** using `AggregateOp::Weighted(weights)`
- **THEN** aggregation MUST compute the weighted sum of contributor values
- **AND** weights MUST be applied in contributor declaration order

### Requirement: Aggregate Builder

The system SHALL provide a builder pattern for declaring aggregates.

#### Scenario: Declare aggregate

- **WHEN** calling `schema.aggregate(name, unit, operation)`
- **THEN** it MUST return an `AggregateBuilder` for configuring contributors

#### Scenario: Add contributor

- **WHEN** calling `aggregate_builder.from(contributor_name)`
- **THEN** it MUST add the contributor to the aggregate's contributor list
- **AND** return self for method chaining

#### Scenario: Build aggregate

- **WHEN** calling `aggregate_builder.build()`
- **THEN** it MUST add the aggregate definition to the schema
- **AND** return the schema for further chaining

#### Scenario: Aggregate with no contributors

- **WHEN** building an aggregate with no `.from()` calls
- **THEN** it MUST be allowed (aggregate will be NaN at runtime)

### Requirement: Schema Validation

The system SHALL validate aggregate definitions within the schema.

#### Scenario: Contributor not in schema

- **WHEN** an aggregate references a contributor not defined in the schema (as a variable or aggregate)
- **THEN** validation MUST fail with `UndefinedContributor` error

#### Scenario: Unit mismatch between contributor and aggregate

- **WHEN** a contributor has a different unit than its aggregate
- **THEN** validation MUST fail with `UnitMismatch` error

#### Scenario: Circular aggregate dependency

- **WHEN** aggregate A references aggregate B which references aggregate A
- **THEN** validation MUST fail with `CircularDependency` error

#### Scenario: Grid type mismatch

- **WHEN** a contributor has a different grid type than its aggregate
- **THEN** validation MUST fail with `GridTypeMismatch` error

#### Scenario: Weighted aggregate weight count mismatch

- **WHEN** a `Weighted` aggregate has a different number of weights than contributors
- **THEN** validation MUST fail with `WeightCountMismatch` error

#### Scenario: Aggregate referencing another aggregate

- **WHEN** aggregate A lists aggregate B as a contributor
- **THEN** it MUST be allowed (aggregates can compose)
- **AND** B MUST be computed before A

### Requirement: ModelBuilder Schema Integration

The system SHALL integrate `VariableSchema` with `ModelBuilder`.

#### Scenario: Build without schema

- **WHEN** building a model without calling `with_schema()`
- **THEN** the model MUST build successfully using existing behaviour
- **AND** no schema validation MUST be performed
- **AND** no aggregation MUST be available

#### Scenario: Build with schema

- **WHEN** calling `model_builder.with_schema(schema)`
- **THEN** the builder MUST store the schema for validation

#### Scenario: Component output not in schema

- **WHEN** a component declares an output not defined in the schema
- **THEN** build MUST fail with `UndefinedVariable` error

#### Scenario: Component input not satisfied

- **WHEN** a component declares an input not in schema and not produced by another component
- **THEN** build MUST fail with `UndefinedVariable` error

#### Scenario: Schema variable not written

- **WHEN** a schema variable has no component writing to it
- **THEN** build MUST succeed
- **AND** the variable MUST contain NaN values at runtime

### Requirement: Aggregate Execution

The system SHALL compute aggregate values during model execution.

#### Scenario: Aggregate computed after contributors

- **WHEN** the model executes a timestep
- **THEN** aggregates MUST be computed after all contributor components have solved

#### Scenario: All contributors present

- **WHEN** all contributors have valid (non-NaN) values
- **THEN** the aggregate MUST be computed using the specified operation

#### Scenario: Some contributors NaN

- **WHEN** some contributors have NaN values
- **THEN** NaN contributors MUST be excluded from the computation
- **AND** the aggregate MUST be computed from remaining valid values
- **AND** for `Mean`, the divisor MUST be the count of valid values (not total contributors)
- **AND** for `Weighted`, the corresponding weights MUST also be excluded

#### Scenario: All contributors NaN

- **WHEN** all contributors have NaN values
- **THEN** the aggregate value MUST be NaN

#### Scenario: Zero contributors defined

- **WHEN** an aggregate has no contributors in its definition
- **THEN** the aggregate value MUST be NaN

### Requirement: Virtual Aggregator Nodes

The system SHALL create virtual nodes in the component graph for aggregates.

#### Scenario: Aggregator in graph

- **WHEN** a model contains aggregates
- **THEN** each aggregate MUST appear as a node in the component graph

#### Scenario: Aggregator dependencies

- **WHEN** viewing the component graph
- **THEN** the aggregator node MUST have edges from all contributor-producing components

#### Scenario: Graph visualisation

- **WHEN** calling `model.to_dot()`
- **THEN** aggregator nodes MUST be visible with a distinguishable style

### Requirement: Schema Serialization

The system SHALL support serialization of `VariableSchema`.

#### Scenario: Serialize to JSON

- **WHEN** serializing a schema with serde
- **THEN** it MUST produce valid JSON
- **AND** include all variables and aggregates

#### Scenario: Deserialize from JSON

- **WHEN** deserializing a schema from JSON
- **THEN** it MUST reconstruct the complete schema
- **AND** validation rules MUST apply on deserialization

### Requirement: Python Bindings

The system SHALL provide Python bindings for `VariableSchema`.

#### Scenario: Create schema from Python

- **WHEN** using `VariableSchema()` in Python
- **THEN** it MUST create an empty schema

#### Scenario: Builder pattern in Python

- **WHEN** using `.variable()` and `.aggregate()` in Python
- **THEN** method chaining MUST work as in Rust

#### Scenario: Pass schema to ModelBuilder

- **WHEN** calling `model_builder.with_schema(schema)` in Python
- **THEN** validation MUST work identically to Rust
