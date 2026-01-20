# variable-schema Specification Delta

## ADDED Requirements

### Requirement: Grid Auto-Aggregation

The system SHALL automatically aggregate variables when a component requests a coarser grid than the schema's native resolution.

#### Scenario: Component reads scalar from FourBox variable

- **WHEN** a schema declares a variable with `GridType::FourBox`
- **AND** a component declares an input for that variable with `GridType::Scalar`
- **THEN** the model MUST build successfully
- **AND** a virtual `GridTransformerComponent` MUST be inserted into the graph
- **AND** the component MUST receive area-weighted aggregated values

#### Scenario: Component reads scalar from Hemispheric variable

- **WHEN** a schema declares a variable with `GridType::Hemispheric`
- **AND** a component declares an input for that variable with `GridType::Scalar`
- **THEN** the model MUST build successfully
- **AND** the component MUST receive area-weighted aggregated values

#### Scenario: Component reads Hemispheric from FourBox variable

- **WHEN** a schema declares a variable with `GridType::FourBox`
- **AND** a component declares an input for that variable with `GridType::Hemispheric`
- **THEN** the model MUST build successfully
- **AND** the component MUST receive per-hemisphere aggregated values

#### Scenario: Component reads at native resolution

- **WHEN** a component declares an input with the same grid type as the schema
- **THEN** no transformer MUST be inserted
- **AND** the component MUST receive values directly

#### Scenario: Component requests finer grid than schema (error)

- **WHEN** a schema declares a variable with `GridType::Scalar`
- **AND** a component declares an input for that variable with `GridType::FourBox`
- **THEN** the build MUST fail with `GridTransformationNotSupported` error
- **AND** the error MUST indicate that broadcast is not supported

#### Scenario: Hemispheric to FourBox not supported (error)

- **WHEN** a schema declares a variable with `GridType::Hemispheric`
- **AND** a component declares an input for that variable with `GridType::FourBox`
- **THEN** the build MUST fail with `GridTransformationNotSupported` error

### Requirement: Grid Auto-Aggregation on Write

The system SHALL automatically aggregate component outputs when a component produces a finer grid than the schema declares.

#### Scenario: Component writes FourBox to Scalar variable

- **WHEN** a schema declares a variable with `GridType::Scalar`
- **AND** a component declares an output for that variable with `GridType::FourBox`
- **THEN** the model MUST build successfully
- **AND** a virtual `GridTransformerComponent` MUST be inserted after the component
- **AND** the FourBox output MUST be aggregated to Scalar before storage

#### Scenario: Component writes Hemispheric to Scalar variable

- **WHEN** a schema declares a variable with `GridType::Scalar`
- **AND** a component declares an output for that variable with `GridType::Hemispheric`
- **THEN** the model MUST build successfully
- **AND** the Hemispheric output MUST be aggregated to Scalar before storage

#### Scenario: Component writes FourBox to Hemispheric variable

- **WHEN** a schema declares a variable with `GridType::Hemispheric`
- **AND** a component declares an output for that variable with `GridType::FourBox`
- **THEN** the model MUST build successfully
- **AND** the FourBox output MUST be aggregated to Hemispheric before storage

#### Scenario: Component writes coarser than schema (error)

- **WHEN** a schema declares a variable with `GridType::FourBox`
- **AND** a component declares an output for that variable with `GridType::Scalar`
- **THEN** the build MUST fail with `GridTransformationNotSupported` error
- **AND** the error MUST indicate that broadcast/disaggregation is not supported on write

### Requirement: GridTransformerComponent

The system SHALL provide a virtual component for performing grid transformations.

#### Scenario: Transformer in component graph

- **WHEN** a model requires grid transformation
- **THEN** a `GridTransformerComponent` node MUST appear in the component graph
- **AND** it MUST have an edge from the component that produces the native-resolution variable

#### Scenario: Timestep access semantics

- **WHEN** the transformer reads from the native-resolution variable
- **THEN** it MUST use `at_end()` to read the value written by the upstream component this timestep
- **AND** fall back to `at_start()` if at the final timestep (when `at_end()` returns `None`)

#### Scenario: Transformer naming convention

- **WHEN** a transformer aggregates variable "Temperature" to Scalar
- **THEN** the transformed output MUST use name "Temperature|_to_scalar"
- **AND** the original variable MUST remain at native resolution

#### Scenario: Graph visualisation

- **WHEN** calling `model.to_dot()`
- **THEN** transformer nodes MUST be visible with a distinguishable style
- **AND** labelled with the transformation (e.g., "FourBox→Scalar")

#### Scenario: Transformer caching

- **WHEN** multiple components request the same transformation
- **THEN** only one transformer MUST be created for that (variable, target_grid) pair
- **AND** all requesting components MUST share the transformed output

### Requirement: Aggregation Weight Configuration

The system SHALL support configurable area weights for grid aggregation.

#### Scenario: Default weights

- **WHEN** no custom weights are configured
- **THEN** FourBox aggregation MUST use equal weights [0.25, 0.25, 0.25, 0.25]
- **AND** Hemispheric aggregation MUST use equal weights [0.5, 0.5]

#### Scenario: Custom weights via ModelBuilder

- **WHEN** calling `model_builder.with_grid_weights(GridType::FourBox, weights)`
- **THEN** all FourBox→Scalar transformations MUST use the provided weights
- **AND** FourBox→Hemispheric transformations MUST derive hemisphere weights from provided weights

#### Scenario: Weight validation

- **WHEN** providing custom weights
- **THEN** weights MUST sum to approximately 1.0 (within 1e-6)
- **AND** build MUST fail if weights are invalid

## MODIFIED Requirements

### Requirement: Schema Validation (modified)

The system SHALL validate component inputs against the schema, allowing coarser grid types for automatic aggregation.

#### Scenario: Input grid validation relaxed for coarser grids

- **WHEN** validating a component input against the schema
- **AND** the component's grid type is coarser than the schema's grid type
- **THEN** validation MUST pass (not fail with `ComponentSchemaGridMismatch`)
- **AND** a transformation MUST be scheduled

#### Scenario: Output grid validation unchanged

- **WHEN** validating a component output against the schema
- **AND** the component's grid type differs from the schema's grid type
- **THEN** validation MUST fail with `ComponentSchemaGridMismatch`
- **AND** writers MUST match the schema's native resolution
