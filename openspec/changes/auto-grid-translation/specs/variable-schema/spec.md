# variable-schema Specification Delta

## ADDED Requirements

### Requirement: Grid Auto-Aggregation on Read

The system SHALL automatically aggregate variables at access time when a component requests a coarser grid than the schema's native resolution. Transformations occur in InputState when the component reads the variable, not via virtual components.

#### Scenario: Component reads scalar from FourBox variable

- **WHEN** a schema declares a variable with `GridType::FourBox`
- **AND** a component declares an input for that variable with `GridType::Scalar`
- **THEN** the model MUST build successfully
- **AND** when the component calls `get_scalar_window(name)`, InputState MUST aggregate the FourBox data
- **AND** the component MUST receive area-weighted aggregated values

#### Scenario: Component reads scalar from Hemispheric variable

- **WHEN** a schema declares a variable with `GridType::Hemispheric`
- **AND** a component declares an input for that variable with `GridType::Scalar`
- **THEN** the model MUST build successfully
- **AND** when the component calls `get_scalar_window(name)`, InputState MUST aggregate the Hemispheric data
- **AND** the component MUST receive area-weighted aggregated values

#### Scenario: Component reads Hemispheric from FourBox variable

- **WHEN** a schema declares a variable with `GridType::FourBox`
- **AND** a component declares an input for that variable with `GridType::Hemispheric`
- **THEN** the model MUST build successfully
- **AND** when the component calls `get_hemispheric_window(name)`, InputState MUST aggregate the FourBox data
- **AND** the component MUST receive per-hemisphere aggregated values

#### Scenario: Component reads at native resolution

- **WHEN** a component declares an input with the same grid type as the schema
- **THEN** no transformation MUST occur
- **AND** the component MUST receive values directly

#### Scenario: Timestep semantics preserved

- **WHEN** a component calls `window.at_start()` on an aggregated view
- **THEN** the aggregation MUST use source data at index N
- **AND** **WHEN** a component calls `window.at_end()` on an aggregated view
- **THEN** the aggregation MUST use source data at index N+1

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

The system SHALL automatically aggregate component outputs at write time when a component produces a finer grid than the schema declares. Transformations occur in the Model's step function when writing outputs to the collection.

#### Scenario: Component writes FourBox to Scalar variable

- **WHEN** a schema declares a variable with `GridType::Scalar`
- **AND** a component declares an output for that variable with `GridType::FourBox`
- **THEN** the model MUST build successfully
- **AND** the Model MUST aggregate FourBox to Scalar when writing to the collection

#### Scenario: Component writes Hemispheric to Scalar variable

- **WHEN** a schema declares a variable with `GridType::Scalar`
- **AND** a component declares an output for that variable with `GridType::Hemispheric`
- **THEN** the model MUST build successfully
- **AND** the Model MUST aggregate Hemispheric to Scalar when writing to the collection

#### Scenario: Component writes FourBox to Hemispheric variable

- **WHEN** a schema declares a variable with `GridType::Hemispheric`
- **AND** a component declares an output for that variable with `GridType::FourBox`
- **THEN** the model MUST build successfully
- **AND** the Model MUST aggregate FourBox to Hemispheric when writing to the collection

#### Scenario: Component writes coarser than schema (error)

- **WHEN** a schema declares a variable with `GridType::FourBox`
- **AND** a component declares an output for that variable with `GridType::Scalar`
- **THEN** the build MUST fail with `GridTransformationNotSupported` error
- **AND** the error MUST indicate that broadcast/disaggregation is not supported on write

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

### Requirement: Transformation Tracking

The system SHALL track required transformations for debugging and introspection, even though virtual components are not used.

#### Scenario: Query active transformations

- **WHEN** calling `model.required_transformations()`
- **THEN** it MUST return a list of (variable, source_grid, target_grid, direction) tuples
- **AND** direction MUST be either `Read` or `Write`

#### Scenario: Logging transformations

- **WHEN** a transformation occurs during model execution
- **THEN** it MAY be logged at debug level for troubleshooting

## MODIFIED Requirements

### Requirement: Schema Validation (modified)

The system SHALL validate component inputs against the schema, allowing coarser grid types for automatic aggregation.

#### Scenario: Input grid validation relaxed for coarser grids

- **WHEN** validating a component input against the schema
- **AND** the component's grid type is coarser than the schema's grid type
- **THEN** validation MUST pass (not fail with `ComponentSchemaGridMismatch`)
- **AND** the transformation MUST be recorded for runtime execution

#### Scenario: Output grid validation relaxed for finer grids

- **WHEN** validating a component output against the schema
- **AND** the component's grid type is finer than the schema's grid type
- **THEN** validation MUST pass
- **AND** the transformation MUST be recorded for runtime execution

#### Scenario: Disaggregation rejected

- **WHEN** validating a component input or output against the schema
- **AND** the transformation would require disaggregation (coarser to finer)
- **THEN** validation MUST fail with `GridTransformationNotSupported`
