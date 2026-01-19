# state Specification

## MODIFIED Requirements

### Requirement: StateValue Type

The system SHALL provide a `StateValue` enum that represents values which can be either scalar or spatially-resolved.

**Modifications from original spec:**

- Replace generic `Grid(Vec<FloatValue>)` variant with typed variants: `FourBox(FourBoxSlice)` and `Hemispheric(HemisphericSlice)`
- Remove `is_grid()` and `as_grid()` methods
- Add `is_four_box()`, `is_hemispheric()`, `as_four_box()`, `as_hemispheric()` methods

#### Scenario: FourBox state value

- **WHEN** creating `StateValue::FourBox(slice)`
- **THEN** `is_scalar()` MUST return false
- **AND** `is_four_box()` MUST return true
- **AND** `as_four_box()` MUST return `Some(&slice)`
- **AND** `to_scalar()` MUST return the mean of all four regions

#### Scenario: Hemispheric state value

- **WHEN** creating `StateValue::Hemispheric(slice)`
- **THEN** `is_scalar()` MUST return false
- **AND** `is_hemispheric()` MUST return true
- **AND** `as_hemispheric()` MUST return `Some(&slice)`
- **AND** `to_scalar()` MUST return the mean of both hemispheres

### Requirement: OutputState Type

The system SHALL provide an `OutputState` type for components to return their computed values.

**Modifications from original spec:**

- Changed from `HashMap<String, FloatValue>` to `HashMap<String, StateValue>`
- Components can now return grid outputs directly without aggregation
- Grid outputs are written to appropriate grid timeseries by the model

#### Scenario: FourBox output

- **WHEN** a component returns a FourBox output
- **THEN** the value MUST be `StateValue::FourBox(slice)`
- **AND** all four regions MUST contain values
- **AND** the model MUST write to a FourBox timeseries without aggregation

#### Scenario: Hemispheric output

- **WHEN** a component returns a Hemispheric output
- **THEN** the value MUST be `StateValue::Hemispheric(slice)`
- **AND** both hemispheres MUST contain values
- **AND** the model MUST write to a Hemispheric timeseries without aggregation
