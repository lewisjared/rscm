# grid-output-state Specification Delta

## Purpose

Extends the component state system to support spatially-resolved (grid) output values.
This enables components to return FourBox and Hemispheric outputs natively without aggregation to scalars.

## MODIFIED Requirements

### Requirement: StateValue Type (Modified)

The system SHALL provide a `StateValue` enum that represents values which can be scalar or grid-typed.

**Modifications from `component-state` spec:**

- Add `FourBox(FourBoxSlice)` variant for four-box regional values
- Add `Hemispheric(HemisphericSlice)` variant for hemispheric values
- Remove generic `Grid(Vec<FloatValue>)` variant (replaced by typed variants)

#### Scenario: Scalar state value

- **WHEN** creating `StateValue::Scalar(value)`
- **THEN** `is_scalar()` MUST return true
- **AND** `as_scalar()` MUST return `Some(value)`
- **AND** `to_scalar()` MUST return the value unchanged

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

#### Scenario: StateValue from FourBoxSlice

- **WHEN** converting `FourBoxSlice` to `StateValue`
- **THEN** `From<FourBoxSlice>` MUST produce `StateValue::FourBox(slice)`

#### Scenario: StateValue from HemisphericSlice

- **WHEN** converting `HemisphericSlice` to `StateValue`
- **THEN** `From<HemisphericSlice>` MUST produce `StateValue::Hemispheric(slice)`

### Requirement: OutputState Type (Modified)

The system SHALL provide an `OutputState` type that supports both scalar and grid values.

**Modifications from `component-state` spec:**

- Change from `HashMap<String, FloatValue>` to `HashMap<String, StateValue>`
- Remove requirement that grid components aggregate before returning

#### Scenario: OutputState with scalar values

- **WHEN** a component returns scalar outputs
- **THEN** values MUST be wrapped as `StateValue::Scalar(value)`
- **AND** the HashMap key MUST match the variable name from `RequirementDefinition`

#### Scenario: OutputState with FourBox values

- **WHEN** a component returns FourBox outputs
- **THEN** values MUST be `StateValue::FourBox(slice)`
- **AND** the slice MUST contain values for all four regions

#### Scenario: OutputState with Hemispheric values

- **WHEN** a component returns Hemispheric outputs
- **THEN** values MUST be `StateValue::Hemispheric(slice)`
- **AND** the slice MUST contain values for both hemispheres

## ADDED Requirements

### Requirement: Model Grid Output Writing

The system SHALL support writing grid outputs from components to the TimeseriesCollection.

#### Scenario: Write scalar output to collection

- **WHEN** the model processes `StateValue::Scalar(value)` for a variable
- **THEN** it MUST write to the scalar timeseries at the next time index
- **AND** use `ScalarRegion::Global` as the region

#### Scenario: Write FourBox output to collection

- **WHEN** the model processes `StateValue::FourBox(slice)` for a variable
- **THEN** it MUST write to the FourBox timeseries at the next time index
- **AND** write values for all four regions
- **AND** the timeseries MUST be a `TimeseriesData::FourBox` variant

#### Scenario: Write Hemispheric output to collection

- **WHEN** the model processes `StateValue::Hemispheric(slice)` for a variable
- **THEN** it MUST write to the Hemispheric timeseries at the next time index
- **AND** write values for both hemispheres
- **AND** the timeseries MUST be a `TimeseriesData::Hemispheric` variant

#### Scenario: Grid type mismatch error

- **WHEN** the model receives a `StateValue` variant that does not match the `TimeseriesData` variant
- **THEN** it MUST return an error
- **AND** the error MUST include the variable name
- **AND** the error MUST indicate the expected and actual grid types

### Requirement: GridTimeseries Bulk Set

The system SHALL provide methods to set all regional values at a time index.

#### Scenario: Set all FourBox values

- **WHEN** calling `grid_timeseries.set_all(time_index, values)`
- **THEN** it MUST set values for all regions at the specified time
- **AND** update the `latest` index if the time_index is newer
- **AND** panic if `values.len()` does not equal `grid.size()`

#### Scenario: Set FourBox from slice

- **WHEN** calling `grid_timeseries.set_from_slice(time_index, slice)`
- **THEN** it MUST set values from the `FourBoxSlice` at the specified time

### Requirement: Grid Initial Values

The system SHALL support grid-typed initial values for state variables.

#### Scenario: ModelBuilder with scalar initial value

- **WHEN** calling `builder.with_initial_value(name, value)`
- **THEN** it MUST store `StateValue::Scalar(value)` for the variable
- **AND** initialise the timeseries with this value at time index 0

#### Scenario: ModelBuilder with FourBox initial value

- **WHEN** calling `builder.with_initial_four_box(name, slice)`
- **THEN** it MUST store `StateValue::FourBox(slice)` for the variable
- **AND** initialise the FourBox timeseries with these values at time index 0

#### Scenario: ModelBuilder with Hemispheric initial value

- **WHEN** calling `builder.with_initial_hemispheric(name, slice)`
- **THEN** it MUST store `StateValue::Hemispheric(slice)` for the variable
- **AND** initialise the Hemispheric timeseries with these values at time index 0

#### Scenario: Initial value grid type mismatch

- **WHEN** the initial value grid type does not match the variable's `RequirementDefinition.grid_type`
- **THEN** building the model MUST return an error
- **AND** the error MUST indicate the expected and actual grid types

### Requirement: ComponentIO Macro Grid Output Support

The system SHALL support grid outputs in the ComponentIO derive macro.

#### Scenario: Macro generates FourBox output conversion

- **GIVEN** a component with `#[outputs(heat_flux { name = "Heat Flux", unit = "W/m^2", grid = "FourBox" })]`
- **WHEN** the macro generates `From<Outputs> for OutputState`
- **THEN** it MUST insert `StateValue::FourBox(outputs.heat_flux)`
- **AND** NOT aggregate to scalar

#### Scenario: Macro generates Hemispheric output conversion

- **GIVEN** a component with `#[outputs(temp { name = "Temperature", unit = "K", grid = "Hemispheric" })]`
- **WHEN** the macro generates `From<Outputs> for OutputState`
- **THEN** it MUST insert `StateValue::Hemispheric(outputs.temp)`

#### Scenario: Macro generates scalar output conversion

- **GIVEN** a component with `#[outputs(co2 { name = "CO2", unit = "ppm" })]` (no grid specified)
- **WHEN** the macro generates `From<Outputs> for OutputState`
- **THEN** it MUST insert `StateValue::Scalar(outputs.co2)`

### Requirement: ModelBuilder Grid Timeseries Creation

The system SHALL create grid timeseries based on RequirementDefinition.grid_type.

#### Scenario: Create FourBox timeseries for FourBox output

- **GIVEN** a component with `RequirementDefinition::four_box_output("Temperature", "K")`
- **WHEN** the model is built
- **THEN** the TimeseriesCollection MUST contain a `TimeseriesData::FourBox` for "Temperature"
- **AND** it MUST use `FourBoxGrid::magicc_standard()`

#### Scenario: Create Hemispheric timeseries for Hemispheric output

- **GIVEN** a component with `RequirementDefinition::hemispheric_output("Precip", "mm/yr")`
- **WHEN** the model is built
- **THEN** the TimeseriesCollection MUST contain a `TimeseriesData::Hemispheric` for "Precip"
- **AND** it MUST use `HemisphericGrid::equal_weights()`

#### Scenario: Create scalar timeseries for scalar output

- **GIVEN** a component with `RequirementDefinition::scalar_output("CO2", "ppm")`
- **WHEN** the model is built
- **THEN** the TimeseriesCollection MUST contain a `TimeseriesData::Scalar` for "CO2"

## Cross-References

- Extends: `component-state` spec (OutputState, StateValue requirements)
- Depends on: `grid-timeseries` spec (GridTimeseries, FourBoxSlice, HemisphericSlice)
- Depends on: `component-dx` spec (ComponentIO macro)
