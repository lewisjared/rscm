# component-dx Specification

## MODIFIED Requirements

### Requirement: Typed Grid Output Slices

The `ComponentIO` macro SHALL generate code that uses typed slice wrappers for grid outputs.

**Modifications from original spec:**

- The generated `Into<OutputState>` implementation now wraps values in `StateValue` variants
- For FourBox outputs, use `StateValue::FourBox(slice)` instead of aggregating to scalar
- For Hemispheric outputs, use `StateValue::Hemispheric(slice)` instead of aggregating to scalar
- Scalar outputs continue to use `StateValue::Scalar(value)`

#### Scenario: FourBoxSlice converted to StateValue::FourBox

- **WHEN** a component declares `#[outputs(temp { name = "Temperature", unit = "K", grid = "FourBox" })]`
- **THEN** the generated `{ComponentName}Outputs` struct has a `temp: FourBoxSlice` field
- **AND** the `Into<OutputState>` implementation inserts `StateValue::FourBox(outputs.temp)`
- **AND** no aggregation to scalar occurs

#### Scenario: HemisphericSlice converted to StateValue::Hemispheric

- **WHEN** a component declares `#[outputs(precip { name = "Precipitation", unit = "mm", grid = "Hemispheric" })]`
- **THEN** the generated `{ComponentName}Outputs` struct has a `precip: HemisphericSlice` field
- **AND** the `Into<OutputState>` implementation inserts `StateValue::Hemispheric(outputs.precip)`

#### Scenario: Scalar output converted to StateValue::Scalar

- **WHEN** a component declares `#[outputs(co2 { name = "CO2", unit = "ppm" })]` (no grid specified)
- **THEN** the `Into<OutputState>` implementation inserts `StateValue::Scalar(outputs.co2)`
