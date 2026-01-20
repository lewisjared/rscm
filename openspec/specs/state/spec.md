# state Specification

## Purpose

Defines how components access input data and produce output data during model execution.
The state system provides type-safe, zero-cost abstractions for accessing timeseries data with support for both scalar and spatially-resolved (grid) variables.

## Requirements

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

### Requirement: InputState Container

The system SHALL provide an `InputState` type that allows components to access their input variables during the solve phase via window accessors with explicit timestep semantics.

### Requirement: TimeseriesWindow for Scalar Access

The system SHALL provide a `TimeseriesWindow` type that provides zero-cost access to scalar timeseries data with temporal navigation, including explicit timestep-semantic methods.

### Requirement: GridTimeseriesWindow for Grid Access

The system SHALL provide a `GridTimeseriesWindow<G>` type that provides zero-cost access to grid timeseries data with both temporal and spatial navigation, including explicit timestep-semantic methods.

#### Scenario: Get all regional values at start of timestep

- **WHEN** calling `window.current_all_at_start()`
- **THEN** it MUST return a Vec containing values for all regions at index N

#### Scenario: Get all regional values at end of timestep

- **WHEN** calling `window.current_all_at_end()`
- **THEN** it MUST return `Some(Vec)` containing values for all regions at index N+1
- **AND** return `None` if at the final timestep

### Requirement: Type-Safe FourBox Window Access

The system SHALL provide type-safe region access for FourBox grid windows.

#### Scenario: Get single region at current time

- **WHEN** calling `window.current(FourBoxRegion::NorthernOcean)`
- **THEN** it MUST return the value for that specific region
- **AND** accept only valid FourBoxRegion variants (compile-time safety)

#### Scenario: Get single region at previous time

- **WHEN** calling `window.previous(FourBoxRegion::SouthernLand)`
- **THEN** it MUST return `Some(value)` for that region at previous timestep
- **AND** return `None` if at the first timestep

#### Scenario: Interpolate single region

- **WHEN** calling `window.interpolate(t, FourBoxRegion::NorthernLand)`
- **THEN** it MUST interpolate only that region to the specified time

### Requirement: Type-Safe Hemispheric Window Access

The system SHALL provide type-safe region access for Hemispheric grid windows.

#### Scenario: Get hemispheric region at current time

- **WHEN** calling `window.current(HemisphericRegion::Northern)`
- **THEN** it MUST return the value for the Northern Hemisphere
- **AND** accept only valid HemisphericRegion variants

### Requirement: InputState Window Accessors

The system SHALL provide methods on InputState to obtain typed windows for variables.

#### Scenario: Get scalar window from InputState

- **WHEN** calling `input_state.get_scalar_window(name)`
- **THEN** it MUST return a `TimeseriesWindow` for that variable
- **AND** panic if the variable is not found
- **AND** panic if the variable is not a scalar timeseries

#### Scenario: Get FourBox window from InputState

- **WHEN** calling `input_state.get_four_box_window(name)`
- **THEN** it MUST return a `GridTimeseriesWindow<FourBoxGrid>` for that variable
- **AND** panic if the variable is not found
- **AND** panic if the variable is not a FourBox timeseries

#### Scenario: Get Hemispheric window from InputState

- **WHEN** calling `input_state.get_hemispheric_window(name)`
- **THEN** it MUST return a `GridTimeseriesWindow<HemisphericGrid>` for that variable
- **AND** panic if the variable is not found
- **AND** panic if the variable is not a Hemispheric timeseries

### Requirement: FourBoxSlice Output Type

The system SHALL provide a `FourBoxSlice` type for type-safe four-box output construction.

#### Scenario: Create with builder pattern

- **WHEN** using `FourBoxSlice::new().with(region, value)`
- **THEN** it MUST set the value for the specified region
- **AND** return self for method chaining
- **AND** initialise unset regions with NaN

#### Scenario: Create uniform slice

- **WHEN** using `FourBoxSlice::uniform(value)`
- **THEN** all four regions MUST have the same value

#### Scenario: FourBoxSlice Access via index operator

- **WHEN** using `slice[FourBoxRegion::NorthernOcean]`
- **THEN** it MUST return the value for that region
- **AND** support both read and write access

#### Scenario: Aggregate to global

- **WHEN** calling `slice.aggregate_global(&grid)`
- **THEN** it MUST use the grid's weights to compute weighted average

### Requirement: HemisphericSlice Output Type

The system SHALL provide a `HemisphericSlice` type for type-safe hemispheric output construction.

#### Scenario: Create hemispheric slice

- **WHEN** using `HemisphericSlice::new().with(region, value)`
- **THEN** it MUST set the value for Northern or Southern hemisphere
- **AND** return self for method chaining

#### Scenario: HemisphericRegion Access via index operator

- **WHEN** using `slice[HemisphericRegion::Northern]`
- **THEN** it MUST return the value for that hemisphere

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

### Requirement: Exogenous vs Endogenous Variable Handling

The system SHALL handle exogenous (external input) and endogenous (model-computed) variables differently.

#### Scenario: Exogenous variable access

- **WHEN** accessing an exogenous variable
- **THEN** the value MUST be interpolated to the current solve time
- **AND** use the timeseries's interpolation strategy

#### Scenario: Endogenous variable access

- **WHEN** accessing an endogenous variable
- **THEN** the latest computed value MUST be returned
- **AND** NOT interpolate (use discrete timestep value)

### Requirement: Timestep Access Semantics

The system SHALL provide explicit methods for accessing values at different points within a timestep, reflecting the forward-Euler time-stepping scheme where values at index N represent state at the start of timestep N, and components write outputs to index N+1.

#### Scenario: Access value at start of timestep

- **WHEN** calling `window.at_start()`
- **THEN** it MUST return the value at the current timestep index (N)
- **AND** NOT allocate memory

#### Scenario: Access value at end of timestep

- **WHEN** calling `window.at_end()`
- **THEN** it MUST return the value at the next timestep index (N+1)
- **AND** return the same value as `at_offset(1)`
- **AND** return `None` if at the final timestep (no N+1 exists)

#### Scenario: Guidance for state variable initial conditions

- **WHEN** a component reads its own state variable (appears in both inputs and outputs)
- **THEN** it SHOULD use `at_start()` because the N+1 slot has not been written yet

#### Scenario: Guidance for upstream component outputs

- **WHEN** a component reads an output from an upstream component in the dependency graph
- **THEN** it SHOULD use `at_end()` to get the value just computed by the upstream component

#### Scenario: Guidance for exogenous inputs

- **WHEN** a component reads exogenous (externally provided) input data
- **THEN** it SHOULD use `at_start()` for the value at the start of the timestep
- **OR** use `interpolate(t)` for values at intermediate times during ODE integration
