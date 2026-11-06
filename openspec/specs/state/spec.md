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

The system SHALL provide an `InputState` type that allows components to access their input variables
during the solve phase via windows.
A window's `get()` returns the value the model wired the input to, so components never choose a time index.

### Requirement: TimeseriesWindow for Scalar Access

The system SHALL provide a `TimeseriesWindow` type that provides zero-cost access to scalar timeseries data
through `get()`, `interval()` and `history(n)`.

### Requirement: GridTimeseriesWindow for Grid Access

The system SHALL provide a `GridTimeseriesWindow<G>` type with zero-cost access to grid timeseries data,
across time and regions, through `get_all()`, `interval()` and `history(n)`.

#### Scenario: Get all regional values

- **WHEN** calling `window.get_all()`
- **THEN** it MUST return a Vec containing values for all regions at the index `get()` reads

#### Scenario: Get earlier regional values

- **WHEN** calling `window.history(n)`
- **THEN** it MUST return `Some(Vec)` containing values for all regions `n` steps before `get_all()`
- **AND** return `None` before the first timestep

### Requirement: Type-Safe FourBox Window Access

The system SHALL provide type-safe region access for FourBox grid windows.

#### Scenario: Get single region

- **WHEN** calling `window.get(FourBoxRegion::NorthernOcean)`
- **THEN** it MUST return the value for that specific region at the index `get()` reads
- **AND** accept only valid FourBoxRegion variants (compile-time safety)

#### Scenario: Interpolate single region

- **WHEN** calling `window.interpolate(t, FourBoxRegion::NorthernLand)`
- **THEN** it MUST interpolate only that region to the specified time

### Requirement: Type-Safe Hemispheric Window Access

The system SHALL provide type-safe region access for Hemispheric grid windows.

#### Scenario: Get hemispheric region

- **WHEN** calling `window.get(HemisphericRegion::Northern)`
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

The system SHALL let components read inputs without choosing a time index,
reflecting the forward-Euler time-stepping scheme,
where values at index N represent state at the start of timestep N,
and components write outputs to index N+1.
The model builder decides which value `get()` returns from the model wiring.

#### Scenario: Read an input

- **WHEN** calling `window.get()`
- **THEN** it MUST return the value at index N+1 for an output written upstream in the same step
- **AND** return the value at index N for exogenous data, a component's own state and a lagged edge
- **AND** NOT allocate memory
- **AND** panic with the variable name if that index lies past the end of the timeseries

#### Scenario: Read a lagged input

- **WHEN** the model wiring lags an edge with `with_lag(variable)` or `with_lag_for(variable, consumer)`
- **THEN** `get()` MUST return the previous step's value for that edge
- **AND** `interval()` MUST return the previous step's start and end values

#### Scenario: Read the start and end of the step

- **WHEN** calling `window.interval()` on a point variable
- **THEN** it MUST return the values at the start and end of the step
- **AND** return an error for an average variable or a component's own state

#### Scenario: Read the past

- **WHEN** calling `window.history(n)`
- **THEN** it MUST return the value `n` steps before the one `get()` returns
- **AND** return `None` before the first timestep
- **AND** never return a value after the one `get()` returns

#### Scenario: Read a fixed calendar year

- **WHEN** a component reads exogenous data at a fixed time, such as a reference year
- **THEN** it SHOULD use `interpolate(t)`, which may read ahead of the step
