# state Specification Delta

## REMOVED Requirements

### ~~Requirement: Exogenous vs Endogenous Variable Handling~~

**Rationale:** This requirement conflated two different concerns (interpolation vs timestep index). Exogenous data is pre-interpolated onto the model time axis at build time, so runtime interpolation is not needed. The real distinction is about which timestep index to read from based on execution order, which is now covered by the "Timestep Access Semantics" requirement.

## ADDED Requirements

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

## MODIFIED Requirements

### Requirement: TimeseriesWindow for Scalar Access

The system SHALL provide a `TimeseriesWindow` type that provides zero-cost access to scalar timeseries data with temporal navigation, including explicit timestep-semantic methods.

### Requirement: GridTimeseriesWindow for Grid Access

The system SHALL provide a `GridTimeseriesWindow<G>` type that provides zero-cost access to grid timeseries data with both temporal and spatial navigation, including explicit timestep-semantic methods.

#### Scenario: Get all regional values at start of timestep

- **WHEN** calling `window.at_start_all()`
- **THEN** it MUST return a Vec containing values for all regions at index N

#### Scenario: Get all regional values at end of timestep

- **WHEN** calling `window.at_end_all()`
- **THEN** it MUST return `Some(Vec)` containing values for all regions at index N+1
- **AND** return `None` if at the final timestep

### Requirement: InputState Container

The system SHALL provide an `InputState` type that allows components to access their input variables during the solve phase via window accessors with explicit timestep semantics.

#### Scenario: Removed methods

- **WHEN** accessing input values
- **THEN** components MUST use window accessors (`get_scalar_window()`, `get_four_box_window()`, `get_hemispheric_window()`) with explicit `at_start()` or `at_end()` methods
- **AND** the methods `get_scalar_value()`, `get_four_box_values()`, `get_hemispheric_values()` SHALL NOT be available
