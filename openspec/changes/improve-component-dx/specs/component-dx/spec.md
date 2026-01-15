## ADDED Requirements

### Requirement: TimeseriesWindow State Access

The system SHALL provide a `TimeseriesWindow` type that gives components zero-cost access to timeseries data at a specific time index.

The window MUST support:

- Current value access
- Previous value access
- Offset-based historical access (last N values)
- Time-based interpolation

#### Scenario: Access current and previous values

- **WHEN** a component receives a `TimeseriesWindow` for a variable
- **THEN** calling `current()` returns the value at the current time index
- **AND** calling `previous()` returns the value at the previous time index or `None` if at the first timestep

#### Scenario: Access historical slice

- **WHEN** a component calls `last_n(5)` on a `TimeseriesWindow`
- **THEN** it receives an array view of the 5 most recent values including current
- **AND** no data is copied (zero-cost view)

#### Scenario: Interpolate at arbitrary time

- **WHEN** a component calls `interpolate(t)` on a `TimeseriesWindow`
- **THEN** it receives the interpolated value using the timeseries interpolation strategy

---

### Requirement: State Requirement Type

The system SHALL support a `RequirementType::State` for variables that read their previous value and write a new value each timestep.

State variables MUST require an initial value to be provided at model build time or runtime configuration.

#### Scenario: State variable in component definition

- **WHEN** a component declares a variable with `RequirementType::State`
- **THEN** the variable appears in both `inputs()` and `outputs()`
- **AND** the model builder validates that an initial value is provided

#### Scenario: State variable without initial value

- **WHEN** a model is built with a component that has a State variable
- **AND** no initial value is provided for that variable
- **THEN** the build fails with a descriptive error identifying the missing variable

---

### Requirement: Spatial Grid Requirements in Definitions

The system SHALL support declaring spatial grid requirements on each input, output, and state variable.

Supported grid types MUST include: Scalar, FourBox, Hemispheric, and Any.

#### Scenario: Component declares grid requirements

- **WHEN** a component declares an input with `grid = FourBox`
- **THEN** the requirement definition includes the grid type
- **AND** the coupler can validate grid compatibility

#### Scenario: Any grid accepts all types

- **WHEN** a component declares an input with `grid = Any`
- **THEN** the component accepts Scalar, FourBox, or Hemispheric inputs
- **AND** the component handles grid-specific logic internally

---

### Requirement: Coupler Grid Validation

The system SHALL validate grid compatibility between connected components at model build time.

#### Scenario: Compatible grids with same type

- **WHEN** a producer outputs `Emissions|CO2` as Scalar
- **AND** a consumer requires `Emissions|CO2` as Scalar
- **THEN** the model builds successfully with no transformation

#### Scenario: Incompatible grids fail build

- **WHEN** a producer outputs `Emissions|CO2` as Scalar
- **AND** a consumer requires `Emissions|CO2` as FourBox
- **THEN** the build fails with a clear error message
- **AND** the error identifies the producer component, consumer component, and variable name
- **AND** the error suggests possible resolutions

---

### Requirement: Coupler Grid Auto-Transform

The system SHALL automatically insert grid transformation nodes when a finer grid connects to a coarser grid.

#### Scenario: FourBox to Scalar auto-aggregation

- **WHEN** a producer outputs `Temperature` as FourBox
- **AND** a consumer requires `Temperature` as Scalar
- **THEN** the coupler automatically inserts an aggregation transform
- **AND** the transform uses the grid's configured weights

#### Scenario: FourBox to Hemispheric auto-transform

- **WHEN** a producer outputs `Temperature` as FourBox
- **AND** a consumer requires `Temperature` as Hemispheric
- **THEN** the coupler automatically inserts a hemispheric aggregation transform

---

### Requirement: Typed Input Accessors (Rust)

The system SHALL provide a derive macro that generates typed input structs from component definitions.

Generated input structs MUST:

- Have fields for each declared input and state variable
- Use `TimeseriesWindow` type for scalar fields
- Use `GridTimeseriesWindow` type for grid fields
- Be lifetime-parameterised to avoid data copying

#### Scenario: Macro generates typed inputs

- **WHEN** a component uses `#[derive(Component)]` with input declarations
- **THEN** an `{ComponentName}Inputs` struct is generated
- **AND** each input variable becomes a field with snake_case naming
- **AND** grid inputs have array-based access methods

#### Scenario: Compile-time validation of variable access

- **WHEN** component code accesses `inputs.emissions_co2`
- **AND** no input named "Emissions|CO2" was declared
- **THEN** compilation fails with an error pointing to the invalid field access

---

### Requirement: Typed Output Accessors (Rust)

The system SHALL provide generated output structs that enforce returning all declared outputs.

Generated output structs MUST:

- Have fields for each declared output and state variable
- Use `FloatValue` for Scalar outputs
- Use typed slice wrappers for grid outputs (e.g., `FourBoxSlice` for FourBox)
- Require all fields to be set

#### Scenario: Macro generates typed outputs

- **WHEN** a component uses `#[derive(Component)]` with output declarations
- **THEN** a `{ComponentName}Outputs` struct is generated
- **AND** the struct implements `Into<OutputState>`

#### Scenario: Missing output causes compile error

- **WHEN** a component's solve method constructs outputs
- **AND** a declared output field is not set
- **THEN** compilation fails due to missing struct field

---

### Requirement: Typed Grid Output Slices

The system SHALL provide zero-cost wrapper types for grid outputs that enforce type-safe region access.

Typed slices MUST:

- Use `#[repr(transparent)]` for zero-cost representation
- Provide region-enum-based accessors instead of raw indices
- Support builder pattern for ergonomic construction
- Be provided for each grid type (FourBoxSlice, HemisphericSlice)

#### Scenario: FourBox output with typed slice

- **WHEN** a component returns a FourBox output
- **THEN** it uses `FourBoxSlice` type
- **AND** values are set using `FourBoxRegion` enum variants
- **AND** IDE autocomplete shows available regions

#### Scenario: Builder pattern construction

- **WHEN** a component constructs a `FourBoxSlice`
- **THEN** it can chain `.with(region, value)` calls
- **AND** unset regions default to NaN

#### Scenario: Type-safe region access

- **WHEN** component code calls `slice.set(FourBoxRegion::NorthernOcean, 1.5)`
- **THEN** the value is stored at the correct array index
- **AND** using an invalid region variant causes a compile error

---

### Requirement: Python Typed Inputs

The system SHALL generate typed dataclasses for Python component inputs.

Generated Python input classes MUST:

- Have attributes for each declared input and state variable
- Provide `TimeseriesWindow`-like access (current, previous, last_n as numpy views)
- Support grid-aware access for grid inputs

#### Scenario: Python component receives typed inputs

- **WHEN** a Python component's `solve()` method is called
- **THEN** it receives an instance of `{ComponentName}.Inputs`
- **AND** scalar attribute access like `inputs.emissions_co2.current` returns a float
- **AND** grid attribute access like `inputs.temperature.current` returns a numpy array

#### Scenario: Python grid region access

- **WHEN** a Python component accesses `inputs.temperature.region(FourBoxRegion.NorthernOcean)`
- **THEN** it receives the scalar value for that specific region

---

### Requirement: Python Typed Outputs

The system SHALL generate typed dataclasses for Python component outputs.

#### Scenario: Python component returns typed outputs

- **WHEN** a Python component's `solve()` method returns
- **THEN** it returns an instance of `{ComponentName}.Outputs`
- **AND** constructing the outputs validates all required fields are provided

#### Scenario: Missing Python output raises error

- **WHEN** a Python component constructs outputs without setting a required field
- **THEN** a validation error is raised with a clear message identifying the missing output

---

### Requirement: Derive Macro in Separate Crate

The system SHALL provide the `#[derive(Component)]` macro in a dedicated `rscm-macros` crate.

#### Scenario: Macro crate is a workspace member

- **WHEN** the workspace is built
- **THEN** `rscm-macros` compiles as a proc-macro crate
- **AND** `rscm-core` re-exports the macro for convenience

#### Scenario: Components use macro via rscm-core

- **WHEN** a component imports from `rscm_core`
- **THEN** the `Component` derive macro is available
- **AND** no direct dependency on `rscm-macros` is required
