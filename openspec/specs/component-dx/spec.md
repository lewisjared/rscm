# component-dx Specification

## Purpose

Defines the `ComponentIO` derive macro and related code generation for type-safe component I/O declarations. Covers compile-time validation of variable access, spatial grid type annotations, and automatic grid transformations in the coupler. Runtime behavior of generated types is specified in `component-state`.

## Requirements

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

The system SHALL provide a derive macro that generates typed input structs from struct-level attribute declarations.

Generated input structs MUST:

- Have fields for each declared input and state variable
- Use `TimeseriesWindow` type for scalar fields (see `component-state` for behavior)
- Use `GridTimeseriesWindow` type for grid fields (see `component-state` for behavior)
- Be lifetime-parameterised to avoid data copying

The macro MUST use struct-level attributes (`#[inputs(...)]`, `#[states(...)]`) rather than phantom fields to avoid:

- Placeholder `()` fields that store nothing
- `#[serde(skip)]` annotations on marker fields
- Underscore-prefixed field names to suppress unused warnings
- Explicit initialization of phantom fields in constructors

#### Scenario: Macro generates typed inputs

- **WHEN** a component uses `#[derive(ComponentIO)]` with `#[inputs(...)]` and `#[states(...)]` attributes
- **THEN** an `{ComponentName}Inputs` struct is generated
- **AND** each input variable becomes a field with snake_case naming
- **AND** the component struct contains only actual parameters (no phantom fields)
- **AND** grid inputs have array-based access methods

#### Scenario: Compile-time validation of variable access

- **WHEN** component code accesses `inputs.emissions_co2`
- **AND** no input named "Emissions|CO2" was declared in `#[inputs(...)]`
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

- **WHEN** a component uses `#[derive(ComponentIO)]` with `#[outputs(...)]` and `#[states(...)]` attributes
- **THEN** a `{ComponentName}Outputs` struct is generated
- **AND** the struct implements `Into<OutputState>`

#### Scenario: Missing output causes compile error

- **WHEN** a component's solve method constructs outputs
- **AND** a declared output field is not set
- **THEN** compilation fails due to missing struct field

---

### Requirement: Typed Grid Output Slices

The `ComponentIO` macro SHALL generate code that uses typed slice wrappers for grid outputs.

The generated `Into<OutputState>` implementation MUST:

- Use `FourBoxSlice` for FourBox grid outputs
- Use `HemisphericSlice` for Hemispheric grid outputs
- Convert slices to the appropriate `StateValue` variant

**Note:** The runtime behavior of `FourBoxSlice` and `HemisphericSlice` types is specified in `component-state`.

#### Scenario: Macro generates FourBoxSlice usage

- **WHEN** a component declares `#[outputs(temp { name = "Temperature", unit = "K", grid = "FourBox" })]`
- **THEN** the generated `{ComponentName}Outputs` struct has a `temp: FourBoxSlice` field
- **AND** the `Into<OutputState>` implementation converts it correctly

#### Scenario: Macro generates HemisphericSlice usage

- **WHEN** a component declares `#[outputs(precip { name = "Precipitation", unit = "mm", grid = "Hemispheric" })]`
- **THEN** the generated `{ComponentName}Outputs` struct has a `precip: HemisphericSlice` field

---

### Requirement: Derive Macro in Separate Crate

The system SHALL provide the `#[derive(ComponentIO)]` macro in a dedicated `rscm-macros` crate.

#### Scenario: Macro crate is a workspace member

- **WHEN** the workspace is built
- **THEN** `rscm-macros` compiles as a proc-macro crate
- **AND** `rscm-core` re-exports the macro for convenience

#### Scenario: Components use macro via rscm-core

- **WHEN** a component imports from `rscm_core`
- **THEN** the `ComponentIO` derive macro is available
- **AND** no direct dependency on `rscm-macros` is required
