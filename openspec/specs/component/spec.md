# component Specification

## Purpose

Defines the `Component` trait and `RequirementDefinition` type that form the fundamental building blocks for climate model components. Covers how components declare their inputs, outputs, and state variables, and the contract for the `solve()` method.

## Requirements

### Requirement: Component Trait

The system SHALL provide a `Component` trait that defines the interface for all climate model components.

#### Scenario: Component declares requirements

- **WHEN** implementing `Component::definitions()`
- **THEN** it MUST return a `Vec<RequirementDefinition>` listing all inputs, outputs, and state variables
- **AND** each definition MUST specify name, unit, requirement type, and grid type

#### Scenario: Component solve method

- **WHEN** calling `component.solve(t_current, t_next, input_state)`
- **THEN** it MUST compute outputs for the timestep from `t_current` to `t_next`
- **AND** return an `OutputState` containing values for all declared outputs
- **AND** return an error if computation fails

#### Scenario: Component is serializable

- **WHEN** a struct implements `Component`
- **THEN** it MUST use `#[typetag::serde]` for trait object serialization
- **AND** be serializable to JSON and TOML formats
- **AND** be deserializable back to the concrete type

#### Scenario: Component is thread-safe

- **WHEN** a struct implements `Component`
- **THEN** it MUST implement `Send + Sync`
- **AND** be safe to use across thread boundaries

### Requirement: RequirementType Enum

The system SHALL provide a `RequirementType` enum to classify component variables.

#### Scenario: Input requirement type

- **WHEN** a variable is declared as `RequirementType::Input`
- **THEN** it represents a read-only variable from external source or other component
- **AND** it appears in `component.inputs()` but NOT in `component.outputs()`

#### Scenario: Output requirement type

- **WHEN** a variable is declared as `RequirementType::Output`
- **THEN** it represents a write-only variable produced each timestep
- **AND** it appears in `component.outputs()` but NOT in `component.inputs()`

#### Scenario: State requirement type

- **WHEN** a variable is declared as `RequirementType::State`
- **THEN** it represents a variable that reads its previous value and writes a new value
- **AND** it appears in BOTH `component.inputs()` AND `component.outputs()`
- **AND** it requires an initial value at model build time

#### Scenario: State variable without initial value

- **WHEN** a model is built with a component that has a State variable
- **AND** no initial value is provided for that variable
- **THEN** the build MUST fail with a descriptive error identifying the missing variable

### Requirement: GridType Enum

The system SHALL provide a `GridType` enum to specify spatial resolution of variables.

#### Scenario: Scalar grid type (default)

- **WHEN** a variable has `GridType::Scalar`
- **THEN** it represents a single global value (non-spatial)
- **AND** this MUST be the default when grid type is not specified

#### Scenario: FourBox grid type

- **WHEN** a variable has `GridType::FourBox`
- **THEN** it represents 4 regional values (NorthernOcean, NorthernLand, SouthernOcean, SouthernLand)

#### Scenario: Hemispheric grid type

- **WHEN** a variable has `GridType::Hemispheric`
- **THEN** it represents 2 regional values (Northern, Southern)

### Requirement: RequirementDefinition Type

The system SHALL provide a `RequirementDefinition` type to declare component I/O.

#### Scenario: Create scalar requirement

- **WHEN** calling `RequirementDefinition::new(name, unit, requirement_type)`
- **THEN** it MUST create a definition with `GridType::Scalar`
- **AND** store the provided name, unit, and requirement type

#### Scenario: Create grid requirement

- **WHEN** calling `RequirementDefinition::with_grid(name, unit, requirement_type, grid_type)`
- **THEN** it MUST create a definition with the specified grid type

#### Scenario: Convenience constructors

- **WHEN** using `RequirementDefinition::scalar_input(name, unit)`
- **THEN** it MUST create a scalar input requirement
- **AND** similar convenience constructors MUST exist for:
  - `scalar_output`, `scalar_state`
  - `four_box_input`, `four_box_output`, `four_box_state`
  - `hemispheric_input`, `hemispheric_output`, `hemispheric_state`

#### Scenario: Check if spatial

- **WHEN** calling `definition.is_spatial()`
- **THEN** it MUST return `true` for FourBox and Hemispheric grid types
- **AND** return `false` for Scalar grid type

### Requirement: Derived Input/Output Lists

The system SHALL derive input and output lists from the definitions.

#### Scenario: inputs() returns Input and State requirements

- **WHEN** calling `component.inputs()`
- **THEN** it MUST return all definitions with `RequirementType::Input`
- **AND** all definitions with `RequirementType::State`
- **AND** NOT include `RequirementType::Output` definitions

#### Scenario: outputs() returns Output and State requirements

- **WHEN** calling `component.outputs()`
- **THEN** it MUST return all definitions with `RequirementType::Output`
- **AND** all definitions with `RequirementType::State`
- **AND** NOT include `RequirementType::Input` definitions

#### Scenario: input_names() and output_names() return name strings

- **WHEN** calling `component.input_names()` or `component.output_names()`
- **THEN** it MUST return a `Vec<String>` of just the variable names

### Requirement: Variable Naming Convention

The system SHALL support hierarchical variable naming using pipe separators.

#### Scenario: Namespaced variable names

- **WHEN** declaring a variable name
- **THEN** it MAY use `|` to create hierarchical namespaces
- **AND** examples include "Emissions|CO2", "Atmospheric Concentration|CO2", "Surface Temperature|Ocean"

#### Scenario: Unique output names within model

- **WHEN** components are combined in a model
- **THEN** no two components MAY produce the same output variable name
- **AND** the model builder MUST validate uniqueness
