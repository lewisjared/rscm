# Component Specification

## Purpose

Defines the `Component` trait and `VariableDefinition` type that form the fundamental building blocks for climate model components. Covers how components declare their inputs, outputs, and state variables, and the contract for the `solve()` method.

## Requirements

### Requirement: Component Trait

The system SHALL provide a `Component` trait that defines the interface for all climate model components.

#### Scenario: Component declares variables

- **WHEN** implementing `Component::definitions()`
- **THEN** it MUST return a `Vec<VariableDefinition>` listing all inputs, outputs, and state variables
- **AND** each definition MUST specify name, unit, role, and grid

#### Scenario: Component solve method

- **WHEN** calling `component.solve(t_current, t_next, input_state)`
- **THEN** it MUST compute outputs for the step from `t_current` to `t_next`
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

### Requirement: VariableRole Enum

The system SHALL provide a `VariableRole` enum to classify component variables.

#### Scenario: Input role

- **WHEN** a variable is declared as `VariableRole::Input`
- **THEN** it represents a read-only variable from prescribed data or another component
- **AND** it appears in `component.inputs()` but NOT in `component.outputs()`

#### Scenario: Output role

- **WHEN** a variable is declared as `VariableRole::Output`
- **THEN** it represents a write-only variable produced each step
- **AND** it appears in `component.outputs()` but NOT in `component.inputs()`

#### Scenario: State role

- **WHEN** a variable is declared as `VariableRole::State`
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

### Requirement: VariableDefinition Type

The system SHALL provide a `VariableDefinition` type to declare component I/O.

#### Scenario: Create scalar definition

- **WHEN** calling `VariableDefinition::new(name, unit, role)`
- **THEN** it MUST create a definition with `GridType::Scalar`
- **AND** store the provided name, unit, and role

#### Scenario: Create grid definition

- **WHEN** calling `VariableDefinition::with_grid(name, unit, role, grid)`
- **THEN** it MUST create a definition with the specified grid

#### Scenario: Convenience constructors

- **WHEN** using `VariableDefinition::scalar_input(name, unit)`
- **THEN** it MUST create a scalar input definition
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

#### Scenario: inputs() returns Input and State definitions

- **WHEN** calling `component.inputs()`
- **THEN** it MUST return all definitions with `VariableRole::Input`
- **AND** all definitions with `VariableRole::State`
- **AND** NOT include `VariableRole::Output` definitions

#### Scenario: outputs() returns Output and State definitions

- **WHEN** calling `component.outputs()`
- **THEN** it MUST return all definitions with `VariableRole::Output`
- **AND** all definitions with `VariableRole::State`
- **AND** NOT include `VariableRole::Input` definitions

#### Scenario: input_names() and output_names() return name strings

- **WHEN** calling `component.input_names()` or `component.output_names()`
- **THEN** it MUST return a `Vec<String>` of just the variable names

### Requirement: Variable Naming Convention

The system SHALL support hierarchical variable naming using pipe separators.

#### Scenario: Namespaced variable names

- **WHEN** declaring a variable name
- **THEN** it MAY use `|` to create hierarchical namespaces
- **AND** examples include "Emissions|CO2", "Atmospheric Concentrations|CO2", "Ocean Temperature|Layer 1"

#### Scenario: Unique output names within model

- **WHEN** components are combined in a model
- **THEN** no two components MAY produce the same output variable name
- **AND** the model builder MUST validate uniqueness
