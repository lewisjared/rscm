## ADDED Requirements

### Requirement: Variable Definition

The system SHALL provide a `VariableDefinition` struct that captures intrinsic metadata for climate model variables.

A `VariableDefinition` SHALL include:
- `name`: Unique identifier using pipe-delimited namespacing (e.g., "Atmospheric Concentration|CO2")
- `unit`: Canonical physical unit string (e.g., "ppm", "W / m^2", "GtC / yr") - for documentation and future unit conversion support
- `time_convention`: When the value applies within a timestep (StartOfYear, MidYear, or Instantaneous)
- `description`: Human-readable description of the variable

**Note on canonical unit**: The `unit` field represents the canonical/preferred unit for the variable. Components may declare different units in their `RequirementDefinition` (e.g., "MtC / yr" instead of "GtC / yr"). Unit validation compares between connected components, not against this canonical unit. The canonical unit serves for:
- Documentation and variable discovery
- Future automatic unit conversion in the coupler

#### Scenario: Define a concentration variable

- **GIVEN** a climate model developer defining CO2 concentration
- **WHEN** they create a VariableDefinition with name "Atmospheric Concentration|CO2", unit "ppm", time_convention StartOfYear, and description "Atmospheric CO2 mole fraction"
- **THEN** the definition is valid and can be used in component requirements

#### Scenario: Define an emissions variable with mid-year convention

- **GIVEN** a climate model developer defining CO2 emissions
- **WHEN** they create a VariableDefinition with name "Emissions|CO2", unit "GtC / yr", time_convention MidYear
- **THEN** the definition captures that emissions are centred on mid-year

---

### Requirement: Time Convention

The system SHALL provide a `TimeConvention` enum to specify when variable values apply within a timestep.

The enum SHALL have three variants:
- `StartOfYear`: Value applies at the start of the year (January 1)
- `MidYear`: Value applies at mid-year (July 1), typically used for emissions
- `Instantaneous`: Value applies at the exact time point with no temporal averaging

#### Scenario: Emissions use mid-year convention

- **GIVEN** MAGICC-style emissions data
- **WHEN** the time convention is queried
- **THEN** it returns MidYear, indicating emissions are centred on July 1

#### Scenario: Concentrations use start-of-year convention

- **GIVEN** atmospheric concentration data
- **WHEN** the time convention is queried
- **THEN** it returns StartOfYear, indicating values apply at January 1

---

### Requirement: Preindustrial Value

The system SHALL provide a `PreindustrialValue` enum to represent preindustrial reference values at different spatial resolutions.

The enum SHALL have three variants:
- `Scalar(f64)`: Global scalar value
- `FourBox([f64; 4])`: Regional values for NorthernOcean, NorthernLand, SouthernOcean, SouthernLand
- `Hemispheric([f64; 2])`: Values for Northern and Southern hemispheres

The enum SHALL provide convenience methods:
- `as_scalar()`: Returns Some(f64) if Scalar, None otherwise
- `as_four_box()`: Returns Some([f64; 4]) if FourBox, None otherwise
- `to_scalar()`: Aggregates any variant to a global scalar using area weights

#### Scenario: Scalar preindustrial for CO2

- **GIVEN** a CO2 concentration timeseries
- **WHEN** preindustrial is set to Scalar(278.0)
- **THEN** `as_scalar()` returns Some(278.0)
- **AND** `to_scalar()` returns 278.0

#### Scenario: Four-box preindustrial for aerosol forcing

- **GIVEN** an aerosol forcing timeseries with regional preindustrial baselines
- **WHEN** preindustrial is set to FourBox([0.1, 0.2, 0.05, 0.15])
- **THEN** `as_four_box()` returns Some([0.1, 0.2, 0.05, 0.15])
- **AND** `to_scalar()` returns the area-weighted global mean

---

### Requirement: Timeseries Preindustrial Metadata

The `TimeseriesItem` struct SHALL support an optional preindustrial reference value as metadata.

TimeseriesItem SHALL include:
- `preindustrial: Option<PreindustrialValue>`: Optional preindustrial reference for this timeseries

The preindustrial value SHALL be:
- Set when loading scenario data or configuring the model
- Accessible via `InputState` during component execution
- Independent of the variable definition (configuration-dependent, not intrinsic)

#### Scenario: Set preindustrial when adding timeseries

- **GIVEN** a TimeseriesCollection being populated with CO2 concentration data
- **WHEN** the timeseries is added with preindustrial Scalar(278.0)
- **THEN** the TimeseriesItem stores the preindustrial value

#### Scenario: Timeseries without preindustrial

- **GIVEN** a derived quantity like effective radiative forcing
- **WHEN** the timeseries is added without a preindustrial value
- **THEN** the TimeseriesItem has preindustrial set to None

#### Scenario: Access preindustrial via InputState

- **GIVEN** a component solving with InputState containing CO2 concentration
- **WHEN** the component calls `input_state.get_preindustrial("Atmospheric Concentration|CO2")`
- **THEN** it returns Some(PreindustrialValue::Scalar(278.0))

---

### Requirement: Variable Registry

The system SHALL provide a `VariableRegistry` that collects and provides lookup for registered variable definitions.

The registry SHALL support:
- Looking up a variable definition by name
- Listing all registered variables
- Checking if a variable name is registered
- Both compile-time (Rust) and runtime (Python) registration

#### Scenario: Look up registered variable

- **GIVEN** a registry containing the CO2 concentration variable
- **WHEN** the user looks up "Atmospheric Concentration|CO2"
- **THEN** the registry returns the corresponding VariableDefinition

#### Scenario: Look up unregistered variable

- **GIVEN** a registry that does not contain "Unknown|Variable"
- **WHEN** the user looks up "Unknown|Variable"
- **THEN** the registry returns None

#### Scenario: List all registered variables

- **GIVEN** a registry containing CO2 concentration and emissions variables
- **WHEN** the user lists all variables
- **THEN** the registry returns an iterator over both definitions

---

### Requirement: Static Variable Registration (Rust)

The system SHALL provide a `define_variable!` macro for declaring variables at compile time with automatic registry registration using the `inventory` crate.

The macro SHALL:
- Create a static `VariableDefinition` constant
- Register the variable in the global registry via inventory
- Support all VariableDefinition fields

#### Scenario: Define variable using macro

- **GIVEN** a Rust developer creating a new variable
- **WHEN** they use `define_variable!(CO2_CONCENTRATION, name = "Atmospheric Concentration|CO2", unit = "ppm", time_convention = TimeConvention::StartOfYear, description = "CO2 concentration")`
- **THEN** a static `CO2_CONCENTRATION` constant is created
- **AND** the variable is automatically registered in the global registry at startup

#### Scenario: Reference macro-defined variable

- **GIVEN** a variable defined with `define_variable!`
- **WHEN** a component references `&CO2_CONCENTRATION`
- **THEN** the reference resolves to the static VariableDefinition

---

### Requirement: Runtime Variable Registration (Python)

The system SHALL allow variables to be registered at runtime from Python.

Runtime registration SHALL:
- Accept a VariableDefinition instance
- Add the variable to the registry
- Fail if a variable with the same name already exists
- Be thread-safe for concurrent access

#### Scenario: Register variable from Python

- **GIVEN** a Python script creating a custom variable
- **WHEN** it calls `rscm.register_variable(VariableDefinition(name="Custom|Variable", unit="kg", ...))`
- **THEN** the variable is added to the registry
- **AND** subsequent lookups for "Custom|Variable" succeed

#### Scenario: Duplicate registration fails

- **GIVEN** a variable "Atmospheric Concentration|CO2" already registered (from Rust)
- **WHEN** Python attempts to register another variable with the same name
- **THEN** registration fails with a DuplicateVariableError
- **AND** the original definition is preserved

#### Scenario: Python-registered variable used in component

- **GIVEN** a variable registered from Python
- **WHEN** a Python-defined component declares it as an input or output
- **THEN** model building succeeds
- **AND** validation uses the Python-registered variable's metadata

---

### Requirement: Requirement Definition with Variable Name and Unit

The `RequirementDefinition` struct SHALL reference variables by name and declare the component's expected/produced unit.

RequirementDefinition SHALL include:
- `variable_name`: Name of the registered variable
- `unit`: The unit this component expects (for inputs) or produces (for outputs)
- `requirement_type`: Whether this is Input, Output, or State
- `grid_type`: Spatial resolution (Scalar, FourBox, Hemispheric)

The struct SHALL provide accessor methods:
- `name()`: Returns the variable name
- `unit()`: Returns the component's declared unit
- `time_convention()`: Returns the time convention from the registry (intrinsic to variable)

#### Scenario: Create scalar input requirement

- **GIVEN** a registered CO2 concentration variable
- **WHEN** a component creates `RequirementDefinition::scalar_input("Atmospheric Concentration|CO2", "ppm")`
- **THEN** the requirement stores the variable name and unit
- **AND** requirement_type is Input
- **AND** grid_type is Scalar

#### Scenario: Create four-box output requirement

- **GIVEN** a registered temperature variable
- **WHEN** a component creates `RequirementDefinition::four_box_output("Surface Temperature", "K")`
- **THEN** the requirement stores the variable name and unit
- **AND** requirement_type is Output
- **AND** grid_type is FourBox

#### Scenario: Component declares different unit than registry canonical

- **GIVEN** a registry with "Emissions|CO2" having canonical unit "GtC / yr"
- **WHEN** a component creates `RequirementDefinition::scalar_output("Emissions|CO2", "MtC / yr")`
- **THEN** the requirement stores "MtC / yr" as the unit
- **AND** the registry canonical unit is not affected

---

### Requirement: Variable Existence Validation

The `ModelBuilder` SHALL validate that all referenced variables exist in the registry.

Validation SHALL:
- Check each RequirementDefinition's variable_name against the registry
- Fail the build if any variable is not registered
- Provide clear error messages identifying the missing variable and component

#### Scenario: All variables registered

- **GIVEN** component A outputs "Atmospheric Concentration|CO2" (registered)
- **AND** component B inputs "Atmospheric Concentration|CO2" (registered)
- **WHEN** the model is built
- **THEN** validation passes

#### Scenario: Missing variable fails build

- **GIVEN** component A outputs "Unregistered|Variable"
- **WHEN** the model is built
- **THEN** validation fails with UnregisteredVariableError
- **AND** the error identifies the component and variable name

---

### Requirement: Unit Compatibility Validation

The `ModelBuilder` SHALL validate unit compatibility when connecting component inputs to outputs.

Validation SHALL:
- Compare unit strings between connected component declarations (output unit vs input unit)
- NOT compare against the registry's canonical unit (registry unit is for documentation only)
- Fail the build if units do not match exactly (future: auto-convert compatible units)
- Provide clear error messages identifying the mismatched components and variables

#### Scenario: Compatible units pass validation

- **GIVEN** component A outputs "Atmospheric Concentration|CO2" with unit "ppm"
- **AND** component B inputs "Atmospheric Concentration|CO2" with unit "ppm"
- **WHEN** the model is built
- **THEN** validation passes without errors

#### Scenario: Incompatible units fail validation

- **GIVEN** component A outputs "Emissions|CO2" with unit "GtC / yr"
- **AND** component B inputs "Emissions|CO2" with unit "MtC / yr"
- **WHEN** the model is built
- **THEN** validation fails with an error indicating the unit mismatch
- **AND** the error identifies both components and the variable name

#### Scenario: Registry canonical unit does not affect validation

- **GIVEN** the registry defines "Emissions|CO2" with canonical unit "GtC / yr"
- **AND** component A outputs "Emissions|CO2" with unit "MtC / yr"
- **AND** component B inputs "Emissions|CO2" with unit "MtC / yr"
- **WHEN** the model is built
- **THEN** validation passes (both components agree on unit, regardless of registry canonical)

---

### Requirement: Time Convention Compatibility Validation

The `ModelBuilder` SHALL validate time convention compatibility when connecting component inputs to outputs.

Validation SHALL:
- Look up time conventions from the registry for each variable
- Compare time conventions between connected requirements
- Fail the build if time conventions do not match
- Provide clear error messages identifying the mismatch

#### Scenario: Matching time conventions pass validation

- **GIVEN** component A outputs concentration with StartOfYear convention
- **AND** component B inputs concentration with StartOfYear convention
- **WHEN** the model is built
- **THEN** validation passes without errors

#### Scenario: Mismatched time conventions fail validation

- **GIVEN** component A outputs emissions with MidYear convention
- **AND** component B inputs emissions expecting StartOfYear convention
- **WHEN** the model is built
- **THEN** validation fails with an error about the time convention mismatch
- **AND** the error identifies both components and the variable name

---

### Requirement: Preindustrial Value Access

The system SHALL provide access to preindustrial reference values via InputState during component execution.

InputState SHALL provide methods:
- `get_preindustrial(name: &str) -> Option<&PreindustrialValue>`: Returns the preindustrial value if set
- `get_preindustrial_scalar(name: &str) -> Option<f64>`: Convenience method returning scalar or aggregated value

#### Scenario: Access scalar preindustrial

- **GIVEN** a timeseries "Atmospheric Concentration|CO2" with preindustrial Scalar(278.0)
- **WHEN** a component calls `input_state.get_preindustrial_scalar("Atmospheric Concentration|CO2")`
- **THEN** it returns Some(278.0)

#### Scenario: Access grid preindustrial

- **GIVEN** a timeseries "Aerosol ERF" with preindustrial FourBox([0.1, 0.2, 0.05, 0.15])
- **WHEN** a component calls `input_state.get_preindustrial("Aerosol ERF")`
- **THEN** it returns Some(PreindustrialValue::FourBox([0.1, 0.2, 0.05, 0.15]))

#### Scenario: No preindustrial set

- **GIVEN** a timeseries "Effective RF|Total" without preindustrial metadata
- **WHEN** a component calls `input_state.get_preindustrial("Effective RF|Total")`
- **THEN** it returns None

---

### Requirement: Python Bindings for Variable System

The variable registration system SHALL be accessible from Python via PyO3 bindings.

Bindings SHALL include:
- `TimeConvention` enum with all variants
- `PreindustrialValue` enum with all variants and convenience methods
- `VariableDefinition` class (constructible from Python)
- `register_variable()` function for runtime registration
- `get_variable()` function for registry lookup
- `list_variables()` function for registry enumeration
- Timeseries creation methods accepting optional preindustrial values

#### Scenario: Create variable definition from Python

- **GIVEN** a Python script
- **WHEN** it creates `rscm.VariableDefinition(name="Custom|Var", unit="kg", time_convention=rscm.TimeConvention.StartOfYear, description="Custom variable")`
- **THEN** a valid VariableDefinition is created

#### Scenario: Register and retrieve variable from Python

- **GIVEN** a Python script that registers a custom variable
- **WHEN** it calls `rscm.register_variable(var)` then `rscm.get_variable("Custom|Var")`
- **THEN** the retrieved variable matches the registered one

#### Scenario: Set preindustrial when creating timeseries from Python

- **GIVEN** a Python script creating a TimeseriesCollection
- **WHEN** it adds a timeseries with `preindustrial=278.0`
- **THEN** the timeseries has PreindustrialValue::Scalar(278.0) set

#### Scenario: List variables from Python

- **GIVEN** a registry with registered variables
- **WHEN** the user calls `rscm.list_variables()`
- **THEN** it returns a list of all registered variable names or definitions
