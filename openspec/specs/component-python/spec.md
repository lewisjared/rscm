# component-python Specification

## Purpose

Defines the Python interface for RSCM components, including protocols for Python-defined components, wrappers for Rust components, model building, and typed input/output access. Enables bidirectional interoperability between Python and Rust components.

## Requirements

### Requirement: Component Protocol

The system SHALL provide a `Component` Protocol that defines the Python interface for all components.

#### Scenario: Component protocol methods

- **WHEN** a class implements the `Component` protocol
- **THEN** it MUST implement `definitions() -> list[RequirementDefinition]`
- **AND** implement `solve(t_current: float, t_next: float, collection: TimeseriesCollection) -> dict[str, StateValue]`

### Requirement: CustomComponent Protocol

The system SHALL provide a `CustomComponent` Protocol for user-defined Python components.

#### Scenario: Dict-based interface (legacy)

- **WHEN** a Python component uses the dict-based interface
- **THEN** `solve()` receives `input_state: dict[str, float]`
- **AND** returns `dict[str, float]` with output values

#### Scenario: Typed interface

- **WHEN** a Python component uses the typed interface
- **THEN** it MUST have a `_component_inputs` attribute (marker for typed detection)
- **AND** have an inner `Inputs` class with `from_input_state(windows: dict)` class method
- **AND** `solve()` receives typed inputs and returns typed outputs
- **AND** outputs MUST have a `to_dict()` method for conversion

### Requirement: PythonComponent Wrapper

The system SHALL provide a `PythonComponent` class that wraps Python-defined components for use in Rust models.

#### Scenario: Build Python component

- **WHEN** calling `PythonComponent.build(component)`
- **THEN** it MUST wrap the `CustomComponent` instance
- **AND** expose it as a `Component` usable by `ModelBuilder`

#### Scenario: Solve delegates to Python

- **WHEN** `PythonComponent.solve()` is called
- **THEN** it MUST detect if the wrapped component is typed or legacy
- **AND** convert `InputState` to appropriate Python format
- **AND** call the Python component's `solve()` method
- **AND** convert the result back to `OutputState`

#### Scenario: Serialization support

- **WHEN** serializing a `PythonComponent`
- **THEN** it MUST call the Python component's `to_json()` method
- **AND** deserialize via `Component.from_json()` class method

### Requirement: RustComponent Wrapper

The system SHALL provide a `RustComponent` class that exposes Rust-defined components to Python.

#### Scenario: RustComponent from builder

- **WHEN** a `ComponentBuilder.build()` is called
- **THEN** it MUST return a `RustComponent` instance
- **AND** the component is usable in Python `ModelBuilder`

#### Scenario: RustComponent methods

- **WHEN** using a `RustComponent` in Python
- **THEN** `definitions()` MUST return `list[RequirementDefinition]`
- **AND** `solve()` MUST accept `TimeseriesCollection` and return `dict[str, StateValue]`

### Requirement: ComponentBuilder Protocol

The system SHALL provide a `ComponentBuilder` Protocol for constructing Rust components from Python.

#### Scenario: Create builder from parameters

- **WHEN** calling `ComponentBuilder.from_parameters(parameters: dict)`
- **THEN** it MUST deserialize the dict into Rust parameter struct
- **AND** return a builder instance

#### Scenario: Build concrete component

- **WHEN** calling `builder.build()`
- **THEN** it MUST return a `RustComponent` instance
- **AND** the component is ready for use in models

### Requirement: Python RequirementDefinition

The system SHALL expose `RequirementDefinition`, `RequirementType`, and `GridType` to Python.

#### Scenario: Create RequirementDefinition

- **WHEN** calling `RequirementDefinition(name, units, requirement_type, grid_type=GridType.Scalar)`
- **THEN** it MUST create a definition with the specified values
- **AND** `grid_type` MUST default to `GridType.Scalar`

#### Scenario: RequirementType enum values

- **WHEN** using `RequirementType` in Python
- **THEN** it MUST have `Input`, `Output`, `State`, and `EmptyLink` variants

#### Scenario: GridType enum values

- **WHEN** using `GridType` in Python
- **THEN** it MUST have `Scalar`, `FourBox`, and `Hemispheric` variants

### Requirement: Python TimeseriesWindow Types

The system SHALL provide Python window types for typed component input access.

#### Scenario: TimeseriesWindow for scalar access

- **WHEN** a typed Python component receives inputs
- **THEN** scalar variables MUST be accessible via `TimeseriesWindow`
- **AND** `window.current` returns the current value
- **AND** `window.previous` returns the previous value (raises if at first timestep)
- **AND** `window.at_offset(n)` returns value at relative offset
- **AND** `window.last_n(n)` returns numpy array of last n values

#### Scenario: FourBoxTimeseriesWindow for grid access

- **WHEN** a typed Python component receives FourBox inputs
- **THEN** the variable MUST be accessible via `FourBoxTimeseriesWindow`
- **AND** `window.current` returns a `FourBoxSlice`
- **AND** `window.previous` returns a `FourBoxSlice`
- **AND** `window.region(index)` returns a `TimeseriesWindow` for that region

#### Scenario: HemisphericTimeseriesWindow for hemispheric access

- **WHEN** a typed Python component receives Hemispheric inputs
- **THEN** the variable MUST be accessible via `HemisphericTimeseriesWindow`
- **AND** `window.current` returns a `HemisphericSlice`
- **AND** `window.region(index)` returns a `TimeseriesWindow` for that region

### Requirement: Python Component Base Class

The system SHALL provide a `Component` base class with metaclass-based code generation for typed Python components.

#### Scenario: Declare inputs with Input descriptor

- **WHEN** a class attribute is set to `Input(name, unit, grid="Scalar")`
- **THEN** it MUST be collected as a component input
- **AND** the field name becomes the accessor name on `Inputs` class
- **AND** `grid` MUST be a `Literal["Scalar", "FourBox", "Hemispheric"]` defaulting to "Scalar"

#### Scenario: Declare outputs with Output descriptor

- **WHEN** a class attribute is set to `Output(name, unit, grid="Scalar")`
- **THEN** it MUST be collected as a component output
- **AND** the field name becomes the required field on `Outputs` class
- **AND** `grid` MUST be a `Literal["Scalar", "FourBox", "Hemispheric"]` defaulting to "Scalar"

#### Scenario: Declare state with State descriptor

- **WHEN** a class attribute is set to `State(name, unit, grid="Scalar")`
- **THEN** it MUST appear in BOTH `Inputs` and `Outputs` classes
- **AND** require an initial value at model build time
- **AND** `grid` MUST be a `Literal["Scalar", "FourBox", "Hemispheric"]` defaulting to "Scalar"

#### Scenario: Metaclass generates Inputs class

- **WHEN** a subclass of `Component` is defined with Input/State declarations
- **THEN** an inner `Inputs` class MUST be generated
- **AND** have typed fields for each input and state (using appropriate window types)
- **AND** provide `from_input_state(mapping)` class method
- **AND** raise `KeyError` for missing required inputs

#### Scenario: Metaclass generates Outputs class

- **WHEN** a subclass of `Component` is defined with Output/State declarations
- **THEN** an inner `Outputs` class MUST be generated
- **AND** have typed fields for each output and state
- **AND** validate all required fields are provided in `__init__`
- **AND** raise `TypeError` for missing or extra fields
- **AND** provide `to_dict() -> dict[str, StateValue]` method for Rust interop

#### Scenario: definitions() auto-generated

- **WHEN** calling `component.definitions()` on a typed Python component
- **THEN** it MUST return `list[RequirementDefinition]` for all inputs, outputs, and states
- **AND** convert descriptors to `RequirementDefinition` objects

#### Scenario: solve() method contract

- **WHEN** implementing `solve(t_current, t_next, inputs)` on a typed component
- **THEN** `inputs` MUST be an instance of the generated `Inputs` class
- **AND** the method MUST return an instance of the generated `Outputs` class

### Requirement: Python Slice Types for Outputs

The system SHALL provide Python slice types for typed component output construction.

#### Scenario: FourBoxSlice construction

- **WHEN** constructing a `FourBoxSlice` in Python
- **THEN** it MUST support keyword arguments for each region
- **AND** `FourBoxSlice.uniform(value)` creates slice with all regions equal
- **AND** `FourBoxSlice.from_array(values)` creates from list of 4 values

#### Scenario: FourBoxSlice access

- **WHEN** using a `FourBoxSlice` in Python
- **THEN** regions MUST be accessible via properties (`northern_ocean`, `northern_land`, etc.)
- **AND** index access via `slice[index]` MUST work
- **AND** `to_array()` returns numpy array
- **AND** `to_dict()` returns dict with region names as keys

#### Scenario: HemisphericSlice construction and access

- **WHEN** using a `HemisphericSlice` in Python
- **THEN** regions MUST be accessible via `northern` and `southern` properties
- **AND** same construction patterns as `FourBoxSlice` MUST be supported

### Requirement: Python StateValue Type

The system SHALL provide a `StateValue` class for representing scalar or spatially-resolved values.

#### Scenario: StateValue factory methods

- **WHEN** creating a `StateValue` in Python
- **THEN** `StateValue.scalar(value)` MUST create a scalar StateValue
- **AND** `StateValue.four_box(slice)` MUST create a FourBox StateValue
- **AND** `StateValue.hemispheric(slice)` MUST create a Hemispheric StateValue

#### Scenario: StateValue type checking

- **WHEN** using a `StateValue` in Python
- **THEN** `is_scalar()`, `is_four_box()`, `is_hemispheric()` MUST return the correct boolean
- **AND** only one type check method returns `True` for any instance

#### Scenario: StateValue accessors

- **WHEN** accessing the inner value of a `StateValue`
- **THEN** `as_scalar()` MUST return `float | None` (None if not scalar)
- **AND** `as_four_box()` MUST return `FourBoxSlice | None`
- **AND** `as_hemispheric()` MUST return `HemisphericSlice | None`
- **AND** `to_scalar()` MUST return a float, aggregating grid values if necessary

#### Scenario: StateValue public export

- **WHEN** importing from the `rscm` package
- **THEN** `StateValue`, `FourBoxSlice`, and `HemisphericSlice` MUST be available as public exports

### Requirement: ModelBuilder Python Interface

The system SHALL provide a `ModelBuilder` class for constructing models in Python.

#### Scenario: Build model with components

- **WHEN** using `ModelBuilder` in Python
- **THEN** `with_time_axis(time_axis)` sets the model time axis
- **AND** `with_py_component(component)` adds a `PythonComponent`
- **AND** `with_rust_component(component)` adds a `RustComponent`
- **AND** `with_initial_values(dict)` sets initial state values
- **AND** `with_exogenous_variable(name, timeseries)` adds exogenous data
- **AND** `with_exogenous_collection(collection)` adds multiple exogenous timeseries
- **AND** `build()` returns a `Model` instance

#### Scenario: ModelBuilder validation

- **WHEN** calling `ModelBuilder.build()`
- **THEN** it MUST validate that all component inputs are satisfied
- **AND** raise an exception with clear message if validation fails

### Requirement: Model Python Interface

The system SHALL provide a `Model` class for running simulations in Python.

#### Scenario: Model execution

- **WHEN** using a `Model` in Python
- **THEN** `step()` advances the model by one timestep
- **AND** `run()` runs the model to completion
- **AND** `finished()` returns whether the model has completed

#### Scenario: Model state access

- **WHEN** querying model state in Python
- **THEN** `current_time()` returns the current simulation time
- **AND** `current_time_bounds()` returns `(start, end)` tuple for current step
- **AND** `timeseries()` returns a `TimeseriesCollection` with all model state

#### Scenario: Model serialization

- **WHEN** serializing a `Model` in Python
- **THEN** `to_toml()` returns TOML string representation
- **AND** `Model.from_toml(string)` reconstructs the model

#### Scenario: Model visualization

- **WHEN** calling `model.as_dot()`
- **THEN** it MUST return a GraphViz DOT string
- **AND** the graph shows component dependencies
