# state-python Specification

## Purpose

Defines the Python interface for state values, enabling Python components to return spatially-resolved values (FourBox/Hemispheric) that are written to grid timeseries, and exposing Rust component grid outputs to Python via native PyO3 types.

## ADDED Requirements

### Requirement: PyStateValue Class

The system SHALL expose `StateValue` to Python as a native PyO3 class with factory methods and accessors.

#### Scenario: Create scalar StateValue

- **WHEN** calling `StateValue.scalar(value)`
- **THEN** it MUST return a `StateValue` wrapping the scalar value
- **AND** `is_scalar()` MUST return `True`
- **AND** `as_scalar()` MUST return the value

#### Scenario: Create FourBox StateValue

- **WHEN** calling `StateValue.four_box(slice)`
- **THEN** it MUST return a `StateValue` wrapping the `FourBoxSlice`
- **AND** `is_four_box()` MUST return `True`
- **AND** `as_four_box()` MUST return the slice

#### Scenario: Create Hemispheric StateValue

- **WHEN** calling `StateValue.hemispheric(slice)`
- **THEN** it MUST return a `StateValue` wrapping the `HemisphericSlice`
- **AND** `is_hemispheric()` MUST return `True`
- **AND** `as_hemispheric()` MUST return the slice

#### Scenario: StateValue to_scalar aggregation

- **WHEN** calling `state_value.to_scalar()` on a grid StateValue
- **THEN** it MUST return the aggregated scalar value (mean of regions)
- **AND** for scalar StateValue it MUST return the value unchanged

#### Scenario: StateValue type checks return False for wrong type

- **WHEN** calling `is_scalar()` on a FourBox StateValue
- **THEN** it MUST return `False`
- **AND** `as_scalar()` MUST return `None`

### Requirement: Python Component Grid Output via StateValue

The system SHALL support Python components returning `StateValue` objects in their output dict.

#### Scenario: Python component returns FourBox output

- **WHEN** a Python component's `Outputs.to_dict()` contains a `FourBoxSlice` value
- **THEN** it MUST be wrapped as `StateValue.four_box(slice)`
- **AND** the Rust wrapper MUST extract as `StateValue::FourBox`

#### Scenario: Python component returns Hemispheric output

- **WHEN** a Python component's `Outputs.to_dict()` contains a `HemisphericSlice` value
- **THEN** it MUST be wrapped as `StateValue.hemispheric(slice)`
- **AND** the Rust wrapper MUST extract as `StateValue::Hemispheric`

#### Scenario: Python component returns scalar output

- **WHEN** a Python component's `Outputs.to_dict()` contains a float value
- **THEN** it MUST be wrapped as `StateValue.scalar(value)`
- **AND** the Rust wrapper MUST extract as `StateValue::Scalar`

## MODIFIED Requirements

### Requirement: PythonComponent Wrapper (from component-python)

The system SHALL extract `StateValue` objects from Python component outputs and convert them to Rust `StateValue` variants.

#### Scenario: Solve extracts StateValue from Python dict

- **WHEN** `PythonComponent.solve()` receives a Python dict output containing `StateValue` objects
- **THEN** it MUST extract each value as `PyStateValue`
- **AND** convert to the inner Rust `StateValue`
- **AND** return in the `OutputState` HashMap

#### Scenario: Legacy float outputs auto-wrapped

- **WHEN** `PythonComponent.solve()` receives a Python dict output containing raw float values
- **THEN** it MUST auto-wrap as `StateValue::Scalar(value)`
- **AND** maintain backwards compatibility with existing dict-based components

### Requirement: RustComponent Wrapper (from component-python)

The system SHALL return `StateValue` objects from `RustComponent.solve()` instead of aggregated scalars.

#### Scenario: RustComponent.solve() returns StateValue dict

- **WHEN** calling `RustComponent.solve()` in Python
- **THEN** the return type MUST be `dict[str, StateValue]`
- **AND** each value MUST be a `StateValue` object preserving grid structure

#### Scenario: RustComponent.solve() FourBox output preserved

- **WHEN** a Rust component's `solve()` returns `StateValue::FourBox(slice)`
- **THEN** the Python return value MUST be `StateValue` with `is_four_box() == True`
- **AND** `as_four_box()` MUST return a `FourBoxSlice` with the original values

#### Scenario: RustComponent.solve() Hemispheric output preserved

- **WHEN** a Rust component's `solve()` returns `StateValue::Hemispheric(slice)`
- **THEN** the Python return value MUST be `StateValue` with `is_hemispheric() == True`
- **AND** `as_hemispheric()` MUST return a `HemisphericSlice` with the original values

#### Scenario: RustComponent.solve() scalar output preserved

- **WHEN** a Rust component's `solve()` returns `StateValue::Scalar(value)`
- **THEN** the Python return value MUST be `StateValue` with `is_scalar() == True`
- **AND** `as_scalar()` MUST return the original value

### Requirement: Python Component Base Class (from component-python)

The system SHALL update the generated `Outputs.to_dict()` method to return `StateValue` objects.

#### Scenario: Outputs.to_dict() returns StateValue dict

- **WHEN** calling `outputs.to_dict()` on a component Outputs instance
- **THEN** the return type MUST be `dict[str, StateValue]`
- **AND** scalar fields MUST be wrapped with `StateValue.scalar()`
- **AND** FourBoxSlice fields MUST be wrapped with `StateValue.four_box()`
- **AND** HemisphericSlice fields MUST be wrapped with `StateValue.hemispheric()`

## ADDED Requirements

### Requirement: Type Stub Accuracy

The Python type stubs SHALL accurately reflect the new `StateValue` class and return types.

#### Scenario: StateValue type stub

- **WHEN** checking the type stub for `StateValue` in `state.pyi`
- **THEN** it MUST include all factory methods (`scalar`, `four_box`, `hemispheric`)
- **AND** include all type check methods (`is_scalar`, `is_four_box`, `is_hemispheric`)
- **AND** include all accessor methods (`as_scalar`, `as_four_box`, `as_hemispheric`, `to_scalar`)

#### Scenario: RustComponent.solve() type stub

- **WHEN** checking the type stub for `RustComponent.solve()`
- **THEN** the return type MUST be `dict[str, StateValue]`

#### Scenario: StateValue exported from core module

- **WHEN** importing from `rscm._lib.core`
- **THEN** `StateValue` MUST be available as a public export
