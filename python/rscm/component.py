"""
Typed Python components for RSCM

This module provides a base class for creating Python components with
type-safe input and output declarations, similar to the Rust ComponentIO
derive macro.

Example
-------
```python
from rscm.component import Component, Input, Output, State

class CarbonCycle(Component):
    # Declare inputs
    emissions = Input("Emissions|CO2", unit="GtCO2")
    temperature = Input("Surface Temperature", unit="K", grid="FourBox")

    # Declare state variables (read previous, write new)
    concentration = State("Atmospheric Concentration|CO2", unit="ppm")

    # Declare outputs
    uptake = Output("Carbon Uptake", unit="GtC")

    def __init__(self, sensitivity: float):
        self.sensitivity = sensitivity

    def solve(
        self,
        t_current: float,
        t_next: float,
        inputs: "CarbonCycle.Inputs"
    ) -> "CarbonCycle.Outputs":
        # Type-safe access to inputs
        emissions = inputs.emissions.current
        temp = inputs.temperature.current  # For grids, .current returns global average
        conc_prev = inputs.concentration.current

        # Calculate outputs
        new_conc = conc_prev + emissions * self.sensitivity
        uptake = emissions * 0.5

        return self.Outputs(
            concentration=new_conc,
            uptake=uptake,
        )
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from rscm.core import (
    FourBoxSlice,
    FourBoxTimeseriesWindow,
    GridType,
    HemisphericSlice,
    HemisphericTimeseriesWindow,
    RequirementDefinition,
    RequirementType,
    TimeseriesWindow,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class Input:
    """Declare an input variable for a component.

    Parameters
    ----------
    name
        The variable name (e.g., "Emissions|CO2")
    unit
        The unit string (e.g., "GtCO2")
    grid
        The spatial grid type: "Scalar" (default), "FourBox", or "Hemispheric"
    """

    name: str
    unit: str = ""
    grid: str = "Scalar"

    def to_requirement(self) -> RequirementDefinition:
        """Convert to a RequirementDefinition."""
        grid_type = _parse_grid_type(self.grid)
        return RequirementDefinition(
            self.name, self.unit, RequirementType.Input, grid_type
        )


@dataclass(frozen=True)
class Output:
    """Declare an output variable for a component.

    Parameters
    ----------
    name
        The variable name (e.g., "Concentration|CO2")
    unit
        The unit string (e.g., "ppm")
    grid
        The spatial grid type: "Scalar" (default), "FourBox", or "Hemispheric"
    """

    name: str
    unit: str = ""
    grid: str = "Scalar"

    def to_requirement(self) -> RequirementDefinition:
        """Convert to a RequirementDefinition."""
        grid_type = _parse_grid_type(self.grid)
        return RequirementDefinition(
            self.name, self.unit, RequirementType.Output, grid_type
        )


@dataclass(frozen=True)
class State:
    """Declare a state variable for a component.

    State variables read their previous value and write a new value each timestep.
    They appear in both inputs and outputs.

    Parameters
    ----------
    name
        The variable name (e.g., "Atmospheric Concentration|CO2")
    unit
        The unit string (e.g., "ppm")
    grid
        The spatial grid type: "Scalar" (default), "FourBox", or "Hemispheric"
    """

    name: str
    unit: str = ""
    grid: str = "Scalar"

    def to_requirement(self) -> RequirementDefinition:
        """Convert to a RequirementDefinition."""
        grid_type = _parse_grid_type(self.grid)
        return RequirementDefinition(
            self.name, self.unit, RequirementType.State, grid_type
        )


def _parse_grid_type(grid: str) -> GridType:
    """Parse a grid type string to GridType enum."""
    grid_lower = grid.lower()
    if grid_lower == "scalar":
        return GridType.Scalar
    elif grid_lower == "fourbox":
        return GridType.FourBox
    elif grid_lower == "hemispheric":
        return GridType.Hemispheric
    else:
        raise ValueError(  # noqa: TRY003
            f"Unknown grid type: {grid}. Must be Scalar, FourBox, or Hemispheric"
        )


def _get_window_type(grid: str) -> type:
    """Get the appropriate TimeseriesWindow type for a grid."""
    grid_lower = grid.lower()
    if grid_lower == "fourbox":
        return FourBoxTimeseriesWindow
    elif grid_lower == "hemispheric":
        return HemisphericTimeseriesWindow
    else:
        return TimeseriesWindow


def _get_output_type(grid: str) -> type:
    """Get the appropriate output type for a grid."""
    grid_lower = grid.lower()
    if grid_lower == "fourbox":
        return FourBoxSlice
    elif grid_lower == "hemispheric":
        return HemisphericSlice
    else:
        return float


class ComponentMeta(type):
    """
    Metaclass for Component that generates typed Inputs and Outputs classes.

    This metaclass collects Input, Output, and State declarations as class
    atributes and generates corresponding typed Inputs and Outputs classes.
    """

    def __new__(  # noqa: D102
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> ComponentMeta:
        # Collect I/O declarations from class attributes
        inputs: dict[str, Input] = {}
        outputs: dict[str, Output] = {}
        states: dict[str, State] = {}

        # Check parent classes for inherited declarations
        for base in bases:
            if hasattr(base, "_component_inputs"):
                inputs.update(base._component_inputs)
            if hasattr(base, "_component_outputs"):
                outputs.update(base._component_outputs)
            if hasattr(base, "_component_states"):
                states.update(base._component_states)

        # Collect from current class
        for attr_name, attr_value in list(namespace.items()):
            if isinstance(attr_value, Input):
                inputs[attr_name] = attr_value
            elif isinstance(attr_value, Output):
                outputs[attr_name] = attr_value
            elif isinstance(attr_value, State):
                states[attr_name] = attr_value

        # Store declarations on the class
        namespace["_component_inputs"] = inputs
        namespace["_component_outputs"] = outputs
        namespace["_component_states"] = states

        # Create the class
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Generate Inputs and Outputs classes (skip for base Component class)
        if name != "Component" and (inputs or outputs or states):
            cls.Inputs = _create_inputs_class(name, inputs, states)  # pyright: ignore[reportAttributeAccessIssue]
            cls.Outputs = _create_outputs_class(name, outputs, states)  # pyright: ignore[reportAttributeAccessIssue]

        return cls


def _create_inputs_class(
    component_name: str,
    inputs: dict[str, Input],
    states: dict[str, State],
) -> type:
    """Create a typed Inputs class for a component."""
    # Build field annotations and defaults
    annotations: dict[str, type] = {}
    field_to_var: dict[str, tuple[str, str]] = {}  # field_name -> (var_name, grid)

    for field_name, inp in inputs.items():
        window_type = _get_window_type(inp.grid)
        annotations[field_name] = window_type
        field_to_var[field_name] = (inp.name, inp.grid)

    for field_name, state in states.items():
        window_type = _get_window_type(state.grid)
        annotations[field_name] = window_type
        field_to_var[field_name] = (state.name, state.grid)

    class InputsBase:
        """Generated inputs class with typed TimeseriesWindow fields."""

        __annotations__ = annotations
        _field_to_var: ClassVar[dict[str, tuple[str, str]]] = field_to_var

        def __init__(self, **kwargs: Any) -> None:
            for name, value in kwargs.items():
                setattr(self, name, value)

        @classmethod
        def from_input_state(cls, input_state: Mapping[str, Any]) -> InputsBase:
            """Construct typed inputs from an input state mapping.

            Parameters
            ----------
            input_state
                Mapping from variable names to TimeseriesWindow objects

            Returns
            -------
            Typed inputs instance with TimeseriesWindow fields
            """
            kwargs = {}
            for field_name, (var_name, _grid) in cls._field_to_var.items():
                if var_name not in input_state:
                    raise KeyError(f"Missing required input: {var_name}")  # noqa: TRY003
                kwargs[field_name] = input_state[var_name]
            return cls(**kwargs)

        def __repr__(self) -> str:
            fields = ", ".join(
                f"{name}={getattr(self, name, None)!r}" for name in self._field_to_var
            )
            return f"{self.__class__.__name__}({fields})"

    InputsBase.__name__ = f"{component_name}Inputs"
    InputsBase.__qualname__ = f"{component_name}.Inputs"

    return InputsBase


def _create_outputs_class(
    component_name: str,
    outputs: dict[str, Output],
    states: dict[str, State],
) -> type:
    """Create a typed Outputs class for a component."""
    # Build field info
    field_info: dict[
        str, tuple[str, str, type]
    ] = {}  # field_name -> (var_name, grid, type)

    for field_name, out in outputs.items():
        out_type = _get_output_type(out.grid)
        field_info[field_name] = (out.name, out.grid, out_type)

    for field_name, state in states.items():
        out_type = _get_output_type(state.grid)
        field_info[field_name] = (state.name, state.grid, out_type)

    required_fields = set(field_info.keys())

    class OutputsBase:
        """Generated outputs class with typed fields and validation."""

        _field_info: ClassVar[dict[str, tuple[str, str, type]]] = field_info
        _required_fields: ClassVar[set[str]] = required_fields

        def __init__(self, **kwargs: Any) -> None:
            # Validate all required fields are provided
            missing = self._required_fields - set(kwargs.keys())
            if missing:
                raise TypeError(  # noqa: TRY003
                    f"Missing required output fields: {', '.join(sorted(missing))}"
                )

            # Validate no extra fields
            extra = set(kwargs.keys()) - self._required_fields
            if extra:
                raise TypeError(f"Unknown output fields: {', '.join(sorted(extra))}")  # noqa: TRY003

            for name, value in kwargs.items():
                setattr(self, name, value)

        def to_dict(self) -> dict[str, Any]:
            """Convert outputs to a dictionary for Rust interop.

            Returns
            -------
            Dictionary mapping variable names to output values
            """
            result: dict[str, Any] = {}
            for field_name, (var_name, grid, _) in self._field_info.items():
                value = getattr(self, field_name)
                # Convert slice types to their underlying values
                if isinstance(value, FourBoxSlice | HemisphericSlice):
                    result[var_name] = value.to_list()
                else:
                    result[var_name] = value
            return result

        def __repr__(self) -> str:
            fields = ", ".join(
                f"{name}={getattr(self, name, None)!r}" for name in self._field_info
            )
            return f"{self.__class__.__name__}({fields})"

    OutputsBase.__name__ = f"{component_name}Outputs"
    OutputsBase.__qualname__ = f"{component_name}.Outputs"

    return OutputsBase


class Component(metaclass=ComponentMeta):
    """Base class for typed Python components.

    Subclasses declare inputs, outputs, and state variables using class-level
    descriptors, and implement the `solve()` method to compute outputs from inputs.

    Example
    -------
    ```python
    class MyComponent(Component):
        emissions = Input("Emissions|CO2", unit="GtCO2")
        concentration = Output("Concentration|CO2", unit="ppm")

        def __init__(self, factor: float):
            self.factor = factor

        def solve(self, t_current, t_next, inputs):
            return self.Outputs(concentration=inputs.emissions.current() * self.factor)
    ```
    """

    # These are populated by the metaclass
    _component_inputs: ClassVar[dict[str, Input]] = {}
    _component_outputs: ClassVar[dict[str, Output]] = {}
    _component_states: ClassVar[dict[str, State]] = {}

    # Generated by metaclass for subclasses
    Inputs: ClassVar[type]
    Outputs: ClassVar[type]

    def definitions(self) -> list[RequirementDefinition]:
        """Return the variable definitions for this component.

        This method is auto-generated from the Input, Output, and State
        class attributes.
        """
        defs: list[RequirementDefinition] = []

        for inp in self._component_inputs.values():
            defs.append(inp.to_requirement())

        for out in self._component_outputs.values():
            defs.append(out.to_requirement())

        for state in self._component_states.values():
            defs.append(state.to_requirement())

        return defs

    def solve(
        self,
        t_current: float,
        t_next: float,
        inputs: Any,
    ) -> Any:
        """Compute outputs from inputs at the current timestep.

        Parameters
        ----------
        t_current
            Current time
        t_next
            Next timestep time
        inputs
            Typed inputs object with TimeseriesWindow fields

        Returns
        -------
        Typed outputs object
        """
        raise NotImplementedError("Subclasses must implement solve()")
