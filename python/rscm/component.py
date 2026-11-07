"""
Typed Python components for RSCM.

This module provides a base class for creating Python components with
type-safe input and output declarations, similar to the Rust ComponentIO
derive macro.

Classes
-------
Component
    Base class for typed Python components. Subclass this to create
    custom climate model components.
Input
    Descriptor for declaring input variables.
Output
    Descriptor for declaring output variables.
State
    Descriptor for declaring state variables (read previous, write new).

Usage Pattern
-------------
1. Subclass ``Component``
2. Declare variables using ``Input``, ``Output``, ``State`` class attributes
3. Implement ``solve(t_current, t_next, inputs)`` returning ``self.Outputs(...)``
4. Pass an instance to ``ModelBuilder.with_component()``

The metaclass automatically generates:

- ``definitions()`` method returning ``list[VariableDefinition]``
- ``Inputs`` inner class with ``TimeseriesWindow`` fields for each input/state
- ``Outputs`` inner class for constructing return values

Examples
--------
```python
from rscm.component import Component, Input, Output, State


class CarbonCycle(Component):
    # Declare inputs
    emissions = Input("Emissions|CO2", unit="Mt CO2/yr")
    temperature = Input("Surface Air Temperature Change", unit="K", grid="four_box")

    # Declare state variables (read previous, write new)
    concentration = State("Atmospheric Concentrations|CO2", unit="ppm")

    # Declare outputs
    uptake = Output("Net Atmosphere to Land Flux|CO2", unit="Mt CO2/yr")

    def __init__(self, sensitivity: float):
        self.sensitivity = sensitivity

    def solve(
        self, t_current: float, t_next: float, inputs: "CarbonCycle.Inputs"
    ) -> "CarbonCycle.Outputs":
        # get() returns the value the model hands this component for the step
        emissions = inputs.emissions.get()
        temp = inputs.temperature.get()  # FourBoxSlice of regional values
        conc_prev = inputs.concentration.get()

        # Calculate outputs
        new_conc = conc_prev + emissions * self.sensitivity
        uptake = emissions * 0.5

        return self.Outputs(
            concentration=new_conc,
            uptake=uptake,
        )
```

Grid Support
------------
Specify spatial grids with the ``grid`` parameter:

- ``grid="scalar"`` (default): Single global value
- ``grid="four_box"``: MAGICC four-box (NO, NL, SO, SL)
- ``grid="hemispheric"``: Northern/Southern hemispheres

Grid outputs should return ``FourBoxSlice`` or ``HemisphericSlice`` objects.

See Also
--------
rscm.core : Core types including ModelBuilder, TimeAxis, Timeseries
rscm.core.FourBoxSlice : Container for four-box regional values
rscm.core.HemisphericSlice : Container for hemispheric values
rscm.core.ModelBuilder : Takes an instance through ``with_component()``
"""

from __future__ import annotations

import functools
import importlib.metadata
import json
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from rscm._lib.core import (
    GridType,
    TimeseriesType,
    VariableDefinition,
    VariableRole,
)
from rscm._lib.core.state import (
    FourBoxSlice,
    FourBoxTimeseriesWindow,
    HemisphericSlice,
    HemisphericTimeseriesWindow,
    StateValue,
    TimeseriesWindow,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

# Valid grid type literals for component I/O declarations
GridLiteral = Literal["scalar", "four_box", "hemispheric"]

# Valid timeseries type literals for component I/O declarations
TimeseriesTypeLiteral = Literal["point", "average"]

# Entry point group that installed packages use to provide components
ENTRY_POINT_GROUP = "rscm.components"


@functools.cache
def _load_entry_points() -> None:
    for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            entry_point.load()
        except Exception as e:
            warnings.warn(
                f"Could not load rscm components from {entry_point.value!r} "
                f"(entry point {entry_point.name!r}): {e!r}",
                stacklevel=3,
            )


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
        The spatial grid: "scalar" (default), "four_box" or "hemispheric"
    timeseries_type
        "point" or "average", or None (default) to take it from the unit
    """

    name: str
    unit: str = ""
    grid: GridLiteral = "scalar"
    timeseries_type: TimeseriesTypeLiteral | None = None

    def to_definition(self) -> VariableDefinition:
        """Convert to a VariableDefinition."""
        return VariableDefinition(
            self.name,
            self.unit,
            VariableRole.Input,
            _parse_grid(self.grid),
            _parse_timeseries_type(self.timeseries_type),
        )


@dataclass(frozen=True)
class Output:
    """Declare an output variable for a component.

    Parameters
    ----------
    name
        The variable name (e.g., "Atmospheric Concentrations|CO2")
    unit
        The unit string (e.g., "ppm")
    grid
        The spatial grid: "scalar" (default), "four_box" or "hemispheric"
    timeseries_type
        "point" or "average", or None (default) to take it from the unit
    """

    name: str
    unit: str = ""
    grid: GridLiteral = "scalar"
    timeseries_type: TimeseriesTypeLiteral | None = None

    def to_definition(self) -> VariableDefinition:
        """Convert to a VariableDefinition."""
        return VariableDefinition(
            self.name,
            self.unit,
            VariableRole.Output,
            _parse_grid(self.grid),
            _parse_timeseries_type(self.timeseries_type),
        )


@dataclass(frozen=True)
class State:
    """Declare a state variable for a component.

    State variables read their previous value and write a new value each step.
    They appear in both inputs and outputs.

    Parameters
    ----------
    name
        The variable name (e.g., "Atmospheric Concentrations|CO2")
    unit
        The unit string (e.g., "ppm")
    grid
        The spatial grid: "scalar" (default), "four_box" or "hemispheric"
    timeseries_type
        "point" or "average", or None (default) to take it from the unit
    """

    name: str
    unit: str = ""
    grid: GridLiteral = "scalar"
    timeseries_type: TimeseriesTypeLiteral | None = None

    def to_definition(self) -> VariableDefinition:
        """Convert to a VariableDefinition."""
        return VariableDefinition(
            self.name,
            self.unit,
            VariableRole.State,
            _parse_grid(self.grid),
            _parse_timeseries_type(self.timeseries_type),
        )


def _parse_grid(grid: GridLiteral) -> GridType:
    """Parse a grid type literal to GridType enum."""
    if grid == "scalar":
        return GridType.Scalar
    elif grid == "four_box":
        return GridType.FourBox
    elif grid == "hemispheric":
        return GridType.Hemispheric
    else:
        raise ValueError(  # noqa: TRY003
            f"Unknown grid: {grid}. Must be scalar, four_box or hemispheric"
        )


def _parse_timeseries_type(
    timeseries_type: TimeseriesTypeLiteral | None,
) -> TimeseriesType | None:
    """Parse a timeseries type literal to TimeseriesType enum."""
    if timeseries_type is None:
        return None
    elif timeseries_type == "point":
        return TimeseriesType.Point
    elif timeseries_type == "average":
        return TimeseriesType.Average
    else:
        raise ValueError(  # noqa: TRY003
            f"Unknown timeseries type: {timeseries_type}. Must be point or average"
        )


def _get_window_type(grid: GridLiteral) -> type:
    """Get the appropriate TimeseriesWindow type for a grid."""
    if grid == "four_box":
        return FourBoxTimeseriesWindow
    elif grid == "hemispheric":
        return HemisphericTimeseriesWindow
    else:
        return TimeseriesWindow


def _get_output_type(grid: GridLiteral) -> type:
    """Get the appropriate output type for a grid."""
    if grid == "four_box":
        return FourBoxSlice
    elif grid == "hemispheric":
        return HemisphericSlice
    else:
        return float


class ComponentMeta(type):
    """
    Metaclass for Component that generates typed Inputs and Outputs classes.

    This metaclass collects Input, Output, and State declarations as class
    attributes and generates corresponding typed Inputs and Outputs classes.
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

        def to_dict(self) -> dict[str, StateValue]:
            """Convert outputs to a dictionary for Rust interop.

            Returns
            -------
            Dictionary mapping variable names to StateValue objects
            """
            result: dict[str, StateValue] = {}
            for field_name, (var_name, _grid, _) in self._field_info.items():
                value = getattr(self, field_name)
                # Wrap values in StateValue for Rust interop
                if isinstance(value, FourBoxSlice):
                    result[var_name] = StateValue.four_box(value)
                elif isinstance(value, HemisphericSlice):
                    result[var_name] = StateValue.hemispheric(value)
                else:
                    result[var_name] = StateValue.scalar(value)
            return result

        def __repr__(self) -> str:
            fields = ", ".join(
                f"{name}={getattr(self, name, None)!r}" for name in self._field_info
            )
            return f"{self.__class__.__name__}({fields})"

    OutputsBase.__name__ = f"{component_name}Outputs"
    OutputsBase.__qualname__ = f"{component_name}.Outputs"

    return OutputsBase


def _slot_names(cls: type) -> list[str]:
    slots = cls.__dict__.get("__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    return [name for name in slots if name not in ("__dict__", "__weakref__")]


class Component(metaclass=ComponentMeta):
    """Base class for typed Python components.

    Subclasses declare inputs, outputs, and state variables using class-level
    descriptors, and implement the `solve()` method to compute outputs from inputs.

    Python components are automatically registered in a class-level registry when
    defined, which can be queried via `Component.get_registered_components()`.
    This registry is used for documentation generation.

    Installed packages can add components to the registry
    by declaring an entry point in the `rscm.components` group.
    The registry imports each entry point the first time it is queried.

    Note
    ----
    This registry only tracks **Python** components (subclasses of this class).
    Rust components use a separate metadata system - see
    `docs.gen_doc_stubs.load_component_metadata()` which loads JSON metadata
    generated by the `rscm-doc-gen` crate.

    Examples
    --------
    ```python
    class MyComponent(Component):
        emissions = Input("Emissions|CO2", unit="Mt CO2/yr")
        concentration = Output("Atmospheric Concentrations|CO2", unit="ppm")

        def __init__(self, factor: float):
            self.factor = factor

        def solve(self, t_current, t_next, inputs):
            return self.Outputs(concentration=inputs.emissions.get() * self.factor)
    ```

    To exclude a component from the registry (e.g., for abstract base classes):
    ```python
    class AbstractComponent(Component, register=False):
        pass
    ```
    """

    # Component registry - maps class name to class
    _registry: ClassVar[dict[str, type[Component]]] = {}

    # These are populated by the metaclass
    _component_inputs: ClassVar[dict[str, Input]] = {}
    _component_outputs: ClassVar[dict[str, Output]] = {}
    _component_states: ClassVar[dict[str, State]] = {}

    # Generated by metaclass for subclasses
    Inputs: ClassVar[type]
    Outputs: ClassVar[type]

    def __init_subclass__(cls, register: bool = True, **kwargs: Any) -> None:
        """Register subclasses in the component registry.

        Parameters
        ----------
        register
            Whether to register this component. Set to False for abstract
            base classes that should not appear in documentation.
        **kwargs
            Passed to parent __init_subclass__
        """
        super().__init_subclass__(**kwargs)
        if register:
            existing = Component._registry.get(cls.__name__)
            if existing is not None and existing.__module__ != cls.__module__:
                warnings.warn(
                    f"Component {cls.__module__}.{cls.__qualname__} replaces "
                    f"{existing.__module__}.{existing.__qualname__} in the registry, "
                    "because components are registered by class name",
                    stacklevel=3,
                )
            Component._registry[cls.__name__] = cls

    @classmethod
    def get_registered_components(cls) -> dict[str, type[Component]]:
        """Return all registered Python component classes.

        This only returns Python components (subclasses of Component).
        For Rust components, see `docs.gen_doc_stubs.load_component_metadata()`.

        Returns
        -------
        Dictionary mapping component names to their classes
        """
        _load_entry_points()
        return dict(cls._registry)

    @classmethod
    def get_component(cls, name: str) -> type[Component]:
        """Get a registered component by name.

        Parameters
        ----------
        name
            The component class name

        Returns
        -------
        The component class

        Raises
        ------
        KeyError
            If no component with the given name is registered
        """
        _load_entry_points()
        if name not in cls._registry:
            raise KeyError(  # noqa: TRY003
                f"No component registered with name '{name}'. "
                f"Available: {', '.join(sorted(cls._registry.keys()))}"
            )
        return cls._registry[name]

    def definitions(self) -> list[VariableDefinition]:
        """Return the variable definitions for this component.

        This method is auto-generated from the Input, Output, and State
        class attributes.
        """
        defs: list[VariableDefinition] = []

        for inp in self._component_inputs.values():
            defs.append(inp.to_definition())

        for out in self._component_outputs.values():
            defs.append(out.to_definition())

        for state in self._component_states.values():
            defs.append(state.to_definition())

        return defs

    def solve(
        self,
        t_current: float,
        t_next: float,
        inputs: Any,
    ) -> Any:
        """Compute outputs from inputs at the current step.

        Parameters
        ----------
        t_current
            Current time
        t_next
            Next step time
        inputs
            Typed inputs object with TimeseriesWindow fields

        Returns
        -------
        Typed outputs object
        """
        raise NotImplementedError("Subclasses must implement solve()")

    def to_json(self) -> str:
        """Serialise the instance attributes for saving a model.

        Override this and `from_json` when an attribute is not JSON-serialisable.
        """
        if any(_slot_names(cls) for cls in type(self).__mro__):
            reason = "attributes in __slots__ are not saved"
        else:
            try:
                return json.dumps(vars(self))
            except TypeError as e:
                reason = str(e)
        msg = (
            f"Cannot serialise {type(self).__qualname__}: {reason}. "
            "Override to_json() and from_json() to save this component."
        )
        raise TypeError(msg)

    @classmethod
    def from_json(cls, data: str) -> Component:
        """Restore an instance from `to_json` output without calling `__init__`."""
        component = cls.__new__(cls)
        component.__dict__.update(json.loads(data))
        return component
