from enum import Enum, auto
from typing import Any, Protocol, Self, TypeVar

import numpy as np
from numpy.typing import NDArray

from .state import FourBoxSlice, HemisphericSlice, StateValue

T = TypeVar("T")

# RSCM uses 64bit floats throughout
Arr = NDArray[np.float64]
F = np.float64 | float

# Re-export StateValue for convenience
# See state.pyi for documentation on the three variants (Scalar, FourBox, Hemispheric)
__all__ = ["FourBoxSlice", "HemisphericSlice", "StateValue"]

class TimeAxis:
    @staticmethod
    def from_values(values: Arr) -> TimeAxis: ...
    @staticmethod
    def from_bounds(values: Arr) -> TimeAxis: ...
    def values(self): ...
    def bounds(self): ...
    def __len__(self) -> int: ...
    def at(self, index: int) -> F: ...
    def at_bounds(self, index: int) -> tuple[F, F]: ...

class InterpolationStrategy(Enum):
    Linear = auto()
    Next = auto()
    Previous = auto()

class Timeseries:
    """Scalar (1D) timeseries with interpolation support."""

    def __init__(
        self, values: Arr, time_axis: TimeAxis, units: str, interpolation_strategy
    ) -> None: ...
    def with_interpolation_strategy(self, interpolation_strategy) -> Timeseries: ...
    def __len__(self) -> int: ...
    def set(self, index: int, value: float): ...
    def values(self) -> Arr: ...
    @property
    def latest(self) -> int: ...
    @property
    def units(self) -> str: ...
    @property
    def time_axis(self) -> TimeAxis: ...
    def latest_value(self) -> F | None: ...
    def at(self, index: int) -> F: ...
    def at_time(self, time: F) -> F:
        """
        Interpolates a value for a given time using the current interpolation strategy.

        Parameters
        ----------
        time
            Time to interpolate (or potentially extrapolate)

        Raises
        ------
        RuntimeError
            Something went wrong during the interpolation.

            See the exception message for more information.

        Returns
        -------
        Interpolated value

        """

class FourBoxTimeseries:
    """FourBox grid timeseries with 4 regional values per timestep."""

    def __len__(self) -> int: ...
    def values(self) -> NDArray[np.float64]:
        """Get all values as a 2D array with shape (n_times, 4)."""
    @property
    def latest(self) -> int: ...
    @property
    def units(self) -> str: ...
    @property
    def time_axis(self) -> TimeAxis: ...

class HemisphericTimeseries:
    """Hemispheric grid timeseries with 2 regional values per timestep."""

    def __len__(self) -> int: ...
    def values(self) -> NDArray[np.float64]:
        """Get all values as a 2D array with shape (n_times, 2)."""
    @property
    def latest(self) -> int: ...
    @property
    def units(self) -> str: ...
    @property
    def time_axis(self) -> TimeAxis: ...

class VariableType(Enum):
    Exogenous = auto()
    Endogenous = auto()

class TimeseriesCollection:
    def __init__(self) -> None: ...
    def add_timeseries(
        self,
        name: str,
        timeseries: Timeseries,
        variable_type: VariableType = VariableType.Exogenous,
    ): ...
    def get_timeseries_by_name(self, name: str) -> Timeseries | None:
        """
        Get a scalar timeseries from the collection by name.

        Any modifications to the returned timeseries will not be reflected
        in the collection as this function returns a cloned timeseries.

        Parameters
        ----------
        name
            Name of the timeseries to query

        Returns
        -------
        A clone of the timeseries or None if the collection doesn't contain
        a scalar timeseries by that name.
        """
    def get_fourbox_timeseries_by_name(self, name: str) -> FourBoxTimeseries | None:
        """
        Get a FourBox grid timeseries from the collection by name.

        Parameters
        ----------
        name
            Name of the timeseries to query

        Returns
        -------
        A clone of the timeseries or None if the collection doesn't contain
        a FourBox timeseries by that name.
        """
    def get_hemispheric_timeseries_by_name(
        self, name: str
    ) -> HemisphericTimeseries | None:
        """
        Get a Hemispheric grid timeseries from the collection by name.

        Parameters
        ----------
        name
            Name of the timeseries to query

        Returns
        -------
        A clone of the timeseries or None if the collection doesn't contain
        a Hemispheric timeseries by that name.
        """
    def names(self) -> list[str]: ...
    def timeseries(self) -> list[Timeseries]:
        """
        Get a list of scalar timeseries stored in the collection.

        These are clones of the original timeseries,
        so they can be modified without affecting the original.

        Note: This only returns scalar timeseries. Use
        get_fourbox_timeseries_by_name() or get_hemispheric_timeseries_by_name()
        to retrieve grid timeseries.

        Returns
        -------
        List of scalar timeseries
        """

class RequirementType(Enum):
    Input = auto()
    Output = auto()
    State = auto()
    EmptyLink = auto()

class GridType(Enum):
    Scalar = auto()
    FourBox = auto()
    Hemispheric = auto()

class RequirementDefinition:
    name: str
    # TODO: fix naming inconsistency between 'unit' and 'units'
    unit: str
    requirement_type: RequirementType
    grid_type: GridType

    def __init__(
        self,
        name: str,
        unit: str,
        requirement_type: RequirementType,
        grid_type: GridType = GridType.Scalar,
    ): ...

class Component(Protocol):
    """A component of the model that can be solved"""

    def definitions(self) -> list[RequirementDefinition]: ...
    def solve(
        self, t_current: float, t_next: float, collection: TimeseriesCollection
    ) -> dict[str, StateValue]: ...

class RustComponent(Component):
    """
    Component that has been defined in Rust
    """

    def definitions(self) -> list[RequirementDefinition]: ...
    def input_names(self) -> list[str]:
        """Get the names of all input variables required by this component."""
    def output_names(self) -> list[str]:
        """Get the names of all output variables produced by this component."""
    def solve(
        self, t_current: float, t_next: float, collection: TimeseriesCollection
    ) -> dict[str, StateValue]: ...

class CustomComponent(Protocol):
    """
    Interface required for registering Python-based component.

    Components can use either:
    - dict-based interface:
        solve(..., input_state: dict[str, float]) -> dict[str, float]
    - Typed interface: solve(..., inputs: TypedInputs) -> TypedOutputs

    The typed interface is provided by inheriting from rscm.component.Component.

    See Also
    --------
    rscm.component.Component : Base class for typed Python components
    """

    def definitions(self) -> list[RequirementDefinition]: ...
    def solve(self, t_current: float, t_next: float, inputs: Any) -> Any: ...

class ComponentBuilder(Protocol):
    """A component of the model that can be solved"""

    @classmethod
    def from_parameters(cls: type[T], parameters: dict[str, F]) -> T:
        """
        Create a builder object from parameters

        Returns
        -------
        Builder that can create a Component
        """
    def build(self) -> RustComponent:
        """
        Create a concrete component

        Returns
        -------
        Component object that can be solved
        or coupled with other components via a `Model`.
        """

class TestComponentBuilder(ComponentBuilder): ...

class PythonComponent(Component):
    """
    A component defined in Python.

    This component must conform with the `CustomComponent` protocol.

    TODO: Example of creating a custom component
    """

    @staticmethod
    def build(component: CustomComponent) -> PythonComponent: ...

class SchemaVariableDefinition:
    """Definition of a single variable in the schema."""

    name: str
    unit: str
    grid_type: GridType

    def __init__(
        self,
        name: str,
        unit: str,
        grid_type: GridType | None = None,
    ) -> None: ...

class AggregateDefinition:
    """Definition of an aggregate variable."""

    name: str
    unit: str
    grid_type: GridType
    contributors: list[str]

    @property
    def operation_type(self) -> str:
        """Get the operation type as a string ("Sum", "Mean", or "Weighted")."""

    @property
    def weights(self) -> list[float] | None:
        """Get the weights for a Weighted operation, or None for Sum/Mean."""

class VariableSchema:
    """
    Complete variable schema for a model.

    The schema declares all variables (regular and aggregates) for a model.
    Components declare which variables they read/write, and the
    ModelBuilder validates consistency.

    Example
    -------
    >>> schema = VariableSchema()
    >>> schema.add_variable("Emissions|CO2", "GtCO2/yr")
    >>> schema.add_variable("Emissions|CH4", "GtCH4/yr")
    >>> schema.add_aggregate(
    ...     "Total Emissions", "GtCO2/yr", "Sum", ["Emissions|CO2", "Emissions|CH4"]
    ... )
    >>> schema.validate()
    """

    variables: dict[str, SchemaVariableDefinition]
    aggregates: dict[str, AggregateDefinition]

    def __init__(self) -> None: ...
    def add_variable(
        self, name: str, unit: str, grid_type: GridType | None = None
    ) -> None:
        """Add a variable to the schema."""

    def add_aggregate(
        self,
        name: str,
        unit: str,
        operation: str,
        contributors: list[str],
        weights: list[float] | None = None,
        grid_type: GridType | None = None,
    ) -> None:
        """
        Add an aggregate to the schema.

        Parameters
        ----------
        name
            Variable identifier for the aggregate result
        unit
            Physical units (must match contributors)
        operation
            Operation type: "Sum", "Mean", or "Weighted"
        contributors
            Names of variables that contribute to this aggregate
        weights
            Weights for Weighted operation (required if operation="Weighted")
        grid_type
            Spatial resolution (defaults to Scalar)

        Raises
        ------
        ValueError
            If operation is "Weighted" but weights not provided, or
            if operation is not one of "Sum", "Mean", "Weighted".
        """

    def contains(self, name: str) -> bool:
        """Check if a name exists in the schema (as variable or aggregate)."""

    def validate(self) -> None:
        """
        Validate the schema for consistency.

        Performs the following checks:
        - All aggregate contributors exist in the schema
        - Unit consistency between contributors and their aggregates
        - Grid type consistency between contributors and their aggregates
        - Weighted aggregate weight counts match contributor counts
        - No circular dependencies between aggregates

        Raises
        ------
        ValueError
            If validation fails.
        """

class ModelBuilder:
    """Builder for a model"""

    def __init__(self): ...
    def with_time_axis(self, time_axis: TimeAxis) -> Self: ...
    def with_py_component(self, component: PythonComponent) -> Self: ...
    def with_rust_component(self, component: RustComponent) -> Self: ...
    def with_initial_values(self, input_state: dict[str, F]) -> Self: ...
    def with_exogenous_variable(self, name: str, timeseries: Timeseries) -> Self: ...
    def with_exogenous_collection(self, timeseries: TimeseriesCollection) -> Self: ...
    def with_schema(self, schema: VariableSchema) -> Self:
        """
        Add a variable schema to the model for validation and aggregation.

        The schema defines the variables the model expects and any aggregates
        that should be computed. Component inputs/outputs are validated against
        the schema at build time.

        Parameters
        ----------
        schema
            The variable schema to use for validation and aggregation

        Returns
        -------
        Self for method chaining
        """
    def build(self) -> Model:
        """
        Build a concrete model from the provided information.

        Raises
        ------
        Exception
            If the model cannot be solved because the provided information is
            inconsistent.

            TODO: improve this error reporting

        Returns
        -------
        Concrete model that can be solved
        """

class Model:
    """
    A coupled set of components that are solved on a common time axis.

    These components are solved over time steps defined by the ['time_axis'].
    Components may pass state between themselves.
    Each component may require information from other components to be
    solved (endogenous) or predefined data (exogenous).

    For example, a component to calculate the
    Effective Radiative Forcing(ERF) of CO_2 may require
    CO_2 concentrations as input state and provide CO_2 ERF.
    The component is agnostic about where/how that state is defined.
    If the model has no components which provide CO_2 concentrations,
    then a CO_2 concentration timeseries must be defined externally.
    If the model also contains a carbon cycle component which produced
    CO_2 concentrations, then the ERF component will be solved after
    the carbon cycle model.
    """

    def current_time(self) -> F: ...
    def current_time_bounds(self) -> (F, F): ...
    def step(self): ...
    def run(self): ...
    def as_dot(self) -> str: ...
    def finished(self) -> bool: ...
    def timeseries(self) -> TimeseriesCollection:
        """
        Get the timeseries associated with the model.

        These timeseries will have the same time axis as the model.
        Any endrogenous values that have not yet been solved for will be NaN.

        Returns
        -------
        Clone of the timeseries held by the model
        """

    def to_toml(self) -> str:
        """
        Serialise the current state of the model to a TOML string.

        This string can be used to recreate the model at a later time using
        `~Model.from_toml`.

        Returns
        -------
        String representation of the model, including the state required to recreate
        the model at a later time.
        """

    @classmethod
    def from_toml(cls: type[T], serialised_model: str) -> T:
        """
        Create a model from a TOML string.

        Parameters
        ----------
        serialised_model
            TOML string representing the model

        Returns
        -------
        New model object with the state as defined in the TOML string.
        """
