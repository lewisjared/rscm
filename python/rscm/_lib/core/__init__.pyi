from enum import Enum, auto
from typing import Any, Literal, Protocol, Self, TypeVar, final

import numpy as np
from numpy.typing import NDArray

from .state import StateValue

T = TypeVar("T")

# RSCM uses 64bit floats throughout
Arr = NDArray[np.float64]
F = np.float64 | float

# Note: __all__ is defined by PyO3 at runtime, not in the stub
# The runtime __all__ includes all exported classes from this module

@final
class TimeAxis:
    """
    Time steps a model is solved on, in decimal years.

    Each value is the start of a step, and each step covers the half-open
    interval up to the next bound.
    """

    @staticmethod
    def from_values(values: Arr) -> TimeAxis:
        """Create from step start times. The last step is as long as the one before."""
    @staticmethod
    def from_bounds(bounds: Arr) -> TimeAxis:
        """Create from step bounds, one more than the number of steps."""
    def values(self) -> Arr:
        """Return the start time of each step."""
    def bounds(self) -> Arr:
        """Return the bounds of the steps."""
    def __len__(self) -> int: ...
    def at(self, index: int) -> F:
        """Return the start time of the step at `index`."""
    def at_bounds(self, index: int) -> tuple[F, F]:
        """Return the start and end of the step at `index`."""

@final
class InterpolationStrategy(Enum):
    """How a timeseries is interpolated between its time points."""

    Linear = auto()
    Next = auto()
    Previous = auto()

@final
class Timeseries:
    """Scalar (1D) timeseries with interpolation support."""

    def __init__(
        self,
        values: Arr,
        time_axis: TimeAxis,
        units: str,
        interpolation_strategy: InterpolationStrategy,
    ) -> None: ...
    @staticmethod
    def from_values(values: Arr, time: TimeAxis) -> Timeseries: ...
    def with_interpolation_strategy(
        self, interpolation_strategy: InterpolationStrategy
    ) -> Timeseries: ...
    def __len__(self) -> int: ...
    def set(self, time_index: int, value: float) -> None: ...
    def values(self) -> Arr: ...
    @property
    def latest(self) -> int: ...
    @property
    def units(self) -> str: ...
    @property
    def time_axis(self) -> TimeAxis: ...
    def latest_value(self) -> F | None: ...
    def at(self, time_index: int) -> F: ...
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

@final
class FourBoxTimeseries:
    """FourBox grid timeseries with 4 regional values per timestep."""

    def __init__(
        self,
        values: NDArray[np.float64],
        time_axis: TimeAxis,
        units: str,
        interpolation_strategy: InterpolationStrategy,
    ) -> None:
        """Create from values with shape (n_times, 4) in grid order.

        The order is NH ocean, NH land, SH ocean, SH land.
        """

    def __len__(self) -> int: ...
    def values(self) -> NDArray[np.float64]:
        """Get all values as a 2D array with shape (n_times, 4)."""
    @property
    def latest(self) -> int: ...
    @property
    def units(self) -> str: ...
    @property
    def time_axis(self) -> TimeAxis: ...

@final
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

@final
class VariableType(Enum):
    Exogenous = auto()
    Endogenous = auto()

@final
class TimeseriesCollection:
    def __init__(self) -> None: ...
    def add_timeseries(
        self,
        name: str,
        timeseries: Timeseries,
        variable_type: VariableType = VariableType.Exogenous,
    ) -> None: ...
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

@final
class RequirementType(Enum):
    """How a component uses a variable."""

    Input = auto()
    Output = auto()
    State = auto()
    EmptyLink = auto()

@final
class GridType(Enum):
    """Spatial resolution of a variable."""

    Scalar = auto()
    FourBox = auto()
    Hemispheric = auto()

@final
class RequirementDefinition:
    """A variable a component reads or writes, with its unit and grid."""

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

class CustomComponent(Protocol):
    """
    Interface required for registering Python-based component.

    Components can use either:

    - a dict-based interface:
      `solve(..., input_state: dict[str, float]) -> dict[str, float]`
    - a typed interface: `solve(..., inputs: TypedInputs) -> TypedOutputs`

    The typed interface is provided by inheriting from rscm.component.Component.

    See Also
    --------
    rscm.component.Component : Base class for typed Python components
    """

    def definitions(self) -> list[RequirementDefinition]: ...
    def solve(self, t_current: float, t_next: float, inputs: Any) -> Any: ...

class ComponentBuilder(Protocol):
    """A component of the model that can be solved"""

    @staticmethod
    def from_parameters(parameters: dict[str, F]) -> ComponentBuilder:
        """
        Create a builder object from parameters

        Returns
        -------
        Builder that can create a Component
        """
    def build(self) -> Component:
        """
        Create a concrete component

        Returns
        -------
        Component object that can be solved
        or coupled with other components via a `Model`.
        """

@final
class TestComponentBuilder(ComponentBuilder): ...

@final
class PythonComponent(Component):
    """
    A component defined in Python.

    This component must conform with the `CustomComponent` protocol.

    TODO: Example of creating a custom component
    """

    @staticmethod
    def build(component: CustomComponent) -> PythonComponent: ...
    def input_names(self) -> list[str]:
        """Get the names of all input variables required by this component."""
    def output_names(self) -> list[str]:
        """Get the names of all output variables produced by this component."""

@final
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

@final
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

@final
class VariableSchema:
    """
    Complete variable schema for a model.

    The schema declares all variables (regular and aggregates) for a model.
    Components declare which variables they read/write, and the
    ModelBuilder validates consistency.

    Examples
    --------
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

@final
class ModelBuilder:
    """Builder for a model"""

    def __init__(self) -> None: ...
    def with_time_axis(self, time_axis: TimeAxis) -> Self:
        """Set the time axis the model is solved on."""
    def with_py_component(self, component: PythonComponent) -> Self:
        """Add a component defined in Python."""
    def with_rust_component(self, component: Component) -> Self:
        """Add a component built by a Rust component builder."""
    def with_initial_values(self, initial_values: dict[str, F]) -> Self:
        """Set initial values for state variables, which are required at build time."""
    def with_initial_grid_values(self, initial_values: dict[str, list[F]]) -> Self:
        """Add initial values for grid variables, one per region in grid order."""
    def with_exogenous_variable(self, name: str, timeseries: Timeseries) -> Self:
        """Supply exogenous data for a variable. Unneeded timeseries are ignored."""
    def with_exogenous_four_box_variable(
        self, name: str, timeseries: FourBoxTimeseries
    ) -> Self:
        """Supply exogenous four-box data for a variable."""
    def with_exogenous_collection(self, timeseries: TimeseriesCollection) -> Self:
        """Supply exogenous data from a collection. Unneeded timeseries are ignored."""
    def with_lag(self, variable: str) -> Self:
        """
        Lag every edge from a variable.

        Each consumer then reads the previous step's value.

        The lag holds whether the variable is produced by a component or exogenous.
        Building fails if no component reads or produces the variable.
        """
    def with_lag_for(self, variable: str, consumer: str) -> Self:
        """
        Lag the edge from a variable into one component.

        ``consumer`` is the component's name, such as ``"OceanCarbon"``.
        Other consumers of the variable still read this step's value.
        Building fails if no component of that name reads the variable as an input.
        """
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
    def with_grid_weights(self, grid_type: GridType, weights: list[float]) -> Self:
        """
        Set custom weights for a grid type.

        These weights override the default grid weights used when:

        - Creating timeseries for grid-based variables
        - Performing automatic grid transformations (aggregation)

        Parameters
        ----------
        grid_type
            The grid type to set weights for (FourBox or Hemispheric)
        weights
            The weights for each region, which must sum to 1.0.
            FourBox order is NorthernOcean, NorthernLand, SouthernOcean, SouthernLand.
            Hemispheric order is Northern, Southern.

        Returns
        -------
        Self for method chaining

        Raises
        ------
        ValueError
            If grid_type is Scalar, weights have wrong length, or don't sum to 1.0

        Examples
        --------
        >>> builder = ModelBuilder()
        >>> builder.with_grid_weights(GridType.FourBox, [0.36, 0.14, 0.36, 0.14])
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

@final
class Unit:
    """A physical unit with parsing, normalization, and conversion support.

    This class provides comprehensive support for working with physical units
    in climate models. It handles parsing of unit strings with flexible syntax,
    dimensional analysis, and conversion factor calculation.

    Parsing
    -------
    The parser accepts several equivalent notations:

    - Exponents: `m^2`, `m**2`, `m2`
    - Division: `W/m^2`, `W m^-2`, `W per m^2`
    - Multiplication: `kg m`, `kg*m`
    - Whitespace: `W/m^2` == `W / m ^ 2`

    Examples
    --------
    >>> from rscm._lib.core import Unit
    >>>
    >>> # Parse and compare units
    >>> u1 = Unit("W/m^2")
    >>> u2 = Unit("W / m ^ 2")
    >>> assert u1 == u2  # Same normalized form
    >>>
    >>> # Convert between compatible units
    >>> gtc = Unit("GtC/yr")
    >>> mtco2 = Unit("MtCO2/yr")
    >>> factor = gtc.conversion_factor(mtco2)  # ~3666.67
    """

    def __init__(self, unit_str: str) -> None:
        """Create a new Unit from a unit string.

        Parameters
        ----------
        unit_str
            The unit string to parse.

        Raises
        ------
        ValueError
            If the unit string cannot be parsed.
        """

    @property
    def original(self) -> str:
        """Return the original input string used to create this unit."""

    def normalized(self) -> str:
        """Return the normalized string representation of this unit."""

    def is_dimensionless(self) -> bool:
        """Check if this unit is dimensionless."""

    def is_compatible(self, other: Unit) -> bool:
        """Check if this unit can be converted to another unit."""

    def conversion_factor(self, target: Unit) -> float:
        """Calculate the conversion factor from this unit to another unit.

        Raises
        ------
        ValueError
            If the units have incompatible dimensions.
        """

    def convert(self, value: float, target: Unit) -> float:
        """Convert a value from this unit to another unit.

        Raises
        ------
        ValueError
            If the units have incompatible dimensions.
        """

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: Unit, /) -> bool: ...  # type: ignore[override]
    def __hash__(self) -> int: ...

@final
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
    def current_time_bounds(self) -> tuple[F, F]: ...
    def step(self) -> None:
        """Advance one interval, raising RuntimeError on execution failure.

        After a failure, rebuild the model before retrying; component state
        may have changed even though the time index has not advanced.
        """
    def run(self) -> None:
        """Run to completion, raising RuntimeError on execution failure.

        Rebuild the model before retrying a failed run.
        """
    def as_dot(self) -> str: ...
    def finished(self) -> bool: ...
    def debug_info(self, format: Literal["rich", "plain", "json"] = "rich") -> str:
        """
        Generate debug information about the model's execution graph.

        Shows execution order, inputs (with exo/upstream/own_state source),
        outputs, states, grid types, grid transforms, and unit conversions.

        Parameters
        ----------
        format
            Output format: "rich" (coloured terminal, default), "plain" (no colours),
            or "json" (structured JSON).
        """
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

    @staticmethod
    def from_toml(string: str) -> Model:
        """
        Create a model from a TOML string.

        Parameters
        ----------
        string
            TOML string representing the model

        Returns
        -------
        New model object with the state as defined in the TOML string.
        """

def build_models(
    builders: list[ModelBuilder], *, threads: int | None = None
) -> list[Model]:
    """
    Build every builder in parallel, in input order.

    Parameters
    ----------
    builders
        Builders to build. Each is left unchanged and can be built again.
    threads
        Number of threads, or None for one per core.

    Raises
    ------
    ValueError
        If a builder fails, naming the first that did.
    """

def run_models(models: list[Model], *, threads: int | None = None) -> None:
    """
    Run every model to completion in parallel.

    Each model runs independently, so the results are the same as running them in turn.
    Python components hold the GIL while they solve, so they run one at a time.

    Parameters
    ----------
    models
        Models to run. A model may appear only once.
    threads
        Number of threads, or None for one per core.

    Raises
    ------
    RuntimeError
        If a model fails, naming the first that did.
        The other models still run to completion.
        Rebuild the failed ones before retrying.
    """
