from collections.abc import Iterable, Mapping
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal, Protocol, Self, TypeVar, final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .state import StateValue

if TYPE_CHECKING:
    from rscm.component import Component

T = TypeVar("T")

# RSCM uses 64bit floats throughout
Arr = NDArray[np.float64]
F = np.float64 | float

# Note: __all__ is defined by PyO3 at runtime, not in the stub
# The runtime __all__ includes all exported classes from this module

@final
class TimeAxis:
    """
    Steps a model is solved on, in decimal years.

    Each value is the start of a step, and each step covers the half-open
    interval up to the next bound.
    """

    @staticmethod
    def from_values(values: ArrayLike) -> TimeAxis:
        """Create from step start times. The last step is as long as the one before.

        Raises
        ------
        ValueError
            If there are fewer than 2 values,
            or they are not finite and strictly increasing.
        """
    @staticmethod
    def from_bounds(bounds: ArrayLike) -> TimeAxis:
        """Create from step bounds, one more than the number of steps.

        Raises
        ------
        ValueError
            If there are fewer than 2 bounds,
            or they are not finite and strictly increasing.
        """
    def values(self) -> Arr:
        """Return the start time of each step."""
    def bounds(self) -> Arr:
        """Return the bounds of the steps."""
    def __len__(self) -> int: ...
    def __eq__(self, other: object, /) -> bool: ...
    def at(self, time_index: int) -> F | None:
        """Return the start time of the step at `time_index`, or None out of range.

        A negative index counts from the end.
        """
    def at_bounds(self, time_index: int) -> tuple[F, F] | None:
        """Return the start and end of the step at `time_index`, or None out of range.

        A negative index counts from the end.
        """

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
        values: ArrayLike,
        time_axis: TimeAxis,
        unit: str,
        interpolation_strategy: InterpolationStrategy,
    ) -> None:
        """Create a timeseries of `values` on `time_axis`.

        Raises
        ------
        ValueError
            If `unit` is not empty and is not a unit `Unit` parses,
            or `values` does not match `time_axis`.
        """
    @staticmethod
    def from_values(values: ArrayLike, time_axis: TimeAxis) -> Timeseries: ...
    def with_interpolation_strategy(
        self, interpolation_strategy: InterpolationStrategy
    ) -> None: ...
    def __len__(self) -> int: ...
    def set(self, time_index: int, value: float) -> None:
        """Set the value at `time_index`, counting a negative index from the end.

        Raises
        ------
        IndexError
            If `time_index` is out of range.
        """
    def values(self) -> Arr: ...
    @property
    def latest(self) -> int: ...
    @property
    def unit(self) -> str: ...
    @property
    def time_axis(self) -> TimeAxis: ...
    def convert_to(self, unit: str) -> Timeseries:
        """
        Return a copy of the series in `unit`.

        Raises `ValueError` if `unit` is not compatible with the series' unit.
        """
    @property
    def interpolation_strategy(self) -> InterpolationStrategy: ...
    def latest_value(self) -> F | None: ...
    def at(self, time_index: int) -> F | None:
        """Return the value at `time_index`, or None out of range.

        A negative index counts from the end.
        """
    def at_time(self, time: F) -> F:
        """
        Interpolates a value for a given time using the current interpolation strategy.

        Parameters
        ----------
        time
            Time to interpolate (or potentially extrapolate)

        Raises
        ------
        ValueError
            If `time` is NaN.
        RuntimeError
            Something went wrong during the interpolation.

            See the exception message for more information.

        Returns
        -------
        Interpolated value

        """

@final
class FourBoxTimeseries:
    """Four-box grid timeseries with 4 regional values per step."""

    def __init__(
        self,
        values: ArrayLike,
        time_axis: TimeAxis,
        unit: str,
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
    def unit(self) -> str: ...
    @property
    def time_axis(self) -> TimeAxis: ...
    def convert_to(self, unit: str) -> FourBoxTimeseries:
        """
        Return a copy of the series in `unit`.

        Raises `ValueError` if `unit` is not compatible with the series' unit.
        """

@final
class HemisphericTimeseries:
    """Hemispheric grid timeseries with 2 regional values per step."""

    def __init__(
        self,
        values: ArrayLike,
        time_axis: TimeAxis,
        unit: str,
        interpolation_strategy: InterpolationStrategy,
    ) -> None:
        """Create from values with shape (n_times, 2) in grid order.

        The order is northern, southern, with equal hemisphere weights.
        """

    def __len__(self) -> int: ...
    def values(self) -> NDArray[np.float64]:
        """Get all values as a 2D array with shape (n_times, 2)."""
    @property
    def latest(self) -> int: ...
    @property
    def unit(self) -> str: ...
    @property
    def time_axis(self) -> TimeAxis: ...
    def convert_to(self, unit: str) -> HemisphericTimeseries:
        """
        Return a copy of the series in `unit`.

        Raises `ValueError` if `unit` is not compatible with the series' unit.
        """

@final
class TimeseriesCollection:
    def __init__(self) -> None: ...
    def add_timeseries(
        self,
        name: str,
        timeseries: Timeseries | FourBoxTimeseries | HemisphericTimeseries,
    ) -> None:
        """Add a timeseries on any grid, raising ValueError if the name is taken."""
    def __getitem__(
        self, name: str
    ) -> Timeseries | FourBoxTimeseries | HemisphericTimeseries:
        """Get a timeseries on any grid, raising KeyError with close matches."""
    def __contains__(self, name: str) -> bool: ...
    def __len__(self) -> int: ...
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
    def get_four_box_timeseries_by_name(self, name: str) -> FourBoxTimeseries | None:
        """
        Get a four-box grid timeseries from the collection by name.

        Parameters
        ----------
        name
            Name of the timeseries to query

        Returns
        -------
        A clone of the timeseries or None if the collection doesn't contain
        a four-box timeseries by that name.
        """
    def get_hemispheric_timeseries_by_name(
        self, name: str
    ) -> HemisphericTimeseries | None:
        """
        Get a hemispheric grid timeseries from the collection by name.

        Parameters
        ----------
        name
            Name of the timeseries to query

        Returns
        -------
        A clone of the timeseries or None if the collection doesn't contain
        a hemispheric timeseries by that name.
        """
    def names(self) -> list[str]: ...
    def timeseries(self) -> list[Timeseries]:
        """
        Get a list of scalar timeseries stored in the collection.

        These are clones of the original timeseries,
        so they can be modified without affecting the original.

        Note: This only returns scalar timeseries. Use
        get_four_box_timeseries_by_name() or get_hemispheric_timeseries_by_name()
        to retrieve grid timeseries.

        Returns
        -------
        List of scalar timeseries
        """

@final
class VariableRole(Enum):
    """How a component uses a variable."""

    Input = auto()
    Output = auto()
    State = auto()

@final
class GridType(Enum):
    """The spatial layout of a variable's values."""

    Scalar = auto()
    FourBox = auto()
    Hemispheric = auto()

@final
class TimeseriesType(Enum):
    """Whether a variable holds at an instant (point) or over a step (average)."""

    Point = auto()
    Average = auto()

@final
class VariableDefinition:
    """A variable a component reads or writes.

    It carries a unit, a grid and a timeseries type.
    """

    name: str
    unit: str
    role: VariableRole
    grid: GridType
    timeseries_type: TimeseriesType | None

    def __init__(
        self,
        name: str,
        unit: str,
        role: VariableRole,
        grid: GridType = GridType.Scalar,
        timeseries_type: TimeseriesType | None = None,
    ): ...
    def __repr__(self) -> str: ...

@final
class PrescribedInput:
    """An input that no component or schema aggregate produces.

    The model must be given prescribed data or an initial value for it.
    """

    definition: VariableDefinition
    """The variable as its first consumer, by name, declares it."""
    consumers: list[str]
    """The components that read it, sorted by name."""
    supplied: bool
    """Whether prescribed data or an initial value supplies it."""

class RustComponent(Protocol):
    """A component defined in Rust, built by a component builder."""

    def definitions(self) -> list[VariableDefinition]: ...
    def solve(
        self, t_current: float, t_next: float, collection: TimeseriesCollection
    ) -> dict[str, StateValue]:
        """Solve one step on its own, reading every input from `collection`.

        Raises
        ------
        KeyError
            If `collection` has no timeseries for an input.
        """

class ComponentBuilder(Protocol):
    """Builder that creates a RustComponent from parameters."""

    @staticmethod
    def from_parameters(parameters: dict[str, F]) -> ComponentBuilder:
        """
        Create a builder object from parameters

        Raises
        ------
        ValueError
            Naming every missing or unknown parameter, or a parameter with a bad value.

        Returns
        -------
        Builder that can create a RustComponent
        """
    def build(self) -> RustComponent:
        """
        Create a concrete component

        Returns
        -------
        RustComponent that can be solved
        or coupled with other components via a `Model`.
        """

@final
class TestComponentBuilder(ComponentBuilder): ...

@final
class PythonComponent(RustComponent):
    """
    The wrapper that lets a Rust model solve a Python component.

    `ModelBuilder.with_component` applies it to an `rscm.component.Component` instance,
    so it is rarely needed directly.
    """

    def __init__(self, component: Component) -> None:
        """Wrap a component instance.

        Raises
        ------
        TypeError
            If `component` is a class rather than an instance.
        ValueError
            If a declared unit names an unknown unit or species.
        """
    def __repr__(self) -> str: ...
    def input_names(self) -> list[str]:
        """Get the names of all input variables required by this component."""
    def output_names(self) -> list[str]:
        """Get the names of all output variables produced by this component."""

@final
class SchemaVariableDefinition:
    """Definition of a single variable in the schema."""

    name: str
    unit: str
    grid: GridType

    def __init__(
        self,
        name: str,
        unit: str,
        grid: GridType | None = None,
    ) -> None: ...

@final
class AggregateDefinition:
    """Definition of an aggregate variable."""

    name: str
    unit: str
    grid: GridType
    contributors: list[str]

    @property
    def operation_type(self) -> str:
        """Get the operation type as a string ("sum", "mean" or "weighted_sum")."""

    @property
    def weights(self) -> list[float] | None:
        """Get the weights of a weighted sum, or None for sum and mean."""

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
    >>> schema.add_variable("Emissions|CO2|Fossil", "Mt CO2/yr")
    >>> schema.add_variable("Emissions|CO2|Biosphere", "Mt CO2/yr")
    >>> schema.add_aggregate(
    ...     "Emissions|CO2",
    ...     "Mt CO2/yr",
    ...     "sum",
    ...     ["Emissions|CO2|Fossil", "Emissions|CO2|Biosphere"],
    ... )
    >>> schema.validate()
    """

    variables: dict[str, SchemaVariableDefinition]
    aggregates: dict[str, AggregateDefinition]

    def __init__(self) -> None: ...
    def add_variable(self, name: str, unit: str, grid: GridType | None = None) -> None:
        """Add a variable to the schema.

        Raises
        ------
        ValueError
            If the schema already defines `name`, or `unit` is not empty
            and is not a unit `Unit` parses.
        """

    def add_aggregate(
        self,
        name: str,
        unit: str,
        operation: str,
        contributors: list[str],
        weights: list[float] | None = None,
        grid: GridType | None = None,
    ) -> None:
        """
        Add an aggregate to the schema.

        Parameters
        ----------
        name
            Variable identifier for the aggregate result
        unit
            Physical unit (must match contributors)
        operation
            Operation type: "sum", "mean" or "weighted_sum"
        contributors
            Names of variables that contribute to this aggregate
        weights
            Weights for the weighted sum (required if operation="weighted_sum")
        grid
            Grid of the aggregate's values (defaults to scalar)

        Raises
        ------
        ValueError
            If operation is "weighted_sum" but weights not provided,
            if operation is not one of "sum", "mean" or "weighted_sum",
            if the schema already defines `name`,
            or if `unit` is not empty and is not a unit `Unit` parses.
        """

    def contains(self, name: str) -> bool:
        """Check if a name exists in the schema (as variable or aggregate)."""

    def validate(self) -> None:
        """
        Validate the schema for consistency.

        Performs the following checks:

        - All aggregate contributors exist in the schema
        - Unit consistency between contributors and their aggregates
        - Grid consistency between contributors and their aggregates
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
    def __repr__(self) -> str: ...
    def with_time_axis(self, time_axis: TimeAxis) -> Self:
        """Set the time axis the model is solved on. `build` fails without one."""
    def with_component(
        self, component: RustComponent | PythonComponent | Component
    ) -> Self:
        """
        Add a component.

        Takes a Rust component, a `PythonComponent`
        or an `rscm.component.Component` instance, which it wraps.
        """
    def with_initial_values(
        self, initial_values: Mapping[str, F | tuple[F, str]]
    ) -> Self:
        """
        Set initial values for state variables, which are required at build time.

        A bare value is in the unit the model stores the variable in.
        A `(value, unit)` pair is converted into that unit.
        `build` fails for a name the model has no variable for,
        or for a unit incompatible with the stored one.
        """
    def with_initial_grid_values(
        self, initial_values: Mapping[str, list[F] | tuple[list[F], str]]
    ) -> Self:
        """
        Add initial values for grid variables, one per region in grid order.

        Units work as for `with_initial_values`.
        Each name holds one initial value, so the latest call for a name wins.
        `build` fails for a name the model has no variable for.
        """
    def with_prescribed_variable(
        self,
        name: str,
        timeseries: Timeseries | FourBoxTimeseries | HemisphericTimeseries,
    ) -> Self:
        """
        Supply prescribed data for a variable on any grid, replacing earlier data.

        `build` fails if no component reads the variable,
        or if a component produces it.
        """
    def with_prescribed_collection(self, timeseries: TimeseriesCollection) -> Self:
        """
        Supply prescribed data from a collection, replacing earlier data.

        Unneeded timeseries are ignored.
        """
    def with_lag(self, variable: str) -> Self:
        """
        Lag every edge from a variable.

        Each consumer then reads the previous step's value.

        The lag holds whether the variable is produced by a component or prescribed.
        Building fails if no component reads or produces the variable.
        """
    def with_lag_for(self, variable: str, consumer: str) -> Self:
        """
        Lag the edge from a variable into one component.

        ``consumer`` is the component's name, such as ``"OceanCarbon"``.
        Other consumers of the variable still read this step's value.
        Building fails if no component of that name reads the variable as an input.
        """
    def with_timeseries_type(
        self, variable: str, timeseries_type: TimeseriesType
    ) -> Self:
        """
        Declare whether a prescribed variable is a point or an average.

        Use it when the unit default is wrong.
        Building fails if a component produces the variable with another type.
        """
    def with_state_type_check(self, enabled: bool) -> Self:
        """
        Turn the check that every state variable is a point on or off.

        The check is on by default.
        """
    def with_missing_prescribed_allowed(self) -> Self:
        """
        Allow inputs that nothing supplies.

        By default, building fails if an input has no producer,
        prescribed data or initial value.
        With this flag, such an input is filled with NaN instead.
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
    def with_grid_weights(self, grid: GridType, weights: list[float]) -> Self:
        """
        Set custom weights for a grid.

        These weights override the default grid weights used when:

        - Creating timeseries for grid-based variables
        - Performing automatic grid transformations (aggregation)

        Parameters
        ----------
        grid
            The grid to set weights for (four-box or hemispheric)
        weights
            The weights for each region, which must sum to 1.0.
            Four-box order is NorthernOcean, NorthernLand, SouthernOcean, SouthernLand.
            Hemispheric order is Northern, Southern.

        Returns
        -------
        Self for method chaining

        Raises
        ------
        ValueError
            If grid is Scalar, weights have wrong length, or don't sum to 1.0

        Examples
        --------
        >>> builder = ModelBuilder()
        >>> builder.with_grid_weights(GridType.FourBox, [0.36, 0.14, 0.36, 0.14])
        """
    def prescribed_inputs(self) -> list[PrescribedInput]:
        """
        Every input that no component or schema aggregate produces, sorted by name.

        Each is marked with whether prescribed data or an initial value supplies it.
        `build` fails if any is not supplied.
        """
    def build(self) -> Model:
        """
        Build a concrete model from the provided information.

        Warns with a `UserWarning` for prescribed data that has to be extrapolated
        to cover the time axis.

        Raises
        ------
        ValueError
            If the provided information is inconsistent or incomplete,
            such as a missing time axis or initial value,
            or a name that matches no variable.

        Returns
        -------
        Concrete model that can be solved
        """

@final
class Unit:
    """A physical unit with parsing, normalisation, and conversion support.

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
    - Species after a mass unit: `Mt CH4/yr`, `Gt C/yr`,
      or fused for carbon: `GtC/yr`, `MtCO2/yr`

    Units of one species family convert by molecular mass ratio,
    as in `Gt C` to `Mt CO2`.
    Units of different species never convert.

    Examples
    --------
    >>> from rscm._lib.core import Unit
    >>>
    >>> # Parse and compare units
    >>> u1 = Unit("W/m^2")
    >>> u2 = Unit("W / m ^ 2")
    >>> assert u1 == u2  # Same normalised form
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
            If the unit string cannot be parsed, including an unknown species token.
        """

    @property
    def original(self) -> str:
        """Return the original input string used to create this unit."""

    def normalised(self) -> str:
        """Return the normalised string representation of this unit.

        The form is compact, as in `W/m^2`.
        Carbon is fused to its mass unit (`GtC/yr`),
        and every other species is spaced (`Mt CO2/yr`).
        """

    def is_dimensionless(self) -> bool:
        """Check if this unit is dimensionless."""

    def is_compatible(self, other: Unit) -> bool:
        """Check if this unit can be converted to another unit.

        Units are compatible if they have the same dimension
        and count species of the same family.
        """

    def conversion_factor(self, target: Unit) -> float:
        """Calculate the conversion factor from this unit to another unit.

        Raises
        ------
        ValueError
            If the units have incompatible dimensions or count different species.
        """

    def convert(self, value: float, target: Unit) -> float:
        """Convert a value from this unit to another unit.

        Raises
        ------
        ValueError
            If the units have incompatible dimensions or count different species.
        """

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: Unit, /) -> bool: ...  # type: ignore[override]
    def __hash__(self) -> int: ...

@final
class Model:
    """
    A coupled set of components that are solved on a common time axis.

    These components are solved over the steps defined by the ['time_axis'].
    Components may pass state between themselves.
    Each component may require information from other components to be
    solved (modelled) or predefined data (prescribed).

    For example, a component to calculate the
    Effective Radiative Forcing(ERF) of CO_2 may require
    CO_2 concentrations as input state and provide CO_2 ERF.
    The component is agnostic about where/how that state is defined.
    If the model has no components which provide CO_2 concentrations,
    then a CO_2 concentration timeseries must be prescribed.
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
    def timeseries_type(self, variable: str) -> TimeseriesType | None:
        """Whether a variable is a point or an average, or None if it is absent."""
    def coupling(self, format: Literal["rich", "plain", "json"] = "rich") -> str:
        """
        Report the model's coupling.

        Lists components in the order the runtime steps them,
        every edge between them with whether it lags,
        and each variable's unit, grid, source and timeseries type.

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
        Any modelled values that have not yet been solved for will be NaN.

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

@final
class ThreadPool:
    """
    A thread pool for building and running models, built once and reused across calls.

    Pass it as ``pool`` to [build_models][], [run_models][] and [run_builders][]
    to avoid starting new threads on every call.
    """

    def __init__(self, n_threads: int | None = None) -> None:
        """
        Start a pool.

        Parameters
        ----------
        n_threads
            Number of threads, or None for rayon's shared pool with one thread per core.

        Raises
        ------
        ValueError
            If ``n_threads`` is 0.
        """
    @property
    def n_threads(self) -> int:
        """Number of threads the pool runs work on."""

def build_models(
    builders: list[ModelBuilder],
    *,
    n_threads: int | None = None,
    pool: ThreadPool | None = None,
) -> list[Model]:
    """
    Build every builder in parallel, in input order.

    Parameters
    ----------
    builders
        Builders to build. Each is left unchanged and can be built again.
    n_threads
        Number of threads, or None for one per core.
    pool
        Pool to run on instead of ``n_threads``.

    Raises
    ------
    ValueError
        If a builder fails, naming the first that did,
        or if both ``n_threads`` and ``pool`` are given.
    """

def run_models(
    models: list[Model],
    *,
    n_threads: int | None = None,
    pool: ThreadPool | None = None,
) -> None:
    """
    Run every model to completion in parallel.

    Each model runs independently, so the results are the same as running them in turn.
    Python components hold the GIL while they solve, so they run one at a time.

    Parameters
    ----------
    models
        Models to run. A model may appear only once.
    n_threads
        Number of threads, or None for one per core.
    pool
        Pool to run on instead of ``n_threads``.

    Raises
    ------
    RuntimeError
        If a model fails, naming the first that did.
        The other models still run to completion.
        Rebuild the failed ones before retrying.
    ValueError
        If both ``n_threads`` and ``pool`` are given.
    """

def run_builders(
    builders: Iterable[ModelBuilder],
    *,
    n_threads: int | None = None,
    pool: ThreadPool | None = None,
) -> list[Model]:
    """
    Build and run every builder in parallel and return the models in input order.

    Each builder starts building and running as soon as the iterable yields it,
    so a generator that makes builders overlaps with the runs of earlier ones.
    The results are the same as [build_models][] followed by [run_models][].
    An error raised by the iterable stops it,
    and is raised once the builders it already yielded have run.

    Parameters
    ----------
    builders
        Builders to build and run. Each is left unchanged and can be built again.
    n_threads
        Number of threads, or None for one per core.
    pool
        Pool to run on instead of ``n_threads``.

    Raises
    ------
    ValueError
        If a builder fails, naming the first that did,
        or if both ``n_threads`` and ``pool`` are given.
    RuntimeError
        If a run fails, naming the first that did.
        The other models still run to completion.
    """
