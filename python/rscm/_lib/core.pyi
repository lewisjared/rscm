from enum import Enum, auto
from typing import Protocol, Self, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")

# RSCM uses 64bit floats throughout
Arr = NDArray[np.float64]
F = np.float64 | float

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
    def __init__(
        self, values: Arr, time_axis: TimeAxis, units: str, interpolation_strategy
    ) -> Timeseries: ...
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

class VariableType(Enum):
    Exogenous = auto()
    Endogenous = auto()

class TimeseriesCollection:
    def __init__(self) -> TimeseriesCollection: ...
    def add_timeseries(
        self,
        name: str,
        timeseries: Timeseries,
        variable_type: VariableType = VariableType.Exogenous,
    ): ...
    def get_timeseries_by_name(self, name: str) -> Timeseries | None:
        """
        Get a timeseries from the collection by name

        Any modifications to the returned timeseries will not be reflected
        in the collection as this function returns a cloned timeseries.

        Parameters
        ----------
        name
            Name of the timeseries to query

        Returns
        -------
        A clone of the timeseries or None if the collection doesn't contain
        a timeseries by that name.
        """
    def names(self) -> list[str]: ...
    def timeseries(self) -> list[Timeseries]:
        """
        Get a list of timeseries stored in the collection.

        These are clones of the original timeseries,
        so they can be modified without affecting the original.

        Returns
        -------
        List of timeseries
        """

class ScalarRegion:
    """Region enum for scalar (global) grid"""

    GLOBAL: int = 0

class FourBoxRegion:
    """Region enum for four-box grid"""

    NORTHERN_OCEAN: int = 0
    NORTHERN_LAND: int = 1
    SOUTHERN_OCEAN: int = 2
    SOUTHERN_LAND: int = 3

class HemisphericRegion:
    """Region enum for hemispheric grid"""

    NORTHERN: int = 0
    SOUTHERN: int = 1

class ScalarGrid:
    """
    Single global region (scalar grid).

    Used for backwards compatibility with scalar timeseries and for
    variables that are truly spatially uniform (e.g., atmospheric CO₂).
    """

    def __init__(self) -> ScalarGrid: ...
    def grid_name(self) -> str: ...
    def size(self) -> int: ...
    def region_names(self) -> list[str]: ...
    def aggregate_global(self, values: list[float]) -> float:
        """
        Aggregate all regional values to a single global value.

        Parameters
        ----------
        values
            Regional values to aggregate (must have length equal to size())

        Returns
        -------
        Aggregated global value
        """

class FourBoxGrid:
    """
    MAGICC standard four-box grid structure.

    Divides the world into:
    - Northern Ocean
    - Northern Land
    - Southern Ocean
    - Southern Land
    """

    def __init__(
        self, weights: tuple[float, float, float, float] | None = None
    ) -> FourBoxGrid: ...
    @staticmethod
    def magicc_standard() -> FourBoxGrid:
        """Create a four-box grid with MAGICC standard (equal) weights"""
    @staticmethod
    def with_weights(weights: tuple[float, float, float, float]) -> FourBoxGrid:
        """Create a four-box grid with custom weights"""
    def grid_name(self) -> str: ...
    def size(self) -> int: ...
    def region_names(self) -> list[str]: ...
    def weights(self) -> tuple[float, float, float, float]: ...
    def aggregate_global(self, values: list[float]) -> float:
        """
        Aggregate all regional values to a single global value using weights.

        Parameters
        ----------
        values
            Regional values to aggregate (must have length 4)

        Returns
        -------
        Weighted global average
        """

class HemisphericGrid:
    """
    Simple north-south hemispheric split.

    Divides the world into:
    - Northern Hemisphere
    - Southern Hemisphere
    """

    def __init__(
        self, weights: tuple[float, float] | None = None
    ) -> HemisphericGrid: ...
    @staticmethod
    def equal_weights() -> HemisphericGrid:
        """Create a hemispheric grid with equal weights (0.5 each)"""
    @staticmethod
    def with_weights(weights: tuple[float, float]) -> HemisphericGrid:
        """Create a hemispheric grid with custom weights"""
    def grid_name(self) -> str: ...
    def size(self) -> int: ...
    def region_names(self) -> list[str]: ...
    def weights(self) -> tuple[float, float]: ...
    def aggregate_global(self, values: list[float]) -> float:
        """
        Aggregate all regional values to a single global value using weights.

        Parameters
        ----------
        values
            Regional values to aggregate (must have length 2)

        Returns
        -------
        Weighted global average
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
    units: str
    requirement_type: RequirementType
    grid_type: GridType

    def __init__(
        self,
        name: str,
        units: str,
        requirement_type: RequirementType,
        grid_type: GridType = GridType.Scalar,
    ): ...

# =============================================================================
# Typed Output Slices
# =============================================================================

class FourBoxSlice:
    northern_ocean: float
    northern_land: float
    southern_ocean: float
    southern_land: float

    def __init__(
        self,
        northern_ocean: float = ...,
        northern_land: float = ...,
        southern_ocean: float = ...,
        southern_land: float = ...,
    ) -> None: ...
    @staticmethod
    def uniform(value: float) -> FourBoxSlice: ...
    @staticmethod
    def from_array(values: list[float]) -> FourBoxSlice: ...
    def get(self, region: int) -> float: ...
    def set(self, region: int, value: float) -> None: ...
    def to_array(self) -> Arr: ...
    def to_list(self) -> list[float]: ...
    def to_dict(self) -> dict[str, float]: ...
    def __getitem__(self, index: int) -> float: ...
    def __setitem__(self, index: int, value: float) -> None: ...
    def __len__(self) -> int: ...

class HemisphericSlice:
    northern: float
    southern: float

    def __init__(
        self,
        northern: float = ...,
        southern: float = ...,
    ) -> None: ...
    @staticmethod
    def uniform(value: float) -> HemisphericSlice: ...
    @staticmethod
    def from_array(values: list[float]) -> HemisphericSlice: ...
    def get(self, region: int) -> float: ...
    def set(self, region: int, value: float) -> None: ...
    def to_array(self) -> Arr: ...
    def to_list(self) -> list[float]: ...
    def to_dict(self) -> dict[str, float]: ...
    def __getitem__(self, index: int) -> float: ...
    def __setitem__(self, index: int, value: float) -> None: ...
    def __len__(self) -> int: ...

# =============================================================================
# TimeseriesWindow Classes
# =============================================================================

class TimeseriesWindow:
    """Zero-cost view into scalar timeseries data."""

    current_index: int

    def __init__(self, values: list[float], current_index: int) -> None: ...
    @property
    def current(self) -> float: ...
    @property
    def previous(self) -> float: ...
    def at_offset(self, offset: int) -> float: ...
    def last_n(self, n: int) -> Arr: ...
    def to_array(self) -> Arr: ...
    def __len__(self) -> int: ...

class FourBoxTimeseriesWindow:
    """View into FourBox grid timeseries data."""

    current_index: int

    def __init__(self, values: list[list[float]], current_index: int) -> None: ...
    @property
    def current(self) -> FourBoxSlice: ...
    @property
    def previous(self) -> FourBoxSlice: ...
    def region(self, region: int) -> TimeseriesWindow: ...
    def __len__(self) -> int: ...

class HemisphericTimeseriesWindow:
    """View into Hemispheric grid timeseries data."""

    current_index: int

    def __init__(self, values: list[list[float]], current_index: int) -> None: ...
    @property
    def current(self) -> HemisphericSlice: ...
    @property
    def previous(self) -> HemisphericSlice: ...
    def region(self, region: int) -> TimeseriesWindow: ...
    def __len__(self) -> int: ...

class Component(Protocol):
    """A component of the model that can be solved"""

    def definitions(self) -> list[RequirementDefinition]: ...
    def solve(
        self, t_current: float, t_next: float, collection: TimeseriesCollection
    ) -> dict[str, float]: ...

class RustComponent(Component):
    """
    Component that has been defined in Rust
    """

class CustomComponent(Protocol):
    """
    Interface required for registering Python-based component

    See Also
    --------
    UserDefinedComponent
    """

    def definitions(self) -> list[RequirementDefinition]: ...
    def solve(
        self, t_current: float, t_next: float, input_state: dict[str, float]
    ) -> dict[str, float]: ...

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

class ModelBuilder:
    """Builder for a model"""

    def __init__(self): ...
    def with_time_axis(self, time_axis: TimeAxis) -> Self: ...
    def with_py_component(self, component: PythonComponent) -> Self: ...
    def with_rust_component(self, component: RustComponent) -> Self: ...
    def with_initial_values(self, input_state: dict[str, F]) -> Self: ...
    def with_exogenous_variable(self, name: str, timeseries: Timeseries) -> Self: ...
    def with_exogenous_collection(self, timeseries: TimeseriesCollection) -> Self: ...
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
