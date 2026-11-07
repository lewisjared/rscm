from enum import Enum, auto
from typing import final

import numpy as np
from numpy.typing import NDArray

from . import TimeseriesType

# RSCM uses 64bit floats throughout
Arr = NDArray[np.float64]
F = np.float64 | float

@final
class StateValue:
    """Represents a value that can be scalar or spatially-resolved.

    In the Rust core, this is an enum with three variants:

    - `Scalar(float)`: a single global value
    - `FourBox(FourBoxSlice)`: four-box regional values
    - `Hemispheric(HemisphericSlice)`: hemispheric values
    """

    @staticmethod
    def scalar(value: float) -> StateValue:
        """Create a scalar StateValue."""
        ...
    @staticmethod
    def four_box(slice: FourBoxSlice) -> StateValue:
        """Create a four-box StateValue from a FourBoxSlice."""
        ...
    @staticmethod
    def hemispheric(slice: HemisphericSlice) -> StateValue:
        """Create a Hemispheric StateValue from a HemisphericSlice."""
        ...
    def is_scalar(self) -> bool:
        """Check if this is a scalar value."""
        ...
    def is_four_box(self) -> bool:
        """Check if this is a four-box grid value."""
        ...
    def is_hemispheric(self) -> bool:
        """Check if this is a Hemispheric grid value."""
        ...
    def as_scalar(self) -> float | None:
        """Get the scalar value if this is a Scalar variant, otherwise None."""
        ...
    def as_four_box(self) -> FourBoxSlice | None:
        """Get the FourBoxSlice if this is a FourBox variant, otherwise None."""
        ...
    def as_hemispheric(self) -> HemisphericSlice | None:
        """Get the HemisphericSlice if this is a Hemispheric variant, otherwise None."""
        ...
    def to_scalar(self) -> float:
        """Convert to a scalar value, aggregating grid values if necessary.

        For Scalar: returns the value directly.
        For FourBox: returns the weighted mean of all 4 regional values.
        For Hemispheric: returns the mean of both hemispheres.
        """
        ...
    def __repr__(self) -> str: ...

@final
class FourBoxSlice:
    @property
    def northern_ocean(self) -> float: ...
    @northern_ocean.setter
    def northern_ocean(self, value: float) -> None: ...
    @property
    def northern_land(self) -> float: ...
    @northern_land.setter
    def northern_land(self, value: float) -> None: ...
    @property
    def southern_ocean(self) -> float: ...
    @southern_ocean.setter
    def southern_ocean(self, value: float) -> None: ...
    @property
    def southern_land(self) -> float: ...
    @southern_land.setter
    def southern_land(self, value: float) -> None: ...
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
    def __getitem__(self, index: int, /) -> float: ...
    def __setitem__(self, index: int, value: float, /) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
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
    def __getitem__(self, index: int, /) -> float: ...
    def __setitem__(self, index: int, value: float, /) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class VariableSource(Enum):
    """Where an input's values come from, which decides the index `get()` reads."""

    Prescribed = auto()
    Upstream = auto()
    Lagged = auto()
    OwnState = auto()

@final
class TimeseriesWindow:
    """Zero-cost view into scalar timeseries data."""

    source: VariableSource
    timeseries_type: TimeseriesType

    def __init__(
        self,
        values: list[float],
        time_index: int,
        *,
        times: list[float] | None = None,
        source: VariableSource = ...,
        timeseries_type: TimeseriesType = ...,
    ) -> None: ...
    def get(self) -> float:
        """Value at the index the model reads for this input.

        A point reads index N+1, or N for own state and a lagged edge.
        An average reads index N, or N-1 for own state.
        """
        ...
    def get_time(self) -> float:
        """Time of the value `get()` returns."""
        ...
    def history(self, n: int) -> float | None:
        """Value `n` steps before the one `get()` returns.

        Returns None before the first step.
        """
        ...
    def history_len(self) -> int:
        """Count the values `history` can read."""
        ...
    def interval(self) -> tuple[float, float]:
        """Values at the start and end of the step.

        A lagged input gives the previous step's interval.
        Raises ValueError on an average and on a component's own state.
        """
        ...
    def interpolate(self, t: float) -> float:
        """Value at time `t`, held at the first or last value beyond the window.

        The window holds values up to the end of the current step.
        """
        ...
    def to_array(self) -> Arr: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class FourBoxTimeseriesWindow:
    """View into four-box grid timeseries data."""

    source: VariableSource
    timeseries_type: TimeseriesType

    def __init__(
        self,
        values: list[list[float]],
        time_index: int,
        *,
        times: list[float] | None = None,
        source: VariableSource = ...,
        timeseries_type: TimeseriesType = ...,
    ) -> None: ...
    def get(self) -> FourBoxSlice:
        """Regional values at the index the model reads for this input.

        A point reads index N+1, or N for own state and a lagged edge.
        An average reads index N, or N-1 for own state.
        """
        ...
    def get_time(self) -> float:
        """Time of the values `get()` returns."""
        ...
    def history(self, n: int) -> FourBoxSlice | None:
        """Regional values `n` steps before those `get()` returns.

        Returns None before the first step.
        """
        ...
    def history_len(self) -> int:
        """Count the values `history` can read."""
        ...
    def interval(self) -> tuple[FourBoxSlice, FourBoxSlice]:
        """Regional values at the start and end of the step.

        A lagged input gives the previous step's interval.
        Raises ValueError on an average and on a component's own state.
        """
        ...
    def interpolate(self, t: float) -> FourBoxSlice:
        """Regional values at time `t`, held at the first or last beyond the window.

        The window holds values up to the end of the current step.
        """
        ...
    def region(self, region: int) -> TimeseriesWindow: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class HemisphericTimeseriesWindow:
    """View into Hemispheric grid timeseries data."""

    source: VariableSource
    timeseries_type: TimeseriesType

    def __init__(
        self,
        values: list[list[float]],
        time_index: int,
        *,
        times: list[float] | None = None,
        source: VariableSource = ...,
        timeseries_type: TimeseriesType = ...,
    ) -> None: ...
    def get(self) -> HemisphericSlice:
        """Regional values at the index the model reads for this input.

        A point reads index N+1, or N for own state and a lagged edge.
        An average reads index N, or N-1 for own state.
        """
        ...
    def get_time(self) -> float:
        """Time of the values `get()` returns."""
        ...
    def history(self, n: int) -> HemisphericSlice | None:
        """Regional values `n` steps before those `get()` returns.

        Returns None before the first step.
        """
        ...
    def history_len(self) -> int:
        """Count the values `history` can read."""
        ...
    def interval(self) -> tuple[HemisphericSlice, HemisphericSlice]:
        """Regional values at the start and end of the step.

        A lagged input gives the previous step's interval.
        Raises ValueError on an average and on a component's own state.
        """
        ...
    def interpolate(self, t: float) -> HemisphericSlice:
        """Regional values at time `t`, held at the first or last beyond the window.

        The window holds values up to the end of the current step.
        """
        ...
    def region(self, region: int) -> TimeseriesWindow: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
