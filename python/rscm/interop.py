"""
Convert timeseries to and from pandas and xarray.

Every conversion goes through one layout, so it works for the results of any model
and for prescribed data alike.

- [to_dataframe][rscm.interop.to_dataframe] gives a wide DataFrame
  with one row per variable and region, indexed by `variable`, `unit` and `region`,
  and a column per step start.
  This is the IAMC layout that scmdata and pandas-indexing read.
- [to_xarray][rscm.interop.to_xarray] gives the same values as an xarray Dataset,
  with one data variable per variable on `time` and `region` dimensions.
- [from_dataframe][rscm.interop.from_dataframe]
  and [from_xarray][rscm.interop.from_xarray] read them back
  into a [TimeseriesCollection][rscm.core.TimeseriesCollection],
  which `ModelBuilder.with_prescribed_collection` takes.
- [to_inputs][rscm.interop.to_inputs] gives the `(values, unit)` inputs
  of `rscm.magicc` and `rscm.fair` `setup_model`,
  and [from_inputs][rscm.interop.from_inputs] reads them into a collection.

A scalar variable has the region `World`.
A grid variable has one row per region, labelled as MAGICC7 output labels them,
see `REGIONS`.
`global_mean=True` adds a `World` row to each grid variable,
its mean weighted by the grid weights.
Reading back takes the finest grid a variable has every region of,
so those `World` rows are ignored.

The time axis, interpolation strategies and grid weights live in `DataFrame.attrs`
and in the Dataset's coordinates and attributes, so a round trip in memory is exact.
A file such as a CSV keeps none of them,
so pass `time_axis`, `interpolation_strategy` and `grid_weights` when reading one.
Without them the last step is as long as the one before, interpolation is linear
and grids have their default weights.

xarray is optional, installed with the `xarray` extra.
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from rscm.core import (
    FourBoxGrid,
    FourBoxTimeseries,
    GridType,
    HemisphericGrid,
    HemisphericTimeseries,
    InterpolationStrategy,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
)

if TYPE_CHECKING:
    import xarray as xr

__all__ = [
    "INDEX",
    "REGIONS",
    "from_dataframe",
    "from_inputs",
    "from_xarray",
    "to_dataframe",
    "to_inputs",
    "to_xarray",
]

AnyTimeseries = Timeseries | FourBoxTimeseries | HemisphericTimeseries
Strategies = InterpolationStrategy | Mapping[str, InterpolationStrategy]

INDEX = ("variable", "unit", "region")
"""Index levels of a DataFrame, in order."""

REGIONS: dict[GridType, tuple[str, ...]] = {
    GridType.Scalar: ("World",),
    GridType.FourBox: (
        "World|Northern Hemisphere|Ocean",
        "World|Northern Hemisphere|Land",
        "World|Southern Hemisphere|Ocean",
        "World|Southern Hemisphere|Land",
    ),
    GridType.Hemispheric: ("World|Northern Hemisphere", "World|Southern Hemisphere"),
}
"""Region labels of each grid, in grid order."""

_FINEST_FIRST = (GridType.FourBox, GridType.Hemispheric, GridType.Scalar)
_KNOWN_REGIONS = frozenset(r for regions in REGIONS.values() for r in regions)
_GRID_BY_SIZE = {len(regions): grid for grid, regions in REGIONS.items()}
_STRATEGIES = (
    ("linear", InterpolationStrategy.Linear),
    ("previous", InterpolationStrategy.Previous),
    ("next", InterpolationStrategy.Next),
)
_GRID_NAMES = (
    ("scalar", GridType.Scalar),
    ("four_box", GridType.FourBox),
    ("hemispheric", GridType.Hemispheric),
)


def _name_of(value: Any, table: Sequence[tuple[str, Any]]) -> str:
    return next(name for name, v in table if v == value)


def _value_of(name: str, table: Sequence[tuple[str, Any]], kind: str) -> Any:
    for key, value in table:
        if key == name:
            return value
    msg = f"Unknown {kind} {name!r}"
    raise ValueError(msg)


def _strategy_for(
    name: str, interpolation_strategy: Strategies | None
) -> InterpolationStrategy | None:
    if isinstance(interpolation_strategy, Mapping):
        return interpolation_strategy.get(name)
    return interpolation_strategy


def _grid_of(ts: AnyTimeseries) -> GridType:
    if isinstance(ts, FourBoxTimeseries):
        return GridType.FourBox
    if isinstance(ts, HemisphericTimeseries):
        return GridType.Hemispheric
    return GridType.Scalar


def _items(
    timeseries: TimeseriesCollection | Mapping[str, AnyTimeseries],
) -> list[tuple[str, AnyTimeseries]]:
    if isinstance(timeseries, TimeseriesCollection):
        return [(name, timeseries[name]) for name in timeseries]
    return list(timeseries.items())


@dataclass
class _Table:
    """Every series as rows of one array on the union of their step starts."""

    variables: list[str]
    units: list[str]
    regions: list[str]
    values: NDArray[np.float64]
    times: NDArray[np.float64]
    rows: dict[str, slice]
    grids: dict[str, GridType]
    strategies: dict[str, str]
    weights: dict[str, list[float]]
    bounds: list[list[float]]
    axis_index: dict[str, int]


def _table(
    timeseries: TimeseriesCollection | Mapping[str, AnyTimeseries],
    global_mean: bool,
) -> _Table:
    variables: list[str] = []
    units: list[str] = []
    regions: list[str] = []
    blocks: list[NDArray[np.float64]] = []
    rows: dict[str, slice] = {}
    grids: dict[str, GridType] = {}
    strategies: dict[str, str] = {}
    weights: dict[str, list[float]] = {}
    axes: list[TimeAxis] = []
    axis_index: dict[str, int] = {}
    for name, ts in _items(timeseries):
        grid = _grid_of(ts)
        values = np.asarray(ts.values(), dtype=np.float64).reshape(len(ts), -1).T
        labels = list(REGIONS[grid])
        if not isinstance(ts, Timeseries):
            weights[name] = [float(w) for w in ts.grid.weights()]
            if global_mean:
                values = np.vstack([ts.global_mean().values(), values])
                labels = ["World", *labels]
        axis = ts.time_axis
        index = next((i for i, known in enumerate(axes) if known == axis), len(axes))
        if index == len(axes):
            axes.append(axis)
        axis_index[name] = index
        rows[name] = slice(len(variables), len(variables) + len(labels))
        variables += [name] * len(labels)
        units += [ts.unit] * len(labels)
        regions += labels
        blocks.append(values)
        grids[name] = grid
        strategies[name] = _name_of(ts.interpolation_strategy, _STRATEGIES)

    starts = [np.asarray(axis.values(), dtype=np.float64) for axis in axes]
    if len(axes) == 1:
        times = starts[0]
        table = np.vstack(blocks)
    else:
        times = np.unique(np.concatenate(starts)) if starts else np.empty(0)
        table = np.full((len(variables), len(times)), np.nan)
        columns = [np.searchsorted(times, s) for s in starts]
        for name, block in zip(rows, blocks):
            table[rows[name], columns[axis_index[name]]] = block
    return _Table(
        variables,
        units,
        regions,
        table,
        times,
        rows,
        grids,
        strategies,
        weights,
        [np.asarray(axis.bounds(), dtype=np.float64).tolist() for axis in axes],
        axis_index,
    )


def _attrs(table: _Table) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "interpolation_strategy": table.strategies,
        "grid_weights": table.weights,
        "time_bounds": table.bounds,
    }
    if len(table.bounds) > 1:
        attrs["time_axis_index"] = table.axis_index
    return attrs


def to_dataframe(
    timeseries: TimeseriesCollection | Mapping[str, AnyTimeseries],
    *,
    global_mean: bool = False,
) -> pd.DataFrame:
    """
    Return every timeseries as a wide DataFrame.

    Parameters
    ----------
    timeseries
        A collection, such as `model.timeseries()`, or timeseries by variable name.
    global_mean
        Add a `World` row to each grid variable,
        its mean over the regions weighted by the series' grid weights.

    Returns
    -------
    A DataFrame indexed by `variable`, `unit` and `region`,
    with a column per step start.
    Series on different time axes share the union of their times, with NaN elsewhere.
    """
    table = _table(timeseries, global_mean)
    df = pd.DataFrame(
        table.values,
        index=pd.MultiIndex.from_arrays(
            [table.variables, table.units, table.regions], names=INDEX
        ),
        columns=pd.Index(table.times, name="time"),
    )
    df.attrs.update(_attrs(table))
    return df


def _wide(df: pd.DataFrame) -> pd.DataFrame:
    """Move `INDEX` columns to the index and drop extra levels with one value."""
    from_columns = [level for level in INDEX if level in df.columns]
    if from_columns:
        append = any(name is not None for name in df.index.names)
        df = df.set_index(from_columns, append=append)
    missing = [level for level in INDEX if level not in df.index.names]
    if missing:
        msg = f"The DataFrame has no {missing} index level or column"
        raise ValueError(msg)
    for level in [n for n in df.index.names if n not in INDEX]:
        values = df.index.unique(level=level)
        if len(values) > 1:
            msg = (
                f"The '{level}' level has several values ({list(values)[:5]}),"
                " so select one before converting"
            )
            raise ValueError(msg)
        df = df.droplevel(level)
    if list(df.index.names) != list(INDEX):
        df = df.reorder_levels(list(INDEX))
    return df


def _times(columns: pd.Index) -> NDArray[np.float64]:
    if isinstance(columns, pd.DatetimeIndex):
        msg = "Time columns must be decimal years, not datetimes"
        raise TypeError(msg)
    try:
        return np.asarray(columns, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = f"Every column must be a time in decimal years, not {list(columns)[:5]}"
        raise ValueError(msg) from error


def _axis(
    times: NDArray[np.float64],
    bounds: Sequence[float] | None,
    time_axis: TimeAxis | None,
) -> TimeAxis:
    if time_axis is not None:
        if not np.array_equal(time_axis.values(), times):
            msg = "time_axis must start a step at each time column"
            raise ValueError(msg)
        return time_axis
    edges = np.asarray(bounds if bounds is not None else [], dtype=np.float64)
    if len(edges) == len(times) + 1 and np.array_equal(edges[:-1], times):
        return TimeAxis.from_bounds(edges)
    return TimeAxis.from_values(times)


def _make(  # noqa: PLR0913
    grid: GridType,
    values: NDArray[np.float64],
    axis: TimeAxis,
    unit: str,
    strategy: InterpolationStrategy,
    weights: Sequence[float] | None,
) -> AnyTimeseries:
    if grid == GridType.FourBox:
        boxes = FourBoxGrid(None if weights is None else tuple(weights))  # type: ignore[arg-type]
        return FourBoxTimeseries(values, axis, unit, strategy, grid=boxes)
    if grid == GridType.Hemispheric:
        halves = HemisphericGrid(None if weights is None else tuple(weights))  # type: ignore[arg-type]
        return HemisphericTimeseries(values, axis, unit, strategy, grid=halves)
    return Timeseries(values[:, 0], axis, unit, strategy)


def _pick_grid(name: str, regions: Sequence[str]) -> GridType:
    unknown = sorted(set(regions) - _KNOWN_REGIONS)
    if unknown:
        msg = f"'{name}' has regions {unknown} on no grid, expected those in REGIONS"
        raise ValueError(msg)
    if len(set(regions)) != len(regions):
        msg = f"'{name}' has a region more than once"
        raise ValueError(msg)
    for grid in _FINEST_FIRST:
        if set(REGIONS[grid]) <= set(regions):
            return grid
    msg = f"'{name}' has regions {list(regions)}, which complete no grid"
    raise ValueError(msg)


@dataclass
class _Entry:
    """One variable read from a DataFrame or Dataset, before it becomes a series."""

    name: str
    unit: str
    regions: list[str]
    values: NDArray[np.float64]
    strategy: str | None
    weights: Sequence[float] | None
    bounds: Sequence[float] | None


def _read(  # noqa: PLR0913
    entries: Iterable[_Entry],
    times: NDArray[np.float64],
    shared_bounds: Sequence[float] | None,
    time_axis: TimeAxis | None,
    interpolation_strategy: Strategies | None,
    grid_weights: Mapping[GridType, Sequence[float]] | None,
) -> TimeseriesCollection:
    """Build a collection from entries whose `values` are (region, time) on `times`."""
    shared = _axis(times, shared_bounds, time_axis)
    own_axes: dict[tuple[float, ...], tuple[TimeAxis, NDArray[np.intp]]] = {}
    collection = TimeseriesCollection()
    for entry in entries:
        name = entry.name
        grid = _pick_grid(name, entry.regions)
        position = {region: i for i, region in enumerate(entry.regions)}
        values = entry.values[[position[r] for r in REGIONS[grid]]]
        axis = shared
        if entry.bounds is not None and time_axis is None:
            key = tuple(float(b) for b in entry.bounds)
            if key not in own_axes:
                own_axes[key] = _own_axis(name, key, times)
            axis, columns = own_axes[key]
            values = values[:, columns]
        strategy = _strategy_for(name, interpolation_strategy)
        if strategy is None:
            strategy = _value_of(
                entry.strategy or "linear", _STRATEGIES, "interpolation strategy"
            )
        weights = (grid_weights or {}).get(grid, entry.weights)
        collection.add_timeseries(
            name, _make(grid, values.T, axis, entry.unit, strategy, weights)
        )
    return collection


def _own_axis(
    name: str, bounds: tuple[float, ...], times: NDArray[np.float64]
) -> tuple[TimeAxis, NDArray[np.intp]]:
    starts = np.asarray(bounds[:-1], dtype=np.float64)
    columns = np.searchsorted(times, starts)
    if np.any(columns >= len(times)) or not np.array_equal(times[columns], starts):
        msg = f"'{name}' has time bounds whose starts are not all time columns"
        raise ValueError(msg)
    return TimeAxis.from_bounds(np.asarray(bounds, dtype=np.float64)), columns


def from_dataframe(
    df: pd.DataFrame,
    *,
    time_axis: TimeAxis | None = None,
    interpolation_strategy: Strategies | None = None,
    grid_weights: Mapping[GridType, Sequence[float]] | None = None,
) -> TimeseriesCollection:
    """
    Read a wide DataFrame, as `to_dataframe` writes it, into a collection.

    `variable`, `unit` and `region` may be index levels or columns.
    Any other index level must hold one value.
    Every other column is a time in decimal years, the start of a step.

    Parameters
    ----------
    df
        One row per variable and region.
    time_axis
        The time axis, which starts a step at each time column.
        By default it comes from `df.attrs`,
        or else the last step is as long as the one before.
    interpolation_strategy
        One strategy for every variable, or a strategy by variable name.
        A variable without one takes it from `df.attrs`, or else `Linear`.
    grid_weights
        The weights of each grid, in grid order.
        A variable without them takes them from `df.attrs`, or else the grid's default.

    Raises
    ------
    ValueError
        If a variable has two units, a region on no grid or an incomplete grid,
        or the index or time columns do not fit the layout.
    """
    attrs = df.attrs
    df = _wide(df)
    times = _times(df.columns)
    values = df.to_numpy(dtype=np.float64)
    strategies = attrs.get("interpolation_strategy", {})
    weights = attrs.get("grid_weights", {})
    bounds = attrs.get("time_bounds", [])
    axis_index = attrs.get("time_axis_index")
    shared_bounds = bounds[0] if axis_index is None and len(bounds) == 1 else None

    positions: dict[str, list[int]] = {}
    units: dict[str, set[str]] = {}
    regions: dict[str, list[str]] = {}
    for i, (variable, unit, region) in enumerate(df.index):
        name = str(variable)
        positions.setdefault(name, []).append(i)
        units.setdefault(name, set()).add(str(unit))
        regions.setdefault(name, []).append(str(region))

    def entries() -> Iterator[_Entry]:
        for name, rows in positions.items():
            if len(units[name]) > 1:
                msg = f"'{name}' has more than one unit: {sorted(units[name])}"
                raise ValueError(msg)
            index = None if axis_index is None else axis_index.get(name)
            yield _Entry(
                name,
                next(iter(units[name])),
                regions[name],
                values[rows],
                strategies.get(name),
                weights.get(name),
                None if index is None else bounds[index],
            )

    return _read(
        entries(), times, shared_bounds, time_axis, interpolation_strategy, grid_weights
    )


def _xarray() -> Any:
    try:
        import xarray as xr  # noqa: PLC0415
    except ImportError as error:
        msg = "xarray is not installed, install it with the rscm[xarray] extra"
        raise ImportError(msg) from error
    return xr


_XARRAY_RESERVED = frozenset({"time", "region", "bound", "time_bounds"})


def to_xarray(
    timeseries: TimeseriesCollection | Mapping[str, AnyTimeseries],
    *,
    global_mean: bool = False,
) -> "xr.Dataset":
    """
    Return every timeseries as an xarray Dataset, with the values of `to_dataframe`.

    Each variable is a data variable on `time` and `region` dimensions,
    NaN in the regions of other grids.
    Its attributes hold its `unit`, `grid`, `interpolation_strategy`
    and any `grid_weights`.
    A `time_bounds` coordinate holds the time axis when every series shares one,
    and otherwise each variable has a `time_bounds` attribute.

    Raises
    ------
    ImportError
        If xarray is not installed.
    ValueError
        If a variable is named `time`, `region`, `bound` or `time_bounds`,
        which the Dataset uses for its coordinates.
    """
    xr = _xarray()
    table = _table(timeseries, global_mean)
    clashes = sorted(_XARRAY_RESERVED & set(table.rows))
    if clashes:
        msg = f"Variables {clashes} clash with the Dataset's coordinate names"
        raise ValueError(msg)
    present = set(table.regions)
    regions = [r for g in REGIONS.values() for r in g if r in present]
    column = {region: i for i, region in enumerate(regions)}
    shared = len(table.bounds) == 1
    data_vars = {}
    for name, rows in table.rows.items():
        data = np.full((len(table.times), len(regions)), np.nan)
        data[:, [column[r] for r in table.regions[rows]]] = table.values[rows].T
        attrs: dict[str, Any] = {
            "unit": table.units[rows.start],
            "grid": _name_of(table.grids[name], _GRID_NAMES),
            "interpolation_strategy": table.strategies[name],
        }
        if name in table.weights:
            attrs["grid_weights"] = table.weights[name]
        if not shared:
            attrs["time_bounds"] = table.bounds[table.axis_index[name]]
        data_vars[name] = (("time", "region"), data, attrs)
    coords: dict[str, Any] = {"time": table.times, "region": regions}
    if shared:
        bounds = np.asarray(table.bounds[0])
        coords["time_bounds"] = (
            ("time", "bound"),
            np.column_stack([bounds[:-1], bounds[1:]]),
        )
    return xr.Dataset(data_vars, coords=coords)


def from_xarray(
    ds: "xr.Dataset",
    *,
    time_axis: TimeAxis | None = None,
    interpolation_strategy: Strategies | None = None,
    grid_weights: Mapping[GridType, Sequence[float]] | None = None,
) -> TimeseriesCollection:
    """
    Read a Dataset, as `to_xarray` writes it, into a collection.

    Each data variable needs a `unit` attribute and a `time` dimension.
    One without a `region` dimension is a scalar.
    One without a `grid` attribute is on the finest grid its non-NaN regions complete.
    The other arguments are as for `from_dataframe`,
    with the Dataset's attributes and `time_bounds` coordinate in place of `df.attrs`.

    Raises
    ------
    ValueError
        As `from_dataframe` does, or if a data variable has no `unit` attribute.
    """
    times = np.asarray(ds["time"].to_numpy(), dtype=np.float64)
    shared_bounds = None
    if "time_bounds" in ds.coords:
        edges = ds["time_bounds"].to_numpy()
        shared_bounds = [*edges[:, 0], edges[-1, 1]]
    labels = [str(r) for r in ds["region"].to_numpy()] if "region" in ds.dims else []

    def entries() -> Iterator[_Entry]:
        for key, array in ds.data_vars.items():
            name = str(key)
            if "unit" not in array.attrs:
                msg = f"'{name}' has no 'unit' attribute"
                raise ValueError(msg)
            if "region" not in array.dims:
                regions = ["World"]
                values = array.transpose("time").to_numpy()[np.newaxis, :]
            else:
                by_region = array.transpose("region", "time").to_numpy()
                if "grid" in array.attrs:
                    grid = _value_of(array.attrs["grid"], _GRID_NAMES, "grid")
                    keep = [labels.index(r) for r in REGIONS[grid]]
                else:
                    keep = np.flatnonzero(~np.isnan(by_region).all(axis=1)).tolist()
                regions = [labels[i] for i in keep]
                values = by_region[keep]
            yield _Entry(
                name,
                str(array.attrs["unit"]),
                regions,
                np.asarray(values, dtype=np.float64),
                array.attrs.get("interpolation_strategy"),
                array.attrs.get("grid_weights"),
                array.attrs.get("time_bounds"),
            )

    return _read(
        entries(), times, shared_bounds, time_axis, interpolation_strategy, grid_weights
    )


def to_inputs(
    data: pd.DataFrame | TimeseriesCollection, time_axis: TimeAxis
) -> dict[str, tuple[NDArray[np.float64], str]]:
    """
    Return the `(values, unit)` inputs of `rscm.magicc` and `rscm.fair` `setup_model`.

    Each series is interpolated with its strategy onto `time_axis`,
    with one value per step start, and four or two columns for a grid variable.
    A DataFrame is read with `from_dataframe` first.
    The inputs keep only values and units,
    so `from_inputs` takes the interpolation strategies and grid weights as arguments.
    """
    collection = (
        data if isinstance(data, TimeseriesCollection) else from_dataframe(data)
    )
    out = {}
    for name, ts in _items(collection):
        on_axis = ts if ts.time_axis == time_axis else ts.interpolate_into(time_axis)
        out[name] = (np.asarray(on_axis.values(), dtype=np.float64), ts.unit)
    return out


def from_inputs(
    inputs: Mapping[str, tuple[ArrayLike, str]],
    time_axis: TimeAxis,
    *,
    interpolation_strategy: Strategies = InterpolationStrategy.Linear,
    grid_weights: Mapping[GridType, Sequence[float]] | None = None,
) -> TimeseriesCollection:
    """
    Read `(values, unit)` inputs on `time_axis` into a collection.

    The values' shape sets the grid: one value per step for a scalar,
    four columns for four-box and two for hemispheric.
    Values one step short, as FaIR's emissions are,
    hold their last value over the last step.
    `interpolation_strategy` and `grid_weights` are as for `from_dataframe`,
    defaulting to `Linear` and each grid's default weights.
    """
    collection = TimeseriesCollection()
    for name, (values, unit) in inputs.items():
        array = np.asarray(values, dtype=np.float64)
        if len(array) == len(time_axis) - 1:
            array = np.concatenate([array, array[-1:]])
        array = array.reshape(len(array), -1)
        if array.shape[1] not in _GRID_BY_SIZE:
            msg = f"'{name}' has {array.shape[1]} columns, expected 4 or 2"
            raise ValueError(msg)
        grid = _GRID_BY_SIZE[array.shape[1]]
        strategy = _strategy_for(name, interpolation_strategy)
        if strategy is None:
            strategy = InterpolationStrategy.Linear
        weights = (grid_weights or {}).get(grid)
        collection.add_timeseries(
            name, _make(grid, array, time_axis, unit, strategy, weights)
        )
    return collection
