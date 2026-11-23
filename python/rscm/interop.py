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

from collections.abc import Mapping, Sequence
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
    blocks: list[pd.DataFrame] = []
    strategies: dict[str, str] = {}
    weights: dict[str, list[float]] = {}
    axes: list[TimeAxis] = []
    for name, ts in _items(timeseries):
        grid = _grid_of(ts)
        values = np.asarray(ts.values(), dtype=np.float64).reshape(len(ts), -1)
        regions = list(REGIONS[grid])
        if not isinstance(ts, Timeseries):
            weights[name] = [float(w) for w in ts.grid.weights()]
            if global_mean:
                values = np.column_stack([ts.global_mean().values(), values])
                regions = ["World", *regions]
        strategies[name] = _name_of(ts.interpolation_strategy, _STRATEGIES)
        rows = [(name, ts.unit, region) for region in regions]
        blocks.append(
            pd.DataFrame(
                values.T,
                index=pd.MultiIndex.from_tuples(rows, names=INDEX),
                columns=ts.time_axis.values(),
            )
        )
        axes.append(ts.time_axis)
    if blocks:
        df = pd.concat(blocks).sort_index(axis="columns")
    else:
        df = pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=INDEX))
    df.columns.name = "time"
    common = bool(axes) and all(axis == axes[0] for axis in axes)
    df.attrs = {"interpolation_strategy": strategies, "grid_weights": weights}
    if common:
        df.attrs["time_bounds"] = [float(b) for b in axes[0].bounds()]
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
    return df.reorder_levels(list(INDEX))


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
    times: NDArray[np.float64], attrs: Mapping[Any, Any], time_axis: TimeAxis | None
) -> TimeAxis:
    if time_axis is not None:
        if not np.array_equal(time_axis.values(), times):
            msg = "time_axis must start a step at each time column"
            raise ValueError(msg)
        return time_axis
    bounds = np.asarray(attrs.get("time_bounds", []), dtype=np.float64)
    if len(bounds) == len(times) + 1 and np.array_equal(bounds[:-1], times):
        return TimeAxis.from_bounds(bounds)
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
    axis = _axis(_times(df.columns), attrs, time_axis)
    strategies = attrs.get("interpolation_strategy", {})
    weights = attrs.get("grid_weights", {})
    collection = TimeseriesCollection()
    for key, rows in df.groupby(level="variable", sort=False):
        name = str(key)
        units = rows.index.unique(level="unit")
        if len(units) > 1:
            msg = f"'{name}' has more than one unit: {list(units)}"
            raise ValueError(msg)
        by_region = rows.droplevel(["variable", "unit"])
        grid = _pick_grid(name, list(by_region.index))
        values = by_region.loc[list(REGIONS[grid])].to_numpy(dtype=np.float64).T
        strategy = _strategy_for(name, interpolation_strategy)
        if strategy is None:
            strategy = _value_of(
                strategies.get(name, "linear"), _STRATEGIES, "interpolation strategy"
            )
        grid_weight = (grid_weights or {}).get(grid, weights.get(name))
        collection.add_timeseries(
            name,
            _make(grid, values, axis, str(units[0]), strategy, grid_weight),
        )
    return collection


def _xarray() -> Any:
    try:
        import xarray as xr  # noqa: PLC0415
    except ImportError as error:
        msg = "xarray is not installed, install it with the rscm[xarray] extra"
        raise ImportError(msg) from error
    return xr


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
    A `time_bounds` coordinate holds the time axis when every series shares one.

    Raises
    ------
    ImportError
        If xarray is not installed.
    """
    xr = _xarray()
    df = to_dataframe(timeseries, global_mean=global_mean)
    present = set(df.index.get_level_values("region"))
    regions = [r for g in REGIONS.values() for r in g if r in present]
    times = df.columns.to_numpy(dtype=np.float64)
    data_vars = {}
    for key, rows in df.groupby(level="variable", sort=False):
        name = str(key)
        by_region = rows.droplevel(["variable", "unit"]).reindex(regions)
        grid = _pick_grid(name, list(rows.index.get_level_values("region")))
        attrs: dict[str, Any] = {
            "unit": str(rows.index.get_level_values("unit")[0]),
            "grid": _name_of(grid, _GRID_NAMES),
            "interpolation_strategy": df.attrs["interpolation_strategy"][name],
        }
        if name in df.attrs["grid_weights"]:
            attrs["grid_weights"] = df.attrs["grid_weights"][name]
        data_vars[name] = xr.DataArray(
            by_region.to_numpy(dtype=np.float64).T,
            dims=("time", "region"),
            attrs=attrs,
        )
    coords: dict[str, Any] = {"time": times, "region": regions}
    if "time_bounds" in df.attrs:
        bounds = np.asarray(df.attrs["time_bounds"])
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
    rows: list[tuple[str, str, str]] = []
    values: list[NDArray[np.float64]] = []
    strategies: dict[str, str] = {}
    weights: dict[str, list[float]] = {}
    for key, array in ds.data_vars.items():
        name = str(key)
        if "unit" not in array.attrs:
            msg = f"'{name}' has no 'unit' attribute"
            raise ValueError(msg)
        unit = str(array.attrs["unit"])
        if "region" not in array.dims:
            rows.append((name, unit, "World"))
            values.append(array.transpose("time").to_numpy())
        else:
            by_region = array.transpose("region", "time")
            if "grid" in array.attrs:
                keep = list(
                    REGIONS[_value_of(array.attrs["grid"], _GRID_NAMES, "grid")]
                )
            else:
                keep = [
                    str(r)
                    for r in by_region["region"].to_numpy()
                    if not np.isnan(by_region.sel(region=r)).all()
                ]
            for region in keep:
                rows.append((name, unit, region))
                values.append(by_region.sel(region=region).to_numpy())
        if "interpolation_strategy" in array.attrs:
            strategies[name] = str(array.attrs["interpolation_strategy"])
        if "grid_weights" in array.attrs:
            weights[name] = [float(w) for w in array.attrs["grid_weights"]]
    df = pd.DataFrame(
        np.vstack(values) if values else None,
        index=pd.MultiIndex.from_tuples(rows, names=INDEX),
        columns=ds["time"].to_numpy(),
    )
    df.attrs = {"interpolation_strategy": strategies, "grid_weights": weights}
    if "time_bounds" in ds.coords:
        bounds = ds["time_bounds"].to_numpy()
        df.attrs["time_bounds"] = [*bounds[:, 0], bounds[-1, 1]]
    return from_dataframe(
        df,
        time_axis=time_axis,
        interpolation_strategy=interpolation_strategy,
        grid_weights=grid_weights,
    )


def to_inputs(
    data: pd.DataFrame | TimeseriesCollection, time_axis: TimeAxis
) -> dict[str, tuple[NDArray[np.float64], str]]:
    """
    Return the `(values, unit)` inputs of `rscm.magicc` and `rscm.fair` `setup_model`.

    Each series is interpolated with its strategy onto `time_axis`,
    with one value per step start, and four or two columns for a grid variable.
    A DataFrame is read with `from_dataframe` first.
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
) -> TimeseriesCollection:
    """
    Read `(values, unit)` inputs on `time_axis` into a collection.

    The values' shape sets the grid: one value per step for a scalar,
    four columns for four-box and two for hemispheric.
    Values one step short, as FaIR's emissions are,
    hold their last value over the last step.
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
        collection.add_timeseries(
            name, _make(grid, array, time_axis, unit, strategy, None)
        )
    return collection
