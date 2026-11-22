"""
Build MAGICC models from a configuration and prescribed inputs.

A configuration is a plain dict:

- ``components``: the Rust parameters of each component, keyed by component name
  such as ``"TerrestrialCarbon"``.
- ``initial_values``: the value at the first time bound of each state variable,
  apart from those of ClimateUDEB and ForcingAggregation, which `setup` calculates
  from the forcing at the first time bound.
  CH4Oxidation's burdens default to all CH4 natural at the initial CH4 concentration.
- ``initial_forcing`` (optional): the forcing of ForcingAggregation agents
  at the first time bound, keyed by agent variable,
  with four values for a regional agent and one otherwise.
  Other agents start from their prescribed forcing, or zero.

The inputs map each prescribed variable to its values and unit,
with one value per time bound.
Four-box values have four columns and hemispheric values two.

`setup` turns a configuration and inputs into a [ModelSetup][rscm.spec.ModelSetup].
[rscm.magicc.emissions_driven][] and [rscm.magicc.concentration_driven][]
set up the two models from the same configuration.
[from_magicc7][rscm.magicc.full.from_magicc7] translates a MAGICC7 configuration
into it.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

import numpy as np

from rscm._lib.magicc import (
    CH4ChemistryBuilder,
    CH4OxidationBuilder,
    ClimateUDEBBuilder,
    CO2BudgetBuilder,
    ForcingAggregationBuilder,
    GHGForcingBuilder,
    HalocarbonChemistryBuilder,
    N2OChemistryBuilder,
    OceanCarbonBuilder,
    OzoneForcingBuilder,
    PermafrostBuilder,
    SeaLevelRiseBuilder,
    TerrestrialCarbonBuilder,
)
from rscm.core import (
    FourBoxTimeseries,
    GridType,
    HemisphericTimeseries,
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
    VariableRole,
    VariableSchema,
)
from rscm.magicc import prescribed as pr
from rscm.names import (
    ATMOSPHERIC_CONCENTRATIONS_CH4,
    ATMOSPHERIC_CONCENTRATIONS_CO2,
    EFFICACY_ADJUSTED_ERF,
    EQUILIBRIUM_CLIMATE_SENSITIVITY,
    SURFACE_AIR_TEMPERATURE_CHANGE,
)
from rscm.spec import ComponentSpec, Lag, ModelSetup, ModelSpec
from rscm.validation import InputValidationError, Issue, IssueKind, Validator

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray

    from rscm._lib.core import RustComponent
    from rscm.core import Model

    Inputs = Mapping[str, tuple[ArrayLike, str]]

__all__ = [
    "BUILDERS",
    "LAGS",
    "ModelConfig",
    "build_from_components",
    "check_axis",
    "check_on_axis",
    "component_specs",
    "setup",
    "with_lags",
]

ModelConfig = dict[str, Any]

BUILDERS: dict[str, Any] = {
    "Permafrost": PermafrostBuilder,
    "CH4Chemistry": CH4ChemistryBuilder,
    "CH4Oxidation": CH4OxidationBuilder,
    "N2OChemistry": N2OChemistryBuilder,
    "HalocarbonChemistry": HalocarbonChemistryBuilder,
    "TerrestrialCarbon": TerrestrialCarbonBuilder,
    "OceanCarbon": OceanCarbonBuilder,
    "CO2Budget": CO2BudgetBuilder,
    "GHGForcing": GHGForcingBuilder,
    "OzoneForcing": OzoneForcingBuilder,
    "ForcingAggregation": ForcingAggregationBuilder,
    "ClimateUDEB": ClimateUDEBBuilder,
    "SeaLevelRise": SeaLevelRiseBuilder,
}
"""Builder of each component a configuration can hold, in the order they are added."""

LAGS = (
    Lag(SURFACE_AIR_TEMPERATURE_CHANGE),
    Lag(ATMOSPHERIC_CONCENTRATIONS_CO2, into=["TerrestrialCarbon", "OceanCarbon"]),
    Lag(EQUILIBRIUM_CLIMATE_SENSITIVITY, into=["ForcingAggregation"]),
)
"""MAGICC7's feedback lags on last year's temperature, CO2 and climate sensitivity."""


def with_lags(builder: ModelBuilder) -> ModelBuilder:
    """Add MAGICC7's feedback `LAGS` to a builder of the MAGICC components."""
    for lag in LAGS:
        lag.add_to(builder)
    return builder


def _grid(values: NDArray[np.float64]) -> GridType:
    if values.ndim == 1:
        return GridType.Scalar
    return GridType.FourBox if values.shape[1] == 4 else GridType.Hemispheric  # noqa: PLR2004


_GRID_RESOLUTION = (GridType.Scalar, GridType.Hemispheric, GridType.FourBox)


def _schema(
    components: Sequence[RustComponent],
    prescribed: Mapping[str, tuple[NDArray[np.float64], str]],
) -> VariableSchema:
    variables = {
        name: (unit, _grid(np.asarray(values)))
        for name, (values, unit) in prescribed.items()
    }
    for component in components:
        for d in component.definitions():
            _, grid = variables.get(d.name, (d.unit, d.grid))
            variables[d.name] = (d.unit, max(grid, d.grid, key=_GRID_RESOLUTION.index))
    schema = VariableSchema()
    for name, (unit, grid) in variables.items():
        schema.add_variable(name, unit, grid)
    return schema


def _grid_setup(
    components: Sequence[RustComponent],
    prescribed: Mapping[str, tuple[NDArray[np.float64], str]],
) -> tuple[VariableSchema | None, dict[GridType, list[float]]]:
    """Return the schema and grid weights, which only four-box components need."""
    if not any(d.grid == GridType.FourBox for c in components for d in c.definitions()):
        return None, {}
    return _schema(components, prescribed), {
        GridType.FourBox: list(pr.area_fractions())
    }


def _timeseries(
    values: NDArray[np.float64], axis: TimeAxis, unit: str
) -> Timeseries | FourBoxTimeseries | HemisphericTimeseries:
    grid = _grid(values)
    if grid == GridType.FourBox:
        return FourBoxTimeseries(values, axis, unit, InterpolationStrategy.Linear)
    if grid == GridType.Hemispheric:
        return HemisphericTimeseries(values, axis, unit, InterpolationStrategy.Linear)
    return Timeseries(values, axis, unit, InterpolationStrategy.Linear)


def build_from_components(  # noqa: PLR0913
    time_axis: TimeAxis,
    components: Sequence[RustComponent],
    *,
    prescribed: Mapping[str, tuple[NDArray[np.float64], str]] = {},
    initial_values: Mapping[str, float] = {},
    initial_grid_values: Mapping[str, Sequence[float]] = {},
    lags: Callable[[ModelBuilder], object] | None = None,
) -> Model:
    """Build an unrun model of components with prescribed inputs.

    ``prescribed`` maps each name to its values and unit.
    The values' shape sets the grid: one value per time bound for a scalar,
    four columns for four-box and two for hemispheric.
    Scalar readers of a four-box variable see its area-weighted mean.
    ``lags`` adds feedback lags to the builder, such as `with_lags`.
    """
    builder = ModelBuilder().with_time_axis(time_axis)
    schema, grid_weights = _grid_setup(components, prescribed)
    if schema is not None:
        builder.with_schema(schema)
    for grid, weights in grid_weights.items():
        builder.with_grid_weights(grid, weights)
    for component in components:
        builder.with_component(component)
    if lags is not None:
        lags(builder)
    for name, (values, unit) in prescribed.items():
        builder.with_prescribed_variable(
            name, _timeseries(np.asarray(values, dtype=np.float64), time_axis, unit)
        )
    builder.with_initial_values(dict(initial_values))
    builder.with_initial_grid_values(
        {k: list(v) for k, v in initial_grid_values.items()}
    )
    return builder.build()


def check_axis(time_axis: TimeAxis) -> None:
    """Raise `InputValidationError` unless ``time_axis`` has two or more annual steps.

    The bounds must be whole years, as MAGICC7 steps from January to January.
    """
    bounds = np.asarray(time_axis.bounds())
    if (
        len(time_axis) < 2  # noqa: PLR2004
        or np.any(np.diff(bounds) != 1.0)
        or np.any(bounds != np.round(bounds))
    ):
        raise InputValidationError(
            [
                Issue(
                    IssueKind.SHAPE,
                    "time_axis",
                    "must have at least two steps of one year, on whole years",
                )
            ]
        )


def check_on_axis(
    v: Validator, name: str, values: Any, time: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """Return ``values`` as an array, or None after reporting a non-numeric value.

    A pandas object's index must equal ``time``, so values are not read out of order.
    """
    index = getattr(values, "index", None)
    if index is not None and not callable(index):
        expected = f"the time axis values {time[0]:g} to {time[-1]:g}"
        if len(index) == 0:
            v.add(IssueKind.SHAPE, name, f"is empty, not {expected}")
            return None
        try:
            on_axis = len(index) == len(time) and np.array_equal(
                np.asarray(index, float), time
            )
        except (TypeError, ValueError):
            on_axis = False
        if not on_axis:
            v.add(
                IssueKind.SHAPE,
                name,
                f"index runs {index[0]} to {index[-1]}, not {expected}",
            )
            return None
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        v.add(IssueKind.TYPE, name, "values must be numbers")
        return None


def _on_axis(
    v: Validator, inputs: Inputs, time: NDArray[np.float64]
) -> dict[str, tuple[NDArray[np.float64], str]]:
    n = len(time)
    out = {}
    for name, (values, unit) in inputs.items():
        array = check_on_axis(v, name, values, time)
        if array is None:
            continue
        if not v.check(
            array.ndim in (1, 2)
            and array.shape[0] == n
            and (array.ndim == 1 or array.shape[1] in (2, 4)),
            IssueKind.SHAPE,
            name,
            f"has shape {array.shape}, expected {n} time bounds"
            " and 2 or 4 columns for a regional variable",
        ):
            continue
        if v.finite(name, array):
            out[name] = (array, unit)
    return out


def _check_read(
    v: Validator, inputs: Collection[str], components: Collection[RustComponent]
) -> None:
    read = {d.name for c in components for d in c.definitions()}
    for name in sorted(set(inputs) - read):
        close = difflib.get_close_matches(name, read, n=1)
        hint = f", did you mean {close[0]!r}?" if close else ""
        v.add(IssueKind.UNKNOWN_KEY, name, f"no component reads this input{hint}")


def component_specs(
    config: ModelConfig, *, required: Collection[str], optional: Collection[str] = ()
) -> dict[str, ComponentSpec]:
    """Return the ``required`` and any ``optional`` components a configuration holds.

    They are in the order of `BUILDERS`.
    Raises `ValueError` if a required component is missing or another is unknown.
    """
    given = config["components"]
    missing = sorted(set(required) - set(given))
    if missing:
        msg = f"configuration has no parameters for components: {', '.join(missing)}"
        raise ValueError(msg)
    unknown = sorted(set(given) - {*required, *optional})
    if unknown:
        msg = f"components not in this model: {', '.join(unknown)}"
        raise ValueError(msg)
    return {
        name: ComponentSpec(builder, given[name])
        for name, builder in BUILDERS.items()
        if name in given
    }


def _initial_forcing(
    agents: Sequence[Mapping[str, Any]],
    first: Mapping[str, NDArray[np.float64]],
    given: Mapping[str, Sequence[float]],
) -> list[list[float]]:
    """Return the forcing of each agent at the first time bound."""
    out = []
    for agent in agents:
        variable = agent["variable"]
        if variable in given:
            out.append([float(v) for v in given[variable]])
            continue
        regional = agent.get("regional", False)
        total = np.zeros(4 if regional else 1)
        for name in agent.get("sum_of") or [variable]:
            if name in first:
                total = total + (first[name] if regional else _global(first[name]))
        out.append(total.tolist())
    return out


def _global(values: NDArray[np.float64]) -> float:
    if values.ndim == 0:
        return float(values)
    if len(values) == 4:  # noqa: PLR2004
        return float(values @ pr.area_fractions())
    return float(values.mean())


def setup(  # noqa: PLR0913
    config: ModelConfig,
    inputs: Inputs,
    time_axis: TimeAxis,
    *,
    required: Collection[str],
    optional: Collection[str] = (),
    name: str = "MAGICC",
) -> ModelSetup:
    """Return the setup of the ``required`` and any ``optional`` components.

    ``time_axis`` has annual steps.
    ClimateUDEB's initial values and the initial efficacy-adjusted forcing
    come from the forcing at the first time bound.
    """
    check_axis(time_axis)
    v = Validator()
    for key in ("components", "initial_values"):
        v.check(key in config, IssueKind.MISSING, key, "not in the configuration")
    v.raise_if_any()
    time = np.asarray(time_axis.values(), dtype=np.float64)
    prescribed = _on_axis(v, inputs, time)
    specs = component_specs(config, required=required, optional=optional)
    components = {key: spec.build() for key, spec in specs.items()}
    _check_read(v, inputs, components.values())
    v.raise_if_any()
    first = {key: values[0] for key, (values, _) in prescribed.items()}
    aggregation = specs["ForcingAggregation"]
    total, boxes = aggregation.configure().aggregate(
        _initial_forcing(
            aggregation.parameters["agents"], first, config.get("initial_forcing", {})
        )
    )
    sea_level = {}
    if "SeaLevelRise" in components:
        sea_level = {
            d.name: 0.0
            for d in components["SeaLevelRise"].definitions()
            if d.role == VariableRole.Output
        }
    schema, grid_weights = _grid_setup(list(components.values()), prescribed)
    burdens = {}
    ch4 = config["initial_values"].get(ATMOSPHERIC_CONCENTRATIONS_CH4)
    if "CH4Oxidation" in specs and ch4 is not None:
        burdens = specs["CH4Oxidation"].configure().initial_values(ch4)
    spec = ModelSpec(
        components=specs,
        lags=LAGS,
        initial_values={
            **burdens,
            **config["initial_values"],
            **sea_level,
            **specs["ClimateUDEB"].configure().initial_values(total),
        },
        initial_grid_values={EFFICACY_ADJUSTED_ERF: boxes},
        grid_weights=grid_weights,
        schema=schema,
        name=name,
    )
    return spec.bind(
        {
            key: _timeseries(values, time_axis, unit)
            for key, (values, unit) in prescribed.items()
        },
        time_axis,
    )
