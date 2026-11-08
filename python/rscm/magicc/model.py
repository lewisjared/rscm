"""
Build MAGICC models from a configuration and prescribed inputs.

A configuration is a plain dict:

- ``components``: the Rust parameters of each component, keyed by component name
  such as ``"TerrestrialCarbon"``.
- ``initial_values``: the value at the first time bound of each state variable,
  apart from those of ClimateUDEB and ForcingAggregation, which the build calculates
  from the forcing at the first time bound.
- ``initial_forcing`` (optional): the forcing of ForcingAggregation agents
  at the first time bound, keyed by agent variable,
  with four values for a regional agent and one otherwise.
  Other agents start from their prescribed forcing, or zero.

The inputs map each prescribed variable to its values and unit,
with one value per time bound.
Four-box values have four columns and hemispheric values two.

[rscm.magicc.emissions_driven][] and [rscm.magicc.concentration_driven][]
build the two models from the same configuration.
[from_magicc7][rscm.magicc.full.from_magicc7] translates a MAGICC7 configuration
into it.
"""

from __future__ import annotations

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
    ATMOSPHERIC_CONCENTRATIONS_CO2,
    EFFICACY_ADJUSTED_ERF,
    EQUILIBRIUM_CLIMATE_SENSITIVITY,
    SURFACE_AIR_TEMPERATURE_CHANGE,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray

    from rscm._lib.core import RustComponent
    from rscm.core import Model

    Inputs = Mapping[str, tuple[ArrayLike, str]]

__all__ = [
    "BUILDERS",
    "ModelConfig",
    "build",
    "build_from_components",
    "check_axis",
    "run_from_components",
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

# MAGICC7 feeds back last year's temperature, CO2 and climate sensitivity.
_LAG = (SURFACE_AIR_TEMPERATURE_CHANGE,)
_LAG_FOR = (
    (ATMOSPHERIC_CONCENTRATIONS_CO2, "TerrestrialCarbon"),
    (ATMOSPHERIC_CONCENTRATIONS_CO2, "OceanCarbon"),
    (EQUILIBRIUM_CLIMATE_SENSITIVITY, "ForcingAggregation"),
)


def with_lags(builder: ModelBuilder) -> ModelBuilder:
    """Add MAGICC7's feedback lags to a builder of the MAGICC components."""
    for variable in _LAG:
        builder.with_lag(variable)
    for variable, consumer in _LAG_FOR:
        builder.with_lag_for(variable, consumer)
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
    if any(d.grid == GridType.FourBox for c in components for d in c.definitions()):
        builder.with_schema(_schema(components, prescribed))
        builder.with_grid_weights(GridType.FourBox, list(pr.area_fractions()))
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


def run_from_components(
    time_axis: TimeAxis,
    components: Sequence[RustComponent],
    **kwargs: Any,
) -> Model:
    """Build and run components and return the model, see `build_from_components`."""
    model = build_from_components(time_axis, components, **kwargs)
    model.run()
    return model


def check_axis(time_axis: TimeAxis) -> None:
    """Raise `ValueError` unless ``time_axis`` has at least two annual steps."""
    if len(time_axis) < 2 or np.any(np.diff(time_axis.bounds()) != 1.0):  # noqa: PLR2004
        msg = "time_axis must have at least two annual steps"
        raise ValueError(msg)


def _on_axis(inputs: Inputs, n: int) -> dict[str, tuple[NDArray[np.float64], str]]:
    out = {}
    for name, (values, unit) in inputs.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim not in (1, 2) or array.shape[0] != n:
            msg = f"input {name!r} has shape {array.shape}, expected {n} time bounds"
            raise ValueError(msg)
        out[name] = (array, unit)
    return out


def _builders(
    config: ModelConfig, required: Collection[str], optional: Collection[str]
) -> dict[str, Any]:
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
        name: builder.from_parameters(given[name])
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


def build(
    config: ModelConfig,
    inputs: Inputs,
    time_axis: TimeAxis,
    *,
    required: Collection[str],
    optional: Collection[str] = (),
) -> Model:
    """Build an unrun model of the ``required`` and any ``optional`` components.

    ``time_axis`` has annual steps.
    """
    check_axis(time_axis)
    prescribed = _on_axis(inputs, len(time_axis))
    builders = _builders(config, required, optional)
    components = {name: builder.build() for name, builder in builders.items()}
    agents = config["components"]["ForcingAggregation"]["agents"]
    first = {name: values[0] for name, (values, _) in prescribed.items()}
    total, boxes = builders["ForcingAggregation"].aggregate(
        _initial_forcing(agents, first, config.get("initial_forcing", {}))
    )
    sea_level = {}
    if "SeaLevelRise" in components:
        sea_level = {
            d.name: 0.0
            for d in components["SeaLevelRise"].definitions()
            if d.role == VariableRole.Output
        }
    return build_from_components(
        time_axis,
        list(components.values()),
        prescribed=prescribed,
        initial_values={
            **config["initial_values"],
            **sea_level,
            **builders["ClimateUDEB"].initial_values(total),
        },
        initial_grid_values={EFFICACY_ADJUSTED_ERF: boxes},
        lags=with_lags,
    )
