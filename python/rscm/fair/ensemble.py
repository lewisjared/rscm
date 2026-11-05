"""Run ensembles of FaIR configurations and scenarios, with results in FaIR's layout."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rscm.fair import naming
from rscm.fair.config import emission_drivers, is_ghg, species_outputs
from rscm.fair.run import run_fair

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    import xarray as xr
    from numpy.typing import ArrayLike, NDArray

    from rscm.core import TimeseriesCollection
    from rscm.fair.config import FairConfig, Species

    Inputs = Mapping[str, tuple[ArrayLike, str]]

#: FaIR attributes on timebounds with a species dimension.
SPECIES_ATTRIBUTES = (
    "concentration",
    "forcing",
    "airborne_emissions",
    "cumulative_emissions",
)
#: FaIR attributes on timebounds without a species or layer dimension.
SCALAR_ATTRIBUTES = {
    "forcing_sum": naming.TOTAL_FORCING,
    "ocean_heat_content_change": naming.OCEAN_HEAT_CONTENT_CHANGE,
    "toa_imbalance": naming.ENERGY_IMBALANCE,
}


def _values(outputs: TimeseriesCollection, variable: str) -> NDArray[np.float64]:
    series = outputs.get_timeseries_by_name(variable)
    if series is None:
        msg = f"The model has no {variable!r}"
        raise KeyError(msg)
    return np.asarray(series.values())


def _emissions(config: FairConfig, s: Species) -> tuple[list[str], slice] | None:
    """Variables summing to a species' emissions and the slice onto timepoints."""
    if s["input_mode"] == "concentration" and is_ghg(s):
        # The inverse emissions of a step are written at its end.
        return [naming.emissions(s["name"])], slice(1, None)
    drivers = emission_drivers(config, s)
    return (drivers, slice(None, -1)) if drivers else None


def run_ensemble(
    configs: Mapping[Hashable, FairConfig],
    scenarios: Mapping[str, Inputs],
    timebounds: ArrayLike,
    *,
    scale_prescribed_forcing: bool = False,
) -> xr.Dataset:
    """Run every configuration on every scenario and collect the results like FaIR.

    The configurations must share their species and layers.
    `scenarios` maps each scenario to its inputs, see `rscm.fair.build_model`.
    The dataset has FaIR's `temperature`, `emissions`, `concentration`, `forcing`,
    `forcing_sum`, `airborne_emissions`, `cumulative_emissions`,
    `ocean_heat_content_change` and `toa_imbalance`,
    with NaN where a species has no value, as in FaIR.
    It needs xarray, which the `fair` extra installs.
    """
    try:
        import xarray as xr  # noqa: PLC0415
    except ImportError as error:
        msg = "run_ensemble needs xarray: pip install 'rscm[fair]'"
        raise ImportError(msg) from error

    config_names = list(configs)
    first = configs[config_names[0]]
    species = [s["name"] for s in first["species"]]
    n_layers = len(first["climate"]["ocean_heat_capacity"])
    for name in config_names[1:]:
        other = configs[name]
        if [s["name"] for s in other["species"]] != species or len(
            other["climate"]["ocean_heat_capacity"]
        ) != n_layers:
            msg = (
                f"Config {name!r} has other species or layers than {config_names[0]!r}"
            )
            raise ValueError(msg)

    bounds = np.asarray(timebounds, dtype=np.float64)
    coords: dict[str, Any] = {
        "timebounds": bounds,
        "timepoints": (bounds[:-1] + bounds[1:]) / 2,
        "scenario": list(scenarios),
        "config": config_names,
        "specie": species,
        "layer": np.arange(n_layers),
    }
    by_specie = ("scenario", "config", "specie")
    dims = {
        "temperature": ("timebounds", "scenario", "config", "layer"),
        "emissions": ("timepoints", *by_specie),
        **{a: ("timebounds", *by_specie) for a in SPECIES_ATTRIBUTES},
        **{a: ("timebounds", "scenario", "config") for a in SCALAR_ATTRIBUTES},
    }
    sizes = {name: len(values) for name, values in coords.items()}
    data = {
        name: np.full([sizes[d] for d in names], np.nan) for name, names in dims.items()
    }

    # FaIR has zero airborne emissions of the species without a gas cycle.
    data["airborne_emissions"][..., [not is_ghg(s) for s in first["species"]]] = 0.0
    timestep = bounds[1] - bounds[0]
    index = {specie: k for k, specie in enumerate(species)}
    species_series = {}
    emission_series = {}
    accumulated = {}
    for name in config_names:
        config = configs[name]
        species_series[name] = [
            (attribute, index[specie], variable)
            for attribute, variables in species_outputs(config).items()
            for specie, variable in variables.items()
        ]
        emission_series[name] = [
            (index[s["name"]], *found)
            for s in config["species"]
            if (found := _emissions(config, s)) is not None
        ]
        # The gas cycle accumulates greenhouse gas emissions.
        accumulated[name] = [
            k for k, *_ in emission_series[name] if not is_ghg(config["species"][k])
        ]

    for i, inputs in enumerate(scenarios.values()):
        for j, name in enumerate(config_names):
            outputs = run_fair(
                configs[name],
                inputs,
                bounds,
                scale_prescribed_forcing=scale_prescribed_forcing,
            ).timeseries()
            for layer in range(n_layers):
                data["temperature"][:, i, j, layer] = _values(
                    outputs, naming.layer_temperature(layer)
                )
            for attribute, variable in SCALAR_ATTRIBUTES.items():
                data[attribute][:, i, j] = _values(outputs, variable)
            for attribute, k, variable in species_series[name]:
                data[attribute][:, i, j, k] = _values(outputs, variable)
            for k, variables, timepoints in emission_series[name]:
                data["emissions"][:, i, j, k] = np.sum(
                    [_values(outputs, v)[timepoints] for v in variables], axis=0
                )
            for k in accumulated[name]:
                cumulative = data["cumulative_emissions"][:, i, j, k]
                cumulative[0] = 0.0
                cumulative[1:] = (
                    np.cumsum(data["emissions"][:, i, j, k]) * timestep + cumulative[0]
                )

    return xr.Dataset(
        {name: (dims[name], array) for name, array in data.items()}, coords=coords
    )
