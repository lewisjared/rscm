"""Run ensembles of FaIR configurations and scenarios, with results in FaIR's layout."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rscm import names
from rscm.fair import naming
from rscm.fair.config import emission_drivers, is_ghg, species_outputs
from rscm.fair.run import _on_timebounds, _setup_model
from rscm.fair.validation import check_config, check_inputs, check_prescribed_scales
from rscm.spec import run_all
from rscm.validation import Validator

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    import xarray as xr
    from numpy.typing import ArrayLike, NDArray

    from rscm.core import TimeAxis, TimeseriesCollection
    from rscm.fair.config import FairConfig, Species

    Inputs = Mapping[str, tuple[ArrayLike, str]]

# FaIR attributes on timebounds with a species dimension.
SPECIES_ATTRIBUTES = (
    "concentration",
    "forcing",
    "airborne_emissions",
    "cumulative_emissions",
)
# FaIR attributes on timebounds without a species or layer dimension.
SCALAR_ATTRIBUTES = {
    "forcing_sum": names.ERF,
    "ocean_heat_content_change": names.OCEAN_HEAT_CONTENT,
    "toa_imbalance": names.NET_ENERGY_IMBALANCE,
}


def _values(
    outputs: TimeseriesCollection, variable: str, inputs: Inputs, n_bounds: int
) -> NDArray[np.float64]:
    """Values of a variable on the timebounds, from the inputs if nothing reads it."""
    series = outputs.get_timeseries_by_name(variable)
    if series is not None:
        return np.asarray(series.values())
    if variable in inputs:
        return _on_timebounds(variable, inputs[variable][0], n_bounds)
    msg = f"The model has no {variable!r}"
    raise KeyError(msg)


def _emissions(config: FairConfig, s: Species) -> list[str] | None:
    """Variables summing to a species' emissions."""
    if s["input_mode"] == "concentration" and is_ghg(s):
        return [naming.inverse_emissions(s["name"])]
    return emission_drivers(config, s) or None


def _validate(
    configs: Mapping[Hashable, FairConfig],
    scenarios: Mapping[str, Inputs],
    time_axis: TimeAxis,
    *,
    scale_prescribed_forcing: bool,
) -> None:
    """Check every configuration and scenario, so a bad one fails before any run."""
    v = Validator()
    # Inputs are checked for every configuration with well-formed species.
    valid = [name for name in configs if check_config(configs[name], v, at=f"{name}: ")]
    if scale_prescribed_forcing:
        for name in valid:
            check_prescribed_scales(configs[name], v, at=f"{name}: ")
    for scenario, inputs in scenarios.items():
        for name in valid:
            check_inputs(configs[name], inputs, time_axis, v, at=f"{scenario}: ")
    # Configurations with the same species share each input issue.
    v.issues[:] = list(dict.fromkeys(v.issues))
    v.raise_if_any()


def run_ensemble(
    configs: Mapping[Hashable, FairConfig],
    scenarios: Mapping[str, Inputs],
    time_axis: TimeAxis,
    *,
    scale_prescribed_forcing: bool = False,
    n_threads: int | None = None,
) -> xr.Dataset:
    """Run every configuration on every scenario and collect the results like FaIR.

    The configurations must share their species and layers.
    `scenarios` maps each scenario to its inputs, see `rscm.fair.setup_model`.
    The configurations of each scenario run in parallel on ``n_threads`` threads,
    or one per core.
    Every configuration and scenario is validated before the first run.
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

    _validate(
        configs,
        scenarios,
        time_axis,
        scale_prescribed_forcing=scale_prescribed_forcing,
    )
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

    bounds = np.asarray(time_axis.values(), dtype=np.float64)
    coords: dict[str, Any] = {
        "timebounds": bounds,
        "timepoints": (bounds[:-1] + bounds[1:]) / 2,
        "scenario": list(scenarios),
        "config": config_names,
        "specie": species,
        "layer": np.arange(n_layers),
    }
    by_species = ("scenario", "config", "specie")
    dims = {
        "temperature": ("timebounds", "scenario", "config", "layer"),
        "emissions": ("timepoints", *by_species),
        **{a: ("timebounds", *by_species) for a in SPECIES_ATTRIBUTES},
        **{a: ("timebounds", "scenario", "config") for a in SCALAR_ATTRIBUTES},
    }
    sizes = {name: len(values) for name, values in coords.items()}
    data = {
        name: np.full([sizes[d] for d in names], np.nan) for name, names in dims.items()
    }

    # FaIR has zero airborne emissions of the species without a gas cycle.
    data["airborne_emissions"][..., [not is_ghg(s) for s in first["species"]]] = 0.0
    n_bounds = len(bounds)
    timestep = bounds[1] - bounds[0]
    index = {s: k for k, s in enumerate(species)}
    species_series = {}
    emission_series = {}
    accumulated = {}
    for name in config_names:
        config = configs[name]
        species_series[name] = [
            (attribute, index[s], variable)
            for attribute, variables in species_outputs(config).items()
            for s, variable in variables.items()
        ]
        emission_series[name] = [
            (index[s["name"]], found)
            for s in config["species"]
            if (found := _emissions(config, s)) is not None
        ]
        # The gas cycle accumulates greenhouse gas emissions.
        accumulated[name] = [
            k for k, _ in emission_series[name] if not is_ghg(config["species"][k])
        ]

    for i, inputs in enumerate(scenarios.values()):
        setups = [
            _setup_model(
                configs[name],
                inputs,
                time_axis,
                scale_prescribed_forcing=scale_prescribed_forcing,
            )
            for name in config_names
        ]
        models = run_all(setups, n_threads=n_threads)
        for j, (name, model) in enumerate(zip(config_names, models, strict=True)):
            outputs = model.timeseries()
            for layer in range(n_layers):
                data["temperature"][:, i, j, layer] = _values(
                    outputs, naming.layer_temperature(layer), inputs, n_bounds
                )
            for attribute, variable in SCALAR_ATTRIBUTES.items():
                data[attribute][:, i, j] = _values(outputs, variable, inputs, n_bounds)
            for attribute, k, variable in species_series[name]:
                data[attribute][:, i, j, k] = _values(
                    outputs, variable, inputs, n_bounds
                )
            for k, variables in emission_series[name]:
                data["emissions"][:, i, j, k] = np.sum(
                    [_values(outputs, v, inputs, n_bounds)[:-1] for v in variables],
                    axis=0,
                )
            for k in accumulated[name]:
                # FaIR starts from its initial cumulative emissions, which from_fair
                # requires to be 0.
                cumulative = data["cumulative_emissions"][:, i, j, k]
                cumulative[0] = 0.0
                cumulative[1:] = np.cumsum(data["emissions"][:, i, j, k]) * timestep

    return xr.Dataset(
        {name: (dims[name], array) for name, array in data.items()}, coords=coords
    )
