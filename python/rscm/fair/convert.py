"""Convert a configured `fair.FAIR` into an RSCM configuration and inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rscm.fair import naming
from rscm.fair.config import CLIMATE_CONFIGS, FairConfig, Species
from rscm.fair.loaders import make_config

if TYPE_CHECKING:
    from collections.abc import Hashable

    from numpy.typing import NDArray

    Inputs = dict[str, tuple[NDArray[np.float64], str]]


def _plain(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim:
        return [float(v) for v in array]
    if array.dtype == bool:
        return bool(array)
    return float(array)


def _only(labels: list[Any], kind: str, given: Hashable | None) -> Hashable:
    if given is not None:
        return given
    if len(labels) != 1:
        msg = f"FAIR has {len(labels)} {kind}s, so choose one"
        raise ValueError(msg)
    return labels[0]  # type: ignore[no-any-return]


def from_fair(
    f: Any,
    *,
    config: Hashable | None = None,
    scenario: Hashable | None = None,
) -> tuple[FairConfig, Inputs, NDArray[np.float64]]:
    """Return the configuration, inputs and timebounds of a config and scenario of `f`.

    `f` is a `fair.FAIR` with its species, configs and inputs filled.
    `config` and `scenario` may be left out when `f` has only one.
    The result runs with `rscm.fair.run_fair`.
    FaIR ignores `forcing_scale` for prescribed forcing, so the inputs are unscaled.
    """
    config = _only(list(f.configs), "config", config)
    scenario = _only(list(f.scenarios), "scenario", scenario)
    climate_configs = f.climate_configs.sel(config=config)
    if bool(climate_configs["stochastic_run"]):
        msg = "Stochastic runs are not supported"
        raise NotImplementedError(msg)
    species_configs = f.species_configs.sel(config=config)

    species: list[Species] = []
    inputs: Inputs = {}
    for name in f.species:
        properties = f.properties[name]
        entry: Species = {
            "name": name,
            "type": properties["type"],
            "input_mode": properties["input_mode"],
            **{
                k: bool(v)
                for k, v in properties.items()
                if k not in ("type", "input_mode")
            },
        }
        for var, data in species_configs.data_vars.items():
            if "specie" in data.dims:
                entry[str(var)] = _plain(data.sel(specie=name).values)
        species.append(entry)

        prescribed = naming.prescribed_variable(entry)
        if prescribed is not None:
            variable, unit = prescribed
            values = getattr(f, entry["input_mode"]).sel(
                specie=name, scenario=scenario, config=config
            )
            inputs[variable] = (np.asarray(values, dtype=np.float64), unit)

    fair_config = make_config(
        species,
        {
            str(var): _plain(data.values)
            for var, data in species_configs.data_vars.items()
            if "specie" not in data.dims
        },
        {name: _plain(climate_configs[name].values) for name in CLIMATE_CONFIGS},
        ch4_method=f.ch4_method,
        ghg_method=f.ghg_method,
        iirf_max=float(f.iirf_max),
        br_cl_ods_potential=float(f.br_cl_ods_potential),
        ch4_min=float(f._ch4_min),
    )
    return fair_config, inputs, np.asarray(f.timebounds, dtype=np.float64)
