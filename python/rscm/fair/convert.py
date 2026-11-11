"""Convert a configured `fair.FAIR` into an RSCM configuration and inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rscm.core import TimeAxis
from rscm.fair import naming
from rscm.fair.config import CLIMATE_CONFIGS, PROPERTY_FLAGS, FairConfig, Species
from rscm.fair.loaders import make_config
from rscm.fair.validation import check_config, check_inputs
from rscm.validation import IssueKind, Validator

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


def _has_default_ghg_forcing_offset(f: Any) -> bool:
    """Whether the run has no offset or the one FaIR calculates for itself."""
    if not hasattr(f, "ghg_forcing_offset"):
        return True
    from fair.forcing.ghg import meinshausen2020  # noqa: PLC0415

    if not hasattr(f, "_co2_indices"):
        f._make_indices()
    sc = f.species_configs
    default = meinshausen2020(
        sc["baseline_concentration"].data[None, None, ...],
        sc["forcing_reference_concentration"].data[None, None, ...],
        (sc["forcing_scale"].data * (1 + sc["tropospheric_adjustment"].data))[
            None, None, ...
        ],
        sc["greenhouse_gas_radiative_efficiency"].data[None, None, ...],
        f._co2_indices,
        f._ch4_indices,
        f._n2o_indices,
        f._minor_ghg_indices,
    )
    offset = np.asarray(f.ghg_forcing_offset)
    return offset.shape == default.shape and np.array_equal(
        offset, default, equal_nan=True
    )


def _check_options(f: Any, v: Validator, config: Hashable, scenario: Hashable) -> None:
    """Record the FaIR options that `from_fair` cannot carry over."""
    climate_configs = f.climate_configs.sel(config=config)
    if bool(climate_configs["stochastic_run"]):
        v.add(
            IssueKind.UNSUPPORTED,
            "climate.stochastic_run",
            "stochastic runs are not supported",
        )
    if f.temperature_prescribed:
        v.add(
            IssueKind.UNSUPPORTED,
            "temperature_prescribed",
            "prescribed temperatures are not supported",
        )
    if not _has_default_ghg_forcing_offset(f):
        v.add(
            IssueKind.UNSUPPORTED,
            "ghg_forcing_offset",
            "a user-set greenhouse gas forcing offset is not supported",
        )
    if np.isfinite(f.temperature.data[1:]).any():
        feedback = f.species_configs["forcing_temperature_feedback"].sel(config=config)
        for name, row in f.properties_df.iterrows():
            if row["input_mode"] == "forcing" and feedback.sel(specie=name) != 0:
                v.add(
                    IssueKind.UNSUPPORTED,
                    f"{name}.forcing",
                    "FAIR has run, which adds the temperature feedback "
                    "to its prescribed forcing in place, so convert it before f.run()",
                )
    _check_initial_state(f, v, config, scenario)


def _check_initial_state(
    f: Any, v: Validator, config: Hashable, scenario: Hashable
) -> None:
    """Record initial values other than FaIR's defaults, which RSCM starts from."""
    at = {"config": config, "scenario": scenario}
    first = {"timebounds": f.timebounds[0], **at}

    def expect(location: str, values: Any, expected: Any) -> None:
        values = np.atleast_1d(np.asarray(values, dtype=np.float64))
        if np.isnan(values).any():
            v.add(
                IssueKind.NON_FINITE,
                location,
                "has no initial value, so initialise it as FaIR needs",
            )
        elif not np.array_equal(values, np.broadcast_to(expected, values.shape)):
            v.add(
                IssueKind.UNSUPPORTED,
                location,
                f"starts at {values.tolist()}, but only FaIR's default initial state "
                f"{np.atleast_1d(expected).tolist()} is supported",
            )

    expect("temperature", f.temperature.sel(first).data, 0.0)
    expect(
        "ocean_heat_content_change",
        f.ocean_heat_content_change.sel(first).data,
        0.0,
    )
    baseline = f.species_configs["baseline_concentration"].sel(config=config)
    for name, row in f.properties_df.iterrows():
        ghg = bool(row["greenhouse_gas"]) and row["type"] in naming.GHG_TYPES
        mode = row["input_mode"]
        calculated_eesc = row["type"] == "eesc" and mode == "calculated"
        if (ghg and mode != "concentration") or calculated_eesc:
            expect(
                f"{name}.concentration",
                f.concentration.sel(first).sel(specie=name).data,
                baseline.sel(specie=name).data,
            )
        if ghg:
            expect(
                f"{name}.airborne_emissions",
                f.airborne_emissions.sel(first).sel(specie=name).data,
                0.0,
            )
        if ghg or mode == "emissions":
            expect(
                f"{name}.cumulative_emissions",
                f.cumulative_emissions.sel(first).sel(specie=name).data,
                0.0,
            )
        forcing = f.forcing.sel(first).sel(specie=name).data
        # FaIR's first total forcing skips NaN.
        if mode == "calculated" and not np.isnan(forcing).all():
            expect(f"{name}.forcing", forcing, 0.0)


def from_fair(
    f: Any,
    *,
    config: Hashable | None = None,
    scenario: Hashable | None = None,
) -> tuple[FairConfig, Inputs, TimeAxis]:
    """Return the configuration, inputs and time axis of a config and scenario of `f`.

    `f` is a `fair.FAIR` with its species, configs and inputs filled,
    and its initial state FaIR's default.
    `config` and `scenario` may be left out when `f` has only one.
    `rscm.fair.setup_model` sets up the model from them.
    FaIR ignores `forcing_scale` for prescribed forcing, so the inputs are unscaled.
    The species properties are read from `f.properties_df`, which FaIR runs from.
    Raises an `InputValidationError` for options `rscm.fair` lacks,
    such as stochastic runs or another initial state, and for invalid inputs.
    """
    config = _only(list(f.configs), "config", config)
    scenario = _only(list(f.scenarios), "scenario", scenario)
    v = Validator()
    _check_options(f, v, config, scenario)
    climate_configs = f.climate_configs.sel(config=config)
    species_configs = f.species_configs.sel(config=config)

    species: list[Species] = []
    inputs: Inputs = {}
    for name in f.species:
        properties = f.properties_df.loc[name]
        entry: Species = {
            "name": name,
            "type": properties["type"],
            "input_mode": properties["input_mode"],
            **{k: bool(properties[k]) for k in PROPERTY_FLAGS},
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
            # FaIR adds the temperature feedback to its prescribed forcing in place.
            inputs[variable] = (np.array(values, dtype=np.float64), unit)

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
    time_axis = TimeAxis.from_values(np.asarray(f.timebounds, dtype=np.float64))
    if check_config(fair_config, v):
        check_inputs(fair_config, inputs, time_axis, v)
    v.raise_if_any()
    return fair_config, inputs, time_axis
