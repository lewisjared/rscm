"""
Read FaIR configurations from FaIR's and fair-calibrate's CSV files.

The species CSV is FaIR's `species_configs_properties.csv`, one row per species.
The calibration CSV is fair-calibrate's `calibrated_constrained_parameters.csv`,
one row per posterior member.
Both are parsed as FaIR parses them, so the values are identical.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

from rscm.fair.config import (
    CLIMATE_CONFIGS,
    GAMMA_AUTOCORRELATION,
    PROPERTY_FLAGS,
    SPECIES_CONFIGS,
    FairConfig,
    Species,
)
from rscm.fair.validation import check_config
from rscm.validation import IssueKind, Validator

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping
    from pathlib import Path

# Calibration columns for stochastic runs and diagnostics, which RSCM does not use.
IGNORED_CALIBRATION = (
    "sigma_eta",
    "sigma_xi",
    "forcing_4co2",
    "seed",
    "use_seed",
    "stochastic_run",
)

MASS_ATMOSPHERE = 5.1352e18
MOLECULAR_WEIGHT_AIR = 28.97

_COLUMN = re.compile(r"(?P<name>[^\[]+)(\[(?P<index>.*)\])?")


def read_species(
    path: str | Path, species: Iterable[str] | None = None
) -> tuple[list[Species], dict[str, float]]:
    """Read species entries and the species configs without a species dimension.

    `species` selects and orders the rows, all of them by default.
    """
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(path, index_col=0)
    names = list(df.index) if species is None else list(species)
    n_gasboxes = sum(c.startswith("partition_fraction") for c in df.columns)
    entries = []
    for name in names:
        row = df.loc[name]
        entry: Species = {
            "name": name,
            "type": row["type"],
            "input_mode": row["input_mode"],
            **{flag: bool(row[flag]) for flag in PROPERTY_FLAGS},
            **{key: float(row[key]) for key in SPECIES_CONFIGS},
        }
        for key in ("partition_fraction", "unperturbed_lifetime"):
            entry[key] = [float(row[f"{key}{i}"]) for i in range(n_gasboxes)]
        entry["concentration_per_emission"] = 1 / (
            MASS_ATMOSPHERE / 1e18 * entry["molecular_weight"] / MOLECULAR_WEIGHT_AIR
        )
        entries.append(entry)

    def only(kind: str, column: str) -> float:
        rows = df.loc[df["type"] == kind, column]
        return float(rows.iloc[0]) if len(rows) else math.nan

    species_configs = {
        "aci_scale": only("aci", "aci_scale"),
        "lifetime_temperature_sensitivity": only(
            "ch4", "lifetime_temperature_sensitivity"
        ),
    }
    return entries, species_configs


def make_config(  # noqa: PLR0913
    species: list[Species],
    species_configs: Mapping[str, float],
    climate: Mapping[str, Any],
    *,
    ch4_method: str = "leach2021",
    ghg_method: str = "meinshausen2020",
    iirf_max: float = 100.0,
    br_cl_ods_potential: float = 45.0,
    ch4_min: float = 0.01,
) -> FairConfig:
    """Assemble a configuration, with FaIR's defaults for the options."""
    return {
        "options": {
            "ch4_method": ch4_method.lower(),
            "ghg_method": ghg_method.lower(),
            "iirf_max": iirf_max,
            "br_cl_ods_potential": br_cl_ods_potential,
            "ch4_min": ch4_min,
        },
        "climate": {"gamma_autocorrelation": GAMMA_AUTOCORRELATION, **climate},
        "species_configs": dict(species_configs),
        "species": species,
    }


def read_config(
    path: str | Path,
    climate: Mapping[str, Any],
    *,
    species: Iterable[str] | None = None,
    **options: Any,
) -> FairConfig:
    """Read a configuration from a species CSV, see `read_species` and `make_config`."""
    entries, species_configs = read_species(path, species)
    return make_config(entries, species_configs, climate, **options)


def read_calibration(
    species_path: str | Path,
    parameters_path: str | Path,
    *,
    members: Iterable[Hashable] | None = None,
    species: Iterable[str] | None = None,
    ch4_method: str = "thornhill2021",
    **options: Any,
) -> dict[Hashable, FairConfig]:
    """Read the configuration of each posterior member, like FaIR's `override_defaults`.

    The stochastic parameters are ignored, so the members run deterministically.
    Parameters of species missing from the configuration are skipped.
    Raises an `InputValidationError` for an unknown column and for every invalid member,
    see `rscm.fair.validation`.
    FaIR ignores `forcing_scale` for prescribed forcing,
    so scale it in the inputs as fair-calibrate does, see `setup_model`.
    """
    import pandas as pd  # noqa: PLC0415

    entries, species_configs = read_species(species_path, species)
    parameters = pd.read_csv(parameters_path, index_col=0)
    if members is not None:
        parameters = parameters.loc[list(members)]
    v = Validator()
    known = {*CLIMATE_CONFIGS, *SPECIES_CONFIGS, *species_configs}
    columns = []
    for column in parameters.columns:
        match = _COLUMN.fullmatch(str(column))
        if match is None or match["name"] not in {*known, *IGNORED_CALIBRATION}:
            v.add(IssueKind.UNKNOWN_KEY, str(column), "not a calibration parameter")
        elif match["name"] not in IGNORED_CALIBRATION:
            columns.append((column, match["name"], match["index"]))
    n_layers = sum(c.startswith("ocean_heat_capacity[") for c in parameters.columns)
    configs = {}
    for member, row in parameters.to_dict("index").items():
        member_species = [
            {
                **s,
                "partition_fraction": list(s["partition_fraction"]),
                "unperturbed_lifetime": list(s["unperturbed_lifetime"]),
            }
            for s in entries
        ]
        by_name = {s["name"]: s for s in member_species}
        member_species_configs = dict(species_configs)
        climate: dict[str, Any] = {
            "ocean_heat_capacity": [math.nan] * n_layers,
            "ocean_heat_transfer": [math.nan] * n_layers,
        }
        for column, name, index in columns:
            value = row[column]
            if name in CLIMATE_CONFIGS:
                if index is None:
                    climate[name] = float(value)
                else:
                    climate[name][int(index)] = float(value)
            elif index is None:
                member_species_configs[name] = float(value)
            elif index in by_name:
                by_name[index][name] = float(value)
        configs[member] = make_config(
            member_species,
            member_species_configs,
            climate,
            ch4_method=ch4_method,
            **options,
        )
        check_config(configs[member], v, at=f"{member}: ")
    v.raise_if_any()
    return configs
