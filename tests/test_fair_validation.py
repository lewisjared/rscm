"""Pre-run validation of `rscm.fair` configurations and inputs."""

from __future__ import annotations

import copy
import math
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from rscm.core import TimeAxis
from rscm.fair import (
    components,
    load_config,
    naming,
    read_calibration,
    run_ensemble,
    setup_model,
)
from rscm.fair.config import linear_drivers
from rscm.fair.validation import validate
from rscm.validation import InputValidationError, IssueKind

if TYPE_CHECKING:
    from collections.abc import Callable

DATA = Path(__file__).parent / "regression" / "data" / "fair"
CALIBRATION = DATA / "calibration"
N_BOUNDS = 11


@cache
def _ssp245() -> dict[str, Any]:
    return load_config(DATA / "ssp245.json")


@pytest.fixture
def config() -> dict[str, Any]:
    return copy.deepcopy(_ssp245())


def _time_axis() -> TimeAxis:
    return TimeAxis.from_values(np.arange(1750.0, 1750.0 + N_BOUNDS))


def _inputs(config: dict[str, Any]) -> dict[str, tuple[Any, str]]:
    inputs = {}
    for s in config["species"]:
        prescribed = naming.prescribed_variable(s)
        if prescribed is not None:
            variable, unit = prescribed
            n = N_BOUNDS - 1 if s["input_mode"] == "emissions" else N_BOUNDS
            inputs[variable] = (np.ones(n), unit)
    return inputs


def _species(config: dict[str, Any], name: str) -> dict[str, Any]:
    return next(s for s in config["species"] if s["name"] == name)


def _drop(config: dict[str, Any], *names: str) -> None:
    config["species"] = [s for s in config["species"] if s["name"] not in names]


def _issues(
    config: dict[str, Any], inputs: dict[str, tuple[Any, str]] | None = None
) -> set[tuple[IssueKind, str]]:
    with pytest.raises(InputValidationError) as info:
        validate(config, inputs, _time_axis() if inputs is not None else None)
    return {(issue.kind, issue.location) for issue in info.value.issues}


def test_valid_config_and_inputs_pass(config: dict[str, Any]) -> None:
    validate(config, _inputs(config), _time_axis())


@pytest.mark.parametrize(
    ("edit", "location"),
    [
        (lambda c: c.update(optoins={}), "config.optoins"),
        (lambda c: c["options"].update(iirf_maximum=100), "options.iirf_maximum"),
        (
            lambda c: c["climate"].update(ocean_heat_capcity=[1]),
            "climate.ocean_heat_capcity",
        ),
        (
            lambda c: c["species_configs"].update(aci_scael=1),
            "species_configs.aci_scael",
        ),
        (lambda c: _species(c, "CO2").update(forcing_scael=1), "CO2.forcing_scael"),
        (
            lambda c: c["options"].update(ch4_method="thornhil2021"),
            "options.ch4_method",
        ),
        (
            lambda c: _species(c, "BC").update(input_mode="concentration"),
            "BC.input_mode",
        ),
    ],
)
def test_unknown_key(
    edit: Callable[[Any], None], location: str, config: dict[str, Any]
) -> None:
    edit(config)
    assert _issues(config) == {(IssueKind.UNKNOWN_KEY, location)}


def test_missing_species_key(config: dict[str, Any]) -> None:
    del _species(config, "CH4")["baseline_concentration"]
    assert _issues(config) == {(IssueKind.MISSING, "CH4.baseline_concentration")}


def test_other_ghg_method_is_unsupported(config: dict[str, Any]) -> None:
    config["options"]["ghg_method"] = "leach2021"
    assert _issues(config) == {(IssueKind.UNSUPPORTED, "options.ghg_method")}


def test_prescribed_ghg_forcing_is_unsupported(config: dict[str, Any]) -> None:
    _species(config, "CH4")["input_mode"] = "forcing"
    assert (IssueKind.UNSUPPORTED, "CH4.input_mode") in _issues(config)


@pytest.mark.parametrize(
    ("key", "value", "kind"),
    [
        ("ocean_heat_capacity", [8.0, math.nan, 100.0], IssueKind.NON_FINITE),
        ("ocean_heat_capacity", [8.0, 0.0, 100.0], IssueKind.OUT_OF_RANGE),
        ("ocean_heat_transfer", [-1.1, 1.6, 0.9], IssueKind.OUT_OF_RANGE),
        ("ocean_heat_transfer", [1.1, 1.6], IssueKind.SHAPE),
        ("deep_ocean_efficacy", math.nan, IssueKind.NON_FINITE),
        ("gamma_autocorrelation", 0.0, IssueKind.OUT_OF_RANGE),
    ],
)
def test_climate_parameters(
    key: str, value: Any, kind: IssueKind, config: dict[str, Any]
) -> None:
    config["climate"][key] = value
    assert _issues(config) == {(kind, f"climate.{key}")}


@pytest.mark.parametrize("key", ["ocean_heat_capacity", "ocean_heat_transfer"])
def test_null_ocean_parameter(key: str, config: dict[str, Any]) -> None:
    config["climate"][key] = None
    assert _issues(config) == {(IssueKind.NON_FINITE, f"climate.{key}")}


@pytest.mark.parametrize("value", [None, math.nan])
@pytest.mark.parametrize("key", ["partition_fraction", "unperturbed_lifetime"])
def test_null_gas_boxes(key: str, value: Any, config: dict[str, Any]) -> None:
    _species(config, "CO2")[key] = value
    assert _issues(config) == {(IssueKind.NON_FINITE, f"CO2.{key}")}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("partition_fraction", [[0.25, 0.25], [0.25, 0.25]]),
        ("partition_fraction", []),
        ("unperturbed_lifetime", 1.0),
    ],
)
def test_gas_boxes_are_one_dimensional(
    key: str, value: Any, config: dict[str, Any]
) -> None:
    _species(config, "CO2")[key] = value
    assert _issues(config) == {(IssueKind.SHAPE, f"CO2.{key}")}


def test_gas_boxes_match(config: dict[str, Any]) -> None:
    _species(config, "CO2")["unperturbed_lifetime"] = [1.0, 2.0]
    assert _issues(config) == {(IssueKind.SHAPE, "CO2.unperturbed_lifetime")}


@pytest.mark.parametrize("key", ["g0", "iirf_0"])
def test_infinite_lifetime_scaling(key: str, config: dict[str, Any]) -> None:
    _species(config, "HFC-134a")[key] = math.inf
    assert _issues(config) == {(IssueKind.NON_FINITE, f"HFC-134a.{key}")}


def test_nan_lifetime_scaling_is_allowed(config: dict[str, Any]) -> None:
    _species(config, "HFC-134a")["g0"] = math.nan
    validate(config)


def test_energy_balance_needs_two_layers(config: dict[str, Any]) -> None:
    config["climate"] |= {"ocean_heat_capacity": [8.0], "ocean_heat_transfer": [1.2]}
    assert _issues(config) == {(IssueKind.OUT_OF_RANGE, "climate.ocean_heat_capacity")}


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("HFC-23", "forcing_reference_concentration"),
        ("HFC-23", "greenhouse_gas_radiative_efficiency"),
        ("CH4", "baseline_concentration"),
        ("BC", "baseline_emissions"),
        ("Sulfur", "baseline_emissions"),
        ("CO", "ozone_radiative_efficiency"),
        ("Ozone", "forcing_scale"),
        ("Solar", "forcing_temperature_feedback"),
        ("CFC-11", "fractional_release"),
        ("NOx", "ch4_lifetime_chemical_sensitivity"),
    ],
)
def test_non_finite_species_parameter(
    name: str, key: str, config: dict[str, Any]
) -> None:
    _species(config, name)[key] = math.nan
    assert _issues(config) == {(IssueKind.NON_FINITE, f"{name}.{key}")}


def test_non_finite_species_configs(config: dict[str, Any]) -> None:
    config["species_configs"]["aci_scale"] = None
    assert _issues(config) == {(IssueKind.NON_FINITE, "species_configs.aci_scale")}


def test_unused_terms_are_skipped(config: dict[str, Any]) -> None:
    """FaIR's NaN-skipping sums drop the terms of a missing or zero coefficient."""
    _species(config, "NH3")["erfari_radiative_efficiency"] = math.nan
    _species(config, "BC")["erfari_radiative_efficiency"] = 0.0
    validate(config)
    drivers = {s["name"] for s, _ in linear_drivers(config, "ari")}
    assert not drivers & {"NH3", "BC"}
    # The iIRF and g parameters may be NaN, as FaIR sets a NaN lifetime scaling to 1.
    _species(config, "HFC-134a")["g1"] = math.nan
    validate(config)


def test_calculated_co2_needs_co2_afolu(config: dict[str, Any]) -> None:
    _drop(config, "CO2 AFOLU", "Land use")
    assert _issues(config) == {(IssueKind.MISSING, "CO2")}


def test_land_use_needs_co2_afolu(config: dict[str, Any]) -> None:
    _drop(config, "CO2 AFOLU")
    _species(config, "CO2")["input_mode"] = "concentration"
    assert _issues(config) == {(IssueKind.MISSING, "Land use")}


def test_calculated_eesc_needs_cfc11(config: dict[str, Any]) -> None:
    _drop(config, "CFC-11")
    eesc = "Equivalent effective stratospheric chlorine"
    assert _issues(config) == {(IssueKind.MISSING, eesc)}


def test_ghg_forcing_needs_n2o(config: dict[str, Any]) -> None:
    _drop(config, "N2O")
    assert _issues(config) == {(IssueKind.MISSING, "N2O")}


@pytest.mark.parametrize(
    "variable",
    [
        "Emissions|BC",
        "Prescribed Effective Radiative Forcing|Volcanic",
    ],
)
def test_non_finite_input(variable: str, config: dict[str, Any]) -> None:
    inputs = _inputs(config)
    values, unit = inputs[variable]
    inputs[variable] = (np.r_[values[:3], np.nan, values[4:]], unit)
    assert _issues(config, inputs) == {(IssueKind.NON_FINITE, variable)}


def test_non_finite_concentration(config: dict[str, Any]) -> None:
    _species(config, "CH4")["input_mode"] = "concentration"
    inputs = _inputs(config)
    variable = naming.concentration("CH4")
    inputs[variable] = (np.r_[np.ones(N_BOUNDS - 1), np.inf], "ppb")
    assert _issues(config, inputs) == {(IssueKind.NON_FINITE, variable)}


def test_emissions_at_the_last_bound_are_unread(config: dict[str, Any]) -> None:
    inputs = _inputs(config)
    inputs["Emissions|BC"] = (np.r_[np.ones(N_BOUNDS - 1), np.nan], "Mt BC/yr")
    validate(config, inputs, _time_axis())


@pytest.mark.parametrize(
    ("variable", "length"),
    [
        ("Emissions|BC", N_BOUNDS - 2),
        ("Prescribed Effective Radiative Forcing|Solar", N_BOUNDS - 1),
    ],
)
def test_input_does_not_cover_time_axis(
    variable: str, length: int, config: dict[str, Any]
) -> None:
    inputs = _inputs(config)
    inputs[variable] = (np.ones(length), inputs[variable][1])
    assert _issues(config, inputs) == {(IssueKind.SHAPE, variable)}


def test_missing_input(config: dict[str, Any]) -> None:
    inputs = _inputs(config)
    del inputs["Emissions|BC"]
    assert _issues(config, inputs) == {(IssueKind.MISSING, "Emissions|BC")}


def test_setup_model_validates(config: dict[str, Any]) -> None:
    config["climate"]["ocean_heat_capacity"] = [8.0, math.nan, 100.0]
    inputs = _inputs(config)
    del inputs["Emissions|BC"]
    with pytest.raises(InputValidationError) as info:
        setup_model(config, inputs, _time_axis())
    assert [(i.kind, i.location) for i in info.value.issues] == [
        (IssueKind.NON_FINITE, "climate.ocean_heat_capacity"),
        (IssueKind.MISSING, "Emissions|BC"),
    ]


def test_scaled_prescribed_forcing_needs_a_finite_scale(
    config: dict[str, Any],
) -> None:
    _species(config, "Solar")["forcing_scale"] = math.nan
    setup_model(config, _inputs(config), _time_axis())
    with pytest.raises(InputValidationError) as info:
        setup_model(
            config, _inputs(config), _time_axis(), scale_prescribed_forcing=True
        )
    assert [(i.kind, i.location) for i in info.value.issues] == [
        (IssueKind.NON_FINITE, "Solar.forcing_scale"),
    ]


def test_components_validate(config: dict[str, Any]) -> None:
    _drop(config, "CFC-11")
    with pytest.raises(InputValidationError, match="calculated EESC needs"):
        components(config)


def test_run_ensemble_validates_before_running(config: dict[str, Any]) -> None:
    pytest.importorskip("xarray")
    bad = _inputs(config)
    bad["Emissions|BC"] = (np.full(N_BOUNDS - 1, np.nan), "Mt BC/yr")
    other = copy.deepcopy(config)
    other["climate"]["deep_ocean_efficacy"] = -1.0
    with pytest.raises(InputValidationError) as info:
        run_ensemble(
            {"a": config, "b": other},
            {"good": _inputs(config), "bad": bad},
            _time_axis(),
        )
    assert [(i.kind, i.location) for i in info.value.issues] == [
        (IssueKind.OUT_OF_RANGE, "b: climate.deep_ocean_efficacy"),
        (IssueKind.NON_FINITE, "bad: Emissions|BC"),
    ]


def test_run_ensemble_checks_prescribed_forcing_scales(config: dict[str, Any]) -> None:
    pytest.importorskip("xarray")
    _species(config, "Solar")["forcing_scale"] = math.nan
    with pytest.raises(InputValidationError) as info:
        run_ensemble(
            {"a": config},
            {"s": _inputs(config)},
            _time_axis(),
            scale_prescribed_forcing=True,
        )
    assert [(i.kind, i.location) for i in info.value.issues] == [
        (IssueKind.NON_FINITE, "a: Solar.forcing_scale"),
    ]


def _calibration(tmp_path: Path, edit: Callable[[pd.DataFrame], None]) -> Path:
    parameters = pd.read_csv(
        CALIBRATION / "calibrated_constrained_parameters.csv", index_col=0
    )
    edit(parameters)
    path = tmp_path / "parameters.csv"
    parameters.to_csv(path)
    return path


def test_read_calibration_rejects_unknown_column(tmp_path: Path) -> None:
    def edit(p: pd.DataFrame) -> None:
        p["ocean_heat_capcity[0]"] = 1.0

    path = _calibration(tmp_path, edit)
    with pytest.raises(InputValidationError) as info:
        read_calibration(CALIBRATION / "species_configs_properties.csv", path)
    assert [(i.kind, i.location) for i in info.value.issues] == [
        (IssueKind.UNKNOWN_KEY, "ocean_heat_capcity[0]"),
    ]


def test_read_calibration_validates_each_member(tmp_path: Path) -> None:
    def edit(p: pd.DataFrame) -> None:
        p.loc[p.index[1], "ocean_heat_transfer[0]"] = np.nan

    path = _calibration(tmp_path, edit)
    member = pd.read_csv(path, index_col=0).index[1]
    with pytest.raises(InputValidationError) as info:
        read_calibration(CALIBRATION / "species_configs_properties.csv", path)
    assert [(i.kind, i.location) for i in info.value.issues] == [
        (IssueKind.NON_FINITE, f"{member}: climate.ocean_heat_transfer"),
    ]
