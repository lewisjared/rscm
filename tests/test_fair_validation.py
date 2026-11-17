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


@pytest.mark.parametrize(
    "fractions", [[0.5, 0.0, 0.0, 0.0], [1.5, -0.5, 0.0, 0.0], [0.5, 0.5, 0.5, -0.5]]
)
def test_partition_fractions_sum_to_one(
    fractions: list[float], config: dict[str, Any]
) -> None:
    _species(config, "CO2")["partition_fraction"] = fractions
    assert _issues(config) == {(IssueKind.OUT_OF_RANGE, "CO2.partition_fraction")}


def test_partition_fractions_allow_rounding(config: dict[str, Any]) -> None:
    _species(config, "CO2")["partition_fraction"] = [0.2173, 0.224, 0.2824, 0.2762]
    validate(config)


@pytest.mark.parametrize("value", [math.nan, math.inf, 0.0, -44.0])
def test_molecular_weight(value: float, config: dict[str, Any]) -> None:
    _species(config, "N2O")["molecular_weight"] = value
    kind = (
        IssueKind.NON_FINITE
        if math.isnan(value) or math.isinf(value)
        else (IssueKind.OUT_OF_RANGE)
    )
    assert _issues(config) == {(kind, "N2O.molecular_weight")}


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


def _axis_issues(
    config: dict[str, Any], bounds: list[float]
) -> set[tuple[IssueKind, str]]:
    with pytest.raises(InputValidationError) as info:
        validate(config, _inputs(config), TimeAxis.from_values(np.asarray(bounds)))
    return {(issue.kind, issue.location) for issue in info.value.issues}


def test_uneven_time_axis_is_unsupported(config: dict[str, Any]) -> None:
    bounds = [1750.0, 1751.0, 1753.0, *np.arange(1754.0, 1762.0)]
    assert _axis_issues(config, bounds) == {(IssueKind.UNSUPPORTED, "time_axis")}


def test_fractional_even_time_axis_passes(config: dict[str, Any]) -> None:
    validate(config, _inputs(config), TimeAxis.from_values(1750 + 0.1 * np.arange(11)))


def test_run_ensemble_rejects_uneven_time_axis(config: dict[str, Any]) -> None:
    pytest.importorskip("xarray")
    bounds = np.r_[1750.0, 1751.0, 1753.0, np.arange(1754.0, 1762.0)]
    with pytest.raises(InputValidationError) as info:
        run_ensemble(
            {"a": config}, {"s": _inputs(config)}, TimeAxis.from_values(bounds)
        )
    assert [(i.kind, i.location) for i in info.value.issues] == [
        (IssueKind.UNSUPPORTED, "time_axis"),
    ]


def _copy(config: dict[str, Any], name: str, new_name: str) -> None:
    entry = copy.deepcopy(_species(config, name))
    entry["name"] = new_name
    config["species"].append(entry)


@pytest.mark.parametrize(
    ("name", "copied"),
    [
        ("CO2", "CO2b"),
        ("CFC-11", "CFC-11b"),
        ("Aerosol-radiation interactions", "ari2"),
        ("Sulfur", "Sulfur2"),
    ],
)
def test_duplicate_species_type_is_unsupported(
    name: str, copied: str, config: dict[str, Any]
) -> None:
    _copy(config, name, copied)
    assert _issues(config) == {(IssueKind.UNSUPPORTED, f"{copied}.type")}


def test_repeatable_species_type_passes(config: dict[str, Any]) -> None:
    _copy(config, "NF3", "NF3b")
    validate(config)


def test_duplicate_species_name(config: dict[str, Any]) -> None:
    config["species"].append(copy.deepcopy(_species(config, "NF3")))
    assert _issues(config) == {(IssueKind.UNSUPPORTED, "NF3.name")}


@pytest.mark.parametrize(
    ("name", "key", "value"),
    [
        ("CO2", "unperturbed_lifetime", [-1.0, 394.4, 36.54, 4.304]),
        ("CO2", "forcing_reference_concentration", 0.0),
        ("CO2", "baseline_concentration", 0.0),
        ("CH4", "baseline_concentration", -10.0),
        ("N2O", "forcing_reference_concentration", -1.0),
    ],
)
def test_gas_parameter_out_of_range(
    name: str, key: str, value: Any, config: dict[str, Any]
) -> None:
    _species(config, name)[key] = value
    assert _issues(config) == {(IssueKind.OUT_OF_RANGE, f"{name}.{key}")}


@pytest.mark.parametrize(
    ("name", "key", "value"),
    [
        ("CO2", "unperturbed_lifetime", [0.0, 394.4, 36.54, 4.304]),
        ("CH4", "baseline_concentration", 0.0),
        ("HFC-23", "forcing_reference_concentration", 0.0),
        ("HFC-23", "baseline_concentration", -10.0),
        ("Sulfur", "aci_shape", -0.01),
    ],
)
def test_gas_parameters_fair_runs_pass(
    name: str, key: str, value: Any, config: dict[str, Any]
) -> None:
    _species(config, name)[key] = value
    validate(config, _inputs(config), _time_axis())


def test_aci_baseline_needs_a_logarithm(config: dict[str, Any]) -> None:
    _species(config, "Sulfur")["aci_shape"] = -1.0
    assert _issues(config) == {(IssueKind.OUT_OF_RANGE, "Aerosol-cloud interactions")}


def test_aci_emissions_need_a_logarithm(config: dict[str, Any]) -> None:
    inputs = _inputs(config)
    inputs["Emissions|SOx"] = (np.r_[np.ones(4), np.full(6, -1e4)], "Mt SO2/yr")
    with pytest.raises(InputValidationError) as info:
        validate(config, inputs, _time_axis())
    (issue,) = info.value.issues
    assert (issue.kind, issue.location) == (
        IssueKind.OUT_OF_RANGE,
        "Aerosol-cloud interactions",
    )
    assert "1754" in issue.message
    assert "Emissions|SOx" in issue.message


@pytest.mark.parametrize(
    ("key", "value", "kind"),
    [
        ("ocean_heat_capacity", ["8", "14", "100"], IssueKind.NON_FINITE),
        ("deep_ocean_efficacy", [1.1, 1.2], IssueKind.SHAPE),
        ("deep_ocean_efficacy", "1.1", IssueKind.NON_FINITE),
    ],
)
def test_climate_parameter_types(
    key: str, value: Any, kind: IssueKind, config: dict[str, Any]
) -> None:
    config["climate"][key] = value
    assert _issues(config) == {(kind, f"climate.{key}")}


def test_scalar_species_parameter(config: dict[str, Any]) -> None:
    _species(config, "HFC-23")["forcing_scale"] = [1.0, 2.0]
    config["options"]["iirf_max"] = "100"
    assert _issues(config) == {
        (IssueKind.SHAPE, "HFC-23.forcing_scale"),
        (IssueKind.NON_FINITE, "options.iirf_max"),
    }


def test_run_ensemble_needs_a_config(config: dict[str, Any]) -> None:
    pytest.importorskip("xarray")
    with pytest.raises(InputValidationError) as info:
        run_ensemble({}, {"s": _inputs(config)}, _time_axis())
    assert [(i.kind, i.location) for i in info.value.issues] == [
        (IssueKind.MISSING, "configs"),
    ]


def test_run_ensemble_needs_a_scenario(config: dict[str, Any]) -> None:
    pytest.importorskip("xarray")
    with pytest.raises(InputValidationError) as info:
        run_ensemble({"a": config}, {}, _time_axis())
    assert [(i.kind, i.location) for i in info.value.issues] == [
        (IssueKind.MISSING, "scenarios"),
    ]


@pytest.mark.parametrize("argument", ["configs", "scenarios"])
def test_run_ensemble_needs_mappings(argument: str, config: dict[str, Any]) -> None:
    pytest.importorskip("xarray")
    arguments = {"configs": {"a": config}, "scenarios": {"s": _inputs(config)}}
    arguments[argument] = list(arguments[argument].values())
    with pytest.raises(TypeError, match=f"{argument} must be a mapping"):
        run_ensemble(**arguments, time_axis=_time_axis())


def _ensemble_issues(
    config: dict[str, Any], other: dict[str, Any]
) -> list[tuple[IssueKind, str, str]]:
    pytest.importorskip("xarray")
    with pytest.raises(InputValidationError) as info:
        run_ensemble({"a": config, "b": other}, {"s": _inputs(config)}, _time_axis())
    return [(i.kind, i.location, i.message) for i in info.value.issues]


def test_run_ensemble_names_missing_species(config: dict[str, Any]) -> None:
    other = copy.deepcopy(config)
    _drop(other, "NF3")
    ((kind, location, message),) = _ensemble_issues(config, other)
    assert (kind, location) == (IssueKind.SHAPE, "b: species")
    assert "NF3" in message


def test_run_ensemble_needs_the_same_species_order(config: dict[str, Any]) -> None:
    other = copy.deepcopy(config)
    other["species"] = other["species"][::-1]
    ((kind, location, message),) = _ensemble_issues(config, other)
    assert (kind, location) == (IssueKind.SHAPE, "b: species")
    assert "order" in message


def test_run_ensemble_needs_the_same_layers(config: dict[str, Any]) -> None:
    other = copy.deepcopy(config)
    other["climate"]["ocean_heat_capacity"] = [8.0, 14.0]
    other["climate"]["ocean_heat_transfer"] = [1.1, 1.6]
    ((kind, location, _),) = _ensemble_issues(config, other)
    assert (kind, location) == (IssueKind.SHAPE, "b: climate.ocean_heat_capacity")


def test_unknown_input(config: dict[str, Any]) -> None:
    inputs = _inputs(config)
    inputs["Emissions|co2|fossil"] = inputs.pop("Emissions|CO2|Fossil")
    inputs["Emissions|Bogus"] = (np.ones(N_BOUNDS - 1), "Mt X/yr")
    with pytest.raises(InputValidationError) as info:
        validate(config, inputs, _time_axis())
    issues = {(i.kind, i.location): i.message for i in info.value.issues}
    assert set(issues) == {
        (IssueKind.MISSING, "Emissions|CO2|Fossil"),
        (IssueKind.UNKNOWN_KEY, "Emissions|co2|fossil"),
        (IssueKind.UNKNOWN_KEY, "Emissions|Bogus"),
    }
    message = issues[IssueKind.UNKNOWN_KEY, "Emissions|co2|fossil"]
    assert "'Emissions|CO2|Fossil'" in message


def test_inputs_of_other_input_modes_are_known(config: dict[str, Any]) -> None:
    """A scenario may hold the inputs of other modes, as ensembles share it."""
    inputs = _inputs(config)
    inputs[naming.concentration("CH4")] = (np.ones(N_BOUNDS), "ppb")
    validate(config, inputs, _time_axis())
