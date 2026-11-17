"""Pre-run validation of `rscm.magicc.full.from_magicc7`."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rscm.core import TimeAxis
from rscm.magicc import checks, emissions_driven, full
from rscm.magicc import prescribed as pr
from rscm.names import SECTORS
from rscm.validation import ExitCode, InputValidationError, IssueKind

YEARS = np.arange(1750.0, 2031.0)
DUMP = Path(__file__).parent / "regression" / "data" / "full_run"


def ramp_emissions(n: int = len(YEARS)) -> dict[str, np.ndarray]:
    """Rising emissions, so harmonisation never divides by a zero change."""
    ramp = np.linspace(0.0, 1.0, n)
    return {
        **{
            f"{s}|{sector}": ramp.copy()
            for s in full.SECTORAL_EMISSIONS
            for sector in SECTORS
        },
        **{s: ramp.copy() for s in full.halocarbon_species()},
    }


def translate(years=YEARS, emissions=None, **kwargs):
    return full.from_magicc7(
        TimeAxis.from_values(years),
        ramp_emissions(len(years)) if emissions is None else emissions,
        **kwargs,
    )


def issues(**kwargs) -> set[tuple[IssueKind, str]]:
    with pytest.raises(InputValidationError) as error:
        translate(**kwargs)
    return {(issue.kind, issue.location) for issue in error.value.issues}


def test_every_magicc7_key_is_classified_once():
    with open(DUMP / "05_ssp245_default_parameters.json") as f:
        dump = json.load(f)
    keys = {*dump["allcfgs"], *dump["outputcfgs"]}
    read = set(full.default_parameters())
    ignored = set(checks.IGNORED_PARAMETERS)
    unsupported = set(checks.UNSUPPORTED_PARAMETERS)

    assert not read & ignored
    assert not read & unsupported
    assert not ignored & unsupported
    assert keys == (read | ignored | unsupported) - {
        "startyear",
        "endyear",
        "stepsperyear",
    }


def test_unknown_key_suggests_a_name():
    with pytest.raises(InputValidationError) as error:
        translate(parameters={"core_climatesensitivty": 3.0})

    (issue,) = error.value.issues
    assert (issue.kind, issue.location) == (
        IssueKind.UNKNOWN_KEY,
        "core_climatesensitivty",
    )
    assert "core_climatesensitivity" in issue.message
    assert error.value.exit_code == ExitCode.CONFIG


REJECTED = {
    "ch4_clathratefeed_apply": 1,
    "oceancc_rad_setting": 1,
    "co2_capconc_apply": 1,
    "core_elnino_apply": 1,
    "core_amv_apply": 1,
    "co2_zeroemis_afterxpgc_apply": 1,
    "rf_total_constantafteryr": 1752,
    "rf_co2_constantafteryr": 1752,
    "rf_aer_constantafteryr": 1752,
    "rf_total_runmodus": "CO2",
    "rf_initialization_method": "ZEROSTARTSHIFT",
    "co2_preindco2conc_apply": 1,
    "rf_volcanic_historic_meanzero": 1,
    "rf_volc_futr_meanlastxyears": 10,
    "core_switch_ocn_tempprofile": 1,
    "ch4_incl_ch4ox": 0,
    "ch4_applyfossfuelfraction": 1,
    "fgas_names": ["CF4"],
    "file_tuningmodel_2": "MAGTUNE_X.CFG",
    "startyear": 1700,
    "endyear": 2100,
}


@pytest.mark.parametrize(("key", "value"), REJECTED.items())
def test_unsupported_values_are_rejected(key, value):
    with pytest.raises(InputValidationError) as error:
        translate(parameters={key: value})

    assert {(i.kind, i.location) for i in error.value.issues} == {
        (IssueKind.UNSUPPORTED, key)
    }
    assert error.value.exit_code == ExitCode.UNSUPPORTED


ACCEPTED = {
    "air_sox_scale": 10.0,
    "cloud_apply_limit_max": 1,
    "scen_histadjust_0no1scale2shift": 1,
    "scen_co2bhistadjust_0no1scale2shift": 1,
    "oceancc_convltn2int_yrsago": 100,
    "co2_feedtemp_landonly_apply": 1,
    "core_maximal_temperature": 1.0,
    "n2o_apply_scaleemis": 1,
    "core_dependence_rlo_on_dt2x": 1,
    "rf_solar_future_meanlastxyears": 30,
    "core_upwelling_scaling_method": "NHSH",
    "rf_total_constantafteryr": 2030,
    "rf_co2_constantafteryr": 10000,
    "ch4_clathratefeed_apply": 0,
    "file_tuningmodel_1": "PYMAGICC",
    "fgas_names": [*full._defaults()["unsupported"]["fgas_names"], "", ""],
    "startyear": 1750,
    "out_sealevel": 0,
    "slr_gis_sid_scaling": 2.0,
}


@pytest.mark.parametrize(("key", "value"), ACCEPTED.items())
def test_ignorable_and_default_values_are_accepted(key, value):
    translate(parameters={key: value})


def test_sea_level_parameters_are_rejected_with_sea_level_on():
    found = issues(parameters={"out_sealevel": 1, "slr_gis_sid_scaling": 2.0})

    assert found == {(IssueKind.UNSUPPORTED, "slr_gis_sid_scaling")}


def test_shared_halocarbon_switches_must_agree():
    found = issues(parameters={"fgas_use_taustrat_var": 0})

    assert found == {(IssueKind.UNSUPPORTED, "fgas_use_taustrat_var")}
    translate(parameters={"fgas_use_taustrat_var": 0, "mhalo_use_taustrat_var": 0})


def test_radiative_efficiencies_may_carry_the_sum_scale():
    """MAGICC7 writes the efficiencies back multiplied by ``rf_fgassum_scale``."""
    default = np.array(full._defaults()["unsupported"]["fgas_radiat_eff"])
    scaled = {"rf_fgassum_scale": 0.5, "fgas_radiat_eff": list(default * 0.5)}

    translate(parameters=scaled)
    found = issues(parameters={**scaled, "fgas_radiat_eff": list(default * 0.4)})
    assert found == {(IssueKind.UNSUPPORTED, "fgas_radiat_eff")}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("co2_plantboxresp_method", 1),
        ("oceancc_model", "BOXDIFF"),
        ("core_co2ch4n2o_rfmethod", "MYHRE"),
        ("rf_efficacy_apply", 3),
        ("rf_soxi_dir_apply", 0),
        ("fgas_n", 22),
        ("ch4_tausoil", 0.0),
    ],
)
def test_unimplemented_choices_are_rejected(key, value):
    assert issues(parameters={key: value}) == {(IssueKind.UNSUPPORTED, key)}


def test_thermal_expansion_file_must_be_known():
    found = issues(
        parameters={"out_sealevel": 1, "file_slr_thexp_params": "SLR_THEXP_OTHER.CFG"}
    )

    assert found == {(IssueKind.UNSUPPORTED, "file_slr_thexp_params")}


@pytest.mark.parametrize(
    ("key", "value", "kind"),
    [
        ("core_climatesensitivity", 0.0, IssueKind.OUT_OF_RANGE),
        ("core_mixedlayer_depth", -60.0, IssueKind.OUT_OF_RANGE),
        ("n2o_tauinit", 0.0, IssueKind.OUT_OF_RANGE),
        ("ch4_taustrat", -1.0, IssueKind.OUT_OF_RANGE),
        ("ch4_tautot_init", 80.0, IssueKind.OUT_OF_RANGE),
        ("rf_regions_ch4", [0.0, 0.0, 0.0, 0.0], IssueKind.OUT_OF_RANGE),
        ("rf_regions_fgas", [1.0, 1.0, 0.0, 0.0], IssueKind.OUT_OF_RANGE),
        ("rf_regions_co2", [1.0, 1.0, 1.0], IssueKind.SHAPE),
        ("rf_regions_n2o", [1.0, np.nan, 1.0, 1.0], IssueKind.NON_FINITE),
        ("rf_regions_ch4", [0.42, -0.58, 0.0, 0.0], IssueKind.OUT_OF_RANGE),
        ("rf_regions_tropoz", [0.42, -0.58, 1.0, 1.0], IssueKind.OUT_OF_RANGE),
    ],
)
def test_out_of_domain_parameters_are_rejected(key, value, kind):
    assert issues(parameters={key: value}) == {(kind, key)}


def test_land_heat_capacity_thickness_only_matters_when_applied():
    parameters = {"core_landheatcapacity_apply": 1, "core_landhc_effthickness": 0.0}

    assert issues(parameters=parameters) == {
        (IssueKind.OUT_OF_RANGE, "core_landhc_effthickness")
    }
    translate(parameters={**parameters, "core_landheatcapacity_apply": 0})


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda e: e.pop("CO2|Fossil"), (IssueKind.MISSING, "CO2|Fossil")),
        (
            lambda e: e.update({"CO3|Fossil": e["CO2|Fossil"]}),
            (IssueKind.UNKNOWN_KEY, "CO3|Fossil"),
        ),
        (lambda e: e.update({"CFC11": e["CFC11"][:-1]}), (IssueKind.SHAPE, "CFC11")),
        (
            lambda e: e["SOx|Fossil"].__setitem__(2, np.nan),
            (IssueKind.NON_FINITE, "SOx|Fossil"),
        ),
        (
            lambda e: e["CH4|Biosphere"].__setitem__(0, np.inf),
            (IssueKind.NON_FINITE, "CH4|Biosphere"),
        ),
    ],
)
def test_bad_emissions_are_rejected(change, expected):
    emissions = ramp_emissions()
    change(emissions)

    assert issues(emissions=emissions) == {expected}


@pytest.mark.parametrize(
    ("given", "kind"),
    [
        ({"HFC4310": 0.0}, IssueKind.UNKNOWN_KEY),  # naming: allow
        ({"CO2": -1.0}, IssueKind.OUT_OF_RANGE),
        ({"CO2": np.nan}, IssueKind.NON_FINITE),
    ],
)
def test_bad_initial_concentrations_are_rejected(given, kind):
    (name,) = given
    found = issues(initial_concentrations=given)

    assert found == {(kind, f"initial_concentrations[{name}]")}


def test_every_issue_is_reported_together():
    emissions = ramp_emissions()
    emissions.pop("CO2|Fossil")
    with pytest.raises(InputValidationError) as error:
        translate(
            emissions=emissions,
            parameters={"core_climatesensitivity": -1.0, "core_elnino_apply": 1},
        )

    assert {(i.kind, i.location) for i in error.value.issues} == {
        (IssueKind.UNSUPPORTED, "core_elnino_apply"),
        (IssueKind.OUT_OF_RANGE, "core_climatesensitivity"),
        (IssueKind.MISSING, "CO2|Fossil"),
    }
    assert error.value.exit_code == ExitCode.UNSUPPORTED


@pytest.fixture(scope="module")
def inputs():
    return full.load_inputs()


def test_solar_file_must_cover_the_run(inputs):
    solar = inputs["file_solar_rf"]
    short = replace(
        solar, years=np.asarray(solar.years)[:3], values=np.asarray(solar.values)[:3]
    )

    found = issues(input_files={**inputs, "file_solar_rf": short})

    assert found == {(IssueKind.SHAPE, "file_solar_rf")}


def test_concentration_file_must_cover_its_prescribed_years(inputs):
    co2 = inputs["file_co2_conc"]
    short = replace(
        co2, years=np.asarray(co2.years)[:3], values=np.asarray(co2.values)[:3]
    )

    found = issues(input_files={**inputs, "file_co2_conc": short})

    assert found == {(IssueKind.SHAPE, "file_co2_conc")}


@pytest.mark.parametrize(
    ("value", "kind"), [(np.nan, IssueKind.NON_FINITE), (-5.0, IssueKind.OUT_OF_RANGE)]
)
def test_concentration_files_must_be_finite_and_positive(inputs, value, kind):
    values = np.array(inputs["file_ch4_conc"].values, dtype=np.float64)
    values[10] = value

    found = issues(
        input_files={
            **inputs,
            "file_ch4_conc": replace(inputs["file_ch4_conc"], values=values),
        }
    )

    assert found == {(kind, "file_ch4_conc")}


def test_missing_concentration_file_is_reported(inputs):
    files = dict(inputs)
    files.pop("file_n2o_conc")

    assert issues(input_files=files) == {(IssueKind.MISSING, "file_n2o_conc")}


def test_non_finite_prescribed_forcing_is_rejected(inputs):
    values = np.array(inputs["file_landuse_rf"].values, dtype=np.float64)
    values[:] = np.nan
    files = {
        **inputs,
        "file_landuse_rf": replace(inputs["file_landuse_rf"], values=values),
    }

    found = issues(input_files=files)

    assert (
        IssueKind.NON_FINITE,
        "Effective Radiative Forcing|Land-use Change",
    ) in found


def test_zero_anthropogenic_nitrate_precursors_give_finite_forcing():
    """MAGICC7 sets nitrate forcing to zero rather than dividing 0 by 0."""
    zero = np.zeros(len(YEARS))

    industrial, biomass = pr.nitrate(
        YEARS, (zero, zero), (zero, zero), zero, [1.0] * 4, factor=1.0
    )

    assert np.all(np.isfinite(industrial))
    assert np.all(np.isfinite(biomass))


def test_initial_concentrations_come_from_the_files(inputs):
    history = inputs["file_co2_conc"]
    values = np.array(history.values, dtype=np.float64)
    values[np.asarray(history.years) == YEARS[0]] = 300.0
    files = {**inputs, "file_co2_conc": replace(history, values=values)}

    config, _ = translate(input_files=files)

    assert config["initial_values"]["Atmospheric Concentrations|CO2"] == 300.0
    assert config["components"]["GHGForcing"]["co2_pi"] == 300.0


def test_input_files_are_not_changed(inputs):
    files = dict(inputs)
    before = {key: copy.deepcopy(history) for key, history in files.items()}

    translate(input_files=files)

    assert files.keys() == before.keys()
    for key, history in files.items():
        assert history is inputs[key]
        np.testing.assert_array_equal(history.years, before[key].years)
        np.testing.assert_array_equal(history.values, before[key].values)


def test_ch4_lifetimes_reach_every_component():
    config, _ = translate(
        parameters={"ch4_tausoil": 60.0, "ch4_tautempsensitivity": 0.1}
    )
    c = config["components"]
    non_oh = 1.0 / (1.0 / 60.0 + 1.0 / 120.0 + 1.0 / 200.0)

    assert c["CH4Chemistry"]["soil_lifetime"] == 60.0
    assert c["CH4Chemistry"]["temperature_sensitivity"] == 0.1
    assert c["CH4Oxidation"]["non_oh_lifetime"] == pytest.approx(non_oh)
    assert c["HalocarbonChemistry"]["initial_ch4_oh_lifetime"] == pytest.approx(
        1.0 / (1.0 / 9.9474 - 1.0 / non_oh)
    )


def test_feedback_start_years_are_moved_into_the_run():
    config, _ = translate(
        parameters={"ch4_feed_yrstart": 1700.0, "n2o_feed_yrstart": 3000.0}
    )
    c = config["components"]

    assert c["CH4Chemistry"]["feedback_start_year"] == YEARS[0]
    assert c["N2OChemistry"]["feedback_start_year"] == YEARS[-1]


def test_thermal_expansion_follows_scaling_and_file():
    parameters = {
        "out_sealevel": 1,
        "slr_expansion_scaling": 2.0,
        "slr_expansion_params": [1.0] * 6,
    }
    config, _ = translate(parameters=parameters)
    expansion = config["components"]["ClimateUDEB"]["thermal_expansion"]

    assert expansion["scaling"] == 2.0
    assert expansion["coefficients"] == list(
        full.THERMAL_EXPANSION_FILES["SLR_THEXP_PARAMS_CMIP5MEAN.CFG"]
    )

    config, _ = translate(parameters={**parameters, "file_slr_thexp_params": ""})
    expansion = config["components"]["ClimateUDEB"]["thermal_expansion"]
    assert expansion["coefficients"] == [1.0] * 6


@pytest.mark.parametrize(
    "name", ["use_tauoh_var", "use_taustrat_var", "taustrat_sens2meridflux"]
)
def test_changing_only_the_montreal_gas_setting_is_rejected(name):
    value = full._defaults()["unsupported"][f"fgas_{name}"]
    changed = 0 if isinstance(value, int) else value * 2

    found = issues(parameters={f"mhalo_{name}": changed})

    assert found == {(IssueKind.UNSUPPORTED, f"fgas_{name}")}


def test_ipcctar_reads_the_radiative_efficiencies():
    parameters = {
        "core_co2ch4n2o_rfmethod": "IPCCTAR",
        "ch4_radeff_wm2perppb": 0.04,
        "n2o_radeff_wm2perppb": 0.13,
    }
    config, _ = translate(parameters=parameters)
    ghg = config["components"]["GHGForcing"]

    assert ghg["ch4_radiative_efficiency"] == 0.04
    assert ghg["n2o_radiative_efficiency"] == 0.13


@pytest.mark.parametrize(
    ("value", "kind"), [(np.nan, IssueKind.NON_FINITE), (-5.0, IssueKind.OUT_OF_RANGE)]
)
def test_initial_values_from_files_are_checked_without_prescribed_years(
    inputs, value, kind
):
    co2 = inputs["file_co2_conc"]
    values = np.array(co2.values, dtype=np.float64)
    values[int(YEARS[0] - np.asarray(co2.years)[0])] = value
    files = {**inputs, "file_co2_conc": replace(co2, values=values)}

    found = issues(
        input_files=files, parameters={"co2_switchfromconc2emis_year": YEARS[0]}
    )

    assert found == {(kind, "file_co2_conc")}


def test_missing_forcing_file_is_reported_before_reading():
    found = issues(parameters={"file_solar_rf": "MISSING_SOLAR_RF.IN"})

    assert found == {(IssueKind.MISSING, "file_solar_rf")}


def test_budget_files_are_checked_without_prescribed_years(inputs):
    files = {k: v for k, v in inputs.items() if k != "file_ch4_conc"}

    found = issues(
        input_files=files, parameters={"ch4_switchfromconc2emis_year": YEARS[0]}
    )

    assert found == {(IssueKind.MISSING, "file_ch4_conc")}


@pytest.mark.parametrize(
    ("value", "kind"),
    [([1.0] * 5, IssueKind.SHAPE), ([np.nan] * 6, IssueKind.NON_FINITE)],
)
def test_bad_thermal_expansion_coefficients_are_rejected(value, kind):
    parameters = {
        "out_sealevel": 1,
        "file_slr_thexp_params": "",
        "slr_expansion_params": value,
    }

    assert issues(parameters=parameters) == {(kind, "slr_expansion_params")}


@pytest.mark.parametrize(
    ("key", "value", "kind"),
    [
        ("core_climatesensitivity", np.inf, IssueKind.NON_FINITE),
        ("core_feedback_qsensitivity", np.nan, IssueKind.NON_FINITE),
        ("core_delq2xco2", 0.0, IssueKind.OUT_OF_RANGE),
        ("core_delq2xco2", np.nan, IssueKind.NON_FINITE),
        ("core_climatesensitivity", "3.0", IssueKind.TYPE),
        ("core_climatesensitivity", None, IssueKind.TYPE),
        ("core_climatesensitivity", [3.0], IssueKind.TYPE),
        ("oceancc_model", 3, IssueKind.TYPE),
        ("rf_regions_ch4", ["1", 1.0, 1.0, 1.0], IssueKind.TYPE),
        ("pf_apply", "0", IssueKind.TYPE),
        ("pf_apply", "False", IssueKind.TYPE),
        ("out_sealevel", 2, IssueKind.OUT_OF_RANGE),
    ],
)
def test_parameters_must_be_finite_values_of_their_type(key, value, kind):
    assert issues(parameters={key: value}) == {(kind, key)}


@pytest.mark.parametrize("value", [0, 1, False, True, np.int64(1)])
def test_switches_take_zero_one_or_bool(value):
    config, _ = translate(parameters={"pf_apply": value})

    assert ("Permafrost" in config["components"]) == bool(value)


def test_numbers_of_any_numeric_type_are_accepted():
    translate(parameters={"core_climatesensitivity": np.float32(3.0)})
    translate(parameters={"core_climatesensitivity": 3})


def test_infinite_years_are_accepted():
    translate(parameters={"co2_switchfromconc2emis_year": np.inf})


@pytest.mark.parametrize(
    ("given", "kind"),
    [
        ({"CO2": 0.0}, IssueKind.OUT_OF_RANGE),
        ({"CO2": 1e-9}, IssueKind.OUT_OF_RANGE),
        ({"CO2": 179.0}, IssueKind.OUT_OF_RANGE),
        ({"CH4": 0.0}, IssueKind.OUT_OF_RANGE),
        ({"CH4": 339.0}, IssueKind.OUT_OF_RANGE),
        ({"N2O": 199.0}, IssueKind.OUT_OF_RANGE),
        ({"CO2": "280"}, IssueKind.TYPE),
        ({"CO2": None}, IssueKind.TYPE),
    ],
)
def test_initial_concentrations_must_be_positive_numbers(given, kind):
    (name,) = given

    assert issues(initial_concentrations=given) == {
        (kind, f"initial_concentrations[{name}]")
    }


@pytest.mark.parametrize(
    ("given", "ok"),
    [
        ({"N2O": 199.0}, True),
        ({"CH4": 1.0}, True),
        ({"CFC11": 0.0}, True),
        ({"CO2": 179.0}, True),
        ({"N2O": 0.0}, False),
        ({"CO2": 0.0}, False),
    ],
)
def test_ipcctar_initial_concentrations(given, ok):
    kwargs = {
        "parameters": {"core_co2ch4n2o_rfmethod": "IPCCTAR"},
        "initial_concentrations": given,
    }
    if ok:
        translate(**kwargs)
    else:
        (name,) = given
        assert issues(**kwargs) == {
            (IssueKind.OUT_OF_RANGE, f"initial_concentrations[{name}]")
        }


def test_zero_halocarbon_initial_concentrations_are_accepted():
    translate(initial_concentrations={"CFC11": 0.0})


def test_initial_concentrations_at_the_fitted_range_are_accepted():
    translate(initial_concentrations={"CO2": 180.0, "CH4": 340.0, "N2O": 200.0})


@pytest.mark.parametrize(
    ("bugs", "kind"),
    [
        (None, IssueKind.TYPE),
        (7, IssueKind.TYPE),
        ([7], IssueKind.TYPE),
        (["bogus"], IssueKind.UNKNOWN_KEY),
        ("7.5.4", IssueKind.UNKNOWN_KEY),
    ],
)
def test_bad_bugs_are_rejected(bugs, kind):
    assert issues(bugs=bugs) == {(kind, "bugs")}


def test_runs_ending_before_the_forcing_files_start_are_rejected():
    with pytest.raises(InputValidationError) as error:
        translate(years=np.arange(1750.0, 1766.0))

    (issue,) = error.value.issues
    assert (issue.kind, issue.location) == (IssueKind.OUT_OF_RANGE, "time_axis")
    assert "1766" in issue.message


def test_runs_ending_once_the_forcing_files_start_are_accepted():
    translate(years=np.arange(1750.0, 1767.0))


@pytest.mark.parametrize("start", [1700.0, 1749.0, 1751.0, 1850.0, 2000.0])
def test_runs_must_start_in_the_preindustrial_reference_year(start):
    with pytest.raises(InputValidationError) as error:
        translate(years=np.arange(start, 2101.0))

    (issue,) = error.value.issues
    assert (issue.kind, issue.location) == (IssueKind.OUT_OF_RANGE, "time_axis")
    assert "must start in 1750" in issue.message


def test_half_year_axis_is_rejected():
    with pytest.raises(InputValidationError, match="time_axis"):
        translate(years=YEARS + 0.5)


@pytest.mark.parametrize("species", ["SOx", "BC", "OC"])
def test_constant_industrial_aerosol_emissions_are_rejected(species):
    emissions = ramp_emissions()
    emissions[f"{species}|Fossil"] = np.full(len(YEARS), 10.0)

    found = issues(emissions=emissions)

    assert found == {(IssueKind.OUT_OF_RANGE, f"{species}|Fossil")}


def test_zero_black_carbon_emissions_are_rejected():
    emissions = ramp_emissions()
    emissions["BC|Fossil"] = np.zeros(len(YEARS))
    emissions["BC|Biosphere"] = np.zeros(len(YEARS))

    found = issues(emissions=emissions)

    assert found == {
        (IssueKind.OUT_OF_RANGE, "BC|Fossil"),
        (IssueKind.OUT_OF_RANGE, "BC|Fossil + BC|Biosphere"),
    }


def test_constant_aerosol_emissions_are_accepted_when_the_run_ends_with_the_files():
    years = np.arange(1750.0, 2016.0)
    emissions = ramp_emissions(len(years))
    emissions["SOx|Fossil"] = np.full(len(years), 10.0)

    translate(years=years, emissions=emissions)


def test_emissions_may_be_lists_but_not_strings():
    emissions = {k: list(v) for k, v in ramp_emissions().items()}
    translate(emissions=emissions)
    emissions["CFC11"] = ["a"] * len(YEARS)

    assert issues(emissions=emissions) == {(IssueKind.TYPE, "CFC11")}


def test_emissions_series_must_follow_the_time_axis():
    emissions = {k: pd.Series(v, index=YEARS) for k, v in ramp_emissions().items()}
    translate(emissions=emissions)
    emissions["CFC11"] = emissions["CFC11"][::-1]

    assert issues(emissions=emissions) == {(IssueKind.SHAPE, "CFC11")}


def test_empty_emissions_series_is_named():
    emissions = {**ramp_emissions(), "CFC11": pd.Series([], dtype=float)}

    assert issues(emissions=emissions) == {(IssueKind.SHAPE, "CFC11")}


def test_non_positive_concentration_stops_the_run():
    years = np.arange(1750.0, 2101.0)
    emissions = ramp_emissions(len(years))
    emissions["CO2|Fossil"][270:] = -1e5
    config, inputs = translate(years=years, emissions=emissions)
    model = emissions_driven.setup_model(config, inputs, TimeAxis.from_values(years))

    with pytest.raises(RuntimeError, match=r"CO2 concentration is -?\d.* ppm in \d{4}"):
        model.run()
