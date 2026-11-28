"""Building the emissions- and concentration-driven MAGICC models."""

import copy
import math

import numpy as np
import pandas as pd
import pytest

from rscm._lib.magicc import N2OChemistryBuilder
from rscm.core import TimeAxis
from rscm.magicc import concentration_driven, emissions_driven, full, model
from rscm.magicc.prescribed import area_fractions
from rscm.names import (
    EFFICACY_ADJUSTED_ERF,
    ERF,
    ERF_F_GASES,
    NITROGEN_FLUX_DEPOSITION,
    SECTORS,
    prescribed,
)
from rscm.spec import diff, run_all
from rscm.validation import InputValidationError, IssueKind

# MAGICC7 harmonises prescribed forcing in 2019.
YEARS = np.arange(1750.0, 2021.0)
N2O_CONCENTRATION = "Atmospheric Concentrations|N2O"


@pytest.fixture(scope="module")
def emissions():
    n = len(YEARS)
    out = {
        f"{species}|{sector}": np.linspace(0.1, 1.0, n)
        for species in full.SECTORAL_EMISSIONS
        for sector in SECTORS
    }
    return out | {species: np.zeros(n) for species in full.halocarbon_species()}


@pytest.fixture(scope="module")
def input_files():
    return full.load_inputs()


@pytest.fixture(scope="module")
def translated(emissions, input_files):
    return full.from_magicc7(
        TimeAxis.from_values(YEARS), emissions, input_files=input_files
    )


def setup_issues(config, inputs, years=YEARS):
    with pytest.raises(InputValidationError) as error:
        emissions_driven.setup_model(config, inputs, TimeAxis.from_values(years))
    return {(i.kind, i.location): i.message for i in error.value.issues}


def test_setup_builds_an_unrun_model(translated):
    config, inputs = translated
    model = emissions_driven.setup_model(
        config, inputs, TimeAxis.from_values(YEARS)
    ).build()

    assert not model.finished()
    model.run()
    assert model.finished()


def test_setup_model_rejects_unknown_components(translated):
    config, inputs = translated
    config = {**config, "components": {**config["components"], "Unknown": {}}}

    with pytest.raises(ValueError, match="not in this model: Unknown"):
        emissions_driven.setup_model(config, inputs, TimeAxis.from_values(YEARS))


def test_setup_model_rejects_missing_components(translated):
    config, inputs = translated
    components = dict(config["components"])
    del components["ClimateUDEB"]

    with pytest.raises(ValueError, match="no parameters for components: ClimateUDEB"):
        emissions_driven.setup_model(
            {**config, "components": components}, inputs, TimeAxis.from_values(YEARS)
        )


def test_setup_model_rejects_inputs_off_the_axis(translated):
    config, inputs = translated
    name = next(iter(inputs))
    values, unit = inputs[name]

    found = setup_issues(config, {**inputs, name: (values[:-1], unit)})

    assert set(found) == {(IssueKind.SHAPE, name)}


def test_setup_model_reports_bad_inputs_together(translated):
    config, inputs = translated
    co2 = "Emissions|CO2|Fossil"
    values, unit = inputs[co2]
    nan = np.array(values, dtype=np.float64)
    nan[100] = np.nan
    columns = np.zeros((len(YEARS), 3))

    found = setup_issues(
        config,
        {
            **inputs,
            co2: (nan, unit),
            "Emissions|CH4": (columns, "Mt CH4 / yr"),
            "Emissions|CO2|Fosil": (values, unit),
        },
    )

    assert set(found) == {
        (IssueKind.NON_FINITE, co2),
        (IssueKind.SHAPE, "Emissions|CH4"),
        (IssueKind.UNKNOWN_KEY, "Emissions|CO2|Fosil"),
    }
    assert "2 or 4 columns" in found[(IssueKind.SHAPE, "Emissions|CH4")]
    assert co2 in found[(IssueKind.UNKNOWN_KEY, "Emissions|CO2|Fosil")]


def test_setup_model_needs_components():
    found = setup_issues({}, {})

    assert set(found) == {
        (IssueKind.MISSING, "components"),
        (IssueKind.MISSING, "initial_values"),
    }


def test_setup_model_rejects_a_series_off_the_axis(translated):
    config, inputs = translated
    name = "Emissions|CO2|Fossil"
    values, unit = inputs[name]
    aligned = pd.Series(values, index=YEARS)
    emissions_driven.setup_model(
        config, {**inputs, name: (aligned, unit)}, TimeAxis.from_values(YEARS)
    )

    found = setup_issues(config, {**inputs, name: (aligned[::-1], unit)})

    assert set(found) == {(IssueKind.SHAPE, name)}


@pytest.mark.parametrize(
    "series",
    [pd.Series([], dtype=float), pd.Series(1.0, index=[f"y{y:g}" for y in YEARS])],
    ids=["empty", "text-index"],
)
def test_setup_model_names_a_series_with_a_bad_index(translated, series):
    config, inputs = translated
    name = "Emissions|CO2|Fossil"
    unit = inputs[name][1]

    found = setup_issues(config, {**inputs, name: (series, unit)})

    assert set(found) == {(IssueKind.SHAPE, name)}


@pytest.mark.parametrize("years", [YEARS + 0.5, np.arange(1750.0, 1761.0, 2.0)])
def test_check_axis_needs_whole_annual_years(years):
    with pytest.raises(InputValidationError, match="time_axis"):
        model.check_axis(TimeAxis.from_values(years))


def test_concentration_driven_follows_prescribed_concentrations(emissions, input_files):
    time_axis = TimeAxis.from_values(YEARS)
    config, inputs = full.from_magicc7(
        time_axis, emissions, input_files=input_files, concentration_driven=True
    )

    assert config["components"]["CO2Budget"]["concentration_switch_year"] == math.inf
    model = concentration_driven.setup_model(config, inputs, time_axis).run()
    for gas in ("CO2", "CH4", "N2O", "CFC11"):
        name = f"Atmospheric Concentrations|{gas}"
        np.testing.assert_array_equal(
            model.timeseries().get_timeseries_by_name(name).values()[1:],
            inputs[prescribed(name)][0][1:],
            err_msg=gas,
        )


def test_setup_model_uses_rust_forcing_agent_defaults(translated):
    config, inputs = translated
    time_axis = TimeAxis.from_values(YEARS)
    agents = copy.deepcopy(config["components"]["ForcingAggregation"]["agents"])
    for agent in agents:
        if agent["variable"] == f"{ERF}|CO2":
            del agent["regional"]
        if agent["variable"] == f"{ERF}|Solar":
            agent["sum_of"] = []
    components = {
        **config["components"],
        "ForcingAggregation": {
            **config["components"]["ForcingAggregation"],
            "agents": agents,
        },
    }

    expected = emissions_driven.setup_model(config, inputs, time_axis).run()
    actual = emissions_driven.setup_model(
        {**config, "components": components}, inputs, time_axis
    ).run()
    np.testing.assert_array_equal(
        actual.timeseries().get_timeseries_by_name(ERF).values(),
        expected.timeseries().get_timeseries_by_name(ERF).values(),
    )


def test_scalar_agents_start_from_their_prescribed_forcing():
    boxes = np.array([1.0, 2.0, 3.0, 4.0])
    first = {"A": np.float64(0.5), "B": boxes, "C": np.array([1.0, 3.0])}
    agents = [{"variable": v} for v in ("A", "B", "C", "D")]

    assert model._initial_forcing(agents, first, {}) == [
        [0.5],
        [float(boxes @ area_fractions())],
        [2.0],
        [0.0],
    ]


def test_setups_differ_only_in_switch_years(translated):
    config, inputs = translated
    time_axis = TimeAxis.from_values(YEARS)
    emissions = emissions_driven.setup_model(config, inputs, time_axis)
    concentrations = concentration_driven.setup_model(config, inputs, time_axis)

    lines = diff(emissions, concentrations).splitlines()[1:]
    assert len(lines) == 6
    assert all("switch_year" in line and line.endswith("-> inf") for line in lines)
    assert all(i.supplied for i in emissions.prescribed_inputs())


def test_run_all_matches_running_each_setup(translated):
    config, inputs = translated
    time_axis = TimeAxis.from_values(YEARS)
    setups = [
        module.setup_model(config, inputs, time_axis)
        for module in (emissions_driven, concentration_driven)
    ] * 2
    for setup, actual in zip(setups, run_all(setups, n_threads=2), strict=True):
        np.testing.assert_array_equal(
            actual.timeseries().get_timeseries_by_name(ERF).values(),
            setup.run().timeseries().get_timeseries_by_name(ERF).values(),
        )


def test_ipcctar_runs_from_an_initial_co2_below_the_olbl_range(emissions, input_files):
    time_axis = TimeAxis.from_values(YEARS)
    config, inputs = full.from_magicc7(
        time_axis,
        emissions,
        input_files=input_files,
        concentration_driven=True,
        parameters={"core_co2ch4n2o_rfmethod": "IPCCTAR"},
        initial_concentrations={"CO2": 179.0},
    )

    model = concentration_driven.setup_model(config, inputs, time_axis).run()
    forcing = model.timeseries().get_timeseries_by_name(f"{ERF}|CO2").values()

    assert np.isfinite(forcing[1:]).all()


def test_run_fails_past_the_maximum_temperature(translated):
    config, inputs = translated
    climate = {**config["components"]["ClimateUDEB"], "maximum_temperature": 0.1}
    components = {**config["components"], "ClimateUDEB": climate}
    setup = emissions_driven.setup_model(
        {**config, "components": components}, inputs, TimeAxis.from_values(YEARS)
    )

    with pytest.raises(RuntimeError, match=r"above the maximum_temperature of 0\.1 K"):
        setup.run()


@pytest.fixture(scope="module")
def fgas_emitted(emissions, input_files):
    """A translation whose short- and long-lived F-gases are emitted from the start."""
    n = len(YEARS)
    emitted = emissions | {
        species: np.linspace(5.0, 50.0, n) for species in ("HFC152a", "HFC134a", "CF4")
    }
    return full.from_magicc7(
        TimeAxis.from_values(YEARS),
        emitted,
        input_files=input_files,
        parameters={"fgas_switchfromconc2emis_year": 1750},
    )


def _with(config, component, **parameters):
    components = copy.deepcopy(config["components"])
    components[component] |= parameters
    return {**config, "components": components}


def _first_bound(config, inputs):
    """Efficacy-adjusted ERF at the first time bound, and the first-step F-gas ERF."""
    setup = emissions_driven.setup_model(config, inputs, TimeAxis.from_values(YEARS))
    fgas = setup.run().timeseries().get_four_box_timeseries_by_name(ERF_F_GASES)
    first_step = [float(v) for v in fgas.values()[1]]
    return setup.spec.initial_grid_values[EFFICACY_ADJUSTED_ERF], first_step


def test_fgas_forcing_starts_from_the_first_step(fgas_emitted):
    config, inputs = fgas_emitted
    actual, first_step = _first_bound(config, inputs)
    given = {**config, "initial_forcing": {ERF_F_GASES: first_step}}

    assert "initial_forcing" not in config
    assert all(v != 0.0 for v in first_step)
    assert actual == _first_bound(given, inputs)[0]


def test_first_fgas_forcing_follows_the_forcing_scale(fgas_emitted):
    config, inputs = fgas_emitted
    scaled = _with(config, "HalocarbonChemistry", fgas_forcing_scale=2.0)

    actual, first_step = _first_bound(scaled, inputs)
    given = {**scaled, "initial_forcing": {ERF_F_GASES: first_step}}

    assert actual == _first_bound(given, inputs)[0]
    assert actual != _first_bound(config, inputs)[0]


def test_given_initial_forcing_overrides_the_first_step(fgas_emitted):
    config, inputs = fgas_emitted
    zero = {**config, "initial_forcing": {ERF_F_GASES: [0.0] * 4}}

    assert _first_bound(zero, inputs)[0] != _first_bound(config, inputs)[0]


def test_v8_fgas_split_follows_the_global_pattern(fgas_emitted):
    config, inputs = fgas_emitted
    v8 = full.v8_fgas_split(config)
    pattern = np.asarray(
        config["components"]["HalocarbonChemistry"]["fgas_regions"]["regional_pattern"]
    )

    _, first_step = _first_bound(v8, inputs)
    _, magicc7 = _first_bound(config, inputs)

    assert v8["initial_forcing"] == {ERF_F_GASES: [0.0] * 4}
    np.testing.assert_allclose(
        np.asarray(first_step) / pattern, first_step[0] / pattern[0]
    )
    assert first_step != magicc7


def test_natural_n2o_in_balance_holds_the_starting_concentration(translated):
    config, inputs = translated
    start = config["initial_values"][N2O_CONCENTRATION]
    first_step = float(inputs["Emissions|N2O"][0][0])
    emissions = _with(config, "N2OChemistry", concentration_switch_year=None)
    inputs = {k: v for k, v in inputs.items() if k != prescribed(N2O_CONCENTRATION)}

    natural = N2OChemistryBuilder.from_parameters(
        emissions["components"]["N2OChemistry"]
    ).natural_emissions_in_balance(start, first_step)
    balanced = _with(emissions, "N2OChemistry", natural_emissions=natural)
    model = emissions_driven.setup_model(
        balanced, inputs, TimeAxis.from_values(YEARS)
    ).run()
    concentration = model.timeseries().get_timeseries_by_name(N2O_CONCENTRATION)

    assert concentration.values()[1] == pytest.approx(start, rel=1e-14)


def test_natural_n2o_in_balance_follows_the_lifetime(translated):
    config, _ = translated
    n2o = config["components"]["N2OChemistry"]
    shorter = {**n2o, "initial_n2o_lifetime": n2o["initial_n2o_lifetime"] / 2}

    def balance(parameters):
        builder = N2OChemistryBuilder.from_parameters(parameters)
        return builder.natural_emissions_in_balance(270.0, 1.0)

    assert balance(shorter) + 1.0 == pytest.approx(2 * (balance(n2o) + 1.0))


@pytest.mark.parametrize("bugs", ["7.5.3", ()])
def test_nitrogen_deposition_from_emissions_matches_a_prescribed_deposition(
    emissions, input_files, bugs
):
    time_axis = TimeAxis.from_values(YEARS)
    parameters = {"ncycle_apply": 1, "ncycle_apply_ndep_from_emis": 1}
    config, inputs = full.from_magicc7(
        time_axis, emissions, parameters=parameters, input_files=input_files, bugs=bugs
    )
    nitrogen = config["components"]["TerrestrialCarbon"]["nitrogen"]
    deposition = nitrogen["deposition_from_emissions"]
    nh3 = "Fossil" if deposition["include_nh3_counted_twice"] else "Biosphere"
    e = {name: values for name, (values, _) in inputs.items()}
    prescribed_deposition = (
        1e-3
        * deposition["land_fraction"]
        * (
            e["Emissions|NOx|Fossil"]
            + e["Emissions|NOx|Biosphere"]
            + (e["Emissions|NH3|Fossil"] + e[f"Emissions|NH3|{nh3}"])
        )
    )
    file_driven = _with(
        config,
        "TerrestrialCarbon",
        nitrogen={**nitrogen, "deposition_from_emissions": None},
    )
    file_inputs = {
        **{k: v for k, v in inputs.items() if not k.startswith("Emissions|NH3")},
        NITROGEN_FLUX_DEPOSITION: (prescribed_deposition, "Gt N/yr"),
    }

    def pool(config, inputs):
        model = emissions_driven.setup_model(config, inputs, time_axis).run()
        return model.timeseries().get_timeseries_by_name("Nitrogen Pool").values()

    assert NITROGEN_FLUX_DEPOSITION not in inputs
    assert deposition["include_nh3_counted_twice"] == (bugs == "7.5.3")
    np.testing.assert_array_equal(pool(config, inputs), pool(file_driven, file_inputs))


def test_nitrogen_deposition_follows_the_land_fraction(emissions, input_files):
    time_axis = TimeAxis.from_values(YEARS)
    parameters = {"ncycle_apply": 1, "ncycle_apply_ndep_from_emis": 1}
    config, inputs = full.from_magicc7(
        time_axis, emissions, parameters=parameters, input_files=input_files
    )
    nitrogen = config["components"]["TerrestrialCarbon"]["nitrogen"]
    halved = _with(
        config,
        "TerrestrialCarbon",
        nitrogen={
            **nitrogen,
            "deposition_from_emissions": {
                **nitrogen["deposition_from_emissions"],
                "land_fraction": 0.35,
            },
        },
    )

    def pool(config):
        model = emissions_driven.setup_model(config, inputs, time_axis).run()
        return model.timeseries().get_timeseries_by_name("Nitrogen Pool").values()

    assert not np.array_equal(pool(config)[1:], pool(halved)[1:])
