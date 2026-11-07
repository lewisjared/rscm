"""Set-up helpers of the full MAGICC7 chain."""

import numpy as np
import pytest

from rscm.core import TimeAxis
from rscm.magicc import full
from rscm.magicc import prescribed as pr
from rscm.names import Convention

YEARS = np.arange(2000.0, 2006.0)


@pytest.fixture
def budget_inputs():
    parameters = full.default_parameters() | {
        "ch4_tautot_init": 10.0,
        "ch4_ppb2tgch4": 2.0,
        "ch4_mixboxsize": 1.0,
        "ch4_lastbudgetyear": 2004,
        "ch4_budget_avgyears": 2,
        "n2o_tauinit": 100.0,
        "n2o_ppb2tgn": 5.0,
        "n2o_lastbudgetyear": 2005,
        "n2o_budget_avgyears": 3,
    }
    emissions = {
        "CH4|Fossil": np.ones(len(YEARS)),
        "CH4|Biosphere": 10.0 * (YEARS - 2000),
        "N2O|Fossil": np.ones(len(YEARS)),
        "N2O|Biosphere": np.full(len(YEARS), 2.0),
    }
    inputs = {
        "file_ch4_conc": pr.AerosolHistory(YEARS, 1000.0 + 10.0 * (YEARS - 2000)),
        "file_n2o_conc": pr.AerosolHistory(YEARS, np.full(len(YEARS), 300.0)),
    }
    return parameters, emissions, inputs


def test_natural_emissions_close_budget(budget_inputs):
    parameters, emissions, inputs = budget_inputs

    out = full.natural_emissions(
        YEARS, emissions, parameters, inputs, Convention.MAGICC7
    )

    # CH4 over 2002 to 2004: 2 * mean(10 + [1025, 1035] / 10) less mean emissions 31
    assert out["CH4"] == pytest.approx(195.0, rel=1e-12)
    # N2O: 5 * 300 / 100 less emissions of 3
    assert out["N2O"] == pytest.approx(12.0, rel=1e-12)


def test_natural_emissions_convert_rscm_units(budget_inputs):
    parameters, emissions, inputs = budget_inputs
    kt_n2o = full.magicc7_units({"N2O|Fossil": np.ones(1)})["N2O|Fossil"][0]
    rscm = {k: v / kt_n2o if k.startswith("N2O") else v for k, v in emissions.items()}

    out = full.natural_emissions(YEARS, rscm, parameters, inputs)

    assert out["N2O"] == pytest.approx(12.0, rel=1e-12)


def test_natural_emissions_skip_budget_outside_years(budget_inputs):
    parameters, emissions, inputs = budget_inputs
    years = YEARS[3:]
    emissions = {k: v[3:] for k, v in emissions.items()}

    out = full.natural_emissions(years, emissions, parameters, inputs)

    assert out == {}


def test_load_inputs_skips_missing_concentration_files():
    p = {**full.default_parameters(), "file_co2_conc": "missing_co2_conc.IN"}

    inputs = full.load_inputs(p)

    assert "file_co2_conc" not in inputs
    with pytest.raises(KeyError, match="file_co2_conc"):
        full.prescribed_concentrations(np.arange(1750.0, 1800.0), p, inputs)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("component_parameters", {"ClimateUDEB": {}}),  # naming: allow
        ("initial_concentrations", {"HFC4310": 0.0}),  # naming: allow
    ],
)
def test_run_magicc7_rejects_unknown_keys(argument, value):
    time_axis = TimeAxis.from_values(np.arange(1750.0, 1753.0))
    with pytest.raises(ValueError, match=f"unknown {argument}"):
        full.run_magicc7(time_axis, {}, **{argument: value})
