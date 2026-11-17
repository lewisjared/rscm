"""Set-up helpers of the full MAGICC7 chain."""

from types import SimpleNamespace

import numpy as np
import pytest

from rscm.core import GridType
from rscm.magicc import full, model
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


def test_schema_keeps_the_finest_grid():
    erf = SimpleNamespace(name="ERF", unit="W/m^2", grid=GridType.Scalar)
    temperature = SimpleNamespace(name="T", unit="K", grid=GridType.FourBox)
    component = SimpleNamespace(definitions=lambda: [erf, temperature])

    schema = model._schema([component], {"ERF": (np.zeros((3, 2)), "W/m^2")})

    assert schema.variables["ERF"].grid == GridType.Hemispheric
    assert schema.variables["T"].grid == GridType.FourBox


def test_reproduced_bugs():
    assert full.reproduced_bugs("7.5.3") == set(full.MAGICC7_BUGS)
    assert full.reproduced_bugs(["unset_lamcalc_error"]) == {"unset_lamcalc_error"}
    assert full.reproduced_bugs(()) == set()
    with pytest.raises(ValueError, match="bugs must be one of"):
        full.reproduced_bugs("7.6.0")
    with pytest.raises(ValueError, match="unknown MAGICC7 bugs"):
        full.reproduced_bugs(["lamcalc"])


def test_nitrogen_deposition_fixes_nh3_double_count():
    p = full.default_parameters() | {
        "ncycle_apply_ndep_from_emis": 1,
        "ncycle_apply_prognostic_nfix": 1,
        "ncycle_frac_ndep_land": 1.0,
    }
    n = len(YEARS)
    emissions = {
        "NOx|Fossil": np.zeros(n),
        "NOx|Biosphere": np.zeros(n),
        "NH3|Fossil": np.ones(n),
        "NH3|Biosphere": np.full(n, 2.0),
    }

    (reproduced,) = full.nitrogen_inputs(YEARS, emissions, p, {}).values()
    (fixed,) = full.nitrogen_inputs(YEARS, emissions, p, {}, bugs=()).values()

    np.testing.assert_allclose(reproduced, 2e-3)
    np.testing.assert_allclose(fixed, 3e-3)
