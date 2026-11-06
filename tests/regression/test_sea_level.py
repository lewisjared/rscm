"""Sea level rise of the full RSCM MAGICC chain against MAGICC7's default run."""

import pytest

from regression import full_run

# RSCM outputs and their MAGICC7 names.
CONTRIBUTIONS = {
    "Sea Level Rise|Thermal Expansion": "SLR_EXPANSION",
    "Sea Level Rise|Glaciers": "SLR_GL",
    "Sea Level Rise|Greenland|Surface Mass Balance": "SLR_GIS_SMB",
    "Sea Level Rise|Greenland|Solid Ice Discharge": "SLR_GIS_SID",
    "Sea Level Rise|Antarctica|Surface Mass Balance": "SLR_AIS_SMB",
    "Sea Level Rise|Antarctica|Solid Ice Discharge": "SLR_AIS_SID",
    "Sea Level Rise": "Sea Level Rise",
}


@pytest.fixture(scope="module")
def sea_level():
    run = full_run.load_sealevel()
    return run, full_run.run_coupled(run)


@pytest.mark.parametrize("name", CONTRIBUTIONS)
def test_contribution(sea_level, name):
    run, model = sea_level
    full_run.assert_close(full_run.output(model, name), run.series(CONTRIBUTIONS[name]))


def test_sea_level_off_by_default():
    model = full_run.default_model()
    assert model.timeseries().get_timeseries_by_name("Sea Level Rise") is None
