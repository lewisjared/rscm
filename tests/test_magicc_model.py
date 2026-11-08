"""Building the emissions- and concentration-driven MAGICC models."""

import math

import numpy as np
import pytest

from rscm.core import TimeAxis
from rscm.magicc import concentration_driven, emissions_driven, full
from rscm.names import SECTORS, prescribed

# MAGICC7 harmonises prescribed forcing in 2019.
YEARS = np.arange(1750.0, 2021.0)


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


def test_build_model_does_not_run(translated):
    config, inputs = translated
    model = emissions_driven.build_model(config, inputs, TimeAxis.from_values(YEARS))

    assert not model.finished()
    model.run()
    assert model.finished()


def test_build_model_rejects_unknown_components(translated):
    config, inputs = translated
    config = {**config, "components": {**config["components"], "Unknown": {}}}

    with pytest.raises(ValueError, match="not in this model: Unknown"):
        emissions_driven.build_model(config, inputs, TimeAxis.from_values(YEARS))


def test_build_model_rejects_missing_components(translated):
    config, inputs = translated
    components = dict(config["components"])
    del components["ClimateUDEB"]

    with pytest.raises(ValueError, match="no parameters for components: ClimateUDEB"):
        emissions_driven.build_model(
            {**config, "components": components}, inputs, TimeAxis.from_values(YEARS)
        )


def test_build_model_rejects_inputs_off_the_axis(translated):
    config, inputs = translated
    name = next(iter(inputs))
    values, unit = inputs[name]

    with pytest.raises(ValueError, match=f"input '{name}'"):
        emissions_driven.build_model(
            config, {**inputs, name: (values[:-1], unit)}, TimeAxis.from_values(YEARS)
        )


def test_concentration_driven_follows_prescribed_concentrations(emissions, input_files):
    time_axis = TimeAxis.from_values(YEARS)
    config, inputs = full.from_magicc7(
        time_axis, emissions, input_files=input_files, concentration_driven=True
    )

    assert config["components"]["CO2Budget"]["concentration_switch_year"] == math.inf
    model = concentration_driven.run_model(config, inputs, time_axis)
    for gas in ("CO2", "CH4", "N2O", "CFC11"):
        name = f"Atmospheric Concentrations|{gas}"
        np.testing.assert_array_equal(
            model.timeseries().get_timeseries_by_name(name).values()[1:],
            inputs[prescribed(name)][0][1:],
            err_msg=gas,
        )
