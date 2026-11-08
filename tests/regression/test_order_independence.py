"""Registration order does not change a model's results."""

from __future__ import annotations

import json
from functools import cache

import numpy as np
import pytest

from regression import full_run, test_fair
from rscm.fair import run as fair_run
from rscm.magicc import concentration_driven, emissions_driven, full
from rscm.magicc import model as magicc_model
from rscm.names import (
    ANNUAL_MEAN_SURFACE_AIR_TEMPERATURE_CHANGE,
    ATMOSPHERIC_CONCENTRATIONS_CO2,
    SURFACE_AIR_TEMPERATURE_CHANGE,
)


def _outputs(model):
    ts = model.timeseries()
    out = {}
    getters = (
        ts.get_timeseries_by_name,
        ts.get_four_box_timeseries_by_name,
        ts.get_hemispheric_timeseries_by_name,
    )
    for name in ts.names():
        series = next(s for s in (get(name) for get in getters) if s is not None)
        out[name] = np.asarray(series.values())
    return out


def _assert_identical(actual, expected):
    assert actual.keys() == expected.keys()
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values, err_msg=name)


def _lags(model):
    coupling = json.loads(model.coupling("json"))
    return {(e["variable"], e["consumer"]): e["lagged"] for e in coupling["edges"]}


def _shuffle(components, seed):
    components = list(components)
    return [
        components[i] for i in np.random.default_rng(seed).permutation(len(components))
    ]


def _shuffled(function, seed):
    def wrapper(*args, **kwargs):
        return dict(_shuffle(function(*args, **kwargs).items(), seed))

    return wrapper


@cache
def _input_files():
    return full.load_inputs(full_run.load().parameters)


def _run_magicc7(*, concentrations=False, **parameters):
    run = full_run.load()
    config, inputs = full.from_magicc7(
        run.time_axis,
        full_run.emissions(run),
        parameters={**run.parameters, **parameters},
        input_files=_input_files(),
        concentration_driven=concentrations,
    )
    model = concentration_driven if concentrations else emissions_driven
    return model.setup_model(config, inputs, run.time_axis).run()


MAGICC7_CASES = {
    "reference": {},
    "permafrost": {"pf_apply": True},
    "sea level": {"out_sealevel": True},
    "permafrost and sea level": {"pf_apply": True, "out_sealevel": True},
    "concentration-driven": {"concentrations": True, "pf_apply": True},
}


@pytest.mark.parametrize("parameters", MAGICC7_CASES.values(), ids=MAGICC7_CASES)
def test_magicc7_shuffled(monkeypatch, parameters):
    expected = _outputs(_run_magicc7(**parameters))
    original = magicc_model.component_specs
    for seed in range(3):
        monkeypatch.setattr(magicc_model, "component_specs", _shuffled(original, seed))
        _assert_identical(_outputs(_run_magicc7(**parameters)), expected)


@pytest.mark.parametrize("case", ["ssp245", "ssp245_prescribed", "calibrated_1939"])
def test_fair_shuffled(monkeypatch, case):
    run = test_fair.load(case)
    inputs = test_fair._inputs(run.config, run.timebounds, run.timepoints)
    expected = _outputs(fair_run.setup_model(run.config, inputs, run.time_axis).run())
    original = fair_run.component_specs
    for seed in range(3):
        monkeypatch.setattr(fair_run, "component_specs", _shuffled(original, seed))
        _assert_identical(
            _outputs(fair_run.setup_model(run.config, inputs, run.time_axis).run()),
            expected,
        )


def test_magicc7_coupling_lags():
    """Forcing sees this year's CO2 and sea level this year's temperature."""
    lags = _lags(_run_magicc7(out_sealevel=True))
    assert not lags[ATMOSPHERIC_CONCENTRATIONS_CO2, "GHGForcing"]
    assert not lags[ANNUAL_MEAN_SURFACE_AIR_TEMPERATURE_CHANGE, "SeaLevelRise"]
    for consumer in ("TerrestrialCarbon", "OceanCarbon"):
        assert lags[ATMOSPHERIC_CONCENTRATIONS_CO2, consumer]
        assert lags[SURFACE_AIR_TEMPERATURE_CHANGE, consumer]
