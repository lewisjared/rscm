"""A FaIR component with a NaN parameter round-trips through a record."""

import copy
import json
import math
import pickle

import numpy as np
import pytest

from rscm.core import (
    InterpolationStrategy,
    Model,
    ModelBuilder,
    Record,
    TimeAxis,
    Timeseries,
)
from rscm.fair import AerosolCloudBuilder

AXIS = TimeAxis.from_values(np.arange(2000.0, 2006.0))
FORCING = "Effective Radiative Forcing|Aerosols|Indirect Effect"


def _prescribed(values, unit: str) -> Timeseries:
    return Timeseries(
        np.asarray(values, dtype=float), AXIS, unit, InterpolationStrategy.Previous
    )


def build() -> Model:
    # A NaN baseline drops the SOx term from the pre-industrial effect, as FaIR does.
    cloud = AerosolCloudBuilder.from_parameters(
        {
            "forcing": FORCING,
            "drivers": [
                {
                    "variable": "Emissions|SOx",
                    "unit": "Mt SO2/yr",
                    "driver": "emissions",
                    "baseline": math.nan,
                    "shape": 0.02,
                },
                {
                    "variable": "Emissions|BC",
                    "unit": "Mt BC/yr",
                    "driver": "emissions",
                    "baseline": 2.0,
                    "shape": 0.1,
                },
            ],
            "scale": -2.0,
            "temperature_feedback": -0.1,
        }
    ).build()
    return (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_component(cloud)
        .with_prescribed_variable(
            "Emissions|SOx", _prescribed(np.linspace(50.0, 100.0, 6), "Mt SO2/yr")
        )
        .with_prescribed_variable(
            "Emissions|BC", _prescribed(np.linspace(2.0, 8.0, 6), "Mt BC/yr")
        )
        .with_prescribed_variable(
            "Surface Air Temperature Change", _prescribed(np.linspace(0, 1, 6), "K")
        )
        .build()
    )


@pytest.mark.parametrize(
    "round_trip",
    [
        lambda m: Model.restore(Record.from_json(m.checkpoint().to_json())),
        copy.deepcopy,
        lambda m: pickle.loads(pickle.dumps(m)),  # noqa: S301
    ],
    ids=["record", "deepcopy", "pickle"],
)
def test_nan_baseline_resumes_to_the_uninterrupted_run(round_trip):
    uninterrupted = build()
    uninterrupted.run()
    expected = uninterrupted.timeseries()[FORCING].values()
    assert np.isfinite(expected[1:]).all()

    model = build()
    model.step()
    model.step()
    resumed = round_trip(model)
    resumed.run()

    np.testing.assert_array_equal(resumed.timeseries()[FORCING].values(), expected)


def test_record_writes_nan_as_a_string():
    record = json.loads(build().checkpoint().to_json())
    (component,) = (
        c for c in record["builder"]["components"] if c["type"] == "fair.AerosolCloud"
    )

    assert component["parameters"]["drivers"][0]["baseline"] == "NaN"
