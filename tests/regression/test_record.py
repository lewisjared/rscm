"""A MAGICC7 run resumed from a mid-run record matches the uninterrupted run exactly."""

import numpy as np
import pytest

from regression import full_run
from rscm.core import Model, Record
from rscm.magicc import emissions_driven, full


@pytest.fixture(scope="module")
def setup():
    run = full_run.load_default()
    config, inputs = full.from_magicc7(
        run.time_axis, full_run.emissions(run), input_files=full.load_inputs()
    )
    return emissions_driven.setup_model(config, inputs, run.time_axis)


def _assert_identical(actual, expected):
    actual, expected = actual.timeseries(), expected.timeseries()
    assert actual.names() == expected.names()
    for name in expected.names():
        np.testing.assert_array_equal(
            actual[name].values(), expected[name].values(), err_msg=name
        )


@pytest.mark.parametrize(
    "resume",
    [
        lambda record: Model.restore(Record.from_json(record.to_json())),
        Model.branch,
    ],
    ids=["restore", "branch"],
)
def test_mid_run_record_resumes_to_the_uninterrupted_run(setup, resume):
    uninterrupted = setup.builder().build()
    uninterrupted.run()

    model = setup.builder().build()
    for _ in range(120):
        model.step()
    resumed = resume(model.checkpoint())
    resumed.run()

    _assert_identical(resumed, uninterrupted)
