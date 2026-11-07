"""Tests for building and running models in parallel."""

import numpy as np
import numpy.testing as npt
import pytest

from rscm._lib.two_layer import TwoLayerBuilder
from rscm.component import Component, Input, Output
from rscm.core import (
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
    build_models,
    run_models,
)

AXIS = TimeAxis.from_values(np.arange(1850.0, 2000.0, 5))


def two_layer(lambda0: float) -> ModelBuilder:
    component = TwoLayerBuilder.from_parameters(
        {
            "lambda0": lambda0,
            "a": 0.0,
            "deep_ocean_efficacy": 1.0,
            "ocean_heat_transfer": 0.7,
            "heat_capacity_deep": 100.0,
            "heat_capacity_surface": 8.0,
        }
    ).build()
    erf = Timeseries(
        np.linspace(0.0, 3.0, len(AXIS)), AXIS, "W/m^2", InterpolationStrategy.Next
    )
    builder = ModelBuilder()
    builder.with_time_axis(AXIS).with_component(component)
    builder.with_prescribed_variable("Effective Radiative Forcing", erf)
    builder.with_initial_values(
        {"Surface Air Temperature Change": 0.0, "Ocean Temperature|Layer 2": 0.0}
    )
    return builder


def temperature(model) -> np.ndarray:
    return (
        model.timeseries()
        .get_timeseries_by_name("Surface Air Temperature Change")
        .values()
    )


class Scaled(Component):
    forcing = Input("Effective Radiative Forcing", unit="W/m^2")
    out = Output("Scaled Forcing", unit="W/m^2")

    def __init__(self, factor: float):
        self.factor = factor

    def solve(self, t_current, t_next, inputs):
        if self.factor < 0:
            raise ValueError("negative factor")  # noqa: TRY003
        return self.Outputs(out=self.factor * inputs.forcing.get())


def scaled(factor: float):
    erf = Timeseries(np.ones(len(AXIS)), AXIS, "W/m^2", InterpolationStrategy.Previous)
    return (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_component(Scaled(factor))
        .with_prescribed_variable("Effective Radiative Forcing", erf)
        .build()
    )


@pytest.mark.parametrize("threads", [None, 1, 4])
def test_parallel_matches_serial(threads):
    builders = [two_layer(lambda0) for lambda0 in np.linspace(0.5, 2.0, 12)]
    models = build_models(builders, threads=threads)
    run_models(models, threads=threads)

    for builder, model in zip(builders, models):
        serial = builder.build()
        serial.run()
        assert model.finished()
        npt.assert_array_equal(temperature(model), temperature(serial))


def test_python_components_run_in_parallel_threads():
    models = [scaled(float(f)) for f in range(8)]
    run_models(models)

    for factor, model in enumerate(models):
        out = model.timeseries().get_timeseries_by_name("Scaled Forcing").values()
        npt.assert_array_equal(out[1:], float(factor))


def test_failure_names_the_model_and_runs_the_rest():
    models = [scaled(1.0), scaled(-1.0), scaled(2.0)]

    with pytest.raises(RuntimeError, match=r"1 of 3 models failed to run\. Model 1:"):
        run_models(models)

    assert models[0].finished()
    assert models[2].finished()


def test_a_model_listed_twice_is_rejected():
    model = scaled(1.0)

    with pytest.raises(RuntimeError, match="borrow"):
        run_models([model, model])


def test_zero_threads_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        run_models([scaled(1.0)], threads=0)
