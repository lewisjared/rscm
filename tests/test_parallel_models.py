"""Tests for building and running models in parallel."""

import subprocess
import sys
import textwrap

import numpy as np
import numpy.testing as npt
import pytest

from rscm._lib.two_layer import TwoLayerBuilder
from rscm.component import Component, Input, Output
from rscm.core import (
    InterpolationStrategy,
    ModelBuilder,
    ThreadPool,
    TimeAxis,
    Timeseries,
    build_models,
    run_builders,
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


def scaled_builder(factor: float) -> ModelBuilder:
    erf = Timeseries(np.ones(len(AXIS)), AXIS, "W/m^2", InterpolationStrategy.Previous)
    return (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_component(Scaled(factor))
        .with_prescribed_variable("Effective Radiative Forcing", erf)
    )


def scaled(factor: float):
    return scaled_builder(factor).build()


def scaled_output(model) -> np.ndarray:
    return model.timeseries().get_timeseries_by_name("Scaled Forcing").values()


@pytest.mark.parametrize("n_threads", [None, 1, 4])
def test_parallel_matches_serial(n_threads):
    builders = [two_layer(lambda0) for lambda0 in np.linspace(0.5, 2.0, 12)]
    models = build_models(builders, n_threads=n_threads)
    run_models(models, n_threads=n_threads)

    for builder, model in zip(builders, models):
        serial = builder.build()
        serial.run()
        assert model.finished()
        npt.assert_array_equal(temperature(model), temperature(serial))


def test_python_components_run_in_parallel_threads():
    models = [scaled(float(f)) for f in range(8)]
    run_models(models)

    for factor, model in enumerate(models):
        npt.assert_array_equal(scaled_output(model)[1:], float(factor))


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
        run_models([scaled(1.0)], n_threads=0)


def test_both_threads_and_pool_is_rejected():
    with pytest.raises(ValueError, match="not both"):
        run_models([scaled(1.0)], n_threads=2, pool=ThreadPool(2))


def test_thread_pool():
    assert ThreadPool(3).n_threads == 3
    assert repr(ThreadPool(2)) == "ThreadPool(n_threads=2)"
    assert ThreadPool().n_threads >= 1
    with pytest.raises(ValueError, match="at least 1"):
        ThreadPool(0)


@pytest.mark.parametrize("pool", [None, ThreadPool(1), ThreadPool(4)])
def test_run_builders_matches_serial_and_reuses_a_pool(pool):
    lambdas = np.linspace(0.5, 2.0, 12)
    for _ in range(2):
        models = run_builders((two_layer(x) for x in lambdas), pool=pool)
        assert len(models) == len(lambdas)
        for lambda0, model in zip(lambdas, models, strict=True):
            serial = two_layer(lambda0).build()
            serial.run()
            assert model.finished()
            npt.assert_array_equal(temperature(model), temperature(serial))


def test_run_builders_leaves_builders_unchanged():
    builders = [two_layer(1.0), two_layer(2.0)]
    first = run_builders(builders, n_threads=2)
    second = run_builders(builders, n_threads=2)
    for a, b in zip(first, second, strict=True):
        npt.assert_array_equal(temperature(a), temperature(b))


def test_run_builders_with_an_empty_iterable():
    assert run_builders(iter([])) == []


def test_run_builders_reports_build_then_run_failures():
    no_time_axis = ModelBuilder().with_component(Scaled(1.0))
    with pytest.raises(ValueError, match=r"1 of 3 models failed to build\. Model 1:"):
        run_builders([scaled_builder(1.0), no_time_axis, scaled_builder(-1.0)])

    with pytest.raises(RuntimeError, match=r"1 of 3 models failed to run\. Model 1:"):
        run_builders([scaled_builder(1.0), scaled_builder(-1.0), scaled_builder(2.0)])


def test_run_builders_raises_the_iterables_error():
    def builders():
        yield scaled_builder(1.0)
        raise KeyError("no more")  # noqa: TRY003

    with pytest.raises(KeyError, match="no more"):
        run_builders(builders())

    with pytest.raises(TypeError, match="ModelBuilder"):
        run_builders([scaled_builder(1.0), "not a builder"])


def test_run_builders_with_python_components_does_not_deadlock():
    # A subprocess, because a deadlock holding the GIL would also hang a test thread.
    script = textwrap.dedent(
        """
        import numpy as np
        from rscm.component import Component, Input, Output
        from rscm.core import (
            InterpolationStrategy, ModelBuilder, ThreadPool, TimeAxis, Timeseries,
            run_builders,
        )

        axis = TimeAxis.from_values(np.arange(1850.0, 2000.0, 5))

        class Scaled(Component):
            forcing = Input("Effective Radiative Forcing", unit="W/m^2")
            out = Output("Scaled Forcing", unit="W/m^2")

            def __init__(self, factor):
                self.factor = factor

            def solve(self, t_current, t_next, inputs):
                return self.Outputs(out=self.factor * inputs.forcing.get())

        def builders():
            erf = Timeseries(
                np.ones(len(axis)), axis, "W/m^2", InterpolationStrategy.Previous
            )
            for factor in range(32):
                sum(range(10_000))
                yield (
                    ModelBuilder()
                    .with_time_axis(axis)
                    .with_component(Scaled(float(factor)))
                    .with_prescribed_variable("Effective Radiative Forcing", erf)
                )

        models = run_builders(builders(), pool=ThreadPool(4))
        for factor, model in enumerate(models):
            series = model.timeseries().get_timeseries_by_name("Scaled Forcing")
            assert (series.values()[1:] == factor).all()
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)  # noqa: S603


class Interrupting(Component):
    forcing = Input("Effective Radiative Forcing", unit="W/m^2")
    out = Output("Scaled Forcing", unit="W/m^2")

    def solve(self, t_current, t_next, inputs):
        raise KeyboardInterrupt


def test_run_builders_raises_an_interrupt_before_a_build_failure():
    erf = Timeseries(np.ones(len(AXIS)), AXIS, "W/m^2", InterpolationStrategy.Previous)
    unbuildable = ModelBuilder().with_time_axis(AXIS).with_component(Scaled(1.0))
    interrupting = (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_component(Interrupting())
        .with_prescribed_variable("Effective Radiative Forcing", erf)
    )

    with pytest.raises(KeyboardInterrupt):
        run_builders([unbuildable, interrupting])
