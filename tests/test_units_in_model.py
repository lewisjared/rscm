"""Units a model stores variables in, and conversions on the way in and out."""

import numpy as np
import numpy.testing as npt
import pytest

from rscm.component import Component, Input, Output, State
from rscm.core import (
    FourBoxTimeseries,
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
    VariableSchema,
)

AXIS = TimeAxis.from_values(np.arange(2000.0, 2004.0))


class Producer(Component):
    forcing = Output("Forcing", unit="kW/m^2")

    def solve(self, t_current, t_next, inputs):
        return self.Outputs(forcing=1.0)


class Consumer(Component):
    forcing = Input("Forcing", unit="kW/m^2")
    copy = Output("Forcing Copy", unit="kW/m^2")

    def solve(self, t_current, t_next, inputs):
        return self.Outputs(copy=inputs.forcing.get())


class Accumulator(Component):
    heat = State("Heat", unit="kJ")

    def solve(self, t_current, t_next, inputs):
        return self.Outputs(heat=inputs.heat.get() + 1.0)


def schema(**units: str) -> VariableSchema:
    s = VariableSchema()
    for name, unit in units.items():
        s.add_variable(name.replace("_", " "), unit)
    return s


def stored(model, name):
    ts = model.timeseries().get_timeseries_by_name(name)
    return ts.values(), ts.unit


def test_python_outputs_are_stored_in_the_schema_unit():
    model = (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_schema(schema(Forcing="W/m^2", Forcing_Copy="W/m^2"))
        .with_component(Producer())
        .with_component(Consumer())
        .build()
    )
    model.run()

    values, unit = stored(model, "Forcing")
    assert unit == "W/m^2"
    npt.assert_allclose(values[1:], 1000.0)
    values, unit = stored(model, "Forcing Copy")
    assert unit == "W/m^2"
    npt.assert_allclose(values[1:], 1000.0)


def test_coupling_shows_conversion_direction():
    model = (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_schema(schema(Forcing="W/m^2", Forcing_Copy="W/m^2"))
        .with_component(Producer())
        .with_component(Consumer())
        .build()
    )
    text = model.coupling("plain")

    assert "Forcing kW/m^2 -> W/m^2 (write by Producer, x1000)" in text
    assert "Forcing W/m^2 -> kW/m^2 (read by Consumer, x0.001)" in text


@pytest.mark.parametrize(
    "initial, expected",
    [
        pytest.param(2000.0, 2000.0, id="bare value in the stored unit"),
        pytest.param((2.0, "kJ"), 2000.0, id="value with a unit"),
    ],
)
def test_initial_values(initial, expected):
    model = (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_schema(schema(Heat="J"))
        .with_component(Accumulator())
        .with_initial_values({"Heat": initial})
        .build()
    )
    model.run()

    values, unit = stored(model, "Heat")
    assert unit == "J"
    npt.assert_allclose(values, expected + 1000.0 * np.arange(4))


def test_bare_initial_value_replaces_one_with_a_unit():
    model = (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_schema(schema(Heat="J"))
        .with_component(Accumulator())
        .with_initial_values({"Heat": (2.0, "kJ")})
        .with_initial_values({"Heat": 3.0})
        .build()
    )

    assert model.timeseries().get_timeseries_by_name("Heat").at(0) == 3.0


def test_initial_value_in_an_incompatible_unit_fails():
    builder = (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_component(Accumulator())
        .with_initial_values({"Heat": (2.0, "GtC")})
    )
    with pytest.raises(ValueError, match="Incompatible units"):
        builder.build()


def test_convert_to():
    ts = Timeseries(
        np.array([1.0, 2.0, 3.0, 4.0]), AXIS, "GtC/yr", InterpolationStrategy.Linear
    )

    converted = ts.convert_to("MtC/yr")

    assert converted.unit == "MtC/yr"
    npt.assert_allclose(converted.values(), [1000.0, 2000.0, 3000.0, 4000.0])
    assert ts.unit == "GtC/yr"
    with pytest.raises(ValueError, match="Cannot convert"):
        ts.convert_to("W/m^2")


def test_convert_to_grid():
    ts = FourBoxTimeseries(
        np.ones((4, 4)),
        AXIS,
        "W/m^2",
        InterpolationStrategy.Linear,
    )

    converted = ts.convert_to("mW/m^2")

    assert converted.unit == "mW/m^2"
    npt.assert_allclose(converted.values(), 1000.0)
