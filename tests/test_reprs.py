"""Reprs of the core Python classes name their type and key fields."""

import numpy as np

from rscm.component import Component, Input, Output
from rscm.core import (
    GridType,
    InterpolationStrategy,
    ModelBuilder,
    PythonComponent,
    TimeAxis,
    Timeseries,
    VariableDefinition,
    VariableRole,
)
from rscm.two_layer import TwoLayerBuilder

AXIS = TimeAxis.from_values(np.arange(2000.0, 2003.0))


class Doubler(Component):
    x = Input("X", unit="K")
    y = Output("Y", unit="K")

    def solve(self, t_current, t_next, inputs):
        return self.Outputs(y=2 * inputs.x.get())


def test_time_axis_repr():
    assert repr(AXIS) == "TimeAxis(len=3, start=2000.0, end=2003.0)"


def test_variable_definition_repr():
    definition = VariableDefinition("X", "K", VariableRole.Input, GridType.FourBox)
    assert repr(definition) == (
        "VariableDefinition(name='X', unit='K', role=Input, grid=FourBox)"
    )


def test_python_component_repr():
    assert repr(PythonComponent(Doubler())) == (
        "PythonComponent(Doubler, inputs=['X'], outputs=['Y'])"
    )


def test_rust_component_repr():
    component = TwoLayerBuilder.from_parameters(
        {
            "lambda0": 1.0,
            "a": 0.0,
            "deep_ocean_efficacy": 1.0,
            "ocean_heat_transfer": 0.7,
            "heat_capacity_surface": 8.0,
            "heat_capacity_deep": 100.0,
        }
    ).build()
    assert repr(component).startswith("RustComponent(TwoLayer, inputs=[")


def test_model_builder_and_model_repr():
    builder = (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_component(Doubler())
        .with_prescribed_variable(
            "X", Timeseries(np.ones(3), AXIS, "K", InterpolationStrategy.Linear)
        )
    )
    assert repr(builder) == (
        "ModelBuilder(time_axis=TimeAxis(len=3, start=2000.0, end=2003.0))"
    )

    model = builder.build()
    assert repr(model) == "Model(current_time=2000.0, finished=False)"
