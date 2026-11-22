"""Labels key lags and coupling, so instances of one type couple independently."""

import json
from typing import Any

import numpy as np
import pytest

from rscm.component import Component, Input, Output
from rscm.core import (
    ModelBuilder,
    PythonComponent,
    TimeAxis,
    VariableDefinition,
    VariableRole,
)
from rscm.spec import ComponentSpec, Lag, ModelSpec

AXIS = TimeAxis.from_values(np.arange(2000.0, 2004.0))


class Counter:
    """Adds one joule to ``Energy`` every step."""

    def definitions(self) -> list[VariableDefinition]:
        return [VariableDefinition("Energy", "J", VariableRole.State)]

    def solve(self, t_current: float, t_next: float, inputs: Any) -> dict[str, float]:
        return {"Energy": inputs["Energy"] + 1.0}


class Reader:
    """Copies ``Energy``, read in ``unit``, to ``output`` in the same unit."""

    def __init__(self, unit: str, output: str):
        self.unit = unit
        self.output = output

    def definitions(self) -> list[VariableDefinition]:
        return [
            VariableDefinition("Energy", self.unit, VariableRole.Input),
            VariableDefinition(self.output, self.unit, VariableRole.Output),
        ]

    def solve(self, t_current: float, t_next: float, inputs: Any) -> dict[str, float]:
        return {self.output: inputs["Energy"]}


def spec(*lags: Lag) -> ModelSpec:
    return ModelSpec(
        components={
            "Counter": ComponentSpec(Counter),
            "Now": ComponentSpec(Reader, {"unit": "J", "output": "Now Energy"}),
            "Before": ComponentSpec(Reader, {"unit": "kJ", "output": "Before Energy"}),
        },
        lags=lags,
        initial_values={"Energy": 0.0},
    )


def values(model: Any, name: str) -> list[float]:
    series = model.timeseries().get_timeseries_by_name(name)
    return list(series.values())


def test_instances_of_one_type_couple_independently():
    model = spec(Lag("Energy", into=["Before"])).bind({}, AXIS).run()

    assert values(model, "Now Energy")[1:] == [1.0, 2.0, 3.0]
    assert values(model, "Before Energy")[1:] == pytest.approx([0.0, 1e-3, 2e-3])
    coupling = json.loads(model.coupling(format="json"))
    assert [c["name"] for c in coupling["components"]] == ["Before", "Counter", "Now"]
    assert [(e["consumer"], e["lagged"]) for e in coupling["edges"]] == [
        ("Before", True),
        ("Now", False),
    ]


def test_a_lag_into_several_labels_lags_each():
    model = spec(Lag("Energy", into=["Now", "Before"])).bind({}, AXIS).run()

    assert values(model, "Now Energy")[1:] == [0.0, 1.0, 2.0]
    assert str(Lag("Energy", into=["Now", "Before"])) == "Energy -> Now, Before"
    assert str(Lag("Energy")) == "Energy -> every consumer"


def test_lag_into_takes_a_sequence_of_labels():
    with pytest.raises(TypeError, match="sequence of labels"):
        Lag("Energy", into="Before")


def test_a_lag_names_a_label_not_a_type():
    with pytest.raises(ValueError, match='no component labelled "Reader"'):
        spec(Lag("Energy", into=["Reader"])).bind({}, AXIS).build()


def test_duplicate_labels_fail_to_build():
    builder = (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_initial_values({"Energy": 0.0})
        .with_component(PythonComponent(Counter()))
        .with_component(PythonComponent(Reader("J", "First")))
        .with_component(PythonComponent(Reader("J", "Second")))
    )
    with pytest.raises(ValueError, match="Two components are labelled 'Reader'"):
        builder.build()


def test_a_python_component_is_labelled_with_its_class_name():
    class Nested(Component):
        energy = Input("Energy", unit="J")
        copy = Output("Copy", unit="J")

        def solve(self, t_current, t_next, inputs):
            return self.Outputs(copy=inputs.energy.get())

    model = (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_initial_values({"Energy": 0.0})
        .with_component(PythonComponent(Counter()))
        .with_component(Nested())
        .with_lag_for("Energy", "Nested")
        .build()
    )
    coupling = json.loads(model.coupling(format="json"))
    assert [c["name"] for c in coupling["components"]] == ["Nested", "Counter"]
