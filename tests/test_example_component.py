import numpy as np
import pytest

from rscm._lib.core import TestComponentBuilder as ExampleComponentBuilder
from rscm._lib.core import VariableType
from rscm.core import (
    InterpolationStrategy,
    PythonComponent,
    RequirementDefinition,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
)


class ExamplePythonComponent:
    def definitions(self) -> list[RequirementDefinition]:
        return []

    def solve(
        self, time_current: float, time_next: float, input_state: dict[str, float]
    ) -> dict[str, float]:
        print(input_state)
        return {"output": input_state.get("input") * 3}


def test_component_definitions():
    component = ExampleComponentBuilder.from_parameters({"p": 12}).build()

    definitions = component.definitions()
    assert len(definitions) == 2
    assert definitions[0].name == "Emissions|CO2"
    assert definitions[1].name == "Concentrations|CO2"


def test_component_invalid():
    with pytest.raises(ValueError, match="missing field `p`"):
        ExampleComponentBuilder.from_parameters({})

    with pytest.raises(
        ValueError,
        match="unexpected type: 'NoneType' object cannot be cast as 'Mapping'",
    ):
        # noinspection PyTypeChecker
        ExampleComponentBuilder.from_parameters(None).build()


@pytest.mark.xfail(reason="component definitions are not implemented")
def test_user_derived_create_and_solve():
    py_component = ExamplePythonComponent()
    component = PythonComponent.build(py_component)

    # TODO: resolve later
    assert component.definitions() == []

    collection = TimeseriesCollection()
    collection.add_timeseries(
        "input",
        Timeseries(
            np.asarray([35.0]),
            TimeAxis.from_bounds(np.asarray([0.0, 1.0])),
            "GtC",
            InterpolationStrategy.Previous,
        ),
        variable_type=VariableType.Exogenous,
    )

    res = component.solve(0, 1, collection)
    assert res["output"] == 35.0 * 3.0
