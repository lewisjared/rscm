import numpy as np
import pytest

from rscm._lib.core import TestComponentBuilder as ExampleComponentBuilder
from rscm.core import (
    InterpolationStrategy,
    PythonComponent,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
    VariableDefinition,
)


class ExamplePythonComponent:
    def definitions(self) -> list[VariableDefinition]:
        return []

    def solve(
        self, t_current: float, t_next: float, inputs: dict[str, float]
    ) -> dict[str, float]:
        print(inputs)
        return {"output": inputs.get("input") * 3}


def test_component_definitions():
    component = ExampleComponentBuilder.from_parameters(
        {"conversion_factor": 0.5}
    ).build()

    definitions = component.definitions()
    assert len(definitions) == 2
    assert definitions[0].name == "Emissions|CO2"
    assert definitions[1].name == "Atmospheric Concentrations|CO2"


def test_component_invalid():
    with pytest.raises(ValueError, match="missing field `conversion_factor`"):
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
    component = PythonComponent(py_component)

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
    )

    res = component.solve(0, 1, collection)
    assert res["output"] == 35.0 * 3.0
