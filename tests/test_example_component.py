import numpy as np
import pytest

from rscm._lib.core import TestComponentBuilder as ExampleComponentBuilder
from rscm.core import (
    InterpolationStrategy,
    ModelBuilder,
    PythonComponent,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
    VariableDefinition,
)
from rscm.example_components import LinearConcentration


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


def test_linear_concentration_runs():
    time_axis = TimeAxis.from_values(np.arange(1850.0, 1901.0))
    emissions = Timeseries(
        np.full(len(time_axis), 10.0),
        time_axis,
        "Gt CO2/yr",
        InterpolationStrategy.Previous,
    )
    model = (
        ModelBuilder()
        .with_time_axis(time_axis)
        .with_component(LinearConcentration(decay_rate=0.0))
        .with_prescribed_variable("Emissions|CO2", emissions)
        .with_initial_values({"Atmospheric Concentrations|CO2": 278.0})
        .build()
    )
    model.run()

    concentration = (
        model.timeseries()
        .get_timeseries_by_name("Atmospheric Concentrations|CO2")
        .values()
    )
    # No decay: each year adds half of 10 Gt CO2/yr.
    np.testing.assert_allclose(concentration, 278.0 + 5.0 * np.arange(51))
