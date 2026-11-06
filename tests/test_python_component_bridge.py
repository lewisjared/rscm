"""Tests for how Python components run inside a Rust model."""

import numpy as np
import pytest

from rscm.component import Component, Input, Output, State
from rscm.core import (
    FourBoxTimeseries,
    GridType,
    InterpolationStrategy,
    ModelBuilder,
    PythonComponent,
    RequirementDefinition,
    RequirementType,
    TimeAxis,
    Timeseries,
    VariableSchema,
)

YEARS = np.arange(2000.0, 2005.0)
AXIS = TimeAxis.from_values(YEARS)


class Accumulator(Component):
    emissions = Input("Emissions|CO2", unit="GtC/yr")
    stock = State("Carbon Stock", unit="GtC")

    def __init__(self, rate: float):
        self.rate = rate

    def solve(self, t_current, t_next, inputs):
        stock = inputs.stock.get() + self.rate * inputs.emissions.get()
        return self.Outputs(stock=stock)


class Failing(Component):
    emissions = Input("Emissions|CO2", unit="GtC/yr")
    out = Output("Out", unit="GtC/yr")

    def solve(self, t_current, t_next, inputs):
        raise ValueError("bad physics")  # noqa: TRY003


class ReturnsDict(Component):
    emissions = Input("Emissions|CO2", unit="GtC/yr")
    out = Output("Out", unit="GtC/yr")

    def solve(self, t_current, t_next, inputs):
        return {"Out": 1.0}


class Producer(Component):
    out = Output("Forcing", unit="W/m^2")

    def solve(self, t_current, t_next, inputs):
        return self.Outputs(out=t_current - 2000.0)


class Recorder(Component):
    forcing = Input("Forcing", unit="W/m^2")
    seen = Output("Seen", unit="W/m^2")

    def __init__(self):
        self.ends = []

    def solve(self, t_current, t_next, inputs):
        self.ends.append(inputs.forcing.interval()[1])
        return self.Outputs(seen=0.0)


class MegatonneReader(Component):
    emissions = Input("Emissions|CO2", unit="MtC/yr")
    out = Output("Out", unit="MtC/yr")

    def solve(self, t_current, t_next, inputs):
        return self.Outputs(out=inputs.emissions.get())


class GlobalReader(Component):
    forcing = Input("Regional Forcing", unit="W/m^2")
    out = Output("Global Forcing", unit="W/m^2")

    def solve(self, t_current, t_next, inputs):
        return self.Outputs(out=inputs.forcing.get())


class NextForcingProducer(Component):
    out = Output("Forcing", unit="W/m^2")

    def solve(self, t_current, t_next, inputs):
        return self.Outputs(out=t_next - 2000.0)


class TypedForcingReader(Component):
    forcing = Input("Forcing", unit="mW/m^2")
    seen = Output("Seen", unit="mW/m^2")

    def __init__(self):
        self.seen_values = []

    def solve(self, t_current, t_next, inputs):
        self.seen_values.append(inputs.forcing.get())
        return self.Outputs(seen=0.0)


class DictForcingReader:
    def __init__(self, grid_type=GridType.Scalar):
        self.grid_type = grid_type
        self.seen_values = []

    def definitions(self):
        return [
            RequirementDefinition(
                "Forcing", "mW/m^2", RequirementType.Input, self.grid_type
            ),
            RequirementDefinition("Seen", "mW/m^2", RequirementType.Output),
        ]

    def solve(self, t_current, t_next, inputs):
        self.seen_values.append(inputs["Forcing"])
        return {"Seen": 0.0}


class Slotted(Accumulator):
    __slots__ = ("cache",)


class Unserialisable(Accumulator):
    def __init__(self, rate: float):
        super().__init__(rate)
        self.table = np.ones(3)


def emissions() -> Timeseries:
    return Timeseries(
        np.arange(1.0, 6.0), AXIS, "GtC/yr", InterpolationStrategy.Previous
    )


def build(component: Component):
    return (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_py_component(PythonComponent.build(component))
        .with_exogenous_variable("Emissions|CO2", emissions())
        .with_initial_values({"Carbon Stock": 0.0})
        .build()
    )


def test_solve_error_raises_runtime_error_with_traceback():
    model = build(Failing())

    with pytest.raises(RuntimeError, match=r"Failing failed during") as excinfo:
        model.run()

    message = str(excinfo.value)
    assert "ValueError: bad physics" in message
    assert 'raise ValueError("bad physics")' in message


def test_solve_returning_plain_dict_names_the_fix():
    model = build(ReturnsDict())

    with pytest.raises(RuntimeError, match=r"must return self\.Outputs"):
        model.run()


def test_build_raises_when_definitions_fail():
    class Broken(Accumulator):
        def definitions(self):
            raise KeyError("missing")

    with pytest.raises(KeyError, match="missing"):
        PythonComponent.build(Broken(1.0))


def test_interval_end_reads_upstream_output_from_this_step():
    recorder = Recorder()
    model = (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_py_component(PythonComponent.build(Producer()))
        .with_py_component(PythonComponent.build(recorder))
        .build()
    )
    model.run()

    assert recorder.ends == [0.0, 1.0, 2.0, 3.0]


def test_inputs_are_converted_to_declared_units():
    schema = VariableSchema()
    schema.add_variable("Emissions|CO2", "GtC/yr")
    schema.add_variable("Out", "MtC/yr")
    model = (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_schema(schema)
        .with_py_component(PythonComponent.build(MegatonneReader()))
        .with_exogenous_variable("Emissions|CO2", emissions())
        .build()
    )
    model.run()

    out = model.timeseries().get_timeseries_by_name("Out").values()
    np.testing.assert_allclose(out, [1000.0, 2000.0, 3000.0, 4000.0, np.nan])


def test_scalar_input_aggregates_four_box_variable():
    schema = VariableSchema()
    schema.add_variable("Regional Forcing", "W/m^2", GridType.FourBox)
    schema.add_variable("Global Forcing", "W/m^2")
    regional = np.tile([1.0, 2.0, 3.0, 4.0], (len(YEARS), 1))
    model = (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_schema(schema)
        .with_grid_weights(GridType.FourBox, [0.4, 0.1, 0.4, 0.1])
        .with_py_component(PythonComponent.build(GlobalReader()))
        .with_exogenous_four_box_variable(
            "Regional Forcing",
            FourBoxTimeseries(regional, AXIS, "W/m^2", InterpolationStrategy.Previous),
        )
        .build()
    )
    model.run()

    out = model.timeseries().get_timeseries_by_name("Global Forcing").values()
    np.testing.assert_allclose(out[1:], 2.2)


def test_toml_round_trip_restores_python_component():
    reference = build(Accumulator(rate=2.0))
    reference.run()

    model = build(Accumulator(rate=2.0))
    model.step()
    model.step()
    restored = type(model).from_toml(model.to_toml())
    restored.run()

    np.testing.assert_allclose(
        restored.timeseries().get_timeseries_by_name("Carbon Stock").values(),
        reference.timeseries().get_timeseries_by_name("Carbon Stock").values(),
    )


def test_to_toml_reports_unserialisable_attributes():
    model = build(Unserialisable(rate=2.0))

    with pytest.raises(ValueError, match=r"Override to_json\(\) and from_json\(\)"):
        model.to_toml()


def test_to_toml_rejects_slotted_attributes():
    model = build(Slotted(rate=2.0))

    with pytest.raises(ValueError, match=r"__slots__.*Override to_json\(\)"):
        model.to_toml()


def run_forcing_reader(reader, forcing: Timeseries | None = None):
    schema = VariableSchema()
    schema.add_variable("Forcing", "W/m^2")
    schema.add_variable("Seen", "mW/m^2")
    builder = (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_schema(schema)
        .with_py_component(PythonComponent.build(reader))
    )
    if forcing is None:
        builder = builder.with_py_component(
            PythonComponent.build(NextForcingProducer())
        )
    else:
        builder = builder.with_exogenous_variable("Forcing", forcing)
    builder.build().run()
    return reader.seen_values


@pytest.mark.parametrize("prescribed", [True, False], ids=["prescribed", "upstream"])
def test_dict_component_reads_inputs_as_typed_component_does(prescribed):
    forcing = (
        Timeseries(np.arange(5.0), AXIS, "W/m^2", InterpolationStrategy.Linear)
        if prescribed
        else None
    )

    typed = run_forcing_reader(TypedForcingReader(), forcing)
    untyped = run_forcing_reader(DictForcingReader(), forcing)

    # Points are read at the step's end and converted to the declared unit.
    np.testing.assert_allclose(typed, [1000.0, 2000.0, 3000.0, 4000.0])
    np.testing.assert_allclose(untyped, typed)


def test_dict_component_aggregates_grid_input_to_global():
    schema = VariableSchema()
    schema.add_variable("Forcing", "W/m^2", GridType.FourBox)
    schema.add_variable("Seen", "mW/m^2")
    regional = np.outer(np.arange(1.0, 6.0), [1.0, 2.0, 3.0, 4.0])
    reader = DictForcingReader(GridType.FourBox)
    (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_schema(schema)
        .with_py_component(PythonComponent.build(reader))
        .with_exogenous_four_box_variable(
            "Forcing",
            FourBoxTimeseries(regional, AXIS, "W/m^2", InterpolationStrategy.Linear),
        )
        .build()
        .run()
    )

    np.testing.assert_allclose(reader.seen_values, 2500.0 * np.arange(2.0, 6.0))
