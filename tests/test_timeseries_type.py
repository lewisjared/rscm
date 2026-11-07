"""Tests for point and average timeseries types."""

import numpy as np
import pytest

from rscm.component import Component, Input, Output, State
from rscm.core import (
    ModelBuilder,
    TimeAxis,
    TimeseriesType,
    VariableDefinition,
    VariableRole,
)

AXIS = TimeAxis.from_values(np.arange(2000.0, 2005.0))


class Chemistry(Component):
    emissions = Input("Emissions|CH4", unit="Mt CH4 / yr")
    lifetime = Output("Lifetime|CH4", unit="yr", timeseries_type="average")
    concentration = Output("Atmospheric Concentrations|CH4", unit="ppb")

    def solve(self, t_current, t_next, inputs):
        return self.Outputs(lifetime=9.0, concentration=1800.0)


class LifetimeReader(Component):
    lifetime = Input("Lifetime|CH4", unit="yr", timeseries_type="point")
    out = Output("Out", unit="1")

    def solve(self, t_current, t_next, inputs):
        return self.Outputs(out=0.0)


class FluxState(Component):
    flux = State("Carbon Flux", unit="GtC / yr")

    def solve(self, t_current, t_next, inputs):
        return self.Outputs(flux=0.0)


def builder(*components: Component) -> ModelBuilder:
    builder = ModelBuilder().with_time_axis(AXIS).with_missing_prescribed_allowed()
    for component in components:
        builder.with_component(component)
    return builder


def test_declaration_reaches_the_definition():
    assert Chemistry.lifetime.to_definition().timeseries_type == (
        TimeseriesType.Average
    )
    assert Chemistry.concentration.to_definition().timeseries_type is None
    definition = VariableDefinition(
        "x", "yr", VariableRole.Input, timeseries_type=TimeseriesType.Average
    )
    assert definition.timeseries_type == TimeseriesType.Average


def test_model_resolves_declared_and_default_types():
    model = builder(Chemistry()).build()

    assert model.timeseries_type("Lifetime|CH4") == TimeseriesType.Average
    assert model.timeseries_type("Emissions|CH4") == TimeseriesType.Average
    assert model.timeseries_type("Atmospheric Concentrations|CH4") == (
        TimeseriesType.Point
    )
    assert model.timeseries_type("Missing") is None


def test_consumer_declaring_another_type_fails():
    with pytest.raises(ValueError, match="Timeseries type mismatch"):
        builder(Chemistry(), LifetimeReader()).build()


def test_prescribed_type_can_be_declared():
    model = (
        builder(LifetimeReader())
        .with_timeseries_type("Lifetime|CH4", TimeseriesType.Point)
        .build()
    )
    assert model.timeseries_type("Lifetime|CH4") == TimeseriesType.Point

    with pytest.raises(ValueError, match="Timeseries type mismatch"):
        (
            builder(LifetimeReader())
            .with_timeseries_type("Lifetime|CH4", TimeseriesType.Average)
            .build()
        )


def test_average_state_fails_unless_the_check_is_off():
    with pytest.raises(ValueError, match="states must be points"):
        builder(FluxState()).with_initial_values({"Carbon Flux": 0.0}).build()

    model = (
        builder(FluxState())
        .with_initial_values({"Carbon Flux": 0.0})
        .with_state_type_check(False)
        .build()
    )
    assert model.timeseries_type("Carbon Flux") == TimeseriesType.Average


def test_unknown_literal_is_rejected():
    with pytest.raises(ValueError, match="Unknown timeseries type"):
        Input("x", timeseries_type="mean").to_definition()  # type: ignore[arg-type]
