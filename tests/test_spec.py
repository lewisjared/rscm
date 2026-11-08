"""Describing, inspecting and building a model from a spec."""

import numpy as np
import pytest

from rscm.core import GridType, InterpolationStrategy, Timeseries
from rscm.example_components import LinearConcentration
from rscm.spec import ComponentSpec, Lag, ModelSpec, _flatten, diff
from rscm.two_layer import TwoLayerBuilder

TWO_LAYER = {
    "lambda0": 1.0,
    "a": 0.0,
    "deep_ocean_efficacy": 1.0,
    "ocean_heat_transfer": 0.5,
    "heat_capacity_deep": 100.0,
    "heat_capacity_surface": 8.0,
}
ERF = "Effective Radiative Forcing"


@pytest.fixture
def spec():
    return ModelSpec(
        components={"TwoLayer": ComponentSpec(TwoLayerBuilder, TWO_LAYER)},
        initial_values={
            "Surface Air Temperature Change": 0.0,
            "Ocean Temperature|Layer 2": 0.0,
        },
        name="Two layer",
    )


@pytest.fixture
def forcing(time_axis):
    return Timeseries(
        np.ones(len(time_axis)), time_axis, "W/m^2", InterpolationStrategy.Linear
    )


def test_setup_lists_prescribed_inputs(spec, forcing, time_axis):
    unbound = spec.bind({}, time_axis)
    (needed,) = unbound.prescribed_inputs()
    assert (needed.definition.name, needed.consumers, needed.supplied) == (
        ERF,
        ["TwoLayer"],
        False,
    )
    assert f"missing inputs: {ERF}" in str(unbound)
    with pytest.raises(ValueError, match=ERF):
        unbound.build()

    setup = spec.bind({ERF: forcing}, time_axis)
    assert [i.supplied for i in setup.prescribed_inputs()] == [True]
    assert setup.run().finished()


def test_describe(spec, forcing, time_axis):
    assert str(spec.bind({ERF: forcing}, time_axis)) == "\n".join(
        [
            "Two layer",
            "  time axis  1850 to 2000, 30 steps",
            "  components (1, in the order added)",
            "    TwoLayer  6 parameters",
            "  lags (0)",
            "  prescribed data (1)",
            f"    {ERF}  1",
            "  missing inputs: none",
            "  initial values: 2 scalar, 0 gridded",
        ]
    )


def test_python_component_takes_keyword_parameters():
    component = ComponentSpec(LinearConcentration, {"decay_rate": 0.1}).build()
    assert component.decay_rate == 0.1


def test_grid_weights_are_keyed_by_grid():
    spec = ModelSpec(components={}, grid_weights={GridType.FourBox: [0.25] * 4})
    assert spec.grid_weights[GridType.FourBox] == [0.25] * 4


def test_diff(spec, forcing, time_axis):
    a = spec.bind({ERF: forcing}, time_axis)
    b = ModelSpec(
        components={
            "TwoLayer": ComponentSpec(TwoLayerBuilder, {**TWO_LAYER, "lambda0": 1.2})
        },
        lags=(Lag("Surface Air Temperature Change"),),
        name="Two layer, stronger feedback",
    ).bind({}, time_axis)

    assert diff(a, b) == "\n".join(
        [
            "Two layer -> Two layer, stronger feedback",
            "  ~ TwoLayer.lambda0: 1.0 -> 1.2",
            "  ~ initial values: 2 removed",
            "  + lag Surface Air Temperature Change -> every consumer",
            f"  - prescribed {ERF}",
        ]
    )
    assert diff(a, a).endswith("identical")


def test_diff_reports_replaced_builders_and_grid_values(spec, time_axis):
    a = spec.bind({}, time_axis)
    b = ModelSpec(
        components={"TwoLayer": ComponentSpec(LinearConcentration, TWO_LAYER)},
        initial_values=spec.initial_values,
        initial_grid_values={"Forcing": [1.0, 2.0, 3.0, 4.0]},
        name="Two layer",
    ).bind({}, time_axis)

    assert diff(a, b).splitlines()[1:] == [
        "  ~ TwoLayer: TwoLayerBuilder -> LinearConcentration",
        "  ~ initial grid values: 1 added",
    ]


def test_spec_keeps_a_copy_of_its_parameters():
    parameters = {"agents": [{"variable": "A", "efficacy": 1.0}]}
    component = ComponentSpec(TwoLayerBuilder, parameters)
    parameters["agents"][0]["efficacy"] = 2.0
    assert component.parameters["agents"][0]["efficacy"] == 1.0


def test_flatten_keeps_identities_that_share_a_last_segment():
    agents = [
        {"variable": "Emissions|CO2", "scale": 1.0},
        {"variable": "Concentrations|CO2", "scale": 2.0},
        {"variable": "Concentrations|CO2", "scale": 3.0},
    ]
    flat = _flatten({"agents": agents}, "")
    assert {k: v for k, v in flat.items() if k.endswith(".scale")} == {
        "agents[Emissions|CO2].scale": 1.0,
        "agents[Concentrations|CO2#1].scale": 2.0,
        "agents[Concentrations|CO2#2].scale": 3.0,
    }
