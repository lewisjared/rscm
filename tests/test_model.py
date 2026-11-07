import json

import numpy as np
import numpy.testing as npt
import pytest

from rscm._lib.core import InterpolationStrategy, Model, Timeseries
from rscm._lib.two_layer import TwoLayerBuilder
from rscm.core import ModelBuilder


def test_model(time_axis):
    component = TwoLayerBuilder.from_parameters(
        dict(
            lambda0=0.0,
            a=0.0,
            deep_ocean_efficacy=0.0,
            ocean_heat_transfer=0.0,
            heat_capacity_deep=0.0,
            heat_capacity_surface=0.0,
        )
    ).build()

    builder = ModelBuilder()
    builder.with_time_axis(time_axis).with_component(component)

    erf = Timeseries(
        np.asarray([1.0] * len(time_axis)),
        time_axis,
        "W / m^2",
        InterpolationStrategy.Next,
    )

    model = (
        builder.with_prescribed_variable("Effective Radiative Forcing", erf)
        .with_initial_values(
            {"Surface Temperature": 0.0, "Deep Ocean Temperature": 0.0}
        )
        .build()
    )

    print(model.as_dot())

    model.run()


def test_missing_prescribed_input(time_axis):
    component = TwoLayerBuilder.from_parameters(
        dict(
            lambda0=0.0,
            a=0.0,
            deep_ocean_efficacy=0.0,
            ocean_heat_transfer=0.0,
            heat_capacity_deep=0.0,
            heat_capacity_surface=0.0,
        )
    ).build()
    builder = (
        ModelBuilder()
        .with_time_axis(time_axis)
        .with_component(component)
        .with_initial_values(
            {"Surface Temperature": 0.0, "Deep Ocean Temperature": 0.0}
        )
    )

    with pytest.raises(
        ValueError, match=r"'Effective Radiative Forcing' \(read by TwoLayer\)"
    ):
        builder.build()

    model = builder.with_missing_prescribed_allowed().build()
    erf = model.timeseries().get_timeseries_by_name("Effective Radiative Forcing")
    assert np.isnan(erf.values()).all()


def test_missing_initial_value_names_the_builder_method(time_axis):
    component = TwoLayerBuilder.from_parameters(
        dict(
            lambda0=0.0,
            a=0.0,
            deep_ocean_efficacy=0.0,
            ocean_heat_transfer=0.0,
            heat_capacity_deep=0.0,
            heat_capacity_surface=0.0,
        )
    ).build()
    builder = (
        ModelBuilder()
        .with_time_axis(time_axis)
        .with_component(component)
        .with_initial_values({"Deep Ocean Temperature": 0.0})
    )

    with pytest.raises(
        ValueError,
        match=r'with_initial_values\(\{"Surface Temperature": value\}\)',
    ):
        builder.build()


def test_model_serialisation(time_axis):
    component = TwoLayerBuilder.from_parameters(
        dict(
            lambda0=0.0,
            a=0.0,
            deep_ocean_efficacy=0.0,
            ocean_heat_transfer=0.0,
            heat_capacity_deep=0.0,
            heat_capacity_surface=0.0,
        )
    ).build()

    builder = ModelBuilder()
    builder.with_time_axis(time_axis).with_component(component)
    erf = Timeseries(
        np.asarray([1.0] * len(time_axis)),
        time_axis,
        "W / m^2",
        InterpolationStrategy.Next,
    )

    model = (
        builder.with_prescribed_variable("Effective Radiative Forcing", erf)
        .with_initial_values(
            {"Surface Temperature": 0.0, "Deep Ocean Temperature": 0.0}
        )
        .build()
    )

    model.step()

    serialised_model = model.to_toml()

    new_model = Model.from_toml(serialised_model)

    assert new_model.as_dot() == model.as_dot()
    assert new_model.current_time() == model.current_time()

    # Run the rest of the model and verify that the results are the same
    model.run()
    new_model.run()

    npt.assert_allclose(
        new_model.timeseries().get_timeseries_by_name("Surface Temperature").values(),
        model.timeseries().get_timeseries_by_name("Surface Temperature").values(),
    )


def test_coupling_formats(time_axis):
    component = TwoLayerBuilder.from_parameters(
        dict(
            lambda0=0.0,
            a=0.0,
            deep_ocean_efficacy=0.0,
            ocean_heat_transfer=0.0,
            heat_capacity_deep=0.0,
            heat_capacity_surface=0.0,
        )
    ).build()
    model = (
        ModelBuilder()
        .with_time_axis(time_axis)
        .with_component(component)
        .with_initial_values(
            {"Surface Temperature": 0.0, "Deep Ocean Temperature": 0.0}
        )
        .with_missing_prescribed_allowed()
        .build()
    )

    coupling = json.loads(model.coupling("json"))
    (two_layer,) = coupling["components"]
    inputs = {v["name"]: v for v in two_layer["inputs"]}
    erf = inputs["Effective Radiative Forcing"]
    assert erf["source"] == "prescribed"
    assert erf["timeseries_type"] == "point"
    assert "[0] TwoLayer" in model.coupling("plain")
    assert "TwoLayer" in model.coupling()
    with pytest.raises(ValueError, match="Unknown format"):
        model.coupling("yaml")
