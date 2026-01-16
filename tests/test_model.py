import numpy as np
import numpy.testing as npt

from rscm._lib.core import InterpolationStrategy, Model, Timeseries
from rscm._lib.two_layer import TwoLayerComponentBuilder
from rscm.core import ModelBuilder


def test_model(time_axis):
    component = TwoLayerComponentBuilder.from_parameters(
        dict(
            lambda0=0.0,
            a=0.0,
            efficacy=0.0,
            eta=0.0,
            heat_capacity_deep=0.0,
            heat_capacity_surface=0.0,
        )
    ).build()

    builder = ModelBuilder()
    builder.with_time_axis(time_axis).with_rust_component(component)

    # Doesn't have any ERF data
    # Need pyo3_runtime.PanicException
    # with pytest.raises(Exception):
    #     builder.build()

    erf = Timeseries(
        np.asarray([1.0] * len(time_axis)),
        time_axis,
        "W / m^2",
        InterpolationStrategy.Next,
    )

    model = builder.with_exogenous_variable("Effective Radiative Forcing", erf).build()

    print(model.as_dot())

    model.run()


def test_model_serialisation(time_axis):
    component = TwoLayerComponentBuilder.from_parameters(
        dict(
            lambda0=0.0,
            a=0.0,
            efficacy=0.0,
            eta=0.0,
            heat_capacity_deep=0.0,
            heat_capacity_surface=0.0,
        )
    ).build()

    builder = ModelBuilder()
    builder.with_time_axis(time_axis).with_rust_component(component)
    erf = Timeseries(
        np.asarray([1.0] * len(time_axis)),
        time_axis,
        "W / m^2",
        InterpolationStrategy.Next,
    )

    model = builder.with_exogenous_variable("Effective Radiative Forcing", erf).build()

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
