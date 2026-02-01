import numpy as np

from rscm._lib.core import (
    InterpolationStrategy,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
)
from rscm._lib.two_layer import TwoLayerBuilder


def test_create_component():
    component = TwoLayerBuilder.from_parameters(
        dict(
            lambda0=0.3,
            efficacy=31,
            a=12,
            eta=12,
            heat_capacity_deep=12,
            heat_capacity_surface=1,
        )
    ).build()

    time_axis = TimeAxis.from_bounds(np.asarray([2000.0, 2010.0]))
    collection = TimeseriesCollection()
    collection.add_timeseries(
        "Effective Radiative Forcing",
        Timeseries(
            np.asarray([12.0]),
            time_axis,
            "W/m^2",
            InterpolationStrategy.Previous,
        ),
    )
    # Add state variables required by TwoLayer
    collection.add_timeseries(
        "Surface Temperature",
        Timeseries(
            np.asarray([0.0]),
            time_axis,
            "K",
            InterpolationStrategy.Previous,
        ),
    )
    collection.add_timeseries(
        "Deep Ocean Temperature",
        Timeseries(
            np.asarray([0.0]),
            time_axis,
            "K",
            InterpolationStrategy.Previous,
        ),
    )

    res = component.solve(2000, 2010, collection)
    assert isinstance(res, dict)
