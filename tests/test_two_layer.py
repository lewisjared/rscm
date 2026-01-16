import numpy as np

from rscm._lib.core import (
    InterpolationStrategy,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
)
from rscm._lib.two_layer import TwoLayerComponentBuilder


def test_create_component():
    component = TwoLayerComponentBuilder.from_parameters(
        dict(
            lambda0=0.3,
            efficacy=31,
            a=12,
            eta=12,
            heat_capacity_deep=12,
            heat_capacity_surface=1,
        )
    ).build()

    collection = TimeseriesCollection()
    collection.add_timeseries(
        "Effective Radiative Forcing",
        Timeseries(
            np.asarray([12.0]),
            TimeAxis.from_bounds(np.asarray([2000.0, 2010.0])),
            "GtC",
            InterpolationStrategy.Previous,
        ),
    )

    res = component.solve(2000, 2010, collection)
    assert isinstance(res, dict)
