import numpy as np
import pytest

from rscm._lib.core import InterpolationStrategy, TimeAxis, Timeseries


@pytest.fixture()
def time_axis():
    return TimeAxis.from_values(np.arange(1850.0, 2000.0, 5))


@pytest.fixture()
def timeseries():
    return Timeseries(
        values=np.arange(1850.0, 2001.0, 5),
        time_axis=TimeAxis.from_values(np.arange(1850.0, 2001.0, 5)),
        unit="K",
        interpolation_strategy=InterpolationStrategy.Next,
    )
