import numpy as np
import numpy.testing as npt
import pytest

from rscm.core import InterpolationStrategy, TimeAxis, Timeseries


class TestTimeAxis:
    def test_time_axis_create(self):
        axis = TimeAxis.from_values(np.asarray([2000.0, 2020.0, 2040.0]))

        npt.assert_allclose(axis.values(), [2000.0, 2020.0, 2040.0])
        npt.assert_allclose(axis.bounds(), [2000.0, 2020.0, 2040.0, 2060.0])
        assert len(axis) == 3

        # This is in rust debug format, but ok for now
        exp = "TimeAxis { bounds: [2000.0, 2020.0, 2040.0, 2060.0], shape=[4], strides=[1], layout=CFcf (0xf), const ndim=1 }"  # noqa: E501
        assert repr(axis) == exp

    def test_time_axis_immutable(self, time_axis):
        values = time_axis.values()

        assert values.flags["OWNDATA"]
        values[0] = 1200

        assert values[0] == 1200.0
        assert time_axis.values()[0] == 1850.0

    def test_time_axis_create_from_list(self):
        match = "'list' object cannot be cast as 'ndarray'"
        with pytest.raises(TypeError, match=match):
            # noinspection PyTypeChecker
            TimeAxis.from_values([2000.0, 2020.0, 2040.0])

    def test_time_axis_at(self, time_axis):
        assert time_axis.at(0) == 1850.0
        with pytest.raises(OverflowError):
            assert time_axis.at(-1) is None
        assert time_axis.at(1) == 1855.0
        assert time_axis.at(10000) is None

    def test_time_axis_at_bounds(self, time_axis):
        assert time_axis.at_bounds(0) == (1850.0, 1855.0)
        with pytest.raises(OverflowError):
            assert time_axis.at_bounds(-1) is None
        assert time_axis.at_bounds(1) == (1855.0, 1860.0)
        assert time_axis.at_bounds(10000) is None


class TestTimeseries:
    def test_create(self, time_axis):
        ts = Timeseries(
            values=time_axis.values(),
            time_axis=time_axis,
            units="Test",
            interpolation_strategy=InterpolationStrategy.Next,
        )

        npt.assert_allclose(ts.values(), time_axis.values())

        ts.set(0, 42.0)
        assert ts.values()[0] == 42.0

    def test_create_invalid(self, time_axis):
        values = np.arange(0.0, 10.0)
        assert len(values) != len(time_axis)

        with pytest.raises(ValueError):
            Timeseries(
                values=values,
                time_axis=time_axis,
                units="Test",
                interpolation_strategy=InterpolationStrategy.Next,
            )

    def test_at_time_previous(self, timeseries):
        timeseries.with_interpolation_strategy(InterpolationStrategy.Previous)
        assert timeseries.at_time(1849) == 1850.0
        assert timeseries.at_time(1850) == 1850.0
        assert timeseries.at_time(1850.5) == 1850.0
        assert timeseries.at_time(1855) == 1855.0
        assert timeseries.at_time(2100.0) == 2000.0

    def test_at_time_next(self, timeseries):
        timeseries.with_interpolation_strategy(InterpolationStrategy.Next)
        assert timeseries.at_time(1849) == 1850.0
        assert timeseries.at_time(1850) == 1850.0
        assert timeseries.at_time(1850.5) == 1855.0
        assert timeseries.at_time(1855) == 1855.0
        assert timeseries.at_time(2100.0) == 2000.0

    def test_at_time_linear(self, timeseries):
        timeseries.with_interpolation_strategy(InterpolationStrategy.Linear)
        assert timeseries.at_time(1849) == 1849.0
        assert timeseries.at_time(1850) == 1850.0
        assert timeseries.at_time(1850.5) == 1850.5
        assert timeseries.at_time(2100.0) == 2100.0
