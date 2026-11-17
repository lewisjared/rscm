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

        assert repr(axis) == "TimeAxis(len=3, start=2000.0, end=2060.0)"

    def test_time_axis_immutable(self, time_axis):
        values = time_axis.values()

        assert values.flags["OWNDATA"]
        values[0] = 1200

        assert values[0] == 1200.0
        assert time_axis.values()[0] == 1850.0

    @pytest.mark.parametrize(
        "values",
        [
            [2000.0, 2020.0, 2040.0],
            np.arange(2000, 2041, 20),
            (2000, 2020, 2040),
            np.arange(2000.0, 2081.0, 20)[::2],
        ],
    )
    def test_time_axis_create_from_array_like(self, values):
        axis = TimeAxis.from_values(values)

        npt.assert_allclose(axis.values(), np.asarray(values, dtype=float))

    def test_time_axis_from_2d(self):
        with pytest.raises(
            ValueError, match=r"values must be 1-dimensional, got shape \(2, 2\)"
        ):
            TimeAxis.from_values(np.ones((2, 2)))
        with pytest.raises(ValueError, match="bounds must be 1-dimensional"):
            TimeAxis.from_bounds(np.ones((2, 2)))

    @pytest.mark.parametrize(
        "values, match",
        [
            ([], "at least 2 values, got 0"),
            ([2000.0], "at least 2 values, got 1"),
            (
                [2000.0, 1999.0, 2001.0],
                "strictly increase, got 1999 at index 1 after 2000",
            ),
            (
                [2000.0, 2001.0, 2001.0],
                "strictly increase, got 2001 at index 2 after 2001",
            ),
            ([2000.0, np.nan, 2002.0], "finite, got NaN at index 1"),
            ([2000.0, np.inf], "finite, got inf at index 1"),
        ],
    )
    def test_time_axis_invalid_values(self, values, match):
        with pytest.raises(ValueError, match=match):
            TimeAxis.from_values(np.asarray(values, dtype=float))

    @pytest.mark.parametrize("bounds", [[], [2000.0]])
    def test_time_axis_too_few_bounds(self, bounds):
        with pytest.raises(ValueError, match="at least 2 bounds"):
            TimeAxis.from_bounds(np.asarray(bounds, dtype=float))

    def test_time_axis_at(self, time_axis):
        assert time_axis.at(0) == 1850.0
        assert time_axis.at(-1) == 1995.0
        assert time_axis.at(1) == 1855.0
        assert time_axis.at(10000) is None
        assert time_axis.at(-10000) is None

    def test_time_axis_at_bounds(self, time_axis):
        assert time_axis.at_bounds(0) == (1850.0, 1855.0)
        assert time_axis.at_bounds(-1) == (1995.0, 2000.0)
        assert time_axis.at_bounds(1) == (1855.0, 1860.0)
        assert time_axis.at_bounds(10000) is None

    def test_time_axis_equality(self, time_axis):
        assert time_axis == TimeAxis.from_values(np.arange(1850.0, 2000.0, 5))
        assert time_axis != TimeAxis.from_values(np.arange(1850.0, 2005.0, 5))
        assert time_axis != "not a time axis"


class TestTimeseries:
    def test_create(self, time_axis):
        ts = Timeseries(
            values=time_axis.values(),
            time_axis=time_axis,
            unit="Test",
            interpolation_strategy=InterpolationStrategy.Next,
        )

        npt.assert_allclose(ts.values(), time_axis.values())

        ts.set(0, 42.0)
        assert ts.values()[0] == 42.0

    def test_create_invalid(self, time_axis):
        values = np.arange(0.0, 10.0)
        assert len(values) != len(time_axis)

        with pytest.raises(
            ValueError, match="values has length 10 but time_axis has length 30"
        ):
            Timeseries(
                values=values,
                time_axis=time_axis,
                unit="Test",
                interpolation_strategy=InterpolationStrategy.Next,
            )

    def test_create_from_array_like(self, time_axis):
        values = list(range(len(time_axis)))
        ts = Timeseries(values, time_axis, "K", InterpolationStrategy.Linear)
        npt.assert_allclose(ts.values(), values)

        ts = Timeseries.from_values(np.arange(len(time_axis)), time_axis)
        npt.assert_allclose(ts.values(), values)

    def test_create_from_2d(self, time_axis):
        with pytest.raises(ValueError, match="values must be 1-dimensional"):
            Timeseries(
                np.ones((len(time_axis), 1)),
                time_axis,
                "K",
                InterpolationStrategy.Linear,
            )

    def test_at_negative_index(self, timeseries):
        assert timeseries.at(-1) == 2000.0
        assert timeseries.at(-len(timeseries)) == 1850.0
        assert timeseries.at(-len(timeseries) - 1) is None

    def test_set_negative_index(self, timeseries):
        timeseries.set(-1, 42.0)
        assert timeseries.values()[-1] == 42.0

    @pytest.mark.parametrize("index", [31, 99, -32])
    def test_set_out_of_range(self, timeseries, index):
        with pytest.raises(
            IndexError, match=f"time index {index} is out of range for 31 steps"
        ):
            timeseries.set(index, 1.0)

    def test_at_time_nan(self, timeseries):
        with pytest.raises(ValueError, match="NaN time"):
            timeseries.at_time(float("nan"))

    def test_at_time_linear_single_step(self):
        axis = TimeAxis.from_bounds(np.array([2000.0, 2001.0]))
        ts = Timeseries(np.array([5.0]), axis, "K", InterpolationStrategy.Linear)
        assert ts.at_time(2000.5) == 5.0
        assert ts.at_time(2010.0) == 5.0

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
