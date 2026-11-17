"""Builder checks that catch silent defaults, typos and extrapolated data."""

import warnings

import numpy as np
import pytest

from rscm.core import (
    FourBoxTimeseries,
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
    build_models,
)
from rscm.two_layer import TwoLayerBuilder

ERF = "Effective Radiative Forcing"
TEMPERATURE = "Surface Air Temperature Change"
AXIS = TimeAxis.from_values(np.arange(2000.0, 2011.0))
PARAMETERS = {
    "lambda0": 1.0,
    "a": 0.0,
    "deep_ocean_efficacy": 1.0,
    "ocean_heat_transfer": 0.7,
    "heat_capacity_surface": 8.0,
    "heat_capacity_deep": 100.0,
}


def forcing(value: float, start: float = 2000.0, end: float = 2011.0) -> Timeseries:
    axis = TimeAxis.from_values(np.arange(start, end))
    return Timeseries(
        np.full(len(axis), value), axis, "W/m^2", InterpolationStrategy.Previous
    )


def builder() -> ModelBuilder:
    return (
        ModelBuilder()
        .with_time_axis(AXIS)
        .with_component(TwoLayerBuilder.from_parameters(PARAMETERS).build())
        .with_initial_values({TEMPERATURE: 0.0, "Ocean Temperature|Layer 2": 0.0})
    )


def test_no_time_axis_is_an_error():
    component = TwoLayerBuilder.from_parameters(PARAMETERS).build()
    b = ModelBuilder().with_component(component)
    with pytest.raises(ValueError, match="no time axis"):
        b.build()


def test_no_components_is_an_error():
    with pytest.raises(ValueError, match="no components"):
        ModelBuilder().with_time_axis(AXIS).build()


def test_unknown_initial_value_suggests_a_close_match():
    b = builder().with_prescribed_variable(ERF, forcing(1.0))
    b.with_initial_values({"Surface Air Temperatur Change": 0.0})
    with pytest.raises(
        ValueError, match="Did you mean 'Surface Air Temperature Change'"
    ):
        b.build()


def test_unread_prescribed_variable_suggests_a_close_match():
    b = builder().with_prescribed_variable(ERF, forcing(1.0))
    b.with_prescribed_variable("Effective Radiative Forcng", forcing(1.0))
    with pytest.raises(
        ValueError,
        match=f"'Effective Radiative Forcng' is not used.* Did you mean '{ERF}'",
    ):
        b.build()


def test_prescribing_an_output_is_an_error():
    b = builder().with_prescribed_variable(ERF, forcing(1.0))
    b.with_prescribed_variable(TEMPERATURE, forcing(1.0))
    with pytest.raises(ValueError, match="is produced by component 'TwoLayer'"):
        b.build()


def test_collection_extras_are_ignored():
    scenario = TimeseriesCollection()
    scenario.add_timeseries(ERF, forcing(1.0))
    scenario.add_timeseries("Emissions|CO2", forcing(1.0))
    builder().with_prescribed_collection(scenario).build()


def test_later_prescribed_data_replaces_earlier():
    scenario = TimeseriesCollection()
    scenario.add_timeseries(ERF, forcing(1.0))
    b = (
        builder()
        .with_prescribed_collection(scenario)
        .with_prescribed_variable(ERF, forcing(2.0))
        .with_prescribed_variable(ERF, forcing(3.0))
    )
    values = b.build().timeseries()[ERF].values()
    np.testing.assert_array_equal(values, np.full(11, 3.0))


def test_extrapolated_prescribed_data_warns():
    b = builder().with_prescribed_variable(ERF, forcing(1.0, start=2005.0))
    with pytest.warns(
        UserWarning,
        match=f"'{ERF}' covers 2005 to 2011, so it is extrapolated back to 2000",
    ):
        b.build()
    with pytest.warns(UserWarning, match="extrapolated back to 2000"):
        build_models([b])


def test_covering_prescribed_data_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        builder().with_prescribed_variable(ERF, forcing(1.0)).build()


class TestCollection:
    def test_duplicate_name_raises(self):
        c = TimeseriesCollection()
        c.add_timeseries(ERF, forcing(1.0))
        with pytest.raises(ValueError, match="already holds a timeseries named"):
            c.add_timeseries(ERF, forcing(2.0))

    def test_getitem_returns_any_grid(self):
        c = TimeseriesCollection()
        c.add_timeseries(ERF, forcing(1.0))
        boxes = FourBoxTimeseries(
            np.ones((11, 4)), AXIS, "K", InterpolationStrategy.Previous
        )
        c.add_timeseries("Regional", boxes)
        assert isinstance(c[ERF], Timeseries)
        assert isinstance(c["Regional"], FourBoxTimeseries)
        assert ERF in c
        assert len(c) == 2

    def test_getitem_missing_suggests_a_close_match(self):
        c = TimeseriesCollection()
        c.add_timeseries(ERF, forcing(1.0))
        with pytest.raises(KeyError, match=f"Did you mean '{ERF}'"):
            c["Effective Radiative Forcng"]
        assert c.get_timeseries_by_name("missing") is None


class TestTwoLayerParameters:
    def test_lists_every_missing_parameter(self):
        with pytest.raises(
            ValueError,
            match=r"TwoLayer parameters: missing lambda0 \(float\), a \(float\)",
        ):
            TwoLayerBuilder.from_parameters({})

    def test_names_the_field_with_a_wrong_type(self):
        with pytest.raises(ValueError, match="'lambda0' must be float, not str"):
            TwoLayerBuilder.from_parameters({**PARAMETERS, "lambda0": "1.0"})

    @pytest.mark.parametrize(
        ("name", "value", "match"),
        [
            ("heat_capacity_surface", 0.0, "must be positive, got 0"),
            ("heat_capacity_deep", -8.0, "heat_capacity_deep must be positive, got -8"),
            ("lambda0", float("nan"), "lambda0 must be finite, got NaN"),
            ("a", float("inf"), "a must be finite, got inf"),
        ],
    )
    def test_rejects_unphysical_values(self, name, value, match):
        with pytest.raises(ValueError, match=match):
            TwoLayerBuilder.from_parameters({**PARAMETERS, name: value})
