"""Round trips between RSCM timeseries and pandas or xarray."""

from typing import Any

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from rscm import interop
from rscm.core import (
    FourBoxGrid,
    FourBoxTimeseries,
    GridType,
    HemisphericGrid,
    HemisphericTimeseries,
    InterpolationStrategy,
    ModelBuilder,
    TimeAxis,
    Timeseries,
    TimeseriesCollection,
)
from rscm.fair import from_fair, setup_model
from rscm.magicc import emissions_driven, full
from rscm.magicc.prescribed import area_fractions
from rscm.names import (
    ERF as EFFECTIVE_RADIATIVE_FORCING,
)
from rscm.names import (
    SURFACE_AIR_TEMPERATURE_CHANGE,
)
from rscm.two_layer import TwoLayerBuilder

AXIS = TimeAxis.from_bounds(np.array([2000.0, 2001.0, 2003.0, 2010.0]))
WEIGHTS = (0.3, 0.2, 0.35, 0.15)
AnySeries = Timeseries | FourBoxTimeseries | HemisphericTimeseries


def _collection() -> TimeseriesCollection:
    c = TimeseriesCollection()
    c.add_timeseries(
        "Emissions|CO2",
        Timeseries(
            [1.0, 2.0, np.nan], AXIS, "Mt CO2/yr", InterpolationStrategy.Previous
        ),
    )
    c.add_timeseries(
        SURFACE_AIR_TEMPERATURE_CHANGE,
        FourBoxTimeseries(
            np.arange(12.0).reshape(3, 4),
            AXIS,
            "K",
            InterpolationStrategy.Linear,
            grid=FourBoxGrid.with_weights(WEIGHTS),
        ),
    )
    c.add_timeseries(
        "Ocean Heat Uptake",
        HemisphericTimeseries(
            np.arange(6.0).reshape(3, 2),
            AXIS,
            "W/m^2",
            InterpolationStrategy.Next,
            grid=HemisphericGrid.with_weights((0.6, 0.4)),
        ),
    )
    return c


def assert_same(actual: AnySeries, expected: AnySeries) -> None:
    assert type(actual) is type(expected)
    npt.assert_array_equal(actual.values(), expected.values())
    assert actual.unit == expected.unit
    assert actual.time_axis == expected.time_axis
    assert actual.interpolation_strategy == expected.interpolation_strategy
    if not isinstance(expected, Timeseries):
        assert actual.grid.weights() == expected.grid.weights()  # type: ignore[union-attr]


def assert_same_collection(
    actual: TimeseriesCollection, expected: TimeseriesCollection
) -> None:
    assert sorted(actual) == sorted(expected)
    for name in expected:
        assert_same(actual[name], expected[name])


class TestDataFrame:
    def test_layout(self):
        df = _collection().to_dataframe()

        assert df.index.names == ["variable", "unit", "region"]
        npt.assert_array_equal(df.columns, AXIS.values())
        assert df.loc[("Emissions|CO2", "Mt CO2/yr", "World")].tolist()[:2] == [
            1.0,
            2.0,
        ]
        regions = df.loc[SURFACE_AIR_TEMPERATURE_CHANGE].index.get_level_values(
            "region"
        )
        assert list(regions) == list(interop.REGIONS[GridType.FourBox])
        assert list(df.loc["Ocean Heat Uptake"].index.get_level_values("region")) == [
            "World|Northern Hemisphere",
            "World|Southern Hemisphere",
        ]

    def test_round_trip(self):
        c = _collection()
        assert_same_collection(interop.from_dataframe(interop.to_dataframe(c)), c)

    def test_round_trip_through_csv_needs_the_time_axis_and_strategy(self, tmp_path):
        c = _collection()
        path = tmp_path / "out.csv"
        c.to_dataframe().to_csv(path)
        df = pd.read_csv(path, index_col=[0, 1, 2])

        back = interop.from_dataframe(
            df,
            time_axis=AXIS,
            interpolation_strategy={
                "Emissions|CO2": InterpolationStrategy.Previous,
                SURFACE_AIR_TEMPERATURE_CHANGE: InterpolationStrategy.Linear,
                "Ocean Heat Uptake": InterpolationStrategy.Next,
            },
            grid_weights={
                GridType.FourBox: WEIGHTS,
                GridType.Hemispheric: (0.6, 0.4),
            },
        )
        assert_same_collection(back, c)

    def test_global_mean_adds_a_world_row_and_round_trips(self):
        c = _collection()
        df = c.to_dataframe(global_mean=True)

        boxes = c[SURFACE_AIR_TEMPERATURE_CHANGE].values()
        npt.assert_allclose(
            df.loc[(SURFACE_AIR_TEMPERATURE_CHANGE, "K", "World")],
            boxes @ np.array(WEIGHTS),
        )
        npt.assert_allclose(
            df.loc[("Ocean Heat Uptake", "W/m^2", "World")],
            c["Ocean Heat Uptake"].values() @ np.array([0.6, 0.4]),
        )
        assert_same_collection(interop.from_dataframe(df), c)

    def test_a_frame_without_attrs_defaults_to_linear_and_its_columns(self):
        df = pd.DataFrame(
            [[1.0, 2.0, 4.0]],
            columns=[2000, 2010, 2020],
            index=pd.MultiIndex.from_tuples(
                [("Emissions|CH4", "Mt CH4/yr", "World")],
                names=["variable", "unit", "region"],
            ),
        )
        ts = interop.from_dataframe(df)["Emissions|CH4"]

        assert ts.interpolation_strategy == InterpolationStrategy.Linear
        npt.assert_array_equal(ts.time_axis.bounds(), [2000, 2010, 2020, 2030])
        assert ts.unit == "Mt CH4/yr"

    def test_index_levels_may_be_columns_in_any_order(self):
        df = _collection().to_dataframe().reset_index()
        df = df[["region", *df.columns.drop("region")]]
        assert_same_collection(interop.from_dataframe(df), _collection())

    def test_rejects_an_unknown_region(self):
        df = _collection().to_dataframe().rename(index={"World": "Mars"})
        with pytest.raises(ValueError, match="Mars"):
            interop.from_dataframe(df)

    def test_rejects_two_units_for_one_variable(self):
        df = interop.to_dataframe(
            {"T": Timeseries.from_values([1.0, 2.0, 3.0], AXIS, unit="K")}
        )
        other = df.rename(index={"K": "mK", "World": "World|Northern Hemisphere"})
        with pytest.raises(ValueError, match="more than one unit"):
            interop.from_dataframe(pd.concat([df, other]))

    def test_rejects_an_extra_level_with_several_values(self):
        df = pd.concat(
            {"a": _collection().to_dataframe(), "b": _collection().to_dataframe()},
            names=["scenario"],
        )
        with pytest.raises(ValueError, match="scenario"):
            interop.from_dataframe(df)

    def test_a_mapping_of_series_converts(self):
        ts = Timeseries.from_values([1.0, 2.0, 3.0], AXIS, unit="K")
        df = interop.to_dataframe({"T": ts})
        assert df.index.tolist() == [("T", "K", "World")]


class TestXarray:
    xr = pytest.importorskip("xarray")

    def test_layout(self):
        ds = _collection().to_xarray()

        assert set(ds.dims) == {"time", "region", "bound"}
        assert ds[SURFACE_AIR_TEMPERATURE_CHANGE].attrs["unit"] == "K"
        assert ds[SURFACE_AIR_TEMPERATURE_CHANGE].attrs["grid"] == "four_box"
        npt.assert_array_equal(ds["time"], AXIS.values())
        npt.assert_array_equal(ds["time_bounds"][:, 1], AXIS.bounds()[1:])
        npt.assert_array_equal(
            ds["Emissions|CO2"].sel(region="World").values[:2], [1.0, 2.0]
        )

    def test_round_trip(self):
        c = _collection()
        assert_same_collection(interop.from_xarray(interop.to_xarray(c)), c)
        assert_same_collection(
            TimeseriesCollection.from_xarray(c.to_xarray(global_mean=True)), c
        )

    def test_matches_the_dataframe(self):
        c = _collection()
        df = c.to_dataframe(global_mean=True)
        ds = c.to_xarray(global_mean=True)
        for (variable, _, region), row in df.iterrows():
            npt.assert_array_equal(ds[variable].sel(region=region), row)


class TestGridSeries:
    def test_global_mean_uses_the_series_weights(self):
        ts = _collection()[SURFACE_AIR_TEMPERATURE_CHANGE]
        mean = ts.global_mean()

        assert isinstance(mean, Timeseries)
        npt.assert_allclose(mean.values(), ts.values() @ np.array(WEIGHTS))
        assert mean.unit == "K"
        assert mean.time_axis == AXIS

    def test_interpolate_into(self):
        ts = _collection()[SURFACE_AIR_TEMPERATURE_CHANGE]
        axis = TimeAxis.from_values(np.array([2000.0, 2002.0]))
        npt.assert_allclose(ts.interpolate_into(axis).values()[1], [6.0, 7, 8, 9])

    def test_from_values_takes_a_unit(self):
        assert Timeseries.from_values([1.0, 2.0, 3.0], AXIS, "K").unit == "K"


class TestCollection:
    def test_iterates_over_every_name_and_grid(self):
        c = _collection()
        assert list(c) == c.names()
        assert dict(zip(c, (c[n] for n in c))).keys() == set(c.names())

    def test_typed_getter_raises_for_another_grid(self):
        c = _collection()
        with pytest.raises(TypeError, match="four_box"):
            c.get_timeseries_by_name(SURFACE_AIR_TEMPERATURE_CHANGE)
        with pytest.raises(TypeError, match="scalar"):
            c.get_four_box_timeseries_by_name("Emissions|CO2")
        assert c.get_timeseries_by_name("missing") is None


class TestInputs:
    def test_to_inputs_interpolates_onto_the_time_axis(self):
        axis = TimeAxis.from_values(np.arange(2000.0, 2006.0))
        inputs = interop.to_inputs(_collection().to_dataframe(), axis)

        values, unit = inputs["Emissions|CO2"]
        assert unit == "Mt CO2/yr"
        npt.assert_array_equal(values, [1.0, 2.0, 2.0, np.nan, np.nan, np.nan])
        assert inputs[SURFACE_AIR_TEMPERATURE_CHANGE][0].shape == (6, 4)

    def test_from_inputs_round_trips(self):
        axis = TimeAxis.from_values(np.arange(2000.0, 2003.0))
        inputs = {
            "A": (np.array([1.0, 2.0, 3.0]), "K"),
            "B": (np.ones((3, 4)), "W/m^2"),
            "C": (np.ones((3, 2)), "W/m^2"),
        }
        c = interop.from_inputs(inputs, axis)
        assert isinstance(c["B"], FourBoxTimeseries)
        assert isinstance(c["C"], HemisphericTimeseries)
        back = interop.to_inputs(c, axis)
        for name, (values, unit) in inputs.items():
            npt.assert_array_equal(back[name][0], values)
            assert back[name][1] == unit


def _two_layer_model(prescribed: TimeseriesCollection) -> Any:
    component = TwoLayerBuilder.from_parameters(
        dict(
            lambda0=1.2,
            deep_ocean_efficacy=1.3,
            a=0.0,
            ocean_heat_transfer=0.7,
            heat_capacity_deep=100.0,
            heat_capacity_surface=8.0,
        )
    ).build()
    return (
        ModelBuilder()
        .with_time_axis(TimeAxis.from_values(np.arange(1850.0, 1901.0)))
        .with_component(component)
        .with_prescribed_collection(prescribed)
        .with_initial_values(
            {SURFACE_AIR_TEMPERATURE_CHANGE: 0.0, "Ocean Temperature|Layer 2": 0.0}
        )
        .build()
    )


def test_two_layer_from_a_scenario_dataframe():
    scenario = pd.DataFrame(
        [[0.0, 1.0, 3.0]],
        columns=[1850, 1870, 1900],
        index=pd.MultiIndex.from_tuples(
            [(EFFECTIVE_RADIATIVE_FORCING, "W/m^2", "World")],
            names=["variable", "unit", "region"],
        ),
    )
    model = _two_layer_model(interop.from_dataframe(scenario))
    model.run()
    out = model.timeseries()
    df = out.to_dataframe()

    assert_same_collection(interop.from_dataframe(df), out)
    erf = df.loc[(EFFECTIVE_RADIATIVE_FORCING, "W/m^2", "World")]
    assert erf[1860] == pytest.approx(0.5)
    temperature = df.loc[(SURFACE_AIR_TEMPERATURE_CHANGE, "K", "World")]
    assert temperature.iloc[-1] > 0


@pytest.fixture(scope="module")
def magicc_run():
    years = np.arange(1750.0, 1781.0)
    time_axis = TimeAxis.from_values(years)
    ramp = np.linspace(0.1, 1.0, len(years))
    emissions = {
        f"{species}|{sector}": ramp
        for species in full.SECTORAL_EMISSIONS
        for sector in ("Fossil", "Biosphere")
    }
    emissions |= {s: np.zeros(len(years)) for s in full.halocarbon_species()}
    config, inputs = full.from_magicc7(time_axis, emissions)
    model = emissions_driven.setup_model(config, inputs, time_axis).run()
    return config, inputs, time_axis, model


def test_magicc_run_round_trips(magicc_run):
    *_, model = magicc_run
    out = model.timeseries()
    df = out.to_dataframe(global_mean=True)

    assert set(df.index.get_level_values("variable")) == set(out.names())
    assert_same_collection(interop.from_dataframe(df), out)
    assert_same_collection(interop.from_xarray(out.to_xarray()), out)
    boxes = out[SURFACE_AIR_TEMPERATURE_CHANGE]
    npt.assert_array_equal(boxes.grid.weights(), area_fractions())
    npt.assert_allclose(
        df.loc[(SURFACE_AIR_TEMPERATURE_CHANGE, "K", "World")],
        boxes.values() @ area_fractions(),
        rtol=1e-15,
    )
    for name in out:
        ts = out[name]
        if isinstance(ts, FourBoxTimeseries):
            npt.assert_array_equal(ts.grid.weights(), area_fractions())


def test_magicc_inputs_from_a_dataframe(magicc_run):
    config, inputs, time_axis, model = magicc_run
    df = interop.from_inputs(inputs, time_axis).to_dataframe()
    rerun = emissions_driven.setup_model(
        config, interop.to_inputs(df, time_axis), time_axis
    ).run()
    pd.testing.assert_frame_equal(
        rerun.timeseries().to_dataframe(), model.timeseries().to_dataframe()
    )


def test_fair_run_round_trips():
    pytest.importorskip("fair")
    from test_fair_convert import _fair  # noqa: PLC0415

    config, inputs, time_axis = from_fair(_fair())
    out = setup_model(config, inputs, time_axis).run().timeseries()
    assert_same_collection(interop.from_dataframe(out.to_dataframe()), out)
    assert_same_collection(interop.from_xarray(out.to_xarray()), out)

    df = interop.from_inputs(inputs, time_axis).to_dataframe()
    rerun = setup_model(config, interop.to_inputs(df, time_axis), time_axis).run()
    pd.testing.assert_frame_equal(rerun.timeseries().to_dataframe(), out.to_dataframe())
