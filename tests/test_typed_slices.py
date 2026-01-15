"""Tests for typed output slices (FourBoxSlice, HemisphericSlice)."""

import math

import numpy as np
import pytest

from rscm.core import (
    FourBoxSlice,
    FourBoxTimeseriesWindow,
    GridType,
    HemisphericSlice,
    HemisphericTimeseriesWindow,
    RequirementDefinition,
    RequirementType,
    TimeseriesWindow,
)


class TestFourBoxSlice:
    """Tests for FourBoxSlice type."""

    def test_constructor_with_kwargs(self):
        """Test construction with keyword arguments."""
        s = FourBoxSlice(
            northern_ocean=15.0,
            northern_land=14.0,
            southern_ocean=10.0,
            southern_land=9.0,
        )
        assert s.northern_ocean == 15.0
        assert s.northern_land == 14.0
        assert s.southern_ocean == 10.0
        assert s.southern_land == 9.0

    def test_constructor_partial_kwargs(self):
        """Test construction with partial kwargs (defaults to NaN)."""
        s = FourBoxSlice(northern_ocean=15.0)
        assert s.northern_ocean == 15.0
        assert math.isnan(s.northern_land)
        assert math.isnan(s.southern_ocean)
        assert math.isnan(s.southern_land)

    def test_default_constructor(self):
        """Test default construction (all NaN)."""
        s = FourBoxSlice()
        assert math.isnan(s.northern_ocean)
        assert math.isnan(s.northern_land)
        assert math.isnan(s.southern_ocean)
        assert math.isnan(s.southern_land)

    def test_uniform(self):
        """Test uniform constructor."""
        s = FourBoxSlice.uniform(15.0)
        assert s.northern_ocean == 15.0
        assert s.northern_land == 15.0
        assert s.southern_ocean == 15.0
        assert s.southern_land == 15.0

    def test_from_array(self):
        """Test from_array constructor."""
        s = FourBoxSlice.from_array([1.0, 2.0, 3.0, 4.0])
        assert s.northern_ocean == 1.0
        assert s.northern_land == 2.0
        assert s.southern_ocean == 3.0
        assert s.southern_land == 4.0

    def test_setters(self):
        """Test property setters."""
        s = FourBoxSlice()
        s.northern_ocean = 15.0
        s.northern_land = 14.0
        s.southern_ocean = 10.0
        s.southern_land = 9.0
        assert s.northern_ocean == 15.0
        assert s.northern_land == 14.0
        assert s.southern_ocean == 10.0
        assert s.southern_land == 9.0

    def test_get_by_index(self):
        """Test get method by region index."""
        s = FourBoxSlice.from_array([15.0, 14.0, 10.0, 9.0])
        assert s.get(0) == 15.0  # northern_ocean
        assert s.get(1) == 14.0  # northern_land
        assert s.get(2) == 10.0  # southern_ocean
        assert s.get(3) == 9.0  # southern_land

    def test_get_invalid_index(self):
        """Test get with invalid index raises error."""
        s = FourBoxSlice()
        with pytest.raises(ValueError, match="Invalid region index"):
            s.get(4)
        with pytest.raises(ValueError, match="Invalid region index"):
            s.get(10)

    def test_set_by_index(self):
        """Test set method by region index."""
        s = FourBoxSlice()
        s.set(0, 15.0)
        s.set(1, 14.0)
        s.set(2, 10.0)
        s.set(3, 9.0)
        assert s.northern_ocean == 15.0
        assert s.northern_land == 14.0
        assert s.southern_ocean == 10.0
        assert s.southern_land == 9.0

    def test_set_invalid_index(self):
        """Test set with invalid index raises error."""
        s = FourBoxSlice()
        with pytest.raises(ValueError, match="Invalid region index"):
            s.set(4, 0.0)

    def test_indexing(self):
        """Test __getitem__ and __setitem__."""
        s = FourBoxSlice.from_array([1.0, 2.0, 3.0, 4.0])
        assert s[0] == 1.0
        assert s[1] == 2.0
        assert s[2] == 3.0
        assert s[3] == 4.0

        s[0] = 10.0
        assert s[0] == 10.0

    def test_len(self):
        """Test __len__."""
        s = FourBoxSlice()
        assert len(s) == 4

    def test_to_array(self):
        """Test conversion to numpy array."""
        s = FourBoxSlice.from_array([1.0, 2.0, 3.0, 4.0])
        arr = s.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4,)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0, 4.0])

    def test_to_list(self):
        """Test conversion to list."""
        s = FourBoxSlice.from_array([1.0, 2.0, 3.0, 4.0])
        lst = s.to_list()
        assert lst == [1.0, 2.0, 3.0, 4.0]

    def test_to_dict(self):
        """Test conversion to dict."""
        s = FourBoxSlice.from_array([1.0, 2.0, 3.0, 4.0])
        d = s.to_dict()
        assert d["northern_ocean"] == 1.0
        assert d["northern_land"] == 2.0
        assert d["southern_ocean"] == 3.0
        assert d["southern_land"] == 4.0

    def test_repr(self):
        """Test string representation."""
        s = FourBoxSlice.from_array([15.0, 14.0, 10.0, 9.0])
        r = repr(s)
        assert "FourBoxSlice" in r
        assert "15" in r
        assert "14" in r
        assert "10" in r
        assert "9" in r


class TestHemisphericSlice:
    """Tests for HemisphericSlice type."""

    def test_constructor_with_kwargs(self):
        """Test construction with keyword arguments."""
        s = HemisphericSlice(northern=15.0, southern=10.0)
        assert s.northern == 15.0
        assert s.southern == 10.0

    def test_constructor_partial_kwargs(self):
        """Test construction with partial kwargs (defaults to NaN)."""
        s = HemisphericSlice(northern=15.0)
        assert s.northern == 15.0
        assert math.isnan(s.southern)

    def test_default_constructor(self):
        """Test default construction (all NaN)."""
        s = HemisphericSlice()
        assert math.isnan(s.northern)
        assert math.isnan(s.southern)

    def test_uniform(self):
        """Test uniform constructor."""
        s = HemisphericSlice.uniform(15.0)
        assert s.northern == 15.0
        assert s.southern == 15.0

    def test_from_array(self):
        """Test from_array constructor."""
        s = HemisphericSlice.from_array([15.0, 10.0])
        assert s.northern == 15.0
        assert s.southern == 10.0

    def test_setters(self):
        """Test property setters."""
        s = HemisphericSlice()
        s.northern = 15.0
        s.southern = 10.0
        assert s.northern == 15.0
        assert s.southern == 10.0

    def test_get_by_index(self):
        """Test get method by region index."""
        s = HemisphericSlice.from_array([15.0, 10.0])
        assert s.get(0) == 15.0  # northern
        assert s.get(1) == 10.0  # southern

    def test_get_invalid_index(self):
        """Test get with invalid index raises error."""
        s = HemisphericSlice()
        with pytest.raises(ValueError, match="Invalid region index"):
            s.get(2)

    def test_set_by_index(self):
        """Test set method by region index."""
        s = HemisphericSlice()
        s.set(0, 15.0)
        s.set(1, 10.0)
        assert s.northern == 15.0
        assert s.southern == 10.0

    def test_indexing(self):
        """Test __getitem__ and __setitem__."""
        s = HemisphericSlice.from_array([15.0, 10.0])
        assert s[0] == 15.0
        assert s[1] == 10.0

        s[0] = 20.0
        assert s[0] == 20.0

    def test_len(self):
        """Test __len__."""
        s = HemisphericSlice()
        assert len(s) == 2

    def test_to_array(self):
        """Test conversion to numpy array."""
        s = HemisphericSlice.from_array([15.0, 10.0])
        arr = s.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2,)
        np.testing.assert_array_equal(arr, [15.0, 10.0])

    def test_to_list(self):
        """Test conversion to list."""
        s = HemisphericSlice.from_array([15.0, 10.0])
        lst = s.to_list()
        assert lst == [15.0, 10.0]

    def test_to_dict(self):
        """Test conversion to dict."""
        s = HemisphericSlice.from_array([15.0, 10.0])
        d = s.to_dict()
        assert d["northern"] == 15.0
        assert d["southern"] == 10.0

    def test_repr(self):
        """Test string representation."""
        s = HemisphericSlice.from_array([15.0, 10.0])
        r = repr(s)
        assert "HemisphericSlice" in r
        assert "15" in r
        assert "10" in r


class TestGridType:
    """Tests for GridType enum."""

    def test_grid_type_exists(self):
        """Test GridType enum values exist."""
        assert hasattr(GridType, "Scalar")
        assert hasattr(GridType, "FourBox")
        assert hasattr(GridType, "Hemispheric")


class TestTimeseriesWindow:
    """Tests for TimeseriesWindow type."""

    def test_constructor(self):
        """Test basic construction."""
        w = TimeseriesWindow([1.0, 2.0, 3.0], 1)
        assert w.current_index == 1
        assert len(w) == 3

    def test_current(self):
        """Test current property."""
        w = TimeseriesWindow([1.0, 2.0, 3.0], 2)
        assert w.current == 3.0

    def test_previous(self):
        """Test previous property."""
        w = TimeseriesWindow([1.0, 2.0, 3.0], 2)
        assert w.previous == 2.0

    def test_previous_at_start_raises(self):
        """Test previous at index 0 raises."""
        w = TimeseriesWindow([1.0, 2.0, 3.0], 0)
        with pytest.raises(ValueError, match="No previous value"):
            _ = w.previous

    def test_at_offset(self):
        """Test at_offset method."""
        w = TimeseriesWindow([1.0, 2.0, 3.0, 4.0, 5.0], 3)
        assert w.at_offset(0) == 4.0
        assert w.at_offset(-1) == 3.0
        assert w.at_offset(-2) == 2.0
        assert w.at_offset(1) == 5.0

    def test_at_offset_out_of_bounds(self):
        """Test at_offset out of bounds raises."""
        w = TimeseriesWindow([1.0, 2.0, 3.0], 1)
        with pytest.raises(ValueError, match="out of bounds"):
            w.at_offset(-5)
        with pytest.raises(ValueError, match="out of bounds"):
            w.at_offset(5)

    def test_last_n(self):
        """Test last_n method."""
        w = TimeseriesWindow([1.0, 2.0, 3.0, 4.0, 5.0], 3)
        arr = w.last_n(3)
        np.testing.assert_array_equal(arr, [2.0, 3.0, 4.0])

    def test_last_n_more_than_available(self):
        """Test last_n when asking for more than available."""
        w = TimeseriesWindow([1.0, 2.0, 3.0], 1)
        arr = w.last_n(5)  # Only 2 values available (indices 0,1)
        np.testing.assert_array_equal(arr, [1.0, 2.0])

    def test_to_array(self):
        """Test to_array method."""
        w = TimeseriesWindow([1.0, 2.0, 3.0], 1)
        arr = w.to_array()
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_repr(self):
        """Test string representation."""
        w = TimeseriesWindow([1.0, 2.0, 3.0], 1)
        r = repr(w)
        assert "TimeseriesWindow" in r
        assert "len=3" in r
        assert "current_index=1" in r

    def test_invalid_current_index(self):
        """Test that invalid current_index raises."""
        with pytest.raises(ValueError, match="out of bounds"):
            TimeseriesWindow([1.0, 2.0], 5)


class TestFourBoxTimeseriesWindow:
    """Tests for FourBoxTimeseriesWindow type."""

    def test_constructor(self):
        """Test basic construction."""
        values = [[1, 2, 3, 4], [5, 6, 7, 8]]
        w = FourBoxTimeseriesWindow(values, 1)
        assert w.current_index == 1
        assert len(w) == 2

    def test_current(self):
        """Test current returns FourBoxSlice."""
        values = [[1, 2, 3, 4], [5, 6, 7, 8]]
        w = FourBoxTimeseriesWindow(values, 1)
        current = w.current
        assert current.northern_ocean == 5.0
        assert current.northern_land == 6.0
        assert current.southern_ocean == 7.0
        assert current.southern_land == 8.0

    def test_previous(self):
        """Test previous returns FourBoxSlice."""
        values = [[1, 2, 3, 4], [5, 6, 7, 8]]
        w = FourBoxTimeseriesWindow(values, 1)
        prev = w.previous
        assert prev.northern_ocean == 1.0

    def test_region(self):
        """Test region returns scalar TimeseriesWindow."""
        values = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        w = FourBoxTimeseriesWindow(values, 2)
        region_0 = w.region(0)  # northern_ocean
        assert region_0.current == 9.0
        assert region_0.previous == 5.0

    def test_region_invalid_index(self):
        """Test region with invalid index raises."""
        values = [[1, 2, 3, 4]]
        w = FourBoxTimeseriesWindow(values, 0)
        with pytest.raises(ValueError, match="Invalid region index"):
            w.region(4)

    def test_repr(self):
        """Test string representation."""
        values = [[1, 2, 3, 4], [5, 6, 7, 8]]
        w = FourBoxTimeseriesWindow(values, 1)
        r = repr(w)
        assert "FourBoxTimeseriesWindow" in r


class TestHemisphericTimeseriesWindow:
    """Tests for HemisphericTimeseriesWindow type."""

    def test_constructor(self):
        """Test basic construction."""
        values = [[1, 2], [3, 4]]
        w = HemisphericTimeseriesWindow(values, 1)
        assert w.current_index == 1
        assert len(w) == 2

    def test_current(self):
        """Test current returns HemisphericSlice."""
        values = [[1, 2], [3, 4]]
        w = HemisphericTimeseriesWindow(values, 1)
        current = w.current
        assert current.northern == 3.0
        assert current.southern == 4.0

    def test_region(self):
        """Test region returns scalar TimeseriesWindow."""
        values = [[1, 2], [3, 4], [5, 6]]
        w = HemisphericTimeseriesWindow(values, 2)
        region_0 = w.region(0)  # northern
        assert region_0.current == 5.0
        assert region_0.previous == 3.0

    def test_region_invalid_index(self):
        """Test region with invalid index raises."""
        values = [[1, 2]]
        w = HemisphericTimeseriesWindow(values, 0)
        with pytest.raises(ValueError, match="Invalid region index"):
            w.region(2)


class TestRequirementDefinitionWithGridType:
    """Tests for RequirementDefinition with grid_type parameter."""

    def test_requirement_definition_default_grid_type(self):
        """Test that default grid_type is Scalar."""
        req = RequirementDefinition("Emissions|CO2", "GtC / yr", RequirementType.Input)
        # PyO3 enums need name comparison
        assert str(req.grid_type) == str(GridType.Scalar)

    def test_requirement_definition_explicit_grid_type(self):
        """Test setting explicit grid_type."""
        req = RequirementDefinition(
            "Temperature", "K", RequirementType.Output, GridType.FourBox
        )
        assert str(req.grid_type) == str(GridType.FourBox)

    def test_requirement_definition_hemispheric_grid_type(self):
        """Test setting hemispheric grid_type."""
        req = RequirementDefinition(
            "Precipitation", "mm / yr", RequirementType.Input, GridType.Hemispheric
        )
        assert str(req.grid_type) == str(GridType.Hemispheric)
