"""Tests for spatial grid types."""

import numpy as np
import pytest

from rscm._lib.core.spatial import (
    FourBoxGrid,
    FourBoxRegion,
    HemisphericGrid,
    HemisphericRegion,
    ScalarGrid,
    ScalarRegion,
)


def test_scalar_grid():
    """Test ScalarGrid basic functionality."""
    grid = ScalarGrid()
    assert grid.size() == 1
    assert grid.grid_name() == "Scalar"
    assert grid.region_names() == ["Global"]

    # Test aggregation (should return the same value)
    value = [288.15]
    assert grid.aggregate_global(value) == 288.15


def test_four_box_grid_default():
    """Test FourBoxGrid with default (equal) weights."""
    grid = FourBoxGrid()
    assert grid.size() == 4
    assert grid.grid_name() == "FourBox"
    assert len(grid.region_names()) == 4
    assert grid.region_names() == [
        "Northern Ocean",
        "Northern Land",
        "Southern Ocean",
        "Southern Land",
    ]

    # Test aggregation with equal weights
    values = [15.0, 14.0, 10.0, 9.0]
    global_value = grid.aggregate_global(values)
    assert global_value == 12.0  # (15 + 14 + 10 + 9) / 4


def test_four_box_grid_magicc_standard():
    """Test FourBoxGrid.magicc_standard() constructor."""
    grid = FourBoxGrid.magicc_standard()
    assert grid.size() == 4
    weights = grid.weights()
    assert len(weights) == 4
    assert sum(weights) == pytest.approx(1.0)


def test_four_box_grid_custom_weights():
    """Test FourBoxGrid with custom weights."""
    custom_weights = [0.3, 0.2, 0.4, 0.1]
    grid = FourBoxGrid.with_weights(custom_weights)
    assert grid.size() == 4

    weights = grid.weights()
    assert np.allclose(weights, custom_weights)

    # Test weighted aggregation
    values = [10.0, 20.0, 30.0, 40.0]
    global_value = grid.aggregate_global(values)
    expected = 10.0 * 0.3 + 20.0 * 0.2 + 30.0 * 0.4 + 40.0 * 0.1
    assert global_value == pytest.approx(expected)


def test_four_box_region_constants():
    """Test FourBoxRegion constants."""
    assert FourBoxRegion.NORTHERN_OCEAN == 0
    assert FourBoxRegion.NORTHERN_LAND == 1
    assert FourBoxRegion.SOUTHERN_OCEAN == 2
    assert FourBoxRegion.SOUTHERN_LAND == 3


def test_hemispheric_grid_default():
    """Test HemisphericGrid with default (equal) weights."""
    grid = HemisphericGrid()
    assert grid.size() == 2
    assert grid.grid_name() == "Hemispheric"
    assert len(grid.region_names()) == 2
    assert grid.region_names() == ["Northern Hemisphere", "Southern Hemisphere"]

    # Test aggregation with equal weights
    values = [15.0, 10.0]
    global_value = grid.aggregate_global(values)
    assert global_value == 12.5  # (15 + 10) / 2


def test_hemispheric_grid_equal_weights():
    """Test HemisphericGrid.equal_weights() constructor."""
    grid = HemisphericGrid.equal_weights()
    assert grid.size() == 2
    weights = grid.weights()
    assert np.allclose(weights, [0.5, 0.5])


def test_hemispheric_grid_custom_weights():
    """Test HemisphericGrid with custom weights."""
    custom_weights = [0.6, 0.4]
    grid = HemisphericGrid.with_weights(custom_weights)
    assert grid.size() == 2

    weights = grid.weights()
    assert np.allclose(weights, custom_weights)

    # Test weighted aggregation
    values = [20.0, 10.0]
    global_value = grid.aggregate_global(values)
    expected = 20.0 * 0.6 + 10.0 * 0.4
    assert global_value == pytest.approx(expected)


def test_hemispheric_region_constants():
    """Test HemisphericRegion constants."""
    assert HemisphericRegion.NORTHERN == 0
    assert HemisphericRegion.SOUTHERN == 1


def test_scalar_region_constants():
    """Test ScalarRegion constants."""
    assert ScalarRegion.GLOBAL == 0
