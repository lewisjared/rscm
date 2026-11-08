"""Forcing derived from MAGICC7 input files before a run."""

import numpy as np
import numpy.testing as npt
import pytest

from rscm.magicc import prescribed as pr


def test_volcanic_pattern_hand_example():
    times = [1999.5, 2000.0, 2000.5, 2001.0, 2001.5, 2002.0]
    northern = [99.0, 1.0, 3.0, np.nan, 5.0, 99.0]
    southern = [99.0, 1.0, 1.0, 7.0, 3.0, 99.0]

    pattern = pr.volcanic_pattern(
        times, northern, southern, start_year=2000, last_year=2001, scale=0.75
    )

    # Months 2000.0, 2000.5 and 2001.5 count.
    # North 3 * 0.75, south 5/3 * 0.75, global mean 1.75
    npt.assert_allclose(pattern, [-0.5, -0.5, -1.5, -1.5], rtol=0, atol=1e-12)


def test_volcanic_pattern_symmetric_forcing():
    times = np.arange(2000.0, 2002.0, 1 / 12)
    forcing = np.linspace(-1.0, -3.0, len(times))

    pattern = pr.volcanic_pattern(
        times, forcing, forcing, start_year=2000, last_year=2001
    )

    npt.assert_allclose(pattern, [-1.0] * 4, rtol=0, atol=1e-12)


def test_volcanic_pattern_sums_global_mean_by_box():
    pattern = pr.volcanic_pattern(
        [2000.0], [0.3], [0.2], start_year=2000, last_year=2000, scale=1.0
    )

    # MAGICC7 sums box times area, which rounds above the hemispheric mean of 0.25.
    assert pattern == (
        0.3 - 0.25000000000000006 - 1,
        0.3 - 0.25000000000000006 - 1,
        0.2 - 0.25000000000000006 - 1,
        0.2 - 0.25000000000000006 - 1,
    )


def test_harmonise_returns_a_given_factor():
    years = np.array([2000.0, 2001.0, 2002.0])

    assert pr._harmonise(np.ones((3, 4)), years, 2001, None, factor=0.5) == 0.5


def test_harmonise_needs_a_target_or_factor():
    years = np.array([2000.0, 2001.0, 2002.0])

    with pytest.raises(ValueError, match="harmonisation_target or factor"):
        pr._harmonise(np.ones((3, 4)), years, 2001, None)
