"""Forcing derived from MAGICC7 input files before a run."""

import numpy as np
import numpy.testing as npt

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
