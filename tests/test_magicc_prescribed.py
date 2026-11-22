"""Forcing derived from MAGICC7 input files before a run."""

import numpy as np
import numpy.testing as npt
import pytest

from rscm.magicc import ClimateUDEBBuilder
from rscm.magicc import prescribed as pr


def _volcanic_efficacy(northern, southern, *, first, last, **kwargs):
    times = 2000.0 + np.arange(1, len(northern) + 1) / 12
    forcing = pr.volcanic_substeps(
        times, northern, southern, regional_pattern_years=(first, last), **kwargs
    )
    return ClimateUDEBBuilder.from_parameters(
        {"subannual_forcing": forcing}
    ).subannual_efficacy()


def test_volcanic_pattern_hand_example():
    # The window is the 12 months ending from 2001.0 to before 2002.0
    northern = [99.0] * 11 + [1.0] * 6 + [5.0] * 6 + [99.0]
    southern = [99.0] * 11 + [3.0] * 12 + [99.0]

    pattern, _ = _volcanic_efficacy(
        northern, southern, first=2001, last=2001, scale=0.5
    )

    # North 1.5, south 1.5, so the global mean is 1.5 and each box is -1
    npt.assert_allclose(pattern, [-1.0] * 4, rtol=0, atol=1e-12)


def test_volcanic_pattern_follows_scale():
    northern = np.linspace(-1.0, -3.0, 24)
    southern = northern / 2

    unscaled, _ = _volcanic_efficacy(
        northern, southern, first=2000, last=2001, scale=1.0
    )
    scaled, _ = _volcanic_efficacy(northern, southern, first=2000, last=2001, scale=0.5)

    npt.assert_allclose(
        np.array(scaled) + 1, (np.array(unscaled) + 1) * 0.5, atol=1e-12
    )


def test_volcanic_pattern_sums_global_mean_by_box():
    pattern, _ = _volcanic_efficacy([0.3], [0.2], first=2000, last=2000, scale=1.0)

    # MAGICC7 sums box times area, which rounds above the hemispheric mean of 0.25.
    assert tuple(pattern) == (
        0.3 - 0.25000000000000006 - 1,
        0.3 - 0.25000000000000006 - 1,
        0.2 - 0.25000000000000006 - 1,
        0.2 - 0.25000000000000006 - 1,
    )


def test_volcanic_internal_efficacy_follows_the_climate():
    forcing = pr.volcanic_substeps(
        2000.0 + np.arange(1, 25) / 12,
        np.full(24, -2.0),
        np.full(24, -1.0),
        regional_pattern_years=(2000, 2001),
    )

    def efficacy(ecs):
        builder = ClimateUDEBBuilder.from_parameters(
            {"ecs": ecs, "subannual_forcing": forcing}
        )
        pattern, internal = builder.subannual_efficacy()
        at_3_71 = ClimateUDEBBuilder.from_parameters({"ecs": ecs, "erf_2xco2": 3.71})
        assert internal == at_3_71.internal_efficacy(pattern)
        return internal

    assert efficacy(2.0) != efficacy(4.5)


def test_volcanic_substeps_build_from_a_shared_array():
    times = np.arange(2000.0, 2002.0, 1 / 12) + 1 / 12
    forcing = pr.volcanic_substeps(times, -np.ones(24), -np.ones(24))
    assert not forcing["values"].flags.writeable
    ClimateUDEBBuilder.from_parameters({"subannual_forcing": forcing}).build()

    with pytest.raises(ValueError, match="2-D float64"):
        ClimateUDEBBuilder.from_parameters(
            {"subannual_forcing": {**forcing, "values": np.ones(4, dtype=np.float32)}}
        )
    with pytest.raises(ValueError, match="4 boxes"):
        ClimateUDEBBuilder.from_parameters(
            {"subannual_forcing": {**forcing, "values": np.ones((24, 3))}}
        )
