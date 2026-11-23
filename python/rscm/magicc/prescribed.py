"""Forcing that MAGICC7 derives from input files and emissions before a run starts.

Aerosol, land-use, solar and volcanic forcing in MAGICC7 depend only on
prescribed inputs, so MAGICC7 calculates them once during initialisation.
These functions reproduce those calculations for an annual axis whose first
year is the preindustrial reference (1750 in MAGICC7).

Four-box arrays have shape ``(years, 4)`` in the order NH ocean, NH land,
SH ocean, SH land. Emissions are annual arrays on the same axis.
Values for a year are at its start, like every MAGICC7 forcing.
The final year needs the following year's emissions, which these functions
replace with the final year's own.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "AerosolHistory",
    "area_fractions",
    "bc_on_snow",
    "direct_aerosol",
    "forward_midyear",
    "land_use",
    "mineral_dust",
    "on_axis",
    "read_input",
    "volcanic_substeps",
]


def area_fractions(nh_land: float = 0.42, sh_land: float = 0.21) -> NDArray[np.float64]:
    """Global area fractions of the four boxes."""
    return np.array([1 - nh_land, nh_land, 1 - sh_land, sh_land]) / 2


def area_mean(boxes: ArrayLike) -> NDArray[np.float64]:
    """Area-weighted mean of the last axis, summed box by box as MAGICC7 does."""
    b = np.asarray(boxes, dtype=np.float64)
    a = area_fractions()
    return b[..., 0] * a[0] + b[..., 1] * a[1] + b[..., 2] * a[2] + b[..., 3] * a[3]


def forward_midyear(values: ArrayLike) -> NDArray[np.float64]:
    """Mean of each year and the next, which MAGICC7 uses to extend forcing.

    The final year has no next year, so it repeats the previous mean,
    as MAGICC7 holds this forcing in the final year of a run.
    """
    x = np.asarray(values, dtype=np.float64)
    mid = (x[:-1] + x[1:]) / 2
    return np.r_[mid, mid[-1:]]


@dataclass(frozen=True)
class AerosolHistory:
    """A historical four-box forcing or optical thickness file."""

    years: ArrayLike
    values: ArrayLike

    def last_year(self) -> float:
        """Return the final year of the file."""
        return float(np.asarray(self.years)[-1])


def read_input(path: Path) -> AerosolHistory:
    """Read a MAGICC7 ``.IN`` or ``.MON`` input file, optionally gzipped."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        lines = f.read().splitlines()
    header = max(i for i, line in enumerate(lines) if line.split()[:1] == ["YEARS"])
    rows = [line.split() for line in lines[header + 1 :] if line.strip()]
    data = np.array(rows, dtype=np.float64)
    return AerosolHistory(data[:, 0], data[:, 1:])


def on_axis(
    history: AerosolHistory, years: ArrayLike, *, before: float | None = None
) -> NDArray[np.float64]:
    """File values on ``years``, which are NaN after the file ends.

    Years before the file starts take ``before``, or the first value if it is None.
    """
    hy = np.asarray(history.years, dtype=np.float64)
    hv = np.asarray(history.values, dtype=np.float64)
    hv = hv.reshape(len(hy), -1)
    y = np.asarray(years, dtype=np.float64)
    out = np.full((len(y), hv.shape[1]), np.nan)
    early = y < hy[0]
    within = ~early & (y <= hy[-1])
    out[early] = hv[0] if before is None else before
    out[within] = hv[np.rint(y[within] - hy[0]).astype(np.intp)]
    return out


def _switch_index(years: NDArray[np.float64], year: float) -> int:
    """Index of ``year``, moved onto the first or last run year like MAGICC7 does."""
    return int(np.clip(round(year - years[0]), 0, len(years) - 1))


def _extend(
    history: AerosolHistory,
    years: ArrayLike,
    extender: ArrayLike,
    *,
    before: float | None = None,
) -> NDArray[np.float64]:
    """Scale the last file value with the forward-midyear ``extender`` afterwards."""
    y = np.asarray(years, dtype=np.float64)
    out = on_axis(history, y, before=before)
    sw = _switch_index(y, history.last_year())
    if sw == len(y) - 1:
        return out
    mid = forward_midyear(extender)[sw:, None]
    out[sw:] = mid * out[sw] / mid[0]
    return out


def direct_aerosol(
    years: ArrayLike,
    industrial: AerosolHistory,
    biomass: AerosolHistory,
    industrial_emissions: ArrayLike,
    biomass_emissions: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unscaled industrial and biomass direct forcing of SOx, BC or OC.

    Industrial forcing scales with emissions above the first year's after the file ends.
    MAGICC7 v7.5.3 extends each biomass box by half the change in midyear
    emissions above the first year's, which is reproduced here.
    The change is negated where the box's final file value and midyear
    emissions differ in sign.
    `ForcingAggregation` scales both by the industrial agent's scale.
    """
    y = np.asarray(years, dtype=np.float64)
    ind = np.asarray(industrial_emissions, dtype=np.float64)
    ind_rf = _extend(industrial, y, ind - ind[0])

    bio_rf = on_axis(biomass, y)
    sw = _switch_index(y, biomass.last_year())
    bio = np.asarray(biomass_emissions, dtype=np.float64)
    mid = forward_midyear(bio - bio[0])
    change = (mid[sw:] - mid[sw])[:, None] * 0.5
    sign = np.where(bio_rf[sw] / mid[sw] < 0, -1.0, 1.0)
    bio_rf[sw:] = sign * change + bio_rf[sw]
    return ind_rf, bio_rf


def bc_on_snow(
    years: ArrayLike, history: AerosolHistory, bc_emissions: ArrayLike
) -> NDArray[np.float64]:
    """Unscaled black carbon on snow forcing, extended with total BC emissions.

    `ForcingAggregation` takes it relative to the first year after scaling.
    """
    return _extend(history, np.asarray(years, dtype=np.float64), bc_emissions)


def _hold_after(
    values: NDArray[np.float64], years: NDArray[np.float64], year: float
) -> NDArray[np.float64]:
    out = values.copy()
    i = round(year - years[0])
    if 0 <= i < len(years) - 1:
        out[i + 1 :] = out[i]
    return out


def land_use(
    years: ArrayLike, history: AerosolHistory, land_use_co2: ArrayLike
) -> NDArray[np.float64]:
    """Unscaled land-use albedo forcing.

    The file is extended with cumulative land-use CO2 emissions.
    Years before the file have no land-use forcing.
    """
    y = np.asarray(years, dtype=np.float64)
    return _extend(history, y, np.cumsum(land_use_co2), before=0.0)


def mineral_dust(
    years: ArrayLike, land_use_history: AerosolHistory, land_use_co2: ArrayLike
) -> NDArray[np.float64]:
    """Unscaled global mineral dust forcing, which follows global land-use forcing."""
    return area_mean(land_use(years, land_use_history, land_use_co2))


def volcanic_substeps(  # noqa: PLR0913
    times: ArrayLike,
    northern: ArrayLike,
    southern: ArrayLike,
    *,
    scale: float = 0.74,
    efficacy: float = 1.0,
    internal_efficacy: float | None = None,
    regional_pattern_years: tuple[float, float] | None = None,
    initial_erf_2xco2: float = 3.71,
    until: float | None = None,
) -> dict[str, object]:
    """Monthly volcanic forcing as ClimateUDEB's ``subannual_forcing``.

    Each valid month in ``times`` (decimal years) drives the substep ending
    at that time, matching MAGICC7's reading of ``.MON`` files.
    ClimateUDEB multiplies the values by ``scale`` and ``efficacy`` over the internal
    efficacy.
    With ``regional_pattern_years`` it takes a regional pattern from the scaled months
    ending in those years and recalculates the internal efficacy each year.
    ``internal_efficacy`` applies only in the first years,
    and unset ClimateUDEB derives it from the pattern at ``initial_erf_2xco2``.
    Months more than a year after ``until`` are left out, as no run to then reads them.
    The values are a read-only array that specs share, so never make it writeable.
    """
    t = np.asarray(times, dtype=np.float64)
    nh = np.asarray(northern, dtype=np.float64)
    sh = np.asarray(southern, dtype=np.float64)
    ok = ~(np.isnan(nh) | np.isnan(sh))
    if until is not None:
        ok &= t <= until + 1
    t, nh, sh = t[ok], nh[ok], sh[ok]
    values = np.stack([nh, nh, sh, sh], axis=1)
    values.flags.writeable = False
    out: dict[str, object] = {
        "start": float(t[0]) - 1 / 12,
        "values": values,
        "scale": scale,
        "internal_efficacy": internal_efficacy,
        "efficacy": efficacy,
        "initial_erf_2xco2": initial_erf_2xco2,
    }
    if regional_pattern_years is not None:
        out["regional_pattern_years"] = list(regional_pattern_years)
    return out
