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

import functools
import gzip
import math
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "AerosolHistory",
    "area_fractions",
    "bc_on_snow",
    "cloud_albedo",
    "direct_aerosol",
    "forward_midyear",
    "land_use",
    "mineral_dust",
    "nitrate",
    "on_axis",
    "read_input",
    "solar",
    "volcanic_pattern",
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
    index = round(year - years[0])
    if not 0 <= index < len(years):
        msg = f"switch year {year} is outside the run"
        raise ValueError(msg)
    return index


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
    mid = forward_midyear(extender)[sw:, None]
    out[sw:] = mid * out[sw] / mid[0]
    return out


def _scale(target: float | None, factor: float | None, reference: float) -> float:
    """``factor``, or the factor that scales ``reference`` to ``target``."""
    if factor is not None:
        return float(factor)
    if target is None:
        msg = "give harmonisation_target or factor"
        raise ValueError(msg)
    return float(target / reference)


def _harmonise(
    boxes: NDArray[np.float64],
    years: NDArray[np.float64],
    year: float,
    target: float | None,
    factor: float | None = None,
) -> float:
    """Factor that sets the global forcing change since the first year to ``target``.

    A given ``factor`` is returned as it is.
    """
    if factor is not None:
        return float(factor)
    i = _switch_index(years, year)
    return _scale(target, None, float(area_mean(boxes[i]) - area_mean(boxes[0])))


def direct_aerosol(  # noqa: PLR0913
    years: ArrayLike,
    industrial: AerosolHistory,
    biomass: AerosolHistory,
    industrial_emissions: ArrayLike,
    biomass_emissions: ArrayLike,
    *,
    harmonisation_target: float | None = None,
    harmonisation_reference_year: float = 2019,
    factor: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Industrial and biomass direct forcing of SOx, BC or OC.

    Industrial forcing scales with emissions above the first year's after the
    file ends, and is harmonised to ``harmonisation_target`` W/m²
    in ``harmonisation_reference_year`` (`RF_*I_DIR_APPLY=1`),
    or scaled by ``factor`` (`RF_*I_DIR_APPLY=2`).
    Biomass forcing (`RF_*B_DIR_APPLY=3`) uses the industrial factor.
    MAGICC7 v7.5.3 extends each biomass box by half the change in midyear
    emissions above the first year's, which is reproduced here.
    The change is negated where the box's final file value and midyear
    emissions differ in sign.
    """
    y = np.asarray(years, dtype=np.float64)
    ind = np.asarray(industrial_emissions, dtype=np.float64)
    ind_rf = _extend(industrial, y, ind - ind[0])
    factor = _harmonise(
        ind_rf, y, harmonisation_reference_year, harmonisation_target, factor
    )

    bio_rf = on_axis(biomass, y)
    sw = _switch_index(y, biomass.last_year())
    bio = np.asarray(biomass_emissions, dtype=np.float64)
    mid = forward_midyear(bio - bio[0])
    change = (mid[sw:] - mid[sw])[:, None] * 0.5
    sign = np.where(bio_rf[sw] / mid[sw] < 0, -1.0, 1.0)
    bio_rf[sw:] = sign * change + bio_rf[sw]
    return ind_rf * factor, bio_rf * factor


def bc_on_snow(  # noqa: PLR0913
    years: ArrayLike,
    history: AerosolHistory,
    bc_emissions: ArrayLike,
    *,
    harmonisation_target: float | None = None,
    harmonisation_reference_year: float = 2019,
    factor: float | None = None,
) -> NDArray[np.float64]:
    """Return black carbon on snow forcing relative to the first year.

    The file is extended with total BC emissions
    and harmonised to ``harmonisation_target``, or scaled by ``factor``.
    """
    y = np.asarray(years, dtype=np.float64)
    rf = _extend(history, y, bc_emissions)
    factor = _harmonise(
        rf, y, harmonisation_reference_year, harmonisation_target, factor
    )
    return rf * factor - rf[0] * factor


def _hold_after(
    values: NDArray[np.float64], years: NDArray[np.float64], year: float
) -> NDArray[np.float64]:
    out = values.copy()
    i = round(year - years[0])
    if 0 <= i < len(years) - 1:
        out[i + 1 :] = out[i]
    return out


def land_use(  # noqa: PLR0913
    years: ArrayLike,
    history: AerosolHistory,
    land_use_co2: ArrayLike,
    *,
    harmonisation_target: float | None = None,
    harmonisation_reference_year: float = 2019,
    constant_after: float = 2018,
    factor: float | None = None,
) -> NDArray[np.float64]:
    """Land-use albedo forcing.

    The file is extended with cumulative land-use CO2 emissions and held
    after ``constant_after``, which does not affect harmonisation.
    It is harmonised to ``harmonisation_target``, or scaled by ``factor``.
    Years before the file have no land-use forcing.
    """
    y = np.asarray(years, dtype=np.float64)
    raw = _extend(history, y, np.cumsum(land_use_co2), before=0.0)
    factor = _harmonise(
        raw, y, harmonisation_reference_year, harmonisation_target, factor
    )
    return _hold_after(raw, y, constant_after) * factor


def mineral_dust(  # noqa: PLR0913
    years: ArrayLike,
    land_use_history: AerosolHistory,
    land_use_co2: ArrayLike,
    pattern: ArrayLike,
    *,
    harmonisation_target: float | None = None,
    harmonisation_reference_year: float = 2019,
    constant_after: float = 2018,
    factor: float | None = None,
) -> NDArray[np.float64]:
    """Mineral dust forcing, following global land-use forcing on its own pattern.

    It is harmonised to ``harmonisation_target``, or scaled by ``factor``.
    """
    y = np.asarray(years, dtype=np.float64)
    raw = area_mean(_extend(land_use_history, y, np.cumsum(land_use_co2), before=0.0))
    if factor is None:
        i = _switch_index(y, harmonisation_reference_year)
        factor = _scale(harmonisation_target, None, raw[i] - raw[0])
    scaled = raw * factor
    p = np.asarray(pattern, dtype=np.float64)
    boxes = scaled[:, None] * p / area_mean(p)
    return _hold_after(boxes, y, constant_after)


def _nitrate_total(  # noqa: PLR0913
    years: ArrayLike,
    nox: tuple[ArrayLike, ArrayLike],
    nh3: tuple[ArrayLike, ArrayLike],
    sox: ArrayLike,
    pattern: ArrayLike,
    *,
    harmonisation_target: float | None = None,
    harmonisation_reference_year: float = 2019,
    natural_nox: float = 10.0,
    natural_nh3: float = 21.0,
    natural_sox: float = 14.0,
    lambda_so2: float = 1.6,
    burden_pi: float = 0.0085,
    optical_thickness_pi: float = 1.41,
    burden_to_optical_thickness_exponent: float = 0.76,
    factor: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Total nitrate forcing boxes, before the source split, and its global value."""
    y = np.asarray(years, dtype=np.float64)
    nox_i, nox_b = (np.asarray(e, dtype=np.float64) for e in nox)
    nh3_i, nh3_b = (np.asarray(e, dtype=np.float64) for e in nh3)
    n = ((nox_i + nox_b) + natural_nox) / 14.0
    a = ((nh3_i + nh3_b) + natural_nh3) / 14.0
    s = (np.asarray(sox, dtype=np.float64) + natural_sox) / 32.07
    alpha = (n * a) / ((s * lambda_so2) / a + 1.0)
    burden = (burden_pi * alpha) / alpha[0]
    # math.pow matches MAGICC7's libm, unlike numpy's vectorised power.
    optical_thickness = np.array(
        [math.pow(b / burden_pi, burden_to_optical_thickness_exponent) for b in burden]
    )
    rf = optical_thickness * optical_thickness_pi - optical_thickness_pi
    if factor is None:
        reference = rf[_switch_index(y, harmonisation_reference_year)]
        factor = _scale(harmonisation_target, None, reference)
    rf = rf * factor

    product = (nh3_i * 0.5 + nh3_b * 0.5) * (nox_b * 0.5 + nox_i * 0.5)
    hemisphere = ((rf + rf) * product) / (product + product)
    w = np.asarray(pattern, dtype=np.float64)
    land = area_fractions()[1::2] * 2
    ocean = 1 - land
    total = np.empty((len(y), 4))
    for h in range(2):
        sea, lnd, wo, wl = ocean[h], land[h], w[2 * h], w[2 * h + 1]
        total[:, 2 * h] = ((hemisphere * (sea + lnd)) * wo) / (sea * wo + lnd * wl)
        total[:, 2 * h + 1] = ((hemisphere * (lnd + sea)) * wl) / (lnd * wl + sea * wo)
    return total, rf


def nitrate(  # noqa: PLR0913
    years: ArrayLike,
    nox: tuple[ArrayLike, ArrayLike],
    nh3: tuple[ArrayLike, ArrayLike],
    sox: ArrayLike,
    pattern: ArrayLike,
    *,
    harmonisation_target: float | None = None,
    harmonisation_reference_year: float = 2019,
    natural_nox: float = 10.0,
    natural_nh3: float = 21.0,
    natural_sox: float = 14.0,
    lambda_so2: float = 1.6,
    burden_pi: float = 0.0085,
    optical_thickness_pi: float = 1.41,
    burden_to_optical_thickness_exponent: float = 0.76,
    factor: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Industrial and biomass nitrate direct forcing (Hauglustaine et al. 2014).

    ``nox`` and ``nh3`` are (industrial, biomass) emissions in Mt N/yr and ``sox``
    the total in Mt S/yr, all without natural sources.
    Each hemisphere's forcing follows its share of NOx x NH3, spread over its
    boxes by ``pattern``, and splits between the sources by the same product.
    Emissions fall half on each hemisphere's land, as MAGICC7 spreads global inputs.
    The forcing is harmonised to ``harmonisation_target``, or scaled by ``factor``.
    """
    total, _ = _nitrate_total(
        years,
        nox,
        nh3,
        sox,
        pattern,
        harmonisation_target=harmonisation_target,
        harmonisation_reference_year=harmonisation_reference_year,
        natural_nox=natural_nox,
        natural_nh3=natural_nh3,
        natural_sox=natural_sox,
        lambda_so2=lambda_so2,
        burden_pi=burden_pi,
        optical_thickness_pi=optical_thickness_pi,
        burden_to_optical_thickness_exponent=burden_to_optical_thickness_exponent,
        factor=factor,
    )
    nox_i, nox_b = (np.asarray(e, dtype=np.float64) for e in nox)
    nh3_i, nh3_b = (np.asarray(e, dtype=np.float64) for e in nh3)
    industrial = (nh3_i * 0.5) * (nox_i * 0.5)
    biomass = (nh3_b * 0.5) * (nox_b * 0.5)
    with np.errstate(invalid="ignore"):
        biomass_share = biomass / (industrial + biomass)
    biomass_share = np.where(np.isnan(biomass_share), 0.5, biomass_share)
    return (
        total * (1.0 - biomass_share)[:, None],
        total * biomass_share[:, None],
    )


def nitrate_pattern(
    *args: Any,
    index: int,
    **kwargs: Any,
) -> list[float]:
    """Regional pattern of nitrate forcing, as MAGICC7 sets ``RF_REGIONS_NO3``.

    It is the total forcing boxes over the global forcing at ``index``.
    Takes the arguments of `nitrate`.
    """
    total, rf = _nitrate_total(*args, **kwargs)
    return (total[index] / rf[index]).tolist()


# Cloud-active species and their components, in MAGICC7's summation order.
CLOUD_GROUPS = {
    "bc": ("BCI", "BCB"),
    "oc": ("OCI", "OCB", "OCN"),
    "sox": ("SOXI", "SOXNB"),
    "ss": ("SS",),
}


def _cloud_drive(
    emissions: ArrayLike, start: int, natural: float | None = None
) -> tuple[NDArray[np.float64], float]:
    """Each year's emissions plus the next year's, and that sum at ``start``."""
    e = np.asarray(emissions, dtype=np.float64)
    if natural is None:
        forward = e[1:] + e[:-1]
        return forward, forward[start]
    # MAGICC7 orders the natural terms differently in the two sums.
    forward = ((e[:-1] + natural) + e[1:]) + natural
    return forward, ((e[start + 1] + e[start]) + natural) + natural


def _scale_from(
    values: NDArray[np.float64],
    start: int,
    forward: NDArray[np.float64],
    base: float,
) -> NDArray[np.float64]:
    out = values.copy()
    out[start:-1] = (forward[start:, None] * values[start]) / base
    out[-1] = out[-2]
    return out


def cloud_albedo(  # noqa: PLR0913
    years: ArrayLike,
    optical_thickness: Mapping[str, AerosolHistory],
    emissions: Mapping[str, ArrayLike],
    nitrate_forcing: ArrayLike,
    industrial_nox: ArrayLike,
    weights: Mapping[str, float],
    pattern: ArrayLike,
    *,
    harmonisation_target: float | None = None,
    harmonisation_reference_year: float = 2019,
    factor: float | None = None,
    bci_soluble_ratio: float = 0.75,
    natural_sox: float = 14.0,
    pattern_reference_year: float = 2000,
    nitrate_switch_year: float = 2014,
) -> NDArray[np.float64]:
    """Cloud albedo forcing from a log-linear aerosol number index.

    ``optical_thickness`` maps the components of ``CLOUD_GROUPS`` to their files.
    Each group is normalised by its area mean when its files end,
    and ``emissions`` extends each emitted component from there
    (SOXNB adds ``natural_sox``), while OCN and SS are held constant.
    ``nitrate_forcing`` is the total nitrate forcing boxes, extended the same way
    with ``industrial_nox`` from ``nitrate_switch_year``.
    ``weights`` has the keys no3, bc, oc, sox and ss.
    The forcing is harmonised to ``harmonisation_target``, or scaled by ``factor``.
    """
    y = np.asarray(years, dtype=np.float64)
    components = {}
    for names in CLOUD_GROUPS.values():
        group = {}
        for name in names:
            history = optical_thickness[name]
            ot = _hold_after(on_axis(history, y), y, history.last_year())
            group[name] = ot * bci_soluble_ratio if name == "BCI" else ot
        last = min(optical_thickness[name].last_year() for name in names)
        sw = _switch_index(y, last)
        norm = area_mean(sum(ot[sw] for ot in group.values()))
        for name, ot in group.items():
            scaled = ot / norm
            if name in emissions:
                natural = natural_sox if name == "SOXNB" else None
                drive = _cloud_drive(emissions[name], sw, natural)
                scaled = _scale_from(scaled, sw, *drive)
            components[name] = scaled

    no3 = np.asarray(nitrate_forcing, dtype=np.float64)
    sw = _switch_index(y, nitrate_switch_year)
    no3 = _scale_from(no3 / area_mean(no3[sw]), sw, *_cloud_drive(industrial_nox, sw))

    # Summed in order like MAGICC7, because builtin sum compensates float rounding.
    total = functools.reduce(operator.add, (weights[g] for g in ("no3", *CLOUD_GROUPS)))

    def number_index(
        c: Mapping[str, NDArray[np.float64]], nitrate: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        out = nitrate * (weights["no3"] / total)
        for group, names in CLOUD_GROUPS.items():
            out = out + sum(c[name] for name in names) * (weights[group] / total)
        return out

    # The first year is MAGICC7's reference, with no nitrate.
    index = number_index(components, no3)
    reference = number_index({k: v[0] for k, v in components.items()}, np.zeros(4))
    # math.log10 matches MAGICC7's libm, unlike numpy's vectorised log10.
    log10 = np.vectorize(math.log10, otypes=[np.float64])
    cdnc = log10(index) - log10(reference)
    pattern_index = _switch_index(y, pattern_reference_year)
    boxes = (cdnc / cdnc[pattern_index]) * np.asarray(pattern)
    if factor is None:
        harmonisation_index = _switch_index(y, harmonisation_reference_year)
        harmonised = float(area_mean(boxes[harmonisation_index]))
        factor = _scale(harmonisation_target, None, harmonised)
    return boxes * factor


def solar(values: ArrayLike, scale: float = 0.66) -> NDArray[np.float64]:
    """Global solar forcing, which MAGICC7 scales without offsetting."""
    return np.asarray(values, dtype=np.float64) * scale


def volcanic_pattern(  # noqa: PLR0913
    times: ArrayLike,
    northern: ArrayLike,
    southern: ArrayLike,
    *,
    start_year: float,
    last_year: float,
    scale: float = 0.74,
) -> tuple[float, float, float, float]:
    """MAGICC7's regional pattern of volcanic forcing, behind its internal efficacy.

    MAGICC7 takes each box's mean forcing over the months from ``start_year``
    to the end of ``last_year``, less the global mean and 1.
    """
    t = np.asarray(times, dtype=np.float64)
    nh = np.asarray(northern, dtype=np.float64)
    sh = np.asarray(southern, dtype=np.float64)
    ok = ~(np.isnan(nh) | np.isnan(sh)) & (t >= start_year) & (t < last_year + 1)
    # Sequential sums of the scaled values, as MAGICC7 adds them.
    north, south = (np.cumsum(x[ok] * scale)[-1] / ok.sum() for x in (nh, sh))
    mean = float(area_mean([north, north, south, south]))
    return (north - mean - 1, north - mean - 1, south - mean - 1, south - mean - 1)


def volcanic_substeps(  # noqa: PLR0913
    times: ArrayLike,
    northern: ArrayLike,
    southern: ArrayLike,
    *,
    scale: float = 0.74,
    efficacy: float = 1.0,
    internal_efficacy: float = 1.0078325986062864,
    regional_pattern: Sequence[float] | None = None,
    until: float | None = None,
) -> dict[str, object]:
    """Monthly volcanic forcing as ClimateUDEB's ``subannual_forcing``.

    Each valid month in ``times`` (decimal years) drives the substep ending
    at that time, matching MAGICC7's reading of ``.MON`` files.
    With ``regional_pattern``, ClimateUDEB divides ``efficacy`` by the internal
    efficacy, which it recalculates each year,
    and ``internal_efficacy`` applies only in the first year.
    Months more than a year after ``until`` are left out, as no run to then reads them.
    """
    t = np.asarray(times, dtype=np.float64)
    nh = np.asarray(northern, dtype=np.float64)
    sh = np.asarray(southern, dtype=np.float64)
    ok = ~(np.isnan(nh) | np.isnan(sh))
    if until is not None:
        ok &= t <= until + 1
    t, nh, sh = t[ok], nh[ok], sh[ok]
    factor = scale
    if regional_pattern is None:
        factor *= efficacy / internal_efficacy
    values = np.stack([nh, nh, sh, sh], axis=1) * factor
    out: dict[str, object] = {"start": float(t[0]) - 1 / 12, "values": values.tolist()}
    if regional_pattern is not None:
        out |= {
            "regional_pattern": list(regional_pattern),
            "internal_efficacy": internal_efficacy,
            "efficacy": efficacy,
        }
    return out
