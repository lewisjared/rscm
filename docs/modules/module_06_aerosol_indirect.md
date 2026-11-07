# Aerosol Indirect Forcing

## Overview

This module calculates the forcing of aerosols through their effect on clouds.
More aerosol gives more cloud condensation nuclei, and so more but smaller cloud droplets.
MAGICC7 has two effects:

- **Cloud albedo** (first indirect or Twomey effect): smaller droplets make clouds brighter.
- **Cloud cover** (second indirect or Albrecht effect): smaller droplets rain out more slowly,
  so clouds last longer.
  Its default target forcing is zero, so the default run has only the albedo effect.

The aerosol number comes from optical thickness, or nitrate forcing, as a proxy.
A log-linear relation turns it into a cloud droplet number,
whose change is spread over the four boxes with a fixed pattern and harmonised to a target forcing.
The Fortran attributes the formula to "a simple formula by xxxxx as used as well in" Hansen et al. (2005),
with the author missing.

Like the [direct aerosol forcing](module_05_aerosol_direct.md),
this depends only on prescribed inputs, so MAGICC7 calculates it once before the run.
The page uses RF, EffRF and ERF as the [forcing aggregation page defines them](module_07b_rf_aggregation.md#forcing-types).
Four-box arrays follow the box order in [initialisation](module_00_initialisation.md).

## RSCM implementation

- **Function**: `rscm.magicc.prescribed.cloud_albedo`, called by `rscm.magicc.full.prescribed_forcing`.
- **Source**: `python/rscm/magicc/prescribed.py`.
- **Status**: Implemented for the cloud albedo effect.
  See the [status table](README.md#rscm-implementation-status).
- **Tests**: `test_prescribed_forcing[Aerosols|Indirect Effect]` in `tests/regression/test_full_run.py`.

`ForcingAggregation` takes the result as the four-box agent
`Effective Radiative Forcing|Aerosols|Indirect Effect`.

## Equations

### Aerosol number index

Each species group $g$ has components $c$, such as industrial and biomass sources:

| Group | Components     | Proxy                                         | After the historical data             |
| ----- | -------------- | --------------------------------------------- | ------------------------------------- |
| NO3   | NO3            | Total nitrate forcing `DAT_NO3T_RF`           | Scaled with industrial NOx emissions  |
| BC    | BCI, BCB       | Optical thickness, BCI times `CLOUD_BCI2BCB_SOLUBLE_RATIO` | Scaled with BC emissions |
| OC    | OCI, OCB, OCN  | Optical thickness                             | OCI and OCB scaled with emissions, OCN constant |
| SOX   | SOXI, SOXNB    | Optical thickness                             | Scaled with emissions, SOXNB including natural SOx |
| SS    | SS             | Sea salt optical thickness                    | Constant                              |

Each component's normalised index divides its proxy by the area-weighted mean of the group total
in the year its data end, $t_g$:

$$
NI_c(t) = \frac{P_c(t)}{\sum_{i} a_i \sum_{c' \in g} P_{c'}(t_g, i)}
$$

where $a_i$ are the box area fractions.
After $t_g$, an emitted component follows the sum of the year's and next year's emissions:

$$
NI_c(t) = NI_c(t_g) \frac{E_c(t) + E_c(t + 1)}{E_c(t_g) + E_c(t_g + 1)}
$$

### Cloud droplet number

The total aerosol number is the weighted sum

$$
N_A(t) = \frac{1}{\sum_g w_g} \sum_g w_g \sum_{c \in g} NI_c(t)
$$

with the weights $w_g$ = `CLOUD_WEIGHT_*`.
The change in cloud droplet number of each box is

$$
\Delta CDNC(t) = \log_{10} N_A(t) - \log_{10} N_A^{PI}
$$

where $N_A^{PI}$ uses the first-year indices and no nitrate.

### Forcing

Each box is scaled to the regional pattern in `RF_REGIONS_NORMYEAR`, $t_n$,
and the result harmonised to the target `RF_CLOUD_ALBEDO_AER_WM2` in `RF_CLOUD_ALBEDO_AER_YR`:

$$
D_i(t) = \frac{\Delta CDNC_i(t)}{\Delta CDNC_i(t_n)} \, \text{RF\_REGIONS\_CLOUD\_ALBEDO}_i
$$

$$
RF_i(t) = D_i(t) \frac{\text{WM2}}{\sum_j a_j D_j(\text{YR})}
$$

The cloud cover effect is the same with `RF_REGIONS_CLOUD_COVER` and `RF_CLOUD_COVER_AER_*`.
The total cloud forcing is the sum of the two.

## Inputs and outputs

| Input                                                  | Use                                       |
| ------------------------------------------------------ | ----------------------------------------- |
| `FILE_{SOXI,SOXNB,OCI,OCB,OCN,BCI,BCB,SS}_OT`          | Historical four-box optical thickness     |
| Total nitrate forcing from [direct aerosol forcing](module_05_aerosol_direct.md) | Nitrate proxy                |
| SOx, OC and BC emissions by sector, industrial NOx     | Extension after the data end              |

| Output                                                 | Description                               |
| ------------------------------------------------------ | ----------------------------------------- |
| `DAT_CLOUD_ALBEDO_RF`, `_EFFRF`, `_ERF`                | Cloud albedo forcing                      |
| `DAT_CLOUD_COVER_RF`, `_EFFRF`, `_ERF`                 | Cloud cover forcing                       |
| `DAT_CLOUD_TOT_RF`, `_EFFRF`, `_ERF`, `_SRF`           | Total cloud forcing                       |

## Parameters

| Parameter                       | Default                              | Description                                    |
| ------------------------------- | ------------------------------------ | ---------------------------------------------- |
| `CLOUD_WEIGHT_SOX`              | 0.2653                               | SOx weight                                     |
| `CLOUD_WEIGHT_OC`               | 0.2653                               | OC weight                                      |
| `CLOUD_WEIGHT_SS`               | 0.2653                               | Sea salt weight                                |
| `CLOUD_WEIGHT_NO3`              | 0.1633                               | Nitrate weight                                 |
| `CLOUD_WEIGHT_BC`               | 0.0408                               | BC weight                                      |
| `CLOUD_BCI2BCB_SOLUBLE_RATIO`   | 0.75                                 | Weight of industrial BC relative to biomass BC |
| `SOX_NATURAL_EMISSIONS`         | 14.0                                 | Natural SOx emissions (Mt S/yr)                |
| `RF_CLOUD_ALBEDO_AER_APPLY`     | 1                                    | Harmonises to the target                       |
| `RF_CLOUD_ALBEDO_AER_YR`        | 2019                                 | Harmonisation year                             |
| `RF_CLOUD_ALBEDO_AER_WM2`       | -0.89                                | Target albedo forcing (W/m²)                   |
| `RF_CLOUD_COVER_AER_WM2`        | 0.0                                  | Target cover forcing (W/m²)                    |
| `RF_REGIONS_CLOUD_ALBEDO`       | -0.96563, -1.39945, -0.34211, -0.62846 | Albedo pattern (W/m²)                        |
| `RF_REGIONS_CLOUD_COVER`        | -1.33278, -1.5806, -0.52942, -0.81127  | Cover pattern (W/m²)                         |
| `RF_REGIONS_NORMYEAR`           | 2000                                 | Year of the regional pattern                   |
| `CLOUD_APPLY_LIMIT_MAX`         | 0                                    | Caps box forcing at `CLOUD_LIMIT_MAX`          |
| `RF_EFFICACY_CLOUD_ALBEDO`      | 1.0                                  | Efficacy                                       |
| `SRF_FACTOR_CLOUD_ALBEDO`       | 0.8                                  | Surface to top-of-atmosphere forcing ratio     |
| `SRF_FACTOR_CLOUD_COVER`        | 0.9                                  | Surface to top-of-atmosphere forcing ratio     |

MAGICC7 renormalises the weights to sum to 1, and the defaults above are those normalised values.
With `RF_CLOUD_ALBEDO_AER_APPLY = 1`, `RF_CLOUD_ALBEDO_AER_FACTOR` is an output.
The patterns come from Hansen et al. (2005).

## Algorithm

`cloud_calc_ind_aerosol` in `cloudstore.f90`:

1. Builds the normalised indices of NO3, OC, BC, SOX and sea salt in turn, as in the table above.
2. Renormalises the species weights.
3. Calculates the total aerosol number, its preindustrial value and the change in cloud droplet number.
4. Scales the change to the albedo and cover patterns.
5. Harmonises each effect in `CLOUD_APPLY_SCALING`, takes the global value as the area-weighted mean,
   applies the first-year offset (see [forcing aggregation](module_07b_rf_aggregation.md#pre-industrial-reference))
   and sets the first year to zero.
6. Caps the box forcing at `CLOUD_LIMIT_MAX` when `CLOUD_APPLY_LIMIT_MAX` is 1.
7. Holds the forcing after `RF_AER_CONSTANTAFTERYR`.
8. Sums the albedo and cover forcing into the total.

## Numerical considerations

RSCM uses `math.log10` to match MAGICC7's rounding.

## Issues in MAGICC7

RSCM reproduces these unless the entry says otherwise.

- **Normalisation years differ by species**: each group is normalised in the last year of its own data.
  NO3 uses the end of the historical NOx emissions, 2014 by default in `cloud_albedo`.
- **Forward emission sums**: future indices follow each year's emissions plus the next year's.
  The final year repeats the one before.
  MAGICC7 adds natural SOx to SOXNB in a different order in the two sums, which RSCM copies.
- **Log of the number index only**: the Gultepe and Isaac parameterisation is commented out
  because the normalisations cancel its coefficients, so only $\log_{10} N_A$ remains.
- **Zero guards**: MAGICC7 normalises a group by 1 when its normalisation is zero,
  and gives a box zero forcing when its $\Delta CDNC$ is zero in `RF_REGIONS_NORMYEAR`.
  RSCM does not, as below.

## Differences from MAGICC7

- **Zero guards**: RSCM has neither guard, which the default inputs do not reach.
- **Not in RSCM**: the cloud cover effect (zero by default), the forcing cap `CLOUD_APPLY_LIMIT_MAX`,
  `RF_AER_CONSTANTAFTERYR`, surface forcing and the sector mode.
  In the sector mode MAGICC7 runs the calculation twice and keeps the normalisation of the first run.
- **Rust component**: `AerosolIndirect` (`crates/rscm-magicc/src/forcing/aerosol_indirect.rs`)
  is a simplified, global version with no MAGICC7 parity test, and `run_magicc7` does not use it.
  It uses $\ln(1 + B/B_0)$ of SOx and OC emissions, with no regional pattern or optical thickness.

## References

MAGICC7 source files:

| Function or section             | File                                          | Lines     |
| ------------------------------- | --------------------------------------------- | --------- |
| `cloud_init`                    | `src/libmagicc/physics/cloudstore.f90`        | 44-65     |
| `cloud_calc_ind_aerosol`        | `src/libmagicc/physics/cloudstore.f90`        | 92-828    |
| `CLOUD_APPLY_SCALING`           | `src/libmagicc/physics/cloudstore.f90`        | 830-916   |
| Parameter declarations          | `src/libmagicc/physics/radiative_forcing.f90` | 1-118     |
| Sector handling (`MOD_SECTOR`)  | `src/libmagicc/core.f90`                      | 44-85     |
| Namelist                        | `src/libmagicc/allcfgs.f90`                   | 190-239   |
| Call site                       | `src/libmagicc/MAGICC7.f90`                   | 1746      |
| ERF and EffRF                   | `src/libmagicc/MAGICC7.f90`                   | 5776-5917 |
| Surface forcing                 | `src/libmagicc/MAGICC7.f90`                   | 6196-6209 |

- Hansen, J. et al. (2005). Efficacy of climate forcings. *Journal of Geophysical Research*, 110, D18104.
