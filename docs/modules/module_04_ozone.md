# Module 04: Ozone Radiative Forcing

## Overview

This module calculates the forcing from changes in ozone.
MAGICC7 does not model ozone concentrations.
It parameterises the forcing from halocarbon loading, precursor emissions, N2O and temperature,
in four independent terms that are summed:

- **Stratospheric ozone** (`STRATOZ`): depletion by chlorine and bromine, measured by EESC.
  The forcing is negative.
- **Tropospheric ozone** (`TROPOZ`): production from CH4, NOx, CO and NMVOC.
  The forcing is positive.
- **Ozone due to N2O** (`OZDUETON2O`): the ozone change caused by N2O through stratospheric chemistry.
- **Ozone due to temperature** (`OZDUETOTEMPERATURE`): the response of ozone to warming, a negative feedback.

None of the terms depends on another, although the real processes interact.

## RSCM implementation

- **Component**: `OzoneForcing` (Python builder `rscm.magicc.OzoneForcingBuilder`).
- **Source**: `crates/rscm-magicc/src/forcing/ozone.rs`,
  with parameters in `crates/rscm-magicc/src/parameters/ozone_forcing.rs`.
- **Status**: Implemented. See the [status table](README.md#rscm-implementation-status).
- **Tests**: `test_ozone` in `tests/regression/test_full_run.py` drives the component with MAGICC7's EESC,
  concentrations, emissions and temperature, and checks the stratospheric, four-box tropospheric,
  N2O and temperature terms.
  `test_coupled_run` checks them through the total.
  `ozone.rs` has unit tests.

`OzoneForcing` outputs each term as `Effective Radiative Forcing|<term>`,
which is MAGICC7's RF before efficacies (see [forcing types](module_07b_rf_aggregation.md#forcing-types)).
`ForcingAggregation` applies the efficacies.
`rscm.magicc.full.ozone_parameters` maps MAGICC7's configuration onto it.

## Equations

### Stratospheric ozone

$$
RF_{strat} = s \left( \max\left(0, \frac{EESC_Y - EESC_{Y_0}}{100}\right) \right)^{\beta}
$$

- $EESC_Y$ is the year's EESC from [module 03](module_03_halocarbon_chemistry.md#eesc) (ppt).
- $Y_0$ = `STRATOZ_THRESHOLD_YEAR`.
  The forcing is zero up to $Y_0$.
- $s$ = `STRATOZ_O3SCALE` is negative, because depletion cools.
- $\beta$ = `STRATOZ_CLEXPON`.

The power is libm `pow`.
The `max` stops EESC below its threshold-year value from producing positive forcing.

### Tropospheric ozone

The forcing combines a logarithmic CH4 term with linear terms in the precursor emissions.
MAGICC7 evaluates it in this order:

$$
\begin{aligned}
C &= (r \, o_{CH4}) \ln\left(\frac{M_Y}{M_0}\right) \\
\Delta X &= (X^{fossil}_{Y-1} - X^{fossil}_0) + (X^{AFOLU}_{Y-1} - X^{AFOLU}_0) \\
S &= (o_{NOx} \Delta NOx + o_{CO} \Delta CO) + o_{VOC} \Delta NMVOC \\
RF_{trop} &= C + S \, r
\end{aligned}
$$

- $r$ = `TROPOZ_RADEFF_WM2PERDU` converts Dobson units to W/m².
- $o_{CH4}$, $o_{NOx}$, $o_{CO}$ and $o_{VOC}$ are
  `TROPOZ_OZCH4`, `TROPOZ_OZNOX`, `TROPOZ_OZCO` and `TROPOZ_OZVOC`.
- $M_Y$ is the year's CH4 concentration (ppb).
- $\Delta X$ is the change in a precursor's emissions, summed over the industrial and biomass burning sectors.
  The emissions are the previous year's.
- The subscript 0 is the first year of the run, MAGICC7's preindustrial reference year (1750).

MAGICC7 spreads $RF_{trop}$ over the [boxes](module_00_initialization.md) with
`RF_REGIONS_TROPOZ` ($w$) normalised within each hemisphere $H$:

$$
RF_{trop,i} = RF_{trop} \frac{w_i}{m_H},
\qquad
m_H = \frac{a_L w_L}{a_O + a_L} + \frac{a_O w_O}{a_O + a_L}
$$

with $a$ the box areas, summing the land term first.
The area mean of each hemisphere is then the global value.
MAGICC7 reports the global value as the area mean of the boxes.

### Ozone due to N2O

$$
RF_{N2O} = k (N_Y - N_0)
$$

with $N$ the N2O concentration (ppb) and $k$ = `OZDUETON2O_RADEFF`.

### Ozone due to temperature

$$
RF_{T} = \gamma \, T_{Y-1}
$$

with $T$ the global mean surface temperature anomaly (K) and $\gamma$ = `OZDUETOTEMPERATURE_SCALE`.
The temperature is the previous year's, because the year's own temperature is not yet known.

### Total

$$
RF_{O3} = RF_{strat} + RF_{trop} + RF_{N2O} + RF_{T}
$$

MAGICC7 reports this as `DAT_OZTOTAL_*`.
RSCM has no total ozone output.
`ForcingAggregation` adds the four terms as one subtotal of the total forcing, in MAGICC7's order.

## Inputs and outputs

| Component     | Input                        | Units                | Fortran variable                                   |
| ------------- | ---------------------------- | -------------------- | -------------------------------------------------- |
| Stratospheric | EESC                         | ppt                  | `DAT_EESC_CONC`                                    |
| Tropospheric  | CH4 concentration            | ppb                  | `DAT_CH4_CONC`                                     |
| Tropospheric  | NOx, CO and NMVOC emissions  | Mt/yr                | `DAT_NOXI_EMIS`, `DAT_NOXB_EMIS`, and so on        |
| N2O           | N2O concentration            | ppb                  | `DAT_N2O_CONC`                                     |
| Temperature   | Surface temperature anomaly  | K                    | `DAT_SURFACE_TEMP`                                 |

Each component writes RF, EffRF and ERF datastores:

| Component     | RF                          | EffRF                          | ERF                          |
| ------------- | --------------------------- | ------------------------------ | ---------------------------- |
| Stratospheric | `DAT_STRATOZ_RF`            | `DAT_STRATOZ_EFFRF`            | `DAT_STRATOZ_ERF`            |
| Tropospheric  | `DAT_TROPOZ_RF`             | `DAT_TROPOZ_EFFRF`             | `DAT_TROPOZ_ERF`             |
| N2O           | `DAT_OZDUETON2O_RF`         | `DAT_OZDUETON2O_EFFRF`         | `DAT_OZDUETON2O_ERF`         |
| Temperature   | `DAT_OZDUETOTEMPERATURE_RF` | `DAT_OZDUETOTEMPERATURE_EFFRF` | `DAT_OZDUETOTEMPERATURE_ERF` |
| Total         | `DAT_OZTOTAL_RF`            | `DAT_OZTOTAL_EFFRF`            | `DAT_OZTOTAL_ERF`            |

`DAT_TROPOZ_CH4_RF` also holds the CH4 term of tropospheric ozone as a diagnostic.

## Parameters

| Component     | Parameter                     | Default   | Units              | Description                                    |
| ------------- | ----------------------------- | --------- | ------------------ | ---------------------------------------------- |
| Stratospheric | `STRATOZ_O3SCALE`             | -0.0043   | W/m²               | $s$                                            |
| Stratospheric | `STRATOZ_CLEXPON`             | 1.7       | 1                  | $\beta$                                        |
| Stratospheric | `STRATOZ_THRESHOLD_YEAR`      | 1979      | year               | $Y_0$                                          |
| Tropospheric  | `TROPOZ_RADEFF_WM2PERDU`      | 0.032     | W/m²/DU            | $r$                                            |
| Tropospheric  | `TROPOZ_OZCH4`                | 5.7       | DU                 | $o_{CH4}$, per unit of $\ln(M_Y / M_0)$        |
| Tropospheric  | `TROPOZ_OZNOX`                | 0.168     | DU per Mt N/yr     | $o_{NOx}$                                      |
| Tropospheric  | `TROPOZ_OZCO`                 | 0.00396   | DU per Mt CO/yr    | $o_{CO}$                                       |
| Tropospheric  | `TROPOZ_OZVOC`                | 0.01008   | DU per Mt NMVOC/yr | $o_{VOC}$                                      |
| Tropospheric  | `AIR_TROPOZ_NOXAIR_SCALE`     | 1.5       | 1                  | Effect of aviation NOx relative to other NOx   |
| Tropospheric  | `AIR_TROPOZ_REFYR`            | 2019      | year               | Calibration year of the aviation NOx ratio     |
| N2O           | `OZDUETON2O_RADEFF`           | 0.0004827 | W/m²/ppb           | $k$                                            |
| Temperature   | `OZDUETOTEMPERATURE_SCALE`    | -0.037    | W/m²/K             | $\gamma$                                       |

The aviation parameters apply only when `SECTOR_INCLUDE` is not `NOSECTOR`.
`STRATOZ_BR_VS_CL_SCALE` belongs to the EESC calculation
in [module 03](module_03_halocarbon_chemistry.md#eesc).

Each term also has a regional pattern, an efficacy and a surface forcing factor:

| Component     | `RF_REGIONS_*` (NO, NL, SO, SL)        | `RF_EFFICACY_*` | `SRF_FACTOR_*` |
| ------------- | -------------------------------------- | --------------- | -------------- |
| Stratospheric | -0.01189, -0.02267, -0.06251, -0.24036 | 1.0             | -0.5           |
| Tropospheric  | 0.46565, 0.51646, 0.17687, 0.23793     | 1.0             | 0.6            |
| N2O           | 1.0, 1.0, 1.0, 1.0                     | 1.0             |                |
| Temperature   | 1.0, 1.0, 1.0, 1.0                     | 1.0             |                |

The stratospheric pattern is weighted to the Southern Hemisphere and the tropospheric one to the Northern.
[Module 07b](module_07b_rf_aggregation.md) describes how the patterns and efficacies are applied.

### Reference values

These follow from the defaults and are quick checks of an implementation:

| Case                                    | Forcing      |
| --------------------------------------- | ------------ |
| CH4 doubled, no precursor change        | 0.1264 W/m²  |
| N2O 60 ppb above its first-year value   | 0.0290 W/m²  |
| 2 K of warming                          | -0.074 W/m²  |

## Algorithm

The terms are calculated in separate places in the `MAGICC7.f90` time loop:

1. After the halocarbon loop, the stratospheric term from the next year's EESC.
2. Straight after, the N2O term from the next year's N2O
   and the temperature term from the current year's temperature.
   In the first year both are also set for the first year itself.
3. Later, with the other forcing agents, the tropospheric term and its box split.
   In the first year `set_rf_first_year_forcing_and_offset` sets its offset,
   which is subtracted every year
   (see [pre-industrial reference](module_07b_rf_aggregation.md#pre-industrial-reference)).
4. Each term is held constant after `RF_*_CONSTANTAFTERYR` and `RF_TOTAL_CONSTANTAFTERYR`.
5. The total sums the four terms for each of RF, EffRF and ERF.

RSCM calculates all four terms in one `solve`, from the same inputs and with the same timing.

## Issues in MAGICC7

RSCM reproduces each of these.

- **Lagged temperature**: the temperature term uses the previous year's temperature,
  so it responds a year late.
- **Lagged precursors**: the tropospheric term uses the previous year's emissions but the year's own CH4.
- **First-year offset**: a TODO in the tropospheric code says
  `check bug - offset not actually pre-industrial or current_year_idx`.
- **Validity of the stratospheric fit**: a comment in `calculate_stratospheric_ozone_rf` warns that
  the parameterisation "is not valid for very high chlorine or bromine loading".
- **Fixed constant**: the 100 that scales the EESC excess is hardcoded.
- **Signs**: `STRATOZ_O3SCALE`, `OZDUETOTEMPERATURE_SCALE` and `RF_REGIONS_STRATOZ` are negative,
  and `RF_REGIONS_TROPOZ` is positive.

## Differences from MAGICC7

- **Aviation NOx**: with `SECTOR_INCLUDE` other than `NOSECTOR`,
  MAGICC7 weights aviation NOx by `AIR_TROPOZ_NOXAIR_SCALE`, and logs that ".SECTOR file use is untested".
  RSCM implements only the `NOSECTOR` path, so aviation NOx is not separated from other NOx.
- **Holds**: RSCM has no `RF_STRATOZ_CONSTANTAFTERYR` or `RF_TROPOZ_CONSTANTAFTERYR`.
  The reference run sets both to 2105, its last year, so they have no effect there.

## References

MAGICC7 source, relative to the MAGICC repository:

| Function or section                                   | File                                            | Lines     |
| ----------------------------------------------------- | ----------------------------------------------- | --------- |
| `MOD_OZONE` parameters                                | `src/libmagicc/core.f90`                        | 87-98     |
| Ozone datastores                                      | `src/libmagicc/utils/datastore.f90`             | 212-246   |
| Regional, efficacy and surface factor declarations    | `src/libmagicc/physics/radiative_forcing.f90`   | 4-48      |
| `calculate_stratospheric_ozone_rf`                    | `src/libmagicc/physics/deltaq_calculations.f90` | 954-1002  |
| `calculate_ozone_due_to_n2o_rf`                       | `src/libmagicc/physics/deltaq_calculations.f90` | 1004-1016 |
| `calculate_ozone_due_to_temperature_rf`               | `src/libmagicc/physics/deltaq_calculations.f90` | 1018-1029 |
| `calculate_tropospheric_ozone_rf_due_to_methane_alone` | `src/libmagicc/physics/deltaq_calculations.f90` | 1337-1354 |
| `calculate_tropospheric_ozone_rf`                     | `src/libmagicc/physics/deltaq_calculations.f90` | 1356-1427 |
| Aviation NOx ratio                                    | `src/libmagicc/MAGICC7.f90`                     | 1782-1798 |
| Stratospheric term                                    | `src/libmagicc/MAGICC7.f90`                     | 4739-4760 |
| N2O and temperature terms                             | `src/libmagicc/MAGICC7.f90`                     | 4762-4816 |
| Tropospheric term, including sectors                  | `src/libmagicc/MAGICC7.f90`                     | 5448-5553 |
| Efficacies of the N2O and temperature terms           | `src/libmagicc/MAGICC7.f90`                     | 5718-5724 |
| Total ozone                                           | `src/libmagicc/MAGICC7.f90`                     | 5999-6013 |
| Efficacy, regional and ozone defaults                 | `run/MAGCFG_DEFAULTALL.CFG`                     | 520-523, 549-552, 655-671 |
