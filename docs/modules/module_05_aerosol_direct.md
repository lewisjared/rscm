# Aerosol Direct Forcing

## Overview

This module calculates the direct forcing of aerosols,
which scatter sunlight (cooling) or absorb it (warming).
It covers:

- sulfate (SOx), black carbon (BC) and organic carbon (OC), each from industrial and biomass burning sources,
- nitrate (NO3), formed from NOx and NH3 emissions,
- mineral dust, which follows land-use change,
- black carbon on snow, which darkens snow and ice.

Aerosol forcing depends only on prescribed inputs, so MAGICC7 calculates it once before the run.
Most species start from a historical four-box forcing file,
extend it with emissions after the file ends,
and harmonise the result to a target forcing in a reference year.
Nitrate instead follows the Hauglustaine et al. (2014) parameterisation from emissions.

The page uses RF, EffRF and ERF as the [forcing aggregation page defines them](module_07b_rf_aggregation.md#forcing-types).
Four-box arrays follow the box order in [initialisation](module_00_initialisation.md).

## RSCM implementation

- **Functions**: `direct_aerosol`, `nitrate`, `mineral_dust` and `bc_on_snow` in `rscm.magicc.prescribed`,
  called by `rscm.magicc.full.prescribed_forcing`.
- **Source**: `python/rscm/magicc/prescribed.py`.
- **Status**: Implemented. See the [status table](README.md#rscm-implementation-status).
- **Tests**: `test_prescribed_forcing` in `tests/regression/test_full_run.py`,
  parametrised over SOx, BC, OC and nitrate for each sector, mineral dust and BC on snow.

`ForcingAggregation` takes the results as four-box agents (see [forcing aggregation](module_07b_rf_aggregation.md)).

## Equations

### Harmonisation

Each species' forcing is its raw forcing times a factor set by its `_APPLY` switch:

| `_APPLY` | Factor                                                                  |
| -------- | ----------------------------------------------------------------------- |
| 0        | 1                                                                       |
| 1        | $\text{WM2} / (RF_{raw}(\text{YR}) - RF_{raw}(t_0))$                     |
| 2        | `_FACTOR`, as given                                                     |
| 3        | The industrial species' factor (biomass species only)                   |

$t_0$ is the first year, the preindustrial reference `RF_PREIND_REFERENCEYR` (1750).
The global value is the area-weighted mean of the boxes.
With `_APPLY = 1`, `_FACTOR` is an output.

### Extension with emissions

After the last year $t_s$ of the forcing file, each box follows emissions:

$$
RF_{box}(t) = RF_{box}(t_s) \frac{\tilde{E}(t)}{\tilde{E}(t_s)}
$$

where $\tilde{E}(t)$ is the mean of the emissions above the first year's in years $t$ and $t + 1$.
BC on snow uses total BC emissions without subtracting the first year.

### Nitrate

Nitrate competes with sulfate for ammonia.
With NOx and NH3 emissions in Mt N/yr and SOx in Mt S/yr, each including its natural source,
the molar amounts $N = E_{NOx} / 14$, $A = E_{NH3} / 14$ and $S = E_{SOx} / 32.07$ give a burden proxy

$$
\alpha(t) = \frac{N A}{1 + \lambda S / A}
$$

with $\lambda$ = `RF_NO3_LAMBDASO2`.
The burden, optical thickness and raw forcing follow from the preindustrial values:

$$
\begin{aligned}
B(t) &= B_{PI} \frac{\alpha(t)}{\alpha(t_0)} \\
OT(t) &= OT_{PI} \left(\frac{B(t)}{B_{PI}}\right)^{k} \\
RF(t) &= \text{WM2} \, \frac{OT(t) - OT_{PI}}{OT(\text{YR}) - OT_{PI}}
\end{aligned}
$$

with $B_{PI}$ = `RF_NO3_PREINDBURDEN`, $OT_{PI}$ = `RF_NO3_PREINDOT` and $k$ = `RF_NO3_BURDEN2OT_EXP`.
Each hemisphere $h$ takes its share of NOx times NH3:

$$
RF_h(t) = 2 RF(t) \frac{N_h A_h}{\sum_{h'} N_{h'} A_{h'}}
$$

Within a hemisphere with ocean and land fractions $f_o$ and $f_l$,
the pattern $w$ = `RF_REGIONS_NO3` spreads $RF_h$ over the boxes:

$$
RF_{o} = RF_h \frac{(f_o + f_l) \, w_o}{f_o w_o + f_l w_l},
\qquad RF_{l} = RF_h \frac{(f_o + f_l) \, w_l}{f_o w_o + f_l w_l}
$$

Global emissions fall half on each hemisphere's land, as MAGICC7 spreads global inputs,
so both hemispheres carry the global forcing.
The industrial and biomass shares follow each source's NOx times NH3 product.

### Mineral dust and BC on snow

Mineral dust follows the global land-use albedo forcing.
The land-use file is extended with cumulative biosphere CO2 emissions,
harmonised to `RF_MINERALDUST_DIR_WM2`, spread over the boxes with `RF_REGIONS_DUST`,
and held constant after `RF_MINERALDUST_CONSTANTAFTERYR`.

BC on snow is its file extended with total BC emissions, harmonised to `RF_BCSNOW_ALBEDO_WM2`,
and taken relative to its first year.

### Total direct aerosol forcing

$$
RF_{TOTAER\_DIR} = RF_{BCI} + RF_{OCI} + RF_{SOXI} + RF_{NO3I} + RF_{BIOMASSAER} + RF_{DUST}
$$

where `DAT_BIOMASSAER_RF` sums the biomass SOx, nitrate, OC and BC forcing.
The order matters for parity, so [forcing aggregation](module_07b_rf_aggregation.md) gives the exact summation order.

## Inputs and outputs

| Input                                  | Unit     | Use                                         |
| -------------------------------------- | -------- | ------------------------------------------- |
| `FILE_{SOXI,SOXB,BCI,BCB,OCI,OCB}_RF`  | W/m²     | Historical four-box forcing                 |
| `FILE_BCSNOW_RF`                       | W/m²     | Historical BC on snow forcing               |
| `FILE_MINERALDUST_RF`                  | W/m²     | Land-use forcing, `GISS_LANDUSE_RF.IN`      |
| SOx, BC and OC emissions by sector     | Mt/yr    | Extension after the files end               |
| NOx and NH3 emissions by sector        | Mt N/yr  | Nitrate                                     |
| Biosphere CO2 emissions                | GtC/yr   | Mineral dust, cumulatively                  |

| Output                                          | Description                                |
| ----------------------------------------------- | ------------------------------------------ |
| `DAT_{SOXI,SOXB,BCI,BCB,OCI,OCB}_RF`            | Forcing by species and source              |
| `DAT_NO3I_RF`, `DAT_NO3B_RF`, `DAT_NO3T_RF`     | Nitrate forcing by sector and in total     |
| `DAT_MINERALDUST_RF`, `DAT_BCSNOW_RF`           | Mineral dust and BC on snow forcing        |
| `DAT_BIOMASSAER_RF`                             | Biomass aerosol forcing                    |
| `DAT_TOTAER_DIR_RF`, `_EFFRF`, `_ERF`, `_SRF`   | Total direct aerosol forcing               |

## Parameters

Harmonisation of each species (MAGICC7 v7.5.3 defaults):

| Species        | `_APPLY` | `_YR` | `_WM2` (W/m²) | Parameter prefix         |
| -------------- | -------- | ----- | ------------- | ------------------------ |
| BC industrial  | 1        | 2019  | 0.155         | `RF_BCI_DIR_`            |
| OC industrial  | 1        | 2019  | -0.1214       | `RF_OCI_DIR_`            |
| SOx industrial | 1        | 2019  | -0.1774       | `RF_SOXI_DIR_`           |
| BC biomass     | 3        | 2019  | output        | `RF_BCB_DIR_`            |
| OC biomass     | 3        | 2019  | output        | `RF_OCB_DIR_`            |
| SOx biomass    | 3        | 2019  | output        | `RF_SOXB_DIR_`           |
| Nitrate        | 1        | 2019  | -0.0444       | `RF_NO3T_DIR_`           |
| Mineral dust   | 1        | 2019  | -0.015        | `RF_MINERALDUST_DIR_`    |
| BC on snow     | 1        | 2019  | 0.08          | `RF_BCSNOW_ALBEDO_`      |

Other parameters:

| Parameter                         | Default | Unit   | Description                                  |
| --------------------------------- | ------- | ------ | -------------------------------------------- |
| `RF_NOXN_EMIS`                    | 10.0    | Mt N/yr | Natural NOx emissions                       |
| `RF_NH3N_EMIS`                    | 21.0    | Mt N/yr | Natural NH3 emissions                       |
| `SOX_NATURAL_EMISSIONS`           | 14.0    | Mt S/yr | Natural SOx emissions                       |
| `RF_NO3_LAMBDASO2`                | 1.6     | 1      | Priority of sulfate for ammonia              |
| `RF_NO3_PREINDBURDEN`             | 0.0085  | Tg N   | Preindustrial nitrate burden                 |
| `RF_NO3_PREINDOT`                 | 1.41    | 1      | Preindustrial nitrate optical thickness      |
| `RF_NO3_BURDEN2OT_EXP`            | 0.76    | 1      | Burden to optical thickness exponent         |
| `RF_MINERALDUST_CONSTANTAFTERYR`  | 2018    | year   | Last year of changing mineral dust forcing   |
| `RF_PREIND_REFERENCEYR`           | 1750    | year   | Preindustrial reference year                 |

The regional patterns `RF_REGIONS_SOX`, `RF_REGIONS_BC`, `RF_REGIONS_OC`, `RF_REGIONS_NO3`,
`RF_REGIONS_DUST` and `RF_REGIONS_AER_DIR`, and the efficacies `RF_EFFICACY_*`,
are listed with the other agents in [forcing aggregation](module_07b_rf_aggregation.md).
Biomass aerosols and mineral dust use the direct aerosol efficacy `RF_EFFICACY_AER_DIR`.

## Algorithm

`JUMPONSTAGE` in `MAGICC7.f90` processes BC, OC, nitrate, SOx and mineral dust in turn.
For BC, OC and SOx it:

1. Reads the industrial and biomass forcing files.
2. Extends them with emissions in `EXTRAP_RF_WITH_EMIS`
   (calling `extend_rf_box_from_switchyear_with_emissions` in `datastore.f90`).
3. Takes the global value as the area-weighted mean of the boxes.
4. Applies the harmonisation factor in `RF_APPLY_SCALING`, with biomass reusing the industrial factor.
5. Subtracts the first-year offset set by `RF_INITIALIZATION_METHOD`,
   which is zero with the default `JUMPSTART`
   (see [forcing aggregation](module_07b_rf_aggregation.md#pre-industrial-reference)).

Nitrate is then calculated from emissions, split by hemisphere and source,
and its regional pattern `RF_REGIONS_NO3` replaced by the total box forcing over the hemispheric forcing
in `RF_REGIONS_NORMYEAR`.
Finally the module sums the total direct aerosol forcing.

## Numerical considerations

RSCM uses `math.pow` for the nitrate optical thickness to match MAGICC7's rounding.

## Issues in MAGICC7

RSCM reproduces each of these.

- **Harmonisation baseline**: `_APPLY = 1` subtracts the preindustrial forcing `PREIND_DATGLOBE`
  in the denominator, so the target is the change since 1750, the first year of the run.
- **No unscaled sulfate or nitrate**: MAGICC7 resets `RF_SOXI_DIR_APPLY`, `RF_SOXB_DIR_APPLY`
  and `RF_NO3T_DIR_APPLY` from 0 to 1 with an error message.
- **Biomass extension**: v7.5.3 extends each biomass box additively, by half the change in midyear emissions
  above the first year's, rather than by the ratio used for industrial sources.
  The change is negated where the box's last file value and midyear emissions differ in sign.
- **Nitrate starts at zero**: the nitrate setup fixes the first-year forcing at zero,
  so `RF_INITIALIZATION_METHOD` has no effect on it (a TODO in the Fortran notes this).

## Differences from MAGICC7

- **Constant-after years**: RSCM applies `RF_MINERALDUST_CONSTANTAFTERYR`,
  but not `RF_AER_CONSTANTAFTERYR` or `RF_TOTAL_CONSTANTAFTERYR`.
- **Surface forcing**: RSCM does not calculate `DAT_TOTAER_DIR_SRF`.
  MAGICC7 leaves BC on snow out of it.
- **Rust component**: `AerosolDirect` (`crates/rscm-magicc/src/forcing/aerosol_direct.rs`)
  is a simplified version with no MAGICC7 parity test, and `run_magicc7` does not use it.
  It takes global emissions, combines sources and scales forcing linearly with emissions.
  It spreads the forcing over the four boxes with species-weighted regional patterns.
  It has no dust, BC on snow or Hauglustaine nitrate.

## References

MAGICC7 source files:

| Function or section                              | File                                  | Lines       |
| ------------------------------------------------ | ------------------------------------- | ----------- |
| Aerosol set-up in `JUMPONSTAGE`                  | `src/libmagicc/MAGICC7.f90`           | 1850-2500   |
| BC, OC                                           | `src/libmagicc/MAGICC7.f90`           | 1851-1904   |
| Nitrate                                          | `src/libmagicc/MAGICC7.f90`           | 1906-2180   |
| SOx                                              | `src/libmagicc/MAGICC7.f90`           | 2186-2220   |
| Mineral dust                                     | `src/libmagicc/MAGICC7.f90`           | 2381-2410   |
| Total direct aerosol                             | `src/libmagicc/MAGICC7.f90`           | 2460-2485   |
| `RF_APPLY_SCALING`                               | `src/libmagicc/MAGICC7.f90`           | 9353-9396   |
| `EXTRAP_RF_WITH_EMIS`                            | `src/libmagicc/MAGICC7.f90`           | 9441-9579   |
| `extend_rf_box_from_switchyear_with_emissions`   | `src/libmagicc/utils/datastore.f90`   | 572-688     |
| Parameters and patterns                          | `src/libmagicc/physics/radiative_forcing.f90` |     |
| Namelist                                         | `src/libmagicc/allcfgs.f90`           |             |
| Defaults                                         | `run/MAGCFG_DEFAULTALL.CFG`           |             |

- Hauglustaine, D. A. et al. (2014). A global model simulation of present and future nitrate aerosols
  and their direct radiative forcing of climate. *Atmospheric Chemistry and Physics*.
