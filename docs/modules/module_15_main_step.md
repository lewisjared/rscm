# Main Step

## Overview

The main step runs every physics module within MAGICC7's annual loop.
It decides the order of the carbon cycle, chemistry, forcing, climate and sea level calculations.

Two subroutines do the work:

- `magicc_run` (MAGICC7.f90 lines 2635 to 2643) loops over the years.
- `magicc_step_year` (lines 2646 to 3727) runs one year, in about 1100 lines.

The order makes concentrations come before forcing and forcing before temperature.
Temperature feeds back on the carbon cycle and chemistry with a lag of one year.
The climate module runs monthly substeps inside the year.

## RSCM implementation

- **Component**: `Model`, built with `rscm.core.ModelBuilder`.
  `rscm.magicc.full.run_magicc7` assembles the MAGICC7 chain with MAGICC7's defaults.
- **Source**: `python/rscm/magicc/full.py` for the chain,
  and `crates/rscm-core/src/model/` for the model.
- **Status**: Implemented. See the [status table](README.md#rscm-implementation-status).
- **Tests**: `tests/regression/test_full_run.py::test_run_magicc7_defaults`
  checks temperature, every concentration and total ERF against MAGICC7 v7.5.3.
  `test_run_magicc7_rejects_bad_emissions` checks the emissions validation.
  `test_coupled_run` covers the chain with permafrost and the parameter variants.
  [MAGICC runs](../magicc_runs.md#full-magicc7-chain) shows how to use it.

RSCM has no `magicc_step_year`.
Each MAGICC7 module is a component, and the `Model` solves them in dependency order every annual step.
Components added producer first are coupled within the step.
Feedbacks such as temperature on the carbon cycle read the value from the start of the step,
through a lag, as in MAGICC7.
[Key concepts](../key_concepts.md#dependency-graph) describes the dependency graph and step semantics.
The monthly loops stay inside their components:
`ClimateUDEB` runs the climate substeps and `OceanCarbon` the ocean carbon months.

## Inputs and outputs

The modules run in the order of their physical dependencies.
Inputs and outputs use MAGICC7's Fortran names.

| Order | Module                                                    | Inputs                                                           | Outputs                                              | RSCM component                                     |
| ----- | --------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------- |
| 1     | [Permafrost](module_12_permafrost.md)                     | `DAT_SURFACE_TEMP` (previous year)                               | `DAT_CH4PF_EMIS`, `DAT_CO2PF_EMIS`                   | `Permafrost`                                       |
| 2     | [CH4 Chemistry](module_01_ch4_chemistry.md)               | CH4 emissions, temperature, NOx, CO and NMVOC                    | CH4 concentration and lifetime                       | `CH4Chemistry`, `CH4Oxidation`                     |
| 3     | [N2O Chemistry](module_02_n2o_chemistry.md)               | N2O emissions                                                    | N2O concentration                                    | `N2OChemistry`                                     |
| 4     | [Terrestrial Carbon](module_09_terrestrial_carbon.md)     | `FEEDBACK_TEMPERATURE`, `DAT_CO2_CONC`, `DAT_CO2B_EMIS`          | `CO2_TERRBIO_AND_FOSSIL_EMIS`, the three pools       | `TerrestrialCarbon`                                |
| 4a    | [Nitrogen Limitation](module_13_nitrogen_limitation.md)   | Called from `TERRCARBON2` when `NCYCLE_APPLY = 1`                | `NCYCLE_LIMIT_FACTOR`, which scales NPP              | `TerrestrialCarbon` with `nitrogen`                |
| 5     | [Ocean Carbon](module_10_ocean_carbon.md)                 | `c_atm_for_oceancc`, `co2_feed_deltatemp`                        | `co2_air2ocean_flux`, `dat_pco2s_conc`               | `OceanCarbon`                                      |
| 6     | [CO2 Budget](module_11_co2_budget.md)                     | Land and ocean fluxes, emissions                                 | `DAT_CO2_CONC`                                       | `CO2Budget`                                        |
| 7     | [Halocarbon Chemistry](module_03_halocarbon_chemistry.md) | Halocarbon emissions                                             | Halocarbon concentrations                            | `HalocarbonChemistry`                              |
| 8     | [RF Aggregation](module_07b_rf_aggregation.md)            | All concentrations and prescribed forcing                        | `dat_total_effrf`, `dat_volcanic_effrf` (four boxes) | `GHGForcing`, `OzoneForcing`, `ForcingAggregation` |
| 9     | [Climate](module_08_climate.md)                           | `dat_total_effrf`, `dat_volcanic_effrf`                          | `DAT_SURFACE_TEMP`, `TEMP_OCEANLAYERS`, heat uptake  | `ClimateUDEB`                                      |
| 10    | [Sea Level Rise](module_14_sea_level_rise.md)             | `TEMP_OCEANLAYERS`, `DAT_SURFACE_TEMP`, `DAT_HEATCONTENT_AGGREG` | Sea level components and total                       | `SeaLevelRise`                                     |

Steps 1 to 8 run inside `DELTAQ`.
`run_magicc7` adds the halocarbons before the terrestrial carbon cycle.
The two are independent, so the order does not change the result.

### Feedback loops

- **Carbon cycle**: temperature changes NPP, respiration and ocean uptake,
  which change CO₂, its forcing and so temperature.
- **Methane**: temperature changes the CH4 lifetime, which changes CH4, its forcing and temperature.
- **Ocean circulation**: warming weakens upwelling, which changes deep-ocean heat uptake
  and the transient temperature response.

The feedback strengths are module parameters.
See [terrestrial carbon](module_09_terrestrial_carbon.md#temperature-feedback-parameters),
[CH4 chemistry](module_01_ch4_chemistry.md) and [ocean carbon](module_10_ocean_carbon.md).

## Algorithm

`magicc_step_year` in outline:

```text
NEXT_YEAR_IDX = CURRENT_YEAR_IDX + 1
PREVIOUS_YEAR_IDX = max(CURRENT_YEAR_IDX - 1, 1)

IF (not last year) CALL DELTAQ            ! concentrations and forcing for NEXT_YEAR_IDX

STARTSTEP = 0 in the first year, else 1
ENDSTEP = STEPSPERYEAR - 1 in the last year, else STEPSPERYEAR
DO CURRENT_STEP = STARTSTEP, ENDSTEP      ! climate substeps, see the climate page
    CURRENT_TIME_IDX = CURRENT_TIME_IDX + 1
    ...
END DO

save climate sensitivities                ! equilibrium and effective
annual means, end-of-year temperature, heat content by depth, upwelling rate
CALL sealevel_calc
```

The [climate algorithm](module_08_climate.md#algorithm) lists the substep steps,
and the [Thomas algorithm](module_08_climate.md#thomas-algorithm) the ocean solve.

`DELTAQ` (lines 3735 to 7042) runs before the substeps:

1. Extrapolate temperature for the feedbacks, and set the CO2 and CH4 feedback temperatures
   from `CO2_TEMPFEEDBACK_YRSTART` and `CH4_FEED_YRSTART`.
   Add CH4 clathrate emissions if `CH4_CLATHRATEFEED_APPLY` is set.
2. Run `permafrost_calc` if `PF_APPLY = 1`.
3. Run the CH4, N2O, terrestrial carbon and ocean carbon modules
   in the order of the [inputs and outputs](#inputs-and-outputs) table.
   The ocean carbon cycle runs a [monthly loop](module_10_ocean_carbon.md#monthly-loop)
   that also updates atmospheric CO₂.
4. Store a calculated concentration from the gas's `*_SWITCHFROMCONC2EMIS_YEAR` on.
   Before that year the concentration stays prescribed.
5. Calculate halocarbon concentrations, then every forcing agent.
6. Aggregate the forcing according to `RF_TOTAL_RUNMODUS` (`CO2`, `GHG`, `ALL` and others),
   and add the annual volcanic forcing.

### Substeps

Monthly substeps (`STEPSPERYEAR = 12`) give a smooth transition between annual forcing values,
resolve sub-annual variability such as El Niño and volcanic eruptions,
and improve the accuracy of the ocean-atmosphere exchange.
The implicit ocean scheme is stable at any step length.

- `CURRENT_YEAR_IDX` runs from 1 to `NYEARS`.
  `NEXT_YEAR_IDX` is capped at `NYEARS` and `PREVIOUS_YEAR_IDX` floored at 1.
- `CURRENT_TIME_IDX` counts substeps from the start, and `NEXT_TIME_IDX` is capped at `NTIMES`.
- The first year starts at substep 0, which includes the initial values.
- The last year stops at substep `STEPSPERYEAR - 1`, so it does not overshoot the time axis.
- In other years substep `STEPSPERYEAR` lands on 1 January of the next year,
  which becomes the starting value of that year.

Forcing is interpolated to each substep:

```fortran
Q = datastore_get_box_with_interpolation(dat_total_effrf, alltimes_d(current_time_idx))
```

`alltimes_d` holds decimal years (1850.0, 1850.083, 1850.167 and so on).

### Call sequence

Line numbers are in `MAGICC7.f90`.
Ranges that end in round hundreds are approximate.

| Line      | Call or action                           | Spec                                                    |
| --------- | ---------------------------------------- | ------------------------------------------------------- |
| 2638      | Set `CURRENT_TIME_IDX = 0`               | This page                                               |
| 2639      | `DO CURRENT_YEAR_IDX = 1, NYEARS`        | This page                                               |
| 2695      | `CALL DELTAQ`                            | This page                                               |
| 3815-3857 | Temperature extrapolation for feedbacks  | This page                                               |
| 3862-3905 | CH4 feedback temperatures                | [CH4 chemistry](module_01_ch4_chemistry.md)                        |
| 3908-3919 | N2O feedback setup                       | [N2O chemistry](module_02_n2o_chemistry.md)                        |
| 3923-3925 | `CALL permafrost_calc`                   | [Permafrost](module_12_permafrost.md)                           |
| 3927-4095 | CH4 chemistry, `CALL METHANE` at 4031    | [CH4 chemistry](module_01_ch4_chemistry.md)                        |
| 4244-4270 | N2O chemistry                            | [N2O chemistry](module_02_n2o_chemistry.md)                        |
| 4820      | `CALL TERRCARBON2`                       | [Terrestrial carbon](module_09_terrestrial_carbon.md), [Nitrogen limitation](module_13_nitrogen_limitation.md) |
| 4849-4952 | Ocean carbon monthly loop                | [Ocean carbon](module_10_ocean_carbon.md), [CO2 budget](module_11_co2_budget.md) |
| 5200-5500 | F-gas and Montreal gas chemistry         | [Halocarbon chemistry](module_03_halocarbon_chemistry.md)                 |
| 5550-5800 | CO2, CH4, N2O and halocarbon forcing     | [GHG forcing](module_07a_ghg_forcing.md)                        |
| 5800-5900 | Stratospheric and tropospheric ozone     | [Ozone](module_04_ozone.md)                                |
| 5900-6000 | Direct and indirect aerosol forcing      | [Direct aerosol forcing](module_05_aerosol_direct.md), [Indirect aerosol forcing](module_06_aerosol_indirect.md) |
| 6000-6200 | Land use, BC on snow, aviation           | [Forcing aggregation](module_07b_rf_aggregation.md)                     |
| 6200-7040 | Forcing aggregation                      | [Forcing aggregation](module_07b_rf_aggregation.md)                     |
| 2718-3390 | `DO CURRENT_STEP = STARTSTEP, ENDSTEP`   | [Climate](module_08_climate.md#algorithm)                    |
| 2724      | Forcing for the substep                  | [Climate](module_08_climate.md)                              |
| 2765-2768 | `CALL LAMCALC`                           | [Climate](module_08_climate.md)                              |
| 2775-3096 | Ocean solve for each hemisphere          | [Climate](module_08_climate.md)                              |
| 3159-3222 | SST to air temperature, land temperature | [Climate](module_08_climate.md)                              |
| 3244-3283 | Store substep temperatures, heat uptake  | [Climate](module_08_climate.md)                              |
| 3300-3388 | Upwelling                                | [Climate](module_08_climate.md)                              |
| 3424-3509 | Annual means                             | [Climate](module_08_climate.md)                              |
| 3586-3720 | Heat content                             | [Climate](module_08_climate.md)                              |
| 3726      | `CALL sealevel_calc`                     | [Sea level rise](module_14_sea_level_rise.md)                       |
| 2641      | End of the year loop                     | This page                                               |

`TERRCARBON2` (lines 7054 to 7542) calls `N_CALC_LIMITATION_FACTOR` at lines 7293 to 7300
when `NCYCLE_APPLY = 1`.

### State updates

Updated every substep:

| Variable                     | Lines     | Unit   |
| ---------------------------- | --------- | ------ |
| `GROUND_HEMISPHERIC_TEMPS`   | 2905-2912 | K      |
| `ELNINO_N34_DOUBLEDELTABOX`  | 2930-2932 | K      |
| `OCN_HEMISPHERIC_LAYERTEMPS` | 3087-3093 | K      |
| `CURRENT_TIME_TEMPERATURE`   | 3159-3222 | K      |
| `CURRENT_TIME_MIXEDLAYERTEMP`| 3207-3240 | K      |
| `HEMISPHERIC_HEATXCHANGE`    | 3288-3291 | W m⁻²  |
| `CURRENT_UPWELLING_RATE`     | 3300-3388 | m yr⁻¹ |

Updated once a year, for the next year's index:

| Variable                        | Line        | Meaning                  |
| ------------------------------- | ----------- | ------------------------ |
| `DAT_CH4_CONC`                  | DELTAQ 4053 | Atmospheric CH4          |
| `DAT_N2O_CONC`                  | DELTAQ 4270 | Atmospheric N2O          |
| `DAT_CO2_CONC`                  | DELTAQ 4955 | Atmospheric CO₂          |
| `DAT_SURFACE_ANNUALMEANTEMP`    | 3426        | Annual mean temperature  |
| `DAT_SURFACE_TEMP`              | 3462        | End-of-year temperature  |
| `DAT_HEATUPTAKE_EBALANCE_TOTAL` | 3572        | Annual heat uptake       |
| `TEMP_OCEANLAYERS`              | 3587        | Ocean profile            |
| `DAT_UPWELLING_RATE`            | 3596        | End-of-year upwelling    |
| `DAT_HEATCONTENT_AGGREG`        | 3717        | Heat content by depth    |

## Numerical considerations

- The year loop has a fixed step of one year.
  The climate and ocean carbon cycle both use `STEPSPERYEAR` substeps (12 by default).
- The climate module is fully implicit.
  The carbon cycle is semi-implicit with stability limits, and the chemistry is explicit.
- LAMCALC takes up to 40 iterations a call.
  The carbon cycle does not iterate within a step.
- Energy conservation requires $\frac{dH}{dt} = Q_{total} - \lambda T_{global}$,
  where $H$ is the heat content of atmosphere, ocean and land.

## Issues in MAGICC7

- **Code organisation**: `magicc_step_year` is about 1100 lines,
  and holds climate physics that belongs in `climate_and_ocean.f90`.
- **Feedback lag**: temperature feedbacks use the previous year's temperature.
  This can underestimate fast feedbacks slightly.
- **Sequential updates**: concentrations, forcing and temperature are updated in turn,
  so each module sees slightly stale values from the others.
  Iterating within the step would be more accurate but costs more.
- **Annual forcing**: most forcing agents (aerosols, land use) are averages over the step,
  applied to every substep.
  Only volcanic forcing varies within the year.
- **Update timing**: upwelling is updated at the end of a substep from that substep's temperature,
  but that temperature used forcing from before the update.

## Differences from MAGICC7

- The last year runs 12 substeps where MAGICC7 runs 11 (see [time conventions](time_conventions.md)).
- `run_magicc7` supports only `co2_plantboxresp_method` 2.

## References

| File                             | Lines     | Content                          |
| -------------------------------- | --------- | -------------------------------- |
| `MAGICC7.f90`                    | 2635-2643 | `magicc_run`                     |
| `MAGICC7.f90`                    | 2646-3727 | `magicc_step_year`               |
| `MAGICC7.f90`                    | 3735-7042 | `DELTAQ`                         |
| `MAGICC7.f90`                    | 7054-7542 | `TERRCARBON2`                    |
| `MAGICC7.f90`                    | 8070-8278 | `LAMCALC`                        |
| `physics/permafrost.f90`         | 308-772   | `permafrost_calc`                |
| `physics/sealevel.f90`           | 239-753   | `sealevel_calc`                  |
| `physics/ocean_carbon_cycle.f90` | -         | Ocean carbon cycle class         |

| Fortran module              | Purpose                       |
| --------------------------- | ----------------------------- |
| `MOD_YEARS`                 | Time indices                  |
| `MOD_CLIMATECORE_AND_OCEAN` | Climate parameters and state  |
| `mod_carbon_cycle`          | Carbon cycle parameters       |
| `MOD_RADIATIVE_FORCING`     | Forcing parameters            |
| `MOD_DATASTORE`             | Data storage                  |
| `MOD_PERMAFROST`            | Permafrost parameters         |
| `MOD_SEALEVEL`              | Sea level parameters          |
| `mod_ocean_carbon_cycle`    | Ocean carbon cycle class      |
