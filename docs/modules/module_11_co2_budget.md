# CO2 Budget

## Overview

The CO2 budget closes the global carbon cycle.
It balances the sources (fossil emissions, CO2 from fossil CH4 oxidation, permafrost)
against the land and ocean sinks, and turns the net source into a change in atmospheric CO2.
It also diagnoses the inverse emissions implied by a concentration path.

MAGICC7 has no separate module for this.
The budget is spread through `MAGICC7.f90`:
the terrestrial carbon cycle (`TERRCARBON2`, [terrestrial carbon](module_09_terrestrial_carbon.md))
supplies the land flux,
the [monthly ocean loop](module_10_ocean_carbon.md#monthly-loop) integrates atmospheric CO2,
and the driver then derives the annual fluxes and the inverse emissions.
This page collects the parts that belong to neither carbon module.

## RSCM implementation

- **Component**: `CO2Budget`, built from Python with `rscm.magicc.CO2BudgetBuilder`.
- **Source**: `crates/rscm-magicc/src/carbon/budget.rs`,
  with parameters in `crates/rscm-magicc/src/parameters/co2_budget.rs`.
- **Status**: Implemented. See the [status table](README.md#rscm-implementation-status).
- **Tests**: `tests/regression/test_full_run.py::test_coupled_run` checks CO2 from emissions,
  and `test_run_magicc7_defaults` checks CO2 prescribed up to 2015 and calculated after.
  `crates/rscm-magicc/tests/carbon_cycle_physics.rs` checks the mass balance
  with the land and ocean components.

`OceanCarbon` integrates atmospheric CO2 over its monthly substeps, as MAGICC7 does.
With `ocean_end_of_year`, which `run_magicc7` sets,
`CO2Budget` takes next year's concentration from that integration.
Without it, the annual mass balance below gives the same value up to rounding.

## Equations

### Mass balance

In emissions-driven years atmospheric CO2 follows

$$
\frac{dC}{dt} = \frac{E_{fos} + E_{ox} + E_{pf} - F_{land} - F_{ocean}}{\gamma}
$$

where:

- $C$ is the atmospheric CO2 concentration (ppm),
- $E_{fos}$ is fossil and industrial CO2 emissions (GtC yr⁻¹),
- $E_{ox}$ is CO2 from the oxidation of fossil CH4 (GtC yr⁻¹),
- $E_{pf}$ is permafrost CO2 (GtC yr⁻¹), with `PF_APPLY` on,
- $F_{land}$ is the net atmosphere-to-land flux (GtC yr⁻¹),
  which includes land-use emissions because deforestation removes carbon from the land pools,
- $F_{ocean}$ is the atmosphere-to-ocean flux (GtC yr⁻¹),
- $\gamma$ = `GTC_PER_PPM` = 2.123 GtC ppm⁻¹, hard-coded in `carboncycle_init`.

The atmospheric pool is $M = \gamma C$ (GtC).

### Budget closure

`TERRCARBON2` combines every source except the ocean into one annual term:

$$
E_{tf} = E_{fos} + E_{ox} + E_{pf} - \Delta P_{land}
$$

where $E_{tf}$ is `CO2_TERRBIO_AND_FOSSIL_EMIS`
and $\Delta P_{land}$ is the year's change in the plant, detritus and soil pools.
The [monthly ocean loop](module_10_ocean_carbon.md#monthly-loop) adds $E_{tf} / (12\gamma)$ each month
and removes the month's ocean uptake.

After the loop, the annual land flux is the change in the land pools, $F_{land} = \Delta P_{land}$,
and the annual ocean flux is the mean of the twelve monthly fluxes times $\gamma$.

### Inverse emissions

MAGICC7 diagnoses the fossil emissions that would explain the concentration change:

$$
E_{inv} = F_{ocean} + F_{land} + \Delta M - E_{pf} - E_{ox}
$$

It also accumulates them in `CO2I_INVERSE_CUMEMIS`.

## State variables

| Variable                     | Fortran name            | Units  | Initial value            |
| ---------------------------- | ----------------------- | ------ | ------------------------ |
| Atmospheric CO2              | `DAT_CO2_CONC%DATGLOBE` | ppm    | From the input file      |
| Atmospheric CO2 pool         | `CO2_ATMOS_POOL`        | GtC    | $\gamma C$               |
| Ocean surface pCO2           | `DAT_PCO2S_CONC%DATGLOBE` | ppm  | First-year CO2           |
| Air-to-ocean flux            | `CO2_AIR2OCEAN_FLUX`    | GtC/yr | 0                        |
| Air-to-land flux             | `CO2_AIR2LAND_FLUX`     | GtC/yr | From pool changes        |
| Net ecosystem exchange       | `CO2_NETECOEXCH_FLUX`   | GtC/yr | From pool changes        |
| Cumulative inverse emissions | `CO2I_INVERSE_CUMEMIS`  | GtC    | 0                        |

## Inputs and outputs

| Direction | Variable                          | Units  | Source or destination      | Fortran variable                                   |
| --------- | --------------------------------- | ------ | -------------------------- | -------------------------------------------------- |
| Input     | Fossil CO2 emissions              | GtC/yr | Emissions                  | `DAT_CO2I_EMIS%DATGLOBE`                           |
| Input     | Fossil CO2 including CH4 oxidation | GtC/yr | CH4 module                | `DAT_CO2I_INCLCH4OX_EMIS%DATGLOBE`                 |
| Input     | Land-use CO2 emissions            | GtC/yr | Emissions                  | `DAT_CO2B_EMIS%DATGLOBE`                           |
| Input     | Permafrost CO2                    | GtC/yr | Permafrost, if `PF_APPLY`  | `DAT_CO2PF_EMIS%DATGLOBE`                          |
| Input     | Land pools                        | GtC    | `TERRCARBON2`              | `CO2_PLANT_POOL`, `CO2_DETRITUS_POOL`, `CO2_SOIL_POOL` |
| Input     | Prescribed CO2                    | ppm    | Input file                 | `DAT_CO2_CONC%DATGLOBE`                            |
| Output    | Atmospheric CO2                   | ppm    | Forcing and carbon cycles  | `DAT_CO2_CONC%DATGLOBE`                            |
| Output    | Atmospheric CO2 pool              | GtC    | Output only                | `CO2_ATMOS_POOL`, `DAT_CO2_ATMOS_POOL%DATGLOBE`    |
| Output    | Air-to-ocean and air-to-land flux | GtC/yr | Output, inverse emissions  | `CO2_AIR2OCEAN_FLUX`, `CO2_AIR2LAND_FLUX`          |
| Output    | Net ecosystem exchange            | GtC/yr | Output                     | `CO2_NETECOEXCH_FLUX`                              |
| Output    | Inverse CO2 emissions             | GtC/yr | Diagnostic                 | `CO2I_INVERSE_EMIS`, `DAT_CO2I_INVERSE_EMMS%DATGLOBE` |

## Parameters

The ocean parameters are on the [ocean carbon](module_10_ocean_carbon.md#parameters) page.

| Parameter                 | Fortran name                    | Units | Default | Description                                           |
| ------------------------- | ------------------------------- | ----- | ------- | ----------------------------------------------------- |
| Switch year               | `CO2_SWITCHFROMCONC2EMIS_YEAR`  | year  | 2015    | Last year of prescribed concentrations                |
| Preindustrial CO2         | `CO2_PREINDCO2CONC`             | ppm   | 278.0   | Preindustrial reference concentration                 |
| Preindustrial CO2 apply   | `CO2_PREINDCO2CONC_APPLY`       | flag  | 0       | Override the input file's preindustrial value         |
| Concentration cap         | `CO2_CAPCONC_PPM`               | ppm   | 2000    | Largest allowed CO2 concentration                     |
| Concentration cap apply   | `CO2_CAPCONC_APPLY`             | flag  | 0       | Apply the cap                                         |
| Zero-emissions threshold  | `CO2_ZEROEMIS_AFTER_PGC`        | PgC   | 1000    | Cumulative emissions after which emissions are zero   |
| Zero-emissions apply      | `CO2_ZEROEMIS_AFTERXPGC_APPLY`  | flag  | 0       | Apply the threshold                                   |

## Algorithm

Each annual step runs:

1. `TERRCARBON2` ([terrestrial carbon](module_09_terrestrial_carbon.md)) updates the land pools and $E_{tf}$.
2. The [monthly ocean loop](module_10_ocean_carbon.md#monthly-loop) integrates atmospheric CO2 and ocean pCO2.
   Before `CO2_SWITCHFROMCONC2EMIS_YEAR` it interpolates the prescribed concentration instead.
   With `CO2_CAPCONC_APPLY`, each month's concentration is capped at `CO2_CAPCONC_PPM`.
3. The driver stores the end-of-year CO2, ocean pCO2 and the atmospheric pool $\gamma C$.
4. It derives the annual land flux and net ecosystem exchange from the pool changes.
5. It calculates the [inverse emissions](#inverse-emissions) and their cumulative total.

## Numerical considerations

- Atmospheric CO2 is integrated by forward Euler over the twelve monthly substeps of the ocean loop.
- The land pools use the implicit trapezoidal scheme of [terrestrial carbon](module_09_terrestrial_carbon.md)
  on the annual step, so the land source is constant within the year.
- The ocean side carries a flux change limiter and two-month averages,
  described with the [monthly loop](module_10_ocean_carbon.md#monthly-loop).

## Issues in MAGICC7

- **No module boundary**: the budget is spread through `MAGICC7.f90`
  (about lines 4818 to 4976 for the ocean loop and CO2 update,
  5000 to 5140 for the land flux and inverse emissions, and 7054 to 7542 for `TERRCARBON2`).
  It shares state with the land, ocean, CH4, permafrost and climate code.
- **First-year overwrite** (lines 4963 to 4974): the first year's pCO2 and atmospheric pool are overwritten
  with the second year's, under "TODO: check why we overwrite first year values".
- **Inverse inconsistency** (lines 5080 to 5081): the source notes that prescribing zero CO2 emissions
  does not give zero inverse emissions, so the inverse is not fully consistent with the forward model.
- **Hard-coded values**: `GTC_PER_PPM` (`carbon_cycle.f90:88`) and the twelve substeps a year
  are not parameters.
- **Overlapping flux variables**: `CO2_AIR2OCEAN_FLUX` (GtC/yr), `CO2_AIR2OCNFLX_ALLT_PPMPERYR`
  (monthly, ppm/yr), `CO2_AIR2LAND_FLUX`, `CO2_NETECOEXCH_FLUX` and `CO2_TERRBIO_AND_FOSSIL_EMIS`
  describe related quantities in different units and steps.

## Differences from MAGICC7

- There is no concentration cap (`CO2_CAPCONC_APPLY`).
- There is no zero-emissions switch (`CO2_ZEROEMIS_AFTERXPGC_APPLY`).
- No inverse fossil emissions are calculated while CO2 is prescribed.
  `Emissions|CO2|Net` is then the net source implied by the prescribed concentrations.
- `CO2Budget` also reports the airborne fraction, the net source over the sum of the sources.

## References

The land and ocean code is listed on the [terrestrial carbon](module_09_terrestrial_carbon.md#references)
and [ocean carbon](module_10_ocean_carbon.md#references) pages.

| File                       | Lines     | Contents                                          |
| -------------------------- | --------- | ------------------------------------------------- |
| `MAGICC7.f90`              | 4818-4820 | Call to `TERRCARBON2`                             |
|                            | 4826-4842 | First-year initialisation and averaging variables |
|                            | 4849-4952 | Monthly ocean loop                                |
|                            | 4916-4931 | Atmospheric CO2 update                            |
|                            | 4934-4938 | Concentration cap                                 |
|                            | 4955-4975 | End-of-year CO2 and atmospheric pool              |
|                            | 5006-5036 | Net human emissions, land flux and NEE            |
|                            | 5082-5139 | Inverse and cumulative inverse emissions          |
|                            | 7509-7522 | `CO2_TERRBIO_AND_FOSSIL_EMIS`                     |
| `physics/carbon_cycle.f90` | 7-37      | Pool, flux and parameter declarations             |
|                            | 65-139    | `carboncycle_init`, with `GTC_PER_PPM` at line 88 |
| `allcfgs.f90`              | 36-54     | CO2 namelist parameters                           |
