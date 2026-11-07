# Initialisation and Data Loading

## Overview

MAGICC7 prepares everything a run needs before its first step.
`magicc_init_run` reads the configuration, loads emissions, concentrations and forcing from input files,
merges scenario data into the historical record and allocates every array.
It also derives quantities that later modules depend on:
the four-box area fractions, the natural CH4 and N2O emissions that close each budget,
and the aerosol, land-use, solar and volcanic forcing that depend only on inputs.

This page covers the containers, conventions and order of that preparation.
The physics of each step lives in the module page that uses it.

## RSCM implementation

RSCM has no single initialisation component.
The work is split between the model framework and the MAGICC7 runner:

- **Model set-up**: `ModelBuilder` assembles components and prescribed data into a `Model`,
  see [Key Concepts](../key_concepts.md#modelbuilder-pattern).
  `Timeseries` and `TimeseriesCollection` replace the `DATASTORE`,
  and the `FourBox` grid replaces `DATBOX`, see [Spatial Grids](../grids.md#four_box-magicc-standard).
- **MAGICC7 initialisation**: `rscm.magicc.full.run_magicc7` (`python/rscm/magicc/full.py`).
  `default_parameters` returns MAGICC7 v7.5.3's defaults,
  packaged in `python/rscm/magicc/data/magicc7/defaults.json`.
  `load_inputs` reads the packaged input files, `prescribed_concentrations` the concentration files
  and `natural_emissions` closes the CH4 and N2O budgets.
  The forcing calculated from inputs comes from `rscm.magicc.prescribed`,
  see [direct aerosol forcing](module_05_aerosol_direct.md) and [indirect aerosol forcing](module_06_aerosol_indirect.md).
- **Climate feedback parameters**: LAMCALC runs inside `ClimateUDEB`
  (`crates/rscm-magicc/src/climate/lamcalc.rs`), see [climate](module_08_climate.md).
- **Status**: Implemented. See the [status table](README.md#rscm-implementation-status).
- **Tests**: every coupled run starts from this set-up, so `test_coupled_run` and `test_run_magicc7_defaults`
  in `tests/regression/test_full_run.py` cover it.
  `tests/test_magicc_full.py` checks that the natural emissions close the CH4 and N2O budgets.
  `crates/rscm-magicc/tests/lamcalc_integration.rs` and `lamcalc_reference_values.rs` test LAMCALC.

## DATASTORE

`DATASTORE` is MAGICC7's container for a timeseries.
Nearly every input and output passes through one.
These are the fields that the module pages refer to:

| Field                     | Shape               | Meaning                                                             |
| ------------------------- | ------------------- | ------------------------------------------------------------------- |
| `DATGLOBE`                | years (or steps)    | Global value                                                        |
| `DATBOX`                  | years (or steps), 4 | Box values, in the [box order](#four-box-geography)                 |
| `FRACTIONSBOX`            | years, 4            | Share of the global value in each box                               |
| `PREIND_DATGLOBE`         | scalar              | Global value in `RF_PREIND_REFERENCEYR`                             |
| `PREIND_DATBOX`           | 4                   | Box values in `RF_PREIND_REFERENCEYR`                               |
| `FIRSTYEAROFFSET_DATGLOBE`| scalar              | Global offset subtracted from forcing, set by the RF initialisation |
| `FIRSTYEAROFFSET_DATBOX`  | 4                   | Box offsets                                                         |
| `EFF_NORMYR_BOX`          | 4                   | Box forcing in `RF_REGIONS_NORMYEAR`, which sets efficacy patterns  |
| `ANNUALSTEPS`             | scalar              | 1 for annual data, 12 for monthly                                   |
| `NULLORAVGORSUM`          | scalar              | How boxes combine: 0 none, 1 area-weighted mean, 2 sum              |
| `FIRSTYEAR`, `LASTYEAR`   | scalar              | Years the data covers                                               |

The RF initialisation method that sets the first-year offsets is described in
[forcing aggregation](module_07b_rf_aggregation.md#pre-industrial-reference).

Some of the main instances:

| Variable              | Contents                                         |
| --------------------- | ------------------------------------------------ |
| `DAT_CO2_CONC`        | CO2 concentration (ppm)                          |
| `DAT_CH4_CONC`        | CH4 concentration (ppb)                          |
| `DAT_N2O_CONC`        | N2O concentration (ppb)                          |
| `DAT_CO2I_EMIS`       | Fossil CO2 emissions (GtC/yr)                    |
| `DAT_CO2B_EMIS`       | Land-use CO2 emissions (GtC/yr)                  |
| `DAT_CH4I_EMIS`       | Industrial CH4 emissions (Tg CH4/yr)             |
| `DAT_SOXI_EMIS`       | Industrial SOx emissions (Tg S/yr)               |
| `DAT_FGAS_EMIS(i)`    | F-gas emissions (kt/yr)                          |
| `DAT_MHALO_EMIS(i)`   | Montreal Protocol gas emissions (kt/yr)          |
| `DAT_TOTAL_EFFRF`     | Total efficacy-adjusted forcing (W/m²)           |
| `DAT_VOLCANIC_RF`     | Volcanic forcing (W/m², monthly)                 |
| `DAT_SURFACE_TEMP`    | Surface air temperature change (K)               |

## Time convention

Every MAGICC7 timeseries holds start-of-year values, except emissions, which are mid-year values.
So `DAT_CO2_CONC%DATGLOBE(i)` is the concentration on 1 January of `STARTYEAR + i - 1`,
while `DAT_CO2I_EMIS%DATGLOBE(i)` is the emissions averaged over that year.

Year `y` has index `y - STARTYEAR + 1`.
Monthly series have `NTIMES = NYEARS * STEPSPERYEAR` entries, with `STEPSPERYEAR = 12`,
and step `s` of year `y` is at `y + (s - 1) / 12`.
`datastore_yr`, `datastore_get_globe` and `datastore_get_box` look up a year in a `DATASTORE`
whose `FIRSTYEAR` may differ from `STARTYEAR`.

How RSCM maps these conventions onto its steps is described in [time conventions](time_conventions.md)
and in [Key Concepts](../key_concepts.md#step-semantics).

## Four-box geography

MAGICC7 divides the surface into four boxes.
RSCM uses the same order.
This table is the reference for the other module pages.

| Index | Box                 | Global area fraction    | Default |
| ----- | ------------------- | ----------------------- | ------- |
| 1     | Northern ocean (NO) | $(1 - f_{NL}) / 2$      | 0.29    |
| 2     | Northern land (NL)  | $f_{NL} / 2$            | 0.21    |
| 3     | Southern ocean (SO) | $(1 - f_{SL}) / 2$      | 0.395   |
| 4     | Southern land (SL)  | $f_{SL} / 2$            | 0.105   |

$f_{NL}$ = `CORE_HEMISFRACTION_NH_LAND` (default 0.42)
and $f_{SL}$ = `CORE_HEMISFRACTION_SH_LAND` (default 0.21) are land fractions of each hemisphere.
MAGICC7 stores the four fractions as `GLOBALAREAFRACTIONS` (aliases `FGNO`, `FGNL`, `FGSO`, `FGSL`).
In RSCM they are `rscm.magicc.prescribed.area_fractions()`
and the `nh_land_fraction` and `sh_land_fraction` parameters of `ClimateUDEB` and `ForcingAggregation`.

`NULLORAVGORSUM` sets how box values combine into the global value $X_g$:

$$
X_g = \sum_{i=1}^{4} a_i X_i \quad \text{(mean, 1)} \qquad X_g = \sum_{i=1}^{4} X_i \quad \text{(sum, 2)}
$$

with $a_i$ the area fractions above.
Concentrations, temperatures and forcing use the mean, emissions the sum.

## Configuration

`magicc_read_config` reads Fortran namelists (`NML_ALLCFGS` in `allcfgs.f90`) in this order,
each file overriding the last:

1. `MAGCFG_DEFAULTALL.CFG`, the defaults of every parameter.
2. `MAGCFG_USER.CFG`, the user's overrides.
3. `FILE_TUNINGMODEL_1` to `FILE_TUNINGMODEL_10`, model tunings such as `MAGTUNE_CMIP6.CFG`.
4. The sea level `FILE_SLR_*` files.

`READ_ALLCFG` reports the offending line when a namelist read fails.

## Parameters

| Parameter                     | Default   | Description                                                   |
| ----------------------------- | --------- | ------------------------------------------------------------- |
| `STARTYEAR`                   | 1750      | First year of the run                                         |
| `ENDYEAR`                     |           | Last year of the run, set per run                             |
| `STEPSPERYEAR`                | 12        | Substeps per year                                             |
| `CORE_HEMISFRACTION_NH_LAND`  | 0.42      | Land fraction of the Northern Hemisphere                      |
| `CORE_HEMISFRACTION_SH_LAND`  | 0.21      | Land fraction of the Southern Hemisphere                      |
| `RF_PREIND_REFERENCEYR`       | 1750      | Year of the preindustrial reference values                    |
| `RF_INITIALIZATION_METHOD`    | JUMPSTART | First-year forcing offset, see [forcing aggregation](module_07b_rf_aggregation.md#pre-industrial-reference) |
| `RF_REGIONS_NORMYEAR`         | 2000      | Year whose box forcing sets the efficacy patterns             |

The defaults of each module's own parameters are on its page.

## Algorithm

`magicc_init_run` runs these steps in order:

1. Reads the ocean area factor and initial temperature profile files when their switches are set,
   and the glacier sea level parameters.
2. Adjusts dependent parameters: the CO2 fertilisation temperature switches,
   the CH4 lifetime feedback from NOx, VOC and CO, the equilibrium climate sensitivity's dependence on RLO
   and the heat capacities.
3. Calculates the [area fractions](#four-box-geography).
4. Allocates the ocean, carbon cycle, methane and cloud arrays.
5. Loads the concentration, emission and forcing files with `datastore_read`
   (`datastore_read_monthly` for volcanic forcing).
   An empty file name gives a zero series.
   Data are interpolated to the model years and the `PREIND_*` values set from `RF_PREIND_REFERENCEYR`.
6. Reads the `.SCEN` or `.SCEN7` scenario files (`scen6_emis_read`, `scen7_emis_read`),
   sums their regions into the four boxes with `SUM_RAWDATA_TOBOX`
   and merges them into the historical emissions with `HANDLE_RAWSCEN_INTO_HISTEMIS`.
7. Closes the CH4 and N2O budgets with natural emissions,
   see [CH4 chemistry](module_01_ch4_chemistry.md#budget-initialisation)
   and [N2O chemistry](module_02_n2o_chemistry.md#budget-initialisation).
8. Scales and extends the aerosol forcing with emissions (`EXTRAP_RF_WITH_EMIS`)
   and calculates nitrate forcing, see [direct aerosol forcing](module_05_aerosol_direct.md).
9. Calculates each halocarbon's molecular mass (`CALC_HALOS_MOLMASS`) and non-OH lifetime,
   see [halocarbon chemistry](module_03_halocarbon_chemistry.md).
10. Allocates and initialises permafrost and nitrogen limitation.

`magicc_dealloc` frees every array afterwards, so one process can run MAGICC7 again.

## Differences from MAGICC7

- **No global state**: RSCM components receive their parameters and inputs explicitly,
  so there is nothing to allocate or free between runs.
- **Configuration**: RSCM reads no namelists.
  `run_magicc7` takes a mapping that overrides `default_parameters`, with lowercase MAGICC7 names.
- **Scenario files**: `run_magicc7` takes emissions as arrays with one value per year,
  instead of reading and merging `.SCEN7` files.
- **Natural emissions**: MAGICC7 adds them to `DAT_CH4N_EMIS` and `DAT_N2ON_EMIS`.
  RSCM passes them to `CH4Chemistry` and `N2OChemistry` as constant parameters.
  `natural_emissions` calculates MAGICC7's budget-closing values, and `run_magicc7` passes them in.

## References

MAGICC7 source files:

| File                                   | Contents                                                                         |
| -------------------------------------- | -------------------------------------------------------------------------------- |
| `src/libmagicc/MAGICC7.f90`            | `magicc_init_run` (lines 131-2569), `magicc_read_config` (2573-2625), `magicc_dealloc` (12007-12493) |
| `src/libmagicc/allcfgs.f90`            | `NML_ALLCFGS`, `READ_ALLCFG`, `READ_TUNING`                                       |
| `src/libmagicc/utils/datastore.f90`    | `DATASTORE`, `datastore_read`, `DATASTORE_DEALLOC`                               |
| `src/libmagicc/utils/years.f90`        | `NYEARS`, `NTIMES`, `yr_idx`                                                     |
| `src/libmagicc/io/scen_file.f90`       | `scen6_emis_read`, `scen7_emis_read`, `scen7_file_read`                           |
| `src/libmagicc/mod_areas.f90`          | Area fractions                                                                   |
| `run/MAGCFG_DEFAULTALL.CFG`            | Default parameters                                                               |
