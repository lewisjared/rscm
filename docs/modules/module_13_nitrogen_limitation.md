# Nitrogen Limitation

## Overview

Plant growth needs nitrogen as well as carbon.
As CO2 rises, fertilisation raises potential NPP, and that extra growth needs more nitrogen.
If the soil cannot supply it, the fertilisation effect is reduced.

This module tracks a plant-available soil nitrogen pool.
Its sources are deposition, recycling from decomposition and biological fixation,
and its sinks are plant uptake and losses (leaching and gaseous).
Each year the pool is set to the steady state of these fluxes,
and a limitation factor derived from it scales the NPP
of the [terrestrial carbon cycle](module_09_terrestrial_carbon.md).

/// warning
MAGICC7 marks this module as unfinished.
With `NCYCLE_APPLY = 1` it logs "Careful using the nitrogen cycle, it's not finished yet"
(`nitrogen_limitation.f90`, line 38).
See [issues in MAGICC7](#issues-in-magicc7).
///

## RSCM implementation

- **Component**: `TerrestrialCarbon` with its `nitrogen` parameters set (off by default).
- **Source**: `crates/rscm-magicc/src/carbon/nitrogen.rs`,
  parameters in `crates/rscm-magicc/src/parameters/nitrogen.rs`.
  Deposition from emissions is calculated in Python by `rscm.magicc.full.nitrogen_inputs`.
- **Status**: Implemented. See the [status table](README.md#rscm-implementation-status).
  `rscm.magicc.full.run_magicc7` reads `ncycle_apply` and the `ncycle_*` parameters.
- **Tests**: `tests/regression/test_full_run.py::test_nitrogen_coupled_run` runs the coupled SSP245 chain
  with deposition from emissions and fixation from `NCYCLE_NFIX3.IN`.
  Temperature and concentrations match MAGICC7 v7.5.3's binary output to a relative 1e-9.
  The limitation factor, nitrogen pool and NPP appear only in `CARBONCYCLE.OUT`,
  so they are compared at its print precision (relative 1e-4).
  Unit tests are in `nitrogen.rs`.

## Equations

### Nitrogen pool balance

Each year the pool $N$ (Gt N) solves

$$
F_{dep} + F_{recyc} + F_{fix}(N) - F_{up}(N) - \frac{N}{\tau_{loss}} = 0
$$

where $\tau_{loss}$ is `NCYCLE_TURNOVERTIME_LOSSES`.
Only fixation and uptake depend on $N$, through the availability ratio below.

### Nitrogen deposition

With `NCYCLE_APPLY_NDEP_FROM_EMIS = 0`, $F_{dep}$ is read from `FILE_NDEP_IN`.
With `NCYCLE_APPLY_NDEP_FROM_EMIS = 1` it comes from NOx and NH3 emissions (Mt N/yr):

$$
F_{dep} = f_{land} \times 10^{-3} \left(E_{NOx,I} + E_{NOx,B} + 2\,E_{NH3,I}\right)
$$

where $f_{land}$ is `NCYCLE_FRAC_NDEP_LAND` and the subscripts I and B are industrial and biomass burning.
The doubled industrial NH3 is a MAGICC7 bug (see [NH3 double count](#nh3-double-count)).

### Nitrogen recycling

$$
F_{recyc} = r_{rec} \, R_h \, \frac{1 + f}{f + \beta} \, \frac{1 + f}{f + f_T^{NPP}}
$$

- $r_{rec}$ is `NCYCLE_NCRATIO_RECYC`.
- $R_h$ is the previous year's total terrestrial respiration (`CO2_TOTALRESPIRATION`),
  or NPP in the first year.
- $\beta$ and $f_T^{NPP}$ are the CO2 fertilisation and NPP temperature factors of
  [terrestrial carbon](module_09_terrestrial_carbon.md#net-primary-production).
- $f$ is `NCYCLE_RECYC2CNRATIO_FACTOR`.
  At its default of 0, recycling is divided by both factors in full.

### Nitrogen availability

The availability function is

$$
\phi(N, NPP) = \frac{1 + A}{A + \exp\left(-C\left(\dfrac{N^2}{N + S} - k_{up}\,NPP\right)\right)}
$$

with $A$ = `NCYCLE_NPP2NAVAIL_FACTOR_A`, $C$ = `NCYCLE_NPP2NAVAIL_FACTOR_C`,
$S$ = `NCYCLE_NCRATIO_UPTAKE_SUPPLY` and $k_{up}$ = `NCYCLE_NCRATIO_UPTAKE`.
$NPP$ is the NPP before limitation.
From `CO2_FERTILIZATION_YRSTART` on, uptake and fixation scale with

$$
\rho(N) = \frac{\phi(N, NPP)}{\phi(N_0, NPP_0)}
$$

where $N_0$ and $NPP_0$ are the pool and unlimited NPP of the last year before `CO2_FERTILIZATION_YRSTART`.
Before that year $\rho = 1$.

### Uptake, fixation and losses

$$
F_{up} = k_{up} \, NPP \, \rho(N)
$$

With `NCYCLE_APPLY_PROGNOSTIC_NFIX = 0`, $F_{fix}$ is read from `FILE_NFIX_IN`.
With `NCYCLE_APPLY_PROGNOSTIC_NFIX = 1` it is a Michaelis-Menten function of the scaled NPP:

$$
F_{fix} = 10^{-3} \, \frac{V_{max} \, NPP \, \rho(N)}{K_m + NPP \, \rho(N)}
$$

with $V_{max}$ = `NCYCLE_NFIX_VMAX` (Tg N/yr) and $K_m$ = `NCYCLE_NFIX_KM` (GtC/yr).
Losses are $N/\tau_{loss}$.

### Nitrogen limitation factor

The limitation factor is the availability ratio at the solved pool, $L = \rho(N)$.
It is 1 before `CO2_FERTILIZATION_YRSTART`.
`MAGICC7.f90` multiplies NPP by it (line 7296):

$$
NPP \leftarrow L \cdot NPP
$$

## Inputs and outputs

| Input                    | RSCM variable                       | Units    | Fortran variable                     |
| ------------------------ | ----------------------------------- | -------- | ------------------------------------ |
| NPP before limitation    | (internal to `TerrestrialCarbon`)   | GtC/yr   | `CO2_CURRENT_NPP`                    |
| Previous respiration     | (internal to `TerrestrialCarbon`)   | GtC/yr   | `CO2_TOTALRESPIRATION(t-1)`          |
| Fertilisation factor     | (internal to `TerrestrialCarbon`)   | 1        | `CO2_EFF_FERTILIZATION_FACTOR`       |
| NPP temperature factor   | (internal to `TerrestrialCarbon`)   | 1        | `CO2_EFF_NPP_TEMPFEEDBACK`           |
| Deposition               | `Nitrogen Deposition`               | Gt N/yr  | `DAT_NCYCLE_NDEP`, or NOx and NH3 emissions |
| Prescribed fixation      | `Nitrogen Fixation`                 | Gt N/yr  | `DAT_NCYCLE_NFIX`                    |

| Output or state          | RSCM variable                       | Units    | Fortran variable                     |
| ------------------------ | ----------------------------------- | -------- | ------------------------------------ |
| Limitation factor        | `Nitrogen Limitation Factor`        | 1        | `NCYCLE_LIMIT_FACTOR`                |
| Nitrogen pool            | `Nitrogen Pool`                     | Gt N     | `NCYCLE_NPOOL`                       |
| Plant uptake             | `Nitrogen Flux|Uptake`              | Gt N/yr  | `NCYCLE_FN_PLANTUPTAKE`              |
| Recycling                | `Nitrogen Flux|Recycling`           | Gt N/yr  | `NCYCLE_FN_RECYCLING`                |
| Losses                   | `Nitrogen Flux|Losses`              | Gt N/yr  | `NCYCLE_FN_LOSSES`                   |
| Fixation                 | `Nitrogen Flux|Fixation`            | Gt N/yr  | `NCYCLE_FN_FIXATION`                 |
| Deposition               | `Nitrogen Flux|Deposition`          | Gt N/yr  | `NCYCLE_FN_DEPOSITION`               |
| Reference pool and NPP   | (component state)                   | Gt N, GtC/yr | `NCYCLE_NPOOL_0`, `CO2_CURRENT_NPP_0` |

MAGICC7 writes the outputs to `CARBONCYCLE.OUT` and `CARBONCYCLE.BINOUT`.

## Parameters

Defaults are those MAGICC7 v7.5.3 runs with.
The three marked * are absent from `MAGCFG_DEFAULTALL.CFG` and take the value 0
(see [issues in MAGICC7](#issues-in-magicc7)).

| Parameter                        | Units      | Default  | Description                                          |
| -------------------------------- | ---------- | -------- | ---------------------------------------------------- |
| `NCYCLE_APPLY`                   | flag       | 0        | Turns nitrogen limitation on                         |
| `NCYCLE_APPLY_CN_INITIAL_VALUES` | flag       | 0        | Use the `NCYCLE_*_INITIAL` pools and NPP below       |
| `NCYCLE_APPLY_PROGNOSTIC_NFIX`   | flag       | 0        | Calculate fixation from NPP instead of reading it    |
| `NCYCLE_APPLY_NDEP_FROM_EMIS`    | flag       | 0        | Calculate deposition from NOx and NH3 emissions      |
| `NCYCLE_NPP2NAVAIL_FACTOR_A`     | 1          | 2.0      | Availability sensitivity $A$                         |
| `NCYCLE_NPP2NAVAIL_FACTOR_C`     | 1          | 2.0      | Availability steepness $C$                           |
| `NCYCLE_RECYC2CNRATIO_FACTOR`*   | 1          | 0        | Damping $f$ of the CO2 and temperature effects on recycling |
| `NCYCLE_NCRATIO_RECYC`           | Gt N/GtC   | 0.013    | Recycling per unit respiration $r_{rec}$             |
| `NCYCLE_NCRATIO_UPTAKE`          | Gt N/GtC   | 0.013    | Uptake per unit NPP $k_{up}$                         |
| `NCYCLE_NCRATIO_UPTAKE_SUPPLY`   | Gt N       | 0.05     | Half saturation of supply $S$                        |
| `NCYCLE_TURNOVERTIME_LOSSES`     | yr         | 2.0      | Residence time against losses $\tau_{loss}$          |
| `NCYCLE_NFIX_VMAX`*              | Tg N/yr    | 0        | Maximum fixation $V_{max}$                           |
| `NCYCLE_NFIX_KM`*                | GtC/yr     | 0        | NPP at half the maximum fixation $K_m$               |
| `NCYCLE_FRAC_NDEP_LAND`          | 1          | 0.7      | Share of emitted nitrogen deposited on land          |
| `NCYCLE_PLANTPOOL_INITIAL`       | GtC        | 884.8584 | Plant pool with C+N initial values                   |
| `NCYCLE_DETRITUSPOOL_INITIAL`    | GtC        | 92.7738  | Detritus pool with C+N initial values                |
| `NCYCLE_SOILPOOL_INITIAL`        | GtC        | 1681.525 | Soil pool with C+N initial values                    |
| `NCYCLE_NPP_INITIAL`             | GtC/yr     | 66.2716  | NPP with C+N initial values                          |

| Input file       | Default                           |
| ---------------- | --------------------------------- |
| `FILE_NFIX_IN`   | `NCYCLE_NFIX3.IN`                 |
| `FILE_NDEP_IN`   | `NCYCLE_NDEP_HISTSSP_CONSTANT.IN` (not shipped with v7.5.3) |
| `FILE_NRECYC_IN` | `NCYCLE_NRECYC.IN` (read, never used) |
| `FILE_NLOSS_IN`  | `NCYCLE_NLOSS.IN` (read, never used)  |

## Algorithm

`nitrogen_init` copies the `NCYCLE_*_INITIAL` values over the carbon cycle's initial pools and NPP
when `NCYCLE_APPLY_CN_INITIAL_VALUES = 1`.
Each year, inside `TERRCARBON2` after the unlimited NPP is known,
`MAGICC7.f90` calls `N_CALC_LIMITATION_FACTOR`:

1. Before `CO2_FERTILIZATION_YRSTART`:
    1. Solve `NITROGEN_BALANCE_INITIAL` for $N$, the [balance](#nitrogen-pool-balance) with $\rho = 1$.
    2. Set $L = 1$.
    3. Store $N$ and the unlimited NPP as the reference $N_0$, $NPP_0$.
       The last year before the start year is the one that counts.
2. From `CO2_FERTILIZATION_YRSTART` on:
    1. Solve `NITROGEN_BALANCE` for $N$.
       It differs from `NITROGEN_BALANCE_INITIAL` only in multiplying uptake,
       and the NPP in prognostic fixation, by $\rho(N)$.
    2. Set $L = \rho(N)$.
3. Write the diagnostic fluxes at the solved $N$.

`TERRCARBON2` then multiplies NPP by $L$ before distributing it to the pools.

Both balance functions are solved by bisection on $[-100, 1000]$ Gt N.
Each iteration halves the bracket, keeping the half whose ends differ in sign
(the lower half when the product is zero),
and stops when the bracket is at most $10^{-5}$ Gt N wide or after 200 iterations.
The root is the midpoint of the final bracket.

## Numerical considerations

The bisection makes the coupled run chaotic, in MAGICC7 as in RSCM.
The pool lands on a grid of $1100/2^{27} \approx 8.2 \times 10^{-6}$ Gt N.
A rounding-level change in the inputs flips the pool by one grid step,
which feeds back through NPP, CO2 and temperature.
A relative change of $10^{-13}$ in one year's fossil CO2 emissions moves 2100 CO2 by 0.03 ppm
and temperature by $3 \times 10^{-5}$ K.
RSCM matches MAGICC7 only because it repeats MAGICC7's floating-point operations closely enough
to land on the same grid steps.

The pool is a steady state each year, so it cannot accumulate or draw down over time.

## Issues in MAGICC7

### NH3 double count

When deposition comes from emissions, `nitrogen_limitation.f90` adds `DAT_NH3I_EMIS` twice
where the second term should be `DAT_NH3B_EMIS`
(lines 106-107, 156-157, 326-327 and 388-389):

```fortran
+ DAT_NH3I_EMIS % DATGLOBE(CURRENT_YEAR_IDX) &
+ DAT_NH3I_EMIS % DATGLOBE(CURRENT_YEAR_IDX))
```

Industrial NH3 is counted twice and biomass-burning NH3 is left out.
RSCM keeps the bug for parity.

### Other issues

- `NCYCLE_NFIX_VMAX`, `NCYCLE_NFIX_KM` and `NCYCLE_RECYC2CNRATIO_FACTOR`
  are absent from `MAGCFG_DEFAULTALL.CFG`.
  MAGICC7 v7.5.3 runs with them at 0, as its recorded run parameters show.
  The parity test confirms the recycling factor.
  With $V_{max} = 0$ prognostic fixation is zero.
- The default deposition file `NCYCLE_NDEP_HISTSSP_CONSTANT.IN` is not shipped with v7.5.3,
  so deposition read from file is zero.
- `FILE_NRECYC_IN` and `FILE_NLOSS_IN` are read into `DAT_NCYCLE_NRECYC` and `DAT_NCYCLE_NLOSS`,
  which no calculation uses.
- With `NCYCLE_APPLY_NDEP_FROM_EMIS = 1`, the diagnostic fluxes in `CARBONCYCLE.OUT` leave out deposition,
  so they do not close.
- The tuning file `MAGTUNE_1PCTCO2_CN.CFG` uses names such as `KN_RECYC`, `KN_UPTAKE` and `N_SENS_FACTOR_A`
  that the code does not read.
- The bisection interval admits a negative pool.
  The routine sets a failure flag when the interval does not bracket a root, and no caller checks it.
  It separates roots from singularities with an arbitrary threshold, `ABS(F(ROOT)) < 1.0`.
- The factor $10^{-3}$ converts Mt N (emissions) and Tg N ($V_{max}$) to Gt N without comment.
- Nothing checks that the pool stays positive or that $L$ stays within 0 to 1.
- `NITROGEN_BALANCE` and `NITROGEN_BALANCE_INITIAL` share most of their code,
  and `N_CALC_LIMITATION_FACTOR` recomputes their fluxes for the diagnostics.

## Differences from MAGICC7

- RSCM's diagnostic fluxes include deposition, so they close.
  The pool and limitation factor match MAGICC7.
- A deposition file missing from the inputs gives zero deposition, as in v7.5.3.

## References

| Section                                  | File                     | Lines                   |
| ---------------------------------------- | ------------------------ | ----------------------- |
| Module                                   | `nitrogen_limitation.f90` | 1-396                  |
| Parameter declarations                   | `nitrogen_limitation.f90` | 5-25                   |
| "Not finished yet" warning               | `nitrogen_limitation.f90` | 38                     |
| `nitrogen_init`                          | `nitrogen_limitation.f90` | 58-78                  |
| `NITROGEN_BALANCE_INITIAL`               | `nitrogen_limitation.f90` | 80-130                 |
| `NITROGEN_BALANCE`                       | `nitrogen_limitation.f90` | 132-193                |
| Availability function                    | `nitrogen_limitation.f90` | 165-170                |
| Bisection                                | `nitrogen_limitation.f90` | 195-251                |
| `N_CALC_LIMITATION_FACTOR`               | `nitrogen_limitation.f90` | 259-393                |
| Limitation factor                        | `nitrogen_limitation.f90` | 336-347                |
| Module use                               | `MAGICC7.f90`            | 151, 7061, 9598, 12025  |
| Input file reading                       | `MAGICC7.f90`            | 496-507                 |
| Call and NPP limitation                  | `MAGICC7.f90`            | 7293-7300               |
| `CARBONCYCLE.OUT` output                 | `MAGICC7.f90`            | 10528-10612             |
| Cleanup                                  | `MAGICC7.f90`            | 12179                   |
| Namelist                                 | `allcfgs.f90`            | 291-301                 |
| Datastore                                | `datastore.f90`          | 332                     |
| Defaults and input files                 | `MAGCFG_DEFAULTALL.CFG`  | 201-213, 318-334        |
