# N2O Chemistry

## Overview

This module calculates the atmospheric concentration and lifetime of nitrous oxide.
N2O is removed in the stratosphere, mainly by photolysis and reaction with O(¹D),
and its lifetime is about 139 years.

Three effects modify the lifetime:

- A power law in the burden.
  With the default exponent the sink becomes more efficient as N2O rises, so the lifetime falls.
- A delay for tropospheric N2O to reach the stratosphere,
  so the sink acts on the concentration of earlier years.
- An optional faster Brewer-Dobson circulation (meridional flux) as the climate warms,
  which also shortens the lifetime.

## RSCM implementation

- **Component**: `N2OChemistry` (Python builder `rscm.magicc.N2OChemistryBuilder`).
- **Source**: `crates/rscm-magicc/src/chemistry/n2o.rs`.
- **Status**: Implemented. See the [status table](README.md#rscm-implementation-status).
- **Tests**: `test_coupled_run` and `test_run_magicc7_defaults` in `tests/regression/test_full_run.py`
  check the N2O concentration.
  `tests/test_magicc_full.py::test_natural_emissions_close_budget` checks the budget-closing
  natural emissions.
  The Rust unit tests in `n2o.rs` include MAGICC7's first emissions-driven year.

## Equations

### Mass balance

$$
\frac{dB}{dt} = E - \frac{\bar{B}_{delayed}}{\tau}
$$

- $B = k C$ is the N2O burden (Tg N) for a concentration $C$ in ppb, with $k$ = `N2O_PPB2TGN`.
- $E$ is total N2O emissions (Tg N/yr).
- $\bar{B}_{delayed}$ is the burden averaged over the delayed years (see below), not the current burden.
- $\tau$ is the lifetime (yr).

### Lifetime

$$
\tau = s \, \tau_{init} \left( \max\left(1, \frac{\bar{B}}{B_{00}}\right) \right)^S
$$

- $\tau_{init}$ = `N2O_TAUINIT`.
- $\bar{B}$ is the estimate of the mean burden over the step.
- $B_{00}$ is the reference burden at the feedback reference year.
- $S$ = `N2O_S`.
  With the default -0.04 the factor is below one whenever $\bar{B} > B_{00}$,
  so the lifetime falls as the burden rises.
  At $\bar{B} = 1.1 B_{00}$ it is $1.1^{-0.04} \approx 0.996$.

### Meridional flux scaling

With `N2O_USE_TAUSTRAT_VAR = 1`, warming shortens the lifetime through

$$
s = \frac{1}{1 + \Delta T_{merid} \, k_{merid} \, \alpha_{N2O}}
$$

where $k_{merid}$ = `GEN_MERIDFLUX_CHNGPERDEG` and $\alpha_{N2O}$ = `N2O_TAUSTRAT_SENS2MERIDFLUX`.
Otherwise $s = 1$.

### Stratospheric mixing delay

The sink uses the mean of two delayed concentrations, with $d$ = `N2O_STRATMIXDELAY`:

$$
\bar{B}_{delayed} = k \, \frac{C_{t-d} + C_{t-d-1}}{2}
$$

Indices before the first year of the run are clamped to the first year.

### Iteration

`NITROUS` runs four fixed iterations from $B_0 = B$:

$$
\begin{aligned}
\bar{B}_n &= \frac{B + B_{n-1}}{2} \quad (\bar{B}_1 = B) \\
\tau_n &= s \, \tau_{init} \left(\max\left(1, \frac{\bar{B}_n}{B_{00}}\right)\right)^S \\
B_n &= B + E - \frac{\bar{B}_{delayed}}{\tau_n}
\end{aligned}
$$

The new concentration is $B_4 / k$ and the reported lifetime is $\tau_4$.
Unlike CH4 there is no correction factor on the lifetime.

### Conventions verified against MAGICC7 output

Reproducing v7.5.3 SSP245 emissions-driven output fixed these details:

- $B_{00}$ tracks the current burden until the year before `N2O_FEED_YRSTART` and is then fixed.
- $\Delta T_{merid}$ is $\max(0, T(t) - T(\text{GEN\_CHANGE\_MERIDIONALFLUX\_YR}))$, zero before that year.
- Natural emissions are constant at the budget-closing value (8.0345 Tg N/yr for the default inputs).

## Inputs and outputs

| Input                          | Unit      | Fortran variable                                   |
| ------------------------------ | --------- | -------------------------------------------------- |
| Current N2O concentration      | ppb       | `DAT_N2O_CONC%DATGLOBE(CURRENT_YEAR_IDX)`          |
| Delayed concentrations         | ppb       | `DAT_N2O_CONC%DATGLOBE(DELAY_ENDYEAR_IDX)` and `(DELAY_STARTYEAR_IDX)` |
| Industrial emissions           | Tg N/yr   | `DAT_N2OI_EMIS%DATGLOBE`                           |
| Biomass burning emissions      | Tg N/yr   | `DAT_N2OB_EMIS%DATGLOBE`                           |
| Natural emissions              | Tg N/yr   | `DAT_N2ON_EMIS%DATGLOBE`                           |
| Lifetime scale $s$             | 1         | `SCALEFACTOR_TAUSTRAT`                             |

| Output                     | Unit      | Fortran variable                          |
| -------------------------- | --------- | ----------------------------------------- |
| Next year's N2O            | ppb       | `DAT_N2O_CONC%DATGLOBE(NEXT_YEAR_IDX)`    |
| Lifetime                   | yr        | `DAT_N2O_TAUTOT%DATGLOBE`                 |
| Inverse emissions          | Tg N/yr   | `N2O_INVERSE_EMIS`                        |
| Total emissions            | Tg N/yr   | `DAT_N2OT_EMIS%DATGLOBE`                  |

## Parameters

| Parameter                      | Unit      | Default  | Description                                          |
| ------------------------------ | --------- | -------- | ---------------------------------------------------- |
| `N2O_TAUINIT`                  | yr        | 139.275  | Reference lifetime                                   |
| `N2O_PPB2TGN`                  | Tg N/ppb  | 4.79     | Concentration to mass                                |
| `N2O_S`                        | 1         | -0.04    | $S$, exponent of the burden feedback                 |
| `N2O_STRATMIXDELAY`            | yr        | 1        | $d$, delay before N2O reaches the stratosphere       |
| `N2O_FEED_YRSTART`             | year      | 1925     | Reference year of the burden feedback                |
| `N2O_USE_TAUSTRAT_VAR`         | flag      | 1        | Enables the meridional flux scaling                  |
| `N2O_TAUSTRAT_SENS2MERIDFLUX`  | 1         | 0.04     | $\alpha_{N2O}$                                       |
| `GEN_MERIDFLUX_CHNGPERDEG`     | 1/K       | 0.15     | $k_{merid}$, shared with the halocarbons             |
| `GEN_CHANGE_MERIDIONALFLUX_YR` | year      | 1980     | Reference year of the meridional flux warming        |
| `N2O_SWITCHFROMCONC2EMIS_YEAR` | year      | 2015     | Last year of prescribed concentrations               |
| `N2O_BUDGET_AVGYEARS`          | yr        | 10       | Length of the budget period                          |
| `N2O_LASTBUDGETYEAR`           | year      | 1991     | Last year of the budget period                       |
| `N2O_APPLY_SCALEEMIS`          | flag      | 0        | Enables `N2O_SCALEEMIS`                              |
| `N2O_SCALEEMIS`                | 1         | 1.0      | Scale on anthropogenic emissions                     |

## Algorithm

### Budget initialisation

During initialisation MAGICC7 closes the budget over the `N2O_BUDGET_AVGYEARS` years
ending in `N2O_LASTBUDGETYEAR`, which is clipped to the concentration file.
With $\Delta C$ and $\bar{C}$ the change and mean of the prescribed concentration between consecutive years,
and $\bar{E}_{anthro}$ the mean of consecutive years' anthropogenic emissions,
each averaged over the budget period, the natural emissions are

$$
E_{nat} = k \left( \Delta C + \frac{\bar{C}}{\tau_{init}} \right) - \bar{E}_{anthro}
$$

MAGICC7 adds $E_{nat}$ to `DAT_N2ON_EMIS` in every year.
[CH4 chemistry](module_01_ch4_chemistry.md#budget-initialisation) closes the CH4 budget the same way,
with its two lifetimes.

### Each year

1. Updates the reference concentration $C_{00}$ until the year before `N2O_FEED_YRSTART`.
2. Calculates the delayed indices, clamped to the first year.
3. Sums industrial, biomass burning and natural emissions.
4. Calculates $s$ from the warming since `GEN_CHANGE_MERIDIONALFLUX_YR`.
5. Calls `NITROUS`, which runs the four iterations above.
6. Stores the lifetime,
   and from `N2O_SWITCHFROMCONC2EMIS_YEAR` the calculated concentration for the next year.
   Before then the prescribed concentration is kept.
7. Diagnoses the inverse emissions as $k (C_{t+1} - C_t + \bar{C}_{delayed} / \tau) - E_{nat}$,
   with $\bar{C}_{delayed} = \bar{B}_{delayed} / k$.

## Issues in MAGICC7

RSCM reproduces each of these.

- **Fixed iterations**: `NITROUS` always runs four iterations, with no convergence check.
- **Delayed sink, current lifetime**: the sink divides the delayed burden $\bar{B}_{delayed}$
  by a lifetime evaluated at the current mid-step burden $\bar{B}_n$.
- **Floor on the burden ratio**: $\max(1, \bar{B}/B_{00})$ keeps the lifetime at $s \, \tau_{init}$
  when N2O is below the reference burden.
- **Global only**: N2O has no regional structure, and MAGICC7 copies the global value to the boxes.

## Differences from MAGICC7

- **Natural emissions**: RSCM takes them as a parameter instead of closing the budget inside the component.
  `rscm.magicc.full.natural_emissions` calculates MAGICC7's values.
- **Inverse emissions**: `N2OChemistry` does not output them.

## References

MAGICC7 source files, with line numbers:

| Function or section                         | File                                          | Lines      |
| ------------------------------------------- | --------------------------------------------- | ---------- |
| Module variables                            | `src/libmagicc/physics/n2o.f90`               | 3-12       |
| `NITROUS` and its iterations                | `src/libmagicc/physics/n2o.f90`               | 19-68      |
| N2O allocation                              | `src/libmagicc/MAGICC7.f90`                   | 566-574    |
| Budget initialisation                       | `src/libmagicc/MAGICC7.f90`                   | 1606-1660  |
| Feedback reference concentration            | `src/libmagicc/MAGICC7.f90`                   | 3907-3919  |
| Delay indices, emissions and scaling        | `src/libmagicc/MAGICC7.f90`                   | 4352-4367  |
| `NITROUS` call and concentration switch     | `src/libmagicc/MAGICC7.f90`                   | 4372-4394  |
| Inverse emissions                           | `src/libmagicc/MAGICC7.f90`                   | 4406-4408  |
| `calculate_feedback_conc`                   | `src/libmagicc/physics/deltaq_calculations.f90` |          |
| Parameter definitions                       | `src/libmagicc/allcfgs.f90`                   |            |
| Defaults                                    | `run/MAGCFG_DEFAULTALL.CFG`                   | 302-315    |
