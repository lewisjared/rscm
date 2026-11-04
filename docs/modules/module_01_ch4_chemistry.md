# CH4 Chemistry

## Overview

This module calculates the atmospheric lifetime and concentration of methane.
The lifetime is not constant.
It depends on the CH4 burden itself, because more CH4 depletes the OH radicals that remove it,
and on emissions of NOx, CO and VOCs, which change OH.
Warming shortens the lifetime through an optional temperature feedback.

MAGICC7 solves the nonlinear mass balance with Prather's iterative method from the IPCC TAR.
A second part of the module tracks the fossil share of atmospheric CH4,
whose oxidation adds CO2 to fossil emissions.

## RSCM implementation

- **Components**: `CH4Chemistry` and `CH4Oxidation`
  (Python builders `rscm.magicc.CH4ChemistryBuilder` and `rscm.magicc.CH4OxidationBuilder`).
- **Source**: `crates/rscm-magicc/src/chemistry/ch4.rs`
  and `crates/rscm-magicc/src/chemistry/ch4_oxidation.rs`.
- **Status**: Implemented. See the [status table](README.md#rscm-implementation-status).
- **Tests**: `test_coupled_run` and `test_run_magicc7_defaults` in `tests/regression/test_full_run.py`
  check the CH4 concentration, and `test_coupled_run` the CO2 that oxidation adds.
  `tests/test_magicc_full.py` checks the budget-closing natural emissions,
  and `tests/test_magicc_emissions.py::test_ch4_budget_shared_between_chemistry_and_oxidation`
  the coupling between the two components.
  Both Rust files have unit tests, including MAGICC7's first emissions-driven year.

## Equations

### Mass balance

$$
\frac{dB}{dt} = E - \frac{B}{\tau_{OH}} - \frac{B}{\tau_{other}}
$$

- $B$ is the atmospheric CH4 burden (Tg CH4).
- $E$ is total CH4 emissions (Tg CH4/yr).
- $\tau_{OH}$ is the tropospheric OH lifetime (yr).
- $\tau_{other}$ combines the soil, stratospheric and tropospheric chlorine sinks:

$$
\frac{1}{\tau_{other}} = \frac{1}{\tau_{soil}} + \frac{1}{\tau_{strat}} + \frac{1}{\tau_{tropcl}}
$$

The burden is $B = k C$ for a concentration $C$ in ppb,
with $k$ = `CH4_PPB2TGCH4` × `CH4_MIXBOXSIZE`.

### OH feedback

The relative change in OH follows TAR Table 4.11:

$$
\frac{\Delta OH}{OH} = S \frac{\Delta B}{B}
  + A_{NOx} \Delta E_{NOx} + A_{CO} \Delta E_{CO} + A_{VOC} \Delta E_{VOC}
$$

The lifetime responds as $\Delta\tau / \tau = -\gamma \, \Delta OH / OH$,
with $\gamma$ = `CH4_SCALEOHSENS` and $S$ = `CH4_S`.
The OH lifetime over a step is then

$$
\bar{\tau} = U \left( \max\left(1, \frac{\bar{B}}{B_{00}}\right) \right)^X
$$

$$
U = \tau_{00} \exp\left(-\gamma \left(A_{NOx} \Delta E_{NOx}
  + A_{CO} \Delta E_{CO} + A_{VOC} \Delta E_{VOC}\right)\right),
\qquad X = -\gamma S
$$

where $B_{00}$ is the reference burden, $\bar{B}$ the mean burden over the step
and $\tau_{00}$ = `CH4_TAUOH_INIT`.

### Temperature feedback

With `CH4_INCLUDE_TEMPFEEDBACK = 1` the lifetime is adjusted for the warming $\Delta T$:

$$
\bar{\tau}_{T} = \frac{\tau_{00}}{\tau_{00} / \bar{\tau} + \alpha_T \Delta T}
$$

with $\alpha_T$ = `CH4_TAUTEMPSENSITIVITY`.

### Prather's iteration

Four fixed iterations solve the implicit equations.
With $B_0 = B$ and $\Delta B_0 = 0$, iteration $n = 1, \dots, 4$ computes

$$
\begin{aligned}
\bar{B}_n &= \frac{B + B_{n-1}}{2} \\
\bar{\tau}_n &= U \left(\max\left(1, \frac{\bar{B}_n}{B_{00}}\right)\right)^X
  \left(1 - 0.5 X \frac{\Delta B_{n-1}}{B}\right) \\
\Delta B_n &= E - \frac{\bar{B}_n}{\bar{\tau}_n} - \frac{\bar{B}_n}{\tau_{other}} \\
B_n &= B + \Delta B_n
\end{aligned}
$$

The temperature adjustment applies to $\bar{\tau}_n$ in every iteration, before $\Delta B_n$.
The new concentration is $B_4 / k$, and the reported OH lifetime is $\bar{\tau}_4$.

### Conventions verified against MAGICC7 output

Reproducing v7.5.3 SSP245 emissions-driven output fixed these details:

- The NOx, CO and VOC deltas are relative to emissions in `CH4_FEED_YRSTART`, not 1750.
- $B_{00}$ tracks the current burden until the year before `CH4_FEED_YRSTART` and is then fixed.
- The temperature feedback uses $\max(0, T(t) - T(\text{CH4\_FEED\_YRSTART} - 1))$.
- $\tau_{00}$ in the temperature formula is the unadjusted `CH4_TAUOH_INIT`.
- Natural emissions are constant at the budget-closing value (182.19 Tg/yr for the default inputs).

## Inputs and outputs

| Input                       | Units      | Fortran variable                                        |
| --------------------------- | ---------- | ------------------------------------------------------- |
| Previous CH4 concentration  | ppb        | `DAT_CH4_CONC%DATGLOBE(CURRENT_YEAR_IDX)`               |
| Total CH4 emissions         | Tg CH4/yr  | `CH4_TOTEMIS`                                           |
| NOx, CO and VOC deltas      | Mt/yr      | `TROPOZ_DELTA_NOXEMIS`, `_COEMIS`, `_NMVOCEMIS`         |
| Warming since the reference | K          | `CH4_FEED_DELTATEMP`                                    |

Total emissions are the sum of industrial (`DAT_CH4I_EMIS`), biomass burning (`DAT_CH4B_EMIS`)
and natural (`DAT_CH4N_EMIS`) emissions,
plus permafrost (`DAT_CH4PF_EMIS`) and clathrate emissions when they are enabled.

| Output              | Units      | Fortran variable                         |
| ------------------- | ---------- | ---------------------------------------- |
| Next year's CH4     | ppb        | `DAT_CH4_CONC%DATGLOBE(NEXT_YEAR_IDX)`   |
| OH lifetime         | yr         | `DAT_CH4_TAUOH%DATGLOBE`                 |
| Total lifetime      | yr         | `DAT_CH4_TAUTOT%DATGLOBE`                |
| Inverse emissions   | Tg CH4/yr  | `CH4_INVERSE_EMIS`                       |

## Parameters

| Parameter                      | Units            | Default    | Description                                              |
| ------------------------------ | ---------------- | ---------- | -------------------------------------------------------- |
| `CH4_TAUTOT_INIT`              | yr               | 9.9474     | Total lifetime in the budget period                      |
| `CH4_TAUSOIL`                  | yr               | 150.0      | Soil sink lifetime                                       |
| `CH4_TAUSTRAT`                 | yr               | 120.0      | Stratospheric sink lifetime                              |
| `CH4_TAUTROPCL`                | yr               | 200.0      | Tropospheric chlorine sink lifetime                      |
| `CH4_PPB2TGCH4`                | Tg CH4/ppb       | 2.824      | Concentration to mass                                    |
| `CH4_MIXBOXSIZE`               | 1                | 0.973      | Mixing box scaling                                       |
| `CH4_SCALEOHSENS`              | 1                | 0.72448    | $\gamma$, scaling of the OH dependence                   |
| `CH4_S`                        | 1                | -0.53775   | $S$, CH4 feedback on OH                                  |
| `CH4_ANOX`                     | (Mt N/yr)⁻¹      | 0.0093376  | $A_{NOx}$                                                |
| `CH4_ACO`                      | (Mt CO/yr)⁻¹     | -0.000113  | $A_{CO}$                                                 |
| `CH4_AVOC`                     | (Mt NMVOC/yr)⁻¹  | -0.0003142 | $A_{VOC}$                                                |
| `CH4_TAUTEMPSENSITIVITY`       | K⁻¹              | 0.07       | $\alpha_T$                                               |
| `CH4_INCLUDE_TEMPFEEDBACK`     | flag             | 1          | Enables the temperature feedback                         |
| `CH4_TAUFEEDBACK_BYNOXVOCCO`   | flag             | 1          | Enables the NOx, CO and VOC feedback                     |
| `CH4_FEED_YRSTART`             | year             | 1927       | Reference year of the feedbacks                          |
| `CH4_SWITCHFROMCONC2EMIS_YEAR` | year             | 2015       | Last year of prescribed concentrations                   |
| `CH4_BUDGET_AVGYEARS`          | yr               | 10         | Length of the budget period                              |
| `CH4_LASTBUDGETYEAR`           | year             | 2004       | Last year of the budget period                           |
| `CH4_INCL_CH4OX`               | flag             | 1          | Adds CO2 from CH4 oxidation                              |
| `CH4_APPLYFOSSFUELFRACTION`    | flag             | 0          | Uses `CH4_FOSSFUELFRACTION` instead of the source pools  |
| `CH4_FOSSFUELFRACTION`         | 1                | 0.18       | Prescribed fossil fraction                               |
| `CH4_WETLAND_SLOPE`            | Mt CH4/yr/K      | 22.4       | Warming response of wetland emissions                    |
| `CH4_CLATHRATEFEED_APPLY`      | flag             | 0          | Enables clathrate release                                |
| `CH4_CLATHRATEFEED_YRSTART`    | year             | 2010       | First year of clathrate release                          |
| `CH4_CLATHRATEFEED_ALPHA`      | 1/yr             | 0.001      | Release rate of the clathrate pool                       |
| `CH4_CLATHRATEFEED_INIPOOL`    | GtC              | 5000.0     | Initial clathrate pool                                   |

## Algorithm

### Budget initialisation

`methane_calc_budget` derives the OH lifetime from the total lifetime,

$$
\tau_{00} = \left(\frac{1}{\text{CH4\_TAUTOT\_INIT}} - \frac{1}{\tau_{other}}\right)^{-1}
$$

It then closes the budget over the `CH4_BUDGET_AVGYEARS` years ending in `CH4_LASTBUDGETYEAR`,
which is clipped to the last year of the concentration file.
With $\Delta C$ and $\bar{C}$ the annual change and mean of the prescribed concentration,
averaged over those years, the natural emissions are

$$
E_{nat} = k \left( \Delta C + \frac{\bar{C}}{\tau_{00}} + \frac{\bar{C}}{\tau_{other}} \right)
  - \bar{E}_{anthro}
$$

MAGICC7 adds $E_{nat}$ to `DAT_CH4N_EMIS` in every year.

### Each year

The main loop in `MAGICC7.f90` prepares the inputs and calls `METHANE`:

1. Calculates the warming since the reference year, clamped at zero.
2. Calculates the NOx, CO and VOC deltas from `CH4_FEED_YRSTART` emissions,
   when `CH4_TAUFEEDBACK_BYNOXVOCCO` is 1.
3. After `CH4_LASTBUDGETYEAR`, adds the wetland feedback to natural emissions:
   `CH4_WETLAND_SLOPE` times the warming over the mean temperature of the budget years.
4. Sums the emissions, adding permafrost and clathrate emissions when they are enabled.
5. Calls `METHANE`, which runs the four iterations above.
6. Stores the OH lifetime and the total lifetime $(1/\bar{\tau} + 1/\tau_{other})^{-1}$.
7. From `CH4_SWITCHFROMCONC2EMIS_YEAR`, writes the calculated concentration for the next year.

### Prescribed concentrations

Before `CH4_SWITCHFROMCONC2EMIS_YEAR`,
`CH4_INVERSE_EMIS` holds the anthropogenic emissions for which `METHANE` reproduces
the prescribed concentration of the following year.
The fossil CH4 burden that feeds `CH4_2CO2_OXIDISATION_EMIS` then grows from
fossil emissions scaled by `CH4_INVERSE_EMIS` over the anthropogenic total,
fossil plus biomass burning.
RSCM matches both, and `CH4Chemistry` finds the inverse emissions with the secant method.

## Issues in MAGICC7

RSCM reproduces each of these.

- **Fixed iterations**: `METHANE` always runs four iterations, with no convergence check.
- **Floor on the burden ratio**: $\max(1, \bar{B}/B_{00})$ stops the lifetime falling
  below its reference value when CH4 drops below the reference burden.
- **Correction factor**: $(1 - 0.5 X \Delta B_{n-1} / B)$ applies from the second iteration.
  Nothing guards against $B = 0$.
- **Early temperature feedback**: `calculate_feedback_temps` is called with `logic_flag=2`,
  so the feedback starts one year before `CH4_FEED_YRSTART`.
  A comment in the Fortran calls this "unintended logic".
- **`CH4_S`**: the Fortran comment cites -0.32 (TAR Table 4.11) or -0.34 (Table 4.2),
  but the default is -0.53775.
  The comment on `CH4_SCALEOHSENS` says it "SHOULD BE =-1 = -1.145", but the default is 0.72448.

## Differences from MAGICC7

- **Wetland and clathrate feedbacks**: MAGICC7 applies both in `MAGICC7.f90`, not in the methane module.
  RSCM implements neither.
- **Natural emissions**: RSCM takes them as a parameter instead of closing the budget inside the component.
  `rscm.magicc.full.natural_emissions` calculates MAGICC7's values.

## References

MAGICC7 source files, with line numbers:

| Function or section                 | File            | Lines      |
| ----------------------------------- | --------------- | ---------- |
| Module variables                    | `methane.f90`   | 4-20       |
| `METHANE` and its iterations        | `methane.f90`   | 34-132     |
| `methane_calc_budget`               | `methane.f90`   | 136-233    |
| $\tau_{other}$ and $\tau_{00}$      | `methane.f90`   | 157-178    |
| Natural emissions budget            | `methane.f90`   | 205-219    |
| Temperature feedback                | `MAGICC7.f90`   | 3874-3883  |
| Emission deltas                     | `MAGICC7.f90`   | 3948-4002  |
| Wetland feedback                    | `MAGICC7.f90`   | 4006-4015  |
| Total emissions and `METHANE` call  | `MAGICC7.f90`   | 4018-4035  |
| Parameter definitions               | `allcfgs.f90`   | 56-65      |

The defaults are in lines 10 to 38 of `run/MAGCFG_DEFAULTALL.CFG`.

- IPCC Third Assessment Report, Working Group I, chapter 4 (Atmospheric Chemistry and Greenhouse Gases),
  Tables 4.2 and 4.11.
