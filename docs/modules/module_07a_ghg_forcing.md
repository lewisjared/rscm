# Well-Mixed GHG Forcing

## Overview

This module calculates the forcing from the three main well-mixed greenhouse gases:
carbon dioxide (CO2), methane (CH4) and nitrous oxide (N2O).
It also calculates the forcing from stratospheric water vapour produced by CH4 oxidation.
It has no state and no chemistry.
The concentrations come from CH4 and N2O chemistry, the carbon cycle or prescribed inputs.

MAGICC7 offers two methods, selected by `CORE_CO2CH4N2O_RFMETHOD`:

- **IPCCTAR** uses the simplified formulas of the IPCC Third Assessment Report (Myhre et al. 1998).
- **OLBL**, the default, uses polynomial fits to line-by-line calculations based on Etminan et al. (2016).

Both account for the overlap of the CH4 and N2O absorption bands.
OLBL also has CO2 and N2O overlaps and applies rapid adjustment factors.
The adjusted forcing is still the module's RF before efficacies
(see [forcing types](module_07b_rf_aggregation.md#forcing-types)).

## RSCM implementation

- **Component**: `GHGForcing` (Python builder `rscm.magicc.GHGForcingBuilder`).
- **Source**: `crates/rscm-magicc/src/forcing/ghg.rs`, with parameters in
  `crates/rscm-magicc/src/parameters/ghg_forcing.rs`.
  The OLBL formulae live in `crates/rscm-components/src/meinshausen2020.rs`, shared with FaIR.
- **Status**: Implemented. See the [status table](README.md#rscm-implementation-status).
- **Tests**: `test_ch4_oxidation_strat_h2o` in `tests/regression/test_full_run.py`
  checks the stratospheric H2O forcing.
  `test_coupled_run` checks the total ERF with OLBL,
  and its `ipcctar` variant the same with IPCCTAR.
  `ghg.rs` has unit tests for both methods.

The regional split happens in `ForcingAggregation` ([forcing aggregation](module_07b_rf_aggregation.md)),
not in `GHGForcing`.

## Equations

### IPCCTAR

CO2 forcing is logarithmic in concentration:

$$
RF_{CO2} = \frac{\Delta Q_{2\times CO2}}{\ln 2} \ln\left(\frac{C}{C_0}\right)
$$

where $C$ and $C_0$ are the current and pre-industrial CO2 (ppm)
and $\Delta Q_{2\times CO2}$ = `CORE_DELQ2XCO2`.
The default 3.71 W/m² gives a coefficient of about 5.35 W/m².

CH4 and N2O forcing is a square-root term less the change in their overlap:

$$
RF_{CH4} = \beta_{CH4} \left(\sqrt{M} - \sqrt{M_0}\right)
  + 0.47 \ln\left(\frac{1 + f(M_0, N_0)}{1 + f(M, N_0)}\right)
$$

$$
RF_{N2O} = \beta_{N2O} \left(\sqrt{N} - \sqrt{N_0}\right)
  + 0.47 \ln\left(\frac{1 + f(M_0, N_0)}{1 + f(M_0, N)}\right)
$$

where $M$ and $N$ are CH4 and N2O (ppb), the subscript 0 marks pre-industrial values,
and $\beta_{CH4}$ and $\beta_{N2O}$ are `CH4_RADEFF_WM2PERPPB` and `N2O_RADEFF_WM2PERPPB`.
The overlap function takes the concentrations in ppm, that is ppb divided by 1000:

$$
f(m, n) = 0.6356 (m n)^{0.75} + 0.007 \, m (m n)^{1.52}
$$

These constants are the TAR coefficients $2.01 \times 10^{-5}$ and $5.31 \times 10^{-15}$
rescaled from ppb to ppm.
The Fortran names the rescaled variables `_ppt`, but they hold ppm.

### OLBL

CO2 forcing has a concentration-dependent coefficient that saturates above $C_{max}$:

$$
RF_{CO2} = f_{CO2} \, \alpha_{CO2} \ln\left(\frac{C}{C_0}\right)
$$

$$
\alpha_{CO2} = c_1 \sqrt{N} +
\begin{cases}
d_1 & C \le C_0 \\
a_1 (C - C_0)^2 + b_1 (C - C_0) + d_1 & C_0 < C < C_{max} \\
d_1 - \dfrac{b_1^2}{4 a_1} & C \ge C_{max}
\end{cases}
$$

$$
C_{max} = \frac{2 a_1 C_0 - b_1}{2 a_1}
$$

$C_{max}$ is the vertex of the quadratic, about 1810 ppm with the default coefficients and $C_0$ = 278 ppm.
CH4 and N2O forcing have coefficients that depend on the other gases:

$$
RF_{CH4} = f_{CH4} \left(a_3 \sqrt{M} + b_3 \sqrt{N} + d_3\right) \left(\sqrt{M} - \sqrt{M_0}\right)
$$

$$
RF_{N2O} = f_{N2O} \left(a_2 \sqrt{C} + b_2 \sqrt{N} + c_2 \sqrt{M} + d_2\right)
  \left(\sqrt{N} - \sqrt{N_0}\right)
$$

$f_{CO2}$, $f_{CH4}$ and $f_{N2O}$ are the rapid adjustment factors `CORE_RFRAPIDADJUST_*`.

### Stratospheric H2O from CH4 oxidation

$$
RF_{strH2O} = RF_{CH4,pure} \cdot \text{CH4\_ADDEDSTRATH2O\_PERCENT}
$$

$RF_{CH4,pure}$ is the CH4 forcing without the N2O overlap.
IPCCTAR drops the overlap term, leaving $\beta_{CH4} (\sqrt{M} - \sqrt{M_0})$.
OLBL evaluates the CH4 formula, rapid adjustment included, at pre-industrial N2O.

### 2xCO2 effective radiative forcing

With OLBL, MAGICC7 recalculates `CORE_DELQ2XCO2` every year as
$f_{CO2} \, \alpha_{CO2} \ln 2$ at the current concentrations,
so the climate feedback parameter changes as CO2 rises.
With IPCCTAR it keeps the configured value.
`GHGForcing` reports this as `2xCO2 Effective Radiative Forcing`, which `ClimateUDEB` reads.

## Inputs and outputs

| Input              | Unit  | Fortran variable                         |
| ------------------ | ----- | ---------------------------------------- |
| CO2 concentration  | ppm   | `DAT_CO2_CONC%DATGLOBE(NEXT_YEAR_IDX)`   |
| CH4 concentration  | ppb   | `DAT_CH4_CONC%DATGLOBE(NEXT_YEAR_IDX)`   |
| N2O concentration  | ppb   | `DAT_N2O_CONC%DATGLOBE(NEXT_YEAR_IDX)`   |
| Pre-industrial CO2, CH4 and N2O | ppm, ppb | `DAT_*_CONC%PREIND_DATGLOBE` |

Forcing is calculated for the end of the step, so it reads the next year's concentrations.
`GHGForcing` reads the end of the step too.

| Output                 | Unit  | Fortran variable                                   |
| ---------------------- | ----- | -------------------------------------------------- |
| CO2 forcing            | W/m²  | `DAT_CO2_RF%DATGLOBE(NEXT_YEAR_IDX)`               |
| CH4 forcing            | W/m²  | `DAT_CH4_RF%DATGLOBE(NEXT_YEAR_IDX)`               |
| N2O forcing            | W/m²  | `DAT_N2O_RF%DATGLOBE(NEXT_YEAR_IDX)`               |
| Strat H2O forcing      | W/m²  | `DAT_CH4OXSTRATH2O_RF%DATGLOBE(NEXT_YEAR_IDX)`     |
| Box forcing            | W/m²  | `DAT_*_RF%DATBOX(NEXT_YEAR_IDX,:)`                 |

## Parameters

| Parameter                   | Unit           | Default     | Description                                          |
| --------------------------- | -------------- | ----------- | ---------------------------------------------------- |
| `CORE_CO2CH4N2O_RFMETHOD`   | string         | `OLBL`      | `IPCCTAR` or `OLBL`                                  |
| `CORE_DELQ2XCO2`            | W/m²           | 3.71        | 2xCO2 effective radiative forcing for IPCCTAR        |
| `CH4_RADEFF_WM2PERPPB`      | W/m²/ppb^0.5   | 0.036       | $\beta_{CH4}$ for IPCCTAR                            |
| `N2O_RADEFF_WM2PERPPB`      | W/m²/ppb^0.5   | 0.12        | $\beta_{N2O}$ for IPCCTAR                            |
| `CORE_OLBL_CO2_A1`          | W/m²/ppm²      | -2.4785e-07 | $a_1$, quadratic term                                |
| `CORE_OLBL_CO2_B1`          | W/m²/ppm       | 0.00075906  | $b_1$, linear term                                   |
| `CORE_OLBL_CO2_C1`          | W/m²/ppb^0.5   | -0.0021492  | $c_1$, N2O overlap                                   |
| `CORE_OLBL_CO2_D1`          | W/m²           | 5.2         | $d_1$, constant                                      |
| `CORE_OLBL_CH4_A3`          | W/m²/ppb       | -8.9603e-05 | $a_3$, CH4 term                                      |
| `CORE_OLBL_CH4_B3`          | W/m²/ppb       | -0.00012462 | $b_3$, N2O overlap                                   |
| `CORE_OLBL_CH4_D3`          | W/m²/ppb^0.5   | 0.045       | $d_3$, constant                                      |
| `CORE_OLBL_N2O_A2`          | W/m²/ppm^0.5/ppb^0.5 | -0.00034197 | $a_2$, CO2 overlap                         |
| `CORE_OLBL_N2O_B2`          | W/m²/ppb       | 0.00025455  | $b_2$, N2O term                                      |
| `CORE_OLBL_N2O_C2`          | W/m²/ppb       | -0.00024357 | $c_2$, CH4 overlap                                   |
| `CORE_OLBL_N2O_D2`          | W/m²/ppb^0.5   | 0.14        | $d_2$, constant                                      |
| `CORE_RFRAPIDADJUST_CO2`    | 1              | 1.05        | $f_{CO2}$, OLBL only                                 |
| `CORE_RFRAPIDADJUST_CH4`    | 1              | 0.86        | $f_{CH4}$, OLBL only                                 |
| `CORE_RFRAPIDADJUST_N2O`    | 1              | 1.0         | $f_{N2O}$, OLBL only                                 |
| `CH4_ADDEDSTRATH2O_PERCENT` | 1              | 0.0923      | Fraction of pure CH4 forcing added as strat H2O      |
| `CO2_PREINDCO2CONC`         | ppm            | 278.0       | Pre-industrial CO2 used with the next switch         |
| `CO2_PREINDCO2CONC_APPLY`   | flag           | 0           | Replaces the file's pre-industrial CO2               |

The rapid adjustment factors follow Smith et al. (2018) and the IPCC AR6 WG1 chapter 7 drafts.
The regional patterns `RF_REGIONS_CO2`, `RF_REGIONS_CH4`, `RF_REGIONS_N2O`
and `RF_REGIONS_CH4OXSTRATH2O` belong to the aggregation,
see [forcing aggregation](module_07b_rf_aggregation.md).

## Algorithm

Each year `DELTAQ` in `MAGICC7.f90`:

1. In the first year, replaces the pre-industrial CO2 with `CO2_PREINDCO2CONC`
   when `CO2_PREINDCO2CONC_APPLY` is 1.
2. Reads the next year's CO2, CH4 and N2O and their pre-industrial values.
3. Calculates the three forcings with the method `CORE_CO2CH4N2O_RFMETHOD` selects,
   and stops with an error for any other value.
4. Calculates the pure CH4 forcing for stratospheric H2O,
   without the overlap for IPCCTAR and at pre-industrial N2O for OLBL.
5. In the first year, repeats steps 3 and 4 at the first year's concentrations to get the first-year offset.
   With the default `JUMPSTART` this offset is zero,
   see [pre-industrial reference](module_07b_rf_aggregation.md#pre-industrial-reference).
6. Splits each global forcing over the four boxes with its `RF_REGIONS_*` pattern,
   normalised by the box area fractions, and stores global and box values.
7. Multiplies the pure CH4 forcing by `CH4_ADDEDSTRATH2O_PERCENT` and stores it the same way.

`GHGForcing` covers steps 2 to 4 and 7, with global outputs only.
`ForcingAggregation` does the regional split of step 6.

## Numerical considerations

Neither MAGICC7 nor `GHGForcing` checks for zero or negative concentrations
before taking logarithms and square roots.

## Issues in MAGICC7

RSCM reproduces each of these.

- **Pure CH4 forcing**: the two methods remove the N2O overlap differently, as above.
- **No CO2 term in OLBL CH4**: the CH4 coefficient has $a_3$, $b_3$ and $d_3$ but no CO2 term,
  unlike the N2O coefficient.
- **CO2 saturation**: above $C_{max}$ the OLBL coefficient stays at its maximum,
  so CO2 forcing still grows logarithmically, with a fixed slope.

## Differences from MAGICC7

- **Pre-industrial concentrations**: MAGICC7 takes them from the concentration files.
  `GHGForcing` takes `co2_pi`, `ch4_pi` and `n2o_pi` as parameters,
  and `run_magicc7` sets them to MAGICC7's 1750 concentrations.
  There is no `CO2_PREINDCO2CONC_APPLY` switch.
- **Rapid adjustment with IPCCTAR**: MAGICC7 applies `CORE_RFRAPIDADJUST_*` only with OLBL.
  `GHGForcing` applies its `adjust_*` factors with either method,
  so `run_magicc7` sets them to 1 for IPCCTAR.

## References

MAGICC7 source files, with line numbers:

| Function or section             | File                                        | Lines           |
| ------------------------------- | ------------------------------------------- | --------------- |
| IPCCTAR forcing                 | `src/libmagicc/physics/deltaq_calculations.f90` | 1031-1087   |
| IPCCTAR CH4 and N2O helper      | `src/libmagicc/physics/deltaq_calculations.f90` | 1089-1106   |
| IPCCTAR overlap                 | `src/libmagicc/physics/deltaq_calculations.f90` | 1108-1147   |
| OLBL forcing                    | `src/libmagicc/physics/deltaq_calculations.f90` | 1149-1276   |
| OLBL CO2 coefficient            | `src/libmagicc/physics/deltaq_calculations.f90` | 1278-1318   |
| Stratospheric H2O               | `src/libmagicc/physics/deltaq_calculations.f90` | 1320-1335   |
| `DELTAQ` call site              | `src/libmagicc/MAGICC7.f90`                 | 5139-5440       |
| Pre-industrial CO2 override     | `src/libmagicc/MAGICC7.f90`                 | 5143-5154       |
| Method selection                | `src/libmagicc/MAGICC7.f90`                 | 5157, 5203      |
| Storage of forcing              | `src/libmagicc/MAGICC7.f90`                 | 5322-5408       |
| Parameter declarations          | `src/libmagicc/physics/climate_and_ocean.f90` | 16-39         |
| Parameter namelist              | `src/libmagicc/allcfgs.f90`                 | 97-109          |
| Defaults                        | `run/MAGCFG_DEFAULTALL.CFG`                 | 86-124, 308     |

- Myhre, G. et al. (1998). New estimates of radiative forcing due to well mixed greenhouse gases.
  *Geophysical Research Letters*.
- Etminan, M. et al. (2016). Radiative forcing of carbon dioxide, methane, and nitrous oxide:
  a significant revision of the methane radiative forcing.
  *Geophysical Research Letters*.
- Smith, C. J. et al. (2018). Understanding rapid adjustments to diverse forcing agents.
  *Geophysical Research Letters*.
