# Halocarbon Chemistry

## Overview

This module calculates the concentrations and forcing of halocarbons,
synthetic gases of carbon with fluorine, chlorine or bromine.
MAGICC7 splits them into two groups:

- **F-gases** (`FGAS`): the HFCs, PFCs, SF6, NF3 and SO2F2 of the Kyoto Protocol.
  They are greenhouse gases that do not deplete ozone.
- **Montreal Protocol gases** (`MHALO`): the CFCs, HCFCs, halons, CCl4, CH3CCl3, CH3Cl and CH3Br.
  They deplete stratospheric ozone as well as adding forcing.

Each species decays with its own lifetime, which can vary with OH and with warming.
Forcing is linear in concentration.
The module also sums the chlorine and bromine of the Montreal Protocol gases into
equivalent effective stratospheric chlorine (EESC),
from which [ozone](module_04_ozone.md#stratospheric-ozone) calculates stratospheric ozone forcing.

## RSCM implementation

- **Component**: `HalocarbonChemistry` (Python builder `rscm.magicc.HalocarbonChemistryBuilder`).
- **Source**: `crates/rscm-magicc/src/chemistry/halocarbon.rs`,
  with the species defaults in `crates/rscm-magicc/src/parameters/halocarbon.rs`.
- **Status**: Implemented. See the [status table](README.md#rscm-implementation-status).
- **Tests**: `test_halocarbons` in `tests/regression/test_full_run.py` runs all 41 species from emissions
  and checks their concentrations, the F-gas and Montreal Protocol forcing, and EESC.
  `test_default_run` checks every concentration in MAGICC7's default run.
  Both Rust files have unit tests.

`HalocarbonChemistry` outputs each species' forcing as `Effective Radiative Forcing|F-Gases|<species>`
or `Effective Radiative Forcing|Montreal Protocol Halogen Gases|<species>`,
which is MAGICC7's RF before efficacies (see [forcing types](module_07b_rf_aggregation.md#forcing-types)).

## Equations

### Concentrations

$$
\frac{dC}{dt} = k E - \frac{C}{\tau}
$$

$C$ is the concentration (ppt), $E$ the emissions (kt/yr) and $\tau$ the effective lifetime (yr).
The conversion factor is

$$
k = \frac{M_{air}}{M_{atm} \, M_{mol} \, f_{mix}}
$$

with $M_{air}$ = `GEN_AIR_GRAMMPROMOL`, $M_{atm}$ = `GEN_ATM_TOTMASS_1E21GRAMM` (in $10^{21}$ g),
$M_{mol}$ the molar mass of the species and $f_{mix}$ = `*_EFF_MIXBOXSIZE` (0.949 for every species).
Each year uses one of two schemes, with the emissions and lifetime of the start of the year:

$$
C_{t+1} =
\begin{cases}
\dfrac{C_t \left(1 - \frac{1}{2\tau}\right) + k E_t}{1 + \frac{1}{2\tau}}
  & \tau \ge 5 \text{ (Crank-Nicolson)} \\[2ex]
\tau k E_t \left(1 - e^{-1/\tau}\right) + C_t e^{-1/\tau} & \tau < 5 \text{ (exact exponential)}
\end{cases}
$$

### Molar mass

`CALC_HALOS_MOLMASS` sums the atomic masses in this order:

$$
M_{mol} = 12.001 n_C + 35.453 n_{Cl} + 18.9984 n_F + 79.904 n_{Br}
  + 1.0079 n_H + 32.06 n_S + 15.999 n_O + 14.007 n_N
$$

### Lifetime

The effective lifetime combines the OH, stratospheric and other sinks, added in this order:

$$
\frac{1}{\tau} = \left( \frac{1}{s_{OH} \tau_{OH}} + \frac{1}{s_{strat} \tau_{strat}} \right)
  + \frac{1}{\tau_{other}},
\qquad
\frac{1}{\tau_{other}} = \frac{1}{\tau_{tot}} - \frac{1}{\tau_{OH}} - \frac{1}{\tau_{strat}}
$$

A term is left out when its lifetime is zero, so $\tau = \tau_{tot}$ when both scale factors are 1.
The scale factors are

$$
s_{OH} = \frac{\tau_{OH}^{CH4}(t)}{\tau_{OH,init}^{CH4}},
\qquad
s_{strat} = \frac{1}{1 + \Delta T \cdot \text{GEN\_MERIDFLUX\_CHNGPERDEG}
  \cdot \text{*\_TAUSTRAT\_SENS2MERIDFLUX}}
$$

$s_{OH}$ uses the CH4 OH lifetime from [CH4 chemistry](module_01_ch4_chemistry.md) as a proxy for OH.
$\Delta T$ is the warming since `GEN_CHANGE_MERIDIONALFLUX_YR` (1980), clamped at zero,
because warming speeds up the stratospheric circulation.
Butchart and Scaife (2001) find the meridional flux rises by about 3% per decade.
`*_USE_TAUOH_VAR` and `*_USE_TAUSTRAT_VAR` switch the two factors.

### Forcing

$$
RF = (C - C_{pi}) \left( \frac{\eta}{1000} \right)
$$

$\eta$ is the radiative efficiency in W/m² per ppb, so $\eta / 1000$ is per ppt.
MAGICC7 divides before it multiplies.
The group totals add the species in the order of the tables below.

### Regional pattern of F-gas forcing

`calculate_halo_rf_box_fractions` spreads each F-gas over the four
[boxes](module_00_initialisation.md), so that short-lived species sit where they are emitted.
With $w$ = `RF_REGIONS_FGAS` and box areas $a$, the global and hemispheric patterns are

$$
g_i = \frac{w_i}{\sum_j w_j a_j},
\qquad
l_i = \frac{g_i}{h_H} \cdot \frac{E_H}{E_{tot}},
\qquad
h_H = a_{O} g_{O} + a_{L} g_{L}
$$

where $h_H$ is the area mean of $g$ over the box's hemisphere $H$ and its ocean and land boxes.
$E$ are the box fossil CO2 emissions of the previous year,
which MAGICC7 splits half to each hemisphere's land box.
$E_H$ is the hemisphere's total and $E_{tot}$ the sum over boxes.
The box fraction for a species with total lifetime $\tau$ is then:

- $g_i$ for $\tau$ above `FGAS_RF_REGIONSCALING_UPPERTAU`, or when $E_{tot}$ is 0.
- $l_i$ for $\tau$ below `FGAS_RF_REGIONSCALING_LOWERTAU`.
- $(g_i - l_i) \frac{\tau - \tau_{lower}}{\tau_{upper} - \tau_{lower}} + l_i$ in between.

The `FGASSUM` boxes add the species in order,
but the reported global `FGASSUM` is the sum of species forcing rather than the area mean of the boxes.

### EESC

$$
EESC = \sum_{species} C_{t-d}
  \left( \alpha_{Br} \left( (n_{Br} f) f_{11} \right) + \left( (n_{Cl} f) f_{11} \right) \right)
$$

The bracketing is MAGICC7's order of operations.

- $C_{t-d}$ is the concentration $d$ = `GEN_EESC_STRATMIXDELAY` years before the year being calculated,
  clamped to the first year of the run.
- $n_{Cl}$ and $n_{Br}$ are the chlorine and bromine atoms per molecule.
- $f$ is the species' fractional release factor (`*_FRACT_RELEASEFACTOR`), zero for every F-gas.
- $f_{11}$ = `MHALO_RELEASEFACTORC11` normalises to CFC-11.
- $\alpha_{Br}$ = `STRATOZ_BR_VS_CL_SCALE` is the ozone destruction by bromine relative to chlorine.

## Inputs and outputs

| Input                    | Unit   | Fortran variable                              |
| ------------------------ | ------ | --------------------------------------------- |
| Species emissions        | kt/yr  | `DAT_FGAS_EMIS(i)`, `DAT_MHALO_EMIS(i)`       |
| Prescribed concentrations | ppt   | `DAT_FGAS_CONC(i)`, `DAT_MHALO_CONC(i)`, up to the switch year |
| CH4 OH lifetime          | yr     | `CH4_TAUOH_EFFECTIVE`                         |
| Warming                  | K      | `TEMP_MERIDIONALFLUX`                         |
| Fossil CO2 emissions     | GtC/yr | `DAT_CO2I_EMIS`, for the F-gas pattern        |

| Output                         | Unit       | Fortran variable                         |
| ------------------------------ | ---------- | ---------------------------------------- |
| Species concentrations         | ppt        | `DAT_FGAS_CONC(i)`, `DAT_MHALO_CONC(i)`  |
| Species forcing                | W/m²       | `DAT_FGAS_RF(i)`, `DAT_MHALO_RF(i)`      |
| Group forcing                  | W/m²       | `DAT_FGASSUM_RF`, `DAT_MHALOSUM_RF`      |
| EESC                           | ppt        | `DAT_EESC_CONC`                          |
| Stratospheric Cl and Br        | ppt        | `DAT_ESC_CONC`, `DAT_ESBR_CONC`          |
| Lifetimes                      | yr         | `DAT_*_TAUTOT(i)`, `_TAUOH(i)`, `_TAUSTRAT(i)` |
| Inverse emissions              | kt/yr      | `DAT_*_INVERSEEMIS(i)`                   |
| CFC-12 and HFC-134a equivalents | ppt       | `DAT_MHALOSUMCFC12EQ_CONC`, `DAT_FGASSUMHFC134AEQ_CONC` |

## Parameters

### Species

`FGAS_N` and `MHALO_N` set the number of species, up to 30 in each group.
The tables give MAGICC7's defaults, with RSCM's species names.
$C_{pi}$ is the 1750 value of MAGICC7's concentration file, which RSCM uses as the initial concentration.
GWP-100 is `*_METRICVALS`, which only MAGICC7's GWP-weighted emission outputs use.

F-gases (`FGAS_*`), none of which contribute to EESC:

| Species   | Formula         | $\tau_{tot}$ (yr) | $\tau_{OH}$ (yr) | $\tau_{strat}$ (yr) | $\eta$ (W/m²/ppb) | $C_{pi}$ (ppt) | GWP-100 |
| --------- | --------------- | ----------------- | ---------------- | ------------------- | ----------------- | -------------- | ------- |
| CF4       | CF4             | 50000             | 0                | 0                   | 0.09              | 34.05          | 7390    |
| C2F6      | C2F6            | 10000             | 0                | 0                   | 0.25              | 0              | 12200   |
| C3F8      | C3F8            | 2600              | 0                | 0                   | 0.28              | 0              | 8830    |
| C4F10     | C4F10           | 2600              | 0                | 0                   | 0.36              | 0              | 8860    |
| C5F12     | n-C5F12         | 4100              | 0                | 0                   | 0.41              | 0              | 9160    |
| C6F14     | n-C6F14         | 3100              | 0                | 0                   | 0.44              | 0              | 9300    |
| C7F16     | n-C7F16         | 3000              | 0                | 0                   | 0.5               | 0              | 0       |
| C8F18     | C8F18           | 3000              | 0                | 0                   | 0.55              | 0              | 0       |
| cC4F8     | c-C4F8          | 3200              | 0                | 0                   | 0.32              | 0              | 10300   |
| HFC23     | CHF3            | 228               | 243              | 4420                | 0.18              | 0              | 14800   |
| HFC32     | CH2F2           | 5.4               | 5.5              | 124                 | 0.11              | 0              | 675     |
| HFC4310   | CF3CHFCHFCF2CF3 | 17                | 17.9             | 365                 | 0.359             | 0              | 1640    |
| HFC125    | CHF2CF3         | 31                | 32               | 351                 | 0.23              | 0              | 3500    |
| HFC134a   | CH2FCF3         | 14                | 14.1             | 267                 | 0.16              | 0              | 1430    |
| HFC143a   | CH3CF3          | 51                | 57               | 612                 | 0.16              | 0              | 4470    |
| HFC152a   | CH3CHF2         | 1.6               | 1.55             | 39                  | 0.1               | 0              | 124     |
| HFC227ea  | CF3CHFCF3       | 36                | 37.5             | 673                 | 0.26              | 0              | 3220    |
| HFC236fa  | CF3CH2CF3       | 213               | 253              | 1350                | 0.24              | 0              | 9810    |
| HFC245fa  | CHF2CH2CF3      | 7.9               | 8.2              | 149                 | 0.24              | 0              | 1030    |
| HFC365mfc | CH3CF2CH2CF3    | 8.9               | 9.3              | 190                 | 0.22              | 0              | 794     |
| NF3       | NF3             | 569               | 0                | 740                 | 0.2               | 0              | 17200   |
| SF6       | SF6             | 850               | 0                | 0                   | 0.57              | 0              | 22800   |
| SO2F2     | SO2F2           | 36                | 300              | 630                 | 0.2               | 0              | 0       |

Montreal Protocol gases (`MHALO_*`):

| Species   | Formula    | $\tau_{tot}$ (yr) | $\tau_{OH}$ (yr) | $\tau_{strat}$ (yr) | $\eta$ (W/m²/ppb) | $C_{pi}$ (ppt) | $n_{Cl}$ | $n_{Br}$ | $f$  | GWP-100 |
| --------- | ---------- | ----------------- | ---------------- | ------------------- | ----------------- | -------------- | -------- | -------- | ---- | ------- |
| CFC11     | CCl3F      | 52                | 0                | 55                  | 0.295             | 0              | 3        | 0        | 0.47 | 4750    |
| CFC12     | CCl2F2     | 102               | 0                | 103                 | 0.364             | 0              | 2        | 0        | 0.23 | 10900   |
| CFC113    | CCl2FCClF2 | 93                | 0                | 94.5                | 0.3               | 0              | 3        | 0        | 0.29 | 6130    |
| CFC114    | CClF2CClF2 | 189               | 0                | 191                 | 0.31              | 0              | 2        | 0        | 0.12 | 10000   |
| CFC115    | CClF2CF3   | 540               | 0                | 664                 | 0.2               | 0              | 1        | 0        | 0.04 | 7370    |
| HCFC22    | CHClF2     | 11.9              | 13               | 161                 | 0.21              | 0              | 1        | 0        | 0.13 | 1810    |
| HCFC141b  | CH3CCl2F   | 9.4               | 10.7             | 72.3                | 0.16              | 0              | 2        | 0        | 0.34 | 725     |
| HCFC142b  | CH3CClF2   | 18                | 19.3             | 212                 | 0.19              | 0              | 1        | 0        | 0.17 | 2310    |
| CH3CCl3   | CH3CCl3    | 5                 | 6.1              | 38                  | 0.07              | 0              | 3        | 0        | 0.67 | 146     |
| CCl4      | CCl4       | 32                | 0                | 44                  | 0.174             | 0.0250004      | 4        | 0        | 0.56 | 1400    |
| CH3Cl     | CH3Cl      | 0.9               | 1.57             | 30.4                | 0.004             | 457            | 1        | 0        | 0.44 | 13      |
| CH2Cl2    | CH2Cl2     | 0.5               | 0.5              | 0                   | 0.028             | 6.91276        | 2        | 0        | 0    | 8.7     |
| CHCl3     | CHCl3      | 0.5               | 0.5              | 0                   | 0.07              | 6              | 3        | 0        | 0    | 0       |
| CH3Br     | CH3Br      | 0.8               | 1.8              | 26.3                | 0.004             | 5.3            | 0        | 1        | 0.6  | 5       |
| Halon1211 | CBrClF2    | 16                | 0                | 41                  | 0.29              | 0.00444657     | 1        | 1        | 0.62 | 1890    |
| Halon1301 | CBrF3      | 72                | 0                | 73.5                | 0.3               | 0              | 0        | 1        | 0.28 | 7140    |
| Halon2402 | CBrF2CBrF2 | 28                | 0                | 41                  | 0.31              | 0              | 0        | 2        | 0.65 | 1640    |
| Halon1202 | CBr2F2     | 2.5               | 0                | 36                  | 0.27              | 0              | 0        | 2        | 0.62 | 0       |

### Global parameters

| Parameter                                                         | Default | Unit      | Description                                       |
| ----------------------------------------------------------------- | ------- | --------- | ------------------------------------------------- |
| `GEN_AIR_GRAMMPROMOL`                                             | 28.984  | g/mol     | Molar mass of air                                 |
| `GEN_ATM_TOTMASS_1E21GRAMM`                                       | 5.133   | $10^{21}$ g | Mass of the atmosphere                          |
| `*_EFF_MIXBOXSIZE`                                                | 0.949   | 1         | Mixing box size of each species                   |
| `GEN_EESC_STRATMIXDELAY`                                          | 3       | yr        | Troposphere to stratosphere delay                 |
| `STRATOZ_BR_VS_CL_SCALE`                                          | 62.5    | 1         | $\alpha_{Br}$                                     |
| `MHALO_RELEASEFACTORC11`                                          | 0.75    | 1         | $f_{11}$                                          |
| `GEN_MERIDFLUX_CHNGPERDEG`                                        | 0.15    | 1/K       | Change in meridional flux per degree of warming   |
| `GEN_CHANGE_MERIDIONALFLUX_YR`                                    | 1980    | year      | Reference year of the warming                     |
| `FGAS_TAUSTRAT_SENS2MERIDFLUX`, `MHALO_TAUSTRAT_SENS2MERIDFLUX`   | 0.3     | 1         | Sensitivity of $\tau_{strat}$ to meridional flux  |
| `FGAS_USE_TAUOH_VAR`, `MHALO_USE_TAUOH_VAR`                       | 1       | flag      | Enables $s_{OH}$                                  |
| `FGAS_USE_TAUSTRAT_VAR`, `MHALO_USE_TAUSTRAT_VAR`                 | 1       | flag      | Enables $s_{strat}$                               |
| `FGAS_SWITCHFROMCONC2EMIS_YEAR`, `MHALO_SWITCHFROMCONC2EMIS_YEAR` | 2015    | year      | Last year of prescribed concentrations            |
| `FGAS_RF_REGIONSCALING_LOWERTAU`                                  | 1.0     | yr        | Below this, the F-gas pattern follows emissions   |
| `FGAS_RF_REGIONSCALING_UPPERTAU`                                  | 8.0     | yr        | Above this, the F-gas pattern is global           |
| `RF_FGASSUM_SCALE`, `RF_MHALOSUM_SCALE`                           | 1.0     | 1         | Scale the radiative efficiencies in the first year |

### Reference values

These follow from the defaults above and are quick checks of an implementation:

| Quantity                                      | Value         |
| --------------------------------------------- | ------------- |
| CFC-11 molar mass                             | 137.3584 g/mol |
| EESC from 200 ppt of CFC-11 (no Br)           | 211.5 ppt     |
| EESC from 3 ppt of Halon-1301                 | 39.375 ppt    |
| Forcing of 200 ppt of CFC-11 above $C_{pi}$   | 0.059 W/m²    |

## Algorithm

`MAGICC7.f90` runs the F-gases and then the Montreal Protocol gases through the same steps each year.

1. Before the run, calculates each species' molar mass and $\tau_{other}$.
2. Calculates $s_{OH}$ and $s_{strat}$ for the year, shared by every species.
3. In the first year, multiplies the radiative efficiencies by `RF_FGASSUM_SCALE` or `RF_MHALOSUM_SCALE`.
4. For each species in order:
   1. Calculates the lifetime.
   2. From the switch year, steps the concentration from emissions,
      and before it keeps the prescribed concentration and calculates inverse emissions.
   3. Calculates the forcing and, for F-gases, its box fractions,
      and adds them to the group total.
   4. Adds the delayed concentration's chlorine and bromine to EESC, ESC and ESBr.
5. Calculates the HFC-134a and CFC-12 equivalent concentrations of the group totals.

Stratospheric ozone forcing follows from EESC,
as the [ozone](module_04_ozone.md#stratospheric-ozone) page describes.

## Numerical considerations

- **Scheme switch**: the change from Crank-Nicolson to the exact exponential at $\tau = 5$ years
  can give a small discontinuity for a species whose lifetime crosses it.
- **Short-lived species**: the annual step is coarse for lifetimes under a year, such as CH2Cl2 (0.5 yr).

## Issues in MAGICC7

RSCM reproduces these unless the entry says otherwise.

- **First F-gas year**: MAGICC7 overwrites each F-gas's first-year concentration
  with its second-year value (`C(1) = C(2)`, marked TODO in the source).
  RSCM does not, as below.
- **HFC-134a index**: MAGICC7 hardcodes `hfc134a_idx = 8` (marked as a bug in the source),
  which is C8F18 in the default list.
  Only the HFC-134a equivalent concentration uses it, and RSCM does not calculate that.
- **Box fractions**: a TODO in `calculate_halo_rf_box_fractions` notes that the interpolation
  between the two lifetime regimes is dimensionally inconsistent.

## Differences from MAGICC7

- **First F-gas year**: RSCM starts from $C_{pi}$, and `test_halocarbons` compares from the second year.
- **Montreal Protocol pattern**: RSCM applies the lifetime-dependent pattern to F-gases only.
  The Montreal Protocol gases reach `ForcingAggregation` as global forcing spread by `RF_REGIONS_MHALO`,
  and `test_coupled_run` reaches parity with MAGICC7's temperature that way.
- **Holds**: RSCM has no `RF_FGAS_CONSTANTAFTERYR` or `RF_MHALO_CONSTANTAFTERYR`.
  The reference run sets both to 2105, its last year, so they have no effect there.
- **Not in RSCM**: inverse emissions and the CFC-12 and HFC-134a equivalent concentrations.

## References

MAGICC7 source, relative to the MAGICC repository:

| Function or section                                   | File                                           | Lines     |
| ----------------------------------------------------- | ---------------------------------------------- | --------- |
| `MOD_HALOS` and `CALC_HALOS_MOLMASS`                  | `src/libmagicc/core.f90`                       | 100-131   |
| FGAS and MHALO datastores                             | `src/libmagicc/utils/datastore.f90`            | 130-196   |
| `calculate_halo_tautot`                               | `src/libmagicc/physics/deltaq_calculations.f90` | 430-453  |
| `calculate_halo_kton_per_yr_to_ppt_conv_factor`       | `src/libmagicc/physics/deltaq_calculations.f90` | 455-490  |
| `calculate_halo_conc`                                 | `src/libmagicc/physics/deltaq_calculations.f90` | 492-536  |
| `calculate_halo_inverse_emis`                         | `src/libmagicc/physics/deltaq_calculations.f90` | 538-581  |
| `calculate_halo_rf`                                   | `src/libmagicc/physics/deltaq_calculations.f90` | 583-594  |
| `calculate_halo_eq_conc_*`                            | `src/libmagicc/physics/deltaq_calculations.f90` | 596-646  |
| `calculate_halo_rf_box_fractions`                     | `src/libmagicc/physics/deltaq_calculations.f90` | 648-756  |
| `add_stratospheric_loading_contributions`             | `src/libmagicc/physics/deltaq_calculations.f90` | 902-951  |
| Molar masses and $\tau_{other}$                       | `src/libmagicc/MAGICC7.f90`                    | 2520-2561 |
| EESC set-up                                           | `src/libmagicc/MAGICC7.f90`                    | 4418-4427 |
| F-gas loop and equivalent concentration               | `src/libmagicc/MAGICC7.f90`                    | 4429-4591 |
| Montreal Protocol loop, equivalents and total         | `src/libmagicc/MAGICC7.f90`                    | 4592-4738 |
| FGAS, MHALO and GEN defaults                          | `run/MAGCFG_DEFAULTALL.CFG`                    | 140-167, 258-301 |

- Butchart, N. and Scaife, A. A. (2001). Removal of chlorofluorocarbons by increased mass exchange
  between the stratosphere and troposphere in a changing climate. *Nature*, 410, 799-802.
