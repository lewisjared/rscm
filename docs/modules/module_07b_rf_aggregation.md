# Radiative Forcing Aggregation and Efficacy

## Overview

The other forcing modules each calculate one agent's forcing.
This module describes how MAGICC7's `DELTAQ` subroutine combines them
into the forcing that drives the climate model.
It covers four steps:

- Spreading each agent's global forcing over the four boxes with a regional pattern.
- Scaling it by the agent's efficacy.
- Summing the agents in a fixed order into subtotals and the total.
- Choosing which agents enter the total with the run mode.

Most of the parity detail is in the summation order,
because floating-point addition is not associative.

## RSCM implementation

- **Component**: `ForcingAggregation` (Python builder `rscm.magicc.ForcingAggregationBuilder`).
- **Source**: `crates/rscm-magicc/src/forcing/aggregation.rs`,
  with parameters in `crates/rscm-magicc/src/parameters/forcing_aggregation.rs`.
  `rscm.magicc.full.forcing_agents` and `rscm.magicc.full.SUMMATION` set MAGICC7's agents and order.
- **Status**: Implemented. See the [status table](README.md#rscm-implementation-status).
- **Tests**: `test_coupled_run` in `tests/regression/test_full_run.py` checks total ERF and temperature,
  including the `no_efficacy` (`RF_EFFICACY_APPLY` 0) and `prescribed_efficacy` (1) variants.
  `aggregation.rs` has Rust unit tests for the patterns and efficacy modes.

Volcanic forcing enters `ClimateUdeb` directly, as in MAGICC7.

## Forcing types

MAGICC7 keeps three forcing datastores for every agent:

| Type  | Datastore     | Definition                                                                           |
| ----- | ------------- | ------------------------------------------------------------------------------------ |
| RF    | `DAT_*_RF`    | Forcing the agent's module calculates, before efficacies                             |
| EffRF | `DAT_*_EFFRF` | RF scaled by the efficacy factor $f$ below, which drives the climate model           |
| ERF   | `DAT_*_ERF`   | EffRF multiplied back by the internal efficacy, which MAGICC7 reports as ERF         |

With the default `RF_EFFICACY_APPLY = 2`, ERF is RF times the prescribed efficacy.
Every prescribed efficacy defaults to 1, so ERF and RF then differ only by rounding.

In RSCM each component's `Effective Radiative Forcing|<agent>` output is MAGICC7's RF.
`ForcingAggregation` writes the box EffRF as `Effective Radiative Forcing|Efficacy Adjusted`,
which `ClimateUdeb` reads,
and the area-weighted sum of the agents' RF as `Effective Radiative Forcing`.
The parity tests rebuild MAGICC7's reported ERF from these outputs
with `Reported` in `tests/regression/full_run.py`.

## Equations

### Regional patterns

The four boxes are in the order NH ocean, NH land, SH ocean, SH land,
with area fractions $a_i$ from [initialisation](module_00_initialisation.md#four-box-geography).
A global forcing $Q$ is spread over the boxes with the agent's pattern $r_i$ (`RF_REGIONS_*`),
normalised so the area-weighted mean is $Q$:

$$
Q_i = Q \, q_i, \qquad q_i = \frac{r_i}{\sum_j a_j r_j}
$$

A pattern whose weighted sum is zero gives $q_i = 1$.
Agents whose module already calculates box forcing, such as tropospheric ozone and the aerosols,
enter as four-box values and skip this step.
Their pattern still sets their internal efficacy.

### Efficacy

The efficacy factor $f$ depends on `RF_EFFICACY_APPLY`:

$$
\text{EffRF}_i = f \, Q_i, \qquad
f = \begin{cases}
1 & \text{RF\_EFFICACY\_APPLY} = 0 \\
1 / \varepsilon_{int} & \text{RF\_EFFICACY\_APPLY} = 1 \\
\varepsilon / \varepsilon_{int} & \text{RF\_EFFICACY\_APPLY} = 2
\end{cases}
$$

$\varepsilon$ is the prescribed efficacy `RF_EFFICACY_*`.
Mode 1 cancels each pattern's own efficacy, so every agent has unit efficacy.
RSCM's `efficacy_method` names modes 0, 1 and 2 `none`, `unit` and `ar6`.

### Internal efficacy

The internal efficacy $\varepsilon_{int}$ (`RF_INTEFFICACY_*`) is the warming of the climate model's
four-box equilibrium for the agent's pattern, relative to its climate sensitivity.
`CALC_INTERNAL_EFFICACY` evaluates it with $M^{-1}$, the inverse coupling matrix from LAMCALC
(see [climate](module_08_climate.md)):

$$
T_i = Q_{2\times} \sum_j \left(M^{-1}\right)_{ij} a_j q_j,
\qquad
\varepsilon_{int} = \frac{1}{\text{ECS}} \sum_i a_i T_i
$$

CO2 takes its internal efficacy from LAMCALC's CO2 response instead.
Patterns weighted towards land, which warms more, give $\varepsilon_{int} > 1$.
MAGICC7 recalculates the internal efficacies each year
from the previous year's climate sensitivity and CO2 doubling forcing.
Before the climate model reports them it uses the configured sensitivity and $Q_{2\times} = 3.71$ W/m².

## Parameters

| Parameter                  | Default | Description                                                        |
| -------------------------- | ------- | ------------------------------------------------------------------ |
| `RF_EFFICACY_APPLY`        | 2       | Efficacy mode, see [Efficacy](#efficacy)                           |
| `RF_EFFICACY_*`            | 1.0     | Prescribed efficacy $\varepsilon$ of each agent                    |
| `RF_INTEFFICACY_*`         | table   | Internal efficacy, overwritten each year by LAMCALC                |
| `RF_REGIONS_*`             | table   | Regional pattern of each agent, in box order                       |
| `RF_REGIONS_NORMYEAR`      | 2000    | Year whose box forcing sets the land use and BC on snow patterns   |
| `RF_TOTAL_RUNMODUS`        | `ALL`   | Agents in the total, see [Run modes](#run-modes)                   |
| `RF_PREIND_REFERENCEYR`    | 1750    | Pre-industrial reference year                                      |
| `RF_INITIALIZATION_METHOD` | `JUMPSTART` | First-year offset, see [Pre-industrial reference](#pre-industrial-reference) |

### Agent inventory

The agents of MAGICC7's default total, with their RSCM input in `run_magicc7`.
"Global" agents are spread with their pattern, and "box" agents enter as four-box values.
Internal efficacies are the defaults in `python/rscm/magicc/data/magicc7/defaults.json`, rounded.
`SRF_FACTOR_*` sets each agent's surface forcing, which RSCM does not calculate.

| Agent (`Effective Radiative Forcing|…`) | Key            | Input  | $\varepsilon_{int}$ | `SRF_FACTOR` |
| ---------------------------------------- | -------------- | ------ | ------------------- | ------------ |
| `CO2`                                    | `CO2`          | global | 1.000               | 0.2          |
| `CH4`                                    | `CH4`          | global | 0.983               | 0.5          |
| `N2O`                                    | `N2O`          | global | 0.993               | 0.2          |
| `F-Gases`                                | `FGAS`         | box    | 1.006               | 0.2          |
| `Montreal Protocol Halogen Gases`        | `MHALO`        | global | 1.006               | 1.0          |
| `Stratospheric Ozone`                    | `STRATOZ`      | global | 0.933               | -0.5         |
| `Tropospheric Ozone`                     | `TROPOZ`       | box    | 1.075               | 0.6          |
| `Ozone due to N2O`                       | `OZDUETON2O`   | global | 1.002               |              |
| `Ozone due to Temperature`               | `OZDUETOTEMPERATURE` | global | 1.002         |              |
| `CH4 Oxidation Stratospheric H2O`        | `CH4OXSTRATH2O`| global | 0.958               | 0.0          |
| `SOx|Fossil`                            | `SOX`          | box    | 1.167               | 1.0          |
| `BC|Fossil`                             | `BC`           | box    | 1.160               | -1.9         |
| `OC|Fossil`                             | `OC`           | box    | 1.104               | 1.0          |
| `NO3|Fossil`                            | `NO3`          | box    | 1.000               | 1.0          |
| `Biomass Burning Aerosols`               | `AER_DIR`      | box    | 1.135               |              |
| `Mineral Dust`                           | `AER_DIR`      | box    | 1.135               | 2.0          |
| `Cloud Albedo`                           | `CLOUD_ALBEDO` | box    | 1.097               | 0.8          |
| `BC on Snow`                             | `BCSNOW`       | box    | 1.263               | 1.0          |
| `Land Use`                               | `LANDUSE`      | box    | 1.204               | 1.5          |
| `Solar`                                  | `SOLAR`        | box    | 1.002               | 1.0          |

Notes on the table:

- **Halocarbons**: the F-gas and Montreal Protocol agents apply the efficacy to each species before summing,
  and each F-gas species has its own box pattern ([halocarbon chemistry](module_03_halocarbon_chemistry.md)).
- **Biomass burning**: `Biomass Burning Aerosols` sums the SOx, NO3, OC and BC `|AFOLU` forcing
  before its efficacy applies.
  Like mineral dust, it takes the direct aerosol efficacy and pattern (`AER_DIR`).
- **Patterns from forcing**: land use and BC on snow take their pattern from their box forcing in
  `RF_REGIONS_NORMYEAR`, and nitrate from its total forcing in that year.
- **Volcanic**: monthly volcanic forcing (`RF_EFFICACY_VOLC`, `SRF_FACTOR_VOLC` 0.7) goes straight to the
  climate model, through `ClimateUdeb`'s `subannual_forcing` in RSCM.
- **Not in the default total**: cloud cover is zero by default ([indirect aerosol forcing](module_06_aerosol_indirect.md)),
  external forcing (`QXTRA`) needs `RF_EXTRA_READ`, and the aviation agents (H2O, contrails, cirrus)
  have no term in the summation order.
  RSCM has none of them.

## Algorithm

### Summation order

MAGICC7 builds every total (RF, EffRF and ERF, globally and per box) from left to right,
adding each nested subtotal first:

```text
TOTAL      = CO2 + CH4 + N2O + FGASSUM + MHALOSUM + TOTAER_DIR + CLOUD_TOT
             + OZTOTAL + CH4OXSTRATH2O + LANDUSE + BCSNOW + SOLAR
TOTAER_DIR = BCI + OCI + SOXI + NO3I + BIOMASSAER + MINERALDUST
BIOMASSAER = SOXB + NO3B + OCB + BCB
CLOUD_TOT  = CLOUD_ALBEDO + CLOUD_COVER
OZTOTAL    = STRATOZ + TROPOZ + OZDUETON2O + OZDUETOTEMPERATURE
TOTAL_INCLVOLCANIC = TOTAL + VOLCANIC_ANNUAL
```

`ForcingAggregation` takes this order as its `summation` parameter, a nested list of agent variables,
and checks that it names every agent once.
`rscm.magicc.full.SUMMATION` holds MAGICC7's order.
`CLOUD_TOT` is the single `Cloud Albedo` agent there, because cloud cover is not modelled.

### Run modes

`RF_TOTAL_RUNMODUS` selects which subtotal becomes the total, for idealised experiments.
The subtotals are built in the same order as above:

- `KYOTOGHG` = CO2 + CH4 + N2O + FGASSUM, and `GHG` = KYOTOGHG + MHALOSUM.
- `CO2CH4N2O` = CO2 + CH4 + N2O.
- `AEROSOL` = TOTAER_DIR + CLOUD_TOT.

| Mode              | Total                                                        |
| ----------------- | ------------------------------------------------------------ |
| `ALL`             | All agents, plus monthly volcanic forcing (the default)      |
| `ANTHROPOGENIC`   | All agents except solar and volcanic                         |
| `NONCO2EMISSIONS` | As `ANTHROPOGENIC`, with CO2 emissions set to zero upstream  |
| `NATURAL`         | Solar, plus monthly volcanic forcing                         |
| `CO2`             | CO2                                                          |
| `CO2CH4N2O`       | CO2, CH4 and N2O                                             |
| `GHG`             | CO2, CH4, N2O and all halocarbons                            |
| `AEROSOL`         | Direct aerosols and cloud albedo                             |
| `STRATO3`         | Stratospheric ozone                                          |
| `TROPO3`          | Tropospheric ozone                                           |
| `QEXTRA`          | External forcing                                             |

Modes without natural forcing also leave out volcanic forcing.
RSCM implements only `ALL`.

### Pre-industrial reference

`RF_PREIND_REFERENCEYR` (1750) is the reference year of the forcing calculations,
and the first year of MAGICC7's default run.
`set_rf_first_year_forcing_and_offset` then sets each agent's first-year offset
from `RF_INITIALIZATION_METHOD`:

- **`ZEROSTARTSHIFT`** stores the first year's forcing as the offset and subtracts it from every year,
  so forcing starts at zero.
- **`JUMPSTART`** (the default) sets the offset to zero, so forcing keeps its first-year value.

RSCM reproduces `JUMPSTART` and has no `ZEROSTARTSHIFT` option.

### Freeze controls

`ensure_constant_after_year` copies the previous year's forcing, globally and per box,
from a configured year onwards.
Several controls can apply to one agent, and the earliest year wins.

| Parameter                        | Default | Holds                    |
| -------------------------------- | ------- | ------------------------ |
| `RF_TOTAL_CONSTANTAFTERYR`       | 2105    | Total forcing            |
| `RF_CO2_CONSTANTAFTERYR`         | 2105    | CO2                      |
| `RF_CH4_CONSTANTAFTERYR`         | 2105    | CH4                      |
| `RF_N2O_CONSTANTAFTERYR`         | 2105    | N2O                      |
| `RF_FGAS_CONSTANTAFTERYR`        | 2105    | F-gases                  |
| `RF_MHALO_CONSTANTAFTERYR`       | 2105    | Montreal Protocol gases  |
| `RF_TROPOZ_CONSTANTAFTERYR`      | 2105    | Tropospheric ozone       |
| `RF_STRATOZ_CONSTANTAFTERYR`     | 2105    | Stratospheric ozone      |
| `RF_AER_CONSTANTAFTERYR`         | 2105    | Aerosols                 |
| `RF_LANDUSE_CONSTANTAFTERYR`     | 2018    | Land use                 |
| `RF_MINERALDUST_CONSTANTAFTERYR` | 2018    | Mineral dust             |

RSCM applies the land use and mineral dust holds in `rscm.magicc.prescribed`, and none of the others.

### Building a total with VariableSchema

The simpler runners do not need MAGICC7's efficacies or summation order.
They declare the total as an aggregate in a [VariableSchema](../key_concepts.md#variable-schema):

```python
from rscm.core import VariableSchema

ERF = "Effective Radiative Forcing"

schema = VariableSchema()
# Individual agents, named as GhgForcing outputs them, plus a prescribed residual
for agent in ("CO2", "CH4", "N2O", "Other"):
    schema.add_variable(f"{ERF}|{agent}", "W/m^2")
schema.add_aggregate(
    f"{ERF}|GHG", "W/m^2", "sum", [f"{ERF}|CO2", f"{ERF}|CH4", f"{ERF}|N2O"]
)
# Top-level aggregate, read by ClimateUdeb
schema.add_aggregate(ERF, "W/m^2", "sum", [f"{ERF}|GHG", f"{ERF}|Other"])
```

`run_magicc` and `run_magicc_emissions` build total ERF this way, as a plain sum.
`ClimateUdeb` then applies the CO2 efficacy and regional pattern to that total
(see [climate](module_08_climate.md)).
`rscm.magicc.full.run_magicc7` uses `ForcingAggregation` instead.
See the [tutorials](../tutorials.md) for a walkthrough.

## Numerical considerations

`CALCULATE_FOUR_BY_FOUR_INVERSE` divides the adjugate by the determinant,
with every cofactor and the determinant expanded along the first row.
RSCM's `invert_by_cofactors` follows the same order,
because Gauss-Jordan elimination differs in the last bit and shifts some internal efficacies by one ulp.

## Issues in MAGICC7

- **Lagged internal efficacies**: each year uses the internal efficacies of the previous year's climate.
  RSCM reproduces this.

## Differences from MAGICC7

- **Global total**: `Effective Radiative Forcing` sums the agents' RF, not MAGICC7's reported ERF.
  The two differ only by rounding while every prescribed efficacy is 1.
- **Run modes, surface forcing and freeze controls**: only the `ALL` run mode is implemented,
  there is no surface forcing and most freeze controls are missing (see [Algorithm](#algorithm)).
- **NaN checks**: MAGICC7 stops with a fatal error that names each agent whose EffRF is NaN.
  `ForcingAggregation` does not check.

## References

MAGICC7 source files, relative to `src/libmagicc/`:

| Function or section                                          | File                              |
| ------------------------------------------------------------ | --------------------------------- |
| `DELTAQ` (lines 3735 to 7042), LAMCALC, `CALC_INTERNAL_EFFICACY` | `MAGICC7.f90`                 |
| Helper functions                                             | `physics/deltaq_calculations.f90` |
| Forcing datastore declarations                               | `physics/radiative_forcing.f90`   |
| `RF_*` configuration parameters                              | `allcfgs.f90`                     |

- Meinshausen, M. et al. (2011). Emulating coupled atmosphere-ocean and carbon cycle models
  with a simpler model, MAGICC6. Part 1: Model description and calibration.
- Hansen, J. et al. (2005). Efficacy of climate forcings.
