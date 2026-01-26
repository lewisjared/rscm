# MAGICC Module Documentation

This directory contains detailed documentation for each scientific module in MAGICC (Model for the Assessment of Greenhouse Gas Induced Climate Change). These documents are intended to guide the Rust reimplementation while maintaining exact numerical fidelity with the Fortran original.

## Architecture Overview

MAGICC is a reduced-complexity climate model that chains several Initial Value Problems (IVPs):

```
EMISSIONS -> CONCENTRATIONS -> RADIATIVE FORCING -> TEMPERATURE -> CARBON CYCLE FEEDBACKS
                                                       ↑                    |
                                                       └────────────────────┘
```

**Key Architectural Finding:** The "feedback loop" is NOT iterative within a timestep. MAGICC uses a **staggered explicit scheme** where carbon cycle uses T(t-1) to compute fluxes for year t. This means modules CAN be standalone steps with clear input/output contracts.

## Time Index Convention

| Variable Type  | Index Convention | Example                                  |
| -------------- | ---------------- | ---------------------------------------- |
| Concentrations | Start-of-year    | CO2(1750) = concentration at Jan 1, 1750 |
| Emissions      | Mid-year         | E(1750) = emissions during year 1750     |
| Temperature    | Start-of-year    | T(1750) = temp at Jan 1, 1750            |
| Forcing        | Start-of-year    | RF(1750) = forcing at Jan 1, 1750        |

## Spatial Structure

MAGICC uses a 4-box spatial discretization (hardcoded):

- **NH-Land** (Northern Hemisphere Land)
- **NH-Ocean** (Northern Hemisphere Ocean)
- **SH-Land** (Southern Hemisphere Land)
- **SH-Ocean** (Southern Hemisphere Ocean)

## Module Summary

### Core Pathway (Critical for any run)

| Module                                               | File                      | Purpose                                     | Iteration Type      |
| ---------------------------------------------------- | ------------------------- | ------------------------------------------- | ------------------- |
| [00 - Initialization](module_00_initialization.md)   | `MAGICC7.f90`             | Initialize DATASTORE, geography, parameters | One-time            |
| [15 - Main Timestep](module_15_main_timestep.md)     | `MAGICC7.f90`             | Orchestrate annual stepping                 | Sequential          |
| [07b - RF Aggregation](module_07b_rf_aggregation.md) | `deltaq_calculations.f90` | Sum forcing agents, apply efficacies        | None                |
| [08 - Climate](module_08_climate.md)                 | `climate_and_ocean.f90`   | Energy balance, ocean heat uptake           | 12 monthly substeps |

### Chemistry Modules (Emissions -> Concentrations)

| Module                                                         | File                      | Purpose                                    | Iteration Type       |
| -------------------------------------------------------------- | ------------------------- | ------------------------------------------ | -------------------- |
| [01 - CH4 Chemistry](module_01_ch4_chemistry.md)               | `methane.f90`             | CH4 concentration with OH feedback         | 4 Prather iterations |
| [02 - N2O Chemistry](module_02_n2o_chemistry.md)               | `n2o.f90`                 | N2O concentration with stratospheric delay | 4 iterations         |
| [03 - Halocarbon Chemistry](module_03_halocarbon_chemistry.md) | `deltaq_calculations.f90` | ~41 species exponential decay              | None                 |
| [04 - Ozone](module_04_ozone.md)                               | `deltaq_calculations.f90` | Tropospheric + stratospheric O3            | None                 |

### Forcing Modules (Concentrations -> W/m^2)

| Module                                                 | File                      | Purpose                       | Swappable Methods       |
| ------------------------------------------------------ | ------------------------- | ----------------------------- | ----------------------- |
| [07a - GHG Forcing](module_07a_ghg_forcing.md)         | `deltaq_calculations.f90` | CO2, CH4, N2O forcing         | IPCCTAR vs OLBL         |
| [05 - Aerosol Direct](module_05_aerosol_direct.md)     | `radiative_forcing.f90`   | BC, OC, SOx, dust, nitrate    | File-based + calculated |
| [06 - Aerosol Indirect](module_06_aerosol_indirect.md) | `cloudstore.f90`          | Cloud albedo/lifetime effects | File-based              |

### Carbon Cycle Modules (Feedbacks)

| Module                                                     | File                       | Purpose                  | Key Feature                        |
| ---------------------------------------------------------- | -------------------------- | ------------------------ | ---------------------------------- |
| [09 - Terrestrial Carbon](module_09_terrestrial_carbon.md) | `carbon_cycle.f90`         | 4-pool land carbon model | CO2 fertilization, Q10 respiration |
| [10 - Ocean Carbon](module_10_ocean_carbon.md)             | `carbon_cycle_ocean/*.f90` | Ocean CO2 uptake         | IRF models (BERN, HILDA, etc.)     |
| [11 - CO2 Budget](module_11_co2_budget.md)                 | `MAGICC7.f90`              | Mass balance integrator  | Closes carbon loop                 |

### Experimental Modules

| Module                                                       | File                      | Purpose                     | Status       |
| ------------------------------------------------------------ | ------------------------- | --------------------------- | ------------ |
| [12 - Permafrost](module_12_permafrost.md)                   | `permafrost.f90`          | Thaw-driven CH4/CO2 release | Experimental |
| [13 - Nitrogen Limitation](module_13_nitrogen_limitation.md) | `nitrogen_limitation.f90` | N constraint on NPP         | Experimental |
| [14 - Sea Level Rise](module_14_sea_level_rise.md)           | `sealevel.f90`            | Thermal + ice sheet SLR     | Experimental |

## Dependency Graph

```
EMISSIONS INPUT
       ↓
   ┌───┴───────────────────────────────────┐
   ↓                                       ↓
CH4 CHEMISTRY ←──────────┐        HALO CHEMISTRY
   ↓                     │              ↓
N2O CHEMISTRY            │         OZONE MODULE
   ↓                     │              ↓
   └──────────-> RADIATIVE FORCING ←─────┘
                    AGGREGATOR
                         ↓
                ┌────────┴────────┐
                ↓                 ↓
        AEROSOL DIRECT    AEROSOL INDIRECT
                ↓                 ↓
                └────────┬────────┘
                         ↓
                  CLIMATE MODEL
                         ↓
            ┌────────────┼────────────┐
            ↓            ↓            ↓
     TERRESTRIAL    OCEAN C      PERMAFROST
       C CYCLE       CYCLE       (optional)
            ↓            ↓            ↓
            └────────────┼────────────┘
                         ↓
                   CO2 BUDGET
                         ↓
                 (loop back to RF)
```

## Iteration Analysis

### Within-Module Iterations (Must Preserve for Exact Match)

| Module        | Iteration Type         | Count  | Why Required                          |
| ------------- | ---------------------- | ------ | ------------------------------------- |
| CH4 Chemistry | Prather fixed-point    | 4      | τ depends on B, implicit equation     |
| N2O Chemistry | Fixed-point            | 4      | Same pattern as CH4                   |
| N-Limitation  | Bisection root-finding | ~20    | Steady-state N pool                   |
| LAMCALC       | Secant/bisection       | ~10-20 | Find λ satisfying ECS+RLO (init only) |

### Sub-Annual Stepping (Hidden from Interface)

| Module         | Substeps          | Why                                    |
| -------------- | ----------------- | -------------------------------------- |
| Climate (UDEB) | 12/year (monthly) | Numerical stability of ocean diffusion |
| Ocean Carbon   | 12/year (monthly) | Stability of air-sea flux              |

## Key Fortran Files

| File                       | Lines   | Contents                          |
| -------------------------- | ------- | --------------------------------- |
| `MAGICC7.f90`              | ~12,599 | Main loop, DELTAQ, initialization |
| `deltaq_calculations.f90`  | ~1,605  | All forcing formulas              |
| `climate_and_ocean.f90`    | ~294    | Ocean diffusion, 4-box temps      |
| `methane.f90`              | ~234    | CH4 chemistry                     |
| `n2o.f90`                  | ~200    | N2O chemistry                     |
| `carbon_cycle.f90`         | ~197    | 4-pool terrestrial model          |
| `carbon_cycle_ocean/*.f90` | ~599    | IRF ocean models                  |
| `permafrost.f90`           | ~931    | Permafrost thaw/decomposition     |
| `sealevel.f90`             | ~831    | Ice sheet dynamics                |
| `cloudstore.f90`           | ~916    | Indirect aerosol effects          |
| `nitrogen_limitation.f90`  | ~396    | NPP constraint                    |
| `allcfgs.f90`              | ~2000   | All 500+ parameters               |

## Document Structure

Each module document follows this structure:

1. **Scientific Purpose** - What physical process is modeled
2. **Mathematical Formulation** - Governing equations in LaTeX
3. **State Variables** - Variables that persist between timesteps
4. **Parameters** - Configurable constants from CFG files
5. **Inputs/Outputs** - Interface contract with other modules
6. **Algorithm** - Step-by-step pseudocode matching Fortran
7. **Fortran Code References** - Specific file:line references
8. **Numerical Considerations** - Stability, edge cases
9. **Test Cases** - Validation approach

## RSCM Framework Compatibility

The existing [RSCM framework](https://github.com/climate-resource/rscm) provides most infrastructure needed:

| Feature            | RSCM Status | Notes                                                       |
| ------------------ | ----------- | ----------------------------------------------------------- |
| Component trait    | ✅          | `solve(t, t_next, input_state) -> RSCMResult<OutputState>`  |
| Timeseries         | ✅          | With interpolation (Linear, Previous, Next)                 |
| State management   | ✅          | TimeseriesCollection                                        |
| PyO3 bindings      | ✅          | Builder pattern                                             |
| Dependency graph   | ✅          | Automatic from requirements                                 |
| Historical access  | ✅          | `previous()` for t-1, `at_offset(-n)` for arbitrary history |
| Sub-annual stepping| ✅          | Component responsibility (e.g., TwoLayer uses RK4 internally) |
| Spatial grids      | ✅          | FourBox and Hemispheric grids with aggregation              |

**Required Additions:**

1. Tridiagonal solver utility (Thomas algorithm for UDEB ocean diffusion)
