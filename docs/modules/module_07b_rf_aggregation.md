# Module 7b: Radiative Forcing Aggregation and Efficacy Framework

## Overview

This module documents the DELTAQ subroutine's radiative forcing aggregation logic in MAGICC7. While individual forcing calculations are handled by other modules (GHG forcing in Module 7a, aerosols in Modules 5-6, etc.), this module focuses on how all forcing agents are combined into total effective radiative forcing that drives the climate system.

**Source Files:**

- `/src/libmagicc/MAGICC7.f90` - DELTAQ subroutine (lines 3735-7042), LAMCALC, CALC_INTERNAL_EFFICACY
- `/src/libmagicc/physics/deltaq_calculations.f90` - Helper functions
- `/src/libmagicc/physics/radiative_forcing.f90` - RF module declarations
- `/src/libmagicc/allcfgs.f90` - RF configuration parameters

---

## 1. Purpose

The DELTAQ subroutine serves as the central orchestrator for radiative forcing calculations in MAGICC. Its primary responsibilities are:

1. **Agent Calculation Coordination**: Calls individual forcing calculation routines for each forcing agent
2. **Aggregation**: Combines individual agent forcings into intermediate totals (GHG, aerosol, anthropogenic) and final total forcing
3. **Efficacy Application**: Converts raw radiative forcing (RF) to effective radiative forcing (ERF) via efficacy factors
4. **Regional Distribution**: Applies 4-box spatial patterns to distribute global forcing across hemispheric land/ocean boxes
5. **Run Mode Selection**: Filters which agents contribute to total forcing based on `RF_TOTAL_RUNMODUS`
6. **Baseline Correction**: Subtracts pre-industrial reference forcing values
7. **Quality Assurance**: Performs NaN checking and error handling

### Forcing Types

MAGICC maintains three parallel forcing streams:

| Type | Description | Usage |
|------|-------------|-------|
| `RF` | Raw radiative forcing (instantaneous) | Diagnostics, intermediate calculations |
| `EFFRF` | Effective radiative forcing (with prescribed efficacy) | Climate response driver |
| `ERF` | Effective radiative forcing (alternative formulation) | Output diagnostics |

The climate model uses `EFFRF` (effective radiative forcing with efficacies applied) to drive temperature response.

---

## 2. Run Modes (RF_TOTAL_RUNMODUS)

The `RF_TOTAL_RUNMODUS` parameter controls which forcing agents are included in the total forcing calculation. This enables idealized experiments isolating specific forcing agents.

### Available Run Modes

| Mode | Included Forcings | Use Case |
|------|-------------------|----------|
| `CO2` | CO2 only | CO2-only sensitivity experiments |
| `GHG` | Kyoto GHGs + Montreal halocarbons (CO2, CH4, N2O, F-gases, CFCs/HFCs) | Long-lived GHG experiments |
| `CO2CH4N2O` | CO2, CH4, N2O only | Major GHG experiments without halogenated species |
| `AEROSOL` | Direct aerosol + cloud indirect effects | Aerosol-only experiments |
| `STRATO3` | Stratospheric ozone only | Ozone hole experiments |
| `TROPO3` | Tropospheric ozone only | Air quality/ozone forcing studies |
| `QEXTRA` | External forcing (user-prescribed) only | Custom forcing experiments |
| `ANTHROPOGENIC` | All anthropogenic forcings (excludes solar, volcanic) | Attribution studies |
| `NONCO2EMISSIONS` | Anthropogenic forcing with CO2 emissions zeroed | Non-CO2 contribution analysis |
| `NATURAL` | Solar only (volcanic added monthly) | Natural forcing experiments |
| `ALL` | All forcings (anthropogenic + solar + volcanic + extra) | Full forcing runs (default) |

### Implementation Logic

```fortran
IF (RF_TOTAL_RUNMODUS == 'CO2') THEN
    ! Only CO2 forcing contributes to total
    dat_total_effrf = dat_co2_effrf

ELSEIF (RF_TOTAL_RUNMODUS == 'GHG') THEN
    ! GHG = Kyoto (CO2+CH4+N2O+Fgas) + Montreal (MHALO)
    dat_total_effrf = dat_ghg_effrf

ELSEIF (RF_TOTAL_RUNMODUS == 'ALL') THEN
    ! Full aggregation
    dat_total_effrf = dat_total_anthro_effrf + dat_solar_effrf + dat_extra_effrf
    ! Note: Volcanic forcing added on monthly basis in main routine
END IF
```

### Special Handling

- **NONCO2EMISSIONS**: Sets CO2 emissions to zero upstream but still includes any CO2 forcing that arises from non-CO2 feedbacks
- **NATURAL**: Volcanic forcing is NOT added here; it is applied on a monthly basis in the main model loop for proper temporal resolution
- Modes that exclude natural forcings (CO2, GHG, AEROSOL, etc.) also exclude volcanic forcing from the output total

---

## 3. Efficacy Framework

Efficacy accounts for the fact that different forcing agents produce different temperature responses per unit forcing due to their spatial distribution and physical properties.

### Definition

```
Effective Radiative Forcing = Raw Radiative Forcing * Efficacy
```

For an agent X:

```
EFFRF_X = RF_X * EFFICACY_X
```

### Efficacy Types

MAGICC supports two efficacy frameworks:

#### 3.1 Prescribed Efficacies (RF_EFFICACY_*)

User-specified efficacy factors for each agent. Applied when `RF_EFFICACY_APPLY = 1`.

**Configuration Parameters:**

| Parameter | Agent | Typical Value |
|-----------|-------|---------------|
| `RF_EFFICACY_CH4` | Methane | 1.0 |
| `RF_EFFICACY_CH4OXSTRATH2O` | CH4 oxidation stratospheric H2O | 1.0 |
| `RF_EFFICACY_N2O` | Nitrous oxide | 1.0 |
| `RF_EFFICACY_STRATOZ` | Stratospheric ozone | 1.0 |
| `RF_EFFICACY_TROPOZ` | Tropospheric ozone | 1.0 |
| `RF_EFFICACY_OZDUETON2O` | Ozone due to N2O | 1.0 |
| `RF_EFFICACY_OZDUETOTEMPERATURE` | Ozone due to temperature | 1.0 |
| `RF_EFFICACY_AER_DIR` | Direct aerosol (aggregate) | 1.0 |
| `RF_EFFICACY_BC` | Black carbon | 1.0 |
| `RF_EFFICACY_OC` | Organic carbon | 1.0 |
| `RF_EFFICACY_SOX` | Sulfate aerosol | 1.0 |
| `RF_EFFICACY_NO3` | Nitrate aerosol | 1.0 |
| `RF_EFFICACY_CLOUD_COVER` | Cloud cover (indirect) | 1.0 |
| `RF_EFFICACY_CLOUD_ALBEDO` | Cloud albedo (indirect) | 1.0 |
| `RF_EFFICACY_SOLAR` | Solar | ~0.8-1.0 |
| `RF_EFFICACY_VOLC` | Volcanic | ~1.0 |
| `RF_EFFICACY_LANDUSE` | Land use albedo | 1.0 |
| `RF_EFFICACY_BCSNOW` | BC on snow | 2.0-3.0 |
| `RF_EFFICACY_FGAS` | F-gases | 1.0 |
| `RF_EFFICACY_MHALO` | Montreal halocarbons | 1.0 |
| `RF_EFFICACY_QXTRA` | External forcing | 1.0 |
| `RF_EFFICACY_AIRH2O` | Aviation H2O | 1.0 |
| `RF_EFFICACY_CONTRAIL` | Contrails | 1.0 |
| `RF_EFFICACY_CIRRUS` | Aviation cirrus | 1.0 |

#### 3.2 Internal (Calculated) Efficacies (RF_INTEFFICACY_*)

Internally calculated efficacies based on the regional forcing pattern and the climate model's land/ocean response. Calculated in LAMCALC subroutine using CALC_INTERNAL_EFFICACY.

**Calculation Algorithm:**

```fortran
SUBROUTINE CALC_INTERNAL_EFFICACY(Q, LAMCORE_CLIMATESENSITIVITY, BMAT, &
                                  RF_REGIONS_X, RF_INTEFFICACY_X)
    ! Q = global mean forcing for 2xCO2
    ! LAMCORE_CLIMATESENSITIVITY = climate sensitivity (K per W/m2)
    ! BMAT = 4x4 inverse feedback matrix from LAMCALC iteration
    ! RF_REGIONS_X = 4-box regional forcing pattern for agent X

    IF (SUM(RF_REGIONS_X * GLOBALAREAFRACTIONS) == 0.0D0) THEN
        RF_INTEFFICACY_X = 1.0D0  ! Default if no pattern
    ELSE
        ! Apply forcing pattern to boxes
        Q_BOX = Q * RF_REGIONS_X / SUM(RF_REGIONS_X * GLOBALAREAFRACTIONS)

        ! Calculate temperature response via inverse matrix
        T_BOX = MATMUL(BMAT, Q_BOX)

        ! Internal efficacy = actual temp response / expected response
        RF_INTEFFICACY_X = SUM(T_BOX * GLOBALAREAFRACTIONS) / LAMCORE_CLIMATESENSITIVITY
    END IF
END SUBROUTINE
```

**Physical Interpretation:**

- If forcing is concentrated in high-latitude land areas (high feedback), efficacy > 1
- If forcing is concentrated in tropical oceans (low feedback), efficacy < 1
- CO2 forcing with uniform pattern serves as reference (efficacy ~1)

**Internal Efficacy Outputs:**

| Parameter | Agent |
|-----------|-------|
| `RF_INTEFFICACY_CO2` | CO2 (calculated directly in LAMCALC) |
| `RF_INTEFFICACY_CH4` | Methane |
| `RF_INTEFFICACY_N2O` | Nitrous oxide |
| `RF_INTEFFICACY_STRATOZ` | Stratospheric ozone |
| `RF_INTEFFICACY_TROPOZ` | Tropospheric ozone |
| `RF_INTEFFICACY_AER_DIR` | Direct aerosol |
| `RF_INTEFFICACY_BC` | Black carbon |
| `RF_INTEFFICACY_OC` | Organic carbon |
| `RF_INTEFFICACY_SOX` | Sulfate |
| `RF_INTEFFICACY_NO3` | Nitrate |
| `RF_INTEFFICACY_CLOUD_COVER` | Cloud cover |
| `RF_INTEFFICACY_CLOUD_ALBEDO` | Cloud albedo |
| `RF_INTEFFICACY_SOLAR` | Solar |
| `RF_INTEFFICACY_VOLC` | Volcanic |
| `RF_INTEFFICACY_LANDUSE` | Land use |
| `RF_INTEFFICACY_BCSNOW` | BC on snow |
| `RF_INTEFFICACY_FGAS` | F-gases |
| `RF_INTEFFICACY_MHALO` | Montreal halos |
| `RF_INTEFFICACY_QXTRA` | Extra forcing |
| `RF_INTEFFICACY_DUST` | Mineral dust |
| `RF_INTEFFICACY_CONTRAIL` | Contrails |
| `RF_INTEFFICACY_CIRRUS` | Aviation cirrus |
| `RF_INTEFFICACY_AIRH2O` | Aviation H2O |

### Efficacy Application Control

```fortran
RF_EFFICACY_APPLY  ! Integer flag
! 0 = No efficacies applied (EFFRF = RF)
! 1 = Apply prescribed efficacies
```

---

## 4. Regional Patterns (RF_REGIONS_*)

MAGICC uses a 4-box representation for regional forcing distribution:

| Box | Index | Description |
|-----|-------|-------------|
| Northern Hemisphere Land | 1 | NH land fraction |
| Northern Hemisphere Ocean | 2 | NH ocean fraction |
| Southern Hemisphere Land | 3 | SH land fraction |
| Southern Hemisphere Ocean | 4 | SH ocean fraction |

### Pattern Normalization

Regional patterns are normalized to sum to 1.0 when weighted by area fractions:

```fortran
! Normalization requirement
SUM(RF_REGIONS_X * GLOBALAREAFRACTIONS) == 1.0

! Where GLOBALAREAFRACTIONS ~ [0.206, 0.294, 0.045, 0.455]
! (NH land, NH ocean, SH land, SH ocean)
```

### Configuration Parameters

| Parameter | Agent | Notes |
|-----------|-------|-------|
| `RF_REGIONS_CO2` | CO2 | Typically [1,1,1,1] (uniform) |
| `RF_REGIONS_CH4` | Methane | Near-uniform |
| `RF_REGIONS_CH4OXSTRATH2O` | CH4 stratospheric H2O | |
| `RF_REGIONS_N2O` | N2O | Near-uniform |
| `RF_REGIONS_MHALO` | Montreal halocarbons | |
| `RF_REGIONS_FGAS` | F-gases | Near-uniform |
| `RF_REGIONS_TROPOZ` | Tropospheric ozone | NH-weighted |
| `RF_REGIONS_STRATOZ` | Stratospheric ozone | Polar-weighted |
| `RF_REGIONS_OZDUETON2O` | Ozone due to N2O | |
| `RF_REGIONS_OZDUETOTEMPERATURE` | Ozone due to temperature | |
| `RF_REGIONS_DUST` | Mineral dust | NH-weighted |
| `RF_REGIONS_SEASALT` | Sea salt | Ocean-weighted |
| `RF_REGIONS_NO3` | Nitrate | NH-weighted |
| `RF_REGIONS_CLOUD_COVER` | Cloud cover indirect | |
| `RF_REGIONS_CLOUD_ALBEDO` | Cloud albedo indirect | |
| `RF_REGIONS_AIRH2O` | Aviation H2O | NH-weighted |
| `RF_REGIONS_CONTRAIL` | Contrails | NH land heavy |
| `RF_REGIONS_CIRRUS` | Aviation cirrus | NH weighted |
| `RF_REGIONS_NORMYEAR` | Reference year for patterns | Integer year |

### Internally Calculated Patterns (Output)

Some regional patterns are calculated internally based on emissions:

| Parameter | Description |
|-----------|-------------|
| `RF_REGIONS_QXTRA` | Extra forcing pattern |
| `RF_REGIONS_AER_DIR` | Aggregate direct aerosol |
| `RF_REGIONS_BC` | Black carbon |
| `RF_REGIONS_OC` | Organic carbon |
| `RF_REGIONS_SOX` | Sulfate |

---

## 5. Forcing Agent Inventory

### Complete Agent List with Parameters

| Agent | On/Off | Scale | Efficacy | Regional |
|-------|--------|-------|----------|----------|
| CO2 | (always on) | RF_CO2_SCALE | (1.0 reference) | RF_REGIONS_CO2 |
| CH4 | (always on) | RF_CH4_SCALE | RF_EFFICACY_CH4 | RF_REGIONS_CH4 |
| N2O | (always on) | RF_N2O_SCALE | RF_EFFICACY_N2O | RF_REGIONS_N2O |
| CH4 strat H2O | (coupled) | - | RF_EFFICACY_CH4OXSTRATH2O | RF_REGIONS_CH4OXSTRATH2O |
| Stratospheric O3 | RF_STRATOZ_APPLY | RF_STRATOZ_SCALE | RF_EFFICACY_STRATOZ | RF_REGIONS_STRATOZ |
| Tropospheric O3 | RF_TROPOZ_APPLY | RF_TROPOZ_SCALE | RF_EFFICACY_TROPOZ | RF_REGIONS_TROPOZ |
| O3 due to N2O | (coupled) | - | RF_EFFICACY_OZDUETON2O | RF_REGIONS_OZDUETON2O |
| O3 due to temp | (coupled) | - | RF_EFFICACY_OZDUETOTEMPERATURE | RF_REGIONS_OZDUETOTEMPERATURE |
| F-gases | (always on) | RF_FGASSUM_SCALE | RF_EFFICACY_FGAS | RF_REGIONS_FGAS |
| Montreal halos | (always on) | RF_MHALOSUM_SCALE | RF_EFFICACY_MHALO | RF_REGIONS_MHALO |
| BC (industrial) | (always on) | RF_BC_SCALE | RF_EFFICACY_BC | RF_REGIONS_BC |
| OC (industrial) | (always on) | RF_OC_SCALE | RF_EFFICACY_OC | RF_REGIONS_OC |
| SOx (industrial) | (always on) | RF_SOX_SCALE | RF_EFFICACY_SOX | RF_REGIONS_SOX |
| NO3 (industrial) | RF_NO3_APPLY | - | RF_EFFICACY_NO3 | RF_REGIONS_NO3 |
| BC (biomass) | (coupled) | - | RF_EFFICACY_BC | RF_REGIONS_BC |
| OC (biomass) | (coupled) | - | RF_EFFICACY_OC | RF_REGIONS_OC |
| SOx (biomass) | (coupled) | - | RF_EFFICACY_SOX | RF_REGIONS_SOX |
| NO3 (biomass) | RF_NO3_APPLY | - | RF_EFFICACY_NO3 | RF_REGIONS_NO3 |
| Mineral dust | (external) | RF_MINERALDUST_SCALE | RF_EFFICACY_DUST | RF_REGIONS_DUST |
| Cloud albedo | RF_CLOUD_ALBEDO_AER_APPLY | RF_CLOUD_ALBEDO_AER_FACTOR | RF_EFFICACY_CLOUD_ALBEDO | RF_REGIONS_CLOUD_ALBEDO |
| Cloud cover | RF_CLOUD_COVER_AER_APPLY | RF_CLOUD_COVER_AER_FACTOR | RF_EFFICACY_CLOUD_COVER | RF_REGIONS_CLOUD_COVER |
| Land use | (external) | RF_LANDUSE_SCALE | RF_EFFICACY_LANDUSE | RF_REGIONS_LANDUSE |
| BC on snow | (external) | RF_BCSNOW_SCALE | RF_EFFICACY_BCSNOW | RF_REGIONS_BCSNOW |
| Aviation H2O | (coupled) | - | RF_EFFICACY_AIRH2O | RF_REGIONS_AIRH2O |
| Contrails | (coupled) | - | RF_EFFICACY_CONTRAIL | RF_REGIONS_CONTRAIL |
| Aviation cirrus | (coupled) | - | RF_EFFICACY_CIRRUS | RF_REGIONS_CIRRUS |
| Solar | (external) | RF_SOLAR_SCALE | RF_EFFICACY_SOLAR | RF_REGIONS_SOLAR |
| Volcanic | (external) | RF_VOLCANIC_SCALE | RF_EFFICACY_VOLC | RF_REGIONS_VOLC |
| Extra/External | RF_EXTRA_READ | RF_EXTRA_FACTOR | RF_EFFICACY_QXTRA | RF_REGIONS_QXTRA |

---

## 6. Aggregation Algorithm

### Pseudocode

```
SUBROUTINE DELTAQ

    ! ===== STEP 1: Calculate individual agent forcings =====
    ! (CH4, N2O chemistry, halocarbons, ozone, aerosols, etc.)
    ! See other module documentation for details

    ! ===== STEP 2: Build intermediate aggregates =====

    ! Direct aerosol total
    dat_totaer_dir_rf = dat_oci_rf + dat_bci_rf + dat_soxi_rf
                      + dat_no3i_rf + dat_biomassaer_rf + dat_mineraldust_rf

    ! CO2+CH4+N2O total
    dat_co2ch4n2o_rf = dat_co2_rf + dat_ch4_rf + dat_n2o_rf

    ! F-gas total
    dat_fgassum_rf = SUM(dat_fgas_rf(1:fgas_n))

    ! Montreal halocarbon total
    dat_mhalosum_rf = SUM(dat_mhalo_rf(1:mhalo_n))

    ! Halocarbon total
    dat_halosum_rf = dat_mhalosum_rf + dat_fgassum_rf

    ! Cloud indirect total
    dat_cloud_tot_rf = dat_cloud_cover_rf + dat_cloud_albedo_rf

    ! Ozone total
    dat_oztotal_rf = dat_stratoz_rf + dat_tropoz_rf
                   + dat_ozdueton2o_rf + dat_ozduetotemperature_rf

    ! Kyoto GHG total (CO2, CH4, N2O, F-gases)
    dat_kyotoghg_rf = dat_co2_rf + dat_ch4_rf + dat_n2o_rf + dat_fgassum_rf

    ! All GHG total (Kyoto + Montreal)
    dat_ghg_rf = dat_kyotoghg_rf + dat_mhalosum_rf

    ! Aerosol total
    dat_aerosol_rf = dat_totaer_dir_rf + dat_cloud_tot_rf

    ! ===== STEP 3: Total anthropogenic =====

    dat_total_anthro_rf = dat_ghg_rf
                        + dat_totaer_dir_rf
                        + dat_cloud_tot_rf
                        + dat_oztotal_rf
                        + dat_ch4oxstrath2o_rf
                        + dat_landuse_rf
                        + dat_bcsnow_rf
                        + dat_air_h2o_rf
                        + dat_air_contrail_rf
                        + dat_air_cirrus_rf

    ! ===== STEP 4: Select total based on run mode =====

    SELECT CASE (RF_TOTAL_RUNMODUS)
        CASE ('CO2')
            dat_total_rf = dat_co2_rf
        CASE ('GHG')
            dat_total_rf = dat_ghg_rf
        CASE ('CO2CH4N2O')
            dat_total_rf = dat_co2ch4n2o_rf
        CASE ('AEROSOL')
            dat_total_rf = dat_aerosol_rf
        CASE ('STRATO3')
            dat_total_rf = dat_stratoz_rf
        CASE ('TROPO3')
            dat_total_rf = dat_tropoz_rf
        CASE ('QEXTRA')
            dat_total_rf = dat_extra_rf
        CASE ('ANTHROPOGENIC')
            dat_total_rf = dat_total_anthro_rf
        CASE ('NONCO2EMISSIONS')
            ! CO2 emissions zeroed upstream
            dat_total_rf = dat_total_anthro_rf
        CASE ('NATURAL')
            dat_total_rf = dat_solar_rf
            ! Volcanic added monthly
        CASE ('ALL')
            dat_total_rf = dat_total_anthro_rf
                         + dat_solar_rf
                         + dat_extra_rf
            ! Volcanic added monthly
    END SELECT

    ! ===== STEP 5: Add volcanic (if applicable) =====

    IF (run_mode_includes_natural) THEN
        dat_total_inclvolcanic_rf = dat_total_rf + dat_volcanic_annual_rf
    ELSE
        dat_total_inclvolcanic_rf = dat_total_rf
    END IF

    ! ===== STEP 6: Apply freeze controls =====

    CALL ensure_constant_after_year(dat_total_rf, rf_total_constantafteryr)
    ! ... and for individual agents

    ! ===== STEP 7: NaN checking =====

    IF (is_nan(dat_total_effrf)) THEN
        ! Log which components are NaN
        ! Fatal error
    END IF

END SUBROUTINE
```

### Aggregation Hierarchy

```
TOTAL_RF
├── TOTAL_ANTHRO_RF
│   ├── GHG_RF
│   │   ├── KYOTOGHG_RF
│   │   │   ├── CO2_RF
│   │   │   ├── CH4_RF
│   │   │   ├── N2O_RF
│   │   │   └── FGASSUM_RF (sum of individual F-gases)
│   │   └── MHALOSUM_RF (sum of Montreal halos)
│   ├── AEROSOL_RF
│   │   ├── TOTAER_DIR_RF
│   │   │   ├── OCI_RF + OCB_RF
│   │   │   ├── BCI_RF + BCB_RF
│   │   │   ├── SOXI_RF + SOXB_RF
│   │   │   ├── NO3I_RF + NO3B_RF
│   │   │   └── MINERALDUST_RF
│   │   └── CLOUD_TOT_RF
│   │       ├── CLOUD_COVER_RF
│   │       └── CLOUD_ALBEDO_RF
│   ├── OZTOTAL_RF
│   │   ├── STRATOZ_RF
│   │   ├── TROPOZ_RF
│   │   ├── OZDUETON2O_RF
│   │   └── OZDUETOTEMPERATURE_RF
│   ├── CH4OXSTRATH2O_RF
│   ├── LANDUSE_RF
│   ├── BCSNOW_RF
│   ├── AIR_H2O_RF
│   ├── AIR_CONTRAIL_RF
│   └── AIR_CIRRUS_RF
├── SOLAR_RF
├── VOLCANIC_RF (added monthly, not in annual total)
└── EXTRA_RF (user-prescribed)
```

---

## 7. Pre-Industrial Reference

### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `RF_PREIND_REFERENCEYR` | Integer | Reference year for pre-industrial baseline (e.g., 1750) |
| `RF_INITIALIZATION_METHOD` | String | How to handle first-year forcing offset |

### Initialization Methods

#### ZEROSTARTSHIFT

Forces the radiative forcing to be zero at the start of the run by subtracting the first-year forcing value.

```fortran
IF (RF_INITIALIZATION_METHOD == 'ZEROSTARTSHIFT') THEN
    DAT_RF % FIRSTYEAROFFSET_DATGLOBE = DAT_RF % DATGLOBE(1)
    DAT_RF % FIRSTYEAROFFSET_DATBOX = DAT_RF % DATBOX(1, :)
END IF

! Then for all subsequent years:
! Effective forcing = Raw forcing - First year offset
```

**Use case:** When you want forcing relative to model start year regardless of actual pre-industrial concentrations.

#### JUMPSTART

Uses the pre-industrial reference value as the offset, allowing non-zero forcing at the start year.

```fortran
IF (RF_INITIALIZATION_METHOD == 'JUMPSTART') THEN
    DAT_RF % FIRSTYEAROFFSET_DATGLOBE = DAT_RF % PREIND_DATGLOBE
    DAT_RF % FIRSTYEAROFFSET_DATBOX = DAT_RF % PREIND_DATBOX
END IF
```

**Use case:** When starting simulations after the pre-industrial period (e.g., starting in 1850 but referencing 1750 pre-industrial).

### Helper Function

```fortran
SUBROUTINE set_rf_first_year_forcing_and_offset( &
    dat_rf, ref_rf_global, ref_rf_box, rf_initialisation_method)

    IF (trim(rf_initialisation_method) == "ZEROSTARTSHIFT") THEN
        aref_rf_global = ref_rf_global
        aref_rf_box = ref_rf_box
    ELSEIF (trim(rf_initialisation_method) == "JUMPSTART") THEN
        aref_rf_global = 0.0D0
        aref_rf_box(:) = 0.0D0
    END IF

    dat_rf % firstyearoffset_datglobe = aref_rf_global
    dat_rf % firstyearoffset_datbox = aref_rf_box

    CALL subtract_firstyearoffset(dat_rf, 1)
END SUBROUTINE
```

---

## 8. Freeze Controls

MAGICC allows forcing to be "frozen" (held constant) after a specified year, useful for scenario analysis and sensitivity experiments.

### Configuration Parameters

| Parameter | Affects |
|-----------|---------|
| `RF_TOTAL_CONSTANTAFTERYR` | Total forcing (all agents) |
| `RF_CO2_CONSTANTAFTERYR` | CO2 forcing |
| `RF_CH4_CONSTANTAFTERYR` | CH4 forcing |
| `RF_N2O_CONSTANTAFTERYR` | N2O forcing |
| `RF_TROPOZ_CONSTANTAFTERYR` | Tropospheric ozone |
| `RF_STRATOZ_CONSTANTAFTERYR` | Stratospheric ozone |
| `RF_AER_CONSTANTAFTERYR` | Aerosol forcing |
| `RF_LANDUSE_CONSTANTAFTERYR` | Land use forcing |
| `RF_MINERALDUST_CONSTANTAFTERYR` | Mineral dust forcing |
| `RF_MHALO_CONSTANTAFTERYR` | Montreal halocarbon forcing |
| `RF_FGAS_CONSTANTAFTERYR` | F-gas forcing |

### Implementation

```fortran
SUBROUTINE ensure_constant_after_year(dat, constantafteryr, year_idx)
    ! If current year >= constantafteryr, copy previous year's value
    IF (allyears(year_idx) >= constantafteryr) THEN
        dat % datglobe(year_idx) = dat % datglobe(year_idx - 1)
        dat % datbox(year_idx, :) = dat % datbox(year_idx - 1, :)
    END IF
END SUBROUTINE
```

### Application Order

Freeze controls are applied after aggregation:

```fortran
! After calculating total forcing
call ensure_constant_after_year(dat_total_rf, rf_total_constantafteryr, next_year_idx)
call ensure_constant_after_year(dat_total_srf, rf_total_constantafteryr, next_year_idx)

! For individual agents
call ensure_constant_after_year(dat_mineraldust_rf, rf_mineraldust_constantafteryr, next_year_idx)
call ensure_constant_after_year(dat_mineraldust_rf, rf_aer_constantafteryr, next_year_idx)
call ensure_constant_after_year(dat_mineraldust_rf, rf_total_constantafteryr, next_year_idx)
! Note: Multiple freeze controls can apply to same agent (most restrictive wins)
```

---

## 9. NaN Checking

DELTAQ performs comprehensive NaN detection to catch numerical errors early.

### Implementation Pattern

```fortran
any_nans = .FALSE.

IF (is_nan(DAT_TOTAL_EFFRF % DATGLOBE(NEXT_YEAR_IDX))) THEN
    any_nans = .TRUE.
    call logger % error('CORE', 'Your Forcing DAT_TOTAL_EFFRF is NaN')

    ! Drill down to identify source
    IF (is_nan(DAT_CO2_EFFRF % DATGLOBE(NEXT_YEAR_IDX))) THEN
        call logger % error('CORE', 'Your Forcing DAT_CO2_EFFRF is NaN')
    END IF
    IF (is_nan(DAT_CH4_EFFRF % DATGLOBE(NEXT_YEAR_IDX))) THEN
        call logger % error('CORE', 'Your Forcing DAT_CH4_EFFRF is NaN')
    END IF
    ! ... check all components

    DO FGAS_I = 1, FGAS_N
        IF (is_nan(DAT_FGAS_EFFRF(FGAS_I) % DATGLOBE(NEXT_YEAR_IDX))) THEN
            call logger % error('CORE', 'Your Forcing is NaN in '//FGAS_NAMES(FGAS_I))
        END IF
    END DO

    DO MHALO_I = 1, MHALO_N
        IF (is_nan(DAT_MHALO_EFFRF(MHALO_I) % DATGLOBE(NEXT_YEAR_IDX))) THEN
            call logger % error('CORE', 'Your Forcing is NaN in '//MHALO_NAMES(MHALO_I))
        END IF
    END DO
END IF

IF (any_nans) THEN
    call logger % fatal('CORE', 'NaN forcing found. Terminating')
END IF
```

### Checks by Run Mode

Each run mode has tailored NaN checking that only examines relevant agents:

| Run Mode | Agents Checked |
|----------|---------------|
| CO2 | CO2 |
| GHG | GHG, CO2, CH4, N2O, F-gases, MHALOs |
| AEROSOL | TOTAER_DIR, CLOUD_TOT |
| ALL | All agents |

---

## 10. Surface Forcing

In addition to top-of-atmosphere (TOA) forcing, MAGICC tracks surface radiative forcing using conversion factors.

### Surface Forcing Factors (SRF_FACTOR_*)

| Factor | Agent |
|--------|-------|
| `SRF_FACTOR_CO2` | CO2 |
| `SRF_FACTOR_CH4` | CH4 |
| `SRF_FACTOR_N2O` | N2O |
| `SRF_FACTOR_FGAS` | F-gases |
| `SRF_FACTOR_MHALO` | Montreal halos |
| `SRF_FACTOR_STRATOZ` | Stratospheric ozone |
| `SRF_FACTOR_TROPOZ` | Tropospheric ozone |
| `SRF_FACTOR_AIRH2O` | Aviation H2O |
| `SRF_FACTOR_CONTRAIL` | Contrails |
| `SRF_FACTOR_CIRRUS` | Cirrus |
| `SRF_FACTOR_CH4OXSTRATH2O` | CH4 stratospheric H2O |
| `SRF_FACTOR_LANDUSE` | Land use |
| `SRF_FACTOR_BCSNOW` | BC on snow |
| `SRF_FACTOR_SOLAR` | Solar |
| `SRF_FACTOR_VOLC` | Volcanic |
| `SRF_FACTOR_QXTRA` | External |

### Calculation

```fortran
! Example for total surface forcing
DAT_TOTAL_SRF % DATGLOBE(NEXT_YEAR_IDX) = &
    DAT_GHG_SRF % DATGLOBE(NEXT_YEAR_IDX) + &
    DAT_TOTAER_DIR_SRF % DATGLOBE(NEXT_YEAR_IDX) + &
    DAT_CLOUD_TOT_SRF % DATGLOBE(NEXT_YEAR_IDX) + &
    DAT_STRATOZ_RF % DATGLOBE(NEXT_YEAR_IDX) * SRF_FACTOR_STRATOZ + &
    DAT_TROPOZ_RF % DATGLOBE(NEXT_YEAR_IDX) * SRF_FACTOR_TROPOZ + &
    ! ... etc
```

---

## 11. Rust Implementation Considerations

### Core Data Structures

```rust
/// Radiative forcing aggregation configuration
pub struct RfAggregationConfig {
    /// Run mode determining which agents contribute to total
    pub run_modus: RfRunModus,

    /// Pre-industrial reference year
    pub preind_reference_yr: i32,

    /// Initialization method (ZEROSTARTSHIFT or JUMPSTART)
    pub initialization_method: RfInitMethod,

    /// Efficacy application flag
    pub efficacy_apply: bool,

    /// Freeze year controls
    pub freeze_years: FreezeYearConfig,
}

#[derive(Clone, Copy, PartialEq)]
pub enum RfRunModus {
    Co2,
    Ghg,
    Co2Ch4N2o,
    Aerosol,
    StratO3,
    TropO3,
    QExtra,
    Anthropogenic,
    NonCo2Emissions,
    Natural,
    All,
}

#[derive(Clone, Copy)]
pub enum RfInitMethod {
    ZeroStartShift,
    JumpStart,
}

/// Per-agent efficacy factors
pub struct EfficacyFactors {
    pub ch4: f64,
    pub n2o: f64,
    pub stratoz: f64,
    pub tropoz: f64,
    pub aer_dir: f64,
    pub bc: f64,
    pub oc: f64,
    pub sox: f64,
    pub no3: f64,
    pub cloud_cover: f64,
    pub cloud_albedo: f64,
    pub solar: f64,
    pub volcanic: f64,
    pub landuse: f64,
    pub bcsnow: f64,
    pub fgas: f64,
    pub mhalo: f64,
    pub qextra: f64,
    pub airh2o: f64,
    pub contrail: f64,
    pub cirrus: f64,
}

/// 4-box regional pattern
pub struct RegionalPattern {
    pub nh_land: f64,
    pub nh_ocean: f64,
    pub sh_land: f64,
    pub sh_ocean: f64,
}

impl RegionalPattern {
    /// Normalize pattern to sum to 1.0 when weighted by area
    pub fn normalize(&self, area_weights: &[f64; 4]) -> Self {
        let weighted_sum = self.nh_land * area_weights[0]
            + self.nh_ocean * area_weights[1]
            + self.sh_land * area_weights[2]
            + self.sh_ocean * area_weights[3];

        Self {
            nh_land: self.nh_land / weighted_sum,
            nh_ocean: self.nh_ocean / weighted_sum,
            sh_land: self.sh_land / weighted_sum,
            sh_ocean: self.sh_ocean / weighted_sum,
        }
    }
}
```

### Aggregation Trait

```rust
pub trait ForcingAggregator {
    /// Aggregate individual agent forcings into total
    fn aggregate(
        &self,
        agent_forcings: &AgentForcings,
        config: &RfAggregationConfig,
        year_idx: usize,
    ) -> Result<AggregatedForcing, AggregationError>;

    /// Apply efficacy factors to convert RF to ERF
    fn apply_efficacies(
        &self,
        rf: f64,
        efficacy: f64,
    ) -> f64;

    /// Calculate internal efficacy from regional pattern
    fn calculate_internal_efficacy(
        &self,
        q: f64,
        climate_sensitivity: f64,
        feedback_matrix: &[[f64; 4]; 4],
        regional_pattern: &RegionalPattern,
        area_fractions: &[f64; 4],
    ) -> f64;
}
```

### Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum AggregationError {
    #[error("NaN detected in {agent} forcing at year index {year_idx}")]
    NaNForcing { agent: String, year_idx: usize },

    #[error("Unknown run modus: {0}")]
    UnknownRunModus(String),

    #[error("Invalid initialization method: {0}")]
    InvalidInitMethod(String),

    #[error("Regional pattern not normalized: sum = {0}, expected 1.0")]
    UnnormalizedPattern(f64),
}
```

### Testing Requirements

1. **Run Mode Tests**: Verify correct agent inclusion/exclusion for each run mode
2. **Efficacy Tests**: Verify prescribed and internal efficacy calculations
3. **Regional Pattern Tests**: Test normalization and application
4. **Aggregation Tests**: Verify hierarchical summation matches Fortran
5. **Baseline Tests**: Test ZEROSTARTSHIFT vs JUMPSTART
6. **Freeze Tests**: Verify forcing held constant after freeze year
7. **NaN Detection Tests**: Ensure proper error propagation

---

## 12. References

1. Meinshausen, M., et al. (2011). "Emulating coupled atmosphere-ocean and carbon cycle models with a simpler model, MAGICC6"
2. Meinshausen, M., et al. (2020). "The shared socio-economic pathway (SSP) greenhouse gas concentrations and their extensions to 2500"
3. IPCC AR5/AR6 radiative forcing assessments
4. Hansen, J., et al. (2005). "Efficacy of climate forcings" - Original efficacy framework
