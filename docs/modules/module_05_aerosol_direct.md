# Module 5: Aerosol Direct Forcing

## 1. Scientific Purpose

The Aerosol Direct Forcing module calculates the radiative forcing from aerosol particles that directly interact with incoming solar radiation through scattering (cooling effect) and absorption (warming effect). This module handles:

- **Sulfate aerosols (SOx)**: Industrial and biomass burning sources - primarily scattering (cooling)
- **Black carbon (BC)**: Industrial and biomass burning sources - absorbing (warming)
- **Organic carbon (OC)**: Industrial and biomass burning sources - primarily scattering (cooling)
- **Mineral dust**: Related to land use changes - mixed scattering/absorbing
- **BC on snow**: Black carbon deposited on snow/ice surfaces - albedo reduction (warming)
- **Nitrate aerosols (NO3)**: Formed from NOx and NH3 emissions - scattering (cooling)

The module uses a harmonization system that allows radiative forcing to be scaled to match observed or target values at specific reference years, then extrapolated based on emission trajectories.

## 2. Mathematical Formulation

### 2.1 Core Harmonization Formula

For each aerosol species, the direct radiative forcing is calculated using a consistent harmonization approach:

```
RF(t) = RF_raw(t) * FACTOR
```

Where `FACTOR` is determined by the `_APPLY` parameter setting:

| APPLY Value | Behavior | Factor Calculation |
|-------------|----------|-------------------|
| 0 | No scaling | `FACTOR = 1.0` |
| 1 | Scale to target W/m^2 | `FACTOR = WM2 / RF_raw(YR)` |
| 2 | Use fixed factor | `FACTOR = FACTOR_param` |
| 3 | Use industrial factor | `FACTOR = RF_XXI_DIR_FACTOR` (for biomass species) |

The raw forcing is derived from emissions or optical thickness data, with extrapolation beyond historical data based on emission scaling.

### 2.2 Emission-Based Extrapolation

For years beyond historical forcing data, the forcing is extrapolated proportionally to emissions:

```
RF_box(i,t) = RF_box(i,switchyear) * (E_midyear(t) / E_midyear(switchyear))
```

Where `E_midyear` is the midyear-averaged emission (average of current and previous year), and `i` denotes the spatial box (NH Ocean, NH Land, SH Ocean, SH Land).

### 2.3 Nitrate Aerosol Formulation

Nitrate forcing uses a more complex parameterization based on Hauglustaine et al. (2014):

**Step 1: Alpha Factor (Burden Proxy)**

The alpha factor uses a multiplicative formula representing ammonia-nitrate equilibrium with sulfate competition:

```
ALPHA(t) = ([NOx-N(t)] * [NH3-N(t)]) / (1 + LAMBDA_SO2 * [SOx-S(t)] / [NH3-N(t)])
```

Where:
- All concentrations are in molar units (Tg divided by molecular weight: N=14, S=32)
- `LAMBDA_SO2` = `RF_NO3_LAMBDASO2` (default 1.6) represents sulfate priority for ammonia neutralization
- Natural emissions (`RF_NOXN_EMIS`, `RF_NH3N_EMIS`, `DAT_SOXN_EMIS`) are added to anthropogenic emissions

The formula captures that nitrate formation requires both NOx and NH3, but sulfate competes for available ammonia. Higher SOx reduces nitrate by consuming NH3 as ammonium sulfate.

Source: MAGICC7.f90 lines 1977-1988.

**Step 2: Atmospheric Burden**
```
B(NO3-)(t) = PREIND_BURDEN * (ALPHA(t) / ALPHA(0))
```

**Step 3: Optical Thickness**
```
OT(NO3-)(t) = PREIND_OT * (B(NO3-) / PREIND_BURDEN)^BURDEN2OT_EXP
```

**Step 4: Radiative Forcing**
```
RF(t) = WM2 * (OT(t) - PREIND_OT) / (OT(YR) - PREIND_OT)
```

**Step 5: Hemispheric Split**
```
RF_NH(t) = 2 * RF(t) * [NOx-NH][NH3-NH] / SUM([NOx-Y][NH3-Y])
RF_SH(t) = 2 * RF(t) * [NOx-SH][NH3-SH] / SUM([NOx-Y][NH3-Y])
```

### 2.4 BC on Snow Forcing

BC on snow forcing is scaled with combined BC emissions (industrial + biomass):

```
RF_BCSNOW(t) = RF_BCSNOW_scaled * (E_BCI(t) + E_BCB(t)) / (E_BCI(YR) + E_BCB(YR))
```

## 3. State Variables

### 3.1 Forcing Datastores

| Variable | Description | Units |
|----------|-------------|-------|
| `DAT_BCI_RF` | Black carbon industrial forcing | W/m^2 |
| `DAT_BCB_RF` | Black carbon biomass forcing | W/m^2 |
| `DAT_OCI_RF` | Organic carbon industrial forcing | W/m^2 |
| `DAT_OCB_RF` | Organic carbon biomass forcing | W/m^2 |
| `DAT_SOXI_RF` | Sulfate industrial forcing | W/m^2 |
| `DAT_SOXB_RF` | Sulfate biomass forcing | W/m^2 |
| `DAT_NO3I_RF` | Nitrate industrial forcing | W/m^2 |
| `DAT_NO3B_RF` | Nitrate biomass forcing | W/m^2 |
| `DAT_NO3T_RF` | Nitrate total forcing | W/m^2 |
| `DAT_MINERALDUST_RF` | Mineral dust forcing | W/m^2 |
| `DAT_BCSNOW_RF` | BC on snow forcing | W/m^2 |
| `DAT_BIOMASSAER_RF` | Combined biomass aerosol forcing | W/m^2 |
| `DAT_TOTAER_DIR_RF` | Total direct aerosol forcing | W/m^2 |
| `DAT_TOTAER_DIR_EFFRF` | Effective total direct forcing | W/m^2 |
| `DAT_TOTAER_DIR_ERF` | ERF total direct forcing | W/m^2 |

### 3.2 Regional Forcing Patterns

| Variable | Description |
|----------|-------------|
| `RF_REGIONS_AER_DIR` | 4-box regional forcing pattern for total direct aerosol |
| `RF_REGIONS_BC` | 4-box regional forcing pattern for BC |
| `RF_REGIONS_OC` | 4-box regional forcing pattern for OC |
| `RF_REGIONS_SOX` | 4-box regional forcing pattern for SOx |
| `RF_REGIONS_NO3` | 4-box regional forcing pattern for nitrate |
| `RF_REGIONS_BCSNOW` | 4-box regional forcing pattern for BC on snow |
| `RF_REGIONS_DUST` | 4-box regional forcing pattern for mineral dust |

### 3.3 Nitrate-Specific Variables

| Variable | Description | Units |
|----------|-------------|-------|
| `RF_NO3_ALPHAFACTOR(:)` | Alpha factor time series | molN equivalent |
| `RF_NO3_BURDEN(:)` | Nitrate burden time series | TgN |
| `RF_NO3_OT(:)` | Optical thickness time series | dimensionless |
| `RF_NO3T_RF_NH(:)` | NH forcing time series | W/m^2 |
| `RF_NO3T_RF_SH(:)` | SH forcing time series | W/m^2 |

## 4. Parameters

### 4.1 Black Carbon Industrial (BCI)

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `RF_BCI_DIR_APPLY` | 1 | flag | Scaling method selector |
| `RF_BCI_DIR_FACTOR` | 0.023 | - | Fixed scaling factor |
| `RF_BCI_DIR_YR` | 2019 | year | Reference year for scaling |
| `RF_BCI_DIR_WM2` | 0.155 | W/m^2 | Target forcing at reference year |

### 4.2 Black Carbon Biomass (BCB)

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `RF_BCB_DIR_APPLY` | 3 | flag | Use BCI factor (option 3) |
| `RF_BCB_DIR_FACTOR` | 1.0 | - | Not used when APPLY=3 |
| `RF_BCB_DIR_YR` | 2019 | year | Reference year |
| `RF_BCB_DIR_WM2` | 0.012 | W/m^2 | Target forcing |

### 4.3 Organic Carbon Industrial (OCI)

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `RF_OCI_DIR_APPLY` | 1 | flag | Scale to target |
| `RF_OCI_DIR_FACTOR` | 0.0074 | - | Fixed scaling factor |
| `RF_OCI_DIR_YR` | 2019 | year | Reference year |
| `RF_OCI_DIR_WM2` | -0.121 | W/m^2 | Target forcing (cooling) |

### 4.4 Organic Carbon Biomass (OCB)

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `RF_OCB_DIR_APPLY` | 3 | flag | Use OCI factor |
| `RF_OCB_DIR_FACTOR` | 1.0 | - | Not used when APPLY=3 |
| `RF_OCB_DIR_YR` | 2019 | year | Reference year |
| `RF_OCB_DIR_WM2` | -0.0064 | W/m^2 | Target forcing |

### 4.5 Sulfate Industrial (SOXI)

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `RF_SOXI_DIR_APPLY` | 1 | flag | Scale to target |
| `RF_SOXI_DIR_FACTOR` | 0.004 | - | Fixed scaling factor |
| `RF_SOXI_DIR_YR` | 2019 | year | Reference year |
| `RF_SOXI_DIR_WM2` | -0.177 | W/m^2 | Target forcing (cooling) |

### 4.6 Sulfate Biomass (SOXB)

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `RF_SOXB_DIR_APPLY` | 3 | flag | Use SOXI factor |
| `RF_SOXB_DIR_FACTOR` | 1.0 | - | Not used when APPLY=3 |
| `RF_SOXB_DIR_YR` | 2019 | year | Reference year |
| `RF_SOXB_DIR_WM2` | -0.0017 | W/m^2 | Target forcing |

### 4.7 Mineral Dust

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `RF_MINERALDUST_DIR_APPLY` | 1 | flag | Scale to target |
| `RF_MINERALDUST_DIR_FACTOR` | 0.159 | - | Fixed scaling factor |
| `RF_MINERALDUST_DIR_YR` | 2019 | year | Reference year |
| `RF_MINERALDUST_DIR_WM2` | -0.015 | W/m^2 | Target forcing |
| `RF_MINERALDUST_CONSTANTAFTERYR` | 2018 | year | Hold constant after this year |

### 4.8 BC on Snow

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `RF_BCSNOW_ALBEDO_APPLY` | 1 | flag | Scale to target |
| `RF_BCSNOW_ALBEDO_FACTOR` | 1.58 | - | Fixed scaling factor |
| `RF_BCSNOW_ALBEDO_YR` | 2019 | year | Reference year |
| `RF_BCSNOW_ALBEDO_WM2` | 0.08 | W/m^2 | Target forcing (warming) |

### 4.9 Nitrate Aerosol

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `RF_NO3T_DIR_APPLY` | 1 | flag | Scale to target |
| `RF_NO3T_DIR_FACTOR` | -0.0063 | - | Fixed scaling factor |
| `RF_NO3T_DIR_YR` | 2019 | year | Reference year |
| `RF_NO3T_DIR_WM2` | -0.044 | W/m^2 | Target forcing |
| `RF_NOXN_EMIS` | 10.0 | TgN/yr | Natural NOx emissions |
| `RF_NH3N_EMIS` | 21.0 | TgN/yr | Natural NH3 emissions |
| `RF_NO3_LAMBDASO2` | 1.6 | molN/molS | SO2-NH3 neutralization ratio |
| `RF_NO3_PREINDBURDEN` | 0.0085 | TgN | Preindustrial NO3 burden |
| `RF_NO3_PREINDOT` | 1.41 | - | Preindustrial optical thickness |
| `RF_NO3_BURDEN2OT_EXP` | 0.76 | - | Burden to optical thickness exponent |

### 4.10 Efficacy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RF_EFFICACY_AER_DIR` | 1.0 | Total direct aerosol efficacy |
| `RF_EFFICACY_BC` | 1.0 | BC efficacy |
| `RF_EFFICACY_OC` | 1.0 | OC efficacy |
| `RF_EFFICACY_SOX` | 1.0 | SOx efficacy |
| `RF_EFFICACY_NO3` | 1.0 | Nitrate efficacy |
| `RF_EFFICACY_BCSNOW` | 1.0 | BC on snow efficacy |
| `RF_EFFICACY_APPLY` | 2 | Efficacy application mode |

## 5. Inputs

### 5.1 Emission Time Series

| Input | Description | Units |
|-------|-------------|-------|
| `DAT_BCI_EMIS` | BC industrial emissions | Mt/yr |
| `DAT_BCB_EMIS` | BC biomass emissions | Mt/yr |
| `DAT_OCI_EMIS` | OC industrial emissions | Mt/yr |
| `DAT_OCB_EMIS` | OC biomass emissions | Mt/yr |
| `DAT_SOXI_EMIS` | SOx industrial emissions | MtS/yr |
| `DAT_SOXB_EMIS` | SOx biomass emissions | MtS/yr |
| `DAT_NOXI_EMIS` | NOx industrial emissions | MtN/yr |
| `DAT_NOXB_EMIS` | NOx biomass emissions | MtN/yr |
| `DAT_NH3I_EMIS` | NH3 industrial emissions | MtN/yr |
| `DAT_NH3B_EMIS` | NH3 biomass emissions | MtN/yr |
| `DAT_CO2B_CUMEMIS` | Cumulative CO2 biomass (for dust scaling) | GtC |

### 5.2 Optical Thickness / Forcing Files

| Input | Description |
|-------|-------------|
| `FILE_BCI_RF` / `FILE_BCI_OT` | BC industrial forcing or optical thickness |
| `FILE_BCB_RF` / `FILE_BCB_OT` | BC biomass forcing or optical thickness |
| `FILE_OCI_RF` / `FILE_OCI_OT` | OC industrial forcing or optical thickness |
| `FILE_OCB_RF` / `FILE_OCB_OT` | OC biomass forcing or optical thickness |
| `FILE_SOXI_RF` / `FILE_SOXI_OT` | SOx industrial forcing or optical thickness |
| `FILE_SOXB_RF` | SOx biomass forcing |
| `FILE_MINERALDUST_RF` | Mineral dust forcing |
| `FILE_BCSNOW_RF` | BC on snow forcing |

## 6. Outputs

### 6.1 Primary Forcing Outputs

| Output | Description | Units |
|--------|-------------|-------|
| `DAT_TOTAER_DIR_RF` | Total direct aerosol RF | W/m^2 |
| `DAT_TOTAER_DIR_EFFRF` | Total direct aerosol effective RF | W/m^2 |
| `DAT_BCI_RF`, `DAT_BCB_RF` | BC forcing by source | W/m^2 |
| `DAT_OCI_RF`, `DAT_OCB_RF` | OC forcing by source | W/m^2 |
| `DAT_SOXI_RF`, `DAT_SOXB_RF` | SOx forcing by source | W/m^2 |
| `DAT_NO3I_RF`, `DAT_NO3B_RF`, `DAT_NO3T_RF` | Nitrate forcing | W/m^2 |
| `DAT_MINERALDUST_RF` | Mineral dust forcing | W/m^2 |
| `DAT_BCSNOW_RF` | BC on snow forcing | W/m^2 |
| `DAT_BIOMASSAER_RF` | Combined biomass aerosol forcing | W/m^2 |

### 6.2 Surface Forcing

| Output | Description | Units |
|--------|-------------|-------|
| `DAT_TOTAER_DIR_SRF` | Total direct aerosol surface forcing | W/m^2 |

## 7. Algorithm

### 7.1 Initialization (JUMPONSTAGE)

```
1. Read or initialize all forcing datastores
2. For each aerosol species:
   a. Read historical forcing/OT data if available
   b. If historical data shorter than emissions:
      - Extend forcing using EXTRAP_RF_WITH_EMIS
   c. Apply scaling via RF_APPLY_SCALING:
      - If APPLY=0: no scaling
      - If APPLY=1: scale to match WM2 at YR
      - If APPLY=2: multiply by FACTOR
      - If APPLY=3: use industrial sector factor
   d. Store regional patterns for efficacy calculation
3. Calculate nitrate forcing using Hauglustaine parameterization
4. Split nitrate into hemispheric and industrial/biomass components
5. Calculate total direct aerosol forcing as sum of components
6. Derive regional forcing patterns for internal efficacy calculation
```

### 7.2 EXTRAP_RF_WITH_EMIS Procedure

```fortran
SUBROUTINE EXTRAP_RF_WITH_EMIS(SWITCH_APPLY, SCALE_FACTOR, SCALE_YR,
                                SCALE_WM2, DAT_EMIS1, DAT_EMIS2,
                                SWITCHYEAR, DAT_RF)

    ! Determine switch year index
    SWITCHYEAR_IDX = SWITCHYEAR - STARTYEAR + 1

    ! Extend forcing with emission scaling
    CALL extend_rf_box_from_switchyear_with_emissions(
        dat_target=dat_rf,
        switchyear_idx=switchyear_idx,
        extender=dat_emis1 % datbox + dat_emis2 % datbox
    )

    ! Calculate global from boxes
    DO I = 1, NYEARS
        DAT_RF % DATGLOBE(I) = SUM(DAT_RF % DATBOX(I,:) * GLOBALAREAFRACTIONS)
    END DO

    ! Apply scaling
    CALL RF_APPLY_SCALING(DAT_RF, SWITCH_APPLY, SCALE_YR, SCALE_WM2, SCALE_FACTOR)

    ! Apply first year offset (ZEROSTARTSHIFT or JUMPSTART)
    DAT_RF % DATGLOBE = DAT_RF % DATGLOBE - DAT_RF % FIRSTYEAROFFSET_DATGLOBE

END SUBROUTINE
```

### 7.3 RF_APPLY_SCALING Procedure

```fortran
SUBROUTINE RF_APPLY_SCALING(DAT_RF, SWITCH_APPLY, SCALE_YR, SCALE_WM2, SCALE_FACTOR)

    IF (SWITCH_APPLY == 0) THEN
        F = 1.0D0  ! No scaling
    ELSEIF (SWITCH_APPLY == 1) THEN
        ! Scale to match target W/m^2
        IF (DAT_RF % DATGLOBE(SCALE_YR_IDX) /= 0.0D0) THEN
            F = SCALE_WM2 / (DAT_RF % DATGLOBE(SCALE_YR_IDX) - DAT_RF % PREIND_DATGLOBE)
        ELSE
            F = 1.0D0
        END IF
    ELSEIF (SWITCH_APPLY == 2 .OR. SWITCH_APPLY == 3) THEN
        F = SCALE_FACTOR  ! Use provided factor
    END IF

    ! Apply scaling
    DAT_RF % DATGLOBE = DAT_RF % DATGLOBE * F
    DAT_RF % DATBOX = DAT_RF % DATBOX * F

END SUBROUTINE
```

### 7.4 Time Loop Contribution

During the main time loop (DELTAQ), the aerosol forcing is:
- Held constant after `RF_AER_CONSTANTAFTERYR` if specified
- Held constant after `RF_TOTAL_CONSTANTAFTERYR` if specified
- Combined with other forcings for total anthropogenic forcing

## 8. Numerical Considerations

### 8.1 Midyear Averaging

Emissions are provided as midyear values, while forcing is calculated for the start of each year. The extrapolation routine averages consecutive years:

```fortran
extender_midyear = (extender(i+1,:) + extender(i,:)) / 2.0D0
```

### 8.2 Division by Zero Protection

The code includes checks for zero denominators:
- In nitrate forcing calculation: `IF (denominator == 0.0D0) THEN rf = 0.0D0`
- In scaling: `IF (DAT_RF % DATGLOBE(YR_IDX) /= 0.0D0) THEN scale ELSE F=1.0`

### 8.3 Preindustrial Reference

The forcing is calculated relative to a preindustrial reference year (`RF_PREIND_REFERENCEYR`, default 1750). Two initialization methods are available:
- `JUMPSTART`: Forcing relative to preindustrial
- `ZEROSTARTSHIFT`: Forcing relative to first simulation year

### 8.4 Constant Forcing After Year

Multiple "constant after year" parameters can constrain the forcing:
- `RF_AER_CONSTANTAFTERYR` (default 10000 = disabled)
- `RF_MINERALDUST_CONSTANTAFTERYR` (default 2018)
- `RF_TOTAL_CONSTANTAFTERYR` (default 10000)

## 9. Issues and Concerns

### 9.1 Critical Issues

**SOx/NO3 APPLY=0 Override**: The code forcibly overrides `RF_SOXI_DIR_APPLY=0`, `RF_SOXB_DIR_APPLY=0`, and `RF_NO3T_DIR_APPLY=0` to `=1` with a warning:
```fortran
IF (RF_SOXI_DIR_APPLY == 0) THEN
    call logger % error("JUMPONSTAGE", "RF_SOXI_DIR_APPLY must not equal 0, setting to 1")
    RF_SOXI_DIR_APPLY = 1
END IF
```
This means users cannot disable scaling for sulfate and nitrate forcing - they are always scaled to a target. This is a silent behavioral override that could cause confusion.

**APPLY=3 Undocumented Behavior**: The `APPLY=3` option for biomass species (BCB, OCB, SOXB) uses the industrial sector's factor, but this is not well-documented and creates an implicit dependency between industrial and biomass parameters.

### 9.2 Architectural Issues

**Scattered Calculation Logic**: Aerosol forcing calculations are spread across:
- `JUMPONSTAGE` for initialization and scaling
- `EXTRAP_RF_WITH_EMIS` for extrapolation
- `RF_APPLY_SCALING` for factor application
- Various helper routines in `datastore.f90`

This makes the full calculation flow difficult to trace.

**Nitrate vs Other Aerosols**: Nitrate uses a fundamentally different calculation approach (emission-based parameterization from Hauglustaine et al.) compared to other aerosols (direct forcing files + emission scaling). This inconsistency is scientifically justified but architecturally awkward.

### 9.3 Numerical Concerns

**Switch Year Edge Cases**: The code has special handling for `SWITCHYEAR_IDX < 2` that directly uses emissions as forcing, which may produce unexpected results if historical forcing data is very short.

**Extrapolation Method Sensitivity**: The `extend_rf_box_from_switchyear_with_emissions` uses land emissions to scale both ocean and land boxes:
```fortran
! always scale with land emissions
extrapolation_values = extender_midyear(:, 2 * hemis_idx + 2)
```
This assumes a constant land-ocean forcing ratio, which may not hold under changing emission patterns.

### 9.4 Documentation Gaps

**Missing Comments**: Key scaling and extrapolation logic lacks inline comments explaining the scientific rationale.

**Parameter Dependencies**: The interplay between APPLY, FACTOR, YR, and WM2 parameters is complex and not well-documented. Users may not understand that:
- If APPLY=1, FACTOR is output (calculated), not input
- If APPLY=2, WM2 is ignored
- If APPLY=3, both FACTOR and WM2 are partially ignored

### 9.5 Potential Bugs

**RF_INITIALIZATION_METHOD Ignored for Nitrate**: A TODO comment indicates:
```fortran
! TODO: check intended. RF_INITIALIZATION_METHOD flag will do nothing as this
! setup hard-codes the first year of forcing to always be zero.
```

**Surface Forcing Excludes BCSNOW**: A comment notes:
```fortran
! note, the surface forcing is now NOT including the BCSnow component,
! as that mainly acts over ice/snow areas.. maybe less important for
! precipitation inference... to be checked.
```
This design decision should be validated.

## 10. Test Cases

### 10.1 Basic Scaling Tests

**Test: APPLY=1 Scales to Target**
```
Input:
  RF_BCI_DIR_APPLY = 1
  RF_BCI_DIR_WM2 = 0.155
  RF_BCI_DIR_YR = 2019
  Raw forcing at 2019 = 6.73 (arbitrary internal units)

Expected:
  FACTOR = 0.155 / 6.73 = 0.023
  Forcing at 2019 = 0.155 W/m^2
  All years scaled by factor 0.023
```

**Test: APPLY=2 Uses Fixed Factor**
```
Input:
  RF_BCI_DIR_APPLY = 2
  RF_BCI_DIR_FACTOR = 0.023

Expected:
  All forcing values multiplied by 0.023
  FACTOR output = 0.023 (unchanged)
```

**Test: APPLY=3 Uses Industrial Factor**
```
Input:
  RF_BCB_DIR_APPLY = 3
  RF_BCI_DIR_FACTOR = 0.023 (calculated from APPLY=1)

Expected:
  BCB forcing scaled by same factor as BCI
  RF_BCB_DIR_FACTOR = RF_BCI_DIR_FACTOR = 0.023
```

### 10.2 Nitrate Forcing Tests

**Test: Zero Emissions Produce Zero Forcing**
```
Input:
  All NOx emissions = 0
  All NH3 emissions = 0
  Natural emissions only

Expected:
  RF_NO3_ALPHAFACTOR = RF_NOXN_EMIS + RF_NH3N_EMIS = 31 TgN/yr
  ALPHA(t) = ALPHA(0) for all t
  RF_NO3T = 0 for all t
```

**Test: Hemispheric Split Conservation**
```
Input:
  Any valid emission scenario

Expected:
  RF_NO3T_RF_NH(t) + RF_NO3T_RF_SH(t) = 2 * RF_NO3T(t) for all t
  (Factor of 2 because hemispheric values are expressed per half-globe)
```

### 10.3 Emission Extrapolation Tests

**Test: Linear Emission Growth**
```
Input:
  Historical forcing to 2015
  Emissions growing linearly 2015-2100
  Emission(2015) = E0
  Emission(2100) = 2*E0

Expected:
  Forcing(2100)/Forcing(2015) approximately 2
  (Not exact due to midyear averaging)
```

**Test: Zero Emission Future**
```
Input:
  Historical forcing to 2015
  All emissions = 0 after 2015

Expected:
  Forcing approaches zero after 2015
  Rate depends on emission averaging window
```

### 10.4 Regional Pattern Tests

**Test: Regional Forcing Sums to Global**
```
Input:
  Any forcing scenario

Expected:
  For all years t:
  SUM(DAT_XXX_RF % DATBOX(t,:) * GLOBALAREAFRACTIONS) = DAT_XXX_RF % DATGLOBE(t)
```

### 10.5 Efficacy Tests

**Test: EFFICACY_APPLY=0 No Modification**
```
Input:
  RF_EFFICACY_APPLY = 0

Expected:
  EFFRF = RF for all forcing components
```

**Test: EFFICACY_APPLY=2 Uses Prescribed Efficacies**
```
Input:
  RF_EFFICACY_APPLY = 2
  RF_EFFICACY_BC = 1.5
  RF_INTEFFICACY_BC = 1.2 (calculated internally)

Expected:
  EFFRF_BC = (1.5 / 1.2) * RF_BC
  ERF_BC = EFFRF_BC * 1.2 = 1.5 * RF_BC
```

## 11. Fortran Code References

### 11.1 Primary Source Files

| File | Content |
|------|---------|
| `src/libmagicc/MAGICC7.f90` | Main calculation in JUMPONSTAGE, EXTRAP_RF_WITH_EMIS, RF_APPLY_SCALING |
| `src/libmagicc/physics/radiative_forcing.f90` | Module with RF parameters and regional patterns |
| `src/libmagicc/allcfgs.f90` | Parameter declarations in NML_ALLCFGS namelist |
| `src/libmagicc/utils/datastore.f90` | extend_rf_box_from_switchyear_with_emissions |
| `run/MAGCFG_DEFAULTALL.CFG` | Default parameter values |

### 11.2 Key Subroutines

**JUMPONSTAGE (MAGICC7.f90)**
- Lines ~1850-2500: Aerosol forcing initialization
- BC processing: lines 1851-1877
- OC processing: lines 1885-1904
- Nitrate processing: lines 1906-2180
- SOx processing: lines 2186-2220
- Mineral dust: lines 2381-2410
- Total direct aerosol: lines 2460-2485

**EXTRAP_RF_WITH_EMIS (MAGICC7.f90)**
- Lines 9441-9579
- Handles emission-based extrapolation and scaling

**RF_APPLY_SCALING (MAGICC7.f90)**
- Lines 9353-9396
- Core scaling logic for APPLY flag

**extend_rf_box_from_switchyear_with_emissions (datastore.f90)**
- Lines 572-688
- Box-level extrapolation with emission scaling

### 11.3 Key Variables Declaration

```fortran
! From mod_radiative_forcing.f90
INTEGER, SAVE :: RF_SOXI_DIR_APPLY, RF_BCI_DIR_APPLY, ...
REAL(8), SAVE :: RF_SOXI_DIR_FACTOR, RF_BCI_DIR_FACTOR, ...
INTEGER, SAVE :: RF_SOXI_DIR_YR, RF_BCI_DIR_YR, ...
REAL(8), SAVE :: RF_SOXI_DIR_WM2, RF_BCI_DIR_WM2, ...

REAL(8), SAVE :: RF_EFFICACY_BC, RF_EFFICACY_OC, RF_EFFICACY_SOX, ...
REAL(8), SAVE :: RF_INTEFFICACY_BC, RF_INTEFFICACY_OC, RF_INTEFFICACY_SOX, ...

REAL(8), DIMENSION(4), SAVE :: RF_REGIONS_AER_DIR, RF_REGIONS_BC, ...
```

### 11.4 Total Direct Aerosol Calculation

```fortran
! From MAGICC7.f90, lines ~2460-2470
DAT_TOTAER_DIR_RF % DATBOX = &
    DAT_OCI_RF % DATBOX + &
    DAT_BCI_RF % DATBOX + &
    DAT_SOXI_RF % DATBOX + &
    DAT_NO3I_RF % DATBOX + &
    DAT_BIOMASSAER_RF % DATBOX + &
    DAT_MINERALDUST_RF % DATBOX
```

---

## Summary

The Aerosol Direct Forcing module is a complex component that handles multiple aerosol species with different scientific formulations. The harmonization system (APPLY, FACTOR, YR, WM2) provides flexibility but adds complexity. Key concerns include:

1. Forced override of APPLY=0 for sulfate/nitrate
2. Implicit dependencies between industrial and biomass parameters (APPLY=3)
3. Different treatment of nitrate vs other aerosols
4. Limited documentation of parameter interactions

For a clean rewrite, consider:
- Making the harmonization system more explicit and documented
- Unifying the aerosol calculation approach where scientifically appropriate
- Providing clearer error messages when parameter combinations are invalid
- Separating the nitrate parameterization into its own submodule
