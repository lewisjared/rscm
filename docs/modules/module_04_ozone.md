# Module 04: Ozone Radiative Forcing

## 1. Scientific Purpose

This module calculates the radiative forcing from changes in atmospheric ozone concentrations. Unlike most forcing agents in MAGICC, ozone is NOT explicitly modeled as a concentration - instead, ozone radiative forcing is parameterized from precursor emissions, halocarbon loading, and temperature changes.

The module handles four distinct ozone forcing components:

1. **Stratospheric Ozone (STRATOZ)**: Ozone depletion from halogenated compound loading (EESC). This is primarily a COOLING (negative) forcing because ozone-depleting substances destroy stratospheric ozone.

2. **Tropospheric Ozone (TROPOZ)**: Ozone production from CH4, NOx, CO, and NMVOC emissions. This is a WARMING (positive) forcing because these precursors increase tropospheric ozone through photochemistry.

3. **Ozone due to N2O (OZDUETON2O)**: A separate component capturing N2O's effect on ozone through stratospheric chemistry. N2O becomes a source of reactive nitrogen (NOy) in the stratosphere which destroys ozone.

4. **Ozone due to Temperature (OZDUETOTEMPERATURE)**: Temperature feedback on ozone radiative forcing. Warming affects ozone photochemistry and transport, providing a (typically negative) feedback.

**Critical Design Decision**: These four components are calculated INDEPENDENTLY and then summed. There is no coupling between them, which is a simplification of the real atmospheric chemistry where these processes interact.

## 2. Mathematical Formulation

### 2.1 Stratospheric Ozone RF

The stratospheric ozone forcing is derived from Equivalent Effective Stratospheric Chlorine (EESC) loading, calculated in the halocarbon chemistry module.

```
RF_stratoz = STRATOZ_O3SCALE * max(0, (EESC - EESC_reference) / 100)^STRATOZ_CLEXPON
```

Where:

- `EESC` = current equivalent effective stratospheric chlorine loading (ppt)
- `EESC_reference` = EESC at the threshold year (typically 1979)
- `STRATOZ_O3SCALE` = scaling factor (default: -0.0043 W/m^2, note negative sign)
- `STRATOZ_CLEXPON` = exponent (default: 1.7)

**Important**: The forcing is zero for years before `STRATOZ_THRESHOLD_YEAR` (default: 1979).

**Source:** `deltaq_calculations.f90` lines 954-1002

### 2.2 Tropospheric Ozone RF

Tropospheric ozone forcing has two components:

#### 2.2.1 Methane Component

```
RF_tropoz_ch4 = TROPOZ_RADEFF_WM2PERDU * TROPOZ_OZCH4 * ln(CH4 / CH4_pi)
```

Where:

- `CH4` = current methane concentration (ppb)
- `CH4_pi` = pre-industrial methane concentration (ppb)
- `TROPOZ_OZCH4` = methane-to-ozone conversion factor (default: 5.7 DU)
- `TROPOZ_RADEFF_WM2PERDU` = radiative efficiency (default: 0.032 W/m^2/DU)

**Source:** `deltaq_calculations.f90` lines 1337-1354

#### 2.2.2 Non-Methane Components (NOx, CO, NMVOC)

```
RF_tropoz_other = TROPOZ_RADEFF_WM2PERDU * (
    TROPOZ_OZNOX * delta_NOx_emis +
    TROPOZ_OZCO * delta_CO_emis +
    TROPOZ_OZVOC * delta_NMVOC_emis
)
```

Where:

- `delta_*_emis` = emissions relative to pre-industrial (Mt/yr for NOx as N, Mt/yr for CO, Mt/yr for NMVOC)
- `TROPOZ_OZNOX` = NOx sensitivity (default: 0.168 DU per MtN/yr)
- `TROPOZ_OZCO` = CO sensitivity (default: 0.00396 DU per MtCO/yr)
- `TROPOZ_OZVOC` = NMVOC sensitivity (default: 0.01008 DU per MtNMVOC/yr)

The total tropospheric ozone forcing per hemisphere:

```
RF_tropoz_box = RF_box_fractions * (RF_tropoz_ch4 + RF_tropoz_other / hemisphere_area_fraction)
```

**Source:** `deltaq_calculations.f90` lines 1356-1427, `MAGICC7.f90` lines 5448-5553

### 2.3 Ozone due to N2O

```
RF_ozdueton2o = OZDUETON2O_RADEFF * (N2O - N2O_pi)
```

Where:

- `N2O` = current N2O concentration (ppb)
- `N2O_pi` = pre-industrial N2O concentration (ppb)
- `OZDUETON2O_RADEFF` = radiative efficiency (default: 0.0004827 W/m^2/ppb)

**Source:** `deltaq_calculations.f90` lines 1004-1016

### 2.4 Ozone due to Temperature

```
RF_ozduetotemperature = OZDUETOTEMPERATURE_SCALE * T_surface
```

Where:

- `T_surface` = global mean surface temperature anomaly (K)
- `OZDUETOTEMPERATURE_SCALE` = scaling factor (default: -0.037 W/m^2/K, note negative sign)

**Source:** `deltaq_calculations.f90` lines 1018-1029

### 2.5 Total Ozone Forcing

```
RF_oztotal = RF_stratoz + RF_tropoz + RF_ozdueton2o + RF_ozduetotemperature
```

**Source:** `MAGICC7.f90` lines 5999-6013

## 3. State Variables

**Critical Issue**: The ozone module does NOT have true state variables in the sense of concentrations that evolve over time. Ozone forcing is calculated diagnostically each timestep from other model outputs.

| Variable | Fortran Name | Units | Description |
|----------|--------------|-------|-------------|
| Stratospheric O3 RF | `DAT_STRATOZ_RF` | W/m^2 | RF from EESC-induced ozone depletion |
| Stratospheric O3 EffRF | `DAT_STRATOZ_EFFRF` | W/m^2 | After efficacy scaling |
| Stratospheric O3 ERF | `DAT_STRATOZ_ERF` | W/m^2 | Effective radiative forcing |
| Tropospheric O3 RF | `DAT_TROPOZ_RF` | W/m^2 | RF from precursor-induced ozone |
| Tropospheric O3 EffRF | `DAT_TROPOZ_EFFRF` | W/m^2 | After efficacy scaling |
| Tropospheric O3 ERF | `DAT_TROPOZ_ERF` | W/m^2 | Effective radiative forcing |
| Tropospheric O3 CH4 component | `DAT_TROPOZ_CH4_RF` | W/m^2 | CH4 contribution to trop O3 (diagnostic) |
| O3 from N2O RF | `DAT_OZDUETON2O_RF` | W/m^2 | N2O-induced ozone forcing |
| O3 from N2O EffRF | `DAT_OZDUETON2O_EFFRF` | W/m^2 | After efficacy scaling |
| O3 from N2O ERF | `DAT_OZDUETON2O_ERF` | W/m^2 | Effective radiative forcing |
| O3 from Temperature RF | `DAT_OZDUETOTEMPERATURE_RF` | W/m^2 | Temperature feedback forcing |
| O3 from Temperature EffRF | `DAT_OZDUETOTEMPERATURE_EFFRF` | W/m^2 | After efficacy scaling |
| O3 from Temperature ERF | `DAT_OZDUETOTEMPERATURE_ERF` | W/m^2 | Effective radiative forcing |
| Total Ozone RF | `DAT_OZTOTAL_RF` | W/m^2 | Sum of all four components |
| Total Ozone EffRF | `DAT_OZTOTAL_EFFRF` | W/m^2 | After efficacy scaling |
| Total Ozone ERF | `DAT_OZTOTAL_ERF` | W/m^2 | Effective radiative forcing |

**Source:** `datastore.f90` lines 212-246

## 4. Parameters

### 4.1 Stratospheric Ozone Parameters

| Parameter | Fortran Name | Default | Units | Description |
|-----------|--------------|---------|-------|-------------|
| EESC exponent | `STRATOZ_CLEXPON` | 1.7 | dimensionless | Power-law exponent for EESC-RF relationship |
| O3 scale factor | `STRATOZ_O3SCALE` | -0.0043 | W/m^2 | Scaling factor (negative for cooling) |
| Threshold year | `STRATOZ_THRESHOLD_YEAR` | 1979 | year | Year before which RF is zero |
| Br vs Cl scale | `STRATOZ_BR_VS_CL_SCALE` | 62.5 | dimensionless | Bromine ozone destruction efficiency relative to chlorine (in EESC calculation) |
| Constant after year | `RF_STRATOZ_CONSTANTAFTERYR` | 10000 | year | Hold RF constant after this year |

**Source:** `MAGCFG_DEFAULTALL.CFG` lines 661-664, `core.f90` line 105

### 4.2 Tropospheric Ozone Parameters

| Parameter | Fortran Name | Default | Units | Description |
|-----------|--------------|---------|-------|-------------|
| Radiative efficiency | `TROPOZ_RADEFF_WM2PERDU` | 0.032 | W/m^2/DU | RF per Dobson Unit |
| CH4 sensitivity | `TROPOZ_OZCH4` | 5.7 | DU | Ozone change per ln(CH4/CH4_pi) |
| NOx sensitivity | `TROPOZ_OZNOX` | 0.168 | DU per MtN/yr | Ozone per NOx emission |
| CO sensitivity | `TROPOZ_OZCO` | 0.00396 | DU per MtCO/yr | Ozone per CO emission |
| NMVOC sensitivity | `TROPOZ_OZVOC` | 0.01008 | DU per MtNMVOC/yr | Ozone per NMVOC emission |
| Constant after year | `RF_TROPOZ_CONSTANTAFTERYR` | 10000 | year | Hold RF constant after this year |

**Source:** `MAGCFG_DEFAULTALL.CFG` lines 665-669, `core.f90` lines 91-92

### 4.3 Aviation Sector Parameters

When sector emissions are included (SECTOR_INCLUDE != 'NOSECTOR'), aviation NOx has different effectiveness:

| Parameter | Fortran Name | Default | Units | Description |
|-----------|--------------|---------|-------|-------------|
| Aviation NOx scale | `AIR_TROPOZ_NOXAIR_SCALE` | 1.5 | dimensionless | Effectiveness ratio of aviation vs ground NOx |
| Aviation reference year | `AIR_TROPOZ_REFYR` | 2019 | year | Year for calibrating aviation NOx ratio |

**Source:** `MAGCFG_DEFAULTALL.CFG` lines 566-567

### 4.4 Ozone due to N2O Parameters

| Parameter | Fortran Name | Default | Units | Description |
|-----------|--------------|---------|-------|-------------|
| N2O radiative efficiency | `OZDUETON2O_RADEFF` | 0.0004827 | W/m^2/ppb | RF per ppb N2O increase |

**Source:** `MAGCFG_DEFAULTALL.CFG` line 670, `core.f90` line 97

### 4.5 Ozone due to Temperature Parameters

| Parameter | Fortran Name | Default | Units | Description |
|-----------|--------------|---------|-------|-------------|
| Temperature scale | `OZDUETOTEMPERATURE_SCALE` | -0.037 | W/m^2/K | RF per K warming (negative for feedback) |

**Source:** `MAGCFG_DEFAULTALL.CFG` line 671, `core.f90` line 97

### 4.6 Regional Distribution Parameters

Each component has a regional distribution pattern (4-box model: NH land, NH ocean, SH land, SH ocean):

| Parameter | Fortran Name | Default Values | Description |
|-----------|--------------|----------------|-------------|
| Strat O3 regions | `RF_REGIONS_STRATOZ` | -0.01189, -0.02267, -0.06251, -0.24036 | Preferentially polar, SH dominated |
| Trop O3 regions | `RF_REGIONS_TROPOZ` | 0.46565, 0.51646, 0.17687, 0.23793 | NH dominated (pollution sources) |
| N2O O3 regions | `RF_REGIONS_OZDUETON2O` | 1.0, 1.0, 1.0, 1.0 | Assumed uniform (well-mixed N2O) |
| Temp O3 regions | `RF_REGIONS_OZDUETOTEMPERATURE` | 1.0, 1.0, 1.0, 1.0 | Assumed uniform |

**Source:** `MAGCFG_DEFAULTALL.CFG` lines 549-552

### 4.7 Efficacy Parameters

| Parameter | Fortran Name | Default | Description |
|-----------|--------------|---------|-------------|
| Stratospheric O3 efficacy | `RF_EFFICACY_STRATOZ` | 1.0 | Prescribed efficacy |
| Tropospheric O3 efficacy | `RF_EFFICACY_TROPOZ` | 1.0 | Prescribed efficacy |
| N2O O3 efficacy | `RF_EFFICACY_OZDUETON2O` | 1.0 | Prescribed efficacy |
| Temperature O3 efficacy | `RF_EFFICACY_OZDUETOTEMPERATURE` | 1.0 | Prescribed efficacy |

**Source:** `MAGCFG_DEFAULTALL.CFG` lines 520-523

### 4.8 Surface Forcing Factors

| Parameter | Fortran Name | Default | Description |
|-----------|--------------|---------|-------------|
| Strat O3 surface factor | `SRF_FACTOR_STRATOZ` | -0.5 | Surface vs TOA forcing ratio |
| Trop O3 surface factor | `SRF_FACTOR_TROPOZ` | 0.6 | Surface vs TOA forcing ratio |

**Source:** `MAGCFG_DEFAULTALL.CFG` lines 655-656

## 5. Inputs (per timestep)

### 5.1 Stratospheric Ozone

| Variable | Units | Source Module | Required? | Fortran Variable |
|----------|-------|---------------|-----------|------------------|
| EESC concentration | ppt | Halocarbon (Module 3) | Yes | `DAT_EESC_CONC % DATGLOBE(next_year_idx)` |
| EESC at threshold year | ppt | Halocarbon (Module 3) | Yes | `DAT_EESC_CONC % DATGLOBE(yr_idx(threshold_year))` |

### 5.2 Tropospheric Ozone

| Variable | Units | Source Module | Required? | Fortran Variable |
|----------|-------|---------------|-----------|------------------|
| CH4 concentration | ppb | CH4 (Module 1) | Yes | `DAT_CH4_CONC` |
| Pre-industrial CH4 | ppb | CH4 (Module 1) | Yes | `DAT_CH4_CONC % PREIND_DATGLOBE` |
| NOx industrial emissions | MtN/yr | Emissions | Yes | `DAT_NOXI_EMIS` |
| NOx biomass emissions | MtN/yr | Emissions | Yes | `DAT_NOXB_EMIS` |
| CO industrial emissions | Mt/yr | Emissions | Yes | `DAT_COI_EMIS` |
| CO biomass emissions | Mt/yr | Emissions | Yes | `DAT_COB_EMIS` |
| NMVOC industrial emissions | Mt/yr | Emissions | Yes | `DAT_NMVOCI_EMIS` |
| NMVOC biomass emissions | Mt/yr | Emissions | Yes | `DAT_NMVOCB_EMIS` |

### 5.3 Ozone due to N2O

| Variable | Units | Source Module | Required? | Fortran Variable |
|----------|-------|---------------|-----------|------------------|
| N2O concentration | ppb | N2O (Module 2) | Yes | `DAT_N2O_CONC` |
| Pre-industrial N2O | ppb | N2O (Module 2) | Yes | `DAT_N2O_CONC % PREIND_DATGLOBE` |

### 5.4 Ozone due to Temperature

| Variable | Units | Source Module | Required? | Fortran Variable |
|----------|-------|---------------|-----------|------------------|
| Surface temperature anomaly | K | Climate core | Yes | `DAT_SURFACE_TEMP % DATGLOBE(current_year_idx)` |

**Note**: Temperature uses CURRENT year, not NEXT year, because next year's temperature hasn't been calculated yet. This introduces a one-year lag in the temperature feedback.

## 6. Outputs (per timestep)

| Variable | Units | Destination | Fortran Variable |
|----------|-------|-------------|------------------|
| Stratospheric O3 RF | W/m^2 | Total forcing | `DAT_STRATOZ_RF` |
| Stratospheric O3 EffRF | W/m^2 | Climate core | `DAT_STRATOZ_EFFRF` |
| Stratospheric O3 ERF | W/m^2 | Diagnostics | `DAT_STRATOZ_ERF` |
| Tropospheric O3 RF | W/m^2 | Total forcing | `DAT_TROPOZ_RF` |
| Tropospheric O3 EffRF | W/m^2 | Climate core | `DAT_TROPOZ_EFFRF` |
| Tropospheric O3 ERF | W/m^2 | Diagnostics | `DAT_TROPOZ_ERF` |
| N2O O3 RF | W/m^2 | Total forcing | `DAT_OZDUETON2O_RF` |
| N2O O3 EffRF | W/m^2 | Climate core | `DAT_OZDUETON2O_EFFRF` |
| N2O O3 ERF | W/m^2 | Diagnostics | `DAT_OZDUETON2O_ERF` |
| Temperature O3 RF | W/m^2 | Total forcing | `DAT_OZDUETOTEMPERATURE_RF` |
| Temperature O3 EffRF | W/m^2 | Climate core | `DAT_OZDUETOTEMPERATURE_EFFRF` |
| Temperature O3 ERF | W/m^2 | Diagnostics | `DAT_OZDUETOTEMPERATURE_ERF` |
| Total O3 RF | W/m^2 | Diagnostics | `DAT_OZTOTAL_RF` |
| Total O3 EffRF | W/m^2 | Diagnostics | `DAT_OZTOTAL_EFFRF` |
| Total O3 ERF | W/m^2 | Diagnostics | `DAT_OZTOTAL_ERF` |
| CH4 component of trop O3 RF | W/m^2 | Diagnostics | `DAT_TROPOZ_CH4_RF` |

## 7. Algorithm (Pseudocode)

```
SUBROUTINE CALCULATE_OZONE_FORCING(current_year_idx)

    next_year_idx = current_year_idx + 1

    ! =========== STRATOSPHERIC OZONE ===========
    ! (Calculated AFTER halogen loop in MAGICC7.f90, ~line 4739)

    ! Check if before threshold year
    IF (year(next_year_idx) <= STRATOZ_THRESHOLD_YEAR) THEN
        RF_stratoz = 0.0
    ELSE
        ! Get EESC values
        eesc_current = EESC(next_year_idx)
        eesc_reference = EESC(threshold_year_idx)

        ! Calculate delta loading
        delta_loading = eesc_current - eesc_reference

        ! Exponential parameterization
        IF (delta_loading > 0) THEN
            RF_stratoz = STRATOZ_O3SCALE * (delta_loading / 100)^STRATOZ_CLEXPON
        ELSE
            RF_stratoz = 0.0
        ENDIF
    ENDIF

    ! Regional distribution
    stratoz_fractions = normalize(RF_REGIONS_STRATOZ, area_weights)
    RF_stratoz_box = RF_stratoz * stratoz_fractions

    ! Apply constant-after-year caps
    ensure_constant_after_year(DAT_STRATOZ_RF, RF_STRATOZ_CONSTANTAFTERYR)
    ensure_constant_after_year(DAT_STRATOZ_RF, RF_TOTAL_CONSTANTAFTERYR)

    ! =========== OZONE DUE TO N2O ===========
    ! (Calculated at ~line 4762)

    ! First year initialization
    IF (current_year_idx == 1) THEN
        RF_ozn2o(1) = OZDUETON2O_RADEFF * (N2O(1) - N2O_pi)
    ENDIF

    RF_ozn2o(next) = OZDUETON2O_RADEFF * (N2O(next) - N2O_pi)

    ! Regional distribution (uniform by default)
    ozn2o_fractions = normalize(RF_REGIONS_OZDUETON2O, area_weights)
    RF_ozn2o_box = RF_ozn2o * ozn2o_fractions

    ! =========== OZONE DUE TO TEMPERATURE ===========
    ! (Calculated at ~line 4790)

    ! First year initialization
    IF (current_year_idx == 1) THEN
        RF_oztemp(1) = OZDUETOTEMPERATURE_SCALE * T_surface(1)
    ENDIF

    ! NOTE: Uses CURRENT year temperature (lagged by 1 year)
    RF_oztemp(next) = OZDUETOTEMPERATURE_SCALE * T_surface(current)

    ! Regional distribution
    oztemp_fractions = normalize(RF_REGIONS_OZDUETOTEMPERATURE, area_weights)
    RF_oztemp_box = RF_oztemp * oztemp_fractions

    ! =========== TROPOSPHERIC OZONE ===========
    ! (Calculated at ~line 5448)

    ! CH4 component (logarithmic)
    RF_tropoz_ch4_global = TROPOZ_RADEFF * TROPOZ_OZCH4 * ln(CH4 / CH4_pi)
    RF_tropoz_ch4_box = RF_tropoz_ch4_global  ! Same in all boxes initially

    ! Precursor emissions (linear, with optional aviation scaling)
    IF (SECTOR_INCLUDE /= 'NOSECTOR') THEN
        ! Complex aviation NOx treatment
        ! Calculate effective NOx with aviation vs ground-based weighting
        ! See lines 5464-5511 for the messy sector-specific code
        ...
    ELSE
        ! Standard calculation using calculate_tropospheric_ozone_rf()
        FOR each hemisphere (NH=1:2, SH=3:4):
            delta_nox = NOX_emis - NOX_pi_emis  ! both I and B sources
            delta_co = CO_emis - CO_pi_emis
            delta_nmvoc = NMVOC_emis - NMVOC_pi_emis

            RF_other_hemi = TROPOZ_RADEFF / hemi_area_fraction * (
                TROPOZ_OZNOX * delta_nox +
                TROPOZ_OZCO * delta_co +
                TROPOZ_OZVOC * delta_nmvoc
            )

            RF_tropoz_box(hemi) = rf_box_fractions(hemi) * (
                RF_tropoz_ch4_box(hemi) + RF_other_hemi
            )
        END FOR
    ENDIF

    ! Global is area-weighted sum
    RF_tropoz_global = sum(RF_tropoz_box * area_weights)

    ! First year offset handling
    IF (current_year_idx == 1) THEN
        set_rf_first_year_forcing_and_offset(...)
    ENDIF

    subtract_firstyearoffset(DAT_TROPOZ_RF, next_year_idx)

    ! Apply constant-after-year caps
    ensure_constant_after_year(DAT_TROPOZ_RF, RF_TROPOZ_CONSTANTAFTERYR)
    ensure_constant_after_year(DAT_TROPOZ_RF, RF_TOTAL_CONSTANTAFTERYR)

    ! =========== TOTAL OZONE ===========
    ! (Calculated at ~line 5999)

    DAT_OZTOTAL_RF = DAT_STRATOZ_RF + DAT_TROPOZ_RF + DAT_OZDUETON2O_RF + DAT_OZDUETOTEMPERATURE_RF
    DAT_OZTOTAL_EFFRF = DAT_STRATOZ_EFFRF + DAT_TROPOZ_EFFRF + DAT_OZDUETON2O_EFFRF + DAT_OZDUETOTEMPERATURE_EFFRF
    DAT_OZTOTAL_ERF = DAT_STRATOZ_ERF + DAT_TROPOZ_ERF + DAT_OZDUETON2O_ERF + DAT_OZDUETOTEMPERATURE_ERF

    ! =========== EFFICACY CONVERSION ===========
    ! (Handled by add_effrf_rf_efficacies_to_conversions at ~line 5718)

    ! Each component gets its own efficacy treatment
    add_to_efficacy_list(DAT_OZDUETON2O_*, RF_EFFICACY_OZDUETON2O, RF_INTEFFICACY_OZDUETON2O)
    add_to_efficacy_list(DAT_OZDUETOTEMPERATURE_*, RF_EFFICACY_OZDUETOTEMPERATURE, RF_INTEFFICACY_OZDUETOTEMPERATURE)
    ! Note: STRATOZ and TROPOZ are handled elsewhere in the efficacy processing

END SUBROUTINE
```

## 8. Numerical Considerations

### 8.1 Stability

All ozone calculations are algebraic (no differential equations solved), so there are no numerical stability concerns.

### 8.2 Potential Issues

1. **One-year lag in temperature feedback**: The temperature-ozone feedback uses `T_surface(current_year_idx)` instead of `T_surface(next_year_idx)` because the next year's temperature hasn't been calculated yet. This introduces a systematic one-year lag in the feedback response.

2. **Negative forcing handling**: The stratospheric ozone formula includes `max(0, delta_loading)` which prevents negative EESC changes (recovery) from producing positive forcing. This is correct physics but could mask errors if EESC goes negative.

3. **Exponential parameterization limits**: The comment in `deltaq_calculations.f90` notes: "This parameterisation needs to be updated because it is not valid for very high chlorine or bromine loading." For extreme scenarios, the exponential relationship may not be appropriate.

4. **First year offset bug (potential)**: At line 5539-5544, there's a TODO comment: "check bug - offset not actually pre-industrial or current_year_idx..." suggesting the first-year offset handling may be incorrect.

## 9. Issues and Concerns

### 9.1 Is This Really 4 Separate Sub-modules?

**Answer: Yes and No.**

From a PHYSICS perspective, these should be treated as coupled processes - stratospheric ozone affects UV reaching the troposphere, N2O and temperature both affect stratospheric chemistry, tropospheric ozone affects temperature which feeds back...

From a CODE perspective, they are entirely INDEPENDENT calculations that are simply summed:

- Stratospheric O3: depends only on EESC (halocarbons)
- Tropospheric O3: depends only on CH4, NOx, CO, NMVOC
- N2O O3: depends only on N2O concentration
- Temperature O3: depends only on temperature

**Recommendation for rewrite**: Consider keeping them as separate functions but grouping them into a single Ozone module class that makes their interdependencies (or lack thereof) explicit.

### 9.2 Where Does This Code Actually Live?

The ozone calculations are **scattered across multiple files**:

1. **Module definition**: `core.f90` (MOD_OZONE, lines 87-98) - just holds parameters
2. **RF calculation functions**: `deltaq_calculations.f90` (lines 954-1029, 1337-1427)
3. **Main calculation loop**: `MAGICC7.f90` (lines 4739-4760 for stratoz, 4762-4816 for N2O/temp, 5448-5553 for tropoz)
4. **Efficacy processing**: `MAGICC7.f90` (lines 5718-5724)
5. **Total aggregation**: `MAGICC7.f90` (lines 5999-6013)
6. **Regional parameters**: `radiative_forcing.f90` (lines 4-6, 23-25, 36-37)

**Recommendation for rewrite**: Consolidate into a single OzoneForcing class with clear method separation.

### 9.3 Hardcoded Values

1. **Division by 100 in EESC formula**: `(delta_loading / 100.0D0)**exponent` - the 100 appears to be a normalization constant but is hardcoded rather than configurable.

2. **Area fractions assumed**: The tropospheric ozone calculation assumes the 4-box model area fractions implicitly in the normalization.

### 9.4 Aviation NOx Complexity

When sector emissions are used (`SECTOR_INCLUDE != 'NOSECTOR'`), the tropospheric ozone calculation becomes much more complex with special treatment for aviation NOx effectiveness. This code (lines 5464-5511) is:

- Poorly documented
- Contains complex reference year adjustments
- Has a warning: `call logger % warning("deltaq", ".SECTOR file use is untested")`

### 9.5 Tropospheric Ozone Pre-industrial Reference

Tropospheric ozone uses `PREIND_DATBOX` for emissions, but these pre-industrial values must be set correctly during initialization. If pre-industrial emissions are wrong, the forcing will be systematically biased.

### 9.6 Missing Ozone Concentration Output

Unlike CH4 and N2O, there is no `DAT_O3_CONC` output. The ozone changes are implicit in the forcing calculation but never output as concentrations (Dobson Units). This makes validation against observations difficult.

### 9.7 Sign Conventions

- `STRATOZ_O3SCALE` is NEGATIVE (-0.0043) because ozone depletion causes cooling
- `OZDUETOTEMPERATURE_SCALE` is NEGATIVE (-0.037) because warming destroys ozone (negative feedback)
- `RF_REGIONS_STRATOZ` values are all NEGATIVE (representing relative cooling pattern)
- `RF_REGIONS_TROPOZ` values are all POSITIVE (representing relative warming pattern)

Be very careful with sign conventions during reimplementation!

## 10. Test Cases

### 10.1 Stratospheric Ozone Tests

```python
def test_stratoz_zero_before_threshold():
    """Stratospheric O3 RF should be zero before threshold year"""
    year = 1970
    threshold_year = 1979
    eesc_current = 1000  # Arbitrary
    eesc_reference = 500

    rf = calculate_stratospheric_ozone_rf(
        eesc_current, eesc_reference,
        exponent=1.7, scale=-0.0043,
        year=year, reference_year=threshold_year
    )

    assert rf == 0.0

def test_stratoz_negative_forcing():
    """Ozone depletion should cause negative (cooling) forcing"""
    year = 2000
    threshold_year = 1979
    eesc_current = 2000  # Peak EESC
    eesc_reference = 1000  # 1979 value

    rf = calculate_stratospheric_ozone_rf(
        eesc_current, eesc_reference,
        exponent=1.7, scale=-0.0043,
        year=year, reference_year=threshold_year
    )

    assert rf < 0  # Negative forcing from ozone depletion

def test_stratoz_recovery():
    """RF magnitude should decrease as EESC decreases"""
    eesc_peak = 2000
    eesc_recovery = 1500
    eesc_ref = 1000

    rf_peak = calculate_stratospheric_ozone_rf(eesc_peak, eesc_ref, ...)
    rf_recovery = calculate_stratospheric_ozone_rf(eesc_recovery, eesc_ref, ...)

    assert abs(rf_recovery) < abs(rf_peak)  # Less negative as EESC recovers

def test_stratoz_exponential_relationship():
    """RF should follow power-law with exponent 1.7"""
    eesc_ref = 1000
    eesc_1 = 1100  # delta = 100
    eesc_2 = 1200  # delta = 200

    rf_1 = calculate_stratospheric_ozone_rf(eesc_1, eesc_ref, exponent=1.7, ...)
    rf_2 = calculate_stratospheric_ozone_rf(eesc_2, eesc_ref, exponent=1.7, ...)

    # rf_2 / rf_1 should be (200/100)^1.7 = 3.25
    ratio = rf_2 / rf_1
    expected_ratio = 2.0 ** 1.7

    assert ratio == pytest.approx(expected_ratio, rel=0.01)
```

### 10.2 Tropospheric Ozone Tests

```python
def test_tropoz_ch4_logarithmic():
    """CH4 component should be logarithmic"""
    ch4_pi = 700  # ppb
    ch4_current = 1400  # Doubled

    radeff = 0.032
    ozch4 = 5.7

    rf = calculate_tropospheric_ozone_rf_due_to_methane_alone(
        ch4_current, ch4_pi, radeff, ozch4
    )

    expected = radeff * ozch4 * np.log(ch4_current / ch4_pi)
    # = 0.032 * 5.7 * ln(2) = 0.126 W/m^2

    assert rf == pytest.approx(expected, rel=0.01)

def test_tropoz_nox_linear():
    """NOx component should be linear in emissions"""
    delta_nox_1 = 10  # MtN/yr
    delta_nox_2 = 20  # MtN/yr

    # Other emissions zero
    delta_co = 0
    delta_nmvoc = 0
    radeff = 0.032
    oznox = 0.168

    rf_1 = radeff * oznox * delta_nox_1
    rf_2 = radeff * oznox * delta_nox_2

    assert rf_2 == pytest.approx(2 * rf_1)

def test_tropoz_preindustrial_zero():
    """RF should be zero at pre-industrial conditions"""
    ch4 = ch4_pi = 700
    nox_emis = nox_pi = 5
    co_emis = co_pi = 100
    nmvoc_emis = nmvoc_pi = 50

    # With all emissions at PI, forcing should be zero
    rf_ch4 = calculate_tropospheric_ozone_rf_due_to_methane_alone(ch4, ch4_pi, ...)
    # delta emissions are all zero
    rf_other = 0

    assert rf_ch4 == 0.0
```

### 10.3 Ozone from N2O Tests

```python
def test_ozn2o_linear():
    """N2O ozone forcing should be linear"""
    n2o_pi = 270  # ppb
    n2o_current = 330  # ppb
    radeff = 0.0004827

    rf = calculate_ozone_due_to_n2o_rf(n2o_current, n2o_pi, radeff)

    expected = radeff * (n2o_current - n2o_pi)
    # = 0.0004827 * 60 = 0.029 W/m^2

    assert rf == pytest.approx(expected)

def test_ozn2o_preindustrial_zero():
    """N2O ozone forcing at PI should be zero"""
    n2o = n2o_pi = 270

    rf = calculate_ozone_due_to_n2o_rf(n2o, n2o_pi, radeff=0.0004827)

    assert rf == 0.0
```

### 10.4 Ozone from Temperature Tests

```python
def test_oztemp_negative_feedback():
    """Temperature-ozone forcing should be negative for positive T"""
    T_anomaly = 2.0  # K warming
    scale = -0.037  # W/m^2/K

    rf = calculate_ozone_due_to_temperature_rf(T_anomaly, scale)

    expected = scale * T_anomaly  # = -0.074 W/m^2
    assert rf == pytest.approx(expected)
    assert rf < 0  # Negative feedback

def test_oztemp_zero_at_baseline():
    """No forcing when temperature anomaly is zero"""
    T_anomaly = 0.0
    scale = -0.037

    rf = calculate_ozone_due_to_temperature_rf(T_anomaly, scale)

    assert rf == 0.0
```

### 10.5 Integration Tests

```python
def test_total_ozone_is_sum():
    """Total ozone RF should be sum of 4 components"""
    # Run model for one timestep
    rf_stratoz = -0.03
    rf_tropoz = 0.15
    rf_ozn2o = 0.02
    rf_oztemp = -0.05

    rf_total = rf_stratoz + rf_tropoz + rf_ozn2o + rf_oztemp

    assert rf_total == pytest.approx(0.09)

def test_ar6_consistency():
    """Year 2019 forcing should be consistent with IPCC AR6"""
    # Run model with historical forcing to 2019
    # AR6 Table 7.5:
    # - Stratospheric O3: -0.02 [-0.10 to 0.03] W/m^2
    # - Tropospheric O3: 0.47 [0.24 to 0.70] W/m^2

    # Extract model values
    rf_stratoz_2019 = model.get("DAT_STRATOZ_RF", year=2019)
    rf_tropoz_2019 = model.get("DAT_TROPOZ_RF", year=2019)

    # Should be within assessed ranges
    assert -0.10 <= rf_stratoz_2019 <= 0.03
    assert 0.24 <= rf_tropoz_2019 <= 0.70
```

## 11. Fortran Code References

### 11.1 Module Definition

- **File:** `/Users/jared/code/magicc/magicc/src/libmagicc/core.f90`
- **Lines 87-98:** `MOD_OZONE` module - contains only parameter declarations

### 11.2 DataStore Definitions

- **File:** `/Users/jared/code/magicc/magicc/src/libmagicc/utils/datastore.f90`
- **Lines 212-214:** Tropospheric O3 data stores (RF, EFFRF, ERF, CH4 component)
- **Lines 243-246:** Stratospheric O3 and other O3 component data stores

### 11.3 Regional and Efficacy Parameters

- **File:** `/Users/jared/code/magicc/magicc/src/libmagicc/physics/radiative_forcing.f90`
- **Lines 4-6:** RF_REGIONS declarations for all ozone components
- **Lines 23-25:** RF_EFFICACY declarations
- **Lines 36-37:** RF_INTEFFICACY declarations
- **Lines 47-48:** SRF_FACTOR declarations

### 11.4 Physics Calculation Functions

- **File:** `/Users/jared/code/magicc/magicc/src/libmagicc/physics/deltaq_calculations.f90`
- **Lines 954-1002:** `calculate_stratospheric_ozone_rf`
- **Lines 1004-1016:** `calculate_ozone_due_to_n2o_rf`
- **Lines 1018-1029:** `calculate_ozone_due_to_temperature_rf`
- **Lines 1337-1354:** `calculate_tropospheric_ozone_rf_due_to_methane_alone`
- **Lines 1356-1427:** `calculate_tropospheric_ozone_rf` (full tropospheric calculation)

### 11.5 Main Calculation Loop

- **File:** `/Users/jared/code/magicc/magicc/src/libmagicc/MAGICC7.f90`
- **Lines 4739-4760:** Stratospheric ozone RF calculation
- **Lines 4762-4788:** N2O ozone component calculation
- **Lines 4790-4816:** Temperature ozone component calculation
- **Lines 5448-5553:** Tropospheric ozone RF calculation (including sector handling)
- **Lines 5718-5724:** Efficacy processing for N2O and temperature components
- **Lines 5999-6013:** Total ozone aggregation

### 11.6 Aviation/Sector Calculations

- **File:** `/Users/jared/code/magicc/magicc/src/libmagicc/MAGICC7.f90`
- **Lines 1782-1798:** Aviation NOx ratio and effectiveness calculations
- **Lines 5464-5511:** Sector-specific tropospheric ozone (when SECTOR_INCLUDE != 'NOSECTOR')

### 11.7 Configuration Defaults

- **File:** `/Users/jared/code/magicc/magicc/run/MAGCFG_DEFAULTALL.CFG`
- **Lines 661-671:** All ozone-related parameters (STRATOZ_*, TROPOZ_*, OZDUE*)
- **Lines 520-523:** Ozone efficacy parameters
- **Lines 549-552:** Regional distribution parameters

---

## Appendix A: Comparison of Ozone Components

| Component | Sign | Magnitude (2019) | Driver | Timescale |
|-----------|------|------------------|--------|-----------|
| Stratospheric O3 | Negative | ~-0.02 W/m^2 | EESC (halocarbons) | Decades |
| Tropospheric O3 | Positive | ~+0.47 W/m^2 | CH4, NOx, CO, NMVOC | Years |
| N2O-induced O3 | Positive | ~+0.03 W/m^2 | N2O concentration | Centuries |
| Temperature feedback | Negative | ~-0.04 W/m^2 | Global temperature | Immediate |

## Appendix B: Historical Context

The separation of ozone into these four components reflects the evolution of understanding:

1. **Original MAGICC (1990s)**: Only tropospheric ozone from precursors
2. **MAGICC5**: Added stratospheric ozone from EESC
3. **MAGICC6**: Refined parameterizations, added efficacies
4. **MAGICC7**: Added N2O ozone component and temperature feedback

The N2O ozone and temperature feedback components were added to match IPCC AR5/AR6 forcing assessments which explicitly quantify these contributions.

## Appendix C: Comparison with Other Models

| Feature | MAGICC | FaIR | Hector |
|---------|--------|------|--------|
| Stratospheric O3 | EESC-based | EESC-based | Not explicit |
| Tropospheric O3 | 4 precursors | CH4 + scaling | CH4 only |
| N2O O3 effect | Explicit | Not explicit | Not explicit |
| Temperature feedback | Explicit | Not explicit | Not explicit |
| Regional distribution | 4-box | Global | Global |
| Aviation sector | Optional | No | No |
