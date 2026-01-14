# Module 06: Aerosol Indirect Effects / Cloud Forcing

## 1. Scientific Purpose

This module calculates the aerosol indirect effects on climate through two mechanisms affecting cloud properties:

1. **Cloud Albedo Effect (First Indirect Effect / Twomey Effect)**: Increased aerosol loading provides more Cloud Condensation Nuclei (CCN), producing more but smaller cloud droplets. This increases cloud reflectivity (albedo) and causes a COOLING (negative) forcing.

2. **Cloud Cover/Lifetime Effect (Second Indirect Effect / Albrecht Effect)**: Smaller droplets take longer to coalesce into raindrops, potentially increasing cloud lifetime and coverage. This also causes a COOLING (negative) forcing.

**Critical Scientific Note**: The code comments reference "a simple formula by xxxxx as used in Hansen et al. (2005) 'Efficacies of Climate Forcing'" but the author attribution is missing ("xxxxx"). The module uses a simplified log-linear relationship between aerosol number concentrations and cloud droplet number concentrations.

### Aerosol Species Considered

The module tracks contributions from five aerosol species, each split by source type:

| Species | Industrial (I) | Biomass/Natural (B/N) | Description |
|---------|---------------|----------------------|-------------|
| Sulfate (SOX) | SOXI | SOXNB (biomass+natural) | Primary CCN source |
| Organic Carbon (OC) | OCI | OCB, OCN | Mixed hydrophobic/hydrophilic |
| Black Carbon (BC) | BCI | BCB | Primarily hydrophobic |
| Nitrate (NO3) | - | NO3 (total) | From NOx/NH3 emissions |
| Sea Salt (SS) | - | SSNAT | Natural only, constant |

**Four-Box Regional Structure**: Like other MAGICC forcing calculations, this module operates on a 4-box spatial structure:
- Box 1: Northern Hemisphere Ocean (NHO)
- Box 2: Northern Hemisphere Land (NHL)
- Box 3: Southern Hemisphere Ocean (SHO)
- Box 4: Southern Hemisphere Land (SHL)

## 2. Mathematical Formulation

### 2.1 Normalized Aerosol Number Index

For each aerosol species, a normalized number index (NI) is calculated using optical thickness (OT) or direct RF as a proxy:

```
NI_species(t, region) = [OT_species(t, region) or RF_species(t, region)] / NORM_species
```

Where `NORM_species` is the area-weighted global average at a normalization year (typically the historical-to-future transition year):

```
NORM_species = FGNO * NI(t_norm, 1) + FGNL * NI(t_norm, 2) + FGSO * NI(t_norm, 3) + FGSL * NI(t_norm, 4)
```

Area fractions (from `mod_areas`):
- FGNO = NH Ocean fraction
- FGNL = NH Land fraction
- FGSO = SH Ocean fraction
- FGSL = SH Land fraction

### 2.2 Total Aerosol Number Concentration

The total normalized aerosol number concentration is a weighted sum of species contributions:

```
NA_TOT(t, region) = w_NO3 * NI_NO3 +
                    w_BC * (NI_BCI + NI_BCB) +
                    w_OC * (NI_OCI + NI_OCB + NI_OCN) +
                    w_SOX * (NI_SOXI + NI_SOXNB) +
                    w_SS * NI_SSNAT
```

**Default Weights** (from MAGCFG_DEFAULTALL.CFG):
| Species | Weight | Normalized Weight |
|---------|--------|-------------------|
| SOX | 0.265 | 26.5% |
| OC | 0.265 | 26.5% |
| SS | 0.265 | 26.5% |
| NO3 | 0.163 | 16.3% |
| BC | 0.041 | 4.1% |

**Note**: BC has low weight because fossil BC (BCI) is primarily hydrophobic and poor CCN. A `CLOUD_BCI2BCB_SOLUBLE_RATIO = 0.75` further reduces BCI contribution relative to BCB.

### 2.3 Cloud Droplet Number Concentration (CDNC)

The CDNC is calculated using a logarithmic relationship:

```
CDNC(t, region) = log10(NA_TOT(t, region))
```

**Historical Note**: The code contains a commented-out Gultepe and Isaac parameterization:
- Ocean: `CDNC = 162 * log10(NA) - 273`
- Land: `CDNC = 298 * log10(NA) - 595`

However, these parameters are effectively removed because subsequent normalization steps negate their effect. The current implementation uses only the log10 relationship.

### 2.4 Change in CDNC

```
DELTA_CDNC(t, region) = CDNC(t, region) - CDNC_preindustrial(region)
```

### 2.5 Albedo Effect Forcing

The albedo forcing is scaled using prescribed regional patterns:

```
DELTA_CDNC_ALBEDO(t, region) = DELTA_CDNC(t, region) / DELTA_CDNC(t_norm, region) * RF_REGIONS_CLOUD_ALBEDO(region)
```

**Default Regional Patterns** (`RF_REGIONS_CLOUD_ALBEDO`):
- NHO: -0.966 W/m^2
- NHL: -1.399 W/m^2
- SHO: -0.342 W/m^2
- SHL: -0.628 W/m^2

These regional patterns are taken from Hansen et al. (2005), Figure 13.

### 2.6 Cloud Cover Effect Forcing

Similarly:

```
DELTA_CDNC_COVER(t, region) = DELTA_CDNC(t, region) / DELTA_CDNC(t_norm, region) * RF_REGIONS_CLOUD_COVER(region)
```

**Default Regional Patterns** (`RF_REGIONS_CLOUD_COVER`):
- NHO: -1.333 W/m^2
- NHL: -1.581 W/m^2
- SHO: -0.529 W/m^2
- SHL: -0.811 W/m^2

### 2.7 Final RF Calculation with Harmonization

The final forcing is harmonized to a target value in a reference year:

```
IF (RF_CLOUD_*_AER_APPLY == 1):
    SCALE_FACTOR = RF_CLOUD_*_AER_WM2 / NORM_DELTA_CDNC_*

RF_DATBOX(t, region) = DELTA_CDNC_*(t, region) * SCALE_FACTOR
```

**Default Harmonization Values** (AR6 calibration):
- Albedo Effect: RF_CLOUD_ALBEDO_AER_WM2 = -0.89 W/m^2 in 2019
- Cover Effect: RF_CLOUD_COVER_AER_WM2 = 0.0 W/m^2 in 2019

**Note**: The default configuration sets cloud cover forcing to ZERO, meaning only the albedo (Twomey) effect is active.

### 2.8 Total Cloud Forcing

```
RF_CLOUD_TOT = RF_CLOUD_ALBEDO + RF_CLOUD_COVER
```

## 3. State Variables

### 3.1 CLOUDSTORE Type Members

| Variable | Dimensions | Description | Units |
|----------|------------|-------------|-------|
| `NI_BCI` | (NYEARS, 4) | Normalized BC industrial number index | dimensionless |
| `NI_BCB` | (NYEARS, 4) | Normalized BC biomass number index | dimensionless |
| `NI_OCI` | (NYEARS, 4) | Normalized OC industrial number index | dimensionless |
| `NI_OCB` | (NYEARS, 4) | Normalized OC biomass number index | dimensionless |
| `NI_OCN` | (NYEARS, 4) | Normalized OC natural number index | dimensionless |
| `NI_SOXI` | (NYEARS, 4) | Normalized SOX industrial number index | dimensionless |
| `NI_SOXNB` | (NYEARS, 4) | Normalized SOX biomass+natural number index | dimensionless |
| `NI_NO3` | (NYEARS, 4) | Normalized nitrate number index | dimensionless |
| `NI_SSNAT` | (NYEARS, 4) | Normalized sea salt natural number index | dimensionless |
| `NA_TOT` | (NYEARS, 4) | Total weighted aerosol number concentration | dimensionless |
| `CDNC` | (NYEARS, 4) | Cloud Droplet Number Concentration | log10(NA_TOT) |
| `DELTA_CDNC` | (NYEARS, 4) | Change in CDNC from pre-industrial | dimensionless |
| `DELTA_CDNC_ALBEDO` | (NYEARS, 4) | CDNC change scaled for albedo effect | W/m^2 |
| `DELTA_CDNC_COVER` | (NYEARS, 4) | CDNC change scaled for cover effect | W/m^2 |
| `RF_DATBOX` | (NYEARS, 4) | Final radiative forcing by box | W/m^2 |

### 3.2 Pre-industrial Reference Values

Each NI variable has a corresponding `PREIND_DATBOX_NI_*` (dimension 4) storing the pre-industrial reference values.

### 3.3 Normalization Factors

| Variable | Description |
|----------|-------------|
| `NORM_NO3` | Global weighted mean NI_NO3 at normyear |
| `NORM_BC` | Global weighted mean NI_BC at normyear |
| `NORM_OC` | Global weighted mean NI_OC at normyear |
| `NORM_SOX` | Global weighted mean NI_SOX at normyear |
| `NORM_SS` | Global weighted mean NI_SS at normyear |
| `NORM_DELTA_CDNC_ALBEDO` | Global weighted delta CDNC for albedo normalization |
| `NORM_DELTA_CDNC_COVER` | Global weighted delta CDNC for cover normalization |

## 4. Parameters

### 4.1 Species Weights (CCN Contribution)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLOUD_WEIGHT_SOX` | 0.265 | SOX contribution weight |
| `CLOUD_WEIGHT_OC` | 0.265 | OC contribution weight |
| `CLOUD_WEIGHT_SS` | 0.265 | Sea salt contribution weight |
| `CLOUD_WEIGHT_NO3` | 0.163 | Nitrate contribution weight |
| `CLOUD_WEIGHT_BC` | 0.041 | BC contribution weight |
| `CLOUD_BCI2BCB_SOLUBLE_RATIO` | 0.75 | Ratio reducing BCI relative to BCB (hydrophobic vs hydrophilic) |

### 4.2 Harmonization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RF_CLOUD_ALBEDO_AER_APPLY` | 1 | Apply harmonization (0/1/2/3) |
| `RF_CLOUD_ALBEDO_AER_YR` | 2019 | Harmonization year |
| `RF_CLOUD_ALBEDO_AER_WM2` | -0.89 | Target forcing in harmonization year (W/m^2) |
| `RF_CLOUD_ALBEDO_AER_FACTOR` | 1.07 | Scaling factor (calculated or prescribed) |
| `RF_CLOUD_COVER_AER_APPLY` | 1 | Apply harmonization (0/1/2/3) |
| `RF_CLOUD_COVER_AER_YR` | 2019 | Harmonization year |
| `RF_CLOUD_COVER_AER_WM2` | 0.0 | Target forcing (W/m^2) - NOTE: DEFAULT IS ZERO |
| `RF_CLOUD_COVER_AER_FACTOR` | 0.0 | Scaling factor |

### 4.3 Regional Forcing Patterns

| Parameter | Default Values (NHO, NHL, SHO, SHL) |
|-----------|-------------------------------------|
| `RF_REGIONS_CLOUD_ALBEDO` | -0.966, -1.399, -0.342, -0.628 |
| `RF_REGIONS_CLOUD_COVER` | -1.333, -1.581, -0.529, -0.811 |
| `RF_REGIONS_NORMYEAR` | Year for regional pattern normalization |

### 4.4 Forcing Limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLOUD_APPLY_LIMIT_MAX` | 0 | Apply maximum forcing limit (0=no, 1=yes) |
| `CLOUD_LIMIT_MAX` | 0.0 | Maximum positive forcing allowed (W/m^2) |

### 4.5 Efficacy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RF_EFFICACY_CLOUD_ALBEDO` | 1.0 | Efficacy factor for albedo effect |
| `RF_EFFICACY_CLOUD_COVER` | 1.0 | Efficacy factor for cover effect |

### 4.6 Surface Forcing Factors

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SRF_FACTOR_CLOUD_ALBEDO` | 0.8 | Surface to TOA forcing ratio for albedo |
| `SRF_FACTOR_CLOUD_COVER` | 0.9 | Surface to TOA forcing ratio for cover |

## 5. Inputs

### 5.1 Optical Thickness Data (Historical)

| Input Datastore | Description |
|-----------------|-------------|
| `DAT_BCI_OT` | Black carbon industrial optical thickness |
| `DAT_BCB_OT` | Black carbon biomass optical thickness |
| `DAT_OCI_OT` | Organic carbon industrial optical thickness |
| `DAT_OCB_OT` | Organic carbon biomass optical thickness |
| `DAT_OCN_OT` | Organic carbon natural optical thickness |
| `DAT_SOXI_OT` | Sulfate industrial optical thickness |
| `DAT_SOXNB_OT` | Sulfate biomass+natural optical thickness |
| `DAT_SS_OT` | Sea salt optical thickness |

### 5.2 Radiative Forcing Data (for NO3)

| Input Datastore | Description |
|-----------------|-------------|
| `DAT_NO3T_RF` | Total nitrate radiative forcing (used as proxy for number index) |

### 5.3 Emission Data (for Future Scaling)

| Input Datastore | Description |
|-----------------|-------------|
| `DAT_NOXI_EMIS` | NOx industrial emissions |
| `DAT_NOXB_EMIS` | NOx biomass emissions |
| `DAT_SOXI_EMIS` | SOX industrial emissions |
| `DAT_SOXB_EMIS` | SOX biomass emissions |
| `DAT_SOXN_EMIS` | SOX natural emissions |
| `DAT_BCI_EMIS` | BC industrial emissions |
| `DAT_BCB_EMIS` | BC biomass emissions |
| `DAT_OCI_EMIS` | OC industrial emissions |
| `DAT_OCB_EMIS` | OC biomass emissions |

## 6. Outputs

### 6.1 Radiative Forcing Outputs

| Output Datastore | Description |
|------------------|-------------|
| `DAT_CLOUD_ALBEDO_RF` | Cloud albedo (Twomey) effect RF |
| `DAT_CLOUD_ALBEDO_EFFRF` | Cloud albedo effective RF (with efficacy) |
| `DAT_CLOUD_ALBEDO_ERF` | Cloud albedo ERF |
| `DAT_CLOUD_COVER_RF` | Cloud cover/lifetime effect RF |
| `DAT_CLOUD_COVER_EFFRF` | Cloud cover effective RF |
| `DAT_CLOUD_COVER_ERF` | Cloud cover ERF |
| `DAT_CLOUD_TOT_RF` | Total cloud indirect RF |
| `DAT_CLOUD_TOT_EFFRF` | Total cloud indirect effective RF |
| `DAT_CLOUD_TOT_ERF` | Total cloud indirect ERF |
| `DAT_CLOUD_TOT_SRF` | Total cloud surface RF |

Each datastore contains:
- `DATGLOBE(NYEARS)` - Global mean time series
- `DATBOX(NYEARS, 4)` - 4-box regional time series

## 7. Algorithm

### 7.1 Main Subroutine: `cloud_calc_ind_aerosol`

**Source:** `cloudstore.f90` lines 92-828

**Step-by-Step Algorithm:**

```
1. PROCESS NITRATE (NO3)
   a. Determine IDX_FUTYR (historical-to-future transition year)
   b. Copy NO3 RF data as proxy for number index: NI_NO3 = DAT_NO3T_RF
   c. Set pre-industrial reference to 0.0
   d. Calculate NORM_NO3 (global weighted average at IDX_FUTYR)
   e. Normalize: NI_NO3 = NI_NO3 / NORM_NO3
   f. Scale future years with NOx emissions

2. PROCESS ORGANIC CARBON (OC)
   a. Update IDX_FUTYR from OC optical thickness data
   b. Copy optical thickness: NI_OCI = DAT_OCI_OT, NI_OCB = DAT_OCB_OT, NI_OCN = DAT_OCN_OT
   c. Copy pre-industrial values
   d. Calculate NORM_OC (sum of all OC components)
   e. Normalize all NI_OC* arrays
   f. Scale future years with OC emissions
   g. Hold natural OC constant after IDX_FUTYR

3. PROCESS BLACK CARBON (BC)
   a. Update IDX_FUTYR from BC optical thickness data
   b. Copy optical thickness with solubility adjustment:
      NI_BCI = DAT_BCI_OT * CLOUD_BCI2BCB_SOLUBLE_RATIO
      NI_BCB = DAT_BCB_OT
   c. Copy pre-industrial values
   d. Calculate NORM_BC
   e. Normalize NI_BC* arrays
   f. Scale future years with BC emissions

4. PROCESS SULFATE (SOX)
   a. Update IDX_FUTYR from SOX optical thickness data
   b. Copy optical thickness: NI_SOXI = DAT_SOXI_OT, NI_SOXNB = DAT_SOXNB_OT
   c. Copy pre-industrial values
   d. Calculate NORM_SOX
   e. Normalize NI_SOX* arrays
   f. Scale future years with SOX emissions

5. PROCESS SEA SALT (SS)
   a. Update IDX_FUTYR
   b. Copy optical thickness: NI_SSNAT = DAT_SS_OT
   c. Copy pre-industrial values
   d. Calculate NORM_SS
   e. Normalize NI_SSNAT
   f. Hold constant after IDX_FUTYR (natural source assumed constant)

6. NORMALIZE SPECIES WEIGHTS
   Sum weights and renormalize to ensure they sum to 1.0

7. CALCULATE TOTAL AEROSOL NUMBER
   NA_TOT = weighted sum of all NI_* species
   Include pre-industrial calculation

8. CALCULATE CDNC
   CDNC = log10(NA_TOT)
   PREIND_DATBOX_CDNC = log10(PREIND_NA_TOT)

9. CALCULATE DELTA CDNC
   DELTA_CDNC = CDNC - PREIND_DATBOX_CDNC

10. CALCULATE ALBEDO EFFECT
    a. Scale DELTA_CDNC by regional patterns (RF_REGIONS_CLOUD_ALBEDO)
    b. Normalize by pattern at normyear
    c. Apply harmonization scaling via CLOUD_APPLY_SCALING subroutine

11. CALCULATE COVER EFFECT
    a. Scale DELTA_CDNC by regional patterns (RF_REGIONS_CLOUD_COVER)
    b. Normalize by pattern at normyear
    c. Apply harmonization scaling

12. APPLY CONSTANT-AFTER-YEAR
    Hold forcing constant after RF_AER_CONSTANTAFTERYR

13. SUM TOTAL CLOUD FORCING
    DAT_CLOUD_TOT_RF = DAT_CLOUD_COVER_RF + DAT_CLOUD_ALBEDO_RF
```

### 7.2 Scaling Subroutine: `CLOUD_APPLY_SCALING`

**Source:** `cloudstore.f90` lines 830-916

**Purpose:** Apply harmonization scaling and initialization method

```
INPUT: SWITCH_APPLY_SCALING, SCALE_YR, SCALE_WM2, SCALE_FACTOR, DAT_RF, CDNC, NORMED

1. DETERMINE SCALING FACTOR
   IF SWITCH=0 or SCALE_YR out of range:
       SCALE_FACTOR = 1.0 (no scaling)
   ELSEIF SWITCH=1:
       SCALE_FACTOR = SCALE_WM2 / NORMED

2. APPLY SCALING TO FORCING
   DAT_RF%DATBOX = CDNC * SCALE_FACTOR

3. CALCULATE GLOBAL FROM BOX VALUES
   DAT_RF%DATGLOBE = sum(DAT_RF%DATBOX * GLOBALAREAFRACTIONS)

4. APPLY INITIALIZATION METHOD
   IF RF_INITIALIZATION_METHOD = 'ZEROSTARTSHIFT':
       Store first year offset
       Subtract offset from all years
   ELSEIF RF_INITIALIZATION_METHOD = 'JUMPSTART':
       No offset applied

5. SET FIRST YEAR TO ZERO
   DAT_RF%DATGLOBE(1) = 0.0
   DAT_RF%DATBOX(1,:) = 0.0

6. APPLY MAXIMUM LIMIT (if enabled)
   IF CLOUD_APPLY_LIMIT_MAX = 1:
       DAT_RF%DATBOX = min(CLOUD_LIMIT_MAX, DAT_RF%DATBOX)
       Recalculate global
```

### 7.3 Sector Emission Handling

The module supports a complex "sector mode" where it runs twice:
1. First run with total emissions - saves normalization factors
2. Second run with sector-adjusted emissions - applies saved factors

This ensures consistent normalization when sectoral emissions modify the totals.

## 8. Numerical Considerations

### 8.1 Division by Zero Protection

Multiple checks prevent division by zero:
```fortran
IF (CLOUD % NORM_NO3 == 0.0D0) CLOUD % NORM_NO3 = 1.0D0
IF (CLOUD % NORM_OC == 0.0D0) CLOUD % NORM_OC = 1.0D0
IF (CLOUD % NORM_BC == 0.0D0) CLOUD % NORM_BC = 1.0D0
IF (CLOUD % NORM_SOX == 0.0D0) CLOUD % NORM_SOX = 1.0D0
IF (CLOUD % NORM_SS == 0.0D0) CLOUD % NORM_SS = 1.0D0
```

### 8.2 Zero Delta CDNC Handling

If DELTA_CDNC at normyear is zero, the entire regional forcing is set to zero:
```fortran
IF (CLOUD % DELTA_CDNC(RF_REGIONS_NORMYEARIDX, I) /= 0.0D0) THEN
    ! Normal calculation
ELSE
    CLOUD % DELTA_CDNC_ALBEDO(:, I) = 0.0D0
END IF
```

### 8.3 Future Year Boundary Handling

The code uses `MIN(IDX_FUTYR + 1, NYEARS)` to prevent array out-of-bounds when scaling future emissions near the end year.

### 8.4 Pointer vs Value Copy

The code explicitly adds `+ 0.0D0` when copying arrays to ensure deep copies rather than pointer references:
```fortran
CLOUD % NI_NO3(:, I) = DAT_NO3T_RF % DATBOX(:, I) + 0.0D0
```

### 8.5 Emission Averaging

Future scaling uses the average of current and next year emissions to provide smoother transitions:
```fortran
(DAT_*_EMIS(FUTYR:NYEARS-1, box) + DAT_*_EMIS(FUTYR+1:NYEARS, box)) / 2.0
```

## 9. Issues and Concerns

### 9.1 CRITICAL: Missing Attribution

**Line 73-74 of cloudstore.f90:**
```fortran
! THIS SUBROUTINE CACULATES THE INDIRECT AERSOSOL RADIATIVE FORCING EFFECT DUE TO
! CLOUD ALBEDO AND LIFETIME CHANGES - USING A SIMPLE FORMULA BY xxxxx AS USED AS WELL IN
```

The scientific source for the formula is marked as "xxxxx" - the attribution is MISSING. This is a significant documentation gap for a key climate process.

### 9.2 MAJOR: Cloud Cover Effect Disabled by Default

The `RF_CLOUD_COVER_AER_WM2 = 0.0` default means the second indirect effect (cloud lifetime) is COMPLETELY DISABLED. Users may not realize this. The code infrastructure exists but produces zero forcing.

### 9.3 MAJOR: Simplified Physics vs Comments

The code contains extensive commented-out Gultepe and Isaac parameterizations (lines 669-706) that would provide more physically meaningful CDNC calculations. Instead, a simple `log10(NA_TOT)` is used because "multiple norming factors applied subsequently... cause that none of the parameters... actually influence the final result."

This reveals that the forcing is essentially empirically scaled rather than physically derived.

### 9.4 MODERATE: Inconsistent Normalization Years

Each species uses a different `IDX_FUTYR` based on data availability:
- NO3: from NO3T_RF or NOXI_EMIS HISTLASTYEAR
- OC: MIN of OCI_OT, OCB_OT, OCN_OT HISTLASTYEAR
- BC: MIN of BCI_OT, BCB_OT HISTLASTYEAR
- SOX: MIN of SOXI_OT, SOXNB_OT HISTLASTYEAR
- SS: from SS_OT LASTYEAR

This could lead to inconsistent behavior if data timeseries have different end years.

### 9.5 MODERATE: Code Duplication

The pattern for processing each aerosol species is nearly identical but repeated 5 times with minor variations. This ~600 lines of code could be refactored to ~100 lines with a common function.

### 9.6 MINOR: Typos in Comments

- Line 73: "CACULATES" should be "CALCULATES"
- Line 669: "GUTEPPE" should be "GULTEPE"

### 9.7 MINOR: Incomplete Comment "Proceed Here"

Multiple occurrences of "proceed here" and "continue here" comments suggest incomplete documentation or development notes:
- Line 76: "PROCEED HERE AND CHECK TEXT"
- Line 115: "continue here, check that lastyear is set correctly"
- Line 148: "continue here: do something for the NOXB emissions"

### 9.8 DESIGN: Optical Thickness as CCN Proxy

The module uses aerosol optical thickness (extinction) as a proxy for particle number concentration. This is a significant simplification - optical thickness depends on particle size and mass, not just number. Different aerosol types with the same optical thickness could have very different CCN contributions.

### 9.9 DESIGN: No Size Distribution

Real CCN activation depends on aerosol size distribution and supersaturation. This module implicitly assumes a fixed relationship between mass/optical thickness and activatable particle number.

### 9.10 DESIGN: Linear Superposition

Species contributions are linearly summed, ignoring:
- Competition for water vapor during activation
- Chemical interactions between species
- Size-dependent activation differences

## 10. Test Cases

### 10.1 Zero Emissions Test

**Purpose:** Verify forcing is zero when all anthropogenic aerosols are zero
```python
# Set all aerosol emissions and optical thickness to pre-industrial
# Expected: DAT_CLOUD_TOT_RF = 0.0 for all years and regions
```

### 10.2 Harmonization Test

**Purpose:** Verify forcing matches target in harmonization year
```python
# Run with default AR6 settings
# Expected: DAT_CLOUD_ALBEDO_RF global mean = -0.89 W/m^2 in 2019
# Expected: DAT_CLOUD_COVER_RF global mean = 0.0 W/m^2 in 2019
```

### 10.3 Single Species Test

**Purpose:** Verify each species contributes according to weight
```python
# Set only SOX emissions, all others zero
# Expected: NA_TOT = CLOUD_WEIGHT_SOX * NI_SOXI (normalized)
```

### 10.4 Regional Pattern Test

**Purpose:** Verify regional patterns are applied correctly
```python
# Uniform global aerosol change
# Expected: Regional RF ratios match RF_REGIONS_CLOUD_ALBEDO ratios
```

### 10.5 BC Solubility Test

**Purpose:** Verify BCI reduction by solubility ratio
```python
# Set only BCI optical thickness, BCB = 0
# Expected: NI_BCI = OT_BCI * CLOUD_BCI2BCB_SOLUBLE_RATIO / NORM_BC
```

### 10.6 Future Scaling Test

**Purpose:** Verify emission-based scaling after IDX_FUTYR
```python
# Double SOX emissions after 2020
# Expected: NI_SOXI roughly doubles after 2020 relative to IDX_FUTYR value
```

### 10.7 Constant Natural Source Test

**Purpose:** Verify sea salt and natural OC remain constant
```python
# Run to 2100
# Expected: NI_SSNAT(2100) = NI_SSNAT(IDX_FUTYR)
# Expected: NI_OCN(2100) = NI_OCN(IDX_FUTYR)
```

### 10.8 Maximum Limit Test

**Purpose:** Verify CLOUD_LIMIT_MAX is applied
```python
# Set CLOUD_APPLY_LIMIT_MAX = 1, CLOUD_LIMIT_MAX = 0.0
# Scenario with decreasing aerosols (forcing becoming less negative)
# Expected: RF never exceeds 0.0 W/m^2
```

### 10.9 Initialization Method Test

**Purpose:** Verify ZEROSTARTSHIFT vs JUMPSTART behavior
```python
# ZEROSTARTSHIFT: First year forcing should be exactly 0.0
# JUMPSTART: First year forcing equals calculated value (no offset)
```

### 10.10 Sector Mode Test

**Purpose:** Verify two-pass sector emission handling
```python
# Run with SECTOR_INCLUDE != 'NOSECTOR'
# First run: Saves SECTOR_CLOUD_NORM_* values
# Second run: Uses saved values, not recalculated
# Expected: Consistent normalization between runs
```

## 11. Fortran Code References

### 11.1 Primary Source Files

| File | Lines | Description |
|------|-------|-------------|
| `src/libmagicc/physics/cloudstore.f90` | 1-916 | Main module, all calculations |
| `src/libmagicc/physics/radiative_forcing.f90` | 1-118 | Parameter declarations |
| `src/libmagicc/core.f90` | 44-85 | Sector handling variables (MOD_SECTOR) |
| `src/libmagicc/allcfgs.f90` | 190-239 | Configuration namelist |
| `src/libmagicc/utils/datastore.f90` | 200-268 | Data structure declarations |

### 11.2 Key Subroutines

| Subroutine | Location | Purpose |
|------------|----------|---------|
| `cloud_init` | cloudstore.f90:44-65 | Allocate arrays, initialize datastores |
| `cloud_calc_ind_aerosol` | cloudstore.f90:92-828 | Main calculation routine |
| `CLOUD_APPLY_SCALING` | cloudstore.f90:830-916 | Harmonization and initialization |

### 11.3 Integration Points

- Called from: `MAGICC7.f90` line 1746
- Outputs used in: `MAGICC7.f90` lines 5776-5917 (ERF/EffRF calculation)
- Surface forcing: `MAGICC7.f90` lines 6196-6209

### 11.4 Related Modules

| Module | Relationship |
|--------|-------------|
| `MOD_DATASTORE` | Optical thickness inputs, RF outputs |
| `MOD_RADIATIVE_FORCING` | Parameters, regional patterns |
| `MOD_SECTOR` | Two-pass sector emission handling |
| `mod_areas` | Area fractions (FGNO, FGNL, etc.) |
| `mod_years` | NYEARS, STARTYEAR, ENDYEAR |

## 12. Summary

This module implements aerosol indirect effects on clouds using a highly parameterized approach. Key characteristics:

**Strengths:**
- Tracks multiple aerosol species with different sources
- Supports AR6-style harmonization to observed/modeled forcing
- Regional forcing patterns from detailed models
- Handles historical data transition to emission-based projections

**Limitations:**
- Simplified log-linear CCN-CDNC relationship
- Cloud cover effect disabled by default
- Optical thickness used as CCN proxy (ignores size distribution)
- Linear species superposition (no interaction effects)
- Missing scientific attribution in code comments

**For Rewrite Considerations:**
1. Consider implementing a more physically-based CCN activation scheme
2. Enable cloud cover effect with appropriate default value
3. Add aerosol size distribution sensitivity
4. Refactor repetitive species processing code
5. Document the scientific basis properly (find the "xxxxx" reference)
6. Consider interaction effects between species
