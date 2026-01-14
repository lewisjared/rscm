# Module 07a: Well-Mixed GHG Forcing

## 1. Scientific Purpose

This module calculates the radiative forcing from the three main well-mixed greenhouse gases: carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O). Radiative forcing quantifies the change in Earth's energy balance due to changes in atmospheric composition, measured in watts per square meter (W/m2).

The module implements two alternative calculation methods:

1. **IPCCTAR Method** - Based on IPCC Third Assessment Report (2001) simplified formulas
2. **OLBL Method** - Based on Etminan et al. (2016) and Meinshausen parameterization of line-by-line radiative transfer calculations

Both methods account for the spectral overlap between CH4 and N2O absorption bands, which reduces their combined radiative forcing compared to treating them independently. The OLBL method additionally accounts for CO2-N2O overlap and includes "rapid adjustment" factors that convert instantaneous radiative forcing to effective radiative forcing.

**Key distinction:** This module calculates *radiative forcing* only. It does not simulate atmospheric chemistry or concentrations - those come from upstream modules (CH4 chemistry module 01, N2O chemistry module 02) or from prescribed concentration inputs.

## 2. Mathematical Formulation

### 2.1 IPCCTAR Method

The IPCC TAR method uses simplified analytical formulas derived from fitting to detailed radiative transfer calculations.

#### 2.1.1 CO2 Radiative Forcing

$$RF_{CO2} = \frac{\Delta Q_{2\times CO2}}{\ln(2)} \cdot \ln\left(\frac{C}{C_0}\right)$$

Where:
- $C$ = current CO2 concentration (ppm)
- $C_0$ = pre-industrial CO2 concentration (ppm)
- $\Delta Q_{2\times CO2}$ = radiative forcing for CO2 doubling (default: 3.71 W/m2)

**Effective alpha coefficient:**

$$\alpha_{CO2} = \frac{\Delta Q_{2\times CO2}}{\ln(2)} \approx 5.35 \text{ W/m}^2$$

#### 2.1.2 CH4 Radiative Forcing

$$RF_{CH4} = \beta_{CH4} \cdot \left(\sqrt{C_{CH4}} - \sqrt{C_{CH4,0}}\right) + 0.47 \cdot \ln\left(\frac{1 + f(C_{CH4,0}, C_{N2O,0})}{1 + f(C_{CH4}, C_{N2O,0})}\right)$$

Where:
- $C_{CH4}$ = current CH4 concentration (ppb)
- $C_{CH4,0}$ = pre-industrial CH4 concentration (ppb)
- $\beta_{CH4}$ = CH4 radiative efficiency (W/m2/ppb), converted from `CH4_RADEFF_WM2PERPPB`
- $f(M, N)$ = CH4-N2O overlap function (see below)

#### 2.1.3 N2O Radiative Forcing

$$RF_{N2O} = \beta_{N2O} \cdot \left(\sqrt{C_{N2O}} - \sqrt{C_{N2O,0}}\right) + 0.47 \cdot \ln\left(\frac{1 + f(C_{CH4,0}, C_{N2O,0})}{1 + f(C_{CH4,0}, C_{N2O})}\right)$$

Where:
- $C_{N2O}$ = current N2O concentration (ppb)
- $C_{N2O,0}$ = pre-industrial N2O concentration (ppb)
- $\beta_{N2O}$ = N2O radiative efficiency (W/m2/ppb), converted from `N2O_RADEFF_WM2PERPPB`

#### 2.1.4 CH4-N2O Overlap Function

The overlap function accounts for spectral band overlap, using concentrations in ppt (ppb/1000):

$$f(M, N) = 0.6356 \cdot (M \cdot N)^{0.75} + 0.007 \cdot M \cdot (M \cdot N)^{1.52}$$

Where $M$ and $N$ are CH4 and N2O concentrations in ppt respectively.

**Note:** The factor 0.47 in the overlap term comes from the TAR Table 6.2 formulation. The constants 0.6356 and 0.007 are derived from TAR coefficients after unit conversion (see code comment lines 1120-1122 in deltaq_calculations.f90).

### 2.2 OLBL Method

The OLBL (Optimized Line-By-Line) method uses polynomial formulas that more accurately capture the concentration dependence and inter-species overlaps.

#### 2.2.1 CO2 Radiative Forcing

$$RF_{CO2} = f_{adj,CO2} \cdot \alpha_{CO2} \cdot \ln\left(\frac{C}{C_0}\right)$$

Where $\alpha_{CO2}$ is a concentration-dependent coefficient:

**For $C \leq C_0$:**
$$\alpha_{CO2} = d_1 + c_1 \cdot \sqrt{N}$$

**For $C_0 < C < C_{max}$:**
$$\alpha_{CO2} = a_1 \cdot (C - C_0)^2 + b_1 \cdot (C - C_0) + d_1 + c_1 \cdot \sqrt{N}$$

**For $C \geq C_{max}$:**
$$\alpha_{CO2} = -\frac{b_1^2}{4 a_1} + d_1 + c_1 \cdot \sqrt{N}$$

Where:
- $C_{max} = \frac{2 a_1 C_0 - b_1}{2 a_1}$ (concentration at which alpha reaches maximum)
- $N$ = N2O concentration (ppb)
- $a_1, b_1, c_1, d_1$ = OLBL coefficients for CO2
- $f_{adj,CO2}$ = rapid adjustment factor (default: 1.05)

#### 2.2.2 CH4 Radiative Forcing

$$RF_{CH4} = f_{adj,CH4} \cdot \alpha_{CH4} \cdot \left(\sqrt{C_{CH4}} - \sqrt{C_{CH4,0}}\right)$$

Where:
$$\alpha_{CH4} = a_3 \cdot \sqrt{C_{CH4}} + b_3 \cdot \sqrt{C_{N2O}} + d_3$$

- $a_3, b_3, d_3$ = OLBL coefficients for CH4
- $f_{adj,CH4}$ = rapid adjustment factor (default: 0.86)

#### 2.2.3 N2O Radiative Forcing

$$RF_{N2O} = f_{adj,N2O} \cdot \alpha_{N2O} \cdot \left(\sqrt{C_{N2O}} - \sqrt{C_{N2O,0}}\right)$$

Where:
$$\alpha_{N2O} = a_2 \cdot \sqrt{C_{CO2}} + b_2 \cdot \sqrt{C_{N2O}} + c_2 \cdot \sqrt{C_{CH4}} + d_2$$

- $a_2, b_2, c_2, d_2$ = OLBL coefficients for N2O
- $f_{adj,N2O}$ = rapid adjustment factor (default: 1.0)

### 2.3 Stratospheric H2O from CH4 Oxidation

Both methods calculate an additional forcing from stratospheric water vapor produced by CH4 oxidation:

$$RF_{strH2O} = RF_{CH4,pure} \cdot f_{strH2O}$$

Where:
- $RF_{CH4,pure}$ = CH4 forcing calculated *without* the overlap correction (pure CH4 effect)
- $f_{strH2O}$ = `CH4_ADDEDSTRATH2O_PERCENT` (default: 0.0923, i.e., ~9.2%)

**Note:** The "pure" methane forcing is calculated separately using the same formula but with the overlap terms set to zero.

## 3. State Variables

This module has **no internal state variables**. It performs a pure calculation at each timestep based on current concentrations.

All forcing values are stored in datastore variables for use by downstream modules (climate module) and output.

## 4. Parameters

### 4.1 Method Selection

| Parameter | Fortran Name | Units | Default | Valid Values | Description |
|-----------|--------------|-------|---------|--------------|-------------|
| RF calculation method | `CORE_CO2CH4N2O_RFMETHOD` | string | "OLBL" | "IPCCTAR", "OLBL" | Selects which forcing calculation method to use |

### 4.2 IPCCTAR Method Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| CO2 2x forcing | `CORE_DELQ2XCO2` | W/m2 | 3.71 | 3.5-4.0 | Radiative forcing for CO2 doubling |
| CH4 radiative efficiency | `CH4_RADEFF_WM2PERPPB` | W/m2/ppb | 0.036 | 0.03-0.04 | CH4 radiative efficiency (10% uncertainty) |
| N2O radiative efficiency | `N2O_RADEFF_WM2PERPPB` | W/m2/ppb | 0.12 | 0.11-0.13 | N2O radiative efficiency (5% uncertainty) |

### 4.3 OLBL Method Parameters

#### 4.3.1 CO2 Coefficients

| Parameter | Fortran Name | Units | Default | Description |
|-----------|--------------|-------|---------|-------------|
| CO2 quadratic | `CORE_OLBL_CO2_A1` | W/m2/ppm2 | -2.4785e-07 | Quadratic term for CO2 alpha |
| CO2 linear | `CORE_OLBL_CO2_B1` | W/m2/ppm | 0.00075906 | Linear term for CO2 alpha |
| CO2-N2O overlap | `CORE_OLBL_CO2_C1` | W/m2/ppb^0.5 | -0.0021492 | N2O overlap term (multiplies sqrt(N2O)) |
| CO2 constant | `CORE_OLBL_CO2_D1` | W/m2 | 5.2 | Constant term (comparable to DELQ2XCO2/ln(2)) |

#### 4.3.2 CH4 Coefficients

| Parameter | Fortran Name | Units | Default | Description |
|-----------|--------------|-------|---------|-------------|
| CH4-CH4 term | `CORE_OLBL_CH4_A3` | W/m2/ppb^0.5 | -8.9603e-05 | Multiplier on sqrt(CH4) in alpha |
| CH4-N2O overlap | `CORE_OLBL_CH4_B3` | W/m2/ppb^0.5 | -0.00012462 | N2O overlap (multiplies sqrt(N2O)) |
| CH4 constant | `CORE_OLBL_CH4_D3` | W/m2 | 0.045 | Constant in alpha term |

#### 4.3.3 N2O Coefficients

| Parameter | Fortran Name | Units | Default | Description |
|-----------|--------------|-------|---------|-------------|
| N2O-CO2 overlap | `CORE_OLBL_N2O_A2` | W/m2/ppm^0.5 | -0.00034197 | CO2 overlap (multiplies sqrt(CO2)) |
| N2O-N2O term | `CORE_OLBL_N2O_B2` | W/m2/ppb^0.5 | 0.00025455 | Multiplier on sqrt(N2O) |
| N2O-CH4 overlap | `CORE_OLBL_N2O_C2` | W/m2/ppb^0.5 | -0.00024357 | CH4 overlap (multiplies sqrt(CH4)) |
| N2O constant | `CORE_OLBL_N2O_D2` | W/m2 | 0.14 | Constant in alpha term |

### 4.4 Rapid Adjustment Factors

These factors convert instantaneous radiative forcing to effective radiative forcing by accounting for rapid atmospheric adjustments (e.g., stratospheric temperature adjustment, cloud responses).

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| CO2 adjustment | `CORE_RFRAPIDADJUST_CO2` | dimensionless | 1.05 | 0.9-1.2 | ERF/RF ratio for CO2 |
| CH4 adjustment | `CORE_RFRAPIDADJUST_CH4` | dimensionless | 0.86 | 0.7-1.0 | ERF/RF ratio for CH4 |
| N2O adjustment | `CORE_RFRAPIDADJUST_N2O` | dimensionless | 1.0 | 0.9-1.1 | ERF/RF ratio for N2O |

**Source:** Smith et al. 2018, IPCC AR6 WG1 Chapter 7 (First Order Draft reference in comments)

### 4.5 Other Related Parameters

| Parameter | Fortran Name | Units | Default | Description |
|-----------|--------------|-------|---------|-------------|
| Pre-industrial CO2 | `CO2_PREINDCO2CONC` | ppm | 278.0 | Reference CO2 for forcing calculation |
| Apply PI override | `CO2_PREINDCO2CONC_APPLY` | flag | 0 | If 1, override file-based PI value |
| Strat H2O fraction | `CH4_ADDEDSTRATH2O_PERCENT` | fraction | 0.0923 | Fraction of CH4 forcing added as strat H2O |

### 4.6 Regional Forcing Distribution

The module also uses regional fraction parameters to distribute global forcing across the 4 MAGICC boxes (Northern Ocean, Northern Land, Southern Ocean, Southern Land):

| Parameter | Fortran Name | Default | Description |
|-----------|--------------|---------|-------------|
| CO2 regions | `RF_REGIONS_CO2` | [1,1,1,1] | Regional weighting for CO2 forcing |
| CH4 regions | `RF_REGIONS_CH4` | [1,1,1,1] | Regional weighting for CH4 forcing |
| N2O regions | `RF_REGIONS_N2O` | [1,1,1,1] | Regional weighting for N2O forcing |
| Strat H2O regions | `RF_REGIONS_CH4OXSTRATH2O` | [1,1,1,1] | Regional weighting for strat H2O |

## 5. Inputs (per timestep)

| Variable | Units | Source Module | Required? | Fortran Variable |
|----------|-------|---------------|-----------|------------------|
| CO2 concentration | ppm | Prescribed or carbon cycle | Yes | `DAT_CO2_CONC%DATGLOBE(NEXT_YEAR_IDX)` |
| CH4 concentration | ppb | Module 01 (CH4 chemistry) | Yes | `DAT_CH4_CONC%DATGLOBE(NEXT_YEAR_IDX)` |
| N2O concentration | ppb | Module 02 (N2O chemistry) | Yes | `DAT_N2O_CONC%DATGLOBE(NEXT_YEAR_IDX)` |
| Pre-industrial CO2 | ppm | Configuration/input file | Yes | `DAT_CO2_CONC%PREIND_DATGLOBE` |
| Pre-industrial CH4 | ppb | Input file | Yes | `DAT_CH4_CONC%PREIND_DATGLOBE` |
| Pre-industrial N2O | ppb | Input file | Yes | `DAT_N2O_CONC%PREIND_DATGLOBE` |

**Note:** Concentrations are read for the "next year" index because forcing is calculated for the end of the current timestep.

## 6. Outputs (per timestep)

| Variable | Units | Destination Module(s) | Fortran Variable |
|----------|-------|----------------------|------------------|
| CO2 radiative forcing | W/m2 | Climate module, output | `DAT_CO2_RF%DATGLOBE(NEXT_YEAR_IDX)` |
| CH4 radiative forcing | W/m2 | Climate module, output | `DAT_CH4_RF%DATGLOBE(NEXT_YEAR_IDX)` |
| N2O radiative forcing | W/m2 | Climate module, output | `DAT_N2O_RF%DATGLOBE(NEXT_YEAR_IDX)` |
| Strat H2O forcing | W/m2 | Climate module, output | `DAT_CH4OXSTRATH2O_RF%DATGLOBE(NEXT_YEAR_IDX)` |
| Box-level forcings | W/m2 | Climate module | `DAT_*_RF%DATBOX(NEXT_YEAR_IDX,:)` |

Each forcing output is also calculated with first-year offsets applied (for `RF_INITIALIZATION_METHOD = "ZEROSTARTSHIFT"`) and stored with regional box breakdowns.

## 7. Algorithm (Pseudocode)

### 7.1 Main Flow

```
SUBROUTINE CALCULATE_GHG_FORCING():
    # 1. Check for PI CO2 override
    IF current_year_idx == 1 AND CO2_PREINDCO2CONC_APPLY == 1:
        DAT_CO2_CONC%PREIND_DATGLOBE = CO2_PREINDCO2CONC
        DAT_CO2_CONC%PREIND_DATBOX = CO2_PREINDCO2CONC

    # 2. Get concentrations
    co2_ppm = DAT_CO2_CONC%DATGLOBE(NEXT_YEAR_IDX)
    ch4_ppb = DAT_CH4_CONC%DATGLOBE(NEXT_YEAR_IDX)
    n2o_ppb = DAT_N2O_CONC%DATGLOBE(NEXT_YEAR_IDX)

    co2_ppm_pi = DAT_CO2_CONC%PREIND_DATGLOBE
    ch4_ppb_pi = DAT_CH4_CONC%PREIND_DATGLOBE
    n2o_ppb_pi = DAT_N2O_CONC%PREIND_DATGLOBE

    # 3. Calculate forcing based on method
    IF CORE_CO2CH4N2O_RFMETHOD == "IPCCTAR":
        co2_ch4_n2o_rf = calculate_co2_ch4_n2o_rf_ipcc_tar(...)
        pure_methane_forcing = calcluate_ch4_or_n2o_ipcc_tar_rf(..., overlap=0, pi_overlap=0)

    ELSEIF CORE_CO2CH4N2O_RFMETHOD == "OLBL":
        co2_ch4_n2o_rf = calculate_co2_ch4_n2o_rf_olbl(...)
        pure_methane_forcing = calculate_ch4_rf_olbl(..., n2o_ppb=n2o_ppb_pi)  # PI N2O for pure

    ELSE:
        FATAL_ERROR("Unknown method")

    # 4. Calculate first year offset (for zero-start shifting)
    IF current_year_idx == 1:
        # Calculate forcing at first year concentrations
        co2_ch4_n2o_rf_offset = [same calculation with CURRENT_YEAR_IDX concentrations]
        pure_methane_forcing_first_year = [same pure calculation]

    # 5. Store CO2 forcing
    CALL set_rf_global_and_boxes_with_fraction_split(
        DAT_CO2_RF, NEXT_YEAR_IDX, co2_ch4_n2o_rf(1),
        normalise_with_weights(RF_REGIONS_CO2, GLOBALAREAFRACTIONS)
    )

    # 6. Store CH4 forcing
    CALL set_rf_global_and_boxes_with_fraction_split(
        DAT_CH4_RF, NEXT_YEAR_IDX, co2_ch4_n2o_rf(2),
        normalise_with_weights(RF_REGIONS_CH4, GLOBALAREAFRACTIONS)
    )

    # 7. Store N2O forcing
    CALL set_rf_global_and_boxes_with_fraction_split(
        DAT_N2O_RF, NEXT_YEAR_IDX, co2_ch4_n2o_rf(3),
        normalise_with_weights(RF_REGIONS_N2O, GLOBALAREAFRACTIONS)
    )

    # 8. First year initialization
    IF current_year_idx == 1:
        CALL set_rf_first_year_forcing_and_offset_from_global(
            DAT_CO2_RF, co2_ch4_n2o_rf_offset(1), ...
        )
        # Similar for CH4, N2O

    # 9. Calculate stratospheric H2O from CH4 oxidation
    ch4oxstrath2o_rf = pure_methane_forcing * CH4_ADDEDSTRATH2O_PERCENT
    CALL set_rf_global_and_boxes_with_fraction_split(
        DAT_CH4OXSTRATH2O_RF, NEXT_YEAR_IDX, ch4oxstrath2o_rf, ...
    )
```

### 7.2 IPCCTAR Method Implementation

```
FUNCTION calculate_co2_ch4_n2o_rf_ipcc_tar(
    co2_ppm, ch4_ppb, n2o_ppb, rf_2xco2,
    co2_ppm_pi, ch4_radeff, n2o_radeff, ch4_ppb_pi, n2o_ppb_pi
) RESULT(rf_array):

    # CO2 forcing (logarithmic)
    co2_rf = rf_2xco2 / ln(2.0) * ln(co2_ppm / co2_ppm_pi)

    # Calculate overlap terms (concentrations in ppt, hence /1000)
    ch4_ppt = ch4_ppb / 1000
    ch4_ppt_pi = ch4_ppb_pi / 1000
    n2o_ppt = n2o_ppb / 1000
    n2o_ppt_pi = n2o_ppb_pi / 1000

    overlap_ch4_n2opi = calculate_overlap(ch4_ppt, n2o_ppt_pi)
    overlap_ch4pi_n2opi = calculate_overlap(ch4_ppt_pi, n2o_ppt_pi)
    overlap_ch4pi_n2o = calculate_overlap(ch4_ppt_pi, n2o_ppt)

    # CH4 forcing with overlap correction
    ch4_rf = ch4_radeff * (sqrt(ch4_ppb) - sqrt(ch4_ppb_pi))
           + 0.47 * ln((1 + overlap_ch4pi_n2opi) / (1 + overlap_ch4_n2opi))

    # N2O forcing with overlap correction
    n2o_rf = n2o_radeff * (sqrt(n2o_ppb) - sqrt(n2o_ppb_pi))
           + 0.47 * ln((1 + overlap_ch4pi_n2opi) / (1 + overlap_ch4pi_n2o))

    RETURN [co2_rf, ch4_rf, n2o_rf]

FUNCTION calculate_overlap(ch4_ppt, n2o_ppt):
    RETURN 0.6356 * (ch4_ppt * n2o_ppt)^0.75
         + 0.007 * ch4_ppt * (ch4_ppt * n2o_ppt)^1.52
```

### 7.3 OLBL Method Implementation

```
FUNCTION calculate_co2_ch4_n2o_rf_olbl(...) RESULT(rf_array):

    # CO2 forcing
    co2_rf = calculate_co2_rf_olbl(co2_ppm, co2_ppm_pi, a1, b1, c1, d1,
                                   rf_radpiadjust_co2, n2o_ppb)

    # CH4 forcing
    ch4_rf = calculate_ch4_rf_olbl(ch4_ppb, ch4_ppb_pi, a3, b3, d3,
                                   rf_rapidadjust_ch4, n2o_ppb)

    # N2O forcing
    n2o_rf = calculate_n2o_rf_olbl(n2o_ppb, n2o_ppb_pi, a2, b2, c2, d2,
                                   rf_rapidadjust_n2o, co2_ppm, ch4_ppb)

    RETURN [co2_rf, ch4_rf, n2o_rf]

FUNCTION calculate_co2_alpha_olbl(co2_ppm, co2_ppm_pi, a1, b1, c1, d1, n2o_ppb):
    # Maximum alpha concentration
    alphamax_co2_conc = (2 * a1 * co2_ppm_pi - b1) / (2 * a1)

    # N2O overlap contribution
    alpha_overlap_n2o = c1 * sqrt(n2o_ppb)

    # CO2 contribution (three regimes)
    IF co2_ppm >= alphamax_co2_conc:
        alpha_wo_n2o = -b1^2 / (4 * a1) + d1
    ELSEIF co2_ppm <= co2_ppm_pi:
        alpha_wo_n2o = d1
    ELSE:
        alpha_wo_n2o = a1 * (co2_ppm - co2_ppm_pi)^2
                     + b1 * (co2_ppm - co2_ppm_pi) + d1

    RETURN alpha_wo_n2o + alpha_overlap_n2o

FUNCTION calculate_co2_rf_olbl(...):
    alpha = calculate_co2_alpha_olbl(...)
    RETURN rf_radpiadjust_co2 * alpha * ln(co2_ppm / co2_ppm_pi)

FUNCTION calculate_ch4_rf_olbl(ch4_ppb, ch4_ppb_pi, a3, b3, d3,
                               rf_rapidadjust, n2o_ppb):
    coeff = a3 * sqrt(ch4_ppb) + b3 * sqrt(n2o_ppb) + d3
    RETURN coeff * rf_rapidadjust * (sqrt(ch4_ppb) - sqrt(ch4_ppb_pi))

FUNCTION calculate_n2o_rf_olbl(n2o_ppb, n2o_ppb_pi, a2, b2, c2, d2,
                               rf_rapidadjust, co2_ppm, ch4_ppb):
    coeff = a2 * sqrt(co2_ppm) + b2 * sqrt(n2o_ppb) + c2 * sqrt(ch4_ppb) + d2
    RETURN coeff * rf_rapidadjust * (sqrt(n2o_ppb) - sqrt(n2o_ppb_pi))
```

## 8. Numerical Considerations

### 8.1 Logarithm of Zero/Negative

The CO2 forcing uses `ln(C/C0)`. If `C <= 0` or `C0 <= 0`, this produces undefined behavior. The code does **not** guard against this.

**Recommendation:** Add checks for `C > 0` and `C0 > 0` before calculating.

### 8.2 Square Root of Negative

CH4 and N2O formulas use `sqrt(C)`. Negative concentrations would cause NaN. No explicit guard exists.

### 8.3 Division in Alpha Calculation

The OLBL alpha calculation includes:
- `alphamax_co2_conc = (2*a1*co2_ppm_pi - b1) / (2*a1)`
- `alpha_wo_n2o = -b1^2 / (4*a1) + d1`

If `a1 = 0`, these divisions would fail. With the default `a1 = -2.4785e-07`, this is not an issue, but parameterization freedom could lead to problems.

### 8.4 Very High Concentrations

For very high CO2 concentrations, the OLBL method caps the alpha at its maximum value (when `C >= alphamax_co2_conc`). This provides implicit saturation behavior and prevents runaway forcing at extreme concentrations.

The threshold concentration is calculated dynamically:
```
alphamax_co2_conc = (2*a1*co2_ppm_pi - b1) / (2*a1)
```
With defaults: `(2*(-2.4785e-07)*278 - 0.00075906) / (2*(-2.4785e-07))` = ~1810 ppm

### 8.5 Precision

The code uses `REAL(8)` (double precision) throughout and `DLOG` for natural logarithm, which should maintain adequate precision for typical climate scenarios.

## 9. Issues and Concerns

### 9.1 Method Equivalence

**Fundamental question:** Do the IPCCTAR and OLBL methods produce equivalent results?

**Answer: No, they are fundamentally different.**

Key differences:
1. **CO2:** IPCCTAR uses a fixed alpha (5.35), OLBL uses concentration-dependent alpha
2. **Rapid adjustments:** Only OLBL includes rapid adjustment factors
3. **Overlap treatment:** Different mathematical forms for the overlap corrections
4. **CO2-N2O overlap:** Only OLBL includes this (IPCCTAR only has CH4-N2O overlap)

**Comparison at pre-industrial + 280 ppm CO2 = 556 ppm (2x):**
- IPCCTAR: RF = 3.71 W/m2 (by definition via CORE_DELQ2XCO2)
- OLBL: RF = 1.05 * alpha * ln(2), where alpha depends on N2O

These methods will give different results for identical inputs.

### 9.2 Effective RF Parameter Disconnect

There is a potential inconsistency in how `CORE_DELQ2XCO2` is used:

1. In IPCCTAR mode, it directly sets the CO2 doubling forcing
2. In OLBL mode, it is **not used** for the primary calculation (OLBL coefficients are used instead)
3. However, there's a derived variable `CORE_DELQ2XCO2_EFF` that may be set differently

Comment in MAGICC7.f90 (line 5298):
```
!   CORE_DELQ2XCO2_EFF SHOULD BE UPDATED WHEN THE OLBL APPROACH IS USED.
```

This suggests the model should dynamically calculate what effective DELQ2XCO2 the OLBL method implies, but this may not be fully implemented.

### 9.3 Hardcoded Magic Numbers

| Value | Location | Concern |
|-------|----------|---------|
| 0.47 | deltaq_calculations.f90:1103 | Hardcoded coefficient in IPCCTAR overlap term |
| 0.6356 | deltaq_calculations.f90:1143 | Derived from TAR but hardcoded |
| 0.007 | deltaq_calculations.f90:1144 | Derived from TAR but hardcoded |
| 1000.0 | deltaq_calculations.f90:1057-1060 | ppb to ppt conversion hardcoded |

### 9.4 Typo in Function Name

The function `calcluate_ch4_or_n2o_ipcc_tar_rf` (line 1089) has a typo: "calcluate" instead of "calculate". This should be fixed for maintainability.

### 9.5 Rapid Adjustment Factor Asymmetry

The rapid adjustment factors have inconsistent sign implications:
- CO2: 1.05 (ERF > RF) - positive adjustment implies tropospheric warming increases forcing
- CH4: 0.86 (ERF < RF) - implies rapid responses reduce net forcing
- N2O: 1.0 (ERF = RF) - no rapid adjustment

The CH4 value of 0.86 is notably different and comes from the fact that CH4 forcing induces rapid cloud adjustments that partially offset the forcing.

### 9.6 First Year Initialization Complexity

The first-year offset calculation is done separately from the main calculation, requiring duplication of function calls. This is error-prone and could lead to inconsistencies if the two code paths diverge.

### 9.7 Pure Methane Forcing Calculation Inconsistency

For stratospheric H2O calculation, the "pure" methane forcing (without N2O overlap) is calculated differently between the two methods:

**IPCCTAR:** Explicitly sets `overlap=0.0, pi_overlap=0.0` parameters
**OLBL:** Uses `n2o_ppb = n2o_ppb_pi` (pre-industrial N2O) instead of current N2O

These are conceptually similar (both remove the N2O overlap effect) but mathematically different due to the different overlap formulations.

### 9.8 Missing C3 Coefficient

The OLBL CH4 formulation uses A3, B3, D3 but **no C3 coefficient**. This suggests either:
1. The C3 term was determined to be negligible
2. The implementation is incomplete
3. The original Etminan formulation doesn't have a CO2-CH4 overlap term for CH4 forcing

The N2O formulation does have all four coefficients (A2, B2, C2, D2).

## 10. Test Cases

### 10.1 Unit Test: CO2 Doubling (IPCCTAR)

**Purpose:** Verify CO2 doubling produces expected forcing.

**Setup:**
```
CORE_CO2CH4N2O_RFMETHOD = "IPCCTAR"
CORE_DELQ2XCO2 = 3.71
co2_ppm = 556.0 (2 x 278)
co2_ppm_pi = 278.0
ch4_ppb = ch4_ppb_pi = 700.0 (to eliminate CH4 contribution)
n2o_ppb = n2o_ppb_pi = 270.0 (to eliminate N2O contribution)
```

**Expected Output:**
- `co2_rf = 3.71 W/m2` (exactly)
- `ch4_rf = 0.0 W/m2`
- `n2o_rf = 0.0 W/m2`

### 10.2 Unit Test: CO2 Doubling (OLBL)

**Purpose:** Verify OLBL produces reasonable CO2 forcing.

**Setup:**
```
CORE_CO2CH4N2O_RFMETHOD = "OLBL"
co2_ppm = 556.0
co2_ppm_pi = 278.0
n2o_ppb = 270.0 (for overlap calculation)
Default OLBL coefficients
```

**Expected Output:**
- `alpha = d1 + c1*sqrt(n2o) = 5.2 + (-0.0021492)*sqrt(270) = 5.165`
- `co2_rf = 1.05 * 5.165 * ln(2) = 3.76 W/m2`

### 10.3 Unit Test: CH4-N2O Overlap (IPCCTAR)

**Purpose:** Verify overlap reduces combined forcing.

**Setup:**
```
CORE_CO2CH4N2O_RFMETHOD = "IPCCTAR"
ch4_ppb = 1800.0
ch4_ppb_pi = 700.0
n2o_ppb = 330.0
n2o_ppb_pi = 270.0
CH4_RADEFF_WM2PERPPB = 0.036
N2O_RADEFF_WM2PERPPB = 0.12
```

**Calculations:**
1. Calculate overlap terms:
   - `f(1800/1000, 270/1000) = 0.6356*(1.8*0.27)^0.75 + 0.007*1.8*(1.8*0.27)^1.52`
   - `f(700/1000, 270/1000) = ...`
   - etc.

2. CH4 forcing without overlap: `0.036 * (sqrt(1800) - sqrt(700)) = 0.58 W/m2`
3. N2O forcing without overlap: `0.12 * (sqrt(330) - sqrt(270)) = 0.21 W/m2`
4. Combined forcing should be LESS than sum due to overlap

**Expected Output:**
- Total CH4+N2O forcing < 0.79 W/m2 (the sum without overlap)

### 10.4 Unit Test: Pre-industrial Concentrations

**Purpose:** Verify zero forcing at pre-industrial.

**Setup:**
```
co2_ppm = co2_ppm_pi = 278.0
ch4_ppb = ch4_ppb_pi = 700.0
n2o_ppb = n2o_ppb_pi = 270.0
```

**Expected Output (both methods):**
- `co2_rf = 0.0 W/m2`
- `ch4_rf = 0.0 W/m2`
- `n2o_rf = 0.0 W/m2`

### 10.5 Unit Test: Stratospheric H2O

**Purpose:** Verify strat H2O forcing calculation.

**Setup:**
```
pure_methane_forcing = 0.5 W/m2
CH4_ADDEDSTRATH2O_PERCENT = 0.0923
```

**Expected Output:**
- `ch4oxstrath2o_rf = 0.5 * 0.0923 = 0.04615 W/m2`

### 10.6 Integration Test: Historical Forcing

**Purpose:** Verify module reproduces AR5/AR6 reported forcing values.

**Setup:** Run from 1750-2020 with observed concentrations.

**Validation (approximate 2019 values from IPCC AR6):**
- CO2 forcing: ~2.1 W/m2
- CH4 forcing: ~0.5 W/m2
- N2O forcing: ~0.2 W/m2
- Strat H2O: ~0.05 W/m2

### 10.7 Comparison Test: IPCCTAR vs OLBL

**Purpose:** Quantify method differences.

**Setup:** Run identical scenario with both methods.

**Expected:** Document the difference in total GHG forcing between methods. Typical difference should be < 10% for modern concentrations.

### 10.8 Edge Case: Very High CO2

**Purpose:** Test OLBL alpha saturation.

**Setup:**
```
co2_ppm = 2000.0 (above alphamax threshold ~1810 ppm)
```

**Expected:** Alpha should be capped at maximum value, forcing should still increase logarithmically but with reduced slope.

## 11. Fortran Code References

### 11.1 Key File Locations

| Function/Section | File | Line Numbers |
|------------------|------|--------------|
| Module public interface | deltaq_calculations.f90 | 34-37 |
| IPCCTAR main function | deltaq_calculations.f90 | 1031-1087 |
| IPCCTAR CH4/N2O helper | deltaq_calculations.f90 | 1089-1106 |
| IPCCTAR overlap terms | deltaq_calculations.f90 | 1108-1147 |
| OLBL main function | deltaq_calculations.f90 | 1149-1222 |
| OLBL CO2 RF | deltaq_calculations.f90 | 1224-1243 |
| OLBL CH4 RF | deltaq_calculations.f90 | 1245-1259 |
| OLBL N2O RF | deltaq_calculations.f90 | 1261-1276 |
| OLBL CO2 alpha | deltaq_calculations.f90 | 1278-1318 |
| Strat H2O calculation | deltaq_calculations.f90 | 1320-1335 |
| Calling site (DELTAQ) | MAGICC7.f90 | 5139-5440 |
| Method selection | MAGICC7.f90 | 5157, 5203 |
| CO2 PI override | MAGICC7.f90 | 5143-5154 |
| Output storage | MAGICC7.f90 | 5322-5408 |
| Parameter declarations | climate_and_ocean.f90 | 16-39 |
| Parameter namelist | allcfgs.f90 | 97-109 |
| Default values | MAGCFG_DEFAULTALL.CFG | 86-124, 308 |

### 11.2 Key Equations by Line

| Equation | File:Line |
|----------|-----------|
| CO2 RF (IPCCTAR): `rf_2xco2/ln(2) * ln(C/C0)` | deltaq_calculations.f90:1053 |
| CH4-N2O overlap formula | deltaq_calculations.f90:1142-1146 |
| CH4/N2O RF formula (IPCCTAR) | deltaq_calculations.f90:1101-1104 |
| CO2 alpha (OLBL): three-regime calculation | deltaq_calculations.f90:1295-1316 |
| CO2 RF (OLBL): `adj * alpha * ln(C/C0)` | deltaq_calculations.f90:1239-1241 |
| CH4 coefficient (OLBL) | deltaq_calculations.f90:1255 |
| CH4 RF (OLBL) | deltaq_calculations.f90:1257 |
| N2O coefficient (OLBL) | deltaq_calculations.f90:1272 |
| N2O RF (OLBL) | deltaq_calculations.f90:1274 |
| Strat H2O RF | deltaq_calculations.f90:1331-1333 |

---

## Summary

The Well-Mixed GHG Forcing module implements two distinct methods for calculating radiative forcing from CO2, CH4, and N2O:

1. **IPCCTAR** - Simple, transparent formulas from TAR (2001) with fixed coefficients
2. **OLBL** - More sophisticated polynomial formulas based on Etminan et al. (2016) with concentration-dependent coefficients and rapid adjustment factors

**Key strengths:**
- Clean separation of calculation functions
- Support for both legacy (IPCCTAR) and modern (OLBL) approaches
- Includes inter-species overlap corrections
- Handles stratospheric H2O from CH4 oxidation

**Key concerns:**
- Methods give different results by design; users should understand which they are using
- Some hardcoded magic numbers that should ideally be configurable
- Typo in function name (`calcluate` -> `calculate`)
- No numerical guards against zero/negative concentrations
- First-year initialization code is duplicative and error-prone

For reimplementation, the mathematical formulations are well-defined and the code is reasonably readable, but care should be taken to:
1. Choose the appropriate method for the application
2. Document which coefficients are used and their sources
3. Add numerical safeguards for edge cases
4. Ensure consistency between the two methods' interpretation of overlap effects
