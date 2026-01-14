# Module 02: N2O Chemistry

## 1. Scientific Purpose

Nitrous oxide (N2O) is a potent greenhouse gas with an atmospheric lifetime of approximately 139 years. This module simulates the atmospheric concentration dynamics of N2O by solving a mass balance equation that accounts for emissions from anthropogenic and natural sources, and stratospheric destruction primarily through photolysis and reaction with O(1D) atoms.

The module implements a concentration-dependent lifetime feedback, recognizing that as N2O concentrations increase, the stratospheric sink becomes relatively less efficient (a negative feedback on concentration growth). This is parameterized using an empirical power-law relationship. The module also accounts for the time delay required for tropospheric N2O to mix into the stratosphere where destruction occurs, and can optionally include sensitivity to changes in the Brewer-Dobson circulation (meridional flux) under climate change.

## 2. Mathematical Formulation

### 2.1 Core Mass Balance Equation

The atmospheric N2O burden evolves according to:

$$\frac{dB}{dt} = E - \frac{\bar{B}_{lagged}}{\tau}$$

Where:

- $B$ = atmospheric N2O burden (TgN)
- $E$ = total emissions (TgN/yr)
- $\bar{B}_{lagged}$ = lagged average burden accounting for stratospheric mixing delay (NOT previous timestep - see Section 2.4)
- $\tau$ = effective atmospheric lifetime (years)

### 2.2 Concentration-Dependent Lifetime

The lifetime depends on the current atmospheric burden:

$$\tau = \tau_{scale} \cdot \tau_{init} \cdot \left( \max\left(1, \frac{\bar{B}}{B_{00}}\right) \right)^S$$

Where:

- $\tau_{init}$ = initial/reference lifetime (default: 139.275 years)
- $\tau_{scale}$ = scaling factor from meridional flux changes (dimensionless, typically 1.0)
- $\bar{B}$ = mid-year burden estimate
- $B_{00}$ = reference burden at N2O_FEED_YRSTART (feedback reference year)
- $S$ = feedback exponent (default: -0.04, negative = lifetime increases with burden)

### 2.3 Meridional Flux Scaling (Optional)

When `N2O_USE_TAUSTRAT_VAR == 1`:

$$\tau_{scale} = \frac{1}{1 + \Delta T_{merid} \cdot k_{merid} \cdot \alpha_{N2O}}$$

Where:

- $\Delta T_{merid}$ = temperature change above reference (K)
- $k_{merid}$ = meridional flux change per degree (`GEN_MERIDFLUX_CHNGPERDEG`)
- $\alpha_{N2O}$ = N2O sensitivity to meridional flux (`N2O_TAUSTRAT_SENS2MERIDFLUX`, default: 0.04)

### 2.4 Stratospheric Mixing Delay

The sink term uses lagged concentrations to account for troposphere-to-stratosphere transport:

$$\bar{B}_{prev} = \frac{C_{t-delay} + C_{t-delay-1}}{2} \cdot f_{ppb2TgN}$$

Where `delay` = `N2O_STRATMIXDELAY` (default: 1 year).

### 2.5 Unit Conversion

$$B = C \cdot f_{ppb2TgN}$$

Where $f_{ppb2TgN}$ = `N2O_PPB2TGN` = 4.79 TgN/ppb (conversion factor from ppb to TgN).

## 3. State Variables

| Variable | Fortran Name | Symbol | Units | Description | Initial Value |
|----------|--------------|--------|-------|-------------|---------------|
| N2O concentration | `DAT_N2O_CONC%DATGLOBE` | $C$ | ppb | Global mean atmospheric N2O concentration | Read from input file |
| N2O lifetime | `DAT_N2O_TAUTOT%DATGLOBE` | $\tau$ | years | Effective atmospheric lifetime | Calculated |
| Reference concentration | `N2O_YRSTART_CONC` | $C_{00}$ | ppb | Concentration at feedback reference year | Calculated from `N2O_FEED_YRSTART` |
| Natural emissions budget | `N2O_NATEMISBUDGET` | $E_{nat}$ | TgN/yr | Inferred natural emissions | Calculated from budget closure |

## 4. Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Initial lifetime | `N2O_TAUINIT` | years | 139.275 | >0 | Reference atmospheric lifetime |
| ppb to TgN conversion | `N2O_PPB2TGN` | TgN/ppb | 4.79 | Fixed | Mass conversion factor |
| Feedback exponent | `N2O_S` | - | -0.04 | typically [-0.1, 0] | Lifetime feedback strength |
| Stratospheric mixing delay | `N2O_STRATMIXDELAY` | years | 1 | integer >= 0 | Transport delay to stratosphere |
| Feedback reference year | `N2O_FEED_YRSTART` | year | 1925.0 | within run period | Year when feedback reference is set |
| Use variable tau_strat | `N2O_USE_TAUSTRAT_VAR` | flag | 1 | 0 or 1 | Enable meridional flux feedback |
| Tau_strat sensitivity | `N2O_TAUSTRAT_SENS2MERIDFLUX` | - | 0.04 | >= 0 | Lifetime sensitivity to circulation changes |
| Conc to emis switch year | `N2O_SWITCHFROMCONC2EMIS_YEAR` | year | 2015 | within run | Year to switch from prescribed to calculated conc |
| Last budget year | `N2O_LASTBUDGETYEAR` | year | 1991.0 | within run | End year for natural emissions budget calc |
| Budget averaging years | `N2O_BUDGET_AVGYEARS` | years | 10 | integer > 0 | Years to average for budget calculation |
| Emissions scaling factor | `N2O_SCALEEMIS` | - | 1.0 | > 0 | Scale applied to anthropogenic emissions |
| Apply emissions scaling | `N2O_APPLY_SCALEEMIS` | flag | 0 | 0 or 1 | Enable emissions scaling |
| Radiative efficiency | `N2O_RADEFF_WM2PERPPB` | W/m2/ppb | 0.12 | > 0 | Radiative forcing per ppb (used elsewhere) |
| GWP | `N2O_GWP` | - | 298.0 | > 0 | 100-year Global Warming Potential |

## 5. Inputs (per timestep)

| Variable | Units | Source Module | Required? | Fortran Variable |
|----------|-------|---------------|-----------|------------------|
| Current N2O concentration | ppb | Previous timestep / input file | Yes | `DAT_N2O_CONC%DATGLOBE(CURRENT_YEAR_IDX)` |
| Delayed concentration (t-delay) | ppb | Previous timestep | Yes | `DAT_N2O_CONC%DATGLOBE(DELAY_ENDYEAR_IDX)` |
| Delayed concentration (t-delay-1) | ppb | Previous timestep | Yes | `DAT_N2O_CONC%DATGLOBE(DELAY_STARTYEAR_IDX)` |
| Industrial emissions | TgN/yr | Emissions module | Yes | `DAT_N2OI_EMIS%DATGLOBE(CURRENT_YEAR_IDX)` |
| Biomass burning emissions | TgN/yr | Emissions module | Yes | `DAT_N2OB_EMIS%DATGLOBE(CURRENT_YEAR_IDX)` |
| Natural emissions | TgN/yr | Calculated during init | Yes | `DAT_N2ON_EMIS%DATGLOBE(CURRENT_YEAR_IDX)` |
| Tau scale factor | - | Temperature module (optional) | No | `SCALEFACTOR_TAUSTRAT` |
| Surface temperature | K | Climate module | Conditional | For meridional flux feedback |

## 6. Outputs (per timestep)

| Variable | Units | Destination Module(s) | Fortran Variable |
|----------|-------|----------------------|------------------|
| Next year N2O concentration | ppb | Stored, RF calculation | `DAT_N2O_CONC%DATGLOBE(NEXT_YEAR_IDX)` |
| Current N2O lifetime | years | Output, diagnostics | `DAT_N2O_TAUTOT%DATGLOBE(CURRENT_YEAR_IDX)` |
| Inverse emissions (diagnosed) | TgN/yr | Output | `N2O_INVERSE_EMIS(CURRENT_YEAR_IDX)` |
| Total emissions | TgN/yr | Output | `DAT_N2OT_EMIS%DATGLOBE` |

## 7. Algorithm (Pseudocode)

### 7.1 Initialization Phase (in `jump_on_stage`)

```
# Calculate budget indices
ADJUST N2O_LASTBUDGETYEAR if outside valid range
CREATE N2O_TAUINITYEARSIDX array (indices for budget averaging years)

# Calculate annual concentration differences and means
FOR i = 1 to NYEARS-1:
    N2O_DCDT[i] = N2O_CONC[i+1] - N2O_CONC[i]
    N2O_CBAR[i] = (N2O_CONC[i+1] + N2O_CONC[i]) / 2

# Derive natural emissions from mass balance closure
N2O_NATEMISBUDGET = N2O_PPB2TGN * (
    SUM(N2O_DCDT[budget_years-1]) +
    SUM(N2O_CBAR[budget_years - STRATMIXDELAY]) / N2O_TAUINIT
) / N_BUDGET_YEARS
- (SUM(anthro_emissions) over budget years) / 2 / N_BUDGET_YEARS

# Add inferred natural emissions to natural emissions array
DAT_N2ON_EMIS = DAT_N2ON_EMIS + N2O_NATEMISBUDGET
```

### 7.2 Timestep Phase (in `magicc_out` main loop)

```
# Calculate feedback reference concentration
IF current_year == startyear:
    N2O_YRSTART_CONC = current_N2O_concentration
ELSE IF current_year < N2O_FEED_YRSTART:
    N2O_YRSTART_CONC = interpolate(current_conc, next_conc, fractional_year)
ELSE:
    N2O_YRSTART_CONC = previous_value  # locked in

# Calculate stratospheric delay indices
DELAY_ENDYEAR_IDX = MAX(1, CURRENT_YEAR_IDX - N2O_STRATMIXDELAY)
DELAY_STARTYEAR_IDX = MAX(1, CURRENT_YEAR_IDX - N2O_STRATMIXDELAY - 1)

# Sum total emissions
N2O_TOTEMIS = N2OI_EMIS + N2OB_EMIS + N2ON_EMIS

# Calculate tau scaling factor (optional meridional flux feedback)
IF N2O_USE_TAUSTRAT_VAR == 1:
    SCALEFACTOR_TAUSTRAT = 1 / (1 + TEMP_MERIDIONALFLUX *
                                  GEN_MERIDFLUX_CHNGPERDEG *
                                  N2O_TAUSTRAT_SENS2MERIDFLUX)
ELSE:
    SCALEFACTOR_TAUSTRAT = 1.0

# Call NITROUS subroutine
CALL NITROUS(SCALEFACTOR_TAUSTRAT, C, CP, CPP, E, C1, TAUBAR)

# Override with prescribed concentration if before switch year
IF current_year < N2O_SWITCHFROMCONC2EMIS_YEAR:
    USE prescribed N2O_CONC from input file
    RECALCULATE CURRENT_N2O_TAU using prescribed concentration

# Store results
DAT_N2O_TAUTOT = CURRENT_N2O_TAU
IF year >= switch_year:
    DAT_N2O_CONC[next] = NEXTYEAR_N2O_CONC

# Calculate inverse emissions (diagnostic)
N2OSINK = (lagged_avg_conc) / CURRENT_N2O_TAU
N2ODIFF = N2O_CONC[next] - N2O_CONC[current]
N2O_INVERSE_EMIS = (N2ODIFF + N2OSINK) * N2O_PPB2TGN - N2ON_EMIS
```

### 7.3 NITROUS Subroutine (Core Calculation)

```fortran
SUBROUTINE NITROUS(SCALETAU, C, CP, CPP, E, C1, TAUBAR)
    ! Inputs:
    !   SCALETAU = tau scaling factor from meridional flux
    !   C = current year concentration (ppb)
    !   CP = delayed concentration at (t - delay) (ppb)
    !   CPP = delayed concentration at (t - delay - 1) (ppb)
    !   E = total emissions (TgN/yr)
    ! Outputs:
    !   C1 = next year concentration (ppb)
    !   TAUBAR = mid-year lifetime (years)

    # Convert concentrations to burdens
    B = C * N2O_PPB2TGN
    B00 = N2O_YRSTART_CONC * N2O_PPB2TGN  ! Reference burden
    BBARPREV = 0.5 * (CP + CPP) * N2O_PPB2TGN  ! Lagged average burden

    # ITERATION 1
    BBAR = B  ! Initial guess: mid-year = start-year
    TAUBAR = SCALETAU * N2O_TAUINIT * MAX(1.0, BBAR/B00)^S
    DB1 = E - BBARPREV / TAUBAR
    B1 = B + DB1

    # ITERATION 2
    BBAR = (B + B1) / 2
    TAUBAR = SCALETAU * N2O_TAUINIT * MAX(1.0, BBAR/B00)^S
    DB2 = E - BBARPREV / TAUBAR
    B2 = B + DB2

    # ITERATION 3
    BBAR = (B + B2) / 2
    TAUBAR = SCALETAU * N2O_TAUINIT * MAX(1.0, BBAR/B00)^S
    DB3 = E - BBARPREV / TAUBAR
    B3 = B + DB3

    # ITERATION 4
    BBAR = (B + B3) / 2
    TAUBAR = SCALETAU * N2O_TAUINIT * MAX(1.0, BBAR/B00)^S
    DB4 = E - BBARPREV / TAUBAR
    B4 = B + DB4

    # Convert final burden back to concentration
    C1 = B4 / N2O_PPB2TGN
```

## 8. Numerical Considerations

### 8.1 Iteration Count

- **Fixed 4 iterations**: The iterative solver uses exactly 4 fixed-point iterations
- No convergence check is performed
- This is adequate for annual timesteps where year-to-year changes are small

### 8.2 Stability Constraints

- **MAX(1.0, ...)**: The lifetime formula includes a floor to prevent lifetime from decreasing below the reference value when concentrations are below the reference concentration
- This prevents negative feedback from accelerating decline during low-concentration periods

### 8.3 Bounds Checking

- Index clamping: `DELAY_ENDYEAR_IDX = MAX(1, ...)` prevents array underflow
- No explicit bounds on calculated concentrations (can go negative in extreme scenarios)

### 8.4 Known Numerical Issues

- **No convergence check**: The 4-iteration scheme assumes convergence without verification
- **Single precision risk**: Although REAL(8) is used, intermediate calculations could lose precision
- **Edge case at run start**: First year uses `CURRENT_YEAR_IDX = 1` which may cause delay indices to collapse to 1

## 9. Issues and Concerns

### 9.1 Hardcoded Magic Numbers

- **4 iterations**: The number of iterations (4) is hardcoded without justification for why this is sufficient
- **0.5D0 factor**: The averaging factor for lagged burden is embedded in the code

### 9.2 Unclear Variable Names

- **B, B1, B2, B3, B4**: Single-letter variable names for burden iterations make code hard to follow
- **CP, CPP**: These represent delayed concentrations but naming suggests "C previous" and "C previous previous" which is misleading - they are actually different delay offsets
- **S**: Module variable `N2O_S` is copied to local `S` without clear reason
- **BBARPREV**: Misleading name - this is the lagged average burden used for the sink term, not the "previous BBAR"

### 9.3 Missing Documentation

- No explanation of why the sink term uses lagged concentrations while the lifetime uses current mid-year estimate
- No references to scientific literature for the lifetime parameterization
- Comment about Brewer-Dobson circulation mentions "NOTE: To update" suggesting incomplete implementation

### 9.4 Potential Bugs or Edge Cases

- **Feedback initialization**: `N2O_YRSTART_CONC` is determined by `calculate_feedback_conc()` which has a TODO comment about whether it should use extrapolated values
- **Budget calculation complexity**: The natural emissions budget calculation (lines 1648-1657 of MAGICC7.f90) uses a complex formula that averages emissions from `years` and `years-1`, making verification difficult
- **TAUBAR output ambiguity**: The subroutine returns the LAST iteration's TAUBAR but this may differ from the mid-year value implied by B4

### 9.5 Design Issues

- **Mixed responsibilities**: The NITROUS subroutine does both mass balance and lifetime calculation in one loop
- **Global state dependency**: The subroutine relies on module-level variables (`N2O_PPB2TGN`, `N2O_YRSTART_CONC`, `N2O_S`, `N2O_TAUINIT`) making it hard to test in isolation
- **Inconsistent sink formulation**: The sink uses `BBARPREV` (lagged) while lifetime uses `BBAR` (current mid-year), creating asymmetry that may not be physically motivated

### 9.6 Coupling Concerns

- **Temperature feedback path unclear**: The meridional flux feedback depends on temperature from the climate module, but the coupling is managed externally in MAGICC7.f90
- **No box/regional structure**: Unlike some other modules, N2O is only treated globally (`DATGLOBE`) with boxes simply copied from global

### 9.7 Code Quality

- **Unrolled loop**: The 4 iterations are copy-pasted rather than in a loop, risking inconsistency if modified
- **Commented-out debug code**: Line 4369-4370 of MAGICC7.f90 has bracketed/commented debug code

## 10. Test Cases

### 10.1 Unit Tests for NITROUS Subroutine

#### Test 1: Steady State (Zero Emissions, Constant Concentration)

```
Input:
  SCALETAU = 1.0
  C = 300.0 ppb
  CP = 300.0 ppb
  CPP = 300.0 ppb
  E = 300.0 * 4.79 / 139.275 = 10.32 TgN/yr (balanced sink)
  N2O_YRSTART_CONC = 300.0 ppb
  N2O_TAUINIT = 139.275 yr
  N2O_S = -0.04
  N2O_PPB2TGN = 4.79

Expected Output:
  C1 ~ 300.0 ppb (steady state maintained)
  TAUBAR ~ 139.275 yr
```

#### Test 2: Concentration Growth

```
Input:
  SCALETAU = 1.0
  C = 310.0 ppb
  CP = 309.0 ppb
  CPP = 308.0 ppb
  E = 15.0 TgN/yr (higher emissions)
  N2O_YRSTART_CONC = 300.0 ppb (lower reference)

Expected:
  C1 > 310.0 ppb (concentration increases)
  TAUBAR > 139.275 yr (lifetime increases due to higher conc)
```

#### Test 3: Lifetime Floor

```
Input:
  C = 280.0 ppb (below reference)
  N2O_YRSTART_CONC = 300.0 ppb

Expected:
  MAX(1.0, 280/300) = 1.0 is used
  TAUBAR = SCALETAU * N2O_TAUINIT * 1.0 (floor applied)
```

#### Test 4: Meridional Flux Scaling

```
Input:
  SCALETAU = 0.95 (5% reduction due to increased circulation)
  C = 320.0 ppb
  N2O_YRSTART_CONC = 300.0 ppb

Expected:
  TAUBAR = 0.95 * 139.275 * (320/300)^(-0.04)
         = 0.95 * 139.275 * 0.9973
         ~ 131.8 yr
```

### 10.2 Integration Tests

#### Test A: Historical Reconstruction (1750-2015)

- Prescribe historical N2O concentrations
- Verify that inverse emissions match historical estimates
- Check that natural emissions budget (~9-12 TgN/yr) is reasonable

#### Test B: Future Projection (SSP2-4.5)

- Run from 2015 with SSP2-4.5 emissions
- Verify concentration trajectory matches expected ~380 ppb by 2100
- Check lifetime evolution (should increase slightly with concentration)

#### Test C: Budget Closure

- Sum of (emissions - sink) should equal concentration change
- Verify: `dC/dt * N2O_PPB2TGN = E_total - C_lagged_avg * N2O_PPB2TGN / tau`

### 10.3 Edge Case Tests

#### Test E1: First Year of Simulation

- Ensure delay indices don't cause array bounds errors
- Verify initialization of N2O_YRSTART_CONC

#### Test E2: Switch Year Transition

- At N2O_SWITCHFROMCONC2EMIS_YEAR, verify smooth transition from prescribed to calculated

#### Test E3: Very High Emissions

- Test with 50 TgN/yr emissions
- Verify no numerical overflow or unrealistic concentrations

## 11. Fortran Code References

### Primary Source Files

| File | Description |
|------|-------------|
| `/Users/jared/code/magicc/magicc/src/libmagicc/physics/n2o.f90` | N2O module with NITROUS subroutine |
| `/Users/jared/code/magicc/magicc/src/libmagicc/MAGICC7.f90` | Main integration, calls NITROUS |
| `/Users/jared/code/magicc/magicc/src/libmagicc/allcfgs.f90` | Parameter namelist definitions |
| `/Users/jared/code/magicc/magicc/src/libmagicc/physics/deltaq_calculations.f90` | Helper functions including `calculate_feedback_conc` |
| `/Users/jared/code/magicc/magicc/run/MAGCFG_DEFAULTALL.CFG` | Default parameter values |

### Key Line Numbers in n2o.f90

| Lines | Description |
|-------|-------------|
| 3-12 | Module variable declarations |
| 19 | NITROUS subroutine signature |
| 22-25 | Local variable declarations |
| 34-37 | Unit conversions and setup |
| 40-44 | First iteration |
| 49-53 | Second iteration |
| 56-60 | Third iteration |
| 63-68 | Fourth iteration and final conversion |

### Key Line Numbers in MAGICC7.f90

| Lines | Description |
|-------|-------------|
| 566-574 | Allocation of N2O arrays |
| 1606-1660 | N2O initialization and budget calculation |
| 3907-3919 | N2O feedback concentration update |
| 4349-4416 | Main N2O timestep calculation |
| 4352-4355 | Stratospheric delay index calculation |
| 4357-4359 | Total emissions summation |
| 4361-4367 | Meridional flux scaling |
| 4372-4373 | NITROUS subroutine call |
| 4375-4380 | Lifetime recalculation for prescribed mode |
| 4389-4394 | Concentration storage/switching |
| 4406-4408 | Inverse emissions calculation |

### Key Line Numbers in MAGCFG_DEFAULTALL.CFG

| Lines | Description |
|-------|-------------|
| 302-315 | All N2O parameter defaults |
