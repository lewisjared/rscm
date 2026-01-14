# Module 01: CH4 Chemistry

## 1. Scientific Purpose

The CH4 Chemistry module simulates the atmospheric lifetime and concentration of methane (CH4), accounting for the complex feedbacks between methane, tropospheric OH radicals, and co-emitted species (NOx, CO, VOCs). Methane is the second most important anthropogenic greenhouse gas after CO2, and its atmospheric lifetime is not constant but depends on its own concentration (self-feedback via OH depletion) and the concentrations of other reactive species.

The module implements Prather's iterative method from the IPCC TAR (Third Assessment Report) to solve the nonlinear mass balance equation for methane. This is critical because higher methane concentrations deplete OH radicals, which in turn increases methane's atmospheric lifetime, creating a positive feedback loop. The module also includes an optional temperature feedback on methane lifetime, representing how warmer temperatures affect oxidation rates.

## 2. Mathematical Formulation

### 2.1 Governing Mass Balance Equation

The fundamental equation solved is:

$$\frac{dB}{dt} = E - \frac{B}{\tau_{OH}} - \frac{B}{\tau_{other}}$$

Where:

- $B$ = atmospheric methane burden (TgCH4)
- $E$ = total methane emissions (TgCH4/yr)
- $\tau_{OH}$ = tropospheric OH sink lifetime (years)
- $\tau_{other}$ = combined lifetime for non-OH sinks (soil, stratosphere, tropospheric Cl)

### 2.2 OH Feedback Mechanism

The relative change in OH concentration is modeled as (TAR Table 4.11):

$$\frac{\Delta OH}{OH} = S \cdot \frac{\Delta B}{B} + A_{NOx} \cdot \Delta E_{NOx} + A_{CO} \cdot \Delta E_{CO} + A_{VOC} \cdot \Delta E_{VOC}$$

Where:

- $S$ = CH4 self-feedback coefficient (default: -0.32 to -0.34)
- $A_{NOx}$, $A_{CO}$, $A_{VOC}$ = sensitivity coefficients for co-emitted species
- $\Delta E_X$ = change in emissions of species X from reference year

### 2.3 Lifetime Response to OH

$$\frac{\Delta \tau}{\tau} = -\gamma \cdot \frac{\Delta OH}{OH}$$

Where $\gamma$ = `CH4_SCALEOHSENS` (should theoretically be 1.0, but calibrated to ~0.72)

### 2.4 Combined Lifetime Adjustment

The effective OH lifetime at each timestep is:

$$\tau_{bar} = U \cdot \left( \max\left(1.0, \frac{\bar{B}}{B_{00}}\right) \right)^X$$

Where:

- $U = \tau_{00} \cdot \exp(-\gamma \cdot (A_{NOx} \Delta E_{NOx} + A_{CO} \Delta E_{CO} + A_{VOC} \Delta E_{VOC}))$
- $X = -\gamma \cdot S$
- $B_{00}$ = reference burden at feedback start year
- $\bar{B}$ = time-averaged burden during timestep

### 2.5 Temperature Feedback on Lifetime

When enabled (`CH4_INCLUDE_TEMPFEEDBACK = 1`):

$$\tau_{bar,adjusted} = \frac{\tau_{00}}{\frac{\tau_{00}}{\tau_{bar}} + \alpha_T \cdot \Delta T}$$

Where:

- $\alpha_T$ = `CH4_TAUTEMPSENSITIVITY` (default: 0.07)
- $\Delta T$ = temperature change from feedback reference year

### 2.6 Combined Non-OH Lifetime

$$\frac{1}{\tau_{other}} = \frac{1}{\tau_{soil}} + \frac{1}{\tau_{strat}} + \frac{1}{\tau_{tropcl}}$$

### 2.7 Prather's Iterative Method

The method uses 4 iterations to solve the implicit equations:

**Iteration n (for n = 1,2,3,4):**
$$\bar{B}_n = \frac{B + B_{n-1}}{2}$$
$$\tau_{bar,n} = U \cdot \left(\max\left(1, \frac{\bar{B}_n}{B_{00}}\right)\right)^X \cdot \left(1 - 0.5 X \frac{\Delta B_{n-1}}{B}\right)$$
$$\Delta B_n = E - \frac{\bar{B}_n}{\tau_{bar,n}} - \frac{\bar{B}_n}{\tau_{other}}$$
$$B_n = B + \Delta B_n$$

Note: The correction factor $(1 - 0.5 X \frac{\Delta B_{n-1}}{B})$ is applied from iteration 2 onwards.

## 3. State Variables

| Variable | Fortran Name | Symbol | Units | Description | Initial Value |
|----------|--------------|--------|-------|-------------|---------------|
| CH4 concentration | `DAT_CH4_CONC%DATGLOBE` | $C$ | ppb | Global mean atmospheric CH4 mixing ratio | From input file |
| Effective OH lifetime | `CH4_TAUOH_EFFECTIVE` | $\tau_{OH}$ | years | Time-varying tropospheric OH sink lifetime | `CH4_TAUOH_INIT` |
| Reference concentration | `CH4_YRSTART_CONC` | $C_{00}$ | ppb | CH4 concentration at feedback start year | Set from `DAT_CH4_CONC` at `CH4_FEED_YRSTART` |
| Reference temperature | `CH4_YRSTART_TEMP` | $T_{00}$ | K (anomaly) | Temperature at feedback start year | Set from `DAT_SURFACE_TEMP` at `CH4_FEED_YRSTART` |
| Initial OH lifetime | `CH4_TAUOH_INIT` | $\tau_{OH,init}$ | years | Derived from `CH4_TAUTOT_INIT` | Calculated in `methane_calc_budget` |
| Combined other lifetime | `CH4_TAUOTHER` | $\tau_{other}$ | years | Combined soil+strat+tropcl lifetime | Calculated in `methane_calc_budget` |
| Natural emission budget | `CH4_NATEMISBUDGET` | - | TgCH4/yr | Inferred natural emissions for budget closure | Calculated in `methane_calc_budget` |

## 4. Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Total initial lifetime | `CH4_TAUTOT_INIT` | years | 9.9474 | 8-12 | Total atmospheric lifetime at reference period |
| Soil sink lifetime | `CH4_TAUSOIL` | years | 150.0 | 100-200 | Partial lifetime for soil uptake sink |
| Stratospheric lifetime | `CH4_TAUSTRAT` | years | 120.0 | 100-150 | Partial lifetime for stratospheric destruction |
| Tropospheric Cl lifetime | `CH4_TAUTROPCL` | years | 200.0 | 150-300 | Partial lifetime for tropospheric Cl reaction |
| PPB to TgCH4 conversion | `CH4_PPB2TGCH4` | TgCH4/ppb | 2.824 | - | Mass conversion factor (16g/mol x 0.1765 Tmol/ppb) |
| Mixing box size | `CH4_MIXBOXSIZE` | dimensionless | 0.973 | 0.9-1.0 | Atmospheric mixing scaling factor |
| OH sensitivity scaling | `CH4_SCALEOHSENS` | dimensionless | 0.72448 | 0.5-1.5 | Scaling for OH-lifetime dependency ($\gamma$) |
| CH4 self-feedback | `CH4_S` | dimensionless | -0.53775 | -0.6 to -0.3 | $S$ coefficient for CH4 feedback on OH |
| NOx feedback coefficient | `CH4_ANOX` | (TgN/yr)^-1 | 0.0093376 | - | $A_{NOx}$ - NOx effect on OH |
| CO feedback coefficient | `CH4_ACO` | (TgCO/yr)^-1 | -0.000113 | - | $A_{CO}$ - CO effect on OH |
| VOC feedback coefficient | `CH4_AVOC` | (TgC/yr)^-1 | -0.0003142 | - | $A_{VOC}$ - VOC effect on OH |
| Temperature sensitivity | `CH4_TAUTEMPSENSITIVITY` | K^-1 | 0.07 | 0-0.15 | Temperature feedback on lifetime |
| Temperature feedback flag | `CH4_INCLUDE_TEMPFEEDBACK` | flag | 1 | 0 or 1 | Enable temperature feedback |
| NOx/VOC/CO feedback flag | `CH4_TAUFEEDBACK_BYNOXVOCCO` | flag | 1 | 0 or 1 | Enable emissions feedback on OH |
| Feedback start year | `CH4_FEED_YRSTART` | year | 1927.0 | - | Year when feedbacks activate |
| Conc-to-emis switch year | `CH4_SWITCHFROMCONC2EMIS_YEAR` | year | 2015 | - | Year to switch from prescribed conc to calculated |
| Budget calculation period | `CH4_BUDGET_AVGYEARS` | years | 10 | 5-20 | Years for budget averaging |
| Last budget year | `CH4_LASTBUDGETYEAR` | year | 2004.0 | - | End year for budget calculation period |
| Radiative efficiency | `CH4_RADEFF_WM2PERPPB` | W/m2/ppb | 0.036 | 0.03-0.04 | Radiative efficiency (not used in this module) |
| Strat H2O percent | `CH4_ADDEDSTRATH2O_PERCENT` | % | 0.0923 | - | Fraction of CH4 oxidation adding strat H2O |
| CH4 include oxidation | `CH4_INCL_CH4OX` | flag | 1 | 0 or 1 | Include CH4 oxidation to CO2 |
| Fossil fuel fraction | `CH4_FOSSFUELFRACTION` | fraction | 0.18 | 0-1 | Prescribed fossil fraction (if applied) |
| Apply fossil fraction | `CH4_APPLYFOSSFUELFRACTION` | flag | 0 | 0 or 1 | Use prescribed vs calculated fossil fraction |
| Clathrate feedback start | `CH4_CLATHRATEFEED_YRSTART` | year | 2010 | - | Year clathrate feedback begins |
| Clathrate feedback apply | `CH4_CLATHRATEFEED_APPLY` | flag | 0 | 0 or 1 | Enable clathrate emissions |
| Clathrate feedback alpha | `CH4_CLATHRATEFEED_ALPHA` | fraction/yr | 0.001 | 0-0.01 | Release rate from clathrate pool |
| Clathrate initial pool | `CH4_CLATHRATEFEED_INIPOOL` | GtC | 5000.0 | - | Initial methane clathrate reservoir |
| GWP | `CH4_GWP` | dimensionless | 25.0 | 20-35 | 100-year Global Warming Potential |
| Wetland slope | `CH4_WETLAND_SLOPE` | MtCH4/yr/K | 22.4 | - | Temperature sensitivity of wetland emissions |

## 5. Inputs (per timestep)

| Variable | Units | Source Module/Data | Required? | Fortran Variable |
|----------|-------|-------------------|-----------|------------------|
| Previous CH4 concentration | ppb | Datastore | Yes | `DAT_CH4_CONC%DATGLOBE(CURRENT_YEAR_IDX)` |
| Total CH4 emissions | TgCH4/yr | Summed from I+B+N+PF+Clath | Yes | `CH4_TOTEMIS` |
| Delta NOx emissions | TgN/yr | Ozone/emissions module | If feedback on | `TROPOZ_DELTA_NOXEMIS` |
| Delta CO emissions | TgCO/yr | Emissions module | If feedback on | `TROPOZ_DELTA_COEMIS` |
| Delta VOC emissions | TgC/yr | Emissions module | If feedback on | `TROPOZ_DELTA_NMVOCEMIS` |
| Temperature change | K | Climate module | If temp feedback on | `CH4_FEED_DELTATEMP` |
| Initial OH lifetime | years | Budget calculation | Yes | `CH4_TAUOH_INIT` |

### Emission Components

| Component | Fortran Variable | Description |
|-----------|------------------|-------------|
| Industrial | `DAT_CH4I_EMIS%DATGLOBE` | Fossil fuel and industrial emissions |
| Biomass burning | `DAT_CH4B_EMIS%DATGLOBE` | Biomass and biofuel burning |
| Natural | `DAT_CH4N_EMIS%DATGLOBE` | Natural sources (wetlands, etc.) + budget adjustment |
| Permafrost | `DAT_CH4PF_EMIS%DATGLOBE` | Permafrost thaw emissions (if enabled) |
| Clathrate | `DAT_CH4CLATHRATE_EMIS%DATGLOBE` | Clathrate release (if enabled) |

## 6. Outputs (per timestep)

| Variable | Units | Destination Module(s) | Fortran Variable |
|----------|-------|----------------------|------------------|
| Next year CH4 concentration | ppb | Stored, radiative forcing | `NEXTYEAR_CH4_CONC` -> `DAT_CH4_CONC%DATGLOBE(NEXT_YEAR_IDX)` |
| Effective OH lifetime | years | Stored for diagnostics | `CH4_TAUOH_EFFECTIVE(CURRENT_YEAR_IDX)` -> `DAT_CH4_TAUOH%DATGLOBE` |
| Total lifetime | years | Diagnostics | `DAT_CH4_TAUTOT%DATGLOBE` (calculated externally) |
| Inverse emissions | TgCH4/yr | Carbon tracking | `CH4_INVERSE_EMIS(CURRENT_YEAR_IDX)` |

## 7. Algorithm (Pseudocode)

### 7.1 Initialization (`methane_calc_budget`)

```
FUNCTION methane_calc_budget():
    # Calculate combined non-OH lifetime
    IF CH4_TAUSOIL != 0 AND CH4_TAUSTRAT != 0 AND CH4_TAUTROPCL != 0:
        CH4_TAUOTHER = 1 / (1/CH4_TAUSOIL + 1/CH4_TAUSTRAT + 1/CH4_TAUTROPCL)
    ELSE:
        # Handle cases with one or more zero lifetimes (multiple branches)
        # ... (see code for full logic)

    # Derive initial OH lifetime from total lifetime
    CH4_TAUOH_INIT = 1 / (1/CH4_TAUTOT_INIT - 1/CH4_TAUOTHER)

    # Adjust CH4_LASTBUDGETYEAR to be within valid range
    IF DAT_CH4_CONC.LASTYEAR <= CH4_LASTBUDGETYEAR:
        CH4_LASTBUDGETYEAR = DAT_CH4_CONC.LASTYEAR
    # ... (additional bounds checking)

    # Create index array for budget years
    FOR i = 1 to SIZE(CH4_TAUINITYEARSIDX):
        CH4_TAUINITYEARSIDX[i] = CH4_LASTBUDGETYEAR - STARTYEAR - SIZE + 1 + i

    # Calculate dC/dt and mean concentrations for budget period
    CH4_DCDT[1:NYEARS-1] = CH4_CONC[2:NYEARS] - CH4_CONC[1:NYEARS-1]
    CH4_CBAR[1:NYEARS-1] = (CH4_CONC[2:NYEARS] + CH4_CONC[1:NYEARS-1]) / 2

    # Calculate natural emissions to close budget
    # (Mass balance: E_nat = dB/dt + B/tau_OH + B/tau_other - E_anthro)
    CH4_NATEMISBUDGET = CH4_PPB2TGCH4 * CH4_MIXBOXSIZE * (
        SUM(CH4_DCDT[budget_years]) +
        SUM(CH4_CBAR[budget_years]) / CH4_TAUOH_INIT +
        SUM(CH4_CBAR[budget_years]) / CH4_TAUOTHER
    ) / N_BUDGET_YEARS - AVG(all_anthro_emissions)

    # Add budget-derived natural emissions to DAT_CH4N_EMIS
    DAT_CH4N_EMIS.DATGLOBE += CH4_NATEMISBUDGET
```

### 7.2 Main Timestep (`METHANE` subroutine)

```
SUBROUTINE METHANE(ICH4F, CPREV, E, DEN, DEC, DEV, CONC, TAU00, TAUOUT, S, AANOX, AACO, AAVOC, TEMP):
    # Input:
    #   ICH4F  = temperature feedback flag (0 or 1)
    #   CPREV  = previous year CH4 concentration (ppb)
    #   E      = total emissions (TgCH4/yr)
    #   DEN    = delta NOx emissions from reference
    #   DEC    = delta CO emissions from reference
    #   DEV    = delta VOC emissions from reference
    #   TAU00  = initial OH lifetime (years)
    #   S      = CH4 self-feedback coefficient
    #   AANOX, AACO, AAVOC = feedback coefficients
    #   TEMP   = temperature change for feedback
    # Output:
    #   CONC   = new CH4 concentration (ppb)
    #   TAUOUT = effective OH lifetime (years)

    # Convert concentration to burden (mass)
    B = CPREV * CH4_PPB2TGCH4 * CH4_MIXBOXSIZE      # Current burden
    B00 = CH4_YRSTART_CONC * CH4_PPB2TGCH4 * CH4_MIXBOXSIZE  # Reference burden

    # Calculate emission-adjusted lifetime factor
    AAA = EXP(-CH4_SCALEOHSENS * (AANOX*DEN + AACO*DEC + AAVOC*DEV))
    X = -CH4_SCALEOHSENS * S
    U = TAU00 * AAA

    # === ITERATION 1 ===
    BBAR = B
    TAUBAR = U * (MAX(1.0, BBAR/B00))^X
    IF ICH4F == 1:
        TAUBAR = TAU00 / (TAU00/TAUBAR + CH4_TAUTEMPSENSITIVITY * TEMP)
    DB1 = E - BBAR/TAUBAR - BBAR/CH4_TAUOTHER
    B1 = B + DB1

    # === ITERATION 2 ===
    BBAR = (B + B1) / 2
    TAUBAR = U * (MAX(1.0, BBAR/B00))^X
    TAUBAR = TAUBAR * (1 - 0.5*X*DB1/B)  # Correction factor added
    IF ICH4F == 1:
        TAUBAR = TAU00 / (TAU00/TAUBAR + CH4_TAUTEMPSENSITIVITY * TEMP)
    DB2 = E - BBAR/TAUBAR - BBAR/CH4_TAUOTHER
    B2 = B + DB2

    # === ITERATION 3 ===
    BBAR = (B + B2) / 2
    TAUBAR = U * (MAX(1.0, BBAR/B00))^X
    TAUBAR = TAUBAR * (1 - 0.5*X*DB2/B)
    IF ICH4F == 1:
        TAUBAR = TAU00 / (TAU00/TAUBAR + CH4_TAUTEMPSENSITIVITY * TEMP)
    DB3 = E - BBAR/TAUBAR - BBAR/CH4_TAUOTHER
    B3 = B + DB3

    # === ITERATION 4 ===
    BBAR = (B + B3) / 2
    TAUBAR = U * (MAX(1.0, BBAR/B00))^X
    TAUBAR = TAUBAR * (1 - 0.5*X*DB3/B)
    IF ICH4F == 1:
        TAUBAR = TAU00 / (TAU00/TAUBAR + CH4_TAUTEMPSENSITIVITY * TEMP)
    DB4 = E - BBAR/TAUBAR - BBAR/CH4_TAUOTHER
    B4 = B + DB4

    # Output results
    TAUOUT = TAUBAR
    CONC = B4 / (CH4_PPB2TGCH4 * CH4_MIXBOXSIZE)

    IF is_nan(CONC):
        LOG_ERROR("Methane concentration is NaN")
```

### 7.3 Calling Context (from MAGICC7.f90)

```
# Before calling METHANE:

# 1. Calculate temperature feedback delta
ch4_feedback_temps = calculate_feedback_temps(
    CH4_FEED_YRSTART,
    CH4_YRSTART_TEMP,
    current_surface_temp,
    extrapolated_temp,
    current_year,
    logic_flag=2
)
CH4_FEED_DELTATEMP = ch4_feedback_temps(1)
CH4_YRSTART_TEMP = ch4_feedback_temps(2)

# 2. Calculate delta emissions from reference year
IF CH4_TAUFEEDBACK_BYNOXVOCCO == 1:
    TROPOZ_DELTA_NOXEMIS = current_NOx - reference_NOx
    TROPOZ_DELTA_COEMIS = current_CO - reference_CO
    TROPOZ_DELTA_NMVOCEMIS = current_VOC - reference_VOC

# 3. Apply wetland temperature feedback (if after budget years)
IF current_year > CH4_LASTBUDGETYEAR:
    avg_temp_budget_years = AVG(temp over CH4_TAUINITYEARSIDX)
    DAT_CH4N_EMIS += CH4_WETLAND_SLOPE * (current_temp - avg_temp_budget_years)

# 4. Sum all emission sources
CH4_TOTEMIS = CH4I + CH4B + CH4N
IF PF_APPLY == 1:
    CH4_TOTEMIS += CH4PF
IF CH4_CLATHRATEFEED_APPLY == 1:
    CH4_TOTEMIS += CH4CLATHRATE

# 5. Call METHANE subroutine
CALL METHANE(...)

# 6. Store results
DAT_CH4_TAUOH = CH4_TAUOH_EFFECTIVE
DAT_CH4_TAUTOT = 1 / (1/CH4_TAUOH_EFFECTIVE + 1/CH4_TAUOTHER)

# 7. Update concentration if past switch year
IF current_year >= CH4_SWITCHFROMCONC2EMIS_YEAR:
    DAT_CH4_CONC(NEXT_YEAR_IDX) = NEXTYEAR_CH4_CONC
```

## 8. Numerical Considerations

### 8.1 Number of Iterations

The code uses exactly **4 iterations** of Prather's method. This is hardcoded and not configurable. The original TAR method may have used fewer iterations; 4 appears to be a design choice for additional accuracy.

### 8.2 Stability Constraints

**Lower bound on concentration ratio:**

```fortran
TAUBAR = U * (MAX(1.0D0, (BBAR / B00))**X)
```

The `MAX(1.0, BBAR/B00)` ensures that the lifetime adjustment factor never decreases below what it would be at the reference concentration. This prevents numerical issues when concentrations drop below reference levels and ensures monotonic behavior.

### 8.3 Clamping and Bounds

1. **Temperature feedback delta:** In `calculate_feedback_temps`, negative delta temperatures are clamped to zero:

   ```fortran
   if (feedback_deltatemp < 0.0D0) feedback_deltatemp = 0.0D0
   ```

2. **NaN check:** The subroutine explicitly checks for NaN in the output concentration and logs an error.

3. **No explicit upper/lower bounds** on concentration or lifetime outputs.

### 8.4 Potential Numerical Issues

1. **Division by zero risk:** If `B` (current burden) is zero, the correction factor `(1 - 0.5*X*DB/B)` will cause division by zero. The code does not guard against this.

2. **Negative concentrations:** If emissions go strongly negative (e.g., aggressive removal scenarios), the iterative method could produce negative concentrations. No guard exists.

3. **Large timesteps:** The method assumes annual timesteps. Sub-annual or multi-year steps would require modification.

## 9. Issues and Concerns

### 9.1 Hardcoded Magic Numbers

| Value | Location | Concern |
|-------|----------|---------|
| 4 iterations | Lines 88-122 | Hardcoded, not configurable. Why 4? No convergence check. |
| `0.5` in correction factor | Lines 100, 109, 118 | Magic number in `(1 - 0.5*X*DB/B)`. Origin unclear. |
| `2.0D0` averaging | Lines 98, 107, 116 | Simple averaging, could be weighted differently. |
| `1.0D0` floor | Lines 91, 99, 108, 117 | Prevents negative feedback, but is this physically justified? |

### 9.2 Unclear Variable Names

| Fortran Name | Issue |
|--------------|-------|
| `AAA` | Cryptic name for emission adjustment factor |
| `U` | Single letter, represents `TAU00 * AAA` |
| `X` | Single letter, represents `-CH4_SCALEOHSENS * S` |
| `B`, `B00` | Burden vs reference burden - ok but could be clearer |
| `DB1`, `DB2`, etc. | Delta burden per iteration |
| `ICH4F` | Unclear abbreviation for temperature feedback flag |

### 9.3 Code Documentation Issues

1. **Commented-out reference:** Lines 30-32 show the calling convention but as a comment, not a docstring.
2. **TAR references:** Comments mention "TAR method" and "Table 4.11" but no formal citation or equation numbers.
3. **CH4_SCALEOHSENS confusion:** Comment on line 57 says "SHOULD BE =-1 = -1.145" but default is +0.72448. The sign convention and value are unclear.

### 9.4 Potential Bugs and Edge Cases

1. **Division by B=0:** No guard against zero burden in correction factor calculation.

2. **CH4_TAUOTHER calculation:** The `methane_calc_budget` function has an extremely verbose if-else chain (lines 157-174) handling combinations of zero lifetimes. One branch (line 167-168) is a no-op:

   ```fortran
   ELSEIF (CH4_TAUOTHER /= 0.0D0) THEN
       CH4_TAUOTHER = CH4_TAUOTHER
   ```

   This appears to be dead code or a bug.

3. **Index off-by-one risk:** The budget calculation uses `CH4_TAUINITYEARSIDX - 1` in some places, which could cause array bounds issues if not carefully managed.

4. **Temperature feedback activation:** The `logic_flag=2` in the calling code means feedback activates one year earlier than the literal interpretation of `CH4_FEED_YRSTART`. This is noted in a comment as "unintended logic."

### 9.5 Design Issues

1. **Module-level state:** Heavy use of module-level `SAVE` variables makes the code difficult to test and reason about. State is scattered between `mod_methane` and the calling code.

2. **Tight coupling:** The METHANE subroutine depends on module-level variables (`CH4_PPB2TGCH4`, `CH4_MIXBOXSIZE`, `CH4_TAUOTHER`, `CH4_YRSTART_CONC`, `CH4_SCALEOHSENS`) that are not passed as parameters.

3. **Input preprocessing:** Much of the input calculation (delta emissions, temperature feedback) happens in the calling code (MAGICC7.f90) rather than in the methane module. This splits the logic across files.

4. **No convergence check:** The 4 iterations are always performed regardless of whether the solution has converged.

5. **Wetland feedback location:** The wetland temperature feedback on natural emissions is applied in MAGICC7.f90 (lines 4006-4015), not in the methane module. This is a leak of methane-specific physics into the main driver.

### 9.6 Parameter Inconsistencies

1. **CH4_S vs S in TAR:** The code comment says S = -0.32 (Table 4.11) or -0.34 (Table 4.2), but the default is -0.53775. The discrepancy is not explained.

2. **CH4_SCALEOHSENS:** Comment says "should be -1 = -1.145" but default is +0.72448. The sign and magnitude differ significantly.

## 10. Test Cases

### 10.1 Unit Test: Steady State

**Purpose:** Verify that constant emissions produce stable concentration.

**Setup:**

```
CPREV = 1800.0 ppb
E = 570.0 TgCH4/yr (typical modern emissions)
DEN = DEC = DEV = 0.0 (no feedback from co-emissions)
TEMP = 0.0 (no temperature feedback)
TAU00 = 9.3 years (typical OH lifetime)
ICH4F = 0
CH4_TAUTOT_INIT = 9.9474 years
CH4_TAUSOIL = 150.0, CH4_TAUSTRAT = 120.0, CH4_TAUTROPCL = 200.0
CH4_PPB2TGCH4 = 2.824, CH4_MIXBOXSIZE = 0.973
CH4_SCALEOHSENS = 0.72448, CH4_S = -0.53775
CH4_YRSTART_CONC = 1800.0 ppb
```

**Expected output:**

- `CONC` should be close to `CPREV` (within ~1%)
- `TAUOUT` should equal `TAU00` (no feedbacks active)

### 10.2 Unit Test: Concentration Increase with Emissions Spike

**Purpose:** Verify concentration rises with increased emissions.

**Setup:**

- Same as 10.1, but `E = 700.0 TgCH4/yr`

**Expected output:**

- `CONC > CPREV`
- `TAUOUT > TAU00` (lifetime increases due to OH depletion from higher CH4)

### 10.3 Unit Test: Temperature Feedback

**Purpose:** Verify temperature feedback shortens lifetime.

**Setup:**

- Same as 10.1, but `ICH4F = 1`, `TEMP = 2.0` K

**Expected output:**

- `TAUOUT < TAU00` (warmer temperature increases OH, reducing lifetime)
- Verify formula: `TAUOUT = TAU00 / (1 + CH4_TAUTEMPSENSITIVITY * TEMP)`

### 10.4 Unit Test: NOx Emissions Feedback

**Purpose:** Verify NOx increases OH and decreases lifetime.

**Setup:**

- Same as 10.1, but `DEN = 10.0` TgN/yr

**Expected output:**

- `TAUOUT < TAU00` (NOx increases OH)
- Should follow: `AAA = exp(-0.72448 * 0.0093376 * 10) ~ 0.935`

### 10.5 Integration Test: Historical Reconstruction

**Purpose:** Verify model reproduces observed CH4 concentrations.

**Setup:**

- Run from 1750-2020 with historical emissions
- Compare to ice core and direct measurements

**Validation:**

- Pre-industrial concentration ~700 ppb
- 1980 concentration ~1570 ppb
- 2020 concentration ~1880 ppb
- All within 5% of observations

### 10.6 Edge Case: Zero Emissions

**Purpose:** Test behavior with no emissions.

**Setup:**

- `E = 0.0`

**Expected output:**

- Concentration should decay exponentially
- `CONC ~ CPREV * exp(-1/tau_total)` approximately

### 10.7 Edge Case: Very Low Concentration

**Purpose:** Test numerical stability near floor.

**Setup:**

- `CPREV = 100.0 ppb` (much lower than reference)
- `CH4_YRSTART_CONC = 1800.0 ppb`

**Expected output:**

- `MAX(1.0, BBAR/B00)` should clamp to 1.0
- Lifetime should equal base lifetime (no concentration feedback)

## 11. Fortran Code References

### Key File Locations

| Function/Section | File | Line Numbers |
|------------------|------|--------------|
| Module variables | `methane.f90` | 4-20 |
| METHANE subroutine signature | `methane.f90` | 34-35 |
| First iteration | `methane.f90` | 88-95 |
| Second iteration | `methane.f90` | 96-103 |
| Third iteration | `methane.f90` | 104-112 |
| Fourth iteration | `methane.f90` | 113-121 |
| Output assignment | `methane.f90` | 125-128 |
| NaN check | `methane.f90` | 130-132 |
| Budget calculation | `methane.f90` | 136-233 |
| CH4_TAUOTHER calculation | `methane.f90` | 157-174 |
| CH4_TAUOH_INIT derivation | `methane.f90` | 178 |
| Natural emissions budget | `methane.f90` | 205-219 |
| METHANE call site | `MAGICC7.f90` | 4031-4035 |
| Temperature feedback calc | `MAGICC7.f90` | 3874-3883 |
| Delta emissions calc | `MAGICC7.f90` | 3948-4002 |
| Wetland feedback | `MAGICC7.f90` | 4006-4015 |
| Total emissions sum | `MAGICC7.f90` | 4018-4029 |
| Parameter definitions | `allcfgs.f90` | 56-65 |
| Default parameter values | `MAGCFG_DEFAULTALL.CFG` | 10-38 |

### Key Equations by Line

| Equation | File:Line |
|----------|-----------|
| Burden conversion: `B = C * ppb2tg * mixbox` | methane.f90:71 |
| Reference burden: `B00 = C00 * ppb2tg * mixbox` | methane.f90:72 |
| Emission factor: `AAA = exp(-gamma * (A_nox*DEN + A_co*DEC + A_voc*DEV))` | methane.f90:73 |
| Feedback exponent: `X = -gamma * S` | methane.f90:74 |
| Base adjusted lifetime: `U = TAU00 * AAA` | methane.f90:75 |
| Lifetime with concentration feedback: `TAUBAR = U * MAX(1,BBAR/B00)^X` | methane.f90:91 |
| Temperature feedback: `TAUBAR = TAU00 / (TAU00/TAUBAR + alpha_T * TEMP)` | methane.f90:92 |
| Mass balance: `DB = E - BBAR/TAUBAR - BBAR/TAUOTHER` | methane.f90:93 |
| Lifetime correction (iter 2+): `TAUBAR = TAUBAR * (1 - 0.5*X*DB_prev/B)` | methane.f90:100 |
| Combined other lifetime: `1/TAUOTHER = 1/soil + 1/strat + 1/tropcl` | methane.f90:158 |
| OH lifetime from total: `TAUOH = 1/(1/TAUTOT - 1/TAUOTHER)` | methane.f90:178 |

---

## Summary

The CH4 chemistry module implements a well-established iterative method (Prather's TAR approach) for solving the nonlinear methane mass balance equation with OH feedback. The core algorithm is sound, but the implementation suffers from:

1. **Tight coupling** between the module and calling code
2. **Hardcoded iteration count** without convergence checking
3. **Cryptic variable names** and sparse inline documentation
4. **Module-level state** that complicates testing
5. **Some apparent bugs** (no-op branch, potential division by zero)
6. **Parameter values** that differ from cited references without explanation

For reimplementation, the mathematical formulation is clear, but careful attention should be paid to the exact sequencing of operations, the floor function on the concentration ratio, and the correction factor in iterations 2-4.
