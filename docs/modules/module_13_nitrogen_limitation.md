# Module 13: Nitrogen Limitation

## 1. Scientific Purpose

The Nitrogen Limitation module represents the constraint that nitrogen availability places on terrestrial carbon uptake in the global carbon cycle. In natural ecosystems, plant growth is often limited not just by CO2 and temperature but by available nitrogen in soils. As atmospheric CO2 increases, the "CO2 fertilization effect" drives higher potential NPP (Net Primary Production), but this enhanced growth requires proportionally more nitrogen. If soil nitrogen cannot keep pace with carbon demand, the effective CO2 fertilization is reduced.

This module calculates a "limitation factor" (0 to 1) that is applied to NPP to account for nitrogen constraints on terrestrial carbon uptake. The module tracks a plant-available soil nitrogen pool and models nitrogen fluxes including biological fixation, atmospheric deposition, plant uptake, recycling through decomposition, and losses through leaching and gaseous emissions. The core assumption is a quasi-steady-state approach where the nitrogen pool adjusts each timestep to balance sources and sinks.

## 2. Experimental Status Assessment

**WARNING: THIS MODULE IS EXPLICITLY MARKED AS UNFINISHED IN THE CODE.**

At line 38 of `nitrogen_limitation.f90`, when the module is activated (`NCYCLE_APPLY == 1`), it logs:

```fortran
CALL logger % warning('NITROGEN_INIT', "Careful using the nitrogen cycle, it's not finished yet")
```

### Known Issues

1. **Copy-Paste Bug**: There is a clear bug at lines 106-107, 156-157, 326-327, and 388-389 where `DAT_NH3I_EMIS` is used twice instead of using both `DAT_NH3I_EMIS` and `DAT_NH3B_EMIS`. This means biomass burning NH3 emissions are ignored when calculating nitrogen deposition from emissions.

2. **Unused Input Files**: `FILE_NRECYC_IN` and `FILE_NLOSS_IN` are read into `DAT_NCYCLE_NRECYC` and `DAT_NCYCLE_NLOSS` datastores, but these are never actually used in any calculations. This suggests incomplete implementation.

3. **Obsolete Tuning Parameters**: The tuning file `MAGTUNE_1PCTCO2_CN.CFG` uses parameter names like `KN_RECYC`, `KN_UPTAKE`, `N_SENS_FACTOR_A`, etc. that do not match the current codebase variable names, suggesting a disconnect between legacy calibration work and the current implementation.

4. **Missing Default Values**: `NCYCLE_NFIX_VMAX` and `NCYCLE_NFIX_KM` (Michaelis-Menten parameters for nitrogen fixation) are not present in `MAGCFG_DEFAULTALL.CFG`, meaning they are uninitialized when `NCYCLE_APPLY_PROGNOSTIC_NFIX == 1`.

5. **Hardcoded Unit Conversions**: Multiple `1.0D-03` conversion factors are embedded in equations without explanation of the unit conversion being performed.

**Recommendation**: Do not use this module for production runs without thorough validation and bug fixes.

## 3. Mathematical Formulation

### 3.1 Nitrogen Pool Balance Equation

The module assumes quasi-steady-state each timestep, solving for the nitrogen pool size N where:

```
dN/dt = 0 = (F_dep + F_recyc + F_fix) - (F_up + F_loss)
```

Where:

- `F_dep` = Nitrogen deposition flux (Pg N yr^-1)
- `F_recyc` = Nitrogen recycling from decomposition (Pg N yr^-1)
- `F_fix` = Biological nitrogen fixation (Pg N yr^-1)
- `F_up` = Plant nitrogen uptake (Pg N yr^-1)
- `F_loss` = Nitrogen losses (leaching + gaseous) (Pg N yr^-1)

### 3.2 Nitrogen Deposition

Two modes:

**Mode 1: External Input** (`NCYCLE_APPLY_NDEP_FROM_EMIS == 0`):

```
F_dep = DAT_NCYCLE_NDEP(t)
```

**Mode 2: From Emissions** (`NCYCLE_APPLY_NDEP_FROM_EMIS == 1`):

```
F_dep = NCYCLE_FRAC_NDEP_LAND * 1e-3 * (NOx_i + NOx_b + NH3_i + NH3_i)
```

Note: The `NH3_i + NH3_i` is a bug - should be `NH3_i + NH3_b`.

### 3.3 Nitrogen Recycling

```
F_recyc = NCYCLE_NCRATIO_RECYC * RH * adj_fert * adj_temp

adj_fert = (1 + NCYCLE_RECYC2CNRATIO_FACTOR) / (NCYCLE_RECYC2CNRATIO_FACTOR + CO2_EFF_FERTILIZATION_FACTOR)

adj_temp = (1 + NCYCLE_RECYC2CNRATIO_FACTOR) / (NCYCLE_RECYC2CNRATIO_FACTOR + CO2_EFF_NPP_TEMPFEEDBACK)
```

Where `RH` is heterotrophic respiration (or NPP in year 1).

### 3.4 Nitrogen Fixation

**Mode 1: External Input** (`NCYCLE_APPLY_PROGNOSTIC_NFIX == 0`):

```
F_fix = DAT_NCYCLE_NFIX(t)
```

**Mode 2: Prognostic** (`NCYCLE_APPLY_PROGNOSTIC_NFIX == 1`):

For initial (pre-fertilization) period:

```
F_fix = 1e-3 * (NCYCLE_NFIX_VMAX * NPP) / (NCYCLE_NFIX_KM + NPP)
```

For active period (Michaelis-Menten with N-availability modulation):

```
phi_N = (1 + A) / (A + exp(-C * (N^2/(N + S) - k_up * NPP)))
phi_N_0 = (1 + A) / (A + exp(-C * (N_0^2/(N_0 + S) - k_up * NPP_0)))

F_fix = 1e-3 * (NCYCLE_NFIX_VMAX * NPP * (phi_N/phi_N_0)) / (NCYCLE_NFIX_KM + NPP * (phi_N/phi_N_0))
```

Where:

- A = `NCYCLE_NPP2NAVAIL_FACTOR_A`
- C = `NCYCLE_NPP2NAVAIL_FACTOR_C`
- S = `NCYCLE_NCRATIO_UPTAKE_SUPPLY`
- k_up = `NCYCLE_NCRATIO_UPTAKE`
- N_0 = Pre-industrial nitrogen pool
- NPP_0 = Pre-industrial NPP

### 3.5 Plant Nitrogen Uptake

For initial period:

```
F_up = NCYCLE_NCRATIO_UPTAKE * NPP
```

For active period:

```
F_up = NCYCLE_NCRATIO_UPTAKE * NPP * (phi_N / phi_N_0)
```

### 3.6 Nitrogen Losses

```
F_loss = N / NCYCLE_TURNOVERTIME_LOSSES
```

### 3.7 Nitrogen Limitation Factor

Before `CO2_FERTILIZATION_YRSTART`:

```
NCYCLE_LIMIT_FACTOR = 1.0
```

After `CO2_FERTILIZATION_YRSTART`:

```
NCYCLE_LIMIT_FACTOR = (phi_N) / (phi_N_0)
```

Where phi_N and phi_N_0 are as defined in section 3.4.

### 3.8 Application to NPP

In MAGICC7.f90 (line 7296):

```
CO2_CURRENT_NPP = NCYCLE_LIMIT_FACTOR * CO2_CURRENT_NPP
```

### 3.9 Numerical Solution

The module uses a **bisection root-finding method** to solve for N where dN/dt = 0:

- Search interval: [-100, 1000] Pg N
- Tolerance: 1e-5 Pg N
- Max iterations: 200

## 4. State Variables

| Variable | Fortran Name | Symbol | Units | Description | Initial Value |
|----------|--------------|--------|-------|-------------|---------------|
| Nitrogen Pool | `NCYCLE_NPOOL(t)` | N | Pg N | Plant-available soil nitrogen pool | Solved from balance |
| Pre-industrial N Pool | `NCYCLE_NPOOL_0` | N_0 | Pg N | Reference nitrogen pool size | Set at CO2_FERTILIZATION_YRSTART |
| Pre-industrial NPP | `CO2_CURRENT_NPP_0` | NPP_0 | Pg C yr^-1 | Reference NPP | Set at CO2_FERTILIZATION_YRSTART |
| N Limitation Factor | `NCYCLE_LIMIT_FACTOR(t)` | phi | dimensionless | NPP scaling factor | 1.0 |
| Plant Uptake Flux | `NCYCLE_FN_PLANTUPTAKE(t)` | F_up | Pg N yr^-1 | N uptake by plants | 0.0 |
| Recycling Flux | `NCYCLE_FN_RECYCLING(t)` | F_recyc | Pg N yr^-1 | N recycling from decomposition | 0.0 |
| Loss Flux | `NCYCLE_FN_LOSSES(t)` | F_loss | Pg N yr^-1 | N losses (leaching, gaseous) | 0.0 |
| Fixation Flux | `NCYCLE_FN_FIXATION(t)` | F_fix | Pg N yr^-1 | Biological N fixation | 0.0 |
| Deposition Flux | `NCYCLE_FN_DEPOSITION(t)` | F_dep | Pg N yr^-1 | Atmospheric N deposition | 0.0 |

## 5. Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Apply N Cycle | `NCYCLE_APPLY` | flag | 0 | 0, 1 | Enable nitrogen limitation (0=off, 1=on) |
| Apply C+N Initial Values | `NCYCLE_APPLY_CN_INITIAL_VALUES` | flag | 0 | 0, 1 | Use separate initial pool values for C+N runs |
| Prognostic N Fixation | `NCYCLE_APPLY_PROGNOSTIC_NFIX` | flag | 0 | 0, 1 | Calculate N fixation internally (0=external, 1=prognostic) |
| N Dep from Emissions | `NCYCLE_APPLY_NDEP_FROM_EMIS` | flag | 0 | 0, 1 | Calculate N deposition from NOx/NH3 emissions |
| Sensitivity Factor A | `NCYCLE_NPP2NAVAIL_FACTOR_A` | dimensionless | 2.0 | >0 | Non-linear NPP-N sensitivity parameter |
| Sensitivity Factor C | `NCYCLE_NPP2NAVAIL_FACTOR_C` | dimensionless | 2.0 | >0 | Non-linear NPP-N sensitivity exponent |
| Recycling-to-CN Factor | `NCYCLE_RECYC2CNRATIO_FACTOR` | dimensionless | (unset) | >0 | Recycling adjustment factor |
| Recycling N:C Ratio | `NCYCLE_NCRATIO_RECYC` | mol N/mol C | 0.013 | >0 | N:C ratio for recycling flux |
| Uptake N:C Ratio | `NCYCLE_NCRATIO_UPTAKE` | mol N/mol C | 0.013 | >0 | N:C ratio for plant uptake |
| Uptake Supply Parameter | `NCYCLE_NCRATIO_UPTAKE_SUPPLY` | Pg N | 0.05 | >0 | Michaelis-Menten half-saturation for N supply |
| Loss Turnover Time | `NCYCLE_TURNOVERTIME_LOSSES` | years | 2.0 | >0 | Residence time for N losses |
| Fixation Vmax | `NCYCLE_NFIX_VMAX` | Pg N yr^-1 | NOT SET | >0 | Maximum N fixation rate |
| Fixation Km | `NCYCLE_NFIX_KM` | Pg C yr^-1 | NOT SET | >0 | Half-saturation for N fixation |
| Land Fraction N Dep | `NCYCLE_FRAC_NDEP_LAND` | fraction | 0.7 | 0-1 | Fraction of emitted N deposited on land |
| Initial Plant Pool | `NCYCLE_PLANTPOOL_INITIAL` | Pg C | 884.8584 | >0 | Initial plant carbon pool (C+N mode) |
| Initial Detritus Pool | `NCYCLE_DETRITUSPOOL_INITIAL` | Pg C | 92.7738 | >0 | Initial detritus carbon pool (C+N mode) |
| Initial Soil Pool | `NCYCLE_SOILPOOL_INITIAL` | Pg C | 1681.525 | >0 | Initial soil carbon pool (C+N mode) |
| Initial NPP | `NCYCLE_NPP_INITIAL` | Pg C yr^-1 | 66.2716 | >0 | Initial NPP (C+N mode) |

### Input File Parameters

| Parameter | Fortran Name | Default File |
|-----------|--------------|--------------|
| N Fixation Input | `FILE_NFIX_IN` | NCYCLE_NFIX3.IN |
| N Deposition Input | `FILE_NDEP_IN` | NCYCLE_NDEP_HISTSSP_CONSTANT.IN |
| N Recycling Input | `FILE_NRECYC_IN` | NCYCLE_NRECYC.IN |
| N Loss Input | `FILE_NLOSS_IN` | NCYCLE_NLOSS.IN |

## 6. Inputs (per timestep)

| Variable | Units | Source Module | Required? | Fortran Variable |
|----------|-------|---------------|-----------|------------------|
| Current NPP | Pg C yr^-1 | Terrestrial Carbon | Yes | `CO2_CURRENT_NPP(t)` |
| Total Respiration | Pg C yr^-1 | Terrestrial Carbon | Yes | `CO2_TOTALRESPIRATION(t-1)` |
| CO2 Fertilization Factor | dimensionless | Terrestrial Carbon | Yes | `CO2_EFF_FERTILIZATION_FACTOR(t)` |
| NPP Temperature Feedback | dimensionless | Terrestrial Carbon | Yes | `CO2_EFF_NPP_TEMPFEEDBACK(t)` |
| N Fixation (external) | Pg N yr^-1 | Input File | Conditional | `DAT_NCYCLE_NFIX` |
| N Deposition (external) | Pg N yr^-1 | Input File | Conditional | `DAT_NCYCLE_NDEP` |
| NOx Industrial Emissions | kt N yr^-1 | Emissions Handler | Conditional | `DAT_NOXI_EMIS` |
| NOx Biomass Emissions | kt N yr^-1 | Emissions Handler | Conditional | `DAT_NOXB_EMIS` |
| NH3 Industrial Emissions | kt N yr^-1 | Emissions Handler | Conditional | `DAT_NH3I_EMIS` |
| NH3 Biomass Emissions | kt N yr^-1 | Emissions Handler | Should be, bug | `DAT_NH3B_EMIS` |
| Current Year | year | Years Module | Yes | `ALLYEARS(CURRENT_YEAR_IDX)` |
| CO2 Fertilization Start Year | year | Config | Yes | `CO2_FERTILIZATION_YRSTART` |

## 7. Outputs (per timestep)

| Variable | Units | Destination Module(s) | Fortran Variable |
|----------|-------|----------------------|------------------|
| N Limitation Factor | dimensionless | Terrestrial Carbon (NPP modifier) | `NCYCLE_LIMIT_FACTOR(t)` |
| N Pool Size | Pg N | Output/Diagnostics | `NCYCLE_NPOOL(t)` |
| Plant Uptake Flux | Pg N yr^-1 | Output/Diagnostics | `NCYCLE_FN_PLANTUPTAKE(t)` |
| Recycling Flux | Pg N yr^-1 | Output/Diagnostics | `NCYCLE_FN_RECYCLING(t)` |
| Loss Flux | Pg N yr^-1 | Output/Diagnostics | `NCYCLE_FN_LOSSES(t)` |
| Fixation Flux | Pg N yr^-1 | Output/Diagnostics | `NCYCLE_FN_FIXATION(t)` |
| Deposition Flux | Pg N yr^-1 | Output/Diagnostics | `NCYCLE_FN_DEPOSITION(t)` |

All outputs are written to `CARBONCYCLE.OUT` and `CARBONCYCLE.BINOUT` files.

## 8. Algorithm (Pseudocode)

```
SUBROUTINE nitrogen_allocate():
    ALLOCATE NCYCLE_LIMIT_FACTOR[NYEARS]
    ALLOCATE NCYCLE_NPOOL[NYEARS]
    ALLOCATE NCYCLE_FN_PLANTUPTAKE[NYEARS]
    ALLOCATE NCYCLE_FN_RECYCLING[NYEARS]
    ALLOCATE NCYCLE_FN_LOSSES[NYEARS]
    ALLOCATE NCYCLE_FN_FIXATION[NYEARS]
    ALLOCATE NCYCLE_FN_DEPOSITION[NYEARS]

    IF NCYCLE_APPLY == 1:
        LOG WARNING: "Careful using the nitrogen cycle, it's not finished yet"

SUBROUTINE nitrogen_init():
    SET all arrays to initial values (1.0 for limit factor, 0.0 for fluxes)

    IF NCYCLE_APPLY == 1 AND NCYCLE_APPLY_CN_INITIAL_VALUES == 1:
        // Override carbon cycle initial values with C+N specific values
        CO2_NPP_INITIAL = NCYCLE_NPP_INITIAL
        CO2_PLANTPOOL_INITIAL = NCYCLE_PLANTPOOL_INITIAL
        CO2_DETRITUSPOOL_INITIAL = NCYCLE_DETRITUSPOOL_INITIAL
        CO2_SOILPOOL_INITIAL = NCYCLE_SOILPOOL_INITIAL

SUBROUTINE N_CALC_LIMITATION_FACTOR():
    // Called from MAGICC7.f90 carbon cycle section

    IF current_year < CO2_FERTILIZATION_YRSTART:
        // PRE-INDUSTRIAL SPINUP PHASE
        // Find N pool that gives dN/dt = 0 using simplified balance equation
        N = BISECTION(NITROGEN_BALANCE_INITIAL, -100, 1000, 1e-5)
        NCYCLE_NPOOL[t] = N
        NCYCLE_LIMIT_FACTOR[t] = 1.0  // No limitation before fertilization starts

        // Store reference values for later use
        NCYCLE_NPOOL_0 = N
        CO2_CURRENT_NPP_0 = CO2_CURRENT_NPP[t]

        // Calculate diagnostic fluxes
        RH = (t == 1) ? CO2_CURRENT_NPP[t] : CO2_TOTALRESPIRATION[t-1]
        Calculate NCYCLE_FN_PLANTUPTAKE, NCYCLE_FN_RECYCLING, NCYCLE_FN_LOSSES
        Calculate NCYCLE_FN_FIXATION (external or prognostic)
        Calculate NCYCLE_FN_DEPOSITION (external or from emissions)

    ELSE:
        // ACTIVE NITROGEN LIMITATION PHASE
        // Find N pool that gives dN/dt = 0 using full balance equation
        N = BISECTION(NITROGEN_BALANCE, -100, 1000, 1e-5)
        NCYCLE_NPOOL[t] = N

        // Calculate limitation factor
        phi_N = (1 + A) / (A + exp(-C * (N^2/(N+S) - k_up*NPP)))
        phi_N_0 = (1 + A) / (A + exp(-C * (N_0^2/(N_0+S) - k_up*NPP_0)))
        NCYCLE_LIMIT_FACTOR[t] = phi_N / phi_N_0

        // Calculate diagnostic fluxes
        Calculate all nitrogen flux diagnostics

FUNCTION NITROGEN_BALANCE_INITIAL(N) -> dN/dt:
    // Simplified balance for pre-industrial period
    RH = (t == 1) ? NPP : TOTALRESPIRATION[t-1]

    // Deposition
    IF NCYCLE_APPLY_NDEP_FROM_EMIS == 0:
        F_dep = DAT_NCYCLE_NDEP[t]
    ELSE:
        F_dep = FRAC_LAND * 1e-3 * (NOx_i + NOx_b + NH3_i + NH3_i)  // BUG: NH3_i twice

    // Recycling
    adj_fert = (1 + RECYC2CNRATIO_FACTOR) / (RECYC2CNRATIO_FACTOR + CO2_EFF_FERTILIZATION_FACTOR)
    adj_temp = (1 + RECYC2CNRATIO_FACTOR) / (RECYC2CNRATIO_FACTOR + CO2_EFF_NPP_TEMPFEEDBACK)
    F_recyc = NCRATIO_RECYC * RH * adj_fert * adj_temp

    // Fixation
    IF NCYCLE_APPLY_PROGNOSTIC_NFIX == 0:
        F_fix = DAT_NCYCLE_NFIX[t]
    ELSE:
        F_fix = 1e-3 * (NFIX_VMAX * NPP) / (NFIX_KM + NPP)

    // Uptake (no limitation in initial period)
    F_up = NCRATIO_UPTAKE * NPP

    // Losses
    F_loss = N / TURNOVERTIME_LOSSES

    RETURN (F_dep + F_recyc + F_fix) - (F_up + F_loss)

FUNCTION NITROGEN_BALANCE(N) -> dN/dt:
    // Full balance with N-availability feedbacks
    RH = (t == 1) ? NPP : TOTALRESPIRATION[t-1]

    // Deposition (same as initial)
    Calculate F_dep

    // Recycling (same as initial)
    Calculate F_recyc

    // N-availability factor
    phi = exp(-C * (N^2/(N+S) - k_up*NPP))
    phi_0 = exp(-C * (N_0^2/(N_0+S) - k_up*NPP_0))
    ratio = ((1+A)/(A+phi)) / ((1+A)/(A+phi_0))

    // Fixation with N-availability feedback
    IF NCYCLE_APPLY_PROGNOSTIC_NFIX == 0:
        F_fix = DAT_NCYCLE_NFIX[t]
    ELSE:
        NPP_eff = NPP * ratio
        F_fix = 1e-3 * (NFIX_VMAX * NPP_eff) / (NFIX_KM + NPP_eff)

    // Uptake with N-availability feedback
    F_up = NCRATIO_UPTAKE * NPP * ratio

    // Losses
    F_loss = N / TURNOVERTIME_LOSSES

    RETURN (F_dep + F_recyc + F_fix) - (F_up + F_loss)

SUBROUTINE BISECTION(F, x1, x2, eps) -> root:
    // Standard bisection root finding
    IF F(x1) * F(x2) > 0: RETURN failure
    a = x1; b = x2
    FOR i = 1 TO 200:
        c = (a + b) / 2
        IF F(a) * F(c) <= 0:
            b = c
        ELSE:
            a = c
        IF |b - a| <= eps: EXIT
    RETURN (a + b) / 2
```

## 9. Integration with Other Modules

### 9.1 Coupling with Terrestrial Carbon Cycle

The nitrogen limitation module is tightly coupled with the terrestrial carbon cycle:

**Inputs from Carbon Cycle** (defined in `mod_carbon_cycle`):

- `CO2_CURRENT_NPP` - Current net primary production
- `CO2_TOTALRESPIRATION` - Heterotrophic respiration (proxy for RH)
- `CO2_EFF_FERTILIZATION_FACTOR` - CO2 fertilization effect
- `CO2_EFF_NPP_TEMPFEEDBACK` - Temperature feedback on NPP

**Outputs to Carbon Cycle**:

- `NCYCLE_LIMIT_FACTOR` is multiplied by `CO2_CURRENT_NPP` in MAGICC7.f90 line 7296

**Timing**: The N_CALC_LIMITATION_FACTOR is called AFTER the carbon cycle calculates NPP but BEFORE the NPP value is finalized, allowing the limitation factor to reduce the effective NPP.

**Initial Value Override**: When `NCYCLE_APPLY == 1` and `NCYCLE_APPLY_CN_INITIAL_VALUES == 1`, the module can override the standard carbon cycle initial pool sizes and NPP with C+N specific values, enabling consistent calibration of the coupled C-N system.

### 9.2 Coupling with Emissions Handler

When `NCYCLE_APPLY_NDEP_FROM_EMIS == 1`:

- Uses NOx emissions (`DAT_NOXI_EMIS`, `DAT_NOXB_EMIS`)
- Uses NH3 emissions (`DAT_NH3I_EMIS`) - note: NH3B is missing due to bug

The emissions are converted to nitrogen deposition using a land fraction parameter (`NCYCLE_FRAC_NDEP_LAND = 0.7` by default).

### 9.3 Module Lifecycle

```
magicc_init_run():
    ...
    nitrogen_allocate()    // Allocate arrays
    nitrogen_init()        // Initialize values
    ...

carbon_cycle_timestep():
    ...
    IF NCYCLE_APPLY == 1:
        IF year > CO2_FERTILIZATION_YRSTART:
            N_CALC_LIMITATION_FACTOR()
            CO2_CURRENT_NPP *= NCYCLE_LIMIT_FACTOR  // Apply limitation
        ELSE:
            N_CALC_LIMITATION_FACTOR()  // Track but don't apply
    ...

magicc_cleanup():
    ...
    nitrogen_cleanup()     // Deallocate arrays
    ...
```

## 10. Honest Assessment / Red Flags

### 10.1 Critical Bugs

1. **NH3 Emissions Bug** (Lines 106-107, 156-157, 326-327, 388-389):

   ```fortran
   + DAT_NH3I_EMIS % DATGLOBE(CURRENT_YEAR_IDX) &
   + DAT_NH3I_EMIS % DATGLOBE(CURRENT_YEAR_IDX))  ! SHOULD BE NH3B_EMIS
   ```

   This doubles industrial NH3 and ignores biomass burning NH3 entirely.

### 10.2 Missing Initialization

1. **Uninitialized Parameters**: `NCYCLE_NFIX_VMAX` and `NCYCLE_NFIX_KM` have no defaults in `MAGCFG_DEFAULTALL.CFG`. Using `NCYCLE_APPLY_PROGNOSTIC_NFIX == 1` will use uninitialized (garbage) values.

2. **Uninitialized NCYCLE_RECYC2CNRATIO_FACTOR**: This parameter is used in calculations but has no visible default value in the CFG file, though it may be set elsewhere.

### 10.3 Unused Code

1. **Unused Input Files**: `FILE_NRECYC_IN` and `FILE_NLOSS_IN` are read but `DAT_NCYCLE_NRECYC` and `DAT_NCYCLE_NLOSS` are never used in any calculation. This suggests either:
   - Incomplete implementation
   - Dead code that should be removed
   - A planned feature that was never completed

### 10.4 Hardcoded Values

1. **Magic Number 1.0D-03**: Unit conversion factor appears repeatedly without documentation. Appears to be kt -> Pg conversion for emissions.

2. **Bisection Search Bounds**: The search range `[-100, 1000]` for nitrogen pool is hardcoded with no scientific justification. A negative nitrogen pool is physically meaningless.

3. **Singularity Check**: The bisection method checks `IF (ABS(F(ROOT)) < 1.0)` to distinguish roots from singularities. The threshold of 1.0 is arbitrary.

### 10.5 Numerical Concerns

1. **Steady-State Assumption**: The module assumes dN/dt = 0 every timestep, which is a strong assumption. The nitrogen pool cannot accumulate or draw down over time, which may not be realistic for transient scenarios.

2. **No Convergence Checking**: The bisection method always returns a result even if the initial condition `F(x1)*F(x2) > 0` fails. The code sets `FLAG = 0` but this flag is never checked by the caller.

3. **Potential Division by Zero**: Several expressions involve divisions that could fail:
    - `N + NCYCLE_NCRATIO_UPTAKE_SUPPLY` (safe if S > 0)
    - `NCYCLE_NFIX_KM + NPP*ratio` (could be near zero)
    - `NCYCLE_TURNOVERTIME_LOSSES` (safe if > 0)

### 10.6 Documentation Issues

1. **Inconsistent Units**: The code mixes Pg and kt units without clear documentation. The 1.0D-03 factor converts kt N to Pg N (since 1 Pg = 10^12 kg = 10^9 Mg = 10^6 kt).

2. **No Physical Bounds Checking**: There is no validation that:
    - Nitrogen pool remains positive
    - Limitation factor stays in [0, 1]
    - Fluxes remain physically reasonable

### 10.7 Code Quality

1. **Code Duplication**: The NITROGEN_BALANCE and NITROGEN_BALANCE_INITIAL functions share ~70% identical code. Should be refactored.

2. **Flux Calculation Duplication**: Flux calculations in N_CALC_LIMITATION_FACTOR duplicate what's computed in the balance functions.

3. **Missing Comments**: Despite the mathematical complexity, there are minimal inline comments explaining the physical meaning of equations.

### 10.8 Legacy Issues

1. **Orphaned Tuning File**: `MAGTUNE_1PCTCO2_CN.CFG` uses parameter names that don't exist in the codebase, suggesting it's obsolete or the module was refactored without updating calibration files.

## 11. Fortran Code References

| Description | File | Lines |
|-------------|------|-------|
| Module definition | nitrogen_limitation.f90 | 1-396 |
| Parameter declarations | nitrogen_limitation.f90 | 5-25 |
| Warning about unfinished module | nitrogen_limitation.f90 | 38 |
| Initialization subroutine | nitrogen_limitation.f90 | 58-78 |
| NITROGEN_BALANCE_INITIAL function | nitrogen_limitation.f90 | 80-130 |
| NH3 bug (first occurrence) | nitrogen_limitation.f90 | 106-107 |
| NITROGEN_BALANCE function | nitrogen_limitation.f90 | 132-193 |
| NH3 bug (second occurrence) | nitrogen_limitation.f90 | 156-157 |
| Phi calculation (N availability factor) | nitrogen_limitation.f90 | 165-170 |
| Bisection root finder | nitrogen_limitation.f90 | 195-251 |
| N_CALC_LIMITATION_FACTOR subroutine | nitrogen_limitation.f90 | 259-393 |
| Pre-industrial phase calculations | nitrogen_limitation.f90 | 273-328 |
| Active phase calculations | nitrogen_limitation.f90 | 330-391 |
| Limitation factor calculation | nitrogen_limitation.f90 | 336-347 |
| Module use in MAGICC7.f90 | MAGICC7.f90 | 151, 7061, 9598, 12025 |
| Nitrogen input file reading | MAGICC7.f90 | 496-507 |
| N_CALC_LIMITATION_FACTOR call | MAGICC7.f90 | 7293-7300 |
| NPP modification by limit factor | MAGICC7.f90 | 7296 |
| Output writing (CARBONCYCLE.OUT) | MAGICC7.f90 | 10528-10612 |
| Cleanup call | MAGICC7.f90 | 12179 |
| Parameter namelist declaration | allcfgs.f90 | 291-301 |
| Datastore declarations | datastore.f90 | 332 |
| Default parameter values | MAGCFG_DEFAULTALL.CFG | 316-330 |
| Default input files | MAGCFG_DEFAULTALL.CFG | 201-211 |
