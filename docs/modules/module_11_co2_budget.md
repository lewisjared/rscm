# Module 11: CO2 Budget Integrator

## 1. Scientific Purpose

The CO2 Budget Integrator closes the global carbon budget by tracking the balance between CO2 sources (fossil fuel emissions, land-use change) and sinks (terrestrial biosphere uptake, ocean uptake). It updates atmospheric CO2 concentrations each timestep based on this mass balance, converting between mass units (GtC) and concentration units (ppm).

**Critical caveat**: This is NOT a cleanly separated module in the MAGICC codebase. The CO2 budget integration is distributed across multiple locations in `MAGICC7.f90`, with tight coupling between the terrestrial carbon cycle (`TERRCARBON2` subroutine), the ocean carbon cycle (run on a sub-annual timestep within the main loop), and the atmospheric CO2 update logic. The "module" described here is a conceptual reconstruction of what would need to be implemented, extracted from the embedded code.

## 2. Mathematical Formulation

### 2.1 Core Mass Balance Equation

In emissions-driven mode, atmospheric CO2 is updated each timestep via:

```
dC_atm/dt = E_fossil + E_landuse + E_permafrost + E_CH4ox - F_ocean - F_land
```

Where:

- `C_atm` = Atmospheric CO2 concentration (ppm)
- `E_fossil` = Fossil fuel and industrial CO2 emissions (GtC/yr)
- `E_landuse` = Land-use change emissions (GtC/yr), though these are handled implicitly through terrestrial pool changes
- `E_permafrost` = Permafrost CO2 emissions (GtC/yr), if enabled
- `E_CH4ox` = CO2 from methane oxidation (GtC/yr), if enabled
- `F_ocean` = Air-to-ocean carbon flux (GtC/yr or ppm/yr)
- `F_land` = Air-to-land carbon flux (GtC/yr)

### 2.2 Unit Conversion

The conversion between atmospheric CO2 mass (GtC) and concentration (ppm) uses:

```
GTC_PER_PPM = 2.123 GtC/ppm
```

This is hardcoded in `carboncycle_init()` at line 88 of `carbon_cycle.f90`:

```fortran
GTC_PER_PPM = 2.123D0  ! THE GTC_PER_PPM TO CONVERT ATMOSPHERIC PPM CONCENTRATION UNITS INTO GTC ABUNDANCE
```

To convert:

- Concentration to mass: `C_mass (GtC) = C_conc (ppm) * 2.123`
- Mass to concentration: `C_conc (ppm) = C_mass (GtC) / 2.123`

### 2.3 Sub-annual Integration (Ocean Carbon Cycle)

The ocean carbon cycle is integrated on a **monthly** timestep within each annual step (lines 4849-4952 of MAGICC7.f90):

```
for month = 1 to 12:
    flux_air2ocean[step] = k * (C_atm - pCO2_ocean)

    if emissions_driven:
        C_atm[next_step] = C_atm[current] + E_total/12/GTC_PER_PPM - flux_air2ocean/12
    else:
        C_atm[next_step] = interpolate(prescribed_concentrations)

    pCO2_ocean = update_ocean_surface_pCO2(flux_history, delta_SST)
```

### 2.4 Stability Constraint

An ad-hoc stability criterion limits the change in ocean flux between successive monthly steps (lines 4887-4898):

```
OCEANCC_STABILITY_LIMIT_DIFFLUX = 0.04 ppm/yr

if (flux_change > 0.04):
    flux[step] = flux[step-1] + 0.04
elif (flux_change < -0.04):
    flux[step] = flux[step-1] - 0.04
```

**Issue**: This is a numerical hack, not physics. The code has TODO comments suggesting this should be replaced with a proper integrating routine.

## 3. State Variables

| Variable | Fortran Name | Symbol | Units | Description | Initial Value |
|----------|--------------|--------|-------|-------------|---------------|
| Atmospheric CO2 concentration | `DAT_CO2_CONC%DATGLOBE` | C_atm | ppm | Global mean atmospheric CO2 | From input file |
| Atmospheric CO2 pool | `CO2_ATMOS_POOL` | M_atm | GtC | Total atmospheric carbon mass | C_atm * 2.123 |
| Ocean surface pCO2 | `DAT_PCO2S_CONC%DATGLOBE` | pCO2_s | ppm | Ocean surface partial pressure of CO2 | Initialized to C_atm(1) |
| Air-to-ocean flux | `CO2_AIR2OCEAN_FLUX` | F_ocean | GtC/yr | Carbon flux from atmosphere to ocean | 0.0 |
| Air-to-land flux | `CO2_AIR2LAND_FLUX` | F_land | GtC/yr | Carbon flux from atmosphere to terrestrial biosphere | Calculated from pool changes |
| Net ecosystem exchange | `CO2_NETECOEXCH_FLUX` | NEE | GtC/yr | Net ecosystem carbon exchange | Calculated |
| Cumulative inverse emissions | `CO2I_INVERSE_CUMEMIS` | E_cum_inv | GtC | Running total of diagnosed emissions | 0.0 |

## 4. Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| GtC per ppm conversion | `GTC_PER_PPM` | GtC/ppm | 2.123 | Fixed | Mass-to-concentration conversion factor |
| Concentration-to-emissions switch year | `CO2_SWITCHFROMCONC2EMIS_YEAR` | year | 2015 | STARTYEAR to 10000 | Year to switch from concentration-driven to emissions-driven |
| Ocean carbon cycle model | `OCEANCC_MODEL` | - | "3D-GFDL" | "BOXDIFF", "HILDA", "2D-BERN", "3D-GFDL" | Impulse response function model selection |
| Ocean gas exchange scaling | `OCEANCC_SCALE_GASXCHANGE` | - | 1.833 | >0 | Scaling factor for ocean gas exchange rate |
| Ocean IRF scaling | `OCEANCC_SCALE_IMPULSERESPONSE` | - | 0.949 | >0 | Scaling factor for ocean impulse response |
| Ocean temperature feedback | `OCEANCC_TEMPFEEDBACK` | 1/K | 0.0372 | - | Sensitivity of ocean pCO2 to SST change |
| Ocean RAD setting | `OCEANCC_RAD_SETTING` | - | 0 | 0 or 1 | If 1, ocean sees only pre-industrial CO2 |
| Pre-industrial CO2 | `CO2_PREINDCO2CONC` | ppm | 278.0 | >0 | Pre-industrial reference concentration |
| Pre-industrial CO2 apply flag | `CO2_PREINDCO2CONC_APPLY` | - | 0 | 0 or 1 | Whether to override file value |
| CO2 concentration cap | `CO2_CAPCONC_PPM` | ppm | 2000 | >0 | Maximum allowed CO2 concentration |
| CO2 cap apply flag | `CO2_CAPCONC_APPLY` | - | 0 | 0 or 1 | Whether to apply concentration cap |
| Zero emissions threshold | `CO2_ZEROEMIS_AFTER_PGC` | PgC | 1000 | >0 | Cumulative emissions threshold for zero-emissions mode |
| Zero emissions apply flag | `CO2_ZEROEMIS_AFTERXPGC_APPLY` | - | 0 | 0 or 1 | Enable zero-emissions mode after threshold |

## 5. Inputs (per timestep)

| Variable | Units | Source Module | Required? | Fortran Variable |
|----------|-------|---------------|-----------|------------------|
| Fossil CO2 emissions | GtC/yr | Emissions input | Yes | `DAT_CO2I_EMIS%DATGLOBE` |
| Fossil CO2 emissions (incl. CH4 ox) | GtC/yr | CH4 module | Conditional | `DAT_CO2I_INCLCH4OX_EMIS%DATGLOBE` |
| Land-use CO2 emissions | GtC/yr | Emissions input | Yes | `DAT_CO2B_EMIS%DATGLOBE` |
| Permafrost CO2 emissions | GtC/yr | Permafrost module | If PF_APPLY=1 | `DAT_CO2PF_EMIS%DATGLOBE` |
| Terrestrial carbon pool changes | GtC | TERRCARBON2 | Yes | `CO2_PLANT_POOL`, `CO2_DETRITUS_POOL`, `CO2_SOIL_POOL` |
| Combined terrestrial + fossil emissions | GtC/yr | TERRCARBON2 | Yes | `CO2_TERRBIO_AND_FOSSIL_EMIS` |
| Temperature feedback | K | Climate module | Yes | `CO2_FEED_DELTATEMP_LAND`, `CO2_FEED_DELTATEMP` |
| Prescribed CO2 concentrations | ppm | Input file | If conc-driven | `DAT_CO2_CONC%DATGLOBE` |

## 6. Outputs (per timestep)

| Variable | Units | Destination Module(s) | Fortran Variable |
|----------|-------|----------------------|------------------|
| Atmospheric CO2 concentration | ppm | Radiative forcing, Terrestrial CC, Ocean CC | `DAT_CO2_CONC%DATGLOBE` |
| Atmospheric CO2 pool | GtC | Output only | `CO2_ATMOS_POOL`, `DAT_CO2_ATMOS_POOL%DATGLOBE` |
| Ocean surface pCO2 | ppm | Ocean CC | `DAT_PCO2S_CONC%DATGLOBE` |
| Air-to-ocean flux | GtC/yr | Output, Inverse calculation | `CO2_AIR2OCEAN_FLUX` |
| Air-to-land flux | GtC/yr | Output, Inverse calculation | `CO2_AIR2LAND_FLUX`, `DAT_CO2_AIR2LAND_FLUX%DATGLOBE` |
| Net ecosystem exchange | GtC/yr | Output | `CO2_NETECOEXCH_FLUX` |
| Inverse CO2 emissions | GtC/yr | Output (diagnostic) | `CO2I_INVERSE_EMIS`, `DAT_CO2I_INVERSE_EMMS%DATGLOBE` |
| Cumulative inverse emissions | GtC | Output | `CO2I_INVERSE_CUMEMIS` |

## 7. Algorithm (Pseudocode)

```
FUNCTION integrate_co2_budget(year_index):

    # Step 1: Terrestrial Carbon Cycle (TERRCARBON2)
    # Updates plant, detritus, soil pools based on NPP, respiration, deforestation
    call TERRCARBON2(temperature_feedback)

    # Calculate air-to-land flux from pool changes
    air2land_flux = (plant_pool[next] + detritus_pool[next] + soil_pool[next])
                  - (plant_pool[current] + detritus_pool[current] + soil_pool[current])

    # Calculate combined emissions term (used for atmospheric update)
    # This is fossil emissions + change in terrestrial pools
    terrbio_and_fossil_emis = fossil_emis_incl_ch4ox - (delta_plant + delta_detritus + delta_soil)
    if permafrost_enabled:
        terrbio_and_fossil_emis += permafrost_co2_emis

    # Step 2: Ocean Carbon Cycle (monthly sub-stepping)
    c_atm_current = co2_conc[current]
    pco2s_current = ocean_pco2[current]

    for month = 1 to 12:
        step = (year_index - 1) * 12 + month

        # Calculate atmosphere-ocean flux (ppm/yr)
        # flux = k * (c_atm - pco2_ocean)
        if ocean_rad_mode:
            c_atm_for_flux = co2_conc[preindustrial]  # Fixed pre-industrial
        else:
            c_atm_for_flux = average(c_atm_current, c_atm_next)

        pco2_for_flux = average(pco2s_current, pco2s_next)

        air2ocean_flux_ppm[step] = 12 * ocean_calculator.calc_flux(c_atm_for_flux, pco2_for_flux)

        # Apply stability constraint (HACK - limits flux change)
        flux_change = air2ocean_flux_ppm[step] - air2ocean_flux_ppm[step-1]
        if abs(flux_change) > 0.04:
            air2ocean_flux_ppm[step] = air2ocean_flux_ppm[step-1] + sign(flux_change) * 0.04

        # Update ocean surface pCO2
        pco2s_next = ocean_calculator.calc_ospp(flux_history, delta_sst)

        # Step 3: Update atmospheric CO2
        if year >= switch_to_emissions_year:
            # Emissions-driven mode
            c_atm_next = c_atm_current
                       + terrbio_and_fossil_emis / 12 / GTC_PER_PPM  # Monthly emissions
                       - air2ocean_flux_ppm[step] / 12               # Monthly ocean uptake
        else:
            # Concentration-driven mode (interpolate prescribed values)
            c_atm_next = linear_interpolate(co2_conc[current], co2_conc[next], month/12)

        # Apply concentration cap if enabled
        if cap_enabled and c_atm_next > cap_value:
            c_atm_next = cap_value

        # Update for next iteration
        c_atm_current = c_atm_next
        pco2s_current = pco2s_next

    # Step 4: Store final annual values
    co2_conc[next] = c_atm_next
    ocean_pco2[next] = pco2s_next
    co2_atmos_pool[next] = c_atm_next * GTC_PER_PPM

    # Average ocean flux over the year for reporting
    air2ocean_flux_gtc = average(air2ocean_flux_ppm[year_steps]) * GTC_PER_PPM

    # Step 5: Calculate inverse emissions (diagnostic)
    # What emissions would be needed to explain concentration change?
    inverse_emis = air2ocean_flux + air2land_flux
                 + (co2_atmos_pool[next] - co2_atmos_pool[current])

    if permafrost_enabled:
        inverse_emis -= permafrost_co2_emis
    if ch4_oxidation_enabled:
        inverse_emis -= ch4_to_co2_emis

    cumulative_inverse_emis += inverse_emis

    return co2_conc[next]
```

## 8. Numerical Considerations

### 8.1 Integration Scheme

- **Terrestrial carbon pools**: Implicit/Crank-Nicolson style scheme using turnover times. The discretization uses the form:

  ```
  pool[next] = (pool[current] * (1 - 0.5/tau) + net_flux) / (1 + 0.5/tau)
  ```

  This is a trapezoidal rule approximation that provides stability.

- **Ocean carbon cycle**: Explicit forward Euler with monthly sub-stepping.

- **Atmospheric CO2**: Explicit forward Euler per monthly substep.

### 8.2 Timestep Constraints

- Annual timestep for terrestrial carbon cycle and overall budget
- Monthly (12 steps/year) for ocean carbon cycle
- The monthly sub-stepping exists due to stability concerns with ocean flux calculation
- TODO comments in code suggest this could be improved with proper adaptive integration

### 8.3 Order of Operations

1. `TERRCARBON2` calculates terrestrial pool changes and `CO2_TERRBIO_AND_FOSSIL_EMIS`
2. Ocean carbon cycle monthly loop updates `CO2_CONC` progressively
3. Air-to-land flux calculated retroactively from pool differences
4. Inverse emissions calculated from budget closure
5. Cumulative emissions updated

### 8.4 Known Numerical Issues

1. **Ad-hoc stability limit** (line 4887): Limits flux change to 0.04 ppm/yr per month - this is a hack
2. **Averaging business** (lines 4833-4842): Unclear why averages of current/next step values are used
3. **First year initialization** (lines 4963-4974): First year values are overwritten with second year values, unclear why

## 9. Issues and Concerns

### 9.1 This is NOT a Separate Module

**Critical Issue**: The CO2 budget integration is deeply embedded in `MAGICC7.f90`. The relevant code is scattered across:

- Lines ~4818-4976: Ocean carbon cycle integration and atmospheric CO2 update
- Lines ~5000-5140: Air-to-land flux calculation and inverse emissions
- Lines ~7054-7542: `TERRCARBON2` subroutine (terrestrial carbon cycle)

There is no clean interface or separation. A reimplementation would need to carefully extract this logic.

### 9.2 Tight Coupling

The budget integrator is tightly coupled to:

- Terrestrial carbon cycle (pool changes needed for air-to-land flux)
- Ocean carbon cycle (monthly sub-stepping interleaved with atmospheric update)
- Methane module (CH4 oxidation adds CO2)
- Permafrost module (optional permafrost CO2 emissions)
- Climate module (temperature feedbacks)

### 9.3 Hardcoded Values

| Value | Location | Issue |
|-------|----------|-------|
| `GTC_PER_PPM = 2.123` | carbon_cycle.f90:88 | Should be configurable? |
| `OCEANCC_STABILITY_LIMIT_DIFFLUX = 0.04` | MAGICC7.f90:4887 | Ad-hoc numerical hack |
| `stepsperyear = 12` | Implicit in loop | Hardcoded monthly sub-stepping |

### 9.4 Unclear Code

1. **Why average c_atm?** (lines 4833-4842, 4860-4867): The code uses averages of current and next step values for ocean flux calculation. The purpose is unclear and marked with TODO.

2. **First year overwrite** (lines 4963-4974): First year pCO2 and CO2 pool are overwritten with second year values with comment "TODO: check why we overwrite first year values"

3. **Inverse emissions inconsistency** (lines 5080-5081): Comment states "prescribing CO2 emissions of zero to the model won't make this come out as zero i.e. our inverse isn't fully consistent with our forward model"

### 9.5 Mode Switching Logic

The switch from concentration-driven to emissions-driven mode (controlled by `CO2_SWITCHFROMCONC2EMIS_YEAR`) is handled inline:

```fortran
if (allyears(current_year_idx) >= co2_switchfromconc2emis_year) then
    ! Emissions-driven: calculate new concentration
else
    ! Concentration-driven: interpolate prescribed values
end if
```

This is clean but embedded in the monthly sub-stepping loop.

### 9.6 Multiple Flux Tracking

The code tracks multiple related flux quantities that can be confusing:

- `CO2_AIR2OCEAN_FLUX`: Air-to-ocean flux in GtC/yr
- `CO2_AIR2OCNFLX_ALLT_PPMPERYR`: Monthly ocean flux history in ppm/yr
- `CO2_AIR2LAND_FLUX`: Net air-to-land flux from terrestrial pool changes
- `CO2_NETECOEXCH_FLUX`: Net ecosystem exchange (= -air2land - deforestation)
- `CO2_TERRBIO_AND_FOSSIL_EMIS`: Combined fossil + terrestrial source term

## 10. Test Cases

### Test 1: Mass Conservation (Closed System)

**Setup**: Zero emissions, zero land-use change, constant temperature
**Expected**: CO2 concentration should remain constant (equilibrium)
**Validation**: `CO2_CONC[t+1] - CO2_CONC[t] < 1e-10 ppm`

### Test 2: GtC-ppm Conversion

**Setup**: Add exactly 2.123 GtC of emissions in one year
**Expected**: Atmospheric CO2 should increase by ~1 ppm (minus sink uptake)
**Validation**: Check `CO2_ATMOS_POOL / CO2_CONC = 2.123` always

### Test 3: Mode Switching

**Setup**:

- Prescribe CO2 concentration = 400 ppm until 2020
- Switch to emissions-driven at 2020
- Provide emissions that would maintain 400 ppm
**Expected**: Smooth transition, no discontinuity at switch year
**Validation**: `|CO2_CONC[2020] - CO2_CONC[2019]| < 1 ppm`

### Test 4: Inverse Emissions Consistency

**Setup**: Run in concentration-driven mode with prescribed concentrations
**Expected**: Inverse emissions should match what emissions-driven mode would need
**Validation**: Run same scenario emissions-driven with diagnosed inverse emissions, get same concentrations

### Test 5: Ocean Uptake Response

**Setup**: Step increase in CO2 from 280 to 400 ppm
**Expected**:

- Immediate increase in air-to-ocean flux
- Gradual decay of flux as ocean equilibrates
**Validation**: Check impulse response matches selected ocean model

### Test 6: Zero Emissions Mode

**Setup**:

- Enable `CO2_ZEROEMIS_AFTERXPGC_APPLY = 1`
- Set `CO2_ZEROEMIS_AFTER_PGC = 500`
- Run with cumulative emissions crossing 500 PgC
**Expected**: Emissions set to zero after threshold, concentrations peak and decline
**Validation**: `DAT_CO2I_EMIS%DATGLOBE[after_threshold:] = 0`

### Test 7: Concentration Cap

**Setup**: Enable `CO2_CAPCONC_APPLY = 1`, set `CO2_CAPCONC_PPM = 500`
**Expected**: CO2 concentration never exceeds 500 ppm
**Validation**: `max(CO2_CONC) <= 500`

## 11. Fortran Code References

### Primary Integration (MAGICC7.f90)

| Lines | Description |
|-------|-------------|
| 4818-4820 | Call to TERRCARBON2 |
| 4826-4831 | First year initialization |
| 4833-4842 | Initialize averaging variables |
| 4849-4952 | Monthly sub-stepping loop for ocean CC |
| 4869-4878 | Air-to-ocean flux calculation |
| 4880-4898 | Stability constraint (HACK) |
| 4901-4912 | Ocean surface pCO2 update |
| 4916-4931 | Atmospheric CO2 update (emissions vs concentration driven) |
| 4934-4938 | Concentration cap application |
| 4955-4956 | Store final CO2 concentration |
| 4970-4975 | Update atmospheric pool |
| 5006-5008 | Net human CO2 emissions |
| 5010-5022 | Air-to-land flux calculation |
| 5030-5036 | Net ecosystem exchange |
| 5082-5117 | Inverse emissions calculation |
| 5122-5139 | Cumulative inverse emissions |

### Terrestrial Carbon Cycle (MAGICC7.f90)

| Lines | Description |
|-------|-------------|
| 7054-7069 | TERRCARBON2 subroutine declaration |
| 7103-7105 | CO2 concentration extrapolation for feedbacks |
| 7141-7148 | Temperature feedback factors |
| 7155-7240 | CO2 fertilization calculation |
| 7267-7291 | NPP and respiration with feedbacks |
| 7311-7316 | NPP partitioning to pools |
| 7357-7362 | Plant pool update |
| 7407-7413 | Detritus pool update |
| 7457-7459 | Soil pool update |
| 7509-7522 | TERRBIO_AND_FOSSIL_EMIS calculation |

### Carbon Cycle Module (carbon_cycle.f90)

| Lines | Description |
|-------|-------------|
| 7-14 | Pool variable declarations |
| 16-24 | Flux variable declarations |
| 30-37 | Parameter declarations including GTC_PER_PPM |
| 65-139 | carboncycle_init() subroutine |
| 88 | GTC_PER_PPM = 2.123 hardcoded value |

### Ocean Carbon Cycle (carbon_cycle_ocean.f90)

| Lines | Description |
|-------|-------------|
| 51-70 | build() method |
| 82-209 | build_from_name_and_parameters() - model selection |
| 351-364 | calc_atm_to_ocn_c_flux() |
| 366-394 | calc_ospp() - ocean surface partial pressure |
| 396-597 | IRF functions for different ocean models |

### Configuration Parameters (allcfgs.f90)

| Lines | Description |
|-------|-------------|
| 36-54 | CO2 parameter declarations in namelist |
| 41 | CO2_SWITCHFROMCONC2EMIS_YEAR |
| 43-45 | OCEANCC_* parameters |

## 12. Recommendations for Reimplementation

1. **Create a dedicated CO2BudgetIntegrator class** with clean interfaces for:
   - `update_atmospheric_co2(emissions, land_uptake, ocean_uptake) -> new_concentration`
   - `calculate_inverse_emissions(concentration_change, land_uptake, ocean_uptake) -> implied_emissions`
   - `switch_mode(year, mode)` for concentration/emissions mode switching

2. **Make GTC_PER_PPM configurable** (even if rarely changed)

3. **Replace ad-hoc stability limit** with proper adaptive time-stepping or implicit solver

4. **Clean up averaging logic** - document or simplify the current/next averaging approach

5. **Separate concerns**:
   - Terrestrial carbon cycle should return air-to-land flux
   - Ocean carbon cycle should return air-to-ocean flux
   - Budget integrator should combine these with emissions

6. **Add proper error checking** for conservation (mass in = mass out + storage change)

7. **Consider making monthly sub-stepping configurable** rather than hardcoded to 12
