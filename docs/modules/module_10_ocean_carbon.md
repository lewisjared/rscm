# Module 10: Ocean Carbon Cycle

## 1. Scientific Purpose

The Ocean Carbon Cycle module simulates the uptake of CO2 by the global ocean through air-sea gas exchange. The ocean is the largest long-term sink for anthropogenic CO2, absorbing approximately 25-30% of annual emissions. This module is critical for determining atmospheric CO2 concentrations over multi-decadal to millennial timescales.

The module implements an **Impulse Response Function (IRF) approach** rather than a full ocean circulation model. This choice allows MAGICC to emulate the behavior of more complex 3D ocean models (like GFDL, HILDA, or the Bern 2.5D model) while maintaining computational efficiency suitable for large ensemble runs and scenario exploration. The IRF captures how the mixed layer "remembers" past carbon uptake events through convolution with the flux history.

Key physical processes represented:
1. **Air-sea gas exchange** driven by the CO2 partial pressure difference between atmosphere and surface ocean
2. **Ocean carbonate chemistry** (Revelle factor / buffering) that reduces uptake efficiency as dissolved inorganic carbon (DIC) accumulates
3. **Temperature sensitivity** of CO2 solubility (warmer water holds less CO2)
4. **Vertical mixing** implicit in the IRF, representing how absorbed carbon is transported to the deep ocean

## 2. Mathematical Formulation

### 2.1 Air-Sea Carbon Flux

The fundamental flux equation is:

$$F_{atm \rightarrow ocn} = k \cdot (pCO2_{atm} - pCO2_{ocn})$$

Where:
- $F_{atm \rightarrow ocn}$ = atmosphere to ocean carbon flux (ppm/month)
- $k$ = gas exchange coefficient (month$^{-1}$), scaled by `OCEANCC_SCALE_GASXCHANGE`
- $pCO2_{atm}$ = atmospheric CO2 partial pressure (ppm)
- $pCO2_{ocn}$ = ocean surface CO2 partial pressure (ppm)

The gas exchange coefficient is model-dependent and defined as:

$$k = \frac{OCEANCC\_SCALE\_GASXCHANGE}{\tau_{exchange} \cdot 12}$$

Where $\tau_{exchange}$ is the characteristic gas exchange timescale in years (converted to months).

### 2.2 Impulse Response Function (IRF)

The IRF represents how a pulse of carbon added to the mixed layer decays over time as it mixes into the deep ocean. For time $t$:

**Before switch time** (short timescales, $t < t_{switch}$):

For 3D-GFDL: Polynomial form
$$IRF(t) = \sum_{i=0}^{6} c_i \cdot t_{years}^i$$

For other models (HILDA, 2D-BERN, BOXDIFF): Exponential sum
$$IRF(t) = \sum_{i} a_i \cdot \exp(-t / \tau_i)$$

**After switch time** (long timescales, $t \geq t_{switch}$):

All models use exponential sum:
$$IRF(t) = \sum_{i} a_i \cdot \exp(-t / \tau_i)$$

The IRF is scaled by `OCEANCC_SCALE_IMPULSERESPONSE` using a nonlinear transformation:

$$IRF_{scaled} = \frac{IRF \cdot f}{IRF \cdot f + 1 - IRF}$$

where $f$ = scaling factor. This ensures $IRF_{scaled}$ remains bounded between 0 and 1.

### 2.3 Change in Dissolved Inorganic Carbon (Delta DIC)

The change in surface ocean DIC is computed via convolution of the flux history with the IRF:

$$\Delta DIC(t) = \frac{\mu}{h \cdot A} \int_0^t F_{atm \rightarrow ocn}(t') \cdot IRF(t - t') \, dt'$$

Where:
- $\mu$ = `OCEAN_MICROMOL_PER_PPM_M3_PER_KG` = unit conversion factor ($\approx 1.72 \times 10^{17}$ micromol ppm$^{-1}$ m$^3$ kg$^{-1}$)
- $h$ = mixed layer depth (m)
- $A$ = ocean surface area (m$^2$)

The integral is computed using a left-hand Riemann sum over the monthly timesteps.

### 2.4 Ocean Surface Partial Pressure (pCO2)

The ocean surface CO2 partial pressure is calculated following Joos et al. (2001), equations A24 and A25:

**Equation A24 - DIC effect on pCO2:**

$$\Delta pCO2_{DIC} = \sum_{i=1}^{5} (b_i + c_i \cdot T_0) \cdot g_i(\Delta DIC)$$

Where:
- $b_i$ = offset coefficients (`delta_ospp_offsets`)
- $c_i$ = temperature-dependent coefficients (`delta_ospp_coefficients`)
- $T_0$ = preindustrial sea surface temperature
- $g_i(\Delta DIC) = [\Delta DIC, (\Delta DIC)^2 \cdot 10^{-3}, -(\Delta DIC)^3 \cdot 10^{-5}, (\Delta DIC)^4 \cdot 10^{-7}, -(\Delta DIC)^5 \cdot 10^{-10}]$

**Equation A25 - Temperature effect on pCO2:**

$$pCO2_{ocn} = (pCO2_0 + \Delta pCO2_{DIC}) \cdot \exp(\alpha_T \cdot \Delta T_{SST})$$

Where:
- $pCO2_0$ = preindustrial ocean pCO2 (set equal to preindustrial atmospheric CO2)
- $\alpha_T$ = `OCEANCC_TEMPFEEDBACK` (default: 0.0423 K$^{-1}$ from Takahashi et al., ~4.23%/K)
- $\Delta T_{SST}$ = change in sea surface temperature from preindustrial

### 2.5 Revelle Factor (Implicit)

The polynomial expansion in equation A24 implicitly captures the Revelle factor (buffer factor), which quantifies how much surface ocean pCO2 changes for a given change in DIC. The Revelle factor is approximately:

$$R = \frac{\partial \ln(pCO2)}{\partial \ln(DIC)} \approx 10-15$$

This means ocean pCO2 increases ~10-15 times faster than DIC, which is why ocean uptake efficiency decreases as the ocean absorbs more carbon.

## 3. State Variables

| Variable | Fortran Name | Symbol | Units | Description | Initial Value |
|----------|--------------|--------|-------|-------------|---------------|
| Ocean pCO2 | `pco2s_currentstep`, `DAT_PCO2S_CONC%DATGLOBE` | $pCO2_{ocn}$ | ppm | Surface ocean CO2 partial pressure | `DAT_CO2_CONC%DATGLOBE(1)` (preindustrial atm CO2) |
| Air-sea flux history | `CO2_AIR2OCNFLX_ALLT_PPMPERYR` | $F(t)$ | ppm/yr | Full history of monthly fluxes | 0.0 |
| Delta sea surface temp | `co2_feed_deltatemp` | $\Delta T_{SST}$ | K | Temperature anomaly from feedback start | 0.0 |
| Atmospheric CO2 | `c_atm_currentstep` | $pCO2_{atm}$ | ppm | Current atmospheric CO2 | From `DAT_CO2_CONC` |
| IRF array | `delta_dioc_calculator%irf` | $IRF(t)$ | dimensionless | Pre-computed IRF at all timesteps | Model-dependent |

## 4. Parameters

### 4.1 User-Configurable Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Ocean model selection | `OCEANCC_MODEL` | string | "3D-GFDL" | "3D-GFDL", "2D-BERN", "HILDA", "BOXDIFF" | Selects which ocean model to emulate |
| Gas exchange scaling | `OCEANCC_SCALE_GASXCHANGE` | dimensionless | 1.833492 | 0.5-3.0 | Scales the gas exchange rate $k$ |
| IRF scaling | `OCEANCC_SCALE_IMPULSERESPONSE` | dimensionless | 0.9492864 | 0.5-1.5 | Scales the impulse response function |
| Temperature feedback | `OCEANCC_TEMPFEEDBACK` | K$^{-1}$ | 0.03717879 | 0.0-0.06 | Sensitivity of pCO2 to SST ($\alpha_T$) |
| Radiative setting | `OCEANCC_RAD_SETTING` | flag | 0 | 0 or 1 | If 1, ocean sees only preindustrial CO2 |

### 4.2 Model-Specific Hardcoded Parameters

These parameters are selected automatically based on `OCEANCC_MODEL`:

| Model | Switch Time (years) | Gas Exchange $\tau$ (years) | Mixed Layer $h$ (m) | Surface Area $A$ (m$^2$) | SST$_0$ (C) |
|-------|---------------------|----------------------------|---------------------|-------------------------|-------------|
| 3D-GFDL | 1.0 | 7.66 | 50.9 | 3.55e14 | 17.7 |
| 2D-BERN | 9.9 | 7.46 | 50.0 | 3.5375e14 | 18.2997 |
| HILDA | 2.0 | 9.06 | 75.0 | 3.62e14 | 18.1716 |
| BOXDIFF | 3.2 | 7.8 | 75.0 | 3.62e14 | 17.7 |

### 4.3 Hardcoded Polynomial/Exponential Coefficients

**pCO2-DIC relationship (all models):**

| Coefficient Set | Values |
|-----------------|--------|
| `delta_ospp_offsets` | [1.5568, 7.4706, 1.2748, 2.4491, 1.5468] |
| `delta_ospp_coefficients` | [-0.013993, -0.20207, -0.12015, -0.12639, -0.15326] |

**IRF coefficients are extensive (6-7 terms per model, 2 functions per model). See Section 11 for full listings.**

## 5. Inputs (per timestep)

| Variable | Units | Source Module/Data | Required? | Fortran Variable |
|----------|-------|-------------------|-----------|------------------|
| Atmospheric CO2 | ppm | Datastore | Yes | `DAT_CO2_CONC%DATGLOBE(current_year_idx)` |
| Previous ocean pCO2 | ppm | Previous timestep | Yes | `pco2s_currentstep` |
| Sea surface temperature delta | K | Climate module | Yes | `co2_feed_deltatemp` |
| Flux history | ppm/month | Accumulated | Yes | `CO2_AIR2OCNFLX_ALLT_PPMPERYR(1:currentstep)` |
| Current month index | - | Time counter | Yes | `currentstep_since_start` |

## 6. Outputs (per timestep)

| Variable | Units | Destination Module(s) | Fortran Variable |
|----------|-------|----------------------|------------------|
| Air-sea carbon flux | ppm/yr | Carbon budget, stored | `CO2_AIR2OCNFLX_ALLT_PPMPERYR(currentstep)` |
| Ocean pCO2 | ppm | Next timestep input | `pco2s_nextstep` -> `DAT_PCO2S_CONC%DATGLOBE` |
| Annual air-sea flux | GtC/yr | Carbon budget output | `CO2_AIR2OCEAN_FLUX(year)` (after unit conversion) |

## 7. Algorithm (Pseudocode)

### 7.1 Initialization (`oceancc_init`)

```
SUBROUTINE oceancc_init():
    # Set hardcoded pCO2-DIC relationship coefficients
    delta_ospp_offsets = [1.5568, 7.4706, 1.2748, 2.4491, 1.5468]
    delta_ospp_coefficients = [-0.013993, -0.20207, -0.12015, -0.12639, -0.15326]

    # Build the ocean carbon cycle calculator
    CALL carbon_cycle_ocean_calculator.build_from_name_and_parameters(
        oceancc_model = OCEANCC_MODEL,
        time = simulation_time_axis,
        scaling_factor_gas_exchange_rate = OCEANCC_SCALE_GASXCHANGE,
        ospp_preindustrial = DAT_CO2_CONC%DATGLOBE(1),  # Preindustrial atm CO2
        sensitivity_ospp_to_temperature = OCEANCC_TEMPFEEDBACK,
        delta_ospp_offsets = delta_ospp_offsets,
        delta_ospp_coefficients = delta_ospp_coefficients,
        scaling_factor_irf = OCEANCC_SCALE_IMPULSERESPONSE
    )
```

### 7.2 Model Selection and IRF Setup (`build_from_name_and_parameters`)

```
SUBROUTINE build_from_name_and_parameters(model_name, ...):
    SELECT CASE (UPPER(model_name))

    CASE ("3D-GFDL"):
        irf_switch_time = 1.0 * 12  # months
        irf_before_switch => polynomial_irf_3dgfdl
        irf_after_switch => exponential_irf_3dgfdl
        gas_exchange_rate = 1.0 / (7.66 * 12)  # per month
        mixed_layer_depth = 50.9
        ocean_surface_area = 3.55e14
        sea_surface_temp_preind = 17.7

    CASE ("2D-BERN"):
        irf_switch_time = 9.9 * 12
        irf_before_switch => exponential_irf_2dbern_early
        irf_after_switch => exponential_irf_2dbern_late
        gas_exchange_rate = 1.0 / (7.46 * 12)
        mixed_layer_depth = 50.0
        ocean_surface_area = 3.5375e14
        sea_surface_temp_preind = 18.2997

    CASE ("HILDA"):
        # ... similar pattern

    CASE ("BOXDIFF"):
        # ... similar pattern

    DEFAULT:
        FATAL_ERROR("Unrecognised oceancc_model: " + model_name)

    END SELECT

    # Pre-compute IRF at all time points
    FOR t = 0 to end_time:
        IF t < irf_switch_time:
            irf[t] = scale_irf(irf_before_switch(t), scaling_factor)
        ELSE:
            irf[t] = scale_factor_post_switch * irf_after_switch(t)
        END IF
    END FOR
```

### 7.3 Main Timestep (Monthly Loop in MAGICC7.f90)

```
# Run on monthly timestep for stability
FOR month = 1 to 12:
    currentstep = (year - 1) * 12 + month

    # === DETERMINE ATMOSPHERIC CO2 FOR OCEAN TO "SEE" ===
    IF OCEANCC_RAD_SETTING == 1:
        # Radiative-only mode: ocean sees preindustrial CO2
        c_atm_for_ocean = DAT_CO2_CONC%DATGLOBE(1)
    ELSE:
        # Normal mode: use time-averaged atmospheric CO2
        c_atm_for_ocean = (c_atm_prev + c_atm_current) / 2
    END IF

    # === CALCULATE AIR-SEA FLUX ===
    # F = k * (pCO2_atm - pCO2_ocn)
    flux_this_step = 12 * calculator.calc_atm_to_ocn_c_flux(
        c_atm = c_atm_for_ocean,
        c_ocn = (pco2s_prev + pco2s_current) / 2
    )
    # Store in ppm/yr units
    CO2_AIR2OCNFLX_ALLT_PPMPERYR[currentstep] = flux_this_step

    # === APPLY STABILITY CONSTRAINT ===
    # Ad-hoc limiter on flux change rate
    flux_change = flux_this_step - CO2_AIR2OCNFLX_ALLT_PPMPERYR[currentstep - 1]
    STABILITY_LIMIT = 0.04  # ppm/yr per timestep

    IF flux_change > STABILITY_LIMIT:
        CO2_AIR2OCNFLX_ALLT_PPMPERYR[currentstep] = prev_flux + STABILITY_LIMIT
    ELSE IF flux_change < -STABILITY_LIMIT:
        CO2_AIR2OCNFLX_ALLT_PPMPERYR[currentstep] = prev_flux - STABILITY_LIMIT
    END IF

    # === CALCULATE NEW OCEAN pCO2 ===
    # This involves:
    # 1. Convolving flux history with IRF to get delta_DIC
    # 2. Applying Joos A24 polynomial to get delta_pCO2 from delta_DIC
    # 3. Applying temperature correction (Joos A25)

    pco2s_next = calculator.calc_ospp(
        atmosphere_to_ocean_carbon_fluxes = flux_history[1:currentstep] / 12,  # Convert to ppm/month
        delta_sea_surface_temperature = co2_feed_deltatemp
    )

    # === UPDATE ATMOSPHERIC CO2 (if in emissions-driven mode) ===
    IF year >= CO2_SWITCHFROMCONC2EMIS_YEAR:
        c_atm_next = c_atm_current
                     + emissions_this_step / gtc_per_ppm / 12  # Add emissions
                     - flux_this_step / 12                      # Subtract ocean uptake
    ELSE:
        # Interpolate from prescribed concentrations
        c_atm_next = interpolate(CO2_conc_data, current_month)
    END IF

    # Update for next iteration
    c_atm_prev = c_atm_current
    c_atm_current = c_atm_next
    pco2s_prev = pco2s_current
    pco2s_current = pco2s_next

END FOR  # monthly loop
```

### 7.4 IRF Convolution for Delta DIC

```
FUNCTION calculate_delta_dic(flux_history):
    # Convolve flux history with reversed IRF
    # IRF is pre-computed on time bounds

    n = SIZE(flux_history)

    # Build reversed IRF for convolution
    # At time n, we need IRF(n-1), IRF(n-2), ..., IRF(0)
    irf_for_convolution = irf[n+1 : 2 : -1]  # Reverse order, skip endpoint

    # Element-wise multiply
    integrand = flux_history * irf_for_convolution

    # Integrate using left-hand Riemann sum
    integral = 0
    FOR i = 1 to n:
        integral = integral + integrand[i] * timestep
    END FOR

    # Convert to DIC units
    delta_dic = OCEAN_MICROMOL_PER_PPM_M3_PER_KG / (h * A) * integral

    RETURN delta_dic
```

### 7.5 Ocean pCO2 Calculation (Joos Equations)

```
FUNCTION calculate_ocean_pco2(delta_dic, delta_sst):
    # Joos et al. 2001 Equation A24: DIC effect
    dic_powers = [
        delta_dic,
        delta_dic^2 * 1e-3,
        -delta_dic^3 * 1e-5,
        delta_dic^4 * 1e-7,
        -delta_dic^5 * 1e-10
    ]

    coefficients = offsets + coeffs * T_preindustrial
    delta_pco2 = DOT_PRODUCT(coefficients, dic_powers)

    # Joos et al. 2001 Equation A25: Temperature effect
    pco2_ocean = (pco2_preindustrial + delta_pco2) * EXP(alpha_T * delta_sst)

    RETURN pco2_ocean
```

## 8. Numerical Considerations

### 8.1 Sub-Annual Stepping

The ocean carbon cycle runs on a **monthly timestep** (12 steps per year), even though the main MAGICC loop is annual. This is stated to be for "stability" reasons:

```fortran
! We run on a monthly timestep because of issues with the stability of the ocean carbon cycle
! TODO: investigate whether this is actually needed (Zeb is extremely skeptical, especially
! if a smart integrating routine is used...)
! (#209)
```

This design choice:
- Increases computational cost by 12x for this module
- Is likely unnecessary with proper numerical integration
- Creates complexity in unit handling (flux is internally in ppm/month, but stored as ppm/yr)

### 8.2 Stability Constraint

An ad-hoc stability limiter is applied:

```fortran
OCEANCC_STABILITY_LIMIT_DIFFLUX = 0.04D0
IF (CHANGE_FLUX_AIR2OCN > OCEANCC_STABILITY_LIMIT_DIFFLUX) THEN
    ! Clamp to previous + limit
```

This limits the air-sea flux to change by at most 0.04 ppm/yr per monthly timestep. Comments indicate this is a stopgap:

```fortran
! TODO: remove this (#208)
! Ad-hoc stability criterion is not the smart way to get stability
```

### 8.3 Averaging for Flux Calculation

The code uses two-timestep averaging for both atmospheric and oceanic pCO2:

```fortran
c_atm_for_oceancc = c_atm_average_prev_two
c_ocn_for_oceancc = pco2s_average_prev_two
```

This is noted as unnecessarily complex:

```fortran
! TODO: remove this average business (unnecessarily complex way to solve) (#208)
```

### 8.4 IRF Truncation

The IRF is pre-computed for the entire simulation length. The convolution integral grows with each timestep, requiring O(N) operations at timestep N, giving O(N^2) total complexity. For multi-century runs with monthly steps, this could become expensive.

### 8.5 Unit Conversion Constant

The key unit conversion is:

```fortran
OCEAN_MICROMOL_PER_PPM_M3_PER_KG = 1.0D6 / OCEAN_PPM_PER_MOL / OCEANCC_DENSITY_KG_PER_M3
```

Where:
- `OCEAN_PPM_PER_MOL` = 5.65770e-15 ppm/mol
- `OCEANCC_DENSITY_KG_PER_M3` = 1026.5 kg/m^3

This evaluates to approximately 1.72e17 micromol ppm^-1 m^3 kg^-1.

## 9. Issues and Concerns

### 9.1 Modularity Assessment

**Claim:** The ocean carbon cycle is a "modular" component.

**Reality:** Partially true. The code is well-encapsulated in the `CarbonCycleOceanCalculatorMAGICCConstantStepTimeAxis` class, BUT:

1. **Hardcoded coefficients:** The pCO2-DIC polynomial coefficients are hardcoded in `oceancc_init` (not exposed as parameters):
   ```fortran
   ! can hard-code for now
   ! TODO: make model/config parameters
   delta_ocean_surface_partial_pressure_offsets = (/ &
       1.5568, 7.4706, 1.2748, 2.4491, 1.5468 &
   /)
   ```

2. **Model selection via string:** Using a string parameter to select fundamentally different models is noted as problematic:
   ```fortran
   ! Zeb's opinion is that this is an abuse of the concept of a model and
   ! this sort of wrapping should be done at a different (much higher)
   ! level via some sort of builder plus director pattern
   ```

3. **Monthly loop in main driver:** The monthly sub-stepping loop lives in MAGICC7.f90, not in the ocean module:
   ```fortran
   do month = 1, stepsperyear
       ! ... ocean carbon calculations ...
   end do
   ```

4. **State split across modules:** The flux history array `CO2_AIR2OCNFLX_ALLT_PPMPERYR` is in `mod_ocean_carbon_cycle`, but the monthly loop and stability logic are in MAGICC7.f90.

### 9.2 Are the Ocean Models Truly Swappable?

**Answer: Yes, but with significant caveats.**

The models (3D-GFDL, 2D-BERN, HILDA, BOXDIFF) share the same interface but differ in:
- IRF functional forms (polynomial vs. exponential)
- IRF coefficients
- Gas exchange timescales
- Mixed layer depths
- Ocean surface areas
- Preindustrial SST

They are swappable at runtime via the `OCEANCC_MODEL` parameter. However:

1. **No biological pump:** All models use abiotic chemistry only. Biological carbon export is not represented.

2. **Single box:** Despite names like "3D-GFDL", these are all single mixed-layer representations. The "3D" character is only in the IRF calibration.

3. **Fixed coefficients:** The polynomial coefficients for pCO2-DIC relationship are the same for all models, but should they be? Different models have different carbonate chemistry.

### 9.3 Hardcoded Values

| Location | Hardcoded Value | Concern |
|----------|-----------------|---------|
| `oceancc_init`, lines 172-177 | pCO2-DIC offsets and coefficients | Should be model parameters |
| `carbon_cycle_ocean.f90`, lines 413-446 | IRF polynomial coefficients (3D-GFDL) | Per-model, extensive |
| `carbon_cycle_ocean.f90`, lines 466-502 | IRF exponential coefficients (2D-BERN) | Per-model |
| `carbon_cycle_ocean.f90`, lines 516-546 | IRF coefficients (HILDA) | Per-model |
| `carbon_cycle_ocean.f90`, lines 564-595 | IRF coefficients (BOXDIFF) | Per-model |
| `MAGICC7.f90`, line 4887 | Stability limit 0.04 | Magic number |
| `units.f90`, lines 8-14 | Unit conversion constants | Physical constants, acceptable |

### 9.4 Complex Coupling

The ocean carbon cycle has non-trivial coupling with:

1. **CO2 concentration state:** Both reads and writes `DAT_CO2_CONC`
2. **Temperature feedback:** Requires `co2_feed_deltatemp` from climate module
3. **Land carbon cycle:** Competes for atmospheric CO2
4. **Time stepping:** Monthly stepping while rest of model is annual

The coupling is bidirectional within a single annual timestep:
- Atmosphere provides CO2 to ocean
- Ocean returns flux that modifies CO2
- This feedback is resolved via the monthly sub-loop

### 9.5 Design Issues

1. **Over-engineered class hierarchy:** The `CarbonCycleOceanCalculatorMAGICCConstantStepTimeAxis` contains three sub-calculators, each with their own build/finalize pattern. This adds complexity without clear benefit.

2. **Manager classes:** The `*_manager.f90` files maintain arrays of 2048 instances for some Python interop purpose. This is infrastructure code mixed with physics code.

3. **TODO comments abound:** The code contains many `TODO` and `#(issue number)` comments indicating known technical debt:
   - #206: Move name to model
   - #207: Push monthly loop into calculator
   - #208: Remove averaging hack and stability limit
   - #209: Investigate monthly stepping necessity
   - #210: Separate radiative setting from physics
   - #211: Remove CO2 cap from physics

4. **Comment about design philosophy:**
   ```fortran
   ! Constants are hard-coded here. It doesn't make sense to expose these
   ! constants because of the 'on the fly' coupling approach that results
   ! from using a model name as a MAGICC parameter.
   ```
   This suggests the developers recognized the design was suboptimal.

### 9.6 Potential Bugs and Edge Cases

1. **IRF scaling continuity:** The IRF scaling applies a nonlinear transformation before the switch time and a linear factor after. There's careful code to ensure continuity at the switch point, but the logic is complex and could harbor edge cases.

2. **Zero flux handling:** If the initial flux is zero, the stability constraint comparison to `currentstep - 1` could access index 0 if not guarded (though `MAX(1, ...)` is used elsewhere).

3. **Negative DIC:** The polynomial expansion for pCO2-DIC is only valid for positive DIC changes. Negative DIC (outgassing scenario) could produce unexpected results.

4. **Temperature extrapolation:** The `co2_feed_deltatemp` comes from temperature feedback logic that has known issues (clamping negative values to zero, unintended timing).

### 9.7 Missing Physics

1. **No biological pump:** No representation of marine biology removing carbon from surface
2. **No ocean circulation changes:** IRF is static, doesn't respond to circulation changes from warming
3. **No alkalinity changes:** Carbonate chemistry assumes constant alkalinity
4. **No regional variation:** Single global mixed layer

## 10. Test Cases

### 10.1 Unit Test: Zero Flux at Equilibrium

**Purpose:** Verify that pCO2_atm = pCO2_ocn produces zero flux.

**Setup:**
```
c_atm = 280.0 ppm
c_ocn = 280.0 ppm
```

**Expected output:**
- `flux = 0.0` exactly
- Ocean pCO2 remains at 280.0 ppm

### 10.2 Unit Test: Positive Flux into Ocean

**Purpose:** Verify ocean absorbs CO2 when atmosphere > ocean.

**Setup:**
```
c_atm = 400.0 ppm
c_ocn = 280.0 ppm
OCEANCC_MODEL = "3D-GFDL"
OCEANCC_SCALE_GASXCHANGE = 1.0
```

**Expected output:**
- `flux > 0` (atmosphere to ocean)
- Flux proportional to (400 - 280) = 120 ppm difference
- With k = 1/(7.66*12), flux ~ 1.3 ppm/month = 15.6 ppm/yr

### 10.3 Unit Test: Temperature Effect on pCO2

**Purpose:** Verify warming increases ocean pCO2.

**Setup:**
```
delta_dic = 0 (no DIC change)
delta_sst = 1.0 K
OCEANCC_TEMPFEEDBACK = 0.0423
```

**Expected output:**
- pCO2_ocean = pCO2_preindustrial * exp(0.0423 * 1.0) ~ 1.043 * pCO2_preindustrial
- ~4.3% increase per degree warming

### 10.4 Unit Test: DIC Effect on pCO2 (Revelle Buffer)

**Purpose:** Verify DIC increase raises pCO2.

**Setup:**
```
delta_sst = 0
delta_dic = 50 micromol/kg
T_preindustrial = 17.7
```

**Expected output:**
- Calculate using Joos A24 polynomial
- pCO2 should increase more than proportionally to DIC (Revelle factor > 1)

### 10.5 Integration Test: IRF Convolution

**Purpose:** Verify IRF convolution produces expected DIC.

**Setup:**
- Constant flux of 1 ppm/yr for 100 years
- 3D-GFDL model

**Expected output:**
- DIC should approach steady state as IRF decays
- Most carbon should remain in mixed layer initially, then penetrate deeper
- After 100 years, significant fraction should have entered deep ocean

### 10.6 Integration Test: Model Intercomparison

**Purpose:** Verify different models give different but reasonable results.

**Setup:**
- Run 1%/yr CO2 increase scenario for 140 years
- Compare 3D-GFDL, 2D-BERN, HILDA, BOXDIFF

**Expected output:**
- All models show ocean uptake
- Uptake rates differ by 10-30%
- HILDA has deeper mixed layer, may show different transient response
- Results should bracket range of full ocean models

### 10.7 Edge Case: Very High CO2

**Purpose:** Test behavior at extreme CO2 levels.

**Setup:**
```
c_atm = 2000.0 ppm (very high)
Run for 500 years
```

**Expected output:**
- Ocean uptake continues but efficiency decreases
- pCO2 polynomial may extrapolate poorly
- Check for numerical stability

### 10.8 Edge Case: Stability Limiter

**Purpose:** Verify stability limiter engages correctly.

**Setup:**
- Sharp step change in atmospheric CO2 (e.g., instant doubling)

**Expected output:**
- Flux should be limited to change by at most 0.04 ppm/yr per month
- No numerical instabilities or oscillations

## 11. Fortran Code References

### 11.1 Key Files

| File | Purpose |
|------|---------|
| `physics/carbon_cycle_ocean/carbon_cycle_ocean.f90` | Main calculator class and IRF functions |
| `physics/carbon_cycle_ocean/atmosphere_to_ocean_carbon_flux.f90` | Air-sea flux calculation |
| `physics/carbon_cycle_ocean/ocean_surface_partial_pressure.f90` | pCO2 calculation (Joos equations) |
| `physics/carbon_cycle_ocean/delta_dissolved_inorganic_carbon.f90` | DIC change via IRF convolution |
| `physics/carbon_cycle.f90` | Module definitions for OCEANCC_* parameters |
| `MAGICC7.f90` | Monthly loop and integration with main model |
| `utils/units.f90` | Unit conversion constants |
| `utils/maths.f90` | exponential_decay and integrate_lh_riemann_sum |

### 11.2 Key Line Numbers

| Function/Section | File | Lines |
|------------------|------|-------|
| Main calculator type | `carbon_cycle_ocean.f90` | 19-49 |
| `build_from_name_and_parameters` | `carbon_cycle_ocean.f90` | 82-209 |
| 3D-GFDL IRF polynomial (before switch) | `carbon_cycle_ocean.f90` | 396-429 |
| 3D-GFDL IRF exponential (after switch) | `carbon_cycle_ocean.f90` | 431-454 |
| 2D-BERN IRF (before switch) | `carbon_cycle_ocean.f90` | 456-479 |
| 2D-BERN IRF (after switch) | `carbon_cycle_ocean.f90` | 481-504 |
| HILDA IRF (before switch) | `carbon_cycle_ocean.f90` | 506-529 |
| HILDA IRF (after switch) | `carbon_cycle_ocean.f90` | 531-554 |
| BOXDIFF IRF (before switch) | `carbon_cycle_ocean.f90` | 556-572 |
| BOXDIFF IRF (after switch) | `carbon_cycle_ocean.f90` | 574-597 |
| `calc_atm_to_ocn_c_flux` | `carbon_cycle_ocean.f90` | 351-364 |
| `calc_ospp` | `carbon_cycle_ocean.f90` | 366-394 |
| Air-sea flux calculator | `atmosphere_to_ocean_carbon_flux.f90` | 39-48 |
| pCO2 calculator (Joos A24) | `ocean_surface_partial_pressure.f90` | 86-115 |
| pCO2 calculator (Joos A25) | `ocean_surface_partial_pressure.f90` | 117-137 |
| DIC convolution integral | `delta_dissolved_inorganic_carbon.f90` | 211-269 |
| IRF scaling function | `delta_dissolved_inorganic_carbon.f90` | 130-143 |
| Module parameters | `carbon_cycle.f90` | 149-156 |
| `oceancc_init` | `carbon_cycle.f90` | 163-189 |
| Monthly loop in main driver | `MAGICC7.f90` | 4849-4939 |
| Stability limiter | `MAGICC7.f90` | 4887-4898 |
| Unit constants | `units.f90` | 8-14 |

### 11.3 IRF Coefficients (Full Listing)

**3D-GFDL Polynomial (before 1 year):**
```fortran
irf_polynomial_coefficients = (/ &
    1.0, -2.2617, 14.002, -48.770, 82.986, -67.527, 21.037 &
/)
```

**3D-GFDL Exponential (after 1 year):**
```fortran
irf_exponential_coefficients = (/ &
    0.01481, 0.019439, 0.038344, 0.066485, 0.24966, 0.70367 &
/)
irf_exponential_lifetimes = MONTHS_PER_YEAR * (/ &
    1.0e10, 347.55, 65.359, 15.281, 2.3488, 0.70177 &
/)
```

**2D-BERN (before 9.9 years):**
```fortran
irf_exponential_coefficients = (/ &
    0.058648, 0.07515, 0.079338, 0.41413, 0.24845, 0.12429 &
/)
irf_exponential_lifetimes = MONTHS_PER_YEAR * (/ &
    1.0e10, 9.62180, 9.23640, 0.7603, 0.16294, 0.0032825 &
/)
```

**2D-BERN (after 9.9 years):**
```fortran
irf_exponential_coefficients = (/ &
    0.01369, 0.012456, 0.026933, 0.026994, 0.036608, 0.06738 &
/)
irf_exponential_lifetimes = MONTHS_PER_YEAR * (/ &
    1.0e10, 331.54, 107.57, 38.946, 11.677, 10.515 &
/)
```

**HILDA (before 2 years):**
```fortran
irf_exponential_coefficients = (/ &
    0.12935, 0.24093, 0.24071, 0.17003, 0.21898 &
/)
irf_exponential_lifetimes = MONTHS_PER_YEAR * (/ &
    1.0e10, 4.9792, 0.96083, 0.26936, 0.034569 &
/)
```

**HILDA (after 2 years):**
```fortran
irf_exponential_coefficients = (/ &
    0.022936, 0.035549, 0.037820, 0.089318, 0.13963, 0.24278 &
/)
irf_exponential_lifetimes = MONTHS_PER_YEAR * (/ &
    1.0e10, 232.30, 68.736, 18.601, 5.2528, 1.2679 &
/)
```

**BOXDIFF (before 3.2 years) - Special power-law form:**
```fortran
irf_val = ( &
    0.1476804 / (t_year + 0.026540147)**0.3881032 &
    + 0.3439660 / (t_year + 0.7751384)**0.5519552 &
)
```

**BOXDIFF (after 3.2 years):**
```fortran
irf_exponential_coefficients = (/ &
    0.0197368421, 0.0315281, 0.0104691, 0.0504693, 0.076817, 0.118034, 0.168507 &
/)
irf_exponential_lifetimes = MONTHS_PER_YEAR * (/ &
    1.0e10, 215.7122, 148.7718, 43.50592, 14.17156, 4.870225, 1.63876 &
/)
```

---

## Summary

The Ocean Carbon Cycle module implements a computationally efficient IRF-based approach to simulate ocean CO2 uptake. The core mathematics (air-sea flux, IRF convolution, Joos pCO2 equations) are well-established and correctly implemented.

**Strengths:**
- Well-encapsulated calculator classes
- Multiple ocean model emulators available
- Physically-based pCO2 calculation with temperature and DIC effects
- Modular design allows model swapping

**Weaknesses:**
- Monthly sub-stepping likely unnecessary with proper integration
- Ad-hoc stability limiter instead of proper numerical method
- Extensive hardcoded coefficients not exposed as parameters
- Complex coupling with main driver code
- Many acknowledged TODOs indicating technical debt
- No biological pump or circulation change feedbacks

For reimplementation, the mathematical formulation is complete in this document. Key attention should be paid to:
1. Unit handling (ppm/yr vs ppm/month internally)
2. IRF scaling function at the switch time
3. Correct implementation of Riemann sum convolution
4. The polynomial coefficients are model-specific and extensive

The four ocean models (3D-GFDL, 2D-BERN, HILDA, BOXDIFF) are genuinely swappable at runtime but represent the same simplified physics with different calibrations, not fundamentally different approaches.
