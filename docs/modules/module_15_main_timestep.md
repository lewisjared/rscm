# Module 15: Main Timestep Integration

## 1. Purpose

The Main Timestep Integration module orchestrates all physics modules within MAGICC's annual simulation loop. It is the "conductor" that coordinates the sequence of calculations across the carbon cycle, atmospheric chemistry, radiative forcing, climate response, and impact modules.

The two key subroutines are:

1. **`magicc_run`** (lines 2635-2643): The outer loop that iterates over all simulation years
2. **`magicc_step_year`** (lines 2646-3727): The ~1100-line main routine that executes all physics for a single year

This module ensures that:

- Dependencies between modules are respected (e.g., concentrations before forcing, forcing before temperature)
- Feedback loops are properly implemented (temperature -> carbon cycle -> CO2 -> forcing -> temperature)
- Sub-annual stepping is handled correctly for numerical stability
- State variables are updated in the correct order

## 2. Module Coordination

### 2.1 High-Level Execution Flow

```
magicc_run
    |
    +-- FOR each year (1 to NYEARS):
        |
        +-- magicc_step_year
                |
                +-- DELTAQ (radiative forcing calculations)
                |       |
                |       +-- Permafrost module (if enabled)
                |       +-- CH4 chemistry and concentrations
                |       +-- N2O chemistry and concentrations
                |       +-- TERRCARBON2 (terrestrial carbon cycle)
                |       +-- Ocean carbon cycle (monthly sub-stepping)
                |       +-- All radiative forcing calculations
                |       +-- Total forcing aggregation
                |
                +-- Climate module (sub-annual stepping)
                |       |
                |       +-- FOR each month (1 to STEPSPERYEAR):
                |       |       +-- LAMCALC (feedback parameters)
                |       |       +-- Ocean diffusion-advection solve
                |       |       +-- Temperature calculations
                |       |       +-- Upwelling rate updates
                |       +-- END FOR
                |
                +-- Sea level calculation
```

### 2.2 Module Dependencies

The order of module execution is determined by physical dependencies:

| Order | Module | Depends On | Provides For |
|-------|--------|------------|--------------|
| 1 | Permafrost | Temperature (prev year) | CH4, CO2 emissions |
| 2 | CH4 Chemistry | CH4 emissions, Temperature | CH4 concentration |
| 3 | N2O Chemistry | N2O emissions | N2O concentration |
| 4 | Terrestrial Carbon | Temperature, CO2 conc | CO2 flux |
| 5 | Ocean Carbon | Temperature, CO2 conc | Ocean CO2 flux |
| 6 | CO2 Budget | Terr + Ocean fluxes | CO2 concentration |
| 7 | Halocarbon Chemistry | Emissions | Halocarbon concentrations |
| 8 | Radiative Forcing | All concentrations | Total forcing |
| 9 | Climate (UDEB) | Total forcing | Temperature, Heat uptake |
| 10 | Sea Level | Temperature, Heat content | Sea level rise |

### 2.3 Feedback Loops

The model implements several important feedback loops:

1. **Temperature -> Carbon Cycle -> CO2 -> Forcing -> Temperature**
   - Global temperature affects NPP, respiration, ocean solubility
   - Carbon fluxes modify atmospheric CO2
   - CO2 concentration affects radiative forcing
   - Forcing drives temperature change

2. **Temperature -> CH4 Lifetime -> CH4 -> Forcing -> Temperature**
   - Temperature affects OH radical chemistry
   - OH affects methane lifetime
   - Methane concentration affects forcing

3. **Temperature -> Upwelling -> Ocean Heat -> Temperature**
   - Warming reduces thermohaline circulation
   - Reduced upwelling changes deep ocean heat uptake
   - Ocean heat content affects transient temperature response

## 3. `magicc_step_year` Algorithm

### 3.1 Complete Pseudocode

```
SUBROUTINE magicc_step_year

    ! === YEAR INDEX SETUP ===
    NEXT_YEAR_IDX = CURRENT_YEAR_IDX + 1
    PREVIOUS_YEAR_IDX = CURRENT_YEAR_IDX - 1 (min 1)

    ! === PHASE 1: RADIATIVE FORCING (DELTAQ) ===
    IF (not last year) THEN
        CALL DELTAQ
            ! 1. Temperature extrapolation for feedbacks
            ! 2. Permafrost calculation (if enabled)
            ! 3. CH4 concentration and lifetime
            ! 4. N2O concentration
            ! 5. Terrestrial carbon cycle (TERRCARBON2)
            ! 6. Ocean carbon cycle (monthly sub-stepping)
            ! 7. CO2 concentration update
            ! 8. Halocarbon concentrations
            ! 9. All radiative forcing components:
            !    - CO2, CH4, N2O forcing
            !    - Halocarbon forcing (F-gases, Montreal gases)
            !    - Ozone forcing (stratospheric, tropospheric)
            !    - Aerosol forcing (direct, indirect)
            !    - Other forcings (land use, BC on snow, aviation)
            ! 10. Total forcing aggregation
    END IF

    ! === PHASE 2: CLIMATE RESPONSE ===
    ! Initialize accumulators
    OCEANT = 0
    LANDT = 0
    GLOBET = 0

    ! Determine sub-annual step range
    STARTSTEP = 1 (or 0 for first year)
    ENDSTEP = STEPSPERYEAR (or STEPSPERYEAR-1 for last year)

    ! === SUB-ANNUAL LOOP ===
    DO CURRENT_STEP = STARTSTEP, ENDSTEP
        CURRENT_TIME_IDX = CURRENT_TIME_IDX + 1

        ! 2.1 Get interpolated forcing for this timestep
        Q(1:4) = interpolate_forcing(dat_total_effrf, current_time)

        ! 2.2 Add volcanic forcing (if applicable)
        IF (volcanics_enabled) THEN
            Q = Q + dat_volcanic_effrf(current_time_idx)
        END IF

        ! 2.3 Calculate cumulative-temperature adjusted climate sensitivity
        CUMULATIVE_T = sum(recent_temperatures)
        CLIMATESENSITIVITY_ADJ = adjust_ecs(CUMULATIVE_T, Q)

        ! 2.4 Calculate ocean/land feedback parameters
        CALL LAMCALC(Q2x, K_lo, K_ns, ECS_adj, RLO, XLAMO, XLAML)

        ! === HEMISPHERE LOOP ===
        DO HEMIS_IDX = 1, 2  ! 1=NH, 2=SH

            ! 2.5 Calculate temperature-dependent vertical diffusivity
            DO L = 1, NLEVELS-1
                K(L) = K_base + dK/dT * (1-depth_frac) * (T_top - T_bottom)
                K(L) = max(K(L), K_min)
            END DO

            ! 2.6 Calculate land feedback denominator
            DENOM = FGO * (K_lo + FGL * XLAML)

            ! 2.7 Build tridiagonal matrix coefficients
            ! --- Mixed layer (layer 1) ---
            A(1) = 1 + dt/dz_mix * (K/dz1 + w*pi + feedback_term)
            B(1) = -dt/dz_mix * (K/dz1 + w)
            D(1) = T_prev + forcing_terms + variable_w_terms

            ! Optional: Add ground heat capacity effect
            IF (land_heat_capacity_enabled) THEN
                D(1) = D(1) - ground_heat_flux_term
                T_ground = update_ground_temperature()
            END IF

            ! Optional: Add El Nino effect
            IF (elnino_enabled) THEN
                D(1) = D(1) + elnino_scale * NINO34_index
                update_elnino_box()
            END IF

            ! --- Layer 2 (below mixed layer) ---
            A(2) = diffusion_from_above
            B(2) = 1 + diffusion_both_ways + upwelling
            C(2) = diffusion_and_upwelling_from_below
            D(2) = T_prev + entrainment_term + variable_w_terms

            ! --- Middle layers (3 to N-1) ---
            DO L = 3, NLEVELS-1
                A(L) = diffusion_from_above
                B(L) = 1 + diffusion_both_ways + upwelling
                C(L) = diffusion_and_upwelling_from_below
                D(L) = T_prev + entrainment_term + variable_w_terms
            END DO

            ! --- Bottom layer ---
            A(N) = diffusion_from_above
            B(N) = 1 + diffusion_above + upwelling
            D(N) = T_prev + entrainment_term + variable_w_terms

            ! 2.8 Solve tridiagonal system (Thomas algorithm)
            ! Forward sweep
            AA(1) = -B(1) / A(1)
            BB(1) = D(1) / A(1)
            DO L = 2, N-1
                VV = A(L) * AA(L-1) + B(L)
                AA(L) = -C(L) / VV
                BB(L) = (D(L) - A(L) * BB(L-1)) / VV
            END DO

            ! Back substitution
            T(N) = (D(N) - A(N)*BB(N-1)) / (A(N)*AA(N-1) + B(N))
            DO I = 1, N-1
                L = N - I
                T(L) = AA(L) * T(L+1) + BB(L)
            END DO

        END DO  ! hemisphere loop

        ! 2.9 Apply temperature safety cap
        IF (any(T > 25K)) THEN
            T = min(T, 25K)
            log_warning("Temperatures exceeded 25K")
        END IF

        ! 2.10 Convert ocean SST to air temperature
        T_air_ocean = alpha * T_sst + gamma * T_sst^2

        ! 2.11 Calculate land temperature (equilibrium with ocean)
        T_land = (FGL * Q_land + K_lo * alpha * T_ocean) / (FGL * XLAML + K_lo)

        ! 2.12 Apply land temperature safety caps
        T = constrain(T, -25K, 25K)

        ! 2.13 Store sub-annual temperatures
        IF (CURRENT_STEP > 0) THEN
            THISYEAR_TEMPS(CURRENT_STEP) = T
            DAT_SURFACE_TEMP_SUBANNUAL(current_time_idx) = T
        END IF

        ! 2.14 Calculate heat uptake (energy balance)
        heat_uptake = Q - (XLAMO*FGO*T_ocean + XLAML*FGL*T_land)

        ! 2.15 Update hemispheric heat exchange
        HEMISPHERIC_HEATXCHANGE(1) = K_ns/FGO(1) * (T_SH - T_NH)
        HEMISPHERIC_HEATXCHANGE(2) = K_ns/FGO(2) * (T_NH - T_SH)

        ! 2.16 Calculate mean temperatures
        OCEANT = area_weighted_ocean_mean(T)
        LANDT = area_weighted_land_mean(T)
        GLOBET = global_area_weighted_mean(T)

        ! 2.17 Update upwelling rate
        SELECT CASE (upwelling_method)
            CASE ("NOSCALING")
                w = w_initial
            CASE ("PRESCRIBED")
                w = dat_upwelling_rate(next_year)
            CASE DEFAULT  ! GLOBE, OCEAN, HEMISPHERIC
                T_key = get_key_temperature(method, GLOBET, OCEANT, T_hem)
                delta_w = -w_var_frac * w_initial * T_key / T_thresh
                w = max(w_initial + delta_w, w_initial * (1 - w_var_frac))
        END SELECT

    END DO  ! sub-annual step loop

    ! === PHASE 3: POST-STEP CALCULATIONS ===

    ! 3.1 Save climate sensitivities
    IF (GLOBET != 0) THEN
        CLIMSENS_EQ = Q2x / effective_lambda
    END IF

    ! 3.2 Calculate annual mean temperatures
    DAT_SURFACE_ANNUALMEANTEMP = mean(THISYEAR_TEMPS)

    ! 3.3 Store end-of-year temperatures
    DAT_SURFACE_TEMP(next_year) = current_temperature

    ! 3.4 Calculate heat content at various depths
    DO depth_idx = 1, 4
        heat_content(depth_idx) = integrate_ocean_heat(depth_idx)
    END DO

    ! 3.5 Store upwelling rate
    DAT_UPWELLING_RATE(next_year) = current_upwelling_rate

    ! === PHASE 4: SEA LEVEL CALCULATION ===
    CALL sealevel_calc

END SUBROUTINE magicc_step_year
```

### 3.2 Detailed DELTAQ Algorithm

The DELTAQ subroutine (lines 3735-7042) calculates all forcing components:

```
SUBROUTINE DELTAQ

    ! === 1. TEMPERATURE FEEDBACK SETUP ===
    temp_extrap = extrapolate_temperature(T_current, T_prev)

    ! CO2 feedback temperatures
    co2_feed_deltatemp = calculate_delta_temp(co2_tempfeedback_yrstart)

    ! CH4 feedback temperatures
    ch4_feed_deltatemp = calculate_delta_temp(ch4_feed_yrstart)

    ! CH4 clathrate emissions (if enabled)
    IF (ch4_clathratefeed_apply) THEN
        ch4_clathrate_emissions = calculate_clathrate_release()
    END IF

    ! === 2. PERMAFROST MODULE ===
    IF (PF_APPLY) THEN
        CALL permafrost_calc
        ! Outputs: CH4 and CO2 emissions from thawing permafrost
    END IF

    ! === 3. METHANE CHEMISTRY ===
    ! Total CH4 emissions
    CH4_TOTEMIS = CH4I + CH4B + CH4N + CH4_permafrost + CH4_clathrate

    ! Calculate CH4 concentration and lifetime
    CALL METHANE(include_feedback, CH4_conc, CH4_emis,
                 delta_NOx, delta_CO, delta_NMVOC,
                 -> next_CH4_conc, CH4_lifetime, ...)

    ! Update CH4 concentration (if emissions-driven)
    IF (year >= CH4_switch_year) THEN
        DAT_CH4_CONC(next_year) = next_CH4_conc
    END IF

    ! === 4. N2O CHEMISTRY ===
    N2O_TOTEMIS = N2OI + N2OB + N2ON + N2O_permafrost

    ! Calculate N2O concentration
    CALL N2O_CONC_CALC(N2O_emis, N2O_lifetime, -> next_N2O_conc)

    ! === 5. TERRESTRIAL CARBON CYCLE ===
    CALL TERRCARBON2(co2_feed_deltatemp_land)
        ! Calculates:
        ! - NPP with temperature and CO2 fertilization feedbacks
        ! - Respiration with temperature feedback
        ! - Plant, detritus, soil pool dynamics
        ! - Net terrestrial CO2 flux

    ! === 6. OCEAN CARBON CYCLE (MONTHLY SUB-STEPPING) ===
    DO month = 1, STEPSPERYEAR
        ! Calculate air-sea CO2 flux
        flux = carbon_cycle_ocean_calculator%calc_flux(c_atm, c_ocn)

        ! Update ocean pCO2
        pco2s = carbon_cycle_ocean_calculator%calc_ospp(flux, delta_SST)

        ! Update atmospheric CO2 (if emissions-driven)
        IF (year >= CO2_switch_year) THEN
            c_atm = c_atm + emissions/stepsperyear - flux/stepsperyear
        END IF
    END DO

    ! Store final CO2 concentration
    DAT_CO2_CONC(next_year) = c_atm

    ! === 7. HALOCARBON CHEMISTRY ===
    ! F-gases (HFCs, PFCs, SF6)
    DO i = 1, FGAS_N
        CALL calc_fgas_conc_and_forcing(i)
    END DO

    ! Montreal Protocol gases (CFCs, HCFCs, etc.)
    DO i = 1, MHALO_N
        CALL calc_mhalo_conc_and_forcing(i)
    END DO

    ! === 8. RADIATIVE FORCING CALCULATIONS ===

    ! 8.1 CO2 forcing
    DAT_CO2_RF = RF_CO2_RADEFF * ln(CO2/CO2_preindustrial)
    DAT_CO2_EFFRF = DAT_CO2_RF * RF_CO2_EFF_ADJ

    ! 8.2 CH4 forcing (with overlap correction)
    DAT_CH4_RF = calculate_ch4_forcing(CH4_conc, N2O_conc)

    ! 8.3 N2O forcing (with overlap correction)
    DAT_N2O_RF = calculate_n2o_forcing(N2O_conc, CH4_conc)

    ! 8.4 Ozone forcing
    DAT_STRATOZ_RF = stratospheric_ozone_forcing()
    DAT_TROPOZ_RF = tropospheric_ozone_forcing(CH4, NOx, CO, NMVOC)

    ! 8.5 Aerosol forcing
    DAT_TOTAER_DIR_RF = direct_aerosol_forcing()
    DAT_CLOUD_TOT_RF = indirect_aerosol_forcing()

    ! 8.6 Other forcings
    DAT_LANDUSE_RF = land_use_forcing()
    DAT_BCSNOW_RF = bc_on_snow_forcing()
    DAT_AIR_*_RF = aviation_forcing()

    ! === 9. FORCING AGGREGATION ===
    ! Aggregate by category
    DAT_GHG_EFFRF = CO2 + CH4 + N2O + F-gases + M-halos
    DAT_TOTAL_ANTHRO_EFFRF = GHG + aerosols + ozone + other

    ! Total forcing (depends on run mode)
    SELECT CASE (RF_TOTAL_RUNMODUS)
        CASE ("CO2")
            DAT_TOTAL_EFFRF = DAT_CO2_EFFRF
        CASE ("GHG")
            DAT_TOTAL_EFFRF = DAT_GHG_EFFRF
        CASE ("ALL")
            DAT_TOTAL_EFFRF = DAT_TOTAL_ANTHRO_EFFRF + natural
        ! ... other modes
    END SELECT

    ! Add volcanic forcing (annual average)
    DAT_TOTAL_INCLVOLCANIC_EFFRF = DAT_TOTAL_EFFRF + DAT_VOLCANIC_ANNUAL_EFFRF

END SUBROUTINE DELTAQ
```

## 4. Forcing -> Temperature -> Feedback Loop

### 4.1 Core Feedback Cycle

The MAGICC model implements a sequential feedback cycle within each annual timestep:

```
+------------------+     +------------------+     +------------------+
|   Emissions      | --> |  Concentrations  | --> | Radiative Forcing|
| (CO2, CH4, N2O,  |     | (calculated from |     | (DELTAQ routine) |
|  aerosols, etc.) |     |  emissions or    |     |                  |
+------------------+     |  prescribed)     |     +--------+---------+
        ^                +------------------+              |
        |                                                  v
+-------+--------+                               +------------------+
| Carbon Cycle   | <---------------------------  | Climate Module   |
| Feedbacks      |        Temperature            | (4-box UDEB)     |
| (TERRCARBON2,  |                               |                  |
|  Ocean CC)     |                               +------------------+
+----------------+
```

### 4.2 Detailed Feedback Sequence

**Step 1: Emissions -> Concentrations**

```
CO2_conc(t+1) = CO2_conc(t) + [fossil_emis + landuse_emis - ocean_uptake - terr_uptake] / GTC_PER_PPM

CH4_conc(t+1) = CH4_conc(t) + [CH4_emis - CH4_sink] * conversion
    where CH4_sink = CH4_conc * (1/tau_OH + 1/tau_other)

N2O_conc(t+1) = N2O_conc(t) + [N2O_emis - N2O_sink] * conversion
    where N2O_sink = N2O_conc / tau_N2O
```

**Step 2: Concentrations -> Radiative Forcing**

```
Q_CO2 = 5.35 * ln(CO2/CO2_0) * efficacy_CO2

Q_CH4 = alpha_CH4 * (sqrt(CH4) - sqrt(CH4_0)) - overlap_CH4_N2O

Q_N2O = alpha_N2O * (sqrt(N2O) - sqrt(N2O_0)) - overlap_CH4_N2O

Q_aerosol = sum(species_forcing * regional_pattern)

Q_total = Q_CO2 + Q_CH4 + Q_N2O + Q_halos + Q_ozone + Q_aerosol + Q_other
```

**Step 3: Forcing -> Temperature (Climate Module)**

```
For each sub-annual step:
    1. Solve ocean diffusion-advection equation:
       C * dT/dt = Q - lambda*T + heat_exchange

    2. Calculate land temperature from ocean:
       T_land = (Q_land + K_lo * T_ocean) / (lambda_land + K_lo/f_land)

    3. Update global temperature:
       T_global = f_ocean * T_ocean + f_land * T_land
```

**Step 4: Temperature -> Carbon Cycle Feedbacks**

```
Terrestrial:
    NPP = NPP_0 * fertilization_factor(CO2) * exp(beta_npp * delta_T)
    Respiration = R_0 * exp(beta_resp * delta_T)
    Net_flux = NPP - Respiration - heterotrophic_respiration

Ocean:
    Solubility = f(SST)  // CO2 less soluble in warmer water
    DIC_uptake = piston_velocity * (pCO2_atm - pCO2_ocean) / Revelle_factor
```

**Step 5: Feedback Loop Closure**

```
CO2_conc(t+1) depends on:
    - Terrestrial flux (depends on T(t))
    - Ocean flux (depends on T(t))
    - Anthropogenic emissions

Next iteration:
    T(t+1) depends on CO2_conc(t+1) through Q(t+1)
```

### 4.3 Temperature Feedback Parameters

| Feedback | Parameter | Typical Value | Effect |
|----------|-----------|---------------|--------|
| NPP temperature | `CO2_FEEDBACKFACTOR_NPP` | 0.0 | Minimal direct T effect |
| Respiration temperature | `CO2_FEEDBACKFACTOR_RESPIRATION` | 0.0393 | ~4% per K increase |
| Detritus decay | `CO2_FEEDBACKFACTOR_DETRITUS` | 0.0 | No direct T effect |
| Soil decay | `CO2_FEEDBACKFACTOR_SOIL` | 0.0393 | ~4% per K increase |
| Ocean solubility | (implicit in ocean CC) | ~-4%/K | Reduced CO2 uptake |
| CH4 lifetime | `CH4_S` | -0.0323 | Shorter lifetime when warm |

## 5. Sub-Annual Stepping

### 5.1 Purpose

Sub-annual stepping (monthly by default with `STEPSPERYEAR = 12`) serves several purposes:

1. **Numerical stability**: The implicit ocean diffusion scheme is stable at any timestep, but accuracy improves with shorter steps
2. **Forcing interpolation**: Allows smooth transition between annual forcing values
3. **Variability capture**: Enables El Nino and other sub-annual phenomena
4. **Ocean-atmosphere coupling**: More realistic energy exchange dynamics

### 5.2 Time Index Management

```fortran
! Annual indices
CURRENT_YEAR_IDX: 1 to NYEARS
NEXT_YEAR_IDX: CURRENT_YEAR_IDX + 1 (capped at NYEARS)
PREVIOUS_YEAR_IDX: CURRENT_YEAR_IDX - 1 (minimum 1)

! Sub-annual indices
CURRENT_TIME_IDX: cumulative step count from start
NEXT_TIME_IDX: CURRENT_TIME_IDX + 1 (capped at NTIMES)

! Step range within year
STARTSTEP: Usually 1, but 0 for first year
ENDSTEP: Usually STEPSPERYEAR, but STEPSPERYEAR-1 for last year
```

### 5.3 Boundary Handling

The sub-annual loop handles year boundaries carefully:

- **First year (CURRENT_YEAR_IDX = 1)**: `STARTSTEP = 0` to include initial state
- **Last year (CURRENT_YEAR_IDX = NYEARS)**: `ENDSTEP = STEPSPERYEAR - 1` to avoid overshooting
- **Normal years**: Loop from step 1 to STEPSPERYEAR, where final step becomes next year's initial state

### 5.4 Forcing Interpolation

Forcing is interpolated between annual values:

```fortran
Q = datastore_get_box_with_interpolation(dat_total_effrf, alltimes_d(current_time_idx))
```

The `alltimes_d` array contains decimal year values (e.g., 1850.0, 1850.083, 1850.167, ...).

## 6. Module Call Sequence

### 6.1 Complete Call Sequence Table

| Line | Subroutine/Action | Module | Description |
|------|-------------------|--------|-------------|
| 2638 | Initialize time index | Years | Set `CURRENT_TIME_IDX = 0` |
| 2639 | Year loop start | Main | `DO CURRENT_YEAR_IDX = 1, NYEARS` |
| 2695 | `CALL DELTAQ` | Main | All forcing calculations |
| 3815-3857 | Temperature extrapolation | DELTAQ | For feedback calculations |
| 3923-3925 | `CALL permafrost_calc` | Permafrost | If `PF_APPLY = 1` |
| 4031-4035 | `CALL METHANE` | CH4 | Calculate CH4 concentration |
| 4244-4270 | N2O calculations | N2O | Calculate N2O concentration |
| 4820 | `CALL TERRCARBON2` | Carbon | Terrestrial carbon cycle |
| 4849-4952 | Ocean CC monthly loop | Ocean Carbon | Monthly sub-stepping |
| 5200-5500 | Halocarbon calculations | Halocarbons | F-gas and M-halo forcing |
| 5600-7040 | Forcing aggregation | Radiative Forcing | Sum all components |
| 2718-3390 | Sub-annual loop | Climate | `DO CURRENT_STEP = STARTSTEP, ENDSTEP` |
| 2724 | Get forcing | Climate | Interpolate for timestep |
| 2765-2768 | `CALL LAMCALC` | Climate | Calculate feedback parameters |
| 2775-3096 | Hemisphere loop | Climate | Ocean temperature solve |
| 3159-3222 | Temperature conversion | Climate | Ocean SST -> air temperature |
| 3244-3283 | Store sub-annual T | Climate | Save temperatures and heat uptake |
| 3300-3388 | Update upwelling | Climate | Temperature-dependent upwelling |
| 3390 | End sub-annual loop | Climate | |
| 3424-3509 | Annual mean calculations | Climate | Average sub-annual values |
| 3586-3720 | Heat content calculation | Climate | Integrate ocean heat |
| 3726 | `CALL sealevel_calc` | Sea Level | Thermal expansion, ice melt |
| 3727 | End year subroutine | Main | |
| 2641 | End year loop | Main | |

### 6.2 DELTAQ Internal Sequence

| Line | Calculation | Description |
|------|-------------|-------------|
| 3815-3857 | Temperature feedback setup | Extrapolate T for feedbacks |
| 3862-3905 | CH4 feedback temps | Calculate CH4-specific feedback T |
| 3908-3919 | N2O feedback setup | Calculate N2O-specific feedback |
| 3923-3925 | Permafrost | `CALL permafrost_calc` (if enabled) |
| 3927-4095 | CH4 chemistry | Full methane cycle |
| 4244-4270 | N2O chemistry | Full N2O cycle |
| 4818-4996 | Carbon cycle | TERRCARBON2 + ocean CC |
| 5200-5300 | F-gas chemistry | HFC, PFC, SF6 |
| 5300-5500 | M-halo chemistry | CFCs, HCFCs |
| 5550-5600 | CO2 forcing | Calculate CO2 RF |
| 5600-5700 | CH4/N2O forcing | Calculate with overlap |
| 5700-5800 | Halo forcing | Sum F-gas and M-halo |
| 5800-5900 | Ozone forcing | Strat + trop ozone |
| 5900-6000 | Aerosol forcing | Direct + indirect |
| 6000-6200 | Other forcing | Land use, BC snow, aviation |
| 6200-7000 | Forcing aggregation | Sum by category and total |

## 7. State Updates

### 7.1 State Variables Updated Each Timestep

| Variable | Update Location | Timing | Units |
|----------|-----------------|--------|-------|
| `OCN_HEMISPHERIC_LAYERTEMPS` | Line 3087-3093 | Each sub-step | K |
| `CURRENT_TIME_TEMPERATURE` | Line 3159-3222 | Each sub-step | K |
| `CURRENT_TIME_MIXEDLAYERTEMP` | Line 3207-3240 | Each sub-step | K |
| `GROUND_HEMISPHERIC_TEMPS` | Line 2905-2912 | Each sub-step | K |
| `CURRENT_UPWELLING_RATE` | Line 3300-3388 | Each sub-step | m/yr |
| `HEMISPHERIC_HEATXCHANGE` | Line 3288-3291 | Each sub-step | W/m^2 |
| `ELNINO_N34_DOUBLEDELTABOX` | Line 2930-2932 | Each sub-step | K |

### 7.2 Annual State Variables

| Variable | Update Location | Description |
|----------|-----------------|-------------|
| `DAT_CO2_CONC` | DELTAQ:4955 | Atmospheric CO2 |
| `DAT_CH4_CONC` | DELTAQ:4053 | Atmospheric CH4 |
| `DAT_N2O_CONC` | DELTAQ:4270 | Atmospheric N2O |
| `DAT_SURFACE_TEMP` | Line 3462 | End-of-year temperature |
| `DAT_SURFACE_ANNUALMEANTEMP` | Line 3426 | Annual mean |
| `DAT_HEATUPTAKE_EBALANCE_TOTAL` | Line 3572 | Annual heat uptake |
| `DAT_HEATCONTENT_AGGREG` | Line 3717 | Heat content by depth |
| `TEMP_OCEANLAYERS` | Line 3587 | Full ocean profile |
| `DAT_UPWELLING_RATE` | Line 3596 | End-of-year upwelling |

### 7.3 State Update Timing Diagram

```
Year N                                          Year N+1
|                                               |
|--+---+---+---+---+---+---+---+---+---+---+---|--
   1   2   3   4   5   6   7   8   9  10  11  12

   |<-- Sub-annual temperature updates ------>|

                                               |
                                               +-- Annual state storage
                                               +-- DAT_SURFACE_TEMP(N+1)
                                               +-- DAT_CO2_CONC(N+1)
                                               +-- etc.
```

## 8. Integration with Other Modules

### 8.1 TERRCARBON2 Integration

The terrestrial carbon cycle (`TERRCARBON2`, lines 7054-7542) is called from DELTAQ:

**Inputs from Main Loop:**

- `FEEDBACK_TEMPERATURE`: Temperature for respiration feedback
- `DAT_CO2_CONC%DATGLOBE`: CO2 for fertilization
- `DAT_CO2B_EMIS`: Land-use emissions

**Outputs to Main Loop:**

- `CO2_TERRBIO_AND_FOSSIL_EMIS`: Net terrestrial flux + fossil
- `CO2_PLANT_POOL`: Updated plant carbon
- `CO2_DETRITUS_POOL`: Updated detritus carbon
- `CO2_SOIL_POOL`: Updated soil carbon

**Call Context:**

```fortran
! In DELTAQ (line 4820)
CALL TERRCARBON2(CO2_FEED_DELTATEMP_LAND)
```

### 8.2 Ocean Carbon Cycle Integration

The ocean carbon cycle runs on a monthly sub-loop within DELTAQ:

**Inputs:**

- `c_atm_for_oceancc`: Atmospheric CO2 (ppm)
- `co2_feed_deltatemp`: Temperature anomaly for solubility

**Outputs:**

- `co2_air2ocean_flux`: Air-sea CO2 flux (GtC/yr)
- `dat_pco2s_conc`: Ocean surface pCO2

**Call Context:**

```fortran
! In DELTAQ (lines 4849-4952)
do month = 1, stepsperyear
    flux = carbon_cycle_ocean_calculator%calc_atm_to_ocn_c_flux(c_atm, c_ocn)
    pco2s = carbon_cycle_ocean_calculator%calc_ospp(flux, delta_SST)
    ! Update atmospheric CO2 if emissions-driven
end do
```

### 8.3 DELTAQ Integration

DELTAQ (lines 3735-7042) is called at the start of each year:

**Called From:** `magicc_step_year` (line 2695)

**Timing:** Before sub-annual climate stepping

**Purpose:**

1. Calculate next year's concentrations and forcing
2. These forcing values are then interpolated during sub-annual stepping

### 8.4 Climate Module Integration

The climate calculations are embedded in `magicc_step_year`:

**Inputs from DELTAQ:**

- `dat_total_effrf`: Total effective forcing (4 boxes)
- `dat_volcanic_effrf`: Volcanic forcing (if enabled)

**Outputs to Other Modules:**

- `DAT_SURFACE_TEMP`: For carbon cycle feedbacks (next year)
- `TEMP_OCEANLAYERS`: For sea level thermal expansion
- `DAT_HEATUPTAKE_EBALANCE_TOTAL`: For energy budget diagnostics

### 8.5 Permafrost Module Integration

When enabled (`PF_APPLY = 1`), permafrost is called from DELTAQ:

**Inputs:**

- `DAT_SURFACE_TEMP`: Global temperature
- Soil carbon pools from terrestrial model

**Outputs:**

- `DAT_CH4PF_EMIS`: CH4 from permafrost
- `DAT_CO2PF_EMIS`: CO2 from permafrost
- Updated permafrost carbon pools

**Call Context:**

```fortran
! In DELTAQ (lines 3923-3925)
IF (PF_APPLY == 1) THEN
    CALL permafrost_calc
END IF
```

### 8.6 Sea Level Integration

Sea level is calculated at the end of each year:

**Inputs:**

- `TEMP_OCEANLAYERS`: Ocean temperature profile
- `DAT_SURFACE_TEMP`: Global temperature
- `DAT_HEATCONTENT_AGGREG`: Ocean heat content

**Outputs:**

- Sea level rise components (thermal, glacier, ice sheet)
- Total sea level rise

**Call Context:**

```fortran
! In magicc_step_year (line 3726)
CALL sealevel_calc
```

### 8.7 N-Limitation Integration

When enabled (`NCYCLE_APPLY = 1`), nitrogen limitation modifies NPP:

**Called From:** `TERRCARBON2` (line 7293-7300)

**Effect:**

```fortran
IF (NCYCLE_APPLY == 1) THEN
    CALL N_CALC_LIMITATION_FACTOR()
    CO2_CURRENT_NPP = NCYCLE_LIMIT_FACTOR * CO2_CURRENT_NPP
END IF
```

## 9. Numerical Considerations

### 9.1 Timestep Constraints

- **Annual loop**: Fixed at 1 year per iteration
- **Sub-annual**: Configurable via `STEPSPERYEAR` (default 12 = monthly)
- **Ocean carbon**: Uses same monthly stepping as climate

### 9.2 Implicit vs Explicit Schemes

- **Climate module**: Fully implicit (backward Euler) for unconditional stability
- **Carbon cycle**: Semi-implicit with stability checks
- **Chemistry**: Explicit with small timesteps

### 9.3 Iteration Requirements

- **LAMCALC**: Up to 40 iterations per call to converge feedback parameters
- **Carbon cycle**: No iteration within timestep (sequential update)

### 9.4 Conservation Checks

Energy conservation should satisfy:

```
dH/dt = Q_total - lambda * T_global
```

Where `H` is total heat content (atmosphere + ocean + land).

## 10. Code References

### 10.1 Primary Files

| File | Lines | Content |
|------|-------|---------|
| `MAGICC7.f90` | 2635-2643 | `magicc_run` outer loop |
| `MAGICC7.f90` | 2646-3727 | `magicc_step_year` main routine |
| `MAGICC7.f90` | 3735-7042 | `DELTAQ` forcing calculations |
| `MAGICC7.f90` | 7054-7542 | `TERRCARBON2` terrestrial carbon |
| `MAGICC7.f90` | 8070-8278 | `LAMCALC` feedback calculation |
| `physics/permafrost.f90` | 308-772 | `permafrost_calc` |
| `physics/sealevel.f90` | 239-753 | `sealevel_calc` |
| `physics/ocean_carbon_cycle.f90` | - | Ocean carbon cycle class |

### 10.2 Key Fortran Modules Used

| Module | Purpose |
|--------|---------|
| `MOD_YEARS` | Time index management |
| `MOD_CLIMATECORE_AND_OCEAN` | Climate parameters and state |
| `mod_carbon_cycle` | Carbon cycle parameters |
| `MOD_RADIATIVE_FORCING` | Forcing parameters |
| `MOD_DATASTORE` | Data storage structures |
| `MOD_PERMAFROST` | Permafrost parameters |
| `MOD_SEALEVEL` | Sea level parameters |
| `mod_ocean_carbon_cycle` | Ocean carbon cycle class |

### 10.3 Important Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `STEPSPERYEAR` | 12 | Monthly sub-stepping |
| `CORE_OCN_NLEVELS` | 50 | Ocean layers |
| `GTC_PER_PPM` | 2.123 | GtC per ppm CO2 |
| `CH4_PPB2TGCH4` | 2.75 | ppb to TgCH4 conversion |

## 11. Issues and Concerns

### 11.1 Code Organization

**Issue**: The `magicc_step_year` routine is ~1100 lines containing climate physics that should be in `climate_and_ocean.f90`.

**Recommendation**: Refactor climate calculations into dedicated module functions.

### 11.2 Feedback Timing

**Issue**: Temperature feedbacks use previous year's temperature, creating a 1-year lag in the feedback loop.

**Implication**: Fast feedbacks may be slightly underestimated.

### 11.3 Sequential vs Simultaneous Updates

**Issue**: The current sequential update scheme (concentrations -> forcing -> temperature) means each component sees slightly stale values from other components.

**Alternative**: Iterative solution within each timestep could improve accuracy but at computational cost.

### 11.4 Sub-Annual Variability

**Issue**: Many forcings (aerosols, land use) are annual averages applied uniformly to all sub-steps.

**Implication**: Sub-annual forcing variability is not captured except for volcanic forcing.

### 11.5 State Variable Coupling

**Issue**: Some state variables are updated at different points in the loop, creating potential inconsistencies.

**Example**: Upwelling rate updated at end of sub-step uses temperature from that step, but temperature used forcing calculated before upwelling update.

## 12. Rust Rewrite Recommendations

### 12.1 Architecture

```rust
pub struct MagiccTimestep {
    year: Year,
    substeps: Vec<SubStep>,
    state: ModelState,
}

pub trait PhysicsModule {
    fn update(&mut self, state: &ModelState, forcing: &Forcing) -> ModuleOutput;
    fn dependencies(&self) -> Vec<ModuleId>;
}

pub struct ModelOrchestrator {
    modules: Vec<Box<dyn PhysicsModule>>,
    execution_order: Vec<ModuleId>,
}
```

### 12.2 Feedback Loop Implementation

```rust
impl ModelOrchestrator {
    pub fn step_year(&mut self, year: Year) -> Result<YearOutput, MagiccError> {
        // Phase 1: Calculate forcing (DELTAQ equivalent)
        let forcing = self.calculate_forcing(year)?;

        // Phase 2: Sub-annual climate stepping
        for substep in 0..self.config.steps_per_year {
            let q = forcing.interpolate(substep);
            self.climate_module.step(q)?;
        }

        // Phase 3: Post-processing
        let annual_mean = self.calculate_annual_mean()?;

        // Phase 4: Sea level
        self.sealevel_module.calculate(&annual_mean)?;

        Ok(YearOutput { ... })
    }
}
```

### 12.3 Key Design Considerations

1. **Explicit Dependencies**: Use a dependency graph to determine execution order
2. **Immutable State Passing**: Pass state snapshots to modules, collect outputs, apply atomically
3. **Error Propagation**: Use `Result` types throughout for clean error handling
4. **Testing**: Design modules with injectable dependencies for unit testing
5. **Parallelization**: Independent modules (within constraints) could run in parallel
