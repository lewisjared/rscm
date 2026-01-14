# Module 08: Climate / Energy Balance Model

## 1. Scientific Purpose

The Climate Module is MAGICC's core temperature response component, implementing an Upwelling-Diffusion-Entrainment Balance (UDEB) ocean model coupled to a 4-box hemispheric atmosphere structure. This module converts radiative forcing into temperature response, accounting for:

1. **Ocean heat uptake** - The primary source of delayed climate response (transient climate response < equilibrium climate sensitivity)
2. **Land-ocean temperature differential** - Land warms faster than ocean due to lower effective heat capacity
3. **Hemispheric asymmetry** - Different responses in Northern and Southern hemispheres
4. **Variable ocean circulation** - Temperature-dependent upwelling rate (thermohaline circulation weakening)
5. **Multi-layer ocean diffusion** - Deep ocean heat sequestration via vertical diffusion and advection

The model captures both fast (atmospheric/mixed layer) and slow (deep ocean) response timescales, essential for projecting transient climate change and ocean heat content.

## 2. Mathematical Formulation

### 2.1 Overview: Energy Balance with 4-Box Atmosphere

The atmosphere is divided into 4 boxes:

- **Box 1**: Northern Hemisphere Ocean (NHO)
- **Box 2**: Northern Hemisphere Land (NHL)
- **Box 3**: Southern Hemisphere Ocean (SHO)
- **Box 4**: Southern Hemisphere Land (SHL)

Each ocean box is coupled to a multi-layer ocean column (by default 50 layers), while land boxes have simpler heat capacity representation.

The fundamental energy balance for each box is:

$$C \frac{dT}{dt} = Q - \lambda T + H_{exchange}$$

Where:

- $C$ = effective heat capacity (J m^-2 K^-1)
- $T$ = temperature anomaly (K)
- $Q$ = radiative forcing (W m^-2)
- $\lambda$ = feedback parameter (W m^-2 K^-1)
- $H_{exchange}$ = heat exchange with other boxes

### 2.2 4-Box Coupled System for Land-Ocean-Hemisphere Exchange

At equilibrium (dT/dt = 0), the 4-box system satisfies:

$$\mathbf{A} \cdot \mathbf{T} = \mathbf{Q}$$

Where the coefficient matrix **A** (solved in `LAMCALC`) is:

```
A(1,1) = FGNO*lambda_o + K_lo*alpha + K_ns    ! NHO self-coupling
A(1,2) = -K_lo                                 ! NHO-NHL coupling
A(1,3) = -K_ns                                 ! NHO-SHO coupling
A(1,4) = 0                                     ! No direct NHO-SHL coupling

A(2,1) = -K_lo*alpha                          ! NHL-NHO coupling
A(2,2) = FGNL*lambda_l + K_lo                 ! NHL self-coupling
A(2,3) = 0                                     ! No direct NHL-SHO coupling
A(2,4) = 0                                     ! No direct NHL-SHL coupling

A(3,1) = -K_ns                                ! SHO-NHO coupling
A(3,2) = 0                                     ! No direct SHO-NHL coupling
A(3,3) = FGSO*lambda_o + K_lo*alpha + K_ns    ! SHO self-coupling
A(3,4) = -K_lo                                 ! SHO-SHL coupling

A(4,1) = 0                                     ! No direct SHL-NHO coupling
A(4,2) = 0                                     ! No direct SHL-NHL coupling
A(4,3) = -K_lo*alpha                          ! SHL-SHO coupling
A(4,4) = FGSL*lambda_l + K_lo                 ! SHL self-coupling
```

Where:

- `K_lo` = `CORE_HEATXCHANGE_LANDOCEAN` (W m^-2 K^-1) - land-ocean heat exchange coefficient
- `K_ns` = `CORE_HEATXCHANGE_NORTHSOUTH` (W m^-2 K^-1) - inter-hemispheric heat exchange
- `alpha` = `CORE_AMPLIFY_OCN2LAND_HEATXCHNG` - amplification factor for ocean-to-land heat flux
- `lambda_o`, `lambda_l` = ocean and land feedback parameters (derived from climate sensitivity and RLO)
- `FGNO`, `FGNL`, `FGSO`, `FGSL` = area fractions of each box

### 2.3 Climate Sensitivity Decomposition

The equilibrium climate sensitivity (`CORE_CLIMATESENSITIVITY`) is decomposed into land and ocean components via the land-ocean warming ratio `CORE_RLO`:

$$\Delta T_{2x} = \frac{Q_{2x}}{\lambda_{global}}$$

$$\lambda_{global} = f_o \cdot \lambda_o + f_l \cdot \lambda_l$$

$$RLO = \frac{\Delta T_l}{\Delta T_o}$$

The feedback parameters are iteratively solved in `LAMCALC` to satisfy both the global sensitivity and the land-ocean warming ratio constraint.

### 2.4 Time-Varying Climate Sensitivity

Climate sensitivity can vary with forcing and cumulative temperature:

$$ECS_{eff}(t) = ECS_0 \cdot \left(1 + \alpha_Q \cdot \frac{Q(t) - Q_{2x}}{Q_{2x}}\right) \cdot \left(1 + \alpha_T \cdot \frac{\sum T(t) - \sum T_{2x}}{\sum T_{2x}}\right)$$

Where:

- `CORE_FEEDBACK_QSENSITIVITY` = $\alpha_Q$ (forcing sensitivity)
- `CORE_FEEDBACK_CUMTSENSITIVITY` = $\alpha_T$ (cumulative temperature sensitivity)
- `CORE_FEEDBACK_CUMTPERIOD` = period for cumulative temperature calculation

### 2.5 Ocean Mixed Layer Energy Balance

For each ocean hemisphere, the mixed layer (layer 1) satisfies:

$$C_{mix} \frac{dT_1}{dt} = Q_{effective} - TERM_{feedback} \cdot T_1 + F_{diffusion} + F_{upwelling} + F_{exchange}$$

Where the feedback term includes coupled land-ocean effects:

$$TERM_{feedback} = \frac{\alpha_{eff}}{C} \cdot \left( \lambda_o + \frac{\lambda_l \cdot K_{lo} \cdot \alpha \cdot f_l}{f_o \cdot (K_{lo} + f_l \cdot \lambda_l)} \right)$$

This couples ocean temperature feedback with land feedback through heat exchange. Variables:

- $C_{mix}$ = `HEAT_CAPACITY_PERM * CORE_MIXEDLAYER_DEPTH` (W yr m^-2 K^-1)
- $\alpha_{eff}$ = `CORE_TEMPADJUST_OCN2ATM_ALPHAEFF` - ocean-to-atmosphere temperature amplification
- $\lambda_o$, $\lambda_l$ = ocean and land feedback parameters
- $K_{lo}$ = `CORE_HEATXCHANGE_LANDOCEAN` - land-ocean heat exchange
- $\alpha$ = `CORE_AMPLIFY_OCN2LAND_HEATXCHNG` - ocean-to-land amplification
- $f_o$, $f_l$ = ocean and land area fractions
- $F_{diffusion}$ = diffusive heat flux from layer below
- $F_{upwelling}$ = advective heat flux from upwelling
- $F_{exchange}$ = land-ocean and inter-hemispheric heat exchange

### 2.6 Ocean Vertical Diffusion-Advection Equation

Below the mixed layer, temperature evolves according to:

$$\frac{\partial T}{\partial t} = \frac{1}{A(z)} \frac{\partial}{\partial z} \left[ A(z) K \frac{\partial T}{\partial z} \right] - w \frac{\partial T}{\partial z} + \frac{w \cdot \pi \cdot (T_1 - T_{cold})}{A(z)} \cdot \frac{dA}{dz}$$

Where:

- $K$ = vertical diffusivity (m^2/yr) - can vary with depth and temperature gradient
- $w$ = upwelling rate (m/yr)
- $A(z)$ = ocean area at depth z (accounting for bathymetry)
- $\pi$ = `CORE_POLARSINKWATER_TEMPRATIO` - fraction of surface temperature entrained in polar sinking
- $T_{cold}$ = temperature of polar sinking water

The third term represents entrainment from the narrowing ocean basin.

### 2.7 Temperature-Dependent Vertical Diffusivity

The vertical diffusivity varies with the top-to-bottom temperature gradient:

$$K(z, t) = \max\left(K_{min}, K_0 + \frac{dK}{dT} \cdot (1 - z/z_{max}) \cdot (T_1 - T_{bottom})\right)$$

Where:

- $K_0$ = `CORE_VERTICALDIFFUSIVITY` (cm^2/s, converted to m^2/yr by factor 3155.76)
- $K_{min}$ = `CORE_VERTICALDIFFUSIVITY_MIN`
- $\frac{dK}{dT}$ = `CORE_VERTICALDIFF_TOP_DKDT`

### 2.8 Tridiagonal Matrix System

The implicit discretization yields a tridiagonal system for each hemisphere:

$$A_l \cdot T_{l-1}^{n+1} + B_l \cdot T_l^{n+1} + C_l \cdot T_{l+1}^{n+1} = D_l$$

**Mixed Layer (l=1):**

```
A(1) = 1 + dt/dz_mix * (K/dz1 * AF_bottom + w*pi*AF_bottom + TERM_OCN_LAND_FEEDBACK*AF_top)
B(1) = -dt/dz_mix * (K/dz1 + w) * AF_bottom
D(1) = T(1) + Q_terms + delta_w_terms + El_Nino_terms + ground_heat_terms
```

**Layer 2 (just below mixed layer):**

```
A(2) = -dt/dz * K(1)/dz1 * AF_top
B(2) = 1 + dt/dz * (K(1)/dz1*AF_top + K(2)/dz*AF_bottom + w*AF_top)
C(2) = -dt/dz * (K(2)/dz + w) * AF_bottom
D(2) = T(2) + entrainment_term + delta_w_terms
```

**Middle Layers (l=3 to N-1):**

```
A(l) = -dt/dz * K(l-1)/dz * AF_top
B(l) = 1 + dt/dz * (K(l-1)/dz*AF_top + K(l)/dz*AF_bottom + w*AF_top)
C(l) = -dt/dz * (K(l)/dz + w) * AF_bottom
D(l) = T(l) + entrainment_term + delta_w_terms
```

**Bottom Layer (l=N):**

```
A(N) = -dt/dz * K(N-1)/dz * AF_top
B(N) = 1 + dt/dz * (K(N-1)/dz + w) * AF_top
D(N) = T(N) + entrainment_term + delta_w_terms
```

Where `AF_top`, `AF_bottom` are area factors accounting for basin narrowing with depth.

### 2.9 Thomas Algorithm for Tridiagonal Solve

The tridiagonal system is solved using the Thomas algorithm (forward elimination, back substitution):

**Forward sweep:**

```fortran
AA(1) = -B(1) / A(1)
BB(1) = D(1) / A(1)
DO L = 2, N-1
    VV = A(L) * AA(L-1) + B(L)
    AA(L) = -C(L) / VV
    BB(L) = (D(L) - A(L) * BB(L-1)) / VV
END DO
```

**Back substitution:**

```fortran
T(N) = (D(N) - A(N)*BB(N-1)) / (A(N)*AA(N-1) + B(N))
DO I = 1, N-1
    L = N - I
    T(L) = AA(L) * T(L+1) + BB(L)
END DO
```

### 2.10 Temperature-Dependent Upwelling

The upwelling rate decreases with warming (simulating thermohaline circulation weakening):

$$w(t) = w_0 \cdot \left(1 - f_{var} \cdot \frac{T_{key}}{T_{thresh}}\right)$$

With a floor at:

$$w_{min} = w_0 \cdot (1 - f_{var})$$

Where:

- $w_0$ = `CORE_INITIAL_UPWELLING_RATE` (m/yr)
- $f_{var}$ = `CORE_UPWELLING_VARIABLE_PART` (fraction that can vary, default 0.7)
- $T_{thresh}$ = `CORE_UPWELL_THRESH_TEMP_NH` or `_SH` (threshold temperature)
- $T_{key}$ depends on `CORE_UPWELLING_SCALING_METHOD`:
  - "GLOBE": Global mean temperature
  - "OCEAN": Ocean mean temperature
  - "HEMISPHERIC": Respective hemisphere ocean temperature
  - "NOSCALING": Constant upwelling
  - "PRESCRIBED": Read from file

### 2.11 Ocean-to-Atmosphere Temperature Amplification

Surface air temperature over ocean differs from sea surface temperature:

$$T_{air,ocean} = \alpha \cdot T_{SST} + \gamma \cdot T_{SST}^2$$

For $T_{SST} < T^*$ (threshold), otherwise linear with offset $\delta_{max}$:

$$T_{air,ocean} = T_{SST} + \delta_{max}$$

Where:

- $\alpha$ = `CORE_TEMPADJUST_OCN2ATM_ALPHA` (default 1.04)
- $\gamma$ = `CORE_TEMPADJUST_OCN2ATM_GAMMA` (default -0.002)
- $T^* = -(\alpha - 1)/(2\gamma)$
- $\delta_{max} = \alpha T^* + \gamma (T^*)^2 - T^*$

This parameterization ensures $T_{air} \geq T_{SST}$ (air warms more than SST).

### 2.12 Land Temperature (Equilibrium with Ocean)

Land temperature is calculated assuming equilibrium with the adjacent ocean box:

$$T_{land} = \frac{f_l \cdot Q_l + K_{lo} \cdot \alpha \cdot T_{ocean}}{f_l \cdot \lambda_l + K_{lo}}$$

### 2.13 Ground Heat Capacity (Optional)

When `CORE_LANDHEATCAPACITY_APPLY = 1`, a ground heat reservoir damps land temperature response:

$$\frac{dT_{ground}}{dt} = \frac{K_{lg} \cdot (T_{land} - T_{ground})}{f_l \cdot C \cdot d_{eff}}$$

Where:

- $K_{lg}$ = `CORE_HEATXCHANGE_LANDGROUND`
- $d_{eff}$ = `CORE_LANDHC_EFFTHICKNESS` (effective depth in meters)

## 3. State Variables

| Variable | Fortran Name | Symbol | Units | Description | Initial Value |
|----------|--------------|--------|-------|-------------|---------------|
| Ocean layer temperatures (NH) | `OCN_HEMISPHERIC_LAYERTEMPS(1,:)` | $T_{NH,l}$ | K | NH ocean temperature anomaly in each layer | 0.0 |
| Ocean layer temperatures (SH) | `OCN_HEMISPHERIC_LAYERTEMPS(2,:)` | $T_{SH,l}$ | K | SH ocean temperature anomaly in each layer | 0.0 |
| 4-box air temperatures | `CURRENT_TIME_TEMPERATURE(1:4)` | $T_{NHO}, T_{NHL}, T_{SHO}, T_{SHL}$ | K | Air temperature anomalies for 4 boxes | 0.0 |
| Mixed layer temperatures | `CURRENT_TIME_MIXEDLAYERTEMP(1:4)` | $T_{mix}$ | K | Sea surface temperature anomalies | 0.0 |
| Ground temperatures | `GROUND_HEMISPHERIC_TEMPS(1:2)` | $T_{ground}$ | K | Ground heat reservoir temperature | 0.0 |
| Current upwelling rate (NH) | `CURRENT_UPWELLING_RATE(1)` | $w_{NH}$ | m/yr | Current NH upwelling rate | `CORE_INITIAL_UPWELLING_RATE` |
| Current upwelling rate (SH) | `CURRENT_UPWELLING_RATE(2)` | $w_{SH}$ | m/yr | Current SH upwelling rate | `CORE_INITIAL_UPWELLING_RATE` |
| Delta upwelling rate | `DELTA_CURRENT_UPWELLING_RATE(1:2)` | $\Delta w$ | m/yr | Change from initial upwelling | 0.0 |
| El Nino heat box | `ELNINO_N34_DOUBLEDELTABOX` | - | K | Heat anomaly in El Nino "parallel box" | 0.0 |
| Inter-hemispheric heat exchange | `HEMISPHERIC_HEATXCHANGE(1:2)` | $H_{NS}$ | W/m^2 | Heat flux between hemispheres | 0.0 |

## 4. Parameters

### 4.1 Climate Sensitivity Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Equilibrium climate sensitivity | `CORE_CLIMATESENSITIVITY` | K | 3.0 | 1.5-6.0 | Equilibrium warming for CO2 doubling |
| Forcing for 2xCO2 | `CORE_DELQ2XCO2` | W/m^2 | 3.71 | 3.5-4.0 | Radiative forcing for doubled CO2 |
| Land-ocean warming ratio | `CORE_RLO` | - | 1.317 | 1.0-2.0 | Ratio of land to ocean equilibrium warming |
| Forcing sensitivity | `CORE_FEEDBACK_QSENSITIVITY` | - | 7.84e-9 | - | Sensitivity of ECS to forcing level |
| Cumulative T sensitivity | `CORE_FEEDBACK_CUMTSENSITIVITY` | - | 0.08 | - | Sensitivity of ECS to cumulative temperature |
| Cumulative T period | `CORE_FEEDBACK_CUMTPERIOD` | years | 300 | - | Period for cumulative temperature integration |

### 4.2 Ocean Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Mixed layer depth | `CORE_MIXEDLAYER_DEPTH` | m | 60 | 40-100 | Depth of ocean mixed layer |
| Number of ocean levels | `CORE_OCN_NLEVELS` | - | 50 | 20-60 | Total ocean layers (including mixed layer) |
| Vertical diffusivity | `CORE_VERTICALDIFFUSIVITY` | cm^2/s | 0.75 | 0.3-2.0 | Base vertical diffusivity |
| Min vertical diffusivity | `CORE_VERTICALDIFFUSIVITY_MIN` | cm^2/s | 0.1 | 0.0-0.5 | Floor for variable diffusivity |
| Diffusivity temperature gradient | `CORE_VERTICALDIFF_TOP_DKDT` | cm^2/s/K | -0.191 | -0.5-0 | dK/dT coefficient |
| Depth-dependent area | `CORE_OCN_DEPTHDEPENDENT` | - | 1.0 | 0-1 | Scale factor for basin narrowing |
| Polar sinking water temp ratio | `CORE_POLARSINKWATER_TEMPRATIO` | - | 0.2 | 0-0.5 | Fraction of surface T in sinking water |

### 4.3 Upwelling Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Initial upwelling rate | `CORE_INITIAL_UPWELLING_RATE` | m/yr | 3.5 | 2-6 | Base upwelling velocity |
| Variable upwelling fraction | `CORE_UPWELLING_VARIABLE_PART` | - | 0.7 | 0-1 | Fraction of upwelling that can vary |
| Upwelling threshold temp (NH) | `CORE_UPWELL_THRESH_TEMP_NH` | K | 8.0 | 4-12 | Temp at which upwelling reaches minimum (NH) |
| Upwelling threshold temp (SH) | `CORE_UPWELL_THRESH_TEMP_SH` | K | 8.0 | 4-12 | Temp at which upwelling reaches minimum (SH) |
| One global threshold | `CORE_UPWELL_THRESH_ONEGLOBAL` | - | 1 | 0,1 | Use same threshold for both hemispheres |
| Upwelling scaling method | `CORE_UPWELLING_SCALING_METHOD` | string | "GLOBE" | see 2.10 | Method for temperature-dependent upwelling |

### 4.4 Heat Exchange Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Land-ocean heat exchange | `CORE_HEATXCHANGE_LANDOCEAN` | W/m^2/K | 1.44 | 0.5-3.0 | Heat exchange coefficient |
| Inter-hemispheric exchange | `CORE_HEATXCHANGE_NORTHSOUTH` | W/m^2/K | 0.31 | 0.1-1.0 | N-S heat exchange coefficient |
| Ocean-to-land amplification | `CORE_AMPLIFY_OCN2LAND_HEATXCHNG` | - | 1.02 | 1.0-1.2 | Amplification of ocean-to-land flux |
| Land-ground exchange | `CORE_HEATXCHANGE_LANDGROUND` | W/m^2/K | 0.1 | 0-1 | Heat exchange with ground reservoir |
| Apply land heat capacity | `CORE_LANDHEATCAPACITY_APPLY` | - | 1 | 0,1 | Enable ground heat reservoir |
| Land heat capacity thickness | `CORE_LANDHC_EFFTHICKNESS` | m | 300 | 50-500 | Effective depth of ground heat pool |

### 4.5 Temperature Adjustment Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Apply temp adjustment | `CORE_SWITCH_TEMPADJUST_OCN2ATM` | - | 1 | 0,1 | Enable ocean-to-atmosphere temp adjustment |
| Temp adjustment alpha | `CORE_TEMPADJUST_OCN2ATM_ALPHA` | - | 1.04 | 1.0-1.5 | Linear coefficient for T_air/T_SST |
| Temp adjustment gamma | `CORE_TEMPADJUST_OCN2ATM_GAMMA` | - | -0.002 | -0.01-0 | Quadratic coefficient for T_air/T_SST |
| Maximum temperature | `CORE_MAXIMAL_TEMPERATURE` | K | 25.0 | 15-50 | Safety cap on temperature anomalies |

### 4.6 Area Fraction Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| NH land fraction | `CORE_HEMISFRACTION_NH_LAND` | - | 0.42 | - | Fraction of NH that is land |
| SH land fraction | `CORE_HEMISFRACTION_SH_LAND` | - | 0.21 | - | Fraction of SH that is land |

### 4.7 El Nino Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Apply El Nino | `CORE_ELNINO_APPLY` | - | 0 | 0,1 | Enable El Nino variability |
| El Nino scale | `CORE_ELNINO_SCALE` | K | 0.0108 | 0-0.05 | Scaling of Nino3.4 index effect |
| El Nino relax factor | `CORE_ELNINO_RELAXFACTOR` | - | 0.0265 | 0-0.1 | Relaxation rate for heat anomaly |

### 4.8 AMV (Atlantic Multidecadal Variability) Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Apply AMV | `CORE_AMV_APPLY` | - | 0 | 0,1 | Enable AMV effect on upwelling |
| AMV scale | `CORE_AMV_SCALE` | K | -4.27 | - | Scaling of AMV index effect |
| AMV SH fraction | `CORE_AMV_SHFRACTION` | - | 0.0 | 0-1 | Fraction of AMV effect in SH |

## 5. Inputs (per timestep)

| Variable | Units | Source Module | Required? | Fortran Variable |
|----------|-------|---------------|-----------|------------------|
| Total effective radiative forcing (4 boxes) | W/m^2 | Radiative Forcing | Yes | `Q(1:4)` via `dat_total_effrf` |
| Volcanic forcing (4 boxes) | W/m^2 | Radiative Forcing | Optional | `dat_volcanic_effrf` |
| El Nino 3.4 index | - | External file | If ELNINO_APPLY=1 | `DAT_ELNINO_N34` |
| AMV index | - | External file | If AMV_APPLY=1 | `DAT_AMV_INDEX` |
| Prescribed upwelling rate | m/yr | External file | If method="PRESCRIBED" | `DAT_UPWELLING_RATE` |
| Prescribed surface temperature | K | External file | If PRESCRTEMP_APPLY=1 | `DAT_RAW_SURFACE_TEMP` |

## 6. Outputs (per timestep)

| Variable | Units | Destination Module(s) | Fortran Variable |
|----------|-------|----------------------|------------------|
| Surface temperature (4 boxes) | K | Carbon cycle, Sea level, Methane | `DAT_SURFACE_TEMP` |
| Annual mean temperature (4 boxes) | K | Output | `DAT_SURFACE_ANNUALMEANTEMP` |
| Mixed layer temperature (4 boxes) | K | Output | `DAT_SURFACE_MIXEDLAYERTEMP` |
| Global temperature | K | Many modules | `DAT_SURFACE_TEMP%DATGLOBE` |
| Ocean layer temperatures | K | Sea level (thermal expansion) | `TEMP_OCEANLAYERS` |
| Heat uptake (energy balance) | W/m^2 | Output | `DAT_HEATUPTAKE_EBALANCE_TOTAL` |
| Heat content (energy balance) | 10^22 J | Output | `DAT_HEATCONTENT_EBALANCE_TOTAL` |
| Heat content by depth | 10^22 J | Output | `DAT_HEATCONTENT_AGGREG(1:4)` |
| Upwelling rate | m/yr | Output | `DAT_UPWELLING_RATE` |
| Equilibrium climate sensitivity | K | Output | `DAT_CLIMSENS_EQ` |
| Effective climate sensitivity | K | Output | `DAT_CLIMSENS_EFF` |
| Ground temperature | K | Output | `DAT_GROUND_TEMP` |
| Sub-annual temperatures | K | Output | `DAT_SURFACE_TEMP_SUBANNUAL` |

## 7. Algorithm (Pseudocode)

### 7.1 Initialization (ocean_init)

```
1. Calculate timestep: DT = 1 / STEPSPERYEAR
2. Set layer spacing: DZ = 100m, DZ1 = DZ/2
3. Initialize all temperatures to 0
4. Calculate depth-dependent ocean areas (interpolate from raw data)
5. Calculate area factors (top flow, bottom flow, diff flow, average)
6. Set initial upwelling rate
7. Calculate initial temperature profiles (exponential decay with depth)
```

### 7.2 Annual Timestep (magicc_step_year)

```
FOR each year:
    1. Calculate radiative forcing from DELTAQ

    2. Calculate cumulative-T adjusted climate sensitivity

    3. Call LAMCALC to get ocean/land feedback parameters

    FOR each sub-annual step (1 to STEPSPERYEAR):
        4. Get interpolated forcing Q for this timestep

        5. Add volcanic forcing if applicable

        FOR each hemisphere (1=NH, 2=SH):
            6. Calculate depth-dependent vertical diffusivity

            7. Calculate land feedback denominator

            8. Build tridiagonal matrix:
               a. Mixed layer (layer 1): A(1), B(1), D(1)
               b. Layer 2 (below mixed layer): A(2), B(2), C(2), D(2)
               c. Middle layers: A(l), B(l), C(l), D(l) for l=3 to N-1
               d. Bottom layer: A(N), B(N), D(N)

            9. Add El Nino terms to D(1) if enabled

            10. Add ground heat capacity terms to D(1) if enabled

            11. Solve tridiagonal system (Thomas algorithm)

        END FOR hemisphere

        12. Apply temperature safety cap (25 K max)

        13. Convert ocean SST to air temperature (alpha + gamma*T^2)

        14. Calculate land temperatures (equilibrium with ocean)

        15. Apply temperature safety caps for land

        16. Store sub-annual temperatures

        17. Calculate heat uptake from energy balance: Q - lambda*T

        18. Update hemispheric heat exchange terms

        19. Update upwelling rate based on temperature

    END FOR step

    20. Calculate annual mean temperatures

    21. Store ocean layer profiles

    22. Calculate heat content by depth

    23. Call sea level calculation

END FOR year
```

### 7.3 LAMCALC Algorithm

```
INPUTS: Q_2x, K_lo, K_ns, ECS, RLO

1. Initialize: lambda_global = Q_2x / ECS
2. Set initial guess: lambda_o(1) = lambda_global
3. Set step size: DLAMO = 0.7

ITERATE (max 40 iterations):
    4. Calculate lambda_l from constraint:
       lambda_l = lambda_global + f_o/f_l * (lambda_global - lambda_o) / RLO

    5. Build 4x4 matrix A (land-ocean-hemisphere coupling)

    6. Invert matrix: B = A^(-1)

    7. Calculate equilibrium temperatures for each box:
       T_box = B * Q (with regional forcing patterns)

    8. Calculate estimated RLO from temperatures:
       RLO_est = T_land_mean / T_ocean_mean

    9. Check convergence: |RLO - RLO_est| < 0.001
       If converged: EXIT

    10. Update lambda_o using secant method:
        - If sign change: interpolate
        - Otherwise: step in direction of improvement

END ITERATE

11. Output: lambda_o_best, lambda_l_best

12. Calculate internal efficacies for all forcing agents
```

## 8. Numerical Considerations

### 8.1 Implicit Scheme Stability

The fully implicit (backward Euler) discretization of the ocean diffusion-advection equation is unconditionally stable. The CFL condition for the explicit scheme would be:

$$\Delta t < \frac{(\Delta z)^2}{2K}$$

With K = 0.75 cm^2/s = 2367 m^2/yr and DZ = 100m, explicit stability would require dt < 2.1 years - violated by annual timesteps. The implicit scheme avoids this constraint.

### 8.2 Tridiagonal Solver Stability

The Thomas algorithm is stable when the matrix is diagonally dominant:

$$|B_l| > |A_l| + |C_l|$$

This is guaranteed by the positive diffusivity and upwelling terms on the diagonal.

### 8.3 Iteration Convergence in LAMCALC

The LAMCALC iteration uses a hybrid secant/bisection method:

- Initial step size DLAMO = 0.7 is empirically chosen
- Too large (>0.9): oscillates around solution
- Too small (<0.4): exceeds 40 iteration limit
- Convergence tolerance: |RLO - RLO_est| < 0.001

### 8.4 Sub-Annual Stepping

Monthly sub-stepping (STEPSPERYEAR = 12) provides:

1. Stable integration of ocean-atmosphere coupling
2. Capture of El Nino variability on sub-annual timescales
3. Proper averaging for annual mean temperatures

### 8.5 Temperature Safety Caps

Hard limits prevent runaway:

- `CORE_MAXIMAL_TEMPERATURE = 25 K` for all temperatures
- Warning logged if caps are hit
- Indicates physically unreasonable parameter combinations

## 9. Issues and Concerns

### 9.1 Code Organization

**CRITICAL ISSUE**: The vast majority of climate physics is implemented directly in `MAGICC7.f90` (lines ~2700-3700), NOT in the dedicated `climate_and_ocean.f90` module. The module file primarily contains:

- Variable declarations
- Parameter storage
- The `ocean_init` subroutine

The actual integration loop, tridiagonal solve, temperature calculations, and upwelling updates are all in the main file. This violates modularity and makes the code difficult to maintain or swap out.

### 9.2 Hardcoded Values

Several values are hardcoded rather than parameterized:

- `DZ = 100.0` m (layer thickness) - not configurable
- `IMAX = 40` (max LAMCALC iterations) - hardcoded
- `DLAMO = 0.7` (iteration step size) - hardcoded
- `DIFFLIM = 0.001` (convergence tolerance) - hardcoded
- `CORE_MAXIMAL_TEMPERATURE = 25.0` - set in code, overriding config
- Ocean area profiles - embedded in code, though option to read from file exists

### 9.3 4-Box Structure

The 4-box atmospheric structure is deeply embedded throughout:

- Array dimensions (4) are hardcoded
- Box indices (1=NHO, 2=NHL, 3=SHO, 4=SHL) assumed in multiple places
- Coupling matrix is explicitly 4x4
- Cannot easily extend to higher resolution

### 9.4 Numerical Scheme Lock-In

The fully implicit scheme cannot be easily swapped:

- Tridiagonal structure assumed throughout
- No option for explicit or Crank-Nicolson schemes
- Area factors pre-computed for implicit scheme only

### 9.5 Mixed Units

The code uses mixed unit systems requiring careful conversion:

- Diffusivity: input in cm^2/s, converted to m^2/yr (factor 3155.76)
- Heat capacity: calculated in W*yr/m^3/K
- Heat content: converted to 10^22 J using `WATTYEARPERM2_2_JOULEE22`
- Temperatures: anomalies in K (not absolute)

### 9.6 Complex Coupling

The land temperature calculation assumes instantaneous equilibrium with ocean, which is inconsistent with the sub-annual ocean evolution. The land heat capacity option partially addresses this but adds complexity.

### 9.7 El Nino "Parallel Box" Model

The El Nino implementation uses a conceptually unclear "double delta box" that:

- Is not physically a separate ocean region
- Has same area and temperature as mixed layer
- Uses relaxation rather than explicit heat exchange
- Implementation is difficult to validate against observations

### 9.8 Variable Upwelling Complications

Variable upwelling requires tracking:

- `DELTA_CURRENT_UPWELLING_RATE` - change from initial
- Initial temperature profiles for each profile type (3 options)
- Separate handling for mixed layer vs. deeper layers

### 9.9 LAMCALC Complexity

The LAMCALC routine:

- Uses iterative matrix inversion (not direct solution)
- Calculates internal efficacies for ~20 forcing agents
- Is called every year (potentially every sub-step)
- Could be simplified if land-ocean warming ratio was derived post-hoc

### 9.10 Dead Code and Comments

Several commented-out code blocks remain:

- Lines 119-135 in `climate_and_ocean.f90` (old initialization)
- Lines 3420-3422 in `MAGICC7.f90` (wrong annual mean calculation)
- Multiple "PROCEED HERE" comments indicating incomplete work

## 10. Test Cases

### 10.1 Equilibrium Climate Sensitivity Test

**Purpose**: Verify that the model achieves the prescribed ECS under 2xCO2.

**Setup**:

- Step forcing: Q = 3.71 W/m^2 at t=0
- Run for 5000 years
- `CORE_CLIMATESENSITIVITY = 3.0`

**Expected**:

- Global temperature approaches 3.0 K
- Land/ocean ratio approaches `CORE_RLO`
- Heat uptake approaches 0

### 10.2 TCR Test (1% CO2/yr)

**Purpose**: Verify Transient Climate Response.

**Setup**:

- Forcing increases as: Q(t) = 3.71 * log2(1.01^t)
- Year 70: Q = 3.71 W/m^2

**Expected**:

- T(year 70) should match published TCR values for given parameters
- TCR/ECS ratio typically 0.4-0.7

### 10.3 Ocean Heat Content Test

**Purpose**: Verify heat uptake and content calculations.

**Setup**:

- Constant forcing Q = 3.71 W/m^2
- Run 100 years

**Expected**:

- Heat content increase = integral of (Q - lambda*T)
- Energy conservation: dH/dt + lambda*T = Q

### 10.4 Upwelling Shutdown Test

**Purpose**: Test variable upwelling behavior.

**Setup**:

- Very high forcing (10 W/m^2)
- `CORE_UPWELL_THRESH_TEMP_NH = 4.0`

**Expected**:

- Upwelling decreases to (1-CORE_UPWELLING_VARIABLE_PART) * initial
- Deep ocean warming accelerates
- Eventually stabilizes at floor

### 10.5 Land-Ocean Warming Ratio Test

**Purpose**: Verify RLO is achieved.

**Setup**:

- Step forcing
- Run to equilibrium

**Expected**:

- T_land / T_ocean = CORE_RLO at equilibrium
- Transient ratio may differ

### 10.6 Hemispheric Heat Exchange Test

**Purpose**: Verify inter-hemispheric coupling.

**Setup**:

- NH-only forcing (set Q(3:4) = 0)

**Expected**:

- SH warms due to heat transport
- Eventually both hemispheres equilibrate

### 10.7 Conservation Test

**Purpose**: Verify energy conservation.

**Setup**:

- Any forcing scenario

**Expected**:

- Sum of (heat uptake + outgoing IR) = total forcing
- No spurious heat sources/sinks

### 10.8 Temperature Profile Initialization Test

**Purpose**: Verify initial ocean profile options.

**Setup**:

- Test CORE_SWITCH_OCN_TEMPPROFILE = 0, 1, 2

**Expected**:

- Profile 0: Analytical exponential profile
- Profile 1: Prescribed exponential profile
- Profile 2: CMIP5-based profile
- All should give similar long-term behavior

## 11. Fortran Code References

### 11.1 Primary Files

| File | Lines | Content |
|------|-------|---------|
| `src/libmagicc/physics/climate_and_ocean.f90` | 1-295 | Module variables, ocean_init |
| `src/libmagicc/physics/core_calculations.f90` | 1-110 | Ocean-atmosphere T conversion |
| `src/libmagicc/MAGICC7.f90` | 2646-3727 | Main integration loop (magicc_step_year) |
| `src/libmagicc/MAGICC7.f90` | 8070-8278 | LAMCALC subroutine |
| `src/libmagicc/MAGICC7.f90` | 8291-8338 | CALC_INTERNAL_EFFICACY |
| `src/libmagicc/allcfgs.f90` | 1-431 | Parameter namelist definitions |
| `src/libmagicc/utils/areas.f90` | 1-8 | Area fraction variables |

### 11.2 Key Code Sections

| Section | File | Lines | Description |
|---------|------|-------|-------------|
| Ocean initialization | climate_and_ocean.f90 | 102-292 | Initialize ocean model |
| Heat capacity calculation | MAGICC7.f90 | 237-246 | HEAT_CAPACITY_PERM setup |
| Area fraction setup | MAGICC7.f90 | 249-269 | FGNO, FGNL, etc. |
| Forcing interpolation | MAGICC7.f90 | 2724 | Get forcing for timestep |
| Climate sensitivity adjustment | MAGICC7.f90 | 2747-2768 | Cumulative-T feedback |
| Vertical diffusivity | MAGICC7.f90 | 2783-2799 | Depth/T-dependent K |
| Tridiagonal matrix setup | MAGICC7.f90 | 2826-3074 | A, B, C, D coefficients |
| Thomas algorithm | MAGICC7.f90 | 3080-3094 | Tridiagonal solve |
| Temperature caps | MAGICC7.f90 | 3102-3112 | Safety limits |
| Ocean-to-air conversion | MAGICC7.f90 | 3159-3187 | Alpha-gamma adjustment |
| Land temperature | MAGICC7.f90 | 3214-3223 | Equilibrium calculation |
| Heat uptake | MAGICC7.f90 | 3258-3281 | Energy balance calculation |
| Upwelling scaling | MAGICC7.f90 | 3300-3388 | Variable upwelling |
| Heat content calculation | MAGICC7.f90 | 3627-3721 | Depth-integrated heat |
| LAMCALC iteration | MAGICC7.f90 | 8116-8207 | Lambda iteration loop |
| Matrix inversion | MAGICC7.f90 | 8142 | 4x4 inverse call |

### 11.3 Physical Constants

| Constant | Value | Location | Description |
|----------|-------|----------|-------------|
| RHO | 1.026D0 | climate_and_ocean.f90:67 | Water density (10^6 g/m^3) |
| SPECHT | 0.9333D0 | climate_and_ocean.f90:67 | Specific heat (cal/g/K) |
| HTCONS | 4.1856D0 | climate_and_ocean.f90:67 | Cal to Joule conversion |
| 3155.76D0 | - | MAGICC7.f90:2789 | cm^2/s to m^2/yr |
| 31.5576D0 | - | MAGICC7.f90:241 | Seconds per year / 10^6 |
| 5.101D0 | - | MAGICC7.f90:244 | Earth surface area (10^14 m^2) |

### 11.4 Configuration File

| File | Description |
|------|-------------|
| `run/MAGCFG_DEFAULTALL.CFG` | Default parameter values (lines 81-139 for CORE_* parameters) |
