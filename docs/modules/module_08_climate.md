# Climate

## Overview

The climate module turns radiative forcing into temperature.
It couples an upwelling-diffusion-entrainment ocean to a four-box atmosphere:
Northern Ocean, Northern Land, Southern Ocean and Southern Land.
It accounts for:

- Ocean heat uptake, which delays the response,
  so the transient response is smaller than the equilibrium one.
- Faster warming over land than over ocean.
- Different responses in the two hemispheres.
- Upwelling that weakens as the climate warms.
- Deep-ocean heat uptake by vertical diffusion and advection through a multi-layer column.

The model resolves both the fast response of the atmosphere and mixed layer
and the slow response of the deep ocean.

## RSCM implementation

- **Component**: `ClimateUDEB`, built from Python with `rscm.magicc.ClimateUDEBBuilder`.
- **Source**: `crates/rscm-magicc/src/climate/udeb/` holds the substep loop (`mod.rs`)
  and the ocean column (`ocean_column.rs`).
  LAMCALC is in `climate/lamcalc.rs`.
  The tridiagonal solve is `thomas_solve` in `crates/rscm-core/src/utils/linear_algebra.rs`.
- **Status**: Implemented. See the [status table](README.md#rscm-implementation-status).
- **Tests**:
  - `tests/regression/test_full_run.py::test_climate` drives `ClimateUDEB` with MAGICC7's own box forcing.
  - `tests/regression/test_full_run.py::test_coupled_run` runs the coupled chain,
    with variants on climate sensitivity, upwelling, diffusivity, ocean geometry, ground heat,
    inter-hemispheric exchange, feedback and efficacy.
  - Rust tests in `crates/rscm-magicc/tests/` (`lamcalc_integration.rs`, `lamcalc_reference_values.rs`,
    `ecs_diagnostics.rs`, `conservation.rs`) check LAMCALC and energy conservation.

`ClimateUDEB` follows MAGICC7 in these details, which are easy to get wrong:

- LAMCALC uses the regional CO₂ forcing pattern `RF_REGIONS_CO2 = [1.4089, 1.37045, 1.43333, 1.33257]`
  from `MAGCFG_DEFAULTALL.CFG`.
- $\alpha_{eff}$ is updated every monthly substep from the previous substep's SST.
  Holding it fixed for the year leaves a cold drift of about $10^{-5}$ K by 2000.
- The CO₂ internal efficacy comes from LAMCALC.
  With `efficacy_method = "ar6"`, effective forcing is scaled by `co2_efficacy / internal_efficacy`.
  `ForcingAggregation` applies an internal efficacy to every other agent.
  As in MAGICC7, they are recalculated each year from the previous year's climate sensitivity,
  so they drift as the climate warms.
- Monthly volcanic forcing enters through the `subannual_forcing` parameter.
  `rscm.magicc.prescribed.volcanic_pattern` reproduces MAGICC7's regional pattern,
  which is derived from the run itself:
  the mean monthly box forcing, scaled by `RF_VOLCANIC_SCALE`,
  from the first year to `RF_VOLC_FORC_PATTERN_MAX_LAST_YEAR`, less its global mean and 1.
  Substeps 1 to 11 take the previous year's volcanic internal efficacy and substep 12 the current one.
  The first year keeps the initial value.

## Equations

### Energy balance

Each box satisfies

$$
C \frac{dT}{dt} = Q - \lambda T + H_{exchange}
$$

where:

- $C$ is the effective heat capacity (J m⁻² K⁻¹).
- $T$ is the temperature anomaly (K).
- $Q$ is the radiative forcing (W m⁻²).
- $\lambda$ is the feedback parameter (W m⁻² K⁻¹).
- $H_{exchange}$ is the heat exchange with other boxes.

Each ocean box sits on a multi-layer ocean column (50 layers by default).
Land boxes have no column of their own.

### Four-box coupling matrix

At equilibrium the four boxes satisfy $\mathbf{A}\,\mathbf{T} = \mathbf{F}\,\mathbf{Q}$,
where $\mathbf{Q}$ holds each box's forcing and $\mathbf{F} = \mathrm{diag}(f_{NO}, f_{NL}, f_{SO}, f_{SL})$,
with the boxes ordered Northern Ocean, Northern Land, Southern Ocean, Southern Land:

$$
\mathbf{A} =
\begin{pmatrix}
f_{NO}\lambda_o + \alpha K_{lo} + K_{ns} & -K_{lo} & -K_{ns} & 0 \\
-\alpha K_{lo} & f_{NL}\lambda_l + K_{lo} & 0 & 0 \\
-K_{ns} & 0 & f_{SO}\lambda_o + \alpha K_{lo} + K_{ns} & -K_{lo} \\
0 & 0 & -\alpha K_{lo} & f_{SL}\lambda_l + K_{lo}
\end{pmatrix}
$$

Land exchanges heat only with the ocean in its hemisphere,
and the hemispheres exchange heat only through their oceans.

- $K_{lo}$ is `CORE_HEATXCHANGE_LANDOCEAN`, the land-ocean heat exchange coefficient (W m⁻² K⁻¹).
- $K_{ns}$ is `CORE_HEATXCHANGE_NORTHSOUTH`, the inter-hemispheric exchange coefficient (W m⁻² K⁻¹).
- $\alpha$ is `CORE_AMPLIFY_OCN2LAND_HEATXCHNG`, the amplification of ocean-to-land heat flux.
- $\lambda_o$ and $\lambda_l$ are the ocean and land feedback parameters.
- $f_{NO}$, $f_{NL}$, $f_{SO}$ and $f_{SL}$ are the global area fractions of each box
  (`FGNO`, `FGNL`, `FGSO`, `FGSL`).
  The initialisation page gives the [box order and area fractions](module_00_initialisation.md#four-box-geography).

### Climate sensitivity decomposition

LAMCALC splits the equilibrium climate sensitivity (`CORE_CLIMATESENSITIVITY`)
into $\lambda_o$ and $\lambda_l$, so that the land-ocean warming ratio `CORE_RLO` holds:

$$
\lambda = \frac{Q_{2x}}{\Delta T_{2x}},
\qquad
\lambda_l = \lambda + \frac{f_o}{f_l} \frac{\lambda - \lambda_o}{RLO},
\qquad
RLO = \frac{\Delta T_l}{\Delta T_o}
$$

Here $f_o$ and $f_l$ are the total ocean and land fractions.
The second relation is the global energy balance with land warming $RLO$ times as much as ocean.
LAMCALC iterates on $\lambda_o$ until the equilibrium of the coupling matrix
gives the target $RLO$ to within 0.001.

The equilibrium forcing in each box follows the regional CO₂ pattern (MAGICC7.f90 lines 8146 to 8275):

$$
q_i = \frac{r_i}{\sum_j r_j f_j}, \qquad Q_i = Q_{2x}\, q_i
$$

where $r_i$ is `RF_REGIONS_CO2` for box $i$.
After LAMCALC converges, MAGICC7 repeats the equilibrium calculation with each agent's regional pattern.
The ratio of the global response to the CO₂ response at the same global-mean forcing
is that agent's internal efficacy.

### Time-varying climate sensitivity

Climate sensitivity varies with forcing and with cumulative temperature:

$$
ECS_{eff} = ECS \left(1 + \alpha_T \frac{\sum T - ECS \cdot P}{ECS \cdot P}\right)
\left(1 + \alpha_Q \left(\max(0, Q) - Q_{2x}\right)\right)
$$

- $\alpha_Q$ is `CORE_FEEDBACK_QSENSITIVITY` (per W m⁻²).
- $\alpha_T$ is `CORE_FEEDBACK_CUMTSENSITIVITY` (dimensionless).
- $P$ is `CORE_FEEDBACK_CUMTPERIOD` (yr), the period over which $\sum T$ (K yr) is summed.

### Mixed layer

The mixed layer (layer 1) of each hemisphere satisfies

$$
C_{mix} \frac{dT_1}{dt} = Q_{eff} - \Lambda\, T_1 + F_{diffusion} + F_{upwelling} + F_{exchange}
$$

with the coupled land-ocean feedback

$$
\Lambda = \frac{\alpha_{eff}}{C}
\left(\lambda_o + \frac{\alpha \lambda_l K_{lo} f_l}{f_o (K_{lo} + f_l \lambda_l)}\right)
$$

- $C_{mix}$ is `HEAT_CAPACITY_PERM * CORE_MIXEDLAYER_DEPTH` (W yr m⁻² K⁻¹).
- $\alpha_{eff}$ is the ratio of ocean air temperature to SST
  (see [ocean-to-atmosphere amplification](#ocean-to-atmosphere-amplification)).
- $f_o$ and $f_l$ are the ocean and land fractions of the hemisphere's boxes.
- $F_{diffusion}$, $F_{upwelling}$ and $F_{exchange}$ are the diffusive, advective
  and land-ocean plus inter-hemispheric fluxes.

The inter-hemispheric exchange uses ocean air temperatures, not SST,
and is divided by the hemisphere's ocean fraction:

$$
H_{NS,N} = \frac{K_{ns}}{f_{NO}}\left(T^{air}_{SO} - T^{air}_{NO}\right),
\qquad
H_{NS,S} = \frac{K_{ns}}{f_{SO}}\left(T^{air}_{NO} - T^{air}_{SO}\right)
$$

### Ocean diffusion and advection

Below the mixed layer:

$$
\frac{\partial T}{\partial t}
= \frac{1}{A(z)} \frac{\partial}{\partial z}\left[A(z) K \frac{\partial T}{\partial z}\right]
- w \frac{\partial T}{\partial z}
+ \frac{w \pi (T_1 - T_{cold})}{A(z)} \frac{dA}{dz}
$$

- $K$ is the vertical diffusivity (m² yr⁻¹).
- $w$ is the upwelling rate (m yr⁻¹).
- $A(z)$ is the ocean area at depth $z$.
- $\pi$ is `CORE_POLARSINKWATER_TEMPRATIO`,
  the fraction of surface temperature carried by polar sinking water.
- $T_{cold}$ is the temperature of polar sinking water.

The last term is entrainment as the basin narrows with depth.

### Temperature-dependent diffusivity

The diffusivity at the base of layer $l$ ($l = 1, \dots, n-1$) depends on the top-to-bottom gradient:

$$
K_l = \max\left(K_{min},\ K_0 + \frac{dK}{dT}\left(1 - \frac{l-1}{n-1}\right)(T_1 - T_n)\right)
$$

- $K_0$ is `CORE_VERTICALDIFFUSIVITY`, $K_{min}$ is `CORE_VERTICALDIFFUSIVITY_MIN`
  and $dK/dT$ is `CORE_VERTICALDIFF_TOP_DKDT`.
  All are given in cm² s⁻¹ and converted to m² yr⁻¹ by the factor 3155.76.
- The scaling uses the boundary index, not depth, so the thicker mixed layer is ignored.

### Tridiagonal system

The implicit discretisation gives a tridiagonal system for each hemisphere:

$$
A_l T_{l-1}^{n+1} + B_l T_l^{n+1} + C_l T_{l+1}^{n+1} = D_l
$$

With substep $\Delta t$, mixed-layer depth $h$,
layer thickness $\Delta z = 100$ m, $\Delta z_1 = \Delta z / 2$,
area factors $a^t_l = A^t_l / \bar{A}_l$, $a^b_l = A^b_l / \bar{A}_l$
and $a^d_l = (A^t_l - A^b_l) / \bar{A}_l$ from the ocean area at the top and bottom of layer $l$
and their mean $\bar{A}_l$,
initial profile $T^0_l$, polar sinking temperature $T_p$ and upwelling change $\Delta w = w - w_0$,
the coefficients are, for the mixed layer:

$$
\begin{aligned}
B_1 &= 1 + \frac{\Delta t}{h}
  \left[\left(\frac{K_1}{\Delta z_1} + \pi w\right) a^b_1 + \Lambda\, a^t_1\right] \\
C_1 &= -\frac{\Delta t}{h}\left(\frac{K_1}{\Delta z_1} + w\right) a^b_1 \\
D_1 &= T_1 + \frac{\Delta t}{h c} F\, a^t_1 + \frac{\Delta t}{h} \Delta w \left(T^0_2 - T_p\right) a^b_1
\end{aligned}
$$

where $c$ is the heat capacity per metre and $F$ is the forcing reaching the mixed layer,

$$
F = \frac{K_{lo} f_l Q_l}{f_o (K_{lo} + f_l \lambda_l)} + Q_o + H_{NS}
$$

For the interior layers $l = 2, \dots, n-1$,
with $\Delta z' = \Delta z_1$ for $l = 2$ and $\Delta z$ otherwise:

$$
\begin{aligned}
A_l &= -\frac{\Delta t}{\Delta z} \frac{K_{l-1}}{\Delta z'} a^t_l \\
B_l &= 1 + \frac{\Delta t}{\Delta z}
  \left[\left(\frac{K_{l-1}}{\Delta z'} + w\right) a^t_l + \frac{K_l}{\Delta z} a^b_l\right] \\
C_l &= -\frac{\Delta t}{\Delta z}\left(\frac{K_l}{\Delta z} + w\right) a^b_l \\
D_l &= T_l + \frac{\Delta t}{\Delta z}\left[\Delta w \left(T^0_{l+1} a^b_l - T^0_l a^t_l\right)
      + a^d_l \left(\pi w T_1 + \Delta w\, T_p\right)\right]
\end{aligned}
$$

and for the bottom layer:

$$
\begin{aligned}
A_n &= -\frac{\Delta t}{\Delta z} \frac{K_{n-1}}{\Delta z} a^t_n \\
B_n &= 1 + \frac{\Delta t}{\Delta z}\left(\frac{K_{n-1}}{\Delta z} + w\right) a^t_n \\
D_n &= T_n + \frac{\Delta t}{\Delta z} a^t_n \left[\pi w T_1 + \Delta w \left(T_p - T^0_n\right)\right]
\end{aligned}
$$

With the ground reservoir on, $D_1$ also loses
$\frac{\Delta t\, a^t_1}{h f_o c} K_{lg} (T_{land} - T_{ground})$.

The Fortran names the mixed-layer diagonal `A(1)` and its super-diagonal `B(1)`,
because the first row has no sub-diagonal.
Layers 2 to $n$ use `A`, `B` and `C` for the sub-diagonal, diagonal and super-diagonal.

### Thomas algorithm

The system is solved by forward elimination and back substitution
(MAGICC7.f90 lines 3080 to 3094), written here in the Fortran's naming:

```fortran
AA(1) = -B(1) / A(1)
BB(1) = D(1) / A(1)
DO L = 2, N-1
    VV = A(L) * AA(L-1) + B(L)
    AA(L) = -C(L) / VV
    BB(L) = (D(L) - A(L) * BB(L-1)) / VV
END DO
T(N) = (D(N) - A(N)*BB(N-1)) / (A(N)*AA(N-1) + B(N))
DO I = 1, N-1
    L = N - I
    T(L) = AA(L) * T(L+1) + BB(L)
END DO
```

### Temperature-dependent upwelling

Upwelling weakens with warming:

$$
w = \max\left(w_0 (1 - f_{var}),\ w_0\left(1 - f_{var} \frac{T_{key}}{T_{thresh}}\right)\right)
$$

- $w_0$ is `CORE_INITIAL_UPWELLING_RATE` (m yr⁻¹).
- $f_{var}$ is `CORE_UPWELLING_VARIABLE_PART`, the fraction that can vary.
- $T_{thresh}$ is `CORE_UPWELL_THRESH_TEMP_NH` or `CORE_UPWELL_THRESH_TEMP_SH`.
- $T_{key}$ depends on `CORE_UPWELLING_SCALING_METHOD`.
  `GLOBE`, the default, uses the area-weighted global air temperature.
  `OCEAN` and `HEMISPHERIC` use ocean temperatures, `NOSCALING` keeps $w_0$
  and `PRESCRIBED` reads the rate from a file.

### Ocean-to-atmosphere amplification

Air temperature over the ocean differs from SST:

$$
T^{air} =
\begin{cases}
\alpha_T T_{SST} + \gamma T_{SST}^2 & T_{SST} < T^* \\
T_{SST} + \delta_{max} & T_{SST} \ge T^*
\end{cases}
$$

with $T^* = -(\alpha_T - 1)/(2\gamma)$ and $\delta_{max} = \alpha_T T^* + \gamma (T^*)^2 - T^*$.
$\alpha_T$ is `CORE_TEMPADJUST_OCN2ATM_ALPHA` and $\gamma$ is `CORE_TEMPADJUST_OCN2ATM_GAMMA`.
Air over the ocean warms more than the SST.
The ratio $\alpha_{eff} = T^{air}/T_{SST}$ enters the next substep's mixed-layer feedback.

### Land temperature

Land is in equilibrium with the ocean of its hemisphere:

$$
T_{land} = \frac{f_l Q_l + K_{lo} \alpha T^{air}_{ocean}}{f_l \lambda_l + K_{lo}}
$$

### Ground heat reservoir

With `CORE_LANDHEATCAPACITY_APPLY = 1`, a ground reservoir damps the land response:

$$
\frac{dT_{ground}}{dt} = \frac{K_{lg} (T_{land} - T_{ground})}{f_l\, c\, d_{eff}}
$$

$K_{lg}$ is `CORE_HEATXCHANGE_LANDGROUND` and $d_{eff}$ is `CORE_LANDHC_EFFTHICKNESS` (m).
It is stepped by forward Euler from the previous substep's land temperature.

## State variables

| Variable                        | Fortran name                        | Units  | Initial value                 |
| ------------------------------- | ----------------------------------- | ------ | ----------------------------- |
| Ocean layer temperatures (NH)   | `OCN_HEMISPHERIC_LAYERTEMPS(1,:)`   | K      | 0                             |
| Ocean layer temperatures (SH)   | `OCN_HEMISPHERIC_LAYERTEMPS(2,:)`   | K      | 0                             |
| Four-box air temperatures       | `CURRENT_TIME_TEMPERATURE(1:4)`     | K      | 0                             |
| Mixed-layer temperatures        | `CURRENT_TIME_MIXEDLAYERTEMP(1:4)`  | K      | 0                             |
| Ground temperatures             | `GROUND_HEMISPHERIC_TEMPS(1:2)`     | K      | 0                             |
| Upwelling rate (NH, SH)         | `CURRENT_UPWELLING_RATE(1:2)`       | m yr⁻¹ | `CORE_INITIAL_UPWELLING_RATE` |
| Change in upwelling             | `DELTA_CURRENT_UPWELLING_RATE(1:2)` | m yr⁻¹ | 0                             |
| El Niño heat box                | `ELNINO_N34_DOUBLEDELTABOX`         | K      | 0                             |
| Inter-hemispheric heat exchange | `HEMISPHERIC_HEATXCHANGE(1:2)`      | W m⁻²  | 0                             |

## Inputs and outputs

| Input                       | Units | Source             | Fortran variable                    |
| --------------------------- | ----- | ------------------ | ----------------------------------- |
| Total effective forcing     | W m⁻² | Forcing aggregation | `dat_total_effrf`, 4 boxes         |
| Volcanic forcing            | W m⁻² | Forcing            | `dat_volcanic_effrf`, 4 boxes       |
| El Niño 3.4 index           | -     | File               | `DAT_ELNINO_N34`, if enabled        |
| AMV index                   | -     | File               | `DAT_AMV_INDEX`, if enabled         |
| Prescribed upwelling rate   | m yr⁻¹ | File              | `DAT_UPWELLING_RATE`, if prescribed |
| Prescribed temperature      | K     | File               | `DAT_RAW_SURFACE_TEMP`, if enabled  |

| Output                          | Units  | Used by                         | Fortran variable                 |
| ------------------------------- | ------ | ------------------------------- | -------------------------------- |
| Surface temperature, 4 boxes    | K      | Carbon cycle, sea level, CH4    | `DAT_SURFACE_TEMP`               |
| Annual mean temperature         | K      | Sea level                       | `DAT_SURFACE_ANNUALMEANTEMP`     |
| Mixed-layer temperature         | K      | Output                          | `DAT_SURFACE_MIXEDLAYERTEMP`     |
| Ocean layer temperatures        | K      | Sea level (thermal expansion)   | `TEMP_OCEANLAYERS`               |
| Heat uptake                     | W m⁻²  | Output                          | `DAT_HEATUPTAKE_EBALANCE_TOTAL`  |
| Heat content                    | 10²² J | Output                          | `DAT_HEATCONTENT_EBALANCE_TOTAL` |
| Heat content by depth           | 10²² J | Output                          | `DAT_HEATCONTENT_AGGREG(1:4)`    |
| Upwelling rate                  | m yr⁻¹ | Output                          | `DAT_UPWELLING_RATE`             |
| Equilibrium climate sensitivity | K      | Output                          | `DAT_CLIMSENS_EQ`                |
| Effective climate sensitivity   | K      | Output                          | `DAT_CLIMSENS_EFF`               |
| Ground temperature              | K      | Output                          | `DAT_GROUND_TEMP`                |
| Monthly temperatures            | K      | Output                          | `DAT_SURFACE_TEMP_SUBANNUAL`     |

`ClimateUDEB` reads `Effective Radiative Forcing` and writes `Surface Air Temperature Change` (four-box),
`Heat Uptake`, `Ocean Heat Content`, `Sea Surface Temperature` and `Equilibrium Climate Sensitivity`.
`Ocean Heat Content` is the global integral in J,
over both hemispheres' ocean area (Earth surface area 5.101 × 10¹⁴ m²).
With `thermal_expansion` set it also writes `Annual Mean Surface Air Temperature Change` for `SeaLevelRise`.

## Parameters

Defaults are MAGICC7 v7.5.3's, from `run/MAGCFG_DEFAULTALL.CFG`.
`run_magicc7` takes them from `python/rscm/magicc/data/magicc7/defaults.json`.
`ClimateUDEBParameters::default()` rounds some of them (for example `CORE_RLO` to 1.317).

### Climate sensitivity

| Parameter                          | Fortran name                    | Units    | Default      |
| ---------------------------------- | ------------------------------- | -------- | ------------ |
| Equilibrium climate sensitivity    | `CORE_CLIMATESENSITIVITY`       | K        | 3.0          |
| Forcing for doubled CO₂            | `CORE_DELQ2XCO2`                | W m⁻²    | 3.71         |
| Land-ocean warming ratio           | `CORE_RLO`                      | -        | 1.317229     |
| Forcing sensitivity                | `CORE_FEEDBACK_QSENSITIVITY`    | per W m⁻² | 7.842035e-9 |
| Cumulative temperature sensitivity | `CORE_FEEDBACK_CUMTSENSITIVITY` | -        | 0.08         |
| Cumulative temperature period      | `CORE_FEEDBACK_CUMTPERIOD`      | yr       | 300          |

### Ocean

| Parameter                    | Fortran name                    | Units       | Default    |
| ---------------------------- | ------------------------------- | ----------- | ---------- |
| Mixed-layer depth            | `CORE_MIXEDLAYER_DEPTH`         | m           | 60         |
| Number of ocean layers       | `CORE_OCN_NLEVELS`              | -           | 50         |
| Vertical diffusivity         | `CORE_VERTICALDIFFUSIVITY`      | cm² s⁻¹     | 0.75       |
| Minimum vertical diffusivity | `CORE_VERTICALDIFFUSIVITY_MIN`  | cm² s⁻¹     | 0.1        |
| Diffusivity gradient         | `CORE_VERTICALDIFF_TOP_DKDT`    | cm² s⁻¹ K⁻¹ | -0.1910108 |
| Depth-dependent area         | `CORE_OCN_DEPTHDEPENDENT`       | -           | 1.0        |
| Polar sinking water ratio    | `CORE_POLARSINKWATER_TEMPRATIO` | -           | 0.2        |

### Upwelling

| Parameter                   | Fortran name                    | Units  | Default |
| --------------------------- | ------------------------------- | ------ | ------- |
| Initial upwelling rate      | `CORE_INITIAL_UPWELLING_RATE`   | m yr⁻¹ | 3.5     |
| Variable upwelling fraction | `CORE_UPWELLING_VARIABLE_PART`  | -      | 0.7     |
| Threshold temperature (NH)  | `CORE_UPWELL_THRESH_TEMP_NH`    | K      | 8.0     |
| Threshold temperature (SH)  | `CORE_UPWELL_THRESH_TEMP_SH`    | K      | 8.0     |
| One global threshold        | `CORE_UPWELL_THRESH_ONEGLOBAL`  | -      | 1       |
| Scaling method              | `CORE_UPWELLING_SCALING_METHOD` | -      | `GLOBE` |

### Heat exchange

| Parameter                    | Fortran name                      | Units     | Default   |
| ---------------------------- | --------------------------------- | --------- | --------- |
| Land-ocean heat exchange     | `CORE_HEATXCHANGE_LANDOCEAN`      | W m⁻² K⁻¹ | 1.442816  |
| Inter-hemispheric exchange   | `CORE_HEATXCHANGE_NORTHSOUTH`     | W m⁻² K⁻¹ | 0.3115475 |
| Ocean-to-land amplification  | `CORE_AMPLIFY_OCN2LAND_HEATXCHNG` | -         | 1.02      |
| Land-ground exchange         | `CORE_HEATXCHANGE_LANDGROUND`     | W m⁻² K⁻¹ | 0.1       |
| Apply ground heat reservoir  | `CORE_LANDHEATCAPACITY_APPLY`     | -         | 1         |
| Ground reservoir thickness   | `CORE_LANDHC_EFFTHICKNESS`        | m         | 300       |

### Temperature adjustment and area

| Parameter                     | Fortran name                     | Units | Default |
| ----------------------------- | -------------------------------- | ----- | ------- |
| Apply ocean-to-air adjustment | `CORE_SWITCH_TEMPADJUST_OCN2ATM` | -     | 1       |
| Linear coefficient            | `CORE_TEMPADJUST_OCN2ATM_ALPHA`  | -     | 1.04    |
| Quadratic coefficient         | `CORE_TEMPADJUST_OCN2ATM_GAMMA`  | K⁻¹   | -0.002  |
| Temperature cap               | `CORE_MAXIMAL_TEMPERATURE`       | K     | 25.0    |
| NH land fraction              | `CORE_HEMISFRACTION_NH_LAND`     | -     | 0.42    |
| SH land fraction              | `CORE_HEMISFRACTION_SH_LAND`     | -     | 0.21    |

### El Niño and AMV

MAGICC7's defaults switch both modes off (`CORE_ELNINO_APPLY = 0`, `CORE_AMV_APPLY = 0`),
and RSCM does not implement them.

| Parameter            | Fortran name              | Units | Default |
| -------------------- | ------------------------- | ----- | ------- |
| Apply El Niño        | `CORE_ELNINO_APPLY`       | -     | 0       |
| El Niño scale        | `CORE_ELNINO_SCALE`       | K     | 0.010825 |
| El Niño relaxation   | `CORE_ELNINO_RELAXFACTOR` | -     | 0.026548 |
| Apply AMV            | `CORE_AMV_APPLY`          | -     | 0       |
| AMV scale            | `CORE_AMV_SCALE`          | K     | -4.26727572 |
| AMV SH fraction      | `CORE_AMV_SHFRACTION`     | -     | 0.0     |

## Algorithm

`ocean_init` sets the substep `DT = 1 / STEPSPERYEAR`, the layer spacing (`DZ = 100` m, `DZ1 = DZ/2`),
the depth-dependent areas and area factors, the initial upwelling rate
and the initial temperature profile (`CORE_SWITCH_OCN_TEMPPROFILE`, CMIP5-based by default).

Each year `magicc_step_year` (MAGICC7.f90 lines 2646 to 3727) runs:

1. `DELTAQ` for concentrations and forcing, except in the last year
   (see [main step](module_15_main_step.md)).
2. For each substep, from substep 1 to `STEPSPERYEAR` (from substep 0 in the first year, to 11 in the last):
    1. Interpolate the box forcing to the substep and add monthly volcanic forcing.
    2. Adjust the climate sensitivity from the substep's global forcing and the cumulative temperature,
       then call LAMCALC for $\lambda_o$, $\lambda_l$ and the CO₂ internal efficacy.
    3. Step the ground reservoir from the previous substep's land temperature.
       The ocean solve below still uses the ground temperature from before this update.
    4. For each hemisphere: compute the diffusivities, build the tridiagonal system,
       add the El Niño and ground heat terms to $D_1$, solve it and cap at `CORE_MAXIMAL_TEMPERATURE`.
    5. Convert SST to ocean air temperature, and keep $\alpha_{eff}$ for the next substep.
    6. Compute the land temperatures and cap them.
    7. Store the substep temperatures and the energy-balance heat uptake $Q - \sum_i f_i \lambda_i T_i$.
    8. Update the inter-hemispheric exchange for the next substep from this substep's air temperatures.
    9. Update upwelling from the global mean air temperature.
3. Average substeps 1 to 12 into the annual mean temperatures.
4. Store the ocean layer profiles and the heat content by depth.
5. Call `sealevel_calc` (see [sea level rise](module_14_sea_level_rise.md)).

LAMCALC (MAGICC7.f90 lines 8070 to 8278) starts from $\lambda_o = \lambda + 0.7$ and steps $\lambda_o$ by 0.7,
reversing when the error in $RLO$ grows.
After the first sign change it switches to the secant method.
It gives up after 40 iterations.
Each iteration derives $\lambda_l$, builds and inverts $\mathbf{A}$,
and compares the area-weighted land and ocean equilibrium temperatures with $RLO$.

## Numerical considerations

- The fully implicit (backward Euler) ocean scheme is unconditionally stable.
  An explicit scheme would need $\Delta t < \Delta z^2 / 2K$.
  With $K = 0.75$ cm² s⁻¹ $\approx 2367$ m² yr⁻¹ and $\Delta z = 100$ m, that is about 2.1 years.
- The Thomas algorithm is stable for a diagonally dominant matrix, $|B_l| > |A_l| + |C_l|$,
  which the positive diffusivity and upwelling terms guarantee.
- LAMCALC's step of 0.7 is empirical.
  Larger steps (above 0.9) oscillate, and smaller ones (below 0.4) can exceed 40 iterations.
- Monthly substeps (`STEPSPERYEAR = 12`) also give the annual means and resolve El Niño variability.
- `CORE_MAXIMAL_TEMPERATURE = 25` K caps every temperature.
  MAGICC7 logs a warning when the cap binds, which signals an unreasonable parameter set.

## Issues in MAGICC7

- **Code organisation**: the climate physics lives in `MAGICC7.f90` (lines about 2700 to 3700),
  not in `climate_and_ocean.f90`, which holds only declarations, parameters and `ocean_init`.
- **Hardcoded values**: `DZ = 100` m, the LAMCALC limits (`IMAX = 40`, `DLAMO = 0.7`, `DIFFLIM = 0.001`)
  and `CORE_MAXIMAL_TEMPERATURE = 25`, which is set in code and overrides the configuration.
  The ocean area profiles are embedded, though they can be read from a file.
- **Fixed structure**: the four boxes (1 = NHO, 2 = NHL, 3 = SHO, 4 = SHL), the 4 × 4 matrix
  and the implicit tridiagonal scheme are assumed throughout.
- **Mixed units**: diffusivity in cm² s⁻¹ (converted with 3155.76), heat capacity in W yr m⁻³ K⁻¹
  and heat content in 10²² J (`WATTYEARPERM2_2_JOULEE22`).
- **Land equilibrium**: land is in instantaneous equilibrium with the ocean.
  The ground reservoir only partly addresses this.
- **El Niño box**: the "double delta box" is not a separate ocean region.
  It has the mixed layer's area and temperature and relaxes rather than exchanging heat,
  which makes it hard to validate.
- **Variable upwelling**: it needs the change from the initial rate and the initial profile
  for each of the three profile types.
- **LAMCALC cost**: it inverts the matrix iteratively and runs every substep,
  because the climate sensitivity changes with each substep's forcing.
- **Dead code**: commented-out blocks remain (`climate_and_ocean.f90` lines 119 to 135,
  `MAGICC7.f90` lines 3420 to 3422), with several "PROCEED HERE" comments.

## Differences from MAGICC7

- No El Niño or AMV modes.
- Upwelling scales with the `GLOBE` method only.
  Prescribed temperatures and prescribed upwelling are not supported.
- The last year runs 12 substeps where MAGICC7 runs 11.
  See [time conventions](time_conventions.md).

## References

| File                                          | Lines     | Content                                   |
| --------------------------------------------- | --------- | ----------------------------------------- |
| `src/libmagicc/physics/climate_and_ocean.f90` | 1-295     | Module variables, `ocean_init` (102-292)  |
| `src/libmagicc/physics/core_calculations.f90` | 1-110     | Ocean-to-air temperature conversion       |
| `src/libmagicc/MAGICC7.f90`                   | 237-246   | `HEAT_CAPACITY_PERM`                      |
| `src/libmagicc/MAGICC7.f90`                   | 249-269   | Area fractions                            |
| `src/libmagicc/MAGICC7.f90`                   | 2646-3727 | `magicc_step_year`                        |
| `src/libmagicc/MAGICC7.f90`                   | 2724      | Forcing interpolation                     |
| `src/libmagicc/MAGICC7.f90`                   | 2747-2768 | Climate sensitivity adjustment            |
| `src/libmagicc/MAGICC7.f90`                   | 2783-2799 | Vertical diffusivity                      |
| `src/libmagicc/MAGICC7.f90`                   | 2826-3074 | Tridiagonal coefficients                  |
| `src/libmagicc/MAGICC7.f90`                   | 3080-3094 | Thomas algorithm                          |
| `src/libmagicc/MAGICC7.f90`                   | 3102-3112 | Temperature caps                          |
| `src/libmagicc/MAGICC7.f90`                   | 3159-3187 | Ocean-to-air conversion                   |
| `src/libmagicc/MAGICC7.f90`                   | 3214-3223 | Land temperature                          |
| `src/libmagicc/MAGICC7.f90`                   | 3258-3281 | Heat uptake                               |
| `src/libmagicc/MAGICC7.f90`                   | 3300-3388 | Upwelling                                 |
| `src/libmagicc/MAGICC7.f90`                   | 3627-3721 | Heat content                              |
| `src/libmagicc/MAGICC7.f90`                   | 8070-8278 | `LAMCALC` (iteration 8116-8207)           |
| `src/libmagicc/MAGICC7.f90`                   | 8291-8338 | `CALC_INTERNAL_EFFICACY`                  |
| `src/libmagicc/allcfgs.f90`                   | 1-431     | Parameter namelist                        |
| `src/libmagicc/utils/areas.f90`               | 1-8       | Area fraction variables                   |
| `run/MAGCFG_DEFAULTALL.CFG`                   | 81-139    | `CORE_*` defaults                         |

| Constant    | Value  | Location                   | Meaning                         |
| ----------- | ------ | -------------------------- | ------------------------------- |
| `RHO`       | 1.026  | `climate_and_ocean.f90:67` | Water density (10⁶ g m⁻³)       |
| `SPECHT`    | 0.9333 | `climate_and_ocean.f90:67` | Specific heat (cal g⁻¹ K⁻¹)     |
| `HTCONS`    | 4.1856 | `climate_and_ocean.f90:67` | Calories to joules              |
| -           | 3155.76 | `MAGICC7.f90:2789`        | cm² s⁻¹ to m² yr⁻¹              |
| -           | 31.5576 | `MAGICC7.f90:241`         | Seconds per year / 10⁶          |
| -           | 5.101  | `MAGICC7.f90:244`          | Earth surface area (10¹⁴ m²)    |
