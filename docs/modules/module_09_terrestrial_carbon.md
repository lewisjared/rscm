# Module 09: Terrestrial Carbon Cycle

## 1. Scientific Purpose

The Terrestrial Carbon Cycle module simulates the exchange of carbon between the atmosphere and land ecosystems using a simplified 4-box model. It represents:

1. **Plant biomass pool** - Living vegetation (leaves, stems, roots)
2. **Detritus/Litter pool** - Dead organic matter in transition
3. **Soil carbon pool** - Long-lived organic carbon in soils
4. **Atmosphere** - Implicitly represented via fluxes

The module captures two critical feedback mechanisms:

1. **CO2 Fertilization**: Higher atmospheric CO2 concentrations enhance photosynthesis and Net Primary Production (NPP), creating a negative feedback that slows atmospheric CO2 accumulation.

2. **Temperature-Respiration Feedback**: Warmer temperatures accelerate decomposition and respiration rates, releasing more CO2 to the atmosphere, creating a positive feedback that accelerates climate change.

The module also handles land-use change emissions (deforestation) with an associated regrowth sink, tracking the separate "no-feedback" carbon pools to properly attribute land-use change vs. climate feedback effects.

## 2. Mathematical Formulation

### 2.1 Overview: 4-Pool Carbon Dynamics

The terrestrial carbon cycle tracks three explicit pools, with atmosphere implicitly represented through net fluxes:

```
                     NPP
        Atmosphere -----> [PLANT]
              ^              |
              |              | turnover (to detritus and soil)
              |              v
        [RESPIRATION] <-- [DETRITUS] --> [SOIL]
                              ^            |
                              |____________|
                            (partial transfer)
```

### 2.2 Net Primary Production (NPP) with Feedbacks

NPP is calculated with both CO2 fertilization and temperature feedbacks:

$$NPP(t) = NPP_0 \cdot \beta(CO_2) \cdot f_T^{NPP}(T)$$

Where:

- $NPP_0$ = `CO2_NPP_INITIAL` = Initial/reference NPP (GtC/yr)
- $\beta(CO_2)$ = CO2 fertilization factor (dimensionless)
- $f_T^{NPP}(T)$ = Temperature effect on NPP

#### 2.2.1 Temperature Effect on NPP

$$f_T^{NPP}(T) = \exp(\gamma_{NPP} \cdot \Delta T)$$

Where:

- $\gamma_{NPP}$ = `CO2_FEEDBACKFACTOR_NPP` (default: +0.0107)
- $\Delta T$ = Temperature change from reference year (K)

#### 2.2.2 CO2 Fertilization Methods

The module supports three fertilization formulations, blended via `CO2_FERTILIZATION_METHOD`:

**Method 1: Logarithmic (Keeling-Bacastow, 1973)**

$$\beta_{log} = 1 + \beta_0 \cdot \ln\left(\frac{CO_2}{CO_{2,ref}}\right)$$

Where $\beta_0$ = `CO2_FERTILIZATION_FACTOR`

**Method 2: Rectangular Hyperbolic (Gifford/Michaelis-Menton)**

$$\beta_{Gif} = \frac{1/C_r + B_{ee}}{1/D_r + B_{ee}}$$

Where:

- $C_r = CO_{2,ref} - CO_{2,zero}$
- $D_r = CO_2 - CO_{2,zero}$
- $CO_{2,zero}$ = `CO2_GIFFORD_CONC_FOR_ZERONPP` (concentration where NPP = 0)
- $B_{ee}$ = derived parameter from matching behavior at 340 and 680 ppm

**Method 3: Saturating Sigmoid (A. Norton)**

$$\beta_{sig} = \frac{A}{1 + \exp(-(CO_2 - CO_{2,ref,sig})/B)}$$

Where:

- $A$ = `CO2_FERTILIZATION_FACTOR` (maximum fertilization)
- $B$ = `CO2_FERTILIZATION_FACTOR2` (curvature parameter)
- $CO_{2,ref,sig}$ = adjusted so $\beta = 1$ at reference concentration

**Blending:**

- If `CO2_FERTILIZATION_METHOD < 1`: No fertilization ($\beta = 1$)
- If `1 <= method <= 2`: Linear blend of log and Gifford
- If `2 < method <= 3`: Linear blend of Gifford and sigmoid

$$\beta = (method - 1) \cdot \beta_{Gif} + (2 - method) \cdot \beta_{log} \quad \text{for } 1 \le method \le 2$$

$$\beta = (method - 2) \cdot \beta_{sig} + (3 - method) \cdot \beta_{Gif} \quad \text{for } 2 < method \le 3$$

### 2.3 Heterotrophic Respiration from Plant Pool

The respiration flux from the plant pool (representing autotrophic respiration and fast heterotrophic decomposition) is:

**Method 1** (`CO2_PLANTBOXRESP_METHOD = 1`):

$$R_h(t) = R_{h,0} \cdot \beta(CO_2) \cdot f_T^{resp}(T)$$

**Method 2** (`CO2_PLANTBOXRESP_METHOD = 2`):

$$R_h(t) = R_{h,0} \cdot \left(1 + \alpha_{resp} \cdot (\beta - 1)\right) \cdot \min\left(1, \frac{C_P}{C_{P,0}}\right) \cdot f_T^{resp}(T)$$

Where:

- $R_{h,0}$ = `CO2_RESPIRATION_INITIAL`
- $f_T^{resp}(T) = \exp(\gamma_{resp} \cdot \Delta T)$
- $\gamma_{resp}$ = `CO2_FEEDBACKFACTOR_RESPIRATION`
- $\alpha_{resp}$ = `CO2_PLANTBOXRESP_FERTSCALE`
- $C_P$ = current plant pool, $C_{P,0}$ = initial plant pool

### 2.4 Pool Dynamics: Implicit Trapezoidal Integration

Each pool uses an implicit trapezoidal (Crank-Nicolson) scheme for stability:

$$C_{t+1} = \frac{C_t \cdot (1 - 0.5/\tau_{eff}) + F_{in}}{ (1 + 0.5/\tau_{eff})}$$

Where:

- $\tau_{eff}$ = effective turnover time (possibly temperature-modified)
- $F_{in}$ = net input flux to the pool

#### 2.4.1 Plant Pool Dynamics

$$\frac{dC_P}{dt} = NPP \cdot f_P - R_h - F_{P \to D+S} - E_{defo,P}$$

Where:

- $f_P$ = `CO2_FRACTION_NPP_2_PLANT` = fraction of NPP to plant pool
- $F_{P \to D+S}$ = turnover flux from plant to detritus and soil
- $E_{defo,P}$ = gross deforestation from plant pool

The effective inverse turnover time for the plant pool is:

$$\frac{1}{\tau_P^{eff}} = \frac{1}{\tau_P(t)}$$

Where $\tau_P(t)$ adjusts based on cumulative land-use emissions:

$$\tau_P(t) = \frac{C_{P,0} - f_{no-regrow} \cdot f_{defo,P} \cdot \sum E_{CO2B}}{F_{net,P,0}}$$

And $F_{net,P,0} = f_P \cdot NPP_0 - R_{h,0}$

#### 2.4.2 Detritus Pool Dynamics

$$\frac{dC_D}{dt} = NPP \cdot f_D + f_{P2D} \cdot F_{P \to D+S} - F_{D \to S+A} - E_{defo,D}$$

Where:

- $f_D$ = `CO2_FRACTION_NPP_2_DETRITUS`
- $f_{P2D}$ = `CO2_FRACTION_PLANT_2_DETRITUS`
- $F_{D \to S+A}$ = flux from detritus to soil and atmosphere

The effective inverse turnover time includes temperature feedback:

$$\frac{1}{\tau_D^{eff}} = \frac{1}{\tau_D(t)} \cdot f_T^{detr}(T)$$

Where $f_T^{detr}(T) = \exp(\gamma_D \cdot \Delta T)$ and $\gamma_D$ = `CO2_FEEDBACKFACTOR_DETRITUS`

#### 2.4.3 Soil Pool Dynamics

$$\frac{dC_S}{dt} = NPP \cdot f_S + (1 - f_{P2D}) \cdot F_{P \to D+S} + f_{D2S} \cdot F_{D \to S+A} - F_{S \to A} - E_{defo,S}$$

Where:

- $f_S$ = `CO2_FRACTION_NPP_2_SOIL` (derived: $1 - f_P - f_D$)
- $f_{D2S}$ = `CO2_FRACTION_DETRITUS_2_SOIL`
- $F_{S \to A}$ = respiration from soil pool

The effective inverse turnover time:

$$\frac{1}{\tau_S^{eff}} = \frac{1}{\tau_S(t)} \cdot f_T^{soil}(T)$$

Where $f_T^{soil}(T) = \exp(\gamma_S \cdot \Delta T)$ and $\gamma_S$ = `CO2_FEEDBACKFACTOR_SOIL`

### 2.5 Land-Use Change (Deforestation and Regrowth)

Deforestation emissions are distributed across pools:

- $E_{defo,P} = f_{defo,P} \cdot E_{CO2B}$ where $f_{defo,P}$ = `CO2_FRACTION_DEFOREST_PLANT`
- $E_{defo,D} = f_{defo,D} \cdot E_{CO2B}$ where $f_{defo,D}$ = `CO2_FRACTION_DEFOREST_DETRITUS`
- $E_{defo,S} = f_{defo,S} \cdot E_{CO2B}$ where $f_{defo,S} = 1 - f_{defo,P} - f_{defo,D}$

"Gross deforestation" includes both the net land-use emissions and the regrowth:

$$E_{gross,X}(t) = f_{defo,X} \cdot E_{CO2B}(t) + Regrowth_X(t)$$

Regrowth is calculated from the difference between the "no-feedback" pool trajectory and the expected trajectory without deforestation.

### 2.6 No-Feedback Reference Pools

The module maintains parallel "no-feedback" pool trajectories (`CO2_NOFEED_*_POOL`) that track what the pools would be without CO2 fertilization or temperature feedbacks. These are used to:

1. Calculate regrowth fluxes consistently
2. Enable attribution of land sink vs. climate feedback effects
3. Apply a correction to ensure land-use emissions are conserved

The correction term is:

$$\Delta_{corr} = E_{CO2B} + (C_P^{nofeed}(t+1) - C_P^{nofeed}(t)) + (C_D^{nofeed}(t+1) - C_D^{nofeed}(t)) + (C_S^{nofeed}(t+1) - C_S^{nofeed}(t))$$

This small correction is added to the plant pool to ensure mass conservation.

## 3. State Variables

| Variable | Fortran Name | Symbol | Units | Description | Initial Value |
|----------|--------------|--------|-------|-------------|---------------|
| Plant carbon pool | `CO2_PLANT_POOL` | $C_P$ | GtC | Carbon stored in living vegetation | `CO2_PLANTPOOL_INITIAL` |
| Detritus carbon pool | `CO2_DETRITUS_POOL` | $C_D$ | GtC | Carbon in litter/dead organic matter | `CO2_DETRITUSPOOL_INITIAL` |
| Soil carbon pool | `CO2_SOIL_POOL` | $C_S$ | GtC | Carbon in soils | `CO2_SOILPOOL_INITIAL` |
| No-feedback plant pool | `CO2_NOFEED_PLANT_POOL` | $C_P^{nf}$ | GtC | Plant pool without feedbacks | `CO2_PLANTPOOL_INITIAL` |
| No-feedback detritus pool | `CO2_NOFEED_DETRITUS_POOL` | $C_D^{nf}$ | GtC | Detritus pool without feedbacks | `CO2_DETRITUSPOOL_INITIAL` |
| No-feedback soil pool | `CO2_NOFEED_SOIL_POOL` | $C_S^{nf}$ | GtC | Soil pool without feedbacks | `CO2_SOILPOOL_INITIAL` |
| Current NPP | `CO2_CURRENT_NPP` | NPP | GtC/yr | Net Primary Production with feedbacks | `CO2_NPP_INITIAL` |
| Respiration | `CO2_RESPIRATION` | $R_h$ | GtC/yr | Heterotrophic respiration from plant pool | `CO2_RESPIRATION_INITIAL` |
| Effective fertilization | `CO2_EFF_FERTILIZATION_FACTOR` | $\beta$ | dimensionless | CO2 fertilization multiplier | 1.0 |
| NPP temperature effect | `CO2_EFF_NPP_TEMPFEEDBACK` | $f_T^{NPP}$ | dimensionless | Temperature effect on NPP | 1.0 |
| Respiration temp effect | `CO2_EFF_RESP_TEMPFEEDBACK` | $f_T^{resp}$ | dimensionless | Temperature effect on respiration | 1.0 |
| Detritus temp effect | `CO2_EFF_DETR_TEMPFEEDBACK` | $f_T^{detr}$ | dimensionless | Temperature effect on detritus decay | 1.0 |
| Soil temp effect | `CO2_EFF_SOIL_TEMPFEEDBACK` | $f_T^{soil}$ | dimensionless | Temperature effect on soil decay | 1.0 |
| Cumulative net emissions | `CO2_NETCUMUL_EMIS` | - | GtC | Cumulative net CO2 emissions to atmosphere | 0.0 |
| Regrowth plant flux | `CO2_REGROWTH_PLANT_FLUX` | - | GtC/yr | Regrowth sink in plant pool | 0.0 |
| Regrowth detritus flux | `CO2_REGROWTH_DETRITUS_FLUX` | - | GtC/yr | Regrowth sink in detritus pool | 0.0 |
| Regrowth soil flux | `CO2_REGROWTH_SOIL_FLUX` | - | GtC/yr | Regrowth sink in soil pool | 0.0 |
| Gross deforestation | `CO2_GROSSDEFO_EMIS` | - | GtC/yr | Gross deforestation emissions | 0.0 |
| Plant turnover time | `CURRENT_TURNOVERTIME_PLANTPOOL` | $\tau_P$ | years | Time-varying plant turnover time | Derived |
| Detritus turnover time | `CURR_TURNOVERT_DETRPOOL` | $\tau_D$ | years | Time-varying detritus turnover time | Derived |
| Soil turnover time | `CURRENT_TURNOVERTIME_SOILPOOL` | $\tau_S$ | years | Time-varying soil turnover time | Derived |

## 4. Parameters

### 4.1 Initial Pool Sizes

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Initial plant pool | `CO2_PLANTPOOL_INITIAL` | GtC | 884.86 | 400-1200 | Pre-industrial plant biomass carbon |
| Initial detritus pool | `CO2_DETRITUSPOOL_INITIAL` | GtC | 92.77 | 50-200 | Pre-industrial litter/detritus carbon |
| Initial soil pool | `CO2_SOILPOOL_INITIAL` | GtC | 1681.53 | 1000-2500 | Pre-industrial soil carbon |
| Initial NPP | `CO2_NPP_INITIAL` | GtC/yr | 66.27 | 50-80 | Pre-industrial Net Primary Production |
| Initial respiration | `CO2_RESPIRATION_INITIAL` | GtC/yr | 12.26 | 5-20 | Pre-industrial heterotrophic respiration (plant pool) |

### 4.2 CO2 Fertilization Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Fertilization factor | `CO2_FERTILIZATION_FACTOR` | dimensionless | 0.6486 | 0.2-1.0 | $\beta_0$ in logarithmic formula |
| Fertilization factor 2 | `CO2_FERTILIZATION_FACTOR2` | ppm | 100.0 | 50-200 | Curvature for sigmoid method |
| Fertilization method | `CO2_FERTILIZATION_METHOD` | dimensionless | 1.10 | 0-3 | Blend between log (1), Gifford (2), sigmoid (3) |
| Fertilization start year | `CO2_FERTILIZATION_YRSTART` | year | 1900.0 | - | Year when fertilization begins |
| Pre-industrial CO2 | `CO2_PREINDCO2CONC` | ppm | 278.0 | 260-290 | Reference CO2 for forcing calculations |
| Gifford zero-NPP conc | `CO2_GIFFORD_CONC_FOR_ZERONPP` | ppm | 80.0 | 30-100 | CO2 at which NPP = 0 (Gifford only) |

### 4.3 Temperature Feedback Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Temperature feedback switch | `CO2_TEMPFEEDBACK_SWITCH` | flag | 1.0 | 0 or 1 | Enable temperature feedbacks |
| Temp feedback start year | `CO2_TEMPFEEDBACK_YRSTART` | year | 1900.0 | - | Year when temp feedbacks begin |
| NPP temp sensitivity | `CO2_FEEDBACKFACTOR_NPP` | K^-1 | +0.0107 | -0.05 to +0.05 | $\gamma_{NPP}$ - positive = warming increases NPP |
| Respiration temp sensitivity | `CO2_FEEDBACKFACTOR_RESPIRATION` | K^-1 | +0.0685 | 0 to 0.15 | $\gamma_{resp}$ - Q10-like factor |
| Detritus temp sensitivity | `CO2_FEEDBACKFACTOR_DETRITUS` | K^-1 | -0.1358 | -0.2 to 0 | $\gamma_D$ - negative = warming speeds decay |
| Soil temp sensitivity | `CO2_FEEDBACKFACTOR_SOIL` | K^-1 | +0.1541 | 0 to 0.2 | $\gamma_S$ - positive = warming speeds decay |
| Use land-only temperature | `CO2_FEEDTEMP_LANDONLY_APPLY` | flag | 0 | 0 or 1 | Use land temp instead of global mean |

**Note on signs:** The sign convention is somewhat confusing:

- For decay processes (detritus, soil), a **positive** $\gamma$ means warming **increases** decay rate (releases more CO2)
- The default `CO2_FEEDBACKFACTOR_DETRITUS` is **negative** (-0.1358), which is physically counterintuitive and may indicate a sign convention issue or compensating effect
- For NPP, a **positive** $\gamma$ means warming **increases** NPP

### 4.4 Carbon Flow Fractions

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| NPP to plant fraction | `CO2_FRACTION_NPP_2_PLANT` | fraction | 0.4483 | 0.3-0.6 | Fraction of NPP going to plant pool |
| NPP to detritus fraction | `CO2_FRACTION_NPP_2_DETRITUS` | fraction | 0.3998 | 0.2-0.5 | Fraction of NPP going directly to detritus |
| NPP to soil fraction | `CO2_FRACTION_NPP_2_SOIL` | fraction | derived | - | = 1 - plant - detritus (derived) |
| Plant to detritus fraction | `CO2_FRACTION_PLANT_2_DETRITUS` | fraction | 0.9989 | 0.8-1.0 | Plant turnover flux going to detritus vs soil |
| Detritus to soil fraction | `CO2_FRACTION_DETRITUS_2_SOIL` | fraction | 0.001 | 0-0.5 | Detritus decay flux going to soil vs atmosphere |

### 4.5 Deforestation Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Deforest plant fraction | `CO2_FRACTION_DEFOREST_PLANT` | fraction | 0.70 | 0.5-0.9 | Fraction of deforestation from plant pool |
| Deforest detritus fraction | `CO2_FRACTION_DEFOREST_DETRITUS` | fraction | 0.05 | 0-0.2 | Fraction of deforestation from detritus |
| Deforest soil fraction | `CO2_FRACTION_DEFOREST_SOIL` | fraction | derived | - | = 1 - plant - detritus (derived) |
| No-regrowth fraction | `CO2_NORGRWTH_FRAC_DEFO` | fraction | (varies) | 0-1 | Fraction of deforestation that is permanent |

### 4.6 Plant Box Respiration Method

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Respiration method | `CO2_PLANTBOXRESP_METHOD` | integer | 1 | 1 or 2 | Method for plant pool respiration |
| Fertilization scaling | `CO2_PLANTBOXRESP_FERTSCALE` | fraction | 0.0 | 0-1 | Scaling of fertilization effect on respiration (method 2) |

### 4.7 Other Parameters

| Parameter | Fortran Name | Units | Default | Description |
|-----------|--------------|-------|---------|-------------|
| GtC per ppm | `GTC_PER_PPM` | GtC/ppm | 2.123 | Conversion factor (hardcoded in `carboncycle_init`) |
| CO2 GWP | `CO2_GWP` | dimensionless | 1.0 | Global Warming Potential (by definition) |

## 5. Inputs (per timestep)

| Variable | Units | Source Module/Data | Required? | Fortran Variable |
|----------|-------|-------------------|-----------|------------------|
| Land-use CO2 emissions | GtC/yr | Emissions input | Yes | `DAT_CO2B_EMIS%DATGLOBE(CURRENT_YEAR_IDX)` |
| Cumulative land-use emissions | GtC | Emissions input | Yes | `DAT_CO2B_CUMEMIS%DATGLOBE(CURRENT_YEAR_IDX)` |
| Atmospheric CO2 concentration | ppm | Carbon cycle | Yes | `DAT_CO2_CONC%DATGLOBE(CURRENT_YEAR_IDX)` |
| Feedback temperature delta | K | Climate module | Yes | `FEEDBACK_TEMPERATURE` (subroutine argument) |
| Previous year index | integer | Time module | Yes | `PREVIOUS_YEAR_IDX` |
| Current year index | integer | Time module | Yes | `CURRENT_YEAR_IDX` |
| Next year index | integer | Time module | Yes | `NEXT_YEAR_IDX` |
| Nitrogen limitation factor | dimensionless | N-cycle module | If enabled | `NCYCLE_LIMIT_FACTOR(CURRENT_YEAR_IDX)` |

**Note:** The feedback temperature is calculated in MAGICC7.f90 before calling TERRCARBON2. It can be either global mean surface temperature or land-only temperature depending on `CO2_FEEDTEMP_LANDONLY_APPLY`.

## 6. Outputs (per timestep)

| Variable | Units | Destination Module(s) | Fortran Variable |
|----------|-------|----------------------|------------------|
| Plant pool (next year) | GtC | Stored | `CO2_PLANT_POOL(NEXT_YEAR_IDX)` |
| Detritus pool (next year) | GtC | Stored | `CO2_DETRITUS_POOL(NEXT_YEAR_IDX)` |
| Soil pool (next year) | GtC | Stored | `CO2_SOIL_POOL(NEXT_YEAR_IDX)` |
| Current NPP | GtC/yr | Stored, output | `CO2_CURRENT_NPP(CURRENT_YEAR_IDX)` -> `DAT_CO2_CURRENT_NPP` |
| Total respiration | GtC/yr | Stored, output | `CO2_TOTALRESPIRATION(CURRENT_YEAR_IDX)` -> `DAT_CO2_TOTALRESPIRATION` |
| Gross deforestation | GtC/yr | Stored | `CO2_GROSSDEFO_EMIS(CURRENT_YEAR_IDX)` |
| Total regrowth | GtC/yr | Stored | `CO2_REGROWTH_TOTAL_FLUX(CURRENT_YEAR_IDX)` |
| Change in terrestrial pools | GtC/yr | CO2 budget | `CO2_DELTA_TERRPOOLS(CURRENT_YEAR_IDX)` |
| Terr bio + fossil emissions | GtC/yr | CO2 budget | `CO2_TERRBIO_AND_FOSSIL_EMIS(CURRENT_YEAR_IDX)` |
| Cumulative net emissions | GtC | Stored | `CO2_NETCUMUL_EMIS(NEXT_YEAR_IDX)` |
| Effective fertilization | dimensionless | Stored | `CO2_EFF_FERTILIZATION_FACTOR(CURRENT_YEAR_IDX)` |
| Effective NPP temp factor | dimensionless | Stored | `CO2_EFF_NPP_TEMPFEEDBACK(CURRENT_YEAR_IDX)` |
| Effective resp temp factor | dimensionless | Stored | `CO2_EFF_RESP_TEMPFEEDBACK(CURRENT_YEAR_IDX)` |
| Effective detr temp factor | dimensionless | Stored | `CO2_EFF_DETR_TEMPFEEDBACK(CURRENT_YEAR_IDX)` |
| Effective soil temp factor | dimensionless | Stored | `CO2_EFF_SOIL_TEMPFEEDBACK(CURRENT_YEAR_IDX)` |

**Post-TERRCARBON2 outputs** (calculated in MAGICC7.f90 after the subroutine call):

| Variable | Units | Fortran Variable |
|----------|-------|------------------|
| Air-to-land flux | GtC/yr | `CO2_AIR2LAND_FLUX(CURRENT_YEAR_IDX)` |
| Net ecosystem exchange | GtC/yr | `CO2_NETECOEXCH_FLUX(CURRENT_YEAR_IDX)` |
| Total land pool | GtC | `CO2_LAND_POOL(CURRENT_YEAR_IDX)` |
| Net atmosphere-land CO2 flux | GtC/yr | `CO2_NETATMOSLANDCO2FLUX(CURRENT_YEAR_IDX)` |

## 7. Algorithm (Pseudocode)

### 7.1 Initialization (`carboncycle_init`)

```
SUBROUTINE carboncycle_init():
    # Initialize pools to pre-industrial values
    CO2_PLANT_POOL(1) = CO2_PLANTPOOL_INITIAL
    CO2_DETRITUS_POOL(1) = CO2_DETRITUSPOOL_INITIAL
    CO2_SOIL_POOL(1) = CO2_SOILPOOL_INITIAL

    # Initialize no-feedback pools identically
    CO2_NOFEED_PLANT_POOL(1) = CO2_PLANTPOOL_INITIAL
    CO2_NOFEED_DETRITUS_POOL(1) = CO2_DETRITUSPOOL_INITIAL
    CO2_NOFEED_SOIL_POOL(1) = CO2_SOILPOOL_INITIAL

    # Initialize fluxes
    CO2_AIR2OCEAN_FLUX(1) = 0.0
    GTC_PER_PPM = 2.123  # HARDCODED conversion factor

    # Derive NPP fractions (ensure they sum to 1)
    CO2_FRACTION_NPP_2_SOIL = 1.0 - CO2_FRACTION_NPP_2_PLANT - CO2_FRACTION_NPP_2_DETRITUS
    IF CO2_FRACTION_NPP_2_SOIL < 0:
        # Normalize plant and detritus fractions to sum to 1
        scale = 1.0 / (CO2_FRACTION_NPP_2_PLANT + CO2_FRACTION_NPP_2_DETRITUS)
        CO2_FRACTION_NPP_2_PLANT *= scale
        CO2_FRACTION_NPP_2_DETRITUS *= scale
        CO2_FRACTION_NPP_2_SOIL = 0.0

    # Similarly for deforestation fractions
    CO2_FRACTION_DEFOREST_SOIL = 1.0 - CO2_FRACTION_DEFOREST_PLANT - CO2_FRACTION_DEFOREST_DETRITUS
    IF CO2_FRACTION_DEFOREST_SOIL < 0:
        # Normalize
        ...

    # Cap plant-to-detritus fraction at 1.0
    IF CO2_FRACTION_PLANT_2_DETRITUS > 1.0:
        CO2_FRACTION_PLANT_2_DETRITUS = 1.0

    # Ensure respiration doesn't exceed NPP flux to plant pool
    IF CO2_FRACTION_NPP_2_PLANT * CO2_NPP_INITIAL < CO2_RESPIRATION_INITIAL:
        CO2_RESPIRATION_INITIAL = 0.99 * CO2_FRACTION_NPP_2_PLANT * CO2_NPP_INITIAL
        LOG_WARNING("Adjusted CO2_RESPIRATION_INITIAL to allow positive plant flux")

    # Calculate initial net flux to plant pool
    NETFLUX_2_PLANTBOX_INITIAL = CO2_FRACTION_NPP_2_PLANT * CO2_NPP_INITIAL - CO2_RESPIRATION_INITIAL

    # Derive initial turnover times from steady-state assumption
    INITIAL_TURNOVERTIME_PLANTPOOL = CO2_PLANTPOOL_INITIAL / NETFLUX_2_PLANTBOX_INITIAL

    INI_TURNOVERTIME_DETRPOOL = CO2_DETRITUSPOOL_INITIAL /
        (CO2_FRACTION_NPP_2_DETRITUS * CO2_NPP_INITIAL +
         CO2_FRACTION_PLANT_2_DETRITUS * NETFLUX_2_PLANTBOX_INITIAL)

    INITIAL_TURNOVERTIME_SOILPOOL = CO2_SOILPOOL_INITIAL /
        (CO2_NPP_INITIAL - CO2_RESPIRATION_INITIAL -
         (1.0 - CO2_FRACTION_DETRITUS_2_SOIL) *
         CO2_DETRITUSPOOL_INITIAL / INI_TURNOVERTIME_DETRPOOL)
```

### 7.2 Main Timestep (`TERRCARBON2`)

```
SUBROUTINE TERRCARBON2(FEEDBACK_TEMPERATURE):
    # Input: FEEDBACK_TEMPERATURE = temperature anomaly for feedbacks (K)

    # ======== STEP 1: CALCULATE EXTRAPOLATED CO2 CONCENTRATION ========
    # Quadratic extrapolation to mid-year
    CO2_EXTRAP = (3*CO2_CONC(t-2) - 10*CO2_CONC(t-1) + 15*CO2_CONC(t)) / 8

    # ======== STEP 2: DETERMINE REFERENCE CO2 FOR FERTILIZATION ========
    IF year < CO2_FERTILIZATION_YRSTART:
        CO2_REF = CO2_EXTRAP
        CO2_REF_MAX = CO2_EXTRAP
    ELSE:
        # Keep reference at start year value, but allow decrease if CO2 drops
        CO2_REF = MIN(CO2_REF_MAX, CO2_EXTRAP)

    # ======== STEP 3: CALCULATE TEMPERATURE FEEDBACK FACTORS ========
    CO2_EFF_NPP_TEMPFEEDBACK = EXP(CO2_FEEDBACKFACTOR_NPP * FEEDBACK_TEMPERATURE)
    CO2_EFF_RESP_TEMPFEEDBACK = EXP(CO2_FEEDBACKFACTOR_RESPIRATION * FEEDBACK_TEMPERATURE)
    CO2_EFF_DETR_TEMPFEEDBACK = EXP(CO2_FEEDBACKFACTOR_DETRITUS * FEEDBACK_TEMPERATURE)
    CO2_EFF_SOIL_TEMPFEEDBACK = EXP(CO2_FEEDBACKFACTOR_SOIL * FEEDBACK_TEMPERATURE)

    # ======== STEP 4: CALCULATE CO2 FERTILIZATION FACTOR ========
    IF CO2_FERTILIZATION_METHOD < 1.0:
        BETA = 1.0  # No fertilization
    ELSE:
        # Calculate all three formulations
        BETA_LOG = 1.0 + CO2_FERTILIZATION_FACTOR * LN(CO2_EXTRAP / CO2_REF)

        # Gifford (Michaelis-Menton) formulation
        R = (1 + beta0 * ln(680/ref)) / (1 + beta0 * ln(340/ref))
        # ... complex calculation of GIFFORD_BEE ...
        BETA_GIFF = (1/C_r + B_ee) / (1/D_r + B_ee)

        # Sigmoid formulation
        CO2_REF_SIG = CO2_REF + CO2_FERTILIZATION_FACTOR2 * LN(CO2_FERTILIZATION_FACTOR - 1)
        BETA_SIG = CO2_FERTILIZATION_FACTOR / (1 + EXP(-(CO2_EXTRAP - CO2_REF_SIG) / CO2_FERTILIZATION_FACTOR2))

        # Blend based on method parameter
        IF CO2_FERTILIZATION_METHOD <= 2.0:
            BETA = (method-1)*BETA_GIFF + (2-method)*BETA_LOG
        ELSE:
            BETA = (method-2)*BETA_SIG + (3-method)*BETA_GIFF

    CO2_EFF_FERTILIZATION_FACTOR(t) = BETA

    # ======== STEP 5: UPDATE TURNOVER TIMES FOR LAND-USE EFFECTS ========
    # Turnover times decrease as cumulative deforestation increases
    TAU_PLANT(t) = (C_P0 - f_noregrow * f_defo_P * CUM_CO2B) / NETFLUX_P0
    TAU_DETR(t) = (C_D0 - f_noregrow * f_defo_D * CUM_CO2B) / flux_to_detr_0
    TAU_SOIL(t) = ... (similar)

    # ======== STEP 6: CALCULATE NPP WITH FEEDBACKS ========
    NPP = CO2_NPP_INITIAL * BETA * CO2_EFF_NPP_TEMPFEEDBACK

    # ======== STEP 7: CALCULATE RESPIRATION ========
    IF CO2_PLANTBOXRESP_METHOD == 1:
        RESP = CO2_RESPIRATION_INITIAL * BETA * CO2_EFF_RESP_TEMPFEEDBACK
    ELSE IF CO2_PLANTBOXRESP_METHOD == 2:
        RESP = CO2_RESPIRATION_INITIAL *
               (1 + FERTSCALE * (BETA - 1)) *
               MIN(1, PLANT_POOL(t) / CO2_PLANTPOOL_INITIAL) *
               CO2_EFF_RESP_TEMPFEEDBACK

    # ======== STEP 8: APPLY NITROGEN LIMITATION (if enabled) ========
    IF NCYCLE_APPLY == 1 AND year > CO2_FERTILIZATION_YRSTART:
        CALL N_CALC_LIMITATION_FACTOR()
        NPP = NPP * NCYCLE_LIMIT_FACTOR(t)

    CO2_CURRENT_NPP(t) = NPP

    # ======== STEP 9: DISTRIBUTE NPP TO POOLS ========
    NPP_2_PLANT = NPP * CO2_FRACTION_NPP_2_PLANT
    NPP_2_DETRITUS = NPP * CO2_FRACTION_NPP_2_DETRITUS
    NPP_2_SOIL = NPP * CO2_FRACTION_NPP_2_SOIL

    # ======== STEP 10: UPDATE PLANT POOL ========
    # 10a: Calculate no-feedback trajectory
    NOFEED_NETFLUX = NPP_0 * f_P - RESP_0
    NOFEED_NEW_PLANT = implicit_step(NOFEED_PLANT(t), TAU_PLANT(t), NOFEED_NETFLUX - defo_P - regrowth_P(t-1))

    # 10b: Calculate regrowth (difference from no-defo trajectory)
    REGROWTH_PLANT(t) = calculate_regrowth(...)

    # 10c: Calculate gross deforestation
    GROSSD_PLANT = f_defo_P * CO2B_EMIS(t) + REGROWTH_PLANT(t)

    # 10d: Update no-feedback plant pool with gross deforestation
    NOFEED_PLANT(t+1) = implicit_step(..., NOFEED_NETFLUX - GROSSD_PLANT)

    # 10e: Calculate net flux to plant pool with feedbacks
    NETFLUX_2_PLANT = NPP_2_PLANT - RESP - GROSSD_PLANT

    # 10f: Update plant pool using implicit trapezoidal scheme
    PLANT_POOL(t+1) = (PLANT_POOL(t) * (1 - 0.5/TAU_PLANT(t)) + NETFLUX_2_PLANT) /
                      (1 + 0.5/TAU_PLANT(t))

    # 10g: Ensure non-negative
    PLANT_POOL(t+1) = MAX(0.0, PLANT_POOL(t+1))

    # 10h: Calculate turnover flux from plant to detritus/soil
    FLUX_PLANT2DETRSOIL = 0.5/TAU_PLANT(t) * (PLANT_POOL(t) + PLANT_POOL(t+1))

    # ======== STEP 11: UPDATE DETRITUS POOL (similar structure) ========
    # ... calculate no-feedback trajectory, regrowth, gross deforestation
    # ... apply temperature feedback to effective inverse turnover time
    EFF_INV_TAU_DETR = (1/TAU_DETR(t)) * CO2_EFF_DETR_TEMPFEEDBACK
    NETFLUX_2_DETR = NPP_2_DETRITUS + f_P2D * FLUX_PLANT2DETRSOIL - GROSSD_DETR
    DETR_POOL(t+1) = implicit_step_with_temp_feedback(...)

    FLUX_DETR2SOILATM = EFF_INV_TAU_DETR * 0.5 * (DETR_POOL(t) + DETR_POOL(t+1))

    # ======== STEP 12: UPDATE SOIL POOL (similar structure) ========
    # ... with temperature feedback on decay
    EFF_INV_TAU_SOIL = (1/TAU_SOIL(t)) * CO2_EFF_SOIL_TEMPFEEDBACK
    NETFLUX_2_SOIL = NPP_2_SOIL +
                     (1 - f_P2D) * FLUX_PLANT2DETRSOIL +
                     f_D2S * FLUX_DETR2SOILATM -
                     GROSSD_SOIL
    SOIL_POOL(t+1) = implicit_step_with_temp_feedback(...)

    # ======== STEP 13: SAVE DIAGNOSTIC OUTPUTS ========
    GROSSDEFO_EMIS(t) = GROSSD_PLANT + GROSSD_DETR + GROSSD_SOIL
    REGROWTH_TOTAL(t) = REGROWTH_PLANT(t) + REGROWTH_DETR(t) + REGROWTH_SOIL(t)

    # ======== STEP 14: APPLY MASS CONSERVATION CORRECTION ========
    # Small correction to ensure no-feedback pools match prescribed land-use emissions
    NOFEED_DELTA = (NOFEED_PLANT(t+1) - NOFEED_PLANT(t)) +
                   (NOFEED_DETR(t+1) - NOFEED_DETR(t)) +
                   (NOFEED_SOIL(t+1) - NOFEED_SOIL(t))
    DIFF = CO2B_EMIS(t) + NOFEED_DELTA  # Should be ~0

    # Add correction to plant pools
    PLANT_POOL(t+1) -= DIFF
    NOFEED_PLANT(t+1) -= DIFF

    # ======== STEP 15: CALCULATE AGGREGATED OUTPUTS ========
    CO2_DELTA_TERRPOOLS(t) = (PLANT(t+1) - PLANT(t)) + (DETR(t+1) - DETR(t)) + (SOIL(t+1) - SOIL(t))

    CO2_TERRBIO_AND_FOSSIL_EMIS(t) = CO2I_EMIS(t) - CO2_DELTA_TERRPOOLS(t)
    IF PF_APPLY:
        CO2_TERRBIO_AND_FOSSIL_EMIS(t) += CO2PF_EMIS(t)

    CO2_NETCUMUL_EMIS(t+1) = CO2_NETCUMUL_EMIS(t) + CO2_TERRBIO_AND_FOSSIL_EMIS(t)

    CO2_TOTALRESPIRATION(t) = NPP - CO2_DELTA_TERRPOOLS(t) - GROSSDEFO_EMIS(t)
```

### 7.3 Post-Processing in MAGICC7.f90

```
# After TERRCARBON2 returns:

# Calculate air-to-land flux
CO2_AIR2LAND_FLUX(t) = PLANT(t+1) + DETR(t+1) + SOIL(t+1) -
                       PLANT(t) - DETR(t) - SOIL(t)

# Calculate net ecosystem exchange (negative of air-to-land minus land-use)
CO2_NETECOEXCH_FLUX(t) = -CO2_AIR2LAND_FLUX(t) - CO2B_EMIS(t)

# Calculate total land pool
CO2_LAND_POOL(t) = PLANT(t) + DETR(t) + SOIL(t)

# Calculate net atmosphere-land flux
CO2_NETATMOSLANDCO2FLUX(t) = CO2_LAND_POOL(t+1) - CO2_LAND_POOL(t)
```

## 8. Numerical Considerations

### 8.1 Integration Scheme

The module uses an **implicit trapezoidal (Crank-Nicolson)** scheme for each pool:

$$C_{t+1} = \frac{C_t \cdot (1 - 0.5 \cdot k_{eff}) + F_{in}}{1 + 0.5 \cdot k_{eff}}$$

Where $k_{eff} = 1/\tau_{eff}$ is the effective decay rate.

**Advantages:**

- Unconditionally stable for linear decay
- Second-order accurate in time
- Handles stiff systems reasonably well

**Limitations:**

- Assumes linear decay within timestep (may be inaccurate for highly nonlinear feedbacks)
- Fixed 1-year timestep (hardcoded)

### 8.2 Stability Constraints

1. **Positive pool constraint:** Plant pool is explicitly floored at zero:

   ```fortran
   CO2_PLANT_POOL(NEXT_YEAR_IDX) = MAX(0.0D0, CO2_PLANT_POOL(NEXT_YEAR_IDX))
   ```

   Detritus and soil pools are NOT similarly protected and could theoretically go negative.

2. **Turnover time calculation:** If `NETFLUX_2_PLANTBOX_INITIAL <= 0`, the initial turnover time calculation will produce negative or infinite values. The code includes a guard:

   ```fortran
   IF (CO2_FRACTION_NPP_2_PLANT * CO2_NPP_INITIAL) < CO2_RESPIRATION_INITIAL:
       CO2_RESPIRATION_INITIAL = 0.99 * (CO2_FRACTION_NPP_2_PLANT * CO2_NPP_INITIAL)
   ```

3. **Fertilization factor bounds:** The logarithmic fertilization can go negative if `CO2 < CO2_REF`, and the Gifford can produce numerical issues near the zero-NPP concentration. No explicit bounds checking is performed on the output.

### 8.3 Known Numerical Issues

1. **Division by near-zero turnover times:** If cumulative deforestation approaches pool sizes, turnover times can become very small, causing numerical instability.

2. **No convergence checking:** Unlike some modules, there is no iteration or convergence checking. The implicit scheme is applied once per timestep.

3. **Correction term magnitude:** The mass conservation correction (Step 14) should be small, but no warning is issued if it becomes large, which could indicate inconsistent parameters or numerical issues.

### 8.4 Timestep Sensitivity

The module is designed for annual timesteps. Sub-annual integration would require:

1. Adjusting the implicit scheme coefficients
2. Interpolating input emissions
3. Handling the mid-year CO2 extrapolation differently

## 9. Issues and Concerns

### 9.1 Module Separation Issues

**Problem: Code is NOT cleanly separated.** The TERRCARBON2 subroutine lives in `MAGICC7.f90` (lines 7054-7542), not in `carbon_cycle.f90`. The carbon_cycle.f90 module only contains:

- Variable declarations
- `carboncycle_init` subroutine

All the actual physics is in the monolithic MAGICC7.f90 file.

**Impact:**

- Cannot test terrestrial carbon cycle in isolation
- Cannot easily swap implementations
- Difficult to understand data dependencies

### 9.2 Hardcoded Values

| Value | Location | Concern |
|-------|----------|---------|
| `2.123` | carbon_cycle.f90:88 | GTC_PER_PPM hardcoded, should be parameter |
| `0.5` | Throughout | Trapezoidal coefficient, not configurable |
| `0.99` | carbon_cycle.f90:120 | Magic number for respiration adjustment |
| `340.0`, `680.0` | MAGICC7.f90:7203-7204 | Hardcoded reference CO2 levels for Gifford |
| `8.0` | MAGICC7.f90:7103-7105 | Coefficient in quadratic extrapolation |
| `3.0`, `10.0`, `15.0` | MAGICC7.f90:7103-7105 | Coefficients in quadratic extrapolation |

### 9.3 Confusing Sign Conventions

The temperature feedback factors have inconsistent sign conventions:

- `CO2_FEEDBACKFACTOR_NPP = +0.0107` (positive means warming increases NPP) - makes physical sense
- `CO2_FEEDBACKFACTOR_RESPIRATION = +0.0685` (positive means warming increases respiration) - makes physical sense
- `CO2_FEEDBACKFACTOR_DETRITUS = -0.1358` (negative means warming DECREASES decay?) - **physically counterintuitive**
- `CO2_FEEDBACKFACTOR_SOIL = +0.1541` (positive means warming increases decay) - makes physical sense

The negative detritus feedback factor is suspicious. Either:

1. The sign convention is reversed for detritus
2. There's a compensating effect being modeled
3. It's a bug or calibration artifact

### 9.4 Unclear Variable Names

| Fortran Name | Issue |
|--------------|-------|
| `NOFEED_NEWPLANTPOOL_NO_REGRWTH` | Extremely long, unclear |
| `CURR_NOFEED_FLUX_DETR2SOIL` | Inconsistent abbreviations |
| `EFF_INVTURNOVERT_DETRPOOL` | Mix of abbreviation styles |
| `CURRENTFLUX_PLANT2DETRSOIL` | Unclear if "current" means this timestep or flow rate |
| `CURNTFLUX_DETRITUS2SOILATMOS` | Typo? "CURNT" vs "CURRENT" |
| `INI_TURNOVERTIME_DETRPOOL` | "INI" vs "INITIAL" inconsistency |

### 9.5 Potential Bugs

1. **Missing floor for detritus and soil pools:** Only plant pool has `MAX(0.0D0, ...)` guard. Detritus and soil could theoretically go negative under extreme scenarios.

2. **Unused variables:** Several variables are declared but appear to be remnants of old code:

   - `CO2_TOTALDEAD_POOL` - allocated but only zeroed
   - `CO2_LAND_POOL` - calculated outside the module
   - `CO2_NETATMOSLANDCO2FLUX` - calculated outside the module

3. **Index boundary:** The code uses `MAX(CURRENT_YEAR_IDX - 2, 1)` for the quadratic extrapolation, which handles the first few years, but the behavior in those years may not be as intended.

4. **Regrowth calculation complexity:** The regrowth calculation is extremely convoluted and difficult to verify. It involves subtracting what didn't happen from what would have happened, then adding back what was subtracted earlier. This is error-prone.

### 9.6 Missing Features

1. **No explicit Q10 parameter:** Temperature feedbacks use exponential form with $\gamma$ parameters, but these are not directly interpretable as Q10 values.

2. **No age/cohort structure:** All plant biomass is treated identically regardless of age.

3. **No explicit distinction between forest types:** A single set of parameters represents all terrestrial ecosystems.

4. **No fire emissions:** Fire is not explicitly modeled; presumably included in land-use emissions.

### 9.7 Documentation Gaps

1. **No scientific references:** Unlike the CH4 module comments that reference TAR tables, the carbon cycle code has almost no references to scientific literature.

2. **TODO comment at line 2:** `! TODO: refactor in carbon cycle functions` - acknowledged but not done.

3. **Cryptic comments:** Many comments describe WHAT the code does but not WHY or what scientific basis exists.

### 9.8 Nitrogen Cycle Interaction

The nitrogen limitation (`NCYCLE_APPLY = 1`) modifies NPP but is called **after** the initial NPP calculation:

```fortran
CO2_CURRENT_NPP(CURRENT_YEAR_IDX) = NCYCLE_LIMIT_FACTOR(CURRENT_YEAR_IDX) * CO2_CURRENT_NPP(CURRENT_YEAR_IDX)
```

This means the nitrogen calculation uses the **pre-limited** NPP internally, which may or may not be intentional.

Also, the nitrogen module's warning:

```fortran
CALL logger % warning('NITROGEN_INIT', "Careful using the nitrogen cycle, it's not finished yet")
```

suggests incomplete implementation.

## 10. Test Cases

### 10.1 Unit Test: Steady State Without Feedbacks

**Purpose:** Verify pools remain constant with zero emissions and no feedbacks.

**Setup:**

```
CO2_PLANTPOOL_INITIAL = 884.86 GtC
CO2_DETRITUSPOOL_INITIAL = 92.77 GtC
CO2_SOILPOOL_INITIAL = 1681.53 GtC
CO2_NPP_INITIAL = 66.27 GtC/yr
CO2_RESPIRATION_INITIAL = 12.26 GtC/yr
CO2_FERTILIZATION_METHOD = 0.0 (no fertilization)
CO2_TEMPFEEDBACK_SWITCH = 0.0 (no temp feedback)
FEEDBACK_TEMPERATURE = 0.0
DAT_CO2B_EMIS = 0.0
DAT_CO2_CONC = 278.0 ppm (constant)
```

**Expected output:**

- All pools should remain within 0.1% of initial values after 100 years
- NPP and respiration should equal initial values
- `CO2_DELTA_TERRPOOLS` should be ~0

### 10.2 Unit Test: CO2 Fertilization Response

**Purpose:** Verify fertilization increases NPP.

**Setup:**

```
Same as 10.1, but:
CO2_FERTILIZATION_METHOD = 1.0 (logarithmic)
CO2_FERTILIZATION_FACTOR = 0.6486
DAT_CO2_CONC = 560.0 ppm (doubled CO2)
```

**Expected output:**

- `CO2_EFF_FERTILIZATION_FACTOR` should equal `1 + 0.6486 * ln(560/278)` = 1.45
- NPP should be ~45% higher than initial
- Plant and soil pools should increase over time

### 10.3 Unit Test: Temperature Feedback Response

**Purpose:** Verify warming increases respiration and decay.

**Setup:**

```
Same as 10.1, but:
CO2_TEMPFEEDBACK_SWITCH = 1.0
FEEDBACK_TEMPERATURE = 2.0 K
```

**Expected output:**

- `CO2_EFF_RESP_TEMPFEEDBACK` = exp(0.0685 * 2.0) = 1.147
- `CO2_EFF_SOIL_TEMPFEEDBACK` = exp(0.1541 * 2.0) = 1.361
- Net carbon loss to atmosphere (pools decrease)
- `CO2_DELTA_TERRPOOLS` should be negative

### 10.4 Unit Test: Deforestation Response

**Purpose:** Verify deforestation removes carbon from pools.

**Setup:**

```
Same as 10.1, but:
DAT_CO2B_EMIS = 2.0 GtC/yr (constant deforestation)
CO2_FRACTION_DEFOREST_PLANT = 0.7
CO2_FRACTION_DEFOREST_DETRITUS = 0.05
```

**Expected output:**

- Plant pool should decrease faster than detritus or soil
- `CO2_GROSSDEFO_EMIS` should be positive
- `CO2_REGROWTH_TOTAL_FLUX` should become increasingly negative (regrowth sink)

### 10.5 Integration Test: Historical Reconstruction

**Purpose:** Verify model produces reasonable land sink over historical period.

**Setup:**

- Run from 1750-2020 with historical CO2 concentrations and land-use emissions
- Use default parameters

**Validation criteria:**

- Cumulative land-use emissions ~200 GtC
- Present-day land sink ~1-3 GtC/yr
- Total terrestrial pool change consistent with observations

### 10.6 Edge Case: Very High CO2

**Purpose:** Test fertilization saturation at high CO2.

**Setup:**

```
DAT_CO2_CONC = 2000.0 ppm
CO2_FERTILIZATION_METHOD = 2.0 (Gifford - should saturate)
```

**Expected output:**

- Fertilization factor should not exceed ~2-3x
- No numerical overflow or NaN

### 10.7 Edge Case: Near-Zero Plant Pool

**Purpose:** Test behavior when plant pool approaches zero.

**Setup:**

```
Very high deforestation rate to deplete plant pool
```

**Expected output:**

- Plant pool should be floored at zero
- Respiration (method 2) should decrease proportionally
- No negative carbon pools

### 10.8 Regression Test: No-Feedback Conservation

**Purpose:** Verify no-feedback pools track land-use emissions exactly.

**Setup:**

- Any realistic scenario with varying land-use emissions

**Expected output:**

- Sum of no-feedback pool changes should equal negative of land-use emissions (within numerical precision)
- Correction term `NOFEED_DIFF_DELTAPOOLS_NETEMIS` should be < 0.01 GtC/yr

## 11. Fortran Code References

### Key File Locations

| Function/Section | File | Line Numbers |
|------------------|------|--------------|
| Module variable declarations | `carbon_cycle.f90` | 1-63 |
| `carboncycle_init` subroutine | `carbon_cycle.f90` | 65-139 |
| TERRCARBON2 main subroutine | `MAGICC7.f90` | 7054-7542 |
| TERRCARBON2 call site | `MAGICC7.f90` | 4820 |
| Feedback temperature calculation | `MAGICC7.f90` | 3826-3858 |
| Post-processing (air2land flux) | `MAGICC7.f90` | 5010-5075 |
| Parameter namelist | `allcfgs.f90` | 36-54 |
| Default values | `MAGCFG_DEFAULTALL.CFG` | 49-80 |

### Key Equations by Line (MAGICC7.f90)

| Equation | Line Numbers |
|----------|--------------|
| CO2 mid-year extrapolation | 7103-7105 |
| Reference CO2 determination | 7111-7135 |
| Temperature feedback factors | 7141-7148 |
| Logarithmic fertilization | 7187-7188 |
| Gifford fertilization | 7203-7224 |
| Sigmoid fertilization | 7179-7183 |
| Fertilization blending | 7232-7240 |
| Turnover time calculation | 7247-7265 |
| NPP with feedbacks | 7270-7271 |
| Respiration (method 1) | 7277-7278 |
| Respiration (method 2) | 7282-7284 |
| Nitrogen limitation application | 7293-7300 |
| Plant pool update | 7357-7362 |
| Detritus pool update | 7407-7409 |
| Soil pool update | 7457-7459 |
| Gross deforestation total | 7464-7465 |
| Delta terrestrial pools | 7504-7507 |
| Total respiration | 7528-7530 |

### Key Equations by Line (carbon_cycle.f90)

| Equation | Line Numbers |
|----------|--------------|
| GTC_PER_PPM hardcode | 88 |
| NPP fraction normalization | 89-98 |
| Deforestation fraction normalization | 100-110 |
| Respiration adjustment guard | 119-124 |
| Initial plant turnover time | 129 |
| Initial detritus turnover time | 131-133 |
| Initial soil turnover time | 135-137 |

---

## Summary

The Terrestrial Carbon Cycle module implements a 4-pool box model with CO2 fertilization and temperature feedbacks. The mathematical formulation is based on standard carbon cycle modeling approaches, but the implementation has significant issues:

**Strengths:**

1. Implicit trapezoidal integration for stability
2. Multiple fertilization formulations available
3. Tracks no-feedback pools for attribution
4. Handles land-use change with regrowth

**Weaknesses:**

1. **Code organization:** Main physics in MAGICC7.f90, not the carbon_cycle module
2. **Hardcoded values:** GTC_PER_PPM and other constants should be parameters
3. **Sign convention confusion:** Detritus feedback factor sign is counterintuitive
4. **Complex regrowth logic:** Difficult to verify correctness
5. **Missing guards:** Detritus and soil pools can potentially go negative
6. **No scientific references:** Code comments lack citations
7. **Incomplete nitrogen cycle:** Warning suggests unfinished feature

For reimplementation, the mathematical formulation is clear, but careful attention should be paid to:

1. The exact fertilization blending logic
2. The regrowth calculation (verify against original)
3. The mass conservation correction
4. The sign conventions for temperature feedbacks
5. Adding proper bounds checking for all pools
