# Module 12: Permafrost Feedback (EXPERIMENTAL)

**STATUS: EXPERIMENTAL - This module is marked as experimental and may not be fit for production use.**

The source code itself contains the warning (line 3 of permafrost.f90):

```fortran
! TODO: update this (all out of date/conflicting with MAGCFG_DEFAULTALL.CFG)
```

## 1. Scientific Purpose

The Permafrost Feedback module simulates the release of carbon (as CO2 and CH4) from thawing permafrost soils in response to Arctic warming. Permafrost contains an estimated 800-1600 GtC of frozen organic matter that can decompose and release greenhouse gases as temperatures rise, creating a positive climate feedback.

The module represents:

1. **Zonal band structure** - The permafrost region is divided into latitudinal bands (default 50), each with a different temperature threshold for thawing
2. **Two soil types** - Mineral soil (faster decomposition) and peat (slower decomposition, more anaerobic)
3. **Two decomposition pathways**:
   - **Aerobic (oxygen-rich)** - Produces CO2 directly
   - **Anaerobic (oxygen-poor)** - Produces CH4 (methane) through methanogenesis
4. **Seasonal temperature cycle** - Summer maximum temperatures determine thawing rates
5. **Arctic amplification** - Temperature changes are scaled by an amplification factor to represent polar amplification of global warming

The key feedback mechanism: As global temperatures rise, Arctic temperatures rise faster (amplification). Summer temperatures exceed melting thresholds in progressively more zonal bands, exposing frozen carbon to decomposition. The released CO2 and CH4 add to atmospheric concentrations, causing further warming.

## 2. Mathematical Formulation

### 2.1 Overview: Carbon Pool Structure

The module tracks carbon in a 2x2x2 matrix structure across N zonal bands:

```
                    FROZEN
                       |
                       | (thawing)
                       v
    +---------+-------------------+
    |         |  AEROBIC (CO2)    |   MINERAL SOIL
    |  THAWED +-------------------+   (80% default)
    |         |  ANAEROBIC (CH4)  |
    +---------+-------------------+
    |         |  AEROBIC (CO2)    |   PEAT
    |  THAWED +-------------------+   (20% default)
    |         |  ANAEROBIC (CH4)  |
    +---------+-------------------+
              x N zonal bands
```

### 2.2 Arctic Temperature and Summer Maximum

The Arctic temperature anomaly is derived from global mean surface temperature using amplification:

$$T_{Arctic}(t) = PF\_ARCTIC\_AMPLIFICATION \times \Delta T_{global}(t)$$

The summer maximum temperature relative to the melting threshold for each zonal band is:

$$T_{summer,max}(t,i) = T_{Arctic}(t) - T_{melt}(i)$$

Where $T_{melt}(i)$ is the melting threshold temperature for zonal band $i$, linearly interpolated between the southernmost and northernmost bands:

$$T_{melt}(i) = PF\_MELTINGTEMP\_MIN + \frac{i-1}{N_{bands}-1} \times (PF\_MELTINGTEMP\_MAX - PF\_MELTINGTEMP\_MIN)$$

Default values: $T_{melt,min} = 1.0$K, $T_{melt,max} = 12.5$K

### 2.3 Thaw/Freeze Rate

The thawing (or refreezing if negative) rate depends on how far the summer temperature exceeds the melting threshold:

$$R_{thaw}(t,i) = sign(T_{summer,max}) \times |T_{summer,max}(t,i)|^{\alpha} \times R_{base}$$

Where:

- $\alpha$ = `PF_x_THAWFREEZE_EXP_TEMP` (default 1.0 for both MS and PEAT)
- $R_{base}$ = `PF_x_THAWFREEZE_PERCPERK_RATE` (fraction per Kelvin per year)

Default rates:

- Mineral soil: 0.1 (10% per K per year)
- Peat: 0.05 (5% per K per year)

**Note:** With exponent = 1.0, this is simply a linear relationship.

### 2.4 Seasonal Cycle and Soil Temperature

A sinusoidal seasonal cycle is applied to determine monthly soil temperatures:

$$T_{soil}(t,i,m) = T_{summer,max}(t,i) + \frac{A_{annual}}{2} \times (\sin(\pi \times (m-1)/11) - 1)$$

Where:

- $m$ = month index (1-12)
- $A_{annual}$ = `PF_TSOILANNUALCYCLE_AMPL` (default 5.0 K)

This creates a seasonal oscillation with maximum at mid-summer (around month 6) and minimum in winter.

### 2.5 Soil Moisture

Soil moisture is a linear function of temperature, bounded between a minimum and 1.0:

$$W_{soil}(t,i,m) = \min(1.0, \max(W_{min}, M \times T_{soil}(t,i,m) + W_{offset}))$$

Where:

- $M$ = `PF_SOILWATER_M` (default 0.02)
- $W_{offset}$ = `PF_SOILWATER_OFFSET` (default 0.2)
- $W_{min}$ = `PF_SOILWATER_MINW` (default 0.2)

A moisture modifier is then calculated:

$$f_{moisture}(W) = \frac{1 - e^{-W}}{1 - e^{-1}}$$

### 2.6 Q10 Temperature Decomposition Response

The decomposition rate depends on temperature via a Q10-like formulation (from Sitch et al. 2003, LPJ):

$$Q_{10}(T) = \exp\left(\alpha \times \left(\frac{1}{T_1} - \frac{1}{T_{soil} + T_2}\right)\right)$$

Where:

- $\alpha$ = `PF_Q10_x_y_ALPHA` (default 308.56 for all four combinations)
- $T_1$ = `PF_Q10_TEMP1` (default 56.02)
- $T_2$ = `PF_Q10_TEMP2` (default 46.02)

**Note:** All four decomposition pathways (MS aerobic, MS anaerobic, PEAT aerobic, PEAT anaerobic) use the same alpha value by default (308.56), meaning there is no differentiation in temperature sensitivity.

### 2.7 Monthly Decomposition Rates

The raw monthly decomposition rates are:

**Aerobic mineral soil (with moisture modifier):**
$$D_{MS,aerob}(m) = \frac{1}{\tau_{MS,aerob}} \times Q_{10,MS,aerob}(m) \times f_{moisture}(m)$$

**Anaerobic mineral soil (no moisture modifier):**
$$D_{MS,anaerob}(m) = \frac{r_{anaerob}}{\tau_{MS,aerob}} \times Q_{10,MS,anaerob}(m)$$

**Aerobic peat:**
$$D_{PEAT,aerob}(m) = \frac{r_{peat}}{\tau_{MS,aerob}} \times Q_{10,PEAT,aerob}(m) \times f_{moisture}(m)$$

**Anaerobic peat:**
$$D_{PEAT,anaerob}(m) = \frac{r_{peat} \times r_{anaerob}}{\tau_{MS,aerob}} \times Q_{10,PEAT,anaerob}(m)$$

Where:

- $\tau_{MS,aerob}$ = `PF_MS_AEROB_DECOMP_TURNOVERTIME` (default 20 years)
- $r_{anaerob}$ = `PF_DECOMPRATE_ANAEROB_OVER_AEROB_RATIO` (default 0.1)
- $r_{peat}$ = `PF_DECOMPRATE_PEAT_OVER_MS_RATIO` (default 0.5)

Annual average decomposition rates are computed as the mean of the 12 monthly values.

### 2.8 Aerobic vs Anaerobic Area Fractions

The fraction of thawed area that is anaerobic (water-saturated) depends on moisture:

$$f_{anaerob}(m) = \max\left(0, \min\left(f_{anaerob,max}, f_{anaerob,init} + (f_{anaerob,max} - f_{anaerob,init}) \times f_{moisture}(m) \times S_{moist}\right)\right)$$

Where:

- $f_{anaerob,init}$ = `PF_x_ANAEROB_INITIAL_AREAFRACTION`
- $f_{anaerob,max}$ = `PF_x_ANAEROB_MAX_AREAFRACTION`
- $S_{moist}$ = `PF_x_ANAEROB_MOISTSENS` (default 0.0, i.e., disabled)

Default values:

| Parameter | Mineral Soil | Peat |
|-----------|--------------|------|
| Initial anaerobic fraction | 0.05 | 0.8 |
| Max anaerobic fraction | 0.3 | 0.9 |
| Moisture sensitivity | 0.0 | 0.0 |

**Note:** With moisture sensitivity = 0.0, the anaerobic fractions remain at their initial values and do not respond to moisture changes. This effectively disables the dynamic aerobic/anaerobic partitioning.

### 2.9 Carbon Pool Updates

The module tracks frozen area and thawed (aerobic + anaerobic) areas, each with associated carbon pools. The update sequence each year:

1. **Calculate new thaw area** from thaw rate and current frozen area
2. **Transfer carbon** from frozen pool to thawed pools proportionally to area
3. **Partition thawed area** between aerobic and anaerobic based on area fractions
4. **Transfer carbon** between aerobic and anaerobic pools if fractions change
5. **Calculate decomposition** (emissions) from thawed pools
6. **Update pools** by subtracting emissions

### 2.10 Methane Oxidation

Methane produced in anaerobic decomposition is partially oxidized before reaching the atmosphere:

- In-soil oxidation: `PF_x_CH4OXIDISATION_FRACTION` (MS: 25%, PEAT: 60%)
- Atmospheric oxidation produces CO2: `PF_CO2FROMCH4OXINATM_FRACTION` (default 1.0)

The net CH4 emissions are:

$$E_{CH4} = \sum_{i} E_{anaerob}(i) \times \frac{16000}{12 \times 2} \times (1 - f_{ox})$$

Where:

- The factor 16000/12 converts from GtC to MtCH4
- The factor 1/2 accounts for methanogenesis producing equal parts CO2 and CH4
- $f_{ox}$ = oxidation fraction

The CO2 emissions include both direct aerobic emissions and CO2 from methanogenesis:

$$E_{CO2} = \sum_{i} E_{aerob}(i) + \sum_{i} E_{anaerob}(i) \times \frac{1 + f_{ox}}{2}$$

### 2.11 Zonal Pool Distribution

The carbon pool can be distributed non-uniformly across zonal bands using `PF_ZONAL_POOLDISTR`:

- 0.0 = Equal distribution (rectangle)
- -1.0 to 0.0 = More carbon in northern bands (triangle pointing south)
- 0.0 to +1.0 = More carbon in southern bands (triangle pointing north)

The fraction of carbon in band $i$ for negative distribution parameter $d$:

$$f_{pool}(i) = \frac{(1+d)/N - d \times i/N^2}{1 + d/2 - d/(2N)}$$

## 3. State Variables

| Variable | Fortran Name | Dimensions | Units | Description |
|----------|--------------|------------|-------|-------------|
| MS Aerobic Area | `PF_MS_AEROB_AREA` | (NYEARS, NBANDS) | fraction | Thawed mineral soil area under aerobic conditions |
| MS Anaerobic Area | `PF_MS_ANAEROB_AREA` | (NYEARS, NBANDS) | fraction | Thawed mineral soil area under anaerobic conditions |
| Peat Aerobic Area | `PF_PEAT_AEROB_AREA` | (NYEARS, NBANDS) | fraction | Thawed peat area under aerobic conditions |
| Peat Anaerobic Area | `PF_PEAT_ANAEROB_AREA` | (NYEARS, NBANDS) | fraction | Thawed peat area under anaerobic conditions |
| MS Aerobic Pool | `PF_MS_AEROB_POOL` | (NYEARS, NBANDS) | GtC | Carbon in thawed aerobic mineral soil |
| MS Anaerobic Pool | `PF_MS_ANAEROB_POOL` | (NYEARS, NBANDS) | GtC | Carbon in thawed anaerobic mineral soil |
| Peat Aerobic Pool | `PF_PEAT_AEROB_POOL` | (NYEARS, NBANDS) | GtC | Carbon in thawed aerobic peat |
| Peat Anaerobic Pool | `PF_PEAT_ANAEROB_POOL` | (NYEARS, NBANDS) | GtC | Carbon in thawed anaerobic peat |
| MS Frozen Area | `PF_MS_FROZEN_AREA` | (NYEARS, NBANDS) | fraction | Frozen mineral soil area |
| Peat Frozen Area | `PF_PEAT_FROZEN_AREA` | (NYEARS, NBANDS) | fraction | Frozen peat area |
| MS Frozen Pool | `PF_MS_FROZEN_POOL` | (NYEARS, NBANDS) | GtC | Carbon in frozen mineral soil |
| Peat Frozen Pool | `PF_PEAT_FROZEN_POOL` | (NYEARS, NBANDS) | GtC | Carbon in frozen peat |
| Summer Max Temp | `PF_SUMMERMAX_TEMP` | (NYEARS, NBANDS) | K | Summer maximum temperature anomaly per band |
| MS Decomp Rate | `PF_MS_AEROB_DECOMPRATE` | (NYEARS, NBANDS) | yr^-1 | Annual decomposition rate, MS aerobic |
| MS Decomp Rate (anaerob) | `PF_MS_ANAEROB_DECOMPRATE` | (NYEARS, NBANDS) | yr^-1 | Annual decomposition rate, MS anaerobic |
| Peat Decomp Rate | `PF_PEAT_AEROB_DECOMPRATE` | (NYEARS, NBANDS) | yr^-1 | Annual decomposition rate, peat aerobic |
| Peat Decomp Rate (anaerob) | `PF_PEAT_ANAEROB_DECOMPRATE` | (NYEARS, NBANDS) | yr^-1 | Annual decomposition rate, peat anaerobic |
| MS Emissions (aerob) | `PF_MS_AEROB_EMIS` | (NYEARS, NBANDS) | GtC/yr | CO2 emissions from MS aerobic |
| MS Emissions (anaerob) | `PF_MS_ANAEROB_EMIS` | (NYEARS, NBANDS) | GtC/yr | CH4 emissions (as C) from MS anaerobic |
| Peat Emissions (aerob) | `PF_PEAT_AEROB_EMIS` | (NYEARS, NBANDS) | GtC/yr | CO2 emissions from peat aerobic |
| Peat Emissions (anaerob) | `PF_PEAT_ANAEROB_EMIS` | (NYEARS, NBANDS) | GtC/yr | CH4 emissions (as C) from peat anaerobic |
| Total Pool | `PF_TOT_TOT_POOL` | (NYEARS) | GtC | Total remaining carbon (frozen + thawed) |
| Pool + Emissions | `PF_TOTPLUSEMIS_POOL` | (NYEARS) | GtC | Pool plus cumulative emissions (conservation check) |
| Thawed Area (total) | `PF_TOT_AREATHAWED` | (NYEARS) | fraction | Total thawed area (carbon-weighted) |

## 4. Parameters

### 4.1 Master Switch and Structure

| Parameter | Fortran Name | Units | Default | Description |
|-----------|--------------|-------|---------|-------------|
| Apply permafrost | `PF_APPLY` | flag | 1 | Enable/disable module |
| Number of bands | `PF_NBANDS` | integer | 50 | Number of zonal latitude bands |

### 4.2 Temperature and Thawing

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Melting temp (min) | `PF_MELTINGTEMP_MIN` | K | 1.0 | 0-5 | Arctic warming threshold for southern band |
| Melting temp (max) | `PF_MELTINGTEMP_MAX` | K | 12.5 | 5-15 | Arctic warming threshold for northern band |
| Arctic amplification | `PF_ARCTIC_AMPLIFICATION` | ratio | 1.7 | 1.5-3.0 | Ratio of Arctic to global warming |
| Annual cycle amplitude | `PF_TSOILANNUALCYCLE_AMPL` | K | 5.0 | 2-10 | Seasonal temperature swing |
| MS thaw rate | `PF_MS_THAWFREEZE_PERCPERK_RATE` | K^-1 | 0.1 | 0.01-0.5 | Thawing rate per degree per year (MS) |
| Peat thaw rate | `PF_PEAT_THAWFREEZE_PERCPERK_RATE` | K^-1 | 0.05 | 0.01-0.5 | Thawing rate per degree per year (peat) |
| MS thaw exponent | `PF_MS_THAWFREEZE_EXP_TEMP` | - | 1.0 | 0.5-2.0 | Temperature exponent for thaw rate (MS) |
| Peat thaw exponent | `PF_PEAT_THAWFREEZE_EXP_TEMP` | - | 1.0 | 0.5-2.0 | Temperature exponent for thaw rate (peat) |

### 4.3 Carbon Pools and Distribution

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| Total pool | `PF_TOT_POOL` | GtC | 800.0 | 500-1600 | Total carbon in permafrost region |
| MS fraction (south) | `PF_MINSOIL_SOUTHERN_POOLFRACTION` | fraction | 0.8 | 0.5-1.0 | Mineral soil fraction in southern bands |
| MS fraction (north) | `PF_MINSOIL_NORTHERN_POOLFRACTION` | fraction | 0.8 | 0.5-1.0 | Mineral soil fraction in northern bands |
| Zonal distribution | `PF_ZONAL_POOLDISTR` | - | 0.0 | -1 to +1 | North-south carbon distribution shape |

### 4.4 Decomposition Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|--------------|-------|---------|-------------|-------------|
| MS aerob turnover | `PF_MS_AEROB_DECOMP_TURNOVERTIME` | years | 20.0 | 5-100 | Base turnover time for MS aerobic |
| Peat/MS ratio | `PF_DECOMPRATE_PEAT_OVER_MS_RATIO` | ratio | 0.5 | 0.1-1.0 | Peat decay relative to MS decay |
| Anaerob/aerob ratio | `PF_DECOMPRATE_ANAEROB_OVER_AEROB_RATIO` | ratio | 0.1 | 0.01-0.5 | Anaerobic decay relative to aerobic |

### 4.5 Q10 Temperature Sensitivity

| Parameter | Fortran Name | Units | Default | Description |
|-----------|--------------|-------|---------|-------------|
| Q10 alpha (MS aerob) | `PF_Q10_MS_AEROB_ALPHA` | K | 308.56 | Q10 exponent for MS aerobic |
| Q10 alpha (MS anaerob) | `PF_Q10_MS_ANAEROB_ALPHA` | K | 308.56 | Q10 exponent for MS anaerobic |
| Q10 alpha (peat aerob) | `PF_Q10_PEAT_AEROB_ALPHA` | K | 308.56 | Q10 exponent for peat aerobic |
| Q10 alpha (peat anaerob) | `PF_Q10_PEAT_ANAEROB_ALPHA` | K | 308.56 | Q10 exponent for peat anaerobic |
| Q10 temp1 | `PF_Q10_TEMP1` | K | 56.02 | Reference temperature 1 |
| Q10 temp2 | `PF_Q10_TEMP2` | K | 46.02 | Reference temperature 2 |

### 4.6 Soil Moisture

| Parameter | Fortran Name | Units | Default | Description |
|-----------|--------------|-------|---------|-------------|
| Moisture slope | `PF_SOILWATER_M` | K^-1 | 0.02 | Temperature-moisture slope |
| Moisture offset | `PF_SOILWATER_OFFSET` | fraction | 0.2 | Base moisture level |
| Minimum moisture | `PF_SOILWATER_MINW` | fraction | 0.2 | Lower bound for moisture |

### 4.7 Aerobic/Anaerobic Partitioning

| Parameter | Fortran Name | Units | Default | Description |
|-----------|--------------|-------|---------|-------------|
| MS anaerob initial | `PF_MS_ANAEROB_INITIAL_AREAFRACTION` | fraction | 0.05 | Initial anaerobic fraction (MS) |
| MS anaerob max | `PF_MS_ANAEROB_MAX_AREAFRACTION` | fraction | 0.3 | Maximum anaerobic fraction (MS) |
| MS anaerob sensitivity | `PF_MS_ANAEROB_MOISTSENS` | - | 0.0 | Moisture sensitivity for anaerobic (MS) |
| Peat anaerob initial | `PF_PEAT_ANAEROB_INITIAL_AREAFRACTION` | fraction | 0.8 | Initial anaerobic fraction (peat) |
| Peat anaerob max | `PF_PEAT_ANAEROB_MAX_AREAFRACTION` | fraction | 0.9 | Maximum anaerobic fraction (peat) |
| Peat anaerob sensitivity | `PF_PEAT_ANAEROB_MOISTSENS` | - | 0.0 | Moisture sensitivity for anaerobic (peat) |

### 4.8 Methane Oxidation

| Parameter | Fortran Name | Units | Default | Description |
|-----------|--------------|-------|---------|-------------|
| MS CH4 oxidation | `PF_MS_CH4OXIDISATION_FRACTION` | fraction | 0.25 | Fraction oxidized in soil (MS) |
| Peat CH4 oxidation | `PF_PEAT_CH4OXIDISATION_FRACTION` | fraction | 0.6 | Fraction oxidized in soil (peat) |
| Atm oxidation to CO2 | `PF_CO2FROMCH4OXINATM_FRACTION` | fraction | 1.0 | Atmospheric CH4 oxidation to CO2 |

## 5. Inputs (per timestep)

| Variable | Units | Source | Fortran Variable |
|----------|-------|--------|------------------|
| Global surface temperature | K | Climate module | `DAT_SURFACE_TEMP%DATGLOBE(CURRENT_YEAR_IDX)` |
| Current year index | integer | Time module | `CURRENT_YEAR_IDX` |
| Next year index | integer | Time module | `NEXT_YEAR_IDX` |

## 6. Outputs (per timestep)

| Variable | Units | Destination | Fortran Variable |
|----------|-------|-------------|------------------|
| CO2 emissions | GtC/yr | CO2 budget | `DAT_CO2PF_EMIS%DATBOX(CURRENT_YEAR_IDX,2)` |
| CH4 emissions | MtCH4/yr | CH4 budget | `DAT_CH4PF_EMIS%DATBOX(CURRENT_YEAR_IDX,2)` |
| Thawed area fraction | fraction | Diagnostics | `DAT_PFTHAWED_AREA%DATBOX(NEXT_YEAR_IDX,2)` |

The emissions are assigned to box 2 (Northern Hemisphere land) and copied to the global totals.

## 7. Algorithm (Pseudocode)

### 7.1 Initialization (`permafrost_init`)

```
IF PF_APPLY == 1:
    # Calculate zonal pool distribution
    FOR each band i:
        PF_FRACPOOL(i) = calculate_fraction(PF_ZONAL_POOLDISTR, i, N_BANDS)

    # Calculate melting thresholds
    FOR each band i:
        PF_MELTING_TEMP(i) = PF_MELTINGTEMP_MIN +
            (i-1)/(N_BANDS-1) * (PF_MELTINGTEMP_MAX - PF_MELTINGTEMP_MIN)

    # Initialize potential pools
    FOR each band i:
        ms_fraction = PF_MINSOIL_SOUTHERN_POOLFRACTION +
            (i-1)/(N_BANDS-1) * (PF_MINSOIL_NORTHERN_POOLFRACTION - PF_MINSOIL_SOUTHERN_POOLFRACTION)
        PF_MS_POT_POOL(i) = ms_fraction * PF_TOT_POOL * PF_FRACPOOL(i)
        PF_PEAT_POT_POOL(i) = PF_TOT_POOL * PF_FRACPOOL(i) - PF_MS_POT_POOL(i)

    # Initialize frozen area and pools (100% frozen at start)
    PF_MS_FROZEN_AREA(1,:) = 1.0
    PF_PEAT_FROZEN_AREA(1,:) = 1.0
    PF_MS_FROZEN_POOL(1,:) = PF_MS_POT_POOL
    PF_PEAT_FROZEN_POOL(1,:) = PF_PEAT_POT_POOL
```

### 7.2 Main Timestep (`permafrost_calc`)

```
FOR each zonal band i:
    # STEP 1: Calculate summer maximum temperature
    T_summer_max(i) = PF_ARCTIC_AMPLIFICATION * T_global - PF_MELTING_TEMP(i)

    # STEP 2: Calculate thaw/freeze rate
    R_MS(i) = sign(T_summer_max) * |T_summer_max|^exp * PF_MS_THAWFREEZE_PERCPERK_RATE
    R_PEAT(i) = sign(T_summer_max) * |T_summer_max|^exp * PF_PEAT_THAWFREEZE_PERCPERK_RATE

    # STEP 3: Calculate monthly values
    FOR each month m:
        T_soil(m) = T_summer_max(i) + seasonal_cycle(m)
        W_soil(m) = bounded_linear(T_soil(m))
        moisture_mod(m) = (1 - exp(-W_soil)) / (1 - exp(-1))

        Q10_MS_aerob(m) = exp(alpha * (1/T1 - 1/(T_soil + T2)))
        # ... similar for other Q10 values

        D_MS_aerob(m) = (1/tau) * Q10_MS_aerob(m) * moisture_mod(m)
        D_MS_anaerob(m) = (r_anaerob/tau) * Q10_MS_anaerob(m)
        # ... similar for peat

        f_anaerob_MS(m) = calculate_anaerob_fraction(moisture_mod(m), MS params)
        f_anaerob_PEAT(m) = calculate_anaerob_fraction(moisture_mod(m), PEAT params)

    # STEP 4: Annual averages
    D_annual_MS_aerob(i) = mean(D_MS_aerob)
    D_annual_MS_anaerob(i) = mean(D_MS_anaerob)
    # ... etc
    f_aerob_MS(i) = 1 - mean(f_anaerob_MS)
    f_aerob_PEAT(i) = 1 - mean(f_anaerob_PEAT)

    # STEP 5: Update aerob/anaerob areas from frozen thaw
    thawed_fraction = 1 - FROZEN_AREA(current)
    AEROB_AREA(next,i) = f_aerob * thawed_fraction
    ANAEROB_AREA(next,i) = (1 - f_aerob) * thawed_fraction

    # STEP 6: Redistribute carbon between aerob and anaerob pools
    # (complex bookkeeping to maintain carbon density)

    # STEP 7: Calculate new thaw area from frozen
    new_area_aerob = R_thaw * f_aerob * FROZEN_AREA * dt
    new_area_anaerob = R_thaw * (1-f_aerob) * FROZEN_AREA * dt
    # (bounded to not exceed frozen area or go negative)

    # STEP 8: Transfer carbon from frozen to thawed pools
    carbon_density_frozen = FROZEN_POOL / FROZEN_AREA
    new_pool_aerob = carbon_density_frozen * new_area_aerob
    new_pool_anaerob = carbon_density_frozen * new_area_anaerob

    # STEP 9: Update pools
    AEROB_POOL(next) = AEROB_POOL(next) + new_pool_aerob
    ANAEROB_POOL(next) = ANAEROB_POOL(next) + new_pool_anaerob
    FROZEN_POOL(next) = FROZEN_POOL(current) - new_pool_aerob - new_pool_anaerob
    FROZEN_AREA(next) = FROZEN_AREA(current) - new_area_aerob - new_area_anaerob

    # STEP 10: Calculate emissions (central differencing)
    E_aerob(i) = D_annual_aerob * (POOL(next) + POOL(current)) / 2
    E_anaerob(i) = D_annual_anaerob * (POOL(next) + POOL(current)) / 2

    # STEP 11: Update pools by subtracting emissions
    POOL(next) = POOL(next) - E

# STEP 12: Aggregate emissions
CO2_total = sum(E_MS_aerob) + sum(E_PEAT_aerob)
          + sum(E_MS_anaerob) * (1 + ox_MS) / 2
          + sum(E_PEAT_anaerob) * (1 + ox_PEAT) / 2

CH4_total = sum(E_MS_anaerob) * 16000/12 / 2 * (1 - ox_MS)
          + sum(E_PEAT_anaerob) * 16000/12 / 2 * (1 - ox_PEAT)

# STEP 13: Conservation check
TOTAL_POOL = sum(all frozen + thawed pools)
TOTAL_PLUS_EMISSIONS = TOTAL_POOL + cumulative(CO2 + CH4 as C)
# Should equal PF_TOT_POOL

# STEP 14: Calculate thawed area (carbon-weighted)
THAWED_AREA = 1 - sum(FROZEN_AREA * POT_POOL) / PF_TOT_POOL
```

## 8. Numerical Considerations

### 8.1 Integration Scheme

The module uses a semi-implicit scheme with central differencing for emissions:

$$E(t) = D(t) \times \frac{C(t) + C(t+1)}{2}$$

This provides better stability than forward Euler but is not fully implicit.

### 8.2 Area and Pool Bounds

All areas are bounded using MAX(0, ...) operations:

```fortran
PF_MS_FROZEN_AREA(NEXT_YEAR_IDX, I) = MAX(0.0D0, ...)
PF_MS_AEROB_POOL(NEXT_YEAR_IDX, I) = MAX(0.0D0, ...)
```

This prevents negative values but may cause mass conservation issues in extreme cases.

### 8.3 Division by Zero Protection

The code includes checks for zero-area conditions:

```fortran
IF (PF_MS_FROZEN_AREA(CURRENT_YEAR_IDX, I) == 0.0D0) THEN
    PF_MS_AEROB_NEWPOOL = 0.0D0
ELSE
    PF_MS_AEROB_NEWPOOL = PF_MS_FROZEN_POOL(...) / PF_MS_FROZEN_AREA(...) * ...
END IF
```

However, these use exact equality (== 0.0D0) rather than a tolerance, which could fail for very small but non-zero areas.

### 8.4 Monthly Loop Overhead

The module loops over 12 months for every zonal band and every year to compute seasonal averages. With 50 bands, this is 600 iterations per year. The monthly values could be precomputed or simplified.

## 9. Issues and Concerns (EXPERIMENTAL STATUS)

### 9.1 Critical: Documentation Mismatch

**Line 3 of permafrost.f90 explicitly states:**

```fortran
! TODO: update this (all out of date/conflicting with MAGCFG_DEFAULTALL.CFG)
```

This means the module's internal documentation does not match the configuration file. The parameter descriptions in the module header may be incorrect or outdated. This is a fundamental documentation problem that undermines confidence in the implementation.

### 9.2 Identical Q10 Parameters

All four decomposition pathways use the same Q10 alpha value (308.56):

| Pool Type | Default Alpha |
|-----------|---------------|
| MS Aerobic | 308.56 |
| MS Anaerobic | 308.56 |
| Peat Aerobic | 308.56 |
| Peat Anaerobic | 308.56 |

This means there is no differentiation in temperature sensitivity between:

- Mineral soil vs. peat
- Aerobic vs. anaerobic conditions

Physically, anaerobic decomposition (methanogenesis) should have different temperature sensitivity than aerobic respiration.

### 9.3 Disabled Moisture Dynamics

The moisture sensitivity parameters are all set to 0.0 by default:

- `PF_MS_ANAEROB_MOISTSENS = 0.0`
- `PF_PEAT_ANAEROB_MOISTSENS = 0.0`

This means the aerobic/anaerobic partitioning does not respond to moisture changes. The module computes moisture but effectively ignores it for area partitioning.

### 9.4 Simple Linear Thaw Model

The thaw rate is simply linear in temperature (exponent = 1.0). Real permafrost thaw involves:

- Active layer deepening (nonlinear with temperature)
- Talik formation (abrupt transitions)
- Thermokarst processes (positive feedback)
- Ice-wedge melting (abrupt subsidence)

The linear approximation may underestimate nonlinear thaw acceleration.

### 9.5 No Refreezing Dynamics

While the code allows for refreezing (negative thaw rates), the physics of refreezing are different from thawing:

- Refreezing takes much longer than thawing
- Refrozen carbon may be more labile
- Hysteresis effects are not captured

### 9.6 Uniform Carbon Density Assumption

Carbon transfer between pools assumes uniform carbon density within each pool:

```fortran
PF_MS_AEROB_NEWPOOL = PF_MS_FROZEN_POOL(...) / PF_MS_FROZEN_AREA(...) * PF_MS_AEROB_NEWAREA
```

In reality, carbon density varies with depth and location. Newly thawed shallow permafrost may have different carbon content than deep permafrost.

### 9.7 No Thermokarst Representation

Thermokarst (abrupt permafrost collapse forming lakes) is a major CH4 emission pathway. The module's gradual thaw model does not capture:

- Abrupt lake formation
- Lake edge erosion
- Anaerobic conditions in lake sediments

### 9.8 No Vegetation Response

Vegetation changes in response to warming are not modeled:

- Shrub expansion affects albedo and snow retention
- Boreal forest advance changes carbon inputs
- Changed evapotranspiration affects soil moisture

### 9.9 Conservation Issues with MAX(0,...) Clipping

The use of MAX(0.0D0, ...) for all pool and area updates can cause mass conservation violations:

- If a pool would go negative (due to excessive decomposition), it is clipped to zero
- The "missing" carbon is not accounted for
- The diagnostic `PF_TOTPLUSEMIS_POOL` may drift from `PF_TOT_POOL`

### 9.10 Hardcoded Box Assignment

Emissions are always assigned to box 2 (Northern Hemisphere land):

```fortran
DAT_CO2PF_EMIS % DATBOX(CURRENT_YEAR_IDX, 2) = ...
```

This is physically correct but hardcoded. The module cannot handle permafrost in other regions (e.g., Antarctic or high-altitude permafrost).

### 9.11 Pi Variable Naming

The variable `PI` is misleadingly named:

```fortran
REAL(8) :: PI = ACOS(0.0D0)  ! Actually stores pi/2, not pi
```

`ACOS(0.0D0)` returns `pi/2` (approximately 1.5708), not `pi`. However, this is **intentional design**, not a bug. The seasonal cycle calculation:

```fortran
PF_TSOIL_X(M) = REAL(M - 1, 8) * PI / 11.0D0
```

Uses `pi/2 / 11` to create the correct phase spacing for 12 monthly values (M=1 to 12), producing angles from 0 to `pi/2` which gives a quarter sine wave cycle. The mathematics is correct; only the variable name is confusing. A better name would be `HALF_PI` or `PI_OVER_2`.

### 9.12 SAVE Attribute Overuse

Many local variables in `permafrost_calc` have the SAVE attribute:

```fortran
REAL(8), DIMENSION(12), SAVE :: PF_TSOIL_X, PF_TSOIL_EFF, ...
```

This means they persist between calls, which could cause issues if the module is called from multiple threads or with different configurations.

### 9.13 Lack of Scientific Validation

The module lacks:

- References to peer-reviewed literature
- Comparison with process-based permafrost models
- Validation against observed permafrost carbon fluxes
- Uncertainty characterization

The comment in MAGICC7.f90 (line 15) explicitly warns:

```fortran
!   and not fit-for-use. This includes sea level rise projections, the permafrost module,
```

## 10. Test Cases

### 10.1 Unit Test: No Warming

**Purpose:** Verify no thaw occurs without temperature increase.

**Setup:**

```
PF_APPLY = 1
PF_TOT_POOL = 800.0
DAT_SURFACE_TEMP = 0.0 (constant)
```

**Expected:**

- All frozen areas remain 1.0
- All thawed areas remain 0.0
- All emissions are zero
- `PF_TOT_TOT_POOL` remains 800.0

### 10.2 Unit Test: Single-Band Thaw

**Purpose:** Verify basic thaw mechanics for southernmost band.

**Setup:**

```
PF_NBANDS = 1
PF_MELTINGTEMP_MIN = 1.0
PF_ARCTIC_AMPLIFICATION = 1.7
DAT_SURFACE_TEMP = 1.0 K (constant, so T_arctic = 1.7 K)
```

**Expected:**

- Summer max temp = 1.7 - 1.0 = 0.7 K
- Thaw rate = 0.7 * 0.1 = 0.07 per year (7%)
- After 10 years, ~70% of area should be thawed
- Some emissions should occur

### 10.3 Unit Test: Conservation Check

**Purpose:** Verify carbon conservation.

**Setup:**

```
Any warming scenario, run 100 years
```

**Expected:**

- `PF_TOT_TOT_POOL + cumulative_CO2 + cumulative_CH4_as_C = PF_TOT_POOL` (within tolerance)
- `PF_TOTPLUSEMIS_POOL(t)` should be approximately constant

### 10.4 Unit Test: CH4 vs CO2 Partitioning

**Purpose:** Verify correct partitioning between aerobic and anaerobic emissions.

**Setup:**

```
PF_MINSOIL_SOUTHERN_POOLFRACTION = 0.5 (equal MS and peat)
PF_MS_ANAEROB_INITIAL_AREAFRACTION = 0.1
PF_PEAT_ANAEROB_INITIAL_AREAFRACTION = 0.9
```

**Expected:**

- MS emissions mostly CO2 (90% aerobic)
- Peat emissions mostly CH4 (90% anaerobic)
- CH4 oxidation reduces net CH4 by 25% (MS) and 60% (peat)

### 10.5 Unit Test: Zonal Band Distribution

**Purpose:** Verify carbon distribution across bands.

**Setup:**

```
PF_NBANDS = 10
PF_ZONAL_POOLDISTR = -1.0 (most carbon in north)
Moderate warming to thaw only southern bands
```

**Expected:**

- Southern bands thaw first
- But have less carbon, so emissions are smaller
- Northern bands with more carbon thaw later

### 10.6 Integration Test: Historical + Scenario

**Purpose:** Verify reasonable emissions under typical scenario.

**Setup:**

```
Run with historical temperatures 1850-2020
Then RCP4.5 scenario to 2100
```

**Validation:**

- Check emissions are in reasonable range (literature: 50-200 GtC cumulative by 2100)
- Check thawed area is reasonable (literature: 30-70% by 2100 under RCP4.5)

### 10.7 Edge Case: Complete Thaw

**Purpose:** Test behavior when all permafrost thaws.

**Setup:**

```
Very high sustained warming (e.g., 10 K global = 17 K Arctic)
Run until all bands thawed
```

**Expected:**

- All frozen areas approach zero
- Thawed pools eventually deplete through decomposition
- No numerical instabilities
- Total emissions approach `PF_TOT_POOL`

### 10.8 Edge Case: Cooling After Warming

**Purpose:** Test refreezing behavior.

**Setup:**

```
Initial warming period (partial thaw)
Then cooling (negative temperature anomaly)
```

**Expected:**

- Thawed areas should decrease (refreeze)
- Carbon transfers from thawed to frozen pools
- Emissions decrease as temperature drops

## 11. Fortran Code References

### Key File Locations

| Function/Section | File | Line Numbers |
|------------------|------|--------------|
| Module definition | `permafrost.f90` | 1-931 |
| Parameter declarations | `permafrost.f90` | 78-95 |
| Variable declarations | `permafrost.f90` | 100-118 |
| `permafrost_alloc` | `permafrost.f90` | 123-145 |
| `permafrost_init` | `permafrost.f90` | 148-284 |
| `permafrost_calc` | `permafrost.f90` | 308-772 |
| `permafrost_dealloc` | `permafrost.f90` | 287-306 |
| `permafrost_write_output` | `permafrost.f90` | 774-836 |
| `permafrost_write_datafield` | `permafrost.f90` | 843-930 |
| Parameter namelist | `allcfgs.f90` | 275-290 |
| Call site (alloc/init) | `MAGICC7.f90` | 524-526 |
| Call site (calc) | `MAGICC7.f90` | 3921-3925 |
| CH4 emissions integration | `MAGICC7.f90` | 4022-4024 |
| CO2 emissions integration | `MAGICC7.f90` | 7513-7517 |
| Call site (dealloc) | `MAGICC7.f90` | 12491-12492 |
| Default parameters | `MAGCFG_DEFAULTALL.CFG` | 385-419 |

### Key Equations by Line (permafrost.f90)

| Equation | Line Numbers |
|----------|--------------|
| Summer max temperature | 331-335 |
| Thaw rate calculation | 338-347 |
| Seasonal cycle | 352-356 |
| Soil temperature | 360 |
| Moisture calculation | 362-364 |
| Moisture modifier | 365-366 |
| Q10 calculation | 369-380 |
| Monthly decomposition rates | 383-395 |
| Anaerobic area fraction | 400-410 |
| Annual average decomposition | 415-418 |
| Area updates (aerob/anaerob) | 427-437 |
| Carbon redistribution | 440-517 |
| New thaw area calculation | 525-537 |
| Carbon pool updates | 540-596 |
| Emissions calculation | 689-716 |
| Total CO2 emissions | 723-724, 737-742 |
| Total CH4 emissions | 730-734 |
| Conservation diagnostic | 749-758 |
| Thawed area diagnostic | 762-771 |

---

## Summary

The Permafrost Feedback module is a simplified representation of permafrost carbon release that:

**Captures:**

- Zonal gradients in permafrost thaw vulnerability
- Distinction between mineral soil and peat
- Aerobic (CO2) vs. anaerobic (CH4) decomposition pathways
- Temperature-dependent decomposition rates
- Seasonal temperature cycles
- Arctic amplification of global warming

**Does not capture:**

- Nonlinear thaw processes (thermokarst, talik formation)
- Depth-dependent carbon profiles
- Vegetation-permafrost interactions
- Microbial community dynamics
- Methane ebullition from lakes

**Key concerns:**

1. Documentation explicitly states it is out of date
2. All Q10 parameters are identical (no process differentiation)
3. Moisture dynamics are effectively disabled
4. Linear thaw model may miss nonlinear feedbacks
5. Module marked as "not fit for use" in MAGICC7.f90 comments

This module should be used with caution and its results should be treated as indicative rather than definitive. Any scientific application should include extensive sensitivity analysis and comparison with process-based permafrost models.
