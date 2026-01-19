# Module 14: Sea Level Rise

## 1. Scientific Purpose

The Sea Level Rise (SLR) module calculates global mean sea level rise from multiple contributing components that respond to climate change. Sea level rise is a critical impact metric for climate projections, affecting coastal communities, infrastructure, and ecosystems worldwide. The module aggregates contributions from:

1. **Thermal expansion** - Warming of ocean water causes volumetric expansion
2. **Glaciers and ice caps** - Mountain glaciers and small ice caps melt in response to warming
3. **Greenland Ice Sheet (GIS)** - Both surface mass balance (SMB) changes and solid ice discharge (SID)
4. **Antarctic Ice Sheet (AIS)** - Both SMB changes and SID with regional basin differentiation
5. **Land water storage** - Human extraction of groundwater contributing to sea level

The module provides a semi-empirical framework combining process-based parameterizations with empirical calibrations from CMIP5 models and ice sheet model ensembles. Each component can be activated independently with separate start years, allowing flexible configuration for different assessment purposes.

## 2. Experimental Status Assessment

**STATUS: EXPERIMENTAL - NOT FIT FOR PRODUCTION USE**

The code explicitly warns (MAGICC7.f90 lines 14-16):
> "There are several components in this program that are of experimental nature and not fit-for-use. This includes sea level rise projections..."

### Known Limitations

1. **Semi-empirical approach**: The module uses simplified parameterizations rather than full physics-based ice sheet models
2. **Limited validation**: Ice sheet dynamics, particularly for Antarctica, are highly uncertain
3. **No marine ice cliff instability**: The DeConto parameterization includes a "fast rate" threshold but does not capture full ice cliff dynamics
4. **Regional homogeneity assumption**: Results are globally uniform (same SLR applied to all ocean boxes)
5. **Levermann AIS method uses convolution**: Complex impulse response function approach that may accumulate numerical errors over long simulations
6. **Equilibrium glacier tables**: Glaciers use static lookup tables (104 points) that cap at 10.3 degrees warming
7. **Land water uses prescribed time series**: Limited to 1900-present historical data extrapolated forward

### Code Quality Issues

1. **Hardcoded magic numbers**: `-361.0` divisor for Fettweis GIS SMB (line 401), `1000.0` conversion (line 657)
2. **Mixed parameterization options**: Two different GIS SMB methods ("DEFAULT" vs "FETTWEIS"), two AIS SID methods ("DECONTO" vs "LEVERMANN")
3. **Complex index arithmetic**: Multiple year indexing schemes (CURRENT_YEAR_IDX, NEXT_YEAR_IDX, AIS_SID_IDX) that are error-prone
4. **SIGN() function usage**: Unusual usage pattern for handling negative temperature anomalies
5. **No conservation checks**: No verification that total ice volume doesn't go negative (relies on MAX(0, ...) guards)

## 3. Mathematical Formulation

### 3.1 Thermal Expansion

Thermal expansion calculates the volumetric change of ocean water due to temperature changes at each ocean layer.

**Expansion coefficient calculation** (empirical polynomial):

```
SLR_EXPANSION_COEFE = SLR_EXPANSION_PARAMS(1) +
    SLR_EXPANSION_PARAMS(2) * TL1 * (12.9635 - 1.0833 * PPL) -
    SLR_EXPANSION_PARAMS(3) * TL2 * (0.1713 - 0.019263 * PPL) +
    SLR_EXPANSION_PARAMS(4) * TL3 * (10.41 - 1.1338 * PPL) +
    SLR_EXPANSION_PARAMS(5) * PPL -
    SLR_EXPANSION_PARAMS(6) * PPL^2
```

Where:

- `TL1` = mean layer temperature (initial profile + delta)
- `TL2` = TL1^2
- `TL3` = TL1^3 / 6000
- `PPL` = pressure at layer from `OCN_PRESSURE_PROFILE`

**Layer expansion**:

```
DLR = (SLR_EXPANSION_COEFE * DELTOC * ZLAYER) / 1000
```

**Total expansion** (area-weighted sum across layers):

```
EXPAN = SUM(DLR * OCN_AREAFACTOR_AVERAGE) / OCN_LAYERBOUND_AREAS(1)
```

**Cumulative contribution**:

```
SLR_EXPANSION(t+1) = SLR_EXPANSION(t) + EXPAN * SLR_EXPANSION_SCALING
```

### 3.2 Glaciers and Ice Caps

Based on Wigley & Raper (2005) modified equation 5, using equilibrium relationships from Marzeion data.

**Equilibrium interpolation**:

1. Look up `SLR_GL_EQUISLRCURRENTTEMP` from temperature using `SLR_GL_EQUITEMP[]` / `SLR_GL_EQUISLR[]` tables
2. Look up `SLR_GL_EQUITEMPCURRENTSLR` from current SLR contribution using same tables

**Rate equation**:

```
SLR_GL(t+1) = SLR_GL(t) + SLR_GL_SENS_MMPYRDEG *
    ((SLR_GL_EQUISLRCURRENTTEMP - SLR_GL(t)) / SLR_GL_NORMPARA_VOL) *
    (SIGN(T_current - SLR_GL_EQUITEMPCURRENTSLR, SLR_GL_EQUITEMPCURRENTSLR) / SLR_GL_NORMPARA_TEMP)^SLR_GL_TEMP_EXPONENT
```

Note: The SIGN function is applied first, then divided by NORMPARA_TEMP, then the exponent is applied to the entire result. The Fortran SIGN(A,B) returns |A| with the sign of B.

### 3.3 Greenland Ice Sheet - Surface Mass Balance (GIS SMB)

Two parameterizations available:

**DEFAULT parameterization**:

```
SLR_GIS_SMB(t+1) = SLR_GIS_SMB(t) +
    SLR_GIS_SMB_COEF1 * (
        SLR_GIS_SMB_COEF2 * T_global +
        (1 - SLR_GIS_SMB_COEF2) * MAX(0, T_global)^SLR_GIS_SMB_SENS_EXPONENT
    ) * MAX(0, 1 - SLR_GIS_SMB(t) / SLR_GIS_SMB_INITIAL_VOLUME_MM)^SLR_GIS_SMB_VOLUME_EXPONENT
```

**FETTWEIS parameterization**:

```
SLR_GIS_SMB(t+1) = SLR_GIS_SMB(t) +
    (SLR_GIS_SMB_COEF_FW1 * T - SLR_GIS_SMB_COEF_FW2 * T^2 - SLR_GIS_SMB_COEF_FW3 * T^3) / (-361)
```

### 3.4 Greenland Ice Sheet - Solid Ice Discharge (GIS SID)

Based on Nick et al. (2013), with LOW/HIGH bounding cases.

**Annual discharge rate**:

```
GIS_SID_DISCHARGE = MIN(0, -DSCHRG_SENS * DSCHRGVOL * EXP(TEMPSENS_EXPONENT * T_global))
GIS_SID_DISCHARGE = MAX(GIS_SID_DISCHARGE, -DSCHRGVOL)  // Cannot exceed remaining volume
```

**Volume depletion**:

```
DSCHRGVOL(t+1) = MAX(DSCHRGVOL(t) + GIS_SID_DISCHARGE, 0)
```

**SLR contribution** (interpolated between LOW/HIGH):

```
SLR_GIS_SID = ((HIGH - LOW) * SLR_GIS_SID_CASE + LOW) * SLR_GIS_SID_SCALING
```

### 3.5 Antarctic Ice Sheet - Surface Mass Balance (AIS SMB)

Simpler than GIS, representing snowfall increase with warming:

```
SLR_AIS_SMB(t+1) = SLR_AIS_SMB(t) +
    SLR_AIS_SMB_COEF1 * (
        SLR_AIS_SMB_COEF2 * T_global +
        (1 - SLR_AIS_SMB_COEF2) * MAX(0, T_global)^SLR_AIS_SMB_SENS_EXPONENT
    )
```

Note: AIS SMB is typically **negative** (mass gain from increased precipitation), offsetting other contributions.

### 3.6 Antarctic Ice Sheet - Solid Ice Discharge (AIS SID)

Two methods available:

#### DECONTO Method

Includes threshold-based "fast rate" for ice cliff instability:

```
IF T_global < SLR_AIS_SID_THRESHOLDTEMP:
    DISCHARGE = DSCHRG_SENS * DSCHRGVOL * SIGN(|T - ZEROTEMP|^TEMPSENS_EXPONENT, T - ZEROTEMP)
ELSE:
    DISCHARGE = DSCHRG_SENS * DSCHRGVOL * SIGN(|T - ZEROTEMP|^TEMPSENS_EXPONENT, T - ZEROTEMP)
             + SLR_AIS_SID_FASTRATE
```

#### LEVERMANN Method (Default)

Uses impulse response functions (IRF) for 4 Antarctic regions with time-delayed convolution:

**Ocean forcing**:

```
OCNFORCE(R,t) = TEMPSCALING(R) * (T(t) - T(start)) * BASALMELT
FORCING(R,t) = OCNFORCE(R,t) - OCNFORCE(R,start)
```

**Impulse response** (4th order polynomial, clamped to 0):

```
R(x) = MAX(0, IRF(1)*x^4 + IRF(2)*x^3 + IRF(3)*x^2 + IRF(4)*x + IRF(5))
```

**Convolution**:

```
CONV_TOTAL(R,t) = FORCING(R,t) * R(0) +
    SUM(FORCING(R,t-i) * R(i) for i=1..t-2) +
    FORCING(R,start) * R(t)
```

**Total AIS SID**:

```
SLR_AIS_SID(t+1) = SUM(CONV_TOTAL(R,t) * 1000) for R in [Amundsen, EastAntarctica, Ross, Weddell]
```

### 3.7 Land Water Storage

Uses prescribed time series with volume depletion after switch year:

```
IF year > SWITCHYEAR:
    SLR_LANDWATER(t+1) = SLR_LANDWATER(t) + MMPYEAR(t) *
        MAX(0, 1 - (SLR_LANDWATER(t) - SWITCHVOL) / (MAXVOLUME - SWITCHVOL))^VOLUME_EXPONENT
ELSE:
    SLR_LANDWATER(t+1) = SLR_LANDWATER(t) + MMPYEAR(t)
```

### 3.8 Semi-Empirical Method (Rahmstorf)

Alternative simple approach:

```
BASETEMP = MEAN(T over TEMPBASEPERIOD)
RATE = RATE_SENS * (T - BASETEMP - ZERORATETEMP)
SLR_SEMIEMPI(t) = SLR_SEMIEMPI(t-1) + RATE(t)
```

### 3.9 Total Sea Level Rise

```
SLR_TOTAL = SLR_EXPANSION + SLR_GL + SLR_GIS_SMB + SLR_GIS_SID + SLR_AIS_SMB + SLR_AIS_SID + SLR_LANDWATER
```

## 4. State Variables

| Variable | Fortran Name | Symbol | Units | Description | Initial Value |
|----------|-------------|--------|-------|-------------|---------------|
| Total SLR contribution | `SLR_TOTAL_CONTRIBUTION` | - | mm | Cumulative total sea level rise | 0.0 |
| Thermal expansion | `SLR_EXPANSION_CONTRIBUTION` | - | mm | Cumulative thermal expansion | 0.0 |
| Glacier contribution | `SLR_GL_CONTRIBUTION` | - | mm | Cumulative glacier/ice cap melt | 0.0 |
| GIS SMB contribution | `SLR_GIS_SMB_CONTRIBUTION` | - | mm | Cumulative Greenland surface mass balance | 0.0 |
| GIS SID contribution | `SLR_GIS_SID_CONTRIBUTION` | - | mm | Cumulative Greenland ice discharge | 0.0 |
| GIS SID contribution LOW | `SLR_GIS_SID_CONTRIBUTION_LOW` | - | mm | Low estimate bound | 0.0 |
| GIS SID contribution HIGH | `SLR_GIS_SID_CONTRIBUTION_HIGH` | - | mm | High estimate bound | 0.0 |
| GIS discharge volume LOW | `SLR_GIS_SID_DSCHRGVOL_LOW` | - | mm | Remaining discharge volume (low) | `SLR_GIS_SID_TOTALVOL_LOW` |
| GIS discharge volume HIGH | `SLR_GIS_SID_DSCHRGVOL_HIGH` | - | mm | Remaining discharge volume (high) | `SLR_GIS_SID_TOTALVOL_HIGH` |
| AIS SMB contribution | `SLR_AIS_SMB_CONTRIBUTION` | - | mm | Cumulative Antarctic surface mass balance | 0.0 |
| AIS SID contribution | `SLR_AIS_SID_CONTRIBUTION` | - | mm | Cumulative Antarctic ice discharge | 0.0 |
| AIS discharge volume | `SLR_AIS_SID_DSCHRGVOL` | - | mm | Remaining AIS discharge volume | `SLR_AIS_SID_TOTALVOL` |
| Land water contribution | `SLR_LANDWATER_CONTRIBUTION` | - | mm | Cumulative land water storage change | 0.0 |
| AIS region response (Amundsen) | `SLR_AIS_SID_R_AMUNDSEN` | - | m/K | IRF response array | Computed from polynomial |
| AIS region response (East Antarctica) | `SLR_AIS_SID_R_EASTANTARCTICA` | - | m/K | IRF response array | Computed from polynomial |
| AIS region response (Ross) | `SLR_AIS_SID_R_ROSS` | - | m/K | IRF response array | Computed from polynomial |
| AIS region response (Weddell) | `SLR_AIS_SID_R_WEDDELL` | - | m/K | IRF response array | Computed from polynomial |
| AIS ocean forcing | `SLR_AIS_SID_OCNFORCE` | - | - | 2D array [4, NYEARS] | 0.0 |
| AIS forcing | `SLR_AIS_SID_FORCING` | - | - | 2D array [4, NYEARS] | 0.0 |
| AIS convolution arrays | `SLR_AIS_SID_CONV_*` | - | - | Multiple 2D arrays for convolution | 0.0 |
| Equilibrium SLR at current temp | `SLR_GL_EQUISLRCURRENTTEMP` | - | mm | Interpolated equilibrium glacier SLR | Computed |
| Equilibrium temp at current SLR | `SLR_GL_EQUITEMPCURRENTSLR` | - | deg C | Interpolated equilibrium temperature | Computed |

## 5. Parameters

### 5.1 Thermal Expansion Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|-------------|-------|---------|-------------|-------------|
| Expansion coefficients | `SLR_EXPANSION_PARAMS` | various | [52.24, 0.99, 1.03, 0.61, 24.26, 0.74] | - | 6-element polynomial coefficients |
| Expansion scaling | `SLR_EXPANSION_SCALING` | - | 0.8824 | 0-2 | Overall scaling factor |
| Expansion start year | `SLR_EXPANSION_STARTYEAR` | year | 1850 | - | Year to begin calculation |
| Expansion params file | `FILE_SLR_THEXP_PARAMS` | - | SLR_THEXP_PARAMS_CMIP5MEAN.CFG | - | Parameter file name |

### 5.2 Glacier Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|-------------|-------|---------|-------------|-------------|
| Sensitivity | `SLR_GL_SENS_MMPYRDEG` | mm/yr/K | 0.625 | 0-5 | Rate sensitivity |
| Temperature exponent | `SLR_GL_TEMP_EXPONENT` | - | 0.82 | 0-2 | Power law exponent |
| Normalization volume | `SLR_GL_NORMPARA_VOL` | - | 1.0 | - | Volume normalization |
| Normalization temp | `SLR_GL_NORMPARA_TEMP` | - | 1.0 | - | Temperature normalization |
| Start year | `SLR_GL_STARTYEAR` | year | 1850 | - | Year to begin calculation |
| Equilibrium temperature table | `SLR_GL_EQUITEMP` | deg C | [0.0, 0.1, ..., 10.3] | - | 104-element lookup (0-10.3 C) |
| Equilibrium SLR table | `SLR_GL_EQUISLR` | mm | [81.2, 96.6, ..., 410.2] | - | 104-element lookup |
| Glacier params file | `FILE_SLR_GL_PARAMS` | - | MAGTUNE_GLACIERTUNE_CMIP5MEAN.CFG | - | Tuning file |
| Glacier extra params file | `FILE_SLR_GL_XTRAPARAMS` | - | SLR_GL_PARAMS_CMIP5MEAN.CFG | - | Equilibrium tables file |

### 5.3 Greenland SMB Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|-------------|-------|---------|-------------|-------------|
| Parameterization | `SLR_GIS_SMB_PARAMETERISATION` | - | "DEFAULT" | DEFAULT/FETTWEIS | Method selection |
| Coefficient 1 | `SLR_GIS_SMB_COEF1` | mm/yr | 0.015 | - | Primary sensitivity |
| Coefficient 2 | `SLR_GIS_SMB_COEF2` | - | 0.9 | 0-1 | Linear/nonlinear weighting |
| Sensitivity exponent | `SLR_GIS_SMB_SENS_EXPONENT` | - | 2.3 | 1-4 | Temperature power |
| Initial volume | `SLR_GIS_SMB_INITIAL_VOLUME_MM` | mm | 7360.0 | - | Total ice equivalent |
| Volume exponent | `SLR_GIS_SMB_VOLUME_EXPONENT` | - | 0.5 | 0-1 | Depletion scaling |
| Fettweis coef 1 | `SLR_GIS_SMB_COEF_FW1` | - | -10.0 | - | Linear term |
| Fettweis coef 2 | `SLR_GIS_SMB_COEF_FW2` | - | 2.0 | - | Quadratic term |
| Fettweis coef 3 | `SLR_GIS_SMB_COEF_FW3` | - | 1.0 | - | Cubic term |
| Start year | `SLR_GIS_SMB_STARTYEAR` | year | 1965 | - | Year to begin calculation |
| SMB params file | `FILE_SLR_GISSMB_PARAMS` | - | MAGTUNE_GISSMBTUNE_CMIP5MEAN.CFG | - | Parameter file |

### 5.4 Greenland SID Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|-------------|-------|---------|-------------|-------------|
| Case selector | `SLR_GIS_SID_CASE` | - | 0.5 | 0-1 | LOW(0) to HIGH(1) interpolation |
| Scaling factor | `SLR_GIS_SID_SCALING` | - | 5.0 | 1-10 | IPCC AR5 upscaling factor |
| Total volume LOW | `SLR_GIS_SID_TOTALVOL_LOW` | mm | 35.98 | - | Low estimate reservoir |
| Total volume HIGH | `SLR_GIS_SID_TOTALVOL_HIGH` | mm | 53.63 | - | High estimate reservoir |
| Discharge sens LOW | `SLR_GIS_SID_DSCHRG_SENS_LOW` | 1/yr | 0.000906 | - | Low discharge rate |
| Discharge sens HIGH | `SLR_GIS_SID_DSCHRG_SENS_HIGH` | 1/yr | 0.000793 | - | High discharge rate |
| Temp sens exp LOW | `SLR_GIS_SID_TEMPSENS_EXPONENT_LOW` | 1/K | 0.389 | - | Temperature sensitivity |
| Temp sens exp HIGH | `SLR_GIS_SID_TEMPSENS_EXPONENT_HIGH` | 1/K | 0.472 | - | Temperature sensitivity |
| Start year | `SLR_GIS_SID_STARTYEAR` | year | 2000 | - | Year to begin calculation |

### 5.5 Antarctic SMB Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|-------------|-------|---------|-------------|-------------|
| Coefficient 1 | `SLR_AIS_SMB_COEF1` | mm/yr | 0.128 | - | Primary sensitivity |
| Coefficient 2 | `SLR_AIS_SMB_COEF2` | - | -0.424 | - | Linear/nonlinear weighting |
| Sensitivity exponent | `SLR_AIS_SMB_SENS_EXPONENT` | - | 0.782 | 0-2 | Temperature power |
| Start year | `SLR_AIS_SMB_STARTYEAR` | year | 1980 | - | Year to begin calculation |
| SMB params file | `FILE_SLR_AISSMB_PARAMS` | - | MAGTUNE_AISSMBTUNE_MEAN.CFG | - | Parameter file |

### 5.6 Antarctic SID Parameters (Common)

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|-------------|-------|---------|-------------|-------------|
| Parameterization | `SLR_AIS_SID_PARAMETERISATION` | - | "LEVERMANN" | DECONTO/LEVERMANN | Method selection |
| Scaling factor | `SLR_AIS_SID_SCALING` | - | 1.0 | 0-2 | Overall scaling |
| Start year | `SLR_AIS_SID_STARTYEAR` | year | 1850 | - | Year to begin calculation |
| Discharge start year | `SLR_AIS_DISCHARGE_STARTYEAR` | year | 1950 | - | Volume depletion start |
| SID params file | `FILE_SLR_AISSID_PARAMS` | - | MAGTUNE_AISSIDTUNE_MIROCESM.CFG | - | Tuning file |

### 5.7 Antarctic SID Parameters (DeConto Method)

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|-------------|-------|---------|-------------|-------------|
| Total volume | `SLR_AIS_SID_TOTALVOL` | mm | 17560.0 | - | Ice reservoir |
| Discharge sensitivity | `SLR_AIS_SID_DSCHRG_SENS` | 1/yr | 5.28e-5 | - | Rate sensitivity |
| Temp sens exponent | `SLR_AIS_SID_TEMPSENS_EXPONENT` | - | 2.0 | 1-3 | Temperature power |
| Threshold temp | `SLR_AIS_SID_THRESHOLDTEMP` | deg C | 1.023 | - | Fast process trigger |
| Zero temp | `SLR_AIS_SID_ZEROTEMP` | deg C | 0.0 | - | Reference temperature |
| Fast rate | `SLR_AIS_SID_FASTRATE` | mm/yr | 13.83 | - | Ice cliff instability rate |

### 5.8 Antarctic SID Parameters (Levermann Method)

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|-------------|-------|---------|-------------|-------------|
| Basal melt | `SLR_AIS_SID_BASALMELT` | m/yr/K | 11.5 | 7-16 | Melt sensitivity |
| IRF year span | `SLR_AIS_SID_IRF_YRSPAN` | years | 500 | 100-1000 | Response function duration |
| Time delay - Amundsen | `SLR_AIS_SID_DT_AMUNDSEN` | years | 0 | 0-100 | Regional delay |
| Time delay - East Ant. | `SLR_AIS_SID_DT_EASTANTARCTICA` | years | 30 | 0-100 | Regional delay |
| Time delay - Ross | `SLR_AIS_SID_DT_ROSS` | years | 20 | 0-100 | Regional delay |
| Time delay - Weddell | `SLR_AIS_SID_DT_WEDDELL` | years | 35 | 0-100 | Regional delay |
| Temp scaling - Amundsen | `SLR_AIS_SID_TEMPSCALING_AMUNDSEN` | - | 0.17 | 0-1 | Regional sensitivity |
| Temp scaling - East Ant. | `SLR_AIS_SID_TEMPSCALING_EASTANTARCTICA` | - | 0.35 | 0-1 | Regional sensitivity |
| Temp scaling - Ross | `SLR_AIS_SID_TEMPSCALING_ROSS` | - | 0.26 | 0-1 | Regional sensitivity |
| Temp scaling - Weddell | `SLR_AIS_SID_TEMPSCALING_WEDDELL` | - | 0.14 | 0-1 | Regional sensitivity |
| IRF polynomial - Amundsen | `SLR_AIS_SID_IRF_AMUNDSEN` | - | [3.8e-15, -1.2e-11, 5.3e-9, -1.1e-7, 2.7e-5] | - | 5 coefficients |
| IRF polynomial - East Ant. | `SLR_AIS_SID_IRF_EASTANTARCTICA` | - | [-4.8e-15, 4.7e-12, -1.3e-9, 1.6e-7, 1.1e-5] | - | 5 coefficients |
| IRF polynomial - Ross | `SLR_AIS_SID_IRF_ROSS` | - | [-6.1e-14, 5.2e-11, -1.4e-8, 1.8e-6, -2.2e-5] | - | 5 coefficients |
| IRF polynomial - Weddell | `SLR_AIS_SID_IRF_WEDDELL` | - | [1.5e-14, -1.5e-11, 5.2e-9, -5.1e-7, 3.5e-5] | - | 5 coefficients |

### 5.9 Land Water Storage Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|-------------|-------|---------|-------------|-------------|
| Switch on/off | `SLR_LANDWATER_SWITCH` | - | 0 | 0/1 | Enable land water component |
| Case selector | `SLR_LANDWATER_CASE` | - | 1.0 | 0-1 | LOW(0) to HIGH(1) interpolation |
| Start year | `SLR_LANDWATER_STARTYEAR` | year | 1900 | - | Year to begin |
| Switch year | `SLR_LANDWATER_SWITCHYEAR` | year | 2100 | - | Transition to depletion mode |
| Max volume | `SLR_LANDWATER_MAXVOLUME_MM` | mm | 1000.0 | - | Total groundwater reservoir |
| Volume exponent | `SLR_LANDWATER_VOLUME_EXPONENT` | - | 0.5 | 0-1 | Depletion scaling |
| High scenario file | `FILE_SLR_LANDWATER_HIGH` | - | SLR_LANDWATER_HIGH.IN | - | High time series |
| Low scenario file | `FILE_SLR_LANDWATER_LOW` | - | SLR_LANDWATER_LOW.IN | - | Low time series |

### 5.10 Semi-Empirical Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|-------------|-------|---------|-------------|-------------|
| Zero rate temp | `SLR_SEMIEMPI_ZERORATETEMP` | deg C | -0.5 | - | Temperature offset |
| Rate sensitivity | `SLR_SEMIEMPI_RATE_SENS` | mm/yr/K | 0.3353 | - | Rahmstorf coefficient |
| Base period | `SLR_SEMIEMPI_TEMPBASEPERIOD` | years | [1980, 1999] | - | Reference period |
| Switch year | `SLR_SEMIEMPI_SWITCHFROMOBS2CALC` | year | 2000 | - | Obs to calc transition |

### 5.11 Output/Control Parameters

| Parameter | Fortran Name | Units | Default | Valid Range | Description |
|-----------|-------------|-------|---------|-------------|-------------|
| Switch year obs/calc | `SLR_SWITCH_CALC2HIST` | year | -10000 | - | When to use calculated vs historic |
| Historic file | `FILE_SLR_HISTORIC` | - | HIST_SEALEVEL_CHURCHWHITE2006_RF.IN | - | Observed SLR data |
| Zero reference period | `OUT_ZERO_SLR_PERIOD` | years | [1986, 2005] | - | Anomaly reference period |

## 6. Inputs (per timestep)

| Variable | Units | Source Module | Required? | Fortran Variable |
|----------|-------|---------------|-----------|------------------|
| Global mean surface temperature | deg C | Climate (datastore) | Yes | `DAT_SURFACE_ANNUALMEANTEMP % DATGLOBE` |
| Ocean layer temperatures | deg C | Climate | Yes (thermal expansion) | `TEMP_OCEANLAYERS(I, CURRENT_YEAR_IDX)` |
| Ocean layer temperatures (next year) | deg C | Climate | Yes (thermal expansion) | `TEMP_OCEANLAYERS(I, NEXT_YEAR_IDX)` |
| Ocean pressure profile | dbar | Climate (init) | Yes (thermal expansion) | `OCN_PRESSURE_PROFILE(I)` |
| Ocean area factors | - | Climate (init) | Yes (thermal expansion) | `OCN_AREAFACTOR_AVERAGE(1:2, I)` |
| Ocean layer bound areas | - | Climate (init) | Yes (thermal expansion) | `OCN_LAYERBOUND_AREAS(1:2, 1)` |
| Initial ocean temperature profile | deg C | Climate (init) | Yes (thermal expansion) | `TEMP_INITIAL_OCEAN_*_PROFILE` |
| Mixed layer depth | m | Climate (param) | Yes (thermal expansion) | `CORE_MIXEDLAYER_DEPTH` |
| Ocean number of levels | - | Climate (param) | Yes (thermal expansion) | `CORE_OCN_NLEVELS` |
| Land water time series | mm/yr | File input | If enabled | `SLR_LANDWATER_MMPYEAR` |
| Current year index | - | Time module | Yes | `CURRENT_YEAR_IDX` |
| Next year index | - | Time module | Yes | `NEXT_YEAR_IDX` |
| Start year | year | Time module | Yes | `STARTYEAR` |
| All years array | year | Time module | Yes | `ALLYEARS` |

## 7. Outputs (per timestep)

| Variable | Units | Destination Module(s) | Fortran Variable |
|----------|-------|----------------------|------------------|
| Total SLR | mm | Output/datastore | `SLR_TOTAL_CONTRIBUTION` |
| Thermal expansion SLR | mm | Output/datastore | `SLR_EXPANSION_CONTRIBUTION` |
| Glacier SLR | mm | Output/datastore | `SLR_GL_CONTRIBUTION` |
| GIS SMB SLR | mm | Output/datastore | `SLR_GIS_SMB_CONTRIBUTION` |
| GIS SID SLR | mm | Output/datastore | `SLR_GIS_SID_CONTRIBUTION` |
| AIS SMB SLR | mm | Output/datastore | `SLR_AIS_SMB_CONTRIBUTION` |
| AIS SID SLR | mm | Output/datastore | `SLR_AIS_SID_CONTRIBUTION` |
| Land water SLR | mm | Output/datastore | `SLR_LANDWATER_CONTRIBUTION` |
| Total SLR (datastore) | mm | Output file | `DAT_SLR_TOT % DATGLOBE` |
| Expansion (datastore) | mm | Output file | `DAT_SLR_EXPANSION % DATGLOBE` |
| Glacier (datastore) | mm | Output file | `DAT_SLR_GL % DATGLOBE` |
| GIS SMB (datastore) | mm | Output file | `DAT_SLR_GIS_SMB % DATGLOBE` |
| GIS SID (datastore) | mm | Output file | `DAT_SLR_GIS_SID % DATGLOBE` |
| AIS SMB (datastore) | mm | Output file | `DAT_SLR_AIS_SMB % DATGLOBE` |
| AIS SID (datastore) | mm | Output file | `DAT_SLR_AIS_SID % DATGLOBE` |
| Land water (datastore) | mm | Output file | `DAT_SLR_LANDWATER % DATGLOBE` |
| Semi-empirical SLR | mm | Output file | `DAT_SLR_SEMIEMPI_TOT % DATGLOBE` |
| Semi-empirical rate | mm/yr | Output file | `DAT_SLR_SEMIEMPI_RATE % DATGLOBE` |

## 8. Algorithm (Pseudocode)

### 8.1 Initialization (`sealevel_init`)

```
SUBROUTINE sealevel_init:
    # Initialize all contribution arrays to zero
    FOR each contribution array:
        SET array = 0.0

    # Build impulse response functions for AIS regions
    FOR i = 1 to NYEARS:
        x = i - 1
        IF i <= SLR_AIS_SID_IRF_YRSPAN:
            SLR_AIS_SID_R_AMUNDSEN(i) = MAX(0, polynomial(IRF_AMUNDSEN, x))
            SLR_AIS_SID_R_EASTANTARCTICA(i) = MAX(0, polynomial(IRF_EASTANTARCTICA, x))
            SLR_AIS_SID_R_ROSS(i) = MAX(0, polynomial(IRF_ROSS, x))
            SLR_AIS_SID_R_WEDDELL(i) = MAX(0, polynomial(IRF_WEDDELL, x))
        ELSE:
            SET all regional R arrays to 0.0

    # Read historic sea level data
    READ FILE_SLR_HISTORIC into DAT_SLR_TOT, DAT_SLR_SEMIEMPI_TOT

    # Initialize empty datastore structures
    FOR each SLR component:
        CALL datastore_read('', DAT_SLR_*, 0)

    # Setup land water time series
    READ FILE_SLR_LANDWATER_HIGH, FILE_SLR_LANDWATER_LOW

    # Interpolate land water based on CASE parameter
    IF SLR_LANDWATER_CASE == 0:
        SLR_LANDWATER_MMPYEAR = DAT_LANDWATER_LOW
    ELSE IF SLR_LANDWATER_CASE == 1:
        SLR_LANDWATER_MMPYEAR = DAT_LANDWATER_HIGH
    ELSE:
        SLR_LANDWATER_MMPYEAR = LINEAR_INTERP(LOW, HIGH, CASE)
```

### 8.2 Main Calculation (`sealevel_calc`)

```
SUBROUTINE sealevel_calc:

    # ========== THERMAL EXPANSION ==========
    IF year > SLR_EXPANSION_STARTYEAR:
        EXPAN = 0.0
        ZLAYER = CORE_MIXEDLAYER_DEPTH

        FOR L = 1 to CORE_OCN_NLEVELS:
            # Temperature change in layer
            DELTOC = SLR_EXPANSION_SCALING * (T_ocean(L, next) - T_ocean(L, current))
            DELTBAR = SLR_EXPANSION_SCALING * (T_ocean(L, next) + T_ocean(L, current)) / 2

            # Get initial temperature for layer
            IF CORE_SWITCH_OCN_TEMPPROFILE == 1:
                TL1 = AVG(TEMP_INITIAL_OCEAN_EXP_PROFILE(:, L)) + DELTBAR
            ELSE IF CORE_SWITCH_OCN_TEMPPROFILE == 2:
                TL1 = AVG(TEMP_INITIAL_OCEAN_CMIP5_PROFILE(:, L)) + DELTBAR
            ELSE:
                TL1 = TEMP_OCEAN_INI_PROFILE_ONE(L) + DELTBAR

            # Calculate expansion coefficient (empirical polynomial)
            TL2 = TL1^2
            TL3 = TL1^3 / 6000
            PPL = OCN_PRESSURE_PROFILE(L)

            COEFE = PARAMS(1) + PARAMS(2)*TL1*(12.9635 - 1.0833*PPL)
                  - PARAMS(3)*TL2*(0.1713 - 0.019263*PPL)
                  + PARAMS(4)*TL3*(10.41 - 1.1338*PPL)
                  + PARAMS(5)*PPL - PARAMS(6)*PPL^2

            # Layer expansion
            DLR = (COEFE * DELTOC * ZLAYER) / 1000
            ZLAYER = DZ  # Update for layers 2+

            # Weight by layer area
            EXPAN = EXPAN + DLR * AVG(OCN_AREAFACTOR_AVERAGE(:, L))

        # Normalize by surface area
        EXPAN = EXPAN / AVG(OCN_LAYERBOUND_AREAS(:, 1))

        SLR_EXPANSION(next) = EXPAN + SLR_EXPANSION(current)

    # ========== GLACIERS ==========
    IF year > SLR_GL_STARTYEAR:
        # Find equilibrium SLR for current temperature
        SLR_GL_EQUISLRCURRENTTEMP = INTERP(T_global, SLR_GL_EQUITEMP, SLR_GL_EQUISLR)

        # Find equilibrium temperature for current SLR
        SLR_GL_EQUITEMPCURRENTSLR = INTERP(SLR_GL(current), SLR_GL_EQUISLR, SLR_GL_EQUITEMP)

        # Apply modified Wigley-Raper equation
        volume_factor = (SLR_GL_EQUISLRCURRENTTEMP - SLR_GL(current)) / SLR_GL_NORMPARA_VOL
        temp_factor = SIGN(|T_global - SLR_GL_EQUITEMPCURRENTSLR|^SLR_GL_TEMP_EXPONENT,
                          SLR_GL_EQUITEMPCURRENTSLR)
        temp_factor = temp_factor / SLR_GL_NORMPARA_TEMP

        SLR_GL(next) = SLR_GL(current) + SLR_GL_SENS_MMPYRDEG * volume_factor * temp_factor

    # ========== GREENLAND SMB ==========
    IF year > SLR_GIS_SMB_STARTYEAR:
        IF SLR_GIS_SMB_PARAMETERISATION == 'DEFAULT':
            T_term = COEF2 * T_global + (1 - COEF2) * MAX(0, T_global)^SENS_EXPONENT
            volume_term = MAX(0, 1 - SLR_GIS_SMB(current) / INITIAL_VOLUME)^VOLUME_EXPONENT
            SLR_GIS_SMB(next) = SLR_GIS_SMB(current) + COEF1 * T_term * volume_term
        ELSE:  # FETTWEIS
            SLR_GIS_SMB(next) = SLR_GIS_SMB(current) +
                (FW1*T - FW2*T^2 - FW3*T^3) / (-361)

    # ========== GREENLAND SID ==========
    IF year > SLR_GIS_SID_STARTYEAR AND T_global != 0:
        # Initialize discharge volumes at start year
        IF year == SLR_GIS_SID_STARTYEAR:
            DSCHRGVOL_LOW(start) = TOTALVOL_LOW
            DSCHRGVOL_HIGH(start) = TOTALVOL_HIGH

        # Calculate discharge for LOW and HIGH cases
        FOR case IN [LOW, HIGH]:
            DISCHARGE = MIN(0, -DSCHRG_SENS * DSCHRGVOL(current) * EXP(TEMPSENS_EXP * T_global))
            DISCHARGE = MAX(DISCHARGE, -DSCHRGVOL(current))  # Cannot exceed volume

            IF year < SLR_GIS_SID_STARTYEAR:
                # Before start year: accumulate but don't deplete
                DSCHRGVOL(next) = TOTALVOL
                CONTRIBUTION(next) = CONTRIBUTION(current) - DISCHARGE
            ELSE:
                # After start year: deplete volume
                DSCHRGVOL(next) = MAX(DISCHARGE + DSCHRGVOL(current), 0)
                CONTRIBUTION(next) = TOTALVOL - DSCHRGVOL(next) + CONTRIBUTION(startyear)

        # Interpolate between LOW and HIGH based on CASE
        SLR_GIS_SID(next) = INTERP(CASE, LOW, HIGH) * SLR_GIS_SID_SCALING

    # ========== ANTARCTIC SMB ==========
    IF year > SLR_AIS_SMB_STARTYEAR:
        T_term = COEF2 * T_global + (1 - COEF2) * MAX(0, T_global)^SENS_EXPONENT
        SLR_AIS_SMB(next) = SLR_AIS_SMB(current) + COEF1 * T_term
        # Note: typically negative (mass gain from increased precipitation)

    # ========== ANTARCTIC SID ==========
    IF SLR_AIS_SID_PARAMETERISATION == 'DECONTO':
        IF year > SLR_AIS_SID_STARTYEAR AND T_global != 0:
            # Initialize at start
            IF year == SLR_AIS_SID_STARTYEAR:
                DSCHRGVOL(start) = TOTALVOL

            # Calculate discharge with optional fast rate
            temp_term = SIGN(|T_global - ZEROTEMP|^TEMPSENS_EXPONENT, T_global - ZEROTEMP)
            DISCHARGE = DSCHRG_SENS * DSCHRGVOL(current) * temp_term

            IF T_global >= THRESHOLDTEMP:
                DISCHARGE = DISCHARGE + FASTRATE  # Ice cliff instability

            DISCHARGE = MIN(DISCHARGE, DSCHRGVOL(current))

            # Update volume and contribution
            IF year < SLR_AIS_DISCHARGE_STARTYEAR:
                DSCHRGVOL(next) = TOTALVOL
            ELSE:
                DSCHRGVOL(next) = DSCHRGVOL(current) - DISCHARGE

            SLR_AIS_SID(next) = (TOTALVOL - DSCHRGVOL(next) + ...) * SCALING

            # Subtract SMB to get pure SID
            SLR_AIS_SID(next) = SLR_AIS_SID(next) - SLR_AIS_SMB(next)

    ELSE:  # LEVERMANN method
        IF year > SLR_AIS_SID_STARTYEAR:
            AIS_SID_IDX = current_year_idx - (SLR_AIS_SID_STARTYEAR - STARTYEAR)

            FOR R = 1 to 4:  # Each Antarctic region
                # Get region-specific parameters
                SELECT region parameters based on R

                IF AIS_SID_IDX >= DT_CURRENTREGION:
                    # Calculate melt forcing
                    OCNFORCE(R, current) = TEMPSCALING * (T(current) - T(start)) * BASALMELT
                    FORCING(R, current) = OCNFORCE(R, current) - OCNFORCE(R, start)

                    # Convolution with impulse response
                    CONV_CORNER_F = FORCING(current) * R(0)
                    IF idx > 1: CONV_CORNER_R = FORCING(start) * R(idx)
                    IF idx > 2:
                        FOR i = 1 to idx-2:
                            CONV_MIDDLE(i) = FORCING(current - i) * R(i)
                        CONV_MIDDLE_CUM = SUM(CONV_MIDDLE)

                    CONV_STEPTOTAL = CONV_CORNER_F + CONV_MIDDLE_CUM + CONV_CORNER_R
                    CONV_TOTAL(R, next) = CONV_STEPTOTAL * 1000  # m to mm

            # Sum all regions
            SLR_AIS_SID(next) = SUM(CONV_TOTAL(:, next))

    # ========== LAND WATER STORAGE ==========
    IF SLR_LANDWATER_SWITCH == 0:
        SLR_LANDWATER(next) = 0
    ELSE IF year > SLR_LANDWATER_STARTYEAR:
        IF year > SLR_LANDWATER_SWITCHYEAR:
            SWITCHVOL = SLR_LANDWATER(switchyear)
            depletion_factor = MAX(0, 1 - (SLR_LANDWATER(current) - SWITCHVOL) /
                                       (MAXVOLUME - SWITCHVOL))^VOLUME_EXPONENT
            SLR_LANDWATER(next) = SLR_LANDWATER(current) + MMPYEAR(current) * depletion_factor
        ELSE:
            SLR_LANDWATER(next) = SLR_LANDWATER(current) + MMPYEAR(current)

    # ========== TOTAL ==========
    SLR_TOTAL(next) = SLR_EXPANSION(next) + SLR_GL(next) + SLR_GIS_SMB(next) +
                      SLR_GIS_SID(next) + SLR_AIS_SMB(next) + SLR_AIS_SID(next) +
                      SLR_LANDWATER(next)

    # Update datastore structures after switch year
    IF year > SLR_SWITCH_CALC2HIST:
        FOR each component:
            DAT_SLR_*(next) = DAT_SLR_*(current) + (SLR_*(next) - SLR_*(current))
```

### 8.3 Semi-Empirical Calculation (`sealevel_calc_semiempi`)

```
SUBROUTINE sealevel_calc_semiempi:
    # Ensure base period is valid
    IF TEMPBASEPERIOD(1) > TEMPBASEPERIOD(2):
        SWAP(TEMPBASEPERIOD(1), TEMPBASEPERIOD(2))

    CHECK_WITHIN_RUNYEARS(TEMPBASEPERIOD(1))
    CHECK_WITHIN_RUNYEARS(TEMPBASEPERIOD(2))

    # Calculate base period mean temperature
    BASETEMP = MEAN(T_surface over TEMPBASEPERIOD)

    # Calculate rate of sea level rise (Rahmstorf regression)
    SLR_RATE = RATE_SENS * (T_surface - BASETEMP - ZERORATETEMP)

    # Handle switch from observations to calculated
    IF DAT_SLR_SEMIEMPI_TOT.LASTYEAR < SWITCHFROMOBS2CALC:
        SWITCHFROMOBS2CALC = DAT_SLR_SEMIEMPI_TOT.LASTYEAR

    IDX_SWITCH = SWITCHFROMOBS2CALC - STARTYEAR + 1

    # Integrate rate to get total SLR
    FOR i = 2 to NYEARS:
        IF i >= IDX_SWITCH:
            DAT_SLR_SEMIEMPI_TOT(i) = DAT_SLR_SEMIEMPI_TOT(i-1) + SLR_RATE(i)

    # Copy to ocean boxes (NH and SH ocean only)
    DAT_SLR_SEMIEMPI_TOT.DATBOX(:, OCEAN_NH) = DAT_SLR_SEMIEMPI_TOT.DATGLOBE
    DAT_SLR_SEMIEMPI_TOT.DATBOX(:, OCEAN_SH) = DAT_SLR_SEMIEMPI_TOT.DATGLOBE
    DAT_SLR_SEMIEMPI_TOT.DATBOX(:, LAND_NH) = 0
    DAT_SLR_SEMIEMPI_TOT.DATBOX(:, LAND_SH) = 0
```

## 9. Integration with Climate Module

### 9.1 Call Sequence

The sea level rise module is called at the end of each timestep after all climate calculations:

```fortran
! In magicc_step_year (MAGICC7.f90 line 3726):
CALL sealevel_calc
```

### 9.2 Temperature Input

The module receives temperature from the climate module via the datastore:

```fortran
DAT_SURFACE_ANNUALMEANTEMP % DATGLOBE(CURRENT_YEAR_IDX)
```

This is the global mean surface temperature anomaly used by all ice sheet and glacier components.

### 9.3 Ocean Temperature Input

For thermal expansion, the module accesses ocean layer temperatures directly from the climate module's state:

```fortran
TEMP_OCEANLAYERS(I, CURRENT_YEAR_IDX)  ! Temperature at current timestep
TEMP_OCEANLAYERS(I, NEXT_YEAR_IDX)     ! Temperature at next timestep
```

### 9.4 Ocean Structure

The module uses ocean structure parameters set during climate initialization:

- `CORE_OCN_NLEVELS` - Number of ocean layers
- `CORE_MIXEDLAYER_DEPTH` - Depth of mixed layer (m)
- `DZ` - Layer thickness (100 m)
- `OCN_PRESSURE_PROFILE(I)` - Pressure at each layer
- `OCN_AREAFACTOR_AVERAGE(1:2, I)` - Hemisphere area fractions
- `OCN_LAYERBOUND_AREAS(1:2, I)` - Layer boundary areas

### 9.5 Initial Ocean Temperature Profile

Three options for initial ocean temperature profile (controlled by `CORE_SWITCH_OCN_TEMPPROFILE`):

1. Exponential profile: `TEMP_INITIAL_OCEAN_EXP_PROFILE`
2. CMIP5 profile: `TEMP_INITIAL_OCEAN_CMIP5_PROFILE`
3. Simple profile: `TEMP_OCEAN_INI_PROFILE_ONE`

## 10. Honest Assessment / Red Flags

### 10.1 Hardcoded Values That Should Be Parameters

| Value | Location | Issue |
|-------|----------|-------|
| `-361.0` | Line 401 | Magic number in Fettweis GIS SMB conversion |
| `1000.0` | Line 657 | m to mm conversion |
| `6000.0` | Line 289 | Divisor in TL3 calculation |
| `12.9635`, `1.0833`, etc. | Lines 292-297 | Thermal expansion polynomial constants |
| `104` | Line 42 | Fixed size of equilibrium lookup tables |
| `4` | Throughout | Number of Antarctic regions |

### 10.2 Missing or Incomplete Implementations

1. **No ice sheet feedback on SMB**: Surface mass balance doesn't account for elevation feedback as ice sheets thin
2. **No marine ice sheet instability physics**: DeConto "fast rate" is a crude threshold approximation
3. **Glacier hypsometry ignored**: Glacier response doesn't account for elevation distribution
4. **No regional sea level**: All outputs are global mean (no fingerprinting, GIA, or regional patterns)
5. **Land water stops at switchyear**: Limited to historical extrapolation

### 10.3 Code Quality Issues

1. **Complex index management**: Multiple index variables (`CURRENT_YEAR_IDX`, `NEXT_YEAR_IDX`, `AIS_SID_IDX`, `AIS_SID_DT_STARTIDX`, `AIS_SID_DT_IDX`) make the code error-prone
2. **Mixed parameterization switching**: Runtime selection between DECONTO/LEVERMANN and DEFAULT/FETTWEIS adds complexity
3. **SAVE attribute on local**: `AIS_SID_DISCHARGE` has `SAVE` attribute (line 258) which persists between calls
4. **Large lookup tables**: 104-element arrays declared as fixed size
5. **No input validation**: No bounds checking on interpolation lookups

### 10.4 Numerical Concerns

1. **Convolution accumulation**: Levermann method accumulates convolution sums that could lose precision
2. **Temperature == 0 guard**: Explicit check for zero temperature (line 412, 510) suggests numerical edge cases
3. **SIGN function usage**: Non-standard use of `SIGN()` to handle negative temperature anomalies
4. **MAX(0, ...)** guards: Used extensively to prevent negative volumes, suggesting potential sign errors

### 10.5 Scientific Limitations

1. **Outdated ice sheet models**: Based on Nick (2013), Levermann (2014), DeConto (2016)
2. **No MICI (Marine Ice Cliff Instability)**: Key process for large Antarctic contributions
3. **Equilibrium glacier assumption**: Uses static temperature-SLR relationship
4. **Semi-empirical alternative**: Rahmstorf method is simple regression, not process-based
5. **No uncertainty propagation**: Only LOW/HIGH bounds for GIS SID, not proper ensemble

### 10.6 Configuration Complexity

- ~80+ configurable parameters across all components
- Multiple tuning files required
- Interdependencies between parameters not documented
- Many parameters have no documented valid ranges

## 11. Fortran Code References

### Key Subroutines

| Subroutine | Lines | Description |
|------------|-------|-------------|
| `sealevel_alloc` | 73-95 | Array allocation |
| `sealevel_dealloc` | 97-114 | Array deallocation |
| `sealevel_init` | 116-237 | Initialization and IRF setup |
| `sealevel_calc` | 239-753 | Main calculation routine |
| `sealevel_calc_semiempi` | 755-830 | Semi-empirical method |

### Component Calculations

| Component | Start Line | End Line | Key Equation(s) |
|-----------|------------|----------|-----------------|
| Thermal Expansion | 266 | 312 | 292-297 (coefficient), 299 (DLR), 303-308 (summation) |
| Glaciers | 319 | 373 | 364-371 (Wigley-Raper) |
| GIS SMB (Default) | 383 | 391 | 386-391 |
| GIS SMB (Fettweis) | 396 | 401 | 396-401 |
| GIS SID | 409 | 483 | 424-436 (discharge), 462-474 (interpolation) |
| AIS SMB | 492 | 499 | 493-498 |
| AIS SID (DeConto) | 507 | 570 | 522-536 (discharge + fast), 556-561 (contribution) |
| AIS SID (Levermann) | 574 | 666 | 618-626 (forcing), 629-657 (convolution) |
| Land Water | 673 | 696 | 683-689 (depletion), 691-693 (simple) |
| Total | 702 | 709 | 702-709 (summation) |
| Datastore Update | 712 | 752 | Copies to datastore structures |

### State Variable Declarations

| Variable Type | Lines |
|---------------|-------|
| Module arrays | 5-20 |
| Module scalars | 21-38 |
| File names | 39-41 |
| Lookup tables | 42-49 |
| IRF arrays | 67-69 |

### Parameter Files Referenced

| Parameter | Configuration File(s) |
|-----------|----------------------|
| Thermal expansion | `SLR_THEXP_PARAMS_*.CFG` |
| Glaciers | `MAGTUNE_GLACIERTUNE_*.CFG`, `SLR_GL_PARAMS_*.CFG` |
| GIS SMB | `MAGTUNE_GISSMBTUNE_*.CFG` |
| AIS SMB | `MAGTUNE_AISSMBTUNE_*.CFG` |
| AIS SID | `MAGTUNE_AISSIDTUNE_*.CFG` |
| AIS IRF | `SLR_AIS_SID_IRF_*_*.CFG` |
| Land water | `SLR_LANDWATER_HIGH.IN`, `SLR_LANDWATER_LOW.IN` |
| Historic | `HIST_SEALEVEL_CHURCHWHITE2006_RF.IN` |
