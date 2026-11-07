# Sea Level Rise

/// warning | Experimental in MAGICC7
MAGICC7 marks this module as experimental. Lines 14 to 16 of `MAGICC7.f90` say:
"There are several components in this program that are of experimental nature and not fit-for-use.
This includes sea level rise projections, the permafrost module, ..."
///

## Overview

The sea level rise module calculates global mean sea level rise from the components that respond to warming:

- **Thermal expansion**: warming ocean water expands.
- **Glaciers and ice caps**: mountain glaciers and small ice caps melt.
- **Greenland ice sheet (GIS)**: surface mass balance (SMB) and solid ice discharge (SID).
- **Antarctic ice sheet (AIS)**: surface mass balance and solid ice discharge from four drainage regions.
- **Land water storage**: groundwater extraction and reservoirs.

Each contribution is a simple parameterisation calibrated to CMIP5 models and ice sheet model ensembles.
Each has its own start year and can be switched independently.

## RSCM implementation

- **Component**: `SeaLevelRise`, built from Python with `rscm.magicc.SeaLevelRiseBuilder`,
  calculates glaciers and the Greenland and Antarctic SMB and SID, and sums them with thermal expansion.
  `ClimateUdeb` calculates thermal expansion when its `thermal_expansion` parameter is set,
  because it needs the ocean layer temperatures.
- **Source**: `crates/rscm-magicc/src/climate/sea_level.rs` and `climate/udeb/mod.rs`,
  parameters in `parameters/sea_level.rs`.
- **Status**: Implemented. See the [status table](README.md#rscm-implementation-status).
  `rscm.magicc.full.run_magicc7` adds both with `out_sealevel=1`. MAGICC7's default is 0.
- **Tests**: `tests/regression/test_sea_level.py::test_contribution` runs the coupled chain
  on MAGICC7 v7.5.3's SSP245 emissions run with sea level (`04_ssp245_emissions_sealevel`)
  and matches every contribution and the total in MAGICC7's binary output.
  `test_sea_level_off_by_default` checks that the default chain has no sea level.
  Unit tests in `sea_level.rs` cover expansion, interpolation, zero warming and the Greenland volume limit.

The equations below are the formulation RSCM reproduces.
Comparison with MAGICC7 output corrected the earlier reading of the Fortran in several places:
the start-year gates, the thermal expansion weighting, and the Levermann forcing and convolution.

## Equations

### Timing

A year's annual mean warming $T(y)$ (`DAT_SURFACE_ANNUALMEANTEMP`, the mean of its 12 monthly steps)
sets the contributions at the start of year $y + 1$.
A contribution runs for $y \ge$ its start year,
so it first changes in the value for the year after its start year.
All temperatures are global mean surface warming in K, and all contributions are in mm.

### Thermal expansion

Ocean layer $l$ has thickness $z_l$: the mixed layer depth (`CORE_MIXEDLAYER_DEPTH`) for the top layer
and `DZ` (100 m) below it.
Its area in hemisphere $h$, $a_{h,l}$, is the mean of the ocean area at its upper and lower bounds
(`OCN_AREAFACTOR_AVERAGE`).
The layer temperature is the mean of the hemispheres weighted by their ocean area in that layer:

$$
\bar T_l = \frac{\sum_h a_{h,l}\, f_h\, T_{h,l}}{\sum_h a_{h,l}\, f_h}
$$

where $f_h$ is the hemisphere's ocean fraction of the globe (`FGO`) and $T_{h,l}$ the layer warming.
With the scaling $s$ = `SLR_EXPANSION_SCALING`, applied once,
the layer warming over the year and the mean layer temperature are

$$
\Delta_l = s \left( \bar T_l(y+1) - \bar T_l(y) \right), \qquad
T_{L1} = s\, \frac{\bar T_l(y+1) + \bar T_l(y)}{2} + \frac{T^0_{N,l} + T^0_{S,l}}{2}
$$

where $T^0_{h,l}$ is the initial ocean temperature profile (`CORE_SWITCH_OCN_TEMPPROFILE`, CMIP5 by default).
The expansion coefficient is an empirical polynomial in $T_{L1}$ and the pressure $P$
with coefficients $p_1 \dots p_6$ = `SLR_EXPANSION_PARAMS`:

$$
\begin{aligned}
c_l = {} & p_1 + p_2\, T_{L1} (12.9635 - 1.0833 P) - p_3\, T_{L1}^2 (0.1713 - 0.019263 P) \\
& + p_4\, \frac{T_{L1}^3}{6000} (10.41 - 1.1338 P) + p_5 P - p_6 P^2
\end{aligned}
$$

$P$ (`OCN_PRESSURE_PROFILE`) is evaluated at the layer centre depth $z$ in m:

$$
P = 0.0098 \left( 10.5 \left( e^{-z/3500} - 1 \right) + 0.1005\, z \right)
$$

The year's expansion weights each layer by the mean of its two hemisphere areas $\bar a_l$,
divided by the mean surface area $\bar a_0$, which is 1:

$$
\Delta S_{exp} = \frac{1}{\bar a_0} \sum_l \bar a_l\, \frac{c_l\, \Delta_l\, z_l}{1000}
$$

$S_{exp}$ accumulates $\Delta S_{exp}$ from `SLR_EXPANSION_STARTYEAR`.

### Glaciers and ice caps

Glaciers follow a modified equation 5 of Wigley and Raper (2005), with equilibrium tables from Marzeion.
Two lookups in the tables `SLR_GL_EQUITEMP` and `SLR_GL_EQUISLR`, linear and clamped at the ends, give
the equilibrium sea level at the current warming, $S_{eq}(T)$ (`SLR_GL_EQUISLRCURRENTTEMP`),
and the equilibrium warming at the current contribution, $T_{eq}(S_{GL})$ (`SLR_GL_EQUITEMPCURRENTSLR`).
Then

$$
S_{GL}(y+1) = S_{GL}(y)
+ k\, \frac{S_{eq} - S_{GL}(y)}{n_V}\, \frac{\operatorname{sign}(T_{eq})\, |T - T_{eq}|^{e}}{n_T}
$$

with $k$ = `SLR_GL_SENS_MMPYRDEG`, $e$ = `SLR_GL_TEMP_EXPONENT`,
and normalisations $n_V$ = `SLR_GL_NORMPARA_VOL` and $n_T$ = `SLR_GL_NORMPARA_TEMP`.
The exponent applies to the absolute difference, and the sign is that of $T_{eq}$,
as Fortran's `SIGN(ABS(T - T_EQ)**E, T_EQ)` gives.

### Greenland surface mass balance

The `DEFAULT` parameterisation,
with $c_1$, $c_2$ and $e$ = `SLR_GIS_SMB_COEF1`, `_COEF2` and `_SENS_EXPONENT`,
shrinks as the ice sheet's volume $V$ (`SLR_GIS_SMB_INITIAL_VOLUME_MM`) is used up:

$$
S_{GIS,SMB}(y+1) = S_{GIS,SMB}(y)
+ c_1 \left( c_2 T + (1 - c_2) \max(0, T)^{e} \right)
\max\left(0, 1 - \frac{S_{GIS,SMB}(y)}{V}\right)^{e_V}
$$

with $e_V$ = `SLR_GIS_SMB_VOLUME_EXPONENT`.
The `FETTWEIS` alternative adds $(a_1 T - a_2 T^2 - a_3 T^3) / (-361)$ each year,
with $a_1 \dots a_3$ = `SLR_GIS_SMB_COEF_FW1` to `_FW3`.

### Greenland solid ice discharge

Discharge follows Nick et al. (2013), with a low and a high case $k$.
Each year with $T \ne 0$ from `SLR_GIS_SID_STARTYEAR`, each case discharges from its remaining volume $V_k$:

$$
q_k = \max\left(-V_k,\ \min\left(0,\ -d_k\, V_k\, e^{x_k T}\right)\right), \qquad
V_k \leftarrow \max(0, V_k + q_k)
$$

with $d_k$ = `SLR_GIS_SID_DSCHRG_SENS_k` and $x_k$ = `SLR_GIS_SID_TEMPSENS_EXPONENT_k`.
$V_k$ starts at `SLR_GIS_SID_TOTALVOL_k`, $V^0_k$.
The contribution interpolates between the cases with $c$ = `SLR_GIS_SID_CASE`
and scales with $\sigma$ = `SLR_GIS_SID_SCALING`:

$$
S_{GIS,SID} = \sigma \left( (S_{high} - S_{low})\, c + S_{low} \right), \qquad S_k = V^0_k - V_k
$$

### Antarctic surface mass balance

Antarctic SMB has the same form as Greenland's without the volume term:

$$
S_{AIS,SMB}(y+1) = S_{AIS,SMB}(y) + c_1 \left( c_2 T + (1 - c_2) \max(0, T)^{e} \right)
$$

Its effective $c_1$ is negative: more snowfall with warming gains mass and lowers sea level.

### Antarctic solid ice discharge: LEVERMANN (default)

Four drainage regions $r$ (Amundsen, East Antarctica, Ross, Weddell) respond to basal melt
through impulse response functions (Levermann et al. 2014).
Region $r$ starts in year $y_r$ = `SLR_AIS_SID_STARTYEAR` + `SLR_AIS_SID_DT_r`.
From then its basal melt forcing is relative to the warming in its first year:

$$
m_r(y) = b\, \theta_r \left( T(y) - T(y_r) \right)
$$

with $b$ = `SLR_AIS_SID_BASALMELT` and $\theta_r$ = `SLR_AIS_SID_TEMPSCALING_r`.
The response to a melt pulse of age $a$ is a quartic, clamped at zero and cut off after
`SLR_AIS_SID_IRF_YRSPAN` years, with coefficients $c_{r,1} \dots c_{r,5}$ = `SLR_AIS_SID_IRF_r`:

$$
R_r(a) = \max\left(0,\ c_{r,1} a^4 + c_{r,2} a^3 + c_{r,3} a^2 + c_{r,4} a + c_{r,5}\right)
$$

Each region's contribution is the full convolution of its forcing with the response,
converted from m to mm, and the regions sum to

$$
S_{AIS,SID}(y+1) = \sigma_{AIS} \sum_r 1000 \sum_{a=0}^{y - y_r} m_r(y - a)\, R_r(a)
$$

with $\sigma_{AIS}$ = `SLR_AIS_SID_SCALING`.
Two MAGICC7 quirks are part of the formulation:

- **Response overflow**: MAGICC7 raises the age to its powers in 32-bit integers,
  so $a^4$ wraps from $a = 216$.
  Ross, with no delay in the default configuration, reaches that age in 2067.
- **Start release**: in the year $y_r - 1$,
  MAGICC7 adds $1000\, b\, \theta_r\, T(y)\, R_r(0)$ for that year alone,
  against zero warming.

### Antarctic solid ice discharge: DECONTO

The alternative adds a fast rate above a threshold, for ice cliff instability (DeConto and Pollard 2016):

$$
q = d\, V \operatorname{sign}(T - T_0)\, |T - T_0|^{e}
+ \begin{cases} r_{fast} & T \ge T_{thr} \\ 0 & T < T_{thr} \end{cases}
$$

bounded by the remaining volume $V$, which depletes from `SLR_AIS_DISCHARGE_STARTYEAR`.
The contribution is the depleted volume times the scaling, minus the Antarctic SMB contribution.

### Land water storage

Off by default (`SLR_LANDWATER_SWITCH = 0`).
The contribution adds a prescribed rate $w(y)$ (mm yr⁻¹, interpolated between the low and high files by
`SLR_LANDWATER_CASE`).
After `SLR_LANDWATER_SWITCHYEAR` the rate shrinks as the reservoir $V_{max}$ is used up:

$$
S_{LW}(y+1) = S_{LW}(y)
+ w(y) \max\left(0,\ 1 - \frac{S_{LW}(y) - S_{switch}}{V_{max} - S_{switch}}\right)^{e_V}
$$

### Total

$$
S_{total} = S_{exp} + S_{GL} + S_{GIS,SMB} + S_{GIS,SID} + S_{AIS,SMB} + S_{AIS,SID} + S_{LW}
$$

## State variables

| Variable                  | Fortran name                          | Units | Description                       | Initial value               |
| ------------------------- | ------------------------------------- | ----- | --------------------------------- | --------------------------- |
| Total                     | `SLR_TOTAL_CONTRIBUTION`              | mm    | Cumulative total sea level rise   | 0                           |
| Thermal expansion         | `SLR_EXPANSION_CONTRIBUTION`          | mm    | $S_{exp}$                         | 0                           |
| Glaciers                  | `SLR_GL_CONTRIBUTION`                 | mm    | $S_{GL}$                          | 0                           |
| GIS SMB                   | `SLR_GIS_SMB_CONTRIBUTION`            | mm    | $S_{GIS,SMB}$                     | 0                           |
| GIS SID                   | `SLR_GIS_SID_CONTRIBUTION`            | mm    | $S_{GIS,SID}$                     | 0                           |
| GIS SID low and high      | `SLR_GIS_SID_CONTRIBUTION_LOW`/`_HIGH` | mm   | $S_{low}$, $S_{high}$             | 0                           |
| GIS discharge volumes     | `SLR_GIS_SID_DSCHRGVOL_LOW`/`_HIGH`   | mm    | $V_{low}$, $V_{high}$             | `SLR_GIS_SID_TOTALVOL_*`    |
| AIS SMB                   | `SLR_AIS_SMB_CONTRIBUTION`            | mm    | $S_{AIS,SMB}$                     | 0                           |
| AIS SID                   | `SLR_AIS_SID_CONTRIBUTION`            | mm    | $S_{AIS,SID}$                     | 0                           |
| AIS discharge volume      | `SLR_AIS_SID_DSCHRGVOL`               | mm    | DECONTO $V$                       | `SLR_AIS_SID_TOTALVOL`      |
| Land water                | `SLR_LANDWATER_CONTRIBUTION`          | mm    | $S_{LW}$                          | 0                           |
| AIS region responses      | `SLR_AIS_SID_R_*`                     | m per m yr⁻¹ | $R_r(a)$ for each region    | From the polynomials        |
| AIS forcing               | `SLR_AIS_SID_OCNFORCE`, `_FORCING`    | m yr⁻¹ | $m_r(y)$, [4, NYEARS]           | 0                           |
| AIS convolution terms     | `SLR_AIS_SID_CONV_*`                  | m     | Convolution partial sums          | 0                           |

RSCM's `SeaLevelRiseState` keeps the glacier and SMB contributions, the two Greenland volumes,
and each Antarctic region's baseline warming and melt history.
`ClimateUdeb` keeps the cumulative thermal expansion.

## Inputs and outputs

| Direction | Variable                  | Units | Fortran variable                       | RSCM variable                                     |
| --------- | ------------------------- | ----- | -------------------------------------- | ------------------------------------------------- |
| Input     | Annual mean warming       | K     | `DAT_SURFACE_ANNUALMEANTEMP%DATGLOBE`  | `Surface Temperature|Annual Mean`                |
| Input     | Ocean layer temperatures  | K     | `TEMP_OCEANLAYERS(I, CURRENT/NEXT_YEAR_IDX)` | `ClimateUdeb` state                         |
| Input     | Land water rate           | mm yr⁻¹ | `SLR_LANDWATER_MMPYEAR`              | Not implemented                                   |
| Output    | Total                     | mm    | `DAT_SLR_TOT`                          | `Sea Level Rise`                                  |
| Output    | Thermal expansion         | mm    | `DAT_SLR_EXPANSION`                    | `Sea Level Rise|Thermal Expansion`               |
| Output    | Glaciers                  | mm    | `DAT_SLR_GL`                           | `Sea Level Rise|Glaciers`                        |
| Output    | GIS SMB                   | mm    | `DAT_SLR_GIS_SMB`                      | `Sea Level Rise|Greenland|Surface Mass Balance` |
| Output    | GIS SID                   | mm    | `DAT_SLR_GIS_SID`                      | `Sea Level Rise|Greenland|Solid Ice Discharge`  |
| Output    | AIS SMB                   | mm    | `DAT_SLR_AIS_SMB`                      | `Sea Level Rise|Antarctica|Surface Mass Balance` |
| Output    | AIS SID                   | mm    | `DAT_SLR_AIS_SID`                      | `Sea Level Rise|Antarctica|Solid Ice Discharge` |
| Output    | Land water                | mm    | `DAT_SLR_LANDWATER`                    | Not implemented                                   |
| Output    | Semi-empirical total and rate | mm, mm yr⁻¹ | `DAT_SLR_SEMIEMPI_TOT`, `_RATE` | Not implemented                                |

Thermal expansion also uses the ocean geometry that the climate module sets up:
`CORE_OCN_NLEVELS`, `CORE_MIXEDLAYER_DEPTH`, `DZ`,
the hemisphere layer areas and the initial temperature profile.

## Parameters

The "CFG default" column lists `MAGCFG_DEFAULTALL.CFG`.
MAGICC7's CMIP5 mean and tuning files override most of these values.
The "Effective default" column lists the values after those files are applied,
which `SeaLevelRiseParameters` and `ThermalExpansionParameters` hold.
A dash marks a parameter RSCM does not have.

### Thermal expansion

| Parameter              | Fortran name              | Units   | CFG default                          | Effective default                                    | Description               |
| ---------------------- | ------------------------- | ------- | ------------------------------------ | ---------------------------------------------------- | ------------------------- |
| Expansion coefficients | `SLR_EXPANSION_PARAMS`    | various | [52.24, 0.99, 1.03, 0.61, 24.26, 0.74] | [52.2368, 0.9938, 1.0333, 0.6065, 24.2643, 0.7408] | $p_1 \dots p_6$           |
| Expansion scaling      | `SLR_EXPANSION_SCALING`   | 1       | 0.8824                               | 0.8823828                                            | $s$                       |
| Start year             | `SLR_EXPANSION_STARTYEAR` | year    | 1850                                 | 1850                                                 | First year of expansion   |
| Parameter file         | `FILE_SLR_THEXP_PARAMS`   |         | `SLR_THEXP_PARAMS_CMIP5MEAN.CFG`     |                                                      | Source of the effective values |

### Glaciers

| Parameter             | Fortran name             | Units | CFG default                          | Effective default    | Description                    |
| --------------------- | ------------------------ | ----- | ------------------------------------ | -------------------- | ------------------------------ |
| Sensitivity           | `SLR_GL_SENS_MMPYRDEG`   | yr⁻¹  | 0.625                                | 1.067337 × 10⁻²      | $k$                            |
| Temperature exponent  | `SLR_GL_TEMP_EXPONENT`   | 1     | 0.82                                 | 7.881273 × 10⁻²      | $e$                            |
| Volume normalisation  | `SLR_GL_NORMPARA_VOL`    | 1     | 1.0                                  |                     | $n_V$                          |
| Temperature normalisation | `SLR_GL_NORMPARA_TEMP` | 1   | 1.0                                  |                     | $n_T$                          |
| Start year            | `SLR_GL_STARTYEAR`       | year  | 1850                                 | 1850                 | First year of melt             |
| Equilibrium warming   | `SLR_GL_EQUITEMP`        | K     | 0.0, 0.1, ..., 10.3 (104 values)     | Same                 | Lookup table                   |
| Equilibrium sea level | `SLR_GL_EQUISLR`         | mm    | 81.2, 96.6, ..., 410.2 (104 values)  | Same                 | Lookup table                   |
| Tuning file           | `FILE_SLR_GL_PARAMS`     |       | `MAGTUNE_GLACIERTUNE_CMIP5MEAN.CFG`  |                      | Source of $k$ and $e$          |
| Table file            | `FILE_SLR_GL_XTRAPARAMS` |       | `SLR_GL_PARAMS_CMIP5MEAN.CFG`        |                      | Source of the tables           |

### Greenland SMB

| Parameter            | Fortran name                    | Units | CFG default | Effective default   | Description              |
| -------------------- | ------------------------------- | ----- | ----------- | ------------------- | ------------------------ |
| Parameterisation     | `SLR_GIS_SMB_PARAMETERISATION`  |       | `DEFAULT`   | `DEFAULT` only      | `DEFAULT` or `FETTWEIS`  |
| Coefficient 1        | `SLR_GIS_SMB_COEF1`             | mm yr⁻¹ | 0.015     | 0.1148748           | $c_1$                    |
| Coefficient 2        | `SLR_GIS_SMB_COEF2`             | 1     | 0.9         | 1.161776 × 10⁻⁷     | $c_2$, linear weight     |
| Sensitivity exponent | `SLR_GIS_SMB_SENS_EXPONENT`     | 1     | 2.3         | 2.016917            | $e$                      |
| Initial volume       | `SLR_GIS_SMB_INITIAL_VOLUME_MM` | mm    | 7360.0      | 7360.0              | $V$                      |
| Volume exponent      | `SLR_GIS_SMB_VOLUME_EXPONENT`   | 1     | 0.5         | 0.5                 | $e_V$                    |
| Fettweis coefficients | `SLR_GIS_SMB_COEF_FW1` to `_FW3` | 1  | -10.0, 2.0, 1.0 |                 | $a_1 \dots a_3$          |
| Start year           | `SLR_GIS_SMB_STARTYEAR`         | year  | 1965        | 1965                | First year of SMB change |
| Tuning file          | `FILE_SLR_GISSMB_PARAMS`        |       | `MAGTUNE_GISSMBTUNE_CMIP5MEAN.CFG` |  | Source of the effective values |

### Greenland SID

| Parameter          | Fortran name                         | Units | CFG default | Effective default | Description                  |
| ------------------ | ------------------------------------ | ----- | ----------- | ----------------- | ---------------------------- |
| Case selector      | `SLR_GIS_SID_CASE`                   | 1     | 0.5         | 0.5               | $c$, low (0) to high (1)     |
| Scaling            | `SLR_GIS_SID_SCALING`                | 1     | 5.0         | 5.0               | $\sigma$, AR5 upscaling      |
| Total volume low   | `SLR_GIS_SID_TOTALVOL_LOW`           | mm    | 35.98       | 35.9803           | $V^0_{low}$                  |
| Total volume high  | `SLR_GIS_SID_TOTALVOL_HIGH`          | mm    | 53.63       | 53.6289           | $V^0_{high}$                 |
| Discharge sens low | `SLR_GIS_SID_DSCHRG_SENS_LOW`        | yr⁻¹  | 0.000906    | 9.0617 × 10⁻⁴     | $d_{low}$                    |
| Discharge sens high | `SLR_GIS_SID_DSCHRG_SENS_HIGH`      | yr⁻¹  | 0.000793    | 7.933 × 10⁻⁴      | $d_{high}$                   |
| Temp exponent low  | `SLR_GIS_SID_TEMPSENS_EXPONENT_LOW`  | K⁻¹   | 0.389       | 0.38919           | $x_{low}$                    |
| Temp exponent high | `SLR_GIS_SID_TEMPSENS_EXPONENT_HIGH` | K⁻¹   | 0.472       | 0.47216           | $x_{high}$                   |
| Start year         | `SLR_GIS_SID_STARTYEAR`              | year  | 2000        | 2000              | First year of discharge      |

### Antarctic SMB

| Parameter            | Fortran name                | Units   | CFG default | Effective default    | Description              |
| -------------------- | --------------------------- | ------- | ----------- | -------------------- | ------------------------ |
| Coefficient 1        | `SLR_AIS_SMB_COEF1`         | mm yr⁻¹ | 0.128       | -0.1208204           | $c_1$                    |
| Coefficient 2        | `SLR_AIS_SMB_COEF2`         | 1       | -0.424      | -7.702955 × 10⁻¹⁵    | $c_2$, linear weight     |
| Sensitivity exponent | `SLR_AIS_SMB_SENS_EXPONENT` | 1       | 0.782       | 1.523378             | $e$                      |
| Start year           | `SLR_AIS_SMB_STARTYEAR`     | year    | 1980        | 1980                 | First year of SMB change |
| Tuning file          | `FILE_SLR_AISSMB_PARAMS`    |         | `MAGTUNE_AISSMBTUNE_MEAN.CFG` |      | Source of the effective values |

### Antarctic SID, common

| Parameter            | Fortran name                   | Units | CFG default   | Effective default | Description                  |
| -------------------- | ------------------------------ | ----- | ------------- | ----------------- | ---------------------------- |
| Parameterisation     | `SLR_AIS_SID_PARAMETERISATION` |       | `LEVERMANN`   | `LEVERMANN` only  | `LEVERMANN` or `DECONTO`     |
| Scaling              | `SLR_AIS_SID_SCALING`          | 1     | 1.0           | 1.0               | $\sigma_{AIS}$               |
| Start year           | `SLR_AIS_SID_STARTYEAR`        | year  | 1850          | 1850              | First year of discharge      |
| Discharge start year | `SLR_AIS_DISCHARGE_STARTYEAR`  | year  | 1950          |                   | DECONTO volume depletion     |
| Tuning file          | `FILE_SLR_AISSID_PARAMS`       |       | `MAGTUNE_AISSIDTUNE_MIROCESM.CFG` |  | Source of the effective values |

### Antarctic SID, DECONTO

| Parameter             | Fortran name                    | Units   | CFG default | Description          |
| --------------------- | ------------------------------- | ------- | ----------- | -------------------- |
| Total volume          | `SLR_AIS_SID_TOTALVOL`          | mm      | 17560.0     | Initial $V$          |
| Discharge sensitivity | `SLR_AIS_SID_DSCHRG_SENS`       | yr⁻¹    | 5.28 × 10⁻⁵ | $d$                  |
| Temp exponent         | `SLR_AIS_SID_TEMPSENS_EXPONENT` | 1       | 2.0         | $e$                  |
| Threshold warming     | `SLR_AIS_SID_THRESHOLDTEMP`     | K       | 1.023       | $T_{thr}$            |
| Zero warming          | `SLR_AIS_SID_ZEROTEMP`          | K       | 0.0         | $T_0$                |
| Fast rate             | `SLR_AIS_SID_FASTRATE`          | mm yr⁻¹ | 13.83       | $r_{fast}$           |

RSCM does not implement DECONTO, so these have no effective default.

### Antarctic SID, LEVERMANN

| Parameter                 | Fortran name                             | Units     | CFG default | Effective default | Description          |
| ------------------------- | ---------------------------------------- | --------- | ----------- | ----------------- | -------------------- |
| Basal melt                | `SLR_AIS_SID_BASALMELT`                  | m yr⁻¹ K⁻¹ | 11.5       | 11.5              | $b$                  |
| Response span             | `SLR_AIS_SID_IRF_YRSPAN`                 | yr        | 500         | 500               | Response cut-off     |
| Delay, Amundsen           | `SLR_AIS_SID_DT_AMUNDSEN`                | yr        | 0           | 60                | Years after start    |
| Delay, East Antarctica    | `SLR_AIS_SID_DT_EASTANTARCTICA`          | yr        | 30          | 50                | Years after start    |
| Delay, Ross               | `SLR_AIS_SID_DT_ROSS`                    | yr        | 20          | 0                 | Years after start    |
| Delay, Weddell            | `SLR_AIS_SID_DT_WEDDELL`                 | yr        | 35          | 65                | Years after start    |
| Temp scaling, Amundsen    | `SLR_AIS_SID_TEMPSCALING_AMUNDSEN`       | 1         | 0.17        | 0.35              | $\theta_r$           |
| Temp scaling, East Antarctica | `SLR_AIS_SID_TEMPSCALING_EASTANTARCTICA` | 1     | 0.35        | 0.24              | $\theta_r$           |
| Temp scaling, Ross        | `SLR_AIS_SID_TEMPSCALING_ROSS`           | 1         | 0.26        | 0.23              | $\theta_r$           |
| Temp scaling, Weddell     | `SLR_AIS_SID_TEMPSCALING_WEDDELL`        | 1         | 0.14        | 0.18              | $\theta_r$           |

The response coefficients `SLR_AIS_SID_IRF_r` are the same in both, highest power first:

| Region          | $c_{r,1}$         | $c_{r,2}$         | $c_{r,3}$        | $c_{r,4}$        | $c_{r,5}$        |
| --------------- | ----------------- | ----------------- | ---------------- | ---------------- | ---------------- |
| Amundsen        | 3.80221625e-15    | -1.20779839e-11   | 5.34616000e-09   | -1.12359405e-07  | 2.69854318e-05   |
| East Antarctica | -4.80672912e-15   | 4.68446533e-12    | -1.32757538e-09  | 1.64696643e-07   | 1.14405405e-05   |
| Ross            | -6.09433726e-14   | 5.15605867e-11    | -1.36480364e-08  | 1.75347964e-06   | -2.21886314e-05  |
| Weddell         | 1.54097171e-14    | -1.53064596e-11   | 5.22078742e-09   | -5.09486184e-07  | 3.54817378e-05   |

### Land water storage

| Parameter       | Fortran name                    | Units | CFG default             | Description                      |
| --------------- | ------------------------------- | ----- | ----------------------- | -------------------------------- |
| Switch          | `SLR_LANDWATER_SWITCH`          | flag  | 0                       | Enable land water                |
| Case selector   | `SLR_LANDWATER_CASE`            | 1     | 1.0                     | Low (0) to high (1)              |
| Start year      | `SLR_LANDWATER_STARTYEAR`       | year  | 1900                    | First year                       |
| Switch year     | `SLR_LANDWATER_SWITCHYEAR`      | year  | 2100                    | Start of volume depletion        |
| Maximum volume  | `SLR_LANDWATER_MAXVOLUME_MM`    | mm    | 1000.0                  | $V_{max}$                        |
| Volume exponent | `SLR_LANDWATER_VOLUME_EXPONENT` | 1     | 0.5                     | $e_V$                            |
| Rate files      | `FILE_SLR_LANDWATER_HIGH`/`_LOW` |      | `SLR_LANDWATER_HIGH.IN`, `SLR_LANDWATER_LOW.IN` | High and low rates |

### Semi-empirical method and output control

| Parameter            | Fortran name                      | Units      | CFG default    | Description                       |
| -------------------- | --------------------------------- | ---------- | -------------- | --------------------------------- |
| Zero rate warming    | `SLR_SEMIEMPI_ZERORATETEMP`       | K          | -0.5           | Warming offset                    |
| Rate sensitivity     | `SLR_SEMIEMPI_RATE_SENS`          | mm yr⁻¹ K⁻¹ | 0.3353        | Rahmstorf coefficient             |
| Base period          | `SLR_SEMIEMPI_TEMPBASEPERIOD`     | years      | [1980, 1999]   | Reference period                  |
| Switch year          | `SLR_SEMIEMPI_SWITCHFROMOBS2CALC` | year       | 2000           | Observed to calculated            |
| Calculated from year | `SLR_SWITCH_CALC2HIST`            | year       | -10000         | Calculated rather than historic   |
| Historic file        | `FILE_SLR_HISTORIC`               |            | `HIST_SEALEVEL_CHURCHWHITE2006_RF.IN` | Observed sea level  |
| Zero period          | `OUT_ZERO_SLR_PERIOD`             | years      | [1986, 2005]   | Anomaly reference period          |

## Algorithm

`sealevel_init` zeroes the contributions and:

- builds each region's response $R_r(a)$ from its polynomial,
- reads the historic sea level file into the total and semi-empirical series,
- reads the land water files and interpolates them by `SLR_LANDWATER_CASE`.

`sealevel_calc` runs at the end of each year's step (`MAGICC7.f90` line 3726), after the climate calculation.
The contributions are independent of each other except at the end, in this order and with these gates:

1. Thermal expansion, from `SLR_EXPANSION_STARTYEAR`.
2. Glaciers, from `SLR_GL_STARTYEAR`.
3. Greenland SMB, from `SLR_GIS_SMB_STARTYEAR`, by `SLR_GIS_SMB_PARAMETERISATION`.
4. Greenland SID, from `SLR_GIS_SID_STARTYEAR` and only while $T \ne 0$.
5. Antarctic SMB, from `SLR_AIS_SMB_STARTYEAR`.
6. Antarctic SID, by `SLR_AIS_SID_PARAMETERISATION`, each Levermann region from $y_r$
   (and the start release in $y_r - 1$).
7. Land water, only with `SLR_LANDWATER_SWITCH`, from `SLR_LANDWATER_STARTYEAR`.
8. Total, then copied to the output series after `SLR_SWITCH_CALC2HIST`.

`sealevel_calc_semiempi` is a separate, observation-driven method after Rahmstorf:

- The rate is `SLR_SEMIEMPI_RATE_SENS` times the warming
  above the base period mean and `SLR_SEMIEMPI_ZERORATETEMP`.
- It integrates the rate from the end of the historic record, or `SLR_SEMIEMPI_SWITCHFROMOBS2CALC` if earlier.
- It reports the result in the two ocean boxes only.

## Issues in MAGICC7

### Hardcoded values

| Value                     | Line           | Use                                       |
| ------------------------- | -------------- | ----------------------------------------- |
| `-361.0`                  | 401            | Fettweis GIS SMB divisor                  |
| `1000.0`                  | 657            | m to mm in the Levermann convolution      |
| `6000.0`                  | 289            | Divisor of the cubic expansion term       |
| `12.9635`, `1.0833`, ...  | 292-297        | Expansion polynomial constants            |
| `104`                     | 42             | Size of the glacier lookup tables         |
| `4`                       | Throughout     | Number of Antarctic regions               |

### Code and numerics

- **Index arithmetic**: `CURRENT_YEAR_IDX`, `NEXT_YEAR_IDX`, `AIS_SID_IDX`, `AIS_SID_DT_STARTIDX`
  and `AIS_SID_DT_IDX` make the timing hard to follow.
  The Levermann convolution sums its first and last terms apart from the rest.
- **Integer overflow** in the response polynomial from an age of 216 years, as above.
- **`SAVE` local**: `AIS_SID_DISCHARGE` keeps its value between calls (line 258).
- **Zero-warming guards**: Greenland SID and DECONTO skip years with $T = 0$ exactly (lines 412, 510).
- **No bounds checks** on the table lookups, and `MAX(0, ...)` guards in place of volume conservation checks.
- **Stale defaults**: the tuning files override most of `MAGCFG_DEFAULTALL.CFG`,
  so the configuration file does not show the values a run uses.

### Scientific limitations

- **Simple parameterisations** in place of ice sheet models.
  SMB has no elevation feedback, and glaciers have no hypsometry.
- **Antarctic instability**: DECONTO's fast rate is a threshold approximation of marine ice cliff instability,
  and LEVERMANN has none.
- **Glacier tables** stop at 10.3 K of warming.
- **Global mean only**: no regional patterns, fingerprints or glacial isostatic adjustment.
- **Uncertainty**: only Greenland SID has low and high cases.
- **Land water** extrapolates a prescribed historical series.

## Differences from MAGICC7

- **Not implemented**: land water (off by default), the semi-empirical method,
  the `FETTWEIS` and `DECONTO` alternatives, and the anomalies to 1986 to 2005 (`ZEROED_*`).
- **Glacier normalisations**: RSCM has no `SLR_GL_NORMPARA_VOL` or `SLR_GL_NORMPARA_TEMP`, so it assumes 1.
- **Initial ocean profile**: `ocean_temp_profile` offers the `analytical` exponential (MAGICC7 1) and `cmip5` (2) profiles.
  MAGICC7's third option, `TEMP_OCEAN_INI_PROFILE_ONE`, is not available.
- **Steps must be one year long.**

## References

### Fortran code (`sealevel.f90`)

| Subroutine               | Lines   | Description                 |
| ------------------------ | ------- | --------------------------- |
| `sealevel_alloc`         | 73-95   | Array allocation            |
| `sealevel_dealloc`       | 97-114  | Array deallocation          |
| `sealevel_init`          | 116-237 | Initialisation and IRF setup |
| `sealevel_calc`          | 239-753 | Main calculation            |
| `sealevel_calc_semiempi` | 755-830 | Semi-empirical method       |

| Component           | Lines   | Key equations                                       |
| ------------------- | ------- | --------------------------------------------------- |
| Thermal expansion   | 266-312 | 292-297 coefficient, 299 layer, 303-308 sum         |
| Glaciers            | 319-373 | 364-371 Wigley and Raper                            |
| GIS SMB (default)   | 383-391 | 386-391                                             |
| GIS SMB (Fettweis)  | 396-401 | 396-401                                             |
| GIS SID             | 409-483 | 424-436 discharge, 462-474 interpolation            |
| AIS SMB             | 492-499 | 493-498                                             |
| AIS SID (DeConto)   | 507-570 | 522-536 discharge and fast rate, 556-561 contribution |
| AIS SID (Levermann) | 574-666 | 618-626 forcing, 629-657 convolution                |
| Land water          | 673-696 | 683-689 depletion, 691-693 simple                   |
| Total               | 702-709 | Sum                                                 |
| Datastore update    | 712-752 | Copies to the output series                         |

| Declarations    | Lines |
| --------------- | ----- |
| Module arrays   | 5-20  |
| Module scalars  | 21-38 |
| File names      | 39-41 |
| Lookup tables   | 42-49 |
| IRF arrays      | 67-69 |

| Contribution      | Configuration files                                      |
| ----------------- | -------------------------------------------------------- |
| Thermal expansion | `SLR_THEXP_PARAMS_*.CFG`                                 |
| Glaciers          | `MAGTUNE_GLACIERTUNE_*.CFG`, `SLR_GL_PARAMS_*.CFG`       |
| GIS SMB           | `MAGTUNE_GISSMBTUNE_*.CFG`                               |
| AIS SMB           | `MAGTUNE_AISSMBTUNE_*.CFG`                               |
| AIS SID           | `MAGTUNE_AISSIDTUNE_*.CFG`                               |
| AIS IRF           | `SLR_AIS_SID_IRF_*_*.CFG`                                |
| Land water        | `SLR_LANDWATER_HIGH.IN`, `SLR_LANDWATER_LOW.IN`          |
| Historic          | `HIST_SEALEVEL_CHURCHWHITE2006_RF.IN`                    |

### Literature

- DeConto and Pollard (2016), for the DECONTO AIS discharge.
- Levermann et al. (2014), for the LEVERMANN AIS discharge response functions.
- Nick et al. (2013), for the GIS solid ice discharge.
- Wigley and Raper (2005), for the glacier equation.
