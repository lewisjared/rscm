# Module 03: Halogenated Gas Chemistry

## 1. Scientific Purpose

This module models the atmospheric chemistry of halocarbons - synthetic compounds containing carbon and halogens (fluorine, chlorine, bromine). These fall into two regulatory categories:

1. **F-Gases (Fluorinated Gases under Kyoto Protocol)**: HFCs, PFCs, SF6, NF3, etc. These are potent greenhouse gases but do not deplete stratospheric ozone.

2. **Montreal Protocol Gases (MHALO)**: CFCs, HCFCs, halons, methyl chloride, methyl bromide, carbon tetrachloride, etc. These deplete stratospheric ozone AND contribute to radiative forcing.

The module tracks:

- Atmospheric concentrations via exponential decay with species-specific lifetimes
- Radiative forcing from infrared absorption
- Equivalent Effective Stratospheric Chlorine (EESC) for ozone depletion calculations
- Inverse emissions (deriving emissions from concentration trajectories)

## 2. Mathematical Formulation

### 2.1 Concentration Time Evolution

The atmospheric concentration follows exponential decay with emissions input:

```
dC/dt = E - C/tau_effective
```

Where:

- `C` = atmospheric concentration (ppt)
- `E` = emissions (converted to ppt/yr)
- `tau_effective` = effective total lifetime (years)

The discrete solution implemented uses two numerical schemes depending on lifetime:

**For long-lived species (tau >= 5 years) - Crank-Nicolson implicit:**

```
C(t+1) = [C(t) * (1 - 1/(2*tau)) + E(t) * conv_factor] / (1 + 1/(2*tau))
```

**For short-lived species (tau < 5 years) - Exact exponential:**

```
ex = exp(-1/tau)
C(t+1) = tau * E(t) * conv_factor * (1 - ex) + C(t) * ex
```

### 2.2 Conversion Factor (Emissions to Concentration)

```
conv_factor = (M_air / (M_atm * M_mol * f_mix))
```

Where:

- `M_air` = molar mass of air (28.984 g/mol)
- `M_atm` = total atmospheric mass (5.133 x 10^21 g)
- `M_mol` = molecular mass of species (calculated from atomic composition)
- `f_mix` = effective mixing box size (default 0.949)

### 2.3 Molecular Mass Calculation

```fortran
MOLMASS = C_atoms*12.001 + Cl_atoms*35.453 + F_atoms*18.9984 + Br_atoms*79.904
        + H_atoms*1.0079 + S_atoms*32.06 + O_atoms*15.999 + N_atoms*14.007
```

**Source:** `core.f90` lines 127-128

### 2.4 Effective Lifetime Calculation

The total effective lifetime combines multiple decay pathways:

```
1/tau_effective = 1/(tau_strat * scale_strat) + 1/(tau_OH * scale_OH) + 1/tau_other
```

Where `tau_other` is calculated at initialization to ensure the unscaled effective lifetime matches `tau_tot`:

```
tau_other = 1 / (1/tau_tot - 1/tau_OH - 1/tau_strat)
```

**Source:** `MAGICC7.f90` lines 2528-2561

### 2.5 Lifetime Scale Factors

**OH-dependent lifetime scaling (optional):**

```
scale_OH = CH4_tauOH_effective(t) / CH4_tauOH_init
```

Uses methane OH lifetime as proxy for oxidation capacity changes.

**Stratospheric lifetime scaling (optional):**

```
scale_strat = 1 / (1 + T_meridional * GEN_MERIDFLUX_CHNGPERDEG * TAUSTRAT_SENS2MERIDFLUX)
```

Based on Butchart & Scaife (2001) - ~3% increase in meridional flux per decade.

### 2.6 Radiative Forcing

Simple linear relationship:

```
RF = (C - C_preindustrial) * radeff / 1000
```

Where `radeff` is in W/m^2 per ppb, divided by 1000 to convert to W/m^2 per ppt.

**Source:** `deltaq_calculations.f90` lines 583-594

### 2.7 EESC Calculation

Equivalent Effective Stratospheric Chlorine combines Cl and Br contributions:

```
EESC += C_delayed * (n_Cl * f_release * f_CFC11 + alpha_Br * n_Br * f_release * f_CFC11)
```

Where:

- `C_delayed` = concentration at (t - delay) years, typically 3 years
- `n_Cl, n_Br` = number of chlorine/bromine atoms in molecule
- `f_release` = fractional release factor (species-specific)
- `f_CFC11` = CFC-11 release factor normalization (0.75 default)
- `alpha_Br` = bromine vs chlorine ozone destruction efficiency (default ~60)

**Source:** `deltaq_calculations.f90` lines 902-951

### 2.8 Stratospheric Ozone Radiative Forcing

Derived from EESC:

```
RF_stratoz = scale_factor * max(0, (EESC - EESC_reference) / 100)^exponent
```

Only calculated after a threshold year (default 1980). Uses an exponential parameterization that requires updating for very high chlorine/bromine loading.

**Source:** `deltaq_calculations.f90` lines 954-1002

## 3. State Variables

### 3.1 F-Gas Species (Default: 23 species)

| Index | Species | Formula | tau_tot (yr) | tau_OH (yr) | tau_strat (yr) | Radeff (W/m2/ppb) | Cl | Br | F | C |
|-------|---------|---------|--------------|-------------|----------------|-------------------|----|----|---|---|
| 1 | CF4 | CF4 | 50000 | 0 | 0 | 0.09 | 0 | 0 | 4 | 1 |
| 2 | C2F6 | C2F6 | 10000 | 0 | 0 | 0.25 | 0 | 0 | 6 | 2 |
| 3 | C3F8 | C3F8 | 2600 | 0 | 0 | 0.28 | 0 | 0 | 8 | 3 |
| 4 | C4F10 | C4F10 | 2600 | 0 | 0 | 0.36 | 0 | 0 | 10 | 4 |
| 5 | C5F12 | n-C5F12 | 4100 | 0 | 0 | 0.41 | 0 | 0 | 12 | 5 |
| 6 | C6F14 | n-C6F14 | 3100 | 0 | 0 | 0.44 | 0 | 0 | 14 | 6 |
| 7 | C7F16 | n-C7F16 | 3000 | 0 | 0 | 0.5 | 0 | 0 | 16 | 7 |
| 8 | C8F18 | C8F18 | 3000 | 0 | 0 | 0.55 | 0 | 0 | 18 | 8 |
| 9 | CC4F8 | c-C4F8 | 3200 | 0 | 0 | 0.32 | 0 | 0 | 8 | 4 |
| 10 | HFC23 | CHF3 | 228 | 243 | 4420 | 0.18 | 0 | 0 | 3 | 1 |
| 11 | HFC32 | CH2F2 | 5.4 | 5.5 | 124 | 0.11 | 0 | 0 | 2 | 1 |
| 12 | HFC4310 | CF3CHFCHFCF2CF3 | 17 | 17.9 | 365 | 0.359 | 0 | 0 | 10 | 5 |
| 13 | HFC125 | CHF2CF3 | 31 | 32 | 351 | 0.23 | 0 | 0 | 5 | 2 |
| 14 | HFC134A | CH2FCF3 | 14 | 14.1 | 267 | 0.16 | 0 | 0 | 4 | 2 |
| 15 | HFC143A | CH3CF3 | 51 | 57 | 612 | 0.16 | 0 | 0 | 3 | 2 |
| 16 | HFC152A | CH3CHF2 | 1.6 | 1.55 | 39 | 0.1 | 0 | 0 | 2 | 2 |
| 17 | HFC227EA | CF3CHFCF3 | 36 | 37.5 | 673 | 0.26 | 0 | 0 | 7 | 3 |
| 18 | HFC236FA | CF3CH2CF3 | 213 | 253 | 1350 | 0.24 | 0 | 0 | 6 | 3 |
| 19 | HFC245FA | CHF2CH2CF3 | 7.9 | 8.2 | 149 | 0.24 | 0 | 0 | 5 | 3 |
| 20 | HFC365MFC | CH3CF2CH2CF3 | 8.9 | 9.3 | 190 | 0.22 | 0 | 0 | 5 | 4 |
| 21 | NF3 | NF3 | 569 | 0 | 740 | 0.2 | 0 | 0 | 3 | 0 |
| 22 | SF6 | SF6 | 850 | 0 | 0 | 0.57 | 0 | 0 | 6 | 0 |
| 23 | SO2F2 | SO2F2 | 36 | 300 | 630 | 0.2 | 0 | 0 | 2 | 0 |

**Note:** F-gases have `FGAS_FRACT_RELEASEFACTOR = 0` for all species, meaning they do not contribute to EESC (no ozone depletion).

### 3.2 Montreal Protocol Species (Default: 18 species)

| Index | Species | Formula | tau_tot (yr) | tau_OH (yr) | tau_strat (yr) | Radeff (W/m2/ppb) | Cl | Br | f_release |
|-------|---------|---------|--------------|-------------|----------------|-------------------|----|----|-----------|
| 1 | CFC11 | CCl3F | 52 | 0 | 55 | 0.295 | 3 | 0 | 0.47 |
| 2 | CFC12 | CCl2F2 | 102 | 0 | 103 | 0.364 | 2 | 0 | 0.23 |
| 3 | CFC113 | CCl2FCClF2 | 93 | 0 | 94.5 | 0.3 | 3 | 0 | 0.29 |
| 4 | CFC114 | CClF2CClF2 | 189 | 0 | 191 | 0.31 | 2 | 0 | 0.12 |
| 5 | CFC115 | CClF2CF3 | 540 | 0 | 664 | 0.2 | 1 | 0 | 0.04 |
| 6 | HCFC22 | CHClF2 | 11.9 | 13 | 161 | 0.21 | 1 | 0 | 0.13 |
| 7 | HCFC141B | CH3CCl2F | 9.4 | 10.7 | 72.3 | 0.16 | 2 | 0 | 0.34 |
| 8 | HCFC142B | CH3CClF2 | 18 | 19.3 | 212 | 0.19 | 1 | 0 | 0.17 |
| 9 | CH3CCL3 | CH3CCl3 | 5 | 6.1 | 38 | 0.07 | 3 | 0 | 0.67 |
| 10 | CCL4 | CCl4 | 32 | 0 | 44 | 0.174 | 4 | 0 | 0.56 |
| 11 | CH3CL | CH3Cl | 0.9 | 1.57 | 30.4 | 0.004 | 1 | 0 | 0.44 |
| 12 | CH2CL2 | CH2Cl2 | 0.5 | 0.5 | 0 | 0.028 | 2 | 0 | 0 |
| 13 | CHCL3 | CHCl3 | 0.5 | 0.5 | 0 | 0.07 | 3 | 0 | 0 |
| 14 | CH3BR | CH3Br | 0.8 | 1.8 | 26.3 | 0.004 | 0 | 1 | 0.6 |
| 15 | HALON1211 | CBrClF2 | 16 | 0 | 41 | 0.29 | 1 | 1 | 0.62 |
| 16 | HALON1301 | CBrF3 | 72 | 0 | 73.5 | 0.3 | 0 | 1 | 0.28 |
| 17 | HALON2402 | CBrF2CBrF2 | 28 | 0 | 41 | 0.31 | 0 | 2 | 0.65 |
| 18 | HALON1202 | CBr2F2 | 2.5 | 0 | 36 | 0.27 | 0 | 2 | 0.62 |

### 3.3 Data Store Variables per Species

For each F-gas and MHALO species, the following DataStore arrays are tracked:

| Variable | Fortran Array | Units | Description |
|----------|---------------|-------|-------------|
| Emissions | `DAT_FGAS_EMIS(i)`, `DAT_MHALO_EMIS(i)` | kt/yr | Input or scenario emissions |
| Concentration | `DAT_FGAS_CONC(i)`, `DAT_MHALO_CONC(i)` | ppt | Atmospheric concentration |
| Radiative Forcing | `DAT_FGAS_RF(i)`, `DAT_MHALO_RF(i)` | W/m^2 | Instantaneous RF |
| Effective RF | `DAT_FGAS_EFFRF(i)`, `DAT_MHALO_EFFRF(i)` | W/m^2 | After efficacy scaling |
| ERF | `DAT_FGAS_ERF(i)`, `DAT_MHALO_ERF(i)` | W/m^2 | Effective radiative forcing |
| Total Lifetime | `DAT_FGAS_TAUTOT(i)`, `DAT_MHALO_TAUTOT(i)` | years | Time-varying total lifetime |
| OH Lifetime | `DAT_FGAS_TAUOH(i)`, `DAT_MHALO_TAUOH(i)` | years | Time-varying OH lifetime |
| Strat Lifetime | `DAT_FGAS_TAUSTRAT(i)`, `DAT_MHALO_TAUSTRAT(i)` | years | Time-varying strat lifetime |
| Inverse Emissions | `DAT_FGAS_INVERSEEMIS(i)`, `DAT_MHALO_INVERSEEMIS(i)` | kt/yr | Derived from conc change |
| GWP Emissions | `DAT_FGAS_GWPEMIS(i)`, `DAT_MHALO_GWPEMIS(i)` | GtCO2eq/yr | GWP-weighted emissions |

### 3.4 Aggregate Variables

| Variable | Fortran Name | Units | Description |
|----------|--------------|-------|-------------|
| Total F-gas RF | `DAT_FGASSUM_RF` | W/m^2 | Sum of all F-gas RF |
| Total MHALO RF | `DAT_MHALOSUM_RF` | W/m^2 | Sum of all MHALO RF |
| Total Halo RF | `DAT_HALOSUM_RF` | W/m^2 | F-gas + MHALO combined |
| EESC | `DAT_EESC_CONC` | ppt | Equivalent Effective Stratospheric Chlorine |
| ESC | `DAT_ESC_CONC` | ppt | Effective Stratospheric Chlorine |
| ESBr | `DAT_ESBR_CONC` | ppt | Effective Stratospheric Bromine |
| CFC12-eq concentration | `DAT_MHALOSUMCFC12EQ_CONC` | ppt | MHALO RF as CFC-12 equivalent |
| HFC134a-eq concentration | `DAT_FGASSUMHFC134AEQ_CONC` | ppt | F-gas RF as HFC-134a equivalent |

## 4. Parameters

### 4.1 Species-Specific Parameters (Arrays)

| Parameter | Fortran Name | Units | Description |
|-----------|--------------|-------|-------------|
| Number of species | `FGAS_N`, `MHALO_N` | - | Runtime configurable (default 23, 18) |
| Species names | `FGAS_NAMES`, `MHALO_NAMES` | - | Character array |
| Total lifetime | `FGAS_TAU_TOT`, `MHALO_TAU_TOT` | years | Base total lifetime |
| OH lifetime | `FGAS_TAU_OH`, `MHALO_TAU_OH` | years | OH-reaction lifetime |
| Stratospheric lifetime | `FGAS_TAU_STRAT`, `MHALO_TAU_STRAT` | years | Stratospheric loss lifetime |
| Radiative efficiency | `FGAS_RADIAT_EFF`, `MHALO_RADIAT_EFF` | W/m^2/ppb | Linear RF per concentration |
| Release factor | `FGAS_FRACT_RELEASEFACTOR`, `MHALO_FRACT_RELEASEFACTOR` | 0-1 | Fraction released in stratosphere |
| Effective mix box | `FGAS_EFF_MIXBOXSIZE`, `MHALO_EFF_MIXBOXSIZE` | 0-1 | Mixing fraction (default 0.949) |
| GWP values | `FGAS_METRICVALS`, `MHALO_METRICVALS` | - | 100-year GWP |
| Atomic composition | `*_CL_ATOMS`, `*_BR_ATOMS`, `*_F_ATOMS`, etc. | count | Atoms per molecule |

### 4.2 Global Parameters

| Parameter | Fortran Name | Default | Units | Description |
|-----------|--------------|---------|-------|-------------|
| Air molar mass | `GEN_AIR_GRAMMPROMOL` | 28.984 | g/mol | Molecular weight of air |
| Atmospheric mass | `GEN_ATM_TOTMASS_1E21GRAMM` | 5.133 | 10^21 g | Total atmospheric mass |
| EESC delay | `GEN_EESC_STRATMIXDELAY` | 3 | years | Troposphere-stratosphere mixing delay |
| Br vs Cl efficiency | `STRATOZ_BR_VS_CL_SCALE` | ~60 | - | Bromine ozone destruction factor |
| CFC-11 release factor | `MHALO_RELEASEFACTORC11` | 0.75 | - | CFC-11 normalization |
| Meridional flux change | `GEN_MERIDFLUX_CHNGPERDEG` | 0.15 | 1/K | Flux change per degree warming |
| Flux change year | `GEN_CHANGE_MERIDIONALFLUX_YR` | 1980 | year | Reference year for flux |

### 4.3 Lifetime Variability Parameters

| Parameter | Fortran Name | Default | Description |
|-----------|--------------|---------|-------------|
| Use variable OH tau | `FGAS_USE_TAUOH_VAR`, `MHALO_USE_TAUOH_VAR` | 1 | Enable OH lifetime scaling |
| Use variable strat tau | `FGAS_USE_TAUSTRAT_VAR`, `MHALO_USE_TAUSTRAT_VAR` | 1 | Enable strat lifetime scaling |
| Strat tau sensitivity | `FGAS_TAUSTRAT_SENS2MERIDFLUX`, `MHALO_TAUSTRAT_SENS2MERIDFLUX` | 0.3 | Strat lifetime sensitivity |

### 4.4 Mode Switches

| Parameter | Fortran Name | Default | Description |
|-----------|--------------|---------|-------------|
| Conc to emis switch year | `FGAS_SWITCHFROMCONC2EMIS_YEAR`, `MHALO_SWITCHFROMCONC2EMIS_YEAR` | 2015 | Year to switch from prescribed conc to calculated |
| RF scaling (lower tau) | `FGAS_RF_REGIONSCALING_LOWERTAU`, `MHALO_RF_BOXSCALE_LOWERTAU` | 1.0 | Tau threshold for regional scaling |
| RF scaling (upper tau) | `FGAS_RF_REGIONSCALING_UPPERTAU`, `MHALO_RF_BOXSCALE_UPPERTAU` | 8.0 | Tau threshold for well-mixed |

## 5. Inputs (per timestep)

| Variable | Units | Source | Required? | Fortran Variable |
|----------|-------|--------|-----------|------------------|
| Species emissions | kt/yr | Scenario file or external | Yes | `DAT_FGAS_EMIS(i)%DATGLOBE`, `DAT_MHALO_EMIS(i)%DATGLOBE` |
| Species concentrations | ppt | Concentration file (before switch year) | Conditional | `DAT_FGAS_CONC(i)%DATGLOBE`, `DAT_MHALO_CONC(i)%DATGLOBE` |
| CH4 OH lifetime | years | CH4 module | If tau_OH variable | `CH4_TAUOH_EFFECTIVE` |
| Temperature for meridional flux | K | Climate module | If tau_strat variable | `TEMP_MERIDIONALFLUX` |
| CO2I emissions pattern | kt/yr | CO2 module | For RF box fractions | `DAT_CO2I_EMIS` |

## 6. Outputs (per timestep)

| Variable | Units | Destination | Fortran Variable |
|----------|-------|-------------|------------------|
| Species concentrations | ppt | Output, internal | `DAT_FGAS_CONC(i)`, `DAT_MHALO_CONC(i)` |
| Species RF | W/m^2 | RF aggregation | `DAT_FGAS_RF(i)`, `DAT_MHALO_RF(i)` |
| Sum RF (F-gases) | W/m^2 | Total RF | `DAT_FGASSUM_RF` |
| Sum RF (MHALO) | W/m^2 | Total RF | `DAT_MHALOSUM_RF` |
| EESC loading | ppt | Stratospheric ozone RF | `DAT_EESC_CONC` |
| ESC loading | ppt | Diagnostics | `DAT_ESC_CONC` |
| ESBr loading | ppt | Diagnostics | `DAT_ESBR_CONC` |
| Stratospheric ozone RF | W/m^2 | Total RF | `DAT_STRATOZ_RF` |
| Inverse emissions | kt/yr | Diagnostics | `DAT_FGAS_INVERSEEMIS(i)`, `DAT_MHALO_INVERSEEMIS(i)` |

## 7. Algorithm (Pseudocode)

```
SUBROUTINE PROCESS_HALOCARBONS(current_year_idx)

    ! Initialize EESC for this timestep
    EESC(next_year) = 0
    ESC(next_year) = 0
    ESBr(next_year) = 0

    ! Calculate delayed EESC year (typically 3 years lag)
    delayed_year_idx = MAX(1, next_year_idx - EESC_DELAY)

    ! =========== F-GASES (Kyoto) ===========

    ! Calculate lifetime scale factors
    IF (use_variable_tau_OH) THEN
        scale_OH = CH4_tau_OH(t) / CH4_tau_OH_init
    ELSE
        scale_OH = 1.0
    ENDIF

    IF (use_variable_tau_strat) THEN
        scale_strat = 1 / (1 + T_meridional * meridional_sensitivity)
    ELSE
        scale_strat = 1.0
    ENDIF

    ! Scale radiative efficiencies on first timestep
    IF (current_year_idx == 1) THEN
        fgas_radeff = fgas_radeff * rf_fgassum_scale
    ENDIF

    FOR each FGAS species i = 1 to FGAS_N:

        ! Calculate effective total lifetime
        tau_eff(i) = calculate_halo_tautot(
            tau_strat(i), scale_strat,
            tau_OH(i), scale_OH,
            tau_other(i)
        )

        ! Store individual lifetime components
        Store tau_OH(i) * scale_OH
        Store tau_strat(i) * scale_strat

        ! Calculate concentration if past switch year
        IF (year >= SWITCHFROMCONC2EMIS_YEAR) THEN
            C(i, t+1) = calculate_halo_conc(
                C(i, t), E(i, t), tau_eff(i),
                air_molar_mass, atm_mass, mol_mass(i), mix_box(i)
            )
        ENDIF

        ! Handle first year initialization (questionable logic - see issues)
        IF (current_year_idx == 1) THEN
            C(i, 1) = C(i, 2)
        ENDIF

        ! Calculate radiative forcing
        delta_C = C(i, t+1) - C_preindustrial(i)
        RF(i, t+1) = delta_C * radeff(i) / 1000  ! W/m^2

        ! Calculate regional RF fractions
        RF_box_fractions = calculate_halo_rf_box_fractions(...)

        ! Store RF with box distribution
        RF_global(i, t+1) = RF(i, t+1)
        RF_box(i, t+1, :) = RF(i, t+1) * RF_box_fractions

        ! Handle first year offset
        IF (current_year_idx == 1) THEN
            set_rf_first_year_forcing_and_offset(...)
        ENDIF

        ! Apply constant-after-year caps
        ensure_constant_after_year(RF(i), rf_fgas_constantafteryr)
        ensure_constant_after_year(RF(i), rf_total_constantafteryr)

        ! Accumulate to sum
        FGASSUM_RF(t+1) += RF(i, t+1)

        ! Add to EESC (F-gases typically have zero contribution)
        add_stratospheric_loading_contributions(
            C(i, delayed_year),
            n_Cl(i), n_Br(i),
            release_factor(i),
            -> EESC, ESC, ESBr
        )

        ! Calculate inverse emissions
        inverse_emis(i, t) = calculate_halo_inverse_emis(...)

    END FOR

    ! Calculate HFC-134a equivalent concentration
    FGASSUM_HFC134AEQ = FGASSUM_RF / radeff(HFC134A)

    ! =========== MONTREAL PROTOCOL GASES ===========

    ! [Same structure as F-gases with MHALO arrays]

    FOR each MHALO species i = 1 to MHALO_N:
        ! Same calculations as F-gases
        ! Key difference: MHALO species have non-zero release factors
        ! so they contribute to EESC
    END FOR

    ! Calculate CFC-12 equivalent concentration
    MHALOSUM_CFC12EQ = MHALOSUM_RF / radeff(CFC12)

    ! Total halocarbon RF
    HALOSUM_RF = FGASSUM_RF + MHALOSUM_RF

    ! =========== STRATOSPHERIC OZONE RF ===========

    RF_stratoz = calculate_stratospheric_ozone_rf(
        EESC(t+1),
        EESC(threshold_year),
        exponent=STRATOZ_CLEXPON,
        scale=STRATOZ_O3SCALE,
        year, threshold_year
    )

    ! Distribute stratospheric ozone RF to boxes
    RF_stratoz_box = RF_stratoz * normalized(RF_REGIONS_STRATOZ)

END SUBROUTINE
```

## 8. Numerical Considerations

### 8.1 Stability

The numerical scheme is unconditionally stable:

- Long-lived species (tau >= 5 yr): Crank-Nicolson implicit is stable for any timestep
- Short-lived species (tau < 5 yr): Exact exponential solution is analytically stable

### 8.2 Accuracy Concerns

1. **5-year threshold**: The switch between numerical schemes at tau=5 years is somewhat arbitrary. Both methods should give similar results, but there may be small discontinuities for species near this threshold.

2. **Annual timestep limitation**: For very short-lived species (tau < 1 year), the annual timestep may introduce errors. Species like CH2Cl2 (tau=0.5 yr) may not be accurately represented.

3. **First year initialization**: The code sets `C(1) = C(2)` on the first timestep, which is questionable and marked with a TODO in the source.

### 8.3 Known Bugs

1. **HFC-134a index hardcoded**: At line 4571, the code has:

   ```fortran
   ! TODO - fix bug: HFC134A is not index 8
   hfc134a_idx = 8
   ```

   This overrides the dynamic lookup and assumes HFC-134a is always at index 8 in `FGAS_NAMES`.

2. **Dimensional inconsistency in RF box fractions**: The `calculate_halo_rf_box_fractions` function has a TODO noting dimensional inconsistency in the interpolation between high and low tau regimes (lines 746-752).

## 9. Issues and Concerns

### 9.1 Species List Configuration

**Strength:** The species list is fully configurable via namelist parameters. `FGAS_N` and `MHALO_N` set the number of species, and all species properties are arrays read from configuration.

**Weakness:** While the number of species is runtime-configurable, the array sizes are fixed at compile time (`FGAS_MAXNGASES = 30`, `MHALO_MAXNGASES = 30`). Adding more than 30 species in either category would require code modification.

### 9.2 Modularity Assessment

**Adding species**: Relatively straightforward - add entries to all parameter arrays in the configuration file.

**Removing species**: Reduce `FGAS_N` or `MHALO_N` - species beyond this count are ignored.

**Changing species order**: Dangerous - indices are hardcoded in some places (e.g., HFC-134a index bug mentioned above).

### 9.3 Parallel Processing Issues

Each species is processed independently in a loop. The loops could potentially be parallelized, but:

- The EESC accumulation uses a shared variable that would need atomic operations
- DataStore writes to `*SUM_RF` variables are accumulating

### 9.4 Code Organization Concerns

1. **Scattered implementation**: The halocarbon calculations are embedded in a very large (~12000 line) subroutine in `MAGICC7.f90` rather than being factored into a separate module.

2. **Repeated code**: The F-gas and MHALO processing loops are nearly identical (~200 lines each), differing only in array names and a few parameters. This could be refactored into a single parameterized subroutine.

3. **Helper functions well-factored**: The actual physics calculations (`calculate_halo_conc`, `calculate_halo_rf`, `add_stratospheric_loading_contributions`, etc.) are properly factored into `deltaq_calculations.f90`.

### 9.5 Scientific Limitations

1. **Linear RF assumption**: Uses simple linear `RF = C * radeff` without saturation effects. Valid for current concentrations but may be inaccurate for very high future concentrations.

2. **EESC parameterization**: The comment in `calculate_stratospheric_ozone_rf` notes: "This parameterisation needs to be updated because it is not valid for very high chlorine or bromine loading."

3. **No bank dynamics**: The module does not explicitly model equipment banks (refrigerators, fire extinguishers, etc.) that can release halocarbons. Bank dynamics are implicit in the emissions input.

4. **No natural sources**: Natural sources of halocarbons (e.g., oceanic CH3Cl, CH3Br) must be included in emissions input - not internally calculated.

## 10. Test Cases

### 10.1 Unit Tests for Concentration Calculation

```python
def test_calculate_halo_conc_long_lived():
    """Test Crank-Nicolson scheme for long-lived species (tau >= 5 yr)"""
    # CF4: tau = 50000 years
    # Starting from equilibrium, zero emissions should maintain concentration
    C_t = 80.0  # ppt (approximate current)
    E = 0.0  # kt/yr
    tau = 50000.0
    air_molar = 28.984
    atm_mass = 5.133e21
    mol_mass = 88.0  # CF4
    mix_box = 0.949

    C_next = calculate_halo_conc(C_t, E, tau, air_molar, atm_mass, mol_mass, mix_box)

    # With zero emissions and very long lifetime, concentration should barely change
    assert abs(C_next - C_t) / C_t < 1e-4

def test_calculate_halo_conc_short_lived():
    """Test exponential scheme for short-lived species (tau < 5 yr)"""
    # HFC-152a: tau = 1.6 years
    C_t = 10.0  # ppt
    E = 0.0  # kt/yr
    tau = 1.6
    # ... parameters ...

    C_next = calculate_halo_conc(C_t, E, tau, ...)

    # Should decay by exp(-1/1.6) ~ 0.535 in one year
    expected = C_t * exp(-1/tau)
    assert abs(C_next - expected) < 0.01

def test_equilibrium_concentration():
    """Verify steady-state: C_eq = E * tau * conv_factor"""
    E = 10.0  # kt/yr constant emissions
    tau = 10.0  # years
    # ... run until equilibrium ...
    # C_eq should satisfy dC/dt = 0 -> C_eq = E * tau * conv
```

### 10.2 EESC Calculation Tests

```python
def test_eesc_cfc11_contribution():
    """CFC-11 contributes 3 Cl atoms with 47% release factor"""
    C_cfc11 = 200.0  # ppt
    n_Cl = 3
    n_Br = 0
    release_factor = 0.47
    cfc11_norm = 0.75
    br_vs_cl = 60.0

    ESC_contrib = C_cfc11 * n_Cl * release_factor * cfc11_norm
    expected_ESC = 200 * 3 * 0.47 * 0.75  # = 211.5 ppt

    assert ESC_contrib == pytest.approx(expected_ESC)

def test_eesc_halon1301_contribution():
    """Halon-1301 contributes 1 Br atom scaled by alpha_Br"""
    C_halon1301 = 3.0  # ppt
    n_Cl = 0
    n_Br = 1
    release_factor = 0.28
    cfc11_norm = 0.75
    br_vs_cl = 60.0

    EESC_contrib = C_halon1301 * br_vs_cl * n_Br * release_factor * cfc11_norm
    expected = 3.0 * 60 * 1 * 0.28 * 0.75  # = 37.8 ppt Cl-equivalent

    assert EESC_contrib == pytest.approx(expected)

def test_eesc_delay():
    """EESC should use concentration from 3 years prior"""
    # Simulate concentration spike at year 2000
    # EESC should show effect starting year 2003
    pass  # Implementation test
```

### 10.3 Radiative Forcing Tests

```python
def test_rf_linear_scaling():
    """RF should scale linearly with concentration above pre-industrial"""
    C_pi = 0.0  # Pre-industrial CFC-11
    C_current = 200.0  # ppt
    radeff = 0.295  # W/m^2 per ppb

    RF = (C_current - C_pi) * radeff / 1000  # Convert ppb to ppt

    expected = 200 * 0.295 / 1000  # = 0.059 W/m^2
    assert RF == pytest.approx(expected)

def test_rf_preindustrial_zero():
    """RF should be zero at pre-industrial concentration"""
    C = C_pi = 50.0  # Both at pre-industrial
    radeff = 0.16

    RF = (C - C_pi) * radeff / 1000
    assert RF == 0.0
```

### 10.4 Molecular Mass Tests

```python
def test_molmass_cfc11():
    """CFC-11 (CCl3F) = 12 + 3*35.453 + 18.998 = 137.36"""
    molmass = calc_halos_molmass(
        atoms_cl=3, atoms_br=0, atoms_f=1, atoms_h=0,
        atoms_c=1, atoms_s=0, atoms_o=0, atoms_n=0
    )
    expected = 12.001 + 3*35.453 + 18.9984
    assert molmass == pytest.approx(expected, rel=1e-3)

def test_molmass_sf6():
    """SF6 = 32.06 + 6*18.998 = 146.05"""
    molmass = calc_halos_molmass(
        atoms_cl=0, atoms_br=0, atoms_f=6, atoms_h=0,
        atoms_c=0, atoms_s=1, atoms_o=0, atoms_n=0
    )
    expected = 32.06 + 6*18.9984
    assert molmass == pytest.approx(expected, rel=1e-3)
```

### 10.5 Integration Tests

```python
def test_historical_cfc_concentrations():
    """CFC-11 and CFC-12 should match WMO observations within 5%"""
    # Run model from 1950 to 2020 with historical emissions
    # Compare against WMO/NOAA observation network data
    pass

def test_eesc_peak_timing():
    """EESC should peak around 1997-2000"""
    # Run with Montreal Protocol scenario
    # Verify EESC peak is in late 1990s
    pass

def test_stratoz_rf_scaling():
    """Stratospheric ozone RF should track EESC with ~3 year lag"""
    pass
```

## 11. Fortran Code References

### 11.1 Core Module Definition

- **File:** `/Users/jared/code/magicc/magicc/src/libmagicc/core.f90`
- **Lines 100-131:** `MOD_HALOS` module with `CALC_HALOS_MOLMASS` subroutine

### 11.2 DataStore Definitions

- **File:** `/Users/jared/code/magicc/magicc/src/libmagicc/utils/datastore.f90`
- **Lines 130-196:** FGAS and MHALO array declarations

### 11.3 Physics Calculations

- **File:** `/Users/jared/code/magicc/magicc/src/libmagicc/physics/deltaq_calculations.f90`
- **Lines 430-453:** `calculate_halo_tautot` - effective lifetime
- **Lines 455-490:** `calculate_halo_kton_per_yr_to_ppt_conv_factor` - unit conversion
- **Lines 492-536:** `calculate_halo_conc` - concentration evolution
- **Lines 538-581:** `calculate_halo_inverse_emis` - inverse emissions
- **Lines 583-594:** `calculate_halo_rf` - radiative forcing
- **Lines 596-646:** `calculate_halo_eq_conc_*` - equivalent concentration
- **Lines 648-756:** `calculate_halo_rf_box_fractions` - regional RF distribution
- **Lines 902-951:** `add_stratospheric_loading_contributions` - EESC calculation
- **Lines 954-1002:** `calculate_stratospheric_ozone_rf` - ozone depletion RF

### 11.4 Main Processing Loop

- **File:** `/Users/jared/code/magicc/magicc/src/libmagicc/MAGICC7.f90`
- **Lines 4418-4427:** EESC initialization and delay setup
- **Lines 4429-4565:** F-gas processing loop
- **Lines 4567-4591:** F-gas equivalent concentration
- **Lines 4592-4718:** Montreal Protocol gas processing loop
- **Lines 4720-4738:** MHALO equivalent concentration and total halo RF
- **Lines 4739-4760:** Stratospheric ozone RF calculation

### 11.5 Initialization

- **File:** `/Users/jared/code/magicc/magicc/src/libmagicc/MAGICC7.f90`
- **Lines 2520-2541:** Molecular mass calculation and tau_other derivation for F-gases
- **Lines 2544-2561:** Same for MHALO

### 11.6 Configuration Defaults

- **File:** `/Users/jared/code/magicc/magicc/run/MAGCFG_DEFAULTALL.CFG`
- **Lines 140-167:** FGAS parameters
- **Lines 273-301:** MHALO parameters
- **Lines 258-264:** General halocarbon parameters (GEN_*)

---

## Appendix A: Species Chemical Formulas and Properties

### F-Gases (PFCs, HFCs, Other)

| Species | Chemical Formula | GWP-100 | Category |
|---------|------------------|---------|----------|
| CF4 | CF4 | 7390 | PFC |
| C2F6 | C2F6 | 12200 | PFC |
| C3F8 | C3F8 | 8830 | PFC |
| C4F10 | C4F10 | 8860 | PFC |
| C5F12 | n-C5F12 | 9160 | PFC |
| C6F14 | n-C6F14 | 9300 | PFC |
| C7F16 | n-C7F16 | 0 | PFC |
| C8F18 | C8F18 | 0 | PFC |
| CC4F8 | c-C4F8 | 10300 | PFC |
| HFC23 | CHF3 | 14800 | HFC |
| HFC32 | CH2F2 | 675 | HFC |
| HFC4310 | CF3CHFCHFCF2CF3 | 1640 | HFC |
| HFC125 | CHF2CF3 | 3500 | HFC |
| HFC134A | CH2FCF3 | 1430 | HFC |
| HFC143A | CH3CF3 | 4470 | HFC |
| HFC152A | CH3CHF2 | 124 | HFC |
| HFC227EA | CF3CHFCF3 | 3220 | HFC |
| HFC236FA | CF3CH2CF3 | 9810 | HFC |
| HFC245FA | CHF2CH2CF3 | 1030 | HFC |
| HFC365MFC | CH3CF2CH2CF3 | 794 | HFC |
| NF3 | NF3 | 17200 | Other |
| SF6 | SF6 | 22800 | Other |
| SO2F2 | SO2F2 | 0 | Other |

### Montreal Protocol Gases (CFCs, HCFCs, Halons, Other)

| Species | Chemical Formula | GWP-100 | ODP | Category |
|---------|------------------|---------|-----|----------|
| CFC11 | CCl3F | 4750 | 1.0 | CFC |
| CFC12 | CCl2F2 | 10900 | 1.0 | CFC |
| CFC113 | CCl2FCClF2 | 6130 | 0.8 | CFC |
| CFC114 | CClF2CClF2 | 10000 | 1.0 | CFC |
| CFC115 | CClF2CF3 | 7370 | 0.6 | CFC |
| HCFC22 | CHClF2 | 1810 | 0.055 | HCFC |
| HCFC141B | CH3CCl2F | 725 | 0.11 | HCFC |
| HCFC142B | CH3CClF2 | 2310 | 0.065 | HCFC |
| CH3CCL3 | CH3CCl3 | 146 | 0.1 | Other |
| CCL4 | CCl4 | 1400 | 1.1 | Other |
| CH3CL | CH3Cl | 13 | 0.02 | Other |
| CH2CL2 | CH2Cl2 | 9 | 0 | VSL |
| CHCL3 | CHCl3 | 0 | 0 | VSL |
| CH3BR | CH3Br | 5 | 0.6 | Bromine |
| HALON1211 | CBrClF2 | 1890 | 3.0 | Halon |
| HALON1301 | CBrF3 | 7140 | 10.0 | Halon |
| HALON2402 | CBrF2CBrF2 | 1640 | 6.0 | Halon |
| HALON1202 | CBr2F2 | 0 | 1.7 | Halon |

**VSL** = Very Short-Lived substance
**ODP** = Ozone Depletion Potential (not directly used in MAGICC but context for release factors)

---

## Appendix B: Comparison with Other Models

| Feature | MAGICC | FaIR | Hector |
|---------|--------|------|--------|
| Species count | 23 F-gas + 18 MHALO | Aggregated | Aggregated |
| Lifetime variability | Yes (OH, strat) | Limited | No |
| EESC calculation | Full | Simplified | No |
| Bank dynamics | Via emissions | Via emissions | Via emissions |
| Configurable species | Yes | No | No |
