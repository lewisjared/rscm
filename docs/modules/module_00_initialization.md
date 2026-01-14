# Module 00: Initialization & Data Loading

## 1. Purpose

The initialization module is the foundation of MAGICC, responsible for:

1. **Memory Allocation**: Allocating all arrays and data structures needed for the simulation
2. **Configuration Loading**: Reading parameters from `.CFG` files using Fortran namelists
3. **Input Data Ingestion**: Loading emissions, concentrations, forcings, and other timeseries from input files
4. **Budget Calculations**: Deriving natural emissions to close mass balance budgets for CH4 and N2O
5. **Data Preprocessing**: Merging historical and scenario data, interpolating between years, scaling radiative forcing patterns
6. **Default Initialization**: Setting initial values for state variables and derived quantities

Understanding initialization is critical for reimplementation because:

- It defines how all input data flows into the model
- It establishes the memory layout and indexing conventions
- It performs non-trivial preprocessing that affects downstream calculations
- Configuration parameters cascade through the model in complex ways

## 2. DATASTORE Structure

The `DATASTORE` type is MAGICC's central data container for timeseries data. Nearly all input/output data flows through DATASTORE instances.

### 2.1 Type Definition

```fortran
TYPE DATASTORE
    ! Primary data arrays
    REAL(8), POINTER :: DATBOX(:, :)        ! (NYEARS|NTIMES, 4) - Regional data
    REAL(8), POINTER :: DATGLOBE(:)         ! (NYEARS|NTIMES) - Global mean
    REAL(8), POINTER :: FRACTIONSBOX(:, :)  ! (NYEARS, 4) - Regional fractions
    INTEGER, POINTER :: NEARESTHISTIDX(:)   ! (NYEARS) - Index to nearest historical year
    INTEGER, POINTER :: SCENYEARS(:)        ! (NYEARS) - Years with scenario data

    ! Pre-industrial reference values
    REAL(8) :: PREIND_DATGLOBE              ! Global value at RF_PREIND_REFERENCEYR
    REAL(8) :: PREIND_DATBOX(4)             ! Regional values at RF_PREIND_REFERENCEYR
    INTEGER :: PREIND_EFF_YR                ! Effective pre-industrial year

    ! First-year offset (for forcing normalization)
    REAL(8) :: FIRSTYEAROFFSET_DATGLOBE     ! Global offset subtracted from all years
    REAL(8) :: FIRSTYEAROFFSET_DATBOX(4)    ! Regional offsets

    ! Efficacy normalization values
    REAL(8) :: EFF_NORMYR_BOX(4)            ! Regional forcings at RF_REGIONS_NORMYEAR

    ! Metadata
    INTEGER :: ANNUALSTEPS                   ! 1 for annual, 12 for monthly
    INTEGER :: DATACOLUMNS                   ! 0=none, 1=global, 2=hemispheric, 4=boxes
    INTEGER :: DATAROWS                      ! Number of data rows
    INTEGER :: NULLORAVGORSUM               ! 0=none, 1=average, 2=sum
    INTEGER :: FIRSTYEAR, LASTYEAR          ! Year bounds
    INTEGER :: HISTFIRSTYEAR, HISTLASTYEAR  ! Historical data bounds
    INTEGER :: SCENFIRSTYEAR, SCENLASTYEAR  ! Scenario data bounds
    INTEGER :: N_SCEN_DATALINES             ! Number of scenario data lines

    CHARACTER(LEN=300) :: NOTE, CODE
    CHARACTER(LEN=30)  :: REGIONMODE        ! 'GLOBAL', 'FOURBOX', 'SRES', 'RCP', etc.
    CHARACTER(LEN=100) :: UNITS, DATTYPE
    CHARACTER(LEN=FILENAME_LEN) :: FNAME

    ! Column metadata for sector files
    CHARACTER(LEN=20), POINTER :: COLCODE_GAS(:)
    CHARACTER(LEN=20), POINTER :: COLCODE_UNITS(:)
    CHARACTER(LEN=20), POINTER :: COLCODE_REGION(:)
    CHARACTER(LEN=20), POINTER :: COLCODE_SECTOR_OPERATOR(:)
ENDTYPE DATASTORE
```

### 2.2 Data Flow Through DATASTORE

```
Input File (.SCEN, .IN, etc.)
         |
         v
    datastore_read() / scen7_emis_read()
         |
         +--> DATGLOBE(:)     Global mean timeseries
         |
         +--> DATBOX(:, 1:4)  Regional breakdown:
         |        |
         |        +--> Box 1: NH-Ocean
         |        +--> Box 2: NH-Land
         |        +--> Box 3: SH-Ocean
         |        +--> Box 4: SH-Land
         |
         +--> FRACTIONSBOX(:, 1:4)  Regional fractions (sum to 1.0)
         |
         +--> PREIND_DATGLOBE, PREIND_DATBOX  Reference values
         |
         v
    Physics Modules (read DATGLOBE/DATBOX)
         |
         v
    Output Files
```

### 2.3 Key DATASTORE Instances

| Variable | Type | Description |
|----------|------|-------------|
| `DAT_CO2_CONC` | Concentration | CO2 mixing ratio (ppm) |
| `DAT_CH4_CONC` | Concentration | CH4 mixing ratio (ppb) |
| `DAT_N2O_CONC` | Concentration | N2O mixing ratio (ppb) |
| `DAT_CO2I_EMIS` | Emission | Fossil CO2 emissions (GtC/yr) |
| `DAT_CO2B_EMIS` | Emission | Land-use CO2 emissions (GtC/yr) |
| `DAT_CH4I_EMIS` | Emission | Anthropogenic CH4 emissions (TgCH4/yr) |
| `DAT_SOXI_EMIS` | Emission | Industrial SOx emissions (TgS/yr) |
| `DAT_TOTAL_EFFRF` | Forcing | Total effective radiative forcing (W/m2) |
| `DAT_SURFACE_TEMP` | Temperature | Surface temperature anomaly (K) |
| `DAT_VOLCANIC_RF` | Forcing | Volcanic radiative forcing (W/m2) - monthly |
| `DAT_FGAS_EMIS(i)` | Emission | F-gas emissions array (kt/yr) |
| `DAT_MHALO_EMIS(i)` | Emission | Montreal halocarbon emissions array (kt/yr) |

## 3. Year Indexing Convention

### 3.1 Critical: Start-of-Year vs Mid-Year Convention

**ALL timeseries in MAGICC are START-OF-YEAR values, EXCEPT emissions which are MID-YEAR values.**

This means:

- `DAT_CO2_CONC%DATGLOBE(i)` = CO2 concentration at January 1st of year `STARTYEAR + i - 1`
- `DAT_SURFACE_TEMP%DATGLOBE(i)` = Temperature at January 1st of year `STARTYEAR + i - 1`
- `DAT_CO2I_EMIS%DATGLOBE(i)` = CO2 emissions averaged over year `STARTYEAR + i - 1` (centered at mid-year)

### 3.2 Index Calculations

```fortran
! Key variables from mod_years
STARTYEAR = 1750        ! First year of simulation
ENDYEAR = 2500          ! Last year of simulation
NYEARS = ENDYEAR - STARTYEAR + 1    ! = 751 years
STEPSPERYEAR = 12       ! Monthly timesteps
NTIMES = NYEARS * STEPSPERYEAR      ! = 9012 timesteps

! Year to index conversion
yr_idx(year) = year - STARTYEAR + 1

! Index to year conversion
year = ALLYEARS(idx)    ! or equivalently: STARTYEAR + idx - 1

! Sub-annual time indexing
! ALLTIMES_D(t) = year + (step - 1) / STEPSPERYEAR
! For January 1750: ALLTIMES_D(1) = 1750.0
! For February 1750: ALLTIMES_D(2) = 1750.0833...
```

### 3.3 DATASTORE Index Helper Functions

```fortran
! Get year index within a DATASTORE (may differ from global if FIRSTYEAR != STARTYEAR)
idx = datastore_yr(dat_in, year)

! Get global value at a year
value = datastore_get_globe(dat_in, year)

! Get box values at a year
values(1:4) = datastore_get_box(dat_in, year)

! Get box values with interpolation for sub-annual
values(1:4) = datastore_get_box_with_interpolation(dat_in, timepoint)
```

## 4. 4-Box Geography

MAGICC uses a simplified 4-box representation of Earth's surface for regional resolution.

### 4.1 Box Definitions

| Box Index | Name | Region |
|-----------|------|--------|
| 1 | NH-Ocean | Northern Hemisphere Ocean |
| 2 | NH-Land | Northern Hemisphere Land |
| 3 | SH-Ocean | Southern Hemisphere Ocean |
| 4 | SH-Land | Southern Hemisphere Land |

### 4.2 Area Fractions

Area fractions are calculated from configuration parameters:

```fortran
! Land fractions (fraction of GLOBE, not hemisphere)
FGL(1) = CORE_HEMISFRACTION_NH_LAND / 2.0  ! NH land fraction of globe
FGL(2) = CORE_HEMISFRACTION_SH_LAND / 2.0  ! SH land fraction of globe

! Ocean fractions (complement of land in each hemisphere)
FGO(1) = 0.5 - FGL(1)  ! NH ocean fraction of globe
FGO(2) = 0.5 - FGL(2)  ! SH ocean fraction of globe

! Named aliases
FGNL = FGL(1)  ! NH Land
FGSL = FGL(2)  ! SH Land
FGNO = FGO(1)  ! NH Ocean
FGSO = FGO(2)  ! SH Ocean

! Global area fractions array (matches DATBOX indexing)
GLOBALAREAFRACTIONS(1) = FGNO  ! Box 1: NH Ocean
GLOBALAREAFRACTIONS(2) = FGNL  ! Box 2: NH Land
GLOBALAREAFRACTIONS(3) = FGSO  ! Box 3: SH Ocean
GLOBALAREAFRACTIONS(4) = FGSL  ! Box 4: SH Land

! Typical default values:
! CORE_HEMISFRACTION_NH_LAND = 0.391
! CORE_HEMISFRACTION_SH_LAND = 0.195
! -> GLOBALAREAFRACTIONS = [0.3045, 0.1955, 0.4025, 0.0975]
```

### 4.3 Global Mean Calculation

For concentrations/temperatures (area-weighted average):

```fortran
global_mean = SUM(DATBOX(i, :) * GLOBALAREAFRACTIONS)
```

For emissions (simple sum):

```fortran
global_total = SUM(DATBOX(i, :))
```

The `NULLORAVGORSUM` field controls this behavior:

- 0 = No data
- 1 = Average (concentrations, temperatures, forcings)
- 2 = Sum (emissions)

## 5. Initialization Sequence

The `magicc_init_run` subroutine executes these steps in order:

### 5.1 Ocean and Sea Level Tuning Data

```fortran
IF (CORE_SWITCH_OCN_AREAFACTOR == 1) THEN
    CALL READ_ALLCFG(FIND_FILE(FILE_OCN_AREAFACTOR))
END IF
IF (CORE_SWITCH_OCN_TEMPPROFILE == 2) THEN
    CALL READ_ALLCFG(FIND_FILE(FILE_OCN_INIPROFILE))
END IF
CALL READ_ALLCFG(FIND_FILE(FILE_SLR_GL_XTRAPARAMS))
```

### 5.2 Parameter Adjustments

- Temperature feedback switches controlling CO2 fertilization
- CH4 lifetime feedback by NOx/VOC/CO
- Climate sensitivity dependency on RLO
- Heat capacity calculations

### 5.3 Area Fraction Setup

- Calculate FGL, FGO, GLOBALAREAFRACTIONS from configuration

### 5.4 Array Allocation

Major allocations include:

```fortran
! Ocean layers
ALLOCATE(OCN_PRESSURE_PROFILE(CORE_OCN_NLEVELS))
ALLOCATE(OCN_HEMISPHERIC_LAYERTEMPS(2, CORE_OCN_NLEVELS))
ALLOCATE(TEMP_OCEANLAYERS(CORE_OCN_NLEVELS, NYEARS))

! Carbon cycle pools
ALLOCATE(CO2_PLANT_POOL(NYEARS), CO2_SOIL_POOL(NYEARS), ...)

! Methane arrays
ALLOCATE(CH4_TAUOH_EFFECTIVE(NYEARS), CH4_DCDT(NYEARS), ...)

! Cloud/aerosol
ALLOCATE(CLOUD%NI_BCI(NYEARS, 4), CLOUD%NI_BCB(NYEARS, 4), ...)
```

### 5.5 DATASTORE Initialization via datastore_read

`datastore_read(filename, dat, nulloravgorsum)` allocates and populates DATASTORE:

- If filename is empty, allocates arrays initialized to zero
- If filename provided, reads data and interpolates to model year grid
- Sets PREIND values from RF_PREIND_REFERENCEYR

### 5.6 Concentration File Loading

```fortran
CALL datastore_read(FILE_CO2_CONC, DAT_CO2_CONC, 1)  ! Average
CALL datastore_read(FILE_CH4_CONC, DAT_CH4_CONC, 1)
CALL datastore_read(FILE_N2O_CONC, DAT_N2O_CONC, 1)
```

### 5.7 Emission File Loading

```fortran
CALL datastore_read(FILE_CO2I_EMIS, DAT_CO2I_EMIS, 2)  ! Sum
CALL datastore_read(FILE_CH4I_EMIS, DAT_CH4I_EMIS, 2)
CALL datastore_read(FILE_SOXI_EMIS, DAT_SOXI_EMIS, 2)
! ... and many more
```

### 5.8 Radiative Forcing File Loading

```fortran
CALL datastore_read_monthly(FILE_VOLCANIC_RF, DAT_VOLCANIC_RF, 1)  ! Monthly
CALL datastore_read(FILE_SOLAR_RF, DAT_SOLAR_RF, 1)                 ! Annual
CALL datastore_read(FILE_LANDUSE_RF, DAT_LANDUSE_RF, 1)
```

### 5.9 SCEN File Processing

The `.SCEN` (or `.SCEN7`) scenario files are read and merged with historical data:

```fortran
! Determine format (.SCEN vs .SCEN7)
IF (scen_format == 7) THEN
    CALL scen7_emis_read(FILE_EMISSCEN)
ELSEIF (scen_format == 6) THEN
    CALL scen6_emis_read(FILE_EMISSCEN)
END IF

! Process additional SCEN files (up to 8)
CALL scen7_emis_read(FILE_EMISSCEN_2)
! ...

! Convert regional data to 4-box
CALL SUM_RAWDATA_TOBOX(RAW_GHGAER_DATA, GHGAER_ALLNGASES)
CALL SUM_RAWDATA_TOBOX(RAW_FGAS_DATA, FGAS_N)
CALL SUM_RAWDATA_TOBOX(RAW_MHALO_DATA, MHALO_N)

! Merge scenario with historical
CALL HANDLE_RAWSCEN_INTO_HISTEMIS('CO2', DAT_CO2I_EMIS, DAT_CO2B_EMIS, ...)
CALL HANDLE_RAWSCEN_INTO_HISTEMIS('CH4', DAT_CH4I_EMIS, DAT_CH4B_EMIS, ...)
! ... for all gases
```

### 5.10 Budget Calculations

For CH4 and N2O, natural emissions are inferred to close the mass balance:

```fortran
! Methane budget (in methane_calc_budget)
CALL methane_calc_budget

! N2O budget
N2O_NATEMISBUDGET = N2O_PPB2TGN * (SUM(N2O_DCDT(...)) + SUM(N2O_CBAR(...)) / N2O_TAUINIT) ...
                   - (anthropogenic emissions)
DAT_N2ON_EMIS%DATGLOBE = DAT_N2ON_EMIS%DATGLOBE + N2O_NATEMISBUDGET
```

### 5.11 Aerosol Forcing Preprocessing

Extensive preprocessing scales and extrapolates aerosol forcing:

```fortran
! Scale sulfate forcing from optical thickness to W/m2
CALL EXTRAP_RF_WITH_EMIS(RF_SOXI_DIR_APPLY, RF_SOXI_DIR_FACTOR, ...)

! Calculate nitrate aerosol forcing from emissions
RF_NO3_ALPHAFACTOR(:) = (NOx * NH3) / (1 + RF_NO3_LAMBDASO2 * SOx / NH3)
```

### 5.12 F-gas and Halocarbon Setup

```fortran
DO FGAS_I = 1, FGAS_N
    CALL CALC_HALOS_MOLMASS(...)  ! Calculate molecular mass
    ! Calculate TAU_OTHER from TAU_TOT, TAU_OH, TAU_STRAT
END DO

DO MHALO_I = 1, MHALO_N
    CALL CALC_HALOS_MOLMASS(...)
END DO
```

### 5.13 Permafrost and Nitrogen Cycle Initialization

```fortran
CALL permafrost_alloc
CALL permafrost_init
CALL nitrogen_allocate
CALL nitrogen_init
```

## 6. Configuration Loading

### 6.1 Configuration File Hierarchy

```
magicc_read_config()
    |
    +-> MAGCFG_DEFAULTALL.CFG     Base defaults for all parameters
    |
    +-> MAGCFG_USER.CFG           User overrides
    |
    +-> FILE_TUNINGMODEL_1..10    Model-specific tuning (e.g., MAGTUNE_CMIP6.CFG)
    |
    +-> FILE_SLR_*.CFG            Sea level rise parameters
```

### 6.2 Configuration Namelist (NML_ALLCFGS)

The `NML_ALLCFGS` namelist in `mod_allcfgs.f90` defines hundreds of parameters organized by category:

| Category | Example Parameters |
|----------|-------------------|
| Run control | `RUNNAME`, `FILE_EMISSCEN`, `PATHNAME_INFILES` |
| CO2/Carbon | `CO2_FERTILIZATION_FACTOR`, `OCEANCC_MODEL`, `CO2_PREINDCO2CONC` |
| Methane | `CH4_TAUTOT_INIT`, `CH4_S`, `CH4_ANOX`, `CH4_ACO`, `CH4_AVOC` |
| N2O | `N2O_TAUINIT`, `N2O_RADEFF_WM2PERPPB` |
| Climate | `CORE_CLIMATESENSITIVITY`, `CORE_VERTICALDIFFUSIVITY`, `CORE_RLO` |
| Aerosols | `RF_SOXI_DIR_FACTOR`, `RF_SOXI_DIR_YR`, `RF_SOXI_DIR_WM2` |
| Sea Level | `SLR_GIS_SMB_COEF1`, `SLR_AIS_SID_PARAMETERISATION` |
| Output | `OUT_EMISSIONS`, `OUT_CONCENTRATIONS`, `OUT_TEMPERATURE` |

### 6.3 READ_ALLCFG Subroutine

```fortran
SUBROUTINE READ_ALLCFG(FNAME)
    OPEN(newunit=IUNIT, file=FNAME, status='OLD')
    READ(iunit, nml=NML_ALLCFGS, iostat=ISTAT)
    IF (ISTAT /= 0) THEN
        ! Error handling - report invalid line
    END IF
    CLOSE(IUNIT)
END SUBROUTINE
```

## 7. Key Variables Initialized

### 7.1 Time Domain

| Variable | Module | Description |
|----------|--------|-------------|
| `STARTYEAR` | mod_years | First simulation year (typically 1750) |
| `ENDYEAR` | mod_years | Last simulation year (typically 2500) |
| `NYEARS` | mod_years | Number of years = ENDYEAR - STARTYEAR + 1 |
| `STEPSPERYEAR` | mod_years | Timesteps per year (12 for monthly) |
| `NTIMES` | mod_years | Total timesteps = NYEARS * STEPSPERYEAR |
| `ALLYEARS(:)` | mod_years | Array of all years [STARTYEAR, STARTYEAR+1, ...] |
| `ALLTIMES_D(:)` | mod_years | Decimal time points for sub-annual |

### 7.2 Climate Core

| Variable | Description | Typical Value |
|----------|-------------|---------------|
| `CORE_CLIMATESENSITIVITY` | Equilibrium climate sensitivity (K) | 3.0 |
| `CORE_VERTICALDIFFUSIVITY` | Ocean vertical diffusivity (cm2/s) | 0.55 |
| `CORE_MIXEDLAYER_DEPTH` | Ocean mixed layer depth (m) | 60-100 |
| `CORE_RLO` | Land-ocean warming ratio | 1.3-1.4 |
| `HEAT_CAPACITY_PERM` | Heat capacity (W*yr/m3/K) | Derived |

### 7.3 Carbon Cycle Initial Pools

| Variable | Description | Typical Value (GtC) |
|----------|-------------|---------------------|
| `CO2_PLANTPOOL_INITIAL` | Initial plant carbon pool | 550 |
| `CO2_SOILPOOL_INITIAL` | Initial soil carbon pool | 1500 |
| `CO2_DETRITUSPOOL_INITIAL` | Initial detritus pool | 55 |
| `CO2_NPP_INITIAL` | Initial NPP (GtC/yr) | 60 |
| `CO2_PREINDCO2CONC` | Pre-industrial CO2 (ppm) | 278 |

### 7.4 Gas Lifetimes

| Variable | Description | Typical Value (years) |
|----------|-------------|----------------------|
| `CH4_TAUTOT_INIT` | Initial CH4 total lifetime | 9.9 |
| `CH4_TAUSOIL` | CH4 soil sink lifetime | 150 |
| `CH4_TAUSTRAT` | CH4 stratospheric lifetime | 120 |
| `N2O_TAUINIT` | Initial N2O lifetime | 121 |

### 7.5 Radiative Forcing Parameters

| Variable | Description |
|----------|-------------|
| `RF_PREIND_REFERENCEYR` | Year for pre-industrial reference (e.g., 1750) |
| `RF_INITIALIZATION_METHOD` | 'ZEROSTARTSHIFT' or 'JUMPSTART' |
| `RF_REGIONS_NORMYEAR` | Year for efficacy normalization |

## 8. Deallocation

The `magicc_dealloc` subroutine frees all allocated memory to allow multiple runs in the same process:

### 8.1 Deallocation Sequence

```fortran
SUBROUTINE magicc_dealloc
    ! Check if already allocated
    IF (.not. allocated(ORIG_RF_DATBOX)) THEN
        call logger % error("magicc_dealloc", "Data have not yet been allocated")
        return
    END IF

    ! Ocean arrays
    DEALLOCATE(OCN_PRESSURE_PROFILE, OCN_HEMISPHERIC_LAYERTEMPS, ...)

    ! F-gas/Halocarbon datastores
    DO FGAS_I = 1, FGAS_N
        CALL DATASTORE_DEALLOC(DAT_FGAS_TAUTOT(FGAS_I))
        ! ...
    END DO

    ! Temperature datastores
    CALL DATASTORE_DEALLOC(DAT_SURFACE_TEMP)
    CALL DATASTORE_DEALLOC(DAT_SURFACE_MIXEDLAYERTEMP)
    ! ...

    ! Carbon cycle arrays
    DEALLOCATE(CO2_PLANT_POOL, CO2_SOIL_POOL, ...)

    ! Methane arrays
    DEALLOCATE(CH4_TAUOH_EFFECTIVE, CH4_DCDT, ...)

    ! Sea level
    CALL sealevel_dealloc

    ! Nitrogen limitation
    CALL nitrogen_cleanup

    ! ... many more deallocations
END SUBROUTINE
```

### 8.2 DATASTORE_DEALLOC

```fortran
SUBROUTINE DATASTORE_DEALLOC(DATSTRUC)
    TYPE(DATASTORE), INTENT(INOUT) :: DATSTRUC

    DEALLOCATE(DATSTRUC%DATBOX)
    DEALLOCATE(DATSTRUC%FRACTIONSBOX)
    DEALLOCATE(DATSTRUC%DATGLOBE)
    DEALLOCATE(DATSTRUC%NEARESTHISTIDX)
    DEALLOCATE(DATSTRUC%SCENYEARS)

    ! Reset all metadata to defaults
    DATSTRUC%FIRSTYEAROFFSET_DATGLOBE = 0.0D0
    DATSTRUC%FIRSTYEAROFFSET_DATBOX = 0.0D0
    DATSTRUC%PREIND_DATGLOBE = 0.0D0
    DATSTRUC%PREIND_DATBOX = 0.0D0
    DATSTRUC%ANNUALSTEPS = 1
    DATSTRUC%DATACOLUMNS = 0
    DATSTRUC%NULLORAVGORSUM = 0
    DATSTRUC%FIRSTYEAR = 0
    DATSTRUC%LASTYEAR = 0
    ! ...
END SUBROUTINE
```

## 9. Implementation Notes for Rust

### 9.1 DATASTORE Equivalent

```rust
pub struct DataStore {
    // Primary timeseries data
    pub datbox: Array2<f64>,        // [nyears|ntimes, 4]
    pub datglobe: Array1<f64>,      // [nyears|ntimes]
    pub fractions_box: Array2<f64>, // [nyears, 4]
    pub nearest_hist_idx: Vec<i32>, // [nyears]
    pub scen_years: Vec<i32>,       // [nyears]

    // Pre-industrial references
    pub preind_datglobe: f64,
    pub preind_datbox: [f64; 4],
    pub preind_eff_yr: i32,

    // First-year offsets
    pub firstyear_offset_datglobe: f64,
    pub firstyear_offset_datbox: [f64; 4],

    // Efficacy normalization
    pub eff_normyr_box: [f64; 4],

    // Metadata
    pub annual_steps: i32,          // 1 or 12
    pub data_columns: i32,          // 0, 1, 2, or 4
    pub null_or_avg_or_sum: NullAvgSum,
    pub first_year: i32,
    pub last_year: i32,
    pub hist_first_year: i32,
    pub hist_last_year: i32,
    pub scen_first_year: i32,
    pub scen_last_year: i32,

    pub region_mode: RegionMode,
    pub units: String,
    pub filename: String,
}

#[derive(Clone, Copy)]
pub enum NullAvgSum {
    None = 0,
    Average = 1,
    Sum = 2,
}

#[derive(Clone, Copy)]
pub enum RegionMode {
    Undefined,
    Global,
    FourBox,
    Sres,
    Rcp,
    RcpPlusBunkers,
    RcpPlusSplitBunkers,
}
```

### 9.2 Time Domain

```rust
pub struct TimeDomain {
    pub start_year: i32,
    pub end_year: i32,
    pub nyears: usize,
    pub steps_per_year: i32,
    pub ntimes: usize,
    pub all_years: Vec<i32>,
    pub all_times_d: Vec<f64>,
}

impl TimeDomain {
    pub fn yr_idx(&self, year: i32) -> usize {
        (year - self.start_year) as usize
    }

    pub fn time_idx(&self, year: i32, step: i32) -> usize {
        self.yr_idx(year) * self.steps_per_year as usize + (step - 1) as usize
    }
}
```

### 9.3 Configuration Loading

Consider using `serde` with a custom deserializer for Fortran namelist format, or the `toml`/`json` crate with a converter from the original `.CFG` files.

### 9.4 Initialization Order

The initialization order matters. A suggested approach:

```rust
impl Magicc {
    pub fn new(config_path: &Path) -> Result<Self, Error> {
        // 1. Load base configuration
        let config = Config::load(config_path)?;

        // 2. Initialize time domain
        let time = TimeDomain::new(config.start_year, config.end_year)?;

        // 3. Calculate area fractions
        let areas = AreaFractions::from_config(&config);

        // 4. Allocate all datastores
        let datastores = DataStores::allocate(&time, &config)?;

        // 5. Load input files
        datastores.load_concentrations(&config)?;
        datastores.load_emissions(&config)?;
        datastores.load_forcings(&config)?;
        datastores.load_scenario(&config)?;

        // 6. Process scenario merging
        datastores.merge_historical_and_scenario(&config)?;

        // 7. Run budget calculations
        datastores.calculate_ch4_budget(&config)?;
        datastores.calculate_n2o_budget(&config)?;

        // 8. Initialize physics modules
        // ...

        Ok(Self { config, time, areas, datastores, ... })
    }
}
```

## 10. Source File References

| File | Key Content |
|------|-------------|
| `src/libmagicc/MAGICC7.f90` | `magicc_init_run` (lines 131-2569), `magicc_read_config` (lines 2573-2625), `magicc_dealloc` (lines 12007-12493) |
| `src/libmagicc/allcfgs.f90` | `NML_ALLCFGS` namelist definition, `READ_ALLCFG`, `READ_TUNING` |
| `src/libmagicc/utils/datastore.f90` | `DATASTORE` type definition, `datastore_read`, `DATASTORE_DEALLOC` |
| `src/libmagicc/utils/years.f90` | Time domain setup, `NYEARS`, `NTIMES`, `yr_idx` |
| `src/libmagicc/io/scen_file.f90` | `scen6_emis_read`, `scen7_emis_read`, `scen7_file_read` |
| `src/libmagicc/mod_areas.f90` | Area fraction calculations |
