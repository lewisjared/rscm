## rscm 0.6.0 (2026-11-04)

### ⚠️ Breaking Changes ⚠️

- Upgrade calibration RNG dependencies to rand 0.10, rand_distr 0.6 and rand_chacha 0.10. Rust callers supplying custom RNGs must use the rand 0.10 traits. Seeded sampling sequences may differ from the previous rand 0.8 implementation. ([#96](https://github.com/lewisjared/rscm/pull/96))
- Migrated from unmaintained `bincode` to actively maintained `postcard` for binary serialization in `rscm-calibrate`.

  **Breaking change**: Checkpoint files (`.checkpoint` and `.chain` files) created with previous versions cannot be loaded with this version due to the serialization format change. Users must either:

  - Complete any in-progress MCMC sampling runs before upgrading
  - Discard existing checkpoint files and restart sampling

  ([#113](https://github.com/lewisjared/rscm/pull/113))
- Restructured terrestrial carbon cycle from 4 pools (plant, detritus, soil, humus) to 3 pools (plant, detritus, soil) to match MAGICC7. The `humus_pool_pi`, `humus_temp_sensitivity`, and `frac_soil_to_humus` parameters have been removed. The `enable_fertilization` and `enable_temp_feedback` boolean parameters have been replaced by `fertilization_yrstart` and `tempfeedback_yrstart` year-based controls. ([#131](https://github.com/lewisjared/rscm/pull/131))
- Rejected the unsupported legacy co2_tempfeedback_switch instead of silently using enabled feedback defaults. Configure temperature feedbacks explicitly on each component. ([#132](https://github.com/lewisjared/rscm/pull/132))
- Updated the ODE solver and nalgebra together to keep IVP vector types compatible. Rust callers using nalgebra vectors must use nalgebra 0.34, which requires Rust 1.87 or newer. ([#137](https://github.com/lewisjared/rscm/pull/137))
- Reworked MAGICC component interfaces to match MAGICC7.

  - `CO2Budget` no longer took `Emissions|CO2|Land Use`, because `Carbon Flux|Terrestrial` already included land-use emissions. It took `Emissions|CO2|CH4 Oxidation` instead.
  - `OceanCarbon` took `Surface Temperature`, fossil emissions, CH4 oxidation and the land flux instead of `Sea Surface Temperature`, so that atmospheric CO2 could evolve within the year. `pco2_pi` defaulted to the initial ocean pCO2.
  - `CH4Chemistry` replaced `ch4_pi`, `tau_oh` and the reference emissions with `feedback_start_year`, `tau_total_init` and `mixbox_size`. Its defaults changed to the MAGICC7 values.
  - `N2OChemistry` replaced `n2o_pi` with `feedback_start_year` and required `Surface Temperature` for the meridional flux feedback.
  - `CH4Chemistry`, `N2OChemistry` and `OceanCarbon` rejected non-annual timesteps.
  - `run_magicc` defaulted the climate `rf_2xco2` to the CO2-doubling ERF of the forcing configuration, about 3.9 W/m² for OLBL.

  ([#158](https://github.com/lewisjared/rscm/pull/158))
- `OzoneForcing` now follows MAGICC7's ozone module.
  It reads N2O and the previous year's fossil and AFOLU NOx, CO and NMVOC emissions
  (`Emissions|<gas>|Fossil` and `Emissions|<gas>|AFOLU`),
  measures changes from the first year of the run,
  and writes the ozone due to N2O and temperature as separate agents.
  The `eesc_reference`, `ch4_pi`, `nox_pi`, `co_pi`, `nmvoc_pi` and `temp_feedback_scale` parameters
  were removed.

  `HalocarbonChemistry` parameters follow MAGICC7's species table.
  Each species now has separate OH and stratospheric lifetimes and `molecular_mass` replaces `molecular_weight`.
  The global settings were renamed, for example `br_multiplier` to `br_vs_cl_scale`.

  `GhgForcing` now also writes `Effective Radiative Forcing|2xCO2`
  and `Effective Radiative Forcing|CH4 Oxidation Stratospheric H2O`,
  and `ClimateUDEB` writes `Equilibrium Climate Sensitivity`.
  A model built with a `VariableSchema` must declare them. ([#159](https://github.com/lewisjared/rscm/pull/159))

### Features

- Added Python regression test framework for validating MAGICC component implementations against Fortran MAGICC7 reference outputs. Tests cover concentration-driven runs, emissions-driven runs, climate sensitivity sweeps, and CO2-only forcing mode. All 9 tests are skipped pending MAGICC component implementation. ([#70](https://github.com/lewisjared/rscm/pull/70))
- Added MAGICC components: `ClimateUDEB` (4-box energy balance), `CH4Chemistry`, `N2OChemistry`, `HalocarbonChemistry`, `TerrestrialCarbon`, `OceanCarbon`, `CO2Budget`, `OzoneForcing`, `AerosolDirect`, and `AerosolIndirect`. ([#73](https://github.com/lewisjared/rscm/pull/73))
- Added TOML-based configuration layer for defining and loading model configurations.
  This includes base config classes, a component registry, parameter metadata introspection, and a model builder.

  Example configurations for the two-layer model are provided in `configs/`. ([#74](https://github.com/lewisjared/rscm/pull/74))
- Added GHG radiative forcing component (`GhgForcing`) supporting IPCCTAR (Myhre et al. 1998) and Etminan (2016) methods for CO2, CH4, and N2O forcing with rapid adjustment factors. ([#111](https://github.com/lewisjared/rscm/pull/111))
- Added LAMCALC iterative feedback parameter solver, temperature-dependent vertical diffusivity, and time-varying ECS to the ClimateUDEB component, improving MAGICC7 parity for ERF-to-temperature calculations. ([#120](https://github.com/lewisjared/rscm/pull/120))
- Aligned ocean UDEB tridiagonal solver with MAGICC v7.5.3 (AREAFACTOR_DIFFFLOW, DZ1 half-thickness, time-varying alpha_eff, area-weighted upwelling temperature). Added 10 progressive ocean regression tests comparing against MAGICC reference data, with 6 passing at 15% tolerance. ([#121](https://github.com/lewisjared/rscm/pull/121))
- Added internal efficacy computation for CO2 forcing. After LAMCALC convergence, the CO2 internal efficacy is now computed from the regional forcing pattern and coupling matrix inverse. A new `efficacy_apply` parameter (default 0, AR6 mode = 2) gates whether effective forcing is adjusted by `prescribed_efficacy / internal_efficacy`. The public `calc_internal_efficacy()` function supports computing efficacy for any agent's spatial forcing pattern.

  Changed the default `rf_regions_co2` from uniform `[1.0, 1.0, 1.0, 1.0]` to the MAGICC7 AR6 calibration values `[1.4089, 1.37045, 1.43333, 1.33257]`. This affects the LAMCALC-derived feedback parameters and equilibrium temperature pattern even when `efficacy_apply = 0`. ([#126](https://github.com/lewisjared/rscm/pull/126))
- Added regional CO2 forcing distribution (qfrac) in the climate solve loop and prescribed efficacy mode (`efficacy_apply=1`). Forcing is now distributed per-box using the CO2 regional pattern, consistent with LAMCALC calibration. ([#128](https://github.com/lewisjared/rscm/pull/128))
- Upgraded terrestrial carbon cycle to near-parity with MAGICC7 `TERRCARBON2` with three CO2 fertilization methods (logarithmic, Gifford, sigmoid with blending), time-varying turnover times, no-feedback reference pools for regrowth attribution, and mass conservation correction. The CO2-plus-temperature scenario retains maximum relative differences of 0.22% for the plant pool and 0.68% for the soil pool because processed feedback temperatures are not yet implemented. ([#131](https://github.com/lewisjared/rscm/pull/131))
- Added a shared MAGICC runner for concentration-driven and partial CO2 emissions-driven experiments with prescribed carbon sinks and residual forcing. Runs now report their input sources and validate finite outputs. ([#144](https://github.com/lewisjared/rscm/pull/144))
- Added an offline SSP245 concentration baseline report with complete annual forcing and temperature comparisons, input validation, and explicit provenance and alignment limitations. ([#145](https://github.com/lewisjared/rscm/pull/145))
- Added `rscm.magicc.run_magicc_emissions` and `Emissions` for emissions-driven runs that calculate CO2, CH4 and N2O with the coupled carbon cycle and chemistry. Temperature feeds back on the land and ocean sinks and on the CH4 and N2O lifetimes. Against MAGICC7 v7.5.3 SSP245 run emissions-driven from 1750, concentrations stay within 0.5 ppm CO2 and 3 ppb CH4, and temperature within 0.02 K. Also added `CH4OxidationBuilder`, which adds the CO2 released by oxidation of fossil CH4. ([#158](https://github.com/lewisjared/rscm/pull/158))
- Added the remaining MAGICC7 processes needed to run its default configuration from emissions.

  - Added a `Permafrost` component reproducing MAGICC7's permafrost carbon feedback.
    Setting `permafrost` on `CH4Chemistry`, `CH4Oxidation`, `OceanCarbon` and `CO2Budget`
    feeds its emissions back.
  - Added `ForcingAggregation`,
    which applies MAGICC7's efficacies and regional patterns to every forcing agent.
    An agent's `sum_of` builds it from other agents before efficacy,
    `summation` sets MAGICC7's order of addition,
    and `ForcingAggregationBuilder.internal_efficacies` reports each agent's internal efficacy.
  - `HalocarbonChemistry` with `fgas_regions` and `OzoneForcing` with `trop_regions`
    write four-box forcing spread as MAGICC7 spreads it.
  - Added `rscm.magicc.full.REWRITTEN_PARAMETERS`,
    the parameters MAGICC7 rewrites at run time and that its configuration file does not show.
  - Added `rscm.magicc.prescribed`, which calculates MAGICC7's prescribed aerosol, land-use,
    solar and volcanic forcing from its input files.
  - `ClimateUDEB` gained regional forcing, a time-varying CO2 doubling forcing
    and monthly forcing such as volcanic eruptions.
  - `HalocarbonChemistry` covers all 41 MAGICC7 species, and `OzoneForcing` follows MAGICC7's ozone module.
  - `ModelBuilder` gained `with_exogenous_four_box_variable` and `with_initial_grid_values`.
  - Added `rscm.magicc.full.run_magicc7`, which runs the full chain from emissions
    with MAGICC7's default parameters and packaged input files.
  - `TerrestrialCarbon` gained MAGICC7's nitrogen limitation of NPP through its `nitrogen` parameters.
    It reads `Nitrogen Deposition` and `Nitrogen Fixation`
    and writes the nitrogen pool, fluxes and limitation factor.
    `run_magicc7` enables it with `ncycle_apply`.
  - Added a `SeaLevelRise` component for MAGICC7's glacier and ice sheet contributions to sea level.
    `ClimateUDEB` calculates thermal expansion and the annual mean warming when `thermal_expansion` is set.
    `run_magicc7` adds both with `out_sealevel`.
  - `CO2Budget`, `OceanCarbon`, `CH4Chemistry`, `CH4Oxidation`, `N2OChemistry` and `HalocarbonChemistry`
    gained MAGICC7's switch from prescribed concentrations to emissions.
    Until `concentration_switch_year` (`fgas_switch_year` and `mhalo_switch_year` for halocarbons)
    they read `Atmospheric Concentration|<gas>|Prescribed`.
    `CH4Chemistry` then writes the inverse emissions `Emissions|CH4|Inverse`.
    `run_magicc7` prescribes concentrations up to 2015 from MAGICC7's packaged files,
    as MAGICC7 does by default.
  - `CH4Chemistry` writes its OH lifetime `Lifetime|CH4|OH`
    and the effective lifetime `Lifetime|CH4|Effective` that `CH4Oxidation` uses.
  - Added `rscm.magicc.full.natural_emissions`,
    which calculates the natural CH4 and N2O emissions that close MAGICC7's budget.
  - Added `rscm.magicc.prescribed.volcanic_pattern`,
    which derives MAGICC7's volcanic forcing pattern from the run's monthly forcing.
  - `run_magicc7` reads MAGICC7's method switches `core_co2ch4n2o_rfmethod`, `co2_fertilization_method`,
    `co2_plantboxresp_fertscale`, `co2_feedbackfactor_*` and `co2_tempfeedback_switch`.
    It rejects `co2_plantboxresp_method` other than 2, the only respiration method implemented.

  The full chain matches MAGICC7's default SSP245 run from 1750 to 2104 in its binary output,
  as do the emissions-driven runs with permafrost, nitrogen limitation or sea level rise.
  Runs that change one method switch stay within 7.1e-15 K of MAGICC7.

  ([#159](https://github.com/lewisjared/rscm/pull/159))

### Improvements

- Added `ComponentState` trait for components requiring persistent internal state across `solve()` calls, with JSON checkpoint/restore support for model serialization. ([#73](https://github.com/lewisjared/rscm/pull/73))
- Coupled ground heat reservoir into land-ocean equilibrium equations, fixing a ~6% warm bias in the ClimateUDEB ground heat test. Tightened GHG forcing regression tolerances from 0.1% to 0.001% and added phased tolerance checking (shock/convergence/final) for ocean regression tests. ([#123](https://github.com/lewisjared/rscm/pull/123))
- Align ClimateUDEB substep forcing timing with MAGICC7 convention, fixing ocean tests 08 (SST-to-SAT) and 10 (full default) and reducing ECS sweep bias from 7.5x to 0.8x. ([#124](https://github.com/lewisjared/rscm/pull/124))
- Switched the default initial ocean temperature profile to the CMIP5 multi-model mean (matching MAGICC7's `CORE_SWITCH_OCN_TEMPPROFILE=2`) with per-hemisphere profiles, and fixed the ground heat coupling to use MAGICC7's explicit formulation instead of absorbing K_LG into the coupled feedback system. Together these reduced parity failures from 11/33 to 1/33, with all ECS sweep values (1.5-4.5) and efficacy tests now passing. Maximum temperature error reduced from ~5% to ~1.2%. ([#129](https://github.com/lewisjared/rscm/pull/129))
- Added multi-model ocean carbon IRF support with `IrfForm` enum (Polynomial, ExponentialSum) and `OceanCarbonModel` enum (3D-GFDL, 2D-BERN, HILDA). Switched the default ocean carbon model from 2D-BERN to 3D-GFDL to match MAGICC7's default configuration, and fixed the temperature sensitivity parameter from 0.0423 to 0.03717879 (`OCEANCC_TEMPFEEDBACK`). ([#130](https://github.com/lewisjared/rscm/pull/130))

### Bug Fixes

- Aligned ClimateUDEB ocean solver with MAGICC7 Fortran source, fixing area factor application and inter-hemispheric heat exchange formulation. Test 07 (K_NS exchange) parity improved by 95% and test 05 (temperature-dependent diffusivity) by 77%. ([#125](https://github.com/lewisjared/rscm/pull/125))
- Reject nonannual terrestrial carbon timesteps through all solver entry points and include the mass conservation correction in the respiration diagnostic. Nonannual model intervals return a component error through the model runner. ([#132](https://github.com/lewisjared/rscm/pull/132))
- Aligned GHG forcing with its concentration boundary and initialized forcing in the MAGICC runner. Model execution now orders producers before consumers and reports component failures through Python exceptions or Rust's fallible run methods. ([#144](https://github.com/lewisjared/rscm/pull/144))
- Fixed biases in the MAGICC chemistry and carbon cycle. These were the main ones:

  - Land-use emissions were counted twice in the CO2 budget.
  - CH4 and N2O integrated from the previous year's concentration.
  - The ocean outgassed at the start of the run.
  - Carbon-cycle and chemistry feedbacks saw an unweighted mean of the four temperature boxes rather than the area-weighted mean.

  ([#158](https://github.com/lewisjared/rscm/pull/158))
- `ClimateUDEB` now matches MAGICC7 from its own forcing.

  - The first year runs the substep at its start, as MAGICC7 does.
  - Ocean diffusivity varies with the layer index rather than depth,
    so the mixed layer thickness no longer shifts it.
  - The ocean takes the ground heat flux from ground temperatures before the substep's update.
  - Each monthly substep takes the internal efficacy of the year it ends in,
    so volcanic forcing within a year uses the previous year's efficacy until the last month.

  `TerrestrialCarbon` includes the current year's land-use emissions in the cumulative emissions
  behind its turnover times, as MAGICC7 does.

  `OceanCarbon` uses the impulse response coefficients of the MAGICC7 binary,
  which carry more digits than Joos et al. (2001).
  This fixes the 2D-BERN early lifetime of 0.0032825 years to 0.032825 and its late constant term to 0.013691.
  The late response is now scaled by the early form at the switch time, as in MAGICC7.

  `Permafrost` now refreezes as MAGICC7 does.
  Each soil type refreezes at most its thawed area before the step,
  at that area's carbon density,
  and pools are clamped at zero after the thaw or refreeze but not after decomposition.
  Thaw and refreeze are split between aerobic and anaerobic soil by the previous step's aerobic fraction,
  and the emissions follow MAGICC7's order of operations,
  so `Permafrost` reproduces MAGICC7's emissions bit for bit from its temperature.

  `ClimateUDEB` and `ForcingAggregation` invert the climate feedback matrix by cofactors, as MAGICC7 does,
  so internal efficacies match it bit for bit.

  `efficacy_apply = 1` in `ClimateUDEB` and `ForcingAggregation` now divides forcing by the internal efficacy
  and ignores the prescribed efficacy, as MAGICC7 does.
  It previously multiplied by the prescribed efficacy.
  `run_magicc7` also applies `rf_efficacy_apply` to the monthly volcanic forcing.

  `GhgForcing` with the IPCCTAR method leaves the N2O overlap out of the stratospheric H2O forcing,
  and computes the CH4 and N2O overlap in MAGICC7's ppm form.
  `run_magicc7` applies no rapid adjustment with IPCCTAR, as MAGICC7 ignores `core_rfrapidadjust_*` there.

  ([#159](https://github.com/lewisjared/rscm/pull/159))

### Improved Documentation

- Added a complex coupled model example with temperature-carbon cycle feedback to the coupled model notebook. ([#77](https://github.com/lewisjared/rscm/pull/77))
- Added a [MAGICC Coupled Model](notebooks/magicc_coupled_model.py) tutorial.
  It assembles MAGICC7's emissions-driven chain from RSCM components,
  shows its module graph and checks it against MAGICC7 v7.5.3.
  It replaces the MAGICC configuration notebook.
  Updated the guides, tutorials and module specs to match current features,
  and added component pages for the MAGICC components. ([#163](https://github.com/lewisjared/rscm/pull/163))

### Trivial/Internal Changes

- [#71](https://github.com/lewisjared/rscm/pull/71), [#72](https://github.com/lewisjared/rscm/pull/72), [#76](https://github.com/lewisjared/rscm/pull/76), [#122](https://github.com/lewisjared/rscm/pull/122), [#164](https://github.com/lewisjared/rscm/pull/164)


## rscm 0.5.0 (2026-01-26)

### ⚠️ Breaking Changes ⚠️

- Replaced ambiguous `current()` and `next()` methods on `TimeseriesWindow` with explicit `at_start()` and `at_end()` accessors that clearly communicate timestep semantics. ([#67](https://github.com/lewisjared/rscm/pull/67))

### Features

- Added VariableSchema for declaring model variables and aggregation relationships. Supports Sum, Mean, and Weighted aggregation operations with hierarchical aggregates, automatic validation, and cycle detection. Virtual aggregator components are inserted into the model graph during build. ([#66](https://github.com/lewisjared/rscm/pull/66))
- Added schema-driven grid auto-aggregation. Components can now read variables at coarser resolutions than the schema declares (e.g., reading a Scalar when the schema stores FourBox data), and write at finer resolutions. The model automatically aggregates using configurable area weights via `ModelBuilder.with_grid_weights()`. ([#68](https://github.com/lewisjared/rscm/pull/68))
- Added model calibration and uncertainty quantification framework (`rscm.calibrate`). Supports MCMC sampling via affine-invariant ensemble sampler (Goodman & Weare 2010), point estimation, convergence diagnostics (R-hat, ESS), and chain persistence. Includes Python bindings with tqdm progress bars and pandas DataFrame integration. ([#69](https://github.com/lewisjared/rscm/pull/69))

### Improvements

- Improved the `/proposal` OpenSpec command with a two-phase workflow that separates investigation from specification, requiring user approval between phases. ([#62](https://github.com/lewisjared/rscm/pull/62))
- Split CI docs into parallel jobs with path filtering for faster builds. ([#63](https://github.com/lewisjared/rscm/pull/63))

### Bug Fixes

- Fixed DOT graph output to properly escape quotes in component labels, enabling compatibility with pydot and other DOT parsers. ([#66](https://github.com/lewisjared/rscm/pull/66))

### Improved Documentation

- Added Variable Schemas tutorial covering aggregation operations (Sum, Mean, Weighted), hierarchical aggregates, NaN handling, and schema validation. ([#66](https://github.com/lewisjared/rscm/pull/66))

### Trivial/Internal Changes

- [#64](https://github.com/lewisjared/rscm/pull/64)


## rscm 0.4.1 (2026-01-17)

No significant changes.

## rscm 0.4.0 (2026-01-17)

### ⚠️ Breaking Changes ⚠️

- Renamed components to remove the trailing "Component" suffix (e.g., `TwoLayerComponent` is now `TwoLayer`, `CarbonCycleComponent` is now `CarbonCycle`). ([#61](https://github.com/lewisjared/rscm/pull/61))

### Improvements

- Added mermaid diagram rendering support in documentation. ([#61](https://github.com/lewisjared/rscm/pull/61))

### Improved Documentation

- Expanded documentation with new tutorials for spatial grids, debugging, and scenario pipelines. Added cross-references from documentation pages to the Python API. ([#61](https://github.com/lewisjared/rscm/pull/61))

## rscm 0.3.0 (2026-01-16)

### ⚠️ Breaking Changes ⚠️

- Replaced `RequirementType::InputAndOutput` with `RequirementType::State` for variables that read previous values and write new values. ([#53](https://github.com/lewisjared/rscm/pull/53))
- Reorganised workspace structure with all Rust crates now under `crates/` directory and improved Python module organisation with dedicated submodules for spatial and state types.
  The public API of the Python package was reorganised. ([#55](https://github.com/lewisjared/rscm/pull/55))

### Features

- Added grid-based timeseries infrastructure for spatially-resolved climate modeling.

  This release introduces support for regional climate data through a flexible grid system:

  - **Grid Types**: Implemented `ScalarGrid` (single global value), `FourBoxGrid` (MAGICC standard with Northern/Southern Ocean/Land regions), and `HemisphericGrid` (Northern/Southern hemispheres)
  - **GridTimeseries**: Generic timeseries type `GridTimeseries<T, G>` with compile-time grid knowledge, supporting interpolation, aggregation, and transformations between grid types
  - **Type-Safe Access**: Region enums (`FourBoxRegion`, `HemisphericRegion`, `ScalarRegion`) provide compile-time safety for regional data access
  - **State System Integration**: Added `StateValue` enum and grid-aware methods (`get_latest_value()`, `get_global()`, `get_region()`) to `InputState` for seamless scalar/grid handling
  - **Python Bindings**: Exposed all grid types to Python with PyO3 wrappers, including region constants and grid operations
  - **Example Component**: Included `FourBoxOceanHeatUptakeComponent` demonstrating scalar-to-grid disaggregation patterns

  The implementation maintains full backwards compatibility with existing scalar timeseries code. Components can continue using scalar values while new components can leverage regional resolution where appropriate. Grid transformations are explicit and type-checked to prevent silent data loss.

  ([#48](https://github.com/lewisjared/rscm/pull/48))
- Added `ComponentIO` derive macro for type-safe component development with struct-level `#[inputs(...)]`, `#[outputs(...)]`, and `#[states(...)]` attributes.

  The macro generates typed input/output structs with compile-time field validation, eliminating stringly-typed component APIs. Components can now access inputs through `TimeseriesWindow` for zero-cost timeseries access and return typed output structs. ([#53](https://github.com/lewisjared/rscm/pull/53))
- Components can now return spatially-resolved grid outputs (FourBox and Hemispheric) natively, enabling regional values to flow between components without aggregation to scalars.
  The `OutputState` type now supports `StateValue` variants for grids, and the ComponentIO macro automatically wraps grid outputs in the appropriate variant. ([#56](https://github.com/lewisjared/rscm/pull/56))
- Added automatic component documentation generation.
   The new `rscm-doc-gen` tool extracts metadata from `ComponentIO` macro attributes and generates documentation pages for each component, including inputs, outputs, states, and mathematical formulations. ([#59](https://github.com/lewisjared/rscm/pull/59))

### Improvements

- Added spatial grid support (`GridType::Scalar`, `GridType::FourBox`, `GridType::Hemispheric`) to component requirements with compile-time grid validation and automatic grid transformation in the model coupler. ([#53](https://github.com/lewisjared/rscm/pull/53))
- Updated legacy components to use the new ComponentIO derive macro. ([#58](https://github.com/lewisjared/rscm/pull/58))

### Trivial/Internal Changes

- [#58](https://github.com/lewisjared/rscm/pull/58)

## rscm 0.2.3 (2026-01-14)

No significant changes.

## rscm 0.2.2 (2026-01-14)

No significant changes.

## rscm 0.2.1 (2026-01-14)

No significant changes.

## rscm 0.2.0 (2026-01-14)

### ⚠️ Breaking Changes ⚠️

- Refactored `InputState` to include references to `TimeSeries` instead of scalar values.
  This is requires a change to the `Component` interface. ([#17](https://github.com/lewisjared/rscm/pull/17))

### Features

- Add Ocean Surface Partial Pressure (OSPP) component to the `rscm-components` crate. ([#10](https://github.com/lewisjared/rscm/pull/10))
- Added automated release workflow for publishing packages to crates.io and PyPI with cross-platform wheel builds. ([#47](https://github.com/lewisjared/rscm/pull/47))

### Bug Fixes

- Fixed CI failure on Python 3.13 by upgrading dependencies (pandas 2.2.2 -> 2.3.3 which includes Python 3.13 wheels). ([#39](https://github.com/lewisjared/rscm/pull/39))

### Improved Documentation

- Add the basic framework for a `mkdocs`-based documentation site in the `docs/` directory. ([#18](https://github.com/lewisjared/rscm/pull/18))
- Added MAGICC module documentation ([#46](https://github.com/lewisjared/rscm/pull/46))

### Trivial/Internal Changes

- [#39](https://github.com/lewisjared/rscm/pull/39)

## rscm 0.1.0 (2024-09-24)

### Improvements

- Add changelog management to the release process ([#9](https://github.com/lewisjared/rscm/pull/9))
