# FaIR in RSCM: investigation

Notes on porting FaIR v2 to RSCM components, written against FaIR v2.2.5 (OMS-NetZero/FAIR, Apache-2.0).

## Summary

A FaIR clone is feasible and much smaller than the MAGICC port.
FaIR's physics is about 1,500 lines of NumPy,
across `fair.py`, `gas_cycle/`, `forcing/` and `energy_balance_model.py`.
Every equation is closed form: exponential gas-box decay, linear or logarithmic forcing,
and a matrix-exponential energy balance step.
The main loop has a fixed order with no within-step cycles, so it maps onto RSCM's dependency graph directly.

The work is mostly plumbing, not physics:

- A species table with 60-odd species and four input modes drives everything.
- Ensemble throughput has to compete with FaIR's NumPy broadcasting over configs and scenarios.
- Bit-level parity is unlikely because NumPy's pairwise summation orders additions differently.
  A tolerance near 1e-12 relative is realistic.

FaIR is pip-installable and deterministic, so parity fixtures are cheap to produce,
unlike MAGICC's reference binary.

## FaIR's structure

`FAIR.run()` precomputes a few things, then loops over timepoints.
Emissions sit on timepoints (interval means) and concentrations, forcing and temperature on timebounds.
Step `i` reads emissions at timepoint `i` and state at timebound `i`, and writes timebound `i+1`.
This is RSCM's `t_current` to `t_next` convention with `Previous` interpolation on emissions.

Before the loop:

- `CO2 = CO2 FFI + CO2 AFOLU` and cumulative emissions are summed.
- `g0`, `g1`, `iirf_0` and `concentration_per_emission` come from lifetimes,
  partition fractions and molecular weight.
  These are parameter transforms and belong in the component constructor.
- The Meinshausen forcing at baseline concentration is stored as `ghg_forcing_offset`.
- The energy balance matrices are discretised with `scipy.linalg.expm`,
  and stochastic noise is drawn for the whole run.

The loop, with the RSCM read for each input:

| FaIR step | Calculation | Reads | Proposed component |
| --- | --- | --- | --- |
| 1 | Lifetime scaling `alpha` from iIRF100 (cumulative uptake, airborne, temperature) | temperature `get()`, own state | `GasCycle` |
| 2 | CH4 lifetime scaling, Thornhill 2021 (aerosol precursors, N2O, EESC, temperature) | everything `get()` | `CH4Lifetime` (added before the gas cycle) |
| 3 | Emissions to concentration over up to 4 gas boxes | emissions `get()` | `GasCycle` |
| 4 | Concentration to inverse emissions (concentration driven) | concentration `get()` (prescribed) | `GasCycle` |
| 5 | GHG forcing: Meinshausen 2020, Etminan 2016, Myhre 1998 or Leach 2021 | concentrations `get()` | `GHGForcing` |
| 6 | ERFari, linear in emissions and concentrations | emissions `get()`, concentrations `get()` | `LinearForcing` |
| 7 | ERFaci, log of a weighted sum | as step 6 | `AerosolCloud` |
| 8 | EESC from halogen concentrations | concentrations `get()` | `EESC` |
| 9 | Ozone, Thornhill 2021, linear | as step 6, plus EESC | `LinearForcing` |
| 10 to 13 | Contrails, LAPSI, stratospheric H2O, land use: all linear | emissions, CH4 concentration, cumulative AFOLU CO2 | `LinearForcing`, one instance per agent |
| 14 | Temperature feedback on each forcing agent | temperature `get()` | `ForcingAggregation` |
| 15 | Total forcing and efficacy-weighted total | forcings `get()` | `ForcingAggregation` |
| 16 | n-layer energy balance step `x' = A_d x + b_d F + noise` | total forcing `get()`, own state | `EnergyBalance` |
| 17 and 18 | TOA imbalance and ocean heat content, after the run | forcing, layer temperatures | `EnergyBalance`, per step |

Steps 2 and 8 form a loop across steps: CH4 lifetime reads last step's EESC and concentrations.
Adding `CH4Lifetime` before the gas cycle gives exactly that,
since RSCM only creates edges to producers added earlier.

## Fit with RSCM

What already maps cleanly:

- **Per-species variables.**
  `HalocarbonChemistry` already builds its definitions from a species list at runtime.
  `GasCycle` can do the same for every GHG and pick input or output per species from its input mode.
- **Gas-box state.** `TerrestrialCarbon` keeps its pools in a `ComponentState`.
  Gas boxes, airborne emissions and cumulative emissions fit there,
  which avoids a 4-wide grid that is not a spatial grid.
- **Matrix exponential.** `rscm-core` already depends on nalgebra, which has `Matrix::exp`.
  `A_d` and `b_d` can be built once in the constructor.
- **The driver of each species is fixed for the run.**
  FaIR never switches a species from concentration to emissions mid-run,
  unlike MAGICC's 2015 switch.
  Each species' driver is a build-time choice, which suits the graph.
- **Shared GHG formula.** FaIR's Meinshausen 2020 and MAGICC's OLBL are the same formulae.
  Both use `rscm_components::meinshausen2020`, with their own coefficients.
  FaIR scales by `forcing_scale * (1 + tropospheric_adjustment)` and subtracts the baseline offset.
- **Variable names.** Use RSCM's existing names (`Emissions|CO2|Fossil`, `Atmospheric Concentrations|CH4`,
  `Effective Radiative Forcing|CH4`, `Surface Air Temperature Change`) so FaIR components can mix with MAGICC ones.
  FaIR's `CO2 FFI` and `CO2 AFOLU` map to `Emissions|CO2|Fossil` and `Emissions|CO2|Biosphere`.

## Gaps and risks

1. **Configuration.** FaIR's behaviour lives in `species_configs_properties.csv` (43 columns per species)
   and, for calibrated runs, fair-calibrate's posterior CSV (841 members, 1.4.x or later).
   A Python loader in `rscm.fair` should read both and build the components.
   Rust parameter structs stay typed per component, not a mirror of the CSV.
2. **Stochastic runs.** FaIR draws noise with `scipy.stats.multivariate_normal` and a NumPy seed.
   A Rust RNG cannot reproduce that stream.
   Pass the noise array in as a parameter, generated in Python with SciPy.
   This keeps parity and keeps Rust deterministic.
   Deterministic runs (the default, and what calibration uses) are unaffected.
3. **Ensemble throughput.** FaIR runs 841 configs times several scenarios as one broadcast array.
   RSCM runs one model per member, and FaIR's per-step work is tiny,
   so framework overhead (about 150 named variables, string lookups) may dominate.
   Benchmark this early against `fair` on the calibrated ensemble.
   The step-plan and lookup work in #165 is the starting point.
4. **Parity precision.** `np.nansum` over the species axis uses pairwise summation,
   so totals can differ from a sequential Rust sum in the last bits.
   Plan for `rtol = 1e-12` and only chase bit-identity if a downstream test needs it.
   NaN handling also needs care: FaIR leans on `nansum` and on `alpha = 1` where `exp` overflows (CF4).
5. **Reported forcing includes the feedback.**
   Step 14 adds `T * forcing_temperature_feedback` to each agent's forcing in place.
   Either each forcing component applies its own feedback,
   or the aggregation component re-emits per-agent forcing.
   The second keeps forcing components free of temperature.
6. **Prescribed forcing is not scaled.**
   FaIR states that `forcing_scale` has no effect on species in forcing mode
   (solar, volcanic), but efficacy and temperature feedback still apply.
   fair-calibrate scales solar and volcanic outside the model, so the Python runner has to do the same.
7. **TOA imbalance uses unweighted forcing.** FaIR flags this itself ("is this correct?").
   Match it and note it.
8. **Restarts.** FaIR can start from a saved state (gas boxes, temperatures, OHC).
   Defer this, but keep the state layout compatible.

## Parity testing

Pin `fair==2.2.5` in a dev dependency group and write a fixture script, like the MAGICC fixtures.
Commit netCDF fixtures so CI does not need FaIR or network access.
`fill_from_rcmip` downloads data through pooch.

Cases, roughly in build order:

- Energy balance alone, driven by prescribed total forcing, for 2-layer and 3-layer configs.
- CO2 only, emissions driven, default AR6 species config, Meinshausen forcing.
- Every default species, SSP2-4.5 and SSP1-1.9, with Leach and Thornhill CH4 methods.
- Concentration-driven minor GHGs with inverse emissions, as in fair-calibrate's historical setup.
- A few members of the calibrated ensemble.
- Stochastic run with FaIR's noise passed in.

## Proposed slices

1. `crates/rscm-fair` with `EnergyBalance`, parity against forcing-driven FaIR.
   This proves the expm discretisation and time convention.
2. `GasCycle` for CO2 and `GHGForcing` for CO2, coupled to slice 1.
   This proves the gas-box state and the temperature feedback on `alpha`.
3. All GHGs, concentration driven with inverse emissions, `CH4Lifetime`, `EESC`.
4. Aerosol, ozone and linear forcings, plus `ForcingAggregation` with efficacy and feedback.
5. `rscm.fair` Python package:
   species table and calibration loaders, `run_fair` runner, `.pyi` stubs, docs page.
6. Stochastic mode.
7. Ensemble benchmark against `fair`, and throughput work if needed.

Slices 1 and 2 are small.
Most of the effort is in slices 3 to 5, because of the species bookkeeping.

## Decisions

Settled on 2026-11-04.

- The port is both a parity clone of FaIR v2.2.5 and a set of components that compose with `rscm-magicc`.
- Parity tests use `rtol = 1e-9`, the MAGICC bar.
- v1 covers the default methods only: Meinshausen 2020 GHG forcing and Thornhill 2021 CH4 lifetime.
- Stochastic runs come in a later slice. Restarts are deferred.
- Any uniform step is supported.
- Components live in a new `crates/rscm-fair` crate, exposed as `rscm.fair`.
- `GasCycle` is one component type configured with a list of species.
- Each forcing component applies its own temperature feedback.
- The Meinshausen formula is shared with MAGICC's OLBL in `rscm_components::meinshausen2020`.
- Variables use RSCM and MAGICC names (`Emissions|HFC134a`, `Emissions|SOx`). Components keep FaIR's units.
- `rscm.fair` has native CSV loaders and `from_fair()`, which converts a configured `fair.FAIR`.
- `run_fair()` returns a `Model`, and an ensemble helper returns xarray in FaIR's layout.
- Calibrated runs target the fair-calibrate v1.6.1 posterior.
- Parity fixtures are committed and regenerated by a script with pinned `fair==2.2.5`.
- Ensembles run as a Python loop over models. A benchmark against `fair` is reported but does not gate.
- One smoke test couples FaIR components with MAGICC's `ClimateUDEB`.

Progress is tracked in [TODO.md](TODO.md).
