# FaIR port: TODO

Decisions are in [investigation.md](investigation.md#decisions).

## Slice 0: setup

- [x] Fixture script `scripts/fair/generate_fixtures.py`, run with pinned `fair==2.2.5` and `numpy==2.4.2`.
- [x] `crates/rscm-fair` crate skeleton, wired into the workspace and the Python bindings.

Fixtures are `csv.gz` plus JSON, like the MAGICC ones, rather than netCDF.
This keeps xarray out of the test dependencies.
The script disables NumPy's AVX-512 `exp`.
It differs from the C library's by an ulp, which CO2's 1e9-year box turns into a 5e-9 relative difference.

## Slice 1: energy balance

- [x] `FairEnergyBalance`: n-layer Cummins model with the matrix exponential from nalgebra.
- [x] Outputs: surface and layer temperatures, TOA imbalance, ocean heat content change.
- [x] Parity against forcing-driven FaIR for 2-layer and 3-layer configs, annual and 5-yearly.

## Slice 2: CO2

- [x] `FairGasCycle` with gas boxes in `ComponentState` and the iIRF100 lifetime scaling.
- [x] `FairGhgForcing` with Meinshausen 2020.
- [x] Parity for CO2, CH4 and N2O emissions-driven (`major_ghg_ssp245`).
  Meinshausen 2020 needs all three, so there is no CO2-only case.

## Slice 3: all greenhouse gases

- [x] Every default GHG in `FairGasCycle`, including inverse (concentration-driven) mode.
- [x] `FairCH4Lifetime` (Thornhill 2021) and `FairEesc`.
- [x] Parity for every default GHG.

## Slice 4: other forcing

- [x] `FairLinearForcing` for aerosol-radiation, ozone, contrails, LAPSI, stratospheric H2O and land use.
- [x] `FairAerosolCloud` for aerosol-cloud interactions.
- [x] Prescribed solar and volcanic forcing, and the efficacy-weighted total (`FairForcingAggregation`).
- [x] Parity for the full default species set, SSP2-4.5 and SSP1-1.9, 5-yearly and concentration-driven.

FaIR calculates EESC after the aerosol forcing, so EESC never reaches ERFari.
`rscm.fair` leaves it out of the aerosol drivers to match.

## Slice 5: Python

- [x] `rscm.fair.build_components`, `initial_values` and `run_fair()` from a FaIR config dict.
- [x] `rscm.fair` loaders for the species CSV and calibration CSV.
- [x] `from_fair()` converter, with fair and xarray in a `fair` extra.
- [x] Ensemble helper returning xarray (`run_ensemble`).
- [x] `.pyi` stubs.
- [ ] Docs page.
- [ ] Changelog fragment, once the PR has a number.
- [x] Parity for members of the fair-calibrate v1.6.1 posterior.
- [x] Smoke test with MAGICC's `ClimateUDEB` (`build_components(..., energy_balance=False)`).

The posterior cases run three members over the CMIP7 history (1750 to 2024) deterministically.
The posterior sets `stochastic_run`, which `read_calibration` ignores.
FaIR ignores `forcing_scale` for prescribed forcing,
so `scale_prescribed_forcing` applies it to the inputs as fair-calibrate does.

## Later

- [ ] Ensemble benchmark against `fair`.
- [ ] Stochastic mode.
- [ ] Other GHG and CH4 methods.
- [ ] Restarts.
