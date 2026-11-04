# FaIR port: TODO

Decisions are in [investigation.md](investigation.md#decisions).

## Slice 0: setup

- [ ] Fixture script `scripts/fair/generate_fixtures.py`, run with pinned `fair==2.2.5`.
- [ ] `crates/rscm-fair` crate skeleton, wired into the workspace and the Python bindings.

## Slice 1: energy balance

- [ ] `FairEnergyBalance`: n-layer Cummins model with the matrix exponential from nalgebra.
- [ ] Outputs: surface and layer temperatures, TOA imbalance, ocean heat content change.
- [ ] Parity against forcing-driven FaIR for 2-layer and 3-layer configs, annual and 5-yearly.

## Slice 2: CO2

- [ ] `FairGasCycle` with gas boxes in `ComponentState` and the iIRF100 lifetime scaling.
- [ ] `FairGhgForcing` with Meinshausen 2020.
- [ ] Parity for a CO2-only emissions-driven run.

## Slice 3: all greenhouse gases

- [ ] Every default GHG in `FairGasCycle`, including inverse (concentration-driven) mode.
- [ ] `FairCH4Lifetime` (Thornhill 2021) and `FairEesc`.
- [ ] Parity for every default GHG.

## Slice 4: other forcing

- [ ] `FairAerosolForcing` (ERFari and ERFaci).
- [ ] `FairOzoneForcing` with temperature feedback.
- [ ] `FairLinearForcing` for contrails, LAPSI, stratospheric H2O and land use.
- [ ] Prescribed solar and volcanic forcing, and the efficacy-weighted total.
- [ ] Parity for the full default species set, SSP2-4.5 and SSP1-1.9.

## Slice 5: Python

- [ ] `rscm.fair` loaders for the species CSV and calibration CSV.
- [ ] `from_fair()` converter.
- [ ] `run_fair()` and the ensemble helper returning xarray.
- [ ] `.pyi` stubs, docs page and changelog fragment.
- [ ] Parity for members of the fair-calibrate v1.6.1 posterior.
- [ ] Smoke test with MAGICC's `ClimateUDEB`.

## Later

- [ ] Ensemble benchmark against `fair`.
- [ ] Stochastic mode.
- [ ] Other GHG and CH4 methods.
- [ ] Restarts.
