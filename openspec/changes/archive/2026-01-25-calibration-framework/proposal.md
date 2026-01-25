# Change: Model Calibration and Constraining Framework

## Why

Climate model development requires calibrating component parameters against observations (IPCC AR6 ranges, historical data). This involves:

- **Calibration**: Point estimation to find best-fit parameters
- **Constraining**: MCMC posterior sampling for uncertainty quantification

Currently RSCM has no calibration infrastructure. The existing Python library (scmcallib) is unmaintained. With millions of model evaluations required, calibration must be a first-class Rust feature.

## What Changes

- Add new `rscm-calibrate` crate with:
  - Prior distributions (Uniform, Normal, Bound, etc.)
  - Parameter specification (bounds, transformations)
  - Affine-invariant ensemble sampler (emcee algorithm)
  - Point estimation optimizers (via argmin)
  - Likelihood/metric computation
  - Parallel model execution (rayon)
  - Convergence diagnostics (R-hat, ESS)
  - Chain persistence

- Add Python bindings for calibration API
- Add Python visualization utilities (trace plots, corner plots)

## Chosen Approach

Pure Rust implementation (Approach B from investigation) with Python bindings for API access and visualization.

Key design decisions:

1. Implement Goodman & Weare stretch move algorithm in Rust
2. Use rayon for embarrassingly parallel model evaluations
3. Export chain data to Python for matplotlib/corner visualisation
4. Support both point estimation and MCMC workflows

## Impact

- Affected specs: None (new capability)
- New spec: `calibration` capability
- Affected code:
  - New crate: `crates/rscm-calibrate/`
  - Python bindings: `crates/rscm/src/python/` (add calibrate submodule)
  - Python package: `python/rscm/calibrate/`
