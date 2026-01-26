# rscm-calibrate

Parameter calibration and uncertainty quantification for reduced-complexity climate models.

## Features

- **Prior distributions**: Uniform, Normal, LogNormal, Bound wrappers
- **Point estimation**: Optimisation-based parameter estimation (planned)
- **Bayesian inference**: Ensemble MCMC sampling (planned)
- **Parallel execution**: Multi-core model evaluation via rayon (planned)
- **Diagnostics**: Convergence checks (Gelman-Rubin, ESS) (planned)

## Current Status

**Implemented**:

- Distribution trait with sampling and log-PDF evaluation
- Core distributions: Uniform, Normal, LogNormal
- Bound<D> wrapper for constraining any distribution

**In Progress**:

- ParameterSet for managing parameter collections
- Target observations and likelihood functions
- Ensemble sampler implementation

## Usage

```rust
use rscm_calibrate::{Uniform, Normal, Bound};

// Create a uniform prior over [0, 1]
let prior = Uniform::new(0.0, 1.0)?;

// Create a bounded normal distribution
let bounded_normal = Bound::new(
    Normal::new(0.0, 1.0)?,
    -2.0,
    2.0
)?;

// Sample from the distribution
let mut rng = rand::thread_rng();
let sample = prior.sample(&mut rng);

// Evaluate log probability density
let log_prob = prior.ln_pdf(0.5);
```

## License

Apache-2.0
