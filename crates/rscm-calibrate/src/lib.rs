//! Parameter calibration and uncertainty quantification for climate models.
//!
//! This crate provides a comprehensive framework for calibrating reduced-complexity
//! climate models against observations using Bayesian inference and optimisation.
//!
//! # Overview
//!
//! Climate model calibration involves two main workflows:
//!
//! - **Point estimation**: Find the single best-fit parameter set that maximises
//!   agreement with observations
//! - **Bayesian constraining**: Sample the posterior distribution to quantify
//!   parameter uncertainties
//!
//! This crate provides both workflows with efficient parallel execution using
//! the affine-invariant ensemble sampler algorithm ([Goodman & Weare 2010]).
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use rscm_calibrate::*;
//!
//! // 1. Define parameter priors
//! let mut params = ParameterSet::new();
//! params.add("climate_sensitivity", Box::new(Uniform::new(2.0, 5.0).unwrap()));
//! params.add("heat_capacity", Box::new(Uniform::new(5.0, 15.0).unwrap()));
//!
//! // 2. Specify observation targets
//! let mut target = Target::new();
//! target.add_variable("Temperature|Global")
//!     .add(2020.0, 1.2, 0.1).unwrap()
//!     .add(2021.0, 1.3, 0.1).unwrap();
//!
//! // 3. Create model runner (user implements ModelRunner trait)
//! # struct MyModelRunner { param_names: Vec<String> }
//! # impl ModelRunner for MyModelRunner {
//! #     fn param_names(&self) -> &[String] { &self.param_names }
//! #     fn run(&self, _params: &[f64]) -> rscm_calibrate::Result<ModelOutput> {
//! #         Ok(ModelOutput { variables: std::collections::HashMap::new() })
//! #     }
//! # }
//! # let runner = MyModelRunner { param_names: vec!["climate_sensitivity".into(), "heat_capacity".into()] };
//!
//! // 4. Configure likelihood
//! let likelihood = GaussianLikelihood::new();
//!
//! // 5. Run MCMC sampling
//! let sampler = EnsembleSampler::new(
//!     params,
//!     runner,      // Takes runner directly (not boxed)
//!     likelihood,  // Takes likelihood directly (not boxed)
//!     target,
//! );
//!
//! let chain = sampler.run(
//!     1000,                          // iterations
//!     WalkerInit::FromPrior,         // initialisation
//!     1,                             // thinning
//! ).unwrap();
//!
//! // 6. Check convergence
//! let r_hat = chain.r_hat(100);  // discard first 100 samples as burn-in
//! println!("R-hat diagnostics: {:?}", r_hat);
//!
//! // 7. Extract results
//! let samples = chain.flat_samples(100);
//! println!("Posterior mean: {:?}", samples.mean_axis(ndarray::Axis(0)));
//! ```
//!
//! # Features
//!
//! - **Prior distributions**: Uniform, Normal, LogNormal, Bounded wrappers
//! - **Optimisers**: Random search, L-BFGS-B, Nelder-Mead, Particle Swarm
//! - **MCMC sampling**: Affine-invariant ensemble sampler with parallel walkers
//! - **Diagnostics**: Gelman-Rubin (R-hat), effective sample size (ESS), autocorrelation time
//! - **Checkpointing**: Save/resume long-running chains
//! - **Parallel execution**: Rayon-based parallel model evaluation
//! - **Python bindings**: Optional PyO3 bindings for Python API
//!
//! # Algorithm References
//!
//! - [Goodman & Weare 2010]: "Ensemble samplers with affine invariance",
//!   Communications in Applied Mathematics and Computational Science, Vol. 5, No. 1, 65-80
//!
//! [Goodman & Weare 2010]: https://doi.org/10.2140/camcos.2010.5.65

pub mod distribution;
pub mod likelihood;
pub mod model_runner;
pub mod optimizer;
pub mod parameter_set;
pub mod point_estimator;
#[cfg(feature = "python")]
pub mod python;
pub mod sampler;
pub mod target;

pub use distribution::{Bound, Distribution, LogNormal, Normal, Uniform};
pub use likelihood::{GaussianLikelihood, LikelihoodFn, ModelOutput, VariableOutput};
pub use model_runner::{DefaultModelRunner, ModelRunner};
pub use optimizer::{OptimizationResult, Optimizer};
pub use parameter_set::ParameterSet;
pub use point_estimator::PointEstimator;
pub use sampler::{Chain, EnsembleSampler, ProgressInfo, SamplerState, StretchMove, WalkerInit};
pub use target::{Observation, Target, VariableTarget};

/// Result type for calibration operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during calibration.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid distribution parameter: {0}")]
    InvalidParameter(String),

    #[error("Sampling failed: {0}")]
    SamplingError(String),

    #[error("Model execution failed: {0}")]
    ModelError(String),
}
