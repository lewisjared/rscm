//! Affine-invariant ensemble sampler (emcee algorithm).
//!
//! Implements the Goodman & Weare (2010) stretch move algorithm for MCMC sampling.
//! This is a parallel MCMC method that uses an ensemble of "walkers" that explore
//! parameter space together, with each walker's proposal distribution informed by
//! the positions of other walkers.
//!
//! # Module Structure
//!
//! - [`state`]: Sampler state management and progress tracking
//! - [`chain`]: MCMC chain storage and serialization
//! - [`diagnostics`]: Convergence diagnostics (R-hat, ESS, autocorrelation)
//! - [`moves`]: Proposal mechanisms (stretch move)
//! - [`init`]: Walker initialization strategies
//! - [`ensemble`]: Main ensemble sampler implementation
//!
//! # Quick Start
//!
//! ```ignore
//! use rscm_calibrate::{EnsembleSampler, ParameterSet, Target, WalkerInit};
//! use rscm_calibrate::likelihood::GaussianLikelihood;
//!
//! // Define priors
//! let mut params = ParameterSet::new();
//! params.add("sensitivity", Box::new(Uniform::new(0.5, 1.5).unwrap()));
//!
//! // Define observations
//! let mut target = Target::new();
//! target.add_variable("Temperature")
//!     .add(2020.0, 1.2, 0.1).unwrap();
//!
//! // Create and run sampler
//! let sampler = EnsembleSampler::new(params, runner, GaussianLikelihood::new(), target);
//! let chain = sampler.run(1000, WalkerInit::FromPrior, 1)?;
//!
//! // Check convergence and extract samples
//! if chain.is_converged(500, 1.1) {
//!     let samples = chain.flat_samples(500);
//! }
//! ```
//!
//! # References
//!
//! Goodman, J., & Weare, J. (2010). Ensemble samplers with affine invariance.
//! Communications in Applied Mathematics and Computational Science, 5(1), 65-80.

mod chain;
mod diagnostics;
mod ensemble;
mod init;
mod moves;
mod state;

// Re-export public API
pub use chain::Chain;
pub use ensemble::EnsembleSampler;
pub use init::WalkerInit;
pub use moves::StretchMove;
pub use state::{ProgressInfo, SamplerState};
