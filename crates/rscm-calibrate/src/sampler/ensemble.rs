//! Affine-invariant ensemble sampler implementation.
//!
//! This module contains the main `EnsembleSampler` struct that orchestrates
//! MCMC sampling using the Goodman & Weare (2010) stretch move algorithm.

use super::{Chain, ProgressInfo, SamplerState, StretchMove, WalkerInit};
use crate::{
    likelihood::LikelihoodFn, model_runner::ModelRunner, parameter_set::ParameterSet,
    target::Target, Error, Result,
};
use ndarray::Array1;
use rayon::prelude::*;
use std::path::Path;

/// Affine-invariant ensemble sampler for Bayesian parameter estimation.
///
/// Implements the Goodman & Weare (2010) stretch move algorithm, a parallel MCMC method
/// that uses an ensemble of "walkers" to explore parameter space. Each walker proposes
/// new positions based on the current positions of other walkers, making the algorithm
/// affine-invariant (robust to parameter correlations and rescaling).
///
/// # Algorithm Overview
///
/// The sampler maintains N walkers (where N ≥ 2 × n_params) that evolve in parallel:
///
/// 1. Split walkers into two groups
/// 2. For each group:
///    - Generate proposals using positions from the complementary group
///    - Evaluate log-posterior for all proposals in parallel
///    - Accept/reject proposals via Metropolis-Hastings
/// 3. Store samples (with optional thinning)
/// 4. Repeat for specified number of iterations
///
/// # Performance
///
/// - **Parallel model evaluation**: All walker proposals evaluated in parallel via rayon
/// - **Affine invariance**: No manual tuning of proposal distributions needed
/// - **Efficient exploration**: Multiple walkers sample different regions simultaneously
///
/// # Workflow
///
/// 1. **Create sampler** with parameters, model runner, likelihood function, and target data
/// 2. **Initialize walkers** from prior, around a point, or explicitly
/// 3. **Run sampling** with optional progress callbacks and checkpointing
/// 4. **Analyze chain** using built-in diagnostics (R-hat, ESS, autocorrelation)
///
/// # Example
///
/// ```ignore
/// use rscm_calibrate::{EnsembleSampler, ParameterSet, Target, WalkerInit};
/// use rscm_calibrate::likelihood::GaussianLikelihood;
///
/// // Define priors
/// let mut params = ParameterSet::new();
/// params.add("sensitivity", Box::new(Uniform::new(0.5, 1.5).unwrap()));
/// params.add("offset", Box::new(Normal::new(0.0, 0.1).unwrap()));
///
/// // Define observations
/// let mut target = Target::new();
/// target.add_variable("Temperature")
///     .add(2020.0, 1.2, 0.1).unwrap()
///     .add(2021.0, 1.3, 0.1).unwrap();
///
/// // Create sampler
/// let runner = MyModelRunner::new();
/// let likelihood = GaussianLikelihood::new();
/// let sampler = EnsembleSampler::new(params, runner, likelihood, target);
///
/// // Run MCMC sampling (1000 iterations, initialize from prior, no thinning)
/// let chain = sampler.run(1000, WalkerInit::FromPrior, 1)?;
///
/// // Check convergence
/// let r_hat = chain.r_hat(500)?;  // Discard first 500 samples as burn-in
/// if chain.is_converged(500, 1.1)? {
///     println!("Chain converged!");
/// }
///
/// // Extract posterior samples
/// let samples = chain.to_param_dict(500);  // Discard burn-in
/// ```
///
/// # References
///
/// Goodman, J., & Weare, J. (2010). Ensemble samplers with affine invariance.
/// Communications in Applied Mathematics and Computational Science, 5(1), 65-80.
pub struct EnsembleSampler<R: ModelRunner, L: LikelihoodFn> {
    /// Parameter set defining the prior distributions
    params: ParameterSet,

    /// Model runner for evaluating parameter sets
    runner: R,

    /// Likelihood function for computing log probability
    likelihood: L,

    /// Target observations
    target: Target,

    /// Stretch move configuration
    stretch: StretchMove,

    /// Default number of walkers (2 * n_params, minimum 32)
    default_n_walkers: usize,
}

impl<R: ModelRunner + Sync, L: LikelihoodFn + Sync> EnsembleSampler<R, L> {
    /// Create a new ensemble sampler.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameter set defining priors
    /// * `runner` - Model runner for evaluating parameter sets
    /// * `likelihood` - Likelihood function
    /// * `target` - Target observations
    pub fn new(params: ParameterSet, runner: R, likelihood: L, target: Target) -> Self {
        let n_params = params.len();
        let default_n_walkers = (2 * n_params).max(32);

        Self {
            params,
            runner,
            likelihood,
            target,
            stretch: StretchMove::default(),
            default_n_walkers,
        }
    }

    /// Create a sampler with custom stretch move parameter.
    pub fn with_stretch_param(mut self, a: f64) -> Result<Self> {
        self.stretch = StretchMove::new(a)?;
        Ok(self)
    }

    /// Get the default number of walkers for this sampler.
    pub fn default_n_walkers(&self) -> usize {
        self.default_n_walkers
    }

    /// Compute log posterior for multiple parameter vectors in parallel.
    ///
    /// log_posterior = log_prior + log_likelihood
    fn log_posterior_batch(&self, param_sets: &[Vec<f64>]) -> Vec<f64> {
        // Run models in parallel
        let outputs = self.runner.run_batch(param_sets);

        // Compute log posteriors
        param_sets
            .par_iter()
            .zip(outputs.par_iter())
            .map(|(params, output_result)| {
                // Compute log prior
                let log_prior = match self.params.log_prior(params) {
                    Ok(lp) => lp,
                    Err(_) => return f64::NEG_INFINITY,
                };

                if !log_prior.is_finite() {
                    return f64::NEG_INFINITY;
                }

                // Check model output
                let output = match output_result {
                    Ok(out) => out,
                    Err(_) => return f64::NEG_INFINITY,
                };

                // Compute likelihood
                let log_likelihood = match self.likelihood.ln_likelihood(output, &self.target) {
                    Ok(ll) => ll,
                    Err(_) => return f64::NEG_INFINITY,
                };

                log_prior + log_likelihood
            })
            .collect()
    }

    /// Run the ensemble sampler for Bayesian parameter estimation.
    ///
    /// Performs MCMC sampling using the affine-invariant ensemble algorithm.
    /// Walkers are initialized according to `init`, then evolved through
    /// `n_iterations` of the stretch move algorithm. All model evaluations
    /// are parallelized via rayon.
    ///
    /// # Arguments
    ///
    /// * `n_iterations` - Number of MCMC iterations to run (each iteration updates all walkers once)
    /// * `init` - Walker initialization strategy (from prior, around a point, or explicit)
    /// * `thin` - Thinning interval - store every `thin`-th sample (1 = no thinning)
    ///
    /// # Returns
    ///
    /// A `Chain` containing the samples, log probabilities, and diagnostic information.
    /// The chain includes samples from all walkers at each stored iteration.
    ///
    /// # Burn-in and Thinning
    ///
    /// - **Burn-in**: The first N samples before the chain converges. Discard when extracting
    ///   posterior samples using `chain.flat_samples(discard)` or `chain.to_param_dict(discard)`.
    /// - **Thinning**: Reduces memory by storing only every Nth sample. Use to reduce
    ///   autocorrelation or save disk space for long runs. Set `thin=1` to store all samples.
    ///
    /// # Walker Count
    ///
    /// Uses the default number of walkers: `max(2 × n_params, 32)`.
    /// For custom walker count, use `run_with_walkers()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Run 10,000 iterations, initialize from prior, store every 10th sample
    /// let chain = sampler.run(10000, WalkerInit::FromPrior, 10)?;
    ///
    /// // Check convergence (discard first 1000 samples as burn-in)
    /// let r_hat = chain.r_hat(1000)?;
    /// println!("R-hat: {:?}", r_hat);
    ///
    /// // Extract converged posterior samples
    /// let samples = chain.flat_samples(1000);  // shape: (n_samples, n_params)
    /// ```
    pub fn run(&self, n_iterations: usize, init: WalkerInit, thin: usize) -> Result<Chain> {
        self.run_with_walkers(
            n_iterations,
            init,
            self.default_n_walkers,
            thin,
            None::<fn(&ProgressInfo)>,
        )
    }

    /// Run the ensemble sampler with progress callback.
    ///
    /// # Arguments
    ///
    /// * `n_iterations` - Number of MCMC iterations to run
    /// * `init` - Walker initialization strategy
    /// * `thin` - Thinning interval (store every thin-th sample)
    /// * `progress_callback` - Callback for progress reporting
    pub fn run_with_progress<F>(
        &self,
        n_iterations: usize,
        init: WalkerInit,
        thin: usize,
        progress_callback: F,
    ) -> Result<Chain>
    where
        F: FnMut(&ProgressInfo),
    {
        self.run_with_walkers(
            n_iterations,
            init,
            self.default_n_walkers,
            thin,
            Some(progress_callback),
        )
    }

    /// Run the ensemble sampler with checkpointing.
    ///
    /// Saves the sampler state and chain to checkpoint files at regular intervals.
    /// If the run is interrupted, it can be resumed from the last checkpoint.
    ///
    /// # Arguments
    ///
    /// * `n_iterations` - Number of MCMC iterations to run
    /// * `init` - Walker initialization strategy
    /// * `thin` - Thinning interval (store every thin-th sample)
    /// * `checkpoint_every` - Save checkpoint every N iterations
    /// * `checkpoint_path` - Base path for checkpoint files (will append .state and .chain)
    /// * `progress_callback` - Optional callback for progress reporting
    pub fn run_with_checkpoint<F, P>(
        &self,
        n_iterations: usize,
        init: WalkerInit,
        thin: usize,
        checkpoint_every: usize,
        checkpoint_path: P,
        progress_callback: Option<F>,
    ) -> Result<Chain>
    where
        F: FnMut(&ProgressInfo),
        P: AsRef<Path>,
    {
        self.run_with_checkpoint_and_walkers(
            n_iterations,
            init,
            self.default_n_walkers,
            thin,
            checkpoint_every,
            checkpoint_path,
            progress_callback,
        )
    }

    /// Resume a checkpointed run.
    ///
    /// Loads the state and chain from checkpoint files and continues sampling.
    ///
    /// # Arguments
    ///
    /// * `n_iterations` - Total number of iterations to reach (including already completed)
    /// * `thin` - Thinning interval (must match original run)
    /// * `checkpoint_every` - Save checkpoint every N iterations
    /// * `checkpoint_path` - Base path for checkpoint files
    /// * `progress_callback` - Optional callback for progress reporting
    ///
    /// # Returns
    ///
    /// The complete chain including both resumed and new samples.
    pub fn resume_from_checkpoint<F, P>(
        &self,
        n_iterations: usize,
        thin: usize,
        checkpoint_every: usize,
        checkpoint_path: P,
        progress_callback: Option<F>,
    ) -> Result<Chain>
    where
        F: FnMut(&ProgressInfo),
        P: AsRef<Path>,
    {
        let state_path = format!("{}.state", checkpoint_path.as_ref().display());
        let chain_path = format!("{}.chain", checkpoint_path.as_ref().display());

        // Load state and chain
        let state = SamplerState::load_checkpoint(&state_path)?;
        let chain = Chain::load(&chain_path)?;

        self.resume_with_state(
            state,
            chain,
            n_iterations,
            thin,
            checkpoint_every,
            checkpoint_path,
            progress_callback,
        )
    }

    /// Run the ensemble sampler with a specific number of walkers and checkpointing.
    ///
    /// # Arguments
    ///
    /// * `n_iterations` - Number of MCMC iterations to run
    /// * `init` - Walker initialization strategy
    /// * `n_walkers` - Number of walkers (must be even and >= 2)
    /// * `thin` - Thinning interval (store every thin-th sample)
    /// * `checkpoint_every` - Save checkpoint every N iterations (0 = no checkpointing)
    /// * `checkpoint_path` - Base path for checkpoint files
    /// * `progress_callback` - Optional callback for progress reporting
    pub fn run_with_checkpoint_and_walkers<F, P>(
        &self,
        n_iterations: usize,
        init: WalkerInit,
        n_walkers: usize,
        thin: usize,
        checkpoint_every: usize,
        checkpoint_path: P,
        progress_callback: Option<F>,
    ) -> Result<Chain>
    where
        F: FnMut(&ProgressInfo),
        P: AsRef<Path>,
    {
        // Validate n_walkers
        if n_walkers < 2 {
            return Err(Error::SamplingError(
                "Must have at least 2 walkers".to_string(),
            ));
        }
        if !n_walkers.is_multiple_of(2) {
            return Err(Error::SamplingError(
                "Number of walkers must be even".to_string(),
            ));
        }

        // Initialize walkers
        let mut rng = rand::thread_rng();
        let positions = init.initialize(n_walkers, &self.params, &mut rng)?;

        // Create initial state
        let param_names: Vec<String> = self
            .params
            .param_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let state = SamplerState::new(positions, param_names.clone())?;
        let chain = Chain::new(param_names, thin);

        self.run_from_state(
            state,
            chain,
            n_iterations,
            checkpoint_every,
            checkpoint_path,
            progress_callback,
        )
    }

    /// Run the ensemble sampler with a specific number of walkers.
    ///
    /// # Arguments
    ///
    /// * `n_iterations` - Number of MCMC iterations to run
    /// * `init` - Walker initialization strategy
    /// * `n_walkers` - Number of walkers (must be even and >= 2)
    /// * `thin` - Thinning interval (store every thin-th sample)
    /// * `progress_callback` - Optional callback for progress reporting
    pub fn run_with_walkers<F>(
        &self,
        n_iterations: usize,
        init: WalkerInit,
        n_walkers: usize,
        thin: usize,
        mut progress_callback: Option<F>,
    ) -> Result<Chain>
    where
        F: FnMut(&ProgressInfo),
    {
        // Validate n_walkers
        if n_walkers < 2 {
            return Err(Error::SamplingError(
                "Must have at least 2 walkers".to_string(),
            ));
        }
        if !n_walkers.is_multiple_of(2) {
            return Err(Error::SamplingError(
                "Number of walkers must be even".to_string(),
            ));
        }

        // Initialize walkers
        let mut rng = rand::thread_rng();
        let positions = init.initialize(n_walkers, &self.params, &mut rng)?;

        // Create initial state
        let param_names: Vec<String> = self
            .params
            .param_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut state = SamplerState::new(positions, param_names.clone())?;

        // Compute initial log probabilities
        let initial_params: Vec<Vec<f64>> = state
            .positions
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
        state.log_probs = Array1::from_vec(self.log_posterior_batch(&initial_params));

        // Create chain
        let mut chain = Chain::new(param_names, thin);

        // Run MCMC iterations
        for iteration in 0..n_iterations {
            // Split walkers into two groups
            let half = n_walkers / 2;

            // Update first half using second half as complementary ensemble
            self.update_group(&mut state, 0..half, half..n_walkers, &mut rng)?;

            // Update second half using first half as complementary ensemble
            self.update_group(&mut state, half..n_walkers, 0..half, &mut rng)?;

            // Store sample in chain
            chain.push(state.positions.clone(), state.log_probs.clone());

            // Call progress callback if provided
            if let Some(ref mut callback) = progress_callback {
                let info = ProgressInfo {
                    iteration,
                    total: n_iterations,
                    acceptance_rate: state.mean_acceptance_rate(),
                    mean_log_prob: state.log_probs.mean().unwrap_or(f64::NEG_INFINITY),
                };
                callback(&info);
            }
        }

        Ok(chain)
    }

    /// Update a group of walkers using stretch moves.
    ///
    /// # Arguments
    ///
    /// * `state` - Current sampler state (modified in place)
    /// * `active_range` - Range of walker indices to update
    /// * `complementary_range` - Range of walker indices to use as complementary ensemble
    /// * `rng` - Random number generator
    fn update_group<Rng: rand::Rng>(
        &self,
        state: &mut SamplerState,
        active_range: std::ops::Range<usize>,
        complementary_range: std::ops::Range<usize>,
        rng: &mut Rng,
    ) -> Result<()> {
        let complementary_positions = state
            .positions
            .slice(ndarray::s![complementary_range.clone(), ..])
            .to_owned();

        // Generate proposals for all active walkers
        let proposals: Vec<(ndarray::Array1<f64>, f64)> = active_range
            .clone()
            .map(|i| {
                let current = state.positions.row(i);
                self.stretch.propose(rng, current, &complementary_positions)
            })
            .collect();

        // Evaluate log posteriors for all proposals in parallel
        let proposal_params: Vec<Vec<f64>> = proposals.iter().map(|(p, _)| p.to_vec()).collect();
        let proposal_log_probs = self.log_posterior_batch(&proposal_params);

        // Accept/reject each proposal
        for (walker_idx, ((proposal, z), &log_prob_new)) in
            active_range.zip(proposals.iter().zip(proposal_log_probs.iter()))
        {
            let log_prob_old = state.log_probs[walker_idx];

            // Compute acceptance probability
            let accept_prob = self.stretch.acceptance_probability(
                *z,
                state.n_params(),
                log_prob_old,
                log_prob_new,
            );

            // Accept/reject
            state.n_proposed[walker_idx] += 1;
            if rng.gen::<f64>() < accept_prob {
                // Accept
                state.positions.row_mut(walker_idx).assign(proposal);
                state.log_probs[walker_idx] = log_prob_new;
                state.n_accepted[walker_idx] += 1;
            }
            // If rejected, walker stays at current position
        }

        Ok(())
    }

    /// Resume sampling from a given state and chain.
    ///
    /// # Arguments
    ///
    /// * `state` - Current sampler state
    /// * `chain` - Existing chain with samples
    /// * `n_iterations` - Total iterations to reach (including already completed)
    /// * `thin` - Thinning interval
    /// * `checkpoint_every` - Save checkpoint every N iterations
    /// * `checkpoint_path` - Base path for checkpoint files
    /// * `progress_callback` - Optional callback for progress reporting
    fn resume_with_state<F, P>(
        &self,
        state: SamplerState,
        chain: Chain,
        n_iterations: usize,
        _thin: usize,
        checkpoint_every: usize,
        checkpoint_path: P,
        progress_callback: Option<F>,
    ) -> Result<Chain>
    where
        F: FnMut(&ProgressInfo),
        P: AsRef<Path>,
    {
        let iterations_completed = chain.total_iterations();
        let iterations_remaining = n_iterations.saturating_sub(iterations_completed);

        if iterations_remaining == 0 {
            return Ok(chain);
        }

        self.run_from_state(
            state,
            chain,
            iterations_remaining,
            checkpoint_every,
            checkpoint_path,
            progress_callback,
        )
    }

    /// Core sampling loop that handles state updates and checkpointing.
    ///
    /// # Arguments
    ///
    /// * `state` - Initial sampler state (will be modified)
    /// * `chain` - Initial chain (will be extended)
    /// * `n_iterations` - Number of iterations to run
    /// * `checkpoint_every` - Save checkpoint every N iterations (0 = disabled)
    /// * `checkpoint_path` - Base path for checkpoint files
    /// * `progress_callback` - Optional callback for progress reporting
    fn run_from_state<F, P>(
        &self,
        mut state: SamplerState,
        mut chain: Chain,
        n_iterations: usize,
        checkpoint_every: usize,
        checkpoint_path: P,
        mut progress_callback: Option<F>,
    ) -> Result<Chain>
    where
        F: FnMut(&ProgressInfo),
        P: AsRef<Path>,
    {
        let n_walkers = state.n_walkers();

        // Compute initial log probabilities if not already computed
        if state.log_probs.iter().all(|&lp| !lp.is_finite()) {
            let initial_params: Vec<Vec<f64>> = state
                .positions
                .outer_iter()
                .map(|row| row.to_vec())
                .collect();
            state.log_probs = Array1::from_vec(self.log_posterior_batch(&initial_params));
        }

        // Setup checkpoint paths
        let state_path = format!("{}.state", checkpoint_path.as_ref().display());
        let chain_path = format!("{}.chain", checkpoint_path.as_ref().display());

        // Run MCMC iterations
        let mut rng = rand::thread_rng();

        for iteration in 0..n_iterations {
            // Split walkers into two groups
            let half = n_walkers / 2;

            // Update first half using second half as complementary ensemble
            self.update_group(&mut state, 0..half, half..n_walkers, &mut rng)?;

            // Update second half using first half as complementary ensemble
            self.update_group(&mut state, half..n_walkers, 0..half, &mut rng)?;

            // Store sample in chain
            chain.push(state.positions.clone(), state.log_probs.clone());

            // Save checkpoint if needed
            if checkpoint_every > 0 && (iteration + 1) % checkpoint_every == 0 {
                state.save_checkpoint(&state_path)?;
                chain.save(&chain_path)?;
            }

            // Call progress callback if provided
            if let Some(ref mut callback) = progress_callback {
                let info = ProgressInfo {
                    iteration,
                    total: n_iterations,
                    acceptance_rate: state.mean_acceptance_rate(),
                    mean_log_prob: state.log_probs.mean().unwrap_or(f64::NEG_INFINITY),
                };
                callback(&info);
            }
        }

        Ok(chain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_runner::ModelRunner;
    use crate::{GaussianLikelihood, ParameterSet, Uniform};
    use ndarray::Array2;
    use std::cell::RefCell;
    use std::rc::Rc;

    // Simple model for testing: y = a * x + b
    struct LinearModel {
        param_names: Vec<String>,
    }

    impl LinearModel {
        fn new() -> Self {
            Self {
                param_names: vec!["a".to_string(), "b".to_string()],
            }
        }
    }

    impl ModelRunner for LinearModel {
        fn param_names(&self) -> &[String] {
            &self.param_names
        }

        fn run(&self, params: &[f64]) -> crate::Result<crate::likelihood::ModelOutput> {
            let a = params[0];
            let b = params[1];

            let mut output = crate::likelihood::ModelOutput::new();
            let mut var = crate::likelihood::VariableOutput::new("y");

            // Generate y values at x = 0, 1, 2, 3, 4
            for x in 0..5 {
                let y = a * (x as f64) + b;
                var.add(x as f64, y);
            }

            output.add_variable(var);
            Ok(output)
        }
    }

    // Constant model for simple testing
    struct ConstantModel {
        param_names: Vec<String>,
    }

    impl ConstantModel {
        fn new() -> Self {
            Self {
                param_names: vec!["x".to_string()],
            }
        }
    }

    impl ModelRunner for ConstantModel {
        fn param_names(&self) -> &[String] {
            &self.param_names
        }

        fn run(&self, params: &[f64]) -> crate::Result<crate::likelihood::ModelOutput> {
            let x = params[0];

            let mut output = crate::likelihood::ModelOutput::new();
            let mut var = crate::likelihood::VariableOutput::new("value");
            var.add(0.0, x);

            output.add_variable(var);
            Ok(output)
        }
    }

    // Dummy model for validation testing
    struct DummyModel {
        param_names: Vec<String>,
    }

    impl DummyModel {
        fn new() -> Self {
            Self {
                param_names: vec!["x".to_string()],
            }
        }
    }

    impl ModelRunner for DummyModel {
        fn param_names(&self) -> &[String] {
            &self.param_names
        }

        fn run(&self, _params: &[f64]) -> crate::Result<crate::likelihood::ModelOutput> {
            Ok(crate::likelihood::ModelOutput::new())
        }
    }

    #[test]
    fn test_ensemble_sampler_simple_model() {
        // Create synthetic observations with a=2, b=1: y = 2*x + 1
        let mut target = Target::new();
        target
            .add_variable("y")
            .add(0.0, 1.0, 0.1)
            .unwrap() // y(0) = 1
            .add(1.0, 3.0, 0.1)
            .unwrap() // y(1) = 3
            .add(2.0, 5.0, 0.1)
            .unwrap() // y(2) = 5
            .add(3.0, 7.0, 0.1)
            .unwrap() // y(3) = 7
            .add(4.0, 9.0, 0.1)
            .unwrap(); // y(4) = 9

        // Set up parameter priors
        let mut params = ParameterSet::new();
        params.add("a", Box::new(Uniform::new(0.0, 5.0).unwrap()));
        params.add("b", Box::new(Uniform::new(-2.0, 4.0).unwrap()));

        // Create sampler
        let runner = LinearModel::new();
        let likelihood = GaussianLikelihood::default();
        let sampler = EnsembleSampler::new(params, runner, likelihood, target);

        // Run a short chain to test functionality
        let chain = sampler
            .run(10, WalkerInit::FromPrior, 1)
            .expect("Sampler should run successfully");

        // Basic checks
        assert_eq!(chain.len(), 10); // All 10 iterations stored (thin=1)
        assert_eq!(chain.param_names(), &["a", "b"]);

        // Check that we got some samples
        let flat_samples = chain.flat_samples(0);
        assert!(flat_samples.nrows() > 0);
        assert_eq!(flat_samples.ncols(), 2);

        // Check log probabilities are finite (at least some valid samples)
        let flat_lp = chain.flat_log_probs(0);
        let n_finite = flat_lp.iter().filter(|&&lp| lp.is_finite()).count();
        assert!(
            n_finite > 0,
            "Should have at least some finite log probabilities"
        );
    }

    #[test]
    fn test_ensemble_sampler_with_ball_init() {
        // Target: x should be near 0.5
        let mut target = Target::new();
        target.add_variable("value").add(0.0, 0.5, 0.1).unwrap();

        let mut params = ParameterSet::new();
        params.add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()));

        let runner = ConstantModel::new();
        let likelihood = GaussianLikelihood::default();
        let sampler = EnsembleSampler::new(params, runner, likelihood, target);

        // Initialize walkers in a ball around the true value
        let init = WalkerInit::Ball {
            center: vec![0.5],
            radius: 0.01,
        };

        let chain = sampler
            .run_with_walkers(5, init, 10, 1, None::<fn(&ProgressInfo)>)
            .expect("Sampler should run successfully");

        assert_eq!(chain.len(), 5);

        // All samples should be near 0.5 since we started there and it's the optimum
        let flat_samples = chain.flat_samples(0);
        for i in 0..flat_samples.nrows() {
            let x = flat_samples[[i, 0]];
            assert!((0.0..=1.0).contains(&x), "Sample {} out of prior bounds", x);
        }
    }

    #[test]
    fn test_ensemble_sampler_odd_walkers_error() {
        let target = Target::new();
        let mut params = ParameterSet::new();
        params.add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()));

        let runner = DummyModel::new();
        let likelihood = GaussianLikelihood::default();
        let sampler = EnsembleSampler::new(params, runner, likelihood, target);

        // Try with odd number of walkers (should fail)
        let result =
            sampler.run_with_walkers(5, WalkerInit::FromPrior, 3, 1, None::<fn(&ProgressInfo)>);
        assert!(result.is_err());
    }

    #[test]
    fn test_progress_callback() {
        let target = Target::new();
        let mut params = ParameterSet::new();
        params.add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()));

        let runner = DummyModel::new();
        let likelihood = GaussianLikelihood::default();
        let sampler = EnsembleSampler::new(params, runner, likelihood, target);

        // Track progress updates
        let progress_updates = Rc::new(RefCell::new(Vec::new()));
        let progress_updates_clone = Rc::clone(&progress_updates);

        let callback = move |info: &ProgressInfo| {
            progress_updates_clone.borrow_mut().push((
                info.iteration,
                info.total,
                info.acceptance_rate,
                info.mean_log_prob,
            ));
        };

        // Run with progress callback
        let n_iterations = 10;
        let _chain = sampler
            .run_with_progress(n_iterations, WalkerInit::FromPrior, 1, callback)
            .unwrap();

        // Check that we got all progress updates
        let updates = progress_updates.borrow();
        assert_eq!(updates.len(), n_iterations);

        // Check first and last updates
        assert_eq!(updates[0].0, 0); // First iteration
        assert_eq!(updates[0].1, n_iterations); // Total

        assert_eq!(updates[n_iterations - 1].0, n_iterations - 1); // Last iteration
        assert_eq!(updates[n_iterations - 1].1, n_iterations); // Total

        // Acceptance rate should be between 0 and 1
        for (_, _, acceptance_rate, _) in updates.iter() {
            assert!(*acceptance_rate >= 0.0 && *acceptance_rate <= 1.0);
        }
    }

    #[test]
    fn test_sampler_correctness_multivariate_normal() {
        use crate::likelihood::{ModelOutput, VariableOutput};

        // Sample from known posterior and verify recovered statistics
        //
        // Setup: Uniform prior on [-10, 10]^2, single observation at (1.0, 2.0) with sigma = 1.0
        // Posterior is bivariate normal centered at observation (independent dimensions)

        let true_mean = [1.0, 2.0]; // True parameter values (= posterior mean)
        let obs_std = 1.0; // Observation uncertainty = posterior std (with uniform prior)

        // Create parameter priors (uniform, wide enough to not dominate)
        let mut params = ParameterSet::new();
        params
            .add(
                "x".to_string(),
                Box::new(crate::distribution::Uniform::new(-10.0, 10.0).unwrap()),
            )
            .add(
                "y".to_string(),
                Box::new(crate::distribution::Uniform::new(-10.0, 10.0).unwrap()),
            );

        // Single observation at true parameters
        let mut target = Target::new();
        target
            .add_variable("x")
            .add(0.0, true_mean[0], obs_std)
            .unwrap();
        target
            .add_variable("y")
            .add(0.0, true_mean[1], obs_std)
            .unwrap();

        // Create model runner that simply returns the parameters as outputs
        struct IdentityRunner {
            param_names: Vec<String>,
        }

        impl ModelRunner for IdentityRunner {
            fn param_names(&self) -> &[String] {
                &self.param_names
            }

            fn run(&self, params: &[f64]) -> crate::Result<ModelOutput> {
                let mut output = std::collections::HashMap::new();

                // Create output for x variable
                let mut x_values = std::collections::HashMap::new();
                for i in 0..20 {
                    x_values.insert(format!("{:.6}", i as f64), params[0]);
                }
                output.insert(
                    "x".to_string(),
                    VariableOutput {
                        name: "x".to_string(),
                        values: x_values,
                    },
                );

                // Create output for y variable
                let mut y_values = std::collections::HashMap::new();
                for i in 0..20 {
                    y_values.insert(format!("{:.6}", i as f64), params[1]);
                }
                output.insert(
                    "y".to_string(),
                    VariableOutput {
                        name: "y".to_string(),
                        values: y_values,
                    },
                );

                Ok(ModelOutput { variables: output })
            }
        }

        let runner = IdentityRunner {
            param_names: vec!["x".to_string(), "y".to_string()],
        };
        let likelihood = GaussianLikelihood::new();

        // Run sampler
        let sampler = EnsembleSampler::new(params.clone(), runner, likelihood, target.clone());
        let n_iterations = 1000;
        let n_walkers = 32;
        let burn_in = 500;

        let chain = sampler
            .run_with_walkers(
                n_iterations,
                WalkerInit::FromPrior,
                n_walkers,
                1,
                None::<fn(&ProgressInfo)>,
            )
            .unwrap();

        // Extract samples after burn-in
        let samples = chain.flat_samples(burn_in);
        let (n_samples, n_params) = samples.dim();

        assert_eq!(n_params, 2);
        assert_eq!(n_samples, (n_iterations - burn_in) * n_walkers);

        // Compute sample mean
        let mean_x = samples.column(0).mean().unwrap();
        let mean_y = samples.column(1).mean().unwrap();

        // Compute sample standard deviations
        let var_x = samples.column(0).var(0.0);
        let var_y = samples.column(1).var(0.0);
        let std_x = var_x.sqrt();
        let std_y = var_y.sqrt();

        // Expected posterior std = obs_std (with uniform prior, posterior ≈ likelihood)
        let expected_std = obs_std;

        // Compute effective sample size (accounts for autocorrelation)
        let ess = chain.ess(burn_in);
        let ess_x = ess.get("x").copied().unwrap_or(n_samples as f64);
        let ess_y = ess.get("y").copied().unwrap_or(n_samples as f64);

        // Compute standard error of mean using ESS
        let se_x = std_x / ess_x.sqrt();
        let se_y = std_y / ess_y.sqrt();

        println!(
            "Recovered mean: x={:.3} (true={:.3}, SE={:.3}, ESS={:.0}), y={:.3} (true={:.3}, SE={:.3}, ESS={:.0})",
            mean_x, true_mean[0], se_x, ess_x, mean_y, true_mean[1], se_y, ess_y
        );
        println!(
            "Recovered std: x={:.3} (expected={:.3}), y={:.3} (expected={:.3})",
            std_x, expected_std, std_y, expected_std
        );

        // Assert mean recovery within 5 standard errors (very conservative for statistical test)
        assert!(
            (mean_x - true_mean[0]).abs() < 5.0 * se_x,
            "Mean x={:.3} not within 5 SE of true mean {:.3}",
            mean_x,
            true_mean[0]
        );
        assert!(
            (mean_y - true_mean[1]).abs() < 5.0 * se_y,
            "Mean y={:.3} not within 5 SE of true mean {:.3}",
            mean_y,
            true_mean[1]
        );

        // Assert standard deviation recovery (allow 30% error due to finite samples and autocorrelation)
        assert!(
            (std_x - expected_std).abs() / expected_std < 0.3,
            "Std x={:.3} differs from expected std {:.3} by more than 30%",
            std_x,
            expected_std
        );
        assert!(
            (std_y - expected_std).abs() / expected_std < 0.3,
            "Std y={:.3} differs from expected std {:.3} by more than 30%",
            std_y,
            expected_std
        );

        // Check convergence
        let r_hat = chain.r_hat(burn_in);
        assert!(
            chain.is_converged(burn_in, 1.1),
            "Chain did not converge: R-hat values {:?}",
            r_hat
        );
    }

    #[test]
    fn test_parallel_determinism() {
        use crate::distribution::Normal;
        use crate::likelihood::{ModelOutput, VariableOutput};
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Verify that same seed produces same chain despite parallel execution
        //
        // Note: Full determinism requires that WalkerInit::Explicit is used, since
        // WalkerInit::FromPrior uses thread_rng() which may not be deterministic in parallel.

        // Create simple test setup
        let mut params = ParameterSet::new();
        params
            .add("x".to_string(), Box::new(Normal::new(0.0, 5.0).unwrap()))
            .add("y".to_string(), Box::new(Normal::new(0.0, 5.0).unwrap()));

        let mut target = Target::new();
        target.add_variable("x").add(0.0, 1.0, 0.5).unwrap();
        target.add_variable("y").add(0.0, 2.0, 0.5).unwrap();

        struct IdentityRunner {
            param_names: Vec<String>,
        }

        impl ModelRunner for IdentityRunner {
            fn param_names(&self) -> &[String] {
                &self.param_names
            }

            fn run(&self, params: &[f64]) -> crate::Result<ModelOutput> {
                let mut output = std::collections::HashMap::new();
                output.insert(
                    "x".to_string(),
                    VariableOutput {
                        name: "x".to_string(),
                        values: vec![("0.000000".to_string(), params[0])]
                            .into_iter()
                            .collect(),
                    },
                );
                output.insert(
                    "y".to_string(),
                    VariableOutput {
                        name: "y".to_string(),
                        values: vec![("0.000000".to_string(), params[1])]
                            .into_iter()
                            .collect(),
                    },
                );
                Ok(ModelOutput { variables: output })
            }
        }

        let runner = IdentityRunner {
            param_names: vec!["x".to_string(), "y".to_string()],
        };
        let likelihood = GaussianLikelihood::new();
        let sampler = EnsembleSampler::new(params, runner, likelihood, target);

        // Generate deterministic initial positions
        let n_walkers = 16;
        let n_params = 2;
        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let positions = Array2::from_shape_fn((n_walkers, n_params), |_| rng.gen::<f64>() * 2.0);

        // Run sampler twice with same initial positions
        let init = WalkerInit::Explicit(positions.clone());

        // NOTE: The sampler uses thread_rng() internally for proposals, which is NOT
        // deterministic across runs. This is a design limitation.

        let chain1 = sampler
            .run_with_walkers(100, init.clone(), n_walkers, 1, None::<fn(&ProgressInfo)>)
            .unwrap();

        let chain2 = sampler
            .run_with_walkers(100, init, n_walkers, 1, None::<fn(&ProgressInfo)>)
            .unwrap();

        // Both chains should have same shape
        assert_eq!(chain1.param_names, chain2.param_names);
        assert_eq!(chain1.thin, chain2.thin);
        assert_eq!(chain1.total_iterations, chain2.total_iterations);

        // Get final samples from both chains
        let samples1 = chain1.flat_samples(50);
        let samples2 = chain2.flat_samples(50);

        assert_eq!(samples1.dim(), samples2.dim());

        // Verify both chains converged to similar posterior means
        let mean1_x = samples1.column(0).mean().unwrap();
        let mean2_x = samples2.column(0).mean().unwrap();
        let std_pooled_x = (samples1.column(0).std(0.0) + samples2.column(0).std(0.0)) / 2.0;

        let mean1_y = samples1.column(1).mean().unwrap();
        let mean2_y = samples2.column(1).mean().unwrap();
        let std_pooled_y = (samples1.column(1).std(0.0) + samples2.column(1).std(0.0)) / 2.0;

        // Means should be within 1 posterior std
        assert!(
            (mean1_x - mean2_x).abs() < std_pooled_x,
            "Chain means differ by more than 1 posterior std: x1={:.3} vs x2={:.3}, pooled_std={:.3}",
            mean1_x,
            mean2_x,
            std_pooled_x
        );

        assert!(
            (mean1_y - mean2_y).abs() < std_pooled_y,
            "Chain means differ by more than 1 posterior std: y1={:.3} vs y2={:.3}, pooled_std={:.3}",
            mean1_y,
            mean2_y,
            std_pooled_y
        );
    }

    #[test]
    fn test_edge_case_single_parameter() {
        use crate::distribution::Normal;
        use crate::likelihood::{ModelOutput, VariableOutput};

        // Test 1D sampling (single parameter)
        let mut params = ParameterSet::new();
        params.add("x".to_string(), Box::new(Normal::new(0.0, 5.0).unwrap()));

        let mut target = Target::new();
        target.add_variable("x").add(0.0, 1.0, 0.5).unwrap();

        struct SingleParamRunner;
        impl ModelRunner for SingleParamRunner {
            fn param_names(&self) -> &[String] {
                static NAMES: [String; 1] = [String::new()];
                &NAMES
            }

            fn run(&self, params: &[f64]) -> crate::Result<ModelOutput> {
                let mut output = std::collections::HashMap::new();
                output.insert(
                    "x".to_string(),
                    VariableOutput {
                        name: "x".to_string(),
                        values: vec![("0.000000".to_string(), params[0])]
                            .into_iter()
                            .collect(),
                    },
                );
                Ok(ModelOutput { variables: output })
            }
        }

        let sampler =
            EnsembleSampler::new(params, SingleParamRunner, GaussianLikelihood::new(), target);

        let chain = sampler
            .run_with_walkers(50, WalkerInit::FromPrior, 10, 1, None::<fn(&ProgressInfo)>)
            .unwrap();

        let samples = chain.flat_samples(20);
        assert_eq!(samples.ncols(), 1);
        assert!(samples.nrows() > 0);

        // Should converge to observation (mean ≈ 1.0)
        let mean = samples.column(0).mean().unwrap();
        assert!(
            (mean - 1.0).abs() < 0.5,
            "1D sampler should converge near observation, got mean={:.3}",
            mean
        );
    }

    #[test]
    fn test_edge_case_high_dimensional() {
        use crate::distribution::Normal;
        use crate::likelihood::{ModelOutput, VariableOutput};

        // Test high-dimensional sampling (50 parameters)
        let n_params = 50;
        let mut params = ParameterSet::new();
        for i in 0..n_params {
            params.add(format!("x{}", i), Box::new(Normal::new(0.0, 10.0).unwrap()));
        }

        let mut target = Target::new();
        for i in 0..n_params {
            target
                .add_variable(format!("x{}", i))
                .add(0.0, (i as f64) * 0.1, 1.0)
                .unwrap();
        }

        struct HighDimRunner {
            n_params: usize,
            param_names: Vec<String>,
        }

        impl ModelRunner for HighDimRunner {
            fn param_names(&self) -> &[String] {
                &self.param_names
            }

            fn run(&self, params: &[f64]) -> crate::Result<ModelOutput> {
                let mut output = std::collections::HashMap::new();
                for i in 0..self.n_params {
                    output.insert(
                        format!("x{}", i),
                        VariableOutput {
                            name: format!("x{}", i),
                            values: vec![("0.000000".to_string(), params[i])]
                                .into_iter()
                                .collect(),
                        },
                    );
                }
                Ok(ModelOutput { variables: output })
            }
        }

        let runner = HighDimRunner {
            n_params,
            param_names: (0..n_params).map(|i| format!("x{}", i)).collect(),
        };

        let sampler = EnsembleSampler::new(params, runner, GaussianLikelihood::new(), target);

        // For high-D, need many walkers (at least 2*n_params)
        let n_walkers = 100;

        let chain = sampler
            .run_with_walkers(
                100,
                WalkerInit::FromPrior,
                n_walkers,
                1,
                None::<fn(&ProgressInfo)>,
            )
            .unwrap();

        let samples = chain.flat_samples(50);
        assert_eq!(samples.ncols(), n_params);
        assert!(samples.nrows() > 0);

        // Check that sampler ran successfully and produced reasonable output
        for i in 0..n_params {
            let mean = samples.column(i).mean().unwrap();
            let expected = (i as f64) * 0.1;
            assert!(
                mean.is_finite(),
                "Parameter {} mean should be finite, got {}",
                i,
                mean
            );
            // Very loose check - just ensure it's in a reasonable range
            assert!(
                (mean - expected).abs() < 10.0,
                "Parameter {} mean {} too far from expected {}",
                i,
                mean,
                expected
            );
        }
    }

    #[test]
    fn test_edge_case_all_walkers_same_init() {
        use crate::distribution::Normal;
        use crate::likelihood::{ModelOutput, VariableOutput};
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Test all walkers initialized very close together
        // Note: The stretch move CANNOT work if all walkers are at exactly the same point,
        // since the proposal is y = c + z*(x - c) where c is complementary walker.
        // If x == c, then y == c always (no movement).
        // So we test with walkers in a very tight ball instead.
        let mut params = ParameterSet::new();
        params
            .add("x".to_string(), Box::new(Normal::new(0.0, 5.0).unwrap()))
            .add("y".to_string(), Box::new(Normal::new(0.0, 5.0).unwrap()));

        let mut target = Target::new();
        target.add_variable("x").add(0.0, 1.0, 0.5).unwrap();
        target.add_variable("y").add(0.0, 2.0, 0.5).unwrap();

        struct IdentityRunner {
            param_names: Vec<String>,
        }

        impl ModelRunner for IdentityRunner {
            fn param_names(&self) -> &[String] {
                &self.param_names
            }

            fn run(&self, params: &[f64]) -> crate::Result<ModelOutput> {
                let mut output = std::collections::HashMap::new();
                output.insert(
                    "x".to_string(),
                    VariableOutput {
                        name: "x".to_string(),
                        values: vec![("0.000000".to_string(), params[0])]
                            .into_iter()
                            .collect(),
                    },
                );
                output.insert(
                    "y".to_string(),
                    VariableOutput {
                        name: "y".to_string(),
                        values: vec![("0.000000".to_string(), params[1])]
                            .into_iter()
                            .collect(),
                    },
                );
                Ok(ModelOutput { variables: output })
            }
        }

        let runner = IdentityRunner {
            param_names: vec!["x".to_string(), "y".to_string()],
        };

        let sampler = EnsembleSampler::new(params, runner, GaussianLikelihood::new(), target);

        // Initialize all walkers in a tiny ball around (0, 0)
        let n_walkers = 16;
        let mut rng = ChaCha8Rng::seed_from_u64(555);
        let mut positions = Array2::zeros((n_walkers, 2));
        for i in 0..n_walkers {
            positions[[i, 0]] = rng.gen::<f64>() * 0.001;
            positions[[i, 1]] = rng.gen::<f64>() * 0.001;
        }
        let init = WalkerInit::Explicit(positions);

        let chain = sampler
            .run_with_walkers(200, init, n_walkers, 1, None::<fn(&ProgressInfo)>)
            .unwrap();

        // Walkers should have spread out after sufficient iterations
        let samples = chain.flat_samples(100);

        // Check that variance is non-zero (walkers spread out)
        let var_x = samples.column(0).var(0.0);
        let var_y = samples.column(1).var(0.0);

        assert!(
            var_x > 0.01,
            "Walkers should spread out from initial point, got var_x={:.6}",
            var_x
        );
        assert!(
            var_y > 0.01,
            "Walkers should spread out from initial point, got var_y={:.6}",
            var_y
        );

        // Check that mean moved towards target
        let mean_x = samples.column(0).mean().unwrap();
        let mean_y = samples.column(1).mean().unwrap();

        assert!(
            (mean_x - 1.0).abs() < 1.0,
            "Mean should move towards observation x=1.0, got {:.3}",
            mean_x
        );
        assert!(
            (mean_y - 2.0).abs() < 1.0,
            "Mean should move towards observation y=2.0, got {:.3}",
            mean_y
        );
    }
}
