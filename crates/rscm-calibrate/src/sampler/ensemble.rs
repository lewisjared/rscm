//! Affine-invariant ensemble sampler implementation.
//!
//! This module contains the main `EnsembleSampler` struct that orchestrates
//! MCMC sampling using the Goodman & Weare (2010) stretch move algorithm.

use super::{Chain, ProgressInfo, SamplerState, StretchMove, WalkerInit};
use crate::{
    likelihood::{log_posterior, LikelihoodFn, ModelOutput},
    model_runner::{ModelRunner, ParameterOrder},
    no_valid_reason,
    parameter_set::ParameterSet,
    rng_from_seed,
    target::Target,
    Error, Failures, Result,
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
/// The sampler maintains N walkers (where N ≥ 2 × n_parameters) that evolve in parallel:
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
/// 2. **Initialise walkers** from prior, around a point, or explicitly
/// 3. **Run sampling** with optional progress callbacks and checkpointing
/// 4. **Analyse chain** using built-in diagnostics (R-hat, ESS, autocorrelation)
///
/// # Example
///
/// ```ignore
/// use rscm_calibrate::{EnsembleSampler, ParameterSet, Target, WalkerInit};
/// use rscm_calibrate::likelihood::GaussianLikelihood;
///
/// // Define priors
/// let mut parameters = ParameterSet::new();
/// parameters.add("sensitivity", Box::new(Uniform::new(0.5, 1.5).unwrap()));
/// parameters.add("offset", Box::new(Normal::new(0.0, 0.1).unwrap()));
///
/// // Define observations
/// let mut target = Target::new();
/// target.add_variable("Surface Air Temperature Change")
///     .add(2020.0, 1.2, 0.1).unwrap()
///     .add(2021.0, 1.3, 0.1).unwrap();
///
/// // Create sampler
/// let runner = MyModelRunner::new();
/// let likelihood = GaussianLikelihood::new();
/// let sampler = EnsembleSampler::new(parameters, runner, likelihood, target)?;
///
/// // Run MCMC sampling (1000 iterations, initialise from prior, no thinning)
/// let chain = sampler.run(1000, WalkerInit::FromPrior, 1)?;
///
/// // Check convergence
/// let r_hat = chain.r_hat(500)?;  // Discard first 500 samples as burn-in
/// if chain.is_converged(500, 1.1)? {
///     println!("Chain converged!");
/// }
///
/// // Extract posterior samples
/// let samples = chain.to_parameter_map(500);  // Discard burn-in
/// ```
///
/// # References
///
/// Goodman, J., & Weare, J. (2010). Ensemble samplers with affine invariance.
/// Communications in Applied Mathematics and Computational Science, 5(1), 65-80.
pub struct EnsembleSampler<R: ModelRunner, L: LikelihoodFn> {
    /// Parameter set defining the prior distributions
    parameters: ParameterSet,

    /// Model runner for evaluating parameter sets
    runner: R,

    /// Likelihood function for computing log probability
    likelihood: L,

    /// Target observations
    target: Target,

    /// Stretch move configuration
    stretch: StretchMove,

    /// Default number of walkers (2 * n_parameters, minimum 32)
    default_n_walkers: usize,

    /// Maps parameter set order to the runner's parameter order
    order: ParameterOrder,
}

impl<R: ModelRunner + Sync, L: LikelihoodFn + Sync> EnsembleSampler<R, L> {
    /// Create a new ensemble sampler.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Parameter set defining priors
    /// * `runner` - Model runner for evaluating parameter sets
    /// * `likelihood` - Likelihood function
    /// * `target` - Target observations
    ///
    /// # Errors
    ///
    /// Fails unless the runner's `parameter_names` name each parameter exactly once.
    /// The runner may list them in a different order: values are passed to it by name.
    pub fn new(parameters: ParameterSet, runner: R, likelihood: L, target: Target) -> Result<Self> {
        let order = ParameterOrder::new(&parameters, runner.parameter_names())?;
        let default_n_walkers = (2 * parameters.len()).max(32);

        Ok(Self {
            parameters,
            runner,
            likelihood,
            target,
            stretch: StretchMove::default(),
            default_n_walkers,
            order,
        })
    }

    /// Get the model runner.
    pub fn runner(&self) -> &R {
        &self.runner
    }

    /// Create a sampler with custom stretch move parameter.
    pub fn with_stretch_parameter(mut self, a: f64) -> Result<Self> {
        self.stretch = StretchMove::new(a)?;
        Ok(self)
    }

    /// Get the default number of walkers for this sampler.
    pub fn default_n_walkers(&self) -> usize {
        self.default_n_walkers
    }

    /// Compute log posterior for multiple parameter vectors in parallel.
    ///
    /// log_posterior = log_prior + log_likelihood.
    /// A point outside the prior support is `Ok(-inf)`.
    /// A failed prior, model or likelihood evaluation is `Err`.
    /// The model only runs for points inside the prior support.
    fn try_log_posterior_batch(&self, parameter_sets: &[Vec<f64>]) -> Vec<Result<f64>> {
        let in_support: Vec<bool> = parameter_sets
            .iter()
            .map(|p| self.parameters.log_prior(p).is_ok_and(|lp| lp.is_finite()))
            .collect();
        let runner_sets: Vec<Vec<f64>> = parameter_sets
            .iter()
            .zip(&in_support)
            .filter(|(_, &run)| run)
            .map(|(p, _)| self.order.apply(p).into_owned())
            .collect();
        let mut outputs = self.runner.run_batch(&runner_sets).into_iter();
        let outputs: Vec<Option<Result<ModelOutput>>> = in_support
            .iter()
            .map(|&run| if run { outputs.next() } else { None })
            .collect();

        parameter_sets
            .par_iter()
            .zip(outputs.into_par_iter())
            .map(|(parameters, output)| {
                log_posterior(
                    &self.parameters,
                    parameters,
                    || output.expect("the model ran for every point in the prior support"),
                    &self.likelihood,
                    &self.target,
                )
                .map(|(log_posterior, _)| log_posterior)
            })
            .collect()
    }

    /// Log posteriors for proposals, where a failed evaluation rejects the proposal.
    ///
    /// Failures are recorded in `failures`. An interrupt stops the run.
    fn log_posterior_batch(
        &self,
        parameter_sets: &[Vec<f64>],
        failures: &mut Failures,
    ) -> Result<Vec<f64>> {
        self.try_log_posterior_batch(parameter_sets)
            .into_iter()
            .map(|lp| lp.or_else(|e| failures.record(e).map(|()| f64::NEG_INFINITY)))
            .collect()
    }

    /// Log posteriors for the initial walkers.
    ///
    /// Fails if no walker has a finite log posterior, because no proposal could then be accepted.
    fn initial_log_probs(
        &self,
        positions: &ndarray::Array2<f64>,
        failures: &mut Failures,
    ) -> Result<Array1<f64>> {
        let parameters: Vec<Vec<f64>> = positions.outer_iter().map(|row| row.to_vec()).collect();
        let log_probs: Array1<f64> = self.log_posterior_batch(&parameters, failures)?.into();

        if !log_probs.iter().any(|v| v.is_finite()) {
            let reason = no_valid_reason(failures.first.as_ref(), "position");
            return Err(Error::SamplingError(format!(
                "All {} initial walker positions have zero posterior probability ({reason})",
                log_probs.len()
            )));
        }
        Ok(log_probs)
    }

    /// Run the ensemble sampler for Bayesian parameter estimation.
    ///
    /// Performs MCMC sampling using the affine-invariant ensemble algorithm.
    /// Walkers are initialised according to `init`, then evolved through
    /// `n_iterations` of the stretch move algorithm. All model evaluations
    /// are parallelised via rayon.
    ///
    /// # Arguments
    ///
    /// * `n_iterations` - Number of MCMC iterations to run (each iteration updates all walkers once)
    /// * `init` - Walker initialisation strategy (from prior, around a point, or explicit)
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
    ///   posterior samples using `chain.flat_samples(discard)` or `chain.to_parameter_map(discard)`.
    /// - **Thinning**: Reduces memory by storing only every Nth sample. Use to reduce
    ///   autocorrelation or save disk space for long runs. Set `thin=1` to store all samples.
    ///
    /// # Walker Count
    ///
    /// Uses the default number of walkers: `max(2 × n_parameters, 32)`.
    /// For custom walker count, use `run_with_walkers()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Run 10,000 iterations, initialise from prior, store every 10th sample
    /// let chain = sampler.run(10000, WalkerInit::FromPrior, 10)?;
    ///
    /// // Check convergence (discard first 1000 samples as burn-in)
    /// let r_hat = chain.r_hat(1000)?;
    /// println!("R-hat: {:?}", r_hat);
    ///
    /// // Extract converged posterior samples
    /// let samples = chain.flat_samples(1000);  // shape: (n_samples, n_parameters)
    /// ```
    pub fn run(&self, n_iterations: usize, init: WalkerInit, thin: usize) -> Result<Chain> {
        self.run_with_walkers(
            n_iterations,
            init,
            self.default_n_walkers,
            thin,
            None::<fn(&ProgressInfo) -> Result<()>>,
        )
    }

    /// Run the ensemble sampler with progress callback.
    ///
    /// # Arguments
    ///
    /// * `n_iterations` - Number of MCMC iterations to run
    /// * `init` - Walker initialisation strategy
    /// * `thin` - Thinning interval (store every thin-th sample)
    /// * `progress_callback` - Callback for progress reporting. An `Err` from it stops the run.
    pub fn run_with_progress<F>(
        &self,
        n_iterations: usize,
        init: WalkerInit,
        thin: usize,
        progress_callback: F,
    ) -> Result<Chain>
    where
        F: FnMut(&ProgressInfo) -> Result<()>,
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
    /// * `init` - Walker initialisation strategy
    /// * `thin` - Thinning interval (store every thin-th sample)
    /// * `checkpoint_every` - Save checkpoint every N iterations
    /// * `checkpoint_path` - Base path for checkpoint files (will append .state and .chain)
    /// * `progress_callback` - Optional callback for progress reporting. An `Err` from it stops the run.
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
        F: FnMut(&ProgressInfo) -> Result<()>,
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
            None,
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
    /// * `progress_callback` - Optional callback for progress reporting. An `Err` from it stops the run.
    /// * `seed` - Seed for the remaining iterations. `None` seeds from the OS.
    ///
    /// # Returns
    ///
    /// The complete chain including both resumed and new samples.
    ///
    /// # Errors
    ///
    /// Fails if the checkpoint's parameter names or thinning interval differ from this run's.
    #[allow(clippy::too_many_arguments)]
    pub fn resume_from_checkpoint<F, P>(
        &self,
        n_iterations: usize,
        thin: usize,
        checkpoint_every: usize,
        checkpoint_path: P,
        progress_callback: Option<F>,
        seed: Option<u64>,
    ) -> Result<Chain>
    where
        F: FnMut(&ProgressInfo) -> Result<()>,
        P: AsRef<Path>,
    {
        let state_path = format!("{}.state", checkpoint_path.as_ref().display());
        let chain_path = format!("{}.chain", checkpoint_path.as_ref().display());

        // Load state and chain
        let state = SamplerState::load_checkpoint(&state_path)?;
        let chain = Chain::load(&chain_path)?;

        let names = self.parameters.parameter_names();
        for (what, checkpoint_names) in [
            ("state", &state.parameter_names),
            ("chain", &chain.parameter_names),
        ] {
            if !names
                .iter()
                .copied()
                .eq(checkpoint_names.iter().map(String::as_str))
            {
                return Err(Error::SamplingError(format!(
                    "Checkpoint {what} parameter names {checkpoint_names:?} do not match \
                     the sampler's parameter names {names:?}"
                )));
            }
        }
        if chain.thin != thin {
            return Err(Error::SamplingError(format!(
                "thin ({thin}) does not match the checkpoint's thin ({})",
                chain.thin
            )));
        }

        let iterations_remaining = n_iterations.saturating_sub(chain.n_iterations());
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
            &mut rng_from_seed(seed),
        )
    }

    /// Run the ensemble sampler with a specific number of walkers and checkpointing.
    ///
    /// # Arguments
    ///
    /// * `n_iterations` - Number of MCMC iterations to run
    /// * `init` - Walker initialisation strategy
    /// * `n_walkers` - Number of walkers (must be even and >= 2)
    /// * `thin` - Thinning interval (store every thin-th sample)
    /// * `checkpoint_every` - Save checkpoint every N iterations (0 = no checkpointing)
    /// * `checkpoint_path` - Base path for checkpoint files
    /// * `progress_callback` - Optional callback for progress reporting. An `Err` from it stops the run.
    /// * `seed` - Seed for walker initialisation and sampling, so equal seeds give equal chains.
    ///   `None` seeds from the OS.
    #[allow(clippy::too_many_arguments)]
    pub fn run_with_checkpoint_and_walkers<F, P>(
        &self,
        n_iterations: usize,
        init: WalkerInit,
        n_walkers: usize,
        thin: usize,
        checkpoint_every: usize,
        checkpoint_path: P,
        progress_callback: Option<F>,
        seed: Option<u64>,
    ) -> Result<Chain>
    where
        F: FnMut(&ProgressInfo) -> Result<()>,
        P: AsRef<Path>,
    {
        if n_walkers < 2 || !n_walkers.is_multiple_of(2) {
            return Err(Error::SamplingError(format!(
                "n_walkers ({n_walkers}) must be even and at least 2"
            )));
        }
        if thin == 0 {
            return Err(Error::SamplingError("thin must be at least 1".to_string()));
        }
        if n_iterations > 0 && thin > n_iterations {
            return Err(Error::SamplingError(format!(
                "thin ({thin}) is greater than n_iterations ({n_iterations}), \
                 so no samples would be stored"
            )));
        }

        let mut rng = rng_from_seed(seed);
        let positions = init.initialise(n_walkers, &self.parameters, &mut rng)?;

        // Create initial state
        let parameter_names: Vec<String> = self
            .parameters
            .parameter_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let state = SamplerState::new(positions, parameter_names.clone())?;
        let chain = Chain::new(parameter_names, thin);

        self.run_from_state(
            state,
            chain,
            n_iterations,
            checkpoint_every,
            checkpoint_path,
            progress_callback,
            &mut rng,
        )
    }

    /// Run the ensemble sampler with a specific number of walkers.
    ///
    /// # Arguments
    ///
    /// * `n_iterations` - Number of MCMC iterations to run
    /// * `init` - Walker initialisation strategy
    /// * `n_walkers` - Number of walkers (must be even and >= 2)
    /// * `thin` - Thinning interval (store every thin-th sample)
    /// * `progress_callback` - Optional callback for progress reporting. An `Err` from it stops the run.
    pub fn run_with_walkers<F>(
        &self,
        n_iterations: usize,
        init: WalkerInit,
        n_walkers: usize,
        thin: usize,
        progress_callback: Option<F>,
    ) -> Result<Chain>
    where
        F: FnMut(&ProgressInfo) -> Result<()>,
    {
        self.run_with_checkpoint_and_walkers(
            n_iterations,
            init,
            n_walkers,
            thin,
            0,
            "",
            progress_callback,
            None,
        )
    }

    /// Update a group of walkers using stretch moves.
    ///
    /// # Arguments
    ///
    /// * `state` - Current sampler state (modified in place)
    /// * `active_range` - Range of walker indices to update
    /// * `complementary_range` - Range of walker indices to use as complementary ensemble
    /// * `rng` - Random number generator
    /// * `failures` - Failed evaluations, which reject their proposals
    fn update_group<Rng: rand::RngExt>(
        &self,
        state: &mut SamplerState,
        active_range: std::ops::Range<usize>,
        complementary_range: std::ops::Range<usize>,
        rng: &mut Rng,
        failures: &mut Failures,
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
        let proposal_log_probs = self.log_posterior_batch(&proposal_params, failures)?;

        // Accept/reject each proposal
        for (walker_idx, ((proposal, z), &log_prob_new)) in
            active_range.zip(proposals.iter().zip(proposal_log_probs.iter()))
        {
            let log_prob_old = state.log_probs[walker_idx];

            // Compute acceptance probability
            let accept_prob = self.stretch.acceptance_probability(
                *z,
                state.n_parameters(),
                log_prob_old,
                log_prob_new,
            );

            // Accept/reject
            state.n_proposed[walker_idx] += 1;
            if rng.random::<f64>() < accept_prob {
                // Accept
                state.positions.row_mut(walker_idx).assign(proposal);
                state.log_probs[walker_idx] = log_prob_new;
                state.n_accepted[walker_idx] += 1;
            }
            // If rejected, walker stays at current position
        }

        Ok(())
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
    /// * `progress_callback` - Optional callback for progress reporting. An `Err` from it stops the run.
    /// * `rng` - Random number generator for proposals and acceptance
    #[allow(clippy::too_many_arguments)]
    fn run_from_state<F, P>(
        &self,
        mut state: SamplerState,
        mut chain: Chain,
        n_iterations: usize,
        checkpoint_every: usize,
        checkpoint_path: P,
        mut progress_callback: Option<F>,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<Chain>
    where
        F: FnMut(&ProgressInfo) -> Result<()>,
        P: AsRef<Path>,
    {
        let n_walkers = state.n_walkers();

        // Compute initial log probabilities if not already computed
        if state.log_probs.iter().all(|&lp| !lp.is_finite()) {
            state.log_probs = self.initial_log_probs(&state.positions, &mut chain.failures)?;
        }

        // Setup checkpoint paths
        let state_path = format!("{}.state", checkpoint_path.as_ref().display());
        let chain_path = format!("{}.chain", checkpoint_path.as_ref().display());

        for iteration in 0..n_iterations {
            // Split walkers into two groups
            let half = n_walkers / 2;

            // Update first half using second half as complementary ensemble
            self.update_group(
                &mut state,
                0..half,
                half..n_walkers,
                rng,
                &mut chain.failures,
            )?;

            // Update second half using first half as complementary ensemble
            self.update_group(
                &mut state,
                half..n_walkers,
                0..half,
                rng,
                &mut chain.failures,
            )?;

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
                    n_iterations,
                    acceptance_rate: state.mean_acceptance_rate(),
                    mean_log_prob: state.log_probs.mean().unwrap_or(f64::NEG_INFINITY),
                };
                callback(&info)?;
            }
        }

        Ok(chain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_runner::ModelRunner;
    use crate::test_support::FailingModel;
    use crate::{GaussianLikelihood, ParameterSet, Uniform};
    use ndarray::Array2;
    use std::cell::RefCell;
    use std::rc::Rc;

    type NoProgress = fn(&ProgressInfo) -> Result<()>;

    // Simple model for testing: y = a * x + b
    struct LinearModel {
        parameter_names: Vec<String>,
    }

    impl LinearModel {
        fn new() -> Self {
            Self {
                parameter_names: vec!["a".to_string(), "b".to_string()],
            }
        }
    }

    impl ModelRunner for LinearModel {
        fn parameter_names(&self) -> &[String] {
            &self.parameter_names
        }

        fn run(&self, parameters: &[f64]) -> crate::Result<crate::likelihood::ModelOutput> {
            let a = parameters[0];
            let b = parameters[1];

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
        parameter_names: Vec<String>,
    }

    impl ConstantModel {
        fn new() -> Self {
            Self {
                parameter_names: vec!["x".to_string()],
            }
        }
    }

    impl ModelRunner for ConstantModel {
        fn parameter_names(&self) -> &[String] {
            &self.parameter_names
        }

        fn run(&self, parameters: &[f64]) -> crate::Result<crate::likelihood::ModelOutput> {
            let x = parameters[0];

            let mut output = crate::likelihood::ModelOutput::new();
            let mut var = crate::likelihood::VariableOutput::new("value");
            var.add(0.0, x);

            output.add_variable(var);
            Ok(output)
        }
    }

    // Dummy model for validation testing
    struct DummyModel {
        parameter_names: Vec<String>,
    }

    impl DummyModel {
        fn new() -> Self {
            Self {
                parameter_names: vec!["x".to_string()],
            }
        }
    }

    impl ModelRunner for DummyModel {
        fn parameter_names(&self) -> &[String] {
            &self.parameter_names
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
        let mut parameters = ParameterSet::new();
        parameters.add("a", Box::new(Uniform::new(0.0, 5.0).unwrap()));
        parameters.add("b", Box::new(Uniform::new(-2.0, 4.0).unwrap()));

        // Create sampler
        let runner = LinearModel::new();
        let likelihood = GaussianLikelihood::default();
        let sampler = EnsembleSampler::new(parameters, runner, likelihood, target).unwrap();

        // Run a short chain to test functionality
        let chain = sampler
            .run(10, WalkerInit::FromPrior, 1)
            .expect("Sampler should run successfully");

        // Basic checks
        assert_eq!(chain.len(), 10); // All 10 iterations stored (thin=1)
        assert_eq!(chain.parameter_names(), &["a", "b"]);

        // Check that we got some samples
        let flat_samples = chain.flat_samples(0);
        assert!(flat_samples.nrows() > 0);
        assert_eq!(flat_samples.ncols(), 2);

        // Check log probabilities are finite (at least some valid samples)
        let flat_lp = chain.flat_log_probs(0);
        let n_finite_values = flat_lp.iter().filter(|&&lp| lp.is_finite()).count();
        assert!(
            n_finite_values > 0,
            "Should have at least some finite log probabilities"
        );
    }

    #[test]
    fn test_ensemble_sampler_with_ball_init() {
        // Target: x should be near 0.5
        let mut target = Target::new();
        target.add_variable("value").add(0.0, 0.5, 0.1).unwrap();

        let mut parameters = ParameterSet::new();
        parameters.add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()));

        let runner = ConstantModel::new();
        let likelihood = GaussianLikelihood::default();
        let sampler = EnsembleSampler::new(parameters, runner, likelihood, target).unwrap();

        // Initialise walkers in a ball around the true value
        let init = WalkerInit::Ball {
            centre: vec![0.5],
            radius: 0.01,
        };

        let chain = sampler
            .run_with_walkers(5, init, 10, 1, None::<NoProgress>)
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
        let mut parameters = ParameterSet::new();
        parameters.add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()));

        let runner = DummyModel::new();
        let likelihood = GaussianLikelihood::default();
        let sampler = EnsembleSampler::new(parameters, runner, likelihood, target).unwrap();

        // Try with odd number of walkers (should fail)
        let result = sampler.run_with_walkers(5, WalkerInit::FromPrior, 3, 1, None::<NoProgress>);
        assert!(result.is_err());
    }

    #[test]
    fn test_progress_callback() {
        let target = Target::new();
        let mut parameters = ParameterSet::new();
        parameters.add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()));

        let runner = DummyModel::new();
        let likelihood = GaussianLikelihood::default();
        let sampler = EnsembleSampler::new(parameters, runner, likelihood, target).unwrap();

        // Track progress updates
        let progress_updates = Rc::new(RefCell::new(Vec::new()));
        let progress_updates_clone = Rc::clone(&progress_updates);

        let callback = move |info: &ProgressInfo| {
            progress_updates_clone.borrow_mut().push((
                info.iteration,
                info.n_iterations,
                info.acceptance_rate,
                info.mean_log_prob,
            ));
            Ok(())
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
        // Posterior is bivariate normal centred at observation (independent dimensions)

        let true_mean = [1.0, 2.0]; // True parameter values (= posterior mean)
        let obs_std = 1.0; // Observation uncertainty = posterior std (with uniform prior)

        // Create parameter priors (uniform, wide enough to not dominate)
        let mut parameters = ParameterSet::new();
        parameters
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
            parameter_names: Vec<String>,
        }

        impl ModelRunner for IdentityRunner {
            fn parameter_names(&self) -> &[String] {
                &self.parameter_names
            }

            fn run(&self, parameters: &[f64]) -> crate::Result<ModelOutput> {
                let mut output = std::collections::HashMap::new();

                // Create output for x variable
                let mut x_values = std::collections::HashMap::new();
                for i in 0..20 {
                    x_values.insert(format!("{:.6}", i as f64), parameters[0]);
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
                    y_values.insert(format!("{:.6}", i as f64), parameters[1]);
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
            parameter_names: vec!["x".to_string(), "y".to_string()],
        };
        let likelihood = GaussianLikelihood::new();

        // Run sampler
        // Use enough iterations for reliable R-hat convergence (stochastic test)
        let sampler =
            EnsembleSampler::new(parameters.clone(), runner, likelihood, target.clone()).unwrap();
        let n_iterations = 2000;
        let n_walkers = 32;
        let burn_in = 1000;

        let chain = sampler
            .run_with_walkers(
                n_iterations,
                WalkerInit::FromPrior,
                n_walkers,
                1,
                None::<NoProgress>,
            )
            .unwrap();

        // Extract samples after burn-in
        let samples = chain.flat_samples(burn_in);
        let (n_samples, n_parameters) = samples.dim();

        assert_eq!(n_parameters, 2);
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
        use rand::RngExt;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Verify that same seed produces same chain despite parallel execution
        //
        // Note: Full determinism requires that WalkerInit::Explicit is used, since
        // WalkerInit::FromPrior uses thread_rng() which may not be deterministic in parallel.

        // Create simple test setup
        let mut parameters = ParameterSet::new();
        parameters
            .add("x".to_string(), Box::new(Normal::new(0.0, 5.0).unwrap()))
            .add("y".to_string(), Box::new(Normal::new(0.0, 5.0).unwrap()));

        let mut target = Target::new();
        target.add_variable("x").add(0.0, 1.0, 0.5).unwrap();
        target.add_variable("y").add(0.0, 2.0, 0.5).unwrap();

        struct IdentityRunner {
            parameter_names: Vec<String>,
        }

        impl ModelRunner for IdentityRunner {
            fn parameter_names(&self) -> &[String] {
                &self.parameter_names
            }

            fn run(&self, parameters: &[f64]) -> crate::Result<ModelOutput> {
                let mut output = std::collections::HashMap::new();
                output.insert(
                    "x".to_string(),
                    VariableOutput {
                        name: "x".to_string(),
                        values: vec![("0.000000".to_string(), parameters[0])]
                            .into_iter()
                            .collect(),
                    },
                );
                output.insert(
                    "y".to_string(),
                    VariableOutput {
                        name: "y".to_string(),
                        values: vec![("0.000000".to_string(), parameters[1])]
                            .into_iter()
                            .collect(),
                    },
                );
                Ok(ModelOutput { variables: output })
            }
        }

        let runner = IdentityRunner {
            parameter_names: vec!["x".to_string(), "y".to_string()],
        };
        let likelihood = GaussianLikelihood::new();
        let sampler = EnsembleSampler::new(parameters, runner, likelihood, target).unwrap();

        // Generate deterministic initial positions
        let n_walkers = 16;
        let n_parameters = 2;
        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let positions =
            Array2::from_shape_fn((n_walkers, n_parameters), |_| rng.random::<f64>() * 2.0);

        // Run sampler twice with same initial positions
        let init = WalkerInit::Explicit(positions.clone());

        // NOTE: The sampler uses thread_rng() internally for proposals, which is NOT
        // deterministic across runs. This is a design limitation.

        let chain1 = sampler
            .run_with_walkers(100, init.clone(), n_walkers, 1, None::<NoProgress>)
            .unwrap();

        let chain2 = sampler
            .run_with_walkers(100, init, n_walkers, 1, None::<NoProgress>)
            .unwrap();

        // Both chains should have same shape
        assert_eq!(chain1.parameter_names, chain2.parameter_names);
        assert_eq!(chain1.thin, chain2.thin);
        assert_eq!(chain1.n_iterations, chain2.n_iterations);

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
        let mut parameters = ParameterSet::new();
        parameters.add("x".to_string(), Box::new(Normal::new(0.0, 5.0).unwrap()));

        let mut target = Target::new();
        target.add_variable("x").add(0.0, 1.0, 0.5).unwrap();

        struct SingleParameterRunner;
        impl ModelRunner for SingleParameterRunner {
            fn parameter_names(&self) -> &[String] {
                static NAMES: std::sync::LazyLock<[String; 1]> =
                    std::sync::LazyLock::new(|| ["x".to_string()]);
                &*NAMES
            }

            fn run(&self, parameters: &[f64]) -> crate::Result<ModelOutput> {
                let mut output = std::collections::HashMap::new();
                output.insert(
                    "x".to_string(),
                    VariableOutput {
                        name: "x".to_string(),
                        values: vec![("0.000000".to_string(), parameters[0])]
                            .into_iter()
                            .collect(),
                    },
                );
                Ok(ModelOutput { variables: output })
            }
        }

        let sampler = EnsembleSampler::new(
            parameters,
            SingleParameterRunner,
            GaussianLikelihood::new(),
            target,
        )
        .unwrap();

        let chain = sampler
            .run_with_walkers(50, WalkerInit::FromPrior, 10, 1, None::<NoProgress>)
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
        let n_parameters = 50;
        let mut parameters = ParameterSet::new();
        for i in 0..n_parameters {
            parameters.add(format!("x{}", i), Box::new(Normal::new(0.0, 10.0).unwrap()));
        }

        let mut target = Target::new();
        for i in 0..n_parameters {
            target
                .add_variable(format!("x{}", i))
                .add(0.0, (i as f64) * 0.1, 1.0)
                .unwrap();
        }

        struct HighDimRunner {
            n_parameters: usize,
            parameter_names: Vec<String>,
        }

        impl ModelRunner for HighDimRunner {
            fn parameter_names(&self) -> &[String] {
                &self.parameter_names
            }

            fn run(&self, parameters: &[f64]) -> crate::Result<ModelOutput> {
                let mut output = std::collections::HashMap::new();
                for (i, &param_val) in parameters.iter().enumerate().take(self.n_parameters) {
                    output.insert(
                        format!("x{}", i),
                        VariableOutput {
                            name: format!("x{}", i),
                            values: vec![("0.000000".to_string(), param_val)]
                                .into_iter()
                                .collect(),
                        },
                    );
                }
                Ok(ModelOutput { variables: output })
            }
        }

        let runner = HighDimRunner {
            n_parameters,
            parameter_names: (0..n_parameters).map(|i| format!("x{}", i)).collect(),
        };

        let sampler =
            EnsembleSampler::new(parameters, runner, GaussianLikelihood::new(), target).unwrap();

        // For high-D, need many walkers (at least 2*n_parameters)
        let n_walkers = 100;

        let chain = sampler
            .run_with_walkers(100, WalkerInit::FromPrior, n_walkers, 1, None::<NoProgress>)
            .unwrap();

        let samples = chain.flat_samples(50);
        assert_eq!(samples.ncols(), n_parameters);
        assert!(samples.nrows() > 0);

        // Check that sampler ran successfully and produced reasonable output
        for i in 0..n_parameters {
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
        use rand::RngExt;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Test all walkers initialised very close together
        // Note: The stretch move CANNOT work if all walkers are at exactly the same point,
        // since the proposal is y = c + z*(x - c) where c is complementary walker.
        // If x == c, then y == c always (no movement).
        // So we test with walkers in a very tight ball instead.
        let mut parameters = ParameterSet::new();
        parameters
            .add("x".to_string(), Box::new(Normal::new(0.0, 5.0).unwrap()))
            .add("y".to_string(), Box::new(Normal::new(0.0, 5.0).unwrap()));

        let mut target = Target::new();
        target.add_variable("x").add(0.0, 1.0, 0.5).unwrap();
        target.add_variable("y").add(0.0, 2.0, 0.5).unwrap();

        struct IdentityRunner {
            parameter_names: Vec<String>,
        }

        impl ModelRunner for IdentityRunner {
            fn parameter_names(&self) -> &[String] {
                &self.parameter_names
            }

            fn run(&self, parameters: &[f64]) -> crate::Result<ModelOutput> {
                let mut output = std::collections::HashMap::new();
                output.insert(
                    "x".to_string(),
                    VariableOutput {
                        name: "x".to_string(),
                        values: vec![("0.000000".to_string(), parameters[0])]
                            .into_iter()
                            .collect(),
                    },
                );
                output.insert(
                    "y".to_string(),
                    VariableOutput {
                        name: "y".to_string(),
                        values: vec![("0.000000".to_string(), parameters[1])]
                            .into_iter()
                            .collect(),
                    },
                );
                Ok(ModelOutput { variables: output })
            }
        }

        let runner = IdentityRunner {
            parameter_names: vec!["x".to_string(), "y".to_string()],
        };

        let sampler =
            EnsembleSampler::new(parameters, runner, GaussianLikelihood::new(), target).unwrap();

        // Initialise all walkers in a tiny ball around (0, 0)
        let n_walkers = 16;
        let mut rng = ChaCha8Rng::seed_from_u64(555);
        let mut positions = Array2::zeros((n_walkers, 2));
        for i in 0..n_walkers {
            positions[[i, 0]] = rng.random::<f64>() * 0.001;
            positions[[i, 1]] = rng.random::<f64>() * 0.001;
        }
        let init = WalkerInit::Explicit(positions);

        let chain = sampler
            .run_with_walkers(200, init, n_walkers, 1, None::<NoProgress>)
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

    fn failing_sampler(
        threshold: Option<f64>,
    ) -> EnsembleSampler<FailingModel, GaussianLikelihood> {
        let mut parameters = ParameterSet::new();
        parameters.add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()));
        let mut target = Target::new();
        target.add_variable("value").add(0.0, 0.2, 0.1).unwrap();
        EnsembleSampler::new(
            parameters,
            FailingModel::new(threshold),
            GaussianLikelihood::default(),
            target,
        )
        .unwrap()
    }

    #[test]
    fn test_run_fails_fast_when_every_initial_position_fails() {
        let sampler = failing_sampler(None);

        let err = sampler
            .run(10, WalkerInit::FromPrior, 1)
            .expect_err("a run with no valid initial walker should fail");

        assert!(
            err.to_string().contains("solver diverged"),
            "error should include the underlying model error, got: {err}"
        );
    }

    #[test]
    fn test_checkpointed_run_fails_fast_when_every_initial_position_fails() {
        let sampler = failing_sampler(None);
        let dir = tempfile::tempdir().unwrap();

        let err = sampler
            .run_with_checkpoint(
                10,
                WalkerInit::FromPrior,
                1,
                5,
                dir.path().join("checkpoint"),
                None::<NoProgress>,
            )
            .expect_err("a run with no valid initial walker should fail");

        assert!(err.to_string().contains("solver diverged"), "got: {err}");
    }

    #[test]
    fn test_failed_proposals_are_rejected_not_fatal() {
        let sampler = failing_sampler(Some(0.5));
        let init = WalkerInit::Ball {
            centre: vec![0.2],
            radius: 0.01,
        };

        let chain = sampler
            .run(20, init, 1)
            .expect("failed proposals are rejected");

        assert!(chain.flat_samples(0).iter().all(|&x| x <= 0.5));
    }

    #[test]
    fn test_progress_callback_error_stops_the_run() {
        let sampler = failing_sampler(Some(10.0));
        let mut calls = 0;

        let err = sampler
            .run_with_progress(10, WalkerInit::FromPrior, 1, |_: &ProgressInfo| {
                calls += 1;
                Err(Error::SamplingError("interrupted".to_string()))
            })
            .expect_err("a failing callback should stop the run");

        assert_eq!(calls, 1);
        assert!(err.to_string().contains("interrupted"), "got: {err}");
    }

    fn seeded_run(sampler: &EnsembleSampler<FailingModel, GaussianLikelihood>, seed: u64) -> Chain {
        sampler
            .run_with_checkpoint_and_walkers(
                5,
                WalkerInit::FromPrior,
                8,
                1,
                0,
                "",
                None::<NoProgress>,
                Some(seed),
            )
            .unwrap()
    }

    #[test]
    fn test_seed_reproduces_chain() {
        let sampler = failing_sampler(Some(10.0));

        let a = seeded_run(&sampler, 42);
        let b = seeded_run(&sampler, 42);
        let c = seeded_run(&sampler, 43);

        assert_eq!(a.flat_samples(0), b.flat_samples(0));
        assert_ne!(a.flat_samples(0), c.flat_samples(0));
    }

    #[test]
    fn test_failed_proposals_are_counted() {
        let sampler = failing_sampler(Some(0.5));
        let chain = seeded_run(&sampler, 1);

        let failures = chain.failures();
        assert!(failures.count > 0);
        assert!(failures
            .first
            .as_deref()
            .unwrap()
            .contains("solver diverged"));
    }

    #[test]
    fn test_interrupt_stops_the_run() {
        struct InterruptingModel(Vec<String>);
        impl ModelRunner for InterruptingModel {
            fn parameter_names(&self) -> &[String] {
                &self.0
            }
            fn run(&self, _: &[f64]) -> crate::Result<crate::likelihood::ModelOutput> {
                Err(Error::Interrupted("KeyboardInterrupt".to_string()))
            }
        }
        let mut parameters = ParameterSet::new();
        parameters.add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()));
        let sampler = EnsembleSampler::new(
            parameters,
            InterruptingModel(vec!["x".to_string()]),
            GaussianLikelihood::default(),
            Target::new(),
        )
        .unwrap();

        let err = sampler.run(5, WalkerInit::FromPrior, 1).unwrap_err();
        assert!(matches!(err, Error::Interrupted(_)), "got: {err}");
    }

    #[test]
    fn test_runner_names_must_match_parameters() {
        let mut parameters = ParameterSet::new();
        parameters.add("y", Box::new(Uniform::new(0.0, 1.0).unwrap()));
        let result = EnsembleSampler::new(
            parameters,
            FailingModel::new(None),
            GaussianLikelihood::default(),
            Target::new(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_thin_is_rejected() {
        let sampler = failing_sampler(Some(10.0));
        for (n_iterations, thin) in [(5, 0), (3, 10)] {
            let err = sampler
                .run(n_iterations, WalkerInit::FromPrior, thin)
                .unwrap_err();
            assert!(err.to_string().contains("thin"), "got: {err}");
        }
    }

    #[test]
    fn test_resume_checks_thin() {
        let sampler = failing_sampler(Some(10.0));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpoint");
        sampler
            .run_with_checkpoint(4, WalkerInit::FromPrior, 2, 2, &path, None::<NoProgress>)
            .unwrap();

        let err = sampler
            .resume_from_checkpoint(8, 1, 2, &path, None::<NoProgress>, None)
            .unwrap_err();
        assert!(err.to_string().contains("thin"), "got: {err}");

        let chain = sampler
            .resume_from_checkpoint(8, 2, 2, &path, None::<NoProgress>, None)
            .unwrap();
        assert_eq!(chain.n_iterations(), 8);
    }
}
