//! Point estimation for finding optimal parameter sets.
//!
//! This module provides infrastructure for point estimation (optimisation)
//! of model parameters by maximising the likelihood function.

use crate::{
    likelihood::LikelihoodFn, model_runner::ModelRunner, parameter_set::ParameterSet,
    target::Target,
};

/// Point estimator for finding optimal parameter sets.
///
/// Wraps the calibration components (parameters, model runner, likelihood, target)
/// and tracks all evaluated points and their likelihoods during optimisation.
pub struct PointEstimator<R: ModelRunner, L: LikelihoodFn> {
    /// Parameter set defining the prior distributions and bounds
    params: ParameterSet,

    /// Model runner for evaluating parameter sets
    runner: R,

    /// Likelihood function for computing log probability
    likelihood: L,

    /// Target observations
    target: Target,

    /// All evaluated parameter vectors (each row is one evaluation)
    evaluated_params: Vec<Vec<f64>>,

    /// Log likelihood values for each evaluated parameter set
    evaluated_log_likelihoods: Vec<f64>,
}

impl<R: ModelRunner, L: LikelihoodFn> PointEstimator<R, L> {
    /// Create a new point estimator.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameter set defining priors and bounds
    /// * `runner` - Model runner for evaluating parameter sets
    /// * `likelihood` - Likelihood function
    /// * `target` - Target observations
    pub fn new(params: ParameterSet, runner: R, likelihood: L, target: Target) -> Self {
        Self {
            params,
            runner,
            likelihood,
            target,
            evaluated_params: Vec::new(),
            evaluated_log_likelihoods: Vec::new(),
        }
    }

    /// Get the number of parameters.
    pub fn n_params(&self) -> usize {
        self.params.len()
    }

    /// Get the parameter names.
    pub fn param_names(&self) -> Vec<&str> {
        self.params.param_names()
    }

    /// Get the parameter bounds for optimisation.
    ///
    /// Returns `(lower_bounds, upper_bounds)` as vectors.
    pub fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        self.params.bounds()
    }

    /// Evaluate the log likelihood for a parameter vector.
    ///
    /// This method runs the model, computes the likelihood, and tracks
    /// the evaluation in the internal history.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameter vector (must match order of param_names)
    ///
    /// # Returns
    ///
    /// The log likelihood value. Returns negative infinity if:
    /// - Parameter is outside prior support
    /// - Model execution fails
    /// - Likelihood computation fails
    pub fn evaluate(&mut self, params: &[f64]) -> f64 {
        // Compute log prior
        let log_prior = match self.params.log_prior(params) {
            Ok(lp) => lp,
            Err(_) => {
                // Outside prior support or dimension mismatch
                self.evaluated_params.push(params.to_vec());
                self.evaluated_log_likelihoods.push(f64::NEG_INFINITY);
                return f64::NEG_INFINITY;
            }
        };

        if !log_prior.is_finite() || log_prior == f64::NEG_INFINITY {
            // Outside prior support
            self.evaluated_params.push(params.to_vec());
            self.evaluated_log_likelihoods.push(f64::NEG_INFINITY);
            return f64::NEG_INFINITY;
        }

        // Run model
        let model_output = match self.runner.run(params) {
            Ok(output) => output,
            Err(_) => {
                // Model failure
                self.evaluated_params.push(params.to_vec());
                self.evaluated_log_likelihoods.push(f64::NEG_INFINITY);
                return f64::NEG_INFINITY;
            }
        };

        // Compute log likelihood
        let log_likelihood = match self.likelihood.ln_likelihood(&model_output, &self.target) {
            Ok(ll) => ll,
            Err(_) => {
                // Likelihood computation failure
                self.evaluated_params.push(params.to_vec());
                self.evaluated_log_likelihoods.push(f64::NEG_INFINITY);
                return f64::NEG_INFINITY;
            }
        };

        // Log posterior (prior + likelihood, no normalization constant)
        let log_posterior = log_prior + log_likelihood;

        // Track evaluation
        self.evaluated_params.push(params.to_vec());
        self.evaluated_log_likelihoods.push(log_likelihood);

        log_posterior
    }

    /// Get all evaluated parameter vectors.
    ///
    /// Returns a reference to the history of all parameter sets evaluated.
    pub fn evaluated_params(&self) -> &[Vec<f64>] {
        &self.evaluated_params
    }

    /// Get all evaluated log likelihood values.
    ///
    /// Returns a reference to the history of log likelihoods.
    /// Note: These are log likelihoods, not log posteriors (prior not included).
    pub fn evaluated_log_likelihoods(&self) -> &[f64] {
        &self.evaluated_log_likelihoods
    }

    /// Get the best parameter set found so far.
    ///
    /// Returns the parameter vector with the highest log likelihood,
    /// along with its log likelihood value.
    ///
    /// Returns `None` if no valid evaluations have been performed yet.
    pub fn best(&self) -> Option<(&[f64], f64)> {
        if self.evaluated_log_likelihoods.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_ll = self.evaluated_log_likelihoods[0];

        for (i, &ll) in self.evaluated_log_likelihoods.iter().enumerate().skip(1) {
            if ll > best_ll {
                best_ll = ll;
                best_idx = i;
            }
        }

        Some((&self.evaluated_params[best_idx], best_ll))
    }

    /// Clear the evaluation history.
    ///
    /// This can be useful when running multiple optimisation attempts
    /// and you want to track each attempt separately.
    pub fn clear_history(&mut self) {
        self.evaluated_params.clear();
        self.evaluated_log_likelihoods.clear();
    }

    /// Get the number of evaluations performed.
    pub fn n_evaluations(&self) -> usize {
        self.evaluated_params.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GaussianLikelihood, Normal, Observation};

    struct ConstantModel {
        param_names: Vec<String>,
        constant_value: f64,
    }

    impl ConstantModel {
        fn new(constant_value: f64) -> Self {
            Self {
                param_names: vec!["x".to_string()],
                constant_value,
            }
        }
    }

    impl ModelRunner for ConstantModel {
        fn param_names(&self) -> &[String] {
            &self.param_names
        }

        fn run(&self, _params: &[f64]) -> crate::Result<crate::likelihood::ModelOutput> {
            let mut output = crate::likelihood::ModelOutput::new();
            let mut var_output = crate::likelihood::VariableOutput::new("y");
            var_output.add(0.0, self.constant_value);
            var_output.add(1.0, self.constant_value);
            output.add_variable(var_output);
            Ok(output)
        }
    }

    #[test]
    fn test_point_estimator_creation() {
        let mut params = ParameterSet::new();
        params.add("x", Box::new(Normal::new(0.0, 1.0).unwrap()));

        let runner = ConstantModel::new(5.0);
        let likelihood = GaussianLikelihood::default();
        let target = Target::new();

        let estimator = PointEstimator::new(params, runner, likelihood, target);

        assert_eq!(estimator.n_params(), 1);
        assert_eq!(estimator.param_names(), vec!["x"]);
        assert_eq!(estimator.n_evaluations(), 0);
    }

    #[test]
    fn test_point_estimator_evaluate() {
        let mut params = ParameterSet::new();
        params.add("x", Box::new(Normal::new(0.0, 1.0).unwrap()));

        let runner = ConstantModel::new(5.0);
        let likelihood = GaussianLikelihood::default();

        let mut target = Target::new();
        let obs1 = Observation::new(0.0, 5.0, 0.1).unwrap();
        let obs2 = Observation::new(1.0, 5.0, 0.1).unwrap();
        target
            .add_variable("y")
            .add_observation(obs1)
            .add_observation(obs2);

        let mut estimator = PointEstimator::new(params, runner, likelihood, target);

        // Evaluate at x=0.0 (mean of prior)
        let log_post = estimator.evaluate(&[0.0]);
        assert!(log_post.is_finite());

        // Check history
        assert_eq!(estimator.n_evaluations(), 1);
        assert_eq!(estimator.evaluated_params(), &[vec![0.0]]);
        assert_eq!(estimator.evaluated_log_likelihoods().len(), 1);

        // Evaluate at another point
        let log_post2 = estimator.evaluate(&[1.0]);
        assert!(log_post2.is_finite());

        // Check history updated
        assert_eq!(estimator.n_evaluations(), 2);
    }

    #[test]
    fn test_point_estimator_best() {
        let mut params = ParameterSet::new();
        params.add("x", Box::new(Normal::new(0.0, 1.0).unwrap()));

        let runner = ConstantModel::new(5.0);
        let likelihood = GaussianLikelihood::default();

        let mut target = Target::new();
        let obs = Observation::new(0.0, 5.0, 0.1).unwrap();
        target.add_variable("y").add_observation(obs);

        let mut estimator = PointEstimator::new(params, runner, likelihood, target);

        // No evaluations yet
        assert!(estimator.best().is_none());

        // Evaluate some points
        estimator.evaluate(&[0.0]);
        estimator.evaluate(&[1.0]);
        estimator.evaluate(&[0.5]);

        // Get best
        let (best_params, best_ll) = estimator.best().unwrap();
        assert_eq!(best_params.len(), 1);
        assert!(best_ll.is_finite());

        // Best should be one of the evaluated points
        assert!(estimator.evaluated_params().contains(&best_params.to_vec()));
    }

    #[test]
    fn test_point_estimator_clear_history() {
        let mut params = ParameterSet::new();
        params.add("x", Box::new(Normal::new(0.0, 1.0).unwrap()));

        let runner = ConstantModel::new(5.0);
        let likelihood = GaussianLikelihood::default();
        let target = Target::new();

        let mut estimator = PointEstimator::new(params, runner, likelihood, target);

        // Evaluate some points
        estimator.evaluate(&[0.0]);
        estimator.evaluate(&[1.0]);
        assert_eq!(estimator.n_evaluations(), 2);

        // Clear history
        estimator.clear_history();
        assert_eq!(estimator.n_evaluations(), 0);
        assert!(estimator.best().is_none());
    }

    #[test]
    fn test_point_estimator_bounds() {
        use crate::Uniform;

        let mut params = ParameterSet::new();
        params.add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()));
        params.add("y", Box::new(Uniform::new(-5.0, 5.0).unwrap()));

        let runner = ConstantModel::new(5.0);
        let likelihood = GaussianLikelihood::default();
        let target = Target::new();

        let estimator = PointEstimator::new(params, runner, likelihood, target);

        let (lower, upper) = estimator.bounds();

        assert_eq!(lower.len(), 2);
        assert_eq!(upper.len(), 2);

        assert_eq!(lower[0], 0.0);
        assert_eq!(upper[0], 1.0);
        assert_eq!(lower[1], -5.0);
        assert_eq!(upper[1], 5.0);
    }

    #[test]
    fn test_point_estimator_model_failure() {
        struct FailingModel {
            param_names: Vec<String>,
        }

        impl FailingModel {
            fn new() -> Self {
                Self {
                    param_names: vec!["x".to_string()],
                }
            }
        }

        impl ModelRunner for FailingModel {
            fn param_names(&self) -> &[String] {
                &self.param_names
            }

            fn run(&self, _params: &[f64]) -> crate::Result<crate::likelihood::ModelOutput> {
                Err(crate::Error::ModelError("Model failed".to_string()))
            }
        }

        let mut params = ParameterSet::new();
        params.add("x", Box::new(Normal::new(0.0, 1.0).unwrap()));

        let runner = FailingModel::new();
        let likelihood = GaussianLikelihood::default();
        let target = Target::new();

        let mut estimator = PointEstimator::new(params, runner, likelihood, target);

        // Model failure should return NEG_INFINITY
        let log_post = estimator.evaluate(&[0.0]);
        assert_eq!(log_post, f64::NEG_INFINITY);

        // Should still track the evaluation
        assert_eq!(estimator.n_evaluations(), 1);
        assert_eq!(estimator.evaluated_log_likelihoods()[0], f64::NEG_INFINITY);
    }
}
