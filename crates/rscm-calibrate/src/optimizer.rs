//! Point estimation optimizers for parameter calibration.
//!
//! This module provides optimization algorithms for finding maximum likelihood
//! or maximum a posteriori (MAP) parameter estimates.
//!
//! Currently implemented:
//! - Random search: Simple baseline global search method
//!
//! # Example
//!
//! ```ignore
//! use rscm_calibrate::{Optimizer, PointEstimator};
//!
//! let estimator = PointEstimator::new(params, runner, likelihood, target);
//! let result = estimator.optimize(Optimizer::RandomSearch, 1000)?;
//!
//! println!("Best parameters: {:?}", result.best_params);
//! println!("Best log likelihood: {}", result.best_log_likelihood);
//! ```

use crate::{point_estimator::PointEstimator, Result};
use rand::Rng;

/// Optimization algorithm to use for point estimation.
#[derive(Debug, Clone, Copy)]
pub enum Optimizer {
    /// Random search baseline
    ///
    /// Samples random parameter sets from within the parameter bounds
    /// and returns the best. Useful as a baseline or for initialization.
    ///
    /// Note: Number of samples controlled by n_samples parameter.
    RandomSearch,
}

/// Result of a point estimation optimization.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Best parameter vector found
    pub best_params: Vec<f64>,

    /// Log likelihood at best parameters (not including prior)
    pub best_log_likelihood: f64,

    /// Log posterior at best parameters (prior + likelihood)
    pub best_log_posterior: f64,

    /// Number of function evaluations performed
    pub n_evaluations: usize,

    /// Whether the optimizer converged
    pub converged: bool,
}

impl<R: crate::model_runner::ModelRunner, L: crate::likelihood::LikelihoodFn> PointEstimator<R, L> {
    /// Run point estimation optimization.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Optimization algorithm to use
    /// * `n_samples` - Number of samples to evaluate
    ///
    /// # Returns
    ///
    /// An `OptimizationResult` with the best parameters and diagnostics.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = estimator.optimize(Optimizer::RandomSearch, 1000)?;
    /// ```
    pub fn optimize(
        &mut self,
        optimizer: Optimizer,
        n_samples: usize,
    ) -> Result<OptimizationResult> {
        match optimizer {
            Optimizer::RandomSearch => self.optimize_random_search(n_samples),
        }
    }

    /// Optimize using random search.
    ///
    /// Samples n_samples parameter sets from the prior and returns the best.
    fn optimize_random_search(&mut self, n_samples: usize) -> Result<OptimizationResult> {
        // Sample parameters from prior
        let samples = self.sample_initial_params(n_samples);

        let mut best_params: Option<Vec<f64>> = None;
        let mut best_log_posterior = f64::NEG_INFINITY;
        let mut best_log_likelihood = f64::NEG_INFINITY;

        // Evaluate all samples
        for params in samples.iter() {
            let log_posterior = self.evaluate(params);

            if log_posterior > best_log_posterior {
                best_log_posterior = log_posterior;
                best_params = Some(params.clone());

                // Find corresponding log likelihood from history
                best_log_likelihood = self
                    .evaluated_log_likelihoods()
                    .iter()
                    .zip(self.evaluated_params().iter())
                    .find(|(_, p)| *p == params)
                    .map(|(ll, _)| *ll)
                    .unwrap_or(f64::NEG_INFINITY);
            }
        }

        let best = best_params.ok_or_else(|| {
            crate::Error::SamplingError("Random search found no valid samples".to_string())
        })?;

        Ok(OptimizationResult {
            best_params: best,
            best_log_likelihood,
            best_log_posterior,
            n_evaluations: n_samples,
            converged: true, // Random search always "converges"
        })
    }

    /// Helper method to sample initial parameter vectors.
    fn sample_initial_params(&self, n: usize) -> Vec<Vec<f64>> {
        // We need access to the parameter set, but it's private
        // For now, use the bounds to sample uniformly
        let (lower, upper) = self.bounds();
        let n_params = lower.len();

        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(n);

        for _ in 0..n {
            let mut sample = Vec::with_capacity(n_params);
            for i in 0..n_params {
                let value = rng.gen::<f64>() * (upper[i] - lower[i]) + lower[i];
                sample.push(value);
            }
            samples.push(sample);
        }

        samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        likelihood::{GaussianLikelihood, ModelOutput, VariableOutput},
        model_runner::ModelRunner,
        parameter_set::ParameterSet,
        target::Target,
        Uniform,
    };

    // Simple quadratic model for testing: y = (x - 2)^2
    // Minimum at x = 2
    struct QuadraticModel {
        param_names: Vec<String>,
    }

    impl QuadraticModel {
        fn new() -> Self {
            Self {
                param_names: vec!["x".to_string()],
            }
        }
    }

    impl ModelRunner for QuadraticModel {
        fn param_names(&self) -> &[String] {
            &self.param_names
        }

        fn run(&self, params: &[f64]) -> crate::Result<ModelOutput> {
            let x = params[0];
            let y = (x - 2.0).powi(2);

            let mut output = ModelOutput::new();
            let mut var = VariableOutput::new("y");
            var.add(0.0, y);

            output.add_variable(var);
            Ok(output)
        }
    }

    #[test]
    fn test_random_search_optimizer() {
        // Create a simple optimization problem
        let mut params = ParameterSet::new();
        params.add("x", Box::new(Uniform::new(0.0, 4.0).unwrap()));

        // Target: y should be 0 (achieved when x=2)
        let mut target = Target::new();
        target.add_variable("y").add(0.0, 0.0, 0.1).unwrap();

        let runner = QuadraticModel::new();
        let likelihood = GaussianLikelihood::default();

        let mut estimator = PointEstimator::new(params, runner, likelihood, target);

        // Run random search
        let result = estimator
            .optimize(Optimizer::RandomSearch, 100)
            .expect("Optimization should succeed");

        // Check result
        assert_eq!(result.n_evaluations, 100);
        assert!(result.converged);
        assert!(result.best_log_posterior.is_finite());

        // Best parameter should be in bounds
        let best_x = result.best_params[0];
        assert!((0.0..=4.0).contains(&best_x), "Best x out of bounds");

        // Should have found something better than worst case
        assert!(result.best_log_posterior > f64::NEG_INFINITY);
    }

    #[test]
    fn test_random_search_multi_dimensional() {
        // 2D quadratic: y = (x1 - 1)^2 + (x2 - 3)^2
        // Minimum at (1, 3)
        struct Quadratic2D {
            param_names: Vec<String>,
        }

        impl Quadratic2D {
            fn new() -> Self {
                Self {
                    param_names: vec!["x1".to_string(), "x2".to_string()],
                }
            }
        }

        impl ModelRunner for Quadratic2D {
            fn param_names(&self) -> &[String] {
                &self.param_names
            }

            fn run(&self, params: &[f64]) -> crate::Result<ModelOutput> {
                let x1 = params[0];
                let x2 = params[1];
                let y = (x1 - 1.0).powi(2) + (x2 - 3.0).powi(2);

                let mut output = ModelOutput::new();
                let mut var = VariableOutput::new("y");
                var.add(0.0, y);

                output.add_variable(var);
                Ok(output)
            }
        }

        let mut params = ParameterSet::new();
        params.add("x1", Box::new(Uniform::new(-2.0, 4.0).unwrap()));
        params.add("x2", Box::new(Uniform::new(0.0, 6.0).unwrap()));

        let mut target = Target::new();
        target.add_variable("y").add(0.0, 0.0, 0.1).unwrap();

        let runner = Quadratic2D::new();
        let likelihood = GaussianLikelihood::default();

        let mut estimator = PointEstimator::new(params, runner, likelihood, target);

        let result = estimator
            .optimize(Optimizer::RandomSearch, 500)
            .expect("Optimization should succeed");

        // Check parameters are in bounds
        let best_x1 = result.best_params[0];
        let best_x2 = result.best_params[1];

        assert!(
            (-2.0..=4.0).contains(&best_x1),
            "x1 out of bounds: {}",
            best_x1
        );
        assert!(
            (0.0..=6.0).contains(&best_x2),
            "x2 out of bounds: {}",
            best_x2
        );

        // With 500 samples, should find something reasonably close
        // (not as precise as gradient-based methods, but should be in the neighborhood)
        assert!(
            result.best_log_posterior.is_finite(),
            "Should find finite log posterior"
        );
    }
}
