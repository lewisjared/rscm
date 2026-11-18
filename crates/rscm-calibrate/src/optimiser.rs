//! Point estimation optimisers for parameter calibration.
//!
//! This module provides optimisation algorithms for finding maximum likelihood
//! or maximum a posteriori (MAP) parameter estimates.
//!
//! Currently implemented:
//! - Random search: Simple baseline global search method
//!
//! # Example
//!
//! ```ignore
//! use rscm_calibrate::{Optimiser, PointEstimator};
//!
//! let estimator = PointEstimator::new(parameters, runner, likelihood, target)?;
//! let result = estimator.optimise(Optimiser::RandomSearch, 1000, None)?;
//!
//! println!("Best parameters: {:?}", result.best_parameters);
//! println!("Best log likelihood: {}", result.best_log_likelihood);
//! ```

use crate::{point_estimator::PointEstimator, rng_from_seed, Error, Failures, Result};

/// Optimisation algorithm to use for point estimation.
#[derive(Debug, Clone, Copy)]
pub enum Optimiser {
    /// Random search baseline
    ///
    /// Samples random parameter sets from the prior
    /// and returns the best. Useful as a baseline or for initialisation.
    ///
    /// Note: Number of samples controlled by n_samples parameter.
    RandomSearch,
}

/// Result of a point estimation optimisation.
#[derive(Debug, Clone)]
pub struct OptimisationResult {
    /// Best parameter vector found
    pub best_parameters: Vec<f64>,

    /// Log likelihood at best parameters (not including prior)
    pub best_log_likelihood: f64,

    /// Log posterior at best parameters (prior + likelihood)
    pub best_log_posterior: f64,

    /// Number of function evaluations performed
    pub n_evaluations: usize,

    /// Whether the optimiser converged
    pub converged: bool,

    /// Evaluations rejected because the prior, model or likelihood failed
    pub failures: Failures,
}

impl<R: crate::model_runner::ModelRunner, L: crate::likelihood::LikelihoodFn> PointEstimator<R, L> {
    /// Run point estimation optimisation.
    ///
    /// # Arguments
    ///
    /// * `optimiser` - Optimisation algorithm to use
    /// * `n_samples` - Number of samples to evaluate, at least 1
    /// * `seed` - Seed for sampling, so equal seeds give equal results. `None` seeds from the OS.
    ///
    /// # Returns
    ///
    /// An `OptimisationResult` with the best parameters and diagnostics.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = estimator.optimise(Optimiser::RandomSearch, 1000, None)?;
    /// ```
    pub fn optimise(
        &mut self,
        optimiser: Optimiser,
        n_samples: usize,
        seed: Option<u64>,
    ) -> Result<OptimisationResult> {
        if n_samples == 0 {
            return Err(Error::InvalidParameter(
                "n_samples must be at least 1".to_string(),
            ));
        }
        match optimiser {
            Optimiser::RandomSearch => self.optimise_random_search(n_samples, seed),
        }
    }

    /// Optimise using random search.
    ///
    /// Samples n_samples parameter sets from the prior and returns the best.
    fn optimise_random_search(
        &mut self,
        n_samples: usize,
        seed: Option<u64>,
    ) -> Result<OptimisationResult> {
        let samples = self.sample_initial_parameters(n_samples, seed);

        let mut best_parameters: Option<Vec<f64>> = None;
        let mut best_log_posterior = f64::NEG_INFINITY;
        let mut best_log_likelihood = f64::NEG_INFINITY;

        let mut failures = Failures::default();

        for parameters in samples.iter() {
            let log_posterior = match self.try_evaluate(parameters) {
                Ok(log_posterior) => log_posterior,
                Err(e) => {
                    failures.record(e)?;
                    f64::NEG_INFINITY
                }
            };

            if log_posterior > best_log_posterior {
                best_log_posterior = log_posterior;
                best_parameters = Some(parameters.clone());

                // Find corresponding log likelihood from history
                best_log_likelihood = self
                    .evaluated_log_likelihoods()
                    .iter()
                    .zip(self.evaluated_parameters().iter())
                    .find(|(_, p)| *p == parameters)
                    .map(|(ll, _)| *ll)
                    .unwrap_or(f64::NEG_INFINITY);
            }
        }

        let best = best_parameters.ok_or_else(|| {
            let reason = crate::no_valid_reason(failures.first.as_ref(), "sample");
            Error::SamplingError(format!("Random search found no valid samples ({reason})"))
        })?;

        Ok(OptimisationResult {
            best_parameters: best,
            best_log_likelihood,
            best_log_posterior,
            n_evaluations: n_samples,
            converged: true, // Random search always "converges"
            failures,
        })
    }

    /// Draw `n` parameter vectors from the prior.
    fn sample_initial_parameters(&self, n: usize, seed: Option<u64>) -> Vec<Vec<f64>> {
        self.parameters
            .sample_random_with_rng(n, &mut rng_from_seed(seed))
            .outer_iter()
            .map(|row| row.to_vec())
            .collect()
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
        parameter_names: Vec<String>,
    }

    impl QuadraticModel {
        fn new() -> Self {
            Self {
                parameter_names: vec!["x".to_string()],
            }
        }
    }

    impl ModelRunner for QuadraticModel {
        fn parameter_names(&self) -> &[String] {
            &self.parameter_names
        }

        fn run(&self, parameters: &[f64]) -> crate::Result<ModelOutput> {
            let x = parameters[0];
            let y = (x - 2.0).powi(2);

            let mut output = ModelOutput::new();
            let mut var = VariableOutput::new("y");
            var.add(0.0, y);

            output.add_variable(var);
            Ok(output)
        }
    }

    #[test]
    fn test_random_search_optimiser() {
        // Create a simple optimisation problem
        let mut parameters = ParameterSet::new();
        parameters.add("x", Box::new(Uniform::new(0.0, 4.0).unwrap()));

        // Target: y should be 0 (achieved when x=2)
        let mut target = Target::new();
        target.add_variable("y").add(0.0, 0.0, 0.1).unwrap();

        let runner = QuadraticModel::new();
        let likelihood = GaussianLikelihood::default();

        let mut estimator = PointEstimator::new(parameters, runner, likelihood, target).unwrap();

        // Run random search
        let result = estimator
            .optimise(Optimiser::RandomSearch, 100, None)
            .expect("Optimisation should succeed");

        // Check result
        assert_eq!(result.n_evaluations, 100);
        assert!(result.converged);
        assert!(result.best_log_posterior.is_finite());

        // Best parameter should be in bounds
        let best_x = result.best_parameters[0];
        assert!((0.0..=4.0).contains(&best_x), "Best x out of bounds");

        // Should have found something better than worst case
        assert!(result.best_log_posterior > f64::NEG_INFINITY);
    }

    #[test]
    fn test_random_search_multi_dimensional() {
        // 2D quadratic: y = (x1 - 1)^2 + (x2 - 3)^2
        // Minimum at (1, 3)
        struct Quadratic2D {
            parameter_names: Vec<String>,
        }

        impl Quadratic2D {
            fn new() -> Self {
                Self {
                    parameter_names: vec!["x1".to_string(), "x2".to_string()],
                }
            }
        }

        impl ModelRunner for Quadratic2D {
            fn parameter_names(&self) -> &[String] {
                &self.parameter_names
            }

            fn run(&self, parameters: &[f64]) -> crate::Result<ModelOutput> {
                let x1 = parameters[0];
                let x2 = parameters[1];
                let y = (x1 - 1.0).powi(2) + (x2 - 3.0).powi(2);

                let mut output = ModelOutput::new();
                let mut var = VariableOutput::new("y");
                var.add(0.0, y);

                output.add_variable(var);
                Ok(output)
            }
        }

        let mut parameters = ParameterSet::new();
        parameters.add("x1", Box::new(Uniform::new(-2.0, 4.0).unwrap()));
        parameters.add("x2", Box::new(Uniform::new(0.0, 6.0).unwrap()));

        let mut target = Target::new();
        target.add_variable("y").add(0.0, 0.0, 0.1).unwrap();

        let runner = Quadratic2D::new();
        let likelihood = GaussianLikelihood::default();

        let mut estimator = PointEstimator::new(parameters, runner, likelihood, target).unwrap();

        let result = estimator
            .optimise(Optimiser::RandomSearch, 500, None)
            .expect("Optimisation should succeed");

        // Check parameters are in bounds
        let best_x1 = result.best_parameters[0];
        let best_x2 = result.best_parameters[1];

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

    #[test]
    fn test_random_search_reports_underlying_error_when_every_evaluation_fails() {
        let mut parameters = ParameterSet::new();
        parameters.add("x", Box::new(Uniform::new(0.0, 4.0).unwrap()));
        let runner = crate::test_support::FailingModel::new(None);
        let mut estimator = PointEstimator::new(
            parameters,
            runner,
            GaussianLikelihood::default(),
            Target::new(),
        )
        .unwrap();

        let err = estimator
            .optimise(Optimiser::RandomSearch, 10, None)
            .expect_err("no evaluation succeeded");

        assert!(err.to_string().contains("no valid samples"), "got: {err}");
        assert!(err.to_string().contains("solver diverged"), "got: {err}");
    }

    #[test]
    fn test_random_search_samples_unbounded_prior() {
        let mut parameters = ParameterSet::new();
        parameters.add("x", Box::new(crate::Normal::new(2.0, 0.5).unwrap()));

        let mut target = Target::new();
        target.add_variable("y").add(0.0, 0.0, 0.1).unwrap();

        let mut estimator = PointEstimator::new(
            parameters,
            QuadraticModel::new(),
            GaussianLikelihood::default(),
            target,
        )
        .unwrap();

        let result = estimator
            .optimise(Optimiser::RandomSearch, 50, None)
            .expect("samples drawn from a Normal prior are valid");

        assert!(result.best_log_posterior.is_finite());
        assert!(estimator
            .evaluated_parameters()
            .iter()
            .all(|p| p[0].is_finite()));
    }

    fn failing_estimator(
        threshold: Option<f64>,
    ) -> PointEstimator<crate::test_support::FailingModel, GaussianLikelihood> {
        let mut parameters = ParameterSet::new();
        parameters.add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()));
        let mut target = Target::new();
        target.add_variable("value").add(0.0, 0.2, 0.1).unwrap();
        PointEstimator::new(
            parameters,
            crate::test_support::FailingModel::new(threshold),
            GaussianLikelihood::default(),
            target,
        )
        .unwrap()
    }

    #[test]
    fn test_random_search_seed_and_failures() {
        let run = |seed| {
            failing_estimator(Some(0.5))
                .optimise(Optimiser::RandomSearch, 40, Some(seed))
                .unwrap()
        };
        let (a, b) = (run(7), run(7));
        assert_eq!(a.best_parameters, b.best_parameters);
        assert_eq!(a.failures, b.failures);
        assert!(a.failures.count > 0);
        assert!(a.failures.first.unwrap().contains("solver diverged"));
    }

    #[test]
    fn test_random_search_rejects_zero_samples() {
        let err = failing_estimator(None)
            .optimise(Optimiser::RandomSearch, 0, None)
            .unwrap_err();
        assert!(err.to_string().contains("n_samples"), "got: {err}");
    }
}
