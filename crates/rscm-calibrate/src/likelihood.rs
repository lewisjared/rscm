//! Likelihood functions for comparing model output to observations.

use crate::parameter_set::ParameterSet;
use crate::target::{Observation, Target, VariableTarget};
use crate::{Error, Result};
use std::collections::HashMap;

/// Model output for a single variable at multiple times.
#[derive(Debug, Clone)]
pub struct VariableOutput {
    /// Variable name
    pub name: String,
    /// Time-indexed values (time -> value)
    pub values: HashMap<String, f64>,
}

impl VariableOutput {
    /// Create a new variable output.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            values: HashMap::new(),
        }
    }

    /// Add a time-value pair.
    pub fn add(&mut self, time: f64, value: f64) -> &mut Self {
        self.values.insert(time_key(time), value);
        self
    }

    /// Get the value at a specific time (with tolerance for floating-point comparison).
    pub fn get(&self, time: f64) -> Option<f64> {
        self.values.get(&time_key(time)).copied()
    }
}

/// Convert a time value to a string key for HashMap lookups.
///
/// This rounds to a reasonable precision to avoid floating-point comparison issues.
fn time_key(time: f64) -> String {
    format!("{:.6}", time)
}

/// Model output collection.
#[derive(Debug, Clone, Default)]
pub struct ModelOutput {
    /// Variables indexed by name
    pub variables: HashMap<String, VariableOutput>,
}

impl ModelOutput {
    /// Create a new empty model output.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    /// Add a variable output.
    pub fn add_variable(&mut self, var: VariableOutput) -> &mut Self {
        self.variables.insert(var.name.clone(), var);
        self
    }

    /// Get a variable by name.
    pub fn get_variable(&self, name: &str) -> Option<&VariableOutput> {
        self.variables.get(name)
    }
}

/// Trait for likelihood functions in Bayesian calibration.
///
/// Likelihood functions quantify how well model output matches observational data.
/// They compute the log-probability of observing the target data given the model output.
///
/// # Bayesian Context
///
/// The likelihood `L(θ) = p(data | θ)` measures how probable the observed data is
/// under model parameters θ. Combined with priors, this forms the posterior:
///
/// ```text
/// p(θ | data) ∝ p(data | θ) × p(θ)
///            ∝ L(θ) × prior(θ)
/// ```
///
/// # Log-Likelihood
///
/// We work with log-likelihoods to avoid numerical underflow with small probabilities.
/// The log of a product becomes a sum:
///
/// ```text
/// ln L(θ) = Σ ln p(y_i | θ)
/// ```
///
/// # Normalisation
///
/// For MCMC sampling, normalisation constants can be omitted since they cancel in
/// acceptance ratios. Implementations may return unnormalised log-likelihoods.
pub trait LikelihoodFn {
    /// Compute the log-likelihood of the model output given the target observations.
    ///
    /// # Arguments
    ///
    /// * `output` - Model output to compare against observations
    /// * `target` - Target observations with uncertainties
    ///
    /// # Returns
    ///
    /// The log-likelihood value. May be unnormalised (missing constant terms).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Required variables are missing from the model output
    /// - Required times are missing from the model output
    /// - Model output contains NaN or infinite values (indicates model failure)
    fn log_likelihood(&self, output: &ModelOutput, target: &Target) -> Result<f64>;
}

/// Gaussian likelihood function with independent errors.
///
/// Assumes each observation has independent Gaussian error with known uncertainty σ:
///
/// ```text
/// p(y_obs | y_model, σ) = (1/√(2πσ²)) exp(-(y_obs - y_model)² / (2σ²))
/// ```
///
/// The log-likelihood sums over all observations:
///
/// ```text
/// ln L = Σ ln p(y_i_obs | y_i_model, σ_i)
///      = -0.5 × Σ[(y_i_obs - y_i_model)² / σ_i²] + normalisation
/// ```
///
/// # Normalisation
///
/// By default, the normalisation constant `(-0.5 × ln(2π) - ln(σ))` is **omitted**
/// since it cancels in MCMC acceptance ratios. Set `normalise: true` for
/// applications requiring absolute log-likelihood values (e.g., model comparison via BIC/AIC).
///
/// # Missing Data
///
/// If the model output is missing a required variable or time point, an error is returned.
/// NaN or infinite model outputs also return errors (indicating model failure).
///
/// # Example
///
/// ```
/// use rscm_calibrate::likelihood::{GaussianLikelihood, LikelihoodFn, ModelOutput, VariableOutput};
/// use rscm_calibrate::Target;
///
/// let likelihood = GaussianLikelihood::new();
///
/// let mut target = Target::new();
/// target.add_variable("Temperature")
///     .add(2020.0, 1.2, 0.1).unwrap();
///
/// let mut output = ModelOutput::new();
/// let mut var = VariableOutput::new("Temperature");
/// var.add(2020.0, 1.15);
/// output.add_variable(var);
///
/// let log_l = likelihood.log_likelihood(&output, &target).unwrap();
/// assert!(log_l.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct GaussianLikelihood {
    /// Whether to include normalisation constant (default: false for MCMC)
    pub normalise: bool,
}

impl GaussianLikelihood {
    /// Create a new Gaussian likelihood function without normalisation.
    ///
    /// Use `with_normalisation()` if you need the full normalised log-likelihood.
    pub fn new() -> Self {
        Self { normalise: false }
    }

    /// Create with normalisation constant included.
    pub fn with_normalisation() -> Self {
        Self { normalise: true }
    }

    /// Compute log-likelihood for a single observation.
    fn observation_log_likelihood(&self, obs: &Observation, model_value: f64) -> f64 {
        let residual = obs.value - model_value;
        let chi_squared = (residual * residual) / (obs.uncertainty * obs.uncertainty);

        let mut log_l = -0.5 * chi_squared;

        if self.normalise {
            log_l -= 0.5 * (2.0 * std::f64::consts::PI).ln();
            log_l -= obs.uncertainty.ln();
        }

        log_l
    }

    /// Compute log-likelihood for a single variable.
    fn variable_log_likelihood(
        &self,
        var_output: &VariableOutput,
        var_target: &VariableTarget,
    ) -> Result<f64> {
        let mut log_l = 0.0;

        for obs in &var_target.observations {
            let model_value = var_output.get(obs.time).ok_or_else(|| {
                Error::ModelError(format!(
                    "Model output missing time {} for variable {}",
                    obs.time, var_target.name
                ))
            })?;

            if !model_value.is_finite() {
                return Err(Error::ModelError(format!(
                    "Model output contains non-finite value for {} at time {}",
                    var_target.name, obs.time
                )));
            }

            log_l += self.observation_log_likelihood(obs, model_value);
        }

        Ok(log_l)
    }
}

impl Default for GaussianLikelihood {
    fn default() -> Self {
        Self::new()
    }
}

impl LikelihoodFn for GaussianLikelihood {
    fn log_likelihood(&self, output: &ModelOutput, target: &Target) -> Result<f64> {
        let mut total_log_l = 0.0;

        for (var_name, var_target) in target.variables() {
            let var_output = output.get_variable(var_name).ok_or_else(|| {
                Error::ModelError(format!("Model output missing variable: {}", var_name))
            })?;

            total_log_l += self.variable_log_likelihood(var_output, var_target)?;
        }

        Ok(total_log_l)
    }
}

/// `(log_posterior, log_likelihood)` of `values`, running the model only inside the prior support.
///
/// A point outside the prior support is `Ok(-inf)`.
/// A failed prior, model or likelihood evaluation is `Err`.
pub(crate) fn log_posterior<L: LikelihoodFn>(
    parameters: &ParameterSet,
    values: &[f64],
    run: impl FnOnce() -> Result<ModelOutput>,
    likelihood: &L,
    target: &Target,
) -> Result<(f64, f64)> {
    let log_prior = parameters.log_prior(values)?;
    if !log_prior.is_finite() {
        return Ok((f64::NEG_INFINITY, f64::NEG_INFINITY));
    }
    let log_likelihood = likelihood.log_likelihood(&run()?, target)?;
    Ok((log_prior + log_likelihood, log_likelihood))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_output_creation() {
        let mut output = VariableOutput::new("Temperature");
        output.add(2020.0, 1.2);
        output.add(2021.0, 1.3);

        assert_eq!(output.get(2020.0), Some(1.2));
        assert_eq!(output.get(2021.0), Some(1.3));
        assert_eq!(output.get(2022.0), None);
    }

    #[test]
    fn time_key_tolerance() {
        let mut output = VariableOutput::new("Temperature");
        output.add(2020.0, 1.2);

        // Should handle small floating-point differences
        assert_eq!(output.get(2020.0), Some(1.2));
        assert_eq!(output.get(2020.0000001), Some(1.2));
    }

    #[test]
    fn model_output_collection() {
        let mut output = ModelOutput::new();

        let mut temp = VariableOutput::new("Temperature");
        temp.add(2020.0, 1.2);

        let mut ohc = VariableOutput::new("OHC");
        ohc.add(2020.0, 200.0);

        output.add_variable(temp);
        output.add_variable(ohc);

        assert!(output.get_variable("Temperature").is_some());
        assert!(output.get_variable("OHC").is_some());
        assert!(output.get_variable("Missing").is_none());
    }

    #[test]
    fn gaussian_likelihood_perfect_match() {
        let likelihood = GaussianLikelihood::new();

        let mut target = Target::new();
        target
            .add_variable("Temperature")
            .add(2020.0, 1.2, 0.1)
            .unwrap()
            .add(2021.0, 1.3, 0.1)
            .unwrap();

        let mut output = ModelOutput::new();
        let mut temp = VariableOutput::new("Temperature");
        temp.add(2020.0, 1.2);
        temp.add(2021.0, 1.3);
        output.add_variable(temp);

        let log_l = likelihood.log_likelihood(&output, &target).unwrap();
        // Perfect match should give log_l = 0 (without normalisation)
        assert_eq!(log_l, 0.0);
    }

    #[test]
    fn gaussian_likelihood_with_residuals() {
        let likelihood = GaussianLikelihood::new();

        let mut target = Target::new();
        target
            .add_variable("Temperature")
            .add(2020.0, 1.0, 0.1)
            .unwrap();

        let mut output = ModelOutput::new();
        let mut temp = VariableOutput::new("Temperature");
        temp.add(2020.0, 1.1); // 0.1 off, 1-sigma error
        output.add_variable(temp);

        let log_l = likelihood.log_likelihood(&output, &target).unwrap();
        // Residual = 0.1, uncertainty = 0.1
        // chi^2 = (0.1/0.1)^2 = 1
        // log_l = -0.5 * 1 = -0.5
        assert!((log_l - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn gaussian_likelihood_multiple_observations() {
        let likelihood = GaussianLikelihood::new();

        let mut target = Target::new();
        target
            .add_variable("Temperature")
            .add(2020.0, 1.0, 0.1)
            .unwrap()
            .add(2021.0, 1.1, 0.1)
            .unwrap();

        let mut output = ModelOutput::new();
        let mut temp = VariableOutput::new("Temperature");
        temp.add(2020.0, 1.1); // chi^2 = 1
        temp.add(2021.0, 1.2); // chi^2 = 1
        output.add_variable(temp);

        let log_l = likelihood.log_likelihood(&output, &target).unwrap();
        // Total chi^2 = 2, log_l = -0.5 * 2 = -1.0
        assert!((log_l - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn gaussian_likelihood_multiple_variables() {
        let likelihood = GaussianLikelihood::new();

        let mut target = Target::new();
        target
            .add_variable("Temperature")
            .add(2020.0, 1.0, 0.1)
            .unwrap();
        target.add_variable("OHC").add(2020.0, 200.0, 10.0).unwrap();

        let mut output = ModelOutput::new();
        let mut temp = VariableOutput::new("Temperature");
        temp.add(2020.0, 1.1); // chi^2 = 1
        let mut ohc = VariableOutput::new("OHC");
        ohc.add(2020.0, 210.0); // residual = 10, uncertainty = 10, chi^2 = 1
        output.add_variable(temp);
        output.add_variable(ohc);

        let log_l = likelihood.log_likelihood(&output, &target).unwrap();
        // Total chi^2 = 2, log_l = -0.5 * 2 = -1.0
        assert!((log_l - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn gaussian_likelihood_missing_variable() {
        let likelihood = GaussianLikelihood::new();

        let mut target = Target::new();
        target
            .add_variable("Temperature")
            .add(2020.0, 1.0, 0.1)
            .unwrap();

        let output = ModelOutput::new(); // Empty output

        let result = likelihood.log_likelihood(&output, &target);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::ModelError(_)));
    }

    #[test]
    fn gaussian_likelihood_missing_time() {
        let likelihood = GaussianLikelihood::new();

        let mut target = Target::new();
        target
            .add_variable("Temperature")
            .add(2020.0, 1.0, 0.1)
            .unwrap()
            .add(2021.0, 1.1, 0.1)
            .unwrap();

        let mut output = ModelOutput::new();
        let mut temp = VariableOutput::new("Temperature");
        temp.add(2020.0, 1.0); // Missing 2021
        output.add_variable(temp);

        let result = likelihood.log_likelihood(&output, &target);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::ModelError(_)));
    }

    #[test]
    fn gaussian_likelihood_nan_value() {
        let likelihood = GaussianLikelihood::new();

        let mut target = Target::new();
        target
            .add_variable("Temperature")
            .add(2020.0, 1.0, 0.1)
            .unwrap();

        let mut output = ModelOutput::new();
        let mut temp = VariableOutput::new("Temperature");
        temp.add(2020.0, f64::NAN);
        output.add_variable(temp);

        let result = likelihood.log_likelihood(&output, &target);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::ModelError(_)));
    }

    #[test]
    fn gaussian_likelihood_with_normalisation() {
        let likelihood = GaussianLikelihood::with_normalisation();

        let mut target = Target::new();
        target
            .add_variable("Temperature")
            .add(2020.0, 1.0, 0.1)
            .unwrap();

        let mut output = ModelOutput::new();
        let mut temp = VariableOutput::new("Temperature");
        temp.add(2020.0, 1.0); // Perfect match
        output.add_variable(temp);

        let log_l = likelihood.log_likelihood(&output, &target).unwrap();
        // With normalisation: -0.5 * ln(2π) - ln(0.1)
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln() - 0.1_f64.ln();
        assert!((log_l - expected).abs() < 1e-10);
    }
}
