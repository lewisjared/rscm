//! Parameter set management for model calibration.
//!
//! This module provides the [`ParameterSet`] type for managing collections of
//! parameter distributions with both dict-based and fluent builder APIs.

use indexmap::IndexMap;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{Distribution, Error, Result};

/// A collection of named parameter distributions for Bayesian calibration.
///
/// ParameterSet manages the prior distributions for all model parameters,
/// maintaining consistent parameter ordering for vector-based operations.
/// Parameter names are preserved in insertion order via `IndexMap`.
///
/// # Construction Patterns
///
/// 1. **Dict-based**: `ParameterSet::from_map(map)` - create from existing map
/// 2. **Fluent builder**: `ParameterSet::new().add("x", dist).add("y", dist)` - build incrementally
///
/// # Parameter Ordering
///
/// Parameter order is **critical** - it determines the index mapping for parameter vectors
/// used in sampling and optimization. The order is:
/// - Preserved from insertion order (fluent builder)
/// - Preserved from IndexMap order (dict-based)
/// - Available via `param_names()` for index lookups
///
/// # Sampling Methods
///
/// - `sample_random(n)` - Simple random sampling from priors
/// - `sample_lhs(n)` - Latin Hypercube Sampling for better parameter space coverage
///
/// LHS is generally preferred for initialization as it provides more uniform exploration
/// of the parameter space with fewer samples.
///
/// # Examples
///
/// ```
/// use rscm_calibrate::{ParameterSet, Uniform, Normal, Distribution};
/// use indexmap::IndexMap;
///
/// // Fluent builder (recommended)
/// let mut params = ParameterSet::new();
/// params.add("sensitivity", Box::new(Uniform::new(0.5, 1.5).unwrap()));
/// params.add("offset", Box::new(Normal::new(0.0, 0.1).unwrap()));
///
/// // Sample initial parameter sets
/// let random_samples = params.sample_random(100);  // 100 random samples
/// let lhs_samples = params.sample_lhs(50);  // 50 LHS samples (better coverage)
///
/// // Get parameter ordering
/// let names = params.param_names();  // ["sensitivity", "offset"]
///
/// // Evaluate log-prior for a parameter vector
/// let params_vec = vec![1.0, 0.05];
/// let log_prior = params.log_prior(&params_vec).unwrap();
/// ```
///
/// ```
/// use rscm_calibrate::{ParameterSet, Uniform, Distribution};
/// use indexmap::IndexMap;
///
/// // Dict-based construction
/// let mut map = IndexMap::new();
/// map.insert("x".to_string(), Box::new(Uniform::new(0.0, 1.0).unwrap()) as Box<dyn Distribution>);
/// map.insert("y".to_string(), Box::new(Uniform::new(-1.0, 1.0).unwrap()) as Box<dyn Distribution>);
/// let params = ParameterSet::from_map(map);
/// ```
#[derive(Serialize, Deserialize)]
pub struct ParameterSet {
    /// Named parameters in insertion order.
    parameters: IndexMap<String, Box<dyn Distribution>>,
}

impl Clone for ParameterSet {
    fn clone(&self) -> Self {
        Self {
            parameters: self
                .parameters
                .iter()
                .map(|(k, v)| (k.clone(), v.clone_box()))
                .collect(),
        }
    }
}

impl ParameterSet {
    /// Create an empty parameter set.
    pub fn new() -> Self {
        Self {
            parameters: IndexMap::new(),
        }
    }

    /// Create a parameter set from a map of distributions.
    ///
    /// Parameter order is preserved from the IndexMap.
    pub fn from_map(parameters: IndexMap<String, Box<dyn Distribution>>) -> Self {
        Self { parameters }
    }

    /// Add a parameter distribution.
    ///
    /// Returns `&mut self` for fluent chaining.
    ///
    /// # Examples
    ///
    /// ```
    /// use rscm_calibrate::{ParameterSet, Uniform, Normal};
    ///
    /// let mut params = ParameterSet::new();
    /// params.add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()));
    /// params.add("y", Box::new(Normal::new(0.0, 1.0).unwrap()));
    /// ```
    pub fn add(
        &mut self,
        name: impl Into<String>,
        distribution: Box<dyn Distribution>,
    ) -> &mut Self {
        self.parameters.insert(name.into(), distribution);
        self
    }

    /// Get parameter names in definition order.
    ///
    /// The order matches the index used in parameter vectors.
    pub fn param_names(&self) -> Vec<&str> {
        self.parameters.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of parameters.
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Check if the parameter set is empty.
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Sample n parameter vectors randomly from the priors.
    ///
    /// Returns an array of shape `(n, n_params)` where each row is a parameter vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rscm_calibrate::{ParameterSet, Uniform};
    ///
    /// let mut params = ParameterSet::new();
    /// params.add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()));
    /// params.add("y", Box::new(Uniform::new(0.0, 1.0).unwrap()));
    ///
    /// let samples = params.sample_random(100);
    /// assert_eq!(samples.shape(), &[100, 2]);
    /// ```
    pub fn sample_random(&self, n: usize) -> ndarray::Array2<f64> {
        let mut rng = rand::thread_rng();
        self.sample_random_with_rng(n, &mut rng)
    }

    /// Sample n parameter vectors with a specific RNG.
    ///
    /// Useful for reproducible sampling with a seeded RNG.
    pub fn sample_random_with_rng<R: Rng>(&self, n: usize, rng: &mut R) -> ndarray::Array2<f64> {
        let n_params = self.len();
        let mut samples = ndarray::Array2::zeros((n, n_params));

        for i in 0..n {
            for (j, dist) in self.parameters.values().enumerate() {
                samples[[i, j]] = dist.sample_dyn(rng);
            }
        }

        samples
    }

    /// Sample n parameter vectors using Latin Hypercube Sampling.
    ///
    /// LHS ensures better coverage of the parameter space than random sampling.
    /// Each parameter dimension is divided into n equal-probability intervals,
    /// and exactly one sample is placed in each interval.
    ///
    /// Returns an array of shape `(n, n_params)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rscm_calibrate::{ParameterSet, Uniform};
    ///
    /// let mut params = ParameterSet::new();
    /// params.add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()));
    /// params.add("y", Box::new(Uniform::new(0.0, 1.0).unwrap()));
    ///
    /// let samples = params.sample_lhs(100);
    /// assert_eq!(samples.shape(), &[100, 2]);
    /// ```
    pub fn sample_lhs(&self, n: usize) -> ndarray::Array2<f64> {
        let mut rng = rand::thread_rng();
        self.sample_lhs_with_rng(n, &mut rng)
    }

    /// Sample n parameter vectors using LHS with a specific RNG.
    pub fn sample_lhs_with_rng<R: Rng>(&self, n: usize, rng: &mut R) -> ndarray::Array2<f64> {
        let n_params = self.len();
        let mut samples = ndarray::Array2::zeros((n, n_params));

        // For each parameter dimension
        for (j, (_, dist)) in self.parameters.iter().enumerate() {
            // Generate stratified uniform samples in [0, 1]
            let mut stratified: Vec<f64> = (0..n)
                .map(|i| {
                    let interval_size = 1.0 / n as f64;
                    let interval_start = i as f64 * interval_size;
                    interval_start + rng.gen::<f64>() * interval_size
                })
                .collect();

            // Shuffle to break correlation between dimensions
            use rand::seq::SliceRandom;
            stratified.shuffle(rng);

            // Transform from [0, 1] to parameter space using quantile function
            for (i, &u) in stratified.iter().enumerate() {
                samples[[i, j]] = quantile_via_sampling(dist.as_ref(), u, rng);
            }
        }

        samples
    }

    /// Compute the log prior probability of a parameter vector.
    ///
    /// The parameter vector must match the order from [`param_names()`](Self::param_names).
    ///
    /// # Errors
    ///
    /// Returns an error if the parameter vector length doesn't match the number of parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use rscm_calibrate::{ParameterSet, Uniform};
    ///
    /// let mut params = ParameterSet::new();
    /// params.add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()));
    /// params.add("y", Box::new(Uniform::new(0.0, 1.0).unwrap()));
    ///
    /// let log_prior = params.log_prior(&[0.5, 0.5]).unwrap();
    /// assert!(log_prior.is_finite());
    /// ```
    pub fn log_prior(&self, params: &[f64]) -> Result<f64> {
        if params.len() != self.len() {
            return Err(Error::InvalidParameter(format!(
                "Parameter vector length {} does not match parameter set size {}",
                params.len(),
                self.len()
            )));
        }

        let mut log_p = 0.0;
        for (value, dist) in params.iter().zip(self.parameters.values()) {
            log_p += dist.ln_pdf(*value);
        }

        Ok(log_p)
    }

    /// Extract bounds for all parameters.
    ///
    /// Returns `(lower_bounds, upper_bounds)` where each vector has length `n_params`.
    /// Unbounded parameters use `f64::NEG_INFINITY` and `f64::INFINITY`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rscm_calibrate::{ParameterSet, Uniform, Normal};
    ///
    /// let mut params = ParameterSet::new();
    /// params.add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()));
    /// params.add("y", Box::new(Normal::new(0.0, 1.0).unwrap()));
    ///
    /// let (lower, upper) = params.bounds();
    /// assert_eq!(lower, vec![0.0, f64::NEG_INFINITY]);
    /// assert_eq!(upper, vec![1.0, f64::INFINITY]);
    /// ```
    pub fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let mut lower = Vec::with_capacity(self.len());
        let mut upper = Vec::with_capacity(self.len());

        for dist in self.parameters.values() {
            match dist.bounds() {
                Some((low, high)) => {
                    lower.push(low);
                    upper.push(high);
                }
                None => {
                    lower.push(f64::NEG_INFINITY);
                    upper.push(f64::INFINITY);
                }
            }
        }

        (lower, upper)
    }
}

impl Default for ParameterSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Approximate quantile function via binary search on CDF.
///
/// This is used for LHS sampling when distributions don't have closed-form quantile functions.
/// For simple distributions like Uniform, this could be optimised, but the general approach
/// works for all distributions.
fn quantile_via_sampling<R: Rng>(dist: &dyn Distribution, u: f64, rng: &mut R) -> f64 {
    // For bounded distributions, use bounds-based binary search
    if let Some((low, high)) = dist.bounds() {
        // Special case: Uniform distribution has closed-form quantile
        // This avoids the iterative search for the most common case
        if low.is_finite() && high.is_finite() {
            // Check if this is approximately uniform by checking PDF is constant
            let pdf_low = dist.ln_pdf(low + (high - low) * 0.1);
            let pdf_high = dist.ln_pdf(low + (high - low) * 0.9);
            if (pdf_low - pdf_high).abs() < 1e-10 {
                return low + u * (high - low);
            }
        }

        // For other bounded distributions, use binary search
        let mut low = low;
        let mut high = if high.is_finite() { high } else { 1e10 };

        for _ in 0..50 {
            let mid = (low + high) / 2.0;
            // Estimate CDF via sampling (crude but works)
            let mut count = 0;
            for _ in 0..100 {
                if dist.sample_dyn(rng) <= mid {
                    count += 1;
                }
            }
            let cdf = count as f64 / 100.0;

            if (cdf - u).abs() < 0.01 {
                return mid;
            }
            if cdf < u {
                low = mid;
            } else {
                high = mid;
            }
        }
        return (low + high) / 2.0;
    }

    // For unbounded distributions, use a wider search
    // This is a fallback - in practice, most distributions should be bounded for calibration
    let mut samples: Vec<f64> = (0..1000).map(|_| dist.sample_dyn(rng)).collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((u * 999.0) as usize).min(999);
    samples[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Normal, Uniform};
    use approx::assert_abs_diff_eq;
    use rand::SeedableRng;

    #[test]
    fn test_from_map() {
        let mut map = IndexMap::new();
        map.insert(
            "x".to_string(),
            Box::new(Uniform::new(0.0, 1.0).unwrap()) as Box<dyn Distribution>,
        );
        map.insert(
            "y".to_string(),
            Box::new(Uniform::new(-1.0, 1.0).unwrap()) as Box<dyn Distribution>,
        );

        let params = ParameterSet::from_map(map);
        assert_eq!(params.len(), 2);
        assert_eq!(params.param_names(), vec!["x", "y"]);
    }

    #[test]
    fn test_fluent_builder() {
        let params = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .add("y", Box::new(Uniform::new(-1.0, 1.0).unwrap()))
            .clone();

        assert_eq!(params.len(), 2);
        assert_eq!(params.param_names(), vec!["x", "y"]);
    }

    #[test]
    fn test_param_names_order() {
        let params = ParameterSet::new()
            .add("a", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .add("b", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .add("c", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .clone();

        assert_eq!(params.param_names(), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_sample_random() {
        let params = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .add("y", Box::new(Uniform::new(-1.0, 1.0).unwrap()))
            .clone();

        let samples = params.sample_random(100);
        assert_eq!(samples.shape(), &[100, 2]);

        // Check all samples are in bounds
        for i in 0..100 {
            assert!(samples[[i, 0]] >= 0.0 && samples[[i, 0]] <= 1.0);
            assert!(samples[[i, 1]] >= -1.0 && samples[[i, 1]] <= 1.0);
        }
    }

    #[test]
    fn test_sample_random_reproducible() {
        let params = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .clone();

        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);

        let samples1 = params.sample_random_with_rng(10, &mut rng1);
        let samples2 = params.sample_random_with_rng(10, &mut rng2);

        assert_eq!(samples1, samples2);
    }

    #[test]
    fn test_sample_lhs() {
        let params = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .add("y", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .clone();

        let samples = params.sample_lhs(10);
        assert_eq!(samples.shape(), &[10, 2]);

        // Check all samples are in bounds
        for i in 0..10 {
            assert!(samples[[i, 0]] >= 0.0 && samples[[i, 0]] <= 1.0);
            assert!(samples[[i, 1]] >= 0.0 && samples[[i, 1]] <= 1.0);
        }
    }

    #[test]
    fn test_sample_lhs_coverage() {
        // LHS should have better coverage than random sampling.
        // For uniform [0, 1] with n=10, each 0.1 interval should have exactly 1 sample.
        let params = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .clone();

        let samples = params.sample_lhs(10);

        // Count samples in each decile
        let mut counts = vec![0; 10];
        for i in 0..10 {
            let val = samples[[i, 0]];
            let bin = (val * 10.0).floor() as usize;
            counts[bin.min(9)] += 1;
        }

        // Each bin should have exactly 1 sample for uniform LHS
        for count in counts {
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn test_log_prior() {
        let params = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .add("y", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .clone();

        // Valid parameters
        let log_p = params.log_prior(&[0.5, 0.5]).unwrap();
        assert!(log_p.is_finite());

        // Out of bounds
        let log_p = params.log_prior(&[1.5, 0.5]).unwrap();
        assert_eq!(log_p, f64::NEG_INFINITY);

        // Wrong length
        assert!(params.log_prior(&[0.5]).is_err());
        assert!(params.log_prior(&[0.5, 0.5, 0.5]).is_err());
    }

    #[test]
    fn test_log_prior_sum() {
        let params = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 2.0).unwrap()))
            .add("y", Box::new(Uniform::new(0.0, 2.0).unwrap()))
            .clone();

        let log_p = params.log_prior(&[1.0, 1.0]).unwrap();

        // log(1/2) + log(1/2) = -log(2) - log(2) = -2*log(2)
        let expected = -2.0 * 2.0_f64.ln();
        assert_abs_diff_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_bounds() {
        let params = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .add("y", Box::new(Normal::new(0.0, 1.0).unwrap()))
            .add("z", Box::new(Uniform::new(-2.0, 2.0).unwrap()))
            .clone();

        let (lower, upper) = params.bounds();

        assert_eq!(lower, vec![0.0, f64::NEG_INFINITY, -2.0]);
        assert_eq!(upper, vec![1.0, f64::INFINITY, 2.0]);
    }

    #[test]
    fn test_empty_parameter_set() {
        let params = ParameterSet::new();
        assert!(params.is_empty());
        assert_eq!(params.len(), 0);
        assert_eq!(params.param_names(), Vec::<&str>::new());

        let samples = params.sample_random(10);
        assert_eq!(samples.shape(), &[10, 0]);
    }
}
