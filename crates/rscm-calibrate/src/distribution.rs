//! Prior distributions for model parameters.
//!
//! Distributions provide:
//! - Random sampling for initialisation and prior generation
//! - Log probability density for likelihood evaluation
//! - Bounds for constrained optimisation

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// A probability distribution for model parameters.
///
/// This trait provides the interface for parameter priors in Bayesian calibration.
/// All distributions are object-safe and support serialization via `typetag::serde`.
///
/// # Usage in Calibration
///
/// Distributions serve two purposes:
/// 1. **Sampling**: Generate initial parameter values and proposals during MCMC
/// 2. **Density evaluation**: Compute log-prior probabilities for Bayesian inference
///
/// # Example
///
/// ```
/// use rscm_calibrate::distribution::{Distribution, Uniform, Normal};
///
/// // Create distributions
/// let uniform = Uniform::new(0.0, 1.0).unwrap();
/// let normal = Normal::new(0.0, 1.0).unwrap();
///
/// // Sample values (use as trait objects for parameterless sample)
/// let uniform_ref: &dyn Distribution = &uniform;
/// let normal_ref: &dyn Distribution = &normal;
/// let x = uniform_ref.sample();
/// let y = normal_ref.sample();
///
/// // Evaluate log-probability density
/// let log_p = uniform.ln_pdf(0.5);
/// assert!(log_p.is_finite());
/// ```
///
/// # Object Safety
///
/// This trait is object-safe and supports dynamic dispatch via trait objects (`Box<dyn Distribution>`).
/// This enables heterogeneous parameter sets with different distribution types.
#[typetag::serde(tag = "type")]
pub trait Distribution: Send + Sync {
    /// Sample a value using the provided RNG.
    ///
    /// This method is object-safe and works with trait objects.
    /// For convenience when using concrete types, use the `sample()` method instead.
    fn sample_dyn(&self, rng: &mut dyn rand::RngCore) -> f64;

    /// Compute the natural logarithm of the probability density at `x`.
    ///
    /// Returns `f64::NEG_INFINITY` for values outside the support.
    ///
    /// # Note on Normalization
    ///
    /// For bounded distributions, the returned value may be unnormalized
    /// (missing the normalization constant). This is acceptable for MCMC
    /// where the constant cancels in acceptance ratios.
    fn ln_pdf(&self, x: f64) -> f64;

    /// Get the support bounds `[min, max]` of the distribution.
    ///
    /// Returns `Some((min, max))` for bounded distributions, or `None`
    /// for distributions with unbounded support (e.g., Normal).
    ///
    /// These bounds are used by optimizers to constrain parameter searches.
    fn bounds(&self) -> Option<(f64, f64)>;

    /// Clone the distribution into a boxed trait object.
    ///
    /// Required for cloning `Box<dyn Distribution>`.
    fn clone_box(&self) -> Box<dyn Distribution>;
}

// Helper extension trait for concrete types with a generic sample method
impl dyn Distribution {
    /// Sample a value from the distribution using the thread-local RNG.
    pub fn sample(&self) -> f64 {
        let mut rng = rand::thread_rng();
        self.sample_dyn(&mut rng)
    }
}

/// Uniform distribution over the interval `[low, high]`.
///
/// All values in the interval have equal probability density `1/(high - low)`.
///
/// # Example
///
/// ```
/// use rscm_calibrate::distribution::{Distribution, Uniform};
///
/// let dist = Uniform::new(0.0, 10.0).unwrap();
///
/// // Sample values are always in bounds (use trait object for parameterless sample)
/// let dist_ref: &dyn Distribution = &dist;
/// let x = dist_ref.sample();
/// assert!(x >= 0.0 && x <= 10.0);
///
/// // Log-PDF is constant inside bounds
/// let log_p = dist.ln_pdf(5.0);
/// assert_eq!(log_p, -(10.0_f64.ln()));
///
/// // Zero probability outside bounds
/// assert_eq!(dist.ln_pdf(-1.0), f64::NEG_INFINITY);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Uniform {
    low: f64,
    high: f64,
}

impl Uniform {
    /// Create a uniform distribution over `[low, high]`.
    ///
    /// # Arguments
    ///
    /// * `low` - Lower bound (inclusive)
    /// * `high` - Upper bound (inclusive)
    ///
    /// # Errors
    ///
    /// Returns an error if `low >= high`.
    pub fn new(low: f64, high: f64) -> Result<Self> {
        if low >= high {
            return Err(Error::InvalidParameter(format!(
                "Uniform: low ({}) must be less than high ({})",
                low, high
            )));
        }
        Ok(Self { low, high })
    }

    /// Get the lower bound.
    pub fn low(&self) -> f64 {
        self.low
    }

    /// Get the upper bound.
    pub fn high(&self) -> f64 {
        self.high
    }
}

#[typetag::serde]
impl Distribution for Uniform {
    fn sample_dyn(&self, rng: &mut dyn rand::RngCore) -> f64 {
        self.low + rng.gen::<f64>() * (self.high - self.low)
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.low || x > self.high {
            f64::NEG_INFINITY
        } else {
            -(self.high - self.low).ln()
        }
    }

    fn bounds(&self) -> Option<(f64, f64)> {
        Some((self.low, self.high))
    }

    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }
}

impl Uniform {
    /// Sample a value from the distribution.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.low + rng.gen::<f64>() * (self.high - self.low)
    }
}

/// Normal (Gaussian) distribution with mean μ and standard deviation σ.
///
/// The probability density function is:
///
/// ```text
/// p(x) = (1 / (σ√(2π))) exp(-(x - μ)² / (2σ²))
/// ```
///
/// This distribution has unbounded support (-∞, +∞).
///
/// # Example
///
/// ```
/// use rscm_calibrate::distribution::{Distribution, Normal};
///
/// let dist = Normal::new(0.0, 1.0).unwrap();
///
/// // Sample from standard normal (use trait object for parameterless sample)
/// let dist_ref: &dyn Distribution = &dist;
/// let x = dist_ref.sample();
///
/// // Evaluate log-PDF (maximum at mean)
/// let log_p_mean = dist.ln_pdf(0.0);
/// let log_p_away = dist.ln_pdf(2.0);
/// assert!(log_p_mean > log_p_away);
///
/// // No bounds (unbounded support)
/// assert_eq!(dist.bounds(), None);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normal {
    mean: f64,
    std_dev: f64,
}

impl Normal {
    /// Create a normal distribution with mean μ and standard deviation σ.
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean (center) of the distribution
    /// * `std_dev` - Standard deviation (must be positive)
    ///
    /// # Errors
    ///
    /// Returns an error if `std_dev <= 0`.
    pub fn new(mean: f64, std_dev: f64) -> Result<Self> {
        if std_dev <= 0.0 {
            return Err(Error::InvalidParameter(format!(
                "Normal: std_dev ({}) must be positive",
                std_dev
            )));
        }
        Ok(Self { mean, std_dev })
    }

    /// Get the mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get the standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }
}

#[typetag::serde]
impl Distribution for Normal {
    fn sample_dyn(&self, rng: &mut dyn rand::RngCore) -> f64 {
        let dist = rand_distr::Normal::new(self.mean, self.std_dev)
            .expect("Normal parameters validated at construction");
        rng.sample(dist)
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        -0.5 * z * z - self.std_dev.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    fn bounds(&self) -> Option<(f64, f64)> {
        None
    }

    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }
}

impl Normal {
    /// Sample a value from the distribution.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let dist = rand_distr::Normal::new(self.mean, self.std_dev)
            .expect("Normal parameters validated at construction");
        rng.sample(dist)
    }
}

/// Log-normal distribution: `ln(X) ~ Normal(μ, σ)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogNormal {
    mu: f64,
    sigma: f64,
}

impl LogNormal {
    /// Create a log-normal distribution where `ln(X) ~ Normal(μ, σ)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `sigma <= 0`.
    pub fn new(mu: f64, sigma: f64) -> Result<Self> {
        if sigma <= 0.0 {
            return Err(Error::InvalidParameter(format!(
                "LogNormal: sigma ({}) must be positive",
                sigma
            )));
        }
        Ok(Self { mu, sigma })
    }

    /// Create a log-normal distribution from the mean and standard deviation of `X`.
    ///
    /// Converts to the underlying log-space parameters μ and σ.
    ///
    /// # Errors
    ///
    /// Returns an error if `mean <= 0` or `std_dev <= 0`.
    pub fn from_mean_std(mean: f64, std_dev: f64) -> Result<Self> {
        if mean <= 0.0 {
            return Err(Error::InvalidParameter(format!(
                "LogNormal: mean ({}) must be positive",
                mean
            )));
        }
        if std_dev <= 0.0 {
            return Err(Error::InvalidParameter(format!(
                "LogNormal: std_dev ({}) must be positive",
                std_dev
            )));
        }

        let variance = std_dev * std_dev;
        let mean_sq = mean * mean;
        let sigma_sq = (variance / mean_sq + 1.0).ln();
        let mu = mean.ln() - 0.5 * sigma_sq;

        Ok(Self {
            mu,
            sigma: sigma_sq.sqrt(),
        })
    }

    /// Get the μ parameter (mean of ln(X)).
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Get the σ parameter (standard deviation of ln(X)).
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

#[typetag::serde]
impl Distribution for LogNormal {
    fn sample_dyn(&self, rng: &mut dyn rand::RngCore) -> f64 {
        let dist = rand_distr::LogNormal::new(self.mu, self.sigma)
            .expect("LogNormal parameters validated at construction");
        rng.sample(dist)
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let ln_x = x.ln();
        let z = (ln_x - self.mu) / self.sigma;
        -0.5 * z * z - ln_x - self.sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    fn bounds(&self) -> Option<(f64, f64)> {
        Some((0.0, f64::INFINITY))
    }

    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }
}

impl LogNormal {
    /// Sample a value from the distribution.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let dist = rand_distr::LogNormal::new(self.mu, self.sigma)
            .expect("LogNormal parameters validated at construction");
        rng.sample(dist)
    }
}

/// Wrapper that applies hard bounds to any distribution via rejection sampling.
///
/// This allows constraining unbounded distributions (e.g., Normal) to a finite interval.
/// Samples are rejected and resampled until they fall within `[low, high]`.
///
/// # Algorithm
///
/// - **Sampling**: Rejection sampling - draw from inner distribution until value is in bounds
/// - **Density**: Returns inner distribution's log-PDF for values in bounds, -∞ otherwise
///
/// # Performance Note
///
/// Rejection sampling can be inefficient if the bounds exclude most of the inner
/// distribution's probability mass. For example, bounding a Normal(0, 1) to [10, 20]
/// will require many samples on average.
///
/// # Example
///
/// ```
/// use rscm_calibrate::distribution::{Distribution, Normal, Bound};
///
/// // Truncated normal: Normal(0, 1) bounded to [-2, 2]
/// let normal = Box::new(Normal::new(0.0, 1.0).unwrap());
/// let bounded = Bound::new(normal, -2.0, 2.0).unwrap();
///
/// // All samples fall within bounds (use trait object for parameterless sample)
/// let bounded_ref: &dyn Distribution = &bounded;
/// let x = bounded_ref.sample();
/// assert!(x >= -2.0 && x <= 2.0);
///
/// // PDF is zero outside bounds
/// assert_eq!(bounded.ln_pdf(-3.0), f64::NEG_INFINITY);
///
/// // Bounds are available for optimizers
/// assert_eq!(bounded.bounds(), Some((-2.0, 2.0)));
/// ```
#[derive(Serialize, Deserialize)]
pub struct Bound {
    distribution: Box<dyn Distribution>,
    low: f64,
    high: f64,
}

impl std::fmt::Debug for Bound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bound")
            .field("low", &self.low)
            .field("high", &self.high)
            .finish()
    }
}

impl Clone for Bound {
    fn clone(&self) -> Self {
        Self {
            distribution: self.distribution.clone_box(),
            low: self.low,
            high: self.high,
        }
    }
}

impl Bound {
    /// Create a bounded version of `distribution` over `[low, high]`.
    ///
    /// # Errors
    ///
    /// Returns an error if `low >= high`.
    pub fn new(distribution: Box<dyn Distribution>, low: f64, high: f64) -> Result<Self> {
        if low >= high {
            return Err(Error::InvalidParameter(format!(
                "Bound: low ({}) must be less than high ({})",
                low, high
            )));
        }
        Ok(Self {
            distribution,
            low,
            high,
        })
    }

    /// Get the lower bound.
    pub fn low(&self) -> f64 {
        self.low
    }

    /// Get the upper bound.
    pub fn high(&self) -> f64 {
        self.high
    }

    /// Get a reference to the inner distribution.
    pub fn inner(&self) -> &dyn Distribution {
        self.distribution.as_ref()
    }
}

#[typetag::serde]
impl Distribution for Bound {
    fn sample_dyn(&self, rng: &mut dyn rand::RngCore) -> f64 {
        // Rejection sampling: keep sampling until we get a value in bounds.
        loop {
            let x = self.distribution.sample_dyn(rng);
            if x >= self.low && x <= self.high {
                return x;
            }
        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.low || x > self.high {
            f64::NEG_INFINITY
        } else {
            // Return unnormalised PDF - normalisation constant doesn't affect MCMC
            self.distribution.ln_pdf(x)
        }
    }

    fn bounds(&self) -> Option<(f64, f64)> {
        Some((self.low, self.high))
    }

    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }
}

impl Bound {
    /// Sample a value from the distribution.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        loop {
            let x = self.distribution.sample_dyn(rng);
            if x >= self.low && x <= self.high {
                return x;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_uniform_sampling() {
        let dist = Uniform::new(0.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let x = dist.sample(&mut rng);
            assert!((0.0..=1.0).contains(&x));
        }
    }

    #[test]
    fn test_uniform_pdf() {
        let dist = Uniform::new(0.0, 2.0).unwrap();

        // Inside bounds
        assert_abs_diff_eq!(dist.ln_pdf(0.5), -(2.0_f64.ln()), epsilon = 1e-10);

        // Outside bounds
        assert_eq!(dist.ln_pdf(-0.1), f64::NEG_INFINITY);
        assert_eq!(dist.ln_pdf(2.1), f64::NEG_INFINITY);
    }

    #[test]
    fn test_uniform_bounds() {
        let dist = Uniform::new(-1.0, 3.0).unwrap();
        assert_eq!(dist.bounds(), Some((-1.0, 3.0)));
    }

    #[test]
    fn test_uniform_validation() {
        assert!(Uniform::new(1.0, 1.0).is_err());
        assert!(Uniform::new(2.0, 1.0).is_err());
    }

    #[test]
    fn test_normal_sampling() {
        let dist = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();

        let mut sum = 0.0;
        let n = 10000;
        for _ in 0..n {
            sum += dist.sample(&mut rng);
        }
        let mean = sum / n as f64;

        // Mean should be close to 0 with this many samples
        assert_abs_diff_eq!(mean, 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_normal_pdf() {
        let dist = Normal::new(0.0, 1.0).unwrap();

        // At mean
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert_abs_diff_eq!(dist.ln_pdf(0.0), expected, epsilon = 1e-10);

        // One sigma away
        let expected = -0.5 - 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert_abs_diff_eq!(dist.ln_pdf(1.0), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_normal_validation() {
        assert!(Normal::new(0.0, 0.0).is_err());
        assert!(Normal::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_lognormal_sampling() {
        let dist = LogNormal::new(0.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let x = dist.sample(&mut rng);
            assert!(x > 0.0);
        }
    }

    #[test]
    fn test_lognormal_pdf() {
        let dist = LogNormal::new(0.0, 1.0).unwrap();

        // Positive values
        assert!(dist.ln_pdf(1.0).is_finite());

        // Non-positive values
        assert_eq!(dist.ln_pdf(0.0), f64::NEG_INFINITY);
        assert_eq!(dist.ln_pdf(-1.0), f64::NEG_INFINITY);
    }

    #[test]
    fn test_lognormal_from_mean_std() {
        let mean = 1.0;
        let std_dev = 0.5;
        let dist = LogNormal::from_mean_std(mean, std_dev).unwrap();

        // Check that samples are positive
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let x = dist.sample(&mut rng);
            assert!(x > 0.0);
        }
    }

    #[test]
    fn test_lognormal_validation() {
        assert!(LogNormal::new(0.0, 0.0).is_err());
        assert!(LogNormal::new(0.0, -1.0).is_err());
        assert!(LogNormal::from_mean_std(0.0, 1.0).is_err());
        assert!(LogNormal::from_mean_std(1.0, 0.0).is_err());
    }

    #[test]
    fn test_bound_sampling() {
        let dist = Bound::new(Box::new(Normal::new(0.0, 1.0).unwrap()), -2.0, 2.0).unwrap();
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let x = dist.sample(&mut rng);
            assert!((-2.0..=2.0).contains(&x));
        }
    }

    #[test]
    fn test_bound_pdf() {
        let dist = Bound::new(Box::new(Normal::new(0.0, 1.0).unwrap()), -1.0, 1.0).unwrap();

        // Inside bounds - should match underlying distribution
        let inner_pdf = dist.inner().ln_pdf(0.5);
        assert_abs_diff_eq!(dist.ln_pdf(0.5), inner_pdf, epsilon = 1e-10);

        // Outside bounds
        assert_eq!(dist.ln_pdf(-1.5), f64::NEG_INFINITY);
        assert_eq!(dist.ln_pdf(1.5), f64::NEG_INFINITY);
    }

    #[test]
    fn test_bound_bounds() {
        let dist = Bound::new(Box::new(Normal::new(0.0, 1.0).unwrap()), -5.0, 5.0).unwrap();
        assert_eq!(dist.bounds(), Some((-5.0, 5.0)));
    }

    #[test]
    fn test_bound_validation() {
        assert!(Bound::new(Box::new(Normal::new(0.0, 1.0).unwrap()), 1.0, 1.0).is_err());
        assert!(Bound::new(Box::new(Normal::new(0.0, 1.0).unwrap()), 2.0, 1.0).is_err());
    }
}
