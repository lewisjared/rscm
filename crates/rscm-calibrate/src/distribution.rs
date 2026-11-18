//! Prior distributions for model parameters.
//!
//! Distributions provide:
//! - Random sampling for initialisation and prior generation
//! - Log probability density for likelihood evaluation
//! - Bounds for constrained optimisation

use rand::RngExt;
use serde::{Deserialize, Serialize};
use statrs::distribution::ContinuousCDF;

use crate::{Error, Result};

/// A probability distribution for model parameters.
///
/// This trait provides the interface for parameter priors in Bayesian calibration.
/// All distributions are object-safe and support serialisation via `typetag::serde`.
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
/// let log_p = uniform.log_pdf(0.5);
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
    fn sample_dyn(&self, rng: &mut dyn rand::Rng) -> f64;

    /// Compute the natural logarithm of the probability density at `x`.
    ///
    /// Returns `f64::NEG_INFINITY` for values outside the support.
    ///
    /// # Note on Normalisation
    ///
    /// For bounded distributions, the returned value may be unnormalised
    /// (missing the normalisation constant). This is acceptable for MCMC
    /// where the constant cancels in acceptance ratios.
    fn log_pdf(&self, x: f64) -> f64;

    /// Get the support bounds `[min, max]` of the distribution.
    ///
    /// Returns `Some((min, max))` for bounded distributions, or `None`
    /// for distributions with unbounded support (e.g., Normal).
    ///
    /// These bounds are used by optimisers to constrain parameter searches.
    fn bounds(&self) -> Option<(f64, f64)>;

    /// Cumulative distribution function at `x`, if it has a closed form.
    ///
    /// [`Bound`] uses it to reject bounds without probability mass and to sample by inverse CDF.
    fn cdf(&self, _x: f64) -> Option<f64> {
        None
    }

    /// Inverse of [`cdf`](Self::cdf) at probability `p`, if it has a closed form.
    fn inverse_cdf(&self, _p: f64) -> Option<f64> {
        None
    }

    /// Clone the distribution into a boxed trait object.
    ///
    /// Required for cloning `Box<dyn Distribution>`.
    fn clone_box(&self) -> Box<dyn Distribution>;
}

// Helper extension trait for concrete types with a generic sample method
impl dyn Distribution {
    /// Sample a value from the distribution using the thread-local RNG.
    pub fn sample(&self) -> f64 {
        let mut rng = rand::rng();
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
/// let log_p = dist.log_pdf(5.0);
/// assert_eq!(log_p, -(10.0_f64.ln()));
///
/// // Zero probability outside bounds
/// assert_eq!(dist.log_pdf(-1.0), f64::NEG_INFINITY);
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
    /// Returns an error unless `low` and `high` are finite and `low < high`.
    pub fn new(low: f64, high: f64) -> Result<Self> {
        if !(low.is_finite() && high.is_finite()) {
            return Err(Error::InvalidParameter(format!(
                "Uniform: low ({low}) and high ({high}) must be finite"
            )));
        }
        if low >= high {
            return Err(Error::InvalidParameter(format!(
                "Uniform: low ({low}) must be less than high ({high})"
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
    fn sample_dyn(&self, rng: &mut dyn rand::Rng) -> f64 {
        self.low + rng.random::<f64>() * (self.high - self.low)
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if (self.low..=self.high).contains(&x) {
            -(self.high - self.low).ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    fn bounds(&self) -> Option<(f64, f64)> {
        Some((self.low, self.high))
    }

    fn cdf(&self, x: f64) -> Option<f64> {
        Some(((x - self.low) / (self.high - self.low)).clamp(0.0, 1.0))
    }

    fn inverse_cdf(&self, p: f64) -> Option<f64> {
        Some(self.low + p * (self.high - self.low))
    }

    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }
}

impl Uniform {
    /// Sample a value from the distribution.
    pub fn sample<R: RngExt + ?Sized>(&self, rng: &mut R) -> f64 {
        self.low + rng.random::<f64>() * (self.high - self.low)
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
/// let log_p_mean = dist.log_pdf(0.0);
/// let log_p_away = dist.log_pdf(2.0);
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
    /// * `mean` - Mean (centre) of the distribution
    /// * `std_dev` - Standard deviation (must be positive)
    ///
    /// # Errors
    ///
    /// Returns an error unless `mean` is finite and `std_dev` is finite and positive.
    pub fn new(mean: f64, std_dev: f64) -> Result<Self> {
        if !mean.is_finite() {
            return Err(Error::InvalidParameter(format!(
                "Normal: mean ({mean}) must be finite"
            )));
        }
        if !(std_dev.is_finite() && std_dev > 0.0) {
            return Err(Error::InvalidParameter(format!(
                "Normal: std_dev ({std_dev}) must be finite and positive"
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

    fn statrs(&self) -> statrs::distribution::Normal {
        statrs::distribution::Normal::new(self.mean, self.std_dev)
            .expect("Normal parameters validated at construction")
    }
}

#[typetag::serde]
impl Distribution for Normal {
    fn sample_dyn(&self, rng: &mut dyn rand::Rng) -> f64 {
        let dist = rand_distr::Normal::new(self.mean, self.std_dev)
            .expect("Normal parameters validated at construction");
        rng.sample(dist)
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NEG_INFINITY;
        }
        let z = (x - self.mean) / self.std_dev;
        -0.5 * z * z - self.std_dev.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    fn bounds(&self) -> Option<(f64, f64)> {
        None
    }

    fn cdf(&self, x: f64) -> Option<f64> {
        Some(self.statrs().cdf(x))
    }

    fn inverse_cdf(&self, p: f64) -> Option<f64> {
        Some(self.statrs().inverse_cdf(p))
    }

    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }
}

impl Normal {
    /// Sample a value from the distribution.
    pub fn sample<R: RngExt + ?Sized>(&self, rng: &mut R) -> f64 {
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
    /// Returns an error unless `mu` is finite and `sigma` is finite and positive.
    pub fn new(mu: f64, sigma: f64) -> Result<Self> {
        if !mu.is_finite() {
            return Err(Error::InvalidParameter(format!(
                "LogNormal: mu ({mu}) must be finite"
            )));
        }
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(Error::InvalidParameter(format!(
                "LogNormal: sigma ({sigma}) must be finite and positive"
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
    /// Returns an error unless `mean` and `std_dev` are finite and positive
    /// and give a finite μ and a finite, positive σ.
    pub fn from_mean_std_dev(mean: f64, std_dev: f64) -> Result<Self> {
        if !(mean.is_finite() && mean > 0.0) {
            return Err(Error::InvalidParameter(format!(
                "LogNormal: mean ({mean}) must be finite and positive"
            )));
        }
        if !(std_dev.is_finite() && std_dev > 0.0) {
            return Err(Error::InvalidParameter(format!(
                "LogNormal: std_dev ({std_dev}) must be finite and positive"
            )));
        }

        let ratio = std_dev / mean;
        let sigma_sq = (ratio * ratio).ln_1p();
        let mu = mean.ln() - 0.5 * sigma_sq;
        let sigma = sigma_sq.sqrt();
        if !(mu.is_finite() && sigma.is_finite() && sigma > 0.0) {
            return Err(Error::InvalidParameter(format!(
                "LogNormal: mean ({mean}) and std_dev ({std_dev}) give mu ({mu}) and sigma ({sigma}), \
                 which must be finite with sigma positive"
            )));
        }
        Ok(Self { mu, sigma })
    }

    /// Get the μ parameter (mean of ln(X)).
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Get the σ parameter (standard deviation of ln(X)).
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    fn statrs(&self) -> statrs::distribution::LogNormal {
        statrs::distribution::LogNormal::new(self.mu, self.sigma)
            .expect("LogNormal parameters validated at construction")
    }
}

#[typetag::serde]
impl Distribution for LogNormal {
    fn sample_dyn(&self, rng: &mut dyn rand::Rng) -> f64 {
        let dist = rand_distr::LogNormal::new(self.mu, self.sigma)
            .expect("LogNormal parameters validated at construction");
        rng.sample(dist)
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if x.is_nan() || x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let log_x = x.ln();
        let z = (log_x - self.mu) / self.sigma;
        -0.5 * z * z - log_x - self.sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    fn bounds(&self) -> Option<(f64, f64)> {
        Some((0.0, f64::INFINITY))
    }

    fn cdf(&self, x: f64) -> Option<f64> {
        Some(self.statrs().cdf(x))
    }

    fn inverse_cdf(&self, p: f64) -> Option<f64> {
        Some(self.statrs().inverse_cdf(p))
    }

    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }
}

impl LogNormal {
    /// Sample a value from the distribution.
    pub fn sample<R: RngExt + ?Sized>(&self, rng: &mut R) -> f64 {
        let dist = rand_distr::LogNormal::new(self.mu, self.sigma)
            .expect("LogNormal parameters validated at construction");
        rng.sample(dist)
    }
}

/// Wrapper that truncates any distribution to `[low, high]`.
///
/// This allows constraining unbounded distributions (e.g., Normal) to an interval.
///
/// # Algorithm
///
/// - **Sampling**: inverse CDF over the inner probability mass in `[low, high]`
///   when the inner distribution has a closed-form CDF, rejection sampling otherwise
/// - **Density**: Returns inner distribution's log-PDF for values in bounds, -∞ otherwise
///
/// Construction fails when `[low, high]` holds no probability mass of the inner distribution
/// to double precision, such as Normal(0, 1) bounded to [10, 20].
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
/// assert_eq!(bounded.log_pdf(-3.0), f64::NEG_INFINITY);
///
/// // Bounds are available for optimisers
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
    /// Returns an error unless `low < high` (either may be infinite)
    /// and `[low, high]` holds probability mass of `distribution`.
    pub fn new(distribution: Box<dyn Distribution>, low: f64, high: f64) -> Result<Self> {
        if low.is_nan() || high.is_nan() || low >= high {
            return Err(Error::InvalidParameter(format!(
                "Bound: low ({low}) must be less than high ({high})"
            )));
        }
        if let Some((support_low, support_high)) = distribution.bounds() {
            if high < support_low || low > support_high {
                return Err(Error::InvalidParameter(format!(
                    "Bound: [{low}, {high}] does not overlap the support \
                     [{support_low}, {support_high}] of the inner distribution"
                )));
            }
        }
        if let (Some(cdf_low), Some(cdf_high)) = (distribution.cdf(low), distribution.cdf(high)) {
            if cdf_high.is_nan() || cdf_low.is_nan() || cdf_high <= cdf_low {
                return Err(Error::InvalidParameter(format!(
                    "Bound: [{low}, {high}] holds no probability mass of the inner distribution"
                )));
            }
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

/// Rejection sampling gives up after this many draws, for inner distributions without a CDF.
const MAX_REJECTION_DRAWS: usize = 1_000_000;

impl Bound {
    /// Inner CDF at `low` and `high`, when the inner distribution has a closed-form CDF.
    fn cdf_range(&self) -> Option<(f64, f64)> {
        Some((
            self.distribution.cdf(self.low)?,
            self.distribution.cdf(self.high)?,
        ))
    }

    fn sample_impl(&self, rng: &mut dyn rand::Rng) -> f64 {
        if let Some((cdf_low, cdf_high)) = self.cdf_range() {
            let p = cdf_low + rng.random::<f64>() * (cdf_high - cdf_low);
            if let Some(x) = self.distribution.inverse_cdf(p) {
                return x.clamp(self.low, self.high);
            }
        }
        for _ in 0..MAX_REJECTION_DRAWS {
            let x = self.distribution.sample_dyn(rng);
            if (self.low..=self.high).contains(&x) {
                return x;
            }
        }
        panic!(
            "Bound: no sample fell in [{}, {}] after {MAX_REJECTION_DRAWS} draws",
            self.low, self.high
        )
    }
}

#[typetag::serde]
impl Distribution for Bound {
    fn sample_dyn(&self, rng: &mut dyn rand::Rng) -> f64 {
        self.sample_impl(rng)
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if (self.low..=self.high).contains(&x) {
            // Unnormalised: the normalisation constant cancels in MCMC.
            self.distribution.log_pdf(x)
        } else {
            f64::NEG_INFINITY
        }
    }

    fn bounds(&self) -> Option<(f64, f64)> {
        Some((self.low, self.high))
    }

    fn cdf(&self, x: f64) -> Option<f64> {
        let (cdf_low, cdf_high) = self.cdf_range()?;
        let inner = self.distribution.cdf(x.clamp(self.low, self.high))?;
        Some((inner - cdf_low) / (cdf_high - cdf_low))
    }

    fn inverse_cdf(&self, p: f64) -> Option<f64> {
        let (cdf_low, cdf_high) = self.cdf_range()?;
        let x = self
            .distribution
            .inverse_cdf(cdf_low + p * (cdf_high - cdf_low))?;
        Some(x.clamp(self.low, self.high))
    }

    fn clone_box(&self) -> Box<dyn Distribution> {
        Box::new(self.clone())
    }
}

impl Bound {
    /// Sample a value from the distribution.
    pub fn sample<R: RngExt>(&self, rng: &mut R) -> f64 {
        self.sample_impl(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_uniform_sampling() {
        let dist = Uniform::new(0.0, 1.0).unwrap();
        let mut rng = rand::rng();

        for _ in 0..100 {
            let x = dist.sample(&mut rng);
            assert!((0.0..=1.0).contains(&x));
        }
    }

    #[test]
    fn test_uniform_pdf() {
        let dist = Uniform::new(0.0, 2.0).unwrap();

        // Inside bounds
        assert_abs_diff_eq!(dist.log_pdf(0.5), -(2.0_f64.ln()), epsilon = 1e-10);

        // Outside bounds
        assert_eq!(dist.log_pdf(-0.1), f64::NEG_INFINITY);
        assert_eq!(dist.log_pdf(2.1), f64::NEG_INFINITY);
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
        let mut rng = rand::rng();

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
        assert_abs_diff_eq!(dist.log_pdf(0.0), expected, epsilon = 1e-10);

        // One sigma away
        let expected = -0.5 - 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert_abs_diff_eq!(dist.log_pdf(1.0), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_normal_validation() {
        assert!(Normal::new(0.0, 0.0).is_err());
        assert!(Normal::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_lognormal_sampling() {
        let dist = LogNormal::new(0.0, 1.0).unwrap();
        let mut rng = rand::rng();

        for _ in 0..100 {
            let x = dist.sample(&mut rng);
            assert!(x > 0.0);
        }
    }

    #[test]
    fn test_lognormal_pdf() {
        let dist = LogNormal::new(0.0, 1.0).unwrap();

        // Positive values
        assert!(dist.log_pdf(1.0).is_finite());

        // Non-positive values
        assert_eq!(dist.log_pdf(0.0), f64::NEG_INFINITY);
        assert_eq!(dist.log_pdf(-1.0), f64::NEG_INFINITY);
    }

    #[test]
    fn test_lognormal_from_mean_std_dev() {
        let mean = 1.0;
        let std_dev = 0.5;
        let dist = LogNormal::from_mean_std_dev(mean, std_dev).unwrap();

        // Check that samples are positive
        let mut rng = rand::rng();
        for _ in 0..100 {
            let x = dist.sample(&mut rng);
            assert!(x > 0.0);
        }
    }

    #[test]
    fn test_lognormal_validation() {
        assert!(LogNormal::new(0.0, 0.0).is_err());
        assert!(LogNormal::new(0.0, -1.0).is_err());
        assert!(LogNormal::from_mean_std_dev(0.0, 1.0).is_err());
        assert!(LogNormal::from_mean_std_dev(1.0, 0.0).is_err());
    }

    #[test]
    fn test_bound_sampling() {
        let dist = Bound::new(Box::new(Normal::new(0.0, 1.0).unwrap()), -2.0, 2.0).unwrap();
        let mut rng = rand::rng();

        for _ in 0..100 {
            let x = dist.sample(&mut rng);
            assert!((-2.0..=2.0).contains(&x));
        }
    }

    #[test]
    fn test_bound_pdf() {
        let dist = Bound::new(Box::new(Normal::new(0.0, 1.0).unwrap()), -1.0, 1.0).unwrap();

        // Inside bounds - should match underlying distribution
        let inner_pdf = dist.inner().log_pdf(0.5);
        assert_abs_diff_eq!(dist.log_pdf(0.5), inner_pdf, epsilon = 1e-10);

        // Outside bounds
        assert_eq!(dist.log_pdf(-1.5), f64::NEG_INFINITY);
        assert_eq!(dist.log_pdf(1.5), f64::NEG_INFINITY);
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
        assert!(Bound::new(Box::new(Normal::new(0.0, 1.0).unwrap()), f64::NAN, 1.0).is_err());
    }

    #[test]
    fn test_bound_without_mass_is_rejected() {
        let normal = Bound::new(Box::new(Normal::new(0.0, 1.0).unwrap()), 100.0, 101.0);
        assert!(normal
            .unwrap_err()
            .to_string()
            .contains("no probability mass"));
        let uniform = Bound::new(Box::new(Uniform::new(0.0, 1.0).unwrap()), 5.0, 6.0);
        assert!(uniform
            .unwrap_err()
            .to_string()
            .contains("does not overlap"));
    }

    #[test]
    fn test_bound_far_tail_sampling() {
        let dist = Bound::new(Box::new(Normal::new(0.0, 1.0).unwrap()), 7.0, 9.0).unwrap();
        let mut rng = rand::rng();
        for _ in 0..100 {
            assert!((7.0..=9.0).contains(&dist.sample(&mut rng)));
        }
    }

    #[test]
    fn test_nested_bound_cdf() {
        let inner = Bound::new(Box::new(Uniform::new(0.0, 4.0).unwrap()), 1.0, 3.0).unwrap();
        let outer = Bound::new(Box::new(inner), 2.0, 5.0).unwrap();
        assert_abs_diff_eq!(outer.cdf(2.5).unwrap(), 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(outer.inverse_cdf(0.5).unwrap(), 2.5, epsilon = 1e-12);
    }

    #[test]
    fn test_non_finite_parameters_rejected() {
        assert!(Uniform::new(f64::NAN, 1.0).is_err());
        assert!(Uniform::new(0.0, f64::INFINITY).is_err());
        assert!(Normal::new(f64::NAN, 1.0).is_err());
        assert!(Normal::new(0.0, f64::NAN).is_err());
        assert!(LogNormal::new(0.0, f64::NAN).is_err());
        assert!(LogNormal::from_mean_std_dev(f64::NAN, 1.0).is_err());
    }

    #[test]
    fn test_lognormal_from_mean_std_dev_degenerate() {
        let tiny = LogNormal::from_mean_std_dev(1.0, 1e-10).unwrap();
        assert!(tiny.sigma() > 0.0);
        assert!(Bound::new(Box::new(tiny), 0.5, 1.5).is_ok());
        assert!(LogNormal::from_mean_std_dev(1.0, 1e-200).is_err());
        assert!(LogNormal::from_mean_std_dev(1.0, 1e200).is_err());
        assert!(LogNormal::from_mean_std_dev(1e300, 1e300).is_ok());
    }

    #[test]
    fn test_log_pdf_nan() {
        let uniform = Uniform::new(0.0, 1.0).unwrap();
        assert_eq!(uniform.log_pdf(f64::NAN), f64::NEG_INFINITY);
        assert_eq!(
            Normal::new(0.0, 1.0).unwrap().log_pdf(f64::NAN),
            f64::NEG_INFINITY
        );
        assert_eq!(
            LogNormal::new(0.0, 1.0).unwrap().log_pdf(f64::NAN),
            f64::NEG_INFINITY
        );
        let bound = Bound::new(Box::new(uniform), 0.1, 0.9).unwrap();
        assert_eq!(bound.log_pdf(f64::NAN), f64::NEG_INFINITY);
    }
}
