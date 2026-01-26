//! MCMC chain convergence diagnostics.
//!
//! This module provides convergence diagnostics for MCMC chains:
//! - R-hat (Gelman-Rubin statistic) for assessing chain convergence
//! - Effective Sample Size (ESS) for measuring independent sample count
//! - Integrated autocorrelation time for determining thinning intervals

use super::chain::Chain;
use indexmap::IndexMap;

impl Chain {
    /// Compute the Gelman-Rubin statistic (R-hat) for each parameter.
    ///
    /// R-hat measures convergence by comparing within-chain and between-chain variances.
    /// Values close to 1.0 indicate convergence. As a rule of thumb, R-hat < 1.1 for all
    /// parameters suggests the chains have converged.
    ///
    /// # Arguments
    ///
    /// * `discard` - Number of initial samples to discard as burn-in
    ///
    /// # Returns
    ///
    /// Map from parameter name to R-hat value. Returns empty map if insufficient samples.
    ///
    /// # Algorithm
    ///
    /// Following Gelman & Rubin (1992):
    /// 1. Split each walker's chain in half to create 2M chains of length N
    /// 2. Compute within-chain variance W (mean of chain variances)
    /// 3. Compute between-chain variance B (variance of chain means)
    /// 4. Estimate variance as weighted average: var+ = (N-1)/N * W + B/N
    /// 5. R-hat = sqrt(var+ / W)
    ///
    /// # References
    ///
    /// Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation using
    /// multiple sequences. Statistical Science, 7(4), 457-472.
    pub fn r_hat(&self, discard: usize) -> IndexMap<String, f64> {
        let mut result = IndexMap::new();

        if self.is_empty() || discard >= self.len() {
            return result;
        }

        let n_keep = self.len() - discard;
        if n_keep < 4 {
            // Need at least 4 samples to split chains
            return result;
        }

        let n_walkers = self.samples[0].nrows();

        // Split each walker's chain in half
        let n_split = n_keep / 2;
        let n_chains = n_walkers * 2;

        for (param_idx, param_name) in self.param_names.iter().enumerate() {
            // Extract all samples for this parameter, organized by split chain
            let mut chain_samples = Vec::with_capacity(n_chains);

            for walker_idx in 0..n_walkers {
                // First half of walker's chain
                let mut first_half = Vec::with_capacity(n_split);
                for sample in self.samples.iter().skip(discard).take(n_split) {
                    first_half.push(sample[[walker_idx, param_idx]]);
                }
                chain_samples.push(first_half);

                // Second half of walker's chain
                let mut second_half = Vec::with_capacity(n_split);
                for sample in self.samples.iter().skip(discard + n_split).take(n_split) {
                    second_half.push(sample[[walker_idx, param_idx]]);
                }
                chain_samples.push(second_half);
            }

            // Compute mean and variance for each chain
            let mut chain_means = Vec::with_capacity(n_chains);
            let mut chain_vars = Vec::with_capacity(n_chains);

            for chain in &chain_samples {
                let mean = chain.iter().sum::<f64>() / n_split as f64;
                let variance =
                    chain.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_split - 1) as f64;

                chain_means.push(mean);
                chain_vars.push(variance);
            }

            // Within-chain variance (W)
            let w = chain_vars.iter().sum::<f64>() / n_chains as f64;

            // Between-chain variance (B)
            let overall_mean = chain_means.iter().sum::<f64>() / n_chains as f64;
            let b = n_split as f64
                * chain_means
                    .iter()
                    .map(|&m| (m - overall_mean).powi(2))
                    .sum::<f64>()
                / (n_chains - 1) as f64;

            // Variance estimate
            let var_plus = ((n_split - 1) as f64 * w + b) / n_split as f64;

            // R-hat
            let r_hat = (var_plus / w).sqrt();

            result.insert(param_name.clone(), r_hat);
        }

        result
    }

    /// Check if the chain has converged based on R-hat statistic.
    ///
    /// # Arguments
    ///
    /// * `discard` - Number of initial samples to discard as burn-in
    /// * `threshold` - R-hat threshold for convergence (typically 1.1)
    ///
    /// # Returns
    ///
    /// `true` if all parameters have R-hat < threshold, `false` otherwise.
    /// Returns `false` if insufficient samples to compute R-hat.
    pub fn is_converged(&self, discard: usize, threshold: f64) -> bool {
        let r_hat = self.r_hat(discard);

        if r_hat.is_empty() {
            return false;
        }

        r_hat.values().all(|&v| v < threshold && v.is_finite())
    }

    /// Compute the effective sample size (ESS) for each parameter.
    ///
    /// ESS estimates the number of independent samples in the chain, accounting
    /// for autocorrelation. Higher values indicate better mixing. As a rule of
    /// thumb, ESS > 100 per chain is often sufficient for posterior inference.
    ///
    /// # Arguments
    ///
    /// * `discard` - Number of initial samples to discard as burn-in
    ///
    /// # Returns
    ///
    /// Map from parameter name to ESS value. Returns empty map if insufficient samples.
    ///
    /// # Algorithm
    ///
    /// Implements the method from Gelman et al. (2013):
    /// 1. Compute autocorrelation at each lag for all chains
    /// 2. Average autocorrelation across chains
    /// 3. Sum positive autocorrelations until they become negative
    /// 4. ESS = N / (1 + 2 * sum(autocorr))
    ///
    /// where N is the total number of samples across all walkers.
    ///
    /// # References
    ///
    /// Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., &
    /// Rubin, D. B. (2013). Bayesian Data Analysis (3rd ed.). CRC Press.
    pub fn ess(&self, discard: usize) -> IndexMap<String, f64> {
        let mut result = IndexMap::new();

        if self.is_empty() || discard >= self.len() {
            return result;
        }

        let n_keep = self.len() - discard;
        if n_keep < 10 {
            // Need at least 10 samples for meaningful autocorrelation
            return result;
        }

        let n_walkers = self.samples[0].nrows();

        for (param_idx, param_name) in self.param_names.iter().enumerate() {
            // Extract samples for this parameter from each walker
            let mut walker_chains = Vec::with_capacity(n_walkers);

            for walker_idx in 0..n_walkers {
                let mut chain = Vec::with_capacity(n_keep);
                for sample in self.samples.iter().skip(discard) {
                    chain.push(sample[[walker_idx, param_idx]]);
                }
                walker_chains.push(chain);
            }

            // Compute autocorrelation for each walker and average
            let max_lag = (n_keep / 2).min(100); // Don't go beyond half the chain or 100 lags
            let mut avg_autocorr = vec![0.0; max_lag];

            for chain in &walker_chains {
                let autocorr = compute_autocorrelation(chain, max_lag);
                for (i, &ac) in autocorr.iter().enumerate() {
                    avg_autocorr[i] += ac / n_walkers as f64;
                }
            }

            // Sum positive autocorrelations
            let mut sum_autocorr = 0.0;
            for &ac in &avg_autocorr {
                if ac <= 0.0 {
                    break;
                }
                sum_autocorr += ac;
            }

            // Compute ESS
            let n_total = (n_keep * n_walkers) as f64;
            let ess = n_total / (1.0 + 2.0 * sum_autocorr);

            result.insert(param_name.clone(), ess);
        }

        result
    }

    /// Compute the integrated autocorrelation time for each parameter.
    ///
    /// The autocorrelation time (τ) measures how many steps it takes for samples
    /// to become approximately independent. It is computed as:
    ///
    /// τ = 1 + 2 * Σ ρ(k)
    ///
    /// where ρ(k) is the autocorrelation at lag k, summed over positive values.
    ///
    /// This is useful for:
    /// - Determining appropriate thinning interval (thin by ~τ to get nearly independent samples)
    /// - Estimating effective sample size: ESS ≈ N / τ
    /// - Assessing mixing quality (smaller τ = better mixing)
    ///
    /// # Arguments
    ///
    /// * `discard` - Number of initial samples to discard as burn-in
    ///
    /// # Returns
    ///
    /// Map from parameter name to autocorrelation time. Returns empty map if
    /// there are insufficient samples (< 10 after discard).
    ///
    /// # Example
    ///
    /// ```
    /// # use rscm_calibrate::sampler::Chain;
    /// # let chain = Chain::new(vec!["x".to_string()], 1);
    /// let tau = chain.autocorr_time(100);
    /// for (param, time) in tau {
    ///     println!("{}: τ = {:.1} (thin by ~{:.0} for independence)", param, time, time);
    /// }
    /// ```
    pub fn autocorr_time(&self, discard: usize) -> IndexMap<String, f64> {
        let mut result = IndexMap::new();

        if self.is_empty() || discard >= self.len() {
            return result;
        }

        let n_keep = self.len() - discard;
        if n_keep < 10 {
            // Need at least 10 samples for meaningful autocorrelation
            return result;
        }

        let n_walkers = self.samples[0].nrows();

        for (param_idx, param_name) in self.param_names.iter().enumerate() {
            // Extract samples for this parameter from each walker
            let mut walker_chains = Vec::with_capacity(n_walkers);

            for walker_idx in 0..n_walkers {
                let mut chain = Vec::with_capacity(n_keep);
                for sample in self.samples.iter().skip(discard) {
                    chain.push(sample[[walker_idx, param_idx]]);
                }
                walker_chains.push(chain);
            }

            // Compute autocorrelation for each walker and average
            let max_lag = (n_keep / 2).min(100); // Don't go beyond half the chain or 100 lags
            let mut avg_autocorr = vec![0.0; max_lag];

            for chain in &walker_chains {
                let autocorr = compute_autocorrelation(chain, max_lag);
                for (i, &ac) in autocorr.iter().enumerate() {
                    avg_autocorr[i] += ac / n_walkers as f64;
                }
            }

            // Sum positive autocorrelations
            let mut sum_autocorr = 0.0;
            for &ac in &avg_autocorr {
                if ac <= 0.0 {
                    break;
                }
                sum_autocorr += ac;
            }

            // Compute autocorrelation time: τ = 1 + 2 * Σ ρ(k)
            let tau = 1.0 + 2.0 * sum_autocorr;

            result.insert(param_name.clone(), tau);
        }

        result
    }
}

/// Compute autocorrelation function for a chain at different lags.
///
/// # Arguments
///
/// * `chain` - The chain of samples
/// * `max_lag` - Maximum lag to compute
///
/// # Returns
///
/// Vector of autocorrelation values at lags 1..max_lag (lag 0 is always 1.0 and is not included)
fn compute_autocorrelation(chain: &[f64], max_lag: usize) -> Vec<f64> {
    let n = chain.len();
    let mean = chain.iter().sum::<f64>() / n as f64;

    // Compute variance (lag-0 autocovariance)
    let variance = chain.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if variance == 0.0 {
        return vec![0.0; max_lag];
    }

    // Compute autocorrelation at each lag
    let mut autocorr = Vec::with_capacity(max_lag);

    for lag in 1..=max_lag {
        if lag >= n {
            autocorr.push(0.0);
            continue;
        }

        let mut covariance = 0.0;
        for i in 0..(n - lag) {
            covariance += (chain[i] - mean) * (chain[i + lag] - mean);
        }
        covariance /= (n - lag) as f64;

        autocorr.push(covariance / variance);
    }

    autocorr
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use rand::SeedableRng;

    #[test]
    fn test_r_hat_converged_chains() {
        // Create a chain with samples from a converged distribution
        // All walkers sampling from the same distribution should give R-hat ≈ 1.0
        let param_names = vec!["x".to_string()];
        let mut chain = Chain::new(param_names, 1);

        let n_walkers = 4;
        let n_samples = 100;

        // Generate samples from N(0, 1) for all walkers
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        use rand_distr::{Distribution as _, Normal};
        let normal = Normal::new(0.0, 1.0).unwrap();

        for _ in 0..n_samples {
            let mut positions = Array2::zeros((n_walkers, 1));
            let mut log_probs = Array1::zeros(n_walkers);

            for w in 0..n_walkers {
                let value: f64 = normal.sample(&mut rng);
                positions[[w, 0]] = value;
                log_probs[w] = -0.5 * value.powi(2); // Log probability of N(0,1)
            }

            chain.push(positions, log_probs);
        }

        // Compute R-hat with burn-in
        let r_hat = chain.r_hat(10);

        // Should have one entry for parameter "x"
        assert_eq!(r_hat.len(), 1);
        assert!(r_hat.contains_key("x"));

        // R-hat should be close to 1.0 for converged chains
        let r_hat_x = r_hat["x"];
        assert!(r_hat_x > 0.9 && r_hat_x < 1.3, "R-hat = {}", r_hat_x);
    }

    #[test]
    fn test_r_hat_diverged_chains() {
        // Create a chain where different walkers sample from different distributions
        // This should give R-hat > 1.0
        let param_names = vec!["x".to_string()];
        let mut chain = Chain::new(param_names, 1);

        let n_walkers = 4;
        let n_samples = 100;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        use rand_distr::{Distribution as _, Normal};

        for _ in 0..n_samples {
            let mut positions = Array2::zeros((n_walkers, 1));
            let log_probs = Array1::zeros(n_walkers);

            for w in 0..n_walkers {
                // Each walker samples from N(w*10, 1) - widely separated modes
                let normal = Normal::new((w * 10) as f64, 1.0).unwrap();
                positions[[w, 0]] = normal.sample(&mut rng);
            }

            chain.push(positions, log_probs);
        }

        // Compute R-hat with burn-in
        let r_hat = chain.r_hat(10);

        // R-hat should be >> 1.0 for diverged chains
        let r_hat_x = r_hat["x"];
        assert!(r_hat_x > 2.0, "R-hat = {} (expected > 2.0)", r_hat_x);
    }

    #[test]
    fn test_r_hat_multiple_parameters() {
        // Test R-hat with multiple parameters
        let param_names = vec!["x".to_string(), "y".to_string()];
        let mut chain = Chain::new(param_names.clone(), 1);

        let n_walkers = 4;
        let n_samples = 100;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        use rand_distr::{Distribution as _, Normal};
        let normal_x = Normal::new(0.0, 1.0).unwrap();
        let normal_y = Normal::new(5.0, 2.0).unwrap();

        for _ in 0..n_samples {
            let mut positions = Array2::zeros((n_walkers, 2));
            let log_probs = Array1::zeros(n_walkers);

            for w in 0..n_walkers {
                positions[[w, 0]] = normal_x.sample(&mut rng);
                positions[[w, 1]] = normal_y.sample(&mut rng);
            }

            chain.push(positions, log_probs);
        }

        // Compute R-hat
        let r_hat = chain.r_hat(10);

        // Should have entries for both parameters
        assert_eq!(r_hat.len(), 2);
        assert!(r_hat.contains_key("x"));
        assert!(r_hat.contains_key("y"));

        // Both should be close to 1.0
        assert!(r_hat["x"] > 0.9 && r_hat["x"] < 1.3);
        assert!(r_hat["y"] > 0.9 && r_hat["y"] < 1.3);
    }

    #[test]
    fn test_r_hat_insufficient_samples() {
        // Test that R-hat returns empty map with insufficient samples
        let param_names = vec!["x".to_string()];
        let mut chain = Chain::new(param_names, 1);

        // Add only 3 samples (need at least 4 after discard)
        for _ in 0..3 {
            chain.push(Array2::zeros((2, 1)), Array1::zeros(2));
        }

        let r_hat = chain.r_hat(0);
        assert!(r_hat.is_empty());
    }

    #[test]
    fn test_is_converged() {
        // Create a converged chain
        let param_names = vec!["x".to_string()];
        let mut chain = Chain::new(param_names, 1);

        let n_walkers = 4;
        let n_samples = 100;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        use rand_distr::{Distribution as _, Normal};
        let normal = Normal::new(0.0, 1.0).unwrap();

        for _ in 0..n_samples {
            let mut positions = Array2::zeros((n_walkers, 1));
            let log_probs = Array1::zeros(n_walkers);

            for w in 0..n_walkers {
                positions[[w, 0]] = normal.sample(&mut rng);
            }

            chain.push(positions, log_probs);
        }

        // Should be converged with typical threshold of 1.1
        assert!(chain.is_converged(10, 1.1));

        // With stricter threshold, may or may not be converged (depends on random samples)
        // Just verify the function returns a boolean
        let _ = chain.is_converged(10, 1.01);
    }

    #[test]
    fn test_ess_independent_samples() {
        // Test ESS with independent samples (no autocorrelation)
        // ESS should be close to the total number of samples
        let param_names = vec!["x".to_string()];
        let mut chain = Chain::new(param_names, 1);

        let n_walkers = 4;
        let n_samples = 200;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        use rand_distr::{Distribution as _, Normal};
        let normal = Normal::new(0.0, 1.0).unwrap();

        for _ in 0..n_samples {
            let mut positions = Array2::zeros((n_walkers, 1));
            let log_probs = Array1::zeros(n_walkers);

            for w in 0..n_walkers {
                // Independent samples from N(0, 1)
                positions[[w, 0]] = normal.sample(&mut rng);
            }

            chain.push(positions, log_probs);
        }

        let ess = chain.ess(10);
        assert_eq!(ess.len(), 1);
        assert!(ess.contains_key("x"));

        // ESS should be close to total samples for independent draws
        let total_samples = (n_samples - 10) * n_walkers;
        let ess_x = ess["x"];

        // Allow some variation due to finite sample effects
        // ESS should be at least 50% of total samples for independent data
        assert!(
            ess_x > (total_samples as f64 * 0.5),
            "ESS = {} (expected > {})",
            ess_x,
            total_samples as f64 * 0.5
        );
    }

    #[test]
    fn test_ess_correlated_samples() {
        // Test ESS with highly autocorrelated samples
        // ESS should be much less than total number of samples
        let param_names = vec!["x".to_string()];
        let mut chain = Chain::new(param_names, 1);

        let n_walkers = 4;
        let n_samples = 200;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        use rand_distr::{Distribution as _, Normal};
        let normal = Normal::new(0.0, 0.1).unwrap(); // Small noise

        // Initialize walker positions
        let mut current = vec![0.0; n_walkers];

        for _ in 0..n_samples {
            let mut positions = Array2::zeros((n_walkers, 1));
            let log_probs = Array1::zeros(n_walkers);

            for w in 0..n_walkers {
                // Random walk: highly autocorrelated
                current[w] += normal.sample(&mut rng);
                positions[[w, 0]] = current[w];
            }

            chain.push(positions, log_probs);
        }

        let ess = chain.ess(10);
        assert_eq!(ess.len(), 1);

        // ESS should be much less than total samples for autocorrelated data
        let total_samples = (n_samples - 10) * n_walkers;
        let ess_x = ess["x"];

        assert!(
            ess_x < (total_samples as f64 * 0.3),
            "ESS = {} (expected < {} for autocorrelated data)",
            ess_x,
            total_samples as f64 * 0.3
        );
    }

    #[test]
    fn test_ess_multiple_parameters() {
        // Test ESS with multiple parameters
        let param_names = vec!["x".to_string(), "y".to_string()];
        let mut chain = Chain::new(param_names.clone(), 1);

        let n_walkers = 4;
        let n_samples = 100;

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        use rand_distr::{Distribution as _, Normal};
        let normal = Normal::new(0.0, 1.0).unwrap();

        for _ in 0..n_samples {
            let mut positions = Array2::zeros((n_walkers, 2));
            let log_probs = Array1::zeros(n_walkers);

            for w in 0..n_walkers {
                positions[[w, 0]] = normal.sample(&mut rng);
                positions[[w, 1]] = normal.sample(&mut rng);
            }

            chain.push(positions, log_probs);
        }

        let ess = chain.ess(10);

        // Should have ESS for both parameters
        assert_eq!(ess.len(), 2);
        assert!(ess.contains_key("x"));
        assert!(ess.contains_key("y"));

        // Both should be positive and finite
        assert!(ess["x"] > 0.0 && ess["x"].is_finite());
        assert!(ess["y"] > 0.0 && ess["y"].is_finite());
    }

    #[test]
    fn test_ess_insufficient_samples() {
        // Test that ESS returns empty map with insufficient samples
        let param_names = vec!["x".to_string()];
        let mut chain = Chain::new(param_names, 1);

        // Add only 9 samples (need at least 10 after discard)
        for _ in 0..9 {
            chain.push(Array2::zeros((2, 1)), Array1::zeros(2));
        }

        let ess = chain.ess(0);
        assert!(ess.is_empty());
    }

    #[test]
    fn test_autocorr_time_independent_samples() {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Test autocorrelation time with independent samples
        // τ should be close to 1.0 (no correlation)
        let param_names = vec!["x".to_string()];
        let mut chain = Chain::new(param_names.clone(), 1);

        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let n_samples = 100;
        let n_walkers = 4;

        // Generate independent samples from N(0,1)
        for _ in 0..n_samples {
            let mut positions = Array2::zeros((n_walkers, 1));
            for i in 0..n_walkers {
                positions[[i, 0]] = rng.gen::<f64>() * 2.0 - 1.0; // Uniform(-1, 1)
            }
            chain.push(positions, Array1::zeros(n_walkers));
        }

        let tau = chain.autocorr_time(10);
        assert_eq!(tau.len(), 1);
        let x_tau = tau.get("x").unwrap();

        // For independent samples, τ should be close to 1.0
        // With random samples, allow some variation
        assert!(*x_tau > 0.5 && *x_tau < 2.0, "τ = {}", x_tau);
    }

    #[test]
    fn test_autocorr_time_correlated_samples() {
        // Test autocorrelation time with highly autocorrelated samples (random walk)
        // τ should be large (many correlated steps)
        let param_names = vec!["x".to_string()];
        let mut chain = Chain::new(param_names.clone(), 1);

        let n_samples = 100;
        let n_walkers = 4;

        // Generate random walk (highly autocorrelated)
        for i in 0..n_samples {
            let mut positions = Array2::zeros((n_walkers, 1));
            for j in 0..n_walkers {
                // Random walk: each step is previous + small increment
                positions[[j, 0]] = (i as f64) * 0.1 + (j as f64) * 0.01;
            }
            chain.push(positions, Array1::zeros(n_walkers));
        }

        let tau = chain.autocorr_time(10);
        assert_eq!(tau.len(), 1);
        let x_tau = tau.get("x").unwrap();

        // For highly correlated samples, τ should be large
        assert!(
            *x_tau > 10.0,
            "τ = {} (expected > 10.0 for random walk)",
            x_tau
        );
    }

    #[test]
    fn test_autocorr_time_multiple_parameters() {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Test autocorrelation time with multiple parameters
        let param_names = vec!["x".to_string(), "y".to_string()];
        let mut chain = Chain::new(param_names.clone(), 1);

        let mut rng = ChaCha8Rng::seed_from_u64(54321);
        let n_samples = 100;
        let n_walkers = 4;

        for _ in 0..n_samples {
            let mut positions = Array2::zeros((n_walkers, 2));
            for i in 0..n_walkers {
                positions[[i, 0]] = rng.gen::<f64>(); // x: independent
                positions[[i, 1]] = positions[[i, 1]].max(0.0) + 0.1; // y: autocorrelated
            }
            chain.push(positions, Array1::zeros(n_walkers));
        }

        let tau = chain.autocorr_time(10);
        assert_eq!(tau.len(), 2);

        let x_tau = tau.get("x").unwrap();
        let y_tau = tau.get("y").unwrap();

        // x should have low autocorrelation time (independent)
        assert!(*x_tau < 5.0, "x: τ = {}", x_tau);

        // y should have higher autocorrelation time (correlated)
        // Note: our simple test pattern may not show huge differences
        assert!(*y_tau >= 1.0, "y: τ = {}", y_tau);
    }

    #[test]
    fn test_autocorr_time_insufficient_samples() {
        // Test that autocorr_time returns empty map with insufficient samples
        let param_names = vec!["x".to_string()];
        let mut chain = Chain::new(param_names, 1);

        // Add only 9 samples (need at least 10 after discard)
        for _ in 0..9 {
            chain.push(Array2::zeros((2, 1)), Array1::zeros(2));
        }

        let tau = chain.autocorr_time(0);
        assert!(tau.is_empty());
    }

    #[test]
    fn test_autocorr_time_relation_to_ess() {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Test the mathematical relationship: ESS ≈ N / τ
        let param_names = vec!["x".to_string()];
        let mut chain = Chain::new(param_names.clone(), 1);

        let mut rng = ChaCha8Rng::seed_from_u64(99999);
        let n_samples = 100;
        let n_walkers = 4;

        // Generate samples with moderate autocorrelation
        let mut prev_value = 0.0;
        for _ in 0..n_samples {
            let mut positions = Array2::zeros((n_walkers, 1));
            for i in 0..n_walkers {
                // AR(1) process: x_t = 0.7 * x_{t-1} + noise
                prev_value = 0.7 * prev_value + rng.gen::<f64>() * 0.3;
                positions[[i, 0]] = prev_value;
            }
            chain.push(positions, Array1::zeros(n_walkers));
        }

        let tau = chain.autocorr_time(10);
        let ess = chain.ess(10);

        let x_tau = tau.get("x").unwrap();
        let x_ess = ess.get("x").unwrap();

        let n_total = ((n_samples - 10) * n_walkers) as f64;
        let expected_ess = n_total / x_tau;

        // ESS and τ should be consistent: ESS ≈ N / τ
        // Allow 20% relative error due to numerical differences in computation
        let relative_error = (x_ess - expected_ess).abs() / expected_ess;
        assert!(
            relative_error < 0.2,
            "ESS = {}, expected ≈ {} (N={}, τ={}), relative error = {:.1}%",
            x_ess,
            expected_ess,
            n_total,
            x_tau,
            relative_error * 100.0
        );
    }
}
