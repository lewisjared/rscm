//! MCMC proposal moves for ensemble sampling.
//!
//! This module contains the stretch move algorithm from Goodman & Weare (2010),
//! used for generating proposals in the affine-invariant ensemble sampler.

use crate::{Error, Result};
use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;

/// Configuration for the stretch move proposal.
///
/// The stretch move is parameterized by a scale parameter `a` that controls
/// the proposal distribution. The default value of 2.0 is recommended by
/// Goodman & Weare (2010).
#[derive(Debug, Clone, Copy)]
pub struct StretchMove {
    /// Scale parameter for the stretch move (typically 2.0)
    pub a: f64,
}

impl Default for StretchMove {
    fn default() -> Self {
        Self { a: 2.0 }
    }
}

impl StretchMove {
    /// Create a new stretch move with custom scale parameter.
    ///
    /// # Arguments
    ///
    /// * `a` - Scale parameter, must be > 1.0. Recommended value is 2.0.
    pub fn new(a: f64) -> Result<Self> {
        if a <= 1.0 {
            return Err(Error::InvalidParameter(format!(
                "Stretch move scale parameter must be > 1.0, got {}",
                a
            )));
        }
        Ok(Self { a })
    }

    /// Sample a stretch factor z from the proposal distribution g(z).
    ///
    /// The distribution is g(z) = 1/sqrt(z) for z in [1/a, a], which can be
    /// sampled by drawing u ~ Uniform(0,1) and setting z = ((a-1)*u + 1)^2 / a.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// A stretch factor z in the range [1/a, a]
    pub fn sample_z<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u: f64 = rng.gen(); // Uniform(0, 1)

        ((self.a - 1.0) * u + 1.0).powi(2) / self.a
    }

    /// Compute the Metropolis-Hastings acceptance probability for a stretch move.
    ///
    /// For the stretch move, the acceptance probability is:
    /// min(1, z^(n_params - 1) * exp(log_prob_new - log_prob_old))
    ///
    /// # Arguments
    ///
    /// * `z` - Stretch factor used for the proposal
    /// * `n_params` - Number of parameters (dimensionality)
    /// * `log_prob_old` - Log probability at current position
    /// * `log_prob_new` - Log probability at proposed position
    ///
    /// # Returns
    ///
    /// Acceptance probability in [0, 1]
    pub fn acceptance_probability(
        &self,
        z: f64,
        n_params: usize,
        log_prob_old: f64,
        log_prob_new: f64,
    ) -> f64 {
        // Handle invalid probabilities
        if !log_prob_new.is_finite() {
            return 0.0;
        }

        let log_ratio = (n_params as f64 - 1.0) * z.ln() + (log_prob_new - log_prob_old);
        let prob = log_ratio.exp();

        prob.min(1.0)
    }

    /// Generate a proposal position using the stretch move.
    ///
    /// The proposal is: y = c + z * (x - c), where:
    /// - x is the current walker position
    /// - c is a complementary walker position (chosen uniformly from the other walkers)
    /// - z is a stretch factor sampled from the proposal distribution
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    /// * `current_pos` - Current position of the walker
    /// * `complementary_positions` - Positions of all walkers in the complementary ensemble
    ///
    /// # Returns
    ///
    /// Tuple of (proposed_position, stretch_factor)
    pub fn propose<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        current_pos: ArrayView1<f64>,
        complementary_positions: &Array2<f64>,
    ) -> (Array1<f64>, f64) {
        // Sample stretch factor
        let z = self.sample_z(rng);

        // Select random complementary walker
        let n_complementary = complementary_positions.nrows();
        let comp_idx = rng.gen_range(0..n_complementary);
        let comp_pos = complementary_positions.row(comp_idx);

        // Compute proposal: y = c + z * (x - c)
        let proposal = &comp_pos + z * (&current_pos - &comp_pos);

        (proposal.to_owned(), z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_stretch_move_creation() {
        // Default should be a=2.0
        let stretch = StretchMove::default();
        assert_eq!(stretch.a, 2.0);

        // Custom value
        let stretch = StretchMove::new(2.5).unwrap();
        assert_eq!(stretch.a, 2.5);

        // Invalid value (a <= 1.0)
        let result = StretchMove::new(1.0);
        assert!(result.is_err());

        let result = StretchMove::new(0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_stretch_move_sample_z() {
        let stretch = StretchMove::default(); // a = 2.0
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Sample many z values and check they're in valid range [1/a, a] = [0.5, 2.0]
        for _ in 0..1000 {
            let z = stretch.sample_z(&mut rng);
            assert!((0.5..=2.0).contains(&z), "z = {} out of range", z);
        }

        // Check distribution properties
        // For g(z) = 1/sqrt(z), the distribution is NOT symmetric around 1.0
        // The mean is biased towards higher values
        // We just verify values are in the correct range
        let samples: Vec<f64> = (0..10000).map(|_| stretch.sample_z(&mut rng)).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;

        // Mean should be between 1/a and a
        assert!(
            (0.5..=2.0).contains(&mean),
            "Mean of z samples {} out of range [0.5, 2.0]",
            mean
        );
    }

    #[test]
    fn test_stretch_move_acceptance_probability() {
        let stretch = StretchMove::default();

        // Perfect acceptance (same log prob, z=1)
        let prob = stretch.acceptance_probability(1.0, 2, -10.0, -10.0);
        assert_eq!(prob, 1.0);

        // Better log prob should increase acceptance
        let prob_better = stretch.acceptance_probability(1.0, 2, -10.0, -5.0);
        assert!(prob_better > 0.99); // Should be very close to 1.0

        // Worse log prob should decrease acceptance
        let prob_worse = stretch.acceptance_probability(1.0, 2, -10.0, -15.0);
        assert!(prob_worse < 0.5);

        // Invalid new log prob (NaN or infinity) should give zero acceptance
        let prob_invalid = stretch.acceptance_probability(1.0, 2, -10.0, f64::NAN);
        assert_eq!(prob_invalid, 0.0);

        let prob_invalid = stretch.acceptance_probability(1.0, 2, -10.0, f64::INFINITY);
        assert_eq!(prob_invalid, 0.0);

        // Check dimensionality factor (z^(n-1))
        // For n=3, z=2.0, same log prob: acceptance = z^2 = 4.0, capped at 1.0
        let prob_dim = stretch.acceptance_probability(2.0, 3, -10.0, -10.0);
        assert_eq!(prob_dim, 1.0);

        // For z=0.5, n=3: z^2 = 0.25
        let prob_dim_small = stretch.acceptance_probability(0.5, 3, -10.0, -10.0);
        assert!((prob_dim_small - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_stretch_move_propose() {
        let stretch = StretchMove::default();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let current = array![1.0, 2.0];
        let complementary = array![[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let (proposal, z) = stretch.propose(&mut rng, current.view(), &complementary);

        // Check z is in valid range
        assert!((0.5..=2.0).contains(&z));

        // Check proposal has correct dimensionality
        assert_eq!(proposal.len(), 2);

        // Test with a specific known case
        let mut rng2 = ChaCha8Rng::seed_from_u64(123);
        let current2 = array![0.0, 0.0];
        let comp2 = array![[1.0, 1.0]];

        let (prop2, z2) = stretch.propose(&mut rng2, current2.view(), &comp2);

        // Proposal should be: c + z * (x - c) = [1,1] + z * ([0,0] - [1,1])
        //                                      = [1,1] + z * [-1,-1]
        //                                      = [1-z, 1-z]
        let expected = array![1.0 - z2, 1.0 - z2];
        assert!((prop2[0] - expected[0]).abs() < 1e-10);
        assert!((prop2[1] - expected[1]).abs() < 1e-10);
    }

    #[test]
    fn test_stretch_move_determinism() {
        let stretch = StretchMove::default();
        let current = array![1.0, 2.0, 3.0];
        let complementary = array![[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        // Same seed should give same results
        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        let (prop1, z1) = stretch.propose(&mut rng1, current.view(), &complementary);

        let mut rng2 = ChaCha8Rng::seed_from_u64(123);
        let (prop2, z2) = stretch.propose(&mut rng2, current.view(), &complementary);

        assert_eq!(z1, z2);
        assert_eq!(prop1, prop2);
    }
}
