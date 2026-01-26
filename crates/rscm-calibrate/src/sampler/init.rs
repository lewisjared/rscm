//! Walker initialization strategies for ensemble sampling.
//!
//! This module provides different strategies for initializing walker positions
//! in the parameter space before MCMC sampling begins.

use crate::{parameter_set::ParameterSet, Error, Result};
use ndarray::Array2;
use rand::Rng;

/// Walker initialization strategy for the ensemble sampler.
#[derive(Debug, Clone)]
pub enum WalkerInit {
    /// Sample walkers from the prior distribution
    FromPrior,

    /// Initialize walkers in a ball around a point
    Ball {
        /// Center point for the ball
        center: Vec<f64>,
        /// Radius of the ball (standard deviation in each dimension)
        radius: f64,
    },

    /// Explicit walker positions
    Explicit(Array2<f64>),
}

impl WalkerInit {
    /// Initialize walker positions.
    ///
    /// # Arguments
    ///
    /// * `n_walkers` - Number of walkers to initialize
    /// * `params` - Parameter set defining the parameter space
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Array of shape (n_walkers, n_params) with initial positions.
    pub fn initialize<R: Rng>(
        &self,
        n_walkers: usize,
        params: &ParameterSet,
        rng: &mut R,
    ) -> Result<Array2<f64>> {
        match self {
            WalkerInit::FromPrior => {
                // Sample from prior
                Ok(params.sample_random_with_rng(n_walkers, rng))
            }
            WalkerInit::Ball { center, radius } => {
                // Validate center length
                if center.len() != params.len() {
                    return Err(Error::InvalidParameter(format!(
                        "Ball center length {} does not match parameter count {}",
                        center.len(),
                        params.len()
                    )));
                }

                let n_params = params.len();
                let mut positions = Array2::zeros((n_walkers, n_params));

                for i in 0..n_walkers {
                    for j in 0..n_params {
                        // Sample from normal distribution around center
                        let offset = rng.gen::<f64>() - 0.5; // Uniform(-0.5, 0.5)
                        positions[[i, j]] = center[j] + offset * radius;
                    }
                }

                Ok(positions)
            }
            WalkerInit::Explicit(positions) => {
                // Validate dimensions
                if positions.nrows() != n_walkers {
                    return Err(Error::InvalidParameter(format!(
                        "Explicit positions have {} walkers, expected {}",
                        positions.nrows(),
                        n_walkers
                    )));
                }
                if positions.ncols() != params.len() {
                    return Err(Error::InvalidParameter(format!(
                        "Explicit positions have {} parameters, expected {}",
                        positions.ncols(),
                        params.len()
                    )));
                }

                Ok(positions.clone())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ParameterSet, Uniform};
    use ndarray::array;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_walker_init_from_prior() {
        let params = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .add("y", Box::new(Uniform::new(-1.0, 1.0).unwrap()))
            .clone();

        let init = WalkerInit::FromPrior;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let positions = init.initialize(10, &params, &mut rng).unwrap();

        assert_eq!(positions.dim(), (10, 2));

        // Check all samples are in bounds
        for i in 0..10 {
            assert!(positions[[i, 0]] >= 0.0 && positions[[i, 0]] <= 1.0);
            assert!(positions[[i, 1]] >= -1.0 && positions[[i, 1]] <= 1.0);
        }
    }

    #[test]
    fn test_walker_init_ball() {
        let params = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 10.0).unwrap()))
            .add("y", Box::new(Uniform::new(0.0, 10.0).unwrap()))
            .clone();

        let init = WalkerInit::Ball {
            center: vec![5.0, 5.0],
            radius: 0.1,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let positions = init.initialize(10, &params, &mut rng).unwrap();

        assert_eq!(positions.dim(), (10, 2));

        // Check all walkers are near center
        for i in 0..10 {
            assert!((positions[[i, 0]] - 5.0).abs() < 0.1);
            assert!((positions[[i, 1]] - 5.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_walker_init_ball_wrong_dimension() {
        let params = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .clone();

        let init = WalkerInit::Ball {
            center: vec![0.5, 0.5], // Wrong dimension
            radius: 0.1,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = init.initialize(10, &params, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_walker_init_explicit() {
        let params = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .add("y", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .clone();

        let explicit_positions = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
        let init = WalkerInit::Explicit(explicit_positions.clone());
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let positions = init.initialize(3, &params, &mut rng).unwrap();

        assert_eq!(positions, explicit_positions);
    }

    #[test]
    fn test_walker_init_explicit_wrong_dimension() {
        let params = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .add("y", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .clone();

        // Wrong number of walkers
        let explicit_positions = array![[0.1, 0.2], [0.3, 0.4]];
        let init = WalkerInit::Explicit(explicit_positions);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = init.initialize(3, &params, &mut rng);
        assert!(result.is_err());
    }
}
