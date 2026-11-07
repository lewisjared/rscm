//! Walker initialisation strategies for ensemble sampling.
//!
//! This module provides different strategies for initialising walker positions
//! in the parameter space before MCMC sampling begins.

use crate::{parameter_set::ParameterSet, Error, Result};
use ndarray::Array2;
use rand::RngExt;

/// Walker initialisation strategy for the ensemble sampler.
#[derive(Debug, Clone)]
pub enum WalkerInit {
    /// Sample walkers from the prior distribution
    FromPrior,

    /// Initialise walkers in a ball around a point
    Ball {
        /// Centre point for the ball
        centre: Vec<f64>,
        /// Radius of the ball (standard deviation in each dimension)
        radius: f64,
    },

    /// Explicit walker positions
    Explicit(Array2<f64>),
}

impl WalkerInit {
    /// Initialise walker positions.
    ///
    /// # Arguments
    ///
    /// * `n_walkers` - Number of walkers to initialise
    /// * `parameters` - Parameter set defining the parameter space
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Array of shape (n_walkers, n_parameters) with initial positions.
    pub fn initialise<R: RngExt>(
        &self,
        n_walkers: usize,
        parameters: &ParameterSet,
        rng: &mut R,
    ) -> Result<Array2<f64>> {
        match self {
            WalkerInit::FromPrior => {
                // Sample from prior
                Ok(parameters.sample_random_with_rng(n_walkers, rng))
            }
            WalkerInit::Ball { centre, radius } => {
                // Validate centre length
                if centre.len() != parameters.len() {
                    return Err(Error::InvalidParameter(format!(
                        "Ball centre length {} does not match parameter count {}",
                        centre.len(),
                        parameters.len()
                    )));
                }

                let n_parameters = parameters.len();
                let mut positions = Array2::zeros((n_walkers, n_parameters));

                for i in 0..n_walkers {
                    for j in 0..n_parameters {
                        // Sample from normal distribution around centre
                        let offset = rng.random::<f64>() - 0.5; // Uniform(-0.5, 0.5)
                        positions[[i, j]] = centre[j] + offset * radius;
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
                if positions.ncols() != parameters.len() {
                    return Err(Error::InvalidParameter(format!(
                        "Explicit positions have {} parameters, expected {}",
                        positions.ncols(),
                        parameters.len()
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
        let parameters = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .add("y", Box::new(Uniform::new(-1.0, 1.0).unwrap()))
            .clone();

        let init = WalkerInit::FromPrior;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let positions = init.initialise(10, &parameters, &mut rng).unwrap();

        assert_eq!(positions.dim(), (10, 2));

        // Check all samples are in bounds
        for i in 0..10 {
            assert!(positions[[i, 0]] >= 0.0 && positions[[i, 0]] <= 1.0);
            assert!(positions[[i, 1]] >= -1.0 && positions[[i, 1]] <= 1.0);
        }
    }

    #[test]
    fn test_walker_init_ball() {
        let parameters = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 10.0).unwrap()))
            .add("y", Box::new(Uniform::new(0.0, 10.0).unwrap()))
            .clone();

        let init = WalkerInit::Ball {
            centre: vec![5.0, 5.0],
            radius: 0.1,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let positions = init.initialise(10, &parameters, &mut rng).unwrap();

        assert_eq!(positions.dim(), (10, 2));

        // Check all walkers are near centre
        for i in 0..10 {
            assert!((positions[[i, 0]] - 5.0).abs() < 0.1);
            assert!((positions[[i, 1]] - 5.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_walker_init_ball_wrong_dimension() {
        let parameters = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .clone();

        let init = WalkerInit::Ball {
            centre: vec![0.5, 0.5], // Wrong dimension
            radius: 0.1,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = init.initialise(10, &parameters, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_walker_init_explicit() {
        let parameters = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .add("y", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .clone();

        let explicit_positions = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
        let init = WalkerInit::Explicit(explicit_positions.clone());
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let positions = init.initialise(3, &parameters, &mut rng).unwrap();

        assert_eq!(positions, explicit_positions);
    }

    #[test]
    fn test_walker_init_explicit_wrong_dimension() {
        let parameters = ParameterSet::new()
            .add("x", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .add("y", Box::new(Uniform::new(0.0, 1.0).unwrap()))
            .clone();

        // Wrong number of walkers
        let explicit_positions = array![[0.1, 0.2], [0.3, 0.4]];
        let init = WalkerInit::Explicit(explicit_positions);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = init.initialise(3, &parameters, &mut rng);
        assert!(result.is_err());
    }
}
