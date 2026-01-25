//! Sampler state management and progress tracking.
//!
//! This module contains:
//! - `ProgressInfo`: Information passed to progress callbacks during sampling
//! - `SamplerState`: Current state of all walkers including positions and acceptance tracking

use crate::{Error, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

/// Information about sampling progress.
///
/// Passed to progress callbacks during MCMC sampling.
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    /// Current iteration number (0-indexed)
    pub iteration: usize,

    /// Total number of iterations
    pub total: usize,

    /// Mean acceptance rate across all walkers
    pub acceptance_rate: f64,

    /// Mean log probability across all walkers
    pub mean_log_prob: f64,
}

/// State of the ensemble sampler at a given iteration.
///
/// Contains the current positions of all walkers, their log probabilities,
/// and acceptance tracking information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerState {
    /// Current positions of walkers: shape (n_walkers, n_params)
    pub positions: Array2<f64>,

    /// Log probabilities at current positions: shape (n_walkers,)
    pub log_probs: Array1<f64>,

    /// Total number of proposals accepted for each walker
    pub n_accepted: Array1<usize>,

    /// Total number of proposals made for each walker
    pub n_proposed: Array1<usize>,

    /// Parameter names in the order they appear in position vectors
    pub param_names: Vec<String>,
}

impl SamplerState {
    /// Create a new sampler state from initial positions.
    ///
    /// Log probabilities will be computed by the sampler on first iteration.
    ///
    /// # Arguments
    ///
    /// * `positions` - Initial walker positions, shape (n_walkers, n_params)
    /// * `param_names` - Names of parameters in order
    ///
    /// # Returns
    ///
    /// A new `SamplerState` with log probabilities set to negative infinity
    /// (indicating they need to be computed) and zero acceptance counts.
    pub fn new(positions: Array2<f64>, param_names: Vec<String>) -> Result<Self> {
        let (n_walkers, n_params) = positions.dim();

        if param_names.len() != n_params {
            return Err(Error::SamplingError(format!(
                "Number of parameter names ({}) does not match positions dimension ({})",
                param_names.len(),
                n_params
            )));
        }

        if n_walkers < 2 {
            return Err(Error::SamplingError(
                "Must have at least 2 walkers for ensemble sampling".to_string(),
            ));
        }

        Ok(Self {
            positions,
            log_probs: Array1::from_elem(n_walkers, f64::NEG_INFINITY),
            n_accepted: Array1::zeros(n_walkers),
            n_proposed: Array1::zeros(n_walkers),
            param_names,
        })
    }

    /// Get the number of walkers.
    pub fn n_walkers(&self) -> usize {
        self.positions.nrows()
    }

    /// Get the number of parameters.
    pub fn n_params(&self) -> usize {
        self.positions.ncols()
    }

    /// Get the acceptance fraction for each walker.
    ///
    /// Returns the ratio of accepted to proposed moves for each walker.
    /// Returns 0.0 for walkers that have not had any proposals yet.
    pub fn acceptance_fraction(&self) -> Array1<f64> {
        let mut fractions = Array1::zeros(self.n_walkers());
        for i in 0..self.n_walkers() {
            if self.n_proposed[i] > 0 {
                fractions[i] = self.n_accepted[i] as f64 / self.n_proposed[i] as f64;
            }
        }
        fractions
    }

    /// Get the mean acceptance rate across all walkers.
    pub fn mean_acceptance_rate(&self) -> f64 {
        let total_accepted: usize = self.n_accepted.iter().sum();
        let total_proposed: usize = self.n_proposed.iter().sum();

        if total_proposed > 0 {
            total_accepted as f64 / total_proposed as f64
        } else {
            0.0
        }
    }

    /// Save the sampler state to a checkpoint file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the checkpoint file to create
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err` if serialization or file writing fails.
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path).map_err(|e| {
            Error::SamplingError(format!("Failed to create checkpoint file: {}", e))
        })?;
        let mut writer = BufWriter::new(file);

        bincode::serialize_into(&mut writer, self)
            .map_err(|e| Error::SamplingError(format!("Failed to serialize checkpoint: {}", e)))?;

        writer
            .flush()
            .map_err(|e| Error::SamplingError(format!("Failed to flush checkpoint file: {}", e)))?;

        Ok(())
    }

    /// Load a sampler state from a checkpoint file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the checkpoint file to read
    ///
    /// # Returns
    ///
    /// The loaded `SamplerState`, or an error if deserialization fails.
    pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::SamplingError(format!("Failed to open checkpoint file: {}", e)))?;
        let mut reader = BufReader::new(file);

        let state: SamplerState = bincode::deserialize_from(&mut reader).map_err(|e| {
            Error::SamplingError(format!("Failed to deserialize checkpoint: {}", e))
        })?;

        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sampler_state_creation() {
        let positions = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];
        let param_names = vec!["x".to_string(), "y".to_string()];

        let state = SamplerState::new(positions.clone(), param_names.clone()).unwrap();

        assert_eq!(state.n_walkers(), 3);
        assert_eq!(state.n_params(), 2);
        assert_eq!(state.param_names, param_names);
        assert_eq!(state.positions, positions);
        assert!(state.log_probs.iter().all(|&lp| lp == f64::NEG_INFINITY));
        assert!(state.n_accepted.iter().all(|&n| n == 0));
        assert!(state.n_proposed.iter().all(|&n| n == 0));
    }

    #[test]
    fn test_sampler_state_validation() {
        let positions = array![[0.0, 1.0], [2.0, 3.0]];

        // Wrong number of parameter names
        let result = SamplerState::new(positions.clone(), vec!["x".to_string()]);
        assert!(result.is_err());

        // Too few walkers
        let positions_single = array![[0.0, 1.0]];
        let result = SamplerState::new(positions_single, vec!["x".to_string(), "y".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_acceptance_tracking() {
        let positions = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];
        let param_names = vec!["x".to_string(), "y".to_string()];

        let mut state = SamplerState::new(positions, param_names).unwrap();

        // No proposals yet
        assert_eq!(state.mean_acceptance_rate(), 0.0);
        assert_eq!(state.acceptance_fraction(), array![0.0, 0.0, 0.0]);

        // Simulate some proposals
        state.n_proposed[0] = 10;
        state.n_accepted[0] = 7;
        state.n_proposed[1] = 10;
        state.n_accepted[1] = 3;
        state.n_proposed[2] = 10;
        state.n_accepted[2] = 5;

        assert_eq!(state.mean_acceptance_rate(), 0.5);
        assert_eq!(state.acceptance_fraction(), array![0.7, 0.3, 0.5]);
    }

    #[test]
    fn test_sampler_state_checkpoint() {
        use std::fs;
        use tempfile::tempdir;

        let positions = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];
        let param_names = vec!["x".to_string(), "y".to_string()];
        let mut state = SamplerState::new(positions.clone(), param_names.clone()).unwrap();

        // Simulate some acceptance tracking
        state.n_proposed[0] = 10;
        state.n_accepted[0] = 7;
        state.log_probs = array![-1.0, -2.0, -3.0];

        // Save checkpoint
        let dir = tempdir().unwrap();
        let checkpoint_path = dir.path().join("test.checkpoint");
        state.save_checkpoint(&checkpoint_path).unwrap();

        // Load checkpoint
        let loaded_state = SamplerState::load_checkpoint(&checkpoint_path).unwrap();

        // Verify state matches
        assert_eq!(loaded_state.positions, state.positions);
        assert_eq!(loaded_state.log_probs, state.log_probs);
        assert_eq!(loaded_state.n_accepted, state.n_accepted);
        assert_eq!(loaded_state.n_proposed, state.n_proposed);
        assert_eq!(loaded_state.param_names, state.param_names);

        // Cleanup
        fs::remove_file(&checkpoint_path).unwrap();
    }
}
