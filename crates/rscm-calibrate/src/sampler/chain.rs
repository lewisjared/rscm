//! MCMC chain storage and serialization.
//!
//! This module contains the `Chain` struct for storing MCMC samples with support
//! for thinning, serialization, and merging chains from checkpointed runs.

use crate::{Error, Result};
use indexmap::IndexMap;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

/// Storage for MCMC chain samples.
///
/// Stores all samples from all walkers, along with their log probabilities.
/// Supports thinning (storing only every Nth sample) to reduce memory usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chain {
    /// Stored samples: shape (n_stored, n_walkers, n_params)
    pub(crate) samples: Vec<Array2<f64>>,

    /// Log probabilities: shape (n_stored, n_walkers)
    pub(crate) log_probs: Vec<Array1<f64>>,

    /// Parameter names in order
    pub(crate) param_names: Vec<String>,

    /// Thinning interval (store every thin-th sample)
    pub(crate) thin: usize,

    /// Total number of iterations run (including thinned samples)
    pub(crate) total_iterations: usize,
}

impl Chain {
    /// Create a new empty chain.
    ///
    /// # Arguments
    ///
    /// * `param_names` - Names of parameters in order
    /// * `thin` - Thinning interval (store every thin-th sample). Default is 1 (no thinning).
    pub fn new(param_names: Vec<String>, thin: usize) -> Self {
        Self {
            samples: Vec::new(),
            log_probs: Vec::new(),
            param_names,
            thin: thin.max(1), // Ensure at least 1
            total_iterations: 0,
        }
    }

    /// Add a sample to the chain if it should be stored (based on thinning).
    ///
    /// # Arguments
    ///
    /// * `positions` - Walker positions, shape (n_walkers, n_params)
    /// * `log_probs` - Log probabilities, shape (n_walkers,)
    ///
    /// # Returns
    ///
    /// `true` if the sample was stored, `false` if it was skipped due to thinning.
    pub fn push(&mut self, positions: Array2<f64>, log_probs: Array1<f64>) -> bool {
        self.total_iterations += 1;

        if self.total_iterations.is_multiple_of(self.thin) {
            self.samples.push(positions);
            self.log_probs.push(log_probs);
            true
        } else {
            false
        }
    }

    /// Get the number of stored samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get the total number of iterations (including thinned samples).
    pub fn total_iterations(&self) -> usize {
        self.total_iterations
    }

    /// Get the thinning interval.
    pub fn thin(&self) -> usize {
        self.thin
    }

    /// Get parameter names.
    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    /// Get flattened samples, optionally discarding initial burn-in samples.
    ///
    /// # Arguments
    ///
    /// * `discard` - Number of initial samples to discard from each walker
    ///
    /// # Returns
    ///
    /// Array of shape ((len - discard) * n_walkers, n_params) containing all
    /// post-burn-in samples from all walkers, concatenated.
    pub fn flat_samples(&self, discard: usize) -> Array2<f64> {
        if self.is_empty() || discard >= self.len() {
            return Array2::zeros((0, self.param_names.len()));
        }

        let n_keep = self.len() - discard;
        let n_walkers = self.samples[0].nrows();
        let n_params = self.param_names.len();

        let mut flat = Array2::zeros((n_keep * n_walkers, n_params));

        for (i, sample) in self.samples.iter().skip(discard).enumerate() {
            for (j, walker) in sample.outer_iter().enumerate() {
                flat.row_mut(i * n_walkers + j).assign(&walker);
            }
        }

        flat
    }

    /// Get flattened log probabilities, optionally discarding initial burn-in samples.
    ///
    /// # Arguments
    ///
    /// * `discard` - Number of initial samples to discard from each walker
    ///
    /// # Returns
    ///
    /// Array of shape ((len - discard) * n_walkers,) containing all
    /// post-burn-in log probabilities from all walkers, concatenated.
    pub fn flat_log_probs(&self, discard: usize) -> Array1<f64> {
        if self.is_empty() || discard >= self.len() {
            return Array1::zeros(0);
        }

        let n_keep = self.len() - discard;
        let n_walkers = self.samples[0].nrows();

        let mut flat = Array1::zeros(n_keep * n_walkers);

        for (i, log_prob) in self.log_probs.iter().skip(discard).enumerate() {
            for (j, &lp) in log_prob.iter().enumerate() {
                flat[i * n_walkers + j] = lp;
            }
        }

        flat
    }

    /// Convert chain to a map of parameter name to sample array.
    ///
    /// Useful for computing diagnostics per parameter.
    ///
    /// # Arguments
    ///
    /// * `discard` - Number of initial samples to discard from each walker
    ///
    /// # Returns
    ///
    /// Map from parameter name to Array1 of all post-burn-in samples for that parameter.
    pub fn to_param_map(&self, discard: usize) -> IndexMap<String, Array1<f64>> {
        let flat = self.flat_samples(discard);
        let mut map = IndexMap::new();

        for (i, name) in self.param_names.iter().enumerate() {
            map.insert(name.clone(), flat.column(i).to_owned());
        }

        map
    }

    /// Save the chain to a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to create
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err` if serialization or file writing fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| Error::SamplingError(format!("Failed to create chain file: {}", e)))?;
        let mut writer = BufWriter::new(file);

        bincode::serialize_into(&mut writer, self)
            .map_err(|e| Error::SamplingError(format!("Failed to serialize chain: {}", e)))?;

        writer
            .flush()
            .map_err(|e| Error::SamplingError(format!("Failed to flush chain file: {}", e)))?;

        Ok(())
    }

    /// Load a chain from a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to read
    ///
    /// # Returns
    ///
    /// The loaded `Chain`, or an error if deserialization fails.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::SamplingError(format!("Failed to open chain file: {}", e)))?;
        let mut reader = BufReader::new(file);

        let chain: Chain = bincode::deserialize_from(&mut reader)
            .map_err(|e| Error::SamplingError(format!("Failed to deserialize chain: {}", e)))?;

        Ok(chain)
    }

    /// Merge another chain into this one.
    ///
    /// This is useful for combining chain segments from checkpointed runs.
    /// The chains must have the same parameter names and thinning interval.
    ///
    /// # Arguments
    ///
    /// * `other` - The chain to merge into this one
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err` if chains are incompatible.
    pub fn merge(&mut self, other: &Chain) -> Result<()> {
        if self.param_names != other.param_names {
            return Err(Error::SamplingError(format!(
                "Cannot merge chains with different parameter names: {:?} vs {:?}",
                self.param_names, other.param_names
            )));
        }

        if self.thin != other.thin {
            return Err(Error::SamplingError(format!(
                "Cannot merge chains with different thinning intervals: {} vs {}",
                self.thin, other.thin
            )));
        }

        // Append samples and log probs
        self.samples.extend(other.samples.iter().cloned());
        self.log_probs.extend(other.log_probs.iter().cloned());
        self.total_iterations += other.total_iterations;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_chain_creation_and_push() {
        let param_names = vec!["x".to_string(), "y".to_string()];
        let mut chain = Chain::new(param_names.clone(), 1);

        assert_eq!(chain.len(), 0);
        assert!(chain.is_empty());
        assert_eq!(chain.total_iterations(), 0);
        assert_eq!(chain.param_names(), param_names.as_slice());

        // Add first sample
        let pos1 = array![[0.0, 1.0], [2.0, 3.0]];
        let lp1 = array![-1.0, -2.0];
        assert!(chain.push(pos1.clone(), lp1.clone()));

        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());
        assert_eq!(chain.total_iterations(), 1);

        // Add second sample
        let pos2 = array![[0.5, 1.5], [2.5, 3.5]];
        let lp2 = array![-1.5, -2.5];
        assert!(chain.push(pos2, lp2));

        assert_eq!(chain.len(), 2);
        assert_eq!(chain.total_iterations(), 2);
    }

    #[test]
    fn test_chain_thinning() {
        let param_names = vec!["x".to_string()];
        let mut chain = Chain::new(param_names, 3);

        // Add 10 samples, only every 3rd should be stored
        for i in 0..10 {
            let pos = array![[i as f64]];
            let lp = array![-(i as f64)];
            let stored = chain.push(pos, lp);

            // Samples 3, 6, 9 should be stored (1-indexed iteration)
            let expected_stored = (i + 1) % 3 == 0;
            assert_eq!(stored, expected_stored, "Sample {} storage mismatch", i);
        }

        assert_eq!(chain.len(), 3); // Stored samples 3, 6, 9
        assert_eq!(chain.total_iterations(), 10);
    }

    #[test]
    fn test_flat_samples() {
        let param_names = vec!["x".to_string(), "y".to_string()];
        let mut chain = Chain::new(param_names, 1);

        // Add 3 samples with 2 walkers each
        chain.push(array![[0.0, 1.0], [2.0, 3.0]], array![-1.0, -2.0]);
        chain.push(array![[0.5, 1.5], [2.5, 3.5]], array![-1.5, -2.5]);
        chain.push(array![[1.0, 2.0], [3.0, 4.0]], array![-2.0, -3.0]);

        // No discard: 3 samples * 2 walkers = 6 total samples
        let flat = chain.flat_samples(0);
        assert_eq!(flat.dim(), (6, 2));

        // Check first walker's samples are in order
        assert_eq!(flat.row(0), array![0.0, 1.0]);
        assert_eq!(flat.row(2), array![0.5, 1.5]);
        assert_eq!(flat.row(4), array![1.0, 2.0]);

        // Discard first 1 sample: 2 samples * 2 walkers = 4 total samples
        let flat_discard = chain.flat_samples(1);
        assert_eq!(flat_discard.dim(), (4, 2));
        assert_eq!(flat_discard.row(0), array![0.5, 1.5]);

        // Discard all samples
        let flat_empty = chain.flat_samples(3);
        assert_eq!(flat_empty.dim(), (0, 2));
    }

    #[test]
    fn test_flat_log_probs() {
        let param_names = vec!["x".to_string()];
        let mut chain = Chain::new(param_names, 1);

        chain.push(array![[0.0], [1.0]], array![-1.0, -2.0]);
        chain.push(array![[0.5], [1.5]], array![-1.5, -2.5]);

        let flat_lp = chain.flat_log_probs(0);
        assert_eq!(flat_lp, array![-1.0, -2.0, -1.5, -2.5]);

        let flat_lp_discard = chain.flat_log_probs(1);
        assert_eq!(flat_lp_discard, array![-1.5, -2.5]);
    }

    #[test]
    fn test_to_param_map() {
        let param_names = vec!["x".to_string(), "y".to_string()];
        let mut chain = Chain::new(param_names.clone(), 1);

        chain.push(array![[0.0, 1.0], [2.0, 3.0]], array![-1.0, -2.0]);
        chain.push(array![[0.5, 1.5], [2.5, 3.5]], array![-1.5, -2.5]);

        let param_map = chain.to_param_map(0);

        assert_eq!(param_map.len(), 2);
        assert_eq!(param_map["x"], array![0.0, 2.0, 0.5, 2.5]);
        assert_eq!(param_map["y"], array![1.0, 3.0, 1.5, 3.5]);
    }

    #[test]
    fn test_chain_serialization() {
        let param_names = vec!["x".to_string(), "y".to_string()];
        let mut chain = Chain::new(param_names, 2);

        chain.push(array![[0.0, 1.0]], array![-1.0]);
        chain.push(array![[0.5, 1.5]], array![-1.5]);

        // Serialize and deserialize
        let serialized = serde_json::to_string(&chain).unwrap();
        let deserialized: Chain = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.len(), chain.len());
        assert_eq!(deserialized.total_iterations(), chain.total_iterations());
        assert_eq!(deserialized.param_names(), chain.param_names());
        assert_eq!(deserialized.thin(), chain.thin());
    }

    #[test]
    fn test_chain_save_load() {
        use std::fs;
        use tempfile::tempdir;

        let param_names = vec!["x".to_string(), "y".to_string()];
        let mut chain = Chain::new(param_names.clone(), 2);

        // Add some samples
        chain.push(array![[0.0, 1.0], [2.0, 3.0]], array![-1.0, -2.0]);
        chain.push(array![[0.5, 1.5], [2.5, 3.5]], array![-1.5, -2.5]);

        // Save chain
        let dir = tempdir().unwrap();
        let chain_path = dir.path().join("test.chain");
        chain.save(&chain_path).unwrap();

        // Load chain
        let loaded_chain = Chain::load(&chain_path).unwrap();

        // Verify chain matches
        assert_eq!(loaded_chain.len(), chain.len());
        assert_eq!(loaded_chain.total_iterations(), chain.total_iterations());
        assert_eq!(loaded_chain.param_names(), chain.param_names());
        assert_eq!(loaded_chain.thin(), chain.thin());
        assert_eq!(loaded_chain.flat_samples(0), chain.flat_samples(0));
        assert_eq!(loaded_chain.flat_log_probs(0), chain.flat_log_probs(0));

        // Cleanup
        fs::remove_file(&chain_path).unwrap();
    }

    #[test]
    fn test_chain_merge() {
        let param_names = vec!["x".to_string(), "y".to_string()];
        let mut chain1 = Chain::new(param_names.clone(), 1);
        let mut chain2 = Chain::new(param_names.clone(), 1);

        // Add samples to both chains
        chain1.push(array![[0.0, 1.0], [2.0, 3.0]], array![-1.0, -2.0]);
        chain1.push(array![[0.5, 1.5], [2.5, 3.5]], array![-1.5, -2.5]);

        chain2.push(array![[1.0, 2.0], [3.0, 4.0]], array![-2.0, -3.0]);
        chain2.push(array![[1.5, 2.5], [3.5, 4.5]], array![-2.5, -3.5]);

        // Merge chain2 into chain1
        chain1.merge(&chain2).unwrap();

        // Verify merged chain
        assert_eq!(chain1.len(), 4);
        assert_eq!(chain1.total_iterations(), 4);

        let flat_samples = chain1.flat_samples(0);
        assert_eq!(flat_samples.nrows(), 8); // 4 samples * 2 walkers

        // First samples from chain1
        assert_eq!(flat_samples.row(0), array![0.0, 1.0]);
        // Last samples from chain2
        assert_eq!(flat_samples.row(6), array![1.5, 2.5]);
    }

    #[test]
    fn test_chain_merge_incompatible() {
        let mut chain1 = Chain::new(vec!["x".to_string()], 1);
        let chain2 = Chain::new(vec!["y".to_string()], 1);

        // Should fail due to different parameter names
        let result = chain1.merge(&chain2);
        assert!(result.is_err());

        // Should fail due to different thinning
        let chain3 = Chain::new(vec!["x".to_string()], 2);
        let result = chain1.merge(&chain3);
        assert!(result.is_err());
    }
}
