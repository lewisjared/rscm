use super::SpatialGrid;
use crate::errors::{RSCMError, RSCMResult};
use crate::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Region enum for hemispheric grid
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HemisphericRegion {
    /// Northern Hemisphere
    Northern = 0,
    /// Southern Hemisphere
    Southern = 1,
}

impl From<HemisphericRegion> for usize {
    fn from(r: HemisphericRegion) -> usize {
        r as usize
    }
}

/// Hemispheric grid (Northern/Southern split)
///
/// Divides the world into two regions based on hemisphere:
/// - Northern Hemisphere
/// - Southern Hemisphere
///
/// This provides an intermediate spatial resolution between scalar (global)
/// and four-box models, useful for representing basic latitudinal gradients.
///
/// # Examples
///
/// ```rust
/// use rscm_core::spatial::{HemisphericGrid, HemisphericRegion, SpatialGrid};
///
/// let grid = HemisphericGrid::equal_weights();
/// assert_eq!(grid.size(), 2);
/// assert_eq!(grid.region_names()[0], "Northern Hemisphere");
///
/// // Aggregate to global
/// let hemispheric = vec![15.0, 10.0]; // degC
/// let global = grid.aggregate_global(&hemispheric);
/// assert_eq!(global, 12.5); // Equal weights = simple average
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HemisphericGrid {
    /// Weights for aggregating hemispheres to global
    ///
    /// Must sum to 1.0. Order: Northern, Southern
    weights: [FloatValue; 2],
}

impl HemisphericGrid {
    /// Create a hemispheric grid with equal weights (0.5 each)
    pub fn equal_weights() -> Self {
        Self {
            weights: [0.5, 0.5],
        }
    }

    /// Create a hemispheric grid with custom weights
    ///
    /// # Panics
    ///
    /// Panics if weights do not sum to approximately 1.0 (within 1e-6)
    pub fn with_weights(weights: [FloatValue; 2]) -> Self {
        let sum: FloatValue = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Weights must sum to 1.0, got {}",
            sum
        );
        Self { weights }
    }

    /// Get the aggregation weights for this grid
    pub fn weights(&self) -> &[FloatValue; 2] {
        &self.weights
    }
}

impl Default for HemisphericGrid {
    fn default() -> Self {
        Self::equal_weights()
    }
}

impl SpatialGrid for HemisphericGrid {
    fn grid_name(&self) -> &'static str {
        "Hemispheric"
    }

    fn size(&self) -> usize {
        2
    }

    fn region_names(&self) -> &[String] {
        static NAMES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        NAMES.get_or_init(|| {
            vec![
                "Northern Hemisphere".to_string(),
                "Southern Hemisphere".to_string(),
            ]
        })
    }

    fn aggregate_global(&self, values: &[FloatValue]) -> FloatValue {
        assert_eq!(values.len(), 2, "HemisphericGrid expects exactly 2 values");

        values
            .iter()
            .zip(self.weights.iter())
            .map(|(v, w)| v * w)
            .sum()
    }

    fn transform_to<G: SpatialGrid>(
        &self,
        values: &[FloatValue],
        target: &G,
    ) -> RSCMResult<Vec<FloatValue>> {
        assert_eq!(
            values.len(),
            self.size(),
            "Values length must match grid size"
        );

        match target.size() {
            1 => {
                // Hemispheric to Scalar: aggregate to global
                Ok(vec![self.aggregate_global(values)])
            }
            2 => {
                // Hemispheric to Hemispheric: identity
                Ok(values.to_vec())
            }
            4 => {
                // Hemispheric to FourBox: not supported (cannot infer ocean/land split)
                Err(RSCMError::UnsupportedGridTransformation {
                    from: self.grid_name().to_string(),
                    to: target.grid_name().to_string(),
                })
            }
            _ => Err(RSCMError::UnsupportedGridTransformation {
                from: self.grid_name().to_string(),
                to: target.grid_name().to_string(),
            }),
        }
    }
}
