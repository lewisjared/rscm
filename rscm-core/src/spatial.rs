//! Spatial grid types for representing spatially-resolved climate data
//!
//! This module provides the [`SpatialGrid`] trait and implementations for common grid structures
//! used in climate models, including:
//!
//! - [`ScalarGrid`]: Single global value (backwards compatible with scalar timeseries)
//! - [`FourBoxGrid`]: MAGICC standard four-box structure (Northern Ocean, Northern Land, Southern Ocean, Southern Land)
//! - [`HemisphericGrid`]: Simple Northern/Southern hemisphere split
//!
//! # Examples
//!
//! ```rust
//! use rscm_core::spatial::{FourBoxGrid, SpatialGrid};
//!
//! let grid = FourBoxGrid::magicc_standard();
//! assert_eq!(grid.size(), 4);
//! assert_eq!(grid.grid_name(), "FourBox");
//!
//! // Aggregate regional values to global
//! let regional_temps = vec![15.0, 14.0, 10.0, 9.0]; // °C
//! let global_temp = grid.aggregate_global(&regional_temps);
//! assert_eq!(global_temp, 12.0); // Equal weights = simple average
//! ```

use crate::errors::{RSCMError, RSCMResult};
use crate::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Trait for spatial grid structures used in climate models
///
/// A spatial grid defines how climate variables are discretized spatially.
/// For example, a four-box grid divides the world into Northern Ocean, Northern Land,
/// Southern Ocean, and Southern Land regions.
///
/// The trait provides methods for:
/// - Querying grid structure (size, region names)
/// - Aggregating regional values to global values
/// - Transforming between different grid types
///
/// Note: This trait is not object-safe due to the generic `transform_to` method.
/// Use the concrete grid types directly rather than trait objects.
pub trait SpatialGrid: Clone + std::fmt::Debug + Send + Sync {
    /// Unique name for this grid type
    ///
    /// Used for error messages and debugging
    fn grid_name(&self) -> &'static str;

    /// Number of spatial regions in this grid
    fn size(&self) -> usize;

    /// Names of regions in this grid
    ///
    /// For example, a four-box grid returns:
    /// `["Northern Ocean", "Northern Land", "Southern Ocean", "Southern Land"]`
    fn region_names(&self) -> &[String];

    /// Aggregate all regional values to a single global value
    ///
    /// Uses grid-specific weights (typically area fractions) to compute
    /// a weighted average of all regions.
    ///
    /// # Arguments
    ///
    /// * `values` - Regional values to aggregate (must have length equal to `self.size()`)
    ///
    /// # Panics
    ///
    /// Panics if `values.len()` does not match `self.size()`
    fn aggregate_global(&self, values: &[FloatValue]) -> FloatValue;

    /// Transform values from this grid to another grid type
    ///
    /// This method performs explicit grid transformations where defined.
    /// Unsupported transformations return an error to prevent silent data loss.
    ///
    /// # Arguments
    ///
    /// * `values` - Regional values in this grid's coordinate system
    /// * `target` - Target grid to transform to
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<FloatValue>)` - Transformed values in target grid's coordinate system
    /// * `Err(RSCMError::UnsupportedGridTransformation)` - If transformation is not defined
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rscm_core::spatial::{FourBoxGrid, ScalarGrid, SpatialGrid};
    ///
    /// let four_box = FourBoxGrid::magicc_standard();
    /// let scalar = ScalarGrid;
    ///
    /// let regional = vec![15.0, 14.0, 10.0, 9.0];
    /// let global = four_box.transform_to(&regional, &scalar).unwrap();
    /// assert_eq!(global.len(), 1);
    /// assert_eq!(global[0], 12.0); // Average of regional values
    /// ```
    fn transform_to<G: SpatialGrid>(
        &self,
        values: &[FloatValue],
        target: &G,
    ) -> RSCMResult<Vec<FloatValue>>;
}

/// Region enum for scalar (global) grid
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScalarRegion {
    /// Global region (only region in scalar grid)
    Global = 0,
}

impl From<ScalarRegion> for usize {
    fn from(r: ScalarRegion) -> usize {
        r as usize
    }
}

/// Single global region (scalar grid)
///
/// This grid type represents a single global value with no spatial structure.
/// It is used for backwards compatibility with scalar timeseries and for
/// variables that are truly spatially uniform (e.g., atmospheric CO₂ concentration).
///
/// # Examples
///
/// ```rust
/// use rscm_core::spatial::{ScalarGrid, ScalarRegion, SpatialGrid};
///
/// let grid = ScalarGrid;
/// assert_eq!(grid.size(), 1);
/// assert_eq!(grid.region_names()[0], "Global");
///
/// let value = vec![288.15]; // K
/// let global = grid.aggregate_global(&value);
/// assert_eq!(global, 288.15);
/// ```
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ScalarGrid;

impl SpatialGrid for ScalarGrid {
    fn grid_name(&self) -> &'static str {
        "Scalar"
    }

    fn size(&self) -> usize {
        1
    }

    fn region_names(&self) -> &[String] {
        static NAMES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        NAMES.get_or_init(|| vec!["Global".to_string()])
    }

    fn aggregate_global(&self, values: &[FloatValue]) -> FloatValue {
        assert_eq!(values.len(), 1, "ScalarGrid expects exactly 1 value");
        values[0]
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
            1 => Ok(values.to_vec()), // Scalar to Scalar (identity)
            _ => {
                // Broadcast scalar to all regions
                Ok(vec![values[0]; target.size()])
            }
        }
    }
}

/// Region enum for four-box grid
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FourBoxRegion {
    /// Northern Ocean region
    NorthernOcean = 0,
    /// Northern Land region
    NorthernLand = 1,
    /// Southern Ocean region
    SouthernOcean = 2,
    /// Southern Land region
    SouthernLand = 3,
}

impl From<FourBoxRegion> for usize {
    fn from(r: FourBoxRegion) -> usize {
        r as usize
    }
}

/// Four-box regional grid (MAGICC standard)
///
/// Divides the world into four regions based on hemisphere and land/ocean:
/// - Northern Ocean
/// - Northern Land
/// - Southern Ocean
/// - Southern Land
///
/// This is the standard regional structure used in MAGICC and provides
/// a simple but physically meaningful spatial discretization for climate models.
///
/// # Examples
///
/// ```rust
/// use rscm_core::spatial::{FourBoxGrid, FourBoxRegion, SpatialGrid};
///
/// // Create with default equal weights
/// let grid = FourBoxGrid::magicc_standard();
/// assert_eq!(grid.size(), 4);
///
/// // Use region enum
/// let region_idx: usize = FourBoxRegion::NorthernOcean.into();
/// assert_eq!(region_idx, 0);
///
/// // Create with custom area-based weights
/// let grid_weighted = FourBoxGrid::with_weights([0.25, 0.25, 0.40, 0.10]);
/// let regional = vec![15.0, 14.0, 10.0, 9.0];
/// let global = grid_weighted.aggregate_global(&regional);
/// // 0.25*15 + 0.25*14 + 0.40*10 + 0.10*9 = 3.75 + 3.5 + 4.0 + 0.9 = 12.15
/// assert!((global - 12.15).abs() < 1e-10);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FourBoxGrid {
    /// Weights for aggregating regions to global (typically area fractions)
    ///
    /// Must sum to 1.0. Order: Northern Ocean, Northern Land, Southern Ocean, Southern Land
    weights: [FloatValue; 4],
}

impl FourBoxGrid {
    /// Index for Northern Ocean region (deprecated, use FourBoxRegion enum)
    #[deprecated(
        since = "0.3.0",
        note = "Use FourBoxRegion::NorthernOcean enum instead"
    )]
    pub const NORTHERN_OCEAN: usize = 0;
    /// Index for Northern Land region (deprecated, use FourBoxRegion enum)
    #[deprecated(since = "0.3.0", note = "Use FourBoxRegion::NorthernLand enum instead")]
    pub const NORTHERN_LAND: usize = 1;
    /// Index for Southern Ocean region (deprecated, use FourBoxRegion enum)
    #[deprecated(
        since = "0.3.0",
        note = "Use FourBoxRegion::SouthernOcean enum instead"
    )]
    pub const SOUTHERN_OCEAN: usize = 2;
    /// Index for Southern Land region (deprecated, use FourBoxRegion enum)
    #[deprecated(since = "0.3.0", note = "Use FourBoxRegion::SouthernLand enum instead")]
    pub const SOUTHERN_LAND: usize = 3;

    /// Create a four-box grid with MAGICC standard equal weights
    ///
    /// All regions are weighted equally (0.25 each) for aggregation.
    /// This is a simple starting point; use [`with_weights`](Self::with_weights)
    /// for physically accurate area-based weights.
    pub fn magicc_standard() -> Self {
        Self {
            weights: [0.25, 0.25, 0.25, 0.25],
        }
    }
}

impl Default for FourBoxGrid {
    fn default() -> Self {
        Self::magicc_standard()
    }
}

impl FourBoxGrid {
    /// Create a four-box grid with custom weights
    ///
    /// Weights should typically be based on the actual surface area fractions
    /// of each region. They must sum to 1.0.
    ///
    /// # Panics
    ///
    /// Panics if weights do not sum to approximately 1.0 (within 1e-6)
    pub fn with_weights(weights: [FloatValue; 4]) -> Self {
        let sum: FloatValue = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Weights must sum to 1.0, got {}",
            sum
        );
        Self { weights }
    }

    /// Get the aggregation weights for this grid
    pub fn weights(&self) -> &[FloatValue; 4] {
        &self.weights
    }
}

impl SpatialGrid for FourBoxGrid {
    fn grid_name(&self) -> &'static str {
        "FourBox"
    }

    fn size(&self) -> usize {
        4
    }

    fn region_names(&self) -> &[String] {
        static NAMES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        NAMES.get_or_init(|| {
            vec![
                "Northern Ocean".to_string(),
                "Northern Land".to_string(),
                "Southern Ocean".to_string(),
                "Southern Land".to_string(),
            ]
        })
    }

    fn aggregate_global(&self, values: &[FloatValue]) -> FloatValue {
        assert_eq!(
            values.len(),
            4,
            "FourBoxGrid expects exactly 4 regional values"
        );

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
                // FourBox to Scalar: aggregate to global
                Ok(vec![self.aggregate_global(values)])
            }
            2 => {
                // FourBox to Hemispheric: aggregate by hemisphere
                let no = FourBoxRegion::NorthernOcean as usize;
                let nl = FourBoxRegion::NorthernLand as usize;
                let so = FourBoxRegion::SouthernOcean as usize;
                let sl = FourBoxRegion::SouthernLand as usize;

                let northern = (values[no] * self.weights[no] + values[nl] * self.weights[nl])
                    / (self.weights[no] + self.weights[nl]);
                let southern = (values[so] * self.weights[so] + values[sl] * self.weights[sl])
                    / (self.weights[so] + self.weights[sl]);
                Ok(vec![northern, southern])
            }
            4 => {
                // FourBox to FourBox: identity
                Ok(values.to_vec())
            }
            _ => Err(RSCMError::UnsupportedGridTransformation {
                from: self.grid_name().to_string(),
                to: target.grid_name().to_string(),
            }),
        }
    }
}

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
/// let hemispheric = vec![15.0, 10.0]; // °C
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
    /// Index for Northern Hemisphere (deprecated, use HemisphericRegion enum)
    #[deprecated(since = "0.3.0", note = "Use HemisphericRegion::Northern enum instead")]
    pub const NORTHERN: usize = 0;
    /// Index for Southern Hemisphere (deprecated, use HemisphericRegion enum)
    #[deprecated(since = "0.3.0", note = "Use HemisphericRegion::Southern enum instead")]
    pub const SOUTHERN: usize = 1;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_grid_basic() {
        let grid = ScalarGrid;
        assert_eq!(grid.grid_name(), "Scalar");
        assert_eq!(grid.size(), 1);
        assert_eq!(grid.region_names().len(), 1);
        assert_eq!(grid.region_names()[0], "Global");
    }

    #[test]
    fn scalar_grid_aggregate() {
        let grid = ScalarGrid;
        let values = vec![288.15];
        assert_eq!(grid.aggregate_global(&values), 288.15);
    }

    #[test]
    fn scalar_grid_transform_identity() {
        let grid = ScalarGrid;
        let target = ScalarGrid;
        let values = vec![288.15];
        let result = grid.transform_to(&values, &target).unwrap();
        assert_eq!(result, vec![288.15]);
    }

    #[test]
    fn scalar_grid_broadcast_to_four_box() {
        let grid = ScalarGrid;
        let target = FourBoxGrid::magicc_standard();
        let values = vec![288.15];
        let result = grid.transform_to(&values, &target).unwrap();
        assert_eq!(result, vec![288.15, 288.15, 288.15, 288.15]);
    }

    #[test]
    fn four_box_grid_basic() {
        let grid = FourBoxGrid::magicc_standard();
        assert_eq!(grid.grid_name(), "FourBox");
        assert_eq!(grid.size(), 4);
        assert_eq!(grid.region_names().len(), 4);
        assert_eq!(
            grid.region_names()[FourBoxGrid::NORTHERN_OCEAN],
            "Northern Ocean"
        );
        assert_eq!(
            grid.region_names()[FourBoxGrid::NORTHERN_LAND],
            "Northern Land"
        );
        assert_eq!(
            grid.region_names()[FourBoxGrid::SOUTHERN_OCEAN],
            "Southern Ocean"
        );
        assert_eq!(
            grid.region_names()[FourBoxGrid::SOUTHERN_LAND],
            "Southern Land"
        );
    }

    #[test]
    fn four_box_grid_aggregate_equal_weights() {
        let grid = FourBoxGrid::magicc_standard();
        let values = vec![15.0, 14.0, 10.0, 9.0];
        let global = grid.aggregate_global(&values);
        assert_eq!(global, 12.0); // (15 + 14 + 10 + 9) / 4
    }

    #[test]
    fn four_box_grid_aggregate_custom_weights() {
        let grid = FourBoxGrid::with_weights([0.25, 0.25, 0.40, 0.10]);
        let values = vec![15.0, 14.0, 10.0, 9.0];
        let global = grid.aggregate_global(&values);
        // 0.25*15 = 3.75, 0.25*14 = 3.5, 0.40*10 = 4.0, 0.10*9 = 0.9
        // Sum = 3.75 + 3.5 + 4.0 + 0.9 = 12.15
        assert!((global - 12.15).abs() < 1e-10);
    }

    #[test]
    fn four_box_grid_transform_to_scalar() {
        let grid = FourBoxGrid::magicc_standard();
        let target = ScalarGrid;
        let values = vec![15.0, 14.0, 10.0, 9.0];
        let result = grid.transform_to(&values, &target).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 12.0);
    }

    #[test]
    fn four_box_grid_transform_to_hemispheric() {
        let grid = FourBoxGrid::magicc_standard();
        let target = HemisphericGrid::equal_weights();
        let values = vec![16.0, 14.0, 12.0, 8.0];
        let result = grid.transform_to(&values, &target).unwrap();
        assert_eq!(result.len(), 2);
        // Northern: (16*0.25 + 14*0.25) / (0.25 + 0.25) = (4 + 3.5) / 0.5 = 15
        // Southern: (12*0.25 + 8*0.25) / (0.25 + 0.25) = (3 + 2) / 0.5 = 10
        assert_eq!(result[0], 15.0);
        assert_eq!(result[1], 10.0);
    }

    #[test]
    fn four_box_grid_transform_identity() {
        let grid = FourBoxGrid::magicc_standard();
        let target = FourBoxGrid::magicc_standard();
        let values = vec![15.0, 14.0, 10.0, 9.0];
        let result = grid.transform_to(&values, &target).unwrap();
        assert_eq!(result, values);
    }

    #[test]
    fn hemispheric_grid_basic() {
        let grid = HemisphericGrid::equal_weights();
        assert_eq!(grid.grid_name(), "Hemispheric");
        assert_eq!(grid.size(), 2);
        assert_eq!(grid.region_names().len(), 2);
        assert_eq!(
            grid.region_names()[HemisphericGrid::NORTHERN],
            "Northern Hemisphere"
        );
        assert_eq!(
            grid.region_names()[HemisphericGrid::SOUTHERN],
            "Southern Hemisphere"
        );
    }

    #[test]
    fn hemispheric_grid_aggregate() {
        let grid = HemisphericGrid::equal_weights();
        let values = vec![15.0, 10.0];
        let global = grid.aggregate_global(&values);
        assert_eq!(global, 12.5);
    }

    #[test]
    fn hemispheric_grid_transform_to_scalar() {
        let grid = HemisphericGrid::equal_weights();
        let target = ScalarGrid;
        let values = vec![15.0, 10.0];
        let result = grid.transform_to(&values, &target).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 12.5);
    }

    #[test]
    fn hemispheric_grid_transform_to_four_box_errors() {
        let grid = HemisphericGrid::equal_weights();
        let target = FourBoxGrid::magicc_standard();
        let values = vec![15.0, 10.0];
        let result = grid.transform_to(&values, &target);
        assert!(result.is_err());
        if let Err(RSCMError::UnsupportedGridTransformation { from, to }) = result {
            assert_eq!(from, "Hemispheric");
            assert_eq!(to, "FourBox");
        }
    }

    #[test]
    #[should_panic(expected = "Weights must sum to 1.0")]
    fn four_box_invalid_weights() {
        FourBoxGrid::with_weights([0.3, 0.3, 0.3, 0.3]);
    }

    #[test]
    #[should_panic(expected = "Weights must sum to 1.0")]
    fn hemispheric_invalid_weights() {
        HemisphericGrid::with_weights([0.6, 0.6]);
    }
}
