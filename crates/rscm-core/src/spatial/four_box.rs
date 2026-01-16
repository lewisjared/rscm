use super::SpatialGrid;
use crate::errors::{RSCMError, RSCMResult};
use crate::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

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
    /// Panics if weights do not sum to approximately 1.0 (within 1e-6), or if
    /// the northern hemisphere weights (NorthernOcean + NorthernLand) or
    /// southern hemisphere weights (SouthernOcean + SouthernLand) sum to zero,
    /// which would cause division by zero in hemispheric transformations.
    pub fn with_weights(weights: [FloatValue; 4]) -> Self {
        let sum: FloatValue = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Weights must sum to 1.0, got {}",
            sum
        );
        let northern_sum = weights[FourBoxRegion::NorthernOcean as usize]
            + weights[FourBoxRegion::NorthernLand as usize];
        let southern_sum = weights[FourBoxRegion::SouthernOcean as usize]
            + weights[FourBoxRegion::SouthernLand as usize];
        assert!(
            northern_sum > 1e-10,
            "Northern hemisphere weights must be non-zero for hemispheric transformation, got {}",
            northern_sum
        );
        assert!(
            southern_sum > 1e-10,
            "Southern hemisphere weights must be non-zero for hemispheric transformation, got {}",
            southern_sum
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
