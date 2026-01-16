use super::SpatialGrid;
use crate::errors::RSCMResult;
use crate::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

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
