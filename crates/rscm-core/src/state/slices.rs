//! Typed output slices for regional values.
//!
//! This module provides zero-cost wrappers for regional output values,
//! enabling type-safe region access instead of raw arrays with magic indices.

use crate::spatial::{FourBoxGrid, FourBoxRegion, HemisphericGrid, HemisphericRegion, SpatialGrid};
use crate::timeseries::FloatValue;

/// A zero-cost wrapper for four-box regional output values.
///
/// `FourBoxSlice` provides type-safe region access instead of raw arrays with magic indices.
/// It uses `#[repr(transparent)]` to ensure zero overhead compared to `[FloatValue; 4]`.
///
/// # Examples
///
/// ```rust
/// use rscm_core::state::FourBoxSlice;
/// use rscm_core::spatial::FourBoxRegion;
///
/// // Builder pattern for ergonomic construction
/// let slice = FourBoxSlice::new()
///     .with(FourBoxRegion::NorthernOcean, 15.0)
///     .with(FourBoxRegion::NorthernLand, 14.0)
///     .with(FourBoxRegion::SouthernOcean, 10.0)
///     .with(FourBoxRegion::SouthernLand, 9.0);
///
/// assert_eq!(slice.get(FourBoxRegion::NorthernOcean), 15.0);
/// ```
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FourBoxSlice(pub [FloatValue; 4]);

impl FourBoxSlice {
    /// Create a new FourBoxSlice initialised with NaN values.
    ///
    /// Using NaN as the initial value ensures that any unset regions will
    /// be immediately apparent in output (as NaN propagates through calculations).
    pub fn new() -> Self {
        Self([FloatValue::NAN; 4])
    }

    /// Create a new FourBoxSlice with all regions set to the same value.
    pub fn uniform(value: FloatValue) -> Self {
        Self([value; 4])
    }

    /// Create a new FourBoxSlice from an array of values.
    ///
    /// Order: [NorthernOcean, NorthernLand, SouthernOcean, SouthernLand]
    pub fn from_array(values: [FloatValue; 4]) -> Self {
        Self(values)
    }

    /// Builder method to set a single region's value.
    ///
    /// Returns `self` for method chaining.
    pub fn with(mut self, region: FourBoxRegion, value: FloatValue) -> Self {
        self.0[region as usize] = value;
        self
    }

    /// Set a region's value (mutating).
    pub fn set(&mut self, region: FourBoxRegion, value: FloatValue) {
        self.0[region as usize] = value;
    }

    /// Get a region's value.
    pub fn get(&self, region: FourBoxRegion) -> FloatValue {
        self.0[region as usize]
    }

    /// Get a mutable reference to a region's value.
    pub fn get_mut(&mut self, region: FourBoxRegion) -> &mut FloatValue {
        &mut self.0[region as usize]
    }

    /// Get the underlying array.
    pub fn as_array(&self) -> &[FloatValue; 4] {
        &self.0
    }

    /// Get the underlying array as a mutable reference.
    pub fn as_array_mut(&mut self) -> &mut [FloatValue; 4] {
        &mut self.0
    }

    /// Convert to a Vec.
    pub fn to_vec(&self) -> Vec<FloatValue> {
        self.0.to_vec()
    }

    /// Compute the global aggregate using a grid's weights.
    pub fn aggregate_global(&self, grid: &FourBoxGrid) -> FloatValue {
        grid.aggregate_global(&self.0)
    }
}

impl Default for FourBoxSlice {
    fn default() -> Self {
        Self::new()
    }
}

impl From<[FloatValue; 4]> for FourBoxSlice {
    fn from(values: [FloatValue; 4]) -> Self {
        Self(values)
    }
}

impl From<FourBoxSlice> for [FloatValue; 4] {
    fn from(slice: FourBoxSlice) -> Self {
        slice.0
    }
}

impl From<FourBoxSlice> for Vec<FloatValue> {
    fn from(slice: FourBoxSlice) -> Self {
        slice.0.to_vec()
    }
}

impl std::ops::Index<FourBoxRegion> for FourBoxSlice {
    type Output = FloatValue;

    fn index(&self, region: FourBoxRegion) -> &Self::Output {
        &self.0[region as usize]
    }
}

impl std::ops::IndexMut<FourBoxRegion> for FourBoxSlice {
    fn index_mut(&mut self, region: FourBoxRegion) -> &mut Self::Output {
        &mut self.0[region as usize]
    }
}

/// A zero-cost wrapper for hemispheric regional output values.
///
/// Similar to `FourBoxSlice` but for the two-region hemispheric grid.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HemisphericSlice(pub [FloatValue; 2]);

impl HemisphericSlice {
    /// Create a new HemisphericSlice initialised with NaN values.
    pub fn new() -> Self {
        Self([FloatValue::NAN; 2])
    }

    /// Create a new HemisphericSlice with both hemispheres set to the same value.
    pub fn uniform(value: FloatValue) -> Self {
        Self([value; 2])
    }

    /// Create a new HemisphericSlice from an array of values.
    ///
    /// Order: [Northern, Southern]
    pub fn from_array(values: [FloatValue; 2]) -> Self {
        Self(values)
    }

    /// Builder method to set a single hemisphere's value.
    pub fn with(mut self, region: HemisphericRegion, value: FloatValue) -> Self {
        self.0[region as usize] = value;
        self
    }

    /// Set a hemisphere's value (mutating).
    pub fn set(&mut self, region: HemisphericRegion, value: FloatValue) {
        self.0[region as usize] = value;
    }

    /// Get a hemisphere's value.
    pub fn get(&self, region: HemisphericRegion) -> FloatValue {
        self.0[region as usize]
    }

    /// Get a mutable reference to a hemisphere's value.
    pub fn get_mut(&mut self, region: HemisphericRegion) -> &mut FloatValue {
        &mut self.0[region as usize]
    }

    /// Get the underlying array.
    pub fn as_array(&self) -> &[FloatValue; 2] {
        &self.0
    }

    /// Get the underlying array as a mutable reference.
    pub fn as_array_mut(&mut self) -> &mut [FloatValue; 2] {
        &mut self.0
    }

    /// Convert to a Vec.
    pub fn to_vec(&self) -> Vec<FloatValue> {
        self.0.to_vec()
    }

    /// Compute the global aggregate using a grid's weights.
    pub fn aggregate_global(&self, grid: &HemisphericGrid) -> FloatValue {
        grid.aggregate_global(&self.0)
    }
}

impl Default for HemisphericSlice {
    fn default() -> Self {
        Self::new()
    }
}

impl From<[FloatValue; 2]> for HemisphericSlice {
    fn from(values: [FloatValue; 2]) -> Self {
        Self(values)
    }
}

impl From<HemisphericSlice> for [FloatValue; 2] {
    fn from(slice: HemisphericSlice) -> Self {
        slice.0
    }
}

impl From<HemisphericSlice> for Vec<FloatValue> {
    fn from(slice: HemisphericSlice) -> Self {
        slice.0.to_vec()
    }
}

impl std::ops::Index<HemisphericRegion> for HemisphericSlice {
    type Output = FloatValue;

    fn index(&self, region: HemisphericRegion) -> &Self::Output {
        &self.0[region as usize]
    }
}

impl std::ops::IndexMut<HemisphericRegion> for HemisphericSlice {
    fn index_mut(&mut self, region: HemisphericRegion) -> &mut Self::Output {
        &mut self.0[region as usize]
    }
}

#[cfg(test)]
mod typed_slice_tests {
    use super::*;
    use crate::spatial::{FourBoxGrid, HemisphericGrid};

    #[test]
    fn test_four_box_slice_new() {
        let slice = FourBoxSlice::new();
        assert!(slice.get(FourBoxRegion::NorthernOcean).is_nan());
        assert!(slice.get(FourBoxRegion::SouthernLand).is_nan());
    }

    #[test]
    fn test_four_box_slice_uniform() {
        let slice = FourBoxSlice::uniform(15.0);
        assert_eq!(slice.get(FourBoxRegion::NorthernOcean), 15.0);
        assert_eq!(slice.get(FourBoxRegion::SouthernLand), 15.0);
    }

    #[test]
    fn test_four_box_slice_builder() {
        let slice = FourBoxSlice::new()
            .with(FourBoxRegion::NorthernOcean, 16.0)
            .with(FourBoxRegion::NorthernLand, 15.0)
            .with(FourBoxRegion::SouthernOcean, 11.0)
            .with(FourBoxRegion::SouthernLand, 10.0);

        assert_eq!(slice.get(FourBoxRegion::NorthernOcean), 16.0);
        assert_eq!(slice.get(FourBoxRegion::NorthernLand), 15.0);
        assert_eq!(slice.get(FourBoxRegion::SouthernOcean), 11.0);
        assert_eq!(slice.get(FourBoxRegion::SouthernLand), 10.0);
    }

    #[test]
    fn test_four_box_slice_mutate() {
        let mut slice = FourBoxSlice::uniform(0.0);
        slice.set(FourBoxRegion::NorthernOcean, 42.0);
        assert_eq!(slice.get(FourBoxRegion::NorthernOcean), 42.0);

        *slice.get_mut(FourBoxRegion::SouthernLand) = 7.0;
        assert_eq!(slice.get(FourBoxRegion::SouthernLand), 7.0);
    }

    #[test]
    fn test_four_box_slice_index() {
        let mut slice = FourBoxSlice::from_array([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(slice[FourBoxRegion::NorthernOcean], 1.0);
        assert_eq!(slice[FourBoxRegion::NorthernLand], 2.0);

        slice[FourBoxRegion::SouthernOcean] = 99.0;
        assert_eq!(slice[FourBoxRegion::SouthernOcean], 99.0);
    }

    #[test]
    fn test_four_box_slice_conversions() {
        let slice = FourBoxSlice::from_array([1.0, 2.0, 3.0, 4.0]);

        let vec: Vec<FloatValue> = slice.into();
        assert_eq!(vec, vec![1.0, 2.0, 3.0, 4.0]);

        let slice2: FourBoxSlice = [5.0, 6.0, 7.0, 8.0].into();
        let arr: [FloatValue; 4] = slice2.into();
        assert_eq!(arr, [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_four_box_slice_aggregate_global() {
        let slice = FourBoxSlice::from_array([16.0, 14.0, 12.0, 10.0]);
        let grid = FourBoxGrid::magicc_standard();
        let global = slice.aggregate_global(&grid);
        // Equal weights: (16 + 14 + 12 + 10) / 4 = 13.0
        assert_eq!(global, 13.0);
    }

    #[test]
    fn test_hemispheric_slice_new() {
        let slice = HemisphericSlice::new();
        assert!(slice.get(HemisphericRegion::Northern).is_nan());
        assert!(slice.get(HemisphericRegion::Southern).is_nan());
    }

    #[test]
    fn test_hemispheric_slice_builder() {
        let slice = HemisphericSlice::new()
            .with(HemisphericRegion::Northern, 15.0)
            .with(HemisphericRegion::Southern, 10.0);

        assert_eq!(slice.get(HemisphericRegion::Northern), 15.0);
        assert_eq!(slice.get(HemisphericRegion::Southern), 10.0);
    }

    #[test]
    fn test_hemispheric_slice_index() {
        let mut slice = HemisphericSlice::from_array([15.0, 10.0]);
        assert_eq!(slice[HemisphericRegion::Northern], 15.0);
        assert_eq!(slice[HemisphericRegion::Southern], 10.0);

        slice[HemisphericRegion::Northern] = 20.0;
        assert_eq!(slice[HemisphericRegion::Northern], 20.0);
    }

    #[test]
    fn test_hemispheric_slice_aggregate_global() {
        let slice = HemisphericSlice::from_array([15.0, 10.0]);
        let grid = HemisphericGrid::equal_weights();
        let global = slice.aggregate_global(&grid);
        // Equal weights: (15 + 10) / 2 = 12.5
        assert_eq!(global, 12.5);
    }

    #[test]
    fn test_slice_default() {
        let four_box = FourBoxSlice::default();
        assert!(four_box.get(FourBoxRegion::NorthernOcean).is_nan());

        let hemispheric = HemisphericSlice::default();
        assert!(hemispheric.get(HemisphericRegion::Northern).is_nan());
    }
}
