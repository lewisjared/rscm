//! Aggregating window types for grid-to-scalar and grid-to-grid transformations.
//!
//! This module provides window types that aggregate data from finer grids (e.g., FourBox)
//! to coarser representations (e.g., Scalar or Hemispheric).

use crate::component::GridType;
use crate::spatial::{FourBoxGrid, HemisphericGrid, HemisphericRegion, SpatialGrid};
use crate::timeseries::{FloatValue, GridTimeseries, Time};

use super::windows::GridTimeseriesWindow;

/// A transformation context for grid aggregation.
///
/// This holds the information needed to transform grid data to scalar values
/// during read operations.
#[derive(Debug, Clone)]
pub struct ReadTransformInfo {
    /// The source grid type (finer resolution data)
    pub source_grid: GridType,
    /// The weights to use for aggregation (from Model's grid_weights)
    /// If None, use the grid's default weights
    pub weights: Option<Vec<f64>>,
}

/// A scalar window that aggregates from a FourBox timeseries.
///
/// This provides the same API as `TimeseriesWindow` but reads from a FourBox
/// timeseries and aggregates to a scalar value on each access.
#[derive(Debug)]
pub struct AggregatingFourBoxWindow<'a> {
    timeseries: &'a GridTimeseries<FloatValue, FourBoxGrid>,
    current_index: usize,
    current_time: Time,
    weights: Option<[f64; 4]>,
}

impl<'a> AggregatingFourBoxWindow<'a> {
    pub fn new(
        timeseries: &'a GridTimeseries<FloatValue, FourBoxGrid>,
        current_index: usize,
        current_time: Time,
        weights: Option<Vec<f64>>,
    ) -> Self {
        let weights = weights.map(|w| {
            let arr: [f64; 4] = w
                .as_slice()
                .try_into()
                .expect("FourBox weights must have 4 elements");
            arr
        });
        Self {
            timeseries,
            current_index,
            current_time,
            weights,
        }
    }

    fn aggregate(&self, values: &[FloatValue]) -> FloatValue {
        match &self.weights {
            Some(weights) => {
                let mut sum = 0.0;
                for (v, w) in values.iter().zip(weights.iter()) {
                    if !v.is_nan() {
                        sum += v * w;
                    }
                }
                sum
            }
            None => self.timeseries.grid().aggregate_global(values),
        }
    }

    /// Get the aggregated scalar value at the start of the timestep (index N).
    pub fn at_start(&self) -> FloatValue {
        let values = self
            .timeseries
            .at_time_index(self.current_index)
            .expect("Current index out of bounds");
        self.aggregate(&values)
    }

    /// Get the aggregated scalar value at the end of the timestep (index N+1), if available.
    pub fn at_end(&self) -> Option<FloatValue> {
        let next_index = self.current_index + 1;
        if next_index >= self.timeseries.len() {
            None
        } else {
            let values = self
                .timeseries
                .at_time_index(next_index)
                .expect("Next index out of bounds");
            Some(self.aggregate(&values))
        }
    }

    /// Get the aggregated value at the previous timestep.
    pub fn previous(&self) -> Option<FloatValue> {
        if self.current_index == 0 {
            None
        } else {
            let values = self
                .timeseries
                .at_time_index(self.current_index - 1)
                .expect("Previous index out of bounds");
            Some(self.aggregate(&values))
        }
    }

    /// Get the current time value.
    pub fn time(&self) -> Time {
        self.current_time
    }

    /// Get the current time index.
    pub fn index(&self) -> usize {
        self.current_index
    }

    /// Get the total length of the underlying timeseries.
    pub fn len(&self) -> usize {
        self.timeseries.len()
    }

    /// Check if the underlying timeseries is empty.
    pub fn is_empty(&self) -> bool {
        self.timeseries.is_empty()
    }
}

/// A scalar window that aggregates from a Hemispheric timeseries.
#[derive(Debug)]
pub struct AggregatingHemisphericWindow<'a> {
    timeseries: &'a GridTimeseries<FloatValue, HemisphericGrid>,
    current_index: usize,
    current_time: Time,
    weights: Option<[f64; 2]>,
}

impl<'a> AggregatingHemisphericWindow<'a> {
    pub fn new(
        timeseries: &'a GridTimeseries<FloatValue, HemisphericGrid>,
        current_index: usize,
        current_time: Time,
        weights: Option<Vec<f64>>,
    ) -> Self {
        let weights = weights.map(|w| {
            let arr: [f64; 2] = w
                .as_slice()
                .try_into()
                .expect("Hemispheric weights must have 2 elements");
            arr
        });
        Self {
            timeseries,
            current_index,
            current_time,
            weights,
        }
    }

    fn aggregate(&self, values: &[FloatValue]) -> FloatValue {
        match &self.weights {
            Some(weights) => {
                let mut sum = 0.0;
                for (v, w) in values.iter().zip(weights.iter()) {
                    if !v.is_nan() {
                        sum += v * w;
                    }
                }
                sum
            }
            None => self.timeseries.grid().aggregate_global(values),
        }
    }

    /// Get the aggregated scalar value at the start of the timestep (index N).
    pub fn at_start(&self) -> FloatValue {
        let values = self
            .timeseries
            .at_time_index(self.current_index)
            .expect("Current index out of bounds");
        self.aggregate(&values)
    }

    /// Get the aggregated scalar value at the end of the timestep (index N+1), if available.
    pub fn at_end(&self) -> Option<FloatValue> {
        let next_index = self.current_index + 1;
        if next_index >= self.timeseries.len() {
            None
        } else {
            let values = self
                .timeseries
                .at_time_index(next_index)
                .expect("Next index out of bounds");
            Some(self.aggregate(&values))
        }
    }

    /// Get the aggregated value at the previous timestep.
    pub fn previous(&self) -> Option<FloatValue> {
        if self.current_index == 0 {
            None
        } else {
            let values = self
                .timeseries
                .at_time_index(self.current_index - 1)
                .expect("Previous index out of bounds");
            Some(self.aggregate(&values))
        }
    }

    /// Get the current time value.
    pub fn time(&self) -> Time {
        self.current_time
    }

    /// Get the current time index.
    pub fn index(&self) -> usize {
        self.current_index
    }

    /// Get the total length of the underlying timeseries.
    pub fn len(&self) -> usize {
        self.timeseries.len()
    }

    /// Check if the underlying timeseries is empty.
    pub fn is_empty(&self) -> bool {
        self.timeseries.is_empty()
    }
}

/// A unified scalar window that can be either direct or aggregating.
///
/// This enum allows `InputState::get_scalar_window()` to return the same type
/// regardless of whether the underlying data is scalar or needs aggregation.
#[derive(Debug)]
pub enum ScalarWindow<'a> {
    /// Direct access to a scalar timeseries
    Direct(super::windows::TimeseriesWindow<'a>),
    /// Aggregating access to a FourBox timeseries
    FromFourBox(AggregatingFourBoxWindow<'a>),
    /// Aggregating access to a Hemispheric timeseries
    FromHemispheric(AggregatingHemisphericWindow<'a>),
}

impl<'a> ScalarWindow<'a> {
    /// Get the scalar value at the start of the timestep (index N).
    pub fn at_start(&self) -> FloatValue {
        match self {
            ScalarWindow::Direct(w) => w.at_start(),
            ScalarWindow::FromFourBox(w) => w.at_start(),
            ScalarWindow::FromHemispheric(w) => w.at_start(),
        }
    }

    /// Get the scalar value at the end of the timestep (index N+1), if available.
    pub fn at_end(&self) -> Option<FloatValue> {
        match self {
            ScalarWindow::Direct(w) => w.at_end(),
            ScalarWindow::FromFourBox(w) => w.at_end(),
            ScalarWindow::FromHemispheric(w) => w.at_end(),
        }
    }

    /// Get the value at the previous timestep, if available.
    pub fn previous(&self) -> Option<FloatValue> {
        match self {
            ScalarWindow::Direct(w) => w.previous(),
            ScalarWindow::FromFourBox(w) => w.previous(),
            ScalarWindow::FromHemispheric(w) => w.previous(),
        }
    }

    /// Get the current time value.
    pub fn time(&self) -> Time {
        match self {
            ScalarWindow::Direct(w) => w.time(),
            ScalarWindow::FromFourBox(w) => w.time(),
            ScalarWindow::FromHemispheric(w) => w.time(),
        }
    }

    /// Get the current time index.
    pub fn index(&self) -> usize {
        match self {
            ScalarWindow::Direct(w) => w.index(),
            ScalarWindow::FromFourBox(w) => w.index(),
            ScalarWindow::FromHemispheric(w) => w.index(),
        }
    }

    /// Get the total length of the underlying timeseries.
    pub fn len(&self) -> usize {
        match self {
            ScalarWindow::Direct(w) => w.len(),
            ScalarWindow::FromFourBox(w) => w.len(),
            ScalarWindow::FromHemispheric(w) => w.len(),
        }
    }

    /// Check if the underlying timeseries is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            ScalarWindow::Direct(w) => w.is_empty(),
            ScalarWindow::FromFourBox(w) => w.is_empty(),
            ScalarWindow::FromHemispheric(w) => w.is_empty(),
        }
    }
}

/// A scalar window that aggregates from a Hemispheric timeseries to Hemispheric output.
///
/// This is used when reading a FourBox timeseries but the component wants Hemispheric data.
#[derive(Debug)]
pub struct AggregatingFourBoxToHemisphericWindow<'a> {
    timeseries: &'a GridTimeseries<FloatValue, FourBoxGrid>,
    current_index: usize,
    current_time: Time,
}

impl<'a> AggregatingFourBoxToHemisphericWindow<'a> {
    pub fn new(
        timeseries: &'a GridTimeseries<FloatValue, FourBoxGrid>,
        current_index: usize,
        current_time: Time,
    ) -> Self {
        Self {
            timeseries,
            current_index,
            current_time,
        }
    }

    fn aggregate_to_hemispheric(&self, values: &[FloatValue]) -> [FloatValue; 2] {
        // FourBox: [NorthernOcean, NorthernLand, SouthernOcean, SouthernLand]
        // Hemispheric: [Northern, Southern]
        // Average ocean+land for each hemisphere
        let northern = (values[0] + values[1]) / 2.0;
        let southern = (values[2] + values[3]) / 2.0;
        [northern, southern]
    }

    /// Get all regional values at the start of the timestep.
    pub fn at_start_all(&self) -> Vec<FloatValue> {
        let values = self
            .timeseries
            .at_time_index(self.current_index)
            .expect("Current index out of bounds");
        self.aggregate_to_hemispheric(&values).to_vec()
    }

    /// Get all regional values at the end of the timestep.
    pub fn at_end_all(&self) -> Option<Vec<FloatValue>> {
        let next_index = self.current_index + 1;
        if next_index >= self.timeseries.len() {
            None
        } else {
            let values = self
                .timeseries
                .at_time_index(next_index)
                .expect("Next index out of bounds");
            Some(self.aggregate_to_hemispheric(&values).to_vec())
        }
    }

    /// Get a single region's value at the start of the timestep.
    pub fn at_start(&self, region: HemisphericRegion) -> FloatValue {
        let values = self.at_start_all();
        values[region as usize]
    }

    /// Get a single region's value at the end of the timestep.
    pub fn at_end(&self, region: HemisphericRegion) -> Option<FloatValue> {
        self.at_end_all().map(|v| v[region as usize])
    }

    /// Get the current time value.
    pub fn time(&self) -> Time {
        self.current_time
    }

    /// Get the current time index.
    pub fn index(&self) -> usize {
        self.current_index
    }

    /// Get the total length of the underlying timeseries.
    pub fn len(&self) -> usize {
        self.timeseries.len()
    }

    /// Check if the underlying timeseries is empty.
    pub fn is_empty(&self) -> bool {
        self.timeseries.is_empty()
    }
}

/// A unified hemispheric window that can be either direct or aggregating.
#[derive(Debug)]
pub enum HemisphericWindow<'a> {
    /// Direct access to a Hemispheric timeseries
    Direct(GridTimeseriesWindow<'a, HemisphericGrid>),
    /// Aggregating access from a FourBox timeseries
    FromFourBox(AggregatingFourBoxToHemisphericWindow<'a>),
}

impl<'a> HemisphericWindow<'a> {
    /// Get all regional values at the start of the timestep.
    pub fn at_start_all(&self) -> Vec<FloatValue> {
        match self {
            HemisphericWindow::Direct(w) => w.at_start_all(),
            HemisphericWindow::FromFourBox(w) => w.at_start_all(),
        }
    }

    /// Get all regional values at the end of the timestep.
    pub fn at_end_all(&self) -> Option<Vec<FloatValue>> {
        match self {
            HemisphericWindow::Direct(w) => w.at_end_all(),
            HemisphericWindow::FromFourBox(w) => w.at_end_all(),
        }
    }

    /// Get a single region's value at the start of the timestep.
    pub fn at_start(&self, region: HemisphericRegion) -> FloatValue {
        match self {
            HemisphericWindow::Direct(w) => w.at_start(region),
            HemisphericWindow::FromFourBox(w) => w.at_start(region),
        }
    }

    /// Get a single region's value at the end of the timestep.
    pub fn at_end(&self, region: HemisphericRegion) -> Option<FloatValue> {
        match self {
            HemisphericWindow::Direct(w) => w.at_end(region),
            HemisphericWindow::FromFourBox(w) => w.at_end(region),
        }
    }

    /// Get the current time value.
    pub fn time(&self) -> Time {
        match self {
            HemisphericWindow::Direct(w) => w.time(),
            HemisphericWindow::FromFourBox(w) => w.time(),
        }
    }

    /// Get the current time index.
    pub fn index(&self) -> usize {
        match self {
            HemisphericWindow::Direct(w) => w.index(),
            HemisphericWindow::FromFourBox(w) => w.index(),
        }
    }

    /// Get the total length of the underlying timeseries.
    pub fn len(&self) -> usize {
        match self {
            HemisphericWindow::Direct(w) => w.len(),
            HemisphericWindow::FromFourBox(w) => w.len(),
        }
    }

    /// Check if the underlying timeseries is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            HemisphericWindow::Direct(w) => w.is_empty(),
            HemisphericWindow::FromFourBox(w) => w.is_empty(),
        }
    }

    /// Get the global aggregate at the start of the timestep.
    pub fn current_global(&self) -> FloatValue {
        let values = self.at_start_all();
        (values[0] + values[1]) / 2.0 // Simple average for now
    }
}

#[cfg(test)]
mod aggregating_window_tests {
    use super::*;
    use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
    use crate::spatial::{FourBoxGrid, HemisphericGrid, ScalarGrid};
    use crate::timeseries::{GridTimeseries, TimeAxis, Timeseries};
    use numpy::array;
    use numpy::ndarray::Array2;
    use std::sync::Arc;

    fn create_four_box_timeseries() -> GridTimeseries<FloatValue, FourBoxGrid> {
        let grid = FourBoxGrid::magicc_standard();
        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0, 2002.0]));
        // Values chosen for easy arithmetic: each timestep increases by 1
        let values = Array2::from_shape_vec(
            (3, 4),
            vec![
                10.0, 20.0, 30.0, 40.0, // 2000: mean = 25.0
                11.0, 21.0, 31.0, 41.0, // 2001: mean = 26.0
                12.0, 22.0, 32.0, 42.0, // 2002: mean = 27.0
            ],
        )
        .unwrap();

        GridTimeseries::new(
            values,
            time_axis,
            grid,
            "W/m^2".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        )
    }

    fn create_hemispheric_timeseries() -> GridTimeseries<FloatValue, HemisphericGrid> {
        let grid = HemisphericGrid::equal_weights();
        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0, 2002.0]));
        let values = Array2::from_shape_vec(
            (3, 2),
            vec![
                100.0, 200.0, // 2000: mean = 150.0
                110.0, 220.0, // 2001: mean = 165.0
                120.0, 240.0, // 2002: mean = 180.0
            ],
        )
        .unwrap();

        GridTimeseries::new(
            values,
            time_axis,
            grid,
            "W/m^2".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        )
    }

    // =========================================================================
    // AggregatingFourBoxWindow tests (FourBox -> Scalar aggregation)
    // =========================================================================

    #[test]
    fn test_aggregating_four_box_window_at_start_default_weights() {
        let ts = create_four_box_timeseries();
        // Index 1 = year 2001, values [11, 21, 31, 41]
        let window = AggregatingFourBoxWindow::new(&ts, 1, 2001.0, None);

        // With equal weights: (11 + 21 + 31 + 41) / 4 = 26.0
        assert_eq!(window.at_start(), 26.0);
    }

    #[test]
    fn test_aggregating_four_box_window_at_start_custom_weights() {
        let ts = create_four_box_timeseries();
        // Custom weights that sum to 1.0
        let weights = vec![0.5, 0.2, 0.2, 0.1];
        let window = AggregatingFourBoxWindow::new(&ts, 1, 2001.0, Some(weights));

        // With custom weights: 11*0.5 + 21*0.2 + 31*0.2 + 41*0.1 = 5.5 + 4.2 + 6.2 + 4.1 = 20.0
        assert_eq!(window.at_start(), 20.0);
    }

    #[test]
    fn test_aggregating_four_box_window_at_end() {
        let ts = create_four_box_timeseries();
        // Index 1, at_end should return value at index 2
        let window = AggregatingFourBoxWindow::new(&ts, 1, 2001.0, None);

        // Index 2 values [12, 22, 32, 42], mean = 27.0
        assert_eq!(window.at_end(), Some(27.0));
    }

    #[test]
    fn test_aggregating_four_box_window_at_end_last_index() {
        let ts = create_four_box_timeseries();
        // At last index, at_end should return None
        let window = AggregatingFourBoxWindow::new(&ts, 2, 2002.0, None);

        assert_eq!(window.at_end(), None);
    }

    #[test]
    fn test_aggregating_four_box_window_previous() {
        let ts = create_four_box_timeseries();
        let window = AggregatingFourBoxWindow::new(&ts, 1, 2001.0, None);

        // Index 0 values [10, 20, 30, 40], mean = 25.0
        assert_eq!(window.previous(), Some(25.0));
    }

    #[test]
    fn test_aggregating_four_box_window_previous_at_first() {
        let ts = create_four_box_timeseries();
        let window = AggregatingFourBoxWindow::new(&ts, 0, 2000.0, None);

        assert_eq!(window.previous(), None);
    }

    #[test]
    fn test_aggregating_four_box_window_metadata() {
        let ts = create_four_box_timeseries();
        let window = AggregatingFourBoxWindow::new(&ts, 1, 2001.0, None);

        assert_eq!(window.time(), 2001.0);
        assert_eq!(window.index(), 1);
        assert_eq!(window.len(), 3);
        assert!(!window.is_empty());
    }

    #[test]
    fn test_aggregating_four_box_window_nan_handling() {
        // Test that NaN values are excluded from aggregation
        let grid = FourBoxGrid::magicc_standard();
        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
        let values = Array2::from_shape_vec(
            (2, 4),
            vec![
                10.0,
                f64::NAN,
                30.0,
                40.0, // 2000: has NaN
                20.0,
                20.0,
                20.0,
                20.0, // 2001: all valid
            ],
        )
        .unwrap();
        let ts = GridTimeseries::new(
            values,
            time_axis,
            grid,
            "test".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );

        let weights = vec![0.25, 0.25, 0.25, 0.25];
        let window = AggregatingFourBoxWindow::new(&ts, 0, 2000.0, Some(weights));

        // NaN is skipped: 10*0.25 + 30*0.25 + 40*0.25 = 2.5 + 7.5 + 10.0 = 20.0
        assert_eq!(window.at_start(), 20.0);
    }

    // =========================================================================
    // AggregatingHemisphericWindow tests (Hemispheric -> Scalar aggregation)
    // =========================================================================

    #[test]
    fn test_aggregating_hemispheric_window_at_start_default_weights() {
        let ts = create_hemispheric_timeseries();
        let window = AggregatingHemisphericWindow::new(&ts, 1, 2001.0, None);

        // Equal weights: (110 + 220) / 2 = 165.0
        assert_eq!(window.at_start(), 165.0);
    }

    #[test]
    fn test_aggregating_hemispheric_window_at_start_custom_weights() {
        let ts = create_hemispheric_timeseries();
        let weights = vec![0.7, 0.3];
        let window = AggregatingHemisphericWindow::new(&ts, 1, 2001.0, Some(weights));

        // 110*0.7 + 220*0.3 = 77.0 + 66.0 = 143.0
        assert_eq!(window.at_start(), 143.0);
    }

    #[test]
    fn test_aggregating_hemispheric_window_at_end() {
        let ts = create_hemispheric_timeseries();
        let window = AggregatingHemisphericWindow::new(&ts, 1, 2001.0, None);

        // Index 2 values [120, 240], mean = 180.0
        assert_eq!(window.at_end(), Some(180.0));
    }

    #[test]
    fn test_aggregating_hemispheric_window_at_end_last_index() {
        let ts = create_hemispheric_timeseries();
        let window = AggregatingHemisphericWindow::new(&ts, 2, 2002.0, None);

        assert_eq!(window.at_end(), None);
    }

    #[test]
    fn test_aggregating_hemispheric_window_previous() {
        let ts = create_hemispheric_timeseries();
        let window = AggregatingHemisphericWindow::new(&ts, 1, 2001.0, None);

        // Index 0 values [100, 200], mean = 150.0
        assert_eq!(window.previous(), Some(150.0));
    }

    #[test]
    fn test_aggregating_hemispheric_window_metadata() {
        let ts = create_hemispheric_timeseries();
        let window = AggregatingHemisphericWindow::new(&ts, 1, 2001.0, None);

        assert_eq!(window.time(), 2001.0);
        assert_eq!(window.index(), 1);
        assert_eq!(window.len(), 3);
        assert!(!window.is_empty());
    }

    // =========================================================================
    // AggregatingFourBoxToHemisphericWindow tests (FourBox -> Hemispheric)
    // =========================================================================

    #[test]
    fn test_aggregating_four_box_to_hemispheric_at_start_all() {
        let ts = create_four_box_timeseries();
        let window = AggregatingFourBoxToHemisphericWindow::new(&ts, 1, 2001.0);

        // Index 1 values [11, 21, 31, 41]
        // Northern = (11 + 21) / 2 = 16.0
        // Southern = (31 + 41) / 2 = 36.0
        let result = window.at_start_all();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 16.0); // Northern
        assert_eq!(result[1], 36.0); // Southern
    }

    #[test]
    fn test_aggregating_four_box_to_hemispheric_at_start_single_region() {
        let ts = create_four_box_timeseries();
        let window = AggregatingFourBoxToHemisphericWindow::new(&ts, 1, 2001.0);

        assert_eq!(window.at_start(HemisphericRegion::Northern), 16.0);
        assert_eq!(window.at_start(HemisphericRegion::Southern), 36.0);
    }

    #[test]
    fn test_aggregating_four_box_to_hemispheric_at_end_all() {
        let ts = create_four_box_timeseries();
        let window = AggregatingFourBoxToHemisphericWindow::new(&ts, 1, 2001.0);

        // Index 2 values [12, 22, 32, 42]
        // Northern = (12 + 22) / 2 = 17.0
        // Southern = (32 + 42) / 2 = 37.0
        let result = window.at_end_all().unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 17.0);
        assert_eq!(result[1], 37.0);
    }

    #[test]
    fn test_aggregating_four_box_to_hemispheric_at_end_last_index() {
        let ts = create_four_box_timeseries();
        let window = AggregatingFourBoxToHemisphericWindow::new(&ts, 2, 2002.0);

        assert_eq!(window.at_end_all(), None);
        assert_eq!(window.at_end(HemisphericRegion::Northern), None);
    }

    #[test]
    fn test_aggregating_four_box_to_hemispheric_metadata() {
        let ts = create_four_box_timeseries();
        let window = AggregatingFourBoxToHemisphericWindow::new(&ts, 1, 2001.0);

        assert_eq!(window.time(), 2001.0);
        assert_eq!(window.index(), 1);
        assert_eq!(window.len(), 3);
        assert!(!window.is_empty());
    }

    // =========================================================================
    // ScalarWindow enum tests (unified scalar interface)
    // =========================================================================

    #[test]
    fn test_scalar_window_direct_variant() {
        use numpy::ndarray::Axis;

        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0, 2002.0]));
        let values = array![100.0, 200.0, 300.0].insert_axis(Axis(1));
        let ts = Timeseries::new(
            values,
            time_axis,
            ScalarGrid,
            "test".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );
        let inner = super::super::windows::TimeseriesWindow::new(&ts, 1, 2001.0);
        let window = ScalarWindow::Direct(inner);

        assert_eq!(window.at_start(), 200.0);
        assert_eq!(window.at_end(), Some(300.0));
        assert_eq!(window.previous(), Some(100.0));
        assert_eq!(window.time(), 2001.0);
        assert_eq!(window.index(), 1);
        assert_eq!(window.len(), 3);
        assert!(!window.is_empty());
    }

    #[test]
    fn test_scalar_window_from_four_box_variant() {
        let ts = create_four_box_timeseries();
        let inner = AggregatingFourBoxWindow::new(&ts, 1, 2001.0, None);
        let window = ScalarWindow::FromFourBox(inner);

        // Same behavior as AggregatingFourBoxWindow
        assert_eq!(window.at_start(), 26.0);
        assert_eq!(window.at_end(), Some(27.0));
        assert_eq!(window.previous(), Some(25.0));
        assert_eq!(window.time(), 2001.0);
        assert_eq!(window.index(), 1);
        assert_eq!(window.len(), 3);
    }

    #[test]
    fn test_scalar_window_from_hemispheric_variant() {
        let ts = create_hemispheric_timeseries();
        let inner = AggregatingHemisphericWindow::new(&ts, 1, 2001.0, None);
        let window = ScalarWindow::FromHemispheric(inner);

        // Same behavior as AggregatingHemisphericWindow
        assert_eq!(window.at_start(), 165.0);
        assert_eq!(window.at_end(), Some(180.0));
        assert_eq!(window.previous(), Some(150.0));
        assert_eq!(window.time(), 2001.0);
    }

    // =========================================================================
    // HemisphericWindow enum tests (unified hemispheric interface)
    // =========================================================================

    #[test]
    fn test_hemispheric_window_direct_variant() {
        let ts = create_hemispheric_timeseries();
        let inner = GridTimeseriesWindow::new(&ts, 1, 2001.0);
        let window = HemisphericWindow::Direct(inner);

        assert_eq!(window.at_start(HemisphericRegion::Northern), 110.0);
        assert_eq!(window.at_start(HemisphericRegion::Southern), 220.0);
        assert_eq!(window.at_start_all(), vec![110.0, 220.0]);
        assert_eq!(window.at_end_all(), Some(vec![120.0, 240.0]));
        assert_eq!(window.time(), 2001.0);
        assert_eq!(window.index(), 1);
        assert_eq!(window.len(), 3);
        assert!(!window.is_empty());
    }

    #[test]
    fn test_hemispheric_window_from_four_box_variant() {
        let ts = create_four_box_timeseries();
        let inner = AggregatingFourBoxToHemisphericWindow::new(&ts, 1, 2001.0);
        let window = HemisphericWindow::FromFourBox(inner);

        // Aggregated values: Northern=(11+21)/2=16, Southern=(31+41)/2=36
        assert_eq!(window.at_start(HemisphericRegion::Northern), 16.0);
        assert_eq!(window.at_start(HemisphericRegion::Southern), 36.0);
        assert_eq!(window.at_start_all(), vec![16.0, 36.0]);
        // Index 2: Northern=(12+22)/2=17, Southern=(32+42)/2=37
        assert_eq!(window.at_end_all(), Some(vec![17.0, 37.0]));
        assert_eq!(window.time(), 2001.0);
        assert_eq!(window.index(), 1);
    }
}
