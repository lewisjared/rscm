use crate::errors::RSCMResult;
use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
use crate::interpolate::Interp1d;
use crate::spatial::SpatialGrid;
use crate::state::{FourBoxSlice, HemisphericSlice};
use num::Float;
use numpy::ndarray::prelude::*;
use numpy::ndarray::{Array, Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::iter::zip;
use std::sync::Arc;

/// The type of float used in time calculations
///
/// Currently, this should be the same as ['FloatValue'] and anything else is untested.
pub type Time = f64;

/// Type of float to use in timeseries and calculations within rscm-core.
///
/// This is a placeholder to make it easier to be able to use a generic representation of value.
pub type FloatValue = f64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeAxis {
    bounds: Array1<Time>,
}

fn check_monotonic_increasing(arr: &Array1<Time>) -> bool {
    let mut zipped_arr = zip(arr.slice(s![0..arr.len() - 1]), arr.slice(s![1..]));

    // Check that [i + 1] > [i]
    zipped_arr.all(|(&a, &b)| b > a)
}

/// Axis for a time series
///
/// The time values must be monotonically increasing with
/// contiguous bounds (i.e. there cannot be any gaps).
///
/// The convention used here is that the value represents the start of a time step.
/// Each time step has a half-open bound that denotes the time period over which that step is
/// calculated.
///
/// Generally, decimal year values are used throughout
impl TimeAxis {
    fn new(bounds: Array1<Time>) -> Self {
        let is_monotonic = check_monotonic_increasing(&bounds);
        assert!(is_monotonic);

        Self { bounds }
    }

    /// Initialise using values
    ///
    /// Assumes that the size of the last time step is equal to the size of the previous time step
    ///
    /// # Example
    ///
    /// ```rust
    /// use numpy::array;
    /// use rscm_core::timeseries::{Time, TimeAxis};
    /// let ta = TimeAxis::from_values(array![1.0, 2.0, 3.0]);
    /// let expected: (Time, Time) = (3.0, 4.0);
    /// assert_eq!(ta.at_bounds(2).unwrap(), expected);
    /// ```
    pub fn from_values(values: Array1<Time>) -> Self {
        assert!(values.len() >= 2);
        let step = values[values.len() - 1] - values[values.len() - 2];

        let mut bounds = Array::zeros(values.len() + 1);
        bounds.slice_mut(s![..values.len()]).assign(&values);

        let last_idx = bounds.len() - 1;

        bounds[last_idx] = bounds[last_idx - 1] + step;
        Self::new(bounds)
    }

    /// Initialise using bounds
    ///
    /// # Example
    ///
    /// ```rust
    /// use numpy::array;
    /// use rscm_core::timeseries::TimeAxis;
    /// let ta = TimeAxis::from_bounds(array![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(ta.len(), 3);
    /// ```
    pub fn from_bounds(bounds: Array1<Time>) -> Self {
        assert!(bounds.len() > 1);

        Self::new(bounds)
    }

    pub fn values(&self) -> ArrayView1<Time> {
        self.bounds.slice(s![0..self.len()])
    }

    pub fn bounds(&self) -> ArrayView1<Time> {
        self.bounds.view()
    }

    /// Get the last time value
    pub fn len(&self) -> usize {
        self.bounds.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// Get the number of bounds
    ///
    /// This is always 1 larger than the number of values
    pub fn len_bounds(&self) -> usize {
        self.bounds.len()
    }

    /// Get the first time value
    // TODO: Investigate Time vs &Time
    pub fn first(&self) -> &Time {
        self.bounds.first().unwrap()
    }

    /// Get the last time value
    pub fn last(&self) -> &Time {
        self.bounds.get(self.len()).unwrap()
    }

    /// Get the time value for a step
    ///
    /// # Example
    ///
    /// ```rust
    /// use numpy::array;
    /// use rscm_core::timeseries::TimeAxis;
    /// let ta = TimeAxis::from_values(array![1.0, 2.0, 3.0]);
    /// assert_eq!(ta.at(1).unwrap(), 2.0);
    /// assert_eq!(ta.at(27), None);
    /// ```
    pub fn at(&self, index: usize) -> Option<Time> {
        if index < self.len() {
            Option::from(self.bounds[index])
        } else {
            None
        }
    }

    /// Get the bounds for a given index
    pub fn at_bounds(&self, index: usize) -> Option<(Time, Time)> {
        if index < self.len() {
            let bound: (Time, Time) = (self.bounds[index], self.bounds[index + 1]);
            Option::from(bound)
        } else {
            None
        }
    }

    pub fn get_index(&self, time: Time) -> usize {
        self.bounds
            .as_slice()
            .unwrap()
            // Have to use binary_search_by as
            .binary_search_by(|v| v.partial_cmp(&time).expect("Couldn't compare values"))
            .unwrap()
    }

    /// Check if the axis contains a given value
    ///
    /// # Example
    ///
    /// ```rust
    /// use numpy::array;
    /// use rscm_core::timeseries::TimeAxis;
    /// let ta = TimeAxis::from_values(array![1.0, 2.0, 3.0]);
    /// assert!(ta.contains(1.0));
    /// assert!(!ta.contains(27.0));
    /// ```
    pub fn contains(&self, value: Time) -> bool {
        let mut found = false;

        for v in self.values().iter() {
            if value == *v {
                found = true;
                break;
            }
        }
        found
    }
}

/// A spatially-resolved timeseries with values on a grid
///
/// `GridTimeseries` extends the concept of a timeseries to include spatial dimensions.
/// It stores values as a 2D array with shape `(time, space)`, where the spatial
/// dimension is defined by a [`SpatialGrid`](crate::spatial::SpatialGrid).
///
/// # Type Parameters
///
/// * `T` - The floating-point type for values (typically `FloatValue`)
/// * `G` - The grid type that defines the spatial structure (e.g., `FourBoxGrid`)
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
/// use numpy::array;
/// use numpy::ndarray::{Array, Array2};
/// use rscm_core::spatial::{FourBoxGrid, SpatialGrid};
/// use rscm_core::timeseries::{FloatValue, TimeAxis, GridTimeseries};
/// use rscm_core::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
///
/// // Create a four-box grid timeseries
/// let grid = FourBoxGrid::magicc_standard();
/// let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0, 2002.0]));
///
/// // Values: shape (3 time steps, 4 regions)
/// let values = Array2::from_shape_vec(
///     (3, 4),
///     vec![
///         15.0, 14.0, 10.0, 9.0,  // Year 2000
///         15.5, 14.5, 10.5, 9.5,  // Year 2001
///         16.0, 15.0, 11.0, 10.0, // Year 2002
///     ]
/// ).unwrap();
///
/// let ts = GridTimeseries::new(
///     values,
///     time_axis,
///     grid,
///     "degC".to_string(),
///     InterpolationStrategy::from(LinearSplineStrategy::new(true)),
/// );
///
/// assert_eq!(ts.len(), 3);
/// assert_eq!(ts.grid().size(), 4);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GridTimeseries<T, G>
where
    T: Float,
    G: SpatialGrid,
{
    #[serde(default, skip)]
    grid: G,
    /// Values stored as Array2<T>: shape (time, space)
    values: Array2<T>,
    time_axis: Arc<TimeAxis>,
    units: String,
    /// Latest timestep with valid (non-NaN) values
    latest: usize,
    interpolation_strategy: InterpolationStrategy,
}

impl<T, G> GridTimeseries<T, G>
where
    T: Float + From<Time>,
    G: SpatialGrid,
{
    /// Create a new grid timeseries from 2D array
    ///
    /// # Arguments
    ///
    /// * `values` - 2D array with shape (time, space)
    /// * `time_axis` - Time axis defining the temporal dimension
    /// * `grid` - Spatial grid defining the spatial dimension
    /// * `units` - Physical units of the values
    /// * `interpolation_strategy` - Strategy for temporal interpolation
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Time dimension of `values` doesn't match `time_axis.len()`
    /// - Space dimension of `values` doesn't match `grid.size()`
    pub fn new(
        values: Array2<T>,
        time_axis: Arc<TimeAxis>,
        grid: G,
        units: String,
        interpolation_strategy: InterpolationStrategy,
    ) -> Self {
        assert_eq!(
            values.nrows(),
            time_axis.len(),
            "Time dimension must match time axis length"
        );
        assert_eq!(
            values.ncols(),
            grid.size(),
            "Space dimension must match grid size"
        );

        // Find latest valid timestep (all regions must be non-NaN)
        let mut latest = 0;
        for (t, row) in values.rows().into_iter().enumerate() {
            if row.iter().all(|v| !v.is_nan()) {
                latest = t;
            }
        }

        Self {
            grid,
            values,
            time_axis,
            units,
            latest,
            interpolation_strategy,
        }
    }

    /// Create an empty grid timeseries filled with NaN
    pub fn new_empty(
        time_axis: Arc<TimeAxis>,
        grid: G,
        units: String,
        interpolation_strategy: InterpolationStrategy,
    ) -> Self {
        let mut values = Array2::zeros((time_axis.len(), grid.size()));
        values.fill(T::nan());

        Self::new(values, time_axis, grid, units, interpolation_strategy)
    }

    /// Get the number of timesteps
    pub fn len(&self) -> usize {
        self.values.nrows()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the spatial grid
    pub fn grid(&self) -> &G {
        &self.grid
    }

    /// Get the time axis
    pub fn time_axis(&self) -> Arc<TimeAxis> {
        self.time_axis.clone()
    }

    /// Get the units
    pub fn units(&self) -> &str {
        &self.units
    }

    /// Get the index of the latest valid timestep
    pub fn latest(&self) -> usize {
        self.latest
    }

    /// Get a value at a specific time index and region index
    ///
    /// Returns `None` if indices are out of bounds
    pub fn at_index(&self, time_index: usize, region_index: usize) -> Option<T> {
        self.values.get((time_index, region_index)).copied()
    }

    /// Set a value at a specific time index and region index
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds
    pub fn set_index(&mut self, time_index: usize, region_index: usize, value: T) {
        self.values[(time_index, region_index)] = value;

        // Update latest if all regions at this timestep are now valid
        if time_index >= self.latest && !value.is_nan() {
            let row = self.values.row(time_index);
            if row.iter().all(|v| !v.is_nan()) {
                self.latest = time_index;
            }
        }
    }

    /// Set all regional values at a specific time index
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `time_index` is out of bounds
    /// - `values` length doesn't match grid size
    pub fn set_all(&mut self, time_index: usize, values: &[T]) {
        assert_eq!(
            values.len(),
            self.grid.size(),
            "Values length ({}) must match grid size ({})",
            values.len(),
            self.grid.size()
        );

        for (region_index, &value) in values.iter().enumerate() {
            self.values[(time_index, region_index)] = value;
        }

        // Update latest if all regions are now valid
        if time_index >= self.latest {
            let row = self.values.row(time_index);
            if row.iter().all(|v| !v.is_nan()) {
                self.latest = time_index;
            }
        }
    }

    /// Get all regional values at a specific time index
    ///
    /// Returns `None` if time index is out of bounds
    pub fn at_time_index(&self, time_index: usize) -> Option<Vec<T>> {
        if time_index < self.len() {
            Some(self.values.row(time_index).to_vec())
        } else {
            None
        }
    }

    /// Get all regional values at the latest valid timestep
    pub fn latest_values(&self) -> Vec<T> {
        self.values.row(self.latest).to_vec()
    }

    /// Interpolate all regional values at a specific time
    ///
    /// Uses the interpolation strategy to compute values for all regions
    /// at the given time point.
    pub fn at_time_all(&self, time: Time) -> RSCMResult<Vec<T>> {
        let mut result = Vec::with_capacity(self.grid.size());

        for region_idx in 0..self.grid.size() {
            let region_values = self.values.column(region_idx);
            let interp = Interp1d::new(
                self.time_axis.values(),
                region_values.view(),
                self.interpolation_strategy.clone(),
            );
            result.push(interp.interpolate(time)?);
        }

        Ok(result)
    }

    /// Aggregate all regions to a single global value at the latest timestep
    pub fn latest_global(&self) -> T {
        let values_f64: Vec<FloatValue> = self
            .latest_values()
            .iter()
            .map(|v| v.to_f64().unwrap())
            .collect();
        <T as From<Time>>::from(self.grid.aggregate_global(&values_f64))
    }

    /// Aggregate all regions to a single global timeseries
    ///
    /// Returns a scalar grid timeseries by aggregating all regions at each timestep
    pub fn aggregate_global(&self) -> GridTimeseries<T, crate::spatial::ScalarGrid> {
        use crate::spatial::ScalarGrid;

        let mut global_values = Array2::zeros((self.len(), 1));

        for (t, row) in self.values.rows().into_iter().enumerate() {
            let values_f64: Vec<FloatValue> = row.iter().map(|v| v.to_f64().unwrap()).collect();
            global_values[(t, 0)] =
                <T as From<Time>>::from(self.grid.aggregate_global(&values_f64));
        }

        GridTimeseries::new(
            global_values,
            self.time_axis.clone(),
            ScalarGrid,
            self.units.clone(),
            self.interpolation_strategy.clone(),
        )
    }

    /// Transform this grid timeseries to a different grid type
    ///
    /// # Errors
    ///
    /// Returns an error if the transformation is not supported
    pub fn transform_to<G2: SpatialGrid>(
        &self,
        target_grid: G2,
    ) -> RSCMResult<GridTimeseries<T, G2>> {
        let target_size = target_grid.size();
        let mut transformed_values = Array2::zeros((self.len(), target_size));

        for (t, row) in self.values.rows().into_iter().enumerate() {
            let values_f64: Vec<FloatValue> = row.iter().map(|v| v.to_f64().unwrap()).collect();
            let transformed = self.grid.transform_to(&values_f64, &target_grid)?;

            for (r, &val) in transformed.iter().enumerate() {
                transformed_values[(t, r)] = <T as From<Time>>::from(val);
            }
        }

        Ok(GridTimeseries::new(
            transformed_values,
            self.time_axis.clone(),
            target_grid,
            self.units.clone(),
            self.interpolation_strategy.clone(),
        ))
    }

    /// Extract a single region as a scalar timeseries
    ///
    /// # Panics
    ///
    /// Panics if region_index is out of bounds
    pub fn region(&self, region_index: usize) -> GridTimeseries<T, crate::spatial::ScalarGrid> {
        use crate::spatial::ScalarGrid;

        assert!(
            region_index < self.grid.size(),
            "Region index out of bounds"
        );

        let mut region_values = Array2::zeros((self.len(), 1));
        for t in 0..self.len() {
            region_values[(t, 0)] = self.values[(t, region_index)];
        }

        GridTimeseries::new(
            region_values,
            self.time_axis.clone(),
            ScalarGrid,
            self.units.clone(),
            self.interpolation_strategy.clone(),
        )
    }

    /// Extract a single region by name
    ///
    /// Returns `None` if no region with that name exists
    pub fn region_by_name(
        &self,
        name: &str,
    ) -> Option<GridTimeseries<T, crate::spatial::ScalarGrid>> {
        self.grid
            .region_names()
            .iter()
            .position(|n| n == name)
            .map(|idx| self.region(idx))
    }

    /// Get the 2D values array
    pub fn values(&self) -> &Array2<T> {
        &self.values
    }

    /// Replace the interpolation strategy
    pub fn with_interpolation_strategy(
        &mut self,
        interpolation_strategy: InterpolationStrategy,
    ) -> &Self {
        self.interpolation_strategy = interpolation_strategy;
        self
    }

    /// Interpolate onto a new time axis
    ///
    /// Creates a new grid timeseries with the same spatial grid but different time axis.
    /// All regions are interpolated independently using the current interpolation strategy.
    pub fn interpolate_into(self, new_time_axis: Arc<TimeAxis>) -> Self {
        let mut new_values = Array2::zeros((new_time_axis.len(), self.grid.size()));

        for region_idx in 0..self.grid.size() {
            let region_values = self.values.column(region_idx);
            let interp = Interp1d::new(
                self.time_axis.values(),
                region_values.view(),
                self.interpolation_strategy.clone(),
            );

            for (t, &time) in new_time_axis.values().iter().enumerate() {
                new_values[(t, region_idx)] = interp.interpolate(time).unwrap();
            }
        }

        Self::new(
            new_values,
            new_time_axis,
            self.grid,
            self.units,
            self.interpolation_strategy,
        )
    }
}

// Convenience methods for scalar grid timeseries
impl<T> GridTimeseries<T, crate::spatial::ScalarGrid>
where
    T: Float + From<Time>,
{
    /// Create a new scalar timeseries from a 1D array and a time axis
    ///
    /// The interpolation strategy for the timeseries defaults to linear with extrapolation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use numpy::array;
    /// use numpy::ndarray::Array;
    /// use rscm_core::timeseries::{FloatValue, TimeAxis, Timeseries};
    /// use rscm_core::spatial::{ScalarRegion};
    ///
    /// let timeseries: Timeseries<FloatValue> = Timeseries::from_values(array![1.0, 2.0, 3.0, 4.0, 5.0], Array::range(2000.0, 2050.0, 10.0));
    ///
    /// assert_eq!(timeseries.len(), 5);
    /// assert_eq!(timeseries.latest_value().unwrap(), 5.0);
    /// assert_eq!(timeseries.at_scalar(0).unwrap(), 1.0);
    /// assert_eq!(timeseries.at_time(2040.0, ScalarRegion::Global).unwrap(), 5.0);
    /// ```
    pub fn from_values(values: Array1<T>, time: Array1<Time>) -> Self {
        use crate::spatial::ScalarGrid;

        // Convert 1D array to 2D with shape (n, 1)
        let values_2d = values.insert_axis(Axis(1));

        Self::new(
            values_2d,
            Arc::new(TimeAxis::from_values(time)),
            ScalarGrid,
            "".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        )
    }

    /// Create an empty scalar timeseries (backwards compatibility method)
    ///
    /// # Arguments
    ///
    /// * `time_axis` - Time axis defining the temporal dimension
    /// * `units` - Physical units of the values
    /// * `interpolation_strategy` - Strategy for temporal interpolation
    pub fn new_empty_scalar(
        time_axis: Arc<TimeAxis>,
        units: String,
        interpolation_strategy: InterpolationStrategy,
    ) -> Self {
        use crate::spatial::ScalarGrid;
        Self::new_empty(time_axis, ScalarGrid, units, interpolation_strategy)
    }

    /// Get the scalar value at a given time index
    ///
    /// # Examples
    /// ```rust
    /// use numpy::array;
    /// use numpy::ndarray::Array;
    /// use rscm_core::timeseries::{Timeseries};
    ///
    /// let timeseries = Timeseries::from_values(array![1.0, 2.0, 3.0, 4.0, 5.0], Array::range(2000.0, 2050.0, 10.0));
    ///
    /// assert_eq!(timeseries.len(), 5);
    /// assert_eq!(timeseries.at_scalar(0).unwrap(), 1.0);
    /// assert_eq!(timeseries.at_scalar(1).unwrap(), 2.0);
    /// assert!(timeseries.at_scalar(12).is_none());
    /// ```
    pub fn at_scalar(&self, index: usize) -> Option<T> {
        self.at(index, crate::spatial::ScalarRegion::Global)
    }

    /// Set the scalar value at a given time index
    pub fn set_scalar(&mut self, time_index: usize, value: T) {
        self.set(time_index, crate::spatial::ScalarRegion::Global, value);
    }

    /// Get the index of the latest valid timestep (backwards compatibility)
    ///
    /// Returns a reference for backwards compatibility
    pub fn latest_ref(&self) -> &usize {
        &self.latest
    }

    /// Get the latest scalar value
    pub fn latest_value(&self) -> Option<T> {
        self.at_scalar(self.latest)
    }
}

// Type-safe accessor methods for ScalarGrid
impl<T> GridTimeseries<T, crate::spatial::ScalarGrid>
where
    T: Float + From<Time>,
{
    /// Get the value at a given time index (type-safe for ScalarGrid)
    ///
    /// # Examples
    /// ```rust
    /// use numpy::array;
    /// use numpy::ndarray::Array;
    /// use rscm_core::timeseries::Timeseries;
    /// use rscm_core::spatial::ScalarRegion;
    ///
    /// let timeseries = Timeseries::from_values(array![1.0, 2.0, 3.0, 4.0, 5.0], Array::range(2000.0, 2050.0, 10.0));
    ///
    /// assert_eq!(timeseries.at(0, ScalarRegion::Global).unwrap(), 1.0);
    /// assert_eq!(timeseries.at(1, ScalarRegion::Global).unwrap(), 2.0);
    /// ```
    pub fn at(&self, time_index: usize, region: crate::spatial::ScalarRegion) -> Option<T> {
        GridTimeseries::at_index(self, time_index, region as usize)
    }

    /// Set the value at a given time index (type-safe for ScalarGrid)
    pub fn set(&mut self, time_index: usize, region: crate::spatial::ScalarRegion, value: T) {
        GridTimeseries::set_index(self, time_index, region as usize, value);
    }

    /// Get the value at a given time with interpolation (type-safe for ScalarGrid)
    ///
    /// This method interpolates using the current interpolation strategy to determine
    /// the value at `time`.
    pub fn at_time(&self, time: Time, region: crate::spatial::ScalarRegion) -> RSCMResult<T> {
        let result = self.at_time_all(time)?;
        Ok(result[region as usize])
    }
}

// Type-safe accessor methods for FourBoxGrid
impl<T> GridTimeseries<T, crate::spatial::FourBoxGrid>
where
    T: Float + From<Time>,
{
    /// Get the value at a given time index and region (type-safe for FourBoxGrid)
    ///
    /// # Examples
    /// ```rust
    /// use std::sync::Arc;
    /// use numpy::array;
    /// use numpy::ndarray::Array2;
    /// use rscm_core::timeseries::GridTimeseries;
    /// use rscm_core::spatial::{FourBoxGrid, FourBoxRegion};
    /// use rscm_core::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
    /// use rscm_core::timeseries::TimeAxis;
    ///
    /// let grid = FourBoxGrid::magicc_standard();
    /// let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
    /// let values = Array2::from_shape_vec((2, 4), vec![15.0, 14.0, 10.0, 9.0, 16.0, 15.0, 11.0, 10.0]).unwrap();
    ///
    /// let ts = GridTimeseries::new(
    ///     values,
    ///     time_axis,
    ///     grid,
    ///     "degC".to_string(),
    ///     InterpolationStrategy::from(LinearSplineStrategy::new(true)),
    /// );
    ///
    /// assert_eq!(ts.at(0, FourBoxRegion::NorthernOcean).unwrap(), 15.0);
    /// assert_eq!(ts.at(0, FourBoxRegion::NorthernLand).unwrap(), 14.0);
    /// ```
    pub fn at(&self, time_index: usize, region: crate::spatial::FourBoxRegion) -> Option<T> {
        GridTimeseries::at_index(self, time_index, region as usize)
    }

    /// Set the value at a given time index and region (type-safe for FourBoxGrid)
    pub fn set(&mut self, time_index: usize, region: crate::spatial::FourBoxRegion, value: T) {
        GridTimeseries::set_index(self, time_index, region as usize, value);
    }

    /// Set all values at a given time index from a FourBoxSlice (type-safe for FourBoxGrid)
    ///
    /// # Panics
    ///
    /// Panics if `time_index` is out of bounds
    pub fn set_from_slice(&mut self, time_index: usize, slice: &FourBoxSlice) {
        let values: Vec<T> = slice
            .as_array()
            .iter()
            .map(|&v| <T as From<Time>>::from(v))
            .collect();
        self.set_all(time_index, &values);
    }

    /// Get the value at a given time with interpolation (type-safe for FourBoxGrid)
    pub fn at_time(&self, time: Time, region: crate::spatial::FourBoxRegion) -> RSCMResult<T> {
        let result = self.at_time_all(time)?;
        Ok(result[region as usize])
    }
}

// Type-safe accessor methods for HemisphericGrid
impl<T> GridTimeseries<T, crate::spatial::HemisphericGrid>
where
    T: Float + From<Time>,
{
    /// Get the value at a given time index and region (type-safe for HemisphericGrid)
    pub fn at(&self, time_index: usize, region: crate::spatial::HemisphericRegion) -> Option<T> {
        GridTimeseries::at_index(self, time_index, region as usize)
    }

    /// Set the value at a given time index and region (type-safe for HemisphericGrid)
    pub fn set(&mut self, time_index: usize, region: crate::spatial::HemisphericRegion, value: T) {
        GridTimeseries::set_index(self, time_index, region as usize, value);
    }

    /// Set all values at a given time index from a HemisphericSlice (type-safe for HemisphericGrid)
    ///
    /// # Panics
    ///
    /// Panics if `time_index` is out of bounds
    pub fn set_from_slice(&mut self, time_index: usize, slice: &HemisphericSlice) {
        let values: Vec<T> = slice
            .as_array()
            .iter()
            .map(|&v| <T as From<Time>>::from(v))
            .collect();
        self.set_all(time_index, &values);
    }

    /// Get the value at a given time with interpolation (type-safe for HemisphericGrid)
    pub fn at_time(&self, time: Time, region: crate::spatial::HemisphericRegion) -> RSCMResult<T> {
        let result = self.at_time_all(time)?;
        Ok(result[region as usize])
    }
}

/// Type alias for scalar timeseries (backwards compatibility)
///
/// A `Timeseries<T>` is simply a `GridTimeseries<T, ScalarGrid>` with a single region.
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
/// use numpy::array;
/// use numpy::ndarray::Array;
/// use rscm_core::timeseries::{FloatValue, TimeAxis, Timeseries};
/// use rscm_core::spatial::ScalarRegion;
///
/// let timeseries: Timeseries<FloatValue> = Timeseries::from_values(array![1.0, 2.0, 3.0, 4.0, 5.0], Array::range(2000.0, 2050.0, 10.0));
///
/// assert_eq!(timeseries.len(), 5);
/// assert_eq!(timeseries.latest_value().unwrap(), 5.0);
/// assert_eq!(timeseries.at(0, ScalarRegion::Global).unwrap(), 1.0);
/// ```
pub type Timeseries<T> = GridTimeseries<T, crate::spatial::ScalarGrid>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolate::strategies::{InterpolationStrategy, PreviousStrategy};
    use crate::spatial::{FourBoxRegion, HemisphericRegion, ScalarRegion};
    use is_close::is_close;

    #[test]
    #[should_panic]
    fn check_monotonic_values() {
        Timeseries::from_values(array![1.0, 2.0, 3.0], array![2020.0, 1.0, 2021.0,]);
    }

    #[test]
    fn get_value() {
        let mut result = Timeseries::from_values(
            array![1.0, 2.0, 3.0, 4.0, 5.0],
            Array::range(2020.0, 2025.0, 1.0),
        );

        result.with_interpolation_strategy(InterpolationStrategy::from(LinearSplineStrategy::new(
            false,
        )));
        assert_eq!(result.at_time(2020.0, ScalarRegion::Global).unwrap(), 1.0);
        assert_eq!(result.at_time(2020.5, ScalarRegion::Global).unwrap(), 1.5);
        assert_eq!(result.at_time(2021.0, ScalarRegion::Global).unwrap(), 2.0);

        // Linear extrapolate isn't allowed by default
        assert!(result.at_time(2026.0, ScalarRegion::Global).is_err());
    }

    #[test]
    fn custom_interpolator() {
        let data = array![1.0, 1.5, 2.0];
        let years = Array::range(2020.0, 2023.0, 1.0);
        let query = 2024.0;

        let mut timeseries = Timeseries::from_values(data, years);

        // Default to linear interpolation
        let result = timeseries.at_time(query, ScalarRegion::Global).unwrap();
        assert_eq!(result, 3.0);

        // Replace interpolation strategy
        timeseries
            .with_interpolation_strategy(InterpolationStrategy::from(PreviousStrategy::new(true)));
        let result = timeseries.at_time(query, ScalarRegion::Global).unwrap();
        assert_eq!(result, 2.0);
    }

    #[test]
    fn serialise_and_deserialise_json() {
        let data = array![1.0, 1.5, 2.0];
        let years = Array::range(2020.0, 2023.0, 1.0);

        let timeseries = Timeseries::from_values(data, years);

        let serialised = serde_json::to_string(&timeseries).unwrap();
        assert_eq!(
            serialised,
            r#"{"values":{"v":1,"dim":[3,1],"data":[1.0,1.5,2.0]},"time_axis":{"bounds":{"v":1,"dim":[4],"data":[2020.0,2021.0,2022.0,2023.0]}},"units":"","latest":2,"interpolation_strategy":"Linear"}"#
        );

        let deserialised = serde_json::from_str::<Timeseries<f64>>(&serialised).unwrap();

        assert_eq!(timeseries.values(), deserialised.values());
    }

    #[test]
    #[should_panic]
    fn serialise_and_deserialise_with_nan_json() {
        let data = array![1.0, 1.5, FloatValue::nan()];
        let years = Array::range(2020.0, 2023.0, 1.0);

        let timeseries = Timeseries::from_values(data, years);

        let serialised = serde_json::to_string(&timeseries).unwrap();
        assert_eq!(
            serialised,
            r#"{"units":"","values":{"v":1,"dim":[3],"data":[1.0,1.5,null]},"time_axis":{"bounds":{"v":1,"dim":[4],"data":[2020.0,2021.0,2022.0,2023.0]}},"latest":2,"interpolation_strategy":"Linear"}"#
        );

        // This panics as it can't handle null -> NaN values
        serde_json::from_str::<Timeseries<f64>>(&serialised).unwrap();
    }

    #[test]
    fn serialise_and_deserialise_with_nan_toml() {
        let data = array![1.0, 1.5, FloatValue::nan()];
        let years = Array::range(2020.0, 2023.0, 1.0);

        let timeseries = Timeseries::from_values(data, years);

        let serialised = toml::to_string(&timeseries).unwrap();

        let expected = "units = \"\"
latest = 1
interpolation_strategy = \"Linear\"

[values]
v = 1
dim = [3, 1]
data = [1.0, 1.5, nan]

[time_axis.bounds]
v = 1
dim = [4]
data = [2020.0, 2021.0, 2022.0, 2023.0]
";

        assert_eq!(serialised, expected);

        let deserialised = toml::from_str::<Timeseries<f64>>(&serialised).unwrap();

        assert!(zip(timeseries.values(), deserialised.values())
            .all(|(x0, x1)| { is_close!(*x0, *x1) || (x0.is_nan() && x0.is_nan()) }))
    }

    mod grid_timeseries_tests {
        use super::*;
        use crate::spatial::{FourBoxGrid, HemisphericGrid};

        #[test]
        fn create_grid_timeseries() {
            let grid = FourBoxGrid::magicc_standard();
            let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0, 2002.0]));

            let values = Array2::from_shape_vec(
                (3, 4),
                vec![
                    15.0, 14.0, 10.0, 9.0, // Year 2000
                    15.5, 14.5, 10.5, 9.5, // Year 2001
                    16.0, 15.0, 11.0, 10.0, // Year 2002
                ],
            )
            .unwrap();

            let ts = GridTimeseries::new(
                values,
                time_axis,
                grid,
                "degC".to_string(),
                InterpolationStrategy::from(LinearSplineStrategy::new(true)),
            );

            assert_eq!(ts.len(), 3);
            assert_eq!(ts.grid().size(), 4);
            assert_eq!(ts.latest(), 2);
        }

        #[test]
        fn grid_timeseries_access() {
            let grid = FourBoxGrid::magicc_standard();
            let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));

            let values = Array2::from_shape_vec(
                (2, 4),
                vec![
                    15.0, 14.0, 10.0, 9.0, // Year 2000
                    16.0, 15.0, 11.0, 10.0, // Year 2001
                ],
            )
            .unwrap();

            let ts = GridTimeseries::new(
                values,
                time_axis,
                grid,
                "degC".to_string(),
                InterpolationStrategy::from(LinearSplineStrategy::new(true)),
            );

            // Test point access
            assert_eq!(ts.at(0, FourBoxRegion::NorthernOcean), Some(15.0));
            assert_eq!(ts.at(0, FourBoxRegion::NorthernLand), Some(14.0));
            assert_eq!(ts.at(1, FourBoxRegion::SouthernOcean), Some(11.0));

            // Test row access
            let row0 = ts.at_time_index(0).unwrap();
            assert_eq!(row0, vec![15.0, 14.0, 10.0, 9.0]);

            // Test latest
            let latest = ts.latest_values();
            assert_eq!(latest, vec![16.0, 15.0, 11.0, 10.0]);
        }

        #[test]
        fn grid_timeseries_set() {
            let grid = FourBoxGrid::magicc_standard();
            let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));

            let values = Array2::from_shape_vec(
                (2, 4),
                vec![
                    15.0,
                    14.0,
                    10.0,
                    9.0,
                    FloatValue::nan(),
                    FloatValue::nan(),
                    FloatValue::nan(),
                    FloatValue::nan(),
                ],
            )
            .unwrap();

            let mut ts = GridTimeseries::new(
                values,
                time_axis,
                grid,
                "degC".to_string(),
                InterpolationStrategy::from(LinearSplineStrategy::new(true)),
            );

            assert_eq!(ts.latest(), 0);

            // Set values for second timestep
            ts.set_index(1, 0, 16.0);
            ts.set_index(1, 1, 15.0);
            ts.set_index(1, 2, 11.0);
            ts.set_index(1, 3, 10.0);

            assert_eq!(ts.latest(), 1);
            assert_eq!(ts.at(1, FourBoxRegion::NorthernOcean), Some(16.0));
        }

        #[test]
        fn grid_timeseries_aggregate_global() {
            let grid = FourBoxGrid::magicc_standard();
            let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));

            let values = Array2::from_shape_vec(
                (2, 4),
                vec![
                    15.0, 14.0, 10.0, 9.0, // Avg = 12.0
                    16.0, 15.0, 11.0, 10.0, // Avg = 13.0
                ],
            )
            .unwrap();

            let ts = GridTimeseries::new(
                values,
                time_axis,
                grid,
                "degC".to_string(),
                InterpolationStrategy::from(LinearSplineStrategy::new(true)),
            );

            let global = ts.aggregate_global();
            assert_eq!(global.grid().size(), 1);
            assert_eq!(global.len(), 2);
            assert_eq!(global.at(0, ScalarRegion::Global), Some(12.0));
            assert_eq!(global.at(1, ScalarRegion::Global), Some(13.0));
        }

        #[test]
        fn grid_timeseries_transform_four_box_to_hemispheric() {
            let grid = FourBoxGrid::magicc_standard();
            let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));

            let values =
                Array2::from_shape_vec((2, 4), vec![16.0, 14.0, 12.0, 8.0, 17.0, 15.0, 13.0, 9.0])
                    .unwrap();

            let ts = GridTimeseries::new(
                values,
                time_axis,
                grid,
                "degC".to_string(),
                InterpolationStrategy::from(LinearSplineStrategy::new(true)),
            );

            let hemispheric = ts.transform_to(HemisphericGrid::equal_weights()).unwrap();

            assert_eq!(hemispheric.grid().size(), 2);
            // Northern: (16*0.25 + 14*0.25) / 0.5 = 15.0
            // Southern: (12*0.25 + 8*0.25) / 0.5 = 10.0
            assert_eq!(hemispheric.at(0, HemisphericRegion::Northern), Some(15.0));
            assert_eq!(hemispheric.at(0, HemisphericRegion::Southern), Some(10.0));
        }

        #[test]
        fn grid_timeseries_region_extraction() {
            let grid = FourBoxGrid::magicc_standard();
            let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));

            let values =
                Array2::from_shape_vec((2, 4), vec![15.0, 14.0, 10.0, 9.0, 16.0, 15.0, 11.0, 10.0])
                    .unwrap();

            let ts = GridTimeseries::new(
                values,
                time_axis,
                grid,
                "degC".to_string(),
                InterpolationStrategy::from(LinearSplineStrategy::new(true)),
            );

            let northern_ocean = ts.region(FourBoxRegion::NorthernOcean as usize);
            assert_eq!(northern_ocean.grid().size(), 1);
            assert_eq!(northern_ocean.at(0, ScalarRegion::Global), Some(15.0));
            assert_eq!(northern_ocean.at(1, ScalarRegion::Global), Some(16.0));

            let northern_land = ts.region_by_name("Northern Land").unwrap();
            assert_eq!(northern_land.at(0, ScalarRegion::Global), Some(14.0));
            assert_eq!(northern_land.at(1, ScalarRegion::Global), Some(15.0));
        }

        #[test]
        fn grid_timeseries_interpolation() {
            let grid = FourBoxGrid::magicc_standard();
            let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2002.0]));

            let values = Array2::from_shape_vec(
                (2, 4),
                vec![10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0],
            )
            .unwrap();

            let ts = GridTimeseries::new(
                values,
                time_axis,
                grid,
                "degC".to_string(),
                InterpolationStrategy::from(LinearSplineStrategy::new(true)),
            );

            // Interpolate at midpoint (2001.0)
            let values_2001 = ts.at_time_all(2001.0).unwrap();
            assert_eq!(values_2001.len(), 4);
            for val in values_2001 {
                assert_eq!(val, 15.0); // Linear interpolation between 10 and 20
            }
        }

        #[test]
        fn grid_timeseries_interpolate_into() {
            let grid = FourBoxGrid::magicc_standard();
            let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2002.0]));

            let values = Array2::from_shape_vec(
                (2, 4),
                vec![10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0],
            )
            .unwrap();

            let ts = GridTimeseries::new(
                values,
                time_axis,
                grid,
                "degC".to_string(),
                InterpolationStrategy::from(LinearSplineStrategy::new(true)),
            );

            let new_time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0, 2002.0]));
            let resampled = ts.interpolate_into(new_time_axis);

            assert_eq!(resampled.len(), 3);
            assert_eq!(resampled.at(0, FourBoxRegion::NorthernOcean), Some(10.0));
            assert_eq!(resampled.at(1, FourBoxRegion::NorthernOcean), Some(15.0));
            assert_eq!(resampled.at(2, FourBoxRegion::NorthernOcean), Some(20.0));
        }

        #[test]
        fn grid_timeseries_serialization_json() {
            let grid = FourBoxGrid::magicc_standard();
            let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));

            let values =
                Array2::from_shape_vec((2, 4), vec![15.0, 14.0, 10.0, 9.0, 16.0, 15.0, 11.0, 10.0])
                    .unwrap();

            let ts = GridTimeseries::new(
                values,
                time_axis,
                grid,
                "degC".to_string(),
                InterpolationStrategy::from(LinearSplineStrategy::new(true)),
            );

            let serialized = serde_json::to_string(&ts).unwrap();
            let deserialized: GridTimeseries<FloatValue, FourBoxGrid> =
                serde_json::from_str(&serialized).unwrap();

            assert_eq!(deserialized.len(), ts.len());
            assert_eq!(deserialized.grid().size(), ts.grid().size());
            assert_eq!(
                deserialized.at(0, FourBoxRegion::NorthernOcean),
                ts.at(0, FourBoxRegion::NorthernOcean)
            );
        }
    }
}
