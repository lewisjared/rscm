//! Core window types for accessing timeseries values.
//!
//! This module provides zero-cost views into timeseries data at specific time indices,
//! supporting both scalar and grid-based timeseries.

use crate::errors::RSCMResult;
use crate::spatial::{
    FourBoxGrid, FourBoxRegion, HemisphericGrid, HemisphericRegion, ScalarGrid, ScalarRegion,
    SpatialGrid,
};
use crate::timeseries::{FloatValue, GridTimeseries, Time, Timeseries};
use ndarray::ArrayView1;

/// A zero-cost view into a scalar timeseries at a specific time index.
///
/// `TimeseriesWindow` provides efficient access to current, historical, and interpolated
/// values without copying data. This is the primary way components access their input
/// variables.
///
/// # Timestep Access Semantics
///
/// Components must explicitly choose which timestep index to read from:
/// - [`at_start()`](Self::at_start) - Value at index N (start of timestep). Use for:
///   - State variables (your own previous state)
///   - Exogenous inputs (external forcing data)
/// - [`at_end()`](Self::at_end) - Value at index N+1 (written this timestep). Use for:
///   - Upstream component outputs (values written before your component ran)
///   - Aggregation (combining outputs from multiple components)
///
/// # Examples
///
/// ```ignore
/// fn solve(&self, inputs: MyComponentInputs) -> MyComponentOutputs {
///     // Read exogenous input at start of timestep
///     let emissions = inputs.emissions_co2.at_start();
///
///     // Read previous value for derivative calculation
///     let previous = inputs.emissions_co2.previous();
///     let derivative = (emissions - previous.unwrap_or(emissions)) / dt;
///
///     // Access historical values
///     let last_5 = inputs.emissions_co2.last_n(5);
///     // ...
/// }
/// ```
#[derive(Debug)]
pub struct TimeseriesWindow<'a> {
    timeseries: &'a Timeseries<FloatValue>,
    current_index: usize,
    current_time: Time,
    /// Unit conversion factor applied to all returned values.
    /// Default is 1.0 (no conversion).
    unit_conversion_factor: f64,
}

impl<'a> TimeseriesWindow<'a> {
    /// Create a new TimeseriesWindow from a scalar timeseries.
    ///
    /// # Arguments
    ///
    /// * `timeseries` - The underlying scalar timeseries
    /// * `current_index` - Index of the current timestep
    /// * `current_time` - Time value at the current timestep
    pub fn new(
        timeseries: &'a Timeseries<FloatValue>,
        current_index: usize,
        current_time: Time,
    ) -> Self {
        Self {
            timeseries,
            current_index,
            current_time,
            unit_conversion_factor: 1.0,
        }
    }

    /// Create a new TimeseriesWindow with unit conversion.
    ///
    /// The returned values will be multiplied by the conversion factor.
    /// This is used when the stored data uses different units than the component expects.
    pub fn with_unit_conversion(
        timeseries: &'a Timeseries<FloatValue>,
        current_index: usize,
        current_time: Time,
        unit_conversion_factor: f64,
    ) -> Self {
        Self {
            timeseries,
            current_index,
            current_time,
            unit_conversion_factor,
        }
    }

    /// Get the value at the start of the timestep (index N).
    ///
    /// This is the primary accessor for reading input values. The "start of timestep"
    /// refers to the state before any components have executed during this timestep.
    ///
    /// # When to use `at_start()`
    ///
    /// - **State variables**: Reading your own component's state from the previous solve
    ///   (e.g., temperature at the beginning of the timestep before you update it)
    /// - **Exogenous inputs**: External forcing data that was pre-populated before the
    ///   model run (e.g., emissions scenarios, solar irradiance)
    /// - **Any input where you need the value at index N**
    ///
    /// # Execution order context
    ///
    /// Components execute in dependency order. When your component runs:
    /// - Index N contains values from before this timestep started
    /// - Index N+1 may contain values written by upstream components (use [`at_end()`])
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState) -> RSCMResult<OutputState> {
    ///     let inputs = MyComponentInputs::from_input_state(input_state);
    ///
    ///     // Read state variable (your own previous output)
    ///     let prev_temperature = inputs.temperature.at_start();
    ///
    ///     // Read exogenous forcing
    ///     let emissions = inputs.emissions.at_start();
    ///
    ///     // Compute new state
    ///     let new_temperature = prev_temperature + emissions * self.sensitivity;
    ///     // ...
    /// }
    /// ```
    ///
    /// [`at_end()`]: TimeseriesWindow::at_end
    pub fn at_start(&self) -> FloatValue {
        self.timeseries
            .at(self.current_index, ScalarRegion::Global)
            .expect("Current index out of bounds")
            * self.unit_conversion_factor
    }

    /// Get the value at the end of the timestep (index N+1), if available.
    ///
    /// This accessor reads values that were written during the current timestep by
    /// components that executed before you in the dependency order.
    ///
    /// # When to use `at_end()`
    ///
    /// - **Upstream component outputs**: When you depend on another component's output
    ///   from this timestep (they ran before you and wrote to index N+1)
    /// - **Aggregation**: Combining outputs from multiple components that all wrote
    ///   during the current timestep
    ///
    /// # Returns
    ///
    /// - `Some(value)` if index N+1 exists in the timeseries
    /// - `None` if at the last timestep (index N+1 is out of bounds)
    ///
    /// # Execution order context
    ///
    /// The model solves components in dependency order. Upstream components write their
    /// outputs to index N+1 before downstream components run. This method lets you read
    /// those freshly-written values.
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn solve(&self, t_current: Time, t_next: Time, input_state: &InputState) -> RSCMResult<OutputState> {
    ///     let inputs = MyComponentInputs::from_input_state(input_state);
    ///
    ///     // Read upstream component output (written this timestep)
    ///     // Fall back to start value if at final timestep
    ///     let erf = inputs.effective_radiative_forcing
    ///         .at_end()
    ///         .unwrap_or_else(|| inputs.effective_radiative_forcing.at_start());
    ///
    ///     // Use the forcing to compute temperature response
    ///     let temp_change = erf * self.climate_sensitivity;
    ///     // ...
    /// }
    /// ```
    ///
    /// [`at_start()`]: TimeseriesWindow::at_start
    pub fn at_end(&self) -> Option<FloatValue> {
        let next_index = self.current_index + 1;
        if next_index >= self.timeseries.len() {
            None
        } else {
            self.timeseries
                .at(next_index, ScalarRegion::Global)
                .map(|v| v * self.unit_conversion_factor)
        }
    }

    /// Get the value at the previous timestep, if available.
    ///
    /// Returns `None` if at the first timestep (no previous value exists).
    pub fn previous(&self) -> Option<FloatValue> {
        if self.current_index == 0 {
            None
        } else {
            self.timeseries
                .at(self.current_index - 1, ScalarRegion::Global)
                .map(|v| v * self.unit_conversion_factor)
        }
    }

    /// Get the value at a relative offset from the current timestep.
    ///
    /// Positive offsets look forward in time, negative offsets look backward.
    /// Returns `None` if the resulting index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let prev = window.at_offset(-1);  // Same as previous()
    /// let two_back = window.at_offset(-2);
    /// let next = window.at_offset(1);   // Future value (if available)
    /// ```
    pub fn at_offset(&self, offset: isize) -> Option<FloatValue> {
        let index = self.current_index as isize + offset;
        if index < 0 || index as usize >= self.timeseries.len() {
            None
        } else {
            self.timeseries
                .at(index as usize, ScalarRegion::Global)
                .map(|v| v * self.unit_conversion_factor)
        }
    }

    /// Get the last N values as an array view, ending at the current timestep.
    ///
    /// This is useful for computing moving averages, derivatives, or any operation
    /// that needs historical context.
    ///
    /// **Note:** This method returns raw values without unit conversion applied.
    /// Use [`last_n_converted()`](Self::last_n_converted) if you need converted values.
    ///
    /// # Panics
    ///
    /// Panics if `n` is greater than `current_index + 1` (not enough history).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let last_5 = window.last_n(5);
    /// let avg = last_5.mean().unwrap();
    /// ```
    pub fn last_n(&self, n: usize) -> ArrayView1<'_, FloatValue> {
        assert!(
            n <= self.current_index + 1,
            "Cannot get {} values when only {} available",
            n,
            self.current_index + 1
        );
        let start = self.current_index + 1 - n;
        let end = self.current_index + 1;
        // Get the values column (shape: [time, 1]) and slice the time dimension
        self.timeseries.values().slice(ndarray::s![start..end, 0])
    }

    /// Get the last N values with unit conversion applied.
    ///
    /// Unlike [`last_n()`](Self::last_n), this returns owned data with the
    /// unit conversion factor applied to each value.
    ///
    /// # Panics
    ///
    /// Panics if `n` is greater than `current_index + 1` (not enough history).
    pub fn last_n_converted(&self, n: usize) -> Vec<FloatValue> {
        self.last_n(n)
            .iter()
            .map(|v| v * self.unit_conversion_factor)
            .collect()
    }

    /// Interpolate the value at an arbitrary time point.
    ///
    /// Uses the timeseries's interpolation strategy to compute the value.
    /// This is useful for sub-timestep calculations or when comparing with
    /// observational data at non-model times.
    ///
    /// The returned value has the unit conversion factor applied.
    pub fn interpolate(&self, t: Time) -> RSCMResult<FloatValue> {
        self.timeseries
            .at_time(t, ScalarRegion::Global)
            .map(|v| v * self.unit_conversion_factor)
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

/// A zero-cost view into a grid timeseries at a specific time index.
///
/// `GridTimeseriesWindow` provides efficient access to regional values for spatially-resolved
/// data. It supports both individual region access and full-grid operations.
///
/// # Type Parameters
///
/// * `G` - The spatial grid type (e.g., `FourBoxGrid`, `HemisphericGrid`)
///
/// # Timestep Access Semantics
///
/// See [`TimeseriesWindow`] for detailed guidance on when to use `at_start()` vs `at_end()`.
///
/// # Examples
///
/// ```ignore
/// fn solve(&self, inputs: MyComponentInputs) -> MyComponentOutputs {
///     // Access individual regions at start of timestep
///     let northern_ocean = inputs.temperature.at_start(FourBoxRegion::NorthernOcean);
///
///     // Get all regions at once
///     let all_temps = inputs.temperature.at_start_all();
///
///     // Compute global aggregate
///     let global_temp = inputs.temperature.current_global();
/// }
/// ```
#[derive(Debug)]
pub struct GridTimeseriesWindow<'a, G>
where
    G: SpatialGrid,
{
    timeseries: &'a GridTimeseries<FloatValue, G>,
    current_index: usize,
    current_time: Time,
    /// Unit conversion factor applied to all returned values.
    /// Default is 1.0 (no conversion).
    unit_conversion_factor: f64,
}

impl<'a, G> GridTimeseriesWindow<'a, G>
where
    G: SpatialGrid,
{
    /// Create a new GridTimeseriesWindow from a grid timeseries.
    pub fn new(
        timeseries: &'a GridTimeseries<FloatValue, G>,
        current_index: usize,
        current_time: Time,
    ) -> Self {
        Self {
            timeseries,
            current_index,
            current_time,
            unit_conversion_factor: 1.0,
        }
    }

    /// Create a new GridTimeseriesWindow with unit conversion.
    ///
    /// The returned values will be multiplied by the conversion factor.
    pub fn with_unit_conversion(
        timeseries: &'a GridTimeseries<FloatValue, G>,
        current_index: usize,
        current_time: Time,
        unit_conversion_factor: f64,
    ) -> Self {
        Self {
            timeseries,
            current_index,
            current_time,
            unit_conversion_factor,
        }
    }

    /// Helper to apply unit conversion to a vector of values.
    fn apply_conversion(&self, values: Vec<FloatValue>) -> Vec<FloatValue> {
        if (self.unit_conversion_factor - 1.0).abs() < f64::EPSILON {
            values
        } else {
            values
                .into_iter()
                .map(|v| v * self.unit_conversion_factor)
                .collect()
        }
    }

    /// Get all regional values at the start of the timestep (index N).
    ///
    /// Returns values for all regions in the grid at the beginning of the timestep,
    /// before any components have executed during this timestep.
    ///
    /// # When to use
    ///
    /// - **State variables**: Reading your component's previous regional state
    /// - **Exogenous inputs**: External forcing data pre-populated before the run
    ///
    /// See [`TimeseriesWindow::at_start()`] for detailed execution order semantics.
    pub fn at_start_all(&self) -> Vec<FloatValue> {
        let values = self
            .timeseries
            .at_time_index(self.current_index)
            .expect("Current index out of bounds");
        self.apply_conversion(values)
    }

    /// Get all regional values at the end of the timestep (index N+1), if available.
    ///
    /// Returns values for all regions written during the current timestep by upstream
    /// components that executed before you.
    ///
    /// # When to use
    ///
    /// - **Upstream component outputs**: Regional values written by components that ran before you
    /// - **Aggregation**: Combining regional outputs from multiple components in the same timestep
    ///
    /// # Returns
    ///
    /// - `Some(Vec<FloatValue>)` with values for all regions if index N+1 exists
    /// - `None` if at the last timestep
    ///
    /// See [`TimeseriesWindow::at_end()`] for detailed execution order semantics.
    pub fn at_end_all(&self) -> Option<Vec<FloatValue>> {
        let next_index = self.current_index + 1;
        if next_index >= self.timeseries.len() {
            None
        } else {
            self.timeseries
                .at_time_index(next_index)
                .map(|v| self.apply_conversion(v))
        }
    }

    /// Get all regional values at the current timestep.
    #[deprecated(
        since = "0.2.0",
        note = "Use `at_start_all()` or `at_end_all()` based on variable semantics."
    )]
    pub fn all(&self) -> Vec<FloatValue> {
        self.at_start_all()
    }

    /// Get all regional values at the previous timestep.
    pub fn previous_all(&self) -> Option<Vec<FloatValue>> {
        if self.current_index == 0 {
            None
        } else {
            self.timeseries
                .at_time_index(self.current_index - 1)
                .map(|v| self.apply_conversion(v))
        }
    }

    /// Get all regional values at a relative offset from the current timestep.
    ///
    /// Positive offsets look forward in time, negative offsets look backward.
    /// Returns `None` if the resulting index is out of bounds.
    pub fn at_offset_all(&self, offset: isize) -> Option<Vec<FloatValue>> {
        let index = self.current_index as isize + offset;
        if index < 0 || index as usize >= self.timeseries.len() {
            None
        } else {
            self.timeseries
                .at_time_index(index as usize)
                .map(|v| self.apply_conversion(v))
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

    /// Get a reference to the underlying spatial grid.
    pub fn grid(&self) -> &G {
        self.timeseries.grid()
    }

    /// Get the total length of the underlying timeseries.
    pub fn len(&self) -> usize {
        self.timeseries.len()
    }

    /// Check if the underlying timeseries is empty.
    pub fn is_empty(&self) -> bool {
        self.timeseries.is_empty()
    }

    /// Interpolate all regional values at an arbitrary time point.
    pub fn interpolate_all(&self, t: Time) -> RSCMResult<Vec<FloatValue>> {
        self.timeseries
            .at_time_all(t)
            .map(|v| self.apply_conversion(v))
    }
}

/// Type-safe accessors for FourBoxGrid windows
impl<'a> GridTimeseriesWindow<'a, FourBoxGrid> {
    /// Get a single region's value at the start of the timestep (index N).
    ///
    /// See [`TimeseriesWindow::at_start()`] for detailed semantics on when to use this method.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let inputs = MyComponentInputs::from_input_state(input_state);
    /// let northern_ocean_temp = inputs.temperature.at_start(FourBoxRegion::NorthernOcean);
    /// ```
    pub fn at_start(&self, region: FourBoxRegion) -> FloatValue {
        self.timeseries
            .at(self.current_index, region)
            .expect("Current index out of bounds")
            * self.unit_conversion_factor
    }

    /// Get a single region's value at the end of the timestep (index N+1), if available.
    ///
    /// See [`TimeseriesWindow::at_end()`] for detailed semantics on when to use this method.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Read upstream forcing written this timestep
    /// let erf = inputs.erf.at_end(FourBoxRegion::NorthernOcean)
    ///     .unwrap_or_else(|| inputs.erf.at_start(FourBoxRegion::NorthernOcean));
    /// ```
    pub fn at_end(&self, region: FourBoxRegion) -> Option<FloatValue> {
        let next_index = self.current_index + 1;
        if next_index >= self.timeseries.len() {
            None
        } else {
            self.timeseries
                .at(next_index, region)
                .map(|v| v * self.unit_conversion_factor)
        }
    }

    /// Get a single region's value at the previous timestep.
    pub fn previous(&self, region: FourBoxRegion) -> Option<FloatValue> {
        if self.current_index == 0 {
            None
        } else {
            self.timeseries
                .at(self.current_index - 1, region)
                .map(|v| v * self.unit_conversion_factor)
        }
    }

    /// Get the global aggregate at the start of the timestep (index N).
    ///
    /// Uses the grid's weights to compute a weighted average of all regions.
    /// Note: Unit conversion is already applied in at_start_all().
    pub fn current_global(&self) -> FloatValue {
        let values = self.at_start_all();
        self.timeseries.grid().aggregate_global(&values)
    }

    /// Get the global aggregate at the previous timestep.
    pub fn previous_global(&self) -> Option<FloatValue> {
        self.previous_all()
            .map(|values| self.timeseries.grid().aggregate_global(&values))
    }

    /// Interpolate a single region's value at an arbitrary time.
    pub fn interpolate(&self, t: Time, region: FourBoxRegion) -> RSCMResult<FloatValue> {
        self.timeseries
            .at_time(t, region)
            .map(|v| v * self.unit_conversion_factor)
    }
}

/// Type-safe accessors for HemisphericGrid windows
impl<'a> GridTimeseriesWindow<'a, HemisphericGrid> {
    /// Get a single region's value at the start of the timestep (index N).
    ///
    /// See [`TimeseriesWindow::at_start()`] for detailed semantics on when to use this method.
    pub fn at_start(&self, region: HemisphericRegion) -> FloatValue {
        self.timeseries
            .at(self.current_index, region)
            .expect("Current index out of bounds")
            * self.unit_conversion_factor
    }

    /// Get a single region's value at the end of the timestep (index N+1), if available.
    ///
    /// See [`TimeseriesWindow::at_end()`] for detailed semantics on when to use this method.
    pub fn at_end(&self, region: HemisphericRegion) -> Option<FloatValue> {
        let next_index = self.current_index + 1;
        if next_index >= self.timeseries.len() {
            None
        } else {
            self.timeseries
                .at(next_index, region)
                .map(|v| v * self.unit_conversion_factor)
        }
    }

    /// Get a single region's value at the previous timestep.
    pub fn previous(&self, region: HemisphericRegion) -> Option<FloatValue> {
        if self.current_index == 0 {
            None
        } else {
            self.timeseries
                .at(self.current_index - 1, region)
                .map(|v| v * self.unit_conversion_factor)
        }
    }

    /// Get the global aggregate at the start of the timestep (index N).
    /// Note: Unit conversion is already applied in at_start_all().
    pub fn current_global(&self) -> FloatValue {
        let values = self.at_start_all();
        self.timeseries.grid().aggregate_global(&values)
    }

    /// Get the global aggregate at the previous timestep.
    pub fn previous_global(&self) -> Option<FloatValue> {
        self.previous_all()
            .map(|values| self.timeseries.grid().aggregate_global(&values))
    }

    /// Interpolate a single region's value at an arbitrary time.
    pub fn interpolate(&self, t: Time, region: HemisphericRegion) -> RSCMResult<FloatValue> {
        self.timeseries
            .at_time(t, region)
            .map(|v| v * self.unit_conversion_factor)
    }
}

/// Type-safe accessors for ScalarGrid windows (convenience wrapper)
impl<'a> GridTimeseriesWindow<'a, ScalarGrid> {
    /// Get the scalar value at the start of the timestep (index N).
    ///
    /// See [`TimeseriesWindow::at_start()`] for detailed semantics on when to use this method.
    pub fn at_start(&self) -> FloatValue {
        self.timeseries
            .at(self.current_index, ScalarRegion::Global)
            .expect("Current index out of bounds")
            * self.unit_conversion_factor
    }

    /// Get the scalar value at the end of the timestep (index N+1), if available.
    ///
    /// See [`TimeseriesWindow::at_end()`] for detailed semantics on when to use this method.
    pub fn at_end(&self) -> Option<FloatValue> {
        let next_index = self.current_index + 1;
        if next_index >= self.timeseries.len() {
            None
        } else {
            self.timeseries
                .at(next_index, ScalarRegion::Global)
                .map(|v| v * self.unit_conversion_factor)
        }
    }

    /// Get the previous scalar value.
    pub fn previous(&self) -> Option<FloatValue> {
        if self.current_index == 0 {
            None
        } else {
            self.timeseries
                .at(self.current_index - 1, ScalarRegion::Global)
                .map(|v| v * self.unit_conversion_factor)
        }
    }

    /// Interpolate the value at an arbitrary time.
    pub fn interpolate(&self, t: Time) -> RSCMResult<FloatValue> {
        self.timeseries
            .at_time(t, ScalarRegion::Global)
            .map(|v| v * self.unit_conversion_factor)
    }
}

#[cfg(test)]
mod timeseries_window_tests {
    use super::*;
    use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
    use crate::timeseries::TimeAxis;
    use numpy::array;
    use numpy::ndarray::{Array, Axis};
    use std::sync::Arc;

    fn create_scalar_timeseries() -> Timeseries<FloatValue> {
        let values = array![1.0, 2.0, 3.0, 4.0, 5.0].insert_axis(Axis(1));
        let time_axis = Arc::new(TimeAxis::from_values(Array::range(2000.0, 2005.0, 1.0)));
        GridTimeseries::new(
            values,
            time_axis,
            ScalarGrid,
            "test".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        )
    }

    #[test]
    fn test_timeseries_window_at_start() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::new(&ts, 2, 2002.0);

        // at_start() returns value at index N (the current index)
        assert_eq!(window.at_start(), 3.0);
        assert_eq!(window.time(), 2002.0);
        assert_eq!(window.index(), 2);

        // At index 0
        let window_start = TimeseriesWindow::new(&ts, 0, 2000.0);
        assert_eq!(window_start.at_start(), 1.0);

        // At last index
        let window_end = TimeseriesWindow::new(&ts, 4, 2004.0);
        assert_eq!(window_end.at_start(), 5.0);
    }

    #[test]
    fn test_timeseries_window_at_end() {
        let ts = create_scalar_timeseries();

        // At index 2, at_end() should return value at index 3
        let window = TimeseriesWindow::new(&ts, 2, 2002.0);
        assert_eq!(window.at_end(), Some(4.0));

        // At index 0, at_end() should return value at index 1
        let window_start = TimeseriesWindow::new(&ts, 0, 2000.0);
        assert_eq!(window_start.at_end(), Some(2.0));

        // At last index, at_end() should return None (out of bounds)
        let window_end = TimeseriesWindow::new(&ts, 4, 2004.0);
        assert_eq!(window_end.at_end(), None);
    }

    #[test]
    fn test_timeseries_window_previous() {
        let ts = create_scalar_timeseries();

        // At index 2, previous should be index 1
        let window = TimeseriesWindow::new(&ts, 2, 2002.0);
        assert_eq!(window.previous(), Some(2.0));

        // At index 0, previous should be None
        let window_start = TimeseriesWindow::new(&ts, 0, 2000.0);
        assert_eq!(window_start.previous(), None);
    }

    #[test]
    fn test_timeseries_window_at_offset() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::new(&ts, 2, 2002.0);

        assert_eq!(window.at_offset(0), Some(3.0)); // Current
        assert_eq!(window.at_offset(-1), Some(2.0)); // Previous
        assert_eq!(window.at_offset(-2), Some(1.0)); // Two back
        assert_eq!(window.at_offset(1), Some(4.0)); // Next
        assert_eq!(window.at_offset(2), Some(5.0)); // Two forward
        assert_eq!(window.at_offset(-3), None); // Out of bounds
        assert_eq!(window.at_offset(3), None); // Out of bounds
    }

    #[test]
    fn test_timeseries_window_last_n() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::new(&ts, 4, 2004.0);

        let last_3 = window.last_n(3);
        assert_eq!(last_3.len(), 3);
        assert_eq!(last_3[0], 3.0);
        assert_eq!(last_3[1], 4.0);
        assert_eq!(last_3[2], 5.0);

        let last_1 = window.last_n(1);
        assert_eq!(last_1[0], 5.0);

        let all = window.last_n(5);
        assert_eq!(all.len(), 5);
    }

    #[test]
    #[should_panic(expected = "Cannot get 6 values when only 5 available")]
    fn test_timeseries_window_last_n_panic() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::new(&ts, 4, 2004.0);
        let _ = window.last_n(6); // Only 5 values available
    }

    #[test]
    fn test_timeseries_window_interpolate() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::new(&ts, 2, 2002.0);

        let mid = window.interpolate(2001.5).unwrap();
        assert_eq!(mid, 2.5); // Linear interpolation between 2.0 and 3.0
    }

    #[test]
    fn test_timeseries_window_len() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::new(&ts, 2, 2002.0);

        assert_eq!(window.len(), 5);
        assert!(!window.is_empty());
    }

    // =========================================================================
    // Unit conversion tests for TimeseriesWindow
    // =========================================================================

    #[test]
    fn test_timeseries_window_unit_conversion_at_start() {
        let ts = create_scalar_timeseries();
        // Apply a 2x conversion factor
        let window = TimeseriesWindow::with_unit_conversion(&ts, 2, 2002.0, 2.0);

        // Raw value is 3.0, converted should be 6.0
        assert_eq!(window.at_start(), 6.0);
    }

    #[test]
    fn test_timeseries_window_unit_conversion_at_end() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::with_unit_conversion(&ts, 2, 2002.0, 2.0);

        // Raw value at index 3 is 4.0, converted should be 8.0
        assert_eq!(window.at_end(), Some(8.0));
    }

    #[test]
    fn test_timeseries_window_unit_conversion_previous() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::with_unit_conversion(&ts, 2, 2002.0, 2.0);

        // Raw value at index 1 is 2.0, converted should be 4.0
        assert_eq!(window.previous(), Some(4.0));
    }

    #[test]
    fn test_timeseries_window_unit_conversion_at_offset() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::with_unit_conversion(&ts, 2, 2002.0, 0.5);

        assert_eq!(window.at_offset(0), Some(1.5)); // 3.0 * 0.5
        assert_eq!(window.at_offset(-1), Some(1.0)); // 2.0 * 0.5
        assert_eq!(window.at_offset(1), Some(2.0)); // 4.0 * 0.5
    }

    #[test]
    fn test_timeseries_window_unit_conversion_interpolate() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::with_unit_conversion(&ts, 2, 2002.0, 3.0);

        // Interpolated raw value at 2001.5 is 2.5, converted should be 7.5
        let mid = window.interpolate(2001.5).unwrap();
        assert_eq!(mid, 7.5);
    }

    #[test]
    fn test_timeseries_window_unit_conversion_last_n_converted() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::with_unit_conversion(&ts, 4, 2004.0, 2.0);

        let converted = window.last_n_converted(3);
        assert_eq!(converted, vec![6.0, 8.0, 10.0]); // [3, 4, 5] * 2
    }

    #[test]
    fn test_timeseries_window_unit_conversion_last_n_raw() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::with_unit_conversion(&ts, 4, 2004.0, 2.0);

        // last_n returns raw values without conversion
        let raw = window.last_n(3);
        assert_eq!(raw[0], 3.0);
        assert_eq!(raw[1], 4.0);
        assert_eq!(raw[2], 5.0);
    }

    #[test]
    fn test_timeseries_window_default_conversion_factor_is_one() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::new(&ts, 2, 2002.0);

        // Should return raw value (factor = 1.0)
        assert_eq!(window.at_start(), 3.0);
    }
}

#[cfg(test)]
mod grid_timeseries_window_tests {
    use super::*;
    use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
    use crate::spatial::FourBoxGrid;
    use crate::timeseries::TimeAxis;
    use numpy::array;
    use numpy::ndarray::Array2;
    use std::sync::Arc;

    fn create_four_box_timeseries() -> GridTimeseries<FloatValue, FourBoxGrid> {
        let grid = FourBoxGrid::magicc_standard();
        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0, 2002.0]));
        let values = Array2::from_shape_vec(
            (3, 4),
            vec![
                15.0, 14.0, 10.0, 9.0, // 2000
                16.0, 15.0, 11.0, 10.0, // 2001
                17.0, 16.0, 12.0, 11.0, // 2002
            ],
        )
        .unwrap();

        GridTimeseries::new(
            values,
            time_axis,
            grid,
            "C".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        )
    }

    #[test]
    fn test_grid_window_at_start() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::new(&ts, 1, 2001.0);

        // at_start() returns value at index N (the current index)
        assert_eq!(window.at_start(FourBoxRegion::NorthernOcean), 16.0);
        assert_eq!(window.at_start(FourBoxRegion::NorthernLand), 15.0);
        assert_eq!(window.at_start(FourBoxRegion::SouthernOcean), 11.0);
        assert_eq!(window.at_start(FourBoxRegion::SouthernLand), 10.0);

        // At first index
        let window_start = GridTimeseriesWindow::new(&ts, 0, 2000.0);
        assert_eq!(window_start.at_start(FourBoxRegion::NorthernOcean), 15.0);

        // At last index
        let window_end = GridTimeseriesWindow::new(&ts, 2, 2002.0);
        assert_eq!(window_end.at_start(FourBoxRegion::NorthernOcean), 17.0);
    }

    #[test]
    fn test_grid_window_at_end() {
        let ts = create_four_box_timeseries();

        // At index 1, at_end() should return value at index 2
        let window = GridTimeseriesWindow::new(&ts, 1, 2001.0);
        assert_eq!(
            window.at_end(FourBoxRegion::NorthernOcean),
            Some(17.0) // 2002 value
        );

        // At index 0, at_end() should return value at index 1
        let window_start = GridTimeseriesWindow::new(&ts, 0, 2000.0);
        assert_eq!(
            window_start.at_end(FourBoxRegion::NorthernOcean),
            Some(16.0)
        );

        // At last index, at_end() should return None (out of bounds)
        let window_end = GridTimeseriesWindow::new(&ts, 2, 2002.0);
        assert_eq!(window_end.at_end(FourBoxRegion::NorthernOcean), None);
    }

    #[test]
    fn test_grid_window_at_start_all() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::new(&ts, 1, 2001.0);

        let all = window.at_start_all();
        assert_eq!(all, vec![16.0, 15.0, 11.0, 10.0]);
    }

    #[test]
    fn test_grid_window_at_end_all() {
        let ts = create_four_box_timeseries();

        // At index 1, at_end returns values at index 2
        let window = GridTimeseriesWindow::new(&ts, 1, 2001.0);
        assert_eq!(window.at_end_all(), Some(vec![17.0, 16.0, 12.0, 11.0]));

        // At last index, returns None
        let window_end = GridTimeseriesWindow::new(&ts, 2, 2002.0);
        assert_eq!(window_end.at_end_all(), None);
    }

    #[test]
    #[allow(deprecated)]
    fn test_grid_window_all() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::new(&ts, 1, 2001.0);

        // all() is deprecated alias for at_start_all()
        let all = window.all();
        assert_eq!(all, vec![16.0, 15.0, 11.0, 10.0]);
        assert_eq!(all, window.at_start_all());
    }

    #[test]
    fn test_grid_window_previous() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::new(&ts, 1, 2001.0);

        assert_eq!(window.previous(FourBoxRegion::NorthernOcean), Some(15.0));
        assert_eq!(window.previous_all(), Some(vec![15.0, 14.0, 10.0, 9.0]));

        let window_start = GridTimeseriesWindow::new(&ts, 0, 2000.0);
        assert_eq!(window_start.previous(FourBoxRegion::NorthernOcean), None);
        assert_eq!(window_start.previous_all(), None);
    }

    #[test]
    fn test_grid_window_current_global() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::new(&ts, 1, 2001.0);

        // Equal weights: (16 + 15 + 11 + 10) / 4 = 13.0
        assert_eq!(window.current_global(), 13.0);
    }

    #[test]
    fn test_grid_window_previous_global() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::new(&ts, 1, 2001.0);

        // (15 + 14 + 10 + 9) / 4 = 12.0
        assert_eq!(window.previous_global(), Some(12.0));

        let window_start = GridTimeseriesWindow::new(&ts, 0, 2000.0);
        assert_eq!(window_start.previous_global(), None);
    }

    #[test]
    fn test_grid_window_interpolate() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::new(&ts, 1, 2001.0);

        // Interpolate at midpoint between 2000 and 2001
        let mid = window
            .interpolate(2000.5, FourBoxRegion::NorthernOcean)
            .unwrap();
        assert_eq!(mid, 15.5); // Linear interpolation between 15.0 and 16.0
    }

    #[test]
    fn test_grid_window_interpolate_all() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::new(&ts, 1, 2001.0);

        let mid = window.interpolate_all(2000.5).unwrap();
        assert_eq!(mid, vec![15.5, 14.5, 10.5, 9.5]);
    }

    #[test]
    fn test_grid_window_metadata() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::new(&ts, 1, 2001.0);

        assert_eq!(window.time(), 2001.0);
        assert_eq!(window.index(), 1);
        assert_eq!(window.len(), 3);
        assert!(!window.is_empty());
        assert_eq!(window.grid().size(), 4);
    }

    // =========================================================================
    // Unit conversion tests for GridTimeseriesWindow
    // =========================================================================

    #[test]
    fn test_grid_window_unit_conversion_at_start() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::with_unit_conversion(&ts, 1, 2001.0, 2.0);

        // Raw values at index 1: [16.0, 15.0, 11.0, 10.0]
        // Converted: [32.0, 30.0, 22.0, 20.0]
        assert_eq!(window.at_start(FourBoxRegion::NorthernOcean), 32.0);
        assert_eq!(window.at_start(FourBoxRegion::NorthernLand), 30.0);
        assert_eq!(window.at_start(FourBoxRegion::SouthernOcean), 22.0);
        assert_eq!(window.at_start(FourBoxRegion::SouthernLand), 20.0);
    }

    #[test]
    fn test_grid_window_unit_conversion_at_start_all() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::with_unit_conversion(&ts, 1, 2001.0, 0.5);

        // Raw: [16.0, 15.0, 11.0, 10.0], Converted: [8.0, 7.5, 5.5, 5.0]
        assert_eq!(window.at_start_all(), vec![8.0, 7.5, 5.5, 5.0]);
    }

    #[test]
    fn test_grid_window_unit_conversion_at_end() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::with_unit_conversion(&ts, 1, 2001.0, 2.0);

        // Raw at index 2: [17.0, 16.0, 12.0, 11.0]
        // Converted: [34.0, 32.0, 24.0, 22.0]
        assert_eq!(window.at_end(FourBoxRegion::NorthernOcean), Some(34.0));
        assert_eq!(window.at_end_all(), Some(vec![34.0, 32.0, 24.0, 22.0]));
    }

    #[test]
    fn test_grid_window_unit_conversion_previous() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::with_unit_conversion(&ts, 1, 2001.0, 2.0);

        // Raw at index 0: [15.0, 14.0, 10.0, 9.0]
        // Converted: [30.0, 28.0, 20.0, 18.0]
        assert_eq!(window.previous(FourBoxRegion::NorthernOcean), Some(30.0));
        assert_eq!(window.previous_all(), Some(vec![30.0, 28.0, 20.0, 18.0]));
    }

    #[test]
    fn test_grid_window_unit_conversion_current_global() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::with_unit_conversion(&ts, 1, 2001.0, 2.0);

        // Raw: [16.0, 15.0, 11.0, 10.0] -> (sum = 52.0) / 4 = 13.0
        // After conversion: 13.0 * 2.0 = 26.0 (conversion applied to values first)
        // Actually: [32.0, 30.0, 22.0, 20.0] -> sum = 104.0 -> mean = 26.0
        assert_eq!(window.current_global(), 26.0);
    }

    #[test]
    fn test_grid_window_unit_conversion_interpolate() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::with_unit_conversion(&ts, 1, 2001.0, 2.0);

        // Raw interpolated at 2000.5: 15.5
        // Converted: 31.0
        let mid = window
            .interpolate(2000.5, FourBoxRegion::NorthernOcean)
            .unwrap();
        assert_eq!(mid, 31.0);
    }

    #[test]
    fn test_grid_window_unit_conversion_interpolate_all() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::with_unit_conversion(&ts, 1, 2001.0, 2.0);

        // Raw: [15.5, 14.5, 10.5, 9.5], Converted: [31.0, 29.0, 21.0, 19.0]
        let mid = window.interpolate_all(2000.5).unwrap();
        assert_eq!(mid, vec![31.0, 29.0, 21.0, 19.0]);
    }
}
