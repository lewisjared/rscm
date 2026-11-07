//! Core window types for accessing timeseries values.
//!
//! This module provides zero-cost views into timeseries data during one step,
//! supporting both scalar and grid-based timeseries.

use super::VariableSource;
use crate::component::TimeseriesType;
use crate::errors::{RscmError, RscmResult};
use crate::spatial::{
    FourBoxGrid, FourBoxRegion, HemisphericGrid, HemisphericRegion, ScalarGrid, ScalarRegion,
    SpatialGrid,
};
use crate::timeseries::{FloatValue, GridTimeseries, Time, TimeAxis, Timeseries};

/// Indices a window reads during the step from index `current` (N) to N+1.
///
/// A point is stored at its time and an average at its step's start bound,
/// so `get()` reads index g: N for an own or lagged point, N+1 for any other point,
/// N-1 for an own average and N for any other average.
/// Prescribed and upstream inputs read the same index, so prescribing a variable never shifts it.
/// A lagged input reads one step behind an unlagged one, so its interval is (N-1, N).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReadIndex {
    pub current: usize,
    pub len: usize,
    pub source: VariableSource,
    pub timeseries_type: TimeseriesType,
}

impl ReadIndex {
    /// Index g that `get()` reads.
    pub fn get(self) -> usize {
        let behind = matches!(
            self.source,
            VariableSource::Lagged | VariableSource::OwnState
        );
        match (self.timeseries_type, behind) {
            (TimeseriesType::Point, false) => self.current + 1,
            // At the first step a component's own average reads its initial value.
            // The builder rejects a lagged average.
            (TimeseriesType::Average, true) => self.current.saturating_sub(1),
            _ => self.current,
        }
    }

    /// Index g - n, or `None` before the first step.
    pub fn history(self, n: usize) -> Option<usize> {
        self.get().checked_sub(n)
    }

    /// Number of values `history` can read, g + 1.
    pub fn history_len(self) -> usize {
        self.get() + 1
    }

    /// Indices of the step's start and end values.
    pub fn interval(self) -> RscmResult<(usize, usize)> {
        if self.timeseries_type == TimeseriesType::Average {
            return Err(RscmError::Error(
                "interval() is undefined for an average, which has one value per step".to_string(),
            ));
        }
        match self.source {
            VariableSource::OwnState => Err(RscmError::Error(
                "interval() is undefined for a component's own state, whose step end it writes"
                    .to_string(),
            )),
            VariableSource::Lagged => Ok((self.current.saturating_sub(1), self.current)),
            VariableSource::Prescribed | VariableSource::Upstream => {
                if self.current + 1 < self.len {
                    Ok((self.current, self.current + 1))
                } else {
                    Err(RscmError::Error(format!(
                        "interval() needs the step's end, but index {} is the last of {}",
                        self.current, self.len
                    )))
                }
            }
        }
    }
}

/// Panics because the series of variable `name` has no value at `index`.
pub(crate) fn out_of_range(name: &str, index: usize, len: usize) -> ! {
    panic!("Variable '{name}' has {len} values, so index {index} is out of range")
}

/// Index of the stored value at `t`: a node of `axis`, or its first or last node beyond its ends.
///
/// `None` when `t` lies between two nodes.
pub(crate) fn stored_index(axis: &TimeAxis, t: Time) -> Option<usize> {
    let values = axis.values();
    let last = values.len().checked_sub(1)?;
    if t <= values[0] {
        Some(0)
    } else if t >= values[last] {
        Some(last)
    } else {
        axis.index_of(t)
    }
}

/// A zero-cost view into a scalar timeseries during one step.
///
/// This is the primary way components read their input variables.
/// A component reads its input with [`get()`](Self::get) and never chooses a time index.
/// The model decides which value `get()` returns:
/// this step's value from an upstream component, or the previous step's value on a lagged edge.
///
/// - [`get()`](Self::get) reads the value the model wired this input to.
/// - [`interval()`](Self::interval) gives a point's values at the start and end of the step.
/// - [`history(n)`](Self::history) reads `n` steps before `get()`, and never the future.
///
/// # Examples
///
/// ```ignore
/// let emissions = inputs.emissions_co2.get();
/// let previous = inputs.emissions_co2.history(1).unwrap_or(emissions);
/// let (erf_start, erf_end) = inputs.erf.interval()?;
/// ```
#[derive(Debug)]
pub struct TimeseriesWindow<'a> {
    timeseries: &'a Timeseries<FloatValue>,
    time_index: usize,
    name: &'a str,
    /// Unit conversion factor applied to all returned values.
    /// Default is 1.0 (no conversion).
    unit_conversion_factor: f64,
    /// Source type determining get() behaviour
    variable_source: VariableSource,
    timeseries_type: TimeseriesType,
}

impl<'a> TimeseriesWindow<'a> {
    /// Create a new TimeseriesWindow from a scalar timeseries.
    ///
    /// # Arguments
    ///
    /// * `timeseries` - The underlying scalar timeseries
    /// * `time_index` - Index of the step's start
    /// * `name` - Name of the variable, used in error messages
    pub fn new(timeseries: &'a Timeseries<FloatValue>, time_index: usize, name: &'a str) -> Self {
        Self::with_source(
            timeseries,
            time_index,
            name,
            1.0,
            VariableSource::Prescribed,
        )
    }

    /// Create a new TimeseriesWindow with unit conversion.
    ///
    /// The returned values will be multiplied by the conversion factor.
    /// This is used when the stored data uses different units than the component expects.
    pub fn with_unit_conversion(
        timeseries: &'a Timeseries<FloatValue>,
        time_index: usize,
        name: &'a str,
        unit_conversion_factor: f64,
    ) -> Self {
        Self::with_source(
            timeseries,
            time_index,
            name,
            unit_conversion_factor,
            VariableSource::Prescribed,
        )
    }

    /// Create a TimeseriesWindow with all options including variable source.
    pub fn with_source(
        timeseries: &'a Timeseries<FloatValue>,
        time_index: usize,
        name: &'a str,
        unit_conversion_factor: f64,
        variable_source: VariableSource,
    ) -> Self {
        Self {
            timeseries,
            time_index,
            name,
            unit_conversion_factor,
            variable_source,
            timeseries_type: TimeseriesType::Point,
        }
    }

    /// Set whether the variable is a point or an average, which decides whether `interval()` exists.
    pub fn with_timeseries_type(mut self, timeseries_type: TimeseriesType) -> Self {
        self.timeseries_type = timeseries_type;
        self
    }

    /// Source of the variable, which decides the index `get()` reads.
    pub fn variable_source(&self) -> VariableSource {
        self.variable_source
    }

    /// Whether the variable is a point or an average.
    pub fn timeseries_type(&self) -> TimeseriesType {
        self.timeseries_type
    }

    pub(crate) fn time_axis(&self) -> &TimeAxis {
        self.timeseries.time_axis_ref()
    }

    fn read_index(&self) -> ReadIndex {
        ReadIndex {
            current: self.time_index,
            len: self.timeseries.len(),
            source: self.variable_source,
            timeseries_type: self.timeseries_type,
        }
    }

    fn value_at(&self, index: usize) -> FloatValue {
        self.timeseries
            .at(index, ScalarRegion::Global)
            .unwrap_or_else(|| out_of_range(self.name, index, self.timeseries.len()))
            * self.unit_conversion_factor
    }

    /// Value the model wired this input to read during the step.
    ///
    /// For a point, this is the step's end value, whether prescribed or from an upstream component,
    /// and the step's start value for a component's own state and a lagged edge.
    /// For an average, it is this step's value, or the previous step's for a component's own output.
    ///
    /// # Panics
    ///
    /// Panics with the variable name if that value lies past the end of the series.
    pub fn get(&self) -> FloatValue {
        self.value_at(self.read_index().get())
    }

    /// Time of the value `get()` returns.
    pub fn get_time(&self) -> Time {
        let index = self.read_index().get();
        self.time_axis()
            .at(index)
            .unwrap_or_else(|| out_of_range(self.name, index, self.timeseries.len()))
    }

    /// Value `n` steps before the one `get()` returns, so `history(0) == get()`.
    ///
    /// Returns `None` before the first step.
    pub fn history(&self, n: usize) -> Option<FloatValue> {
        self.read_index().history(n).map(|i| self.value_at(i))
    }

    /// Number of values `history` can read.
    pub fn history_len(&self) -> usize {
        self.read_index().history_len()
    }

    /// Values at the start and end of the step.
    ///
    /// A lagged input gives the previous step's interval.
    /// Errors on an average, which has one value per step, and on a component's own state.
    pub fn interval(&self) -> RscmResult<(FloatValue, FloatValue)> {
        let (start, end) = self.read_index().interval()?;
        Ok((self.value_at(start), self.value_at(end)))
    }

    /// Value of the stored series at time `t`, past or future.
    ///
    /// Returns the stored value at a node and holds the first or last value beyond the axis.
    /// Between nodes it uses the series's interpolation strategy.
    /// This suits fixed reference years of prescribed data,
    /// but a future value of an upstream output is not yet written.
    pub fn interpolate(&self, t: Time) -> RscmResult<FloatValue> {
        match stored_index(self.time_axis(), t) {
            Some(index) => Ok(self.value_at(index)),
            None => self
                .timeseries
                .at_time(t, ScalarRegion::Global)
                .map(|v| v * self.unit_conversion_factor),
        }
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

/// A zero-cost view into a grid timeseries during one step.
///
/// `GridTimeseriesWindow` reads regional values for spatially-resolved data,
/// one region at a time or the whole grid at once.
///
/// # Type Parameters
///
/// * `G` - The spatial grid type (e.g., `FourBoxGrid`, `HemisphericGrid`)
///
/// # Reading values
///
/// See [`TimeseriesWindow`] for how the model decides which value `get()` returns.
///
/// # Examples
///
/// ```ignore
/// let northern_ocean = inputs.temperature.get(FourBoxRegion::NorthernOcean);
/// let all_temps = inputs.temperature.get_all();
/// let previous = inputs.temperature.history(1);
/// ```
#[derive(Debug)]
pub struct GridTimeseriesWindow<'a, G>
where
    G: SpatialGrid,
{
    timeseries: &'a GridTimeseries<FloatValue, G>,
    time_index: usize,
    name: &'a str,
    /// Unit conversion factor applied to all returned values.
    /// Default is 1.0 (no conversion).
    unit_conversion_factor: f64,
    /// Source type determining get() behaviour
    variable_source: VariableSource,
    timeseries_type: TimeseriesType,
}

impl<'a, G> GridTimeseriesWindow<'a, G>
where
    G: SpatialGrid,
{
    /// Create a new GridTimeseriesWindow from a grid timeseries.
    ///
    /// `name` is the variable's name, used in error messages.
    pub fn new(
        timeseries: &'a GridTimeseries<FloatValue, G>,
        time_index: usize,
        name: &'a str,
    ) -> Self {
        Self::with_source(
            timeseries,
            time_index,
            name,
            1.0,
            VariableSource::Prescribed,
        )
    }

    /// Create a new GridTimeseriesWindow with unit conversion.
    ///
    /// The returned values will be multiplied by the conversion factor.
    pub fn with_unit_conversion(
        timeseries: &'a GridTimeseries<FloatValue, G>,
        time_index: usize,
        name: &'a str,
        unit_conversion_factor: f64,
    ) -> Self {
        Self::with_source(
            timeseries,
            time_index,
            name,
            unit_conversion_factor,
            VariableSource::Prescribed,
        )
    }

    /// Create a GridTimeseriesWindow with all options including variable source.
    pub fn with_source(
        timeseries: &'a GridTimeseries<FloatValue, G>,
        time_index: usize,
        name: &'a str,
        unit_conversion_factor: f64,
        variable_source: VariableSource,
    ) -> Self {
        Self {
            timeseries,
            time_index,
            name,
            unit_conversion_factor,
            variable_source,
            timeseries_type: TimeseriesType::Point,
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

    /// Set whether the variable is a point or an average, which decides whether `interval()` exists.
    pub fn with_timeseries_type(mut self, timeseries_type: TimeseriesType) -> Self {
        self.timeseries_type = timeseries_type;
        self
    }

    /// Source of the variable, which decides the index `get()` reads.
    pub fn variable_source(&self) -> VariableSource {
        self.variable_source
    }

    /// Whether the variable is a point or an average.
    pub fn timeseries_type(&self) -> TimeseriesType {
        self.timeseries_type
    }

    pub(crate) fn time_axis(&self) -> &TimeAxis {
        self.timeseries.time_axis_ref()
    }

    fn read_index(&self) -> ReadIndex {
        ReadIndex {
            current: self.time_index,
            len: self.timeseries.len(),
            source: self.variable_source,
            timeseries_type: self.timeseries_type,
        }
    }

    fn out_of_range(&self, index: usize) -> ! {
        out_of_range(self.name, index, self.timeseries.len())
    }

    fn values_at(&self, index: usize) -> Vec<FloatValue> {
        let values = self
            .timeseries
            .at_time_index(index)
            .unwrap_or_else(|| self.out_of_range(index));
        self.apply_conversion(values)
    }

    /// All regional values the model wired this input to read during the step.
    ///
    /// See [`TimeseriesWindow::get()`] for which step's values these are.
    pub fn get_all(&self) -> Vec<FloatValue> {
        self.values_at(self.read_index().get())
    }

    /// Time of the values `get_all()` returns.
    pub fn get_time(&self) -> Time {
        let index = self.read_index().get();
        self.time_axis()
            .at(index)
            .unwrap_or_else(|| self.out_of_range(index))
    }

    /// Regional values `n` steps before those `get_all()` returns, so `history(0) == get_all()`.
    ///
    /// Returns `None` before the first step.
    pub fn history(&self, n: usize) -> Option<Vec<FloatValue>> {
        self.read_index().history(n).map(|i| self.values_at(i))
    }

    /// Number of values `history` can read.
    pub fn history_len(&self) -> usize {
        self.read_index().history_len()
    }

    /// Regional values at the start and end of the step.
    ///
    /// See [`TimeseriesWindow::interval()`] for lagged inputs and errors.
    pub fn interval(&self) -> RscmResult<(Vec<FloatValue>, Vec<FloatValue>)> {
        let (start, end) = self.read_index().interval()?;
        Ok((self.values_at(start), self.values_at(end)))
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

    /// Regional values of the stored series at time `t`, past or future.
    ///
    /// See [`TimeseriesWindow::interpolate()`] for nodes, the axis ends and upstream outputs.
    pub fn interpolate_all(&self, t: Time) -> RscmResult<Vec<FloatValue>> {
        match stored_index(self.time_axis(), t) {
            Some(index) => Ok(self.values_at(index)),
            None => self
                .timeseries
                .at_time_all(t)
                .map(|v| self.apply_conversion(v)),
        }
    }
}

/// Type-safe accessors for FourBoxGrid windows
impl<'a> GridTimeseriesWindow<'a, FourBoxGrid> {
    fn region_at(&self, index: usize, region: FourBoxRegion) -> FloatValue {
        self.timeseries
            .at(index, region)
            .unwrap_or_else(|| self.out_of_range(index))
            * self.unit_conversion_factor
    }

    /// A single region's value that the model wired this input to read during the step.
    ///
    /// See [`TimeseriesWindow::get()`] for which step's value this is.
    pub fn get(&self, region: FourBoxRegion) -> FloatValue {
        self.region_at(self.read_index().get(), region)
    }

    /// A single region's value of the stored series at time `t`, past or future.
    ///
    /// See [`TimeseriesWindow::interpolate()`] for nodes, the axis ends and upstream outputs.
    pub fn interpolate(&self, t: Time, region: FourBoxRegion) -> RscmResult<FloatValue> {
        match stored_index(self.time_axis(), t) {
            Some(index) => Ok(self.region_at(index, region)),
            None => self
                .timeseries
                .at_time(t, region)
                .map(|v| v * self.unit_conversion_factor),
        }
    }
}

/// Type-safe accessors for HemisphericGrid windows
impl<'a> GridTimeseriesWindow<'a, HemisphericGrid> {
    fn region_at(&self, index: usize, region: HemisphericRegion) -> FloatValue {
        self.timeseries
            .at(index, region)
            .unwrap_or_else(|| self.out_of_range(index))
            * self.unit_conversion_factor
    }

    /// A single region's value that the model wired this input to read during the step.
    ///
    /// See [`TimeseriesWindow::get()`] for which step's value this is.
    pub fn get(&self, region: HemisphericRegion) -> FloatValue {
        self.region_at(self.read_index().get(), region)
    }

    /// A single region's value of the stored series at time `t`, past or future.
    ///
    /// See [`TimeseriesWindow::interpolate()`] for nodes, the axis ends and upstream outputs.
    pub fn interpolate(&self, t: Time, region: HemisphericRegion) -> RscmResult<FloatValue> {
        match stored_index(self.time_axis(), t) {
            Some(index) => Ok(self.region_at(index, region)),
            None => self
                .timeseries
                .at_time(t, region)
                .map(|v| v * self.unit_conversion_factor),
        }
    }
}

/// Type-safe accessors for ScalarGrid windows (convenience wrapper)
impl<'a> GridTimeseriesWindow<'a, ScalarGrid> {
    fn scalar_at(&self, index: usize) -> FloatValue {
        self.timeseries
            .at(index, ScalarRegion::Global)
            .unwrap_or_else(|| self.out_of_range(index))
            * self.unit_conversion_factor
    }

    /// Value the model wired this input to read during the step.
    ///
    /// See [`TimeseriesWindow::get()`] for which step's value this is.
    pub fn get(&self) -> FloatValue {
        self.scalar_at(self.read_index().get())
    }

    /// Value of the stored series at time `t`, past or future.
    ///
    /// See [`TimeseriesWindow::interpolate()`] for nodes, the axis ends and upstream outputs.
    pub fn interpolate(&self, t: Time) -> RscmResult<FloatValue> {
        match stored_index(self.time_axis(), t) {
            Some(index) => Ok(self.scalar_at(index)),
            None => self
                .timeseries
                .at_time(t, ScalarRegion::Global)
                .map(|v| v * self.unit_conversion_factor),
        }
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
    fn test_timeseries_window_get_by_source() {
        let ts = create_scalar_timeseries();
        let window = |index, source| TimeseriesWindow::with_source(&ts, index, "test", 1.0, source);

        assert_eq!(window(2, VariableSource::Prescribed).get(), 4.0);
        assert_eq!(window(2, VariableSource::OwnState).get(), 3.0);
        assert_eq!(window(2, VariableSource::Lagged).get(), 3.0);
        assert_eq!(window(2, VariableSource::Upstream).get(), 4.0);

        assert_eq!(window(0, VariableSource::Prescribed).get(), 2.0);
        assert_eq!(window(4, VariableSource::OwnState).get(), 5.0);
        assert_eq!(window(3, VariableSource::Prescribed).get(), 5.0);
        assert_eq!(window(3, VariableSource::Upstream).get(), 5.0);
    }

    #[test]
    #[should_panic(expected = "Variable 'Emissions|CO2' has 5 values, so index 5 is out of range")]
    fn test_timeseries_window_get_upstream_at_last_index_panics() {
        let ts = create_scalar_timeseries();
        let window =
            TimeseriesWindow::with_source(&ts, 4, "Emissions|CO2", 1.0, VariableSource::Upstream);
        let _ = window.get();
    }

    #[test]
    fn test_timeseries_window_interpolate() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::new(&ts, 2, "test");

        let mid = window.interpolate(2001.5).unwrap();
        assert_eq!(mid, 2.5); // Linear interpolation between 2.0 and 3.0
    }

    #[test]
    fn test_timeseries_window_len() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::new(&ts, 2, "test");

        assert_eq!(window.len(), 5);
        assert!(!window.is_empty());
    }

    // =========================================================================
    // Unit conversion tests for TimeseriesWindow
    // =========================================================================

    #[test]
    fn test_timeseries_window_unit_conversion_get() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::with_unit_conversion(&ts, 1, "test", 2.0);

        // Raw value is 3.0, converted should be 6.0
        assert_eq!(window.get(), 6.0);

        let upstream = TimeseriesWindow::with_source(&ts, 2, "test", 2.0, VariableSource::Upstream);
        // Raw value at index 3 is 4.0, converted should be 8.0
        assert_eq!(upstream.get(), 8.0);
    }

    #[test]
    fn test_timeseries_window_unit_conversion_history() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::with_unit_conversion(&ts, 1, "test", 0.5);

        assert_eq!(window.history(0), Some(1.5)); // 3.0 * 0.5
        assert_eq!(window.history(1), Some(1.0)); // 2.0 * 0.5
    }

    #[test]
    fn test_timeseries_window_unit_conversion_interpolate() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::with_unit_conversion(&ts, 2, "test", 3.0);

        // Interpolated raw value at 2001.5 is 2.5, converted should be 7.5
        let mid = window.interpolate(2001.5).unwrap();
        assert_eq!(mid, 7.5);
    }

    #[test]
    fn test_timeseries_window_default_conversion_factor_is_one() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::new(&ts, 1, "test");

        // Should return raw value (factor = 1.0)
        assert_eq!(window.get(), 3.0);
    }

    // =========================================================================
    // get_time(), history(), history_len() and interval()
    // =========================================================================

    fn window_from(
        ts: &Timeseries<FloatValue>,
        index: usize,
        source: VariableSource,
    ) -> TimeseriesWindow<'_> {
        TimeseriesWindow::with_source(ts, index, "test", 1.0, source)
    }

    #[test]
    fn test_window_prescribed_point_reads_step_end() {
        let ts = create_scalar_timeseries();
        let window = window_from(&ts, 2, VariableSource::Prescribed);

        assert_eq!(window.get_time(), 2003.0);
        assert_eq!(window.history(0), Some(4.0));
        assert_eq!(window.history(3), Some(1.0));
        assert_eq!(window.history(4), None);
        assert_eq!(window.history_len(), 4);
        assert_eq!(window.interval().unwrap(), (3.0, 4.0));
    }

    #[test]
    fn test_window_point_reads_same_index_prescribed_or_upstream() {
        let ts = create_scalar_timeseries();
        let prescribed = window_from(&ts, 2, VariableSource::Prescribed);
        let upstream = window_from(&ts, 2, VariableSource::Upstream);

        assert_eq!(prescribed.get(), upstream.get());
        assert_eq!(prescribed.interval().unwrap(), upstream.interval().unwrap());
        assert_eq!(prescribed.history(1), upstream.history(1));
    }

    #[test]
    fn test_window_upstream_reads_step_end() {
        let ts = create_scalar_timeseries();
        let window = window_from(&ts, 2, VariableSource::Upstream);

        assert_eq!(window.get_time(), 2003.0);
        assert_eq!(window.history(0), Some(4.0));
        assert_eq!(window.history(1), Some(3.0));
        assert_eq!(window.history(3), Some(1.0));
        assert_eq!(window.history(4), None);
        assert_eq!(window.history_len(), 4);
        assert_eq!(window.interval().unwrap(), (3.0, 4.0));
    }

    #[test]
    fn test_window_lagged_shifts_back_a_step() {
        let ts = create_scalar_timeseries();
        let window = window_from(&ts, 2, VariableSource::Lagged);

        assert_eq!(window.get_time(), 2002.0);
        assert_eq!(window.history(0), Some(3.0));
        assert_eq!(window.history_len(), 3);
        assert_eq!(window.interval().unwrap(), (2.0, 3.0));

        let first = window_from(&ts, 0, VariableSource::Lagged);
        assert_eq!(first.interval().unwrap(), (1.0, 1.0));
    }

    #[test]
    fn test_window_own_state_has_no_interval() {
        let ts = create_scalar_timeseries();
        let window = window_from(&ts, 2, VariableSource::OwnState);

        assert_eq!(window.get_time(), 2002.0);
        assert_eq!(window.history(1), Some(2.0));
        assert_eq!(window.history_len(), 3);
        let err = window.interval().unwrap_err().to_string();
        assert!(err.contains("own state"), "{err}");
    }

    #[test]
    fn test_window_history_at_first_step() {
        let ts = create_scalar_timeseries();
        for source in [VariableSource::OwnState, VariableSource::Lagged] {
            let window = window_from(&ts, 0, source);
            assert_eq!(window.history(0), Some(1.0));
            assert_eq!(window.history(1), None);
            assert_eq!(window.history_len(), 1);
        }
        for source in [VariableSource::Prescribed, VariableSource::Upstream] {
            let window = window_from(&ts, 0, source);
            assert_eq!(window.history(1), Some(1.0));
            assert_eq!(window.history(2), None);
            assert_eq!(window.history_len(), 2);
        }
    }

    #[test]
    fn test_window_interval_errors_on_average() {
        let ts = create_scalar_timeseries();
        for source in [VariableSource::Prescribed, VariableSource::Upstream] {
            let window = window_from(&ts, 2, source).with_timeseries_type(TimeseriesType::Average);
            let err = window.interval().unwrap_err().to_string();
            assert!(err.contains("average"), "{err}");
            assert_eq!(window.get(), window.history(0).unwrap());
        }
    }

    #[test]
    fn test_window_average_reads_step_start_bound() {
        let ts = create_scalar_timeseries();
        let average = |index, source| {
            window_from(&ts, index, source).with_timeseries_type(TimeseriesType::Average)
        };

        assert_eq!(average(2, VariableSource::Prescribed).get(), 3.0);
        assert_eq!(average(2, VariableSource::Upstream).get(), 3.0);
        assert_eq!(average(2, VariableSource::Lagged).get(), 2.0);
        assert_eq!(average(2, VariableSource::OwnState).get(), 2.0);
        assert_eq!(average(0, VariableSource::OwnState).get(), 1.0);

        let upstream = average(2, VariableSource::Upstream);
        assert_eq!(upstream.get_time(), 2002.0);
        assert_eq!(upstream.history(1), Some(2.0));
        assert_eq!(upstream.history_len(), 3);
    }

    #[test]
    fn test_window_interval_errors_at_last_index() {
        let ts = create_scalar_timeseries();
        let window = window_from(&ts, 4, VariableSource::Prescribed);
        assert!(window.interval().is_err());
    }

    #[test]
    fn test_window_new_accessors_convert_units() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::with_source(&ts, 2, "test", 2.0, VariableSource::Prescribed);

        assert_eq!(window.history(1), Some(6.0));
        assert_eq!(window.interval().unwrap(), (6.0, 8.0));
        assert_eq!(window.interpolate(2004.0).unwrap(), 10.0);
    }

    #[test]
    fn test_window_interpolate_reads_stored_series_at_any_time() {
        let ts = create_scalar_timeseries();
        let window = window_from(&ts, 1, VariableSource::Prescribed);

        assert_eq!(window.interpolate(2000.0).unwrap(), 1.0);
        assert_eq!(window.interpolate(2003.0).unwrap(), 4.0);
        assert_eq!(window.interpolate(2004.0).unwrap(), 5.0);
        assert_eq!(window.interpolate(2003.5).unwrap(), 4.5);
        // Held at the ends rather than extrapolated
        assert_eq!(window.interpolate(1990.0).unwrap(), 1.0);
        assert_eq!(window.interpolate(2010.0).unwrap(), 5.0);
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
    fn test_grid_window_get() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::new(&ts, 0, "test");

        assert_eq!(window.get(FourBoxRegion::NorthernOcean), 16.0);
        assert_eq!(window.get(FourBoxRegion::NorthernLand), 15.0);
        assert_eq!(window.get(FourBoxRegion::SouthernOcean), 11.0);
        assert_eq!(window.get(FourBoxRegion::SouthernLand), 10.0);
        assert_eq!(window.get_all(), vec![16.0, 15.0, 11.0, 10.0]);

        let window_end = GridTimeseriesWindow::new(&ts, 1, "test");
        assert_eq!(window_end.get(FourBoxRegion::NorthernOcean), 17.0);
    }

    #[test]
    fn test_grid_window_get_upstream() {
        let ts = create_four_box_timeseries();
        let window =
            GridTimeseriesWindow::with_source(&ts, 1, "test", 1.0, VariableSource::Upstream);

        assert_eq!(window.get(FourBoxRegion::NorthernOcean), 17.0);
        assert_eq!(window.get_all(), vec![17.0, 16.0, 12.0, 11.0]);
    }

    #[test]
    #[should_panic(
        expected = "Variable 'Surface Temperature' has 3 values, so index 3 is out of range"
    )]
    fn test_grid_window_get_upstream_at_last_index_panics() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::with_source(
            &ts,
            2,
            "Surface Temperature",
            1.0,
            VariableSource::Upstream,
        );
        let _ = window.get(FourBoxRegion::NorthernOcean);
    }

    #[test]
    fn test_grid_window_lagged() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::with_source(&ts, 1, "test", 1.0, VariableSource::Lagged);

        assert_eq!(window.get_all(), vec![16.0, 15.0, 11.0, 10.0]);
        assert_eq!(window.get(FourBoxRegion::NorthernOcean), 16.0);
    }

    #[test]
    fn test_grid_window_interpolate() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::new(&ts, 1, "test");

        // Interpolate at midpoint between 2000 and 2001
        let mid = window
            .interpolate(2000.5, FourBoxRegion::NorthernOcean)
            .unwrap();
        assert_eq!(mid, 15.5); // Linear interpolation between 15.0 and 16.0
    }

    #[test]
    fn test_grid_window_interpolate_all() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::new(&ts, 1, "test");

        let mid = window.interpolate_all(2000.5).unwrap();
        assert_eq!(mid, vec![15.5, 14.5, 10.5, 9.5]);
    }

    #[test]
    fn test_grid_window_metadata() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::new(&ts, 1, "test");

        assert_eq!(window.len(), 3);
        assert!(!window.is_empty());
        assert_eq!(window.grid().size(), 4);
    }

    // =========================================================================
    // Unit conversion tests for GridTimeseriesWindow
    // =========================================================================

    #[test]
    fn test_grid_window_unit_conversion_get() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::with_unit_conversion(&ts, 0, "test", 2.0);

        // Raw values at index 1: [16.0, 15.0, 11.0, 10.0]
        // Converted: [32.0, 30.0, 22.0, 20.0]
        assert_eq!(window.get(FourBoxRegion::NorthernOcean), 32.0);
        assert_eq!(window.get(FourBoxRegion::NorthernLand), 30.0);
        assert_eq!(window.get(FourBoxRegion::SouthernOcean), 22.0);
        assert_eq!(window.get(FourBoxRegion::SouthernLand), 20.0);
    }

    #[test]
    fn test_grid_window_unit_conversion_get_all() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::with_unit_conversion(&ts, 0, "test", 0.5);

        // Raw: [16.0, 15.0, 11.0, 10.0], Converted: [8.0, 7.5, 5.5, 5.0]
        assert_eq!(window.get_all(), vec![8.0, 7.5, 5.5, 5.0]);
    }

    #[test]
    fn test_grid_window_unit_conversion_get_upstream() {
        let ts = create_four_box_timeseries();
        let window =
            GridTimeseriesWindow::with_source(&ts, 1, "test", 2.0, VariableSource::Upstream);

        // Raw at index 2: [17.0, 16.0, 12.0, 11.0]
        // Converted: [34.0, 32.0, 24.0, 22.0]
        assert_eq!(window.get(FourBoxRegion::NorthernOcean), 34.0);
        assert_eq!(window.get_all(), vec![34.0, 32.0, 24.0, 22.0]);
    }

    #[test]
    fn test_grid_window_unit_conversion_interpolate() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::with_unit_conversion(&ts, 1, "test", 2.0);

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
        let window = GridTimeseriesWindow::with_unit_conversion(&ts, 1, "test", 2.0);

        // Raw: [15.5, 14.5, 10.5, 9.5], Converted: [31.0, 29.0, 21.0, 19.0]
        let mid = window.interpolate_all(2000.5).unwrap();
        assert_eq!(mid, vec![31.0, 29.0, 21.0, 19.0]);
    }

    // =========================================================================
    // get_time(), history(), history_len() and interval() on grid windows
    // =========================================================================

    #[test]
    fn test_grid_window_new_accessors_by_source() {
        let ts = create_four_box_timeseries();
        let at_2000 = vec![15.0, 14.0, 10.0, 9.0];
        let at_2001 = vec![16.0, 15.0, 11.0, 10.0];
        let at_2002 = vec![17.0, 16.0, 12.0, 11.0];
        let window = |source| GridTimeseriesWindow::with_source(&ts, 1, "test", 1.0, source);

        let prescribed = window(VariableSource::Prescribed);
        assert_eq!(prescribed.get_time(), 2002.0);
        assert_eq!(prescribed.history(0), Some(prescribed.get_all()));
        assert_eq!(prescribed.history(1), Some(at_2001.clone()));
        assert_eq!(prescribed.history_len(), 3);
        assert_eq!(
            prescribed.interval().unwrap(),
            (at_2001.clone(), at_2002.clone())
        );

        let upstream = window(VariableSource::Upstream);
        assert_eq!(upstream.get_time(), 2002.0);
        assert_eq!(upstream.history(0), Some(at_2002.clone()));
        assert_eq!(upstream.history_len(), 3);
        assert_eq!(upstream.interval().unwrap(), (at_2001.clone(), at_2002));

        let lagged = window(VariableSource::Lagged);
        assert_eq!(lagged.get_time(), 2001.0);
        assert_eq!(lagged.interval().unwrap(), (at_2000.clone(), at_2001));

        let own_state = window(VariableSource::OwnState);
        assert!(own_state.interval().is_err());

        let average =
            window(VariableSource::Prescribed).with_timeseries_type(TimeseriesType::Average);
        assert!(average.interval().is_err());

        let first =
            GridTimeseriesWindow::with_source(&ts, 0, "test", 1.0, VariableSource::Prescribed);
        assert_eq!(first.history(1), Some(at_2000));
        assert_eq!(first.history(2), None);
    }

    #[test]
    fn test_grid_window_new_accessors_convert_units() {
        let ts = create_four_box_timeseries();
        let window =
            GridTimeseriesWindow::with_source(&ts, 1, "test", 2.0, VariableSource::Prescribed);

        assert_eq!(window.history(1), Some(vec![32.0, 30.0, 22.0, 20.0]));
        assert_eq!(
            window.interval().unwrap(),
            (vec![32.0, 30.0, 22.0, 20.0], vec![34.0, 32.0, 24.0, 22.0])
        );
        assert_eq!(
            window.interpolate_all(2005.0).unwrap(),
            vec![34.0, 32.0, 24.0, 22.0]
        );
        assert_eq!(
            window
                .interpolate(1990.0, FourBoxRegion::NorthernOcean)
                .unwrap(),
            30.0
        );
    }
}
