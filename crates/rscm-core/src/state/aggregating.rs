//! Aggregating window types for grid-to-scalar and grid-to-grid transformations.
//!
//! This module provides window types that aggregate data from finer grids (e.g., FourBox)
//! to coarser representations (e.g., Scalar or Hemispheric).

use crate::component::{GridType, TimeseriesType};
use crate::errors::RSCMResult;
use crate::spatial::{FourBoxGrid, HemisphericGrid, HemisphericRegion, SpatialGrid};
use crate::timeseries::{FloatValue, GridTimeseries, Time, TimeAxis};

use super::windows::{out_of_range, stored_index, GridTimeseriesWindow, ReadIndex};
use super::VariableSource;

/// A transformation context for grid aggregation and unit conversion.
///
/// This holds the information needed to transform grid data to scalar values
/// and apply unit conversions during read operations.
#[derive(Debug, Clone)]
pub struct ReadTransformInfo {
    /// The source grid type (finer resolution data)
    pub source_grid: GridType,
    /// The weights to use for aggregation (from Model's grid_weights)
    /// If None, use the grid's default weights
    pub weights: Option<Vec<f64>>,
    /// Unit conversion factor to apply when reading values.
    ///
    /// The stored value is multiplied by this factor to convert from the
    /// producer's unit to the consumer's unit.
    ///
    /// For example, if a schema stores emissions in GtC/yr but a component
    /// requests MtCO2/yr, the factor would be 3666.67.
    ///
    /// Default is 1.0 (no conversion).
    pub unit_conversion_factor: f64,
    /// The source type of this variable, determining how `get()` reads values.
    pub variable_source: VariableSource,
    /// Whether the variable is a point or an average, determining whether `interval()` exists.
    pub timeseries_type: TimeseriesType,
}

impl Default for ReadTransformInfo {
    fn default() -> Self {
        Self {
            source_grid: GridType::Scalar,
            weights: None,
            unit_conversion_factor: 1.0,
            variable_source: VariableSource::Exogenous,
            timeseries_type: TimeseriesType::Point,
        }
    }
}

impl ReadTransformInfo {
    /// Create a new ReadTransformInfo with just grid transformation.
    pub fn new(source_grid: GridType, weights: Option<Vec<f64>>) -> Self {
        Self {
            source_grid,
            weights,
            unit_conversion_factor: 1.0,
            variable_source: VariableSource::Exogenous,
            timeseries_type: TimeseriesType::Point,
        }
    }

    /// Create a new ReadTransformInfo with unit conversion only (no grid transform).
    pub fn with_unit_conversion(factor: f64) -> Self {
        Self {
            source_grid: GridType::Scalar,
            weights: None,
            unit_conversion_factor: factor,
            variable_source: VariableSource::Exogenous,
            timeseries_type: TimeseriesType::Point,
        }
    }

    /// Add unit conversion to an existing transform.
    pub fn with_conversion_factor(mut self, factor: f64) -> Self {
        self.unit_conversion_factor = factor;
        self
    }

    /// Set the variable source type for `get()` behavior.
    pub fn with_source(mut self, source: VariableSource) -> Self {
        self.variable_source = source;
        self
    }

    /// Set whether the variable is a point or an average.
    pub fn with_timeseries_type(mut self, timeseries_type: TimeseriesType) -> Self {
        self.timeseries_type = timeseries_type;
        self
    }
}

/// A scalar window that aggregates from a FourBox timeseries.
///
/// This provides the same API as `TimeseriesWindow` but reads from a FourBox
/// timeseries and aggregates to a scalar value on each access.
///
/// Optionally applies a unit conversion factor after aggregation.
#[derive(Debug)]
pub struct AggregatingFourBoxWindow<'a> {
    timeseries: &'a GridTimeseries<FloatValue, FourBoxGrid>,
    current_index: usize,
    name: &'a str,
    weights: Option<[f64; 4]>,
    /// Unit conversion factor applied to all returned values.
    unit_conversion_factor: f64,
    /// Source type determining get() behavior
    variable_source: VariableSource,
    timeseries_type: TimeseriesType,
}

impl<'a> AggregatingFourBoxWindow<'a> {
    pub fn new(
        timeseries: &'a GridTimeseries<FloatValue, FourBoxGrid>,
        current_index: usize,
        name: &'a str,
        weights: Option<Vec<f64>>,
    ) -> Self {
        Self::with_source(
            timeseries,
            current_index,
            name,
            weights,
            1.0,
            VariableSource::Exogenous,
        )
    }

    /// Create a new window with unit conversion.
    pub fn with_unit_conversion(
        timeseries: &'a GridTimeseries<FloatValue, FourBoxGrid>,
        current_index: usize,
        name: &'a str,
        weights: Option<Vec<f64>>,
        unit_conversion_factor: f64,
    ) -> Self {
        Self::with_source(
            timeseries,
            current_index,
            name,
            weights,
            unit_conversion_factor,
            VariableSource::Exogenous,
        )
    }

    /// Create a new window with all options including variable source.
    pub fn with_source(
        timeseries: &'a GridTimeseries<FloatValue, FourBoxGrid>,
        current_index: usize,
        name: &'a str,
        weights: Option<Vec<f64>>,
        unit_conversion_factor: f64,
        variable_source: VariableSource,
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
            name,
            weights,
            unit_conversion_factor,
            variable_source,
            timeseries_type: TimeseriesType::Point,
        }
    }

    fn aggregate(&self, values: &[FloatValue]) -> FloatValue {
        let aggregated = match &self.weights {
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
        };
        aggregated * self.unit_conversion_factor
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
            current: self.current_index,
            len: self.timeseries.len(),
            source: self.variable_source,
            timeseries_type: self.timeseries_type,
        }
    }

    /// Aggregated value the model wired this input to read during the step.
    ///
    /// See [`TimeseriesWindow::get()`](super::TimeseriesWindow::get) for which step's value this is.
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

    /// Number of values `history` can read.
    pub fn history_len(&self) -> usize {
        self.read_index().history_len()
    }

    fn value_at(&self, index: usize) -> FloatValue {
        let values = self
            .timeseries
            .at_time_index(index)
            .unwrap_or_else(|| out_of_range(self.name, index, self.timeseries.len()));
        self.aggregate(&values)
    }

    /// Value `n` steps before the one `get()` returns, so `history(0) == get()`.
    ///
    /// Returns `None` before the first step.
    pub fn history(&self, n: usize) -> Option<FloatValue> {
        self.read_index().history(n).map(|i| self.value_at(i))
    }

    /// Values at the start and end of the step.
    ///
    /// See [`TimeseriesWindow::interval()`](super::TimeseriesWindow::interval) for lagged inputs and errors.
    pub fn interval(&self) -> RSCMResult<(FloatValue, FloatValue)> {
        let (start, end) = self.read_index().interval()?;
        Ok((self.value_at(start), self.value_at(end)))
    }

    /// Aggregated value of the stored series at time `t`, past or future.
    ///
    /// See [`TimeseriesWindow::interpolate()`](super::TimeseriesWindow::interpolate)
    /// for nodes, the axis ends and upstream outputs.
    pub fn interpolate(&self, t: Time) -> RSCMResult<FloatValue> {
        match stored_index(self.time_axis(), t) {
            Some(index) => Ok(self.value_at(index)),
            None => Ok(self.aggregate(&self.timeseries.at_time_all(t)?)),
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

/// A scalar window that aggregates from a Hemispheric timeseries.
///
/// Optionally applies a unit conversion factor after aggregation.
#[derive(Debug)]
pub struct AggregatingHemisphericWindow<'a> {
    timeseries: &'a GridTimeseries<FloatValue, HemisphericGrid>,
    current_index: usize,
    name: &'a str,
    weights: Option<[f64; 2]>,
    /// Unit conversion factor applied to all returned values.
    unit_conversion_factor: f64,
    /// Source type determining get() behavior
    variable_source: VariableSource,
    timeseries_type: TimeseriesType,
}

impl<'a> AggregatingHemisphericWindow<'a> {
    pub fn new(
        timeseries: &'a GridTimeseries<FloatValue, HemisphericGrid>,
        current_index: usize,
        name: &'a str,
        weights: Option<Vec<f64>>,
    ) -> Self {
        Self::with_source(
            timeseries,
            current_index,
            name,
            weights,
            1.0,
            VariableSource::Exogenous,
        )
    }

    /// Create a new window with unit conversion.
    pub fn with_unit_conversion(
        timeseries: &'a GridTimeseries<FloatValue, HemisphericGrid>,
        current_index: usize,
        name: &'a str,
        weights: Option<Vec<f64>>,
        unit_conversion_factor: f64,
    ) -> Self {
        Self::with_source(
            timeseries,
            current_index,
            name,
            weights,
            unit_conversion_factor,
            VariableSource::Exogenous,
        )
    }

    /// Create a new window with all options including variable source.
    pub fn with_source(
        timeseries: &'a GridTimeseries<FloatValue, HemisphericGrid>,
        current_index: usize,
        name: &'a str,
        weights: Option<Vec<f64>>,
        unit_conversion_factor: f64,
        variable_source: VariableSource,
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
            name,
            weights,
            unit_conversion_factor,
            variable_source,
            timeseries_type: TimeseriesType::Point,
        }
    }

    fn aggregate(&self, values: &[FloatValue]) -> FloatValue {
        let aggregated = match &self.weights {
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
        };
        aggregated * self.unit_conversion_factor
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
            current: self.current_index,
            len: self.timeseries.len(),
            source: self.variable_source,
            timeseries_type: self.timeseries_type,
        }
    }

    /// Aggregated value the model wired this input to read during the step.
    ///
    /// See [`TimeseriesWindow::get()`](super::TimeseriesWindow::get) for which step's value this is.
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

    /// Number of values `history` can read.
    pub fn history_len(&self) -> usize {
        self.read_index().history_len()
    }

    fn value_at(&self, index: usize) -> FloatValue {
        let values = self
            .timeseries
            .at_time_index(index)
            .unwrap_or_else(|| out_of_range(self.name, index, self.timeseries.len()));
        self.aggregate(&values)
    }

    /// Value `n` steps before the one `get()` returns, so `history(0) == get()`.
    ///
    /// Returns `None` before the first step.
    pub fn history(&self, n: usize) -> Option<FloatValue> {
        self.read_index().history(n).map(|i| self.value_at(i))
    }

    /// Values at the start and end of the step.
    ///
    /// See [`TimeseriesWindow::interval()`](super::TimeseriesWindow::interval) for lagged inputs and errors.
    pub fn interval(&self) -> RSCMResult<(FloatValue, FloatValue)> {
        let (start, end) = self.read_index().interval()?;
        Ok((self.value_at(start), self.value_at(end)))
    }

    /// Aggregated value of the stored series at time `t`, past or future.
    ///
    /// See [`TimeseriesWindow::interpolate()`](super::TimeseriesWindow::interpolate)
    /// for nodes, the axis ends and upstream outputs.
    pub fn interpolate(&self, t: Time) -> RSCMResult<FloatValue> {
        match stored_index(self.time_axis(), t) {
            Some(index) => Ok(self.value_at(index)),
            None => Ok(self.aggregate(&self.timeseries.at_time_all(t)?)),
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
    /// Set whether the variable is a point or an average, which decides whether `interval()` exists.
    pub fn with_timeseries_type(self, timeseries_type: TimeseriesType) -> Self {
        match self {
            ScalarWindow::Direct(w) => {
                ScalarWindow::Direct(w.with_timeseries_type(timeseries_type))
            }
            ScalarWindow::FromFourBox(w) => {
                ScalarWindow::FromFourBox(w.with_timeseries_type(timeseries_type))
            }
            ScalarWindow::FromHemispheric(w) => {
                ScalarWindow::FromHemispheric(w.with_timeseries_type(timeseries_type))
            }
        }
    }

    /// Source of the variable, which decides the index `get()` reads.
    pub fn variable_source(&self) -> VariableSource {
        match self {
            ScalarWindow::Direct(w) => w.variable_source(),
            ScalarWindow::FromFourBox(w) => w.variable_source(),
            ScalarWindow::FromHemispheric(w) => w.variable_source(),
        }
    }

    /// Whether the variable is a point or an average.
    pub fn timeseries_type(&self) -> TimeseriesType {
        match self {
            ScalarWindow::Direct(w) => w.timeseries_type(),
            ScalarWindow::FromFourBox(w) => w.timeseries_type(),
            ScalarWindow::FromHemispheric(w) => w.timeseries_type(),
        }
    }

    pub(crate) fn time_axis(&self) -> &TimeAxis {
        match self {
            ScalarWindow::Direct(w) => w.time_axis(),
            ScalarWindow::FromFourBox(w) => w.time_axis(),
            ScalarWindow::FromHemispheric(w) => w.time_axis(),
        }
    }

    /// Value the model wired this input to read during the step.
    ///
    /// See [`TimeseriesWindow::get()`](super::TimeseriesWindow::get) for which step's value this is.
    pub fn get(&self) -> FloatValue {
        match self {
            ScalarWindow::Direct(w) => w.get(),
            ScalarWindow::FromFourBox(w) => w.get(),
            ScalarWindow::FromHemispheric(w) => w.get(),
        }
    }

    /// Time of the value `get()` returns.
    pub fn get_time(&self) -> Time {
        match self {
            ScalarWindow::Direct(w) => w.get_time(),
            ScalarWindow::FromFourBox(w) => w.get_time(),
            ScalarWindow::FromHemispheric(w) => w.get_time(),
        }
    }

    /// Value `n` steps before the one `get()` returns, so `history(0) == get()`.
    ///
    /// Returns `None` before the first step.
    pub fn history(&self, n: usize) -> Option<FloatValue> {
        match self {
            ScalarWindow::Direct(w) => w.history(n),
            ScalarWindow::FromFourBox(w) => w.history(n),
            ScalarWindow::FromHemispheric(w) => w.history(n),
        }
    }

    /// Number of values `history` can read.
    pub fn history_len(&self) -> usize {
        match self {
            ScalarWindow::Direct(w) => w.history_len(),
            ScalarWindow::FromFourBox(w) => w.history_len(),
            ScalarWindow::FromHemispheric(w) => w.history_len(),
        }
    }

    /// Values at the start and end of the step.
    ///
    /// See [`TimeseriesWindow::interval()`](super::TimeseriesWindow::interval) for lagged inputs and errors.
    pub fn interval(&self) -> RSCMResult<(FloatValue, FloatValue)> {
        match self {
            ScalarWindow::Direct(w) => w.interval(),
            ScalarWindow::FromFourBox(w) => w.interval(),
            ScalarWindow::FromHemispheric(w) => w.interval(),
        }
    }

    /// Value of the stored series at time `t`, past or future.
    ///
    /// See [`TimeseriesWindow::interpolate()`](super::TimeseriesWindow::interpolate)
    /// for nodes, the axis ends and upstream outputs.
    pub fn interpolate(&self, t: Time) -> RSCMResult<FloatValue> {
        match self {
            ScalarWindow::Direct(w) => w.interpolate(t),
            ScalarWindow::FromFourBox(w) => w.interpolate(t),
            ScalarWindow::FromHemispheric(w) => w.interpolate(t),
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

/// A window that aggregates from a FourBox timeseries to Hemispheric output.
///
/// This is used when reading a FourBox timeseries but the component wants Hemispheric data.
/// Optionally applies a unit conversion factor after aggregation.
#[derive(Debug)]
pub struct AggregatingFourBoxToHemisphericWindow<'a> {
    timeseries: &'a GridTimeseries<FloatValue, FourBoxGrid>,
    current_index: usize,
    name: &'a str,
    /// Unit conversion factor applied to all returned values.
    unit_conversion_factor: f64,
    /// Source type determining get() behavior
    variable_source: VariableSource,
    timeseries_type: TimeseriesType,
}

impl<'a> AggregatingFourBoxToHemisphericWindow<'a> {
    pub fn new(
        timeseries: &'a GridTimeseries<FloatValue, FourBoxGrid>,
        current_index: usize,
        name: &'a str,
    ) -> Self {
        Self::with_source(
            timeseries,
            current_index,
            name,
            1.0,
            VariableSource::Exogenous,
        )
    }

    /// Create a new window with unit conversion.
    pub fn with_unit_conversion(
        timeseries: &'a GridTimeseries<FloatValue, FourBoxGrid>,
        current_index: usize,
        name: &'a str,
        unit_conversion_factor: f64,
    ) -> Self {
        Self::with_source(
            timeseries,
            current_index,
            name,
            unit_conversion_factor,
            VariableSource::Exogenous,
        )
    }

    /// Create a new window with all options including variable source.
    pub fn with_source(
        timeseries: &'a GridTimeseries<FloatValue, FourBoxGrid>,
        current_index: usize,
        name: &'a str,
        unit_conversion_factor: f64,
        variable_source: VariableSource,
    ) -> Self {
        Self {
            timeseries,
            current_index,
            name,
            unit_conversion_factor,
            variable_source,
            timeseries_type: TimeseriesType::Point,
        }
    }

    fn aggregate_to_hemispheric(&self, values: &[FloatValue]) -> [FloatValue; 2] {
        // FourBox: [NorthernOcean, NorthernLand, SouthernOcean, SouthernLand]
        // Hemispheric: [Northern, Southern]
        // Average ocean+land for each hemisphere, then apply unit conversion
        let northern = (values[0] + values[1]) / 2.0 * self.unit_conversion_factor;
        let southern = (values[2] + values[3]) / 2.0 * self.unit_conversion_factor;
        [northern, southern]
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
            current: self.current_index,
            len: self.timeseries.len(),
            source: self.variable_source,
            timeseries_type: self.timeseries_type,
        }
    }

    /// Time of the values `get_all()` returns.
    pub fn get_time(&self) -> Time {
        let index = self.read_index().get();
        self.time_axis()
            .at(index)
            .unwrap_or_else(|| out_of_range(self.name, index, self.timeseries.len()))
    }

    /// Number of values `history` can read.
    pub fn history_len(&self) -> usize {
        self.read_index().history_len()
    }

    fn values_at(&self, index: usize) -> Vec<FloatValue> {
        let values = self
            .timeseries
            .at_time_index(index)
            .unwrap_or_else(|| out_of_range(self.name, index, self.timeseries.len()));
        self.aggregate_to_hemispheric(&values).to_vec()
    }

    /// Regional values `n` steps before those `get_all()` returns, so `history(0) == get_all()`.
    ///
    /// Returns `None` before the first step.
    pub fn history(&self, n: usize) -> Option<Vec<FloatValue>> {
        self.read_index().history(n).map(|i| self.values_at(i))
    }

    /// Regional values at the start and end of the step.
    ///
    /// See [`TimeseriesWindow::interval()`](super::TimeseriesWindow::interval) for lagged inputs and errors.
    pub fn interval(&self) -> RSCMResult<(Vec<FloatValue>, Vec<FloatValue>)> {
        let (start, end) = self.read_index().interval()?;
        Ok((self.values_at(start), self.values_at(end)))
    }

    /// Regional values of the stored series at time `t`, past or future.
    ///
    /// See [`TimeseriesWindow::interpolate()`](super::TimeseriesWindow::interpolate)
    /// for nodes, the axis ends and upstream outputs.
    pub fn interpolate_all(&self, t: Time) -> RSCMResult<Vec<FloatValue>> {
        match stored_index(self.time_axis(), t) {
            Some(index) => Ok(self.values_at(index)),
            None => Ok(self
                .aggregate_to_hemispheric(&self.timeseries.at_time_all(t)?)
                .to_vec()),
        }
    }

    /// A single region's value that the model wired this input to read during the step.
    ///
    /// See [`TimeseriesWindow::get()`](super::TimeseriesWindow::get) for which step's value this is.
    pub fn get(&self, region: HemisphericRegion) -> FloatValue {
        self.get_all()[region as usize]
    }

    /// All regional values that the model wired this input to read during the step.
    pub fn get_all(&self) -> Vec<FloatValue> {
        self.values_at(self.read_index().get())
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
    /// Set whether the variable is a point or an average, which decides whether `interval()` exists.
    pub fn with_timeseries_type(self, timeseries_type: TimeseriesType) -> Self {
        match self {
            HemisphericWindow::Direct(w) => {
                HemisphericWindow::Direct(w.with_timeseries_type(timeseries_type))
            }
            HemisphericWindow::FromFourBox(w) => {
                HemisphericWindow::FromFourBox(w.with_timeseries_type(timeseries_type))
            }
        }
    }

    /// Source of the variable, which decides the index `get()` reads.
    pub fn variable_source(&self) -> VariableSource {
        match self {
            HemisphericWindow::Direct(w) => w.variable_source(),
            HemisphericWindow::FromFourBox(w) => w.variable_source(),
        }
    }

    /// Whether the variable is a point or an average.
    pub fn timeseries_type(&self) -> TimeseriesType {
        match self {
            HemisphericWindow::Direct(w) => w.timeseries_type(),
            HemisphericWindow::FromFourBox(w) => w.timeseries_type(),
        }
    }

    pub(crate) fn time_axis(&self) -> &TimeAxis {
        match self {
            HemisphericWindow::Direct(w) => w.time_axis(),
            HemisphericWindow::FromFourBox(w) => w.time_axis(),
        }
    }

    /// A single region's value that the model wired this input to read during the step.
    pub fn get(&self, region: HemisphericRegion) -> FloatValue {
        match self {
            HemisphericWindow::Direct(w) => w.get(region),
            HemisphericWindow::FromFourBox(w) => w.get(region),
        }
    }

    /// All regional values that the model wired this input to read during the step.
    pub fn get_all(&self) -> Vec<FloatValue> {
        match self {
            HemisphericWindow::Direct(w) => w.get_all(),
            HemisphericWindow::FromFourBox(w) => w.get_all(),
        }
    }

    /// Time of the values `get_all()` returns.
    pub fn get_time(&self) -> Time {
        match self {
            HemisphericWindow::Direct(w) => w.get_time(),
            HemisphericWindow::FromFourBox(w) => w.get_time(),
        }
    }

    /// Regional values `n` steps before those `get_all()` returns, so `history(0) == get_all()`.
    ///
    /// Returns `None` before the first step.
    pub fn history(&self, n: usize) -> Option<Vec<FloatValue>> {
        match self {
            HemisphericWindow::Direct(w) => w.history(n),
            HemisphericWindow::FromFourBox(w) => w.history(n),
        }
    }

    /// Number of values `history` can read.
    pub fn history_len(&self) -> usize {
        match self {
            HemisphericWindow::Direct(w) => w.history_len(),
            HemisphericWindow::FromFourBox(w) => w.history_len(),
        }
    }

    /// Regional values at the start and end of the step.
    ///
    /// See [`TimeseriesWindow::interval()`](super::TimeseriesWindow::interval) for lagged inputs and errors.
    pub fn interval(&self) -> RSCMResult<(Vec<FloatValue>, Vec<FloatValue>)> {
        match self {
            HemisphericWindow::Direct(w) => w.interval(),
            HemisphericWindow::FromFourBox(w) => w.interval(),
        }
    }

    /// Regional values of the stored series at time `t`, past or future.
    ///
    /// See [`TimeseriesWindow::interpolate()`](super::TimeseriesWindow::interpolate)
    /// for nodes, the axis ends and upstream outputs.
    pub fn interpolate_all(&self, t: Time) -> RSCMResult<Vec<FloatValue>> {
        match self {
            HemisphericWindow::Direct(w) => w.interpolate_all(t),
            HemisphericWindow::FromFourBox(w) => w.interpolate_all(t),
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
    fn test_aggregating_four_box_window_get_default_weights() {
        let ts = create_four_box_timeseries();
        // Index 1 = year 2001, values [11, 21, 31, 41]
        let window = AggregatingFourBoxWindow::new(&ts, 1, "test", None);

        // With equal weights: (11 + 21 + 31 + 41) / 4 = 26.0
        assert_eq!(window.get(), 26.0);
    }

    #[test]
    fn test_aggregating_four_box_window_get_custom_weights() {
        let ts = create_four_box_timeseries();
        // Custom weights that sum to 1.0
        let weights = vec![0.5, 0.2, 0.2, 0.1];
        let window = AggregatingFourBoxWindow::new(&ts, 1, "test", Some(weights));

        // With custom weights: 11*0.5 + 21*0.2 + 31*0.2 + 41*0.1 = 5.5 + 4.2 + 6.2 + 4.1 = 20.0
        assert_eq!(window.get(), 20.0);
    }

    #[test]
    fn test_aggregating_four_box_window_get_upstream() {
        let ts = create_four_box_timeseries();
        let window = AggregatingFourBoxWindow::with_source(
            &ts,
            1,
            "test",
            None,
            1.0,
            VariableSource::UpstreamOutput,
        );

        // Index 2 values [12, 22, 32, 42], mean = 27.0
        assert_eq!(window.get(), 27.0);
    }

    #[test]
    #[should_panic(expected = "Variable 'Effective Radiative Forcing' has 3 values, so index 3")]
    fn test_aggregating_four_box_window_get_upstream_at_last_index_panics() {
        let ts = create_four_box_timeseries();
        let window = AggregatingFourBoxWindow::with_source(
            &ts,
            2,
            "Effective Radiative Forcing",
            None,
            1.0,
            VariableSource::UpstreamOutput,
        );
        let _ = window.get();
    }

    #[test]
    fn test_aggregating_four_box_window_history() {
        let ts = create_four_box_timeseries();
        let window = AggregatingFourBoxWindow::new(&ts, 1, "test", None);

        // Index 0 values [10, 20, 30, 40], mean = 25.0
        assert_eq!(window.history(1), Some(25.0));

        let first = AggregatingFourBoxWindow::new(&ts, 0, "test", None);
        assert_eq!(first.history(1), None);
    }

    #[test]
    fn test_aggregating_four_box_window_metadata() {
        let ts = create_four_box_timeseries();
        let window = AggregatingFourBoxWindow::new(&ts, 1, "test", None);

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
        let window = AggregatingFourBoxWindow::new(&ts, 0, "test", Some(weights));

        // NaN is skipped: 10*0.25 + 30*0.25 + 40*0.25 = 2.5 + 7.5 + 10.0 = 20.0
        assert_eq!(window.get(), 20.0);
    }

    // =========================================================================
    // AggregatingHemisphericWindow tests (Hemispheric -> Scalar aggregation)
    // =========================================================================

    #[test]
    fn test_aggregating_hemispheric_window_get_default_weights() {
        let ts = create_hemispheric_timeseries();
        let window = AggregatingHemisphericWindow::new(&ts, 1, "test", None);

        // Equal weights: (110 + 220) / 2 = 165.0
        assert_eq!(window.get(), 165.0);
    }

    #[test]
    fn test_aggregating_hemispheric_window_get_custom_weights() {
        let ts = create_hemispheric_timeseries();
        let weights = vec![0.7, 0.3];
        let window = AggregatingHemisphericWindow::new(&ts, 1, "test", Some(weights));

        // 110*0.7 + 220*0.3 = 77.0 + 66.0 = 143.0
        assert_eq!(window.get(), 143.0);
    }

    #[test]
    fn test_aggregating_hemispheric_window_get_upstream() {
        let ts = create_hemispheric_timeseries();
        let window = AggregatingHemisphericWindow::with_source(
            &ts,
            1,
            "test",
            None,
            1.0,
            VariableSource::UpstreamOutput,
        );

        // Index 2 values [120, 240], mean = 180.0
        assert_eq!(window.get(), 180.0);
    }

    #[test]
    fn test_aggregating_hemispheric_window_history() {
        let ts = create_hemispheric_timeseries();
        let window = AggregatingHemisphericWindow::new(&ts, 1, "test", None);

        // Index 0 values [100, 200], mean = 150.0
        assert_eq!(window.history(1), Some(150.0));
    }

    #[test]
    fn test_aggregating_hemispheric_window_metadata() {
        let ts = create_hemispheric_timeseries();
        let window = AggregatingHemisphericWindow::new(&ts, 1, "test", None);

        assert_eq!(window.len(), 3);
        assert!(!window.is_empty());
    }

    // =========================================================================
    // AggregatingFourBoxToHemisphericWindow tests (FourBox -> Hemispheric)
    // =========================================================================

    #[test]
    fn test_aggregating_four_box_to_hemispheric_get_all() {
        let ts = create_four_box_timeseries();
        let window = AggregatingFourBoxToHemisphericWindow::new(&ts, 1, "test");

        // Index 1 values [11, 21, 31, 41]
        // Northern = (11 + 21) / 2 = 16.0
        // Southern = (31 + 41) / 2 = 36.0
        assert_eq!(window.get_all(), vec![16.0, 36.0]);
        assert_eq!(window.get(HemisphericRegion::Northern), 16.0);
        assert_eq!(window.get(HemisphericRegion::Southern), 36.0);
    }

    #[test]
    fn test_aggregating_four_box_to_hemispheric_get_upstream() {
        let ts = create_four_box_timeseries();
        let window = AggregatingFourBoxToHemisphericWindow::with_source(
            &ts,
            1,
            "test",
            1.0,
            VariableSource::UpstreamOutput,
        );

        // Index 2 values [12, 22, 32, 42]
        // Northern = (12 + 22) / 2 = 17.0
        // Southern = (32 + 42) / 2 = 37.0
        assert_eq!(window.get_all(), vec![17.0, 37.0]);
        assert_eq!(window.get(HemisphericRegion::Southern), 37.0);
    }

    #[test]
    fn test_aggregating_four_box_to_hemispheric_metadata() {
        let ts = create_four_box_timeseries();
        let window = AggregatingFourBoxToHemisphericWindow::new(&ts, 1, "test");

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
        let inner = super::super::windows::TimeseriesWindow::new(&ts, 1, "test");
        let window = ScalarWindow::Direct(inner);

        assert_eq!(window.get(), 200.0);
        assert_eq!(window.history(1), Some(100.0));
        assert_eq!(window.get_time(), 2001.0);
        assert_eq!(window.len(), 3);
        assert!(!window.is_empty());
    }

    #[test]
    fn test_scalar_window_from_four_box_variant() {
        let ts = create_four_box_timeseries();
        let inner = AggregatingFourBoxWindow::new(&ts, 1, "test", None);
        let window = ScalarWindow::FromFourBox(inner);

        // Same behavior as AggregatingFourBoxWindow
        assert_eq!(window.get(), 26.0);
        assert_eq!(window.history(1), Some(25.0));
        assert_eq!(window.get_time(), 2001.0);
        assert_eq!(window.len(), 3);
    }

    #[test]
    fn test_scalar_window_from_hemispheric_variant() {
        let ts = create_hemispheric_timeseries();
        let inner = AggregatingHemisphericWindow::new(&ts, 1, "test", None);
        let window = ScalarWindow::FromHemispheric(inner);

        // Same behavior as AggregatingHemisphericWindow
        assert_eq!(window.get(), 165.0);
        assert_eq!(window.history(1), Some(150.0));
        assert_eq!(window.get_time(), 2001.0);
    }

    // =========================================================================
    // HemisphericWindow enum tests (unified hemispheric interface)
    // =========================================================================

    #[test]
    fn test_hemispheric_window_direct_variant() {
        let ts = create_hemispheric_timeseries();
        let inner = GridTimeseriesWindow::new(&ts, 1, "test");
        let window = HemisphericWindow::Direct(inner);

        assert_eq!(window.get(HemisphericRegion::Northern), 110.0);
        assert_eq!(window.get(HemisphericRegion::Southern), 220.0);
        assert_eq!(window.get_all(), vec![110.0, 220.0]);
        assert_eq!(window.get_time(), 2001.0);
        assert_eq!(window.len(), 3);
        assert!(!window.is_empty());
    }

    #[test]
    fn test_hemispheric_window_from_four_box_variant() {
        let ts = create_four_box_timeseries();
        let inner = AggregatingFourBoxToHemisphericWindow::new(&ts, 1, "test");
        let window = HemisphericWindow::FromFourBox(inner);

        // Aggregated values: Northern=(11+21)/2=16, Southern=(31+41)/2=36
        assert_eq!(window.get(HemisphericRegion::Northern), 16.0);
        assert_eq!(window.get(HemisphericRegion::Southern), 36.0);
        assert_eq!(window.get_all(), vec![16.0, 36.0]);
        // Index 0: Northern=(10+20)/2=15, Southern=(30+40)/2=35
        assert_eq!(window.history(1), Some(vec![15.0, 35.0]));
        assert_eq!(window.get_time(), 2001.0);
    }

    // =========================================================================
    // Unit conversion tests for aggregating windows
    // =========================================================================

    #[test]
    fn test_aggregating_four_box_window_unit_conversion() {
        let ts = create_four_box_timeseries();
        // Index 1 values [11, 21, 31, 41], mean = 26.0
        // With 2.0 conversion factor: 52.0
        let window = AggregatingFourBoxWindow::with_unit_conversion(&ts, 1, "test", None, 2.0);

        assert_eq!(window.get(), 52.0);
        assert_eq!(window.history(1), Some(50.0)); // 25.0 * 2.0
    }

    #[test]
    fn test_aggregating_four_box_window_unit_conversion_with_custom_weights() {
        let ts = create_four_box_timeseries();
        // Custom weights that sum to 1.0
        let weights = vec![0.5, 0.2, 0.2, 0.1];
        // Index 1: 11*0.5 + 21*0.2 + 31*0.2 + 41*0.1 = 5.5 + 4.2 + 6.2 + 4.1 = 20.0
        // With 3.0 conversion factor: 60.0
        let window =
            AggregatingFourBoxWindow::with_unit_conversion(&ts, 1, "test", Some(weights), 3.0);

        assert_eq!(window.get(), 60.0);
    }

    #[test]
    fn test_aggregating_hemispheric_window_unit_conversion() {
        let ts = create_hemispheric_timeseries();
        // Index 1: [110, 220], mean = 165.0
        // With 0.5 conversion factor: 82.5
        let window = AggregatingHemisphericWindow::with_unit_conversion(&ts, 1, "test", None, 0.5);

        assert_eq!(window.get(), 82.5);
        assert_eq!(window.history(1), Some(75.0)); // 150.0 * 0.5
    }

    #[test]
    fn test_aggregating_four_box_to_hemispheric_unit_conversion() {
        let ts = create_four_box_timeseries();
        // Index 1 values [11, 21, 31, 41]
        // Northern = (11 + 21) / 2 = 16.0
        // Southern = (31 + 41) / 2 = 36.0
        // With 2.0 conversion factor: [32.0, 72.0]
        let window =
            AggregatingFourBoxToHemisphericWindow::with_unit_conversion(&ts, 1, "test", 2.0);

        assert_eq!(window.get(HemisphericRegion::Northern), 32.0);
        assert_eq!(window.get(HemisphericRegion::Southern), 72.0);
        assert_eq!(window.get_all(), vec![32.0, 72.0]);
    }

    // =========================================================================
    // get_time(), history(), history_len() and interval() on aggregating windows
    // =========================================================================

    #[test]
    fn test_aggregating_four_box_window_new_accessors() {
        let ts = create_four_box_timeseries();
        let window =
            |source| AggregatingFourBoxWindow::with_source(&ts, 1, "test", None, 2.0, source);

        let exogenous = window(VariableSource::Exogenous);
        assert_eq!(exogenous.get_time(), 2001.0);
        assert_eq!(exogenous.history(0), Some(52.0));
        assert_eq!(exogenous.history(1), Some(50.0));
        assert_eq!(exogenous.history(2), None);
        assert_eq!(exogenous.history_len(), 2);
        assert_eq!(exogenous.interval().unwrap(), (52.0, 54.0));
        assert_eq!(exogenous.interpolate(2002.0).unwrap(), 54.0);

        let upstream = window(VariableSource::UpstreamOutput);
        assert_eq!(upstream.get_time(), 2002.0);
        assert_eq!(upstream.history(0), Some(54.0));
        assert_eq!(upstream.history_len(), 3);
        assert_eq!(upstream.interval().unwrap(), (52.0, 54.0));

        assert_eq!(
            window(VariableSource::Lagged).interval().unwrap(),
            (50.0, 52.0)
        );
        assert!(window(VariableSource::OwnState).interval().is_err());
        assert!(window(VariableSource::Exogenous)
            .with_timeseries_type(TimeseriesType::Average)
            .interval()
            .is_err());
    }

    #[test]
    fn test_aggregating_hemispheric_window_new_accessors() {
        let ts = create_hemispheric_timeseries();
        let window =
            |source| AggregatingHemisphericWindow::with_source(&ts, 1, "test", None, 1.0, source);

        let exogenous = window(VariableSource::Exogenous);
        assert_eq!(exogenous.get_time(), 2001.0);
        assert_eq!(exogenous.history(1), Some(150.0));
        assert_eq!(exogenous.history_len(), 2);
        assert_eq!(exogenous.interval().unwrap(), (165.0, 180.0));

        let upstream = window(VariableSource::UpstreamOutput);
        assert_eq!(upstream.get_time(), 2002.0);
        assert_eq!(upstream.history(0), Some(180.0));

        assert_eq!(
            window(VariableSource::Lagged).interval().unwrap(),
            (150.0, 165.0)
        );
        assert!(window(VariableSource::OwnState).interval().is_err());
        assert_eq!(
            window(VariableSource::Lagged).interpolate(1999.0).unwrap(),
            150.0
        );
    }

    #[test]
    fn test_aggregating_four_box_to_hemispheric_new_accessors() {
        let ts = create_four_box_timeseries();
        let window = |source| {
            AggregatingFourBoxToHemisphericWindow::with_source(&ts, 1, "test", 2.0, source)
        };

        let exogenous = window(VariableSource::Exogenous);
        assert_eq!(exogenous.get_time(), 2001.0);
        assert_eq!(exogenous.history(0), Some(exogenous.get_all()));
        assert_eq!(exogenous.history(1), Some(vec![30.0, 70.0]));
        assert_eq!(exogenous.history_len(), 2);
        assert_eq!(
            exogenous.interval().unwrap(),
            (vec![32.0, 72.0], vec![34.0, 74.0])
        );
        assert_eq!(exogenous.interpolate_all(2010.0).unwrap(), vec![34.0, 74.0]);

        let upstream = window(VariableSource::UpstreamOutput);
        assert_eq!(upstream.get_time(), 2002.0);
        assert_eq!(upstream.history(0), Some(vec![34.0, 74.0]));

        assert_eq!(
            window(VariableSource::Lagged).interval().unwrap(),
            (vec![30.0, 70.0], vec![32.0, 72.0])
        );
        assert!(window(VariableSource::OwnState).interval().is_err());
    }

    #[test]
    fn test_window_enums_delegate_new_accessors() {
        let four_box = create_four_box_timeseries();
        let scalar = ScalarWindow::FromFourBox(AggregatingFourBoxWindow::with_source(
            &four_box,
            1,
            "test",
            None,
            1.0,
            VariableSource::UpstreamOutput,
        ))
        .with_timeseries_type(TimeseriesType::Average);
        assert_eq!(scalar.variable_source(), VariableSource::UpstreamOutput);
        assert_eq!(scalar.timeseries_type(), TimeseriesType::Average);
        assert_eq!(scalar.get_time(), 2002.0);
        assert_eq!(scalar.history(0), Some(27.0));
        assert_eq!(scalar.history_len(), 3);
        assert!(scalar.interval().is_err());
        assert_eq!(scalar.interpolate(2000.0).unwrap(), 25.0);

        let hemispheric = create_hemispheric_timeseries();
        let window = HemisphericWindow::Direct(GridTimeseriesWindow::with_source(
            &hemispheric,
            1,
            "test",
            1.0,
            VariableSource::Lagged,
        ));
        assert_eq!(window.get_time(), 2001.0);
        assert_eq!(window.history(1), Some(vec![100.0, 200.0]));
        assert_eq!(window.history_len(), 2);
        assert_eq!(
            window.interval().unwrap(),
            (vec![100.0, 200.0], vec![110.0, 220.0])
        );
        assert_eq!(window.interpolate_all(2002.0).unwrap(), vec![120.0, 240.0]);
    }

    #[test]
    fn test_read_transform_info_default() {
        let info = ReadTransformInfo::default();
        assert_eq!(info.source_grid, GridType::Scalar);
        assert!(info.weights.is_none());
        assert_eq!(info.unit_conversion_factor, 1.0);
        assert_eq!(info.variable_source, VariableSource::Exogenous);
        assert_eq!(info.timeseries_type, TimeseriesType::Point);
    }

    #[test]
    fn test_read_transform_info_with_unit_conversion() {
        let info = ReadTransformInfo::with_unit_conversion(3.667);
        assert_eq!(info.source_grid, GridType::Scalar);
        assert!(info.weights.is_none());
        assert!((info.unit_conversion_factor - 3.667).abs() < f64::EPSILON);
    }

    #[test]
    fn test_read_transform_info_builder() {
        let info = ReadTransformInfo::new(GridType::FourBox, Some(vec![0.25, 0.25, 0.25, 0.25]))
            .with_conversion_factor(2.5);

        assert_eq!(info.source_grid, GridType::FourBox);
        assert!(info.weights.is_some());
        assert_eq!(info.unit_conversion_factor, 2.5);
    }
}
