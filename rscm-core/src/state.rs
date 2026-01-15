use crate::errors::RSCMResult;
use crate::spatial::{
    FourBoxGrid, FourBoxRegion, HemisphericGrid, HemisphericRegion, ScalarGrid, ScalarRegion,
    SpatialGrid,
};
use crate::timeseries::{FloatValue, GridTimeseries, Time, Timeseries};
use crate::timeseries_collection::{TimeseriesData, TimeseriesItem, VariableType};
use crate::variable::PreindustrialValue;
use ndarray::ArrayView1;
use num::Float;
use std::collections::HashMap;

/// A zero-cost view into a scalar timeseries at a specific time index.
///
/// `TimeseriesWindow` provides efficient access to current, historical, and interpolated
/// values without copying data. This is the primary way components access their input
/// variables.
///
/// # Examples
///
/// ```ignore
/// fn solve(&self, inputs: MyComponentInputs) -> MyComponentOutputs {
///     let current_co2 = inputs.emissions_co2.current();
///     let previous_co2 = inputs.emissions_co2.previous();
///     let last_5 = inputs.emissions_co2.last_n(5);
///     let derivative = (current_co2 - previous_co2.unwrap_or(current_co2)) / dt;
///     // ...
/// }
/// ```
#[derive(Debug)]
pub struct TimeseriesWindow<'a> {
    timeseries: &'a Timeseries<FloatValue>,
    current_index: usize,
    current_time: Time,
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
        }
    }

    /// Get the value at the current timestep.
    ///
    /// This is the most common operation - getting the current value of an input.
    pub fn current(&self) -> FloatValue {
        self.timeseries
            .at(self.current_index, ScalarRegion::Global)
            .expect("Current index out of bounds")
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
            self.timeseries.at(index as usize, ScalarRegion::Global)
        }
    }

    /// Get the last N values as an array view, ending at the current timestep.
    ///
    /// This is useful for computing moving averages, derivatives, or any operation
    /// that needs historical context.
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

    /// Interpolate the value at an arbitrary time point.
    ///
    /// Uses the timeseries's interpolation strategy to compute the value.
    /// This is useful for sub-timestep calculations or when comparing with
    /// observational data at non-model times.
    pub fn interpolate(&self, t: Time) -> RSCMResult<FloatValue> {
        self.timeseries.at_time(t, ScalarRegion::Global)
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
/// # Examples
///
/// ```ignore
/// fn solve(&self, inputs: MyComponentInputs) -> MyComponentOutputs {
///     // Access individual regions
///     let northern_ocean = inputs.temperature.current(FourBoxRegion::NorthernOcean);
///
///     // Get all regions at once
///     let all_temps = inputs.temperature.current_all();
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
        }
    }

    /// Get all regional values at the current timestep.
    pub fn current_all(&self) -> Vec<FloatValue> {
        self.timeseries
            .at_time_index(self.current_index)
            .expect("Current index out of bounds")
    }

    /// Get all regional values at the previous timestep.
    pub fn previous_all(&self) -> Option<Vec<FloatValue>> {
        if self.current_index == 0 {
            None
        } else {
            self.timeseries.at_time_index(self.current_index - 1)
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
        self.timeseries.at_time_all(t)
    }
}

/// Type-safe accessors for FourBoxGrid windows
impl<'a> GridTimeseriesWindow<'a, FourBoxGrid> {
    /// Get a single region's value at the current timestep.
    pub fn current(&self, region: FourBoxRegion) -> FloatValue {
        self.timeseries
            .at(self.current_index, region)
            .expect("Current index out of bounds")
    }

    /// Get a single region's value at the previous timestep.
    pub fn previous(&self, region: FourBoxRegion) -> Option<FloatValue> {
        if self.current_index == 0 {
            None
        } else {
            self.timeseries.at(self.current_index - 1, region)
        }
    }

    /// Get the global aggregate at the current timestep.
    ///
    /// Uses the grid's weights to compute a weighted average of all regions.
    pub fn current_global(&self) -> FloatValue {
        let values = self.current_all();
        self.timeseries.grid().aggregate_global(&values)
    }

    /// Get the global aggregate at the previous timestep.
    pub fn previous_global(&self) -> Option<FloatValue> {
        self.previous_all()
            .map(|values| self.timeseries.grid().aggregate_global(&values))
    }

    /// Interpolate a single region's value at an arbitrary time.
    pub fn interpolate(&self, t: Time, region: FourBoxRegion) -> RSCMResult<FloatValue> {
        self.timeseries.at_time(t, region)
    }
}

/// Type-safe accessors for HemisphericGrid windows
impl<'a> GridTimeseriesWindow<'a, HemisphericGrid> {
    /// Get a single region's value at the current timestep.
    pub fn current(&self, region: HemisphericRegion) -> FloatValue {
        self.timeseries
            .at(self.current_index, region)
            .expect("Current index out of bounds")
    }

    /// Get a single region's value at the previous timestep.
    pub fn previous(&self, region: HemisphericRegion) -> Option<FloatValue> {
        if self.current_index == 0 {
            None
        } else {
            self.timeseries.at(self.current_index - 1, region)
        }
    }

    /// Get the global aggregate at the current timestep.
    pub fn current_global(&self) -> FloatValue {
        let values = self.current_all();
        self.timeseries.grid().aggregate_global(&values)
    }

    /// Get the global aggregate at the previous timestep.
    pub fn previous_global(&self) -> Option<FloatValue> {
        self.previous_all()
            .map(|values| self.timeseries.grid().aggregate_global(&values))
    }

    /// Interpolate a single region's value at an arbitrary time.
    pub fn interpolate(&self, t: Time, region: HemisphericRegion) -> RSCMResult<FloatValue> {
        self.timeseries.at_time(t, region)
    }
}

/// Type-safe accessors for ScalarGrid windows (convenience wrapper)
impl<'a> GridTimeseriesWindow<'a, ScalarGrid> {
    /// Get the current scalar value.
    pub fn current(&self) -> FloatValue {
        self.timeseries
            .at(self.current_index, ScalarRegion::Global)
            .expect("Current index out of bounds")
    }

    /// Get the previous scalar value.
    pub fn previous(&self) -> Option<FloatValue> {
        if self.current_index == 0 {
            None
        } else {
            self.timeseries
                .at(self.current_index - 1, ScalarRegion::Global)
        }
    }

    /// Interpolate the value at an arbitrary time.
    pub fn interpolate(&self, t: Time) -> RSCMResult<FloatValue> {
        self.timeseries.at_time(t, ScalarRegion::Global)
    }
}

// =============================================================================
// Typed Output Slices
// =============================================================================

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
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
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

// =============================================================================
// State Value Types
// =============================================================================

/// Represents a value that can be either scalar or spatially-resolved
#[derive(Debug, Clone, PartialEq)]
pub enum StateValue {
    /// A single scalar value (global average or non-spatial variable)
    Scalar(FloatValue),
    /// Multiple regional values from a grid timeseries
    Grid(Vec<FloatValue>),
}

/// Input state for a component
///
/// A state is a collection of values
/// that can be used to represent the state of a system at a given time.
///
/// This is very similar to a Hashmap (with likely worse performance),
/// but provides strong type separation.
#[derive(Debug, Clone)]
pub struct InputState<'a> {
    current_time: Time,
    state: Vec<&'a TimeseriesItem>,
}

impl<'a> InputState<'a> {
    pub fn build(values: Vec<&'a TimeseriesItem>, current_time: Time) -> Self {
        Self {
            current_time,
            state: values,
        }
    }

    pub fn empty() -> Self {
        Self {
            current_time: Time::nan(),
            state: vec![],
        }
    }

    /// Get the latest scalar value for a variable
    ///
    /// This method assumes the variable is scalar (single global value).
    /// For grid variables, use `get_latest_value()` or `get_global()`.
    ///
    /// # Panics
    /// Panics if the variable is not found or is not scalar.
    ///
    /// # Deprecated
    /// Use `get_scalar_window(name).current()` for typed access with history support.
    #[deprecated(
        since = "0.3.0",
        note = "Use get_scalar_window(name).current() for typed, compile-time safe access"
    )]
    pub fn get_latest(&self, name: &str) -> FloatValue {
        let item = self
            .iter()
            .find(|item| item.name == name)
            .expect("No item found");

        match &item.data {
            TimeseriesData::Scalar(ts) => match item.variable_type {
                VariableType::Exogenous => {
                    ts.at_time(self.current_time, ScalarRegion::Global).unwrap()
                }
                VariableType::Endogenous => ts.latest_value().unwrap(),
            },
            _ => panic!("Variable {} is not scalar", name),
        }
    }

    /// Get the latest value as a StateValue (scalar or grid)
    ///
    /// For grid timeseries, returns all regional values.
    /// For scalar timeseries, returns a single value wrapped in StateValue::Scalar.
    pub fn get_latest_value(&self, name: &str) -> Option<StateValue> {
        let item = self.iter().find(|item| item.name == name)?;

        match &item.data {
            TimeseriesData::Scalar(ts) => {
                let value = match item.variable_type {
                    VariableType::Exogenous => {
                        ts.at_time(self.current_time, ScalarRegion::Global).ok()?
                    }
                    VariableType::Endogenous => ts.latest_value()?,
                };
                Some(StateValue::Scalar(value))
            }
            TimeseriesData::FourBox(ts) => {
                let values = match item.variable_type {
                    VariableType::Exogenous => ts.at_time_all(self.current_time).ok()?,
                    VariableType::Endogenous => ts.latest_values(),
                };
                Some(StateValue::Grid(values))
            }
            TimeseriesData::Hemispheric(ts) => {
                let values = match item.variable_type {
                    VariableType::Exogenous => ts.at_time_all(self.current_time).ok()?,
                    VariableType::Endogenous => ts.latest_values(),
                };
                Some(StateValue::Grid(values))
            }
        }
    }

    /// Get the global aggregated value for a variable
    ///
    /// For scalar variables, returns the scalar value.
    /// For grid variables, aggregates all regions to a single global value using the grid's weights.
    pub fn get_global(&self, name: &str) -> Option<FloatValue> {
        let item = self.iter().find(|item| item.name == name)?;

        match &item.data {
            TimeseriesData::Scalar(ts) => match item.variable_type {
                VariableType::Exogenous => ts.at_time(self.current_time, ScalarRegion::Global).ok(),
                VariableType::Endogenous => ts.latest_value(),
            },
            TimeseriesData::FourBox(ts) => {
                let values = match item.variable_type {
                    VariableType::Exogenous => ts.at_time_all(self.current_time).ok()?,
                    VariableType::Endogenous => ts.latest_values(),
                };
                Some(ts.grid().aggregate_global(&values))
            }
            TimeseriesData::Hemispheric(ts) => {
                let values = match item.variable_type {
                    VariableType::Exogenous => ts.at_time_all(self.current_time).ok()?,
                    VariableType::Endogenous => ts.latest_values(),
                };
                Some(ts.grid().aggregate_global(&values))
            }
        }
    }

    /// Get a specific region's value for a grid variable
    ///
    /// For scalar variables, always returns the scalar value regardless of region_index.
    /// For grid variables, returns the value for the specified region.
    ///
    /// # Arguments
    /// * `name` - Variable name
    /// * `region_index` - Index of the region (0 for scalar variables)
    pub fn get_region(&self, name: &str, region_index: usize) -> Option<FloatValue> {
        self.get_latest_value(name).and_then(|sv| match sv {
            StateValue::Scalar(v) => {
                if region_index == 0 {
                    Some(v)
                } else {
                    None
                }
            }
            StateValue::Grid(values) => values.get(region_index).copied(),
        })
    }

    /// Get the preindustrial value for a variable.
    ///
    /// Returns the preindustrial reference value if one is set for this variable.
    /// Preindustrial values are scenario-dependent and stored with the timeseries data.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(pi) = input_state.get_preindustrial("Atmospheric Concentration|CO2") {
    ///     let delta = current_conc - pi.to_scalar();
    /// }
    /// ```
    pub fn get_preindustrial(&self, name: &str) -> Option<&PreindustrialValue> {
        self.iter()
            .find(|item| item.name == name)
            .and_then(|item| item.preindustrial.as_ref())
    }

    /// Get the preindustrial value as a scalar.
    ///
    /// This is a convenience method for the common case of needing a scalar preindustrial value.
    /// For grid preindustrial values, uses area-weighted averaging to compute the global value.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pi_co2 = input_state.get_preindustrial_scalar("Atmospheric Concentration|CO2")
    ///     .unwrap_or(278.0);  // Default preindustrial CO2 in ppm
    /// ```
    pub fn get_preindustrial_scalar(&self, name: &str) -> Option<FloatValue> {
        self.get_preindustrial(name).map(|pi| pi.to_scalar())
    }

    /// Test if the state contains a value with the given name
    pub fn has(&self, name: &str) -> bool {
        self.state.iter().any(|x| x.name == name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &&TimeseriesItem> {
        self.state.iter()
    }

    /// Get the current time
    pub fn current_time(&self) -> Time {
        self.current_time
    }

    /// Get a scalar TimeseriesWindow for the named variable
    ///
    /// This provides zero-cost access to current, previous, and historical values.
    ///
    /// # Panics
    ///
    /// Panics if the variable is not found or is not a scalar timeseries.
    pub fn get_scalar_window(&self, name: &str) -> TimeseriesWindow<'_> {
        let item = self
            .iter()
            .find(|item| item.name == name)
            .unwrap_or_else(|| panic!("Variable '{}' not found in input state", name));

        let ts = item
            .data
            .as_scalar()
            .unwrap_or_else(|| panic!("Variable '{}' is not a scalar timeseries", name));

        // Use latest() for all variable types - the model interpolates exogenous
        // data to the common time axis before solving, so latest() is correct.
        let current_index = ts.latest();

        TimeseriesWindow::new(ts, current_index, self.current_time)
    }

    /// Get a FourBox GridTimeseriesWindow for the named variable
    ///
    /// # Panics
    ///
    /// Panics if the variable is not found or is not a FourBox timeseries.
    pub fn get_four_box_window(&self, name: &str) -> GridTimeseriesWindow<'_, FourBoxGrid> {
        let item = self
            .iter()
            .find(|item| item.name == name)
            .unwrap_or_else(|| panic!("Variable '{}' not found in input state", name));

        let ts = item
            .data
            .as_four_box()
            .unwrap_or_else(|| panic!("Variable '{}' is not a FourBox timeseries", name));

        let current_index = ts.latest();

        GridTimeseriesWindow::new(ts, current_index, self.current_time)
    }

    /// Get a Hemispheric GridTimeseriesWindow for the named variable
    ///
    /// # Panics
    ///
    /// Panics if the variable is not found or is not a Hemispheric timeseries.
    pub fn get_hemispheric_window(&self, name: &str) -> GridTimeseriesWindow<'_, HemisphericGrid> {
        let item = self
            .iter()
            .find(|item| item.name == name)
            .unwrap_or_else(|| panic!("Variable '{}' not found in input state", name));

        let ts = item
            .data
            .as_hemispheric()
            .unwrap_or_else(|| panic!("Variable '{}' is not a Hemispheric timeseries", name));

        let current_index = ts.latest();

        GridTimeseriesWindow::new(ts, current_index, self.current_time)
    }

    /// Converts the state into an equivalent hashmap
    ///
    /// For grid variables, aggregates to global values using grid weights.
    pub fn to_hashmap(self) -> HashMap<String, FloatValue> {
        HashMap::from_iter(self.state.into_iter().map(|item| {
            let value = match &item.data {
                TimeseriesData::Scalar(ts) => ts.latest_value().unwrap(),
                TimeseriesData::FourBox(ts) => ts.grid().aggregate_global(&ts.latest_values()),
                TimeseriesData::Hemispheric(ts) => ts.grid().aggregate_global(&ts.latest_values()),
            };
            (item.name.clone(), value)
        }))
    }
}

impl<'a> IntoIterator for InputState<'a> {
    type Item = &'a TimeseriesItem;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.state.into_iter()
    }
}

/// Output state from a component
///
/// Currently stores only scalar values for backwards compatibility.
/// In the future, this may be updated to support `StateValue` to handle grid values.
///
/// Components that produce grid outputs can aggregate to global values before returning,
/// or we can introduce a new `GridOutputState` type in the future.
pub type OutputState = HashMap<String, FloatValue>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_value_scalar() {
        let sv = StateValue::Scalar(42.0);
        assert!(sv.is_scalar());
        assert!(!sv.is_grid());
        assert_eq!(sv.as_scalar(), Some(42.0));
        assert_eq!(sv.as_grid(), None);
        assert_eq!(sv.to_scalar(), 42.0);
    }

    #[test]
    fn test_state_value_grid() {
        let sv = StateValue::Grid(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(!sv.is_scalar());
        assert!(sv.is_grid());
        assert_eq!(sv.as_scalar(), None);
        assert_eq!(sv.as_grid(), Some(&[1.0, 2.0, 3.0, 4.0][..]));
        assert_eq!(sv.to_scalar(), 2.5); // Mean of [1, 2, 3, 4]
    }

    #[test]
    fn test_state_value_grid_aggregation() {
        let sv = StateValue::Grid(vec![10.0, 20.0]);
        assert_eq!(sv.to_scalar(), 15.0);
    }

    #[test]
    fn test_input_state_get_global() {
        use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
        use crate::timeseries::{TimeAxis, Timeseries};
        use numpy::array;
        use numpy::ndarray::Axis;
        use std::sync::Arc;

        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
        let values = array![280.0, 285.0].insert_axis(Axis(1));
        let ts = Timeseries::new(
            values,
            time_axis,
            crate::spatial::ScalarGrid,
            "ppm".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );

        let item = TimeseriesItem {
            data: TimeseriesData::Scalar(ts),
            name: "CO2".to_string(),
            variable_type: VariableType::Endogenous,
            preindustrial: None,
        };

        let state = InputState::build(vec![&item], 2000.5);

        // For Endogenous variables, get_latest returns the latest_value (index 1)
        assert_eq!(state.get_global("CO2"), Some(285.0));
        assert_eq!(state.get_latest("CO2"), 285.0);
    }

    #[test]
    fn test_input_state_get_region_scalar() {
        use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
        use crate::timeseries::{TimeAxis, Timeseries};
        use numpy::array;
        use numpy::ndarray::Axis;
        use std::sync::Arc;

        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
        let values = array![15.0, 16.0].insert_axis(Axis(1));
        let ts = Timeseries::new(
            values,
            time_axis,
            crate::spatial::ScalarGrid,
            "°C".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );

        let item = TimeseriesItem {
            data: TimeseriesData::Scalar(ts),
            name: "Temperature".to_string(),
            variable_type: VariableType::Endogenous,
            preindustrial: None,
        };

        let state = InputState::build(vec![&item], 2000.5);

        // For Endogenous variables, returns latest_value (index 1)
        // Scalar values accessible as region 0
        assert_eq!(state.get_region("Temperature", 0), Some(16.0));
        // Other regions return None for scalar
        assert_eq!(state.get_region("Temperature", 1), None);
    }

    #[test]
    fn test_input_state_grid_values() {
        use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
        use crate::spatial::FourBoxGrid;
        use crate::timeseries::{GridTimeseries, TimeAxis};
        use numpy::array;
        use numpy::ndarray::Array2;
        use std::sync::Arc;

        let grid = FourBoxGrid::magicc_standard();
        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
        let values =
            Array2::from_shape_vec((2, 4), vec![15.0, 14.0, 10.0, 9.0, 16.0, 15.0, 11.0, 10.0])
                .unwrap();

        let ts = GridTimeseries::new(
            values,
            time_axis,
            grid,
            "°C".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );

        let item = TimeseriesItem {
            data: TimeseriesData::FourBox(ts),
            name: "Temperature".to_string(),
            variable_type: VariableType::Endogenous,
            preindustrial: None,
        };

        let state = InputState::build(vec![&item], 2000.5);

        // Test get_latest_value returns Grid variant
        let value = state.get_latest_value("Temperature").unwrap();
        assert!(value.is_grid());
        assert_eq!(value.as_grid(), Some(&[16.0, 15.0, 11.0, 10.0][..]));

        // Test get_global aggregates using weights (equal weights = mean)
        let global = state.get_global("Temperature").unwrap();
        assert_eq!(global, 13.0); // (16 + 15 + 11 + 10) / 4

        // Test get_region for individual regions
        assert_eq!(state.get_region("Temperature", 0), Some(16.0));
        assert_eq!(state.get_region("Temperature", 1), Some(15.0));
        assert_eq!(state.get_region("Temperature", 2), Some(11.0));
        assert_eq!(state.get_region("Temperature", 3), Some(10.0));
        assert_eq!(state.get_region("Temperature", 4), None); // Out of bounds
    }

    #[test]
    fn test_input_state_to_hashmap_with_grid() {
        use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
        use crate::spatial::FourBoxGrid;
        use crate::timeseries::{GridTimeseries, TimeAxis};
        use numpy::array;
        use numpy::ndarray::Array2;
        use std::sync::Arc;

        let grid = FourBoxGrid::magicc_standard();
        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
        let values =
            Array2::from_shape_vec((2, 4), vec![15.0, 14.0, 10.0, 9.0, 16.0, 15.0, 11.0, 10.0])
                .unwrap();

        let ts = GridTimeseries::new(
            values,
            time_axis,
            grid,
            "°C".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );

        let item = TimeseriesItem {
            data: TimeseriesData::FourBox(ts),
            name: "Temperature".to_string(),
            variable_type: VariableType::Endogenous,
            preindustrial: None,
        };

        let state = InputState::build(vec![&item], 2000.5);
        let hashmap = state.to_hashmap();

        // Should contain aggregated global value
        assert_eq!(hashmap.get("Temperature"), Some(&13.0));
    }
}

impl StateValue {
    /// Convert to a scalar value, aggregating if necessary
    ///
    /// For Scalar variants, returns the value directly.
    /// For Grid variants, computes the mean of all regional values.
    ///
    /// Note: This simple averaging may not be physically appropriate for all variables.
    /// Use grid-aware aggregation methods when the grid weights are known.
    pub fn to_scalar(&self) -> FloatValue {
        match self {
            StateValue::Scalar(v) => *v,
            StateValue::Grid(values) => {
                let sum: FloatValue = values.iter().sum();
                sum / (values.len() as FloatValue)
            }
        }
    }

    /// Check if this is a scalar value
    pub fn is_scalar(&self) -> bool {
        matches!(self, StateValue::Scalar(_))
    }

    /// Check if this is a grid value
    pub fn is_grid(&self) -> bool {
        matches!(self, StateValue::Grid(_))
    }

    /// Get the scalar value if this is a Scalar variant
    pub fn as_scalar(&self) -> Option<FloatValue> {
        match self {
            StateValue::Scalar(v) => Some(*v),
            StateValue::Grid(_) => None,
        }
    }

    /// Get the grid values if this is a Grid variant
    pub fn as_grid(&self) -> Option<&[FloatValue]> {
        match self {
            StateValue::Scalar(_) => None,
            StateValue::Grid(values) => Some(values),
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
    fn test_timeseries_window_current() {
        let ts = create_scalar_timeseries();
        let window = TimeseriesWindow::new(&ts, 2, 2002.0);

        assert_eq!(window.current(), 3.0);
        assert_eq!(window.time(), 2002.0);
        assert_eq!(window.index(), 2);
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
    fn test_grid_window_current() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::new(&ts, 1, 2001.0);

        assert_eq!(window.current(FourBoxRegion::NorthernOcean), 16.0);
        assert_eq!(window.current(FourBoxRegion::NorthernLand), 15.0);
        assert_eq!(window.current(FourBoxRegion::SouthernOcean), 11.0);
        assert_eq!(window.current(FourBoxRegion::SouthernLand), 10.0);
    }

    #[test]
    fn test_grid_window_current_all() {
        let ts = create_four_box_timeseries();
        let window = GridTimeseriesWindow::new(&ts, 1, 2001.0);

        let all = window.current_all();
        assert_eq!(all, vec![16.0, 15.0, 11.0, 10.0]);
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

#[cfg(test)]
mod input_state_window_tests {
    use super::*;
    use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
    use crate::spatial::{FourBoxGrid, HemisphericGrid};
    use crate::timeseries::{GridTimeseries, TimeAxis, Timeseries};
    use numpy::array;
    use numpy::ndarray::{Array2, Axis};
    use std::sync::Arc;

    fn create_scalar_item(name: &str, values: Vec<FloatValue>) -> TimeseriesItem {
        // Create time axis that matches values length
        let n = values.len();
        let time_vals: Vec<f64> = (0..n).map(|i| 2000.0 + i as f64).collect();
        let time_axis = Arc::new(TimeAxis::from_values(ndarray::Array1::from_vec(time_vals)));
        let values_arr = ndarray::Array1::from_vec(values).insert_axis(Axis(1));
        let ts = Timeseries::new(
            values_arr,
            time_axis,
            ScalarGrid,
            "unit".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );
        TimeseriesItem {
            data: TimeseriesData::Scalar(ts),
            name: name.to_string(),
            variable_type: VariableType::Endogenous,
            preindustrial: None,
        }
    }

    fn create_four_box_item(name: &str) -> TimeseriesItem {
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
        let ts = GridTimeseries::new(
            values,
            time_axis,
            grid,
            "C".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );
        TimeseriesItem {
            data: TimeseriesData::FourBox(ts),
            name: name.to_string(),
            variable_type: VariableType::Endogenous,
            preindustrial: None,
        }
    }

    fn create_hemispheric_item(name: &str) -> TimeseriesItem {
        let grid = HemisphericGrid::equal_weights();
        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
        let values = Array2::from_shape_vec(
            (2, 2),
            vec![
                1000.0, 500.0, // 2000
                1100.0, 550.0, // 2001
            ],
        )
        .unwrap();
        let ts = GridTimeseries::new(
            values,
            time_axis,
            grid,
            "mm/yr".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );
        TimeseriesItem {
            data: TimeseriesData::Hemispheric(ts),
            name: name.to_string(),
            variable_type: VariableType::Endogenous,
            preindustrial: None,
        }
    }

    #[test]
    fn test_get_scalar_window() {
        let item = create_scalar_item("CO2", vec![280.0, 285.0, 290.0, 295.0, 300.0]);
        let state = InputState::build(vec![&item], 2002.0);

        let window = state.get_scalar_window("CO2");

        // latest() returns the highest index with valid data = 4 (300.0)
        assert_eq!(window.current(), 300.0);
        assert_eq!(window.previous(), Some(295.0));
        assert_eq!(window.len(), 5);
    }

    #[test]
    fn test_get_four_box_window() {
        let item = create_four_box_item("Temperature");
        let state = InputState::build(vec![&item], 2001.0);

        let window = state.get_four_box_window("Temperature");

        // latest() = 2 (2002 values: [17.0, 16.0, 12.0, 11.0])
        assert_eq!(window.current(FourBoxRegion::NorthernOcean), 17.0);
        assert_eq!(window.current(FourBoxRegion::SouthernLand), 11.0);
        assert_eq!(window.current_all(), vec![17.0, 16.0, 12.0, 11.0]);
        assert_eq!(window.previous_all(), Some(vec![16.0, 15.0, 11.0, 10.0]));
    }

    #[test]
    fn test_get_hemispheric_window() {
        let item = create_hemispheric_item("Precipitation");
        let state = InputState::build(vec![&item], 2000.0);

        let window = state.get_hemispheric_window("Precipitation");

        // latest() = 1 (2001 values: [1100.0, 550.0])
        assert_eq!(window.current(HemisphericRegion::Northern), 1100.0);
        assert_eq!(window.current(HemisphericRegion::Southern), 550.0);
        assert_eq!(window.current_global(), 825.0); // Equal weights mean
    }

    #[test]
    #[should_panic(expected = "Variable 'NonExistent' not found")]
    fn test_get_scalar_window_missing_variable() {
        let item = create_scalar_item("CO2", vec![280.0, 285.0]);
        let state = InputState::build(vec![&item], 2000.0);
        let _ = state.get_scalar_window("NonExistent");
    }

    #[test]
    #[should_panic(expected = "not a scalar timeseries")]
    fn test_get_scalar_window_wrong_type() {
        let item = create_four_box_item("Temperature");
        let state = InputState::build(vec![&item], 2000.0);
        // Attempting to get scalar window for a FourBox variable should panic
        let _ = state.get_scalar_window("Temperature");
    }

    #[test]
    #[should_panic(expected = "not a FourBox timeseries")]
    fn test_get_four_box_window_wrong_type() {
        let item = create_scalar_item("CO2", vec![280.0, 285.0]);
        let state = InputState::build(vec![&item], 2000.0);
        let _ = state.get_four_box_window("CO2");
    }

    #[test]
    fn test_current_time_accessor() {
        let item = create_scalar_item("CO2", vec![280.0, 285.0]);
        let state = InputState::build(vec![&item], 2023.5);
        assert_eq!(state.current_time(), 2023.5);
    }

    #[test]
    fn test_get_preindustrial_scalar() {
        use crate::variable::PreindustrialValue;

        // Create time axis that matches values length
        let time_axis = Arc::new(TimeAxis::from_values(ndarray::Array1::from_vec(vec![
            2000.0, 2001.0,
        ])));
        let values_arr = ndarray::Array1::from_vec(vec![280.0, 285.0]).insert_axis(Axis(1));
        let ts = Timeseries::new(
            values_arr,
            time_axis,
            ScalarGrid,
            "ppm".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );
        let item = TimeseriesItem {
            data: TimeseriesData::Scalar(ts),
            name: "CO2".to_string(),
            variable_type: VariableType::Endogenous,
            preindustrial: Some(PreindustrialValue::Scalar(278.0)),
        };

        let state = InputState::build(vec![&item], 2000.0);

        // Test get_preindustrial returns the full PreindustrialValue
        let pi = state.get_preindustrial("CO2").unwrap();
        assert_eq!(pi.to_scalar(), 278.0);

        // Test get_preindustrial_scalar returns the scalar directly
        assert_eq!(state.get_preindustrial_scalar("CO2"), Some(278.0));
    }

    #[test]
    fn test_get_preindustrial_four_box() {
        use crate::variable::PreindustrialValue;

        let grid = FourBoxGrid::magicc_standard();
        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
        let values =
            Array2::from_shape_vec((2, 4), vec![15.0, 14.0, 10.0, 9.0, 16.0, 15.0, 11.0, 10.0])
                .unwrap();
        let ts = GridTimeseries::new(
            values,
            time_axis,
            grid,
            "K".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );

        let pi_values = [14.0, 13.0, 9.0, 8.0];
        let item = TimeseriesItem {
            data: TimeseriesData::FourBox(ts),
            name: "Temperature".to_string(),
            variable_type: VariableType::Endogenous,
            preindustrial: Some(PreindustrialValue::FourBox(pi_values)),
        };

        let state = InputState::build(vec![&item], 2000.0);

        // Test get_preindustrial returns the full PreindustrialValue
        let pi = state.get_preindustrial("Temperature").unwrap();
        assert_eq!(pi.as_four_box(), Some(pi_values));

        // Test get_preindustrial_scalar returns the weighted average
        let scalar = state.get_preindustrial_scalar("Temperature").unwrap();
        // MAGICC standard weights are equal, so average is (14 + 13 + 9 + 8) / 4 = 11.0
        assert_eq!(scalar, 11.0);
    }

    #[test]
    fn test_get_preindustrial_none() {
        let item = create_scalar_item("CO2", vec![280.0, 285.0]);
        let state = InputState::build(vec![&item], 2000.0);

        // Item without preindustrial should return None
        assert!(state.get_preindustrial("CO2").is_none());
        assert!(state.get_preindustrial_scalar("CO2").is_none());
    }

    #[test]
    fn test_get_preindustrial_missing_variable() {
        let item = create_scalar_item("CO2", vec![280.0, 285.0]);
        let state = InputState::build(vec![&item], 2000.0);

        // Non-existent variable should return None
        assert!(state.get_preindustrial("NonExistent").is_none());
        assert!(state.get_preindustrial_scalar("NonExistent").is_none());
    }
}
