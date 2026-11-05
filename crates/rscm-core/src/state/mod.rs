//! State management for model components.
//!
//! This module provides types for managing input and output state in model components,
//! including:
//!
//! - [`StateValue`]: Enum for scalar, four-box, or hemispheric values
//! - [`InputState`]: Input state passed to components during solve
//! - [`OutputState`]: Output state returned from components
//! - Window types for efficient timeseries access
//! - Slice types for typed regional values

mod aggregating;
mod slices;
mod windows;

// Re-export slice types
pub use slices::{FourBoxSlice, HemisphericSlice};

// Re-export window types
pub(crate) use windows::ReadIndex;
pub use windows::{GridTimeseriesWindow, TimeseriesWindow};

// Re-export aggregating window types
pub use aggregating::{
    AggregatingFourBoxToHemisphericWindow, AggregatingFourBoxWindow, AggregatingHemisphericWindow,
    HemisphericWindow, ReadTransformInfo, ScalarWindow,
};

use crate::component::{GridType, TimeseriesType};
use crate::spatial::{FourBoxGrid, ScalarRegion, SpatialGrid};
use crate::timeseries::{FloatValue, Time, TimeAxis};
use crate::timeseries_collection::{TimeseriesData, TimeseriesItem, VariableType};
use pyo3::pyclass;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;

// =============================================================================
// State Value Types
// =============================================================================

/// Represents a value that can be either scalar or spatially-resolved
///
/// `StateValue` is the enum used for both input state retrieval and output state
/// in components. It provides type-safe handling of scalar and grid-based values.
///
/// # Examples
///
/// ```rust
/// use rscm_core::state::{StateValue, FourBoxSlice, HemisphericSlice};
///
/// // Scalar value
/// let scalar = StateValue::Scalar(288.0);
/// assert_eq!(scalar.to_scalar(), 288.0);
///
/// // FourBox value
/// let four_box = StateValue::FourBox(FourBoxSlice::from_array([15.0, 14.0, 10.0, 9.0]));
/// assert_eq!(four_box.to_scalar(), 12.0); // Mean of all regions
///
/// // Hemispheric value
/// let hemispheric = StateValue::Hemispheric(HemisphericSlice::from_array([15.0, 10.0]));
/// assert_eq!(hemispheric.to_scalar(), 12.5); // Mean of both hemispheres
/// ```
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum StateValue {
    /// A single scalar value (global average or non-spatial variable)
    Scalar(FloatValue),
    /// Four-box regional values (Northern Ocean, Northern Land, Southern Ocean, Southern Land)
    FourBox(FourBoxSlice),
    /// Hemispheric values (Northern, Southern)
    Hemispheric(HemisphericSlice),
}

impl StateValue {
    /// Convert to a scalar value, aggregating if necessary
    ///
    /// For Scalar variants, returns the value directly.
    /// For FourBox variants, computes the mean of all 4 regional values.
    /// For Hemispheric variants, computes the mean of both hemispheres.
    ///
    /// Note: This simple averaging may not be physically appropriate for all variables.
    /// Use grid-aware aggregation methods when the grid weights are known.
    pub fn to_scalar(&self) -> FloatValue {
        match self {
            StateValue::Scalar(v) => *v,
            StateValue::FourBox(slice) => {
                let values = slice.as_array();
                values.iter().sum::<FloatValue>() / 4.0
            }
            StateValue::Hemispheric(slice) => {
                let values = slice.as_array();
                values.iter().sum::<FloatValue>() / 2.0
            }
        }
    }

    /// Check if this is a scalar value
    pub fn is_scalar(&self) -> bool {
        matches!(self, StateValue::Scalar(_))
    }

    /// Check if this is a FourBox grid value
    pub fn is_four_box(&self) -> bool {
        matches!(self, StateValue::FourBox(_))
    }

    /// Check if this is a Hemispheric grid value
    pub fn is_hemispheric(&self) -> bool {
        matches!(self, StateValue::Hemispheric(_))
    }

    /// Get the scalar value if this is a Scalar variant
    pub fn as_scalar(&self) -> Option<FloatValue> {
        match self {
            StateValue::Scalar(v) => Some(*v),
            _ => None,
        }
    }

    /// Get the FourBoxSlice if this is a FourBox variant
    pub fn as_four_box(&self) -> Option<&FourBoxSlice> {
        match self {
            StateValue::FourBox(slice) => Some(slice),
            _ => None,
        }
    }

    /// Get the HemisphericSlice if this is a Hemispheric variant
    pub fn as_hemispheric(&self) -> Option<&HemisphericSlice> {
        match self {
            StateValue::Hemispheric(slice) => Some(slice),
            _ => None,
        }
    }
}

impl From<FloatValue> for StateValue {
    fn from(value: FloatValue) -> Self {
        StateValue::Scalar(value)
    }
}

impl From<FourBoxSlice> for StateValue {
    fn from(slice: FourBoxSlice) -> Self {
        StateValue::FourBox(slice)
    }
}

impl From<HemisphericSlice> for StateValue {
    fn from(slice: HemisphericSlice) -> Self {
        StateValue::Hemispheric(slice)
    }
}

/// Describes the source/origin of a variable in the model.
///
/// This is used by `get()` to determine the correct timestep index to read from.
/// The source is computed at model build time based on the dependency graph.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum VariableSource {
    /// Variable is provided externally (exogenous input).
    /// `get()` reads index N.
    #[default]
    Exogenous,

    /// Variable is written by an upstream component in the dependency graph.
    /// `get()` reads index N+1, which the upstream component wrote earlier in this step.
    UpstreamOutput,

    /// Variable is this component's own state from the previous timestep.
    /// `get()` reads index N.
    OwnState,

    /// Variable is read from the previous step because the model wiring lags this edge.
    /// `get()` reads index N, and `interval()` gives the previous step's (N-1, N).
    Lagged,
}

/// Transform context for automatic grid aggregation.
///
/// This holds the information needed to transform grid data during read operations.
/// It is passed to InputState when the model has configured grid transformations.
#[derive(Debug, Clone, Default)]
pub struct TransformContext {
    /// Map from variable name to the read transformation info
    pub read_transforms: HashMap<String, ReadTransformInfo>,
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
    /// Index of `current_time` in the time axis that most items share.
    time_index: Option<usize>,
    state: Vec<&'a TimeseriesItem>,
    /// Optional transform context for grid aggregation
    transform_context: Option<Cow<'a, TransformContext>>,
}

/// Order names by length and then by name, which settles most comparisons by length alone.
pub(crate) fn lookup_order(a: &str, b: &str) -> std::cmp::Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

impl<'a> InputState<'a> {
    pub fn build(values: Vec<&'a TimeseriesItem>, current_time: Time) -> Self {
        Self {
            current_time,
            time_index: None,
            state: values,
            transform_context: None,
        }
    }

    /// Build an InputState with transform context for grid aggregation.
    pub fn build_with_transforms(
        values: Vec<&'a TimeseriesItem>,
        current_time: Time,
        transform_context: TransformContext,
    ) -> Self {
        Self {
            current_time,
            time_index: None,
            state: values,
            transform_context: Some(Cow::Owned(transform_context)),
        }
    }

    /// Build an InputState that borrows an optional transform context.
    pub fn build_with_transforms_ref(
        values: Vec<&'a TimeseriesItem>,
        current_time: Time,
        transform_context: Option<&'a TransformContext>,
    ) -> Self {
        Self {
            current_time,
            time_index: None,
            state: values,
            transform_context: transform_context.map(Cow::Borrowed),
        }
    }

    pub fn empty() -> Self {
        Self {
            current_time: f64::NAN,
            time_index: None,
            state: vec![],
            transform_context: None,
        }
    }

    /// Get the global aggregated value for a variable
    ///
    /// For scalar variables, returns the scalar value.
    /// For grid variables, aggregates all regions to a single global value using the grid's weights.
    pub fn get_global(&self, name: &str) -> Option<FloatValue> {
        let item = self.find(name)?;

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

    /// Test if the state contains a value with the given name
    pub fn has(&self, name: &str) -> bool {
        self.find(name).is_some()
    }

    /// Set the index of the current time in the time axis that most items share.
    pub fn with_time_index(mut self, time_index: usize) -> Self {
        self.time_index = Some(time_index);
        self
    }

    /// Index of the current time in `axis`.
    pub(crate) fn time_index(&self, axis: &TimeAxis, name: &str) -> usize {
        self.time_index
            .filter(|&index| axis.at(index) == Some(self.current_time))
            .or_else(|| axis.index_of(self.current_time))
            .unwrap_or_else(|| {
                panic!(
                    "Time {} not found in timeseries '{}' time axis",
                    self.current_time, name
                )
            })
    }

    /// Item named `name`, by binary search when the items are in [`lookup_order`].
    fn find(&self, name: &str) -> Option<&'a TimeseriesItem> {
        let index = self
            .state
            .binary_search_by(|item| lookup_order(&item.name, name))
            .ok()
            .or_else(|| self.state.iter().position(|item| item.name == name))?;
        Some(self.state[index])
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
    /// If the underlying data is stored at a finer grid resolution (FourBox or Hemispheric)
    /// and a read transform is configured, the data will be automatically aggregated
    /// to a scalar value on each access.
    ///
    /// # Panics
    ///
    /// Panics if the variable is not found or cannot be accessed as scalar.
    pub fn get_scalar_window(&self, name: &str) -> ScalarWindow<'_> {
        self.scalar_window(name)
            .with_timeseries_type(self.timeseries_type(name))
    }

    /// Point or average type of `name`, from its read transform.
    fn timeseries_type(&self, name: &str) -> TimeseriesType {
        self.transform_context
            .as_ref()
            .and_then(|ctx| ctx.read_transforms.get(name))
            .map_or(TimeseriesType::Point, |t| t.timeseries_type)
    }

    fn scalar_window(&self, name: &str) -> ScalarWindow<'_> {
        let item = self
            .find(name)
            .unwrap_or_else(|| panic!("Variable '{}' not found in input state", name));

        // Check if there's a read transform for this variable
        let transform = self
            .transform_context
            .as_ref()
            .and_then(|ctx| ctx.read_transforms.get(name));

        // If there's a transform, use the source grid type and unit conversion factor
        if let Some(transform) = transform {
            let unit_factor = transform.unit_conversion_factor;

            match transform.source_grid {
                GridType::FourBox => {
                    let ts = item.data.as_four_box().unwrap_or_else(|| {
                        panic!(
                            "Variable '{}' requires FourBox->Scalar transform but is not FourBox",
                            name
                        )
                    });

                    let current_index = self.time_index(ts.time_axis_ref(), name);

                    return ScalarWindow::FromFourBox(AggregatingFourBoxWindow::with_source(
                        ts,
                        current_index,
                        &item.name,
                        transform.weights.clone(),
                        unit_factor,
                        transform.variable_source,
                    ));
                }
                GridType::Hemispheric => {
                    let ts = item
                        .data
                        .as_hemispheric()
                        .unwrap_or_else(|| {
                            panic!(
                                "Variable '{}' requires Hemispheric->Scalar transform but is not Hemispheric",
                                name
                            )
                        });

                    let current_index = self.time_index(ts.time_axis_ref(), name);

                    return ScalarWindow::FromHemispheric(
                        AggregatingHemisphericWindow::with_source(
                            ts,
                            current_index,
                            &item.name,
                            transform.weights.clone(),
                            unit_factor,
                            transform.variable_source,
                        ),
                    );
                }
                GridType::Scalar => {
                    // Scalar->Scalar with possible unit conversion
                    let ts = item.data.as_scalar().unwrap_or_else(|| {
                        panic!("Variable '{}' is not a scalar timeseries", name)
                    });

                    let current_index = self.time_index(ts.time_axis_ref(), name);

                    return ScalarWindow::Direct(TimeseriesWindow::with_source(
                        ts,
                        current_index,
                        &item.name,
                        unit_factor,
                        transform.variable_source,
                    ));
                }
            }
        }

        // Direct scalar access (no transform needed)
        let ts = item
            .data
            .as_scalar()
            .unwrap_or_else(|| panic!("Variable '{}' is not a scalar timeseries", name));

        let current_index = self.time_index(ts.time_axis_ref(), name);

        // Get variable source from transform context if available
        let source = self
            .transform_context
            .as_ref()
            .and_then(|ctx| ctx.read_transforms.get(name))
            .map(|t| t.variable_source)
            .unwrap_or(VariableSource::Exogenous);

        ScalarWindow::Direct(TimeseriesWindow::with_source(
            ts,
            current_index,
            &item.name,
            1.0, // no unit conversion for direct access
            source,
        ))
    }

    /// Get a FourBox GridTimeseriesWindow for the named variable
    ///
    /// If a read transform with unit conversion is configured for this variable,
    /// the conversion factor will be applied to all returned values.
    ///
    /// # Panics
    ///
    /// Panics if the variable is not found or is not a FourBox timeseries.
    pub fn get_four_box_window(&self, name: &str) -> GridTimeseriesWindow<'_, FourBoxGrid> {
        let item = self
            .find(name)
            .unwrap_or_else(|| panic!("Variable '{}' not found in input state", name));

        let ts = item
            .data
            .as_four_box()
            .unwrap_or_else(|| panic!("Variable '{}' is not a FourBox timeseries", name));

        // Find the index corresponding to current_time in the timeseries
        let current_index = self.time_index(ts.time_axis_ref(), name);

        // Check if there's a unit conversion factor and variable source
        let (unit_factor, source) = self
            .transform_context
            .as_ref()
            .and_then(|ctx| ctx.read_transforms.get(name))
            .map(|t| (t.unit_conversion_factor, t.variable_source))
            .unwrap_or((1.0, VariableSource::Exogenous));

        GridTimeseriesWindow::with_source(ts, current_index, &item.name, unit_factor, source)
            .with_timeseries_type(self.timeseries_type(name))
    }

    /// Get a Hemispheric GridTimeseriesWindow for the named variable
    ///
    /// If the underlying data is stored at FourBox resolution and a read transform
    /// is configured, the data will be automatically aggregated to Hemispheric
    /// on each access.
    ///
    /// If a read transform with unit conversion is configured for this variable,
    /// the conversion factor will be applied to all returned values.
    ///
    /// # Panics
    ///
    /// Panics if the variable is not found or cannot be accessed as Hemispheric.
    pub fn get_hemispheric_window(&self, name: &str) -> HemisphericWindow<'_> {
        self.hemispheric_window(name)
            .with_timeseries_type(self.timeseries_type(name))
    }

    fn hemispheric_window(&self, name: &str) -> HemisphericWindow<'_> {
        let item = self
            .find(name)
            .unwrap_or_else(|| panic!("Variable '{}' not found in input state", name));

        // Check if there's a read transform for this variable
        let transform = self
            .transform_context
            .as_ref()
            .and_then(|ctx| ctx.read_transforms.get(name));

        // If there's a transform from FourBox, aggregate
        if let Some(transform) = transform {
            let unit_factor = transform.unit_conversion_factor;

            if transform.source_grid == GridType::FourBox {
                let ts = item.data.as_four_box().unwrap_or_else(|| {
                    panic!(
                        "Variable '{}' requires FourBox->Hemispheric transform but is not FourBox",
                        name
                    )
                });

                let current_index = self.time_index(ts.time_axis_ref(), name);

                return HemisphericWindow::FromFourBox(
                    AggregatingFourBoxToHemisphericWindow::with_source(
                        ts,
                        current_index,
                        &item.name,
                        unit_factor,
                        transform.variable_source,
                    ),
                );
            }

            // Hemispheric->Hemispheric with possible unit conversion
            if transform.source_grid == GridType::Hemispheric {
                let ts = item.data.as_hemispheric().unwrap_or_else(|| {
                    panic!("Variable '{}' is not a Hemispheric timeseries", name)
                });

                let current_index = self.time_index(ts.time_axis_ref(), name);

                return HemisphericWindow::Direct(GridTimeseriesWindow::with_source(
                    ts,
                    current_index,
                    &item.name,
                    unit_factor,
                    transform.variable_source,
                ));
            }
        }

        // Direct hemispheric access (no transform)
        let ts = item
            .data
            .as_hemispheric()
            .unwrap_or_else(|| panic!("Variable '{}' is not a Hemispheric timeseries", name));

        let current_index = self.time_index(ts.time_axis_ref(), name);

        // Get variable source from transform context if available
        let source = self
            .transform_context
            .as_ref()
            .and_then(|ctx| ctx.read_transforms.get(name))
            .map(|t| t.variable_source)
            .unwrap_or(VariableSource::Exogenous);

        HemisphericWindow::Direct(GridTimeseriesWindow::with_source(
            ts,
            current_index,
            &item.name,
            1.0,
            source,
        ))
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
/// A collection of named values that a component produces. Each value can be:
/// - `StateValue::Scalar` for global/non-spatial values
/// - `StateValue::FourBox` for four-box regional values
/// - `StateValue::Hemispheric` for hemispheric values
///
/// The model writes these values to the appropriate timeseries based on the
/// variable's grid type in `RequirementDefinition`.
pub type OutputState = HashMap<String, StateValue>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_value_scalar() {
        let sv = StateValue::Scalar(42.0);
        assert!(sv.is_scalar());
        assert!(!sv.is_four_box());
        assert!(!sv.is_hemispheric());
        assert_eq!(sv.as_scalar(), Some(42.0));
        assert_eq!(sv.as_four_box(), None);
        assert_eq!(sv.as_hemispheric(), None);
        assert_eq!(sv.to_scalar(), 42.0);
    }

    #[test]
    fn test_state_value_four_box() {
        let slice = FourBoxSlice::from_array([1.0, 2.0, 3.0, 4.0]);
        let sv = StateValue::FourBox(slice);
        assert!(!sv.is_scalar());
        assert!(sv.is_four_box());
        assert!(!sv.is_hemispheric());
        assert_eq!(sv.as_scalar(), None);
        assert_eq!(sv.as_four_box(), Some(&slice));
        assert_eq!(sv.as_hemispheric(), None);
        assert_eq!(sv.to_scalar(), 2.5); // Mean of [1, 2, 3, 4]
    }

    #[test]
    fn test_state_value_hemispheric() {
        let slice = HemisphericSlice::from_array([10.0, 20.0]);
        let sv = StateValue::Hemispheric(slice);
        assert!(!sv.is_scalar());
        assert!(!sv.is_four_box());
        assert!(sv.is_hemispheric());
        assert_eq!(sv.as_scalar(), None);
        assert_eq!(sv.as_four_box(), None);
        assert_eq!(sv.as_hemispheric(), Some(&slice));
        assert_eq!(sv.to_scalar(), 15.0); // Mean of [10, 20]
    }

    #[test]
    fn test_state_value_from_impls() {
        // Test From<FloatValue> for StateValue
        let sv: StateValue = 42.0.into();
        assert!(sv.is_scalar());
        assert_eq!(sv.as_scalar(), Some(42.0));

        // Test From<FourBoxSlice> for StateValue
        let slice = FourBoxSlice::from_array([1.0, 2.0, 3.0, 4.0]);
        let sv: StateValue = slice.into();
        assert!(sv.is_four_box());

        // Test From<HemisphericSlice> for StateValue
        let slice = HemisphericSlice::from_array([10.0, 20.0]);
        let sv: StateValue = slice.into();
        assert!(sv.is_hemispheric());
    }

    #[test]
    fn test_input_state_get_global() {
        use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
        use crate::spatial::ScalarGrid;
        use crate::timeseries::{TimeAxis, Timeseries};
        use numpy::array;
        use numpy::ndarray::Axis;
        use std::sync::Arc;

        let time_axis = Arc::new(TimeAxis::from_values(array![2000.0, 2001.0]));
        let values = array![280.0, 285.0].insert_axis(Axis(1));
        let ts = Timeseries::new(
            values,
            time_axis,
            ScalarGrid,
            "ppm".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );

        let item = TimeseriesItem {
            data: TimeseriesData::Scalar(ts),
            name: "CO2".to_string(),
            variable_type: VariableType::Endogenous,
        };

        // Use a time that exists in the time axis
        let state = InputState::build(vec![&item], 2001.0);

        // get() returns the value at the index of current_time (index 1)
        assert_eq!(state.get_global("CO2"), Some(285.0));
        assert_eq!(state.get_scalar_window("CO2").get(), 285.0);
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
            "degC".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );

        let item = TimeseriesItem {
            data: TimeseriesData::FourBox(ts),
            name: "Temperature".to_string(),
            variable_type: VariableType::Endogenous,
        };

        // Use a time that exists in the time axis
        let state = InputState::build(vec![&item], 2001.0);

        // Test get_four_box_window returns values at index 1 using get_all()
        let window = state.get_four_box_window("Temperature");
        let values = window.get_all();
        assert_eq!(values, [16.0, 15.0, 11.0, 10.0]);

        // Test get_global aggregates using weights (equal weights = mean)
        let global = state.get_global("Temperature").unwrap();
        assert_eq!(global, 13.0); // (16 + 15 + 11 + 10) / 4
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
            "degC".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );

        let item = TimeseriesItem {
            data: TimeseriesData::FourBox(ts),
            name: "Temperature".to_string(),
            variable_type: VariableType::Endogenous,
        };

        let state = InputState::build(vec![&item], 2000.5);
        let hashmap = state.to_hashmap();

        // Should contain aggregated global value
        assert_eq!(hashmap.get("Temperature"), Some(&13.0));
    }
}

#[cfg(test)]
mod input_state_window_tests {
    use super::*;
    use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
    use crate::spatial::{
        FourBoxGrid, FourBoxRegion, HemisphericGrid, HemisphericRegion, ScalarGrid,
    };
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
        }
    }

    #[test]
    fn test_get_scalar_window() {
        let item = create_scalar_item("CO2", vec![280.0, 285.0, 290.0, 295.0, 300.0]);
        // Time 2002.0 corresponds to index 2 in the timeseries [2000, 2001, 2002, 2003, 2004]
        let state = InputState::build(vec![&item], 2002.0);

        let window = state.get_scalar_window("CO2");

        // get() returns the value at the index of current_time
        assert_eq!(window.get(), 290.0);
        assert_eq!(window.interval().unwrap(), (290.0, 295.0));
        assert_eq!(window.history(1), Some(285.0));
        assert_eq!(window.len(), 5);
    }

    #[test]
    fn test_get_four_box_window() {
        let item = create_four_box_item("Temperature");
        // Time 2001.0 corresponds to index 1 in the timeseries [2000, 2001, 2002]
        let state = InputState::build(vec![&item], 2001.0);

        let window = state.get_four_box_window("Temperature");

        // get() returns values at index 1 (2001 values: [16.0, 15.0, 11.0, 10.0])
        assert_eq!(window.get(FourBoxRegion::NorthernOcean), 16.0);
        assert_eq!(window.get(FourBoxRegion::SouthernLand), 10.0);
        assert_eq!(window.get_all(), vec![16.0, 15.0, 11.0, 10.0]);
        // history(1) is index 0 (2000 values: [15.0, 14.0, 10.0, 9.0])
        assert_eq!(window.history(1), Some(vec![15.0, 14.0, 10.0, 9.0]));
    }

    #[test]
    fn test_get_hemispheric_window() {
        let item = create_hemispheric_item("Precipitation");
        // Time 2001.0 corresponds to index 1 in the timeseries [2000, 2001]
        let state = InputState::build(vec![&item], 2001.0);

        let window = state.get_hemispheric_window("Precipitation");

        // get() returns values at index 1 (2001 values: [1100.0, 550.0])
        assert_eq!(window.get(HemisphericRegion::Northern), 1100.0);
        assert_eq!(window.get(HemisphericRegion::Southern), 550.0);
        assert_eq!(state.get_global("Precipitation"), Some(825.0)); // Equal weights mean
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

    // =========================================================================
    // Unit conversion integration tests through InputState
    // =========================================================================

    #[test]
    fn test_input_state_scalar_window_with_unit_conversion() {
        use std::collections::HashMap;

        let item = create_scalar_item("Emissions", vec![1.0, 2.0, 3.0]);

        // Create transform context with unit conversion factor
        let mut read_transforms = HashMap::new();
        read_transforms.insert(
            "Emissions".to_string(),
            ReadTransformInfo::with_unit_conversion(1000.0), // Convert Gt to Mt
        );
        let transform_ctx = TransformContext { read_transforms };

        let state = InputState::build_with_transforms(vec![&item], 2001.0, transform_ctx);

        let window = state.get_scalar_window("Emissions");

        // Raw value at index 1 is 2.0, converted should be 2000.0
        assert_eq!(window.get(), 2000.0);
        assert_eq!(window.interval().unwrap(), (2000.0, 3000.0));
        assert_eq!(window.history(1), Some(1000.0));
    }

    #[test]
    fn test_input_state_four_box_window_with_unit_conversion() {
        use std::collections::HashMap;

        let item = create_four_box_item("Temperature");

        // Create transform context with unit conversion factor
        let mut read_transforms = HashMap::new();
        read_transforms.insert(
            "Temperature".to_string(),
            ReadTransformInfo::with_unit_conversion(0.1), // Convert from deci-degrees to degrees
        );
        let transform_ctx = TransformContext { read_transforms };

        let state = InputState::build_with_transforms(vec![&item], 2001.0, transform_ctx);

        let window = state.get_four_box_window("Temperature");

        // Raw values at index 1: [16.0, 15.0, 11.0, 10.0]
        // Converted: [1.6, 1.5, 1.1, 1.0]
        assert_eq!(window.get(FourBoxRegion::NorthernOcean), 1.6);
        assert_eq!(window.get_all(), vec![1.6, 1.5, 1.1, 1.0]);
    }

    #[test]
    fn test_input_state_four_box_to_scalar_with_unit_conversion() {
        use crate::component::GridType;
        use std::collections::HashMap;

        let item = create_four_box_item("Forcing");

        // Create transform context with grid aggregation AND unit conversion
        let mut read_transforms = HashMap::new();
        read_transforms.insert(
            "Forcing".to_string(),
            ReadTransformInfo::new(GridType::FourBox, None).with_conversion_factor(2.0),
        );
        let transform_ctx = TransformContext { read_transforms };

        let state = InputState::build_with_transforms(vec![&item], 2001.0, transform_ctx);

        let window = state.get_scalar_window("Forcing");

        // Raw values at index 1: [16.0, 15.0, 11.0, 10.0], mean = 13.0
        // Converted: 13.0 * 2.0 = 26.0
        assert_eq!(window.get(), 26.0);
    }

    #[test]
    fn test_input_state_hemispheric_window_with_unit_conversion() {
        use crate::component::GridType;
        use std::collections::HashMap;

        let item = create_hemispheric_item("Precipitation");

        // Create transform context with unit conversion
        let mut read_transforms = HashMap::new();
        read_transforms.insert(
            "Precipitation".to_string(),
            ReadTransformInfo {
                source_grid: GridType::Hemispheric,
                weights: None,
                unit_conversion_factor: 0.001, // Convert mm/yr to m/yr
                variable_source: VariableSource::Exogenous,
                timeseries_type: TimeseriesType::Point,
            },
        );
        let transform_ctx = TransformContext { read_transforms };

        let state = InputState::build_with_transforms(vec![&item], 2001.0, transform_ctx);

        let window = state.get_hemispheric_window("Precipitation");

        // Raw values at index 1: [1100.0, 550.0]
        // Converted: [1.1, 0.55]
        assert_eq!(window.get(HemisphericRegion::Northern), 1.1);
        assert_eq!(window.get(HemisphericRegion::Southern), 0.55);
    }

    #[test]
    fn test_input_state_four_box_to_hemispheric_with_unit_conversion() {
        use crate::component::GridType;
        use std::collections::HashMap;

        let item = create_four_box_item("Temperature");

        // Create transform context to aggregate FourBox -> Hemispheric with conversion
        let mut read_transforms = HashMap::new();
        read_transforms.insert(
            "Temperature".to_string(),
            ReadTransformInfo::new(GridType::FourBox, None).with_conversion_factor(2.0),
        );
        let transform_ctx = TransformContext { read_transforms };

        let state = InputState::build_with_transforms(vec![&item], 2001.0, transform_ctx);

        let window = state.get_hemispheric_window("Temperature");

        // Raw values at index 1: [16.0, 15.0, 11.0, 10.0]
        // Hemispheric aggregation: Northern = (16+15)/2 = 15.5, Southern = (11+10)/2 = 10.5
        // With 2.0 conversion: [31.0, 21.0]
        assert_eq!(window.get(HemisphericRegion::Northern), 31.0);
        assert_eq!(window.get(HemisphericRegion::Southern), 21.0);
    }

    #[test]
    fn test_input_state_windows_carry_timeseries_type() {
        use crate::component::GridType;
        use std::collections::HashMap;

        let emissions = create_scalar_item("Emissions", vec![1.0, 2.0, 3.0]);
        let temperature = create_four_box_item("Temperature");
        let read_transforms = HashMap::from([
            (
                "Emissions".to_string(),
                ReadTransformInfo::default().with_timeseries_type(TimeseriesType::Average),
            ),
            (
                "Temperature".to_string(),
                ReadTransformInfo::new(GridType::FourBox, None)
                    .with_source(VariableSource::UpstreamOutput),
            ),
        ]);
        let state = InputState::build_with_transforms(
            vec![&emissions, &temperature],
            2001.0,
            TransformContext { read_transforms },
        );

        let emissions = state.get_scalar_window("Emissions");
        assert_eq!(emissions.timeseries_type(), TimeseriesType::Average);
        assert!(emissions.interval().is_err());

        let temperature = state.get_hemispheric_window("Temperature");
        assert_eq!(temperature.timeseries_type(), TimeseriesType::Point);
        assert_eq!(temperature.get_time(), 2002.0);
        assert_eq!(
            temperature.interval().unwrap(),
            (vec![15.5, 10.5], vec![16.5, 11.5])
        );
        assert_eq!(
            state
                .get_four_box_window("Temperature")
                .interval()
                .unwrap()
                .1,
            vec![17.0, 16.0, 12.0, 11.0]
        );
    }
}
