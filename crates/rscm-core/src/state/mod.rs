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
pub use windows::{GridTimeseriesWindow, TimeseriesWindow};

// Re-export aggregating window types
pub use aggregating::{
    AggregatingFourBoxToHemisphericWindow, AggregatingFourBoxWindow, AggregatingHemisphericWindow,
    HemisphericWindow, ReadTransformInfo, ScalarWindow,
};

use crate::component::GridType;
use crate::spatial::{FourBoxGrid, ScalarRegion, SpatialGrid};
use crate::timeseries::{FloatValue, Time};
use crate::timeseries_collection::{TimeseriesData, TimeseriesItem, VariableType};
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
    state: Vec<&'a TimeseriesItem>,
    /// Optional transform context for grid aggregation
    transform_context: Option<TransformContext>,
}

impl<'a> InputState<'a> {
    pub fn build(values: Vec<&'a TimeseriesItem>, current_time: Time) -> Self {
        Self {
            current_time,
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
            state: values,
            transform_context: Some(transform_context),
        }
    }

    pub fn empty() -> Self {
        Self {
            current_time: f64::NAN,
            state: vec![],
            transform_context: None,
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
    /// If the underlying data is stored at a finer grid resolution (FourBox or Hemispheric)
    /// and a read transform is configured, the data will be automatically aggregated
    /// to a scalar value on each access.
    ///
    /// # Panics
    ///
    /// Panics if the variable is not found or cannot be accessed as scalar.
    pub fn get_scalar_window(&self, name: &str) -> ScalarWindow<'_> {
        let item = self
            .iter()
            .find(|item| item.name == name)
            .unwrap_or_else(|| panic!("Variable '{}' not found in input state", name));

        // Check if there's a read transform for this variable
        let transform = self
            .transform_context
            .as_ref()
            .and_then(|ctx| ctx.read_transforms.get(name));

        // If there's a transform, use the source grid type
        if let Some(transform) = transform {
            match transform.source_grid {
                GridType::FourBox => {
                    let ts = item.data.as_four_box().unwrap_or_else(|| {
                        panic!(
                            "Variable '{}' requires FourBox->Scalar transform but is not FourBox",
                            name
                        )
                    });

                    let current_index =
                        ts.time_axis()
                            .index_of(self.current_time)
                            .unwrap_or_else(|| {
                                panic!(
                                    "Time {} not found in timeseries '{}' time axis",
                                    self.current_time, name
                                )
                            });

                    return ScalarWindow::FromFourBox(AggregatingFourBoxWindow::new(
                        ts,
                        current_index,
                        self.current_time,
                        transform.weights.clone(),
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

                    let current_index =
                        ts.time_axis()
                            .index_of(self.current_time)
                            .unwrap_or_else(|| {
                                panic!(
                                    "Time {} not found in timeseries '{}' time axis",
                                    self.current_time, name
                                )
                            });

                    return ScalarWindow::FromHemispheric(AggregatingHemisphericWindow::new(
                        ts,
                        current_index,
                        self.current_time,
                        transform.weights.clone(),
                    ));
                }
                GridType::Scalar => {
                    // No transform needed, fall through to direct access
                }
            }
        }

        // Direct scalar access (no transform needed)
        let ts = item
            .data
            .as_scalar()
            .unwrap_or_else(|| panic!("Variable '{}' is not a scalar timeseries", name));

        let current_index = ts
            .time_axis()
            .index_of(self.current_time)
            .unwrap_or_else(|| {
                panic!(
                    "Time {} not found in timeseries '{}' time axis",
                    self.current_time, name
                )
            });

        ScalarWindow::Direct(TimeseriesWindow::new(ts, current_index, self.current_time))
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

        // Find the index corresponding to current_time in the timeseries
        let current_index = ts
            .time_axis()
            .index_of(self.current_time)
            .unwrap_or_else(|| {
                panic!(
                    "Time {} not found in timeseries '{}' time axis",
                    self.current_time, name
                )
            });

        GridTimeseriesWindow::new(ts, current_index, self.current_time)
    }

    /// Get a Hemispheric GridTimeseriesWindow for the named variable
    ///
    /// # Panics
    ///
    /// If the underlying data is stored at FourBox resolution and a read transform
    /// is configured, the data will be automatically aggregated to Hemispheric
    /// on each access.
    ///
    /// # Panics
    ///
    /// Panics if the variable is not found or cannot be accessed as Hemispheric.
    pub fn get_hemispheric_window(&self, name: &str) -> HemisphericWindow<'_> {
        let item = self
            .iter()
            .find(|item| item.name == name)
            .unwrap_or_else(|| panic!("Variable '{}' not found in input state", name));

        // Check if there's a read transform for this variable
        let transform = self
            .transform_context
            .as_ref()
            .and_then(|ctx| ctx.read_transforms.get(name));

        // If there's a transform from FourBox, aggregate
        if let Some(transform) = transform {
            if transform.source_grid == GridType::FourBox {
                let ts = item.data.as_four_box().unwrap_or_else(|| {
                    panic!(
                        "Variable '{}' requires FourBox->Hemispheric transform but is not FourBox",
                        name
                    )
                });

                let current_index =
                    ts.time_axis()
                        .index_of(self.current_time)
                        .unwrap_or_else(|| {
                            panic!(
                                "Time {} not found in timeseries '{}' time axis",
                                self.current_time, name
                            )
                        });

                return HemisphericWindow::FromFourBox(AggregatingFourBoxToHemisphericWindow::new(
                    ts,
                    current_index,
                    self.current_time,
                ));
            }
        }

        // Direct hemispheric access
        let ts = item
            .data
            .as_hemispheric()
            .unwrap_or_else(|| panic!("Variable '{}' is not a Hemispheric timeseries", name));

        let current_index = ts
            .time_axis()
            .index_of(self.current_time)
            .unwrap_or_else(|| {
                panic!(
                    "Time {} not found in timeseries '{}' time axis",
                    self.current_time, name
                )
            });

        HemisphericWindow::Direct(GridTimeseriesWindow::new(
            ts,
            current_index,
            self.current_time,
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

        // at_start() returns value at index corresponding to current_time (index 1)
        assert_eq!(state.get_global("CO2"), Some(285.0));
        assert_eq!(state.get_scalar_window("CO2").at_start(), 285.0);
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

        // Test get_four_box_window returns values at index 1 using at_start()
        let window = state.get_four_box_window("Temperature");
        let values = window.at_start_all();
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

        // at_start() returns the value at the index corresponding to current_time
        assert_eq!(window.at_start(), 290.0);
        assert_eq!(window.at_end().unwrap(), 295.0);
        assert_eq!(window.previous(), Some(285.0));
        assert_eq!(window.len(), 5);
    }

    #[test]
    fn test_get_four_box_window() {
        let item = create_four_box_item("Temperature");
        // Time 2001.0 corresponds to index 1 in the timeseries [2000, 2001, 2002]
        let state = InputState::build(vec![&item], 2001.0);

        let window = state.get_four_box_window("Temperature");

        // at_start() returns values at index 1 (2001 values: [16.0, 15.0, 11.0, 10.0])
        assert_eq!(window.at_start(FourBoxRegion::NorthernOcean), 16.0);
        assert_eq!(window.at_start(FourBoxRegion::SouthernLand), 10.0);
        assert_eq!(window.at_start_all(), vec![16.0, 15.0, 11.0, 10.0]);
        // previous is index 0 (2000 values: [15.0, 14.0, 10.0, 9.0])
        assert_eq!(window.previous_all(), Some(vec![15.0, 14.0, 10.0, 9.0]));
    }

    #[test]
    fn test_get_hemispheric_window() {
        let item = create_hemispheric_item("Precipitation");
        // Time 2001.0 corresponds to index 1 in the timeseries [2000, 2001]
        let state = InputState::build(vec![&item], 2001.0);

        let window = state.get_hemispheric_window("Precipitation");

        // at_start() returns values at index 1 (2001 values: [1100.0, 550.0])
        assert_eq!(window.at_start(HemisphericRegion::Northern), 1100.0);
        assert_eq!(window.at_start(HemisphericRegion::Southern), 550.0);
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
}
