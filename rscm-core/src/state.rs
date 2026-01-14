use crate::spatial::ScalarRegion;
use crate::timeseries::{FloatValue, Time};
use crate::timeseries_collection::{TimeseriesItem, VariableType};
use num::Float;
use std::collections::HashMap;

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
    /// For grid variables, use `get_latest_grid()` or `get_latest_global()`.
    ///
    /// # Panics
    /// Panics if the variable is not found in the state.
    pub fn get_latest(&self, name: &str) -> FloatValue {
        let item = self
            .iter()
            .find(|item| item.name == name)
            .expect("No item found");

        match item.variable_type {
            VariableType::Exogenous => item
                .timeseries
                .at_time(self.current_time, ScalarRegion::Global)
                .unwrap(),
            VariableType::Endogenous => item.timeseries.latest_value().unwrap(),
        }
    }

    /// Get the latest value as a StateValue (scalar or grid)
    ///
    /// For grid timeseries, returns all regional values.
    /// For scalar timeseries, returns a single value wrapped in StateValue::Scalar.
    pub fn get_latest_value(&self, name: &str) -> Option<StateValue> {
        let item = self.iter().find(|item| item.name == name)?;

        // For now, all timeseries are scalar since TimeseriesItem holds Timeseries<FloatValue>
        // In the future when we support grid timeseries in TimeseriesCollection,
        // this would check the grid type and return Grid variant
        let value = match item.variable_type {
            VariableType::Exogenous => item
                .timeseries
                .at_time(self.current_time, ScalarRegion::Global)
                .ok()?,
            VariableType::Endogenous => item.timeseries.latest_value()?,
        };

        Some(StateValue::Scalar(value))
    }

    /// Get the global aggregated value for a variable
    ///
    /// For scalar variables, returns the scalar value.
    /// For grid variables, aggregates all regions to a single global value using the grid's weights.
    pub fn get_global(&self, name: &str) -> Option<FloatValue> {
        self.get_latest_value(name).map(|sv| match sv {
            StateValue::Scalar(v) => v,
            StateValue::Grid(_) => {
                // In the future, this would aggregate using grid weights
                // For now, we only have scalar values
                unreachable!("Grid values not yet supported in TimeseriesCollection")
            }
        })
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

    /// Test if the state contains a value with the given name
    pub fn has(&self, name: &str) -> bool {
        self.state.iter().any(|x| x.name == name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &&TimeseriesItem> {
        self.state.iter()
    }

    /// Converts the state into an equivalent hashmap
    pub fn to_hashmap(self) -> HashMap<String, FloatValue> {
        HashMap::from_iter(
            self.state
                .into_iter()
                .map(|item| (item.name.clone(), item.timeseries.latest_value().unwrap())),
        )
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
            timeseries: ts,
            name: "CO2".to_string(),
            variable_type: VariableType::Endogenous,
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
            timeseries: ts,
            name: "Temperature".to_string(),
            variable_type: VariableType::Endogenous,
        };

        let state = InputState::build(vec![&item], 2000.5);

        // For Endogenous variables, returns latest_value (index 1)
        // Scalar values accessible as region 0
        assert_eq!(state.get_region("Temperature", 0), Some(16.0));
        // Other regions return None for scalar
        assert_eq!(state.get_region("Temperature", 1), None);
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
