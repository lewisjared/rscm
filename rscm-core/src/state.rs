use crate::spatial::ScalarRegion;
use crate::timeseries::{FloatValue, Time};
use crate::timeseries_collection::{TimeseriesItem, VariableType};
use num::Float;
use std::collections::HashMap;

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

pub type OutputState = HashMap<String, FloatValue>;
