use crate::errors::RSCMResult;
use crate::timeseries::{FloatValue, Time};
use crate::timeseries_collection::{TimeseriesCollection, VariableType};
use pyo3::pyclass;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::zip;

/// Generic state representation
///
/// A state is a collection of values
/// that can be used to represent the state of a system at a given time.
///
/// This is very similar to a Hashmap (with likely worse performance),
/// but provides strong type separation.
pub trait State<T> {
    fn get(&self, name: &str) -> &T;
}

#[derive(Debug, Clone)]
pub struct InputState {
    state: Vec<(String, FloatValue)>,
}

impl InputState {
    pub fn from_vectors(values: Vec<FloatValue>, names: Vec<String>) -> Self {
        assert_eq!(values.len(), names.len());
        let state: Vec<(String, FloatValue)> = zip(names, values).collect();
        Self { state }
    }

    pub fn empty() -> Self {
        Self { state: vec![] }
    }

    pub fn from_hashmap(items: HashMap<String, FloatValue>) -> Self {
        let mut state = vec![];
        items.into_iter().for_each(|(name, value)| {
            state.push((name, value));
        });
        Self { state }
    }

    pub fn from_hashmap_and_verify(
        items: HashMap<String, FloatValue>,
        expected_items: Vec<String>,
    ) -> Self {
        let mut keys: Vec<&String> = items.keys().collect();
        keys.sort_unstable();
        let mut expected_items: Vec<&String> = expected_items.iter().collect();
        expected_items.sort_unstable();

        assert_eq!(keys, expected_items);

        Self::from_hashmap(items)
    }

    pub fn has(&self, name: &str) -> bool {
        self.state.iter().any(|(n, _)| *n == name)
    }

    /// Merge state into this state
    ///
    /// Overrides any existing values with the same name
    pub fn merge(&mut self, state: InputState) -> &mut Self {
        state.into_iter().for_each(|(key, value)| {
            let existing = self.state.iter_mut().find(|(n, _)| *n == key);

            match existing {
                Some(item) => *item = (key, value),
                None => self.state.push((key, value)),
            }
        });
        self
    }

    pub fn iter(&self) -> impl Iterator<Item = &(String, FloatValue)> {
        self.state.iter()
    }

    /// Converts the state into an equivalent hashmap
    pub fn to_hashmap(self) -> HashMap<String, FloatValue> {
        HashMap::from_iter(self.state)
    }
}
impl State<FloatValue> for InputState {
    fn get(&self, name: &str) -> &FloatValue {
        let found = self.state.iter().find(|(n, _)| *n == name).map(|(_, v)| v);
        match found {
            Some(val) => val,
            None => panic!("No state named {} found in {:?}", name, self),
        }
    }
}

impl IntoIterator for InputState {
    type Item = (String, FloatValue);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.state.into_iter()
    }
}

pub type OutputState = InputState;

#[pyclass]
#[derive(Debug, Eq, PartialEq, Clone, Hash, Serialize, Deserialize)]
pub enum RequirementType {
    Input,
    Output,
    InputAndOutput, // TODO: Figure out how to compose input and output together
    EmptyLink,
}

#[pyclass]
#[derive(Debug, Eq, PartialEq, Clone, Hash, Serialize, Deserialize)]
pub struct RequirementDefinition {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub unit: String,
    #[pyo3(get, set)]
    pub requirement_type: RequirementType,
}

impl RequirementDefinition {
    pub fn new(name: &str, unit: &str, requirement_type: RequirementType) -> Self {
        Self {
            name: name.to_string(),
            unit: unit.to_string(),
            requirement_type,
        }
    }
}

/// Component of a reduced complexity climate model
///
/// Each component encapsulates some set of physics that can be solved for a given time step.
/// Generally these components can be modelled as a set of Ordinary Differential Equations (ODEs)
/// with an input state that can be solved as an initial value problem over a given time domain.
///
/// The resulting state of a component can then be used by other components as part of a `Model`
/// or solved alone during calibration.
///
/// Each component contains:
/// * parameters: Time invariant constants used to parameterize the components physics
/// * inputs: State information required to solve the model. This come from either other
///   components as part of a coupled system or from exogenous data.
/// * outputs: Information that is solved by the component
///
/// Structs implementing the `Component` trait should be serializable and deserializable
/// and use the `#[typetag::serde]` macro when implementing the trait to enable
/// serialisation/deserialisation when using `Component` as an object trait
/// (i.e. where `dyn Component` is used; see `models.rs`).
#[typetag::serde(tag = "type")]
pub trait Component: Debug + Send + Sync {
    fn definitions(&self) -> Vec<RequirementDefinition>;

    /// Variables that are required to solve this component
    fn inputs(&self) -> Vec<RequirementDefinition> {
        self.definitions()
            .iter()
            .filter(|d| {
                (d.requirement_type == RequirementType::Input)
                    || (d.requirement_type == RequirementType::InputAndOutput)
            })
            .cloned()
            .collect()
    }
    fn input_names(&self) -> Vec<String> {
        self.inputs().into_iter().map(|d| d.name).collect()
    }

    /// Variables that are solved by this component
    ///
    /// The names of the solved variables must be unique for a given model.
    /// i.e. No two components within a model can produce the same variable names.
    /// These names can contain '|' to namespace variables to avoid collisions,
    /// for example, 'Emissions|CO2' and 'Atmospheric Concentrations|CO2'
    fn outputs(&self) -> Vec<RequirementDefinition> {
        self.definitions()
            .iter()
            .filter(|d| {
                (d.requirement_type == RequirementType::Output)
                    || (d.requirement_type == RequirementType::InputAndOutput)
            })
            .cloned()
            .collect()
    }
    fn output_names(&self) -> Vec<String> {
        self.outputs().into_iter().map(|d| d.name).collect()
    }

    /// Extract the input state for the current time step
    ///
    /// By default, for endogenous variables which are calculated as part of the model
    /// the most recent value is used, whereas, for exogenous variables the values are linearly
    /// interpolated.
    /// This ensures that state calculated from previous components within the same timestep
    /// is used.
    ///
    /// The result should contain values for the current time step for all input variable
    fn extract_state(&self, collection: &TimeseriesCollection, t_current: Time) -> InputState {
        let mut state = HashMap::new();

        self.input_names().into_iter().for_each(|name| {
            let ts = collection
                .get_by_name(name.as_str())
                .unwrap_or_else(|| panic!("No timeseries with variable='{}'", name));

            let result = match ts.variable_type {
                VariableType::Exogenous => ts.timeseries.at_time(t_current).unwrap(),
                VariableType::Endogenous => ts.timeseries.latest_value().unwrap(),
            };
            state.insert(name, result);
        });

        InputState::from_hashmap_and_verify(state, self.input_names())
    }

    /// Solve the component until `t_next`
    ///
    /// The result should contain values for the current time step for all output variables
    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example_components::{TestComponent, TestComponentParameters};

    #[test]
    fn solve() {
        let component = TestComponent::from_parameters(TestComponentParameters { p: 2.0 });

        let input_state = component.extract_state(&TimeseriesCollection::new(), 2020.0);
        let output_state = component.solve(2020.0, 2021.0, &input_state).unwrap();

        assert_eq!(*output_state.get("Concentrations|CO2"), 2.0 * 1.3);
    }
}
