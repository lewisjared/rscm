use crate::errors::RSCMResult;
pub use crate::state::{InputState, OutputState};
use crate::timeseries::Time;
use pyo3::pyclass;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

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
    use crate::timeseries::Timeseries;
    use crate::timeseries_collection::{TimeseriesItem, VariableType};
    use ndarray::array;

    #[test]
    fn solve() {
        let component = TestComponent::from_parameters(TestComponentParameters { p: 2.0 });

        let emissions_co2 = TimeseriesItem {
            timeseries: Timeseries::from_values(array![1.1, 1.3], array![2020.0, 2021.0]),
            name: "Emissions|CO2".to_string(),
            variable_type: VariableType::Exogenous,
        };

        let input_state = InputState::build(vec![&emissions_co2], 2020.0);
        assert_eq!(input_state.get_latest("Emissions|CO2"), 1.1);

        let output_state = component.solve(2020.0, 2021.0, &input_state).unwrap();

        assert_eq!(*output_state.get("Concentrations|CO2").unwrap(), 1.1 * 2.0);
    }
}
