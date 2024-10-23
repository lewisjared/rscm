#![allow(dead_code)]

use crate::component::{
    Component, InputState, OutputState, RequirementDefinition, RequirementType,
};
use crate::errors::RSCMResult;
use crate::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TestComponentParameters {
    pub p: FloatValue,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TestComponent {
    parameters: TestComponentParameters,
}

impl TestComponent {
    pub fn from_parameters(parameters: TestComponentParameters) -> Self {
        Self { parameters }
    }
}

#[typetag::serde]
impl Component for TestComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::new("Emissions|CO2", "GtCO2", RequirementType::Input),
            RequirementDefinition::new("Concentrations|CO2", "ppm", RequirementType::Output),
        ]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let emission_co2 = input_state.get_latest("Emissions|CO2");

        println!("Solving {:?} with state: {:?}", self, input_state);

        let mut output_state = OutputState::new();
        output_state.insert(
            "Concentrations|CO2".to_string(),
            emission_co2 * self.parameters.p,
        );
        Ok(output_state)
    }
}
