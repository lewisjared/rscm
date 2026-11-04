//! Total and efficacy-adjusted forcing (FaIR steps 14 and 15)
//!
//! Sums each forcing agent at the end of the step, skipping missing values,
//! with and without its efficacy.
//! Prescribed agents can carry a feedback on the surface temperature at the start of the step,
//! which calculated agents apply themselves.

use crate::constants::{
    efficacy_adjusted_forcing, end, forcing_unit, one, start, surface_temperature,
    temperature_unit, total_forcing,
};
use rscm_core::component::{Component, InputState, OutputState, RequirementDefinition};
use rscm_core::errors::RSCMResult;
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};

/// One forcing agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ForcingAgent {
    /// Forcing input.
    pub forcing: String,
    /// Efficacy relative to CO2.
    #[serde(default = "one")]
    pub efficacy: FloatValue,
    /// Forcing per kelvin of surface temperature, added to the input.
    #[serde(default)]
    pub temperature_feedback: FloatValue,
}

/// Parameters of [`FairForcingAggregation`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FairForcingAggregationParameters {
    /// The agents.
    pub agents: Vec<ForcingAgent>,
    /// Total forcing output.
    #[serde(default = "total_forcing")]
    pub total: String,
    /// Efficacy-adjusted total forcing output.
    #[serde(default = "efficacy_adjusted_forcing")]
    pub efficacy_adjusted: String,
    /// Surface temperature input, read when an agent has a feedback.
    #[serde(default = "surface_temperature")]
    pub surface_temperature: String,
}

/// FaIR's forcing sums.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairForcingAggregation {
    parameters: FairForcingAggregationParameters,
}

impl FairForcingAggregation {
    /// Create the component.
    pub fn from_parameters(parameters: FairForcingAggregationParameters) -> Self {
        Self { parameters }
    }

    fn has_feedback(&self) -> bool {
        self.parameters
            .agents
            .iter()
            .any(|a| a.temperature_feedback != 0.0)
    }
}

#[typetag::serde]
impl Component for FairForcingAggregation {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        let p = &self.parameters;
        let mut definitions: Vec<_> = p
            .agents
            .iter()
            .map(|a| RequirementDefinition::scalar_input(&a.forcing, &forcing_unit()))
            .collect();
        if self.has_feedback() {
            definitions.push(RequirementDefinition::scalar_input(
                &p.surface_temperature,
                &temperature_unit(),
            ));
        }
        definitions.push(RequirementDefinition::scalar_output(
            &p.total,
            &forcing_unit(),
        ));
        definitions.push(RequirementDefinition::scalar_output(
            &p.efficacy_adjusted,
            &forcing_unit(),
        ));
        definitions
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let p = &self.parameters;
        let temperature = if self.has_feedback() {
            start(input_state, &p.surface_temperature)
        } else {
            0.0
        };
        let mut total = 0.0;
        let mut adjusted = 0.0;
        for agent in &p.agents {
            let forcing =
                end(input_state, &agent.forcing) + temperature * agent.temperature_feedback;
            if !forcing.is_nan() {
                total += forcing;
                adjusted += forcing * agent.efficacy;
            }
        }
        let mut output = OutputState::with_capacity(2);
        output.insert(p.total.clone(), StateValue::Scalar(total));
        output.insert(p.efficacy_adjusted.clone(), StateValue::Scalar(adjusted));
        Ok(output)
    }
}
