//! Total and efficacy-adjusted effective radiative forcing (FaIR steps 14 and 15)
//!
//! Sums the effective radiative forcing of every forcing agent into a total,
//! an efficacy-adjusted total and the greenhouse gas, aerosol and anthropogenic subtotals.
//!
//! It reads each agent at the end of the step and skips missing values.
//! A prescribed agent reads its effective radiative forcing from a separate input,
//! adds a feedback on the surface air temperature change at the start of the step
//! and writes the result as its forcing.
//! A calculated agent applies its own feedback in the component that produces it.
//! Each subtotal sums the agents whose `category` counts towards it.
//!
//! # Inputs
//!
//! - The `forcing` of each calculated agent (W/m^2).
//! - The `prescribed` input of each prescribed agent (W/m^2).
//!   rscm.fair names it with the `Prescribed` qualifier,
//!   such as `Prescribed Effective Radiative Forcing|Solar`.
//! - The surface air temperature change named by `surface_temperature` (K),
//!   only when an agent has a nonzero `temperature_feedback`.
//!   Its default is `Surface Air Temperature Change`.
//!
//! # Outputs
//!
//! - The `forcing` of each prescribed agent (W/m^2), its prescribed input plus the feedback.
//! - The total named by `total` (W/m^2), by default `Effective Radiative Forcing`.
//! - The efficacy-adjusted total named by `efficacy_adjusted` (W/m^2),
//!   by default `Efficacy Adjusted Effective Radiative Forcing`.
//! - The subtotals `Effective Radiative Forcing|Greenhouse Gases`,
//!   `Effective Radiative Forcing|Aerosols`
//!   and `Effective Radiative Forcing|Anthropogenic` (W/m^2).

use crate::constants::{
    efficacy_adjusted_forcing, forcing_unit, get, one, surface_temperature, temperature_unit,
    total_forcing,
};
use rscm_core::component::{Component, InputState, OutputState, VariableDefinition};
use rscm_core::errors::RSCMResult;
use rscm_core::forcing::{ForcingCategory, ForcingSubtotals};
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentMetadata;
use serde::{Deserialize, Serialize};

/// One forcing agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ForcingAgent {
    /// Name of the agent's effective radiative forcing,
    /// an output when `prescribed` is set and an input otherwise.
    pub forcing: String,
    /// Subtotals the agent's effective radiative forcing counts towards.
    pub category: ForcingCategory,
    /// Name of the prescribed effective radiative forcing input,
    /// for an agent no component calculates.
    #[serde(default)]
    pub prescribed: Option<String>,
    /// Efficacy relative to CO2 (FaIR `forcing_efficacy`).
    #[serde(default = "one")]
    pub efficacy: FloatValue,
    /// Effective radiative forcing per kelvin of surface air temperature change,
    /// added to the prescribed input (FaIR `forcing_temperature_feedback`).
    /// unit: W/m^2/K
    #[serde(default)]
    pub temperature_feedback: FloatValue,
}

/// Parameters of [`ForcingAggregation`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ForcingAggregationParameters {
    /// Forcing agents, in the order they are summed.
    pub agents: Vec<ForcingAgent>,
    /// Name of the total effective radiative forcing output.
    #[serde(default = "total_forcing")]
    pub total: String,
    /// Name of the efficacy-adjusted total effective radiative forcing output.
    #[serde(default = "efficacy_adjusted_forcing")]
    pub efficacy_adjusted: String,
    /// Name of the surface air temperature change input, read only when an agent has a feedback.
    #[serde(default = "surface_temperature")]
    pub surface_temperature: String,
}

/// FaIR's effective radiative forcing sums.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentMetadata)]
#[component(tags = ["forcing", "aggregation", "fair"], category = "Radiative Forcing")]
pub struct ForcingAggregation {
    parameters: ForcingAggregationParameters,
}

impl ForcingAggregation {
    /// Create the component.
    pub fn from_parameters(parameters: ForcingAggregationParameters) -> Self {
        Self { parameters }
    }

    fn has_feedback(&self) -> bool {
        self.parameters
            .agents
            .iter()
            .any(|a| a.temperature_feedback != 0.0)
    }
}

#[typetag::serde(name = "fair.ForcingAggregation")]
impl Component for ForcingAggregation {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let p = &self.parameters;
        let mut definitions = Vec::with_capacity(p.agents.len() + 6);
        for a in &p.agents {
            match &a.prescribed {
                Some(input) => {
                    definitions.push(VariableDefinition::scalar_input(input, &forcing_unit()));
                    definitions.push(VariableDefinition::scalar_output(
                        &a.forcing,
                        &forcing_unit(),
                    ));
                }
                None => definitions.push(VariableDefinition::scalar_input(
                    &a.forcing,
                    &forcing_unit(),
                )),
            }
        }
        if self.has_feedback() {
            definitions.push(VariableDefinition::scalar_input(
                &p.surface_temperature,
                &temperature_unit(),
            ));
        }
        definitions.push(VariableDefinition::scalar_output(&p.total, &forcing_unit()));
        definitions.extend(ForcingSubtotals::definitions(&forcing_unit()));
        definitions.push(VariableDefinition::scalar_output(
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
            get(input_state, &p.surface_temperature)
        } else {
            0.0
        };
        let mut output = OutputState::with_capacity(p.agents.len() + 5);
        let mut total = 0.0;
        let mut adjusted = 0.0;
        let mut subtotals = ForcingSubtotals::default();
        for agent in &p.agents {
            let input = agent.prescribed.as_ref().unwrap_or(&agent.forcing);
            let forcing = get(input_state, input) + temperature * agent.temperature_feedback;
            if agent.prescribed.is_some() {
                output.insert(agent.forcing.clone(), StateValue::Scalar(forcing));
            }
            if !forcing.is_nan() {
                total += forcing;
                adjusted += forcing * agent.efficacy;
                subtotals = subtotals + agent.category.subtotals(forcing);
            }
        }
        output.insert(p.total.clone(), StateValue::Scalar(total));
        output.insert(p.efficacy_adjusted.clone(), StateValue::Scalar(adjusted));
        subtotals.insert_into(&mut output);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rscm_core::names::{ERF_AEROSOLS, ERF_ANTHROPOGENIC, ERF_GREENHOUSE_GASES};
    use rscm_core::timeseries::Timeseries;
    use rscm_core::timeseries_collection::{TimeseriesData, TimeseriesItem};

    fn agent(forcing: &str, category: ForcingCategory) -> ForcingAgent {
        ForcingAgent {
            forcing: forcing.to_string(),
            category,
            prescribed: None,
            efficacy: 2.0,
            temperature_feedback: 0.0,
        }
    }

    fn item(name: &str, value: FloatValue) -> TimeseriesItem {
        TimeseriesItem {
            data: TimeseriesData::Scalar(Timeseries::from_values(
                ndarray::array![value, value],
                ndarray::array![2000.0, 2001.0],
            )),
            name: name.to_string(),
        }
    }

    #[test]
    fn subtotals_skip_missing_forcing() {
        let c = ForcingAggregation::from_parameters(ForcingAggregationParameters {
            agents: vec![
                agent("co2", ForcingCategory::GreenhouseGas),
                agent("ari", ForcingCategory::Aerosol),
                agent("aci", ForcingCategory::Aerosol),
                agent("ozone", ForcingCategory::Anthropogenic),
                agent("solar", ForcingCategory::Natural),
            ],
            total: total_forcing(),
            efficacy_adjusted: efficacy_adjusted_forcing(),
            surface_temperature: surface_temperature(),
        });
        let items = [
            item("co2", 2.0),
            item("ari", -0.25),
            item("aci", FloatValue::NAN),
            item("ozone", 0.5),
            item("solar", 0.125),
        ];
        let input_state = InputState::build(items.iter().collect(), 2000.0);
        let output = c.solve(2000.0, 2001.0, &input_state).unwrap();
        let scalar = |name: &str| match &output[name] {
            StateValue::Scalar(value) => *value,
            other => panic!("expected a scalar, got {other:?}"),
        };
        assert_eq!(scalar(ERF_GREENHOUSE_GASES), 2.0);
        assert_eq!(scalar(ERF_AEROSOLS), -0.25);
        assert_eq!(scalar(ERF_ANTHROPOGENIC), 2.25);
        assert_eq!(scalar(&total_forcing()), 2.375);
        assert_eq!(scalar(&efficacy_adjusted_forcing()), 4.75);
    }
}
