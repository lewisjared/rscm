//! Methane lifetime from its chemical precursors and temperature (FaIR step 2)
//!
//! Following Thornhill et al. (2021), the CH4 lifetime scales by
//!
//! $$\alpha = \exp\Big(\sum_i \max(\ln(1 + s_i (x_i - x_{i,0})), \ln f)
//!   + \ln \max(1 + s_T T, \ln f)\Big)$$
//!
//! over precursor emissions and concentrations $x_i$ at the start of the step,
//! where $f$ is the floor `minimum_scaling`.

use crate::constants::{get, surface_temperature, temperature_unit};
use rscm_core::component::{
    Component, InputState, OutputState, TimeseriesType, VariableDefinition,
};
use rscm_core::errors::RscmResult;
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};

/// Default name of the CH4 lifetime scaling output.
pub const CH4_LIFETIME_SCALING: &str = "Lifetime Scaling|CH4";

/// A precursor of the CH4 lifetime.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Precursor {
    /// Emissions or concentration variable.
    pub variable: String,
    /// Its unit.
    pub unit: String,
    /// Value with no effect on the lifetime.
    pub baseline: FloatValue,
    /// Fractional lifetime change per unit.
    pub sensitivity: FloatValue,
}

/// Parameters of [`CH4Lifetime`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CH4LifetimeParameters {
    /// Precursor emissions, summed before the concentrations.
    pub emissions: Vec<Precursor>,
    /// Precursor concentrations.
    pub concentrations: Vec<Precursor>,
    /// Fractional lifetime change per kelvin.
    /// FaIR: `lifetime_temperature_sensitivity`
    pub temperature_sensitivity: FloatValue,
    /// Floor on each scaling term.
    /// FaIR: `_ch4_min`
    #[serde(default = "default_minimum_scaling")]
    pub minimum_scaling: FloatValue,
    /// Surface temperature input.
    #[serde(default = "surface_temperature")]
    pub surface_temperature: String,
    /// Lifetime scaling output.
    #[serde(default = "default_output")]
    pub output: String,
}

fn default_minimum_scaling() -> FloatValue {
    0.01
}

fn default_output() -> String {
    CH4_LIFETIME_SCALING.to_string()
}

/// Thornhill et al. (2021) CH4 lifetime scaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CH4Lifetime {
    parameters: CH4LifetimeParameters,
}

impl CH4Lifetime {
    /// Create the component.
    pub fn from_parameters(parameters: CH4LifetimeParameters) -> Self {
        Self { parameters }
    }

    /// Lifetime scaling for precursor values in the order of the parameters.
    pub fn alpha(
        &self,
        emissions: &[FloatValue],
        concentrations: &[FloatValue],
        temperature: FloatValue,
    ) -> FloatValue {
        let p = &self.parameters;
        let log_floor = p.minimum_scaling.ln();
        let term = |precursors: &[Precursor], values: &[FloatValue]| -> FloatValue {
            precursors
                .iter()
                .zip(values)
                .map(|(x, v)| (1.0 + (v - x.baseline) * x.sensitivity).ln().max(log_floor))
                .sum()
        };
        let log_scaling = term(&p.emissions, emissions)
            + term(&p.concentrations, concentrations)
            + (1.0 + temperature * p.temperature_sensitivity)
                .max(log_floor)
                .ln();
        log_scaling.exp()
    }
}

#[typetag::serde(name = "fair.CH4Lifetime")]
impl Component for CH4Lifetime {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let p = &self.parameters;
        let mut definitions = vec![VariableDefinition::scalar_input(
            &p.surface_temperature,
            &temperature_unit(),
        )];
        for x in p.emissions.iter().chain(&p.concentrations) {
            definitions.push(VariableDefinition::scalar_input(&x.variable, &x.unit));
        }
        definitions.push(
            VariableDefinition::scalar_output(&p.output, "1")
                .with_timeseries_type(TimeseriesType::Average),
        );
        definitions
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RscmResult<OutputState> {
        let p = &self.parameters;
        let get_of = |x: &Precursor| get(input_state, &x.variable);
        let emissions: Vec<_> = p.emissions.iter().map(get_of).collect();
        let concentrations: Vec<_> = p.concentrations.iter().map(get_of).collect();
        let temperature = get(input_state, &p.surface_temperature);
        let mut output = OutputState::with_capacity(1);
        output.insert(
            p.output.clone(),
            StateValue::Scalar(self.alpha(&emissions, &concentrations, temperature)),
        );
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn precursor(baseline: FloatValue, sensitivity: FloatValue) -> Precursor {
        Precursor {
            variable: "x".into(),
            unit: "1".into(),
            baseline,
            sensitivity,
        }
    }

    #[test]
    fn baseline_gives_unit_scaling() {
        let c = CH4Lifetime::from_parameters(CH4LifetimeParameters {
            emissions: vec![precursor(2.0, 0.1)],
            concentrations: vec![precursor(700.0, 0.0003)],
            temperature_sensitivity: -0.0408,
            minimum_scaling: 0.01,
            surface_temperature: surface_temperature(),
            output: default_output(),
        });
        assert_relative_eq!(c.alpha(&[2.0], &[700.0], 0.0), 1.0);
        assert_relative_eq!(
            c.alpha(&[3.0], &[700.0], 1.0),
            1.1 * (1.0 - 0.0408),
            max_relative = 1e-14
        );
    }

    #[test]
    fn terms_are_floored() {
        let c = CH4Lifetime::from_parameters(CH4LifetimeParameters {
            emissions: vec![precursor(0.0, -1.0)],
            concentrations: vec![],
            temperature_sensitivity: 0.0,
            minimum_scaling: 0.01,
            surface_temperature: surface_temperature(),
            output: default_output(),
        });
        assert_relative_eq!(c.alpha(&[0.999999], &[], 0.0), 0.01, max_relative = 1e-12);
    }
}
