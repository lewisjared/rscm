//! Methane lifetime from its chemical precursors and temperature (FaIR step 2)
//!
//! Calculates a scaling of the CH4 atmospheric lifetime from the emissions and concentrations
//! of its chemical precursors and from the surface air temperature change.
//! The method follows Thornhill et al. (2021).
//! [`crate::GasCycle`] reads the scaling when a gas sets `lifetime_scaling`,
//! as rscm.fair does for CH4 with the `thornhill2021` CH4 method.
//!
//! $$\alpha = \exp\Big(\sum_i \max(\ln(1 + s_i (x_i - x_{i,0})), \ln f)
//!   + \ln \max(1 + s_T T, \ln f)\Big)$$
//!
//! over precursor emissions and concentrations $x_i$ at the start of the step,
//! with sensitivities $s_i$ and baselines $x_{i,0}$.
//! $s_T$ is `temperature_sensitivity`, $T$ the surface air temperature change
//! and $f$ the floor `minimum_scaling`.
//!
//! # Inputs
//!
//! - The surface air temperature change named by `surface_temperature` (K).
//!   Its default is `Surface Air Temperature Change`.
//! - Each precursor's `variable`, in its `unit`, from `emissions` then `concentrations`.
//!   rscm.fair passes each species FaIR marks for aerosol chemistry,
//!   such as `Emissions|NOx` and `Atmospheric Concentrations|CH4`.
//!
//! # Outputs
//!
//! - The lifetime scaling named by `output` (1), an average over the step.
//!   Its default is `Lifetime Scaling|CH4`.

use crate::constants::{get, surface_temperature, temperature_unit};
use rscm_core::component::{
    Component, InputState, OutputState, TimeseriesType, VariableDefinition,
};
use rscm_core::errors::RSCMResult;
use rscm_core::names::LIFETIME_SCALING_CH4;
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentMetadata;
use serde::{Deserialize, Serialize};

/// A precursor of the CH4 lifetime.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Precursor {
    /// Name of the emissions or concentration input.
    pub variable: String,
    /// Unit of the input.
    pub unit: String,
    /// Value of the input with no effect on the lifetime, in its unit.
    pub baseline: FloatValue,
    /// Fractional lifetime change per unit of the input (FaIR `ch4_lifetime_chemical_sensitivity`).
    pub sensitivity: FloatValue,
}

/// Parameters of [`CH4Lifetime`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CH4LifetimeParameters {
    /// Precursor emissions, summed before the concentrations.
    pub emissions: Vec<Precursor>,
    /// Precursor concentrations.
    pub concentrations: Vec<Precursor>,
    /// Fractional lifetime change per kelvin of surface air temperature change
    /// (FaIR `lifetime_temperature_sensitivity`).
    /// unit: 1/K
    pub temperature_sensitivity: FloatValue,
    /// Floor on each scaling term (FaIR `_ch4_min`).
    #[serde(default = "default_minimum_scaling")]
    pub minimum_scaling: FloatValue,
    /// Name of the surface air temperature change input.
    #[serde(default = "surface_temperature")]
    pub surface_temperature: String,
    /// Name of the lifetime scaling output.
    #[serde(default = "default_output")]
    pub output: String,
}

fn default_minimum_scaling() -> FloatValue {
    0.01
}

fn default_output() -> String {
    LIFETIME_SCALING_CH4.to_string()
}

/// Thornhill et al. (2021) CH4 lifetime scaling.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentMetadata)]
#[component(tags = ["chemistry", "ch4", "fair"], category = "Atmospheric Chemistry")]
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
    ) -> RSCMResult<OutputState> {
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
