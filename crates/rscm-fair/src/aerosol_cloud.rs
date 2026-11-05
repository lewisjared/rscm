//! Aerosol-cloud interactions (FaIR step 7)
//!
//! $$F = \big(\beta \ln(1 + \textstyle\sum_i s_i x_i) - \beta \ln(1 + \sum_i s_i x_{i,0})\big)\, s$$
//!
//! over emissions and concentrations $x_i$ with shapes $s_i$, scale $\beta$ and forcing
//! scale $s$, plus a feedback on the surface temperature at the start of the step
//! (Smith et al. 2021).

use crate::constants::{driver_definitions, end, get, one, surface_temperature};
use rscm_core::component::{Component, InputState, OutputState, RequirementDefinition};
use rscm_core::errors::RSCMResult;
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// What a driver is, which sets the time it is read at.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CloudDriverKind {
    /// Emissions over the step.
    Emissions,
    /// Concentration at the end of the step.
    Concentration,
}

/// One driver of aerosol-cloud interactions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CloudDriver {
    /// Input variable.
    pub variable: String,
    /// Its unit.
    pub unit: String,
    /// Emissions or concentration.
    pub kind: CloudDriverKind,
    /// Pre-industrial value.
    #[serde(default)]
    pub baseline: FloatValue,
    /// Sensitivity of the cloud response.
    pub shape: FloatValue,
}

/// Parameters of [`FairAerosolCloud`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FairAerosolCloudParameters {
    /// Forcing output.
    pub forcing: String,
    /// Drivers; emissions are summed before concentrations.
    pub drivers: Vec<CloudDriver>,
    /// Scale of the logarithmic response (W m-2).
    pub scale: FloatValue,
    /// Forcing scaling.
    #[serde(default = "one")]
    pub forcing_scale: FloatValue,
    /// Tropospheric adjustment.
    #[serde(default)]
    pub tropospheric_adjustment: FloatValue,
    /// Forcing per kelvin of surface temperature.
    #[serde(default)]
    pub temperature_feedback: FloatValue,
    /// Surface temperature input.
    #[serde(default = "surface_temperature")]
    pub surface_temperature: String,
}

/// FaIR's logarithmic aerosol-cloud forcing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairAerosolCloud {
    parameters: FairAerosolCloudParameters,
    #[serde(skip)]
    baseline_effect: OnceLock<FloatValue>,
}

impl FairAerosolCloud {
    /// Create the component.
    pub fn from_parameters(parameters: FairAerosolCloudParameters) -> Self {
        Self {
            parameters,
            baseline_effect: OnceLock::new(),
        }
    }

    /// Radiative effect of values in driver order.
    fn effect(&self, values: &[FloatValue]) -> FloatValue {
        let p = &self.parameters;
        let sum = |kind| -> FloatValue {
            p.drivers
                .iter()
                .zip(values)
                .filter(|(d, _)| d.kind == kind)
                .map(|(d, v)| d.shape * v)
                .sum()
        };
        p.scale * (1.0 + sum(CloudDriverKind::Emissions) + sum(CloudDriverKind::Concentration)).ln()
    }

    /// Forcing before the temperature feedback, for values in driver order.
    pub fn forcing(&self, values: &[FloatValue]) -> FloatValue {
        let p = &self.parameters;
        let baseline_effect = *self.baseline_effect.get_or_init(|| {
            let baselines: Vec<_> = p.drivers.iter().map(|d| d.baseline).collect();
            self.effect(&baselines)
        });
        (self.effect(values) - baseline_effect)
            * p.forcing_scale
            * (1.0 + p.tropospheric_adjustment)
    }
}

#[typetag::serde]
impl Component for FairAerosolCloud {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        let p = &self.parameters;
        driver_definitions(
            &p.surface_temperature,
            p.drivers
                .iter()
                .map(|d| (d.variable.as_str(), d.unit.as_str())),
            &p.forcing,
        )
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let p = &self.parameters;
        let values = p
            .drivers
            .iter()
            .map(|d| match d.kind {
                CloudDriverKind::Emissions => Ok(get(input_state, &d.variable)),
                CloudDriverKind::Concentration => end(input_state, &d.variable),
            })
            .collect::<RSCMResult<Vec<_>>>()?;
        let temperature = get(input_state, &p.surface_temperature);
        let mut output = OutputState::with_capacity(1);
        output.insert(
            p.forcing.clone(),
            StateValue::Scalar(self.forcing(&values) + temperature * p.temperature_feedback),
        );
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn baseline_has_no_forcing_and_more_sulfur_cools() {
        let c = FairAerosolCloud::from_parameters(FairAerosolCloudParameters {
            forcing: "ERFaci".into(),
            drivers: vec![CloudDriver {
                variable: "Emissions|SOx".into(),
                unit: "Mt SO2/yr".into(),
                kind: CloudDriverKind::Emissions,
                baseline: 2.4,
                shape: 0.0169,
            }],
            scale: -1.03,
            forcing_scale: 1.0,
            tropospheric_adjustment: 0.0,
            temperature_feedback: 0.0,
            surface_temperature: surface_temperature(),
        });
        assert_relative_eq!(c.forcing(&[2.4]), 0.0);
        assert!(c.forcing(&[100.0]) < 0.0);
    }
}
