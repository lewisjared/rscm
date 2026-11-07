//! Aerosol-cloud interactions (FaIR step 7)
//!
//! $$F = \big(\beta \ln(1 + \textstyle\sum_i s_i x_i) - \beta \ln(1 + \sum_i s_i x_{i,0})\big)\, s$$
//!
//! over emissions and concentrations $x_i$ with shapes $s_i$, scale $\beta$ and forcing
//! scale $s$, plus a feedback on the surface temperature at the start of the step
//! (Smith et al. 2021).

use crate::constants::{driver_definitions, get, one, surface_temperature};
use crate::driver::Driver;
use rscm_core::component::{Component, InputState, OutputState, VariableDefinition};
use rscm_core::errors::{RscmError, RscmResult};
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// One driver of aerosol-cloud interactions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CloudDriver {
    /// Input variable.
    pub variable: String,
    /// Its unit.
    pub unit: String,
    /// Emissions or concentration.
    pub driver: Driver,
    /// Pre-industrial value.
    #[serde(default)]
    pub baseline: FloatValue,
    /// Sensitivity of the cloud response.
    pub shape: FloatValue,
}

/// Parameters of [`AerosolCloud`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AerosolCloudParameters {
    /// Forcing output.
    pub forcing: String,
    /// Drivers. Emissions are summed before concentrations.
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
#[serde(try_from = "Unchecked")]
pub struct AerosolCloud {
    parameters: AerosolCloudParameters,
    #[serde(skip)]
    baseline_effect: OnceLock<FloatValue>,
}

#[derive(Deserialize)]
struct Unchecked {
    parameters: AerosolCloudParameters,
}

impl TryFrom<Unchecked> for AerosolCloud {
    type Error = RscmError;

    fn try_from(unchecked: Unchecked) -> RscmResult<Self> {
        Self::from_parameters(unchecked.parameters)
    }
}

impl AerosolCloud {
    /// Create the component, checking each driver is emissions or a concentration.
    pub fn from_parameters(parameters: AerosolCloudParameters) -> RscmResult<Self> {
        if let Some(d) = parameters
            .drivers
            .iter()
            .find(|d| d.driver == Driver::CumulativeEmissions)
        {
            return Err(RscmError::Error(format!(
                "aerosol-cloud driver {} cannot be cumulative emissions",
                d.variable
            )));
        }
        Ok(Self {
            parameters,
            baseline_effect: OnceLock::new(),
        })
    }

    /// Radiative effect of values in driver order.
    fn effect(&self, values: &[FloatValue]) -> FloatValue {
        let p = &self.parameters;
        let sum = |driver| -> FloatValue {
            p.drivers
                .iter()
                .zip(values)
                .filter(|(d, _)| d.driver == driver)
                .map(|(d, v)| d.shape * v)
                .sum()
        };
        p.scale * (1.0 + sum(Driver::Emissions) + sum(Driver::Concentration)).ln()
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

#[typetag::serde(name = "fair.AerosolCloud")]
impl Component for AerosolCloud {
    fn definitions(&self) -> Vec<VariableDefinition> {
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
    ) -> RscmResult<OutputState> {
        let p = &self.parameters;
        let values: Vec<_> = p
            .drivers
            .iter()
            .map(|d| get(input_state, &d.variable))
            .collect();
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
    use rscm_core::names::{EMISSIONS_SOX, ERF_AEROSOLS_INDIRECT_EFFECT};

    #[test]
    fn baseline_has_no_forcing_and_more_sulfur_cools() {
        let c = AerosolCloud::from_parameters(AerosolCloudParameters {
            forcing: ERF_AEROSOLS_INDIRECT_EFFECT.into(),
            drivers: vec![CloudDriver {
                variable: EMISSIONS_SOX.into(),
                unit: "Mt SO2/yr".into(),
                driver: Driver::Emissions,
                baseline: 2.4,
                shape: 0.0169,
            }],
            scale: -1.03,
            forcing_scale: 1.0,
            tropospheric_adjustment: 0.0,
            temperature_feedback: 0.0,
            surface_temperature: surface_temperature(),
        })
        .unwrap();
        assert_relative_eq!(c.forcing(&[2.4]), 0.0);
        assert!(c.forcing(&[100.0]) < 0.0);
    }

    #[test]
    fn rejects_cumulative_emissions() {
        let parameters = AerosolCloudParameters {
            forcing: ERF_AEROSOLS_INDIRECT_EFFECT.into(),
            drivers: vec![CloudDriver {
                variable: EMISSIONS_SOX.into(),
                unit: "Mt SO2/yr".into(),
                driver: Driver::CumulativeEmissions,
                baseline: 0.0,
                shape: 0.0169,
            }],
            scale: -1.03,
            forcing_scale: 1.0,
            tropospheric_adjustment: 0.0,
            temperature_feedback: 0.0,
            surface_temperature: surface_temperature(),
        };
        assert!(AerosolCloud::from_parameters(parameters).is_err());
    }
}
