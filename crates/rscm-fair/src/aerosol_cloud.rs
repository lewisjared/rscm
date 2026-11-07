//! Aerosol-cloud interactions (FaIR step 7)
//!
//! Calculates the effective radiative forcing from aerosol-cloud interactions,
//! which grows with the logarithm of a weighted sum of aerosol emissions and concentrations.
//! The method follows Smith et al. (2021).
//!
//! $$F = \big(\beta \ln(1 + \textstyle\sum_i s_i x_i) - \beta \ln(1 + \sum_i s_i x_{i,0})\big)\, s$$
//!
//! over emissions and concentrations $x_i$ with shapes $s_i$, pre-industrial values $x_{i,0}$,
//! scale $\beta$ and forcing scale $s$.
//! The forcing scale is `forcing_scale * (1 + tropospheric_adjustment)`.
//! It adds a feedback of `temperature_feedback` times the surface air temperature change
//! at the start of the step.
//! Each driver is an emissions rate or a concentration, never cumulative emissions.
//!
//! # Inputs
//!
//! - The surface air temperature change named by `surface_temperature` (K).
//!   Its default is `Surface Air Temperature Change`.
//! - Each driver's `variable`, in the driver's `unit`.
//!   rscm.fair passes the emissions or concentration of each species with an `aci_shape`,
//!   such as `Emissions|SOx`.
//!
//! # Outputs
//!
//! - The effective radiative forcing named by `forcing` (W/m^2).
//!   rscm.fair names it `Effective Radiative Forcing|Aerosols|Indirect Effect`.

use crate::constants::{driver_definitions, get, one, surface_temperature};
use crate::driver::Driver;
use rscm_core::component::{Component, InputState, OutputState, VariableDefinition};
use rscm_core::errors::{RSCMError, RSCMResult};
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentMetadata;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// One driver of aerosol-cloud interactions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CloudDriver {
    /// Name of the emissions or concentration input.
    pub variable: String,
    /// Unit of the input.
    pub unit: String,
    /// Whether the input is emissions or a concentration.
    pub driver: Driver,
    /// Pre-industrial value of the input, in its unit
    /// (FaIR `baseline_emissions` or `baseline_concentration`).
    #[serde(default)]
    pub baseline: FloatValue,
    /// Weight of the input in the logarithmic response, per unit of the input (FaIR `aci_shape`).
    pub shape: FloatValue,
}

/// Parameters of [`AerosolCloud`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AerosolCloudParameters {
    /// Name of the effective radiative forcing output.
    pub forcing: String,
    /// Emissions and concentrations that drive the response.
    /// Emissions are summed before concentrations.
    pub drivers: Vec<CloudDriver>,
    /// Scale of the logarithmic response (FaIR `aci_scale`).
    /// unit: W/m^2
    pub scale: FloatValue,
    /// Multiplier on the effective radiative forcing (FaIR `forcing_scale`).
    #[serde(default = "one")]
    pub forcing_scale: FloatValue,
    /// Fractional tropospheric adjustment, applied as `1 + tropospheric_adjustment`
    /// (FaIR `tropospheric_adjustment`).
    #[serde(default)]
    pub tropospheric_adjustment: FloatValue,
    /// Effective radiative forcing per kelvin of surface air temperature change
    /// (FaIR `forcing_temperature_feedback`).
    /// unit: W/m^2/K
    #[serde(default)]
    pub temperature_feedback: FloatValue,
    /// Name of the surface air temperature change input.
    #[serde(default = "surface_temperature")]
    pub surface_temperature: String,
}

/// FaIR's logarithmic aerosol-cloud effective radiative forcing.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentMetadata)]
#[component(tags = ["forcing", "aerosol", "cloud", "fair"], category = "Radiative Forcing")]
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
    type Error = RSCMError;

    fn try_from(unchecked: Unchecked) -> RSCMResult<Self> {
        Self::from_parameters(unchecked.parameters)
    }
}

impl AerosolCloud {
    /// Create the component, checking each driver is emissions or a concentration.
    pub fn from_parameters(parameters: AerosolCloudParameters) -> RSCMResult<Self> {
        if let Some(d) = parameters
            .drivers
            .iter()
            .find(|d| d.driver == Driver::CumulativeEmissions)
        {
            return Err(RSCMError::Error(format!(
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
    ) -> RSCMResult<OutputState> {
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
