//! Effective radiative forcing linear in its drivers (FaIR steps 6 and 9 to 13)
//!
//! Calculates an effective radiative forcing that is linear in a set of emissions,
//! concentrations or cumulative emissions.
//! One instance each gives FaIR's aerosol-radiation interactions, ozone, contrail,
//! light absorbing particles on snow, stratospheric water vapour and land-use change forcing.
//!
//! $$F = \sum_i (x_i - x_{i,0})\, \eta_i\, s + \lambda T$$
//!
//! with drivers $x_i$, baselines $x_{i,0}$, radiative efficiencies $\eta_i$, forcing scale $s$
//! and a feedback $\lambda$ on the surface air temperature change $T$ at the start of the step.
//! The forcing scale is `forcing_scale * (1 + tropospheric_adjustment)`.
//! A cumulative emissions driver reads an emissions rate and accumulates it over the run.
//! With `skip_nan_terms`, a NaN term drops out of the sum, as in FaIR's `np.nansum`.
//! FaIR sums every linear forcing this way apart from ozone.
//!
//! # Inputs
//!
//! - The surface air temperature change named by `surface_temperature` (K).
//!   Its default is `Surface Air Temperature Change`.
//! - Each driver's `variable`, in the driver's `unit`.
//!   rscm.fair passes, for example, `Emissions|SOx` for aerosol-radiation interactions
//!   and `Emissions|CO2|Biosphere` for land-use change.
//!
//! # Outputs
//!
//! - The effective radiative forcing named by `forcing` (W/m^2).
//!   rscm.fair names it after the forcing agent,
//!   such as `Effective Radiative Forcing|Aerosols|Direct Effect`
//!   or `Effective Radiative Forcing|Ozone`.

use crate::constants::{driver_definitions, get, one, surface_temperature};
use crate::driver::Driver;
use rscm_core::component::{
    Component, ComponentState, InputState, OutputState, VariableDefinition,
};
use rscm_core::errors::{RSCMError, RSCMResult};
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentMetadata;
use serde::{Deserialize, Serialize};
use std::any::Any;

/// One driver of the forcing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LinearDriver {
    /// Name of the input. A cumulative emissions driver reads the emissions rate.
    pub variable: String,
    /// Unit of the input.
    pub unit: String,
    /// Whether the input is emissions, a concentration or cumulative emissions.
    pub driver: Driver,
    /// Value of the driver with zero effective radiative forcing.
    #[serde(default)]
    pub baseline: FloatValue,
    /// Effective radiative forcing per unit of the driver.
    pub radiative_efficiency: FloatValue,
}

/// Parameters of [`LinearForcing`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LinearForcingParameters {
    /// Name of the effective radiative forcing output.
    pub forcing: String,
    /// Emissions, concentrations or cumulative emissions the forcing is linear in, summed in order.
    pub drivers: Vec<LinearDriver>,
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
    /// Whether a NaN term drops out of the sum rather than making the forcing NaN.
    #[serde(default)]
    pub skip_nan_terms: bool,
}

/// Running sums of the emission rates of cumulative drivers, in driver order.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinearForcingState {
    /// One sum per driver, zero for other drivers.
    pub emissions_sum: Vec<FloatValue>,
}

#[typetag::serde(name = "fair.LinearForcingState")]
impl ComponentState for LinearForcingState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Effective radiative forcing linear in emissions, concentrations or cumulative emissions.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentMetadata)]
#[component(tags = ["forcing", "aerosol", "ozone", "fair"], category = "Radiative Forcing")]
pub struct LinearForcing {
    parameters: LinearForcingParameters,
}

impl LinearForcing {
    /// Create the component.
    pub fn from_parameters(parameters: LinearForcingParameters) -> Self {
        Self { parameters }
    }

    /// Forcing before the temperature feedback, for driver values in driver order.
    pub fn forcing(&self, values: &[FloatValue]) -> FloatValue {
        let p = &self.parameters;
        let scale = p.forcing_scale * (1.0 + p.tropospheric_adjustment);
        p.drivers
            .iter()
            .zip(values)
            .map(|(driver, value)| (value - driver.baseline) * driver.radiative_efficiency * scale)
            .filter(|term| !(p.skip_nan_terms && term.is_nan()))
            .fold(0.0, |forcing, term| forcing + term)
    }

    fn initial_state(&self) -> LinearForcingState {
        LinearForcingState {
            emissions_sum: vec![0.0; self.parameters.drivers.len()],
        }
    }

    fn solve_impl(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
        state: &mut LinearForcingState,
    ) -> RSCMResult<OutputState> {
        let p = &self.parameters;
        let dt = t_next - t_current;
        let values: Vec<_> = p
            .drivers
            .iter()
            .zip(state.emissions_sum.iter_mut())
            .map(|(driver, sum)| match driver.driver {
                Driver::Emissions | Driver::Concentration => get(input_state, &driver.variable),
                Driver::CumulativeEmissions => {
                    *sum += get(input_state, &driver.variable);
                    *sum * dt
                }
            })
            .collect();
        let forcing = self.forcing(&values);
        let temperature = get(input_state, &p.surface_temperature);
        let mut output = OutputState::with_capacity(1);
        output.insert(
            p.forcing.clone(),
            StateValue::Scalar(forcing + temperature * p.temperature_feedback),
        );
        Ok(output)
    }
}

#[typetag::serde(name = "fair.LinearForcing")]
impl Component for LinearForcing {
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

    /// Stateless solve with no emissions before this step, so only correct for the first step
    /// when a driver is cumulative.
    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        self.solve_impl(t_current, t_next, input_state, &mut self.initial_state())
    }

    fn create_initial_state(&self) -> Box<dyn ComponentState> {
        Box::new(self.initial_state())
    }

    fn solve_with_state(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
        internal_state: &mut dyn ComponentState,
    ) -> RSCMResult<OutputState> {
        let state = internal_state
            .as_any_mut()
            .downcast_mut::<LinearForcingState>()
            .ok_or_else(|| RSCMError::Error("Internal state is not LinearForcingState".into()))?;
        self.solve_impl(t_current, t_next, input_state, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear(skip_nan_terms: bool) -> LinearForcing {
        let driver = |variable: &str, baseline| LinearDriver {
            variable: variable.into(),
            unit: "Mt/yr".into(),
            driver: Driver::Emissions,
            baseline,
            radiative_efficiency: 0.5,
        };
        LinearForcing::from_parameters(LinearForcingParameters {
            forcing: "Effective Radiative Forcing|Aerosols|Direct Effect".into(),
            drivers: vec![
                driver("Emissions|BC", 1.0),
                driver("Emissions|OC", f64::NAN),
            ],
            forcing_scale: 2.0,
            tropospheric_adjustment: 0.0,
            temperature_feedback: 0.0,
            surface_temperature: surface_temperature(),
            skip_nan_terms,
        })
    }

    #[test]
    fn nan_terms_drop_out_when_skipped() {
        assert_eq!(linear(true).forcing(&[3.0, 4.0]), 2.0);
    }

    #[test]
    fn nan_terms_make_the_forcing_nan_by_default() {
        assert!(linear(false).forcing(&[3.0, 4.0]).is_nan());
    }
}
