//! Forcing linear in its drivers (FaIR steps 6 and 9 to 13)
//!
//! $$F = \sum_i (x_i - x_{i,0})\, \eta_i\, s + \lambda T$$
//!
//! with radiative efficiencies $\eta_i$, forcing scale $s$ and a feedback $\lambda$ on the
//! surface temperature at the start of the step.
//! One instance each gives FaIR's aerosol-radiation, ozone, contrail, light absorbing
//! particle, stratospheric water vapour and land use forcing.

use crate::constants::{driver_definitions, end, get, one, surface_temperature};
use rscm_core::component::{
    Component, ComponentState, InputState, OutputState, RequirementDefinition,
};
use rscm_core::errors::{RSCMError, RSCMResult};
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};
use std::any::Any;

/// What a driver is, which sets the time it is read at.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DriverKind {
    /// Emissions over the step.
    Emissions,
    /// Concentration at the end of the step.
    Concentration,
    /// Emissions summed to the end of the step.
    CumulativeEmissions,
}

/// One driver of the forcing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Driver {
    /// Input variable. Cumulative drivers read the emissions rate.
    pub variable: String,
    /// Its unit.
    pub unit: String,
    /// What the driver is.
    pub kind: DriverKind,
    /// Value with zero forcing.
    #[serde(default)]
    pub baseline: FloatValue,
    /// Forcing per unit of the driver.
    pub radiative_efficiency: FloatValue,
}

/// Parameters of [`FairLinearForcing`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FairLinearForcingParameters {
    /// Forcing output.
    pub forcing: String,
    /// Drivers, summed in order.
    pub drivers: Vec<Driver>,
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

/// Running sums of the emission rates of cumulative drivers, in driver order.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FairLinearForcingState {
    /// One sum per driver, zero for other kinds.
    pub emissions_sum: Vec<FloatValue>,
}

#[typetag::serde]
impl ComponentState for FairLinearForcingState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Forcing linear in emissions, concentrations or cumulative emissions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairLinearForcing {
    parameters: FairLinearForcingParameters,
}

impl FairLinearForcing {
    /// Create the component.
    pub fn from_parameters(parameters: FairLinearForcingParameters) -> Self {
        Self { parameters }
    }

    fn initial_state(&self) -> FairLinearForcingState {
        FairLinearForcingState {
            emissions_sum: vec![0.0; self.parameters.drivers.len()],
        }
    }

    fn solve_impl(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
        state: &mut FairLinearForcingState,
    ) -> RSCMResult<OutputState> {
        let p = &self.parameters;
        let dt = t_next - t_current;
        let scale = p.forcing_scale * (1.0 + p.tropospheric_adjustment);
        let mut forcing = 0.0;
        for (driver, sum) in p.drivers.iter().zip(state.emissions_sum.iter_mut()) {
            let value = match driver.kind {
                DriverKind::Emissions => get(input_state, &driver.variable),
                DriverKind::Concentration => end(input_state, &driver.variable)?,
                DriverKind::CumulativeEmissions => {
                    *sum += get(input_state, &driver.variable);
                    *sum * dt
                }
            };
            forcing += (value - driver.baseline) * driver.radiative_efficiency * scale;
        }
        let temperature = get(input_state, &p.surface_temperature);
        let mut output = OutputState::with_capacity(1);
        output.insert(
            p.forcing.clone(),
            StateValue::Scalar(forcing + temperature * p.temperature_feedback),
        );
        Ok(output)
    }
}

#[typetag::serde]
impl Component for FairLinearForcing {
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
            .downcast_mut::<FairLinearForcingState>()
            .ok_or_else(|| {
                RSCMError::Error("Internal state is not FairLinearForcingState".into())
            })?;
        self.solve_impl(t_current, t_next, input_state, state)
    }
}
