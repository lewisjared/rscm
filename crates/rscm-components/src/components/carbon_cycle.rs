//! Carbon cycle component
//!
//! A simple one-box carbon cycle model that tracks atmospheric CO2 concentrations
//! and land uptake based on emissions and temperature.

use crate::constants::GTC_PER_PPM;
use ode_solvers::Vector3;
use rscm_core::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
    TimeseriesWindow,
};
use rscm_core::errors::RSCMResult;
use rscm_core::ivp::{get_last_step, IVPBuilder, IVP};
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

type ModelState = Vector3<FloatValue>;

/// Parameters for the one-box carbon cycle component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonCycleParameters {
    /// Timescale of the box's response
    /// unit: yr
    pub tau: FloatValue,
    /// Pre-industrial atmospheric CO2 concentration
    /// unit: ppm
    pub conc_pi: FloatValue,
    /// Sensitivity of lifetime to changes in global-mean temperature
    /// unit: 1 / K
    pub alpha_temperature: FloatValue,
}

/// Solver options for the ODE integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverOptions {
    pub step_size: FloatValue,
}

/// One-box carbon cycle component
///
/// This component models the carbon cycle using a simple one-box model where:
/// - CO2 emissions increase atmospheric concentrations
/// - Land uptake removes CO2 at a rate that depends on the concentration anomaly
/// - The uptake rate is temperature-dependent
///
/// The governing equations are:
/// $$ \frac{dC}{dt} = E - \frac{C - C_0}{\tau \exp(\alpha_T \cdot T)} $$
///
/// Where:
/// - $C$ is atmospheric CO2 concentration (ppm)
/// - $E$ is emissions (GtC/yr converted to ppm/yr)
/// - $C_0$ is pre-industrial concentration (ppm)
/// - $\tau$ is the baseline lifetime (yr)
/// - $\alpha_T$ is the temperature sensitivity (1/K)
/// - $T$ is the surface temperature anomaly (K)
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["carbon-cycle", "simple", "stable"], category = "Carbon Cycle")]
#[parameters(CarbonCycleParameters)]
#[inputs(
    emissions { name = "Emissions|CO2|Anthropogenic", unit = "GtC / yr" },
    temperature { name = "Surface Temperature", unit = "K" },
)]
#[states(
    concentration { name = "Atmospheric Concentration|CO2", unit = "ppm" },
    cumulative_emissions { name = "Cumulative Emissions|CO2", unit = "Gt C" },
    cumulative_uptake { name = "Cumulative Land Uptake", unit = "Gt C" },
)]
pub struct CarbonCycle {
    parameters: CarbonCycleParameters,
    solver_options: SolverOptions,
}

impl CarbonCycle {
    /// Create a new carbon cycle component from parameters
    pub fn from_parameters(parameters: CarbonCycleParameters) -> Self {
        Self {
            parameters,
            solver_options: SolverOptions { step_size: 0.1 },
        }
    }

    /// Set custom solver options
    pub fn with_solver_options(self, solver_options: SolverOptions) -> Self {
        Self {
            parameters: self.parameters,
            solver_options,
        }
    }
}

#[typetag::serde]
impl Component for CarbonCycle {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let inputs = CarbonCycleInputs::from_input_state(input_state);

        let y0 = ModelState::new(
            inputs.concentration.current(),
            inputs.cumulative_uptake.current(),
            inputs.cumulative_emissions.current(),
        );

        let solver = IVPBuilder::new(Arc::new(self.to_owned()), input_state, y0);

        let mut solver = solver.to_rk4(t_current, t_next, self.solver_options.step_size);
        solver.integrate().expect("Failed solving");

        let results = get_last_step(solver.results(), t_next);

        let outputs = CarbonCycleOutputs {
            concentration: results[0],
            cumulative_uptake: results[1],
            cumulative_emissions: results[2],
        };

        Ok(outputs.into())
    }
}

impl IVP<Time, ModelState> for CarbonCycle {
    fn calculate_dy_dt(
        &self,
        _t: Time,
        input_state: &InputState,
        y: &Vector3<FloatValue>,
        dy_dt: &mut Vector3<FloatValue>,
    ) {
        let inputs = CarbonCycleInputs::from_input_state(input_state);

        // Inputs come from input_state
        let emissions = inputs.emissions.current();
        let temperature = inputs.temperature.current();

        // State variables come from the ODE state vector y
        let conc = y[0];

        // dC / dt = E - (C - C_0) / (tau * exp(alpha_temperature * temperature))
        let lifetime =
            self.parameters.tau * (self.parameters.alpha_temperature * temperature).exp();
        let uptake = (conc - self.parameters.conc_pi) / lifetime; // ppm / yr

        dy_dt[0] = emissions / GTC_PER_PPM - uptake; // ppm / yr
        dy_dt[1] = uptake * GTC_PER_PPM; // GtC / yr
        dy_dt[2] = emissions // GtC / yr
    }
}
