//! One-box carbon cycle component
//!
//! A single box tracks the atmospheric CO2 concentration,
//! which CO2 emissions raise and land uptake lowers.
//! Land uptake removes the concentration excess over pre-industrial at a temperature-dependent rate.

use crate::constants::GTC_PER_PPM;
use ode_solvers::Vector3;
use rscm_core::component::{
    Component, GridType, InputState, OutputState, ScalarWindow, VariableDefinition, VariableRole,
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
#[serde(deny_unknown_fields)]
pub struct CarbonCycleParameters {
    /// Response timescale of the box
    /// unit: yr
    pub tau: FloatValue,
    /// Pre-industrial atmospheric CO2 concentration
    /// unit: ppm
    pub co2_pi: FloatValue,
    /// Exponential scaling of the response timescale with surface air temperature change
    /// unit: 1/K
    pub alpha_temperature: FloatValue,
}

/// Solver options for the ODE integration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverOptions {
    pub step_size: FloatValue,
}

/// One-box carbon cycle component
///
/// The concentration follows
///
/// $$ \frac{dC}{dt} = E - \frac{C - C_0}{\tau \exp(\alpha_T \cdot T)} $$
///
/// where:
///
/// - $C$ is the atmospheric CO2 concentration (ppm)
/// - $E$ is the CO2 emissions (GtC/yr converted to ppm/yr)
/// - $C_0$ is the pre-industrial concentration (ppm)
/// - $\tau$ is the response timescale (yr)
/// - $\alpha_T$ scales the response timescale with warming (1/K)
/// - $T$ is the surface air temperature change (K)
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["carbon-cycle", "simple", "stable"], category = "Carbon Cycle")]
#[inputs(
    emissions { name = "Emissions|CO2", unit = "GtC/yr", description = "CO2 emissions, added to the atmosphere" },
    temperature { name = "Surface Air Temperature Change", unit = "K", description = "Surface air temperature change, which scales the response timescale" },
)]
#[states(
    concentration { name = "Atmospheric Concentrations|CO2", unit = "ppm", description = "Atmospheric CO2 concentration at the end of the step" },
    cumulative_emissions { name = "Cumulative Emissions|CO2", unit = "GtC", description = "CO2 emitted up to the end of the step" },
    cumulative_uptake { name = "Cumulative Carbon Uptake|Land", unit = "GtC", description = "Carbon taken up by land up to the end of the step" },
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
    fn definitions(&self) -> Vec<VariableDefinition> {
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
            inputs.concentration.get(),
            inputs.cumulative_uptake.get(),
            inputs.cumulative_emissions.get(),
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

        // Inputs come from input_state (could be prescribed or upstream)
        let emissions = inputs.emissions.get();
        let temperature = inputs.temperature.get();

        // State variables come from the ODE state vector y
        let conc = y[0];

        // dC / dt = E - (C - C_0) / (tau * exp(alpha_temperature * temperature))
        let timescale =
            self.parameters.tau * (self.parameters.alpha_temperature * temperature).exp();
        let uptake = (conc - self.parameters.co2_pi) / timescale; // ppm/yr

        dy_dt[0] = emissions / GTC_PER_PPM - uptake; // ppm/yr
        dy_dt[1] = uptake * GTC_PER_PPM; // GtC/yr
        dy_dt[2] = emissions // GtC/yr
    }
}
