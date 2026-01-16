//! Carbon cycle component
//!
//! A simple one-box carbon cycle model that tracks atmospheric CO2 concentrations
//! and land uptake based on emissions and temperature.

use crate::constants::GTC_PER_PPM;
use ode_solvers::Vector3;
use rscm_core::component::{Component, InputState, OutputState, RequirementDefinition};
use rscm_core::errors::RSCMResult;
use rscm_core::ivp::{get_last_step, IVPBuilder, IVP};
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

type ModelState = Vector3<FloatValue>;

// Variable name constants
const VAR_EMISSIONS_CO2: &str = "Emissions|CO2|Anthropogenic";
const VAR_SURFACE_TEMP: &str = "Surface Temperature";
const VAR_CONC_CO2: &str = "Atmospheric Concentration|CO2";
const VAR_CUM_EMISSIONS: &str = "Cumulative Emissions|CO2";
const VAR_CUM_UPTAKE: &str = "Cumulative Land Uptake";

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonCycleComponent {
    parameters: CarbonCycleParameters,
    solver_options: SolverOptions,
}

impl CarbonCycleComponent {
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
impl Component for CarbonCycleComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::scalar_input(VAR_EMISSIONS_CO2, "GtC / yr"),
            RequirementDefinition::scalar_input(VAR_SURFACE_TEMP, "K"),
            RequirementDefinition::scalar_state(VAR_CONC_CO2, "ppm"),
            RequirementDefinition::scalar_state(VAR_CUM_EMISSIONS, "Gt C"),
            RequirementDefinition::scalar_state(VAR_CUM_UPTAKE, "Gt C"),
        ]
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let y0 = ModelState::new(
            input_state.get_latest(VAR_CONC_CO2),
            input_state.get_latest(VAR_CUM_UPTAKE),
            input_state.get_latest(VAR_CUM_EMISSIONS),
        );

        let solver = IVPBuilder::new(Arc::new(self.to_owned()), input_state, y0);

        let mut solver = solver.to_rk4(t_current, t_next, self.solver_options.step_size);
        solver.integrate().expect("Failed solving");

        let results = get_last_step(solver.results(), t_next);

        let mut output = HashMap::new();
        output.insert(VAR_CONC_CO2.to_string(), StateValue::Scalar(results[0]));
        output.insert(VAR_CUM_UPTAKE.to_string(), StateValue::Scalar(results[1]));
        output.insert(
            VAR_CUM_EMISSIONS.to_string(),
            StateValue::Scalar(results[2]),
        );

        Ok(output)
    }
}

impl IVP<Time, ModelState> for CarbonCycleComponent {
    fn calculate_dy_dt(
        &self,
        _t: Time,
        input_state: &InputState,
        _y: &Vector3<FloatValue>,
        dy_dt: &mut Vector3<FloatValue>,
    ) {
        let emissions = input_state.get_latest(VAR_EMISSIONS_CO2);
        let temperature = input_state.get_latest(VAR_SURFACE_TEMP);
        let conc = input_state.get_latest(VAR_CONC_CO2);

        // dC / dt = E - (C - C_0) / (tau * exp(alpha_temperature * temperature))
        let lifetime =
            self.parameters.tau * (self.parameters.alpha_temperature * temperature).exp();
        let uptake = (conc - self.parameters.conc_pi) / lifetime; // ppm / yr

        dy_dt[0] = emissions / GTC_PER_PPM - uptake; // ppm / yr
        dy_dt[1] = uptake * GTC_PER_PPM; // GtC / yr
        dy_dt[2] = emissions // GtC / yr
    }
}
