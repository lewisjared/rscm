use crate::constants::GTC_PER_PPM;
use ode_solvers::Vector3;
use rscm_core::component::{
    Component, InputState, OutputState, RequirementDefinition, RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::ivp::{get_last_step, IVPBuilder, IVP};
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
type ModelState = Vector3<FloatValue>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonCycleParameters {
    /// Timescale of the box's response
    /// unit: yr
    pub tau: FloatValue,
    /// Pre-industrial atmospheric CO_2 concentration
    /// unit: ppm
    pub conc_pi: FloatValue,
    /// Sensitivity of lifetime to changes in global-mean temperature
    /// unit: 1 / K
    pub alpha_temperature: FloatValue,
}

// TODO: Move this into core
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverOptions {
    pub step_size: FloatValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonCycleComponent {
    parameters: CarbonCycleParameters,
    solver_options: SolverOptions,
}

impl CarbonCycleComponent {
    pub fn from_parameters(parameters: CarbonCycleParameters) -> Self {
        Self {
            parameters,
            solver_options: SolverOptions { step_size: 0.1 },
        }
    }

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
            RequirementDefinition::new(
                "Emissions|CO2|Anthropogenic",
                "GtC / yr",
                RequirementType::Input,
            ),
            RequirementDefinition::new("Surface Temperature", "K", RequirementType::Input),
            RequirementDefinition::new(
                "Atmospheric Concentration|CO2",
                "ppm",
                RequirementType::InputAndOutput,
            ),
            RequirementDefinition::new(
                "Cumulative Emissions|CO2",
                "Gt C",
                RequirementType::InputAndOutput,
            ),
            RequirementDefinition::new(
                "Cumulative Land Uptake",
                "Gt C",
                RequirementType::InputAndOutput,
            ),
        ]
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let y0 = ModelState::new(
            input_state.get_latest("Atmospheric Concentration|CO2"),
            input_state.get_latest("Cumulative Land Uptake"),
            input_state.get_latest("Cumulative Emissions|CO2"),
        );

        let solver = IVPBuilder::new(Arc::new(self.to_owned()), &input_state, y0);

        let mut solver = solver.to_rk4(t_current, t_next, self.solver_options.step_size);
        solver.integrate().expect("Failed solving");

        let results = get_last_step(solver.results(), t_next);

        let mut output = HashMap::new();
        output.insert("Atmospheric Concentration|CO2".to_string(), results[0]);
        output.insert("Cumulative Land Uptake".to_string(), results[1]);
        output.insert("Cumulative Emissions|CO2".to_string(), results[2]);

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
        let emissions = input_state.get_latest("Emissions|CO2|Anthropogenic");
        let temperature = input_state.get_latest("Surface Temperature");
        let conc = input_state.get_latest("Atmospheric Concentration|CO2");

        // dC / dt = E - (C - C_0) / (\tau \exp(alpha_temperature * temperature))
        let lifetime =
            self.parameters.tau * (self.parameters.alpha_temperature * temperature).exp();
        let uptake = (conc - self.parameters.conc_pi) / lifetime; // ppm / yr

        dy_dt[0] = emissions / GTC_PER_PPM - uptake; // ppm / yr
        dy_dt[1] = uptake * GTC_PER_PPM; // GtC / yr
        dy_dt[2] = emissions // GtC / yr
    }
}
