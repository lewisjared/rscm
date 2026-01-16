#![allow(dead_code)]

use ode_solvers::*;
use std::collections::HashMap;
use std::sync::Arc;

use rscm_core::component::{
    Component, InputState, OutputState, RequirementDefinition, RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::ivp::{IVPBuilder, IVP};
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};

// Define some types that are used by OdeSolvers
type ModelState = Vector3<FloatValue>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TwoLayerComponentParameters {
    pub lambda0: FloatValue,
    pub a: FloatValue,
    pub efficacy: FloatValue,
    pub eta: FloatValue,
    pub heat_capacity_surface: FloatValue,
    pub heat_capacity_deep: FloatValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoLayerComponent {
    parameters: TwoLayerComponentParameters,
}

// Create the set of ODEs to represent the two layer model
impl IVP<Time, ModelState> for TwoLayerComponent {
    fn calculate_dy_dt(
        &self,
        _t: Time,
        input_state: &InputState,
        y: &ModelState,
        dy_dt: &mut ModelState,
    ) {
        let temperature_surface = y[0];
        let temperature_deep = y[1];
        let erf = input_state.get_latest("Effective Radiative Forcing");

        let temperature_difference = temperature_surface - temperature_deep;

        let lambda_eff = self.parameters.lambda0 - self.parameters.a * temperature_surface;
        let heat_exchange_surface =
            self.parameters.efficacy * self.parameters.eta * temperature_difference;
        let dtemperature_surface_dt =
            (erf - lambda_eff * temperature_surface - heat_exchange_surface)
                / self.parameters.heat_capacity_surface;

        let heat_exchange_deep = self.parameters.eta * temperature_difference;
        let dtemperature_deep_dt = heat_exchange_deep / self.parameters.heat_capacity_deep;

        dy_dt[0] = dtemperature_surface_dt;
        dy_dt[1] = dtemperature_deep_dt;
        dy_dt[2] = self.parameters.heat_capacity_surface * dtemperature_surface_dt
            + self.parameters.heat_capacity_deep * dtemperature_deep_dt;
    }
}

impl TwoLayerComponent {
    pub fn from_parameters(parameters: TwoLayerComponentParameters) -> Self {
        Self { parameters }
    }
}

#[typetag::serde]
impl Component for TwoLayerComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::new(
                "Effective Radiative Forcing",
                "W/m^2",
                RequirementType::Input,
            ),
            RequirementDefinition::new("Surface Temperature", "K", RequirementType::Output),
        ]
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let erf = input_state.get_latest("Effective Radiative Forcing");

        let y0 = ModelState::new(0.0, 0.0, 0.0);

        let solver = IVPBuilder::new(Arc::new(self.to_owned()), input_state, y0);
        println!("Solving {:?} with state: {:?}", self, input_state);

        let mut solver = solver.to_rk4(t_current, t_next, 0.1);
        let stats = solver.integrate().expect("Failed solving");

        let results = solver.results();

        println!("Stats {:?}", stats);
        println!("Results {:?}", results);

        // Create the solver

        Ok(HashMap::from([(
            "Surface Temperature".to_string(),
            StateValue::Scalar(erf * self.parameters.lambda0),
        )]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::array;
    use rscm_core::model::extract_state;
    use rscm_core::timeseries::Timeseries;
    use rscm_core::timeseries_collection::{TimeseriesCollection, VariableType};

    #[test]
    fn it_works() {
        // Solve the two layer component in isolation
        let component = TwoLayerComponent::from_parameters(TwoLayerComponentParameters {
            lambda0: 0.5,
            a: 0.01,
            efficacy: 0.5,
            eta: 0.1,
            heat_capacity_surface: 1.0,
            heat_capacity_deep: 100.0,
        });

        let mut ts_collection = TimeseriesCollection::new();
        ts_collection.add_timeseries(
            "Effective Radiative Forcing".to_string(),
            Timeseries::from_values(
                array![1.0, 1.5, 2.0, 2.0],
                array![1848.0, 1849.0, 1850.0, 1900.0],
            ),
            VariableType::Exogenous,
        );

        let input_state = extract_state(&ts_collection, component.input_names(), 1848.0);

        // Create the solver
        let output_state = component.solve(1848.0, 1849.0, &input_state);

        println!("Output: {:?}", output_state);
        let output_state = output_state.unwrap();
        assert_eq!(
            *output_state.get("Surface Temperature").unwrap(),
            StateValue::Scalar(0.5)
        );
    }
}
