use ode_solvers::*;
use std::sync::Arc;

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

#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[inputs(
    erf { name = "Effective Radiative Forcing", unit = "W/m^2" },
)]
#[outputs(
    surface_temperature { name = "Surface Temperature", unit = "K" },
)]
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
        let inputs = TwoLayerComponentInputs::from_input_state(input_state);
        let erf = inputs.erf.current();

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
        Self::generated_definitions()
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let y0 = ModelState::new(0.0, 0.0, 0.0);

        let solver = IVPBuilder::new(Arc::new(self.to_owned()), input_state, y0);

        let mut solver = solver.to_rk4(t_current, t_next, 0.1);
        solver.integrate().expect("Failed solving");

        let results = get_last_step(solver.results(), t_next);

        let outputs = TwoLayerComponentOutputs {
            surface_temperature: results[0],
        };

        Ok(outputs.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::array;
    use rscm_core::model::extract_state;
    use rscm_core::state::StateValue;
    use rscm_core::timeseries::Timeseries;
    use rscm_core::timeseries_collection::{TimeseriesCollection, VariableType};

    fn create_component() -> TwoLayerComponent {
        TwoLayerComponent::from_parameters(TwoLayerComponentParameters {
            lambda0: 1.0,               // W/(m^2 K) - climate feedback parameter
            a: 0.0,                     // No nonlinear feedback for simpler testing
            efficacy: 1.0,              // Ocean heat uptake efficacy
            eta: 0.7,                   // W/(m^2 K) - heat exchange coefficient
            heat_capacity_surface: 8.0, // W yr / (m^2 K) - realistic ocean mixed layer
            heat_capacity_deep: 100.0,  // W yr / (m^2 K) - deep ocean
        })
    }

    fn create_input_state_with_erf(
        erf_value: FloatValue,
        t_start: Time,
        t_end: Time,
    ) -> TimeseriesCollection {
        let mut ts_collection = TimeseriesCollection::new();
        ts_collection.add_timeseries(
            "Effective Radiative Forcing".to_string(),
            Timeseries::from_values(array![erf_value, erf_value], array![t_start, t_end]),
            VariableType::Exogenous,
        );
        ts_collection
    }

    #[test]
    fn test_positive_erf_causes_warming() {
        let component = create_component();
        let ts_collection = create_input_state_with_erf(4.0, 2000.0, 2001.0);
        let input_state = extract_state(&ts_collection, component.input_names(), 2000.0);

        let output_state = component.solve(2000.0, 2001.0, &input_state).unwrap();
        let temperature = match output_state.get("Surface Temperature").unwrap() {
            StateValue::Scalar(t) => *t,
            _ => panic!("Expected scalar output"),
        };

        // Positive ERF should cause warming (T > 0)
        assert!(
            temperature > 0.0,
            "Positive ERF should cause warming, got T = {}",
            temperature
        );

        // Temperature should be less than equilibrium value (ERF/lambda0 = 4.0/1.0 = 4.0 K)
        // since we're only integrating for 1 year
        assert!(
            temperature < 4.0,
            "Temperature {} should be below equilibrium (4.0 K)",
            temperature
        );
    }

    #[test]
    fn test_zero_erf_no_warming() {
        let component = create_component();
        let ts_collection = create_input_state_with_erf(0.0, 2000.0, 2001.0);
        let input_state = extract_state(&ts_collection, component.input_names(), 2000.0);

        let output_state = component.solve(2000.0, 2001.0, &input_state).unwrap();
        let temperature = match output_state.get("Surface Temperature").unwrap() {
            StateValue::Scalar(t) => *t,
            _ => panic!("Expected scalar output"),
        };

        // Zero ERF from zero initial state should stay at zero
        assert!(
            temperature.abs() < 1e-10,
            "Zero ERF should cause no warming, got T = {}",
            temperature
        );
    }

    #[test]
    fn test_negative_erf_causes_cooling() {
        let component = create_component();
        let ts_collection = create_input_state_with_erf(-2.0, 2000.0, 2001.0);
        let input_state = extract_state(&ts_collection, component.input_names(), 2000.0);

        let output_state = component.solve(2000.0, 2001.0, &input_state).unwrap();
        let temperature = match output_state.get("Surface Temperature").unwrap() {
            StateValue::Scalar(t) => *t,
            _ => panic!("Expected scalar output"),
        };

        // Negative ERF should cause cooling (T < 0)
        assert!(
            temperature < 0.0,
            "Negative ERF should cause cooling, got T = {}",
            temperature
        );
    }

    #[test]
    fn test_larger_erf_causes_more_warming() {
        let component = create_component();

        // Integrate with ERF = 2.0
        let ts_collection_small = create_input_state_with_erf(2.0, 2000.0, 2001.0);
        let input_state_small =
            extract_state(&ts_collection_small, component.input_names(), 2000.0);
        let output_small = component.solve(2000.0, 2001.0, &input_state_small).unwrap();
        let temp_small = match output_small.get("Surface Temperature").unwrap() {
            StateValue::Scalar(t) => *t,
            _ => panic!("Expected scalar output"),
        };

        // Integrate with ERF = 4.0
        let ts_collection_large = create_input_state_with_erf(4.0, 2000.0, 2001.0);
        let input_state_large =
            extract_state(&ts_collection_large, component.input_names(), 2000.0);
        let output_large = component.solve(2000.0, 2001.0, &input_state_large).unwrap();
        let temp_large = match output_large.get("Surface Temperature").unwrap() {
            StateValue::Scalar(t) => *t,
            _ => panic!("Expected scalar output"),
        };

        // Larger ERF should cause more warming
        assert!(
            temp_large > temp_small,
            "Larger ERF ({}) should cause more warming than smaller ERF ({})",
            temp_large,
            temp_small
        );

        // For linear system (a=0), doubling ERF should approximately double the response
        let ratio = temp_large / temp_small;
        assert!(
            (ratio - 2.0).abs() < 0.1,
            "Doubling ERF should approximately double temperature response, got ratio = {}",
            ratio
        );
    }
}
