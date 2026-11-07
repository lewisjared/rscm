//! Integration test for multi-component variable source disambiguation.
//!
//! This test verifies that the same variable can have different sources for different components:
//! - For the producer component, a state variable is `OwnState`
//! - For the consumer component, the same variable is `Upstream`

use crate::component::{
    Component, GridType, InputState, OutputState, VariableDefinition, VariableRole,
};
use crate::errors::RscmResult;
use crate::interpolate::strategies::{InterpolationStrategy, PreviousStrategy};
use crate::model::ModelBuilder;
use crate::spatial::{ScalarGrid, ScalarRegion};
use crate::state::{ScalarWindow, StateValue, VariableSource};
use crate::timeseries::{FloatValue, Time, TimeAxis, Timeseries};
use crate::ComponentIO;
use numpy::array;
use numpy::ndarray::{Array, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Producer component: writes and reads its own state variable
// ============================================================================

/// Component that produces a state variable (reads own previous state, writes new state)
#[derive(Debug, Serialize, Deserialize, ComponentIO)]
#[states(
    temperature { name = "Surface Air Temperature Change", unit = "degC" },
)]
struct TemperatureProducer {
    /// Rate of temperature increase per step
    pub warming_rate: FloatValue,
}

#[typetag::serde]
impl Component for TemperatureProducer {
    fn definitions(&self) -> Vec<VariableDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RscmResult<OutputState> {
        let inputs = TemperatureProducerInputs::from_input_state(input_state);

        // Read previous temperature (own state)
        let prev_temp = inputs.temperature.get();

        // Calculate new temperature
        let new_temp = prev_temp + self.warming_rate;

        let outputs = TemperatureProducerOutputs {
            temperature: new_temp,
        };

        Ok(outputs.into())
    }
}

// ============================================================================
// Consumer component: reads the state variable produced by TemperatureProducer
// ============================================================================

/// Component that consumes the temperature produced by TemperatureProducer
#[derive(Debug, Serialize, Deserialize, ComponentIO)]
#[inputs(
    temperature { name = "Surface Air Temperature Change", unit = "degC" },
)]
#[outputs(
    heat_content { name = "Ocean Heat Content", unit = "J" },
)]
struct TemperatureConsumer {
    /// Conversion factor from temperature to heat content
    pub heat_capacity: FloatValue,
}

#[typetag::serde]
impl Component for TemperatureConsumer {
    fn definitions(&self) -> Vec<VariableDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RscmResult<OutputState> {
        let inputs = TemperatureConsumerInputs::from_input_state(input_state);

        // Read temperature from upstream producer using get()
        // The framework knows this is Upstream, so get() reads this step's value
        let temperature = inputs.temperature.get();

        // Calculate heat content
        let heat_content = temperature * self.heat_capacity;

        let outputs = TemperatureConsumerOutputs { heat_content };

        Ok(outputs.into())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_variable_sources_disambiguation() {
    // Create a model with both producer and consumer
    let time_axis = TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0));

    // Initial temperature
    let mut initial_values = HashMap::new();
    initial_values.insert("Surface Air Temperature Change".to_string(), 15.0);

    let model = ModelBuilder::new()
        .with_time_axis(time_axis)
        .with_component(Arc::new(TemperatureProducer { warming_rate: 0.1 }))
        .with_component(Arc::new(TemperatureConsumer {
            heat_capacity: 100.0,
        }))
        .with_initial_values(initial_values)
        .build()
        .unwrap();

    // Verify variable sources
    let sources = model.variable_sources();

    // Producer component sees "Surface Air Temperature Change" as OwnState (reads its own previous state)
    let producer_source = sources.get(&(
        "Surface Air Temperature Change".to_string(),
        "TemperatureProducer".to_string(),
    ));
    assert_eq!(
        producer_source,
        Some(&VariableSource::OwnState),
        "Producer should see Surface Air Temperature Change as OwnState"
    );

    // Consumer component sees "Surface Air Temperature Change" as Upstream (reads producer's output)
    let consumer_source = sources.get(&(
        "Surface Air Temperature Change".to_string(),
        "TemperatureConsumer".to_string(),
    ));
    assert_eq!(
        consumer_source,
        Some(&VariableSource::Upstream),
        "Consumer should see Surface Air Temperature Change as Upstream"
    );
}

#[test]
fn test_variable_sources_with_get_method() {
    // Create a model with both producer and consumer
    let time_axis = TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0));

    // Initial temperature
    let mut initial_values = HashMap::new();
    initial_values.insert("Surface Air Temperature Change".to_string(), 15.0);

    let mut model = ModelBuilder::new()
        .with_time_axis(time_axis)
        .with_component(Arc::new(TemperatureProducer { warming_rate: 0.1 }))
        .with_component(Arc::new(TemperatureConsumer {
            heat_capacity: 100.0,
        }))
        .with_initial_values(initial_values)
        .build()
        .unwrap();

    // Run the model
    model.step(); // 2020 -> 2021
    model.step(); // 2021 -> 2022

    // Verify that both components ran correctly
    let timeseries = model.timeseries();

    // Check producer output (state variable)
    let temp_data = timeseries
        .get_data("Surface Air Temperature Change")
        .and_then(|data| data.as_scalar())
        .expect("Surface Air Temperature Change should exist");

    // Initial value at index 0
    let temp_0 = temp_data.at(0, ScalarRegion::Global).unwrap();
    assert_eq!(temp_0, 15.0, "Initial temperature should be 15.0");

    // After first step (index 1): 15.0 + 0.1 = 15.1
    let temp_1 = temp_data.at(1, ScalarRegion::Global).unwrap();
    assert_eq!(temp_1, 15.1, "Temperature after step 1 should be 15.1");

    // After second step (index 2): 15.1 + 0.1 = 15.2
    let temp_2 = temp_data.at(2, ScalarRegion::Global).unwrap();
    assert_eq!(temp_2, 15.2, "Temperature after step 2 should be 15.2");

    // Check consumer output (should use upstream temperature)
    let heat_data = timeseries
        .get_data("Ocean Heat Content")
        .and_then(|data| data.as_scalar())
        .expect("Ocean Heat Content should exist");

    // First value should be NaN (no previous state at index 0)
    let heat_0 = heat_data.at(0, ScalarRegion::Global).unwrap();
    assert!(heat_0.is_nan(), "Heat content at index 0 should be NaN");

    // After first step (index 1): 15.1 * 100.0 = 1510.0
    let heat_1 = heat_data.at(1, ScalarRegion::Global).unwrap();
    assert_eq!(heat_1, 1510.0, "Heat content after step 1 should be 1510.0");

    // After second step (index 2): 15.2 * 100.0 = 1520.0
    let heat_2 = heat_data.at(2, ScalarRegion::Global).unwrap();
    assert_eq!(heat_2, 1520.0, "Heat content after step 2 should be 1520.0");
}

#[test]
fn test_variable_sources_with_prescribed_input() {
    // Create a model with a prescribed input and a component that reads it
    let time_axis = TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0));

    // Create prescribed forcing timeseries
    let forcing_values = array![1.0, 2.0].insert_axis(Axis(1));
    let forcing_ts = Timeseries::new(
        forcing_values,
        Arc::new(TimeAxis::from_bounds(array![2020.0, 2022.5, 2025.0])),
        ScalarGrid,
        "W/m^2".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    );

    /// Component that reads prescribed forcing
    #[derive(Debug, Serialize, Deserialize, ComponentIO)]
    #[inputs(
        forcing { name = "Effective Radiative Forcing", unit = "W/m^2" },
    )]
    #[outputs(
        response { name = "Climate Response", unit = "K" },
    )]
    struct ForcingConsumer {
        sensitivity: FloatValue,
    }

    #[typetag::serde]
    impl Component for ForcingConsumer {
        fn definitions(&self) -> Vec<VariableDefinition> {
            Self::generated_definitions()
        }

        fn solve(
            &self,
            _t_current: Time,
            _t_next: Time,
            input_state: &InputState,
        ) -> RscmResult<OutputState> {
            let inputs = ForcingConsumerInputs::from_input_state(input_state);

            // Read prescribed forcing
            let forcing = inputs.forcing.get();

            let outputs = ForcingConsumerOutputs {
                response: forcing * self.sensitivity,
            };

            Ok(outputs.into())
        }
    }

    let model = ModelBuilder::new()
        .with_time_axis(time_axis)
        .with_component(Arc::new(ForcingConsumer { sensitivity: 2.0 }))
        .with_prescribed_variable("Effective Radiative Forcing", forcing_ts)
        .build()
        .unwrap();

    // Verify variable source for prescribed input
    let sources = model.variable_sources();

    let consumer_source = sources.get(&(
        "Effective Radiative Forcing".to_string(),
        "ForcingConsumer".to_string(),
    ));
    assert_eq!(
        consumer_source,
        Some(&VariableSource::Prescribed),
        "Consumer should see Effective Radiative Forcing as Prescribed"
    );
}

#[test]
fn test_multiple_consumers_same_upstream() {
    // Create a model with one producer and multiple consumers
    let time_axis = TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0));

    let mut initial_values = HashMap::new();
    initial_values.insert("Surface Air Temperature Change".to_string(), 15.0);

    /// Another consumer component
    #[derive(Debug, Serialize, Deserialize, ComponentIO)]
    #[inputs(
        temperature { name = "Surface Air Temperature Change", unit = "degC" },
    )]
    #[outputs(
        albedo { name = "Surface Albedo", unit = "1" },
    )]
    struct AlbedoCalculator {
        temp_coefficient: FloatValue,
    }

    #[typetag::serde]
    impl Component for AlbedoCalculator {
        fn definitions(&self) -> Vec<VariableDefinition> {
            Self::generated_definitions()
        }

        fn solve(
            &self,
            _t_current: Time,
            _t_next: Time,
            input_state: &InputState,
        ) -> RscmResult<OutputState> {
            let inputs = AlbedoCalculatorInputs::from_input_state(input_state);

            // Use get() to automatically resolve upstream temperature
            let temperature = inputs.temperature.get();

            let outputs = AlbedoCalculatorOutputs {
                albedo: 0.3 - (temperature * self.temp_coefficient),
            };

            Ok(outputs.into())
        }
    }

    let model = ModelBuilder::new()
        .with_time_axis(time_axis)
        .with_component(Arc::new(TemperatureProducer { warming_rate: 0.1 }))
        .with_component(Arc::new(TemperatureConsumer {
            heat_capacity: 100.0,
        }))
        .with_component(Arc::new(AlbedoCalculator {
            temp_coefficient: 0.01,
        }))
        .with_initial_values(initial_values)
        .build()
        .unwrap();

    // Verify variable sources
    let sources = model.variable_sources();

    // Producer sees it as OwnState
    let producer_source = sources.get(&(
        "Surface Air Temperature Change".to_string(),
        "TemperatureProducer".to_string(),
    ));
    assert_eq!(
        producer_source,
        Some(&VariableSource::OwnState),
        "Producer should see Surface Air Temperature Change as OwnState"
    );

    // First consumer sees it as Upstream
    let consumer1_source = sources.get(&(
        "Surface Air Temperature Change".to_string(),
        "TemperatureConsumer".to_string(),
    ));
    assert_eq!(
        consumer1_source,
        Some(&VariableSource::Upstream),
        "TemperatureConsumer should see Surface Air Temperature Change as Upstream"
    );

    // Second consumer also sees it as Upstream
    let consumer2_source = sources.get(&(
        "Surface Air Temperature Change".to_string(),
        "AlbedoCalculator".to_string(),
    ));
    assert_eq!(
        consumer2_source,
        Some(&VariableSource::Upstream),
        "AlbedoCalculator should see Surface Air Temperature Change as Upstream"
    );
}

#[test]
fn test_scalar_state_initialised_from_grid_values() {
    let time_axis = TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0));
    let initial = HashMap::from([("Surface Air Temperature Change".to_string(), vec![15.0])]);

    let model = ModelBuilder::new()
        .with_time_axis(time_axis)
        .with_component(Arc::new(TemperatureProducer { warming_rate: 0.1 }))
        .with_initial_grid_values(initial)
        .build()
        .unwrap();

    let temperature = model
        .timeseries()
        .get_data("Surface Air Temperature Change")
        .and_then(|data| data.as_scalar())
        .expect("Surface Air Temperature Change should exist");
    assert_eq!(temperature.at(0, ScalarRegion::Global).unwrap(), 15.0);
}
