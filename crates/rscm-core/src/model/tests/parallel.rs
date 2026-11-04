//! Parallel build and run tests.

use super::aggregate_execution::FailingForcing;
use super::basic::get_emissions;
use crate::example_components::{TestComponent, TestComponentParameters};
use crate::model::{build_models, run_models, Model, ModelBuilder};
use crate::schema::VariableSchema;
use crate::timeseries::{FloatValue, TimeAxis};
use numpy::array;
use numpy::ndarray::Array;
use std::sync::Arc;

fn builder(conversion_factor: FloatValue) -> ModelBuilder {
    let mut builder = ModelBuilder::new();
    builder
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2030.0, 1.0)))
        .with_component(Arc::new(TestComponent::from_parameters(
            TestComponentParameters { conversion_factor },
        )))
        .with_exogenous_variable("Emissions|CO2", get_emissions());
    builder
}

fn concentration_bits(model: &Model) -> Vec<u64> {
    model
        .timeseries()
        .get_data("Concentrations|CO2")
        .and_then(|data| data.as_scalar())
        .unwrap()
        .values()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

#[test]
fn parallel_runs_match_serial_runs() {
    let builders: Vec<_> = (1..=16).map(|i| builder(0.1 * i as FloatValue)).collect();
    let mut models: Vec<Model> = build_models(&builders)
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert!(run_models(&mut models).iter().all(Result::is_ok));

    for (builder, model) in builders.iter().zip(&models) {
        let mut serial = builder.build().unwrap();
        serial.run();
        assert!(model.finished());
        assert_eq!(concentration_bits(model), concentration_bits(&serial));
    }
}

#[test]
fn a_failed_run_leaves_the_others_finished() {
    let mut failing = ModelBuilder::new();
    failing
        .with_time_axis(TimeAxis::from_values(array![2020.0, 2021.0]))
        .with_component(Arc::new(FailingForcing));
    let mut models = [
        builder(0.5).build().unwrap(),
        failing.build().unwrap(),
        builder(0.5).build().unwrap(),
    ];
    let mut refs: Vec<&mut Model> = models.iter_mut().collect();

    let results = run_models(&mut refs);

    assert!(results[0].is_ok() && results[2].is_ok());
    assert!(results[1]
        .as_ref()
        .unwrap_err()
        .to_string()
        .contains("forcing unavailable"));
    assert!(models[0].finished() && models[2].finished());
}

#[test]
fn build_errors_keep_their_position() {
    let mut wrong_units = builder(0.5);
    wrong_units.with_schema(
        VariableSchema::new()
            .variable("Emissions|CO2", "GtCO2")
            .variable("Concentrations|CO2", "GtC"),
    );
    let builders = [builder(0.5), wrong_units];

    let results = build_models(&builders);

    assert!(results[0].is_ok());
    assert!(results[1].is_err());
}
