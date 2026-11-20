//! Parallel build and run tests.

use super::aggregate_execution::FailingForcing;
use super::basic::get_emissions;
use crate::errors::RSCMError;
use crate::example_components::{TestComponent, TestComponentParameters};
use crate::model::{build_models, run_each, run_models, Model, ModelBuilder, Pool};
use crate::schema::VariableSchema;
use crate::timeseries::{FloatValue, TimeAxis};
use numpy::array;
use numpy::ndarray::Array;
use std::collections::HashSet;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn builder(conversion_factor: FloatValue) -> ModelBuilder {
    let mut builder = ModelBuilder::new();
    builder
        .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2030.0, 1.0)))
        .with_component(Arc::new(TestComponent::from_parameters(
            TestComponentParameters { conversion_factor },
        )))
        .with_prescribed_variable("Emissions|CO2", get_emissions());
    builder
}

fn concentration_bits(model: &Model) -> Vec<u64> {
    model
        .timeseries()
        .get_data("Atmospheric Concentrations|CO2")
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
            .variable("Emissions|CO2", "GtC / yr")
            .variable("Atmospheric Concentrations|CO2", "GtC"),
    );
    let builders = [builder(0.5), wrong_units];

    let results = build_models(&builders);

    assert!(results[0].is_ok());
    assert!(results[1].is_err());
}

#[test]
fn run_each_matches_serial_runs_in_input_order() {
    let pool = Pool::new(Some(4)).unwrap();
    let factors: Vec<FloatValue> = (1..=16).map(|i| 0.1 * i as FloatValue).collect();
    // Slow producer, so late items are still being made while early ones run.
    let source = factors.iter().map(|&f| {
        thread::sleep(Duration::from_millis(2));
        Ok(builder(f))
    });

    let results = run_each(&pool, source, |index, model| {
        (index, model.map(|m| concentration_bits(&m)))
    });

    assert_eq!(results.len(), factors.len());
    for (i, (&factor, (index, bits))) in factors.iter().zip(results).enumerate() {
        let mut serial = builder(factor).build().unwrap();
        serial.run();
        assert_eq!(index, i);
        assert_eq!(bits.unwrap(), concentration_bits(&serial));
    }
}

#[test]
fn run_each_isolates_failures() {
    let mut failing = ModelBuilder::new();
    failing
        .with_time_axis(TimeAxis::from_values(array![2020.0, 2021.0]))
        .with_component(Arc::new(FailingForcing));
    let source = vec![
        Ok(builder(0.5)),
        Err(RSCMError::Error("no builder".into())),
        Ok(failing),
        Ok(ModelBuilder::new()),
        Ok(builder(0.5)),
    ];

    let results = run_each(&Pool::default(), source, |_, model| {
        model.map(|m| m.finished())
    });

    assert!(results[0].as_ref().is_ok_and(|&finished| finished));
    assert!(results[1]
        .as_ref()
        .unwrap_err()
        .to_string()
        .contains("no builder"));
    assert!(results[2]
        .as_ref()
        .unwrap_err()
        .to_string()
        .contains("forcing unavailable"));
    assert!(matches!(results[3], Err(RSCMError::MissingTimeAxis)));
    assert!(results[4].as_ref().is_ok_and(|&finished| finished));
}

#[test]
fn run_each_with_an_empty_source_returns_nothing() {
    let results = run_each(&Pool::default(), std::iter::empty(), |_, model| {
        model.is_ok()
    });
    assert!(results.is_empty());
}

#[test]
fn a_pool_is_reused_across_calls() {
    let pool = Pool::new(Some(3)).unwrap();
    let thread_ids = |pool: &Pool| {
        pool.map_each(0..64, |_, _| {
            thread::sleep(Duration::from_millis(1));
            thread::current().id()
        })
    };
    // Thread ids are never reused, so a rebuilt pool would add new ones.
    let ids: HashSet<_> = thread_ids(&pool)
        .into_iter()
        .chain(thread_ids(&pool.clone()))
        .collect();

    assert!(ids.len() <= 3);
    assert_eq!(pool.n_threads(), 3);
    assert_eq!(pool.install(rayon::current_num_threads), 3);
}

#[test]
fn a_pool_needs_at_least_one_thread() {
    assert!(Pool::new(Some(0))
        .unwrap_err()
        .to_string()
        .contains("at least 1"));
}
