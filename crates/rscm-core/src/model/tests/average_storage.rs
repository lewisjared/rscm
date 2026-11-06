//! Averages are stored at their step's start bound, whoever produced them.

use crate::component::{Component, InputState, OutputState, RequirementDefinition};
use crate::errors::{RSCMError, RSCMResult};
use crate::interpolate::strategies::{InterpolationStrategy, PreviousStrategy};
use crate::model::{Model, ModelBuilder};
use crate::spatial::ScalarGrid;
use crate::state::StateValue;
use crate::timeseries::{FloatValue, Time, TimeAxis, Timeseries};
use numpy::ndarray::{Array, Array1, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Emits a flux that depends on the step's start time.
#[derive(Debug, Serialize, Deserialize)]
struct FluxProducer;

#[typetag::serde]
impl Component for FluxProducer {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![RequirementDefinition::scalar_output("Flux", "GtC / yr")]
    }

    fn solve(&self, t: Time, _t_next: Time, _inputs: &InputState) -> RSCMResult<OutputState> {
        Ok(HashMap::from([(
            "Flux".to_string(),
            StateValue::Scalar(1.0 + (t - 2000.0).powi(2)),
        )]))
    }
}

/// Integrates the flux into a pool and records the flux it read and the one before.
#[derive(Debug, Serialize, Deserialize)]
struct PoolIntegrator;

#[typetag::serde]
impl Component for PoolIntegrator {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![
            RequirementDefinition::scalar_input("Flux", "GtC / yr"),
            RequirementDefinition::scalar_state("Pool", "GtC"),
            RequirementDefinition::scalar_output("Flux Read", "GtC / yr"),
            RequirementDefinition::scalar_output("Previous Flux", "GtC / yr"),
        ]
    }

    fn solve(&self, t: Time, t_next: Time, inputs: &InputState) -> RSCMResult<OutputState> {
        let flux = inputs.get_scalar_window("Flux");
        let pool = inputs.get_scalar_window("Pool").get();
        Ok(HashMap::from([
            (
                "Pool".to_string(),
                StateValue::Scalar(pool + flux.get() * (t_next - t)),
            ),
            ("Flux Read".to_string(), StateValue::Scalar(flux.get())),
            (
                "Previous Flux".to_string(),
                StateValue::Scalar(flux.history(1).unwrap_or(-1.0)),
            ),
        ]))
    }
}

/// Holds an average as its own state, adding one each step.
#[derive(Debug, Serialize, Deserialize)]
struct AverageCounter;

#[typetag::serde]
impl Component for AverageCounter {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![RequirementDefinition::scalar_state("Flux", "GtC / yr")]
    }

    fn solve(&self, _t: Time, _t_next: Time, inputs: &InputState) -> RSCMResult<OutputState> {
        let previous = inputs.get_scalar_window("Flux").get();
        Ok(HashMap::from([(
            "Flux".to_string(),
            StateValue::Scalar(previous + 1.0),
        )]))
    }
}

fn axis() -> TimeAxis {
    TimeAxis::from_values(Array::range(2000.0, 2006.0, 1.0))
}

fn builder(components: Vec<Arc<dyn Component + Send + Sync>>) -> ModelBuilder {
    let mut builder = ModelBuilder::new();
    builder
        .with_time_axis(axis())
        .with_initial_values(HashMap::from([("Pool".to_string(), 0.0)]));
    for component in components {
        builder.with_component(component);
    }
    builder
}

fn values(model: &Model, variable: &str) -> Vec<FloatValue> {
    model
        .timeseries()
        .get_data(variable)
        .and_then(|data| data.as_scalar())
        .unwrap()
        .values()
        .iter()
        .copied()
        .collect()
}

fn bits(values: &[FloatValue]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn flux(t: Time) -> FloatValue {
    1.0 + (t - 2000.0).powi(2)
}

#[test]
fn an_average_output_is_stored_at_its_step_start() {
    let mut model = builder(vec![Arc::new(FluxProducer), Arc::new(PoolIntegrator)])
        .build()
        .unwrap();
    model.run();

    let stored = values(&model, "Flux");
    let expected: Vec<_> = (0..5).map(|i| flux(2000.0 + i as f64)).collect();
    assert_eq!(stored[..5], expected[..]);
    assert!(stored[5].is_nan(), "no step starts at the last bound");
    assert_eq!(values(&model, "Flux Read")[..5], expected[..]);
    assert_eq!(
        values(&model, "Previous Flux")[..5],
        [-1.0, expected[0], expected[1], expected[2], expected[3]]
    );

    // The pool is a point, so each step's value lands at its end
    let pool = values(&model, "Pool");
    let mut total = 0.0;
    for (i, step_flux) in expected.iter().enumerate() {
        total += step_flux;
        assert_eq!(pool[i + 1], total);
    }
}

#[test]
fn prescribing_a_modelled_average_gives_identical_results() {
    let mut modelled = builder(vec![Arc::new(FluxProducer), Arc::new(PoolIntegrator)])
        .build()
        .unwrap();
    modelled.run();

    let stored = values(&modelled, "Flux");
    let prescribed_flux = Timeseries::new(
        Array1::from(stored.clone()).insert_axis(Axis(1)),
        Arc::new(axis()),
        ScalarGrid,
        "GtC / yr".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    );
    let mut prescribed = builder(vec![Arc::new(PoolIntegrator)])
        .with_exogenous_variable("Flux", prescribed_flux)
        .build()
        .unwrap();
    prescribed.run();

    for variable in ["Pool", "Flux Read", "Previous Flux"] {
        assert_eq!(
            bits(&values(&prescribed, variable)),
            bits(&values(&modelled, variable)),
            "{variable}"
        );
    }
}

#[test]
fn a_lagged_average_fails_to_build() {
    let components = || -> Vec<Arc<dyn Component + Send + Sync>> {
        vec![Arc::new(FluxProducer), Arc::new(PoolIntegrator)]
    };
    for (builder, lag) in [
        (
            builder(components()).with_lag("Flux").build(),
            "with_lag(\"Flux\")",
        ),
        (
            builder(components())
                .with_lag_for("Flux", "PoolIntegrator")
                .build(),
            "with_lag_for(\"Flux\", \"PoolIntegrator\")",
        ),
    ] {
        let error = builder.expect_err("a lagged average must not build");
        assert!(
            matches!(&error, RSCMError::InvalidLag { lag: l, .. } if l == lag),
            "{error}"
        );
        assert!(error.to_string().contains("average"), "{error}");
    }
}

#[test]
fn an_average_state_reads_its_previous_value() {
    let mut builder = builder(vec![Arc::new(AverageCounter)]);
    builder
        .with_initial_values(HashMap::from([("Flux".to_string(), 0.0)]))
        .with_state_type_check(false);
    let mut model = builder.build().unwrap();
    model.run();

    let stored = values(&model, "Flux");
    assert_eq!(stored[..5], [1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(stored[5].is_nan());
}
