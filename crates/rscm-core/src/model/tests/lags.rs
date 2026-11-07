//! Global producer resolution and feedback lags.

use crate::component::{Component, InputState, OutputState, VariableDefinition, VariableRole};
use crate::errors::{RscmError, RscmResult};
use crate::model::{Model, ModelBuilder};
use crate::state::{StateValue, VariableSource};
use crate::timeseries::{FloatValue, Time, TimeAxis};
use numpy::ndarray::Array;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Defines a component that writes one plus the sum of its inputs (and its own state).
macro_rules! link {
    ($name:ident, inputs: [$($input:expr),*], output: $output:expr, state: $state:expr) => {
        #[derive(Debug, Serialize, Deserialize)]
        struct $name;

        #[typetag::serde]
        impl Component for $name {
            fn definitions(&self) -> Vec<VariableDefinition> {
                let mut definitions: Vec<VariableDefinition> = vec![
                    $(VariableDefinition::new($input, "1", VariableRole::Input),)*
                ];
                let output_type = if $state {
                    VariableRole::State
                } else {
                    VariableRole::Output
                };
                definitions.push(VariableDefinition::new($output, "1", output_type));
                definitions
            }

            fn solve(
                &self,
                _t_current: Time,
                _t_next: Time,
                input_state: &InputState,
            ) -> RscmResult<OutputState> {
                let mut total = 1.0;
                $(total += input_state.get_scalar_window($input).get();)*
                if $state {
                    total += input_state.get_scalar_window($output).get();
                }
                Ok(HashMap::from([($output.to_string(), StateValue::Scalar(total))]))
            }
        }
    };
}

// A -> B -> C, a chain without loops.
link!(ChainA, inputs: [], output: "a", state: false);
link!(ChainB, inputs: ["a"], output: "b", state: false);
link!(ChainC, inputs: ["b"], output: "c", state: false);

// Climate and Carbon form a loop through temperature and CO2, and Ozone also reads temperature.
link!(Climate, inputs: ["co2"], output: "temperature", state: true);
link!(Carbon, inputs: ["temperature"], output: "co2", state: false);
link!(Ozone, inputs: ["temperature"], output: "ozone", state: false);
link!(OtherCarbon, inputs: [], output: "co2", state: false);

fn builder(components: Vec<Arc<dyn Component + Send + Sync>>) -> ModelBuilder {
    let mut builder = ModelBuilder::new();
    builder
        .with_time_axis(TimeAxis::from_values(Array::range(2000.0, 2006.0, 1.0)))
        .with_initial_values(HashMap::from([
            ("temperature".to_string(), 0.0),
            ("co2".to_string(), 0.0),
            ("a".to_string(), 0.0),
            ("b".to_string(), 0.0),
            ("c".to_string(), 0.0),
            ("ozone".to_string(), 0.0),
        ]));
    for component in components {
        builder.with_component(component);
    }
    builder
}

fn source(model: &Model, variable: &str, component: &str) -> VariableSource {
    model.variable_sources()[&(variable.to_string(), component.to_string())]
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

#[test]
fn unlagged_loop_fails_to_build() {
    let error = builder(vec![Arc::new(Climate), Arc::new(Carbon)])
        .build()
        .expect_err("a loop without a lag must not build");
    let RscmError::FeedbackLoop {
        components,
        variables,
        suggestions,
    } = &error
    else {
        panic!("expected FeedbackLoop, got {error}");
    };
    assert_eq!(components, &["Carbon", "Climate"]);
    assert_eq!(variables, &["co2", "temperature"]);
    // Climate holds its own state, so the edge into it ranks first.
    assert_eq!(
        suggestions,
        &[
            ("co2".to_string(), "Climate".to_string()),
            ("temperature".to_string(), "Carbon".to_string()),
        ]
    );
    assert_eq!(
        error.to_string(),
        "Feedback loop among components [Carbon, Climate] through variables [co2, temperature]. \
         Break it with one of: with_lag_for(\"co2\", \"Climate\"), \
         with_lag_for(\"temperature\", \"Carbon\")"
    );
}

#[test]
fn lagged_loop_builds_in_any_order() {
    let run = |components: Vec<Arc<dyn Component + Send + Sync>>| {
        let mut model = builder(components)
            .with_lag_for("temperature", "Carbon")
            .build()
            .unwrap();
        model.run();
        model
    };
    let forward = run(vec![Arc::new(Climate), Arc::new(Carbon)]);
    let backward = run(vec![Arc::new(Carbon), Arc::new(Climate)]);

    assert_eq!(
        source(&forward, "temperature", "Carbon"),
        VariableSource::Lagged
    );
    assert_eq!(source(&forward, "co2", "Climate"), VariableSource::Upstream);
    assert_eq!(forward.variable_sources(), backward.variable_sources());
    for variable in ["temperature", "co2"] {
        assert_eq!(values(&forward, variable), values(&backward, variable));
    }
    // Carbon reads last step's temperature, so co2 = 1 + T[n] and T[n+1] = 1 + co2 + T[n].
    assert_eq!(values(&forward, "co2"), [0.0, 1.0, 3.0, 7.0, 15.0, 31.0]);
    assert_eq!(
        values(&forward, "temperature"),
        [0.0, 2.0, 6.0, 14.0, 30.0, 62.0]
    );
}

#[test]
fn shuffled_chain_gives_identical_sources() {
    let orders: [Vec<Arc<dyn Component + Send + Sync>>; 3] = [
        vec![Arc::new(ChainA), Arc::new(ChainB), Arc::new(ChainC)],
        vec![Arc::new(ChainC), Arc::new(ChainB), Arc::new(ChainA)],
        vec![Arc::new(ChainB), Arc::new(ChainC), Arc::new(ChainA)],
    ];
    let models: Vec<Model> = orders
        .into_iter()
        .map(|components| {
            let mut model = builder(components).build().unwrap();
            model.run();
            model
        })
        .collect();
    for model in &models {
        assert_eq!(source(model, "a", "ChainB"), VariableSource::Upstream);
        assert_eq!(source(model, "b", "ChainC"), VariableSource::Upstream);
        assert_eq!(model.variable_sources(), models[0].variable_sources());
        assert_eq!(values(model, "c"), values(&models[0], "c"));
    }
    assert_eq!(values(&models[0], "c"), [0.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
}

#[test]
fn lag_for_leaves_sibling_consumer_unlagged() {
    let model = builder(vec![Arc::new(Ozone), Arc::new(Carbon), Arc::new(Climate)])
        .with_lag_for("temperature", "Carbon")
        .build()
        .unwrap();
    assert_eq!(
        source(&model, "temperature", "Carbon"),
        VariableSource::Lagged
    );
    assert_eq!(
        source(&model, "temperature", "Ozone"),
        VariableSource::Upstream
    );
}

#[test]
fn lag_marks_every_consumer() {
    let model = builder(vec![Arc::new(Ozone), Arc::new(Carbon), Arc::new(Climate)])
        .with_lag("temperature")
        .build()
        .unwrap();
    assert_eq!(
        source(&model, "temperature", "Carbon"),
        VariableSource::Lagged
    );
    assert_eq!(
        source(&model, "temperature", "Ozone"),
        VariableSource::Lagged
    );
    assert_eq!(
        source(&model, "temperature", "Climate"),
        VariableSource::OwnState
    );
}

#[test]
fn lag_holds_for_a_prescribed_variable() {
    let model = builder(vec![Arc::new(Ozone)])
        .with_lag("temperature")
        .build()
        .unwrap();
    assert_eq!(
        source(&model, "temperature", "Ozone"),
        VariableSource::Lagged
    );
}

#[test]
fn invalid_lags_fail_to_build() {
    let cases = [
        (
            builder(vec![Arc::new(Carbon)])
                .with_lag_for("temperature", "Nobody")
                .build(),
            "no component named \"Nobody\"",
        ),
        (
            builder(vec![Arc::new(Carbon)])
                .with_lag_for("co2", "Carbon")
                .build(),
            "Carbon does not read \"co2\" as an input",
        ),
        (
            builder(vec![Arc::new(Carbon)]).with_lag("nitrogen").build(),
            "no component reads or produces this variable",
        ),
    ];
    for (result, reason) in cases {
        let error = result.expect_err("an invalid lag must not build");
        assert!(
            matches!(error, RscmError::InvalidLag { .. }) && error.to_string().contains(reason),
            "{error}"
        );
    }
}

#[test]
fn two_producers_fail_to_build() {
    let error = builder(vec![Arc::new(Carbon), Arc::new(OtherCarbon)])
        .build()
        .expect_err("two producers must not build");
    assert!(
        matches!(&error, RscmError::MultipleProducers { variable, .. } if variable == "co2"),
        "{error}"
    );
}
