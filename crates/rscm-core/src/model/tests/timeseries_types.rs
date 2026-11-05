//! Point and average resolution of each variable's timeseries type.

use crate::component::{Component, InputState, OutputState, RequirementDefinition, TimeseriesType};
use crate::errors::{RSCMError, RSCMResult};
use crate::model::{Model, ModelBuilder};
use crate::spatial::ScalarRegion;
use crate::state::StateValue;
use crate::timeseries::{Time, TimeAxis, Timeseries};
use numpy::ndarray::{array, Array};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Defines a component with the given definitions that writes `t_next` to every output.
macro_rules! declared {
    ($name:ident) => {
        #[derive(Debug, Serialize, Deserialize)]
        struct $name(Vec<RequirementDefinition>);

        #[typetag::serde]
        impl Component for $name {
            fn definitions(&self) -> Vec<RequirementDefinition> {
                self.0.clone()
            }

            fn solve(
                &self,
                _t_current: Time,
                t_next: Time,
                _input_state: &InputState,
            ) -> RSCMResult<OutputState> {
                Ok(self
                    .outputs()
                    .into_iter()
                    .map(|d| (d.name, StateValue::Scalar(t_next)))
                    .collect())
            }
        }
    };
}

declared!(Producer);
declared!(Consumer);

fn input(name: &str, unit: &str) -> RequirementDefinition {
    RequirementDefinition::scalar_input(name, unit)
}

fn output(name: &str, unit: &str) -> RequirementDefinition {
    RequirementDefinition::scalar_output(name, unit)
}

fn builder(components: Vec<Arc<dyn Component + Send + Sync>>) -> ModelBuilder {
    let mut builder = ModelBuilder::new();
    builder.with_time_axis(TimeAxis::from_values(Array::range(2000.0, 2006.0, 1.0)));
    for component in components {
        builder.with_component(component);
    }
    builder
}

fn build(components: Vec<Arc<dyn Component + Send + Sync>>) -> RSCMResult<Model> {
    builder(components).build()
}

fn lifetime_producer() -> Arc<Producer> {
    Arc::new(Producer(vec![
        output("Lifetime", "yr").with_timeseries_type(TimeseriesType::Average)
    ]))
}

#[test]
fn types_default_from_the_unit() {
    let model = build(vec![
        Arc::new(Producer(vec![
            output("Emissions", "GtC / yr"),
            output("Temperature", "K"),
            output("Forcing", "W / m^2"),
        ])),
        Arc::new(Consumer(vec![input("Prescribed Emissions", "Mt CO2 / yr")])),
    ])
    .unwrap();
    let types = [
        ("Emissions", TimeseriesType::Average),
        ("Temperature", TimeseriesType::Point),
        ("Forcing", TimeseriesType::Point),
        ("Prescribed Emissions", TimeseriesType::Average),
    ];
    for (variable, expected) in types {
        assert_eq!(
            model.timeseries_type(variable),
            Some(expected),
            "{variable}"
        );
    }
    assert_eq!(model.timeseries_type("Missing"), None);
}

#[test]
fn consumers_inherit_the_producer_type() {
    let model = build(vec![
        lifetime_producer(),
        Arc::new(Consumer(vec![input("Lifetime", "yr")])),
    ])
    .unwrap();
    assert_eq!(
        model.timeseries_type("Lifetime"),
        Some(TimeseriesType::Average)
    );

    let agreeing = Consumer(vec![
        input("Lifetime", "yr").with_timeseries_type(TimeseriesType::Average)
    ]);
    build(vec![lifetime_producer(), Arc::new(agreeing)]).unwrap();
}

#[test]
fn a_consumer_declaring_another_type_is_an_error() {
    let disagreeing = Consumer(vec![
        input("Lifetime", "yr").with_timeseries_type(TimeseriesType::Point)
    ]);
    let err = build(vec![lifetime_producer(), Arc::new(disagreeing)]).unwrap_err();
    match err {
        RSCMError::TimeseriesTypeMismatch {
            variable,
            producer,
            producer_type,
            consumer,
            consumer_type,
        } => {
            assert_eq!(variable, "Lifetime");
            assert_eq!(producer, "Producer");
            assert_eq!(producer_type, "Average");
            assert_eq!(consumer, "Consumer");
            assert_eq!(consumer_type, "Point");
        }
        other => panic!("expected TimeseriesTypeMismatch, got {other:?}"),
    }
}

#[test]
fn exogenous_type_comes_from_the_builder_then_consumers_then_the_unit() {
    let consumer = || Arc::new(Consumer(vec![input("Lifetime", "yr")]));

    let model = build(vec![consumer()]).unwrap();
    assert_eq!(
        model.timeseries_type("Lifetime"),
        Some(TimeseriesType::Point)
    );

    let model = builder(vec![consumer()])
        .with_timeseries_type("Lifetime", TimeseriesType::Average)
        .build()
        .unwrap();
    assert_eq!(
        model.timeseries_type("Lifetime"),
        Some(TimeseriesType::Average)
    );

    let declaring = Arc::new(Producer(vec![
        input("Lifetime", "yr").with_timeseries_type(TimeseriesType::Average)
    ]));
    let model = build(vec![consumer(), declaring]).unwrap();
    assert_eq!(
        model.timeseries_type("Lifetime"),
        Some(TimeseriesType::Average)
    );

    let disagreeing = Arc::new(Producer(vec![
        input("Lifetime", "yr").with_timeseries_type(TimeseriesType::Point)
    ]));
    let err = builder(vec![disagreeing])
        .with_timeseries_type("Lifetime", TimeseriesType::Average)
        .build()
        .unwrap_err();
    assert!(
        matches!(err, RSCMError::TimeseriesTypeMismatch { ref producer, .. }
            if producer == "ModelBuilder::with_timeseries_type"),
        "{err:?}"
    );
}

#[test]
fn a_builder_declaration_must_match_the_producer() {
    let err = builder(vec![lifetime_producer()])
        .with_timeseries_type("Lifetime", TimeseriesType::Point)
        .build()
        .unwrap_err();
    assert!(
        matches!(err, RSCMError::TimeseriesTypeMismatch { ref consumer, .. }
            if consumer == "ModelBuilder::with_timeseries_type"),
        "{err:?}"
    );
}

#[test]
fn an_average_state_is_an_error_unless_the_check_is_off() {
    let stateful = || {
        Arc::new(Producer(vec![RequirementDefinition::scalar_state(
            "Flux", "GtC / yr",
        )]))
    };
    let mut builder = builder(vec![stateful()]);
    builder.with_initial_values(HashMap::from([("Flux".to_string(), 0.0)]));

    let err = builder.build().unwrap_err();
    assert!(
        matches!(err, RSCMError::AverageState { ref variable, ref component }
            if variable == "Flux" && component == "Producer"),
        "{err:?}"
    );

    let model = builder.with_state_type_check(false).build().unwrap();
    assert_eq!(model.timeseries_type("Flux"), Some(TimeseriesType::Average));
}

#[test]
fn a_point_state_passes_the_check() {
    let point = Producer(vec![RequirementDefinition::scalar_state(
        "Flux", "GtC / yr",
    )
    .with_timeseries_type(TimeseriesType::Point)]);
    builder(vec![Arc::new(point)])
        .with_initial_values(HashMap::from([("Flux".to_string(), 0.0)]))
        .build()
        .unwrap();
}

#[test]
fn builder_series_interpolate_by_type() {
    let mut model = build(vec![Arc::new(Producer(vec![
        output("Emissions", "GtC / yr"),
        output("Concentration", "ppm"),
    ]))])
    .unwrap();
    model.run();
    let at = |variable: &str| {
        model
            .timeseries()
            .get_data(variable)
            .and_then(|data| data.as_scalar())
            .unwrap()
            .at_time(2002.5, ScalarRegion::Global)
            .unwrap()
    };
    assert_eq!(at("Emissions"), 2002.0);
    assert_eq!(at("Concentration"), 2002.5);
}

#[test]
fn exogenous_series_keep_their_interpolation() {
    let data = Timeseries::from_values(array![0.0, 10.0], array![2000.0, 2010.0]);
    let model = builder(vec![Arc::new(Consumer(vec![input(
        "Emissions",
        "GtC / yr",
    )]))])
    .with_exogenous_variable("Emissions", data)
    .build()
    .unwrap();
    let value = model
        .timeseries()
        .get_data("Emissions")
        .and_then(|data| data.as_scalar())
        .unwrap()
        .at_time(2002.5, ScalarRegion::Global)
        .unwrap();
    assert_eq!(value, 2.5);
}
