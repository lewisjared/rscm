//! Records: checkpoints, restore and branching.

use crate::component::{
    Component, ComponentState, InputState, OutputState, VariableDefinition, VariableRole,
};
use crate::errors::RSCMResult;
use crate::interpolate::strategies::{InterpolationStrategy, PreviousStrategy};
use crate::model::{Branch, Model, ModelBuilder, Record, RECORD_FORMAT};
use crate::spatial::ScalarGrid;
use crate::state::StateValue;
use crate::timeseries::{FloatValue, Time, TimeAxis, Timeseries};
use crate::timeseries_collection::TimeseriesCollection;
use numpy::ndarray::{Array, Array1, Axis};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

#[derive(Debug, Serialize, Deserialize)]
struct Calls(usize);

#[typetag::serde]
impl ComponentState for Calls {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Adds `rate` times the emissions and its call count to its stock each step.
#[derive(Debug, Serialize, Deserialize)]
struct Accumulate {
    rate: FloatValue,
    unit: String,
}

#[typetag::serde]
impl Component for Accumulate {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![
            VariableDefinition::scalar_input("Emissions", "GtC / yr"),
            VariableDefinition::new("Stock", &self.unit, VariableRole::State),
        ]
    }

    fn create_initial_state(&self) -> Box<dyn ComponentState> {
        Box::new(Calls(0))
    }

    fn solve(&self, _: Time, _: Time, _: &InputState) -> RSCMResult<OutputState> {
        unreachable!("solved with state")
    }

    fn solve_with_state(
        &self,
        _: Time,
        _: Time,
        input_state: &InputState,
        internal_state: &mut dyn ComponentState,
    ) -> RSCMResult<OutputState> {
        let calls = internal_state.as_any_mut().downcast_mut::<Calls>().unwrap();
        calls.0 += 1;
        let stock = input_state.get_scalar_window("Stock").get()
            + self.rate * input_state.get_scalar_window("Emissions").get()
            + calls.0 as FloatValue;
        Ok(HashMap::from([("Stock".into(), StateValue::Scalar(stock))]))
    }
}

fn time_axis() -> Arc<TimeAxis> {
    Arc::new(TimeAxis::from_values(Array::range(2000.0, 2010.0, 1.0)))
}

fn emissions(values: Vec<FloatValue>) -> Timeseries<FloatValue> {
    Timeseries::new(
        Array1::from(values).insert_axis(Axis(1)),
        time_axis(),
        ScalarGrid,
        "GtC / yr".to_string(),
        InterpolationStrategy::from(PreviousStrategy::new(true)),
    )
}

fn base_emissions() -> Vec<FloatValue> {
    (0..10).map(|i| i as FloatValue).collect()
}

fn builder(emission_values: Vec<FloatValue>) -> ModelBuilder {
    let mut builder = ModelBuilder::new();
    builder
        .with_time_axis(time_axis())
        .with_component_as(
            "Fast",
            Arc::new(Accumulate {
                rate: 2.0,
                unit: "GtC".into(),
            }),
        )
        .with_prescribed_variable("Emissions", emissions(emission_values))
        .with_initial_values(HashMap::from([("Stock".to_string(), 1.0)]));
    builder
}

fn stock(model: &Model) -> Vec<FloatValue> {
    model
        .timeseries()
        .get_data("Stock")
        .and_then(|data| data.as_scalar())
        .unwrap()
        .values()
        .iter()
        .copied()
        .collect()
}

fn through_json(record: Record) -> Record {
    Record::from_json(&record.to_json().unwrap()).unwrap()
}

#[test]
fn a_mid_run_checkpoint_resumes_to_the_uninterrupted_output() {
    let mut uninterrupted = builder(base_emissions()).build().unwrap();
    uninterrupted.run();

    let mut model = builder(base_emissions()).build().unwrap();
    for _ in 0..4 {
        model.step();
    }
    let mut resumed = Model::restore(&through_json(model.checkpoint().unwrap())).unwrap();
    assert_eq!(resumed.time_index(), 4);
    resumed.run();

    assert_eq!(stock(&resumed), stock(&uninterrupted));
    let output = through_json(resumed.output().unwrap());
    assert_eq!(output.time_index, 9);
    assert_eq!(
        output
            .values
            .get_data("Stock")
            .unwrap()
            .as_scalar()
            .unwrap()
            .values(),
        uninterrupted
            .timeseries()
            .get_data("Stock")
            .unwrap()
            .as_scalar()
            .unwrap()
            .values()
    );
}

#[test]
fn states_are_keyed_by_label() {
    let mut model = builder(base_emissions()).build().unwrap();
    model.step();
    let record = model.checkpoint().unwrap();
    assert_eq!(record.states.keys().collect::<Vec<_>>(), ["Fast"]);
    assert_eq!(record.states["Fast"]["state_type"], "Calls");
}

#[test]
fn a_branch_with_new_future_prescribed_data_equals_a_fresh_run_with_it() {
    let mut new_values = base_emissions();
    for value in &mut new_values[5..] {
        *value *= 10.0;
    }
    let mut fresh = builder(new_values.clone()).build().unwrap();
    fresh.run();

    let mut model = builder(base_emissions()).build().unwrap();
    for _ in 0..5 {
        model.step();
    }
    let mut prescribed = TimeseriesCollection::new();
    prescribed.add_timeseries("Emissions".into(), emissions(new_values));
    let record = through_json(model.checkpoint().unwrap());
    let mut branched = Model::branch(
        &record,
        Branch {
            prescribed,
            ..Branch::default()
        },
    )
    .unwrap();
    branched.run();

    assert_eq!(stock(&branched), stock(&fresh));
    let mut unbranched = Model::restore(&record).unwrap();
    unbranched.run();
    assert_ne!(stock(&branched), stock(&unbranched));
}

#[test]
fn saved_history_wins_before_the_branch_point() {
    let mut model = builder(base_emissions()).build().unwrap();
    for _ in 0..5 {
        model.step();
    }
    let mut prescribed = TimeseriesCollection::new();
    prescribed.add_timeseries("Emissions".into(), emissions(vec![100.0; 10]));
    let branched = Model::branch(
        &model.checkpoint().unwrap(),
        Branch {
            prescribed,
            ..Branch::default()
        },
    )
    .unwrap();

    let emitted = |m: &Model| -> Vec<FloatValue> {
        let data = m.timeseries().get_data("Emissions").unwrap();
        data.as_scalar().unwrap().values().iter().copied().collect()
    };
    // Emissions are an average, so the step at the branch point is already new.
    assert_eq!(emitted(&branched)[..5], emitted(&model)[..5]);
    assert_eq!(emitted(&branched)[5..], [100.0; 5]);
    assert_eq!(stock(&branched)[..6], stock(&model)[..6]);
}

#[test]
fn a_branch_replaces_parameters_by_label() {
    let mut fresh_builder = builder(base_emissions());
    fresh_builder
        .replace_component(
            "Fast",
            Arc::new(Accumulate {
                rate: 3.0,
                unit: "GtC".into(),
            }),
        )
        .unwrap();
    let mut model = builder(base_emissions()).build().unwrap();
    model.step();
    let record = model.checkpoint().unwrap();

    let mut branched = Model::branch(
        &record,
        Branch {
            parameters: BTreeMap::from([(
                "Fast".to_string(),
                Arc::new(Accumulate {
                    rate: 3.0,
                    unit: "GtC".into(),
                }) as Arc<dyn Component>,
            )]),
            ..Branch::default()
        },
    )
    .unwrap();
    branched.run();
    let mut fresh = fresh_builder.build().unwrap();
    fresh.run();

    // The first step ran with the old rate, so only the later increments match.
    let (branched, fresh) = (stock(&branched), stock(&fresh));
    assert_eq!(branched[..2], stock(&model)[..2]);
    for i in 2..branched.len() {
        assert_eq!(branched[i] - branched[i - 1], fresh[i] - fresh[i - 1]);
    }
}

#[test]
fn a_branch_rejects_another_type_or_label() {
    let record = builder(base_emissions())
        .build()
        .unwrap()
        .checkpoint()
        .unwrap();
    let replace = |label: &str, component: Arc<dyn Component>| {
        Model::branch(
            &record,
            Branch {
                parameters: BTreeMap::from([(label.to_string(), component)]),
                ..Branch::default()
            },
        )
        .unwrap_err()
        .to_string()
    };
    let other = Arc::new(crate::model::null_component::NullComponent {});
    assert!(replace("Fast", other.clone()).contains("'Fast' has type Accumulate"));
    assert!(replace("Slow", other).contains("No component is labelled 'Slow'"));
}

#[test]
fn a_changed_declared_unit_fails_restore_naming_the_variable() {
    let mut record = builder(base_emissions())
        .build()
        .unwrap()
        .checkpoint()
        .unwrap();
    Arc::make_mut(&mut record.builder)
        .replace_component(
            "Fast",
            Arc::new(Accumulate {
                rate: 2.0,
                unit: "MtC".into(),
            }),
        )
        .unwrap();

    let err = Model::restore(&record).unwrap_err().to_string();
    assert!(err.contains("'Stock'"), "{err}");
    assert!(err.contains("'MtC'") && err.contains("'GtC'"), "{err}");
}

#[test]
fn a_format_mismatch_errors_with_both_numbers() {
    let mut record = builder(base_emissions())
        .build()
        .unwrap()
        .checkpoint()
        .unwrap();
    record.format = RECORD_FORMAT + 1;
    let expected = format!(
        "The record has format {}, but this rscm reads format {RECORD_FORMAT}",
        RECORD_FORMAT + 1
    );

    assert!(Model::restore(&record)
        .unwrap_err()
        .to_string()
        .contains(&expected));
    let err = Record::from_json(&record.to_json().unwrap()).unwrap_err();
    assert!(err.to_string().contains(&expected), "{err}");
}

#[test]
fn another_rscm_version_restores_with_a_warning() {
    let mut record = builder(base_emissions())
        .build()
        .unwrap()
        .checkpoint()
        .unwrap();
    record.rscm = "0.0.1".into();

    let (_, warnings) = Model::restore_with_warnings(&record).unwrap();
    assert_eq!(
        warnings,
        [format!(
            "The record was written by rscm 0.0.1, this is rscm {}, so results may differ",
            env!("CARGO_PKG_VERSION")
        )]
    );
}

#[test]
fn a_branch_rejects_prescribed_data_for_a_modelled_variable() {
    let record = builder(base_emissions())
        .build()
        .unwrap()
        .checkpoint()
        .unwrap();
    let mut prescribed = TimeseriesCollection::new();
    prescribed.add_timeseries("Stock".into(), emissions(vec![0.0; 10]));

    let err = Model::branch(
        &record,
        Branch {
            prescribed,
            ..Branch::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("'Stock'"), "{err}");
}
