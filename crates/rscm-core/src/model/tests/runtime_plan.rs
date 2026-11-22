//! The runtime steps the plan the builder resolved.

use crate::component::{
    Component, GridType, InputState, OutputState, VariableDefinition, VariableRole,
};
use crate::errors::RSCMResult;
use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
use crate::model::{Model, ModelBuilder};
use crate::schema::VariableSchema;
use crate::spatial::ScalarGrid;
use crate::state::{FourBoxSlice, HemisphericSlice, StateValue};
use crate::timeseries::{FloatValue, Time, TimeAxis, Timeseries};
use is_close::is_close;
use numpy::ndarray::{array, Array};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

const WEIGHTS: [f64; 4] = [0.1, 0.2, 0.3, 0.4];

/// Writes `value` to `output` every step.
#[derive(Debug, Serialize, Deserialize)]
struct Emit {
    output: VariableDefinition,
    value: StateValue,
}

#[typetag::serde]
impl Component for Emit {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![self.output.clone()]
    }

    fn solve(&self, _: Time, _: Time, _: &InputState) -> RSCMResult<OutputState> {
        Ok(HashMap::from([(
            self.output.name.clone(),
            self.value.clone(),
        )]))
    }
}

/// Writes the start of each step to `x`.
#[derive(Debug, Serialize, Deserialize)]
struct Clock;

#[typetag::serde]
impl Component for Clock {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![VariableDefinition::scalar_output("x", "1")]
    }

    fn solve(&self, t_current: Time, _: Time, _: &InputState) -> RSCMResult<OutputState> {
        Ok(HashMap::from([("x".into(), StateValue::Scalar(t_current))]))
    }
}

/// Copies what it reads of `input` to `output`, on the input's grid.
#[derive(Debug, Serialize, Deserialize)]
struct Mirror {
    input: VariableDefinition,
    output: VariableDefinition,
}

impl Mirror {
    fn new(input: VariableDefinition, output: &str) -> Arc<Self> {
        let output =
            VariableDefinition::with_grid(output, &input.unit, VariableRole::Output, input.grid);
        Arc::new(Self { input, output })
    }
}

#[typetag::serde]
impl Component for Mirror {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![self.input.clone(), self.output.clone()]
    }

    fn solve(&self, _: Time, _: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        let name = &self.input.name;
        let value = match self.input.grid {
            GridType::Scalar => StateValue::Scalar(input_state.get_scalar_window(name).get()),
            GridType::Hemispheric => {
                let all = input_state.get_hemispheric_window(name).get_all();
                StateValue::Hemispheric(HemisphericSlice::from_array([all[0], all[1]]))
            }
            GridType::FourBox => {
                let all = input_state.get_four_box_window(name).get_all();
                StateValue::FourBox(FourBoxSlice::from_array([all[0], all[1], all[2], all[3]]))
            }
        };
        Ok(HashMap::from([(self.output.name.clone(), value)]))
    }
}

/// Adds one of its own unit to its state `s` each step.
#[derive(Debug, Serialize, Deserialize)]
struct Tally {
    unit: String,
}

#[typetag::serde]
impl Component for Tally {
    fn definitions(&self) -> Vec<VariableDefinition> {
        vec![VariableDefinition::new(
            "s",
            &self.unit,
            VariableRole::State,
        )]
    }

    fn solve(&self, _: Time, _: Time, input_state: &InputState) -> RSCMResult<OutputState> {
        let s = input_state.get_scalar_window("s").get();
        Ok(HashMap::from([("s".into(), StateValue::Scalar(s + 1.0))]))
    }
}

fn builder() -> ModelBuilder {
    let mut builder = ModelBuilder::new();
    builder.with_time_axis(TimeAxis::from_values(Array::range(2000.0, 2004.0, 1.0)));
    builder
}

fn scalar(model: &Model, name: &str) -> Vec<FloatValue> {
    let data = model.timeseries().get_data(name).unwrap();
    let ts = data.as_scalar().unwrap();
    (0..ts.len()).map(|i| ts.at_scalar(i).unwrap()).collect()
}

#[test]
fn two_consumers_at_different_grids_read_with_the_model_weights() {
    let mut model = builder()
        .with_grid_weights(GridType::FourBox, WEIGHTS.to_vec())
        .with_component_as(
            "Producer",
            Arc::new(Emit {
                output: VariableDefinition::four_box_output("T", "K"),
                value: StateValue::FourBox(FourBoxSlice::from_array([1.0, 2.0, 3.0, 4.0])),
            }),
        )
        .with_component_as(
            "Global",
            Mirror::new(VariableDefinition::scalar_input("T", "mK"), "Global T"),
        )
        .with_component_as(
            "Hemispheres",
            Mirror::new(
                VariableDefinition::hemispheric_input("T", "K"),
                "Hemispheric T",
            ),
        )
        .build()
        .unwrap();
    model.run();

    // (0.1 + 0.4 + 0.9 + 1.6) K, read in mK.
    let global = scalar(&model, "Global T")[1];
    assert!(is_close!(global, 3000.0), "{global}");
    let hemispheric = model
        .timeseries()
        .get_data("Hemispheric T")
        .unwrap()
        .as_hemispheric()
        .unwrap()
        .at_time_index(1)
        .unwrap();
    assert!(is_close!(hemispheric[0], 0.5 / 0.3), "{hemispheric:?}");
    assert!(is_close!(hemispheric[1], 2.5 / 0.7), "{hemispheric:?}");
}

#[test]
fn a_lagged_point_reads_the_previous_step_when_modelled() {
    let mut model = builder()
        .with_component(Arc::new(Clock))
        .with_component(Mirror::new(
            VariableDefinition::scalar_input("x", "1"),
            "seen",
        ))
        .with_lag("x")
        .with_initial_values(HashMap::from([("x".to_string(), -1.0)]))
        .build()
        .unwrap();
    model.run();

    assert_eq!(scalar(&model, "x")[1..], [2000.0, 2001.0, 2002.0]);
    assert_eq!(scalar(&model, "seen")[1..], [-1.0, 2000.0, 2001.0]);
}

#[test]
fn a_lagged_point_reads_the_previous_step_when_prescribed() {
    let mut model = builder()
        .with_component(Mirror::new(
            VariableDefinition::scalar_input("x", "1"),
            "seen",
        ))
        .with_lag("x")
        .with_prescribed_variable(
            "x",
            Timeseries::new(
                array![[10.0], [11.0], [12.0], [13.0]],
                Arc::new(TimeAxis::from_values(Array::range(2000.0, 2004.0, 1.0))),
                ScalarGrid,
                "1".to_string(),
                InterpolationStrategy::from(LinearSplineStrategy::new(true)),
            ),
        )
        .build()
        .unwrap();
    model.run();

    assert_eq!(scalar(&model, "seen")[1..], [10.0, 11.0, 12.0]);
}

#[test]
fn a_state_converts_on_read_and_on_write() {
    let mut model = builder()
        .with_schema(VariableSchema::new().variable("s", "K"))
        .with_component(Arc::new(Tally {
            unit: "mK".to_string(),
        }))
        .with_initial_values(HashMap::from([("s".to_string(), 1.0)]))
        .build()
        .unwrap();
    let step = &model.plan().steps[0];
    assert_eq!(step.reads[0].unit.as_ref().map(|u| u.factor), Some(1000.0));
    assert_eq!(step.writes[0].unit.as_ref().map(|u| u.factor), Some(0.001));
    model.run();

    let stored = scalar(&model, "s");
    for (value, expected) in stored.iter().zip([1.0, 1.001, 1.002, 1.003]) {
        assert!(is_close!(*value, expected), "{stored:?}");
    }
}

#[test]
fn a_save_without_a_plan_steps_like_the_original() {
    let build = || {
        builder()
            .with_component(Arc::new(Clock))
            .with_component(Mirror::new(
                VariableDefinition::scalar_input("x", "1"),
                "seen",
            ))
            .build()
            .unwrap()
    };
    let mut original = build();
    let saved = toml::to_string(&build()).unwrap();
    let without_plan = &saved[..saved.find("[[step_plan.steps]]").unwrap()];
    let mut restored: Model = toml::from_str(without_plan).unwrap();
    assert!(restored.plan().steps.is_empty());

    original.run();
    restored.run();
    assert_eq!(restored.plan(), original.plan());
    assert_eq!(
        scalar(&restored, "seen")[1..],
        scalar(&original, "seen")[1..]
    );
}
