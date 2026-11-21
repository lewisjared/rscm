//! `resolve` makes every coupling decision per labelled read and write, and the report projects it.

use crate::component::{Component, GridType, InputState, OutputState, VariableDefinition};
use crate::errors::RSCMResult;
use crate::model::{GridConversion, Model, ModelBuilder, TransformDirection, UnitConversion};
use crate::schema::VariableSchema;
use crate::state::VariableSource;
use crate::timeseries::{Time, TimeAxis};
use numpy::ndarray::Array;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

const WEIGHTS: [f64; 4] = [0.1, 0.2, 0.3, 0.4];

/// Declares `definitions` and writes nothing.
#[derive(Debug, Serialize, Deserialize)]
struct Declares(Vec<VariableDefinition>);

#[typetag::serde]
impl Component for Declares {
    fn definitions(&self) -> Vec<VariableDefinition> {
        self.0.clone()
    }

    fn solve(&self, _: Time, _: Time, _: &InputState) -> RSCMResult<OutputState> {
        Ok(OutputState::new())
    }
}

fn declares(definitions: Vec<VariableDefinition>) -> Arc<Declares> {
    Arc::new(Declares(definitions))
}

/// A four-box `T` read globally by one consumer and per hemisphere by another.
fn two_grids() -> Model {
    let mut builder = ModelBuilder::new();
    builder
        .with_time_axis(TimeAxis::from_values(Array::range(2000.0, 2003.0, 1.0)))
        .with_grid_weights(GridType::FourBox, WEIGHTS.to_vec())
        .with_component_as(
            "Producer",
            declares(vec![VariableDefinition::four_box_output("T", "K")]),
        )
        .with_component_as(
            "Global",
            declares(vec![
                VariableDefinition::scalar_input("T", "mK"),
                VariableDefinition::scalar_output("Global T", "K"),
            ]),
        )
        .with_component_as(
            "Hemispheres",
            declares(vec![
                VariableDefinition::hemispheric_input("T", "K"),
                VariableDefinition::scalar_output("Hemispheric T", "K"),
            ]),
        );
    builder.build().unwrap()
}

#[test]
fn two_consumers_at_different_grids_report_two_reads() {
    let coupling = two_grids().coupling();
    let reads: Vec<_> = coupling
        .conversions
        .iter()
        .filter(|c| c.variable == "T")
        .map(|c| (c.component.as_str(), c.direction, c.grid, c.unit.clone()))
        .collect();
    let grid = |to| {
        Some(GridConversion {
            from: GridType::FourBox,
            to,
        })
    };
    let millikelvin = UnitConversion {
        from: "K".into(),
        to: "mK".into(),
        factor: 1000.0,
    };
    assert_eq!(
        reads,
        [
            (
                "Global",
                TransformDirection::Read,
                grid(GridType::Scalar),
                Some(millikelvin)
            ),
            (
                "Hemispheres",
                TransformDirection::Read,
                grid(GridType::Hemispheric),
                None
            ),
        ]
    );
    let text = coupling.to_string();
    assert!(
        text.contains("T four_box -> scalar (read by Global)"),
        "{text}"
    );
    assert!(
        text.contains("T four_box -> hemispheric (read by Hemispheres)"),
        "{text}"
    );
}

#[test]
fn each_read_carries_the_weights_of_the_stored_grid() {
    let model = two_grids();
    let slot = model.timeseries().position("T").unwrap();
    for step in &model.plan().steps[1..] {
        let read = &step.reads[0];
        assert_eq!(read.slot, slot);
        assert_eq!(read.producer.as_deref(), Some("Producer"));
        assert_eq!(read.source, VariableSource::Upstream);
        assert_eq!(read.stored_grid, GridType::FourBox);
        assert_eq!(read.weights.as_deref(), Some(&WEIGHTS[..]));
        assert!(!read.plain);
    }
}

#[test]
fn a_finer_write_carries_the_weights_of_its_grid() {
    let mut builder = ModelBuilder::new();
    builder
        .with_time_axis(TimeAxis::from_values(Array::range(2000.0, 2003.0, 1.0)))
        .with_grid_weights(GridType::FourBox, WEIGHTS.to_vec())
        .with_schema(VariableSchema::new().variable("T", "mK"))
        .with_component_as(
            "Producer",
            declares(vec![VariableDefinition::four_box_output("T", "K")]),
        );
    let model = builder.build().unwrap();

    let write = &model.plan().steps[0].writes[0];
    assert_eq!(write.stored_grid, GridType::Scalar);
    assert_eq!(
        write.grid,
        Some(GridConversion {
            from: GridType::FourBox,
            to: GridType::Scalar
        })
    );
    assert_eq!(write.weights.as_deref(), Some(&WEIGHTS[..]));
    assert_eq!(write.unit.as_ref().map(|u| u.factor), Some(1000.0));
}

#[test]
fn only_a_prescribed_scalar_point_in_its_stored_unit_is_a_plain_read() {
    let mut builder = ModelBuilder::new();
    builder
        .with_time_axis(TimeAxis::from_values(Array::range(2000.0, 2003.0, 1.0)))
        .with_schema(VariableSchema::new().variable("Converted", "K"))
        .with_component_as(
            "Producer",
            declares(vec![VariableDefinition::scalar_output("Upstream", "K")]),
        )
        .with_component_as(
            "Reader",
            declares(vec![
                VariableDefinition::scalar_input("Plain", "K"),
                VariableDefinition::scalar_input("Converted", "mK"),
                VariableDefinition::scalar_input("Upstream", "K"),
                VariableDefinition::scalar_output("Out", "K"),
            ]),
        )
        .with_initial_values(HashMap::from([
            ("Plain".to_string(), 1.0),
            ("Converted".to_string(), 1.0),
        ]))
        .with_missing_prescribed_allowed();
    let model = builder.build().unwrap();

    let reader = model
        .plan()
        .steps
        .iter()
        .find(|s| s.label == "Reader")
        .unwrap();
    let plain: Vec<_> = reader
        .reads
        .iter()
        .map(|r| (r.definition.name.as_str(), r.plain))
        .collect();
    assert_eq!(
        plain,
        [("Plain", true), ("Converted", false), ("Upstream", false)]
    );
}

#[test]
fn the_plan_survives_serialisation() {
    let model = two_grids();
    let restored: Model = toml::from_str(&toml::to_string(&model).unwrap()).unwrap();
    assert_eq!(restored.plan(), model.plan());
}
