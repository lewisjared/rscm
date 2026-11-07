//! `Model::coupling` reports the runtime's execution plan.

use crate::component::{Component, InputState, OutputState, VariableDefinition, VariableRole};
use crate::errors::RSCMResult;
use crate::model::{CouplingEdge, Model, ModelBuilder};
use crate::state::StateValue;
use crate::timeseries::{Time, TimeAxis};
use crate::VariableSource;
use numpy::ndarray::Array;
use petgraph::visit::Bfs;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

macro_rules! link {
    ($name:ident, inputs: [$($input:expr),*], output: $output:expr) => {
        #[derive(Debug, Serialize, Deserialize)]
        struct $name;

        #[typetag::serde]
        impl Component for $name {
            fn definitions(&self) -> Vec<VariableDefinition> {
                vec![
                    $(VariableDefinition::new($input, "1", VariableRole::Input),)*
                    VariableDefinition::new($output, "1", VariableRole::Output),
                ]
            }

            fn solve(
                &self,
                _t_current: Time,
                _t_next: Time,
                _input_state: &InputState,
            ) -> RSCMResult<OutputState> {
                Ok(HashMap::from([($output.to_string(), StateValue::Scalar(1.0))]))
            }
        }
    };
}

// Sink reads Source directly and through Mid, and Source reads Sink's output from the previous step.
link!(Source, inputs: ["c"], output: "a");
link!(Mid, inputs: ["a"], output: "b");
link!(Sink, inputs: ["a", "b"], output: "c");

fn model() -> Model {
    let mut builder = ModelBuilder::new();
    builder
        .with_time_axis(TimeAxis::from_values(Array::range(2000.0, 2003.0, 1.0)))
        .with_component(Arc::new(Source))
        .with_component(Arc::new(Mid))
        .with_component(Arc::new(Sink))
        .with_lag("c")
        .with_initial_values(HashMap::from([("c".to_string(), 0.0)]));
    builder.build().unwrap()
}

fn names(
    model: &Model,
    nodes: impl IntoIterator<Item = petgraph::graph::NodeIndex>,
) -> Vec<String> {
    nodes
        .into_iter()
        .filter(|&nx| nx != model.initial_node)
        .map(|nx| Model::extract_component_name(&model.components[nx]))
        .collect()
}

#[test]
fn order_matches_runtime_plan_where_bfs_differs() {
    let mut model = model();
    let reported: Vec<String> = model
        .coupling()
        .unwrap()
        .components
        .into_iter()
        .map(|c| c.name)
        .collect();

    let mut bfs = Bfs::new(&model.components, model.initial_node);
    let mut visited = Vec::new();
    while let Some(nx) = bfs.next(&model.components) {
        visited.push(nx);
    }
    assert_eq!(names(&model, visited), ["Source", "Sink", "Mid"]);

    model.step();
    let planned = names(&model, model.planned_order().unwrap());
    assert_eq!(reported, planned);
    assert_eq!(reported, ["Source", "Mid", "Sink"]);
}

#[test]
fn edges_flag_lags() {
    let coupling = model().coupling().unwrap();
    let edge = |producer: &str, consumer: &str, variable: &str, lagged| CouplingEdge {
        producer: producer.into(),
        consumer: consumer.into(),
        variable: variable.into(),
        lagged,
    };

    assert_eq!(
        coupling.edges,
        [
            edge("Sink", "Source", "c", true),
            edge("Source", "Mid", "a", false),
            edge("Source", "Sink", "a", false),
            edge("Mid", "Sink", "b", false),
        ]
    );
    assert_eq!(
        coupling.components[0].inputs[0].source,
        Some(VariableSource::Lagged)
    );
    assert!(coupling
        .to_string()
        .contains("Lagged Edges\n  Sink -> Source: c\n"));
}
