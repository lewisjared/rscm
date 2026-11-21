//! The step plan: every coupling decision for a model, made once when it is built.
//!
//! [`resolve`] decides, for each component in execution order, where each input and output is stored,
//! where it comes from and how it converts between the stored and declared unit and grid.
//! [`StepPlan::report`] projects the plan into the [`Coupling`] report.

use std::collections::{BTreeMap, HashMap};

use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};

use crate::component::{
    DeclaredVariable, GridType, TimeseriesType, VariableDefinition, VariableRole,
};
use crate::errors::{RSCMError, RSCMResult};
use crate::state::{lookup_order, ReadTransformInfo, VariableSource};
use crate::timeseries_collection::TimeseriesCollection;
use crate::units;

use super::coupling::{ComponentCoupling, Coupling, CouplingEdge};
use super::runtime::execution_order;
use super::types::{CGraph, RequiredTransformation, TransformDirection, UnitConversionInfo};

/// How a model steps: each component in execution order, with how it reads and writes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StepPlan {
    /// Components in the order the runtime steps them, without the graph's root node.
    pub steps: Vec<Step>,
}

/// One component's part of a step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub node: NodeIndex,
    pub label: String,
    /// One per input and state, aligned with the component's `input_names()`.
    pub reads: Vec<Read>,
    /// Positions in `reads`, sorted by [`lookup_order`] of their names.
    pub lookup_order: Vec<u32>,
    /// One per output and state, aligned with the component's `output_names()`.
    pub writes: Vec<Write>,
}

/// How a component reads one variable from storage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Read {
    /// The variable as the component declares it.
    pub definition: VariableDefinition,
    /// Position of the stored series in the model's collection.
    pub slot: usize,
    /// Label of the component that writes the variable, `None` if it is prescribed.
    pub producer: Option<String>,
    pub source: VariableSource,
    pub timeseries_type: TimeseriesType,
    pub stored_grid: GridType,
    /// The model's weights for `stored_grid`, `None` for the grid's defaults.
    pub weights: Option<Vec<f64>>,
    /// Stored unit to declared unit.
    pub unit: Option<UnitConversion>,
    /// Stored grid to declared grid.
    pub grid: Option<GridConversion>,
    /// Whether the read needs no conversion, timing or weights, so the runtime can skip its transform.
    pub plain: bool,
}

/// How a component's output is written to storage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Write {
    /// The variable as the component declares it.
    pub definition: VariableDefinition,
    /// Position of the stored series in the model's collection.
    pub slot: usize,
    pub timeseries_type: TimeseriesType,
    pub stored_grid: GridType,
    /// The model's weights for the declared grid when it aggregates, `None` for the grid's defaults.
    pub weights: Option<Vec<f64>>,
    /// Declared unit to stored unit.
    pub unit: Option<UnitConversion>,
    /// Declared grid to stored grid.
    pub grid: Option<GridConversion>,
}

/// A unit conversion, `converted = value * factor`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnitConversion {
    pub from: String,
    pub to: String,
    pub factor: f64,
}

/// An aggregation from a finer grid to a coarser one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridConversion {
    pub from: GridType,
    pub to: GridType,
}

/// The conversions one component applies when it reads or writes one variable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conversion {
    pub variable: String,
    /// The label of the component that reads or writes.
    pub component: String,
    pub direction: TransformDirection,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<UnitConversion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grid: Option<GridConversion>,
}

/// What the builder decided, for [`resolve`] to turn into a plan.
pub(crate) struct Decisions<'a> {
    pub graph: &'a CGraph,
    pub initial_node: NodeIndex,
    pub labels: &'a BTreeMap<NodeIndex, String>,
    pub collection: &'a TimeseriesCollection,
    /// Source of each input, keyed by variable and consumer label.
    pub sources: &'a HashMap<(String, String), VariableSource>,
    /// Label of the component or aggregate producing each modelled variable.
    pub producers: &'a HashMap<String, String>,
    pub read_conversions: &'a [UnitConversionInfo],
    /// Grid aggregations, each with the label of the component that applies it.
    pub transformations: &'a [(String, RequiredTransformation)],
    pub timeseries_types: &'a BTreeMap<String, TimeseriesType>,
    pub grid_weights: &'a HashMap<GridType, Vec<f64>>,
}

/// Make every step coupling decision for a built graph and its storage.
pub(crate) fn resolve(decisions: &Decisions) -> RSCMResult<StepPlan> {
    let order = execution_order(decisions.graph).map_err(|node| RSCMError::FeedbackLoop {
        components: vec![decisions.labels[&node].clone()],
        variables: vec![],
        suggestions: vec![],
    })?;
    let steps = order
        .into_iter()
        .filter(|&node| node != decisions.initial_node)
        .map(|node| resolve_step(decisions, node))
        .collect::<RSCMResult<_>>()?;
    Ok(StepPlan { steps })
}

fn resolve_step(d: &Decisions, node: NodeIndex) -> RSCMResult<Step> {
    let component = &d.graph[node];
    let label = d.labels[&node].clone();
    let grid_conversion = |variable: &str, direction| {
        d.transformations
            .iter()
            .find(|(component, t)| {
                component == &label && t.variable == variable && t.direction == direction
            })
            .map(|(_, t)| GridConversion {
                from: t.source_grid,
                to: t.target_grid,
            })
    };

    let mut reads = Vec::new();
    for definition in component.inputs() {
        let (slot, stored) = stored(d.collection, &definition.name)?;
        let source = d
            .sources
            .get(&(definition.name.clone(), label.clone()))
            .copied()
            .unwrap_or_default();
        let timeseries_type = timeseries_type(d, &definition.name);
        let stored_grid = stored.grid();
        let unit = d
            .read_conversions
            .iter()
            .find(|c| c.component == label && c.variable == definition.name)
            .map(|c| UnitConversion {
                from: c.source_unit.clone(),
                to: c.target_unit.clone(),
                factor: c.factor,
            });
        let factor = unit.as_ref().map_or(1.0, |u| u.factor);
        // A grid input always carries the model's weights, so a global read of it uses them.
        let plain = stored_grid == GridType::Scalar
            && (factor - 1.0).abs() <= f64::EPSILON
            && source == VariableSource::Prescribed
            && timeseries_type == TimeseriesType::Point;
        reads.push(Read {
            producer: d.producers.get(&definition.name).cloned(),
            grid: grid_conversion(&definition.name, TransformDirection::Read),
            weights: d.grid_weights.get(&stored_grid).cloned(),
            definition,
            slot,
            source,
            timeseries_type,
            stored_grid,
            unit,
            plain,
        });
    }
    let mut lookup: Vec<u32> = (0..reads.len() as u32).collect();
    lookup.sort_by(|&a, &b| {
        lookup_order(
            &reads[a as usize].definition.name,
            &reads[b as usize].definition.name,
        )
    });

    let mut writes = Vec::new();
    for definition in component.outputs() {
        let (slot, stored) = stored(d.collection, &definition.name)?;
        let stored_unit = stored.unit();
        let unit = units::factor(&definition.name, &definition.unit, stored_unit)?.map(|factor| {
            UnitConversion {
                from: definition.unit.clone(),
                to: stored_unit.to_string(),
                factor,
            }
        });
        let grid = grid_conversion(&definition.name, TransformDirection::Write);
        writes.push(Write {
            slot,
            timeseries_type: timeseries_type(d, &definition.name),
            stored_grid: stored.grid(),
            weights: grid.and_then(|g| d.grid_weights.get(&g.from).cloned()),
            unit,
            grid,
            definition,
        });
    }

    Ok(Step {
        node,
        label,
        reads,
        lookup_order: lookup,
        writes,
    })
}

fn stored<'a>(
    collection: &'a TimeseriesCollection,
    name: &str,
) -> RSCMResult<(usize, &'a crate::timeseries_collection::TimeseriesData)> {
    let missing = || RSCMError::Error(format!("No timeseries with variable='{name}'"));
    let slot = collection.position(name).ok_or_else(missing)?;
    Ok((slot, &collection.items()[slot].data))
}

fn timeseries_type(d: &Decisions, name: &str) -> TimeseriesType {
    d.timeseries_types
        .get(name)
        .copied()
        .unwrap_or(TimeseriesType::Point)
}

impl Read {
    /// How the runtime reads this input, or `None` for a plain read.
    pub fn transform_info(&self) -> Option<ReadTransformInfo> {
        (!self.plain).then(|| {
            ReadTransformInfo::new(self.stored_grid, self.weights.clone())
                .with_conversion_factor(self.unit.as_ref().map_or(1.0, |u| u.factor))
                .with_source(self.source)
                .with_timeseries_type(self.timeseries_type)
        })
    }
}

impl StepPlan {
    /// Report the coupling: execution order, edges, each input's source and every conversion.
    pub fn report(&self) -> Coupling {
        let mut components = Vec::new();
        let mut edges = Vec::new();
        let mut conversions = Vec::new();
        for (order, step) in self.steps.iter().enumerate() {
            let mut step_edges: Vec<CouplingEdge> = step
                .reads
                .iter()
                .filter(|r| matches!(r.source, VariableSource::Upstream | VariableSource::Lagged))
                .filter_map(|r| {
                    let producer = r.producer.as_ref()?;
                    Some(CouplingEdge {
                        producer: producer.clone(),
                        consumer: step.label.clone(),
                        variable: r.definition.name.clone(),
                        lagged: r.source == VariableSource::Lagged,
                    })
                })
                .collect();
            step_edges.sort_by(|a, b| (&a.variable, &a.producer).cmp(&(&b.variable, &b.producer)));
            edges.extend(step_edges);

            let declared = |definition: &VariableDefinition, timeseries_type, source| {
                let mut info = DeclaredVariable::from_definition(definition);
                info.timeseries_type = timeseries_type;
                info.source = source;
                info
            };
            let mut inputs = Vec::new();
            let mut states = Vec::new();
            for read in &step.reads {
                let info = declared(&read.definition, read.timeseries_type, Some(read.source));
                match read.definition.role {
                    VariableRole::State => states.push(info),
                    _ => inputs.push(info),
                }
                if read.unit.is_some() || read.grid.is_some() {
                    conversions.push(Conversion {
                        variable: read.definition.name.clone(),
                        component: step.label.clone(),
                        direction: TransformDirection::Read,
                        unit: read.unit.clone(),
                        grid: read.grid,
                    });
                }
            }
            let mut outputs = Vec::new();
            for write in &step.writes {
                if write.definition.role == VariableRole::Output {
                    outputs.push(declared(&write.definition, write.timeseries_type, None));
                }
                if write.unit.is_some() || write.grid.is_some() {
                    conversions.push(Conversion {
                        variable: write.definition.name.clone(),
                        component: step.label.clone(),
                        direction: TransformDirection::Write,
                        unit: write.unit.clone(),
                        grid: write.grid,
                    });
                }
            }
            components.push(ComponentCoupling {
                order,
                name: step.label.clone(),
                inputs,
                outputs,
                states,
            });
        }
        conversions.sort_by(|a, b| {
            (&a.variable, &a.component, &a.direction).cmp(&(
                &b.variable,
                &b.component,
                &b.direction,
            ))
        });
        Coupling {
            components,
            edges,
            conversions,
        }
    }
}
