//! Model struct and runtime execution.

use crate::component::{ComponentState, GridType, TimeseriesType};
use crate::errors::{RSCMError, RSCMResult};
use crate::state::{
    FourBoxSlice, HemisphericSlice, InputPlan, InputState, OutputBuffer, StateValue,
};
use crate::timeseries::{FloatValue, Time, TimeAxis};
use crate::timeseries_collection::TimeseriesCollection;
use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeFiltered, EdgeRef};
use std::collections::BTreeMap;
use std::sync::Arc;

use super::builder::ModelBuilder;
use super::plan::{Step, StepPlan, Write};
use super::transformations::aggregate_state_value;
use super::types::CGraph;

/// A coupled set of components that are solved on a common time axis.
///
/// These components are solved over steps defined by the [`TimeAxis`].
/// Components may pass state between themselves.
/// Each component may require information from other components to be solved (modelled) or
/// predefined data (prescribed).
///
/// For example, a component to calculate the Effective Radiative Forcing(ERF) of CO_2 may
/// require CO_2 concentrations as input state and provide CO_2 ERF.
/// The component is agnostic about where/how that state is defined.
/// If the model has no components which provide CO_2 concentrations,
/// then a CO_2 concentration timeseries must be prescribed.
/// If the model also contains a carbon cycle component which produced CO_2 concentrations,
/// then the ERF component will be solved after the carbon cycle model.
#[derive(Debug)]
pub struct Model {
    /// Components as nodes, with an edge for each variable one passes to another.
    pub(crate) components: CGraph,
    /// The model state.
    ///
    /// Variable names within the model are unique and these variable names are used by
    /// components to request state.
    pub(crate) collection: TimeseriesCollection,
    time_axis: Arc<TimeAxis>,
    pub(crate) time_index: usize,
    /// Internal state for each component, keyed by node index.
    /// Uses Box<dyn ComponentState> with typetag for serialisation.
    pub(crate) component_states: BTreeMap<NodeIndex, Box<dyn ComponentState>>,
    /// Every coupling decision, resolved by the builder.
    pub(crate) step_plan: StepPlan,
    /// Scratch for stepping each component of the plan, made from it on the first step.
    buffers: Option<Vec<StepBuffers>>,
    /// The builder this model was built from, which a [`super::Record`] keeps as its spec.
    pub(crate) builder: Arc<ModelBuilder>,
}

/// Buffers for stepping one component, aligned with its [`Step`].
#[derive(Debug)]
struct StepBuffers {
    inputs: InputPlan,
    outputs: OutputBuffer,
    /// Converted outputs of the current step, written only once all of them are valid.
    staged: Vec<StagedOutput>,
}

impl StepBuffers {
    fn new(step: &Step) -> Self {
        Self {
            inputs: InputPlan::from_step(step),
            outputs: OutputBuffer::new(
                step.writes
                    .iter()
                    .map(|write| write.definition.name.clone())
                    .collect(),
            ),
            staged: Vec::new(),
        }
    }
}

/// An output converted to its stored unit and grid, with the position it came from.
#[derive(Debug)]
struct StagedOutput {
    position: usize,
    index: usize,
    at: usize,
    value: StateValue,
}

/// `value` multiplied by `factor` in every region.
fn scale_state_value(value: &StateValue, factor: FloatValue) -> StateValue {
    match value {
        StateValue::Scalar(v) => StateValue::Scalar(v * factor),
        StateValue::FourBox(slice) => StateValue::FourBox(FourBoxSlice::from_array(
            slice.as_array().map(|v| v * factor),
        )),
        StateValue::Hemispheric(slice) => StateValue::Hemispheric(HemisphericSlice::from_array(
            slice.as_array().map(|v| v * factor),
        )),
    }
}

/// The order components run in within a step, or a node on a cycle.
///
/// Self-edges describe a component's own state and do not constrain the order.
pub(crate) fn execution_order(graph: &CGraph) -> Result<Vec<NodeIndex>, NodeIndex> {
    let graph = EdgeFiltered::from_fn(graph, |edge| edge.source() != edge.target());
    petgraph::algo::toposort(&graph, None).map_err(|cycle| cycle.node_id())
}

impl Model {
    /// A model at its first step, stepping `step_plan` with a fresh state for each component.
    pub(crate) fn from_plan(
        components: CGraph,
        collection: TimeseriesCollection,
        time_axis: Arc<TimeAxis>,
        step_plan: StepPlan,
        builder: Arc<ModelBuilder>,
    ) -> Self {
        let component_states = components
            .node_indices()
            .map(|node| (node, components[node].create_initial_state()))
            .collect();
        Self {
            components,
            collection,
            time_axis,
            time_index: 0,
            component_states,
            step_plan,
            buffers: None,
            builder,
        }
    }

    /// Get a reference to a component's internal state.
    pub fn get_component_state(&self, node_index: NodeIndex) -> Option<&dyn ComponentState> {
        self.component_states.get(&node_index).map(|b| b.as_ref())
    }

    /// The number of steps on the time axis.
    pub(crate) fn timeseries_len(&self) -> usize {
        self.time_axis.len()
    }

    /// Get the current time index (useful for checkpointing).
    pub fn time_index(&self) -> usize {
        self.time_index
    }

    /// Get the configured grid weights.
    ///
    /// Returns the custom weights if configured, or None if using defaults.
    pub fn get_grid_weights(&self, grid: GridType) -> Option<&Vec<f64>> {
        self.builder.grid_weights(grid)
    }

    /// Whether `variable` is a point or an average, or `None` if the model has no such variable.
    pub fn timeseries_type(&self, variable: &str) -> Option<TimeseriesType> {
        self.step_plan.timeseries_types.get(variable).copied()
    }

    /// The type name of a component, its default label.
    ///
    /// Extracts the type name (before the first '{', ' ', or '(') from the Debug output.
    pub(crate) fn extract_component_name(component: &(impl std::fmt::Debug + ?Sized)) -> String {
        /// Keeps output up to the first delimiter, then stops the formatting.
        struct TypeName(String);
        impl std::fmt::Write for TypeName {
            fn write_str(&mut self, s: &str) -> std::fmt::Result {
                match s.find(['{', ' ', '(']) {
                    Some(end) => {
                        self.0.push_str(&s[..end]);
                        Err(std::fmt::Error)
                    }
                    None => {
                        self.0.push_str(s);
                        Ok(())
                    }
                }
            }
        }
        let mut name = TypeName(String::new());
        // The error only signals that the name is complete
        let _ = std::fmt::Write::write_fmt(&mut name, format_args!("{component:?}"));
        name.0
    }

    /// Gets the time value at the current step.
    pub fn current_time(&self) -> Time {
        self.time_axis.at(self.time_index).unwrap()
    }

    /// Gets the time bounds at the current step.
    pub fn current_time_bounds(&self) -> (Time, Time) {
        self.time_axis.at_bounds(self.time_index).unwrap()
    }

    /// Solve a single component for the current step.
    ///
    /// The updated state from the component is then pushed into the model's timeseries collection
    /// to be later used by other components.
    /// A point output is stored at the step's end and an average at its start.
    fn step_model_component(&mut self, step: &Step, buffers: &mut StepBuffers) -> RSCMResult<()> {
        let component = &self.components[step.node];
        let component_name = &step.label;
        let input_state = InputState::planned(
            self.collection.items(),
            &buffers.inputs,
            self.current_time(),
        )
        .with_time_index(self.time_index);

        let (start, end) = self.current_time_bounds();

        // Get component state - must exist since we initialise all states in build()
        let internal_state = self
            .component_states
            .get_mut(&step.node)
            .expect("Component state not initialised, a bug in Model::build()");

        buffers.outputs.clear();
        component
            .solve_into(
                start,
                end,
                &input_state,
                internal_state.as_mut(),
                &mut buffers.outputs.writer(),
            )
            .map_err(|err| match err {
                RSCMError::Interrupt(_) => err,
                err => RSCMError::Error(format!(
                    "{component_name} failed during {start}..{end}: {err}"
                )),
            })?;

        let undeclared = buffers.outputs.take_undeclared();
        if !undeclared.is_empty() {
            let names: Vec<&str> = undeclared.iter().map(|(name, _)| name.as_str()).collect();
            return Err(RSCMError::Error(format!(
                "{component_name} returned undeclared output {} during {start}..{end}",
                names.join(", ")
            )));
        }

        buffers.staged.clear();
        for (position, write) in step.writes.iter().enumerate() {
            if let Some(value) = buffers.outputs.take(position) {
                let staged = self.stage_output(component_name, position, write, value)?;
                buffers.staged.push(staged);
            }
        }
        for staged in buffers.staged.drain(..) {
            let key = &step.writes[staged.position].definition.name;
            let data = self.collection.data_mut(staged.index);
            match &staged.value {
                StateValue::Scalar(v) => data.set_scalar(key, staged.at, *v),
                StateValue::FourBox(slice) => data.set_four_box(key, staged.at, slice),
                StateValue::Hemispheric(slice) => data.set_hemispheric(key, staged.at, slice),
            }
            .expect("grid checked while staging");
        }
        Ok(())
    }

    fn grid_mismatch(
        &self,
        component_name: &str,
        key: &str,
        expected: GridType,
        actual: GridType,
    ) -> RSCMError {
        let (start, end) = self.current_time_bounds();
        let err = RSCMError::GridOutputMismatch {
            variable: key.to_string(),
            expected_grid: format!("{expected:?}"),
            component_grid: format!("{actual:?}"),
        };
        RSCMError::Error(format!(
            "{component_name} could not write {key} during {start}..{end}: {err}"
        ))
    }

    /// A component's output converted to the unit and grid it is stored in.
    fn stage_output(
        &self,
        component_name: &str,
        position: usize,
        write: &Write,
        value: StateValue,
    ) -> RSCMResult<StagedOutput> {
        let key = &write.definition.name;
        let expected_grid = write.grid.map_or(write.stored_grid, |g| g.from);
        if value.grid() != expected_grid {
            return Err(self.grid_mismatch(component_name, key, expected_grid, value.grid()));
        }
        let value = match &write.unit {
            Some(unit) if (unit.factor - 1.0).abs() > FloatValue::EPSILON => {
                scale_state_value(&value, unit.factor)
            }
            _ => value,
        };
        let value = match write.grid {
            Some(grid) => aggregate_state_value(&value, grid.from, grid.to, write.weights.as_ref())
                .map_err(|err| {
                    let (start, end) = self.current_time_bounds();
                    RSCMError::Error(format!(
                        "{component_name} could not aggregate {key} during {start}..{end}: {err}"
                    ))
                })?,
            None => value,
        };
        if value.grid() != write.stored_grid {
            return Err(self.grid_mismatch(component_name, key, write.stored_grid, value.grid()));
        }
        let at = match write.timeseries_type {
            TimeseriesType::Point => self.time_index + 1,
            TimeseriesType::Average => self.time_index,
        };
        Ok(StagedOutput {
            position,
            index: write.slot,
            at,
            value,
        })
    }

    /// Every coupling decision the builder made, in execution order.
    pub fn plan(&self) -> &StepPlan {
        &self.step_plan
    }

    /// Step the model forward a step by solving each component for the current step.
    fn step_model(&mut self) -> RSCMResult<()> {
        let plan = std::mem::take(&mut self.step_plan);
        let mut buffers = self
            .buffers
            .take()
            .unwrap_or_else(|| plan.steps.iter().map(StepBuffers::new).collect());
        let result = plan
            .steps
            .iter()
            .zip(&mut buffers)
            .try_for_each(|(step, buffers)| self.step_model_component(step, buffers));
        self.step_plan = plan;
        self.buffers = Some(buffers);
        result
    }

    /// Steps the model forward one step.
    ///
    /// This solves the current step and then updates the index.
    pub fn step(&mut self) {
        self.try_step().expect("Model step failed");
    }

    /// Fallible step execution. On failure the time index is not advanced.
    /// Components already executed may have updated their state; rebuild the
    /// model before retrying a failed run.
    pub fn try_step(&mut self) -> RSCMResult<()> {
        if self.time_index + 1 >= self.time_axis.len() {
            return Err(RSCMError::Error(
                "Model is already at its final time bound".into(),
            ));
        }
        self.step_model()?;
        self.time_index += 1;
        Ok(())
    }

    /// Steps the model until the end of the time axis, panicking on failure.
    /// Use `try_run` to handle component errors.
    pub fn run(&mut self) {
        self.try_run().expect("Model run failed");
    }

    /// Run to completion, returning the first component or output error.
    /// As with `try_step`, discard the model after a failed run.
    pub fn try_run(&mut self) -> RSCMResult<()> {
        while self.time_index + 1 < self.time_axis.len() {
            self.try_step()?;
        }
        Ok(())
    }

    /// Create a diagram that represents the component graph.
    ///
    /// Useful for debugging.
    pub fn as_dot(&self) -> Dot<'_, &CGraph> {
        Dot::with_attr_getters(
            &self.components,
            &[Config::NodeNoLabel, Config::EdgeNoLabel],
            &|_, er| format!("label = {:?}", er.weight().name),
            &|_, (_, component)| {
                // Escape quotes and backslashes for DOT format
                let debug_str = format!("{:?}", component);
                let escaped = debug_str.replace('\\', "\\\\").replace('"', "\\\"");
                format!("label = \"{}\"", escaped)
            },
        )
    }

    /// Returns true if the model has no more steps to process.
    pub fn finished(&self) -> bool {
        self.time_index == self.time_axis.len() - 1
    }

    /// Returns a reference to the timeseries collection.
    pub fn timeseries(&self) -> &TimeseriesCollection {
        &self.collection
    }
}
