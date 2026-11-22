//! Model struct and runtime execution.

use crate::component::{ComponentState, GridType, TimeseriesType, VariableDefinition};
use crate::errors::{RSCMError, RSCMResult};
use crate::state::{
    FourBoxSlice, HemisphericSlice, InputPlan, InputState, OutputBuffer, StateValue, VariableSource,
};
use crate::timeseries::{FloatValue, Time, TimeAxis};
use crate::timeseries_collection::TimeseriesCollection;
use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeFiltered, EdgeRef};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use super::plan::{resolve, Decisions, Step, StepPlan, Write};
use super::transformations::aggregate_state_value;
use super::types::{CGraph, RequiredTransformation, TransformDirection, UnitConversionInfo};

/// Custom serialisation for HashMap<(String, String), T> to work with JSON.
/// Serialises as a Vec of (key1, key2, value) tuples.
mod tuple_map_serde {
    use super::*;

    pub fn serialize<S, T>(
        map: &HashMap<(String, String), T>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize,
    {
        let vec: Vec<(&str, &str, &T)> = map
            .iter()
            .map(|((k1, k2), v)| (k1.as_str(), k2.as_str(), v))
            .collect();
        vec.serialize(serializer)
    }

    pub fn deserialize<'de, D, T>(deserializer: D) -> Result<HashMap<(String, String), T>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        let vec: Vec<(String, String, T)> = Vec::deserialize(deserializer)?;
        Ok(vec.into_iter().map(|(k1, k2, v)| ((k1, k2), v)).collect())
    }
}

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
#[derive(Debug, Serialize, Deserialize)]
pub struct Model {
    /// A directed graph with components as nodes and the edges defining the state dependencies
    /// between nodes.
    /// This graph is traversed on every step to ensure that any state dependencies are
    /// solved before another component needs the state.
    pub(crate) components: CGraph,
    /// The base node of the graph from where to begin traversing.
    pub(crate) initial_node: NodeIndex,
    /// The model state.
    ///
    /// Variable names within the model are unique and these variable names are used by
    /// components to request state.
    collection: TimeseriesCollection,
    time_axis: Arc<TimeAxis>,
    time_index: usize,
    /// Custom weights for grid aggregation, keyed by grid.
    ///
    /// Used for grid transformations during model execution.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    grid_weights: HashMap<GridType, Vec<f64>>,
    /// Read-side transformations: variable name -> transformation needed when component reads.
    ///
    /// When a component reads a variable at a coarser grid than the schema declares,
    /// this maps the variable name to the transformation needed (e.g., FourBox -> Scalar).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    read_transforms: HashMap<String, RequiredTransformation>,
    /// Write-side transformations: variable name -> transformation needed when component writes.
    ///
    /// When a component writes a variable at a finer grid than the schema declares,
    /// this maps the variable name to the transformation needed (e.g., FourBox -> Scalar).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    write_transforms: HashMap<String, RequiredTransformation>,
    /// Unit conversions needed at runtime.
    ///
    /// Key: (variable_name, component_name) -> conversion factor
    /// The factor converts from the stored unit to the component's unit when reading,
    /// and its inverse converts a producer's output into the stored unit when writing.
    #[serde(
        default,
        skip_serializing_if = "HashMap::is_empty",
        with = "tuple_map_serde"
    )]
    unit_conversions: HashMap<(String, String), f64>,
    /// Variable source types for get() behaviour.
    /// Key: (variable_name, component_name) -> source type.
    /// This matches the unit_conversions pattern since the same variable
    /// can have different sources for different components.
    #[serde(
        default,
        skip_serializing_if = "HashMap::is_empty",
        with = "tuple_map_serde"
    )]
    variable_sources: HashMap<(String, String), VariableSource>,
    /// Whether each variable is a point or an average.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) timeseries_types: BTreeMap<String, TimeseriesType>,
    /// Label of each component node, set by the builder.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) labels: BTreeMap<NodeIndex, String>,
    /// Internal state for each component, keyed by node index.
    /// Uses Box<dyn ComponentState> with typetag for serialisation.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) component_states: BTreeMap<NodeIndex, Box<dyn ComponentState>>,
    /// Every coupling decision, resolved by the builder.
    #[serde(default)]
    pub(crate) step_plan: StepPlan,
    /// Scratch for stepping each component of the plan, made from it on the first step.
    #[serde(skip)]
    buffers: Option<Vec<StepBuffers>>,
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

fn transform(
    def: &VariableDefinition,
    source_grid: GridType,
    target_grid: GridType,
    direction: TransformDirection,
) -> RequiredTransformation {
    RequiredTransformation {
        variable: def.name.clone(),
        unit: def.unit.clone(),
        source_grid,
        target_grid,
        direction,
    }
}

impl Model {
    /// Create a new Model with the given components and collection.
    pub fn new(
        components: CGraph,
        initial_node: NodeIndex,
        collection: TimeseriesCollection,
        time_axis: Arc<TimeAxis>,
    ) -> Self {
        Self::with_grid_weights(
            components,
            initial_node,
            collection,
            time_axis,
            HashMap::new(),
        )
    }

    /// Create a new Model with custom grid weights.
    ///
    /// The grid_weights are used for grid transformations during model execution.
    pub fn with_grid_weights(
        components: CGraph,
        initial_node: NodeIndex,
        collection: TimeseriesCollection,
        time_axis: Arc<TimeAxis>,
        grid_weights: HashMap<GridType, Vec<f64>>,
    ) -> Self {
        Self::with_transforms(
            components,
            initial_node,
            collection,
            time_axis,
            grid_weights,
            HashMap::new(),
            HashMap::new(),
        )
    }

    /// Create a new Model with grid weights and transformations.
    ///
    /// This is the full constructor that includes both grid weights and the
    /// read/write transformations for automatic grid aggregation.
    pub fn with_transforms(
        components: CGraph,
        initial_node: NodeIndex,
        collection: TimeseriesCollection,
        time_axis: Arc<TimeAxis>,
        grid_weights: HashMap<GridType, Vec<f64>>,
        read_transforms: HashMap<String, RequiredTransformation>,
        write_transforms: HashMap<String, RequiredTransformation>,
    ) -> Self {
        Self::with_unit_conversions(
            components,
            initial_node,
            collection,
            time_axis,
            grid_weights,
            read_transforms,
            write_transforms,
            HashMap::new(),
        )
    }

    /// Create a new Model with all transformations including unit conversions.
    ///
    /// This is the complete constructor that includes grid weights, read/write
    /// transformations, and unit conversions for automatic unit translation at runtime.
    ///
    /// # Arguments
    ///
    /// * `unit_conversions` - Map from (variable_name, component_name) to conversion factor.
    ///   The factor is multiplied with the stored value to convert from schema units to
    ///   component units when reading.
    #[allow(clippy::too_many_arguments)]
    pub fn with_unit_conversions(
        components: CGraph,
        initial_node: NodeIndex,
        collection: TimeseriesCollection,
        time_axis: Arc<TimeAxis>,
        grid_weights: HashMap<GridType, Vec<f64>>,
        read_transforms: HashMap<String, RequiredTransformation>,
        write_transforms: HashMap<String, RequiredTransformation>,
        unit_conversions: HashMap<(String, String), f64>,
    ) -> Self {
        Self::with_variable_sources(
            components,
            initial_node,
            collection,
            time_axis,
            grid_weights,
            read_transforms,
            write_transforms,
            unit_conversions,
            HashMap::new(),
        )
    }

    /// Create a new Model with all transformations including unit conversions and variable sources.
    ///
    /// This is the most complete constructor that includes grid weights, read/write
    /// transformations, unit conversions, and variable source information.
    ///
    /// # Arguments
    ///
    /// * `unit_conversions` - Map from (variable_name, component_name) to conversion factor.
    ///   The factor is multiplied with the stored value to convert from schema units to
    ///   component units when reading.
    /// * `variable_sources` - Map from (variable_name, component_name) to source type.
    ///   Used by TimeseriesWindow to pick the index `get()` reads.
    #[allow(clippy::too_many_arguments)]
    pub fn with_variable_sources(
        components: CGraph,
        initial_node: NodeIndex,
        collection: TimeseriesCollection,
        time_axis: Arc<TimeAxis>,
        grid_weights: HashMap<GridType, Vec<f64>>,
        read_transforms: HashMap<String, RequiredTransformation>,
        write_transforms: HashMap<String, RequiredTransformation>,
        unit_conversions: HashMap<(String, String), f64>,
        variable_sources: HashMap<(String, String), VariableSource>,
    ) -> Self {
        Self {
            components,
            initial_node,
            collection,
            time_axis,
            time_index: 0,
            grid_weights,
            read_transforms,
            write_transforms,
            unit_conversions,
            variable_sources,
            timeseries_types: BTreeMap::new(),
            labels: BTreeMap::new(),
            component_states: BTreeMap::new(),
            step_plan: StepPlan::default(),
            buffers: None,
        }
    }

    /// Get a reference to a component's internal state.
    pub fn get_component_state(&self, node_index: NodeIndex) -> Option<&dyn ComponentState> {
        self.component_states.get(&node_index).map(|b| b.as_ref())
    }

    /// Create a checkpoint of the model state that can be serialised.
    ///
    /// The checkpoint includes:
    /// - All timeseries data (including current state)
    /// - Component internal states (flux_history, etc.)
    /// - Current time index
    ///
    /// Use this to save model state mid-run for later resumption.
    pub fn checkpoint(&self) -> RSCMResult<String> {
        serde_json::to_string(self)
            .map_err(|e| RSCMError::Error(format!("Serialisation failed: {}", e)))
    }

    /// Restore a model from a JSON checkpoint string.
    ///
    /// This recreates the exact state of the model at checkpoint time,
    /// including component internal states.
    pub fn from_checkpoint(json: &str) -> RSCMResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| RSCMError::Error(format!("Deserialisation failed: {}", e)))
    }

    /// Get the current time index (useful for checkpointing).
    pub fn time_index(&self) -> usize {
        self.time_index
    }

    /// Get the configured grid weights.
    ///
    /// Returns the custom weights if configured, or None if using defaults.
    pub fn get_grid_weights(&self, grid: GridType) -> Option<&Vec<f64>> {
        self.grid_weights.get(&grid)
    }

    /// Get all required transformations for introspection.
    ///
    /// Returns a vector of all required transformations, both read-side and write-side.
    /// This is useful for debugging and understanding what grid aggregations will occur.
    pub fn required_transformations(&self) -> Vec<&RequiredTransformation> {
        self.read_transforms
            .values()
            .chain(self.write_transforms.values())
            .collect()
    }

    /// Get read-side transformations.
    ///
    /// These transformations aggregate data when components read variables at coarser
    /// resolutions than the schema declares.
    pub fn read_transforms(&self) -> &HashMap<String, RequiredTransformation> {
        &self.read_transforms
    }

    /// Get write-side transformations.
    ///
    /// These transformations aggregate data when components write variables at finer
    /// resolutions than the schema declares.
    pub fn write_transforms(&self) -> &HashMap<String, RequiredTransformation> {
        &self.write_transforms
    }

    /// Get unit conversions.
    ///
    /// Returns the unit conversion factors keyed by (variable_name, component_name).
    /// Each factor converts from schema units to component units when reading.
    pub fn unit_conversions(&self) -> &HashMap<(String, String), f64> {
        &self.unit_conversions
    }

    /// Get variable sources.
    ///
    /// Returns the variable source types keyed by (variable_name, component_name).
    /// This enables different components to have different source classifications for the same variable.
    /// For example, a state variable is `OwnState` for the producer component but `Upstream` for consumers.
    pub fn variable_sources(&self) -> &HashMap<(String, String), VariableSource> {
        &self.variable_sources
    }

    /// Whether `variable` is a point or an average, or `None` if the model has no such variable.
    pub fn timeseries_type(&self, variable: &str) -> Option<TimeseriesType> {
        self.timeseries_types.get(variable).copied()
    }

    /// The label of the component at `node`, which keys its coupling.
    ///
    /// A model made without the builder falls back to the component's type name.
    pub fn component_label(&self, node: NodeIndex) -> String {
        self.labels
            .get(&node)
            .cloned()
            .unwrap_or_else(|| Self::extract_component_name(&self.components[node]))
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

    /// Grid of the stored series `name`, or `Scalar` if there is none.
    fn stored_grid(&self, name: &str) -> GridType {
        self.collection
            .get_by_name(name)
            .map_or(GridType::Scalar, |item| item.data.grid())
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
    ///
    /// Empty for a model made without the builder or restored from a save that has no plan.
    pub fn plan(&self) -> &StepPlan {
        &self.step_plan
    }

    /// The plan the builder resolved, or for a model without one, the plan its coupling maps give.
    pub(crate) fn resolved_plan(&self) -> RSCMResult<std::borrow::Cow<'_, StepPlan>> {
        if !self.step_plan.steps.is_empty() {
            return Ok(std::borrow::Cow::Borrowed(&self.step_plan));
        }
        let labels: BTreeMap<NodeIndex, String> = self
            .components
            .node_indices()
            .map(|node| (node, self.component_label(node)))
            .collect();
        let mut producers = HashMap::new();
        let mut read_conversions = Vec::new();
        let mut transformations = Vec::new();
        for (&node, label) in &labels {
            let component = &self.components[node];
            for def in component.outputs() {
                producers.insert(def.name.clone(), label.clone());
                let stored_grid = self.stored_grid(&def.name);
                if self.write_transforms.contains_key(&def.name) && def.grid != stored_grid {
                    transformations.push((
                        label.clone(),
                        transform(&def, def.grid, stored_grid, TransformDirection::Write),
                    ));
                }
            }
            for def in component.inputs() {
                let key = (def.name.clone(), label.clone());
                if let (Some(&factor), Some(stored)) = (
                    self.unit_conversions.get(&key),
                    self.collection.get_data(&def.name),
                ) {
                    read_conversions.push(UnitConversionInfo {
                        variable: def.name.clone(),
                        component: label.clone(),
                        factor,
                        source_unit: stored.unit().to_string(),
                        target_unit: def.unit.clone(),
                    });
                }
                let stored_grid = self.stored_grid(&def.name);
                if self.read_transforms.contains_key(&def.name) && def.grid != stored_grid {
                    transformations.push((
                        label.clone(),
                        transform(&def, stored_grid, def.grid, TransformDirection::Read),
                    ));
                }
            }
        }
        resolve(&Decisions {
            graph: &self.components,
            initial_node: self.initial_node,
            labels: &labels,
            collection: &self.collection,
            sources: &self.variable_sources,
            producers: &producers,
            read_conversions: &read_conversions,
            transformations: &transformations,
            timeseries_types: &self.timeseries_types,
            grid_weights: &self.grid_weights,
        })
        .map(std::borrow::Cow::Owned)
    }

    /// Step the model forward a step by solving each component for the current step.
    fn step_model(&mut self) -> RSCMResult<()> {
        if self.step_plan.steps.is_empty() {
            self.step_plan = self.resolved_plan()?.into_owned();
        }
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
            &|_, er| format!("label = {:?}", er.weight().name()),
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
