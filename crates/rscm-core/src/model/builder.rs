//! Model builder for constructing models from components.

use crate::component::{
    reads_and_writes, Component, GridType, TimeseriesType, VariableDefinition, VariableRole,
};
use crate::errors::{close_matches, RSCMError, RSCMResult};
use crate::interpolate::strategies::{
    InterpolationStrategy, LinearSplineStrategy, PreviousStrategy,
};
use crate::schema::VariableSchema;
use crate::spatial::SpatialGrid;
use crate::state::VariableSource;
use crate::timeseries::{FloatValue, GridTimeseries, TimeAxis, Timeseries};
use crate::timeseries_collection::{TimeseriesCollection, TimeseriesData};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Graph;
use pyo3::pyclass;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use super::plan::{resolve, Decisions};
use super::runtime::Model;
use super::storage::{resolve_storage, Storage};
use super::types::{CGraph, ResolvedVariable, C};
use super::validation::resolve_timeseries_types;

/// An input that no component or schema aggregate produces, so the model must be given it.
#[pyclass(frozen, get_all)]
#[derive(Debug, Clone, PartialEq)]
pub struct PrescribedInput {
    /// The variable as its first consumer, by label, declares it.
    pub definition: VariableDefinition,
    /// The labels of the components that read it, sorted.
    pub consumers: Vec<String>,
    /// Whether prescribed data or an initial value supplies it.
    pub supplied: bool,
}

/// A variable's value at the first step, the same in every region or one per region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RegionValues {
    Uniform(FloatValue),
    PerRegion(Vec<FloatValue>),
}

/// An initial value and, when not given in the stored unit, its unit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct InitialValue {
    values: RegionValues,
    unit: Option<String>,
}

impl InitialValue {
    /// One value per region of `grid`, converted into `storage_unit`.
    fn resolve(
        &self,
        name: &str,
        storage_unit: &str,
        grid: GridType,
    ) -> RSCMResult<Vec<FloatValue>> {
        let factor = match &self.unit {
            Some(unit) => crate::units::factor(name, unit, storage_unit)?.unwrap_or(1.0),
            None => 1.0,
        };
        let expected_len = match grid {
            GridType::Scalar => 1,
            GridType::Hemispheric => 2,
            GridType::FourBox => 4,
        };
        match &self.values {
            RegionValues::Uniform(value) => Ok(vec![value * factor; expected_len]),
            RegionValues::PerRegion(values) if values.len() == expected_len => {
                Ok(values.iter().map(|v| v * factor).collect())
            }
            RegionValues::PerRegion(values) => Err(RSCMError::Error(format!(
                "Initial grid values for '{name}' on the {grid} grid need \
                 {expected_len} value{}, got {}",
                if expected_len == 1 { "" } else { "s" },
                values.len()
            ))),
        }
    }
}

/// Build a new model from a set of components.
///
/// The builder generates a graph that defines the inter-component dependencies
/// and determines what variables are modelled and prescribed to the model.
/// This graph is used by the model to define the order in which components are solved.
///
/// A builder serialises with its components and data, so it is the spec a [`super::Record`] keeps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBuilder {
    components: Vec<C>,
    /// Label of each component, aligned with `components`.
    labels: Vec<String>,
    pub(crate) prescribed_variables: TimeseriesCollection,
    /// Names given to `with_prescribed_variable`, which must be read by the model.
    explicit_prescribed: BTreeSet<String>,
    initial_values: HashMap<String, InitialValue>,
    time_axis: Option<Arc<TimeAxis>>,
    schema: Option<VariableSchema>,
    /// Custom weights for grid aggregation, keyed by grid.
    ///
    /// When provided, these override the default weights used when creating
    /// timeseries and performing grid transformations. Weights must sum to 1.0.
    grid_weights: HashMap<GridType, Vec<f64>>,
    lagged_variables: BTreeSet<String>,
    lagged_edges: BTreeSet<(String, String)>,
    timeseries_types: HashMap<String, TimeseriesType>,
    state_type_check: bool,
    missing_prescribed_allowed: bool,
}

/// Check weights suit a grid: one per region of a four-box or hemispheric grid,
/// accepted by its `try_with_weights`.
pub fn check_grid_weights(grid: GridType, weights: &[f64]) -> Result<(), String> {
    use crate::spatial::{FourBoxGrid, HemisphericGrid};
    let expected = match grid {
        GridType::Scalar => {
            return Err("Cannot set weights for the scalar grid, which has no regions".into())
        }
        GridType::FourBox => 4,
        GridType::Hemispheric => 2,
    };
    if weights.len() != expected {
        return Err(format!(
            "Weights length {} does not match {grid} grid size {expected}",
            weights.len()
        ));
    }
    match grid {
        GridType::FourBox => FourBoxGrid::try_with_weights(weights.try_into().unwrap()).map(|_| ()),
        _ => HemisphericGrid::try_with_weights(weights.try_into().unwrap()).map(|_| ()),
    }
}

impl ModelBuilder {
    /// Create a new model builder with default settings.
    pub fn new() -> Self {
        Self {
            components: vec![],
            labels: vec![],
            initial_values: HashMap::new(),
            prescribed_variables: TimeseriesCollection::new(),
            explicit_prescribed: BTreeSet::new(),
            time_axis: None,
            schema: None,
            grid_weights: HashMap::new(),
            lagged_variables: BTreeSet::new(),
            lagged_edges: BTreeSet::new(),
            timeseries_types: HashMap::new(),
            state_type_check: true,
            missing_prescribed_allowed: false,
        }
    }

    /// Allow inputs that no component produces and no prescribed data or initial value supplies.
    ///
    /// Such an input is filled with NaN instead of failing the build.
    pub fn with_missing_prescribed_allowed(&mut self) -> &mut Self {
        self.missing_prescribed_allowed = true;
        self
    }

    /// Lag every edge from `variable`, so each of its consumers reads the previous step's value.
    ///
    /// The lag holds whether the variable is produced by a component or prescribed.
    pub fn with_lag(&mut self, variable: &str) -> &mut Self {
        self.lagged_variables.insert(variable.to_string());
        self
    }

    /// Lag the edge from `variable` into the component labelled `consumer` alone.
    ///
    /// `consumer` is a label, such as `"OceanCarbon"`, or the name of a schema aggregate.
    /// Other consumers of `variable` still read this step's value.
    pub fn with_lag_for(&mut self, variable: &str, consumer: &str) -> &mut Self {
        self.lagged_edges
            .insert((variable.to_string(), consumer.to_string()));
        self
    }

    fn is_lagged(&self, variable: &str, consumer: &str) -> bool {
        self.lagged_variables.contains(variable)
            || self
                .lagged_edges
                .contains(&(variable.to_string(), consumer.to_string()))
    }

    /// Check that every lag names a variable and an edge of the model.
    fn validate_lags(
        &self,
        labels: &[String],
        component_defs: &[(Vec<VariableDefinition>, Vec<VariableDefinition>)],
    ) -> RSCMResult<()> {
        let aggregates = self.schema.as_ref().map(|s| &s.aggregates);
        for variable in &self.lagged_variables {
            let known = component_defs
                .iter()
                .flat_map(|(requires, provides)| requires.iter().chain(provides))
                .any(|r| &r.name == variable)
                || aggregates.is_some_and(|a| a.contains_key(variable));
            if !known {
                return Err(RSCMError::InvalidLag {
                    lag: format!("with_lag({variable:?})"),
                    reason: "no component reads or produces this variable".to_string(),
                });
            }
        }
        for (variable, consumer) in &self.lagged_edges {
            let lag = format!("with_lag_for({variable:?}, {consumer:?})");
            let reads = if let Some(index) = labels.iter().position(|l| l == consumer) {
                component_defs[index]
                    .0
                    .iter()
                    .any(|r| &r.name == variable && r.role == VariableRole::Input)
            } else if let Some(aggregate) = aggregates.and_then(|a| a.get(consumer)) {
                aggregate.contributors.contains(variable)
            } else {
                let mut known: BTreeSet<&str> = labels.iter().map(String::as_str).collect();
                known.extend(
                    aggregates
                        .into_iter()
                        .flat_map(|a| a.keys().map(String::as_str)),
                );
                return Err(RSCMError::InvalidLag {
                    lag,
                    reason: format!("no component labelled {consumer:?}, labels are {known:?}"),
                });
            };
            if !reads {
                return Err(RSCMError::InvalidLag {
                    lag,
                    reason: format!("{consumer} does not read {variable:?} as an input"),
                });
            }
        }
        Ok(())
    }

    /// Check that no two components, or a component and a schema aggregate, share a label.
    fn check_unique_labels(&self) -> RSCMResult<()> {
        let aggregates = self.schema.iter().flat_map(|s| s.aggregates.keys());
        let mut seen = HashSet::new();
        for label in self.labels.iter().chain(aggregates) {
            if !seen.insert(label.as_str()) {
                return Err(RSCMError::DuplicateLabel {
                    label: label.clone(),
                });
            }
        }
        Ok(())
    }

    /// Check that no lag is on an average, which has no previous-step value at the first step.
    fn check_no_lagged_average(
        &self,
        variable_sources: &HashMap<(String, String), VariableSource>,
        timeseries_types: &BTreeMap<String, TimeseriesType>,
    ) -> RSCMResult<()> {
        let lagged_average = variable_sources
            .iter()
            .filter(|((variable, _), &source)| {
                source == VariableSource::Lagged
                    && timeseries_types.get(variable) == Some(&TimeseriesType::Average)
            })
            .map(|(edge, _)| edge)
            .min();
        let Some((variable, consumer)) = lagged_average else {
            return Ok(());
        };
        let lag = if self.lagged_variables.contains(variable) {
            format!("with_lag({variable:?})")
        } else {
            format!("with_lag_for({variable:?}, {consumer:?})")
        };
        Err(RSCMError::InvalidLag {
            lag,
            reason: format!(
                "{variable:?} is an average read by {consumer}, and lagged averages are not supported yet"
            ),
        })
    }

    /// Check that every input without a producer has prescribed data or an initial value.
    fn check_prescribed_supplied(
        &self,
        names: &[String],
        component_defs: &[(Vec<VariableDefinition>, Vec<VariableDefinition>)],
    ) -> RSCMResult<()> {
        let missing: Vec<(String, Vec<String>)> = self
            .collect_prescribed_inputs(names, component_defs)
            .into_iter()
            .filter(|input| !input.supplied)
            .map(|input| (input.definition.name, input.consumers))
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        Err(RSCMError::MissingPrescribedInput { inputs: missing })
    }

    /// Every input that no component or schema aggregate produces, sorted by name.
    ///
    /// These are the prescribed variables the model needs,
    /// each marked with whether prescribed data or an initial value supplies it.
    /// `build` fails if any is not supplied, unless missing inputs are allowed.
    pub fn prescribed_inputs(&self) -> Vec<PrescribedInput> {
        let component_defs: Vec<_> = self
            .components
            .iter()
            .map(|component| reads_and_writes(&component.definitions()))
            .collect();
        self.collect_prescribed_inputs(&self.labels, &component_defs)
    }

    fn collect_prescribed_inputs(
        &self,
        names: &[String],
        component_defs: &[(Vec<VariableDefinition>, Vec<VariableDefinition>)],
    ) -> Vec<PrescribedInput> {
        let mut produced: HashSet<&str> = self
            .schema
            .as_ref()
            .map(|schema| schema.aggregates.keys().map(String::as_str).collect())
            .unwrap_or_default();
        for (_, provides) in component_defs {
            produced.extend(provides.iter().map(|def| def.name.as_str()));
        }
        let mut inputs: BTreeMap<String, (VariableDefinition, BTreeSet<String>)> = BTreeMap::new();
        for (consumer, (requires, _)) in names.iter().zip(component_defs) {
            for def in requires {
                if def.role == VariableRole::Input && !produced.contains(def.name.as_str()) {
                    inputs
                        .entry(def.name.clone())
                        .or_insert_with(|| (def.clone(), BTreeSet::new()))
                        .1
                        .insert(consumer.clone());
                }
            }
        }
        inputs
            .into_iter()
            .map(|(name, (definition, consumers))| PrescribedInput {
                supplied: self.prescribed_variables.get_data(&name).is_some()
                    || self.initial_values.contains_key(&name),
                definition,
                consumers: consumers.into_iter().collect(),
            })
            .collect()
    }

    /// Set custom weights for a grid.
    ///
    /// These weights override the default grid weights used when:
    /// - Creating timeseries for grid-based variables
    /// - Performing automatic grid transformations (when enabled)
    ///
    /// # Arguments
    ///
    /// * `grid` - The grid to configure (four-box or hemispheric)
    /// * `weights` - Area-based weights that must sum to 1.0
    ///
    /// # Panics
    ///
    /// Panics where [`check_grid_weights`] returns an error.
    pub fn with_grid_weights(&mut self, grid: GridType, weights: Vec<f64>) -> &mut Self {
        check_grid_weights(grid, &weights).unwrap_or_else(|e| panic!("{e}"));
        self.grid_weights.insert(grid, weights);
        self
    }

    /// Create a FourBoxGrid using custom weights if configured, otherwise use defaults.
    fn create_four_box_grid(&self) -> crate::spatial::FourBoxGrid {
        use crate::spatial::FourBoxGrid;
        match self.grid_weights.get(&GridType::FourBox) {
            Some(weights) => {
                let weights_arr: [f64; 4] = weights
                    .as_slice()
                    .try_into()
                    .expect("four-box grid weights should have 4 elements");
                FourBoxGrid::with_weights(weights_arr)
            }
            None => FourBoxGrid::magicc_standard(),
        }
    }

    /// Create a HemisphericGrid using custom weights if configured, otherwise use defaults.
    fn create_hemispheric_grid(&self) -> crate::spatial::HemisphericGrid {
        use crate::spatial::HemisphericGrid;
        match self.grid_weights.get(&GridType::Hemispheric) {
            Some(weights) => {
                let weights_arr: [f64; 2] = weights
                    .as_slice()
                    .try_into()
                    .expect("Hemispheric weights should have 2 elements");
                HemisphericGrid::with_weights(weights_arr)
            }
            None => HemisphericGrid::equal_weights(),
        }
    }

    /// Set the variable schema for the model.
    ///
    /// Each schema entry sets the unit and grid its variable is stored in,
    /// and each aggregate adds a component that computes it.
    /// A variable without an entry is stored as its producer declares it, or as its prescribed data is.
    /// Variables defined in the schema but not produced by any component will
    /// be initialised to NaN.
    pub fn with_schema(&mut self, schema: VariableSchema) -> &mut Self {
        self.schema = Some(schema);
        self
    }

    /// Register a component, labelled with its type name.
    ///
    /// [`Self::build`] fails if another component has the same label,
    /// so register a second instance of a type with [`Self::with_component_as`].
    pub fn with_component(&mut self, component: Arc<dyn Component + Send + Sync>) -> &mut Self {
        let label = Model::extract_component_name(&component);
        self.with_component_as(label, component)
    }

    /// Register a component under `label`, which lags, the coupling report and errors name it by.
    pub fn with_component_as(
        &mut self,
        label: impl Into<String>,
        component: Arc<dyn Component + Send + Sync>,
    ) -> &mut Self {
        self.labels.push(label.into());
        self.components.push(component);
        self
    }

    /// A copy whose components are rebuilt from their serialised form, so it shares no mutable state.
    pub(crate) fn snapshot(&self) -> RSCMResult<ModelBuilder> {
        let mut copy = self.clone();
        for (component, label) in copy.components.iter_mut().zip(&self.labels) {
            let value = serde_json::to_value(&**component).map_err(|e| {
                RSCMError::Error(format!("Could not serialise component {label}: {e}"))
            })?;
            let fresh: Box<dyn Component> = serde_json::from_value(value)
                .map_err(|e| super::record::read_error(&format!("component {label}"), e))?;
            *component = Arc::from(fresh);
        }
        Ok(copy)
    }

    /// Replace the component labelled `label` with another of the same type.
    pub(crate) fn replace_component(&mut self, label: &str, component: C) -> RSCMResult<()> {
        let index = self.labels.iter().position(|l| l == label).ok_or_else(|| {
            RSCMError::Error(format!(
                "No component is labelled '{label}', the labels are {}",
                self.labels.join(", ")
            ))
        })?;
        let existing = &self.components[index];
        let type_name = |c: &C| (c.typetag_name(), Model::extract_component_name(c));
        if type_name(existing) != type_name(&component) {
            return Err(RSCMError::Error(format!(
                "Component '{label}' has type {}, so it cannot be replaced by a {}",
                type_name(existing).1,
                type_name(&component).1
            )));
        }
        self.components[index] = component;
        Ok(())
    }

    /// Supply prescribed data for a variable, on any grid, replacing any earlier data for it.
    ///
    /// [`Self::build`] fails if no component reads the variable or a component produces it.
    /// It stores the data in the unit and grid supplied, unless a schema entry or a producer sets them,
    /// and fails if a reader's unit cannot be converted from the stored one.
    pub fn with_prescribed_variable(
        &mut self,
        name: &str,
        timeseries: impl Into<TimeseriesData>,
    ) -> &mut Self {
        self.prescribed_variables
            .set_data(name.to_string(), timeseries);
        self.explicit_prescribed.insert(name.to_string());
        self
    }

    /// Supply prescribed data from a collection, replacing any earlier data for the same variables.
    ///
    /// Any unneeded timeseries will be ignored.
    /// [`Self::build`] stores the data as [`Self::with_prescribed_variable`] does.
    pub fn with_prescribed_collection(&mut self, collection: TimeseriesCollection) -> &mut Self {
        for item in collection {
            self.prescribed_variables.set_data(item.name, item.data);
        }
        self
    }

    /// Adds some state to the set of initial values.
    ///
    /// These initial values are used to provide some initial values at `t_0`.
    /// Initial values are used for variables whose role is `VariableRole::State`.
    /// State variables read their value from the previous step in order to generate a new value
    /// for the next step.
    /// Building a model where any variables which have `VariableRole::State`, but
    /// do not have an initial value will result in an error.
    /// Building also fails for a name no component reads or produces.
    pub fn with_initial_values(
        &mut self,
        initial_values: HashMap<String, FloatValue>,
    ) -> &mut Self {
        for (name, value) in initial_values {
            self.with_initial_value(name, RegionValues::Uniform(value), None);
        }
        self
    }

    /// Like [`Self::with_initial_values`], with each value given in its own unit.
    ///
    /// Building converts the values into the unit the model stores each variable in,
    /// and fails if the units are incompatible.
    pub fn with_initial_values_in_units(
        &mut self,
        initial_values: HashMap<String, (FloatValue, String)>,
    ) -> &mut Self {
        for (name, (value, unit)) in initial_values {
            self.with_initial_value(name, RegionValues::Uniform(value), Some(unit));
        }
        self
    }

    /// Adds initial values for grid variables, one per region in grid order.
    ///
    /// Each name holds one initial value, so the latest call for a name replaces
    /// a value given to [`Self::with_initial_values`] and the reverse.
    /// Building fails if the number of values does not match the variable's grid,
    /// or for a name no component reads or produces.
    pub fn with_initial_grid_values(
        &mut self,
        initial_values: HashMap<String, Vec<FloatValue>>,
    ) -> &mut Self {
        for (name, values) in initial_values {
            self.with_initial_value(name, RegionValues::PerRegion(values), None);
        }
        self
    }

    /// Like [`Self::with_initial_grid_values`], with each variable's values given in their own unit.
    pub fn with_initial_grid_values_in_units(
        &mut self,
        initial_values: HashMap<String, (Vec<FloatValue>, String)>,
    ) -> &mut Self {
        for (name, (values, unit)) in initial_values {
            self.with_initial_value(name, RegionValues::PerRegion(values), Some(unit));
        }
        self
    }

    /// Set one variable's initial value, in the stored unit when `unit` is `None`.
    pub(crate) fn with_initial_value(
        &mut self,
        name: String,
        values: RegionValues,
        unit: Option<String>,
    ) -> &mut Self {
        self.initial_values
            .insert(name, InitialValue { values, unit });
        self
    }

    /// Specify the time axis that will be used by the model.
    ///
    /// This time axis defines the steps (including bounds) on which the model will be iterated.
    /// [`Self::build`] fails without one.
    /// The time axis set with `with_time_axis`, if any.
    pub fn time_axis(&self) -> Option<&Arc<TimeAxis>> {
        self.time_axis.as_ref()
    }

    pub fn with_time_axis(&mut self, time_axis: impl Into<Arc<TimeAxis>>) -> &mut Self {
        self.time_axis = Some(time_axis.into());
        self
    }

    /// Declare whether a prescribed variable is a point or an average.
    ///
    /// Use it when the unit default is wrong. A component that produces the variable must agree.
    pub fn with_timeseries_type(
        &mut self,
        variable: &str,
        timeseries_type: TimeseriesType,
    ) -> &mut Self {
        self.timeseries_types
            .insert(variable.to_string(), timeseries_type);
        self
    }

    /// Turn the check that every state variable is a point on or off.
    pub fn with_state_type_check(&mut self, enabled: bool) -> &mut Self {
        self.state_type_check = enabled;
        self
    }

    /// Interpolate prescribed data onto the model's time axis in the unit the model stores it in.
    ///
    /// Returns an error if the data's unit cannot be converted into `unit`.
    /// Pushes a warning onto `warnings` if the data has to be extrapolated to cover the axis.
    fn prescribed_in_unit<G: SpatialGrid>(
        name: &str,
        ts: &GridTimeseries<FloatValue, G>,
        unit: &str,
        time_axis: &Arc<TimeAxis>,
        warnings: &mut Vec<String>,
    ) -> RSCMResult<GridTimeseries<FloatValue, G>> {
        let factor = crate::units::factor(name, ts.unit(), unit)?;
        let data_axis = ts.time_axis();
        let (start, end) = (*data_axis.first(), *data_axis.last());
        let times = time_axis.values();
        let (model_start, model_end) = (times[0], times[times.len() - 1]);
        let mut directions = Vec::new();
        if model_start < start {
            directions.push(format!("back to {model_start}"));
        }
        if model_end > end {
            directions.push(format!("forward to {model_end}"));
        }
        if !directions.is_empty() {
            warnings.push(format!(
                "Prescribed data for '{name}' covers {start} to {end}, so it is extrapolated {}",
                directions.join(" and ")
            ));
        }
        let ts = ts.to_owned().interpolate_into(time_axis.clone());
        Ok(match factor {
            Some(factor) => ts.convert_unit(factor, unit.to_string()),
            None => ts,
        })
    }

    /// The custom weights for `grid`, if any.
    pub(crate) fn grid_weights(&self, grid: GridType) -> Option<&Vec<f64>> {
        self.grid_weights.get(&grid)
    }

    /// Builds the component graph for the registered components and creates a concrete model.
    ///
    /// Returns an error if the component definitions are inconsistent.
    /// Logs any warnings from [`Self::build_with_warnings`].
    pub fn build(&self) -> RSCMResult<Model> {
        let (model, warnings) = self.build_with_warnings()?;
        for message in warnings {
            tracing::warn!("{message}");
        }
        Ok(model)
    }

    /// Like [`Self::build`], but also returns warnings, such as prescribed data that had to be extrapolated.
    pub fn build_with_warnings(&self) -> RSCMResult<(Model, Vec<String>)> {
        use crate::spatial::ScalarRegion;

        let time_axis = self.time_axis.clone().ok_or(RSCMError::MissingTimeAxis)?;
        let has_aggregates = self
            .schema
            .as_ref()
            .is_some_and(|s| !s.aggregates.is_empty());
        if self.components.is_empty() && !has_aggregates {
            return Err(RSCMError::NoComponents);
        }
        let mut warnings = Vec::new();

        // todo: refactor once this is more stable
        let mut graph: CGraph = Graph::new();
        let mut modelled: HashMap<String, NodeIndex> = HashMap::new();
        let mut prescribed: HashSet<String> = HashSet::new();
        // Track which component owns each variable for better error messages
        let mut variable_owners: HashMap<String, String> = HashMap::new();
        // Track variable sources keyed by (variable_name, component_name)
        // This enables different components to have different source classifications
        // for the same variable (e.g., State for owner, Upstream for consumers)
        let mut variable_sources: HashMap<(String, String), VariableSource> = HashMap::new();
        let mut aggregate_nodes: HashSet<NodeIndex> = HashSet::new();

        // Track pending aggregate dependencies: (component_node, variable_name, definition)
        // These will be resolved after aggregate components are added to the graph
        let mut pending_aggregate_deps: Vec<(NodeIndex, String, VariableDefinition)> = vec![];

        // Get aggregate names for forward reference (if schema exists)
        let aggregate_names: HashSet<String> = self
            .schema
            .as_ref()
            .map(|s| s.aggregates.keys().cloned().collect())
            .unwrap_or_default();

        // Each call builds the definitions afresh, so read them once.
        let component_defs: Vec<_> = self
            .components
            .iter()
            .map(|component| reads_and_writes(&component.definitions()))
            .collect();

        let names = &self.labels;
        self.check_unique_labels()?;
        self.validate_lags(names, &component_defs)?;

        let nodes: Vec<NodeIndex> = self
            .components
            .iter()
            .map(|component| graph.add_node(component.clone()))
            .collect();
        let mut node_labels: BTreeMap<NodeIndex, String> =
            nodes.iter().copied().zip(names.iter().cloned()).collect();

        // Every producer is known before any input is classified, so registration order is irrelevant.
        for (index, (_, provides)) in component_defs.iter().enumerate() {
            for def in provides {
                if let Some(other) = variable_owners.get(&def.name) {
                    return Err(RSCMError::MultipleProducers {
                        variable: def.name.clone(),
                        components: vec![other.clone(), names[index].clone()],
                    });
                }
                modelled.insert(def.name.clone(), nodes[index]);
                variable_owners.insert(def.name.clone(), names[index].clone());
            }
        }

        for (index, (requires, _)) in component_defs.iter().enumerate() {
            let node = nodes[index];
            let component_name = &names[index];

            for def in requires {
                let name = &def.name;
                let lagged = self.is_lagged(name, component_name);
                let producer = modelled.get(name).copied();
                let source = if def.role == VariableRole::State || producer == Some(node) {
                    prescribed.insert(name.clone());
                    VariableSource::OwnState
                } else if lagged {
                    // A lagged read sees index N, which prescribed data may fill.
                    prescribed.insert(name.clone());
                    VariableSource::Lagged
                } else if let Some(producer) = producer {
                    graph.add_edge(producer, node, def.clone());
                    VariableSource::Upstream
                } else if aggregate_names.contains(name) {
                    pending_aggregate_deps.push((node, name.clone(), def.clone()));
                    VariableSource::Upstream
                } else {
                    prescribed.insert(name.clone());
                    VariableSource::Prescribed
                };
                variable_sources.insert((name.clone(), component_name.clone()), source);
            }
        }

        if let Some(schema) = &self.schema {
            schema.validate()?;
            // A schema variable no component reads or produces is still stored, empty or prescribed.
            prescribed.extend(
                schema
                    .variables
                    .keys()
                    .filter(|name| !modelled.contains_key(*name))
                    .cloned(),
            );
        }
        let Storage {
            mut definitions,
            read_units,
            grids,
        } = resolve_storage(
            self.schema.as_ref(),
            names,
            &component_defs,
            &self.prescribed_variables,
        )?;

        if let Some(schema) = &self.schema {
            // Add a component for each aggregate definition
            // Process aggregates in topological order to handle chained aggregates
            let ordered_aggregates = schema.topological_order_aggregates();
            for agg_name in &ordered_aggregates {
                let agg_def = schema.get_aggregate(agg_name).unwrap();

                // Create the aggregate's component
                let agg_component = crate::schema::AggregateComponent::from_definition(agg_def);

                // Add to graph
                let agg_node = graph.add_node(Arc::new(agg_component));
                aggregate_nodes.insert(agg_node);

                // An aggregate's label is its schema name.
                variable_owners.insert(agg_name.clone(), agg_name.clone());
                node_labels.insert(agg_node, agg_name.clone());

                // Add dependency edges from contributor sources to the aggregate's component
                for contributor in &agg_def.contributors {
                    let source = if self.is_lagged(contributor, agg_name) {
                        VariableSource::Lagged
                    } else if let Some(&producer_node) = modelled.get(contributor) {
                        graph.add_edge(
                            producer_node,
                            agg_node,
                            VariableDefinition::with_grid(
                                contributor,
                                &agg_def.unit,
                                VariableRole::Input,
                                agg_def.grid,
                            ),
                        );
                        VariableSource::Upstream
                    } else {
                        VariableSource::Prescribed
                    };
                    variable_sources.insert((contributor.clone(), agg_name.clone()), source);
                }

                // Register the aggregate output as modelled
                modelled.insert(agg_name.clone(), agg_node);
            }

            // Resolve pending aggregate dependencies now that aggregate components are in the graph
            for (component_node, var_name, def) in pending_aggregate_deps {
                if let Some(&agg_node) = modelled.get(&var_name) {
                    graph.add_edge(agg_node, component_node, def);
                }
            }
        }

        self.check_supplied_values_are_read(&definitions, &prescribed, &variable_owners)?;
        check_acyclic(
            &graph,
            &component_defs,
            names,
            &node_labels,
            &aggregate_nodes,
        )?;
        resolve_timeseries_types(
            &mut definitions,
            names,
            &component_defs,
            &self.timeseries_types,
            self.state_type_check,
        )?;
        let timeseries_types: BTreeMap<String, TimeseriesType> = definitions
            .iter()
            .map(|(name, definition)| (name.clone(), definition.timeseries_type))
            .collect();
        self.check_no_lagged_average(&variable_sources, &timeseries_types)?;

        // Validate that all state variables have initial values
        for (name, def) in &definitions {
            if def.role == VariableRole::State && !self.initial_values.contains_key(name) {
                let component = variable_owners
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());
                return Err(RSCMError::MissingInitialValue {
                    variable: name.clone(),
                    component,
                });
            }
        }

        if !self.missing_prescribed_allowed {
            self.check_prescribed_supplied(names, &component_defs)?;
        }

        // Create the timeseries collection using the information from the components
        let mut collection = TimeseriesCollection::new();

        for (name, definition) in definitions {
            assert_eq!(definition.name, name);
            let interp = match definition.timeseries_type {
                TimeseriesType::Point => {
                    InterpolationStrategy::from(LinearSplineStrategy::new(true))
                }
                TimeseriesType::Average => InterpolationStrategy::from(PreviousStrategy::new(true)),
            };

            let storage_grid = definition.grid;

            // Try to populate from: (1) prescribed data, (2) initial value, (3) empty
            let prescribed_data = if prescribed.contains(&name) {
                self.prescribed_variables.get_data(&name)
            } else {
                None
            };
            let storage_unit = definition.unit.as_str();
            let initial = self
                .initial_values
                .get(&name)
                .map(|initial| initial.resolve(&name, storage_unit, storage_grid))
                .transpose()?;
            match storage_grid {
                GridType::Scalar => {
                    if let Some(TimeseriesData::Scalar(ts)) = prescribed_data {
                        let ts = Self::prescribed_in_unit(
                            &name,
                            ts,
                            storage_unit,
                            &time_axis,
                            &mut warnings,
                        )?;
                        collection.add_timeseries(name, ts);
                    } else {
                        let mut ts = Timeseries::new_empty_scalar(
                            time_axis.clone(),
                            storage_unit.to_string(),
                            interp.clone(),
                        );
                        if let Some(values) = &initial {
                            ts.set(0, ScalarRegion::Global, values[0]);
                        }
                        collection.add_timeseries(name, ts);
                    }
                }
                GridType::FourBox => {
                    if let Some(TimeseriesData::FourBox(ts)) = prescribed_data {
                        let ts = Self::prescribed_in_unit(
                            &name,
                            ts,
                            storage_unit,
                            &time_axis,
                            &mut warnings,
                        )?;
                        collection.add_four_box_timeseries(name, ts);
                    } else {
                        let mut ts = GridTimeseries::new_empty(
                            time_axis.clone(),
                            self.create_four_box_grid(),
                            storage_unit.to_string(),
                            interp.clone(),
                        );
                        if let Some(values) = &initial {
                            ts.set_all(0, values);
                        }
                        collection.add_four_box_timeseries(name, ts);
                    }
                }
                GridType::Hemispheric => {
                    if let Some(TimeseriesData::Hemispheric(ts)) = prescribed_data {
                        let ts = Self::prescribed_in_unit(
                            &name,
                            ts,
                            storage_unit,
                            &time_axis,
                            &mut warnings,
                        )?;
                        collection.add_hemispheric_timeseries(name, ts);
                    } else {
                        let mut ts = GridTimeseries::new_empty(
                            time_axis.clone(),
                            self.create_hemispheric_grid(),
                            storage_unit.to_string(),
                            interp.clone(),
                        );
                        if let Some(values) = &initial {
                            ts.set_all(0, values);
                        }
                        collection.add_hemispheric_timeseries(name, ts);
                    }
                }
            }
        }

        let step_plan = resolve(&Decisions {
            graph: &graph,
            labels: &node_labels,
            collection: &collection,
            sources: &variable_sources,
            producers: &variable_owners,
            read_units: &read_units,
            grids: &grids,
            timeseries_types: &timeseries_types,
            grid_weights: &self.grid_weights,
        })?;
        let model = Model::from_plan(
            graph,
            collection,
            time_axis,
            step_plan,
            Arc::new(self.clone()),
        );
        Ok((model, warnings))
    }

    /// Check that every initial value names a model variable,
    /// and that every variable prescribed by name is read and produced by no component.
    ///
    /// `prescribed` holds the variables the model takes from prescribed data.
    fn check_supplied_values_are_read(
        &self,
        definitions: &HashMap<String, ResolvedVariable>,
        prescribed: &HashSet<String>,
        owners: &HashMap<String, String>,
    ) -> RSCMResult<()> {
        let initial_names: BTreeSet<&String> = self.initial_values.keys().collect();
        if let Some(variable) = initial_names
            .into_iter()
            .find(|n| !definitions.contains_key(*n))
        {
            return Err(RSCMError::UnknownInitialValue {
                variable: variable.clone(),
                suggestions: close_matches(variable, &sorted_names(definitions.keys())),
            });
        }
        for variable in &self.explicit_prescribed {
            if let Some(component) = owners.get(variable) {
                return Err(RSCMError::PrescribedOutput {
                    variable: variable.clone(),
                    component: component.clone(),
                });
            }
            if !prescribed.contains(variable) {
                let unproduced = prescribed.iter().filter(|n| !owners.contains_key(*n));
                return Err(RSCMError::UnusedPrescribedVariable {
                    variable: variable.clone(),
                    suggestions: close_matches(variable, &sorted_names(unproduced)),
                });
            }
        }
        Ok(())
    }
}

fn sorted_names<'a>(names: impl Iterator<Item = &'a String>) -> Vec<&'a str> {
    let mut names: Vec<&str> = names.map(String::as_str).collect();
    names.sort_unstable();
    names
}

/// Check that no feedback loop is left unlagged.
///
/// A cycle is a feedback loop no lag breaks. The error lists the components and variables in each loop,
/// and the `with_lag_for` calls that would break it, edges into a component with its own state first.
fn check_acyclic(
    graph: &CGraph,
    component_defs: &[(Vec<VariableDefinition>, Vec<VariableDefinition>)],
    names: &[String],
    labels: &BTreeMap<NodeIndex, String>,
    aggregate_nodes: &HashSet<NodeIndex>,
) -> RSCMResult<()> {
    if !petgraph::algo::is_cyclic_directed(graph) {
        return Ok(());
    }
    let stateful: HashSet<&str> = names
        .iter()
        .zip(component_defs)
        .filter(|(_, (requires, _))| requires.iter().any(|r| r.role == VariableRole::State))
        .map(|(name, _)| name.as_str())
        .collect();
    let mut components = BTreeSet::new();
    let mut variables = BTreeSet::new();
    let mut suggestions = BTreeSet::new();
    for scc in petgraph::algo::tarjan_scc(graph) {
        if scc.len() < 2 {
            continue;
        }
        let members: HashSet<NodeIndex> = scc.iter().copied().collect();
        components.extend(scc.iter().map(|n| labels[n].clone()));
        for edge in graph.edge_references() {
            if !(members.contains(&edge.source()) && members.contains(&edge.target())) {
                continue;
            }
            let variable = edge.weight().name.clone();
            variables.insert(variable.clone());
            // Suggest lags into components only, since a lagged aggregate shifts every reader of its total.
            if !aggregate_nodes.contains(&edge.target()) {
                let consumer = labels[&edge.target()].clone();
                let rank = !stateful.contains(consumer.as_str());
                suggestions.insert((rank, variable, consumer));
            }
        }
    }
    Err(RSCMError::FeedbackLoop {
        components: components.into_iter().collect(),
        variables: variables.into_iter().collect(),
        suggestions: suggestions
            .into_iter()
            .map(|(_, variable, consumer)| (variable, consumer))
            .collect(),
    })
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}
