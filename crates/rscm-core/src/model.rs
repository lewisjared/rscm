/// A model consists of a series of coupled components which are solved together.
/// The model orchastrates the passing of state between different components.
/// Each component is solved for a given time step in an order determined by their
/// dependencies.
/// Once all components and state is solved for, the model will move to the next time step.
/// The state from previous steps is preserved as it is useful as output or in the case where
/// a component needs previous values.
///
/// The model also holds all of the exogenous variables required by the model.
/// The required variables are identified when building the model.
/// If a required exogenous variable isn't provided, then the build step will fail.
use crate::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
};
use crate::errors::{RSCMError, RSCMResult};
use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
use crate::schema::VariableSchema;
use crate::spatial::{FourBoxGrid, FourBoxRegion, HemisphericGrid, ScalarRegion};
use crate::state::{HemisphericSlice, StateValue};
use crate::timeseries::{FloatValue, Time, TimeAxis, Timeseries};
use crate::timeseries_collection::{TimeseriesCollection, VariableType};
use numpy::ndarray::Array;
use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::visit::{Bfs, IntoNeighbors, IntoNodeIdentifiers, Visitable};
use petgraph::Graph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Index;
use std::sync::Arc;

type C = Arc<dyn Component>;
type CGraph = Graph<C, RequirementDefinition>;

#[derive(Debug)]
struct VariableDefinition {
    name: String,
    unit: String,
    grid_type: GridType,
}

impl VariableDefinition {
    fn from_requirement_definition(definition: &RequirementDefinition) -> Self {
        Self {
            name: definition.name.clone(),
            unit: definition.unit.clone(),
            grid_type: definition.grid_type,
        }
    }
}

/// Direction of a grid transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransformDirection {
    /// Read-side: aggregating schema data before component reads it
    /// (e.g., schema has FourBox, component wants Scalar)
    Read,
    /// Write-side: aggregating component output before storing in schema
    /// (e.g., component produces FourBox, schema wants Scalar)
    Write,
}

/// A required grid transformation identified during validation.
///
/// These are collected during component validation against the schema
/// and used to configure runtime grid aggregation (transform-on-read/write).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequiredTransformation {
    /// The variable name being transformed
    pub variable: String,
    /// The unit of the variable
    pub unit: String,
    /// The source grid type (finer resolution)
    pub source_grid: GridType,
    /// The target grid type (coarser resolution)
    pub target_grid: GridType,
    /// Direction of the transformation
    pub direction: TransformDirection,
}

/// A null component that does nothing
///
/// Used as an initial component to ensure that the model is connected
#[derive(Debug, Serialize, Deserialize)]
struct NullComponent {}

#[typetag::serde]
impl Component for NullComponent {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        vec![]
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        _input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        Ok(OutputState::new())
    }
}

/// Build a new model from a set of components
///
/// The builder generates a graph that defines the inter-component dependencies
/// and determines what variables are endogenous and exogenous to the model.
/// This graph is used by the model to define the order in which components are solved.
///
/// # Examples
/// TODO: figure out how to share example components throughout the docs
pub struct ModelBuilder {
    components: Vec<C>,
    exogenous_variables: TimeseriesCollection,
    initial_values: HashMap<String, FloatValue>,
    pub time_axis: Arc<TimeAxis>,
    schema: Option<VariableSchema>,
    /// Custom weights for grid aggregation, keyed by grid type
    ///
    /// When provided, these override the default weights used when creating
    /// timeseries and performing grid transformations. Weights must sum to 1.0.
    grid_weights: HashMap<GridType, Vec<f64>>,
}

/// Checks if the new definition is valid
///
/// If any definitions share a name then the units and grid types must be equivalent.
/// When `has_schema` is true, grid type checking is skipped because the schema validation
/// will handle grid compatibility with relaxed rules (allowing aggregation).
///
/// Returns an error if the parameter definition is inconsistent with any existing definitions.
fn verify_definition(
    definitions: &mut HashMap<String, VariableDefinition>,
    definition: &RequirementDefinition,
    component_name: &str,
    existing_component_name: Option<&str>,
    has_schema: bool,
) -> RSCMResult<()> {
    let existing = definitions.get(&definition.name);
    match existing {
        Some(existing) => {
            if existing.unit != definition.unit {
                return Err(RSCMError::Error(format!(
                    "Unit mismatch for variable '{}': component '{}' uses '{}' but component '{}' uses '{}'. \
                     All producers and consumers of a variable must use the same unit.",
                    definition.name,
                    existing_component_name.unwrap_or("unknown"),
                    existing.unit,
                    component_name,
                    definition.unit
                )));
            }

            // Skip grid type check when schema is present - schema validation handles it
            // with relaxed rules that allow aggregation
            if !has_schema && existing.grid_type != definition.grid_type {
                return Err(RSCMError::GridTypeMismatch {
                    variable: definition.name.clone(),
                    producer_component: existing_component_name.unwrap_or("unknown").to_string(),
                    consumer_component: component_name.to_string(),
                    producer_grid: existing.grid_type.to_string(),
                    consumer_grid: definition.grid_type.to_string(),
                });
            }
        }
        None => {
            definitions.insert(
                definition.name.clone(),
                VariableDefinition::from_requirement_definition(definition),
            );
        }
    }
    Ok(())
}

use crate::state::{ReadTransformInfo, TransformContext};

/// Extract the input state for the current time step
///
/// By default, for endogenous variables which are calculated as part of the model
/// the most recent value is used, whereas, for exogenous variables the values are linearly
/// interpolated.
/// This ensures that state calculated from previous components within the same timestep
/// is used.
///
/// The result should contain values for the current time step for all input variable
pub fn extract_state(
    collection: &TimeseriesCollection,
    input_names: Vec<String>,
    t_current: Time,
) -> InputState<'_> {
    let mut state = Vec::new();

    input_names.into_iter().for_each(|name| {
        let ts = collection
            .get_by_name(name.as_str())
            .unwrap_or_else(|| panic!("No timeseries with variable='{}'", name));
        state.push(ts);
    });

    InputState::build(state, t_current)
}

/// Extract the input state with transform context for grid aggregation.
///
/// Like `extract_state`, but includes transformation context for automatic
/// grid aggregation when reading variables at coarser resolutions.
pub fn extract_state_with_transforms(
    collection: &TimeseriesCollection,
    input_names: Vec<String>,
    t_current: Time,
    transform_context: TransformContext,
) -> InputState<'_> {
    let mut state = Vec::new();

    input_names.into_iter().for_each(|name| {
        let ts = collection
            .get_by_name(name.as_str())
            .unwrap_or_else(|| panic!("No timeseries with variable='{}'", name));
        state.push(ts);
    });

    InputState::build_with_transforms(state, t_current, transform_context)
}

/// Aggregate a StateValue from a finer grid to a coarser grid.
///
/// This function performs write-side grid transformation, aggregating component output
/// at a finer resolution to the schema's declared coarser resolution before storage.
///
/// # Arguments
///
/// * `value` - The StateValue to aggregate
/// * `source_grid` - The grid type the component produced (finer resolution)
/// * `target_grid` - The grid type the schema expects (coarser resolution)
/// * `weights` - Optional custom weights for aggregation; uses default grid weights if None
///
/// # Supported transformations
///
/// * FourBox -> Scalar: weighted average of all 4 regions
/// * FourBox -> Hemispheric: weighted average within each hemisphere
/// * Hemispheric -> Scalar: weighted average of 2 hemispheres
///
/// # Errors
///
/// Returns an error if:
/// * The source grid is coarser than the target (disaggregation not supported)
/// * The StateValue variant doesn't match the declared source_grid
fn aggregate_state_value(
    value: &StateValue,
    source_grid: GridType,
    target_grid: GridType,
    weights: Option<&Vec<f64>>,
) -> RSCMResult<StateValue> {
    match (source_grid, target_grid) {
        // FourBox -> Scalar: aggregate all 4 regions to global
        (GridType::FourBox, GridType::Scalar) => {
            let slice = match value {
                StateValue::FourBox(s) => s,
                _ => panic!(
                    "StateValue type mismatch: expected FourBox but got {:?}",
                    value
                ),
            };
            let grid = match weights {
                Some(w) => {
                    let arr: [f64; 4] = w.as_slice().try_into().unwrap_or_else(|_| {
                        panic!("FourBox weights must have 4 elements, got {}", w.len())
                    });
                    FourBoxGrid::with_weights(arr)
                }
                None => FourBoxGrid::magicc_standard(),
            };
            Ok(StateValue::Scalar(slice.aggregate_global(&grid)))
        }

        // FourBox -> Hemispheric: aggregate by hemisphere
        (GridType::FourBox, GridType::Hemispheric) => {
            let slice = match value {
                StateValue::FourBox(s) => s,
                _ => panic!(
                    "StateValue type mismatch: expected FourBox but got {:?}",
                    value
                ),
            };
            let grid_weights = match weights {
                Some(w) => {
                    let arr: [f64; 4] = w.as_slice().try_into().unwrap_or_else(|_| {
                        panic!("FourBox weights must have 4 elements, got {}", w.len())
                    });
                    arr
                }
                None => [0.25, 0.25, 0.25, 0.25],
            };

            // Aggregate by hemisphere using weighted averages
            let values = slice.as_array();
            let no = FourBoxRegion::NorthernOcean as usize;
            let nl = FourBoxRegion::NorthernLand as usize;
            let so = FourBoxRegion::SouthernOcean as usize;
            let sl = FourBoxRegion::SouthernLand as usize;

            let northern = (values[no] * grid_weights[no] + values[nl] * grid_weights[nl])
                / (grid_weights[no] + grid_weights[nl]);
            let southern = (values[so] * grid_weights[so] + values[sl] * grid_weights[sl])
                / (grid_weights[so] + grid_weights[sl]);

            Ok(StateValue::Hemispheric(HemisphericSlice::from([
                northern, southern,
            ])))
        }

        // Hemispheric -> Scalar: aggregate both hemispheres to global
        (GridType::Hemispheric, GridType::Scalar) => {
            let slice = match value {
                StateValue::Hemispheric(s) => s,
                _ => panic!(
                    "StateValue type mismatch: expected Hemispheric but got {:?}",
                    value
                ),
            };
            let grid = match weights {
                Some(w) => {
                    let arr: [f64; 2] = w.as_slice().try_into().unwrap_or_else(|_| {
                        panic!("Hemispheric weights must have 2 elements, got {}", w.len())
                    });
                    HemisphericGrid::with_weights(arr)
                }
                None => HemisphericGrid::default(),
            };
            Ok(StateValue::Scalar(slice.aggregate_global(&grid)))
        }

        // Same grid type: no transformation needed (identity)
        (s, t) if s == t => Ok(value.clone()),

        // Any other combination is not supported (disaggregation)
        _ => Err(RSCMError::GridTransformationNotSupported {
            source_grid: format!("{:?}", source_grid),
            target_grid: format!("{:?}", target_grid),
            variable: "unknown".to_string(),
        }),
    }
}

/// Check that a component graph is valid
///
/// We require a directed acyclic graph which doesn't contain any cycles (other than a self-referential node).
/// This avoids the case where component `A` depends on a component `B`,
/// but component `B` also depends on component `A`.
fn is_valid_graph<G>(g: G) -> bool
where
    G: IntoNodeIdentifiers + IntoNeighbors + Visitable,
{
    use petgraph::visit::{depth_first_search, DfsEvent};

    depth_first_search(g, g.node_identifiers(), |event| match event {
        DfsEvent::BackEdge(a, b) => {
            // If the cycle is self-referential then that is fine
            match a == b {
                true => Ok(()),
                false => Err(()),
            }
        }
        _ => Ok(()),
    })
    .is_err()
}

impl ModelBuilder {
    pub fn new() -> Self {
        Self {
            components: vec![],
            initial_values: HashMap::new(),
            exogenous_variables: TimeseriesCollection::new(),
            time_axis: Arc::new(TimeAxis::from_values(Array::range(2000.0, 2100.0, 1.0))),
            schema: None,
            grid_weights: HashMap::new(),
        }
    }

    /// Set custom weights for a grid type
    ///
    /// These weights override the default grid weights used when:
    /// - Creating timeseries for grid-based variables
    /// - Performing automatic grid transformations (when enabled)
    ///
    /// # Arguments
    ///
    /// * `grid_type` - The grid type to configure (FourBox or Hemispheric)
    /// * `weights` - Area-based weights that must sum to 1.0
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `grid_type` is `Scalar` (scalars have no weights)
    /// - `weights` length does not match the grid size (4 for FourBox, 2 for Hemispheric)
    /// - `weights` do not sum to approximately 1.0 (within 1e-6)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rscm_core::model::ModelBuilder;
    /// use rscm_core::component::GridType;
    ///
    /// let mut builder = ModelBuilder::new();
    /// // Area-based weights for FourBox grid
    /// builder.with_grid_weights(GridType::FourBox, vec![0.36, 0.14, 0.36, 0.14]);
    /// ```
    pub fn with_grid_weights(&mut self, grid_type: GridType, weights: Vec<f64>) -> &mut Self {
        // Validate grid type
        let expected_size = match grid_type {
            GridType::Scalar => {
                panic!("Cannot set weights for Scalar grid type (scalars have no regional weights)")
            }
            GridType::FourBox => 4,
            GridType::Hemispheric => 2,
        };

        // Validate weights length
        assert_eq!(
            weights.len(),
            expected_size,
            "Weights length {} does not match {} grid size {}",
            weights.len(),
            grid_type,
            expected_size
        );

        // Validate weights sum to 1.0
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Weights must sum to 1.0, got {}",
            sum
        );

        self.grid_weights.insert(grid_type, weights);
        self
    }

    /// Create a FourBoxGrid using custom weights if configured, otherwise use defaults
    fn create_four_box_grid(&self) -> crate::spatial::FourBoxGrid {
        use crate::spatial::FourBoxGrid;
        match self.grid_weights.get(&GridType::FourBox) {
            Some(weights) => {
                let weights_arr: [f64; 4] = weights
                    .as_slice()
                    .try_into()
                    .expect("FourBox weights should have 4 elements");
                FourBoxGrid::with_weights(weights_arr)
            }
            None => FourBoxGrid::magicc_standard(),
        }
    }

    /// Create a HemisphericGrid using custom weights if configured, otherwise use defaults
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

    /// Set the variable schema for the model
    ///
    /// The schema defines all variables (including aggregates) that the model uses.
    /// When a schema is provided, the builder validates:
    /// - Component outputs are defined in the schema
    /// - Component inputs are defined in the schema or produced by other components
    /// - Units and grid types match between components and schema
    ///
    /// Variables defined in the schema but not produced by any component will
    /// be initialised to NaN.
    pub fn with_schema(&mut self, schema: VariableSchema) -> &mut Self {
        self.schema = Some(schema);
        self
    }

    /// Register a component with the builder
    pub fn with_component(&mut self, component: Arc<dyn Component + Send + Sync>) -> &mut Self {
        self.components.push(component);
        self
    }

    /// Supply exogenous data to be used by the model
    ///
    /// Any unneeded timeseries will be ignored.
    pub fn with_exogenous_variable(
        &mut self,
        name: &str,
        timeseries: Timeseries<FloatValue>,
    ) -> &mut Self {
        self.exogenous_variables.add_timeseries(
            name.to_string(),
            timeseries,
            VariableType::Exogenous,
        );
        self
    }

    /// Supply exogenous data to be used by the model
    ///
    /// Any unneeded timeseries will be ignored.
    pub fn with_exogenous_collection(&mut self, collection: TimeseriesCollection) -> &mut Self {
        self.exogenous_variables.extend(collection);
        self
    }

    /// Adds some state to the set of initial values
    ///
    /// These initial values are used to provide some initial values at `t_0`.
    /// Initial values are used for requirements which have a type of `RequirementType::State`.
    /// State variables read their value from the previous timestep in order to generate a new value
    /// for the next timestep.
    /// Building a model where any variables which have `RequirementType::State`, but
    /// do not have an initial value will result in an error.
    pub fn with_initial_values(
        &mut self,
        initial_values: HashMap<String, FloatValue>,
    ) -> &mut Self {
        for (name, value) in initial_values.into_iter() {
            self.initial_values.insert(name, value);
        }
        self
    }

    /// Specify the time axis that will be used by the model
    ///
    /// This time axis defines the time steps (including bounds) on which the model will be iterated.
    pub fn with_time_axis(&mut self, time_axis: TimeAxis) -> &mut Self {
        self.time_axis = Arc::new(time_axis);
        self
    }

    /// Validate a component's requirements against the schema
    ///
    /// Checks that:
    /// - All outputs are defined in the schema (as variables or aggregates)
    /// - All inputs are defined in the schema (as variables or aggregates)
    /// - Units match between component and schema
    /// - Grid types are compatible (allowing aggregation where valid)
    ///
    /// Returns a list of required grid transformations for mismatched grids.
    fn validate_component_against_schema(
        &self,
        schema: &VariableSchema,
        component_name: &str,
        inputs: &[RequirementDefinition],
        outputs: &[RequirementDefinition],
        endogenous: &HashMap<String, NodeIndex>,
    ) -> RSCMResult<Vec<RequiredTransformation>> {
        let mut transformations = Vec::new();

        // Validate outputs
        // Write-side: component produces finer grid than schema → aggregate before storage
        for output in outputs {
            // Check if output is defined in schema
            if !schema.contains(&output.name) {
                return Err(RSCMError::SchemaUndefinedOutput {
                    component: component_name.to_string(),
                    variable: output.name.clone(),
                    unit: output.unit.clone(),
                });
            }

            // Check unit matches
            if let Some(schema_unit) = schema.get_unit(&output.name) {
                if schema_unit != output.unit {
                    return Err(RSCMError::ComponentSchemaUnitMismatch {
                        variable: output.name.clone(),
                        component: component_name.to_string(),
                        component_unit: output.unit.clone(),
                        schema_unit: schema_unit.to_string(),
                    });
                }
            }

            // Check grid type compatibility
            if let Some(schema_grid) = schema.get_grid_type(&output.name) {
                if schema_grid != output.grid_type {
                    // Write-side: component grid can aggregate to schema grid?
                    // Component produces finer data → aggregation to schema is OK
                    if output.grid_type.can_aggregate_to(schema_grid) {
                        // Valid write-side aggregation needed
                        transformations.push(RequiredTransformation {
                            variable: output.name.clone(),
                            unit: output.unit.clone(),
                            source_grid: output.grid_type,
                            target_grid: schema_grid,
                            direction: TransformDirection::Write,
                        });
                    } else {
                        // Invalid: would require disaggregation (broadcast)
                        // Component produces coarser data than schema expects
                        return Err(RSCMError::GridTransformationNotSupported {
                            variable: output.name.clone(),
                            source_grid: output.grid_type.to_string(),
                            target_grid: schema_grid.to_string(),
                        });
                    }
                }
            }
        }

        // Validate inputs
        // Read-side: component wants coarser grid than schema → aggregate before read
        for input in inputs {
            // Skip empty links
            if input.requirement_type == RequirementType::EmptyLink {
                continue;
            }

            // Input is valid if:
            // 1. It's defined in the schema (as variable or aggregate), OR
            // 2. It's produced by another component (endogenous)
            if !schema.contains(&input.name) && !endogenous.contains_key(&input.name) {
                return Err(RSCMError::SchemaUndefinedInput {
                    component: component_name.to_string(),
                    variable: input.name.clone(),
                    unit: input.unit.clone(),
                });
            }

            // If it's in the schema, check unit and grid type compatibility
            if schema.contains(&input.name) {
                if let Some(schema_unit) = schema.get_unit(&input.name) {
                    if schema_unit != input.unit {
                        return Err(RSCMError::ComponentSchemaUnitMismatch {
                            variable: input.name.clone(),
                            component: component_name.to_string(),
                            component_unit: input.unit.clone(),
                            schema_unit: schema_unit.to_string(),
                        });
                    }
                }

                if let Some(schema_grid) = schema.get_grid_type(&input.name) {
                    if schema_grid != input.grid_type {
                        // Read-side: schema grid can aggregate to component grid?
                        // Schema has finer data → aggregation for component is OK
                        if schema_grid.can_aggregate_to(input.grid_type) {
                            // Valid read-side aggregation needed
                            transformations.push(RequiredTransformation {
                                variable: input.name.clone(),
                                unit: input.unit.clone(),
                                source_grid: schema_grid,
                                target_grid: input.grid_type,
                                direction: TransformDirection::Read,
                            });
                        } else {
                            // Invalid: would require disaggregation (broadcast)
                            // Component wants finer data than schema provides
                            return Err(RSCMError::GridTransformationNotSupported {
                                variable: input.name.clone(),
                                source_grid: schema_grid.to_string(),
                                target_grid: input.grid_type.to_string(),
                            });
                        }
                    }
                }
            }
        }

        Ok(transformations)
    }

    /// Builds the component graph for the registered components and creates a concrete model
    ///
    /// Returns an error if the component definitions are inconsistent.
    pub fn build(&self) -> RSCMResult<Model> {
        // todo: refactor once this is more stable
        let mut graph: CGraph = Graph::new();
        let mut endrogoneous: HashMap<String, NodeIndex> = HashMap::new();
        let mut exogenous: Vec<String> = vec![];
        let mut definitions: HashMap<String, VariableDefinition> = HashMap::new();
        // Track which component owns each variable for better error messages
        let mut variable_owners: HashMap<String, String> = HashMap::new();
        let initial_node = graph.add_node(Arc::new(NullComponent {}));

        for component in &self.components {
            let node = graph.add_node(component.clone());
            let mut has_dependencies = false;

            // Get component name from Debug implementation
            let component_name = format!("{:?}", component);
            // Extract just the type name (before the first '{' or ' ')
            let component_name = component_name
                .split(['{', ' ', '('])
                .next()
                .unwrap_or("UnknownComponent")
                .to_string();

            let requires = component.inputs();
            let provides = component.outputs();

            for requirement in requires {
                let existing_owner = variable_owners.get(&requirement.name).map(|s| s.as_str());
                verify_definition(
                    &mut definitions,
                    &requirement,
                    &component_name,
                    existing_owner,
                    self.schema.is_some(),
                )?;

                if let Some(&producer_node) = endrogoneous.get(&requirement.name) {
                    // Link to the node that provides the requirement
                    graph.add_edge(producer_node, node, requirement.clone());
                    has_dependencies = true;
                } else {
                    // Add a new variable that must be defined outside of the model
                    if !exogenous.contains(&requirement.name) {
                        exogenous.push(requirement.name.clone());
                    }
                }
            }

            if !has_dependencies {
                // If the node has no dependencies on other components,
                // create a link to the initial node.
                // This ensures that we have a single connected graph
                // There might be smarter ways to iterate over the nodes, but this is fine for now
                graph.add_edge(
                    initial_node,
                    node,
                    RequirementDefinition::new("", "", RequirementType::EmptyLink),
                );
            }

            for requirement in provides {
                let existing_owner = variable_owners.get(&requirement.name).map(|s| s.as_str());
                verify_definition(
                    &mut definitions,
                    &requirement,
                    &component_name,
                    existing_owner,
                    self.schema.is_some(),
                )?;

                // Track this component as the owner of this variable
                variable_owners.insert(requirement.name.clone(), component_name.clone());

                let val = endrogoneous.get(&requirement.name);

                match val {
                    None => {
                        endrogoneous.insert(requirement.name.clone(), node);
                    }
                    Some(node_index) => {
                        graph.add_edge(*node_index, node, requirement.clone());
                        endrogoneous.insert(requirement.name.clone(), node);
                    }
                }
            }
        }

        // Check that the component graph doesn't contain any loops
        assert!(!is_valid_graph(&graph));

        // Collect all required grid transformations
        let mut all_transformations: Vec<RequiredTransformation> = Vec::new();

        // Validate against schema if provided
        if let Some(schema) = &self.schema {
            // First validate the schema itself
            schema.validate()?;

            // Validate each component against the schema and collect transformations
            for component in &self.components {
                let component_name = format!("{:?}", component);
                let component_name = component_name
                    .split(['{', ' ', '('])
                    .next()
                    .unwrap_or("UnknownComponent")
                    .to_string();

                let component_transforms = self.validate_component_against_schema(
                    schema,
                    &component_name,
                    &component.inputs(),
                    &component.outputs(),
                    &endrogoneous,
                )?;
                all_transformations.extend(component_transforms);
            }

            // Add schema variables not produced by components to definitions (4.5)
            // These will be initialised to NaN
            for (name, var_def) in &schema.variables {
                if !definitions.contains_key(name) {
                    definitions.insert(
                        name.clone(),
                        VariableDefinition {
                            name: name.clone(),
                            unit: var_def.unit.clone(),
                            grid_type: var_def.grid_type,
                        },
                    );
                    // Mark as exogenous since it's not produced by any component
                    exogenous.push(name.clone());
                }
            }

            // Add aggregator components for each aggregate definition (5.1, 5.2, 5.3)
            // Process aggregates in topological order to handle chained aggregates
            let ordered_aggregates = schema.topological_order_aggregates();
            for agg_name in &ordered_aggregates {
                let agg_def = schema.get_aggregate(agg_name).unwrap();

                // Create the aggregator component (5.1)
                let aggregator = crate::schema::AggregatorComponent::from_definition(agg_def);

                // Add to graph (5.2)
                let agg_node = graph.add_node(Arc::new(aggregator.clone()));

                // Track the component name for variable ownership
                let agg_component_name = format!("Aggregator:{}", agg_name);
                variable_owners.insert(agg_name.clone(), agg_component_name.clone());

                // Add dependency edges from contributor sources to aggregator (5.3)
                let mut has_dependencies = false;
                for contributor in &agg_def.contributors {
                    // Find the node that produces this contributor
                    if let Some(&producer_node) = endrogoneous.get(contributor) {
                        // Add edge from producer to aggregator
                        graph.add_edge(
                            producer_node,
                            agg_node,
                            RequirementDefinition::with_grid(
                                contributor,
                                &agg_def.unit,
                                RequirementType::Input,
                                agg_def.grid_type,
                            ),
                        );
                        has_dependencies = true;
                    }
                    // If contributor is exogenous, the aggregator will read from the
                    // timeseries collection which will have been populated
                }

                // If aggregator has no component dependencies, link to initial node
                if !has_dependencies {
                    graph.add_edge(
                        initial_node,
                        agg_node,
                        RequirementDefinition::new("", "", RequirementType::EmptyLink),
                    );
                }

                // Register the aggregate output as endogenous
                endrogoneous.insert(agg_name.clone(), agg_node);

                // Add aggregate variable to definitions
                definitions.insert(
                    agg_name.clone(),
                    VariableDefinition {
                        name: agg_name.clone(),
                        unit: agg_def.unit.clone(),
                        grid_type: agg_def.grid_type,
                    },
                );
            }
        }

        // Store transformations for runtime grid auto-aggregation
        // Split into read and write transforms for efficient lookup during execution
        let mut read_transforms: HashMap<String, RequiredTransformation> = HashMap::new();
        let mut write_transforms: HashMap<String, RequiredTransformation> = HashMap::new();

        for transform in all_transformations {
            match transform.direction {
                TransformDirection::Read => {
                    read_transforms.insert(transform.variable.clone(), transform);
                }
                TransformDirection::Write => {
                    write_transforms.insert(transform.variable.clone(), transform);
                }
            }
        }

        // Create the timeseries collection using the information from the components
        let mut collection = TimeseriesCollection::new();
        for (name, definition) in definitions {
            assert_eq!(definition.name, name);

            if exogenous.contains(&name) {
                // Exogenous variable is expected to be supplied
                if self.initial_values.contains_key(&name) {
                    // An initial value was provided
                    let mut ts = Timeseries::new_empty_scalar(
                        self.time_axis.clone(),
                        definition.unit,
                        InterpolationStrategy::from(LinearSplineStrategy::new(true)),
                    );
                    ts.set(0, ScalarRegion::Global, self.initial_values[&name]);

                    // Note that timeseries that are initialised are defined as Endogenous
                    // all but the first time point come from the model.
                    // This could potentially be defined as a different VariableType if needed.
                    collection.add_timeseries(name, ts, VariableType::Endogenous)
                } else {
                    // Check if the timeseries is available in the provided exogenous variables
                    // then interpolate to the right timebase
                    let timeseries = self
                        .exogenous_variables
                        .get_data(&name)
                        .and_then(|data| data.as_scalar());

                    match timeseries {
                        Some(timeseries) => collection.add_timeseries(
                            name,
                            timeseries
                                .to_owned()
                                .interpolate_into(self.time_axis.clone()),
                            VariableType::Exogenous,
                        ),
                        None => {
                            // No exogenous data provided - create empty timeseries (all NaN)
                            // This is expected for schema variables without writers
                            match definition.grid_type {
                                GridType::Scalar => collection.add_timeseries(
                                    definition.name,
                                    Timeseries::new_empty_scalar(
                                        self.time_axis.clone(),
                                        definition.unit,
                                        InterpolationStrategy::from(LinearSplineStrategy::new(
                                            true,
                                        )),
                                    ),
                                    VariableType::Exogenous,
                                ),
                                GridType::FourBox => collection.add_four_box_timeseries(
                                    definition.name,
                                    crate::timeseries::GridTimeseries::new_empty(
                                        self.time_axis.clone(),
                                        self.create_four_box_grid(),
                                        definition.unit,
                                        InterpolationStrategy::from(LinearSplineStrategy::new(
                                            true,
                                        )),
                                    ),
                                    VariableType::Exogenous,
                                ),
                                GridType::Hemispheric => collection.add_hemispheric_timeseries(
                                    definition.name,
                                    crate::timeseries::GridTimeseries::new_empty(
                                        self.time_axis.clone(),
                                        self.create_hemispheric_grid(),
                                        definition.unit,
                                        InterpolationStrategy::from(LinearSplineStrategy::new(
                                            true,
                                        )),
                                    ),
                                    VariableType::Exogenous,
                                ),
                            }
                        }
                    }
                }
            } else {
                // Create a placeholder for data that will be generated by the model
                // If there's a write transform, use the target grid type (schema's type)
                // instead of the component's declared output type
                let storage_grid_type = write_transforms
                    .get(&name)
                    .map(|t| t.target_grid)
                    .unwrap_or(definition.grid_type);

                match storage_grid_type {
                    GridType::Scalar => collection.add_timeseries(
                        definition.name,
                        Timeseries::new_empty_scalar(
                            self.time_axis.clone(),
                            definition.unit,
                            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
                        ),
                        VariableType::Endogenous,
                    ),
                    GridType::FourBox => collection.add_four_box_timeseries(
                        definition.name,
                        crate::timeseries::GridTimeseries::new_empty(
                            self.time_axis.clone(),
                            self.create_four_box_grid(),
                            definition.unit,
                            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
                        ),
                        VariableType::Endogenous,
                    ),
                    GridType::Hemispheric => collection.add_hemispheric_timeseries(
                        definition.name,
                        crate::timeseries::GridTimeseries::new_empty(
                            self.time_axis.clone(),
                            self.create_hemispheric_grid(),
                            definition.unit,
                            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
                        ),
                        VariableType::Endogenous,
                    ),
                }
            }
        }

        // Add the components to the graph
        Ok(Model::with_transforms(
            graph,
            initial_node,
            collection,
            self.time_axis.clone(),
            self.grid_weights.clone(),
            read_transforms,
            write_transforms,
        ))
    }
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A coupled set of components that are solved on a common time axis.
///
/// These components are solved over time steps defined by the ['time_axis'].
/// Components may pass state between themselves.
/// Each component may require information from other components to be solved (endogenous) or
/// predefined data (exogenous).
///
/// For example, a component to calculate the Effective Radiative Forcing(ERF) of CO_2 may
/// require CO_2 concentrations as input state and provide CO_2 ERF.
/// The component is agnostic about where/how that state is defined.
/// If the model has no components which provide CO_2 concentrations,
/// then a CO_2 concentration timeseries must be defined externally.
/// If the model also contains a carbon cycle component which produced CO_2 concentrations,
/// then the ERF component will be solved after the carbon cycle model.
#[derive(Debug, Serialize, Deserialize)]
pub struct Model {
    /// A directed graph with components as nodes and the edges defining the state dependencies
    /// between nodes.
    /// This graph is traversed on every time step to ensure that any state dependencies are
    /// solved before another component needs the state.
    components: CGraph,
    /// The base node of the graph from where to begin traversing.
    initial_node: NodeIndex,
    /// The model state
    ///
    /// Variable names within the model are unique and these variable names are used by
    /// components to request state.
    collection: TimeseriesCollection,
    time_axis: Arc<TimeAxis>,
    time_index: usize,
    /// Custom weights for grid aggregation, keyed by grid type
    ///
    /// Used for grid transformations during model execution.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    grid_weights: HashMap<GridType, Vec<f64>>,
    /// Read-side transformations: variable name -> transformation needed when component reads
    ///
    /// When a component reads a variable at a coarser grid than the schema declares,
    /// this maps the variable name to the transformation needed (e.g., FourBox -> Scalar).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    read_transforms: HashMap<String, RequiredTransformation>,
    /// Write-side transformations: variable name -> transformation needed when component writes
    ///
    /// When a component writes a variable at a finer grid than the schema declares,
    /// this maps the variable name to the transformation needed (e.g., FourBox -> Scalar).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    write_transforms: HashMap<String, RequiredTransformation>,
}

impl Model {
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

    /// Create a new Model with custom grid weights
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

    /// Create a new Model with grid weights and transformations
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
        Self {
            components,
            initial_node,
            collection,
            time_axis,
            time_index: 0,
            grid_weights,
            read_transforms,
            write_transforms,
        }
    }

    /// Get the configured grid weights
    ///
    /// Returns the custom weights if configured, or None if using defaults.
    pub fn get_grid_weights(&self, grid_type: GridType) -> Option<&Vec<f64>> {
        self.grid_weights.get(&grid_type)
    }

    /// Get all required transformations for introspection
    ///
    /// Returns a vector of all required transformations, both read-side and write-side.
    /// This is useful for debugging and understanding what grid aggregations will occur.
    pub fn required_transformations(&self) -> Vec<&RequiredTransformation> {
        self.read_transforms
            .values()
            .chain(self.write_transforms.values())
            .collect()
    }

    /// Get read-side transformations
    ///
    /// These transformations aggregate data when components read variables at coarser
    /// resolutions than the schema declares.
    pub fn read_transforms(&self) -> &HashMap<String, RequiredTransformation> {
        &self.read_transforms
    }

    /// Get write-side transformations
    ///
    /// These transformations aggregate data when components write variables at finer
    /// resolutions than the schema declares.
    pub fn write_transforms(&self) -> &HashMap<String, RequiredTransformation> {
        &self.write_transforms
    }

    /// Gets the time value at the current step
    pub fn current_time(&self) -> Time {
        self.time_axis.at(self.time_index).unwrap()
    }
    pub fn current_time_bounds(&self) -> (Time, Time) {
        self.time_axis.at_bounds(self.time_index).unwrap()
    }

    /// Solve a single component for the current timestep
    ///
    /// The updated state from the component is then pushed into the model's timeseries collection
    /// to be later used by other components.
    /// The output state defines the values at the next time index as it represents the state
    /// at the start of the next timestep.
    fn step_model_component(&mut self, component: C) {
        // Build transform context for read-side aggregation
        let input_names = component.input_names();
        let input_state = if self.read_transforms.is_empty() {
            extract_state(&self.collection, input_names, self.current_time())
        } else {
            // Build transform context with only the transforms relevant to this component's inputs
            let mut read_transform_info = HashMap::new();
            for name in &input_names {
                if let Some(transform) = self.read_transforms.get(name) {
                    read_transform_info.insert(
                        name.clone(),
                        ReadTransformInfo {
                            source_grid: transform.source_grid,
                            weights: self.grid_weights.get(&transform.source_grid).cloned(),
                        },
                    );
                }
            }

            if read_transform_info.is_empty() {
                extract_state(&self.collection, input_names, self.current_time())
            } else {
                let context = TransformContext {
                    read_transforms: read_transform_info,
                };
                extract_state_with_transforms(
                    &self.collection,
                    input_names,
                    self.current_time(),
                    context,
                )
            }
        };

        let (start, end) = self.current_time_bounds();

        let result = component.solve(start, end, &input_state);

        match result {
            Ok(output_state) => {
                for (key, state_value) in output_state.iter() {
                    let data = self.collection.get_data_mut(key).unwrap();

                    // Apply write-side transformation if needed (component produces finer grid
                    // than schema expects)
                    let final_value = if let Some(transform) = self.write_transforms.get(key) {
                        let weights = self.grid_weights.get(&transform.source_grid);
                        match aggregate_state_value(
                            state_value,
                            transform.source_grid,
                            transform.target_grid,
                            weights,
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                println!("Write-side aggregation failed for {}: {}", key, e);
                                continue;
                            }
                        }
                    } else {
                        state_value.clone()
                    };

                    // The next time index is used as this output state represents the value of a
                    // variable at the end of the current time step.
                    // This is the same as the start of the next timestep.
                    let result = match &final_value {
                        StateValue::Scalar(v) => data.set_scalar(key, self.time_index + 1, *v),
                        StateValue::FourBox(slice) => {
                            data.set_four_box(key, self.time_index + 1, slice)
                        }
                        StateValue::Hemispheric(slice) => {
                            data.set_hemispheric(key, self.time_index + 1, slice)
                        }
                    };
                    if let Err(e) = result {
                        println!("Failed to set output {}: {}", key, e);
                    }
                }
            }
            Err(err) => {
                println!("Solving failed: {}", err)
            }
        }
    }

    /// Step the model forward a step by solving each component for the current time step.
    ///
    /// A breadth-first search across the component graph starting at the initial node
    /// will solve the components in a way that ensures any models with dependencies are solved
    /// after the dependent component is first solved.
    fn step_model(&mut self) {
        let mut bfs = Bfs::new(&self.components, self.initial_node);
        while let Some(nx) = bfs.next(&self.components) {
            let c = self.components.index(nx);
            self.step_model_component(c.clone())
        }
    }

    /// Steps the model forward one time step
    ///
    /// This solves the current time step and then updates the index.
    pub fn step(&mut self) {
        assert!(self.time_index < self.time_axis.len() - 1);
        self.step_model();

        self.time_index += 1;
    }

    /// Steps the model until the end of the time axis
    pub fn run(&mut self) {
        while self.time_index < self.time_axis.len() - 1 {
            self.step();
        }
    }

    /// Create a diagram the represents the component graph
    ///
    /// Useful for debugging
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

    /// Returns true if the model has no more time steps to process
    pub fn finished(&self) -> bool {
        self.time_index == self.time_axis.len() - 1
    }

    pub fn timeseries(&self) -> &TimeseriesCollection {
        &self.collection
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example_components::{TestComponent, TestComponentParameters};
    use crate::interpolate::strategies::PreviousStrategy;
    use is_close::is_close;
    use numpy::array;
    use numpy::ndarray::{Array, Axis};
    use std::iter::zip;

    fn get_emissions() -> Timeseries<FloatValue> {
        use crate::spatial::ScalarGrid;
        let values = array![0.0, 10.0].insert_axis(Axis(1));
        Timeseries::new(
            values,
            Arc::new(TimeAxis::from_bounds(array![1800.0, 1850.0, 2100.0])),
            ScalarGrid,
            "GtC / yr".to_string(),
            InterpolationStrategy::from(PreviousStrategy::new(true)),
        )
    }

    #[test]
    fn step() {
        let time_axis = TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0));
        let mut model = ModelBuilder::new()
            .with_time_axis(time_axis)
            .with_component(Arc::new(TestComponent::from_parameters(
                TestComponentParameters {
                    conversion_factor: 0.5,
                },
            )))
            .with_exogenous_variable("Emissions|CO2", get_emissions())
            .build()
            .unwrap();

        assert_eq!(model.time_index, 0);
        model.step();
        model.step();
        assert_eq!(model.time_index, 2);
        assert_eq!(model.current_time(), 2022.0);
        model.run();
        assert_eq!(model.time_index, 4);
        assert!(model.finished());

        let concentrations = model
            .collection
            .get_data("Concentrations|CO2")
            .and_then(|data| data.as_scalar())
            .unwrap();

        println!("{:?}", concentrations.values());

        // The first value for an endogenous timeseries without a y0 value is NaN.
        // This is because the values in the timeseries represents the state at the start
        // of a time step.
        // Since the values from t-1 aren't known we can't solve for y0
        assert!(concentrations.at(0, ScalarRegion::Global).unwrap().is_nan());
        let mut iter = concentrations.values().into_iter();
        iter.next(); // Skip the first value
        assert!(iter.all(|x| !x.is_nan()));
    }

    #[test]
    fn dot() {
        let time_axis = TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0));
        let model = ModelBuilder::new()
            .with_time_axis(time_axis)
            .with_component(Arc::new(TestComponent::from_parameters(
                TestComponentParameters {
                    conversion_factor: 0.5,
                },
            )))
            .with_exogenous_variable("Emissions|CO2", get_emissions())
            .build()
            .unwrap();

        let exp = r#"digraph {
    0 [ label = "NullComponent"]
    1 [ label = "TestComponent { parameters: TestComponentParameters { conversion_factor: 0.5 } }"]
    0 -> 1 [ label = ""]
}
"#;

        let res = format!("{:?}", model.as_dot());
        assert_eq!(res, exp);
    }

    #[test]
    fn serialise_and_deserialise_model() {
        let mut model = ModelBuilder::new()
            .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
            .with_component(Arc::new(TestComponent::from_parameters(
                TestComponentParameters {
                    conversion_factor: 0.5,
                },
            )))
            .with_exogenous_variable("Emissions|CO2", get_emissions())
            .build()
            .unwrap();

        model.step();

        let serialised = serde_json::to_string_pretty(&model).unwrap();
        println!("Pretty JSON");
        println!("{}", serialised);
        let serialised = toml::to_string(&model).unwrap();
        println!("TOML");
        println!("{}", serialised);

        let expected = r#"initial_node = 0
time_index = 1

[components]
node_holes = []
edge_property = "directed"
edges = [[0, 1, { name = "", unit = "", requirement_type = "EmptyLink", grid_type = "Scalar" }]]

[[components.nodes]]
type = "NullComponent"

[[components.nodes]]
type = "TestComponent"

[components.nodes.parameters]
conversion_factor = 0.5

[[collection.timeseries]]
name = "Concentrations|CO2"
variable_type = "Endogenous"

[collection.timeseries.data.Scalar]
units = "ppm"
latest = 1
interpolation_strategy = "Linear"

[collection.timeseries.data.Scalar.values]
v = 1
dim = [5, 1]
data = [nan, 5.0, nan, nan, nan]

[collection.timeseries.data.Scalar.time_axis.bounds]
v = 1
dim = [6]
data = [2020.0, 2021.0, 2022.0, 2023.0, 2024.0, 2025.0]

[[collection.timeseries]]
name = "Emissions|CO2"
variable_type = "Exogenous"

[collection.timeseries.data.Scalar]
units = "GtC / yr"
latest = 4
interpolation_strategy = "Previous"

[collection.timeseries.data.Scalar.values]
v = 1
dim = [5, 1]
data = [10.0, 10.0, 10.0, 10.0, 10.0]

[collection.timeseries.data.Scalar.time_axis.bounds]
v = 1
dim = [6]
data = [2020.0, 2021.0, 2022.0, 2023.0, 2024.0, 2025.0]

[time_axis.bounds]
v = 1
dim = [6]
data = [2020.0, 2021.0, 2022.0, 2023.0, 2024.0, 2025.0]
"#;

        assert_eq!(serialised, expected);

        let deserialised = toml::from_str::<Model>(&serialised).unwrap();

        assert!(zip(
            model
                .collection
                .get_data("Emissions|CO2")
                .and_then(|data| data.as_scalar())
                .unwrap()
                .values(),
            deserialised
                .collection
                .get_data("Emissions|CO2")
                .and_then(|data| data.as_scalar())
                .unwrap()
                .values()
        )
        .all(|(x0, x1)| { is_close!(*x0, *x1) || (x0.is_nan() && x0.is_nan()) }));

        assert_eq!(model.current_time_bounds(), (2021.0, 2022.0));
        assert_eq!(deserialised.current_time_bounds(), (2021.0, 2022.0));
    }

    mod grid_validation_tests {
        use super::*;

        /// A component that produces a FourBox output
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct FourBoxProducer;

        #[typetag::serde]
        impl Component for FourBoxProducer {
            fn definitions(&self) -> Vec<RequirementDefinition> {
                vec![RequirementDefinition::four_box_output("Temperature", "K")]
            }

            fn solve(
                &self,
                _t_current: Time,
                _t_next: Time,
                _input_state: &InputState,
            ) -> RSCMResult<OutputState> {
                use crate::state::{FourBoxSlice, StateValue};
                let mut output = OutputState::new();
                output.insert(
                    "Temperature".to_string(),
                    StateValue::FourBox(FourBoxSlice::from_array([288.0, 290.0, 287.0, 285.0])),
                );
                Ok(output)
            }
        }

        /// A component that expects a Scalar input
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct ScalarConsumer;

        #[typetag::serde]
        impl Component for ScalarConsumer {
            fn definitions(&self) -> Vec<RequirementDefinition> {
                vec![
                    RequirementDefinition::scalar_input("Temperature", "K"),
                    RequirementDefinition::scalar_output("Result", "W / m^2"),
                ]
            }

            fn solve(
                &self,
                _t_current: Time,
                _t_next: Time,
                input_state: &InputState,
            ) -> RSCMResult<OutputState> {
                use crate::state::StateValue;
                let temp = input_state.get_scalar_window("Temperature").at_start();
                let mut output = OutputState::new();
                output.insert("Result".to_string(), StateValue::Scalar(temp * 2.0));
                Ok(output)
            }
        }

        /// A component that expects a FourBox input (compatible with FourBoxProducer)
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct FourBoxConsumer;

        #[typetag::serde]
        impl Component for FourBoxConsumer {
            fn definitions(&self) -> Vec<RequirementDefinition> {
                vec![
                    RequirementDefinition::four_box_input("Temperature", "K"),
                    RequirementDefinition::scalar_output("GlobalTemperature", "K"),
                ]
            }

            fn solve(
                &self,
                _t_current: Time,
                _t_next: Time,
                input_state: &InputState,
            ) -> RSCMResult<OutputState> {
                use crate::state::StateValue;
                let temp = input_state
                    .get_four_box_window("Temperature")
                    .current_global();
                let mut output = OutputState::new();
                output.insert("GlobalTemperature".to_string(), StateValue::Scalar(temp));
                Ok(output)
            }
        }

        #[test]
        fn test_grid_type_mismatch_returns_error() {
            // This should return an error because FourBoxProducer outputs FourBox
            // but ScalarConsumer expects Scalar
            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_component(Arc::new(FourBoxProducer))
                .with_component(Arc::new(ScalarConsumer))
                .build();

            assert!(result.is_err());
            let err = result.unwrap_err();
            let err_msg = err.to_string();
            assert!(err_msg.contains("Grid type mismatch for variable 'Temperature'"));
            assert!(err_msg.contains("FourBoxProducer"));
            assert!(err_msg.contains("ScalarConsumer"));
            assert!(err_msg.contains("FourBox"));
            assert!(err_msg.contains("Scalar"));
        }

        #[test]
        fn test_matching_grid_types_ok() {
            // This should work because both use FourBox for Temperature
            let _model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_component(Arc::new(FourBoxProducer))
                .with_component(Arc::new(FourBoxConsumer))
                .build()
                .unwrap();
        }
    }

    mod schema_validation_tests {
        use super::grid_validation_tests::{FourBoxConsumer, FourBoxProducer};
        use super::*;
        use crate::schema::{AggregateOp, VariableSchema};

        #[test]
        fn test_model_with_valid_schema() {
            // Schema that matches component requirements
            let schema = VariableSchema::new()
                .variable("Emissions|CO2", "GtCO2")
                .variable("Concentrations|CO2", "ppm");

            let _model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(TestComponent::from_parameters(
                    TestComponentParameters {
                        conversion_factor: 0.5,
                    },
                )))
                .with_exogenous_variable("Emissions|CO2", get_emissions())
                .build()
                .unwrap();
        }

        #[test]
        fn test_schema_rejects_undefined_output() {
            // Schema missing the output variable
            let schema = VariableSchema::new().variable("Emissions|CO2", "GtCO2");
            // Missing "Concentrations|CO2"

            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(TestComponent::from_parameters(
                    TestComponentParameters {
                        conversion_factor: 0.5,
                    },
                )))
                .with_exogenous_variable("Emissions|CO2", get_emissions())
                .build();

            assert!(result.is_err());
            let err = result.unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("Concentrations|CO2"),
                "Error should mention missing variable: {}",
                msg
            );
            assert!(
                msg.contains("not defined in the schema"),
                "Error should indicate schema issue: {}",
                msg
            );
        }

        #[test]
        fn test_schema_rejects_undefined_input() {
            // Schema missing the input variable (and no component produces it)
            let schema = VariableSchema::new().variable("Concentrations|CO2", "ppm");
            // Missing "Emissions|CO2"

            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(TestComponent::from_parameters(
                    TestComponentParameters {
                        conversion_factor: 0.5,
                    },
                )))
                .build();

            assert!(result.is_err());
            let err = result.unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("Emissions|CO2"),
                "Error should mention missing variable: {}",
                msg
            );
        }

        #[test]
        fn test_schema_rejects_unit_mismatch() {
            // Schema with wrong unit for output variable
            let schema = VariableSchema::new()
                .variable("Emissions|CO2", "GtCO2")
                .variable("Concentrations|CO2", "GtC"); // Wrong unit - should be "ppm"

            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(TestComponent::from_parameters(
                    TestComponentParameters {
                        conversion_factor: 0.5,
                    },
                )))
                .with_exogenous_variable("Emissions|CO2", get_emissions())
                .build();

            assert!(result.is_err());
            let err = result.unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("Unit mismatch"),
                "Error should indicate unit mismatch: {}",
                msg
            );
            assert!(
                msg.contains("Concentrations|CO2"),
                "Error should mention the variable: {}",
                msg
            );
        }

        #[test]
        fn test_schema_rejects_disaggregation_on_read() {
            // Schema has Scalar temperature, but FourBoxConsumer wants FourBox input
            // This is a disaggregation (broadcast) attempt and should fail
            let schema = VariableSchema::new()
                .variable_with_grid("Temperature", "K", GridType::Scalar)
                .variable("GlobalTemperature", "K");

            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(FourBoxProducer)) // Writes FourBox, aggregated to Scalar (OK)
                .with_component(Arc::new(FourBoxConsumer)) // Reads FourBox from Scalar schema (ERROR)
                .build();

            assert!(result.is_err());
            let err = result.unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("Grid transformation not supported"),
                "Error should indicate unsupported transformation: {}",
                msg
            );
            assert!(
                msg.contains("Scalar") && msg.contains("FourBox"),
                "Error should mention the grid types: {}",
                msg
            );
        }

        #[test]
        fn test_schema_with_aggregate_validates() {
            // Schema with an aggregate definition
            let schema = VariableSchema::new()
                .variable("Emissions|CO2", "GtCO2")
                .variable("Concentrations|CO2", "ppm")
                .aggregate("Total Concentrations", "ppm", AggregateOp::Sum)
                .from("Concentrations|CO2")
                .build();

            let _model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(TestComponent::from_parameters(
                    TestComponentParameters {
                        conversion_factor: 0.5,
                    },
                )))
                .with_exogenous_variable("Emissions|CO2", get_emissions())
                .build()
                .unwrap();
        }

        #[test]
        fn test_schema_creates_nan_for_unwritten_variables() {
            // Schema has a variable that no component writes to
            let schema = VariableSchema::new()
                .variable("Emissions|CO2", "GtCO2")
                .variable("Concentrations|CO2", "ppm")
                .variable("Concentrations|CH4", "ppb"); // No component writes this

            let model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(TestComponent::from_parameters(
                    TestComponentParameters {
                        conversion_factor: 0.5,
                    },
                )))
                .with_exogenous_variable("Emissions|CO2", get_emissions())
                .build()
                .unwrap();

            // The CH4 variable should exist but be all NaN
            let ch4 = model
                .timeseries()
                .get_data("Concentrations|CH4")
                .and_then(|d| d.as_scalar());
            assert!(
                ch4.is_some(),
                "CH4 timeseries should exist even though no component writes it"
            );
            let ch4 = ch4.unwrap();
            assert!(
                ch4.values().iter().all(|v| v.is_nan()),
                "All CH4 values should be NaN since no component writes to it"
            );
        }

        #[test]
        fn test_schema_invalid_aggregate_fails() {
            // Schema with invalid aggregate (circular dependency)
            let mut schema = VariableSchema::new();
            schema.aggregates.insert(
                "A".to_string(),
                crate::schema::AggregateDefinition {
                    name: "A".to_string(),
                    unit: "units".to_string(),
                    grid_type: GridType::Scalar,
                    operation: AggregateOp::Sum,
                    contributors: vec!["B".to_string()],
                },
            );
            schema.aggregates.insert(
                "B".to_string(),
                crate::schema::AggregateDefinition {
                    name: "B".to_string(),
                    unit: "units".to_string(),
                    grid_type: GridType::Scalar,
                    operation: AggregateOp::Sum,
                    contributors: vec!["A".to_string()],
                },
            );

            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .build();

            assert!(result.is_err());
            let err = result.unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("Circular dependency"),
                "Error should indicate circular dependency: {}",
                msg
            );
        }

        #[test]
        fn test_model_without_schema_still_works() {
            // Ensure models without schema still work as before
            let _model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_component(Arc::new(TestComponent::from_parameters(
                    TestComponentParameters {
                        conversion_factor: 0.5,
                    },
                )))
                .with_exogenous_variable("Emissions|CO2", get_emissions())
                .build()
                .unwrap();
        }
    }

    mod aggregate_execution_tests {
        use super::*;
        use crate::schema::{AggregateOp, VariableSchema};
        use crate::spatial::ScalarRegion;
        use crate::state::StateValue;

        /// A simple component that produces ERF|CO2
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct CO2ERFComponent {
            forcing_per_ppm: f64,
        }

        #[typetag::serde]
        impl Component for CO2ERFComponent {
            fn definitions(&self) -> Vec<RequirementDefinition> {
                vec![
                    RequirementDefinition::scalar_input("Concentrations|CO2", "ppm"),
                    RequirementDefinition::scalar_output("ERF|CO2", "W/m^2"),
                ]
            }

            fn solve(
                &self,
                _t_current: Time,
                _t_next: Time,
                input_state: &InputState,
            ) -> RSCMResult<OutputState> {
                let conc = input_state
                    .get_scalar_window("Concentrations|CO2")
                    .at_start();
                let mut output = OutputState::new();
                output.insert(
                    "ERF|CO2".to_string(),
                    StateValue::Scalar(conc * self.forcing_per_ppm),
                );
                Ok(output)
            }
        }

        /// A simple component that produces ERF|CH4
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct CH4ERFComponent {
            forcing_per_ppb: f64,
        }

        #[typetag::serde]
        impl Component for CH4ERFComponent {
            fn definitions(&self) -> Vec<RequirementDefinition> {
                vec![
                    RequirementDefinition::scalar_input("Concentrations|CH4", "ppb"),
                    RequirementDefinition::scalar_output("ERF|CH4", "W/m^2"),
                ]
            }

            fn solve(
                &self,
                _t_current: Time,
                _t_next: Time,
                input_state: &InputState,
            ) -> RSCMResult<OutputState> {
                let conc = input_state
                    .get_scalar_window("Concentrations|CH4")
                    .at_start();
                let mut output = OutputState::new();
                output.insert(
                    "ERF|CH4".to_string(),
                    StateValue::Scalar(conc * self.forcing_per_ppb),
                );
                Ok(output)
            }
        }

        fn get_co2_concentrations() -> Timeseries<FloatValue> {
            use crate::spatial::ScalarGrid;
            let values = array![280.0, 400.0].insert_axis(Axis(1));
            Timeseries::new(
                values,
                Arc::new(TimeAxis::from_bounds(array![1800.0, 1850.0, 2100.0])),
                ScalarGrid,
                "ppm".to_string(),
                InterpolationStrategy::from(PreviousStrategy::new(true)),
            )
        }

        fn get_ch4_concentrations() -> Timeseries<FloatValue> {
            use crate::spatial::ScalarGrid;
            let values = array![700.0, 1800.0].insert_axis(Axis(1));
            Timeseries::new(
                values,
                Arc::new(TimeAxis::from_bounds(array![1800.0, 1850.0, 2100.0])),
                ScalarGrid,
                "ppb".to_string(),
                InterpolationStrategy::from(PreviousStrategy::new(true)),
            )
        }

        #[test]
        fn test_aggregate_sum_execution() {
            // Schema with aggregate summing two ERF components
            let schema = VariableSchema::new()
                .variable("Concentrations|CO2", "ppm")
                .variable("Concentrations|CH4", "ppb")
                .variable("ERF|CO2", "W/m^2")
                .variable("ERF|CH4", "W/m^2")
                .aggregate("ERF|Total", "W/m^2", AggregateOp::Sum)
                .from("ERF|CO2")
                .from("ERF|CH4")
                .build();

            let mut model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(CO2ERFComponent {
                    forcing_per_ppm: 0.01, // 1 W/m^2 per 100 ppm
                }))
                .with_component(Arc::new(CH4ERFComponent {
                    forcing_per_ppb: 0.001, // 1 W/m^2 per 1000 ppb
                }))
                .with_exogenous_variable("Concentrations|CO2", get_co2_concentrations())
                .with_exogenous_variable("Concentrations|CH4", get_ch4_concentrations())
                .build()
                .unwrap();

            // Run the model
            model.run();

            // Check that the aggregate was computed
            let total_erf = model
                .timeseries()
                .get_data("ERF|Total")
                .and_then(|d| d.as_scalar())
                .expect("ERF|Total should exist");

            // At 2021+ (after first step):
            // CO2: 400 ppm * 0.01 = 4.0 W/m^2
            // CH4: 1800 ppb * 0.001 = 1.8 W/m^2
            // Total: 5.8 W/m^2
            let value = total_erf.at(1, ScalarRegion::Global).unwrap();
            assert!(
                (value - 5.8).abs() < 1e-10,
                "ERF|Total should be 5.8, got {}",
                value
            );
        }

        #[test]
        fn test_aggregate_mean_execution() {
            // Schema with mean aggregate
            let schema = VariableSchema::new()
                .variable("Concentrations|CO2", "ppm")
                .variable("Concentrations|CH4", "ppb")
                .variable("ERF|CO2", "W/m^2")
                .variable("ERF|CH4", "W/m^2")
                .aggregate("ERF|Mean", "W/m^2", AggregateOp::Mean)
                .from("ERF|CO2")
                .from("ERF|CH4")
                .build();

            let mut model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(CO2ERFComponent {
                    forcing_per_ppm: 0.01,
                }))
                .with_component(Arc::new(CH4ERFComponent {
                    forcing_per_ppb: 0.001,
                }))
                .with_exogenous_variable("Concentrations|CO2", get_co2_concentrations())
                .with_exogenous_variable("Concentrations|CH4", get_ch4_concentrations())
                .build()
                .unwrap();

            model.run();

            let mean_erf = model
                .timeseries()
                .get_data("ERF|Mean")
                .and_then(|d| d.as_scalar())
                .expect("ERF|Mean should exist");

            // Mean of 4.0 and 1.8 = 2.9 W/m^2
            let value = mean_erf.at(1, ScalarRegion::Global).unwrap();
            assert!(
                (value - 2.9).abs() < 1e-10,
                "ERF|Mean should be 2.9, got {}",
                value
            );
        }

        #[test]
        fn test_aggregate_weighted_execution() {
            // Schema with weighted aggregate (80% CO2, 20% CH4)
            let schema = VariableSchema::new()
                .variable("Concentrations|CO2", "ppm")
                .variable("Concentrations|CH4", "ppb")
                .variable("ERF|CO2", "W/m^2")
                .variable("ERF|CH4", "W/m^2")
                .aggregate(
                    "ERF|Weighted",
                    "W/m^2",
                    AggregateOp::Weighted(vec![0.8, 0.2]),
                )
                .from("ERF|CO2")
                .from("ERF|CH4")
                .build();

            let mut model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(CO2ERFComponent {
                    forcing_per_ppm: 0.01,
                }))
                .with_component(Arc::new(CH4ERFComponent {
                    forcing_per_ppb: 0.001,
                }))
                .with_exogenous_variable("Concentrations|CO2", get_co2_concentrations())
                .with_exogenous_variable("Concentrations|CH4", get_ch4_concentrations())
                .build()
                .unwrap();

            model.run();

            let weighted_erf = model
                .timeseries()
                .get_data("ERF|Weighted")
                .and_then(|d| d.as_scalar())
                .expect("ERF|Weighted should exist");

            // Weighted: 4.0 * 0.8 + 1.8 * 0.2 = 3.2 + 0.36 = 3.56 W/m^2
            let value = weighted_erf.at(1, ScalarRegion::Global).unwrap();
            assert!(
                (value - 3.56).abs() < 1e-10,
                "ERF|Weighted should be 3.56, got {}",
                value
            );
        }

        #[test]
        fn test_aggregate_with_nan_contributor() {
            // Schema where one contributor has no writer (all NaN)
            let schema = VariableSchema::new()
                .variable("Concentrations|CO2", "ppm")
                .variable("ERF|CO2", "W/m^2")
                .variable("ERF|N2O", "W/m^2") // No component writes this
                .aggregate("ERF|Total", "W/m^2", AggregateOp::Sum)
                .from("ERF|CO2")
                .from("ERF|N2O")
                .build();

            let mut model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(CO2ERFComponent {
                    forcing_per_ppm: 0.01,
                }))
                .with_exogenous_variable("Concentrations|CO2", get_co2_concentrations())
                .build()
                .unwrap();

            model.run();

            let total_erf = model
                .timeseries()
                .get_data("ERF|Total")
                .and_then(|d| d.as_scalar())
                .expect("ERF|Total should exist");

            // ERF|N2O is NaN, so Sum should just be ERF|CO2 = 4.0 W/m^2
            let value = total_erf.at(1, ScalarRegion::Global).unwrap();
            assert!(
                (value - 4.0).abs() < 1e-10,
                "ERF|Total should be 4.0 (NaN excluded), got {}",
                value
            );
        }

        #[test]
        fn test_chained_aggregates_execution() {
            // Schema with chained aggregates: Total depends on GHG, GHG depends on CO2+CH4
            let schema = VariableSchema::new()
                .variable("Concentrations|CO2", "ppm")
                .variable("Concentrations|CH4", "ppb")
                .variable("ERF|CO2", "W/m^2")
                .variable("ERF|CH4", "W/m^2")
                .variable("ERF|Other", "W/m^2") // Will be NaN
                .aggregate("ERF|GHG", "W/m^2", AggregateOp::Sum)
                .from("ERF|CO2")
                .from("ERF|CH4")
                .build()
                .aggregate("ERF|Total", "W/m^2", AggregateOp::Sum)
                .from("ERF|GHG")
                .from("ERF|Other")
                .build();

            let mut model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(CO2ERFComponent {
                    forcing_per_ppm: 0.01,
                }))
                .with_component(Arc::new(CH4ERFComponent {
                    forcing_per_ppb: 0.001,
                }))
                .with_exogenous_variable("Concentrations|CO2", get_co2_concentrations())
                .with_exogenous_variable("Concentrations|CH4", get_ch4_concentrations())
                .build()
                .unwrap();

            model.run();

            // Check ERF|GHG = CO2 + CH4 = 4.0 + 1.8 = 5.8
            let ghg_erf = model
                .timeseries()
                .get_data("ERF|GHG")
                .and_then(|d| d.as_scalar())
                .expect("ERF|GHG should exist");
            let ghg_value = ghg_erf.at(1, ScalarRegion::Global).unwrap();
            assert!(
                (ghg_value - 5.8).abs() < 1e-10,
                "ERF|GHG should be 5.8, got {}",
                ghg_value
            );

            // Check ERF|Total = GHG + Other(NaN) = 5.8
            let total_erf = model
                .timeseries()
                .get_data("ERF|Total")
                .and_then(|d| d.as_scalar())
                .expect("ERF|Total should exist");
            let total_value = total_erf.at(1, ScalarRegion::Global).unwrap();
            assert!(
                (total_value - 5.8).abs() < 1e-10,
                "ERF|Total should be 5.8, got {}",
                total_value
            );
        }

        #[test]
        fn test_aggregate_appears_in_dot_graph() {
            let schema = VariableSchema::new()
                .variable("Concentrations|CO2", "ppm")
                .variable("ERF|CO2", "W/m^2")
                .aggregate("ERF|Total", "W/m^2", AggregateOp::Sum)
                .from("ERF|CO2")
                .build();

            let model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(CO2ERFComponent {
                    forcing_per_ppm: 0.01,
                }))
                .with_exogenous_variable("Concentrations|CO2", get_co2_concentrations())
                .build()
                .unwrap();

            let dot = format!("{:?}", model.as_dot());

            // The aggregator component should appear in the graph
            assert!(
                dot.contains("AggregatorComponent"),
                "Graph should contain AggregatorComponent: {}",
                dot
            );
        }
    }

    mod grid_weight_tests {
        use super::*;

        #[test]
        fn test_with_grid_weights_fourbox_valid() {
            let mut builder = ModelBuilder::new();
            builder.with_grid_weights(GridType::FourBox, vec![0.36, 0.14, 0.36, 0.14]);

            assert_eq!(
                builder.grid_weights.get(&GridType::FourBox),
                Some(&vec![0.36, 0.14, 0.36, 0.14])
            );
        }

        #[test]
        fn test_with_grid_weights_hemispheric_valid() {
            let mut builder = ModelBuilder::new();
            builder.with_grid_weights(GridType::Hemispheric, vec![0.6, 0.4]);

            assert_eq!(
                builder.grid_weights.get(&GridType::Hemispheric),
                Some(&vec![0.6, 0.4])
            );
        }

        #[test]
        #[should_panic(expected = "Cannot set weights for Scalar")]
        fn test_with_grid_weights_scalar_panics() {
            let mut builder = ModelBuilder::new();
            builder.with_grid_weights(GridType::Scalar, vec![1.0]);
        }

        #[test]
        #[should_panic(expected = "Weights length")]
        fn test_with_grid_weights_wrong_length_panics() {
            let mut builder = ModelBuilder::new();
            builder.with_grid_weights(GridType::FourBox, vec![0.5, 0.5]); // Wrong: 2 instead of 4
        }

        #[test]
        #[should_panic(expected = "Weights must sum to 1.0")]
        fn test_with_grid_weights_wrong_sum_panics() {
            let mut builder = ModelBuilder::new();
            builder.with_grid_weights(GridType::FourBox, vec![0.3, 0.3, 0.3, 0.3]);
            // Sum = 1.2
        }

        #[test]
        fn test_custom_weights_applied_to_fourbox_timeseries() {
            use crate::schema::VariableSchema;

            // Custom weights: different from default [0.25, 0.25, 0.25, 0.25]
            let custom_weights = vec![0.36, 0.14, 0.36, 0.14];

            let schema =
                VariableSchema::new().variable_with_grid("Temperature", "K", GridType::FourBox);

            let model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_grid_weights(GridType::FourBox, custom_weights.clone())
                .build()
                .unwrap();

            // Verify custom weights are stored in the Model
            let model_weights = model.get_grid_weights(GridType::FourBox);
            assert_eq!(model_weights, Some(&custom_weights));
        }

        #[test]
        fn test_model_get_grid_weights_returns_none_for_unset() {
            let model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .build()
                .unwrap();

            assert!(model.get_grid_weights(GridType::FourBox).is_none());
            assert!(model.get_grid_weights(GridType::Hemispheric).is_none());
            assert!(model.get_grid_weights(GridType::Scalar).is_none());
        }

        #[test]
        fn test_grid_weights_serialisation_roundtrip() {
            use crate::schema::VariableSchema;

            let custom_weights = vec![0.36, 0.14, 0.36, 0.14];
            let schema =
                VariableSchema::new().variable_with_grid("Temperature", "K", GridType::FourBox);

            let model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_grid_weights(GridType::FourBox, custom_weights.clone())
                .build()
                .unwrap();

            // Serialise and deserialise
            let serialised = toml::to_string(&model).unwrap();
            let deserialised: Model = toml::from_str(&serialised).unwrap();

            // Verify weights are preserved
            assert_eq!(
                deserialised.get_grid_weights(GridType::FourBox),
                Some(&custom_weights)
            );
        }

        #[test]
        fn test_empty_grid_weights_not_serialised() {
            let model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .build()
                .unwrap();

            let serialised = toml::to_string(&model).unwrap();

            // The grid_weights section should not appear
            assert!(
                !serialised.contains("grid_weights"),
                "Empty grid_weights should not be serialised: {}",
                serialised
            );
        }
    }

    mod relaxed_grid_validation_tests {
        use super::*;
        use crate::schema::VariableSchema;
        use crate::state::{FourBoxSlice, HemisphericSlice, StateValue};

        /// A component that produces FourBox output
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct FourBoxProducer {
            var_name: String,
        }

        #[typetag::serde]
        impl Component for FourBoxProducer {
            fn definitions(&self) -> Vec<RequirementDefinition> {
                vec![RequirementDefinition::four_box_output(&self.var_name, "K")]
            }

            fn solve(
                &self,
                _t_current: Time,
                _t_next: Time,
                _input_state: &InputState,
            ) -> RSCMResult<OutputState> {
                let mut output = OutputState::new();
                output.insert(
                    self.var_name.clone(),
                    StateValue::FourBox(FourBoxSlice::from_array([1.0, 2.0, 3.0, 4.0])),
                );
                Ok(output)
            }
        }

        /// A component that produces Hemispheric output
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct HemisphericProducer {
            var_name: String,
        }

        #[typetag::serde]
        impl Component for HemisphericProducer {
            fn definitions(&self) -> Vec<RequirementDefinition> {
                vec![RequirementDefinition::hemispheric_output(
                    &self.var_name,
                    "K",
                )]
            }

            fn solve(
                &self,
                _t_current: Time,
                _t_next: Time,
                _input_state: &InputState,
            ) -> RSCMResult<OutputState> {
                let mut output = OutputState::new();
                output.insert(
                    self.var_name.clone(),
                    StateValue::Hemispheric(HemisphericSlice::from_array([1.0, 2.0])),
                );
                Ok(output)
            }
        }

        /// A component that produces Scalar output
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct ScalarProducer {
            var_name: String,
        }

        #[typetag::serde]
        impl Component for ScalarProducer {
            fn definitions(&self) -> Vec<RequirementDefinition> {
                vec![RequirementDefinition::scalar_output(&self.var_name, "K")]
            }

            fn solve(
                &self,
                _t_current: Time,
                _t_next: Time,
                _input_state: &InputState,
            ) -> RSCMResult<OutputState> {
                let mut output = OutputState::new();
                output.insert(self.var_name.clone(), StateValue::Scalar(1.5));
                Ok(output)
            }
        }

        /// A component that consumes Scalar input
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct ScalarConsumer {
            input_var: String,
            output_var: String,
        }

        #[typetag::serde]
        impl Component for ScalarConsumer {
            fn definitions(&self) -> Vec<RequirementDefinition> {
                vec![
                    RequirementDefinition::scalar_input(&self.input_var, "K"),
                    RequirementDefinition::scalar_output(&self.output_var, "K"),
                ]
            }

            fn solve(
                &self,
                _t_current: Time,
                _t_next: Time,
                input_state: &InputState,
            ) -> RSCMResult<OutputState> {
                let value = input_state.get_scalar_window(&self.input_var).at_start();
                let mut output = OutputState::new();
                output.insert(self.output_var.clone(), StateValue::Scalar(value * 2.0));
                Ok(output)
            }
        }

        /// A component that consumes FourBox input
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct FourBoxConsumer {
            input_var: String,
            output_var: String,
        }

        #[typetag::serde]
        impl Component for FourBoxConsumer {
            fn definitions(&self) -> Vec<RequirementDefinition> {
                vec![
                    RequirementDefinition::four_box_input(&self.input_var, "K"),
                    RequirementDefinition::scalar_output(&self.output_var, "K"),
                ]
            }

            fn solve(
                &self,
                _t_current: Time,
                _t_next: Time,
                input_state: &InputState,
            ) -> RSCMResult<OutputState> {
                let value = input_state
                    .get_four_box_window(&self.input_var)
                    .current_global();
                let mut output = OutputState::new();
                output.insert(self.output_var.clone(), StateValue::Scalar(value));
                Ok(output)
            }
        }

        // Write-side aggregation tests

        #[test]
        fn test_write_side_fourbox_to_scalar_allowed() {
            // Schema declares Scalar, component produces FourBox
            // Should be allowed (write-side aggregation)
            let schema =
                VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Scalar);

            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(FourBoxProducer {
                    var_name: "Temperature".to_string(),
                }))
                .build();

            assert!(
                result.is_ok(),
                "Write-side FourBox->Scalar aggregation should be allowed: {:?}",
                result.err()
            );
        }

        #[test]
        fn test_write_side_fourbox_to_hemispheric_allowed() {
            // Schema declares Hemispheric, component produces FourBox
            // Should be allowed (write-side aggregation)
            let schema =
                VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Hemispheric);

            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(FourBoxProducer {
                    var_name: "Temperature".to_string(),
                }))
                .build();

            assert!(
                result.is_ok(),
                "Write-side FourBox->Hemispheric aggregation should be allowed: {:?}",
                result.err()
            );
        }

        #[test]
        fn test_write_side_hemispheric_to_scalar_allowed() {
            // Schema declares Scalar, component produces Hemispheric
            // Should be allowed (write-side aggregation)
            let schema =
                VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Scalar);

            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(HemisphericProducer {
                    var_name: "Temperature".to_string(),
                }))
                .build();

            assert!(
                result.is_ok(),
                "Write-side Hemispheric->Scalar aggregation should be allowed: {:?}",
                result.err()
            );
        }

        #[test]
        fn test_write_side_scalar_to_fourbox_rejected() {
            // Schema declares FourBox, component produces Scalar
            // Should be rejected (cannot broadcast/disaggregate)
            let schema =
                VariableSchema::new().variable_with_grid("Temperature", "K", GridType::FourBox);

            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(ScalarProducer {
                    var_name: "Temperature".to_string(),
                }))
                .build();

            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("Grid transformation not supported"),
                "Should indicate transformation not supported: {}",
                err
            );
        }

        #[test]
        fn test_write_side_scalar_to_hemispheric_rejected() {
            // Schema declares Hemispheric, component produces Scalar
            // Should be rejected (cannot broadcast/disaggregate)
            let schema =
                VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Hemispheric);

            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(ScalarProducer {
                    var_name: "Temperature".to_string(),
                }))
                .build();

            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("Grid transformation not supported"),
                "Should indicate transformation not supported: {}",
                err
            );
        }

        #[test]
        fn test_write_side_hemispheric_to_fourbox_rejected() {
            // Schema declares FourBox, component produces Hemispheric
            // Should be rejected (cannot disaggregate)
            let schema =
                VariableSchema::new().variable_with_grid("Temperature", "K", GridType::FourBox);

            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(HemisphericProducer {
                    var_name: "Temperature".to_string(),
                }))
                .build();

            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("Grid transformation not supported"),
                "Should indicate transformation not supported: {}",
                err
            );
        }

        // Read-side aggregation tests

        #[test]
        fn test_read_side_fourbox_schema_scalar_consumer_allowed() {
            // Schema declares FourBox, component consumes Scalar
            // Should be allowed (read-side aggregation)
            let schema = VariableSchema::new()
                .variable_with_grid("Temperature", "K", GridType::FourBox)
                .variable("Output", "K");

            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(FourBoxProducer {
                    var_name: "Temperature".to_string(),
                }))
                .with_component(Arc::new(ScalarConsumer {
                    input_var: "Temperature".to_string(),
                    output_var: "Output".to_string(),
                }))
                .build();

            assert!(
                result.is_ok(),
                "Read-side FourBox->Scalar aggregation should be allowed: {:?}",
                result.err()
            );
        }

        #[test]
        fn test_read_side_hemispheric_schema_scalar_consumer_allowed() {
            // Schema declares Hemispheric, component consumes Scalar
            // Should be allowed (read-side aggregation)
            let schema = VariableSchema::new()
                .variable_with_grid("Temperature", "K", GridType::Hemispheric)
                .variable("Output", "K");

            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(HemisphericProducer {
                    var_name: "Temperature".to_string(),
                }))
                .with_component(Arc::new(ScalarConsumer {
                    input_var: "Temperature".to_string(),
                    output_var: "Output".to_string(),
                }))
                .build();

            assert!(
                result.is_ok(),
                "Read-side Hemispheric->Scalar aggregation should be allowed: {:?}",
                result.err()
            );
        }

        #[test]
        fn test_read_side_scalar_schema_fourbox_consumer_rejected() {
            // Schema declares Scalar, component consumes FourBox
            // Should be rejected (cannot disaggregate/broadcast)
            let schema = VariableSchema::new()
                .variable_with_grid("Temperature", "K", GridType::Scalar)
                .variable("Output", "K");

            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(ScalarProducer {
                    var_name: "Temperature".to_string(),
                }))
                .with_component(Arc::new(FourBoxConsumer {
                    input_var: "Temperature".to_string(),
                    output_var: "Output".to_string(),
                }))
                .build();

            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("Grid transformation not supported"),
                "Should indicate transformation not supported: {}",
                err
            );
        }

        #[test]
        fn test_same_grid_always_allowed() {
            // Same grid types should always be allowed
            let schema = VariableSchema::new()
                .variable_with_grid("Temperature", "K", GridType::FourBox)
                .variable("Output", "K");

            let result = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(FourBoxProducer {
                    var_name: "Temperature".to_string(),
                }))
                .with_component(Arc::new(FourBoxConsumer {
                    input_var: "Temperature".to_string(),
                    output_var: "Output".to_string(),
                }))
                .build();

            assert!(
                result.is_ok(),
                "Same grid types should always be allowed: {:?}",
                result.err()
            );
        }
    }

    /// Integration tests for write-side grid aggregation during model execution
    mod write_side_integration_tests {
        use super::*;
        use crate::state::FourBoxSlice;
        use is_close::is_close;

        /// Component that produces FourBox output with known values
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct FourBoxWriter {
            var_name: String,
            /// Values to produce: [NO, NL, SO, SL]
            values: [FloatValue; 4],
        }

        #[typetag::serde]
        impl Component for FourBoxWriter {
            fn definitions(&self) -> Vec<RequirementDefinition> {
                vec![RequirementDefinition::four_box_output(&self.var_name, "K")]
            }

            fn solve(
                &self,
                _t_current: Time,
                _t_next: Time,
                _input_state: &InputState,
            ) -> RSCMResult<OutputState> {
                let mut output = OutputState::new();
                output.insert(
                    self.var_name.clone(),
                    StateValue::FourBox(FourBoxSlice::from_array(self.values)),
                );
                Ok(output)
            }
        }

        /// Component that produces Hemispheric output with known values
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct HemisphericWriter {
            var_name: String,
            /// Values to produce: [Northern, Southern]
            values: [FloatValue; 2],
        }

        #[typetag::serde]
        impl Component for HemisphericWriter {
            fn definitions(&self) -> Vec<RequirementDefinition> {
                vec![RequirementDefinition::hemispheric_output(
                    &self.var_name,
                    "K",
                )]
            }

            fn solve(
                &self,
                _t_current: Time,
                _t_next: Time,
                _input_state: &InputState,
            ) -> RSCMResult<OutputState> {
                let mut output = OutputState::new();
                output.insert(
                    self.var_name.clone(),
                    StateValue::Hemispheric(HemisphericSlice::from(self.values)),
                );
                Ok(output)
            }
        }

        #[test]
        fn test_write_aggregation_fourbox_to_scalar_execution() {
            // Schema declares Scalar, component produces FourBox
            // The model should aggregate FourBox values to Scalar on write
            let schema =
                VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Scalar);

            let mut model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(FourBoxWriter {
                    var_name: "Temperature".to_string(),
                    values: [10.0, 20.0, 30.0, 40.0], // [NO, NL, SO, SL]
                }))
                .build()
                .expect("Model should build");

            // Verify write transform is registered
            assert!(
                model.write_transforms().contains_key("Temperature"),
                "Write transform should be registered"
            );

            // Run one step
            model.step();

            // Check the collection has scalar data (not FourBox)
            let data = model.timeseries().get_data("Temperature").unwrap();
            let ts = data.as_scalar().expect("Should be stored as Scalar");

            // Get the value at index 1 (after first step)
            let value = ts.at_scalar(1).expect("Should have value at index 1");

            // With default equal weights [0.25, 0.25, 0.25, 0.25]:
            // 10*0.25 + 20*0.25 + 30*0.25 + 40*0.25 = 25.0
            assert!(
                is_close!(value, 25.0),
                "Expected aggregated value 25.0, got {}",
                value
            );
        }

        #[test]
        fn test_write_aggregation_fourbox_to_scalar_custom_weights() {
            // Schema declares Scalar, component produces FourBox
            // Use custom weights for aggregation
            let schema =
                VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Scalar);

            let mut model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
                .with_schema(schema)
                .with_grid_weights(GridType::FourBox, vec![0.36, 0.14, 0.36, 0.14])
                .with_component(Arc::new(FourBoxWriter {
                    var_name: "Temperature".to_string(),
                    values: [10.0, 20.0, 30.0, 40.0],
                }))
                .build()
                .expect("Model should build");

            model.step();

            let data = model.timeseries().get_data("Temperature").unwrap();
            let ts = data.as_scalar().expect("Should be stored as Scalar");
            let value = ts.at_scalar(1).expect("Should have value at index 1");

            // With custom weights [0.36, 0.14, 0.36, 0.14]:
            // 10*0.36 + 20*0.14 + 30*0.36 + 40*0.14 = 3.6 + 2.8 + 10.8 + 5.6 = 22.8
            assert!(
                is_close!(value, 22.8),
                "Expected aggregated value 22.8, got {}",
                value
            );
        }

        #[test]
        fn test_write_aggregation_fourbox_to_hemispheric_execution() {
            // Schema declares Hemispheric, component produces FourBox
            let schema =
                VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Hemispheric);

            let mut model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(FourBoxWriter {
                    var_name: "Temperature".to_string(),
                    values: [10.0, 20.0, 30.0, 40.0],
                }))
                .build()
                .expect("Model should build");

            model.step();

            let data = model.timeseries().get_data("Temperature").unwrap();
            let ts = data
                .as_hemispheric()
                .expect("Should be stored as Hemispheric");

            let northern = ts.at_index(1, 0).expect("Should have northern value");
            let southern = ts.at_index(1, 1).expect("Should have southern value");

            // With equal weights [0.25, 0.25, 0.25, 0.25]:
            // Northern = (10*0.25 + 20*0.25) / (0.25 + 0.25) = 7.5 / 0.5 = 15.0
            // Southern = (30*0.25 + 40*0.25) / (0.25 + 0.25) = 17.5 / 0.5 = 35.0
            assert!(
                is_close!(northern, 15.0),
                "Expected northern 15.0, got {}",
                northern
            );
            assert!(
                is_close!(southern, 35.0),
                "Expected southern 35.0, got {}",
                southern
            );
        }

        #[test]
        fn test_write_aggregation_hemispheric_to_scalar_execution() {
            // Schema declares Scalar, component produces Hemispheric
            let schema =
                VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Scalar);

            let mut model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(HemisphericWriter {
                    var_name: "Temperature".to_string(),
                    values: [15.0, 35.0], // [Northern, Southern]
                }))
                .build()
                .expect("Model should build");

            model.step();

            let data = model.timeseries().get_data("Temperature").unwrap();
            let ts = data.as_scalar().expect("Should be stored as Scalar");
            let value = ts.at_scalar(1).expect("Should have value at index 1");

            // With default equal weights [0.5, 0.5]:
            // 15*0.5 + 35*0.5 = 25.0
            assert!(
                is_close!(value, 25.0),
                "Expected aggregated value 25.0, got {}",
                value
            );
        }

        #[test]
        fn test_write_aggregation_multiple_steps() {
            // Run multiple steps to ensure aggregation works consistently
            let schema =
                VariableSchema::new().variable_with_grid("Temperature", "K", GridType::Scalar);

            let mut model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2025.0, 1.0)))
                .with_schema(schema)
                .with_component(Arc::new(FourBoxWriter {
                    var_name: "Temperature".to_string(),
                    values: [10.0, 20.0, 30.0, 40.0],
                }))
                .build()
                .expect("Model should build");

            // Run all steps
            model.run();

            let data = model.timeseries().get_data("Temperature").unwrap();
            let ts = data.as_scalar().expect("Should be stored as Scalar");

            // Check values at all indices after initial
            for i in 1..5 {
                let value = ts
                    .at_scalar(i)
                    .unwrap_or_else(|| panic!("Should have value at index {}", i));
                assert!(
                    is_close!(value, 25.0),
                    "Expected aggregated value 25.0 at index {}, got {}",
                    i,
                    value
                );
            }
        }

        #[test]
        fn test_no_schema_no_aggregation() {
            // Without a schema, no aggregation should happen
            // Component produces FourBox, it should stay as FourBox
            let mut model = ModelBuilder::new()
                .with_time_axis(TimeAxis::from_values(Array::range(2020.0, 2023.0, 1.0)))
                .with_component(Arc::new(FourBoxWriter {
                    var_name: "Temperature".to_string(),
                    values: [10.0, 20.0, 30.0, 40.0],
                }))
                .build()
                .expect("Model should build without schema");

            // Verify no write transforms
            assert!(
                model.write_transforms().is_empty(),
                "Should have no write transforms without schema"
            );

            model.step();

            // Data should remain as FourBox
            let data = model.timeseries().get_data("Temperature").unwrap();
            assert!(
                data.as_four_box().is_some(),
                "Should be stored as FourBox without schema"
            );
        }
    }

    /// Tests for the aggregate_state_value helper function
    mod aggregate_state_value_tests {
        use super::*;
        use crate::state::FourBoxSlice;
        use is_close::is_close;

        #[test]
        fn test_fourbox_to_scalar_default_weights() {
            // With equal weights (default MAGICC standard), result is mean
            let fourbox = StateValue::FourBox(FourBoxSlice::from([10.0, 20.0, 30.0, 40.0]));

            let result =
                aggregate_state_value(&fourbox, GridType::FourBox, GridType::Scalar, None).unwrap();

            match result {
                StateValue::Scalar(v) => {
                    // Default weights are equal [0.25, 0.25, 0.25, 0.25]
                    // 10*0.25 + 20*0.25 + 30*0.25 + 40*0.25 = 25.0
                    assert!(is_close!(v, 25.0), "Expected 25.0, got {}", v);
                }
                _ => panic!("Expected Scalar, got {:?}", result),
            }
        }

        #[test]
        fn test_fourbox_to_scalar_custom_weights() {
            let fourbox = StateValue::FourBox(FourBoxSlice::from([10.0, 20.0, 30.0, 40.0]));
            let weights = vec![0.36, 0.14, 0.36, 0.14]; // Ocean-biased weights

            let result = aggregate_state_value(
                &fourbox,
                GridType::FourBox,
                GridType::Scalar,
                Some(&weights),
            )
            .unwrap();

            match result {
                StateValue::Scalar(v) => {
                    // 10*0.36 + 20*0.14 + 30*0.36 + 40*0.14 = 3.6 + 2.8 + 10.8 + 5.6 = 22.8
                    assert!(is_close!(v, 22.8), "Expected 22.8, got {}", v);
                }
                _ => panic!("Expected Scalar, got {:?}", result),
            }
        }

        #[test]
        fn test_fourbox_to_hemispheric_default_weights() {
            let fourbox = StateValue::FourBox(FourBoxSlice::from([10.0, 20.0, 30.0, 40.0]));

            let result =
                aggregate_state_value(&fourbox, GridType::FourBox, GridType::Hemispheric, None)
                    .unwrap();

            match result {
                StateValue::Hemispheric(slice) => {
                    // With equal weights [0.25, 0.25, 0.25, 0.25]:
                    // Northern = (10*0.25 + 20*0.25) / (0.25 + 0.25) = 7.5 / 0.5 = 15.0
                    // Southern = (30*0.25 + 40*0.25) / (0.25 + 0.25) = 17.5 / 0.5 = 35.0
                    assert!(
                        is_close!(slice.as_array()[0], 15.0),
                        "Expected Northern=15.0, got {}",
                        slice.as_array()[0]
                    );
                    assert!(
                        is_close!(slice.as_array()[1], 35.0),
                        "Expected Southern=35.0, got {}",
                        slice.as_array()[1]
                    );
                }
                _ => panic!("Expected Hemispheric, got {:?}", result),
            }
        }

        #[test]
        fn test_fourbox_to_hemispheric_custom_weights() {
            let fourbox = StateValue::FourBox(FourBoxSlice::from([10.0, 20.0, 30.0, 40.0]));
            let weights = vec![0.36, 0.14, 0.36, 0.14]; // Ocean-biased

            let result = aggregate_state_value(
                &fourbox,
                GridType::FourBox,
                GridType::Hemispheric,
                Some(&weights),
            )
            .unwrap();

            match result {
                StateValue::Hemispheric(slice) => {
                    // Northern = (10*0.36 + 20*0.14) / (0.36 + 0.14) = (3.6 + 2.8) / 0.5 = 12.8
                    // Southern = (30*0.36 + 40*0.14) / (0.36 + 0.14) = (10.8 + 5.6) / 0.5 = 32.8
                    assert!(
                        is_close!(slice.as_array()[0], 12.8),
                        "Expected Northern=12.8, got {}",
                        slice.as_array()[0]
                    );
                    assert!(
                        is_close!(slice.as_array()[1], 32.8),
                        "Expected Southern=32.8, got {}",
                        slice.as_array()[1]
                    );
                }
                _ => panic!("Expected Hemispheric, got {:?}", result),
            }
        }

        #[test]
        fn test_hemispheric_to_scalar_default_weights() {
            let hemispheric = StateValue::Hemispheric(HemisphericSlice::from([15.0, 35.0]));

            let result =
                aggregate_state_value(&hemispheric, GridType::Hemispheric, GridType::Scalar, None)
                    .unwrap();

            match result {
                StateValue::Scalar(v) => {
                    // Default weights [0.5, 0.5] -> mean
                    // 15*0.5 + 35*0.5 = 25.0
                    assert!(is_close!(v, 25.0), "Expected 25.0, got {}", v);
                }
                _ => panic!("Expected Scalar, got {:?}", result),
            }
        }

        #[test]
        fn test_hemispheric_to_scalar_custom_weights() {
            let hemispheric = StateValue::Hemispheric(HemisphericSlice::from([10.0, 30.0]));
            let weights = vec![0.4, 0.6]; // Southern-biased

            let result = aggregate_state_value(
                &hemispheric,
                GridType::Hemispheric,
                GridType::Scalar,
                Some(&weights),
            )
            .unwrap();

            match result {
                StateValue::Scalar(v) => {
                    // 10*0.4 + 30*0.6 = 4 + 18 = 22.0
                    assert!(is_close!(v, 22.0), "Expected 22.0, got {}", v);
                }
                _ => panic!("Expected Scalar, got {:?}", result),
            }
        }

        #[test]
        fn test_identity_transformation_scalar() {
            let scalar = StateValue::Scalar(42.0);

            let result =
                aggregate_state_value(&scalar, GridType::Scalar, GridType::Scalar, None).unwrap();

            match result {
                StateValue::Scalar(v) => assert_eq!(v, 42.0),
                _ => panic!("Expected Scalar, got {:?}", result),
            }
        }

        #[test]
        fn test_identity_transformation_fourbox() {
            let fourbox = StateValue::FourBox(FourBoxSlice::from([1.0, 2.0, 3.0, 4.0]));

            let result =
                aggregate_state_value(&fourbox, GridType::FourBox, GridType::FourBox, None)
                    .unwrap();

            match result {
                StateValue::FourBox(slice) => {
                    assert_eq!(*slice.as_array(), [1.0, 2.0, 3.0, 4.0]);
                }
                _ => panic!("Expected FourBox, got {:?}", result),
            }
        }

        #[test]
        fn test_disaggregation_scalar_to_fourbox_rejected() {
            let scalar = StateValue::Scalar(25.0);

            let result = aggregate_state_value(&scalar, GridType::Scalar, GridType::FourBox, None);

            assert!(
                result.is_err(),
                "Disaggregation Scalar->FourBox should be rejected"
            );
        }

        #[test]
        fn test_disaggregation_scalar_to_hemispheric_rejected() {
            let scalar = StateValue::Scalar(25.0);

            let result =
                aggregate_state_value(&scalar, GridType::Scalar, GridType::Hemispheric, None);

            assert!(
                result.is_err(),
                "Disaggregation Scalar->Hemispheric should be rejected"
            );
        }

        #[test]
        fn test_disaggregation_hemispheric_to_fourbox_rejected() {
            let hemispheric = StateValue::Hemispheric(HemisphericSlice::from([15.0, 35.0]));

            let result =
                aggregate_state_value(&hemispheric, GridType::Hemispheric, GridType::FourBox, None);

            assert!(
                result.is_err(),
                "Disaggregation Hemispheric->FourBox should be rejected"
            );
        }
    }
}
