//! Model builder for constructing models from components.

use crate::component::{Component, GridType, RequirementDefinition, RequirementType};
use crate::errors::{RSCMError, RSCMResult};
use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
use crate::schema::VariableSchema;
use crate::timeseries::{FloatValue, TimeAxis, Timeseries};
use crate::timeseries_collection::{TimeseriesCollection, TimeseriesData, VariableType};
use numpy::ndarray::Array;
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use std::collections::HashMap;
use std::sync::Arc;

use super::null_component::NullComponent;
use super::runtime::Model;
use super::types::{CGraph, RequiredTransformation, TransformDirection, VariableDefinition, C};
use super::validation::verify_definition;

/// Build a new model from a set of components.
///
/// The builder generates a graph that defines the inter-component dependencies
/// and determines what variables are endogenous and exogenous to the model.
/// This graph is used by the model to define the order in which components are solved.
pub struct ModelBuilder {
    components: Vec<C>,
    pub(crate) exogenous_variables: TimeseriesCollection,
    initial_values: HashMap<String, FloatValue>,
    /// The time axis for the model.
    pub time_axis: Arc<TimeAxis>,
    schema: Option<VariableSchema>,
    /// Custom weights for grid aggregation, keyed by grid type.
    ///
    /// When provided, these override the default weights used when creating
    /// timeseries and performing grid transformations. Weights must sum to 1.0.
    grid_weights: HashMap<GridType, Vec<f64>>,
}

impl ModelBuilder {
    /// Create a new model builder with default settings.
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

    /// Set custom weights for a grid type.
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

    /// Create a FourBoxGrid using custom weights if configured, otherwise use defaults.
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

    /// Register a component with the builder.
    pub fn with_component(&mut self, component: Arc<dyn Component + Send + Sync>) -> &mut Self {
        self.components.push(component);
        self
    }

    /// Supply exogenous data to be used by the model.
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

    /// Supply exogenous data to be used by the model.
    ///
    /// Any unneeded timeseries will be ignored.
    pub fn with_exogenous_collection(&mut self, collection: TimeseriesCollection) -> &mut Self {
        self.exogenous_variables.extend(collection);
        self
    }

    /// Adds some state to the set of initial values.
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

    /// Specify the time axis that will be used by the model.
    ///
    /// This time axis defines the time steps (including bounds) on which the model will be iterated.
    pub fn with_time_axis(&mut self, time_axis: TimeAxis) -> &mut Self {
        self.time_axis = Arc::new(time_axis);
        self
    }

    /// Validate a component's requirements against the schema.
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
        // Write-side: component produces finer grid than schema -> aggregate before storage
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
                    // Component produces finer data -> aggregation to schema is OK
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
        // Read-side: component wants coarser grid than schema -> aggregate before read
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
                        // Schema has finer data -> aggregation for component is OK
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

    /// Builds the component graph for the registered components and creates a concrete model.
    ///
    /// Returns an error if the component definitions are inconsistent.
    pub fn build(&self) -> RSCMResult<Model> {
        use crate::spatial::ScalarRegion;

        // todo: refactor once this is more stable
        let mut graph: CGraph = Graph::new();
        let mut endogenous: HashMap<String, NodeIndex> = HashMap::new();
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

                if let Some(&producer_node) = endogenous.get(&requirement.name) {
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

                let val = endogenous.get(&requirement.name);

                match val {
                    None => {
                        endogenous.insert(requirement.name.clone(), node);
                    }
                    Some(node_index) => {
                        graph.add_edge(*node_index, node, requirement.clone());
                        endogenous.insert(requirement.name.clone(), node);
                    }
                }
            }
        }

        // Check that the component graph doesn't contain any loops
        assert!(!super::validation::is_valid_graph(&graph));

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
                    &endogenous,
                )?;
                all_transformations.extend(component_transforms);
            }

            // Handle schema variables (4.5)
            // For variables only declared as inputs (not produced by any component),
            // add them to definitions using the schema's grid type.
            // For variables that are produced by components, update the grid type
            // to match the schema (the schema is the source of truth for storage grid type).
            for (name, var_def) in &schema.variables {
                if !definitions.contains_key(name) {
                    // Variable not produced by any component - add it as exogenous
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
                } else {
                    // Variable exists (from component input declaration) - update grid type to match schema
                    // This ensures storage uses schema's grid type, and read transforms will handle conversion
                    if let Some(def) = definitions.get_mut(name) {
                        if def.grid_type != var_def.grid_type && !endogenous.contains_key(name) {
                            // Only update if this variable is exogenous (input-only)
                            // If a component outputs this variable, the write transform will handle conversion
                            def.grid_type = var_def.grid_type;
                            exogenous.push(name.clone());
                        }
                    }
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
                    if let Some(&producer_node) = endogenous.get(contributor) {
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
                endogenous.insert(agg_name.clone(), agg_node);

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
                    // Look for exogenous data matching the schema's grid type
                    let exogenous_data = self.exogenous_variables.get_data(&name);

                    match (exogenous_data, definition.grid_type) {
                        (Some(TimeseriesData::Scalar(ts)), GridType::Scalar) => {
                            collection.add_timeseries(
                                name,
                                ts.to_owned().interpolate_into(self.time_axis.clone()),
                                VariableType::Exogenous,
                            );
                        }
                        (Some(TimeseriesData::FourBox(ts)), GridType::FourBox) => {
                            collection.add_four_box_timeseries(
                                name,
                                ts.to_owned().interpolate_into(self.time_axis.clone()),
                                VariableType::Exogenous,
                            );
                        }
                        (Some(TimeseriesData::Hemispheric(ts)), GridType::Hemispheric) => {
                            collection.add_hemispheric_timeseries(
                                name,
                                ts.to_owned().interpolate_into(self.time_axis.clone()),
                                VariableType::Exogenous,
                            );
                        }
                        _ => {
                            // No exogenous data provided or grid type mismatch
                            // Create empty timeseries (all NaN) matching the schema's grid type
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
        let mut model = Model::with_transforms(
            graph,
            initial_node,
            collection,
            self.time_axis.clone(),
            self.grid_weights.clone(),
            read_transforms,
            write_transforms,
        );

        // Initialize component states for each node
        for node_idx in model.components.node_indices() {
            let component = &model.components[node_idx];
            let state = component.create_initial_state();
            model.component_states.insert(node_idx, state);
        }

        Ok(model)
    }
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}
