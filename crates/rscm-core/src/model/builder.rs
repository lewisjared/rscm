//! Model builder for constructing models from components.

use crate::component::{Component, GridType, RequirementDefinition, RequirementType};
use crate::errors::{RSCMError, RSCMResult};
use crate::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
use crate::schema::VariableSchema;
use crate::spatial::FourBoxGrid;
use crate::state::VariableSource;
use crate::timeseries::{FloatValue, GridTimeseries, TimeAxis, Timeseries};
use crate::timeseries_collection::{TimeseriesCollection, TimeseriesData, VariableType};
use crate::units::Unit;
use numpy::ndarray::Array;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Graph;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use super::null_component::NullComponent;
use super::runtime::Model;
use super::types::{
    CGraph, RequiredTransformation, TransformDirection, UnitConversionInfo, VariableDefinition, C,
};
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
    initial_grid_values: HashMap<String, Vec<FloatValue>>,
    /// The time axis for the model.
    pub time_axis: Arc<TimeAxis>,
    schema: Option<VariableSchema>,
    /// Custom weights for grid aggregation, keyed by grid type.
    ///
    /// When provided, these override the default weights used when creating
    /// timeseries and performing grid transformations. Weights must sum to 1.0.
    grid_weights: HashMap<GridType, Vec<f64>>,
    lagged_variables: BTreeSet<String>,
    lagged_edges: BTreeSet<(String, String)>,
}

impl ModelBuilder {
    /// Create a new model builder with default settings.
    pub fn new() -> Self {
        Self {
            components: vec![],
            initial_values: HashMap::new(),
            initial_grid_values: HashMap::new(),
            exogenous_variables: TimeseriesCollection::new(),
            time_axis: Arc::new(TimeAxis::from_values(Array::range(2000.0, 2100.0, 1.0))),
            schema: None,
            grid_weights: HashMap::new(),
            lagged_variables: BTreeSet::new(),
            lagged_edges: BTreeSet::new(),
        }
    }

    /// Lag every edge from `variable`, so each of its consumers reads the previous step's value.
    ///
    /// The lag holds whether the variable is produced by a component or exogenous.
    pub fn with_lag(&mut self, variable: &str) -> &mut Self {
        self.lagged_variables.insert(variable.to_string());
        self
    }

    /// Lag the edge from `variable` into `consumer` alone.
    ///
    /// `consumer` is the component's name, such as `"OceanCarbon"`.
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
        names: &[String],
        requirements: &[(Vec<RequirementDefinition>, Vec<RequirementDefinition>)],
    ) -> RSCMResult<()> {
        let aggregates = self.schema.as_ref().map(|s| &s.aggregates);
        for variable in &self.lagged_variables {
            let known = requirements
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
            if !names.contains(consumer) {
                let known: BTreeSet<&str> = names.iter().map(String::as_str).collect();
                return Err(RSCMError::InvalidLag {
                    lag,
                    reason: format!("no component named {consumer:?}, components are {known:?}"),
                });
            }
            let reads = names
                .iter()
                .zip(requirements)
                .filter(|(n, _)| *n == consumer)
                .any(|(_, (requires, _))| {
                    requires.iter().any(|r| {
                        &r.name == variable && r.requirement_type == RequirementType::Input
                    })
                });
            if !reads {
                return Err(RSCMError::InvalidLag {
                    lag,
                    reason: format!("{consumer} does not read {variable:?} as an input"),
                });
            }
        }
        Ok(())
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

    /// Adds a four-box exogenous variable.
    pub fn with_exogenous_four_box_variable(
        &mut self,
        name: &str,
        timeseries: GridTimeseries<FloatValue, FourBoxGrid>,
    ) -> &mut Self {
        self.exogenous_variables.add_four_box_timeseries(
            name.to_string(),
            timeseries,
            VariableType::Exogenous,
        );
        self
    }

    /// Adds initial values for grid variables, one per region in grid order.
    ///
    /// These take precedence over a uniform value given to [`Self::with_initial_values`].
    /// Building fails if the number of values does not match the variable's grid.
    pub fn with_initial_grid_values(
        &mut self,
        initial_values: HashMap<String, Vec<FloatValue>>,
    ) -> &mut Self {
        self.initial_grid_values.extend(initial_values);
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
    /// - Units are dimensionally compatible between component and schema
    /// - Grid types are compatible (allowing aggregation where valid)
    ///
    /// Returns a tuple of:
    /// - A list of required grid transformations for mismatched grids
    /// - A list of unit conversions needed at runtime
    fn validate_component_against_schema(
        &self,
        schema: &VariableSchema,
        component_name: &str,
        inputs: &[RequirementDefinition],
        outputs: &[RequirementDefinition],
        endogenous: &HashMap<String, NodeIndex>,
    ) -> RSCMResult<(Vec<RequiredTransformation>, Vec<UnitConversionInfo>)> {
        let mut transformations = Vec::new();
        let mut unit_conversions = Vec::new();

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

            // Check unit compatibility (not just string equality)
            if let Some(schema_unit) = schema.get_unit(&output.name) {
                if let Some(conversion) = Self::check_unit_compatibility(
                    &output.name,
                    component_name,
                    schema_unit,
                    &output.unit,
                )? {
                    unit_conversions.push(conversion);
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
                    if let Some(conversion) = Self::check_unit_compatibility(
                        &input.name,
                        component_name,
                        schema_unit,
                        &input.unit,
                    )? {
                        unit_conversions.push(conversion);
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

        Ok((transformations, unit_conversions))
    }

    /// Check if two units are compatible and return conversion info if needed.
    ///
    /// Returns:
    /// - `Ok(None)` if units are identical (no conversion needed)
    /// - `Ok(Some(conversion))` if units are compatible but different
    /// - `Err(...)` if units are incompatible (different dimensions)
    fn check_unit_compatibility(
        variable: &str,
        component: &str,
        schema_unit: &str,
        component_unit: &str,
    ) -> RSCMResult<Option<UnitConversionInfo>> {
        // Fast path: exact string match
        if schema_unit == component_unit {
            return Ok(None);
        }

        // Parse both units
        let parsed_schema = Unit::parse(schema_unit).map_err(|e| RSCMError::UnitParseError {
            variable: variable.to_string(),
            unit_string: schema_unit.to_string(),
            details: e.to_string(),
        })?;

        let parsed_component =
            Unit::parse(component_unit).map_err(|e| RSCMError::UnitParseError {
                variable: variable.to_string(),
                unit_string: component_unit.to_string(),
                details: e.to_string(),
            })?;

        // Check if they're the same after normalization
        if parsed_schema == parsed_component {
            return Ok(None);
        }

        // Check dimensional compatibility
        if !parsed_schema.is_compatible(&parsed_component) {
            let dim_schema = parsed_schema
                .dimension()
                .map(|d| d.to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            let dim_component = parsed_component
                .dimension()
                .map(|d| d.to_string())
                .unwrap_or_else(|_| "unknown".to_string());

            return Err(RSCMError::IncompatibleUnits {
                variable: variable.to_string(),
                unit1: schema_unit.to_string(),
                unit2: component_unit.to_string(),
                dim1: dim_schema,
                dim2: dim_component,
            });
        }

        // Calculate conversion factor from schema unit to component unit
        let factor = parsed_schema
            .conversion_factor(&parsed_component)
            .map_err(|e| RSCMError::UnitParseError {
                variable: variable.to_string(),
                unit_string: format!("{schema_unit} -> {component_unit}"),
                details: e.to_string(),
            })?;

        Ok(Some(UnitConversionInfo {
            variable: variable.to_string(),
            component: component.to_string(),
            factor,
            source_unit: schema_unit.to_string(),
            target_unit: component_unit.to_string(),
        }))
    }

    /// Builds the component graph for the registered components and creates a concrete model.
    ///
    /// Returns an error if the component definitions are inconsistent.
    pub fn build(&self) -> RSCMResult<Model> {
        use crate::spatial::ScalarRegion;

        // todo: refactor once this is more stable
        let mut graph: CGraph = Graph::new();
        let mut endogenous: HashMap<String, NodeIndex> = HashMap::new();
        let mut exogenous: HashSet<String> = HashSet::new();
        let mut definitions: HashMap<String, VariableDefinition> = HashMap::new();
        // Track which component owns each variable for better error messages
        let mut variable_owners: HashMap<String, String> = HashMap::new();
        // Collect unit conversion factors needed at runtime
        // Key: (variable_name, component_name) -> conversion info
        let mut unit_conversions: Vec<UnitConversionInfo> = Vec::new();
        // Track variable sources keyed by (variable_name, component_name)
        // This enables different components to have different source classifications
        // for the same variable (e.g., State for owner, UpstreamOutput for consumers)
        let mut variable_sources: HashMap<(String, String), VariableSource> = HashMap::new();
        let initial_node = graph.add_node(Arc::new(NullComponent {}));
        let mut aggregator_nodes: HashSet<NodeIndex> = HashSet::new();

        // Track pending aggregate dependencies: (component_node, variable_name, requirement)
        // These will be resolved after aggregators are added to the graph
        let mut pending_aggregate_deps: Vec<(NodeIndex, String, RequirementDefinition)> = vec![];

        // Get aggregate names for forward reference (if schema exists)
        let aggregate_names: HashSet<String> = self
            .schema
            .as_ref()
            .map(|s| s.aggregates.keys().cloned().collect())
            .unwrap_or_default();

        // Each call builds the definitions afresh, so read them once.
        let requirements: Vec<_> = self
            .components
            .iter()
            .map(|component| (component.inputs(), component.outputs()))
            .collect();

        let names: Vec<String> = self
            .components
            .iter()
            .map(Model::extract_component_name)
            .collect();
        self.validate_lags(&names, &requirements)?;

        let nodes: Vec<NodeIndex> = self
            .components
            .iter()
            .map(|component| graph.add_node(component.clone()))
            .collect();

        // Every producer is known before any input is classified, so registration order is irrelevant.
        for (index, (_, provides)) in requirements.iter().enumerate() {
            for requirement in provides {
                if let Some(other) = variable_owners.get(&requirement.name) {
                    return Err(RSCMError::MultipleProducers {
                        variable: requirement.name.clone(),
                        components: vec![other.clone(), names[index].clone()],
                    });
                }
                endogenous.insert(requirement.name.clone(), nodes[index]);
                variable_owners.insert(requirement.name.clone(), names[index].clone());
            }
        }

        // Producers define each variable first, so consumers get the unit conversions.
        for (index, (_, provides)) in requirements.iter().enumerate() {
            for requirement in provides {
                if let Some(conversion) = verify_definition(
                    &mut definitions,
                    requirement,
                    &names[index],
                    Some(&names[index]),
                    self.schema.is_some(),
                )? {
                    unit_conversions.push(conversion);
                }
            }
        }

        for (index, (requires, _)) in requirements.iter().enumerate() {
            let node = nodes[index];
            let component_name = &names[index];
            let mut has_dependencies = false;

            for requirement in requires {
                if requirement.requirement_type == RequirementType::EmptyLink {
                    continue;
                }
                let existing_owner = variable_owners.get(&requirement.name).map(|s| s.as_str());
                if let Some(conversion) = verify_definition(
                    &mut definitions,
                    requirement,
                    component_name,
                    existing_owner,
                    self.schema.is_some(),
                )? {
                    unit_conversions.push(conversion);
                }

                let name = &requirement.name;
                let lagged = self.is_lagged(name, component_name);
                let producer = endogenous.get(name).copied();
                let source = if requirement.requirement_type == RequirementType::State
                    || producer == Some(node)
                {
                    exogenous.insert(name.clone());
                    VariableSource::OwnState
                } else if lagged {
                    // A lagged read sees index N, which exogenous data may fill.
                    exogenous.insert(name.clone());
                    VariableSource::Lagged
                } else if let Some(producer) = producer {
                    graph.add_edge(producer, node, requirement.clone());
                    has_dependencies = true;
                    VariableSource::UpstreamOutput
                } else if aggregate_names.contains(name) {
                    pending_aggregate_deps.push((node, name.clone(), requirement.clone()));
                    has_dependencies = true;
                    VariableSource::UpstreamOutput
                } else {
                    exogenous.insert(name.clone());
                    VariableSource::Exogenous
                };
                variable_sources.insert((name.clone(), component_name.clone()), source);
            }

            if !has_dependencies {
                // Link to the initial node so the graph stays connected.
                graph.add_edge(
                    initial_node,
                    node,
                    RequirementDefinition::new("", "", RequirementType::EmptyLink),
                );
            }
        }

        // Collect all required grid transformations
        let mut all_transformations: Vec<RequiredTransformation> = Vec::new();

        // Validate against schema if provided
        if let Some(schema) = &self.schema {
            // First validate the schema itself
            schema.validate()?;

            // Validate each component against the schema and collect transformations
            for (component, (requires, provides)) in self.components.iter().zip(&requirements) {
                let component_name = Model::extract_component_name(component);

                let (component_transforms, component_unit_conversions) = self
                    .validate_component_against_schema(
                        schema,
                        &component_name,
                        requires,
                        provides,
                        &endogenous,
                    )?;
                all_transformations.extend(component_transforms);
                unit_conversions.extend(component_unit_conversions);
            }

            // Handle schema variables (4.5)
            // For variables only declared as inputs (not produced by any component),
            // add them to definitions using the schema's grid type.
            // For variables that are produced by components, update the grid type
            // to match the schema (the schema is the source of truth for storage grid type).
            for (name, var_def) in &schema.variables {
                if !definitions.contains_key(name) {
                    // Variable not produced by any component - add it as exogenous
                    let parsed_unit = Unit::parse(&var_def.unit).ok();
                    definitions.insert(
                        name.clone(),
                        VariableDefinition {
                            name: name.clone(),
                            unit: var_def.unit.clone(),
                            parsed_unit,
                            grid_type: var_def.grid_type,
                            requirement_type: RequirementType::Input, // Schema-only variables are inputs
                        },
                    );
                    // Mark as exogenous since it's not produced by any component
                    exogenous.insert(name.clone());
                } else {
                    // Variable exists (from component input declaration) - update grid type to match schema
                    // This ensures storage uses schema's grid type, and read transforms will handle conversion
                    if let Some(def) = definitions.get_mut(name) {
                        if def.grid_type != var_def.grid_type {
                            // Update storage grid type to match schema
                            // Read/write transforms handle conversion between component and schema grid types
                            def.grid_type = var_def.grid_type;
                            if !endogenous.contains_key(name) {
                                exogenous.insert(name.clone());
                            }
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
                aggregator_nodes.insert(agg_node);

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
                let parsed_unit = Unit::parse(&agg_def.unit).ok();
                definitions.insert(
                    agg_name.clone(),
                    VariableDefinition {
                        name: agg_name.clone(),
                        unit: agg_def.unit.clone(),
                        parsed_unit,
                        grid_type: agg_def.grid_type,
                        requirement_type: RequirementType::Output, // Aggregates are outputs
                    },
                );
            }

            // Resolve pending aggregate dependencies now that aggregators are in the graph
            for (component_node, var_name, requirement) in pending_aggregate_deps {
                if let Some(&agg_node) = endogenous.get(&var_name) {
                    graph.add_edge(agg_node, component_node, requirement);
                }
            }
        }

        check_acyclic(&graph, &requirements, &names, &aggregator_nodes)?;

        // Validate that all state variables have initial values
        for (name, def) in &definitions {
            if def.requirement_type == RequirementType::State
                && !self.initial_values.contains_key(name)
                && !self.initial_grid_values.contains_key(name)
            {
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
        let interp = InterpolationStrategy::from(LinearSplineStrategy::new(true));

        for (name, definition) in definitions {
            assert_eq!(definition.name, name);

            let is_endogenous = endogenous.contains_key(&name);
            let var_type = if is_endogenous {
                VariableType::Endogenous
            } else {
                VariableType::Exogenous
            };

            // Determine storage grid type: write transforms override the definition
            let storage_grid = write_transforms
                .get(&name)
                .map(|t| t.target_grid)
                .unwrap_or(definition.grid_type);

            // Try to populate from: (1) exogenous data, (2) initial value, (3) empty
            let exogenous_data = if exogenous.contains(&name) {
                self.exogenous_variables.get_data(&name)
            } else {
                None
            };
            let initial_val = self.initial_values.get(&name).copied();
            let initial_grid = self.initial_grid_values.get(&name);
            let expected_len = match storage_grid {
                GridType::Scalar => 1,
                GridType::Hemispheric => 2,
                GridType::FourBox => 4,
            };
            if let Some(values) = initial_grid {
                if values.len() != expected_len {
                    return Err(RSCMError::Error(format!(
                        "Initial values for {name} must have {expected_len} entries, got {}",
                        values.len()
                    )));
                }
            }

            match storage_grid {
                GridType::Scalar => {
                    if let Some(TimeseriesData::Scalar(ts)) = exogenous_data {
                        collection.add_timeseries(
                            name,
                            ts.to_owned().interpolate_into(self.time_axis.clone()),
                            var_type,
                        );
                    } else {
                        let mut ts = Timeseries::new_empty_scalar(
                            self.time_axis.clone(),
                            definition.unit,
                            interp.clone(),
                        );
                        if let Some(val) = initial_grid.map(|v| v[0]).or(initial_val) {
                            ts.set(0, ScalarRegion::Global, val);
                        }
                        collection.add_timeseries(name, ts, var_type);
                    }
                }
                GridType::FourBox => {
                    if let Some(TimeseriesData::FourBox(ts)) = exogenous_data {
                        collection.add_four_box_timeseries(
                            name,
                            ts.to_owned().interpolate_into(self.time_axis.clone()),
                            var_type,
                        );
                    } else {
                        let mut ts = GridTimeseries::new_empty(
                            self.time_axis.clone(),
                            self.create_four_box_grid(),
                            definition.unit,
                            interp.clone(),
                        );
                        use crate::spatial::FourBoxRegion;
                        let regions = [
                            FourBoxRegion::NorthernOcean,
                            FourBoxRegion::NorthernLand,
                            FourBoxRegion::SouthernOcean,
                            FourBoxRegion::SouthernLand,
                        ];
                        if let Some(values) = initial_grid {
                            ts.set_all(0, values);
                        } else if let Some(val) = initial_val {
                            for region in regions {
                                ts.set(0, region, val);
                            }
                        }
                        collection.add_four_box_timeseries(name, ts, var_type);
                    }
                }
                GridType::Hemispheric => {
                    if let Some(TimeseriesData::Hemispheric(ts)) = exogenous_data {
                        collection.add_hemispheric_timeseries(
                            name,
                            ts.to_owned().interpolate_into(self.time_axis.clone()),
                            var_type,
                        );
                    } else {
                        let mut ts = GridTimeseries::new_empty(
                            self.time_axis.clone(),
                            self.create_hemispheric_grid(),
                            definition.unit,
                            interp.clone(),
                        );
                        use crate::spatial::HemisphericRegion;
                        let regions = [HemisphericRegion::Northern, HemisphericRegion::Southern];
                        if let Some(values) = initial_grid {
                            ts.set_all(0, values);
                        } else if let Some(val) = initial_val {
                            for region in regions {
                                ts.set(0, region, val);
                            }
                        }
                        collection.add_hemispheric_timeseries(name, ts, var_type);
                    }
                }
            }
        }

        // Build unit conversion map from collected conversion info
        // Key: (variable_name, component_name) -> conversion factor
        let unit_conversion_map: HashMap<(String, String), f64> = unit_conversions
            .into_iter()
            .map(|info| ((info.variable.clone(), info.component.clone()), info.factor))
            .collect();

        // Add the components to the graph
        let mut model = Model::with_variable_sources(
            graph,
            initial_node,
            collection,
            self.time_axis.clone(),
            self.grid_weights.clone(),
            read_transforms,
            write_transforms,
            unit_conversion_map,
            variable_sources,
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

/// Check that no feedback loop is left unlagged.
///
/// A cycle is a feedback loop no lag breaks. The error lists the components and variables in each loop,
/// and the `with_lag_for` calls that would break it, edges into a component with its own state first.
fn check_acyclic(
    graph: &CGraph,
    requirements: &[(Vec<RequirementDefinition>, Vec<RequirementDefinition>)],
    names: &[String],
    aggregator_nodes: &HashSet<NodeIndex>,
) -> RSCMResult<()> {
    if !petgraph::algo::is_cyclic_directed(graph) {
        return Ok(());
    }
    let stateful: HashSet<&str> = names
        .iter()
        .zip(requirements)
        .filter(|(_, (requires, _))| {
            requires
                .iter()
                .any(|r| r.requirement_type == RequirementType::State)
        })
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
        components.extend(
            scc.iter()
                .map(|&n| Model::extract_component_name(&graph[n])),
        );
        for edge in graph.edge_references() {
            if !(members.contains(&edge.source()) && members.contains(&edge.target())) {
                continue;
            }
            let variable = edge.weight().name.clone();
            variables.insert(variable.clone());
            // Aggregators always read this step's contributions, so their inputs cannot lag.
            if !aggregator_nodes.contains(&edge.target()) {
                let consumer = Model::extract_component_name(&graph[edge.target()]);
                let rank = !stateful.contains(consumer.as_str());
                suggestions.insert((rank, variable, consumer));
            }
        }
    }
    Err(RSCMError::CircularDependency {
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
