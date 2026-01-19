use thiserror::Error;

/// Error type for invalid operations.
#[derive(Error, Debug)]
pub enum RSCMError {
    #[error("{0}")]
    Error(String),

    #[error("Extrapolation is not allowed. Target={0}, {1} interpolation range={2}")]
    ExtrapolationNotAllowed(f32, String, f32),

    #[error("Wrong input units. Expected {0}, got {1}")]
    WrongUnits(String, String),

    /// Grid transformation errors for unsupported conversions
    #[error("Unsupported grid transformation from {from} to {to}. This transformation is not defined because it would require additional physical assumptions. Consider creating a custom component that explicitly handles this disaggregation, or use an intermediate transformation.")]
    UnsupportedGridTransformation { from: String, to: String },

    /// Grid type mismatch between connected components
    #[error("Grid type mismatch for variable '{variable}': producer component '{producer_component}' outputs {producer_grid} but consumer component '{consumer_component}' expects {consumer_grid}. \n\nPossible resolutions:\n  1. Change {producer_component} to output {consumer_grid}\n  2. Change {consumer_component} to accept {producer_grid}\n  3. Insert a grid transformation component between them\n  4. If aggregation is acceptable, the model coupler can auto-transform from finer to coarser grids")]
    GridTypeMismatch {
        variable: String,
        producer_component: String,
        consumer_component: String,
        producer_grid: String,
        consumer_grid: String,
    },

    /// Missing initial value for a state variable
    #[error("Missing initial value for state variable '{variable}' in component '{component}'. State variables require an initial value. Use ModelBuilder::with_initial_value(\"{variable}\", value) to provide one, or set a default in the component's parameter configuration.")]
    MissingInitialValue { variable: String, component: String },

    /// Variable not found in state
    #[error("Variable '{name}' not found in state. Available variables: {available}. Ensure the variable is produced by a component or provided as exogenous input.")]
    VariableNotFound { name: String, available: String },

    /// Invalid region index for grid type
    #[error(
        "Invalid region index {index} for grid type {grid_type}. Valid indices are 0..{max_index}."
    )]
    InvalidRegionIndex {
        index: usize,
        grid_type: String,
        max_index: usize,
    },

    /// Component cycle detected in dependency graph
    #[error("Circular dependency detected in component graph: {cycle}. Components cannot form cycles. Consider splitting the cycle by introducing intermediate state variables or restructuring the component dependencies.")]
    CircularDependency { cycle: String },

    /// Grid output type mismatch between component output and variable definition
    #[error("Grid output mismatch for variable '{variable}': component tried to output {component_grid} but variable expects {expected_grid}. Ensure the component's StateValue variant matches the RequirementDefinition's grid_type.")]
    GridOutputMismatch {
        variable: String,
        expected_grid: String,
        component_grid: String,
    },

    /// Aggregate contributor not defined in schema
    #[error("Undefined contributor '{contributor}' in aggregate '{aggregate}'. The contributor must be defined as a variable or aggregate in the schema before it can be used.")]
    UndefinedContributor {
        contributor: String,
        aggregate: String,
    },

    /// Unit mismatch between schema variable/aggregate and its contributor
    #[error("Unit mismatch in aggregate '{aggregate}': contributor '{contributor}' has unit '{contributor_unit}' but aggregate expects '{aggregate_unit}'.")]
    SchemaUnitMismatch {
        aggregate: String,
        contributor: String,
        contributor_unit: String,
        aggregate_unit: String,
    },

    /// Grid type mismatch between schema aggregate and its contributor
    #[error("Grid type mismatch in aggregate '{aggregate}': contributor '{contributor}' has grid type '{contributor_grid}' but aggregate expects '{aggregate_grid}'.")]
    SchemaGridTypeMismatch {
        aggregate: String,
        contributor: String,
        contributor_grid: String,
        aggregate_grid: String,
    },

    /// Weight count mismatch for weighted aggregate
    #[error("Weight count mismatch in weighted aggregate '{aggregate}': {weight_count} weights provided but {contributor_count} contributors defined.")]
    WeightCountMismatch {
        aggregate: String,
        weight_count: usize,
        contributor_count: usize,
    },

    /// Circular dependency detected in aggregate graph
    #[error(
        "Circular dependency detected in aggregate schema: {cycle}. Aggregates cannot form cycles."
    )]
    AggregateCircularDependency { cycle: String },

    /// Component output not defined in schema
    #[error("Component '{component}' outputs variable '{variable}' which is not defined in the schema. Add it with schema.variable(\"{variable}\", \"{unit}\") or remove the schema constraint.")]
    SchemaUndefinedOutput {
        component: String,
        variable: String,
        unit: String,
    },

    /// Component input not defined in schema or produced by any component
    #[error("Component '{component}' requires variable '{variable}' which is not defined in the schema. Add it with schema.variable(\"{variable}\", \"{unit}\") or remove the schema constraint.")]
    SchemaUndefinedInput {
        component: String,
        variable: String,
        unit: String,
    },

    /// Component output unit mismatch with schema
    #[error("Unit mismatch for variable '{variable}': component '{component}' uses '{component_unit}' but schema defines '{schema_unit}'.")]
    ComponentSchemaUnitMismatch {
        variable: String,
        component: String,
        component_unit: String,
        schema_unit: String,
    },

    /// Component output grid type mismatch with schema
    #[error("Grid type mismatch for variable '{variable}': component '{component}' uses '{component_grid}' but schema defines '{schema_grid}'.")]
    ComponentSchemaGridMismatch {
        variable: String,
        component: String,
        component_grid: String,
        schema_grid: String,
    },
}

/// Convenience type for `Result<T, RSCMError>`.
pub type RSCMResult<T> = Result<T, RSCMError>;
