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
    #[error("Missing initial value for state variable '{variable}' in component '{component}'. State variables require an initial value. Provide one with builder.with_initial_values({{\"{variable}\": value}}), or with_initial_grid_values for a grid variable.")]
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

    /// A feedback loop that no lag breaks.
    ///
    /// `suggestions` holds the `(variable, consumer)` pairs whose `with_lag_for` call would break it,
    /// edges into a component with its own state first.
    #[error(
        "Feedback loop among components [{}] through variables [{}]. Break it with one of: {}",
        components.join(", "),
        variables.join(", "),
        format_lag_suggestions(suggestions)
    )]
    CircularDependency {
        components: Vec<String>,
        variables: Vec<String>,
        suggestions: Vec<(String, String)>,
    },

    /// A `with_lag` or `with_lag_for` call that names no edge of the model, or lags an average.
    #[error("Invalid lag {lag}: {reason}")]
    InvalidLag { lag: String, reason: String },

    /// Inputs that no component produces and no exogenous data or initial value supplies.
    ///
    /// `inputs` holds each `(variable, consumers)` pair, sorted by variable.
    #[error(
        "Inputs with no producer, exogenous data or initial value: {}. Supply them as exogenous data, or call with_missing_exogenous_allowed() to fill them with NaN.",
        format_missing_inputs(inputs)
    )]
    MissingExogenousInput { inputs: Vec<(String, Vec<String>)> },

    /// Two components produce the same variable.
    #[error("Variable '{variable}' has more than one producer: {}", components.join(", "))]
    MultipleProducers {
        variable: String,
        components: Vec<String>,
    },

    /// A declared timeseries type that disagrees with the variable's resolved type.
    #[error("Timeseries type mismatch for variable '{variable}': {producer} gives {producer_type} but {consumer} declares {consumer_type}. Consumers inherit the producer's type, so drop the declaration or make them agree.")]
    TimeseriesTypeMismatch {
        variable: String,
        producer: String,
        producer_type: String,
        consumer: String,
        consumer_type: String,
    },

    /// A state variable that resolves to an average.
    #[error("State variable '{variable}' of component '{component}' is an average, but states must be points. Declare it a point, or turn the check off with ModelBuilder::with_state_type_check(false).")]
    AverageState { variable: String, component: String },

    /// Grid output type mismatch between component output and variable definition
    #[error("Grid output mismatch for variable '{variable}': component tried to output {component_grid} but variable expects {expected_grid}. Ensure the component's StateValue variant matches the RequirementDefinition's grid_type.")]
    GridOutputMismatch {
        variable: String,
        expected_grid: String,
        component_grid: String,
    },

    /// Grid transformation not supported (disaggregation/broadcast attempt)
    ///
    /// This error occurs when a component requires a finer grid resolution than
    /// the schema or producer provides. Disaggregation (broadcasting coarse data
    /// to finer grids) is not supported because it would require inventing
    /// spatial structure that doesn't exist in the source data.
    #[error("Grid transformation not supported for variable '{variable}': cannot transform from {source_grid} to {target_grid}. Disaggregation (broadcasting from coarser to finer grids) is not supported because it would require inventing spatial structure. \n\nPossible resolutions:\n  1. Change the consumer component to accept {source_grid} resolution\n  2. Change the producer component or schema to provide {target_grid} resolution\n  3. Create an explicit disaggregation component with domain-specific assumptions")]
    GridTransformationNotSupported {
        variable: String,
        source_grid: String,
        target_grid: String,
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

    /// Failed to parse a unit string
    #[error("Failed to parse unit '{unit_string}' for variable '{variable}': {details}")]
    UnitParseError {
        variable: String,
        unit_string: String,
        details: String,
    },

    /// Units have incompatible physical dimensions
    ///
    /// This error occurs when two components use units with different physical dimensions
    /// for the same variable. For example, one component using "W/m^2" (radiative flux)
    /// and another using "GtC/yr" (mass rate) - these cannot be converted between.
    #[error("Incompatible units for variable '{variable}': '{unit1}' has dimension {dim1} but '{unit2}' has dimension {dim2}. These units measure different physical quantities and cannot be converted.")]
    IncompatibleUnits {
        variable: String,
        unit1: String,
        unit2: String,
        dim1: String,
        dim2: String,
    },
}

/// Convenience type for `Result<T, RSCMError>`.
pub type RSCMResult<T> = Result<T, RSCMError>;

fn format_lag_suggestions(suggestions: &[(String, String)]) -> String {
    suggestions
        .iter()
        .map(|(variable, consumer)| format!("with_lag_for({variable:?}, {consumer:?})"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_missing_inputs(inputs: &[(String, Vec<String>)]) -> String {
    inputs
        .iter()
        .map(|(variable, consumers)| format!("'{variable}' (read by {})", consumers.join(", ")))
        .collect::<Vec<_>>()
        .join(", ")
}
