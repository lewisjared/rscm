use crate::units::Unit;
use thiserror::Error;

/// Error type for invalid operations.
#[derive(Error, Debug)]
pub enum RSCMError {
    #[error("{0}")]
    Error(String),

    /// A Python `KeyboardInterrupt`, `SystemExit` or other non-`Exception` raised in a component,
    /// re-raised unchanged at the Python boundary.
    #[error("{0}")]
    Interrupt(pyo3::PyErr),

    #[error("Extrapolation is not allowed. Target={0}, {1} interpolation range={2}")]
    ExtrapolationNotAllowed(f32, String, f32),

    #[error("Wrong input unit. Expected {0}, got {1}")]
    WrongUnit(String, String),

    /// Grid transformation errors for unsupported conversions
    #[error("Unsupported grid transformation from {from} to {to}. This transformation is not defined because it would require additional physical assumptions. Consider writing a component that explicitly handles this disaggregation, or use an intermediate transformation.")]
    UnsupportedGridTransformation { from: String, to: String },

    /// Grid mismatch between connected components
    #[error("Grid mismatch for variable '{variable}': producer component '{producer_component}' outputs {producer_grid} but consumer component '{consumer_component}' expects {consumer_grid}. \n\nPossible resolutions:\n  1. Change {producer_component} to output {consumer_grid}\n  2. Change {consumer_component} to accept {producer_grid}\n  3. Insert a grid transformation component between them\n  4. If aggregation is acceptable, the model coupler can auto-transform from finer to coarser grids")]
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
    #[error("Variable '{name}' not found in state. Available variables: {available}. Ensure the variable is produced by a component or provided as prescribed input.")]
    VariableNotFound { name: String, available: String },

    /// Invalid region index for a grid
    #[error("Invalid region index {index} for grid {grid}. Valid indices are 0..{max_index}.")]
    InvalidRegionIndex {
        index: usize,
        grid: String,
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
    FeedbackLoop {
        components: Vec<String>,
        variables: Vec<String>,
        suggestions: Vec<(String, String)>,
    },

    /// A `with_lag` or `with_lag_for` call that names no edge of the model, or lags an average.
    #[error("Invalid lag {lag}: {reason}")]
    InvalidLag { lag: String, reason: String },

    /// Inputs that no component produces and no prescribed data or initial value supplies.
    ///
    /// `inputs` holds each `(variable, consumers)` pair, sorted by variable.
    #[error(
        "Inputs with no producer, prescribed data or initial value: {}. Supply them as prescribed data, or call with_missing_prescribed_allowed() to fill them with NaN.",
        format_missing_inputs(inputs)
    )]
    MissingPrescribedInput { inputs: Vec<(String, Vec<String>)> },

    /// A builder with no time axis.
    #[error("The model has no time axis. Set one with with_time_axis() before building.")]
    MissingTimeAxis,

    /// A builder with no components.
    #[error("The model has no components. Add one with with_component() before building.")]
    NoComponents,

    /// An initial value for a variable no component reads or produces.
    #[error(
        "Initial value for '{variable}' is not used: no component reads or produces it.{}",
        format_close_matches(suggestions)
    )]
    UnknownInitialValue {
        variable: String,
        suggestions: Vec<String>,
    },

    /// Prescribed data, given by name, for a variable no component reads.
    #[error(
        "Prescribed variable '{variable}' is not used: no component reads it.{}",
        format_close_matches(suggestions)
    )]
    UnusedPrescribedVariable {
        variable: String,
        suggestions: Vec<String>,
    },

    /// Prescribed data, given by name, for a variable a component produces.
    #[error("Prescribed variable '{variable}' is produced by component '{component}', so the prescribed data would be overwritten. Drop the prescribed data, or remove the component.")]
    PrescribedOutput { variable: String, component: String },

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
    #[error("Grid output mismatch for variable '{variable}': component tried to output {component_grid} but variable expects {expected_grid}. Ensure the component's StateValue variant matches the VariableDefinition's grid.")]
    GridOutputMismatch {
        variable: String,
        expected_grid: String,
        component_grid: String,
    },

    /// Grid transformation not supported (disaggregation/broadcast attempt)
    ///
    /// This error occurs when a component requires a finer grid than
    /// the schema or producer provides. Disaggregation (broadcasting coarse data
    /// to finer grids) is not supported because it would require inventing
    /// spatial structure that doesn't exist in the source data.
    #[error("Grid transformation not supported for variable '{variable}': cannot transform from {source_grid} to {target_grid}. Disaggregation (broadcasting from coarser to finer grids) is not supported because it would require inventing spatial structure. \n\nPossible resolutions:\n  1. Change the consumer component to accept the {source_grid} grid\n  2. Change the producer component or schema to provide the {target_grid} grid\n  3. Create an explicit disaggregation component with domain-specific assumptions")]
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

    /// Grid mismatch between schema aggregate and its contributor
    #[error("Grid mismatch in aggregate '{aggregate}': contributor '{contributor}' has grid '{contributor_grid}' but aggregate expects '{aggregate_grid}'.")]
    SchemaGridTypeMismatch {
        aggregate: String,
        contributor: String,
        contributor_grid: String,
        aggregate_grid: String,
    },

    /// Weight count mismatch for weighted aggregate
    #[error("Weight count mismatch in weighted aggregate '{aggregate}': {n_weights} weights provided but {n_contributors} contributors defined.")]
    WeightCountMismatch {
        aggregate: String,
        n_weights: usize,
        n_contributors: usize,
    },

    /// Circular dependency detected in aggregate graph
    #[error(
        "Circular dependency detected in aggregate schema: {cycle}. Aggregates cannot form cycles."
    )]
    AggregateCircularDependency { cycle: String },

    /// Component output not defined in schema
    #[error("Component '{component}' outputs variable '{variable}' which is not defined in the schema. Add it to the schema with unit '{unit}', or remove the schema constraint.")]
    SchemaUndefinedOutput {
        component: String,
        variable: String,
        unit: String,
    },

    /// Component input not defined in schema or produced by any component
    #[error("Component '{component}' requires variable '{variable}' which is not defined in the schema. Add it to the schema with unit '{unit}', or remove the schema constraint.")]
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

    /// Component output grid mismatch with schema
    #[error("Grid mismatch for variable '{variable}': component '{component}' uses '{component_grid}' but schema defines '{schema_grid}'.")]
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
    #[error("Incompatible units for variable '{variable}': '{unit1}' (normalised '{normalised1}', dimension {dim1}) cannot be converted to '{unit2}' (normalised '{normalised2}', dimension {dim2}).")]
    IncompatibleUnits {
        variable: String,
        unit1: String,
        unit2: String,
        normalised1: String,
        normalised2: String,
        dim1: String,
        dim2: String,
    },
}

impl RSCMError {
    /// Error for two units of `variable` that cannot be converted into each other.
    pub fn incompatible_units(variable: &str, unit1: &Unit, unit2: &Unit) -> Self {
        let dimension = |unit: &Unit| {
            unit.dimension()
                .map(|d| d.to_string())
                .unwrap_or_else(|_| "unknown".to_string())
        };
        RSCMError::IncompatibleUnits {
            variable: variable.to_string(),
            unit1: unit1.original().to_string(),
            unit2: unit2.original().to_string(),
            normalised1: unit1.normalised(),
            normalised2: unit2.normalised(),
            dim1: dimension(unit1),
            dim2: dimension(unit2),
        }
    }
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

/// " Did you mean ...?" for `suggestions`, or nothing if there are none.
pub(crate) fn format_close_matches(suggestions: &[String]) -> String {
    match suggestions {
        [] => String::new(),
        [one] => format!(" Did you mean '{one}'?"),
        many => format!(
            " Did you mean one of {}?",
            many.iter()
                .map(|s| format!("'{s}'"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Up to three of `candidates` that look like `name`, closest first.
pub fn close_matches(name: &str, candidates: &[&str]) -> Vec<String> {
    let mut scored: Vec<(f64, &str)> = candidates
        .iter()
        .map(|c| (strsim::normalized_levenshtein(name, c), *c))
        .filter(|(score, _)| *score >= 0.6)
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(b.1)));
    scored
        .into_iter()
        .take(3)
        .map(|(_, c)| c.to_string())
        .collect()
}
