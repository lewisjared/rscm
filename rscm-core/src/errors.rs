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
    #[error("Grid type mismatch for variable '{variable}': producer outputs {producer_grid} but consumer expects {consumer_grid}. Use a grid transformation component or ensure matching grid types.")]
    GridTypeMismatch {
        variable: String,
        producer_grid: String,
        consumer_grid: String,
    },

    /// Missing initial value for a state variable
    #[error("Missing initial value for state variable '{variable}' in component '{component}'. State variables (InputAndOutput) require an initial value. Use ModelBuilder::with_initial_value(\"{variable}\", value) to provide one, or set a default in the component's parameter configuration.")]
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
}

/// Convenience type for `Result<T, EosError>`.
pub type RSCMResult<T> = Result<T, RSCMError>;
