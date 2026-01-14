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
    #[error("Unsupported grid transformation from {from} to {to}. This transformation is not defined because it would require additional physical assumptions. Consider creating a custom component that explicitly handles this disaggregation, or use an intermediate transformation.")]
    UnsupportedGridTransformation { from: String, to: String },
}

/// Convenience type for `Result<T, EosError>`.
pub type RSCMResult<T> = Result<T, RSCMError>;
