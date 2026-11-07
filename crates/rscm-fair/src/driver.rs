//! What drives a species or a forcing.

use serde::{Deserialize, Serialize};

/// What drives a species or a forcing, which sets the time it is read at.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Driver {
    /// Emissions over the step.
    Emissions,
    /// Concentration at the end of the step.
    Concentration,
    /// Emissions summed to the end of the step.
    CumulativeEmissions,
}
