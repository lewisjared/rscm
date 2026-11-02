//! CH4 Oxidation Parameters
//!
//! Parameters for the CO2 released by oxidation of fossil methane.

use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Parameters for the fossil CH4 oxidation source of CO2
///
/// MAGICC7 adds the CO2 produced by oxidising fossil CH4 to fossil CO2
/// emissions (`CH4_INCL_CH4OX`). Biogenic CH4 is excluded because its carbon
/// was recently taken from the atmosphere.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CH4OxidationParameters {
    /// Mass of carbon per mass of CH4 oxidised
    /// unit: dimensionless
    /// default: 0.75 (12/16)
    pub carbon_per_ch4: FloatValue,
}

impl Default for CH4OxidationParameters {
    fn default() -> Self {
        Self {
            carbon_per_ch4: 12.0 / 16.0,
        }
    }
}
