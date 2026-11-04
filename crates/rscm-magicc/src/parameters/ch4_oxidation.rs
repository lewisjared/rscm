//! CH4 Oxidation Parameters
//!
//! Parameters for the CO2 released by oxidation of fossil methane.

use crate::parameters::CH4ChemistryParameters;
use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Parameters for the fossil CH4 oxidation source of CO2
///
/// MAGICC7 adds the CO2 produced by oxidising fossil CH4 to fossil CO2
/// emissions (`CH4_INCL_CH4OX`). Biogenic CH4 is excluded because its carbon
/// was recently taken from the atmosphere.
/// The CH4 budget parameters must match `CH4Chemistry`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CH4OxidationParameters {
    /// Mass of carbon per mass of CH4 oxidised
    /// unit: dimensionless
    /// default: 0.75 (12/16)
    pub carbon_per_ch4: FloatValue,

    /// Add `Emissions|CH4|Permafrost` to the CH4 budget.
    /// default: false
    pub permafrost: bool,

    /// Fraction of permafrost CH4 counted as fossil (`PF_CO2FROMCH4OXINATM_FRACTION`)
    /// unit: dimensionless
    /// default: 1.0
    pub permafrost_fraction: FloatValue,

    /// Conversion from ppb to Tg CH4 (`CH4_PPB2TGCH4`)
    /// unit: Tg CH4/ppb
    /// default: 2.824
    pub ppb_to_tg: FloatValue,

    /// Fraction of the atmosphere mixed with surface CH4 (`CH4_MIXBOXSIZE`)
    /// unit: dimensionless
    /// default: 0.973
    pub mixbox_size: FloatValue,

    /// Combined soil, stratospheric and tropospheric chlorine lifetime
    /// unit: yr
    /// default: `CH4ChemistryParameters::tau_other`
    pub tau_other: FloatValue,

    /// Natural CH4 emissions
    /// unit: Mt CH4/yr
    /// default: 182.1908
    pub natural_emissions: FloatValue,
}

impl Default for CH4OxidationParameters {
    fn default() -> Self {
        let chemistry = CH4ChemistryParameters::default();
        Self {
            carbon_per_ch4: 12.0 / 16.0,
            permafrost: false,
            permafrost_fraction: 1.0,
            ppb_to_tg: chemistry.ppb_to_tg,
            mixbox_size: chemistry.mixbox_size,
            tau_other: chemistry.tau_other(),
            natural_emissions: chemistry.natural_emissions,
        }
    }
}
