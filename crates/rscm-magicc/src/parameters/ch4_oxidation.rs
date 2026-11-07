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
#[serde(deny_unknown_fields)]
pub struct CH4OxidationParameters {
    /// Mass of carbon per mass of CH4 oxidised, 12/16 by default
    ///
    /// unit: dimensionless
    pub carbon_per_ch4: FloatValue,

    /// Add `Emissions|CH4|Permafrost` to the CH4 budget (MAGICC7 `PF_APPLY`)
    pub include_permafrost: bool,

    /// Fraction of permafrost CH4 counted as fossil (MAGICC7 `PF_CO2FROMCH4OXINATM_FRACTION`)
    ///
    /// unit: dimensionless
    pub permafrost_fraction: FloatValue,

    /// Conversion factor from ppb to Tg CH4 (MAGICC7 `CH4_PPB2TGCH4`)
    ///
    /// unit: Tg CH4/ppb
    pub tg_per_ppb: FloatValue,

    /// Fraction of the atmosphere mixed with surface CH4 (MAGICC7 `CH4_MIXBOXSIZE`)
    ///
    /// unit: dimensionless
    pub mixing_box_size: FloatValue,

    /// Combined soil, stratospheric and tropospheric chlorine lifetime.
    ///
    /// unit: yr
    ///
    /// The default combines the `CH4ChemistryParameters` sink lifetimes
    /// (MAGICC7 `CH4_TAUSOIL`, `CH4_TAUSTRAT` and `CH4_TAUTROPCL`).
    pub non_oh_lifetime: FloatValue,

    /// Natural CH4 emissions
    ///
    /// unit: Mt CH4/yr
    pub natural_emissions: FloatValue,
}

impl Default for CH4OxidationParameters {
    fn default() -> Self {
        let chemistry = CH4ChemistryParameters::default();
        Self {
            carbon_per_ch4: 12.0 / 16.0,
            include_permafrost: false,
            permafrost_fraction: 1.0,
            tg_per_ppb: chemistry.tg_per_ppb,
            mixing_box_size: chemistry.mixing_box_size,
            non_oh_lifetime: chemistry.non_oh_lifetime(),
            natural_emissions: chemistry.natural_emissions,
        }
    }
}
