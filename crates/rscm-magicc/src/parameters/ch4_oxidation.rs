//! CH4 Oxidation Parameters
//!
//! Parameters for the CO2 released by oxidation of fossil methane.

use crate::parameters::{ch4_non_oh_lifetime, CH4ChemistryParameters};
use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Parameters for the fossil CH4 oxidation source of CO2
///
/// MAGICC7 adds the CO2 produced by oxidising fossil CH4 to fossil CO2
/// emissions (`CH4_INCL_CH4OX`). Biogenic CH4 is excluded because its carbon
/// was recently taken from the atmosphere.
/// The CH4 budget parameters must match `CH4Chemistry`.
#[rscm_core::params]
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

    /// Soil sink lifetime (MAGICC7 `CH4_TAUSOIL`)
    ///
    /// unit: yr
    pub soil_lifetime: FloatValue,

    /// Stratospheric sink lifetime (MAGICC7 `CH4_TAUSTRAT`)
    ///
    /// unit: yr
    pub stratospheric_lifetime: FloatValue,

    /// Tropospheric Cl sink lifetime (MAGICC7 `CH4_TAUTROPCL`)
    ///
    /// unit: yr
    pub tropospheric_cl_lifetime: FloatValue,

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
            soil_lifetime: chemistry.soil_lifetime,
            stratospheric_lifetime: chemistry.stratospheric_lifetime,
            tropospheric_cl_lifetime: chemistry.tropospheric_cl_lifetime,
            natural_emissions: chemistry.natural_emissions,
        }
    }
}

impl CH4OxidationParameters {
    /// Combined soil, stratospheric and tropospheric chlorine lifetime, see [`ch4_non_oh_lifetime`]
    pub fn non_oh_lifetime(&self) -> FloatValue {
        ch4_non_oh_lifetime(
            self.soil_lifetime,
            self.stratospheric_lifetime,
            self.tropospheric_cl_lifetime,
        )
    }

    /// Initial natural CH4 burden at `concentration` ppb, with no anthropogenic CH4 in the atmosphere
    ///
    /// unit: Mt CH4
    pub fn initial_natural_burden(&self, concentration: FloatValue) -> FloatValue {
        self.tg_per_ppb * concentration * self.mixing_box_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::HalocarbonChemistryParameters;

    #[test]
    fn test_derived_lifetimes_follow_the_stated_ch4_lifetimes() {
        let chemistry = CH4ChemistryParameters {
            soil_lifetime: 60.0,
            stratospheric_lifetime: 100.0,
            tropospheric_cl_lifetime: 250.0,
            initial_ch4_lifetime: 9.0,
            ..Default::default()
        };
        let oxidation = CH4OxidationParameters {
            soil_lifetime: 60.0,
            stratospheric_lifetime: 100.0,
            tropospheric_cl_lifetime: 250.0,
            ..Default::default()
        };
        let halocarbon = HalocarbonChemistryParameters {
            initial_ch4_lifetime: 9.0,
            ch4_soil_lifetime: 60.0,
            ch4_stratospheric_lifetime: 100.0,
            ch4_tropospheric_cl_lifetime: 250.0,
            ..Default::default()
        };
        let non_oh = 1.0 / (1.0 / 60.0 + 1.0 / 100.0 + 1.0 / 250.0);
        assert_eq!(oxidation.non_oh_lifetime(), non_oh);
        assert_eq!(oxidation.non_oh_lifetime(), chemistry.non_oh_lifetime());
        assert_eq!(
            halocarbon.initial_ch4_oh_lifetime(),
            1.0 / (1.0 / 9.0 - 1.0 / non_oh)
        );
        assert_eq!(
            halocarbon.initial_ch4_oh_lifetime(),
            chemistry.initial_oh_lifetime()
        );
    }

    #[test]
    fn test_initial_natural_burden() {
        let p = CH4OxidationParameters {
            tg_per_ppb: 2.0,
            mixing_box_size: 0.5,
            ..Default::default()
        };
        assert_eq!(p.initial_natural_burden(700.0), 2.0 * 700.0 * 0.5);
    }
}
