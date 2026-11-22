//! Nitrogen limitation parameters
//!
//! Parameters for the MAGICC7 nitrogen limitation of NPP.

use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Michaelis-Menten nitrogen fixation, calculated from NPP.
#[rscm_core::params]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrognosticFixation {
    /// Maximum fixation rate (MAGICC7 `NCYCLE_NFIX_VMAX`)
    ///
    /// unit: Tg N/yr
    pub maximum_fixation_rate: FloatValue,
    /// NPP at half the maximum fixation rate (MAGICC7 `NCYCLE_NFIX_KM`)
    ///
    /// unit: GtC/yr
    pub half_saturation_npp: FloatValue,
}

/// Nitrogen deposition calculated from NOx and NH3 emissions.
#[rscm_core::params]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct DepositionFromEmissions {
    /// Share of the emitted nitrogen deposited on land (MAGICC7 `NCYCLE_FRAC_NDEP_LAND`)
    ///
    /// unit: dimensionless
    pub land_fraction: FloatValue,
    /// Reproduce MAGICC7 v7.5.3 counting industrial NH3 twice and leaving out biomass NH3
    pub include_nh3_counted_twice: bool,
}

impl Default for DepositionFromEmissions {
    fn default() -> Self {
        Self {
            land_fraction: 0.7,
            include_nh3_counted_twice: true,
        }
    }
}

impl DepositionFromEmissions {
    /// Deposition (Gt N/yr) from fossil and biosphere NOx and NH3 emissions (Mt N/yr).
    pub fn deposition(&self, nox: [FloatValue; 2], nh3: [FloatValue; 2]) -> FloatValue {
        let nh3_biosphere = if self.include_nh3_counted_twice {
            nh3[0]
        } else {
            nh3[1]
        };
        1e-3 * self.land_fraction * ((nox[0] + nox[1]) + (nh3[0] + nh3_biosphere))
    }
}

/// Parameters for the nitrogen limitation of NPP.
///
/// Each year the plant-available nitrogen pool is solved for the steady state of
/// deposition, recycling and fixation against plant uptake and losses.
/// From `fertilisation_start_year` NPP is scaled by the nitrogen availability
/// relative to the last year before it.
#[rscm_core::params]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct NitrogenParameters {
    /// Plant nitrogen uptake per unit NPP (MAGICC7 `NCYCLE_NCRATIO_UPTAKE`)
    ///
    /// unit: Gt N/GtC
    pub uptake_nc_ratio: FloatValue,
    /// Nitrogen recycled per unit heterotrophic respiration (MAGICC7 `NCYCLE_NCRATIO_RECYC`)
    ///
    /// unit: Gt N/GtC
    pub recycling_nc_ratio: FloatValue,
    /// Damping of the CO2 and temperature effects on the recycled N:C ratio
    /// (MAGICC7 `NCYCLE_RECYC2CNRATIO_FACTOR`)
    ///
    /// unit: dimensionless
    pub recycling_cn_factor: FloatValue,
    /// Half saturation of nitrogen supply in the availability function
    /// (MAGICC7 `NCYCLE_NCRATIO_UPTAKE_SUPPLY`)
    ///
    /// unit: Gt N
    pub uptake_supply: FloatValue,
    /// Sensitivity of nitrogen availability, $A$ (MAGICC7 `NCYCLE_NPP2NAVAIL_FACTOR_A`)
    ///
    /// unit: dimensionless
    pub availability_factor_a: FloatValue,
    /// Steepness of nitrogen availability, $C$ (MAGICC7 `NCYCLE_NPP2NAVAIL_FACTOR_C`)
    ///
    /// unit: dimensionless
    pub availability_factor_c: FloatValue,
    /// Residence time of the nitrogen pool against losses (MAGICC7 `NCYCLE_TURNOVERTIME_LOSSES`)
    ///
    /// unit: yr
    pub loss_turnover_time: FloatValue,
    /// Fixation from NPP, or `None` to read `Nitrogen Flux|Fixation`
    /// (MAGICC7 `NCYCLE_APPLY_PROGNOSTIC_NFIX`)
    pub prognostic_fixation: Option<PrognosticFixation>,
    /// Deposition from NOx and NH3 emissions, or `None` to read `Nitrogen Flux|Deposition`
    /// (MAGICC7 `NCYCLE_APPLY_NDEP_FROM_EMIS`)
    pub deposition_from_emissions: Option<DepositionFromEmissions>,
}

impl Default for NitrogenParameters {
    fn default() -> Self {
        Self {
            uptake_nc_ratio: 0.013,
            recycling_nc_ratio: 0.013,
            recycling_cn_factor: 0.0,
            uptake_supply: 0.05,
            availability_factor_a: 2.0,
            availability_factor_c: 2.0,
            loss_turnover_time: 2.0,
            prognostic_fixation: None,
            deposition_from_emissions: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deposition_counts_nh3_as_magicc7_does() {
        let reproduced = DepositionFromEmissions {
            land_fraction: 1.0,
            include_nh3_counted_twice: true,
        };
        let fixed = DepositionFromEmissions {
            include_nh3_counted_twice: false,
            ..reproduced.clone()
        };
        assert_eq!(reproduced.deposition([0.0, 0.0], [1.0, 2.0]), 2e-3);
        assert_eq!(fixed.deposition([0.0, 0.0], [1.0, 2.0]), 3e-3);
    }

    #[test]
    fn test_deposition_scales_with_land_fraction() {
        let deposition = DepositionFromEmissions {
            land_fraction: 0.5,
            include_nh3_counted_twice: false,
        };
        assert_eq!(
            deposition.deposition([2.0, 4.0], [1.0, 1.0]),
            1e-3 * 0.5 * 8.0
        );
    }
}
