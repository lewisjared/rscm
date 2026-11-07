//! Nitrogen limitation parameters
//!
//! Parameters for the MAGICC7 nitrogen limitation of NPP.

use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Michaelis-Menten nitrogen fixation, calculated from NPP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrognosticFixationParameters {
    /// Maximum fixation rate
    /// unit: Tg N/yr
    /// MAGICC7: NCYCLE_NFIX_VMAX
    pub maximum_fixation_rate: FloatValue,
    /// NPP at half the maximum fixation rate
    /// unit: GtC/yr
    /// MAGICC7: NCYCLE_NFIX_KM
    pub half_saturation_npp: FloatValue,
}

/// Parameters for the nitrogen limitation of NPP.
///
/// Each year the plant-available nitrogen pool is solved for the steady state of
/// deposition, recycling and fixation against plant uptake and losses.
/// From `fertilisation_start_year` NPP is scaled by the nitrogen availability
/// relative to the last year before it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NitrogenParameters {
    /// Plant nitrogen uptake per unit NPP
    /// unit: Gt N/GtC
    /// MAGICC7: NCYCLE_NCRATIO_UPTAKE
    pub uptake_nc_ratio: FloatValue,
    /// Nitrogen recycled per unit heterotrophic respiration
    /// unit: Gt N/GtC
    /// MAGICC7: NCYCLE_NCRATIO_RECYC
    pub recycling_nc_ratio: FloatValue,
    /// Damping of the CO2 and temperature effects on the recycled N:C ratio
    /// MAGICC7: NCYCLE_RECYC2CNRATIO_FACTOR
    pub recycling_cn_factor: FloatValue,
    /// Half saturation of nitrogen supply in the availability function
    /// unit: Gt N
    /// MAGICC7: NCYCLE_NCRATIO_UPTAKE_SUPPLY
    pub uptake_supply: FloatValue,
    /// Sensitivity of nitrogen availability (A)
    /// MAGICC7: NCYCLE_NPP2NAVAIL_FACTOR_A
    pub availability_factor_a: FloatValue,
    /// Steepness of nitrogen availability (C)
    /// MAGICC7: NCYCLE_NPP2NAVAIL_FACTOR_C
    pub availability_factor_c: FloatValue,
    /// Residence time of the nitrogen pool against losses
    /// unit: yr
    /// MAGICC7: NCYCLE_TURNOVERTIME_LOSSES
    pub loss_turnover_time: FloatValue,
    /// Fixation from NPP, or `None` to read `Nitrogen Flux|Fixation`
    /// MAGICC7: NCYCLE_APPLY_PROGNOSTIC_NFIX
    pub prognostic_fixation: Option<PrognosticFixationParameters>,
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
        }
    }
}
