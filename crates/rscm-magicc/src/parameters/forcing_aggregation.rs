//! Forcing Aggregation Parameters

use super::EfficacyMethod;
use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// A forcing agent read by `ForcingAggregation`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForcingAgent {
    /// Name of the agent's ERF variable (W/m²).
    pub variable: String,

    /// Whether the ERF is a four-box variable. Global ERF is spread with `regional_pattern`.
    #[serde(default)]
    pub regional: bool,

    /// Variables summed in order into this agent before its efficacy applies,
    /// as MAGICC7 groups biomass burning aerosols. Empty reads `variable` itself.
    #[serde(default)]
    pub sum_of: Vec<String>,

    /// Apply the efficacy to each `sum_of` variable before summing, as MAGICC7 does for halocarbons.
    #[serde(default)]
    pub efficacy_per_variable: bool,

    /// Regional forcing pattern (NH ocean, NH land, SH ocean, SH land), as `RF_REGIONS_*`.
    /// It spreads global ERF over the boxes and, with `climate`, sets the internal efficacy.
    #[serde(default = "uniform")]
    pub regional_pattern: [FloatValue; 4],

    /// Normalise `regional_pattern` within each hemisphere instead of globally,
    /// so both hemispheres carry the global mean (MAGICC7 tropospheric ozone).
    #[serde(default)]
    pub hemispheric: bool,

    /// Take the internal efficacy from LAMCALC's CO2 response, as MAGICC7 does for CO2.
    #[serde(default)]
    pub co2_efficacy: bool,

    /// Prescribed efficacy (`RF_EFFICACY_*`).
    #[serde(default = "one")]
    pub efficacy: FloatValue,

    /// Internal efficacy of the agent's pattern (`RF_INTEFFICACY_*`), unused with `climate`.
    #[serde(default = "one")]
    pub internal_efficacy: FloatValue,
}

/// Parameters for `ForcingAggregation`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ForcingAggregationParameters {
    /// Agents summed into the total.
    pub agents: Vec<ForcingAgent>,

    /// Order in which agents are summed, by `variable`, with nested lists summed first.
    /// MAGICC7 sums its forcing groups this way. Empty sums the agents in order.
    pub summation: Vec<ForcingTerm>,

    /// How efficacies scale each agent's forcing (MAGICC7 `RF_EFFICACY_APPLY`).
    pub efficacy_method: EfficacyMethod,

    /// Northern Hemisphere land fraction.
    pub nh_land_fraction: FloatValue,

    /// Southern Hemisphere land fraction.
    pub sh_land_fraction: FloatValue,

    /// Recalculate internal efficacies each year from the climate sensitivity,
    /// as MAGICC7 does, instead of using each agent's `internal_efficacy`.
    pub climate: Option<EfficacyClimate>,
}

/// An agent or a nested sum in `ForcingAggregationParameters::summation`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ForcingTerm {
    /// An agent, by its `variable`.
    Agent(String),
    /// Terms summed left to right.
    Sum(Vec<ForcingTerm>),
}

/// Climate settings for recalculating internal efficacies.
///
/// Each year the efficacies come from LAMCALC at the previous year's
/// `Equilibrium Climate Sensitivity` and `Effective Radiative Forcing|2xCO2`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EfficacyClimate {
    /// Climate sensitivity used before the climate model reports one (K).
    pub ecs: FloatValue,
    /// CO2 doubling forcing used before it is first calculated (W/m²).
    pub erf_2xco2: FloatValue,
    /// Land-ocean heat exchange coefficient (W/m²/K).
    /// MAGICC7: CORE_HEATXCHANGE_LANDOCEAN
    pub land_ocean_heat_exchange: FloatValue,
    /// Inter-hemispheric heat exchange coefficient (W/m²/K).
    /// MAGICC7: CORE_HEATXCHANGE_NORTHSOUTH
    pub north_south_heat_exchange: FloatValue,
    /// Land-ocean warming ratio.
    /// MAGICC7: CORE_RLO
    pub land_ocean_warming_ratio: FloatValue,
    /// Ocean-to-land heat exchange amplification.
    pub amplify_ocean_to_land: FloatValue,
    /// Regional CO2 forcing pattern.
    /// MAGICC7: RF_REGIONS_CO2
    pub co2_regional_pattern: [FloatValue; 4],
}

impl Default for EfficacyClimate {
    fn default() -> Self {
        let p = crate::parameters::ClimateUdebParameters::default();
        Self {
            ecs: p.ecs,
            erf_2xco2: p.erf_2xco2,
            land_ocean_heat_exchange: p.land_ocean_heat_exchange,
            north_south_heat_exchange: p.north_south_heat_exchange,
            land_ocean_warming_ratio: p.land_ocean_warming_ratio,
            amplify_ocean_to_land: p.amplify_ocean_to_land,
            co2_regional_pattern: p.co2_regional_pattern,
        }
    }
}

impl Default for ForcingAggregationParameters {
    fn default() -> Self {
        Self {
            agents: Vec::new(),
            summation: Vec::new(),
            efficacy_method: EfficacyMethod::Ar6,
            nh_land_fraction: 0.42,
            sh_land_fraction: 0.21,
            climate: None,
        }
    }
}

impl ForcingAggregationParameters {
    /// Global area fractions of the four boxes.
    pub fn area_fractions(&self) -> [FloatValue; 4] {
        [
            (1.0 - self.nh_land_fraction) / 2.0,
            self.nh_land_fraction / 2.0,
            (1.0 - self.sh_land_fraction) / 2.0,
            self.sh_land_fraction / 2.0,
        ]
    }
}

fn uniform() -> [FloatValue; 4] {
    [1.0; 4]
}

fn one() -> FloatValue {
    1.0
}
