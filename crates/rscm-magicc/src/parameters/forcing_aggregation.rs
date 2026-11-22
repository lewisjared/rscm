//! Forcing Aggregation Parameters

use super::EfficacyMethod;
use rscm_core::forcing::ForcingCategory;
use rscm_core::spatial::FourBoxGrid;
use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// A forcing agent read by `ForcingAggregation`.
#[rscm_core::params]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForcingAgent {
    /// Name of the agent's ERF variable, in W/m^2
    pub variable: String,

    /// Subtotals the agent's ERF counts towards.
    pub category: ForcingCategory,

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

    /// Regional ERF pattern in each region of the four-box grid (MAGICC7 `RF_REGIONS_*`).
    ///
    /// unit: dimensionless
    ///
    /// It spreads global ERF over the regions and, with `climate`, sets the internal efficacy.
    #[serde(default = "uniform")]
    pub regional_pattern: [FloatValue; 4],

    /// Normalise `regional_pattern` within each hemisphere instead of globally,
    /// so both hemispheres carry the global mean (MAGICC7 tropospheric ozone).
    #[serde(default)]
    pub hemispheric: bool,

    /// Take the internal efficacy from LAMCALC's CO2 response, as MAGICC7 does for CO2.
    #[serde(default)]
    pub use_co2_internal_efficacy: bool,

    /// Factor on each variable's ERF before the efficacy applies, such as MAGICC7's `RF_SOLAR_SCALE`.
    #[serde(default = "one")]
    pub scale: FloatValue,

    /// Efficacy (MAGICC7 `RF_EFFICACY_*`)
    ///
    /// unit: dimensionless
    #[serde(default = "one")]
    pub efficacy: FloatValue,

    /// Internal efficacy of the agent's pattern, unused with `climate` (MAGICC7 `RF_INTEFFICACY_*`)
    ///
    /// unit: dimensionless
    #[serde(default = "one")]
    pub internal_efficacy: FloatValue,

    /// Scale the ERF to a target change between two years instead of by `scale`
    /// (MAGICC7 `RF_*_APPLY = 1`). Needs a prescribed `variable` and no `sum_of`.
    #[serde(default)]
    pub harmonisation: Option<Harmonisation>,

    /// For each `sum_of` variable, the `variable` of the agent whose scale applies to it,
    /// or an empty string for this agent's own, as MAGICC7 scales biomass aerosols by industrial factors.
    #[serde(default)]
    pub sum_of_scale_from: Vec<String>,

    /// Pattern that spreads a global ERF over the regions in proportion to its area mean,
    /// in place of `regional_pattern`, which then sets only the internal efficacy
    /// (MAGICC7 `RF_REGIONS_DUST`).
    ///
    /// unit: dimensionless
    #[serde(default)]
    pub spreading_pattern: Option<[FloatValue; 4]>,

    /// Hold the ERF at its value in this year afterwards (MAGICC7 `RF_*_CONSTANTAFTERYR`).
    /// Needs a prescribed `variable`.
    ///
    /// unit: yr
    #[serde(default)]
    pub constant_after_year: Option<FloatValue>,

    /// Subtract the scaled ERF in this year, which MAGICC7 does for black carbon on snow.
    /// Needs a prescribed `variable`.
    ///
    /// unit: yr
    #[serde(default)]
    pub relative_to_year: Option<FloatValue>,
}

/// Harmonisation of an agent's ERF to a target change since a pre-industrial year.
///
/// The scale is `target` over the change in the agent's global ERF
/// from `preindustrial_reference_year` to `reference_year`, read from its prescribed input.
#[rscm_core::params]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Harmonisation {
    /// Global ERF change to harmonise to (MAGICC7 `RF_*_WM2`)
    ///
    /// unit: W/m^2
    pub target: FloatValue,

    /// Year of the target (MAGICC7 `RF_*_YR`)
    ///
    /// unit: yr
    pub reference_year: FloatValue,

    /// Year the change is measured from (MAGICC7 `RF_PREIND_REFERENCEYR`)
    ///
    /// unit: yr
    #[serde(default = "preindustrial_reference_year")]
    pub preindustrial_reference_year: FloatValue,
}

fn preindustrial_reference_year() -> FloatValue {
    1750.0
}

/// Parameters for `ForcingAggregation`.
#[rscm_core::params]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct ForcingAggregationParameters {
    /// Agents summed into the total.
    ///
    /// `rscm.magicc.full.forcing_agents` builds MAGICC7's agents.
    pub agents: Vec<ForcingAgent>,

    /// Order in which agents are summed, by `variable`, with nested lists summed first.
    ///
    /// MAGICC7 sums its forcing groups this way.
    /// Empty sums the agents in order.
    pub summation: Vec<ForcingTerm>,

    /// How efficacies scale each agent's ERF (MAGICC7 `RF_EFFICACY_APPLY`)
    pub efficacy_method: EfficacyMethod,

    /// Northern Hemisphere land fraction
    ///
    /// unit: dimensionless
    pub nh_land_fraction: FloatValue,

    /// Southern Hemisphere land fraction
    ///
    /// unit: dimensionless
    pub sh_land_fraction: FloatValue,

    /// Recalculate internal efficacies each year from the equilibrium climate sensitivity.
    ///
    /// MAGICC7 does this, instead of using each agent's `internal_efficacy`.
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
/// `Equilibrium Climate Sensitivity` and `2xCO2 Effective Radiative Forcing`.
#[rscm_core::params]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct EfficacyClimate {
    /// Equilibrium climate sensitivity used before the climate component reports one
    ///
    /// unit: K
    pub ecs: FloatValue,
    /// 2xCO2 effective radiative forcing used before it is first calculated
    ///
    /// unit: W/m^2
    pub erf_2xco2: FloatValue,
    /// Land-ocean heat exchange coefficient (MAGICC7 `CORE_HEATXCHANGE_LANDOCEAN`)
    ///
    /// unit: W/m^2/K
    pub land_ocean_heat_exchange: FloatValue,
    /// Inter-hemispheric heat exchange coefficient (MAGICC7 `CORE_HEATXCHANGE_NORTHSOUTH`)
    ///
    /// unit: W/m^2/K
    pub north_south_heat_exchange: FloatValue,
    /// Land-ocean warming ratio (MAGICC7 `CORE_RLO`)
    ///
    /// unit: dimensionless
    pub land_ocean_warming_ratio: FloatValue,
    /// Ocean-to-land heat exchange amplification
    ///
    /// unit: dimensionless
    pub amplify_ocean_to_land: FloatValue,
    /// Regional CO2 ERF pattern in each region of the four-box grid (MAGICC7 `RF_REGIONS_CO2`)
    ///
    /// unit: dimensionless
    pub co2_regional_pattern: [FloatValue; 4],
    /// Reproduce MAGICC7 v7.5.3 reading LAMCALC's first error, `DIFF(1)`, unset
    pub include_unset_lamcalc_error: bool,
}

impl Default for EfficacyClimate {
    fn default() -> Self {
        let p = crate::parameters::ClimateUDEBParameters::default();
        Self {
            ecs: p.ecs,
            erf_2xco2: p.erf_2xco2,
            land_ocean_heat_exchange: p.land_ocean_heat_exchange,
            north_south_heat_exchange: p.north_south_heat_exchange,
            land_ocean_warming_ratio: p.land_ocean_warming_ratio,
            amplify_ocean_to_land: p.amplify_ocean_to_land,
            co2_regional_pattern: p.co2_regional_pattern,
            include_unset_lamcalc_error: p.include_unset_lamcalc_error,
        }
    }
}

impl Default for ForcingAggregationParameters {
    fn default() -> Self {
        Self {
            agents: Vec::new(),
            summation: Vec::new(),
            efficacy_method: EfficacyMethod::AR6,
            nh_land_fraction: 0.42,
            sh_land_fraction: 0.21,
            climate: None,
        }
    }
}

impl ForcingAggregationParameters {
    /// Four-box grid weighted by the global area fraction of each box.
    pub fn grid(&self) -> FourBoxGrid {
        FourBoxGrid::with_weights([
            (1.0 - self.nh_land_fraction) / 2.0,
            self.nh_land_fraction / 2.0,
            (1.0 - self.sh_land_fraction) / 2.0,
            self.sh_land_fraction / 2.0,
        ])
    }
}

fn uniform() -> [FloatValue; 4] {
    [1.0; 4]
}

fn one() -> FloatValue {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_keys() {
        let err = serde_json::from_str::<ForcingAggregationParameters>(r#"{"efficacy_apply": 0}"#)
            .unwrap_err();
        assert!(
            err.to_string().contains("unknown field `efficacy_apply`"),
            "{err}"
        );
    }
}
