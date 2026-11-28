//! Nitrate and cloud albedo forcing parameters

use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Parameters for `NitrateCloudForcing`, named after their MAGICC7 counterparts.
#[rscm_core::params]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct NitrateCloudForcingParameters {
    /// Nitrate direct forcing.
    pub nitrate: NitrateParameters,

    /// Cloud albedo forcing.
    pub cloud_albedo: CloudAlbedoParameters,

    /// Natural SOx emissions, added to the anthropogenic ones (MAGICC7 `SOX_NATURAL_EMISSIONS`).
    ///
    /// unit: Mt S/yr
    pub natural_sox_emissions: FloatValue,

    /// Northern Hemisphere land fraction
    ///
    /// unit: dimensionless
    pub nh_land_fraction: FloatValue,

    /// Southern Hemisphere land fraction
    ///
    /// unit: dimensionless
    pub sh_land_fraction: FloatValue,
}

/// Nitrate direct forcing (Hauglustaine et al. 2014).
#[rscm_core::params]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct NitrateParameters {
    /// Natural NOx emissions (MAGICC7 `RF_NOXN_EMIS`).
    ///
    /// unit: Mt N/yr
    pub natural_nox_emissions: FloatValue,

    /// Natural NH3 emissions (MAGICC7 `RF_NH3N_EMIS`).
    ///
    /// unit: Mt N/yr
    pub natural_nh3_emissions: FloatValue,

    /// Competition of SO2 for NH3 (MAGICC7 `RF_NO3_LAMBDASO2`).
    ///
    /// unit: dimensionless
    pub lambda_so2: FloatValue,

    /// Pre-industrial nitrate burden (MAGICC7 `RF_NO3_PREINDBURDEN`).
    ///
    /// unit: Tg
    pub burden_pi: FloatValue,

    /// Pre-industrial nitrate optical thickness (MAGICC7 `RF_NO3_PREINDOT`).
    ///
    /// unit: dimensionless
    pub optical_thickness_pi: FloatValue,

    /// Exponent from relative burden to relative optical thickness (MAGICC7 `RF_NO3_BURDEN2OT_EXP`).
    ///
    /// unit: dimensionless
    pub burden_to_optical_thickness_exponent: FloatValue,

    /// Pattern that spreads each hemisphere's forcing over its boxes (MAGICC7 `RF_REGIONS_NO3`).
    ///
    /// unit: dimensionless
    pub regional_pattern: [FloatValue; 4],

    /// Year whose forcing sets nitrate's pattern for its internal efficacy (MAGICC7 `RF_REGIONS_NORMYEAR`).
    ///
    /// unit: yr
    ///
    /// None leaves the `ForcingAggregation` pattern as configured.
    /// The Python model setup applies it, see `NitrateCloudForcing::nitrate_pattern`.
    pub efficacy_pattern_year: Option<FloatValue>,

    /// Factor on the forcing (MAGICC7 `RF_NO3T_DIR_FACTOR`).
    ///
    /// unit: dimensionless
    pub scale: FloatValue,

    /// Global forcing in a year that sets the factor instead of `scale` (MAGICC7 `RF_NO3T_DIR_WM2`).
    pub harmonisation: Option<ForcingTarget>,
}

/// Cloud albedo forcing from a log-linear aerosol number index.
#[rscm_core::params]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct CloudAlbedoParameters {
    /// Weight of each species in the aerosol number index (MAGICC7 `CLOUD_WEIGHT_<X>`).
    pub weights: CloudSpecies,

    /// Year from which each species follows emissions rather than its optical thickness.
    ///
    /// The optical thickness records end in this year.
    /// Nitrate follows industrial NOx emissions.
    pub switch_years: CloudSpecies,

    /// Soluble share of industrial BC optical thickness (MAGICC7 `CLOUD_BCI2BCB_SOLUBLE_RATIO`).
    ///
    /// unit: dimensionless
    pub bci_soluble_ratio: FloatValue,

    /// Box forcing per unit change in log10 of the number index (MAGICC7 `RF_REGIONS_CLOUD_ALBEDO`).
    ///
    /// unit: dimensionless
    pub regional_pattern: [FloatValue; 4],

    /// Year in which the forcing equals `regional_pattern`, before the factor.
    ///
    /// unit: yr
    pub pattern_year: FloatValue,

    /// Factor on the forcing (MAGICC7 `RF_CLOUD_ALBEDO_AER_FACTOR`).
    ///
    /// unit: dimensionless
    pub scale: FloatValue,

    /// Global forcing in a year that sets the factor instead of `scale` (MAGICC7 `RF_CLOUD_ALBEDO_AER_WM2`).
    pub harmonisation: Option<ForcingTarget>,
}

/// A value for each species of the aerosol number index.
#[rscm_core::params]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloudSpecies {
    /// Nitrate
    pub nitrate: FloatValue,
    /// Black carbon
    pub bc: FloatValue,
    /// Organic carbon
    pub oc: FloatValue,
    /// Sulfate
    pub sox: FloatValue,
    /// Sea salt
    pub sea_salt: FloatValue,
}

/// Global forcing that a forcing is scaled to in a year.
#[rscm_core::params]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForcingTarget {
    /// Global forcing in `reference_year`
    ///
    /// unit: W/m^2
    pub target: FloatValue,

    /// Year the forcing equals `target`
    ///
    /// unit: yr
    pub reference_year: FloatValue,
}

impl Default for NitrateCloudForcingParameters {
    fn default() -> Self {
        Self {
            nitrate: NitrateParameters::default(),
            cloud_albedo: CloudAlbedoParameters::default(),
            natural_sox_emissions: 14.0,
            nh_land_fraction: 0.42,
            sh_land_fraction: 0.21,
        }
    }
}

impl Default for NitrateParameters {
    fn default() -> Self {
        Self {
            natural_nox_emissions: 10.0,
            natural_nh3_emissions: 21.0,
            lambda_so2: 1.6,
            burden_pi: 0.0085,
            optical_thickness_pi: 1.41,
            burden_to_optical_thickness_exponent: 0.76,
            regional_pattern: [-1.53572293, -1.35919761, -0.482044172, -0.553476135],
            efficacy_pattern_year: None,
            scale: 1.0,
            harmonisation: Some(ForcingTarget {
                target: -0.0444,
                reference_year: 2019.0,
            }),
        }
    }
}

impl Default for CloudAlbedoParameters {
    fn default() -> Self {
        Self {
            weights: CloudSpecies {
                nitrate: 0.163265306122449,
                bc: 0.04081632653061225,
                oc: 0.26530612244898,
                sox: 0.26530612244898,
                sea_salt: 0.26530612244898,
            },
            switch_years: CloudSpecies {
                nitrate: 2014.0,
                bc: 2015.0,
                oc: 2015.0,
                sox: 2015.0,
                sea_salt: 2015.0,
            },
            bci_soluble_ratio: 0.75,
            regional_pattern: [-0.96563, -1.39945, -0.34211, -0.62846],
            pattern_year: 2000.0,
            scale: 1.0,
            harmonisation: Some(ForcingTarget {
                target: -0.89,
                reference_year: 2019.0,
            }),
        }
    }
}
