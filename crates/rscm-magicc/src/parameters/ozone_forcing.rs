//! Ozone forcing parameters

use crate::climate::lamcalc::hemispheric_qfrac;
use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Parameters for `OzoneForcing`, named after their MAGICC7 counterparts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OzoneForcingParameters {
    /// `STRATOZ_O3SCALE` (W/m²): negative because depletion cools.
    pub stratospheric_o3_scale: FloatValue,

    /// `STRATOZ_CLEXPON`: exponent of the EESC power law.
    pub stratospheric_cl_exponent: FloatValue,

    /// `STRATOZ_THRESHOLD_YEAR`: stratospheric forcing is zero before this year
    /// and relative to its EESC afterwards.
    pub stratospheric_reference_year: FloatValue,

    /// `TROPOZ_RADEFF_WM2PERDU` (W/m² per DU).
    pub tropospheric_ozone_radiative_efficiency: FloatValue,

    /// `TROPOZ_OZCH4` (DU per unit ln(CH4/CH4₀)).
    pub tropospheric_ozone_ch4_sensitivity: FloatValue,

    /// `TROPOZ_OZNOX` (DU per Mt N/yr).
    pub tropospheric_ozone_nox_sensitivity: FloatValue,

    /// `TROPOZ_OZCO` (DU per Mt CO/yr).
    pub tropospheric_ozone_co_sensitivity: FloatValue,

    /// `TROPOZ_OZVOC` (DU per Mt NMVOC/yr).
    pub tropospheric_ozone_nmvoc_sensitivity: FloatValue,

    /// `OZDUETON2O_RADEFF` (W/m² per ppb N2O).
    pub n2o_ozone_radiative_efficiency: FloatValue,

    /// `OZDUETOTEMPERATURE_SCALE` (W/m² per K).
    pub temperature_scale: FloatValue,

    /// Regional pattern of tropospheric ozone forcing, which makes
    /// `Effective Radiative Forcing|Tropospheric Ozone` a four-box output.
    /// MAGICC7: RF_REGIONS_TROPOZ
    pub tropospheric_ozone_regions: Option<TroposphericOzoneRegions>,
}

/// Regional pattern of tropospheric ozone forcing, normalised within each hemisphere.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TroposphericOzoneRegions {
    /// Pattern (NH ocean, NH land, SH ocean, SH land).
    /// MAGICC7: RF_REGIONS_TROPOZ
    pub regional_pattern: [FloatValue; 4],
    /// Northern Hemisphere land fraction.
    pub nh_land_fraction: FloatValue,
    /// Southern Hemisphere land fraction.
    pub sh_land_fraction: FloatValue,
}

impl Default for TroposphericOzoneRegions {
    fn default() -> Self {
        Self {
            regional_pattern: [0.46565, 0.51646, 0.17687, 0.23793],
            nh_land_fraction: 0.42,
            sh_land_fraction: 0.21,
        }
    }
}

impl TroposphericOzoneRegions {
    /// Box forcing per unit global forcing.
    pub fn pattern(&self) -> [FloatValue; 4] {
        let area = [
            (1.0 - self.nh_land_fraction) / 2.0,
            self.nh_land_fraction / 2.0,
            (1.0 - self.sh_land_fraction) / 2.0,
            self.sh_land_fraction / 2.0,
        ];
        hemispheric_qfrac(&self.regional_pattern, &area)
    }
}

impl Default for OzoneForcingParameters {
    fn default() -> Self {
        Self {
            stratospheric_o3_scale: -0.0043,
            stratospheric_cl_exponent: 1.7,
            stratospheric_reference_year: 1979.0,
            tropospheric_ozone_radiative_efficiency: 0.032,
            tropospheric_ozone_ch4_sensitivity: 5.7,
            tropospheric_ozone_nox_sensitivity: 0.168,
            tropospheric_ozone_co_sensitivity: 0.00396,
            tropospheric_ozone_nmvoc_sensitivity: 0.01008,
            n2o_ozone_radiative_efficiency: 0.0004827,
            temperature_scale: -0.037,
            tropospheric_ozone_regions: None,
        }
    }
}
