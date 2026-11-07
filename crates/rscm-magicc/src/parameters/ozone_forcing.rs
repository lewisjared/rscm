//! Ozone forcing parameters

use crate::climate::lamcalc::hemispheric_qfrac;
use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Parameters for `OzoneForcing`, named after their MAGICC7 counterparts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct OzoneForcingParameters {
    /// Stratospheric ozone scale, $s$ (MAGICC7 `STRATOZ_O3SCALE`).
    ///
    /// unit: W/m^2
    ///
    /// Negative because depletion cools.
    pub stratospheric_o3_scale: FloatValue,

    /// Exponent of the EESC power law, $\beta$ (MAGICC7 `STRATOZ_CLEXPON`)
    ///
    /// unit: dimensionless
    pub stratospheric_cl_exponent: FloatValue,

    /// Threshold year $Y_0$ of the stratospheric ozone ERF (MAGICC7 `STRATOZ_THRESHOLD_YEAR`).
    ///
    /// unit: yr
    ///
    /// The ERF is zero before this year and relative to its EESC afterwards.
    pub stratospheric_reference_year: FloatValue,

    /// Radiative efficiency of tropospheric ozone, $r$ (MAGICC7 `TROPOZ_RADEFF_WM2PERDU`)
    ///
    /// unit: W/m^2 per DU
    pub tropospheric_ozone_radiative_efficiency: FloatValue,

    /// Tropospheric ozone sensitivity to CH4, $o_{CH4}$ (MAGICC7 `TROPOZ_OZCH4`)
    ///
    /// unit: DU per unit ln(CH4/CH4_0)
    pub tropospheric_ozone_ch4_sensitivity: FloatValue,

    /// Tropospheric ozone sensitivity to NOx emissions, $o_{NOx}$ (MAGICC7 `TROPOZ_OZNOX`)
    ///
    /// unit: DU per Mt N/yr
    pub tropospheric_ozone_nox_sensitivity: FloatValue,

    /// Tropospheric ozone sensitivity to CO emissions, $o_{CO}$ (MAGICC7 `TROPOZ_OZCO`)
    ///
    /// unit: DU per Mt CO/yr
    pub tropospheric_ozone_co_sensitivity: FloatValue,

    /// Tropospheric ozone sensitivity to NMVOC emissions, $o_{VOC}$ (MAGICC7 `TROPOZ_OZVOC`)
    ///
    /// unit: DU per Mt NMVOC/yr
    pub tropospheric_ozone_nmvoc_sensitivity: FloatValue,

    /// Ozone ERF per unit N2O concentration, $k$ (MAGICC7 `OZDUETON2O_RADEFF`)
    ///
    /// unit: W/m^2 per ppb
    pub n2o_ozone_radiative_efficiency: FloatValue,

    /// Ozone ERF per unit warming, $\gamma$ (MAGICC7 `OZDUETOTEMPERATURE_SCALE`)
    ///
    /// unit: W/m^2 per K
    pub temperature_scale: FloatValue,

    /// Regional pattern of tropospheric ozone ERF (MAGICC7 `RF_REGIONS_TROPOZ`).
    ///
    /// Setting it makes `Effective Radiative Forcing|Ozone|Tropospheric` a four-box output.
    pub tropospheric_ozone_regions: Option<TroposphericOzoneRegions>,
}

/// Regional pattern of tropospheric ozone ERF, normalised within each hemisphere.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct TroposphericOzoneRegions {
    /// Pattern in each region of the four-box grid (MAGICC7 `RF_REGIONS_TROPOZ`)
    ///
    /// unit: dimensionless
    pub regional_pattern: [FloatValue; 4],
    /// Northern Hemisphere land fraction
    ///
    /// unit: dimensionless
    pub nh_land_fraction: FloatValue,
    /// Southern Hemisphere land fraction
    ///
    /// unit: dimensionless
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
