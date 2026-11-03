//! Ozone forcing parameters

use crate::climate::lamcalc::hemispheric_qfrac;
use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Parameters for `OzoneForcing`, named after their MAGICC7 counterparts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OzoneForcingParameters {
    /// `STRATOZ_O3SCALE` (W/m²): negative because depletion cools.
    pub strat_o3_scale: FloatValue,

    /// `STRATOZ_CLEXPON`: exponent of the EESC power law.
    pub strat_cl_exponent: FloatValue,

    /// `STRATOZ_THRESHOLD_YEAR`: stratospheric forcing is zero before this year
    /// and relative to its EESC afterwards.
    pub strat_threshold_year: FloatValue,

    /// `TROPOZ_RADEFF_WM2PERDU` (W/m² per DU).
    pub trop_radeff: FloatValue,

    /// `TROPOZ_OZCH4` (DU per unit ln(CH4/CH4₀)).
    pub trop_oz_ch4: FloatValue,

    /// `TROPOZ_OZNOX` (DU per Mt N/yr).
    pub trop_oz_nox: FloatValue,

    /// `TROPOZ_OZCO` (DU per Mt CO/yr).
    pub trop_oz_co: FloatValue,

    /// `TROPOZ_OZVOC` (DU per Mt NMVOC/yr).
    pub trop_oz_voc: FloatValue,

    /// `OZDUETON2O_RADEFF` (W/m² per ppb N2O).
    pub n2o_radeff: FloatValue,

    /// `OZDUETOTEMPERATURE_SCALE` (W/m² per K).
    pub temperature_scale: FloatValue,

    /// Regional pattern of tropospheric ozone forcing, which makes
    /// `Effective Radiative Forcing|Tropospheric Ozone` a four-box output.
    pub trop_regions: Option<TropOzoneRegions>,
}

/// Regional pattern of tropospheric ozone forcing, normalised within each hemisphere.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TropOzoneRegions {
    /// Pattern (NH ocean, NH land, SH ocean, SH land).
    /// MAGICC7: RF_REGIONS_TROPOZ
    pub rf_regions: [FloatValue; 4],
    /// Northern Hemisphere land fraction.
    pub nh_land_fraction: FloatValue,
    /// Southern Hemisphere land fraction.
    pub sh_land_fraction: FloatValue,
}

impl Default for TropOzoneRegions {
    fn default() -> Self {
        Self {
            rf_regions: [0.46565, 0.51646, 0.17687, 0.23793],
            nh_land_fraction: 0.42,
            sh_land_fraction: 0.21,
        }
    }
}

impl TropOzoneRegions {
    /// Box forcing per unit global forcing.
    pub fn pattern(&self) -> [FloatValue; 4] {
        let area = [
            (1.0 - self.nh_land_fraction) / 2.0,
            self.nh_land_fraction / 2.0,
            (1.0 - self.sh_land_fraction) / 2.0,
            self.sh_land_fraction / 2.0,
        ];
        hemispheric_qfrac(&self.rf_regions, &area)
    }
}

impl Default for OzoneForcingParameters {
    fn default() -> Self {
        Self {
            strat_o3_scale: -0.0043,
            strat_cl_exponent: 1.7,
            strat_threshold_year: 1979.0,
            trop_radeff: 0.032,
            trop_oz_ch4: 5.7,
            trop_oz_nox: 0.168,
            trop_oz_co: 0.00396,
            trop_oz_voc: 0.01008,
            n2o_radeff: 0.0004827,
            temperature_scale: -0.037,
            trop_regions: None,
        }
    }
}
