//! Ozone forcing parameters
//!
//! Parameters for stratospheric and tropospheric ozone radiative forcing
//! calculations.

use serde::{Deserialize, Serialize};

/// Parameters for ozone radiative forcing
///
/// # Stratospheric Ozone
///
/// Stratospheric ozone forcing is derived from EESC (Equivalent Effective
/// Stratospheric Chlorine) using a power-law relationship:
///
/// $$RF_{strat} = \alpha \cdot \max(0, EESC - EESC_{ref})^{\beta}$$
///
/// This produces negative (cooling) forcing because ozone-depleting substances
/// destroy stratospheric ozone.
///
/// # Tropospheric Ozone
///
/// Tropospheric ozone forcing has two components:
///
/// 1. CH4 contribution (logarithmic in concentration)
/// 2. Precursor contributions (linear in emissions: NOx, CO, NMVOC)
///
/// $$RF_{trop} = \eta \cdot OZCH4 \cdot \ln(CH4/CH4_{pi}) + \eta \cdot (\alpha_{NOx} \cdot \Delta NOx + ...)$$
///
/// # Temperature Feedback
///
/// Warming affects ozone photochemistry and transport, providing a
/// (typically negative) feedback:
///
/// $$RF_{temp} = \gamma \cdot \Delta T$$
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OzoneForcingParameters {
    // === Stratospheric Ozone Parameters ===
    /// EESC reference value (ppt) at threshold year (1979).
    ///
    /// Stratospheric ozone forcing is relative to this baseline.
    /// Default: 1420.0 ppt (approximate EESC at 1979)
    pub eesc_reference: f64,

    /// Scaling factor for stratospheric ozone forcing (W/m² per (ppt/100)^exponent).
    ///
    /// Note: This is NEGATIVE because ozone depletion causes cooling.
    /// Default: -0.0043 W/m²
    pub strat_o3_scale: f64,

    /// Power-law exponent for EESC-RF relationship.
    ///
    /// Default: 1.7 (from MAGICC7 default)
    pub strat_cl_exponent: f64,

    // === Tropospheric Ozone Parameters ===
    /// Radiative efficiency (W/m² per DU).
    ///
    /// Default: 0.032 W/m²/DU
    pub trop_radeff: f64,

    /// Ozone change per ln(CH4/CH4_pi) in DU.
    ///
    /// Default: 5.7 DU
    pub trop_oz_ch4: f64,

    /// NOx sensitivity (DU per Mt N/yr).
    ///
    /// Default: 0.168 DU/(Mt N/yr)
    pub trop_oz_nox: f64,

    /// CO sensitivity (DU per Mt CO/yr).
    ///
    /// Default: 0.00396 DU/(Mt CO/yr)
    pub trop_oz_co: f64,

    /// NMVOC sensitivity (DU per Mt NMVOC/yr).
    ///
    /// Default: 0.01008 DU/(Mt NMVOC/yr)
    pub trop_oz_voc: f64,

    /// Pre-industrial CH4 concentration (ppb).
    ///
    /// Default: 700.0 ppb
    pub ch4_pi: f64,

    /// Pre-industrial NOx emissions (Mt N/yr).
    ///
    /// Default: 0.0 Mt N/yr
    pub nox_pi: f64,

    /// Pre-industrial CO emissions (Mt CO/yr).
    ///
    /// Default: 0.0 Mt CO/yr
    pub co_pi: f64,

    /// Pre-industrial NMVOC emissions (Mt NMVOC/yr).
    ///
    /// Default: 0.0 Mt NMVOC/yr
    pub nmvoc_pi: f64,

    // === Temperature Feedback Parameters ===
    /// Temperature feedback coefficient (W/m² per K).
    ///
    /// Note: This is NEGATIVE because warming destroys ozone (negative feedback).
    /// Default: -0.037 W/m²/K
    pub temp_feedback_scale: f64,
}

impl Default for OzoneForcingParameters {
    fn default() -> Self {
        Self {
            // Stratospheric ozone
            eesc_reference: 1420.0,  // ppt at 1979
            strat_o3_scale: -0.0043, // W/m² (negative for cooling)
            strat_cl_exponent: 1.7,  // power-law exponent

            // Tropospheric ozone
            trop_radeff: 0.032,   // W/m² per DU
            trop_oz_ch4: 5.7,     // DU per ln ratio
            trop_oz_nox: 0.168,   // DU per Mt N/yr
            trop_oz_co: 0.00396,  // DU per Mt CO/yr
            trop_oz_voc: 0.01008, // DU per Mt NMVOC/yr
            ch4_pi: 700.0,        // ppb
            nox_pi: 0.0,          // Mt N/yr
            co_pi: 0.0,           // Mt CO/yr
            nmvoc_pi: 0.0,        // Mt NMVOC/yr

            // Temperature feedback
            temp_feedback_scale: -0.037, // W/m² per K (negative for feedback)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() {
        let params = OzoneForcingParameters::default();

        // Stratospheric parameters should be set
        assert!(params.eesc_reference > 0.0);
        assert!(params.strat_o3_scale < 0.0); // Negative for cooling
        assert!(params.strat_cl_exponent > 0.0);

        // Tropospheric parameters should be set
        assert!(params.trop_radeff > 0.0);
        assert!(params.trop_oz_ch4 > 0.0);
        assert!(params.ch4_pi > 0.0);

        // Temperature feedback should be negative
        assert!(params.temp_feedback_scale < 0.0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let params = OzoneForcingParameters::default();
        let json = serde_json::to_string(&params).unwrap();
        let restored: OzoneForcingParameters = serde_json::from_str(&json).unwrap();

        assert!((params.eesc_reference - restored.eesc_reference).abs() < 1e-10);
        assert!((params.strat_o3_scale - restored.strat_o3_scale).abs() < 1e-10);
        assert!((params.trop_radeff - restored.trop_radeff).abs() < 1e-10);
    }
}
