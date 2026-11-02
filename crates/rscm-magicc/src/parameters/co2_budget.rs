//! CO2 Budget parameters
//!
//! Parameters for the CO2 mass balance calculation that closes the carbon cycle.

use serde::{Deserialize, Serialize};

/// MAGICC7 conversion from atmospheric carbon mass to CO2 concentration (GtC/ppm).
pub const GTC_PER_PPM: f64 = 2.123;

/// Parameters for CO2 budget calculation.
///
/// The CO2 budget integrates emissions and uptakes to calculate the change
/// in atmospheric CO2 concentration:
///
/// $$\Delta C_{atm} = \frac{E_{fossil} + E_{CH_4\,ox} - F_{land} - F_{ocean}}{GtC\_per\_ppm}$$
///
/// where fluxes are positive when removing CO2 from the atmosphere and the
/// land flux includes land-use emissions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CO2BudgetParameters {
    /// Conversion factor: GtC per ppm CO2.
    ///
    /// This relates changes in atmospheric CO2 mass (GtC) to changes in
    /// concentration (ppm). The default value of 2.123 GtC/ppm is standard
    /// in carbon cycle modelling.
    ///
    /// Default: 2.123 GtC/ppm
    pub gtc_per_ppm: f64,

    /// Pre-industrial CO2 concentration (ppm).
    ///
    /// Used for calculating airborne fraction relative to pre-industrial.
    ///
    /// Default: 278.0 ppm
    pub co2_pi: f64,
}

impl Default for CO2BudgetParameters {
    fn default() -> Self {
        Self {
            gtc_per_ppm: GTC_PER_PPM,
            co2_pi: 278.0,
        }
    }
}
