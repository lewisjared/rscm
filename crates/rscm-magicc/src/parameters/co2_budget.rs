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
/// land flux includes Biosphere CO2 emissions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct CO2BudgetParameters {
    /// Conversion factor from atmospheric carbon mass to CO2 concentration
    ///
    /// unit: GtC/ppm
    ///
    /// Default: 2.123 GtC/ppm
    pub gtc_per_ppm: f64,

    /// Pre-industrial CO2 concentration (MAGICC7 `CO2_PREINDCO2CONC`).
    ///
    /// unit: ppm
    ///
    /// The component does not use it yet.
    ///
    /// Default: 278.0 ppm
    /// MAGICC7: CO2_PREINDCO2CONC
    pub co2_pi: f64,

    /// Add `Emissions|CO2|Permafrost` from the `Permafrost` component to the sources (MAGICC7 `PF_APPLY`)
    ///
    /// Default: false
    pub include_permafrost: bool,

    /// Last year of concentrations prescribed by `Prescribed Atmospheric Concentrations|CO2`
    /// (MAGICC7 `CO2_SWITCHFROMCONC2EMIS_YEAR`).
    ///
    /// unit: yr
    ///
    /// `None` calculates concentrations from emissions throughout.
    pub concentration_switch_year: Option<f64>,

    /// Take next year's concentration from `Ocean Carbon Atmospheric Concentrations|CO2`.
    ///
    /// `OceanCarbon` integrates it over its substeps, as MAGICC7 does.
    ///
    /// Default: false
    pub ocean_end_of_year: bool,
}

impl Default for CO2BudgetParameters {
    fn default() -> Self {
        Self {
            gtc_per_ppm: GTC_PER_PPM,
            co2_pi: 278.0,
            include_permafrost: false,
            concentration_switch_year: None,
            ocean_end_of_year: false,
        }
    }
}
