//! CH4 Chemistry Parameters
//!
//! Parameters for atmospheric methane chemistry calculations, implementing
//! the Prather iteration method with OH feedback.
//!
//! # Reference
//!
//! Based on MAGICC7's CH4 chemistry, which implements Prather's
//! method from IPCC TAR Table 4.11 for solving the nonlinear methane mass
//! balance equation.

use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Parameters for CH4 chemistry calculations
///
/// Controls the methane atmospheric lifetime calculation including:
/// - OH sink with concentration-dependent feedback (Prather method)
/// - Non-OH sinks (soil, stratosphere, tropospheric Cl)
/// - Temperature feedback on OH oxidation rates
/// - Emissions feedback from co-emitted species (NOx, CO, NMVOC)
///
/// # Lifetime Calculation
///
/// The effective CH4 lifetime is calculated as:
///
/// $$\frac{1}{\tau_{total}} = \frac{1}{\tau_{OH}} + \frac{1}{\tau_{soil}} + \frac{1}{\tau_{strat}} + \frac{1}{\tau_{trop\_cl}}$$
///
/// where ${\tau_{OH}}$ varies with CH4 concentration and co-emitted species.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct CH4ChemistryParameters {
    /// Year from which the concentration and temperature feedbacks are measured (MAGICC7 `CH4_FEED_YRSTART`).
    ///
    /// unit: yr
    ///
    /// The reference concentration and temperature are taken from the year before.
    /// NOx, CO and NMVOC feedbacks are relative to emissions in this year.
    pub feedback_start_year: FloatValue,

    /// Natural CH4 emissions from wetlands, termites, oceans and other sources.
    ///
    /// unit: Mt CH4/yr
    ///
    /// MAGICC7 closes the 1995-2004 budget against observed concentrations.
    /// The default is the value it derives for its standard historical inputs.
    pub natural_emissions: FloatValue,

    /// Add `Emissions|CH4|Permafrost` from the `Permafrost` component to the emissions (MAGICC7 `PF_APPLY`)
    pub include_permafrost: bool,

    /// Last year of concentrations prescribed by `Prescribed Atmospheric Concentrations|CH4`
    /// (MAGICC7 `CH4_SWITCHFROMCONC2EMIS_YEAR`).
    ///
    /// unit: yr
    ///
    /// `None` calculates concentrations from emissions throughout.
    pub concentration_switch_year: Option<FloatValue>,

    /// Total atmospheric lifetime at the reference state (MAGICC7 `CH4_TAUTOT_INIT`).
    ///
    /// unit: yr
    ///
    /// The OH lifetime is derived by removing the non-OH sinks.
    pub initial_ch4_lifetime: FloatValue,

    /// Soil uptake sink lifetime (MAGICC7 `CH4_TAUSOIL`)
    ///
    /// unit: yr
    pub soil_lifetime: FloatValue,

    /// Stratospheric sink lifetime (MAGICC7 `CH4_TAUSTRAT`)
    ///
    /// unit: yr
    pub stratospheric_lifetime: FloatValue,

    /// Tropospheric Cl sink lifetime (MAGICC7 `CH4_TAUTROPCL`)
    ///
    /// unit: yr
    pub tropospheric_cl_lifetime: FloatValue,

    /// CH4 self-feedback coefficient, S in TAR Table 4.11.
    ///
    /// unit: dimensionless
    ///
    /// How CH4 concentration affects OH and so its own lifetime.
    /// Negative because higher CH4 lowers OH and lengthens the lifetime (a positive feedback).
    pub ch4_self_feedback: FloatValue,

    /// OH sensitivity scaling factor, $\gamma$ in the equations.
    ///
    /// unit: dimensionless
    ///
    /// Scales the response of the OH lifetime to the feedbacks.
    pub oh_sensitivity_scale: FloatValue,

    /// Effect of NOx emissions on OH, $A_{NOx}$.
    ///
    /// unit: (Mt N/yr)^-1
    ///
    /// Positive because more NOx gives more OH and a shorter CH4 lifetime.
    pub oh_nox_sensitivity: FloatValue,

    /// Effect of CO emissions on OH, $A_{CO}$.
    ///
    /// unit: (Mt CO/yr)^-1
    ///
    /// Negative because more CO gives less OH and a longer CH4 lifetime.
    pub oh_co_sensitivity: FloatValue,

    /// Effect of NMVOC emissions on OH, $A_{VOC}$.
    ///
    /// unit: (Mt NMVOC/yr)^-1
    ///
    /// Negative because more NMVOC gives less OH and a longer CH4 lifetime.
    pub oh_nmvoc_sensitivity: FloatValue,

    /// Temperature sensitivity of the lifetime (MAGICC7 `CH4_TAUTEMPSENSITIVITY`).
    ///
    /// unit: K^-1
    ///
    /// Positive because warming increases OH activity and shortens the lifetime.
    pub temperature_sensitivity: FloatValue,

    /// Apply the temperature feedback on the lifetime (MAGICC7 `CH4_INCLUDE_TEMPFEEDBACK`)
    pub include_temperature_feedback: bool,

    /// Apply the NOx, CO and NMVOC emissions feedback on OH
    pub include_emissions_feedback: bool,

    /// Conversion factor from ppb to Tg CH4 (MAGICC7 `CH4_PPB2TGCH4`)
    ///
    /// unit: Tg CH4/ppb
    pub tg_per_ppb: FloatValue,

    /// Fraction of the atmosphere mixed with surface CH4 (MAGICC7 `CH4_MIXBOXSIZE`)
    ///
    /// unit: dimensionless
    pub mixing_box_size: FloatValue,
}

impl Default for CH4ChemistryParameters {
    fn default() -> Self {
        Self {
            feedback_start_year: 1927.0,
            natural_emissions: 182.1908,
            include_permafrost: false,
            concentration_switch_year: None,
            initial_ch4_lifetime: 9.9474,
            soil_lifetime: 150.0,
            stratospheric_lifetime: 120.0,
            tropospheric_cl_lifetime: 200.0,
            ch4_self_feedback: -0.53775,
            oh_sensitivity_scale: 0.72448,
            oh_nox_sensitivity: 0.0093376,
            oh_co_sensitivity: -0.000113,
            oh_nmvoc_sensitivity: -0.0003142,
            temperature_sensitivity: 0.07,
            include_temperature_feedback: true,
            include_emissions_feedback: true,
            tg_per_ppb: 2.824,
            mixing_box_size: 0.973,
        }
    }
}

impl CH4ChemistryParameters {
    /// Calculate the combined non-OH lifetime
    ///
    /// $$\frac{1}{\tau_{other}} = \frac{1}{\tau_{soil}} + \frac{1}{\tau_{strat}} + \frac{1}{\tau_{trop\_cl}}$$
    pub fn non_oh_lifetime(&self) -> FloatValue {
        1.0 / (1.0 / self.soil_lifetime
            + 1.0 / self.stratospheric_lifetime
            + 1.0 / self.tropospheric_cl_lifetime)
    }

    /// Initial OH lifetime, $1 / (1/\tau_{total,init} - 1/\tau_{other})$
    pub fn initial_oh_lifetime(&self) -> FloatValue {
        1.0 / (1.0 / self.initial_ch4_lifetime - 1.0 / self.non_oh_lifetime())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() {
        let params = CH4ChemistryParameters::default();
        assert!((params.feedback_start_year - 1927.0).abs() < 1e-10);
        assert!((params.initial_ch4_lifetime - 9.9474).abs() < 1e-10);
        assert!(params.include_temperature_feedback);
    }

    #[test]
    fn test_non_oh_lifetime_calculation() {
        let params = CH4ChemistryParameters::default();
        let non_oh_lifetime = params.non_oh_lifetime();

        // Expected: 1/(1/150 + 1/120 + 1/200) = 50
        let expected = 1.0 / (1.0 / 150.0 + 1.0 / 120.0 + 1.0 / 200.0);
        assert!(
            (non_oh_lifetime - expected).abs() < 1e-10,
            "non_oh_lifetime should be {:.2}, got {:.2}",
            expected,
            non_oh_lifetime
        );
    }

    #[test]
    fn test_initial_oh_lifetime_matches_magicc() {
        // MAGICC7 reports 12.4179 years for the default configuration
        let params = CH4ChemistryParameters::default();
        assert!((params.initial_oh_lifetime() - 12.41792).abs() < 1e-4);
    }

    #[test]
    fn test_ch4_self_feedback_sign() {
        let params = CH4ChemistryParameters::default();
        // Self-feedback should be negative (positive feedback loop)
        assert!(
            params.ch4_self_feedback < 0.0,
            "CH4 self-feedback should be negative"
        );
    }

    #[test]
    fn test_emissions_feedback_signs() {
        let params = CH4ChemistryParameters::default();
        // NOx increases OH (positive coefficient)
        assert!(params.oh_nox_sensitivity > 0.0);
        // CO and NMVOC decrease OH (negative coefficients)
        assert!(params.oh_co_sensitivity < 0.0);
        assert!(params.oh_nmvoc_sensitivity < 0.0);
    }

    #[test]
    fn test_partial_deserialisation() {
        // Test that #[serde(default)] allows partial deserialisation
        let json = r#"{"initial_ch4_lifetime": 10.5, "include_temperature_feedback": false}"#;
        let params: CH4ChemistryParameters =
            serde_json::from_str(json).expect("Partial deserialisation failed");

        // Specified fields from JSON
        assert!((params.initial_ch4_lifetime - 10.5).abs() < 1e-10);
        assert!(!params.include_temperature_feedback);

        // Other fields should be defaults
        assert!((params.feedback_start_year - 1927.0).abs() < 1e-10);
        assert!((params.soil_lifetime - 150.0).abs() < 1e-10);
    }
}
