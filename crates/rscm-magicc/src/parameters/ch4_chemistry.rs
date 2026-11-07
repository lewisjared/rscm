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
pub struct CH4ChemistryParameters {
    /// Year from which the concentration and temperature feedbacks are measured
    /// (`CH4_FEED_YRSTART`). The reference concentration and temperature are
    /// taken from the year before. NOx, CO and NMVOC deltas are relative to
    /// emissions in this year.
    /// unit: year
    /// default: 1927.0
    pub feedback_start_year: FloatValue,

    /// Natural CH4 emissions (wetlands, termites, oceans, etc.)
    /// MAGICC7 closes the 1995-2004 budget against observed concentrations.
    /// The default is the value it derives for its standard historical inputs.
    /// unit: Mt CH4/yr
    /// default: 182.1908
    pub natural_emissions: FloatValue,

    /// Add `Emissions|CH4|Permafrost` from the `Permafrost` component to the emissions.
    /// MAGICC7: PF_APPLY
    /// default: false
    pub include_permafrost: bool,

    /// Last year of concentrations prescribed by `Atmospheric Concentration|CH4|Prescribed`,
    /// or `None` to calculate them from emissions throughout.
    /// MAGICC7: CH4_SWITCHFROMCONC2EMIS_YEAR
    pub concentration_switch_year: Option<FloatValue>,

    /// Total atmospheric lifetime at the reference state (`CH4_TAUTOT_INIT`).
    /// The OH lifetime is derived by removing the non-OH sinks.
    /// unit: years
    /// default: 9.9474
    pub initial_ch4_lifetime: FloatValue,

    /// Soil uptake sink lifetime
    /// unit: years
    /// default: 150.0
    /// MAGICC7: CH4_TAUSOIL
    pub soil_lifetime: FloatValue,

    /// Stratospheric sink lifetime
    /// unit: years
    /// default: 120.0
    /// MAGICC7: CH4_TAUSTRAT
    pub stratospheric_lifetime: FloatValue,

    /// Tropospheric Cl sink lifetime
    /// unit: years
    /// default: 200.0
    /// MAGICC7: CH4_TAUTROPCL
    pub tropospheric_cl_lifetime: FloatValue,

    /// CH4 self-feedback coefficient (S in TAR Table 4.11)
    /// Represents how CH4 concentration affects OH and hence its own lifetime
    /// Negative value: higher CH4 → lower OH → longer lifetime (positive feedback)
    /// unit: dimensionless
    /// default: -0.53775
    pub ch4_self_feedback: FloatValue,

    /// OH sensitivity scaling factor (γ in the equations)
    /// Scales the OH-lifetime feedback response
    /// unit: dimensionless
    /// default: 0.72448
    pub oh_sensitivity_scale: FloatValue,

    /// NOx emissions effect on OH (A_NOx)
    /// Positive: more NOx → more OH → shorter CH4 lifetime
    /// unit: (Mt N/yr)^-1
    /// default: 0.0093376
    pub oh_nox_sensitivity: FloatValue,

    /// CO emissions effect on OH (A_CO)
    /// Negative: more CO → less OH → longer CH4 lifetime
    /// unit: (Mt CO/yr)^-1
    /// default: -0.000113
    pub oh_co_sensitivity: FloatValue,

    /// NMVOC emissions effect on OH (A_VOC)
    /// Negative: more VOC → less OH → longer CH4 lifetime
    /// unit: (Mt NMVOC/yr)^-1
    /// default: -0.0003142
    pub oh_nmvoc_sensitivity: FloatValue,

    /// Temperature sensitivity of lifetime
    /// Positive: warmer → shorter lifetime (more OH activity)
    /// unit: K^-1
    /// default: 0.07
    /// MAGICC7: CH4_TAUTEMPSENSITIVITY
    pub temperature_sensitivity: FloatValue,

    /// Enable temperature feedback on lifetime
    /// default: true
    /// MAGICC7: CH4_INCLUDE_TEMPFEEDBACK
    pub include_temperature_feedback: bool,

    /// Enable NOx/CO/NMVOC emissions feedback on OH
    /// default: true
    pub include_emissions_feedback: bool,

    /// Conversion factor from ppb to Tg CH4
    /// unit: Tg CH4/ppb
    /// default: 2.824
    /// MAGICC7: CH4_PPB2TGCH4
    pub tg_per_ppb: FloatValue,

    /// Fraction of the atmosphere mixed with surface CH4 (`CH4_MIXBOXSIZE`)
    /// unit: dimensionless
    /// default: 0.973
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
