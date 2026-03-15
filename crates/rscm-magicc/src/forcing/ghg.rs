//! GHG Radiative Forcing Component
//!
//! Calculates effective radiative forcing (ERF) from well-mixed greenhouse
//! gases (CO2, CH4, N2O) following MAGICC7 forcing methods.
//!
//! # What This Component Does
//!
//! 1. Calculates instantaneous radiative forcing from CO2, CH4, and N2O
//!    concentrations using either IPCCTAR or Etminan methods
//! 2. Accounts for CH4-N2O absorption band overlap
//! 3. Applies rapid adjustment factors to convert to effective radiative forcing
//!
//! # Inputs
//!
//! - `Atmospheric Concentration|CO2` (ppm) - CO2 concentration
//! - `Atmospheric Concentration|CH4` (ppb) - CH4 concentration
//! - `Atmospheric Concentration|N2O` (ppb) - N2O concentration
//!
//! # Outputs
//!
//! - `Effective Radiative Forcing|CO2` (W/m^2) - CO2 ERF
//! - `Effective Radiative Forcing|CH4` (W/m^2) - CH4 ERF
//! - `Effective Radiative Forcing|N2O` (W/m^2) - N2O ERF
//!
//! # Methods
//!
//! ## IPCCTAR (Myhre et al. 1998)
//!
//! Simple analytical formulae: logarithmic for CO2, square-root for CH4/N2O,
//! with overlap correction.
//!
//! ## Etminan (Etminan et al. 2016)
//!
//! Concentration-dependent coefficients derived from line-by-line radiative
//! transfer calculations. Accounts for shortwave absorption in CO2 and
//! state-dependent forcing efficiency for CH4/N2O. Used as the OLBL proxy
//! in AR6 assessments.

use crate::parameters::{ForcingMethod, GhgForcingParameters};
use rscm_core::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// GHG radiative forcing component
///
/// Calculates effective radiative forcing from CO2, CH4, and N2O
/// concentrations using configurable forcing methods.
///
/// # CO2 Forcing (IPCCTAR)
///
/// $$F_{CO2} = \frac{\Delta Q_{2 \times CO2}}{\ln 2} \cdot \ln\left(\frac{C}{C_0}\right)$$
///
/// # CO2 Forcing (Etminan)
///
/// $$(a_1 \cdot (\Delta C)^2 + a_2 \cdot |\Delta C| + a_3) \cdot \ln\left(\frac{C}{C_0}\right)$$
///
/// where $a_3 = a_{3,N2O} \cdot \bar{N} + a_{3,offset}$
///
/// # CH4-N2O Overlap (IPCCTAR)
///
/// $$f(M, N) = 0.47 \cdot \ln\left(1 + 2.01 \times 10^{-5} (MN)^{0.75}
///     + 5.31 \times 10^{-15} M (MN)^{1.52}\right)$$
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["forcing", "ghg", "co2", "ch4", "n2o", "magicc"], category = "Radiative Forcing")]
#[inputs(
    co2_concentration { name = "Atmospheric Concentration|CO2", unit = "ppm" },
    ch4_concentration { name = "Atmospheric Concentration|CH4", unit = "ppb" },
    n2o_concentration { name = "Atmospheric Concentration|N2O", unit = "ppb" },
)]
#[outputs(
    co2_erf { name = "Effective Radiative Forcing|CO2", unit = "W/m^2" },
    ch4_erf { name = "Effective Radiative Forcing|CH4", unit = "W/m^2" },
    n2o_erf { name = "Effective Radiative Forcing|N2O", unit = "W/m^2" },
)]
pub struct GhgForcing {
    parameters: GhgForcingParameters,
}

/// Result of GHG forcing calculations
#[derive(Debug, Clone, Copy)]
pub struct GhgForcingResult {
    /// CO2 effective radiative forcing (W/m2)
    pub co2_erf: FloatValue,
    /// CH4 effective radiative forcing (W/m2)
    pub ch4_erf: FloatValue,
    /// N2O effective radiative forcing (W/m2)
    pub n2o_erf: FloatValue,
}

impl GhgForcingResult {
    /// Total GHG forcing
    pub fn total(&self) -> FloatValue {
        self.co2_erf + self.ch4_erf + self.n2o_erf
    }
}

impl GhgForcing {
    /// Create a new GHG forcing component with default parameters
    pub fn new() -> Self {
        Self::from_parameters(GhgForcingParameters::default())
    }

    /// Create from parameters
    pub fn from_parameters(parameters: GhgForcingParameters) -> Self {
        Self { parameters }
    }

    /// Get the parameters
    pub fn parameters(&self) -> &GhgForcingParameters {
        &self.parameters
    }

    /// IPCCTAR CH4-N2O overlap function (Myhre et al. 1998)
    ///
    /// $$f(M, N) = 0.47 \cdot \ln\left(1 + 2.01 \times 10^{-5} (MN)^{0.75}
    ///     + 5.31 \times 10^{-15} M (MN)^{1.52}\right)$$
    ///
    /// where $M = \sqrt{CH4}$, $N = \sqrt{N2O}$ (both in ppb)
    fn overlap_f(m: FloatValue, n: FloatValue) -> FloatValue {
        let mn = m * n;
        0.47 * (1.0 + 2.01e-5 * mn.powf(0.75) + 5.31e-15 * m * mn.powf(1.52)).ln()
    }

    /// Calculate CO2 forcing (dispatches to method)
    ///
    /// The `n2o` parameter is needed for the Etminan method (N2O affects CO2
    /// forcing coefficients). For IPCCTAR it is unused.
    pub fn calculate_co2_forcing(&self, co2: FloatValue, n2o: FloatValue) -> FloatValue {
        match self.parameters.method {
            ForcingMethod::Ipcctar => self.co2_forcing_ipcctar(co2),
            ForcingMethod::Etminan => self.co2_forcing_etminan(co2, n2o),
        }
    }

    /// Calculate CH4 forcing (dispatches to method)
    pub fn calculate_ch4_forcing(&self, ch4: FloatValue, n2o: FloatValue) -> FloatValue {
        match self.parameters.method {
            ForcingMethod::Ipcctar => self.ch4_forcing_ipcctar(ch4, n2o),
            ForcingMethod::Etminan => self.ch4_forcing_etminan(ch4, n2o),
        }
    }

    /// Calculate N2O forcing (dispatches to method)
    pub fn calculate_n2o_forcing(&self, ch4: FloatValue, n2o: FloatValue) -> FloatValue {
        match self.parameters.method {
            ForcingMethod::Ipcctar => self.n2o_forcing_ipcctar(ch4, n2o),
            ForcingMethod::Etminan => self.n2o_forcing_etminan(ch4, n2o),
        }
    }

    // === IPCCTAR implementations ===

    fn co2_forcing_ipcctar(&self, co2: FloatValue) -> FloatValue {
        let p = &self.parameters;
        let alpha = p.delq2xco2 / 2.0_f64.ln();
        alpha * (co2 / p.co2_pi).ln()
    }

    fn ch4_forcing_ipcctar(&self, ch4: FloatValue, n2o: FloatValue) -> FloatValue {
        let p = &self.parameters;
        let m = ch4.sqrt();
        let m0 = p.ch4_pi.sqrt();
        let n = n2o.sqrt();

        // Direct term
        let direct = p.ch4_radeff * (m - m0);

        // Overlap correction using current N2O (Myhre et al. 1998)
        // The overlap between CH4 and N2O absorption bands means that
        // the marginal forcing from CH4 depends on N2O concentration.
        let overlap = Self::overlap_f(m, n) - Self::overlap_f(m0, n);

        direct - overlap
    }

    fn n2o_forcing_ipcctar(&self, ch4: FloatValue, n2o: FloatValue) -> FloatValue {
        let p = &self.parameters;
        let n = n2o.sqrt();
        let n0 = p.n2o_pi.sqrt();
        let m = ch4.sqrt();

        // Direct term
        let direct = p.n2o_radeff * (n - n0);

        // Overlap correction using current CH4 (Myhre et al. 1998)
        let overlap = Self::overlap_f(m, n) - Self::overlap_f(m, n0);

        direct - overlap
    }

    // === Etminan implementations ===

    fn co2_forcing_etminan(&self, co2: FloatValue, n2o: FloatValue) -> FloatValue {
        let p = &self.parameters;
        let co2_pi = p.co2_pi;
        let n2o_bar = (n2o + p.n2o_pi) / 2.0;
        let delta_co2 = co2 - co2_pi;

        let a1 = p.etminan_co2_a1 * delta_co2 * delta_co2;
        let a2 = p.etminan_co2_a2 * delta_co2.abs();
        let a3 = p.etminan_co2_a3_n2o * n2o_bar + p.etminan_co2_a3_offset;

        (a1 + a2 + a3) * (co2 / co2_pi).ln()
    }

    fn ch4_forcing_etminan(&self, ch4: FloatValue, n2o: FloatValue) -> FloatValue {
        let p = &self.parameters;
        let ch4_bar = (ch4 + p.ch4_pi) / 2.0;
        let n2o_bar = (n2o + p.n2o_pi) / 2.0;

        let coeff = p.etminan_ch4_b1 * ch4_bar + p.etminan_ch4_b2 * n2o_bar + p.etminan_ch4_b3;

        coeff * (ch4.sqrt() - p.ch4_pi.sqrt())
    }

    fn n2o_forcing_etminan(&self, ch4: FloatValue, n2o: FloatValue) -> FloatValue {
        let p = &self.parameters;
        let ch4_bar = (ch4 + p.ch4_pi) / 2.0;
        let n2o_bar = (n2o + p.n2o_pi) / 2.0;

        let coeff = p.etminan_n2o_c1 * ch4_bar + p.etminan_n2o_c2 * n2o_bar + p.etminan_n2o_c3;

        coeff * (n2o.sqrt() - p.n2o_pi.sqrt())
    }

    /// Calculate all three forcings with rapid adjustment applied
    pub fn calculate_forcings(
        &self,
        co2: FloatValue,
        ch4: FloatValue,
        n2o: FloatValue,
    ) -> GhgForcingResult {
        let p = &self.parameters;

        let co2_raw = self.calculate_co2_forcing(co2, n2o);
        let ch4_raw = self.calculate_ch4_forcing(ch4, n2o);
        let n2o_raw = self.calculate_n2o_forcing(ch4, n2o);

        GhgForcingResult {
            co2_erf: co2_raw * p.adjust_co2,
            ch4_erf: ch4_raw * p.adjust_ch4,
            n2o_erf: n2o_raw * p.adjust_n2o,
        }
    }
}

impl Default for GhgForcing {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde]
impl Component for GhgForcing {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let inputs = GhgForcingInputs::from_input_state(input_state);

        let co2 = inputs.co2_concentration.get();
        let ch4 = inputs.ch4_concentration.get();
        let n2o = inputs.n2o_concentration.get();

        let result = self.calculate_forcings(co2, ch4, n2o);

        let outputs = GhgForcingOutputs {
            co2_erf: result.co2_erf,
            ch4_erf: result.ch4_erf,
            n2o_erf: result.n2o_erf,
        };

        Ok(outputs.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::ForcingMethod;
    use rscm_core::component::RequirementType;

    fn ipcctar_component() -> GhgForcing {
        GhgForcing::from_parameters(GhgForcingParameters {
            method: ForcingMethod::Ipcctar,
            adjust_co2: 1.0, // No adjustment for pure forcing tests
            adjust_ch4: 1.0,
            adjust_n2o: 1.0,
            ..GhgForcingParameters::default()
        })
    }

    fn etminan_component() -> GhgForcing {
        GhgForcing::from_parameters(GhgForcingParameters {
            method: ForcingMethod::Etminan,
            adjust_co2: 1.0,
            adjust_ch4: 1.0,
            adjust_n2o: 1.0,
            ..GhgForcingParameters::default()
        })
    }

    // ===== IPCCTAR CO2 Tests =====

    #[test]
    fn test_ipcctar_co2_zero_at_preindustrial() {
        let c = ipcctar_component();
        let f = c.calculate_co2_forcing(278.0, 270.0);
        assert!(
            f.abs() < 1e-10,
            "CO2 forcing at PI should be zero, got {}",
            f
        );
    }

    #[test]
    fn test_ipcctar_co2_at_2x() {
        let c = ipcctar_component();
        let f = c.calculate_co2_forcing(556.0, 270.0);
        // Should equal delq2xco2 = 3.71
        assert!(
            (f - 3.71).abs() < 0.01,
            "CO2 forcing at 2xCO2 should be ~3.71, got {}",
            f
        );
    }

    #[test]
    fn test_ipcctar_co2_logarithmic() {
        let c = ipcctar_component();
        let f_2x = c.calculate_co2_forcing(556.0, 270.0);
        let f_4x = c.calculate_co2_forcing(1112.0, 270.0);
        // 4x should be exactly 2 * 2x for log relationship
        assert!(
            (f_4x - 2.0 * f_2x).abs() < 0.01,
            "4xCO2 forcing should be 2 * 2xCO2: got {} vs {}",
            f_4x,
            2.0 * f_2x
        );
    }

    // ===== IPCCTAR CH4 Tests =====

    #[test]
    fn test_ipcctar_ch4_zero_at_preindustrial() {
        let c = ipcctar_component();
        let f = c.calculate_ch4_forcing(722.0, 270.0);
        assert!(
            f.abs() < 1e-10,
            "CH4 forcing at PI should be zero, got {}",
            f
        );
    }

    #[test]
    fn test_ipcctar_ch4_positive_above_pi() {
        let c = ipcctar_component();
        let f = c.calculate_ch4_forcing(1900.0, 270.0);
        assert!(
            f > 0.0,
            "CH4 forcing above PI should be positive, got {}",
            f
        );
    }

    #[test]
    fn test_ipcctar_ch4_realistic_modern() {
        let c = ipcctar_component();
        let f = c.calculate_ch4_forcing(1900.0, 270.0);
        // AR6: CH4 forcing ~0.54 W/m2 (raw forcing without adjustment)
        assert!(
            f > 0.3 && f < 0.8,
            "Modern CH4 forcing should be in realistic range, got {} W/m2",
            f
        );
    }

    // ===== IPCCTAR N2O Tests =====

    #[test]
    fn test_ipcctar_n2o_zero_at_preindustrial() {
        let c = ipcctar_component();
        let f = c.calculate_n2o_forcing(722.0, 270.0);
        assert!(
            f.abs() < 1e-10,
            "N2O forcing at PI should be zero, got {}",
            f
        );
    }

    #[test]
    fn test_ipcctar_n2o_positive_above_pi() {
        let c = ipcctar_component();
        let f = c.calculate_n2o_forcing(722.0, 332.0);
        assert!(
            f > 0.0,
            "N2O forcing above PI should be positive, got {}",
            f
        );
    }

    #[test]
    fn test_ipcctar_n2o_realistic_modern() {
        let c = ipcctar_component();
        let f = c.calculate_n2o_forcing(722.0, 332.0);
        // AR6: N2O forcing ~0.21 W/m2
        assert!(
            f > 0.1 && f < 0.4,
            "Modern N2O forcing should be in realistic range, got {} W/m2",
            f
        );
    }

    // ===== CH4-N2O Overlap Tests =====

    #[test]
    fn test_ipcctar_overlap_nonzero() {
        let c = ipcctar_component();

        let ch4 = 1900.0;
        let n2o = 332.0;

        let f_ch4_with_overlap = c.calculate_ch4_forcing(ch4, n2o);
        let f_ch4_no_overlap = c.calculate_ch4_forcing(ch4, 270.0);

        let f_n2o_with_overlap = c.calculate_n2o_forcing(ch4, n2o);
        let f_n2o_no_overlap = c.calculate_n2o_forcing(722.0, n2o);

        // Both overlaps should be non-zero
        let ch4_overlap = f_ch4_no_overlap - f_ch4_with_overlap;
        let n2o_overlap = f_n2o_no_overlap - f_n2o_with_overlap;

        assert!(
            ch4_overlap.abs() > 1e-6 || n2o_overlap.abs() > 1e-6,
            "Overlap corrections should be non-zero: CH4={}, N2O={}",
            ch4_overlap,
            n2o_overlap
        );
    }

    // ===== Etminan CO2 Tests =====

    #[test]
    fn test_etminan_co2_zero_at_preindustrial() {
        let c = etminan_component();
        let f = c.calculate_co2_forcing(278.0, 270.0);
        assert!(
            f.abs() < 1e-10,
            "CO2 forcing at PI should be zero, got {}",
            f
        );
    }

    #[test]
    fn test_etminan_co2_positive_above_pi() {
        let c = etminan_component();
        let f = c.calculate_co2_forcing(400.0, 270.0);
        assert!(
            f > 0.0,
            "CO2 forcing above PI should be positive, got {}",
            f
        );
    }

    #[test]
    fn test_etminan_co2_differs_from_ipcctar() {
        let ipcc = ipcctar_component();
        let etm = etminan_component();

        let f_ipcc = ipcc.calculate_co2_forcing(560.0, 270.0);
        let f_etm = etm.calculate_co2_forcing(560.0, 270.0);

        // They should give similar but not identical results
        assert!(
            (f_ipcc - f_etm).abs() > 0.01,
            "IPCCTAR and Etminan should give different results at 2xCO2: {} vs {}",
            f_ipcc,
            f_etm
        );
        assert!(
            (f_ipcc - f_etm).abs() < 1.0,
            "But the difference should be modest: {} vs {}",
            f_ipcc,
            f_etm
        );
    }

    // ===== Etminan CH4 Tests =====

    #[test]
    fn test_etminan_ch4_zero_at_preindustrial() {
        let c = etminan_component();
        let f = c.calculate_ch4_forcing(722.0, 270.0);
        assert!(
            f.abs() < 1e-10,
            "CH4 forcing at PI should be zero, got {}",
            f
        );
    }

    #[test]
    fn test_etminan_ch4_positive_above_pi() {
        let c = etminan_component();
        let f = c.calculate_ch4_forcing(1900.0, 270.0);
        assert!(
            f > 0.0,
            "CH4 forcing above PI should be positive, got {}",
            f
        );
    }

    // ===== Etminan N2O Tests =====

    #[test]
    fn test_etminan_n2o_zero_at_preindustrial() {
        let c = etminan_component();
        let f = c.calculate_n2o_forcing(722.0, 270.0);
        assert!(
            f.abs() < 1e-10,
            "N2O forcing at PI should be zero, got {}",
            f
        );
    }

    #[test]
    fn test_etminan_n2o_positive_above_pi() {
        let c = etminan_component();
        let f = c.calculate_n2o_forcing(722.0, 332.0);
        assert!(
            f > 0.0,
            "N2O forcing above PI should be positive, got {}",
            f
        );
    }

    // ===== Rapid Adjustment Tests =====

    #[test]
    fn test_rapid_adjustment_applied() {
        let c = GhgForcing::from_parameters(GhgForcingParameters {
            method: ForcingMethod::Ipcctar,
            adjust_co2: 1.05,
            adjust_ch4: 0.86,
            adjust_n2o: 0.93,
            ..GhgForcingParameters::default()
        });

        let result = c.calculate_forcings(400.0, 1900.0, 332.0);

        // Build a no-adjustment version for comparison
        let c_raw = GhgForcing::from_parameters(GhgForcingParameters {
            method: ForcingMethod::Ipcctar,
            adjust_co2: 1.0,
            adjust_ch4: 1.0,
            adjust_n2o: 1.0,
            ..GhgForcingParameters::default()
        });

        let raw = c_raw.calculate_forcings(400.0, 1900.0, 332.0);

        assert!(
            (result.co2_erf - raw.co2_erf * 1.05).abs() < 1e-10,
            "CO2 adjustment: expected {}, got {}",
            raw.co2_erf * 1.05,
            result.co2_erf
        );
        assert!(
            (result.ch4_erf - raw.ch4_erf * 0.86).abs() < 1e-10,
            "CH4 adjustment: expected {}, got {}",
            raw.ch4_erf * 0.86,
            result.ch4_erf
        );
        assert!(
            (result.n2o_erf - raw.n2o_erf * 0.93).abs() < 1e-10,
            "N2O adjustment: expected {}, got {}",
            raw.n2o_erf * 0.93,
            result.n2o_erf
        );
    }

    // ===== Component Trait Tests =====

    #[test]
    fn test_definitions() {
        let c = ipcctar_component();
        let defs = c.definitions();

        assert_eq!(defs.len(), 6); // 3 inputs + 3 outputs

        let input_names: Vec<_> = defs
            .iter()
            .filter(|d| d.requirement_type == RequirementType::Input)
            .map(|d| d.name.as_str())
            .collect();

        assert!(input_names.contains(&"Atmospheric Concentration|CO2"));
        assert!(input_names.contains(&"Atmospheric Concentration|CH4"));
        assert!(input_names.contains(&"Atmospheric Concentration|N2O"));

        let output_names: Vec<_> = defs
            .iter()
            .filter(|d| d.requirement_type == RequirementType::Output)
            .map(|d| d.name.as_str())
            .collect();

        assert!(output_names.contains(&"Effective Radiative Forcing|CO2"));
        assert!(output_names.contains(&"Effective Radiative Forcing|CH4"));
        assert!(output_names.contains(&"Effective Radiative Forcing|N2O"));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let component = GhgForcing::from_parameters(GhgForcingParameters {
            method: ForcingMethod::Ipcctar,
            delq2xco2: 3.80,
            ..GhgForcingParameters::default()
        });

        let json = serde_json::to_string(&component).unwrap();
        let restored: GhgForcing = serde_json::from_str(&json).unwrap();

        assert_eq!(component.parameters().method, restored.parameters().method);
        assert!((component.parameters().delq2xco2 - restored.parameters().delq2xco2).abs() < 1e-10);
    }

    // ===== Total Forcing Tests =====

    #[test]
    fn test_total_forcing_is_sum() {
        let c = ipcctar_component();
        let result = c.calculate_forcings(400.0, 1900.0, 332.0);

        let expected_total = result.co2_erf + result.ch4_erf + result.n2o_erf;
        assert!(
            (result.total() - expected_total).abs() < 1e-10,
            "Total should be sum of components"
        );
    }

    #[test]
    fn test_all_forcings_positive_above_pi() {
        let c = ipcctar_component();
        let result = c.calculate_forcings(400.0, 1900.0, 332.0);

        assert!(
            result.co2_erf > 0.0,
            "CO2 forcing should be positive above PI"
        );
        assert!(
            result.ch4_erf > 0.0,
            "CH4 forcing should be positive above PI"
        );
        assert!(
            result.n2o_erf > 0.0,
            "N2O forcing should be positive above PI"
        );
    }

    #[test]
    fn test_etminan_all_forcings_positive_above_pi() {
        let c = etminan_component();
        let result = c.calculate_forcings(400.0, 1900.0, 332.0);

        assert!(
            result.co2_erf > 0.0,
            "CO2 forcing should be positive above PI"
        );
        assert!(
            result.ch4_erf > 0.0,
            "CH4 forcing should be positive above PI"
        );
        assert!(
            result.n2o_erf > 0.0,
            "N2O forcing should be positive above PI"
        );
    }
}
