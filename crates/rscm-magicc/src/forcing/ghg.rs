//! GHG Radiative Forcing Component
//!
//! Calculates effective radiative forcing (ERF) from well-mixed greenhouse
//! gases (CO2, CH4, N2O) following MAGICC7 forcing methods.
//!
//! # What This Component Does
//!
//! 1. Calculates instantaneous radiative forcing from CO2, CH4, and N2O
//!    concentrations using either IPCCTAR or OLBL methods
//! 2. Accounts for CH4-N2O absorption band overlap
//! 3. Applies rapid adjustment factors to convert to effective radiative forcing
//!
//! # Inputs
//!
//! - `Atmospheric Concentrations|CO2` (ppm) - CO2 concentration
//! - `Atmospheric Concentrations|CH4` (ppb) - CH4 concentration
//! - `Atmospheric Concentrations|N2O` (ppb) - N2O concentration
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
//! ## Oslo line-by-line (OLBL)
//!
//! Meinshausen et al. (2020), shared with FaIR in [`rscm_components::meinshausen2020`],
//! with MAGICC7's coefficients and the pre-industrial concentrations as reference.

use crate::parameters::{ForcingMethod, GhgForcingParameters};
use rscm_core::component::{
    Component, GridType, InputState, OutputState, VariableDefinition, VariableRole,
};
use rscm_core::errors::RscmResult;
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
/// # CO2 Forcing (OLBL)
///
/// $$\alpha = a_1 (\Delta C)^2 + b_1 \Delta C + d_1 + c_1 \sqrt{N_2O}$$
/// $$F_{CO2} = \alpha \cdot \ln(C / C_0)$$
///
/// # CH4-N2O Overlap (IPCCTAR)
///
/// $$f(M, N) = 0.47 \cdot \ln\left(1 + 2.01 \times 10^{-5} (MN)^{0.75}
///     + 5.31 \times 10^{-15} M (MN)^{1.52}\right)$$
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["forcing", "ghg", "co2", "ch4", "n2o", "magicc"], category = "Radiative Forcing")]
#[inputs(
    co2_concentration { name = "Atmospheric Concentrations|CO2", unit = "ppm" },
    ch4_concentration { name = "Atmospheric Concentrations|CH4", unit = "ppb" },
    n2o_concentration { name = "Atmospheric Concentrations|N2O", unit = "ppb" },
)]
#[outputs(
    co2_erf { name = "Effective Radiative Forcing|CO2", unit = "W/m^2" },
    ch4_erf { name = "Effective Radiative Forcing|CH4", unit = "W/m^2" },
    n2o_erf { name = "Effective Radiative Forcing|N2O", unit = "W/m^2" },
    erf_2xco2 { name = "2xCO2 Effective Radiative Forcing", unit = "W/m^2" },
    ch4_strat_h2o_erf { name = "Effective Radiative Forcing|CH4 Oxidation Stratospheric H2O", unit = "W/m^2" },
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

    /// Argument of the IPCCTAR CH4-N2O overlap (Myhre et al. 1998), in MAGICC7's ppm form.
    ///
    /// $$g(M, N) = 1 + 0.6356 (MN)^{0.75} + 0.007 M (MN)^{1.52}$$
    ///
    /// where $M$ and $N$ are CH4 and N2O concentrations in ppm.
    /// The overlap between two states is $0.47 \ln(g_0 / g)$.
    fn overlap_g(ch4_ppb: FloatValue, n2o_ppb: FloatValue) -> FloatValue {
        let (m, n) = (ch4_ppb / 1000.0, n2o_ppb / 1000.0);
        let mn = m * n;
        (0.6356 * mn.powf(0.75) + (0.007 * m) * mn.powf(1.52)) + 1.0
    }

    /// Calculate CO2 forcing (dispatches to method)
    ///
    /// The `n2o` parameter is needed for the OLBL method (N2O affects CO2
    /// forcing coefficients via overlap). For IPCCTAR it is unused.
    pub fn calculate_co2_forcing(&self, co2: FloatValue, n2o: FloatValue) -> FloatValue {
        match self.parameters.method {
            ForcingMethod::Ipcctar => self.co2_forcing_ipcctar(co2),
            ForcingMethod::Olbl => {
                let p = &self.parameters;
                p.olbl().co2(co2, n2o, p.co2_pi, 1.0)
            }
        }
    }

    /// Calculate CH4 forcing (dispatches to method)
    pub fn calculate_ch4_forcing(&self, ch4: FloatValue, n2o: FloatValue) -> FloatValue {
        match self.parameters.method {
            ForcingMethod::Ipcctar => self.ch4_forcing_ipcctar(ch4, n2o),
            ForcingMethod::Olbl => {
                let p = &self.parameters;
                p.olbl().ch4(ch4, n2o, p.ch4_pi, 1.0)
            }
        }
    }

    /// Calculate N2O forcing (dispatches to method)
    ///
    /// The `co2` parameter is needed for the OLBL method (CO2 affects N2O
    /// forcing via overlap). For IPCCTAR it is unused.
    pub fn calculate_n2o_forcing(
        &self,
        co2: FloatValue,
        ch4: FloatValue,
        n2o: FloatValue,
    ) -> FloatValue {
        match self.parameters.method {
            ForcingMethod::Ipcctar => self.n2o_forcing_ipcctar(ch4, n2o),
            ForcingMethod::Olbl => {
                let p = &self.parameters;
                p.olbl().n2o(co2, ch4, n2o, p.n2o_pi, 1.0)
            }
        }
    }

    // === IPCCTAR implementations ===

    fn co2_forcing_ipcctar(&self, co2: FloatValue) -> FloatValue {
        let p = &self.parameters;
        let alpha = p.erf_2xco2 / 2.0_f64.ln();
        alpha * (co2 / p.co2_pi).ln()
    }

    fn ch4_forcing_ipcctar(&self, ch4: FloatValue, _n2o: FloatValue) -> FloatValue {
        let p = &self.parameters;
        let direct = (ch4.sqrt() - p.ch4_pi.sqrt()) * p.ch4_radiative_efficiency;
        let overlap = Self::overlap_g(p.ch4_pi, p.n2o_pi) / Self::overlap_g(ch4, p.n2o_pi);
        0.47 * overlap.ln() + direct
    }

    fn n2o_forcing_ipcctar(&self, _ch4: FloatValue, n2o: FloatValue) -> FloatValue {
        let p = &self.parameters;
        let direct = (n2o.sqrt() - p.n2o_pi.sqrt()) * p.n2o_radiative_efficiency;
        let overlap = Self::overlap_g(p.ch4_pi, p.n2o_pi) / Self::overlap_g(p.ch4_pi, n2o);
        direct + 0.47 * overlap.ln()
    }

    // === OLBL implementations (MAGICC7 v7.5.3) ===

    /// 2xCO2 effective radiative forcing at the current concentrations (`CORE_DELQ2XCO2`).
    ///
    /// MAGICC7 recalculates this every year for OLBL, which changes the climate
    /// feedback parameter as CO2 rises. IPCCTAR keeps the configured value.
    pub fn erf_2xco2(&self, co2: FloatValue, n2o: FloatValue) -> FloatValue {
        let p = &self.parameters;
        match p.method {
            ForcingMethod::Ipcctar => p.erf_2xco2 * p.adjust_co2,
            ForcingMethod::Olbl => {
                p.olbl().co2_alpha(co2, n2o, p.co2_pi) * p.adjust_co2 * 2.0_f64.ln()
            }
        }
    }

    /// ERF of stratospheric H2O from CH4 oxidation (`CH4OXSTRATH2O`).
    pub fn ch4_strat_h2o_erf(&self, ch4: FloatValue) -> FloatValue {
        let p = &self.parameters;
        let ch4_alone = match p.method {
            // MAGICC7 leaves out the N2O overlap for IPCCTAR
            ForcingMethod::Ipcctar => {
                p.adjust_ch4 * p.ch4_radiative_efficiency * (ch4.sqrt() - p.ch4_pi.sqrt())
            }
            ForcingMethod::Olbl => p.olbl().ch4(ch4, p.n2o_pi, p.ch4_pi, p.adjust_ch4),
        };
        ch4_alone * p.ch4_stratospheric_h2o_fraction
    }

    /// Calculate all three forcings with rapid adjustment applied
    pub fn calculate_forcings(
        &self,
        co2: FloatValue,
        ch4: FloatValue,
        n2o: FloatValue,
    ) -> GhgForcingResult {
        let p = &self.parameters;

        match p.method {
            ForcingMethod::Ipcctar => GhgForcingResult {
                co2_erf: self.co2_forcing_ipcctar(co2) * p.adjust_co2,
                ch4_erf: self.ch4_forcing_ipcctar(ch4, n2o) * p.adjust_ch4,
                n2o_erf: self.n2o_forcing_ipcctar(ch4, n2o) * p.adjust_n2o,
            },
            ForcingMethod::Olbl => {
                let m = p.olbl();
                GhgForcingResult {
                    co2_erf: m.co2(co2, n2o, p.co2_pi, p.adjust_co2),
                    ch4_erf: m.ch4(ch4, n2o, p.ch4_pi, p.adjust_ch4),
                    n2o_erf: m.n2o(co2, ch4, n2o, p.n2o_pi, p.adjust_n2o),
                }
            }
        }
    }
}

impl Default for GhgForcing {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde(name = "magicc.GhgForcing")]
impl Component for GhgForcing {
    fn definitions(&self) -> Vec<VariableDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RscmResult<OutputState> {
        let inputs = GhgForcingInputs::from_input_state(input_state);

        let co2 = inputs.co2_concentration.get();
        let ch4 = inputs.ch4_concentration.get();
        let n2o = inputs.n2o_concentration.get();

        let result = self.calculate_forcings(co2, ch4, n2o);

        let outputs = GhgForcingOutputs {
            co2_erf: result.co2_erf,
            ch4_erf: result.ch4_erf,
            n2o_erf: result.n2o_erf,
            erf_2xco2: self.erf_2xco2(co2, n2o),
            ch4_strat_h2o_erf: self.ch4_strat_h2o_erf(ch4),
        };

        Ok(outputs.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::ForcingMethod;
    use rscm_core::component::VariableRole;

    fn ipcctar_component() -> GhgForcing {
        GhgForcing::from_parameters(GhgForcingParameters {
            method: ForcingMethod::Ipcctar,
            adjust_co2: 1.0, // No adjustment for pure forcing tests
            adjust_ch4: 1.0,
            adjust_n2o: 1.0,
            ..GhgForcingParameters::default()
        })
    }

    fn olbl_component() -> GhgForcing {
        GhgForcing::from_parameters(GhgForcingParameters {
            method: ForcingMethod::Olbl,
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
        // Should equal erf_2xco2 = 3.71
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
        let f = c.calculate_n2o_forcing(278.0, 722.0, 270.0);
        assert!(
            f.abs() < 1e-10,
            "N2O forcing at PI should be zero, got {}",
            f
        );
    }

    #[test]
    fn test_ipcctar_n2o_positive_above_pi() {
        let c = ipcctar_component();
        let f = c.calculate_n2o_forcing(278.0, 722.0, 332.0);
        assert!(
            f > 0.0,
            "N2O forcing above PI should be positive, got {}",
            f
        );
    }

    #[test]
    fn test_ipcctar_n2o_realistic_modern() {
        let c = ipcctar_component();
        let f = c.calculate_n2o_forcing(278.0, 722.0, 332.0);
        // AR6: N2O forcing ~0.21 W/m2
        assert!(
            f > 0.1 && f < 0.4,
            "Modern N2O forcing should be in realistic range, got {} W/m2",
            f
        );
    }

    // ===== CH4-N2O Overlap Tests =====

    #[test]
    fn test_ipcctar_overlap_reduces_forcing() {
        let c = ipcctar_component();
        let p = c.parameters();

        // The overlap correction reduces the forcing from the direct term.
        // At elevated CH4 (above PI), the overlap should make the CH4
        // forcing smaller than the pure direct term alone.
        let ch4: f64 = 1900.0;
        let direct_ch4 = p.ch4_radiative_efficiency * (ch4.sqrt() - p.ch4_pi.sqrt());
        let actual_ch4 = c.calculate_ch4_forcing(ch4, p.n2o_pi);

        assert!(
            actual_ch4 < direct_ch4,
            "Overlap should reduce CH4 forcing: actual={}, direct={}",
            actual_ch4,
            direct_ch4
        );

        // Same for N2O
        let n2o: f64 = 332.0;
        let direct_n2o = p.n2o_radiative_efficiency * (n2o.sqrt() - p.n2o_pi.sqrt());
        let actual_n2o = c.calculate_n2o_forcing(p.co2_pi, p.ch4_pi, n2o);

        assert!(
            actual_n2o < direct_n2o,
            "Overlap should reduce N2O forcing: actual={}, direct={}",
            actual_n2o,
            direct_n2o
        );
    }

    // ===== OLBL CO2 Tests =====

    #[test]
    fn test_olbl_co2_zero_at_preindustrial() {
        let c = olbl_component();
        let f = c.calculate_co2_forcing(278.0, 270.0);
        assert!(
            f.abs() < 1e-10,
            "CO2 forcing at PI should be zero, got {}",
            f
        );
    }

    #[test]
    fn test_olbl_co2_positive_above_pi() {
        let c = olbl_component();
        let f = c.calculate_co2_forcing(400.0, 270.0);
        assert!(
            f > 0.0,
            "CO2 forcing above PI should be positive, got {}",
            f
        );
    }

    #[test]
    fn test_olbl_co2_differs_from_ipcctar() {
        let ipcc = ipcctar_component();
        let olbl = olbl_component();

        let f_ipcc = ipcc.calculate_co2_forcing(560.0, 270.0);
        let f_olbl = olbl.calculate_co2_forcing(560.0, 270.0);

        // They should give similar but not identical results
        assert!(
            (f_ipcc - f_olbl).abs() > 1e-4,
            "IPCCTAR and OLBL should give different results at 2xCO2: {} vs {}",
            f_ipcc,
            f_olbl
        );
        assert!(
            (f_ipcc - f_olbl).abs() < 1.0,
            "But the difference should be modest: {} vs {}",
            f_ipcc,
            f_olbl
        );
    }

    // ===== OLBL CH4 Tests =====

    #[test]
    fn test_olbl_ch4_zero_at_preindustrial() {
        let c = olbl_component();
        let f = c.calculate_ch4_forcing(722.0, 270.0);
        assert!(
            f.abs() < 1e-10,
            "CH4 forcing at PI should be zero, got {}",
            f
        );
    }

    #[test]
    fn test_olbl_ch4_positive_above_pi() {
        let c = olbl_component();
        let f = c.calculate_ch4_forcing(1900.0, 270.0);
        assert!(
            f > 0.0,
            "CH4 forcing above PI should be positive, got {}",
            f
        );
    }

    // ===== OLBL N2O Tests =====

    #[test]
    fn test_olbl_n2o_zero_at_preindustrial() {
        let c = olbl_component();
        let f = c.calculate_n2o_forcing(278.0, 722.0, 270.0);
        assert!(
            f.abs() < 1e-10,
            "N2O forcing at PI should be zero, got {}",
            f
        );
    }

    #[test]
    fn test_olbl_n2o_positive_above_pi() {
        let c = olbl_component();
        let f = c.calculate_n2o_forcing(278.0, 722.0, 332.0);
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

        assert_eq!(defs.len(), 8); // 3 inputs + 5 outputs

        let input_names: Vec<_> = defs
            .iter()
            .filter(|d| d.role == VariableRole::Input)
            .map(|d| d.name.as_str())
            .collect();

        assert!(input_names.contains(&"Atmospheric Concentrations|CO2"));
        assert!(input_names.contains(&"Atmospheric Concentrations|CH4"));
        assert!(input_names.contains(&"Atmospheric Concentrations|N2O"));

        let output_names: Vec<_> = defs
            .iter()
            .filter(|d| d.role == VariableRole::Output)
            .map(|d| d.name.as_str())
            .collect();

        assert!(output_names.contains(&"Effective Radiative Forcing|CO2"));
        assert!(output_names.contains(&"Effective Radiative Forcing|CH4"));
        assert!(output_names.contains(&"Effective Radiative Forcing|N2O"));
        assert!(output_names.contains(&"2xCO2 Effective Radiative Forcing"));
    }

    #[test]
    fn test_serialisation_roundtrip() {
        let component = GhgForcing::from_parameters(GhgForcingParameters {
            method: ForcingMethod::Ipcctar,
            erf_2xco2: 3.80,
            ..GhgForcingParameters::default()
        });

        let json = serde_json::to_string(&component).unwrap();
        let restored: GhgForcing = serde_json::from_str(&json).unwrap();

        assert_eq!(component.parameters().method, restored.parameters().method);
        assert!((component.parameters().erf_2xco2 - restored.parameters().erf_2xco2).abs() < 1e-10);
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
    fn test_olbl_all_forcings_positive_above_pi() {
        let c = olbl_component();
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
