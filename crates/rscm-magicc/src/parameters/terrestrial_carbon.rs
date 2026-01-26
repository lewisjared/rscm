//! Terrestrial Carbon Parameters
//!
//! Parameters for the 4-pool terrestrial carbon cycle model with CO2 fertilization
//! and temperature feedbacks.
//!
//! # Reference
//!
//! Based on MAGICC7 Module 09 (Terrestrial Carbon Cycle) which implements a
//! simplified 4-box model representing plant biomass, detritus, soil, and humus pools.

use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Parameters for terrestrial carbon cycle calculations.
///
/// The terrestrial carbon cycle tracks four carbon pools:
/// 1. Plant biomass - living vegetation (leaves, stems, roots)
/// 2. Detritus - dead organic matter in transition
/// 3. Soil - medium-lived organic carbon in soils
/// 4. Humus - long-lived soil carbon (slow pool)
///
/// # Carbon Flows
///
/// ```text
///                      NPP
///         Atmosphere -----> [PLANT]
///               ^              |
///               |              | turnover
///               |              v
///         [RESPIRATION] <-- [DETRITUS] --> [SOIL] --> [HUMUS]
/// ```
///
/// # Key Feedbacks
///
/// 1. **CO2 Fertilization**: Higher atmospheric CO2 enhances NPP:
///    $$\text{NPP} = \text{NPP}_0 \times (1 + \beta \times \ln(\text{CO}_2/\text{CO}_{2,\text{pi}}))$$
///
/// 2. **Temperature-Respiration Feedback**: Warmer temperatures accelerate decay:
///    $$f_T = \exp(\gamma \times \Delta T)$$
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrestrialCarbonParameters {
    /// Pre-industrial Net Primary Production (NPP)
    /// unit: GtC/yr
    /// default: 66.27 (MAGICC7 default)
    pub npp_pi: FloatValue,

    /// Pre-industrial CO2 concentration used as reference for fertilization
    /// unit: ppm
    /// default: 278.0
    pub co2_pi: FloatValue,

    /// CO2 fertilization factor (β in logarithmic formula)
    /// Controls the strength of CO2 fertilization effect on NPP.
    /// Higher values mean stronger fertilization response.
    /// unit: dimensionless
    /// default: 0.6486
    pub beta: FloatValue,

    /// NPP temperature sensitivity coefficient (γ_NPP)
    /// Positive means warming increases NPP.
    /// unit: K⁻¹
    /// default: 0.0107
    pub npp_temp_sensitivity: FloatValue,

    /// Respiration temperature sensitivity coefficient (γ_resp)
    /// Controls how plant pool respiration responds to temperature.
    /// Positive means warming increases respiration.
    /// unit: K⁻¹
    /// default: 0.0685
    pub resp_temp_sensitivity: FloatValue,

    /// Detritus decay temperature sensitivity coefficient (γ_detritus)
    /// Controls how detritus decay responds to temperature.
    /// Positive means warming increases decay rate (releases CO2).
    /// unit: K⁻¹
    /// default: 0.1358 (note: some MAGICC7 configs use negative value)
    pub detritus_temp_sensitivity: FloatValue,

    /// Soil decay temperature sensitivity coefficient (γ_soil)
    /// Controls how soil decay responds to temperature.
    /// Positive means warming increases decay rate (releases CO2).
    /// unit: K⁻¹
    /// default: 0.1541
    pub soil_temp_sensitivity: FloatValue,

    /// Humus decay temperature sensitivity coefficient (γ_humus)
    /// Controls how humus decay responds to temperature.
    /// Positive means warming increases decay rate (releases CO2).
    /// unit: K⁻¹
    /// default: 0.05 (slow pool, less sensitive)
    pub humus_temp_sensitivity: FloatValue,

    /// Pre-industrial plant pool carbon content
    /// unit: GtC
    /// default: 884.86
    pub plant_pool_pi: FloatValue,

    /// Pre-industrial detritus pool carbon content
    /// unit: GtC
    /// default: 92.77
    pub detritus_pool_pi: FloatValue,

    /// Pre-industrial soil pool carbon content
    /// unit: GtC
    /// default: 1681.53
    pub soil_pool_pi: FloatValue,

    /// Pre-industrial humus pool carbon content
    /// unit: GtC
    /// default: 836.0 (derived to give ~1000yr turnover)
    pub humus_pool_pi: FloatValue,

    /// Pre-industrial respiration from plant pool (R_h0)
    /// unit: GtC/yr
    /// default: 12.26
    pub respiration_pi: FloatValue,

    /// Fraction of NPP going directly to plant pool
    /// unit: dimensionless
    /// default: 0.4483
    pub frac_npp_to_plant: FloatValue,

    /// Fraction of NPP going directly to detritus pool
    /// unit: dimensionless
    /// default: 0.3998
    pub frac_npp_to_detritus: FloatValue,

    /// Fraction of plant turnover flux going to detritus (vs soil)
    /// unit: dimensionless
    /// default: 0.9989
    pub frac_plant_to_detritus: FloatValue,

    /// Fraction of detritus decay going to soil (vs atmosphere)
    /// unit: dimensionless
    /// default: 0.3
    pub frac_detritus_to_soil: FloatValue,

    /// Fraction of soil decay going to humus (vs atmosphere)
    /// unit: dimensionless
    /// default: 0.1
    pub frac_soil_to_humus: FloatValue,

    /// Enable CO2 fertilization feedback
    /// default: true
    pub enable_fertilization: bool,

    /// Enable temperature feedback on decay rates
    /// default: true
    pub enable_temp_feedback: bool,
}

impl Default for TerrestrialCarbonParameters {
    fn default() -> Self {
        Self {
            // NPP and CO2
            npp_pi: 66.27,
            co2_pi: 278.0,
            beta: 0.6486,

            // Temperature sensitivities
            npp_temp_sensitivity: 0.0107,
            resp_temp_sensitivity: 0.0685,
            detritus_temp_sensitivity: 0.1358,
            soil_temp_sensitivity: 0.1541,
            humus_temp_sensitivity: 0.05,

            // Pre-industrial pool sizes
            plant_pool_pi: 884.86,
            detritus_pool_pi: 92.77,
            soil_pool_pi: 1681.53,
            humus_pool_pi: 836.0,

            // Respiration
            respiration_pi: 12.26,

            // Carbon flow fractions
            frac_npp_to_plant: 0.4483,
            frac_npp_to_detritus: 0.3998,
            frac_plant_to_detritus: 0.9989,
            frac_detritus_to_soil: 0.3,
            frac_soil_to_humus: 0.1,

            // Feedback switches
            enable_fertilization: true,
            enable_temp_feedback: true,
        }
    }
}

impl TerrestrialCarbonParameters {
    /// Fraction of NPP going directly to soil pool (derived).
    ///
    /// Calculated as `1 - frac_npp_to_plant - frac_npp_to_detritus`.
    pub fn frac_npp_to_soil(&self) -> FloatValue {
        let f = 1.0 - self.frac_npp_to_plant - self.frac_npp_to_detritus;
        f.max(0.0)
    }

    /// Net flux to plant pool at pre-industrial (NPP to plant - respiration).
    ///
    /// This is used to derive the initial turnover time for the plant pool.
    pub fn net_flux_to_plant_pi(&self) -> FloatValue {
        self.frac_npp_to_plant * self.npp_pi - self.respiration_pi
    }

    /// Initial turnover time for plant pool at pre-industrial (years).
    ///
    /// Derived from steady-state assumption: pool_size / net_flux
    pub fn tau_plant_pi(&self) -> FloatValue {
        let net_flux = self.net_flux_to_plant_pi();
        if net_flux > 1e-10 {
            self.plant_pool_pi / net_flux
        } else {
            100.0 // Fallback value
        }
    }

    /// Initial turnover time for detritus pool at pre-industrial (years).
    ///
    /// Derived from steady-state: detritus_pool / flux_into_detritus
    pub fn tau_detritus_pi(&self) -> FloatValue {
        let net_flux_plant = self.net_flux_to_plant_pi();
        let flux_into_detritus =
            self.frac_npp_to_detritus * self.npp_pi + self.frac_plant_to_detritus * net_flux_plant;
        if flux_into_detritus > 1e-10 {
            self.detritus_pool_pi / flux_into_detritus
        } else {
            3.0 // Fallback value
        }
    }

    /// Initial turnover time for soil pool at pre-industrial (years).
    ///
    /// Derived from steady-state assumption.
    pub fn tau_soil_pi(&self) -> FloatValue {
        // At steady state, flux into soil = flux out of soil
        // Flux in = NPP_to_soil + plant_to_soil + detritus_to_soil
        let net_flux_plant = self.net_flux_to_plant_pi();
        let flux_detritus_out = self.detritus_pool_pi / self.tau_detritus_pi();

        let flux_into_soil = self.frac_npp_to_soil() * self.npp_pi
            + (1.0 - self.frac_plant_to_detritus) * net_flux_plant
            + self.frac_detritus_to_soil * flux_detritus_out;

        if flux_into_soil > 1e-10 {
            self.soil_pool_pi / flux_into_soil
        } else {
            50.0 // Fallback value
        }
    }

    /// Initial turnover time for humus pool at pre-industrial (years).
    ///
    /// Derived from steady-state assumption.
    pub fn tau_humus_pi(&self) -> FloatValue {
        // Flux into humus = fraction of soil decay
        let flux_soil_out = self.soil_pool_pi / self.tau_soil_pi();
        let flux_into_humus = self.frac_soil_to_humus * flux_soil_out;

        if flux_into_humus > 1e-10 {
            self.humus_pool_pi / flux_into_humus
        } else {
            1000.0 // Fallback value
        }
    }

    /// Total pre-industrial terrestrial carbon pool (GtC).
    pub fn total_pool_pi(&self) -> FloatValue {
        self.plant_pool_pi + self.detritus_pool_pi + self.soil_pool_pi + self.humus_pool_pi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() {
        let params = TerrestrialCarbonParameters::default();
        assert!((params.npp_pi - 66.27).abs() < 1e-10);
        assert!((params.co2_pi - 278.0).abs() < 1e-10);
        assert!(params.enable_fertilization);
        assert!(params.enable_temp_feedback);
    }

    #[test]
    fn test_npp_fractions_sum_to_one() {
        let params = TerrestrialCarbonParameters::default();
        let sum =
            params.frac_npp_to_plant + params.frac_npp_to_detritus + params.frac_npp_to_soil();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "NPP fractions should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_net_flux_to_plant_positive() {
        let params = TerrestrialCarbonParameters::default();
        let net_flux = params.net_flux_to_plant_pi();
        assert!(
            net_flux > 0.0,
            "Net flux to plant should be positive at steady state, got {}",
            net_flux
        );
    }

    #[test]
    fn test_turnover_times_positive() {
        let params = TerrestrialCarbonParameters::default();

        let tau_plant = params.tau_plant_pi();
        let tau_detritus = params.tau_detritus_pi();
        let tau_soil = params.tau_soil_pi();
        let tau_humus = params.tau_humus_pi();

        assert!(tau_plant > 0.0, "Plant turnover time should be positive");
        assert!(
            tau_detritus > 0.0,
            "Detritus turnover time should be positive"
        );
        assert!(tau_soil > 0.0, "Soil turnover time should be positive");
        assert!(tau_humus > 0.0, "Humus turnover time should be positive");

        // Check relative ordering (plant < detritus < soil < humus typically)
        assert!(
            tau_detritus < tau_soil,
            "Detritus turnover should be faster than soil"
        );
        assert!(
            tau_soil < tau_humus,
            "Soil turnover should be faster than humus"
        );
    }

    #[test]
    fn test_total_pool_calculation() {
        let params = TerrestrialCarbonParameters::default();
        let total = params.total_pool_pi();
        let expected = params.plant_pool_pi
            + params.detritus_pool_pi
            + params.soil_pool_pi
            + params.humus_pool_pi;
        assert!(
            (total - expected).abs() < 1e-10,
            "Total pool calculation incorrect"
        );
    }

    #[test]
    fn test_temperature_sensitivities_positive() {
        let params = TerrestrialCarbonParameters::default();
        // Respiration and decay should increase with temperature (positive sensitivity)
        assert!(
            params.resp_temp_sensitivity > 0.0,
            "Respiration temp sensitivity should be positive"
        );
        assert!(
            params.soil_temp_sensitivity > 0.0,
            "Soil temp sensitivity should be positive"
        );
        assert!(
            params.humus_temp_sensitivity > 0.0,
            "Humus temp sensitivity should be positive"
        );
    }

    #[test]
    fn test_beta_fertilization_reasonable() {
        let params = TerrestrialCarbonParameters::default();
        // Beta should give reasonable fertilization at doubled CO2
        // At 2x CO2: factor = 1 + beta * ln(2) ≈ 1 + 0.6486 * 0.693 ≈ 1.45
        let doubled_co2_factor = 1.0 + params.beta * (2.0_f64).ln();
        assert!(
            (doubled_co2_factor - 1.45).abs() < 0.1,
            "Doubled CO2 should give ~45% NPP increase, got {:.1}%",
            (doubled_co2_factor - 1.0) * 100.0
        );
    }
}
