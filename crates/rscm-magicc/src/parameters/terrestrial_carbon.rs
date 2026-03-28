//! Terrestrial Carbon Parameters
//!
//! Parameters for the MAGICC7 3-pool terrestrial carbon cycle model with
//! three CO2 fertilization methods, two respiration methods, and temperature feedbacks.
//!
//! # Reference
//!
//! Based on MAGICC7 Module 09 (Terrestrial Carbon Cycle) which implements a
//! 3-box model representing plant biomass, detritus, and soil pools.

use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Parameters for terrestrial carbon cycle calculations.
///
/// The terrestrial carbon cycle tracks three carbon pools:
/// 1. Plant biomass - living vegetation (leaves, stems, roots)
/// 2. Detritus - dead organic matter in transition
/// 3. Soil - organic carbon in soils
///
/// # Carbon Flows
///
/// ```text
///                      NPP
///         Atmosphere -----> [PLANT]
///               ^              |
///               |              | turnover
///               |              v
///         [RESPIRATION] <-- [DETRITUS] --> [SOIL]
/// ```
///
/// # Key Feedbacks
///
/// 1. **CO2 Fertilization**: Higher atmospheric CO2 enhances NPP via three
///    blendable methods (logarithmic, Gifford/Michaelis-Menten, sigmoid).
///
/// 2. **Temperature-Respiration Feedback**: Warmer temperatures accelerate decay:
///    $$f_T = \exp(\gamma \times \Delta T)$$
///
/// 3. **Time-varying turnover**: Cumulative deforestation reduces pool turnover times.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TerrestrialCarbonParameters {
    /// Pre-industrial Net Primary Production (NPP)
    /// unit: GtC/yr
    /// MAGICC7: CO2_NPP_INITIAL
    pub npp_pi: FloatValue,

    /// Pre-industrial CO2 concentration used as reference for fertilization
    /// unit: ppm
    /// MAGICC7: CO2_PREINDCO2CONC
    pub co2_pi: FloatValue,

    /// CO2 fertilization factor (beta in logarithmic formula, also used by Gifford/sigmoid)
    /// unit: dimensionless
    /// MAGICC7: CO2_FERTILIZATION_FACTOR
    pub beta: FloatValue,

    /// CO2 fertilization method selector (float for blending):
    /// `< 1.0` = no fertilization,
    /// `1.0` = logarithmic (Keeling-Bacastow 1973),
    /// `2.0` = Gifford rectangular hyperbolic (Michaelis-Menten),
    /// `3.0` = saturating sigmoid (Norton).
    /// Fractional values blend adjacent methods.
    /// unit: dimensionless.
    /// MAGICC7: CO2_FERTILIZATION_METHOD
    pub fertilization_method: FloatValue,

    /// Curvature parameter for sigmoid fertilization method.
    /// unit: ppm
    /// MAGICC7: CO2_FERTILIZATION_FACTOR2
    pub fertilization_factor2: FloatValue,

    /// CO2 concentration at which NPP would be zero (Gifford method).
    /// unit: ppm
    /// MAGICC7: CO2_GIFFORD_CONC_FOR_ZERONPP
    pub gifford_conc_for_zero_npp: FloatValue,

    /// Year before which CO2 fertilization reference tracks current CO2.
    /// unit: year
    /// MAGICC7: CO2_FERTILIZATION_YRSTART
    pub fertilization_yrstart: FloatValue,

    /// NPP temperature sensitivity coefficient (gamma_NPP)
    /// Positive means warming increases NPP.
    /// unit: K^-1
    /// MAGICC7: CO2_FEEDBACKFACTOR_NPP
    pub npp_temp_sensitivity: FloatValue,

    /// Respiration temperature sensitivity coefficient (gamma_resp)
    /// Controls how plant pool respiration responds to temperature.
    /// unit: K^-1
    /// MAGICC7: CO2_FEEDBACKFACTOR_RESPIRATION
    pub resp_temp_sensitivity: FloatValue,

    /// Detritus decay temperature sensitivity coefficient (gamma_detritus)
    /// Positive means warming increases decay rate.
    /// unit: K^-1
    /// MAGICC7: CO2_FEEDBACKFACTOR_DETRITUS
    pub detritus_temp_sensitivity: FloatValue,

    /// Soil decay temperature sensitivity coefficient (gamma_soil)
    /// Positive means warming increases decay rate.
    /// unit: K^-1
    /// MAGICC7: CO2_FEEDBACKFACTOR_SOIL
    pub soil_temp_sensitivity: FloatValue,

    /// Year before which temperature feedbacks are disabled.
    /// unit: year
    /// MAGICC7: CO2_TEMPFEEDBACK_YRSTART
    pub tempfeedback_yrstart: FloatValue,

    /// Pre-industrial plant pool carbon content
    /// unit: GtC
    /// MAGICC7: CO2_PLANTPOOL_INITIAL
    pub plant_pool_pi: FloatValue,

    /// Pre-industrial detritus pool carbon content
    /// unit: GtC
    /// MAGICC7: CO2_DETRITUSPOOL_INITIAL
    pub detritus_pool_pi: FloatValue,

    /// Pre-industrial soil pool carbon content
    /// unit: GtC
    /// MAGICC7: CO2_SOILPOOL_INITIAL
    pub soil_pool_pi: FloatValue,

    /// Pre-industrial respiration from plant pool (R_h0)
    /// unit: GtC/yr
    /// MAGICC7: CO2_RESPIRATION_INITIAL
    pub respiration_pi: FloatValue,

    /// Plant box respiration method.
    /// - 1: R_h = R_h0 * beta * f_T_resp
    /// - 2: R_h = R_h0 * (1 + alpha*(beta-1)) * min(1, C_P/C_P0) * f_T_resp
    /// MAGICC7: CO2_PLANTBOXRESP_METHOD
    pub plantbox_resp_method: u8,

    /// Scaling of fertilization effect on respiration (method 2 only).
    /// unit: dimensionless
    /// MAGICC7: CO2_PLANTBOXRESP_FERTSCALE
    pub plantbox_resp_fertscale: FloatValue,

    /// Fraction of NPP going directly to plant pool
    /// unit: dimensionless
    /// MAGICC7: CO2_FRACTION_NPP_2_PLANT
    pub frac_npp_to_plant: FloatValue,

    /// Fraction of NPP going directly to detritus pool
    /// unit: dimensionless
    /// MAGICC7: CO2_FRACTION_NPP_2_DETRITUS
    pub frac_npp_to_detritus: FloatValue,

    /// Fraction of plant turnover flux going to detritus (vs soil)
    /// unit: dimensionless
    /// MAGICC7: CO2_FRACTION_PLANT_2_DETRITUS
    pub frac_plant_to_detritus: FloatValue,

    /// Fraction of detritus decay going to soil (vs atmosphere)
    /// unit: dimensionless
    /// MAGICC7: CO2_FRACTION_DETRITUS_2_SOIL
    pub frac_detritus_to_soil: FloatValue,

    /// Fraction of deforestation emissions from plant pool
    /// unit: dimensionless
    /// MAGICC7: CO2_FRACTION_DEFOREST_PLANT
    pub frac_deforest_plant: FloatValue,

    /// Fraction of deforestation emissions from detritus pool
    /// unit: dimensionless
    /// MAGICC7: CO2_FRACTION_DEFOREST_DETRITUS
    pub frac_deforest_detritus: FloatValue,

    /// Fraction of deforestation that is permanent (no regrowth)
    /// unit: dimensionless
    /// MAGICC7: CO2_NORGRWTH_FRAC_DEFO
    pub norgrwth_frac_defo: FloatValue,
}

impl Default for TerrestrialCarbonParameters {
    fn default() -> Self {
        Self {
            npp_pi: 66.2716,
            co2_pi: 278.0,
            beta: 0.6485981,
            fertilization_method: 1.100486,
            fertilization_factor2: 100.0,
            gifford_conc_for_zero_npp: 80.0,
            fertilization_yrstart: 1900.0,

            npp_temp_sensitivity: 0.01070037,
            resp_temp_sensitivity: 0.06845893,
            detritus_temp_sensitivity: -0.1357817,
            soil_temp_sensitivity: 0.1540879,
            tempfeedback_yrstart: 1900.0,

            plant_pool_pi: 884.8584,
            detritus_pool_pi: 92.7738,
            soil_pool_pi: 1681.525,

            respiration_pi: 12.26025,
            plantbox_resp_method: 1,
            plantbox_resp_fertscale: 0.0,

            frac_npp_to_plant: 0.4482615,
            frac_npp_to_detritus: 0.3998165,
            frac_plant_to_detritus: 0.9989021,
            frac_detritus_to_soil: 0.00100763,

            frac_deforest_plant: 0.70,
            frac_deforest_detritus: 0.05,
            norgrwth_frac_defo: 0.5,
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

    /// Fraction of deforestation emissions from soil pool (derived).
    ///
    /// Clamps the sum of plant + detritus fractions to [0, 1] before deriving
    /// the soil fraction, preventing total deforestation fraction from exceeding 1.0.
    pub fn frac_deforest_soil(&self) -> FloatValue {
        let total = (self.frac_deforest_plant + self.frac_deforest_detritus).clamp(0.0, 1.0);
        1.0 - total
    }

    /// Net flux to plant pool at pre-industrial (NPP to plant - respiration).
    ///
    /// This is used to derive the initial turnover time for the plant pool.
    pub fn net_flux_to_plant_pi(&self) -> FloatValue {
        self.frac_npp_to_plant * self.npp_pi - self.respiration_pi
    }

    /// Initial turnover time for plant pool at pre-industrial (years).
    ///
    /// Derived from steady-state assumption: pool_size / net_flux.
    /// MAGICC7: INITIAL_TURNOVERTIME_PLANTPOOL
    pub fn tau_plant_pi(&self) -> FloatValue {
        let net_flux = self.net_flux_to_plant_pi();
        if net_flux > 1e-10 {
            self.plant_pool_pi / net_flux
        } else {
            100.0
        }
    }

    /// Initial turnover time for detritus pool at pre-industrial (years).
    ///
    /// MAGICC7: INI_TURNOVERTIME_DETRPOOL
    pub fn tau_detritus_pi(&self) -> FloatValue {
        let net_flux_plant = self.net_flux_to_plant_pi();
        let flux_into_detritus =
            self.frac_npp_to_detritus * self.npp_pi + self.frac_plant_to_detritus * net_flux_plant;
        if flux_into_detritus > 1e-10 {
            self.detritus_pool_pi / flux_into_detritus
        } else {
            3.0
        }
    }

    /// Flux into detritus at pre-industrial steady state (GtC/yr).
    pub fn flux_into_detritus_pi(&self) -> FloatValue {
        let net_flux_plant = self.net_flux_to_plant_pi();
        self.frac_npp_to_detritus * self.npp_pi + self.frac_plant_to_detritus * net_flux_plant
    }

    /// Initial turnover time for soil pool at pre-industrial (years).
    ///
    /// Derived from steady-state: uses the MAGICC7 formulation.
    /// MAGICC7: INITIAL_TURNOVERTIME_SOILPOOL
    pub fn tau_soil_pi(&self) -> FloatValue {
        let net_flux_plant = self.net_flux_to_plant_pi();
        let flux_detritus_out = self.detritus_pool_pi / self.tau_detritus_pi();

        let flux_into_soil = self.frac_npp_to_soil() * self.npp_pi
            + (1.0 - self.frac_plant_to_detritus) * net_flux_plant
            + self.frac_detritus_to_soil * flux_detritus_out;

        if flux_into_soil > 1e-10 {
            self.soil_pool_pi / flux_into_soil
        } else {
            50.0
        }
    }

    /// Flux into soil at pre-industrial steady state (GtC/yr).
    pub fn flux_into_soil_pi(&self) -> FloatValue {
        let net_flux_plant = self.net_flux_to_plant_pi();
        let flux_detritus_out = self.detritus_pool_pi / self.tau_detritus_pi();
        self.frac_npp_to_soil() * self.npp_pi
            + (1.0 - self.frac_plant_to_detritus) * net_flux_plant
            + self.frac_detritus_to_soil * flux_detritus_out
    }

    /// Total pre-industrial terrestrial carbon pool (GtC).
    pub fn total_pool_pi(&self) -> FloatValue {
        self.plant_pool_pi + self.detritus_pool_pi + self.soil_pool_pi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() {
        let params = TerrestrialCarbonParameters::default();
        assert!((params.npp_pi - 66.2716).abs() < 1e-10);
        assert!((params.co2_pi - 278.0).abs() < 1e-10);
        assert!((params.fertilization_method - 1.100486).abs() < 1e-10);
        assert_eq!(params.plantbox_resp_method, 1);
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
    fn test_deforest_fractions_sum_to_one() {
        let params = TerrestrialCarbonParameters::default();
        let sum = params.frac_deforest_plant
            + params.frac_deforest_detritus
            + params.frac_deforest_soil();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Deforestation fractions should sum to 1.0, got {}",
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

        assert!(tau_plant > 0.0, "Plant turnover time should be positive");
        assert!(
            tau_detritus > 0.0,
            "Detritus turnover time should be positive"
        );
        assert!(tau_soil > 0.0, "Soil turnover time should be positive");

        assert!(
            tau_detritus < tau_soil,
            "Detritus turnover should be faster than soil"
        );
    }

    #[test]
    fn test_total_pool_calculation() {
        let params = TerrestrialCarbonParameters::default();
        let total = params.total_pool_pi();
        let expected = params.plant_pool_pi + params.detritus_pool_pi + params.soil_pool_pi;
        assert!(
            (total - expected).abs() < 1e-10,
            "Total pool calculation incorrect"
        );
    }

    #[test]
    fn test_temperature_sensitivities_positive() {
        let params = TerrestrialCarbonParameters::default();
        assert!(
            params.resp_temp_sensitivity > 0.0,
            "Respiration temp sensitivity should be positive"
        );
        assert!(
            params.soil_temp_sensitivity > 0.0,
            "Soil temp sensitivity should be positive"
        );
    }

    #[test]
    fn test_beta_fertilization_reasonable() {
        let params = TerrestrialCarbonParameters::default();
        let doubled_co2_factor = 1.0 + params.beta * (2.0_f64).ln();
        assert!(
            (doubled_co2_factor - 1.45).abs() < 0.1,
            "Doubled CO2 should give ~45% NPP increase, got {:.1}%",
            (doubled_co2_factor - 1.0) * 100.0
        );
    }
}
