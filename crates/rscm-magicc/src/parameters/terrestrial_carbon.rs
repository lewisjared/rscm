//! Terrestrial Carbon Parameters
//!
//! Parameters for the MAGICC7 3-pool terrestrial carbon cycle model with
//! three CO2 fertilisation methods, two respiration methods, and temperature feedbacks.
//!
//! # Reference
//!
//! Based on MAGICC7's terrestrial carbon cycle, which implements a
//! 3-box model representing plant biomass, detritus, and soil pools.

use super::NitrogenParameters;
use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// CO2 fertilisation method selector.
///
/// MAGICC7 uses a continuous float (1.0-3.0) that blends adjacent methods.
/// This enum provides named variants for the three pure methods plus
/// a `Blended` variant for fractional values.
///
/// MAGICC7: CO2_FERTILIZATION_METHOD
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "FloatValue", into = "FloatValue")]
pub enum FertilisationMethod {
    /// No CO2 fertilisation (MAGICC7 value < 1.0)
    None,
    /// Logarithmic (Keeling-Bacastow 1973) (MAGICC7 value = 1.0)
    Logarithmic,
    /// Gifford rectangular hyperbolic / Michaelis-Menten (MAGICC7 value = 2.0)
    Gifford,
    /// Saturating sigmoid, Norton (MAGICC7 value = 3.0)
    Sigmoid,
    /// Blend between adjacent methods using MAGICC7 continuous parameter.
    /// Value in range [1.0, 3.0]: 1.0-2.0 blends log/Gifford, 2.0-3.0 blends Gifford/sigmoid.
    Blended(FloatValue),
}

impl FertilisationMethod {
    /// MAGICC7 continuous parameter value.
    pub fn as_f64(&self) -> FloatValue {
        match self {
            Self::None => 0.0,
            Self::Logarithmic => 1.0,
            Self::Gifford => 2.0,
            Self::Sigmoid => 3.0,
            Self::Blended(v) => *v,
        }
    }
}

impl From<FloatValue> for FertilisationMethod {
    fn from(v: FloatValue) -> Self {
        if v < 1.0 {
            Self::None
        } else if (v - 1.0).abs() < f64::EPSILON {
            Self::Logarithmic
        } else if (v - 2.0).abs() < f64::EPSILON {
            Self::Gifford
        } else if (v - 3.0).abs() < f64::EPSILON {
            Self::Sigmoid
        } else {
            Self::Blended(v.clamp(1.0, 3.0))
        }
    }
}

impl From<FertilisationMethod> for FloatValue {
    fn from(m: FertilisationMethod) -> Self {
        m.as_f64()
    }
}

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
/// 1. **CO2 Fertilisation**: Higher atmospheric CO2 enhances NPP via three
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

    /// Pre-industrial CO2 concentration used as reference for fertilisation
    /// unit: ppm
    /// MAGICC7: CO2_PREINDCO2CONC
    pub co2_pi: FloatValue,

    /// CO2 fertilisation factor (beta in logarithmic formula, also used by Gifford/sigmoid)
    /// unit: dimensionless
    /// MAGICC7: CO2_FERTILIZATION_FACTOR
    pub fertilisation_factor: FloatValue,

    /// CO2 fertilisation method.
    /// MAGICC7: CO2_FERTILIZATION_METHOD
    pub fertilisation_method: FertilisationMethod,

    /// Curvature parameter for sigmoid fertilisation method.
    /// unit: ppm
    /// MAGICC7: CO2_FERTILIZATION_FACTOR2
    pub fertilisation_factor2: FloatValue,

    /// CO2 concentration at which NPP would be zero (Gifford method).
    /// unit: ppm
    /// MAGICC7: CO2_GIFFORD_CONC_FOR_ZERONPP
    pub gifford_concentration_for_zero_npp: FloatValue,

    /// Year before which CO2 fertilisation reference tracks current CO2.
    /// unit: year
    /// MAGICC7: CO2_FERTILIZATION_YRSTART
    pub fertilisation_start_year: FloatValue,

    /// NPP temperature sensitivity coefficient (gamma_NPP)
    /// Positive means warming increases NPP.
    /// unit: K^-1
    /// MAGICC7: CO2_FEEDBACKFACTOR_NPP
    pub npp_temperature_sensitivity: FloatValue,

    /// Respiration temperature sensitivity coefficient (gamma_resp)
    /// Controls how plant pool respiration responds to temperature.
    /// unit: K^-1
    /// MAGICC7: CO2_FEEDBACKFACTOR_RESPIRATION
    pub respiration_temperature_sensitivity: FloatValue,

    /// Detritus decay temperature sensitivity coefficient (gamma_detritus)
    /// Positive means warming increases decay rate.
    /// unit: K^-1
    /// MAGICC7: CO2_FEEDBACKFACTOR_DETRITUS
    pub detritus_temperature_sensitivity: FloatValue,

    /// Soil decay temperature sensitivity coefficient (gamma_soil)
    /// Positive means warming increases decay rate.
    /// unit: K^-1
    /// MAGICC7: CO2_FEEDBACKFACTOR_SOIL
    pub soil_temperature_sensitivity: FloatValue,

    /// Year from which warming drives the temperature feedbacks.
    /// Feedbacks use the warming since this year, floored at zero.
    /// unit: year
    /// MAGICC7: CO2_TEMPFEEDBACK_YRSTART
    pub temperature_feedback_start_year: FloatValue,

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

    /// Scaling of fertilisation effect on plant respiration.
    /// R_h = R_h0 * (1 + alpha*(beta-1)) * min(1, C_P/C_P0) * f_T_resp
    /// unit: dimensionless
    /// MAGICC7: CO2_PLANTBOXRESP_FERTSCALE
    pub plant_respiration_fertilisation_scale: FloatValue,

    /// Fraction of NPP going directly to plant pool
    /// unit: dimensionless
    /// MAGICC7: CO2_FRACTION_NPP_2_PLANT
    pub npp_to_plant_fraction: FloatValue,

    /// Fraction of NPP going directly to detritus pool
    /// unit: dimensionless
    /// MAGICC7: CO2_FRACTION_NPP_2_DETRITUS
    pub npp_to_detritus_fraction: FloatValue,

    /// Fraction of plant turnover flux going to detritus (vs soil)
    /// unit: dimensionless
    /// MAGICC7: CO2_FRACTION_PLANT_2_DETRITUS
    pub plant_to_detritus_fraction: FloatValue,

    /// Fraction of detritus decay going to soil (vs atmosphere)
    /// unit: dimensionless
    /// MAGICC7: CO2_FRACTION_DETRITUS_2_SOIL
    pub detritus_to_soil_fraction: FloatValue,

    /// Fraction of deforestation emissions from plant pool
    /// unit: dimensionless
    /// MAGICC7: CO2_FRACTION_DEFOREST_PLANT
    pub deforestation_plant_fraction: FloatValue,

    /// Fraction of deforestation emissions from detritus pool
    /// unit: dimensionless
    /// MAGICC7: CO2_FRACTION_DEFOREST_DETRITUS
    pub deforestation_detritus_fraction: FloatValue,

    /// Fraction of deforestation that is permanent (no regrowth)
    /// unit: dimensionless
    /// MAGICC7: CO2_NORGRWTH_FRAC_DEFO
    pub deforestation_no_regrowth_fraction: FloatValue,

    /// Nitrogen limitation of NPP, off when `None`
    /// MAGICC7: NCYCLE_APPLY
    pub nitrogen: Option<NitrogenParameters>,
}

impl Default for TerrestrialCarbonParameters {
    fn default() -> Self {
        Self {
            npp_pi: 66.2716,
            co2_pi: 278.0,
            fertilisation_factor: 0.6485981,
            fertilisation_method: FertilisationMethod::Blended(1.100486),
            fertilisation_factor2: 100.0,
            gifford_concentration_for_zero_npp: 80.0,
            fertilisation_start_year: 1900.0,

            npp_temperature_sensitivity: 0.01070037,
            respiration_temperature_sensitivity: 0.06845893,
            detritus_temperature_sensitivity: -0.1357817,
            soil_temperature_sensitivity: 0.1540879,
            temperature_feedback_start_year: 1900.0,

            plant_pool_pi: 884.8584,
            detritus_pool_pi: 92.7738,
            soil_pool_pi: 1681.525,

            respiration_pi: 12.26025,
            plant_respiration_fertilisation_scale: 0.95,

            npp_to_plant_fraction: 0.4482615,
            npp_to_detritus_fraction: 0.3998165,
            plant_to_detritus_fraction: 0.9989021,
            detritus_to_soil_fraction: 0.00100763,

            deforestation_plant_fraction: 0.70,
            deforestation_detritus_fraction: 0.05,
            deforestation_no_regrowth_fraction: 0.5,
            nitrogen: None,
        }
    }
}

impl TerrestrialCarbonParameters {
    /// Fraction of NPP going directly to soil pool (derived).
    ///
    /// Calculated as `1 - npp_to_plant_fraction - npp_to_detritus_fraction`.
    pub fn npp_to_soil_fraction(&self) -> FloatValue {
        let f = 1.0 - self.npp_to_plant_fraction - self.npp_to_detritus_fraction;
        f.max(0.0)
    }

    /// Fraction of deforestation emissions from soil pool (derived).
    ///
    /// Validates that plant + detritus fractions do not exceed 1.0 and derives
    /// the soil fraction as the residual.
    pub fn deforestation_soil_fraction(&self) -> FloatValue {
        let total = self.deforestation_plant_fraction + self.deforestation_detritus_fraction;
        assert!(
            total <= 1.0 + 1e-9,
            "deforestation_plant_fraction ({}) + deforestation_detritus_fraction ({}) must not exceed 1.0, got {}",
            self.deforestation_plant_fraction,
            self.deforestation_detritus_fraction,
            total
        );
        (1.0 - self.deforestation_plant_fraction - self.deforestation_detritus_fraction).max(0.0)
    }

    /// Net flux to plant pool at pre-industrial (NPP to plant - respiration).
    ///
    /// This is used to derive the initial turnover time for the plant pool.
    pub fn net_flux_to_plant_pi(&self) -> FloatValue {
        self.npp_to_plant_fraction * self.npp_pi - self.respiration_pi
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
        let flux_into_detritus = self.npp_to_detritus_fraction * self.npp_pi
            + self.plant_to_detritus_fraction * net_flux_plant;
        if flux_into_detritus > 1e-10 {
            self.detritus_pool_pi / flux_into_detritus
        } else {
            3.0
        }
    }

    /// Flux into detritus at pre-industrial steady state (GtC/yr).
    pub fn flux_into_detritus_pi(&self) -> FloatValue {
        let net_flux_plant = self.net_flux_to_plant_pi();
        self.npp_to_detritus_fraction * self.npp_pi
            + self.plant_to_detritus_fraction * net_flux_plant
    }

    /// Initial turnover time for soil pool at pre-industrial (years).
    ///
    /// Derived from steady-state: uses the MAGICC7 formulation.
    /// MAGICC7: INITIAL_TURNOVERTIME_SOILPOOL
    pub fn tau_soil_pi(&self) -> FloatValue {
        let flux_into_soil = self.flux_into_soil_pi();
        if flux_into_soil > 1e-10 {
            self.soil_pool_pi / flux_into_soil
        } else {
            50.0
        }
    }

    /// Flux into soil at pre-industrial steady state (GtC/yr).
    pub fn flux_into_soil_pi(&self) -> FloatValue {
        self.flux_into_soil(self.detritus_pool_pi, self.tau_detritus_pi())
    }

    /// Steady-state flux into soil for a detritus pool and turnover time (GtC/yr).
    ///
    /// NPP less plant respiration and detritus respiration, in MAGICC7's order of operations.
    pub fn flux_into_soil(
        &self,
        detritus_pool: FloatValue,
        tau_detritus: FloatValue,
    ) -> FloatValue {
        (self.npp_pi - self.respiration_pi)
            - (1.0 - self.detritus_to_soil_fraction) * detritus_pool / tau_detritus
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
        assert!((params.fertilisation_method.as_f64() - 1.100486).abs() < 1e-10);
        assert!((params.plant_respiration_fertilisation_scale - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_npp_fractions_sum_to_one() {
        let params = TerrestrialCarbonParameters::default();
        let sum = params.npp_to_plant_fraction
            + params.npp_to_detritus_fraction
            + params.npp_to_soil_fraction();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "NPP fractions should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_deforest_fractions_sum_to_one() {
        let params = TerrestrialCarbonParameters::default();
        let sum = params.deforestation_plant_fraction
            + params.deforestation_detritus_fraction
            + params.deforestation_soil_fraction();
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
            params.respiration_temperature_sensitivity > 0.0,
            "Respiration temp sensitivity should be positive"
        );
        assert!(
            params.soil_temperature_sensitivity > 0.0,
            "Soil temp sensitivity should be positive"
        );
    }

    #[test]
    fn test_beta_fertilisation_reasonable() {
        let params = TerrestrialCarbonParameters::default();
        let doubled_co2_factor = 1.0 + params.fertilisation_factor * (2.0_f64).ln();
        assert!(
            (doubled_co2_factor - 1.45).abs() < 0.1,
            "Doubled CO2 should give ~45% NPP increase, got {:.1}%",
            (doubled_co2_factor - 1.0) * 100.0
        );
    }
}
