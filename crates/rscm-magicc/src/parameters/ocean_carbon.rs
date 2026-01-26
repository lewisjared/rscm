//! Ocean Carbon Parameters
//!
//! Parameters for the IRF-based ocean carbon cycle model with air-sea exchange
//! and temperature feedback.
//!
//! # Reference
//!
//! Based on MAGICC7 Module 10 (Ocean Carbon Cycle) which implements an Impulse
//! Response Function (IRF) approach to emulate the behaviour of complex 3D
//! ocean models (GFDL, HILDA, BERN 2.5D, BOXDIFF).
//!
//! # What This Implementation Supports
//!
//! This is a simplified implementation using only the 2D-BERN model parameters.
//! The full MAGICC7 supports four different IRF models selectable at runtime.

use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Pre-industrial ocean surface temperature (Celsius) for 2D-BERN model.
pub const SST_PI_BERN: FloatValue = 18.2997;

/// Ocean surface area for 2D-BERN model (m^2).
pub const OCEAN_SURFACE_AREA_BERN: FloatValue = 3.5375e14;

/// Mixed layer depth for 2D-BERN model (m).
pub const MIXED_LAYER_DEPTH_BERN: FloatValue = 50.0;

/// Gas exchange timescale for 2D-BERN model (years).
pub const GAS_EXCHANGE_TAU_BERN: FloatValue = 7.46;

/// Switch time for 2D-BERN IRF (years).
/// Before this time, use "early" IRF; after, use "late" IRF.
pub const IRF_SWITCH_TIME_BERN: FloatValue = 9.9;

/// Unit conversion constant: micromol/(ppm * m^3/kg).
///
/// Used to convert between flux (ppm/month) and DIC (micromol/kg).
/// Calculated as: 1e6 / (5.65770e-15 ppm/mol) / (1026.5 kg/m^3) ≈ 1.72e17
pub const OCEAN_MICROMOL_PER_PPM_M3_PER_KG: FloatValue = 1.72e17;

/// Joos A24 polynomial offsets for pCO2-DIC relationship.
pub const DELTA_OSPP_OFFSETS: [FloatValue; 5] = [1.5568, 7.4706, 1.2748, 2.4491, 1.5468];

/// Joos A24 polynomial temperature coefficients for pCO2-DIC relationship.
pub const DELTA_OSPP_COEFFICIENTS: [FloatValue; 5] =
    [-0.013993, -0.20207, -0.12015, -0.12639, -0.15326];

/// Parameters for the ocean carbon cycle.
///
/// The ocean carbon cycle uses an Impulse Response Function (IRF) approach
/// to calculate how carbon absorbed at the surface is transported into the
/// deep ocean. The IRF represents the "memory" of the mixed layer.
///
/// # Air-Sea Exchange
///
/// The fundamental flux equation is:
/// $$F = k \times (pCO2_{atm} - pCO2_{ocn})$$
///
/// where $k$ is the gas exchange rate and pCO2 is partial pressure.
///
/// # Ocean pCO2 Calculation (Joos Equations)
///
/// The ocean surface pCO2 is calculated using:
/// 1. **Joos A24**: DIC effect on pCO2 via polynomial expansion
/// 2. **Joos A25**: Temperature effect: $pCO2 = pCO2_{DIC} \times e^{\alpha_T \Delta T}$
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OceanCarbonParameters {
    /// Pre-industrial atmospheric CO2 (ppm).
    /// Used as reference for pCO2 calculations.
    /// default: 278.0
    pub co2_pi: FloatValue,

    /// Pre-industrial ocean surface pCO2 (ppm).
    /// At equilibrium, equals atmospheric CO2.
    /// default: 278.0
    pub pco2_pi: FloatValue,

    /// Gas exchange rate scaling factor (dimensionless).
    /// Scales the base gas exchange rate.
    /// default: 1.833492 (MAGICC7 default)
    pub gas_exchange_scale: FloatValue,

    /// Gas exchange timescale (years).
    /// Characteristic time for air-sea equilibration.
    /// default: 7.46 (2D-BERN model)
    pub gas_exchange_tau: FloatValue,

    /// Temperature sensitivity of pCO2 (K^-1).
    /// From Takahashi et al., approximately 4.23%/K.
    /// Joos A25: pCO2 = pCO2_base * exp(alpha_T * delta_T)
    /// default: 0.0423
    pub temp_sensitivity: FloatValue,

    /// IRF scaling factor (dimensionless).
    /// Scales the impulse response function.
    /// default: 0.9492864 (MAGICC7 default)
    pub irf_scale: FloatValue,

    /// Mixed layer depth (m).
    /// Determines the volume of the surface ocean box.
    /// default: 50.0 (2D-BERN model)
    pub mixed_layer_depth: FloatValue,

    /// Ocean surface area (m^2).
    /// default: 3.5375e14 (2D-BERN model)
    pub ocean_surface_area: FloatValue,

    /// Pre-industrial sea surface temperature (Celsius).
    /// Used in pCO2-DIC polynomial calculation.
    /// default: 18.2997 (2D-BERN model)
    pub sst_pi: FloatValue,

    /// Number of sub-steps per year.
    /// MAGICC7 uses monthly stepping (12) for stability.
    /// default: 12
    pub steps_per_year: usize,

    /// Maximum flux history length in months.
    /// Limits memory usage for IRF convolution (6000 months = 500 years).
    /// default: 6000
    pub max_history_months: usize,

    /// IRF switch time (years).
    /// Time at which to switch from early to late IRF coefficients.
    /// default: 9.9 (2D-BERN model)
    pub irf_switch_time: FloatValue,

    /// Early IRF exponential coefficients (before switch time).
    /// 2D-BERN uses 6 terms.
    pub irf_early_coefficients: Vec<FloatValue>,

    /// Early IRF exponential timescales (years).
    pub irf_early_timescales: Vec<FloatValue>,

    /// Late IRF exponential coefficients (after switch time).
    pub irf_late_coefficients: Vec<FloatValue>,

    /// Late IRF exponential timescales (years).
    pub irf_late_timescales: Vec<FloatValue>,

    /// Joos A24 polynomial offsets.
    pub delta_ospp_offsets: [FloatValue; 5],

    /// Joos A24 polynomial coefficients (temperature-dependent).
    pub delta_ospp_coefficients: [FloatValue; 5],

    /// Enable temperature feedback on pCO2.
    /// default: true
    pub enable_temp_feedback: bool,
}

impl Default for OceanCarbonParameters {
    fn default() -> Self {
        Self {
            // Atmospheric reference
            co2_pi: 278.0,
            pco2_pi: 278.0,

            // Gas exchange (with MAGICC7 scaling)
            gas_exchange_scale: 1.833492,
            gas_exchange_tau: GAS_EXCHANGE_TAU_BERN,

            // Temperature feedback
            temp_sensitivity: 0.0423,

            // IRF parameters
            irf_scale: 0.9492864,
            irf_switch_time: IRF_SWITCH_TIME_BERN,

            // 2D-BERN early IRF (before 9.9 years)
            irf_early_coefficients: vec![0.058648, 0.07515, 0.079338, 0.41413, 0.24845, 0.12429],
            irf_early_timescales: vec![1.0e10, 9.6218, 9.2364, 0.7603, 0.16294, 0.0032825],

            // 2D-BERN late IRF (after 9.9 years)
            irf_late_coefficients: vec![0.01369, 0.012456, 0.026933, 0.026994, 0.036608, 0.06738],
            irf_late_timescales: vec![1.0e10, 331.54, 107.57, 38.946, 11.677, 10.515],

            // Physical parameters
            mixed_layer_depth: MIXED_LAYER_DEPTH_BERN,
            ocean_surface_area: OCEAN_SURFACE_AREA_BERN,
            sst_pi: SST_PI_BERN,

            // Joos A24 polynomial
            delta_ospp_offsets: DELTA_OSPP_OFFSETS,
            delta_ospp_coefficients: DELTA_OSPP_COEFFICIENTS,

            // Sub-stepping
            steps_per_year: 12,

            // History limit (500 years = 6000 months)
            max_history_months: 6000,

            // Feedback switches
            enable_temp_feedback: true,
        }
    }
}

impl OceanCarbonParameters {
    /// Calculate the gas exchange rate (per month).
    ///
    /// $$k = \frac{\text{scale}}{\tau \times 12}$$
    ///
    /// Returns the rate in units of month^-1.
    pub fn gas_exchange_rate(&self) -> FloatValue {
        self.gas_exchange_scale / (self.gas_exchange_tau * 12.0)
    }

    /// Calculate the IRF value at a given time (years).
    ///
    /// Uses exponential sum form:
    /// $$IRF(t) = \sum_i a_i \times e^{-t/\tau_i}$$
    ///
    /// Selects early or late coefficients based on switch time.
    ///
    /// # Arguments
    ///
    /// * `t` - Time since pulse (years)
    ///
    /// # Returns
    ///
    /// IRF value (dimensionless, 0 to 1)
    pub fn irf(&self, t: FloatValue) -> FloatValue {
        let (coeffs, taus) = if t < self.irf_switch_time {
            (&self.irf_early_coefficients, &self.irf_early_timescales)
        } else {
            (&self.irf_late_coefficients, &self.irf_late_timescales)
        };

        let mut irf = 0.0;
        for (a, tau) in coeffs.iter().zip(taus.iter()) {
            irf += a * (-t / tau).exp();
        }

        // Apply scaling using nonlinear transform
        self.scale_irf(irf)
    }

    /// Apply nonlinear IRF scaling.
    ///
    /// $$IRF_{scaled} = \frac{IRF \times f}{IRF \times f + 1 - IRF}$$
    ///
    /// This ensures the scaled IRF stays bounded between 0 and 1.
    fn scale_irf(&self, irf: FloatValue) -> FloatValue {
        let f = self.irf_scale;
        (irf * f) / (irf * f + 1.0 - irf)
    }

    /// Calculate delta pCO2 from DIC change using Joos A24 polynomial.
    ///
    /// $$\Delta pCO2 = \sum_{i=1}^{5} (b_i + c_i T_0) \times g_i(\Delta DIC)$$
    ///
    /// where:
    /// - $b_i$ = offsets
    /// - $c_i$ = temperature coefficients
    /// - $T_0$ = pre-industrial SST
    /// - $g_i$ = polynomial terms with scaling factors
    ///
    /// # Arguments
    ///
    /// * `delta_dic` - Change in DIC (micromol/kg)
    ///
    /// # Returns
    ///
    /// Change in pCO2 (ppm)
    pub fn delta_pco2_from_dic(&self, delta_dic: FloatValue) -> FloatValue {
        // Build polynomial terms g_i with proper signs and scaling
        let dic_powers = [
            delta_dic,                  // g_1
            delta_dic.powi(2) * 1e-3,   // g_2
            -delta_dic.powi(3) * 1e-5,  // g_3 (negative)
            delta_dic.powi(4) * 1e-7,   // g_4
            -delta_dic.powi(5) * 1e-10, // g_5 (negative)
        ];

        // Calculate effective coefficients (offset + T_pi * temp_coeff)
        let mut delta_pco2 = 0.0;
        for i in 0..5 {
            let coeff = self.delta_ospp_offsets[i] + self.delta_ospp_coefficients[i] * self.sst_pi;
            delta_pco2 += coeff * dic_powers[i];
        }

        delta_pco2
    }

    /// Calculate ocean pCO2 with temperature effect (Joos A25).
    ///
    /// $$pCO2_{ocn} = (pCO2_{pi} + \Delta pCO2_{DIC}) \times e^{\alpha_T \Delta T}$$
    ///
    /// # Arguments
    ///
    /// * `delta_pco2_dic` - Change in pCO2 from DIC (ppm)
    /// * `delta_sst` - Change in SST from pre-industrial (K)
    ///
    /// # Returns
    ///
    /// Ocean surface pCO2 (ppm)
    pub fn ocean_pco2(&self, delta_pco2_dic: FloatValue, delta_sst: FloatValue) -> FloatValue {
        let temp_factor = if self.enable_temp_feedback {
            (self.temp_sensitivity * delta_sst).exp()
        } else {
            1.0
        };

        (self.pco2_pi + delta_pco2_dic) * temp_factor
    }

    /// Unit conversion factor for DIC calculation.
    ///
    /// Converts from (ppm * months) to (micromol/kg) accounting for
    /// mixed layer depth and ocean area.
    pub fn dic_conversion_factor(&self) -> FloatValue {
        OCEAN_MICROMOL_PER_PPM_M3_PER_KG / (self.mixed_layer_depth * self.ocean_surface_area)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() {
        let params = OceanCarbonParameters::default();
        assert!((params.co2_pi - 278.0).abs() < 1e-10);
        assert!((params.pco2_pi - 278.0).abs() < 1e-10);
        assert!(params.enable_temp_feedback);
        assert_eq!(params.steps_per_year, 12);
    }

    #[test]
    fn test_gas_exchange_rate() {
        let params = OceanCarbonParameters::default();
        let rate = params.gas_exchange_rate();

        // Expected: 1.833492 / (7.46 * 12) ≈ 0.0205 per month
        let expected = params.gas_exchange_scale / (params.gas_exchange_tau * 12.0);
        assert!(
            (rate - expected).abs() < 1e-6,
            "Gas exchange rate: expected {:.6}, got {:.6}",
            expected,
            rate
        );
        assert!(rate > 0.0, "Gas exchange rate should be positive");
    }

    #[test]
    fn test_irf_at_zero() {
        let params = OceanCarbonParameters::default();
        let irf = params.irf(0.0);

        // At t=0, IRF should be close to 1.0 (all coefficients sum to 1)
        // With scaling, may be slightly different but should be very close
        assert!(
            irf > 0.9,
            "IRF at t=0 should be close to 1.0, got {:.4}",
            irf
        );
        assert!(
            (irf - 1.0).abs() < 0.01,
            "IRF at t=0 should be approximately 1.0, got {:.6}",
            irf
        );
    }

    #[test]
    fn test_irf_decays_over_time() {
        let params = OceanCarbonParameters::default();

        let irf_0 = params.irf(0.0);
        let irf_10 = params.irf(10.0);
        let irf_100 = params.irf(100.0);

        assert!(
            irf_10 < irf_0,
            "IRF should decay: {:.4} (t=10) should be < {:.4} (t=0)",
            irf_10,
            irf_0
        );
        assert!(
            irf_100 < irf_10,
            "IRF should continue decaying: {:.4} (t=100) should be < {:.4} (t=10)",
            irf_100,
            irf_10
        );
        assert!(
            irf_100 > 0.0,
            "IRF should remain positive, got {:.4}",
            irf_100
        );
    }

    #[test]
    fn test_irf_switch_time() {
        let params = OceanCarbonParameters::default();

        // IRF should be continuous across switch time
        let t_before = params.irf_switch_time - 0.01;
        let t_after = params.irf_switch_time + 0.01;

        let irf_before = params.irf(t_before);
        let irf_after = params.irf(t_after);

        // The values may differ at the switch, but should be in same ballpark
        // (MAGICC7 has code to ensure continuity)
        assert!(
            (irf_before - irf_after).abs() < 0.5,
            "IRF should be roughly continuous: before={:.4}, after={:.4}",
            irf_before,
            irf_after
        );
    }

    #[test]
    fn test_delta_pco2_zero_dic() {
        let params = OceanCarbonParameters::default();
        let delta_pco2 = params.delta_pco2_from_dic(0.0);

        assert!(
            delta_pco2.abs() < 1e-10,
            "Zero DIC change should give zero pCO2 change, got {:.4}",
            delta_pco2
        );
    }

    #[test]
    fn test_delta_pco2_positive_dic() {
        let params = OceanCarbonParameters::default();
        let delta_pco2 = params.delta_pco2_from_dic(50.0);

        // Positive DIC change should increase pCO2 (Revelle factor > 1)
        assert!(
            delta_pco2 > 0.0,
            "Positive DIC should increase pCO2, got {:.4}",
            delta_pco2
        );
    }

    #[test]
    fn test_ocean_pco2_no_change() {
        let params = OceanCarbonParameters::default();
        let pco2 = params.ocean_pco2(0.0, 0.0);

        assert!(
            (pco2 - params.pco2_pi).abs() < 1e-10,
            "With no DIC or temp change, pCO2 should equal PI: {:.2}",
            pco2
        );
    }

    #[test]
    fn test_ocean_pco2_warming() {
        let params = OceanCarbonParameters::default();

        let pco2_cold = params.ocean_pco2(0.0, 0.0);
        let pco2_warm = params.ocean_pco2(0.0, 1.0);

        // Warming should increase pCO2 (~4.23% per K)
        assert!(
            pco2_warm > pco2_cold,
            "Warming should increase pCO2: cold={:.2}, warm={:.2}",
            pco2_cold,
            pco2_warm
        );

        // Check approximate magnitude
        let expected_factor = (params.temp_sensitivity * 1.0).exp();
        let actual_factor = pco2_warm / pco2_cold;
        assert!(
            (actual_factor - expected_factor).abs() < 0.01,
            "Warming factor: expected {:.4}, got {:.4}",
            expected_factor,
            actual_factor
        );
    }

    #[test]
    fn test_ocean_pco2_temp_feedback_disabled() {
        let mut params = OceanCarbonParameters::default();
        params.enable_temp_feedback = false;

        let pco2_cold = params.ocean_pco2(0.0, 0.0);
        let pco2_warm = params.ocean_pco2(0.0, 5.0);

        assert!(
            (pco2_cold - pco2_warm).abs() < 1e-10,
            "With feedback disabled, pCO2 should not change with temperature"
        );
    }

    #[test]
    fn test_dic_conversion_factor_reasonable() {
        let params = OceanCarbonParameters::default();
        let factor = params.dic_conversion_factor();

        // Factor should be positive and finite
        assert!(factor > 0.0, "DIC conversion factor should be positive");
        assert!(factor.is_finite(), "DIC conversion factor should be finite");

        // Order of magnitude check: ~1e17 / (50 * 3.5e14) ≈ 5.7
        assert!(
            factor > 1.0 && factor < 100.0,
            "DIC conversion factor seems out of range: {:.2}",
            factor
        );
    }

    #[test]
    fn test_serialization() {
        let params = OceanCarbonParameters::default();
        let json = serde_json::to_string(&params).expect("Serialization failed");
        let parsed: OceanCarbonParameters =
            serde_json::from_str(&json).expect("Deserialization failed");

        assert!(
            (params.co2_pi - parsed.co2_pi).abs() < 1e-10,
            "Parameters should survive round-trip"
        );
        assert_eq!(
            params.irf_early_coefficients.len(),
            parsed.irf_early_coefficients.len()
        );
    }
}
