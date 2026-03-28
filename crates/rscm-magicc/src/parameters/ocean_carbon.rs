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
//! # Ocean Model Selection
//!
//! MAGICC7 supports four ocean carbon IRF models with runtime selection via
//! `OCEANCC_MODEL`. Each model has different IRF functional forms, physical
//! parameters, and characteristic timescales. The default is 3D-GFDL.
//!
//! The IRF form varies by model and time regime:
//! - **3D-GFDL**: Polynomial (pre-switch, first year) + exponential sum (post-switch)
//! - **2D-BERN**: Exponential sum for both regimes (switch at 9.9 years)
//! - **HILDA**: Exponential sum for both regimes (switch at 2.0 years)
//!
//! See `docs/modules/module_10_ocean_carbon.md` for full specification.

use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Unit conversion constant: micromol/(ppm * m^3/kg).
///
/// Used to convert between flux (ppm/month) and DIC (micromol/kg).
/// Calculated as: 1e6 / (5.65770e-15 ppm/mol) / (1026.5 kg/m^3)
pub const OCEAN_MICROMOL_PER_PPM_M3_PER_KG: FloatValue = 1.72e17;

/// Joos A24 polynomial offsets for pCO2-DIC relationship.
pub const DELTA_OSPP_OFFSETS: [FloatValue; 5] = [1.5568, 7.4706, 1.2748, 2.4491, 1.5468];

/// Joos A24 polynomial temperature coefficients for pCO2-DIC relationship.
pub const DELTA_OSPP_COEFFICIENTS: [FloatValue; 5] =
    [-0.013993, -0.20207, -0.12015, -0.12639, -0.15326];

/// Ocean carbon IRF model selection.
///
/// Controls which impulse response function and physical parameters are used
/// for the ocean carbon cycle. Corresponds to MAGICC7's `OCEANCC_MODEL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OceanCarbonModel {
    /// 3D-GFDL model (MAGICC7 default).
    ///
    /// Uses polynomial IRF for the first year, exponential sum thereafter.
    /// Physical parameters: h=50.9m, A=3.55e14 m^2, SST_pi=17.7C.
    #[serde(rename = "3D-GFDL")]
    GFDL3D,

    /// 2D-BERN model.
    ///
    /// Uses exponential sum IRF for both early and late regimes.
    /// Physical parameters: h=50.0m, A=3.5375e14 m^2, SST_pi=18.2997C.
    #[serde(rename = "2D-BERN")]
    BERN2D,

    /// HILDA model.
    ///
    /// Uses exponential sum IRF for both early and late regimes.
    /// Physical parameters: h=75.0m, A=3.62e14 m^2, SST_pi=18.1716C.
    #[serde(rename = "HILDA")]
    HILDA,
}

/// Impulse Response Function form.
///
/// Represents the mathematical form of the IRF used to calculate how a pulse
/// of carbon in the mixed layer decays over time. Different ocean models use
/// different functional forms for different time regimes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum IrfForm {
    /// Polynomial: $IRF(t) = \sum_i c_i \cdot t^i$ where $t$ is in years.
    ///
    /// Used by 3D-GFDL for the pre-switch regime (first year).
    Polynomial {
        /// Polynomial coefficients in ascending order of power: [c_0, c_1, ..., c_n].
        coefficients: Vec<FloatValue>,
    },

    /// Exponential sum: $IRF(t) = \sum_i a_i \cdot \exp(-t / \tau_i)$.
    ///
    /// Used by all models for the post-switch regime, and by 2D-BERN/HILDA/BOXDIFF
    /// for the pre-switch regime.
    ExponentialSum {
        /// Amplitude coefficients $a_i$.
        coefficients: Vec<FloatValue>,
        /// Characteristic timescales $\tau_i$ in years.
        timescales: Vec<FloatValue>,
    },
}

impl IrfForm {
    /// Evaluate the IRF at time `t` (years).
    pub fn evaluate(&self, t: FloatValue) -> FloatValue {
        match self {
            IrfForm::Polynomial { coefficients } => {
                // Horner's method for numerical stability
                let mut result = 0.0;
                for &c in coefficients.iter().rev() {
                    result = result * t + c;
                }
                result
            }
            IrfForm::ExponentialSum {
                coefficients,
                timescales,
            } => {
                debug_assert_eq!(
                    coefficients.len(),
                    timescales.len(),
                    "ExponentialSum coefficients ({}) and timescales ({}) must have equal length",
                    coefficients.len(),
                    timescales.len()
                );
                let mut sum = 0.0;
                for (a, tau) in coefficients.iter().zip(timescales.iter()) {
                    sum += a * (-t / tau).exp();
                }
                sum
            }
        }
    }
}

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
#[serde(default)]
pub struct OceanCarbonParameters {
    /// Ocean carbon model selection.
    ///
    /// This field is informational only; it does not automatically select the
    /// corresponding IRF coefficients or physical parameters. Use the named
    /// constructors (`gfdl_3d()`, `bern_2d()`, `hilda()`) to get a
    /// self-consistent parameter set for a given model. Setting only this
    /// field via serde partial deserialization will leave the IRF parameters
    /// at their default (GFDL3D) values regardless of the variant chosen.
    ///
    /// default: GFDL3D (matches MAGICC7 OCEANCC_MODEL = "3D-GFDL")
    pub model: OceanCarbonModel,

    /// Pre-industrial atmospheric CO2 (ppm).
    /// default: 278.0
    pub co2_pi: FloatValue,

    /// Pre-industrial ocean surface pCO2 (ppm).
    /// At equilibrium, equals atmospheric CO2.
    /// default: 278.0
    pub pco2_pi: FloatValue,

    /// Gas exchange rate scaling factor (dimensionless).
    /// Corresponds to MAGICC7 OCEANCC_SCALE_GASXCHANGE.
    /// default: 1.833492
    pub gas_exchange_scale: FloatValue,

    /// Gas exchange timescale (years).
    /// Model-dependent characteristic time for air-sea equilibration.
    pub gas_exchange_tau: FloatValue,

    /// Temperature sensitivity of pCO2 (K^-1).
    /// Joos A25 exponent. Corresponds to MAGICC7 OCEANCC_TEMPFEEDBACK.
    /// default: 0.03717879
    pub temp_sensitivity: FloatValue,

    /// IRF scaling factor (dimensionless).
    /// Corresponds to MAGICC7 OCEANCC_SCALE_IMPULSERESPONSE.
    /// default: 0.9492864
    pub irf_scale: FloatValue,

    /// Mixed layer depth (m). Model-dependent.
    pub mixed_layer_depth: FloatValue,

    /// Ocean surface area (m^2). Model-dependent.
    pub ocean_surface_area: FloatValue,

    /// Pre-industrial sea surface temperature (Celsius). Model-dependent.
    /// Used in pCO2-DIC polynomial calculation (Joos A24).
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
    /// Time at which to switch from early to late IRF form.
    pub irf_switch_time: FloatValue,

    /// IRF form for the early regime (before switch time).
    pub irf_early: IrfForm,

    /// IRF form for the late regime (after switch time).
    pub irf_late: IrfForm,

    /// Joos A24 polynomial offsets.
    pub delta_ospp_offsets: [FloatValue; 5],

    /// Joos A24 polynomial coefficients (temperature-dependent).
    pub delta_ospp_coefficients: [FloatValue; 5],

    /// Enable temperature feedback on pCO2.
    /// default: true
    pub enable_temp_feedback: bool,
}

impl OceanCarbonParameters {
    /// Construct parameters for the 3D-GFDL ocean carbon model (MAGICC7 default).
    ///
    /// 3D-GFDL uses a polynomial IRF for the first year and exponential sum
    /// thereafter. Physical parameters from MAGICC7 Fortran source
    /// (`carbon_cycle_ocean.f90` lines 124-132).
    pub fn gfdl_3d() -> Self {
        Self {
            model: OceanCarbonModel::GFDL3D,

            co2_pi: 278.0,
            pco2_pi: 278.0,

            gas_exchange_scale: 1.833492,
            gas_exchange_tau: 7.66,

            temp_sensitivity: 0.03717879,

            irf_scale: 0.9492864,
            irf_switch_time: 1.0,

            // Polynomial IRF for first year (MAGICC7 lines 413-414)
            irf_early: IrfForm::Polynomial {
                coefficients: vec![1.0, -2.2617, 14.002, -48.770, 82.986, -67.527, 21.037],
            },

            // Exponential sum IRF after first year (MAGICC7 lines 441-446)
            irf_late: IrfForm::ExponentialSum {
                coefficients: vec![0.01481, 0.019439, 0.038344, 0.066485, 0.24966, 0.70367],
                timescales: vec![1.0e10, 347.55, 65.359, 15.281, 2.3488, 0.70177],
            },

            mixed_layer_depth: 50.9,
            ocean_surface_area: 3.55e14,
            sst_pi: 17.7,

            delta_ospp_offsets: DELTA_OSPP_OFFSETS,
            delta_ospp_coefficients: DELTA_OSPP_COEFFICIENTS,

            steps_per_year: 12,
            max_history_months: 6000,
            enable_temp_feedback: true,
        }
    }

    /// Construct parameters for the 2D-BERN ocean carbon model.
    ///
    /// 2D-BERN uses exponential sum IRF for both early and late regimes.
    /// Physical parameters from MAGICC7 Fortran source
    /// (`carbon_cycle_ocean.f90` lines 134-142).
    pub fn bern_2d() -> Self {
        Self {
            model: OceanCarbonModel::BERN2D,

            co2_pi: 278.0,
            pco2_pi: 278.0,

            gas_exchange_scale: 1.833492,
            gas_exchange_tau: 7.46,

            temp_sensitivity: 0.03717879,

            irf_scale: 0.9492864,
            irf_switch_time: 9.9,

            // Early IRF: exponential sum (MAGICC7 lines 466-471)
            irf_early: IrfForm::ExponentialSum {
                coefficients: vec![0.058648, 0.07515, 0.079338, 0.41413, 0.24845, 0.12429],
                timescales: vec![1.0e10, 9.6218, 9.2364, 0.7603, 0.16294, 0.0032825],
            },

            // Late IRF: exponential sum (MAGICC7 lines 491-496)
            irf_late: IrfForm::ExponentialSum {
                coefficients: vec![0.01369, 0.012456, 0.026933, 0.026994, 0.036608, 0.06738],
                timescales: vec![1.0e10, 331.54, 107.57, 38.946, 11.677, 10.515],
            },

            mixed_layer_depth: 50.0,
            ocean_surface_area: 3.5375e14,
            sst_pi: 18.2997,

            delta_ospp_offsets: DELTA_OSPP_OFFSETS,
            delta_ospp_coefficients: DELTA_OSPP_COEFFICIENTS,

            steps_per_year: 12,
            max_history_months: 6000,
            enable_temp_feedback: true,
        }
    }

    /// Construct parameters for the HILDA ocean carbon model.
    ///
    /// HILDA uses exponential sum IRF for both early and late regimes.
    /// Physical parameters from MAGICC7 Fortran source
    /// (`carbon_cycle_ocean.f90` lines 144-152, 506-554).
    pub fn hilda() -> Self {
        Self {
            model: OceanCarbonModel::HILDA,

            co2_pi: 278.0,
            pco2_pi: 278.0,

            gas_exchange_scale: 1.833492,
            gas_exchange_tau: 9.06,

            temp_sensitivity: 0.03717879,

            irf_scale: 0.9492864,
            irf_switch_time: 2.0,

            // Early IRF: exponential sum (MAGICC7 lines 516-521)
            irf_early: IrfForm::ExponentialSum {
                coefficients: vec![0.12935, 0.24093, 0.24071, 0.17003, 0.21898],
                timescales: vec![1.0e10, 4.9792, 0.96083, 0.26936, 0.034569],
            },

            // Late IRF: exponential sum (MAGICC7 lines 541-546)
            irf_late: IrfForm::ExponentialSum {
                coefficients: vec![0.022936, 0.035549, 0.037820, 0.089318, 0.13963, 0.24278],
                timescales: vec![1.0e10, 232.30, 68.736, 18.601, 5.2528, 1.2679],
            },

            mixed_layer_depth: 75.0,
            ocean_surface_area: 3.62e14,
            sst_pi: 18.1716,

            delta_ospp_offsets: DELTA_OSPP_OFFSETS,
            delta_ospp_coefficients: DELTA_OSPP_COEFFICIENTS,

            steps_per_year: 12,
            max_history_months: 6000,
            enable_temp_feedback: true,
        }
    }

    /// Calculate the gas exchange rate (per month).
    ///
    /// $$k = \frac{\text{scale}}{\tau \times 12}$$
    pub fn gas_exchange_rate(&self) -> FloatValue {
        self.gas_exchange_scale / (self.gas_exchange_tau * 12.0)
    }

    /// Calculate the IRF value at a given time (years).
    ///
    /// Selects early or late IRF form based on switch time, evaluates it,
    /// then applies nonlinear scaling.
    pub fn irf(&self, t: FloatValue) -> FloatValue {
        let raw = if t < self.irf_switch_time {
            self.irf_early.evaluate(t)
        } else {
            self.irf_late.evaluate(t)
        };

        self.scale_irf(raw)
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
    pub fn delta_pco2_from_dic(&self, delta_dic: FloatValue) -> FloatValue {
        let dic_powers = [
            delta_dic,                  // g_1
            delta_dic.powi(2) * 1e-3,   // g_2
            -delta_dic.powi(3) * 1e-5,  // g_3 (negative)
            delta_dic.powi(4) * 1e-7,   // g_4
            -delta_dic.powi(5) * 1e-10, // g_5 (negative)
        ];

        let mut delta_pco2 = 0.0;
        for (i, &dic_power) in dic_powers.iter().enumerate() {
            let coeff = self.delta_ospp_offsets[i] + self.delta_ospp_coefficients[i] * self.sst_pi;
            delta_pco2 += coeff * dic_power;
        }

        delta_pco2
    }

    /// Calculate ocean pCO2 with temperature effect (Joos A25).
    ///
    /// $$pCO2_{ocn} = (pCO2_{pi} + \Delta pCO2_{DIC}) \times e^{\alpha_T \Delta T}$$
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

impl Default for OceanCarbonParameters {
    fn default() -> Self {
        Self::gfdl_3d()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_gfdl_3d() {
        let params = OceanCarbonParameters::default();
        assert_eq!(params.model, OceanCarbonModel::GFDL3D);
        assert!((params.co2_pi - 278.0).abs() < 1e-10);
        assert!((params.pco2_pi - 278.0).abs() < 1e-10);
        assert!((params.temp_sensitivity - 0.03717879).abs() < 1e-10);
        assert!((params.mixed_layer_depth - 50.9).abs() < 1e-10);
        assert!((params.irf_switch_time - 1.0).abs() < 1e-10);
        assert!(params.enable_temp_feedback);
        assert_eq!(params.steps_per_year, 12);
    }

    #[test]
    fn test_bern_2d_constructor() {
        let params = OceanCarbonParameters::bern_2d();
        assert_eq!(params.model, OceanCarbonModel::BERN2D);
        assert!((params.mixed_layer_depth - 50.0).abs() < 1e-10);
        assert!((params.sst_pi - 18.2997).abs() < 1e-10);
        assert!((params.irf_switch_time - 9.9).abs() < 1e-10);
        assert!((params.gas_exchange_tau - 7.46).abs() < 1e-10);
    }

    #[test]
    fn test_hilda_constructor() {
        let params = OceanCarbonParameters::hilda();
        assert_eq!(params.model, OceanCarbonModel::HILDA);
        assert!((params.mixed_layer_depth - 75.0).abs() < 1e-10);
        assert!((params.sst_pi - 18.1716).abs() < 1e-10);
        assert!((params.irf_switch_time - 2.0).abs() < 1e-10);
        assert!((params.gas_exchange_tau - 9.06).abs() < 1e-10);
        assert!((params.ocean_surface_area - 3.62e14).abs() < 1e10);
    }

    #[test]
    fn test_irf_hilda_decays() {
        let params = OceanCarbonParameters::hilda();

        let irf_0 = params.irf(0.0);
        let irf_10 = params.irf(10.0);
        let irf_100 = params.irf(100.0);

        assert!(
            irf_0 > 0.9,
            "HILDA IRF at t=0 should be close to 1.0, got {:.4}",
            irf_0
        );
        assert!(irf_10 < irf_0, "HILDA IRF should decay at t=10");
        assert!(
            irf_100 < irf_10,
            "HILDA IRF should continue decaying at t=100"
        );
        assert!(irf_100 > 0.0, "HILDA IRF should remain positive");
    }

    #[test]
    fn test_irf_switch_time_gfdl_3d() {
        // 3D-GFDL switches from polynomial to exponential sum at 1.0 year.
        // The forms are different, so a small discontinuity is expected,
        // but both values should be positive and in a reasonable range.
        let params = OceanCarbonParameters::gfdl_3d();
        let eps = 1e-6;

        let before = params.irf(params.irf_switch_time - eps);
        let after = params.irf(params.irf_switch_time + eps);

        assert!(
            before > 0.0 && before < 1.5,
            "IRF just before switch should be in (0, 1.5), got {:.6}",
            before
        );
        assert!(
            after > 0.0 && after < 1.5,
            "IRF just after switch should be in (0, 1.5), got {:.6}",
            after
        );
        // Both sides should be in the same order of magnitude
        assert!(
            (before / after) > 0.1 && (before / after) < 10.0,
            "IRF should not jump by more than 10x at switch: before={:.6}, after={:.6}",
            before,
            after
        );
    }

    #[test]
    fn test_irf_switch_time_bern_2d() {
        // 2D-BERN uses exponential sums on both sides of switch at 9.9 years.
        let params = OceanCarbonParameters::bern_2d();
        let eps = 1e-6;

        let before = params.irf(params.irf_switch_time - eps);
        let after = params.irf(params.irf_switch_time + eps);

        assert!(
            before > 0.0,
            "IRF just before switch should be positive, got {:.6}",
            before
        );
        assert!(
            after > 0.0,
            "IRF just after switch should be positive, got {:.6}",
            after
        );
    }

    #[test]
    fn test_irf_switch_time_hilda() {
        // HILDA uses exponential sums on both sides of switch at 2.0 years.
        let params = OceanCarbonParameters::hilda();
        let eps = 1e-6;

        let before = params.irf(params.irf_switch_time - eps);
        let after = params.irf(params.irf_switch_time + eps);

        assert!(
            before > 0.0,
            "IRF just before switch should be positive, got {:.6}",
            before
        );
        assert!(
            after > 0.0,
            "IRF just after switch should be positive, got {:.6}",
            after
        );
    }

    #[test]
    fn test_gas_exchange_rate() {
        let params = OceanCarbonParameters::default();
        let rate = params.gas_exchange_rate();

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
    fn test_polynomial_irf_at_zero() {
        // 3D-GFDL polynomial: at t=0, only the constant term survives = 1.0
        let irf_form = IrfForm::Polynomial {
            coefficients: vec![1.0, -2.2617, 14.002, -48.770, 82.986, -67.527, 21.037],
        };
        let val = irf_form.evaluate(0.0);
        assert!(
            (val - 1.0).abs() < 1e-10,
            "Polynomial IRF at t=0 should be 1.0, got {:.6}",
            val
        );
    }

    #[test]
    fn test_polynomial_irf_at_half_year() {
        // Evaluate 3D-GFDL polynomial at t=0.5 years
        let irf_form = IrfForm::Polynomial {
            coefficients: vec![1.0, -2.2617, 14.002, -48.770, 82.986, -67.527, 21.037],
        };
        let val = irf_form.evaluate(0.5);
        // Manual: 1 + (-2.2617)(0.5) + 14.002(0.25) + (-48.77)(0.125)
        //       + 82.986(0.0625) + (-67.527)(0.03125) + 21.037(0.015625)
        // = 1.0 - 1.13085 + 3.5005 - 6.09625 + 5.186625 - 2.11021875 + 0.328703125
        // ≈ 0.6788
        assert!(
            val > 0.5 && val < 1.0,
            "Polynomial IRF at t=0.5yr should be between 0.5 and 1.0, got {:.4}",
            val
        );
    }

    #[test]
    fn test_exponential_sum_irf_at_zero() {
        // 2D-BERN early: coefficients sum to ~1.0 at t=0
        let irf_form = IrfForm::ExponentialSum {
            coefficients: vec![0.058648, 0.07515, 0.079338, 0.41413, 0.24845, 0.12429],
            timescales: vec![1.0e10, 9.6218, 9.2364, 0.7603, 0.16294, 0.0032825],
        };
        let val = irf_form.evaluate(0.0);
        let expected_sum: FloatValue = [0.058648, 0.07515, 0.079338, 0.41413, 0.24845, 0.12429]
            .iter()
            .sum();
        assert!(
            (val - expected_sum).abs() < 1e-10,
            "ExponentialSum at t=0 should equal coefficient sum {:.6}, got {:.6}",
            expected_sum,
            val
        );
    }

    #[test]
    fn test_irf_at_zero() {
        let params = OceanCarbonParameters::default();
        let irf = params.irf(0.0);

        // At t=0, 3D-GFDL polynomial gives 1.0. With scaling (f=0.9492864):
        // scaled = (1.0 * f) / (1.0 * f + 1.0 - 1.0) = 1.0
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
    fn test_irf_bern_2d_decays() {
        let params = OceanCarbonParameters::bern_2d();

        let irf_0 = params.irf(0.0);
        let irf_10 = params.irf(10.0);
        let irf_100 = params.irf(100.0);

        assert!(
            irf_0 > 0.9,
            "BERN IRF at t=0 should be close to 1.0, got {:.4}",
            irf_0
        );
        assert!(irf_10 < irf_0, "BERN IRF should decay at t=10");
        assert!(
            irf_100 < irf_10,
            "BERN IRF should continue decaying at t=100"
        );
        assert!(irf_100 > 0.0, "BERN IRF should remain positive");
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

        assert!(
            pco2_warm > pco2_cold,
            "Warming should increase pCO2: cold={:.2}, warm={:.2}",
            pco2_cold,
            pco2_warm
        );

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
        let params = OceanCarbonParameters {
            enable_temp_feedback: false,
            ..Default::default()
        };

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

        assert!(factor > 0.0, "DIC conversion factor should be positive");
        assert!(factor.is_finite(), "DIC conversion factor should be finite");

        // Order of magnitude check: ~1e17 / (50.9 * 3.55e14) ≈ 5.5
        assert!(
            factor > 1.0 && factor < 100.0,
            "DIC conversion factor seems out of range: {:.2}",
            factor
        );
    }

    #[test]
    fn test_serialization_roundtrip() {
        let params = OceanCarbonParameters::default();
        let json = serde_json::to_string(&params).expect("Serialization failed");
        let parsed: OceanCarbonParameters =
            serde_json::from_str(&json).expect("Deserialization failed");

        assert!(
            (params.co2_pi - parsed.co2_pi).abs() < 1e-10,
            "Parameters should survive round-trip"
        );
        assert_eq!(params.model, parsed.model);
    }

    #[test]
    fn test_serialization_bern_2d_roundtrip() {
        let params = OceanCarbonParameters::bern_2d();
        let json = serde_json::to_string(&params).expect("Serialization failed");
        let parsed: OceanCarbonParameters =
            serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(parsed.model, OceanCarbonModel::BERN2D);
        assert!((parsed.irf_switch_time - 9.9).abs() < 1e-10);
    }

    #[test]
    fn test_horner_matches_naive_polynomial() {
        // Verify Horner's method gives same result as naive evaluation
        let coeffs = vec![1.0, -2.2617, 14.002, -48.770, 82.986, -67.527, 21.037];
        let irf = IrfForm::Polynomial {
            coefficients: coeffs.clone(),
        };

        let t = 0.75;
        let horner = irf.evaluate(t);

        // Naive evaluation
        let naive: FloatValue = coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| c * t.powi(i as i32))
            .sum();

        assert!(
            (horner - naive).abs() < 1e-10,
            "Horner ({:.8}) should match naive ({:.8})",
            horner,
            naive
        );
    }
}
