//! GHG radiative forcing parameters
//!
//! Parameters for well-mixed greenhouse gas (CO2, CH4, N2O) radiative
//! forcing calculations using IPCCTAR or Etminan methods.

use serde::{Deserialize, Serialize};

/// Forcing calculation method
///
/// Selects which radiative transfer parameterisation to use for
/// CO2/CH4/N2O forcing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForcingMethod {
    /// IPCCTAR method (Myhre et al. 1998)
    ///
    /// Simple logarithmic/square-root formulae with CH4-N2O overlap.
    /// Used for backward compatibility and simpler validation.
    Ipcctar,

    /// Etminan method (Etminan et al. 2016)
    ///
    /// Concentration-dependent forcing coefficients that account for
    /// shortwave absorption. Used as the OLBL proxy in AR6 assessments.
    Etminan,
}

/// Parameters for GHG radiative forcing
///
/// # Forcing Methods
///
/// Two methods are supported:
///
/// ## IPCCTAR (Myhre et al. 1998)
///
/// $$F_{CO2} = \frac{\Delta Q_{2 \times CO2}}{\ln 2} \cdot \ln\left(\frac{C}{C_0}\right)$$
///
/// ## Etminan (Etminan et al. 2016)
///
/// Concentration-dependent coefficients that capture shortwave
/// absorption effects not present in the simpler IPCCTAR formulae.
///
/// # Rapid Adjustment
///
/// Both methods calculate instantaneous forcing. To obtain effective
/// radiative forcing (ERF), rapid adjustment factors are applied:
///
/// $$ERF_X = F_X \cdot \alpha_X$$
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GhgForcingParameters {
    /// Forcing calculation method
    pub method: ForcingMethod,

    // === Pre-industrial concentrations ===
    /// Pre-industrial CO2 concentration (ppm).
    /// Default: 278.0 ppm
    pub co2_pi: f64,

    /// Pre-industrial CH4 concentration (ppb).
    /// Default: 722.0 ppb
    pub ch4_pi: f64,

    /// Pre-industrial N2O concentration (ppb).
    /// Default: 270.0 ppb
    pub n2o_pi: f64,

    // === IPCCTAR-specific parameters ===
    /// Forcing for CO2 doubling (W/m2).
    ///
    /// Used by IPCCTAR method: `alpha = delq2xco2 / ln(2)`.
    /// Default: 3.71 W/m2
    pub delq2xco2: f64,

    /// CH4 radiative efficiency (W/m2 per sqrt(ppb)).
    ///
    /// IPCCTAR coefficient for CH4 forcing.
    /// Default: 0.036
    pub ch4_radeff: f64,

    /// N2O radiative efficiency (W/m2 per sqrt(ppb)).
    ///
    /// IPCCTAR coefficient for N2O forcing.
    /// Default: 0.12
    pub n2o_radeff: f64,

    // === Etminan-specific coefficients ===
    // These follow Etminan et al. (2016) Table 1
    /// Etminan CO2 coefficient a1 scale (per ppm^2).
    /// Default: -2.4e-7
    pub etminan_co2_a1: f64,

    /// Etminan CO2 coefficient a2 scale (per ppm).
    /// Default: 7.2e-4
    pub etminan_co2_a2: f64,

    /// Etminan CO2 a3 N2O sensitivity (per ppb).
    /// Default: -2.1e-4
    pub etminan_co2_a3_n2o: f64,

    /// Etminan CO2 a3 offset.
    /// Default: 5.36
    pub etminan_co2_a3_offset: f64,

    /// Etminan CH4 coefficient b1 (per ppb).
    /// Default: -1.3e-6
    pub etminan_ch4_b1: f64,

    /// Etminan CH4 coefficient b2 (per ppb).
    /// Default: -8.2e-6
    pub etminan_ch4_b2: f64,

    /// Etminan CH4 coefficient b3.
    /// Default: 0.043
    pub etminan_ch4_b3: f64,

    /// Etminan N2O coefficient c1 (per ppb).
    /// Default: -8.0e-6
    pub etminan_n2o_c1: f64,

    /// Etminan N2O coefficient c2 (per ppb).
    /// Default: 4.2e-6
    pub etminan_n2o_c2: f64,

    /// Etminan N2O coefficient c3.
    /// Default: 0.12
    pub etminan_n2o_c3: f64,

    // === Rapid adjustment factors ===
    /// Rapid adjustment factor for CO2 forcing.
    ///
    /// Converts instantaneous forcing to ERF. AR6 default: 1.05
    pub adjust_co2: f64,

    /// Rapid adjustment factor for CH4 forcing.
    ///
    /// AR6 default: 0.86 (CH4 has negative tropospheric adjustment)
    pub adjust_ch4: f64,

    /// Rapid adjustment factor for N2O forcing.
    ///
    /// AR6 default: 0.93
    pub adjust_n2o: f64,
}

impl Default for GhgForcingParameters {
    fn default() -> Self {
        Self {
            method: ForcingMethod::Etminan,

            // Pre-industrial concentrations
            co2_pi: 278.0, // ppm
            ch4_pi: 722.0, // ppb
            n2o_pi: 270.0, // ppb

            // IPCCTAR parameters
            delq2xco2: 3.71,
            ch4_radeff: 0.036,
            n2o_radeff: 0.12,

            // Etminan coefficients (Table 1, Etminan et al. 2016)
            etminan_co2_a1: -2.4e-7,
            etminan_co2_a2: 7.2e-4,
            etminan_co2_a3_n2o: -2.1e-4,
            etminan_co2_a3_offset: 5.36,
            etminan_ch4_b1: -1.3e-6,
            etminan_ch4_b2: -8.2e-6,
            etminan_ch4_b3: 0.043,
            etminan_n2o_c1: -8.0e-6,
            etminan_n2o_c2: 4.2e-6,
            etminan_n2o_c3: 0.12,

            // Rapid adjustment factors (AR6 defaults)
            adjust_co2: 1.05,
            adjust_ch4: 0.86,
            adjust_n2o: 0.93,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() {
        let params = GhgForcingParameters::default();

        assert_eq!(params.method, ForcingMethod::Etminan);
        assert!((params.co2_pi - 278.0).abs() < 1e-10);
        assert!((params.ch4_pi - 722.0).abs() < 1e-10);
        assert!((params.n2o_pi - 270.0).abs() < 1e-10);
        assert!((params.delq2xco2 - 3.71).abs() < 1e-10);

        // Adjustment factors should be close to 1
        assert!(params.adjust_co2 > 0.5 && params.adjust_co2 < 2.0);
        assert!(params.adjust_ch4 > 0.5 && params.adjust_ch4 < 2.0);
        assert!(params.adjust_n2o > 0.5 && params.adjust_n2o < 2.0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let params = GhgForcingParameters::default();
        let json = serde_json::to_string(&params).unwrap();
        let restored: GhgForcingParameters = serde_json::from_str(&json).unwrap();

        assert_eq!(params.method, restored.method);
        assert!((params.co2_pi - restored.co2_pi).abs() < 1e-10);
        assert!((params.adjust_co2 - restored.adjust_co2).abs() < 1e-10);
    }

    #[test]
    fn test_partial_deserialization() {
        let json = r#"{"method": "Ipcctar", "delq2xco2": 3.80}"#;
        let params: GhgForcingParameters = serde_json::from_str(json).unwrap();

        assert_eq!(params.method, ForcingMethod::Ipcctar);
        assert!((params.delq2xco2 - 3.80).abs() < 1e-10);
        // Defaults for unspecified fields
        assert!((params.co2_pi - 278.0).abs() < 1e-10);
    }
}
