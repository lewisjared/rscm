//! GHG effective radiative forcing parameters
//!
//! Parameters for well-mixed greenhouse gas (CO2, CH4, N2O) effective radiative
//! forcing calculations using the IPCCTAR or OLBL methods.

use rscm_components::meinshausen2020::Meinshausen2020;
use serde::{Deserialize, Serialize};

/// Forcing calculation method
///
/// Selects which radiative transfer parameterisation to use for
/// CO2/CH4/N2O forcing. Strings use MAGICC7's `CORE_CO2CH4N2O_RFMETHOD` spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForcingMethod {
    /// IPCCTAR method (Myhre et al. 1998)
    ///
    /// Simple logarithmic/square-root formulae with CH4-N2O overlap.
    /// Used for backward compatibility and simpler validation.
    #[serde(rename = "IPCCTAR")]
    IPCCTAR,

    /// OLBL method (Oslo line-by-line)
    ///
    /// Concentration-dependent forcing coefficients calibrated against
    /// line-by-line radiative transfer calculations. This is the method
    /// used in MAGICC7 v7.5.3 for AR6 assessments.
    #[serde(rename = "OLBL")]
    OLBL,
}

/// Parameters for GHG effective radiative forcing
///
/// # Forcing Methods
///
/// Two methods are supported:
///
/// ## IPCCTAR (Myhre et al. 1998)
///
/// $$F_{CO2} = \frac{\Delta Q_{2 \times CO2}}{\ln 2} \cdot \ln\left(\frac{C}{C_0}\right)$$
///
/// ## OLBL (MAGICC7 v7.5.3)
///
/// Concentration-dependent alpha coefficients calibrated against
/// line-by-line radiative transfer. CO2 uses a quadratic alpha
/// with saturation, CH4 and N2O use square-root overlap terms.
///
/// # Rapid Adjustment
///
/// Both methods calculate instantaneous forcing. To obtain effective
/// radiative forcing (ERF), rapid adjustment factors are applied:
///
/// $$ERF_X = F_X \cdot \alpha_X$$
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct GHGForcingParameters {
    /// Forcing method (MAGICC7 `CORE_CO2CH4N2O_RFMETHOD`)
    pub method: ForcingMethod,

    // === Pre-industrial concentrations ===
    /// Pre-industrial CO2 concentration (MAGICC7 `CO2_PREINDCO2CONC`)
    ///
    /// unit: ppm
    ///
    /// Default: 278.0 ppm
    /// MAGICC7: CO2_PREINDCO2CONC
    pub co2_pi: f64,

    /// Pre-industrial CH4 concentration
    ///
    /// unit: ppb
    ///
    /// Default: 722.0 ppb
    pub ch4_pi: f64,

    /// Pre-industrial N2O concentration
    ///
    /// unit: ppb
    ///
    /// Default: 270.0 ppb
    pub n2o_pi: f64,

    // === IPCCTAR-specific parameters ===
    /// 2xCO2 effective radiative forcing (MAGICC7 `CORE_DELQ2XCO2`).
    ///
    /// unit: W/m^2
    ///
    /// The IPCCTAR method uses `alpha = erf_2xco2 / ln(2)`.
    ///
    /// Default: 3.71 W/m2
    /// MAGICC7: CORE_DELQ2XCO2
    pub erf_2xco2: f64,

    /// CH4 radiative efficiency, the IPCCTAR coefficient for CH4 (MAGICC7 `CH4_RADEFF_WM2PERPPB`)
    ///
    /// unit: W/m^2 per sqrt(ppb)
    ///
    /// Default: 0.036
    pub ch4_radiative_efficiency: f64,

    /// N2O radiative efficiency, the IPCCTAR coefficient for N2O (MAGICC7 `N2O_RADEFF_WM2PERPPB`)
    ///
    /// unit: W/m^2 per sqrt(ppb)
    ///
    /// Default: 0.12
    pub n2o_radiative_efficiency: f64,

    // === OLBL coefficients (MAGICC7 v7.5.3) ===
    // CO2: alpha = a1*(C-C0)^2 + b1*(C-C0) + d1 + c1*sqrt(N2O)
    /// OLBL CO2 quadratic coefficient $a_1$
    ///
    /// unit: W/m^2/ppm^2
    ///
    /// Default: -2.4785e-7
    pub olbl_co2_a1: f64,

    /// OLBL CO2 linear coefficient $b_1$
    ///
    /// unit: W/m^2/ppm
    ///
    /// Default: 7.5906e-4
    pub olbl_co2_b1: f64,

    /// OLBL CO2 N2O overlap coefficient $c_1$
    ///
    /// unit: W/m^2/sqrt(ppb)
    ///
    /// Default: -2.1492e-3
    pub olbl_co2_c1: f64,

    /// OLBL CO2 constant $d_1$
    ///
    /// unit: W/m^2
    ///
    /// Default: 5.2
    pub olbl_co2_d1: f64,

    // CH4: RF = (a3*sqrt(CH4) + b3*sqrt(N2O) + d3) * (sqrt(CH4) - sqrt(CH4_pi))
    /// OLBL CH4 self-overlap coefficient $a_3$
    ///
    /// unit: W/m^2/sqrt(ppb)
    ///
    /// Default: -8.9603e-5
    pub olbl_ch4_a3: f64,

    /// OLBL CH4 N2O overlap coefficient $b_3$
    ///
    /// unit: W/m^2/sqrt(ppb)
    ///
    /// Default: -1.2462e-4
    pub olbl_ch4_b3: f64,

    /// OLBL CH4 constant $d_3$
    ///
    /// unit: W/m^2
    ///
    /// Default: 0.045
    pub olbl_ch4_d3: f64,

    // N2O: RF = (a2*sqrt(CO2) + b2*sqrt(N2O) + c2*sqrt(CH4) + d2) * (sqrt(N2O) - sqrt(N2O_pi))
    /// OLBL N2O CO2 overlap coefficient $a_2$
    ///
    /// unit: W/m^2/sqrt(ppm)
    ///
    /// Default: -3.4197e-4
    pub olbl_n2o_a2: f64,

    /// OLBL N2O self-overlap coefficient $b_2$
    ///
    /// unit: W/m^2/sqrt(ppb)
    ///
    /// Default: 2.5455e-4
    pub olbl_n2o_b2: f64,

    /// OLBL N2O CH4 overlap coefficient $c_2$
    ///
    /// unit: W/m^2/sqrt(ppb)
    ///
    /// Default: -2.4357e-4
    pub olbl_n2o_c2: f64,

    /// OLBL N2O constant $d_2$
    ///
    /// unit: W/m^2
    ///
    /// Default: 0.14
    pub olbl_n2o_d2: f64,

    // === Rapid adjustment factors ===
    /// Rapid adjustment factor that converts instantaneous CO2 forcing to ERF
    ///
    /// unit: dimensionless
    pub adjust_co2: f64,

    /// Rapid adjustment factor that converts instantaneous CH4 forcing to ERF.
    ///
    /// unit: dimensionless
    ///
    /// Below 1 because CH4 has a negative tropospheric adjustment.
    ///
    /// Default: 0.86 (CH4 has negative tropospheric adjustment)
    pub adjust_ch4: f64,

    /// Rapid adjustment factor that converts instantaneous N2O forcing to ERF.
    ///
    /// unit: dimensionless
    ///
    /// MAGICC7 v7.5.3 applies no adjustment.
    ///
    /// Default: 1.0 (no adjustment in MAGICC7 v7.5.3)
    pub adjust_n2o: f64,

    /// Stratospheric H2O ERF from CH4 oxidation as a fraction of CH4 ERF at pre-industrial N2O
    /// (MAGICC7 `CH4_ADDEDSTRATH2O_PERCENT`)
    ///
    /// unit: dimensionless
    pub ch4_stratospheric_h2o_fraction: f64,
}

impl Default for GHGForcingParameters {
    fn default() -> Self {
        Self {
            method: ForcingMethod::OLBL,

            // Pre-industrial concentrations
            co2_pi: 278.0, // ppm
            ch4_pi: 722.0, // ppb
            n2o_pi: 270.0, // ppb

            // IPCCTAR parameters
            erf_2xco2: 3.71,
            ch4_radiative_efficiency: 0.036,
            n2o_radiative_efficiency: 0.12,

            // OLBL coefficients (MAGICC7 v7.5.3 MAGCFG_DEFAULTALL.CFG)
            olbl_co2_a1: -2.4785e-7,
            olbl_co2_b1: 7.5906e-4,
            olbl_co2_c1: -2.1492e-3,
            olbl_co2_d1: 5.2,
            olbl_ch4_a3: -8.9603e-5,
            olbl_ch4_b3: -1.2462e-4,
            olbl_ch4_d3: 0.045,
            olbl_n2o_a2: -3.4197e-4,
            olbl_n2o_b2: 2.5455e-4,
            olbl_n2o_c2: -2.4357e-4,
            olbl_n2o_d2: 0.14,

            // Rapid adjustment factors (MAGICC7 v7.5.3 defaults)
            adjust_co2: 1.05,
            adjust_ch4: 0.86,
            adjust_n2o: 1.0,

            ch4_stratospheric_h2o_fraction: 0.0923,
        }
    }
}

impl GHGForcingParameters {
    /// OLBL coefficients.
    pub fn olbl(&self) -> Meinshausen2020 {
        Meinshausen2020 {
            a1: self.olbl_co2_a1,
            b1: self.olbl_co2_b1,
            c1: self.olbl_co2_c1,
            d1: self.olbl_co2_d1,
            a2: self.olbl_n2o_a2,
            b2: self.olbl_n2o_b2,
            c2: self.olbl_n2o_c2,
            d2: self.olbl_n2o_d2,
            a3: self.olbl_ch4_a3,
            b3: self.olbl_ch4_b3,
            d3: self.olbl_ch4_d3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() {
        let params = GHGForcingParameters::default();

        assert_eq!(params.method, ForcingMethod::OLBL);
        assert!((params.co2_pi - 278.0).abs() < 1e-10);
        assert!((params.ch4_pi - 722.0).abs() < 1e-10);
        assert!((params.n2o_pi - 270.0).abs() < 1e-10);
        assert!((params.erf_2xco2 - 3.71).abs() < 1e-10);

        // Adjustment factors should be close to 1
        assert!(params.adjust_co2 > 0.5 && params.adjust_co2 < 2.0);
        assert!(params.adjust_ch4 > 0.5 && params.adjust_ch4 < 2.0);
        assert!(params.adjust_n2o > 0.5 && params.adjust_n2o < 2.0);
    }

    #[test]
    fn test_serialisation_roundtrip() {
        let params = GHGForcingParameters::default();
        let json = serde_json::to_string(&params).unwrap();
        let restored: GHGForcingParameters = serde_json::from_str(&json).unwrap();

        assert_eq!(params.method, restored.method);
        assert!((params.co2_pi - restored.co2_pi).abs() < 1e-10);
        assert!((params.adjust_co2 - restored.adjust_co2).abs() < 1e-10);
    }

    #[test]
    fn test_partial_deserialisation() {
        let json = r#"{"method": "IPCCTAR", "erf_2xco2": 3.80}"#;
        let params: GHGForcingParameters = serde_json::from_str(json).unwrap();

        assert_eq!(params.method, ForcingMethod::IPCCTAR);
        assert!((params.erf_2xco2 - 3.80).abs() < 1e-10);
        // Defaults for unspecified fields
        assert!((params.co2_pi - 278.0).abs() < 1e-10);
    }
}
