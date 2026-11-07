//! Aerosol forcing parameters
//!
//! Parameters for direct and indirect aerosol effective radiative forcing.

use serde::{Deserialize, Serialize};

/// Parameters for direct aerosol effective radiative forcing
///
/// Direct aerosol ERF is calculated as a linear combination of emissions
/// from different aerosol species, each with its own radiative efficiency:
///
/// $$ERF_{direct} = \sum_i \alpha_i \cdot E_i$$
///
/// where $\alpha_i$ is the forcing coefficient and $E_i$ is the emission rate
/// for species $i$.
///
/// # Regional Distribution
///
/// Forcing is distributed across four regions using species-specific patterns.
/// The global forcing is first calculated, then distributed proportionally
/// to regional weights that sum to 1.0.
///
/// # Harmonisation
///
/// Optional harmonisation scales the forcing to match a target value at a
/// reference year. This is useful for calibrating to observed/modelled forcing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct AerosolDirectParameters {
    // === Species Forcing Coefficients ===
    /// SOx forcing coefficient.
    ///
    /// unit: W/m^2 per Mt S/yr
    ///
    /// Negative (cooling) because sulfate aerosols scatter incoming solar radiation.
    ///
    /// Default: -0.0035 W/m²/(Mt S/yr), calibrated to the AR6 best estimate.
    pub sox_coefficient: f64,

    /// Black carbon forcing coefficient.
    ///
    /// unit: W/m^2 per Mt BC/yr
    ///
    /// Positive (warming) because BC absorbs solar radiation.
    ///
    /// Default: 0.0077 W/m²/(Mt BC/yr)
    pub bc_coefficient: f64,

    /// Organic carbon forcing coefficient.
    ///
    /// unit: W/m^2 per Mt OC/yr
    ///
    /// Negative (cooling) because OC mostly scatters radiation.
    ///
    /// Default: -0.002 W/m²/(Mt OC/yr)
    pub oc_coefficient: f64,

    /// Nitrate forcing coefficient.
    ///
    /// unit: W/m^2 per Mt N/yr
    ///
    /// Negative (cooling) because nitrate aerosols scatter radiation.
    ///
    /// Default: -0.001 W/m²/(Mt N/yr)
    pub nitrate_coefficient: f64,

    // === Regional Distribution Weights ===
    // Order: [Northern Ocean, Northern Land, Southern Ocean, Southern Land]
    /// SOx regional pattern, as a fraction in each region of the four-box grid.
    ///
    /// unit: dimensionless
    ///
    /// The default is concentrated in the Northern Hemisphere, especially over land.
    pub sox_regional: [f64; 4],

    /// BC regional pattern, as a fraction in each region of the four-box grid.
    ///
    /// unit: dimensionless
    ///
    /// The default is concentrated in the Northern Hemisphere.
    pub bc_regional: [f64; 4],

    /// OC regional pattern, as a fraction in each region of the four-box grid.
    ///
    /// unit: dimensionless
    ///
    /// The default is like SOx but more uniform.
    pub oc_regional: [f64; 4],

    /// Nitrate regional pattern, as a fraction in each region of the four-box grid.
    ///
    /// unit: dimensionless
    ///
    /// The default is concentrated in the Northern Hemisphere.
    pub nitrate_regional: [f64; 4],

    // === Pre-industrial emissions (baseline) ===
    /// Pre-industrial SOx emissions, from natural volcanic sources by default
    ///
    /// unit: Mt S/yr
    ///
    /// Default: 1.0 Mt S/yr (natural volcanic)
    pub sox_pi: f64,

    /// Pre-industrial BC emissions, from biomass burning by default
    ///
    /// unit: Mt BC/yr
    ///
    /// Default: 2.5 Mt BC/yr (biomass burning)
    pub bc_pi: f64,

    /// Pre-industrial OC emissions, from biomass burning by default
    ///
    /// unit: Mt OC/yr
    ///
    /// Default: 10.0 Mt OC/yr (biomass burning)
    pub oc_pi: f64,

    /// Pre-industrial NOx emissions for nitrate, from natural sources by default
    ///
    /// unit: Mt N/yr
    ///
    /// Default: 10.0 Mt N/yr (natural sources)
    pub nox_pi: f64,

    // === Harmonisation ===
    /// Scale the ERF to a target value in a reference year.
    ///
    /// When true, the ERF is scaled so that its value at `harmonisation_reference_year`
    /// equals `harmonisation_target`.
    /// The component does not apply it yet.
    pub include_harmonisation: bool,

    /// Reference year for harmonisation
    ///
    /// unit: yr
    ///
    /// Default: 2019.0
    pub harmonisation_reference_year: f64,

    /// Target ERF in the reference year.
    ///
    /// unit: W/m^2
    ///
    ///
    /// Default: -0.22 W/m² (AR6 central estimate for direct aerosol)
    pub harmonisation_target: f64,
}

impl Default for AerosolDirectParameters {
    fn default() -> Self {
        Self {
            // Forcing coefficients (W/m² per emission unit)
            sox_coefficient: -0.0035,    // Cooling
            bc_coefficient: 0.0077,      // Warming
            oc_coefficient: -0.002,      // Cooling
            nitrate_coefficient: -0.001, // Cooling

            // Regional patterns (must sum to 1.0)
            // Order: [NH Ocean, NH Land, SH Ocean, SH Land]
            sox_regional: [0.15, 0.55, 0.10, 0.20],
            bc_regional: [0.15, 0.50, 0.15, 0.20],
            oc_regional: [0.15, 0.45, 0.15, 0.25],
            nitrate_regional: [0.15, 0.50, 0.15, 0.20],

            // Pre-industrial emissions
            sox_pi: 1.0,  // Mt S/yr
            bc_pi: 2.5,   // Mt BC/yr
            oc_pi: 10.0,  // Mt OC/yr
            nox_pi: 10.0, // Mt N/yr

            // Harmonisation disabled by default
            include_harmonisation: false,
            harmonisation_reference_year: 2019.0,
            harmonisation_target: -0.22, // W/m²
        }
    }
}

/// Parameters for indirect aerosol effective radiative forcing (cloud effects)
///
/// Indirect aerosol ERF arises from aerosol effects on cloud properties:
///
/// 1. **Cloud Albedo Effect (First Indirect / Twomey Effect)**: More aerosols
///    provide more cloud condensation nuclei (CCN), producing more but smaller
///    cloud droplets, increasing cloud reflectivity (cooling).
///
/// 2. **Cloud Lifetime Effect (Second Indirect / Albrecht Effect)**: Smaller
///    droplets take longer to coalesce into rain, potentially increasing cloud
///    lifetime and coverage (cooling). Note: This effect is uncertain and often
///    set to zero in default configurations.
///
/// # Formulation
///
/// The forcing uses a logarithmic relationship with aerosol burden:
///
/// $$ERF_{indirect} = \alpha \cdot \ln(1 + B / B_0)$$
///
/// where $B$ is the aerosol burden (proxy: SOx + OC emissions) and $B_0$ is
/// a reference burden scale.
///
/// # Differences from MAGICC7
///
/// This is a simplified implementation:
///
/// - **Species weights**: MAGICC7 uses detailed species weights for CCN
///   contribution (SOx, OC, BC, nitrate, sea salt). Here, only SOx and OC
///   are considered as the dominant CCN sources.
/// - **Regional patterns**: MAGICC7 applies detailed regional forcing patterns.
///   Here, forcing is scalar (global mean).
/// - **Cloud lifetime effect**: MAGICC7 has separate cloud albedo and lifetime
///   parameters. Here, they are combined into a single coefficient.
/// - **Optical thickness proxy**: MAGICC7 uses aerosol optical thickness.
///   Here, emissions are used directly as a proxy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct AerosolIndirectParameters {
    /// Cloud albedo effect coefficient.
    ///
    /// unit: W/m^2
    ///
    /// Sets the strength of the first indirect effect.
    /// Negative (cooling) because more aerosol increases cloud albedo.
    ///
    /// Default: -1.0 W/m² (calibrated to AR6 central estimate)
    pub cloud_albedo_coefficient: f64,

    /// Reference aerosol burden for the logarithmic scaling.
    ///
    /// unit: Tg/yr
    ///
    /// At this burden $\ln(1 + B/B_0) = \ln(2)$, so the ERF equals the coefficient.
    ///
    /// Default: 50.0 Tg/yr
    pub reference_burden: f64,

    /// SOx weight in the aerosol burden.
    ///
    /// unit: dimensionless
    ///
    /// SOx is a primary CCN source and contributes strongly to indirect effects.
    ///
    /// Default: 1.0 (emissions used directly)
    pub sox_weight: f64,

    /// OC weight in the aerosol burden.
    ///
    /// unit: dimensionless
    ///
    /// OC contributes to CCN, but less effectively than sulfate per unit mass.
    ///
    /// Default: 0.3 (30% as effective as SOx per mass)
    pub oc_weight: f64,

    // === Pre-industrial emissions (baseline) ===
    /// Pre-industrial SOx emissions
    ///
    /// unit: Mt S/yr
    ///
    /// Default: 1.0 Mt S/yr
    pub sox_pi: f64,

    /// Pre-industrial OC emissions
    ///
    /// unit: Mt OC/yr
    ///
    /// Default: 10.0 Mt OC/yr
    pub oc_pi: f64,

    // === Harmonisation ===
    /// Scale the ERF to a target value in a reference year.
    ///
    /// The component does not apply it yet.
    pub include_harmonisation: bool,

    /// Reference year for harmonisation
    ///
    /// unit: yr
    ///
    /// Default: 2019.0
    pub harmonisation_reference_year: f64,

    /// Target ERF in the reference year.
    ///
    /// unit: W/m^2
    ///
    ///
    /// Default: -0.89 W/m² (AR6 central estimate for cloud albedo effect)
    pub harmonisation_target: f64,
}

impl Default for AerosolIndirectParameters {
    fn default() -> Self {
        Self {
            // Cloud effect coefficient
            cloud_albedo_coefficient: -1.0, // W/m² (cooling)
            reference_burden: 50.0,         // Tg/yr

            // Species weights for CCN contribution
            sox_weight: 1.0, // SOx is primary CCN source
            oc_weight: 0.3,  // OC less effective

            // Pre-industrial emissions
            sox_pi: 1.0, // Mt S/yr
            oc_pi: 10.0, // Mt OC/yr

            // Harmonisation disabled by default
            include_harmonisation: false,
            harmonisation_reference_year: 2019.0,
            harmonisation_target: -0.89, // W/m² (AR6 central estimate)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_default_parameters() {
        let params = AerosolDirectParameters::default();

        // Cooling species should have negative coefficients
        assert!(params.sox_coefficient < 0.0, "SOx should cause cooling");
        assert!(params.oc_coefficient < 0.0, "OC should cause cooling");
        assert!(
            params.nitrate_coefficient < 0.0,
            "Nitrate should cause cooling"
        );

        // BC should have positive coefficient (warming)
        assert!(params.bc_coefficient > 0.0, "BC should cause warming");

        // Regional patterns should sum to ~1.0
        let sox_sum: f64 = params.sox_regional.iter().sum();
        assert!(
            (sox_sum - 1.0).abs() < 0.01,
            "SOx regional weights should sum to 1.0, got {}",
            sox_sum
        );

        let bc_sum: f64 = params.bc_regional.iter().sum();
        assert!(
            (bc_sum - 1.0).abs() < 0.01,
            "BC regional weights should sum to 1.0, got {}",
            bc_sum
        );

        // Pre-industrial values should be positive
        assert!(params.sox_pi > 0.0);
        assert!(params.bc_pi > 0.0);
    }

    #[test]
    fn test_indirect_default_parameters() {
        let params = AerosolIndirectParameters::default();

        // Cloud effects should be cooling (negative)
        assert!(
            params.cloud_albedo_coefficient < 0.0,
            "Cloud albedo effect should cause cooling"
        );

        // Reference burden should be positive
        assert!(params.reference_burden > 0.0);

        // Weights should be positive
        assert!(params.sox_weight > 0.0);
        assert!(params.oc_weight > 0.0);

        // SOx should be more effective than OC
        assert!(
            params.sox_weight > params.oc_weight,
            "SOx should be more effective CCN than OC"
        );
    }

    #[test]
    fn test_direct_serialisation_roundtrip() {
        let params = AerosolDirectParameters::default();
        let json = serde_json::to_string(&params).unwrap();
        let restored: AerosolDirectParameters = serde_json::from_str(&json).unwrap();

        assert!((params.sox_coefficient - restored.sox_coefficient).abs() < 1e-10);
        assert!((params.bc_coefficient - restored.bc_coefficient).abs() < 1e-10);
        assert_eq!(params.include_harmonisation, restored.include_harmonisation);
    }

    #[test]
    fn test_indirect_serialisation_roundtrip() {
        let params = AerosolIndirectParameters::default();
        let json = serde_json::to_string(&params).unwrap();
        let restored: AerosolIndirectParameters = serde_json::from_str(&json).unwrap();

        assert!(
            (params.cloud_albedo_coefficient - restored.cloud_albedo_coefficient).abs() < 1e-10
        );
        assert!((params.reference_burden - restored.reference_burden).abs() < 1e-10);
    }

    #[test]
    fn test_partial_deserialisation() {
        // Test that #[serde(default)] allows partial deserialisation for both structs

        // Direct parameters
        let json_direct = r#"{"sox_coefficient": -0.005, "include_harmonisation": true}"#;
        let params: AerosolDirectParameters =
            serde_json::from_str(json_direct).expect("Partial deserialisation failed");

        assert!((params.sox_coefficient + 0.005).abs() < 1e-10);
        assert!(params.include_harmonisation);
        // Defaults for unspecified fields
        assert!((params.bc_coefficient - 0.0077).abs() < 1e-10);
        assert!((params.harmonisation_reference_year - 2019.0).abs() < 1e-10);

        // Indirect parameters
        let json_indirect = r#"{"cloud_albedo_coefficient": -1.5}"#;
        let params: AerosolIndirectParameters =
            serde_json::from_str(json_indirect).expect("Partial deserialisation failed");

        assert!((params.cloud_albedo_coefficient + 1.5).abs() < 1e-10);
        // Defaults for unspecified fields
        assert!((params.reference_burden - 50.0).abs() < 1e-10);
        assert!((params.sox_weight - 1.0).abs() < 1e-10);
    }
}
