//! Halocarbon Chemistry Parameters
//!
//! Parameters for atmospheric halocarbon chemistry calculations, implementing
//! exponential decay for F-gases and Montreal Protocol gases.
//!
//! # Reference
//!
//! Based on MAGICC7 Module 03 (Halogenated Gas Chemistry). Species properties
//! are derived from IPCC AR6 and WMO Ozone Assessment reports.
//!
//! # Species Categories
//!
//! - **F-gases**: HFCs, PFCs, SF6, NF3 - potent greenhouse gases, no ozone depletion
//! - **Montreal gases**: CFCs, HCFCs, halons - both GHGs and ozone depleters

use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Data for a single halocarbon species.
///
/// Contains all physical and chemical properties needed for:
/// - Exponential decay calculation
/// - Radiative forcing calculation
/// - EESC calculation (for Montreal gases)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HalocarbonSpecies {
    /// Species identifier (e.g., "CFC-11", "HFC-134a")
    pub name: String,

    /// Total atmospheric lifetime
    /// unit: years
    pub lifetime: FloatValue,

    /// Radiative efficiency - forcing per unit concentration change
    /// unit: W/m^2 per ppb
    pub radiative_efficiency: FloatValue,

    /// Pre-industrial concentration (usually 0 for synthetic species)
    /// unit: ppt
    pub concentration_pi: FloatValue,

    /// Molecular weight (used for emissions to concentration conversion)
    /// unit: g/mol
    pub molecular_weight: FloatValue,

    /// Number of chlorine atoms (for EESC calculation)
    pub n_cl: u8,

    /// Number of bromine atoms (for EESC calculation)
    pub n_br: u8,

    /// Fractional release factor for EESC calculation (0-1)
    /// Represents fraction of species that releases halogens in the stratosphere
    /// F-gases typically have 0, Montreal gases have species-specific values
    pub fractional_release: FloatValue,
}

impl HalocarbonSpecies {
    /// Create a new halocarbon species with the given properties
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        lifetime: FloatValue,
        radiative_efficiency: FloatValue,
        concentration_pi: FloatValue,
        molecular_weight: FloatValue,
        n_cl: u8,
        n_br: u8,
        fractional_release: FloatValue,
    ) -> Self {
        Self {
            name: name.into(),
            lifetime,
            radiative_efficiency,
            concentration_pi,
            molecular_weight,
            n_cl,
            n_br,
            fractional_release,
        }
    }
}

/// Parameters for halocarbon chemistry calculations.
///
/// Contains species lists and global parameters for calculating:
/// - Atmospheric concentrations via exponential decay
/// - Radiative forcing from all halocarbons
/// - Equivalent Effective Stratospheric Chlorine (EESC)
///
/// # Exponential Decay
///
/// Each species follows simple exponential decay with emissions:
///
/// $$\frac{dC}{dt} = E - \frac{C}{\tau}$$
///
/// Analytical solution for one timestep:
///
/// $$C(t+\Delta t) = C(t) \cdot e^{-\Delta t/\tau} + E \cdot \tau \cdot (1 - e^{-\Delta t/\tau})$$
///
/// # EESC Calculation
///
/// $$\text{EESC} = \sum_i C_i \cdot (n_{Cl,i} + \alpha_{Br} \cdot n_{Br,i}) \cdot f_{release,i}$$
///
/// where $\alpha_{Br}$ is the bromine vs chlorine efficiency multiplier.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HalocarbonParameters {
    /// F-gas species (HFCs, PFCs, SF6, etc.)
    /// These are greenhouse gases but do not deplete stratospheric ozone.
    pub fgases: Vec<HalocarbonSpecies>,

    /// Montreal Protocol gases (CFCs, HCFCs, halons, etc.)
    /// These both contribute to radiative forcing AND deplete stratospheric ozone.
    pub montreal_gases: Vec<HalocarbonSpecies>,

    /// Bromine efficiency multiplier for EESC calculation
    /// Bromine is more effective at destroying ozone than chlorine.
    /// unit: dimensionless
    /// default: 60.0 (per WMO Scientific Assessment)
    pub br_multiplier: FloatValue,

    /// CFC-11 fractional release factor for EESC normalisation
    /// Used to normalise other species' release factors.
    /// unit: dimensionless
    /// default: 0.47
    pub cfc11_release_normalisation: FloatValue,

    /// EESC stratospheric mixing delay
    /// Time for tropospheric concentrations to affect stratospheric chemistry.
    /// unit: years
    /// default: 3.0
    pub eesc_delay: FloatValue,

    /// Molar mass of air
    /// unit: g/mol
    /// default: 28.97
    pub air_molar_mass: FloatValue,

    /// Total atmospheric mass
    /// unit: Tg
    /// default: 5.133e9
    pub atmospheric_mass_tg: FloatValue,

    /// Effective mixing box fraction
    /// Fraction of atmosphere available for mixing.
    /// unit: dimensionless
    /// default: 0.949
    pub mixing_box_fraction: FloatValue,
}

impl HalocarbonParameters {
    /// Calculate conversion factor from kt/yr emissions to ppt/yr concentration change.
    ///
    /// $$\text{conv} = \frac{M_{air}}{M_{atm} \cdot M_{mol} \cdot f_{mix}}$$
    ///
    /// where:
    /// - $M_{air}$ = molar mass of air (g/mol)
    /// - $M_{atm}$ = atmospheric mass (Tg)
    /// - $M_{mol}$ = molecular mass of species (g/mol)
    /// - $f_{mix}$ = mixing box fraction
    pub fn emission_to_concentration_factor(&self, molecular_weight: FloatValue) -> FloatValue {
        // Convert atmospheric mass from Tg to g: 5.133e9 Tg * 1e12 g/Tg = 5.133e21 g
        let atm_mass_g = self.atmospheric_mass_tg * 1e12;

        // Calculate conversion: result is in ppt per (kt/yr)
        // Emissions are in kt/yr = 1e9 g/yr
        // We want ppt = 1e-12 molar ratio
        // conv = (M_air / M_mol) * (1e9 g emissions / atm_mass_g) * 1e12 ppt_factor / f_mix
        (self.air_molar_mass / molecular_weight) * (1e9 / atm_mass_g) * 1e12
            / self.mixing_box_fraction
    }

    /// Get all species (F-gases + Montreal gases) as a single iterator
    pub fn all_species(&self) -> impl Iterator<Item = &HalocarbonSpecies> {
        self.fgases.iter().chain(self.montreal_gases.iter())
    }

    /// Get a species by name
    pub fn get_species(&self, name: &str) -> Option<&HalocarbonSpecies> {
        self.all_species().find(|s| s.name == name)
    }
}

impl Default for HalocarbonParameters {
    fn default() -> Self {
        Self {
            fgases: default_fgases(),
            montreal_gases: default_montreal_gases(),
            br_multiplier: 60.0,
            cfc11_release_normalisation: 0.47,
            eesc_delay: 3.0,
            air_molar_mass: 28.97,
            atmospheric_mass_tg: 5.133e9,
            mixing_box_fraction: 0.949,
        }
    }
}

/// Create default F-gas species list.
///
/// Species data from IPCC AR6 and MAGICC7 defaults.
fn default_fgases() -> Vec<HalocarbonSpecies> {
    vec![
        // PFCs (perfluorocarbons)
        HalocarbonSpecies::new("CF4", 50000.0, 0.09, 0.0, 88.0, 0, 0, 0.0),
        HalocarbonSpecies::new("C2F6", 10000.0, 0.25, 0.0, 138.0, 0, 0, 0.0),
        HalocarbonSpecies::new("C3F8", 2600.0, 0.28, 0.0, 188.0, 0, 0, 0.0),
        HalocarbonSpecies::new("C4F10", 2600.0, 0.36, 0.0, 238.0, 0, 0, 0.0),
        HalocarbonSpecies::new("C5F12", 4100.0, 0.41, 0.0, 288.0, 0, 0, 0.0),
        HalocarbonSpecies::new("C6F14", 3100.0, 0.44, 0.0, 338.0, 0, 0, 0.0),
        HalocarbonSpecies::new("C7F16", 3000.0, 0.50, 0.0, 388.0, 0, 0, 0.0),
        HalocarbonSpecies::new("C8F18", 3000.0, 0.55, 0.0, 438.0, 0, 0, 0.0),
        HalocarbonSpecies::new("c-C4F8", 3200.0, 0.32, 0.0, 200.0, 0, 0, 0.0),
        // HFCs (hydrofluorocarbons)
        HalocarbonSpecies::new("HFC-23", 228.0, 0.18, 0.0, 70.0, 0, 0, 0.0),
        HalocarbonSpecies::new("HFC-32", 5.4, 0.11, 0.0, 52.0, 0, 0, 0.0),
        HalocarbonSpecies::new("HFC-43-10mee", 17.0, 0.359, 0.0, 252.0, 0, 0, 0.0),
        HalocarbonSpecies::new("HFC-125", 31.0, 0.23, 0.0, 120.0, 0, 0, 0.0),
        HalocarbonSpecies::new("HFC-134a", 14.0, 0.16, 0.0, 102.0, 0, 0, 0.0),
        HalocarbonSpecies::new("HFC-143a", 51.0, 0.16, 0.0, 84.0, 0, 0, 0.0),
        HalocarbonSpecies::new("HFC-152a", 1.6, 0.10, 0.0, 66.0, 0, 0, 0.0),
        HalocarbonSpecies::new("HFC-227ea", 36.0, 0.26, 0.0, 170.0, 0, 0, 0.0),
        HalocarbonSpecies::new("HFC-236fa", 213.0, 0.24, 0.0, 152.0, 0, 0, 0.0),
        HalocarbonSpecies::new("HFC-245fa", 7.9, 0.24, 0.0, 134.0, 0, 0, 0.0),
        HalocarbonSpecies::new("HFC-365mfc", 8.9, 0.22, 0.0, 148.0, 0, 0, 0.0),
        // Other F-gases
        HalocarbonSpecies::new("NF3", 569.0, 0.20, 0.0, 71.0, 0, 0, 0.0),
        HalocarbonSpecies::new("SF6", 850.0, 0.57, 0.0, 146.0, 0, 0, 0.0),
        HalocarbonSpecies::new("SO2F2", 36.0, 0.20, 0.0, 102.0, 0, 0, 0.0),
    ]
}

/// Create default Montreal Protocol gas species list.
///
/// Species data from WMO Scientific Assessment and MAGICC7 defaults.
/// Fractional release factors are species-specific values representing
/// the fraction released in the stratosphere.
fn default_montreal_gases() -> Vec<HalocarbonSpecies> {
    vec![
        // CFCs
        HalocarbonSpecies::new("CFC-11", 52.0, 0.295, 0.0, 137.4, 3, 0, 0.47),
        HalocarbonSpecies::new("CFC-12", 102.0, 0.364, 0.0, 120.9, 2, 0, 0.23),
        HalocarbonSpecies::new("CFC-113", 93.0, 0.30, 0.0, 187.4, 3, 0, 0.29),
        HalocarbonSpecies::new("CFC-114", 189.0, 0.31, 0.0, 170.9, 2, 0, 0.12),
        HalocarbonSpecies::new("CFC-115", 540.0, 0.20, 0.0, 154.5, 1, 0, 0.04),
        // HCFCs
        HalocarbonSpecies::new("HCFC-22", 11.9, 0.21, 0.0, 86.5, 1, 0, 0.13),
        HalocarbonSpecies::new("HCFC-141b", 9.4, 0.16, 0.0, 116.9, 2, 0, 0.34),
        HalocarbonSpecies::new("HCFC-142b", 18.0, 0.19, 0.0, 100.5, 1, 0, 0.17),
        // Other chlorinated
        HalocarbonSpecies::new("CH3CCl3", 5.0, 0.07, 0.0, 133.4, 3, 0, 0.67),
        HalocarbonSpecies::new("CCl4", 32.0, 0.174, 0.0, 153.8, 4, 0, 0.56),
        HalocarbonSpecies::new("CH3Cl", 0.9, 0.004, 500.0, 50.5, 1, 0, 0.44),
        HalocarbonSpecies::new("CH2Cl2", 0.5, 0.028, 0.0, 84.9, 2, 0, 0.0),
        HalocarbonSpecies::new("CHCl3", 0.5, 0.07, 0.0, 119.4, 3, 0, 0.0),
        // Brominated
        HalocarbonSpecies::new("CH3Br", 0.8, 0.004, 5.0, 94.9, 0, 1, 0.60),
        HalocarbonSpecies::new("Halon-1211", 16.0, 0.29, 0.0, 165.4, 1, 1, 0.62),
        HalocarbonSpecies::new("Halon-1301", 72.0, 0.30, 0.0, 148.9, 0, 1, 0.28),
        HalocarbonSpecies::new("Halon-2402", 28.0, 0.31, 0.0, 259.8, 0, 2, 0.65),
        HalocarbonSpecies::new("Halon-1202", 2.5, 0.27, 0.0, 209.8, 0, 2, 0.62),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() {
        let params = HalocarbonParameters::default();

        // Check species counts
        assert_eq!(params.fgases.len(), 23, "Should have 23 F-gas species");
        assert_eq!(
            params.montreal_gases.len(),
            18,
            "Should have 18 Montreal gas species"
        );

        // Check global parameters
        assert!((params.br_multiplier - 60.0).abs() < 1e-10);
        assert!((params.eesc_delay - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_fgas_properties() {
        let params = HalocarbonParameters::default();

        // CF4 should have very long lifetime
        let cf4 = params.get_species("CF4").expect("CF4 should exist");
        assert!((cf4.lifetime - 50000.0).abs() < 1e-10);
        assert_eq!(cf4.n_cl, 0);
        assert_eq!(cf4.n_br, 0);
        assert!((cf4.fractional_release - 0.0).abs() < 1e-10);

        // HFC-134a commonly used species
        let hfc134a = params
            .get_species("HFC-134a")
            .expect("HFC-134a should exist");
        assert!((hfc134a.lifetime - 14.0).abs() < 1e-10);
        assert!((hfc134a.radiative_efficiency - 0.16).abs() < 1e-10);
    }

    #[test]
    fn test_montreal_gas_properties() {
        let params = HalocarbonParameters::default();

        // CFC-11 - the reference species for EESC
        let cfc11 = params.get_species("CFC-11").expect("CFC-11 should exist");
        assert_eq!(cfc11.n_cl, 3);
        assert_eq!(cfc11.n_br, 0);
        assert!((cfc11.fractional_release - 0.47).abs() < 1e-10);

        // Halon-1301 - contains bromine
        let halon1301 = params
            .get_species("Halon-1301")
            .expect("Halon-1301 should exist");
        assert_eq!(halon1301.n_cl, 0);
        assert_eq!(halon1301.n_br, 1);
        assert!(halon1301.fractional_release > 0.0);
    }

    #[test]
    fn test_emission_conversion_factor() {
        let params = HalocarbonParameters::default();

        // Test for CFC-11 (molecular weight 137.4)
        let conv = params.emission_to_concentration_factor(137.4);

        // Should be positive and reasonable magnitude
        assert!(conv > 0.0, "Conversion factor should be positive");

        // Calculate expected value:
        // 1 kt = 1e9 g emissions
        // Atmospheric mass = 5.133e9 Tg * 1e12 g/Tg = 5.133e21 g
        // Moles of CFC-11 = 1e9 g / 137.4 g/mol = 7.28e6 mol
        // Moles of air = 5.133e21 g / 28.97 g/mol = 1.77e20 mol
        // Mixing ratio = 7.28e6 / (1.77e20 * 0.949) = 4.3e-14 = 43 ppt
        // So factor ~ 0.04 ppt per kt/yr
        assert!(
            conv > 0.01 && conv < 0.1,
            "Conversion factor should be ~0.04 for CFC-11: {}",
            conv
        );
    }

    #[test]
    fn test_all_species_iterator() {
        let params = HalocarbonParameters::default();

        let all_count = params.all_species().count();
        assert_eq!(all_count, params.fgases.len() + params.montreal_gases.len());
    }

    #[test]
    fn test_fgases_have_zero_release() {
        let params = HalocarbonParameters::default();

        for species in &params.fgases {
            assert!(
                species.fractional_release.abs() < 1e-10,
                "F-gas {} should have zero fractional release",
                species.name
            );
        }
    }

    #[test]
    fn test_ch3cl_preindustrial() {
        // CH3Cl has natural sources and a pre-industrial concentration
        let params = HalocarbonParameters::default();
        let ch3cl = params.get_species("CH3Cl").expect("CH3Cl should exist");
        assert!(
            ch3cl.concentration_pi > 0.0,
            "CH3Cl should have non-zero pre-industrial concentration"
        );
    }

    #[test]
    fn test_ch3br_preindustrial() {
        // CH3Br also has natural sources
        let params = HalocarbonParameters::default();
        let ch3br = params.get_species("CH3Br").expect("CH3Br should exist");
        assert!(
            ch3br.concentration_pi > 0.0,
            "CH3Br should have non-zero pre-industrial concentration"
        );
    }
}
