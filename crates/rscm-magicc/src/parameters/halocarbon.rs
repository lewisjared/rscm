//! Halocarbon Chemistry Parameters
//!
//! Parameters for MAGICC7's F-gases and Montreal Protocol gases.
//! Species defaults are MAGICC7 v7.5.3's `FGAS_*` and `MHALO_*` configuration.

use crate::climate::lamcalc::compute_qfrac;
use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Atomic masses (g/mol) of C, H, N, O, F, S, Cl and Br, as in MAGICC7.
const ATOMIC_MASSES: [FloatValue; 8] = [
    12.001, 1.0079, 14.007, 15.999, 18.9984, 32.06, 35.453, 79.904,
];

/// MAGICC7 sums atomic masses as C, Cl, F, Br, H, S, O, N.
const MOLAR_MASS_ORDER: [usize; 8] = [0, 6, 4, 7, 1, 5, 3, 2];

/// A halocarbon species.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HalocarbonSpecies {
    /// Species name, as used in variable names.
    pub name: String,

    /// Unperturbed total lifetime (yr).
    /// MAGICC7: FGAS_TAU_TOT, MHALO_TAU_TOT
    pub total_lifetime: FloatValue,

    /// Unperturbed OH lifetime (yr), or 0 if not destroyed by OH.
    /// MAGICC7: FGAS_TAU_OH, MHALO_TAU_OH
    pub oh_lifetime: FloatValue,

    /// Unperturbed stratospheric lifetime (yr), or 0 if not destroyed there.
    /// MAGICC7: FGAS_TAU_STRAT, MHALO_TAU_STRAT
    pub stratospheric_lifetime: FloatValue,

    /// Radiative efficiency (W/m² per ppb).
    pub radiative_efficiency: FloatValue,

    /// Initial and reference concentration (ppt), the 1750 value of MAGICC7's concentration file.
    pub concentration_pi: FloatValue,

    /// Molecular mass (g/mol).
    pub molecular_mass: FloatValue,

    /// Chlorine atoms per molecule.
    /// MAGICC7: FGAS_CL_ATOMS, MHALO_CL_ATOMS
    pub cl_atoms: u8,

    /// Bromine atoms per molecule.
    /// MAGICC7: FGAS_BR_ATOMS, MHALO_BR_ATOMS
    pub br_atoms: u8,

    /// Fractional release factor in the stratosphere.
    pub fractional_release: FloatValue,

    /// Fraction of the atmosphere the species mixes into.
    /// MAGICC7: FGAS_EFF_MIXBOXSIZE, MHALO_EFF_MIXBOXSIZE
    #[serde(default = "default_mixing_box_size")]
    pub mixing_box_size: FloatValue,
}

impl HalocarbonSpecies {
    /// Lifetime not attributed to OH or the stratosphere (yr), or `None` if there is none.
    pub fn other_lifetime(&self) -> Option<FloatValue> {
        let mut rate = 1.0 / self.total_lifetime;
        if self.oh_lifetime > 0.0 {
            rate -= 1.0 / self.oh_lifetime;
        }
        if self.stratospheric_lifetime > 0.0 {
            rate -= 1.0 / self.stratospheric_lifetime;
        }
        (rate.abs() > 1e-12).then(|| 1.0 / rate)
    }
}

/// Parameters for halocarbon chemistry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct HalocarbonChemistryParameters {
    /// F-gases, which do not deplete ozone.
    pub fgases: Vec<HalocarbonSpecies>,

    /// Montreal Protocol gases.
    pub montreal_gases: Vec<HalocarbonSpecies>,

    /// Ozone depletion by bromine relative to chlorine (`STRATOZ_BR_VS_CL_SCALE`).
    pub br_cl_ratio: FloatValue,

    /// EESC scaling, the CFC-11 release factor (`MHALO_RELEASEFACTORC11`).
    pub cfc11_release_factor: FloatValue,

    /// Years from troposphere to stratosphere (`GEN_EESC_STRATMIXDELAY`).
    pub stratospheric_delay: usize,

    /// Molar mass of air (g/mol, `GEN_AIR_GRAMMPROMOL`).
    pub air_molar_mass: FloatValue,

    /// Mass of the atmosphere (1e21 g, `GEN_ATM_TOTMASS_1E21GRAMM`).
    pub atmosphere_mass: FloatValue,

    /// Scale OH lifetimes with the CH4 OH lifetime (`*_USE_TAUOH_VAR`).
    pub include_variable_oh_lifetime: bool,

    /// Unperturbed CH4 OH lifetime (yr), the reference for OH scaling.
    /// MAGICC7: from CH4_TAUTOT_INIT and the other CH4 lifetimes
    pub initial_ch4_oh_lifetime: FloatValue,

    /// Scale stratospheric lifetimes with warming (`*_USE_TAUSTRAT_VAR`).
    pub include_variable_stratospheric_lifetime: bool,

    /// Year from which warming speeds up the meridional circulation
    /// (`GEN_CHANGE_MERIDIONALFLUX_YR`).
    pub meridional_flux_reference_year: FloatValue,

    /// Change in meridional flux per degree of warming (`GEN_MERIDFLUX_CHNGPERDEG`).
    pub meridional_flux_change_per_degree: FloatValue,

    /// Sensitivity of stratospheric lifetimes to meridional flux
    /// (`*_TAUSTRAT_SENS2MERIDFLUX`).
    pub stratospheric_lifetime_sensitivity: FloatValue,

    /// Last year of F-gas concentrations prescribed by `Prescribed Atmospheric Concentrations|<species>`,
    /// or `None` to calculate them from emissions throughout.
    /// MAGICC7: FGAS_SWITCHFROMCONC2EMIS_YEAR
    pub fgas_switch_year: Option<FloatValue>,

    /// As `fgas_switch_year`, for the Montreal Protocol gases.
    /// MAGICC7: MHALO_SWITCHFROMCONC2EMIS_YEAR
    pub montreal_gases_switch_year: Option<FloatValue>,

    /// Regional pattern of F-gas forcing, which makes `Effective Radiative Forcing|F-Gases`
    /// a four-box output.
    pub fgas_regions: Option<FgasRegions>,
}

/// Regional pattern of F-gas forcing.
///
/// Species with lifetimes of at least `upper_lifetime` follow `regional_pattern` normalised globally.
/// Those below `lower_lifetime` follow it normalised within each hemisphere,
/// scaled by each hemisphere's share of fossil CO2 emissions, as MAGICC7 does.
/// Lifetimes in between mix the two linearly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct FgasRegions {
    /// Pattern (NH ocean, NH land, SH ocean, SH land).
    /// MAGICC7: RF_REGIONS_FGAS
    pub regional_pattern: [FloatValue; 4],
    /// unit: yr
    /// MAGICC7: FGAS_RF_REGIONSCALING_LOWERTAU
    pub lower_lifetime: FloatValue,
    /// unit: yr
    /// MAGICC7: FGAS_RF_REGIONSCALING_UPPERTAU
    pub upper_lifetime: FloatValue,
    /// Northern Hemisphere land fraction.
    pub nh_land_fraction: FloatValue,
    /// Southern Hemisphere land fraction.
    pub sh_land_fraction: FloatValue,
}

impl Default for FgasRegions {
    fn default() -> Self {
        Self {
            regional_pattern: [0.16695, 0.16218, 0.15508, 0.1463],
            lower_lifetime: 1.0,
            upper_lifetime: 8.0,
            nh_land_fraction: 0.42,
            sh_land_fraction: 0.21,
        }
    }
}

impl FgasRegions {
    /// Box forcing per unit global forcing of a species with lifetime `tau`.
    ///
    /// `fossil_co2` is the global fossil CO2 emission, split equally between the land boxes
    /// as MAGICC7 splits global emissions. Without emissions the pattern is global.
    pub fn pattern(&self, tau: FloatValue, fossil_co2: FloatValue) -> [FloatValue; 4] {
        let area = [
            (1.0 - self.nh_land_fraction) / 2.0,
            self.nh_land_fraction / 2.0,
            (1.0 - self.sh_land_fraction) / 2.0,
            self.sh_land_fraction / 2.0,
        ];
        let global = compute_qfrac(&self.regional_pattern, &area);
        if tau > self.upper_lifetime || fossil_co2 == 0.0 {
            return global;
        }
        let emissions = [0.0, fossil_co2 / 2.0, 0.0, fossil_co2 / 2.0];
        let total = emissions.iter().fold(0.0, |sum, e| sum + e);
        let mut short_lived = [0.0; 4];
        for ocean in [0, 2] {
            let land = ocean + 1;
            let mean = area[ocean] * global[ocean] + area[land] * global[land];
            let share = emissions[ocean] + emissions[land];
            for i in [ocean, land] {
                short_lived[i] = global[i] / mean * share / total;
            }
        }
        if tau < self.lower_lifetime {
            return short_lived;
        }
        let w = (tau - self.lower_lifetime) / (self.upper_lifetime - self.lower_lifetime);
        std::array::from_fn(|i| (global[i] - short_lived[i]) * w + short_lived[i])
    }
}

impl HalocarbonChemistryParameters {
    /// Concentration change (ppt) per kt of emissions.
    pub fn emission_to_concentration_factor(&self, species: &HalocarbonSpecies) -> FloatValue {
        self.air_molar_mass
            / (self.atmosphere_mass * species.molecular_mass * species.mixing_box_size)
    }

    /// All species, F-gases first.
    pub fn all_species(&self) -> impl Iterator<Item = &HalocarbonSpecies> {
        self.fgases.iter().chain(self.montreal_gases.iter())
    }

    /// A species by name.
    pub fn get_species(&self, name: &str) -> Option<&HalocarbonSpecies> {
        self.all_species().find(|s| s.name == name)
    }
}

impl Default for HalocarbonChemistryParameters {
    fn default() -> Self {
        Self {
            fgases: default_fgases(),
            montreal_gases: default_montreal_gases(),
            br_cl_ratio: 62.5,
            cfc11_release_factor: 0.75,
            stratospheric_delay: 3,
            air_molar_mass: 28.984,
            atmosphere_mass: 5.133,
            include_variable_oh_lifetime: true,
            // 1 / (1/CH4_TAUTOT_INIT - 1/CH4_TAUSOIL - 1/CH4_TAUSTRAT - 1/CH4_TAUTROPCL)
            initial_ch4_oh_lifetime: 1.0 / (1.0 / 9.9474 - 1.0 / 150.0 - 1.0 / 120.0 - 1.0 / 200.0),
            include_variable_stratospheric_lifetime: true,
            meridional_flux_reference_year: 1980.0,
            meridional_flux_change_per_degree: 0.15,
            stratospheric_lifetime_sensitivity: 0.3,
            fgas_switch_year: None,
            montreal_gases_switch_year: None,
            fgas_regions: None,
        }
    }
}

fn default_mixing_box_size() -> FloatValue {
    0.949
}

/// Builds a species from lifetimes `[total, OH, stratospheric]` and atom counts
/// `[C, H, N, O, F, S, Cl, Br]`.
fn species(
    name: &str,
    [total_lifetime, oh_lifetime, stratospheric_lifetime]: [FloatValue; 3],
    radiative_efficiency: FloatValue,
    concentration_pi: FloatValue,
    atoms: [u8; 8],
    fractional_release: FloatValue,
) -> HalocarbonSpecies {
    HalocarbonSpecies {
        name: name.to_string(),
        total_lifetime,
        oh_lifetime,
        stratospheric_lifetime,
        radiative_efficiency,
        concentration_pi,
        molecular_mass: MOLAR_MASS_ORDER
            .iter()
            .map(|&i| FloatValue::from(atoms[i]) * ATOMIC_MASSES[i])
            .sum(),
        cl_atoms: atoms[6],
        br_atoms: atoms[7],
        fractional_release,
        mixing_box_size: default_mixing_box_size(),
    }
}

/// MAGICC7 default F-gases (`FGAS_*`).
fn default_fgases() -> Vec<HalocarbonSpecies> {
    vec![
        species(
            "CF4",
            [50000.0, 0.0, 0.0],
            0.09,
            34.05,
            [1, 0, 0, 0, 4, 0, 0, 0],
            0.0,
        ),
        species(
            "C2F6",
            [10000.0, 0.0, 0.0],
            0.25,
            0.0,
            [2, 0, 0, 0, 6, 0, 0, 0],
            0.0,
        ),
        species(
            "C3F8",
            [2600.0, 0.0, 0.0],
            0.28,
            0.0,
            [3, 0, 0, 0, 8, 0, 0, 0],
            0.0,
        ),
        species(
            "C4F10",
            [2600.0, 0.0, 0.0],
            0.36,
            0.0,
            [4, 0, 0, 0, 10, 0, 0, 0],
            0.0,
        ),
        species(
            "C5F12",
            [4100.0, 0.0, 0.0],
            0.41,
            0.0,
            [5, 0, 0, 0, 12, 0, 0, 0],
            0.0,
        ),
        species(
            "C6F14",
            [3100.0, 0.0, 0.0],
            0.44,
            0.0,
            [6, 0, 0, 0, 14, 0, 0, 0],
            0.0,
        ),
        species(
            "C7F16",
            [3000.0, 0.0, 0.0],
            0.5,
            0.0,
            [7, 0, 0, 0, 16, 0, 0, 0],
            0.0,
        ),
        species(
            "C8F18",
            [3000.0, 0.0, 0.0],
            0.55,
            0.0,
            [8, 0, 0, 0, 18, 0, 0, 0],
            0.0,
        ),
        species(
            "cC4F8",
            [3200.0, 0.0, 0.0],
            0.32,
            0.0,
            [4, 0, 0, 0, 8, 0, 0, 0],
            0.0,
        ),
        species(
            "HFC23",
            [228.0, 243.0, 4420.0],
            0.18,
            0.0,
            [1, 1, 0, 0, 3, 0, 0, 0],
            0.0,
        ),
        species(
            "HFC32",
            [5.4, 5.5, 124.0],
            0.11,
            0.0,
            [1, 2, 0, 0, 2, 0, 0, 0],
            0.0,
        ),
        species(
            "HFC4310mee",
            [17.0, 17.9, 365.0],
            0.359,
            0.0,
            [5, 2, 0, 0, 10, 0, 0, 0],
            0.0,
        ),
        species(
            "HFC125",
            [31.0, 32.0, 351.0],
            0.23,
            0.0,
            [2, 1, 0, 0, 5, 0, 0, 0],
            0.0,
        ),
        species(
            "HFC134a",
            [14.0, 14.1, 267.0],
            0.16,
            0.0,
            [2, 2, 0, 0, 4, 0, 0, 0],
            0.0,
        ),
        species(
            "HFC143a",
            [51.0, 57.0, 612.0],
            0.16,
            0.0,
            [2, 3, 0, 0, 3, 0, 0, 0],
            0.0,
        ),
        species(
            "HFC152a",
            [1.6, 1.55, 39.0],
            0.1,
            0.0,
            [2, 4, 0, 0, 2, 0, 0, 0],
            0.0,
        ),
        species(
            "HFC227ea",
            [36.0, 37.5, 673.0],
            0.26,
            0.0,
            [3, 1, 0, 0, 7, 0, 0, 0],
            0.0,
        ),
        species(
            "HFC236fa",
            [213.0, 253.0, 1350.0],
            0.24,
            0.0,
            [3, 2, 0, 0, 6, 0, 0, 0],
            0.0,
        ),
        species(
            "HFC245fa",
            [7.9, 8.2, 149.0],
            0.24,
            0.0,
            [3, 3, 0, 0, 5, 0, 0, 0],
            0.0,
        ),
        species(
            "HFC365mfc",
            [8.9, 9.3, 190.0],
            0.22,
            0.0,
            [4, 5, 0, 0, 5, 0, 0, 0],
            0.0,
        ),
        species(
            "NF3",
            [569.0, 0.0, 740.0],
            0.2,
            0.0,
            [0, 0, 1, 0, 3, 0, 0, 0],
            0.0,
        ),
        species(
            "SF6",
            [850.0, 0.0, 0.0],
            0.57,
            0.0,
            [0, 0, 0, 0, 6, 1, 0, 0],
            0.0,
        ),
        species(
            "SO2F2",
            [36.0, 300.0, 630.0],
            0.2,
            0.0,
            [0, 0, 0, 2, 2, 1, 0, 0],
            0.0,
        ),
    ]
}

/// MAGICC7 default Montreal Protocol gases (`MHALO_*`).
fn default_montreal_gases() -> Vec<HalocarbonSpecies> {
    vec![
        species(
            "CFC11",
            [52.0, 0.0, 55.0],
            0.295,
            0.0,
            [1, 0, 0, 0, 1, 0, 3, 0],
            0.47,
        ),
        species(
            "CFC12",
            [102.0, 0.0, 103.0],
            0.364,
            0.0,
            [1, 0, 0, 0, 2, 0, 2, 0],
            0.23,
        ),
        species(
            "CFC113",
            [93.0, 0.0, 94.5],
            0.3,
            0.0,
            [2, 0, 0, 0, 3, 0, 3, 0],
            0.29,
        ),
        species(
            "CFC114",
            [189.0, 0.0, 191.0],
            0.31,
            0.0,
            [2, 0, 0, 0, 4, 0, 2, 0],
            0.12,
        ),
        species(
            "CFC115",
            [540.0, 0.0, 664.0],
            0.2,
            0.0,
            [2, 0, 0, 0, 5, 0, 1, 0],
            0.04,
        ),
        species(
            "HCFC22",
            [11.9, 13.0, 161.0],
            0.21,
            0.0,
            [1, 1, 0, 0, 2, 0, 1, 0],
            0.13,
        ),
        species(
            "HCFC141b",
            [9.4, 10.7, 72.3],
            0.16,
            0.0,
            [2, 3, 0, 0, 1, 0, 2, 0],
            0.34,
        ),
        species(
            "HCFC142b",
            [18.0, 19.3, 212.0],
            0.19,
            0.0,
            [2, 3, 0, 0, 2, 0, 1, 0],
            0.17,
        ),
        species(
            "CH3CCl3",
            [5.0, 6.1, 38.0],
            0.07,
            0.0,
            [2, 3, 0, 0, 0, 0, 3, 0],
            0.67,
        ),
        species(
            "CCl4",
            [32.0, 0.0, 44.0],
            0.174,
            0.0250004,
            [1, 0, 0, 0, 0, 0, 4, 0],
            0.56,
        ),
        species(
            "CH3Cl",
            [0.9, 1.57, 30.4],
            0.004,
            457.0,
            [1, 3, 0, 0, 0, 0, 1, 0],
            0.44,
        ),
        species(
            "CH2Cl2",
            [0.5, 0.5, 0.0],
            0.028,
            6.91276,
            [1, 2, 0, 0, 0, 0, 2, 0],
            0.0,
        ),
        species(
            "CHCl3",
            [0.5, 0.5, 0.0],
            0.07,
            6.0,
            [1, 1, 0, 0, 0, 0, 3, 0],
            0.0,
        ),
        species(
            "CH3Br",
            [0.8, 1.8, 26.3],
            0.004,
            5.3,
            [1, 3, 0, 0, 0, 0, 0, 1],
            0.6,
        ),
        species(
            "Halon1211",
            [16.0, 0.0, 41.0],
            0.29,
            0.00444657,
            [1, 0, 0, 0, 2, 0, 1, 1],
            0.62,
        ),
        species(
            "Halon1301",
            [72.0, 0.0, 73.5],
            0.3,
            0.0,
            [1, 0, 0, 0, 3, 0, 0, 1],
            0.28,
        ),
        species(
            "Halon2402",
            [28.0, 0.0, 41.0],
            0.31,
            0.0,
            [2, 0, 0, 0, 4, 0, 0, 2],
            0.65,
        ),
        species(
            "Halon1202",
            [2.5, 0.0, 36.0],
            0.27,
            0.0,
            [1, 0, 0, 0, 2, 0, 0, 2],
            0.62,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_species_counts() {
        let params = HalocarbonChemistryParameters::default();
        assert_eq!(params.fgases.len(), 23);
        assert_eq!(params.montreal_gases.len(), 18);
    }

    #[test]
    fn test_molecular_mass_from_atoms() {
        let params = HalocarbonChemistryParameters::default();
        let cfc11 = params.get_species("CFC11").unwrap();
        assert!((cfc11.molecular_mass - (12.001 + 3.0 * 35.453 + 18.9984)).abs() < 1e-10);
    }

    #[test]
    fn test_other_lifetime_balances_total() {
        let params = HalocarbonChemistryParameters::default();
        let hcfc22 = params.get_species("HCFC22").unwrap();
        let rate = 1.0 / hcfc22.oh_lifetime
            + 1.0 / hcfc22.stratospheric_lifetime
            + 1.0 / hcfc22.other_lifetime().unwrap();
        assert!((1.0 / rate - hcfc22.total_lifetime).abs() < 1e-10);
        let cfc11 = params.get_species("CFC11").unwrap();
        assert!(cfc11.other_lifetime().is_some());
        let sf6 = params.get_species("SF6").unwrap();
        assert!((sf6.other_lifetime().unwrap() - 850.0).abs() < 1e-9);
    }

    fn equal_area_regions() -> FgasRegions {
        FgasRegions {
            regional_pattern: [1.0, 2.0, 3.0, 4.0],
            lower_lifetime: 1.0,
            upper_lifetime: 8.0,
            nh_land_fraction: 0.5,
            sh_land_fraction: 0.5,
        }
    }

    fn assert_close(actual: [FloatValue; 4], expected: [FloatValue; 4]) {
        for i in 0..4 {
            assert!((actual[i] - expected[i]).abs() < 1e-12, "{actual:?}");
        }
    }

    #[test]
    fn test_fgas_pattern_long_lived_is_global() {
        // Equal box areas: the pattern over its global mean of 2.5
        let regions = equal_area_regions();
        let global = [0.4, 0.8, 1.2, 1.6];
        assert_close(regions.pattern(8.0, 1.0), global);
        assert_close(regions.pattern(100.0, 1.0), global);
        assert_close(regions.pattern(1.0, 0.0), global);
    }

    #[test]
    fn test_fgas_pattern_short_lived_is_hemispheric() {
        // Each hemisphere over its own mean, 1.5 in the north and 3.5 in the south
        let regions = equal_area_regions();
        let hemispheric = [2.0 / 3.0, 4.0 / 3.0, 6.0 / 7.0, 8.0 / 7.0];
        assert_close(regions.pattern(1.0, 1.0), hemispheric);
        assert_close(regions.pattern(0.1, 5.0), hemispheric);
    }

    #[test]
    fn test_fgas_pattern_mixes_linearly_between_lifetimes() {
        let regions = equal_area_regions();
        let mid = regions.pattern(4.5, 1.0);
        let (global, hemispheric) = (regions.pattern(8.0, 1.0), regions.pattern(1.0, 1.0));
        assert_close(
            mid,
            std::array::from_fn(|i| (global[i] + hemispheric[i]) / 2.0),
        );
    }

    #[test]
    fn test_fgas_pattern_keeps_global_mean() {
        let regions = FgasRegions::default();
        let area = [0.29, 0.21, 0.395, 0.105];
        for tau in [0.5, 1.0, 3.0, 8.0, 50.0] {
            let mean: FloatValue = (0..4).map(|i| regions.pattern(tau, 1.0)[i] * area[i]).sum();
            assert!((mean - 1.0).abs() < 1e-12, "tau={tau}: {mean}");
        }
    }
}
