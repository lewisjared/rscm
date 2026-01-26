//! Halocarbon Chemistry Component
//!
//! Simulates atmospheric chemistry of halocarbons (F-gases and Montreal Protocol gases)
//! with exponential decay lifetimes.
//!
//! # What This Component Does
//!
//! 1. Calculates atmospheric concentrations for each halocarbon species using
//!    exponential decay with species-specific lifetimes
//! 2. Computes total radiative forcing from all halocarbons
//! 3. Calculates Equivalent Effective Stratospheric Chlorine (EESC) for ozone
//!    depletion calculations
//!
//! # Inputs
//!
//! For each species:
//! - `Emissions|<species>` (kt/yr) - Emissions for each halocarbon species
//!
//! # Outputs
//!
//! For each species:
//! - `Concentration|<species>` (ppt) - Atmospheric concentration (state variable)
//!
//! Aggregates:
//! - `Forcing|Halocarbons` (W/m^2) - Total radiative forcing from all halocarbons
//! - `Forcing|F-gases` (W/m^2) - Forcing from F-gases only
//! - `Forcing|Montreal Gases` (W/m^2) - Forcing from Montreal gases only
//! - `EESC` (ppt) - Equivalent Effective Stratospheric Chlorine
//!
//! # Differences from MAGICC7 Module 03
//!
//! This is a simplified implementation:
//!
//! - **Variable lifetimes**: MAGICC7 supports time-varying lifetimes based on OH
//!   and stratospheric changes. Here, lifetimes are fixed constants.
//! - **Regional forcing**: MAGICC7 distributes forcing across boxes based on
//!   species lifetime. Here, forcing is scalar (global mean).
//! - **EESC delay**: MAGICC7 uses concentration from 3 years prior for EESC.
//!   Here, we use current concentration (delay should be handled externally).
//! - **Inverse emissions**: MAGICC7 can derive emissions from concentration
//!   trajectories. Not implemented here.
//! - **Bank dynamics**: Equipment banks are not modelled; they must be included
//!   in the emissions input.

use crate::parameters::{HalocarbonParameters, HalocarbonSpecies};
use rscm_core::component::{Component, InputState, OutputState, RequirementDefinition};
use rscm_core::errors::RSCMResult;
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Halocarbon atmospheric chemistry component
///
/// Implements exponential decay for all halocarbon species and calculates:
/// - Species concentrations from emissions and decay
/// - Total radiative forcing (linear in concentration)
/// - EESC for stratospheric ozone calculations
///
/// # Exponential Decay
///
/// For each species, the concentration evolves according to:
///
/// $$\frac{dC}{dt} = E \cdot \text{conv} - \frac{C}{\tau}$$
///
/// Analytical solution for one timestep $\Delta t$:
///
/// $$C(t+\Delta t) = C(t) \cdot e^{-\Delta t/\tau} + E \cdot \text{conv} \cdot \tau \cdot (1 - e^{-\Delta t/\tau})$$
///
/// where conv is the emission-to-concentration conversion factor.
///
/// # Radiative Forcing
///
/// $$RF_i = (C_i - C_{i,PI}) \cdot \eta_i / 1000$$
///
/// where $\eta_i$ is the radiative efficiency in W/m² per ppb, divided by 1000
/// to convert to per ppt.
///
/// # EESC Calculation
///
/// $$\text{EESC} = \sum_i C_i \cdot (n_{Cl,i} + \alpha_{Br} \cdot n_{Br,i}) \cdot f_{release,i}$$
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HalocarbonChemistry {
    parameters: HalocarbonParameters,
}

impl HalocarbonChemistry {
    /// Create a new halocarbon chemistry component with default parameters
    pub fn new() -> Self {
        Self::from_parameters(HalocarbonParameters::default())
    }

    /// Create a new halocarbon chemistry component from parameters
    pub fn from_parameters(parameters: HalocarbonParameters) -> Self {
        Self { parameters }
    }

    /// Get the parameters
    pub fn parameters(&self) -> &HalocarbonParameters {
        &self.parameters
    }

    /// Get emissions variable name for a species
    fn emissions_name(species: &str) -> String {
        format!("Emissions|{}", species)
    }

    /// Get concentration variable name for a species
    fn concentration_name(species: &str) -> String {
        format!("Atmospheric Concentration|{}", species)
    }

    /// Calculate concentration after one timestep using analytical exponential decay
    ///
    /// Returns the new concentration in ppt.
    pub fn decay_species(
        &self,
        species: &HalocarbonSpecies,
        concentration: FloatValue,
        emissions: FloatValue,
        dt: FloatValue,
    ) -> FloatValue {
        // Exponential decay factor
        let decay = (-dt / species.lifetime).exp();

        // Convert emissions from kt/yr to ppt/yr
        let conv = self
            .parameters
            .emission_to_concentration_factor(species.molecular_weight);
        let emissions_ppt = emissions * conv;

        // Analytical solution: C_new = C * exp(-dt/tau) + E * tau * (1 - exp(-dt/tau))
        concentration * decay + emissions_ppt * species.lifetime * (1.0 - decay)
    }

    /// Calculate radiative forcing for a single species (W/m²)
    pub fn species_forcing(
        &self,
        species: &HalocarbonSpecies,
        concentration: FloatValue,
    ) -> FloatValue {
        let delta_conc = concentration - species.concentration_pi;
        // radiative_efficiency is in W/m² per ppb, divide by 1000 for ppt
        delta_conc * species.radiative_efficiency / 1000.0
    }

    /// Calculate total radiative forcing from all species
    pub fn calculate_total_forcing(
        &self,
        concentrations: &HashMap<String, FloatValue>,
    ) -> FloatValue {
        self.parameters
            .all_species()
            .map(|species| {
                let conc = *concentrations
                    .get(&species.name)
                    .unwrap_or(&species.concentration_pi);
                self.species_forcing(species, conc)
            })
            .sum()
    }

    /// Calculate forcing from F-gases only
    pub fn calculate_fgas_forcing(
        &self,
        concentrations: &HashMap<String, FloatValue>,
    ) -> FloatValue {
        self.parameters
            .fgases
            .iter()
            .map(|species| {
                let conc = *concentrations
                    .get(&species.name)
                    .unwrap_or(&species.concentration_pi);
                self.species_forcing(species, conc)
            })
            .sum()
    }

    /// Calculate forcing from Montreal gases only
    pub fn calculate_montreal_forcing(
        &self,
        concentrations: &HashMap<String, FloatValue>,
    ) -> FloatValue {
        self.parameters
            .montreal_gases
            .iter()
            .map(|species| {
                let conc = *concentrations
                    .get(&species.name)
                    .unwrap_or(&species.concentration_pi);
                self.species_forcing(species, conc)
            })
            .sum()
    }

    /// Calculate Equivalent Effective Stratospheric Chlorine (EESC)
    ///
    /// EESC combines chlorine and bromine loading weighted by their
    /// ozone destruction efficiency and fractional release in the stratosphere.
    ///
    /// $$\text{EESC} = \sum_i C_i \cdot (n_{Cl,i} + \alpha_{Br} \cdot n_{Br,i}) \cdot f_{release,i} / f_{CFC11}$$
    pub fn calculate_eesc(&self, concentrations: &HashMap<String, FloatValue>) -> FloatValue {
        let mut eesc = 0.0;

        for species in self.parameters.all_species() {
            let conc = *concentrations
                .get(&species.name)
                .unwrap_or(&species.concentration_pi);

            // Only species with non-zero release factor contribute to EESC
            if species.fractional_release > 0.0 {
                let halogen_loading =
                    species.n_cl as f64 + self.parameters.br_multiplier * species.n_br as f64;

                // Normalise by CFC-11 release factor
                let normalised_release =
                    species.fractional_release / self.parameters.cfc11_release_normalisation;

                eesc += conc * halogen_loading * normalised_release;
            }
        }

        eesc
    }

    /// Step all species forward in time and return new concentrations
    pub fn step_concentrations(
        &self,
        current_concentrations: &HashMap<String, FloatValue>,
        emissions: &HashMap<String, FloatValue>,
        dt: FloatValue,
    ) -> HashMap<String, FloatValue> {
        let mut new_concentrations = HashMap::new();

        for species in self.parameters.all_species() {
            let current_conc = *current_concentrations
                .get(&species.name)
                .unwrap_or(&species.concentration_pi);

            let emission = *emissions.get(&species.name).unwrap_or(&0.0);

            let new_conc = self.decay_species(species, current_conc, emission, dt);
            new_concentrations.insert(species.name.clone(), new_conc);
        }

        new_concentrations
    }
}

impl Default for HalocarbonChemistry {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde]
impl Component for HalocarbonChemistry {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        let mut defs = Vec::new();

        // Emissions inputs and concentration states for each species
        for species in self.parameters.all_species() {
            // Emissions input
            defs.push(RequirementDefinition::scalar_input(
                &Self::emissions_name(&species.name),
                "kt/yr",
            ));

            // Concentration state (reads previous, writes new)
            defs.push(RequirementDefinition::scalar_state(
                &Self::concentration_name(&species.name),
                "ppt",
            ));
        }

        // Aggregate outputs
        defs.push(RequirementDefinition::scalar_output(
            "Forcing|Halocarbons",
            "W/m^2",
        ));
        defs.push(RequirementDefinition::scalar_output(
            "Forcing|F-gases",
            "W/m^2",
        ));
        defs.push(RequirementDefinition::scalar_output(
            "Forcing|Montreal Gases",
            "W/m^2",
        ));
        defs.push(RequirementDefinition::scalar_output("EESC", "ppt"));

        defs
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let dt = t_next - t_current;

        // Read current concentrations and emissions for each species
        let mut current_concentrations = HashMap::new();
        let mut emissions = HashMap::new();

        for species in self.parameters.all_species() {
            // Get current concentration (from state) or use pre-industrial
            let conc_name = Self::concentration_name(&species.name);
            let current_conc = input_state
                .get_global(&conc_name)
                .unwrap_or(species.concentration_pi);
            current_concentrations.insert(species.name.clone(), current_conc);

            // Get emissions
            let emis_name = Self::emissions_name(&species.name);
            let emission = input_state.get_global(&emis_name).unwrap_or(0.0);
            emissions.insert(species.name.clone(), emission);
        }

        // Calculate new concentrations
        let new_concentrations = self.step_concentrations(&current_concentrations, &emissions, dt);

        // Calculate outputs
        let total_forcing = self.calculate_total_forcing(&new_concentrations);
        let fgas_forcing = self.calculate_fgas_forcing(&new_concentrations);
        let montreal_forcing = self.calculate_montreal_forcing(&new_concentrations);
        let eesc = self.calculate_eesc(&new_concentrations);

        // Build output state
        let mut output = OutputState::new();

        // Add concentration outputs for each species
        for species in self.parameters.all_species() {
            let conc_name = Self::concentration_name(&species.name);
            if let Some(&conc) = new_concentrations.get(&species.name) {
                output.insert(conc_name, StateValue::Scalar(conc));
            }
        }

        // Add aggregate outputs
        output.insert(
            "Forcing|Halocarbons".to_string(),
            StateValue::Scalar(total_forcing),
        );
        output.insert(
            "Forcing|F-gases".to_string(),
            StateValue::Scalar(fgas_forcing),
        );
        output.insert(
            "Forcing|Montreal Gases".to_string(),
            StateValue::Scalar(montreal_forcing),
        );
        output.insert("EESC".to_string(), StateValue::Scalar(eesc));

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_component() -> HalocarbonChemistry {
        HalocarbonChemistry::from_parameters(HalocarbonParameters::default())
    }

    // ===== Species Decay Tests =====

    #[test]
    fn test_exponential_decay_no_emissions() {
        let component = default_component();
        let cf4 = component
            .parameters()
            .get_species("CF4")
            .expect("CF4 should exist");

        let initial_conc = 100.0; // ppt
        let dt = 1.0; // 1 year

        let new_conc = component.decay_species(cf4, initial_conc, 0.0, dt);

        // With 50000 year lifetime, decay should be minimal
        let expected_decay = (-dt / cf4.lifetime).exp();
        let expected = initial_conc * expected_decay;

        assert!(
            (new_conc - expected).abs() < 1e-10,
            "CF4 decay without emissions: expected {}, got {}",
            expected,
            new_conc
        );

        // Should be very close to initial (long lifetime)
        assert!(
            (new_conc - initial_conc).abs() / initial_conc < 0.001,
            "CF4 should barely decay in 1 year: {} -> {}",
            initial_conc,
            new_conc
        );
    }

    #[test]
    fn test_exponential_decay_short_lived() {
        let component = default_component();
        let hfc152a = component
            .parameters()
            .get_species("HFC-152a")
            .expect("HFC-152a should exist");

        let initial_conc = 100.0; // ppt
        let dt = 1.0; // 1 year

        let new_conc = component.decay_species(hfc152a, initial_conc, 0.0, dt);

        // With 1.6 year lifetime, should decay significantly
        let expected_decay = (-dt / hfc152a.lifetime).exp();
        let expected = initial_conc * expected_decay;

        assert!(
            (new_conc - expected).abs() < 1e-8,
            "HFC-152a decay: expected {:.4}, got {:.4}",
            expected,
            new_conc
        );

        // Should decay by ~47% in 1 year (exp(-1/1.6) ≈ 0.535)
        assert!(
            new_conc < initial_conc * 0.6,
            "HFC-152a should decay significantly: {} -> {}",
            initial_conc,
            new_conc
        );
    }

    #[test]
    fn test_decay_with_emissions_equilibrium() {
        let component = default_component();
        let hfc134a = component
            .parameters()
            .get_species("HFC-134a")
            .expect("HFC-134a should exist");

        // At equilibrium: dC/dt = 0 => C_eq = E * tau * conv
        let emissions = 100.0; // kt/yr
        let conv = component
            .parameters()
            .emission_to_concentration_factor(hfc134a.molecular_weight);
        let expected_equilibrium = emissions * conv * hfc134a.lifetime;

        // Start from zero and run many timesteps to approach equilibrium
        let dt = 1.0;
        let mut conc = 0.0;
        for _ in 0..500 {
            conc = component.decay_species(hfc134a, conc, emissions, dt);
        }

        // Should be close to equilibrium
        let relative_error = ((conc - expected_equilibrium) / expected_equilibrium).abs();
        assert!(
            relative_error < 0.01,
            "Should approach equilibrium: got {:.2}, expected {:.2} (error: {:.1}%)",
            conc,
            expected_equilibrium,
            relative_error * 100.0
        );
    }

    #[test]
    fn test_zero_emissions_decays_to_zero() {
        let component = default_component();
        let cfc11 = component
            .parameters()
            .get_species("CFC-11")
            .expect("CFC-11 should exist");

        let mut conc = 250.0; // ppt (approximate current level)
        let dt = 1.0;

        // Run for 10 lifetimes
        for _ in 0..(cfc11.lifetime as usize * 10) {
            conc = component.decay_species(cfc11, conc, 0.0, dt);
        }

        // Should be negligible (< 0.1% of initial)
        assert!(conc < 0.25, "CFC-11 should decay to near zero: {}", conc);
    }

    // ===== Radiative Forcing Tests =====

    #[test]
    fn test_species_forcing_zero_at_pi() {
        let component = default_component();
        let cfc11 = component
            .parameters()
            .get_species("CFC-11")
            .expect("CFC-11 should exist");

        // At pre-industrial concentration, forcing should be zero
        let forcing = component.species_forcing(cfc11, cfc11.concentration_pi);
        assert!(
            forcing.abs() < 1e-10,
            "Forcing at PI should be zero: {}",
            forcing
        );
    }

    #[test]
    fn test_species_forcing_linear() {
        let component = default_component();
        let cfc11 = component
            .parameters()
            .get_species("CFC-11")
            .expect("CFC-11 should exist");

        let conc1 = 100.0;
        let conc2 = 200.0;

        let forcing1 = component.species_forcing(cfc11, conc1);
        let forcing2 = component.species_forcing(cfc11, conc2);

        // Forcing should scale linearly with concentration
        assert!(
            (forcing2 - 2.0 * forcing1).abs() < 1e-10,
            "Forcing should be linear: F(100)={}, F(200)={}",
            forcing1,
            forcing2
        );
    }

    #[test]
    fn test_total_forcing_calculation() {
        let component = default_component();

        // Set up concentrations with some non-zero values
        let mut concentrations = HashMap::new();
        concentrations.insert("CFC-11".to_string(), 250.0); // ~current level
        concentrations.insert("CFC-12".to_string(), 520.0); // ~current level
        concentrations.insert("HFC-134a".to_string(), 100.0);

        let total = component.calculate_total_forcing(&concentrations);
        let fgas = component.calculate_fgas_forcing(&concentrations);
        let montreal = component.calculate_montreal_forcing(&concentrations);

        // Total should equal sum of components
        assert!(
            (total - (fgas + montreal)).abs() < 1e-10,
            "Total should equal F-gas + Montreal: {} vs {}",
            total,
            fgas + montreal
        );

        // Forcing should be positive (above pre-industrial)
        assert!(total > 0.0, "Total forcing should be positive: {}", total);
    }

    // ===== EESC Tests =====

    #[test]
    fn test_eesc_calculation_basic() {
        let component = default_component();

        // Set up with just CFC-11, but also need to account for natural halocarbons
        // (CH3Cl and CH3Br have non-zero pre-industrial concentrations)
        let mut concentrations = HashMap::new();
        concentrations.insert("CFC-11".to_string(), 200.0); // ppt

        // Zero out natural halocarbons to isolate CFC-11 contribution
        concentrations.insert("CH3Cl".to_string(), 0.0);
        concentrations.insert("CH3Br".to_string(), 0.0);

        let eesc = component.calculate_eesc(&concentrations);

        // CFC-11: 3 Cl atoms, 0 Br, f_release = 0.47
        // EESC = 200 * 3 * (0.47 / 0.47) = 600
        let expected = 200.0 * 3.0 * 1.0; // normalised release = 1.0 for CFC-11
        assert!(
            (eesc - expected).abs() < 1e-6,
            "CFC-11 EESC: expected {}, got {}",
            expected,
            eesc
        );
    }

    #[test]
    fn test_eesc_bromine_contribution() {
        let component = default_component();

        // Halon-1301: 0 Cl, 1 Br, f_release = 0.28
        let mut concentrations = HashMap::new();
        concentrations.insert("Halon-1301".to_string(), 3.0); // ppt

        // Zero out natural halocarbons to isolate Halon-1301 contribution
        concentrations.insert("CH3Cl".to_string(), 0.0);
        concentrations.insert("CH3Br".to_string(), 0.0);

        let eesc = component.calculate_eesc(&concentrations);

        // EESC = 3 * (60 * 1) * (0.28 / 0.47) = 3 * 60 * 0.596 ≈ 107
        let br_eff = component.parameters().br_multiplier;
        let normalised_release = 0.28 / 0.47;
        let expected = 3.0 * br_eff * normalised_release;

        assert!(
            (eesc - expected).abs() < 1e-6,
            "Halon-1301 EESC: expected {:.2}, got {:.2}",
            expected,
            eesc
        );
    }

    #[test]
    fn test_fgases_no_eesc_contribution() {
        let component = default_component();

        // F-gases have zero fractional release, should not contribute to EESC
        let mut concentrations = HashMap::new();
        concentrations.insert("HFC-134a".to_string(), 1000.0);
        concentrations.insert("SF6".to_string(), 100.0);

        // Zero out natural halocarbons to isolate F-gas contribution
        concentrations.insert("CH3Cl".to_string(), 0.0);
        concentrations.insert("CH3Br".to_string(), 0.0);

        // Also zero out other Montreal gases to ensure only F-gases
        for species in component.parameters().montreal_gases.iter() {
            concentrations.insert(species.name.clone(), 0.0);
        }

        let eesc = component.calculate_eesc(&concentrations);

        assert!(
            eesc.abs() < 1e-10,
            "F-gases should not contribute to EESC: {}",
            eesc
        );
    }

    // ===== Component Trait Tests =====

    #[test]
    fn test_definitions_count() {
        let component = default_component();
        let defs = component.definitions();

        // Should have: 41 species * 2 (emissions + concentration) + 4 aggregates
        let expected_count = 41 * 2 + 4;
        assert_eq!(
            defs.len(),
            expected_count,
            "Should have {} definitions, got {}",
            expected_count,
            defs.len()
        );
    }

    #[test]
    fn test_definitions_contain_expected_names() {
        let component = default_component();
        let defs = component.definitions();

        let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();

        // Check some expected names
        assert!(names.contains(&"Emissions|CFC-11"));
        assert!(names.contains(&"Atmospheric Concentration|CFC-11"));
        assert!(names.contains(&"Emissions|HFC-134a"));
        assert!(names.contains(&"Atmospheric Concentration|HFC-134a"));
        assert!(names.contains(&"Forcing|Halocarbons"));
        assert!(names.contains(&"EESC"));
    }

    // ===== Integration Tests =====

    #[test]
    fn test_step_concentrations_all_species() {
        let component = default_component();

        // Start with zero concentrations
        let mut current = HashMap::new();
        let mut emissions = HashMap::new();

        for species in component.parameters().all_species() {
            current.insert(species.name.clone(), species.concentration_pi);
            emissions.insert(species.name.clone(), 0.1); // Small emissions
        }

        let new_concs = component.step_concentrations(&current, &emissions, 1.0);

        // All species should have new concentrations
        assert_eq!(
            new_concs.len(),
            41,
            "Should have concentrations for all 41 species"
        );

        // With positive emissions, concentrations should increase from PI
        for species in component.parameters().all_species() {
            if species.concentration_pi == 0.0 {
                let conc = *new_concs.get(&species.name).unwrap();
                assert!(
                    conc > 0.0,
                    "Species {} should have positive concentration with emissions",
                    species.name
                );
            }
        }
    }

    #[test]
    fn test_realistic_scenario() {
        let component = default_component();

        // Use rough current concentrations (circa 2020)
        let mut concentrations = HashMap::new();
        concentrations.insert("CFC-11".to_string(), 230.0);
        concentrations.insert("CFC-12".to_string(), 510.0);
        concentrations.insert("CFC-113".to_string(), 70.0);
        concentrations.insert("HCFC-22".to_string(), 245.0);
        concentrations.insert("HFC-134a".to_string(), 100.0);
        concentrations.insert("SF6".to_string(), 10.0);
        concentrations.insert("CH3Cl".to_string(), 540.0); // Natural background

        let total_forcing = component.calculate_total_forcing(&concentrations);
        let eesc = component.calculate_eesc(&concentrations);

        // Total halocarbon forcing ~0.4 W/m² (rough estimate)
        assert!(
            total_forcing > 0.1 && total_forcing < 1.0,
            "Total forcing should be reasonable: {} W/m²",
            total_forcing
        );

        // EESC should be in the range of ~1500-2000 ppt equivalent Cl
        assert!(
            eesc > 500.0 && eesc < 5000.0,
            "EESC should be in reasonable range: {} ppt",
            eesc
        );
    }
}
