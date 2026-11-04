//! Halocarbon Chemistry Component
//!
//! Concentrations, forcing and EESC of F-gases and Montreal Protocol gases,
//! following MAGICC7 Module 03.
//!
//! # Inputs
//!
//! - `Emissions|<species>` (kt/yr) for each species
//! - `Lifetime|CH4|OH` (yr), which scales OH lifetimes
//! - `Surface Temperature` (K), which shortens stratospheric lifetimes
//! - `Emissions|CO2|Fossil` (GtC/yr) with `fgas_regions`, which spreads short-lived F-gases
//!
//! # States
//!
//! - `Atmospheric Concentration|<species>` (ppt)
//!
//! # Outputs
//!
//! - `Effective Radiative Forcing|F-Gases` (W/m²), four-box with `fgas_regions`
//! - `Effective Radiative Forcing|Montreal Protocol Halogen Gases` (W/m²)
//! - Each species' share of these, as `Effective Radiative Forcing|<group>|<species>`
//! - `EESC` (ppt): equivalent effective stratospheric chlorine
//!
//! # Lifetimes
//!
//! $$\frac{1}{\tau} = \frac{1}{s_{OH} \tau_{OH}} + \frac{1}{s_{strat} \tau_{strat}} + \frac{1}{\tau_{other}}$$
//!
//! where $s_{OH}$ is the CH4 OH lifetime relative to its unperturbed value and
//! $s_{strat} = 1 / (1 + f \cdot k \cdot \max(0, T - T_{1980}))$ after 1980.
//!
//! # Concentrations
//!
//! Crank-Nicolson for lifetimes of at least 5 years, otherwise the exact
//! exponential solution, with the emissions and lifetime of the start of the year.

use crate::parameters::{HalocarbonParameters, HalocarbonSpecies};
use crate::reference::{
    feedback_warming, prescribed_definition, prescribed_input, prescribed_name, require_annual,
};
use rscm_core::component::{Component, InputState, OutputState, RequirementDefinition};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{FourBoxSlice, ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// F-gas ERF output.
pub const FGAS_FORCING: &str = "Effective Radiative Forcing|F-Gases";
/// Montreal Protocol gas ERF output.
pub const MHALO_FORCING: &str = "Effective Radiative Forcing|Montreal Protocol Halogen Gases";
/// EESC output.
pub const EESC: &str = "EESC";
/// CH4 OH lifetime input.
pub const CH4_OH_LIFETIME: &str = "Lifetime|CH4|OH";
const FOSSIL_CO2: &str = "Emissions|CO2|Fossil";

/// Halocarbon atmospheric chemistry component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HalocarbonChemistry {
    parameters: HalocarbonParameters,
    #[serde(skip)]
    names: OnceLock<[Vec<SpeciesNames>; 2]>,
}

/// Variables of one species, built once as every step reads or writes them.
#[derive(Debug, Clone)]
struct SpeciesNames {
    emissions: String,
    concentration: String,
    prescribed: String,
    forcing: String,
}

impl HalocarbonChemistry {
    /// Create a component with MAGICC7 default parameters.
    pub fn new() -> Self {
        Self::from_parameters(HalocarbonParameters::default())
    }

    /// Create a component from parameters.
    pub fn from_parameters(parameters: HalocarbonParameters) -> Self {
        Self {
            parameters,
            names: OnceLock::new(),
        }
    }

    /// Parameters of this component.
    pub fn parameters(&self) -> &HalocarbonParameters {
        &self.parameters
    }

    fn emissions_name(species: &str) -> String {
        format!("Emissions|{species}")
    }

    fn concentration_name(species: &str) -> String {
        format!("Atmospheric Concentration|{species}")
    }

    /// ERF variable of a species in group 0 (F-gases) or 1 (Montreal gases).
    pub fn forcing_name(group: usize, species: &str) -> String {
        format!("{}|{species}", [FGAS_FORCING, MHALO_FORCING][group])
    }

    /// Lifetime (yr) given the OH and stratospheric lifetime scale factors.
    pub fn lifetime(
        &self,
        species: &HalocarbonSpecies,
        oh_scale: FloatValue,
        strat_scale: FloatValue,
    ) -> FloatValue {
        let mut rate = 0.0;
        if species.tau_oh > 0.0 {
            rate += 1.0 / (species.tau_oh * oh_scale);
        }
        if species.tau_strat > 0.0 {
            rate += 1.0 / (species.tau_strat * strat_scale);
        }
        if let Some(tau) = species.tau_other() {
            rate += 1.0 / tau;
        }
        1.0 / rate
    }

    /// Concentration after one year (ppt).
    pub fn step_concentration(
        &self,
        species: &HalocarbonSpecies,
        concentration: FloatValue,
        emissions: FloatValue,
        tau: FloatValue,
    ) -> FloatValue {
        let factor = self.parameters.emission_to_concentration_factor(species);
        if tau >= 5.0 {
            let half = 1.0 / (2.0 * tau);
            (concentration * (1.0 - half) + emissions * factor) / (1.0 + half)
        } else {
            let decay = (-1.0 / tau).exp();
            tau * emissions * factor * (1.0 - decay) + concentration * decay
        }
    }

    /// ERF of a species at a concentration (W/m²).
    pub fn species_forcing(species: &HalocarbonSpecies, concentration: FloatValue) -> FloatValue {
        (concentration - species.concentration_pi) * (species.radiative_efficiency / 1000.0)
    }

    /// EESC contribution of a species at a concentration (ppt).
    pub fn eesc_contribution(
        &self,
        species: &HalocarbonSpecies,
        concentration: FloatValue,
    ) -> FloatValue {
        let p = &self.parameters;
        let release = |atoms: u8| {
            FloatValue::from(atoms) * species.fractional_release * p.cfc11_release_factor
        };
        concentration * (p.br_vs_cl_scale * release(species.n_br) + release(species.n_cl))
    }

    fn scale_factors(&self, t_current: Time, input_state: &InputState) -> (FloatValue, FloatValue) {
        let p = &self.parameters;
        let oh_scale = if p.variable_oh_lifetime {
            let window = input_state.get_scalar_window(CH4_OH_LIFETIME);
            window.at_end().unwrap_or_else(|| window.at_start()) / p.ch4_oh_lifetime_init
        } else {
            1.0
        };
        let strat_scale = if p.variable_strat_lifetime {
            let temperature = input_state.get_scalar_window("Surface Temperature");
            let warming = feedback_warming(&temperature, t_current, p.meridional_flux_year);
            1.0 / (1.0 + warming * p.meridional_flux_per_degree * p.strat_lifetime_sensitivity)
        } else {
            1.0
        };
        (oh_scale, strat_scale)
    }
}

impl HalocarbonChemistry {
    /// Names of each species in each group of [`Self::switch_years`].
    fn names(&self) -> &[Vec<SpeciesNames>; 2] {
        self.names.get_or_init(|| {
            let mut groups = self.switch_years().into_iter().enumerate();
            std::array::from_fn(|_| {
                let (group, (list, _)) = groups.next().expect("two species groups");
                list.iter()
                    .map(|species| {
                        let concentration = Self::concentration_name(&species.name);
                        SpeciesNames {
                            emissions: Self::emissions_name(&species.name),
                            prescribed: prescribed_name(&concentration),
                            forcing: Self::forcing_name(group, &species.name),
                            concentration,
                        }
                    })
                    .collect()
            })
        })
    }

    /// F-gases and Montreal gases with the last year of their prescribed concentrations.
    fn switch_years(&self) -> [(&[HalocarbonSpecies], Option<Time>); 2] {
        let p = &self.parameters;
        [
            (&p.fgases, p.fgas_switch_year),
            (&p.montreal_gases, p.mhalo_switch_year),
        ]
    }
}

impl Default for HalocarbonChemistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Value `lag` steps before the end of the step, clamped to the first.
fn lagged(window: &ScalarWindow, end: FloatValue, lag: usize) -> FloatValue {
    if lag == 0 {
        return end;
    }
    let offset = (lag as isize - 1).min(window.index() as isize);
    window
        .at_offset(-offset)
        .unwrap_or_else(|| window.at_start())
}

#[typetag::serde]
impl Component for HalocarbonChemistry {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        let mut definitions = Vec::new();
        for species in self.parameters.all_species() {
            definitions.push(RequirementDefinition::scalar_input(
                &Self::emissions_name(&species.name),
                "kt/yr",
            ));
            definitions.push(RequirementDefinition::scalar_state(
                &Self::concentration_name(&species.name),
                "ppt",
            ));
        }
        for (list, switch_year) in self.switch_years() {
            for species in list {
                definitions.extend(prescribed_definition(
                    &Self::concentration_name(&species.name),
                    "ppt",
                    switch_year,
                ));
            }
        }
        if self.parameters.variable_oh_lifetime {
            definitions.push(RequirementDefinition::scalar_input(CH4_OH_LIFETIME, "yr"));
        }
        if self.parameters.variable_strat_lifetime {
            definitions.push(RequirementDefinition::scalar_input(
                "Surface Temperature",
                "K",
            ));
        }
        if self.parameters.fgas_regions.is_some() {
            definitions.push(RequirementDefinition::scalar_input(FOSSIL_CO2, "GtC/yr"));
        }
        let regional = self.parameters.fgas_regions.is_some();
        let forcing = |name: &str, four_box: bool| {
            if four_box {
                RequirementDefinition::four_box_output(name, "W/m^2")
            } else {
                RequirementDefinition::scalar_output(name, "W/m^2")
            }
        };
        definitions.push(forcing(FGAS_FORCING, regional));
        definitions.push(forcing(MHALO_FORCING, false));
        for (group, (list, _)) in self.switch_years().into_iter().enumerate() {
            for species in list {
                let name = Self::forcing_name(group, &species.name);
                definitions.push(forcing(&name, regional && group == 0));
            }
        }
        definitions.push(RequirementDefinition::scalar_output(EESC, "ppt"));
        definitions
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        require_annual("HalocarbonChemistry", t_current, t_next)?;
        let (oh_scale, strat_scale) = self.scale_factors(t_current, input_state);
        let names = self.names();
        let mut output = OutputState::with_capacity(2 * (names[0].len() + names[1].len()) + 3);
        let mut forcing = [0.0; 2];
        let mut fgas_boxes = [0.0; 4];
        let mut eesc = 0.0;
        let fossil_co2 = self
            .parameters
            .fgas_regions
            .as_ref()
            .map(|_| input_state.get_scalar_window(FOSSIL_CO2).at_start());

        for (group, (list, switch_year)) in self.switch_years().into_iter().enumerate() {
            for (species, names) in list.iter().zip(&names[group]) {
                let window = input_state.get_scalar_window(&names.concentration);
                let emissions = input_state.get_scalar_window(&names.emissions).at_start();
                let tau = self.lifetime(species, oh_scale, strat_scale);
                let next = prescribed_input(input_state, &names.prescribed, switch_year, t_next)
                    .unwrap_or_else(|| {
                        self.step_concentration(species, window.at_start(), emissions, tau)
                    });
                let species_forcing = Self::species_forcing(species, next);
                forcing[group] += species_forcing;
                let value = if let (0, Some(regions), Some(fossil_co2)) =
                    (group, &self.parameters.fgas_regions, fossil_co2)
                {
                    let boxes = regions
                        .pattern(tau, fossil_co2)
                        .map(|s| species_forcing * s);
                    for (total, b) in fgas_boxes.iter_mut().zip(boxes) {
                        *total += b;
                    }
                    StateValue::FourBox(FourBoxSlice::from_array(boxes))
                } else {
                    StateValue::Scalar(species_forcing)
                };
                output.insert(names.forcing.clone(), value);
                let delayed = lagged(&window, next, self.parameters.eesc_delay);
                eesc += self.eesc_contribution(species, delayed);
                output.insert(names.concentration.clone(), StateValue::Scalar(next));
            }
        }

        let fgas = if self.parameters.fgas_regions.is_some() {
            StateValue::FourBox(FourBoxSlice::from_array(fgas_boxes))
        } else {
            StateValue::Scalar(forcing[0])
        };
        output.insert(FGAS_FORCING.to_string(), fgas);
        output.insert(MHALO_FORCING.to_string(), StateValue::Scalar(forcing[1]));
        output.insert(EESC.to_string(), StateValue::Scalar(eesc));
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn component() -> HalocarbonChemistry {
        HalocarbonChemistry::new()
    }

    fn species(name: &str) -> HalocarbonSpecies {
        component().parameters().get_species(name).unwrap().clone()
    }

    #[test]
    fn test_unscaled_lifetime_is_total() {
        let c = component();
        for s in c.parameters().all_species() {
            assert!(
                (c.lifetime(s, 1.0, 1.0) - s.tau_total).abs() < 1e-9,
                "{}",
                s.name
            );
        }
    }

    #[test]
    fn test_oh_scaling_lengthens_lifetime() {
        let c = component();
        let hcfc22 = species("HCFC22");
        assert!(c.lifetime(&hcfc22, 1.1, 1.0) > hcfc22.tau_total);
        let cfc11 = species("CFC11");
        assert!((c.lifetime(&cfc11, 1.1, 1.0) - cfc11.tau_total).abs() < 1e-9);
    }

    #[test]
    fn test_long_lived_decay_is_crank_nicolson() {
        let c = component();
        let cfc11 = species("CFC11");
        let tau = 52.0;
        let half = 1.0 / (2.0 * tau);
        let expected = 100.0 * (1.0 - half) / (1.0 + half);
        assert!((c.step_concentration(&cfc11, 100.0, 0.0, tau) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_short_lived_decay_is_exponential() {
        let c = component();
        let ch3cl = species("CH3Cl");
        let next = c.step_concentration(&ch3cl, 100.0, 0.0, 0.9);
        assert!((next - 100.0 * (-1.0 / 0.9_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn test_steady_state_short_lived() {
        let c = component();
        let ch3cl = species("CH3Cl");
        let conv = c.parameters().emission_to_concentration_factor(&ch3cl);
        let steady = 1000.0 * conv * 0.9;
        assert!((c.step_concentration(&ch3cl, steady, 1000.0, 0.9) - steady).abs() < 1e-9);
    }

    #[test]
    fn test_forcing_relative_to_initial() {
        let cfc12 = species("CFC12");
        assert_eq!(
            HalocarbonChemistry::species_forcing(&cfc12, cfc12.concentration_pi),
            0.0
        );
        assert!((HalocarbonChemistry::species_forcing(&cfc12, 500.0) - 0.182).abs() < 1e-12);
    }

    #[test]
    fn test_eesc_contribution() {
        let c = component();
        let halon1301 = species("Halon1301");
        let expected = 3.0 * 62.5 * 0.28 * 0.75;
        assert!((c.eesc_contribution(&halon1301, 3.0) - expected).abs() < 1e-12);
    }
}
