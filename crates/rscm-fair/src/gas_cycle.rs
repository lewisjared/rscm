//! Greenhouse gas cycle (FaIR steps 1, 3 and 4)
//!
//! Each gas is held in up to four boxes that decay with lifetimes $\alpha \tau_i$.
//! The lifetime scaling $\alpha = g_0 \exp(r / g_1)$ follows the 100-year integrated impulse
//! response
//!
//! $$r = r_0 + r_u (C_{cum} - C_{air}) + r_T T + r_a C_{air}$$
//!
//! capped at `iirf_max`, from the cumulative and airborne emissions at the start of the step
//! and the surface temperature (Leach et al. 2021).
//! A gas can instead read its lifetime scaling from another component, as CH4 does with
//! [`crate::FairCH4Lifetime`].
//!
//! Emissions-driven gases update the boxes with the exact solution for constant emissions over
//! the step. Concentration-driven gases invert that solution to give the emissions.

use crate::constants::{end, start, surface_temperature, temperature_unit};
use rscm_core::component::{
    Component, ComponentState, InputState, OutputState, RequirementDefinition,
};
use rscm_core::errors::{RSCMError, RSCMResult};
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use serde::{Deserialize, Serialize};
use std::any::Any;

/// How a gas is driven.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GasMode {
    /// Emissions in, concentration out.
    Emissions,
    /// Concentration in, emissions out.
    Concentration,
}

/// One greenhouse gas.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GasSpecies {
    /// How the gas is driven.
    pub mode: GasMode,
    /// Emissions variables, summed. Inputs in emissions mode.
    pub emissions: Vec<String>,
    /// Emissions output in concentration mode.
    #[serde(default)]
    pub inverse_emissions: Option<String>,
    /// Emissions unit.
    pub emissions_unit: String,
    /// Unit of airborne and cumulative emissions.
    pub amount_unit: String,
    /// Concentration variable: an output in emissions mode and an input in concentration mode.
    pub concentration: String,
    /// Concentration unit.
    pub concentration_unit: String,
    /// Airborne emissions output, if wanted.
    #[serde(default)]
    pub airborne_emissions: Option<String>,
    /// Cumulative emissions output, if wanted.
    #[serde(default)]
    pub cumulative_emissions: Option<String>,
    /// Fraction of emissions entering each box.
    pub partition_fraction: Vec<FloatValue>,
    /// Lifetime of each box (yr).
    pub unperturbed_lifetime: Vec<FloatValue>,
    /// Concentration with no airborne emissions.
    pub baseline_concentration: FloatValue,
    /// Emissions that leave the concentration unchanged.
    pub baseline_emissions: FloatValue,
    /// Concentration change per unit emission.
    pub concentration_per_emission: FloatValue,
    /// Pre-industrial 100-year integrated impulse response (yr).
    pub iirf_0: FloatValue,
    /// Sensitivity of `iirf` to airborne emissions.
    pub iirf_airborne: FloatValue,
    /// Sensitivity of `iirf` to emissions taken up by the Earth system.
    pub iirf_uptake: FloatValue,
    /// Sensitivity of `iirf` to surface temperature (yr/K).
    pub iirf_temperature: FloatValue,
    /// Lifetime scaling parameter `g0`.
    pub g0: FloatValue,
    /// Lifetime scaling parameter `g1`.
    pub g1: FloatValue,
    /// Lifetime scaling input that replaces the impulse response scaling.
    #[serde(default)]
    pub lifetime_scaling: Option<String>,
}

/// Parameters of [`FairGasCycle`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FairGasCycleParameters {
    /// The gases.
    pub species: Vec<GasSpecies>,
    /// Upper limit on the integrated impulse response (yr).
    #[serde(default = "default_iirf_max")]
    pub iirf_max: FloatValue,
    /// Surface temperature input.
    #[serde(default = "surface_temperature")]
    pub surface_temperature: String,
}

fn default_iirf_max() -> FloatValue {
    100.0
}

/// Box contents and emission totals of one gas.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GasState {
    /// Airborne emissions in each box.
    pub boxes: Vec<FloatValue>,
    /// Airborne emissions.
    pub airborne: FloatValue,
    /// Cumulative emissions.
    pub cumulative: FloatValue,
    /// Sum of the emission rates so far, which FaIR scales by the timestep for the cumulative
    /// emissions of emissions-driven gases.
    pub emissions_sum: FloatValue,
}

/// State of every gas, in species order.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FairGasCycleState {
    /// One entry per gas.
    pub gases: Vec<GasState>,
}

#[typetag::serde]
impl ComponentState for FairGasCycleState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// FaIR's greenhouse gas cycle for any set of gases.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "Unchecked")]
pub struct FairGasCycle {
    parameters: FairGasCycleParameters,
}

#[derive(Deserialize)]
struct Unchecked {
    parameters: FairGasCycleParameters,
}

impl TryFrom<Unchecked> for FairGasCycle {
    type Error = RSCMError;

    fn try_from(unchecked: Unchecked) -> RSCMResult<Self> {
        Self::from_parameters(unchecked.parameters)
    }
}

impl FairGasCycle {
    /// Create the component, checking each gas.
    pub fn from_parameters(parameters: FairGasCycleParameters) -> RSCMResult<Self> {
        for species in &parameters.species {
            let n = species.partition_fraction.len();
            if n == 0 || species.unperturbed_lifetime.len() != n {
                return Err(RSCMError::Error(format!(
                    "{}: partition_fraction and unperturbed_lifetime need one value per box",
                    species.concentration
                )));
            }
            match species.mode {
                GasMode::Emissions if species.emissions.is_empty() => {
                    return Err(RSCMError::Error(format!(
                        "{}: an emissions-driven gas needs an emissions input",
                        species.concentration
                    )))
                }
                GasMode::Concentration if species.inverse_emissions.is_none() => {
                    return Err(RSCMError::Error(format!(
                        "{}: a concentration-driven gas needs inverse_emissions",
                        species.concentration
                    )))
                }
                _ => {}
            }
        }
        Ok(Self { parameters })
    }

    /// Parameters of this component.
    pub fn parameters(&self) -> &FairGasCycleParameters {
        &self.parameters
    }

    fn initial_state(&self) -> FairGasCycleState {
        FairGasCycleState {
            gases: self
                .parameters
                .species
                .iter()
                .map(|s| GasState {
                    boxes: vec![0.0; s.partition_fraction.len()],
                    ..GasState::default()
                })
                .collect(),
        }
    }

    /// Lifetime scaling from the integrated impulse response.
    pub fn alpha(
        &self,
        species: &GasSpecies,
        state: &GasState,
        temperature: FloatValue,
    ) -> FloatValue {
        let iirf = species.iirf_0
            + species.iirf_uptake * (state.cumulative - state.airborne)
            + species.iirf_temperature * temperature
            + species.iirf_airborne * state.airborne;
        let max = self.parameters.iirf_max;
        // FaIR's cap, which gives 0 at exactly the maximum.
        let iirf = if iirf > max { max } else { 0.0 } + iirf * if iirf < max { 1.0 } else { 0.0 };
        let alpha = species.g0 * (iirf / species.g1).exp();
        if alpha.is_nan() {
            1.0
        } else {
            alpha
        }
    }

    /// Advance an emissions-driven gas; returns the concentration at the end of the step.
    pub fn step_emissions(
        species: &GasSpecies,
        state: &mut GasState,
        emissions: FloatValue,
        alpha: FloatValue,
        dt: FloatValue,
    ) -> FloatValue {
        let mut airborne = 0.0;
        for (i, b) in state.boxes.iter_mut().enumerate() {
            let decay_rate = dt / (alpha * species.unperturbed_lifetime[i]);
            let decay_factor = (-decay_rate).exp();
            *b = species.partition_fraction[i] * (emissions - species.baseline_emissions) * 1.0
                / decay_rate
                * (1.0 - decay_factor)
                * dt
                + *b * decay_factor;
            airborne += *b;
        }
        state.airborne = airborne;
        state.emissions_sum += emissions;
        state.cumulative = state.emissions_sum * dt;
        species.baseline_concentration + species.concentration_per_emission * airborne
    }

    /// Advance a concentration-driven gas; returns the emissions over the step.
    pub fn step_concentration(
        species: &GasSpecies,
        state: &mut GasState,
        concentration: FloatValue,
        alpha: FloatValue,
        dt: FloatValue,
    ) -> FloatValue {
        let n = state.boxes.len();
        let mut decay_rate = vec![0.0; n];
        let mut decay_factor = vec![0.0; n];
        let mut remaining = 0.0;
        let mut uptake = 0.0;
        for i in 0..n {
            decay_rate[i] = dt / (alpha * species.unperturbed_lifetime[i]);
            decay_factor[i] = (-decay_rate[i]).exp();
            remaining += state.boxes[i] * decay_factor[i];
            uptake += species.partition_fraction[i] / decay_rate[i] * (1.0 - decay_factor[i]) * dt;
        }
        let airborne =
            (concentration - species.baseline_concentration) / species.concentration_per_emission;
        let emissions = (airborne - remaining) / uptake;
        for i in 0..n {
            state.boxes[i] = dt * emissions * species.partition_fraction[i] * 1.0 / decay_rate[i]
                * (1.0 - decay_factor[i])
                + state.boxes[i] * decay_factor[i];
        }
        let emissions = emissions + species.baseline_emissions;
        state.airborne = airborne;
        state.cumulative += emissions * dt;
        emissions
    }

    fn solve_impl(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
        state: &mut FairGasCycleState,
    ) -> RSCMResult<OutputState> {
        let dt = t_next - t_current;
        let temperature = start(input_state, &self.parameters.surface_temperature);
        let mut output = OutputState::with_capacity(3 * self.parameters.species.len());
        for (species, gas) in self.parameters.species.iter().zip(state.gases.iter_mut()) {
            let alpha = match &species.lifetime_scaling {
                Some(name) => end(input_state, name),
                None => self.alpha(species, gas, temperature),
            };
            match species.mode {
                GasMode::Emissions => {
                    let emissions = species
                        .emissions
                        .iter()
                        .map(|name| start(input_state, name))
                        .reduce(|a, b| a + b)
                        .unwrap_or(0.0);
                    let concentration = Self::step_emissions(species, gas, emissions, alpha, dt);
                    output.insert(
                        species.concentration.clone(),
                        StateValue::Scalar(concentration),
                    );
                }
                GasMode::Concentration => {
                    let concentration = end(input_state, &species.concentration);
                    let emissions =
                        Self::step_concentration(species, gas, concentration, alpha, dt);
                    if let Some(name) = &species.inverse_emissions {
                        output.insert(name.clone(), StateValue::Scalar(emissions));
                    }
                }
            }
            if let Some(name) = &species.airborne_emissions {
                output.insert(name.clone(), StateValue::Scalar(gas.airborne));
            }
            if let Some(name) = &species.cumulative_emissions {
                output.insert(name.clone(), StateValue::Scalar(gas.cumulative));
            }
        }
        Ok(output)
    }
}

#[typetag::serde]
impl Component for FairGasCycle {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        let mut definitions = vec![RequirementDefinition::scalar_input(
            &self.parameters.surface_temperature,
            &temperature_unit(),
        )];
        for s in &self.parameters.species {
            match s.mode {
                GasMode::Emissions => {
                    for name in &s.emissions {
                        definitions
                            .push(RequirementDefinition::scalar_input(name, &s.emissions_unit));
                    }
                    definitions.push(RequirementDefinition::scalar_output(
                        &s.concentration,
                        &s.concentration_unit,
                    ));
                }
                GasMode::Concentration => {
                    definitions.push(RequirementDefinition::scalar_input(
                        &s.concentration,
                        &s.concentration_unit,
                    ));
                    if let Some(name) = &s.inverse_emissions {
                        definitions.push(RequirementDefinition::scalar_output(
                            name,
                            &s.emissions_unit,
                        ));
                    }
                }
            }
            for name in [&s.airborne_emissions, &s.cumulative_emissions]
                .into_iter()
                .flatten()
            {
                definitions.push(RequirementDefinition::scalar_output(name, &s.amount_unit));
            }
            if let Some(name) = &s.lifetime_scaling {
                definitions.push(RequirementDefinition::scalar_input(name, "1"));
            }
        }
        definitions
    }

    /// Stateless solve from empty boxes, so only correct for the first step.
    /// Models use `solve_with_state`.
    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let mut state = self.initial_state();
        self.solve_impl(t_current, t_next, input_state, &mut state)
    }

    fn create_initial_state(&self) -> Box<dyn ComponentState> {
        Box::new(self.initial_state())
    }

    fn solve_with_state(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
        internal_state: &mut dyn ComponentState,
    ) -> RSCMResult<OutputState> {
        let state = internal_state
            .as_any_mut()
            .downcast_mut::<FairGasCycleState>()
            .ok_or_else(|| RSCMError::Error("Internal state is not FairGasCycleState".into()))?;
        self.solve_impl(t_current, t_next, input_state, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn ch4() -> GasSpecies {
        GasSpecies {
            mode: GasMode::Emissions,
            emissions: vec!["Emissions|CH4".into()],
            inverse_emissions: Some("Emissions|CH4".into()),
            emissions_unit: "Mt CH4/yr".into(),
            amount_unit: "Mt CH4".into(),
            concentration: "Atmospheric Concentration|CH4".into(),
            concentration_unit: "ppb".into(),
            airborne_emissions: None,
            cumulative_emissions: None,
            partition_fraction: vec![1.0, 0.0, 0.0, 0.0],
            unperturbed_lifetime: vec![8.25; 4],
            baseline_concentration: 729.2,
            baseline_emissions: 0.0,
            concentration_per_emission: 1.0 / (5.1352 * 16.043 / 28.97),
            iirf_0: 8.249955097,
            iirf_airborne: 0.00032,
            iirf_uptake: 0.0,
            iirf_temperature: -0.3,
            g0: 0.36785517,
            g1: 8.249410814,
            lifetime_scaling: None,
        }
    }

    #[test]
    fn inverse_recovers_emissions() {
        let species = ch4();
        let mut forward = GasState {
            boxes: vec![0.0; 4],
            ..GasState::default()
        };
        let mut inverse = forward.clone();
        for emissions in [100.0, 250.0, 400.0] {
            let c = FairGasCycle::step_emissions(&species, &mut forward, emissions, 1.2, 1.0);
            let e = FairGasCycle::step_concentration(&species, &mut inverse, c, 1.2, 1.0);
            assert_relative_eq!(e, emissions, max_relative = 1e-10);
        }
    }

    #[test]
    fn deserialising_checks_the_parameters() {
        let mut parameters = FairGasCycleParameters {
            species: vec![ch4()],
            iirf_max: default_iirf_max(),
            surface_temperature: surface_temperature(),
        };
        let valid =
            serde_json::to_string(&FairGasCycle::from_parameters(parameters.clone()).unwrap())
                .unwrap();
        assert!(serde_json::from_str::<FairGasCycle>(&valid).is_ok());
        parameters.species[0].unperturbed_lifetime.pop();
        let json = format!(
            r#"{{"parameters": {}}}"#,
            serde_json::to_string(&parameters).unwrap()
        );
        assert!(serde_json::from_str::<FairGasCycle>(&json).is_err());
    }

    #[test]
    fn single_box_decays_with_lifetime() {
        let species = ch4();
        let mut state = GasState {
            boxes: vec![100.0, 0.0, 0.0, 0.0],
            ..GasState::default()
        };
        FairGasCycle::step_emissions(&species, &mut state, 0.0, 1.0, 1.0);
        assert_relative_eq!(state.boxes[0], 100.0 * (-1.0 / 8.25_f64).exp());
    }
}
