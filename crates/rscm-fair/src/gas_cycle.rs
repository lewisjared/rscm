//! Greenhouse gas cycle (FaIR steps 1, 3 and 4)
//!
//! Calculates the atmospheric concentration of each emissions-driven greenhouse gas,
//! and the inverse emissions of each concentration-driven one.
//! It also tracks the airborne and cumulative emissions of every gas.
//!
//! Each gas is held in up to four boxes that decay with lifetimes $\alpha \tau_i$.
//! The lifetime scaling $\alpha = g_0 \exp(r / g_1)$ follows the 100-year integrated impulse
//! response
//!
//! $$r = r_0 + r_u (C_{cum} - C_{air}) + r_T T + r_a C_{air}$$
//!
//! capped at `maximum_iirf`, from the cumulative and airborne emissions at the start of the step
//! and the surface air temperature change (Leach et al. 2021).
//! A gas can instead read its lifetime scaling from another component, as CH4 does with
//! [`crate::CH4Lifetime`].
//!
//! Emissions-driven gases update the boxes with the exact solution for constant emissions over
//! the step.
//! Concentration-driven gases invert that solution to give the inverse emissions.
//!
//! # Inputs
//!
//! - The surface air temperature change named by `surface_temperature` (K).
//!   Its default is `Surface Air Temperature Change`.
//! - Each of a gas's `emissions`, in its `emissions_unit`, when the gas is emissions driven.
//!   rscm.fair passes `Emissions|<species>`,
//!   or `Emissions|CO2|Fossil` and `Emissions|CO2|Biosphere` when FaIR calculates CO2 from them.
//! - A gas's `concentration`, in its `concentration_unit`, when the gas is concentration driven.
//!   rscm.fair names it `Atmospheric Concentrations|<species>`.
//! - The lifetime scaling named by a gas's `lifetime_scaling` (1), an average over the step,
//!   when it is set.
//!   rscm.fair sets `Lifetime Scaling|CH4` for CH4 with the `thornhill2021` CH4 method.
//!
//! # Outputs
//!
//! - A gas's `concentration`, in its `concentration_unit`, when the gas is emissions driven.
//!   rscm.fair names it `Atmospheric Concentrations|<species>`.
//! - A gas's `inverse_emissions`, in its `emissions_unit`, when the gas is concentration driven.
//!   rscm.fair names it `Inverse Emissions|<species>`.
//! - A gas's `airborne_emissions` and `cumulative_emissions`, in its `amount_unit`,
//!   when they are set.
//!   rscm.fair names them `Airborne Emissions|<species>` and `Cumulative Emissions|<species>`.

use crate::constants::{get, surface_temperature, temperature_unit};
use crate::driver::Driver;
use rscm_core::component::{
    Component, ComponentState, InputState, OutputState, TimeseriesType, VariableDefinition,
};
use rscm_core::errors::{RSCMError, RSCMResult};
use rscm_core::state::StateValue;
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentMetadata;
use serde::{Deserialize, Serialize};
use std::any::Any;

/// One greenhouse gas.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GasSpecies {
    /// How the gas is driven: emissions in and concentration out,
    /// or concentration in and inverse emissions out.
    pub driver: Driver,
    /// Names of the emissions inputs, summed, read when the gas is emissions driven.
    pub emissions: Vec<String>,
    /// Name of the inverse emissions output, written when the gas is concentration driven.
    #[serde(default)]
    pub inverse_emissions: Option<String>,
    /// Unit of the emissions and inverse emissions.
    pub emissions_unit: String,
    /// Unit of the airborne and cumulative emissions.
    pub amount_unit: String,
    /// Name of the concentration,
    /// an output when the gas is emissions driven and an input when it is concentration driven.
    pub concentration: String,
    /// Unit of the concentration.
    pub concentration_unit: String,
    /// Name of the airborne emissions output, if wanted.
    #[serde(default)]
    pub airborne_emissions: Option<String>,
    /// Name of the cumulative emissions output, if wanted.
    #[serde(default)]
    pub cumulative_emissions: Option<String>,
    /// Fraction of emissions entering each box (FaIR `partition_fraction`).
    pub partition_fraction: Vec<FloatValue>,
    /// Lifetime of each box before scaling (FaIR `unperturbed_lifetime`).
    /// unit: yr
    pub unperturbed_lifetime: Vec<FloatValue>,
    /// Concentration with no airborne emissions, in `concentration_unit`
    /// (FaIR `baseline_concentration`).
    pub baseline_concentration: FloatValue,
    /// Emissions that leave the concentration unchanged, in `emissions_unit`
    /// (FaIR `baseline_emissions`).
    pub baseline_emissions: FloatValue,
    /// Concentration change per unit of airborne emissions (FaIR `concentration_per_emission`).
    pub concentration_per_emission: FloatValue,
    /// Pre-industrial 100-year integrated impulse response (FaIR `iirf_0`).
    /// unit: yr
    pub iirf_0: FloatValue,
    /// Change in the integrated impulse response per unit of airborne emissions
    /// (FaIR `iirf_airborne`).
    pub iirf_airborne: FloatValue,
    /// Change in the integrated impulse response per unit of emissions taken up by the Earth
    /// system (FaIR `iirf_uptake`).
    pub iirf_uptake: FloatValue,
    /// Change in the integrated impulse response per kelvin of surface air temperature change
    /// (FaIR `iirf_temperature`).
    /// unit: yr/K
    pub iirf_temperature: FloatValue,
    /// Lifetime scaling parameter $g_0$ (FaIR `g0`).
    pub g0: FloatValue,
    /// Lifetime scaling parameter $g_1$ (FaIR `g1`).
    /// unit: yr
    pub g1: FloatValue,
    /// Name of a lifetime scaling input that replaces the impulse response scaling, if set.
    #[serde(default)]
    pub lifetime_scaling: Option<String>,
}

/// Parameters of [`GasCycle`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GasCycleParameters {
    /// Greenhouse gases the component tracks.
    pub species: Vec<GasSpecies>,
    /// Upper limit on the 100-year integrated impulse response (FaIR `iirf_max`).
    /// unit: yr
    #[serde(default = "default_maximum_iirf")]
    pub maximum_iirf: FloatValue,
    /// Name of the surface air temperature change input.
    #[serde(default = "surface_temperature")]
    pub surface_temperature: String,
}

fn default_maximum_iirf() -> FloatValue {
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
    /// Sum of the emission rates so far, which FaIR scales by the step for the cumulative
    /// emissions of emissions-driven gases.
    pub emissions_sum: FloatValue,
}

/// State of every gas, in species order.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GasCycleState {
    /// One entry per gas.
    pub gases: Vec<GasState>,
}

#[typetag::serde(name = "fair.GasCycleState")]
impl ComponentState for GasCycleState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// FaIR's greenhouse gas cycle for any set of gases.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentMetadata)]
#[component(tags = ["chemistry", "carbon-cycle", "gas-cycle", "fair"], category = "Atmospheric Chemistry")]
#[serde(try_from = "Unchecked")]
pub struct GasCycle {
    parameters: GasCycleParameters,
}

#[derive(Deserialize)]
struct Unchecked {
    parameters: GasCycleParameters,
}

impl TryFrom<Unchecked> for GasCycle {
    type Error = RSCMError;

    fn try_from(unchecked: Unchecked) -> RSCMResult<Self> {
        Self::from_parameters(unchecked.parameters)
    }
}

impl GasCycle {
    /// Create the component, checking each gas.
    pub fn from_parameters(parameters: GasCycleParameters) -> RSCMResult<Self> {
        for species in &parameters.species {
            let n = species.partition_fraction.len();
            if n == 0 || species.unperturbed_lifetime.len() != n {
                return Err(RSCMError::Error(format!(
                    "{}: partition_fraction and unperturbed_lifetime need one value per box",
                    species.concentration
                )));
            }
            match species.driver {
                Driver::Emissions if species.emissions.is_empty() => {
                    return Err(RSCMError::Error(format!(
                        "{}: an emissions-driven gas needs an emissions input",
                        species.concentration
                    )))
                }
                Driver::Concentration if species.inverse_emissions.is_none() => {
                    return Err(RSCMError::Error(format!(
                        "{}: a concentration-driven gas needs inverse_emissions",
                        species.concentration
                    )))
                }
                Driver::CumulativeEmissions => {
                    return Err(RSCMError::Error(format!(
                    "{}: a gas is driven by emissions or concentration, not cumulative emissions",
                    species.concentration
                )))
                }
                _ => {}
            }
        }
        Ok(Self { parameters })
    }

    /// Parameters of this component.
    pub fn parameters(&self) -> &GasCycleParameters {
        &self.parameters
    }

    fn initial_state(&self) -> GasCycleState {
        GasCycleState {
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
        let max = self.parameters.maximum_iirf;
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
        state: &mut GasCycleState,
    ) -> RSCMResult<OutputState> {
        let dt = t_next - t_current;
        let temperature = get(input_state, &self.parameters.surface_temperature);
        let mut output = OutputState::with_capacity(3 * self.parameters.species.len());
        for (species, gas) in self.parameters.species.iter().zip(state.gases.iter_mut()) {
            let alpha = match &species.lifetime_scaling {
                Some(name) => get(input_state, name),
                None => self.alpha(species, gas, temperature),
            };
            match species.driver {
                Driver::Emissions => {
                    let emissions = species
                        .emissions
                        .iter()
                        .map(|name| get(input_state, name))
                        .reduce(|a, b| a + b)
                        .unwrap_or(0.0);
                    let concentration = Self::step_emissions(species, gas, emissions, alpha, dt);
                    output.insert(
                        species.concentration.clone(),
                        StateValue::Scalar(concentration),
                    );
                }
                Driver::Concentration => {
                    let concentration = get(input_state, &species.concentration);
                    let emissions =
                        Self::step_concentration(species, gas, concentration, alpha, dt);
                    if let Some(name) = &species.inverse_emissions {
                        output.insert(name.clone(), StateValue::Scalar(emissions));
                    }
                }
                Driver::CumulativeEmissions => unreachable!("rejected by from_parameters"),
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

#[typetag::serde(name = "fair.GasCycle")]
impl Component for GasCycle {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let mut definitions = vec![VariableDefinition::scalar_input(
            &self.parameters.surface_temperature,
            &temperature_unit(),
        )];
        for s in &self.parameters.species {
            match s.driver {
                Driver::Emissions => {
                    for name in &s.emissions {
                        definitions.push(VariableDefinition::scalar_input(name, &s.emissions_unit));
                    }
                    definitions.push(VariableDefinition::scalar_output(
                        &s.concentration,
                        &s.concentration_unit,
                    ));
                }
                Driver::Concentration => {
                    definitions.push(VariableDefinition::scalar_input(
                        &s.concentration,
                        &s.concentration_unit,
                    ));
                    if let Some(name) = &s.inverse_emissions {
                        definitions
                            .push(VariableDefinition::scalar_output(name, &s.emissions_unit));
                    }
                }
                Driver::CumulativeEmissions => unreachable!("rejected by from_parameters"),
            }
            for name in [&s.airborne_emissions, &s.cumulative_emissions]
                .into_iter()
                .flatten()
            {
                definitions.push(VariableDefinition::scalar_output(name, &s.amount_unit));
            }
            if let Some(name) = &s.lifetime_scaling {
                definitions.push(
                    VariableDefinition::scalar_input(name, "1")
                        .with_timeseries_type(TimeseriesType::Average),
                );
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
            .downcast_mut::<GasCycleState>()
            .ok_or_else(|| RSCMError::Error("Internal state is not GasCycleState".into()))?;
        self.solve_impl(t_current, t_next, input_state, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rscm_core::names::{ATMOSPHERIC_CONCENTRATIONS_CH4, EMISSIONS_CH4, INVERSE_EMISSIONS_CH4};

    fn ch4() -> GasSpecies {
        GasSpecies {
            driver: Driver::Emissions,
            emissions: vec![EMISSIONS_CH4.into()],
            inverse_emissions: Some(INVERSE_EMISSIONS_CH4.into()),
            emissions_unit: "Mt CH4/yr".into(),
            amount_unit: "Mt CH4".into(),
            concentration: ATMOSPHERIC_CONCENTRATIONS_CH4.into(),
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
            let c = GasCycle::step_emissions(&species, &mut forward, emissions, 1.2, 1.0);
            let e = GasCycle::step_concentration(&species, &mut inverse, c, 1.2, 1.0);
            assert_relative_eq!(e, emissions, max_relative = 1e-10);
        }
    }

    #[test]
    fn deserialising_checks_the_parameters() {
        let mut parameters = GasCycleParameters {
            species: vec![ch4()],
            maximum_iirf: default_maximum_iirf(),
            surface_temperature: surface_temperature(),
        };
        let valid =
            serde_json::to_string(&GasCycle::from_parameters(parameters.clone()).unwrap()).unwrap();
        assert!(serde_json::from_str::<GasCycle>(&valid).is_ok());
        parameters.species[0].unperturbed_lifetime.pop();
        let json = format!(
            r#"{{"parameters": {}}}"#,
            serde_json::to_string(&parameters).unwrap()
        );
        assert!(serde_json::from_str::<GasCycle>(&json).is_err());
    }

    #[test]
    fn rejects_cumulative_emissions() {
        let mut species = ch4();
        species.driver = Driver::CumulativeEmissions;
        let parameters = GasCycleParameters {
            species: vec![species],
            maximum_iirf: default_maximum_iirf(),
            surface_temperature: surface_temperature(),
        };
        assert!(GasCycle::from_parameters(parameters).is_err());
    }

    #[test]
    fn single_box_decays_with_lifetime() {
        let species = ch4();
        let mut state = GasState {
            boxes: vec![100.0, 0.0, 0.0, 0.0],
            ..GasState::default()
        };
        GasCycle::step_emissions(&species, &mut state, 0.0, 1.0, 1.0);
        assert_relative_eq!(state.boxes[0], 100.0 * (-1.0 / 8.25_f64).exp());
    }
}
