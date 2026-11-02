//! Ocean Carbon Cycle Component
//!
//! Simulates the uptake of CO2 by the global ocean through air-sea gas exchange
//! using an Impulse Response Function (IRF) approach.
//!
//! # What This Component Does
//!
//! 1. Calculates air-sea CO2 flux based on the partial pressure difference
//!    between atmosphere and ocean surface
//!
//! 2. Tracks how absorbed carbon mixes into the deep ocean using an IRF
//!    convolution that "remembers" past flux history
//!
//! 3. Updates ocean surface pCO2 accounting for:
//!    - DIC accumulation (Joos A24 polynomial)
//!    - Temperature effect on solubility (Joos A25)
//!
//! 4. Uses monthly sub-stepping (12 steps per year). As in MAGICC7, atmospheric
//!    CO2 evolves within the year from the other carbon sources and the ocean
//!    uptake, so the uptake responds to the rising concentration.
//!
//! # Inputs
//!
//! - `Atmospheric Concentration|CO2` (ppm) - Concentration at the start of the year
//! - `Surface Temperature` (K) - Drives the pCO2 feedback through the warming
//!   since `tempfeedback_yrstart`, floored at zero
//! - `Emissions|CO2|Fossil` (GtC/yr) - Fossil and industrial emissions
//! - `Emissions|CO2|CH4 Oxidation` (GtC/yr) - CO2 from oxidised fossil CH4
//! - `Carbon Flux|Terrestrial` (GtC/yr) - Net land uptake, including land use
//!
//! # States (tracked between timesteps)
//!
//! - `Ocean Surface pCO2` (ppm) - Ocean surface CO2 partial pressure
//! - `Cumulative Ocean Uptake` (GtC) - Total carbon absorbed by ocean
//!
//! # Outputs
//!
//! - `Carbon Flux|Ocean` (GtC/yr) - Annual mean air-sea flux (positive = ocean uptake)
//!
//! # Differences from MAGICC7 Module 10
//!
//! - **Three IRF models**: 3D-GFDL (default), 2D-BERN, and HILDA are
//!   implemented. MAGICC7 also supports BOXDIFF.
//! - **No radiative-only mode**: MAGICC7 can make ocean "see" only PI CO2.
//!   Not implemented.
//! - **No biological pump**: Only abiotic chemistry is represented.
//! - **No circulation changes**: IRF is static, doesn't respond to warming.

use std::any::Any;
use std::collections::VecDeque;

use crate::parameters::OceanCarbonParameters;
use crate::reference::{feedback_warming, require_annual};
use rscm_core::component::{
    Component, ComponentState, GridType, InputState, OutputState, RequirementDefinition,
    RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Internal state for OceanCarbon component.
///
/// Holds the flux history and the last monthly sub-step of the previous year,
/// which persist across solve() calls. Unlike coupled state
/// (RequirementType::State), this is private to the component.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OceanCarbonState {
    /// Flux history for IRF convolution (ppm/month).
    /// Each element represents the flux in one month.
    pub flux_history: VecDeque<FloatValue>,

    /// Atmospheric CO2 one sub-step before the start of the year (ppm).
    pub co2_previous: Option<FloatValue>,

    /// Ocean pCO2 one sub-step before the start of the year (ppm).
    pub pco2_previous: Option<FloatValue>,

    /// Unperturbed ocean pCO2 (ppm), fixed on the first step.
    pub pco2_reference: Option<FloatValue>,
}

#[typetag::serde]
impl ComponentState for OceanCarbonState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Result of one annual ocean solve.
#[derive(Debug, Clone, Copy)]
pub struct OceanStepResult {
    /// Ocean pCO2 at the end of the year (ppm)
    pub pco2: FloatValue,
    /// Cumulative ocean uptake at the end of the year (GtC)
    pub cumulative_uptake: FloatValue,
    /// Annual mean air-sea flux (GtC/yr)
    pub flux: FloatValue,
    /// Atmospheric CO2 at the end of the year implied by the sub-steps (ppm)
    pub co2: FloatValue,
}

/// Ocean carbon cycle component using IRF-based deep ocean mixing.
///
/// Implements a MAGICC7-style ocean carbon model using an Impulse Response
/// Function (IRF) to represent carbon transport from the mixed layer to
/// the deep ocean. Supports 3D-GFDL (default), 2D-BERN, and HILDA models.
///
/// # Algorithm
///
/// For each monthly sub-step:
///
/// 1. Calculate air-sea flux from the mean of the last two sub-steps:
///    $$F = k \times (\overline{pCO2}_{atm} - \overline{pCO2}_{ocn})$$
///    limiting its change from the previous sub-step to `flux_change_limit`
///
/// 2. Add flux to history and convolve with IRF to get $\Delta DIC$:
///    $$\Delta DIC = \int_0^t F(t') \times IRF(t - t') \, dt'$$
///
/// 3. Calculate $\Delta pCO2$ from $\Delta DIC$ using Joos A24 polynomial
///
/// 4. Apply temperature effect (Joos A25):
///    $$pCO2_{ocn} = (pCO2_{pi} + \Delta pCO2) \times e^{\alpha_T \Delta T}$$
///
/// 5. Update atmospheric CO2 with the other sources and the ocean uptake
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["carbon-cycle", "ocean", "magicc"], category = "Carbon Cycle")]
#[inputs(
    co2_concentration { name = "Atmospheric Concentration|CO2", unit = "ppm" },
    temperature { name = "Surface Temperature", unit = "K" },
    fossil_emissions { name = "Emissions|CO2|Fossil", unit = "GtC/yr" },
    ch4_oxidation { name = "Emissions|CO2|CH4 Oxidation", unit = "GtC/yr" },
    terrestrial_flux { name = "Carbon Flux|Terrestrial", unit = "GtC/yr" },
)]
#[states(
    ocean_pco2 { name = "Ocean Surface pCO2", unit = "ppm" },
    cumulative_uptake { name = "Cumulative Ocean Uptake", unit = "GtC" },
)]
#[outputs(
    air_sea_flux { name = "Carbon Flux|Ocean", unit = "GtC/yr" },
)]
pub struct OceanCarbon {
    parameters: OceanCarbonParameters,
}

impl OceanCarbon {
    /// Create a new ocean carbon component with default parameters.
    pub fn new() -> Self {
        Self::from_parameters(OceanCarbonParameters::default())
    }

    /// Create a new ocean carbon component from parameters.
    pub fn from_parameters(parameters: OceanCarbonParameters) -> Self {
        Self { parameters }
    }

    /// Calculate air-sea carbon flux (ppm/month).
    ///
    /// $$F = k \times (pCO2_{atm} - pCO2_{ocn})$$
    ///
    /// Positive values are fluxes into the ocean.
    fn calculate_flux(&self, pco2_atm: FloatValue, pco2_ocn: FloatValue) -> FloatValue {
        self.parameters.gas_exchange_rate() * (pco2_atm - pco2_ocn)
    }

    /// Calculate delta DIC via IRF convolution.
    ///
    /// $$\Delta DIC = \text{factor} \times \sum_{i=1}^{n} F_i \times IRF(t_n - t_i) \times \Delta t$$
    ///
    /// Uses a left-hand Riemann sum over the sub-step fluxes (ppm per sub-step).
    /// Returns the change in DIC (micromol/kg).
    fn calculate_delta_dic(&self, flux_history: &VecDeque<FloatValue>) -> FloatValue {
        let n = flux_history.len();
        let steps = self.parameters.steps_per_year as FloatValue;
        let irf = self.parameters.irf_evaluator();
        let integral: FloatValue = flux_history
            .iter()
            .enumerate()
            .map(|(i, &flux)| flux * irf((n - 1 - i) as FloatValue / steps))
            .sum();
        integral * self.parameters.dic_conversion_factor()
    }

    /// Solve the ocean carbon cycle for one timestep with monthly sub-steps.
    ///
    /// `other_sources` is the net carbon added to the atmosphere by everything
    /// except the ocean (GtC/yr) and `warming` drives the pCO2 temperature
    /// feedback (K).
    #[allow(clippy::too_many_arguments)]
    pub fn solve_ocean(
        &self,
        state: &mut OceanCarbonState,
        co2_start: FloatValue,
        other_sources: FloatValue,
        warming: FloatValue,
        pco2_start: FloatValue,
        cumulative_start: FloatValue,
        dt: FloatValue,
    ) -> OceanStepResult {
        let p = &self.parameters;
        let steps = p.steps_per_year as FloatValue;
        let pco2_reference = *state
            .pco2_reference
            .get_or_insert(p.pco2_pi.unwrap_or(pco2_start));

        let mut co2 = co2_start;
        let mut co2_previous = state.co2_previous.unwrap_or(co2_start);
        let mut pco2 = pco2_start;
        let mut pco2_previous = state.pco2_previous.unwrap_or(pco2_start);
        let sources_per_step = other_sources * dt / steps / p.gtc_per_ppm;
        let flux_change_limit = p.flux_change_limit * 12.0 / steps;
        let mut total_flux = 0.0;

        for _ in 0..p.steps_per_year {
            // Flux in ppm/yr from the monthly exchange rate, using the means of the last two sub-steps
            let mut flux = 12.0
                * self.calculate_flux((co2_previous + co2) / 2.0, (pco2_previous + pco2) / 2.0);
            if let (Some(&last), true) = (state.flux_history.back(), flux_change_limit > 0.0) {
                let last = last * steps;
                flux = flux.clamp(last - flux_change_limit, last + flux_change_limit);
            }

            state.flux_history.push_back(flux / steps);
            if state.flux_history.len() > p.max_history_months {
                state.flux_history.pop_front();
            }
            total_flux += flux;

            let delta_pco2 = p.delta_pco2_from_dic(self.calculate_delta_dic(&state.flux_history));
            let pco2_next = p.ocean_pco2(pco2_reference, delta_pco2, warming);
            let co2_next = co2 + sources_per_step - flux * dt / steps;

            co2_previous = co2;
            co2 = co2_next;
            pco2_previous = pco2;
            pco2 = pco2_next;
        }

        state.co2_previous = Some(co2_previous);
        state.pco2_previous = Some(pco2_previous);

        let mean_flux = total_flux / steps * p.gtc_per_ppm;
        OceanStepResult {
            pco2,
            cumulative_uptake: cumulative_start + mean_flux * dt,
            flux: mean_flux,
            co2,
        }
    }

    /// Reset the flux history and sub-step memory (for initialization or testing).
    pub fn reset_history(&self, state: &mut OceanCarbonState) {
        *state = OceanCarbonState::default();
    }

    /// Get the current flux history length (months).
    pub fn history_length(&self, state: &OceanCarbonState) -> usize {
        state.flux_history.len()
    }

    /// Internal solve implementation used by both solve() and solve_with_state().
    fn solve_impl(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
        state: &mut OceanCarbonState,
    ) -> RSCMResult<OutputState> {
        require_annual("OceanCarbon", t_current, t_next)?;
        let inputs = OceanCarbonInputs::from_input_state(input_state);

        let warming = feedback_warming(
            &inputs.temperature,
            t_current,
            self.parameters.tempfeedback_yrstart,
        );
        let other_sources = inputs.fossil_emissions.get() + inputs.ch4_oxidation.get()
            - inputs.terrestrial_flux.get();
        if state.pco2_reference.is_none() {
            let first = -(inputs.ocean_pco2.index() as isize);
            state.pco2_reference = Some(
                self.parameters
                    .pco2_pi
                    .unwrap_or_else(|| inputs.ocean_pco2.at_offset(first).unwrap_or(0.0)),
            );
        }

        let result = self.solve_ocean(
            state,
            inputs.co2_concentration.at_start(),
            other_sources,
            warming,
            inputs.ocean_pco2.at_start(),
            inputs.cumulative_uptake.at_start(),
            t_next - t_current,
        );

        let outputs = OceanCarbonOutputs {
            ocean_pco2: result.pco2,
            cumulative_uptake: result.cumulative_uptake,
            air_sea_flux: result.flux,
        };

        Ok(outputs.into())
    }
}

impl Default for OceanCarbon {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde]
impl Component for OceanCarbon {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        // For standalone use, create a temporary state
        let mut state = OceanCarbonState::default();
        self.solve_impl(t_current, t_next, input_state, &mut state)
    }

    fn create_initial_state(&self) -> Box<dyn ComponentState> {
        Box::new(OceanCarbonState::default())
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
            .downcast_mut::<OceanCarbonState>()
            .expect("OceanCarbon: invalid state type (expected OceanCarbonState)");
        self.solve_impl(t_current, t_next, input_state, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI: FloatValue = 278.0;

    fn default_component() -> OceanCarbon {
        OceanCarbon::from_parameters(OceanCarbonParameters::default())
    }

    /// Run `years` annual steps with constant sources, returning the final result
    fn run_years(
        component: &OceanCarbon,
        state: &mut OceanCarbonState,
        co2: FloatValue,
        other_sources: FloatValue,
        warming: FloatValue,
        years: usize,
    ) -> OceanStepResult {
        let (mut co2, mut pco2, mut cumulative) = (co2, PI, 0.0);
        let mut result = None;
        for _ in 0..years {
            let r =
                component.solve_ocean(state, co2, other_sources, warming, pco2, cumulative, 1.0);
            co2 = r.co2;
            pco2 = r.pco2;
            cumulative = r.cumulative_uptake;
            result = Some(r);
        }
        result.unwrap()
    }

    fn fresh_state() -> OceanCarbonState {
        OceanCarbonState {
            pco2_reference: Some(PI),
            ..Default::default()
        }
    }

    #[test]
    fn test_zero_flux_at_equilibrium() {
        let component = default_component();
        assert!(component.calculate_flux(PI, PI).abs() < 1e-15);
    }

    #[test]
    fn test_flux_sign_follows_gradient() {
        let component = default_component();
        assert!(component.calculate_flux(PI + 100.0, PI) > 0.0);
        assert!(component.calculate_flux(PI, PI + 50.0) < 0.0);
    }

    #[test]
    fn test_flux_proportional_to_gradient() {
        let component = default_component();
        let flux_50 = component.calculate_flux(PI + 50.0, PI);
        let flux_100 = component.calculate_flux(PI + 100.0, PI);
        assert!((flux_100 / flux_50 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_delta_dic_zero_for_empty_history() {
        let component = default_component();
        assert!(component.calculate_delta_dic(&VecDeque::new()).abs() < 1e-15);
    }

    #[test]
    fn test_delta_dic_positive_for_positive_flux() {
        let component = default_component();
        let history: VecDeque<_> = std::iter::repeat_n(0.1, 12).collect();
        assert!(component.calculate_delta_dic(&history) > 0.0);
    }

    #[test]
    fn test_steady_state_at_equilibrium() {
        let component = default_component();
        let mut state = fresh_state();
        let result = run_years(&component, &mut state, PI, 0.0, 0.0, 10);
        assert!(result.flux.abs() < 1e-12);
        assert!((result.co2 - PI).abs() < 1e-12);
        assert!((result.pco2 - PI).abs() < 1e-12);
    }

    #[test]
    fn test_uptake_lowers_atmospheric_co2_within_year() {
        // With no other sources, uptake from an elevated atmosphere draws it down
        let component = default_component();
        let mut state = fresh_state();
        let result = component.solve_ocean(&mut state, 400.0, 0.0, 0.0, PI, 0.0, 1.0);
        assert!(result.flux > 0.0);
        let expected = 400.0 - result.flux / component.parameters.gtc_per_ppm;
        assert!((result.co2 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_sources_raise_uptake() {
        let component = default_component();
        let low = run_years(&component, &mut fresh_state(), PI, 2.0, 0.0, 20);
        let high = run_years(&component, &mut fresh_state(), PI, 8.0, 0.0, 20);
        assert!(high.flux > low.flux && low.flux > 0.0);
        assert!(high.cumulative_uptake > low.cumulative_uptake);
    }

    #[test]
    fn test_warming_reduces_uptake() {
        let component = default_component();
        let cold = run_years(&component, &mut fresh_state(), PI, 8.0, 0.0, 20);
        let warm = run_years(&component, &mut fresh_state(), PI, 8.0, 2.0, 20);
        assert!(warm.flux < cold.flux);
        assert!(warm.pco2 > cold.pco2);
    }

    #[test]
    fn test_flux_change_limited() {
        let component = default_component();
        let mut state = fresh_state();
        component.solve_ocean(&mut state, PI, 0.0, 0.0, PI, 0.0, 1.0);
        // A sudden jump in atmospheric CO2 can only raise the flux gradually
        component.solve_ocean(&mut state, 1000.0, 0.0, 0.0, PI, 0.0, 1.0);
        let months: Vec<_> = state.flux_history.iter().map(|f| f * 12.0).collect();
        for pair in months.windows(2) {
            assert!(pair[1] - pair[0] <= 0.04 + 1e-12);
        }
    }

    #[test]
    fn test_substep_count_only_changes_resolution() {
        let monthly = run_years(&default_component(), &mut fresh_state(), PI, 8.0, 0.0, 20);
        let fine = OceanCarbon::from_parameters(OceanCarbonParameters {
            steps_per_year: 24,
            max_history_months: 12000,
            ..Default::default()
        });
        let fine = run_years(&fine, &mut fresh_state(), PI, 8.0, 0.0, 20);
        assert!((fine.flux / monthly.flux - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_flux_magnitude_reasonable() {
        // Around 2 GtC/yr uptake for a ~100 ppm imbalance after decades of emissions
        let component = default_component();
        let result = run_years(&component, &mut fresh_state(), PI, 8.0, 0.0, 50);
        assert!(
            result.flux > 0.5 && result.flux < 6.0,
            "Got {}",
            result.flux
        );
    }

    #[test]
    fn test_flux_history_bounded() {
        let params = OceanCarbonParameters {
            max_history_months: 24,
            ..Default::default()
        };
        let component = OceanCarbon::from_parameters(params);
        let mut state = fresh_state();
        run_years(&component, &mut state, PI, 5.0, 0.0, 5);
        assert_eq!(component.history_length(&state), 24);
    }

    #[test]
    fn test_history_reset() {
        let component = default_component();
        let mut state = fresh_state();
        run_years(&component, &mut state, PI, 5.0, 0.0, 2);
        component.reset_history(&mut state);
        assert_eq!(component.history_length(&state), 0);
        assert!(state.co2_previous.is_none());
    }

    #[test]
    fn test_definitions() {
        let component = default_component();
        let names: Vec<_> = component
            .definitions()
            .iter()
            .map(|d| d.name.clone())
            .collect();
        for name in [
            "Atmospheric Concentration|CO2",
            "Surface Temperature",
            "Emissions|CO2|Fossil",
            "Emissions|CO2|CH4 Oxidation",
            "Carbon Flux|Terrestrial",
            "Ocean Surface pCO2",
            "Carbon Flux|Ocean",
        ] {
            assert!(names.contains(&name.to_string()), "Missing {name}");
        }
    }

    #[test]
    fn test_component_state_serialization_roundtrip() {
        let component = default_component();
        let mut state = fresh_state();
        run_years(&component, &mut state, PI, 5.0, 0.0, 2);
        let json = serde_json::to_string(&state).unwrap();
        let parsed: OceanCarbonState = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.flux_history.len(), state.flux_history.len());
        assert_eq!(parsed.co2_previous, state.co2_previous);
    }
}
