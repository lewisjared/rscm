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
//! 4. Integrates over monthly substeps (12 substeps per year). As in MAGICC7, atmospheric
//!    CO2 evolves within the year from the other carbon sources and the ocean
//!    uptake, so the uptake responds to the rising concentration.
//!    Up to `concentration_switch_year` it is instead interpolated linearly to
//!    `Prescribed Atmospheric Concentrations|CO2`.
//!
//! Biosphere CO2 emissions are part of the net atmosphere to land flux, as in `CO2Budget`.
//!
//! # Inputs
//!
//! - `Emissions|CO2|Permafrost` (GtC/yr), only with the `include_permafrost` parameter
//! - `Prescribed Atmospheric Concentrations|CO2` (ppm), only with the `concentration_switch_year` parameter
//!
//! # Differences from MAGICC7
//!
//! - **Three IRF methods**: 3D-GFDL (default), 2D-BERN, and HILDA are
//!   implemented. MAGICC7 also supports BOXDIFF.
//! - **No radiative-only mode**: MAGICC7 can make the ocean "see" only pre-industrial CO2.
//!   Not implemented.
//! - **No biological pump**: Only abiotic chemistry is represented.
//! - **No circulation changes**: the IRF is static and does not respond to warming.

use rscm_core::names::{ATMOSPHERIC_CONCENTRATIONS_CO2, EMISSIONS_CO2_PERMAFROST};
use std::any::Any;
use std::collections::VecDeque;

use crate::parameters::OceanCarbonParameters;
use crate::reference::{
    feedback_warming, first_value, permafrost_emissions, prescribed_concentration,
    prescribed_definition, require_annual,
};
use rscm_core::component::{
    Component, ComponentState, GridType, InputState, OutputState, VariableDefinition, VariableRole,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Internal state for OceanCarbon component.
///
/// Holds the flux history and the last monthly substep of the previous year,
/// which persist across solve() calls. Unlike coupled state
/// (VariableRole::State), this is private to the component.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OceanCarbonState {
    /// Air-sea flux of each substep for the IRF convolution (ppm/yr).
    pub flux_history: VecDeque<FloatValue>,

    /// First year of the run, which times the IRF lags as MAGICC7 does.
    pub start_year: Option<FloatValue>,

    /// IRF at each lag in substeps, filled as the history grows.
    #[serde(skip)]
    pub irf: Vec<FloatValue>,

    /// Atmospheric CO2 one substep before the start of the year (ppm).
    pub co2_previous: Option<FloatValue>,

    /// Ocean pCO2 one substep before the start of the year (ppm).
    pub pco2_previous: Option<FloatValue>,

    /// Unperturbed ocean pCO2 (ppm), fixed on the first step.
    pub pco2_reference: Option<FloatValue>,
}

#[typetag::serde(name = "magicc.OceanCarbonState")]
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
    /// Atmospheric CO2 at the end of the year implied by the substeps (ppm)
    pub co2: FloatValue,
}

/// Ocean carbon cycle component.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["carbon-cycle", "ocean", "magicc"], category = "Carbon Cycle")]
#[inputs(
    co2_concentration { name = "Atmospheric Concentrations|CO2", unit = "ppm", description = "Atmospheric CO2 concentration at the start of the year" },
    temperature { name = "Surface Air Temperature Change", unit = "K", description = "Drives the pCO2 feedback through the warming since `temperature_feedback_start_year`, floored at zero" },
    fossil_emissions { name = "Emissions|CO2|Fossil", unit = "GtC/yr", description = "Fossil CO2 emissions over the step, which raise CO2 within the year" },
    ch4_oxidation { name = "Emissions|CO2|CH4 Oxidation", unit = "GtC/yr", description = "CO2 from oxidised fossil CH4 over the step" },
    terrestrial_flux { name = "Net Atmosphere to Land Flux|CO2", unit = "GtC/yr", description = "Net atmosphere to land flux over the step, positive for uptake" },
)]
#[states(
    ocean_pco2 { name = "Ocean Surface Partial Pressure|CO2", unit = "ppm", description = "Ocean surface CO2 partial pressure at the end of the step" },
    cumulative_uptake { name = "Cumulative Carbon Uptake|Ocean", unit = "GtC", description = "Total carbon absorbed by the ocean by the end of the step" },
)]
#[outputs(
    air_sea_flux { name = "Net Atmosphere to Ocean Flux|CO2", unit = "GtC/yr", description = "Mean air-sea flux over the step, positive for ocean uptake" },
    co2_end_of_year { name = "Ocean Carbon Atmospheric Concentrations|CO2", unit = "ppm", description = "Atmospheric CO2 after the last substep, which `CO2Budget` carries into the next year" },
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

    /// Calculate air-sea carbon flux (ppm/yr).
    ///
    /// $$F = k \times (pCO2_{atm} - pCO2_{ocn})$$
    ///
    /// Positive values are fluxes into the ocean.
    fn calculate_flux(&self, pco2_atm: FloatValue, pco2_ocn: FloatValue) -> FloatValue {
        (pco2_atm - pco2_ocn) * self.parameters.gas_exchange_rate()
    }

    /// Calculate delta DIC via IRF convolution.
    ///
    /// $$\Delta DIC = \text{factor} \times \sum_{i=1}^{n} F_i \times IRF(t_n - t_i) \times \Delta t$$
    ///
    /// Uses a left-hand Riemann sum over the substep fluxes (ppm/yr).
    /// Returns the change in DIC (micromol/kg).
    fn calculate_delta_dic(&self, state: &mut OceanCarbonState) -> FloatValue {
        let n = state.flux_history.len();
        self.extend_irf(state, n);
        let integral: FloatValue = state
            .flux_history
            .iter()
            .enumerate()
            .map(|(i, &flux)| flux * state.irf[n - 1 - i])
            .sum();
        self.parameters.delta_dic(integral)
    }

    /// Tabulate the IRF up to `len` lags at MAGICC7's substep times, `year + i / steps`.
    fn extend_irf(&self, state: &mut OceanCarbonState, len: usize) {
        if state.irf.len() >= len {
            return;
        }
        let steps = self.parameters.steps_per_year;
        let start = state.start_year.unwrap_or(0.0);
        let irf = self.parameters.irf_evaluator();
        for lag in state.irf.len()..len {
            let year = start + (lag / steps) as FloatValue;
            let time = year + (lag % steps) as FloatValue / steps as FloatValue;
            state.irf.push(irf(time - start));
        }
    }

    /// Solve the ocean carbon cycle for one step with monthly substeps.
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
        self.solve_ocean_prescribed(
            state,
            co2_start,
            None,
            other_sources,
            warming,
            pco2_start,
            cumulative_start,
            dt,
        )
    }

    /// As [`Self::solve_ocean`], but with `co2_end` set atmospheric CO2 is
    /// interpolated linearly to it over the substeps instead of evolving.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_ocean_prescribed(
        &self,
        state: &mut OceanCarbonState,
        co2_start: FloatValue,
        co2_end: Option<FloatValue>,
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
        let limit = p.flux_change_limit;

        for step in 1..=p.steps_per_year {
            let mut flux =
                self.calculate_flux((co2_previous + co2) * 0.5, (pco2_previous + pco2) * 0.5);
            if let (Some(&last), true) = (state.flux_history.back(), limit > 0.0) {
                let change = flux - last;
                if change > limit {
                    flux = limit + last;
                } else if change < -limit {
                    flux = last - limit;
                }
            }

            state.flux_history.push_back(flux);
            if state.flux_history.len() > p.maximum_history_months {
                state.flux_history.pop_front();
            }

            let delta_pco2 = p.delta_pco2_from_dic(self.calculate_delta_dic(state));
            let pco2_next = p.ocean_pco2(pco2_reference, delta_pco2, warming);
            let co2_next = match co2_end {
                Some(end) => (end - co2_start) / steps * step as FloatValue + co2_start,
                None => co2 + (sources_per_step - flux * dt / steps),
            };

            co2_previous = co2;
            co2 = co2_next;
            pco2_previous = pco2;
            pco2 = pco2_next;
        }

        state.co2_previous = Some(co2_previous);
        state.pco2_previous = Some(pco2_previous);

        let year = state.flux_history.len().saturating_sub(p.steps_per_year);
        let total_flux = state.flux_history.range(year..).fold(0.0, |sum, f| sum + f);
        let mean_flux = total_flux * p.gtc_per_ppm / steps;
        OceanStepResult {
            pco2,
            cumulative_uptake: cumulative_start + mean_flux * dt,
            flux: mean_flux,
            co2,
        }
    }

    /// Reset the flux history and substep memory (for initialisation or testing).
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
            self.parameters.temperature_feedback_start_year,
        );
        state.start_year.get_or_insert(t_current.floor());
        let other_sources = (inputs.fossil_emissions.get() + inputs.ch4_oxidation.get()
            - inputs.terrestrial_flux.get())
            + permafrost_emissions(
                self.parameters.include_permafrost,
                input_state,
                EMISSIONS_CO2_PERMAFROST,
            );
        if state.pco2_reference.is_none() {
            state.pco2_reference = Some(
                self.parameters
                    .pco2_pi
                    .unwrap_or_else(|| first_value(&inputs.ocean_pco2)),
            );
        }

        let co2_end = prescribed_concentration(
            input_state,
            ATMOSPHERIC_CONCENTRATIONS_CO2,
            self.parameters.concentration_switch_year,
            t_next,
        );
        let result = self.solve_ocean_prescribed(
            state,
            inputs.co2_concentration.get(),
            co2_end,
            other_sources,
            warming,
            inputs.ocean_pco2.get(),
            inputs.cumulative_uptake.get(),
            t_next - t_current,
        );

        let outputs = OceanCarbonOutputs {
            ocean_pco2: result.pco2,
            cumulative_uptake: result.cumulative_uptake,
            air_sea_flux: result.flux,
            co2_end_of_year: result.co2,
        };

        Ok(outputs.into())
    }
}

impl Default for OceanCarbon {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde(name = "magicc.OceanCarbon")]
impl Component for OceanCarbon {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let mut definitions = Self::generated_definitions();
        if self.parameters.include_permafrost {
            definitions.push(VariableDefinition::scalar_input(
                EMISSIONS_CO2_PERMAFROST,
                "GtC/yr",
            ));
        }
        definitions.extend(prescribed_definition(
            ATMOSPHERIC_CONCENTRATIONS_CO2,
            "ppm",
            self.parameters.concentration_switch_year,
        ));
        definitions
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
        assert!(component.calculate_delta_dic(&mut fresh_state()).abs() < 1e-15);
    }

    #[test]
    fn test_delta_dic_positive_for_positive_flux() {
        let component = default_component();
        let mut state = OceanCarbonState {
            flux_history: std::iter::repeat_n(1.2, 12).collect(),
            ..fresh_state()
        };
        assert!(component.calculate_delta_dic(&mut state) > 0.0);
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
    fn test_prescribed_co2_ignores_sources() {
        let component = default_component();
        let solve = |sources| {
            component.solve_ocean_prescribed(
                &mut fresh_state(),
                400.0,
                Some(402.0),
                sources,
                0.0,
                PI,
                0.0,
                1.0,
            )
        };
        let (with_sources, without) = (solve(50.0), solve(0.0));
        assert_eq!(with_sources.co2, 402.0);
        assert_eq!(with_sources.flux, without.flux);
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
        let fluxes: Vec<_> = state.flux_history.iter().copied().collect();
        for pair in fluxes.windows(2) {
            assert!(pair[1] - pair[0] <= 0.04 + 1e-12);
        }
    }

    #[test]
    fn test_substep_count_only_changes_resolution() {
        let monthly = run_years(&default_component(), &mut fresh_state(), PI, 8.0, 0.0, 20);
        let fine = OceanCarbon::from_parameters(OceanCarbonParameters {
            steps_per_year: 24,
            maximum_history_months: 12000,
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
            maximum_history_months: 24,
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
            "Atmospheric Concentrations|CO2",
            "Surface Air Temperature Change",
            "Emissions|CO2|Fossil",
            "Emissions|CO2|CH4 Oxidation",
            "Net Atmosphere to Land Flux|CO2",
            "Ocean Surface Partial Pressure|CO2",
            "Net Atmosphere to Ocean Flux|CO2",
        ] {
            assert!(names.contains(&name.to_string()), "Missing {name}");
        }
    }

    #[test]
    fn test_component_state_serialisation_roundtrip() {
        let component = default_component();
        let mut state = fresh_state();
        run_years(&component, &mut state, PI, 5.0, 0.0, 2);
        let json = serde_json::to_string(&state).unwrap();
        let parsed: OceanCarbonState = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.flux_history.len(), state.flux_history.len());
        assert_eq!(parsed.co2_previous, state.co2_previous);
    }
}
