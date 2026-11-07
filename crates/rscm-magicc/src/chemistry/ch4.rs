//! CH4 Chemistry Component
//!
//! Simulates atmospheric methane chemistry with concentration-dependent lifetime
//! using Prather's iterative method from IPCC TAR.
//!
//! # What This Component Does
//!
//! 1. Calculates the effective CH4 atmospheric lifetime including:
//!    - OH sink with self-feedback (higher CH4 → depleted OH → longer lifetime)
//!    - Non-OH sinks (soil uptake, stratospheric destruction, tropospheric Cl)
//!    - Temperature feedback (warmer → more OH → shorter lifetime)
//!    - Emissions feedback from co-emitted species (NOx, CO, NMVOC)
//!
//! 2. Solves the mass balance equation using 4 Prather iterations:
//!    $$\frac{dB}{dt} = E - \frac{B}{\tau_{OH}} - \frac{B}{\tau_{other}}$$
//!
//! # Inputs
//!
//! - `Emissions|CH4` (Tg CH4/yr) - Anthropogenic methane emissions
//! - `Surface Air Temperature Change` (K) - Global mean surface air temperature change
//! - `Emissions|NOx` (Tg N/yr) - NOx emissions for OH feedback
//! - `Emissions|CO` (Tg CO/yr) - CO emissions for OH feedback
//! - `Emissions|NMVOC` (Tg NMVOC/yr) - NMVOC emissions for OH feedback
//! - `Emissions|CH4|Permafrost` (Mt CH4/yr), only with the `include_permafrost` parameter
//!
//! # Outputs
//!
//! - `Atmospheric Concentrations|CH4` (ppb) - Updated CH4 concentration
//! - `Lifetime|CH4` (yr) - Effective total CH4 lifetime
//! - `Effective Lifetime|CH4` (yr): start-of-year burden over the CH4 removed during the step
//!
//! # Feedback references
//!
//! As in MAGICC7, the concentration and temperature feedbacks are measured from
//! the year before `feedback_start_year` and the NOx, CO and NMVOC feedbacks from
//! emissions in `feedback_start_year`. Before then the reference concentration
//! tracks the current concentration and the temperature feedback is zero.
//! References outside the run are clamped to its first or last year.
//!
//! # Differences from MAGICC7
//!
//! - **Budget initialisation**: natural emissions are a fixed parameter.
//!   `rscm.magicc.full.natural_emissions` calculates MAGICC7's budget-closing value for it.
//! - **Wetland feedback**: MAGICC7 includes temperature-dependent wetland CH4
//!   emissions. These are not implemented, as they need a natural emissions component.
//! - **Clathrate release**: MAGICC7 has optional methane clathrate feedback. Not implemented.
//!
//! With `concentration_switch_year` the concentration follows
//! `Prescribed Atmospheric Concentrations|CH4` until that year, and `Inverse Emissions|CH4`
//! holds the anthropogenic emissions that would reproduce it.

use crate::parameters::CH4ChemistryParameters;
use crate::reference::{
    feedback_warming, permafrost_emissions, prescribed_concentration, prescribed_definition,
    reference_value, require_annual, value_at_year,
};
use rscm_core::component::{
    Component, GridType, InputState, OutputState, VariableDefinition, VariableRole,
};
use rscm_core::errors::RSCMResult;
use rscm_core::names::{
    ATMOSPHERIC_CONCENTRATIONS_CH4, EMISSIONS_CH4_PERMAFROST, INVERSE_EMISSIONS_CH4,
};
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Number of Prather iterations (fixed per MAGICC7 implementation)
const PRATHER_ITERATIONS: usize = 4;

/// Root of a near-linear `f` by the secant method from two starting guesses.
fn secant(
    f: impl Fn(FloatValue) -> FloatValue,
    mut x0: FloatValue,
    mut x1: FloatValue,
) -> FloatValue {
    let mut f0 = f(x0);
    for _ in 0..50 {
        let f1 = f(x1);
        if f1 == 0.0 || f1 == f0 {
            break;
        }
        let x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        (x0, f0, x1) = (x1, f1, x2);
        if (x1 - x0).abs() < 1e-12 * x1.abs().max(1.0) {
            break;
        }
    }
    x1
}

/// CH4 atmospheric chemistry component
///
/// Implements Prather's iterative method for solving the nonlinear methane
/// mass balance equation. The method accounts for the positive feedback where
/// higher CH4 concentrations deplete tropospheric OH, which in turn increases
/// CH4 lifetime.
///
/// # Algorithm
///
/// For each step, starting from the burden $B$ at the start of the year,
/// performs 4 fixed-point iterations:
///
/// 1. Mean burden: $\bar{B}_1 = B$, then $\bar{B}_n = (B + B_{n-1}) / 2$
/// 2. Calculate effective OH lifetime with feedbacks
/// 3. Calculate burden change: $\Delta B_n = E - \bar{B}_n/\tau_{OH} - \bar{B}_n/\tau_{other}$
/// 4. Update burden: $B_n = B + \Delta B_n$
///
/// From iteration 2 onwards, a correction factor $(1 - 0.5 X \Delta B_{n-1}/B)$
/// is applied to improve convergence.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["chemistry", "ch4", "magicc"], category = "Atmospheric Chemistry")]
#[inputs(
    ch4_emissions { name = "Emissions|CH4", unit = "Mt CH4/yr" },
    temperature { name = "Surface Air Temperature Change", unit = "K" },
    nox_emissions { name = "Emissions|NOx", unit = "Mt N/yr" },
    co_emissions { name = "Emissions|CO", unit = "Mt CO/yr" },
    nmvoc_emissions { name = "Emissions|NMVOC", unit = "Mt NMVOC/yr" },
)]
#[states(
    ch4_concentration { name = "Atmospheric Concentrations|CH4", unit = "ppb" },
)]
#[outputs(
    ch4_lifetime { name = "Lifetime|CH4", unit = "yr", timeseries_type = "average" },
    ch4_oh_lifetime { name = "Lifetime|CH4|OH", unit = "yr", timeseries_type = "average" },
    ch4_effective_lifetime { name = "Effective Lifetime|CH4", unit = "yr", timeseries_type = "average" },
)]
pub struct CH4Chemistry {
    parameters: CH4ChemistryParameters,
}

impl CH4Chemistry {
    /// Create a new CH4 chemistry component with default parameters
    pub fn new() -> Self {
        Self::from_parameters(CH4ChemistryParameters::default())
    }

    /// Create a new CH4 chemistry component from parameters
    pub fn from_parameters(parameters: CH4ChemistryParameters) -> Self {
        Self { parameters }
    }

    /// Convert concentration (ppb) to burden (Tg CH4)
    fn concentration_to_burden(&self, concentration: FloatValue) -> FloatValue {
        concentration * self.parameters.tg_per_ppb * self.parameters.mixing_box_size
    }

    /// Convert burden (Tg CH4) to concentration (ppb)
    fn burden_to_concentration(&self, burden: FloatValue) -> FloatValue {
        burden / (self.parameters.tg_per_ppb * self.parameters.mixing_box_size)
    }

    /// Feedback exponent $X = -\gamma S$
    fn feedback_exponent(&self) -> FloatValue {
        -self.parameters.oh_sensitivity_scale * self.parameters.ch4_self_feedback
    }

    /// Calculate emissions-adjusted base lifetime factor (U in MAGICC7)
    ///
    /// $$U = \tau_{OH,0} \cdot \exp(-\gamma \cdot (A_{NOx} \Delta E_{NOx} + A_{CO} \Delta E_{CO} + A_{VOC} \Delta E_{VOC}))$$
    fn calculate_base_lifetime_factor(
        &self,
        delta_nox: FloatValue,
        delta_co: FloatValue,
        delta_nmvoc: FloatValue,
    ) -> FloatValue {
        let oh_lifetime = self.parameters.initial_oh_lifetime();
        if !self.parameters.include_emissions_feedback {
            return oh_lifetime;
        }

        let gamma = self.parameters.oh_sensitivity_scale;
        let exponent = -gamma
            * (self.parameters.oh_nox_sensitivity * delta_nox
                + self.parameters.oh_co_sensitivity * delta_co
                + self.parameters.oh_nmvoc_sensitivity * delta_nmvoc);

        oh_lifetime * exponent.exp()
    }

    /// Apply temperature feedback to lifetime
    ///
    /// $$\tau_{adjusted} = \frac{\tau_0}{\tau_0/\tau + \alpha_T \cdot \max(0, \Delta T)}$$
    fn apply_temperature_feedback(
        &self,
        oh_lifetime: FloatValue,
        temperature_change: FloatValue,
    ) -> FloatValue {
        if !self.parameters.include_temperature_feedback {
            return oh_lifetime;
        }

        let alpha = self.parameters.temperature_sensitivity;
        let initial_oh_lifetime = self.parameters.initial_oh_lifetime();
        initial_oh_lifetime
            / (initial_oh_lifetime / oh_lifetime + alpha * temperature_change.max(0.0))
    }

    /// Calculate total lifetime from OH lifetime and non-OH sinks
    fn calculate_total_lifetime(&self, oh_lifetime: FloatValue) -> FloatValue {
        let non_oh_lifetime = self.parameters.non_oh_lifetime();
        1.0 / (1.0 / oh_lifetime + 1.0 / non_oh_lifetime)
    }

    /// Run the full Prather iteration scheme (4 iterations)
    ///
    /// `temperature_change` and the emission deltas are measured from the
    /// feedback references. Returns the concentration at the end of the year
    /// and the total and OH lifetimes.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_concentration(
        &self,
        ch4_start: FloatValue,
        ch4_reference: FloatValue,
        anthropogenic_emissions: FloatValue,
        temperature_change: FloatValue,
        delta_nox: FloatValue,
        delta_co: FloatValue,
        delta_nmvoc: FloatValue,
    ) -> (FloatValue, FloatValue, FloatValue) {
        self.solve_with_total_emissions(
            ch4_start,
            ch4_reference,
            anthropogenic_emissions + self.parameters.natural_emissions,
            temperature_change,
            delta_nox,
            delta_co,
            delta_nmvoc,
        )
    }

    /// As [`Self::solve_concentration`], with natural emissions included in `total_emissions`.
    #[allow(clippy::too_many_arguments)]
    fn solve_with_total_emissions(
        &self,
        ch4_start: FloatValue,
        ch4_reference: FloatValue,
        total_emissions: FloatValue,
        temperature_change: FloatValue,
        delta_nox: FloatValue,
        delta_co: FloatValue,
        delta_nmvoc: FloatValue,
    ) -> (FloatValue, FloatValue, FloatValue) {
        let burden = self.concentration_to_burden(ch4_start);
        let burden_reference = self.concentration_to_burden(ch4_reference);
        let base_lifetime = self.calculate_base_lifetime_factor(delta_nox, delta_co, delta_nmvoc);
        let x = self.feedback_exponent();
        let non_oh_lifetime = self.parameters.non_oh_lifetime();

        let mut burden_next = burden;
        let mut delta_burden = 0.0;
        let mut oh_lifetime = base_lifetime;
        for iteration in 0..PRATHER_ITERATIONS {
            let burden_mean = if iteration == 0 {
                burden
            } else {
                (burden + burden_next) / 2.0
            };
            oh_lifetime = base_lifetime * (burden_mean / burden_reference).max(1.0).powf(x);
            if iteration > 0 {
                oh_lifetime *= 1.0 - 0.5 * x * delta_burden / burden;
            }
            oh_lifetime = self.apply_temperature_feedback(oh_lifetime, temperature_change);
            delta_burden =
                total_emissions - burden_mean / oh_lifetime - burden_mean / non_oh_lifetime;
            burden_next = burden + delta_burden;
        }

        (
            self.burden_to_concentration(burden_next),
            self.calculate_total_lifetime(oh_lifetime),
            oh_lifetime,
        )
    }
}

impl Default for CH4Chemistry {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde(name = "magicc.CH4Chemistry")]
impl Component for CH4Chemistry {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let mut definitions = Self::generated_definitions();
        if self.parameters.include_permafrost {
            definitions.push(VariableDefinition::scalar_input(
                EMISSIONS_CH4_PERMAFROST,
                "Mt CH4/yr",
            ));
        }
        definitions.extend(prescribed_definition(
            ATMOSPHERIC_CONCENTRATIONS_CH4,
            "ppb",
            self.parameters.concentration_switch_year,
        ));
        if self.parameters.concentration_switch_year.is_some() {
            definitions.push(VariableDefinition::scalar_output(
                INVERSE_EMISSIONS_CH4,
                "Mt CH4/yr",
            ));
        }
        definitions
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        require_annual("CH4Chemistry", t_current, t_next)?;
        let inputs = CH4ChemistryInputs::from_input_state(input_state);
        let feedback_year = self.parameters.feedback_start_year;

        let ch4_start = inputs.ch4_concentration.get();
        let ch4_reference = reference_value(&inputs.ch4_concentration, t_current, feedback_year);
        let warming = feedback_warming(&inputs.temperature, t_current, feedback_year - 1.0);
        let delta =
            |window: &ScalarWindow| window.get() - value_at_year(window, t_current, feedback_year);

        let anthropogenic = inputs.ch4_emissions.get();
        let permafrost = permafrost_emissions(
            self.parameters.include_permafrost,
            input_state,
            EMISSIONS_CH4_PERMAFROST,
        );
        let (delta_nox, delta_co, delta_nmvoc) = (
            delta(&inputs.nox_emissions),
            delta(&inputs.co_emissions),
            delta(&inputs.nmvoc_emissions),
        );
        // MAGICC7 adds permafrost emissions after natural ones
        let solve = |emissions: FloatValue| {
            self.solve_with_total_emissions(
                ch4_start,
                ch4_reference,
                emissions + self.parameters.natural_emissions + permafrost,
                warming,
                delta_nox,
                delta_co,
                delta_nmvoc,
            )
        };
        let (mut new_concentration, lifetime, oh_lifetime) = solve(anthropogenic);

        let mut inverse_emissions = anthropogenic;
        if let Some(prescribed) = prescribed_concentration(
            input_state,
            ATMOSPHERIC_CONCENTRATIONS_CH4,
            self.parameters.concentration_switch_year,
            t_next,
        ) {
            inverse_emissions = secant(
                |e| solve(e).0 - prescribed,
                anthropogenic,
                anthropogenic + self.concentration_to_burden(prescribed - new_concentration),
            );
            new_concentration = prescribed;
        }
        let burden = self.concentration_to_burden(ch4_start);
        let sink = burden + inverse_emissions + permafrost + self.parameters.natural_emissions
            - self.concentration_to_burden(new_concentration);

        let mut outputs: OutputState = CH4ChemistryOutputs {
            ch4_concentration: new_concentration,
            ch4_lifetime: lifetime,
            ch4_oh_lifetime: oh_lifetime,
            ch4_effective_lifetime: burden / sink,
        }
        .into();
        if self.parameters.concentration_switch_year.is_some() {
            outputs.insert(
                INVERSE_EMISSIONS_CH4.to_string(),
                StateValue::Scalar(inverse_emissions),
            );
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_component() -> CH4Chemistry {
        CH4Chemistry::from_parameters(CH4ChemistryParameters::default())
    }

    /// Concentration from which the default natural emissions alone are in balance
    fn natural_steady_state(component: &CH4Chemistry) -> FloatValue {
        let p = &component.parameters;
        let tau = 1.0 / (1.0 / p.initial_oh_lifetime() + 1.0 / p.non_oh_lifetime());
        component.burden_to_concentration(p.natural_emissions * tau)
    }

    #[test]
    fn test_steady_state_with_natural_emissions() {
        let component = default_component();
        let ch4 = natural_steady_state(&component);
        let (new_conc, _, _) = component.solve_concentration(ch4, ch4, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(
            (new_conc - ch4).abs() < 1e-9,
            "Expected steady state at {ch4}, got {new_conc}"
        );
    }

    #[test]
    fn test_matches_magicc_first_year() {
        // MAGICC7 v7.5.3 SSP245, emissions-driven from 1750: 731.406 -> 731.767 ppb.
        // Emission deltas are relative to 1927 emissions.
        let component = default_component();
        let (new_conc, lifetime, _) = component.solve_concentration(
            731.406,
            731.406,
            19.01975,
            0.0,
            3.8759306 - 8.49682,
            348.5278 - 515.091,
            60.02186 - 86.5276,
        );
        assert!((new_conc - 731.76674).abs() < 1e-4, "Got {new_conc}");
        assert!((lifetime - 10.040079).abs() < 1e-5, "Got {lifetime}");
    }

    #[test]
    fn test_secant_inverts_prather_iteration() {
        let component = default_component();
        let solve = |e| {
            component
                .solve_concentration(1800.0, 1800.0, e, 0.5, 0.0, 0.0, 0.0)
                .0
        };
        let target = solve(350.0);
        let emissions = secant(|e| solve(e) - target, 300.0, 310.0);
        assert!((emissions - 350.0).abs() < 1e-8, "Got {emissions}");
    }

    #[test]
    fn test_emissions_increase_raises_concentration() {
        let component = default_component();
        let ch4 = natural_steady_state(&component);
        let (new_conc, _, _) = component.solve_concentration(ch4, ch4, 300.0, 0.0, 0.0, 0.0, 0.0);
        assert!(new_conc > ch4);
    }

    #[test]
    fn test_higher_emissions_higher_concentration() {
        let component = default_component();
        let (conc_low, _, _) =
            component.solve_concentration(722.0, 722.0, 200.0, 0.0, 0.0, 0.0, 0.0);
        let (conc_high, _, _) =
            component.solve_concentration(722.0, 722.0, 400.0, 0.0, 0.0, 0.0, 0.0);
        assert!(conc_high > conc_low);
    }

    #[test]
    fn test_warming_shortens_lifetime() {
        let component = default_component();
        let (_, lifetime_cold, _) =
            component.solve_concentration(722.0, 722.0, 300.0, 0.0, 0.0, 0.0, 0.0);
        let (_, lifetime_warm, _) =
            component.solve_concentration(722.0, 722.0, 300.0, 2.0, 0.0, 0.0, 0.0);
        assert!(lifetime_warm < lifetime_cold);
    }

    #[test]
    fn test_cooling_does_not_change_lifetime() {
        let component = default_component();
        let (_, lifetime_ref, _) =
            component.solve_concentration(722.0, 722.0, 300.0, 0.0, 0.0, 0.0, 0.0);
        let (_, lifetime_cool, _) =
            component.solve_concentration(722.0, 722.0, 300.0, -1.0, 0.0, 0.0, 0.0);
        assert!((lifetime_cool - lifetime_ref).abs() < 1e-12);
    }

    #[test]
    fn test_temperature_feedback_disabled() {
        let params = CH4ChemistryParameters {
            include_temperature_feedback: false,
            ..Default::default()
        };
        let component = CH4Chemistry::from_parameters(params);
        let (_, lifetime_cold, _) =
            component.solve_concentration(722.0, 722.0, 300.0, 0.0, 0.0, 0.0, 0.0);
        let (_, lifetime_warm, _) =
            component.solve_concentration(722.0, 722.0, 300.0, 2.0, 0.0, 0.0, 0.0);
        assert!((lifetime_warm - lifetime_cold).abs() < 1e-10);
    }

    #[test]
    fn test_higher_concentration_longer_lifetime() {
        let component = default_component();
        let (_, lifetime_low, _) =
            component.solve_concentration(722.0, 722.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (_, lifetime_high, _) =
            component.solve_concentration(1800.0, 722.0, 300.0, 0.0, 0.0, 0.0, 0.0);
        assert!(lifetime_high > lifetime_low);
    }

    #[test]
    fn test_nox_emissions_shorten_lifetime() {
        let component = default_component();
        let (_, lifetime_no_nox, _) =
            component.solve_concentration(722.0, 722.0, 300.0, 0.0, 0.0, 0.0, 0.0);
        let (_, lifetime_with_nox, _) =
            component.solve_concentration(722.0, 722.0, 300.0, 0.0, 50.0, 0.0, 0.0);
        assert!(lifetime_with_nox < lifetime_no_nox);
    }

    #[test]
    fn test_co_emissions_lengthen_lifetime() {
        let component = default_component();
        let (_, lifetime_no_co, _) =
            component.solve_concentration(722.0, 722.0, 300.0, 0.0, 0.0, 0.0, 0.0);
        let (_, lifetime_with_co, _) =
            component.solve_concentration(722.0, 722.0, 300.0, 0.0, 0.0, 1000.0, 0.0);
        assert!(lifetime_with_co > lifetime_no_co);
    }

    #[test]
    fn test_definitions() {
        let component = default_component();
        let defs = component.definitions();
        let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"Emissions|CH4"));
        assert!(names.contains(&"Surface Air Temperature Change"));
        assert!(names.contains(&"Atmospheric Concentrations|CH4"));
        assert!(names.contains(&"Lifetime|CH4"));
    }

    #[test]
    fn test_zero_emissions_decay() {
        let params = CH4ChemistryParameters {
            natural_emissions: 0.0,
            ..Default::default()
        };
        let component = CH4Chemistry::from_parameters(params);
        let (new_conc, _, _) =
            component.solve_concentration(1800.0, 1800.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(new_conc < 1800.0);
    }

    #[test]
    fn test_very_high_concentration() {
        let component = default_component();
        let (new_conc, lifetime, _) =
            component.solve_concentration(10000.0, 722.0, 300.0, 0.0, 0.0, 0.0, 0.0);
        assert!(new_conc > 0.0 && new_conc.is_finite());
        assert!(lifetime.is_finite());
    }

    #[test]
    fn test_low_concentration_floor() {
        // Below the reference the concentration feedback is clamped, so the
        // reference value no longer matters
        let component = default_component();
        let (_, lifetime_a, _) =
            component.solve_concentration(100.0, 722.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (_, lifetime_b, _) =
            component.solve_concentration(100.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((lifetime_a - lifetime_b).abs() < 1e-12);
    }

    fn constant_item(
        name: &str,
        value: FloatValue,
    ) -> rscm_core::timeseries_collection::TimeseriesItem {
        use rscm_core::timeseries_collection::{TimeseriesData, TimeseriesItem};
        TimeseriesItem {
            data: TimeseriesData::Scalar(rscm_core::timeseries::Timeseries::from_values(
                ndarray::array![value, value],
                ndarray::array![2000.0, 2001.0],
            )),
            name: name.to_string(),
        }
    }

    /// Outputs of one 2000 to 2001 step with constant inputs.
    fn solve_constant(
        component: &CH4Chemistry,
        ch4: FloatValue,
        emissions: FloatValue,
    ) -> OutputState {
        let items = [
            constant_item(ATMOSPHERIC_CONCENTRATIONS_CH4, ch4),
            constant_item("Emissions|CH4", emissions),
            constant_item("Surface Air Temperature Change", 0.0),
            constant_item("Emissions|NOx", 0.0),
            constant_item("Emissions|CO", 0.0),
            constant_item("Emissions|NMVOC", 0.0),
        ];
        let input_state = InputState::build(items.iter().collect(), 2000.0);
        component.solve(2000.0, 2001.0, &input_state).unwrap()
    }

    fn scalar(output: &OutputState, name: &str) -> FloatValue {
        output[name].as_scalar().unwrap()
    }

    #[test]
    fn test_effective_lifetime_in_steady_state_is_total_lifetime() {
        let component = default_component();
        let p = &component.parameters;
        let tau = 1.0 / (1.0 / p.initial_oh_lifetime() + 1.0 / p.non_oh_lifetime());
        let output = solve_constant(&component, natural_steady_state(&component), 0.0);
        let effective = scalar(&output, "Effective Lifetime|CH4");
        assert!((effective / tau - 1.0).abs() < 1e-9, "{effective} vs {tau}");
    }

    #[test]
    fn test_effective_lifetime_closes_burden_budget() {
        // B_next = B + E - B / tau_eff, with E the anthropogenic and natural emissions
        let component = default_component();
        let output = solve_constant(&component, 1800.0, 300.0);
        let effective = scalar(&output, "Effective Lifetime|CH4");
        let burden = component.concentration_to_burden(1800.0);
        let burden_next =
            component.concentration_to_burden(scalar(&output, ATMOSPHERIC_CONCENTRATIONS_CH4));
        let emissions = 300.0 + component.parameters.natural_emissions;
        assert!((burden + emissions - burden / effective - burden_next).abs() < 1e-9);
        // It differs from the total lifetime `Lifetime|CH4`
        assert!((effective - scalar(&output, "Lifetime|CH4")).abs() > 1e-3);
    }
}
