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
//! - `Surface Temperature` (K) - Global mean surface temperature anomaly
//! - `Emissions|NOx` (Tg N/yr) - NOx emissions for OH feedback
//! - `Emissions|CO` (Tg CO/yr) - CO emissions for OH feedback
//! - `Emissions|NMVOC` (Tg NMVOC/yr) - NMVOC emissions for OH feedback
//!
//! # Outputs
//!
//! - `Atmospheric Concentration|CH4` (ppb) - Updated CH4 concentration
//! - `Lifetime|CH4` (yr) - Effective total CH4 lifetime
//!
//! # Feedback references
//!
//! As in MAGICC7, the concentration and temperature feedbacks are measured from
//! the year before `feedback_start_year` and the NOx, CO and NMVOC feedbacks from
//! emissions in `feedback_start_year`. Before then the reference concentration
//! tracks the current concentration and the temperature feedback is zero.
//! References outside the run are clamped to its first or last year.
//!
//! # Differences from MAGICC7 Module 01
//!
//! - **Budget initialisation**: MAGICC7 calculates natural emissions to close the
//!   budget over a reference period. Here, natural emissions are a fixed parameter.
//! - **Wetland feedback**: MAGICC7 includes temperature-dependent wetland CH4
//!   emissions. Not implemented here - would need separate natural emissions component.
//! - **Clathrate release**: MAGICC7 has optional methane clathrate feedback. Not implemented.
//! - **Prescribed/calculated switch**: MAGICC7 can switch between prescribed concentrations
//!   and emissions-driven calculation by year. Here, always emissions-driven.

use crate::parameters::CH4ChemistryParameters;
use crate::reference::value_at_year;
use rscm_core::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Number of Prather iterations (fixed per MAGICC7 implementation)
const PRATHER_ITERATIONS: usize = 4;

/// CH4 atmospheric chemistry component
///
/// Implements Prather's iterative method for solving the nonlinear methane
/// mass balance equation. The method accounts for the positive feedback where
/// higher CH4 concentrations deplete tropospheric OH, which in turn increases
/// CH4 lifetime.
///
/// # Algorithm
///
/// For each timestep, starting from the burden $B$ at the start of the year,
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
    temperature { name = "Surface Temperature", unit = "K" },
    nox_emissions { name = "Emissions|NOx", unit = "Mt N/yr" },
    co_emissions { name = "Emissions|CO", unit = "Mt CO/yr" },
    nmvoc_emissions { name = "Emissions|NMVOC", unit = "Mt NMVOC/yr" },
)]
#[states(
    ch4_concentration { name = "Atmospheric Concentration|CH4", unit = "ppb" },
)]
#[outputs(
    ch4_lifetime { name = "Lifetime|CH4", unit = "yr" },
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
        concentration * self.parameters.ppb_to_tg * self.parameters.mixbox_size
    }

    /// Convert burden (Tg CH4) to concentration (ppb)
    fn burden_to_concentration(&self, burden: FloatValue) -> FloatValue {
        burden / (self.parameters.ppb_to_tg * self.parameters.mixbox_size)
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
        let tau_oh = self.parameters.tau_oh_init();
        if !self.parameters.include_emissions_feedback {
            return tau_oh;
        }

        let gamma = self.parameters.oh_sensitivity_scale;
        let exponent = -gamma
            * (self.parameters.oh_nox_sensitivity * delta_nox
                + self.parameters.oh_co_sensitivity * delta_co
                + self.parameters.oh_nmvoc_sensitivity * delta_nmvoc);

        tau_oh * exponent.exp()
    }

    /// Apply temperature feedback to lifetime
    ///
    /// $$\tau_{adjusted} = \frac{\tau_0}{\tau_0/\tau + \alpha_T \cdot \max(0, \Delta T)}$$
    fn apply_temperature_feedback(
        &self,
        tau_oh: FloatValue,
        temperature_change: FloatValue,
    ) -> FloatValue {
        if !self.parameters.include_temp_feedback {
            return tau_oh;
        }

        let alpha = self.parameters.temp_sensitivity;
        let tau_0 = self.parameters.tau_oh_init();
        tau_0 / (tau_0 / tau_oh + alpha * temperature_change.max(0.0))
    }

    /// Calculate total lifetime from OH lifetime and non-OH sinks
    fn calculate_total_lifetime(&self, tau_oh: FloatValue) -> FloatValue {
        let tau_other = self.parameters.tau_other();
        1.0 / (1.0 / tau_oh + 1.0 / tau_other)
    }

    /// Run the full Prather iteration scheme (4 iterations)
    ///
    /// `temperature_change` and the emission deltas are measured from the
    /// feedback references. Returns the concentration at the end of the year
    /// and the total lifetime.
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
    ) -> (FloatValue, FloatValue) {
        let total_emissions = anthropogenic_emissions + self.parameters.natural_emissions;
        let burden = self.concentration_to_burden(ch4_start);
        let burden_reference = self.concentration_to_burden(ch4_reference);
        let base_lifetime = self.calculate_base_lifetime_factor(delta_nox, delta_co, delta_nmvoc);
        let x = self.feedback_exponent();
        let tau_other = self.parameters.tau_other();

        let mut burden_next = burden;
        let mut delta_burden = 0.0;
        let mut tau_oh = base_lifetime;
        for iteration in 0..PRATHER_ITERATIONS {
            let burden_mean = if iteration == 0 {
                burden
            } else {
                (burden + burden_next) / 2.0
            };
            tau_oh = base_lifetime * (burden_mean / burden_reference).max(1.0).powf(x);
            if iteration > 0 {
                tau_oh *= 1.0 - 0.5 * x * delta_burden / burden;
            }
            tau_oh = self.apply_temperature_feedback(tau_oh, temperature_change);
            delta_burden = total_emissions - burden_mean / tau_oh - burden_mean / tau_other;
            burden_next = burden + delta_burden;
        }

        (
            self.burden_to_concentration(burden_next),
            self.calculate_total_lifetime(tau_oh),
        )
    }
}

impl Default for CH4Chemistry {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde]
impl Component for CH4Chemistry {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let inputs = CH4ChemistryInputs::from_input_state(input_state);
        let feedback_year = self.parameters.feedback_start_year;

        let ch4_start = inputs.ch4_concentration.at_start();
        let temperature = inputs.temperature.at_start();
        let (ch4_reference, temperature_reference) = if t_current < feedback_year {
            (ch4_start, temperature)
        } else {
            (
                value_at_year(&inputs.ch4_concentration, t_current, feedback_year - 1.0),
                value_at_year(&inputs.temperature, t_current, feedback_year - 1.0),
            )
        };
        let delta =
            |window: &ScalarWindow| window.get() - value_at_year(window, t_current, feedback_year);

        let (new_concentration, lifetime) = self.solve_concentration(
            ch4_start,
            ch4_reference,
            inputs.ch4_emissions.get(),
            temperature - temperature_reference,
            delta(&inputs.nox_emissions),
            delta(&inputs.co_emissions),
            delta(&inputs.nmvoc_emissions),
        );

        let outputs = CH4ChemistryOutputs {
            ch4_concentration: new_concentration,
            ch4_lifetime: lifetime,
        };

        Ok(outputs.into())
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
        let tau = 1.0 / (1.0 / p.tau_oh_init() + 1.0 / p.tau_other());
        component.burden_to_concentration(p.natural_emissions * tau)
    }

    #[test]
    fn test_steady_state_with_natural_emissions() {
        let component = default_component();
        let ch4 = natural_steady_state(&component);
        let (new_conc, _) = component.solve_concentration(ch4, ch4, 0.0, 0.0, 0.0, 0.0, 0.0);
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
        let (new_conc, lifetime) = component.solve_concentration(
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
    fn test_emissions_increase_raises_concentration() {
        let component = default_component();
        let ch4 = natural_steady_state(&component);
        let (new_conc, _) = component.solve_concentration(ch4, ch4, 300.0, 0.0, 0.0, 0.0, 0.0);
        assert!(new_conc > ch4);
    }

    #[test]
    fn test_higher_emissions_higher_concentration() {
        let component = default_component();
        let (conc_low, _) = component.solve_concentration(722.0, 722.0, 200.0, 0.0, 0.0, 0.0, 0.0);
        let (conc_high, _) = component.solve_concentration(722.0, 722.0, 400.0, 0.0, 0.0, 0.0, 0.0);
        assert!(conc_high > conc_low);
    }

    #[test]
    fn test_warming_shortens_lifetime() {
        let component = default_component();
        let (_, lifetime_cold) =
            component.solve_concentration(722.0, 722.0, 300.0, 0.0, 0.0, 0.0, 0.0);
        let (_, lifetime_warm) =
            component.solve_concentration(722.0, 722.0, 300.0, 2.0, 0.0, 0.0, 0.0);
        assert!(lifetime_warm < lifetime_cold);
    }

    #[test]
    fn test_cooling_does_not_change_lifetime() {
        let component = default_component();
        let (_, lifetime_ref) =
            component.solve_concentration(722.0, 722.0, 300.0, 0.0, 0.0, 0.0, 0.0);
        let (_, lifetime_cool) =
            component.solve_concentration(722.0, 722.0, 300.0, -1.0, 0.0, 0.0, 0.0);
        assert!((lifetime_cool - lifetime_ref).abs() < 1e-12);
    }

    #[test]
    fn test_temperature_feedback_disabled() {
        let params = CH4ChemistryParameters {
            include_temp_feedback: false,
            ..Default::default()
        };
        let component = CH4Chemistry::from_parameters(params);
        let (_, lifetime_cold) =
            component.solve_concentration(722.0, 722.0, 300.0, 0.0, 0.0, 0.0, 0.0);
        let (_, lifetime_warm) =
            component.solve_concentration(722.0, 722.0, 300.0, 2.0, 0.0, 0.0, 0.0);
        assert!((lifetime_warm - lifetime_cold).abs() < 1e-10);
    }

    #[test]
    fn test_higher_concentration_longer_lifetime() {
        let component = default_component();
        let (_, lifetime_low) =
            component.solve_concentration(722.0, 722.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (_, lifetime_high) =
            component.solve_concentration(1800.0, 722.0, 300.0, 0.0, 0.0, 0.0, 0.0);
        assert!(lifetime_high > lifetime_low);
    }

    #[test]
    fn test_nox_emissions_shorten_lifetime() {
        let component = default_component();
        let (_, lifetime_no_nox) =
            component.solve_concentration(722.0, 722.0, 300.0, 0.0, 0.0, 0.0, 0.0);
        let (_, lifetime_with_nox) =
            component.solve_concentration(722.0, 722.0, 300.0, 0.0, 50.0, 0.0, 0.0);
        assert!(lifetime_with_nox < lifetime_no_nox);
    }

    #[test]
    fn test_co_emissions_lengthen_lifetime() {
        let component = default_component();
        let (_, lifetime_no_co) =
            component.solve_concentration(722.0, 722.0, 300.0, 0.0, 0.0, 0.0, 0.0);
        let (_, lifetime_with_co) =
            component.solve_concentration(722.0, 722.0, 300.0, 0.0, 0.0, 1000.0, 0.0);
        assert!(lifetime_with_co > lifetime_no_co);
    }

    #[test]
    fn test_definitions() {
        let component = default_component();
        let defs = component.definitions();
        let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"Emissions|CH4"));
        assert!(names.contains(&"Surface Temperature"));
        assert!(names.contains(&"Atmospheric Concentration|CH4"));
        assert!(names.contains(&"Lifetime|CH4"));
    }

    #[test]
    fn test_zero_emissions_decay() {
        let params = CH4ChemistryParameters {
            natural_emissions: 0.0,
            ..Default::default()
        };
        let component = CH4Chemistry::from_parameters(params);
        let (new_conc, _) = component.solve_concentration(1800.0, 1800.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(new_conc < 1800.0);
    }

    #[test]
    fn test_very_high_concentration() {
        let component = default_component();
        let (new_conc, lifetime) =
            component.solve_concentration(10000.0, 722.0, 300.0, 0.0, 0.0, 0.0, 0.0);
        assert!(new_conc > 0.0 && new_conc.is_finite());
        assert!(lifetime.is_finite());
    }

    #[test]
    fn test_low_concentration_floor() {
        // Below the reference the concentration feedback is clamped, so the
        // reference value no longer matters
        let component = default_component();
        let (_, lifetime_a) = component.solve_concentration(100.0, 722.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (_, lifetime_b) = component.solve_concentration(100.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((lifetime_a - lifetime_b).abs() < 1e-12);
    }
}
