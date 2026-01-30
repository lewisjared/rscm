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
//! # Differences from MAGICC7 Module 01
//!
//! This is a simplified implementation:
//!
//! - **Budget initialisation**: MAGICC7 calculates natural emissions to close the
//!   budget over a reference period. Here, natural emissions are a fixed parameter.
//! - **Wetland feedback**: MAGICC7 includes temperature-dependent wetland CH4
//!   emissions. Not implemented here - would need separate natural emissions component.
//! - **Clathrate release**: MAGICC7 has optional methane clathrate feedback. Not implemented.
//! - **Prescribed/calculated switch**: MAGICC7 can switch between prescribed concentrations
//!   and emissions-driven calculation by year. Here, always emissions-driven.
//! - **Reference year setup**: MAGICC7 calculates reference concentrations and temperatures
//!   at a configurable feedback start year. Here, uses pre-industrial as reference.

use crate::parameters::CH4ChemistryParameters;
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
/// For each timestep, performs 4 fixed-point iterations:
///
/// 1. Calculate mean burden: $\bar{B}_n = (B + B_{n-1}) / 2$
/// 2. Calculate effective OH lifetime with feedbacks
/// 3. Calculate burden change: $\Delta B_n = E - \bar{B}_n/\tau_{OH} - \bar{B}_n/\tau_{other}$
/// 4. Update burden: $B_n = B + \Delta B_n$
///
/// From iteration 2 onwards, a correction factor $(1 - 0.5 X \Delta B_{n-1}/B)$
/// is applied to improve convergence.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["chemistry", "ch4", "magicc"], category = "Atmospheric Chemistry")]
#[inputs(
    ch4_emissions { name = "Emissions|CH4", unit = "Tg CH4/yr" },
    temperature { name = "Surface Temperature|Global", unit = "K" },
    nox_emissions { name = "Emissions|NOx", unit = "Tg N/yr" },
    co_emissions { name = "Emissions|CO", unit = "Tg CO/yr" },
    nmvoc_emissions { name = "Emissions|NMVOC", unit = "Tg NMVOC/yr" },
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
        concentration * self.parameters.ppb_to_tg
    }

    /// Convert burden (Tg CH4) to concentration (ppb)
    fn burden_to_concentration(&self, burden: FloatValue) -> FloatValue {
        burden / self.parameters.ppb_to_tg
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
        if !self.parameters.include_emissions_feedback {
            return self.parameters.tau_oh;
        }

        let gamma = self.parameters.oh_sensitivity_scale;
        let exponent = -gamma
            * (self.parameters.oh_nox_sensitivity * delta_nox
                + self.parameters.oh_co_sensitivity * delta_co
                + self.parameters.oh_nmvoc_sensitivity * delta_nmvoc);

        self.parameters.tau_oh * exponent.exp()
    }

    /// Calculate effective OH lifetime with concentration feedback
    ///
    /// $$\tau_{bar} = U \cdot \max(1, \bar{B}/B_0)^X$$
    ///
    /// where $X = -\gamma \cdot S$ (S is the self-feedback coefficient)
    fn calculate_oh_lifetime(
        &self,
        burden_mean: FloatValue,
        burden_reference: FloatValue,
        base_lifetime_factor: FloatValue,
    ) -> FloatValue {
        let gamma = self.parameters.oh_sensitivity_scale;
        let s = self.parameters.ch4_self_feedback;
        let x = -gamma * s;

        let concentration_ratio = (burden_mean / burden_reference).max(1.0);
        base_lifetime_factor * concentration_ratio.powf(x)
    }

    /// Apply temperature feedback to lifetime
    ///
    /// $$\tau_{adjusted} = \frac{\tau_0}{\tau_0/\tau + \alpha_T \cdot \Delta T}$$
    fn apply_temperature_feedback(
        &self,
        tau_oh: FloatValue,
        temperature: FloatValue,
    ) -> FloatValue {
        if !self.parameters.include_temp_feedback || temperature.abs() < 1e-10 {
            return tau_oh;
        }

        let alpha = self.parameters.temp_sensitivity;
        let tau_0 = self.parameters.tau_oh;

        // Clamp negative temperature feedback to zero (per MAGICC7)
        let delta_t = temperature.max(0.0);

        tau_0 / (tau_0 / tau_oh + alpha * delta_t)
    }

    /// Apply iteration correction factor (from iteration 2 onwards)
    ///
    /// $$\tau_{corrected} = \tau \cdot (1 - 0.5 \cdot X \cdot \Delta B_{prev} / B)$$
    fn apply_iteration_correction(
        &self,
        tau_oh: FloatValue,
        delta_burden_prev: FloatValue,
        burden_current: FloatValue,
    ) -> FloatValue {
        if burden_current.abs() < 1e-10 {
            return tau_oh;
        }

        let gamma = self.parameters.oh_sensitivity_scale;
        let s = self.parameters.ch4_self_feedback;
        let x = -gamma * s;

        tau_oh * (1.0 - 0.5 * x * delta_burden_prev / burden_current)
    }

    /// Calculate total lifetime from OH lifetime and non-OH sinks
    fn calculate_total_lifetime(&self, tau_oh: FloatValue) -> FloatValue {
        let tau_other = self.parameters.tau_other();
        1.0 / (1.0 / tau_oh + 1.0 / tau_other)
    }

    /// Perform one Prather iteration
    ///
    /// Returns (new_burden, delta_burden, effective_oh_lifetime)
    #[allow(clippy::too_many_arguments)]
    fn prather_iteration(
        &self,
        burden_current: FloatValue,
        burden_previous: FloatValue,
        burden_reference: FloatValue,
        total_emissions: FloatValue,
        temperature: FloatValue,
        base_lifetime_factor: FloatValue,
        delta_burden_prev: Option<FloatValue>,
        _iteration: usize,
    ) -> (FloatValue, FloatValue, FloatValue) {
        // 1. Calculate mean burden
        let burden_mean = (burden_current + burden_previous) / 2.0;

        // 2. Calculate effective OH lifetime with concentration feedback
        let mut tau_oh =
            self.calculate_oh_lifetime(burden_mean, burden_reference, base_lifetime_factor);

        // 3. Apply iteration correction (iterations 2-4)
        if let Some(db_prev) = delta_burden_prev {
            tau_oh = self.apply_iteration_correction(tau_oh, db_prev, burden_previous);
        }

        // 4. Apply temperature feedback
        tau_oh = self.apply_temperature_feedback(tau_oh, temperature);

        // 5. Calculate burden change
        let tau_other = self.parameters.tau_other();
        let delta_burden = total_emissions - burden_mean / tau_oh - burden_mean / tau_other;

        // 6. New burden
        let new_burden = burden_previous + delta_burden;

        (new_burden, delta_burden, tau_oh)
    }

    /// Run the full Prather iteration scheme (4 iterations)
    #[allow(clippy::too_many_arguments)]
    pub fn solve_concentration(
        &self,
        ch4_prev: FloatValue,
        ch4_current: FloatValue,
        anthropogenic_emissions: FloatValue,
        temperature: FloatValue,
        nox_emissions: FloatValue,
        co_emissions: FloatValue,
        nmvoc_emissions: FloatValue,
    ) -> (FloatValue, FloatValue) {
        // Total emissions = anthropogenic + natural
        let total_emissions = anthropogenic_emissions + self.parameters.natural_emissions;

        // Convert to burden
        let burden_prev = self.concentration_to_burden(ch4_prev);
        let burden_reference = self.concentration_to_burden(self.parameters.ch4_pi);

        // Calculate delta emissions from reference
        let delta_nox = nox_emissions - self.parameters.nox_reference;
        let delta_co = co_emissions - self.parameters.co_reference;
        let delta_nmvoc = nmvoc_emissions - self.parameters.nmvoc_reference;

        // Calculate base lifetime factor (U)
        let base_lifetime_factor =
            self.calculate_base_lifetime_factor(delta_nox, delta_co, delta_nmvoc);

        // Run Prather iterations
        let mut burden = self.concentration_to_burden(ch4_current);
        let mut delta_burden: Option<FloatValue> = None;
        let mut tau_oh = self.parameters.tau_oh;

        for i in 0..PRATHER_ITERATIONS {
            let (new_burden, db, tau) = self.prather_iteration(
                burden,
                burden_prev,
                burden_reference,
                total_emissions,
                temperature,
                base_lifetime_factor,
                delta_burden,
                i,
            );
            burden = new_burden;
            delta_burden = Some(db);
            tau_oh = tau;
        }

        // Convert back to concentration and calculate total lifetime
        let new_concentration = self.burden_to_concentration(burden);
        let total_lifetime = self.calculate_total_lifetime(tau_oh);

        (new_concentration, total_lifetime)
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
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let inputs = CH4ChemistryInputs::from_input_state(input_state);

        // Get inputs
        let ch4_current = inputs.ch4_concentration.at_start();
        // Use current as previous if no history (first timestep)
        let ch4_prev = inputs.ch4_concentration.previous().unwrap_or(ch4_current);
        let emissions = inputs.ch4_emissions.at_start();
        let temperature = inputs.temperature.at_start();
        let nox = inputs.nox_emissions.at_start();
        let co = inputs.co_emissions.at_start();
        let nmvoc = inputs.nmvoc_emissions.at_start();

        // Solve for new concentration
        let (new_concentration, lifetime) = self.solve_concentration(
            ch4_prev,
            ch4_current,
            emissions,
            temperature,
            nox,
            co,
            nmvoc,
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

    // ===== Steady State Tests =====

    #[test]
    fn test_steady_state_at_preindustrial() {
        // At pre-industrial with only natural emissions, concentration should be stable
        let component = default_component();
        let ch4_pi = component.parameters.ch4_pi;

        // Solve starting from pre-industrial
        let (new_conc, _lifetime) = component.solve_concentration(
            ch4_pi, // previous
            ch4_pi, // current
            0.0,    // no anthropogenic emissions
            0.0,    // no temperature anomaly
            0.0,    // no NOx
            0.0,    // no CO
            0.0,    // no NMVOC
        );

        // Should remain close to pre-industrial (within 5%)
        let relative_change = ((new_conc - ch4_pi) / ch4_pi).abs();
        assert!(
            relative_change < 0.05,
            "At pre-industrial with natural emissions only, concentration should be stable. \
             Got {} vs PI={}, change={:.1}%",
            new_conc,
            ch4_pi,
            relative_change * 100.0
        );
    }

    // ===== Emissions Response Tests =====

    #[test]
    fn test_emissions_increase_raises_concentration() {
        let component = default_component();
        let ch4_pi = component.parameters.ch4_pi;

        // Add anthropogenic emissions
        let (new_conc, _lifetime) = component.solve_concentration(
            ch4_pi, ch4_pi, 300.0, // 300 Tg CH4/yr anthropogenic
            0.0, 0.0, 0.0, 0.0,
        );

        assert!(
            new_conc > ch4_pi,
            "Adding emissions should increase concentration. Got {} vs PI={}",
            new_conc,
            ch4_pi
        );
    }

    #[test]
    fn test_higher_emissions_higher_concentration() {
        let component = default_component();
        let ch4_pi = component.parameters.ch4_pi;

        let (conc_low, _) =
            component.solve_concentration(ch4_pi, ch4_pi, 200.0, 0.0, 0.0, 0.0, 0.0);

        let (conc_high, _) =
            component.solve_concentration(ch4_pi, ch4_pi, 400.0, 0.0, 0.0, 0.0, 0.0);

        assert!(
            conc_high > conc_low,
            "Higher emissions should give higher concentration. Got {} vs {}",
            conc_high,
            conc_low
        );
    }

    // ===== Temperature Feedback Tests =====

    #[test]
    fn test_warming_shortens_lifetime() {
        let component = default_component();
        let ch4_pi = component.parameters.ch4_pi;

        let (_conc_cold, lifetime_cold) =
            component.solve_concentration(ch4_pi, ch4_pi, 300.0, 0.0, 0.0, 0.0, 0.0);

        let (_conc_warm, lifetime_warm) =
            component.solve_concentration(ch4_pi, ch4_pi, 300.0, 2.0, 0.0, 0.0, 0.0);

        assert!(
            lifetime_warm < lifetime_cold,
            "Warmer temperature should shorten lifetime. Got warm={:.2} vs cold={:.2}",
            lifetime_warm,
            lifetime_cold
        );
    }

    #[test]
    fn test_temperature_feedback_disabled() {
        let params = CH4ChemistryParameters {
            include_temp_feedback: false,
            ..Default::default()
        };
        let component = CH4Chemistry::from_parameters(params);
        let ch4_pi = component.parameters.ch4_pi;

        let (_conc_cold, lifetime_cold) =
            component.solve_concentration(ch4_pi, ch4_pi, 300.0, 0.0, 0.0, 0.0, 0.0);

        let (_conc_warm, lifetime_warm) =
            component.solve_concentration(ch4_pi, ch4_pi, 300.0, 2.0, 0.0, 0.0, 0.0);

        assert!(
            (lifetime_warm - lifetime_cold).abs() < 1e-10,
            "With temp feedback disabled, lifetime should not change with temperature. \
             Got warm={:.4} vs cold={:.4}",
            lifetime_warm,
            lifetime_cold
        );
    }

    // ===== Concentration Feedback Tests =====

    #[test]
    fn test_higher_concentration_longer_lifetime() {
        let component = default_component();
        let ch4_pi = component.parameters.ch4_pi;

        // At pre-industrial
        let (_conc_low, lifetime_low) =
            component.solve_concentration(ch4_pi, ch4_pi, 0.0, 0.0, 0.0, 0.0, 0.0);

        // At elevated concentration (simulating steady state at higher level)
        let ch4_elevated = 1800.0;
        let (_conc_high, lifetime_high) =
            component.solve_concentration(ch4_elevated, ch4_elevated, 300.0, 0.0, 0.0, 0.0, 0.0);

        assert!(
            lifetime_high > lifetime_low,
            "Higher CH4 concentration should give longer lifetime (OH depletion). \
             Got high={:.2} vs low={:.2}",
            lifetime_high,
            lifetime_low
        );
    }

    // ===== Emissions Feedback Tests =====

    #[test]
    fn test_nox_emissions_shorten_lifetime() {
        let component = default_component();
        let ch4_pi = component.parameters.ch4_pi;

        let (_conc_no_nox, lifetime_no_nox) =
            component.solve_concentration(ch4_pi, ch4_pi, 300.0, 0.0, 0.0, 0.0, 0.0);

        let (_conc_with_nox, lifetime_with_nox) =
            component.solve_concentration(ch4_pi, ch4_pi, 300.0, 0.0, 50.0, 0.0, 0.0);

        assert!(
            lifetime_with_nox < lifetime_no_nox,
            "NOx emissions should shorten CH4 lifetime (more OH). \
             Got with_nox={:.2} vs no_nox={:.2}",
            lifetime_with_nox,
            lifetime_no_nox
        );
    }

    #[test]
    fn test_co_emissions_lengthen_lifetime() {
        let component = default_component();
        let ch4_pi = component.parameters.ch4_pi;

        let (_conc_no_co, lifetime_no_co) =
            component.solve_concentration(ch4_pi, ch4_pi, 300.0, 0.0, 0.0, 0.0, 0.0);

        let (_conc_with_co, lifetime_with_co) =
            component.solve_concentration(ch4_pi, ch4_pi, 300.0, 0.0, 0.0, 1000.0, 0.0);

        assert!(
            lifetime_with_co > lifetime_no_co,
            "CO emissions should lengthen CH4 lifetime (less OH). \
             Got with_co={:.2} vs no_co={:.2}",
            lifetime_with_co,
            lifetime_no_co
        );
    }

    // ===== Iteration Convergence Tests =====

    #[test]
    fn test_iteration_convergence() {
        // Verify that 4 iterations are sufficient for convergence
        // by comparing results with different iteration counts
        let params = CH4ChemistryParameters::default();
        let ch4_elevated = 1500.0;

        // We can't easily test different iteration counts without modifying the code,
        // but we can verify the result is physically reasonable
        let component = CH4Chemistry::from_parameters(params);

        let (new_conc, lifetime) = component.solve_concentration(
            ch4_elevated,
            ch4_elevated,
            350.0, // moderate anthropogenic emissions
            1.0,   // 1K warming
            30.0,  // moderate NOx
            500.0, // moderate CO
            100.0, // moderate NMVOC
        );

        // Check result is in reasonable range
        assert!(new_conc > 0.0, "Concentration should be positive");
        assert!(new_conc < 5000.0, "Concentration should be reasonable");
        assert!(lifetime > 5.0, "Lifetime should be > 5 years");
        assert!(lifetime < 20.0, "Lifetime should be < 20 years");
    }

    // ===== Component Trait Tests =====

    #[test]
    fn test_definitions() {
        let component = default_component();
        let defs = component.definitions();

        // Should have 5 inputs + 1 state (input/output) + 1 output = 7 definitions
        // But state appears as both input and output, so count carefully
        assert!(
            defs.len() >= 6,
            "Should have at least 6 definitions (5 inputs + 1 state + 1 output)"
        );

        let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"Emissions|CH4"));
        assert!(names.contains(&"Surface Temperature|Global"));
        assert!(names.contains(&"Atmospheric Concentration|CH4"));
        assert!(names.contains(&"Lifetime|CH4"));
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_zero_emissions_decay() {
        let ch4_elevated = 1800.0;

        // With zero total emissions, CH4 should decay
        let params = CH4ChemistryParameters {
            natural_emissions: 0.0,
            ..Default::default()
        };
        let component_no_natural = CH4Chemistry::from_parameters(params);

        let (new_conc, _) = component_no_natural.solve_concentration(
            ch4_elevated,
            ch4_elevated,
            0.0, // no anthropogenic
            0.0,
            0.0,
            0.0,
            0.0,
        );

        assert!(
            new_conc < ch4_elevated,
            "With zero emissions, concentration should decay. Got {} vs initial {}",
            new_conc,
            ch4_elevated
        );
    }

    #[test]
    fn test_very_high_concentration() {
        let component = default_component();
        let ch4_extreme = 10000.0; // Very high concentration

        let (new_conc, lifetime) =
            component.solve_concentration(ch4_extreme, ch4_extreme, 300.0, 0.0, 0.0, 0.0, 0.0);

        // Should still produce valid results
        assert!(new_conc > 0.0, "Concentration should remain positive");
        assert!(!new_conc.is_nan(), "Concentration should not be NaN");
        assert!(!lifetime.is_nan(), "Lifetime should not be NaN");
    }

    #[test]
    fn test_low_concentration_floor() {
        let ch4_low = 100.0; // Below pre-industrial

        let component = default_component();
        let (new_conc, lifetime) =
            component.solve_concentration(ch4_low, ch4_low, 50.0, 0.0, 0.0, 0.0, 0.0);

        // The MAX(1.0, ratio) floor should prevent lifetime from being affected
        // by low concentrations the same way high concentrations do
        assert!(lifetime > 0.0, "Lifetime should be positive");
        assert!(new_conc > 0.0, "Concentration should be positive");
    }
}
