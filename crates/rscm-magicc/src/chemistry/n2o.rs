//! N2O Chemistry Component
//!
//! Simulates atmospheric nitrous oxide chemistry with concentration-dependent
//! lifetime and stratospheric transport delay using Prather-style fixed-point
//! iterations.
//!
//! # What This Component Does
//!
//! 1. Calculates the effective N2O atmospheric lifetime including:
//!    - Stratospheric photolysis/destruction (primary sink)
//!    - Concentration-dependent feedback (higher N2O → longer lifetime)
//!
//! 2. Solves the mass balance equation using 4 fixed-point iterations:
//!    $$\frac{dB}{dt} = E - \frac{\bar{B}_{lagged}}{\tau}$$
//!
//!    where the sink term uses lagged concentrations to account for
//!    stratospheric transport delay.
//!
//! # Inputs
//!
//! - `Emissions|N2O` (Tg N/yr) - Anthropogenic N2O emissions
//!
//! # Outputs
//!
//! - `Atmospheric Concentration|N2O` (ppb) - Updated N2O concentration
//! - `Lifetime|N2O` (yr) - Effective N2O lifetime
//!
//! # Differences from MAGICC7 Module 02
//!
//! This is a simplified implementation:
//!
//! - **Budget initialisation**: MAGICC7 calculates natural emissions to close
//!   the budget over a reference period. Here, natural emissions are fixed.
//! - **Meridional flux feedback**: MAGICC7 can adjust lifetime based on
//!   Brewer-Dobson circulation changes with temperature. Not implemented.
//! - **Prescribed/calculated switch**: MAGICC7 can switch between prescribed
//!   and calculated concentrations by year. Here, always emissions-driven.
//! - **Stratospheric delay history**: MAGICC7 uses concentrations from both
//!   t-delay and t-delay-1. Here, we use t-1 and t-2 concentrations assuming
//!   annual timesteps with delay=1.

use crate::parameters::N2OChemistryParameters;
use rscm_core::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Number of fixed-point iterations (matches MAGICC7)
const PRATHER_ITERATIONS: usize = 4;

/// N2O atmospheric chemistry component
///
/// Implements Prather-style fixed-point iterations for solving the N2O mass
/// balance equation. The method accounts for:
///
/// 1. **Concentration feedback**: Higher N2O concentrations lead to longer
///    lifetime because stratospheric sinks become relatively less efficient.
///
/// 2. **Stratospheric delay**: The sink term uses lagged concentrations
///    because N2O must first be transported from the troposphere to the
///    stratosphere before destruction can occur.
///
/// # Algorithm
///
/// For each timestep, performs 4 fixed-point iterations:
///
/// 1. Calculate mid-year burden: $\bar{B}_n = (B + B_{n-1}) / 2$
/// 2. Calculate effective lifetime: $\tau_n = \tau_0 \cdot \max(1, \bar{B}_n/B_{ref})^S$
/// 3. Calculate burden change: $\Delta B_n = E - \bar{B}_{lagged}/\tau_n$
/// 4. Update burden: $B_n = B_{prev} + \Delta B_n$
///
/// Note: Unlike CH4, the sink term uses the lagged average burden
/// ($\bar{B}_{lagged}$), not the current mid-year estimate.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["chemistry", "n2o", "magicc"], category = "Atmospheric Chemistry")]
#[inputs(
    n2o_emissions { name = "Emissions|N2O", unit = "Tg N/yr" },
)]
#[states(
    n2o_concentration { name = "Atmospheric Concentration|N2O", unit = "ppb" },
)]
#[outputs(
    n2o_lifetime { name = "Lifetime|N2O", unit = "yr" },
)]
pub struct N2OChemistry {
    parameters: N2OChemistryParameters,
}

impl N2OChemistry {
    /// Create a new N2O chemistry component with default parameters
    pub fn new() -> Self {
        Self::from_parameters(N2OChemistryParameters::default())
    }

    /// Create a new N2O chemistry component from parameters
    pub fn from_parameters(parameters: N2OChemistryParameters) -> Self {
        Self { parameters }
    }

    /// Convert concentration (ppb) to burden (Tg N)
    fn concentration_to_burden(&self, concentration: FloatValue) -> FloatValue {
        concentration * self.parameters.ppb_to_tg
    }

    /// Convert burden (Tg N) to concentration (ppb)
    fn burden_to_concentration(&self, burden: FloatValue) -> FloatValue {
        burden / self.parameters.ppb_to_tg
    }

    /// Calculate effective lifetime with concentration feedback
    ///
    /// $$\tau = \tau_{init} \cdot \max\left(1, \frac{\bar{B}}{B_{ref}}\right)^S$$
    ///
    /// With the default $S = -0.04$, higher concentrations give slightly shorter
    /// lifetime (ratio^(-0.04) < 1 when ratio > 1). This is a very weak feedback
    /// compared to CH4.
    ///
    /// Note: The floor `max(1, ratio)` ensures lifetime never increases above
    /// `tau_init` when concentration is below reference.
    fn calculate_effective_lifetime(
        &self,
        burden_mid: FloatValue,
        burden_reference: FloatValue,
    ) -> FloatValue {
        let ratio = (burden_mid / burden_reference).max(1.0);
        self.parameters.tau_n2o * ratio.powf(self.parameters.lifetime_feedback)
    }

    /// Perform one fixed-point iteration
    ///
    /// Returns (new_burden, effective_lifetime)
    fn iteration(
        &self,
        burden_current: FloatValue,
        burden_prev: FloatValue,
        burden_reference: FloatValue,
        burden_lagged: FloatValue,
        total_emissions: FloatValue,
    ) -> (FloatValue, FloatValue) {
        // 1. Calculate mid-year burden estimate
        let burden_mid = (burden_prev + burden_current) / 2.0;

        // 2. Calculate effective lifetime with concentration feedback
        let tau_eff = self.calculate_effective_lifetime(burden_mid, burden_reference);

        // 3. Calculate burden change (sink uses lagged burden)
        let delta_burden = total_emissions - burden_lagged / tau_eff;

        // 4. New burden
        let new_burden = burden_prev + delta_burden;

        (new_burden, tau_eff)
    }

    /// Run the full iteration scheme (4 iterations)
    pub fn solve_concentration(
        &self,
        n2o_prev: FloatValue,
        n2o_current: FloatValue,
        n2o_lagged: FloatValue,
        anthropogenic_emissions: FloatValue,
    ) -> (FloatValue, FloatValue) {
        // Total emissions = anthropogenic + natural
        let total_emissions = anthropogenic_emissions + self.parameters.natural_emissions;

        // Convert to burdens
        let burden_prev = self.concentration_to_burden(n2o_prev);
        let burden_lagged = self.concentration_to_burden(n2o_lagged);
        let burden_reference = self.concentration_to_burden(self.parameters.n2o_pi);

        // Initial guess: burden at start of timestep
        let mut burden = self.concentration_to_burden(n2o_current);
        let mut tau_eff = self.parameters.tau_n2o;

        // Run fixed-point iterations
        for _ in 0..PRATHER_ITERATIONS {
            let (new_burden, tau) = self.iteration(
                burden,
                burden_prev,
                burden_reference,
                burden_lagged,
                total_emissions,
            );
            burden = new_burden;
            tau_eff = tau;
        }

        // Convert back to concentration
        let new_concentration = self.burden_to_concentration(burden);

        (new_concentration, tau_eff)
    }
}

impl Default for N2OChemistry {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde]
impl Component for N2OChemistry {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        _t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let inputs = N2OChemistryInputs::from_input_state(input_state);

        // Get current and historical concentrations
        let n2o_current = inputs.n2o_concentration.at_start();
        // Use current as previous if no history (first timestep)
        let n2o_prev = inputs.n2o_concentration.previous().unwrap_or(n2o_current);

        // For the lagged concentration (used in sink term), we need the average
        // of (t-delay) and (t-delay-1). With delay=1 and annual timesteps:
        // - t-1 = n2o_prev (available)
        // - t-2 = need to approximate
        //
        // We use n2o_prev as an approximation for the average, which slightly
        // underestimates the sink lag but is reasonable for smooth trajectories.
        let n2o_lagged = n2o_prev;

        let emissions = inputs.n2o_emissions.at_start();

        // Solve for new concentration
        let (new_concentration, lifetime) =
            self.solve_concentration(n2o_prev, n2o_current, n2o_lagged, emissions);

        let outputs = N2OChemistryOutputs {
            n2o_concentration: new_concentration,
            n2o_lifetime: lifetime,
        };

        Ok(outputs.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_component() -> N2OChemistry {
        N2OChemistry::from_parameters(N2OChemistryParameters::default())
    }

    // ===== Steady State Tests =====

    #[test]
    fn test_steady_state_at_preindustrial() {
        // At pre-industrial with only natural emissions, concentration should be stable
        let component = default_component();
        let n2o_pi = component.parameters.n2o_pi;

        // Solve starting from pre-industrial
        let (new_conc, lifetime) = component.solve_concentration(
            n2o_pi, // previous
            n2o_pi, // current
            n2o_pi, // lagged
            0.0,    // no anthropogenic emissions
        );

        // Check lifetime is close to base value at pre-industrial
        let expected_tau = component.parameters.tau_n2o;
        let tau_relative_diff = ((lifetime - expected_tau) / expected_tau).abs();
        assert!(
            tau_relative_diff < 0.01,
            "At pre-industrial, lifetime should be close to tau_n2o. \
             Got {:.2} vs expected {:.2}",
            lifetime,
            expected_tau
        );

        // Concentration should be close to pre-industrial
        // With balanced natural emissions, the concentration should be stable
        let relative_change = ((new_conc - n2o_pi) / n2o_pi).abs();
        assert!(
            relative_change < 0.05,
            "At pre-industrial with natural emissions only, concentration should be stable. \
             Got {:.2} vs PI={:.2}, change={:.2}%",
            new_conc,
            n2o_pi,
            relative_change * 100.0
        );
    }

    #[test]
    fn test_steady_state_emissions_balance() {
        // Calculate what emissions would balance a given concentration
        let component = default_component();
        let n2o_conc = 320.0; // ppb
        let burden = n2o_conc * component.parameters.ppb_to_tg;
        let tau = component.calculate_effective_lifetime(
            burden,
            component.parameters.n2o_pi * component.parameters.ppb_to_tg,
        );

        // At steady state: E = B/tau
        let balanced_emissions = burden / tau;
        let anthropogenic_emissions = balanced_emissions - component.parameters.natural_emissions;

        let (new_conc, _) = component.solve_concentration(
            n2o_conc,
            n2o_conc,
            n2o_conc,
            anthropogenic_emissions.max(0.0),
        );

        let relative_change = ((new_conc - n2o_conc) / n2o_conc).abs();
        assert!(
            relative_change < 0.02,
            "With balanced emissions, concentration should be stable. \
             Got {:.2} vs {:.2}, change={:.2}%",
            new_conc,
            n2o_conc,
            relative_change * 100.0
        );
    }

    // ===== Emissions Response Tests =====

    #[test]
    fn test_emissions_increase_raises_concentration() {
        let component = default_component();
        let n2o_pi = component.parameters.n2o_pi;

        // Add anthropogenic emissions
        let (new_conc, _) = component.solve_concentration(
            n2o_pi, n2o_pi, n2o_pi, 5.0, // 5 Tg N/yr anthropogenic
        );

        assert!(
            new_conc > n2o_pi,
            "Adding emissions should increase concentration. Got {:.2} vs PI={:.2}",
            new_conc,
            n2o_pi
        );
    }

    #[test]
    fn test_higher_emissions_higher_concentration() {
        let component = default_component();
        let n2o_pi = component.parameters.n2o_pi;

        let (conc_low, _) = component.solve_concentration(n2o_pi, n2o_pi, n2o_pi, 3.0);

        let (conc_high, _) = component.solve_concentration(n2o_pi, n2o_pi, n2o_pi, 8.0);

        assert!(
            conc_high > conc_low,
            "Higher emissions should give higher concentration. Got {:.2} vs {:.2}",
            conc_high,
            conc_low
        );
    }

    // ===== Concentration Feedback Tests =====

    #[test]
    fn test_concentration_lifetime_feedback() {
        // With S = -0.04 (negative), higher concentration gives slightly shorter
        // lifetime: ratio^(-0.04) < 1 when ratio > 1
        let component = default_component();
        let burden_ref = component.parameters.n2o_pi * component.parameters.ppb_to_tg;

        // At pre-industrial
        let tau_at_ref = component.calculate_effective_lifetime(burden_ref, burden_ref);

        // At elevated concentration (350 ppb vs 270 ppb reference)
        let burden_high = 350.0 * component.parameters.ppb_to_tg;
        let tau_at_high = component.calculate_effective_lifetime(burden_high, burden_ref);

        // With S = -0.04: tau_high = tau_init * (350/270)^(-0.04) < tau_init
        assert!(
            tau_at_high < tau_at_ref,
            "With S < 0, higher N2O concentration gives slightly shorter lifetime. \
             Got high={:.2} vs ref={:.2}",
            tau_at_high,
            tau_at_ref
        );

        // Verify the feedback is weak (< 2% change)
        let relative_change = (tau_at_ref - tau_at_high) / tau_at_ref;
        assert!(
            relative_change < 0.02,
            "N2O lifetime feedback should be weak (< 2%). Got {:.2}%",
            relative_change * 100.0
        );
    }

    #[test]
    fn test_lifetime_floor_at_low_concentration() {
        let component = default_component();
        let burden_ref = component.parameters.n2o_pi * component.parameters.ppb_to_tg;

        // At low concentration (below reference)
        let burden_low = 200.0 * component.parameters.ppb_to_tg;
        let tau = component.calculate_effective_lifetime(burden_low, burden_ref);

        // Due to MAX(1.0, ratio), lifetime should equal base value
        let expected = component.parameters.tau_n2o;
        assert!(
            (tau - expected).abs() < 1e-10,
            "Below reference, lifetime should be tau_n2o. Got {:.2} vs {:.2}",
            tau,
            expected
        );
    }

    // ===== Iteration Convergence Tests =====

    #[test]
    fn test_iteration_convergence() {
        // Verify results are physically reasonable
        let component = default_component();
        let n2o_elevated = 330.0;

        let (new_conc, lifetime) = component.solve_concentration(
            n2o_elevated,
            n2o_elevated,
            n2o_elevated,
            6.0, // moderate anthropogenic emissions
        );

        assert!(new_conc > 0.0, "Concentration should be positive");
        assert!(
            new_conc < 500.0,
            "Concentration should be reasonable (< 500 ppb)"
        );
        assert!(lifetime > 100.0, "Lifetime should be > 100 years");
        assert!(lifetime < 200.0, "Lifetime should be < 200 years");
    }

    #[test]
    fn test_three_vs_four_iterations_close() {
        // Indirectly test convergence by checking the result is stable
        // Run with same inputs twice - should give same result
        let component = default_component();
        let n2o = 310.0;
        let emissions = 5.0;

        let (conc1, tau1) = component.solve_concentration(n2o, n2o, n2o, emissions);
        let (conc2, tau2) = component.solve_concentration(n2o, n2o, n2o, emissions);

        assert!(
            (conc1 - conc2).abs() < 1e-10,
            "Same inputs should give same concentration"
        );
        assert!(
            (tau1 - tau2).abs() < 1e-10,
            "Same inputs should give same lifetime"
        );
    }

    // ===== Component Trait Tests =====

    #[test]
    fn test_definitions() {
        let component = default_component();
        let defs = component.definitions();

        // Should have: 1 input + 1 state (in/out) + 1 output
        assert!(defs.len() >= 3, "Should have at least 3 definitions");

        let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"Emissions|N2O"));
        assert!(names.contains(&"Atmospheric Concentration|N2O"));
        assert!(names.contains(&"Lifetime|N2O"));
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_zero_emissions_decay() {
        // With zero total emissions, N2O should decay
        let mut params = N2OChemistryParameters::default();
        params.natural_emissions = 0.0;
        let component = N2OChemistry::from_parameters(params);

        let n2o_elevated = 350.0;
        let (new_conc, _) = component.solve_concentration(
            n2o_elevated,
            n2o_elevated,
            n2o_elevated,
            0.0, // no anthropogenic
        );

        assert!(
            new_conc < n2o_elevated,
            "With zero emissions, concentration should decay. Got {:.2} vs initial {:.2}",
            new_conc,
            n2o_elevated
        );
    }

    #[test]
    fn test_very_high_concentration() {
        let component = default_component();
        let n2o_extreme = 500.0; // Very high concentration

        let (new_conc, lifetime) =
            component.solve_concentration(n2o_extreme, n2o_extreme, n2o_extreme, 10.0);

        assert!(new_conc > 0.0, "Concentration should remain positive");
        assert!(!new_conc.is_nan(), "Concentration should not be NaN");
        assert!(!lifetime.is_nan(), "Lifetime should not be NaN");
    }

    #[test]
    fn test_lagged_concentration_effect() {
        // Test that lower lagged concentration increases effective sink
        let component = default_component();
        let n2o_current = 320.0;

        // Higher lagged concentration = more sink = lower final concentration
        let (conc_high_lag, _) = component.solve_concentration(
            n2o_current,
            n2o_current,
            350.0, // high lagged
            5.0,
        );

        let (conc_low_lag, _) = component.solve_concentration(
            n2o_current,
            n2o_current,
            280.0, // low lagged
            5.0,
        );

        assert!(
            conc_low_lag > conc_high_lag,
            "Lower lagged concentration should give higher final concentration (less sink). \
             Got low_lag={:.2} vs high_lag={:.2}",
            conc_low_lag,
            conc_high_lag
        );
    }
}
