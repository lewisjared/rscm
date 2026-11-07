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
//!    - Concentration-dependent feedback (with default parameters, higher N2O → shorter lifetime)
//!    - Faster stratospheric overturning with warming (meridional flux feedback)
//!
//! 2. Solves the mass balance equation using 4 fixed-point iterations:
//!    $$\frac{dB}{dt} = E - \frac{\bar{B}_{lagged}}{\tau}$$
//!
//!    where the sink term uses lagged concentrations to account for
//!    stratospheric transport delay.
//!
//! # Inputs
//!
//! - `Emissions|N2O` (Mt N/yr) - Anthropogenic N2O emissions
//! - `Surface Temperature` (K) - Global mean surface temperature anomaly
//!
//! # Outputs
//!
//! - `Atmospheric Concentration|N2O` (ppb) - Updated N2O concentration
//! - `Lifetime|N2O` (yr) - Effective N2O lifetime
//!
//! # Feedback references
//!
//! As in MAGICC7, the concentration feedback is measured from the year before
//! `feedback_start_year`, and warming changes the meridional flux relative to
//! `meridional_flux_reference_year`. References outside the run are clamped to
//! its first or last year.
//!
//! # Differences from MAGICC7
//!
//! - **Budget initialisation**: natural emissions are a fixed parameter.
//!   `rscm.magicc.full.natural_emissions` calculates MAGICC7's budget-closing value for it.
//!
//! With `concentration_switch_year` the concentration follows
//! `Atmospheric Concentration|N2O|Prescribed` until that year.

use crate::parameters::N2OChemistryParameters;
use crate::reference::{
    feedback_warming, prescribed_concentration, prescribed_definition, reference_value,
    require_annual, value_at_year,
};
use rscm_core::component::{
    Component, GridType, InputState, OutputState, VariableDefinition, VariableRole,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Number of fixed-point iterations (fixed per MAGICC7 implementation)
const PRATHER_ITERATIONS: usize = 4;
const N2O_CONCENTRATION: &str = "Atmospheric Concentration|N2O";

/// N2O atmospheric chemistry component
///
/// Implements the MAGICC7 `NITROUS` scheme. Starting from the burden $B$ at the
/// start of the year, each iteration estimates the mid-year burden, evaluates
/// the lifetime from it and integrates the sink using the burden averaged over
/// the delayed years.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["chemistry", "n2o", "magicc"], category = "Atmospheric Chemistry")]
#[inputs(
    n2o_emissions { name = "Emissions|N2O", unit = "Mt N/yr" },
    temperature { name = "Surface Temperature", unit = "K" },
)]
#[states(
    n2o_concentration { name = "Atmospheric Concentration|N2O", unit = "ppb" },
)]
#[outputs(
    n2o_lifetime { name = "Lifetime|N2O", unit = "yr", timeseries_type = "average" },
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

    /// Lifetime scaling from faster stratospheric overturning with warming
    fn meridional_flux_scale(&self, temperature_change: FloatValue) -> FloatValue {
        if !self.parameters.include_meridional_flux_feedback {
            return 1.0;
        }
        let p = &self.parameters;
        1.0 / (1.0
            + temperature_change.max(0.0)
                * p.meridional_flux_change_per_degree
                * p.tau_sensitivity_to_meridional_flux)
    }

    /// Solve one step with 4 fixed-point iterations
    ///
    /// `n2o_lagged` is the mean concentration over the delayed years and
    /// `temperature_change` is the warming since the meridional flux reference.
    /// Returns the concentration at the end of the step and the lifetime.
    pub fn solve_concentration(
        &self,
        n2o_start: FloatValue,
        n2o_reference: FloatValue,
        n2o_lagged: FloatValue,
        anthropogenic_emissions: FloatValue,
        temperature_change: FloatValue,
        dt: FloatValue,
    ) -> (FloatValue, FloatValue) {
        let total_emissions = anthropogenic_emissions + self.parameters.natural_emissions;
        let burden = self.concentration_to_burden(n2o_start);
        let burden_reference = self.concentration_to_burden(n2o_reference);
        let burden_lagged = self.concentration_to_burden(n2o_lagged);
        let tau_base = self.meridional_flux_scale(temperature_change) * self.parameters.tau_n2o;

        let mut burden_next = burden;
        let mut tau = tau_base;
        for iteration in 0..PRATHER_ITERATIONS {
            let burden_mean = if iteration == 0 {
                burden
            } else {
                (burden + burden_next) / 2.0
            };
            tau = tau_base
                * (burden_mean / burden_reference)
                    .max(1.0)
                    .powf(self.parameters.lifetime_feedback);
            burden_next = burden + (total_emissions - burden_lagged / tau) * dt;
        }

        (self.burden_to_concentration(burden_next), tau)
    }
}

impl Default for N2OChemistry {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde]
impl Component for N2OChemistry {
    fn definitions(&self) -> Vec<VariableDefinition> {
        let mut definitions = Self::generated_definitions();
        definitions.extend(prescribed_definition(
            N2O_CONCENTRATION,
            "ppb",
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
        require_annual("N2OChemistry", t_current, t_next)?;
        let inputs = N2OChemistryInputs::from_input_state(input_state);
        let p = &self.parameters;
        let concentration = &inputs.n2o_concentration;

        let n2o_start = concentration.get();
        let n2o_reference = reference_value(concentration, t_current, p.feedback_start_year);

        // MAGICC7 clamps delayed indices to the first year of the run
        let delay = p.strat_delay as Time;
        let delayed = |lag: Time| value_at_year(concentration, t_current, t_current - lag);
        let n2o_lagged = (delayed(delay) + delayed(delay + 1.0)) / 2.0;

        let temperature_change = feedback_warming(
            &inputs.temperature,
            t_current,
            p.meridional_flux_reference_year,
        );

        let (new_concentration, lifetime) = self.solve_concentration(
            n2o_start,
            n2o_reference,
            n2o_lagged,
            inputs.n2o_emissions.get(),
            temperature_change,
            t_next - t_current,
        );

        let new_concentration = prescribed_concentration(
            input_state,
            N2O_CONCENTRATION,
            p.concentration_switch_year,
            t_next,
        )
        .unwrap_or(new_concentration);

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

    /// Concentration at which natural emissions balance the base lifetime
    fn natural_steady_state(component: &N2OChemistry) -> FloatValue {
        let p = &component.parameters;
        component.burden_to_concentration(p.natural_emissions * p.tau_n2o)
    }

    #[test]
    fn test_steady_state_with_natural_emissions() {
        let component = default_component();
        let c = natural_steady_state(&component);
        let (new_conc, lifetime) = component.solve_concentration(c, c, c, 0.0, 0.0, 1.0);
        assert!((new_conc - c).abs() < 1e-9, "Expected {c}, got {new_conc}");
        assert!((lifetime - component.parameters.tau_n2o).abs() < 1e-9);
    }

    #[test]
    fn test_matches_magicc_first_year() {
        // MAGICC7 v7.5.3 SSP245, emissions-driven from 1750: 273.865 -> 273.587 ppb
        let component = default_component();
        let (new_conc, lifetime) =
            component.solve_concentration(273.865, 273.865, 273.865, 0.05474145, 0.0, 1.0);
        assert!((new_conc - 273.587).abs() < 1e-3, "Got {new_conc}");
        assert!((lifetime - 139.275).abs() < 1e-9);
    }

    #[test]
    fn test_emissions_increase_raises_concentration() {
        let component = default_component();
        let c = natural_steady_state(&component);
        let (new_conc, _) = component.solve_concentration(c, c, c, 5.0, 0.0, 1.0);
        assert!(new_conc > c);
    }

    #[test]
    fn test_concentration_lifetime_feedback() {
        let component = default_component();
        let (_, tau_ref) = component.solve_concentration(270.0, 270.0, 270.0, 3.0, 0.0, 1.0);
        let (_, tau_high) = component.solve_concentration(330.0, 270.0, 330.0, 3.0, 0.0, 1.0);
        assert!(tau_high < tau_ref, "Lifetime should shorten with burden");
    }

    #[test]
    fn test_lifetime_floor_at_low_concentration() {
        let component = default_component();
        let (_, tau) = component.solve_concentration(200.0, 270.0, 200.0, 0.0, 0.0, 1.0);
        assert!((tau - component.parameters.tau_n2o).abs() < 1e-9);
    }

    #[test]
    fn test_warming_shortens_lifetime() {
        let component = default_component();
        let (_, tau_cold) = component.solve_concentration(300.0, 270.0, 300.0, 5.0, 0.0, 1.0);
        let (_, tau_warm) = component.solve_concentration(300.0, 270.0, 300.0, 5.0, 2.0, 1.0);
        let expected = tau_cold / (1.0 + 2.0 * 0.15 * 0.04);
        assert!((tau_warm / expected - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_meridional_feedback_disabled() {
        let component = N2OChemistry::from_parameters(N2OChemistryParameters {
            include_meridional_flux_feedback: false,
            ..Default::default()
        });
        let (_, tau_cold) = component.solve_concentration(300.0, 270.0, 300.0, 5.0, 0.0, 1.0);
        let (_, tau_warm) = component.solve_concentration(300.0, 270.0, 300.0, 5.0, 2.0, 1.0);
        assert!((tau_warm - tau_cold).abs() < 1e-12);
    }

    #[test]
    fn test_lagged_concentration_sets_sink() {
        let component = default_component();
        let (conc_low_lag, _) = component.solve_concentration(300.0, 270.0, 290.0, 5.0, 0.0, 1.0);
        let (conc_high_lag, _) = component.solve_concentration(300.0, 270.0, 310.0, 5.0, 0.0, 1.0);
        assert!(conc_low_lag > conc_high_lag);
    }

    #[test]
    fn test_zero_emissions_decay() {
        let component = N2OChemistry::from_parameters(N2OChemistryParameters {
            natural_emissions: 0.0,
            ..Default::default()
        });
        let (new_conc, _) = component.solve_concentration(320.0, 270.0, 320.0, 0.0, 0.0, 1.0);
        assert!(new_conc < 320.0);
    }

    #[test]
    fn test_definitions() {
        let component = default_component();
        let names: Vec<_> = component
            .definitions()
            .iter()
            .map(|d| d.name.clone())
            .collect();
        assert!(names.contains(&"Emissions|N2O".to_string()));
        assert!(names.contains(&"Surface Temperature".to_string()));
        assert!(names.contains(&"Atmospheric Concentration|N2O".to_string()));
        assert!(names.contains(&"Lifetime|N2O".to_string()));
    }
}
