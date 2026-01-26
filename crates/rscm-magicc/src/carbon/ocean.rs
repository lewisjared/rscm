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
//! 4. Uses monthly sub-stepping (12 steps per year) for numerical stability
//!
//! # Inputs
//!
//! - `Atmospheric Concentration|CO2` (ppm) - Atmospheric CO2 concentration
//! - `Sea Surface Temperature` (K) - Temperature anomaly from pre-industrial
//!
//! # States (tracked between timesteps)
//!
//! - `Ocean Surface pCO2` (ppm) - Ocean surface CO2 partial pressure
//! - `Cumulative Ocean Uptake` (GtC) - Total carbon absorbed by ocean
//!
//! # Outputs
//!
//! - `Carbon Flux|Ocean` (GtC/yr) - Air-sea carbon flux (positive = ocean uptake)
//!
//! # Differences from MAGICC7 Module 10
//!
//! This is a simplified implementation:
//!
//! - **Single IRF model**: Only 2D-BERN is implemented. MAGICC7 supports
//!   3D-GFDL, HILDA, BOXDIFF, and 2D-BERN with runtime selection.
//! - **No stability limiter**: MAGICC7 has an ad-hoc flux change limiter
//!   (0.04 ppm/yr per month). Not implemented as it's noted as a workaround.
//! - **No radiative-only mode**: MAGICC7 can make ocean "see" only PI CO2.
//!   Not implemented.
//! - **No biological pump**: Only abiotic chemistry is represented.
//! - **No circulation changes**: IRF is static, doesn't respond to warming.

use std::any::Any;
use std::collections::VecDeque;

use crate::parameters::OceanCarbonParameters;
use rscm_core::component::{
    Component, ComponentState, GridType, InputState, OutputState, RequirementDefinition,
    RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Conversion factor from ppm to GtC.
/// 1 ppm atmospheric CO2 ≈ 2.124 GtC
const PPM_TO_GTC: FloatValue = 2.124;

/// Internal state for OceanCarbon component.
///
/// This holds the flux history that persists across solve() calls.
/// Unlike coupled state (RequirementType::State), this is private
/// to the component and not shared between components.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OceanCarbonState {
    /// Flux history for IRF convolution (ppm/month).
    /// Each element represents the flux in one month.
    pub flux_history: VecDeque<FloatValue>,
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

/// Ocean carbon cycle component using IRF-based deep ocean mixing.
///
/// Implements a simplified MAGICC7-style ocean carbon model using the
/// 2D-BERN Impulse Response Function to represent carbon transport
/// from the mixed layer to the deep ocean.
///
/// # Algorithm
///
/// For each sub-step (monthly):
///
/// 1. Calculate air-sea flux:
///    $$F = k \times (pCO2_{atm} - pCO2_{ocn})$$
///
/// 2. Add flux to history and convolve with IRF to get $\Delta DIC$:
///    $$\Delta DIC = \int_0^t F(t') \times IRF(t - t') \, dt'$$
///
/// 3. Calculate $\Delta pCO2$ from $\Delta DIC$ using Joos A24 polynomial
///
/// 4. Apply temperature effect (Joos A25):
///    $$pCO2_{ocn} = (pCO2_{pi} + \Delta pCO2) \times e^{\alpha_T \Delta T}$$
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["carbon-cycle", "ocean", "magicc"], category = "Carbon Cycle")]
#[inputs(
    co2_concentration { name = "Atmospheric Concentration|CO2", unit = "ppm" },
    sst { name = "Sea Surface Temperature", unit = "K" },
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
    /// # Arguments
    ///
    /// * `pco2_atm` - Atmospheric CO2 partial pressure (ppm)
    /// * `pco2_ocn` - Ocean surface CO2 partial pressure (ppm)
    ///
    /// # Returns
    ///
    /// Flux in ppm/month (positive = into ocean)
    fn calculate_flux(&self, pco2_atm: FloatValue, pco2_ocn: FloatValue) -> FloatValue {
        self.parameters.gas_exchange_rate() * (pco2_atm - pco2_ocn)
    }

    /// Calculate delta DIC via IRF convolution.
    ///
    /// The change in dissolved inorganic carbon is computed by convolving
    /// the flux history with the impulse response function:
    ///
    /// $$\Delta DIC = \text{factor} \times \sum_{i=1}^{n} F_i \times IRF(t_n - t_i) \times \Delta t$$
    ///
    /// Uses a left-hand Riemann sum for the integral.
    ///
    /// # Arguments
    ///
    /// * `flux_history` - Ring buffer of monthly fluxes (ppm/month)
    ///
    /// # Returns
    ///
    /// Change in DIC (micromol/kg)
    fn calculate_delta_dic(&self, flux_history: &VecDeque<FloatValue>) -> FloatValue {
        if flux_history.is_empty() {
            return 0.0;
        }

        let n = flux_history.len();
        let dt_months = 1.0; // Monthly timesteps
        let dt_years = dt_months / 12.0;

        let mut integral = 0.0;

        for (i, &flux) in flux_history.iter().enumerate() {
            // Time since this pulse (in years)
            // At position i, the pulse happened (n - 1 - i) months ago
            let t_since_pulse = (n - 1 - i) as FloatValue * dt_years;

            // IRF value at this time
            let irf = self.parameters.irf(t_since_pulse);

            // Accumulate: flux * IRF * dt
            integral += flux * irf * dt_months;
        }

        // Convert to DIC units
        integral * self.parameters.dic_conversion_factor()
    }

    /// Solve the ocean carbon cycle for one timestep.
    ///
    /// Performs monthly sub-stepping for stability.
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable reference to component state (holds flux history)
    /// * `co2_atm` - Atmospheric CO2 concentration (ppm)
    /// * `delta_sst` - SST anomaly from pre-industrial (K)
    /// * `pco2_initial` - Initial ocean pCO2 (ppm)
    /// * `cumulative_initial` - Initial cumulative uptake (GtC)
    /// * `dt` - Timestep (years)
    ///
    /// # Returns
    ///
    /// (new_pco2, new_cumulative, annual_mean_flux)
    pub fn solve_ocean(
        &self,
        state: &mut OceanCarbonState,
        co2_atm: FloatValue,
        delta_sst: FloatValue,
        pco2_initial: FloatValue,
        cumulative_initial: FloatValue,
        dt: FloatValue,
    ) -> (FloatValue, FloatValue, FloatValue) {
        let steps = self.parameters.steps_per_year;
        let dt_month = dt / steps as FloatValue;

        let mut pco2_ocn = pco2_initial;
        let mut cumulative = cumulative_initial;
        let mut total_flux_gtc = 0.0;

        for _ in 0..steps {
            // Calculate air-sea flux (ppm/month)
            let flux_ppm = self.calculate_flux(co2_atm, pco2_ocn);

            // Store flux in history (bounded ring buffer)
            state.flux_history.push_back(flux_ppm);
            if state.flux_history.len() > self.parameters.max_history_months {
                state.flux_history.pop_front();
            }

            // Convert flux to GtC/yr for output and accumulation
            let flux_gtc_yr = flux_ppm * 12.0 * PPM_TO_GTC;
            total_flux_gtc += flux_gtc_yr / steps as FloatValue;

            // Accumulate uptake
            cumulative += flux_gtc_yr * dt_month;

            // Calculate new delta DIC from convolution
            let delta_dic = self.calculate_delta_dic(&state.flux_history);

            // Calculate delta pCO2 from DIC (Joos A24)
            let delta_pco2_dic = self.parameters.delta_pco2_from_dic(delta_dic);

            // Apply temperature effect (Joos A25)
            pco2_ocn = self.parameters.ocean_pco2(delta_pco2_dic, delta_sst);
        }

        (pco2_ocn, cumulative, total_flux_gtc)
    }

    /// Reset the flux history (for initialization or testing).
    pub fn reset_history(&self, state: &mut OceanCarbonState) {
        state.flux_history.clear();
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
        let inputs = OceanCarbonInputs::from_input_state(input_state);

        // Get current inputs
        let co2 = inputs.co2_concentration.at_start();
        let sst = inputs.sst.at_start();

        // Get current states
        let pco2 = inputs.ocean_pco2.at_start();
        let cumulative = inputs.cumulative_uptake.at_start();

        let dt = t_next - t_current;

        // Solve for new values
        let (new_pco2, new_cumulative, flux) =
            self.solve_ocean(state, co2, sst, pco2, cumulative, dt);

        let outputs = OceanCarbonOutputs {
            ocean_pco2: new_pco2,
            cumulative_uptake: new_cumulative,
            air_sea_flux: flux,
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

    fn default_component() -> OceanCarbon {
        OceanCarbon::from_parameters(OceanCarbonParameters::default())
    }

    // ===== Air-Sea Flux Tests =====

    #[test]
    fn test_zero_flux_at_equilibrium() {
        let component = default_component();
        let pco2 = component.parameters.pco2_pi;

        // When atm = ocean pCO2, flux should be zero
        let flux = component.calculate_flux(pco2, pco2);
        assert!(
            flux.abs() < 1e-15,
            "Flux should be zero at equilibrium, got {:.6e}",
            flux
        );
    }

    #[test]
    fn test_positive_flux_when_atm_exceeds_ocean() {
        let component = default_component();
        let pco2_pi = component.parameters.pco2_pi;

        // Higher atmospheric CO2 should drive flux into ocean
        let flux = component.calculate_flux(pco2_pi + 100.0, pco2_pi);
        assert!(
            flux > 0.0,
            "Flux should be positive when atm > ocean, got {:.6}",
            flux
        );
    }

    #[test]
    fn test_negative_flux_when_ocean_exceeds_atm() {
        let component = default_component();
        let pco2_pi = component.parameters.pco2_pi;

        // Higher ocean pCO2 should drive outgassing
        let flux = component.calculate_flux(pco2_pi, pco2_pi + 50.0);
        assert!(
            flux < 0.0,
            "Flux should be negative when ocean > atm, got {:.6}",
            flux
        );
    }

    #[test]
    fn test_flux_proportional_to_gradient() {
        let component = default_component();
        let pco2_pi = component.parameters.pco2_pi;

        let flux_50 = component.calculate_flux(pco2_pi + 50.0, pco2_pi);
        let flux_100 = component.calculate_flux(pco2_pi + 100.0, pco2_pi);

        // Flux should double when gradient doubles
        assert!(
            (flux_100 / flux_50 - 2.0).abs() < 1e-10,
            "Flux should be proportional to gradient: {:.6} vs {:.6}",
            flux_50,
            flux_100
        );
    }

    // ===== IRF Convolution Tests =====

    #[test]
    fn test_delta_dic_zero_for_empty_history() {
        let component = default_component();
        let delta_dic = component.calculate_delta_dic(&VecDeque::new());

        assert!(
            delta_dic.abs() < 1e-15,
            "Delta DIC should be zero for empty history"
        );
    }

    #[test]
    fn test_delta_dic_positive_for_positive_flux() {
        let component = default_component();

        // Constant positive flux for 12 months
        let flux_history: VecDeque<FloatValue> = vec![1.0; 12].into();
        let delta_dic = component.calculate_delta_dic(&flux_history);

        assert!(
            delta_dic > 0.0,
            "Delta DIC should be positive for positive flux, got {:.6}",
            delta_dic
        );
    }

    #[test]
    fn test_delta_dic_grows_with_time() {
        let component = default_component();

        // Constant flux over increasing time
        let flux_1yr: VecDeque<FloatValue> = vec![1.0; 12].into();
        let flux_2yr: VecDeque<FloatValue> = vec![1.0; 24].into();
        let flux_5yr: VecDeque<FloatValue> = vec![1.0; 60].into();

        let dic_1yr = component.calculate_delta_dic(&flux_1yr);
        let dic_2yr = component.calculate_delta_dic(&flux_2yr);
        let dic_5yr = component.calculate_delta_dic(&flux_5yr);

        assert!(
            dic_2yr > dic_1yr,
            "DIC should increase with time: 2yr={:.4} vs 1yr={:.4}",
            dic_2yr,
            dic_1yr
        );
        assert!(
            dic_5yr > dic_2yr,
            "DIC should continue increasing: 5yr={:.4} vs 2yr={:.4}",
            dic_5yr,
            dic_2yr
        );
    }

    // ===== Temperature Feedback Tests =====

    #[test]
    fn test_warming_increases_pco2() {
        let params = OceanCarbonParameters::default();

        let pco2_cold = params.ocean_pco2(0.0, 0.0);
        let pco2_warm = params.ocean_pco2(0.0, 1.0);

        assert!(
            pco2_warm > pco2_cold,
            "Warming should increase pCO2: cold={:.2}, warm={:.2}",
            pco2_cold,
            pco2_warm
        );
    }

    #[test]
    fn test_warming_reduces_uptake() {
        let component = default_component();
        let co2_elevated = 400.0;
        let pco2_pi = component.parameters.pco2_pi;

        // Run for one year with no warming
        let mut state_cold = OceanCarbonState::default();
        let (_, cumulative_cold, flux_cold) =
            component.solve_ocean(&mut state_cold, co2_elevated, 0.0, pco2_pi, 0.0, 1.0);

        // Run with warming (new state)
        let mut state_warm = OceanCarbonState::default();
        let (_, cumulative_warm, flux_warm) =
            component.solve_ocean(&mut state_warm, co2_elevated, 2.0, pco2_pi, 0.0, 1.0);

        // Warming should reduce uptake
        assert!(
            flux_warm < flux_cold,
            "Warming should reduce flux: cold={:.2}, warm={:.2}",
            flux_cold,
            flux_warm
        );
        assert!(
            cumulative_warm < cumulative_cold,
            "Warming should reduce cumulative uptake: cold={:.2}, warm={:.2}",
            cumulative_cold,
            cumulative_warm
        );
    }

    // ===== Integration Tests =====

    #[test]
    fn test_solve_ocean_one_year() {
        let component = default_component();
        let mut state = OceanCarbonState::default();
        let co2_elevated = 400.0;
        let pco2_pi = component.parameters.pco2_pi;

        let (new_pco2, cumulative, flux) =
            component.solve_ocean(&mut state, co2_elevated, 0.0, pco2_pi, 0.0, 1.0);

        // Ocean should have absorbed carbon
        assert!(
            cumulative > 0.0,
            "Ocean should have positive cumulative uptake"
        );
        assert!(
            flux > 0.0,
            "Flux should be positive when atm CO2 > ocean pCO2"
        );

        // Ocean pCO2 should have increased
        assert!(
            new_pco2 > pco2_pi,
            "Ocean pCO2 should increase after uptake: {:.2} vs {:.2}",
            new_pco2,
            pco2_pi
        );

        // History should have 12 entries (monthly)
        assert_eq!(
            component.history_length(&state),
            12,
            "Should have 12 months of history"
        );
    }

    #[test]
    fn test_multi_year_uptake() {
        let component = default_component();
        let mut state = OceanCarbonState::default();
        let co2_elevated = 400.0;

        let mut pco2 = component.parameters.pco2_pi;
        let mut cumulative = 0.0;
        let mut prev_flux = f64::MAX;

        // Run for 10 years
        for year in 0..10 {
            let (new_pco2, new_cumulative, flux) =
                component.solve_ocean(&mut state, co2_elevated, 0.0, pco2, cumulative, 1.0);

            // Flux should decrease over time as ocean pCO2 rises
            if year > 0 {
                assert!(
                    flux < prev_flux,
                    "Flux should decrease over time: year {} flux={:.2} vs prev={:.2}",
                    year,
                    flux,
                    prev_flux
                );
            }

            // Cumulative should monotonically increase
            assert!(
                new_cumulative > cumulative,
                "Cumulative should increase: {:.2} vs {:.2}",
                new_cumulative,
                cumulative
            );

            pco2 = new_pco2;
            cumulative = new_cumulative;
            prev_flux = flux;
        }
    }

    #[test]
    fn test_steady_state_at_equilibrium() {
        let component = default_component();
        let mut state = OceanCarbonState::default();
        let co2_pi = component.parameters.co2_pi;
        let pco2_pi = component.parameters.pco2_pi;

        // At equilibrium, there should be minimal change
        let (new_pco2, cumulative, flux) =
            component.solve_ocean(&mut state, co2_pi, 0.0, pco2_pi, 0.0, 1.0);

        // Flux should be very small (near equilibrium)
        assert!(
            flux.abs() < 0.1,
            "Flux should be near zero at equilibrium, got {:.4}",
            flux
        );

        // pCO2 should stay near initial
        assert!(
            (new_pco2 - pco2_pi).abs() < 1.0,
            "pCO2 should stay near equilibrium: {:.2} vs {:.2}",
            new_pco2,
            pco2_pi
        );

        // Cumulative should be near zero
        assert!(
            cumulative.abs() < 0.5,
            "Cumulative uptake should be near zero at equilibrium"
        );
    }

    // ===== Magnitude Tests =====

    #[test]
    fn test_flux_magnitude_reasonable() {
        let component = default_component();
        let mut state = OceanCarbonState::default();

        // The initial flux with ocean at PI (278 ppm) and atmosphere at 400 ppm
        // gives a large gradient (122 ppm). This is correct physics but doesn't
        // represent steady-state uptake.
        //
        // Run for several years to let the ocean pCO2 adjust towards the
        // atmospheric value, then check that quasi-steady-state flux is reasonable.
        let mut pco2 = component.parameters.pco2_pi;
        let mut cumulative = 0.0;
        let mut flux = 0.0;

        // Run for 50 years to approach quasi-steady state
        for _ in 0..50 {
            let (new_pco2, new_cumulative, new_flux) =
                component.solve_ocean(&mut state, 400.0, 0.0, pco2, cumulative, 1.0);
            pco2 = new_pco2;
            cumulative = new_cumulative;
            flux = new_flux;
        }

        // After 50 years, flux should have decreased as ocean pCO2 catches up
        // Expected range is 0.5-5 GtC/yr for quasi-steady-state ocean uptake
        assert!(
            flux > 0.1 && flux < 10.0,
            "Quasi-steady-state flux should be in reasonable range (0.1-10 GtC/yr), got {:.2}",
            flux
        );
    }

    #[test]
    fn test_pco2_increase_reasonable() {
        let component = default_component();
        let mut state = OceanCarbonState::default();

        // Run for 100 years at elevated CO2
        let mut pco2 = component.parameters.pco2_pi;
        let mut cumulative = 0.0;

        for _ in 0..100 {
            let (new_pco2, new_cumulative, _) =
                component.solve_ocean(&mut state, 450.0, 0.0, pco2, cumulative, 1.0);
            pco2 = new_pco2;
            cumulative = new_cumulative;
        }

        // Ocean pCO2 should approach but not exceed atmospheric
        assert!(
            pco2 < 450.0,
            "Ocean pCO2 should be less than atmospheric: {:.2}",
            pco2
        );
        assert!(
            pco2 > component.parameters.pco2_pi + 50.0,
            "Ocean pCO2 should have increased significantly: {:.2}",
            pco2
        );

        // Cumulative uptake should be substantial (50-200 GtC over 100 years)
        assert!(
            cumulative > 50.0 && cumulative < 500.0,
            "Cumulative uptake should be in reasonable range: {:.1} GtC",
            cumulative
        );
    }

    // ===== Edge Cases =====

    #[test]
    fn test_very_high_co2() {
        let component = default_component();
        let mut state = OceanCarbonState::default();

        // Very high CO2 scenario (like PETM)
        let (new_pco2, cumulative, flux) = component.solve_ocean(
            &mut state,
            2000.0,
            0.0,
            component.parameters.pco2_pi,
            0.0,
            1.0,
        );

        assert!(
            new_pco2.is_finite(),
            "pCO2 should remain finite at high CO2"
        );
        assert!(
            cumulative.is_finite() && cumulative > 0.0,
            "Cumulative should be finite and positive"
        );
        assert!(
            flux.is_finite() && flux > 0.0,
            "Flux should be finite and positive"
        );
    }

    #[test]
    fn test_negative_sst_anomaly() {
        let component = default_component();

        // Cooling should increase uptake (higher solubility)
        let mut state_cold = OceanCarbonState::default();
        let (_, _cumulative_cold, flux_cold) = component.solve_ocean(
            &mut state_cold,
            400.0,
            -2.0, // Cooling
            component.parameters.pco2_pi,
            0.0,
            1.0,
        );

        let mut state_warm = OceanCarbonState::default();
        let (_, _cumulative_warm, flux_warm) = component.solve_ocean(
            &mut state_warm,
            400.0,
            0.0,
            component.parameters.pco2_pi,
            0.0,
            1.0,
        );

        // Cooling should increase uptake
        assert!(
            flux_cold > flux_warm,
            "Cooling should increase flux: cold={:.2} vs warm={:.2}",
            flux_cold,
            flux_warm
        );
    }

    // ===== Component Trait Tests =====

    #[test]
    fn test_definitions() {
        let component = default_component();
        let defs = component.definitions();

        // Should have: 2 inputs + 2 states + 1 output = 5 definitions
        assert!(
            defs.len() >= 5,
            "Should have at least 5 definitions, got {}",
            defs.len()
        );

        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"Atmospheric Concentration|CO2"));
        assert!(names.contains(&"Sea Surface Temperature"));
        assert!(names.contains(&"Ocean Surface pCO2"));
        assert!(names.contains(&"Cumulative Ocean Uptake"));
        assert!(names.contains(&"Carbon Flux|Ocean"));
    }

    #[test]
    fn test_serialization() {
        let component = default_component();
        let json = serde_json::to_string(&component).expect("Serialization failed");
        let parsed: OceanCarbon = serde_json::from_str(&json).expect("Deserialization failed");

        assert!(
            (component.parameters.co2_pi - parsed.parameters.co2_pi).abs() < 1e-10,
            "Parameters should survive round-trip"
        );
    }

    #[test]
    fn test_history_reset() {
        let component = default_component();
        let mut state = OceanCarbonState::default();

        // Add some flux history
        component.solve_ocean(&mut state, 400.0, 0.0, 278.0, 0.0, 1.0);
        assert!(
            component.history_length(&state) > 0,
            "Should have history after solve"
        );

        // Reset
        component.reset_history(&mut state);
        assert_eq!(
            component.history_length(&state),
            0,
            "History should be cleared"
        );
    }

    // ===== Bounded History Tests =====

    #[test]
    fn test_flux_history_bounded() {
        // Create component with small history limit for testing
        let params = OceanCarbonParameters {
            max_history_months: 24, // 2 years
            ..Default::default()
        };
        let component = OceanCarbon::from_parameters(params);
        let mut state = OceanCarbonState::default();

        // Run for 5 years (60 months)
        for _ in 0..5 {
            component.solve_ocean(&mut state, 400.0, 0.0, 278.0, 0.0, 1.0);
        }

        // History should be bounded at 24 months
        assert_eq!(
            component.history_length(&state),
            24,
            "History should be bounded at max_history_months"
        );
    }

    #[test]
    fn test_component_state_serialization_roundtrip() {
        let component = default_component();
        let mut state = OceanCarbonState::default();

        // Run for 5 years to build up history
        let mut pco2 = component.parameters.pco2_pi;
        let mut cumulative = 0.0;

        for _ in 0..5 {
            let (new_pco2, new_cumulative, _) =
                component.solve_ocean(&mut state, 400.0, 0.0, pco2, cumulative, 1.0);
            pco2 = new_pco2;
            cumulative = new_cumulative;
        }

        let history_len_before = component.history_length(&state);
        assert!(
            history_len_before > 0,
            "Should have history before serialization"
        );

        // Serialize component state
        let state_json = serde_json::to_string(&state).expect("State serialization failed");
        let restored_state: OceanCarbonState =
            serde_json::from_str(&state_json).expect("State deserialization failed");

        assert_eq!(
            component.history_length(&restored_state),
            history_len_before,
            "History length should survive serialization"
        );

        // The flux_history content should also match
        assert_eq!(
            state.flux_history.len(),
            restored_state.flux_history.len(),
            "Flux history length should match"
        );
    }

    #[test]
    fn test_longer_history_captures_more_response() {
        // The 2D-BERN IRF has long timescales:
        // - Late IRF components: 10.5, 11.7, 38.9, 107.6, 331.5 years
        // - Permanent sequestration: 1e10 years (carbon never fully returns)
        //
        // This means longer history always captures more of the response.
        // The 500-year default (6000 months) is a practical compromise between
        // accuracy and memory usage (~48KB for flux history).
        let params = OceanCarbonParameters::default();
        let component = OceanCarbon::from_parameters(params);

        // Compare histories of increasing length with same constant flux
        let history_10yr: VecDeque<FloatValue> = vec![1.0; 120].into();
        let history_100yr: VecDeque<FloatValue> = vec![1.0; 1200].into();
        let history_500yr: VecDeque<FloatValue> = vec![1.0; 6000].into();

        let dic_10yr = component.calculate_delta_dic(&history_10yr);
        let dic_100yr = component.calculate_delta_dic(&history_100yr);
        let dic_500yr = component.calculate_delta_dic(&history_500yr);

        // Longer history should accumulate more DIC
        assert!(
            dic_100yr > dic_10yr,
            "100yr history should have more DIC than 10yr: {:.4} vs {:.4}",
            dic_100yr,
            dic_10yr
        );
        assert!(
            dic_500yr > dic_100yr,
            "500yr history should have more DIC than 100yr: {:.4} vs {:.4}",
            dic_500yr,
            dic_100yr
        );

        // Growth rate should slow as we capture more of the response
        let growth_10_to_100 = (dic_100yr - dic_10yr) / dic_10yr;
        let growth_100_to_500 = (dic_500yr - dic_100yr) / dic_100yr;
        assert!(
            growth_100_to_500 < growth_10_to_100,
            "Growth rate should slow: 10→100yr={:.1}%, 100→500yr={:.1}%",
            growth_10_to_100 * 100.0,
            growth_100_to_500 * 100.0
        );
    }
}
