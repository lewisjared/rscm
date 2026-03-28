//! Terrestrial Carbon Cycle Component
//!
//! Implements the MAGICC7 `TERRCARBON2` subroutine: a 3-pool terrestrial carbon
//! model with three CO2 fertilization methods, two respiration methods,
//! time-varying turnover times, no-feedback reference pools for regrowth
//! attribution, and mass conservation correction.
//!
//! # Pools
//!
//! - Plant biomass (living vegetation)
//! - Detritus/litter (dead organic matter)
//! - Soil carbon (long-lived organic carbon)
//!
//! # Feedbacks
//!
//! 1. **CO2 Fertilization**: Three blendable methods (logarithmic, Gifford, sigmoid)
//! 2. **Temperature on NPP**: Exponential sensitivity
//! 3. **Temperature on decay**: Separate sensitivities for detritus and soil
//! 4. **Time-varying turnover**: Cumulative deforestation reduces turnover times
//!
//! # Inputs
//!
//! - `Atmospheric Concentration|CO2` (ppm)
//! - `Surface Temperature` (K) - temperature anomaly
//! - `Emissions|CO2|Land Use` (GtC/yr)
//!
//! # States (tracked between timesteps)
//!
//! - `Carbon Pool|Plant` (GtC)
//! - `Carbon Pool|Detritus` (GtC)
//! - `Carbon Pool|Soil` (GtC)
//!
//! # Outputs
//!
//! - `Carbon Flux|Terrestrial` (GtC/yr) - net flux (positive = land uptake)
//! - `Emissions|CO2|Gross Deforestation` (GtC/yr)
//! - `Carbon Flux|Regrowth` (GtC/yr)
//! - `Net Primary Production` (GtC/yr)
//! - `Respiration|Terrestrial` (GtC/yr)

use std::any::Any;

use crate::parameters::TerrestrialCarbonParameters;
use rscm_core::component::{
    Component, ComponentState, GridType, InputState, OutputState, RequirementDefinition,
    RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Internal state persisted across timesteps.
///
/// Tracks no-feedback reference pools, regrowth history, CO2 reference,
/// and cumulative land-use emissions for the MAGICC7 attribution scheme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrestrialCarbonState {
    /// No-feedback reference pools [plant, detritus, soil] (GtC).
    /// Evolve without CO2 fertilization or temperature feedbacks.
    pub nofeedback_pools: [FloatValue; 3],

    /// No-feedback, no-deforestation reference pools [plant, detritus, soil] (GtC).
    /// Used to compute regrowth.
    pub nofeedback_nodefo_pools: [FloatValue; 3],

    /// Previous year regrowth fluxes [plant, detritus, soil] (GtC/yr).
    pub prev_regrowth: [FloatValue; 3],

    /// Reference CO2 concentration for fertilization (ppm).
    /// Tracks current CO2 before fertilization start year, then stays fixed.
    pub co2_ref: FloatValue,

    /// Maximum reference CO2 seen (ppm). Used to allow decrease but not increase.
    pub co2_ref_max: FloatValue,

    /// Cumulative land-use emissions (GtC).
    pub cumulative_landuse: FloatValue,

    /// CO2 history for quadratic extrapolation: [t-2, t-1] (ppm).
    pub co2_history: [FloatValue; 2],

    /// Whether the state has been initialized (first timestep completed).
    pub initialized: bool,
}

impl Default for TerrestrialCarbonState {
    fn default() -> Self {
        Self {
            nofeedback_pools: [0.0; 3],
            nofeedback_nodefo_pools: [0.0; 3],
            prev_regrowth: [0.0; 3],
            co2_ref: 0.0,
            co2_ref_max: 0.0,
            cumulative_landuse: 0.0,
            co2_history: [0.0; 2],
            initialized: false,
        }
    }
}

#[typetag::serde]
impl ComponentState for TerrestrialCarbonState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Terrestrial carbon cycle component implementing MAGICC7 TERRCARBON2.
///
/// Uses implicit trapezoidal (Crank-Nicolson) integration for numerical stability.
///
/// # Algorithm
///
/// For each timestep:
///
/// 1. Extrapolate CO2 to mid-year (quadratic)
/// 2. Calculate CO2 fertilization factor (3 methods with blending)
/// 3. Calculate temperature feedback factors
/// 4. Calculate NPP and respiration with feedbacks
/// 5. Compute time-varying turnover times
/// 6. Update pools using implicit trapezoidal scheme
/// 7. Advance no-feedback reference pools and compute regrowth
/// 8. Apply mass conservation correction
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["carbon-cycle", "terrestrial", "magicc"], category = "Carbon Cycle")]
#[inputs(
    co2_concentration { name = "Atmospheric Concentration|CO2", unit = "ppm" },
    temperature { name = "Surface Temperature", unit = "K" },
    landuse_emissions { name = "Emissions|CO2|Land Use", unit = "GtC/yr" },
)]
#[states(
    plant_pool { name = "Carbon Pool|Plant", unit = "GtC" },
    detritus_pool { name = "Carbon Pool|Detritus", unit = "GtC" },
    soil_pool { name = "Carbon Pool|Soil", unit = "GtC" },
)]
#[outputs(
    net_flux { name = "Carbon Flux|Terrestrial", unit = "GtC/yr" },
    gross_deforestation { name = "Emissions|CO2|Gross Deforestation", unit = "GtC/yr" },
    total_regrowth { name = "Carbon Flux|Regrowth", unit = "GtC/yr" },
    npp { name = "Net Primary Production", unit = "GtC/yr" },
    total_respiration { name = "Respiration|Terrestrial", unit = "GtC/yr" },
)]
pub struct TerrestrialCarbon {
    parameters: TerrestrialCarbonParameters,
}

impl TerrestrialCarbon {
    /// Create a new terrestrial carbon component with default parameters.
    pub fn new() -> Self {
        Self::from_parameters(TerrestrialCarbonParameters::default())
    }

    /// Create a new terrestrial carbon component from parameters.
    pub fn from_parameters(parameters: TerrestrialCarbonParameters) -> Self {
        Self { parameters }
    }

    /// Extrapolate CO2 to mid-year using quadratic formula.
    ///
    /// MAGICC7: `CO2_EXTRAP = (3*CO2(t-2) - 10*CO2(t-1) + 15*CO2(t)) / 8`
    fn extrapolate_co2(&self, state: &TerrestrialCarbonState, co2: FloatValue) -> FloatValue {
        if !state.initialized || state.co2_history[0] == 0.0 || state.co2_history[1] == 0.0 {
            return co2;
        }
        (3.0 * state.co2_history[0] - 10.0 * state.co2_history[1] + 15.0 * co2) / 8.0
    }

    /// Logarithmic fertilization (Keeling-Bacastow 1973).
    ///
    /// $$\beta_{log} = 1 + \beta_0 \ln(CO_2 / CO_{2,ref})$$
    fn fert_logarithmic(&self, co2: FloatValue, co2_ref: FloatValue) -> FloatValue {
        if co2 <= 0.0 || co2_ref <= 0.0 {
            return 1.0;
        }
        1.0 + self.parameters.beta * (co2 / co2_ref).ln()
    }

    /// Gifford rectangular hyperbolic / Michaelis-Menten fertilization.
    ///
    /// Wigley (1993) Equations A2 and A5.
    fn fert_gifford(&self, co2: FloatValue, co2_ref: FloatValue) -> FloatValue {
        if co2 <= 0.0 || co2_ref <= 0.0 {
            return 1.0;
        }

        let zero = self.parameters.gifford_conc_for_zero_npp;

        // Gifford R: ratio of NPP at 680 to NPP at 340, using logarithmic formula
        let r = (1.0 + self.parameters.beta * (680.0 / co2_ref).ln())
            / (1.0 + self.parameters.beta * (340.0 / co2_ref).ln());

        let ar = 680.0 - zero;
        let br = 340.0 - zero;

        if (r - 1.0).abs() < 1e-15 {
            return 1.0;
        }

        // Wigley (1993) Eq A5: derive B_ee
        let bee = (ar / br - r) / ((r - 1.0) * ar);

        let dr = co2 - zero;
        let cr = co2_ref - zero;

        if dr.abs() < 1e-15 || cr.abs() < 1e-15 {
            return 1.0;
        }

        // Wigley (1993) Eq A2
        (1.0 / cr + bee) / (1.0 / dr + bee)
    }

    /// Saturating sigmoid fertilization (A. Norton).
    ///
    /// $$\beta_{sig} = A / (1 + \exp(-(CO_2 - CO_{2,ref,sig}) / B))$$
    fn fert_sigmoid(&self, co2: FloatValue, co2_ref: FloatValue) -> FloatValue {
        let a = self.parameters.beta.max(1.0 + 1e-10); // Must be > 1
        let b = self.parameters.fertilization_factor2;

        if b.abs() < 1e-15 {
            return 1.0;
        }

        // Calculate reference so that beta = 1 at co2_ref
        let co2_ref_sig = co2_ref + b * (a - 1.0).ln();

        a / (1.0 + (-(co2 - co2_ref_sig) / b).exp())
    }

    /// Calculate CO2 fertilization factor with method blending.
    ///
    /// MAGICC7: CO2_EFF_FERTILIZATION_FACTOR
    fn fertilization_factor(&self, co2: FloatValue, co2_ref: FloatValue) -> FloatValue {
        let method = self.parameters.fertilization_method;

        if method < 1.0 {
            return 1.0;
        }

        let method = method.min(3.0);

        let result = if method <= 2.0 {
            let w = method - 1.0;
            let beta_log = self.fert_logarithmic(co2, co2_ref);
            let beta_giff = self.fert_gifford(co2, co2_ref);
            (1.0 - w) * beta_log + w * beta_giff
        } else {
            let w = method - 2.0;
            let beta_giff = self.fert_gifford(co2, co2_ref);
            let beta_sig = self.fert_sigmoid(co2, co2_ref);
            (1.0 - w) * beta_giff + w * beta_sig
        };

        result.max(0.1) // Floor to prevent negative/zero fertilization
    }

    /// Calculate temperature effect multiplier.
    ///
    /// $$f_T(\Delta T) = e^{\gamma \times \Delta T}$$
    fn temperature_factor(&self, temperature: FloatValue, sensitivity: FloatValue) -> FloatValue {
        (sensitivity * temperature).exp()
    }

    /// Calculate respiration from plant pool with feedbacks.
    ///
    /// Supports MAGICC7 Method 1 and Method 2.
    fn calculate_respiration(
        &self,
        beta_fert: FloatValue,
        temperature: FloatValue,
        plant_pool: FloatValue,
    ) -> FloatValue {
        let temp_effect =
            self.temperature_factor(temperature, self.parameters.resp_temp_sensitivity);

        match self.parameters.plantbox_resp_method {
            2 => {
                let alpha = self.parameters.plantbox_resp_fertscale;
                let pool_ratio = (plant_pool / self.parameters.plant_pool_pi).min(1.0);
                self.parameters.respiration_pi
                    * (1.0 + alpha * (beta_fert - 1.0))
                    * pool_ratio
                    * temp_effect
            }
            _ => {
                // Method 1 (default)
                self.parameters.respiration_pi * beta_fert * temp_effect
            }
        }
    }

    /// Update a carbon pool using implicit trapezoidal integration.
    ///
    /// $$C_{n+1} = \frac{C_n (1 - 0.5 k_{eff} \Delta t) + F_{in} \Delta t}{1 + 0.5 k_{eff} \Delta t}$$
    fn implicit_pool_step(
        pool_current: FloatValue,
        tau: FloatValue,
        flux_in: FloatValue,
        temp_factor: FloatValue,
        dt: FloatValue,
    ) -> (FloatValue, FloatValue) {
        let k_eff = temp_factor / tau;
        let half_k = 0.5 * k_eff * dt;

        let new_pool = ((1.0 - half_k) * pool_current + flux_in * dt) / (1.0 + half_k);
        let new_pool = new_pool.max(0.0);

        // Turnover flux (average over timestep)
        let turnover = 0.5 * k_eff * (pool_current + new_pool);

        (new_pool, turnover)
    }

    /// Calculate time-varying turnover time for a pool.
    ///
    /// Turnover decreases as cumulative deforestation increases.
    /// MAGICC7: `tau(t) = (C0 - f_norgrwth * f_defo * cum_CO2B) / flux0`
    fn time_varying_tau(
        pool_pi: FloatValue,
        norgrwth_frac: FloatValue,
        defo_frac: FloatValue,
        cumulative_landuse: FloatValue,
        flux_pi: FloatValue,
    ) -> FloatValue {
        if flux_pi.abs() < 1e-10 {
            return 100.0;
        }
        let numerator = pool_pi - norgrwth_frac * defo_frac * cumulative_landuse;
        let tau = numerator / flux_pi;
        tau.max(1.0) // Floor at 1 year for stability
    }

    /// Advance a set of 3 pools by one timestep.
    ///
    /// Returns (new_pools, turnover_plant, turnover_detritus, turnover_soil).
    fn advance_pools(
        &self,
        pools: &[FloatValue; 3],
        npp: FloatValue,
        respiration: FloatValue,
        gross_defo: &[FloatValue; 3],
        tau_plant: FloatValue,
        tau_detritus: FloatValue,
        tau_soil: FloatValue,
        temp_factor_detritus: FloatValue,
        temp_factor_soil: FloatValue,
        dt: FloatValue,
    ) -> ([FloatValue; 3], FloatValue, FloatValue, FloatValue) {
        let [plant, detritus, soil] = *pools;

        let npp_to_plant = npp * self.parameters.frac_npp_to_plant;
        let flux_in_plant = npp_to_plant - respiration - gross_defo[0];
        let (new_plant, turnover_plant) =
            Self::implicit_pool_step(plant, tau_plant, flux_in_plant, 1.0, dt);

        let npp_to_detritus = npp * self.parameters.frac_npp_to_detritus;
        let flux_in_detritus = npp_to_detritus
            + self.parameters.frac_plant_to_detritus * turnover_plant
            - gross_defo[1];
        let (new_detritus, turnover_detritus) = Self::implicit_pool_step(
            detritus,
            tau_detritus,
            flux_in_detritus,
            temp_factor_detritus,
            dt,
        );

        let npp_to_soil = npp * self.parameters.frac_npp_to_soil();
        let plant_to_soil = (1.0 - self.parameters.frac_plant_to_detritus) * turnover_plant;
        let detritus_to_soil = self.parameters.frac_detritus_to_soil * turnover_detritus;
        let flux_in_soil = npp_to_soil + plant_to_soil + detritus_to_soil - gross_defo[2];
        let (new_soil, turnover_soil) =
            Self::implicit_pool_step(soil, tau_soil, flux_in_soil, temp_factor_soil, dt);

        (
            [new_plant, new_detritus, new_soil],
            turnover_plant,
            turnover_detritus,
            turnover_soil,
        )
    }

    /// Solve the terrestrial carbon cycle for one timestep (public API without state).
    ///
    /// This is a simplified interface that creates a temporary state. For full
    /// MAGICC7 fidelity (regrowth, mass conservation), use `solve_with_state`.
    pub fn solve_pools(
        &self,
        co2: FloatValue,
        temperature: FloatValue,
        landuse_emissions: FloatValue,
        pools: [FloatValue; 3],
        dt: FloatValue,
    ) -> ([FloatValue; 3], FloatValue) {
        let mut state = TerrestrialCarbonState::default();
        let result = self.solve_terrestrial(
            &mut state,
            co2,
            temperature,
            landuse_emissions,
            pools,
            dt,
            None,
        );
        (result.new_pools, result.net_flux)
    }

    /// Full solve returning all diagnostics.
    fn solve_terrestrial(
        &self,
        state: &mut TerrestrialCarbonState,
        co2: FloatValue,
        temperature: FloatValue,
        landuse_emissions: FloatValue,
        pools: [FloatValue; 3],
        dt: FloatValue,
        current_year: Option<FloatValue>,
    ) -> TerrestrialResult {
        let p = &self.parameters;

        // Initialize state on first call
        if !state.initialized {
            state.nofeedback_pools = [p.plant_pool_pi, p.detritus_pool_pi, p.soil_pool_pi];
            state.nofeedback_nodefo_pools = [p.plant_pool_pi, p.detritus_pool_pi, p.soil_pool_pi];
            state.co2_ref = p.co2_pi;
            state.co2_ref_max = p.co2_pi;
            state.co2_history = [co2, co2];
            state.initialized = true;
        }

        // Step 1: CO2 extrapolation to mid-year
        let co2_extrap = self.extrapolate_co2(state, co2);

        // Step 2: Update reference CO2 for fertilization
        let year = current_year.unwrap_or(p.fertilization_yrstart + 1.0);
        if year < p.fertilization_yrstart {
            state.co2_ref = co2_extrap;
            state.co2_ref_max = co2_extrap;
        } else {
            state.co2_ref = co2_extrap.min(state.co2_ref_max);
        }
        let co2_ref = state.co2_ref;

        // Step 3: Temperature feedback factors
        let apply_temp = current_year.map_or(true, |y| y >= p.tempfeedback_yrstart);
        let temp = if apply_temp { temperature } else { 0.0 };

        let temp_npp = self.temperature_factor(temp, p.npp_temp_sensitivity);
        let temp_detritus = self.temperature_factor(temp, p.detritus_temp_sensitivity);
        let temp_soil = self.temperature_factor(temp, p.soil_temp_sensitivity);

        // Step 4: CO2 fertilization
        let beta_fert = self.fertilization_factor(co2_extrap, co2_ref);

        // Step 5: NPP with feedbacks
        let npp = p.npp_pi * beta_fert * temp_npp;

        // Step 6: Respiration (recomputes temp factor internally from `temp`)
        let respiration = self.calculate_respiration(beta_fert, temp, pools[0]);

        // Step 7: Time-varying turnover times
        let cum_lu = state.cumulative_landuse;
        let tau_plant = Self::time_varying_tau(
            p.plant_pool_pi,
            p.norgrwth_frac_defo,
            p.frac_deforest_plant,
            cum_lu,
            p.net_flux_to_plant_pi(),
        );
        let tau_detritus = Self::time_varying_tau(
            p.detritus_pool_pi,
            p.norgrwth_frac_defo,
            p.frac_deforest_detritus,
            cum_lu,
            p.flux_into_detritus_pi(),
        );
        let tau_soil = Self::time_varying_tau(
            p.soil_pool_pi,
            p.norgrwth_frac_defo,
            p.frac_deforest_soil(),
            cum_lu,
            p.flux_into_soil_pi(),
        );

        // Step 8: Advance no-feedback, no-defo pools (for regrowth calculation)
        let nf_npp = p.npp_pi; // No fertilization, no temp feedback
        let nf_resp = p.respiration_pi;
        let zero_defo = [0.0; 3];
        let (new_nf_nodefo, _, _, _) = self.advance_pools(
            &state.nofeedback_nodefo_pools,
            nf_npp,
            nf_resp,
            &zero_defo,
            tau_plant,
            tau_detritus,
            tau_soil,
            1.0, // no temp feedback
            1.0,
            dt,
        );

        // Step 9: Advance nofeedback pools with deforestation, then compute regrowth
        let mut regrowth = [0.0; 3];
        let nf_defo = [
            p.frac_deforest_plant * landuse_emissions,
            p.frac_deforest_detritus * landuse_emissions,
            p.frac_deforest_soil() * landuse_emissions,
        ];

        // Add previous regrowth to get gross deforestation for no-feedback pools
        let nf_gross_defo = std::array::from_fn(|i| nf_defo[i] + state.prev_regrowth[i]);

        let (new_nf_pools, _, _, _) = self.advance_pools(
            &state.nofeedback_pools,
            nf_npp,
            nf_resp,
            &nf_gross_defo,
            tau_plant,
            tau_detritus,
            tau_soil,
            1.0,
            1.0,
            dt,
        );

        // Update regrowth: the difference in pool changes between no-defo and with-defo
        for i in 0..3 {
            let delta_nodefo = new_nf_nodefo[i] - state.nofeedback_nodefo_pools[i];
            let delta_withdefo = new_nf_pools[i] - state.nofeedback_pools[i];
            regrowth[i] = delta_nodefo - delta_withdefo;
        }

        // Step 10: Gross deforestation for main pools
        let gross_defo = std::array::from_fn(|i| nf_defo[i] + regrowth[i]);

        // Step 11: Advance main pools with feedbacks
        let (mut new_pools, _turnover_plant, turnover_detritus, turnover_soil) = self
            .advance_pools(
                &pools,
                npp,
                respiration,
                &gross_defo,
                tau_plant,
                tau_detritus,
                tau_soil,
                temp_detritus,
                temp_soil,
                dt,
            );

        // Step 12: Mass conservation correction
        let nf_delta: FloatValue = (0..3)
            .map(|i| new_nf_pools[i] - state.nofeedback_pools[i])
            .sum();
        let correction = landuse_emissions * dt + nf_delta;
        new_pools[0] -= correction;
        let mut corrected_nf = new_nf_pools;
        corrected_nf[0] -= correction;

        // Step 13: Calculate net flux and diagnostics
        let detritus_to_atm = (1.0 - p.frac_detritus_to_soil) * turnover_detritus;
        let soil_to_atm = turnover_soil;
        let total_respiration = respiration + detritus_to_atm + soil_to_atm;
        let total_gross_defo: FloatValue = gross_defo.iter().sum();
        let total_regrowth: FloatValue = regrowth.iter().sum();

        // Net flux = change in total pool (positive = land uptake)
        let pool_change: FloatValue = (0..3).map(|i| new_pools[i] - pools[i]).sum();
        let net_flux = pool_change / dt;

        // Update state for next timestep
        state.nofeedback_pools = corrected_nf;
        state.nofeedback_nodefo_pools = new_nf_nodefo;
        state.prev_regrowth = regrowth;
        state.cumulative_landuse += landuse_emissions * dt;
        state.co2_history[0] = state.co2_history[1];
        state.co2_history[1] = co2;

        TerrestrialResult {
            new_pools,
            net_flux,
            gross_deforestation: total_gross_defo,
            total_regrowth,
            npp,
            total_respiration,
        }
    }

    /// Internal solve used by both `solve()` and `solve_with_state()`.
    fn solve_impl(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
        state: &mut TerrestrialCarbonState,
    ) -> RSCMResult<OutputState> {
        let inputs = TerrestrialCarbonInputs::from_input_state(input_state);

        let co2 = inputs.co2_concentration.get();
        let temperature = inputs.temperature.get();
        let landuse = inputs.landuse_emissions.get();

        let plant = inputs.plant_pool.at_start();
        let detritus = inputs.detritus_pool.at_start();
        let soil = inputs.soil_pool.at_start();

        let dt = t_next - t_current;

        let result = self.solve_terrestrial(
            state,
            co2,
            temperature,
            landuse,
            [plant, detritus, soil],
            dt,
            Some(t_current),
        );

        let outputs = TerrestrialCarbonOutputs {
            plant_pool: result.new_pools[0],
            detritus_pool: result.new_pools[1],
            soil_pool: result.new_pools[2],
            net_flux: result.net_flux,
            gross_deforestation: result.gross_deforestation,
            total_regrowth: result.total_regrowth,
            npp: result.npp,
            total_respiration: result.total_respiration,
        };

        Ok(outputs.into())
    }
}

/// Internal result struct for solve_terrestrial.
struct TerrestrialResult {
    new_pools: [FloatValue; 3],
    net_flux: FloatValue,
    gross_deforestation: FloatValue,
    total_regrowth: FloatValue,
    npp: FloatValue,
    total_respiration: FloatValue,
}

impl Default for TerrestrialCarbon {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde]
impl Component for TerrestrialCarbon {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let mut state = TerrestrialCarbonState::default();
        self.solve_impl(t_current, t_next, input_state, &mut state)
    }

    fn create_initial_state(&self) -> Box<dyn ComponentState> {
        Box::new(TerrestrialCarbonState::default())
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
            .downcast_mut::<TerrestrialCarbonState>()
            .expect("Internal state should be TerrestrialCarbonState");
        self.solve_impl(t_current, t_next, input_state, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_component() -> TerrestrialCarbon {
        TerrestrialCarbon::from_parameters(TerrestrialCarbonParameters::default())
    }

    fn pi_pools() -> [FloatValue; 3] {
        let params = TerrestrialCarbonParameters::default();
        [
            params.plant_pool_pi,
            params.detritus_pool_pi,
            params.soil_pool_pi,
        ]
    }

    #[test]
    fn test_steady_state_at_preindustrial() {
        let component = default_component();
        let pools = pi_pools();
        let co2_pi = component.parameters.co2_pi;

        let (new_pools, net_flux) = component.solve_pools(co2_pi, 0.0, 0.0, pools, 1.0);

        for (i, (old, new)) in pools.iter().zip(new_pools.iter()).enumerate() {
            let rel_change = ((new - old) / old).abs();
            assert!(
                rel_change < 0.05,
                "Pool {} should be stable at PI: {:.2} -> {:.2} ({:.1}% change)",
                i,
                old,
                new,
                rel_change * 100.0
            );
        }

        assert!(
            net_flux.abs() < 1.0,
            "Net flux should be ~0 at steady state, got {:.2} GtC/yr",
            net_flux
        );
    }

    #[test]
    fn test_fert_logarithmic_at_pi() {
        let component = default_component();
        let co2_pi = component.parameters.co2_pi;
        let factor = component.fert_logarithmic(co2_pi, co2_pi);
        assert!(
            (factor - 1.0).abs() < 1e-10,
            "Log fertilization should be 1.0 at PI CO2"
        );
    }

    #[test]
    fn test_fert_logarithmic_at_doubled_co2() {
        let component = default_component();
        let co2_pi = component.parameters.co2_pi;
        let factor = component.fert_logarithmic(co2_pi * 2.0, co2_pi);
        let expected = 1.0 + component.parameters.beta * 2.0_f64.ln();
        assert!(
            (factor - expected).abs() < 0.01,
            "Log fertilization at 2xCO2 should be {:.3}, got {:.3}",
            expected,
            factor
        );
    }

    #[test]
    fn test_fert_gifford_at_pi() {
        let component = default_component();
        let co2_pi = component.parameters.co2_pi;
        let factor = component.fert_gifford(co2_pi, co2_pi);
        assert!(
            (factor - 1.0).abs() < 1e-10,
            "Gifford fertilization should be 1.0 at PI CO2, got {:.6}",
            factor
        );
    }

    #[test]
    fn test_fert_gifford_increases_with_co2() {
        let component = default_component();
        let co2_pi = component.parameters.co2_pi;
        let factor_350 = component.fert_gifford(350.0, co2_pi);
        let factor_500 = component.fert_gifford(500.0, co2_pi);
        assert!(
            factor_500 > factor_350,
            "Gifford factor should increase with CO2: {:.4} vs {:.4}",
            factor_350,
            factor_500
        );
    }

    #[test]
    fn test_fert_sigmoid_at_pi() {
        let component = default_component();
        let co2_pi = component.parameters.co2_pi;
        let factor = component.fert_sigmoid(co2_pi, co2_pi);
        assert!(
            (factor - 1.0).abs() < 0.01,
            "Sigmoid fertilization should be ~1.0 at PI CO2, got {:.6}",
            factor
        );
    }

    #[test]
    fn test_fertilization_blending_method_1_pure_log() {
        let params = TerrestrialCarbonParameters {
            fertilization_method: 1.0,
            ..Default::default()
        };
        let component = TerrestrialCarbon::from_parameters(params);
        let co2_pi = component.parameters.co2_pi;

        let blended = component.fertilization_factor(400.0, co2_pi);
        let pure_log = component.fert_logarithmic(400.0, co2_pi);
        assert!(
            (blended - pure_log).abs() < 1e-10,
            "Method 1.0 should be pure logarithmic: {:.6} vs {:.6}",
            blended,
            pure_log
        );
    }

    #[test]
    fn test_fertilization_blending_method_2_pure_gifford() {
        let params = TerrestrialCarbonParameters {
            fertilization_method: 2.0,
            ..Default::default()
        };
        let component = TerrestrialCarbon::from_parameters(params);
        let co2_pi = component.parameters.co2_pi;

        let blended = component.fertilization_factor(400.0, co2_pi);
        let pure_giff = component.fert_gifford(400.0, co2_pi);
        assert!(
            (blended - pure_giff).abs() < 1e-10,
            "Method 2.0 should be pure Gifford: {:.6} vs {:.6}",
            blended,
            pure_giff
        );
    }

    #[test]
    fn test_fertilization_no_fert_below_1() {
        let params = TerrestrialCarbonParameters {
            fertilization_method: 0.5,
            ..Default::default()
        };
        let component = TerrestrialCarbon::from_parameters(params);
        let factor = component.fertilization_factor(500.0, 278.0);
        assert!(
            (factor - 1.0).abs() < 1e-10,
            "Method < 1.0 should give factor 1.0, got {:.6}",
            factor
        );
    }

    #[test]
    fn test_respiration_method2_at_pi() {
        let params = TerrestrialCarbonParameters {
            plantbox_resp_method: 2,
            plantbox_resp_fertscale: 0.5,
            ..Default::default()
        };
        let component = TerrestrialCarbon::from_parameters(params.clone());

        // At PI: beta=1, temp=0, pool=pool_pi -> should match method 1
        let resp = component.calculate_respiration(1.0, 0.0, params.plant_pool_pi);
        assert!(
            (resp - params.respiration_pi).abs() < 1e-10,
            "Method 2 at PI should match R_h0: {:.6} vs {:.6}",
            resp,
            params.respiration_pi
        );
    }

    #[test]
    fn test_higher_co2_increases_uptake() {
        let component = default_component();
        let pools = pi_pools();
        let co2_pi = component.parameters.co2_pi;

        let (_, flux_pi) = component.solve_pools(co2_pi, 0.0, 0.0, pools, 1.0);
        let (_, flux_high) = component.solve_pools(co2_pi * 1.5, 0.0, 0.0, pools, 1.0);

        assert!(
            flux_high > flux_pi,
            "Higher CO2 should increase net uptake: {:.2} vs {:.2}",
            flux_high,
            flux_pi
        );
    }

    #[test]
    fn test_warming_reduces_net_uptake() {
        let component = default_component();
        let pools = pi_pools();
        let co2_high = component.parameters.co2_pi * 1.5;

        let (_, flux_cold) = component.solve_pools(co2_high, 0.0, 0.0, pools, 1.0);
        let (_, flux_warm) = component.solve_pools(co2_high, 3.0, 0.0, pools, 1.0);

        assert!(
            flux_warm < flux_cold,
            "Warming should reduce net uptake: {:.2} vs {:.2}",
            flux_warm,
            flux_cold
        );
    }

    #[test]
    fn test_pools_remain_positive() {
        let component = default_component();
        let pools = pi_pools();

        let (new_pools, _) =
            component.solve_pools(component.parameters.co2_pi, 10.0, 5.0, pools, 1.0);

        for (i, pool) in new_pools.iter().enumerate() {
            assert!(
                *pool >= 0.0,
                "Pool {} should remain non-negative, got {}",
                i,
                pool
            );
        }
    }

    #[test]
    fn test_deforestation_distributed_across_pools() {
        let component = default_component();
        let pools = pi_pools();

        let (pools_no_lu, _) =
            component.solve_pools(component.parameters.co2_pi, 0.0, 0.0, pools, 1.0);
        let (pools_with_lu, _) =
            component.solve_pools(component.parameters.co2_pi, 0.0, 5.0, pools, 1.0);

        // Plant pool should decrease most (70% of deforestation)
        assert!(
            pools_with_lu[0] < pools_no_lu[0],
            "Land use should reduce plant pool"
        );
    }

    #[test]
    fn test_multi_year_stability() {
        let component = default_component();
        let mut pools = pi_pools();
        let co2_pi = component.parameters.co2_pi;

        for _ in 0..100 {
            let (new_pools, _) = component.solve_pools(co2_pi, 0.0, 0.0, pools, 1.0);
            pools = new_pools;
        }

        let initial = pi_pools();
        let total_initial: FloatValue = initial.iter().sum();
        let total_final: FloatValue = pools.iter().sum();

        let rel_change = ((total_final - total_initial) / total_initial).abs();
        assert!(
            rel_change < 0.1,
            "Total pool should be stable over 100 years: {:.1} -> {:.1} ({:.1}% change)",
            total_initial,
            total_final,
            rel_change * 100.0
        );
    }

    #[test]
    fn test_elevated_co2_increases_total_pool() {
        let component = default_component();
        let mut pools = pi_pools();
        let co2_elevated = component.parameters.co2_pi * 1.5;

        for _ in 0..50 {
            let (new_pools, _) = component.solve_pools(co2_elevated, 0.0, 0.0, pools, 1.0);
            pools = new_pools;
        }

        let initial_total: FloatValue = pi_pools().iter().sum();
        let final_total: FloatValue = pools.iter().sum();

        assert!(
            final_total > initial_total,
            "Elevated CO2 should increase total terrestrial carbon: {:.1} -> {:.1}",
            initial_total,
            final_total
        );
    }

    #[test]
    fn test_definitions() {
        let component = default_component();
        let defs = component.definitions();

        // 3 inputs + 3 states + 5 outputs = 11 definitions
        assert!(
            defs.len() >= 11,
            "Should have at least 11 definitions, got {}",
            defs.len()
        );

        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"Atmospheric Concentration|CO2"));
        assert!(names.contains(&"Surface Temperature"));
        assert!(names.contains(&"Emissions|CO2|Land Use"));
        assert!(names.contains(&"Carbon Pool|Plant"));
        assert!(names.contains(&"Carbon Pool|Detritus"));
        assert!(names.contains(&"Carbon Pool|Soil"));
        assert!(names.contains(&"Carbon Flux|Terrestrial"));
        assert!(names.contains(&"Net Primary Production"));
        assert!(names.contains(&"Respiration|Terrestrial"));
    }

    #[test]
    fn test_serialization() {
        let component = default_component();
        let json = serde_json::to_string(&component).expect("Serialization failed");
        let parsed: TerrestrialCarbon =
            serde_json::from_str(&json).expect("Deserialization failed");

        assert!(
            (component.parameters.npp_pi - parsed.parameters.npp_pi).abs() < 1e-10,
            "Parameters should survive round-trip serialization"
        );
    }

    #[test]
    fn test_state_serialization() {
        let mut state = TerrestrialCarbonState::default();
        state.initialized = true;
        state.cumulative_landuse = 42.0;
        state.co2_ref = 300.0;

        let json = serde_json::to_string(&state).expect("State serialization failed");
        let parsed: TerrestrialCarbonState =
            serde_json::from_str(&json).expect("State deserialization failed");

        assert!(parsed.initialized);
        assert!((parsed.cumulative_landuse - 42.0).abs() < 1e-10);
        assert!((parsed.co2_ref - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_very_low_co2() {
        let component = default_component();
        let pools = pi_pools();

        let (new_pools, net_flux) = component.solve_pools(100.0, 0.0, 0.0, pools, 1.0);

        assert!(
            new_pools.iter().all(|p| p.is_finite()),
            "Pools should remain finite at low CO2"
        );
        assert!(
            net_flux.is_finite(),
            "Net flux should remain finite at low CO2"
        );
    }

    #[test]
    fn test_very_high_co2() {
        let component = default_component();
        let pools = pi_pools();

        let (new_pools, net_flux) = component.solve_pools(2000.0, 0.0, 0.0, pools, 1.0);

        assert!(
            new_pools.iter().all(|p| p.is_finite() && *p > 0.0),
            "Pools should remain finite and positive at high CO2"
        );
        assert!(
            net_flux.is_finite() && net_flux > 0.0,
            "Net flux should be positive at high CO2"
        );
    }

    #[test]
    fn test_extreme_warming() {
        let component = default_component();
        let pools = pi_pools();

        let (new_pools, net_flux) =
            component.solve_pools(component.parameters.co2_pi, 10.0, 0.0, pools, 1.0);

        assert!(
            new_pools.iter().all(|p| p.is_finite() && *p >= 0.0),
            "Pools should remain finite and non-negative under extreme warming"
        );
        assert!(
            net_flux.is_finite(),
            "Net flux should remain finite under extreme warming"
        );
        assert!(
            net_flux < 0.0,
            "Extreme warming should cause net carbon release, got {:.2}",
            net_flux
        );
    }

    #[test]
    fn test_time_varying_turnover() {
        let p = TerrestrialCarbonParameters::default();

        let tau_0 = TerrestrialCarbon::time_varying_tau(
            p.plant_pool_pi,
            p.norgrwth_frac_defo,
            p.frac_deforest_plant,
            0.0,
            p.net_flux_to_plant_pi(),
        );
        let tau_100 = TerrestrialCarbon::time_varying_tau(
            p.plant_pool_pi,
            p.norgrwth_frac_defo,
            p.frac_deforest_plant,
            100.0,
            p.net_flux_to_plant_pi(),
        );

        assert!(
            tau_100 < tau_0,
            "Cumulative deforestation should reduce turnover time: {:.2} vs {:.2}",
            tau_100,
            tau_0
        );
        assert!(
            tau_100 >= 1.0,
            "Turnover time should be floored at 1.0 year"
        );
    }
}
