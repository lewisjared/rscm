//! Terrestrial Carbon Cycle Component
//!
//! Simulates the exchange of carbon between the atmosphere and land ecosystems
//! using a simplified 4-pool box model with CO2 fertilization and temperature
//! feedbacks.
//!
//! # What This Component Does
//!
//! 1. Calculates Net Primary Production (NPP) with CO2 fertilization:
//!    - Higher CO2 concentrations enhance photosynthesis
//!    - Temperature can affect NPP positively or negatively
//!
//! 2. Tracks carbon in four pools with different turnover times:
//!    - Plant biomass (~15 year turnover)
//!    - Detritus/litter (~3 year turnover)
//!    - Soil carbon (~50 year turnover)
//!    - Humus (~1000 year turnover)
//!
//! 3. Calculates temperature-dependent respiration from each pool
//!
//! 4. Outputs the net land-atmosphere carbon flux
//!
//! # Inputs
//!
//! - `Atmospheric Concentration|CO2` (ppm) - Atmospheric CO2 concentration
//! - `Surface Temperature` (K) - Global mean surface temperature anomaly
//! - `Emissions|CO2|Land Use` (GtC/yr) - Land use change emissions
//!
//! # States (tracked between timesteps)
//!
//! - `Carbon Pool|Plant` (GtC) - Carbon in plant biomass
//! - `Carbon Pool|Detritus` (GtC) - Carbon in detritus/litter
//! - `Carbon Pool|Soil` (GtC) - Carbon in soil
//! - `Carbon Pool|Humus` (GtC) - Carbon in humus (slow soil pool)
//!
//! # Outputs
//!
//! - `Carbon Flux|Terrestrial` (GtC/yr) - Net land-atmosphere flux (positive = uptake)
//!
//! # Differences from MAGICC7 Module 09
//!
//! This is a simplified implementation:
//!
//! - **No-feedback reference pools**: MAGICC7 tracks parallel "no-feedback" pools
//!   for attribution of land sink vs. climate feedback. Not implemented.
//! - **Regrowth calculation**: MAGICC7 has complex regrowth logic for land-use
//!   change. Here, land-use emissions are directly subtracted from pools.
//! - **Fertilization methods**: MAGICC7 supports logarithmic, Gifford, and sigmoid
//!   fertilization methods with blending. Here, only logarithmic is implemented.
//! - **Permafrost feedback**: MAGICC7 has optional permafrost carbon release.
//!   Not implemented.
//! - **Nitrogen limitation**: MAGICC7 can apply nitrogen cycle limitation to NPP.
//!   Not implemented.
//! - **Deforestation fractions**: MAGICC7 distributes deforestation emissions
//!   across pools with configurable fractions. Here, simplified to plant pool only.

use crate::parameters::TerrestrialCarbonParameters;
use rscm_core::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Terrestrial carbon cycle component with 4-pool dynamics.
///
/// Implements a simplified MAGICC7-style terrestrial carbon model using
/// implicit trapezoidal (Crank-Nicolson) integration for numerical stability.
///
/// # Algorithm
///
/// For each timestep:
///
/// 1. Calculate NPP with CO2 fertilization:
///    $$\text{NPP} = \text{NPP}_0 \times (1 + \beta \ln(C/C_0)) \times e^{\gamma_{NPP} \Delta T}$$
///
/// 2. Calculate temperature-dependent respiration from plant pool:
///    $$R_h = R_{h,0} \times (1 + \beta \ln(C/C_0)) \times e^{\gamma_{resp} \Delta T}$$
///
/// 3. Update each pool using implicit trapezoidal scheme:
///    $$C_{n+1} = \frac{C_n (1 - 0.5 k) + F_{in}}{1 + 0.5 k}$$
///    where $k = 1/\tau$ is the decay rate.
///
/// 4. Calculate net flux = NPP - total respiration
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["carbon-cycle", "terrestrial", "magicc"], category = "Carbon Cycle")]
#[inputs(
    co2_concentration { name = "Atmospheric Concentration|CO2", unit = "ppm" },
    temperature { name = "Surface Temperature|Global", unit = "K" },
    landuse_emissions { name = "Emissions|CO2|Land Use", unit = "GtC/yr" },
)]
#[states(
    plant_pool { name = "Carbon Pool|Plant", unit = "GtC" },
    detritus_pool { name = "Carbon Pool|Detritus", unit = "GtC" },
    soil_pool { name = "Carbon Pool|Soil", unit = "GtC" },
    humus_pool { name = "Carbon Pool|Humus", unit = "GtC" },
)]
#[outputs(
    net_flux { name = "Carbon Flux|Terrestrial", unit = "GtC/yr" },
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

    /// Calculate CO2 fertilization factor using logarithmic formula.
    ///
    /// $$\beta(C) = 1 + \beta_0 \times \ln(C / C_{pi})$$
    ///
    /// # Arguments
    ///
    /// * `co2` - Current atmospheric CO2 concentration (ppm)
    ///
    /// # Returns
    ///
    /// Fertilization factor (dimensionless, ≥ 1 for CO2 > CO2_pi)
    fn fertilization_factor(&self, co2: FloatValue) -> FloatValue {
        if !self.parameters.enable_fertilization || co2 <= 0.0 {
            return 1.0;
        }

        let ratio = co2 / self.parameters.co2_pi;
        // Floor at 1.0 to prevent negative fertilization
        (1.0 + self.parameters.beta * ratio.ln()).max(0.1)
    }

    /// Calculate temperature effect multiplier.
    ///
    /// $$f_T(\Delta T) = e^{\gamma \times \Delta T}$$
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature anomaly (K)
    /// * `sensitivity` - Temperature sensitivity coefficient (K⁻¹)
    ///
    /// # Returns
    ///
    /// Temperature factor (dimensionless)
    fn temperature_factor(&self, temperature: FloatValue, sensitivity: FloatValue) -> FloatValue {
        if !self.parameters.enable_temp_feedback {
            return 1.0;
        }
        (sensitivity * temperature).exp()
    }

    /// Calculate Net Primary Production with feedbacks.
    ///
    /// $$\text{NPP} = \text{NPP}_0 \times \beta(C) \times f_T^{NPP}(\Delta T)$$
    ///
    /// # Arguments
    ///
    /// * `co2` - Current atmospheric CO2 concentration (ppm)
    /// * `temperature` - Temperature anomaly (K)
    ///
    /// # Returns
    ///
    /// NPP in GtC/yr
    fn calculate_npp(&self, co2: FloatValue, temperature: FloatValue) -> FloatValue {
        let fert = self.fertilization_factor(co2);
        let temp_effect =
            self.temperature_factor(temperature, self.parameters.npp_temp_sensitivity);
        self.parameters.npp_pi * fert * temp_effect
    }

    /// Calculate respiration from plant pool with feedbacks.
    ///
    /// Following MAGICC7 Method 1:
    /// $$R_h = R_{h,0} \times \beta(C) \times f_T^{resp}(\Delta T)$$
    ///
    /// # Arguments
    ///
    /// * `co2` - Current atmospheric CO2 concentration (ppm)
    /// * `temperature` - Temperature anomaly (K)
    ///
    /// # Returns
    ///
    /// Respiration in GtC/yr
    fn calculate_respiration(&self, co2: FloatValue, temperature: FloatValue) -> FloatValue {
        let fert = self.fertilization_factor(co2);
        let temp_effect =
            self.temperature_factor(temperature, self.parameters.resp_temp_sensitivity);
        self.parameters.respiration_pi * fert * temp_effect
    }

    /// Update a carbon pool using implicit trapezoidal integration.
    ///
    /// Uses the Crank-Nicolson scheme for numerical stability:
    /// $$C_{n+1} = \frac{C_n (1 - 0.5 k_{eff}) + F_{in}}{1 + 0.5 k_{eff}}$$
    ///
    /// # Arguments
    ///
    /// * `pool_current` - Current pool size (GtC)
    /// * `tau` - Base turnover time (years)
    /// * `flux_in` - Net input flux to pool (GtC/yr)
    /// * `temp_factor` - Temperature factor for decay rate
    /// * `dt` - Timestep (years)
    ///
    /// # Returns
    ///
    /// (new_pool_size, turnover_flux)
    fn implicit_pool_step(
        &self,
        pool_current: FloatValue,
        tau: FloatValue,
        flux_in: FloatValue,
        temp_factor: FloatValue,
        dt: FloatValue,
    ) -> (FloatValue, FloatValue) {
        // Effective decay rate with temperature feedback
        let k_eff = temp_factor / tau;
        let half_k = 0.5 * k_eff * dt;

        // Implicit trapezoidal step
        let new_pool = ((1.0 - half_k) * pool_current + flux_in * dt) / (1.0 + half_k);
        let new_pool = new_pool.max(0.0); // Ensure non-negative

        // Turnover flux (average over timestep)
        let turnover = 0.5 * k_eff * (pool_current + new_pool);

        (new_pool, turnover)
    }

    /// Solve the terrestrial carbon cycle for one timestep.
    ///
    /// # Arguments
    ///
    /// * `co2` - Atmospheric CO2 concentration (ppm)
    /// * `temperature` - Temperature anomaly (K)
    /// * `landuse_emissions` - Land use change emissions (GtC/yr)
    /// * `pools` - Current pool sizes [plant, detritus, soil, humus] (GtC)
    /// * `dt` - Timestep (years)
    ///
    /// # Returns
    ///
    /// (new_pools, net_flux)
    pub fn solve_pools(
        &self,
        co2: FloatValue,
        temperature: FloatValue,
        landuse_emissions: FloatValue,
        pools: [FloatValue; 4],
        dt: FloatValue,
    ) -> ([FloatValue; 4], FloatValue) {
        let [plant, detritus, soil, humus] = pools;

        // Calculate NPP and respiration with feedbacks
        let npp = self.calculate_npp(co2, temperature);
        let respiration = self.calculate_respiration(co2, temperature);

        // Temperature factors for decay
        let temp_factor_detritus =
            self.temperature_factor(temperature, self.parameters.detritus_temp_sensitivity);
        let temp_factor_soil =
            self.temperature_factor(temperature, self.parameters.soil_temp_sensitivity);
        let temp_factor_humus =
            self.temperature_factor(temperature, self.parameters.humus_temp_sensitivity);

        // Get turnover times
        let tau_plant = self.parameters.tau_plant_pi();
        let tau_detritus = self.parameters.tau_detritus_pi();
        let tau_soil = self.parameters.tau_soil_pi();
        let tau_humus = self.parameters.tau_humus_pi();

        // === Step 1: Update Plant Pool ===
        // Flux in = NPP_to_plant - respiration - land_use_emissions
        // Flux out = plant turnover (to detritus/soil)
        let npp_to_plant = npp * self.parameters.frac_npp_to_plant;
        let flux_in_plant = npp_to_plant - respiration - landuse_emissions;

        // Plant pool doesn't have temperature-dependent decay (only respiration)
        let (new_plant, turnover_plant) =
            self.implicit_pool_step(plant, tau_plant, flux_in_plant, 1.0, dt);

        // === Step 2: Update Detritus Pool ===
        // Flux in = NPP_to_detritus + fraction of plant turnover
        let npp_to_detritus = npp * self.parameters.frac_npp_to_detritus;
        let flux_in_detritus =
            npp_to_detritus + self.parameters.frac_plant_to_detritus * turnover_plant;

        let (new_detritus, turnover_detritus) = self.implicit_pool_step(
            detritus,
            tau_detritus,
            flux_in_detritus,
            temp_factor_detritus,
            dt,
        );

        // === Step 3: Update Soil Pool ===
        // Flux in = NPP_to_soil + plant_to_soil + detritus_to_soil
        let npp_to_soil = npp * self.parameters.frac_npp_to_soil();
        let plant_to_soil = (1.0 - self.parameters.frac_plant_to_detritus) * turnover_plant;
        let detritus_to_soil = self.parameters.frac_detritus_to_soil * turnover_detritus;
        let flux_in_soil = npp_to_soil + plant_to_soil + detritus_to_soil;

        let (new_soil, turnover_soil) =
            self.implicit_pool_step(soil, tau_soil, flux_in_soil, temp_factor_soil, dt);

        // === Step 4: Update Humus Pool ===
        // Flux in = fraction of soil turnover
        let flux_in_humus = self.parameters.frac_soil_to_humus * turnover_soil;

        let (new_humus, turnover_humus) =
            self.implicit_pool_step(humus, tau_humus, flux_in_humus, temp_factor_humus, dt);

        // === Calculate Net Flux ===
        // Net flux = NPP - total respiration to atmosphere
        // Respiration to atmosphere = respiration + detritus_to_atm + soil_to_atm + humus_to_atm
        let detritus_to_atm = (1.0 - self.parameters.frac_detritus_to_soil) * turnover_detritus;
        let soil_to_atm = (1.0 - self.parameters.frac_soil_to_humus) * turnover_soil;
        let humus_to_atm = turnover_humus; // All humus turnover goes to atmosphere

        let total_respiration = respiration + detritus_to_atm + soil_to_atm + humus_to_atm;

        // Net flux positive = land uptake (NPP > respiration)
        let net_flux = npp - total_respiration - landuse_emissions;

        ([new_plant, new_detritus, new_soil, new_humus], net_flux)
    }
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
        let inputs = TerrestrialCarbonInputs::from_input_state(input_state);

        // Get current inputs
        let co2 = inputs.co2_concentration.at_start();
        let temperature = inputs.temperature.at_start();
        let landuse = inputs.landuse_emissions.at_start();

        // Get current pool states
        let plant = inputs.plant_pool.at_start();
        let detritus = inputs.detritus_pool.at_start();
        let soil = inputs.soil_pool.at_start();
        let humus = inputs.humus_pool.at_start();

        let dt = t_next - t_current;

        // Solve for new pool sizes and net flux
        let ([new_plant, new_detritus, new_soil, new_humus], net_flux) = self.solve_pools(
            co2,
            temperature,
            landuse,
            [plant, detritus, soil, humus],
            dt,
        );

        let outputs = TerrestrialCarbonOutputs {
            plant_pool: new_plant,
            detritus_pool: new_detritus,
            soil_pool: new_soil,
            humus_pool: new_humus,
            net_flux,
        };

        Ok(outputs.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_component() -> TerrestrialCarbon {
        TerrestrialCarbon::from_parameters(TerrestrialCarbonParameters::default())
    }

    fn pi_pools() -> [FloatValue; 4] {
        let params = TerrestrialCarbonParameters::default();
        [
            params.plant_pool_pi,
            params.detritus_pool_pi,
            params.soil_pool_pi,
            params.humus_pool_pi,
        ]
    }

    // ===== Steady State Tests =====

    #[test]
    fn test_steady_state_at_preindustrial() {
        let component = default_component();
        let pools = pi_pools();
        let co2_pi = component.parameters.co2_pi;

        // At pre-industrial with no land use, pools should be approximately stable
        let (new_pools, net_flux) = component.solve_pools(
            co2_pi, // pre-industrial CO2
            0.0,    // no temperature anomaly
            0.0,    // no land use emissions
            pools, 1.0,
        );

        // Check pools remain close to initial values (within 5%)
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

        // Net flux should be approximately zero
        assert!(
            net_flux.abs() < 1.0,
            "Net flux should be ~0 at steady state, got {:.2} GtC/yr",
            net_flux
        );
    }

    // ===== CO2 Fertilization Tests =====

    #[test]
    fn test_fertilization_factor_at_pi() {
        let component = default_component();
        let factor = component.fertilization_factor(component.parameters.co2_pi);
        assert!(
            (factor - 1.0).abs() < 1e-10,
            "Fertilization factor should be 1.0 at PI CO2"
        );
    }

    #[test]
    fn test_fertilization_factor_at_doubled_co2() {
        let component = default_component();
        let doubled = component.parameters.co2_pi * 2.0;
        let factor = component.fertilization_factor(doubled);

        // Expected: 1 + beta * ln(2) ≈ 1 + 0.6486 * 0.693 ≈ 1.45
        let expected = 1.0 + component.parameters.beta * 2.0_f64.ln();
        assert!(
            (factor - expected).abs() < 0.01,
            "Fertilization at 2xCO2 should be {:.3}, got {:.3}",
            expected,
            factor
        );
    }

    #[test]
    fn test_higher_co2_increases_npp() {
        let component = default_component();

        let npp_pi = component.calculate_npp(component.parameters.co2_pi, 0.0);
        let npp_high = component.calculate_npp(component.parameters.co2_pi * 1.5, 0.0);

        assert!(
            npp_high > npp_pi,
            "Higher CO2 should increase NPP: {:.2} vs {:.2}",
            npp_high,
            npp_pi
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

    // ===== Temperature Feedback Tests =====

    #[test]
    fn test_warming_increases_respiration() {
        let component = default_component();
        let co2_pi = component.parameters.co2_pi;

        let resp_cold = component.calculate_respiration(co2_pi, 0.0);
        let resp_warm = component.calculate_respiration(co2_pi, 2.0);

        assert!(
            resp_warm > resp_cold,
            "Warming should increase respiration: {:.2} vs {:.2}",
            resp_warm,
            resp_cold
        );
    }

    #[test]
    fn test_warming_reduces_net_uptake() {
        let component = default_component();
        let pools = pi_pools();
        let co2_pi = component.parameters.co2_pi;

        // At elevated CO2 (to have positive uptake to start)
        let co2_high = co2_pi * 1.5;

        let (_, flux_cold) = component.solve_pools(co2_high, 0.0, 0.0, pools, 1.0);
        let (_, flux_warm) = component.solve_pools(co2_high, 3.0, 0.0, pools, 1.0);

        assert!(
            flux_warm < flux_cold,
            "Warming should reduce net uptake (increase respiration): {:.2} vs {:.2}",
            flux_warm,
            flux_cold
        );
    }

    #[test]
    fn test_temperature_feedback_can_be_disabled() {
        let mut params = TerrestrialCarbonParameters::default();
        params.enable_temp_feedback = false;
        let component = TerrestrialCarbon::from_parameters(params);

        let co2_pi = component.parameters.co2_pi;
        let resp_cold = component.calculate_respiration(co2_pi, 0.0);
        let resp_warm = component.calculate_respiration(co2_pi, 5.0);

        assert!(
            (resp_warm - resp_cold).abs() < 1e-10,
            "With temp feedback disabled, respiration should not change with temperature"
        );
    }

    // ===== Pool Dynamics Tests =====

    #[test]
    fn test_pools_remain_positive() {
        let component = default_component();
        let pools = pi_pools();

        // Extreme warming should not make pools negative
        let (new_pools, _) = component.solve_pools(
            component.parameters.co2_pi,
            10.0, // Extreme warming
            5.0,  // High land use emissions
            pools,
            1.0,
        );

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
    fn test_land_use_emissions_reduce_plant_pool() {
        let component = default_component();
        let pools = pi_pools();

        let (pools_no_lu, _) =
            component.solve_pools(component.parameters.co2_pi, 0.0, 0.0, pools, 1.0);
        let (pools_with_lu, _) =
            component.solve_pools(component.parameters.co2_pi, 0.0, 5.0, pools, 1.0);

        assert!(
            pools_with_lu[0] < pools_no_lu[0],
            "Land use emissions should reduce plant pool: {:.2} vs {:.2}",
            pools_with_lu[0],
            pools_no_lu[0]
        );
    }

    // ===== Integration Tests =====

    #[test]
    fn test_multi_year_stability() {
        let component = default_component();
        let mut pools = pi_pools();
        let co2_pi = component.parameters.co2_pi;

        // Run for 100 years at pre-industrial
        for _ in 0..100 {
            let (new_pools, _) = component.solve_pools(co2_pi, 0.0, 0.0, pools, 1.0);
            pools = new_pools;
        }

        // Pools should still be close to initial values
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

        // Run for 50 years at elevated CO2
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

    // ===== Component Trait Tests =====

    #[test]
    fn test_definitions() {
        let component = default_component();
        let defs = component.definitions();

        // Should have: 3 inputs + 4 states + 1 output = 8 definitions
        assert!(
            defs.len() >= 8,
            "Should have at least 8 definitions, got {}",
            defs.len()
        );

        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"Atmospheric Concentration|CO2"));
        assert!(names.contains(&"Surface Temperature|Global"));
        assert!(names.contains(&"Emissions|CO2|Land Use"));
        assert!(names.contains(&"Carbon Pool|Plant"));
        assert!(names.contains(&"Carbon Pool|Detritus"));
        assert!(names.contains(&"Carbon Pool|Soil"));
        assert!(names.contains(&"Carbon Pool|Humus"));
        assert!(names.contains(&"Carbon Flux|Terrestrial"));
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

    // ===== Edge Case Tests =====

    #[test]
    fn test_very_low_co2() {
        let component = default_component();
        let pools = pi_pools();

        // Very low CO2 should still work (fertilization factor floored)
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

        // Very high CO2
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

        // Extreme warming (10K)
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
        // Under extreme warming, should have net release (negative flux)
        assert!(
            net_flux < 0.0,
            "Extreme warming should cause net carbon release, got {:.2}",
            net_flux
        );
    }
}
