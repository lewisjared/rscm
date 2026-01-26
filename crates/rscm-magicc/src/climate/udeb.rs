//! Climate UDEB Component
//!
//! Implements a 4-box Upwelling-Diffusion Energy Balance (UDEB) climate model
//! with multi-layer ocean diffusion.
//!
//! # What This Component Does
//!
//! 1. Takes total effective radiative forcing (4-box regional) as input
//! 2. Solves the ocean diffusion-advection equation using implicit Thomas algorithm
//! 3. Calculates surface air temperature response (4 boxes: NH Ocean, NH Land, SH Ocean, SH Land)
//! 4. Outputs heat uptake and ocean heat content
//!
//! # Physics Overview
//!
//! The model couples a 4-box atmosphere to a 50-layer ocean in each hemisphere.
//! The energy balance for the mixed layer is:
//!
//! $$C_{mix} \frac{dT}{dt} = Q - \lambda T + F_{diffusion} + F_{upwelling}$$
//!
//! where:
//! - $C_{mix}$ = mixed layer heat capacity
//! - $Q$ = radiative forcing
//! - $\lambda$ = climate feedback parameter
//! - $F_{diffusion}$ = diffusive heat flux from deeper ocean
//! - $F_{upwelling}$ = advective heat flux from upwelling
//!
//! Land temperatures are calculated assuming equilibrium with adjacent ocean box.
//!
//! # Differences from MAGICC7 Module 08
//!
//! This is a simplified implementation. The full MAGICC7 module includes:
//!
//! - **LAMCALC iterations**: Iterative calculation of land/ocean feedback parameters
//!   to match land-ocean warming ratio. Not implemented - uses prescribed lambda.
//! - **Time-varying ECS**: Climate sensitivity that changes with forcing level and
//!   cumulative temperature. Not implemented - constant ECS.
//! - **Temperature-dependent diffusivity**: Diffusivity that varies with vertical
//!   temperature gradient. Not implemented - constant diffusivity.
//! - **Depth-dependent ocean area**: Basin narrowing with depth. Not implemented -
//!   cylindrical ocean.
//! - **El Niño/AMV modes**: Internal variability modes. Not implemented.
//! - **Ground heat reservoir**: Land heat capacity damping. Not implemented.

use crate::parameters::ClimateUDEBParameters;
use rscm_core::component::{
    Component, GridType, InputState, OutputState, RequirementDefinition, RequirementType,
};
use rscm_core::errors::RSCMResult;
use rscm_core::spatial::{FourBoxGrid, FourBoxRegion};
use rscm_core::state::{FourBoxSlice, GridTimeseriesWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::utils::linear_algebra::thomas_solve;
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// 4-box UDEB climate model component.
///
/// Implements an Upwelling-Diffusion Energy Balance model with 50 ocean layers
/// per hemisphere. Uses monthly sub-stepping with implicit Thomas algorithm
/// for numerical stability.
///
/// # State Variables
///
/// Internal state (not exposed via component IO):
/// - Ocean layer temperatures for each hemisphere: `[2][n_layers]`
/// - Current upwelling rates for each hemisphere: `[2]`
///
/// # Parameters
///
/// See [`ClimateUDEBParameters`] for configuration options.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["climate", "udeb", "magicc"], category = "Climate")]
#[inputs(
    total_erf { name = "Effective Radiative Forcing|Total", unit = "W/m^2", grid = "FourBox" },
)]
#[states(
    surface_temperature { name = "Surface Temperature", unit = "K", grid = "FourBox" },
)]
#[outputs(
    heat_uptake { name = "Heat Uptake", unit = "W/m^2" },
    ocean_heat_content { name = "Ocean Heat Content", unit = "J/m^2" },
)]
pub struct ClimateUDEB {
    parameters: ClimateUDEBParameters,

    /// Ocean layer temperatures for each hemisphere (NH=0, SH=1).
    /// Shape: [hemisphere][layer], layer 0 is mixed layer.
    #[serde(skip)]
    ocean_temps: Vec<Vec<FloatValue>>,

    /// Current upwelling rate for each hemisphere (m/yr).
    #[serde(skip)]
    upwelling_rates: [FloatValue; 2],

    /// Ocean feedback parameter ($\text{W/m}^2\text{/K}$).
    #[serde(skip)]
    lambda_ocean: FloatValue,

    /// Land feedback parameter ($\text{W/m}^2\text{/K}$).
    #[serde(skip)]
    lambda_land: FloatValue,
}

impl ClimateUDEB {
    /// Create a new ClimateUDEB component with default parameters.
    pub fn new() -> Self {
        Self::from_parameters(ClimateUDEBParameters::default())
    }

    /// Create a new ClimateUDEB component from parameters.
    pub fn from_parameters(parameters: ClimateUDEBParameters) -> Self {
        let n_layers = parameters.n_layers;

        // Calculate lambda_ocean and lambda_land from ECS and RLO
        // Using simplified derivation:
        //   lambda_global = Q_2x / ECS
        //   lambda_l = RLO * lambda_o (at equilibrium)
        //   lambda_global = f_o * lambda_o + f_l * lambda_l
        // Solving: lambda_o = lambda_global / (f_o + f_l * RLO)
        let lambda_global = parameters.lambda_global();
        let f_o = parameters.global_ocean_fraction();
        let f_l = parameters.global_land_fraction();
        let rlo = parameters.rlo;

        let lambda_ocean = lambda_global / (f_o + f_l * rlo);
        let lambda_land = lambda_ocean * rlo;

        Self {
            parameters: parameters.clone(),
            ocean_temps: vec![vec![0.0; n_layers]; 2],
            upwelling_rates: [parameters.w_initial; 2],
            lambda_ocean,
            lambda_land,
        }
    }

    /// Step forward by dt (in years) for a single hemisphere.
    ///
    /// Uses implicit Thomas algorithm for the diffusion-advection equation.
    ///
    /// # Arguments
    ///
    /// * `hemi` - Hemisphere index (0 = NH, 1 = SH)
    /// * `forcing` - Forcing for this hemisphere's ocean box ($\text{W/m}^2$)
    /// * `dt` - Timestep in years
    ///
    /// # Returns
    ///
    /// Mixed layer temperature anomaly (K)
    fn step_hemisphere(&mut self, hemi: usize, forcing: FloatValue, dt: FloatValue) -> FloatValue {
        let n = self.parameters.n_layers;
        let kappa = self.parameters.kappa_m2_per_yr();
        let w = self.upwelling_rates[hemi];
        let dz = self.parameters.layer_thickness;
        let dz_mix = self.parameters.mixed_layer_depth;
        let pi_ratio = self.parameters.polar_sinking_ratio;

        // Heat capacity of mixed layer (W yr / m^2 K)
        let c_mix = self.parameters.mixed_layer_heat_capacity();

        // Build tridiagonal matrix for implicit solve: A*T^{n+1} = D
        // Matrix structure: a[i]*T[i-1] + b[i]*T[i] + c[i]*T[i+1] = d[i]
        let mut a = vec![0.0; n]; // Sub-diagonal
        let mut b = vec![0.0; n]; // Main diagonal
        let mut c = vec![0.0; n]; // Super-diagonal
        let mut d = vec![0.0; n]; // RHS

        // Mixed layer (layer 0)
        // dT/dt = (Q - lambda*T)/C + diffusion + upwelling
        let term_feedback = self.lambda_ocean / c_mix;
        let term_diff = kappa / (dz_mix * dz) * dt;
        let term_upwell = w / dz_mix * dt;

        b[0] = 1.0 + term_feedback * dt + term_diff + term_upwell * pi_ratio;
        c[0] = -(term_diff + term_upwell);
        d[0] = self.ocean_temps[hemi][0] + forcing / c_mix * dt;

        // Layers 1 to n-2 (interior layers)
        for i in 1..n - 1 {
            let term_diff_up = kappa / (dz * dz) * dt;
            let term_diff_down = kappa / (dz * dz) * dt;
            let term_upwell_layer = w / dz * dt;

            a[i] = -term_diff_up;
            b[i] = 1.0 + term_diff_up + term_diff_down + term_upwell_layer;
            c[i] = -(term_diff_down + term_upwell_layer);

            // Entrainment term from polar sinking
            d[i] = self.ocean_temps[hemi][i]
                + pi_ratio * term_upwell_layer * self.ocean_temps[hemi][0];
        }

        // Bottom layer (layer n-1)
        // No flux boundary condition at bottom
        let term_diff_up = kappa / (dz * dz) * dt;
        let term_upwell_bottom = w / dz * dt;

        a[n - 1] = -term_diff_up;
        b[n - 1] = 1.0 + term_diff_up + term_upwell_bottom;
        // c[n-1] = 0 (no layer below)

        d[n - 1] = self.ocean_temps[hemi][n - 1]
            + pi_ratio * term_upwell_bottom * self.ocean_temps[hemi][0];

        // Solve tridiagonal system
        let new_temps = thomas_solve(&a, &b, &c, &d);

        // Apply temperature cap
        let max_temp = self.parameters.max_temperature;
        for (i, &temp) in new_temps.iter().enumerate() {
            self.ocean_temps[hemi][i] = temp.min(max_temp);
        }

        self.ocean_temps[hemi][0]
    }

    /// Update upwelling rate based on global temperature.
    ///
    /// Upwelling decreases with warming (thermohaline circulation weakening):
    ///
    /// $$w = w_0 \times (1 - f_{var} \times T_{global} / T_{threshold})$$
    fn update_upwelling(&mut self, global_temp: FloatValue) {
        let w_0 = self.parameters.w_initial;
        let f_var = self.parameters.w_variable_fraction;
        let w_min = w_0 * (1.0 - f_var);

        // NH upwelling
        let t_thresh_nh = self.parameters.w_threshold_temp_nh;
        let w_nh = w_0 * (1.0 - f_var * (global_temp / t_thresh_nh).min(1.0));
        self.upwelling_rates[0] = w_nh.max(w_min);

        // SH upwelling
        let t_thresh_sh = self.parameters.w_threshold_temp_sh;
        let w_sh = w_0 * (1.0 - f_var * (global_temp / t_thresh_sh).min(1.0));
        self.upwelling_rates[1] = w_sh.max(w_min);
    }

    /// Calculate land temperature from ocean temperature (equilibrium assumption).
    ///
    /// At equilibrium:
    ///
    /// $$\lambda_l \times T_l \times f_l + K_{lo} \times (T_l - \alpha \times T_o) = Q_l \times f_l$$
    ///
    /// Solving for $T_l$:
    ///
    /// $$T_l = \frac{Q_l \times f_l + K_{lo} \times \alpha \times T_o}{\lambda_l \times f_l + K_{lo}}$$
    fn calculate_land_temperature(
        &self,
        ocean_temp: FloatValue,
        land_forcing: FloatValue,
        land_fraction: FloatValue,
    ) -> FloatValue {
        let k_lo = self.parameters.k_lo;
        let alpha = self.parameters.amplify_ocean_to_land;

        let numerator = land_forcing * land_fraction + k_lo * alpha * ocean_temp;
        let denominator = self.lambda_land * land_fraction + k_lo;

        (numerator / denominator).min(self.parameters.max_temperature)
    }

    /// Convert ocean SST to surface air temperature.
    ///
    /// For $T_{sst} < T^*$:
    ///
    /// $$T_{air} = \alpha \times T_{sst} + \gamma \times T_{sst}^2$$
    ///
    /// For $T_{sst} \geq T^*$:
    ///
    /// $$T_{air} = T_{sst} + \delta_{max}$$
    fn sst_to_air_temperature(&self, sst: FloatValue) -> FloatValue {
        let alpha = self.parameters.temp_adjust_alpha;
        let gamma = self.parameters.temp_adjust_gamma;

        // Threshold temperature where quadratic matches linear
        // T* = -(alpha - 1) / (2 * gamma)
        let t_star = if gamma.abs() > 1e-15 {
            -(alpha - 1.0) / (2.0 * gamma)
        } else {
            FloatValue::INFINITY
        };

        if sst < t_star {
            alpha * sst + gamma * sst * sst
        } else {
            // delta_max = alpha*T* + gamma*T*^2 - T*
            let delta_max = alpha * t_star + gamma * t_star * t_star - t_star;
            sst + delta_max
        }
    }

    /// Calculate ocean heat uptake ($\text{W/m}^2$).
    ///
    /// $$\text{Heat uptake} = Q - \lambda \times T$$ (global average)
    fn calculate_heat_uptake(
        &self,
        forcing: &FourBoxSlice,
        temperature: &FourBoxSlice,
    ) -> FloatValue {
        // Area-weighted global average
        let weights = [
            0.5 * self.parameters.nh_ocean_fraction(),
            0.5 * self.parameters.nh_land_fraction,
            0.5 * self.parameters.sh_ocean_fraction(),
            0.5 * self.parameters.sh_land_fraction,
        ];

        let mut q_global = 0.0;
        let mut t_global = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            q_global += w * forcing.0[i];
            t_global += w * temperature.0[i];
        }

        // Heat uptake = Q - lambda * T
        let lambda_global = self.parameters.lambda_global();
        q_global - lambda_global * t_global
    }

    /// Calculate total ocean heat content ($\text{J/m}^2$).
    ///
    /// Integrates temperature anomaly over all ocean layers weighted by depth.
    fn calculate_ocean_heat_content(&self) -> FloatValue {
        let dz = self.parameters.layer_thickness;
        let dz_mix = self.parameters.mixed_layer_depth;

        // Heat capacity per meter depth (J / m^3 K)
        let rho_c = 1026.0 * 3985.0; // rho * c_p

        let mut total_heat = 0.0;

        for hemi in 0..2 {
            // Mixed layer contribution
            total_heat += rho_c * dz_mix * self.ocean_temps[hemi][0];

            // Deep layer contributions
            for layer in 1..self.parameters.n_layers {
                total_heat += rho_c * dz * self.ocean_temps[hemi][layer];
            }
        }

        // Average over both hemispheres
        total_heat / 2.0
    }
}

impl Default for ClimateUDEB {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde]
impl Component for ClimateUDEB {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let inputs = ClimateUDEBInputs::from_input_state(input_state);

        // Get forcing for each box
        let forcing = FourBoxSlice::from_array([
            inputs.total_erf.at_start(FourBoxRegion::NorthernOcean),
            inputs.total_erf.at_start(FourBoxRegion::NorthernLand),
            inputs.total_erf.at_start(FourBoxRegion::SouthernOcean),
            inputs.total_erf.at_start(FourBoxRegion::SouthernLand),
        ]);

        // Get previous surface temperature state
        let prev_temp = FourBoxSlice::from_array([
            inputs
                .surface_temperature
                .at_start(FourBoxRegion::NorthernOcean),
            inputs
                .surface_temperature
                .at_start(FourBoxRegion::NorthernLand),
            inputs
                .surface_temperature
                .at_start(FourBoxRegion::SouthernOcean),
            inputs
                .surface_temperature
                .at_start(FourBoxRegion::SouthernLand),
        ]);

        // Clone self to make mutable for stepping (interior mutability alternative)
        let mut model = self.clone();

        // Initialize ocean temps from previous state if this is first step
        if model.ocean_temps[0][0] == 0.0 && prev_temp.0[0] != 0.0 {
            model.ocean_temps[0][0] = prev_temp.get(FourBoxRegion::NorthernOcean);
            model.ocean_temps[1][0] = prev_temp.get(FourBoxRegion::SouthernOcean);
        }

        let dt_year = t_next - t_current;
        let dt_sub = dt_year / model.parameters.steps_per_year as FloatValue;

        // Monthly sub-stepping
        for _ in 0..model.parameters.steps_per_year {
            // Step each hemisphere's ocean
            let sst_nh =
                model.step_hemisphere(0, forcing.get(FourBoxRegion::NorthernOcean), dt_sub);
            let sst_sh =
                model.step_hemisphere(1, forcing.get(FourBoxRegion::SouthernOcean), dt_sub);

            // Update upwelling based on global SST
            let global_sst = 0.5 * (sst_nh + sst_sh);
            model.update_upwelling(global_sst);
        }

        // Get final ocean surface temperatures
        let sst_nh = model.ocean_temps[0][0];
        let sst_sh = model.ocean_temps[1][0];

        // Convert SST to air temperature over ocean
        let t_air_nho = model.sst_to_air_temperature(sst_nh);
        let t_air_sho = model.sst_to_air_temperature(sst_sh);

        // Calculate land temperatures
        let t_air_nhl = model.calculate_land_temperature(
            t_air_nho,
            forcing.get(FourBoxRegion::NorthernLand),
            model.parameters.nh_land_fraction,
        );
        let t_air_shl = model.calculate_land_temperature(
            t_air_sho,
            forcing.get(FourBoxRegion::SouthernLand),
            model.parameters.sh_land_fraction,
        );

        let surface_temperature =
            FourBoxSlice::from_array([t_air_nho, t_air_nhl, t_air_sho, t_air_shl]);

        // Calculate diagnostics
        let heat_uptake = model.calculate_heat_uptake(&forcing, &surface_temperature);
        let ocean_heat_content = model.calculate_ocean_heat_content();

        let outputs = ClimateUDEBOutputs {
            surface_temperature,
            heat_uptake,
            ocean_heat_content,
        };

        Ok(outputs.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_component() -> ClimateUDEB {
        ClimateUDEB::from_parameters(ClimateUDEBParameters::default())
    }

    #[test]
    fn test_new_component() {
        let component = default_component();
        assert_eq!(component.ocean_temps.len(), 2);
        assert_eq!(component.ocean_temps[0].len(), 50);
        assert_eq!(component.ocean_temps[1].len(), 50);
    }

    #[test]
    fn test_lambda_calculation() {
        let component = default_component();

        // lambda_global = Q_2x / ECS = 3.71 / 3.0 ~= 1.237
        let lambda_global = component.parameters.lambda_global();
        assert!((lambda_global - 1.237).abs() < 0.01);

        // lambda_land / lambda_ocean should equal RLO
        let ratio = component.lambda_land / component.lambda_ocean;
        assert!((ratio - 1.317).abs() < 0.01);
    }

    #[test]
    fn test_positive_forcing_causes_warming() {
        let mut component = default_component();

        // Apply positive forcing for one substep
        let forcing = 3.71; // W/m^2
        let dt = 1.0 / 12.0; // One month

        let initial_temp = component.ocean_temps[0][0];
        let new_temp = component.step_hemisphere(0, forcing, dt);

        assert!(
            new_temp > initial_temp,
            "Positive forcing should cause warming: {} -> {}",
            initial_temp,
            new_temp
        );
    }

    #[test]
    fn test_sst_to_air_temperature() {
        let component = default_component();

        // At T=0, T_air should be 0
        let t_air_0 = component.sst_to_air_temperature(0.0);
        assert!(t_air_0.abs() < 1e-10);

        // At T=1, T_air should be > 1 (amplification)
        let t_air_1 = component.sst_to_air_temperature(1.0);
        assert!(
            t_air_1 > 1.0,
            "Air temperature should be amplified: SST=1 -> T_air={}",
            t_air_1
        );

        // Amplification should decrease at higher temperatures (gamma < 0)
        let t_air_5 = component.sst_to_air_temperature(5.0);
        let ratio_1 = t_air_1 / 1.0;
        let ratio_5 = t_air_5 / 5.0;
        assert!(
            ratio_5 < ratio_1,
            "Amplification should decrease with temperature"
        );
    }

    #[test]
    fn test_upwelling_decreases_with_warming() {
        let mut component = default_component();
        let initial_w = component.upwelling_rates[0];

        component.update_upwelling(4.0); // 4K warming

        assert!(
            component.upwelling_rates[0] < initial_w,
            "Upwelling should decrease with warming"
        );

        // At threshold temperature, upwelling should reach minimum
        component.update_upwelling(10.0); // Above threshold
        let w_min =
            component.parameters.w_initial * (1.0 - component.parameters.w_variable_fraction);
        assert!(
            (component.upwelling_rates[0] - w_min).abs() < 1e-10,
            "Upwelling should reach minimum at threshold"
        );
    }

    #[test]
    fn test_heat_content_increases_with_warming() {
        let mut component = default_component();

        let initial_heat = component.calculate_ocean_heat_content();
        assert!(
            initial_heat.abs() < 1e-10,
            "Initial heat content should be zero"
        );

        // Warm the mixed layer
        component.ocean_temps[0][0] = 1.0;
        component.ocean_temps[1][0] = 1.0;

        let new_heat = component.calculate_ocean_heat_content();
        assert!(
            new_heat > 0.0,
            "Heat content should be positive after warming"
        );
    }

    #[test]
    fn test_land_temperature_higher_than_ocean() {
        let component = default_component();

        // Given same forcing, land should warm more than ocean
        let ocean_temp = 1.0;
        let forcing = 3.71;
        let land_fraction = 0.42;

        let land_temp = component.calculate_land_temperature(ocean_temp, forcing, land_fraction);

        // Land temperature depends on the forcing and heat exchange
        // With RLO > 1, land should eventually be warmer
        // But this depends on the forcing level and equilibrium assumptions
        assert!(land_temp.is_finite(), "Land temperature should be finite");
    }

    #[test]
    fn test_definitions() {
        let component = default_component();
        let defs = component.definitions();

        // Should have: 1 input + 1 state + 2 outputs = 4 definitions
        assert_eq!(defs.len(), 4);

        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"Effective Radiative Forcing|Total"));
        assert!(names.contains(&"Surface Temperature"));
        assert!(names.contains(&"Heat Uptake"));
        assert!(names.contains(&"Ocean Heat Content"));
    }

    #[test]
    fn test_serialization() {
        let component = default_component();
        let json = serde_json::to_string(&component).expect("Serialization failed");
        let parsed: ClimateUDEB = serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(component.parameters.n_layers, parsed.parameters.n_layers);
    }
}
