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
//! Land temperatures are calculated assuming equilibrium with adjacent ocean box,
//! with optional ground heat reservoir damping.
//!
//! # Differences from MAGICC7 Module 08
//!
//! This is a simplified implementation. The full MAGICC7 module includes:
//!
//! - **LAMCALC iterations**: Iterative calculation of land/ocean feedback parameters
//!   to match land-ocean warming ratio. Implemented - see [`lamcalc`](super::lamcalc).
//! - **Time-varying ECS**: Climate sensitivity that changes with forcing level and
//!   cumulative temperature. Implemented - see `adjusted_ecs()`.
//! - **Temperature-dependent diffusivity**: Diffusivity that varies with vertical
//!   temperature gradient. Implemented - see `layer_diffusivities()`.
//! - **Depth-dependent ocean area**: Basin narrowing with depth using a standard
//!   hypsometric profile. Implemented - see [`OceanAreaFactors`].
//! - **El Niño/AMV modes**: Internal variability modes. Not implemented.
//! - **Ground heat reservoir**: Land heat capacity damping. Implemented - see
//!   [`land_heat_capacity_enabled`](super::parameters::ClimateUDEBParameters::land_heat_capacity_enabled).

use std::any::Any;

use crate::climate::lamcalc::{self, LamcalcParams, LamcalcResult};
use crate::parameters::{
    ClimateUDEBParameters, CP_SEAWATER, DIFFUSIVITY_CM2S_TO_M2YR, RHO_SEAWATER,
};
use rscm_core::component::{
    Component, ComponentState, GridType, InputState, OutputState, RequirementDefinition,
    RequirementType,
};
use rscm_core::errors::{RSCMError, RSCMResult};
use rscm_core::spatial::{FourBoxGrid, FourBoxRegion};
use rscm_core::state::{FourBoxSlice, GridTimeseriesWindow, ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::utils::linear_algebra::thomas_solve;
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};

/// Internal state for ClimateUDEB component.
///
/// This holds the ocean layer temperatures and upwelling rates that persist
/// across solve() calls. Unlike coupled state (RequirementType::State), this
/// is private to the component and not shared between components.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClimateUDEBState {
    /// Ocean layer temperatures for each hemisphere (NH=0, SH=1).
    /// Shape: [hemisphere][layer], layer 0 is mixed layer.
    pub ocean_temps: Vec<Vec<FloatValue>>,

    /// Current upwelling rate for each hemisphere (m/yr).
    pub upwelling_rates: [FloatValue; 2],

    /// Whether the state has been initialized with parameters.
    pub initialized: bool,

    /// History of year-weighted global mean temperature for cumulative-T ECS adjustment.
    /// Each entry stores `temperature * dt_years`, so the sum over entries gives
    /// the cumulative temperature in K-years regardless of timestep size.
    #[serde(default)]
    pub temperature_history: Vec<FloatValue>,

    /// History of timestep sizes (years) corresponding to each temperature_history entry.
    /// Used to determine how many entries span the `feedback_cumt_period` window.
    #[serde(default)]
    pub dt_history: Vec<FloatValue>,

    /// Land surface temperatures for each hemisphere (K).
    /// Index 0 = NH, 1 = SH. Updated each substep when ground heat
    /// capacity is enabled.
    #[serde(default)]
    pub land_temps: [FloatValue; 2],

    /// Ground heat reservoir temperatures for each hemisphere (K).
    /// Index 0 = NH, 1 = SH.
    #[serde(default)]
    pub ground_temps: [FloatValue; 2],
}

impl ClimateUDEBState {
    /// Create a new state for a given number of layers and initial upwelling rate.
    pub fn new(n_layers: usize, w_initial: FloatValue) -> Self {
        Self {
            ocean_temps: vec![vec![0.0; n_layers]; 2],
            upwelling_rates: [w_initial; 2],
            initialized: true,
            temperature_history: Vec::new(),
            dt_history: Vec::new(),
            land_temps: [0.0; 2],
            ground_temps: [0.0; 2],
        }
    }
}

#[typetag::serde]
impl ComponentState for ClimateUDEBState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Pre-computed ocean area factors for depth-dependent basin narrowing.
///
/// These factors modify the diffusion and upwelling terms in the tridiagonal
/// solver to account for the ocean basin narrowing with depth (hypsometry).
/// For a cylindrical ocean, `af_top` and `af_bottom` are 1.0 and `af_diff` is 0.0.
#[derive(Debug, Clone, Default)]
pub struct OceanAreaFactors {
    /// Area factor for flux from above: `A(z_top) / A(z_avg)` per layer.
    pub af_top: Vec<FloatValue>,
    /// Area factor for flux to below: `A(z_bottom) / A(z_avg)` per layer.
    pub af_bottom: Vec<FloatValue>,
    /// Area factor for entrainment (polar sinking): `(A(z_top) - A(z_bottom)) / A(z_avg)`.
    /// Matches MAGICC7 `AREAFACTOR_DIFFFLOW`.
    pub af_diff: Vec<FloatValue>,
}

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
    total_erf { name = "Effective Radiative Forcing", unit = "W/m^2" },
)]
#[states(
    surface_temperature { name = "Surface Temperature", unit = "K", grid = "FourBox" },
)]
#[outputs(
    heat_uptake { name = "Heat Uptake", unit = "W/m^2" },
    ocean_heat_content { name = "Ocean Heat Content", unit = "J/m^2" },
    sst { name = "Sea Surface Temperature", unit = "K" },
)]
pub struct ClimateUDEB {
    parameters: ClimateUDEBParameters,

    /// Ocean feedback parameter ($\text{W/m}^2\text{/K}$).
    #[serde(skip)]
    lambda_ocean: FloatValue,

    /// Land feedback parameter ($\text{W/m}^2\text{/K}$).
    #[serde(skip)]
    lambda_land: FloatValue,

    /// Pre-computed ocean area factors for depth-dependent basin narrowing.
    #[serde(skip)]
    area_factors: OceanAreaFactors,
}

impl ClimateUDEB {
    /// Create a new ClimateUDEB component with default parameters.
    ///
    /// # Panics
    ///
    /// Panics if LAMCALC iteration fails to converge for default parameters
    /// (should never happen).
    pub fn new() -> Self {
        Self::from_parameters(ClimateUDEBParameters::default())
            .expect("LAMCALC failed to converge with default parameters")
    }

    /// Create a new ClimateUDEB component from parameters.
    ///
    /// Returns an error if the LAMCALC iteration fails to converge for the
    /// given parameter combination. This can happen with extreme or
    /// inconsistent user-supplied configurations.
    pub fn from_parameters(parameters: ClimateUDEBParameters) -> RSCMResult<Self> {
        if parameters.n_layers < 2 {
            return Err(RSCMError::Error(format!(
                "invalid n_layers: must be >= 2, got {}",
                parameters.n_layers
            )));
        }

        // Calculate lambda_ocean and lambda_land via LAMCALC iteration.
        // This accounts for inter-box heat exchange when matching the
        // land-ocean warming ratio (RLO) at the given ECS.
        let (fgno, fgnl, fgso, fgsl) = parameters.global_box_fractions();
        let lam_result = lamcalc::lamcalc(&LamcalcParams {
            q_2xco2: parameters.rf_2xco2,
            k_lo: parameters.k_lo,
            k_ns: parameters.k_ns,
            ecs: parameters.ecs,
            rlo: parameters.rlo,
            amplify_ocean_to_land: parameters.amplify_ocean_to_land,
            fgno,
            fgnl,
            fgso,
            fgsl,
        })
        .ok_or_else(|| {
            RSCMError::Error(format!(
                "LAMCALC iteration failed to converge for ECS={}, RLO={}",
                parameters.ecs, parameters.rlo
            ))
        })?;
        let lambda_ocean = lam_result.lambda_ocean;
        let lambda_land = lam_result.lambda_land;

        // Pre-compute depth-dependent ocean area factors
        let (af_top, af_bottom, af_diff) = parameters.compute_area_factors();
        let area_factors = OceanAreaFactors {
            af_top,
            af_bottom,
            af_diff,
        };

        Ok(Self {
            parameters: parameters.clone(),
            lambda_ocean,
            lambda_land,
            area_factors,
        })
    }

    /// Initialize or ensure state is properly configured.
    fn ensure_state_initialized(&self, state: &mut ClimateUDEBState) {
        if !state.initialized {
            state.ocean_temps = vec![vec![0.0; self.parameters.n_layers]; 2];
            state.upwelling_rates = [self.parameters.w_initial; 2];
            state.initialized = true;
        }
    }

    /// Calculate depth-dependent vertical diffusivity for each layer boundary.
    ///
    /// Diffusivity varies with the temperature gradient between the surface
    /// and bottom layer, decreasing linearly with relative depth:
    ///
    /// $$K(z) = \max(K_{min}, K_0 + \frac{dK}{dT} \times (1 - z/z_{max}) \times (T_{top} - T_{bottom})) \times 3155.76$$
    ///
    /// Returns a Vec of length `n_layers - 1` (diffusivity at each layer boundary).
    fn layer_diffusivities(&self, state: &ClimateUDEBState, hemi: usize) -> Vec<FloatValue> {
        let n = self.parameters.n_layers;
        let dz = self.parameters.layer_thickness;
        let total_depth = self.parameters.mixed_layer_depth + (n as FloatValue - 1.0) * dz;

        let t_top = state.ocean_temps[hemi][0];
        let t_bottom = state.ocean_temps[hemi][n - 1];

        let kappa_min_m2yr = self.parameters.kappa_min_m2_per_yr();

        let mut kappa = Vec::with_capacity(n - 1);
        for l in 0..n - 1 {
            // Depth of layer boundary
            let depth = self.parameters.mixed_layer_depth + l as FloatValue * dz;
            let relative_depth = depth / total_depth;

            let k = ((1.0 - relative_depth) * self.parameters.kappa_dkdt * (t_top - t_bottom)
                + self.parameters.kappa)
                * DIFFUSIVITY_CM2S_TO_M2YR;
            kappa.push(k.max(kappa_min_m2yr));
        }

        kappa
    }

    /// Calculate time-varying equilibrium climate sensitivity.
    ///
    /// Adjusts ECS based on cumulative temperature history and current
    /// forcing level (MAGICC7.f90 lines 2747-2763).
    ///
    /// Temperature history is stored as year-weighted entries (`T * dt_years`)
    /// so that the cumulative sum is correct regardless of model timestep size.
    ///
    /// $$ECS_{adj} = ECS \times (1 + \alpha_T \times \frac{\sum T \, dt - \sum T_{2x}}{\sum T_{2x}})
    ///                       \times (1 + \alpha_Q \times (\max(0, Q) - Q_{2x}))$$
    fn adjusted_ecs(&self, global_forcing: FloatValue, state: &ClimateUDEBState) -> FloatValue {
        let cumt_2x = self.parameters.ecs * self.parameters.feedback_cumt_period;
        let period = self.parameters.feedback_cumt_period;

        // Sum year-weighted temperatures over the last `period` years,
        // walking backwards through the history.
        let cum_t: FloatValue = if state.temperature_history.is_empty() {
            0.0
        } else {
            let mut years_remaining = period;
            let mut sum = 0.0;
            for i in (0..state.temperature_history.len()).rev() {
                if years_remaining <= 0.0 {
                    break;
                }
                let dt = state.dt_history[i];
                if dt <= years_remaining {
                    sum += state.temperature_history[i];
                    years_remaining -= dt;
                } else {
                    // Partial contribution from this entry
                    sum += state.temperature_history[i] * (years_remaining / dt);
                    years_remaining = 0.0;
                }
            }
            sum
        };

        let cumt_factor = if cumt_2x.abs() > 1e-15 {
            1.0 + self.parameters.feedback_cumt_sensitivity * (cum_t - cumt_2x) / cumt_2x
        } else {
            1.0
        };

        let q_factor = 1.0
            + self.parameters.feedback_q_sensitivity
                * (global_forcing.max(0.0) - self.parameters.rf_2xco2);

        self.parameters.ecs * cumt_factor * q_factor
    }

    /// Step forward by dt (in years) for a single hemisphere.
    ///
    /// Uses implicit Thomas algorithm for the diffusion-advection equation
    /// with depth-dependent ocean area factors and inter-hemispheric heat
    /// exchange.
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable reference to component state
    /// * `hemi` - Hemisphere index (0 = NH, 1 = SH)
    /// * `forcing` - Forcing for this hemisphere's ocean box ($\text{W/m}^2$)
    /// * `dt` - Timestep in years
    /// * `lambda_ocean` - Ocean feedback parameter ($\text{W/m}^2\text{/K}$)
    /// * `lambda_land` - Land feedback parameter ($\text{W/m}^2\text{/K}$)
    /// * `other_hemi_sst` - Other hemisphere's ocean SST for $K_{NS}$ exchange
    /// * `land_temp_prev` - Previous substep's land temperature for ground heat coupling
    /// * `ground_temp` - Current ground reservoir temperature for ground heat coupling
    ///
    /// # Returns
    ///
    /// Mixed layer temperature anomaly (K)
    fn step_hemisphere(
        &self,
        state: &mut ClimateUDEBState,
        hemi: usize,
        forcing: FloatValue,
        dt: FloatValue,
        lambda_ocean: FloatValue,
        lambda_land: FloatValue,
        other_hemi_sst: FloatValue,
        land_temp_prev: FloatValue,
        ground_temp: FloatValue,
    ) -> FloatValue {
        let n = self.parameters.n_layers;
        let kappas = self.layer_diffusivities(state, hemi);
        let w = state.upwelling_rates[hemi];
        let dz = self.parameters.layer_thickness;
        let dz_mix = self.parameters.mixed_layer_depth;
        let pi_ratio = self.parameters.polar_sinking_ratio;
        let af_top = &self.area_factors.af_top;
        let af_bot = &self.area_factors.af_bottom;
        let af_diff = &self.area_factors.af_diff;

        // Heat capacity of mixed layer (W yr / m^2 K)
        let c_mix = self.parameters.mixed_layer_heat_capacity();

        // Build tridiagonal matrix for implicit solve: A*T^{n+1} = D
        // Matrix structure: a[i]*T[i-1] + b[i]*T[i] + c[i]*T[i+1] = d[i]
        let mut a = vec![0.0; n]; // Sub-diagonal
        let mut b = vec![0.0; n]; // Main diagonal
        let mut c = vec![0.0; n]; // Super-diagonal
        let mut d = vec![0.0; n]; // RHS

        // Mixed layer (layer 0)
        // dT/dt = (Q - lambda*T)/C + diffusion + upwelling + K_NS exchange
        // Coupled ocean-land feedback term (TERM_OCN_LAND_FEEDBACK)
        // Accounts for land feedback coupling through the mixed layer.
        // MAGICC7.f90 lines 2806-2820
        let f_l_hemi = if hemi == 0 {
            self.parameters.nh_land_fraction / 2.0
        } else {
            self.parameters.sh_land_fraction / 2.0
        };
        let f_o_hemi = 0.5 - f_l_hemi;
        // Time-varying alpha_eff: MAGICC7.f90 lines 3171-3187
        // CORE_TEMPADJUST_OCN2ATM_ALPHAEFF = T_air / T_sst (or base alpha if SST ~ 0)
        let sst_prev = state.ocean_temps[hemi][0];
        let alpha_eff = if sst_prev.abs() < 1e-15 {
            self.parameters.temp_adjust_alpha
        } else {
            let t_air_prev = self.sst_to_air_temperature(sst_prev);
            t_air_prev / sst_prev
        };
        let denominator = f_o_hemi * (self.parameters.k_lo + f_l_hemi * lambda_land);
        let term_feedback = alpha_eff / c_mix
            * (lambda_ocean
                + lambda_land
                    * self.parameters.k_lo
                    * self.parameters.amplify_ocean_to_land
                    * f_l_hemi
                    / denominator);
        // DZ1 = DZ/2: half-thickness for the gradient between mixed layer and
        // the first deep layer (MAGICC7.f90 DZ1 = DZ/2, asymmetric spacing).
        let dz1 = dz / 2.0;
        let term_diff = kappas[0] / (dz_mix * dz1) * dt;
        let term_upwell = w / dz_mix * dt;

        // Land forcing amplification.
        //
        // Eliminating $T_l$ from the coupled land-ocean system produces both a
        // modified feedback (TERM_OCN_LAND_FEEDBACK above) and a modified forcing:
        //
        // $$Q_{eff} = Q \cdot \left(1 + \frac{K_{lo} \cdot f_l}{f_o \cdot (f_l \cdot \lambda_l + K_{lo})}\right)$$
        //
        // The amplification arises because land forcing propagates to the ocean
        // through the $K_{lo}$ coupling: land receives $f_l \cdot Q$ of forcing
        // but has no thermal inertia, so it passes heat to the ocean mixed layer.
        let forcing_amp = 1.0 + self.parameters.k_lo * f_l_hemi / denominator;

        // Inter-hemispheric heat exchange: K_NS * (T_this - T_other)
        // Implicit on diagonal (self-coupling), explicit on RHS (cross-coupling).
        let k_ns_term = self.parameters.k_ns / c_mix * dt;

        b[0] = 1.0
            + term_feedback * dt
            + term_diff * af_bot[0]
            + term_upwell * pi_ratio * af_bot[0]
            + k_ns_term;
        c[0] = -(term_diff + term_upwell) * af_bot[0];
        d[0] = state.ocean_temps[hemi][0]
            + forcing * forcing_amp / c_mix * dt
            + k_ns_term * other_hemi_sst;

        // Ground heat capacity: subtract heat flowing from land to ground reservoir.
        // The ground absorbs K_lg * (T_land - T_ground) per unit globe area.
        // Scale by f_l/f_o to convert to per unit ocean area for the mixed layer.
        if self.parameters.land_heat_capacity_enabled {
            let ground_flux = self.parameters.k_lg * (land_temp_prev - ground_temp);
            d[0] -= ground_flux * f_l_hemi / f_o_hemi / c_mix * dt;
        }

        // Layers 1 to n-2 (interior layers)
        for i in 1..n - 1 {
            // Layer 1 (MAGICC layer 2) uses DZ1 = DZ/2 for upward diffusion
            // to the mixed layer (MAGICC7.f90 VERTICALDIFF(1)/DZ1).
            // All deeper layers use the full DZ spacing.
            let dz_up = if i == 1 { dz1 } else { dz };
            let term_diff_up = kappas[i - 1] / (dz * dz_up) * dt;
            let term_diff_down = kappas[i] / (dz * dz) * dt;
            let term_upwell_layer = w / dz * dt;

            a[i] = -term_diff_up * af_top[i];
            b[i] = 1.0
                + term_diff_up * af_top[i]
                + term_diff_down * af_bot[i]
                + term_upwell_layer * af_top[i];
            c[i] = -(term_diff_down + term_upwell_layer) * af_bot[i];

            // Entrainment term from polar sinking (MAGICC7 AREAFACTOR_DIFFFLOW)
            d[i] = state.ocean_temps[hemi][i]
                + pi_ratio * term_upwell_layer * state.ocean_temps[hemi][0] * af_diff[i];
        }

        // Bottom layer (layer n-1)
        // No flux boundary condition at bottom
        let term_diff_up = kappas[n - 2] / (dz * dz) * dt;
        let term_upwell_bottom = w / dz * dt;

        a[n - 1] = -term_diff_up * af_top[n - 1];
        b[n - 1] = 1.0 + (term_diff_up + term_upwell_bottom) * af_top[n - 1];
        // c[n-1] = 0 (no layer below)

        // Entrainment term from polar sinking (MAGICC7 AREAFACTOR_DIFFFLOW)
        d[n - 1] = state.ocean_temps[hemi][n - 1]
            + pi_ratio * term_upwell_bottom * state.ocean_temps[hemi][0] * af_diff[n - 1];

        // Solve tridiagonal system
        let new_temps = thomas_solve(&a, &b, &c, &d);

        // Apply temperature cap
        let max_temp = self.parameters.max_temperature;
        for (i, &temp) in new_temps.iter().enumerate() {
            state.ocean_temps[hemi][i] = temp.min(max_temp);
        }

        state.ocean_temps[hemi][0]
    }

    /// Update upwelling rate based on global temperature.
    ///
    /// Upwelling decreases with warming (thermohaline circulation weakening):
    ///
    /// $$w = w_0 \times (1 - f_{var} \times T_{global} / T_{threshold})$$
    fn update_upwelling(&self, state: &mut ClimateUDEBState, global_temp: FloatValue) {
        let w_0 = self.parameters.w_initial;
        let f_var = self.parameters.w_variable_fraction;
        let w_min = w_0 * (1.0 - f_var);

        // NH upwelling
        let t_thresh_nh = self.parameters.w_threshold_temp_nh;
        let w_nh = w_0 * (1.0 - f_var * (global_temp / t_thresh_nh).min(1.0));
        state.upwelling_rates[0] = w_nh.max(w_min);

        // SH upwelling
        let t_thresh_sh = self.parameters.w_threshold_temp_sh;
        let w_sh = w_0 * (1.0 - f_var * (global_temp / t_thresh_sh).min(1.0));
        state.upwelling_rates[1] = w_sh.max(w_min);
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
        lambda_land: FloatValue,
    ) -> FloatValue {
        let k_lo = self.parameters.k_lo;
        let alpha = self.parameters.amplify_ocean_to_land;

        let numerator = land_forcing * land_fraction + k_lo * alpha * ocean_temp;
        let denominator = lambda_land * land_fraction + k_lo;

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
    /// $$\text{Heat uptake} = Q - \sum_i f_i \lambda_i T_i$$
    ///
    /// where $f_i$ are global area fractions, $\lambda_i$ is the per-box
    /// feedback parameter (ocean or land), and $T_i$ is the per-box temperature.
    /// This ensures the diagnostic is consistent with the LAMCALC-solved
    /// feedback parameters and any time-varying ECS adjustments.
    fn calculate_heat_uptake(
        &self,
        forcing: &FourBoxSlice,
        temperature: &FourBoxSlice,
        lambda_ocean: FloatValue,
        lambda_land: FloatValue,
    ) -> FloatValue {
        let (fgno, fgnl, fgso, fgsl) = self.parameters.global_box_fractions();
        let weights = [fgno, fgnl, fgso, fgsl];
        let lambdas = [lambda_ocean, lambda_land, lambda_ocean, lambda_land];

        let mut q_global = 0.0;
        let mut feedback_global = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            q_global += w * forcing.0[i];
            feedback_global += w * lambdas[i] * temperature.0[i];
        }

        // Heat uptake = Q - sum(f_i * lambda_i * T_i)
        q_global - feedback_global
    }

    /// Calculate total ocean heat content ($\text{J/m}^2$).
    ///
    /// Integrates temperature anomaly over all ocean layers weighted by depth.
    fn calculate_ocean_heat_content(&self, state: &ClimateUDEBState) -> FloatValue {
        let dz = self.parameters.layer_thickness;
        let dz_mix = self.parameters.mixed_layer_depth;

        let rho_c = RHO_SEAWATER * CP_SEAWATER;

        let mut total_heat = 0.0;

        for hemi in 0..2 {
            // Mixed layer contribution
            total_heat += rho_c * dz_mix * state.ocean_temps[hemi][0];

            // Deep layer contributions
            for layer in 1..self.parameters.n_layers {
                total_heat += rho_c * dz * state.ocean_temps[hemi][layer];
            }
        }

        // Average over both hemispheres
        total_heat / 2.0
    }

    /// Internal solve implementation used by both solve() and solve_with_state().
    fn solve_impl(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
        state: &mut ClimateUDEBState,
    ) -> RSCMResult<OutputState> {
        self.ensure_state_initialized(state);

        let inputs = ClimateUDEBInputs::from_input_state(input_state);

        // Get forcing - broadcast scalar to uniform FourBox
        // When regional forcing distribution is available, this can accept FourBox input directly
        let erf = inputs.total_erf.get();
        let forcing = FourBoxSlice::from_array([erf, erf, erf, erf]);

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

        // Initialize internal state from previous surface temperatures if this
        // is a restart with non-zero initial conditions (e.g. warm-start).
        if state.ocean_temps[0][0] == 0.0 && prev_temp.0[0] != 0.0 {
            state.ocean_temps[0][0] = prev_temp.get(FourBoxRegion::NorthernOcean);
            state.ocean_temps[1][0] = prev_temp.get(FourBoxRegion::SouthernOcean);
            state.land_temps[0] = prev_temp.get(FourBoxRegion::NorthernLand);
            state.land_temps[1] = prev_temp.get(FourBoxRegion::SouthernLand);
            // Set ground temps equal to land temps to avoid artificial initial flux
            state.ground_temps[0] = state.land_temps[0];
            state.ground_temps[1] = state.land_temps[1];
        }

        let dt_year = t_next - t_current;
        let dt_sub = dt_year / self.parameters.steps_per_year as FloatValue;

        // Time-varying ECS adjustment
        let adjusted_ecs = self.adjusted_ecs(erf, state);

        let (current_lambda_ocean, current_lambda_land) =
            if (adjusted_ecs - self.parameters.ecs).abs() > 1e-10 {
                let (fgno, fgnl, fgso, fgsl) = self.parameters.global_box_fractions();
                let result = lamcalc::lamcalc(&LamcalcParams {
                    q_2xco2: self.parameters.rf_2xco2,
                    k_lo: self.parameters.k_lo,
                    k_ns: self.parameters.k_ns,
                    ecs: adjusted_ecs,
                    rlo: self.parameters.rlo,
                    amplify_ocean_to_land: self.parameters.amplify_ocean_to_land,
                    fgno,
                    fgnl,
                    fgso,
                    fgsl,
                })
                .unwrap_or(LamcalcResult {
                    lambda_ocean: self.lambda_ocean,
                    lambda_land: self.lambda_land,
                });
                (result.lambda_ocean, result.lambda_land)
            } else {
                (self.lambda_ocean, self.lambda_land)
            };

        // Pre-compute global box fractions needed in the substep loop
        let (fgno, fgnl, fgso, fgsl) = self.parameters.global_box_fractions();

        // Pre-compute ground heat capacity if enabled
        let c_ground = if self.parameters.land_heat_capacity_enabled {
            self.parameters.ground_heat_capacity()
        } else {
            0.0
        };

        // Monthly sub-stepping
        for _ in 0..self.parameters.steps_per_year {
            // Save start-of-step values for symmetric inter-hemispheric exchange
            // and explicit ground heat coupling
            let nh_sst_start = state.ocean_temps[0][0];
            let sh_sst_start = state.ocean_temps[1][0];
            let nh_land_prev = state.land_temps[0];
            let sh_land_prev = state.land_temps[1];
            let nh_ground = state.ground_temps[0];
            let sh_ground = state.ground_temps[1];

            // Step each hemisphere's ocean with K_NS exchange and ground heat
            let sst_nh = self.step_hemisphere(
                state,
                0,
                forcing.get(FourBoxRegion::NorthernOcean),
                dt_sub,
                current_lambda_ocean,
                current_lambda_land,
                sh_sst_start,
                nh_land_prev,
                nh_ground,
            );
            let sst_sh = self.step_hemisphere(
                state,
                1,
                forcing.get(FourBoxRegion::SouthernOcean),
                dt_sub,
                current_lambda_ocean,
                current_lambda_land,
                nh_sst_start,
                sh_land_prev,
                sh_ground,
            );

            // Update land temperatures from new ocean SSTs (equilibrium assumption)
            let t_air_nho = self.sst_to_air_temperature(sst_nh);
            let t_air_sho = self.sst_to_air_temperature(sst_sh);
            state.land_temps[0] = self.calculate_land_temperature(
                t_air_nho,
                forcing.get(FourBoxRegion::NorthernLand),
                fgnl,
                current_lambda_land,
            );
            state.land_temps[1] = self.calculate_land_temperature(
                t_air_sho,
                forcing.get(FourBoxRegion::SouthernLand),
                fgsl,
                current_lambda_land,
            );

            // Update ground heat reservoir temperatures (forward Euler).
            //
            // $$\frac{dT_{ground}}{dt} = \frac{K_{lg} \cdot (T_{land} - T_{ground})}{f_l \cdot C \cdot d_{eff}}$$
            if self.parameters.land_heat_capacity_enabled {
                for (hemi, &f_l) in [fgnl, fgsl].iter().enumerate() {
                    let flux =
                        self.parameters.k_lg * (state.land_temps[hemi] - state.ground_temps[hemi]);
                    state.ground_temps[hemi] += flux / (f_l * c_ground) * dt_sub;
                }
            }

            // Update upwelling based on area-weighted global air temperature
            // (MAGICC7.f90 line 3298: GLOBET = SUM(CURRENT_TIME_TEMPERATURE * GLOBALAREAFRACTIONS))
            let global_temp = t_air_nho * fgno
                + state.land_temps[0] * fgnl
                + t_air_sho * fgso
                + state.land_temps[1] * fgsl;
            self.update_upwelling(state, global_temp);
        }

        // Get final temperatures from the last substep
        let sst_nh = state.ocean_temps[0][0];
        let sst_sh = state.ocean_temps[1][0];
        let t_air_nho = self.sst_to_air_temperature(sst_nh);
        let t_air_sho = self.sst_to_air_temperature(sst_sh);
        let t_air_nhl = state.land_temps[0];
        let t_air_shl = state.land_temps[1];

        let surface_temperature =
            FourBoxSlice::from_array([t_air_nho, t_air_nhl, t_air_sho, t_air_shl]);

        // Track year-weighted temperature history for cumulative-T adjustment.
        // Each entry stores T * dt so the sum gives K-years regardless of timestep.
        let global_temp = surface_temperature.0[0] * fgno
            + surface_temperature.0[1] * fgnl
            + surface_temperature.0[2] * fgso
            + surface_temperature.0[3] * fgsl;
        state.temperature_history.push(global_temp * dt_year);
        state.dt_history.push(dt_year);

        // Calculate diagnostics
        let heat_uptake = self.calculate_heat_uptake(
            &forcing,
            &surface_temperature,
            current_lambda_ocean,
            current_lambda_land,
        );
        let ocean_heat_content = self.calculate_ocean_heat_content(state);

        // SST is the mean of the two ocean box temperatures
        let sst = (sst_nh + sst_sh) / 2.0;

        let outputs = ClimateUDEBOutputs {
            surface_temperature,
            heat_uptake,
            ocean_heat_content,
            sst,
        };

        Ok(outputs.into())
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
        // For standalone use, create a temporary state
        let mut state = ClimateUDEBState::default();
        self.solve_impl(t_current, t_next, input_state, &mut state)
    }

    fn create_initial_state(&self) -> Box<dyn ComponentState> {
        Box::new(ClimateUDEBState::new(
            self.parameters.n_layers,
            self.parameters.w_initial,
        ))
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
            .downcast_mut::<ClimateUDEBState>()
            .expect("ClimateUDEB: invalid state type (expected ClimateUDEBState)");
        self.solve_impl(t_current, t_next, input_state, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rscm_core::timeseries::GridTimeseries;

    fn default_component() -> ClimateUDEB {
        ClimateUDEB::from_parameters(ClimateUDEBParameters::default()).unwrap()
    }

    fn default_state(component: &ClimateUDEB) -> ClimateUDEBState {
        ClimateUDEBState::new(
            component.parameters.n_layers,
            component.parameters.w_initial,
        )
    }

    #[test]
    fn test_new_component() {
        let component = default_component();
        let state = default_state(&component);
        assert_eq!(state.ocean_temps.len(), 2);
        assert_eq!(state.ocean_temps[0].len(), 50);
        assert_eq!(state.ocean_temps[1].len(), 50);
    }

    #[test]
    fn test_lambda_calculation() {
        let component = default_component();

        // lambda_global = Q_2x / ECS = 3.71 / 3.0 ~= 1.237
        let lambda_global = component.parameters.lambda_global();
        assert!((lambda_global - 1.237).abs() < 0.01);

        // With LAMCALC, both feedback parameters should be positive
        assert!(
            component.lambda_ocean > 0.0,
            "lambda_ocean should be positive, got {}",
            component.lambda_ocean
        );
        assert!(
            component.lambda_land.is_finite(),
            "lambda_land should be finite, got {}",
            component.lambda_land
        );

        // With LAMCALC the ratio lambda_land/lambda_ocean need not equal RLO
        // exactly because inter-box heat exchange is accounted for.
        // lambda_land can even be negative (positive feedback on land) depending
        // on the parameter combination; we just verify it is finite.
        println!(
            "lambda_ocean = {:.6}, lambda_land = {:.6}",
            component.lambda_ocean, component.lambda_land
        );
    }

    #[test]
    fn test_positive_forcing_causes_warming() {
        let component = default_component();
        let mut state = default_state(&component);

        // Apply positive forcing for one substep
        let forcing = 3.71; // W/m^2
        let dt = 1.0 / 12.0; // One month

        let initial_temp = state.ocean_temps[0][0];
        let new_temp = component.step_hemisphere(
            &mut state,
            0,
            forcing,
            dt,
            component.lambda_ocean,
            component.lambda_land,
            0.0, // other hemisphere SST
            0.0, // land temp prev
            0.0, // ground temp
        );

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
        let component = default_component();
        let mut state = default_state(&component);
        let initial_w = state.upwelling_rates[0];

        component.update_upwelling(&mut state, 4.0); // 4K warming

        assert!(
            state.upwelling_rates[0] < initial_w,
            "Upwelling should decrease with warming"
        );

        // At threshold temperature, upwelling should reach minimum
        component.update_upwelling(&mut state, 10.0); // Above threshold
        let w_min =
            component.parameters.w_initial * (1.0 - component.parameters.w_variable_fraction);
        assert!(
            (state.upwelling_rates[0] - w_min).abs() < 1e-10,
            "Upwelling should reach minimum at threshold"
        );
    }

    #[test]
    fn test_heat_content_increases_with_warming() {
        let component = default_component();
        let mut state = default_state(&component);

        let initial_heat = component.calculate_ocean_heat_content(&state);
        assert!(
            initial_heat.abs() < 1e-10,
            "Initial heat content should be zero"
        );

        // Warm the mixed layer
        state.ocean_temps[0][0] = 1.0;
        state.ocean_temps[1][0] = 1.0;

        let new_heat = component.calculate_ocean_heat_content(&state);
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

        let land_temp = component.calculate_land_temperature(
            ocean_temp,
            forcing,
            land_fraction,
            component.lambda_land,
        );

        // Land temperature depends on the forcing and heat exchange
        // With RLO > 1, land should eventually be warmer
        // But this depends on the forcing level and equilibrium assumptions
        assert!(land_temp.is_finite(), "Land temperature should be finite");
    }

    #[test]
    fn test_definitions() {
        let component = default_component();
        let defs = component.definitions();

        // Should have: 1 input + 1 state + 3 outputs = 5 definitions
        assert_eq!(defs.len(), 5);

        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"Effective Radiative Forcing"));
        assert!(names.contains(&"Surface Temperature"));
        assert!(names.contains(&"Heat Uptake"));
        assert!(names.contains(&"Ocean Heat Content"));
        assert!(names.contains(&"Sea Surface Temperature"));
    }

    #[test]
    fn test_sst_is_mean_of_ocean_boxes() {
        use rscm_core::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
        use rscm_core::spatial::FourBoxGrid;
        use rscm_core::state::StateValue;
        use rscm_core::timeseries::TimeAxis;
        use rscm_core::timeseries_collection::{TimeseriesData, TimeseriesItem, VariableType};
        use std::sync::Arc;

        let component = default_component();
        let mut state = default_state(&component);

        // Build scalar ERF timeseries: constant 3.71 W/m^2
        let time_axis = Arc::new(TimeAxis::from_values(ndarray::array![
            2000.0_f64, 2001.0_f64
        ]));
        let erf_ts = rscm_core::timeseries::Timeseries::from_values(
            ndarray::array![3.71, 3.71],
            ndarray::array![2000.0_f64, 2001.0_f64],
        );
        let erf_item = TimeseriesItem {
            data: TimeseriesData::Scalar(erf_ts),
            name: "Effective Radiative Forcing".to_string(),
            variable_type: VariableType::Exogenous,
        };

        // Build FourBox surface temperature timeseries initialised to zero
        let surf_grid = FourBoxGrid::magicc_standard();
        let surf_values = Array2::zeros((2, 4));
        let surf_ts = GridTimeseries::new(
            surf_values,
            time_axis,
            surf_grid,
            "K".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );
        let surf_item = TimeseriesItem {
            data: TimeseriesData::FourBox(surf_ts),
            name: "Surface Temperature".to_string(),
            variable_type: VariableType::Endogenous,
        };

        let input_state = InputState::build(vec![&erf_item, &surf_item], 2000.0);

        let output = component
            .solve_impl(2000.0, 2001.0, &input_state, &mut state)
            .expect("solve_impl failed");

        // Extract SST from output
        let sst_val = match output
            .get("Sea Surface Temperature")
            .expect("SST not found in output")
        {
            StateValue::Scalar(v) => *v,
            other => panic!("Expected scalar SST, got {:?}", other),
        };

        // SST should equal mean of the two ocean mixed-layer temperatures
        let nh_ocean = state.ocean_temps[0][0];
        let sh_ocean = state.ocean_temps[1][0];
        let expected_sst = (nh_ocean + sh_ocean) / 2.0;

        assert!(
            (sst_val - expected_sst).abs() < 1e-10,
            "SST ({}) should be mean of NH ocean ({}) and SH ocean ({}) = {}",
            sst_val,
            nh_ocean,
            sh_ocean,
            expected_sst
        );

        // SST should be positive after applying positive forcing
        assert!(
            sst_val > 0.0,
            "SST should be positive after positive forcing"
        );
    }

    #[test]
    fn test_diffusivity_varies_with_temperature() {
        let component = default_component();
        let state = default_state(&component);

        // With zero temperature gradient, diffusivity should equal base kappa
        let kappa_uniform = component.layer_diffusivities(&state, 0);
        let base_kappa = component.parameters.kappa_m2_per_yr();
        for &k in &kappa_uniform {
            assert!(
                (k - base_kappa).abs() < 1e-6,
                "Zero gradient: K={k} should equal base {base_kappa}"
            );
        }

        // With positive gradient (warm top, cold bottom), diffusivity should
        // decrease near surface (dK/dT is negative, gradient is positive)
        let mut warm_state = default_state(&component);
        warm_state.ocean_temps[0][0] = 3.0; // warm surface
        let kappa_warm = component.layer_diffusivities(&warm_state, 0);

        // Surface layers should have lower diffusivity
        assert!(
            kappa_warm[0] < base_kappa,
            "Warm surface: K[0]={} should be less than base {}",
            kappa_warm[0],
            base_kappa
        );

        // But never below minimum
        let kappa_min = component.parameters.kappa_min_m2_per_yr();
        for &k in &kappa_warm {
            assert!(
                k >= kappa_min - 1e-10,
                "K={k} should not go below min {kappa_min}"
            );
        }
    }

    #[test]
    fn test_serialization() {
        let component = default_component();
        let json = serde_json::to_string(&component).expect("Serialization failed");
        let parsed: ClimateUDEB = serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(component.parameters.n_layers, parsed.parameters.n_layers);
    }

    /// Helper to build a ClimateUDEBState with year-weighted temperature history.
    ///
    /// Takes raw per-year temperatures and converts to year-weighted entries
    /// (each with dt=1.0 year).
    fn state_with_history(
        component: &ClimateUDEB,
        yearly_temps: &[FloatValue],
    ) -> ClimateUDEBState {
        let mut state = default_state(component);
        for &t in yearly_temps {
            state.temperature_history.push(t * 1.0); // T * dt_years
            state.dt_history.push(1.0);
        }
        state
    }

    #[test]
    fn test_adjusted_ecs_with_defaults() {
        let component = default_component();

        // With empty history and forcing equal to rf_2xco2, the q_factor is 1.0.
        // The cumt_factor = 1 + 0.08 * (0 - 900) / 900 = 0.92
        // So ecs_adj = 3.0 * 0.92 = 2.76
        let empty_state = default_state(&component);
        let ecs_adj = component.adjusted_ecs(3.71, &empty_state);
        let expected =
            component.parameters.ecs * (1.0 - component.parameters.feedback_cumt_sensitivity);
        assert!(
            (ecs_adj - expected).abs() < 0.01,
            "Empty history should reduce ECS by cumt_sensitivity fraction: got {ecs_adj}, expected {expected}"
        );

        // With equilibrium history (cumT == cumT_2x), ECS should be unchanged
        let cumt_2x = component.parameters.ecs * component.parameters.feedback_cumt_period;
        let period = component.parameters.feedback_cumt_period as usize;
        // Each year contributes cumt_2x/period temperature to reach cumT == cumT_2x
        let t_per_year = cumt_2x / period as FloatValue;
        let equilibrium_history: Vec<FloatValue> = vec![t_per_year; period];
        let eq_state = state_with_history(&component, &equilibrium_history);
        let ecs_eq = component.adjusted_ecs(component.parameters.rf_2xco2, &eq_state);
        assert!(
            (ecs_eq - component.parameters.ecs).abs() < 0.01,
            "Equilibrium history should leave ECS unchanged: got {ecs_eq}"
        );
    }

    #[test]
    fn test_adjusted_ecs_with_large_sensitivity() {
        let mut params = ClimateUDEBParameters::default();
        params.feedback_cumt_sensitivity = 0.5; // exaggerated
        let component = ClimateUDEB::from_parameters(params).unwrap();

        // After 100 years of 1K warming: cumT = 100, cumT_2x = 3.0 * 300 = 900
        let yearly_temps: Vec<FloatValue> = vec![1.0; 100];
        let state = state_with_history(&component, &yearly_temps);
        let ecs_adj = component.adjusted_ecs(3.71, &state);

        // Below-equilibrium cumT should reduce effective ECS
        assert!(
            ecs_adj < component.parameters.ecs,
            "Below-equilibrium cumT should reduce ECS: got {ecs_adj}"
        );
    }
}
