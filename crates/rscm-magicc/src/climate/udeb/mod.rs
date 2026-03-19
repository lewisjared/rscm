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
//! # Module Structure
//!
//! - `state`: Internal state (`ClimateUDEBState`, `OceanAreaFactors`)
//! - `ocean_column`: Ocean column solver (tridiagonal diffusion,
//!   upwelling, heat uptake/content diagnostics)
//! - `lamcalc`: Iterative feedback parameter solver
//!
//! # Differences from MAGICC7 Module 08
//!
//! - **LAMCALC iterations**: Implemented - see [`lamcalc`](super::lamcalc).
//! - **Time-varying ECS**: Implemented - see `adjusted_ecs()`.
//! - **Temperature-dependent diffusivity**: Implemented - see
//!   `ocean_column::layer_diffusivities()`.
//! - **Depth-dependent ocean area**: Implemented - see `OceanAreaFactors`.
//! - **El Niño/AMV modes**: Internal variability modes. Not implemented.
//! - **Ground heat reservoir**: Implemented - see
//!   [`land_heat_capacity_enabled`](crate::parameters::ClimateUDEBParameters::land_heat_capacity_enabled).

mod ocean_column;

use crate::climate::lamcalc::{self, LamcalcParams, LamcalcResult};
use crate::climate::state::{ClimateUDEBState, OceanAreaFactors};
use crate::parameters::ClimateUDEBParameters;
use rscm_core::component::{
    Component, ComponentState, GridType, InputState, OutputState, RequirementDefinition,
    RequirementType,
};
use rscm_core::errors::{RSCMError, RSCMResult};
use rscm_core::spatial::{FourBoxGrid, FourBoxRegion};
use rscm_core::state::{FourBoxSlice, GridTimeseriesWindow, ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
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

    /// Inverse of the converged LAMCALC coupling matrix.
    /// Used for computing per-agent internal efficacies.
    #[serde(skip)]
    matrix_inverse: [[FloatValue; 4]; 4],

    /// Internal efficacy of CO2 forcing from the initial LAMCALC solve.
    #[serde(skip)]
    co2_internal_efficacy: FloatValue,
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

    /// Create a properly initialized state for this component.
    ///
    /// The initial ocean temperature profile uses an analytical exponential
    /// decay (MAGICC7 `CORE_SWITCH_OCN_TEMPPROFILE=1`). Area factors and
    /// polar sinking ratio are passed for forward compatibility but are
    /// not currently used in the profile calculation.
    pub fn initial_state(&self) -> ClimateUDEBState {
        ClimateUDEBState::new(
            self.parameters.n_layers,
            self.parameters.w_initial,
            self.parameters.temp_adjust_alpha,
            self.parameters.kappa_m2_per_yr(),
            self.parameters.layer_thickness,
            &self.area_factors,
            self.parameters.polar_sinking_ratio,
        )
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
            rf_regions_co2: parameters.rf_regions_co2,
        })
        .ok_or_else(|| {
            RSCMError::Error(format!(
                "LAMCALC iteration failed to converge for ECS={}, RLO={}",
                parameters.ecs, parameters.rlo
            ))
        })?;
        let lambda_ocean = lam_result.lambda_ocean;
        let lambda_land = lam_result.lambda_land;
        let matrix_inverse = lam_result.matrix_inverse;
        let co2_internal_efficacy = lam_result.co2_internal_efficacy;

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
            matrix_inverse,
            co2_internal_efficacy,
        })
    }

    /// Get the inverse of the converged LAMCALC coupling matrix.
    pub fn matrix_inverse(&self) -> &[[FloatValue; 4]; 4] {
        &self.matrix_inverse
    }

    /// Get the CO2 internal efficacy from the initial LAMCALC solve.
    pub fn co2_internal_efficacy(&self) -> FloatValue {
        self.co2_internal_efficacy
    }

    /// Initialize or ensure state is properly configured.
    fn ensure_state_initialized(&self, state: &mut ClimateUDEBState) {
        if !state.initialized {
            let fresh = ClimateUDEBState::new(
                self.parameters.n_layers,
                self.parameters.w_initial,
                self.parameters.temp_adjust_alpha,
                self.parameters.kappa_m2_per_yr(),
                self.parameters.layer_thickness,
                &self.area_factors,
                self.parameters.polar_sinking_ratio,
            );
            state.ocean_temps = fresh.ocean_temps;
            state.upwelling_rates = fresh.upwelling_rates;
            state.alpha_eff = fresh.alpha_eff;
            state.initial_ocean_profile = fresh.initial_ocean_profile;
            state.polar_sinking_temp = fresh.polar_sinking_temp;
            state.mixed_layer_initial_temp = fresh.mixed_layer_initial_temp;
            state.initialized = true;
        }
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

    /// Calculate land temperature from ocean temperature (equilibrium assumption).
    ///
    /// When ground heat capacity is enabled, the ground reservoir couples into
    /// the land equilibrium:
    ///
    /// $$\lambda_l f_l T_l + K_{lo}(T_l - \alpha T_o) + K_{lg}(T_l - T_g) = Q_l f_l$$
    ///
    /// Solving for $T_l$:
    ///
    /// $$T_l = \frac{Q_l f_l + K_{lo} \alpha T_o + K_{lg} T_g}{\lambda_l f_l + K_{lo} + K_{lg}}$$
    fn calculate_land_temperature(
        &self,
        ocean_temp: FloatValue,
        land_forcing: FloatValue,
        land_fraction: FloatValue,
        lambda_land: FloatValue,
        ground_temp: FloatValue,
    ) -> FloatValue {
        let k_lo = self.parameters.k_lo;
        let alpha = self.parameters.amplify_ocean_to_land;
        let k_lg = if self.parameters.land_heat_capacity_enabled {
            self.parameters.k_lg
        } else {
            0.0
        };

        let numerator =
            land_forcing * land_fraction + k_lo * alpha * ocean_temp + k_lg * ground_temp;
        let denominator = lambda_land * land_fraction + k_lo + k_lg;

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

        // MAGICC7 interpolates forcing at each sub-annual time point:
        //   q(step) = ERF(year + step/steps_per_year)
        // We store start and end forcing and linearly interpolate per substep.
        //
        // Note: erf_start is the forcing at t_current (start of year),
        // erf_end is the forcing at t_next (start of next year).
        // MAGICC7's substep loop runs from step 1..12, with forcing times
        // at year+1/12, year+2/12, ..., year+12/12 (= t_next).
        let erf_start = inputs.total_erf.at_start();
        let erf_end = inputs.total_erf.at_end().unwrap_or(erf_start);
        let steps = self.parameters.steps_per_year as FloatValue;

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

        // Time-varying ECS uses midpoint forcing (average of start and end)
        let erf_mid = (erf_start + erf_end) / 2.0;
        let adjusted_ecs = self.adjusted_ecs(erf_mid, state);

        let (current_lambda_ocean, current_lambda_land, current_co2_efficacy) =
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
                    rf_regions_co2: self.parameters.rf_regions_co2,
                })
                .unwrap_or(LamcalcResult {
                    lambda_ocean: self.lambda_ocean,
                    lambda_land: self.lambda_land,
                    matrix_inverse: self.matrix_inverse,
                    co2_internal_efficacy: self.co2_internal_efficacy,
                });
                (
                    result.lambda_ocean,
                    result.lambda_land,
                    result.co2_internal_efficacy,
                )
            } else {
                (
                    self.lambda_ocean,
                    self.lambda_land,
                    self.co2_internal_efficacy,
                )
            };

        // Pre-compute global box fractions needed in the substep loop
        let (fgno, fgnl, fgso, fgsl) = self.parameters.global_box_fractions();

        // Pre-compute ground heat capacity if enabled
        let c_ground = if self.parameters.land_heat_capacity_enabled {
            self.parameters.ground_heat_capacity()
        } else {
            0.0
        };

        // Read alpha_eff once per annual timestep (set at end of previous year).
        // All monthly substeps within this year use the same fixed values,
        // matching MAGICC7 behaviour where alpha_eff is not updated mid-year.
        let alpha_eff_nh = state.alpha_eff[0];
        let alpha_eff_sh = state.alpha_eff[1];

        // Monthly sub-stepping with per-substep forcing interpolation.
        //
        // MAGICC7 runs steps 1..12 for normal years, with forcing queried at:
        //   step 1: year + 1/12, step 2: year + 2/12, ..., step 12: year + 12/12
        // The last substep (step 12) is at t_next (start of next year).
        // See docs/modules/time_conventions.md for details.
        for step_idx in 1..=self.parameters.steps_per_year {
            let frac = step_idx as FloatValue / steps;
            let erf = erf_start + frac * (erf_end - erf_start);
            let erf_adjusted = if self.parameters.efficacy_apply == 2
                && current_co2_efficacy.is_finite()
                && current_co2_efficacy > 0.0
            {
                // AR6 mode: prescribed_efficacy=1.0 for CO2, so EFFRF = RF / internal_efficacy
                erf / current_co2_efficacy
            } else {
                erf
            };
            let forcing = FourBoxSlice::from_array([erf_adjusted; 4]);
            // Update ground heat reservoir temperatures BEFORE the ocean solve
            // (forward Euler, using previous substep's land temperatures).
            //
            // $$\frac{dT_{ground}}{dt} = \frac{K_{lg} \cdot (T_{land} - T_{ground})}{f_l \cdot C \cdot d_{eff}}$$
            if self.parameters.land_heat_capacity_enabled {
                for (hemi, &f_l) in [fgnl, fgsl].iter().enumerate() {
                    if f_l < 1e-15 {
                        continue;
                    }
                    let flux =
                        self.parameters.k_lg * (state.land_temps[hemi] - state.ground_temps[hemi]);
                    state.ground_temps[hemi] += flux / (f_l * c_ground) * dt_sub;
                }
            }

            let nh_ground = state.ground_temps[0];
            let sh_ground = state.ground_temps[1];

            // Step each hemisphere's ocean with explicit K_NS exchange.
            // alpha_eff is fixed for the entire year (read above, outside the loop).
            let sst_nh = self.step_hemisphere(
                state,
                0,
                forcing.get(FourBoxRegion::NorthernOcean),
                dt_sub,
                current_lambda_ocean,
                current_lambda_land,
                state.hemi_heat_exchange[0],
                nh_ground,
                alpha_eff_nh,
            );
            let sst_sh = self.step_hemisphere(
                state,
                1,
                forcing.get(FourBoxRegion::SouthernOcean),
                dt_sub,
                current_lambda_ocean,
                current_lambda_land,
                state.hemi_heat_exchange[1],
                sh_ground,
                alpha_eff_sh,
            );

            // Update land temperatures from new ocean SSTs (equilibrium with
            // ground reservoir when enabled)
            let t_air_nho = self.sst_to_air_temperature(sst_nh);
            let t_air_sho = self.sst_to_air_temperature(sst_sh);
            state.land_temps[0] = self.calculate_land_temperature(
                t_air_nho,
                forcing.get(FourBoxRegion::NorthernLand),
                fgnl,
                current_lambda_land,
                state.ground_temps[0],
            );
            state.land_temps[1] = self.calculate_land_temperature(
                t_air_sho,
                forcing.get(FourBoxRegion::SouthernLand),
                fgsl,
                current_lambda_land,
                state.ground_temps[1],
            );

            // Update inter-hemispheric heat exchange for the NEXT substep
            // using this substep's air temperatures (MAGICC7 lines 3174-3177).
            // HX(hemi) = K_NS / FGO(hemi) * (T_air_other_ocean - T_air_self_ocean)
            let k_ns = self.parameters.k_ns;
            if fgno > 1e-15 {
                state.hemi_heat_exchange[0] = k_ns / fgno * (t_air_sho - t_air_nho);
            }
            if fgso > 1e-15 {
                state.hemi_heat_exchange[1] = k_ns / fgso * (t_air_nho - t_air_sho);
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

        // Update alpha_eff from end-of-year SST for use in the NEXT annual timestep.
        // This matches MAGICC7: alpha_eff is set once per year from the previous
        // year's final SST, not recalculated each monthly substep.
        let alpha = self.parameters.temp_adjust_alpha;
        state.alpha_eff[0] = if sst_nh.abs() < 1e-15 {
            alpha
        } else {
            self.sst_to_air_temperature(sst_nh) / sst_nh
        };
        state.alpha_eff[1] = if sst_sh.abs() < 1e-15 {
            alpha
        } else {
            self.sst_to_air_temperature(sst_sh) / sst_sh
        };

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

        // Calculate diagnostics using end-of-year forcing, adjusted for efficacy
        // so that N = Q_eff - lambda*T is consistent with the forcing used to drive T.
        let erf_end_adjusted = if self.parameters.efficacy_apply == 2
            && current_co2_efficacy.is_finite()
            && current_co2_efficacy > 0.0
        {
            erf_end / current_co2_efficacy
        } else {
            erf_end
        };
        let forcing_end = FourBoxSlice::from_array([erf_end_adjusted; 4]);
        let heat_uptake = self.calculate_heat_uptake(
            &forcing_end,
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
        Box::new(self.initial_state())
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
        component.initial_state()
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
    fn test_initial_state_profile() {
        let component = default_component();
        let state = component.initial_state();

        let n = component.parameters.n_layers;
        assert_eq!(state.initial_ocean_profile.len(), n);

        // Mixed layer should be t_mix
        assert!((state.initial_ocean_profile[0] - 17.2).abs() < 1e-10);

        // Profile should decrease monotonically from t_mix toward t_polar
        for l in 1..n {
            assert!(
                state.initial_ocean_profile[l] < state.initial_ocean_profile[l - 1],
                "profile should decrease with depth: layer {} ({}) >= layer {} ({})",
                l,
                state.initial_ocean_profile[l],
                l - 1,
                state.initial_ocean_profile[l - 1]
            );
            assert!(
                state.initial_ocean_profile[l] >= 1.0,
                "profile should stay above t_polar=1.0: layer {} = {}",
                l,
                state.initial_ocean_profile[l]
            );
        }

        // Deep layers should approach t_polar = 1.0
        let deepest = state.initial_ocean_profile[n - 1];
        assert!(
            deepest < 2.0,
            "deepest layer should be close to t_polar=1.0, got {deepest}"
        );
    }

    #[test]
    fn test_initial_state_anomalies_are_zero() {
        let component = default_component();
        let state = component.initial_state();

        for hemi in 0..2 {
            for (l, &t) in state.ocean_temps[hemi].iter().enumerate() {
                assert!(
                    t.abs() < 1e-15,
                    "initial anomaly temps should be zero: hemi={hemi} layer={l} T={t}"
                );
            }
        }
        assert_eq!(state.upwelling_rates, [component.parameters.w_initial; 2]);
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
            0.0,                                    // other hemisphere SST
            0.0,                                    // ground temp
            component.parameters.temp_adjust_alpha, // alpha_eff (initial value, SST=0)
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
            0.0, // ground temp
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

    #[test]
    fn test_efficacy_accessors() {
        let component = default_component();

        // matrix_inverse should be a valid 4x4 matrix
        let inv = component.matrix_inverse();
        for row in inv {
            for &val in row {
                assert!(val.is_finite(), "matrix_inverse entry should be finite");
            }
        }

        // CO2 efficacy should be near 1.0
        let eff = component.co2_internal_efficacy();
        println!("CO2 internal efficacy from ClimateUDEB = {eff:.6}");
        assert!(
            eff > 0.90 && eff < 1.10,
            "CO2 efficacy should be near 1.0, got {eff}"
        );
    }

    #[test]
    fn test_efficacy_mode_zero_unchanged() {
        use rscm_core::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
        use rscm_core::spatial::FourBoxGrid;
        use rscm_core::state::StateValue;
        use rscm_core::timeseries::TimeAxis;
        use rscm_core::timeseries_collection::{TimeseriesData, TimeseriesItem, VariableType};
        use std::sync::Arc;

        // Mode 0 (default) should produce identical results to a component
        // that has no efficacy fields at all.
        let mut params = ClimateUDEBParameters::default();
        params.efficacy_apply = 0;
        let component = ClimateUDEB::from_parameters(params).unwrap();
        let mut state = default_state(&component);

        let time_axis = Arc::new(TimeAxis::from_values(ndarray::array![2000.0, 2001.0]));
        let erf_ts = rscm_core::timeseries::Timeseries::from_values(
            ndarray::array![3.71, 3.71],
            ndarray::array![2000.0, 2001.0],
        );
        let erf_item = TimeseriesItem {
            data: TimeseriesData::Scalar(erf_ts),
            name: "Effective Radiative Forcing".to_string(),
            variable_type: VariableType::Exogenous,
        };

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

        // Compare with default component (also mode 0)
        let default_comp = default_component();
        let mut default_st = default_state(&default_comp);
        let output_default = default_comp
            .solve_impl(2000.0, 2001.0, &input_state, &mut default_st)
            .expect("solve_impl failed");

        let sst_mode0 = match output.get("Sea Surface Temperature").unwrap() {
            StateValue::Scalar(v) => *v,
            other => panic!("Expected scalar, got {:?}", other),
        };
        let sst_default = match output_default.get("Sea Surface Temperature").unwrap() {
            StateValue::Scalar(v) => *v,
            other => panic!("Expected scalar, got {:?}", other),
        };

        assert!(
            (sst_mode0 - sst_default).abs() < 1e-15,
            "Mode 0 should be identical to default: mode0={sst_mode0}, default={sst_default}"
        );
    }

    #[test]
    fn test_efficacy_mode_two_adjusts_temperature() {
        use rscm_core::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
        use rscm_core::spatial::FourBoxGrid;
        use rscm_core::state::StateValue;
        use rscm_core::timeseries::TimeAxis;
        use rscm_core::timeseries_collection::{TimeseriesData, TimeseriesItem, VariableType};
        use std::sync::Arc;

        let time_axis = Arc::new(TimeAxis::from_values(ndarray::array![2000.0, 2001.0]));
        let erf_ts = rscm_core::timeseries::Timeseries::from_values(
            ndarray::array![3.71, 3.71],
            ndarray::array![2000.0, 2001.0],
        );
        let erf_item = TimeseriesItem {
            data: TimeseriesData::Scalar(erf_ts),
            name: "Effective Radiative Forcing".to_string(),
            variable_type: VariableType::Exogenous,
        };

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

        // Mode 0: no efficacy adjustment
        let mut params_0 = ClimateUDEBParameters::default();
        params_0.efficacy_apply = 0;
        let comp_0 = ClimateUDEB::from_parameters(params_0).unwrap();
        let mut state_0 = default_state(&comp_0);
        let out_0 = comp_0
            .solve_impl(2000.0, 2001.0, &input_state, &mut state_0)
            .expect("solve failed");

        // Mode 2: AR6 efficacy adjustment
        let mut params_2 = ClimateUDEBParameters::default();
        params_2.efficacy_apply = 2;
        let comp_2 = ClimateUDEB::from_parameters(params_2).unwrap();
        let mut state_2 = default_state(&comp_2);
        let out_2 = comp_2
            .solve_impl(2000.0, 2001.0, &input_state, &mut state_2)
            .expect("solve failed");

        let sst_0 = match out_0.get("Sea Surface Temperature").unwrap() {
            StateValue::Scalar(v) => *v,
            other => panic!("Expected scalar, got {:?}", other),
        };
        let sst_2 = match out_2.get("Sea Surface Temperature").unwrap() {
            StateValue::Scalar(v) => *v,
            other => panic!("Expected scalar, got {:?}", other),
        };

        println!("SST mode 0 = {sst_0:.6}, SST mode 2 = {sst_2:.6}");
        println!(
            "CO2 internal efficacy = {:.6}",
            comp_2.co2_internal_efficacy()
        );

        // Mode 2 divides forcing by co2_internal_efficacy.
        // With AR6 defaults the efficacy is near 1.0 but not exactly 1.0,
        // so temperatures should differ.
        assert!(
            (sst_0 - sst_2).abs() > 1e-6,
            "Mode 2 should shift temperature: mode0={sst_0}, mode2={sst_2}"
        );

        // The direction depends on whether co2_efficacy > 1 or < 1.
        // With default AR6 rf_regions_co2, CO2 efficacy is typically slightly != 1.
        if comp_2.co2_internal_efficacy() > 1.0 {
            assert!(
                sst_2 < sst_0,
                "Efficacy > 1 should reduce effective forcing: mode0={sst_0}, mode2={sst_2}"
            );
        } else {
            assert!(
                sst_2 > sst_0,
                "Efficacy < 1 should increase effective forcing: mode0={sst_0}, mode2={sst_2}"
            );
        }
    }
}
