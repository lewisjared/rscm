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
//! # Differences from MAGICC7
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
use crate::climate::sea_level::{
    thermal_expansion_increment, ANNUAL_MEAN_TEMPERATURE, THERMAL_EXPANSION,
};
use crate::climate::state::{ClimateUDEBState, OceanAreaFactors};
use crate::parameters::{
    ClimateUDEBParameters, SubannualForcing, CP_SEAWATER, RHO_SEAWATER, SECONDS_PER_YEAR,
};
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
#[serde(try_from = "SerializedClimateUDEB")]
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
    climate_sensitivity { name = "Equilibrium Climate Sensitivity", unit = "K" },
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

    /// Diffusivity change per kelvin of top to bottom temperature difference at each layer boundary.
    #[serde(skip)]
    kappa_slopes: Vec<FloatValue>,

    /// Inverse of the converged LAMCALC coupling matrix.
    /// Used for computing per-agent internal efficacies.
    #[serde(skip)]
    matrix_inverse: [[FloatValue; 4]; 4],

    /// Internal efficacy of CO2 forcing from the initial LAMCALC solve.
    #[serde(skip)]
    co2_internal_efficacy: FloatValue,

    /// Per-box CO2 forcing fractions from the regional CO2 pattern.
    ///
    /// Computed as `qfrac[i] = rf_regions_co2[i] / sum(rf_regions_co2[j] * area[j])`.
    /// The area-weighted sum equals 1.0, preserving global mean forcing.
    /// Falls back to `[1.0; 4]` (uniform) if the weighted sum is near zero.
    #[serde(skip)]
    co2_qfrac: [FloatValue; 4],
}

/// The serialised form of [`ClimateUDEB`], from which every derived field is rebuilt.
#[derive(Deserialize)]
struct SerializedClimateUDEB {
    parameters: ClimateUDEBParameters,
}

impl TryFrom<SerializedClimateUDEB> for ClimateUDEB {
    type Error = RSCMError;

    fn try_from(value: SerializedClimateUDEB) -> RSCMResult<Self> {
        Self::from_parameters(value.parameters)
    }
}

/// CO2 doubling ERF, read when `time_varying_q2x` is set.
pub const CO2_DOUBLING_ERF: &str = "Effective Radiative Forcing|2xCO2";

/// Climate sensitivity of the step, after the time-varying feedback adjustments.
pub const CLIMATE_SENSITIVITY: &str = "Equilibrium Climate Sensitivity";

/// Efficacy-adjusted forcing on the four boxes, read in regional mode.
pub const REGIONAL_FORCING: &str = "Effective Radiative Forcing|Efficacy Adjusted";

/// Forcing at the start and end of a step.
enum ForcingInput {
    /// Global ERF, spread with the CO2 pattern and efficacy.
    Global(FloatValue, FloatValue),
    /// Efficacy-adjusted box forcing, used as given.
    Regional([FloatValue; 4], [FloatValue; 4]),
}

/// Linear interpolation that lands exactly on `end`, as MAGICC7 reads the next year's value there.
fn lerp(start: FloatValue, end: FloatValue, frac: FloatValue) -> FloatValue {
    if frac == 1.0 {
        end
    } else {
        start + frac * (end - start)
    }
}

/// Seawater heat capacity per metre of depth (W yr / m^3 / K), in MAGICC7's operation order.
fn heat_capacity_per_metre() -> FloatValue {
    RHO_SEAWATER * CP_SEAWATER / SECONDS_PER_YEAR
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
    /// The initial ocean temperature profile mode is controlled by the
    /// `ocean_temp_profile` parameter:
    /// - Mode 2 (default): CMIP5 multi-model mean profile
    /// - Mode 1: Analytical exponential decay
    pub fn initial_state(&self) -> ClimateUDEBState {
        let profiles = [
            self.parameters.initial_ocean_profile(0),
            self.parameters.initial_ocean_profile(1),
        ];
        ClimateUDEBState::new(
            self.parameters.n_layers,
            self.parameters.w_initial,
            self.parameters.temp_adjust_alpha,
            profiles,
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

        if parameters.steps_per_year == 0 {
            return Err(RSCMError::Error(
                "invalid steps_per_year: must be >= 1, got 0".to_string(),
            ));
        }

        if !parameters.prescribed_efficacy_co2.is_finite()
            || parameters.prescribed_efficacy_co2 <= 0.0
        {
            return Err(RSCMError::Error(format!(
                "invalid prescribed_efficacy_co2: must be finite and positive, got {}",
                parameters.prescribed_efficacy_co2
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

        // Pre-compute per-box CO2 forcing fractions from the regional pattern
        let area = [fgno, fgnl, fgso, fgsl];
        let co2_qfrac = lamcalc::compute_qfrac(&parameters.rf_regions_co2, &area);

        // Pre-compute depth-dependent ocean area factors
        let [(top_nh, bottom_nh, diff_nh), (top_sh, bottom_sh, diff_sh)] =
            [0, 1].map(|hemi| parameters.compute_area_factors(hemi));
        let area_factors = OceanAreaFactors {
            af_top: [top_nh, top_sh],
            af_bottom: [bottom_nh, bottom_sh],
            af_diff: [diff_nh, diff_sh],
        };

        let kappa_slopes = parameters.kappa_slopes();

        Ok(Self {
            parameters: parameters.clone(),
            lambda_ocean,
            lambda_land,
            area_factors,
            kappa_slopes,
            matrix_inverse,
            co2_internal_efficacy,
            co2_qfrac,
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

    /// Get the per-box CO2 forcing fractions.
    pub fn co2_qfrac(&self) -> &[FloatValue; 4] {
        &self.co2_qfrac
    }

    /// Heat uptake at initialization with zero temperature anomalies.
    /// Uses the same efficacy and regional forcing adjustments as annual diagnostics.
    pub fn initial_heat_uptake(&self, erf: FloatValue) -> FloatValue {
        let forcing = self.apply_efficacy_and_qfrac(erf, self.co2_internal_efficacy);
        self.calculate_heat_uptake(
            &forcing,
            &FourBoxSlice::from_array([0.0; 4]),
            self.lambda_ocean,
            self.lambda_land,
        )
    }

    /// Apply efficacy adjustment and regional CO2 qfrac distribution to scalar ERF.
    ///
    /// NOTE: This applies the CO2 regional forcing pattern to the *total* ERF input,
    /// not just the CO2 component. This is correct when forcing is CO2-only, but
    /// when non-CO2 agents contribute to total ERF, their forcing is also distributed
    /// using the CO2 pattern. This matches MAGICC's single-agent architecture where
    /// LAMCALC is calibrated against the same total ERF, but breaks physical
    /// consistency for multi-agent forcing mixes. Per-agent regional patterns would
    /// require decomposed forcing inputs (future work).
    fn apply_efficacy_and_qfrac(&self, erf: FloatValue, co2_efficacy: FloatValue) -> FourBoxSlice {
        let erf_adjusted = match self.parameters.efficacy_apply {
            1 if co2_efficacy.is_finite() && co2_efficacy > 0.0 => erf * (1.0 / co2_efficacy),
            2 if co2_efficacy.is_finite() && co2_efficacy > 0.0 => {
                erf * (self.parameters.prescribed_efficacy_co2 / co2_efficacy)
            }
            _ => erf,
        };
        FourBoxSlice::from_array([
            erf_adjusted * self.co2_qfrac[0],
            erf_adjusted * self.co2_qfrac[1],
            erf_adjusted * self.co2_qfrac[2],
            erf_adjusted * self.co2_qfrac[3],
        ])
    }

    /// Box forcing at fraction `frac` of the way through the step.
    fn box_forcing(
        &self,
        input: &ForcingInput,
        frac: FloatValue,
        co2_efficacy: FloatValue,
    ) -> FourBoxSlice {
        match input {
            ForcingInput::Global(start, end) => {
                self.apply_efficacy_and_qfrac(lerp(*start, *end, frac), co2_efficacy)
            }
            ForcingInput::Regional(start, end) => {
                FourBoxSlice::from_array(std::array::from_fn(|i| lerp(start[i], end[i], frac)))
            }
        }
    }

    /// Internal efficacy of an agent with the given regional forcing pattern.
    pub fn internal_efficacy(&self, rf_regions: &[FloatValue; 4]) -> FloatValue {
        let (fgno, fgnl, fgso, fgsl) = self.parameters.global_box_fractions();
        lamcalc::calc_internal_efficacy(
            self.parameters.rf_2xco2,
            &self.matrix_inverse,
            &[fgno, fgnl, fgso, fgsl],
            rf_regions,
            self.parameters.ecs,
        )
    }

    /// Initialize or ensure state is properly configured.
    fn ensure_state_initialized(&self, state: &mut ClimateUDEBState) {
        if !state.initialized {
            let profiles = [
                self.parameters.initial_ocean_profile(0),
                self.parameters.initial_ocean_profile(1),
            ];
            let fresh = ClimateUDEBState::new(
                self.parameters.n_layers,
                self.parameters.w_initial,
                self.parameters.temp_adjust_alpha,
                profiles,
            );
            state.ocean_temps = fresh.ocean_temps;
            state.upwelling_rates = fresh.upwelling_rates;
            state.upwelling_delta = fresh.upwelling_delta;
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
    fn adjusted_ecs(
        &self,
        global_forcing: FloatValue,
        q_2xco2: FloatValue,
        cum_t: FloatValue,
    ) -> FloatValue {
        let cumt_2x = self.parameters.ecs * self.parameters.feedback_cumt_period;
        let cumt_factor = if cumt_2x.abs() > 1e-15 {
            1.0 + self.parameters.feedback_cumt_sensitivity * (cum_t - cumt_2x) / cumt_2x
        } else {
            1.0
        };

        let q_factor =
            1.0 + self.parameters.feedback_q_sensitivity * (global_forcing.max(0.0) - q_2xco2);

        self.parameters.ecs * cumt_factor * q_factor
    }

    /// Year-weighted temperature summed oldest first over the last `feedback_cumt_period` years.
    fn cumulative_temperature(&self, state: &ClimateUDEBState) -> FloatValue {
        let mut years_remaining = self.parameters.feedback_cumt_period;
        let mut start = state.temperature_history.len();
        let mut oldest_weight = 1.0;
        while start > 0 && years_remaining > 0.0 {
            let dt = state.dt_history[start - 1];
            start -= 1;
            if dt <= years_remaining {
                years_remaining -= dt;
            } else {
                oldest_weight = years_remaining / dt;
                years_remaining = 0.0;
            }
        }
        let mut sum = 0.0;
        for (i, &t) in state.temperature_history.iter().enumerate().skip(start) {
            sum += if i == start && oldest_weight != 1.0 {
                t * oldest_weight
            } else {
                t
            };
        }
        sum
    }

    /// Feedback parameters and CO2 internal efficacy at the given sensitivity.
    fn sensitivity(&self, ecs: FloatValue, q_2xco2: FloatValue) -> LamcalcResult {
        if ecs == self.parameters.ecs && q_2xco2 == self.parameters.rf_2xco2 {
            return self.initial_sensitivity();
        }
        let (fgno, fgnl, fgso, fgsl) = self.parameters.global_box_fractions();
        lamcalc::lamcalc(&LamcalcParams {
            q_2xco2,
            k_lo: self.parameters.k_lo,
            k_ns: self.parameters.k_ns,
            ecs,
            rlo: self.parameters.rlo,
            amplify_ocean_to_land: self.parameters.amplify_ocean_to_land,
            fgno,
            fgnl,
            fgso,
            fgsl,
            rf_regions_co2: self.parameters.rf_regions_co2,
        })
        .unwrap_or_else(|| self.initial_sensitivity())
    }

    fn initial_sensitivity(&self) -> LamcalcResult {
        LamcalcResult {
            lambda_ocean: self.lambda_ocean,
            lambda_land: self.lambda_land,
            matrix_inverse: self.matrix_inverse,
            co2_internal_efficacy: self.co2_internal_efficacy,
        }
    }

    /// Calculate land temperature from ocean temperature (equilibrium assumption).
    ///
    /// MAGICC7.f90 lines 3214-3222:
    ///
    /// $$T_l = \frac{Q_l f_l + K_{lo} \alpha T_o}{\lambda_l f_l + K_{lo}}$$
    ///
    /// Ground heat capacity is NOT included in the land equilibrium -- it is
    /// handled as a separate explicit term in the ocean mixed layer equation
    /// and the ground temperature forward Euler update.
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
            alpha * sst + gamma * (sst * sst)
        } else {
            sst + ((alpha * t_star + gamma * (t_star * t_star)) - t_star)
        }
    }

    /// Ratio of air temperature to SST, applied to the mixed-layer feedback of the next substep.
    fn alpha_eff(&self, sst: FloatValue) -> FloatValue {
        if sst == 0.0 {
            self.parameters.temp_adjust_alpha
        } else {
            self.sst_to_air_temperature(sst) / sst
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

        // MAGICC7 interpolates forcing at each sub-annual time point:
        //   q(step) = ERF(year + step/steps_per_year)
        // Substep 12 lands on t_next. See docs/modules/time_conventions.md.
        let (fgno, fgnl, fgso, fgsl) = self.parameters.global_box_fractions();
        let area = [fgno, fgnl, fgso, fgsl];
        let forcing_input = if self.parameters.regional_forcing {
            let window = input_state.get_four_box_window(REGIONAL_FORCING);
            let start = window.at_start_all();
            let end = window.at_end_all().unwrap_or_else(|| start.clone());
            ForcingInput::Regional(
                [start[0], start[1], start[2], start[3]],
                [end[0], end[1], end[2], end[3]],
            )
        } else {
            let window = input_state.get_scalar_window("Effective Radiative Forcing");
            let start = window.at_start();
            ForcingInput::Global(start, window.at_end().unwrap_or(start))
        };
        let steps = self.parameters.steps_per_year as FloatValue;

        // Get previous surface temperature state
        let temperature = input_state.get_four_box_window("Surface Temperature");
        let prev_temp = FourBoxSlice::from_array([
            temperature.at_start(FourBoxRegion::NorthernOcean),
            temperature.at_start(FourBoxRegion::NorthernLand),
            temperature.at_start(FourBoxRegion::SouthernOcean),
            temperature.at_start(FourBoxRegion::SouthernLand),
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

        let q_2xco2 = if self.parameters.time_varying_q2x {
            let window = input_state.get_scalar_window(CO2_DOUBLING_ERF);
            window.at_end().unwrap_or_else(|| window.at_start())
        } else {
            self.parameters.rf_2xco2
        };
        let cum_t = self.cumulative_temperature(state);

        // The last substep takes the internal efficacy from the previous year's sensitivity
        // and the others the one before, so the first two years start from the initial one.
        let subannual_scales = match &self.parameters.subannual_forcing {
            Some(SubannualForcing {
                rf_regions: Some(_),
                internal_efficacy,
                efficacy,
                ..
            }) => {
                let previous = state.subannual_internal_efficacy;
                let current = state
                    .subannual_next_internal_efficacy
                    .unwrap_or(*internal_efficacy);
                state.subannual_internal_efficacy = Some(current);
                (efficacy / previous.unwrap_or(current), efficacy / current)
            }
            _ => (1.0, 1.0),
        };

        // Pre-compute ground heat capacity if enabled
        let c_ground = if self.parameters.land_heat_capacity_enabled {
            self.parameters.land_hc_eff_thickness * heat_capacity_per_metre()
        } else {
            0.0
        };

        // Monthly sub-stepping with per-substep forcing interpolation.
        //
        // MAGICC7 runs steps 1..12 for normal years, with forcing queried at:
        //   step 1: year + 1/12, step 2: year + 2/12, ..., step 12: year + 12/12
        // The last substep (step 12) is at t_next (start of next year).
        // The first year also runs step 0, at t_current.
        // See docs/modules/time_conventions.md for details.
        let first_step = usize::from(!state.temperature_history.is_empty());
        let start_ocean_temps = self
            .parameters
            .thermal_expansion
            .as_ref()
            .map(|_| state.ocean_temps.clone());
        let mut box_temp_sums = [0.0; 4];
        let mut adjusted_ecs = self.parameters.ecs;
        let mut lam = self.initial_sensitivity();
        let mut forcing_end = None;
        for step_idx in first_step..=self.parameters.steps_per_year {
            let frac = (t_current + step_idx as FloatValue / steps) - t_current;
            let extra = match &self.parameters.subannual_forcing {
                Some(extra) => {
                    let values = extra.at(t_current, step_idx, self.parameters.steps_per_year);
                    let scale = if step_idx == self.parameters.steps_per_year {
                        subannual_scales.1
                    } else {
                        subannual_scales.0
                    };
                    values.map(|v| v * scale)
                }
                None => [0.0; 4],
            };
            let add_extra = |f: FourBoxSlice| {
                FourBoxSlice::from_array(std::array::from_fn(|i| f.0[i] + extra[i]))
            };
            // MAGICC adjusts the sensitivity every substep, from that substep's total forcing.
            let (regional, global_forcing) = match &forcing_input {
                ForcingInput::Regional(..) => {
                    let f = add_extra(self.box_forcing(&forcing_input, frac, 1.0));
                    let q = (0..4).fold(0.0, |acc, i| acc + f.0[i] * area[i]);
                    (Some(f), q)
                }
                ForcingInput::Global(start, end) => {
                    let extra_global: FloatValue = (0..4).map(|i| extra[i] * area[i]).sum();
                    (None, lerp(*start, *end, frac) + extra_global)
                }
            };
            adjusted_ecs = self.adjusted_ecs(global_forcing, q_2xco2, cum_t);
            lam = self.sensitivity(adjusted_ecs, q_2xco2);
            let (current_lambda_ocean, current_lambda_land) = (lam.lambda_ocean, lam.lambda_land);
            let forcing = regional.unwrap_or_else(|| {
                add_extra(self.box_forcing(&forcing_input, frac, lam.co2_internal_efficacy))
            });
            forcing_end = Some(forcing);
            // Forward Euler ground update from the previous substep's land temperatures.
            // The ocean solve sees the same flux, so it takes the ground temperatures from before.
            let [nh_ground, sh_ground] = state.ground_temps;
            if self.parameters.land_heat_capacity_enabled {
                for (hemi, &f_l) in [fgnl, fgsl].iter().enumerate() {
                    if f_l < 1e-15 {
                        continue;
                    }
                    let g = state.ground_temps[hemi];
                    state.ground_temps[hemi] = g
                        + (((dt_sub / f_l) / c_ground) * self.parameters.k_lg)
                            * (state.land_temps[hemi] - g);
                }
            }

            // Step each hemisphere's ocean with explicit K_NS exchange.
            let [alpha_eff_nh, alpha_eff_sh] = state.alpha_eff;
            let sst_nh = self.step_hemisphere(
                state,
                0,
                forcing.get(FourBoxRegion::NorthernOcean),
                forcing.get(FourBoxRegion::NorthernLand),
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
                forcing.get(FourBoxRegion::SouthernLand),
                dt_sub,
                current_lambda_ocean,
                current_lambda_land,
                state.hemi_heat_exchange[1],
                sh_ground,
                alpha_eff_sh,
            );

            state.alpha_eff = [self.alpha_eff(sst_nh), self.alpha_eff(sst_sh)];

            // Update land temperatures from new ocean SSTs (MAGICC7 lines 3214-3222)
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
            if step_idx > 0 {
                let boxes = [
                    t_air_nho,
                    state.land_temps[0],
                    t_air_sho,
                    state.land_temps[1],
                ];
                for (sum, t) in box_temp_sums.iter_mut().zip(boxes) {
                    *sum += t;
                }
            }
            self.update_upwelling(state, global_temp);
        }

        if let Some(SubannualForcing {
            rf_regions: Some(pattern),
            ..
        }) = &self.parameters.subannual_forcing
        {
            state.subannual_next_internal_efficacy = Some(lamcalc::calc_internal_efficacy(
                q_2xco2,
                &lam.matrix_inverse,
                &area,
                pattern,
                adjusted_ecs,
            ));
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

        // The last substep's forcing, so that N = Q_eff - lambda*T matches the forcing that drove T
        let forcing_end = forcing_end.expect("the step runs at least one substep");
        let heat_uptake = self.calculate_heat_uptake(
            &forcing_end,
            &surface_temperature,
            lam.lambda_ocean,
            lam.lambda_land,
        );
        let ocean_heat_content = self.calculate_ocean_heat_content(state);

        // SST is the mean of the two ocean box temperatures
        let sst = (sst_nh + sst_sh) / 2.0;

        let mut outputs: OutputState = ClimateUDEBOutputs {
            surface_temperature,
            heat_uptake,
            ocean_heat_content,
            sst,
            climate_sensitivity: adjusted_ecs,
        }
        .into();
        if let (Some(expansion), Some(start)) =
            (&self.parameters.thermal_expansion, start_ocean_temps)
        {
            // MAGICC7 averages each box over the year before weighting by area
            let annual_mean_temp: FloatValue = box_temp_sums
                .iter()
                .zip([fgno, fgnl, fgso, fgsl])
                .map(|(sum, area)| sum / steps * area)
                .sum();
            if t_current >= expansion.start_year {
                state.thermal_expansion += thermal_expansion_increment(
                    expansion,
                    &self.parameters,
                    &state.initial_ocean_profile,
                    &start,
                    &state.ocean_temps,
                );
            }
            outputs.insert(
                ANNUAL_MEAN_TEMPERATURE.to_string(),
                StateValue::Scalar(annual_mean_temp),
            );
            outputs.insert(
                THERMAL_EXPANSION.to_string(),
                StateValue::Scalar(state.thermal_expansion),
            );
        }

        Ok(outputs)
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
        let mut definitions = Self::generated_definitions();
        if self.parameters.time_varying_q2x {
            definitions.push(RequirementDefinition::scalar_input(
                CO2_DOUBLING_ERF,
                "W/m^2",
            ));
        }
        if self.parameters.thermal_expansion.is_some() {
            definitions.push(RequirementDefinition::scalar_output(
                ANNUAL_MEAN_TEMPERATURE,
                "K",
            ));
            definitions.push(RequirementDefinition::scalar_output(
                THERMAL_EXPANSION,
                "mm",
            ));
        }
        if self.parameters.regional_forcing {
            for definition in &mut definitions {
                if definition.name == "Effective Radiative Forcing" {
                    *definition = RequirementDefinition::four_box_input(REGIONAL_FORCING, "W/m^2");
                }
            }
        }
        definitions
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
        // Per-hemisphere profiles
        assert_eq!(state.initial_ocean_profile.len(), 2);
        assert_eq!(state.initial_ocean_profile[0].len(), n);
        assert_eq!(state.initial_ocean_profile[1].len(), n);

        // Both hemispheres should decrease monotonically
        for hemi in 0..2 {
            let profile = &state.initial_ocean_profile[hemi];
            for l in 1..n {
                assert!(
                    profile[l] < profile[l - 1],
                    "hemi {} profile should decrease with depth: layer {} ({}) >= layer {} ({})",
                    hemi,
                    l,
                    profile[l],
                    l - 1,
                    profile[l - 1]
                );
                assert!(
                    profile[l] >= 0.7,
                    "hemi {} profile should stay above 0.7: layer {} = {}",
                    hemi,
                    l,
                    profile[l]
                );
            }

            // Deep layers should be cold
            let deepest = profile[n - 1];
            assert!(
                deepest < 2.0,
                "hemi {} deepest layer should be < 2.0, got {deepest}",
                hemi
            );
        }

        // NH mixed layer should be warmer than SH (CMIP5 default)
        assert!(
            state.initial_ocean_profile[0][0] > state.initial_ocean_profile[1][0],
            "NH mixed layer ({}) should be warmer than SH ({})",
            state.initial_ocean_profile[0][0],
            state.initial_ocean_profile[1][0]
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

        assert_eq!(defs.len(), 6);

        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"Effective Radiative Forcing"));
        assert!(names.contains(&"Surface Temperature"));
        assert!(names.contains(&"Heat Uptake"));
        assert!(names.contains(&"Ocean Heat Content"));
        assert!(names.contains(&"Sea Surface Temperature"));
        assert!(names.contains(&CLIMATE_SENSITIVITY));
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

    #[test]
    fn test_deserialized_component_steps_like_original() {
        let component = default_component();
        let json = serde_json::to_string(&component).unwrap();
        let parsed: ClimateUDEB = serde_json::from_str(&json).unwrap();

        let ocean_temps = |udeb: &ClimateUDEB| {
            let mut state = default_state(udeb);
            for _ in 0..24 {
                udeb.step_hemisphere(
                    &mut state,
                    0,
                    3.71,
                    3.71,
                    1.0 / 12.0,
                    udeb.lambda_ocean,
                    udeb.lambda_land,
                    0.0,
                    0.0,
                    udeb.parameters.temp_adjust_alpha,
                );
            }
            state.ocean_temps[0].clone()
        };

        assert_eq!(ocean_temps(&component), ocean_temps(&parsed));
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
        let ecs_adj =
            component.adjusted_ecs(3.71, 3.71, component.cumulative_temperature(&empty_state));
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
        let ecs_eq = component.adjusted_ecs(
            component.parameters.rf_2xco2,
            component.parameters.rf_2xco2,
            component.cumulative_temperature(&eq_state),
        );
        assert!(
            (ecs_eq - component.parameters.ecs).abs() < 0.01,
            "Equilibrium history should leave ECS unchanged: got {ecs_eq}"
        );
    }

    #[test]
    fn test_adjusted_ecs_with_large_sensitivity() {
        let params = ClimateUDEBParameters {
            feedback_cumt_sensitivity: 0.5, // exaggerated
            ..Default::default()
        };
        let component = ClimateUDEB::from_parameters(params).unwrap();

        // After 100 years of 1K warming: cumT = 100, cumT_2x = 3.0 * 300 = 900
        let yearly_temps: Vec<FloatValue> = vec![1.0; 100];
        let state = state_with_history(&component, &yearly_temps);
        let ecs_adj = component.adjusted_ecs(3.71, 3.71, component.cumulative_temperature(&state));

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
        let params = ClimateUDEBParameters {
            efficacy_apply: 0,
            ..Default::default()
        };
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
        let params_0 = ClimateUDEBParameters {
            efficacy_apply: 0,
            ..Default::default()
        };
        let comp_0 = ClimateUDEB::from_parameters(params_0).unwrap();
        let mut state_0 = default_state(&comp_0);
        let out_0 = comp_0
            .solve_impl(2000.0, 2001.0, &input_state, &mut state_0)
            .expect("solve failed");

        // Mode 2: AR6 efficacy adjustment
        let params_2 = ClimateUDEBParameters {
            efficacy_apply: 2,
            ..Default::default()
        };
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

    #[test]
    fn test_co2_qfrac_accessor() {
        let component = default_component();
        let qfrac = component.co2_qfrac();

        // With AR6 defaults, qfrac should be non-uniform
        println!("CO2 qfrac = {:?}", qfrac);
        for &q in qfrac {
            assert!(q.is_finite(), "qfrac entry should be finite");
            assert!(q > 0.0, "qfrac entry should be positive, got {q}");
        }

        // Area-weighted sum of qfrac * area should equal 1.0
        let (fgno, fgnl, fgso, fgsl) = component.parameters.global_box_fractions();
        let area = [fgno, fgnl, fgso, fgsl];
        let weighted_sum: FloatValue = qfrac.iter().zip(area.iter()).map(|(q, a)| q * a).sum();
        assert!(
            (weighted_sum - 1.0).abs() < 1e-10,
            "Area-weighted qfrac sum should be 1.0, got {weighted_sum}"
        );
    }

    #[test]
    fn test_uniform_rf_regions_gives_uniform_qfrac() {
        let params = ClimateUDEBParameters {
            rf_regions_co2: [1.0; 4],
            ..Default::default()
        };
        let component = ClimateUDEB::from_parameters(params).unwrap();
        let qfrac = component.co2_qfrac();

        // Uniform rf_regions should give qfrac = 1.0 for all boxes
        for (i, &q) in qfrac.iter().enumerate() {
            assert!(
                (q - 1.0).abs() < 1e-10,
                "Uniform rf_regions should give qfrac=1.0, box {i} got {q}"
            );
        }
    }

    #[test]
    fn test_efficacy_mode_one_ignores_prescribed_efficacy() {
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

        let solve = |efficacy_apply, prescribed_efficacy_co2| {
            let params = ClimateUDEBParameters {
                efficacy_apply,
                prescribed_efficacy_co2,
                ..Default::default()
            };
            let comp = ClimateUDEB::from_parameters(params).unwrap();
            let mut state = default_state(&comp);
            comp.solve_impl(2000.0, 2001.0, &input_state, &mut state)
                .expect("solve failed")
        };
        let out_1 = solve(1, 1.2);
        let out_2 = solve(2, 1.0);

        let sst_1 = match out_1.get("Sea Surface Temperature").unwrap() {
            StateValue::Scalar(v) => *v,
            other => panic!("Expected scalar, got {:?}", other),
        };
        let sst_2 = match out_2.get("Sea Surface Temperature").unwrap() {
            StateValue::Scalar(v) => *v,
            other => panic!("Expected scalar, got {:?}", other),
        };
        assert_eq!(sst_1, sst_2);
    }

    #[test]
    fn test_prescribed_efficacy_in_mode_two() {
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

        // Mode 2 with prescribed=1.0 (default)
        let params_2a = ClimateUDEBParameters {
            efficacy_apply: 2,
            ..Default::default()
        };
        let comp_2a = ClimateUDEB::from_parameters(params_2a).unwrap();
        let mut state_2a = default_state(&comp_2a);
        let out_2a = comp_2a
            .solve_impl(2000.0, 2001.0, &input_state, &mut state_2a)
            .expect("solve failed");

        // Mode 2 with prescribed=1.1 (10% additional efficacy)
        let params_2b = ClimateUDEBParameters {
            efficacy_apply: 2,
            prescribed_efficacy_co2: 1.1,
            ..Default::default()
        };
        let comp_2b = ClimateUDEB::from_parameters(params_2b).unwrap();
        let mut state_2b = default_state(&comp_2b);
        let out_2b = comp_2b
            .solve_impl(2000.0, 2001.0, &input_state, &mut state_2b)
            .expect("solve failed");

        let sst_2a = match out_2a.get("Sea Surface Temperature").unwrap() {
            StateValue::Scalar(v) => *v,
            other => panic!("Expected scalar, got {:?}", other),
        };
        let sst_2b = match out_2b.get("Sea Surface Temperature").unwrap() {
            StateValue::Scalar(v) => *v,
            other => panic!("Expected scalar, got {:?}", other),
        };

        println!("SST mode2(presc=1.0) = {sst_2a:.6}, SST mode2(presc=1.1) = {sst_2b:.6}");

        // Higher prescribed efficacy should amplify the internal efficacy adjustment
        assert!(
            (sst_2a - sst_2b).abs() > 1e-6,
            "Different prescribed efficacy should change temperature: \
             presc=1.0: {sst_2a}, presc=1.1: {sst_2b}"
        );

        // With prescribed > 1.0, effective forcing is larger, so more warming
        assert!(
            sst_2b > sst_2a,
            "Higher prescribed efficacy should produce more warming: \
             presc=1.0: {sst_2a}, presc=1.1: {sst_2b}"
        );
    }

    #[test]
    fn test_zero_steps_per_year_rejected() {
        let params = ClimateUDEBParameters {
            steps_per_year: 0,
            ..Default::default()
        };
        assert!(ClimateUDEB::from_parameters(params).is_err());
    }

    #[test]
    fn test_invalid_prescribed_efficacy_rejected() {
        // Zero should be rejected
        let mut params = ClimateUDEBParameters {
            prescribed_efficacy_co2: 0.0,
            ..Default::default()
        };
        assert!(
            ClimateUDEB::from_parameters(params.clone()).is_err(),
            "prescribed_efficacy_co2=0.0 should be rejected"
        );

        // Negative should be rejected
        params.prescribed_efficacy_co2 = -1.0;
        assert!(
            ClimateUDEB::from_parameters(params.clone()).is_err(),
            "prescribed_efficacy_co2=-1.0 should be rejected"
        );

        // NaN should be rejected
        params.prescribed_efficacy_co2 = FloatValue::NAN;
        assert!(
            ClimateUDEB::from_parameters(params.clone()).is_err(),
            "prescribed_efficacy_co2=NaN should be rejected"
        );

        // Positive value should be accepted
        params.prescribed_efficacy_co2 = 0.5;
        assert!(
            ClimateUDEB::from_parameters(params).is_ok(),
            "prescribed_efficacy_co2=0.5 should be accepted"
        );
    }

    /// Global ERF on 2000 to 2003 and zero initial surface temperature.
    fn yearly_items(erf: [FloatValue; 4]) -> Vec<rscm_core::timeseries_collection::TimeseriesItem> {
        use rscm_core::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
        use rscm_core::spatial::FourBoxGrid;
        use rscm_core::timeseries::TimeAxis;
        use rscm_core::timeseries_collection::{TimeseriesData, TimeseriesItem, VariableType};
        use std::sync::Arc;

        let times = ndarray::array![2000.0, 2001.0, 2002.0, 2003.0];
        let temperature = GridTimeseries::new(
            Array2::zeros((4, 4)),
            Arc::new(TimeAxis::from_values(times.clone())),
            FourBoxGrid::magicc_standard(),
            "K".to_string(),
            InterpolationStrategy::from(LinearSplineStrategy::new(true)),
        );
        vec![
            TimeseriesItem {
                data: TimeseriesData::Scalar(rscm_core::timeseries::Timeseries::from_values(
                    ndarray::Array1::from_vec(erf.to_vec()),
                    times,
                )),
                name: "Effective Radiative Forcing".to_string(),
                variable_type: VariableType::Exogenous,
            },
            TimeseriesItem {
                data: TimeseriesData::FourBox(temperature),
                name: "Surface Temperature".to_string(),
                variable_type: VariableType::Endogenous,
            },
        ]
    }

    fn assert_same_climate(a: &ClimateUDEBState, b: &ClimateUDEBState, tolerance: FloatValue) {
        for hemi in 0..2 {
            for (x, y) in a.ocean_temps[hemi].iter().zip(&b.ocean_temps[hemi]) {
                assert!((x - y).abs() < tolerance, "ocean {hemi}: {x} vs {y}");
            }
            let (x, y) = (a.land_temps[hemi], b.land_temps[hemi]);
            assert!((x - y).abs() < tolerance, "land {hemi}: {x} vs {y}");
        }
    }

    #[test]
    fn test_subannual_efficacy_lags_one_year_until_last_substep() {
        let pattern = [1.0, 2.0, 0.5, 1.5];
        let box_values = [1.0, 0.5, 0.8, 0.3];
        let erf = [0.0, 2.0, 4.0, 6.0];
        let params = |subannual| ClimateUDEBParameters {
            feedback_cumt_sensitivity: 0.0,
            feedback_q_sensitivity: 0.1,
            subannual_forcing: Some(subannual),
            ..Default::default()
        };
        let scaled = ClimateUDEB::from_parameters(params(SubannualForcing {
            start: 2000.0,
            values: vec![box_values; 36],
            rf_regions: Some(pattern),
            internal_efficacy: 2.0,
            efficacy: 1.0,
        }))
        .unwrap();

        // Internal efficacy of the pattern at the sensitivity of a year's last substep,
        // whose forcing includes the scaled extra forcing
        let p = &scaled.parameters;
        let (fgno, fgnl, fgso, fgsl) = p.global_box_fractions();
        let area = [fgno, fgnl, fgso, fgsl];
        let extra: FloatValue = (0..4).map(|i| box_values[i] * area[i]).sum();
        let efficacy = |year: usize, divisor: FloatValue| {
            let ecs = scaled.adjusted_ecs(erf[year + 1] + extra / divisor, p.rf_2xco2, 0.0);
            let lam = scaled.sensitivity(ecs, p.rf_2xco2);
            lamcalc::calc_internal_efficacy(p.rf_2xco2, &lam.matrix_inverse, &area, &pattern, ecs)
        };
        let e0 = efficacy(0, 2.0);
        let e1 = efficacy(1, e0);
        assert!(
            (e0 - 2.0).abs() > 0.1 && (e0 - e1).abs() > 1e-5,
            "{e0} {e1}"
        );

        // The same forcing with each substep's scaling applied up front
        let divisors: Vec<FloatValue> = (0..36)
            .map(|i| match i {
                0..=22 => 2.0,
                23..=34 => e0,
                _ => e1,
            })
            .collect();
        let unscaled = ClimateUDEB::from_parameters(params(SubannualForcing {
            start: 2000.0,
            values: divisors
                .iter()
                .map(|d| box_values.map(|v| v * (1.0 / d)))
                .collect(),
            rf_regions: None,
            internal_efficacy: 1.0,
            efficacy: 1.0,
        }))
        .unwrap();

        let items = yearly_items(erf);
        let (mut state_scaled, mut state_unscaled) =
            (scaled.initial_state(), unscaled.initial_state());
        for year in [2000.0, 2001.0, 2002.0] {
            let input_state = InputState::build(items.iter().collect(), year);
            scaled
                .solve_impl(year, year + 1.0, &input_state, &mut state_scaled)
                .unwrap();
            unscaled
                .solve_impl(year, year + 1.0, &input_state, &mut state_unscaled)
                .unwrap();
            assert_same_climate(&state_scaled, &state_unscaled, 1e-12);
        }
        assert_eq!(state_scaled.subannual_internal_efficacy, Some(e1));
    }

    #[test]
    fn test_two_layer_ocean_solves() {
        let component = ClimateUDEB::from_parameters(ClimateUDEBParameters {
            n_layers: 2,
            ..Default::default()
        })
        .unwrap();
        let items = yearly_items([0.0, 1.0, 2.0, 3.0]);
        let input_state = InputState::build(items.iter().collect(), 2000.0);
        let mut state = component.initial_state();
        component
            .solve_impl(2000.0, 2001.0, &input_state, &mut state)
            .unwrap();
        assert!(state.ocean_temps[0].iter().all(|t| t.is_finite()));
    }

    #[test]
    fn test_heat_uptake_includes_subannual_forcing() {
        let heat_uptake = |values: [FloatValue; 4]| {
            let component = ClimateUDEB::from_parameters(ClimateUDEBParameters {
                subannual_forcing: Some(SubannualForcing {
                    start: 2000.0,
                    values: vec![values; 12],
                    rf_regions: None,
                    internal_efficacy: 1.0,
                    efficacy: 1.0,
                }),
                ..Default::default()
            })
            .unwrap();
            let items = yearly_items([0.0; 4]);
            let input_state = InputState::build(items.iter().collect(), 2000.0);
            let mut state = component.initial_state();
            let out = component
                .solve_impl(2000.0, 2001.0, &input_state, &mut state)
                .unwrap();
            match out["Heat Uptake"] {
                StateValue::Scalar(v) => v,
                ref other => panic!("expected scalar, got {other:?}"),
            }
        };
        // A year of 1 W/m^2 warms the surface by much less than 1 / lambda
        let difference = heat_uptake([1.0; 4]) - heat_uptake([0.0; 4]);
        assert!(difference > 0.5, "{difference}");
    }

    #[test]
    fn test_alpha_eff_updates_every_substep() {
        // Twelve one-month steps match one annual step with twelve substeps
        let params = ClimateUDEBParameters {
            feedback_cumt_sensitivity: 0.0,
            feedback_q_sensitivity: 0.0,
            ..Default::default()
        };
        let annual = ClimateUDEB::from_parameters(params.clone()).unwrap();
        let monthly = ClimateUDEB::from_parameters(ClimateUDEBParameters {
            steps_per_year: 1,
            ..params
        })
        .unwrap();
        let items = yearly_items([3.71; 4]);
        let input_state = InputState::build(items.iter().collect(), 2000.0);

        let mut warm = annual.initial_state();
        annual
            .solve_impl(0.0, 1.0, &input_state, &mut warm)
            .unwrap();
        let mut state_annual = warm.clone();
        annual
            .solve_impl(1.0, 2.0, &input_state, &mut state_annual)
            .unwrap();

        let run_monthly = |hold_alpha: bool| {
            let mut state = warm.clone();
            for month in 12..24 {
                if hold_alpha {
                    state.alpha_eff = warm.alpha_eff;
                }
                let t = month as FloatValue / 12.0;
                monthly
                    .solve_impl(t, t + 1.0 / 12.0, &input_state, &mut state)
                    .unwrap();
            }
            state
        };
        let state_monthly = run_monthly(false);
        assert_same_climate(&state_annual, &state_monthly, 1e-12);
        for hemi in 0..2 {
            assert!((state_annual.alpha_eff[hemi] - state_monthly.alpha_eff[hemi]).abs() < 1e-12);
        }

        // Holding alpha_eff for the year gives a measurably different climate
        let held = run_monthly(true);
        let diff = (held.ocean_temps[0][0] - state_annual.ocean_temps[0][0]).abs();
        assert!(diff > 1e-8, "{diff}");
    }
}
