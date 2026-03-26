//! Ocean column solver for the UDEB climate model.
//!
//! Contains the tridiagonal diffusion-advection solver and related ocean
//! physics: depth-dependent diffusivity, upwelling dynamics, heat uptake,
//! and ocean heat content diagnostics.

use super::ClimateUDEB;
use crate::climate::state::ClimateUDEBState;
use crate::parameters::{CP_SEAWATER, DIFFUSIVITY_CM2S_TO_M2YR, RHO_SEAWATER};
use rscm_core::state::FourBoxSlice;
use rscm_core::timeseries::FloatValue;
use rscm_core::utils::linear_algebra::thomas_solve;

impl ClimateUDEB {
    /// Calculate depth-dependent vertical diffusivity for each layer boundary.
    ///
    /// Diffusivity varies with the temperature gradient between the surface
    /// and bottom layer, decreasing linearly with relative depth:
    ///
    /// $$K(z) = \max(K_{min}, K_0 + \frac{dK}{dT} \times (1 - z/z_{max}) \times (T_{top} - T_{bottom})) \times 3155.76$$
    ///
    /// Returns a Vec of length `n_layers - 1` (diffusivity at each layer boundary).
    pub(super) fn layer_diffusivities(
        &self,
        state: &ClimateUDEBState,
        hemi: usize,
    ) -> Vec<FloatValue> {
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

    /// Step forward by dt (in years) for a single hemisphere.
    ///
    /// Uses implicit Thomas algorithm for the diffusion-advection equation
    /// with depth-dependent ocean area factors and inter-hemispheric heat
    /// exchange.
    ///
    /// When ground heat capacity is enabled, $K_{lg}$ is incorporated into
    /// the coupled land-ocean feedback equations rather than applied as a
    /// separate correction. This modifies both the feedback parameter and
    /// forcing amplification to account for the ground reservoir absorbing
    /// heat from the land surface.
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable reference to component state
    /// * `hemi` - Hemisphere index (0 = NH, 1 = SH)
    /// * `forcing` - Forcing for this hemisphere's ocean box ($\text{W/m}^2$)
    /// * `dt` - Timestep in years
    /// * `lambda_ocean` - Ocean feedback parameter ($\text{W/m}^2\text{/K}$)
    /// * `lambda_land` - Land feedback parameter ($\text{W/m}^2\text{/K}$)
    /// * `hemi_heat_exchange` - Inter-hemispheric heat exchange ($\text{W/m}^2$), computed
    ///   explicitly from previous substep air temperatures (MAGICC7 lines 3174-3177)
    /// * `ground_temp` - Current ground reservoir temperature for ground heat coupling
    /// * `alpha_eff` - Effective SST-to-air-temperature ratio, computed once per annual
    ///   timestep from the end-of-previous-year SST (MAGICC7 behaviour)
    ///
    /// # Returns
    ///
    /// Mixed layer temperature anomaly (K)
    pub(super) fn step_hemisphere(
        &self,
        state: &mut ClimateUDEBState,
        hemi: usize,
        forcing: FloatValue,
        dt: FloatValue,
        lambda_ocean: FloatValue,
        lambda_land: FloatValue,
        hemi_heat_exchange: FloatValue,
        ground_temp: FloatValue,
        alpha_eff: FloatValue,
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
        // Coupled ocean-land feedback term (TERM_OCN_LAND_FEEDBACK)
        // MAGICC7.f90 lines 2803-2820.
        let f_l_hemi = if hemi == 0 {
            self.parameters.nh_land_fraction / 2.0
        } else {
            self.parameters.sh_land_fraction / 2.0
        };
        let f_o_hemi = 0.5 - f_l_hemi;

        // Land feedback denominator (MAGICC7 line 2803-2804).
        // Ground heat (K_lg) is NOT included here -- it is handled as a
        // separate explicit term on the RHS (MAGICC7 lines 2893-2902).
        let denominator = f_o_hemi * (self.parameters.k_lo + f_l_hemi * lambda_land);

        // Feedback term: MAGICC7 lines 2806-2820.
        // lambda_l * K_lo * alpha * f_l / DENOM
        let term_feedback = alpha_eff / c_mix
            * (lambda_ocean
                + lambda_land
                    * self.parameters.k_lo
                    * self.parameters.amplify_ocean_to_land
                    * f_l_hemi
                    / denominator);

        let dz1 = dz / 2.0;
        let term_diff = kappas[0] / (dz_mix * dz1) * dt;
        let term_upwell = w / dz_mix * dt;

        // Land forcing amplification (MAGICC7 lines 2843-2845).
        let forcing_amp = 1.0 + self.parameters.k_lo * f_l_hemi / denominator;

        // MAGICC7 applies af_top to the feedback term and af_top to the
        // forcing/exchange terms on the RHS (lines 2826-2856).
        // Inter-hemispheric heat exchange is explicit on the RHS (not implicit
        // on the diagonal), folded into the forcing term alongside ocean and
        // land forcing (MAGICC7 lines 2793-2800).
        b[0] = 1.0
            + term_feedback * dt * af_top[0]
            + term_diff * af_bot[0]
            + term_upwell * pi_ratio * af_bot[0];
        c[0] = -(term_diff + term_upwell) * af_bot[0];
        d[0] = state.ocean_temps[hemi][0]
            + (forcing * forcing_amp + hemi_heat_exchange) / c_mix * dt * af_top[0];

        // Ground heat as explicit subtraction on D(1) (MAGICC7 lines 2893-2902).
        // The ground reservoir heat flux damps the mixed layer by subtracting
        // K_lg * (T_land - T_ground) / (C_mix * FGO) from the RHS.
        // Uses previous-substep land and ground temperatures (explicit).
        if self.parameters.land_heat_capacity_enabled {
            let land_temp = state.land_temps[hemi];
            d[0] -= self.parameters.k_lg * (land_temp - ground_temp) / (c_mix * f_o_hemi)
                * dt
                * af_top[0];
        }

        // Layers 1 to n-2 (interior layers)
        for i in 1..n - 1 {
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
        // MAGICC7 uses af_top (not af_diff) for the bottom layer entrainment
        // (lines 3055-3058), since there is no layer below for bottom flow.
        let term_diff_up = kappas[n - 2] / (dz * dz) * dt;
        let term_upwell_bottom = w / dz * dt;

        a[n - 1] = -term_diff_up * af_top[n - 1];
        b[n - 1] = 1.0 + (term_diff_up + term_upwell_bottom) * af_top[n - 1];

        d[n - 1] = state.ocean_temps[hemi][n - 1]
            + pi_ratio * term_upwell_bottom * state.ocean_temps[hemi][0] * af_top[n - 1];

        // Variable upwelling correction terms (MAGICC7.f90 lines 2858-2874, 2957-2991, 3012-3039).
        //
        // When upwelling changes from its initial value, the equilibrium ocean temperature
        // profile shifts. These terms compensate for that shift by adjusting the RHS (D)
        // based on the initial temperature profile and the upwelling change.
        let delta_w = w - self.parameters.w_initial;
        if delta_w.abs() > 1e-15 {
            let init = &state.initial_ocean_profile[hemi];
            let t_polar = state.polar_sinking_temp;

            // Mixed layer (layer 0): MAGICC7 line 2858-2874
            let dt_per_dz_mix = dt / dz_mix;
            d[0] += dt_per_dz_mix * delta_w * (init[1] - t_polar) * af_bot[0];

            // Interior layers 1..n-2
            let dt_per_dz = dt / dz;
            for i in 1..n - 1 {
                d[i] += dt_per_dz * delta_w * (init[i + 1] * af_bot[i] - init[i] * af_top[i]);
                d[i] += dt_per_dz * delta_w * t_polar * af_diff[i];
            }

            // Bottom layer (n-1): MAGICC7 combines advection and entrainment
            // into a single term with af_top (lines 3059-3063).
            d[n - 1] += dt_per_dz * delta_w * (t_polar - init[n - 1]) * af_top[n - 1];
        }

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
    pub(super) fn update_upwelling(&self, state: &mut ClimateUDEBState, global_temp: FloatValue) {
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

    /// Calculate ocean heat uptake ($\text{W/m}^2$).
    ///
    /// $$\text{Heat uptake} = Q - \sum_i f_i \lambda_i T_i$$
    pub(super) fn calculate_heat_uptake(
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

        q_global - feedback_global
    }

    /// Calculate total ocean heat content ($\text{J/m}^2$).
    ///
    /// Integrates temperature anomaly over all ocean layers weighted by depth.
    pub(super) fn calculate_ocean_heat_content(&self, state: &ClimateUDEBState) -> FloatValue {
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
}
