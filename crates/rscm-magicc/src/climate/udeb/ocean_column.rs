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
    pub(super) fn step_hemisphere(
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
    ///
    /// where $f_i$ are global area fractions, $\lambda_i$ is the per-box
    /// feedback parameter (ocean or land), and $T_i$ is the per-box temperature.
    /// This ensures the diagnostic is consistent with the LAMCALC-solved
    /// feedback parameters and any time-varying ECS adjustments.
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

        // Heat uptake = Q - sum(f_i * lambda_i * T_i)
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
