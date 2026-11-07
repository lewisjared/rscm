//! Ocean column solver for the UDEB climate model.
//!
//! Contains the tridiagonal diffusion-advection solver and related ocean
//! physics: depth-dependent diffusivity, upwelling dynamics, heat uptake,
//! and ocean heat content diagnostics.

use super::{heat_capacity_per_metre, ClimateUdeb};
use crate::climate::state::ClimateUdebState;
use crate::parameters::{CP_SEAWATER, DIFFUSIVITY_CM2S_TO_M2YR, EARTH_SURFACE_AREA, RHO_SEAWATER};
use rscm_core::state::FourBoxSlice;
use rscm_core::timeseries::FloatValue;
use rscm_core::utils::linear_algebra::thomas_solve_inplace;

impl ClimateUdeb {
    /// Calculate depth-dependent vertical diffusivity for each layer boundary.
    ///
    /// Diffusivity varies with the temperature gradient between the surface
    /// and bottom layer, decreasing linearly with relative depth:
    ///
    /// $$K_l = \max(K_{min}, K_0 + \frac{dK}{dT} \times (1 - l/(n-1)) \times (T_{top} - T_{bottom})) \times 3155.76$$
    ///
    /// Returns a Vec of length `n_layers - 1` (diffusivity at each layer boundary).
    pub(super) fn layer_diffusivities(
        &self,
        state: &ClimateUdebState,
        hemi: usize,
    ) -> Vec<FloatValue> {
        let n = self.parameters.n_layers;
        let t_top = state.ocean_temperatures[hemi][0];
        let t_bottom = state.ocean_temperatures[hemi][n - 1];

        let kappa_min_m2yr = self.parameters.kappa_min_m2_per_yr();

        self.kappa_slopes
            .iter()
            .map(|slope| {
                let k = (slope * (t_top - t_bottom) + self.parameters.vertical_diffusivity)
                    * DIFFUSIVITY_CM2S_TO_M2YR;
                k.max(kappa_min_m2yr)
            })
            .collect()
    }

    /// Step forward by dt (in years) for a single hemisphere.
    ///
    /// Uses implicit Thomas algorithm for the diffusion-advection equation
    /// with depth-dependent ocean area factors and inter-hemispheric heat
    /// exchange.
    /// The coefficients follow MAGICC7's operation order so the result matches it bit for bit.
    ///
    /// When ground heat capacity is enabled, the ground reservoir flux
    /// $K_{lg} (T_{land} - T_{ground})$ is subtracted explicitly from the mixed-layer RHS.
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable reference to component state
    /// * `hemi` - Hemisphere index (0 = NH, 1 = SH)
    /// * `ocean_forcing` - Forcing for this hemisphere's ocean box ($\text{W/m}^2$)
    /// * `land_forcing` - Forcing for this hemisphere's land box ($\text{W/m}^2$)
    /// * `dt` - Step in years
    /// * `lambda_ocean` - Ocean feedback parameter ($\text{W/m}^2\text{/K}$)
    /// * `lambda_land` - Land feedback parameter ($\text{W/m}^2\text{/K}$)
    /// * `hemispheric_heat_exchange` - Inter-hemispheric heat exchange ($\text{W/m}^2$), computed
    ///   explicitly from previous substep air temperatures (MAGICC7 lines 3174-3177)
    /// * `ground_temp` - Ground reservoir temperature before this substep's update
    /// * `alpha_eff` - Effective SST-to-air-temperature ratio from the previous substep's SST
    ///
    /// # Returns
    ///
    /// Mixed layer temperature anomaly (K)
    #[allow(clippy::too_many_arguments)]
    pub(super) fn step_hemisphere(
        &self,
        state: &mut ClimateUdebState,
        hemi: usize,
        ocean_forcing: FloatValue,
        land_forcing: FloatValue,
        dt: FloatValue,
        lambda_ocean: FloatValue,
        lambda_land: FloatValue,
        hemispheric_heat_exchange: FloatValue,
        ground_temp: FloatValue,
        alpha_eff: FloatValue,
    ) -> FloatValue {
        let p = &self.parameters;
        let n = p.n_layers;
        let kappas = self.layer_diffusivities(state, hemi);
        let w = state.upwelling_rates[hemi];
        let dw = state.upwelling_delta[hemi];
        let dz = p.layer_thickness;
        let dz1 = dz / 2.0;
        let pi_ratio = p.polar_sinking_ratio;
        let aft = &self.area_factors.af_top[hemi];
        let afb = &self.area_factors.af_bottom[hemi];
        let afd = &self.area_factors.af_diff[hemi];
        let init = &state.initial_ocean_profile[hemi];
        let t_polar = state.initial_polar_sinking_temperature;
        let temps = &state.ocean_temperatures[hemi];

        let c = heat_capacity_per_metre();
        let (fgno, fgnl, fgso, fgsl) = p.global_box_fractions();
        let (f_o, f_l) = if hemi == 0 {
            (fgno, fgnl)
        } else {
            (fgso, fgsl)
        };
        let denom = (lambda_land * f_l + p.land_ocean_heat_exchange) * f_o;

        let mut coefficients = vec![0.0; 4 * n];
        let (a, rest) = coefficients.split_at_mut(n);
        let (b, rest) = rest.split_at_mut(n);
        let (cc, d) = rest.split_at_mut(n);

        let dtpdm = dt / p.mixed_layer_depth;
        let feedback =
            ((p.amplify_ocean_to_land * (lambda_land * p.land_ocean_heat_exchange)) * f_l) / denom
                + lambda_ocean;
        b[0] = ((kappas[0] / dz1) * afb[0]
            + (pi_ratio * w) * afb[0]
            + (feedback * ((1.0 / c) * alpha_eff)) * aft[0])
            * dtpdm
            + 1.0;
        cc[0] = -((((kappas[0] / dz1) + w) * dtpdm) * afb[0]);
        let flux = ((p.land_ocean_heat_exchange * land_forcing) * f_l) / denom
            + (hemispheric_heat_exchange + ocean_forcing);
        d[0] = (((flux * dtpdm) / c) * aft[0] + temps[0])
            + ((init[1] - t_polar) * (dtpdm * dw)) * afb[0];
        if p.include_land_heat_capacity {
            d[0] -= (((aft[0] * dtpdm) / (f_o * c)) * p.land_ground_heat_exchange)
                * (state.land_temperatures[hemi] - ground_temp);
        }

        let dtpdz = dt / dz;
        let dw_dtpdz = dw * dtpdz;
        let up = ((temps[0] * pi_ratio) * dtpdz) * w;
        let polar = t_polar * dw_dtpdz;

        // Diffusion and advection terms of a boundary, shared by the rows above and below it.
        let boundary = |kappa: FloatValue| ((dtpdz * kappa) / dz, ((kappa / dz) + w) * dtpdz);
        let (mut diffusion_above, mut advection_above) = boundary(kappas[0]);
        for i in 1..n - 1 {
            let (diffusion_below, advection_below) = boundary(kappas[i]);
            if i == 1 {
                a[1] = -(((kappas[0] * dtpdz) / dz1) * aft[1]);
                b[1] = ((((kappas[0] * dtpdz) / dz1) * aft[1] + 1.0) + diffusion_below * afb[1])
                    + (w * dtpdz) * aft[1];
            } else {
                a[i] = -(diffusion_above * aft[i]);
                b[i] = advection_above * aft[i] + (diffusion_below * afb[i] + 1.0);
            }
            cc[i] = -(advection_below * afb[i]);
            d[i] = ((temps[i] + (init[i + 1] * afb[i] - init[i] * aft[i]) * dw_dtpdz)
                + up * afd[i])
                + afd[i] * polar;
            (diffusion_above, advection_above) = (diffusion_below, advection_below);
        }

        a[n - 1] = -(diffusion_above * aft[n - 1]);
        b[n - 1] = advection_above * aft[n - 1] + 1.0;
        d[n - 1] =
            aft[n - 1] * (dw_dtpdz * (t_polar - init[n - 1])) + (temps[n - 1] + up * aft[n - 1]);

        thomas_solve_inplace(a, b, cc, d);

        let max_temp = p.maximum_temperature;
        for (old, &temp) in state.ocean_temperatures[hemi].iter_mut().zip(d.iter()) {
            *old = temp.min(max_temp);
        }

        state.ocean_temperatures[hemi][0]
    }

    /// Update upwelling rate based on global temperature.
    ///
    /// Upwelling decreases with warming (thermohaline circulation weakening):
    ///
    /// $$w = \max(w_0 (1 - f_{var}), w_0 - f_{var} w_0 T_{global} / T_{threshold})$$
    ///
    /// The change from $w_0$ is kept for the next substep's RHS correction.
    pub(super) fn update_upwelling(&self, state: &mut ClimateUdebState, global_temp: FloatValue) {
        let w_0 = self.parameters.initial_upwelling_rate;
        let fw = self.parameters.upwelling_variable_fraction * w_0;
        let w_min = (1.0 - self.parameters.upwelling_variable_fraction) * w_0;
        let thresholds = [
            self.parameters.upwelling_threshold_temperature_nh,
            self.parameters.upwelling_threshold_temperature_sh,
        ];
        for (hemi, thr) in thresholds.into_iter().enumerate() {
            let x = (fw * global_temp) / thr;
            let (w, delta) = if w_min > w_0 - x {
                (w_min, -fw)
            } else {
                (w_0 - x, -x)
            };
            state.upwelling_rates[hemi] = w;
            state.upwelling_delta[hemi] = delta;
        }
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

    /// Calculate the global ocean heat content ($\text{J}$).
    ///
    /// Integrates the temperature anomaly over the ocean layers and the ocean area of each hemisphere.
    pub(super) fn calculate_ocean_heat_content(&self, state: &ClimateUdebState) -> FloatValue {
        let dz = self.parameters.layer_thickness;
        let dz_mix = self.parameters.mixed_layer_depth;

        let rho_c = RHO_SEAWATER * CP_SEAWATER;
        let ocean_area = [
            self.parameters.nh_ocean_fraction(),
            self.parameters.sh_ocean_fraction(),
        ]
        .map(|fraction| 0.5 * EARTH_SURFACE_AREA * fraction);

        let mut total_heat = 0.0;

        for (hemi, area) in ocean_area.iter().enumerate() {
            let mut column = dz_mix * state.ocean_temperatures[hemi][0];
            for layer in 1..self.parameters.n_layers {
                column += dz * state.ocean_temperatures[hemi][layer];
            }
            total_heat += rho_c * column * area;
        }

        total_heat
    }
}
