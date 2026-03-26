//! Climate UDEB Parameters
//!
//! Parameters for the Upwelling-Diffusion Energy Balance (UDEB) climate model
//! following MAGICC7 Module 08.

use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Initial ocean temperature profile mode (`CORE_SWITCH_OCN_TEMPPROFILE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OceanTempProfile {
    /// Analytical exponential decay (uniform diffusivity + upwelling).
    #[serde(rename = "1")]
    Analytical = 1,
    /// CMIP5 multi-model mean observational profile (default).
    #[serde(rename = "2")]
    Cmip5 = 2,
}

/// Parameters for the 4-box UDEB climate model.
///
/// The UDEB model couples a 4-box atmosphere (Northern Ocean, Northern Land,
/// Southern Ocean, Southern Land) to a multi-layer ocean with vertical diffusion
/// and upwelling.
///
/// # Physical Constants
///
/// Several conversion factors are used internally:
/// - Diffusivity: $\text{cm}^2/\text{s}$ to $\text{m}^2/\text{yr}$ uses factor 3155.76 (= 100 * 31.5576)
/// - Heat capacity: $\text{W yr / m}^2\text{ K}$
///
/// # Default Values
///
/// Default values match MAGICC7 defaults from `MAGCFG_DEFAULTALL.CFG`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ClimateUDEBParameters {
    // Ocean structure parameters
    /// Number of ocean layers (including mixed layer).
    /// Default: 50
    pub n_layers: usize,

    /// Mixed layer depth (m).
    /// Default: 60.0
    pub mixed_layer_depth: FloatValue,

    /// Layer thickness for deeper ocean layers (m).
    /// Default: 100.0
    pub layer_thickness: FloatValue,

    // Diffusivity parameters
    /// Base vertical diffusivity ($\text{cm}^2/\text{s}$).
    /// Converted to $\text{m}^2/\text{yr}$ internally using factor 3155.76.
    /// Default: 0.75
    pub kappa: FloatValue,

    /// Minimum vertical diffusivity ($\text{cm}^2/\text{s}$).
    /// Floor for temperature-dependent diffusivity.
    /// Default: 0.1
    pub kappa_min: FloatValue,

    /// Temperature gradient coefficient for diffusivity ($\text{cm}^2/\text{s/K}$).
    /// Controls how diffusivity changes with vertical temperature gradient.
    /// Default: -0.191
    pub kappa_dkdt: FloatValue,

    // Upwelling parameters
    /// Initial upwelling rate (m/yr).
    /// Base rate before temperature-dependent adjustment.
    /// Default: 3.5
    pub w_initial: FloatValue,

    /// Variable fraction of upwelling (dimensionless).
    /// Fraction of upwelling that can vary with temperature.
    /// Default: 0.7
    pub w_variable_fraction: FloatValue,

    /// Temperature threshold for NH upwelling shutdown (K).
    /// Temperature anomaly at which upwelling reaches minimum.
    /// Default: 8.0
    pub w_threshold_temp_nh: FloatValue,

    /// Temperature threshold for SH upwelling shutdown (K).
    /// Default: 8.0
    pub w_threshold_temp_sh: FloatValue,

    // Climate sensitivity parameters
    /// Equilibrium climate sensitivity (K).
    /// Equilibrium warming for CO2 doubling.
    /// Default: 3.0
    pub ecs: FloatValue,

    /// Radiative forcing for $2 \times \text{CO}_2$ ($\text{W/m}^2$).
    /// Default: 3.71
    pub rf_2xco2: FloatValue,

    /// Land-ocean warming ratio (dimensionless).
    /// Ratio of land to ocean equilibrium warming.
    /// Default: 1.317
    pub rlo: FloatValue,

    /// Sensitivity of ECS to forcing level (dimensionless).
    /// Default: 7.84e-9
    pub feedback_q_sensitivity: FloatValue,

    /// Sensitivity of ECS to cumulative temperature (dimensionless).
    /// Default: 0.08
    pub feedback_cumt_sensitivity: FloatValue,

    /// Period for cumulative temperature integration (years).
    /// Default: 300.0
    pub feedback_cumt_period: FloatValue,

    // Heat exchange parameters
    /// Land-ocean heat exchange coefficient ($\text{W/m}^2\text{/K}$).
    /// Default: 1.44
    pub k_lo: FloatValue,

    /// Inter-hemispheric (North-South) heat exchange coefficient ($\text{W/m}^2\text{/K}$).
    /// Default: 0.31
    pub k_ns: FloatValue,

    /// Ocean-to-land heat exchange amplification factor (dimensionless).
    /// Default: 1.02
    pub amplify_ocean_to_land: FloatValue,

    // Area fractions
    /// Northern Hemisphere land fraction (dimensionless).
    /// Default: 0.42
    pub nh_land_fraction: FloatValue,

    /// Southern Hemisphere land fraction (dimensionless).
    /// Default: 0.21
    pub sh_land_fraction: FloatValue,

    /// Depth-dependent ocean area scale factor (0-1).
    ///
    /// Controls the strength of basin narrowing with depth:
    /// - 0.0 = cylindrical ocean (no narrowing)
    /// - 1.0 = full hypsometric profile
    ///
    /// Default: 1.0
    pub depth_dependent_area: FloatValue,

    // Ocean temperature adjustment
    /// Ocean-to-atmosphere temperature adjustment alpha (dimensionless).
    /// Linear coefficient for SAT/SST relationship.
    /// Default: 1.04
    pub temp_adjust_alpha: FloatValue,

    /// Ocean-to-atmosphere temperature adjustment gamma (1/K).
    /// Quadratic coefficient for SAT/SST relationship.
    /// Default: -0.002
    pub temp_adjust_gamma: FloatValue,

    // Polar sinking
    /// Polar sinking water temperature ratio (dimensionless).
    /// Fraction of surface temperature in polar sinking water.
    /// Default: 0.2
    pub polar_sinking_ratio: FloatValue,

    // Ground heat reservoir parameters
    /// Enable ground heat reservoir (`CORE_LANDHEATCAPACITY_APPLY`).
    ///
    /// When enabled, a ground heat reservoir damps the land temperature
    /// response by absorbing/releasing heat from the land surface.
    /// Default: true
    pub land_heat_capacity_enabled: bool,

    /// Land-ground heat exchange coefficient ($\text{W/m}^2\text{/K}$)
    /// (`CORE_HEATXCHANGE_LANDGROUND`).
    ///
    /// Per unit globe area. Controls the rate of heat exchange between
    /// the land surface and the ground heat reservoir.
    /// Default: 0.1
    pub k_lg: FloatValue,

    /// Effective thickness of ground heat reservoir (m)
    /// (`CORE_LANDHC_EFFTHICKNESS`).
    ///
    /// Ocean-water-equivalent depth parameterising the total land heat
    /// reservoir thermal inertia.
    /// Default: 300.0
    pub land_hc_eff_thickness: FloatValue,

    // Regional CO2 forcing
    /// Regional CO2 radiative forcing pattern (NH ocean, NH land, SH ocean, SH land).
    ///
    /// Relative weights used to compute per-box CO2 forcing fractions for the
    /// LAMCALC solver. Normalized against global-area-weighted sum following
    /// MAGICC7.f90 lines 8146-8165.
    ///
    /// Default values from `MAGCFG_DEFAULTALL.CFG` (AR6 calibration):
    /// `[1.4089, 1.37045, 1.43333, 1.33257]`.
    pub rf_regions_co2: [FloatValue; 4],

    // Efficacy parameters
    /// Efficacy application mode (matches MAGICC7 `RF_EFFICACY_APPLY`).
    ///
    /// - `0` = disabled (default): all agents treated with uniform spatial efficacy.
    /// - `1` = prescribed mode: effective forcing scaled by `prescribed_efficacy_co2`.
    /// - `2` = AR6 mode: effective forcing adjusted by `prescribed_efficacy_co2 / internal_efficacy`.
    #[serde(default)]
    pub efficacy_apply: u8,

    /// Prescribed CO2 efficacy factor (matches MAGICC7 `RF_EFFICACY_CO2`).
    ///
    /// In mode 1: $\text{EFFRF} = \text{ERF} \times \text{prescribed\_efficacy\_co2}$.
    /// In mode 2: $\text{EFFRF} = \text{ERF} \times \text{prescribed\_efficacy\_co2} / \text{internal\_efficacy}$.
    ///
    /// Default: 1.0 (no-op).
    #[serde(default = "default_prescribed_efficacy_co2")]
    pub prescribed_efficacy_co2: FloatValue,

    // Initial ocean temperature profile
    /// Initial ocean temperature profile mode (`CORE_SWITCH_OCN_TEMPPROFILE`).
    ///
    /// The profile is used as a reference baseline for variable upwelling
    /// correction terms. When upwelling changes from its initial value, these
    /// corrections compensate for the shift in equilibrium profile. Using the
    /// CMIP5 profile matches MAGICC7's default behaviour.
    pub ocean_temp_profile: OceanTempProfile,

    // Integration parameters
    /// Steps per year for sub-annual integration.
    /// Default: 12 (monthly)
    pub steps_per_year: usize,

    /// Maximum temperature anomaly cap (K).
    /// Safety limit to prevent runaway temperatures.
    /// Default: 25.0
    pub max_temperature: FloatValue,
}

impl Default for ClimateUDEBParameters {
    fn default() -> Self {
        Self {
            // Ocean structure
            n_layers: 50,
            mixed_layer_depth: 60.0,
            layer_thickness: 100.0,

            // Diffusivity
            kappa: 0.75,
            kappa_min: 0.1,
            kappa_dkdt: -0.191,

            // Upwelling
            w_initial: 3.5,
            w_variable_fraction: 0.7,
            w_threshold_temp_nh: 8.0,
            w_threshold_temp_sh: 8.0,

            // Climate sensitivity
            ecs: 3.0,
            rf_2xco2: 3.71,
            rlo: 1.317,
            feedback_q_sensitivity: 7.84e-9,
            feedback_cumt_sensitivity: 0.08,
            feedback_cumt_period: 300.0,

            // Heat exchange
            k_lo: 1.44,
            k_ns: 0.31,
            amplify_ocean_to_land: 1.02,

            // Area fractions
            nh_land_fraction: 0.42,
            sh_land_fraction: 0.21,
            depth_dependent_area: 1.0,

            // Temperature adjustment
            temp_adjust_alpha: 1.04,
            temp_adjust_gamma: -0.002,

            // Polar sinking
            polar_sinking_ratio: 0.2,

            // Ground heat reservoir
            land_heat_capacity_enabled: true,
            k_lg: 0.1,
            land_hc_eff_thickness: 300.0,

            // Regional CO2 forcing (MAGCFG_DEFAULTALL.CFG)
            rf_regions_co2: [1.4089, 1.37045, 1.43333, 1.33257],

            // Efficacy
            efficacy_apply: 0,
            prescribed_efficacy_co2: 1.0,

            // Initial ocean profile
            ocean_temp_profile: OceanTempProfile::Cmip5,

            // Integration
            steps_per_year: 12,
            max_temperature: 25.0,
        }
    }
}

fn default_prescribed_efficacy_co2() -> FloatValue {
    1.0
}

/// Conversion factor from $\text{cm}^2/\text{s}$ to $\text{m}^2/\text{yr}$.
///
/// $100 \, (\text{cm}^2 \to \text{m}^2) \times 31.5576 \, (\text{s} \to \text{yr}) = 3155.76$
pub const DIFFUSIVITY_CM2S_TO_M2YR: FloatValue = 3155.76;

/// Density of seawater ($\text{kg/m}^3$).
pub const RHO_SEAWATER: FloatValue = 1026.0;

/// Specific heat capacity of seawater ($\text{J/(kg K)}$).
pub const CP_SEAWATER: FloatValue = 3985.0;

/// Seconds per Julian year ($\text{s/yr}$).
pub const SECONDS_PER_YEAR: FloatValue = 31557600.0;

/// Volumetric heat capacity of seawater per unit depth ($\text{W yr / m}^2\text{ K}$).
///
/// $$\frac{\rho \, c_p \, d}{\text{seconds per year}}$$
pub fn heat_capacity_per_unit_area(depth_m: FloatValue) -> FloatValue {
    RHO_SEAWATER * CP_SEAWATER * depth_m / SECONDS_PER_YEAR
}

impl ClimateUDEBParameters {
    /// Convert vertical diffusivity from $\text{cm}^2/\text{s}$ to $\text{m}^2/\text{yr}$.
    pub fn kappa_m2_per_yr(&self) -> FloatValue {
        self.kappa * DIFFUSIVITY_CM2S_TO_M2YR
    }

    /// Convert minimum vertical diffusivity from $\text{cm}^2/\text{s}$ to $\text{m}^2/\text{yr}$.
    pub fn kappa_min_m2_per_yr(&self) -> FloatValue {
        self.kappa_min * DIFFUSIVITY_CM2S_TO_M2YR
    }

    /// Get the global climate feedback parameter ($\text{W/m}^2\text{/K}$).
    ///
    /// $$\lambda = \Delta Q_{2x} / \text{ECS}$$
    pub fn lambda_global(&self) -> FloatValue {
        self.rf_2xco2 / self.ecs
    }

    /// Get the ocean area fraction for Northern Hemisphere.
    pub fn nh_ocean_fraction(&self) -> FloatValue {
        1.0 - self.nh_land_fraction
    }

    /// Get the ocean area fraction for Southern Hemisphere.
    pub fn sh_ocean_fraction(&self) -> FloatValue {
        1.0 - self.sh_land_fraction
    }

    /// Get the global ocean fraction.
    ///
    /// Assumes equal hemisphere areas (0.5 each).
    pub fn global_ocean_fraction(&self) -> FloatValue {
        0.5 * (self.nh_ocean_fraction() + self.sh_ocean_fraction())
    }

    /// Get per-box global area fractions (FGNO, FGNL, FGSO, FGSL).
    ///
    /// These are fractions of the total globe, not of each hemisphere.
    /// Order: (NH ocean, NH land, SH ocean, SH land).
    /// Matches MAGICC's GLOBALAREAFRACTIONS ordering.
    pub fn global_box_fractions(&self) -> (FloatValue, FloatValue, FloatValue, FloatValue) {
        let fgnl = self.nh_land_fraction / 2.0;
        let fgno = 0.5 - fgnl;
        let fgsl = self.sh_land_fraction / 2.0;
        let fgso = 0.5 - fgsl;
        (fgno, fgnl, fgso, fgsl)
    }

    /// Get the global land fraction.
    pub fn global_land_fraction(&self) -> FloatValue {
        0.5 * (self.nh_land_fraction + self.sh_land_fraction)
    }

    /// Compute the ocean area fraction at a given depth using the hypsometric profile.
    ///
    /// Returns the fraction of ocean surface area that is still ocean at depth `z` metres.
    /// The result is scaled by [`depth_dependent_area`](Self::depth_dependent_area):
    /// when that parameter is 0 the ocean is cylindrical ($A = 1$), when 1 the full
    /// hypsometric profile is used.
    ///
    /// Based on global ocean bathymetry (ETOPO/GEBCO).
    pub fn ocean_area_at_depth(&self, depth_m: FloatValue) -> FloatValue {
        /// Depth breakpoints (metres) for the hypsometric lookup table.
        const DEPTH: [FloatValue; 12] = [
            0.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0,
            5000.0,
        ];
        /// Fraction of ocean surface area remaining at each depth breakpoint.
        const AREA: [FloatValue; 12] = [
            1.0, 0.975, 0.95, 0.92, 0.91, 0.87, 0.81, 0.72, 0.55, 0.38, 0.18, 0.05,
        ];

        // Linearly interpolate the hypsometric profile
        let hydro = if depth_m <= DEPTH[0] {
            AREA[0]
        } else if depth_m >= DEPTH[DEPTH.len() - 1] {
            AREA[AREA.len() - 1]
        } else {
            let mut a = AREA[0];
            for i in 1..DEPTH.len() {
                if depth_m <= DEPTH[i] {
                    let frac = (depth_m - DEPTH[i - 1]) / (DEPTH[i] - DEPTH[i - 1]);
                    a = AREA[i - 1] + frac * (AREA[i] - AREA[i - 1]);
                    break;
                }
            }
            a
        };

        // Blend between cylindrical (1.0) and full hypsometric profile
        1.0 + self.depth_dependent_area * (hydro - 1.0)
    }

    /// Compute area factors for the tridiagonal ocean solver.
    ///
    /// Returns `(af_top, af_bottom, af_diff)` vectors of length `n_layers` where:
    /// - `af_top[l]`  = area at top boundary of layer `l` / average area of layer `l`
    /// - `af_bottom[l]` = area at bottom boundary of layer `l` / average area of layer `l`
    /// - `af_diff[l]` = (area at top - area at bottom) / average area of layer `l`
    ///
    /// `af_diff` is used for the entrainment terms (polar sinking water entering
    /// each deep layer). Matches MAGICC7 AREAFACTOR_DIFFFLOW.
    ///
    /// For a cylindrical ocean (`depth_dependent_area = 0`), all factors are 1.0
    /// (and `af_diff` is 0.0).
    pub fn compute_area_factors(&self) -> (Vec<FloatValue>, Vec<FloatValue>, Vec<FloatValue>) {
        let n = self.n_layers;
        let mut af_top = Vec::with_capacity(n);
        let mut af_bottom = Vec::with_capacity(n);
        let mut af_diff = Vec::with_capacity(n);

        for l in 0..n {
            let (z_top, z_bottom) = if l == 0 {
                (0.0, self.mixed_layer_depth)
            } else {
                let top = self.mixed_layer_depth + (l as FloatValue - 1.0) * self.layer_thickness;
                let bot = top + self.layer_thickness;
                (top, bot)
            };

            let a_top = self.ocean_area_at_depth(z_top);
            let a_bottom = self.ocean_area_at_depth(z_bottom);
            let a_avg = (a_top + a_bottom) / 2.0;

            af_top.push(a_top / a_avg);
            af_bottom.push(a_bottom / a_avg);
            // MAGICC7 AREAFACTOR_DIFFFLOW: (A_top - A_bottom) / A_avg
            af_diff.push((a_top - a_bottom) / a_avg);
        }

        (af_top, af_bottom, af_diff)
    }

    /// Calculate heat capacity of the mixed layer per unit area ($\text{W yr / m}^2\text{ K}$).
    pub fn mixed_layer_heat_capacity(&self) -> FloatValue {
        heat_capacity_per_unit_area(self.mixed_layer_depth)
    }

    /// Calculate heat capacity of the ground reservoir per unit area
    /// ($\text{W yr / m}^2\text{ K}$).
    ///
    /// Uses ocean-water-equivalent heat capacity with the effective ground
    /// thickness parameterising the total land heat reservoir.
    pub fn ground_heat_capacity(&self) -> FloatValue {
        heat_capacity_per_unit_area(self.land_hc_eff_thickness)
    }

    /// Compute the initial ocean temperature profile for a given hemisphere.
    ///
    /// Returns a Vec of length `n_layers` with the initial temperature (K)
    /// for each ocean layer. Layer 0 is the mixed layer.
    ///
    /// When `ocean_temp_profile` is [`OceanTempProfile::Cmip5`] (default),
    /// returns the CMIP5 multi-model mean profile. When
    /// [`OceanTempProfile::Analytical`], returns the analytical exponential
    /// decay profile.
    pub fn initial_ocean_profile(&self, hemi: usize) -> Vec<FloatValue> {
        assert!(
            hemi < 2,
            "initial_ocean_profile: hemi must be 0 (NH) or 1 (SH), got {}",
            hemi
        );

        match self.ocean_temp_profile {
            OceanTempProfile::Cmip5 => {
                let cmip5 = if hemi == 0 {
                    &CMIP5_PROFILE_NH
                } else {
                    &CMIP5_PROFILE_SH
                };
                // Truncate or pad to n_layers
                let mut profile = vec![0.0; self.n_layers];
                for (i, val) in profile.iter_mut().enumerate() {
                    *val = if i < cmip5.len() {
                        cmip5[i]
                    } else {
                        // Extrapolate with last value
                        cmip5[cmip5.len() - 1]
                    };
                }
                profile
            }
            OceanTempProfile::Analytical => {
                let t_mix = 17.2_f64;
                let t_polar = 1.0_f64;
                let kappa = self.kappa_m2_per_yr();
                let w = self.w_initial;
                let dz = self.layer_thickness;

                let mut profile = vec![0.0; self.n_layers];
                profile[0] = t_mix;
                for l in 1..self.n_layers {
                    let depth = (l as f64 - 1.0) * dz + 0.5 * dz;
                    profile[l] = t_polar + (t_mix - t_polar) * (-w * depth / kappa).exp();
                }
                profile
            }
        }
    }
}

/// CMIP5 multi-model mean initial ocean temperature profile for the
/// Northern Hemisphere (50 layers, K).
///
/// Source: MAGICC7 `OCN_INITIAL_PROFILE_CMIP5MEAN.CFG`.
/// Layer 0 is the mixed layer (~19 K), layer 49 is the deepest (~1.2 K).
const CMIP5_PROFILE_NH: [FloatValue; 50] = [
    1.89503822e+01,
    1.58484640e+01,
    1.27692938e+01,
    1.11237631e+01,
    9.93378544e+00,
    8.89700890e+00,
    8.01173782e+00,
    7.24060631e+00,
    6.58022213e+00,
    5.99888515e+00,
    5.47700644e+00,
    5.02416515e+00,
    4.62269211e+00,
    4.27446032e+00,
    3.95875454e+00,
    3.70120311e+00,
    3.47130036e+00,
    3.26678157e+00,
    3.08187413e+00,
    2.93045211e+00,
    2.79141068e+00,
    2.66952801e+00,
    2.55478907e+00,
    2.44816899e+00,
    2.35198379e+00,
    2.26331019e+00,
    2.18005610e+00,
    2.10292435e+00,
    2.02744699e+00,
    1.95637441e+00,
    1.89118743e+00,
    1.82867718e+00,
    1.76954043e+00,
    1.71074319e+00,
    1.65469503e+00,
    1.60236323e+00,
    1.55269921e+00,
    1.50864816e+00,
    1.47147048e+00,
    1.44045138e+00,
    1.41173756e+00,
    1.38347185e+00,
    1.35783422e+00,
    1.33539736e+00,
    1.31498563e+00,
    1.29516900e+00,
    1.27472460e+00,
    1.25263810e+00,
    1.22954643e+00,
    1.20586693e+00,
];

/// CMIP5 multi-model mean initial ocean temperature profile for the
/// Southern Hemisphere (50 layers, K).
///
/// Source: MAGICC7 `OCN_INITIAL_PROFILE_CMIP5MEAN.CFG`.
const CMIP5_PROFILE_SH: [FloatValue; 50] = [
    1.62849369e+01,
    1.35041571e+01,
    1.10637445e+01,
    9.45342350e+00,
    8.30402851e+00,
    7.37928152e+00,
    6.60113478e+00,
    5.90550613e+00,
    5.29829597e+00,
    4.77080584e+00,
    4.31242418e+00,
    3.93976259e+00,
    3.62348270e+00,
    3.35576391e+00,
    3.11617875e+00,
    2.93644977e+00,
    2.77795982e+00,
    2.63738632e+00,
    2.50925493e+00,
    2.40222931e+00,
    2.30221725e+00,
    2.21322107e+00,
    2.12794638e+00,
    2.04543614e+00,
    1.96889246e+00,
    1.89580762e+00,
    1.82651293e+00,
    1.75886285e+00,
    1.69188118e+00,
    1.62586987e+00,
    1.56049752e+00,
    1.49373257e+00,
    1.42720032e+00,
    1.35796928e+00,
    1.28947854e+00,
    1.22542751e+00,
    1.16357803e+00,
    1.10515058e+00,
    1.05139232e+00,
    1.00322735e+00,
    9.58882809e-01,
    9.15422320e-01,
    8.75476420e-01,
    8.43416333e-01,
    8.16016912e-01,
    7.90101945e-01,
    7.68699825e-01,
    7.51805604e-01,
    7.36583769e-01,
    7.25481987e-01,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() {
        let params = ClimateUDEBParameters::default();

        // Check key defaults
        assert_eq!(params.n_layers, 50);
        assert!((params.ecs - 3.0).abs() < 1e-10);
        assert!((params.rf_2xco2 - 3.71).abs() < 1e-10);
        assert_eq!(params.steps_per_year, 12);
    }

    #[test]
    fn test_kappa_conversion() {
        let params = ClimateUDEBParameters::default();
        let kappa_m2_yr = params.kappa_m2_per_yr();

        // 0.75 cm^2/s * 3155.76 ~= 2366.82 m^2/yr
        assert!((kappa_m2_yr - 2366.82).abs() < 0.1);
    }

    #[test]
    fn test_lambda_global() {
        let params = ClimateUDEBParameters::default();
        let lambda = params.lambda_global();

        // 3.71 / 3.0 ~= 1.237 W/m^2/K
        assert!((lambda - 1.2367).abs() < 0.001);
    }

    #[test]
    fn test_area_fractions() {
        let params = ClimateUDEBParameters::default();

        // NH: 42% land, 58% ocean
        assert!((params.nh_ocean_fraction() - 0.58).abs() < 1e-10);

        // SH: 21% land, 79% ocean
        assert!((params.sh_ocean_fraction() - 0.79).abs() < 1e-10);

        // Global: average of hemispheres
        let global_ocean = params.global_ocean_fraction();
        let global_land = params.global_land_fraction();

        assert!((global_ocean + global_land - 1.0).abs() < 1e-10);
        assert!((global_ocean - 0.685).abs() < 1e-10);
    }

    #[test]
    fn test_mixed_layer_heat_capacity() {
        let params = ClimateUDEBParameters::default();
        let c_mix = params.mixed_layer_heat_capacity();

        // For 60m mixed layer, expect ~7.8 W yr / m^2 K
        // (1026 * 3985 * 60) / 31557600 ~= 7.77
        assert!(c_mix > 7.0 && c_mix < 8.5, "C_mix = {}", c_mix);
    }

    #[test]
    fn test_serialization() {
        let params = ClimateUDEBParameters::default();
        let json = serde_json::to_string(&params).expect("Serialization failed");
        let parsed: ClimateUDEBParameters =
            serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(params.n_layers, parsed.n_layers);
        assert!((params.ecs - parsed.ecs).abs() < 1e-10);
    }

    #[test]
    fn test_per_box_global_fractions() {
        let params = ClimateUDEBParameters::default();
        let (fgno, fgnl, fgso, fgsl) = params.global_box_fractions();

        assert!((fgnl - 0.21).abs() < 1e-10);
        assert!((fgno - 0.29).abs() < 1e-10);
        assert!((fgsl - 0.105).abs() < 1e-10);
        assert!((fgso - 0.395).abs() < 1e-10);
        assert!((fgno + fgnl + fgso + fgsl - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_varying_ecs_defaults() {
        let params = ClimateUDEBParameters::default();
        assert!((params.feedback_q_sensitivity - 7.84e-9).abs() < 1e-12);
        assert!((params.feedback_cumt_sensitivity - 0.08).abs() < 1e-10);
        assert!((params.feedback_cumt_period - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_ground_heat_capacity() {
        let params = ClimateUDEBParameters::default();
        let c_ground = params.ground_heat_capacity();

        // For 300m effective thickness, expect ~38.8 W yr / m^2 K
        // (1026 * 3985 * 300) / 31557600 ~= 38.8
        assert!(
            c_ground > 35.0 && c_ground < 42.0,
            "C_ground = {}",
            c_ground
        );

        // Ground capacity should scale linearly with thickness
        let ratio = c_ground / params.mixed_layer_heat_capacity();
        let expected_ratio = params.land_hc_eff_thickness / params.mixed_layer_depth;
        assert!(
            (ratio - expected_ratio).abs() < 1e-10,
            "Ground/mixed capacity ratio should equal thickness ratio: {} vs {}",
            ratio,
            expected_ratio
        );
    }

    #[test]
    fn test_ground_heat_defaults() {
        let params = ClimateUDEBParameters::default();
        assert!(params.land_heat_capacity_enabled);
        assert!((params.k_lg - 0.1).abs() < 1e-10);
        assert!((params.land_hc_eff_thickness - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_partial_deserialization() {
        // Test that #[serde(default)] allows partial deserialization
        let json = r#"{"ecs": 4.5}"#;
        let params: ClimateUDEBParameters =
            serde_json::from_str(json).expect("Partial deserialization failed");

        // ecs should be from JSON
        assert!((params.ecs - 4.5).abs() < 1e-10);

        // Other fields should be defaults
        assert_eq!(params.n_layers, 50);
        assert!((params.kappa - 0.75).abs() < 1e-10);
        assert!((params.rf_2xco2 - 3.71).abs() < 1e-10);
    }
}
