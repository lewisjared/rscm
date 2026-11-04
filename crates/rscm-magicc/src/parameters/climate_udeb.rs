//! Climate UDEB Parameters
//!
//! Parameters for the Upwelling-Diffusion Energy Balance (UDEB) climate model
//! following MAGICC7.

use super::ThermalExpansionParameters;
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
    CMIP5 = 2,
}

/// Box forcing given at every substep, such as monthly volcanic forcing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubannualForcing {
    /// Time of the start of the first substep (decimal year).
    pub start: FloatValue,
    /// Values per box (NH ocean, NH land, SH ocean, SH land), one per substep
    /// with `steps_per_year` substeps a year. Zero outside the given range.
    pub values: Vec<[FloatValue; 4]>,
    /// Regional pattern whose internal efficacy divides `efficacy` for the values of each year,
    /// taken from the climate sensitivity of the year before.
    /// A substep belongs to the year it ends in.
    #[serde(default)]
    pub rf_regions: Option<[FloatValue; 4]>,
    /// Prescribed efficacy over the internal efficacy when `rf_regions` is set.
    #[serde(default = "one")]
    pub efficacy: FloatValue,
    /// Internal efficacy dividing the values of the first two years when `rf_regions` is set.
    #[serde(default = "one")]
    pub internal_efficacy: FloatValue,
}

fn one() -> FloatValue {
    1.0
}

impl SubannualForcing {
    /// Forcing for substep `step` (1-based) of the year starting at `year_start`.
    pub fn at(
        &self,
        year_start: FloatValue,
        step: usize,
        steps_per_year: usize,
    ) -> [FloatValue; 4] {
        let index = ((year_start - self.start) * steps_per_year as FloatValue).round() as i64
            + step as i64
            - 1;
        usize::try_from(index)
            .ok()
            .and_then(|i| self.values.get(i))
            .copied()
            .unwrap_or([0.0; 4])
    }
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
    /// - `1` = unit efficacy: effective forcing divided by the internal efficacy.
    /// - `2` = AR6 mode: effective forcing adjusted by `prescribed_efficacy_co2 / internal_efficacy`.
    #[serde(default)]
    pub efficacy_apply: u8,

    /// Prescribed CO2 efficacy factor (matches MAGICC7 `RF_EFFICACY_CO2`).
    ///
    /// Unused in mode 1.
    /// In mode 2: $\text{EFFRF} = \text{ERF} \times \text{prescribed\_efficacy\_co2} / \text{internal\_efficacy}$.
    ///
    /// Default: 1.0 (no-op).
    #[serde(default = "default_prescribed_efficacy_co2")]
    pub prescribed_efficacy_co2: FloatValue,

    /// Read efficacy-adjusted four-box forcing instead of global ERF.
    ///
    /// When set, the forcing is used as given and the efficacy settings above
    /// are ignored. See `ForcingAggregation`.
    pub regional_forcing: bool,

    /// Efficacy-adjusted box forcing applied at each substep, on top of the
    /// annual input. MAGICC7 applies volcanic forcing this way.
    pub subannual_forcing: Option<SubannualForcing>,

    /// Write the annual mean warming and the thermal expansion of the ocean.
    pub thermal_expansion: Option<ThermalExpansionParameters>,

    /// Read the CO2 doubling ERF each year instead of using `rf_2xco2`.
    ///
    /// MAGICC7 recalculates it from the current concentrations with OLBL,
    /// which changes the feedback parameters as CO2 rises.
    pub time_varying_q2x: bool,

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
            regional_forcing: false,
            time_varying_q2x: false,
            subannual_forcing: None,
            thermal_expansion: None,

            // Initial ocean profile
            ocean_temp_profile: OceanTempProfile::CMIP5,

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
///
/// MAGICC7's `SPECHT` (0.9333 cal/g/K) times `HTCONS` (4.1856 J/cal).
pub const CP_SEAWATER: FloatValue = 0.9333 * 4.1856 * 1000.0;

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

    /// Diffusivity change per kelvin of top to bottom temperature difference at each layer boundary.
    pub fn kappa_slopes(&self) -> Vec<FloatValue> {
        let n = self.n_layers;
        // MAGICC7 scales by boundary index, not depth, so the thicker mixed layer is ignored.
        (0..n - 1)
            .map(|l| (1.0 - l as FloatValue / (n as FloatValue - 1.0)) * self.kappa_dkdt)
            .collect()
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

    /// Relative ocean area of a hemisphere at depth `depth_m`.
    ///
    /// Interpolates the CMIP5 mean profile of `OCN_REF_AREAFRAC_PROFILE_CMIP5MEAN.CFG`,
    /// blended towards a cylindrical ocean by
    /// [`depth_dependent_area`](Self::depth_dependent_area).
    pub fn ocean_area_at_depth(&self, hemi: usize, depth_m: FloatValue) -> FloatValue {
        let area = if hemi == 0 {
            &OCEAN_AREA_NH
        } else {
            &OCEAN_AREA_SH
        };
        let d = &OCEAN_AREA_DEPTHS;
        let raw = match d.windows(2).position(|w| w[0] <= depth_m && depth_m < w[1]) {
            Some(i) => (area[i + 1] - area[i]) * (depth_m - d[i]) / (d[i + 1] - d[i]) + area[i],
            None if depth_m < d[0] => area[0],
            None => area[area.len() - 1],
        };
        (1.0 - (1.0 - raw) * self.depth_dependent_area).clamp(0.0, 1.0)
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
    pub fn compute_area_factors(
        &self,
        hemi: usize,
    ) -> (Vec<FloatValue>, Vec<FloatValue>, Vec<FloatValue>) {
        let n = self.n_layers;
        let mut af_top = Vec::with_capacity(n);
        let mut af_bottom = Vec::with_capacity(n);
        let mut af_diff = Vec::with_capacity(n);

        for l in 0..n {
            let (z_top, z_bottom) = if l == 0 {
                (0.0, self.mixed_layer_depth)
            } else {
                let top = self.mixed_layer_depth + (l as FloatValue - 1.0) * self.layer_thickness;
                (top, top + self.layer_thickness)
            };

            let a_top = self.ocean_area_at_depth(hemi, z_top);
            let a_bottom = self.ocean_area_at_depth(hemi, z_bottom);
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
    /// When `ocean_temp_profile` is [`OceanTempProfile::CMIP5`] (default),
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
            OceanTempProfile::CMIP5 => {
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

/// Depths (m) of the CMIP5 mean ocean area profiles.
const OCEAN_AREA_DEPTHS: [FloatValue; 41] = [
    0.0, 30.0, 110.0, 210.0, 310.0, 410.0, 510.0, 610.0, 710.0, 810.0, 910.0, 1010.0, 1110.0,
    1210.0, 1310.0, 1410.0, 1510.0, 1610.0, 1710.0, 1810.0, 1910.0, 2010.0, 2110.0, 2210.0, 2310.0,
    2410.0, 2510.0, 3010.0, 3510.0, 4010.0, 4510.0, 5010.0, 5510.0, 6010.0, 6510.0, 7010.0, 7510.0,
    8010.0, 8510.0, 9010.0, 9510.0,
];

/// NH ocean area relative to the surface (`OCN_AREAS_RAW_RELATIVE_PRESCR_NH`).
const OCEAN_AREA_NH: [FloatValue; 41] = [
    1.00000000e+00,
    9.92768049e-01,
    9.33580875e-01,
    9.05554295e-01,
    8.88706923e-01,
    8.77777398e-01,
    8.69542181e-01,
    8.62955093e-01,
    8.57065380e-01,
    8.51579487e-01,
    8.46125364e-01,
    8.40779603e-01,
    8.35111797e-01,
    8.29301596e-01,
    8.23629618e-01,
    8.17975044e-01,
    8.12225044e-01,
    8.06298792e-01,
    8.00089359e-01,
    7.93866634e-01,
    7.87548840e-01,
    7.81117976e-01,
    7.74461150e-01,
    7.67656147e-01,
    7.60640740e-01,
    7.53213227e-01,
    7.45444179e-01,
    6.98505878e-01,
    6.27339005e-01,
    5.30937970e-01,
    4.15673941e-01,
    2.88527399e-01,
    1.59321889e-01,
    5.03621809e-02,
    2.64205802e-02,
    2.63157897e-02,
    2.63157897e-02,
    2.63157897e-02,
    2.63157897e-02,
    2.63157897e-02,
    2.63157897e-02,
];

/// SH ocean area relative to the surface (`OCN_AREAS_RAW_RELATIVE_PRESCR_SH`).
const OCEAN_AREA_SH: [FloatValue; 41] = [
    1.00000000e+00,
    9.98092234e-01,
    9.82268870e-01,
    9.75749671e-01,
    9.71684515e-01,
    9.67262864e-01,
    9.62361991e-01,
    9.57554638e-01,
    9.53565300e-01,
    9.50394392e-01,
    9.47427332e-01,
    9.44458604e-01,
    9.41365719e-01,
    9.38220859e-01,
    9.34938848e-01,
    9.31435645e-01,
    9.27700162e-01,
    9.23839390e-01,
    9.19681549e-01,
    9.15324748e-01,
    9.10636783e-01,
    9.05802965e-01,
    9.00624514e-01,
    8.95213604e-01,
    8.89417052e-01,
    8.82735789e-01,
    8.75507176e-01,
    8.21786106e-01,
    7.18526065e-01,
    5.64145565e-01,
    3.78261954e-01,
    2.01682448e-01,
    4.64582741e-02,
    2.65591927e-02,
    2.63590403e-02,
    2.63157897e-02,
    2.63157897e-02,
    2.63157897e-02,
    2.63157897e-02,
    2.63157897e-02,
    2.63157897e-02,
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
        // (1026 * 3906 * 60) / 31557600 ~= 7.62
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

        // For 300m effective thickness, expect ~38.1 W yr / m^2 K
        // (1026 * 3906 * 300) / 31557600 ~= 38.1
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
