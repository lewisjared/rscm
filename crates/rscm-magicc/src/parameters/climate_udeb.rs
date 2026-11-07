//! Climate UDEB Parameters
//!
//! Parameters for the Upwelling-Diffusion Energy Balance (UDEB) climate model
//! following MAGICC7.

use super::ThermalExpansion;
use rscm_core::spatial::FourBoxGrid;
use rscm_core::timeseries::FloatValue;
use serde::{Deserialize, Serialize};

/// Initial ocean temperature profile (MAGICC7 `CORE_SWITCH_OCN_TEMPPROFILE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OceanTemperatureProfile {
    /// Analytical exponential decay from uniform diffusivity and upwelling (MAGICC7 1).
    Analytical,
    /// CMIP5 multi-model mean observational profile (MAGICC7 2, the default).
    #[serde(rename = "cmip5")]
    CMIP5,
}

/// How efficacies and internal efficacies scale an agent's forcing (MAGICC7 `RF_EFFICACY_APPLY`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EfficacyMethod {
    /// Forcing is left unchanged (MAGICC7 0).
    #[default]
    None,
    /// Forcing is divided by the internal efficacy, so every agent has unit efficacy (MAGICC7 1).
    Unit,
    /// Forcing is scaled by the efficacy over the internal efficacy, as in AR6 (MAGICC7 2).
    #[serde(rename = "ar6")]
    AR6,
}

/// Box forcing given at every substep, such as monthly volcanic forcing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubannualForcing {
    /// Time of the start of the first substep (decimal year).
    pub start: FloatValue,
    /// Values per box (NH ocean, NH land, SH ocean, SH land), one per substep
    /// with `steps_per_year` substeps a year. Zero outside the given range.
    pub values: Vec<[FloatValue; 4]>,
    /// Regional pattern whose internal efficacy divides `efficacy` for the values of each year,
    /// taken from the equilibrium climate sensitivity of the year before.
    /// A substep belongs to the year it ends in.
    #[serde(default)]
    pub regional_pattern: Option<[FloatValue; 4]>,
    /// Efficacy over the internal efficacy when `regional_pattern` is set.
    #[serde(default = "one")]
    pub efficacy: FloatValue,
    /// Internal efficacy dividing the values of the first two years when `regional_pattern` is set.
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

/// Parameters for the four-box UDEB climate component.
///
/// `ClimateUDEB` couples a four-box atmosphere (Northern Ocean, Northern Land,
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
/// Default values are the MAGICC7 defaults from `MAGCFG_DEFAULTALL.CFG`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct ClimateUDEBParameters {
    // Ocean structure parameters
    /// Number of ocean layers, including the mixed layer
    ///
    /// Default: 50
    pub n_layers: usize,

    /// Mixed layer depth
    ///
    /// unit: m
    ///
    /// Default: 60.0
    pub mixed_layer_depth: FloatValue,

    /// Thickness of each deeper ocean layer
    ///
    /// unit: m
    ///
    /// Default: 100.0
    pub layer_thickness: FloatValue,

    // Diffusivity parameters
    /// Base vertical diffusivity.
    ///
    /// unit: cm^2/s
    ///
    /// Converted to m^2/yr internally with the factor 3155.76.
    ///
    /// Default: 0.75
    pub vertical_diffusivity: FloatValue,

    /// Minimum vertical diffusivity, the floor for the temperature-dependent diffusivity
    ///
    /// unit: cm^2/s
    ///
    /// Default: 0.1
    pub minimum_vertical_diffusivity: FloatValue,

    /// Change in diffusivity with the vertical temperature gradient
    ///
    /// unit: cm^2/s/K
    ///
    /// Default: -0.191
    pub vertical_diffusivity_gradient_sensitivity: FloatValue,

    // Upwelling parameters
    /// Initial upwelling rate, before the temperature-dependent adjustment
    /// (MAGICC7 `CORE_INITIAL_UPWELLING_RATE`)
    ///
    /// unit: m/yr
    ///
    /// Default: 3.5
    /// MAGICC7: CORE_INITIAL_UPWELLING_RATE
    pub initial_upwelling_rate: FloatValue,

    /// Fraction of upwelling that varies with temperature (MAGICC7 `CORE_UPWELLING_VARIABLE_PART`)
    ///
    /// unit: dimensionless
    ///
    /// Default: 0.7
    /// MAGICC7: CORE_UPWELLING_VARIABLE_PART
    pub upwelling_variable_fraction: FloatValue,

    /// Temperature change at which NH upwelling reaches its minimum (MAGICC7 `CORE_UPWELL_THRESH_TEMP_NH`)
    ///
    /// unit: K
    ///
    /// Default: 8.0
    /// MAGICC7: CORE_UPWELL_THRESH_TEMP_NH
    pub upwelling_threshold_temperature_nh: FloatValue,

    /// Temperature change at which SH upwelling reaches its minimum (MAGICC7 `CORE_UPWELL_THRESH_TEMP_SH`)
    ///
    /// unit: K
    ///
    /// Default: 8.0
    /// MAGICC7: CORE_UPWELL_THRESH_TEMP_SH
    pub upwelling_threshold_temperature_sh: FloatValue,

    // Climate sensitivity parameters
    /// Equilibrium climate sensitivity, the equilibrium warming after CO2 doubles
    ///
    /// unit: K
    ///
    /// Default: 3.0
    pub ecs: FloatValue,

    /// 2xCO2 effective radiative forcing
    ///
    /// unit: W/m^2
    ///
    /// Default: 3.71
    pub erf_2xco2: FloatValue,

    /// Ratio of land to ocean equilibrium warming (MAGICC7 `CORE_RLO`)
    ///
    /// unit: dimensionless
    ///
    /// Default: 1.317
    /// MAGICC7: CORE_RLO
    pub land_ocean_warming_ratio: FloatValue,

    /// Change in equilibrium climate sensitivity with the ERF level (MAGICC7 `CORE_FEEDBACK_QSENSITIVITY`)
    ///
    /// unit: dimensionless
    ///
    /// Default: 7.84e-9
    /// MAGICC7: CORE_FEEDBACK_QSENSITIVITY
    pub feedback_forcing_sensitivity: FloatValue,

    /// Change in equilibrium climate sensitivity with cumulative warming
    /// (MAGICC7 `CORE_FEEDBACK_CUMTSENSITIVITY`)
    ///
    /// unit: dimensionless
    ///
    /// Default: 0.08
    /// MAGICC7: CORE_FEEDBACK_CUMTSENSITIVITY
    pub feedback_cumulative_temperature_sensitivity: FloatValue,

    /// Period over which warming is accumulated (MAGICC7 `CORE_FEEDBACK_CUMTPERIOD`)
    ///
    /// unit: yr
    ///
    /// Default: 300.0
    /// MAGICC7: CORE_FEEDBACK_CUMTPERIOD
    pub feedback_cumulative_temperature_period: FloatValue,

    // Heat exchange parameters
    /// Land-ocean heat exchange coefficient
    ///
    /// unit: W/m^2/K
    ///
    /// Default: 1.44
    pub land_ocean_heat_exchange: FloatValue,

    /// Inter-hemispheric (north-south) heat exchange coefficient
    ///
    /// unit: W/m^2/K
    ///
    /// Default: 0.31
    pub north_south_heat_exchange: FloatValue,

    /// Ocean-to-land heat exchange amplification factor
    ///
    /// unit: dimensionless
    ///
    /// Default: 1.02
    pub amplify_ocean_to_land: FloatValue,

    // Area fractions
    /// Northern Hemisphere land fraction
    ///
    /// unit: dimensionless
    ///
    /// Default: 0.42
    pub nh_land_fraction: FloatValue,

    /// Southern Hemisphere land fraction
    ///
    /// unit: dimensionless
    ///
    /// Default: 0.21
    pub sh_land_fraction: FloatValue,

    /// Strength of the ocean basin narrowing with depth, from 0 to 1.
    ///
    /// unit: dimensionless
    ///
    /// 0 is a cylindrical ocean with no narrowing and 1 is the full hypsometric profile.
    ///
    /// Default: 1.0
    pub depth_dependent_area: FloatValue,

    // Ocean temperature adjustment
    /// Linear coefficient of the air temperature to SST relationship
    /// (MAGICC7 `CORE_TEMPADJUST_OCN2ATM_ALPHA`)
    ///
    /// unit: dimensionless
    ///
    /// Default: 1.04
    /// MAGICC7: CORE_TEMPADJUST_OCN2ATM_ALPHA
    pub temperature_adjustment_alpha: FloatValue,

    /// Quadratic coefficient of the air temperature to SST relationship
    /// (MAGICC7 `CORE_TEMPADJUST_OCN2ATM_GAMMA`)
    ///
    /// unit: K^-1
    ///
    /// Default: -0.002
    /// MAGICC7: CORE_TEMPADJUST_OCN2ATM_GAMMA
    pub temperature_adjustment_gamma: FloatValue,

    // Polar sinking
    /// Fraction of the surface temperature change carried by polar sinking water
    ///
    /// unit: dimensionless
    ///
    /// Default: 0.2
    pub polar_sinking_ratio: FloatValue,

    // Ground heat reservoir parameters
    /// Apply the ground heat reservoir (MAGICC7 `CORE_LANDHEATCAPACITY_APPLY`).
    ///
    /// The reservoir damps the land temperature response by absorbing and releasing heat
    /// from the land surface.
    ///
    /// Default: true
    pub include_land_heat_capacity: bool,

    /// Land-ground heat exchange coefficient, per unit globe area (MAGICC7 `CORE_HEATXCHANGE_LANDGROUND`).
    ///
    /// unit: W/m^2/K
    ///
    /// Sets the rate of heat exchange between the land surface and the ground heat reservoir.
    ///
    /// Default: 0.1
    pub land_ground_heat_exchange: FloatValue,

    /// Effective thickness of the ground heat reservoir (MAGICC7 `CORE_LANDHC_EFFTHICKNESS`).
    ///
    /// unit: m
    ///
    /// The ocean-water-equivalent depth that sets the thermal inertia of the land heat reservoir.
    ///
    /// Default: 300.0
    pub land_heat_capacity_effective_thickness: FloatValue,

    // Regional CO2 forcing
    /// Regional CO2 ERF pattern in each region of the four-box grid (MAGICC7 `RF_REGIONS_CO2`).
    ///
    /// unit: dimensionless
    ///
    /// Relative weights for the CO2 ERF fraction of each region in the LAMCALC solver,
    /// normalised by their global area-weighted sum as in MAGICC7.
    ///
    /// Default values from `MAGCFG_DEFAULTALL.CFG` (AR6 calibration):
    /// `[1.4089, 1.37045, 1.43333, 1.33257]`.
    pub co2_regional_pattern: [FloatValue; 4],

    // Efficacy parameters
    /// How efficacies scale the ERF (MAGICC7 `RF_EFFICACY_APPLY`).
    ///
    /// `none` leaves the ERF unchanged, `unit` divides it by the internal efficacy,
    /// and `ar6` scales it by `co2_efficacy / internal_efficacy`.
    #[serde(default)]
    pub efficacy_method: EfficacyMethod,

    /// CO2 efficacy (MAGICC7 `RF_EFFICACY_CO2`).
    ///
    /// unit: dimensionless
    ///
    /// Used only by the `ar6` method:
    /// $\text{EFFRF} = \text{ERF} \times \text{co2\_efficacy} / \text{internal\_efficacy}$.
    ///
    /// Default: 1.0 (no-op).
    #[serde(default = "default_co2_efficacy")]
    pub co2_efficacy: FloatValue,

    /// Read the four-box efficacy-adjusted ERF from `ForcingAggregation` instead of the global ERF.
    ///
    /// The ERF is then used as given and the efficacy parameters above are ignored.
    pub include_regional_forcing: bool,

    /// Efficacy-adjusted regional ERF applied at each substep, on top of the annual input.
    ///
    /// MAGICC7 applies volcanic forcing this way.
    pub subannual_forcing: Option<SubannualForcing>,

    /// Write the annual mean warming and the thermal expansion of the ocean
    pub thermal_expansion: Option<ThermalExpansion>,

    /// Read `2xCO2 Effective Radiative Forcing` each year instead of using `erf_2xco2`.
    ///
    /// MAGICC7 recalculates it from the current concentrations with OLBL,
    /// which changes the feedback parameters as CO2 rises.
    pub time_varying_erf_2xco2: bool,

    // Initial ocean temperature profile
    /// Initial ocean temperature profile (MAGICC7 `CORE_SWITCH_OCN_TEMPPROFILE`).
    ///
    /// The profile is the reference for the variable upwelling correction terms.
    /// When upwelling changes from its initial value, these corrections compensate
    /// for the shift in the equilibrium profile.
    /// MAGICC7 uses the CMIP5 profile by default.
    pub ocean_temperature_profile: OceanTemperatureProfile,

    // Integration parameters
    /// Substeps per year, 12 (monthly) by default
    ///
    /// Default: 12 (monthly)
    pub steps_per_year: usize,

    /// Cap on the temperature change, to prevent runaway temperatures (MAGICC7 `CORE_MAXIMAL_TEMPERATURE`)
    ///
    /// unit: K
    ///
    /// Default: 25.0
    /// MAGICC7: CORE_MAXIMAL_TEMPERATURE
    pub maximum_temperature: FloatValue,
}

impl Default for ClimateUDEBParameters {
    fn default() -> Self {
        Self {
            // Ocean structure
            n_layers: 50,
            mixed_layer_depth: 60.0,
            layer_thickness: 100.0,

            // Diffusivity
            vertical_diffusivity: 0.75,
            minimum_vertical_diffusivity: 0.1,
            vertical_diffusivity_gradient_sensitivity: -0.191,

            // Upwelling
            initial_upwelling_rate: 3.5,
            upwelling_variable_fraction: 0.7,
            upwelling_threshold_temperature_nh: 8.0,
            upwelling_threshold_temperature_sh: 8.0,

            // Climate sensitivity
            ecs: 3.0,
            erf_2xco2: 3.71,
            land_ocean_warming_ratio: 1.317,
            feedback_forcing_sensitivity: 7.84e-9,
            feedback_cumulative_temperature_sensitivity: 0.08,
            feedback_cumulative_temperature_period: 300.0,

            // Heat exchange
            land_ocean_heat_exchange: 1.44,
            north_south_heat_exchange: 0.31,
            amplify_ocean_to_land: 1.02,

            // Area fractions
            nh_land_fraction: 0.42,
            sh_land_fraction: 0.21,
            depth_dependent_area: 1.0,

            // Temperature adjustment
            temperature_adjustment_alpha: 1.04,
            temperature_adjustment_gamma: -0.002,

            // Polar sinking
            polar_sinking_ratio: 0.2,

            // Ground heat reservoir
            include_land_heat_capacity: true,
            land_ground_heat_exchange: 0.1,
            land_heat_capacity_effective_thickness: 300.0,

            // Regional CO2 forcing (MAGCFG_DEFAULTALL.CFG)
            co2_regional_pattern: [1.4089, 1.37045, 1.43333, 1.33257],

            // Efficacy
            efficacy_method: EfficacyMethod::None,
            co2_efficacy: 1.0,
            include_regional_forcing: false,
            time_varying_erf_2xco2: false,
            subannual_forcing: None,
            thermal_expansion: None,

            // Initial ocean profile
            ocean_temperature_profile: OceanTemperatureProfile::CMIP5,

            // Integration
            steps_per_year: 12,
            maximum_temperature: 25.0,
        }
    }
}

fn default_co2_efficacy() -> FloatValue {
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

/// Earth surface area ($\text{m}^2$), MAGICC7's value.
pub const EARTH_SURFACE_AREA: FloatValue = 5.101e14;

/// Volumetric heat capacity of seawater per unit depth ($\text{W yr / m}^2\text{ K}$).
///
/// $$\frac{\rho \, c_p \, d}{\text{seconds per year}}$$
pub fn heat_capacity_per_unit_area(depth_m: FloatValue) -> FloatValue {
    RHO_SEAWATER * CP_SEAWATER * depth_m / SECONDS_PER_YEAR
}

impl ClimateUDEBParameters {
    /// Convert vertical diffusivity from $\text{cm}^2/\text{s}$ to $\text{m}^2/\text{yr}$.
    pub fn kappa_m2_per_yr(&self) -> FloatValue {
        self.vertical_diffusivity * DIFFUSIVITY_CM2S_TO_M2YR
    }

    /// Convert minimum vertical diffusivity from $\text{cm}^2/\text{s}$ to $\text{m}^2/\text{yr}$.
    pub fn kappa_min_m2_per_yr(&self) -> FloatValue {
        self.minimum_vertical_diffusivity * DIFFUSIVITY_CM2S_TO_M2YR
    }

    /// Diffusivity change per kelvin of top to bottom temperature difference at each layer boundary.
    pub fn kappa_slopes(&self) -> Vec<FloatValue> {
        let n = self.n_layers;
        // MAGICC7 scales by boundary index, not depth, so the thicker mixed layer is ignored.
        (0..n - 1)
            .map(|l| {
                (1.0 - l as FloatValue / (n as FloatValue - 1.0))
                    * self.vertical_diffusivity_gradient_sensitivity
            })
            .collect()
    }

    /// Get the global climate feedback parameter ($\text{W/m}^2\text{/K}$).
    ///
    /// $$\lambda = \Delta Q_{2x} / \text{ECS}$$
    pub fn lambda_global(&self) -> FloatValue {
        self.erf_2xco2 / self.ecs
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

    /// Four-box grid weighted by the global area fraction of each box.
    ///
    /// The weights are fractions of the total globe, not of each hemisphere.
    /// MAGICC7 stores them as GLOBALAREAFRACTIONS (FGNO, FGNL, FGSO, FGSL).
    pub fn grid(&self) -> FourBoxGrid {
        let nh_land = self.nh_land_fraction / 2.0;
        let sh_land = self.sh_land_fraction / 2.0;
        FourBoxGrid::with_weights([0.5 - nh_land, nh_land, 0.5 - sh_land, sh_land])
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
        heat_capacity_per_unit_area(self.land_heat_capacity_effective_thickness)
    }

    /// Compute the initial ocean temperature profile for a given hemisphere.
    ///
    /// Returns a Vec of length `n_layers` with the initial temperature (K)
    /// for each ocean layer. Layer 0 is the mixed layer.
    ///
    /// When `ocean_temperature_profile` is [`OceanTemperatureProfile::CMIP5`] (default),
    /// returns the CMIP5 multi-model mean profile. When
    /// [`OceanTemperatureProfile::Analytical`], returns the analytical exponential
    /// decay profile.
    pub fn initial_ocean_profile(&self, hemi: usize) -> Vec<FloatValue> {
        assert!(
            hemi < 2,
            "initial_ocean_profile: hemi must be 0 (NH) or 1 (SH), got {}",
            hemi
        );

        match self.ocean_temperature_profile {
            OceanTemperatureProfile::CMIP5 => {
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
            OceanTemperatureProfile::Analytical => {
                let t_mix = 17.2_f64;
                let t_polar = 1.0_f64;
                let kappa = self.kappa_m2_per_yr();
                let w = self.initial_upwelling_rate;
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
        assert!((params.erf_2xco2 - 3.71).abs() < 1e-10);
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
    fn test_serialisation() {
        let params = ClimateUDEBParameters::default();
        let json = serde_json::to_string(&params).expect("Serialisation failed");
        let parsed: ClimateUDEBParameters =
            serde_json::from_str(&json).expect("Deserialisation failed");

        assert_eq!(params.n_layers, parsed.n_layers);
        assert!((params.ecs - parsed.ecs).abs() < 1e-10);
    }

    #[test]
    fn test_per_box_global_fractions() {
        let params = ClimateUDEBParameters::default();
        let [nh_ocean, nh_land, sh_ocean, sh_land] = *params.grid().weights();

        assert!((nh_land - 0.21).abs() < 1e-10);
        assert!((nh_ocean - 0.29).abs() < 1e-10);
        assert!((sh_land - 0.105).abs() < 1e-10);
        assert!((sh_ocean - 0.395).abs() < 1e-10);
        assert!((nh_ocean + nh_land + sh_ocean + sh_land - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_varying_ecs_defaults() {
        let params = ClimateUDEBParameters::default();
        assert!((params.feedback_forcing_sensitivity - 7.84e-9).abs() < 1e-12);
        assert!((params.feedback_cumulative_temperature_sensitivity - 0.08).abs() < 1e-10);
        assert!((params.feedback_cumulative_temperature_period - 300.0).abs() < 1e-10);
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
        let expected_ratio =
            params.land_heat_capacity_effective_thickness / params.mixed_layer_depth;
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
        assert!(params.include_land_heat_capacity);
        assert!((params.land_ground_heat_exchange - 0.1).abs() < 1e-10);
        assert!((params.land_heat_capacity_effective_thickness - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_partial_deserialisation() {
        // Test that #[serde(default)] allows partial deserialisation
        let json = r#"{"ecs": 4.5}"#;
        let params: ClimateUDEBParameters =
            serde_json::from_str(json).expect("Partial deserialisation failed");

        // ecs should be from JSON
        assert!((params.ecs - 4.5).abs() < 1e-10);

        // Other fields should be defaults
        assert_eq!(params.n_layers, 50);
        assert!((params.vertical_diffusivity - 0.75).abs() < 1e-10);
        assert!((params.erf_2xco2 - 3.71).abs() < 1e-10);
    }
}
