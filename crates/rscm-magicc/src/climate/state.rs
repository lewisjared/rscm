//! Internal state for the ClimateUDEB component.
//!
//! This module contains the private state structures that persist across
//! solve() calls, including ocean layer temperatures, upwelling rates,
//! and temperature history for time-varying ECS.

use std::any::Any;

use rscm_core::component::ComponentState;
use rscm_core::timeseries::FloatValue;
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

    /// Effective SST-to-air-temperature ratio for each hemisphere.
    /// Computed once per annual timestep from the end-of-year SST, then held
    /// fixed for all monthly substeps of the following year (MAGICC7 behaviour).
    /// Index 0 = NH, 1 = SH.
    #[serde(default)]
    pub alpha_eff: [FloatValue; 2],

    /// Inter-hemispheric heat exchange (W/m^2) from the previous substep.
    /// Persists across annual timesteps (MAGICC7 module-level variable,
    /// initialized once at startup in climate_and_ocean.f90 line 140).
    /// Index 0 = NH, 1 = SH.
    #[serde(default)]
    pub hemi_heat_exchange: [FloatValue; 2],

    /// Initial equilibrium ocean temperature profile (K).
    ///
    /// Computed once at initialization using an exponential profile
    /// (MAGICC7 CORE_SWITCH_OCN_TEMPPROFILE=1). Index 0 is the mixed layer.
    /// Used to compute variable upwelling correction terms in the tridiagonal RHS.
    #[serde(default)]
    pub initial_ocean_profile: Vec<FloatValue>,

    /// Initial temperature of polar sinking water (K).
    /// MAGICC7 default: 1.0 K.
    #[serde(default = "default_polar_sinking_temp")]
    pub polar_sinking_temp: FloatValue,

    /// Initial mixed-layer temperature for the exponential profile (K).
    /// MAGICC7 default: 17.2 K.
    #[serde(default = "default_mixed_layer_initial_temp")]
    pub mixed_layer_initial_temp: FloatValue,
}

fn default_polar_sinking_temp() -> FloatValue {
    1.0
}

fn default_mixed_layer_initial_temp() -> FloatValue {
    17.2
}

impl ClimateUDEBState {
    /// Create a new state for a given number of layers and initial conditions.
    ///
    /// `alpha_initial` is the SST-to-air-temperature ratio used before any
    /// ocean temperature has developed (i.e. `temp_adjust_alpha` from parameters).
    /// `kappa_m2_per_yr` is the base vertical diffusivity in m2/yr used to
    /// compute the exponential initial ocean profile.
    /// `layer_thickness` is the thickness of deep layers in metres.
    /// `area_factors` is accepted for forward compatibility but the initial
    /// profile always uses the analytical exponential (MAGICC7
    /// `CORE_SWITCH_OCN_TEMPPROFILE=1`). The variable upwelling correction
    /// terms in `ocean_column.rs` are calibrated against this profile.
    /// `polar_sinking_ratio` is accepted for forward compatibility.
    pub fn new(
        n_layers: usize,
        w_initial: FloatValue,
        alpha_initial: FloatValue,
        kappa_m2_per_yr: FloatValue,
        layer_thickness: FloatValue,
        _area_factors: &OceanAreaFactors,
        _polar_sinking_ratio: FloatValue,
    ) -> Self {
        let t_mix = 17.2_f64;
        let t_polar = 1.0_f64;

        // Exponential decay profile (MAGICC7 CORE_SWITCH_OCN_TEMPPROFILE=1).
        // This is the analytical steady-state for a cylindrical ocean with
        // uniform diffusivity and upwelling. The variable upwelling correction
        // terms use this profile as a reference baseline — they already apply
        // area factors to the transport coefficients, so using an area-factor-
        // aware equilibrium here would double-count the geometry.
        let mut initial_ocean_profile = vec![0.0; n_layers];
        initial_ocean_profile[0] = t_mix;
        for l in 1..n_layers {
            let depth = (l as f64 - 1.0) * layer_thickness + 0.5 * layer_thickness;
            initial_ocean_profile[l] =
                t_polar + (t_mix - t_polar) * (-w_initial * depth / kappa_m2_per_yr).exp();
        }

        Self {
            ocean_temps: vec![vec![0.0; n_layers]; 2],
            upwelling_rates: [w_initial; 2],
            initialized: true,
            temperature_history: Vec::new(),
            dt_history: Vec::new(),
            land_temps: [0.0; 2],
            ground_temps: [0.0; 2],
            alpha_eff: [alpha_initial; 2],
            hemi_heat_exchange: [0.0; 2],
            initial_ocean_profile,
            polar_sinking_temp: t_polar,
            mixed_layer_initial_temp: t_mix,
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
