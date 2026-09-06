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

    /// Initial equilibrium ocean temperature profile per hemisphere (K).
    ///
    /// Shape: `[hemisphere][layer]`, where hemisphere 0 = NH, 1 = SH.
    /// Layer 0 is the mixed layer.
    ///
    /// By default uses the CMIP5 multi-model mean profile
    /// (`CORE_SWITCH_OCN_TEMPPROFILE=2`). Falls back to an analytical
    /// exponential profile when mode 1 is selected.
    ///
    /// Used to compute variable upwelling correction terms in the tridiagonal RHS.
    #[serde(default)]
    pub initial_ocean_profile: [Vec<FloatValue>; 2],

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
    /// `profiles` is the per-hemisphere initial ocean temperature profiles,
    /// computed by [`ClimateUDEBParameters::initial_ocean_profile()`].
    pub fn new(
        n_layers: usize,
        w_initial: FloatValue,
        alpha_initial: FloatValue,
        profiles: [Vec<FloatValue>; 2],
    ) -> Self {
        assert!(
            n_layers >= 2,
            "ClimateUDEBState::new requires n_layers >= 2, got {}",
            n_layers
        );
        for (i, hemi_profile) in profiles.iter().enumerate() {
            assert!(
                hemi_profile.len() == n_layers,
                "ClimateUDEBState::new requires hemisphere {} profile to have {} layers, got {}",
                i,
                n_layers,
                hemi_profile.len()
            );
        }

        let t_polar = 1.0_f64;
        let t_mix = profiles[0][0];

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
            initial_ocean_profile: profiles,
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
