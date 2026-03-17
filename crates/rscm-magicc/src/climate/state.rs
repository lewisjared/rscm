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
}

impl ClimateUDEBState {
    /// Create a new state for a given number of layers and initial upwelling rate.
    pub fn new(n_layers: usize, w_initial: FloatValue) -> Self {
        Self {
            ocean_temps: vec![vec![0.0; n_layers]; 2],
            upwelling_rates: [w_initial; 2],
            initialized: true,
            temperature_history: Vec::new(),
            dt_history: Vec::new(),
            land_temps: [0.0; 2],
            ground_temps: [0.0; 2],
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
