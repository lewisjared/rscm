//! Sea Level Rise Component
//!
//! Global mean sea level rise from land ice and thermal expansion.
//!
//! # What This Component Does
//!
//! Each contribution is driven by the annual mean global surface warming:
//!
//! 1. Glaciers relax towards an equilibrium sea level that rises with warming
//! 2. Greenland and Antarctic surface mass balance follow a power of warming,
//!    Greenland's shrinking with its remaining volume
//! 3. Greenland solid ice discharge depletes a low and a high volume case
//! 4. Antarctic solid ice discharge convolves basal melt with each drainage region's
//!    response function (Levermann et al. 2014)
//!
//! Thermal expansion needs the ocean layer temperatures, so `ClimateUDEB` calculates it
//! when its `thermal_expansion` parameter is set, see [`thermal_expansion_increment`].
//!
//! # Inputs
//!
//! - `Surface Temperature|Annual Mean` (K) - Mean warming over the step, from `ClimateUDEB`
//! - `Sea Level Rise|Thermal Expansion` (mm) - From `ClimateUDEB`
//!
//! # Outputs
//!
//! - `Sea Level Rise` (mm) - Sum of all contributions
//! - `Sea Level Rise|Glaciers`, `Sea Level Rise|Greenland|Surface Mass Balance`,
//!   `Sea Level Rise|Greenland|Solid Ice Discharge`, `Sea Level Rise|Antarctica|Surface Mass Balance`
//!   and `Sea Level Rise|Antarctica|Solid Ice Discharge` (mm)
//!
//! A step's warming sets the sea level at its end, as in MAGICC7.
//! Steps must be one year long.
//!
//! # Differences from MAGICC7
//!
//! - **Land water** is not implemented. MAGICC7's default run leaves it off.
//! - **Antarctic discharge** keeps two MAGICC7 quirks: the response functions overflow
//!   from an age of 216 years, and a delayed region releases one year of melt before it starts.

use crate::parameters::{
    ClimateUDEBParameters, SeaLevelRiseParameters, SurfaceMassBalanceParameters,
    ThermalExpansionParameters,
};
use rscm_core::component::{
    Component, ComponentState, GridType, InputState, OutputState, RequirementDefinition,
    RequirementType,
};
use rscm_core::errors::{RSCMError, RSCMResult};
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};
use std::any::Any;

/// Annual mean global surface warming, written by `ClimateUDEB` at the end of each step.
pub const ANNUAL_MEAN_TEMPERATURE: &str = "Surface Temperature|Annual Mean";

/// Cumulative thermal expansion, written by `ClimateUDEB`.
pub const THERMAL_EXPANSION: &str = "Sea Level Rise|Thermal Expansion";

/// Sea level rise from thermal expansion over one step, in mm.
///
/// `start` and `end` are the ocean layer warming of each hemisphere at the ends of the step
/// and `profile` the initial layer temperatures.
pub fn thermal_expansion_increment(
    expansion: &ThermalExpansionParameters,
    climate: &ClimateUDEBParameters,
    profile: &[Vec<FloatValue>; 2],
    start: &[Vec<FloatValue>],
    end: &[Vec<FloatValue>],
) -> FloatValue {
    let (fgno, _, fgso, _) = climate.global_box_fractions();
    let [p1, p2, p3, p4, p5, p6] = expansion.coefficients;
    let sc = expansion.scaling;
    let (mld, dz) = (climate.mixed_layer_depth, climate.layer_thickness);
    let bound = |b: usize| match b {
        0 => 0.0,
        _ => (b - 1) as FloatValue * dz + mld,
    };
    let area = |hemi, l| {
        (climate.ocean_area_at_depth(hemi, bound(l + 1))
            + climate.ocean_area_at_depth(hemi, bound(l)))
            * 0.5
    };
    let total: FloatValue = (0..climate.n_layers)
        .map(|l| {
            let (area_nh, area_sh) = (area(0, l), area(1, l));
            let (w_nh, w_sh) = (area_nh * fgno, area_sh * fgso);
            let world = |t: &[Vec<FloatValue>]| (w_nh * t[0][l] + w_sh * t[1][l]) / (w_nh + w_sh);
            let (t_start, t_end) = (world(start), world(end));
            let warming = (t_end - t_start) * sc;
            let tl1 = (t_end + t_start) * 0.5 * sc + (profile[0][l] + profile[1][l]) * 0.5;
            let tl2 = tl1 * tl1;
            let p = layer_pressure(l, mld, dz);
            let coefficient = p1 + (12.9635 - 1.0833 * p) * (p2 * tl1)
                - p3 * tl2 * (0.1713 - 0.019263 * p)
                + tl1 * tl2 / 6000.0 * p4 * (10.41 - 1.1338 * p)
                + p5 * p
                - p * (p6 * p);
            let thickness = if l == 0 { mld } else { dz };
            (area_nh + area_sh) * 0.5 * (warming * coefficient * thickness / 1000.0)
        })
        .sum();
    total / ((climate.ocean_area_at_depth(0, 0.0) + climate.ocean_area_at_depth(1, 0.0)) * 0.5)
}

/// Pressure at the centre of ocean layer `l`, as MAGICC7 calculates it.
fn layer_pressure(l: usize, mld: FloatValue, dz: FloatValue) -> FloatValue {
    let z = match l {
        0 => 0.5 * mld,
        _ => l as FloatValue * dz + mld - 0.5 * dz,
    };
    (((-(z / 3500.0)).exp() - 1.0) * 10.5 + 0.1005 * z) * 0.0098
}

/// Linear interpolation in a table with increasing `xs`, clamped at its ends.
fn interpolate(x: FloatValue, xs: &[FloatValue], ys: &[FloatValue]) -> FloatValue {
    match xs.iter().position(|&v| v > x) {
        Some(0) => ys[0],
        Some(i) => {
            let w = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
            (1.0 - w) * ys[i - 1] + w * ys[i]
        }
        None => ys[ys.len() - 1],
    }
}

/// Basal melt history of one Antarctic drainage region.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AntarcticRegionState {
    /// Warming in the region's first year
    pub baseline: Option<FloatValue>,
    /// Basal melt (m/yr) of each year since the region started
    pub basal_melt: Vec<FloatValue>,
}

/// Internal state of [`SeaLevelRise`], contributions in mm.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SeaLevelRiseState {
    pub glaciers: FloatValue,
    pub greenland_smb: FloatValue,
    pub antarctic_smb: FloatValue,
    /// Remaining volume of the low and high Greenland discharge cases
    pub greenland_discharge_volume: [FloatValue; 2],
    pub antarctic_regions: Vec<AntarcticRegionState>,
}

#[typetag::serde]
impl ComponentState for SeaLevelRiseState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Sea level rise component.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["climate", "sea-level", "magicc"], category = "Climate")]
#[inputs(
    temperature { name = "Surface Temperature|Annual Mean", unit = "K" },
    thermal_expansion { name = "Sea Level Rise|Thermal Expansion", unit = "mm" },
)]
#[outputs(
    total { name = "Sea Level Rise", unit = "mm" },
    glaciers { name = "Sea Level Rise|Glaciers", unit = "mm" },
    greenland_smb { name = "Sea Level Rise|Greenland|Surface Mass Balance", unit = "mm" },
    greenland_discharge { name = "Sea Level Rise|Greenland|Solid Ice Discharge", unit = "mm" },
    antarctic_smb { name = "Sea Level Rise|Antarctica|Surface Mass Balance", unit = "mm" },
    antarctic_discharge { name = "Sea Level Rise|Antarctica|Solid Ice Discharge", unit = "mm" },
)]
pub struct SeaLevelRise {
    parameters: SeaLevelRiseParameters,
}

impl SeaLevelRise {
    pub fn new() -> Self {
        Self::from_parameters(SeaLevelRiseParameters::default())
    }

    pub fn from_parameters(parameters: SeaLevelRiseParameters) -> Self {
        Self { parameters }
    }

    pub fn parameters(&self) -> &SeaLevelRiseParameters {
        &self.parameters
    }

    pub fn initial_state(&self) -> SeaLevelRiseState {
        let discharge = &self.parameters.greenland_discharge;
        SeaLevelRiseState {
            greenland_discharge_volume: [discharge.low.volume, discharge.high.volume],
            antarctic_regions: vec![
                AntarcticRegionState::default();
                self.parameters.antarctic_discharge.regions.len()
            ],
            ..Default::default()
        }
    }

    fn glacier_step(&self, glaciers: FloatValue, temperature: FloatValue) -> FloatValue {
        let p = &self.parameters.glaciers;
        let (temps, levels) = (&p.equilibrium_temperature, &p.equilibrium_sea_level);
        let equilibrium = interpolate(temperature, temps, levels);
        let equilibrium_temperature = interpolate(glaciers, levels, temps);
        let response = (temperature - equilibrium_temperature)
            .abs()
            .powf(p.exponent)
            .copysign(equilibrium_temperature);
        glaciers + p.sensitivity * (equilibrium - glaciers) * response
    }

    fn smb_step(
        p: &SurfaceMassBalanceParameters,
        smb: FloatValue,
        temperature: FloatValue,
    ) -> FloatValue {
        let melt = p.linear_fraction * temperature
            + (1.0 - p.linear_fraction) * temperature.max(0.0).powf(p.exponent);
        let remaining = p
            .volume
            .map_or(1.0, |v| (1.0 - smb / v).max(0.0).powf(p.volume_exponent));
        smb + p.sensitivity * melt * remaining
    }

    fn greenland_discharge(
        &self,
        state: &mut SeaLevelRiseState,
        year: Time,
        temperature: FloatValue,
    ) -> FloatValue {
        let p = &self.parameters.greenland_discharge;
        let cases = [&p.low, &p.high];
        if year >= p.start_year && temperature != 0.0 {
            for (volume, case) in state.greenland_discharge_volume.iter_mut().zip(cases) {
                let discharge = (-case.sensitivity * *volume * (case.exponent * temperature).exp())
                    .min(0.0)
                    .max(-*volume);
                *volume = (*volume + discharge).max(0.0);
            }
        }
        let [low, high] = [0, 1].map(|i| cases[i].volume - state.greenland_discharge_volume[i]);
        ((high - low) * p.case + low) * p.scaling
    }

    fn antarctic_discharge(
        &self,
        state: &mut SeaLevelRiseState,
        year: Time,
        temperature: FloatValue,
    ) -> FloatValue {
        let p = &self.parameters.antarctic_discharge;
        let mut total = 0.0;
        for (region, history) in p.regions.iter().zip(state.antarctic_regions.iter_mut()) {
            let response = |age: usize| {
                if age >= p.response_span {
                    return 0.0;
                }
                let c = region.response;
                // MAGICC7 raises the age to powers in int32, so the fourth power wraps from 216
                let x = age as i32;
                let x2 = x.wrapping_mul(x);
                let [x, x2, x3, x4] =
                    [x, x2, x2.wrapping_mul(x), x2.wrapping_mul(x2)].map(FloatValue::from);
                (c[0] * x4 + c[1] * x3 + c[2] * x2 + c[3] * x + c[4]).max(0.0)
            };
            let region_start = p.start_year + region.delay as FloatValue;
            if year < region_start {
                // MAGICC7 releases one year of melt against zero warming the year before a region starts
                if year >= p.start_year && year + 1.0 >= region_start {
                    let melt = temperature * region.temperature_scaling * p.basal_melt_sensitivity;
                    total += melt * response(0) * 1000.0;
                }
                continue;
            }
            let baseline = *history.baseline.get_or_insert(temperature);
            let melt = &mut history.basal_melt;
            melt.push(
                (temperature - baseline) * region.temperature_scaling * p.basal_melt_sensitivity,
            );
            // MAGICC7 sums the first and last years of the convolution apart from the rest
            let n = melt.len();
            let mut convolution = melt[n - 1] * response(0);
            if n > 1 {
                let middle: FloatValue = (1..n - 1)
                    .map(|age| melt[n - 1 - age] * response(age))
                    .sum();
                convolution = convolution + middle + melt[0] * response(n - 1);
            }
            total += convolution * 1000.0;
        }
        total * p.scaling
    }

    fn solve_impl(
        &self,
        t_current: Time,
        input_state: &InputState,
        state: &mut SeaLevelRiseState,
    ) -> RSCMResult<OutputState> {
        if state.antarctic_regions.len() != self.parameters.antarctic_discharge.regions.len() {
            *state = self.initial_state();
        }
        let inputs = SeaLevelRiseInputs::from_input_state(input_state);
        let (_, temperature) = inputs.temperature.interval()?;
        let (_, thermal_expansion) = inputs.thermal_expansion.interval()?;
        let p = &self.parameters;
        let year = t_current;
        if year >= p.glaciers.start_year {
            state.glaciers = self.glacier_step(state.glaciers, temperature);
        }
        if year >= p.greenland_smb.start_year {
            state.greenland_smb =
                Self::smb_step(&p.greenland_smb, state.greenland_smb, temperature);
        }
        if year >= p.antarctic_smb.start_year {
            state.antarctic_smb =
                Self::smb_step(&p.antarctic_smb, state.antarctic_smb, temperature);
        }
        let greenland_discharge = self.greenland_discharge(state, year, temperature);
        let antarctic_discharge = self.antarctic_discharge(state, year, temperature);
        Ok(SeaLevelRiseOutputs {
            total: thermal_expansion
                + state.glaciers
                + state.greenland_smb
                + greenland_discharge
                + state.antarctic_smb
                + antarctic_discharge,
            glaciers: state.glaciers,
            greenland_smb: state.greenland_smb,
            greenland_discharge,
            antarctic_smb: state.antarctic_smb,
            antarctic_discharge,
        }
        .into())
    }
}

impl Default for SeaLevelRise {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde]
impl Component for SeaLevelRise {
    fn definitions(&self) -> Vec<RequirementDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        t_current: Time,
        _t_next: Time,
        input_state: &InputState,
    ) -> RSCMResult<OutputState> {
        let mut state = self.initial_state();
        self.solve_impl(t_current, input_state, &mut state)
    }

    fn create_initial_state(&self) -> Box<dyn ComponentState> {
        Box::new(self.initial_state())
    }

    fn solve_with_state(
        &self,
        t_current: Time,
        _t_next: Time,
        input_state: &InputState,
        internal_state: &mut dyn ComponentState,
    ) -> RSCMResult<OutputState> {
        let state = internal_state
            .as_any_mut()
            .downcast_mut::<SeaLevelRiseState>()
            .ok_or_else(|| RSCMError::Error("Internal state should be SeaLevelRiseState".into()))?;
        self.solve_impl(t_current, input_state, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_expansion_follows_warming() {
        let climate = ClimateUDEBParameters::default();
        let expansion = ThermalExpansionParameters::default();
        let profile = [0, 1].map(|hemi| climate.initial_ocean_profile(hemi));
        let cold = vec![vec![0.0; climate.n_layers]; 2];
        let warm = vec![vec![0.1; climate.n_layers]; 2];
        let rise = thermal_expansion_increment(&expansion, &climate, &profile, &cold, &warm);
        let fall = thermal_expansion_increment(&expansion, &climate, &profile, &warm, &cold);
        assert!(rise > 0.0);
        assert!((rise + fall).abs() < 1e-12);
        assert_eq!(
            thermal_expansion_increment(&expansion, &climate, &profile, &warm, &warm),
            0.0
        );
    }

    #[test]
    fn test_interpolate_clamps() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [10.0, 20.0, 40.0];
        assert_eq!(interpolate(-1.0, &xs, &ys), 10.0);
        assert_eq!(interpolate(1.5, &xs, &ys), 30.0);
        assert_eq!(interpolate(3.0, &xs, &ys), 40.0);
    }

    #[test]
    fn test_no_warming_leaves_ice_sheets_unchanged() {
        let component = SeaLevelRise::new();
        let mut state = component.initial_state();
        for year in 1850..2100 {
            component.greenland_discharge(&mut state, year as Time, 0.0);
            assert_eq!(
                component.antarctic_discharge(&mut state, year as Time, 0.0),
                0.0
            );
            state.greenland_smb = SeaLevelRise::smb_step(
                &component.parameters.greenland_smb,
                state.greenland_smb,
                0.0,
            );
        }
        assert_eq!(state.greenland_smb, 0.0);
        assert_eq!(state.greenland_discharge_volume, [35.9803, 53.6289]);
    }

    #[test]
    fn test_greenland_smb_limited_by_volume() {
        let p = SurfaceMassBalanceParameters::greenland();
        let mut smb = 0.0;
        for _ in 0..100_000 {
            smb = SeaLevelRise::smb_step(&p, smb, 10.0);
        }
        // The last step can overshoot the volume slightly before melt stops.
        assert!((smb - 7360.0).abs() < 0.1, "{smb}");
    }
}
