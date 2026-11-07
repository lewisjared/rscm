//! Permafrost Component
//!
//! Carbon released from thawing permafrost.
//!
//! # What This Component Does
//!
//! The permafrost region is divided into zonal bands, each holding mineral soil
//! and peat carbon. For each band and soil type:
//!
//! 1. Arctic warming above the band's melting threshold thaws frozen area
//!    (or refreezes thawed area when below it), moving carbon with the area
//! 2. Thawed area splits into aerobic and anaerobic parts
//! 3. Thawed carbon decomposes at Q10 and moisture dependent rates averaged over
//!    a seasonal soil temperature cycle
//! 4. Aerobic decomposition releases CO2. Anaerobic decomposition releases equal
//!    parts CO2 and CH4, part of the CH4 being oxidised to CO2 in the soil
//!
//! # Inputs
//!
//! - `Surface Temperature` (K) - Global mean surface temperature anomaly
//!
//! # Outputs
//!
//! - `Emissions|CO2|Permafrost` (GtC/yr) - CO2 released from permafrost
//! - `Emissions|CH4|Permafrost` (Mt CH4/yr) - CH4 released from permafrost
//! - `Permafrost Thawed Area` (1) - Carbon weighted thawed fraction after the step
//!
//! Emissions use the temperature at the start of the step and the mean of the
//! thawed pools before and after the step, as MAGICC7 does.
//! The emissions enter `CH4Chemistry`, `CH4Oxidation`, `OceanCarbon` and
//! `CO2Budget` when their `permafrost` parameter is set.
//!
//! # Rounding sensitivity
//!
//! A band that refreezes completely keeps thawed areas of rounding size, as in MAGICC7.
//! Carbon moved between them divides one rounding residue by another,
//! so emissions after such a refreeze depend on the last bits of the temperature.

use crate::parameters::{PermafrostParameters, PermafrostSoilParameters};
use rscm_core::component::{
    Component, ComponentState, GridType, InputState, OutputState, VariableDefinition, VariableRole,
};
use rscm_core::errors::{RscmError, RscmResult};
use rscm_core::state::{ScalarWindow, StateValue};
use rscm_core::timeseries::{FloatValue, Time};
use rscm_core::ComponentIO;
use serde::{Deserialize, Serialize};
use std::any::Any;

/// Mt CH4 per GtC of methane carbon.
const MTCH4_PER_GTC_NUMERATOR: FloatValue = 16000.0;
const MTCH4_PER_GTC_DENOMINATOR: FloatValue = 12.0;
#[cfg(test)]
const MTCH4_PER_GTC: FloatValue = MTCH4_PER_GTC_NUMERATOR / MTCH4_PER_GTC_DENOMINATOR;

/// Carbon and area of one soil type in one zonal band.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SoilBand {
    pub frozen_area: FloatValue,
    pub frozen_pool: FloatValue,
    pub aerobic_area: FloatValue,
    pub anaerobic_area: FloatValue,
    pub aerobic_pool: FloatValue,
    pub anaerobic_pool: FloatValue,
    /// Aerobic fraction of thawed area from the previous step, which splits this step's thaw.
    pub aerobic_fraction: FloatValue,
}

/// Per band carbon state of the permafrost region.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PermafrostState {
    pub mineral_soil: Vec<SoilBand>,
    pub peat: Vec<SoilBand>,
}

#[typetag::serde(name = "magicc.PermafrostState")]
impl ComponentState for PermafrostState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Monthly Q10 temperature term and moisture of one band, which both soil types share.
#[derive(Debug)]
struct Season {
    summer_max: FloatValue,
    q10_temperature: [FloatValue; 12],
    moisture: [FloatValue; 12],
}

/// Aerobic and anaerobic decomposition (GtC/yr) from one soil type.
#[derive(Debug, Clone, Copy, Default)]
struct Decomposition {
    aerobic: FloatValue,
    anaerobic: FloatValue,
}

/// Permafrost carbon feedback component.
#[derive(Debug, Clone, Serialize, Deserialize, ComponentIO)]
#[component(tags = ["carbon-cycle", "permafrost", "magicc"], category = "Carbon Cycle")]
#[inputs(
    temperature { name = "Surface Temperature", unit = "K" },
)]
#[outputs(
    co2_emissions { name = "Emissions|CO2|Permafrost", unit = "GtC/yr" },
    ch4_emissions { name = "Emissions|CH4|Permafrost", unit = "Mt CH4/yr" },
    thawed_area { name = "Permafrost Thawed Area", unit = "1" },
)]
pub struct Permafrost {
    parameters: PermafrostParameters,
}

impl Permafrost {
    pub fn new() -> Self {
        Self::from_parameters(PermafrostParameters::default())
    }

    pub fn from_parameters(parameters: PermafrostParameters) -> Self {
        Self { parameters }
    }

    pub fn parameters(&self) -> &PermafrostParameters {
        &self.parameters
    }

    /// Position of band `i` between the southernmost (0) and northernmost (1) band.
    fn band_position(&self, i: usize) -> FloatValue {
        let n = self.parameters.n_bands;
        if n > 1 {
            i as FloatValue / (n - 1) as FloatValue
        } else {
            0.0
        }
    }

    fn melting_temp(&self, i: usize) -> FloatValue {
        let p = &self.parameters;
        let n = p.n_bands.max(2) - 1;
        i as FloatValue * (p.melting_temp_max - p.melting_temp_min) / n as FloatValue
            + p.melting_temp_min
    }

    /// Fraction of the total pool in band `i`, a triangle shaped by `zonal_pool_distribution`.
    fn pool_fraction(&self, i: usize) -> FloatValue {
        let n = self.parameters.n_bands as FloatValue;
        let d = self.parameters.zonal_pool_distribution;
        // Band index from the end the triangle's base points away from.
        let k = if d <= 0.0 {
            i + 1
        } else {
            self.parameters.n_bands - i
        } as FloatValue;
        let d = -d.abs();
        ((1.0 + d) / n - d * k / (n * n)) / (1.0 + d / 2.0 - d / (2.0 * n))
    }

    /// Mineral soil and peat carbon of band `i` before any thaw (GtC).
    fn potential_pools(&self, i: usize) -> (FloatValue, FloatValue) {
        let p = &self.parameters;
        let fraction = self.pool_fraction(i);
        let ms_fraction = self.band_position(i)
            * (p.mineral_soil_fraction_north - p.mineral_soil_fraction_south)
            + p.mineral_soil_fraction_south;
        let ms = ms_fraction * p.total_pool * fraction;
        (ms, fraction * p.total_pool - ms)
    }

    /// Fully frozen permafrost.
    pub fn initial_state(&self) -> PermafrostState {
        let frozen = |pool, soil: &PermafrostSoilParameters| SoilBand {
            frozen_area: 1.0,
            frozen_pool: pool,
            aerobic_fraction: 1.0 - soil.anaerobic_initial_fraction,
            ..Default::default()
        };
        let p = &self.parameters;
        let (mineral_soil, peat) = (0..p.n_bands)
            .map(|i| {
                let (ms, peat) = self.potential_pools(i);
                (frozen(ms, &p.mineral_soil), frozen(peat, &p.peat))
            })
            .unzip();
        PermafrostState { mineral_soil, peat }
    }

    /// Carbon weighted fraction of the permafrost area that has thawed.
    pub fn thawed_area(&self, state: &PermafrostState) -> FloatValue {
        let frozen: FloatValue = (0..self.parameters.n_bands)
            .map(|i| {
                let (ms, peat) = self.potential_pools(i);
                state.mineral_soil[i].frozen_area * ms + state.peat[i].frozen_area * peat
            })
            .sum();
        1.0 - frozen / self.parameters.total_pool
    }

    /// Soil temperature below the summer maximum in each month of the seasonal cycle.
    fn seasonal_offsets(&self) -> [FloatValue; 12] {
        let half_amplitude = 0.5 * self.parameters.soil_annual_cycle_amplitude;
        std::array::from_fn(|month| {
            // MAGICC7's seasonal cycle spans a quarter period, from sin(0) to sin(pi/2).
            let phase = month as FloatValue * std::f64::consts::PI / 22.0;
            phase.sin() * half_amplitude - half_amplitude
        })
    }

    fn season(&self, offsets: &[FloatValue; 12], summer_max: FloatValue) -> Season {
        let p = &self.parameters;
        let soil_temp = offsets.map(|offset| summer_max + offset);
        let moisture = soil_temp.map(|soil_temp| {
            let water = (p.soil_water_slope * soil_temp + p.soil_water_offset)
                .max(p.soil_water_min)
                .min(1.0);
            (1.0 - (-water).exp()) / (1.0 - (-1.0 as FloatValue).exp())
        });
        Season {
            summer_max,
            q10_temperature: soil_temp
                .map(|soil_temp| 1.0 / p.q10_temp1 - 1.0 / (soil_temp + p.q10_temp2)),
            moisture,
        }
    }

    /// Annual mean aerobic fraction of the thawed area of a soil type.
    fn aerobic_fraction(&self, soil: &PermafrostSoilParameters, season: &Season) -> FloatValue {
        let anaerobic_fraction: FloatValue = season
            .moisture
            .iter()
            .map(|moisture| {
                (soil.anaerobic_initial_fraction
                    + (soil.anaerobic_max_fraction - soil.anaerobic_initial_fraction)
                        * moisture
                        * soil.anaerobic_moisture_sensitivity)
                    .min(soil.anaerobic_max_fraction)
                    .max(0.0)
            })
            .sum::<FloatValue>();
        1.0 - anaerobic_fraction / 12.0
    }

    /// Annual mean (aerobic, anaerobic) decomposition rate of a soil type.
    fn decomposition_rates(
        &self,
        soil: &PermafrostSoilParameters,
        season: &Season,
    ) -> (FloatValue, FloatValue) {
        let p = &self.parameters;
        let base = soil.decomposition_ratio / p.aerobic_turnover_time;
        let anaerobic_base =
            p.anaerobic_decomposition_ratio * soil.decomposition_ratio / p.aerobic_turnover_time;
        let (mut aerobic, mut anaerobic) = (0.0, 0.0);
        for (temperature, moisture) in season.q10_temperature.iter().zip(&season.moisture) {
            let q10 = |alpha: FloatValue| (alpha * temperature).exp();
            aerobic += base * q10(soil.q10_aerobic_alpha) * moisture;
            anaerobic += q10(soil.q10_anaerobic_alpha) * anaerobic_base;
        }
        (aerobic / 12.0, anaerobic / 12.0)
    }

    /// Advance one soil type in one band by `dt` years.
    fn step_band(
        &self,
        soil: &PermafrostSoilParameters,
        band: &mut SoilBand,
        season: &Season,
        dt: FloatValue,
    ) -> Decomposition {
        let summer_max = season.summer_max;
        let aerobic_fraction = self.aerobic_fraction(soil, season);
        let thaw_rate =
            summer_max.signum() * summer_max.abs().powf(soil.thaw_exponent) * soil.thaw_rate;

        let frozen = band.frozen_area;
        let thawed = 1.0 - frozen;
        let (aerobic_pool, anaerobic_pool) = (band.aerobic_pool, band.anaerobic_pool);
        let (old_aerobic_area, old_anaerobic_area) = (band.aerobic_area, band.anaerobic_area);
        let aerobic_area = aerobic_fraction * thawed;
        let anaerobic_area = (1.0 - aerobic_fraction) * thawed;
        let (new_aerobic_pool, new_anaerobic_pool) = redistribute(band, aerobic_area);

        // Thaw is bounded by the frozen area and refreezing by each type's area before the step.
        let thaw = thaw_rate * dt;
        let split = band.aerobic_fraction;
        let new_aerobic = (thaw * split * frozen)
            .min(frozen * split)
            .max(-old_aerobic_area);
        let new_anaerobic = (thaw * (1.0 - split) * frozen)
            .min((1.0 - split) * frozen)
            .max(-old_anaerobic_area);
        let (moved_aerobic, moved_anaerobic) = if new_aerobic >= 0.0 {
            let density = density(band.frozen_pool, frozen);
            (density * new_aerobic, density * new_anaerobic)
        } else {
            (
                density(aerobic_pool, old_aerobic_area) * new_aerobic,
                density(anaerobic_pool, old_anaerobic_area) * new_anaerobic,
            )
        };
        let new_aerobic_pool = (new_aerobic_pool + moved_aerobic).max(0.0);
        let new_anaerobic_pool = (new_anaerobic_pool + moved_anaerobic).max(0.0);
        let aerobic_area = (aerobic_area + new_aerobic).max(0.0);
        let anaerobic_area = (anaerobic_area + new_anaerobic).max(0.0);
        band.frozen_pool = (band.frozen_pool - moved_aerobic - moved_anaerobic).max(0.0);
        band.frozen_area =
            (frozen - (aerobic_area - old_aerobic_area) - (anaerobic_area - old_anaerobic_area))
                .max(0.0);

        let (aerobic_sum, anaerobic_sum) = (
            new_aerobic_pool + aerobic_pool,
            new_anaerobic_pool + anaerobic_pool,
        );
        // Frozen bands have no thawed carbon, so their rates are not needed.
        let decomposition = if aerobic_sum == 0.0 && anaerobic_sum == 0.0 {
            Decomposition::default()
        } else {
            let (aerobic_rate, anaerobic_rate) = self.decomposition_rates(soil, season);
            Decomposition {
                aerobic: aerobic_rate * aerobic_sum / 2.0,
                anaerobic: anaerobic_rate * anaerobic_sum / 2.0,
            }
        };
        band.aerobic_area = aerobic_area;
        band.anaerobic_area = anaerobic_area;
        band.aerobic_fraction = aerobic_fraction;
        band.aerobic_pool = new_aerobic_pool - decomposition.aerobic * dt;
        band.anaerobic_pool = new_anaerobic_pool - decomposition.anaerobic * dt;
        decomposition
    }

    /// Advance the permafrost by one step.
    ///
    /// Returns (CO2 emissions in GtC/yr, CH4 emissions in Mt CH4/yr).
    pub fn step(
        &self,
        state: &mut PermafrostState,
        temperature: FloatValue,
        dt: FloatValue,
    ) -> (FloatValue, FloatValue) {
        let p = &self.parameters;
        let arctic = p.arctic_amplification * temperature;
        // Each pool is summed over bands before the soil types are combined, as in MAGICC7.
        let (mut ms, mut peat) = (Decomposition::default(), Decomposition::default());
        let offsets = self.seasonal_offsets();
        for i in 0..p.n_bands {
            let season = self.season(&offsets, arctic - self.melting_temp(i));
            for (soil, band, total) in [
                (&p.mineral_soil, &mut state.mineral_soil[i], &mut ms),
                (&p.peat, &mut state.peat[i], &mut peat),
            ] {
                let d = self.step_band(soil, band, &season, dt);
                total.aerobic += d.aerobic;
                total.anaerobic += d.anaerobic;
            }
        }
        let (ms_ox, peat_ox) = (
            p.mineral_soil.ch4_oxidation_fraction,
            p.peat.ch4_oxidation_fraction,
        );
        let ch4 = |anaerobic: FloatValue, ox: FloatValue| {
            anaerobic * MTCH4_PER_GTC_NUMERATOR / MTCH4_PER_GTC_DENOMINATOR * 0.5 * (1.0 - ox)
        };
        let co2 = (peat.aerobic + ms.aerobic)
            + ms.anaerobic * (ms_ox + 1.0) * 0.5
            + peat.anaerobic * (1.0 + peat_ox) * 0.5;
        (co2, ch4(ms.anaerobic, ms_ox) + ch4(peat.anaerobic, peat_ox))
    }

    fn solve_impl(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
        state: &mut PermafrostState,
    ) -> RscmResult<OutputState> {
        if state.mineral_soil.len() != self.parameters.n_bands {
            *state = self.initial_state();
        }
        let inputs = PermafrostInputs::from_input_state(input_state);
        let (co2_emissions, ch4_emissions) =
            self.step(state, inputs.temperature.get(), t_next - t_current);
        Ok(PermafrostOutputs {
            co2_emissions,
            ch4_emissions,
            thawed_area: self.thawed_area(state),
        }
        .into())
    }
}

/// Carbon per unit area, zero for an empty area.
fn density(pool: FloatValue, area: FloatValue) -> FloatValue {
    if area > 0.0 {
        pool / area
    } else {
        0.0
    }
}

/// Thawed pools after the aerobic area changes to `aerobic_area`.
///
/// Area changing type takes the carbon density of the type it leaves.
fn redistribute(band: &SoilBand, aerobic_area: FloatValue) -> (FloatValue, FloatValue) {
    let to_aerobic = aerobic_area - band.aerobic_area;
    let moved = if to_aerobic >= 0.0 {
        density(band.anaerobic_pool, band.anaerobic_area) * to_aerobic
    } else {
        density(band.aerobic_pool, band.aerobic_area) * to_aerobic
    };
    (band.aerobic_pool + moved, band.anaerobic_pool - moved)
}

impl Default for Permafrost {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde(name = "magicc.Permafrost")]
impl Component for Permafrost {
    fn definitions(&self) -> Vec<VariableDefinition> {
        Self::generated_definitions()
    }

    fn solve(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
    ) -> RscmResult<OutputState> {
        let mut state = self.initial_state();
        self.solve_impl(t_current, t_next, input_state, &mut state)
    }

    fn create_initial_state(&self) -> Box<dyn ComponentState> {
        Box::new(self.initial_state())
    }

    fn solve_with_state(
        &self,
        t_current: Time,
        t_next: Time,
        input_state: &InputState,
        internal_state: &mut dyn ComponentState,
    ) -> RscmResult<OutputState> {
        let state = internal_state
            .as_any_mut()
            .downcast_mut::<PermafrostState>()
            .ok_or_else(|| RscmError::Error("Internal state should be PermafrostState".into()))?;
        self.solve_impl(t_current, t_next, input_state, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total_carbon(state: &PermafrostState) -> FloatValue {
        state
            .mineral_soil
            .iter()
            .chain(&state.peat)
            .map(|b| b.frozen_pool + b.aerobic_pool + b.anaerobic_pool)
            .sum()
    }

    #[test]
    fn test_initial_state_holds_total_pool() {
        let component = Permafrost::new();
        let state = component.initial_state();
        assert!((total_carbon(&state) - 800.0).abs() < 1e-9);
        assert_eq!(component.thawed_area(&state), 0.0);
    }

    #[test]
    fn test_zonal_distribution_sums_to_one() {
        for d in [-1.0, -0.4, 0.0, 0.3, 1.0] {
            let component = Permafrost::from_parameters(PermafrostParameters {
                zonal_pool_distribution: d,
                ..Default::default()
            });
            let sum: FloatValue = (0..50).map(|i| component.pool_fraction(i)).sum();
            assert!((sum - 1.0).abs() < 1e-12, "d={d}");
            let (south, north) = (component.pool_fraction(0), component.pool_fraction(49));
            assert!(d >= 0.0 || north > south);
            assert!(d <= 0.0 || south > north);
        }
    }

    #[test]
    fn test_no_thaw_without_warming() {
        let component = Permafrost::new();
        let mut state = component.initial_state();
        for _ in 0..50 {
            let (co2, ch4) = component.step(&mut state, 0.0, 1.0);
            assert_eq!((co2, ch4), (0.0, 0.0));
        }
        assert_eq!(component.thawed_area(&state), 0.0);
    }

    #[test]
    fn test_onset_matches_magicc() {
        // MAGICC7 SSP245: global warming of 0.60088 K in 1989 thaws 3.8698e-5 of the region.
        let component = Permafrost::new();
        let mut state = component.initial_state();
        let (co2, ch4) = component.step(&mut state, 0.600_881_81, 1.0);
        assert!((component.thawed_area(&state) - 3.869_833_1e-5).abs() < 1e-11);
        assert!((co2 / 5.164_095_9e-5 - 1.0).abs() < 1e-6);
        assert!((ch4 / 6.990_3e-4 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_carbon_is_conserved() {
        let component = Permafrost::new();
        let mut state = component.initial_state();
        let mut emitted = 0.0;
        for year in 0..200 {
            let temperature = 0.03 * year as FloatValue;
            let (co2, ch4) = component.step(&mut state, temperature, 1.0);
            emitted += co2 + ch4 / MTCH4_PER_GTC;
        }
        assert!(component.thawed_area(&state) > 0.3);
        let err = total_carbon(&state) + emitted - 800.0;
        assert!(err.abs() < 1e-9, "{err}");
    }

    #[test]
    fn test_cooling_refreezes() {
        let component = Permafrost::new();
        let mut state = component.initial_state();
        for _ in 0..20 {
            component.step(&mut state, 2.0, 1.0);
        }
        let thawed = component.thawed_area(&state);
        component.step(&mut state, 0.0, 1.0);
        assert!(component.thawed_area(&state) < thawed);
    }

    fn season(component: &Permafrost, summer_max: FloatValue) -> Season {
        component.season(&component.seasonal_offsets(), summer_max)
    }

    /// Mineral soil band, half thawed, with thawed carbon densities 20 (aerobic) and 10 (anaerobic).
    fn half_thawed(aerobic_fraction: FloatValue) -> SoilBand {
        let aerobic_area = 0.5 * aerobic_fraction;
        let anaerobic_area = 0.5 - aerobic_area;
        SoilBand {
            frozen_area: 0.5,
            frozen_pool: 30.0,
            aerobic_area,
            anaerobic_area,
            aerobic_pool: 20.0 * aerobic_area,
            anaerobic_pool: 10.0 * anaerobic_area,
            aerobic_fraction,
        }
    }

    #[test]
    fn test_refreeze_bounded_by_area_before_step() {
        let component = Permafrost::new();
        let soil = &component.parameters.mineral_soil;
        let season = season(&component, -20.0);
        let (aerobic_rate, anaerobic_rate) = component.decomposition_rates(soil, &season);
        let aerobic_fraction = component.aerobic_fraction(soil, &season);
        assert!((aerobic_fraction - 0.95).abs() < 1e-12);
        let mut band = half_thawed(0.5);

        // Ten years at -2/yr would refreeze 9.5 and 0.5 of area, more than the 0.25 of each type.
        let d = component.step_band(soil, &mut band, &season, 10.0);

        // The split moves 0.225 of area and 2.25 GtC to aerobic before the refreeze.
        assert!((band.aerobic_area - 0.225).abs() < 1e-12);
        assert_eq!(band.anaerobic_area, 0.0);
        assert!((band.frozen_area - 0.775).abs() < 1e-12);
        // Refrozen carbon leaves at the pre-step densities: 5 + 2.5 GtC
        assert!((band.frozen_pool - 37.5).abs() < 1e-12);
        assert!((d.aerobic - aerobic_rate * (2.25 + 5.0) / 2.0).abs() < 1e-12);
        assert!((d.anaerobic - anaerobic_rate * (0.0 + 2.5) / 2.0).abs() < 1e-12);
        assert!((band.aerobic_pool - (2.25 - 10.0 * d.aerobic)).abs() < 1e-12);
        assert!((band.anaerobic_pool + 10.0 * d.anaerobic).abs() < 1e-12);
    }

    #[test]
    fn test_thaw_split_uses_previous_aerobic_fraction() {
        let component = Permafrost::new();
        let soil = &component.parameters.mineral_soil;
        let mut band = SoilBand {
            frozen_area: 1.0,
            frozen_pool: 10.0,
            aerobic_fraction: 0.25,
            ..Default::default()
        };
        let season = season(&component, 1.0);
        let aerobic_fraction = component.aerobic_fraction(soil, &season);
        component.step_band(soil, &mut band, &season, 1.0);
        let thaw = soil.thaw_rate;
        assert!((band.aerobic_area - 0.25 * thaw).abs() < 1e-15);
        assert!((band.anaerobic_area - 0.75 * thaw).abs() < 1e-15);
        assert_eq!(band.aerobic_fraction, aerobic_fraction);
    }

    #[test]
    fn test_complete_refreeze_emits_negative_carbon_next_year() {
        let component = Permafrost::new();
        let soil = &component.parameters.mineral_soil;
        let season = season(&component, -20.0);
        let mut band = half_thawed(component.aerobic_fraction(soil, &season));

        let first = component.step_band(soil, &mut band, &season, 1.0);
        assert!(first.aerobic > 0.0 && first.anaerobic > 0.0);
        assert!(band.aerobic_area.abs() < 1e-12 && band.anaerobic_area.abs() < 1e-12);
        assert!((band.frozen_area - 1.0).abs() < 1e-12);
        assert!((band.frozen_pool - 39.75).abs() < 1e-12);
        // Decomposition of the mean pool leaves the thawed pools negative
        let (aerobic_pool, anaerobic_pool) = (band.aerobic_pool, band.anaerobic_pool);
        assert!((aerobic_pool + first.aerobic).abs() < 1e-12);
        assert!(aerobic_pool < 0.0 && anaerobic_pool < 0.0);

        let second = component.step_band(soil, &mut band, &season, 1.0);
        assert!(second.aerobic < 0.0 && second.anaerobic < 0.0);
        let (aerobic_rate, _) = component.decomposition_rates(soil, &season);
        assert!((second.aerobic - aerobic_rate * aerobic_pool / 2.0).abs() < 1e-12);
    }
}
