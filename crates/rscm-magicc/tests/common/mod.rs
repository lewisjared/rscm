//! Shared test helpers for rscm-magicc integration tests.
//!
//! Provides common parameter construction, equilibrium temperature computation,
//! and ClimateUDEB input state builders used across multiple test files.
#![allow(dead_code)] // Each test binary uses a different subset of these helpers.

use ndarray::Array2;
use rscm_core::component::{Component, InputState};
use rscm_core::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
use rscm_core::spatial::FourBoxGrid;
use rscm_core::state::{FourBoxSlice, StateValue};
use rscm_core::timeseries::{FloatValue, GridTimeseries, TimeAxis, Timeseries};
use rscm_core::timeseries_collection::{TimeseriesData, TimeseriesItem};
use rscm_core::utils::linear_algebra::invert_4x4;
use rscm_magicc::climate::lamcalc::{build_coupling_matrix, compute_qfrac, LAMCALCParameters};
use rscm_magicc::climate::ClimateUDEB;
use rscm_magicc::parameters::ClimateUDEBParameters;
use std::sync::Arc;

/// Convergence tolerance used by LAMCALC internally.
pub const RLO_TOLERANCE: FloatValue = 0.001;

// ---------------------------------------------------------------------------
// LAMCALC parameter construction
// ---------------------------------------------------------------------------

/// Create default LAMCALC parameters matching MAGICC defaults.
pub fn default_lamcalc_params() -> LAMCALCParameters {
    let nh_land = 0.42;
    let sh_land = 0.21;
    LAMCALCParameters {
        erf_2xco2: 3.71,
        land_ocean_heat_exchange: 1.44,
        north_south_heat_exchange: 0.31,
        ecs: 3.0,
        land_ocean_warming_ratio: 1.317,
        amplify_ocean_to_land: 1.02,
        grid: FourBoxGrid::with_weights([
            0.5 - nh_land / 2.0,
            nh_land / 2.0,
            0.5 - sh_land / 2.0,
            sh_land / 2.0,
        ]),
        co2_regional_pattern: [1.0; 4],
        include_unset_lamcalc_error: true,
    }
}

/// Build LAMCALC parameters from ClimateUDEB parameters for consistency.
pub fn lamcalc_params_from_udeb(params: &ClimateUDEBParameters) -> LAMCALCParameters {
    params.lamcalc_parameters(params.ecs, params.erf_2xco2)
}

// ---------------------------------------------------------------------------
// ClimateUDEB parameter construction
// ---------------------------------------------------------------------------

/// Create [`ClimateUDEBParameters`] with time-varying ECS adjustments disabled.
///
/// This isolates the base ECS for equilibrium tests by zeroing out the
/// cumulative-temperature and forcing sensitivity coefficients.
pub fn params_with_fixed_ecs(ecs: FloatValue) -> ClimateUDEBParameters {
    ClimateUDEBParameters {
        ecs,
        feedback_cumulative_temperature_sensitivity: 0.0,
        feedback_forcing_sensitivity: 0.0,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Equilibrium temperature computation
// ---------------------------------------------------------------------------

/// Result of computing equilibrium temperatures from LAMCALC parameters.
///
/// Contains per-box temperatures and area-weighted aggregates for ocean,
/// land, and global means, plus the actual land-ocean warming ratio.
#[derive(Debug, Clone)]
pub struct EquilibriumTemperatures {
    /// Per-box temperatures: $[T_{NO}, T_{NL}, T_{SO}, T_{SL}]$.
    pub box_temps: [FloatValue; 4],
    /// Area-weighted ocean mean temperature.
    pub ocean_mean: FloatValue,
    /// Area-weighted land mean temperature.
    pub land_mean: FloatValue,
    /// Area-weighted global mean temperature.
    pub global_mean: FloatValue,
    /// Actual land-ocean warming ratio (land_mean / ocean_mean).
    pub rlo_actual: FloatValue,
}

/// Compute equilibrium box temperatures from LAMCALC parameters and solved
/// lambda values, returning both the per-box temperatures and the aggregated
/// ocean, land, and global means.
///
/// Solves $M \cdot T = Q \cdot f \cdot qfrac$ where $M$ is the coupling matrix,
/// $Q$ is the 2xCO2 effective radiative forcing, $f$ is the area fraction
/// vector, and $qfrac$ accounts for regional CO2 forcing patterns.
pub fn compute_equilibrium_temperatures(
    params: &LAMCALCParameters,
    lam_o: FloatValue,
    lam_l: FloatValue,
) -> EquilibriumTemperatures {
    let matrix = build_coupling_matrix(params, lam_o, lam_l);
    let inv = invert_4x4(&matrix).unwrap_or_else(|| {
        panic!(
            "Coupling matrix should be invertible for lam_o={}, lam_l={}",
            lam_o, lam_l
        )
    });

    let area = *params.grid.weights();
    let [nh_ocean, nh_land, sh_ocean, sh_land] = area;
    let q = params.erf_2xco2;

    let qfrac = compute_qfrac(&params.co2_regional_pattern, &area);

    let mut box_temps = [0.0_f64; 4];
    for row in 0..4 {
        let mut sum = 0.0;
        for col in 0..4 {
            sum += inv[row][col] * area[col] * qfrac[col];
        }
        box_temps[row] = q * sum;
    }

    let ocean_mean = (nh_ocean * box_temps[0] + sh_ocean * box_temps[2]) / (nh_ocean + sh_ocean);
    let land_mean = (nh_land * box_temps[1] + sh_land * box_temps[3]) / (nh_land + sh_land);
    let global_mean = area[0] * box_temps[0]
        + area[1] * box_temps[1]
        + area[2] * box_temps[2]
        + area[3] * box_temps[3];
    let rlo_actual = land_mean / ocean_mean;

    EquilibriumTemperatures {
        box_temps,
        ocean_mean,
        land_mean,
        global_mean,
        rlo_actual,
    }
}

// ---------------------------------------------------------------------------
// ClimateUDEB input state construction
// ---------------------------------------------------------------------------

/// Build the [`InputState`] needed by `ClimateUDEB::solve_with_state` for a
/// single step.
///
/// Provides a scalar ERF timeseries and a four-box surface temperature state.
/// Both timeseries span `[t_current, t_next]` with constant values.
pub fn build_udeb_input_state(
    erf: FloatValue,
    prev_temps: &FourBoxSlice,
    t_current: FloatValue,
    t_next: FloatValue,
) -> (TimeseriesItem, TimeseriesItem) {
    let time_axis = Arc::new(TimeAxis::from_values(ndarray::array![t_current, t_next]));

    // Scalar ERF timeseries (constant)
    let erf_ts = Timeseries::from_values(
        ndarray::array![erf, erf],
        ndarray::array![t_current, t_next],
    );
    let erf_item = TimeseriesItem {
        data: TimeseriesData::Scalar(erf_ts),
        name: "Effective Radiative Forcing".to_string(),
    };

    // Four-box surface temperature state
    let surf_grid = FourBoxGrid::magicc_standard();
    let mut surf_values = Array2::zeros((2, 4));
    for col in 0..4 {
        surf_values[[0, col]] = prev_temps.0[col];
        surf_values[[1, col]] = prev_temps.0[col]; // same for next step placeholder
    }
    let surf_ts = GridTimeseries::new(
        surf_values,
        time_axis,
        surf_grid,
        "K".to_string(),
        InterpolationStrategy::from(LinearSplineStrategy::new(true)),
    );
    let surf_item = TimeseriesItem {
        data: TimeseriesData::FourBox(surf_ts),
        name: "Surface Air Temperature Change".to_string(),
    };

    (erf_item, surf_item)
}

// ---------------------------------------------------------------------------
// ClimateUDEB simulation runner
// ---------------------------------------------------------------------------

/// Per-year record from a ClimateUDEB simulation under constant forcing.
#[allow(dead_code)]
pub struct YearRecord {
    /// Area-weighted global surface air temperature.
    pub global_sat: FloatValue,
    /// Sea surface temperature (scalar output).
    pub sst: FloatValue,
    /// Heat uptake into the deep ocean.
    pub heat_uptake: FloatValue,
    /// Four-box surface temperature state.
    pub surface_temperature: FourBoxSlice,
}

/// Run ClimateUDEB for `n_years` under constant forcing and return per-year
/// records of global SAT, SST, heat uptake, and surface temperature.
///
/// Starts from zero initial surface temperatures at year 2000.
pub fn run_udeb_simulation(
    params: &ClimateUDEBParameters,
    erf: FloatValue,
    n_years: usize,
) -> Vec<YearRecord> {
    let component = ClimateUDEB::from_parameters(params.clone()).unwrap();
    let mut state = component.initial_state();
    let mut prev_temps = FourBoxSlice::from_array([0.0, 0.0, 0.0, 0.0]);
    let mut records = Vec::with_capacity(n_years);

    let [nh_ocean, nh_land, sh_ocean, sh_land] = *params.grid().weights();

    for year in 0..n_years {
        let t_current = 2000.0 + year as FloatValue;
        let t_next = t_current + 1.0;

        let (erf_item, surf_item) = build_udeb_input_state(erf, &prev_temps, t_current, t_next);
        let input_state = InputState::build(vec![&erf_item, &surf_item], t_current);

        let output = component
            .solve_with_state(t_current, t_next, &input_state, &mut state)
            .unwrap_or_else(|e| {
                panic!("solve_with_state failed at year {}: {:?}", year, e);
            });

        if let Some(StateValue::FourBox(temps)) = output.get("Surface Air Temperature Change") {
            prev_temps = *temps;
        } else {
            panic!(
                "Year {}: Surface Air Temperature Change not found in output",
                year
            );
        }

        let global_sat = prev_temps.0[0] * nh_ocean
            + prev_temps.0[1] * nh_land
            + prev_temps.0[2] * sh_ocean
            + prev_temps.0[3] * sh_land;

        let sst = match output.get("Sea Surface Temperature") {
            Some(StateValue::Scalar(v)) => *v,
            _ => panic!("Year {}: SST not found in output", year),
        };

        let heat_uptake = match output.get("Heat Uptake") {
            Some(StateValue::Scalar(v)) => *v,
            _ => panic!("Year {}: Heat Uptake not found in output", year),
        };

        records.push(YearRecord {
            global_sat,
            sst,
            heat_uptake,
            surface_temperature: prev_temps,
        });
    }

    records
}
