//! Shared test helpers for rscm-magicc integration tests.
//!
//! Provides common parameter construction, equilibrium temperature computation,
//! and ClimateUDEB input state builders used across multiple test files.

use ndarray::Array2;
use rscm_core::component::{Component, InputState};
use rscm_core::interpolate::strategies::{InterpolationStrategy, LinearSplineStrategy};
use rscm_core::spatial::FourBoxGrid;
use rscm_core::state::{FourBoxSlice, StateValue};
use rscm_core::timeseries::{FloatValue, GridTimeseries, TimeAxis, Timeseries};
use rscm_core::timeseries_collection::{TimeseriesData, TimeseriesItem, VariableType};
use rscm_core::utils::linear_algebra::invert_4x4;
use rscm_magicc::climate::lamcalc::{build_coupling_matrix, LamcalcParams};
use rscm_magicc::climate::{ClimateUDEB, ClimateUDEBState};
use rscm_magicc::parameters::ClimateUDEBParameters;
use std::sync::Arc;

/// Convergence tolerance used by LAMCALC internally.
pub const RLO_TOLERANCE: FloatValue = 0.001;

// ---------------------------------------------------------------------------
// LAMCALC parameter construction
// ---------------------------------------------------------------------------

/// Create default LAMCALC parameters matching MAGICC defaults.
pub fn default_lamcalc_params() -> LamcalcParams {
    let nh_land = 0.42;
    let sh_land = 0.21;
    LamcalcParams {
        q_2xco2: 3.71,
        k_lo: 1.44,
        k_ns: 0.31,
        ecs: 3.0,
        rlo: 1.317,
        amplify_ocean_to_land: 1.02,
        fgno: 0.5 - nh_land / 2.0,
        fgnl: nh_land / 2.0,
        fgso: 0.5 - sh_land / 2.0,
        fgsl: sh_land / 2.0,
    }
}

/// Build LAMCALC params from ClimateUDEB parameters for consistency.
pub fn lamcalc_params_from_udeb(params: &ClimateUDEBParameters) -> LamcalcParams {
    let (fgno, fgnl, fgso, fgsl) = params.global_box_fractions();
    LamcalcParams {
        q_2xco2: params.rf_2xco2,
        k_lo: params.k_lo,
        k_ns: params.k_ns,
        ecs: params.ecs,
        rlo: params.rlo,
        amplify_ocean_to_land: params.amplify_ocean_to_land,
        fgno,
        fgnl,
        fgso,
        fgsl,
    }
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
        feedback_cumt_sensitivity: 0.0,
        feedback_q_sensitivity: 0.0,
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
/// Solves $M \cdot T = Q \cdot f$ where $M$ is the coupling matrix, $Q$ is the
/// radiative forcing for CO2 doubling, and $f$ is the area fraction vector.
pub fn compute_equilibrium_temperatures(
    params: &LamcalcParams,
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

    let area = [params.fgno, params.fgnl, params.fgso, params.fgsl];
    let q = params.q_2xco2;

    let mut box_temps = [0.0_f64; 4];
    for row in 0..4 {
        let mut sum = 0.0;
        for col in 0..4 {
            sum += inv[row][col] * area[col];
        }
        box_temps[row] = q * sum;
    }

    let ocean_mean =
        (params.fgno * box_temps[0] + params.fgso * box_temps[2]) / (params.fgno + params.fgso);
    let land_mean =
        (params.fgnl * box_temps[1] + params.fgsl * box_temps[3]) / (params.fgnl + params.fgsl);
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
/// single timestep.
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
        variable_type: VariableType::Exogenous,
    };

    // FourBox surface temperature state
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
        name: "Surface Temperature".to_string(),
        variable_type: VariableType::Endogenous,
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
    let mut state = ClimateUDEBState::new(params.n_layers, params.w_initial);
    let mut prev_temps = FourBoxSlice::from_array([0.0, 0.0, 0.0, 0.0]);
    let mut records = Vec::with_capacity(n_years);

    let (fgno, fgnl, fgso, fgsl) = params.global_box_fractions();

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

        if let Some(StateValue::FourBox(temps)) = output.get("Surface Temperature") {
            prev_temps = *temps;
        } else {
            panic!("Year {}: Surface Temperature not found in output", year);
        }

        let global_sat = prev_temps.0[0] * fgno
            + prev_temps.0[1] * fgnl
            + prev_temps.0[2] * fgso
            + prev_temps.0[3] * fgsl;

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
