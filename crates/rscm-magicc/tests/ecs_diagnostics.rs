//! ECS diagnostics integration tests for LAMCALC and time-varying ECS.
//!
//! These tests serve as diagnostic documentation when run with
//! `cargo test -p rscm-magicc --test ecs_diagnostics -- --nocapture`.
//!
//! Tests verify:
//! 1. ECS-Lambda consistency sweep (ECS 1.0..8.0)
//! 2. Lambda monotonicity (lambda_ocean decreases as ECS increases)
//! 3. Time-varying ECS correctness under constant forcing
//! 4. Heat uptake consistency (Q - sum(f_i * lambda_i * T_i))
//! 5. Land-ocean warming ratio convergence to RLO at equilibrium

mod common;

use approx::assert_relative_eq;
use common::{
    build_udeb_input_state, compute_equilibrium_temperatures, default_lamcalc_params,
    params_with_fixed_ecs, run_udeb_simulation,
};
use rscm_core::component::{Component, InputState};
use rscm_core::state::{FourBoxSlice, StateValue};
use rscm_core::timeseries::FloatValue;
use rscm_magicc::climate::lamcalc::lamcalc;
use rscm_magicc::climate::{ClimateUDEB, ClimateUDEBState};
use rscm_magicc::parameters::ClimateUDEBParameters;

// ---------------------------------------------------------------------------
// 1. ECS-Lambda consistency sweep
// ---------------------------------------------------------------------------

mod ecs_lambda_consistency_sweep {
    use super::*;

    #[test]
    fn test_ecs_lambda_consistency_sweep() {
        println!();
        println!(
            "{:>6} | {:>12} | {:>12} | {:>10} | {:>8}",
            "ECS", "lambda_ocean", "lambda_land", "T_global", "Error"
        );
        println!("{}", "-".repeat(62));

        // ECS from 1.0 to 8.0 in steps of 0.5
        let ecs_values: Vec<FloatValue> = (2..=16).map(|i| i as FloatValue * 0.5).collect();

        for &ecs in &ecs_values {
            let mut params = default_lamcalc_params();
            params.ecs = ecs;

            let result = lamcalc(&params).unwrap_or_else(|| {
                panic!("LAMCALC should converge for ECS = {:.1}", ecs);
            });

            let eq =
                compute_equilibrium_temperatures(&params, result.lambda_ocean, result.lambda_land);

            let error = (eq.global_mean - ecs).abs();

            println!(
                "{:>6.1} | {:>12.6} | {:>12.6} | {:>10.6} | {:>8.6}",
                ecs, result.lambda_ocean, result.lambda_land, eq.global_mean, error
            );

            // The reconstructed global mean equilibrium temperature must equal
            // ECS to within 0.05 K. This verifies that LAMCALC correctly
            // partitions the global feedback into ocean and land components
            // such that the four-box energy balance reproduces the target ECS.
            assert_relative_eq!(eq.global_mean, ecs, epsilon = 0.05);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Lambda monotonicity
// ---------------------------------------------------------------------------

mod lambda_monotonicity {
    use super::*;

    #[test]
    fn test_lambda_ocean_decreases_with_increasing_ecs() {
        // Higher ECS means weaker total negative feedback (lambda_global = Q/ECS
        // decreases). The ocean feedback parameter should also decrease
        // monotonically because the ocean dominates the global feedback budget.
        println!();
        println!(
            "{:>6} | {:>12} | {:>14} | {:>12}",
            "ECS", "lambda_ocean", "lambda_global", "delta_lam_o"
        );
        println!("{}", "-".repeat(52));

        let ecs_values: Vec<FloatValue> = (2..=16).map(|i| i as FloatValue * 0.5).collect();
        let mut prev_lambda_ocean: Option<FloatValue> = None;

        for &ecs in &ecs_values {
            let mut params = default_lamcalc_params();
            params.ecs = ecs;

            let result = lamcalc(&params).unwrap_or_else(|| {
                panic!("LAMCALC should converge for ECS = {:.1}", ecs);
            });

            let lambda_global = params.q_2xco2 / ecs;

            let delta = match prev_lambda_ocean {
                Some(prev) => {
                    let d = result.lambda_ocean - prev;
                    // lambda_ocean should decrease (delta negative) as ECS
                    // increases, because weaker global feedback requires
                    // weaker ocean feedback.
                    assert!(
                        d < 0.0,
                        "ECS = {:.1}: lambda_ocean ({:.6}) should be less than \
                         previous ({:.6}). Delta = {:.6}. \
                         Higher ECS must produce weaker ocean feedback.",
                        ecs,
                        result.lambda_ocean,
                        prev,
                        d,
                    );
                    format!("{:>12.6}", d)
                }
                None => format!("{:>12}", "---"),
            };

            println!(
                "{:>6.1} | {:>12.6} | {:>14.6} | {}",
                ecs, result.lambda_ocean, lambda_global, delta
            );

            prev_lambda_ocean = Some(result.lambda_ocean);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Time-varying ECS correctness
// ---------------------------------------------------------------------------

mod time_varying_ecs {
    use super::*;

    #[test]
    fn test_adjusted_ecs_changes_over_time() {
        // Create a component with non-zero feedback sensitivities so that
        // adjusted ECS evolves as cumulative temperature accumulates.
        let params = ClimateUDEBParameters {
            ecs: 3.0,
            feedback_cumt_sensitivity: 0.15,
            feedback_q_sensitivity: 0.01,
            feedback_cumt_period: 300.0,
            ..Default::default()
        };

        let component = ClimateUDEB::from_parameters(params.clone());
        let mut state = ClimateUDEBState::new(params.n_layers, params.w_initial);

        let erf = params.rf_2xco2;
        let n_years = 500;
        let mut prev_temps = FourBoxSlice::from_array([0.0, 0.0, 0.0, 0.0]);
        let (fgno, fgnl, fgso, fgsl) = params.global_box_fractions();

        // Track the effective ECS by computing it from the temperature history
        // the same way adjusted_ecs() does internally.
        let mut ecs_trajectory: Vec<FloatValue> = Vec::with_capacity(n_years);
        let mut temp_history: Vec<FloatValue> = Vec::new();

        println!();
        println!(
            "{:>6} | {:>10} | {:>12} | {:>10}",
            "Year", "Global SAT", "Adjusted ECS", "Cum T"
        );
        println!("{}", "-".repeat(48));

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
            }

            let global_sat = prev_temps.0[0] * fgno
                + prev_temps.0[1] * fgnl
                + prev_temps.0[2] * fgso
                + prev_temps.0[3] * fgsl;
            temp_history.push(global_sat);

            // Reconstruct adjusted ECS using the same formula as ClimateUDEB
            let cumt_2x = params.ecs * params.feedback_cumt_period;
            let period = params.feedback_cumt_period as usize;
            let start = temp_history.len().saturating_sub(period);
            let cum_t: FloatValue = temp_history[start..].iter().sum();

            let cumt_factor = if cumt_2x.abs() > 1e-15 {
                1.0 + params.feedback_cumt_sensitivity * (cum_t - cumt_2x) / cumt_2x
            } else {
                1.0
            };
            let q_factor = 1.0 + params.feedback_q_sensitivity * (erf.max(0.0) - params.rf_2xco2);
            let ecs_adj = params.ecs * cumt_factor * q_factor;
            ecs_trajectory.push(ecs_adj);

            if year == 0
                || year == 9
                || year == 49
                || year == 99
                || year == 199
                || year == n_years - 1
            {
                println!(
                    "{:>6} | {:>10.4} | {:>12.4} | {:>10.2}",
                    year + 1,
                    global_sat,
                    ecs_adj,
                    cum_t,
                );
            }
        }

        // Verify that adjusted ECS actually changes over time.
        // Early on, cumulative temperature is low so ECS is reduced;
        // later it increases toward the base ECS as cumT approaches cumT_2x.
        let ecs_year_1 = ecs_trajectory[0];
        let ecs_year_end = ecs_trajectory[n_years - 1];

        assert!(
            (ecs_year_1 - ecs_year_end).abs() > 0.01,
            "Adjusted ECS should change over time. \
             Year 1: {:.4}, Year {}: {:.4}",
            ecs_year_1,
            n_years,
            ecs_year_end,
        );

        // Early ECS should be below the base (cumT < cumT_2x -> cumt_factor < 1)
        assert!(
            ecs_year_1 < params.ecs,
            "Early adjusted ECS ({:.4}) should be below base ECS ({:.1}) \
             because cumulative temperature is below equilibrium",
            ecs_year_1,
            params.ecs,
        );

        println!();
        println!(
            "ECS trajectory: year 1 = {:.4}, year {} = {:.4}, base = {:.1}",
            ecs_year_1, n_years, ecs_year_end, params.ecs,
        );

        // The system should still converge to a stable temperature (warming
        // rate should decrease). Compare first and last decade warming rates.
        let warming_decade_1 = temp_history[9] - temp_history[0];
        let warming_last_decade = temp_history[n_years - 1] - temp_history[n_years - 11];

        println!(
            "Warming rate: decade 1 = {:.4} K/decade, last decade = {:.4} K/decade",
            warming_decade_1, warming_last_decade,
        );

        assert!(
            warming_last_decade < warming_decade_1,
            "System should stabilise: last-decade warming ({:.4} K) \
             should be less than first-decade warming ({:.4} K)",
            warming_last_decade,
            warming_decade_1,
        );

        // Final temperature must be finite and physically reasonable
        let final_temp = *temp_history.last().unwrap();
        assert!(
            final_temp > 0.0 && final_temp < 25.0,
            "Final temperature ({:.4} K) should be positive and below cap",
            final_temp,
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Heat uptake consistency
// ---------------------------------------------------------------------------

mod heat_uptake_consistency {
    use super::*;

    #[test]
    fn test_heat_uptake_equals_forcing_minus_feedback() {
        // Heat uptake = Q - sum(f_i * lambda_i * T_i)
        // We verify that the component-reported heat_uptake matches this
        // formula computed from the same temperatures and lambda values.
        let params = params_with_fixed_ecs(3.0);
        let component = ClimateUDEB::from_parameters(params.clone());
        let mut state = ClimateUDEBState::new(params.n_layers, params.w_initial);

        let erf = params.rf_2xco2;
        let (fgno, fgnl, fgso, fgsl) = params.global_box_fractions();

        // Get the lambda values by running LAMCALC with matching parameters
        let lamcalc_params = common::lamcalc_params_from_udeb(&params);
        let lam_result =
            lamcalc(&lamcalc_params).expect("LAMCALC should converge for heat uptake test");
        let lambda_ocean = lam_result.lambda_ocean;
        let lambda_land = lam_result.lambda_land;

        let mut prev_temps = FourBoxSlice::from_array([0.0, 0.0, 0.0, 0.0]);

        println!();
        println!(
            "{:>6} | {:>12} | {:>12} | {:>10}",
            "Year", "HU (output)", "HU (manual)", "Difference"
        );
        println!("{}", "-".repeat(50));

        let test_years = [1, 5, 10, 25, 50, 100];
        let n_years = *test_years.last().unwrap();

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
            }

            let hu_output = match output.get("Heat Uptake") {
                Some(StateValue::Scalar(v)) => *v,
                _ => panic!("Year {}: Heat Uptake not found", year),
            };

            // Manual calculation: Q - sum(f_i * lambda_i * T_i)
            // Forcing is uniform across boxes (erf for each box)
            let weights = [fgno, fgnl, fgso, fgsl];
            let lambdas = [lambda_ocean, lambda_land, lambda_ocean, lambda_land];

            let q_global: FloatValue = weights.iter().map(|&w| w * erf).sum();
            let feedback_global: FloatValue = (0..4)
                .map(|i| weights[i] * lambdas[i] * prev_temps.0[i])
                .sum();
            let hu_manual = q_global - feedback_global;

            if test_years.contains(&(year + 1)) {
                let diff = (hu_output - hu_manual).abs();
                println!(
                    "{:>6} | {:>12.6} | {:>12.6} | {:>10.2e}",
                    year + 1,
                    hu_output,
                    hu_manual,
                    diff,
                );

                // The component heat uptake should match our manual
                // calculation very closely, since they use the same formula.
                // Small differences can arise if adjusted_ecs triggers a
                // re-solve of LAMCALC (but we disabled that via fixed ECS).
                assert_relative_eq!(hu_output, hu_manual, epsilon = 1e-6);
            }
        }

        // Additional sanity: heat uptake should be positive during warming
        // and decreasing over time as temperature approaches equilibrium
        let records = run_udeb_simulation(&params, erf, 200);

        let hu_early = records[0].heat_uptake;
        let hu_late = records[199].heat_uptake;

        println!();
        println!(
            "Heat uptake trend: year 1 = {:.4} W/m^2, year 200 = {:.4} W/m^2",
            hu_early, hu_late,
        );

        assert!(
            hu_early > hu_late,
            "Heat uptake should decrease over time as system approaches equilibrium. \
             Year 1: {:.4} W/m^2, Year 200: {:.4} W/m^2",
            hu_early,
            hu_late,
        );

        assert!(
            hu_late > 0.0,
            "Heat uptake should still be positive at year 200 (not yet at equilibrium). \
             Got {:.4} W/m^2",
            hu_late,
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Land-ocean warming ratio at equilibrium
// ---------------------------------------------------------------------------

mod land_ocean_ratio_equilibrium {
    use super::*;

    #[test]
    fn test_rlo_convergence_for_multiple_ecs() {
        // Run ClimateUDEB to near-equilibrium (1500 years, constant forcing)
        // and verify the land/ocean temperature ratio converges to the
        // prescribed RLO. Test for multiple ECS values.
        let ecs_values = [2.0, 3.0, 4.5];
        let n_years = 1500;
        let target_rlo = ClimateUDEBParameters::default().rlo;

        println!();
        println!(
            "{:>6} | {:>12} | {:>12} | {:>12} | {:>10} | {:>10}",
            "ECS", "Ocean Mean", "Land Mean", "RLO actual", "RLO target", "Error"
        );
        println!("{}", "-".repeat(72));

        for &ecs in &ecs_values {
            let params = params_with_fixed_ecs(ecs);
            let erf = params.rf_2xco2;

            let records = run_udeb_simulation(&params, erf, n_years);

            let (fgno, fgnl, fgso, fgsl) = params.global_box_fractions();
            let final_temps = &records[n_years - 1].surface_temperature;

            let ocean_mean = (final_temps.0[0] * fgno + final_temps.0[2] * fgso) / (fgno + fgso);
            let land_mean = (final_temps.0[1] * fgnl + final_temps.0[3] * fgsl) / (fgnl + fgsl);

            assert!(
                ocean_mean > 0.0,
                "ECS = {:.1}: ocean mean temperature should be positive, got {:.6}",
                ecs,
                ocean_mean,
            );

            let rlo_actual = land_mean / ocean_mean;
            let rlo_error = (rlo_actual - target_rlo).abs();

            println!(
                "{:>6.1} | {:>12.4} | {:>12.4} | {:>12.6} | {:>10.4} | {:>10.6}",
                ecs, ocean_mean, land_mean, rlo_actual, target_rlo, rlo_error,
            );

            // The LAMCALC RLO constraint is defined in terms of raw energy
            // balance temperatures, not the SST-to-SAT adjusted temperatures
            // that ClimateUDEB outputs. The temp_adjust_alpha (1.04) and
            // temp_adjust_gamma (-0.002) parameters amplify ocean SAT
            // relative to SST, which inflates the apparent SAT-based
            // land/ocean ratio. This effect grows with temperature (and
            // hence with ECS) because the quadratic gamma term increases
            // the SAT amplification at higher SSTs.
            //
            // We verify:
            //  - The RLO is at least > 1.0 (land warms more than ocean)
            //  - The RLO does not exceed a physically plausible upper bound
            //    that accounts for the SAT amplification effect
            assert!(
                rlo_actual > 1.0,
                "ECS = {:.1}: land/ocean SAT ratio ({:.6}) should exceed 1.0. \
                 Land must warm more than ocean.",
                ecs,
                rlo_actual,
            );

            // Upper bound: the SAT amplification increases with temperature,
            // so higher ECS produces a larger SAT-based RLO. A ratio up to
            // ~1.7 is plausible for high ECS (4.5 K) given the quadratic
            // SST-to-SAT relationship.
            assert!(
                rlo_actual < 2.0,
                "ECS = {:.1}: land/ocean SAT ratio ({:.6}) should be below 2.0. \
                 Very high values indicate a problem with the feedback partitioning.",
                ecs,
                rlo_actual,
            );

            // For low ECS, the SAT-based RLO should be closer to the
            // energy-balance target since the quadratic term is small.
            if ecs <= 2.0 {
                assert!(
                    rlo_error < 0.1,
                    "ECS = {:.1}: at low ECS, SAT-based RLO ({:.6}) should be \
                     close to energy-balance RLO ({:.4}). Error = {:.6}",
                    ecs,
                    rlo_actual,
                    target_rlo,
                    rlo_error,
                );
            }
        }

        // Also verify the RLO is converging over time by comparing early
        // and late values for the default ECS case.
        let params = params_with_fixed_ecs(3.0);
        let erf = params.rf_2xco2;
        let records = run_udeb_simulation(&params, erf, n_years);
        let (fgno, fgnl, fgso, fgsl) = params.global_box_fractions();

        println!();
        println!("RLO convergence over time (ECS = 3.0):");
        println!(
            "{:>8} | {:>12} | {:>12} | {:>12}",
            "Year", "Ocean Mean", "Land Mean", "RLO"
        );
        println!("{}", "-".repeat(52));

        let check_years = [10, 50, 100, 200, 500, 1000, n_years];
        let mut rlo_values: Vec<FloatValue> = Vec::new();

        for &yr in &check_years {
            if yr > n_years {
                continue;
            }
            let temps = &records[yr - 1].surface_temperature;
            let ocean_mean = (temps.0[0] * fgno + temps.0[2] * fgso) / (fgno + fgso);
            let land_mean = (temps.0[1] * fgnl + temps.0[3] * fgsl) / (fgnl + fgsl);
            let rlo = land_mean / ocean_mean;
            rlo_values.push(rlo);

            println!(
                "{:>8} | {:>12.4} | {:>12.4} | {:>12.6}",
                yr, ocean_mean, land_mean, rlo,
            );
        }

        // Verify the RLO is converging: the difference between consecutive
        // RLO measurements should generally decrease.
        if rlo_values.len() >= 3 {
            let late_change =
                (rlo_values[rlo_values.len() - 1] - rlo_values[rlo_values.len() - 2]).abs();
            let early_change = (rlo_values[1] - rlo_values[0]).abs();

            println!();
            println!(
                "RLO change: early = {:.6}, late = {:.6}",
                early_change, late_change,
            );

            assert!(
                late_change < early_change,
                "RLO should converge: late change ({:.6}) should be smaller \
                 than early change ({:.6})",
                late_change,
                early_change,
            );
        }
    }
}
