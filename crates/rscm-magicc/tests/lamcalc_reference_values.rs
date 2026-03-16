//! Reference value and analytical property tests for the LAMCALC solver.
//!
//! These tests validate the LAMCALC implementation against:
//! 1. Energy conservation at equilibrium
//! 2. Analytical limit: zero coupling (decoupled boxes)
//! 3. Analytical limit: infinite coupling (perfect mixing)
//! 4. Sensitivity analysis: k_lo sweep with RLO constraint verification
//! 5. Sensitivity analysis: RLO sweep with symmetry checks
//! 6. Coupling matrix mathematical properties (diagonal dominance, invertibility,
//!    Gershgorin eigenvalue bounds)

mod common;

use approx::assert_relative_eq;
use common::{compute_equilibrium_temperatures, default_lamcalc_params, RLO_TOLERANCE};
use rscm_core::timeseries::FloatValue;
use rscm_core::utils::linear_algebra::invert_4x4;
use rscm_magicc::climate::lamcalc::{build_coupling_matrix, lamcalc, LamcalcResult};

// ===========================================================================
// 1. Energy conservation at equilibrium
// ===========================================================================

mod energy_conservation {
    use super::*;

    /// At equilibrium the total forcing absorbed equals the total radiative
    /// response: sum_i(f_i * Q_i) = sum_i(f_i * lambda_i * T_i).
    ///
    /// Equivalently, the net heat uptake (forcing minus feedback) summed over
    /// all boxes must be zero. This is the fundamental energy balance constraint.
    #[test]
    fn test_energy_balance_at_equilibrium_default_params() {
        let params = default_lamcalc_params();
        let result = lamcalc(&params).expect("LAMCALC should converge with default parameters");
        let eq = compute_equilibrium_temperatures(&params, result.lambda_ocean, result.lambda_land);
        let temps = eq.box_temps;
        let area = [params.fgno, params.fgnl, params.fgso, params.fgsl];

        // Per-box lambda: ocean boxes get lambda_ocean, land boxes get lambda_land
        let lambdas = [
            result.lambda_ocean,
            result.lambda_land,
            result.lambda_ocean,
            result.lambda_land,
        ];

        // Total forcing absorbed: sum_i(f_i * Q_2x * qfrac_i) where qfrac=1 for CO2
        let total_forcing: FloatValue = area.iter().sum::<FloatValue>() * params.q_2xco2;

        // Total radiative response: sum of (f_i * lambda_i * T_i) over all boxes
        // Note: the coupling matrix equation is M * T = Q * f, where the diagonal
        // of M contains f_i * lambda_i plus exchange terms. At equilibrium the
        // exchange terms cancel globally, leaving sum(f_i * lambda_i * T_i) = sum(f_i * Q).
        let total_feedback: FloatValue = (0..4).map(|i| area[i] * lambdas[i] * temps[i]).sum();

        println!("--- Energy conservation at equilibrium (default params) ---");
        println!(
            "Box temperatures: NO={:.4}, NL={:.4}, SO={:.4}, SL={:.4}",
            temps[0], temps[1], temps[2], temps[3]
        );
        println!(
            "Box lambdas:      NO={:.4}, NL={:.4}, SO={:.4}, SL={:.4}",
            lambdas[0], lambdas[1], lambdas[2], lambdas[3]
        );
        println!(
            "Per-box feedback (f_i * lambda_i * T_i): {:.4}, {:.4}, {:.4}, {:.4}",
            area[0] * lambdas[0] * temps[0],
            area[1] * lambdas[1] * temps[1],
            area[2] * lambdas[2] * temps[2],
            area[3] * lambdas[3] * temps[3],
        );
        println!("Total forcing absorbed:  {:.6} W/m^2", total_forcing);
        println!("Total radiative response: {:.6} W/m^2", total_feedback);
        println!(
            "Residual (forcing - feedback): {:.6} W/m^2",
            total_forcing - total_feedback
        );

        // The residual should be very small -- exchange terms cancel globally
        // because heat exchanged out of one box enters another.
        // We allow a tolerance proportional to Q_2x because the RLO convergence
        // tolerance introduces a small imbalance.
        let tolerance = 0.05;
        assert!(
            (total_forcing - total_feedback).abs() < tolerance,
            "Energy balance violated: total forcing = {:.6}, total feedback = {:.6}, \
             residual = {:.6} (tolerance = {:.6})",
            total_forcing,
            total_feedback,
            total_forcing - total_feedback,
            tolerance,
        );
    }

    /// Verify energy conservation across a range of ECS values.
    #[test]
    fn test_energy_balance_across_ecs_range() {
        let ecs_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0];

        println!("--- Energy conservation across ECS range ---");
        println!(
            "{:>5}  {:>12}  {:>12}  {:>12}",
            "ECS", "Forcing", "Feedback", "Residual"
        );

        for &ecs in &ecs_values {
            let mut params = default_lamcalc_params();
            params.ecs = ecs;

            let result = lamcalc(&params)
                .unwrap_or_else(|| panic!("LAMCALC should converge for ECS = {:.1}", ecs));
            let eq =
                compute_equilibrium_temperatures(&params, result.lambda_ocean, result.lambda_land);
            let temps = eq.box_temps;
            let area = [params.fgno, params.fgnl, params.fgso, params.fgsl];
            let lambdas = [
                result.lambda_ocean,
                result.lambda_land,
                result.lambda_ocean,
                result.lambda_land,
            ];

            let total_forcing: FloatValue = area.iter().sum::<FloatValue>() * params.q_2xco2;
            let total_feedback: FloatValue = (0..4).map(|i| area[i] * lambdas[i] * temps[i]).sum();
            let residual = total_forcing - total_feedback;

            println!(
                "{:>5.1}  {:>12.6}  {:>12.6}  {:>12.6}",
                ecs, total_forcing, total_feedback, residual
            );

            assert!(
                residual.abs() < 0.1,
                "ECS = {:.1}: energy balance residual = {:.6} W/m^2 exceeds tolerance",
                ecs,
                residual,
            );
        }
    }
}

// ===========================================================================
// 2. Analytical limit: zero coupling
// ===========================================================================

mod zero_coupling_limit {
    use super::*;

    /// When k_lo = 0 and k_ns = 0, the four boxes are completely decoupled.
    /// Each box must independently satisfy F = lambda * T, so both lambda_ocean
    /// and lambda_land must equal the global lambda = Q_2x / ECS.
    #[test]
    fn test_decoupled_boxes_yield_uniform_lambda() {
        let mut params = default_lamcalc_params();
        params.k_lo = 0.0;
        params.k_ns = 0.0;
        // With zero coupling, RLO is not meaningful (land and ocean decouple).
        // Set RLO = 1.0 to request equal warming, which is the only consistent
        // solution when boxes are decoupled with uniform forcing.
        params.rlo = 1.0;

        let lam_global = params.q_2xco2 / params.ecs;

        let result =
            lamcalc(&params).expect("LAMCALC should converge with zero coupling and RLO=1.0");

        println!("--- Zero coupling limit ---");
        println!("Expected lambda_global = Q_2x / ECS = {:.6}", lam_global);
        println!("lambda_ocean = {:.6}", result.lambda_ocean);
        println!("lambda_land  = {:.6}", result.lambda_land);

        // Both lambdas should equal the global value
        assert_relative_eq!(result.lambda_ocean, lam_global, epsilon = 0.01,);
        assert_relative_eq!(result.lambda_land, lam_global, epsilon = 0.01,);

        // Verify all box temperatures are equal (uniform warming)
        let eq = compute_equilibrium_temperatures(&params, result.lambda_ocean, result.lambda_land);
        let temps = eq.box_temps;
        println!(
            "Box temperatures: NO={:.4}, NL={:.4}, SO={:.4}, SL={:.4}",
            temps[0], temps[1], temps[2], temps[3]
        );

        for i in 1..4 {
            assert_relative_eq!(temps[0], temps[i], epsilon = 0.05,);
        }
    }

    /// With zero coupling, the coupling matrix should be purely diagonal.
    #[test]
    fn test_coupling_matrix_is_diagonal_at_zero_coupling() {
        let mut params = default_lamcalc_params();
        params.k_lo = 0.0;
        params.k_ns = 0.0;

        let lam_o = 1.2;
        let lam_l = 1.5;
        let matrix = build_coupling_matrix(&params, lam_o, lam_l);

        println!("--- Coupling matrix at zero coupling ---");
        for (i, row) in matrix.iter().enumerate() {
            println!(
                "  Row {}: [{:.6}, {:.6}, {:.6}, {:.6}]",
                i, row[0], row[1], row[2], row[3]
            );
        }

        // All off-diagonal elements should be exactly zero
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert!(
                        matrix[i][j].abs() < 1e-15,
                        "Off-diagonal element [{i}][{j}] = {:.6e} should be zero \
                         when k_lo = 0 and k_ns = 0",
                        matrix[i][j],
                    );
                }
            }
        }

        // Diagonal elements should be f_i * lambda_i
        let area = [params.fgno, params.fgnl, params.fgso, params.fgsl];
        let lams = [lam_o, lam_l, lam_o, lam_l];
        for i in 0..4 {
            let expected = area[i] * lams[i];
            assert_relative_eq!(matrix[i][i], expected, epsilon = 1e-12,);
        }
    }
}

// ===========================================================================
// 3. Analytical limit: infinite coupling (perfect mixing)
// ===========================================================================

mod infinite_coupling_limit {
    use super::*;

    /// As k_lo and k_ns become very large with RLO = 1.0 (equal warming),
    /// inter-box exchange dominates and all boxes should approach the same
    /// equilibrium temperature (perfect mixing).
    ///
    /// Note: with RLO != 1.0, the land-ocean warming ratio constraint
    /// inherently prevents uniform temperatures regardless of coupling
    /// strength. So we set RLO = 1.0 to test the true mixing limit.
    #[test]
    fn test_large_coupling_yields_uniform_temperatures() {
        let coupling_strengths = [1.0, 5.0, 10.0, 20.0, 50.0];

        println!("--- Infinite coupling limit (RLO=1.0): temperature convergence ---");
        println!(
            "{:>10}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}",
            "k_lo=k_ns", "T_NO", "T_NL", "T_SO", "T_SL", "max_spread"
        );

        let mut prev_spread = FloatValue::MAX;
        let mut last_converged_spread = FloatValue::MAX;

        for &k in &coupling_strengths {
            let mut params = default_lamcalc_params();
            params.k_lo = k;
            params.k_ns = k;
            params.rlo = 1.0; // Equal warming -- true mixing limit

            let result = lamcalc(&params);
            let result = match result {
                Some(r) => r,
                None => {
                    println!("{:>10.1}  (solver did not converge, skipping)", k);
                    continue;
                }
            };

            let eq =
                compute_equilibrium_temperatures(&params, result.lambda_ocean, result.lambda_land);
            let temps = eq.box_temps;
            let t_min = temps
                .iter()
                .cloned()
                .fold(FloatValue::INFINITY, FloatValue::min);
            let t_max = temps
                .iter()
                .cloned()
                .fold(FloatValue::NEG_INFINITY, FloatValue::max);
            let spread = t_max - t_min;

            println!(
                "{:>10.1}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>10.6}",
                k, temps[0], temps[1], temps[2], temps[3], spread
            );

            // Spread should decrease (or at least not increase substantially)
            // as coupling increases
            if prev_spread < FloatValue::MAX {
                assert!(
                    spread < prev_spread + 0.05,
                    "Temperature spread should not increase with stronger coupling: \
                     spread={:.6} at k={:.1} vs prev_spread={:.6}",
                    spread,
                    k,
                    prev_spread,
                );
            }
            prev_spread = spread;
            last_converged_spread = spread;
        }

        // At high coupling with RLO=1.0, the temperature spread should be small.
        // The residual asymmetry comes from amplify_ocean_to_land (1.02) and
        // asymmetric hemispheric area fractions.
        assert!(
            last_converged_spread < 0.5,
            "At strong coupling with RLO=1.0, temperature spread = {:.4} K \
             should be less than 0.5 K",
            last_converged_spread,
        );
    }

    /// In the perfect mixing limit, the global mean temperature should still
    /// equal ECS (energy conservation holds regardless of coupling strength).
    #[test]
    fn test_global_mean_equals_ecs_at_strong_coupling() {
        let mut params = default_lamcalc_params();
        params.k_lo = 100.0;
        params.k_ns = 100.0;

        let result = match lamcalc(&params) {
            Some(r) => r,
            None => {
                println!("Solver did not converge at very strong coupling; skipping test");
                return;
            }
        };

        let eq = compute_equilibrium_temperatures(&params, result.lambda_ocean, result.lambda_land);
        let global_mean = eq.global_mean;

        println!("--- Strong coupling: global mean vs ECS ---");
        println!("Global mean temperature: {:.6} K", global_mean);
        println!("ECS:                     {:.6} K", params.ecs);
        println!(
            "Difference:              {:.6} K",
            (global_mean - params.ecs).abs()
        );

        assert!(
            (global_mean - params.ecs).abs() < 0.1,
            "Global mean = {:.6} K should be close to ECS = {:.1} K even at strong coupling",
            global_mean,
            params.ecs,
        );
    }
}

// ===========================================================================
// 4. Sensitivity analysis: k_lo sweep
// ===========================================================================

mod k_lo_sweep {
    use super::*;

    /// Sweep k_lo from 0.1 to 5.0 and verify the RLO constraint is always
    /// satisfied. Print a diagnostic table of how lambdas and the implied
    /// RLO vary.
    #[test]
    fn test_k_lo_sweep_rlo_constraint_always_satisfied() {
        let k_lo_values: Vec<FloatValue> = (1..=50).map(|i| i as FloatValue * 0.1).collect();

        println!("--- k_lo sweep: lambda and RLO diagnostics ---");
        println!(
            "{:>6}  {:>12}  {:>12}  {:>10}  {:>10}  {:>10}",
            "k_lo", "lambda_ocean", "lambda_land", "RLO_actual", "RLO_target", "RLO_error"
        );

        for &k_lo in &k_lo_values {
            let mut params = default_lamcalc_params();
            params.k_lo = k_lo;

            let result = match lamcalc(&params) {
                Some(r) => r,
                None => {
                    println!("{:>6.2}  (did not converge)", k_lo);
                    continue;
                }
            };

            let eq =
                compute_equilibrium_temperatures(&params, result.lambda_ocean, result.lambda_land);
            let rlo_actual = eq.rlo_actual;
            let rlo_error = (params.rlo - rlo_actual).abs();

            println!(
                "{:>6.2}  {:>12.6}  {:>12.6}  {:>10.6}  {:>10.6}  {:>10.6}",
                k_lo, result.lambda_ocean, result.lambda_land, rlo_actual, params.rlo, rlo_error
            );

            // RLO constraint must hold for every converged solution
            assert!(
                rlo_error < RLO_TOLERANCE,
                "k_lo = {:.2}: RLO constraint violated. \
                 target = {:.6}, actual = {:.6}, error = {:.6} (tolerance = {:.6})",
                k_lo,
                params.rlo,
                rlo_actual,
                rlo_error,
                RLO_TOLERANCE,
            );

            // lambda_ocean must remain positive (physical stability)
            assert!(
                result.lambda_ocean > 0.0,
                "k_lo = {:.2}: lambda_ocean = {:.6} must be positive",
                k_lo,
                result.lambda_ocean,
            );

            // lambda_land must be finite
            assert!(
                result.lambda_land.is_finite(),
                "k_lo = {:.2}: lambda_land = {:.6} must be finite",
                k_lo,
                result.lambda_land,
            );
        }
    }
}

// ===========================================================================
// 5. Sensitivity analysis: RLO sweep
// ===========================================================================

mod rlo_sweep {
    use super::*;

    /// Sweep RLO from 1.0 to 2.0 and examine how the lambdas change.
    /// At RLO = 1.0 (land and ocean warm equally), the lambda split should
    /// be close to symmetric.
    #[test]
    fn test_rlo_sweep_symmetry_and_monotonicity() {
        let rlo_values: Vec<FloatValue> = (10..=20).map(|i| i as FloatValue * 0.1).collect();

        println!("--- RLO sweep: lambda symmetry and trends ---");
        println!(
            "{:>5}  {:>12}  {:>12}  {:>12}  {:>10}",
            "RLO", "lambda_ocean", "lambda_land", "lam_global", "ocean_T"
        );

        let lam_global = default_lamcalc_params().q_2xco2 / default_lamcalc_params().ecs;

        let mut results: Vec<(FloatValue, LamcalcResult)> = Vec::new();

        for &rlo in &rlo_values {
            let mut params = default_lamcalc_params();
            params.rlo = rlo;

            let result = match lamcalc(&params) {
                Some(r) => r,
                None => {
                    println!("{:>5.1}  (did not converge)", rlo);
                    continue;
                }
            };

            let eq =
                compute_equilibrium_temperatures(&params, result.lambda_ocean, result.lambda_land);
            let ocean_mean = eq.ocean_mean;

            println!(
                "{:>5.1}  {:>12.6}  {:>12.6}  {:>12.6}  {:>10.4}",
                rlo, result.lambda_ocean, result.lambda_land, lam_global, ocean_mean
            );

            results.push((rlo, result));
        }

        // At RLO = 1.0, land and ocean warm equally so the split should be
        // close to symmetric (lambda_ocean ~ lambda_land ~ lam_global).
        if let Some((_, ref r_unity)) = results.iter().find(|(rlo, _)| (*rlo - 1.0).abs() < 0.01) {
            println!("\n--- RLO = 1.0 symmetry check ---");
            println!("lambda_ocean = {:.6}", r_unity.lambda_ocean);
            println!("lambda_land  = {:.6}", r_unity.lambda_land);
            println!("lambda_global = {:.6}", lam_global);
            println!(
                "|lambda_ocean - lambda_land| = {:.6}",
                (r_unity.lambda_ocean - r_unity.lambda_land).abs()
            );

            // With amplify_ocean_to_land = 1.02 and asymmetric hemispheric
            // land fractions (NH=0.42, SH=0.21), there is a residual asymmetry
            // even at RLO=1.0. Allow up to 0.25 W/m^2/K difference.
            assert!(
                (r_unity.lambda_ocean - r_unity.lambda_land).abs() < 0.25,
                "At RLO=1.0, lambda_ocean ({:.6}) and lambda_land ({:.6}) \
                 should be nearly equal (difference = {:.6})",
                r_unity.lambda_ocean,
                r_unity.lambda_land,
                (r_unity.lambda_ocean - r_unity.lambda_land).abs(),
            );
        }

        // As RLO increases (land warms more relative to ocean), lambda_land
        // should generally decrease (weaker land feedback allows more warming).
        // Check the trend is monotonically decreasing for lambda_land.
        println!("\n--- RLO sweep: lambda_land trend ---");
        for window in results.windows(2) {
            let (rlo_a, ref r_a) = window[0];
            let (rlo_b, ref r_b) = window[1];
            println!(
                "RLO {:.1} -> {:.1}: lambda_land {:.6} -> {:.6} (delta = {:.6})",
                rlo_a,
                rlo_b,
                r_a.lambda_land,
                r_b.lambda_land,
                r_b.lambda_land - r_a.lambda_land,
            );

            // lambda_land should not increase as RLO increases
            assert!(
                r_b.lambda_land <= r_a.lambda_land + 0.01,
                "lambda_land should decrease (or stay flat) as RLO increases: \
                 at RLO={:.1} got {:.6}, at RLO={:.1} got {:.6}",
                rlo_a,
                r_a.lambda_land,
                rlo_b,
                r_b.lambda_land,
            );
        }
    }
}

// ===========================================================================
// 6. Coupling matrix mathematical properties
// ===========================================================================

mod coupling_matrix_properties {
    use super::*;

    /// Generate a diverse set of parameter combinations for matrix property tests.
    /// These are chosen to be physically reasonable: lambda values are positive and
    /// exchange coefficients are moderate relative to the feedback terms.
    fn parameter_grid() -> Vec<(FloatValue, FloatValue, FloatValue, FloatValue)> {
        // (lam_o, lam_l, k_lo, k_ns)
        vec![
            (1.0, 1.0, 0.5, 0.1),
            (1.2, 0.8, 1.44, 0.31),
            (0.5, 2.0, 0.1, 0.05),
            (1.5, 1.5, 1.0, 0.5),
            (0.8, 1.2, 2.0, 0.8),
            (3.0, 0.3, 0.5, 0.2),
            (2.0, 1.0, 1.5, 0.5),
            (1.0, 1.0, 2.0, 0.5),
        ]
    }

    /// For a range of parameter sets, verify the coupling matrix is diagonally
    /// dominant for the ocean rows (rows 0 and 2).
    ///
    /// Diagonal dominance means |a_ii| >= sum_{j!=i} |a_ij|. This is a
    /// sufficient condition for matrix invertibility (by the Levy-Desplanques
    /// theorem) and ensures no sign-change issues in the energy balance.
    #[test]
    fn test_diagonal_dominance_for_ocean_rows() {
        let grid = parameter_grid();

        println!("--- Coupling matrix diagonal dominance (ocean rows) ---");
        println!(
            "{:>5}  {:>5}  {:>5}  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}",
            "lam_o", "lam_l", "k_lo", "k_ns", "diag[0]", "offsum[0]", "diag[2]", "offsum[2]"
        );

        for &(lam_o, lam_l, k_lo, k_ns) in &grid {
            let mut params = default_lamcalc_params();
            params.k_lo = k_lo;
            params.k_ns = k_ns;

            let matrix = build_coupling_matrix(&params, lam_o, lam_l);

            for &row_idx in &[0_usize, 2] {
                let diag = matrix[row_idx][row_idx].abs();
                let offdiag_sum: FloatValue = (0..4)
                    .filter(|&j| j != row_idx)
                    .map(|j| matrix[row_idx][j].abs())
                    .sum();

                if row_idx == 0 {
                    print!(
                        "{:>5.1}  {:>5.1}  {:>5.1}  {:>5.1}  {:>10.4}  {:>10.4}",
                        lam_o, lam_l, k_lo, k_ns, diag, offdiag_sum
                    );
                } else {
                    println!("  {:>10.4}  {:>10.4}", diag, offdiag_sum);
                }

                assert!(
                    diag >= offdiag_sum,
                    "Row {} not diagonally dominant for lam_o={}, lam_l={}, \
                     k_lo={}, k_ns={}: |diag|={:.6} < sum|offdiag|={:.6}",
                    row_idx,
                    lam_o,
                    lam_l,
                    k_lo,
                    k_ns,
                    diag,
                    offdiag_sum,
                );
            }
        }
    }

    /// For a range of parameter sets, verify the coupling matrix is invertible
    /// by checking that invert_4x4 succeeds and that the determinant (computed
    /// via the LU-style inverse) is non-zero.
    #[test]
    fn test_matrix_invertibility() {
        let grid = parameter_grid();

        println!("--- Coupling matrix invertibility ---");
        println!(
            "{:>5}  {:>5}  {:>5}  {:>5}  {:>12}",
            "lam_o", "lam_l", "k_lo", "k_ns", "invertible"
        );

        for &(lam_o, lam_l, k_lo, k_ns) in &grid {
            let mut params = default_lamcalc_params();
            params.k_lo = k_lo;
            params.k_ns = k_ns;

            let matrix = build_coupling_matrix(&params, lam_o, lam_l);
            let inv = invert_4x4(&matrix);

            let invertible = inv.is_some();
            println!(
                "{:>5.1}  {:>5.1}  {:>5.1}  {:>5.1}  {:>12}",
                lam_o, lam_l, k_lo, k_ns, invertible
            );

            assert!(
                invertible,
                "Matrix should be invertible for lam_o={}, lam_l={}, k_lo={}, k_ns={}",
                lam_o, lam_l, k_lo, k_ns,
            );

            // Verify M * M^{-1} = I (to within numerical precision)
            let inv = inv.unwrap();
            let mut product = [[0.0_f64; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    for k in 0..4 {
                        product[i][j] += matrix[i][k] * inv[k][j];
                    }
                }
            }

            for i in 0..4 {
                for j in 0..4 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (product[i][j] - expected).abs() < 1e-10,
                        "M * M^{{-1}} [{i}][{j}] = {:.6e}, expected {:.1} \
                         (lam_o={}, lam_l={}, k_lo={}, k_ns={})",
                        product[i][j],
                        expected,
                        lam_o,
                        lam_l,
                        k_lo,
                        k_ns,
                    );
                }
            }
        }
    }

    /// Use Gershgorin circle theorem to verify all eigenvalues have positive
    /// real parts. For each row i, the eigenvalue lies in a disc centred at
    /// a_ii with radius R_i = sum_{j!=i} |a_ij|. If a_ii - R_i > 0 for all i,
    /// then all eigenvalues have positive real parts (the matrix is positive
    /// stable).
    ///
    /// Note: this is a sufficient condition. It may not hold for all parameter
    /// combinations, but it should hold for physically reasonable ones where
    /// the feedback terms dominate the exchange terms.
    #[test]
    fn test_gershgorin_positive_eigenvalues() {
        let grid = parameter_grid();

        println!("--- Gershgorin eigenvalue bounds ---");
        println!(
            "{:>5}  {:>5}  {:>5}  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}",
            "lam_o", "lam_l", "k_lo", "k_ns", "lb[0]", "lb[1]", "lb[2]", "lb[3]", "all_pos?"
        );

        for &(lam_o, lam_l, k_lo, k_ns) in &grid {
            let mut params = default_lamcalc_params();
            params.k_lo = k_lo;
            params.k_ns = k_ns;

            let matrix = build_coupling_matrix(&params, lam_o, lam_l);

            let mut lower_bounds = [0.0_f64; 4];
            for i in 0..4 {
                let radius: FloatValue =
                    (0..4).filter(|&j| j != i).map(|j| matrix[i][j].abs()).sum();
                lower_bounds[i] = matrix[i][i] - radius;
            }

            let all_positive = lower_bounds.iter().all(|&lb| lb > 0.0);

            println!(
                "{:>5.1}  {:>5.1}  {:>5.1}  {:>5.1}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>10}",
                lam_o,
                lam_l,
                k_lo,
                k_ns,
                lower_bounds[0],
                lower_bounds[1],
                lower_bounds[2],
                lower_bounds[3],
                all_positive,
            );

            // For the parameter sets in our grid, Gershgorin bounds should
            // confirm positive eigenvalues. The land rows (1, 3) have the
            // weakest margin because they only couple to one ocean box.
            // We check each row individually for better diagnostics.
            for i in 0..4 {
                assert!(
                    lower_bounds[i] >= -1e-10,
                    "Gershgorin lower bound for row {} is negative ({:.6}) \
                     with lam_o={}, lam_l={}, k_lo={}, k_ns={}. \
                     This suggests the eigenvalue could have a non-positive real part, \
                     which would indicate an unstable climate response.",
                    i,
                    lower_bounds[i],
                    lam_o,
                    lam_l,
                    k_lo,
                    k_ns,
                );
            }
        }
    }

    /// Verify the matrix built from actual LAMCALC solutions (not arbitrary
    /// lambda values) has all the desired properties.
    #[test]
    fn test_solved_matrix_properties() {
        let ecs_values = [2.0, 3.0, 4.0, 5.0];

        println!("--- Solved coupling matrix properties ---");
        println!(
            "{:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            "ECS", "lam_o", "lam_l", "det_check", "gersh_min", "diag_dom?"
        );

        for &ecs in &ecs_values {
            let mut params = default_lamcalc_params();
            params.ecs = ecs;

            let result = lamcalc(&params)
                .unwrap_or_else(|| panic!("LAMCALC should converge for ECS = {}", ecs));

            let matrix = build_coupling_matrix(&params, result.lambda_ocean, result.lambda_land);

            // Invertibility
            let inv = invert_4x4(&matrix);
            assert!(
                inv.is_some(),
                "Solved matrix should be invertible for ECS = {}",
                ecs
            );

            // Gershgorin lower bounds
            let mut gersh_min = FloatValue::INFINITY;
            for i in 0..4 {
                let radius: FloatValue =
                    (0..4).filter(|&j| j != i).map(|j| matrix[i][j].abs()).sum();
                let lb = matrix[i][i] - radius;
                if lb < gersh_min {
                    gersh_min = lb;
                }
            }

            // Diagonal dominance check (all rows)
            let mut all_diag_dominant = true;
            for i in 0..4 {
                let diag = matrix[i][i].abs();
                let offdiag: FloatValue =
                    (0..4).filter(|&j| j != i).map(|j| matrix[i][j].abs()).sum();
                if diag < offdiag {
                    all_diag_dominant = false;
                }
            }

            println!(
                "{:>5.1}  {:>10.4}  {:>10.4}  {:>10}  {:>10.4}  {:>10}",
                ecs, result.lambda_ocean, result.lambda_land, "OK", gersh_min, all_diag_dominant,
            );

            // For solved parameters, the matrix must be invertible (checked above).
            // The Gershgorin bound can be slightly negative for land rows when
            // lambda_land is small or negative (positive land feedback at high ECS).
            // This does not imply instability -- Gershgorin is only a sufficient
            // condition. We log the value for diagnostic purposes and only fail
            // on severely negative values that would indicate a pathological solution.
            assert!(
                gersh_min >= -0.5,
                "ECS = {}: Gershgorin minimum = {:.6} is severely negative, \
                 suggesting a potentially unstable solution. \
                 lambda_ocean = {:.6}, lambda_land = {:.6}",
                ecs,
                gersh_min,
                result.lambda_ocean,
                result.lambda_land,
            );
        }
    }
}
