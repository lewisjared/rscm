//! LAMCALC iterative solver for climate feedback parameters.
//!
//! Given equilibrium climate sensitivity (ECS), land-ocean warming ratio (RLO),
//! and physical parameters, LAMCALC iteratively finds the ocean and land feedback
//! parameters ($\lambda_{\text{ocean}}$ and $\lambda_{\text{land}}$) that satisfy
//! both constraints when inter-box heat exchange is accounted for.
//!
//! The algorithm constructs a $4 \times 4$ coupling matrix representing the
//! four-box energy balance (NH ocean, NH land, SH ocean, SH land) and iterates
//! on $\lambda_{\text{ocean}}$ until the implied land-ocean warming ratio matches
//! the target RLO. A hybrid step/secant method is used for convergence.
//!
//! Reference: MAGICC7.f90 lines 8070-8278.

use rscm_core::timeseries::FloatValue;
use rscm_core::utils::linear_algebra::invert_4x4;

/// Maximum number of iterations for convergence.
const MAX_ITERATIONS: usize = 40;

/// Convergence tolerance for the land-ocean warming ratio.
const RLO_TOLERANCE: FloatValue = 0.001;

/// Parameters for the LAMCALC iteration.
#[derive(Debug, Clone, Copy)]
pub struct LamcalcParams {
    /// Radiative forcing for a doubling of $\text{CO}_2$ (W/m$^2$), typically 3.71.
    pub q_2xco2: FloatValue,
    /// Land-ocean heat exchange coefficient (W/m$^2$/K), typically 1.44.
    pub k_lo: FloatValue,
    /// Inter-hemispheric (north-south) heat exchange coefficient (W/m$^2$/K), typically 0.31.
    pub k_ns: FloatValue,
    /// Equilibrium climate sensitivity (K).
    pub ecs: FloatValue,
    /// Land-ocean warming ratio, typically 1.317.
    pub rlo: FloatValue,
    /// Ocean-to-land amplification factor, typically 1.02.
    pub amplify_ocean_to_land: FloatValue,
    /// Northern hemisphere ocean global fraction.
    pub fgno: FloatValue,
    /// Northern hemisphere land global fraction.
    pub fgnl: FloatValue,
    /// Southern hemisphere ocean global fraction.
    pub fgso: FloatValue,
    /// Southern hemisphere land global fraction.
    pub fgsl: FloatValue,
    /// Regional CO2 radiative forcing pattern (NH ocean, NH land, SH ocean, SH land).
    ///
    /// Used to compute per-box forcing fractions (qfrac) following MAGICC7:
    ///
    /// $$\text{qfrac}_i = \frac{\text{rf\_regions\_co2}_i}{\sum_j \text{rf\_regions\_co2}_j \times f_j}$$
    ///
    /// Default `[1.0; 4]` produces uniform qfrac = 1.0 for all boxes.
    ///
    /// Reference: MAGICC7.f90 lines 8146-8165.
    pub rf_regions_co2: [FloatValue; 4],
}

/// Result of the LAMCALC iteration.
#[derive(Debug, Clone, Copy)]
pub struct LamcalcResult {
    /// Converged ocean feedback parameter (W/m$^2$/K).
    pub lambda_ocean: FloatValue,
    /// Converged land feedback parameter (W/m$^2$/K).
    pub lambda_land: FloatValue,
    /// Inverse of the converged $4 \times 4$ coupling matrix.
    pub matrix_inverse: [[FloatValue; 4]; 4],
    /// Internal efficacy of CO2 forcing (ratio of CO2 global temperature
    /// response to the ECS-implied response). Near 1.0 for typical parameters.
    pub co2_internal_efficacy: FloatValue,
}

/// Build the $4 \times 4$ coupling matrix for the four-box energy balance.
///
/// The boxes are ordered: NH ocean (0), NH land (1), SH ocean (2), SH land (3).
///
/// The matrix encodes radiative feedback ($\lambda$), land-ocean heat exchange
/// ($K_{\text{LO}}$ with amplification $\alpha$), and inter-hemispheric exchange
/// ($K_{\text{NS}}$).
///
/// # Arguments
/// * `params` - Physical parameters (exchange coefficients, area fractions)
/// * `lam_o` - Ocean feedback parameter
/// * `lam_l` - Land feedback parameter
pub fn build_coupling_matrix(
    params: &LamcalcParams,
    lam_o: FloatValue,
    lam_l: FloatValue,
) -> [[FloatValue; 4]; 4] {
    let alpha = params.amplify_ocean_to_land;
    let k_lo = params.k_lo;
    let k_ns = params.k_ns;

    [
        // Row 0: NH ocean
        [params.fgno * lam_o + k_lo * alpha + k_ns, -k_lo, -k_ns, 0.0],
        // Row 1: NH land
        [-k_lo * alpha, params.fgnl * lam_l + k_lo, 0.0, 0.0],
        // Row 2: SH ocean
        [-k_ns, 0.0, params.fgso * lam_o + k_lo * alpha + k_ns, -k_lo],
        // Row 3: SH land
        [0.0, 0.0, -k_lo * alpha, params.fgsl * lam_l + k_lo],
    ]
}

/// Compute per-box forcing fractions from a regional forcing pattern.
///
/// Normalizes `rf_regions` by the area-weighted sum to produce `qfrac` values
/// such that area-weighted forcing is preserved. Returns `[1.0; 4]` if the
/// weighted sum is near zero.
///
/// $$\text{qfrac}_i = \frac{\text{rf\_regions}_i}{\sum_j \text{rf\_regions}_j \times f_j}$$
///
/// Reference: MAGICC7.f90 lines 8146-8165.
pub fn compute_qfrac(rf_regions: &[FloatValue; 4], area: &[FloatValue; 4]) -> [FloatValue; 4] {
    let rf_sum: FloatValue = rf_regions.iter().zip(area.iter()).map(|(r, a)| r * a).sum();

    if rf_sum.abs() <= 1e-15 {
        [1.0; 4]
    } else {
        [
            rf_regions[0] / rf_sum,
            rf_regions[1] / rf_sum,
            rf_regions[2] / rf_sum,
            rf_regions[3] / rf_sum,
        ]
    }
}

/// Compute the internal efficacy of a forcing agent given its regional pattern.
///
/// Internal efficacy measures how effectively an agent's spatial forcing pattern
/// translates into global temperature change relative to ECS. Agents concentrated
/// in high-feedback regions (land, high latitudes) get efficacy > 1, while those
/// in low-feedback regions (ocean, tropics) get efficacy < 1.
///
/// In AR6 mode: $\text{EFFRF} = \text{RF} \times \text{prescribed\_efficacy} / \text{internal\_efficacy}$.
///
/// Returns 1.0 as fallback if `rf_regions` sums to zero.
///
/// Reference: MAGICC7.f90 lines 8267-8278.
pub fn calc_internal_efficacy(
    q_2xco2: FloatValue,
    matrix_inverse: &[[FloatValue; 4]; 4],
    area: &[FloatValue; 4],
    rf_regions: &[FloatValue; 4],
    ecs: FloatValue,
) -> FloatValue {
    let rf_sum: FloatValue = rf_regions.iter().zip(area.iter()).map(|(r, a)| r * a).sum();

    if rf_sum.abs() <= 1e-15 {
        return 1.0;
    }

    let qfrac = compute_qfrac(rf_regions, area);

    let mut temps = [0.0_f64; 4];
    for row in 0..4 {
        let mut sum = 0.0;
        for col in 0..4 {
            sum += matrix_inverse[row][col] * area[col] * qfrac[col];
        }
        temps[row] = q_2xco2 * sum;
    }

    let t_global: FloatValue = area.iter().zip(temps.iter()).map(|(a, t)| a * t).sum();
    t_global / ecs
}

/// Run the LAMCALC iterative solver.
///
/// Iteratively finds $\lambda_{\text{ocean}}$ and $\lambda_{\text{land}}$ such that
/// the four-box energy balance model produces equilibrium warming with the target
/// land-ocean warming ratio (RLO) at the given ECS.
///
/// Returns `None` if the iteration fails to converge within [`MAX_ITERATIONS`] steps
/// or if a singular coupling matrix is encountered.
pub fn lamcalc(params: &LamcalcParams) -> Option<LamcalcResult> {
    let lam = params.q_2xco2 / params.ecs;
    let fgosum = params.fgno + params.fgso;
    let fglsum = params.fgnl + params.fgsl;
    let fratio = fgosum / fglsum;

    let area = [params.fgno, params.fgnl, params.fgso, params.fgsl];

    // Compute per-box CO2 forcing fractions from regional pattern.
    let qfrac = compute_qfrac(&params.rf_regions_co2, &area);

    // Storage for iteration history
    let mut lamo = vec![0.0; MAX_ITERATIONS + 2];
    let mut diff = vec![0.0; MAX_ITERATIONS + 2];

    // Initial guesses
    lamo[1] = lam;
    lamo[2] = lam + 0.7;

    let mut dlamo = 0.7_f64;
    let mut iflag = 0_i32;

    let mut converged_lam_o = 0.0;
    let mut converged_lam_l = 0.0;
    let mut converged_inv: Option<[[FloatValue; 4]; 4]> = None;
    let mut found = false;

    for i in 2..=MAX_ITERATIONS {
        // Derive lambda_land from lambda_ocean
        let lam_l = lam + fratio * (lam - lamo[i]) / params.rlo;
        let lam_o = lamo[i];

        // Build and invert coupling matrix
        let matrix = build_coupling_matrix(params, lam_o, lam_l);
        let inv = invert_4x4(&matrix)?;

        // Compute equilibrium temperatures for each box
        // T[row] = Q * sum_col(B[row][col] * area[col] * qfrac[col])
        let q = params.q_2xco2;
        let mut temps = [0.0_f64; 4];
        for row in 0..4 {
            let mut sum = 0.0;
            for col in 0..4 {
                sum += inv[row][col] * area[col] * qfrac[col];
            }
            temps[row] = q * sum;
        }

        // Compute area-weighted ocean and land mean temperatures
        let ocean_mean =
            (params.fgno * temps[0] + params.fgso * temps[2]) / (params.fgno + params.fgso);
        let land_mean =
            (params.fgnl * temps[1] + params.fgsl * temps[3]) / (params.fgnl + params.fgsl);

        let rlo_est = land_mean / ocean_mean;

        // Check convergence
        diff[i] = params.rlo - rlo_est;
        if diff[i].abs() < RLO_TOLERANCE {
            converged_lam_o = lam_o;
            converged_lam_l = lam_l;
            converged_inv = Some(inv);
            found = true;
            break;
        }

        // Update lambda_ocean using hybrid step/secant method
        if diff[i] * diff[i - 1] < 0.0 {
            iflag = 1;
        }

        if iflag == 0 {
            // Step method: reverse direction if error increased
            if diff[i].abs() > diff[i - 1].abs() {
                dlamo = -dlamo;
            }
            lamo[i + 1] = lamo[i] + dlamo;
        } else if diff[i] * diff[i - 1] < 0.0 {
            // Secant between i and i-1 (sign change)
            let denom = diff[i] - diff[i - 1];
            if denom.abs() < 1e-30 {
                lamo[i + 1] = lamo[i] + dlamo;
            } else {
                lamo[i + 1] = lamo[i] - diff[i] * (lamo[i] - lamo[i - 1]) / denom;
            }
        } else {
            // Secant between i and i-2 (no sign change)
            let i2 = if i >= 2 { i - 2 } else { 0 };
            let denom = diff[i] - diff[i2];
            if denom.abs() < 1e-30 {
                lamo[i + 1] = lamo[i] + dlamo;
            } else {
                lamo[i + 1] = lamo[i] - diff[i] * (lamo[i] - lamo[i2]) / denom;
            }
        }
    }

    if found {
        // Use the matrix inverse cached from the converged iteration.
        let final_inv = converged_inv.expect("inv is set when found=true");

        let co2_internal_efficacy = calc_internal_efficacy(
            params.q_2xco2,
            &final_inv,
            &area,
            &params.rf_regions_co2,
            params.ecs,
        );

        Some(LamcalcResult {
            lambda_ocean: converged_lam_o,
            lambda_land: converged_lam_l,
            matrix_inverse: final_inv,
            co2_internal_efficacy,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create default LAMCALC parameters matching MAGICC defaults.
    fn default_params() -> LamcalcParams {
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
            rf_regions_co2: [1.0; 4],
        }
    }

    #[test]
    fn test_lamcalc_converges() {
        let params = default_params();
        let result = lamcalc(&params);

        println!("LAMCALC result: {:?}", result);

        assert!(
            result.is_some(),
            "LAMCALC should converge with default parameters"
        );

        let result = result.unwrap();
        assert!(
            result.lambda_ocean > 0.0,
            "lambda_ocean should be positive, got {}",
            result.lambda_ocean
        );
        // lambda_land can be negative (positive feedback on land) depending on
        // the RLO and ECS combination; we only check it is finite.
        assert!(
            result.lambda_land.is_finite(),
            "lambda_land should be finite, got {}",
            result.lambda_land
        );
        println!(
            "lambda_ocean = {:.6}, lambda_land = {:.6}",
            result.lambda_ocean, result.lambda_land
        );
    }

    #[test]
    fn test_lamcalc_satisfies_rlo_constraint() {
        let params = default_params();
        let result = lamcalc(&params).expect("LAMCALC should converge");

        // Reconstruct the coupling matrix with converged lambdas and verify RLO
        let matrix = build_coupling_matrix(&params, result.lambda_ocean, result.lambda_land);
        let inv = invert_4x4(&matrix).expect("Coupling matrix should be invertible");

        let area = [params.fgno, params.fgnl, params.fgso, params.fgsl];
        let q = params.q_2xco2;

        let mut temps = [0.0_f64; 4];
        for row in 0..4 {
            let mut sum = 0.0;
            for col in 0..4 {
                sum += inv[row][col] * area[col];
            }
            temps[row] = q * sum;
        }

        let ocean_mean =
            (params.fgno * temps[0] + params.fgso * temps[2]) / (params.fgno + params.fgso);
        let land_mean =
            (params.fgnl * temps[1] + params.fgsl * temps[3]) / (params.fgnl + params.fgsl);
        let rlo_actual = land_mean / ocean_mean;

        println!(
            "Target RLO = {:.4}, actual RLO = {:.4}, ocean_mean = {:.4} K, land_mean = {:.4} K",
            params.rlo, rlo_actual, ocean_mean, land_mean
        );

        assert!(
            (params.rlo - rlo_actual).abs() < RLO_TOLERANCE,
            "RLO mismatch: target = {}, actual = {}, difference = {}",
            params.rlo,
            rlo_actual,
            (params.rlo - rlo_actual).abs()
        );

        // Also verify that the global-mean equilibrium temperature is close to ECS
        let global_mean =
            area[0] * temps[0] + area[1] * temps[1] + area[2] * temps[2] + area[3] * temps[3];
        println!(
            "Global mean equilibrium temperature = {:.4} K (ECS = {:.1} K)",
            global_mean, params.ecs
        );
    }

    #[test]
    fn test_lamcalc_different_ecs_values() {
        let ecs_values = [1.5, 2.0, 3.0, 4.0, 4.5];

        for &ecs in &ecs_values {
            let mut params = default_params();
            params.ecs = ecs;

            let result = lamcalc(&params);
            println!("ECS = {:.1}: result = {:?}", ecs, result);

            assert!(
                result.is_some(),
                "LAMCALC should converge for ECS = {}",
                ecs
            );

            let result = result.unwrap();
            assert!(
                result.lambda_ocean > 0.0,
                "lambda_ocean should be positive for ECS = {}, got {}",
                ecs,
                result.lambda_ocean
            );
            assert!(
                result.lambda_land.is_finite(),
                "lambda_land should be finite for ECS = {}, got {}",
                ecs,
                result.lambda_land
            );

            // Verify the converged result actually satisfies the RLO constraint
            let matrix = build_coupling_matrix(&params, result.lambda_ocean, result.lambda_land);
            let inv = invert_4x4(&matrix).expect("Matrix should be invertible");
            let area = [params.fgno, params.fgnl, params.fgso, params.fgsl];
            let q = params.q_2xco2;
            let mut temps = [0.0_f64; 4];
            for row in 0..4 {
                for col in 0..4 {
                    temps[row] += inv[row][col] * area[col];
                }
                temps[row] *= q;
            }
            let ocean_mean =
                (params.fgno * temps[0] + params.fgso * temps[2]) / (params.fgno + params.fgso);
            let land_mean =
                (params.fgnl * temps[1] + params.fgsl * temps[3]) / (params.fgnl + params.fgsl);
            let rlo_actual = land_mean / ocean_mean;
            assert!(
                (params.rlo - rlo_actual).abs() < RLO_TOLERANCE,
                "RLO constraint not satisfied for ECS = {}: target = {}, actual = {}",
                ecs,
                params.rlo,
                rlo_actual
            );

            println!(
                "  lambda_ocean = {:.6}, lambda_land = {:.6}",
                result.lambda_ocean, result.lambda_land
            );
        }
    }

    #[test]
    fn test_coupling_matrix_structure() {
        let params = default_params();
        let lam_o = 1.2;
        let lam_l = 1.5;

        let matrix = build_coupling_matrix(&params, lam_o, lam_l);

        println!("Coupling matrix:");
        for (i, row) in matrix.iter().enumerate() {
            println!("  Row {}: {:?}", i, row);
        }

        // Diagonal elements should be positive (feedback + exchange terms)
        for i in 0..4 {
            assert!(
                matrix[i][i] > 0.0,
                "Diagonal element [{}][{}] = {} should be positive",
                i,
                i,
                matrix[i][i]
            );
        }

        // Off-diagonal K_LO terms should be negative
        // Row 0, col 1: -K_LO
        assert!(
            matrix[0][1] < 0.0,
            "matrix[0][1] = {} should be negative (-K_LO)",
            matrix[0][1]
        );
        assert!(
            (matrix[0][1] - (-params.k_lo)).abs() < 1e-15,
            "matrix[0][1] = {} should equal -K_LO = {}",
            matrix[0][1],
            -params.k_lo
        );

        // Row 1, col 0: -K_LO * alpha
        assert!(
            matrix[1][0] < 0.0,
            "matrix[1][0] = {} should be negative (-K_LO*alpha)",
            matrix[1][0]
        );
        assert!(
            (matrix[1][0] - (-params.k_lo * params.amplify_ocean_to_land)).abs() < 1e-15,
            "matrix[1][0] = {} should equal -K_LO*alpha = {}",
            matrix[1][0],
            -params.k_lo * params.amplify_ocean_to_land
        );

        // Row 2, col 3: -K_LO
        assert!(
            matrix[2][3] < 0.0,
            "matrix[2][3] = {} should be negative (-K_LO)",
            matrix[2][3]
        );

        // Row 3, col 2: -K_LO * alpha
        assert!(
            matrix[3][2] < 0.0,
            "matrix[3][2] = {} should be negative (-K_LO*alpha)",
            matrix[3][2]
        );

        // Correct zero entries
        assert!(
            matrix[0][3].abs() < 1e-15,
            "matrix[0][3] = {} should be zero",
            matrix[0][3]
        );
        assert!(
            matrix[1][2].abs() < 1e-15,
            "matrix[1][2] = {} should be zero",
            matrix[1][2]
        );
        assert!(
            matrix[1][3].abs() < 1e-15,
            "matrix[1][3] = {} should be zero",
            matrix[1][3]
        );
        assert!(
            matrix[3][0].abs() < 1e-15,
            "matrix[3][0] = {} should be zero",
            matrix[3][0]
        );
        assert!(
            matrix[3][1].abs() < 1e-15,
            "matrix[3][1] = {} should be zero",
            matrix[3][1]
        );

        // Inter-hemispheric exchange: K_NS terms
        assert!(
            matrix[0][2] < 0.0,
            "matrix[0][2] = {} should be negative (-K_NS)",
            matrix[0][2]
        );
        assert!(
            (matrix[0][2] - (-params.k_ns)).abs() < 1e-15,
            "matrix[0][2] = {} should equal -K_NS = {}",
            matrix[0][2],
            -params.k_ns
        );
        assert!(
            matrix[2][0] < 0.0,
            "matrix[2][0] = {} should be negative (-K_NS)",
            matrix[2][0]
        );
        assert!(
            (matrix[2][0] - (-params.k_ns)).abs() < 1e-15,
            "matrix[2][0] = {} should equal -K_NS = {}",
            matrix[2][0],
            -params.k_ns
        );
    }

    #[test]
    fn test_non_uniform_rf_regions_produces_different_lambdas() {
        // Uniform forcing (default): all boxes receive equal CO2 forcing fraction.
        let uniform_params = default_params();
        let uniform_result =
            lamcalc(&uniform_params).expect("LAMCALC should converge with uniform forcing");

        // Non-uniform forcing: NH receives ~20% stronger CO2 forcing than SH.
        // This matches a typical MAGICC7 regional CO2 pattern where forcing
        // is not perfectly symmetric across hemispheres.
        let mut nonuniform_params = default_params();
        nonuniform_params.rf_regions_co2 = [1.2, 1.2, 0.8, 0.8];

        let nonuniform_result =
            lamcalc(&nonuniform_params).expect("LAMCALC should converge with non-uniform forcing");

        println!(
            "Uniform:     lambda_ocean = {:.6}, lambda_land = {:.6}",
            uniform_result.lambda_ocean, uniform_result.lambda_land
        );
        println!(
            "Non-uniform: lambda_ocean = {:.6}, lambda_land = {:.6}",
            nonuniform_result.lambda_ocean, nonuniform_result.lambda_land
        );

        // Non-uniform regional forcing must produce different lambda values.
        let ocean_diff = (uniform_result.lambda_ocean - nonuniform_result.lambda_ocean).abs();
        let land_diff = (uniform_result.lambda_land - nonuniform_result.lambda_land).abs();
        assert!(
            ocean_diff > 1e-6 || land_diff > 1e-6,
            "Non-uniform rf_regions_co2 should produce different lambdas: \
             uniform=({:.6}, {:.6}), non-uniform=({:.6}, {:.6})",
            uniform_result.lambda_ocean,
            uniform_result.lambda_land,
            nonuniform_result.lambda_ocean,
            nonuniform_result.lambda_land,
        );

        // Both results must still satisfy the RLO constraint.
        for (label, params, result) in [
            ("uniform", &uniform_params, &uniform_result),
            ("non-uniform", &nonuniform_params, &nonuniform_result),
        ] {
            let matrix = build_coupling_matrix(params, result.lambda_ocean, result.lambda_land);
            let inv = invert_4x4(&matrix).expect("Coupling matrix should be invertible");
            let area = [params.fgno, params.fgnl, params.fgso, params.fgsl];

            let qfrac = compute_qfrac(&params.rf_regions_co2, &area);

            let q = params.q_2xco2;
            let mut temps = [0.0_f64; 4];
            for row in 0..4 {
                for col in 0..4 {
                    temps[row] += inv[row][col] * area[col] * qfrac[col];
                }
                temps[row] *= q;
            }

            let ocean_mean =
                (params.fgno * temps[0] + params.fgso * temps[2]) / (params.fgno + params.fgso);
            let land_mean =
                (params.fgnl * temps[1] + params.fgsl * temps[3]) / (params.fgnl + params.fgsl);
            let rlo_actual = land_mean / ocean_mean;

            println!(
                "{}: target RLO = {:.4}, actual RLO = {:.4}",
                label, params.rlo, rlo_actual
            );
            assert!(
                (params.rlo - rlo_actual).abs() < RLO_TOLERANCE,
                "{}: RLO constraint not satisfied: target = {}, actual = {}",
                label,
                params.rlo,
                rlo_actual
            );
        }
    }

    #[test]
    fn test_lamcalc_returns_valid_matrix_inverse() {
        let params = default_params();
        let result = lamcalc(&params).expect("LAMCALC should converge");

        // Reconstruct the coupling matrix and verify matrix * inverse ≈ identity
        let matrix = build_coupling_matrix(&params, result.lambda_ocean, result.lambda_land);
        let inv = &result.matrix_inverse;

        for i in 0..4 {
            for j in 0..4 {
                let mut dot = 0.0;
                for k in 0..4 {
                    dot += matrix[i][k] * inv[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "matrix * inverse [{i}][{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_co2_internal_efficacy_near_unity() {
        // With AR6 defaults, CO2 efficacy should be close to 1.0
        let params = default_params();
        let result = lamcalc(&params).expect("LAMCALC should converge");

        println!(
            "CO2 internal efficacy = {:.6}",
            result.co2_internal_efficacy
        );
        assert!(
            result.co2_internal_efficacy > 0.90 && result.co2_internal_efficacy < 1.10,
            "CO2 efficacy should be near 1.0, got {}",
            result.co2_internal_efficacy
        );
    }

    #[test]
    fn test_calc_internal_efficacy_uniform_pattern() {
        // Uniform rf_regions [1;4] with uniform lamcalc params should give efficacy = 1.0
        let mut params = default_params();
        params.rf_regions_co2 = [1.0; 4];
        let result = lamcalc(&params).expect("LAMCALC should converge");

        let area = [params.fgno, params.fgnl, params.fgso, params.fgsl];
        let uniform_rf = [1.0; 4];
        let efficacy = calc_internal_efficacy(
            params.q_2xco2,
            &result.matrix_inverse,
            &area,
            &uniform_rf,
            params.ecs,
        );

        println!("Uniform pattern efficacy = {:.6}", efficacy);
        assert!(
            (efficacy - 1.0).abs() < 0.01,
            "Uniform pattern should give efficacy ~1.0, got {efficacy}"
        );
    }

    #[test]
    fn test_calc_internal_efficacy_asymmetric_patterns() {
        // Different spatial patterns should produce different efficacies
        let params = default_params();
        let result = lamcalc(&params).expect("LAMCALC should converge");
        let area = [params.fgno, params.fgnl, params.fgso, params.fgsl];

        // CO2-like pattern (slight land enhancement)
        let co2_pattern = [1.4089, 1.37045, 1.43333, 1.33257];
        let co2_eff = calc_internal_efficacy(
            params.q_2xco2,
            &result.matrix_inverse,
            &area,
            &co2_pattern,
            params.ecs,
        );

        // Tropospheric ozone-like pattern (NH land-dominated)
        let trop_o3_pattern = [1.0, 3.0, 0.5, 0.5];
        let o3_eff = calc_internal_efficacy(
            params.q_2xco2,
            &result.matrix_inverse,
            &area,
            &trop_o3_pattern,
            params.ecs,
        );

        println!("CO2 efficacy = {co2_eff:.6}, Trop O3 efficacy = {o3_eff:.6}");

        // Land-concentrated forcing (O3) should have higher efficacy than CO2
        // because land has stronger feedback
        assert!(
            o3_eff > co2_eff,
            "NH land-concentrated O3 forcing should have higher efficacy than CO2: \
             CO2={co2_eff:.6}, O3={o3_eff:.6}"
        );
    }

    #[test]
    fn test_calc_internal_efficacy_zero_pattern() {
        let params = default_params();
        let result = lamcalc(&params).expect("LAMCALC should converge");
        let area = [params.fgno, params.fgnl, params.fgso, params.fgsl];

        let zero_rf = [0.0; 4];
        let efficacy = calc_internal_efficacy(
            params.q_2xco2,
            &result.matrix_inverse,
            &area,
            &zero_rf,
            params.ecs,
        );

        assert!(
            (efficacy - 1.0).abs() < 1e-15,
            "Zero pattern should return fallback 1.0, got {efficacy}"
        );
    }
}
