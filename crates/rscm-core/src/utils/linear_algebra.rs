//! Linear algebra utilities.

/// Solve tridiagonal system Ax = d using Thomas algorithm.
///
/// The matrix A has the form:
/// ```text
/// | b[0]  c[0]   0     0    ...   0   |
/// | a[1]  b[1]  c[1]   0    ...   0   |
/// |  0    a[2]  b[2]  c[2]  ...   0   |
/// | ...   ...   ...   ...   ...  ... |
/// |  0     0     0   a[n-1] b[n-1]   |
/// ```
///
/// # Arguments
/// * `a` - Sub-diagonal coefficients (length n, a[0] is unused)
/// * `b` - Main diagonal coefficients (length n)
/// * `c` - Super-diagonal coefficients (length n, c[n-1] is unused)
/// * `d` - Right-hand side vector (length n)
///
/// # Returns
/// Solution vector x (length n)
///
/// # Panics
/// Panics if array lengths are inconsistent or if a zero pivot is encountered.
///
/// # Example
/// ```
/// use rscm_core::utils::linear_algebra::thomas_solve;
///
/// // Solve simple 3x3 system
/// let a = vec![0.0, -1.0, -1.0];
/// let b = vec![2.0, 2.0, 2.0];
/// let c = vec![-1.0, -1.0, 0.0];
/// let d = vec![1.0, 0.0, 1.0];
///
/// let x = thomas_solve(&a, &b, &c, &d);
/// assert!((x[0] - 1.0).abs() < 1e-10);
/// assert!((x[1] - 1.0).abs() < 1e-10);
/// assert!((x[2] - 1.0).abs() < 1e-10);
/// ```
pub fn thomas_solve(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = b.len();
    assert_eq!(a.len(), n, "a must have same length as b");
    assert_eq!(c.len(), n, "c must have same length as b");
    assert_eq!(d.len(), n, "d must have same length as b");
    assert!(n > 0, "System must have at least one equation");

    // Modified coefficients
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    // Forward sweep
    assert!(b[0].abs() > 1e-15, "Zero pivot encountered at row 0");
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let denom = b[i] - a[i] * c_prime[i - 1];
        assert!(denom.abs() > 1e-15, "Zero pivot encountered at row {}", i);

        if i < n - 1 {
            c_prime[i] = c[i] / denom;
        }
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];

    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    x
}

/// In-place version of Thomas algorithm (avoids allocations).
///
/// Modifies `d` in-place to contain the solution.
///
/// # Arguments
/// * `a` - Sub-diagonal coefficients (length n)
/// * `b` - Main diagonal coefficients (length n) - modified in-place
/// * `c` - Super-diagonal coefficients (length n) - modified in-place
/// * `d` - Right-hand side (length n) - replaced with solution
pub fn thomas_solve_inplace(a: &[f64], b: &mut [f64], c: &mut [f64], d: &mut [f64]) {
    let n = b.len();
    assert_eq!(a.len(), n, "a must have same length as b");
    assert_eq!(c.len(), n, "c must have same length as b");
    assert_eq!(d.len(), n, "d must have same length as b");
    assert!(n > 0, "System must have at least one equation");

    // Forward sweep
    assert!(b[0].abs() > 1e-15, "Zero pivot encountered at row 0");
    c[0] /= b[0];
    d[0] /= b[0];

    for i in 1..n {
        let denom = b[i] - a[i] * c[i - 1];
        assert!(denom.abs() > 1e-15, "Zero pivot encountered at row {}", i);
        if i < n - 1 {
            c[i] /= denom;
        }
        d[i] = (d[i] - a[i] * d[i - 1]) / denom;
    }

    // Back substitution (d now contains x)
    for i in (0..n - 1).rev() {
        d[i] -= c[i] * d[i + 1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thomas_identity() {
        // Diagonal matrix (identity-like) - trivial case
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0];
        let c = vec![0.0, 0.0, 0.0];
        let d = vec![1.0, 2.0, 3.0];

        let x = thomas_solve(&a, &b, &c, &d);
        assert_eq!(x, d);
    }

    #[test]
    fn test_thomas_3x3_known_solution() {
        // System from the doctest: solution should be [1, 1, 1]
        // Matrix:
        // | 2 -1  0 |   | 1 |
        // |-1  2 -1 | = | 0 |
        // | 0 -1  2 |   | 1 |
        let a = vec![0.0, -1.0, -1.0];
        let b = vec![2.0, 2.0, 2.0];
        let c = vec![-1.0, -1.0, 0.0];
        let d = vec![1.0, 0.0, 1.0];

        let x = thomas_solve(&a, &b, &c, &d);

        println!("Solution: {:?}", x);
        for (i, &xi) in x.iter().enumerate() {
            assert!((xi - 1.0).abs() < 1e-10, "x[{}] = {} (expected 1.0)", i, xi);
        }
    }

    #[test]
    fn test_thomas_larger_system() {
        // 50-element diffusion-like system: -u[i-1] + 2u[i] - u[i+1] = h^2 * f[i]
        // with boundary conditions u[0] = u[n-1] = 0
        // This models heat diffusion on a 1D grid
        let n = 50;
        let h = 1.0 / (n as f64 + 1.0);
        let h2 = h * h;

        // Set up tridiagonal system
        let a: Vec<f64> = (0..n).map(|i| if i == 0 { 0.0 } else { -1.0 }).collect();
        let b: Vec<f64> = vec![2.0; n];
        let c: Vec<f64> = (0..n)
            .map(|i| if i == n - 1 { 0.0 } else { -1.0 })
            .collect();

        // RHS: constant forcing f(x) = 1, so d[i] = h^2
        let d: Vec<f64> = vec![h2; n];

        let x = thomas_solve(&a, &b, &c, &d);

        // Verify solution satisfies the system
        // For interior points: -x[i-1] + 2*x[i] - x[i+1] should equal h^2
        for i in 1..n - 1 {
            let residual = -x[i - 1] + 2.0 * x[i] - x[i + 1];
            assert!(
                (residual - h2).abs() < 1e-10,
                "Residual at {} is {} (expected {})",
                i,
                residual,
                h2
            );
        }

        // Check boundary consistency
        // At i=0: 2*x[0] - x[1] = h^2 (since a[0] is unused)
        let residual_0 = 2.0 * x[0] - x[1];
        assert!(
            (residual_0 - h2).abs() < 1e-10,
            "Residual at 0: {} (expected {})",
            residual_0,
            h2
        );

        // At i=n-1: -x[n-2] + 2*x[n-1] = h^2 (since c[n-1] is unused)
        let residual_n = -x[n - 2] + 2.0 * x[n - 1];
        assert!(
            (residual_n - h2).abs() < 1e-10,
            "Residual at n-1: {} (expected {})",
            residual_n,
            h2
        );

        println!(
            "Solved 50-element system. Max value: {:.6}",
            x.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );
    }

    #[test]
    fn test_thomas_inplace_matches_allocating() {
        let a = vec![0.0, -1.0, -1.0, -1.0];
        let b_orig = vec![2.0, 3.0, 3.0, 2.0];
        let c_orig = vec![-1.0, -1.0, -1.0, 0.0];
        let d_orig = vec![1.0, 2.0, 2.0, 1.0];

        // Allocating version
        let x_alloc = thomas_solve(&a, &b_orig, &c_orig, &d_orig);

        // In-place version (need mutable copies)
        let mut b = b_orig.clone();
        let mut c = c_orig.clone();
        let mut d = d_orig.clone();
        thomas_solve_inplace(&a, &mut b, &mut c, &mut d);

        println!("Allocating solution: {:?}", x_alloc);
        println!("In-place solution: {:?}", d);

        for i in 0..x_alloc.len() {
            assert!(
                (x_alloc[i] - d[i]).abs() < 1e-10,
                "Mismatch at {}: alloc={}, inplace={}",
                i,
                x_alloc[i],
                d[i]
            );
        }
    }

    #[test]
    fn test_thomas_single_equation() {
        // n=1 edge case: b[0] * x[0] = d[0]
        let a = vec![0.0];
        let b = vec![3.0];
        let c = vec![0.0];
        let d = vec![6.0];

        let x = thomas_solve(&a, &b, &c, &d);

        assert_eq!(x.len(), 1);
        assert!((x[0] - 2.0).abs() < 1e-10, "x[0] = {} (expected 2.0)", x[0]);
    }

    #[test]
    fn test_thomas_two_equations() {
        // n=2 system:
        // | 4  1 |   | x0 |   | 1 |
        // | 1  3 | * | x1 | = | 2 |
        // Solution: x0 = 1/11, x1 = 7/11
        let a = vec![0.0, 1.0];
        let b = vec![4.0, 3.0];
        let c = vec![1.0, 0.0];
        let d = vec![1.0, 2.0];

        let x = thomas_solve(&a, &b, &c, &d);

        let expected_x0 = 1.0 / 11.0;
        let expected_x1 = 7.0 / 11.0;

        assert!(
            (x[0] - expected_x0).abs() < 1e-10,
            "x[0] = {} (expected {})",
            x[0],
            expected_x0
        );
        assert!(
            (x[1] - expected_x1).abs() < 1e-10,
            "x[1] = {} (expected {})",
            x[1],
            expected_x1
        );
    }

    #[test]
    #[should_panic(expected = "Zero pivot")]
    fn test_thomas_zero_pivot_panics() {
        // Create a system that will have a zero pivot
        let a = vec![0.0, 1.0];
        let b = vec![1.0, 1.0]; // After elimination: b[1] - a[1]*c[0] = 1 - 1*1 = 0
        let c = vec![1.0, 0.0];
        let d = vec![1.0, 1.0];

        let _ = thomas_solve(&a, &b, &c, &d);
    }

    #[test]
    #[should_panic(expected = "a must have same length as b")]
    fn test_thomas_length_mismatch_panics() {
        let a = vec![0.0, 1.0]; // length 2
        let b = vec![1.0, 2.0, 3.0]; // length 3
        let c = vec![1.0, 1.0, 0.0];
        let d = vec![1.0, 1.0, 1.0];

        let _ = thomas_solve(&a, &b, &c, &d);
    }
}
