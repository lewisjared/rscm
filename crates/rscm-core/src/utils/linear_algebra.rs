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

/// Invert a $4 \times 4$ matrix using Gauss-Jordan elimination with partial pivoting.
///
/// Returns `None` if the matrix is singular (i.e. a pivot element is smaller
/// than $10^{-15}$ in absolute value).
///
/// # Arguments
/// * `m` - The $4 \times 4$ matrix to invert, stored as row-major nested arrays.
///
/// # Returns
/// `Some(inverse)` if the matrix is invertible, `None` otherwise.
///
/// # Example
/// ```
/// use rscm_core::utils::linear_algebra::invert_4x4;
///
/// let identity = [
///     [1.0, 0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0, 1.0],
/// ];
/// let inv = invert_4x4(&identity).unwrap();
/// assert!((inv[0][0] - 1.0).abs() < 1e-15);
/// ```
pub fn invert_4x4(m: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    // Augmented matrix [A | I]
    let mut aug = [[0.0f64; 8]; 4];
    for i in 0..4 {
        for j in 0..4 {
            aug[i][j] = m[i][j];
        }
        aug[i][i + 4] = 1.0;
    }

    // Gauss-Jordan elimination with partial pivoting
    for col in 0..4 {
        // Find pivot row (largest absolute value in this column)
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..4 {
            let val = aug[row][col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return None;
        }

        // Swap rows
        if max_row != col {
            aug.swap(col, max_row);
        }

        // Scale pivot row
        let pivot = aug[col][col];
        for j in 0..8 {
            aug[col][j] /= pivot;
        }

        // Eliminate column in all other rows
        for row in 0..4 {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..8 {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Extract inverse from augmented matrix
    let mut inv = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            inv[i][j] = aug[i][j + 4];
        }
    }

    Some(inv)
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

    #[test]
    fn test_invert_4x4_identity() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let inv = invert_4x4(&identity).expect("Identity matrix should be invertible");

        println!("Inverse of identity:");
        for row in &inv {
            println!("  {:?}", row);
        }

        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv[i][j] - expected).abs() < 1e-15,
                    "inv[{}][{}] = {} (expected {})",
                    i,
                    j,
                    inv[i][j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_invert_4x4_known_matrix() {
        // Coupling matrix similar to the LAMCALC four-box structure
        let m = [
            [2.0, -0.5, -0.3, 0.0],
            [-0.5, 1.5, 0.0, 0.0],
            [-0.3, 0.0, 2.0, -0.5],
            [0.0, 0.0, -0.5, 1.5],
        ];

        let inv = invert_4x4(&m).expect("Coupling matrix should be invertible");

        println!("Inverse of coupling matrix:");
        for row in &inv {
            println!("  {:?}", row);
        }

        // Verify A * A_inv = I
        let mut product = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    product[i][j] += m[i][k] * inv[k][j];
                }
            }
        }

        println!("Product A * A_inv:");
        for row in &product {
            println!("  {:?}", row);
        }

        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i][j] - expected).abs() < 1e-12,
                    "product[{}][{}] = {} (expected {})",
                    i,
                    j,
                    product[i][j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_invert_4x4_singular_returns_none() {
        // Matrix with a zero row is singular
        let singular = [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ];

        let result = invert_4x4(&singular);

        println!("Inversion of singular matrix returned: {:?}", result);
        assert!(
            result.is_none(),
            "Singular matrix should return None, got {:?}",
            result
        );
    }
}
