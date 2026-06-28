//! Lawson-Hanson NNLS (Non-Negative Least Squares).
//!
//! Solves min ||A x - y||² subject to x >= 0.
//! Optionally with Tikhonov regularisation: min ||A x - y||² + λ²||L x||²
//! by augmenting [A; λL] and [y; 0] before the solve.

use nalgebra::{DMatrix, DVector};

/// Tolerance for the KKT condition (gradient is non-positive on active set).
const TOL: f64 = 1e-8;

/// Lawson-Hanson NNLS. Returns the non-negative least-squares solution.
pub fn nnls(a: &DMatrix<f64>, y: &DVector<f64>, max_iter: usize) -> DVector<f64> {
    let n = a.ncols();
    let mut x = DVector::<f64>::zeros(n);
    // active[j] = true means j is in the "passive" (free) set.
    let mut passive = vec![false; n];

    for _outer in 0..max_iter {
        // Compute gradient w = A^T (y - A x).
        let r = y - a * &x;
        let w = a.transpose() * &r;

        // Find best active (zero) coordinate to add.
        let mut best_j = None;
        let mut best_w = TOL;
        for j in 0..n {
            if !passive[j] && w[j] > best_w {
                best_w = w[j];
                best_j = Some(j);
            }
        }
        let j = match best_j {
            Some(j) => j,
            None => return x, // KKT satisfied
        };
        passive[j] = true;

        // Inner loop: unconstrained LS on the passive set.
        for _inner in 0..3 * n {
            let p_idx: Vec<usize> = (0..n).filter(|i| passive[*i]).collect();
            if p_idx.is_empty() {
                break;
            }
            let a_p = a.select_columns(p_idx.iter());
            // Solve (a_p^T a_p) s_p = a_p^T y
            let atb = a_p.transpose() * y;
            let ata = a_p.transpose() * &a_p;
            let s_p = match ata.lu().solve(&atb) {
                Some(s) => s,
                None => return x,
            };

            // If all s_p > 0, accept and break inner loop.
            if s_p.iter().all(|&v| v > 0.0) {
                let mut s_full = DVector::<f64>::zeros(n);
                for (k, &idx) in p_idx.iter().enumerate() {
                    s_full[idx] = s_p[k];
                }
                x = s_full;
                break;
            }

            // Otherwise compute alpha and shrink toward boundary.
            let mut alpha = f64::INFINITY;
            for (k, &idx) in p_idx.iter().enumerate() {
                if s_p[k] <= 0.0 {
                    let denom = x[idx] - s_p[k];
                    if denom > 1e-20 {
                        let a_candidate = x[idx] / denom;
                        if a_candidate < alpha {
                            alpha = a_candidate;
                        }
                    }
                }
            }
            if !alpha.is_finite() {
                return x;
            }

            // x = x + alpha * (s_full - x)
            for (k, &idx) in p_idx.iter().enumerate() {
                x[idx] = x[idx] + alpha * (s_p[k] - x[idx]);
            }
            // Move any (numerically) zero coordinates back to active.
            for (k, &idx) in p_idx.iter().enumerate() {
                if x[idx] < 1e-12 || s_p[k] <= 0.0 && x[idx] < 1e-9 {
                    x[idx] = 0.0;
                    passive[idx] = false;
                }
            }
        }
    }
    x
}

/// Build a finite-difference smoothness regulariser L (second derivative).
/// Shape: (n-2, n).
pub fn second_difference_matrix(n: usize) -> DMatrix<f64> {
    let mut l = DMatrix::<f64>::zeros(n.saturating_sub(2), n);
    for i in 0..n.saturating_sub(2) {
        l[(i, i)] = 1.0;
        l[(i, i + 1)] = -2.0;
        l[(i, i + 2)] = 1.0;
    }
    l
}

/// Tikhonov-regularised NNLS. Solves min ||A x - y||² + λ²||L x||² s.t. x>=0
/// by augmenting [A; λL] and [y; 0].
pub fn nnls_tikhonov(
    a: &DMatrix<f64>,
    y: &DVector<f64>,
    l: &DMatrix<f64>,
    lambda: f64,
    max_iter: usize,
) -> DVector<f64> {
    if lambda <= 0.0 {
        return nnls(a, y, max_iter);
    }
    let n_rows = a.nrows() + l.nrows();
    let n_cols = a.ncols();
    let mut a_aug = DMatrix::<f64>::zeros(n_rows, n_cols);
    let mut y_aug = DVector::<f64>::zeros(n_rows);
    for i in 0..a.nrows() {
        for j in 0..n_cols {
            a_aug[(i, j)] = a[(i, j)];
        }
        y_aug[i] = y[i];
    }
    for i in 0..l.nrows() {
        for j in 0..n_cols {
            a_aug[(a.nrows() + i, j)] = lambda * l[(i, j)];
        }
    }
    nnls(&a_aug, &y_aug, max_iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nnls_recovers_simple() {
        // A = I, y = [1, 2, 3] -> x = y
        let a = DMatrix::<f64>::identity(3, 3);
        let y = DVector::<f64>::from_row_slice(&[1.0, 2.0, 3.0]);
        let x = nnls(&a, &y, 50);
        for i in 0..3 {
            assert!((x[i] - y[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn nnls_clips_negatives() {
        // If true x would be [1, -1] the NNLS solution must zero out the second.
        let a = DMatrix::<f64>::identity(2, 2);
        let y = DVector::<f64>::from_row_slice(&[1.0, -1.0]);
        let x = nnls(&a, &y, 50);
        assert!((x[0] - 1.0).abs() < 1e-9);
        assert!(x[1].abs() < 1e-9);
    }
}
