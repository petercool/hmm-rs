//! KL divergence computations for variational inference.
//!
//! Port of hmmlearn's `_kl_divergence.py`.

use ndarray::{Array1, Array2};
use statrs::function::gamma::{digamma, ln_gamma};

use crate::stats::cholesky_lower;

/// KL divergence between two Dirichlet distributions.
///
/// KL(q || p) = ln[Γ(sum(q))/Γ(sum(p))] - sum[ln(Γ(q_j)/Γ(p_j))]
///              + sum[(q_j - p_j)(ψ(q_j) - ψ(sum(q)))]
pub fn kl_dirichlet(q: &Array1<f64>, p: &Array1<f64>) -> f64 {
    let qsum = q.sum();
    let psum = p.sum();

    let mut sum_gammaln_diff = 0.0;
    let mut einsum_term = 0.0;
    let digamma_qsum = digamma(qsum);

    for i in 0..q.len() {
        sum_gammaln_diff += ln_gamma(q[i]) - ln_gamma(p[i]);
        einsum_term += (q[i] - p[i]) * (digamma(q[i]) - digamma_qsum);
    }

    ln_gamma(qsum) - ln_gamma(psum) - sum_gammaln_diff + einsum_term
}

/// KL divergence between two 1D normal distributions.
pub fn kl_normal(mean_q: f64, var_q: f64, mean_p: f64, var_p: f64) -> f64 {
    ((var_p / var_q).ln()) / 2.0 + ((mean_q - mean_p).powi(2) + var_q) / (2.0 * var_p) - 0.5
}

/// KL divergence between two multivariate normal distributions.
pub fn kl_multivariate_normal(
    mean_q: &Array1<f64>,
    covar_q: &Array2<f64>,
    mean_p: &Array1<f64>,
    covar_p: &Array2<f64>,
) -> f64 {
    let d = mean_q.len();

    // Precision of p
    let precision_p = matrix_inverse(covar_p);
    let mean_diff = mean_q - mean_p;

    let logdet_p = logdet(covar_p);
    let logdet_q = logdet(covar_q);
    let trace_term = matrix_trace(&precision_p.dot(covar_q));
    let quad_form = mean_diff.dot(&precision_p.dot(&mean_diff));

    0.5 * (logdet_p - logdet_q + trace_term + quad_form - d as f64)
}

/// KL divergence between two Gamma distributions.
pub fn kl_gamma(b_q: f64, c_q: f64, b_p: f64, c_p: f64) -> f64 {
    (b_q - b_p) * digamma(b_q) - ln_gamma(b_q) + ln_gamma(b_p)
        + b_p * (c_q.ln() - c_p.ln())
        + b_q * (c_p - c_q) / c_q
}

/// KL divergence between two Wishart distributions.
pub fn kl_wishart(
    dof_q: f64,
    scale_q: &Array2<f64>,
    dof_p: f64,
    scale_p: &Array2<f64>,
) -> f64 {
    let d = scale_p.nrows();

    let e_q = expected_log_det_wishart(dof_q, scale_q);
    let inv_scale_q = matrix_inverse(scale_q);
    let trace_term = matrix_trace(&scale_p.dot(&inv_scale_q));

    (dof_q - dof_p) / 2.0 * e_q - d as f64 * dof_q / 2.0
        + dof_q / 2.0 * trace_term
        + log_partition_wishart(dof_p, scale_p)
        - log_partition_wishart(dof_q, scale_q)
}

/// E[log |Γ|] for Wishart distribution.
fn expected_log_det_wishart(dof: f64, scale: &Array2<f64>) -> f64 {
    let d = scale.nrows();
    let mut digamma_sum = 0.0;
    for i in 0..d {
        digamma_sum += digamma((dof - i as f64) / 2.0);
    }
    -logdet(&(scale / 2.0)) + digamma_sum
}

/// Log partition function of Wishart distribution.
fn log_partition_wishart(dof: f64, scale: &Array2<f64>) -> f64 {
    let d = scale.nrows();
    let mut gammaln_sum = 0.0;
    for i in 0..d {
        gammaln_sum += ln_gamma((dof - i as f64) / 2.0);
    }
    (d as f64 * (d as f64 - 1.0) / 4.0) * std::f64::consts::PI.ln()
        - dof / 2.0 * logdet(&(scale / 2.0))
        + gammaln_sum
}

/// Compute log determinant of a matrix via Cholesky.
pub fn logdet(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    match cholesky_lower(a) {
        Some(l) => {
            let mut ld = 0.0;
            for i in 0..n {
                ld += l[[i, i]].ln();
            }
            2.0 * ld
        }
        None => {
            // Fall back to LU-like computation
            // For now, use the product of diagonal if symmetric
            let mut prod = 0.0;
            for i in 0..n {
                prod += a[[i, i]].ln();
            }
            prod // approximate
        }
    }
}

/// Simple matrix inverse for small matrices using Cholesky.
fn matrix_inverse(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let l = cholesky_lower(a).unwrap_or_else(|| {
        // Add small regularization
        let mut a_reg = a.clone();
        for i in 0..n {
            a_reg[[i, i]] += 1e-10;
        }
        cholesky_lower(&a_reg).expect("matrix must be positive definite")
    });

    // Solve L * L^T * X = I using forward/back substitution
    let mut inv = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        // Solve L * y = e_col
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut val = if i == col { 1.0 } else { 0.0 };
            for j in 0..i {
                val -= l[[i, j]] * y[j];
            }
            y[i] = val / l[[i, i]];
        }

        // Solve L^T * x = y
        let mut x = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            let mut val = y[i];
            for j in (i + 1)..n {
                val -= l[[j, i]] * x[j];
            }
            x[i] = val / l[[i, i]];
        }

        inv.column_mut(col).assign(&x);
    }
    inv
}

/// Trace of a matrix.
fn matrix_trace(a: &Array2<f64>) -> f64 {
    let n = a.nrows().min(a.ncols());
    (0..n).map(|i| a[[i, i]]).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kl_dirichlet_same() {
        let p = array![1.0, 1.0, 1.0];
        let kl = kl_dirichlet(&p, &p);
        assert!(kl.abs() < 1e-12, "KL of same distribution should be 0, got {}", kl);
    }

    #[test]
    fn test_kl_dirichlet_different() {
        let q = array![2.0, 1.0, 1.0];
        let p = array![1.0, 1.0, 1.0];
        let kl = kl_dirichlet(&q, &p);
        assert!(kl >= -1e-12, "KL should be non-negative, got {}", kl);
    }

    #[test]
    fn test_kl_normal_same() {
        let kl = kl_normal(0.0, 1.0, 0.0, 1.0);
        assert!(kl.abs() < 1e-12);
    }

    #[test]
    fn test_kl_normal_different() {
        let kl = kl_normal(1.0, 1.0, 0.0, 1.0);
        assert!(kl > 0.0);
        // Known value: KL(N(1,1) || N(0,1)) = 0.5
        assert!((kl - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_kl_multivariate_normal_same() {
        let mean = array![0.0, 0.0];
        let covar = array![[1.0, 0.0], [0.0, 1.0]];
        let kl = kl_multivariate_normal(&mean, &covar, &mean, &covar);
        assert!(kl.abs() < 1e-10, "KL should be ~0, got {}", kl);
    }

    #[test]
    fn test_logdet_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let ld = logdet(&eye);
        assert!(ld.abs() < 1e-14);
    }

    #[test]
    fn test_matrix_inverse() {
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let inv = matrix_inverse(&a);
        let product = a.dot(&inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[[i, j]] - expected).abs() < 1e-12,
                    "product[{},{}] = {}",
                    i, j, product[[i, j]]
                );
            }
        }
    }
}
