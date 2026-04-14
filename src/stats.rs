//! Statistical utility functions for HMM emission models.
//!
//! Port of hmmlearn's `stats.py` — provides log multivariate normal density
//! computation for all four covariance types.

use ndarray::{Array1, Array2, Array3, Axis};
use std::f64::consts::PI;

use crate::emissions::CovarianceType;

/// Compute the log probability under a multivariate Gaussian distribution.
///
/// # Arguments
/// * `x` - Observation matrix (n_samples, n_features).
/// * `means` - Mean vectors (n_components, n_features).
/// * `covars` - Covariance parameters (shape depends on `covariance_type`).
/// * `covariance_type` - Type of covariance parameterization.
///
/// # Returns
/// Log probability matrix (n_samples, n_components).
pub fn log_multivariate_normal_density(
    x: &Array2<f64>,
    means: &Array2<f64>,
    covars: &CovarsArg,
    covariance_type: CovarianceType,
) -> Array2<f64> {
    match covariance_type {
        CovarianceType::Diag => log_mvn_density_diag(x, means, covars.as_diag()),
        CovarianceType::Spherical => log_mvn_density_spherical(x, means, covars.as_spherical()),
        CovarianceType::Tied => log_mvn_density_tied(x, means, covars.as_tied()),
        CovarianceType::Full => log_mvn_density_full(x, means, covars.as_full()),
    }
}

/// Wrapper for passing covariance matrices with different shapes.
pub enum CovarsArg<'a> {
    Full(&'a Array3<f64>),
    Tied(&'a Array2<f64>),
    Diag(&'a Array2<f64>),
    Spherical(&'a Array1<f64>),
}

impl<'a> CovarsArg<'a> {
    pub fn as_full(&self) -> &Array3<f64> {
        match self {
            CovarsArg::Full(c) => c,
            _ => panic!("expected Full covariance"),
        }
    }
    pub fn as_tied(&self) -> &Array2<f64> {
        match self {
            CovarsArg::Tied(c) => c,
            _ => panic!("expected Tied covariance"),
        }
    }
    pub fn as_diag(&self) -> &Array2<f64> {
        match self {
            CovarsArg::Diag(c) => c,
            _ => panic!("expected Diag covariance"),
        }
    }
    pub fn as_spherical(&self) -> &Array1<f64> {
        match self {
            CovarsArg::Spherical(c) => c,
            _ => panic!("expected Spherical covariance"),
        }
    }
}

/// Diagonal covariance: covars shape (n_components, n_features).
///
/// log p(x|c) = -0.5 * (nf * ln(2π) + sum(ln(covars_c)) + sum((x - μ_c)² / covars_c))
fn log_mvn_density_diag(
    x: &Array2<f64>,
    means: &Array2<f64>,
    covars: &Array2<f64>,
) -> Array2<f64> {
    let nc = means.nrows();
    let nf = means.ncols();
    let ns = x.nrows();

    // Floor covars at f64::MIN_POSITIVE to avoid log(0).
    // Matches: np.maximum(covars, np.finfo(float).tiny)
    let covars_safe = covars.mapv(|v| v.max(f64::MIN_POSITIVE));

    let log_covars_sum: Array1<f64> = covars_safe.mapv(f64::ln).sum_axis(Axis(1));

    let mut result = Array2::<f64>::zeros((ns, nc));

    for c in 0..nc {
        for s in 0..ns {
            let mut sq_maha = 0.0;
            for f in 0..nf {
                let diff = x[[s, f]] - means[[c, f]];
                sq_maha += diff * diff / covars_safe[[c, f]];
            }
            result[[s, c]] = -0.5 * (nf as f64 * (2.0 * PI).ln() + log_covars_sum[c] + sq_maha);
        }
    }

    result
}

/// Spherical covariance: covars shape (n_components,).
/// Each component has a single variance applied to all features.
fn log_mvn_density_spherical(
    x: &Array2<f64>,
    means: &Array2<f64>,
    covars: &Array1<f64>,
) -> Array2<f64> {
    let nc = means.nrows();
    let nf = means.ncols();

    // Broadcast spherical to diagonal: (nc,) -> (nc, nf)
    let mut diag_covars = Array2::<f64>::zeros((nc, nf));
    for c in 0..nc {
        for f in 0..nf {
            diag_covars[[c, f]] = covars[c];
        }
    }

    log_mvn_density_diag(x, means, &diag_covars)
}

/// Tied covariance: single covariance matrix (n_features, n_features) shared by all components.
fn log_mvn_density_tied(
    x: &Array2<f64>,
    means: &Array2<f64>,
    covar: &Array2<f64>,
) -> Array2<f64> {
    let nc = means.nrows();
    let nf = means.ncols();

    // Broadcast tied to full: (nf, nf) -> (nc, nf, nf)
    let mut full_covars = Array3::<f64>::zeros((nc, nf, nf));
    for c in 0..nc {
        for i in 0..nf {
            for j in 0..nf {
                full_covars[[c, i, j]] = covar[[i, j]];
            }
        }
    }

    log_mvn_density_full(x, means, &full_covars)
}

/// Full covariance: covars shape (n_components, n_features, n_features).
///
/// Uses Cholesky decomposition for numerical stability.
/// Matches scipy's linalg.cholesky(cv, lower=True) convention.
fn log_mvn_density_full(
    x: &Array2<f64>,
    means: &Array2<f64>,
    covars: &Array3<f64>,
) -> Array2<f64> {
    let nc = means.nrows();
    let nf = means.ncols();
    let ns = x.nrows();
    let min_covar = 1e-7; // matches hmmlearn's default

    let mut result = Array2::<f64>::zeros((ns, nc));

    for c in 0..nc {
        // Extract the covariance matrix for component c
        let cv = covars.index_axis(Axis(0), c);

        // Cholesky decomposition: L such that cv = L @ L^T
        let l = match cholesky_lower(&cv.to_owned()) {
            Some(l) => l,
            None => {
                // Add min_covar to diagonal and retry
                let mut cv_reg = cv.to_owned();
                for i in 0..nf {
                    cv_reg[[i, i]] += min_covar;
                }
                cholesky_lower(&cv_reg)
                    .expect("covars must be symmetric, positive-definite")
            }
        };

        // log_det = 2 * sum(log(diag(L)))
        let cv_log_det: f64 = (0..nf).map(|i| l[[i, i]].ln()).sum::<f64>() * 2.0;

        // Solve L * y = (x - mu) for y via forward substitution, then ||y||^2
        for s in 0..ns {
            // diff = x[s] - means[c]
            let mut diff = Array1::<f64>::zeros(nf);
            for f in 0..nf {
                diff[f] = x[[s, f]] - means[[c, f]];
            }

            // Forward substitution: L * y = diff
            let mut y = Array1::<f64>::zeros(nf);
            for i in 0..nf {
                let mut val = diff[i];
                for j in 0..i {
                    val -= l[[i, j]] * y[j];
                }
                y[i] = val / l[[i, i]];
            }

            // ||y||^2 = Mahalanobis distance squared
            let sq_maha: f64 = y.iter().map(|&v| v * v).sum();

            result[[s, c]] = -0.5 * (nf as f64 * (2.0 * PI).ln() + sq_maha + cv_log_det);
        }
    }

    result
}

/// Compute lower-triangular Cholesky decomposition: A = L @ L^T.
/// Returns None if the matrix is not positive definite.
pub fn cholesky_lower(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    assert_eq!(n, a.ncols());
    let mut l = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let diag = a[[i, i]] - sum;
                if diag <= 0.0 {
                    return None;
                }
                l[[i, j]] = diag.sqrt();
            } else {
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Some(l)
}

/// Solve a lower-triangular system L * x = b via forward substitution.
pub fn solve_triangular_lower(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut x = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut val = b[i];
        for j in 0..i {
            val -= l[[i, j]] * x[j];
        }
        x[i] = val / l[[i, i]];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cholesky_lower() {
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let l = cholesky_lower(&a).unwrap();

        // Verify L @ L^T = A
        let lt = l.t();
        let result = l.dot(&lt);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (result[[i, j]] - a[[i, j]]).abs() < 1e-14,
                    "mismatch at [{},{}]",
                    i,
                    j
                );
            }
        }

        // L should be lower triangular
        assert!((l[[0, 1]]).abs() < 1e-14);
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        let a = array![[-1.0, 0.0], [0.0, 1.0]];
        assert!(cholesky_lower(&a).is_none());
    }

    #[test]
    fn test_diag_identity_covar() {
        let x = array![[0.0, 0.0]];
        let means = array![[0.0, 0.0]];
        let covars = array![[1.0, 1.0]];

        let result = log_mvn_density_diag(&x, &means, &covars);
        let expected = -0.5 * 2.0 * (2.0 * PI).ln();
        assert!(
            (result[[0, 0]] - expected).abs() < 1e-12,
            "got {} expected {}",
            result[[0, 0]],
            expected
        );
    }

    #[test]
    fn test_diag_off_center() {
        let x = array![[1.0, 0.0]];
        let means = array![[0.0, 0.0]];
        let covars = array![[1.0, 1.0]];

        let result = log_mvn_density_diag(&x, &means, &covars);
        let expected = -0.5 * (2.0 * (2.0 * PI).ln() + 1.0);
        assert!(
            (result[[0, 0]] - expected).abs() < 1e-12,
            "got {} expected {}",
            result[[0, 0]],
            expected
        );
    }

    #[test]
    fn test_full_identity_matches_diag() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let means = array![[0.0, 0.0], [1.0, 1.0]];
        let diag_covars = array![[1.0, 2.0], [0.5, 1.5]];

        let nc = 2;
        let nf = 2;
        let mut full_covars = Array3::<f64>::zeros((nc, nf, nf));
        for c in 0..nc {
            for f in 0..nf {
                full_covars[[c, f, f]] = diag_covars[[c, f]];
            }
        }

        let result_diag = log_mvn_density_diag(&x, &means, &diag_covars);
        let result_full = log_mvn_density_full(&x, &means, &full_covars);

        for s in 0..2 {
            for c in 0..2 {
                assert!(
                    (result_diag[[s, c]] - result_full[[s, c]]).abs() < 1e-12,
                    "mismatch at [{},{}]: diag={} full={}",
                    s,
                    c,
                    result_diag[[s, c]],
                    result_full[[s, c]]
                );
            }
        }
    }

    #[test]
    fn test_spherical_matches_diag() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let means = array![[0.0, 0.0], [1.0, 1.0]];
        let spherical = array![2.0, 0.5];
        let diag_covars = array![[2.0, 2.0], [0.5, 0.5]];

        let result_sph = log_mvn_density_spherical(&x, &means, &spherical);
        let result_diag = log_mvn_density_diag(&x, &means, &diag_covars);

        for s in 0..2 {
            for c in 0..2 {
                assert!(
                    (result_sph[[s, c]] - result_diag[[s, c]]).abs() < 1e-12,
                    "mismatch at [{},{}]",
                    s,
                    c
                );
            }
        }
    }

    #[test]
    fn test_tied_matches_full() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let means = array![[0.0, 0.0], [1.0, 1.0]];
        let tied = array![[2.0, 0.5], [0.5, 3.0]];

        let nc = 2;
        let nf = 2;
        let mut full_covars = Array3::<f64>::zeros((nc, nf, nf));
        for c in 0..nc {
            for i in 0..nf {
                for j in 0..nf {
                    full_covars[[c, i, j]] = tied[[i, j]];
                }
            }
        }

        let result_tied = log_mvn_density_tied(&x, &means, &tied);
        let result_full = log_mvn_density_full(&x, &means, &full_covars);

        for s in 0..2 {
            for c in 0..2 {
                assert!(
                    (result_tied[[s, c]] - result_full[[s, c]]).abs() < 1e-12,
                    "mismatch at [{},{}]",
                    s,
                    c
                );
            }
        }
    }

    #[test]
    fn test_full_non_diagonal() {
        let x = array![[1.0, 0.0]];
        let means = array![[0.0, 0.0]];
        let covars = Array3::from_shape_vec((1, 2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();

        let result = log_mvn_density_full(&x, &means, &covars);

        // Manual: det = 3, inv = [[2/3, -1/3], [-1/3, 2/3]]
        // Mahalanobis = [1,0] @ inv @ [1,0]^T = 2/3
        // log_prob = -0.5 * (2*ln(2π) + ln(3) + 2/3)
        let expected = -0.5 * (2.0 * (2.0 * PI).ln() + 3.0_f64.ln() + 2.0 / 3.0);
        assert!(
            (result[[0, 0]] - expected).abs() < 1e-12,
            "got {} expected {}",
            result[[0, 0]],
            expected
        );
    }
}
