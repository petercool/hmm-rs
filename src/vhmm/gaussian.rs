//! Variational Gaussian emission model.
//!
//! Port of hmmlearn's VariationalGaussianHMM.
//! Supports all 4 covariance types.
//! Internally stores scale_posterior/prior as full (nc, nf, nf) matrices
//! for computational simplicity (matching hmmlearn's approach).

use std::collections::HashMap;

use ndarray::{Array1, Array2, Array3, ArrayD, Axis, IxDyn};
use rand::Rng;
use statrs::function::gamma::digamma;

use crate::base::{EmissionModel, ParamFlags, SufficientStatistics};
use crate::emissions::CovarianceType;
use crate::error::Result;
use crate::kl_divergence;
use crate::stats;
use crate::vhmm::VariationalEmissionModel;

/// Variational Gaussian emission model with Normal-Wishart posterior.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VariationalGaussianEmissions {
    pub covariance_type: CovarianceType,
    pub n_features: Option<usize>,

    // Prior parameters
    pub means_prior_: Option<Array2<f64>>,
    pub beta_prior_: Option<Array1<f64>>,
    pub dof_prior_: Option<Array1<f64>>,
    /// Scale prior stored as full (nc, nf, nf) regardless of covariance_type.
    pub scale_prior_: Option<Array3<f64>>,

    // Posterior parameters
    pub means_posterior_: Option<Array2<f64>>,
    pub beta_posterior_: Option<Array1<f64>>,
    pub dof_posterior_: Option<Array1<f64>>,
    /// Scale posterior stored as full (nc, nf, nf) regardless of covariance_type.
    pub scale_posterior_: Option<Array3<f64>>,

    /// Point-estimate covariance (nc, nf, nf) — always full internally.
    covars_: Option<Array3<f64>>,
}

impl VariationalGaussianEmissions {
    pub fn new(covariance_type: CovarianceType) -> Self {
        Self {
            covariance_type,
            n_features: None,
            means_prior_: None,
            beta_prior_: None,
            dof_prior_: None,
            scale_prior_: None,
            means_posterior_: None,
            beta_posterior_: None,
            dof_posterior_: None,
            scale_posterior_: None,
            covars_: None,
        }
    }

}

impl EmissionModel for VariationalGaussianEmissions {
    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn set_n_features(&mut self, n: usize) {
        self.n_features = Some(n);
    }

    fn init(
        &mut self,
        _x: &Array2<f64>,
        _n_components: usize,
        _params: &ParamFlags,
        _rng: &mut impl Rng,
    ) {
        // Variational init happens in init_variational
    }

    fn check(&self, _n_components: usize) -> Result<()> {
        Ok(())
    }

    fn compute_log_likelihood(&self, x: &Array2<f64>) -> Array2<f64> {
        let means = self.means_posterior_.as_ref().unwrap();
        let covars = self.covars_.as_ref().unwrap();
        stats::log_multivariate_normal_density(
            x,
            means,
            &stats::CovarsArg::Full(covars),
            CovarianceType::Full,
        )
    }

    fn generate_sample_from_state(&self, state: usize, rng: &mut impl Rng) -> Array1<f64> {
        let means = self.means_posterior_.as_ref().unwrap();
        let covars = self.covars_.as_ref().unwrap();
        let nf = means.ncols();
        let cov = covars.index_axis(Axis(0), state).to_owned();
        let l = stats::cholesky_lower(&cov).unwrap();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let z: Array1<f64> = Array1::from_shape_fn(nf, |_| rng.sample(normal));
        let mut sample = means.row(state).to_owned();
        for i in 0..nf {
            for j in 0..=i {
                sample[i] += l[[i, j]] * z[j];
            }
        }
        sample
    }

    fn initialize_sufficient_statistics(&self, n_components: usize) -> SufficientStatistics {
        let nf = self.n_features.unwrap();
        let mut stats = SufficientStatistics::new();
        stats.insert("post".to_string(), ArrayD::zeros(IxDyn(&[n_components])));
        stats.insert(
            "obs".to_string(),
            ArrayD::zeros(IxDyn(&[n_components, nf])),
        );
        stats.insert(
            "obs**2".to_string(),
            ArrayD::zeros(IxDyn(&[n_components, nf])),
        );
        stats.insert(
            "obs*obs.T".to_string(),
            ArrayD::zeros(IxDyn(&[n_components, nf, nf])),
        );
        stats
    }

    fn accumulate_sufficient_statistics(
        &self,
        stats: &mut SufficientStatistics,
        x: &Array2<f64>,
        posteriors: &Array2<f64>,
        params: &ParamFlags,
    ) {
        let nc = posteriors.ncols();
        let ns = x.nrows();
        let nf = x.ncols();

        if params.contains('m') || params.contains('c') {
            let post = stats.get_mut("post").unwrap();
            for c in 0..nc {
                let s: f64 = (0..ns).map(|s| posteriors[[s, c]]).sum();
                post[IxDyn(&[c])] += s;
            }

            let obs = stats.get_mut("obs").unwrap();
            for c in 0..nc {
                for f in 0..nf {
                    let s: f64 = (0..ns).map(|s| posteriors[[s, c]] * x[[s, f]]).sum();
                    obs[IxDyn(&[c, f])] += s;
                }
            }
        }

        if params.contains('c') {
            // Always accumulate obs**2 (needed for diag/spherical M-step)
            let obs2 = stats.get_mut("obs**2").unwrap();
            for c in 0..nc {
                for f in 0..nf {
                    let s: f64 = (0..ns).map(|s| posteriors[[s, c]] * x[[s, f]] * x[[s, f]]).sum();
                    obs2[IxDyn(&[c, f])] += s;
                }
            }

            // Always accumulate obs*obs.T (needed for full/tied M-step)
            let obs_obt = stats.get_mut("obs*obs.T").unwrap();
            for c in 0..nc {
                for k in 0..nf {
                    for l in 0..nf {
                        let s: f64 =
                            (0..ns).map(|s| posteriors[[s, c]] * x[[s, k]] * x[[s, l]]).sum();
                        obs_obt[IxDyn(&[c, k, l])] += s;
                    }
                }
            }
        }
    }

    fn do_mstep(&mut self, _stats: &SufficientStatistics, _params: &ParamFlags) {}

    fn n_fit_scalars_per_param(&self, n_components: usize) -> HashMap<char, usize> {
        let nc = n_components;
        let nf = self.n_features.unwrap_or(0);
        let mut m = HashMap::new();
        m.insert('m', nc * nf + nc); // means + beta
        m.insert(
            'c',
            match self.covariance_type {
                CovarianceType::Full => nc + nc * nf * (nf + 1) / 2,
                CovarianceType::Tied => 1 + nf * (nf + 1) / 2,
                CovarianceType::Diag => nc + nc * nf,
                CovarianceType::Spherical => nc + nc,
            },
        );
        m
    }
}

impl VariationalEmissionModel for VariationalGaussianEmissions {
    fn compute_subnorm_log_likelihood(&self, x: &Array2<f64>) -> Array2<f64> {
        let nf = self.n_features.unwrap();
        let means = self.means_posterior_.as_ref().unwrap();
        let nc = means.nrows();
        let ns = x.nrows();

        let dof = self.dof_posterior_.as_ref().unwrap();
        let scale_post = self.scale_posterior_.as_ref().unwrap();
        let beta = self.beta_posterior_.as_ref().unwrap();

        // term1 = sum(digamma((dof - arange(nf)) / 2))
        let mut term1 = Array1::<f64>::zeros(nc);
        for c in 0..nc {
            for i in 0..nf {
                term1[c] += digamma((dof[c] - i as f64) / 2.0);
            }
        }

        // W_k = inv(scale_posterior) — scale_posterior is always (nc, nf, nf)
        let mut w_k = Array3::<f64>::zeros((nc, nf, nf));
        for c in 0..nc {
            let sp = scale_post.index_axis(Axis(0), c).to_owned();
            let inv = matrix_inverse_cholesky(&sp);
            for i in 0..nf {
                for j in 0..nf {
                    w_k[[c, i, j]] = inv[[i, j]];
                }
            }
        }

        // term1 += nf * ln(2) + logdet(W_k)
        for c in 0..nc {
            let wk_c = w_k.index_axis(Axis(0), c).to_owned();
            term1[c] += nf as f64 * 2.0_f64.ln() + kl_divergence::logdet(&wk_c);
        }
        term1 /= 2.0;

        // term3 = nf / beta
        let term3 = Array1::from_shape_fn(nc, |c| nf as f64 / beta[c]);

        // Compute Mahalanobis: (X - means) * W_k * (X - means)^T * dof
        let mut result = Array2::<f64>::zeros((ns, nc));
        for c in 0..nc {
            for s in 0..ns {
                let mut maha = 0.0;
                for i in 0..nf {
                    for j in 0..nf {
                        let di = x[[s, i]] - means[[c, i]];
                        let dj = x[[s, j]] - means[[c, j]];
                        maha += di * w_k[[c, i, j]] * dj;
                    }
                }
                maha *= dof[c];
                result[[s, c]] = term1[c] - 0.5 * (maha + term3[c]);
            }
        }

        result
    }

    fn estep_begin(&mut self) {}

    fn emission_kl_divergence(&self) -> f64 {
        let nc = self.means_posterior_.as_ref().unwrap().nrows();
        let nf = self.n_features.unwrap();
        let means_post = self.means_posterior_.as_ref().unwrap();
        let means_prior = self.means_prior_.as_ref().unwrap();
        let beta_post = self.beta_posterior_.as_ref().unwrap();
        let beta_prior = self.beta_prior_.as_ref().unwrap();
        let dof_post = self.dof_posterior_.as_ref().unwrap();
        let dof_prior = self.dof_prior_.as_ref().unwrap();
        let scale_post = self.scale_posterior_.as_ref().unwrap();
        let scale_prior = self.scale_prior_.as_ref().unwrap();

        let mut kl = 0.0;
        for c in 0..nc {
            let sp = scale_post.index_axis(Axis(0), c).to_owned();
            let w_k = matrix_inverse_cholesky(&sp);
            let precision = &w_k * dof_post[c];

            // KL for normal
            let mut covar_q = Array2::<f64>::zeros((nf, nf));
            let mut covar_p = Array2::<f64>::zeros((nf, nf));
            for i in 0..nf {
                for j in 0..nf {
                    covar_q[[i, j]] = if precision[[i, j]].abs() > 1e-300 {
                        sp[[i, j]] / (beta_post[c] * dof_post[c])
                    } else if i == j {
                        1.0 / beta_post[c]
                    } else {
                        0.0
                    };
                    covar_p[[i, j]] = if precision[[i, j]].abs() > 1e-300 {
                        sp[[i, j]] / (beta_prior[c] * dof_post[c])
                    } else if i == j {
                        1.0 / beta_prior[c]
                    } else {
                        0.0
                    };
                }
            }

            kl += kl_divergence::kl_multivariate_normal(
                &means_post.row(c).to_owned(),
                &covar_q,
                &means_prior.row(c).to_owned(),
                &covar_p,
            );

            // KL for Wishart
            kl += kl_divergence::kl_wishart(
                dof_post[c],
                &sp,
                dof_prior[c],
                &scale_prior.index_axis(Axis(0), c).to_owned(),
            );
        }
        kl
    }

    fn init_variational(
        &mut self,
        x: &Array2<f64>,
        _lengths: &[usize],
        n_components: usize,
        _rng: &mut impl Rng,
    ) {
        let nf = x.ncols();
        self.n_features = Some(nf);
        let nc = n_components;
        let ns = x.nrows();

        let x_mean = x.mean_axis(Axis(0)).unwrap();

        // Means
        self.means_prior_ = Some(Array2::from_shape_fn((nc, nf), |(_, f)| x_mean[f]));
        let mut means_post = Array2::<f64>::zeros((nc, nf));
        for c in 0..nc {
            let idx = (c * ns) / nc;
            means_post.row_mut(c).assign(&x.row(idx.min(ns - 1)));
        }
        self.means_posterior_ = Some(means_post);

        // Beta
        self.beta_prior_ = Some(Array1::from_elem(nc, 1.0));
        self.beta_posterior_ = Some(Array1::from_elem(nc, (ns / nc).max(1) as f64));

        // DOF
        self.dof_prior_ = Some(Array1::from_elem(nc, nf as f64));
        self.dof_posterior_ = Some(Array1::from_elem(nc, (ns / nc).max(1) as f64));

        // Sample covariance
        let mut cv = Array2::<f64>::zeros((nf, nf));
        for s in 0..ns {
            for i in 0..nf {
                for j in 0..nf {
                    cv[[i, j]] += (x[[s, i]] - x_mean[i]) * (x[[s, j]] - x_mean[j]);
                }
            }
        }
        if ns > 1 {
            cv /= (ns - 1) as f64;
        }
        for i in 0..nf {
            cv[[i, i]] += 1e-3;
        }

        // Scale prior — always stored as (nc, nf, nf)
        let mut scale_prior = Array3::<f64>::zeros((nc, nf, nf));
        for c in 0..nc {
            for i in 0..nf {
                scale_prior[[c, i, i]] = 1e-3;
            }
        }
        self.scale_prior_ = Some(scale_prior);

        // Scale posterior: cv * dof — stored as (nc, nf, nf)
        let dof_post = self.dof_posterior_.as_ref().unwrap();
        let mut scale_post = Array3::<f64>::zeros((nc, nf, nf));
        for c in 0..nc {
            for i in 0..nf {
                for j in 0..nf {
                    // For diag/spherical, only set diagonal elements from cv
                    match self.covariance_type {
                        CovarianceType::Full | CovarianceType::Tied => {
                            scale_post[[c, i, j]] = cv[[i, j]] * dof_post[c];
                        }
                        CovarianceType::Diag => {
                            if i == j {
                                scale_post[[c, i, j]] = cv[[i, i]] * dof_post[c];
                            }
                        }
                        CovarianceType::Spherical => {
                            if i == j {
                                let mean_var =
                                    (0..nf).map(|k| cv[[k, k]]).sum::<f64>() / nf as f64;
                                scale_post[[c, i, j]] = mean_var * dof_post[c];
                            }
                        }
                    }
                }
            }
        }
        self.scale_posterior_ = Some(scale_post.clone());

        // Point estimate covars — always full (nc, nf, nf)
        let mut covars = Array3::<f64>::zeros((nc, nf, nf));
        for c in 0..nc {
            for i in 0..nf {
                for j in 0..nf {
                    covars[[c, i, j]] = scale_post[[c, i, j]] / dof_post[c];
                }
            }
        }
        self.covars_ = Some(covars);
    }

    fn do_mstep_variational(&mut self, stats: &SufficientStatistics, params: &ParamFlags) {
        let nc = self.means_posterior_.as_ref().unwrap().nrows();
        let nf = self.n_features.unwrap();

        let post = stats.get("post").unwrap();
        let obs = stats.get("obs").unwrap();

        if params.contains('m') {
            let beta_prior = self.beta_prior_.as_ref().unwrap();
            let means_prior = self.means_prior_.as_ref().unwrap();
            let beta_post = self.beta_posterior_.as_mut().unwrap();
            let means_post = self.means_posterior_.as_mut().unwrap();

            for c in 0..nc {
                beta_post[c] = beta_prior[c] + post[IxDyn(&[c])];
                for f in 0..nf {
                    means_post[[c, f]] =
                        (beta_prior[c] * means_prior[[c, f]] + obs[IxDyn(&[c, f])]) / beta_post[c];
                }
            }
        }

        if params.contains('c') {
            let dof_prior = self.dof_prior_.as_ref().unwrap();
            let scale_prior = self.scale_prior_.as_ref().unwrap();
            let beta_prior = self.beta_prior_.as_ref().unwrap();
            let means_prior = self.means_prior_.as_ref().unwrap();
            let beta_post = self.beta_posterior_.as_ref().unwrap();
            let means_post = self.means_posterior_.as_ref().unwrap();
            let dof_post = self.dof_posterior_.as_mut().unwrap();
            let scale_post = self.scale_posterior_.as_mut().unwrap();
            let covars = self.covars_.as_mut().unwrap();

            match self.covariance_type {
                CovarianceType::Full => {
                    let obs_obt = stats.get("obs*obs.T").unwrap();
                    for c in 0..nc {
                        dof_post[c] = dof_prior[c] + post[IxDyn(&[c])];
                        for k in 0..nf {
                            for l in 0..nf {
                                scale_post[[c, k, l]] = scale_prior[[c, k, l]]
                                    + obs_obt[IxDyn(&[c, k, l])]
                                    + beta_prior[c]
                                        * means_prior[[c, k]]
                                        * means_prior[[c, l]]
                                    - beta_post[c] * means_post[[c, k]] * means_post[[c, l]];
                            }
                        }
                        for k in 0..nf {
                            for l in 0..nf {
                                covars[[c, k, l]] = scale_post[[c, k, l]] / dof_post[c];
                            }
                        }
                    }
                }
                CovarianceType::Tied => {
                    let obs_obt = stats.get("obs*obs.T").unwrap();
                    let total_dof = dof_prior[0] + post.iter().sum::<f64>();
                    for c in 0..nc {
                        dof_post[c] = total_dof; // shared
                    }
                    // Sum scale across components
                    let mut scale_sum = Array2::<f64>::zeros((nf, nf));
                    for c in 0..nc {
                        for k in 0..nf {
                            for l in 0..nf {
                                scale_sum[[k, l]] += obs_obt[IxDyn(&[c, k, l])]
                                    + beta_prior[c]
                                        * means_prior[[c, k]]
                                        * means_prior[[c, l]]
                                    - beta_post[c] * means_post[[c, k]] * means_post[[c, l]];
                            }
                        }
                    }
                    for c in 0..nc {
                        for k in 0..nf {
                            for l in 0..nf {
                                scale_post[[c, k, l]] =
                                    scale_prior[[c, k, l]] + scale_sum[[k, l]];
                                covars[[c, k, l]] = scale_post[[c, k, l]] / total_dof;
                            }
                        }
                    }
                }
                CovarianceType::Diag => {
                    let obs2 = stats.get("obs**2").unwrap();
                    for c in 0..nc {
                        dof_post[c] = dof_prior[c] + post[IxDyn(&[c])];
                        for f in 0..nf {
                            let s = scale_prior[[c, f, f]]
                                + obs2[IxDyn(&[c, f])]
                                + beta_prior[c]
                                    * means_prior[[c, f]]
                                    * means_prior[[c, f]]
                                - beta_post[c] * means_post[[c, f]] * means_post[[c, f]];
                            scale_post[[c, f, f]] = s;
                            covars[[c, f, f]] = s / dof_post[c];
                            // Zero off-diagonals
                            for g in 0..nf {
                                if g != f {
                                    scale_post[[c, f, g]] = 0.0;
                                    covars[[c, f, g]] = 0.0;
                                }
                            }
                        }
                    }
                }
                CovarianceType::Spherical => {
                    let obs2 = stats.get("obs**2").unwrap();
                    for c in 0..nc {
                        dof_post[c] = dof_prior[c] + post[IxDyn(&[c])];
                        // Compute mean of diagonal scale elements
                        let mut mean_s = 0.0;
                        for f in 0..nf {
                            mean_s += scale_prior[[c, f, f]]
                                + obs2[IxDyn(&[c, f])]
                                + beta_prior[c]
                                    * means_prior[[c, f]]
                                    * means_prior[[c, f]]
                                - beta_post[c] * means_post[[c, f]] * means_post[[c, f]];
                        }
                        mean_s /= nf as f64;
                        for f in 0..nf {
                            scale_post[[c, f, f]] = mean_s;
                            covars[[c, f, f]] = mean_s / dof_post[c];
                            for g in 0..nf {
                                if g != f {
                                    scale_post[[c, f, g]] = 0.0;
                                    covars[[c, f, g]] = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Matrix inverse via Cholesky.
fn matrix_inverse_cholesky(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let l = stats::cholesky_lower(a).unwrap_or_else(|| {
        let mut a_reg = a.clone();
        for i in 0..n {
            a_reg[[i, i]] += 1e-10;
        }
        stats::cholesky_lower(&a_reg).expect("matrix inverse failed")
    });

    let mut inv = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut val = if i == col { 1.0 } else { 0.0 };
            for j in 0..i {
                val -= l[[i, j]] * y[j];
            }
            y[i] = val / l[[i, i]];
        }
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

pub type VariationalGaussianHmm =
    crate::vhmm::VariationalBaseHmm<VariationalGaussianEmissions>;

impl VariationalGaussianHmm {
    pub fn variational_gaussian(
        n_components: usize,
        covariance_type: CovarianceType,
    ) -> Self {
        Self::new(
            n_components,
            VariationalGaussianEmissions::new(covariance_type),
        )
        .with_params("stmc")
        .with_init_params("stmc")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn test_data() -> Array2<f64> {
        array![
            [0.1, 0.2],
            [-0.1, 0.3],
            [0.2, -0.1],
            [5.1, 5.2],
            [4.9, 5.3],
            [5.2, 4.9],
            [0.0, 0.1],
            [5.0, 5.0],
            [-0.2, 0.1],
        ]
    }

    #[test]
    fn test_variational_gaussian_full() {
        let x = test_data();
        let mut model =
            VariationalGaussianHmm::variational_gaussian(2, CovarianceType::Full)
                .with_n_iter(30)
                .with_tol(1e-6);
        model.fit(&x, &[x.nrows()]).unwrap();
        let score = model.score(&x, &[x.nrows()]).unwrap();
        assert!(score.is_finite(), "Full: score={}", score);
    }

    #[test]
    fn test_variational_gaussian_diag() {
        let x = test_data();
        let mut model =
            VariationalGaussianHmm::variational_gaussian(2, CovarianceType::Diag)
                .with_n_iter(30)
                .with_tol(1e-6);
        model.fit(&x, &[x.nrows()]).unwrap();
        let score = model.score(&x, &[x.nrows()]).unwrap();
        assert!(score.is_finite(), "Diag: score={}", score);
    }

    #[test]
    fn test_variational_gaussian_tied() {
        let x = test_data();
        let mut model =
            VariationalGaussianHmm::variational_gaussian(2, CovarianceType::Tied)
                .with_n_iter(30)
                .with_tol(1e-6);
        model.fit(&x, &[x.nrows()]).unwrap();
        let score = model.score(&x, &[x.nrows()]).unwrap();
        assert!(score.is_finite(), "Tied: score={}", score);
    }

    #[test]
    fn test_variational_gaussian_spherical() {
        let x = test_data();
        let mut model =
            VariationalGaussianHmm::variational_gaussian(2, CovarianceType::Spherical)
                .with_n_iter(30)
                .with_tol(1e-6);
        model.fit(&x, &[x.nrows()]).unwrap();
        let score = model.score(&x, &[x.nrows()]).unwrap();
        assert!(score.is_finite(), "Spherical: score={}", score);
    }
}
