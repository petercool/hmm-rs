//! Gaussian Mixture Model emission model.
//!
//! Port of hmmlearn's GMMHMM emission logic.
//! Each state emits from a mixture of Gaussians.

use std::collections::HashMap;

use ndarray::{Array1, Array2, Array3, ArrayD, Axis, IxDyn};
use rand::Rng;
use rand_distr::Normal;

use crate::base::{EmissionModel, ParamFlags, SufficientStatistics};
use crate::emissions::CovarianceType;
use crate::error::{HmmError, Result};
use crate::stats::{self, CovarsArg};
use crate::utils;

/// GMM emission model: each HMM state has a mixture of Gaussians.
///
/// Note: "tied" covariance for GMMHMM means all mixture components within each state
/// share the same covariance — different from GaussianHMM's "tied" which means all states share.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GmmEmissions {
    pub covariance_type: CovarianceType,
    pub min_covar: f64,
    pub n_features: Option<usize>,
    pub n_mix: usize,

    /// Mixture weights: (n_components, n_mix).
    pub weights_: Option<Array2<f64>>,
    /// Means: (n_components, n_mix, n_features).
    pub means_: Option<Array3<f64>>,

    // Covariances — shape depends on covariance_type:
    // Full: (n_components, n_mix, nf, nf)
    // Tied: (n_components, nf, nf)
    // Diag: (n_components, n_mix, nf)
    // Spherical: (n_components, n_mix)
    covars_full_: Option<ndarray::Array4<f64>>,
    covars_tied_: Option<Array3<f64>>,
    covars_diag_: Option<Array3<f64>>,
    covars_spherical_: Option<Array2<f64>>,

    // Priors
    pub weights_prior: f64,
    pub means_prior: f64,
    pub means_weight: f64,
    pub covars_prior: f64,
    pub covars_weight: f64,
}

impl GmmEmissions {
    pub fn new(n_mix: usize, covariance_type: CovarianceType) -> Self {
        Self {
            covariance_type,
            min_covar: 1e-3,
            n_features: None,
            n_mix,
            weights_: None,
            means_: None,
            covars_full_: None,
            covars_tied_: None,
            covars_diag_: None,
            covars_spherical_: None,
            weights_prior: 1.0,
            means_prior: 0.0,
            means_weight: 0.0,
            covars_prior: 0.0,
            covars_weight: 0.0,
        }
    }

    /// Compute log-weighted Gaussian densities for a single HMM component.
    fn compute_log_weighted_gaussian_densities(
        &self,
        x: &Array2<f64>,
        i_comp: usize,
    ) -> Array2<f64> {
        let means = self.means_.as_ref().unwrap();
        let weights = self.weights_.as_ref().unwrap();
        let nm = self.n_mix;
        let nf = means.shape()[2];

        // Extract means for this component: (n_mix, n_features)
        let cur_means = means.index_axis(Axis(0), i_comp).to_owned();

        // Build covars arg for this component's mixtures
        let log_densities = match self.covariance_type {
            CovarianceType::Diag => {
                let c = self.covars_diag_.as_ref().unwrap();
                let cur_covs = c.index_axis(Axis(0), i_comp).to_owned();
                stats::log_multivariate_normal_density(
                    x,
                    &cur_means,
                    &CovarsArg::Diag(&cur_covs),
                    CovarianceType::Diag,
                )
            }
            CovarianceType::Spherical => {
                let c = self.covars_spherical_.as_ref().unwrap();
                let cur_covs = c.row(i_comp).to_owned();
                stats::log_multivariate_normal_density(
                    x,
                    &cur_means,
                    &CovarsArg::Spherical(&cur_covs),
                    CovarianceType::Spherical,
                )
            }
            CovarianceType::Full => {
                let c = self.covars_full_.as_ref().unwrap();
                // Extract (n_mix, nf, nf)
                let mut cur_covs = Array3::<f64>::zeros((nm, nf, nf));
                for m in 0..nm {
                    for i in 0..nf {
                        for j in 0..nf {
                            cur_covs[[m, i, j]] = c[[i_comp, m, i, j]];
                        }
                    }
                }
                stats::log_multivariate_normal_density(
                    x,
                    &cur_means,
                    &CovarsArg::Full(&cur_covs),
                    CovarianceType::Full,
                )
            }
            CovarianceType::Tied => {
                let c = self.covars_tied_.as_ref().unwrap();
                let cur_cov = c.index_axis(Axis(0), i_comp).to_owned();
                stats::log_multivariate_normal_density(
                    x,
                    &cur_means,
                    &CovarsArg::Tied(&cur_cov),
                    CovarianceType::Tied,
                )
            }
        };

        // Add log weights
        let ns = x.nrows();
        let mut result = log_densities;
        for s in 0..ns {
            for m in 0..nm {
                result[[s, m]] += weights[[i_comp, m]].ln();
            }
        }
        result
    }
}

impl EmissionModel for GmmEmissions {
    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn set_n_features(&mut self, n: usize) {
        self.n_features = Some(n);
    }

    fn init(
        &mut self,
        x: &Array2<f64>,
        n_components: usize,
        params: &ParamFlags,
        _rng: &mut impl Rng,
    ) {
        let nf = x.ncols();
        self.n_features = Some(nf);
        let nm = self.n_mix;

        if params.contains('w') || self.weights_.is_none() {
            self.weights_ = Some(Array2::from_elem((n_components, nm), 1.0 / nm as f64));
        }

        if params.contains('m') || self.means_.is_none() {
            // Simple init: spread data evenly across components and mixtures
            let ns = x.nrows();
            let mut means = Array3::<f64>::zeros((n_components, nm, nf));
            for c in 0..n_components {
                for m in 0..nm {
                    let idx = ((c * nm + m) * ns) / (n_components * nm);
                    means
                        .index_axis_mut(Axis(0), c)
                        .row_mut(m)
                        .assign(&x.row(idx.min(ns - 1)));
                }
            }
            self.means_ = Some(means);
        }

        if params.contains('c') || !self.has_covars() {
            // Sample covariance
            let mean = x.mean_axis(Axis(0)).unwrap();
            let mut cv = Array2::<f64>::zeros((nf, nf));
            for s in 0..x.nrows() {
                for i in 0..nf {
                    for j in 0..nf {
                        cv[[i, j]] += (x[[s, i]] - mean[i]) * (x[[s, j]] - mean[j]);
                    }
                }
            }
            if x.nrows() > 1 {
                cv /= (x.nrows() - 1) as f64;
            }
            for i in 0..nf {
                cv[[i, i]] += self.min_covar;
            }

            match self.covariance_type {
                CovarianceType::Diag => {
                    let mut covars = Array3::<f64>::zeros((n_components, nm, nf));
                    for c in 0..n_components {
                        for m in 0..nm {
                            for f in 0..nf {
                                covars[[c, m, f]] = cv[[f, f]];
                            }
                        }
                    }
                    self.covars_diag_ = Some(covars);
                }
                CovarianceType::Spherical => {
                    let mean_var = (0..nf).map(|i| cv[[i, i]]).sum::<f64>() / nf as f64;
                    self.covars_spherical_ = Some(Array2::from_elem((n_components, nm), mean_var));
                }
                CovarianceType::Full => {
                    let mut covars = ndarray::Array4::<f64>::zeros((n_components, nm, nf, nf));
                    for c in 0..n_components {
                        for m in 0..nm {
                            for i in 0..nf {
                                for j in 0..nf {
                                    covars[[c, m, i, j]] = cv[[i, j]];
                                }
                            }
                        }
                    }
                    self.covars_full_ = Some(covars);
                }
                CovarianceType::Tied => {
                    let mut covars = Array3::<f64>::zeros((n_components, nf, nf));
                    for c in 0..n_components {
                        for i in 0..nf {
                            for j in 0..nf {
                                covars[[c, i, j]] = cv[[i, j]];
                            }
                        }
                    }
                    self.covars_tied_ = Some(covars);
                }
            }
        }
    }

    fn check(&self, n_components: usize) -> Result<()> {
        let means = self
            .means_
            .as_ref()
            .ok_or_else(|| HmmError::NotFitted("means_ not set".into()))?;
        if means.shape()[0] != n_components || means.shape()[1] != self.n_mix {
            return Err(HmmError::InvalidParameter("means_ shape mismatch".into()));
        }
        let weights = self
            .weights_
            .as_ref()
            .ok_or_else(|| HmmError::NotFitted("weights_ not set".into()))?;
        if weights.shape() != [n_components, self.n_mix] {
            return Err(HmmError::InvalidParameter("weights_ shape mismatch".into()));
        }
        if !self.has_covars() {
            return Err(HmmError::NotFitted("covars not set".into()));
        }
        Ok(())
    }

    fn compute_log_likelihood(&self, x: &Array2<f64>) -> Array2<f64> {
        let nc = self.weights_.as_ref().unwrap().nrows();
        let ns = x.nrows();
        let mut logprobs = Array2::<f64>::zeros((ns, nc));

        for i in 0..nc {
            let log_denses = self.compute_log_weighted_gaussian_densities(x, i);
            // logsumexp across mixture components
            for s in 0..ns {
                let row: Vec<f64> = (0..self.n_mix).map(|m| log_denses[[s, m]]).collect();
                logprobs[[s, i]] = crate::algorithms::logsumexp(&row);
            }
        }

        logprobs
    }

    fn generate_sample_from_state(&self, state: usize, rng: &mut impl Rng) -> Array1<f64> {
        let weights = self.weights_.as_ref().unwrap();
        let means = self.means_.as_ref().unwrap();
        let nf = means.shape()[2];

        // Choose mixture component
        let u: f64 = rng.random();
        let mut cumsum = 0.0;
        let mut i_gauss = self.n_mix - 1;
        for m in 0..self.n_mix {
            cumsum += weights[[state, m]];
            if cumsum > u {
                i_gauss = m;
                break;
            }
        }

        // Get mean and covariance for chosen mixture
        let mean: Array1<f64> = means.index_axis(Axis(0), state).row(i_gauss).to_owned();

        // Build full covariance for this mixture
        let cov = self.get_full_cov_for_mix(state, i_gauss);
        let l = stats::cholesky_lower(&cov).unwrap();

        let normal = Normal::new(0.0, 1.0).unwrap();
        let z: Array1<f64> = Array1::from_shape_fn(nf, |_| rng.sample(normal));
        let mut sample = mean.clone();
        for i in 0..nf {
            for j in 0..=i {
                sample[i] += l[[i, j]] * z[j];
            }
        }
        sample
    }

    fn initialize_sufficient_statistics(&self, n_components: usize) -> SufficientStatistics {
        let nf = self.n_features.unwrap();
        let nm = self.n_mix;
        let mut stats = SufficientStatistics::new();
        stats.insert(
            "post_mix_sum".to_string(),
            ArrayD::zeros(IxDyn(&[n_components, nm])),
        );
        stats.insert(
            "post_sum".to_string(),
            ArrayD::zeros(IxDyn(&[n_components])),
        );
        stats.insert(
            "m_n".to_string(),
            ArrayD::zeros(IxDyn(&[n_components, nm, nf])),
        );
        stats.insert(
            "c_n".to_string(),
            match self.covariance_type {
                CovarianceType::Full => ArrayD::zeros(IxDyn(&[n_components, nm, nf, nf])),
                CovarianceType::Diag => ArrayD::zeros(IxDyn(&[n_components, nm, nf])),
                CovarianceType::Spherical => ArrayD::zeros(IxDyn(&[n_components, nm])),
                CovarianceType::Tied => ArrayD::zeros(IxDyn(&[n_components, nf, nf])),
            },
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
        let nm = self.n_mix;
        let means = self.means_.as_ref().unwrap();

        // Compute post_mix: (ns, nc, nm)
        // For each component, compute mixture responsibilities
        let mut post_comp_mix = vec![0.0f64; ns * nc * nm];

        for p in 0..nc {
            let log_denses = self.compute_log_weighted_gaussian_densities(x, p);
            // Normalize across mixtures (log_normalize each row)
            let mut normalized = log_denses.clone();
            utils::log_normalize_rows(&mut normalized);
            let mix_post = normalized.mapv(f64::exp);

            for s in 0..ns {
                for m in 0..nm {
                    post_comp_mix[s * nc * nm + p * nm + m] = posteriors[[s, p]] * mix_post[[s, m]];
                }
            }
        }

        // Accumulate post_mix_sum and post_sum
        {
            // Compute sums first to avoid double mutable borrow
            let mut mix_sums = vec![0.0f64; nc * nm];
            let mut comp_sums = vec![0.0f64; nc];
            for c in 0..nc {
                for m in 0..nm {
                    let mut ms = 0.0;
                    for s in 0..ns {
                        ms += post_comp_mix[s * nc * nm + c * nm + m];
                    }
                    mix_sums[c * nm + m] = ms;
                    comp_sums[c] += ms;
                }
            }
            let pms = stats.get_mut("post_mix_sum").unwrap();
            for c in 0..nc {
                for m in 0..nm {
                    pms[IxDyn(&[c, m])] += mix_sums[c * nm + m];
                }
            }
            let ps = stats.get_mut("post_sum").unwrap();
            for c in 0..nc {
                ps[IxDyn(&[c])] += comp_sums[c];
            }
        }

        // Accumulate m_n (means stats)
        if params.contains('m') {
            let m_n = stats.get_mut("m_n").unwrap();
            for c in 0..nc {
                for m in 0..nm {
                    for f in 0..nf {
                        let mut sum = 0.0;
                        for s in 0..ns {
                            sum += post_comp_mix[s * nc * nm + c * nm + m] * x[[s, f]];
                        }
                        m_n[IxDyn(&[c, m, f])] += sum;
                    }
                }
            }
        }

        // Accumulate c_n (covariance stats)
        if params.contains('c') {
            let c_n = stats.get_mut("c_n").unwrap();

            for s in 0..ns {
                for c in 0..nc {
                    for m in 0..nm {
                        let w = post_comp_mix[s * nc * nm + c * nm + m];
                        if w < 1e-300 {
                            continue;
                        }

                        match self.covariance_type {
                            CovarianceType::Full => {
                                for k in 0..nf {
                                    let dk = x[[s, k]] - means[[c, m, k]];
                                    for l in 0..nf {
                                        let dl = x[[s, l]] - means[[c, m, l]];
                                        c_n[IxDyn(&[c, m, k, l])] += w * dk * dl;
                                    }
                                }
                            }
                            CovarianceType::Diag => {
                                for f in 0..nf {
                                    let d = x[[s, f]] - means[[c, m, f]];
                                    c_n[IxDyn(&[c, m, f])] += w * d * d;
                                }
                            }
                            CovarianceType::Spherical => {
                                let mut norm2 = 0.0;
                                for f in 0..nf {
                                    let d = x[[s, f]] - means[[c, m, f]];
                                    norm2 += d * d;
                                }
                                c_n[IxDyn(&[c, m])] += w * norm2;
                            }
                            CovarianceType::Tied => {
                                for k in 0..nf {
                                    let dk = x[[s, k]] - means[[c, m, k]];
                                    for l in 0..nf {
                                        let dl = x[[s, l]] - means[[c, m, l]];
                                        c_n[IxDyn(&[c, k, l])] += w * dk * dl;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn do_mstep(&mut self, stats: &SufficientStatistics, params: &ParamFlags) {
        let nc = self.weights_.as_ref().unwrap().nrows();
        let nm = self.n_mix;
        let nf = self.n_features.unwrap();

        let pms = stats.get("post_mix_sum").unwrap();
        let ps = stats.get("post_sum").unwrap();

        // Update weights
        if params.contains('w') {
            let weights = self.weights_.as_mut().unwrap();
            let alpha_m1 = self.weights_prior - 1.0;
            for c in 0..nc {
                let denom = ps[IxDyn(&[c])] + alpha_m1 * nm as f64;
                for m in 0..nm {
                    weights[[c, m]] = (pms[IxDyn(&[c, m])] + alpha_m1) / denom;
                }
            }
        }

        // Update means
        if params.contains('m') {
            let m_n = stats.get("m_n").unwrap();
            let means = self.means_.as_mut().unwrap();
            for c in 0..nc {
                for m in 0..nm {
                    let denom = pms[IxDyn(&[c, m])] + self.means_weight;
                    let denom = if denom == 0.0 { 1.0 } else { denom };
                    for f in 0..nf {
                        means[[c, m, f]] =
                            (self.means_weight * self.means_prior + m_n[IxDyn(&[c, m, f])]) / denom;
                    }
                }
            }
        }

        // Update covariances
        if params.contains('c') {
            let c_n = stats.get("c_n").unwrap();

            match self.covariance_type {
                CovarianceType::Diag => {
                    let covars = self.covars_diag_.as_mut().unwrap();
                    for c in 0..nc {
                        for m in 0..nm {
                            let denom = pms[IxDyn(&[c, m])] + 1.0;
                            for f in 0..nf {
                                covars[[c, m, f]] =
                                    (self.covars_prior + c_n[IxDyn(&[c, m, f])]) / denom;
                            }
                        }
                    }
                }
                CovarianceType::Spherical => {
                    let covars = self.covars_spherical_.as_mut().unwrap();
                    for c in 0..nc {
                        for m in 0..nm {
                            let denom = nf as f64 * (pms[IxDyn(&[c, m])] + 1.0);
                            covars[[c, m]] = (self.covars_prior + c_n[IxDyn(&[c, m])]) / denom;
                        }
                    }
                }
                CovarianceType::Full => {
                    let covars = self.covars_full_.as_mut().unwrap();
                    for c in 0..nc {
                        for m in 0..nm {
                            let denom = pms[IxDyn(&[c, m])] + 1.0 + nf as f64 + 1.0;
                            for k in 0..nf {
                                for l in 0..nf {
                                    covars[[c, m, k, l]] =
                                        (self.covars_prior + c_n[IxDyn(&[c, m, k, l])]) / denom;
                                }
                            }
                        }
                    }
                }
                CovarianceType::Tied => {
                    let covars = self.covars_tied_.as_mut().unwrap();
                    for c in 0..nc {
                        let denom = ps[IxDyn(&[c])] + nm as f64 + nf as f64 + 1.0;
                        for k in 0..nf {
                            for l in 0..nf {
                                covars[[c, k, l]] =
                                    (self.covars_prior + c_n[IxDyn(&[c, k, l])]) / denom;
                            }
                        }
                    }
                }
            }
        }
    }

    fn n_fit_scalars_per_param(&self, n_components: usize) -> HashMap<char, usize> {
        let nc = n_components;
        let nf = self.n_features.unwrap_or(0);
        let nm = self.n_mix;
        let mut m = HashMap::new();
        m.insert('m', nc * nm * nf);
        m.insert(
            'c',
            match self.covariance_type {
                CovarianceType::Spherical => nc * nm,
                CovarianceType::Diag => nc * nm * nf,
                CovarianceType::Full => nc * nm * nf * (nf + 1) / 2,
                CovarianceType::Tied => nc * nf * (nf + 1) / 2,
            },
        );
        m.insert('w', nm - 1); // weights: shared constraint, not per-component
        m
    }
}

impl GmmEmissions {
    fn has_covars(&self) -> bool {
        match self.covariance_type {
            CovarianceType::Full => self.covars_full_.is_some(),
            CovarianceType::Tied => self.covars_tied_.is_some(),
            CovarianceType::Diag => self.covars_diag_.is_some(),
            CovarianceType::Spherical => self.covars_spherical_.is_some(),
        }
    }

    fn get_full_cov_for_mix(&self, state: usize, mix: usize) -> Array2<f64> {
        let nf = self.n_features.unwrap();
        match self.covariance_type {
            CovarianceType::Full => {
                let c = self.covars_full_.as_ref().unwrap();
                let mut cov = Array2::<f64>::zeros((nf, nf));
                for i in 0..nf {
                    for j in 0..nf {
                        cov[[i, j]] = c[[state, mix, i, j]];
                    }
                }
                cov
            }
            CovarianceType::Tied => self
                .covars_tied_
                .as_ref()
                .unwrap()
                .index_axis(Axis(0), state)
                .to_owned(),
            CovarianceType::Diag => {
                let c = self.covars_diag_.as_ref().unwrap();
                let mut cov = Array2::<f64>::zeros((nf, nf));
                for f in 0..nf {
                    cov[[f, f]] = c[[state, mix, f]];
                }
                cov
            }
            CovarianceType::Spherical => {
                let v = self.covars_spherical_.as_ref().unwrap()[[state, mix]];
                let mut cov = Array2::<f64>::zeros((nf, nf));
                for f in 0..nf {
                    cov[[f, f]] = v;
                }
                cov
            }
        }
    }
}

pub type GmmHmm = crate::base::BaseHmm<GmmEmissions>;

impl GmmHmm {
    pub fn gmm(n_components: usize, n_mix: usize, covariance_type: CovarianceType) -> Self {
        Self::new(n_components, GmmEmissions::new(n_mix, covariance_type))
            .with_params("stmcw")
            .with_init_params("stmcw")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gmm_diag_fit_score() {
        let x = array![
            [0.1, 0.2],
            [-0.1, 0.3],
            [0.2, -0.1],
            [0.0, 0.1],
            [0.3, 0.0],
            [5.1, 5.2],
            [4.9, 5.3],
            [5.2, 4.9],
            [5.0, 5.1],
            [5.3, 5.0],
        ];
        let lengths = [x.nrows()];

        let mut model = GmmHmm::gmm(2, 2, CovarianceType::Diag)
            .with_n_iter(20)
            .with_tol(1e-4);

        model.fit(&x, &lengths).unwrap();

        let score = model.score(&x, &lengths).unwrap();
        assert!(score.is_finite(), "score not finite: {}", score);

        let (_, states) = model.decode(&x, &lengths, None).unwrap();
        assert_eq!(states.len(), x.nrows());
    }

    #[test]
    fn test_gmm_single_mix() {
        // n_mix=1 should behave like GaussianHMM
        let x = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [4.9, 5.1],];
        let lengths = [4];

        let mut model = GmmHmm::gmm(2, 1, CovarianceType::Diag)
            .with_n_iter(10)
            .with_tol(1e-4);

        model.fit(&x, &lengths).unwrap();
        let score = model.score(&x, &lengths).unwrap();
        assert!(score.is_finite());
    }
}
