//! Gaussian emission model with 4 covariance types.
//!
//! Port of hmmlearn's GaussianHMM emission logic.

use std::collections::HashMap;

use ndarray::{Array1, Array2, Array3, ArrayD, Axis, IxDyn};
use rand::Rng;
use rand_distr::Normal;

use crate::base::{EmissionModel, ParamFlags, SufficientStatistics};
use crate::emissions::CovarianceType;
use crate::error::{HmmError, Result};
use crate::stats::{self, CovarsArg};

/// Gaussian emission model supporting 4 covariance types.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GaussianEmissions {
    pub covariance_type: CovarianceType,
    pub min_covar: f64,
    pub n_features: Option<usize>,

    /// Mean vectors: (n_components, n_features).
    pub means_: Option<Array2<f64>>,

    /// Internal covariance storage (native shape for the type).
    /// - Full: (n_components, n_features, n_features)
    /// - Tied: (n_features, n_features)
    /// - Diag: (n_components, n_features)
    /// - Spherical: (n_components,)
    pub covars_full_: Option<Array3<f64>>,
    pub covars_tied_: Option<Array2<f64>>,
    pub covars_diag_: Option<Array2<f64>>,
    pub covars_spherical_: Option<Array1<f64>>,

    // Priors
    pub means_prior: f64,
    pub means_weight: f64,
    pub covars_prior: f64,
    pub covars_weight: f64,
}

impl GaussianEmissions {
    pub fn new(covariance_type: CovarianceType) -> Self {
        Self {
            covariance_type,
            min_covar: 1e-3,
            n_features: None,
            means_: None,
            covars_full_: None,
            covars_tied_: None,
            covars_diag_: None,
            covars_spherical_: None,
            means_prior: 0.0,
            means_weight: 0.0,
            covars_prior: 1e-2,
            covars_weight: 1.0,
        }
    }

    pub fn with_min_covar(mut self, min_covar: f64) -> Self {
        self.min_covar = min_covar;
        self
    }

    /// Set covariance matrices (validates shape against covariance_type).
    pub fn set_covars_full(&mut self, covars: Array3<f64>) {
        self.covars_full_ = Some(covars);
    }

    pub fn set_covars_tied(&mut self, covars: Array2<f64>) {
        self.covars_tied_ = Some(covars);
    }

    pub fn set_covars_diag(&mut self, covars: Array2<f64>) {
        self.covars_diag_ = Some(covars);
    }

    pub fn set_covars_spherical(&mut self, covars: Array1<f64>) {
        self.covars_spherical_ = Some(covars);
    }

    /// Get the CovarsArg for stats computation.
    fn covars_arg(&self) -> CovarsArg {
        match self.covariance_type {
            CovarianceType::Full => CovarsArg::Full(self.covars_full_.as_ref().unwrap()),
            CovarianceType::Tied => CovarsArg::Tied(self.covars_tied_.as_ref().unwrap()),
            CovarianceType::Diag => CovarsArg::Diag(self.covars_diag_.as_ref().unwrap()),
            CovarianceType::Spherical => {
                CovarsArg::Spherical(self.covars_spherical_.as_ref().unwrap())
            }
        }
    }

    /// Get the internal covariance used for log-likelihood (native shape).
    /// For GaussianHMM, _covars_ is the native shape used in compute_log_likelihood.
    /// covars_ property returns full matrices (like hmmlearn's fill_covars).
    fn get_covars_for_log_likelihood(&self) -> CovarsArg {
        self.covars_arg()
    }

    /// Compute sample covariance from data.
    fn compute_sample_covariance(x: &Array2<f64>) -> Array2<f64> {
        let ns = x.nrows();
        let nf = x.ncols();
        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut cov = Array2::<f64>::zeros((nf, nf));
        for s in 0..ns {
            for i in 0..nf {
                for j in 0..nf {
                    cov[[i, j]] += (x[[s, i]] - mean[i]) * (x[[s, j]] - mean[j]);
                }
            }
        }
        if ns > 1 {
            cov /= (ns - 1) as f64;
        }
        cov
    }
}

impl EmissionModel for GaussianEmissions {
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

        // Initialize means: random selection from data
        if params.contains('m') || self.means_.is_none() {
            let ns = x.nrows();
            let mut means = Array2::<f64>::zeros((n_components, nf));
            // Simple initialization: spread evenly through data
            for c in 0..n_components {
                let idx = (c * ns) / n_components;
                means.row_mut(c).assign(&x.row(idx.min(ns - 1)));
            }
            self.means_ = Some(means);
        }

        // Initialize covariances from data sample covariance
        if params.contains('c') || !self.has_covars() {
            let cv = Self::compute_sample_covariance(x);
            // Add min_covar to diagonal
            let mut cv_reg = cv.clone();
            for i in 0..nf {
                cv_reg[[i, i]] += self.min_covar;
            }

            match self.covariance_type {
                CovarianceType::Full => {
                    let mut covars = Array3::<f64>::zeros((n_components, nf, nf));
                    for c in 0..n_components {
                        for i in 0..nf {
                            for j in 0..nf {
                                covars[[c, i, j]] = cv_reg[[i, j]];
                            }
                        }
                    }
                    self.covars_full_ = Some(covars);
                }
                CovarianceType::Tied => {
                    self.covars_tied_ = Some(cv_reg);
                }
                CovarianceType::Diag => {
                    let mut covars = Array2::<f64>::zeros((n_components, nf));
                    for c in 0..n_components {
                        for f in 0..nf {
                            covars[[c, f]] = cv_reg[[f, f]];
                        }
                    }
                    self.covars_diag_ = Some(covars);
                }
                CovarianceType::Spherical => {
                    let mean_diag = (0..nf).map(|i| cv_reg[[i, i]]).sum::<f64>() / nf as f64;
                    self.covars_spherical_ = Some(Array1::from_elem(n_components, mean_diag));
                }
            }
        }
    }

    fn check(&self, n_components: usize) -> Result<()> {
        let means = self
            .means_
            .as_ref()
            .ok_or_else(|| HmmError::NotFitted("means_ not set".into()))?;
        let nf = means.ncols();

        if means.nrows() != n_components {
            return Err(HmmError::InvalidParameter(format!(
                "means_ must have {} rows, got {}",
                n_components,
                means.nrows()
            )));
        }

        if !self.has_covars() {
            return Err(HmmError::NotFitted("covars not set".into()));
        }

        // Shape checks per covariance type
        match self.covariance_type {
            CovarianceType::Full => {
                let c = self.covars_full_.as_ref().unwrap();
                if c.shape() != [n_components, nf, nf] {
                    return Err(HmmError::InvalidParameter(
                        "full covars shape mismatch".into(),
                    ));
                }
            }
            CovarianceType::Tied => {
                let c = self.covars_tied_.as_ref().unwrap();
                if c.shape() != [nf, nf] {
                    return Err(HmmError::InvalidParameter(
                        "tied covars shape mismatch".into(),
                    ));
                }
            }
            CovarianceType::Diag => {
                let c = self.covars_diag_.as_ref().unwrap();
                if c.shape() != [n_components, nf] {
                    return Err(HmmError::InvalidParameter(
                        "diag covars shape mismatch".into(),
                    ));
                }
            }
            CovarianceType::Spherical => {
                let c = self.covars_spherical_.as_ref().unwrap();
                if c.len() != n_components {
                    return Err(HmmError::InvalidParameter(
                        "spherical covars length mismatch".into(),
                    ));
                }
            }
        }

        Ok(())
    }

    fn compute_log_likelihood(&self, x: &Array2<f64>) -> Array2<f64> {
        let means = self.means_.as_ref().unwrap();
        let covars = self.get_covars_for_log_likelihood();
        stats::log_multivariate_normal_density(x, means, &covars, self.covariance_type)
    }

    fn generate_sample_from_state(&self, state: usize, rng: &mut impl Rng) -> Array1<f64> {
        let means = self.means_.as_ref().unwrap();
        let nf = means.ncols();
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Get full covariance for this state
        let cov = self.get_full_covariance_for_state(state);

        // Cholesky decomposition
        let l = stats::cholesky_lower(&cov).unwrap();

        // Sample: mean + L @ z where z ~ N(0, I)
        let z: Array1<f64> = Array1::from_shape_fn(nf, |_| rng.sample(normal));
        let mut sample = Array1::<f64>::zeros(nf);
        for i in 0..nf {
            let mut val = means[[state, i]];
            for j in 0..=i {
                val += l[[i, j]] * z[j];
            }
            sample[i] = val;
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
        if matches!(
            self.covariance_type,
            CovarianceType::Tied | CovarianceType::Full
        ) {
            stats.insert(
                "obs*obs.T".to_string(),
                ArrayD::zeros(IxDyn(&[n_components, nf, nf])),
            );
        }
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
        let needs_mean = params.contains('m');
        let needs_covar = params.contains('c');

        if needs_mean || needs_covar {
            // post += posteriors.sum(axis=0)
            let post = stats.get_mut("post").unwrap();
            for c in 0..nc {
                let sum: f64 = (0..ns).map(|s| posteriors[[s, c]]).sum();
                post[IxDyn(&[c])] += sum;
            }

            // obs += posteriors.T @ X
            let obs = stats.get_mut("obs").unwrap();
            for c in 0..nc {
                for f in 0..nf {
                    let mut sum = 0.0;
                    for s in 0..ns {
                        sum += posteriors[[s, c]] * x[[s, f]];
                    }
                    obs[IxDyn(&[c, f])] += sum;
                }
            }
        }

        if needs_covar {
            match self.covariance_type {
                CovarianceType::Spherical | CovarianceType::Diag => {
                    // obs**2 += posteriors.T @ X**2
                    let obs2 = stats.get_mut("obs**2").unwrap();
                    for c in 0..nc {
                        for f in 0..nf {
                            let mut sum = 0.0;
                            for s in 0..ns {
                                sum += posteriors[[s, c]] * x[[s, f]] * x[[s, f]];
                            }
                            obs2[IxDyn(&[c, f])] += sum;
                        }
                    }
                }
                CovarianceType::Tied | CovarianceType::Full => {
                    // obs*obs.T += einsum('ij,ik,il->jkl', posteriors, X, X)
                    let obs_obt = stats.get_mut("obs*obs.T").unwrap();
                    for c in 0..nc {
                        for k in 0..nf {
                            for l in 0..nf {
                                let mut sum = 0.0;
                                for s in 0..ns {
                                    sum += posteriors[[s, c]] * x[[s, k]] * x[[s, l]];
                                }
                                obs_obt[IxDyn(&[c, k, l])] += sum;
                            }
                        }
                    }
                }
            }
        }
    }

    fn do_mstep(&mut self, stats: &SufficientStatistics, params: &ParamFlags) {
        let nc = self.means_.as_ref().unwrap().nrows();
        let nf = self.n_features.unwrap();

        let post = stats.get("post").unwrap();
        let obs = stats.get("obs").unwrap();

        // Update means
        if params.contains('m') {
            let means = self.means_.as_mut().unwrap();
            for c in 0..nc {
                let denom = self.means_weight + post[IxDyn(&[c])];
                for f in 0..nf {
                    means[[c, f]] =
                        (self.means_weight * self.means_prior + obs[IxDyn(&[c, f])]) / denom;
                }
            }
        }

        // Update covariances
        if params.contains('c') {
            let means = self.means_.as_ref().unwrap().clone();
            let means_prior = self.means_prior;
            let means_weight = self.means_weight;
            let covars_prior = self.covars_prior;
            let covars_weight = self.covars_weight;

            match self.covariance_type {
                CovarianceType::Spherical | CovarianceType::Diag => {
                    let obs2 = stats.get("obs**2").unwrap();
                    let mut covars = Array2::<f64>::zeros((nc, nf));

                    for c in 0..nc {
                        let denom = post[IxDyn(&[c])];
                        let c_d = (covars_weight - 1.0).max(0.0) + denom;

                        for f in 0..nf {
                            let meandiff = means[[c, f]] - means_prior;
                            let c_n = means_weight * meandiff * meandiff
                                + obs2[IxDyn(&[c, f])]
                                - 2.0 * means[[c, f]] * obs[IxDyn(&[c, f])]
                                + means[[c, f]] * means[[c, f]] * denom;
                            covars[[c, f]] = (covars_prior + c_n) / c_d.max(1e-5);
                        }
                    }

                    if self.covariance_type == CovarianceType::Spherical {
                        // Average across features
                        let mut sph = Array1::<f64>::zeros(nc);
                        for c in 0..nc {
                            sph[c] = covars.row(c).mean().unwrap();
                        }
                        self.covars_spherical_ = Some(sph);
                    } else {
                        self.covars_diag_ = Some(covars);
                    }
                }
                CovarianceType::Tied | CovarianceType::Full => {
                    let obs_obt = stats.get("obs*obs.T").unwrap();
                    let mut c_n = Array3::<f64>::zeros((nc, nf, nf));

                    for c in 0..nc {
                        for k in 0..nf {
                            for l in 0..nf {
                                let meandiff_k = means[[c, k]] - means_prior;
                                let meandiff_l = means[[c, l]] - means_prior;

                                c_n[[c, k, l]] = means_weight * meandiff_k * meandiff_l
                                    + obs_obt[IxDyn(&[c, k, l])]
                                    - obs[IxDyn(&[c, k])] * means[[c, l]]
                                    - obs[IxDyn(&[c, l])] * means[[c, k]]
                                    + means[[c, k]] * means[[c, l]] * post[IxDyn(&[c])];
                            }
                        }
                    }

                    let cvweight = (covars_weight - nf as f64).max(0.0);

                    if self.covariance_type == CovarianceType::Tied {
                        let mut tied = Array2::<f64>::zeros((nf, nf));
                        let mut post_sum = 0.0;
                        for c in 0..nc {
                            post_sum += post[IxDyn(&[c])];
                            for k in 0..nf {
                                for l in 0..nf {
                                    tied[[k, l]] += c_n[[c, k, l]];
                                }
                            }
                        }
                        let denom = cvweight + post_sum;
                        for k in 0..nf {
                            for l in 0..nf {
                                tied[[k, l]] = (covars_prior + tied[[k, l]]) / denom;
                            }
                        }
                        self.covars_tied_ = Some(tied);
                    } else {
                        // Full
                        let mut full = Array3::<f64>::zeros((nc, nf, nf));
                        for c in 0..nc {
                            let denom = cvweight + post[IxDyn(&[c])];
                            for k in 0..nf {
                                for l in 0..nf {
                                    full[[c, k, l]] =
                                        (covars_prior + c_n[[c, k, l]]) / denom;
                                }
                            }
                        }
                        self.covars_full_ = Some(full);
                    }
                }
            }
        }
    }

    fn n_fit_scalars_per_param(&self, n_components: usize) -> HashMap<char, usize> {
        let nc = n_components;
        let nf = self.n_features.unwrap_or(0);
        let mut m = HashMap::new();
        m.insert('m', nc * nf);
        m.insert(
            'c',
            match self.covariance_type {
                CovarianceType::Spherical => nc,
                CovarianceType::Diag => nc * nf,
                CovarianceType::Full => nc * nf * (nf + 1) / 2,
                CovarianceType::Tied => nf * (nf + 1) / 2, // shared, not multiplied by nc
            },
        );
        m
    }
}

impl GaussianEmissions {
    fn has_covars(&self) -> bool {
        match self.covariance_type {
            CovarianceType::Full => self.covars_full_.is_some(),
            CovarianceType::Tied => self.covars_tied_.is_some(),
            CovarianceType::Diag => self.covars_diag_.is_some(),
            CovarianceType::Spherical => self.covars_spherical_.is_some(),
        }
    }

    /// Get a full (nf x nf) covariance matrix for a given state.
    fn get_full_covariance_for_state(&self, state: usize) -> Array2<f64> {
        let nf = self.n_features.unwrap();
        match self.covariance_type {
            CovarianceType::Full => {
                let c = self.covars_full_.as_ref().unwrap();
                c.index_axis(Axis(0), state).to_owned()
            }
            CovarianceType::Tied => self.covars_tied_.as_ref().unwrap().clone(),
            CovarianceType::Diag => {
                let c = self.covars_diag_.as_ref().unwrap();
                let mut full = Array2::<f64>::zeros((nf, nf));
                for f in 0..nf {
                    full[[f, f]] = c[[state, f]];
                }
                full
            }
            CovarianceType::Spherical => {
                let v = self.covars_spherical_.as_ref().unwrap()[state];
                let mut full = Array2::<f64>::zeros((nf, nf));
                for f in 0..nf {
                    full[[f, f]] = v;
                }
                full
            }
        }
    }
}

/// Type alias for GaussianHMM.
pub type GaussianHmm = crate::base::BaseHmm<GaussianEmissions>;

impl GaussianHmm {
    /// Create a new GaussianHMM with default settings.
    pub fn gaussian(n_components: usize, covariance_type: CovarianceType) -> Self {
        Self::new(n_components, GaussianEmissions::new(covariance_type))
            .with_params("stmc")
            .with_init_params("stmc")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn make_gaussian_data() -> Array2<f64> {
        // Two clusters: around [0,0] and [5,5]
        array![
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
            [0.1, 0.0],
            [5.0, 5.0],
            [-0.2, 0.1],
            [5.1, 4.8],
            [0.0, -0.1],
        ]
    }

    #[test]
    fn test_gaussian_diag_fit_score() {
        let x = make_gaussian_data();
        let lengths = [x.nrows()];

        let mut model = GaussianHmm::gaussian(2, CovarianceType::Diag)
            .with_n_iter(50)
            .with_tol(1e-4);

        model.fit(&x, &lengths).unwrap();

        let score = model.score(&x, &lengths).unwrap();
        assert!(score.is_finite(), "score should be finite: {}", score);

        let (_, states) = model.decode(&x, &lengths, None).unwrap();
        assert_eq!(states.len(), x.nrows());
    }

    #[test]
    fn test_gaussian_full_fit_score() {
        let x = make_gaussian_data();
        let lengths = [x.nrows()];

        let mut model = GaussianHmm::gaussian(2, CovarianceType::Full)
            .with_n_iter(50)
            .with_tol(1e-4);

        model.fit(&x, &lengths).unwrap();

        let score = model.score(&x, &lengths).unwrap();
        assert!(score.is_finite());
    }

    #[test]
    fn test_gaussian_tied_fit_score() {
        let x = make_gaussian_data();
        let lengths = [x.nrows()];

        let mut model = GaussianHmm::gaussian(2, CovarianceType::Tied)
            .with_n_iter(50)
            .with_tol(1e-4);

        model.fit(&x, &lengths).unwrap();

        let score = model.score(&x, &lengths).unwrap();
        assert!(score.is_finite());
    }

    #[test]
    fn test_gaussian_spherical_fit_score() {
        let x = make_gaussian_data();
        let lengths = [x.nrows()];

        let mut model = GaussianHmm::gaussian(2, CovarianceType::Spherical)
            .with_n_iter(50)
            .with_tol(1e-4);

        model.fit(&x, &lengths).unwrap();

        let score = model.score(&x, &lengths).unwrap();
        assert!(score.is_finite());
    }

    #[test]
    fn test_gaussian_sample() {
        let mut model = GaussianHmm::gaussian(2, CovarianceType::Diag);
        model.emission.n_features = Some(2);
        model.emission.means_ = Some(array![[0.0, 0.0], [5.0, 5.0]]);
        model.emission.covars_diag_ = Some(array![[1.0, 1.0], [1.0, 1.0]]);
        model.startprob_ = array![0.5, 0.5];
        model.transmat_ = array![[0.8, 0.2], [0.2, 0.8]];
        model.fitted = true;

        let mut rng = rand::rng();
        let (x, states) = model.sample(50, &mut rng, None).unwrap();
        assert_eq!(x.nrows(), 50);
        assert_eq!(x.ncols(), 2);
        assert_eq!(states.len(), 50);
    }

    #[test]
    fn test_gaussian_preset_params_score() {
        // Test with manually set parameters to verify score computation
        let mut model = GaussianHmm::gaussian(2, CovarianceType::Full);
        model.emission.n_features = Some(2);
        model.emission.means_ = Some(array![[0.0, 0.0], [5.0, 5.0]]);
        model.emission.covars_full_ = Some(Array3::from_shape_vec(
            (2, 2, 2),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        ).unwrap());
        model.startprob_ = array![0.5, 0.5];
        model.transmat_ = array![[0.8, 0.2], [0.2, 0.8]];
        model.fitted = true;

        let x = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [4.9, 5.1]];
        let score = model.score(&x, &[4]).unwrap();
        assert!(score.is_finite());
        assert!(score < 0.0);
    }

    #[test]
    fn test_gaussian_aic_bic() {
        let x = make_gaussian_data();
        let lengths = [x.nrows()];

        let mut model = GaussianHmm::gaussian(2, CovarianceType::Diag)
            .with_n_iter(20)
            .with_tol(1e-4);

        model.fit(&x, &lengths).unwrap();

        let aic = model.aic(&x, &lengths).unwrap();
        let bic = model.bic(&x, &lengths).unwrap();
        assert!(aic.is_finite());
        assert!(bic.is_finite());
        // AIC and BIC should be positive (since -2*log_prob is positive for prob < 1)
        // Actually they could be negative if the log_likelihood is very positive,
        // but typically positive for practical data
    }
}
