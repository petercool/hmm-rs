//! Variational Categorical emission model.

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use rand::Rng;
use statrs::function::gamma::digamma;

use crate::base::{EmissionModel, ParamFlags, SufficientStatistics, sample_dirichlet};
use crate::error::Result;
use crate::kl_divergence;
use crate::utils;
use crate::vhmm::VariationalEmissionModel;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VariationalCategoricalEmissions {
    pub n_features: Option<usize>,
    pub emissionprob_prior_: Option<Array2<f64>>,
    pub emissionprob_posterior_: Option<Array2<f64>>,
    emissionprob_log_subnorm_: Option<Array2<f64>>,
    pub emissionprob_prior: Option<f64>,
}

impl Default for VariationalCategoricalEmissions {
    fn default() -> Self {
        Self::new()
    }
}

impl VariationalCategoricalEmissions {
    pub fn new() -> Self {
        Self {
            n_features: None,
            emissionprob_prior_: None,
            emissionprob_posterior_: None,
            emissionprob_log_subnorm_: None,
            emissionprob_prior: None,
        }
    }

    pub fn with_n_features(mut self, n: usize) -> Self {
        self.n_features = Some(n);
        self
    }
}

impl EmissionModel for VariationalCategoricalEmissions {
    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn set_n_features(&mut self, _n: usize) {}

    fn init(
        &mut self,
        x: &Array2<f64>,
        _n_components: usize,
        _params: &ParamFlags,
        _rng: &mut impl Rng,
    ) {
        if self.n_features.is_none() {
            self.n_features = Some(x.iter().cloned().fold(0.0_f64, f64::max) as usize + 1);
        }
    }

    fn check(&self, _n_components: usize) -> Result<()> {
        Ok(())
    }

    fn compute_log_likelihood(&self, x: &Array2<f64>) -> Array2<f64> {
        // Use the normalized posterior as point estimate
        let post = self.emissionprob_posterior_.as_ref().unwrap();
        let nc = post.nrows();
        let ns = x.nrows();
        let mut ep = post.clone();
        utils::normalize_rows(&mut ep);
        let mut result = Array2::<f64>::zeros((ns, nc));
        for s in 0..ns {
            let symbol = x[[s, 0]] as usize;
            for c in 0..nc {
                result[[s, c]] = ep[[c, symbol]].max(1e-300).ln();
            }
        }
        result
    }

    fn generate_sample_from_state(&self, state: usize, rng: &mut impl Rng) -> Array1<f64> {
        let post = self.emissionprob_posterior_.as_ref().unwrap();
        let nf = post.ncols();
        let row_sum = post.row(state).sum();
        let u: f64 = rng.random::<f64>() * row_sum;
        let mut cumsum = 0.0;
        for j in 0..nf {
            cumsum += post[[state, j]];
            if cumsum > u {
                return Array1::from_vec(vec![j as f64]);
            }
        }
        Array1::from_vec(vec![(nf - 1) as f64])
    }

    fn initialize_sufficient_statistics(&self, n_components: usize) -> SufficientStatistics {
        let nf = self.n_features.unwrap();
        let mut stats = SufficientStatistics::new();
        stats.insert("obs".to_string(), ArrayD::zeros(IxDyn(&[n_components, nf])));
        stats
    }

    fn accumulate_sufficient_statistics(
        &self,
        stats: &mut SufficientStatistics,
        x: &Array2<f64>,
        posteriors: &Array2<f64>,
        params: &ParamFlags,
    ) {
        if params.contains('e') {
            let obs = stats.get_mut("obs").unwrap();
            let ns = x.nrows();
            let nc = posteriors.ncols();
            for s in 0..ns {
                let symbol = x[[s, 0]] as usize;
                for c in 0..nc {
                    obs[IxDyn(&[c, symbol])] += posteriors[[s, c]];
                }
            }
        }
    }

    fn do_mstep(&mut self, _stats: &SufficientStatistics, _params: &ParamFlags) {
        // For variational, use do_mstep_variational instead
    }

    fn n_fit_scalars_per_param(&self, n_components: usize) -> HashMap<char, usize> {
        let nf = self.n_features.unwrap_or(0);
        let mut m = HashMap::new();
        m.insert('e', n_components * (nf - 1));
        m
    }
}

impl VariationalEmissionModel for VariationalCategoricalEmissions {
    fn compute_subnorm_log_likelihood(&self, x: &Array2<f64>) -> Array2<f64> {
        let log_subnorm = self.emissionprob_log_subnorm_.as_ref().unwrap();
        let nc = log_subnorm.nrows();
        let ns = x.nrows();
        let mut result = Array2::<f64>::zeros((ns, nc));
        for s in 0..ns {
            let symbol = x[[s, 0]] as usize;
            for c in 0..nc {
                result[[s, c]] = log_subnorm[[c, symbol]];
            }
        }
        result
    }

    fn estep_begin(&mut self) {
        let post = self.emissionprob_posterior_.as_ref().unwrap();
        let nc = post.nrows();
        let nf = post.ncols();
        let mut log_subnorm = Array2::<f64>::zeros((nc, nf));
        for c in 0..nc {
            let row_sum = post.row(c).sum();
            let digamma_sum = digamma(row_sum);
            for f in 0..nf {
                log_subnorm[[c, f]] = digamma(post[[c, f]]) - digamma_sum;
            }
        }
        self.emissionprob_log_subnorm_ = Some(log_subnorm);
    }

    fn emission_kl_divergence(&self) -> f64 {
        let prior = self.emissionprob_prior_.as_ref().unwrap();
        let post = self.emissionprob_posterior_.as_ref().unwrap();
        let nc = prior.nrows();
        let mut kl = 0.0;
        for c in 0..nc {
            kl += kl_divergence::kl_dirichlet(&post.row(c).to_owned(), &prior.row(c).to_owned());
        }
        kl
    }

    fn init_variational(
        &mut self,
        x: &Array2<f64>,
        lengths: &[usize],
        n_components: usize,
        rng: &mut impl Rng,
    ) {
        if self.n_features.is_none() {
            self.n_features = Some(x.iter().cloned().fold(0.0_f64, f64::max) as usize + 1);
        }
        let nf = self.n_features.unwrap();
        let total: usize = lengths.iter().sum();

        let ep_init = self.emissionprob_prior.unwrap_or(1.0 / nf as f64);

        self.emissionprob_prior_ = Some(Array2::from_elem((n_components, nf), ep_init));
        // Initialize posterior via random Dirichlet
        let mut posterior = Array2::<f64>::zeros((n_components, nf));
        for c in 0..n_components {
            let row = sample_dirichlet(nf, ep_init, rng);
            posterior
                .row_mut(c)
                .assign(&(row * total as f64 / n_components as f64));
        }
        self.emissionprob_posterior_ = Some(posterior);
    }

    fn do_mstep_variational(&mut self, stats: &SufficientStatistics, params: &ParamFlags) {
        if params.contains('e') {
            let obs = stats.get("obs").unwrap();
            let prior = self.emissionprob_prior_.as_ref().unwrap();
            let post = self.emissionprob_posterior_.as_mut().unwrap();
            let nc = post.nrows();
            let nf = post.ncols();

            for c in 0..nc {
                for f in 0..nf {
                    post[[c, f]] = prior[[c, f]] + obs[IxDyn(&[c, f])];
                }
            }
        }
    }
}

pub type VariationalCategoricalHmm =
    crate::vhmm::VariationalBaseHmm<VariationalCategoricalEmissions>;

impl VariationalCategoricalHmm {
    pub fn variational_categorical(n_components: usize) -> Self {
        Self::new(n_components, VariationalCategoricalEmissions::new())
            .with_params("ste")
            .with_init_params("ste")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_variational_categorical_fit() {
        let mut model = VariationalCategoricalHmm::variational_categorical(2)
            .with_n_iter(50)
            .with_tol(1e-6);
        model.emission.n_features = Some(3);

        let x = array![
            [0.0],
            [1.0],
            [2.0],
            [0.0],
            [1.0],
            [0.0],
            [0.0],
            [1.0],
            [2.0],
            [2.0]
        ];
        let lengths = [10];

        model.fit(&x, &lengths).unwrap();

        // Should produce finite score
        let score = model.score(&x, &lengths).unwrap();
        assert!(score.is_finite(), "score = {}", score);

        // Should decode
        let (_, states) = model.decode(&x, &lengths).unwrap();
        assert_eq!(states.len(), 10);
    }

    #[test]
    fn test_variational_categorical_lower_bound_monotonic() {
        let mut model = VariationalCategoricalHmm::variational_categorical(2)
            .with_n_iter(30)
            .with_tol(1e-12); // very tight to force many iterations
        model.emission.n_features = Some(3);

        let x = array![
            [0.0],
            [1.0],
            [2.0],
            [0.0],
            [1.0],
            [0.0],
            [0.0],
            [1.0],
            [2.0],
            [2.0],
            [1.0],
            [0.0],
            [2.0],
            [1.0],
            [0.0],
        ];
        let lengths = [15];

        model.fit(&x, &lengths).unwrap();

        // Check that the lower bound is (approximately) non-decreasing
        let history: Vec<f64> = model.monitor_.history.iter().cloned().collect();
        for i in 1..history.len() {
            // Allow small numerical noise
            assert!(
                history[i] >= history[i - 1] - 1e-6,
                "lower bound decreased at iter {}: {} -> {}",
                i,
                history[i - 1],
                history[i]
            );
        }
    }
}
