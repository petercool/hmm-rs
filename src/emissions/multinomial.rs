//! Multinomial emission model.
//!
//! Port of hmmlearn's MultinomialHMM emission logic.
//! Unlike CategoricalHMM (single symbol per timestep), MultinomialHMM
//! models counts of outcomes across n_trials.

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use rand::Rng;
use statrs::function::factorial::ln_factorial;

use crate::base::{EmissionModel, ParamFlags, SufficientStatistics};
use crate::error::{HmmError, Result};
use crate::utils;

/// Multinomial emission model.
///
/// Observations are count vectors: (n_samples, n_features) where each row sums to n_trials.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultinomialEmissions {
    pub n_features: Option<usize>,
    pub n_trials: Option<usize>,
    /// Emission probability matrix (n_components, n_features).
    pub emissionprob_: Option<Array2<f64>>,
    pub emissionprob_prior: f64,
}

impl Default for MultinomialEmissions {
    fn default() -> Self {
        Self::new()
    }
}

impl MultinomialEmissions {
    pub fn new() -> Self {
        Self {
            n_features: None,
            n_trials: None,
            emissionprob_: None,
            emissionprob_prior: 1.0,
        }
    }
}

/// Compute multinomial log PMF: log P(x | n, p)
///
/// Uses xlogy semantics: 0 * ln(0) = 0, k * ln(0) = -inf for k > 0.
fn multinomial_logpmf(x: &[f64], p: &[f64]) -> f64 {
    let n: f64 = x.iter().sum();
    let n_u = n as u64;
    let mut log_prob = ln_factorial(n_u);
    for i in 0..x.len() {
        let k = x[i];
        let k_u = k as u64;
        // xlogy: 0 * ln(0) = 0, avoids NaN from 0.0 * (-inf)
        let term = if k == 0.0 {
            0.0
        } else if p[i] <= 0.0 {
            f64::NEG_INFINITY
        } else {
            k * p[i].ln()
        };
        log_prob += term - ln_factorial(k_u);
    }
    log_prob
}

impl EmissionModel for MultinomialEmissions {
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
        rng: &mut impl Rng,
    ) {
        let nf = x.ncols();
        self.n_features = Some(nf);

        if self.n_trials.is_none() {
            // Infer n_trials from data (assume consistent)
            self.n_trials = Some(x.row(0).sum() as usize);
        }

        if params.contains('e') || self.emissionprob_.is_none() {
            let mut emissionprob = Array2::<f64>::zeros((n_components, nf));
            for i in 0..n_components {
                for j in 0..nf {
                    emissionprob[[i, j]] = rng.random::<f64>();
                }
            }
            utils::normalize_rows(&mut emissionprob);
            self.emissionprob_ = Some(emissionprob);
        }
    }

    fn check(&self, n_components: usize) -> Result<()> {
        let ep = self
            .emissionprob_
            .as_ref()
            .ok_or_else(|| HmmError::NotFitted("emissionprob_ not set".into()))?;
        let nf = self.n_features.unwrap_or(ep.ncols());
        if ep.shape() != [n_components, nf] {
            return Err(HmmError::InvalidParameter(
                "emissionprob_ shape mismatch".into(),
            ));
        }
        Ok(())
    }

    fn compute_log_likelihood(&self, x: &Array2<f64>) -> Array2<f64> {
        let ep = self.emissionprob_.as_ref().unwrap();
        let nc = ep.nrows();
        let ns = x.nrows();
        let nf = x.ncols();

        let mut result = Array2::<f64>::zeros((ns, nc));
        for c in 0..nc {
            let p: Vec<f64> = (0..nf).map(|f| ep[[c, f]]).collect();
            for s in 0..ns {
                let obs: Vec<f64> = (0..nf).map(|f| x[[s, f]]).collect();
                result[[s, c]] = multinomial_logpmf(&obs, &p);
            }
        }
        result
    }

    fn generate_sample_from_state(&self, state: usize, rng: &mut impl Rng) -> Array1<f64> {
        let ep = self.emissionprob_.as_ref().unwrap();
        let nf = ep.ncols();
        let n_trials = self.n_trials.unwrap_or(1);

        // Sample multinomial: n_trials draws from categorical
        let mut counts = Array1::<f64>::zeros(nf);
        for _ in 0..n_trials {
            let u: f64 = rng.random();
            let mut cumsum = 0.0;
            for j in 0..nf {
                cumsum += ep[[state, j]];
                if cumsum > u {
                    counts[j] += 1.0;
                    break;
                }
            }
        }
        counts
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
            let nc = posteriors.ncols();
            let ns = x.nrows();
            let nf = x.ncols();

            // obs += posteriors.T @ X
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
    }

    fn do_mstep(&mut self, stats: &SufficientStatistics, params: &ParamFlags) {
        if params.contains('e') {
            let obs = stats.get("obs").unwrap();
            let ep = self.emissionprob_.as_mut().unwrap();
            let nc = ep.nrows();
            let nf = ep.ncols();

            for i in 0..nc {
                for j in 0..nf {
                    ep[[i, j]] = (self.emissionprob_prior - 1.0 + obs[IxDyn(&[i, j])]).max(0.0);
                }
            }
            utils::normalize_rows(ep);
        }
    }

    fn n_fit_scalars_per_param(&self, n_components: usize) -> HashMap<char, usize> {
        let nf = self.n_features.unwrap_or(0);
        let mut m = HashMap::new();
        m.insert('e', n_components * (nf - 1));
        m
    }
}

pub type MultinomialHmm = crate::base::BaseHmm<MultinomialEmissions>;

impl MultinomialHmm {
    pub fn multinomial(n_components: usize) -> Self {
        Self::new(n_components, MultinomialEmissions::new())
            .with_params("ste")
            .with_init_params("ste")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_multinomial_fit_and_score() {
        let mut model = MultinomialHmm::multinomial(2)
            .with_n_iter(20)
            .with_tol(1e-4);

        // Each row: count vector summing to 5 (n_trials=5), 3 categories
        let x = array![
            [3.0, 1.0, 1.0],
            [2.0, 2.0, 1.0],
            [4.0, 0.0, 1.0],
            [1.0, 1.0, 3.0],
            [0.0, 2.0, 3.0],
            [1.0, 3.0, 1.0],
            [2.0, 1.0, 2.0],
            [3.0, 2.0, 0.0],
        ];
        let lengths = [8];

        model.emission.n_trials = Some(5);
        model.fit(&x, &lengths).unwrap();

        let score = model.score(&x, &lengths).unwrap();
        assert!(score.is_finite());
    }
}
