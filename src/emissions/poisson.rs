//! Poisson emission model.
//!
//! Port of hmmlearn's PoissonHMM emission logic.

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use rand::Rng;
use rand_distr::Poisson;
use statrs::function::factorial::ln_factorial;

use crate::base::{EmissionModel, ParamFlags, SufficientStatistics};
use crate::error::{HmmError, Result};

/// Poisson emission model.
///
/// Each state emits counts from a Poisson distribution with rate lambda.
/// Supports multivariate Poisson (independent Poisson per feature).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PoissonEmissions {
    pub n_features: Option<usize>,
    /// Poisson rates: (n_components, n_features).
    pub lambdas_: Option<Array2<f64>>,
    pub lambdas_prior: f64,
    pub lambdas_weight: f64,
}

impl Default for PoissonEmissions {
    fn default() -> Self {
        Self::new()
    }
}

impl PoissonEmissions {
    pub fn new() -> Self {
        Self {
            n_features: None,
            lambdas_: None,
            lambdas_prior: 0.0,
            lambdas_weight: 0.0,
        }
    }
}

impl EmissionModel for PoissonEmissions {
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
        let nf = self.n_features.unwrap_or(x.ncols());
        self.n_features = Some(nf);

        if params.contains('l') || self.lambdas_.is_none() {
            // Initialize lambdas randomly around the data mean
            let col_means: Array1<f64> = x.mean_axis(ndarray::Axis(0)).unwrap();
            let mut lambdas = Array2::<f64>::zeros((n_components, nf));
            for c in 0..n_components {
                for f in 0..nf {
                    lambdas[[c, f]] = (col_means[f] * (0.5 + rng.random::<f64>())).max(0.1);
                }
            }
            self.lambdas_ = Some(lambdas);
        }
    }

    fn check(&self, n_components: usize) -> Result<()> {
        let lambdas = self
            .lambdas_
            .as_ref()
            .ok_or_else(|| HmmError::NotFitted("lambdas_ not set".into()))?;
        let nf = self.n_features.unwrap_or(lambdas.ncols());

        if lambdas.shape() != [n_components, nf] {
            return Err(HmmError::InvalidParameter(format!(
                "lambdas_ must have shape ({n_components}, {nf})"
            )));
        }

        Ok(())
    }

    fn compute_log_likelihood(&self, x: &Array2<f64>) -> Array2<f64> {
        let lambdas = self.lambdas_.as_ref().unwrap();
        let nc = lambdas.nrows();
        let nf = lambdas.ncols();
        let ns = x.nrows();

        let mut result = Array2::<f64>::zeros((ns, nc));

        for c in 0..nc {
            for s in 0..ns {
                let mut log_prob = 0.0;
                for f in 0..nf {
                    let k = x[[s, f]];
                    let lam = lambdas[[c, f]];
                    // Poisson log PMF: k * ln(lam) - lam - ln(k!)
                    log_prob += k * lam.ln() - lam - ln_factorial(k as u64);
                }
                result[[s, c]] = log_prob;
            }
        }

        result
    }

    fn generate_sample_from_state(&self, state: usize, rng: &mut impl Rng) -> Array1<f64> {
        let lambdas = self.lambdas_.as_ref().unwrap();
        let nf = lambdas.ncols();
        let mut sample = Array1::<f64>::zeros(nf);
        for f in 0..nf {
            let pois = Poisson::new(lambdas[[state, f]]).unwrap();
            sample[f] = rng.sample::<f64, _>(&pois);
        }
        sample
    }

    fn initialize_sufficient_statistics(&self, n_components: usize) -> SufficientStatistics {
        let nf = self.n_features.unwrap();
        let mut stats = SufficientStatistics::new();
        stats.insert(
            "post".to_string(),
            ArrayD::zeros(IxDyn(&[n_components])),
        );
        stats.insert(
            "obs".to_string(),
            ArrayD::zeros(IxDyn(&[n_components, nf])),
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
        if params.contains('l') {
            let nc = posteriors.ncols();
            let ns = x.nrows();
            let nf = x.ncols();

            let post = stats.get_mut("post").unwrap();
            for c in 0..nc {
                let sum: f64 = (0..ns).map(|s| posteriors[[s, c]]).sum();
                post[IxDyn(&[c])] += sum;
            }

            let obs = stats.get_mut("obs").unwrap();
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
        if params.contains('l') {
            let post = stats.get("post").unwrap();
            let obs = stats.get("obs").unwrap();
            let lambdas = self.lambdas_.as_mut().unwrap();
            let nc = lambdas.nrows();
            let nf = lambdas.ncols();

            for c in 0..nc {
                for f in 0..nf {
                    let num = self.lambdas_prior * self.lambdas_weight + obs[IxDyn(&[c, f])];
                    let den = self.lambdas_weight + post[IxDyn(&[c])];
                    lambdas[[c, f]] = if den > 0.0 { num / den } else { 0.1 };
                }
            }
        }
    }

    fn n_fit_scalars_per_param(&self, n_components: usize) -> HashMap<char, usize> {
        let nf = self.n_features.unwrap_or(0);
        let mut m = HashMap::new();
        m.insert('l', n_components * nf);
        m
    }
}

/// Type alias for a PoissonHMM.
pub type PoissonHmm = crate::base::BaseHmm<PoissonEmissions>;

impl PoissonHmm {
    pub fn poisson(n_components: usize) -> Self {
        Self::new(n_components, PoissonEmissions::new())
            .with_params("stl")
            .with_init_params("stl")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_poisson_fit_and_score() {
        let mut model = PoissonHmm::poisson(2)
            .with_n_iter(20)
            .with_tol(1e-4);

        // Generate Poisson-like data
        let x = array![
            [1.0], [0.0], [2.0], [1.0], [3.0],
            [5.0], [4.0], [6.0], [5.0], [7.0]
        ];
        let lengths = [10];

        model.fit(&x, &lengths).unwrap();

        let score = model.score(&x, &lengths).unwrap();
        assert!(score.is_finite());
        assert!(score < 0.0);

        let (_, states) = model.decode(&x, &lengths, None).unwrap();
        assert_eq!(states.len(), 10);
        for &s in states.iter() {
            assert!(s < 2);
        }
    }
}
