//! Categorical (discrete) emission model.
//!
//! Port of hmmlearn's CategoricalHMM emission logic.

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use rand::Rng;

use crate::base::{EmissionModel, ParamFlags, SufficientStatistics};
use crate::error::{HmmError, Result};
use crate::utils;

/// Categorical emission model.
///
/// Each state emits a single integer symbol from {0, 1, ..., n_features - 1}.
/// Observations are expected as (n_samples, 1) integer-valued arrays stored as f64.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CategoricalEmissions {
    /// Number of possible symbols.
    pub n_features: Option<usize>,
    /// Emission probability matrix (n_components, n_features).
    pub emissionprob_: Option<Array2<f64>>,
    /// Dirichlet prior for emission probabilities.
    pub emissionprob_prior: f64,
}

impl Default for CategoricalEmissions {
    fn default() -> Self {
        Self::new()
    }
}

impl CategoricalEmissions {
    pub fn new() -> Self {
        Self {
            n_features: None,
            emissionprob_: None,
            emissionprob_prior: 1.0,
        }
    }

    pub fn with_n_features(mut self, n: usize) -> Self {
        self.n_features = Some(n);
        self
    }

    pub fn with_emissionprob_prior(mut self, prior: f64) -> Self {
        self.emissionprob_prior = prior;
        self
    }
}

impl EmissionModel for CategoricalEmissions {
    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn set_n_features(&mut self, _n: usize) {
        // For categorical, n_features is the number of symbols, not input columns.
        // It's inferred from data or set explicitly.
    }

    fn init(
        &mut self,
        x: &Array2<f64>,
        n_components: usize,
        params: &ParamFlags,
        rng: &mut impl Rng,
    ) {
        // Infer n_features from data if not set
        if self.n_features.is_none() {
            let max_symbol = x.iter().cloned().fold(0.0_f64, f64::max) as usize;
            self.n_features = Some(max_symbol + 1);
        }

        let nf = self.n_features.unwrap();

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
            return Err(HmmError::InvalidParameter(format!(
                "emissionprob_ must have shape ({}, {}), got {:?}",
                n_components,
                nf,
                ep.shape()
            )));
        }

        // Check rows sum to 1
        for i in 0..n_components {
            let row_sum = ep.row(i).sum();
            if (row_sum - 1.0).abs() > 1e-4 {
                return Err(HmmError::InvalidParameter(format!(
                    "emissionprob_ row {i} sums to {row_sum}"
                )));
            }
        }

        Ok(())
    }

    fn compute_log_likelihood(&self, x: &Array2<f64>) -> Array2<f64> {
        let ep = self.emissionprob_.as_ref().unwrap();
        let nc = ep.nrows();
        let ns = x.nrows();

        let mut result = Array2::<f64>::zeros((ns, nc));
        for s in 0..ns {
            let symbol = x[[s, 0]] as usize;
            for c in 0..nc {
                result[[s, c]] = ep[[c, symbol]].ln();
            }
        }
        result
    }

    fn compute_likelihood(&self, x: &Array2<f64>) -> Array2<f64> {
        let ep = self.emissionprob_.as_ref().unwrap();
        let nc = ep.nrows();
        let ns = x.nrows();

        let mut result = Array2::<f64>::zeros((ns, nc));
        for s in 0..ns {
            let symbol = x[[s, 0]] as usize;
            for c in 0..nc {
                result[[s, c]] = ep[[c, symbol]];
            }
        }
        result
    }

    fn generate_sample_from_state(&self, state: usize, rng: &mut impl Rng) -> Array1<f64> {
        let ep = self.emissionprob_.as_ref().unwrap();
        let nf = ep.ncols();

        // CDF sampling
        let u: f64 = rng.random();
        let mut cumsum = 0.0;
        for j in 0..nf {
            cumsum += ep[[state, j]];
            if cumsum > u {
                return Array1::from_vec(vec![j as f64]);
            }
        }
        Array1::from_vec(vec![(nf - 1) as f64])
    }

    fn initialize_sufficient_statistics(&self, n_components: usize) -> SufficientStatistics {
        let nf = self.n_features.unwrap();
        let mut stats = SufficientStatistics::new();
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

/// Type alias for a CategoricalHMM.
pub type CategoricalHmm = crate::base::BaseHmm<CategoricalEmissions>;

impl CategoricalHmm {
    /// Create a new CategoricalHMM with default settings.
    pub fn categorical(n_components: usize) -> Self {
        Self::new(n_components, CategoricalEmissions::new())
            .with_params("ste")
            .with_init_params("ste")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_categorical_fit_and_score() {
        // Create a simple 2-state, 3-symbol CategoricalHMM
        let mut model = CategoricalHmm::categorical(2)
            .with_n_iter(20)
            .with_tol(1e-4);

        // Set n_features
        model.emission.n_features = Some(3);

        // Generate some fake data: sequence of symbols
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

        // Fit the model
        model.fit(&x, &lengths).unwrap();

        // Score should return a finite value
        let score = model.score(&x, &lengths).unwrap();
        assert!(score.is_finite(), "score should be finite: {}", score);
        assert!(score < 0.0, "log probability should be negative");

        // Decode should return valid states
        let (log_prob, states) = model.decode(&x, &lengths, None).unwrap();
        assert!(log_prob.is_finite());
        assert_eq!(states.len(), 10);
        for &s in states.iter() {
            assert!(s < 2);
        }

        // Posteriors should sum to 1 per row
        let posteriors = model.predict_proba(&x, &lengths).unwrap();
        assert_eq!(posteriors.shape(), &[10, 2]);
        for row in posteriors.rows() {
            let sum = row.sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "posterior row should sum to 1, got {}",
                sum
            );
        }
    }

    #[test]
    fn test_categorical_sample() {
        let mut model = CategoricalHmm::categorical(2);
        model.emission.n_features = Some(3);

        // Set parameters manually
        model.startprob_ = array![0.6, 0.4];
        model.transmat_ = array![[0.7, 0.3], [0.4, 0.6]];
        model.emission.emissionprob_ = Some(array![
            [0.5, 0.3, 0.2],
            [0.1, 0.4, 0.5]
        ]);
        model.fitted = true;

        let mut rng = rand::rng();
        let (x, states) = model.sample(100, &mut rng, None).unwrap();
        assert_eq!(x.nrows(), 100);
        assert_eq!(x.ncols(), 1);
        assert_eq!(states.len(), 100);

        // All samples should be valid symbols
        for &v in x.iter() {
            assert!(v >= 0.0 && v < 3.0);
        }
        for &s in states.iter() {
            assert!(s < 2);
        }
    }

    #[test]
    fn test_categorical_multiple_sequences() {
        let mut model = CategoricalHmm::categorical(2)
            .with_n_iter(10)
            .with_tol(1e-4);
        model.emission.n_features = Some(3);

        let x = array![
            [0.0], [1.0], [2.0], [0.0], [1.0],
            [2.0], [2.0], [1.0], [0.0], [0.0]
        ];
        let lengths = [5, 5];

        model.fit(&x, &lengths).unwrap();
        let score = model.score(&x, &lengths).unwrap();
        assert!(score.is_finite());
    }
}
