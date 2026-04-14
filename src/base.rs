//! Base HMM infrastructure: EmissionModel trait, ParamFlags, and BaseHmm<E>.
//!
//! Port of hmmlearn's `base.py` — BaseHMM class with EM training loop.

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Gamma;
use serde::{Deserialize, Serialize};

use crate::algorithms;
use crate::error::{HmmError, Result};
use crate::monitor::ConvergenceMonitor;
use crate::utils;

/// Sufficient statistics accumulated during the E-step.
pub type SufficientStatistics = HashMap<String, ArrayD<f64>>;

/// E-step result for a single sequence: (lattice, log_prob, posteriors, fwdlattice, bwdlattice).
type EStepResult = (Array2<f64>, f64, Array2<f64>, Array2<f64>, Array2<f64>);

/// Decoder algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecoderAlgorithm {
    Viterbi,
    Map,
}

/// Forward-backward implementation selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Implementation {
    Log,
    Scaling,
}

/// Flags controlling which parameters to update during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamFlags {
    flags: String,
}

impl ParamFlags {
    pub fn new(s: &str) -> Self {
        Self {
            flags: s.to_string(),
        }
    }

    pub fn contains(&self, c: char) -> bool {
        self.flags.contains(c)
    }
}

/// Trait for emission model implementations.
///
/// Each concrete emission model (Gaussian, Categorical, etc.) implements this trait.
/// The BaseHmm struct is generic over this trait.
pub trait EmissionModel: Clone {
    /// Number of features, if known.
    fn n_features(&self) -> Option<usize>;

    /// Set the number of features after observing data.
    fn set_n_features(&mut self, n: usize);

    /// Initialize emission parameters from data.
    fn init(
        &mut self,
        x: &Array2<f64>,
        n_components: usize,
        params: &ParamFlags,
        rng: &mut impl Rng,
    );

    /// Validate emission parameters.
    fn check(&self, n_components: usize) -> Result<()>;

    /// Compute log emission probability for each (sample, component) pair.
    /// Returns shape (n_samples, n_components).
    fn compute_log_likelihood(&self, x: &Array2<f64>) -> Array2<f64>;

    /// Compute emission probability (not log).
    /// Default: exp(compute_log_likelihood(x)).
    fn compute_likelihood(&self, x: &Array2<f64>) -> Array2<f64> {
        self.compute_log_likelihood(x).mapv(f64::exp)
    }

    /// Generate a random sample from the emission distribution of a given state.
    fn generate_sample_from_state(&self, state: usize, rng: &mut impl Rng) -> Array1<f64>;

    /// Initialize sufficient statistics for the E-step.
    fn initialize_sufficient_statistics(&self, n_components: usize) -> SufficientStatistics;

    /// Accumulate sufficient statistics from one sequence.
    fn accumulate_sufficient_statistics(
        &self,
        stats: &mut SufficientStatistics,
        x: &Array2<f64>,
        posteriors: &Array2<f64>,
        params: &ParamFlags,
    );

    /// Perform emission-specific M-step updates.
    fn do_mstep(&mut self, stats: &SufficientStatistics, params: &ParamFlags);

    /// Total number of free scalar parameters per param flag character (for AIC/BIC).
    /// Must return the **total** count (already multiplied by n_components where appropriate).
    fn n_fit_scalars_per_param(&self, n_components: usize) -> HashMap<char, usize>;
}

/// Hidden Markov Model with EM training.
///
/// Generic over the emission model `E`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "E: Serialize + for<'a> Deserialize<'a>")]
pub struct BaseHmm<E: EmissionModel> {
    // Configuration
    pub n_components: usize,
    pub algorithm: DecoderAlgorithm,
    pub implementation: Implementation,
    pub n_iter: usize,
    pub tol: f64,
    pub verbose: bool,
    pub params: ParamFlags,
    pub init_params: ParamFlags,

    // RNG seed for reproducibility
    pub random_state: Option<u64>,

    // Priors
    pub startprob_prior: f64,
    pub transmat_prior: f64,

    // Learned parameters
    pub startprob_: Array1<f64>,
    pub transmat_: Array2<f64>,

    // Emission model
    pub emission: E,

    // Convergence monitor
    pub monitor_: ConvergenceMonitor,

    // Internal state
    pub fitted: bool,
}

impl<E: EmissionModel> BaseHmm<E> {
    /// Create a new BaseHmm with default configuration.
    pub fn new(n_components: usize, emission: E) -> Self {
        Self {
            n_components,
            algorithm: DecoderAlgorithm::Viterbi,
            implementation: Implementation::Log,
            n_iter: 10,
            tol: 1e-2,
            verbose: false,
            params: ParamFlags::new(""),      // will be set by subtype
            init_params: ParamFlags::new(""), // will be set by subtype
            random_state: None,
            startprob_prior: 1.0,
            transmat_prior: 1.0,
            startprob_: Array1::zeros(0),
            transmat_: Array2::zeros((0, 0)),
            emission,
            monitor_: ConvergenceMonitor::new(1e-2, 10, false),
            fitted: false,
        }
    }

    /// Builder: set decoder algorithm.
    pub fn with_algorithm(mut self, alg: DecoderAlgorithm) -> Self {
        self.algorithm = alg;
        self
    }

    /// Builder: set forward-backward implementation.
    pub fn with_implementation(mut self, imp: Implementation) -> Self {
        self.implementation = imp;
        self
    }

    /// Builder: set max EM iterations.
    pub fn with_n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self.monitor_ = ConvergenceMonitor::new(self.tol, n, self.verbose);
        self
    }

    /// Builder: set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self.monitor_ = ConvergenceMonitor::new(tol, self.n_iter, self.verbose);
        self
    }

    /// Builder: set verbose mode.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self.monitor_ = ConvergenceMonitor::new(self.tol, self.n_iter, verbose);
        self
    }

    /// Builder: set parameter update flags.
    pub fn with_params(mut self, params: &str) -> Self {
        self.params = ParamFlags::new(params);
        self
    }

    /// Builder: set initialization flags.
    pub fn with_init_params(mut self, params: &str) -> Self {
        self.init_params = ParamFlags::new(params);
        self
    }

    /// Builder: set priors.
    pub fn with_startprob_prior(mut self, prior: f64) -> Self {
        self.startprob_prior = prior;
        self
    }

    pub fn with_transmat_prior(mut self, prior: f64) -> Self {
        self.transmat_prior = prior;
        self
    }

    /// Builder: set random seed for reproducibility.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Initialize model parameters before fitting.
    fn init(&mut self, x: &Array2<f64>, rng: &mut impl Rng) {
        // Set n_features from data
        let n_features = x.ncols();
        self.emission.set_n_features(n_features);

        let nc = self.n_components;
        let init = 1.0 / nc as f64;

        // Initialize startprob
        if self.init_params.contains('s') || self.startprob_.len() != nc {
            self.startprob_ = sample_dirichlet(nc, init, rng);
        }

        // Initialize transmat
        if self.init_params.contains('t') || self.transmat_.shape() != [nc, nc] {
            let mut transmat = Array2::<f64>::zeros((nc, nc));
            for i in 0..nc {
                let row = sample_dirichlet(nc, init, rng);
                transmat.row_mut(i).assign(&row);
            }
            self.transmat_ = transmat;
        }

        // Initialize emission parameters
        self.emission.init(x, nc, &self.init_params, rng);
    }

    /// Validate model parameters.
    fn check(&self) -> Result<()> {
        let nc = self.n_components;

        if self.startprob_.len() != nc {
            return Err(HmmError::InvalidParameter(
                "startprob_ must have length n_components".into(),
            ));
        }
        // Normalize startprob/transmat if they don't sum to 1
        // (can happen with degenerate data)

        if self.transmat_.shape() != [nc, nc] {
            return Err(HmmError::InvalidParameter(
                "transmat_ must have shape (n_components, n_components)".into(),
            ));
        }

        self.emission.check(nc)
    }

    /// Ensure startprob/transmat are valid distributions. Fix degenerate cases.
    fn fix_params(&mut self) {
        // Fix startprob
        let sp_sum = self.startprob_.sum();
        if sp_sum == 0.0 || !sp_sum.is_finite() {
            let nc = self.n_components;
            self.startprob_.fill(1.0 / nc as f64);
        } else if (sp_sum - 1.0).abs() > 1e-10 {
            self.startprob_ /= sp_sum;
        }

        // Fix transmat rows
        let nc = self.n_components;
        for i in 0..nc {
            let row_sum = self.transmat_.row(i).sum();
            if row_sum == 0.0 || !row_sum.is_finite() {
                for j in 0..nc {
                    self.transmat_[[i, j]] = 1.0 / nc as f64;
                }
            } else if (row_sum - 1.0).abs() > 1e-10 {
                for j in 0..nc {
                    self.transmat_[[i, j]] /= row_sum;
                }
            }
        }
    }

    /// Estimate model parameters using EM.
    pub fn fit(&mut self, x: &Array2<f64>, lengths: &[usize]) -> Result<&mut Self> {
        match self.random_state {
            Some(seed) => {
                let mut rng = StdRng::seed_from_u64(seed);
                self.init(x, &mut rng);
            }
            None => {
                let mut rng = rand::rng();
                self.init(x, &mut rng);
            }
        }
        self.fix_params();
        self.check()?;
        self.monitor_ = ConvergenceMonitor::new(self.tol, self.n_iter, self.verbose);
        self.monitor_.reset();

        for _iter in 0..self.n_iter {
            let (stats, curr_logprob) = self.do_estep(x, lengths)?;

            // M-step: update base parameters
            self.do_mstep(&stats);

            // M-step: update emission parameters
            self.emission.do_mstep(&stats, &self.params);

            // Fix any degenerate parameters
            self.fix_params();

            self.monitor_.report(curr_logprob);
            if self.monitor_.converged() {
                break;
            }
        }

        self.fitted = true;
        Ok(self)
    }

    /// Perform the E-step: compute sufficient statistics and log-likelihood.
    fn do_estep(&self, x: &Array2<f64>, lengths: &[usize]) -> Result<(SufficientStatistics, f64)> {
        let mut stats = self.initialize_sufficient_statistics();
        let mut curr_logprob = 0.0;

        for sub_x in utils::split_x_lengths(x, lengths) {
            let sub_x_owned = sub_x.to_owned();
            let (lattice, logprob, posteriors, fwdlattice, bwdlattice) = match self.implementation {
                Implementation::Log => self.fit_log(&sub_x_owned)?,
                Implementation::Scaling => self.fit_scaling(&sub_x_owned)?,
            };

            // Accumulate base sufficient statistics
            self.accumulate_base_sufficient_statistics(
                &mut stats,
                &lattice,
                &posteriors,
                &fwdlattice,
                &bwdlattice,
            );

            // Accumulate emission sufficient statistics
            self.emission.accumulate_sufficient_statistics(
                &mut stats,
                &sub_x_owned,
                &posteriors,
                &self.params,
            );

            curr_logprob += logprob;
        }

        Ok((stats, curr_logprob))
    }

    /// Log implementation of the E-step for a single sequence.
    fn fit_log(&self, x: &Array2<f64>) -> Result<EStepResult> {
        let log_frameprob = self.emission.compute_log_likelihood(x);
        let (log_prob, fwdlattice) =
            algorithms::forward_log(&self.startprob_, &self.transmat_, &log_frameprob);
        let bwdlattice =
            algorithms::backward_log(&self.startprob_, &self.transmat_, &log_frameprob);
        let posteriors = compute_posteriors_log(&fwdlattice, &bwdlattice);
        Ok((log_frameprob, log_prob, posteriors, fwdlattice, bwdlattice))
    }

    /// Scaling implementation of the E-step for a single sequence.
    fn fit_scaling(&self, x: &Array2<f64>) -> Result<EStepResult> {
        let frameprob = self.emission.compute_likelihood(x);
        let (log_prob, fwdlattice, scaling_factors) =
            algorithms::forward_scaling(&self.startprob_, &self.transmat_, &frameprob)?;
        let bwdlattice = algorithms::backward_scaling(
            &self.startprob_,
            &self.transmat_,
            &frameprob,
            &scaling_factors,
        );
        let posteriors = compute_posteriors_scaling(&fwdlattice, &bwdlattice);
        Ok((frameprob, log_prob, posteriors, fwdlattice, bwdlattice))
    }

    /// Initialize sufficient statistics.
    fn initialize_sufficient_statistics(&self) -> SufficientStatistics {
        let nc = self.n_components;
        let mut stats = self.emission.initialize_sufficient_statistics(nc);
        stats.insert("nobs".to_string(), ArrayD::zeros(IxDyn(&[1])));
        stats.insert("start".to_string(), ArrayD::zeros(IxDyn(&[nc])));
        stats.insert("trans".to_string(), ArrayD::zeros(IxDyn(&[nc, nc])));
        stats
    }

    /// Accumulate base (non-emission) sufficient statistics.
    fn accumulate_base_sufficient_statistics(
        &self,
        stats: &mut SufficientStatistics,
        lattice: &Array2<f64>,
        posteriors: &Array2<f64>,
        fwdlattice: &Array2<f64>,
        bwdlattice: &Array2<f64>,
    ) {
        // nobs
        stats.get_mut("nobs").unwrap().as_slice_mut().unwrap()[0] += 1.0;

        // startprob: accumulate first frame's posteriors
        if self.params.contains('s') {
            let start = stats.get_mut("start").unwrap();
            let start_slice = start.as_slice_mut().unwrap();
            for i in 0..self.n_components {
                start_slice[i] += posteriors[[0, i]];
            }
        }

        // transmat: accumulate transition counts
        if self.params.contains('t') {
            let ns = lattice.nrows();
            if ns <= 1 {
                return;
            }

            let xi_sum = match self.implementation {
                Implementation::Log => {
                    let log_xi = algorithms::compute_log_xi_sum(
                        fwdlattice,
                        &self.transmat_,
                        bwdlattice,
                        lattice,
                    );
                    log_xi.mapv(f64::exp)
                }
                Implementation::Scaling => algorithms::compute_scaling_xi_sum(
                    fwdlattice,
                    &self.transmat_,
                    bwdlattice,
                    lattice,
                ),
            };

            let trans = stats.get_mut("trans").unwrap();
            let nc = self.n_components;
            for i in 0..nc {
                for j in 0..nc {
                    trans[IxDyn(&[i, j])] += xi_sum[[i, j]];
                }
            }
        }
    }

    /// Perform the base M-step: update startprob and transmat.
    fn do_mstep(&mut self, stats: &SufficientStatistics) {
        if self.params.contains('s') {
            let start = stats.get("start").unwrap();
            let nc = self.n_components;
            for i in 0..nc {
                let val = self.startprob_prior - 1.0 + start[IxDyn(&[i])];
                self.startprob_[i] = if self.startprob_[i] == 0.0 {
                    0.0
                } else {
                    val.max(0.0)
                };
            }
            utils::normalize(&mut self.startprob_);
        }

        if self.params.contains('t') {
            let trans = stats.get("trans").unwrap();
            let nc = self.n_components;
            for i in 0..nc {
                for j in 0..nc {
                    let val = self.transmat_prior - 1.0 + trans[IxDyn(&[i, j])];
                    self.transmat_[[i, j]] = if self.transmat_[[i, j]] == 0.0 {
                        0.0
                    } else {
                        val.max(0.0)
                    };
                }
            }
            utils::normalize_rows(&mut self.transmat_);
        }
    }

    fn check_fitted(&self) -> Result<()> {
        if !self.fitted {
            return Err(HmmError::NotFitted("model has not been fitted yet".into()));
        }
        Ok(())
    }

    /// Compute the log probability under the model.
    pub fn score(&self, x: &Array2<f64>, lengths: &[usize]) -> Result<f64> {
        self.check_fitted()?;
        let (log_prob, _) = self.score_samples(x, lengths)?;
        Ok(log_prob)
    }

    /// Compute log probability and posteriors.
    pub fn score_samples(&self, x: &Array2<f64>, lengths: &[usize]) -> Result<(f64, Array2<f64>)> {
        self.check_fitted()?;
        self.check()?;

        let mut log_prob = 0.0;
        let mut all_posteriors: Vec<Array2<f64>> = Vec::new();

        for sub_x in utils::split_x_lengths(x, lengths) {
            let sub_x_owned = sub_x.to_owned();
            match self.implementation {
                Implementation::Log => {
                    let log_frameprob = self.emission.compute_log_likelihood(&sub_x_owned);
                    let (log_probij, fwdlattice) =
                        algorithms::forward_log(&self.startprob_, &self.transmat_, &log_frameprob);
                    let bwdlattice =
                        algorithms::backward_log(&self.startprob_, &self.transmat_, &log_frameprob);
                    let posteriors = compute_posteriors_log(&fwdlattice, &bwdlattice);
                    log_prob += log_probij;
                    all_posteriors.push(posteriors);
                }
                Implementation::Scaling => {
                    let frameprob = self.emission.compute_likelihood(&sub_x_owned);
                    let (log_probij, fwdlattice, scaling) =
                        algorithms::forward_scaling(&self.startprob_, &self.transmat_, &frameprob)?;
                    let bwdlattice = algorithms::backward_scaling(
                        &self.startprob_,
                        &self.transmat_,
                        &frameprob,
                        &scaling,
                    );
                    let posteriors = compute_posteriors_scaling(&fwdlattice, &bwdlattice);
                    log_prob += log_probij;
                    all_posteriors.push(posteriors);
                }
            }
        }

        let posteriors = ndarray::concatenate(
            ndarray::Axis(0),
            &all_posteriors.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .unwrap();

        Ok((log_prob, posteriors))
    }

    /// Find most likely state sequence.
    pub fn decode(
        &self,
        x: &Array2<f64>,
        lengths: &[usize],
        algorithm: Option<DecoderAlgorithm>,
    ) -> Result<(f64, Array1<usize>)> {
        self.check_fitted()?;
        self.check()?;

        let algo = algorithm.unwrap_or(self.algorithm);

        let mut log_prob = 0.0;
        let mut all_states: Vec<Array1<usize>> = Vec::new();

        for sub_x in utils::split_x_lengths(x, lengths) {
            let sub_x_owned = sub_x.to_owned();
            let (sub_log_prob, sub_states) = match algo {
                DecoderAlgorithm::Viterbi => {
                    let log_frameprob = self.emission.compute_log_likelihood(&sub_x_owned);
                    algorithms::viterbi(&self.startprob_, &self.transmat_, &log_frameprob)
                }
                DecoderAlgorithm::Map => {
                    let (_, posteriors) =
                        self.score_samples(&sub_x_owned, &[sub_x_owned.nrows()])?;
                    let map_log_prob: f64 = posteriors
                        .rows()
                        .into_iter()
                        .map(|row| row.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
                        .sum();
                    let map_states = Array1::from_vec(
                        posteriors
                            .rows()
                            .into_iter()
                            .map(|row| {
                                row.iter()
                                    .enumerate()
                                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                    .unwrap()
                                    .0
                            })
                            .collect(),
                    );
                    (map_log_prob, map_states)
                }
            };
            log_prob += sub_log_prob;
            all_states.push(sub_states);
        }

        let states = ndarray::concatenate(
            ndarray::Axis(0),
            &all_states.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .unwrap();

        Ok((log_prob, states))
    }

    /// Predict most likely state sequence (convenience wrapper for decode).
    pub fn predict(&self, x: &Array2<f64>, lengths: &[usize]) -> Result<Array1<usize>> {
        let (_, states) = self.decode(x, lengths, None)?;
        Ok(states)
    }

    /// Compute posterior state probabilities.
    pub fn predict_proba(&self, x: &Array2<f64>, lengths: &[usize]) -> Result<Array2<f64>> {
        let (_, posteriors) = self.score_samples(x, lengths)?;
        Ok(posteriors)
    }

    /// Generate random samples from the model.
    pub fn sample(
        &self,
        n_samples: usize,
        rng: &mut impl Rng,
        currstate: Option<usize>,
    ) -> Result<(Array2<f64>, Array1<usize>)> {
        self.check_fitted()?;
        self.check()?;

        // Build CDF for transitions
        let nc = self.n_components;
        let mut transmat_cdf = Array2::<f64>::zeros((nc, nc));
        for i in 0..nc {
            let mut cumsum = 0.0;
            for j in 0..nc {
                cumsum += self.transmat_[[i, j]];
                transmat_cdf[[i, j]] = cumsum;
            }
        }

        // Initial state
        let mut state = match currstate {
            Some(s) => s,
            None => {
                let mut startprob_cdf = Array1::<f64>::zeros(nc);
                let mut cumsum = 0.0;
                for i in 0..nc {
                    cumsum += self.startprob_[i];
                    startprob_cdf[i] = cumsum;
                }
                let u: f64 = rng.random();
                startprob_cdf.iter().position(|&c| c > u).unwrap_or(nc - 1)
            }
        };

        let mut states = Vec::with_capacity(n_samples);
        let mut samples: Vec<Array1<f64>> = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            states.push(state);
            samples.push(self.emission.generate_sample_from_state(state, rng));

            // Transition to next state
            let u: f64 = rng.random();
            state = transmat_cdf
                .row(state)
                .iter()
                .position(|&c| c > u)
                .unwrap_or(nc - 1);
        }

        // Stack samples into (n_samples, n_features)
        let n_features = if samples.is_empty() {
            0
        } else {
            samples[0].len()
        };
        let mut x = Array2::<f64>::zeros((n_samples, n_features));
        for (i, sample) in samples.iter().enumerate() {
            x.row_mut(i).assign(sample);
        }

        Ok((x, Array1::from_vec(states)))
    }

    /// Akaike Information Criterion.
    pub fn aic(&self, x: &Array2<f64>, lengths: &[usize]) -> Result<f64> {
        let n_params = self.n_free_params();
        let log_l = self.score(x, lengths)?;
        Ok(-2.0 * log_l + 2.0 * n_params as f64)
    }

    /// Bayesian Information Criterion.
    pub fn bic(&self, x: &Array2<f64>, lengths: &[usize]) -> Result<f64> {
        let n_params = self.n_free_params();
        let log_l = self.score(x, lengths)?;
        Ok(-2.0 * log_l + n_params as f64 * (x.nrows() as f64).ln())
    }

    /// Total number of free parameters.
    fn n_free_params(&self) -> usize {
        let nc = self.n_components;
        let mut base = HashMap::new();
        base.insert('s', nc - 1);
        base.insert('t', nc * (nc - 1));

        let emission_params = self.emission.n_fit_scalars_per_param(nc);

        let mut total = 0;
        for c in self.params.flags.chars() {
            if let Some(&n) = base.get(&c) {
                total += n;
            } else if let Some(&n) = emission_params.get(&c) {
                total += n;
            }
        }
        total
    }

    /// Compute the stationary distribution of the transition matrix.
    pub fn get_stationary_distribution(&self) -> Result<Array1<f64>> {
        self.check_fitted()?;

        let nc = self.n_components;
        // The stationary distribution is the left eigenvector of transmat
        // associated with eigenvalue 1. We find it by solving (A^T - I) pi = 0
        // with the constraint sum(pi) = 1.
        // Simple power iteration approach:
        let mut pi = Array1::from_elem(nc, 1.0 / nc as f64);
        for _ in 0..1000 {
            let mut new_pi = Array1::<f64>::zeros(nc);
            for j in 0..nc {
                for i in 0..nc {
                    new_pi[j] += pi[i] * self.transmat_[[i, j]];
                }
            }
            utils::normalize(&mut new_pi);
            let diff: f64 = (&new_pi - &pi).mapv(f64::abs).sum();
            pi = new_pi;
            if diff < 1e-14 {
                break;
            }
        }
        Ok(pi)
    }
}

/// Sample from a Dirichlet distribution with uniform alpha.
/// Uses the Gamma distribution method: sample x_i ~ Gamma(alpha, 1), then normalize.
pub fn sample_dirichlet(n: usize, alpha: f64, rng: &mut impl Rng) -> Array1<f64> {
    let gamma = Gamma::new(alpha, 1.0).unwrap();
    let mut result = Array1::<f64>::zeros(n);
    for i in 0..n {
        result[i] = rng.sample(gamma);
    }
    let sum = result.sum();
    if sum > 0.0 {
        result /= sum;
    } else {
        result.fill(1.0 / n as f64);
    }
    result
}

/// Compute posteriors from log-space forward and backward lattices.
fn compute_posteriors_log(fwdlattice: &Array2<f64>, bwdlattice: &Array2<f64>) -> Array2<f64> {
    let mut log_gamma = fwdlattice + bwdlattice;
    utils::log_normalize_rows(&mut log_gamma);
    log_gamma.mapv(f64::exp)
}

/// Compute posteriors from scaling-space forward and backward lattices.
fn compute_posteriors_scaling(fwdlattice: &Array2<f64>, bwdlattice: &Array2<f64>) -> Array2<f64> {
    let mut posteriors = fwdlattice * bwdlattice;
    utils::normalize_rows(&mut posteriors);
    posteriors
}
