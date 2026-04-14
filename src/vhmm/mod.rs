pub mod categorical;
pub mod gaussian;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use statrs::function::gamma::digamma;

use crate::algorithms;
use crate::base::{EmissionModel, ParamFlags, SufficientStatistics, sample_dirichlet};
use crate::error::{HmmError, Result};
use crate::kl_divergence;
use crate::monitor::ConvergenceMonitor;
use crate::utils;

/// Trait for variational emission models.
pub trait VariationalEmissionModel: EmissionModel + Clone {
    /// Compute sub-normalized log-likelihood using posterior expectations.
    fn compute_subnorm_log_likelihood(&self, x: &Array2<f64>) -> Array2<f64>;

    /// Pre-compute sub-normalized parameters at the beginning of each E-step.
    fn estep_begin(&mut self);

    /// Compute KL divergence between emission posteriors and priors.
    fn emission_kl_divergence(&self) -> f64;

    /// Initialize variational posterior parameters.
    fn init_variational(
        &mut self,
        x: &Array2<f64>,
        lengths: &[usize],
        n_components: usize,
        rng: &mut impl Rng,
    );

    /// M-step: update posterior parameters.
    fn do_mstep_variational(&mut self, stats: &SufficientStatistics, params: &ParamFlags);
}

/// Variational Bayes HMM.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "E: serde::Serialize + for<'a> serde::Deserialize<'a>")]
pub struct VariationalBaseHmm<E: VariationalEmissionModel> {
    pub n_components: usize,
    pub algorithm: crate::base::DecoderAlgorithm,
    pub implementation: crate::base::Implementation,
    pub n_iter: usize,
    pub tol: f64,
    pub verbose: bool,
    pub random_state: Option<u64>,
    pub params: ParamFlags,
    pub init_params: ParamFlags,

    // Prior and posterior for startprob
    pub startprob_prior_: Array1<f64>,
    pub startprob_posterior_: Array1<f64>,

    // Prior and posterior for transmat
    pub transmat_prior_: Array2<f64>,
    pub transmat_posterior_: Array2<f64>,

    // Sub-normalized parameters (computed each E-step)
    startprob_subnorm_: Array1<f64>,
    transmat_subnorm_: Array2<f64>,

    // Compat: point estimates for decode/predict
    pub startprob_: Array1<f64>,
    pub transmat_: Array2<f64>,

    pub emission: E,
    pub monitor_: ConvergenceMonitor,
    pub fitted: bool,
}

impl<E: VariationalEmissionModel> VariationalBaseHmm<E> {
    pub fn new(n_components: usize, emission: E) -> Self {
        Self {
            n_components,
            algorithm: crate::base::DecoderAlgorithm::Viterbi,
            implementation: crate::base::Implementation::Log,
            n_iter: 100,
            tol: 1e-6,
            verbose: false,
            random_state: None,
            params: ParamFlags::new(""),
            init_params: ParamFlags::new(""),
            startprob_prior_: Array1::zeros(0),
            startprob_posterior_: Array1::zeros(0),
            transmat_prior_: Array2::zeros((0, 0)),
            transmat_posterior_: Array2::zeros((0, 0)),
            startprob_subnorm_: Array1::zeros(0),
            transmat_subnorm_: Array2::zeros((0, 0)),
            startprob_: Array1::zeros(0),
            transmat_: Array2::zeros((0, 0)),
            emission,
            monitor_: ConvergenceMonitor::new(1e-6, 100, false),
            fitted: false,
        }
    }

    pub fn with_params(mut self, p: &str) -> Self {
        self.params = ParamFlags::new(p);
        self
    }

    pub fn with_init_params(mut self, p: &str) -> Self {
        self.init_params = ParamFlags::new(p);
        self
    }

    pub fn with_n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self.monitor_ = ConvergenceMonitor::new(self.tol, n, self.verbose);
        self
    }

    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self.monitor_ = ConvergenceMonitor::new(tol, self.n_iter, self.verbose);
        self
    }

    fn init(&mut self, x: &Array2<f64>, lengths: &[usize], rng: &mut impl Rng) {
        let nc = self.n_components;
        let nf = x.ncols();
        self.emission.set_n_features(nf);

        let uniform = 1.0 / nc as f64;

        // Initialize startprob prior/posterior
        if self.init_params.contains('s') || self.startprob_posterior_.len() != nc {
            self.startprob_prior_ = Array1::from_elem(nc, uniform);
            self.startprob_posterior_ = sample_dirichlet(nc, uniform, rng) * lengths.len() as f64;
        }

        // Initialize transmat prior/posterior
        if self.init_params.contains('t') || self.transmat_posterior_.shape() != [nc, nc] {
            self.transmat_prior_ = Array2::from_elem((nc, nc), uniform);
            let mut tp = Array2::<f64>::zeros((nc, nc));
            let total: usize = lengths.iter().sum();
            for i in 0..nc {
                let row = sample_dirichlet(nc, uniform, rng);
                tp.row_mut(i).assign(&(row * (total as f64 / nc as f64)));
            }
            self.transmat_posterior_ = tp;
        }

        // Initialize point estimates for decode/score
        let sp_sum = self.startprob_posterior_.sum();
        self.startprob_ = &self.startprob_posterior_ / sp_sum;
        self.transmat_ = Array2::<f64>::zeros((nc, nc));
        for i in 0..nc {
            let row_sum = self.transmat_posterior_.row(i).sum();
            for j in 0..nc {
                self.transmat_[[i, j]] = self.transmat_posterior_[[i, j]] / row_sum;
            }
        }

        // Initialize emission parameters
        self.emission.init_variational(x, lengths, nc, rng);
    }

    fn estep_begin(&mut self) {
        let nc = self.n_components;

        // Compute sub-normalized startprob
        let sp_sum = self.startprob_posterior_.sum();
        self.startprob_subnorm_ = Array1::from_shape_fn(nc, |i| {
            (digamma(self.startprob_posterior_[i]) - digamma(sp_sum)).exp()
        });

        // Compute sub-normalized transmat
        self.transmat_subnorm_ = Array2::from_shape_fn((nc, nc), |(i, j)| {
            let row_sum = self.transmat_posterior_.row(i).sum();
            (digamma(self.transmat_posterior_[[i, j]]) - digamma(row_sum)).exp()
        });

        // Emission-specific estep_begin
        self.emission.estep_begin();
    }

    fn compute_lower_bound(&self, curr_logprob: f64) -> f64 {
        let nc = self.n_components;

        // KL for startprob
        let startprob_kl =
            kl_divergence::kl_dirichlet(&self.startprob_posterior_, &self.startprob_prior_);

        // KL for transmat (per row)
        let mut transmat_kl = 0.0;
        for i in 0..nc {
            transmat_kl += kl_divergence::kl_dirichlet(
                &self.transmat_posterior_.row(i).to_owned(),
                &self.transmat_prior_.row(i).to_owned(),
            );
        }

        // Emission KL
        let emission_kl = self.emission.emission_kl_divergence();

        curr_logprob - startprob_kl - transmat_kl - emission_kl
    }

    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    pub fn fit(&mut self, x: &Array2<f64>, lengths: &[usize]) -> Result<&mut Self> {
        match self.random_state {
            Some(seed) => {
                let mut rng = StdRng::seed_from_u64(seed);
                self.init(x, lengths, &mut rng);
            }
            None => {
                let mut rng = rand::rng();
                self.init(x, lengths, &mut rng);
            }
        }
        self.monitor_ = ConvergenceMonitor::new(self.tol, self.n_iter, self.verbose);
        self.monitor_.reset();

        for _iter in 0..self.n_iter {
            self.estep_begin();

            // E-step
            let mut stats = self.initialize_sufficient_statistics();
            let mut curr_logprob = 0.0;

            for sub_x in utils::split_x_lengths(x, lengths) {
                let sub_x_owned = sub_x.to_owned();
                let framelogprob = self.emission.compute_subnorm_log_likelihood(&sub_x_owned);

                let (logprob, fwdlattice) = algorithms::forward_log(
                    &self.startprob_subnorm_,
                    &self.transmat_subnorm_,
                    &framelogprob,
                );
                let bwdlattice = algorithms::backward_log(
                    &self.startprob_subnorm_,
                    &self.transmat_subnorm_,
                    &framelogprob,
                );

                let mut log_gamma = &fwdlattice + &bwdlattice;
                utils::log_normalize_rows(&mut log_gamma);
                let posteriors = log_gamma.mapv(f64::exp);

                // Accumulate base stats
                {
                    stats.get_mut("nobs").unwrap().as_slice_mut().unwrap()[0] += 1.0;
                    if self.params.contains('s') {
                        let start = stats.get_mut("start").unwrap();
                        for i in 0..self.n_components {
                            start[IxDyn(&[i])] += posteriors[[0, i]];
                        }
                    }
                    if self.params.contains('t') && framelogprob.nrows() > 1 {
                        let log_xi = algorithms::compute_log_xi_sum(
                            &fwdlattice,
                            &self.transmat_subnorm_,
                            &bwdlattice,
                            &framelogprob,
                        );
                        let xi = log_xi.mapv(f64::exp);
                        let trans = stats.get_mut("trans").unwrap();
                        for i in 0..self.n_components {
                            for j in 0..self.n_components {
                                trans[IxDyn(&[i, j])] += xi[[i, j]];
                            }
                        }
                    }
                }

                // Accumulate emission stats
                self.emission.accumulate_sufficient_statistics(
                    &mut stats,
                    &sub_x_owned,
                    &posteriors,
                    &self.params,
                );

                curr_logprob += logprob;
            }

            // Compute lower bound
            let lower_bound = self.compute_lower_bound(curr_logprob);

            // M-step: update startprob/transmat posteriors
            if self.params.contains('s') {
                let start = stats.get("start").unwrap();
                for i in 0..self.n_components {
                    self.startprob_posterior_[i] = self.startprob_prior_[i] + start[IxDyn(&[i])];
                }
                // Compat point estimate
                let sum = self.startprob_posterior_.sum();
                self.startprob_ = &self.startprob_posterior_ / sum;
            }

            if self.params.contains('t') {
                let trans = stats.get("trans").unwrap();
                for i in 0..self.n_components {
                    for j in 0..self.n_components {
                        self.transmat_posterior_[[i, j]] =
                            self.transmat_prior_[[i, j]] + trans[IxDyn(&[i, j])];
                    }
                    // Compat point estimate
                    let row_sum = self.transmat_posterior_.row(i).sum();
                    for j in 0..self.n_components {
                        self.transmat_[[i, j]] = self.transmat_posterior_[[i, j]] / row_sum;
                    }
                }
            }

            // M-step: update emission posteriors
            self.emission.do_mstep_variational(&stats, &self.params);

            self.monitor_.report(lower_bound);
            if self.monitor_.converged() {
                break;
            }
        }

        self.fitted = true;
        Ok(self)
    }

    fn initialize_sufficient_statistics(&self) -> SufficientStatistics {
        let nc = self.n_components;
        let mut stats = self.emission.initialize_sufficient_statistics(nc);
        stats.insert("nobs".to_string(), ArrayD::zeros(IxDyn(&[1])));
        stats.insert("start".to_string(), ArrayD::zeros(IxDyn(&[nc])));
        stats.insert("trans".to_string(), ArrayD::zeros(IxDyn(&[nc, nc])));
        stats
    }

    pub fn score(&self, x: &Array2<f64>, lengths: &[usize]) -> Result<f64> {
        if !self.fitted {
            return Err(HmmError::NotFitted("not fitted".into()));
        }
        let mut log_prob = 0.0;
        for sub_x in utils::split_x_lengths(x, lengths) {
            let sub_x_owned = sub_x.to_owned();
            let log_frameprob = self.emission.compute_log_likelihood(&sub_x_owned);
            let (logprob, _) =
                algorithms::forward_log(&self.startprob_, &self.transmat_, &log_frameprob);
            log_prob += logprob;
        }
        Ok(log_prob)
    }

    pub fn decode(&self, x: &Array2<f64>, lengths: &[usize]) -> Result<(f64, Array1<usize>)> {
        if !self.fitted {
            return Err(HmmError::NotFitted("not fitted".into()));
        }
        let mut log_prob = 0.0;
        let mut all_states: Vec<Array1<usize>> = Vec::new();
        for sub_x in utils::split_x_lengths(x, lengths) {
            let sub_x_owned = sub_x.to_owned();
            let log_frameprob = self.emission.compute_log_likelihood(&sub_x_owned);
            let (lp, states) =
                algorithms::viterbi(&self.startprob_, &self.transmat_, &log_frameprob);
            log_prob += lp;
            all_states.push(states);
        }
        let states = ndarray::concatenate(
            ndarray::Axis(0),
            &all_states.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        Ok((log_prob, states))
    }
}
