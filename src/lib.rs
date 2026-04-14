//! # hmm-rs
//!
//! A Rust implementation of Hidden Markov Models, functionally equivalent to
//! the Python [hmmlearn](https://github.com/hmmlearn/hmmlearn) library.
//!
//! ## Supported Models
//!
//! ### EM-based (Maximum Likelihood)
//! - [`CategoricalHmm`] — Discrete symbol emissions
//! - [`GaussianHmm`] — Gaussian emissions (4 covariance types)
//! - [`GmmHmm`] — Gaussian Mixture Model emissions
//! - [`MultinomialHmm`] — Multinomial count emissions
//! - [`PoissonHmm`] — Poisson count emissions
//!
//! ### Variational Bayes
//! - [`VariationalCategoricalHmm`] — VB Categorical HMM
//! - [`VariationalGaussianHmm`] — VB Gaussian HMM
//!
//! ## Quick Start
//!
//! ```rust
//! use hmm_rs::prelude::*;
//! use ndarray::array;
//!
//! // Create a 2-state Gaussian HMM with diagonal covariance
//! let mut model = GaussianHmm::gaussian(2, CovarianceType::Diag)
//!     .with_n_iter(100)
//!     .with_tol(1e-4);
//!
//! // Fit to data
//! let x = array![[0.1, 0.2], [0.0, 0.1], [5.0, 5.1], [4.9, 5.0]];
//! model.fit(&x, &[4]).unwrap();
//!
//! // Decode most likely state sequence
//! let (log_prob, states) = model.decode(&x, &[4], None).unwrap();
//! ```

pub mod algorithms;
pub mod base;
pub mod emissions;
pub mod error;
pub mod kl_divergence;
pub mod monitor;
pub mod stats;
pub mod utils;
pub mod vhmm;

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use crate::base::{BaseHmm, DecoderAlgorithm, EmissionModel, Implementation, ParamFlags};
    pub use crate::emissions::CovarianceType;
    pub use crate::emissions::categorical::{CategoricalEmissions, CategoricalHmm};
    pub use crate::emissions::gaussian::{GaussianEmissions, GaussianHmm};
    pub use crate::emissions::gmm::{GmmEmissions, GmmHmm};
    pub use crate::emissions::multinomial::{MultinomialEmissions, MultinomialHmm};
    pub use crate::emissions::poisson::{PoissonEmissions, PoissonHmm};
    pub use crate::error::{HmmError, Result};
    pub use crate::monitor::ConvergenceMonitor;
    pub use crate::vhmm::VariationalBaseHmm;
    pub use crate::vhmm::categorical::{
        VariationalCategoricalEmissions, VariationalCategoricalHmm,
    };
    pub use crate::vhmm::gaussian::{VariationalGaussianEmissions, VariationalGaussianHmm};
}
