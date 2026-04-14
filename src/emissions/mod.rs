pub mod categorical;
pub mod gaussian;
pub mod gmm;
pub mod multinomial;
pub mod poisson;

use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};

/// Covariance matrix parameterization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CovarianceType {
    Full,
    Tied,
    Diag,
    Spherical,
}

/// Type-safe covariance storage for Gaussian models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CovarsStorage {
    Full(Array3<f64>),
    Tied(Array2<f64>),
    Diag(Array2<f64>),
    Spherical(Array1<f64>),
}
