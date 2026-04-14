use thiserror::Error;

#[derive(Debug, Error)]
pub enum HmmError {
    #[error("model not fitted: {0}")]
    NotFitted(String),
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("singular covariance matrix for component {0}")]
    SingularCovariance(usize),
    #[error("forward pass failed with underflow; consider using implementation='log'")]
    ForwardUnderflow,
    #[error("linear algebra error: {0}")]
    LinAlg(String),
}

pub type Result<T> = std::result::Result<T, HmmError>;
