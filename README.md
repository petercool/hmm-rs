# hmm-rs

A Rust implementation of Hidden Markov Models, functionally equivalent to
Python's [hmmlearn](https://github.com/hmmlearn/hmmlearn).

## Features

- **7 model types**: CategoricalHMM, GaussianHMM, GMMHMM, MultinomialHMM, PoissonHMM, VariationalCategoricalHMM, VariationalGaussianHMM
- **4 covariance types** for Gaussian models: full, diagonal, tied, spherical
- **Both EM and Variational Bayes** training
- **Both log and scaling** forward-backward implementations
- **Viterbi and MAP** decoders
- **Multiple sequences** via the `lengths` parameter
- **Serde serialization** — save/load models as JSON
- **Reproducible** — optional `random_state` seed
- **Pure Rust** — no C/Fortran/LAPACK dependencies

## Quick Start

```rust
use hmm_rs::prelude::*;
use ndarray::array;

// Create and fit a 2-state Gaussian HMM
let mut model = GaussianHmm::gaussian(2, CovarianceType::Diag)
    .with_n_iter(100)
    .with_tol(1e-4)
    .with_random_state(42);

let x = array![[0.1, 0.2], [0.0, 0.1], [5.0, 5.1], [4.9, 5.0]];
model.fit(&x, &[4]).unwrap();

// Decode most likely state sequence
let (log_prob, states) = model.decode(&x, &[4], None).unwrap();

// Get posterior probabilities
let posteriors = model.predict_proba(&x, &[4]).unwrap();

// Model selection
let aic = model.aic(&x, &[4]).unwrap();
let bic = model.bic(&x, &[4]).unwrap();

// Save and load
let json = serde_json::to_string(&model).unwrap();
let loaded: GaussianHmm = serde_json::from_str(&json).unwrap();
```

## All Model Types

| Model | Emissions | Training | Params |
|---|---|---|---|
| `CategoricalHmm` | Discrete symbols | EM | `"ste"` |
| `GaussianHmm` | Multivariate Gaussian | EM | `"stmc"` |
| `GmmHmm` | Gaussian Mixture | EM | `"stmcw"` |
| `MultinomialHmm` | Multinomial counts | EM | `"ste"` |
| `PoissonHmm` | Poisson counts | EM | `"stl"` |
| `VariationalCategoricalHmm` | Discrete symbols | VB | `"ste"` |
| `VariationalGaussianHmm` | Multivariate Gaussian | VB | `"stmc"` |

## Examples

Five runnable examples with [Plotly](https://docs.rs/plotly) HTML visualizations:

```bash
cargo run --example stock_regime_detection   # Market regime detection
cargo run --example weather_hmm              # Classic weather/activity HMM
cargo run --example gaussian_clustering      # Covariance type comparison
cargo run --example model_selection          # AIC/BIC for optimal state count
cargo run --example sequence_generation      # Sampling and serialization
```

## Running Tests

```bash
cargo test          # 71 tests (52 unit + 18 integration + 1 doc)
cargo bench         # Criterion benchmarks
```

## License

MIT
