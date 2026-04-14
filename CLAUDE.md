# CLAUDE.md

## Project: hmm-rs

Rust port of Python's hmmlearn library for Hidden Markov Models.

## Build & Test

```bash
cargo build          # Build the library
cargo test           # Run all tests (52 unit + integration)
cargo bench          # Run benchmarks (criterion)
cargo run --example stock_regime_detection  # Run an example
```

## Architecture

- **Composition over inheritance**: Python's class hierarchy maps to `BaseHmm<E: EmissionModel>` where emission-specific logic lives in the generic parameter.
- **`EmissionModel` trait**: Core abstraction. Each model type implements this trait for its specific emission distribution.
- **`VariationalBaseHmm<E>`**: Separate struct for VB-trained models (different from `BaseHmm`).
- **No LAPACK dependency**: Uses hand-rolled Cholesky decomposition for portability. No external C/Fortran libraries needed.

## Module Layout

```
src/
  lib.rs           - Public API, prelude re-exports
  algorithms.rs    - Forward/backward/Viterbi (port of _hmmc.cpp)
  base.rs          - EmissionModel trait, BaseHmm<E>, EM loop
  stats.rs         - Gaussian log-density, Cholesky
  kl_divergence.rs - KL divergence functions for VB
  utils.rs         - normalize, log_normalize, split_x_lengths
  monitor.rs       - ConvergenceMonitor
  emissions/       - Concrete emission models
  vhmm/            - Variational HMM models
```

## Conventions

- All computations use `f64` exclusively
- Array operations use `ndarray` (version 0.16)
- Serialization via `serde` + `serde_json`
- Error handling via `thiserror` with `HmmError` enum
- Tests are inline `#[cfg(test)]` for unit tests, `tests/` for integration
- Examples use `plotly` crate for visualization
