//! Core HMM algorithms: forward, backward, Viterbi, and transition accumulation.
//!
//! This module is a direct port of hmmlearn's `_hmmc.cpp` (C++/pybind11).
//! All loop structures and operation orders match the reference implementation
//! to ensure numerical parity.

use ndarray::{Array1, Array2};

use crate::error::{HmmError, Result};

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Numerically stable log(exp(a) + exp(b)).
/// Matches hmmlearn's `logaddexp` in _hmmc.cpp.
#[inline]
pub fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        b
    } else if b == f64::NEG_INFINITY {
        a
    } else {
        let (a, b) = (a.max(b), a.min(b));
        a + (b - a).exp().ln_1p()
    }
}

/// Numerically stable log(sum(exp(v))).
/// Matches hmmlearn's `logsumexp` in _hmmc.cpp.
#[inline]
pub fn logsumexp(v: &[f64]) -> f64 {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return max;
    }
    let acc: f64 = v.iter().map(|&x| (x - max).exp()).sum();
    acc.ln() + max
}

// ---------------------------------------------------------------------------
// Forward algorithm — log implementation
// ---------------------------------------------------------------------------

/// Forward algorithm in log space.
///
/// # Arguments
/// * `startprob` - Initial state probabilities (n_components,). NOT in log space.
/// * `transmat` - Transition matrix (n_components, n_components). NOT in log space.
/// * `log_frameprob` - Log emission probabilities (n_samples, n_components).
///
/// # Returns
/// * `log_prob` - Total log probability.
/// * `fwdlattice` - Forward lattice in log space (n_samples, n_components).
pub fn forward_log(
    startprob: &Array1<f64>,
    transmat: &Array2<f64>,
    log_frameprob: &Array2<f64>,
) -> (f64, Array2<f64>) {
    let ns = log_frameprob.nrows();
    let nc = log_frameprob.ncols();

    let log_startprob = startprob.mapv(f64::ln);
    let log_transmat = transmat.mapv(f64::ln);

    let mut fwdlattice = Array2::<f64>::zeros((ns, nc));
    let mut buf = vec![0.0f64; nc];

    // t = 0
    for i in 0..nc {
        fwdlattice[[0, i]] = log_startprob[i] + log_frameprob[[0, i]];
    }

    // t = 1..ns
    for t in 1..ns {
        for j in 0..nc {
            for i in 0..nc {
                buf[i] = fwdlattice[[t - 1, i]] + log_transmat[[i, j]];
            }
            fwdlattice[[t, j]] = logsumexp(&buf) + log_frameprob[[t, j]];
        }
    }

    let log_prob = logsumexp(fwdlattice.row(ns - 1).as_slice().unwrap());
    (log_prob, fwdlattice)
}

// ---------------------------------------------------------------------------
// Backward algorithm — log implementation
// ---------------------------------------------------------------------------

/// Backward algorithm in log space.
///
/// # Arguments
/// * `startprob` - Initial state probabilities. NOT in log space.
/// * `transmat` - Transition matrix. NOT in log space.
/// * `log_frameprob` - Log emission probabilities (n_samples, n_components).
///
/// # Returns
/// * `bwdlattice` - Backward lattice in log space (n_samples, n_components).
pub fn backward_log(
    startprob: &Array1<f64>,
    transmat: &Array2<f64>,
    log_frameprob: &Array2<f64>,
) -> Array2<f64> {
    let ns = log_frameprob.nrows();
    let nc = log_frameprob.ncols();

    let _log_startprob = startprob.mapv(f64::ln);
    let log_transmat = transmat.mapv(f64::ln);

    let mut bwdlattice = Array2::<f64>::zeros((ns, nc));
    let mut buf = vec![0.0f64; nc];

    // t = ns - 1: bwd[ns-1, i] = 0 (log(1) = 0)
    // Already zero-initialized.

    for t in (0..ns - 1).rev() {
        for i in 0..nc {
            for j in 0..nc {
                buf[j] =
                    log_transmat[[i, j]] + log_frameprob[[t + 1, j]] + bwdlattice[[t + 1, j]];
            }
            bwdlattice[[t, i]] = logsumexp(&buf);
        }
    }

    bwdlattice
}

// ---------------------------------------------------------------------------
// Forward algorithm — scaling implementation
// ---------------------------------------------------------------------------

/// Forward algorithm with scaling factors.
///
/// # Arguments
/// * `startprob` - Initial state probabilities (n_components,).
/// * `transmat` - Transition matrix (n_components, n_components).
/// * `frameprob` - Emission probabilities (NOT log) (n_samples, n_components).
///
/// # Returns
/// * `log_prob` - Total log probability.
/// * `fwdlattice` - Scaled forward lattice (n_samples, n_components).
/// * `scaling` - Scaling factors (n_samples,).
pub fn forward_scaling(
    startprob: &Array1<f64>,
    transmat: &Array2<f64>,
    frameprob: &Array2<f64>,
) -> Result<(f64, Array2<f64>, Array1<f64>)> {
    let min_sum = 1e-300;
    let ns = frameprob.nrows();
    let nc = frameprob.ncols();

    let mut fwdlattice = Array2::<f64>::zeros((ns, nc));
    let mut scaling = Array1::<f64>::zeros(ns);
    let mut log_prob = 0.0f64;

    // t = 0
    for i in 0..nc {
        fwdlattice[[0, i]] = startprob[i] * frameprob[[0, i]];
    }
    let sum: f64 = fwdlattice.row(0).sum();
    if sum < min_sum {
        return Err(HmmError::ForwardUnderflow);
    }
    let scale = 1.0 / sum;
    scaling[0] = scale;
    log_prob -= scale.ln();
    for i in 0..nc {
        fwdlattice[[0, i]] *= scale;
    }

    // t = 1..ns
    for t in 1..ns {
        for j in 0..nc {
            let mut acc = 0.0;
            for i in 0..nc {
                acc += fwdlattice[[t - 1, i]] * transmat[[i, j]];
            }
            fwdlattice[[t, j]] = acc * frameprob[[t, j]];
        }
        let sum: f64 = fwdlattice.row(t).sum();
        if sum < min_sum {
            return Err(HmmError::ForwardUnderflow);
        }
        let scale = 1.0 / sum;
        scaling[t] = scale;
        log_prob -= scale.ln();
        for j in 0..nc {
            fwdlattice[[t, j]] *= scale;
        }
    }

    Ok((log_prob, fwdlattice, scaling))
}

// ---------------------------------------------------------------------------
// Backward algorithm — scaling implementation
// ---------------------------------------------------------------------------

/// Backward algorithm with scaling factors.
///
/// # Arguments
/// * `startprob` - Initial state probabilities.
/// * `transmat` - Transition matrix.
/// * `frameprob` - Emission probabilities (NOT log).
/// * `scaling` - Scaling factors from forward_scaling.
///
/// # Returns
/// * `bwdlattice` - Scaled backward lattice (n_samples, n_components).
pub fn backward_scaling(
    _startprob: &Array1<f64>,
    transmat: &Array2<f64>,
    frameprob: &Array2<f64>,
    scaling: &Array1<f64>,
) -> Array2<f64> {
    let ns = frameprob.nrows();
    let nc = frameprob.ncols();

    let mut bwdlattice = Array2::<f64>::zeros((ns, nc));

    // t = ns - 1
    for i in 0..nc {
        bwdlattice[[ns - 1, i]] = scaling[ns - 1];
    }

    for t in (0..ns - 1).rev() {
        for i in 0..nc {
            let mut acc = 0.0;
            for j in 0..nc {
                acc += transmat[[i, j]] * frameprob[[t + 1, j]] * bwdlattice[[t + 1, j]];
            }
            bwdlattice[[t, i]] = acc * scaling[t];
        }
    }

    bwdlattice
}

// ---------------------------------------------------------------------------
// Viterbi algorithm
// ---------------------------------------------------------------------------

/// Viterbi algorithm for MAP state sequence decoding.
///
/// # Arguments
/// * `startprob` - Initial state probabilities. NOT in log space.
/// * `transmat` - Transition matrix. NOT in log space.
/// * `log_frameprob` - Log emission probabilities (n_samples, n_components).
///
/// # Returns
/// * `log_prob` - Log probability of the most likely state sequence.
/// * `state_sequence` - Most likely state sequence (n_samples,).
pub fn viterbi(
    startprob: &Array1<f64>,
    transmat: &Array2<f64>,
    log_frameprob: &Array2<f64>,
) -> (f64, Array1<usize>) {
    let ns = log_frameprob.nrows();
    let nc = log_frameprob.ncols();

    let log_startprob = startprob.mapv(f64::ln);
    let log_transmat = transmat.mapv(f64::ln);

    let mut viterbi_lattice = Array2::<f64>::zeros((ns, nc));
    let mut state_sequence = Array1::<usize>::zeros(ns);

    // t = 0
    for i in 0..nc {
        viterbi_lattice[[0, i]] = log_startprob[i] + log_frameprob[[0, i]];
    }

    // t = 1..ns: forward pass
    for t in 1..ns {
        for i in 0..nc {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..nc {
                let val = viterbi_lattice[[t - 1, j]] + log_transmat[[j, i]];
                if val > max_val {
                    max_val = val;
                }
            }
            viterbi_lattice[[t, i]] = max_val + log_frameprob[[t, i]];
        }
    }

    // Backtrace
    let last_row = viterbi_lattice.row(ns - 1);
    let mut prev = last_row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    state_sequence[ns - 1] = prev;

    for t in (0..ns - 1).rev() {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_idx = 0;
        for i in 0..nc {
            let val = viterbi_lattice[[t, i]] + log_transmat[[i, prev]];
            if val > best_val || (val == best_val && i > best_idx) {
                best_val = val;
                best_idx = i;
            }
        }
        state_sequence[t] = best_idx;
        prev = best_idx;
    }

    let log_prob = viterbi_lattice[[ns - 1, state_sequence[ns - 1]]];
    (log_prob, state_sequence)
}

// ---------------------------------------------------------------------------
// Transition count accumulation — scaling implementation
// ---------------------------------------------------------------------------

/// Compute xi_sum (expected transition counts) using the scaling implementation.
///
/// # Arguments
/// * `fwdlattice` - Scaled forward lattice (n_samples, n_components).
/// * `transmat` - Transition matrix (n_components, n_components).
/// * `bwdlattice` - Scaled backward lattice (n_samples, n_components).
/// * `frameprob` - Emission probabilities (NOT log) (n_samples, n_components).
///
/// # Returns
/// * `xi_sum` - Expected transition counts (n_components, n_components).
pub fn compute_scaling_xi_sum(
    fwdlattice: &Array2<f64>,
    transmat: &Array2<f64>,
    bwdlattice: &Array2<f64>,
    frameprob: &Array2<f64>,
) -> Array2<f64> {
    let ns = frameprob.nrows();
    let nc = frameprob.ncols();

    let mut xi_sum = Array2::<f64>::zeros((nc, nc));

    for t in 0..ns - 1 {
        for i in 0..nc {
            for j in 0..nc {
                xi_sum[[i, j]] += fwdlattice[[t, i]]
                    * transmat[[i, j]]
                    * frameprob[[t + 1, j]]
                    * bwdlattice[[t + 1, j]];
            }
        }
    }

    xi_sum
}

// ---------------------------------------------------------------------------
// Transition count accumulation — log implementation
// ---------------------------------------------------------------------------

/// Compute log_xi_sum (expected transition counts in log space) using the log implementation.
///
/// # Arguments
/// * `fwdlattice` - Forward lattice in log space (n_samples, n_components).
/// * `transmat` - Transition matrix. NOT in log space.
/// * `bwdlattice` - Backward lattice in log space (n_samples, n_components).
/// * `log_frameprob` - Log emission probabilities (n_samples, n_components).
///
/// # Returns
/// * `log_xi_sum` - Log of expected transition counts (n_components, n_components).
pub fn compute_log_xi_sum(
    fwdlattice: &Array2<f64>,
    transmat: &Array2<f64>,
    bwdlattice: &Array2<f64>,
    log_frameprob: &Array2<f64>,
) -> Array2<f64> {
    let ns = log_frameprob.nrows();
    let nc = log_frameprob.ncols();

    let log_transmat = transmat.mapv(f64::ln);
    let log_prob = logsumexp(fwdlattice.row(ns - 1).as_slice().unwrap());

    let mut log_xi_sum = Array2::from_elem((nc, nc), f64::NEG_INFINITY);

    for t in 0..ns - 1 {
        for i in 0..nc {
            for j in 0..nc {
                let log_xi = fwdlattice[[t, i]]
                    + log_transmat[[i, j]]
                    + log_frameprob[[t + 1, j]]
                    + bwdlattice[[t + 1, j]]
                    - log_prob;
                log_xi_sum[[i, j]] = logaddexp(log_xi_sum[[i, j]], log_xi);
            }
        }
    }

    log_xi_sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_logaddexp_basic() {
        let a = 1.0_f64;
        let b = 2.0_f64;
        let expected = (a.exp() + b.exp()).ln();
        assert!((logaddexp(a, b) - expected).abs() < 1e-14);
    }

    #[test]
    fn test_logaddexp_neg_inf() {
        assert_eq!(logaddexp(f64::NEG_INFINITY, 1.0), 1.0);
        assert_eq!(logaddexp(1.0, f64::NEG_INFINITY), 1.0);
        assert_eq!(
            logaddexp(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn test_logaddexp_equal() {
        let a = 5.0;
        let result = logaddexp(a, a);
        let expected = a + 2.0_f64.ln();
        assert!((result - expected).abs() < 1e-14);
    }

    #[test]
    fn test_logsumexp_basic() {
        let v = [1.0, 2.0, 3.0];
        let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
        assert!((logsumexp(&v) - expected).abs() < 1e-14);
    }

    #[test]
    fn test_logsumexp_large_values() {
        // Should not overflow
        let v = [1000.0, 1001.0, 1002.0];
        let result = logsumexp(&v);
        assert!(result.is_finite());
        // The result should be close to 1002 + ln(1 + exp(-1) + exp(-2))
        let expected = 1002.0 + (1.0 + (-1.0_f64).exp() + (-2.0_f64).exp()).ln();
        assert!((result - expected).abs() < 1e-12);
    }

    #[test]
    fn test_logsumexp_all_neg_inf() {
        let v = [f64::NEG_INFINITY, f64::NEG_INFINITY];
        assert_eq!(logsumexp(&v), f64::NEG_INFINITY);
    }

    #[test]
    fn test_forward_backward_log_trivial() {
        // 2-state model with known parameters.
        // startprob = [0.6, 0.4]
        // transmat = [[0.7, 0.3], [0.4, 0.6]]
        // log_frameprob for 3 timesteps:
        //   t=0: [ln(0.5), ln(0.5)]
        //   t=1: [ln(0.1), ln(0.9)]
        //   t=2: [ln(0.8), ln(0.2)]
        let startprob = array![0.6, 0.4];
        let transmat = array![[0.7, 0.3], [0.4, 0.6]];
        let log_frameprob = array![
            [0.5_f64.ln(), 0.5_f64.ln()],
            [0.1_f64.ln(), 0.9_f64.ln()],
            [0.8_f64.ln(), 0.2_f64.ln()]
        ];

        let (log_prob, fwd) = forward_log(&startprob, &transmat, &log_frameprob);
        let bwd = backward_log(&startprob, &transmat, &log_frameprob);

        // log_prob should be finite and negative
        assert!(log_prob.is_finite());
        assert!(log_prob < 0.0);

        // fwdlattice shape
        assert_eq!(fwd.shape(), &[3, 2]);
        assert_eq!(bwd.shape(), &[3, 2]);

        // Verify: sum(exp(fwd[t] + bwd[t])) should be approximately exp(log_prob) for each t
        for t in 0..3 {
            let gamma_sum: f64 = (0..2)
                .map(|i| (fwd[[t, i]] + bwd[[t, i]]).exp())
                .sum();
            let prob = log_prob.exp();
            assert!(
                (gamma_sum - prob).abs() / prob < 1e-10,
                "t={}: gamma_sum={} vs prob={}",
                t,
                gamma_sum,
                prob
            );
        }
    }

    #[test]
    fn test_forward_scaling_matches_log() {
        let startprob = array![0.6, 0.4];
        let transmat = array![[0.7, 0.3], [0.4, 0.6]];
        let log_frameprob = array![
            [0.5_f64.ln(), 0.5_f64.ln()],
            [0.1_f64.ln(), 0.9_f64.ln()],
            [0.8_f64.ln(), 0.2_f64.ln()]
        ];
        let frameprob = log_frameprob.mapv(f64::exp);

        let (log_prob_log, _) = forward_log(&startprob, &transmat, &log_frameprob);
        let (log_prob_scaling, _, _) =
            forward_scaling(&startprob, &transmat, &frameprob).unwrap();

        assert!(
            (log_prob_log - log_prob_scaling).abs() < 1e-10,
            "log={} scaling={}",
            log_prob_log,
            log_prob_scaling
        );
    }

    #[test]
    fn test_viterbi_trivial() {
        let startprob = array![0.6, 0.4];
        let transmat = array![[0.7, 0.3], [0.4, 0.6]];
        let log_frameprob = array![
            [0.5_f64.ln(), 0.5_f64.ln()],
            [0.1_f64.ln(), 0.9_f64.ln()],
            [0.8_f64.ln(), 0.2_f64.ln()]
        ];

        let (log_prob, states) = viterbi(&startprob, &transmat, &log_frameprob);

        assert!(log_prob.is_finite());
        assert_eq!(states.len(), 3);
        // Each state should be 0 or 1
        for &s in states.iter() {
            assert!(s < 2);
        }
    }

    #[test]
    fn test_log_xi_sum() {
        let startprob = array![0.6, 0.4];
        let transmat = array![[0.7, 0.3], [0.4, 0.6]];
        let log_frameprob = array![
            [0.5_f64.ln(), 0.5_f64.ln()],
            [0.1_f64.ln(), 0.9_f64.ln()],
            [0.8_f64.ln(), 0.2_f64.ln()]
        ];

        let (_, fwd) = forward_log(&startprob, &transmat, &log_frameprob);
        let bwd = backward_log(&startprob, &transmat, &log_frameprob);

        let log_xi = compute_log_xi_sum(&fwd, &transmat, &bwd, &log_frameprob);

        // xi_sum should be (2, 2) and all entries finite
        assert_eq!(log_xi.shape(), &[2, 2]);
        for &v in log_xi.iter() {
            assert!(v.is_finite(), "non-finite in log_xi_sum: {}", v);
        }

        // exp(log_xi_sum) should be non-negative and finite
        let xi = log_xi.mapv(f64::exp);
        for &v in xi.iter() {
            assert!(v >= 0.0 && v.is_finite());
        }
        // Total sum of xi should be approximately n_samples - 1 = 2
        // (each timestep t has transition probabilities summing to 1)
        let total = xi.sum();
        let ns = log_frameprob.nrows();
        assert!(
            (total - (ns - 1) as f64).abs() < 1e-10,
            "xi total should be ~{}, got {}",
            ns - 1,
            total
        );
    }

    #[test]
    fn test_scaling_xi_sum_matches_log() {
        let startprob = array![0.6, 0.4];
        let transmat = array![[0.7, 0.3], [0.4, 0.6]];
        let log_frameprob = array![
            [0.5_f64.ln(), 0.5_f64.ln()],
            [0.1_f64.ln(), 0.9_f64.ln()],
            [0.8_f64.ln(), 0.2_f64.ln()]
        ];
        let frameprob = log_frameprob.mapv(f64::exp);

        // Log implementation
        let (_, fwd_log) = forward_log(&startprob, &transmat, &log_frameprob);
        let bwd_log = backward_log(&startprob, &transmat, &log_frameprob);
        let log_xi = compute_log_xi_sum(&fwd_log, &transmat, &bwd_log, &log_frameprob);
        let xi_from_log = log_xi.mapv(f64::exp);

        // Scaling implementation
        let (_, fwd_sc, scaling) =
            forward_scaling(&startprob, &transmat, &frameprob).unwrap();
        let bwd_sc = backward_scaling(&startprob, &transmat, &frameprob, &scaling);
        let xi_from_scaling = compute_scaling_xi_sum(&fwd_sc, &transmat, &bwd_sc, &frameprob);

        // They should match within tolerance
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (xi_from_log[[i, j]] - xi_from_scaling[[i, j]]).abs() < 1e-10,
                    "xi[{},{}]: log={} scaling={}",
                    i,
                    j,
                    xi_from_log[[i, j]],
                    xi_from_scaling[[i, j]]
                );
            }
        }
    }
}
