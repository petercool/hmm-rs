use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Monitors and reports convergence of the EM algorithm.
///
/// Port of hmmlearn's `ConvergenceMonitor`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMonitor {
    pub tol: f64,
    pub n_iter: usize,
    pub verbose: bool,
    pub history: VecDeque<f64>,
    pub iter: usize,
}

impl ConvergenceMonitor {
    pub fn new(tol: f64, n_iter: usize, verbose: bool) -> Self {
        Self {
            tol,
            n_iter,
            verbose,
            history: VecDeque::new(),
            iter: 0,
        }
    }

    /// Reset the monitor's state for a new fit.
    pub fn reset(&mut self) {
        self.iter = 0;
        self.history.clear();
    }

    /// Report the current log probability and advance the iteration counter.
    ///
    /// Matches hmmlearn's `ConvergenceMonitor.report`.
    pub fn report(&mut self, log_prob: f64) {
        if self.verbose {
            let delta = if let Some(&last) = self.history.back() {
                log_prob - last
            } else {
                f64::NAN
            };
            eprintln!("{:>10} {:>16.8} {:>+16.8}", self.iter + 1, log_prob, delta);
        }

        // Warn if not converging (matching hmmlearn's precision check).
        let precision = f64::EPSILON.sqrt();
        if let Some(&last) = self.history.back()
            && (log_prob - last) < -precision
        {
            log::warn!(
                "Model is not converging. Current: {} is not greater than {}. Delta is {}",
                log_prob,
                last,
                log_prob - last
            );
        }

        self.history.push_back(log_prob);
        self.iter += 1;
    }

    /// Whether the EM algorithm has converged.
    ///
    /// Converged if max iterations reached OR the improvement between the
    /// last two iterations is below the tolerance threshold.
    pub fn converged(&self) -> bool {
        self.iter == self.n_iter
            || (self.history.len() >= 2 && {
                let len = self.history.len();
                self.history[len - 1] - self.history[len - 2] < self.tol
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_by_tol() {
        let mut mon = ConvergenceMonitor::new(0.01, 100, false);
        mon.report(-100.0);
        assert!(!mon.converged());
        mon.report(-50.0);
        assert!(!mon.converged());
        mon.report(-49.999);
        assert!(mon.converged()); // delta = 0.001 < 0.01
    }

    #[test]
    fn test_convergence_by_iter() {
        let mut mon = ConvergenceMonitor::new(1e-10, 3, false);
        mon.report(-100.0);
        mon.report(-50.0);
        assert!(!mon.converged());
        mon.report(-10.0);
        assert!(mon.converged()); // iter == n_iter
    }

    #[test]
    fn test_reset() {
        let mut mon = ConvergenceMonitor::new(0.01, 10, false);
        mon.report(-100.0);
        mon.report(-50.0);
        assert_eq!(mon.iter, 2);
        mon.reset();
        assert_eq!(mon.iter, 0);
        assert!(mon.history.is_empty());
    }
}
