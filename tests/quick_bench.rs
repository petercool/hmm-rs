//! Quick performance comparison numbers.
//! Run with: cargo test --release --test quick_bench -- --nocapture --ignored

use hmm_rs::prelude::*;
use ndarray::Array2;
use std::time::Instant;

#[test]
#[ignore] // slow: runs full model fits with up to 100 EM iterations
fn performance_report() {
    let configs = [
        (100, 2, 2, 10),
        (500, 5, 3, 10),
        (1000, 10, 5, 10),
        (1000, 10, 5, 100),
    ];

    println!("\n=== hmm-rs Performance Report ===\n");
    println!("GaussianHMM.fit() (diag covariance):");

    for &(n_s, n_f, n_c, n_i) in &configs {
        let data = Array2::from_shape_fn((n_s, n_f), |(i, j)| {
            ((i * 7 + j * 13) % 100) as f64 / 50.0 - 1.0
        });
        let lengths = [n_s];

        let mut times = Vec::new();
        for _ in 0..5 {
            let mut model = GaussianHmm::gaussian(n_c, CovarianceType::Diag)
                .with_n_iter(n_i)
                .with_tol(1e-6)
                .with_random_state(42);
            let start = Instant::now();
            model.fit(&data, &lengths).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[2];
        println!(
            "  {:>5}x{} nc={} iter={:>3}: {:>8.2} ms",
            n_s, n_f, n_c, n_i, median
        );
    }

    println!("\nGaussianHMM.score():");
    for &n_s in &[100, 500, 1000, 5000] {
        let data = Array2::from_shape_fn((n_s, 5), |(i, j)| {
            ((i * 7 + j * 13) % 100) as f64 / 50.0 - 1.0
        });
        let lengths = [n_s];

        let mut model = GaussianHmm::gaussian(3, CovarianceType::Diag)
            .with_n_iter(10)
            .with_tol(1e-4)
            .with_random_state(42);
        model.fit(&data, &lengths).unwrap();

        let mut times = Vec::new();
        for _ in 0..20 {
            let start = Instant::now();
            model.score(&data, &lengths).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[10];
        println!("  {:>5} samples, nc=3, nf=5:     {:>8.4} ms", n_s, median);
    }

    println!("\nCategoricalHMM.fit():");
    for &n_s in &[100, 500, 1000] {
        let data = Array2::from_shape_fn((n_s, 1), |(i, _)| (i % 5) as f64);
        let lengths = [n_s];

        let mut times = Vec::new();
        for _ in 0..5 {
            let mut model = CategoricalHmm::categorical(3)
                .with_n_iter(10)
                .with_tol(1e-4)
                .with_random_state(42);
            model.emission.n_features = Some(5);
            let start = Instant::now();
            model.fit(&data, &lengths).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[2];
        println!("  {:>5} samples, nc=3, nf=5:     {:>8.2} ms", n_s, median);
    }
}
