//! # Model Selection: Finding the Optimal Number of States
//!
//! This example fits Gaussian HMMs with different numbers of hidden states
//! and uses AIC/BIC to select the best model. Demonstrates a common
//! workflow for determining model complexity.
//!
//! Run: `cargo run --example model_selection`

use hmm_rs::prelude::*;
use ndarray::Array2;
use plotly::common::{Mode, Title};
use plotly::layout::{Axis, Layout};
use plotly::{Plot, Scatter};

fn main() {
    println!("=== Model Selection via AIC/BIC ===\n");

    // Generate data from a known 3-state model
    let data = generate_three_state_data();
    let n = data.nrows();
    let lengths = [n];

    println!("Generated {} samples from a 3-state model", n);

    // Try models with 1 to 6 states
    let max_states = 6;
    let n_trials = 3; // Multiple random restarts per state count

    let mut best_scores: Vec<(usize, f64, f64, f64)> = Vec::new();

    for n_states in 1..=max_states {
        let mut best_ll = f64::NEG_INFINITY;
        let mut best_aic = f64::INFINITY;
        let mut best_bic = f64::INFINITY;

        for trial in 0..n_trials {
            let mut model = GaussianHmm::gaussian(n_states, CovarianceType::Diag)
                .with_n_iter(100)
                .with_tol(1e-6);

            match model.fit(&data, &lengths) {
                Ok(_) => {
                    let ll = model.score(&data, &lengths).unwrap_or(f64::NEG_INFINITY);
                    let aic = model.aic(&data, &lengths).unwrap_or(f64::INFINITY);
                    let bic = model.bic(&data, &lengths).unwrap_or(f64::INFINITY);

                    if ll > best_ll {
                        best_ll = ll;
                        best_aic = aic;
                        best_bic = bic;
                    }
                }
                Err(e) => {
                    eprintln!("  Trial {} with {} states failed: {}", trial, n_states, e);
                }
            }
        }

        println!(
            "  n_states={}: LL={:>10.2}  AIC={:>10.2}  BIC={:>10.2}",
            n_states, best_ll, best_aic, best_bic
        );

        best_scores.push((n_states, best_ll, best_aic, best_bic));
    }

    // Find optimal
    let optimal_aic = best_scores
        .iter()
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();
    let optimal_bic = best_scores
        .iter()
        .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
        .unwrap();

    println!(
        "\nOptimal by AIC: {} states (AIC={:.2})",
        optimal_aic.0, optimal_aic.2
    );
    println!(
        "Optimal by BIC: {} states (BIC={:.2})",
        optimal_bic.0, optimal_bic.3
    );

    // Create visualization
    create_model_selection_plot(&best_scores);
    println!("\nPlot saved to 'model_selection.html'");
}

fn generate_three_state_data() -> Array2<f64> {
    let mut rng = rand::rng();
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    let centers = [[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]];
    let stds = [1.0, 1.5, 0.8];
    let trans = [[0.93, 0.05, 0.02], [0.03, 0.92, 0.05], [0.04, 0.03, 0.93]];

    let n = 300;
    let mut data = Array2::<f64>::zeros((n, 2));
    let mut state = 0;

    for t in 0..n {
        data[[t, 0]] =
            centers[state][0] + stds[state] * rand::Rng::sample::<f64, _>(&mut rng, &normal);
        data[[t, 1]] =
            centers[state][1] + stds[state] * rand::Rng::sample::<f64, _>(&mut rng, &normal);

        let u: f64 = rand::Rng::random(&mut rng);
        let mut cumsum = 0.0;
        for next in 0..3 {
            cumsum += trans[state][next];
            if cumsum > u {
                state = next;
                break;
            }
        }
    }

    data
}

fn create_model_selection_plot(scores: &[(usize, f64, f64, f64)]) {
    let mut plot = Plot::new();

    let n_states: Vec<f64> = scores.iter().map(|s| s.0 as f64).collect();

    // Log-likelihood
    let ll_trace = Scatter::new(
        n_states.clone(),
        scores.iter().map(|s| s.1).collect::<Vec<_>>(),
    )
    .mode(Mode::LinesMarkers)
    .name("Log-Likelihood")
    .y_axis("y2");
    plot.add_trace(ll_trace);

    // AIC
    let aic_trace = Scatter::new(
        n_states.clone(),
        scores.iter().map(|s| s.2).collect::<Vec<_>>(),
    )
    .mode(Mode::LinesMarkers)
    .name("AIC");
    plot.add_trace(aic_trace);

    // BIC
    let bic_trace = Scatter::new(
        n_states.clone(),
        scores.iter().map(|s| s.3).collect::<Vec<_>>(),
    )
    .mode(Mode::LinesMarkers)
    .name("BIC");
    plot.add_trace(bic_trace);

    let layout = Layout::new()
        .title(Title::with_text(
            "Model Selection: AIC & BIC vs Number of States",
        ))
        .x_axis(Axis::new().title(Title::with_text("Number of Hidden States")))
        .y_axis(Axis::new().title(Title::with_text("Information Criterion")))
        .y_axis2(
            Axis::new()
                .title(Title::with_text("Log-Likelihood"))
                .overlaying("y")
                .side(plotly::common::AxisSide::Right),
        );

    plot.set_layout(layout);
    plot.write_html("model_selection.html");
}
