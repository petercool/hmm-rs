//! # Gaussian HMM for Sequential Clustering
//!
//! Demonstrates fitting a Gaussian HMM to 2D data with multiple clusters,
//! showing how covariance type affects the fit quality.
//!
//! Run: `cargo run --example gaussian_clustering`

use hmm_rs::prelude::*;
use ndarray::Array2;
use plotly::common::{Marker, MarkerSymbol, Mode, Title};
use plotly::layout::{Axis as PlotAxis, Layout};
use plotly::{Plot, Scatter};

fn main() {
    println!("=== Gaussian HMM Clustering ===\n");

    // Generate 2D sequential data with 3 distinct clusters
    let (data, true_states) = generate_clustered_data();
    let n = data.nrows();
    let lengths = [n];

    println!("Generated {} samples with 3 clusters", n);

    // Fit with different covariance types and compare
    let cov_types = [
        ("Full", CovarianceType::Full),
        ("Diagonal", CovarianceType::Diag),
        ("Spherical", CovarianceType::Spherical),
        ("Tied", CovarianceType::Tied),
    ];

    let mut results = Vec::new();

    for (name, cov_type) in &cov_types {
        let mut model = GaussianHmm::gaussian(3, *cov_type)
            .with_n_iter(100)
            .with_tol(1e-6);

        model.fit(&data, &lengths).unwrap();

        let score = model.score(&data, &lengths).unwrap();
        let aic = model.aic(&data, &lengths).unwrap();
        let bic = model.bic(&data, &lengths).unwrap();
        let (_, states) = model.decode(&data, &lengths, None).unwrap();

        // Compute accuracy (accounting for label permutation)
        let accuracy = compute_best_accuracy(&true_states, &states, 3);

        println!(
            "{:<12} Score: {:>10.2}  AIC: {:>10.2}  BIC: {:>10.2}  Acc: {:.1}%",
            name,
            score,
            aic,
            bic,
            accuracy * 100.0
        );

        let means = model.emission.means_.as_ref().unwrap().clone();
        results.push((*name, states, means, score, aic, bic));
    }

    // Create visualization
    create_clustering_plot(&data, &true_states, &results);
    println!("\nPlot saved to 'gaussian_clustering.html'");
}

fn generate_clustered_data() -> (Array2<f64>, Vec<usize>) {
    let mut rng = rand::rng();
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    // Three cluster centers
    let centers = [[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]];
    let stds = [0.8, 1.2, 0.6];

    let n_per_cluster = 80;
    let n_total = n_per_cluster * 3;
    let mut data = Array2::<f64>::zeros((n_total, 2));
    let mut states = Vec::with_capacity(n_total);

    // Generate in temporal order with regime switching
    let mut state = 0;
    let trans = [[0.96, 0.02, 0.02], [0.02, 0.96, 0.02], [0.02, 0.02, 0.96]];

    for t in 0..n_total {
        states.push(state);
        data[[t, 0]] =
            centers[state][0] + stds[state] * rand::Rng::sample::<f64, _>(&mut rng, &normal);
        data[[t, 1]] =
            centers[state][1] + stds[state] * rand::Rng::sample::<f64, _>(&mut rng, &normal);

        // Transition
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

    (data, states)
}

fn compute_best_accuracy(
    true_states: &[usize],
    pred_states: &ndarray::Array1<usize>,
    k: usize,
) -> f64 {
    // Try all permutations of labels to find best match
    let perms = generate_permutations(k);
    let n = true_states.len();

    perms
        .iter()
        .map(|perm| {
            let correct = (0..n)
                .filter(|&i| perm[pred_states[i]] == true_states[i])
                .count();
            correct as f64 / n as f64
        })
        .fold(0.0_f64, f64::max)
}

fn generate_permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut result = Vec::new();
    let sub_perms = generate_permutations(n - 1);
    for perm in sub_perms {
        for i in 0..=perm.len() {
            let mut new_perm = perm.clone();
            new_perm.insert(i, n - 1);
            result.push(new_perm);
        }
    }
    result
}

fn create_clustering_plot(
    data: &Array2<f64>,
    true_states: &[usize],
    results: &[(&str, ndarray::Array1<usize>, Array2<f64>, f64, f64, f64)],
) {
    let n = data.nrows();
    let colors = ["#e74c3c", "#2ecc71", "#3498db"];
    let state_names = ["Cluster A", "Cluster B", "Cluster C"];

    let mut plot = Plot::new();

    // True clusters
    for s in 0..3 {
        let x_vals: Vec<f64> = (0..n)
            .filter(|&t| true_states[t] == s)
            .map(|t| data[[t, 0]])
            .collect();
        let y_vals: Vec<f64> = (0..n)
            .filter(|&t| true_states[t] == s)
            .map(|t| data[[t, 1]])
            .collect();
        let trace = Scatter::new(x_vals, y_vals)
            .mode(Mode::Markers)
            .name(format!("True {}", state_names[s]))
            .marker(
                Marker::new()
                    .color(colors[s])
                    .size(6)
                    .symbol(MarkerSymbol::Circle),
            );
        plot.add_trace(trace);
    }

    // Add fitted means for each covariance type
    let symbols = [
        MarkerSymbol::Star,
        MarkerSymbol::Diamond,
        MarkerSymbol::Square,
        MarkerSymbol::Cross,
    ];

    for (idx, (name, _states, means, _score, aic, _bic)) in results.iter().enumerate() {
        let x_means: Vec<f64> = (0..3).map(|c| means[[c, 0]]).collect();
        let y_means: Vec<f64> = (0..3).map(|c| means[[c, 1]]).collect();
        let trace = Scatter::new(x_means, y_means)
            .mode(Mode::Markers)
            .name(format!("{} means (AIC={:.0})", name, aic))
            .marker(
                Marker::new()
                    .size(15)
                    .symbol(symbols[idx].clone())
                    .line(plotly::common::Line::new().width(2.0).color("black")),
            );
        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title(Title::with_text(
            "Gaussian HMM Clustering: Covariance Type Comparison",
        ))
        .x_axis(PlotAxis::new().title(Title::with_text("Feature 1")))
        .y_axis(PlotAxis::new().title(Title::with_text("Feature 2")));

    plot.set_layout(layout);
    plot.write_html("gaussian_clustering.html");
}
