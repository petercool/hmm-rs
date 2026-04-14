//! # Sequence Generation and Visualization
//!
//! Demonstrates generating sequences from trained HMMs and visualizing
//! the state transitions and emission distributions.
//!
//! Run: `cargo run --example sequence_generation`

use hmm_rs::prelude::*;
use ndarray::{Array2, Array3, array};
use plotly::common::{Marker, Mode, Title};
use plotly::layout::{Axis, Layout};
use plotly::{Plot, Scatter};

fn main() {
    println!("=== HMM Sequence Generation ===\n");

    // 1. Create and demonstrate a Gaussian HMM
    demo_gaussian_hmm();

    // 2. Create and demonstrate a Poisson HMM
    demo_poisson_hmm();

    // 3. Demonstrate serialization round-trip
    demo_serialization();
}

fn demo_gaussian_hmm() {
    println!("--- Gaussian HMM: 2D Trajectory ---");

    let mut model = GaussianHmm::gaussian(3, CovarianceType::Full);

    // Set up a 3-state model with distinct means
    model.emission.n_features = Some(2);
    model.emission.means_ = Some(array![
        [0.0, 0.0], // State 0: origin
        [5.0, 0.0], // State 1: right
        [2.5, 4.0]  // State 2: top
    ]);
    model.emission.covars_full_ = Some(
        Array3::from_shape_vec(
            (3, 2, 2),
            vec![
                0.5, 0.2, 0.2, 0.5, // State 0: correlated
                1.0, 0.0, 0.0, 0.3, // State 1: elongated horizontal
                0.3, 0.0, 0.0, 1.0, // State 2: elongated vertical
            ],
        )
        .unwrap(),
    );

    model.startprob_ = array![0.5, 0.3, 0.2];
    model.transmat_ = array![[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]];
    model.fitted = true;

    // Generate multiple trajectories
    let mut rng = rand::rng();
    let mut plot = Plot::new();
    let colors = ["#e74c3c", "#2ecc71", "#3498db"];
    let state_names = ["Origin", "Right", "Top"];

    for traj in 0..5 {
        let (samples, states) = model.sample(50, &mut rng, None).unwrap();

        // Color each point by state
        for s in 0..3 {
            let x_vals: Vec<f64> = (0..50)
                .filter(|&t| states[t] == s)
                .map(|t| samples[[t, 0]])
                .collect();
            let y_vals: Vec<f64> = (0..50)
                .filter(|&t| states[t] == s)
                .map(|t| samples[[t, 1]])
                .collect();

            if !x_vals.is_empty() {
                let trace = Scatter::new(x_vals, y_vals)
                    .mode(Mode::Markers)
                    .marker(Marker::new().color(colors[s]).size(5))
                    .name(format!("Traj {} - {}", traj + 1, state_names[s]))
                    .show_legend(traj == 0);
                plot.add_trace(trace);
            }
        }

        // Draw trajectory path
        let path_x: Vec<f64> = (0..50).map(|t| samples[[t, 0]]).collect();
        let path_y: Vec<f64> = (0..50).map(|t| samples[[t, 1]]).collect();
        let path = Scatter::new(path_x, path_y)
            .mode(Mode::Lines)
            .line(plotly::common::Line::new().width(0.5).color("gray"))
            .name(format!("Path {}", traj + 1))
            .show_legend(false);
        plot.add_trace(path);
    }

    // Add mean markers
    let means = model.emission.means_.as_ref().unwrap();
    for s in 0..3 {
        let trace = Scatter::new(vec![means[[s, 0]]], vec![means[[s, 1]]])
            .mode(Mode::Markers)
            .marker(
                Marker::new()
                    .color(colors[s])
                    .size(20)
                    .symbol(plotly::common::MarkerSymbol::Star)
                    .line(plotly::common::Line::new().width(2.0).color("black")),
            )
            .name(format!("{} mean", state_names[s]));
        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title(Title::with_text("Gaussian HMM: Generated 2D Trajectories"))
        .x_axis(Axis::new().title(Title::with_text("X")))
        .y_axis(Axis::new().title(Title::with_text("Y")));
    plot.set_layout(layout);
    plot.write_html("sequence_generation.html");

    println!("  Generated 5 trajectories of 50 samples each");
    println!("  Plot saved to 'sequence_generation.html'\n");

    // Stationary distribution
    let pi = model.get_stationary_distribution().unwrap();
    println!(
        "  Stationary distribution: [{:.3}, {:.3}, {:.3}]",
        pi[0], pi[1], pi[2]
    );
}

fn demo_poisson_hmm() {
    println!("\n--- Poisson HMM: Event Counts ---");

    let mut model = PoissonHmm::poisson(2).with_n_iter(50).with_tol(1e-4);

    // Generate training data: low-rate and high-rate states
    let mut rng = rand::rng();
    let pois_low = rand_distr::Poisson::new(2.0).unwrap();
    let pois_high = rand_distr::Poisson::new(8.0).unwrap();

    let n = 200;
    let mut data = Array2::<f64>::zeros((n, 1));
    let mut state = 0;
    for t in 0..n {
        data[[t, 0]] = if state == 0 {
            rand::Rng::sample::<f64, _>(&mut rng, &pois_low)
        } else {
            rand::Rng::sample::<f64, _>(&mut rng, &pois_high)
        };
        // Transition
        let u: f64 = rand::Rng::random(&mut rng);
        if state == 0 && u < 0.05 {
            state = 1;
        } else if state == 1 && u < 0.08 {
            state = 0;
        }
    }

    model.fit(&data, &[n]).unwrap();

    let lambdas = model.emission.lambdas_.as_ref().unwrap();
    println!(
        "  Learned Poisson rates: [{:.2}, {:.2}]",
        lambdas[[0, 0]],
        lambdas[[1, 0]]
    );

    let score = model.score(&data, &[n]).unwrap();
    println!("  Log-likelihood: {:.2}", score);
}

fn demo_serialization() {
    println!("\n--- Serialization Round-Trip ---");

    // Create and fit a model
    let mut model = CategoricalHmm::categorical(2)
        .with_n_iter(10)
        .with_tol(1e-4);
    model.emission.n_features = Some(3);

    let data = array![
        [0.0],
        [1.0],
        [2.0],
        [0.0],
        [1.0],
        [2.0],
        [0.0],
        [0.0],
        [1.0],
        [2.0]
    ];
    model.fit(&data, &[10]).unwrap();

    let score_before = model.score(&data, &[10]).unwrap();

    // Serialize to JSON
    let json = serde_json::to_string(&model).unwrap();
    println!("  Serialized model: {} bytes", json.len());

    // Deserialize
    let loaded: CategoricalHmm = serde_json::from_str(&json).unwrap();
    let score_after = loaded.score(&data, &[10]).unwrap();

    println!("  Score before: {:.6}", score_before);
    println!("  Score after:  {:.6}", score_after);
    assert!(
        (score_before - score_after).abs() < 1e-10,
        "Scores should match after serialization round-trip"
    );
    println!("  Round-trip: PASSED (scores match exactly)");
}
