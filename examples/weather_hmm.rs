//! # Weather Prediction with Categorical HMM
//!
//! Classic HMM example: hidden states are weather conditions (Sunny, Rainy),
//! and observations are activities (Walk, Shop, Clean).
//!
//! Run: `cargo run --example weather_hmm`

use hmm_rs::prelude::*;
use ndarray::{array, Array1, Array2};
use plotly::common::{Mode, Title};
use plotly::layout::{Axis, Layout};
use plotly::{Plot, Scatter};

fn main() {
    println!("=== Weather HMM (Categorical) ===\n");

    // Define a known weather HMM
    let mut model = CategoricalHmm::categorical(2)
        .with_n_iter(1)
        .with_tol(1e-10);

    model.emission.n_features = Some(3);

    // Set known parameters
    // States: 0=Sunny, 1=Rainy
    model.startprob_ = array![0.6, 0.4];
    model.transmat_ = array![[0.7, 0.3], [0.4, 0.6]];

    // Emissions: P(activity | weather)
    // Activities: 0=Walk, 1=Shop, 2=Clean
    model.emission.emissionprob_ = Some(array![
        [0.6, 0.3, 0.1], // Sunny: prefer walking
        [0.1, 0.4, 0.5]  // Rainy: prefer cleaning
    ]);
    model.fitted = true;

    // Generate a sequence
    let mut rng = rand::rng();
    let (observations, true_states) = model.sample(30, &mut rng, None).unwrap();

    let activity_names = ["Walk", "Shop", "Clean"];
    let weather_names = ["Sunny", "Rainy"];

    println!("Generated sequence of 30 days:");
    for t in 0..30 {
        println!(
            "  Day {:2}: Weather={:<6} Activity={}",
            t + 1,
            weather_names[true_states[t]],
            activity_names[observations[[t, 0]] as usize]
        );
    }

    // Now decode (pretend we only see activities)
    let (log_prob, decoded_states) =
        model.decode(&observations, &[30], None).unwrap();

    println!("\nDecoding results (log-prob: {:.4}):", log_prob);
    let mut correct = 0;
    for t in 0..30 {
        let match_str = if decoded_states[t] == true_states[t] {
            correct += 1;
            "ok"
        } else {
            "MISS"
        };
        println!(
            "  Day {:2}: True={:<6} Predicted={:<6} [{}]",
            t + 1,
            weather_names[true_states[t]],
            weather_names[decoded_states[t]],
            match_str
        );
    }
    println!("\nAccuracy: {}/{} ({:.1}%)", correct, 30, correct as f64 / 30.0 * 100.0);

    // Get posterior probabilities
    let posteriors = model.predict_proba(&observations, &[30]).unwrap();

    // Score the sequence
    let score = model.score(&observations, &[30]).unwrap();
    println!("Sequence log-probability: {:.4}", score);

    // Now demonstrate fitting from scratch
    println!("\n--- Training from scratch ---");
    let mut learner = CategoricalHmm::categorical(2)
        .with_n_iter(50)
        .with_tol(1e-4);
    learner.emission.n_features = Some(3);

    // Generate more training data
    let (train_data, _) = model.sample(500, &mut rng, None).unwrap();
    learner.fit(&train_data, &[500]).unwrap();

    let learned_ep = learner.emission.emissionprob_.as_ref().unwrap();
    println!("Learned emission probabilities:");
    for s in 0..2 {
        println!(
            "  State {}: Walk={:.3} Shop={:.3} Clean={:.3}",
            s, learned_ep[[s, 0]], learned_ep[[s, 1]], learned_ep[[s, 2]]
        );
    }

    // Create visualization
    create_weather_plot(&observations, &true_states, &decoded_states, &posteriors);
    println!("\nPlot saved to 'weather_hmm.html'");
}

fn create_weather_plot(
    observations: &Array2<f64>,
    true_states: &Array1<usize>,
    decoded_states: &Array1<usize>,
    posteriors: &Array2<f64>,
) {
    let n = observations.nrows();
    let days: Vec<f64> = (1..=n).map(|i| i as f64).collect();

    let mut plot = Plot::new();

    // True weather states
    let true_weather: Vec<f64> = true_states.iter().map(|&s| s as f64).collect();
    let true_trace = Scatter::new(days.clone(), true_weather)
        .mode(Mode::LinesMarkers)
        .name("True Weather")
        .line(plotly::common::Line::new().dash(plotly::common::DashType::Dash));
    plot.add_trace(true_trace);

    // Decoded weather states (offset slightly for visibility)
    let decoded_weather: Vec<f64> = decoded_states.iter().map(|&s| s as f64 + 0.05).collect();
    let decoded_trace = Scatter::new(days.clone(), decoded_weather)
        .mode(Mode::LinesMarkers)
        .name("Decoded Weather");
    plot.add_trace(decoded_trace);

    // P(Rainy) posterior
    let p_rainy: Vec<f64> = (0..n).map(|t| posteriors[[t, 1]]).collect();
    let posterior_trace = Scatter::new(days.clone(), p_rainy)
        .mode(Mode::Lines)
        .name("P(Rainy)")
        .y_axis("y2");
    plot.add_trace(posterior_trace);

    // Observations (activities)
    let activities: Vec<f64> = (0..n).map(|t| observations[[t, 0]]).collect();
    let obs_trace = Scatter::new(days.clone(), activities)
        .mode(Mode::Markers)
        .name("Activity (0=Walk,1=Shop,2=Clean)")
        .y_axis("y3");
    plot.add_trace(obs_trace);

    let layout = Layout::new()
        .title(Title::with_text("Weather HMM: Hidden States and Observations"))
        .x_axis(Axis::new().title(Title::with_text("Day")))
        .y_axis(Axis::new().title(Title::with_text("State (0=Sunny, 1=Rainy)")))
        .y_axis2(
            Axis::new()
                .title(Title::with_text("P(Rainy)"))
                .overlaying("y")
                .side(plotly::common::AxisSide::Right),
        );

    plot.set_layout(layout);
    plot.write_html("weather_hmm.html");
}
