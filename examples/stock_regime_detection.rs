//! # Stock Market Regime Detection with Gaussian HMM
//!
//! This example demonstrates using a Gaussian HMM to detect market regimes
//! (bull/bear/sideways) from synthetic stock return data. The model identifies
//! hidden states that correspond to different volatility regimes.
//!
//! Run: `cargo run --example stock_regime_detection`

use hmm_rs::prelude::*;
use ndarray::{Array1, Array2};
use plotly::common::{Fill, Line, Mode, Title};
use plotly::layout::{Axis, Layout, RangeSlider, Shape, ShapeLayer, ShapeLine, ShapeType};
use plotly::{Bar, Plot, Scatter};

fn main() {
    println!("=== Stock Market Regime Detection ===\n");

    // Generate synthetic stock returns with regime switching
    let (returns, true_regimes) = generate_synthetic_returns();
    let n_samples = returns.nrows();

    println!("Generated {} days of synthetic stock returns", n_samples);
    println!(
        "True regimes: Bull={}, Bear={}, Sideways={}",
        true_regimes.iter().filter(|&&r| r == 0).count(),
        true_regimes.iter().filter(|&&r| r == 1).count(),
        true_regimes.iter().filter(|&&r| r == 2).count(),
    );

    // Fit a 3-state Gaussian HMM (bull, bear, sideways)
    let mut model = GaussianHmm::gaussian(3, CovarianceType::Full)
        .with_n_iter(100)
        .with_tol(1e-6);

    let lengths = [n_samples];
    model.fit(&returns, &lengths).unwrap();

    // Decode the most likely state sequence
    let (log_prob, predicted_states) = model.decode(&returns, &lengths, None).unwrap();
    println!("\nModel log-likelihood: {:.2}", log_prob);

    // Print learned parameters
    let means = model.emission.means_.as_ref().unwrap();
    println!("\nLearned state means (return, volatility proxy):");
    for i in 0..3 {
        println!(
            "  State {}: [{:.4}, {:.4}]",
            i,
            means[[i, 0]],
            means[[i, 1]]
        );
    }

    // Compute model quality metrics
    let aic = model.aic(&returns, &lengths).unwrap();
    let bic = model.bic(&returns, &lengths).unwrap();
    println!("\nAIC: {:.2}", aic);
    println!("BIC: {:.2}", bic);

    // Posterior probabilities
    let posteriors = model.predict_proba(&returns, &lengths).unwrap();

    // Serialize the model
    let json = serde_json::to_string_pretty(&model).unwrap();
    println!("\nModel serialized to JSON ({} bytes)", json.len());

    // Create Plotly visualizations
    create_regime_plot(&returns, &predicted_states, &posteriors, &true_regimes);

    println!("\nPlot saved to 'stock_regimes.html'");
}

fn generate_synthetic_returns() -> (Array2<f64>, Vec<usize>) {
    let mut rng = rand::rng();
    let n = 500;

    // Three regimes: bull (high return, low vol), bear (negative return, high vol), sideways
    let regime_means = [[0.001, 0.005], [-0.002, 0.02], [0.0, 0.008]];
    let regime_vols = [0.01, 0.03, 0.005];

    let mut returns = Array2::<f64>::zeros((n, 2));
    let mut regimes = Vec::with_capacity(n);
    let mut state = 0usize;

    // Transition probabilities: high self-transition (sticky regimes)
    let trans = [[0.95, 0.03, 0.02], [0.03, 0.94, 0.03], [0.04, 0.02, 0.94]];

    for t in 0..n {
        regimes.push(state);

        // Generate return and volatility proxy
        let normal = rand_distr::Normal::new(regime_means[state][0], regime_vols[state]).unwrap();
        let vol_normal =
            rand_distr::Normal::new(regime_means[state][1], regime_vols[state] * 0.5).unwrap();
        returns[[t, 0]] = rand::Rng::sample(&mut rng, &normal);
        returns[[t, 1]] = rand::Rng::sample::<f64, _>(&mut rng, &vol_normal).abs();

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

    (returns, regimes)
}

fn create_regime_plot(
    returns: &Array2<f64>,
    states: &Array1<usize>,
    posteriors: &Array2<f64>,
    _true_regimes: &[usize],
) {
    let n = returns.nrows();
    let days: Vec<f64> = (0..n).map(|i| i as f64).collect();

    // Compute cumulative returns for price series
    let mut prices = vec![100.0_f64];
    for t in 0..n {
        let next = prices.last().unwrap() * (1.0 + returns[[t, 0]]);
        prices.push(next);
    }
    let price_days: Vec<f64> = (0..=n).map(|i| i as f64).collect();

    let mut plot = Plot::new();

    // 1. Price series colored by regime
    let regime_colors = [
        "rgba(46,204,113,0.3)",
        "rgba(231,76,60,0.3)",
        "rgba(52,152,219,0.3)",
    ];
    let regime_names = ["Bull", "Bear", "Sideways"];

    // Add price trace
    let price_trace = Scatter::new(price_days.clone(), prices.clone())
        .mode(Mode::Lines)
        .name("Price")
        .line(Line::new().color("black").width(1.5));
    plot.add_trace(price_trace);

    // Add colored background regions for each regime
    // Group consecutive same-state regions
    let mut shapes = Vec::new();
    let mut region_start = 0;
    let mut current_state = states[0];
    for t in 1..n {
        if states[t] != current_state || t == n - 1 {
            let end = if t == n - 1 { t } else { t - 1 };
            shapes.push(
                Shape::new()
                    .shape_type(ShapeType::Rect)
                    .x0(region_start as f64)
                    .x1(end as f64)
                    .y0(0.0)
                    .y1(1.0)
                    .y_ref("paper")
                    .fill_color(regime_colors[current_state])
                    .layer(ShapeLayer::Below)
                    .line(ShapeLine::new().width(0.0)),
            );
            region_start = t;
            current_state = states[t];
        }
    }

    // 2. Posterior probabilities as stacked area
    for s in 0..3 {
        let probs: Vec<f64> = (0..n).map(|t| posteriors[[t, s]]).collect();
        let trace = Scatter::new(days.clone(), probs)
            .mode(Mode::Lines)
            .name(format!("P({})", regime_names[s]))
            .fill(Fill::ToZeroY)
            .visible(plotly::common::Visible::LegendOnly);
        plot.add_trace(trace);
    }

    // 3. Returns histogram by regime
    for s in 0..3 {
        let regime_returns: Vec<f64> = (0..n)
            .filter(|&t| states[t] == s)
            .map(|t| returns[[t, 0]])
            .collect();
        let trace = Bar::new(
            (0..regime_returns.len())
                .map(|i| i as f64)
                .collect::<Vec<_>>(),
            regime_returns,
        )
        .name(format!("{} returns", regime_names[s]))
        .visible(plotly::common::Visible::LegendOnly);
        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title(Title::with_text(
            "Stock Market Regime Detection with Gaussian HMM",
        ))
        .x_axis(
            Axis::new()
                .title(Title::with_text("Trading Day"))
                .range_slider(RangeSlider::new().visible(true)),
        )
        .y_axis(Axis::new().title(Title::with_text("Price")))
        .shapes(shapes);

    plot.set_layout(layout);
    plot.write_html("stock_regimes.html");
}
