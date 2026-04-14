//! Detailed parity report: print exact Rust vs Python values for all models.
//! Run with: cargo test --release --test parity_report -- --nocapture --ignored

use hmm_rs::prelude::*;
use ndarray::{Array1, Array2, Array3, Axis};
use std::fs;

fn load_fixture(name: &str) -> serde_json::Value {
    let path = format!("tests/fixtures/{}", name);
    let data = fs::read_to_string(&path).unwrap();
    serde_json::from_str(&data).unwrap()
}

fn json_to_array1(val: &serde_json::Value) -> Array1<f64> {
    Array1::from_vec(serde_json::from_value(val.clone()).unwrap())
}

fn json_to_array2(val: &serde_json::Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = serde_json::from_value(val.clone()).unwrap();
    let (nr, nc) = (rows.len(), rows[0].len());
    Array2::from_shape_fn((nr, nc), |(i, j)| rows[i][j])
}

fn json_to_array3(val: &serde_json::Value) -> Array3<f64> {
    let d: Vec<Vec<Vec<f64>>> = serde_json::from_value(val.clone()).unwrap();
    let (d0, d1, d2) = (d.len(), d[0].len(), d[0][0].len());
    Array3::from_shape_fn((d0, d1, d2), |(i, j, k)| d[i][j][k])
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b)
        .mapv(f64::abs)
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max)
}

#[test]
#[ignore] // verbose report; use parity_tests.rs for CI
fn full_parity_report() {
    println!("\n========================================================================");
    println!("           hmm-rs vs hmmlearn Numerical Parity Report");
    println!("========================================================================\n");

    let mut all_pass = true;

    // ── CategoricalHMM ──────────────────────────────────────────────
    {
        let fix = load_fixture("categorical.json");
        let mut model = CategoricalHmm::categorical(2);
        model.emission.n_features = Some(3);
        model.startprob_ = json_to_array1(&fix["startprob"]);
        model.transmat_ = json_to_array2(&fix["transmat"]);
        model.emission.emissionprob_ = Some(json_to_array2(&fix["emissionprob"]));
        model.fitted = true;

        let x = json_to_array2(&fix["X"]);
        let lengths = [x.nrows()];

        let rs_score = model.score(&x, &lengths).unwrap();
        let py_score = fix["expected"]["score"].as_f64().unwrap();
        let score_diff = (rs_score - py_score).abs();

        let rs_post = model.predict_proba(&x, &lengths).unwrap();
        let py_post = json_to_array2(&fix["expected"]["posteriors"]);
        let post_diff = max_abs_diff(&rs_post, &py_post);

        let (rs_lp, rs_states) = model.decode(&x, &lengths, None).unwrap();
        let py_lp = fix["expected"]["decode_log_prob"].as_f64().unwrap();
        let py_states: Vec<usize> = fix["expected"]["decode_states"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let states_match = rs_states.as_slice().unwrap() == &py_states[..];

        let pass = score_diff < 1e-9 && post_diff < 1e-9 && states_match;
        if !pass {
            all_pass = false;
        }

        println!(
            "CategoricalHMM (2 states, 3 symbols)        {}",
            if pass { "PASS" } else { "FAIL" }
        );
        println!(
            "  score:      Rust={:.12}  Python={:.12}  diff={:.2e}",
            rs_score, py_score, score_diff
        );
        println!(
            "  decode_lp:  Rust={:.12}  Python={:.12}  diff={:.2e}",
            rs_lp,
            py_lp,
            (rs_lp - py_lp).abs()
        );
        println!("  states:     match={}", states_match);
        println!("  posteriors: max_diff={:.2e}", post_diff);
        println!();
    }

    // ── GaussianHMM (full) ──────────────────────────────────────────
    {
        let fix = load_fixture("gaussian_full.json");
        let mut model = GaussianHmm::gaussian(3, CovarianceType::Full);
        model.emission.n_features = Some(2);
        model.startprob_ = json_to_array1(&fix["startprob"]);
        model.transmat_ = json_to_array2(&fix["transmat"]);
        model.emission.means_ = Some(json_to_array2(&fix["means"]));
        model.emission.covars_full_ = Some(json_to_array3(&fix["covars"]));
        model.fitted = true;

        let x = json_to_array2(&fix["X"]);
        let lengths = [x.nrows()];

        let rs_score = model.score(&x, &lengths).unwrap();
        let py_score = fix["expected"]["score"].as_f64().unwrap();
        let score_diff = (rs_score - py_score).abs();

        let rs_post = model.predict_proba(&x, &lengths).unwrap();
        let py_post = json_to_array2(&fix["expected"]["posteriors"]);
        let post_diff = max_abs_diff(&rs_post, &py_post);

        let (rs_lp, rs_states) = model.decode(&x, &lengths, None).unwrap();
        let py_lp = fix["expected"]["decode_log_prob"].as_f64().unwrap();
        let py_states: Vec<usize> = fix["expected"]["decode_states"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let states_match = rs_states.as_slice().unwrap() == &py_states[..];

        let pass = score_diff < 1e-9 && post_diff < 1e-9 && states_match;
        if !pass {
            all_pass = false;
        }

        println!(
            "GaussianHMM full (3 states, 2 features)      {}",
            if pass { "PASS" } else { "FAIL" }
        );
        println!(
            "  score:      Rust={:.12}  Python={:.12}  diff={:.2e}",
            rs_score, py_score, score_diff
        );
        println!(
            "  decode_lp:  Rust={:.12}  Python={:.12}  diff={:.2e}",
            rs_lp,
            py_lp,
            (rs_lp - py_lp).abs()
        );
        println!("  states:     match={}", states_match);
        println!("  posteriors: max_diff={:.2e}", post_diff);
        println!();
    }

    // ── GaussianHMM (diag) ──────────────────────────────────────────
    {
        let fix = load_fixture("gaussian_diag.json");
        let nc = 2;
        let nf = 2;
        let mut model = GaussianHmm::gaussian(nc, CovarianceType::Diag);
        model.emission.n_features = Some(nf);
        model.startprob_ = json_to_array1(&fix["startprob"]);
        model.transmat_ = json_to_array2(&fix["transmat"]);
        model.emission.means_ = Some(json_to_array2(&fix["means"]));
        let full = json_to_array3(&fix["covars"]);
        let mut diag = Array2::<f64>::zeros((nc, nf));
        for c in 0..nc {
            for f in 0..nf {
                diag[[c, f]] = full[[c, f, f]];
            }
        }
        model.emission.covars_diag_ = Some(diag);
        model.fitted = true;

        let x = json_to_array2(&fix["X"]);
        let rs_score = model.score(&x, &[x.nrows()]).unwrap();
        let py_score = fix["expected"]["score"].as_f64().unwrap();
        let diff = (rs_score - py_score).abs();
        let pass = diff < 1e-9;
        if !pass {
            all_pass = false;
        }

        println!(
            "GaussianHMM diag (2 states, 2 features)      {}",
            if pass { "PASS" } else { "FAIL" }
        );
        println!(
            "  score:      Rust={:.12}  Python={:.12}  diff={:.2e}",
            rs_score, py_score, diff
        );
        println!();
    }

    // ── GaussianHMM (spherical) ─────────────────────────────────────
    {
        let fix = load_fixture("gaussian_spherical.json");
        let nc = 2;
        let nf = 2;
        let mut model = GaussianHmm::gaussian(nc, CovarianceType::Spherical);
        model.emission.n_features = Some(nf);
        model.startprob_ = json_to_array1(&fix["startprob"]);
        model.transmat_ = json_to_array2(&fix["transmat"]);
        model.emission.means_ = Some(json_to_array2(&fix["means"]));
        let full = json_to_array3(&fix["covars"]);
        let mut sph = Array1::<f64>::zeros(nc);
        for c in 0..nc {
            let mut s = 0.0;
            for f in 0..nf {
                s += full[[c, f, f]];
            }
            sph[c] = s / nf as f64;
        }
        model.emission.covars_spherical_ = Some(sph);
        model.fitted = true;

        let x = json_to_array2(&fix["X"]);
        let rs_score = model.score(&x, &[x.nrows()]).unwrap();
        let py_score = fix["expected"]["score"].as_f64().unwrap();
        let diff = (rs_score - py_score).abs();
        let pass = diff < 1e-9;
        if !pass {
            all_pass = false;
        }

        println!(
            "GaussianHMM spherical (2 states, 2 features)  {}",
            if pass { "PASS" } else { "FAIL" }
        );
        println!(
            "  score:      Rust={:.12}  Python={:.12}  diff={:.2e}",
            rs_score, py_score, diff
        );
        println!();
    }

    // ── GaussianHMM (tied) ──────────────────────────────────────────
    {
        let fix = load_fixture("gaussian_tied.json");
        let nc = 2;
        let mut model = GaussianHmm::gaussian(nc, CovarianceType::Tied);
        model.emission.n_features = Some(2);
        model.startprob_ = json_to_array1(&fix["startprob"]);
        model.transmat_ = json_to_array2(&fix["transmat"]);
        model.emission.means_ = Some(json_to_array2(&fix["means"]));
        let full = json_to_array3(&fix["covars"]);
        model.emission.covars_tied_ = Some(full.index_axis(Axis(0), 0).to_owned());
        model.fitted = true;

        let x = json_to_array2(&fix["X"]);
        let rs_score = model.score(&x, &[x.nrows()]).unwrap();
        let py_score = fix["expected"]["score"].as_f64().unwrap();
        let diff = (rs_score - py_score).abs();
        let pass = diff < 1e-9;
        if !pass {
            all_pass = false;
        }

        println!(
            "GaussianHMM tied (2 states, 2 features)      {}",
            if pass { "PASS" } else { "FAIL" }
        );
        println!(
            "  score:      Rust={:.12}  Python={:.12}  diff={:.2e}",
            rs_score, py_score, diff
        );
        println!();
    }

    // ── PoissonHMM ──────────────────────────────────────────────────
    {
        let fix = load_fixture("poisson.json");
        let mut model = PoissonHmm::poisson(2);
        model.emission.n_features = Some(1);
        model.startprob_ = json_to_array1(&fix["startprob"]);
        model.transmat_ = json_to_array2(&fix["transmat"]);
        model.emission.lambdas_ = Some(json_to_array2(&fix["lambdas"]));
        model.fitted = true;

        let x = json_to_array2(&fix["X"]);
        let lengths = [x.nrows()];

        let rs_score = model.score(&x, &lengths).unwrap();
        let py_score = fix["expected"]["score"].as_f64().unwrap();
        let score_diff = (rs_score - py_score).abs();

        let (_, rs_states) = model.decode(&x, &lengths, None).unwrap();
        let py_states: Vec<usize> = fix["expected"]["decode_states"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let states_match = rs_states.as_slice().unwrap() == &py_states[..];

        let pass = score_diff < 1e-9 && states_match;
        if !pass {
            all_pass = false;
        }

        println!(
            "PoissonHMM (2 states, 1 feature)             {}",
            if pass { "PASS" } else { "FAIL" }
        );
        println!(
            "  score:      Rust={:.12}  Python={:.12}  diff={:.2e}",
            rs_score, py_score, score_diff
        );
        println!("  states:     match={}", states_match);
        println!();
    }

    // ── MultinomialHMM ──────────────────────────────────────────────
    {
        let fix = load_fixture("multinomial.json");
        let mut model = MultinomialHmm::multinomial(2);
        model.emission.n_features = Some(3);
        model.emission.n_trials = Some(5);
        model.startprob_ = json_to_array1(&fix["startprob"]);
        model.transmat_ = json_to_array2(&fix["transmat"]);
        model.emission.emissionprob_ = Some(json_to_array2(&fix["emissionprob"]));
        model.fitted = true;

        let x = json_to_array2(&fix["X"]);
        let rs_score = model.score(&x, &[x.nrows()]).unwrap();
        let py_score = fix["expected"]["score"].as_f64().unwrap();
        let diff = (rs_score - py_score).abs();
        let pass = diff < 1e-9;
        if !pass {
            all_pass = false;
        }

        println!(
            "MultinomialHMM (2 states, 3 symbols, 5 trials) {}",
            if pass { "PASS" } else { "FAIL" }
        );
        println!(
            "  score:      Rust={:.12}  Python={:.12}  diff={:.2e}",
            rs_score, py_score, diff
        );
        println!();
    }

    // ── Summary ─────────────────────────────────────────────────────
    println!("========================================================================");
    if all_pass {
        println!("ALL PARITY CHECKS PASSED (tolerance: 1e-9)");
    } else {
        println!("SOME PARITY CHECKS FAILED");
    }
    println!("========================================================================");

    assert!(all_pass, "Parity check failures detected");
}
