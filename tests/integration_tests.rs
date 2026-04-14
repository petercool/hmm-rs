//! Comprehensive integration tests for hmm-rs.

use hmm_rs::prelude::*;
use ndarray::array;

// ==========================================================================
// Serialization round-trip tests
// ==========================================================================

#[test]
fn test_categorical_serialization_roundtrip() {
    let mut model = CategoricalHmm::categorical(2)
        .with_n_iter(10)
        .with_tol(1e-4);
    model.emission.n_features = Some(3);

    let x = array![[0.0], [1.0], [2.0], [0.0], [1.0], [0.0], [1.0], [2.0]];
    model.fit(&x, &[8]).unwrap();

    let score_before = model.score(&x, &[8]).unwrap();
    let json = serde_json::to_string(&model).unwrap();
    let loaded: CategoricalHmm = serde_json::from_str(&json).unwrap();
    let score_after = loaded.score(&x, &[8]).unwrap();

    assert!(
        (score_before - score_after).abs() < 1e-10,
        "score mismatch: {} vs {}",
        score_before,
        score_after
    );
}

#[test]
fn test_gaussian_serialization_roundtrip() {
    let mut model = GaussianHmm::gaussian(2, CovarianceType::Diag)
        .with_n_iter(10)
        .with_tol(1e-4);

    let x = array![
        [0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [4.9, 5.1],
        [0.2, -0.1], [5.1, 4.9], [0.0, 0.2], [5.0, 5.2],
    ];
    model.fit(&x, &[8]).unwrap();

    let score_before = model.score(&x, &[8]).unwrap();
    let json = serde_json::to_string(&model).unwrap();
    let loaded: GaussianHmm = serde_json::from_str(&json).unwrap();
    let score_after = loaded.score(&x, &[8]).unwrap();

    assert!(
        (score_before - score_after).abs() < 1e-10,
        "score mismatch: {} vs {}",
        score_before,
        score_after
    );
}

#[test]
fn test_poisson_serialization_roundtrip() {
    let mut model = PoissonHmm::poisson(2)
        .with_n_iter(10)
        .with_tol(1e-4);

    let x = array![[1.0], [0.0], [2.0], [5.0], [6.0], [4.0], [1.0], [0.0]];
    model.fit(&x, &[8]).unwrap();

    let score_before = model.score(&x, &[8]).unwrap();
    let json = serde_json::to_string(&model).unwrap();
    let loaded: PoissonHmm = serde_json::from_str(&json).unwrap();
    let score_after = loaded.score(&x, &[8]).unwrap();

    assert!((score_before - score_after).abs() < 1e-10);
}

// ==========================================================================
// Edge case tests
// ==========================================================================

#[test]
fn test_single_state_categorical() {
    let mut model = CategoricalHmm::categorical(1)
        .with_n_iter(10)
        .with_tol(1e-4);
    model.emission.n_features = Some(3);

    let x = array![[0.0], [1.0], [2.0], [0.0], [1.0]];
    model.fit(&x, &[5]).unwrap();

    let score = model.score(&x, &[5]).unwrap();
    assert!(score.is_finite());

    let (_, states) = model.decode(&x, &[5], None).unwrap();
    assert!(states.iter().all(|&s| s == 0), "single-state model should decode all to state 0");
}

#[test]
fn test_single_sample_sequence() {
    let mut model = GaussianHmm::gaussian(2, CovarianceType::Diag)
        .with_n_iter(10)
        .with_tol(1e-4);

    // Train on normal data
    let x = array![
        [0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [4.9, 5.1],
        [0.2, -0.1], [5.1, 4.9], [0.0, 0.2], [5.0, 5.2],
    ];
    model.fit(&x, &[8]).unwrap();

    // Score a single-sample sequence
    let single = array![[2.5, 2.5]];
    let score = model.score(&single, &[1]).unwrap();
    assert!(score.is_finite());

    let (_, states) = model.decode(&single, &[1], None).unwrap();
    assert_eq!(states.len(), 1);
}

#[test]
fn test_length_one_sequences() {
    // Multiple sequences each of length 1 — no transitions
    let mut model = CategoricalHmm::categorical(2)
        .with_n_iter(20)
        .with_tol(1e-4);
    model.emission.n_features = Some(3);

    let x = array![[0.0], [1.0], [2.0], [0.0], [1.0]];
    let lengths = [1, 1, 1, 1, 1];

    model.fit(&x, &lengths).unwrap();
    let score = model.score(&x, &lengths).unwrap();
    assert!(score.is_finite());
}

// ==========================================================================
// Multi-sequence tests
// ==========================================================================

#[test]
fn test_gaussian_multi_sequence_fit() {
    let mut model = GaussianHmm::gaussian(2, CovarianceType::Full)
        .with_n_iter(50)
        .with_tol(1e-4);

    let x = array![
        // Sequence 1
        [0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 4.9],
        // Sequence 2
        [5.0, 5.0], [4.9, 5.1], [0.0, 0.0], [-0.1, 0.1],
        // Sequence 3
        [0.0, 0.1], [5.0, 5.0],
    ];
    let lengths = [4, 4, 2];

    model.fit(&x, &lengths).unwrap();
    let score = model.score(&x, &lengths).unwrap();
    assert!(score.is_finite());

    let (_, states) = model.decode(&x, &lengths, None).unwrap();
    assert_eq!(states.len(), 10);
}

// ==========================================================================
// Covariance type comparison tests
// ==========================================================================

#[test]
fn test_all_covariance_types_converge() {
    let x = array![
        [0.1, 0.2], [-0.1, 0.3], [0.2, -0.1], [0.0, 0.1],
        [5.1, 5.2], [4.9, 5.3], [5.2, 4.9], [5.0, 5.1],
        [0.0, 0.0], [5.0, 5.0], [0.1, -0.1], [4.8, 5.2],
    ];
    let lengths = [x.nrows()];

    for cov_type in [
        CovarianceType::Full,
        CovarianceType::Diag,
        CovarianceType::Tied,
        CovarianceType::Spherical,
    ] {
        let mut model = GaussianHmm::gaussian(2, cov_type)
            .with_n_iter(50)
            .with_tol(1e-4);

        model.fit(&x, &lengths).unwrap();
        let score = model.score(&x, &lengths).unwrap();
        assert!(
            score.is_finite(),
            "{:?} covariance produced non-finite score: {}",
            cov_type,
            score
        );
    }
}

// ==========================================================================
// Decoder comparison tests
// ==========================================================================

#[test]
fn test_viterbi_vs_map_consistency() {
    let mut model = GaussianHmm::gaussian(2, CovarianceType::Diag)
        .with_n_iter(20)
        .with_tol(1e-4);

    let x = array![
        [0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 4.9],
        [0.0, 0.0], [5.0, 5.0], [0.1, -0.1], [5.0, 5.1],
    ];
    model.fit(&x, &[8]).unwrap();

    let (_, viterbi_states) = model.decode(&x, &[8], Some(DecoderAlgorithm::Viterbi)).unwrap();
    let (_, map_states) = model.decode(&x, &[8], Some(DecoderAlgorithm::Map)).unwrap();

    // Viterbi and MAP should produce similar (not necessarily identical) results
    assert_eq!(viterbi_states.len(), map_states.len());

    // But they should agree on most states
    let agree: usize = (0..8)
        .filter(|&i| viterbi_states[i] == map_states[i])
        .count();
    assert!(
        agree >= 6,
        "Viterbi and MAP should agree on most states: {}/8",
        agree
    );
}

// ==========================================================================
// Implementation comparison (log vs scaling)
// ==========================================================================

#[test]
fn test_log_vs_scaling_score_match() {
    // Create model with known parameters
    let emission = GaussianEmissions::new(CovarianceType::Diag);
    let mut model_log = GaussianHmm::gaussian(2, CovarianceType::Diag)
        .with_implementation(Implementation::Log);
    let mut model_scaling = GaussianHmm::gaussian(2, CovarianceType::Diag)
        .with_implementation(Implementation::Scaling);

    // Set identical parameters on both
    let startprob = array![0.6, 0.4];
    let transmat = array![[0.7, 0.3], [0.4, 0.6]];
    let means = array![[0.0, 0.0], [5.0, 5.0]];
    let covars = array![[1.0, 1.0], [1.0, 1.0]];

    for model in [&mut model_log, &mut model_scaling] {
        model.startprob_ = startprob.clone();
        model.transmat_ = transmat.clone();
        model.emission.n_features = Some(2);
        model.emission.means_ = Some(means.clone());
        model.emission.covars_diag_ = Some(covars.clone());
        model.fitted = true;
    }

    let x = array![
        [0.1, 0.2], [0.0, -0.1], [5.0, 5.1], [4.9, 5.0],
        [0.2, 0.1], [5.1, 4.9],
    ];

    let score_log = model_log.score(&x, &[6]).unwrap();
    let score_scaling = model_scaling.score(&x, &[6]).unwrap();

    assert!(
        (score_log - score_scaling).abs() < 1e-8,
        "Log ({}) and scaling ({}) scores should match",
        score_log,
        score_scaling
    );
}

// ==========================================================================
// Convergence tests
// ==========================================================================

#[test]
fn test_gaussian_likelihood_non_decreasing() {
    let mut model = GaussianHmm::gaussian(2, CovarianceType::Diag)
        .with_n_iter(30)
        .with_tol(1e-12); // very tight to force many iterations

    let x = array![
        [0.0, 0.0], [0.1, 0.1], [0.2, -0.1], [0.0, 0.2],
        [5.0, 5.0], [5.1, 4.9], [4.9, 5.1], [5.0, 5.0],
        [0.1, 0.0], [5.0, 5.1], [-0.1, 0.1], [4.9, 5.0],
    ];
    model.fit(&x, &[12]).unwrap();

    let history: Vec<f64> = model.monitor_.history.iter().cloned().collect();
    assert!(history.len() >= 2, "should have at least 2 iterations");

    for i in 1..history.len() {
        // Allow tiny numerical wiggle
        assert!(
            history[i] >= history[i - 1] - 1e-6,
            "likelihood should be non-decreasing: iter {}: {} -> {}",
            i,
            history[i - 1],
            history[i]
        );
    }
}

// ==========================================================================
// Posterior validity tests
// ==========================================================================

#[test]
fn test_posteriors_sum_to_one() {
    let mut model = GaussianHmm::gaussian(3, CovarianceType::Diag)
        .with_n_iter(20)
        .with_tol(1e-4);

    let x = array![
        [0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [10.0, 0.0],
        [0.0, 0.0], [5.0, 5.0], [10.0, 0.0], [0.1, -0.1],
    ];
    model.fit(&x, &[8]).unwrap();

    let posteriors = model.predict_proba(&x, &[8]).unwrap();
    assert_eq!(posteriors.shape(), &[8, 3]);

    for t in 0..8 {
        let row_sum = posteriors.row(t).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-6,
            "posterior row {} should sum to 1, got {}",
            t,
            row_sum
        );
    }

    // All values should be in [0, 1]
    for &v in posteriors.iter() {
        assert!(v >= -1e-10 && v <= 1.0 + 1e-10, "posterior value out of range: {}", v);
    }
}

// ==========================================================================
// Stationary distribution tests
// ==========================================================================

#[test]
fn test_stationary_distribution() {
    let mut model = CategoricalHmm::categorical(3);
    model.emission.n_features = Some(2);
    model.startprob_ = array![0.33, 0.34, 0.33];
    model.transmat_ = array![
        [0.7, 0.2, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.1, 0.7]
    ];
    model.emission.emissionprob_ = Some(array![[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]]);
    model.fitted = true;

    let pi = model.get_stationary_distribution().unwrap();

    // Should sum to 1
    assert!((pi.sum() - 1.0).abs() < 1e-10);

    // All entries should be positive
    for &v in pi.iter() {
        assert!(v > 0.0);
    }

    // Should satisfy pi @ transmat = pi
    let nc = 3;
    for j in 0..nc {
        let mut val = 0.0;
        for i in 0..nc {
            val += pi[i] * model.transmat_[[i, j]];
        }
        assert!(
            (val - pi[j]).abs() < 1e-10,
            "stationary distribution should satisfy pi*T = pi"
        );
    }
}

// ==========================================================================
// GMM-specific tests
// ==========================================================================

#[test]
fn test_gmm_different_covariance_types() {
    // GMM needs more data than simple Gaussian HMM due to mixture components
    let x = array![
        [0.0, 0.0], [0.1, 0.1], [-0.1, 0.2], [0.2, -0.1], [0.0, 0.3],
        [5.0, 5.0], [5.1, 4.9], [4.9, 5.1], [5.2, 5.0], [5.0, 4.8],
        [0.1, 0.0], [5.0, 5.1], [-0.2, 0.1], [4.8, 5.2], [0.0, -0.1],
        [5.1, 5.1], [0.2, 0.2], [4.9, 4.9], [0.0, 0.0], [5.0, 5.0],
    ];
    let lengths = [x.nrows()];

    for cov_type in [
        CovarianceType::Diag,
        CovarianceType::Spherical,
        CovarianceType::Full,
        CovarianceType::Tied,
    ] {
        let mut model = GmmHmm::gmm(2, 2, cov_type)
            .with_n_iter(30)
            .with_tol(1e-4);

        model.fit(&x, &lengths).unwrap();
        let score = model.score(&x, &lengths).unwrap();
        assert!(
            score.is_finite(),
            "GMM with {:?} covariance should produce finite score, got {}",
            cov_type,
            score
        );
    }
}

// ==========================================================================
// Variational tests
// ==========================================================================

#[test]
fn test_variational_categorical_convergence() {
    let mut model = VariationalCategoricalHmm::variational_categorical(2)
        .with_n_iter(50)
        .with_tol(1e-8);
    model.emission.n_features = Some(3);

    let x = array![
        [0.0], [1.0], [2.0], [0.0], [1.0],
        [0.0], [0.0], [1.0], [2.0], [2.0],
        [1.0], [0.0], [1.0], [2.0], [0.0],
    ];
    model.fit(&x, &[15]).unwrap();

    // Model should have converged (or reached max iterations)
    assert!(model.monitor_.iter > 0);

    // Score should be finite
    let score = model.score(&x, &[15]).unwrap();
    assert!(score.is_finite());
}

#[test]
fn test_variational_gaussian_produces_finite_results() {
    let x = array![
        [0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 4.9],
        [0.0, 0.2], [5.0, 5.1], [-0.1, 0.0], [4.9, 5.0],
        [0.1, -0.1], [5.0, 5.0],
    ];

    let mut model = VariationalGaussianHmm::variational_gaussian(2, CovarianceType::Full)
        .with_n_iter(30)
        .with_tol(1e-6);

    model.fit(&x, &[10]).unwrap();

    let score = model.score(&x, &[10]).unwrap();
    assert!(score.is_finite());

    let (_, states) = model.decode(&x, &[10]).unwrap();
    assert_eq!(states.len(), 10);
}

// ==========================================================================
// AIC/BIC tests
// ==========================================================================

#[test]
fn test_aic_bic_ordering() {
    // More complex model should have better likelihood but may have worse BIC
    let x = array![
        [0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 4.9],
        [0.0, 0.2], [5.0, 5.1], [-0.1, 0.0], [4.9, 5.0],
        [0.1, -0.1], [5.0, 5.0], [0.0, 0.0], [5.1, 5.1],
    ];
    let lengths = [12];

    let mut model_2 = GaussianHmm::gaussian(2, CovarianceType::Diag)
        .with_n_iter(50)
        .with_tol(1e-4);
    model_2.fit(&x, &lengths).unwrap();

    let aic_2 = model_2.aic(&x, &lengths).unwrap();
    let bic_2 = model_2.bic(&x, &lengths).unwrap();

    assert!(aic_2.is_finite());
    assert!(bic_2.is_finite());
    // BIC >= AIC for n >= 8 (since ln(n) > 2 for n >= 8)
    assert!(
        bic_2 >= aic_2 - 1.0, // small tolerance
        "BIC ({}) should be >= AIC ({}) for n=12",
        bic_2,
        aic_2
    );
}

// ==========================================================================
// Sample and fit consistency
// ==========================================================================

#[test]
fn test_sample_then_fit_recovers_structure() {
    // Create a model with well-separated states
    let mut source = CategoricalHmm::categorical(2);
    source.emission.n_features = Some(4);
    source.startprob_ = array![0.8, 0.2];
    source.transmat_ = array![[0.9, 0.1], [0.1, 0.9]];
    source.emission.emissionprob_ = Some(array![
        [0.7, 0.2, 0.05, 0.05],  // State 0: mostly symbol 0
        [0.05, 0.05, 0.2, 0.7]   // State 1: mostly symbol 3
    ]);
    source.fitted = true;

    // Generate training data
    let mut rng = rand::rng();
    let (train_data, _) = source.sample(500, &mut rng, None).unwrap();

    // Fit a new model
    let mut learned = CategoricalHmm::categorical(2)
        .with_n_iter(100)
        .with_tol(1e-6);
    learned.emission.n_features = Some(4);
    learned.fit(&train_data, &[500]).unwrap();

    // The learned model should score the training data well
    let source_score = source.score(&train_data, &[500]).unwrap();
    let learned_score = learned.score(&train_data, &[500]).unwrap();

    // Learned model should achieve reasonable score on training data
    // (not necessarily better than source due to random init and local optima)
    assert!(
        learned_score.is_finite(),
        "learned score should be finite, got {}",
        learned_score
    );
    // The learned score should be in a reasonable range (not wildly worse)
    assert!(
        learned_score > source_score - 200.0,
        "learned score ({}) should be in reasonable range of source score ({})",
        learned_score,
        source_score
    );
}

// ==========================================================================
// Scaling vs Log coverage for multiple model types
// ==========================================================================

#[test]
fn test_scaling_vs_log_categorical() {
    let mut model_log = CategoricalHmm::categorical(2)
        .with_implementation(Implementation::Log);
    let mut model_sc = CategoricalHmm::categorical(2)
        .with_implementation(Implementation::Scaling);
    model_log.emission.n_features = Some(3);
    model_sc.emission.n_features = Some(3);

    let sp = array![0.6, 0.4];
    let tm = array![[0.7, 0.3], [0.4, 0.6]];
    let ep = array![[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]];
    for m in [&mut model_log, &mut model_sc] {
        m.startprob_ = sp.clone();
        m.transmat_ = tm.clone();
        m.emission.emissionprob_ = Some(ep.clone());
        m.fitted = true;
    }

    let x = array![[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]];
    let s_log = model_log.score(&x, &[6]).unwrap();
    let s_sc = model_sc.score(&x, &[6]).unwrap();
    assert!(
        (s_log - s_sc).abs() < 1e-8,
        "Categorical log ({}) vs scaling ({})",
        s_log,
        s_sc
    );
}

#[test]
fn test_scaling_vs_log_gaussian_all_covariance_types() {
    let x = array![
        [0.1, 0.2], [0.0, -0.1], [5.0, 5.1], [4.9, 5.0],
        [0.2, 0.1], [5.1, 4.9],
    ];

    for cov_type in [
        CovarianceType::Full,
        CovarianceType::Diag,
        CovarianceType::Tied,
        CovarianceType::Spherical,
    ] {
        // Fit with log, then score with both
        let mut model = GaussianHmm::gaussian(2, cov_type)
            .with_n_iter(10)
            .with_tol(1e-4)
            .with_random_state(42);
        model.fit(&x, &[6]).unwrap();

        let s_log = model.score(&x, &[6]).unwrap();
        model.implementation = Implementation::Scaling;
        let s_sc = model.score(&x, &[6]).unwrap();

        assert!(
            (s_log - s_sc).abs() < 1e-6,
            "{:?}: log ({}) vs scaling ({})",
            cov_type,
            s_log,
            s_sc
        );
    }
}

// ==========================================================================
// Reproducibility test (random_state)
// ==========================================================================

#[test]
fn test_random_state_reproducibility() {
    let x = array![
        [0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 4.9],
        [0.0, 0.2], [5.0, 5.1], [-0.1, 0.0], [4.9, 5.0],
    ];
    let lengths = [8];

    let mut model1 = GaussianHmm::gaussian(2, CovarianceType::Diag)
        .with_n_iter(20)
        .with_tol(1e-6)
        .with_random_state(123);
    model1.fit(&x, &lengths).unwrap();
    let score1 = model1.score(&x, &lengths).unwrap();

    let mut model2 = GaussianHmm::gaussian(2, CovarianceType::Diag)
        .with_n_iter(20)
        .with_tol(1e-6)
        .with_random_state(123);
    model2.fit(&x, &lengths).unwrap();
    let score2 = model2.score(&x, &lengths).unwrap();

    assert!(
        (score1 - score2).abs() < 1e-10,
        "same random_state should produce identical results: {} vs {}",
        score1,
        score2
    );
}

// ==========================================================================
// AIC/BIC correctness test
// ==========================================================================

#[test]
fn test_aic_bic_parameter_count() {
    // CategoricalHMM with nc=2, nf=3
    // Free params: s=(2-1)=1, t=2*(2-1)=2, e=2*(3-1)=4 => total=7
    let mut model = CategoricalHmm::categorical(2)
        .with_n_iter(5)
        .with_tol(1e-4);
    model.emission.n_features = Some(3);

    let x = array![[0.0], [1.0], [2.0], [0.0], [1.0], [2.0], [0.0], [1.0]];
    model.fit(&x, &[8]).unwrap();

    let score = model.score(&x, &[8]).unwrap();
    let aic = model.aic(&x, &[8]).unwrap();
    let bic = model.bic(&x, &[8]).unwrap();

    let expected_n_params = 7; // 1 + 2 + 4
    let expected_aic = -2.0 * score + 2.0 * expected_n_params as f64;
    let expected_bic = -2.0 * score + expected_n_params as f64 * (8.0_f64).ln();

    assert!(
        (aic - expected_aic).abs() < 1e-6,
        "AIC: got {} expected {} (n_params={})",
        aic,
        expected_aic,
        expected_n_params
    );
    assert!(
        (bic - expected_bic).abs() < 1e-6,
        "BIC: got {} expected {} (n_params={})",
        bic,
        expected_bic,
        expected_n_params
    );
}
