#!/usr/bin/env python3
"""Generate JSON fixtures from hmmlearn for Rust parity testing.

Usage:
    pip install hmmlearn
    python tests/generate_fixtures.py

This creates JSON files in tests/fixtures/ with pre-set model parameters
and their expected outputs (score, decode, posteriors).
"""

import json
import numpy as np
from hmmlearn import hmm


def ndarray_to_list(arr):
    """Convert numpy array to nested Python lists for JSON serialization."""
    return arr.tolist()


def generate_categorical_fixture():
    """Generate fixture for CategoricalHMM with 2 states, 3 symbols."""
    model = hmm.CategoricalHMM(n_components=2, n_iter=1, random_state=42)
    model.startprob_ = np.array([0.6, 0.4])
    model.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
    model.emissionprob_ = np.array([
        [0.5, 0.3, 0.2],
        [0.1, 0.4, 0.5]
    ])
    model.n_features = 3

    # Test data
    X = np.array([[0], [1], [2], [0], [1], [2], [0], [0], [1], [2]])

    # Compute outputs
    log_prob = float(model.score(X))
    decode_log_prob, decode_states = model.decode(X)
    posteriors = model.predict_proba(X)

    fixture = {
        "model_type": "CategoricalHMM",
        "n_components": 2,
        "n_features": 3,
        "startprob": ndarray_to_list(model.startprob_),
        "transmat": ndarray_to_list(model.transmat_),
        "emissionprob": ndarray_to_list(model.emissionprob_),
        "X": ndarray_to_list(X),
        "expected": {
            "score": log_prob,
            "decode_log_prob": float(decode_log_prob),
            "decode_states": ndarray_to_list(decode_states),
            "posteriors": ndarray_to_list(posteriors),
        }
    }

    with open("tests/fixtures/categorical.json", "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"  categorical.json: score={log_prob:.6f}")


def generate_gaussian_full_fixture():
    """Generate fixture for GaussianHMM with full covariance."""
    model = hmm.GaussianHMM(
        n_components=3, covariance_type="full",
        n_iter=1, random_state=42
    )
    model.startprob_ = np.array([0.6, 0.3, 0.1])
    model.transmat_ = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.3, 0.3, 0.4]
    ])
    model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
    model.covars_ = np.tile(np.identity(2), (3, 1, 1))

    # Test data: deterministic points near each mean
    X = np.array([
        [0.1, 0.2], [-0.1, 0.3], [0.2, -0.1],
        [3.1, -2.9], [2.9, -3.1], [3.0, -3.0],
        [5.1, 10.2], [4.9, 9.8], [5.0, 10.0],
        [0.0, 0.0], [3.0, -3.0], [5.0, 10.0],
    ])

    log_prob = float(model.score(X))
    decode_log_prob, decode_states = model.decode(X)
    posteriors = model.predict_proba(X)

    fixture = {
        "model_type": "GaussianHMM",
        "n_components": 3,
        "n_features": 2,
        "covariance_type": "full",
        "startprob": ndarray_to_list(model.startprob_),
        "transmat": ndarray_to_list(model.transmat_),
        "means": ndarray_to_list(model.means_),
        "covars": ndarray_to_list(model.covars_),
        "X": ndarray_to_list(X),
        "expected": {
            "score": log_prob,
            "decode_log_prob": float(decode_log_prob),
            "decode_states": ndarray_to_list(decode_states),
            "posteriors": ndarray_to_list(posteriors),
        }
    }

    with open("tests/fixtures/gaussian_full.json", "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"  gaussian_full.json: score={log_prob:.6f}")


def generate_gaussian_diag_fixture():
    """Generate fixture for GaussianHMM with diagonal covariance."""
    model = hmm.GaussianHMM(
        n_components=2, covariance_type="diag",
        n_iter=1, random_state=42
    )
    model.startprob_ = np.array([0.7, 0.3])
    model.transmat_ = np.array([[0.8, 0.2], [0.3, 0.7]])
    model.means_ = np.array([[0.0, 0.0], [5.0, 5.0]])
    model.covars_ = np.array([[1.0, 2.0], [0.5, 1.5]])

    X = np.array([
        [0.1, 0.2], [-0.1, 0.3], [5.0, 5.1], [4.9, 5.0],
        [0.0, 0.0], [5.0, 5.0], [0.2, -0.1], [5.1, 4.9],
    ])

    log_prob = float(model.score(X))
    decode_log_prob, decode_states = model.decode(X)
    posteriors = model.predict_proba(X)

    fixture = {
        "model_type": "GaussianHMM",
        "n_components": 2,
        "n_features": 2,
        "covariance_type": "diag",
        "startprob": ndarray_to_list(model.startprob_),
        "transmat": ndarray_to_list(model.transmat_),
        "means": ndarray_to_list(model.means_),
        "covars": ndarray_to_list(model.covars_),
        "X": ndarray_to_list(X),
        "expected": {
            "score": log_prob,
            "decode_log_prob": float(decode_log_prob),
            "decode_states": ndarray_to_list(decode_states),
            "posteriors": ndarray_to_list(posteriors),
        }
    }

    with open("tests/fixtures/gaussian_diag.json", "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"  gaussian_diag.json: score={log_prob:.6f}")


def generate_poisson_fixture():
    """Generate fixture for PoissonHMM."""
    model = hmm.PoissonHMM(n_components=2, n_iter=1, random_state=42)
    model.startprob_ = np.array([0.6, 0.4])
    model.transmat_ = np.array([[0.8, 0.2], [0.3, 0.7]])
    model.lambdas_ = np.array([[2.0], [8.0]])

    X = np.array([[1], [2], [0], [3], [7], [9], [6], [8], [1], [2]])

    log_prob = float(model.score(X))
    decode_log_prob, decode_states = model.decode(X)

    fixture = {
        "model_type": "PoissonHMM",
        "n_components": 2,
        "n_features": 1,
        "startprob": ndarray_to_list(model.startprob_),
        "transmat": ndarray_to_list(model.transmat_),
        "lambdas": ndarray_to_list(model.lambdas_),
        "X": ndarray_to_list(X),
        "expected": {
            "score": log_prob,
            "decode_log_prob": float(decode_log_prob),
            "decode_states": ndarray_to_list(decode_states),
        }
    }

    with open("tests/fixtures/poisson.json", "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"  poisson.json: score={log_prob:.6f}")


def generate_gaussian_spherical_fixture():
    """Generate fixture for GaussianHMM with spherical covariance."""
    model = hmm.GaussianHMM(
        n_components=2, covariance_type="spherical",
        n_iter=1, random_state=42
    )
    model.startprob_ = np.array([0.7, 0.3])
    model.transmat_ = np.array([[0.8, 0.2], [0.3, 0.7]])
    model.means_ = np.array([[0.0, 0.0], [5.0, 5.0]])
    model.covars_ = np.array([1.5, 0.8])

    X = np.array([
        [0.1, 0.2], [-0.1, 0.3], [5.0, 5.1], [4.9, 5.0],
        [0.0, 0.0], [5.0, 5.0], [0.2, -0.1], [5.1, 4.9],
    ])

    log_prob = float(model.score(X))
    decode_log_prob, decode_states = model.decode(X)

    fixture = {
        "model_type": "GaussianHMM",
        "n_components": 2,
        "n_features": 2,
        "covariance_type": "spherical",
        "startprob": ndarray_to_list(model.startprob_),
        "transmat": ndarray_to_list(model.transmat_),
        "means": ndarray_to_list(model.means_),
        "covars": ndarray_to_list(model.covars_),
        "X": ndarray_to_list(X),
        "expected": {
            "score": log_prob,
            "decode_log_prob": float(decode_log_prob),
            "decode_states": ndarray_to_list(decode_states),
        }
    }

    with open("tests/fixtures/gaussian_spherical.json", "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"  gaussian_spherical.json: score={log_prob:.6f}")


def generate_gaussian_tied_fixture():
    """Generate fixture for GaussianHMM with tied covariance."""
    model = hmm.GaussianHMM(
        n_components=2, covariance_type="tied",
        n_iter=1, random_state=42
    )
    model.startprob_ = np.array([0.7, 0.3])
    model.transmat_ = np.array([[0.8, 0.2], [0.3, 0.7]])
    model.means_ = np.array([[0.0, 0.0], [5.0, 5.0]])
    model.covars_ = np.array([[1.5, 0.2], [0.2, 1.0]])

    X = np.array([
        [0.1, 0.2], [-0.1, 0.3], [5.0, 5.1], [4.9, 5.0],
        [0.0, 0.0], [5.0, 5.0], [0.2, -0.1], [5.1, 4.9],
    ])

    log_prob = float(model.score(X))
    decode_log_prob, decode_states = model.decode(X)

    fixture = {
        "model_type": "GaussianHMM",
        "n_components": 2,
        "n_features": 2,
        "covariance_type": "tied",
        "startprob": ndarray_to_list(model.startprob_),
        "transmat": ndarray_to_list(model.transmat_),
        "means": ndarray_to_list(model.means_),
        "covars": ndarray_to_list(model.covars_),
        "X": ndarray_to_list(X),
        "expected": {
            "score": log_prob,
            "decode_log_prob": float(decode_log_prob),
            "decode_states": ndarray_to_list(decode_states),
        }
    }

    with open("tests/fixtures/gaussian_tied.json", "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"  gaussian_tied.json: score={log_prob:.6f}")


def generate_multinomial_fixture():
    """Generate fixture for MultinomialHMM."""
    model = hmm.MultinomialHMM(n_components=2, n_trials=5,
                                n_iter=1, random_state=42)
    model.startprob_ = np.array([0.6, 0.4])
    model.transmat_ = np.array([[0.8, 0.2], [0.3, 0.7]])
    model.n_features = 3
    model.emissionprob_ = np.array([
        [0.5, 0.3, 0.2],
        [0.1, 0.4, 0.5]
    ])

    # Each row must sum to n_trials=5
    X = np.array([
        [3, 1, 1],
        [2, 2, 1],
        [1, 1, 3],
        [0, 2, 3],
        [2, 1, 2],
        [4, 1, 0],
    ])

    log_prob = float(model.score(X))

    fixture = {
        "model_type": "MultinomialHMM",
        "n_components": 2,
        "n_features": 3,
        "startprob": ndarray_to_list(model.startprob_),
        "transmat": ndarray_to_list(model.transmat_),
        "emissionprob": ndarray_to_list(model.emissionprob_),
        "X": ndarray_to_list(X),
        "expected": {
            "score": log_prob,
        }
    }

    with open("tests/fixtures/multinomial.json", "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"  multinomial.json: score={log_prob:.6f}")


if __name__ == "__main__":
    print("Generating hmm-rs parity fixtures from hmmlearn...")
    generate_categorical_fixture()
    generate_gaussian_full_fixture()
    generate_gaussian_diag_fixture()
    generate_gaussian_spherical_fixture()
    generate_gaussian_tied_fixture()
    generate_poisson_fixture()
    generate_multinomial_fixture()
    print("Done! Fixtures written to tests/fixtures/")
