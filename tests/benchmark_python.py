#!/usr/bin/env python3
"""Benchmark hmmlearn for comparison against hmm-rs.

Usage:
    python tests/benchmark_python.py
"""

import time
import numpy as np
from hmmlearn import hmm


def bench_gaussian_fit(n_samples, n_features, n_components, n_iter):
    """Benchmark GaussianHMM.fit()."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    times = []
    for _ in range(5):
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=42,
        )
        start = time.perf_counter()
        model.fit(X)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.median(times)


def bench_gaussian_score(n_samples, n_features, n_components):
    """Benchmark GaussianHMM.score()."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=10,
        random_state=42,
    )
    model.fit(X)

    times = []
    for _ in range(20):
        start = time.perf_counter()
        model.score(X)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.median(times)


def bench_categorical_fit(n_samples, n_features, n_components, n_iter):
    """Benchmark CategoricalHMM.fit()."""
    np.random.seed(42)
    X = np.random.randint(0, n_features, size=(n_samples, 1))

    times = []
    for _ in range(5):
        model = hmm.CategoricalHMM(
            n_components=n_components,
            n_iter=n_iter,
            random_state=42,
        )
        start = time.perf_counter()
        model.fit(X)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.median(times)


if __name__ == "__main__":
    print("Python hmmlearn benchmarks")
    print("=" * 60)

    configs = [
        (100, 2, 2, 10),
        (500, 5, 3, 10),
        (1000, 10, 5, 10),
        (1000, 10, 5, 100),
    ]

    print("\nGaussianHMM.fit() (diag covariance):")
    for n_s, n_f, n_c, n_i in configs:
        t = bench_gaussian_fit(n_s, n_f, n_c, n_i)
        print(f"  {n_s:>5}x{n_f} nc={n_c} iter={n_i:>3}: {t*1000:>8.2f} ms")

    print("\nGaussianHMM.score():")
    for n_s in [100, 500, 1000, 5000]:
        t = bench_gaussian_score(n_s, 5, 3)
        print(f"  {n_s:>5} samples, nc=3, nf=5:     {t*1000:>8.4f} ms")

    print("\nCategoricalHMM.fit():")
    for n_s in [100, 500, 1000]:
        t = bench_categorical_fit(n_s, 5, 3, 10)
        print(f"  {n_s:>5} samples, nc=3, nf=5:     {t*1000:>8.2f} ms")
