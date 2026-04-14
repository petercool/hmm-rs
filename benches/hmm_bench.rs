use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use hmm_rs::algorithms;
use hmm_rs::prelude::*;
use ndarray::{Array1, Array2};

fn generate_gaussian_data(
    n_samples: usize,
    n_features: usize,
    _n_components: usize,
) -> Array2<f64> {
    let mut rng = rand::rng();
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    Array2::from_shape_fn((n_samples, n_features), |_| {
        rand::Rng::sample::<f64, _>(&mut rng, &normal)
    })
}

fn bench_forward_log(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_log");

    for &n_samples in &[100, 500, 1000, 5000] {
        for &n_components in &[2, 5, 10] {
            let startprob = Array1::from_elem(n_components, 1.0 / n_components as f64);
            let transmat =
                Array2::from_elem((n_components, n_components), 1.0 / n_components as f64);
            let log_frameprob = Array2::from_elem((n_samples, n_components), -2.0);

            group.bench_with_input(
                BenchmarkId::new(format!("nc{}", n_components), n_samples),
                &n_samples,
                |b, _| {
                    b.iter(|| {
                        algorithms::forward_log(
                            black_box(&startprob),
                            black_box(&transmat),
                            black_box(&log_frameprob),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_viterbi(c: &mut Criterion) {
    let mut group = c.benchmark_group("viterbi");

    for &n_samples in &[100, 500, 1000] {
        for &n_components in &[2, 5, 10] {
            let startprob = Array1::from_elem(n_components, 1.0 / n_components as f64);
            let transmat =
                Array2::from_elem((n_components, n_components), 1.0 / n_components as f64);
            let log_frameprob = Array2::from_elem((n_samples, n_components), -2.0);

            group.bench_with_input(
                BenchmarkId::new(format!("nc{}", n_components), n_samples),
                &n_samples,
                |b, _| {
                    b.iter(|| {
                        algorithms::viterbi(
                            black_box(&startprob),
                            black_box(&transmat),
                            black_box(&log_frameprob),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_gaussian_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_fit");
    group.sample_size(10);

    for &(n_samples, n_features, n_components) in &[(100, 2, 2), (500, 5, 3), (1000, 10, 5)] {
        let data = generate_gaussian_data(n_samples, n_features, n_components);
        let lengths = [n_samples];

        group.bench_with_input(
            BenchmarkId::new(
                format!("{}x{}_nc{}", n_samples, n_features, n_components),
                n_samples,
            ),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut model = GaussianHmm::gaussian(n_components, CovarianceType::Diag)
                        .with_n_iter(10)
                        .with_tol(1e-4);
                    model.fit(black_box(&data), black_box(&lengths)).unwrap();
                })
            },
        );
    }
    group.finish();
}

fn bench_categorical_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("categorical_fit");
    group.sample_size(10);

    for &n_samples in &[100, 500, 1000] {
        let mut rng = rand::rng();
        let n_features = 5;
        let data = Array2::from_shape_fn((n_samples, 1), |_| {
            (rand::Rng::random::<f64>(&mut rng) * n_features as f64).floor()
        });
        let lengths = [n_samples];

        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut model = CategoricalHmm::categorical(3)
                        .with_n_iter(10)
                        .with_tol(1e-4);
                    model.emission.n_features = Some(n_features);
                    model.fit(black_box(&data), black_box(&lengths)).unwrap();
                })
            },
        );
    }
    group.finish();
}

fn bench_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("score");

    for &n_samples in &[100, 500, 1000, 5000] {
        let data = generate_gaussian_data(n_samples, 5, 3);
        let lengths = [n_samples];

        // Pre-fit the model
        let mut model = GaussianHmm::gaussian(3, CovarianceType::Diag)
            .with_n_iter(10)
            .with_tol(1e-4);
        model.fit(&data, &lengths).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, _| b.iter(|| model.score(black_box(&data), black_box(&lengths)).unwrap()),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_forward_log,
    bench_viterbi,
    bench_gaussian_fit,
    bench_categorical_fit,
    bench_score,
);
criterion_main!(benches);
