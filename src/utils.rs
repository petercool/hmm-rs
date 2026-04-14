use ndarray::{Array1, Array2, ArrayView2, s};

/// Normalize array in-place so that it sums to 1 along the given axis.
/// If a slice sums to zero, it is left unchanged (matching hmmlearn behavior
/// where zero-sum rows stay zero after division by 1).
pub fn normalize(a: &mut Array1<f64>) {
    let sum = a.sum();
    if sum != 0.0 {
        *a /= sum;
    }
}

/// Normalize each row of a 2D array in-place so rows sum to 1.
pub fn normalize_rows(a: &mut Array2<f64>) {
    for mut row in a.rows_mut() {
        let sum = row.sum();
        if sum == 0.0 {
            // hmmlearn sets zero-sum to 1 to avoid division by zero,
            // effectively leaving the row as zeros.
            continue;
        }
        row /= sum;
    }
}

/// Log-normalize array in-place so that sum(exp(a)) == 1 along the given axis.
/// Operates on a 1D array.
pub fn log_normalize(a: &mut Array1<f64>) {
    if a.len() == 1 {
        // Handle single-state degenerate case: normalize single -inf to 0.
        a[0] = 0.0;
        return;
    }
    let lse = logsumexp_slice(a.as_slice().unwrap());
    *a -= lse;
}

/// Log-normalize each row of a 2D array in-place.
pub fn log_normalize_rows(a: &mut Array2<f64>) {
    let ncols = a.ncols();
    for mut row in a.rows_mut() {
        if ncols == 1 {
            row[0] = 0.0;
            continue;
        }
        let lse = logsumexp_slice(row.as_slice().unwrap());
        row -= lse;
    }
}

/// Compute logsumexp over a slice.
fn logsumexp_slice(v: &[f64]) -> f64 {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return max;
    }
    let sum: f64 = v.iter().map(|&x| (x - max).exp()).sum();
    max + sum.ln()
}

/// Split concatenated observation matrix X into sub-sequences according to lengths.
/// Returns a Vec of array views, one per sequence.
pub fn split_x_lengths<'a>(x: &'a Array2<f64>, lengths: &[usize]) -> Vec<ArrayView2<'a, f64>> {
    let n_samples = x.nrows();
    let total: usize = lengths.iter().sum();
    assert_eq!(
        total, n_samples,
        "lengths sum {total} doesn't match n_samples {n_samples}"
    );

    let mut result = Vec::with_capacity(lengths.len());
    let mut start = 0;
    for &len in lengths {
        result.push(x.slice(s![start..start + len, ..]));
        start += len;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_normalize() {
        let mut a = array![1.0, 2.0, 3.0];
        normalize(&mut a);
        let expected = array![1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0];
        assert!((a.clone() - &expected).mapv(f64::abs).sum() < 1e-14);
    }

    #[test]
    fn test_normalize_zero_sum() {
        let mut a = array![0.0, 0.0, 0.0];
        normalize(&mut a);
        assert_eq!(a, array![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_rows() {
        let mut a = array![[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]];
        normalize_rows(&mut a);
        assert!((a[[0, 0]] - 1.0 / 6.0).abs() < 1e-14);
        assert_eq!(a[[1, 0]], 0.0);
    }

    #[test]
    fn test_log_normalize() {
        let mut a = array![0.0, 0.0, 0.0];
        log_normalize(&mut a);
        // After log_normalize, exp(a) should sum to 1
        let sum: f64 = a.mapv(f64::exp).sum();
        assert!((sum - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_log_normalize_single() {
        let mut a = array![f64::NEG_INFINITY];
        log_normalize(&mut a);
        assert_eq!(a[0], 0.0);
    }

    #[test]
    fn test_split_x_lengths() {
        let x = Array2::zeros((10, 3));
        let parts = split_x_lengths(&x, &[4, 6]);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].nrows(), 4);
        assert_eq!(parts[1].nrows(), 6);
    }
}
