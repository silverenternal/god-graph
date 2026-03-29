//! SVD Decomposition for tensor analysis
//!
//! Implements Singular Value Decomposition: A = U Σ V^T
//! Uses the one-sided Jacobi algorithm for numerical stability.

use crate::tensor::DenseTensor;
use crate::tensor::TensorBase;
use crate::tensor::TensorError;

/// Perform SVD decomposition on a 2D tensor using Jacobi algorithm
///
/// # Arguments
///
/// * `tensor` - Input tensor with shape [m, n]
/// * `k` - Optional number of singular values/vectors to compute (for truncated SVD)
///
/// # Returns
///
/// A tuple (U, S, V) where:
/// - U: Left singular vectors with shape [m, k] or [m, min(m,n)]
/// - S: Singular values with shape [k] or [min(m,n)]
/// - V: Right singular vectors with shape [n, k] or [n, min(m,n)]
///
/// # Errors
///
/// Returns an error if the tensor is not 2D or computation fails
pub fn svd_decompose(
    tensor: &DenseTensor,
    k: Option<usize>,
) -> Result<(DenseTensor, DenseTensor, DenseTensor), TensorError> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(TensorError::DimensionMismatch {
            expected: 2,
            got: shape.len(),
        });
    }

    let (m, n) = (shape[0], shape[1]);
    let min_dim = std::cmp::min(m, n);
    let k = k.unwrap_or(min_dim);

    if k > min_dim {
        return Err(TensorError::ShapeMismatch {
            expected: vec![min_dim],
            got: vec![k],
        });
    }

    let data = tensor.data();

    // Work with A^T A for the Jacobi algorithm
    // Compute A^T A (n x n symmetric matrix)
    let mut ata = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let mut sum = 0.0;
            for l in 0..m {
                sum += data[l * n + i] * data[l * n + j];
            }
            ata[i * n + j] = sum;
            ata[j * n + i] = sum; // Symmetric
        }
    }

    // Initialize V as identity
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // One-sided Jacobi algorithm for eigendecomposition of A^T A
    let max_iter = 50;
    let tol = 1e-10;

    for _ in 0..max_iter {
        let mut converged = true;

        // Sweep through all pairs (p, q) with p < q
        for p in 0..n {
            for q in (p + 1)..n {
                let app = ata[p * n + p];
                let aqq = ata[q * n + q];
                let apq = ata[p * n + q];

                if apq.abs() < tol * (app * aqq).sqrt() {
                    continue;
                }
                converged = false;

                // Compute Jacobi rotation angle
                // tan(2*theta) = 2*a_pq / (a_qq - a_pp)
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply rotation to A^T A
                // Update diagonal elements
                let new_app = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                let new_aqq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                let new_apq = (c * c - s * s) * apq + s * c * (app - aqq);

                ata[p * n + p] = new_app;
                ata[q * n + q] = new_aqq;
                ata[p * n + q] = new_apq;
                ata[q * n + p] = new_apq;

                // Update off-diagonal elements in rows/columns p and q
                for r in 0..n {
                    if r != p && r != q {
                        let apr = ata[p * n + r];
                        let aqr = ata[q * n + r];
                        ata[p * n + r] = c * apr - s * aqr;
                        ata[r * n + p] = ata[p * n + r];
                        ata[q * n + r] = s * apr + c * aqr;
                        ata[r * n + q] = ata[q * n + r];
                    }
                }

                // Update eigenvector matrix V
                for i in 0..n {
                    let vip = v[i * n + p];
                    let viq = v[i * n + q];
                    v[i * n + p] = c * vip - s * viq;
                    v[i * n + q] = s * vip + c * viq;
                }
            }
        }

        if converged {
            break;
        }
    }

    // Extract singular values (square root of eigenvalues of A^T A)
    let mut s = vec![0.0; n];
    for i in 0..n {
        s[i] = if ata[i * n + i] > 0.0 {
            ata[i * n + i].sqrt()
        } else {
            0.0
        };
    }

    // Sort singular values and vectors in descending order
    for i in 0..n {
        for j in (i + 1)..n {
            if s[j] > s[i] {
                s.swap(i, j);
                // Swap columns of V
                for row in 0..n {
                    v.swap(row * n + i, row * n + j);
                }
            }
        }
    }

    // Compute U = A V S^{-1}
    let mut u_data = vec![0.0; m * k];
    for i in 0..k {
        if s[i] < 1e-10 {
            // Handle zero singular value
            for j in 0..m {
                u_data[j * k + i] = 0.0;
            }
            continue;
        }
        for j in 0..m {
            let mut sum = 0.0;
            for l in 0..n {
                sum += data[j * n + l] * v[l * n + i];
            }
            u_data[j * k + i] = sum / s[i];
        }
    }

    // Truncate to k components
    let u_tensor = DenseTensor::from_vec(u_data, vec![m, k]);
    let s_tensor = DenseTensor::from_vec(s[..k].to_vec(), vec![k]);
    let v_data: Vec<f64> = v.iter().take(n * k).cloned().collect();
    let v_tensor = DenseTensor::from_vec(v_data, vec![n, k]);

    Ok((u_tensor, s_tensor, v_tensor))
}

/// Low-rank approximation using SVD
///
/// # Arguments
///
/// * `tensor` - Input tensor to approximate
/// * `rank` - Target rank for approximation
///
/// # Returns
///
/// Low-rank approximation of the input tensor
pub fn low_rank_approx(tensor: &DenseTensor, rank: usize) -> Result<DenseTensor, TensorError> {
    let (u, s, v) = svd_decompose(tensor, Some(rank))?;

    // Reconstruct: A ≈ U @ diag(S) @ V^T
    let u_data = u.data();
    let s_data = s.data();
    let v_data = v.data();

    let (m, k) = (u.shape()[0], u.shape()[1]);
    let n = v.shape()[0];

    let mut result = vec![0.0; m * n];

    // U @ diag(S)
    let mut us = vec![0.0; m * k];
    for i in 0..m {
        for j in 0..k {
            us[i * k + j] = u_data[i * k + j] * s_data[j];
        }
    }

    // (U @ diag(S)) @ V^T
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                result[i * n + j] += us[i * k + l] * v_data[j * k + l];
            }
        }
    }

    Ok(DenseTensor::from_vec(result, vec![m, n]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd_decomposition() {
        let tensor = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        let (u, s, v) = svd_decompose(&tensor, None).unwrap();

        // Check shapes
        assert_eq!(u.shape(), &[3, 2]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(v.shape(), &[2, 2]);

        // Singular values should be positive and decreasing
        let s_data = s.data();
        assert!(s_data[0] > 0.0);
        assert!(s_data[1] > 0.0);
        assert!(s_data[0] >= s_data[1]);
    }

    #[test]
    fn test_low_rank_approx() {
        let original =
            DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2]);

        let approx = low_rank_approx(&original, 1).unwrap();

        // Approximation should have same shape
        assert_eq!(approx.shape(), &[4, 2]);

        // Approximation error should be reasonable
        let orig_data = original.data();
        let approx_data = approx.data();
        let mse: f64 = orig_data
            .iter()
            .zip(approx_data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / orig_data.len() as f64;

        assert!(mse < 1.0);
    }

    #[test]
    fn test_svd_reconstruction() {
        // Test that SVD can reconstruct the original matrix
        let original = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        let (u, s, v) = svd_decompose(&original, None).unwrap();

        // Reconstruct: A = U @ diag(S) @ V^T
        let u_data = u.data();
        let s_data = s.data();
        let v_data = v.data();

        let (m, k) = (u.shape()[0], u.shape()[1]);
        let n = v.shape()[0];

        let mut reconstructed = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    reconstructed[i * n + j] += u_data[i * k + l] * s_data[l] * v_data[j * k + l];
                }
            }
        }

        let orig_data = original.data();
        for (a, b) in orig_data.iter().zip(reconstructed.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Reconstruction failed: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_svd_orthogonality() {
        let tensor =
            DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2]);

        let (u, _s, v) = svd_decompose(&tensor, None).unwrap();

        // Check U^T U = I
        let u_data = u.data();
        let (m, k) = (u.shape()[0], u.shape()[1]);

        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0;
                for l in 0..m {
                    dot += u_data[l * k + i] * u_data[l * k + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-5,
                    "U orthogonality failed at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Check V^T V = I
        let v_data = v.data();
        let n = v.shape()[0];

        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0;
                for l in 0..n {
                    dot += v_data[l * k + i] * v_data[l * k + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-5,
                    "V orthogonality failed at ({}, {})",
                    i,
                    j
                );
            }
        }
    }
}
