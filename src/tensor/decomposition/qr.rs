//! QR Decomposition for tensor orthogonalization
//!
//! Implements QR decomposition using modified Gram-Schmidt with reorthogonalization.
//! For a matrix A ∈ R^(m×n) with m ≥ n, computes A = QR where:
//! - Q ∈ R^(m×n) has orthonormal columns (Q^T Q = I)
//! - R ∈ R^(n×n) is upper triangular with non-negative diagonal entries

use crate::tensor::DenseTensor;
use crate::tensor::TensorBase;
use crate::tensor::TensorError;

/// Perform QR decomposition on a 2D tensor using modified Gram-Schmidt with reorthogonalization
pub fn qr_decompose(tensor: &DenseTensor) -> Result<(DenseTensor, DenseTensor), TensorError> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(TensorError::DimensionMismatch {
            expected: 2,
            got: shape.len(),
        });
    }

    let (m, n) = (shape[0], shape[1]);
    if m < n {
        return Err(TensorError::ShapeMismatch {
            expected: vec![n, m],
            got: vec![m, n],
        });
    }

    let data = tensor.data();

    // Initialize Q as a copy of A (will be orthogonalized in place)
    let mut q = data.to_vec();

    // Initialize R as zeros
    let mut r = vec![0.0; n * n];

    // Modified Gram-Schmidt with reorthogonalization for numerical stability
    for j in 0..n {
        // Copy column j to work with
        let mut col_norm = 0.0;
        for k in 0..m {
            col_norm += q[k * n + j] * q[k * n + j];
        }
        col_norm = col_norm.sqrt();

        // Modified Gram-Schmidt: orthogonalize against previous columns
        for i in 0..j {
            // Compute r[i,j] = q[:,i]^T * q[:,j]
            let mut dot = 0.0;
            for k in 0..m {
                dot += q[k * n + i] * q[k * n + j];
            }
            r[i * n + j] = dot;

            // Subtract projection: q[:,j] -= r[i,j] * q[:,i]
            for k in 0..m {
                q[k * n + j] -= dot * q[k * n + i];
            }
        }

        // Reorthogonalization pass (critical for numerical stability)
        // Check if we lost orthogonality and correct
        for i in 0..j {
            let mut dot = 0.0;
            for k in 0..m {
                dot += q[k * n + i] * q[k * n + j];
            }
            if dot.abs() > 1e-14 {
                // Second correction
                for k in 0..m {
                    q[k * n + j] -= dot * q[k * n + i];
                }
                r[i * n + j] += dot;
            }
        }

        // Normalize q[:, j]
        let mut new_norm = 0.0;
        for k in 0..m {
            new_norm += q[k * n + j] * q[k * n + j];
        }
        new_norm = new_norm.sqrt();

        // Use relative norm change to detect linear dependence
        let rel_norm = if col_norm > 1e-14 { new_norm / col_norm } else { new_norm };

        if rel_norm > 1e-12 && new_norm > 1e-14 {
            r[j * n + j] = new_norm;
            for k in 0..m {
                q[k * n + j] /= new_norm;
            }
        } else {
            // Column is linearly dependent or nearly so
            r[j * n + j] = 0.0;
            // Set column to zero to avoid NaN
            for k in 0..m {
                q[k * n + j] = 0.0;
            }
        }
    }

    // Ensure R has non-negative diagonal
    for i in 0..n {
        if r[i * n + i] < 0.0 {
            // Multiply row i of R by -1
            for j in i..n {
                r[i * n + j] = -r[i * n + j];
            }
            // Multiply column i of Q by -1
            for j in 0..m {
                q[j * n + i] = -q[j * n + i];
            }
        }
    }

    let q_tensor = DenseTensor::from_vec(q, vec![m, n]);
    let r_tensor = DenseTensor::from_vec(r, vec![n, n]);

    Ok((q_tensor, r_tensor))
}

/// Orthogonalize a matrix using QR decomposition
pub fn orthogonalize(tensor: &DenseTensor) -> Result<DenseTensor, TensorError> {
    let (q, _r) = qr_decompose(tensor)?;
    Ok(q)
}

/// Orthogonalize a matrix in-place using QR decomposition
///
/// This function performs orthogonalization directly on the input data slice,
/// avoiding unnecessary allocations. The result overwrites the input data.
///
/// # Arguments
/// * `data` - Mutable slice of tensor data (will be modified in place)
/// * `shape` - Tensor shape [m, n] where m >= n
///
/// # Returns
/// * `Ok(f64)` - Orthogonalization error (deviation from perfect orthogonality)
/// * `Err(TensorError)` - Error if the shape is invalid
///
/// # Example
///
/// ```no_run
/// use god_gragh::tensor::decomposition::qr::orthogonalize_in_place;
///
/// let mut data = vec![1.0, 0.0, 0.0, 1.0];
/// let shape = vec![2, 2];
/// let error = orthogonalize_in_place(&mut data, &shape).unwrap();
/// assert!(error < 1e-10);
/// ```
pub fn orthogonalize_in_place(data: &mut [f64], shape: &[usize]) -> Result<f64, TensorError> {
    if shape.len() != 2 {
        return Err(TensorError::DimensionMismatch {
            expected: 2,
            got: shape.len(),
        });
    }

    let (m, n) = (shape[0], shape[1]);
    if m < n {
        return Err(TensorError::ShapeMismatch {
            expected: vec![n, m],
            got: vec![m, n],
        });
    }

    if data.len() != m * n {
        return Err(TensorError::DimensionMismatch {
            expected: m * n,
            got: data.len(),
        });
    }

    let mut max_error: f64 = 0.0;

    // Modified Gram-Schmidt with reorthogonalization (in-place)
    for j in 0..n {
        // Compute column norm
        let mut col_norm = 0.0;
        for k in 0..m {
            col_norm += data[k * n + j] * data[k * n + j];
        }
        col_norm = col_norm.sqrt();

        // Orthogonalize against previous columns
        for i in 0..j {
            // Compute dot product: data[:,i]^T * data[:,j]
            let mut dot = 0.0;
            for k in 0..m {
                dot += data[k * n + i] * data[k * n + j];
            }

            // Subtract projection: data[:,j] -= dot * data[:,i]
            for k in 0..m {
                data[k * n + j] -= dot * data[k * n + i];
            }
        }

        // Reorthogonalization pass for numerical stability
        for i in 0..j {
            let mut dot = 0.0;
            for k in 0..m {
                dot += data[k * n + i] * data[k * n + j];
            }
            if dot.abs() > 1e-14 {
                // Second correction
                for k in 0..m {
                    data[k * n + j] -= dot * data[k * n + i];
                }
            }
        }

        // Normalize column j
        let mut new_norm = 0.0;
        for k in 0..m {
            new_norm += data[k * n + j] * data[k * n + j];
        }
        new_norm = new_norm.sqrt();

        let rel_norm = if col_norm > 1e-14 { new_norm / col_norm } else { new_norm };

        if rel_norm > 1e-12 && new_norm > 1e-14 {
            for k in 0..m {
                data[k * n + j] /= new_norm;
            }
        } else {
            // Column is linearly dependent - set to zero
            for k in 0..m {
                data[k * n + j] = 0.0;
            }
        }
    }

    // Compute orthogonality error: max|Q^T Q - I|
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for k in 0..m {
                dot += data[k * n + i] * data[k * n + j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            let error = (dot - expected).abs();
            max_error = max_error.max(error);
        }
    }

    Ok(max_error)
}

/// Check if a matrix is orthogonal (Q^T Q ≈ I)
pub fn is_orthogonal(tensor: &DenseTensor, tolerance: f64) -> bool {
    let shape = tensor.shape();
    if shape.len() != 2 {
        return false;
    }

    let (m, n) = (shape[0], shape[1]);
    if m < n {
        return false;
    }

    let data = tensor.data();

    // For row-major layout: data[row * n + col]
    // Q^T Q: sum_k Q[k,i] * Q[k,j] for k in 0..m
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for k in 0..m {
                // Row-major: Q[k,i] = data[k * n + i]
                dot += data[k * n + i] * data[k * n + j];
            }

            let expected = if i == j { 1.0 } else { 0.0 };
            if (dot - expected).abs() > tolerance {
                return false;
            }
        }
    }

    true
}

/// Debug helper: print matrix statistics
pub fn debug_matrix(tensor: &DenseTensor, label: &str) {
    let shape = tensor.shape();
    let data = tensor.data();
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_val: f64 = data.iter().sum::<f64>() / data.len() as f64;
    println!("{}: shape={:?}, min={:.6}, max={:.6}, mean={:.6}", 
             label, shape, min_val, max_val, mean_val);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::traits::TensorOps;

    #[test]
    fn test_qr_decomposition() {
        let tensor = DenseTensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![4, 2],
        );

        let (q, r) = qr_decompose(&tensor).unwrap();

        assert_eq!(q.shape(), &[4, 2]);
        assert_eq!(r.shape(), &[2, 2]);
        assert!(is_orthogonal(&q, 1e-5));
    }

    #[test]
    fn test_orthogonalize() {
        let tensor = DenseTensor::from_vec(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2],
        );

        let ortho = orthogonalize(&tensor).unwrap();
        assert!(is_orthogonal(&ortho, 1e-5));
    }

    #[test]
    fn test_qr_reconstruction() {
        let original = DenseTensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        );

        let (q, r) = qr_decompose(&original).unwrap();
        let reconstructed = q.matmul(&r);

        let orig_data: &[f64] = original.data();
        let recon_data: &[f64] = reconstructed.data();
        for (a, b) in orig_data.iter().zip(recon_data.iter()) {
            assert!((a - b).abs() < 1e-5, "Reconstruction failed: {} vs {}", a, b);
        }
    }
    
    #[test]
    fn test_qr_square() {
        let tensor = DenseTensor::from_vec(
            vec![4.0, 1.0, 2.0, 3.0],
            vec![2, 2],
        );
        
        let (q, r) = qr_decompose(&tensor).unwrap();
        
        assert_eq!(q.shape(), &[2, 2]);
        assert_eq!(r.shape(), &[2, 2]);
        assert!(is_orthogonal(&q, 1e-5));
        
        let reconstructed = q.matmul(&r);
        let orig_data: &[f64] = tensor.data();
        let recon_data: &[f64] = reconstructed.data();
        for (a, b) in orig_data.iter().zip(recon_data.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
    
    #[test]
    fn test_qr_identity() {
        let tensor = DenseTensor::from_vec(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2],
        );
        
        let (q, r) = qr_decompose(&tensor).unwrap();
        
        assert!(is_orthogonal(&q, 1e-5));
        
        let r_data: &[f64] = r.data();
        assert!((r_data[0] - 1.0).abs() < 1e-5);
        assert!(r_data[1].abs() < 1e-5);
        assert!(r_data[2].abs() < 1e-5);
        assert!((r_data[3] - 1.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_qr_positive_diagonal() {
        let tensor = DenseTensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        );
        
        let (_, r) = qr_decompose(&tensor).unwrap();
        let r_data: &[f64] = r.data();
        
        assert!(r_data[0] >= 0.0, "R[0,0] should be non-negative");
        assert!(r_data[3] >= 0.0, "R[1,1] should be non-negative");
    }

    #[test]
    fn test_orthogonalize_in_place() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![3, 2];

        let error = orthogonalize_in_place(&mut data, &shape).unwrap();

        // Check orthogonality error is small
        assert!(error < 1e-10, "Orthogonalization error too large: {}", error);

        // Verify the result is orthogonal
        let tensor = DenseTensor::from_vec(data, shape);
        assert!(is_orthogonal(&tensor, 1e-10));
    }

    #[test]
    fn test_orthogonalize_in_place_identity() {
        let mut data = vec![1.0, 0.0, 0.0, 1.0];
        let shape = vec![2, 2];

        let error = orthogonalize_in_place(&mut data, &shape).unwrap();

        // Identity matrix should remain unchanged (already orthogonal)
        assert!(error < 1e-10);
        assert_eq!(data, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_orthogonalize_in_place_error() {
        // Invalid shape (1D)
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4];
        assert!(orthogonalize_in_place(&mut data, &shape).is_err());

        // m < n (invalid for QR)
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 4];
        assert!(orthogonalize_in_place(&mut data, &shape).is_err());

        // Data length mismatch
        let mut data = vec![1.0, 2.0, 3.0];
        let shape = vec![2, 2];
        assert!(orthogonalize_in_place(&mut data, &shape).is_err());
    }
}
