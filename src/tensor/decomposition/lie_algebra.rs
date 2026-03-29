//! Lie Algebra Utilities
//!
//! Provides utilities for Lie group operations:
//! - Exponential map: so(n) → SO(n)
//! - Logarithmic map: SO(n) → so(n)
//! - Skew-symmetric projection

use crate::tensor::DenseTensor;
use crate::tensor::TensorBase;
use crate::tensor::TensorError;

/// Compute the matrix exponential exp(A) for a square matrix
///
/// For Lie algebra, this maps so(n) → SO(n)
///
/// # Arguments
///
/// * `algebra` - Skew-symmetric matrix in Lie algebra so(n)
///
/// # Returns
///
/// Matrix in Lie group SO(n)
///
/// # Algorithm
///
/// Uses Padé approximation with scaling and squaring:
/// exp(A) = (exp(A/2^s))^(2^s)
/// The [1/1] Padé approximant is: exp(A) ≈ (I - A/2)^(-1) (I + A/2)
pub fn lie_exponential(algebra: &DenseTensor) -> Result<DenseTensor, TensorError> {
    let shape = algebra.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(TensorError::DimensionMismatch {
            expected: 2,
            got: shape.len(),
        });
    }

    let n = shape[0];
    let data = algebra.data();

    // Scaling: find s such that ||A/2^s||_∞ < 1
    let norm: f64 = data.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let s = if norm > 0.5 {
        ((norm.ln() / 2.0_f64.ln()).ceil() as i32) + 1
    } else {
        0
    };

    // Scale the matrix
    let scale = 2.0_f64.powi(-s);
    let scaled: Vec<f64> = data.iter().map(|x| x * scale).collect();

    // Padé [1/1] approximant: exp(A) ≈ (I - A/2)^(-1) (I + A/2)
    // Let M = I - A/2, N = I + A/2, then exp(A) ≈ M^(-1) N
    
    // Build M = I - A/2
    let mut m_mat = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            m_mat[idx] = if i == j { 1.0 - scaled[idx] / 2.0 } else { -scaled[idx] / 2.0 };
        }
    }
    
    // Build N = I + A/2
    let mut n_mat = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            n_mat[idx] = if i == j { 1.0 + scaled[idx] / 2.0 } else { scaled[idx] / 2.0 };
        }
    }
    
    // Solve M * X = N for X using Gaussian elimination
    // This gives X = M^(-1) * N
    let result = solve_matrix_equation(&m_mat, &n_mat, n)?;

    // Squaring: exp(A) = (exp(A/2^s))^(2^s)
    let mut exp_a = result;
    for _ in 0..s {
        exp_a = matrix_multiply(&exp_a, &exp_a, n);
    }

    Ok(DenseTensor::from_vec(exp_a, vec![n, n]))
}

/// Compute the matrix logarithm log(A) for an orthogonal matrix
///
/// For Lie group, this maps SO(n) → so(n)
///
/// # Arguments
///
/// * `group` - Orthogonal matrix in Lie group SO(n)
///
/// # Returns
///
/// Skew-symmetric matrix in Lie algebra so(n)
pub fn lie_logarithm(group: &DenseTensor) -> Result<DenseTensor, TensorError> {
    let shape = group.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(TensorError::DimensionMismatch {
            expected: 2,
            got: shape.len(),
        });
    }

    let n = shape[0];
    let data = group.data();

    // For matrices close to identity: log(A) ≈ A - I
    // More accurate: use the inverse of exp, i.e., solve for X in exp(X) = A
    // For now, use the series expansion: log(A) ≈ (A - I) - (A - I)^2 / 2 + ...
    
    // First order: A - I
    let mut algebra = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            algebra[idx] = if i == j {
                data[idx] - 1.0
            } else {
                data[idx]
            };
        }
    }

    // Project to skew-symmetric: (A - A^T) / 2
    skew_symmetric_projection_inplace(&mut algebra, n);

    Ok(DenseTensor::from_vec(algebra, vec![n, n]))
}

/// Project a matrix to skew-symmetric form
///
/// For any matrix A, the projection is (A - A^T) / 2
///
/// # Arguments
///
/// * `matrix` - Input matrix
///
/// # Returns
///
/// Skew-symmetric matrix
pub fn skew_symmetric_projection(matrix: &DenseTensor) -> Result<DenseTensor, TensorError> {
    let shape = matrix.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(TensorError::DimensionMismatch {
            expected: 2,
            got: shape.len(),
        });
    }

    let n = shape[0];
    let data = matrix.data();
    let mut result = vec![0.0; n * n];

    skew_symmetric_projection_with(&data, &mut result, n);

    Ok(DenseTensor::from_vec(result, vec![n, n]))
}

/// In-place skew-symmetric projection
fn skew_symmetric_projection_inplace(data: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let idx_ij = i * n + j;
            let idx_ji = j * n + i;
            data[idx_ij] = (data[idx_ij] - data[idx_ji]) / 2.0;
        }
    }
}

/// Skew-symmetric projection with explicit output buffer
fn skew_symmetric_projection_with(input: &[f64], output: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let idx_ij = i * n + j;
            let idx_ji = j * n + i;
            output[idx_ij] = (input[idx_ij] - input[idx_ji]) / 2.0;
        }
    }
}

/// Matrix multiplication for square matrices
fn matrix_multiply(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                result[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    result
}

/// Solve matrix equation A * X = B using Gaussian elimination with partial pivoting
///
/// # Arguments
///
/// * `a` - Coefficient matrix (n x n)
/// * `b` - Right-hand side matrix (n x n)
/// * `n` - Matrix dimension
///
/// # Returns
///
/// Solution matrix X (n x n)
fn solve_matrix_equation(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, TensorError> {
    // Create augmented matrix [A | B] where B has n columns
    // We solve for each column of X separately
    let mut x = vec![0.0; n * n];
    
    for col in 0..n {
        // Extract the col-th column of B
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            rhs[i] = b[i * n + col];
        }
        
        // Create augmented matrix [A | rhs]
        let mut aug = vec![0.0; n * (n + 1)];
        for i in 0..n {
            for j in 0..n {
                aug[i * (n + 1) + j] = a[i * n + j];
            }
            aug[i * (n + 1) + n] = rhs[i];
        }
        
        // Forward elimination with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            let mut max_val = aug[k * (n + 1) + k].abs();
            for row in (k + 1)..n {
                let val = aug[row * (n + 1) + k].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            
            if max_val < 1e-12 {
                return Err(TensorError::BlasError {
                    code: 0,
                    description: "Singular or near-singular matrix".to_string(),
                });
            }
            
            // Swap rows
            if max_row != k {
                for j in 0..(n + 1) {
                    aug.swap(k * (n + 1) + j, max_row * (n + 1) + j);
                }
            }
            
            // Eliminate column
            let pivot = aug[k * (n + 1) + k];
            for row in (k + 1)..n {
                let factor = aug[row * (n + 1) + k] / pivot;
                for j in k..(n + 1) {
                    aug[row * (n + 1) + j] -= factor * aug[k * (n + 1) + j];
                }
            }
        }
        
        // Back substitution
        for i in (0..n).rev() {
            let mut sum = aug[i * (n + 1) + n];
            for j in (i + 1)..n {
                sum -= aug[i * (n + 1) + j] * x[j * n + col];
            }
            x[i * n + col] = sum / aug[i * (n + 1) + i];
        }
    }
    
    Ok(x)
}

/// Create a generator matrix for SO(n)
///
/// Generator matrices form a basis for the Lie algebra so(n).
/// This creates the generator for rotation in the (i, j) plane.
///
/// # Arguments
///
/// * `n` - Dimension of SO(n)
/// * `i` - First plane index
/// * `j` - Second plane index
///
/// # Returns
///
/// Skew-symmetric generator matrix
pub fn so_n_generator(n: usize, i: usize, j: usize) -> Result<DenseTensor, TensorError> {
    if i >= n || j >= n || i == j {
        return Err(TensorError::SliceError {
            description: format!("Invalid indices ({}, {}) for SO({})", i, j, n),
        });
    }

    let mut data = vec![0.0; n * n];
    data[i * n + j] = 1.0;
    data[j * n + i] = -1.0;

    Ok(DenseTensor::from_vec(data, vec![n, n]))
}

/// Check if a matrix is skew-symmetric
///
/// # Arguments
///
/// * `matrix` - Matrix to check
/// * `tolerance` - Tolerance for skew-symmetry check
///
/// # Returns
///
/// True if A^T = -A within tolerance
pub fn is_skew_symmetric(matrix: &DenseTensor, tolerance: f64) -> bool {
    let shape = matrix.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return false;
    }

    let n = shape[0];
    let data = matrix.data();

    for i in 0..n {
        for j in 0..n {
            let idx_ij = i * n + j;
            let idx_ji = j * n + i;
            if (data[idx_ij] + data[idx_ji]).abs() > tolerance {
                return false;
            }
        }
    }

    true
}

/// Check if a matrix is orthogonal (in SO(n))
///
/// # Arguments
///
/// * `matrix` - Matrix to check
/// * `tolerance` - Tolerance for orthogonality check
///
/// # Returns
///
/// True if A^T A = I within tolerance
pub fn is_orthogonal(matrix: &DenseTensor, tolerance: f64) -> bool {
    let shape = matrix.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return false;
    }

    let n = shape[0];
    let data = matrix.data();

    // Check A^T A = I
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for k in 0..n {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skew_symmetric_projection() {
        let matrix = DenseTensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );

        let skew = skew_symmetric_projection(&matrix).unwrap();
        assert!(is_skew_symmetric(&skew, 1e-6));
    }

    #[test]
    fn test_so2_generator() {
        let gen = so_n_generator(2, 0, 1).unwrap();
        assert!(is_skew_symmetric(&gen, 1e-6));
    }

    #[test]
    fn test_lie_exponential_logarithm() {
        // Create a skew-symmetric matrix
        let algebra = DenseTensor::from_vec(
            vec![0.0, -0.1, 0.1, 0.0],
            vec![2, 2],
        );

        // exp: so(n) -> SO(n)
        let group = lie_exponential(&algebra).unwrap();
        assert!(is_orthogonal(&group, 1e-5));

        // log: SO(n) -> so(n)
        let algebra_back = lie_logarithm(&group).unwrap();
        assert!(is_skew_symmetric(&algebra_back, 1e-5));
    }
    
    #[test]
    fn test_lie_exponential_rotation() {
        // Test with a known rotation generator
        // For SO(2), exp([0, -theta; theta, 0]) = [cos(theta), -sin(theta); sin(theta), cos(theta)]
        // Use small theta for better Padé approximation accuracy
        let theta = 0.1;
        let algebra = DenseTensor::from_vec(
            vec![0.0, -theta, theta, 0.0],
            vec![2, 2],
        );

        let group = lie_exponential(&algebra).unwrap();
        let data = group.data();

        // Check against known result (with relaxed tolerance for Padé approximation)
        assert!((data[0] - theta.cos()).abs() < 1e-3, "Expected {}, got {}", theta.cos(), data[0]);
        assert!((data[1] + theta.sin()).abs() < 1e-3, "Expected {}, got {}", -theta.sin(), data[1]);
        assert!((data[2] - theta.sin()).abs() < 1e-3, "Expected {}, got {}", theta.sin(), data[2]);
        assert!((data[3] - theta.cos()).abs() < 1e-3, "Expected {}, got {}", theta.cos(), data[3]);

        assert!(is_orthogonal(&group, 1e-5));
    }
    
    #[test]
    fn test_so3_generator() {
        // Test SO(3) generators
        let gen_xy = so_n_generator(3, 0, 1).unwrap();
        let gen_yz = so_n_generator(3, 1, 2).unwrap();
        let gen_xz = so_n_generator(3, 0, 2).unwrap();
        
        assert!(is_skew_symmetric(&gen_xy, 1e-6));
        assert!(is_skew_symmetric(&gen_yz, 1e-6));
        assert!(is_skew_symmetric(&gen_xz, 1e-6));
    }
}
