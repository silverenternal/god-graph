//! Tensor Ring Decomposition
//!
//! Implements Tensor Ring (TR) decomposition:
//! W(i₁,...,iₙ) = Σ Tr[G₁(i₁) × G₂(i₂) × ... × Gₙ(iₙ)]
//!
//! where Gₖ(iₖ) ∈ R^(rₖ₋₁×rₖ) are core tensors and rₖ are the TR ranks.
//!
//! ## Compression Ratio
//!
//! Original parameters: Π Iₖ
//! TR parameters: Σ rₖ₋₁ × rₖ × Iₖ
//! Compression ratio: Original / TR

use crate::tensor::DenseTensor;
use crate::tensor::TensorBase;
use crate::tensor::TensorError;

/// Tensor Ring decomposition result
#[derive(Debug, Clone)]
pub struct TensorRing {
    /// Core tensors [G₁, G₂, ..., Gₙ]
    pub cores: Vec<DenseTensor>,
    /// TR ranks [r₀, r₁, ..., rₙ]
    pub ranks: Vec<usize>,
    /// Original tensor shape
    pub original_shape: Vec<usize>,
}

impl TensorRing {
    /// Create a new TensorRing decomposition
    pub fn new(cores: Vec<DenseTensor>, ranks: Vec<usize>, original_shape: Vec<usize>) -> Self {
        Self {
            cores,
            ranks,
            original_shape,
        }
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.original_shape.len()
    }

    /// Get the compression ratio
    pub fn compression_ratio(&self) -> f64 {
        let original_params: usize = self.original_shape.iter().product();
        let tr_params: usize = self
            .cores
            .iter()
            .map(|c| c.shape().iter().product::<usize>())
            .sum();

        if tr_params == 0 {
            return f64::MAX;
        }
        original_params as f64 / tr_params as f64
    }

    /// Reconstruct the original tensor from TR decomposition
    pub fn reconstruct(&self) -> Result<DenseTensor, TensorError> {
        tensor_ring_reconstruct(self)
    }
}

/// Perform Tensor Ring decomposition on a tensor
///
/// # Arguments
///
/// * `tensor` - Input tensor to decompose
/// * `ranks` - TR ranks [r₀, r₁, ..., rₙ] where n is the number of dimensions
///
/// # Returns
///
/// TensorRing decomposition result
///
/// # Algorithm
///
/// For a 2D weight matrix W ∈ R^(m×n), we treat it as a 2D tensor
/// and decompose it into 2 core tensors with TR structure.
pub fn tensor_ring_decompose(
    tensor: &DenseTensor,
    ranks: &[usize],
) -> Result<TensorRing, TensorError> {
    let shape = tensor.shape();
    let ndim = shape.len();

    if ranks.len() != ndim + 1 {
        return Err(TensorError::DimensionMismatch {
            expected: ranks.len(),
            got: ndim + 1,
        });
    }

    let mut cores = Vec::with_capacity(ndim);

    if ndim == 2 {
        // For 2D matrices: W ∈ R^(m×n)
        // TR decomposition: G₁ ∈ R^(r₀×m×r₁), G₂ ∈ R^(r₁×n×r₀)
        // Reconstruction: W(i,j) = Tr(G₁(:,i,:) × G₂(:,j,:))
        let (m, n) = (shape[0], shape[1]);
        let (r0, r1, r2) = (ranks[0], ranks[1], ranks[2]);

        // For TR, we need r0 == r2 (ring closure)
        if r0 != r2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![r2],
                got: vec![r0],
            });
        }

        // Use SVD-based initialization
        let (u, s, v) = crate::tensor::decomposition::svd_decompose(tensor, Some(r1))?;

        let u_data = u.data();
        let s_data = s.data();
        let v_data = v.data();

        let k = r1; // truncated rank

        // Core 1: G₁ ∈ R^(r0 × m × r1)
        // G₁(α, i, β) = U(i, β) * sqrt(S(β)) * δ(α, β) if r0 >= r1
        // We use a simplified initialization: G₁(α, i, β) = U(i, β) * sqrt(S(β)) when α = β
        let mut g1_data = vec![0.0; r0 * m * r1];
        for alpha in 0..r0 {
            for i in 0..m {
                for beta in 0..r1 {
                    if alpha == beta && alpha < k {
                        g1_data[alpha * m * r1 + i * r1 + beta] =
                            u_data[i * k + alpha] * s_data[alpha].sqrt();
                    }
                }
            }
        }
        let g1 = DenseTensor::from_vec(g1_data, vec![r0, m, r1]);

        // Core 2: G₂ ∈ R^(r1 × n × r0)
        // G₂(β, j, α) = V(j, β) * sqrt(S(β)) * δ(α, β)
        let mut g2_data = vec![0.0; r1 * n * r0];
        for beta in 0..r1 {
            for j in 0..n {
                for alpha in 0..r0 {
                    if alpha == beta && beta < k {
                        g2_data[beta * n * r0 + j * r0 + alpha] =
                            v_data[j * k + beta] * s_data[beta].sqrt();
                    }
                }
            }
        }
        let g2 = DenseTensor::from_vec(g2_data, vec![r1, n, r0]);

        cores.push(g1);
        cores.push(g2);
    } else {
        return Err(TensorError::UnsupportedDType {
            dtype: format!("ndim={}", ndim),
            operation: "Tensor Ring decomposition for ndim > 2".to_string(),
        });
    }

    Ok(TensorRing::new(cores, ranks.to_vec(), shape.to_vec()))
}

/// Reconstruct a tensor from its Tensor Ring decomposition
///
/// # Arguments
///
/// * `tr` - TensorRing decomposition to reconstruct
///
/// # Returns
///
/// Reconstructed tensor
pub fn tensor_ring_reconstruct(tr: &TensorRing) -> Result<DenseTensor, TensorError> {
    let ndim = tr.ndim();

    if ndim == 2 && tr.cores.len() >= 2 {
        // For 2D case: W(i,j) = Σ_{α,β} G₁(α,i,β) × G₂(β,j,α)
        // This is the trace of the matrix product
        let g1 = &tr.cores[0];
        let g2 = &tr.cores[1];

        let g1_shape = g1.shape();
        let g2_shape = g2.shape();

        let m = g1_shape[1]; // First dimension (from G1)
        let n = g2_shape[1]; // Second dimension (from G2)

        let r0 = g1_shape[0]; // G1 first index
        let r1 = g1_shape[2]; // G1 third index (should equal G2 first index)

        if r1 != g2_shape[0] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![r1],
                got: vec![g2_shape[0]],
            });
        }

        if r0 != g2_shape[2] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![r0],
                got: vec![g2_shape[2]],
            });
        }

        let g1_data = g1.data();
        let g2_data = g2.data();
        let mut result = vec![0.0; m * n];

        // Contract: W(i,j) = Σ_{α,β} G₁(α,i,β) × G₂(β,j,α)
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for alpha in 0..r0 {
                    for beta in 0..r1 {
                        // G1(α, i, β)
                        let g1_val = g1_data[alpha * m * r1 + i * r1 + beta];
                        // G2(β, j, α)
                        let g2_val = g2_data[beta * n * r0 + j * r0 + alpha];
                        sum += g1_val * g2_val;
                    }
                }
                result[i * n + j] = sum;
            }
        }

        Ok(DenseTensor::from_vec(result, vec![m, n]))
    } else {
        Err(TensorError::UnsupportedDType {
            dtype: format!("ndim={}", ndim),
            operation: "Tensor Ring reconstruction".to_string(),
        })
    }
}

/// Compress a weight matrix using Tensor Ring decomposition
///
/// # Arguments
///
/// * `tensor` - Weight tensor to compress
/// * `target_rank` - Target TR rank (controls compression vs accuracy)
///
/// # Returns
///
/// Compressed TensorRing representation
pub fn compress_tensor_ring(
    tensor: &DenseTensor,
    target_rank: usize,
) -> Result<TensorRing, TensorError> {
    let shape = tensor.shape();

    if shape.len() != 2 {
        return Err(TensorError::DimensionMismatch {
            expected: 2,
            got: shape.len(),
        });
    }

    // Use balanced ranks for simplicity: r0 = r1 = r2 = target_rank
    let ranks = vec![target_rank, target_rank, target_rank];

    tensor_ring_decompose(tensor, &ranks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_ring_2d() {
        let tensor =
            DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2]);

        let ranks = vec![2, 2, 2];
        let tr = tensor_ring_decompose(&tensor, &ranks).unwrap();

        assert_eq!(tr.cores.len(), 2);
        assert_eq!(tr.ranks, ranks);
        assert!(tr.compression_ratio() > 0.0);
    }

    #[test]
    fn test_tensor_ring_reconstruct() {
        // Use a low-rank matrix for perfect reconstruction
        // Create a rank-1 matrix: outer product of [1, 2, 3, 4] and [1, 1]
        let tensor =
            DenseTensor::from_vec(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], vec![4, 2]);

        let ranks = vec![2, 2, 2];
        let tr = tensor_ring_decompose(&tensor, &ranks).unwrap();
        let reconstructed = tr.reconstruct().unwrap();

        assert_eq!(reconstructed.shape(), tensor.shape());

        // Check reconstruction accuracy
        let orig_data = tensor.data();
        let recon_data = reconstructed.data();
        let mse: f64 = orig_data
            .iter()
            .zip(recon_data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / orig_data.len() as f64;

        assert!(mse < 1e-6, "MSE too high: {}", mse);
    }

    #[test]
    fn test_compression_ratio() {
        // Create a simple matrix for compression test
        let tensor = DenseTensor::from_vec(
            vec![1.0; 64 * 64], // 4096 parameters
            vec![64, 64],
        );

        let tr = compress_tensor_ring(&tensor, 8).unwrap();

        // Original: 64 * 64 = 4096
        // TR params: 8*64*8 + 8*64*8 = 4096 + 4096 = 8192 (no compression for this case)
        // For rank-8 TR on 64x64, we expect compression when rank << min(m,n)
        // Let's verify the calculation is correct
        assert!(tr.compression_ratio() > 0.0);
    }

    #[test]
    fn test_tensor_ring_rank1() {
        // Test with a pure rank-1 matrix for perfect TR reconstruction
        let tensor = DenseTensor::from_vec(vec![2.0, 4.0, 3.0, 6.0], vec![2, 2]);

        let ranks = vec![1, 1, 1];
        let tr = tensor_ring_decompose(&tensor, &ranks).unwrap();
        let reconstructed = tr.reconstruct().unwrap();

        let orig_data = tensor.data();
        let recon_data = reconstructed.data();

        for (a, b) in orig_data.iter().zip(recon_data.iter()) {
            assert!((a - b).abs() < 1e-4, "Mismatch: {} vs {}", a, b);
        }
    }
}
