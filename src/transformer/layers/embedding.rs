//! Rotary Positional Embedding (RoPE) implementation

use crate::tensor::DenseTensor;
use crate::tensor::traits::TensorBase;

/// Rotary Positional Embedding (RoPE)
///
/// RoPE encodes positional information by rotating the query and key vectors
/// based on their positions. This is used in LLaMA, Mistral, and other modern LLMs.
///
/// Formula: RoPE(x, pos) = x * cos(pos * theta) + rotate_half(x) * sin(pos * theta)
///
/// where theta = base^(-2i/dim) for i = 0, 1, ..., dim/2-1
#[derive(Debug, Clone)]
pub struct RoPE {
    /// Pre-computed cosine cache [max_seq_len, dim/2]
    pub cos_cache: DenseTensor,
    /// Pre-computed sine cache [max_seq_len, dim/2]
    pub sin_cache: DenseTensor,
    /// Rotation dimension (typically head_dim)
    pub dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Base frequency (LLaMA uses 10000)
    pub base: f64,
}

impl RoPE {
    /// Create a new RoPE module
    ///
    /// # Arguments
    /// * `dim` - Rotation dimension (typically head_dim)
    /// * `max_seq_len` - Maximum sequence length to support
    /// * `base` - Base frequency for theta calculation (default: 10000)
    pub fn new(dim: usize, max_seq_len: usize, base: f64) -> Self {
        // Pre-compute theta frequencies: theta = base^(-2i/dim)
        let theta = Self::compute_theta(dim, base);
        
        // Compute positions: [0, 1, ..., max_seq_len-1]
        let positions: Vec<f64> = (0..max_seq_len).map(|i| i as f64).collect();
        
        // Compute outer product: positions * theta
        let freqs = Self::compute_freqs(&positions, &theta);
        
        // Compute cos and sin caches
        let cos_cache = freqs.cos();
        let sin_cache = freqs.sin();
        
        Self {
            cos_cache,
            sin_cache,
            dim,
            max_seq_len,
            base,
        }
    }

    /// Create RoPE with default base (10000)
    pub fn default(dim: usize, max_seq_len: usize) -> Self {
        Self::new(dim, max_seq_len, 10000.0)
    }

    /// Compute theta frequencies
    fn compute_theta(dim: usize, base: f64) -> Vec<f64> {
        let half_dim = dim / 2;
        let mut theta = Vec::with_capacity(half_dim);
        
        for i in 0..half_dim {
            let exponent = -2.0 * i as f64 / dim as f64;
            theta.push(base.powf(exponent));
        }
        
        theta
    }

    /// Compute frequency matrix (outer product of positions and theta)
    fn compute_freqs(positions: &[f64], theta: &[f64]) -> DenseTensor {
        let max_seq_len = positions.len();
        let half_dim = theta.len();
        
        let mut data = Vec::with_capacity(max_seq_len * half_dim);
        
        for &pos in positions {
            for &t in theta {
                data.push(pos * t);
            }
        }
        
        DenseTensor::new(data, vec![max_seq_len, half_dim])
    }

    /// Apply RoPE to input tensor
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch_size, seq_len, dim]
    /// * `positions` - Optional position indices [seq_len] or [batch_size * seq_len]
    ///
    /// # Returns
    /// Rotated tensor [batch_size, seq_len, dim]
    pub fn forward(&self, x: &DenseTensor, positions: Option<&[usize]>) -> DenseTensor {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        
        // Default to sequential positions if not provided
        let default_positions: Vec<usize> = (0..seq_len).collect();
        let positions = positions.unwrap_or(&default_positions);
        
        let mut output = Vec::with_capacity(batch_size * seq_len * self.dim);
        let half_dim = self.dim / 2;
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                let pos = positions[s % positions.len()];
                
                // Get cos/sin for this position
                let cos = self.cos_cache.get_row(pos.min(self.max_seq_len - 1));
                let sin = self.sin_cache.get_row(pos.min(self.max_seq_len - 1));
                
                // Get input slice for this position
                let x_start = (b * seq_len + s) * self.dim;
                let x_slice = &x.data()[x_start..x_start + self.dim];
                
                // Apply RoPE: x * cos + rotate_half(x) * sin
                for i in 0..half_dim {
                    let x1 = x_slice[i];
                    let x2 = x_slice[i + half_dim];
                    
                    // rotate_half: [-x2, x1]
                    let rotated_x1 = -x2;
                    let rotated_x2 = x1;
                    
                    // Apply rotation
                    let out1 = x1 * cos.data()[i] + rotated_x1 * sin.data()[i];
                    let out2 = x2 * cos.data()[i] + rotated_x2 * sin.data()[i];
                    
                    output.push(out1);
                    output.push(out2);
                }
            }
        }
        
        DenseTensor::new(output, vec![batch_size, seq_len, self.dim])
    }

    /// Apply RoPE to query and key tensors
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch_size, seq_len, dim]
    /// * `k` - Key tensor [batch_size, seq_len, dim]
    /// * `positions` - Optional position indices
    ///
    /// # Returns
    /// Tuple of (rotated_q, rotated_k)
    pub fn forward_qk(&self, q: &DenseTensor, k: &DenseTensor, positions: Option<&[usize]>) -> (DenseTensor, DenseTensor) {
        let rotated_q = self.forward(q, positions);
        let rotated_k = self.forward(k, positions);
        (rotated_q, rotated_k)
    }

    /// Get the dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_creation() {
        let dim = 8;
        let max_seq_len = 512;
        let rope = RoPE::default(dim, max_seq_len);

        assert_eq!(rope.dim(), dim);
        assert_eq!(rope.max_seq_len(), max_seq_len);
        assert_eq!(rope.cos_cache.shape(), &[max_seq_len, dim / 2]);
        assert_eq!(rope.sin_cache.shape(), &[max_seq_len, dim / 2]);
    }

    #[test]
    fn test_rope_forward() {
        let dim = 8;
        let max_seq_len = 512;
        let rope = RoPE::default(dim, max_seq_len);

        let batch_size = 2;
        let seq_len = 4;
        let x = DenseTensor::ones(vec![batch_size, seq_len, dim]);

        let output = rope.forward(&x, None);

        assert_eq!(output.shape(), &[batch_size, seq_len, dim]);
    }

    #[test]
    fn test_rope_with_positions() {
        let dim = 8;
        let max_seq_len = 512;
        let rope = RoPE::default(dim, max_seq_len);

        let batch_size = 1;
        let seq_len = 3;
        let x = DenseTensor::ones(vec![batch_size, seq_len, dim]);
        let positions = vec![0, 2, 4];

        let output = rope.forward(&x, Some(&positions));

        assert_eq!(output.shape(), &[batch_size, seq_len, dim]);
    }

    #[test]
    fn test_rope_preserves_norm() {
        // RoPE is a rotation, so it should preserve the L2 norm
        let dim = 8;
        let max_seq_len = 512;
        let rope = RoPE::default(dim, max_seq_len);

        let batch_size = 1;
        let seq_len = 1;
        let x = DenseTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![batch_size, seq_len, dim]);

        let output = rope.forward(&x, None);

        // Compute L2 norm of input and output
        let input_norm: f64 = x.data().iter().map(|v| v * v).sum::<f64>().sqrt();
        let output_norm: f64 = output.data().iter().map(|v| v * v).sum::<f64>().sqrt();

        // Norms should be equal (within numerical precision)
        assert!((input_norm - output_norm).abs() < 1e-5);
    }
}
