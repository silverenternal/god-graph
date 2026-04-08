//! Inference optimizations for Transformer models
//!
//! This module provides utilities for optimized inference:
//! - Weight caching (pre-transposed weights for faster matmul)
//! - Memory pool integration
//! - Batched inference support

#![allow(dead_code)] // Allow unused fields for future extension

use crate::tensor::dense::DenseTensor;
use crate::tensor::traits::TensorBase;

/// Cached linear layer for fast inference
///
/// This struct pre-transposes weights once during initialization,
/// avoiding runtime transpose overhead during inference.
///
/// # Example
/// ```
/// use god_graph::tensor::DenseTensor;
/// use god_graph::transformer::inference::CachedLinear;
///
/// let weight = DenseTensor::ones(vec![128, 256]); // [in_features, out_features]
/// let linear = CachedLinear::new(&weight, None);
///
/// let input = DenseTensor::ones(vec![2, 10, 128]); // [batch, seq, in_features]
/// let output = linear.forward(&input);
/// ```
#[derive(Debug, Clone)]
pub struct CachedLinear {
    /// Transposed weight matrix [out_features, in_features]
    weight_t: DenseTensor,
    /// Optional bias [out_features]
    bias: Option<DenseTensor>,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
}

impl CachedLinear {
    /// Create a new cached linear layer
    ///
    /// # Arguments
    /// * `weight` - Weight matrix [in_features, out_features]
    /// * `bias` - Optional bias vector [out_features]
    pub fn new(weight: &DenseTensor, bias: Option<&DenseTensor>) -> Self {
        assert_eq!(weight.ndim(), 2, "Weight must be 2D");

        let in_features = weight.shape()[0];
        let out_features = weight.shape()[1];

        // Transpose weight once during initialization
        let weight_t = Self::transpose_weight(weight);

        Self {
            weight_t,
            bias: bias.cloned(),
            in_features,
            out_features,
        }
    }

    /// Transpose weight matrix from [in, out] to [out, in]
    fn transpose_weight(weight: &DenseTensor) -> DenseTensor {
        let in_features = weight.shape()[0];
        let out_features = weight.shape()[1];
        let weight_data = weight.data();

        let mut weight_t_data = Vec::with_capacity(in_features * out_features);

        // Transpose: weight_t[out][in] = weight[in][out]
        for o in 0..out_features {
            for i in 0..in_features {
                weight_t_data.push(weight_data[i * out_features + o]);
            }
        }

        DenseTensor::new(weight_t_data, vec![out_features, in_features])
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq, in_features]
    ///
    /// # Returns
    /// Output tensor [batch, seq, out_features]
    pub fn forward(&self, x: &DenseTensor) -> DenseTensor {
        // Use pre-transposed weight for faster matmul
        let mut output = x.bmm_weight_transposed(&self.weight_t);

        // Add bias if present (broadcast along last dimension)
        if let Some(ref bias) = self.bias {
            Self::add_bias_inplace(&mut output, bias);
        }

        output
    }

    /// Add bias to tensor (in-place, broadcast along last dimension)
    #[inline]
    fn add_bias_inplace(x: &mut DenseTensor, bias: &DenseTensor) {
        let shape = x.shape();
        let hidden_dim = shape[shape.len() - 1];
        let data = x.data_mut();
        let bias_data = bias.data();

        #[cfg(feature = "simd")]
        {
            use wide::f64x4;

            let data_len = data.len();
            
            // Process 4 elements at a time with SIMD
            let num_chunks = data_len / 4;
            let remainder_len = data_len % 4;
            
            for i in 0..num_chunks {
                let base_idx = i * 4;
                let i_mod = base_idx % hidden_dim;
                
                // Handle alignment: if not aligned to 4, process element-wise
                if i_mod % 4 != 0 || i_mod + 4 > hidden_dim {
                    for j in 0..4 {
                        data[base_idx + j] += bias_data[(i_mod + j) % hidden_dim];
                    }
                } else {
                    // Load 4 bias values and add with SIMD
                    let bias_vec = f64x4::from([
                        bias_data[i_mod],
                        bias_data[i_mod + 1],
                        bias_data[i_mod + 2],
                        bias_data[i_mod + 3],
                    ]);
                    let data_vec = f64x4::from([
                        data[base_idx],
                        data[base_idx + 1],
                        data[base_idx + 2],
                        data[base_idx + 3],
                    ]);
                    let result = data_vec + bias_vec;
                    let arr = result.to_array();
                    data[base_idx..base_idx + 4].copy_from_slice(&arr);
                }
            }
            
            // Handle remainder
            let remainder_start = num_chunks * 4;
            for i in 0..remainder_len {
                let idx = (remainder_start + i) % hidden_dim;
                data[remainder_start + i] += bias_data[idx];
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            for (i, val) in data.iter_mut().enumerate() {
                *val += bias_data[i % hidden_dim];
            }
        }
    }

    /// Get input features
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get transposed weight matrix
    pub fn weight_transposed(&self) -> &DenseTensor {
        &self.weight_t
    }
}

/// Cached multi-head attention for fast inference
///
/// Pre-transposes all projection weights for faster attention computation.
#[derive(Debug, Clone)]
pub struct CachedMultiHeadAttention {
    /// Cached Q projection
    cached_w_q: CachedLinear,
    /// Cached K projection
    cached_w_k: CachedLinear,
    /// Cached V projection
    cached_w_v: CachedLinear,
    /// Cached output projection
    cached_w_o: CachedLinear,
    /// Number of query heads
    num_heads: usize,
    /// Number of KV heads (for GQA)
    num_kv_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Scaling factor
    scale: f64,
}

impl CachedMultiHeadAttention {
    /// Create cached multi-head attention
    ///
    /// # Arguments
    /// * `w_q` - Query weight [hidden_dim, hidden_dim]
    /// * `w_k` - Key weight [hidden_dim, num_kv_heads * head_dim]
    /// * `w_v` - Value weight [hidden_dim, num_kv_heads * head_dim]
    /// * `w_o` - Output weight [hidden_dim, hidden_dim]
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of KV heads
    pub fn new(
        w_q: &DenseTensor,
        w_k: &DenseTensor,
        w_v: &DenseTensor,
        w_o: &DenseTensor,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Self {
        let hidden_dim = w_q.shape()[0];
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        Self {
            cached_w_q: CachedLinear::new(w_q, None),
            cached_w_k: CachedLinear::new(w_k, None),
            cached_w_v: CachedLinear::new(w_v, None),
            cached_w_o: CachedLinear::new(w_o, None),
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        }
    }

    /// Get number of heads
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get scale factor
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

/// Cached feed-forward network for fast inference
#[derive(Debug, Clone)]
pub struct CachedFeedForward {
    /// Cached gate projection (for SwiGLU/GeGLU)
    cached_gate_proj: Option<CachedLinear>,
    /// Cached up projection
    cached_up_proj: Option<CachedLinear>,
    /// Cached down projection
    cached_down_proj: Option<CachedLinear>,
    /// Cached FC1 (for standard FFN)
    cached_fc1: Option<CachedLinear>,
    /// Cached FC2 (for standard FFN)
    cached_fc2: Option<CachedLinear>,
    /// Intermediate dimension
    intermediate_dim: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// FFN type
    ffn_type: FFNType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum FFNType {
    Standard,
    SwiGLU,
    GeGLU,
}

impl CachedFeedForward {
    /// Create cached standard FFN
    pub fn standard(fc1: &DenseTensor, fc2: &DenseTensor) -> Self {
        let hidden_dim = fc1.shape()[0];
        let intermediate_dim = fc1.shape()[1];

        Self {
            cached_fc1: Some(CachedLinear::new(fc1, None)),
            cached_fc2: Some(CachedLinear::new(fc2, None)),
            cached_gate_proj: None,
            cached_up_proj: None,
            cached_down_proj: None,
            intermediate_dim,
            hidden_dim,
            ffn_type: FFNType::Standard,
        }
    }

    /// Create cached SwiGLU FFN
    pub fn swiglu(gate_proj: &DenseTensor, up_proj: &DenseTensor, down_proj: &DenseTensor) -> Self {
        let hidden_dim = gate_proj.shape()[0];
        let intermediate_dim = gate_proj.shape()[1];

        Self {
            cached_fc1: None,
            cached_fc2: None,
            cached_gate_proj: Some(CachedLinear::new(gate_proj, None)),
            cached_up_proj: Some(CachedLinear::new(up_proj, None)),
            cached_down_proj: Some(CachedLinear::new(down_proj, None)),
            intermediate_dim,
            hidden_dim,
            ffn_type: FFNType::SwiGLU,
        }
    }

    /// Get intermediate dimension
    pub fn intermediate_dim(&self) -> usize {
        self.intermediate_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cached_linear() {
        let in_features = 64;
        let out_features = 128;

        let weight = DenseTensor::ones(vec![in_features, out_features]);
        let linear = CachedLinear::new(&weight, None);

        assert_eq!(linear.in_features(), in_features);
        assert_eq!(linear.out_features(), out_features);
        assert_eq!(linear.weight_transposed().shape(), &[out_features, in_features]);

        let batch = 2;
        let seq = 10;
        let input = DenseTensor::ones(vec![batch, seq, in_features]);
        let output = linear.forward(&input);

        assert_eq!(output.shape(), &[batch, seq, out_features]);
    }

    #[test]
    fn test_cached_linear_with_bias() {
        let in_features = 64;
        let out_features = 128;

        let weight = DenseTensor::ones(vec![in_features, out_features]);
        let bias = DenseTensor::ones(vec![out_features]);
        let linear = CachedLinear::new(&weight, Some(&bias));

        let batch = 2;
        let seq = 10;
        let input = DenseTensor::ones(vec![batch, seq, in_features]);
        let output = linear.forward(&input);

        assert_eq!(output.shape(), &[batch, seq, out_features]);
    }
}
