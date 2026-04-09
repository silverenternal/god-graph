//! Multi-Head Attention implementation

use crate::tensor::DenseTensor;
use crate::tensor::traits::{TensorOps, TensorBase};

/// Multi-Head Attention layer
///
/// This implements the standard multi-head attention mechanism:
/// 1. Linear projections for Q, K, V
/// 2. Split into multiple heads
/// 3. Scaled dot-product attention for each head
/// 4. Concatenate heads and output projection
///
/// Supports Grouped-Query Attention (GQA) where multiple Q heads share a single K/V head.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Query projection matrix [hidden_dim, hidden_dim]
    pub w_q: DenseTensor,
    /// Key projection matrix [hidden_dim, num_kv_heads * head_dim]
    pub w_k: DenseTensor,
    /// Value projection matrix [hidden_dim, num_kv_heads * head_dim]
    pub w_v: DenseTensor,
    /// Output projection matrix [hidden_dim, hidden_dim]
    pub w_o: DenseTensor,
    /// Number of query heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA, can be less than num_heads)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Scaling factor (1.0 / sqrt(head_dim))
    pub scale: f64,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    ///
    /// # Arguments
    /// * `w_q` - Query projection matrix [hidden_dim, hidden_dim]
    /// * `w_k` - Key projection matrix [hidden_dim, num_kv_heads * head_dim]
    /// * `w_v` - Value projection matrix [hidden_dim, num_kv_heads * head_dim]
    /// * `w_o` - Output projection matrix [hidden_dim, hidden_dim]
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of KV heads (use same as num_heads for standard MHA)
    pub fn new(
        w_q: DenseTensor,
        w_k: DenseTensor,
        w_v: DenseTensor,
        w_o: DenseTensor,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Self {
        let hidden_dim = w_q.shape()[0];
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        }
    }

    /// Create multi-head attention with standard configuration (num_kv_heads = num_heads)
    pub fn standard(w_q: DenseTensor, w_k: DenseTensor, w_v: DenseTensor, w_o: DenseTensor, num_heads: usize) -> Self {
        Self::new(w_q, w_k, w_v, w_o, num_heads, num_heads)
    }

    /// Forward pass for multi-head attention
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch_size, seq_len, hidden_dim]
    ///
    /// # Returns
    /// Output tensor [batch_size, seq_len, hidden_dim]
    pub fn forward(&self, x: &DenseTensor) -> DenseTensor {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let hidden_dim = self.num_heads * self.head_dim;

        // Linear projections using batched matmul
        let q = x.bmm_broadcast_weight(&self.w_q);
        let k = x.bmm_broadcast_weight(&self.w_k);
        let v = x.bmm_broadcast_weight(&self.w_v);

        // Reshape for multi-head: [batch, seq_len, num_heads, head_dim]
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, num_heads, seq_len, head_dim]
        let q = q.transpose_2d();
        let k = k.transpose_2d();
        let v = v.transpose_2d();

        // Apply attention
        let attn_output = self.scaled_dot_product_attention(&q, &k, &v, None);

        // Transpose back: [batch, seq_len, num_heads, head_dim]
        let attn_output = attn_output.transpose_2d();

        // Reshape: [batch, seq_len, hidden_dim]
        let attn_output = attn_output.reshape(&[batch_size, seq_len, hidden_dim]);

        // Output projection using batched matmul
        attn_output.bmm_broadcast_weight(&self.w_o)
    }

    /// Forward pass with optional mask
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch_size, seq_len, hidden_dim]
    /// * `mask` - Optional attention mask [batch_size, seq_len, seq_len] or [seq_len, seq_len]
    ///
    /// # Returns
    /// Output tensor [batch_size, seq_len, hidden_dim]
    pub fn forward_with_mask(&self, x: &DenseTensor, mask: Option<&DenseTensor>) -> DenseTensor {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let hidden_dim = self.num_heads * self.head_dim;

        // Linear projections using batched matmul
        let q = x.bmm_broadcast_weight(&self.w_q);
        let k = x.bmm_broadcast_weight(&self.w_k);
        let v = x.bmm_broadcast_weight(&self.w_v);

        // Reshape for multi-head
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, num_heads, seq_len, head_dim]
        let q = q.transpose_2d();
        let k = k.transpose_2d();
        let v = v.transpose_2d();

        // Apply attention with mask
        let attn_output = self.scaled_dot_product_attention(&q, &k, &v, mask);

        // Transpose back and reshape
        let attn_output = attn_output.transpose_2d();
        let attn_output = attn_output.reshape(&[batch_size, seq_len, hidden_dim]);

        // Output projection using batched matmul
        attn_output.bmm_broadcast_weight(&self.w_o)
    }

    /// Scaled dot-product attention implementation
    ///
    /// # Arguments
    /// * `q` - Query [batch, num_heads, seq_len, head_dim]
    /// * `k` - Key [batch, num_kv_heads, seq_len, head_dim]
    /// * `v` - Value [batch, num_kv_heads, seq_len, head_dim]
    /// * `mask` - Optional attention mask
    fn scaled_dot_product_attention(
        &self,
        q: &DenseTensor,
        k: &DenseTensor,
        v: &DenseTensor,
        mask: Option<&DenseTensor>,
    ) -> DenseTensor {
        // For GQA, repeat K/V heads to match Q heads if needed
        let k_expanded = if self.num_heads != self.num_kv_heads {
            self.repeat_kv(k)
        } else {
            k.clone()
        };

        let v_expanded = if self.num_heads != self.num_kv_heads {
            self.repeat_kv(v)
        } else {
            v.clone()
        };

        // Reshape for batched matmul: [batch, num_heads, seq_len, head_dim] -> [batch * num_heads, seq_len, head_dim]
        let batch = q.shape()[0];
        let num_heads = q.shape()[1];
        let seq_len = q.shape()[2];
        let head_dim = q.shape()[3];

        let q_reshaped = q.reshape(&[batch * num_heads, seq_len, head_dim]);
        let k_reshaped = k_expanded.reshape(&[batch * num_heads, seq_len, head_dim]);
        let v_reshaped = v_expanded.reshape(&[batch * num_heads, seq_len, head_dim]);

        // Compute attention scores: Q @ K.T * scale
        // For 3D @ 3D transpose, we need to do batched transpose and matmul
        let k_transposed = Self::batch_transpose_3d(&k_reshaped);
        let mut scores = Self::batch_matmul_3d(&q_reshaped, &k_transposed);
        scores = scores.scale(self.scale);

        // Apply mask if provided
        if let Some(mask) = mask {
            scores = scores.mask_fill(mask, f64::NEG_INFINITY);
        }

        // Apply softmax
        let attn_weights = scores.softmax(-1);

        // Apply attention to values: attn @ V
        let attn_output = Self::batch_matmul_3d(&attn_weights, &v_reshaped);

        // Reshape back to [batch, num_heads, seq_len, head_dim]
        attn_output.reshape(&[batch, num_heads, seq_len, head_dim])
    }

    /// Batch transpose for 3D tensor: [batch, A, B] -> [batch, B, A]
    fn batch_transpose_3d(x: &DenseTensor) -> DenseTensor {
        assert_eq!(x.ndim(), 3, "Must be 3D tensor");
        let batch = x.shape()[0];
        let a = x.shape()[1];
        let b = x.shape()[2];

        let mut data = Vec::with_capacity(batch * a * b);

        for batch_idx in 0..batch {
            for j in 0..b {
                for i in 0..a {
                    let src_idx = (batch_idx * a + i) * b + j;
                    data.push(x.data()[src_idx]);
                }
            }
        }

        DenseTensor::new(data, vec![batch, b, a])
    }

    /// Batched matrix multiplication for 3D tensors
    /// [batch, A, B] @ [batch, B, C] -> [batch, A, C]
    fn batch_matmul_3d(a: &DenseTensor, b: &DenseTensor) -> DenseTensor {
        assert_eq!(a.ndim(), 3, "First tensor must be 3D");
        assert_eq!(b.ndim(), 3, "Second tensor must be 3D");
        assert_eq!(a.shape()[0], b.shape()[0], "Batch dimensions must match");
        assert_eq!(a.shape()[2], b.shape()[1], "Inner dimensions must match");

        let batch = a.shape()[0];
        let m = a.shape()[1];
        let k = a.shape()[2];
        let n = b.shape()[2];

        let mut result = vec![0.0; batch * m * n];

        #[cfg(feature = "simd")]
        {
            // SIMD-optimized batched matmul using wide::f64x4
            use wide::f64x4;

            for batch_idx in 0..batch {
                for i in 0..m {
                    let a_row_offset = (batch_idx * m + i) * k;
                    
                    for j in (0..n).step_by(4) {
                        if j + 4 <= n {
                            // SIMD: process 4 columns at once
                            let mut sum_simd = f64x4::new([0.0; 4]);
                            
                            for p in 0..k {
                                let a_val = a.data()[a_row_offset + p];
                                let a_simd = f64x4::new([a_val; 4]);
                                
                                let b_vals = [
                                    b.data()[(batch_idx * k + p) * n + j],
                                    b.data()[(batch_idx * k + p) * n + j + 1],
                                    b.data()[(batch_idx * k + p) * n + j + 2],
                                    b.data()[(batch_idx * k + p) * n + j + 3],
                                ];
                                let b_simd = f64x4::new(b_vals);
                                
                                sum_simd += a_simd * b_simd;
                            }
                            
                            let sums = sum_simd.to_array();
                            result[(batch_idx * m + i) * n + j] = sums[0];
                            result[(batch_idx * m + i) * n + j + 1] = sums[1];
                            result[(batch_idx * m + i) * n + j + 2] = sums[2];
                            result[(batch_idx * m + i) * n + j + 3] = sums[3];
                        } else {
                            // Handle remainder
                            for rem_j in j..n {
                                let mut sum = 0.0;
                                for p in 0..k {
                                    let a_val = a.data()[a_row_offset + p];
                                    let b_val = b.data()[(batch_idx * k + p) * n + rem_j];
                                    sum += a_val * b_val;
                                }
                                result[(batch_idx * m + i) * n + rem_j] = sum;
                            }
                        }
                    }
                }
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            // Fallback: naive implementation
            for batch_idx in 0..batch {
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for p in 0..k {
                            let a_val = a.data()[(batch_idx * m + i) * k + p];
                            let b_val = b.data()[(batch_idx * k + p) * n + j];
                            sum += a_val * b_val;
                        }
                        result[(batch_idx * m + i) * n + j] = sum;
                    }
                }
            }
        }

        DenseTensor::new(result, vec![batch, m, n])
    }

    /// Repeat KV heads for GQA
    ///
    /// # Arguments
    /// * `x` - KV tensor [batch, num_kv_heads, seq_len, head_dim]
    ///
    /// # Returns
    /// Expanded tensor [batch, num_heads, seq_len, head_dim]
    fn repeat_kv(&self, x: &DenseTensor) -> DenseTensor {
        if self.num_heads == self.num_kv_heads {
            return x.clone();
        }

        let batch = x.shape()[0];
        let seq_len = x.shape()[2];
        let head_dim = x.shape()[3];
        let repeats = self.num_heads / self.num_kv_heads;

        let mut data = Vec::with_capacity(batch * self.num_heads * seq_len * head_dim);

        for b in 0..batch {
            for kv_head in 0..self.num_kv_heads {
                for _ in 0..repeats {
                    for s in 0..seq_len {
                        let offset = ((b * self.num_kv_heads + kv_head) * seq_len + s) * head_dim;
                        let slice = &x.data()[offset..offset + head_dim];
                        data.extend_from_slice(slice);
                    }
                }
            }
        }

        DenseTensor::new(data, vec![batch, self.num_heads, seq_len, head_dim])
    }

    /// Get number of query heads
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get number of KV heads
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get the number of parameters in this attention layer
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;
        total += self.w_q.shape().iter().product::<usize>();
        total += self.w_k.shape().iter().product::<usize>();
        total += self.w_v.shape().iter().product::<usize>();
        total += self.w_o.shape().iter().product::<usize>();
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_head_attention_standard() {
        let batch_size = 2;
        let seq_len = 4;
        let hidden_dim = 8;
        let num_heads = 2;
        let _head_dim = hidden_dim / num_heads;

        // Initialize weight matrices
        let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_k = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_v = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);

        let attn = MultiHeadAttention::standard(w_q, w_k, w_v, w_o, num_heads);

        // Create input
        let x = DenseTensor::ones(vec![batch_size, seq_len, hidden_dim]);
        let output = attn.forward(&x);

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_dim]);
    }

    #[test]
    fn test_multi_head_attention_gqa() {
        let batch_size = 2;
        let seq_len = 4;
        let hidden_dim = 8;
        let num_heads = 4;
        let num_kv_heads = 2; // GQA: fewer KV heads
        let head_dim = hidden_dim / num_heads;

        // Initialize weight matrices
        let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_k = DenseTensor::ones(vec![hidden_dim, num_kv_heads * head_dim]);
        let w_v = DenseTensor::ones(vec![hidden_dim, num_kv_heads * head_dim]);
        let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);

        let attn = MultiHeadAttention::new(w_q, w_k, w_v, w_o, num_heads, num_kv_heads);

        // Create input
        let x = DenseTensor::ones(vec![batch_size, seq_len, hidden_dim]);
        let output = attn.forward(&x);

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_dim]);
    }
}
