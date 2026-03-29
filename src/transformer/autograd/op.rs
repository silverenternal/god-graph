//! Operation traits and implementations for autograd

use crate::tensor::traits::{TensorBase, TensorOps};
use crate::tensor::DenseTensor;

/// Trait for differentiable operations
pub trait DifferentiableOp {
    /// Forward pass
    fn forward(&self, inputs: &[&DenseTensor]) -> DenseTensor;

    /// Backward pass (compute gradients)
    fn backward(&self, inputs: &[&DenseTensor], grad_output: &DenseTensor) -> Vec<DenseTensor>;
}

/// Linear layer operations: y = xW + b
#[derive(Debug, Clone)]
pub struct LinearOp {
    /// Weight matrix
    pub weight: DenseTensor,
    /// Bias vector (optional)
    pub bias: Option<DenseTensor>,
}

impl LinearOp {
    /// Create a new linear operation
    pub fn new(weight: DenseTensor, bias: Option<DenseTensor>) -> Self {
        Self { weight, bias }
    }

    /// Forward pass
    pub fn forward(&self, input: &DenseTensor) -> DenseTensor {
        let output = input.matmul(&self.weight);
        if let Some(bias) = &self.bias {
            output.add(bias)
        } else {
            output
        }
    }
}

/// Embedding lookup operation
#[derive(Debug, Clone)]
pub struct EmbeddingOp {
    /// Embedding table
    pub embeddings: DenseTensor,
}

impl EmbeddingOp {
    /// Create a new embedding operation
    pub fn new(embeddings: DenseTensor) -> Self {
        Self { embeddings }
    }

    /// Forward pass - lookup embeddings for given indices
    pub fn forward(&self, indices: &[usize]) -> DenseTensor {
        let dim = self.embeddings.shape()[1];
        let mut data = Vec::with_capacity(indices.len() * dim);

        for &idx in indices {
            let start = idx * dim;
            let end = start + dim;
            data.extend_from_slice(&self.embeddings.data()[start..end]);
        }

        DenseTensor::new(data, vec![indices.len(), dim])
    }
}

/// Scaled dot-product attention operation
#[derive(Debug, Clone)]
pub struct ScaledDotProductOp {
    /// Scaling factor
    pub scale: f64,
}

impl ScaledDotProductOp {
    /// Create a new scaled dot-product operation
    pub fn new(head_dim: usize) -> Self {
        Self {
            scale: 1.0 / (head_dim as f64).sqrt(),
        }
    }

    /// Forward pass without mask
    pub fn forward(
        &self,
        query: &DenseTensor,
        key: &DenseTensor,
        value: &DenseTensor,
    ) -> DenseTensor {
        // Compute attention scores: Q @ K.T * scale
        let key_t = key.transpose(None);
        let mut scores = query.matmul(&key_t);
        scores = scores.scale(self.scale);

        // Apply softmax
        let attn_weights = scores.softmax(-1);

        // Apply attention to values: attn @ V
        attn_weights.matmul(value)
    }

    /// Forward pass with optional mask
    pub fn forward_with_mask(
        &self,
        query: &DenseTensor,
        key: &DenseTensor,
        value: &DenseTensor,
        mask: Option<&DenseTensor>,
    ) -> DenseTensor {
        // Compute attention scores
        let key_t = key.transpose(None);
        let mut scores = query.matmul(&key_t);
        scores = scores.scale(self.scale);

        // Apply mask if provided
        if let Some(mask) = mask {
            scores = scores.mask_fill(mask, f64::NEG_INFINITY);
        }

        // Apply softmax
        let attn_weights = scores.softmax(-1);

        // Apply attention to values
        attn_weights.matmul(value)
    }
}

/// Multi-head attention operation
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionOp {
    /// Query projection weights
    pub w_q: DenseTensor,
    /// Key projection weights
    pub w_k: DenseTensor,
    /// Value projection weights
    pub w_v: DenseTensor,
    /// Output projection weights
    pub w_o: DenseTensor,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Scaling factor
    pub scale: f64,
}

impl MultiHeadAttentionOp {
    /// Create a new multi-head attention operation
    pub fn new(
        w_q: DenseTensor,
        w_k: DenseTensor,
        w_v: DenseTensor,
        w_o: DenseTensor,
        num_heads: usize,
    ) -> Self {
        let head_dim = w_q.shape()[0] / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            num_heads,
            head_dim,
            scale,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &DenseTensor) -> DenseTensor {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        // Linear projections
        let q = x.matmul(&self.w_q);
        let k = x.matmul(&self.w_k);
        let v = x.matmul(&self.w_v);

        // Reshape for multi-head: [batch, seq_len, num_heads, head_dim]
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim]);
        let v = v.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim]);

        // Transpose to [batch, num_heads, seq_len, head_dim]
        let q = q.transpose_2d();
        let k = k.transpose_2d();
        let v = v.transpose_2d();

        // Scaled dot-product attention for each head
        let k_t = k.transpose(None);
        let mut scores = q.matmul(&k_t);
        scores = scores.scale(self.scale);
        let attn_weights = scores.softmax(-1);
        let attn_output = attn_weights.matmul(&v);

        // Transpose back: [batch, seq_len, num_heads, head_dim]
        let attn_output = attn_output.transpose_2d();

        // Reshape: [batch, seq_len, hidden_dim]
        let hidden_dim = self.num_heads * self.head_dim;
        let attn_output = attn_output.reshape(&[batch_size, seq_len, hidden_dim]);

        // Output projection
        attn_output.matmul(&self.w_o)
    }
}

/// Feed-forward network operation (SwiGLU variant used in LLaMA/Mistral)
#[derive(Debug, Clone)]
pub struct SwiGLUOp {
    /// Gate projection weights
    pub gate_proj: DenseTensor,
    /// Up projection weights
    pub up_proj: DenseTensor,
    /// Down projection weights
    pub down_proj: DenseTensor,
}

impl SwiGLUOp {
    /// Create a new SwiGLU operation
    pub fn new(gate_proj: DenseTensor, up_proj: DenseTensor, down_proj: DenseTensor) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &DenseTensor) -> DenseTensor {
        // gate = SiLU(x @ W_gate)
        let gate = x.matmul(&self.gate_proj);
        let gate = gate.silu();

        // up = x @ W_up
        let up = x.matmul(&self.up_proj);

        // output = (gate * up) @ W_down
        let intermediate = gate.mul(&up);
        intermediate.matmul(&self.down_proj)
    }
}

/// Layer normalization operation
#[derive(Debug, Clone)]
pub struct LayerNormOp {
    /// Weight (gamma)
    pub weight: DenseTensor,
    /// Bias (beta)
    pub bias: DenseTensor,
    /// Epsilon for numerical stability
    pub epsilon: f64,
}

impl LayerNormOp {
    /// Create a new layer normalization operation
    pub fn new(weight: DenseTensor, bias: DenseTensor, epsilon: f64) -> Self {
        Self {
            weight,
            bias,
            epsilon,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &DenseTensor) -> DenseTensor {
        let mean = x.mean_dim(-1);
        let var = x.var_dim(-1);

        let normalized = x.sub(&mean).div(
            &var.add(&DenseTensor::full(var.shape(), self.epsilon))
                .sqrt(),
        );
        normalized.mul(&self.weight).add(&self.bias)
    }
}

/// RMS normalization operation (used in LLaMA/Mistral)
#[derive(Debug, Clone)]
pub struct RMSNormOp {
    /// Weight
    pub weight: DenseTensor,
    /// Epsilon for numerical stability
    pub epsilon: f64,
}

impl RMSNormOp {
    /// Create a new RMS normalization operation
    pub fn new(weight: DenseTensor, epsilon: f64) -> Self {
        Self { weight, epsilon }
    }

    /// Forward pass
    pub fn forward(&self, x: &DenseTensor) -> DenseTensor {
        let ndim = x.ndim();

        if ndim == 3 {
            // 3D tensor: [batch, seq, hidden]
            let batch = x.shape()[0];
            let seq = x.shape()[1];
            let hidden = x.shape()[2];

            // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
            let x_squared = x.mul(x);
            let mean_square = x_squared.mean_dim(-1);

            // Expand from [batch, seq, 1] to [batch, seq, hidden]
            let mean_square_expanded = mean_square.expand_last_dim(hidden);
            let eps_tensor = DenseTensor::full(&[batch, seq, hidden], self.epsilon);
            let rms_input = mean_square_expanded.add(&eps_tensor);
            let rms = rms_input.sqrt();

            let normalized = x.div(&rms);

            // Expand weight from [hidden] to [batch, seq, hidden]
            let weight_expanded = self.weight.expand_to_3d(batch, seq);
            normalized.mul(&weight_expanded)
        } else if ndim == 2 {
            // 2D tensor: [seq, hidden]
            let seq = x.shape()[0];
            let hidden = x.shape()[1];

            // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
            let x_squared = x.mul(x);
            let mean_square = x_squared.mean_dim(-1);

            // Expand from [seq, 1] to [seq, hidden]
            let mean_square_expanded = mean_square.expand_last_dim_2d(hidden);
            let eps_tensor = DenseTensor::full(&[seq, hidden], self.epsilon);
            let rms_input = mean_square_expanded.add(&eps_tensor);
            let rms = rms_input.sqrt();

            let normalized = x.div(&rms);

            // Expand weight from [hidden] to [seq, hidden]
            let weight_expanded = self.weight.expand_to_2d(seq);
            normalized.mul(&weight_expanded)
        } else {
            panic!("RMSNormOp only supports 2D or 3D tensors");
        }
    }
}

/// Rotary positional embedding operation
#[derive(Debug, Clone)]
pub struct RoPEOp {
    /// Cosine cache (pre-computed)
    pub cos_cache: DenseTensor,
    /// Sine cache (pre-computed)
    pub sin_cache: DenseTensor,
}

impl RoPEOp {
    /// Create a new RoPE operation
    pub fn new(cos_cache: DenseTensor, sin_cache: DenseTensor) -> Self {
        Self {
            cos_cache,
            sin_cache,
        }
    }

    /// Forward pass - apply rotary position embedding
    pub fn forward(&self, x: &DenseTensor, positions: &[usize]) -> DenseTensor {
        // Apply RoPE: rotate_half(x) * sin + x * cos
        let mut output = x.clone();

        for (i, &pos) in positions.iter().enumerate() {
            let cos = self.cos_cache.get_row(pos);
            let sin = self.sin_cache.get_row(pos);

            // Apply to each position in the sequence
            let x_row = x.get_row(i);
            let rotated = self.rotate_half(&x_row);

            let result = x_row.mul(&cos).add(&rotated.mul(&sin));
            output.set_row(i, &result);
        }

        output
    }

    fn rotate_half(&self, x: &DenseTensor) -> DenseTensor {
        let dim = x.shape()[0];
        let half_dim = dim / 2;

        let mut data = vec![0.0; dim];
        let x_data = x.data();

        // Rotate first half
        for i in 0..half_dim {
            data[i] = -x_data[i + half_dim];
            data[i + half_dim] = x_data[i];
        }

        DenseTensor::new(data, vec![1, dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_op() {
        let weight = DenseTensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
        let bias = DenseTensor::new(vec![0.1, 0.1], vec![1, 2]);
        let linear = LinearOp::new(weight, Some(bias));

        let input = DenseTensor::new(vec![1.0, 2.0], vec![1, 2]);
        let output = linear.forward(&input);

        assert_eq!(output.shape(), &[1, 2]);
    }

    #[test]
    fn test_embedding_op() {
        let embeddings = DenseTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let embedding = EmbeddingOp::new(embeddings);

        let indices = vec![0, 2];
        let output = embedding.forward(&indices);

        assert_eq!(output.shape(), &[2, 2]);
    }

    #[test]
    fn test_scaled_dot_product() {
        let op = ScaledDotProductOp::new(4);

        let q = DenseTensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![1, 4]);
        let k = DenseTensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![1, 4]);
        let v = DenseTensor::new(vec![1.0, 2.0], vec![1, 2]);

        let output = op.forward(&q, &k, &v);

        assert_eq!(output.shape(), &[1, 2]);
    }

    #[test]
    fn test_rms_norm() {
        let weight = DenseTensor::ones(vec![4]);
        let rms_norm = RMSNormOp::new(weight, 1e-6);

        let x = DenseTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let output = rms_norm.forward(&x);

        assert_eq!(output.shape(), &[1, 4]);

        // RMS should be close to 1 after normalization
        let rms = output.clone().mul(&output).mean_dim(-1);
        let rms_val = rms.data()[0];
        assert!((rms_val - 1.0).abs() < 0.1);
    }
}
