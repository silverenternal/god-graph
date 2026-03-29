//! Normalization layer implementations

use crate::tensor::DenseTensor;
use crate::tensor::traits::{TensorOps, TensorBase};

/// Layer Normalization
///
/// Implements: y = (x - mean) / sqrt(var + eps) * weight + bias
///
/// Used in standard Transformer architectures.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Weight parameter [hidden_dim]
    pub weight: DenseTensor,
    /// Bias parameter [hidden_dim]
    pub bias: DenseTensor,
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Hidden dimension
    pub hidden_dim: usize,
}

impl LayerNorm {
    /// Create a new layer normalization
    ///
    /// # Arguments
    /// * `weight` - Weight parameter [hidden_dim]
    /// * `bias` - Bias parameter [hidden_dim]
    /// * `epsilon` - Epsilon for numerical stability (default: 1e-5)
    pub fn new(weight: DenseTensor, bias: DenseTensor, epsilon: f64) -> Self {
        let hidden_dim = weight.shape()[0];
        Self {
            weight,
            bias,
            epsilon,
            hidden_dim,
        }
    }

    /// Create layer normalization with default parameters
    pub fn default(hidden_dim: usize) -> Self {
        let weight = DenseTensor::ones(vec![hidden_dim]);
        let bias = DenseTensor::zeros(vec![hidden_dim]);
        Self::new(weight, bias, 1e-5)
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
    ///
    /// # Returns
    /// Normalized tensor [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
    pub fn forward(&self, x: &DenseTensor) -> DenseTensor {
        let ndim = x.ndim();
        
        if ndim == 3 {
            // 3D tensor: [batch_size, seq_len, hidden_dim]
            let batch_size = x.shape()[0];
            let seq_len = x.shape()[1];
            let hidden_dim = x.shape()[2];

            // Compute mean along last dimension
            let mean = x.mean_dim(-1);
            
            // Expand mean from [batch, seq, 1] to [batch, seq, hidden] for broadcasting
            let mean_expanded = mean.expand_last_dim(hidden_dim);

            // Compute variance along last dimension
            let var = x.var_dim(-1);
            
            // Expand var and add epsilon
            let var_expanded = var.expand_last_dim(hidden_dim);
            let eps_tensor = DenseTensor::full(&[batch_size, seq_len, hidden_dim], self.epsilon);
            let var_with_eps = var_expanded.add(&eps_tensor);

            // Normalize: (x - mean) / sqrt(var + eps)
            let std = var_with_eps.sqrt();
            let centered = x.sub(&mean_expanded);
            let normalized = centered.div(&std);

            // Scale and shift: * weight + bias
            let weight_expanded = self.weight.expand_to_3d(batch_size, seq_len);
            let bias_expanded = self.bias.expand_to_3d(batch_size, seq_len);
            
            normalized.mul(&weight_expanded).add(&bias_expanded)
        } else if ndim == 2 {
            // 2D tensor: [seq_len, hidden_dim]
            let seq_len = x.shape()[0];
            let hidden_dim = x.shape()[1];

            // Compute mean along last dimension
            let mean = x.mean_dim(-1);
            
            // Expand mean from [seq, 1] to [seq, hidden] for broadcasting
            let mean_expanded = mean.expand_last_dim_2d(hidden_dim);

            // Compute variance along last dimension
            let var = x.var_dim(-1);
            
            // Expand var and add epsilon
            let var_expanded = var.expand_last_dim_2d(hidden_dim);
            let eps_tensor = DenseTensor::full(&[seq_len, hidden_dim], self.epsilon);
            let var_with_eps = var_expanded.add(&eps_tensor);

            // Normalize: (x - mean) / sqrt(var + eps)
            let std = var_with_eps.sqrt();
            let centered = x.sub(&mean_expanded);
            let normalized = centered.div(&std);

            // Scale and shift: * weight + bias
            let weight_expanded = self.weight.expand_to_2d(seq_len);
            let bias_expanded = self.bias.expand_to_2d(seq_len);
            
            normalized.mul(&weight_expanded).add(&bias_expanded)
        } else {
            panic!("LayerNorm only supports 2D or 3D tensors");
        }
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}

/// RMS (Root Mean Square) Normalization
///
/// Implements: y = x / sqrt(mean(x^2) + eps) * weight
///
/// Used in LLaMA and Mistral models. Faster than LayerNorm as it doesn't center the data.
#[derive(Debug, Clone)]
pub struct RMSNorm {
    /// Weight parameter [hidden_dim]
    pub weight: DenseTensor,
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Hidden dimension
    pub hidden_dim: usize,
}

impl RMSNorm {
    /// Create a new RMS normalization
    ///
    /// # Arguments
    /// * `weight` - Weight parameter [hidden_dim]
    /// * `epsilon` - Epsilon for numerical stability (LLaMA uses 1e-6)
    pub fn new(weight: DenseTensor, epsilon: f64) -> Self {
        let hidden_dim = weight.shape()[0];
        Self {
            weight,
            epsilon,
            hidden_dim,
        }
    }

    /// Create RMS normalization with default parameters
    pub fn default(hidden_dim: usize) -> Self {
        let weight = DenseTensor::ones(vec![hidden_dim]);
        Self::new(weight, 1e-6)
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
    ///
    /// # Returns
    /// Normalized tensor [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
    pub fn forward(&self, x: &DenseTensor) -> DenseTensor {
        let ndim = x.ndim();
        
        if ndim == 3 {
            let batch_size = x.shape()[0];
            let seq_len = x.shape()[1];
            let hidden_dim = x.shape()[2];

            // Compute mean of squares: mean(x^2)
            let x_squared = x.mul(x);
            let mean_square = x_squared.mean_dim(-1);

            // Expand mean_square from [batch, seq, 1] to [batch, seq, hidden]
            let mean_square_expanded = mean_square.expand_last_dim(hidden_dim);
            
            // Add epsilon
            let eps_tensor = DenseTensor::full(&[batch_size, seq_len, hidden_dim], self.epsilon);
            let rms_input = mean_square_expanded.add(&eps_tensor);

            // Compute RMS: sqrt(mean(x^2) + eps)
            let rms = rms_input.sqrt();

            // Normalize: x / rms
            let normalized = x.div(&rms);

            // Scale: * weight
            let weight_expanded = self.weight.expand_to_3d(batch_size, seq_len);
            normalized.mul(&weight_expanded)
        } else if ndim == 2 {
            let seq_len = x.shape()[0];
            let hidden_dim = x.shape()[1];

            // Compute mean of squares: mean(x^2)
            let x_squared = x.mul(x);
            let mean_square = x_squared.mean_dim(-1);

            // Expand mean_square from [seq, 1] to [seq, hidden]
            let mean_square_expanded = mean_square.expand_last_dim_2d(hidden_dim);
            
            // Add epsilon
            let eps_tensor = DenseTensor::full(&[seq_len, hidden_dim], self.epsilon);
            let rms_input = mean_square_expanded.add(&eps_tensor);

            // Compute RMS: sqrt(mean(x^2) + eps)
            let rms = rms_input.sqrt();

            // Normalize: x / rms
            let normalized = x.div(&rms);

            // Scale: * weight
            let weight_expanded = self.weight.expand_to_2d(seq_len);
            normalized.mul(&weight_expanded)
        } else {
            panic!("RMSNorm only supports 2D or 3D tensors");
        }
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let hidden_dim = 8;
        let layer_norm = LayerNorm::default(hidden_dim);

        let batch_size = 2;
        let seq_len = 4;
        let x = DenseTensor::new(
            (0..batch_size * seq_len * hidden_dim).map(|i| (i % 10) as f64).collect(),
            vec![batch_size, seq_len, hidden_dim],
        );
        
        let output = layer_norm.forward(&x);

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_dim]);

        // Check that output has mean close to 0 and std close to 1
        let mean = output.mean_dim(-1);
        let var = output.var_dim(-1);
        
        for i in 0..mean.shape()[0] {
            for j in 0..mean.shape()[1] {
                let m = mean.data()[i * mean.shape()[1] + j];
                assert!(m.abs() < 1e-5, "Mean should be close to 0, got {}", m);
            }
        }
        
        for i in 0..var.shape()[0] {
            for j in 0..var.shape()[1] {
                let v = var.data()[i * var.shape()[1] + j];
                assert!((v - 1.0).abs() < 0.1, "Variance should be close to 1, got {}", v);
            }
        }
    }

    #[test]
    fn test_rms_norm() {
        let hidden_dim = 8;
        let rms_norm = RMSNorm::default(hidden_dim);

        let batch_size = 2;
        let seq_len = 4;
        let x = DenseTensor::new(
            (0..batch_size * seq_len * hidden_dim).map(|i| (i % 10) as f64).collect(),
            vec![batch_size, seq_len, hidden_dim],
        );
        
        let output = rms_norm.forward(&x);

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_dim]);

        // Check that RMS is close to 1
        let output_squared = output.mul(&output);
        let mean_square = output_squared.mean_dim(-1);
        
        for i in 0..mean_square.shape()[0] {
            for j in 0..mean_square.shape()[1] {
                let ms = mean_square.data()[i * mean_square.shape()[1] + j];
                assert!((ms - 1.0).abs() < 0.1, "RMS^2 should be close to 1, got {}", ms);
            }
        }
    }

    #[test]
    fn test_layer_norm_with_custom_params() {
        let hidden_dim = 4;
        let weight = DenseTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![hidden_dim]);
        let bias = DenseTensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![hidden_dim]);
        let layer_norm = LayerNorm::new(weight, bias, 1e-5);

        let x = DenseTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, hidden_dim]);
        let output = layer_norm.forward(&x);

        assert_eq!(output.shape(), &[1, hidden_dim]);
    }
}
