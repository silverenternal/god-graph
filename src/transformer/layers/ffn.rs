//! Feed-Forward Network implementations

use crate::tensor::DenseTensor;
use crate::tensor::traits::{TensorOps, TensorBase};

/// Feed-Forward Network types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedForwardType {
    /// Standard FFN: FC2(activation(FC1(x)))
    Standard,
    /// SwiGLU: down_proj(SiLU(gate_proj(x)) * up_proj(x))
    SwiGLU,
    /// GeGLU: down_proj(GELU(gate_proj(x)) * up_proj(x))
    GeGLU,
}

/// Feed-Forward Network layer
///
/// Supports multiple variants:
/// - Standard: Used in original Transformer
/// - SwiGLU: Used in LLaMA, Mistral
/// - GeGLU: Used in some variants
#[derive(Debug, Clone)]
pub struct FeedForward {
    /// Type of FFN
    pub ff_type: FeedForwardType,

    /// For standard FFN
    /// First linear layer
    pub fc1: Option<DenseTensor>,
    /// Second linear layer
    pub fc2: Option<DenseTensor>,
    /// Activation function
    pub activation: Option<Activation>,

    /// For SwiGLU/GeGLU
    /// Gate projection
    pub gate_proj: Option<DenseTensor>,
    /// Up projection
    pub up_proj: Option<DenseTensor>,
    /// Down projection
    pub down_proj: Option<DenseTensor>,

    /// Intermediate dimension
    pub intermediate_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
}

/// Activation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Rectified Linear Unit
    ReLU,
    /// Gaussian Error Linear Unit
    GELU,
    /// Sigmoid Linear Unit
    SiLU,
}

impl FeedForward {
    /// Create a standard feed-forward network
    ///
    /// # Arguments
    /// * `fc1` - First linear layer [hidden_dim, intermediate_dim]
    /// * `fc2` - Second linear layer [intermediate_dim, hidden_dim]
    /// * `activation` - Activation function
    pub fn standard(fc1: DenseTensor, fc2: DenseTensor, activation: Activation) -> Self {
        let hidden_dim = fc1.shape()[0];
        let intermediate_dim = fc1.shape()[1];
        
        Self {
            ff_type: FeedForwardType::Standard,
            fc1: Some(fc1),
            fc2: Some(fc2),
            activation: Some(activation),
            gate_proj: None,
            up_proj: None,
            down_proj: None,
            intermediate_dim,
            hidden_dim,
        }
    }

    /// Create a SwiGLU feed-forward network (used in LLaMA/Mistral)
    ///
    /// # Arguments
    /// * `gate_proj` - Gate projection [hidden_dim, intermediate_dim]
    /// * `up_proj` - Up projection [hidden_dim, intermediate_dim]
    /// * `down_proj` - Down projection [intermediate_dim, hidden_dim]
    pub fn swiglu(gate_proj: DenseTensor, up_proj: DenseTensor, down_proj: DenseTensor) -> Self {
        let hidden_dim = gate_proj.shape()[0];
        let intermediate_dim = gate_proj.shape()[1];
        
        Self {
            ff_type: FeedForwardType::SwiGLU,
            fc1: None,
            fc2: None,
            activation: None,
            gate_proj: Some(gate_proj),
            up_proj: Some(up_proj),
            down_proj: Some(down_proj),
            intermediate_dim,
            hidden_dim,
        }
    }

    /// Create a GeGLU feed-forward network
    ///
    /// # Arguments
    /// * `gate_proj` - Gate projection [hidden_dim, intermediate_dim]
    /// * `up_proj` - Up projection [hidden_dim, intermediate_dim]
    /// * `down_proj` - Down projection [intermediate_dim, hidden_dim]
    pub fn geglu(gate_proj: DenseTensor, up_proj: DenseTensor, down_proj: DenseTensor) -> Self {
        let hidden_dim = gate_proj.shape()[0];
        let intermediate_dim = gate_proj.shape()[1];
        
        Self {
            ff_type: FeedForwardType::GeGLU,
            fc1: None,
            fc2: None,
            activation: None,
            gate_proj: Some(gate_proj),
            up_proj: Some(up_proj),
            down_proj: Some(down_proj),
            intermediate_dim,
            hidden_dim,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch_size, seq_len, hidden_dim]
    ///
    /// # Returns
    /// Output tensor [batch_size, seq_len, hidden_dim]
    pub fn forward(&self, x: &DenseTensor) -> DenseTensor {
        match self.ff_type {
            FeedForwardType::Standard => self.forward_standard(x),
            FeedForwardType::SwiGLU => self.forward_swiglu(x),
            FeedForwardType::GeGLU => self.forward_geglu(x),
        }
    }

    /// Standard FFN forward: FC2(activation(FC1(x)))
    fn forward_standard(&self, x: &DenseTensor) -> DenseTensor {
        let fc1 = self.fc1.as_ref().expect("FC1 not initialized");
        let fc2 = self.fc2.as_ref().expect("FC2 not initialized");
        let activation = self.activation.expect("Activation not set");

        // FC1 using batched matmul
        let hidden = x.bmm_broadcast_weight(fc1);

        // Activation
        let activated = match activation {
            Activation::ReLU => hidden.relu(),
            Activation::GELU => hidden.gelu(),
            Activation::SiLU => hidden.silu(),
        };

        // FC2 using batched matmul
        activated.bmm_broadcast_weight(fc2)
    }

    /// SwiGLU forward: down_proj(SiLU(gate_proj(x)) * up_proj(x))
    fn forward_swiglu(&self, x: &DenseTensor) -> DenseTensor {
        let gate_proj = self.gate_proj.as_ref().expect("gate_proj not initialized");
        let up_proj = self.up_proj.as_ref().expect("up_proj not initialized");
        let down_proj = self.down_proj.as_ref().expect("down_proj not initialized");

        // gate = SiLU(x @ gate_proj) using batched matmul
        let gate = x.bmm_broadcast_weight(gate_proj);
        let gate = gate.silu();

        // up = x @ up_proj using batched matmul
        let up = x.bmm_broadcast_weight(up_proj);

        // output = (gate * up) @ down_proj using batched matmul
        let intermediate = gate.mul(&up);
        intermediate.bmm_broadcast_weight(down_proj)
    }

    /// GeGLU forward: down_proj(GELU(gate_proj(x)) * up_proj(x))
    fn forward_geglu(&self, x: &DenseTensor) -> DenseTensor {
        let gate_proj = self.gate_proj.as_ref().expect("gate_proj not initialized");
        let up_proj = self.up_proj.as_ref().expect("up_proj not initialized");
        let down_proj = self.down_proj.as_ref().expect("down_proj not initialized");

        // gate = GELU(x @ gate_proj) using batched matmul
        let gate = x.bmm_broadcast_weight(gate_proj);
        let gate = gate.gelu();

        // up = x @ up_proj using batched matmul
        let up = x.bmm_broadcast_weight(up_proj);

        // output = (gate * up) @ down_proj using batched matmul
        let intermediate = gate.mul(&up);
        intermediate.bmm_broadcast_weight(down_proj)
    }

    /// Get intermediate dimension
    pub fn intermediate_dim(&self) -> usize {
        self.intermediate_dim
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get the number of parameters in this FFN layer
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;

        match self.ff_type {
            FeedForwardType::Standard => {
                if let Some(ref fc1) = self.fc1 {
                    total += fc1.shape().iter().product::<usize>();
                }
                if let Some(ref fc2) = self.fc2 {
                    total += fc2.shape().iter().product::<usize>();
                }
            }
            FeedForwardType::SwiGLU | FeedForwardType::GeGLU => {
                if let Some(ref gate_proj) = self.gate_proj {
                    total += gate_proj.shape().iter().product::<usize>();
                }
                if let Some(ref up_proj) = self.up_proj {
                    total += up_proj.shape().iter().product::<usize>();
                }
                if let Some(ref down_proj) = self.down_proj {
                    total += down_proj.shape().iter().product::<usize>();
                }
            }
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_ffn() {
        let hidden_dim = 8;
        let intermediate_dim = 32;

        let fc1 = DenseTensor::ones(vec![hidden_dim, intermediate_dim]);
        let fc2 = DenseTensor::ones(vec![intermediate_dim, hidden_dim]);

        let ffn = FeedForward::standard(fc1, fc2, Activation::GELU);

        let batch_size = 2;
        let seq_len = 4;
        let x = DenseTensor::ones(vec![batch_size, seq_len, hidden_dim]);
        let output = ffn.forward(&x);

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_dim]);
    }

    #[test]
    fn test_swiglu_ffn() {
        let hidden_dim = 8;
        let intermediate_dim = 32;

        let gate_proj = DenseTensor::ones(vec![hidden_dim, intermediate_dim]);
        let up_proj = DenseTensor::ones(vec![hidden_dim, intermediate_dim]);
        let down_proj = DenseTensor::ones(vec![intermediate_dim, hidden_dim]);

        let ffn = FeedForward::swiglu(gate_proj, up_proj, down_proj);

        let batch_size = 2;
        let seq_len = 4;
        let x = DenseTensor::ones(vec![batch_size, seq_len, hidden_dim]);
        let output = ffn.forward(&x);

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_dim]);
    }
}
