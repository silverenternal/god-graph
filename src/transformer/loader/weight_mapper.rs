//! Weight mapper for converting Safetensors weights to model structures
//!
//! This module provides mapping from HuggingFace Safetensors weight names
//! to GodGraph model structures (LlamaModel, etc.).

use crate::errors::{GraphError, GraphResult};
use crate::tensor::traits::TensorBase;
use crate::tensor::DenseTensor;
use crate::transformer::layers::{FeedForward, MultiHeadAttention, RMSNorm, RoPE};
use crate::transformer::loader::config::LlamaConfig;
use crate::transformer::model::{LlamaDecoderLayer, LlamaModel as LlamaModelStruct};
use std::collections::HashMap;

/// Weight mapper for LLaMA models
///
/// Maps Safetensors weight names to LlamaModel components
pub struct LlamaWeightMapper {
    config: LlamaConfig,
}

impl LlamaWeightMapper {
    /// Create a new LLaMA weight mapper
    pub fn new(config: LlamaConfig) -> Self {
        Self { config }
    }

    /// Get the model configuration
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Build a complete LlamaModel from loaded tensors
    ///
    /// # Arguments
    ///
    /// * `tensors` - Map of tensor names to tensor data
    ///
    /// # Returns
    ///
    /// Complete LlamaModelStruct with all weights loaded
    pub fn build_model(
        &self,
        tensors: &HashMap<String, DenseTensor>,
    ) -> GraphResult<LlamaModelStruct> {
        // Extract token embeddings
        let embed_tokens = tensors
            .get("model.embed_tokens.weight")
            .ok_or_else(|| GraphError::NotFound("model.embed_tokens.weight".to_string()))?
            .clone();

        // Build decoder layers
        let mut layers = Vec::new();
        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = self.build_layer(layer_idx, tensors)?;
            layers.push(layer);
        }

        // Extract final layer norm
        let norm = RMSNorm::new(
            tensors
                .get("model.norm.weight")
                .ok_or_else(|| GraphError::NotFound("model.norm.weight".to_string()))?
                .clone(),
            self.config.rms_norm_eps,
        );

        // Extract language model head
        let lm_head = tensors
            .get("lm_head.weight")
            .ok_or_else(|| GraphError::NotFound("lm_head.weight".to_string()))?
            .clone();

        Ok(LlamaModelStruct {
            embed_tokens: DenseTensor::new(
                embed_tokens.data().to_vec(),
                embed_tokens.shape().to_vec(),
            ),
            layers,
            norm,
            lm_head: Some(DenseTensor::new(
                lm_head.data().to_vec(),
                lm_head.shape().to_vec(),
            )),
            config: self.config.clone(),
            rope: RoPE::new(
                self.config.head_dim(),
                self.config.max_position_embeddings,
                self.config.rope_theta,
            ),
        })
    }

    /// Build a single decoder layer
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Layer index (0-based)
    /// * `tensors` - Map of tensor names to tensor data
    ///
    /// # Returns
    ///
    /// LlamaDecoderLayer with weights loaded
    pub fn build_layer(
        &self,
        layer_idx: usize,
        tensors: &HashMap<String, DenseTensor>,
    ) -> GraphResult<LlamaDecoderLayer> {
        let prefix = format!("model.layers.{}", layer_idx);

        // Extract attention weights
        let q_proj = tensors
            .get(&format!("{}.self_attn.q_proj.weight", prefix))
            .ok_or_else(|| GraphError::NotFound(format!("{}.self_attn.q_proj.weight", prefix)))?
            .clone();

        let k_proj = tensors
            .get(&format!("{}.self_attn.k_proj.weight", prefix))
            .ok_or_else(|| GraphError::NotFound(format!("{}.self_attn.k_proj.weight", prefix)))?
            .clone();

        let v_proj = tensors
            .get(&format!("{}.self_attn.v_proj.weight", prefix))
            .ok_or_else(|| GraphError::NotFound(format!("{}.self_attn.v_proj.weight", prefix)))?
            .clone();

        let o_proj = tensors
            .get(&format!("{}.self_attn.o_proj.weight", prefix))
            .ok_or_else(|| GraphError::NotFound(format!("{}.self_attn.o_proj.weight", prefix)))?
            .clone();

        // Build multi-head attention
        let self_attn = MultiHeadAttention::new(
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            self.config.num_attention_heads,
            self.config.get_num_key_value_heads(),
        );

        // Extract FFN weights (SwiGLU)
        let gate_proj = tensors
            .get(&format!("{}.mlp.gate_proj.weight", prefix))
            .ok_or_else(|| GraphError::NotFound(format!("{}.mlp.gate_proj.weight", prefix)))?
            .clone();

        let up_proj = tensors
            .get(&format!("{}.mlp.up_proj.weight", prefix))
            .ok_or_else(|| GraphError::NotFound(format!("{}.mlp.up_proj.weight", prefix)))?
            .clone();

        let down_proj = tensors
            .get(&format!("{}.mlp.down_proj.weight", prefix))
            .ok_or_else(|| GraphError::NotFound(format!("{}.mlp.down_proj.weight", prefix)))?
            .clone();

        // Build SwiGLU feed-forward network
        let mlp = FeedForward::swiglu(gate_proj, up_proj, down_proj);

        // Extract normalization weights
        let input_layernorm = RMSNorm::new(
            tensors
                .get(&format!("{}.input_layernorm.weight", prefix))
                .ok_or_else(|| GraphError::NotFound(format!("{}.input_layernorm.weight", prefix)))?
                .clone(),
            self.config.rms_norm_eps,
        );

        let post_attention_layernorm = RMSNorm::new(
            tensors
                .get(&format!("{}.post_attention_layernorm.weight", prefix))
                .ok_or_else(|| {
                    GraphError::NotFound(format!("{}.post_attention_layernorm.weight", prefix))
                })?
                .clone(),
            self.config.rms_norm_eps,
        );

        Ok(LlamaDecoderLayer::new(
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        ))
    }

    /// Extract a specific weight tensor by layer and component
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Layer index
    /// * `component` - Component name (e.g., "q_proj", "k_proj", "mlp.gate_proj")
    /// * `tensors` - Map of tensor names to tensor data
    ///
    /// # Returns
    ///
    /// The requested weight tensor
    pub fn get_weight<'a>(
        &self,
        layer_idx: usize,
        component: &str,
        tensors: &'a HashMap<String, DenseTensor>,
    ) -> GraphResult<&'a DenseTensor> {
        let name = format!("model.layers.{}.{}", layer_idx, component);
        tensors.get(&name).ok_or(GraphError::NotFound(name))
    }
}

/// LLaMA model structure with loaded weights
#[derive(Debug, Clone)]
pub struct LlamaModel {
    /// Token embedding matrix
    pub embed_tokens: DenseTensor,
    /// Decoder layers
    pub layers: Vec<LlamaDecoderLayer>,
    /// Final layer normalization
    pub norm: RMSNorm,
    /// Language model head
    pub lm_head: DenseTensor,
    /// Model configuration
    pub config: LlamaConfig,
}

impl LlamaModel {
    /// Get the number of parameters in the model
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Embeddings
        total += self.embed_tokens.shape().iter().product::<usize>();

        // Each decoder layer
        for layer in &self.layers {
            total += layer.num_parameters();
        }

        // Final norm
        total += self.norm.weight.shape().iter().product::<usize>();

        // LM head
        total += self.lm_head.shape().iter().product::<usize>();

        total
    }

    /// Get model size in MB (assuming f64)
    pub fn size_mb(&self) -> f64 {
        (self.num_parameters() * 8) as f64 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_weight_mapper_creation() {
        let config = LlamaConfig::llama_7b();
        let mapper = LlamaWeightMapper::new(config.clone());

        assert_eq!(mapper.config().vocab_size, config.vocab_size);
        assert_eq!(mapper.config().hidden_size, config.hidden_size);
    }

    #[test]
    fn test_llama_model_structure() {
        // Create a minimal mock model for testing
        let config = LlamaConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: Some(8),
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            attention_bias: false,
        };

        let embed_tokens = DenseTensor::from_vec(
            vec![1.0; config.vocab_size * config.hidden_size],
            vec![config.vocab_size, config.hidden_size],
        );

        let lm_head = DenseTensor::from_vec(
            vec![1.0; config.vocab_size * config.hidden_size],
            vec![config.vocab_size, config.hidden_size],
        );

        let norm_weight =
            DenseTensor::from_vec(vec![1.0; config.hidden_size], vec![config.hidden_size]);

        let norm = RMSNorm::new(norm_weight, config.rms_norm_eps);

        // Create mock layers (this would normally come from the mapper)
        let layers = Vec::new(); // Empty for this test

        let rope = RoPE::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
        );

        let model = LlamaModelStruct {
            embed_tokens,
            layers,
            norm,
            lm_head: Some(lm_head),
            config: config.clone(),
            rope,
        };

        // Verify parameter count calculation
        assert!(model.num_parameters() > 0);
        assert!(model.size_mb() > 0.0);
    }
}
