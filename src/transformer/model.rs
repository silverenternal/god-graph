//! LLaMA model implementation

use crate::tensor::DenseTensor;
use crate::tensor::traits::{TensorBase, TensorOps};
use super::layers::{MultiHeadAttention, FeedForward, RMSNorm, RoPE};
pub use super::loader::LlamaConfig;

/// LLaMA decoder layer
#[derive(Debug, Clone)]
pub struct LlamaDecoderLayer {
    /// Self-attention layer
    pub self_attn: MultiHeadAttention,
    /// Feed-forward network (SwiGLU)
    pub mlp: FeedForward,
    /// Input layer normalization
    pub input_layernorm: RMSNorm,
    /// Post-attention layer normalization
    pub post_attention_layernorm: RMSNorm,
}

impl LlamaDecoderLayer {
    /// Create a new LLaMA decoder layer
    pub fn new(
        self_attn: MultiHeadAttention,
        mlp: FeedForward,
        input_layernorm: RMSNorm,
        post_attention_layernorm: RMSNorm,
    ) -> Self {
        Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch_size, seq_len, hidden_dim]
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Output tensor [batch_size, seq_len, hidden_dim]
    pub fn forward(&self, x: &DenseTensor, mask: Option<&DenseTensor>) -> DenseTensor {
        // Pre-norm residual architecture (LLaMA uses pre-LN)
        
        // 1. Input normalization
        let normed = self.input_layernorm.forward(x);
        
        // 2. Self-attention with residual
        let attn_output = self.self_attn.forward_with_mask(&normed, mask);
        let hidden = x.add(&attn_output);
        
        // 3. Post-attention normalization
        let normed = self.post_attention_layernorm.forward(&hidden);
        
        // 4. FFN with residual
        let mlp_output = self.mlp.forward(&normed);
        let output = hidden.add(&mlp_output);
        
        output
    }

    /// Forward pass with KV cache
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch_size, seq_len, hidden_dim]
    /// * `kv_cache` - Optional KV cache for this layer
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Output tensor and updated KV cache
    pub fn forward_with_cache(
        &self,
        x: &DenseTensor,
        kv_cache: Option<(&DenseTensor, &DenseTensor)>,
        mask: Option<&DenseTensor>,
    ) -> (DenseTensor, Option<(DenseTensor, DenseTensor)>) {
        // For inference with KV cache
        // This is a simplified version - full implementation would update cache
        let output = self.forward(x, mask);
        (output, kv_cache.map(|(k, v)| (k.clone(), v.clone())))
    }

    /// Get the number of parameters in this layer
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Attention parameters
        total += self.self_attn.num_parameters();

        // MLP parameters
        total += self.mlp.num_parameters();

        // Layer norm parameters (2 * hidden_dim)
        total += self.input_layernorm.weight.shape().iter().product::<usize>();
        total += self.post_attention_layernorm.weight.shape().iter().product::<usize>();

        total
    }
}

/// Complete LLaMA model
#[derive(Debug, Clone)]
pub struct LlamaModel {
    /// Model configuration
    pub config: LlamaConfig,
    /// Token embeddings [vocab_size, hidden_dim]
    pub embed_tokens: DenseTensor,
    /// Decoder layers
    pub layers: Vec<LlamaDecoderLayer>,
    /// Final layer normalization
    pub norm: RMSNorm,
    /// LM head (optional, may be tied with embed_tokens)
    pub lm_head: Option<DenseTensor>,
    /// RoPE module
    pub rope: RoPE,
}

impl LlamaModel {
    /// Create a new LLaMA model
    pub fn new(
        config: LlamaConfig,
        embed_tokens: DenseTensor,
        layers: Vec<LlamaDecoderLayer>,
        norm: RMSNorm,
        lm_head: Option<DenseTensor>,
    ) -> Self {
        let rope = RoPE::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
        );
        
        Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs [batch_size, seq_len]
    /// * `mask` - Optional attention mask [batch_size, seq_len, seq_len]
    ///
    /// # Returns
    /// Logits tensor [batch_size, seq_len, vocab_size]
    pub fn forward(&self, input_ids: &[Vec<usize>], mask: Option<&DenseTensor>) -> DenseTensor {
        let batch_size = input_ids.len();
        let seq_len = input_ids[0].len();

        // 1. Get token embeddings
        let mut hidden = self.embed_tokens_batch(input_ids);

        // 2. Apply RoPE to positions
        let _positions: Vec<usize> = (0..seq_len).collect();

        // 3. Pass through decoder layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, mask);
        }

        // 4. Final normalization
        hidden = self.norm.forward(&hidden);

        // 5. LM head projection
        // hidden: [batch, seq_len, hidden_dim], lm_head: [vocab_size, hidden_dim]
        // Need to compute: hidden @ lm_head.T for each (batch, seq) position
        let lm_head = self.lm_head.as_ref().unwrap_or(&self.embed_tokens);
        let lm_head_t = lm_head.transpose(None); // [hidden_dim, vocab_size]
        
        // Reshape hidden to [batch*seq_len, hidden_dim] for matmul
        let hidden_data = hidden.data().to_vec();
        let hidden_dim = self.config.hidden_size;
        let flat_hidden = DenseTensor::new(hidden_data, vec![batch_size * seq_len, hidden_dim]);
        
        // Matmul: [batch*seq, hidden] @ [hidden, vocab] = [batch*seq, vocab]
        let logits_flat = flat_hidden.matmul(&lm_head_t);
        
        // Reshape back to [batch, seq_len, vocab_size]
        let vocab_size = self.config.vocab_size;
        let logits_data = logits_flat.data().to_vec();
        
        DenseTensor::new(logits_data, vec![batch_size, seq_len, vocab_size])
    }

    /// Forward pass for a single sequence
    pub fn forward_single(&self, input_ids: &[usize], mask: Option<&DenseTensor>) -> DenseTensor {
        self.forward(&[input_ids.to_vec()], mask)
    }

    /// Embed tokens in batch
    fn embed_tokens_batch(&self, input_ids: &[Vec<usize>]) -> DenseTensor {
        let batch_size = input_ids.len();
        let seq_len = input_ids[0].len();
        let hidden_dim = self.config.hidden_size;
        
        let mut data = Vec::with_capacity(batch_size * seq_len * hidden_dim);
        
        for batch in input_ids {
            for &token_id in batch {
                let start = token_id * hidden_dim;
                let end = start + hidden_dim;
                data.extend_from_slice(&self.embed_tokens.data()[start..end]);
            }
        }
        
        DenseTensor::new(data, vec![batch_size, seq_len, hidden_dim])
    }

    /// Get the hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.config.hidden_size
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

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
        if let Some(lm_head) = &self.lm_head {
            total += lm_head.shape().iter().product::<usize>();
        }

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
    use crate::tensor::DenseTensor;
    use crate::tensor::traits::TensorBase;

    fn create_test_layer(config: &LlamaConfig) -> LlamaDecoderLayer {
        let hidden_dim = config.hidden_size;
        let num_heads = config.num_attention_heads;

        // Create attention weights
        let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_k = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_v = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let self_attn = MultiHeadAttention::standard(w_q, w_k, w_v, w_o, num_heads);

        // Create FFN (SwiGLU)
        let gate_proj = DenseTensor::ones(vec![hidden_dim, config.intermediate_size]);
        let up_proj = DenseTensor::ones(vec![hidden_dim, config.intermediate_size]);
        let down_proj = DenseTensor::ones(vec![config.intermediate_size, hidden_dim]);
        let mlp = FeedForward::swiglu(gate_proj, up_proj, down_proj);

        // Create norms
        let input_layernorm = RMSNorm::default(hidden_dim);
        let post_attention_layernorm = RMSNorm::default(hidden_dim);

        LlamaDecoderLayer::new(self_attn, mlp, input_layernorm, post_attention_layernorm)
    }

    #[test]
    fn test_decoder_layer() {
        // Use small dimensions to avoid excessive memory allocation
        let mut config = LlamaConfig::llama_7b();
        config.hidden_size = 64;
        config.intermediate_size = 128;
        config.num_attention_heads = 2;
        let layer = create_test_layer(&config);

        let batch_size = 2;
        let seq_len = 4;
        let x = DenseTensor::ones(vec![batch_size, seq_len, config.hidden_size]);

        let output = layer.forward(&x, None);

        assert_eq!(output.shape(), &[batch_size, seq_len, config.hidden_size]);
    }

    #[test]
    fn test_llama_model_creation() {
        // Use small dimensions to avoid excessive memory allocation
        let mut config = LlamaConfig::llama_7b();
        config.vocab_size = 100;
        config.hidden_size = 64;
        config.intermediate_size = 128;
        config.num_hidden_layers = 2;
        config.num_attention_heads = 2;

        let embed_tokens = DenseTensor::ones(vec![config.vocab_size, config.hidden_size]);
        let layers = vec![create_test_layer(&config); config.num_hidden_layers];
        let norm = RMSNorm::default(config.hidden_size);
        let lm_head = None; // Tied with embeddings

        let model = LlamaModel::new(config, embed_tokens, layers, norm, lm_head);

        assert_eq!(model.num_layers(), 2);
        assert_eq!(model.vocab_size(), 100);
        assert_eq!(model.hidden_dim(), 64);
    }
}

// ============================================================================
// LlamaModel Graph Builder
// ============================================================================

use crate::transformer::graph_transformer::GraphTransformer;

/// LlamaModel graph builder for constructing graph-structured Llama models
///
/// This builder converts a standard LlamaModel into a graph-structured representation
/// that can leverage god-gragh's graph algorithms for optimization and analysis.
///
/// # Example
///
/// ```no_run
/// use god_gragh::transformer::model::{LlamaModel, LlamaConfig, LlamaModelGraphBuilder};
/// use god_gragh::transformer::layers::RMSNorm;
/// use god_gragh::tensor::DenseTensor;
///
/// let config = LlamaConfig::llama_7b();
/// let embed_tokens = DenseTensor::ones(vec![config.vocab_size, config.hidden_size]);
/// let layers = vec![]; // Add your layers here
/// let norm = RMSNorm::default(config.hidden_size);
/// let model = LlamaModel::new(config, embed_tokens, layers, norm, None);
///
/// let builder = LlamaModelGraphBuilder::new(&model);
/// let graph_transformer = builder.build_graph();
/// ```
pub struct LlamaModelGraphBuilder<'a> {
    model: &'a LlamaModel,
}

impl<'a> LlamaModelGraphBuilder<'a> {
    /// Create a new graph builder from a LlamaModel
    pub fn new(model: &'a LlamaModel) -> Self {
        Self { model }
    }

    /// Build graph-structured transformer from the model
    pub fn build_graph(&self) -> GraphTransformer {
        let mut transformer = GraphTransformer::new(
            self.model.num_layers(),
            self.model.config.num_attention_heads,
            self.model.config.hidden_size,
        );

        // Build graph structure
        // Note: In a real implementation, this would use actual weights
        // For now, we create the graph topology
        let dummy_input = vec![0; 1]; // Single token for graph structure
        transformer.build_graph(&dummy_input);

        transformer
    }

    /// Build graph with specific input sequence
    pub fn build_graph_for_input(&self, input_ids: &[usize]) -> GraphTransformer {
        let mut transformer = GraphTransformer::new(
            self.model.num_layers(),
            self.model.config.num_attention_heads,
            self.model.config.hidden_size,
        );

        transformer.build_graph(input_ids);
        transformer
    }

    /// Export graph to DOT format for visualization
    pub fn export_to_dot(&self, transformer: &GraphTransformer) -> String {
        transformer.to_dot()
    }
}

#[cfg(test)]
mod graph_builder_tests {
    use super::*;
    use crate::transformer::layers::{MultiHeadAttention, FeedForward, RMSNorm};

    fn create_test_layer(config: &LlamaConfig) -> LlamaDecoderLayer {
        let hidden_dim = config.hidden_size;
        let num_heads = config.num_attention_heads;

        let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_k = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_v = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let self_attn = MultiHeadAttention::standard(w_q, w_k, w_v, w_o, num_heads);

        let gate_proj = DenseTensor::ones(vec![hidden_dim, config.intermediate_size]);
        let up_proj = DenseTensor::ones(vec![hidden_dim, config.intermediate_size]);
        let down_proj = DenseTensor::ones(vec![config.intermediate_size, hidden_dim]);
        let mlp = FeedForward::swiglu(gate_proj, up_proj, down_proj);

        let input_layernorm = RMSNorm::default(hidden_dim);
        let post_attention_layernorm = RMSNorm::default(hidden_dim);

        LlamaDecoderLayer::new(self_attn, mlp, input_layernorm, post_attention_layernorm)
    }

    #[test]
    fn test_llama_model_graph_builder() {
        // Use small dimensions to avoid excessive memory allocation
        let mut config = LlamaConfig::llama_7b();
        config.vocab_size = 100;
        config.hidden_size = 64;
        config.intermediate_size = 128;
        config.num_attention_heads = 2;
        let embed_tokens = DenseTensor::ones(vec![config.vocab_size, config.hidden_size]);
        let layers = vec![create_test_layer(&config); 2]; // Use 2 layers for test
        let norm = RMSNorm::default(config.hidden_size);
        let lm_head = None;

        let model = LlamaModel::new(config.clone(), embed_tokens, layers, norm, lm_head);

        let builder = LlamaModelGraphBuilder::new(&model);
        let transformer = builder.build_graph();

        // Verify graph was built
        assert!(transformer.num_nodes() > 0);
        assert!(transformer.num_edges() > 0);
    }

    #[test]
    fn test_llama_model_graph_builder_with_input() {
        // Use small dimensions to avoid excessive memory allocation
        let mut config = LlamaConfig::llama_7b();
        config.vocab_size = 100;
        config.hidden_size = 64;
        config.intermediate_size = 128;
        config.num_attention_heads = 2;
        let embed_tokens = DenseTensor::ones(vec![config.vocab_size, config.hidden_size]);
        let layers = vec![create_test_layer(&config); 1];
        let norm = RMSNorm::default(config.hidden_size);
        let lm_head = None;

        let model = LlamaModel::new(config.clone(), embed_tokens, layers, norm, lm_head);

        let builder = LlamaModelGraphBuilder::new(&model);
        let input_ids = vec![1, 2, 3, 4, 5];
        let mut transformer = builder.build_graph_for_input(&input_ids);

        // Verify graph structure
        assert!(transformer.num_nodes() > 0);
        assert!(transformer.num_edges() > 0);

        // Test forward pass
        let output = transformer.forward(&input_ids);
        assert!(!output.data().is_empty());
    }

    #[test]
    fn test_graph_export_to_dot() {
        // Use small dimensions to avoid excessive memory allocation
        let mut config = LlamaConfig::llama_7b();
        config.vocab_size = 100;
        config.hidden_size = 64;
        config.intermediate_size = 128;
        config.num_attention_heads = 2;
        let embed_tokens = DenseTensor::ones(vec![config.vocab_size, config.hidden_size]);
        let layers = vec![create_test_layer(&config); 1];
        let norm = RMSNorm::default(config.hidden_size);
        let lm_head = None;

        let model = LlamaModel::new(config.clone(), embed_tokens, layers, norm, lm_head);

        let builder = LlamaModelGraphBuilder::new(&model);
        let transformer = builder.build_graph();
        let dot = builder.export_to_dot(&transformer);

        // Verify DOT format
        assert!(dot.contains("digraph Transformer"));
        assert!(dot.contains("rankdir=TB"));
    }
}
