//! Transformer Integration Tests
//!
//! End-to-end tests for transformer model inference and generation

#[cfg(feature = "transformer")]
mod transformer_tests {
    use god_gragh::tensor::traits::TensorBase;
    use god_gragh::tensor::DenseTensor;
    use god_gragh::transformer::batch::BatchData;
    use god_gragh::transformer::generation::{GenerationConfig, TextGenerator};
    use god_gragh::transformer::kv_cache::KVCache;
    use god_gragh::transformer::layers::{FeedForward, MultiHeadAttention, RMSNorm};
    use god_gragh::transformer::loader::LlamaConfig;
    use god_gragh::transformer::model::{LlamaDecoderLayer, LlamaModel};
    use god_gragh::transformer::quantization::{QuantizationConfig, QuantizedTensor};
    use god_gragh::transformer::sparse_attention::SparseMask;

    /// Test creating and running a small Llama model
    #[test]
    fn test_llama_model_creation() {
        let mut config = LlamaConfig::llama_7b();
        // Use smaller dimensions for testing
        config.vocab_size = 1000;
        config.hidden_size = 128;
        config.intermediate_size = 256;
        config.num_hidden_layers = 1;
        config.num_attention_heads = 4;

        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;

        // Create token embeddings
        let embed_tokens = DenseTensor::ones(vec![vocab_size, hidden_size]);

        // Create a single decoder layer
        let hidden_dim = hidden_size;
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

        let layer =
            LlamaDecoderLayer::new(self_attn, mlp, input_layernorm, post_attention_layernorm);

        let layers = vec![layer];
        let norm = RMSNorm::default(hidden_dim);

        let model = LlamaModel::new(config, embed_tokens, layers, norm, None);

        assert_eq!(model.config.vocab_size, vocab_size);
        assert_eq!(model.layers.len(), 1);
    }

    /// Test model forward pass with batched input
    #[test]
    fn test_llama_forward_pass() {
        let mut config = LlamaConfig::llama_7b();
        config.vocab_size = 100;
        config.hidden_size = 64;
        config.intermediate_size = 128;
        config.num_hidden_layers = 2;
        config.num_attention_heads = 4;

        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;

        // Create token embeddings
        let embed_tokens = DenseTensor::ones(vec![vocab_size, hidden_size]);

        // Create decoder layers
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            let hidden_dim = hidden_size;
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

            let layer =
                LlamaDecoderLayer::new(self_attn, mlp, input_layernorm, post_attention_layernorm);
            layers.push(layer);
        }

        let norm = RMSNorm::default(hidden_size);
        let model = LlamaModel::new(config, embed_tokens, layers, norm, None);

        // Create batched input [batch_size=2, seq_len=4]
        let input_ids = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];

        let logits = model.forward(&input_ids, None);

        // Output shape should be [batch_size, seq_len, vocab_size]
        let shape = logits.shape();
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[0], 2); // batch_size
        assert_eq!(shape[1], 4); // seq_len
        assert_eq!(shape[2], vocab_size);
    }

    /// Test text generation with greedy decoding
    #[test]
    fn test_greedy_generation() {
        let mut config = LlamaConfig::llama_7b();
        config.vocab_size = 50;
        config.hidden_size = 32;
        config.intermediate_size = 64;
        config.num_hidden_layers = 1;
        config.num_attention_heads = 2;

        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;

        let embed_tokens = DenseTensor::ones(vec![vocab_size, hidden_size]);

        let hidden_dim = hidden_size;
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

        let layer =
            LlamaDecoderLayer::new(self_attn, mlp, input_layernorm, post_attention_layernorm);

        let layers = vec![layer];
        let norm = RMSNorm::default(hidden_size);

        let model = LlamaModel::new(config, embed_tokens, layers, norm, None);

        // Create text generator
        let gen_config = GenerationConfig::greedy();
        let generator = TextGenerator::new(&model, gen_config);

        // Generate from a prompt
        let prompt = vec![1, 2, 3];
        let output = generator.generate(&prompt);

        assert!(!output.is_empty());
        assert!(output.len() >= prompt.len());
    }

    /// Test batched inference
    #[test]
    fn test_batched_inference() {
        let mut config = LlamaConfig::llama_7b();
        config.vocab_size = 100;
        config.hidden_size = 64;
        config.intermediate_size = 128;
        config.num_hidden_layers = 1;
        config.num_attention_heads = 4;

        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;

        let embed_tokens = DenseTensor::ones(vec![vocab_size, hidden_size]);

        let hidden_dim = hidden_size;
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

        let layer =
            LlamaDecoderLayer::new(self_attn, mlp, input_layernorm, post_attention_layernorm);

        let layers = vec![layer];
        let norm = RMSNorm::default(hidden_dim);

        let model = LlamaModel::new(config, embed_tokens, layers, norm, None);

        // Create batched data
        let sequences = vec![vec![1, 2, 3, 4, 5], vec![6, 7, 8], vec![9, 10, 11, 12]];

        let batch = BatchData::new(sequences);

        assert_eq!(batch.batch_size(), 3);

        // Test forward pass with batch
        let _logits = model.forward(batch.padded_input_ids(), None);
    }

    /// Test KV Cache functionality
    #[test]
    fn test_kv_cache() {
        let num_layers = 2;
        let seq_len = 128;
        let hidden_dim = 64;
        let num_kv_heads = 4;
        let head_dim = hidden_dim / num_kv_heads; // = 16

        let mut cache = KVCache::new(num_layers, seq_len, hidden_dim, num_kv_heads);

        // Create dummy key and value tensors for layer 0
        // Shape should be [batch_size=1, num_kv_heads, head_dim]
        let key = DenseTensor::ones(vec![1, num_kv_heads, head_dim]);
        let value = DenseTensor::ones(vec![1, num_kv_heads, head_dim]);

        // Update cache at position 0 for layer 0
        cache.update(0, &key, &value, 0);

        // Retrieve cached key and value for layer 0
        let (cached_key, cached_value) = cache.get(0, None).unwrap();

        // Cache shape should be [current_len=1, num_kv_heads, head_dim]
        assert_eq!(cached_key.shape(), &[1, num_kv_heads, head_dim]);
        assert_eq!(cached_value.shape(), &[1, num_kv_heads, head_dim]);
        assert_eq!(cache.current_len(), 1);
    }

    /// Test sparse attention mask
    #[test]
    fn test_sparse_attention() {
        // Create a sliding window mask
        let seq_len = 64;
        let window_size = 16;

        let mask = SparseMask::sliding_window(seq_len, window_size, true);

        assert!(!mask.row_offsets.is_empty());
        assert_eq!(mask.row_offsets.len(), seq_len + 1);
    }

    /// Test quantization (INT8 per-tensor)
    #[test]
    fn test_int8_quantization() {
        // Create a tensor with known values
        let data: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let shape = vec![4, 4];
        let tensor = DenseTensor::new(data.clone(), shape);

        // Quantize to INT8 (per-tensor)
        let config = QuantizationConfig::int8();
        let quantized = QuantizedTensor::from_tensor(&tensor, config);

        // Dequantize and check error is bounded
        let dequantized = quantized.dequantize();

        assert_eq!(dequantized.shape(), tensor.shape());

        // Check reconstruction error
        let orig_data = tensor.data();
        let recon_data = dequantized.data();

        for (orig, recon) in orig_data.iter().zip(recon_data.iter()) {
            let error = (orig - recon).abs();
            assert!(error < 0.15, "Reconstruction error too large: {}", error);
        }
    }
}
