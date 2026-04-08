//! Transformer Model Usage Example
//!
//! This example demonstrates how to create and use a small transformer model
//! for inference with the god-gragh library.

#[cfg(feature = "transformer")]
fn main() {
    use god_graph::tensor::traits::TensorBase;
    use god_graph::tensor::DenseTensor;
    use god_graph::transformer::generation::{GenerationConfig, TextGenerator};
    use god_graph::transformer::layers::{FeedForward, MultiHeadAttention, RMSNorm};
    use god_graph::transformer::loader::LlamaConfig;
    use god_graph::transformer::model::{LlamaDecoderLayer, LlamaModel};

    println!("=== Transformer Model Example ===\n");

    // Create a small model configuration for demonstration
    let mut config = LlamaConfig::llama_7b();
    config.vocab_size = 1000; // Small vocab for demo
    config.hidden_size = 128; // Small hidden dim for demo
    config.intermediate_size = 256; // Small FFN dim for demo
    config.num_hidden_layers = 2; // Only 2 layers for demo
    config.num_attention_heads = 4; // 4 attention heads

    println!("Model Configuration:");
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Intermediate size: {}", config.intermediate_size);
    println!("  Num layers: {}", config.num_hidden_layers);
    println!("  Num heads: {}", config.num_attention_heads);
    println!();

    // Create token embeddings
    let vocab_size = config.vocab_size;
    let hidden_size = config.hidden_size;
    let embed_tokens = DenseTensor::ones(vec![vocab_size, hidden_size]);
    println!("Created token embeddings: {:?}", embed_tokens.shape());

    // Create decoder layers
    let mut layers = Vec::new();
    for layer_idx in 0..config.num_hidden_layers {
        let hidden_dim = hidden_size;
        let num_heads = config.num_attention_heads;

        // Create attention weights
        let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_k = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_v = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let self_attn = MultiHeadAttention::standard(w_q, w_k, w_v, w_o, num_heads);

        // Create SwiGLU FFN weights
        let gate_proj = DenseTensor::ones(vec![hidden_dim, config.intermediate_size]);
        let up_proj = DenseTensor::ones(vec![hidden_dim, config.intermediate_size]);
        let down_proj = DenseTensor::ones(vec![config.intermediate_size, hidden_dim]);
        let mlp = FeedForward::swiglu(gate_proj, up_proj, down_proj);

        // Create normalization layers
        let input_layernorm = RMSNorm::default(hidden_dim);
        let post_attention_layernorm = RMSNorm::default(hidden_dim);

        // Create decoder layer
        let layer =
            LlamaDecoderLayer::new(self_attn, mlp, input_layernorm, post_attention_layernorm);

        layers.push(layer);
        println!("Created decoder layer {}", layer_idx);
    }

    // Create final normalization
    let norm = RMSNorm::default(hidden_size);
    println!("Created final RMSNorm\n");

    // Build the model
    let model = LlamaModel::new(config, embed_tokens, layers, norm, None);
    println!("Model created successfully!\n");

    // Example 1: Forward pass with a single sequence
    println!("=== Example 1: Forward Pass ===");
    let input_ids = vec![vec![1, 2, 3, 4, 5]]; // Single sequence of length 5
    let logits = model.forward(&input_ids, None);
    println!("Input shape: [1, 5]");
    println!("Output logits shape: {:?}", logits.shape());
    println!();

    // Example 2: Forward pass with batched input
    println!("=== Example 2: Batched Forward Pass ===");
    let batch_input = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]];
    let batch_logits = model.forward(&batch_input, None);
    println!("Batch input shape: [3, 4]");
    println!("Batch output logits shape: {:?}", batch_logits.shape());
    println!();

    // Example 3: Text generation with greedy decoding
    println!("=== Example 3: Text Generation (Greedy) ===");
    let gen_config = GenerationConfig::greedy();
    let generator = TextGenerator::new(&model, gen_config);

    let prompt = vec![1, 2, 3];
    let output = generator.generate(&prompt);
    println!("Prompt: {:?}", prompt);
    println!("Generated output: {:?}", output);
    println!("Output length: {}", output.len());
    println!();

    // Example 4: Text generation with sampling
    println!("=== Example 4: Text Generation (Sampling) ===");
    let sample_config = GenerationConfig::sampling(0.8); // Temperature = 0.8
    let sampler = TextGenerator::new(&model, sample_config);

    let prompt2 = vec![10, 20, 30];
    let sampled_output = sampler.generate(&prompt2);
    println!("Prompt: {:?}", prompt2);
    println!("Sampled output: {:?}", sampled_output);
    println!("Output length: {}", sampled_output.len());
    println!();

    println!("=== Example Complete ===");
}

#[cfg(not(feature = "transformer"))]
fn main() {
    println!("This example requires the 'transformer' feature to be enabled.");
    println!("Run with: cargo run --example transformer_basic --features transformer");
}
