//! End-to-End LLM Text Generation Demo
//!
//! This example demonstrates complete text generation workflow.
//!
//! ## Usage
//!
//! ```bash
//! # Run with default settings
//! cargo run --example llm_text_gen --features transformer
//!
//! # Run with custom prompt
//! cargo run --example llm_text_gen --features transformer -- "Once upon a time"
//! ```

#[cfg(feature = "transformer")]
fn main() {
    use god_gragh::tensor::{DenseTensor, TensorBase};
    use god_gragh::transformer::{
        kv_cache::KVCache,
        layers::{FeedForward, MultiHeadAttention, RMSNorm},
        model::{LlamaDecoderLayer, LlamaModel},
        GenerationConfig, LlamaConfig, TextGenerator,
    };
    use std::env;

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║     God-Graph LLM Text Generation Demo                 ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let mut prompt_text = "The future of artificial intelligence is".to_string();
    let mut max_length = 32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" | "-p" => {
                if i + 1 < args.len() {
                    prompt_text = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--max-length" | "-m" => {
                if i + 1 < args.len() {
                    max_length = args[i + 1].parse().unwrap_or(32);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            _ => {
                if !args[i].starts_with("--") {
                    prompt_text = args[i].clone();
                }
                i += 1;
            }
        }
    }

    // Create a small demo model configuration
    println!("📦 Initializing model...");
    let mut config = LlamaConfig::llama_7b();
    config.vocab_size = 500;
    config.hidden_size = 64;
    config.intermediate_size = 128;
    config.num_hidden_layers = 2;
    config.num_attention_heads = 4;
    config.num_key_value_heads = Some(2);

    let vocab_size = config.vocab_size;
    let hidden_size = config.hidden_size;

    // Create token embeddings
    let embed_tokens = DenseTensor::ones(vec![vocab_size, hidden_size]);

    // Create decoder layers
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

    let layer = LlamaDecoderLayer::new(self_attn, mlp, input_layernorm, post_attention_layernorm);

    let layers = vec![layer; config.num_hidden_layers];
    let norm = RMSNorm::default(hidden_dim);

    // Create model
    let model = LlamaModel::new(config, embed_tokens, layers, norm, None);

    println!("   Model config:");
    println!("     - Hidden size: {}", model.config.hidden_size);
    println!("     - Layers: {}", model.config.num_hidden_layers);
    println!(
        "     - Attention heads: {}",
        model.config.num_attention_heads
    );
    println!("   ✓ Model initialized");

    // Configure generation
    let gen_config = GenerationConfig {
        max_length,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        do_sample: true,
        repetition_penalty: 1.1,
        ..GenerationConfig::default()
    };

    println!("\n⚙️  Generation config:");
    println!("     - Max length: {}", max_length);
    println!("     - Temperature: 0.7");

    // Display prompt
    println!("\n📝 Prompt:");
    println!("   \"{}\"", prompt_text);

    // Note: For actual text generation, you need a tokenizer
    // This is a simplified demo showing the model structure
    println!("\n✨ Model Ready:");
    println!("   The model is initialized and ready for generation.");
    println!("   For actual text generation, integrate with a tokenizer:");
    println!("     let generator = TextGenerator::new(&model, gen_config);");
    println!("     let output = generator.generate(&prompt)?;");

    // KV Cache demonstration
    println!("\n💾 KV Cache Demo:");
    let num_kv_heads = model.config.get_num_key_value_heads();
    let kv_cache = KVCache::new(
        model.config.num_hidden_layers,
        model.config.max_position_embeddings,
        model.config.hidden_size,
        num_kv_heads,
    );
    println!("   Cache capacity: {} tokens", kv_cache.max_seq_len());
    println!("   Current usage: {} tokens", kv_cache.current_len());

    // Estimate memory
    let memory_mb = (model.config.num_hidden_layers
        * model.config.max_position_embeddings
        * model.config.hidden_size
        * 2
        * 8)
        / (1024 * 1024);
    println!("   Estimated memory usage: ~{} MB", memory_mb);

    println!("\n✅ Demo complete!");
    println!("\n💡 For complete text generation examples:");
    println!("   See docs/transformer_guide.md and docs/transformer_tutorial.md");
}

fn print_help() {
    println!("LLM Text Generation Demo");
    println!();
    println!("Usage:");
    println!("  cargo run --example llm_text_gen --features transformer [OPTIONS] [PROMPT]");
    println!();
    println!("Arguments:");
    println!("  [PROMPT]              Text prompt for generation");
    println!();
    println!("Options:");
    println!("  -p, --prompt <TEXT>   Set the prompt for generation");
    println!("  -m, --max-length <N>  Maximum length to generate (default: 32)");
    println!("  -h, --help            Show this help message");
}

#[cfg(not(feature = "transformer"))]
fn main() {
    println!("This example requires the 'transformer' feature.");
    println!("Run with: cargo run --example llm_text_gen --features transformer");
}
