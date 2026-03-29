//! LLaMA Forward Pass Example
//!
//! This example demonstrates how to:
//! 1. Load a LlamaModel from configuration
//! 2. Build a graph-structured representation
//! 3. Run forward pass using the graph executor
//! 4. Export the computation graph to DOT format for visualization
//!
//! ## Usage
//!
//! ```bash
//! # Run with default tiny model
//! cargo run --features "tensor,safetensors,tensor-pool,transformer" --example llama_forward
//!
//! # Run with specific model size
//! cargo run --features "tensor,safetensors,tensor-pool,transformer" --example llama_forward -- --model 7b
//!
//! # Export graph to DOT
//! cargo run --features "tensor,safetensors,tensor-pool,transformer" --example llama_forward -- --export-dot graph.dot
//! ```

use god_gragh::tensor::traits::TensorBase;
use god_gragh::tensor::DenseTensor;
use god_gragh::transformer::layers::{FeedForward, MultiHeadAttention, RMSNorm};
use god_gragh::transformer::model::{
    LlamaConfig, LlamaDecoderLayer, LlamaModel, LlamaModelGraphBuilder,
};
use std::path::Path;

/// Model size presets
enum ModelSize {
    Tiny,   // 1 layer for testing
    Small,  // 2 layers
    Medium, // 12 layers
    Large,  // 32 layers (7B)
}

impl ModelSize {
    fn to_config(&self) -> LlamaConfig {
        match self {
            ModelSize::Tiny => {
                let mut config = LlamaConfig::llama_7b();
                config.vocab_size = 1000;
                config.hidden_size = 128;
                config.intermediate_size = 256;
                config.num_hidden_layers = 1;
                config.num_attention_heads = 4;
                config.max_position_embeddings = 512;
                config
            }
            ModelSize::Small => {
                let mut config = LlamaConfig::llama_7b();
                config.vocab_size = 1000;
                config.hidden_size = 256;
                config.intermediate_size = 512;
                config.num_hidden_layers = 2;
                config.num_attention_heads = 8;
                config.max_position_embeddings = 512;
                config
            }
            ModelSize::Medium => {
                let mut config = LlamaConfig::llama_7b();
                config.num_hidden_layers = 12;
                config
            }
            ModelSize::Large => LlamaConfig::llama_7b(),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            ModelSize::Tiny => "tiny (1 layer)",
            ModelSize::Small => "small (2 layers)",
            ModelSize::Medium => "medium (12 layers)",
            ModelSize::Large => "large/7B (32 layers)",
        }
    }
}

/// Create a test LlamaModel with random weights
fn create_test_model(config: &LlamaConfig) -> LlamaModel {
    println!("  Creating model...");

    // Create embedding layer
    let embed_tokens = DenseTensor::ones(vec![config.vocab_size, config.hidden_size]);

    // Create decoder layers
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let layer = create_decoder_layer(config, i);
        layers.push(layer);
    }

    // Create final layer norm
    let norm = RMSNorm::default(config.hidden_size);

    // LM head (tied with embeddings for simplicity)
    let lm_head = None;

    LlamaModel::new(config.clone(), embed_tokens, layers, norm, lm_head)
}

/// Create a single decoder layer with random weights
fn create_decoder_layer(config: &LlamaConfig, layer_idx: usize) -> LlamaDecoderLayer {
    let hidden_dim = config.hidden_size;
    let num_heads = config.num_attention_heads;

    // Initialize attention weights with ones (simplified)
    let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_k = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_v = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let self_attn = MultiHeadAttention::standard(w_q, w_k, w_v, w_o, num_heads);

    // Initialize FFN weights
    let gate_proj = DenseTensor::ones(vec![hidden_dim, config.intermediate_size]);
    let up_proj = DenseTensor::ones(vec![hidden_dim, config.intermediate_size]);
    let down_proj = DenseTensor::ones(vec![config.intermediate_size, hidden_dim]);
    let mlp = FeedForward::swiglu(gate_proj, up_proj, down_proj);

    // Layer norms
    let input_layernorm = RMSNorm::default(hidden_dim);
    let post_attention_layernorm = RMSNorm::default(hidden_dim);

    LlamaDecoderLayer::new(self_attn, mlp, input_layernorm, post_attention_layernorm)
}

/// Parse command line arguments
struct Args {
    model_size: ModelSize,
    export_dot: Option<String>,
    input_ids: Vec<usize>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    let mut model_size = ModelSize::Tiny;
    let mut export_dot = None;
    let mut input_ids = vec![1, 2, 3, 4, 5]; // Default input

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                if i + 1 < args.len() {
                    model_size = match args[i + 1].to_lowercase().as_str() {
                        "tiny" => ModelSize::Tiny,
                        "small" => ModelSize::Small,
                        "medium" | "med" => ModelSize::Medium,
                        "large" | "7b" => ModelSize::Large,
                        _ => {
                            eprintln!("Unknown model size: {}. Using tiny.", args[i + 1]);
                            ModelSize::Tiny
                        }
                    };
                    i += 1;
                }
            }
            "--export-dot" | "-e" => {
                if i + 1 < args.len() {
                    export_dot = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--input" | "-i" => {
                if i + 1 < args.len() {
                    input_ids = args[i + 1]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
            }
        }
        i += 1;
    }

    Args {
        model_size,
        export_dot,
        input_ids,
    }
}

fn print_help() {
    println!("LLaMA Forward Pass Example");
    println!();
    println!("Usage: cargo run --example llama_forward [OPTIONS]");
    println!();
    println!("Options:");
    println!("  -m, --model <SIZE>     Model size: tiny, small, medium, large (default: tiny)");
    println!("  -e, --export-dot <FILE> Export computation graph to DOT file");
    println!("  -i, --input <IDS>      Input token IDs (comma-separated, default: 1,2,3,4,5)");
    println!("  -h, --help             Show this help message");
    println!();
    println!("Examples:");
    println!("  cargo run --example llama_forward");
    println!("  cargo run --example llama_forward -- -m small");
    println!("  cargo run --example llama_forward -- -m medium -e graph.dot");
    println!("  cargo run --example llama_forward -- -i 10,20,30,40,50");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║           LLaMA Forward Pass Example                    ║");
    println!("╠══════════════════════════════════════════════════════════╣");

    // Parse arguments
    let args = parse_args();

    println!("║ Model: {}", args.model_size.name());
    println!("║ Input: {:?}", args.input_ids);

    if let Some(ref dot_file) = args.export_dot {
        println!("║ Export DOT: {}", dot_file);
    }

    println!("╠══════════════════════════════════════════════════════════╣");

    // Get configuration
    let config = args.model_size.to_config();

    // Create model
    println!("║ Building model...");
    let model = create_test_model(&config);
    println!("║ ✓ Model created with {} layers", model.num_layers());

    // Build graph-structured representation
    println!("║ Building computation graph...");
    let builder = LlamaModelGraphBuilder::new(&model);
    let mut transformer = builder.build_graph_for_input(&args.input_ids);
    println!(
        "║ ✓ Graph built with {} nodes, {} edges",
        transformer.num_nodes(),
        transformer.num_edges()
    );

    // Export to DOT if requested
    if let Some(dot_file) = &args.export_dot {
        println!("║ Exporting graph to {}...", dot_file);
        let dot = builder.export_to_dot(&transformer);
        std::fs::write(dot_file, &dot)?;
        println!("║ ✓ Graph exported to {}", dot_file);
    }

    // Run forward pass
    println!("║ Running forward pass...");
    let output = transformer.forward(&args.input_ids);
    println!("║ ✓ Forward pass completed");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Output shape: {:?}", output.shape());
    println!("║ Output stats:");
    println!(
        "║   - Min: {:.6}",
        output.data().iter().cloned().fold(f64::INFINITY, f64::min)
    );
    println!(
        "║   - Max: {:.6}",
        output
            .data()
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );
    println!(
        "║   - Mean: {:.6}",
        output.data().iter().sum::<f64>() / output.data().len() as f64
    );
    println!("╚══════════════════════════════════════════════════════════╝");

    Ok(())
}
