//! Example: Loading LLaMA/Mistral Models from HuggingFace
//!
//! This example demonstrates how to load pre-trained model weights.
//!
//! ## Usage
//!
//! ```bash
//! # Show help
//! cargo run --example llm_model_loader --features "transformer,safetensors"
//!
//! # Load from local directory
//! cargo run --example llm_model_loader --features "transformer,safetensors" -- local /path/to/model
//! ```
//!
//! Requires the `transformer` and `safetensors` features.

#[cfg(all(feature = "transformer", feature = "safetensors"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use god_graph::transformer::loader::ModelConfig;
    use std::env;

    println!("=== God-Graph LLM Model Loader ===\n");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Usage:");
        println!(
            "  cargo run --example llm_model_loader --features transformer,safetensors -- [MODEL]"
        );
        println!();
        println!("Models:");
        println!("  llama-2-7b    - LLaMA-2-7B");
        println!("  mistral-7b    - Mistral-7B");
        println!("  local <path>  - Load from local directory");
        println!();
        println!("Note: Download models using huggingface-cli:");
        println!("  huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./model");
        return Ok(());
    }

    let model_name = &args[1];

    let model_path = if model_name == "local" {
        if args.len() < 3 {
            eprintln!("Error: Local model requires path argument");
            return Ok(());
        }
        std::path::PathBuf::from(&args[2])
    } else {
        let hf_repo = get_hf_repo(model_name)?;
        let cache_dir = get_hf_cache_dir();
        let model_dir = cache_dir.join(hf_repo.replace('/', "--"));

        if !model_dir.exists() {
            println!("\nModel not found. Please download using:");
            println!(
                "  huggingface-cli download {} --local-dir {}",
                hf_repo,
                model_dir.display()
            );
            return Ok(());
        }

        model_dir
    };

    println!("Loading model from: {}", model_path.display());

    // Load model configuration
    let config_path = model_path.join("config.json");
    if !config_path.exists() {
        return Err(format!("Config not found: {}", config_path.display()).into());
    }

    println!("Loading configuration...");
    let model_config = ModelConfig::from_file(&config_path)?;

    // Extract LlamaConfig if it's a LLaMA model
    if let Some(config) = model_config.as_llama() {
        println!("  Vocab size: {}", config.vocab_size);
        println!("  Hidden size: {}", config.hidden_size);
        println!("  Num layers: {}", config.num_hidden_layers);
        println!("  Num heads: {}", config.num_attention_heads);
    } else if let Some(config) = model_config.as_mistral() {
        println!("  Model type: Mistral");
        println!("  Vocab size: {}", config.vocab_size);
        println!("  Hidden size: {}", config.hidden_size);
        println!("  Num layers: {}", config.num_hidden_layers);
        println!("  Num heads: {}", config.num_attention_heads);
    }

    // Find Safetensors files
    let model_files = find_safetensors_files(&model_path)?;

    if model_files.is_empty() {
        return Err("No .safetensors files found".into());
    }

    println!("\nFound {} Safetensors file(s):", model_files.len());
    for file in &model_files {
        println!("  - {}", file.display());
    }

    // Load weights (demonstration - actual loading requires complete model setup)
    println!("\nTo load weights, use SafetensorsLoader:");
    println!("  let loader = SafetensorsLoader::from_file(&model_files[0])?;");
    println!("  let weights = loader.tensors();");

    // Estimate parameters
    if let Some(config) = model_config.as_llama() {
        let param_count = estimate_parameters_llama(config);
        println!("\n✅ Model info:");
        println!(
            "  Estimated parameters: ~{:.1}B",
            param_count as f64 / 1_000_000_000.0
        );
        println!("  Model type: LLaMA architecture");
    } else if model_config.as_mistral().is_some() {
        println!("\n✅ Model info:");
        println!("  Model type: Mistral architecture");
    }

    println!("\n=== Model Loading Complete ===");
    println!("\n💡 See docs/transformer_guide.md for complete usage examples.");

    Ok(())
}

fn get_hf_repo(model_name: &str) -> Result<String, Box<dyn std::error::Error>> {
    match model_name.to_lowercase().as_str() {
        "llama-2-7b" => Ok("meta-llama/Llama-2-7b-hf".to_string()),
        "llama-2-7b-chat" => Ok("meta-llama/Llama-2-7b-chat-hf".to_string()),
        "mistral-7b" => Ok("mistralai/Mistral-7B-v0.1".to_string()),
        _ => Err(format!("Unknown model: {}", model_name).into()),
    }
}

fn get_hf_cache_dir() -> std::path::PathBuf {
    if let Ok(cache_dir) = std::env::var("HF_HOME") {
        std::path::PathBuf::from(cache_dir).join("hub")
    } else if let Ok(cache_dir) = std::env::var("XDG_CACHE_HOME") {
        std::path::PathBuf::from(cache_dir)
            .join("huggingface")
            .join("hub")
    } else {
        std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
            .join(".cache")
            .join("huggingface")
            .join("hub")
    }
}

fn find_safetensors_files(
    dir: &std::path::Path,
) -> Result<Vec<std::path::PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();

    if dir.is_file() && dir.extension().is_some_and(|e| e == "safetensors") {
        files.push(dir.to_path_buf());
        return Ok(files);
    }

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|e| e == "safetensors") {
                files.push(path);
            }
        }
    }

    files.sort();
    Ok(files)
}

fn estimate_parameters_llama(config: &god_graph::transformer::loader::LlamaConfig) -> usize {
    let embed_params = config.vocab_size * config.hidden_size;
    let attn_params = config.hidden_size * config.hidden_size * 4;
    let ffn_params = config.hidden_size * config.intermediate_size * 3;
    let layer_params = attn_params + ffn_params + config.hidden_size * 2;
    let total_layer_params = layer_params * config.num_hidden_layers;

    embed_params + total_layer_params + config.hidden_size
}

#[cfg(not(all(feature = "transformer", feature = "safetensors")))]
fn main() {
    println!("This example requires the 'transformer' and 'safetensors' features.");
    println!("Run with: cargo run --example llm_model_loader --features transformer,safetensors");
}
