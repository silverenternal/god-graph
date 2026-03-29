# Transformer Tutorial: From Basics to Advanced

This tutorial walks you through using the Transformer module in god-gragh, from basic concepts to advanced optimization techniques.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [Basic Text Generation](#basic-text-generation)
4. [Loading Pre-trained Models](#loading-pre-trained-models)
5. [Optimization Techniques](#optimization-techniques)
6. [Advanced Features](#advanced-features)
7. [Production Deployment](#production-deployment)

---

## Getting Started

### Prerequisites

- Rust 1.85 or later
- Basic understanding of neural networks
- (Optional) Familiarity with Transformer architecture

### Installation

Create a new Rust project:

```bash
cargo new my-transformer-app
cd my-transformer-app
```

Add god-gragh to your dependencies:

```toml
[dependencies]
god-gragh = { version = "0.4.3-beta", features = ["transformer"] }
```

For full functionality:

```toml
[dependencies]
god-gragh = { version = "0.4.3-beta", features = [
    "transformer",
    "safetensors",
    "tokenizer",
    "simd",
    "tensor-pool",
] }
```

### Your First Transformer

Create `src/main.rs`:

```rust
use god_gragh::transformer::{
    LlamaModel, LlamaConfig, TextGenerator, GenerationConfig,
};
use god_gragh::tensor::DenseTensor;

fn main() {
    println!("Hello, Transformer!");

    // Create a small model for testing
    let config = LlamaConfig {
        vocab_size: 1000,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
    };

    // Initialize model
    let mut model = LlamaModel::new(&config);
    
    // Initialize with small random weights for testing
    model.embed_tokens = DenseTensor::randn(
        vec![config.vocab_size, config.hidden_size],
        0.0,
        0.02,
    );

    // Configure generation
    let gen_config = GenerationConfig {
        max_new_tokens: 20,
        temperature: 0.7,
        top_p: 0.9,
        do_sample: true,
        ..Default::default()
    };

    // Create generator
    let mut generator = TextGenerator::new(model, gen_config);

    // Generate text
    match generator.generate("Hello, world!", None) {
        Ok(output) => println!("Generated: {}", output),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

Run it:

```bash
cargo run --features transformer
```

---

## Understanding the Architecture

### Transformer Building Blocks

A Transformer model consists of several key components:

```
Input → Embeddings → [Decoder Layer × N] → Output
                        ↓
                    ┌─────────────┐
                    │ 1. LayerNorm│
                    │ 2. Attention│
                    │ 3. Residual │
                    │ 4. LayerNorm│
                    │ 5. FFN      │
                    │ 6. Residual │
                    └─────────────┘
```

### Component Breakdown

Let's examine each component:

```rust
use god_gragh::transformer::layers::{
    MultiHeadAttention, RMSNorm, RoPE, FeedForward,
};
use god_gragh::tensor::DenseTensor;

// 1. Multi-Head Attention
let hidden_dim = 512;
let num_heads = 8;

let w_q = DenseTensor::randn(vec![hidden_dim, hidden_dim], 0.0, 0.02);
let w_k = DenseTensor::randn(vec![hidden_dim, hidden_dim], 0.0, 0.02);
let w_v = DenseTensor::randn(vec![hidden_dim, hidden_dim], 0.0, 0.02);
let w_o = DenseTensor::randn(vec![hidden_dim, hidden_dim], 0.0, 0.02);

let attention = MultiHeadAttention::standard(w_q, w_k, w_v, w_o, num_heads);

// 2. RMSNorm (Layer Normalization variant)
let norm_weight = DenseTensor::ones(vec![hidden_dim]);
let norm = RMSNorm::new(norm_weight, 1e-6);

// 3. RoPE (Rotary Positional Embedding)
let head_dim = hidden_dim / num_heads;
let rope = RoPE::new(head_dim, 512, 10000.0);

// 4. Feed-Forward Network (SwiGLU variant)
let intermediate_dim = 2048;
let gate_proj = DenseTensor::randn(vec![hidden_dim, intermediate_dim], 0.0, 0.02);
let up_proj = DenseTensor::randn(vec![hidden_dim, intermediate_dim], 0.0, 0.02);
let down_proj = DenseTensor::randn(vec![intermediate_dim, hidden_dim], 0.0, 0.02);

let ffn = FeedForward::swiglu(gate_proj, up_proj, down_proj);
```

### Understanding Data Flow

```rust
use god_gragh::tensor::{DenseTensor, TensorOps};

// Input: [batch_size=2, seq_len=10, hidden_dim=512]
let x = DenseTensor::randn(vec![2, 10, 512], 0.0, 0.02);

// 1. Normalize
let x_norm = norm.forward(&x);  // [2, 10, 512]

// 2. Self-Attention
let attn_out = attention.forward(&x_norm);  // [2, 10, 512]

// 3. Residual connection
let h = x.add(&attn_out);  // [2, 10, 512]

// 4. Normalize again
let h_norm = norm.forward(&h);  // [2, 10, 512]

// 5. FFN
let ffn_out = ffn.forward(&h_norm);  // [2, 10, 512]

// 6. Final residual
let output = h.add(&ffn_out);  // [2, 10, 512]
```

---

## Basic Text Generation

### Tokenization Basics

Before text generation, you need to convert text to token IDs:

```rust
// Simple character-level tokenizer for demonstration
fn tokenize(text: &str) -> Vec<usize> {
    text.chars()
        .map(|c| c as usize)
        .collect()
}

fn detokenize(tokens: &[usize]) -> String {
    tokens
        .iter()
        .filter_map(|&t| char::from_u32(t as u32))
        .collect()
}

// Usage
let text = "Hello, world!";
let tokens = tokenize(text);
println!("Tokens: {:?}", tokens);

let recovered = detokenize(&tokens);
println!("Recovered: {}", recovered);
```

**Note**: For production use, use a proper tokenizer like the HuggingFace `tokenizers` crate.

### Simple Generation Loop

```rust
use god_gragh::transformer::{LlamaModel, LlamaConfig, GenerationConfig};
use god_gragh::tensor::DenseTensor;

fn generate_simple(
    model: &LlamaModel,
    prompt: &str,
    max_tokens: usize,
) -> String {
    let mut tokens = vec![1, 2, 3]; // Simplified: assume tokenized prompt
    
    for _ in 0..max_tokens {
        // Forward pass
        let logits = model.forward(&[tokens.clone()], 1, tokens.len());
        
        // Get last token logits
        let last_logits = &logits.data()[(tokens.len() - 1) * model.config.vocab_size..];
        
        // Argmax (greedy)
        let next_token = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        tokens.push(next_token);
    }
    
    detokenize(&tokens)
}
```

### Using GenerationConfig

```rust
use god_gragh::transformer::{TextGenerator, GenerationConfig};

// Greedy decoding (deterministic)
let greedy_config = GenerationConfig {
    max_new_tokens: 50,
    temperature: 0.0,
    do_sample: false,
    ..Default::default()
};

// Random sampling (high diversity)
let random_config = GenerationConfig {
    max_new_tokens: 50,
    temperature: 1.0,
    do_sample: true,
    ..Default::default()
};

// Balanced (top-p sampling)
let balanced_config = GenerationConfig {
    max_new_tokens: 50,
    temperature: 0.7,
    top_p: 0.9,
    do_sample: true,
    ..Default::default()
};

// Controlled (top-k sampling)
let controlled_config = GenerationConfig {
    max_new_tokens: 50,
    temperature: 0.7,
    top_k: 40,
    do_sample: true,
    ..Default::default()
};
```

---

## Loading Pre-trained Models

### Understanding Model Formats

Pre-trained models are typically distributed in these formats:

| Format | Description | Size (LLaMA-7B) |
|--------|-------------|-----------------|
| **Safetensors** | Safe tensor format (recommended) | ~14 GB (FP16) |
| **PyTorch** | .bin or .pt files | ~14 GB (FP16) |
| **GGUF** | Quantized format (llama.cpp) | ~4 GB (Q4_K_M) |

### Downloading from HuggingFace

```bash
# Install huggingface-cli
pip install huggingface_hub

# Download LLaMA-2-7B (requires access request)
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./models/llama-2-7b

# Download Mistral-7B (open access)
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ./models/mistral-7b
```

### Loading Safetensors Weights

```rust
use god_gragh::transformer::{
    LlamaModel, LlamaConfig, TextGenerator,
    loader::SafetensorsLoader,
};
use std::path::Path;

fn load_model(model_path: &str) -> Result<LlamaModel, Box<dyn std::error::Error>> {
    // 1. Load configuration
    let config_path = Path::new(model_path).join("config.json");
    let config = LlamaConfig::from_file(&config_path)?;
    println!("Loaded config: {:?}", config);

    // 2. Initialize model architecture
    let mut model = LlamaModel::new(&config);

    // 3. Find all Safetensors files
    let model_dir = Path::new(model_path);
    let mut safetensors_files = Vec::new();
    
    for entry in std::fs::read_dir(model_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "safetensors") {
            safetensors_files.push(path);
        }
    }

    // 4. Load weights from each file
    for file_path in &safetensors_files {
        println!("Loading: {:?}", file_path);
        let loader = SafetensorsLoader::from_file(file_path)?;
        model = loader.load_into_llama(model, &config)?;
    }

    Ok(model)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = load_model("./models/mistral-7b")?;
    
    let generator = TextGenerator::new(model, GenerationConfig::default());
    let output = generator.generate("The future of AI", None)?;
    
    println!("{}", output);
    Ok(())
}
```

### Weight Mapping

Understanding weight names helps with debugging:

```rust
// Common weight mappings for LLaMA/Mistral

// Embeddings
"model.embed_tokens.weight"  // [vocab_size, hidden_size]

// Each decoder layer (model.layers.{i})
"model.layers.{i}.self_attn.q_proj.weight"  // [hidden_size, hidden_size]
"model.layers.{i}.self_attn.k_proj.weight"  // [hidden_size, hidden_size]
"model.layers.{i}.self_attn.v_proj.weight"  // [hidden_size, hidden_size]
"model.layers.{i}.self_attn.o_proj.weight"  // [hidden_size, hidden_size]
"model.layers.{i}.mlp.gate_proj.weight"     // [intermediate_size, hidden_size]
"model.layers.{i}.mlp.up_proj.weight"       // [intermediate_size, hidden_size]
"model.layers.{i}.mlp.down_proj.weight"     // [hidden_size, intermediate_size]
"model.layers.{i}.input_layernorm.weight"   // [hidden_size]
"model.layers.{i}.post_attention_layernorm.weight" // [hidden_size]

// Final norm and head
"model.norm.weight"  // [hidden_size]
"lm_head.weight"     // [vocab_size, hidden_size] (optional, often tied with embeddings)
```

---

## Optimization Techniques

### 1. KV Cache

Avoid recomputing keys and values for previous tokens:

```rust
use god_gragh::transformer::kv_cache::KVCache;

// Create cache
let num_layers = 32;
let max_seq_len = 2048;
let hidden_dim = 4096;
let num_kv_heads = 8;

let mut kv_cache = KVCache::new(num_layers, max_seq_len, hidden_dim, num_kv_heads);

// Autoregressive generation with cache
fn generate_with_cache(
    model: &LlamaModel,
    kv_cache: &mut KVCache,
    prompt: &[usize],
    max_tokens: usize,
) -> Vec<usize> {
    let mut tokens = prompt.to_vec();
    
    for i in 0..max_tokens {
        // Only process new token(s)
        let input = if i == 0 {
            prompt.to_vec()  // First pass: process full prompt
        } else {
            vec![*tokens.last().unwrap()]  // Subsequent: single token
        };
        
        // Forward pass with cache
        let logits = model.forward_with_cache(&[input], kv_cache);
        
        // Sample next token
        let next_token = sample_from_logits(&logits);
        tokens.push(next_token);
    }
    
    tokens
}
```

**Performance Impact**: 5-10x speedup for long sequences

### 2. Batch Processing

Process multiple sequences simultaneously:

```rust
use god_gragh::transformer::batch::{BatchData, BatchInference};

// Create batch inference engine
let model = LlamaModel::new(&config);
let mut batch_infer = BatchInference::new(&model, 8, 512);

// Batch multiple sequences
let batch_data = BatchData::new(vec![
    vec![1, 2, 3, 4, 5],
    vec![6, 7, 8],
    vec![9, 10, 11, 12],
    vec![13, 14],
]);

// Single forward pass for all sequences
let logits = batch_infer.forward(&batch_data);

// Each sequence gets its own output
println!("Batch output shape: {:?}", logits.shape());
```

**Performance Impact**: 4-8x throughput improvement (batch_size=8)

### 3. Quantization

Reduce memory usage with lower precision:

```rust
use god_gragh::transformer::quantization::{QuantizedTensor, QuantizationConfig};

// INT8 quantization (4x compression)
let weight = DenseTensor::randn(vec![4096, 4096], 0.0, 0.02);
let q_config = QuantizationConfig::int8();
let quantized = QuantizedTensor::from_tensor(&weight, q_config);

println!("Original: {} MB", weight.data().len() * 8 / 1024 / 1024);
println!("Quantized: {} MB", quantized.memory_bytes() / 1024 / 1024);
println!("Compression: {:.1}x", quantized.compression_ratio());

// Use quantized weights in model
// Note: Requires modifying model to use QuantizedTensor instead of DenseTensor
```

**Performance Impact**:
- Memory: 4x reduction (INT8) or 8x (INT4)
- Speed: 2-3x faster (reduced memory bandwidth)

### 4. SIMD Acceleration

Enable CPU vector instructions:

```toml
# Cargo.toml
[dependencies]
god-gragh = { version = "0.4.3-beta", features = ["simd"] }

# Enable target-cpu=native for best performance
[profile.release]
opt-level = 3
target-cpu = "native"
```

```rust
use god_gragh::transformer::perf::{softmax_inplace_simd, matmul_with_buffer};

// SIMD-optimized operations
let mut data = vec![1.0f64; 1024 * 1024];
let shape = vec![1024, 1024];

// Uses AVX2/AVX-512 if available
softmax_inplace_simd(&mut data, &shape, 1);
```

**Performance Impact**: 2-4x speedup on supported CPUs

### 5. Memory Pool

Reuse buffers to reduce allocation overhead:

```rust
use god_gragh::transformer::perf::TransformerMemoryPool;

// Create pool (allocate once)
let mut pool = TransformerMemoryPool::new(4, 512, 4096, 32);

// Reuse across multiple forward passes
for _ in 0..100 {
    let qkv_buf = pool.get_qkv_buffer();
    let attn_score_buf = pool.get_attn_score_buffer();
    // ... use buffers
}

// Much faster than allocating new Vecs each iteration
```

**Performance Impact**: 1.5-2x speedup for iterative algorithms

---

## Advanced Features

### Sparse Attention

For long sequences, use sparse attention patterns:

```rust
use god_gragh::transformer::sparse_attention::{SparseMask, SparseAttention};

// Sliding window attention (local context)
let seq_len = 4096;
let window_size = 256;
let sliding_mask = SparseMask::sliding_window(seq_len, window_size, true);

// Block sparse (learned sparse patterns)
let block_sparse = SparseMask::block_sparse(seq_len, 128, 4);

// Create sparse attention module
let sparse_attn = SparseAttention::new(sliding_mask);

// Use in place of regular attention
let output = sparse_attn.forward(&query, &key, &value);
```

**Use Cases**:
- Long document processing (10K+ tokens)
- Genomic sequence analysis
- Time series forecasting

### Graph Transformer

Apply Transformer to graph-structured data:

```rust
use god_gragh::transformer::graph_transformer::{
    GraphTransformer, GraphNode, GraphEdge, GraphExecutor,
};
use god_gragh::graph::Graph;

// Create graph
let mut graph = Graph::<Vec<f64>, f64>::directed();

let n0 = graph.add_node(vec![1.0, 0.0, 0.0]).unwrap();
let n1 = graph.add_node(vec![0.0, 1.0, 0.0]).unwrap();
let n2 = graph.add_node(vec![0.0, 0.0, 1.0]).unwrap();

let _ = graph.add_edge(n0, n1, 1.0);
let _ = graph.add_edge(n1, n2, 1.0);

// Convert to GraphTransformer format
let nodes: Vec<GraphNode> = graph.nodes()
    .map(|n| GraphNode::new(n.index(), n.data().clone()))
    .collect();

let edges: Vec<GraphEdge> = graph.edges()
    .map(|e| GraphEdge::new(e.source(), e.target(), *e.weight()))
    .collect();

// Create and run GraphTransformer
let transformer = GraphTransformer::new(
    nodes,
    edges,
    3,  // num_layers
    64, // hidden_dim
);

let executor = GraphExecutor::new(transformer);
let output = executor.forward();
```

**Use Cases**:
- Molecular property prediction
- Social network analysis
- Knowledge graph reasoning

### Continuous Batching (vLLM-style)

Efficiently handle multiple generation requests:

```rust
use god_gragh::transformer::batch::{RequestScheduler, InferenceRequest, GenerationConfig};

// Create scheduler
let mut scheduler = RequestScheduler::new(8); // max 8 concurrent requests

// Add requests dynamically
let id1 = scheduler.add_request(vec![1, 2, 3], GenerationConfig::default());
let id2 = scheduler.add_request(vec![4, 5, 6, 7], GenerationConfig::default());

// Generation loop
while scheduler.num_active() > 0 || scheduler.num_pending() > 0 {
    // Schedule requests (fills empty slots with pending requests)
    let active_requests = scheduler.schedule();
    
    if active_requests.is_empty() {
        break;
    }
    
    // Process batch
    // ... (forward pass, sampling)
    
    // Update requests
    for req in active_requests {
        if req.completed {
            // Return result to user
            println!("Request {} completed: {:?}", req.id, req.generated);
        }
    }
}
```

**Benefits**:
- High throughput for server deployment
- Dynamic request scheduling
- Efficient GPU utilization

---

## Production Deployment

### Memory Management

For large models, manage memory carefully:

```rust
fn estimate_memory(config: &LlamaConfig, precision_bits: usize) -> usize {
    let param_count = config.hidden_size * config.hidden_size * config.num_hidden_layers * 12; // Rough estimate
    let bytes_per_param = precision_bits / 8;
    param_count * bytes_per_param
}

// LLaMA-7B memory estimates:
// - FP64: ~336 GB
// - FP32: ~168 GB
// - FP16: ~84 GB
// - INT8: ~21 GB
// - INT4: ~10.5 GB
```

### Model Serving

Example HTTP server with Actix:

```rust
use actix_web::{web, App, HttpServer, post, HttpResponse};
use god_gragh::transformer::{LlamaModel, TextGenerator, GenerationConfig};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
}

#[derive(Serialize)]
struct GenerateResponse {
    text: String,
}

async fn generate(
    model: web::Data<LlamaModel>,
    req: web::Json<GenerateRequest>,
) -> HttpResponse {
    let config = GenerationConfig {
        max_new_tokens: req.max_tokens.unwrap_or(100),
        temperature: req.temperature.unwrap_or(0.7),
        ..Default::default()
    };
    
    let mut generator = TextGenerator::new(model.get_ref().clone(), config);
    
    match generator.generate(&req.prompt, None) {
        Ok(text) => HttpResponse::Ok().json(GenerateResponse { text }),
        Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Load model once
    let model = load_model("./models/mistral-7b").unwrap();
    let model_data = web::Data::new(model);
    
    HttpServer::new(move || {
        App::new()
            .app_data(model_data.clone())
            .route("/generate", post(generate))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

### Monitoring and Logging

```rust
use std::time::Instant;
use tracing::{info, warn, error};

fn generate_with_logging(
    generator: &mut TextGenerator,
    prompt: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let start = Instant::now();
    
    info!("Generating for prompt: {}", prompt);
    
    match generator.generate(prompt, None) {
        Ok(output) => {
            let elapsed = start.elapsed();
            let tokens = output.split_whitespace().count();
            let tokens_per_sec = tokens as f64 / elapsed.as_secs_f64();
            
            info!(
                "Generated {} tokens in {:.2}s ({:.1} tokens/sec)",
                tokens, elapsed.as_secs_f64(), tokens_per_sec
            );
            
            Ok(output)
        }
        Err(e) => {
            error!("Generation failed: {}", e);
            Err(e.into())
        }
    }
}
```

### Error Handling

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GenerationError {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),
    
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
    
    #[error("Generation failed: {0}")]
    GenerationError(String),
    
    #[error("Out of memory: {0}")]
    OutOfMemory(String),
}

pub type Result<T> = std::result::Result<T, GenerationError>;

fn safe_generate(prompt: &str) -> Result<String> {
    // Validate input
    if prompt.is_empty() {
        return Err(GenerationError::ConfigError("Empty prompt".to_string()));
    }
    
    if prompt.len() > 10000 {
        return Err(GenerationError::ConfigError("Prompt too long".to_string()));
    }
    
    // Generate with timeout
    let result = std::thread::spawn(move || {
        // ... generation code
    })
    .join()
    .map_err(|_| GenerationError::GenerationError("Thread panicked".to_string()))?;
    
    result
}
```

---

## Next Steps

### Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Mistral 7B](https://arxiv.org/abs/2310.06825) - Efficient small models
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://arxiv.org/abs/2309.06180)

### Example Projects

- `examples/llm_model_loader.rs` - Loading pre-trained models
- `examples/llm_text_gen.rs` - Text generation demo
- `examples/llm_batch_simd.rs` - SIMD optimization
- `benches/transformer_inference.rs` - Performance benchmarks

### Community Resources

- [god-gragh GitHub](https://github.com/silverenternal/god-graph)
- [API Documentation](https://docs.rs/god-gragh)
- [HuggingFace Model Hub](https://huggingface.co/models)

---

## Conclusion

You've now learned:
- ✅ Basic Transformer architecture and components
- ✅ How to create and run a simple model
- ✅ Loading pre-trained weights
- ✅ Optimization techniques (KV cache, batching, quantization, SIMD)
- ✅ Advanced features (sparse attention, graph transformers)
- ✅ Production deployment considerations

Start building your own Transformer applications with god-gragh!
