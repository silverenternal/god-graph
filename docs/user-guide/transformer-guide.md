# Transformer Module User Guide

## Overview

The `transformer` module in god-gragh provides a complete implementation of Transformer-based language model inference, optimized for CPU execution with support for:

- **LLaMA 2/3** architecture
- **Mistral** architecture  
- **Qwen** architecture
- **Phi-2/3** architecture

### Key Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Autograd Engine** | Automatic differentiation for training | ✅ |
| **Transformer Layers** | Multi-head attention, FFN (SwiGLU), RMSNorm, RoPE | ✅ |
| **Model Loading** | Safetensors format support | ✅ |
| **Graph Transformer** | Graph-structured attention | ✅ |
| **KV Cache** | Efficient autoregressive generation | ✅ |
| **Batch Inference** | Continuous batching (vLLM-style) | ✅ |
| **Quantization** | INT8/INT4 weight quantization | ✅ |
| **Sparse Attention** | Sliding window, block sparse patterns | ✅ |
| **SIMD Optimization** | AVX2/AVX-512 acceleration | ✅ |
| **Memory Pool** | Buffer reuse for reduced allocation | ✅ |

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
god-gragh = { version = "0.4.3-beta", features = ["transformer"] }
```

### Optional Features

```toml
[dependencies]
god-gragh = { version = "0.4.3-beta", features = [
    "transformer",      # Core transformer support
    "safetensors",      # Model weight loading
    "tokenizer",        # HuggingFace tokenizer integration
    "simd",             # SIMD acceleration (AVX2/AVX-512)
    "tensor-pool",      # Memory pool optimization
] }
```

## Quick Start

### Basic Text Generation

```rust
use god_graph::transformer::{
    LlamaModel, LlamaConfig, TextGenerator, GenerationConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a small model for demonstration
    let config = LlamaConfig {
        vocab_size: 1000,
        hidden_size: 256,
        intermediate_size: 512,
        num_hidden_layers: 4,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        max_position_embeddings: 512,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
    };

    // Initialize model (with random weights for demo)
    let model = LlamaModel::new(&config);

    // Configure generation
    let gen_config = GenerationConfig {
        max_new_tokens: 50,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        do_sample: true,
        ..Default::default()
    };

    // Create generator and generate
    let mut generator = TextGenerator::new(model, gen_config);
    let output = generator.generate("The future of AI", None)?;

    println!("{}", output);
    Ok(())
}
```

### Loading Pre-trained Weights

```rust
use god_graph::transformer::{
    LlamaModel, LlamaConfig, TextGenerator, GenerationConfig,
    loader::SafetensorsLoader,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = LlamaConfig::from_file("model/config.json")?;

    // Initialize model architecture
    let mut model = LlamaModel::new(&config);

    // Load weights from Safetensors
    let loader = SafetensorsLoader::from_file("model/model-00001-of-00002.safetensors")?;
    model = loader.load_into_llama(model, &config)?;

    // Generate text
    let generator = TextGenerator::new(model, GenerationConfig::default());
    let output = generator.generate("Hello, world!", None)?;

    println!("{}", output);
    Ok(())
}
```

## Architecture

### Model Components

```
┌─────────────────────────────────────────────────────────┐
│                    LlamaModel                           │
├─────────────────────────────────────────────────────────┤
│  embed_tokens: [vocab_size, hidden_size]                │
│  ┌──────────────────────────────────────────────────┐   │
│  │  LlamaDecoderLayer × num_hidden_layers           │   │
│  │  ┌────────────────────────────────────────────┐  │   │
│  │  │  input_layernorm (RMSNorm)                 │  │   │
│  │  │  self_attn (MultiHeadAttention)            │  │   │
│  │  │    - w_q, w_k, w_v, w_o                    │  │   │
│  │  │    - RoPE positional encoding              │  │   │
│  │  │  post_attention_layernorm (RMSNorm)        │  │   │
│  │  │  mlp (FeedForward SwiGLU)                  │  │   │
│  │  │    - gate_proj, up_proj, down_proj         │  │   │
│  │  └────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
│  norm (RMSNorm)                                         │
│  lm_head (optional)                                     │
└─────────────────────────────────────────────────────────┘
```

### Forward Pass Flow

```
Input IDs → Embeddings → [DecoderLayer × N] → Norm → LM Head → Logits
                              ↓
                    (Residual Connection)
                    1. Input Norm
                    2. Self-Attention + RoPE
                    3. Residual Add
                    4. Post-Attention Norm
                    5. FFN (SwiGLU)
                    6. Residual Add
```

## Module Structure

### Core Modules

| Module | Description | Key Types |
|--------|-------------|-----------|
| `autograd` | Automatic differentiation engine | `ComputeGraph`, `DifferentiableTensor`, `Op` |
| `layers` | Transformer layer implementations | `MultiHeadAttention`, `RMSNorm`, `RoPE`, `FeedForward` |
| `model` | Complete model implementations | `LlamaModel`, `LlamaConfig`, `LlamaDecoderLayer` |
| `loader` | Model weight loading | `SafetensorsLoader`, `ModelConfig` |
| `generation` | Text generation logic | `TextGenerator`, `GenerationConfig` |
| `kv_cache` | KV cache optimization | `KVCache` |
| `batch` | Batch inference | `BatchData`, `BatchInference`, `RequestScheduler` |
| `quantization` | Weight quantization | `QuantizedTensor`, `QuantizationConfig` |
| `sparse_attention` | Sparse attention patterns | `SparseMask`, `SparseAttention` |
| `perf` | Performance optimizations | `TransformerMemoryPool`, `softmax_inplace_simd` |

## Advanced Usage

### KV Cache for Efficient Generation

KV Cache stores key and value projections to avoid recomputing them for previous tokens:

```rust
use god_graph::transformer::kv_cache::KVCache;

// Create KV cache
let num_layers = 32;
let max_seq_len = 2048;
let hidden_dim = 4096;
let num_kv_heads = 8;

let mut kv_cache = KVCache::new(num_layers, max_seq_len, hidden_dim, num_kv_heads);

// During generation, update cache with new tokens
let k = /* key projection [batch, 1, hidden_dim] */;
let v = /* value projection [batch, 1, hidden_dim] */;
kv_cache.update(layer_idx, &k, &v);

// Retrieve cached keys/values for attention
let (k_cached, v_cached) = kv_cache.get(layer_idx);
```

**Memory Usage**: `num_layers × max_seq_len × hidden_dim × 2 (k+v) × 8 (f64) bytes`

For LLaMA-7B with 2048 seq len: ~128 MB

### Batch Inference with Continuous Caching

```rust
use god_graph::transformer::batch::{BatchData, BatchInference, RequestScheduler};
use god_graph::transformer::model::LlamaModel;

// Create model
let model = LlamaModel::new(&config);

// Create batch inference engine
let max_batch_size = 8;
let max_seq_len = 512;
let mut batch_infer = BatchInference::new(&model, max_batch_size, max_seq_len);

// Create batch
let input_ids = vec![
    vec![1, 2, 3, 4, 5],
    vec![6, 7, 8],
    vec![9, 10, 11, 12],
];
let batch = BatchData::new(input_ids);

// Forward pass
let logits = batch_infer.forward(&batch);

// For continuous batching, use RequestScheduler
let mut scheduler = RequestScheduler::new(max_batch_size);

// Add requests
scheduler.add_request(vec![1, 2, 3], gen_config);
scheduler.add_request(vec![4, 5], gen_config);

// Generation loop
while scheduler.num_active() > 0 {
    let requests = scheduler.schedule();
    let tokens = batch_infer.step(&requests);
    // ... update requests with generated tokens
}
```

### Quantization for Memory Efficiency

```rust
use god_graph::transformer::quantization::{QuantizedTensor, QuantizationConfig};
use god_graph::tensor::DenseTensor;

// Create tensor
let weight = DenseTensor::randn(vec![4096, 4096], 0.0, 0.02);

// INT8 quantization (4x compression)
let q_config = QuantizationConfig::int8();
let quantized = QuantizedTensor::from_tensor(&weight, q_config);

println!("Original size: {} bytes", weight.data().len() * 8);
println!("Quantized size: {} bytes", quantized.memory_bytes());
println!("Compression ratio: {:.1}x", quantized.compression_ratio());

// Use quantized matrix multiplication
let result = QuantizedMatMul::gemm_int8(&quantized, &input);
```

**Compression Ratios**:
- INT8: 4x (64-bit → 8-bit per weight)
- INT4: 8x (64-bit → 4-bit per weight)

### Sparse Attention Patterns

```rust
use god_graph::transformer::sparse_attention::{SparseMask, SparseAttention};

// Sliding window attention (local attention)
let seq_len = 1024;
let window_size = 128;
let sliding_mask = SparseMask::sliding_window(seq_len, window_size, true);

// Block sparse attention
let block_sparse = SparseMask::block_sparse(seq_len, 64, 4);

// Apply sparse attention
let sparse_attn = SparseAttention::new(sliding_mask);
let output = sparse_attn.forward(&query, &key, &value);
```

**Sparsity Examples**:

| Pattern | Sequence Length | Non-zero % | Memory Savings |
|---------|-----------------|------------|----------------|
| Dense | 1024 | 100% | - |
| Sliding Window (128) | 1024 | 12.5% | 8x |
| Block Sparse (4 blocks) | 1024 | 6.25% | 16x |

### SIMD Optimization

Enable SIMD for 2-4x speedup on supported CPUs:

```toml
[dependencies]
god-gragh = { version = "0.4.3-beta", features = ["transformer", "simd"] }
```

```rust
use god_graph::transformer::perf::{softmax_inplace_simd, matmul_with_buffer};
use god_graph::tensor::DenseTensor;

// SIMD-optimized softmax
let mut data = vec![1.0f64; 512 * 512];
let shape = vec![512, 512];
softmax_inplace_simd(&mut data, &shape, 1);

// SIMD-optimized matmul with buffer reuse
let a = DenseTensor::randn(vec![256, 512], 0.0, 0.02);
let b = DenseTensor::randn(vec![512, 256], 0.0, 0.02);
let mut buffer = vec![0.0f64; 256 * 256];
let result = matmul_with_buffer(&a, &b, &mut buffer);
```

### Memory Pool for Reduced Allocation

```rust
use god_graph::transformer::perf::TransformerMemoryPool;

// Create pool
let mut pool = TransformerMemoryPool::new(
    4,   // batch_size
    512, // seq_len
    4096, // hidden_dim
    32,  // num_heads
);

// Get buffers (allocated once, reused)
let qkv_buf = pool.get_qkv_buffer();
let attn_score_buf = pool.get_attn_score_buffer();
let attn_weight_buf = pool.get_attn_weight_buffer();
let output_buf = pool.get_output_buffer();

// Buffers automatically resized if needed
pool.resize(8, 1024, 4096, 32);
```

## Generation Strategies

### Greedy Decoding

```rust
let config = GenerationConfig {
    max_new_tokens: 100,
    temperature: 0.0,  // Greedy
    do_sample: false,
    ..Default::default()
};
```

### Random Sampling

```rust
let config = GenerationConfig {
    max_new_tokens: 100,
    temperature: 1.0,  // Random
    do_sample: true,
    ..Default::default()
};
```

### Top-k Sampling

```rust
let config = GenerationConfig {
    max_new_tokens: 100,
    temperature: 0.7,
    top_k: 40,       // Sample from top 40 tokens
    do_sample: true,
    ..Default::default()
};
```

### Nucleus (Top-p) Sampling

```rust
let config = GenerationConfig {
    max_new_tokens: 100,
    temperature: 0.7,
    top_p: 0.9,      // Sample from top 90% probability mass
    do_sample: true,
    ..Default::default()
};
```

### Repetition Penalty

```rust
let config = GenerationConfig {
    max_new_tokens: 100,
    temperature: 0.7,
    repetition_penalty: 1.1,  // Discourage repetition
    ..Default::default()
};
```

## Performance Optimization

### Benchmarking

Run benchmarks with:

```bash
# All benchmarks
cargo bench --features "transformer,simd" --bench transformer_inference

# Specific benchmark
cargo bench --features "transformer,simd" --bench transformer_inference -- bench_llama_forward

# Compare SIMD vs naive
cargo bench --features transformer --bench transformer_inference  # naive
cargo bench --features "transformer,simd" --bench transformer_inference  # SIMD
```

### Optimization Tips

| Optimization | Speedup | Memory | Use Case |
|--------------|---------|--------|----------|
| **SIMD** | 2-4x | - | All inference |
| **INT8 Quantization** | 2-3x | 4x reduction | Production deployment |
| **KV Cache** | 5-10x (autoregressive) | +128MB/7B | Text generation |
| **Batch Inference** | 4-8x (batch=8) | Linear | High throughput |
| **Memory Pool** | 1.5-2x | - | Iterative algorithms |
| **Sparse Attention** | 2-4x | 4-16x reduction | Long sequences |

### Memory Usage Estimates

| Model | Precision | VRAM Required |
|-------|-----------|---------------|
| LLaMA-7B | FP64 | ~56 GB |
| LLaMA-7B | FP32 | ~28 GB |
| LLaMA-7B | FP16 | ~14 GB |
| LLaMA-7B | INT8 | ~7 GB |
| LLaMA-7B | INT4 | ~3.5 GB |
| LLaMA-13B | FP16 | ~26 GB |
| LLaMA-70B | INT4 | ~35 GB |

## Examples

See the `examples/` directory for complete examples:

| Example | Description | Features Required |
|---------|-------------|-------------------|
| `llm_model_loader` | Load models from HuggingFace | `transformer,safetensors,tokenizer` |
| `llm_text_gen` | End-to-end text generation | `transformer` |
| `llm_batch_simd` | SIMD-optimized batch inference | `transformer,simd` |
| `transformer_basic` | Basic transformer usage | `transformer` |
| `transformer_advanced` | Advanced features demo | `transformer,safetensors,tensor-pool` |

Run examples:

```bash
# Text generation demo
cargo run --example llm_text_gen --features transformer

# Load real model weights
cargo run --example llm_model_loader --features transformer,safetensors,tokenizer

# SIMD batch inference
cargo run --example llm_batch_simd --features transformer,simd
```

## Troubleshooting

### Common Issues

**Out of Memory**: Reduce batch size or use quantization (INT8/INT4)

```rust
// Use INT4 for 8x memory reduction
let config = QuantizationConfig::int4();
```

**Slow Inference**: Enable SIMD and batch processing

```toml
[dependencies]
god-gragh = { version = "0.4.3-beta", features = ["transformer", "simd", "tensor-pool"] }
```

**Model Loading Fails**: Ensure Safetensors format and correct file paths

```rust
// Check file exists
assert!(std::path::Path::new("model.safetensors").exists());

// Verify config
let config = LlamaConfig::from_file("config.json")?;
```

## API Reference

### Key Types

```rust
// Model configuration
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

// Complete model
pub struct LlamaModel {
    pub config: LlamaConfig,
    pub embed_tokens: DenseTensor,
    pub layers: Vec<LlamaDecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<DenseTensor>,
}

// Generation configuration
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub do_sample: bool,
    pub repetition_penalty: f64,
    pub eos_token_id: Option<usize>,
}
```

### Main Functions

```rust
// Create model
LlamaModel::new(&config) -> LlamaModel

// Load weights
SafetensorsLoader::from_file(path) -> Result<SafetensorsLoader>
loader.load_into_llama(model, &config) -> Result<LlamaModel>

// Generate text
TextGenerator::new(model, config) -> TextGenerator
generator.generate(prompt, None) -> Result<String>

// Batch inference
BatchInference::new(&model, batch_size, max_seq_len) -> BatchInference
batch_infer.forward(&batch) -> DenseTensor

// Quantization
QuantizedTensor::from_tensor(&tensor, config) -> QuantizedTensor
QuantizedMatMul::gemm_int8(&a, &b) -> DenseTensor
```

## License

This project is dual-licensed: MIT or Apache-2.0.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Additional Resources

- [Performance Benchmarks](../benches/transformer_inference.rs)
- [Example Code](../examples/)
- [API Documentation](https://docs.rs/god-gragh)
