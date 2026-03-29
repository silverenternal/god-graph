//! Transformer module for LLM support
//!
//! This module provides:
//! - Autograd engine for automatic differentiation
//! - Transformer layer implementations (Attention, FFN, Norm)
//! - Model loading from Safetensors format
//! - Graph-structured Transformer inference
//! - KV Cache and batch inference optimizations
//! - Sparse attention optimizations
//! - Quantization support
//! - Performance optimizations (SIMD, memory pool, optimized kernels)

#![cfg(feature = "tensor")]

pub mod autograd;
pub mod batch;
pub mod generation;
pub mod graph_transformer;
pub mod kv_cache;
pub mod layers;
pub mod loader;
pub mod model;
pub mod optimization;
pub mod perf;
#[cfg(feature = "tensor")]
pub mod quantization;
#[cfg(feature = "tensor-sparse")]
pub mod sparse_attention;

// Re-export commonly used types
pub use autograd::{ComputeGraph, DifferentiableTensor, Op, Optimizer};
pub use batch::BatchInference;
pub use generation::{GenerationConfig, TextGenerator};
pub use graph_transformer::{GraphEdge, GraphExecutor, GraphNode, GraphTransformer};
pub use kv_cache::KVCache;
pub use layers::{FeedForward, LayerNorm, MultiHeadAttention, RMSNorm, RoPE};
pub use loader::{ModelConfig, SafetensorsLoader};
pub use model::{LlamaConfig, LlamaModel};
pub use perf::{matmul_with_buffer, softmax_inplace_simd, TransformerMemoryPool};
pub use quantization::{QuantizationConfig, QuantizedTensor};
#[cfg(feature = "tensor-sparse")]
pub use sparse_attention::SparseAttention;
