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
pub mod layers;
pub mod loader;
pub mod model;
pub mod generation;
pub mod graph_transformer;
pub mod kv_cache;
#[cfg(feature = "tensor-sparse")]
pub mod sparse_attention;
pub mod batch;
#[cfg(feature = "tensor")]
pub mod quantization;
pub mod perf;
pub mod optimization;

// Re-export commonly used types
pub use autograd::{ComputeGraph, DifferentiableTensor, Op, Optimizer};
pub use layers::{MultiHeadAttention, RMSNorm, LayerNorm, RoPE, FeedForward};
pub use loader::{SafetensorsLoader, ModelConfig};
pub use model::{LlamaModel, LlamaConfig};
pub use generation::{GenerationConfig, TextGenerator};
pub use graph_transformer::{GraphExecutor, GraphTransformer, GraphNode, GraphEdge};
pub use kv_cache::KVCache;
#[cfg(feature = "tensor-sparse")]
pub use sparse_attention::SparseAttention;
pub use batch::BatchInference;
pub use quantization::{QuantizedTensor, QuantizationConfig};
pub use perf::{TransformerMemoryPool, softmax_inplace_simd, matmul_with_buffer};
