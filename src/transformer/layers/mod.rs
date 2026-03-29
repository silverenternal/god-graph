//! Transformer layer implementations
//!
//! This module provides the core building blocks for Transformer models:
//! - Multi-Head Attention
//! - Feed-Forward Networks (including SwiGLU variant)
//! - Normalization layers (LayerNorm, RMSNorm)
//! - Positional encodings (RoPE)

pub mod attention;
pub mod embedding;
pub mod ffn;
pub mod norm;

pub use attention::MultiHeadAttention;
pub use embedding::RoPE;
pub use ffn::FeedForward;
pub use norm::{LayerNorm, RMSNorm};
