//! Model loader for loading pre-trained weights
//!
//! This module provides:
//! - Safetensors format parsing
//! - Model configuration (LLaMA, Mistral)
//! - Weight mapping to model structures
//! - HuggingFace Hub integration (optional)

pub mod config;
pub mod safetensors;
pub mod weight_mapper;

pub use config::{LlamaConfig, MistralConfig, ModelConfig};
pub use safetensors::SafetensorsLoader;
pub use weight_mapper::LlamaWeightMapper;
