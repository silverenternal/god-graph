//! Model loader for loading pre-trained weights
//!
//! This module provides:
//! - Safetensors format parsing
//! - Model configuration (LLaMA, Mistral)
//! - Weight mapping to model structures
//! - HuggingFace Hub integration (optional)

pub mod safetensors;
pub mod config;
pub mod weight_mapper;

pub use safetensors::SafetensorsLoader;
pub use config::{ModelConfig, LlamaConfig, MistralConfig};
pub use weight_mapper::LlamaWeightMapper;
