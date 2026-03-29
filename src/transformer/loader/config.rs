//! Model configuration for LLaMA and Mistral

use serde::{Deserialize, Serialize};

/// Base model configuration trait
pub trait ModelConfigTrait {
    /// Get vocabulary size
    fn vocab_size(&self) -> usize;
    
    /// Get hidden dimension
    fn hidden_size(&self) -> usize;
    
    /// Get intermediate dimension (FFN)
    fn intermediate_size(&self) -> usize;
    
    /// Get number of hidden layers
    fn num_hidden_layers(&self) -> usize;
    
    /// Get number of attention heads
    fn num_attention_heads(&self) -> usize;
    
    /// Get number of KV heads (for GQA)
    fn num_key_value_heads(&self) -> Option<usize>;
    
    /// Get maximum position embeddings
    fn max_position_embeddings(&self) -> usize;
    
    /// Get RMS norm epsilon
    fn rms_norm_eps(&self) -> f64;
    
    /// Get RoPE theta base
    fn rope_theta(&self) -> f64;
}

/// LLaMA model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate dimension (FFN)
    pub intermediate_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of KV heads (for GQA, LLaMA-2/3)
    pub num_key_value_heads: Option<usize>,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f64,
    /// RoPE theta base
    pub rope_theta: f64,
    /// Whether to use tied word embeddings
    pub tie_word_embeddings: bool,
    /// Attention bias
    pub attention_bias: bool,
}

impl LlamaConfig {
    /// Create a new LLaMA config with default values for LLaMA-2 7B
    pub fn llama_7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None, // LLaMA-1 uses same KV heads as Q
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            attention_bias: false,
        }
    }

    /// Create config for LLaMA-2 7B
    pub fn llama_2_7b() -> Self {
        let mut config = Self::llama_7b();
        config.num_key_value_heads = Some(32); // LLaMA-2 uses GQA
        config.max_position_embeddings = 4096;
        config
    }

    /// Create config for LLaMA-3 8B
    pub fn llama_3_8b() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8), // GQA with 8 KV heads
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            tie_word_embeddings: false,
            attention_bias: false,
        }
    }

    /// Get number of KV heads (defaults to num_attention_heads if not specified)
    pub fn get_num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get number of Q heads per KV head (for GQA)
    pub fn q_per_kv(&self) -> usize {
        self.num_attention_heads / self.get_num_key_value_heads()
    }
}

impl ModelConfigTrait for LlamaConfig {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    fn num_key_value_heads(&self) -> Option<usize> {
        self.num_key_value_heads
    }

    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }

    fn rope_theta(&self) -> f64 {
        self.rope_theta
    }
}

/// Mistral model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate dimension (FFN)
    pub intermediate_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_key_value_heads: usize,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f64,
    /// RoPE theta base
    pub rope_theta: f64,
    /// Sliding window size (Mistral specific)
    pub sliding_window: Option<usize>,
    /// Whether to use tied word embeddings
    pub tie_word_embeddings: bool,
    /// Attention bias
    pub attention_bias: bool,
}

impl MistralConfig {
    /// Create a new Mistral config with default values for Mistral 7B
    pub fn mistral_7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8, // GQA with 8 KV heads
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            sliding_window: Some(4096),
            tie_word_embeddings: false,
            attention_bias: false,
        }
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get number of Q heads per KV head
    pub fn q_per_kv(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

impl ModelConfigTrait for MistralConfig {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    fn num_key_value_heads(&self) -> Option<usize> {
        Some(self.num_key_value_heads)
    }

    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }

    fn rope_theta(&self) -> f64 {
        self.rope_theta
    }
}

/// Enum for any supported model config
#[derive(Debug, Clone)]
pub enum ModelConfig {
    /// LLaMA model configuration
    Llama(LlamaConfig),
    /// Mistral model configuration
    Mistral(MistralConfig),
}

impl ModelConfig {
    /// Load config from a JSON file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let value: serde_json::Value = serde_json::from_reader(reader)?;
        
        // Try to detect model type from config
        if value.get("sliding_window").is_some() {
            // Mistral has sliding_window
            let config: MistralConfig = serde_json::from_value(value)?;
            Ok(ModelConfig::Mistral(config))
        } else {
            // Default to LLaMA
            let config: LlamaConfig = serde_json::from_value(value)?;
            Ok(ModelConfig::Llama(config))
        }
    }

    /// Get the config as LlamaConfig if applicable
    pub fn as_llama(&self) -> Option<&LlamaConfig> {
        match self {
            ModelConfig::Llama(config) => Some(config),
            _ => None,
        }
    }

    /// Get the config as MistralConfig if applicable
    pub fn as_mistral(&self) -> Option<&MistralConfig> {
        match self {
            ModelConfig::Mistral(config) => Some(config),
            _ => None,
        }
    }
}

impl ModelConfigTrait for ModelConfig {
    fn vocab_size(&self) -> usize {
        match self {
            ModelConfig::Llama(c) => c.vocab_size(),
            ModelConfig::Mistral(c) => c.vocab_size(),
        }
    }

    fn hidden_size(&self) -> usize {
        match self {
            ModelConfig::Llama(c) => c.hidden_size(),
            ModelConfig::Mistral(c) => c.hidden_size(),
        }
    }

    fn intermediate_size(&self) -> usize {
        match self {
            ModelConfig::Llama(c) => c.intermediate_size(),
            ModelConfig::Mistral(c) => c.intermediate_size(),
        }
    }

    fn num_hidden_layers(&self) -> usize {
        match self {
            ModelConfig::Llama(c) => c.num_hidden_layers(),
            ModelConfig::Mistral(c) => c.num_hidden_layers(),
        }
    }

    fn num_attention_heads(&self) -> usize {
        match self {
            ModelConfig::Llama(c) => c.num_attention_heads(),
            ModelConfig::Mistral(c) => c.num_attention_heads(),
        }
    }

    fn num_key_value_heads(&self) -> Option<usize> {
        match self {
            ModelConfig::Llama(c) => c.num_key_value_heads(),
            ModelConfig::Mistral(c) => c.num_key_value_heads(),
        }
    }

    fn max_position_embeddings(&self) -> usize {
        match self {
            ModelConfig::Llama(c) => c.max_position_embeddings(),
            ModelConfig::Mistral(c) => c.max_position_embeddings(),
        }
    }

    fn rms_norm_eps(&self) -> f64 {
        match self {
            ModelConfig::Llama(c) => c.rms_norm_eps(),
            ModelConfig::Mistral(c) => c.rms_norm_eps(),
        }
    }

    fn rope_theta(&self) -> f64 {
        match self {
            ModelConfig::Llama(c) => c.rope_theta(),
            ModelConfig::Mistral(c) => c.rope_theta(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_7b_config() {
        let config = LlamaConfig::llama_7b();
        
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.intermediate_size, 11008);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn test_llama_2_7b_config() {
        let config = LlamaConfig::llama_2_7b();
        
        assert_eq!(config.num_key_value_heads, Some(32));
        assert_eq!(config.max_position_embeddings, 4096);
    }

    #[test]
    fn test_llama_3_8b_config() {
        let config = LlamaConfig::llama_3_8b();
        
        assert_eq!(config.vocab_size, 128256);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.intermediate_size, 14336);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, Some(8));
        assert_eq!(config.q_per_kv(), 4);
        assert_eq!(config.max_position_embeddings, 8192);
    }

    #[test]
    fn test_mistral_7b_config() {
        let config = MistralConfig::mistral_7b();
        
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.sliding_window, Some(4096));
        assert_eq!(config.q_per_kv(), 4);
    }
}
