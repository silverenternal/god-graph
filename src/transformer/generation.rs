//! Text generation utilities

use super::model::LlamaModel;
use crate::tensor::DenseTensor;

/// Generation configuration
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum length to generate
    pub max_length: usize,
    /// Minimum length to generate
    pub min_length: usize,
    /// Temperature for sampling (higher = more random)
    pub temperature: f64,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) sampling (0.0 = disabled)
    pub top_p: f64,
    /// Repetition penalty
    pub repetition_penalty: f64,
    /// EOS token ID
    pub eos_token_id: Option<usize>,
    /// Pad token ID
    pub pad_token_id: Option<usize>,
    /// Do sample (if false, use greedy decoding)
    pub do_sample: bool,
    /// Number of beams for beam search (1 = disabled)
    pub num_beams: usize,
    /// Length penalty for beam search
    pub length_penalty: f64,
}

impl GenerationConfig {
    /// Create a default generation config
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            max_length: 256,
            min_length: 0,
            temperature: 1.0,
            top_k: 0,
            top_p: 0.0,
            repetition_penalty: 1.0,
            eos_token_id: None,
            pad_token_id: None,
            do_sample: false,
            num_beams: 1,
            length_penalty: 1.0,
        }
    }

    /// Create config for greedy decoding
    pub fn greedy() -> Self {
        Self {
            do_sample: false,
            ..Self::default()
        }
    }

    /// Create config for sampling
    pub fn sampling(temperature: f64) -> Self {
        Self {
            do_sample: true,
            temperature,
            ..Self::default()
        }
    }

    /// Create config for beam search
    pub fn beam_search(num_beams: usize) -> Self {
        Self {
            do_sample: false,
            num_beams,
            ..Self::default()
        }
    }

    /// Set maximum length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-k
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set top-p
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set EOS token ID
    pub fn with_eos_token_id(mut self, eos_token_id: usize) -> Self {
        self.eos_token_id = Some(eos_token_id);
        self
    }
}

/// Text generator for LLaMA models
pub struct TextGenerator<'a> {
    /// Reference to the model
    model: &'a LlamaModel,
    /// Generation configuration
    config: GenerationConfig,
}

impl<'a> TextGenerator<'a> {
    /// Create a new text generator
    pub fn new(model: &'a LlamaModel, config: GenerationConfig) -> Self {
        Self { model, config }
    }

    /// Generate text from input prompt
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs [seq_len]
    ///
    /// # Returns
    /// Generated token IDs
    pub fn generate(&self, input_ids: &[usize]) -> Vec<usize> {
        if self.config.num_beams > 1 {
            self.generate_beam_search(input_ids)
        } else if self.config.do_sample {
            self.generate_sampling(input_ids)
        } else {
            self.generate_greedy(input_ids)
        }
    }

    /// Greedy decoding (always pick the highest probability token)
    fn generate_greedy(&self, input_ids: &[usize]) -> Vec<usize> {
        let mut current_ids = input_ids.to_vec();

        for _ in 0..self.config.max_length {
            // Forward pass
            let logits = self.model.forward_single(&current_ids, None);

            // Get logits for the last position
            let seq_len = current_ids.len();
            let last_logits = logits.get_row(seq_len - 1);

            // Apply temperature
            let mut probs = last_logits.clone();
            if self.config.temperature != 1.0 {
                probs = probs.scale(1.0 / self.config.temperature);
            }

            // Apply softmax
            probs = probs.softmax(-1);

            // Greedy: pick the token with highest probability
            let next_token = self.argmax(probs.data());

            // Check for EOS
            if Some(next_token) == self.config.eos_token_id {
                break;
            }

            current_ids.push(next_token);
        }

        current_ids
    }

    /// Sampling-based generation
    fn generate_sampling(&self, input_ids: &[usize]) -> Vec<usize> {
        let mut current_ids = input_ids.to_vec();
        let mut rng = rand::thread_rng();

        for _ in 0..self.config.max_length {
            // Forward pass
            let logits = self.model.forward_single(&current_ids, None);

            // Get logits for the last position
            let seq_len = current_ids.len();
            let last_logits = logits.get_row(seq_len - 1);

            // Apply temperature
            let mut probs = last_logits.clone();
            if self.config.temperature != 1.0 {
                probs = probs.scale(1.0 / self.config.temperature);
            }

            // Apply softmax
            probs = probs.softmax(-1);

            // Apply top-k filtering
            if self.config.top_k > 0 {
                probs = self.top_k_filtering(&probs, self.config.top_k);
            }

            // Apply top-p (nucleus) filtering
            if self.config.top_p > 0.0 {
                probs = self.top_p_filtering(&probs, self.config.top_p);
            }

            // Sample from the distribution
            let next_token = self.sample_from_probs(probs.data(), &mut rng);

            // Check for EOS
            if Some(next_token) == self.config.eos_token_id {
                break;
            }

            current_ids.push(next_token);
        }

        current_ids
    }

    /// Beam search generation
    fn generate_beam_search(&self, input_ids: &[usize]) -> Vec<usize> {
        // Simplified beam search implementation
        // A full implementation would track multiple hypotheses

        let mut beams: Vec<(Vec<usize>, f64)> = vec![(input_ids.to_vec(), 0.0)];

        for _ in 0..self.config.max_length {
            let mut candidates: Vec<(Vec<usize>, f64)> = Vec::new();

            for (beam_ids, beam_score) in &beams {
                // Forward pass
                let logits = self.model.forward_single(beam_ids, None);

                // Get logits for the last position
                let seq_len = beam_ids.len();
                let last_logits = logits.get_row(seq_len - 1);

                // Get top-k candidates
                let top_indices = self.topk_indices(last_logits.data(), self.config.num_beams);

                for &next_token in &top_indices {
                    let mut new_beam = beam_ids.clone();
                    new_beam.push(next_token);

                    // Update score (log probability)
                    let token_prob = last_logits.data()[next_token];
                    let new_score = beam_score + token_prob.ln();

                    candidates.push((new_beam, new_score));
                }
            }

            // Keep top-k beams
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            beams = candidates.into_iter().take(self.config.num_beams).collect();

            // Check if all beams reached EOS
            if beams
                .iter()
                .all(|(ids, _)| ids.last() == self.config.eos_token_id.as_ref())
            {
                break;
            }
        }

        // Return the best beam
        beams
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(ids, _)| ids)
            .unwrap_or_else(|| input_ids.to_vec())
    }

    /// Argmax: find index of maximum value
    fn argmax(&self, data: &[f64]) -> usize {
        data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Top-k indices
    fn topk_indices(&self, data: &[f64], k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, &f64)> = data.iter().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// Top-k filtering: zero out probabilities outside top-k
    fn top_k_filtering(&self, probs: &DenseTensor, k: usize) -> DenseTensor {
        let data = probs.data();
        let top_indices = self.topk_indices(data, k);
        let threshold = top_indices
            .iter()
            .map(|&i| data[i])
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let mut filtered = probs.clone();
        for (i, &prob) in data.iter().enumerate() {
            if prob < threshold {
                filtered.data_mut()[i] = 0.0;
            }
        }

        // Re-normalize
        let sum: f64 = filtered.data().iter().sum();
        if sum > 0.0 {
            filtered = filtered.scale(1.0 / sum);
        }

        filtered
    }

    /// Top-p (nucleus) filtering: keep smallest set of tokens with cumulative prob >= p
    fn top_p_filtering(&self, probs: &DenseTensor, p: f64) -> DenseTensor {
        let data = probs.data();
        let mut indexed: Vec<(usize, &f64)> = data.iter().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumulative_prob = 0.0;
        let mut cutoff_index = indexed.len();

        for (i, (_, &prob)) in indexed.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob >= p {
                cutoff_index = i + 1;
                break;
            }
        }

        let threshold = indexed
            .into_iter()
            .take(cutoff_index)
            .map(|(_, &prob)| prob)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let mut filtered = probs.clone();
        for (i, &prob) in data.iter().enumerate() {
            if prob < threshold {
                filtered.data_mut()[i] = 0.0;
            }
        }

        // Re-normalize
        let sum: f64 = filtered.data().iter().sum();
        if sum > 0.0 {
            filtered = filtered.scale(1.0 / sum);
        }

        filtered
    }

    /// Sample from probability distribution
    fn sample_from_probs(&self, probs: &[f64], rng: &mut impl rand::Rng) -> usize {
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;

        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if r < cumulative {
                return i;
            }
        }

        probs.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model() -> LlamaModel {
        use super::super::layers::{FeedForward, MultiHeadAttention, RMSNorm};
        use super::super::loader::LlamaConfig;
        use super::super::model::LlamaModel;
        use crate::tensor::DenseTensor;

        // Use tiny dimensions to avoid OOM in tests
        let mut config = LlamaConfig::llama_7b();
        config.vocab_size = 64;
        config.hidden_size = 16;
        config.intermediate_size = 32;
        config.num_attention_heads = 2;

        let embed_tokens = DenseTensor::ones(vec![config.vocab_size, config.hidden_size]);

        let hidden_dim = config.hidden_size;
        let num_heads = config.num_attention_heads;

        let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_k = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_v = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
        let self_attn = MultiHeadAttention::standard(w_q, w_k, w_v, w_o, num_heads);

        let gate_proj = DenseTensor::ones(vec![hidden_dim, config.intermediate_size]);
        let up_proj = DenseTensor::ones(vec![hidden_dim, config.intermediate_size]);
        let down_proj = DenseTensor::ones(vec![config.intermediate_size, hidden_dim]);
        let mlp = FeedForward::swiglu(gate_proj, up_proj, down_proj);

        let input_layernorm = RMSNorm::default(hidden_dim);
        let post_attention_layernorm = RMSNorm::default(hidden_dim);

        let layer = super::super::model::LlamaDecoderLayer::new(
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        );

        let layers = vec![layer; 2];
        let norm = RMSNorm::default(hidden_dim);

        LlamaModel::new(config, embed_tokens, layers, norm, None)
    }

    #[test]
    fn test_generation_config() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_length, 256);
        assert_eq!(config.temperature, 1.0);
        assert!(!config.do_sample);

        let greedy = GenerationConfig::greedy();
        assert!(!greedy.do_sample);

        let sampling = GenerationConfig::sampling(0.8);
        assert!(sampling.do_sample);
        assert_eq!(sampling.temperature, 0.8);
    }

    #[test]
    fn test_argmax() {
        let model = create_test_model();
        let generator = TextGenerator::new(&model, GenerationConfig::default());

        let data = vec![0.1, 0.3, 0.5, 0.2, 0.4];
        assert_eq!(generator.argmax(&data), 2);
    }

    #[test]
    fn test_topk_indices() {
        let model = create_test_model();
        let generator = TextGenerator::new(&model, GenerationConfig::default());

        let data = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let top2 = generator.topk_indices(&data, 2);
        assert_eq!(top2, vec![3, 1]);
    }
}
