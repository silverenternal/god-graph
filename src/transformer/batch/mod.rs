//! Batch inference module for efficient throughput
//!
//! This module provides:
//! - Batch forward pass
//! - Continuous batching (vLLM-style)
//! - Request scheduling

use crate::tensor::DenseTensor;
use super::model::LlamaModel;
use super::generation::GenerationConfig;
use super::kv_cache::KVCache;

/// Batch data for inference
#[derive(Debug, Clone)]
pub struct BatchData {
    /// Input token IDs for each sequence in batch
    pub input_ids: Vec<Vec<usize>>,
    /// Attention mask [batch_size, seq_len, seq_len]
    pub attention_mask: Option<DenseTensor>,
    /// Position IDs [batch_size, seq_len]
    pub position_ids: Option<Vec<Vec<usize>>>,
    /// Sequence lengths
    pub seq_lengths: Vec<usize>,
}

impl BatchData {
    /// Create a new batch from input sequences
    ///
    /// # Arguments
    /// * `input_ids` - List of input sequences
    pub fn new(input_ids: Vec<Vec<usize>>) -> Self {
        let seq_lengths: Vec<usize> = input_ids.iter().map(|ids| ids.len()).collect();
        let max_len = seq_lengths.iter().max().copied().unwrap_or(0);

        // Pad sequences to max length
        let mut padded_ids = Vec::new();
        for ids in &input_ids {
            let mut padded = ids.clone();
            while padded.len() < max_len {
                padded.push(0); // Pad with 0
            }
            padded_ids.push(padded);
        }

        // Create attention mask
        let batch_size = input_ids.len();
        let mut mask_data = Vec::with_capacity(batch_size * max_len * max_len);

        for &seq_len in seq_lengths.iter() {
            for j in 0..max_len {
                for k in 0..max_len {
                    // Valid positions can attend to each other
                    let can_attend = (j < seq_len && k < seq_len) as u8 as f64;
                    mask_data.push(if can_attend == 1.0 { 0.0 } else { f64::NEG_INFINITY });
                }
            }
        }

        let attention_mask = Some(DenseTensor::new(mask_data, vec![batch_size, max_len, max_len]));

        Self {
            input_ids: padded_ids,
            attention_mask,
            position_ids: None,
            seq_lengths,
        }
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.input_ids.len()
    }

    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.seq_lengths.iter().max().copied().unwrap_or(0)
    }

    /// Get padded input IDs as 2D vector
    pub fn padded_input_ids(&self) -> &[Vec<usize>] {
        &self.input_ids
    }
}

/// Inference request
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Request ID
    pub id: usize,
    /// Input token IDs
    pub input_ids: Vec<usize>,
    /// Generation configuration
    pub config: GenerationConfig,
    /// Generated tokens so far
    pub generated: Vec<usize>,
    /// Whether request is complete
    pub completed: bool,
    /// Priority (lower = higher priority)
    pub priority: usize,
}

impl InferenceRequest {
    /// Create a new inference request
    pub fn new(id: usize, input_ids: Vec<usize>, config: GenerationConfig) -> Self {
        Self {
            id,
            input_ids: input_ids.clone(),
            config,
            generated: input_ids,
            completed: false,
            priority: 0,
        }
    }

    /// Add generated token
    pub fn append_token(&mut self, token: usize) {
        self.generated.push(token);

        // Check completion
        if self.generated.len() >= self.config.max_length {
            self.completed = true;
        }
        if let Some(eos) = self.config.eos_token_id {
            if token == eos {
                self.completed = true;
            }
        }
    }

    /// Get current sequence length
    pub fn current_len(&self) -> usize {
        self.generated.len()
    }
}

/// Request scheduler for continuous batching
#[derive(Debug)]
pub struct RequestScheduler {
    /// Pending requests
    pending: Vec<InferenceRequest>,
    /// Active requests
    active: Vec<InferenceRequest>,
    /// Completed requests
    completed: Vec<InferenceRequest>,
    /// Next request ID
    next_id: usize,
    /// Maximum batch size
    max_batch_size: usize,
}

impl RequestScheduler {
    /// Create a new scheduler
    ///
    /// # Arguments
    /// * `max_batch_size` - Maximum number of concurrent requests
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            pending: Vec::new(),
            active: Vec::new(),
            completed: Vec::new(),
            next_id: 0,
            max_batch_size,
        }
    }

    /// Add a new request
    pub fn add_request(&mut self, input_ids: Vec<usize>, config: GenerationConfig) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let request = InferenceRequest::new(id, input_ids, config);
        self.pending.push(request);

        id
    }

    /// Schedule requests for next batch
    pub fn schedule(&mut self) -> Vec<&mut InferenceRequest> {
        // Move completed active requests to completed
        self.active.retain(|req| {
            !req.completed
        });

        // Move pending to active if there's capacity
        while !self.pending.is_empty() && self.active.len() < self.max_batch_size {
            let request = self.pending.remove(0);
            self.active.push(request);
        }

        // Return mutable references to active requests
        self.active.iter_mut().collect()
    }

    /// Get number of pending requests
    pub fn num_pending(&self) -> usize {
        self.pending.len()
    }

    /// Get number of active requests
    pub fn num_active(&self) -> usize {
        self.active.len()
    }

    /// Get number of completed requests
    pub fn num_completed(&self) -> usize {
        self.completed.len()
    }

    /// Remove and return completed requests
    pub fn pop_completed(&mut self) -> Vec<InferenceRequest> {
        
        std::mem::take(&mut self.completed)
    }
}

/// Batch inference engine
#[derive(Debug)]
pub struct BatchInference<'a> {
    /// Reference to model
    model: &'a LlamaModel,
    /// KV caches for each layer
    kv_caches: Vec<KVCache>,
    /// Current batch size
    batch_size: usize,
}

impl<'a> BatchInference<'a> {
    /// Create a new batch inference engine
    ///
    /// # Arguments
    /// * `model` - Reference to LlamaModel
    /// * `max_batch_size` - Maximum batch size
    /// * `max_seq_len` - Maximum sequence length
    pub fn new(model: &'a LlamaModel, max_batch_size: usize, max_seq_len: usize) -> Self {
        let kv_caches = vec![
            KVCache::new(
                model.num_layers(),
                max_seq_len,
                model.hidden_dim(),
                model.config.get_num_key_value_heads(),
            );
            max_batch_size
        ];

        Self {
            model,
            kv_caches,
            batch_size: 0,
        }
    }

    /// Run batch forward pass
    ///
    /// # Arguments
    /// * `batch` - Batch data
    ///
    /// # Returns
    /// Logits for each sequence in batch [batch_size, seq_len, vocab_size]
    pub fn forward(&mut self, batch: &BatchData) -> DenseTensor {
        let batch_size = batch.batch_size();
        self.batch_size = batch_size;

        // Run model forward pass with batched input
        self.model.forward(&batch.input_ids, batch.attention_mask.as_ref())
    }

    /// Run single step of generation for batch
    ///
    /// # Arguments
    /// * `requests` - Active inference requests
    ///
    /// # Returns
    /// Generated tokens for each request
    pub fn step(&mut self, requests: &[&mut InferenceRequest]) -> Vec<usize> {
        // Collect current tokens
        let input_ids: Vec<Vec<usize>> = requests
            .iter()
            .map(|req| vec![*req.generated.last().unwrap()])
            .collect();

        let batch = BatchData::new(input_ids);

        // Forward pass
        let logits = self.forward(&batch);

        // Sample tokens
        let mut tokens = Vec::new();
        for (i, req) in requests.iter().enumerate() {
            let seq_len = req.current_len();
            let token_logits = logits.get_row(i * seq_len + seq_len - 1);

            // Apply temperature
            let mut probs = token_logits.clone();
            if req.config.temperature != 1.0 {
                probs = probs.scale(1.0 / req.config.temperature);
            }

            // Softmax
            probs = probs.softmax(-1);

            // Sample or greedy
            let token = if req.config.do_sample {
                self.sample_from_probs(probs.data())
            } else {
                self.argmax(probs.data())
            };

            tokens.push(token);
        }

        tokens
    }

    /// Run continuous batching generation
    ///
    /// # Arguments
    /// * `scheduler` - Request scheduler
    ///
    /// # Returns
    /// Generated sequences for each request
    pub fn generate_continuous(&mut self, scheduler: &mut RequestScheduler) -> Vec<Vec<usize>> {
        let mut results: Vec<Option<Vec<usize>>> = Vec::new();

        // Initialize results
        for _ in 0..scheduler.next_id {
            results.push(None);
        }

        // Generation loop
        while scheduler.num_active() > 0 || scheduler.num_pending() > 0 {
            // Schedule requests
            let mut active_requests = scheduler.schedule();

            if active_requests.is_empty() {
                break;
            }

            // Generate step
            let tokens = self.step(&active_requests);

            // Update requests
            for (req, token) in active_requests.iter_mut().zip(tokens) {
                req.append_token(token);

                if req.completed {
                    // Store result
                    results[req.id] = Some(req.generated.clone());
                }
            }
        }

        // Collect results
        results.into_iter().flatten().collect()
    }

    /// Reset KV caches
    pub fn reset(&mut self) {
        for cache in &mut self.kv_caches {
            cache.reset();
        }
    }

    /// Argmax sampling
    fn argmax(&self, probs: &[f64]) -> usize {
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Sample from probability distribution
    fn sample_from_probs(&self, probs: &[f64]) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();
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

/// Utility functions for batch processing
pub mod utils {
    use super::*;

    /// Pad sequences to same length
    pub fn pad_sequences(sequences: &[Vec<usize>], pad_token: usize) -> (Vec<Vec<usize>>, Vec<usize>) {
        let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut padded = Vec::new();
        let mut lengths = Vec::new();

        for seq in sequences {
            lengths.push(seq.len());
            let mut padded_seq = seq.clone();
            while padded_seq.len() < max_len {
                padded_seq.push(pad_token);
            }
            padded.push(padded_seq);
        }

        (padded, lengths)
    }

    /// Create attention mask from lengths
    pub fn create_attention_mask(lengths: &[usize]) -> DenseTensor {
        let batch_size = lengths.len();
        let max_len = lengths.iter().max().copied().unwrap_or(0);

        let mut data = Vec::with_capacity(batch_size * max_len * max_len);

        for &seq_len in lengths.iter() {
            for j in 0..max_len {
                for k in 0..max_len {
                    let can_attend = (j < seq_len && k < seq_len) as u8 as f64;
                    data.push(if can_attend == 1.0 { 0.0 } else { f64::NEG_INFINITY });
                }
            }
        }

        DenseTensor::new(data, vec![batch_size, max_len, max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::model::LlamaModel;
    use crate::transformer::layers::{MultiHeadAttention, FeedForward, RMSNorm};
    use crate::transformer::loader::LlamaConfig;
    use crate::tensor::DenseTensor;

    fn create_test_model() -> LlamaModel {
        let config = LlamaConfig::llama_7b();
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
            self_attn, mlp, input_layernorm, post_attention_layernorm
        );

        let layers = vec![layer; 2];
        let norm = RMSNorm::default(hidden_dim);

        LlamaModel::new(config, embed_tokens, layers, norm, None)
    }

    #[test]
    fn test_batch_data_creation() {
        let input_ids = vec![
            vec![1, 2, 3],
            vec![4, 5],
            vec![6, 7, 8, 9],
        ];

        let batch = BatchData::new(input_ids.clone());

        assert_eq!(batch.batch_size(), 3);
        assert_eq!(batch.max_seq_len(), 4);
        assert_eq!(batch.seq_lengths, vec![3, 2, 4]);
    }

    #[test]
    fn test_inference_request() {
        let config = GenerationConfig::greedy();
        let mut request = InferenceRequest::new(0, vec![1, 2, 3], config);

        assert!(!request.completed);
        assert_eq!(request.current_len(), 3);

        request.append_token(4);
        assert_eq!(request.current_len(), 4);
    }

    #[test]
    fn test_request_scheduler() {
        let mut scheduler = RequestScheduler::new(2);

        let _id1 = scheduler.add_request(vec![1, 2, 3], GenerationConfig::greedy());
        let _id2 = scheduler.add_request(vec![4, 5], GenerationConfig::greedy());
        let _id3 = scheduler.add_request(vec![6, 7, 8], GenerationConfig::greedy());

        assert_eq!(scheduler.num_pending(), 3);
        assert_eq!(scheduler.num_active(), 0);

        let active = scheduler.schedule();
        assert_eq!(active.len(), 2); // max_batch_size = 2
        assert_eq!(scheduler.num_pending(), 1);
        assert_eq!(scheduler.num_active(), 2);
    }

    #[test]
    fn test_batch_inference_creation() {
        let model = create_test_model();
        let batch_infer = BatchInference::new(&model, 4, 512);

        assert_eq!(batch_infer.kv_caches.len(), 4);
    }

    #[test]
    fn test_pad_sequences() {
        let sequences = vec![
            vec![1, 2],
            vec![3, 4, 5],
            vec![6],
        ];

        let (padded, lengths) = utils::pad_sequences(&sequences, 0);

        assert_eq!(padded, vec![
            vec![1, 2, 0],
            vec![3, 4, 5],
            vec![6, 0, 0],
        ]);
        assert_eq!(lengths, vec![2, 3, 1]);
    }
}
