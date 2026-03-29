//! Sparse Attention module for efficient attention computation
//!
//! This module provides various sparse attention patterns:
//! - Sliding window attention (used in Mistral)
//! - Block sparse attention
//! - Star attention
//! - Head-wise sparse attention

use crate::tensor::DenseTensor;
use crate::tensor::traits::{TensorOps, TensorBase};
use crate::tensor::sparse::SparseTensor;

/// Sparse attention pattern types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparsePattern {
    /// Sliding window attention
    SlidingWindow,
    /// Block sparse attention
    BlockSparse,
    /// Star attention (center node attends to all)
    Star,
    /// Head-wise sparse (different heads use different patterns)
    HeadSparse,
}

/// Configuration for sliding window attention
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Window size (number of tokens to attend to)
    pub window_size: usize,
    /// Left-only (causal) or bidirectional
    pub causal: bool,
}

impl SlidingWindowConfig {
    /// Create a new sliding window config
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            causal: true,
        }
    }

    /// Create with bidirectional attention
    pub fn bidirectional(window_size: usize) -> Self {
        Self {
            window_size,
            causal: false,
        }
    }
}

/// Configuration for block sparse attention
#[derive(Debug, Clone)]
pub struct BlockSparseConfig {
    /// Block size
    pub block_size: usize,
    /// Number of blocks to attend to per query block
    pub num_blocks: usize,
}

impl BlockSparseConfig {
    /// Create a new block sparse config
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        Self {
            block_size,
            num_blocks,
        }
    }
}

/// Sparse attention mask
#[derive(Debug, Clone)]
pub struct SparseMask {
    /// Row offsets for CSR format [seq_len + 1]
    pub row_offsets: Vec<usize>,
    /// Column indices for CSR format [nnz]
    pub col_indices: Vec<usize>,
    /// Sequence length
    pub seq_len: usize,
    /// Number of non-zero elements
    pub nnz: usize,
}

impl SparseMask {
    /// Create a sliding window mask
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length
    /// * `window_size` - Window size
    /// * `causal` - Whether to use causal masking
    pub fn sliding_window(seq_len: usize, window_size: usize, causal: bool) -> Self {
        let mut row_offsets = Vec::with_capacity(seq_len + 1);
        let mut col_indices = Vec::new();

        row_offsets.push(0);

        for i in 0..seq_len {
            let start = if causal {
                (i + 1).saturating_sub(window_size)
            } else {
                i.saturating_sub(window_size)
            };
            let end = if causal {
                i + 1
            } else {
                (i + window_size).min(seq_len)
            };

            for j in start..end {
                col_indices.push(j);
            }

            row_offsets.push(col_indices.len());
        }

        let nnz = col_indices.len();

        Self {
            row_offsets,
            col_indices,
            seq_len,
            nnz,
        }
    }

    /// Create a block sparse mask
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length
    /// * `block_size` - Block size
    /// * `num_blocks` - Number of blocks to attend to
    pub fn block_sparse(seq_len: usize, block_size: usize, num_blocks: usize) -> Self {
        let _num_blocks_total = (seq_len + block_size - 1) / block_size;
        let mut row_offsets = Vec::with_capacity(seq_len + 1);
        let mut col_indices = Vec::new();

        row_offsets.push(0);

        for i in 0..seq_len {
            let block_id = i / block_size;

            // Attend to current block and previous blocks
            for b in 0..num_blocks.min(block_id + 1) {
                let src_block = block_id - b;
                let start = src_block * block_size;
                let end = (start + block_size).min(seq_len);

                for j in start..end {
                    col_indices.push(j);
                }
            }

            row_offsets.push(col_indices.len());
        }

        let nnz = col_indices.len();

        Self {
            row_offsets,
            col_indices,
            seq_len,
            nnz,
        }
    }

    /// Create a star attention mask
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length
    /// * `center_ratio` - Ratio of center tokens (e.g., 0.1 for 10%)
    pub fn star(seq_len: usize, center_ratio: f64) -> Self {
        let num_centers = (seq_len as f64 * center_ratio).ceil() as usize;
        let mut row_offsets = Vec::with_capacity(seq_len + 1);
        let mut col_indices = Vec::new();

        row_offsets.push(0);

        for i in 0..seq_len {
            // Center tokens attend to all
            if i < num_centers {
                for j in 0..seq_len {
                    col_indices.push(j);
                }
            } else {
                // Non-center tokens attend to centers and local window
                // Attend to centers
                for j in 0..num_centers {
                    col_indices.push(j);
                }
                // Attend to local window
                let window_start = i.saturating_sub(64);
                let window_end = (i + 64).min(seq_len);
                for j in window_start..window_end {
                    if !col_indices.contains(&j) {
                        col_indices.push(j);
                    }
                }
            }

            row_offsets.push(col_indices.len());
        }

        let nnz = col_indices.len();

        Self {
            row_offsets,
            col_indices,
            seq_len,
            nnz,
        }
    }

    /// Convert to sparse tensor
    pub fn to_sparse_tensor(&self, values: Vec<f64>) -> SparseTensor {
        let values_tensor = DenseTensor::new(values, vec![self.nnz]);
        SparseTensor::csr(
            self.row_offsets.clone(),
            self.col_indices.clone(),
            values_tensor,
            [self.seq_len, self.seq_len],
        )
    }

    /// Get sparsity ratio
    pub fn sparsity(&self) -> f64 {
        let total = self.seq_len * self.seq_len;
        1.0 - (self.nnz as f64 / total as f64)
    }

    /// Apply mask to attention scores
    ///
    /// # Arguments
    /// * `scores` - Attention scores [batch, heads, seq_len, seq_len]
    pub fn apply(&self, scores: &DenseTensor) -> DenseTensor {
        let mut masked = scores.clone();
        let data = masked.data_mut();

        // Set masked positions to -inf
        for i in 0..self.seq_len {
            let start = self.row_offsets[i];
            let end = self.row_offsets[i + 1];

            for j in 0..self.seq_len {
                // Check if (i, j) is in the mask
                let is_valid = self.col_indices[start..end].contains(&j);

                if !is_valid {
                    // Set to -inf (use a large negative number)
                    let offset = i * self.seq_len + j;
                    if offset < data.len() {
                        data[offset] = f64::NEG_INFINITY;
                    }
                }
            }
        }

        masked
    }
}

/// Sparse attention module
#[derive(Debug, Clone)]
pub struct SparseAttention {
    /// Sparse pattern
    pub pattern: SparsePattern,
    /// Sparse mask
    pub mask: Option<SparseMask>,
    /// Window size (for sliding window)
    pub window_size: Option<usize>,
    /// Block size (for block sparse)
    pub block_size: Option<usize>,
    /// Number of blocks (for block sparse)
    pub num_blocks: Option<usize>,
    /// Scale factor
    pub scale: f64,
}

impl SparseAttention {
    /// Create a new sparse attention module
    ///
    /// # Arguments
    /// * `pattern` - Sparse pattern type
    /// * `head_dim` - Head dimension for scaling
    pub fn new(pattern: SparsePattern, head_dim: usize) -> Self {
        Self {
            pattern,
            mask: None,
            window_size: None,
            block_size: None,
            num_blocks: None,
            scale: 1.0 / (head_dim as f64).sqrt(),
        }
    }

    /// Create sliding window attention
    ///
    /// # Arguments
    /// * `head_dim` - Head dimension
    /// * `window_size` - Window size
    pub fn sliding_window(head_dim: usize, window_size: usize) -> Self {
        let mut self_ = Self::new(SparsePattern::SlidingWindow, head_dim);
        self_.window_size = Some(window_size);
        self_
    }

    /// Create block sparse attention
    ///
    /// # Arguments
    /// * `head_dim` - Head dimension
    /// * `block_size` - Block size
    /// * `num_blocks` - Number of blocks to attend to
    pub fn block_sparse(head_dim: usize, block_size: usize, num_blocks: usize) -> Self {
        let mut self_ = Self::new(SparsePattern::BlockSparse, head_dim);
        self_.block_size = Some(block_size);
        self_.num_blocks = Some(num_blocks);
        self_
    }

    /// Create star attention
    ///
    /// # Arguments
    /// * `head_dim` - Head dimension
    /// * `center_ratio` - Ratio of center tokens
    pub fn star(head_dim: usize, _center_ratio: f64) -> Self {
        let self_ = Self::new(SparsePattern::Star, head_dim);
        self_
    }

    /// Build sparse mask for given sequence length
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length
    pub fn build_mask(&mut self, seq_len: usize) {
        self.mask = Some(match self.pattern {
            SparsePattern::SlidingWindow => {
                let window_size = self.window_size.unwrap_or(seq_len);
                SparseMask::sliding_window(seq_len, window_size, true)
            }
            SparsePattern::BlockSparse => {
                let block_size = self.block_size.unwrap_or(64);
                let num_blocks = self.num_blocks.unwrap_or(4);
                SparseMask::block_sparse(seq_len, block_size, num_blocks)
            }
            SparsePattern::Star => {
                SparseMask::star(seq_len, 0.1)
            }
            SparsePattern::HeadSparse => {
                // Default to sliding window for head-sparse
                SparseMask::sliding_window(seq_len, 64, true)
            }
        });
    }

    /// Compute sparse attention
    ///
    /// # Arguments
    /// * `query` - Query tensor [batch, heads, seq_len, head_dim]
    /// * `key` - Key tensor [batch, heads, seq_len, head_dim]
    /// * `value` - Value tensor [batch, heads, seq_len, head_dim]
    ///
    /// # Returns
    /// Attention output [batch, heads, seq_len, head_dim]
    pub fn forward(
        &mut self,
        query: &DenseTensor,
        key: &DenseTensor,
        value: &DenseTensor,
    ) -> DenseTensor {
        let seq_len = query.shape()[2];

        // Build mask if not already built
        if self.mask.is_none() || self.mask.as_ref().unwrap().seq_len != seq_len {
            self.build_mask(seq_len);
        }

        // Compute attention scores
        let key_t = key.transpose(None);
        let mut scores = query.matmul(&key_t);
        scores = scores.scale(self.scale);

        // Apply sparse mask
        if let Some(mask) = &self.mask {
            scores = mask.apply(&scores);
        }

        // Apply softmax
        let attn_weights = scores.softmax(-1);

        // Apply attention to values
        attn_weights.matmul(value)
    }

    /// Get sparsity ratio
    pub fn sparsity(&self) -> f64 {
        self.mask.as_ref().map(|m| m.sparsity()).unwrap_or(0.0)
    }
}

/// Sliding window attention helper
pub struct SlidingWindowAttention {
    window_size: usize,
    scale: f64,
}

impl SlidingWindowAttention {
    /// Create a new sliding window attention
    pub fn new(window_size: usize, head_dim: usize) -> Self {
        Self {
            window_size,
            scale: 1.0 / (head_dim as f64).sqrt(),
        }
    }

    /// Compute sliding window attention efficiently
    ///
    /// # Arguments
    /// * `query` - Query [batch, heads, seq_len, head_dim]
    /// * `key` - Key [batch, heads, seq_len, head_dim]
    /// * `value` - Value [batch, heads, seq_len, head_dim]
    pub fn forward(&self, query: &DenseTensor, key: &DenseTensor, value: &DenseTensor) -> DenseTensor {
        let batch_size = query.shape()[0];
        let num_heads = query.shape()[1];
        let seq_len = query.shape()[2];
        let head_dim = query.shape()[3];

        let mut output_data = Vec::with_capacity(batch_size * num_heads * seq_len * head_dim);

        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    // Compute attention for position i
                    let mut attn_output = vec![0.0; head_dim];
                    let mut total_weight = 0.0;

                    // Only attend to window
                    let start = i.saturating_sub(self.window_size);
                    let end = i + 1;

                    for j in start..end {
                        // Compute dot product
                        let q_slice = &query.data()[(b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim)..];
                        let k_slice = &key.data()[(b * num_heads * seq_len * head_dim + h * seq_len * head_dim + j * head_dim)..];

                        let mut score = 0.0;
                        for d in 0..head_dim {
                            score += q_slice[d] * k_slice[d];
                        }
                        score *= self.scale;

                        // Softmax weight
                        let weight = score.exp();

                        // Weighted sum of values
                        let v_slice = &value.data()[(b * num_heads * seq_len * head_dim + h * seq_len * head_dim + j * head_dim)..];
                        for d in 0..head_dim {
                            attn_output[d] += weight * v_slice[d];
                        }
                        total_weight += weight;
                    }

                    // Normalize
                    if total_weight > 0.0 {
                        for d in 0..head_dim {
                            attn_output[d] /= total_weight;
                        }
                    }

                    output_data.extend(attn_output);
                }
            }
        }

        DenseTensor::new(output_data, vec![batch_size, num_heads, seq_len, head_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sliding_window_mask() {
        let mask = SparseMask::sliding_window(10, 3, true);

        assert_eq!(mask.seq_len, 10);
        assert!(mask.nnz < 10 * 10); // Should be sparse
        assert_eq!(mask.row_offsets.len(), 11);
    }

    #[test]
    fn test_block_sparse_mask() {
        let mask = SparseMask::block_sparse(16, 4, 2);

        assert_eq!(mask.seq_len, 16);
        assert!(mask.nnz < 16 * 16);
    }

    #[test]
    fn test_star_mask() {
        let mask = SparseMask::star(20, 0.1);

        assert_eq!(mask.seq_len, 20);
        // Center tokens (2) should attend to all
        // Non-center tokens should attend to centers + local
    }

    #[test]
    fn test_sparsity_calculation() {
        let mask = SparseMask::sliding_window(100, 10, true);
        let sparsity = mask.sparsity();

        // Should be approximately 90% sparse
        assert!(sparsity > 0.8);
        assert!(sparsity < 1.0);
    }

    #[test]
    fn test_sparse_attention_sliding_window() {
        let mut attn = SparseAttention::sliding_window(64, 10);
        attn.build_mask(20);

        assert_eq!(attn.pattern, SparsePattern::SlidingWindow);
        assert!(attn.mask.is_some());
    }

    #[test]
    fn test_sliding_window_attention_forward() {
        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 8;
        let head_dim = 16;

        let query = DenseTensor::ones(vec![batch_size, num_heads, seq_len, head_dim]);
        let key = DenseTensor::ones(vec![batch_size, num_heads, seq_len, head_dim]);
        let value = DenseTensor::ones(vec![batch_size, num_heads, seq_len, head_dim]);

        let attn = SlidingWindowAttention::new(4, head_dim);
        let output = attn.forward(&query, &key, &value);

        assert_eq!(output.shape(), &[batch_size, num_heads, seq_len, head_dim]);
    }
}
