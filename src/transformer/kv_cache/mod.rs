//! KV Cache module for efficient autoregressive generation
//!
//! KV Cache caches key and value states to avoid recomputing them
//! during autoregressive generation, significantly improving inference speed.

use crate::tensor::traits::TensorBase;
use crate::tensor::DenseTensor;

/// KV Cache for caching key and value states during generation
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Key cache [num_layers, max_seq_len, hidden_dim]
    key_cache: Vec<DenseTensor>,
    /// Value cache [num_layers, max_seq_len, hidden_dim]
    value_cache: Vec<DenseTensor>,
    /// Current sequence length
    current_len: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Number of layers
    num_layers: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of KV heads (for GQA)
    num_kv_heads: usize,
}

impl KVCache {
    /// Create a new KV cache
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers
    /// * `max_seq_len` - Maximum sequence length to cache
    /// * `hidden_dim` - Hidden dimension
    /// * `num_kv_heads` - Number of KV heads (for GQA)
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        hidden_dim: usize,
        num_kv_heads: usize,
    ) -> Self {
        let head_dim = hidden_dim / num_kv_heads;
        let key_cache =
            vec![DenseTensor::zeros(vec![max_seq_len, num_kv_heads, head_dim]); num_layers];
        let value_cache =
            vec![DenseTensor::zeros(vec![max_seq_len, num_kv_heads, head_dim]); num_layers];

        Self {
            key_cache,
            value_cache,
            current_len: 0,
            max_seq_len,
            num_layers,
            hidden_dim,
            num_kv_heads,
        }
    }

    /// Update cache with new key and value
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `key` - New key [batch_size, num_kv_heads, head_dim]
    /// * `value` - New value [batch_size, num_kv_heads, head_dim]
    /// * `position` - Position to cache at
    pub fn update(
        &mut self,
        layer: usize,
        key: &DenseTensor,
        value: &DenseTensor,
        position: usize,
    ) {
        if layer >= self.num_layers || position >= self.max_seq_len {
            return;
        }

        // Update key cache
        if let Some(layer_key) = self.key_cache.get_mut(layer) {
            Self::copy_to_cache_static(layer_key, key, position, self.num_kv_heads);
        }

        // Update value cache
        if let Some(layer_value) = self.value_cache.get_mut(layer) {
            Self::copy_to_cache_static(layer_value, value, position, self.num_kv_heads);
        }

        // Update current length
        if position >= self.current_len {
            self.current_len = position + 1;
        }
    }

    /// Copy tensor to cache at specified position (static method to avoid borrow issues)
    #[inline]
    fn copy_to_cache_static(
        cache: &mut DenseTensor,
        tensor: &DenseTensor,
        position: usize,
        num_kv_heads: usize,
    ) {
        let batch_size = tensor.shape()[0];
        let head_dim = tensor.shape()[2];

        for b in 0..batch_size {
            for h in 0..num_kv_heads {
                let src_offset = (b * num_kv_heads + h) * head_dim;
                let dst_offset = (position * num_kv_heads + h) * head_dim;

                let src_slice = &tensor.data()[src_offset..src_offset + head_dim];
                let cache_data = cache.data_mut();
                cache_data[dst_offset..dst_offset + head_dim].copy_from_slice(src_slice);
            }
        }
    }

    /// Get cached keys and values for a layer
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `length` - Number of cached positions to retrieve
    ///
    /// # Returns
    /// Tuple of (key_cache, value_cache) with shape [batch_size, length, num_kv_heads, head_dim]
    pub fn get(&self, layer: usize, length: Option<usize>) -> Option<(DenseTensor, DenseTensor)> {
        if layer >= self.num_layers {
            return None;
        }

        let key_cache = self.key_cache.get(layer)?;
        let value_cache = self.value_cache.get(layer)?;

        let seq_len = length.unwrap_or(self.current_len);

        // Slice cache to current length
        let key = self.slice_cache(key_cache, seq_len);
        let value = self.slice_cache(value_cache, seq_len);

        Some((key, value))
    }

    /// Slice cache to specified length
    fn slice_cache(&self, cache: &DenseTensor, length: usize) -> DenseTensor {
        let num_kv_heads = cache.shape()[1];
        let head_dim = cache.shape()[2];

        let mut data = Vec::with_capacity(length * num_kv_heads * head_dim);

        for pos in 0..length {
            for h in 0..num_kv_heads {
                let offset = (pos * num_kv_heads + h) * head_dim;
                data.extend_from_slice(&cache.data()[offset..offset + head_dim]);
            }
        }

        DenseTensor::new(data, vec![length, num_kv_heads, head_dim])
    }

    /// Get all cached keys and values for a layer (full history)
    ///
    /// # Arguments
    /// * `layer` - Layer index
    ///
    /// # Returns
    /// Tuple of (key_cache, value_cache)
    pub fn get_all(&self, layer: usize) -> Option<(DenseTensor, DenseTensor)> {
        self.get(layer, Some(self.current_len))
    }

    /// Reset cache for new sequence
    pub fn reset(&mut self) {
        self.current_len = 0;

        // Zero out caches
        for key_cache in &mut self.key_cache {
            *key_cache = DenseTensor::zeros(key_cache.shape().to_vec());
        }
        for value_cache in &mut self.value_cache {
            *value_cache = DenseTensor::zeros(value_cache.shape().to_vec());
        }
    }

    /// Get current sequence length
    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get number of KV heads
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Check if cache is full
    pub fn is_full(&self) -> bool {
        self.current_len >= self.max_seq_len
    }

    /// Get remaining capacity
    pub fn remaining_capacity(&self) -> usize {
        self.max_seq_len - self.current_len
    }

    /// Append new token's KV without position argument (auto-increment)
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `key` - New key [1, num_kv_heads, head_dim]
    /// * `value` - New value [1, num_kv_heads, head_dim]
    pub fn append(&mut self, layer: usize, key: &DenseTensor, value: &DenseTensor) {
        if self.is_full() {
            return;
        }
        self.update(layer, key, value, self.current_len);
    }

    /// Concatenate cached KV with new KV for attention computation
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `new_key` - New key to append
    /// * `new_value` - New value to append
    ///
    /// # Returns
    /// Tuple of (concatenated_key, concatenated_value)
    pub fn get_with_new(
        &self,
        layer: usize,
        new_key: &DenseTensor,
        new_value: &DenseTensor,
    ) -> Option<(DenseTensor, DenseTensor)> {
        let (cached_key, cached_value) = self.get(layer, None)?;

        // Concatenate along sequence dimension
        let key = self.concat_along_seq(&cached_key, new_key);
        let value = self.concat_along_seq(&cached_value, new_value);

        Some((key, value))
    }

    /// Concatenate two tensors along sequence dimension
    fn concat_along_seq(&self, cached: &DenseTensor, new: &DenseTensor) -> DenseTensor {
        let cached_len = cached.shape()[0];
        let num_kv_heads = cached.shape()[1];
        let head_dim = cached.shape()[2];

        let new_len = new.shape()[0];
        let total_len = cached_len + new_len;

        let mut data = Vec::with_capacity(total_len * num_kv_heads * head_dim);

        // Copy cached data
        data.extend_from_slice(cached.data());

        // Copy new data
        data.extend_from_slice(new.data());

        DenseTensor::new(data, vec![total_len, num_kv_heads, head_dim])
    }
}

/// Paged KV Cache for vLLM-style memory management
#[derive(Debug, Clone)]
pub struct PagedKVCache {
    /// Block size (tokens per block)
    block_size: usize,
    /// Key blocks [num_blocks, block_size, num_kv_heads, head_dim]
    key_blocks: Vec<DenseTensor>,
    /// Value blocks
    value_blocks: Vec<DenseTensor>,
    /// Block table: logical block -> physical block
    block_table: Vec<usize>,
    /// Current sequence length
    current_len: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Number of layers
    #[allow(dead_code)]
    num_layers: usize,
    /// Hidden dimension
    #[allow(dead_code)]
    hidden_dim: usize,
    /// Number of KV heads
    num_kv_heads: usize,
}

impl PagedKVCache {
    /// Create a new paged KV cache
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers
    /// * `max_seq_len` - Maximum sequence length
    /// * `hidden_dim` - Hidden dimension
    /// * `num_kv_heads` - Number of KV heads
    /// * `block_size` - Tokens per block (typical: 16 or 32)
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        hidden_dim: usize,
        num_kv_heads: usize,
        block_size: usize,
    ) -> Self {
        let num_blocks = max_seq_len.div_ceil(block_size);
        let head_dim = hidden_dim / num_kv_heads;

        let key_blocks =
            vec![
                DenseTensor::zeros(vec![num_blocks, block_size, num_kv_heads, head_dim]);
                num_layers
            ];
        let value_blocks =
            vec![
                DenseTensor::zeros(vec![num_blocks, block_size, num_kv_heads, head_dim]);
                num_layers
            ];

        Self {
            block_size,
            key_blocks,
            value_blocks,
            block_table: Vec::new(),
            current_len: 0,
            max_seq_len,
            num_layers,
            hidden_dim,
            num_kv_heads,
        }
    }

    /// Allocate a new block
    fn allocate_block(&mut self) -> Option<usize> {
        if self.block_table.len() * self.block_size >= self.max_seq_len {
            return None; // No more capacity
        }

        let block_id = self.block_table.len();
        self.block_table.push(block_id);
        Some(block_id)
    }

    /// Update cache with new key and value
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `key` - New key [1, num_kv_heads, head_dim]
    /// * `value` - New value [1, num_kv_heads, head_dim]
    pub fn append(&mut self, layer: usize, key: &DenseTensor, value: &DenseTensor) {
        if self.current_len >= self.max_seq_len {
            return;
        }

        // Check if we need a new block
        if self.current_len % self.block_size == 0 {
            self.allocate_block();
        }

        let block_id = self.block_table.len().saturating_sub(1);
        let block_offset = self.current_len % self.block_size;

        if let Some(key_block) = self.key_blocks.get_mut(layer) {
            Self::copy_to_block_static(
                key_block,
                block_id,
                block_offset,
                key,
                self.block_size,
                self.num_kv_heads,
            );
        }

        if let Some(value_block) = self.value_blocks.get_mut(layer) {
            Self::copy_to_block_static(
                value_block,
                block_id,
                block_offset,
                value,
                self.block_size,
                self.num_kv_heads,
            );
        }

        self.current_len += 1;
    }

    /// Copy tensor to block at specified offset (static method to avoid borrow issues)
    #[inline]
    fn copy_to_block_static(
        block: &mut DenseTensor,
        block_id: usize,
        offset: usize,
        tensor: &DenseTensor,
        block_size: usize,
        num_kv_heads: usize,
    ) {
        let head_dim = tensor.shape()[2];

        for h in 0..num_kv_heads {
            let src_offset = h * head_dim;
            let dst_offset = ((block_id * block_size + offset) * num_kv_heads + h) * head_dim;

            let src_slice = &tensor.data()[src_offset..src_offset + head_dim];
            let block_data = block.data_mut();
            block_data[dst_offset..dst_offset + head_dim].copy_from_slice(src_slice);
        }
    }

    /// Get current sequence length
    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Get number of allocated blocks
    pub fn num_blocks(&self) -> usize {
        self.block_table.len()
    }

    /// Get block table
    pub fn block_table(&self) -> &[usize] {
        &self.block_table
    }

    /// Reset cache
    pub fn reset(&mut self) {
        self.current_len = 0;
        self.block_table.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_creation() {
        let cache = KVCache::new(2, 512, 4096, 8);

        assert_eq!(cache.num_layers(), 2);
        assert_eq!(cache.max_seq_len(), 512);
        assert_eq!(cache.hidden_dim(), 4096);
        assert_eq!(cache.num_kv_heads(), 8);
        assert_eq!(cache.current_len(), 0);
    }

    #[test]
    fn test_kv_cache_update() {
        let mut cache = KVCache::new(2, 512, 4096, 8);

        let key = DenseTensor::ones(vec![1, 8, 512]);
        let value = DenseTensor::ones(vec![1, 8, 512]);

        cache.update(0, &key, &value, 0);

        assert_eq!(cache.current_len(), 1);

        let (cached_key, cached_value) = cache.get(0, Some(1)).unwrap();
        assert_eq!(cached_key.shape(), &[1, 8, 512]);
        assert_eq!(cached_value.shape(), &[1, 8, 512]);
    }

    #[test]
    fn test_kv_cache_append() {
        let mut cache = KVCache::new(2, 512, 4096, 8);

        for i in 0..5 {
            let key = DenseTensor::full(&[1, 8, 512], i as f64);
            let value = DenseTensor::full(&[1, 8, 512], i as f64 * 2.0);
            cache.append(0, &key, &value);
        }

        assert_eq!(cache.current_len(), 5);
        assert!(!cache.is_full());
        assert_eq!(cache.remaining_capacity(), 512 - 5);
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = KVCache::new(2, 512, 4096, 8);

        let key = DenseTensor::ones(vec![1, 8, 512]);
        let value = DenseTensor::ones(vec![1, 8, 512]);
        cache.update(0, &key, &value, 0);

        assert_eq!(cache.current_len(), 1);

        cache.reset();

        assert_eq!(cache.current_len(), 0);
    }

    #[test]
    fn test_paged_kv_cache() {
        let mut cache = PagedKVCache::new(2, 128, 4096, 8, 16);

        for i in 0..20 {
            let key = DenseTensor::full(&[1, 8, 512], i as f64);
            let value = DenseTensor::full(&[1, 8, 512], i as f64);
            cache.append(0, &key, &value);
        }

        assert_eq!(cache.current_len(), 20);
        assert_eq!(cache.num_blocks(), 2); // 20 tokens / 16 block_size = 2 blocks
    }

    #[test]
    fn test_gqa_kv_cache() {
        // LLaMA-3 8B: 32 Q heads, 8 KV heads
        let mut cache = KVCache::new(32, 8192, 4096, 8);

        let key = DenseTensor::ones(vec![1, 8, 512]);
        let value = DenseTensor::ones(vec![1, 8, 512]);

        for layer in 0..32 {
            cache.update(layer, &key, &value, 0);
        }

        assert_eq!(cache.num_layers(), 32);
        assert_eq!(cache.num_kv_heads(), 8);
    }
}
