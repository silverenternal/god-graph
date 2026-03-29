//! Tensor Ring Compression for LLM weights
//!
//! This module implements tensor ring compression for LLM weight matrices,
//! achieving parameter reduction while maintaining accuracy.
//!
//! ## Mathematical Foundation
//!
//! Tensor Ring decomposition represents a high-dimensional tensor as a ring of
//! 3D core tensors:
//!
//! W(i₁,...,iₙ) = Σ Tr[G₁(i₁) × G₂(i₂) × ... × Gₙ(iₙ)]
//!
//! where Gₖ(iₖ) ∈ R^(rₖ₋₁×rₖ) and rₖ are the TR ranks controlling compression.
//!
//! ## Compression Ratio
//!
//! For a weight matrix W ∈ R^(m×n) with TR ranks [r₀, r₁, r₂]:
//! - Original parameters: m × n
//! - TR parameters: r₀×m×r₁ + r₁×n×r₂
//! - Compression ratio: (m × n) / (r₀×m×r₁ + r₁×n×r₂)
//!
//! ## Example
//!
//! ```no_run
//! use god_gragh::transformer::optimization::{TensorRingCompressor, CompressionConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = CompressionConfig::new()
//!     .with_target_ranks(vec![32, 64])
//!     .with_layers(vec!["qkv".to_string(), "mlp".to_string()]);
//!
//! let compressor = TensorRingCompressor::new(config);
//!
//! // Compress a weight matrix
//! // let compressed_graph = compressor.compress_graph(&graph)?;
//!
//! // Query compression ratio
//! // println!("Compression ratio: {:.2}x", compressor.compression_ratio());
//! # Ok(())
//! # }
//! ```

use crate::errors::{GraphError, GraphResult};
use crate::graph::Graph;
use crate::tensor::decomposition::tensor_ring::TensorRing;
use crate::tensor::DenseTensor;
use crate::tensor::TensorBase;
use crate::transformer::optimization::switch::{OperatorType, WeightTensor};
use std::cell::RefCell;
use std::collections::HashMap;

/// Configuration for tensor ring compression
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Target TR ranks for each dimension
    pub target_ranks: Vec<usize>,
    /// Layers to compress (by name pattern)
    pub layers: Vec<String>,
    /// Minimum rank (lower bound)
    pub min_rank: usize,
    /// Maximum rank (upper bound)
    pub max_rank: usize,
    /// Target compression ratio (adaptive rank selection)
    pub target_ratio: Option<f64>,
}

impl CompressionConfig {
    /// Create a new compression config with default values
    pub fn new() -> Self {
        Self {
            target_ranks: vec![64],
            layers: vec![".*".to_string()], // Match all layers by default
            min_rank: 16,
            max_rank: 256,
            target_ratio: None,
        }
    }

    /// Set target ranks
    pub fn with_target_ranks(mut self, ranks: Vec<usize>) -> Self {
        self.target_ranks = ranks;
        self
    }

    /// Set layers to compress
    pub fn with_layers(mut self, layers: Vec<String>) -> Self {
        self.layers = layers;
        self
    }

    /// Set minimum rank
    pub fn with_min_rank(mut self, rank: usize) -> Self {
        self.min_rank = rank;
        self
    }

    /// Set maximum rank
    pub fn with_max_rank(mut self, rank: usize) -> Self {
        self.max_rank = rank;
        self
    }

    /// Set target compression ratio
    pub fn with_target_ratio(mut self, ratio: f64) -> Self {
        self.target_ratio = Some(ratio.clamp(1.5, 10.0));
        self
    }

    /// Check if a layer name matches the compression pattern
    pub fn matches_layer(&self, layer_name: &str) -> bool {
        self.layers.iter().any(|pattern| {
            if pattern == ".*" {
                true
            } else {
                layer_name.contains(pattern)
            }
        })
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Tensor Ring compressor for LLM weights
pub struct TensorRingCompressor {
    config: CompressionConfig,
    compressed_tensors: RefCell<HashMap<String, TensorRing>>,
    original_params: RefCell<usize>,
    compressed_params: RefCell<usize>,
}

impl TensorRingCompressor {
    /// Create a new tensor ring compressor
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            compressed_tensors: RefCell::new(HashMap::new()),
            original_params: RefCell::new(0),
            compressed_params: RefCell::new(0),
        }
    }

    /// Get the compression configuration
    pub fn config(&self) -> &CompressionConfig {
        &self.config
    }

    /// Compress a single tensor
    ///
    /// # Arguments
    ///
    /// * `tensor` - Weight tensor to compress
    ///
    /// # Returns
    ///
    /// TensorRing decomposition of the input tensor
    pub fn decompose(&self, tensor: &DenseTensor) -> Result<TensorRing, crate::tensor::TensorError> {
        use crate::tensor::decomposition::tensor_ring::compress_tensor_ring;

        let shape = tensor.shape();

        // Select rank based on tensor dimensions and config
        let rank = self.select_rank(shape);

        compress_tensor_ring(tensor, rank)
    }

    /// Reconstruct a tensor from its Tensor Ring decomposition
    ///
    /// # Arguments
    ///
    /// * `ring` - TensorRing decomposition to reconstruct
    ///
    /// # Returns
    ///
    /// Reconstructed dense tensor
    pub fn reconstruct(&self, ring: &TensorRing) -> Result<DenseTensor, crate::tensor::TensorError> {
        ring.reconstruct()
    }

    /// Compress all weights in a graph
    ///
    /// # Arguments
    ///
    /// * `graph` - Graph containing weights to compress
    ///
    /// # Returns
    ///
    /// Compression report with statistics
    pub fn compress_graph(
        &self,
        graph: &Graph<OperatorType, WeightTensor>,
    ) -> GraphResult<CompressionReport> {
        use crate::graph::traits::GraphQuery;
        use crate::tensor::decomposition::tensor_ring::compress_tensor_ring;

        let mut total_original_params = 0usize;
        let mut total_compressed_params = 0usize;
        let mut layer_reports = Vec::new();
        let mut compressed_map = HashMap::new();

        // Calculate compression statistics for each edge weight
        for edge_ref in graph.edges() {
            let weight = edge_ref.data();

            // Decompose into Tensor Ring format
            let weight_tensor = DenseTensor::new(
                weight.data.to_vec(),
                weight.shape.to_vec(),
            );

            // Select rank based on config
            let rank = self.select_rank(weight_tensor.shape());

            let ring = compress_tensor_ring(&weight_tensor, rank)
                .map_err(|e| GraphError::InvalidFormat(e.to_string()))?;

            // Count parameters
            let original_params = weight_tensor.shape().iter().product::<usize>();
            let compressed_params = ring.cores.iter()
                .map(|c| c.shape().iter().product::<usize>())
                .sum::<usize>();

            total_original_params += original_params;
            total_compressed_params += compressed_params;

            // Store compressed tensor
            compressed_map.insert(weight.name.clone(), ring.clone());

            layer_reports.push(LayerCompressionReport {
                layer_name: weight.name.clone(),
                original_params,
                compressed_params,
                compression_ratio: original_params as f64 / compressed_params as f64,
                ranks: ring.ranks.clone(),
            });
        }

        let overall_ratio = if total_compressed_params > 0 {
            total_original_params as f64 / total_compressed_params as f64
        } else {
            1.0
        };

        // Store compressed tensors and statistics
        *self.compressed_tensors.borrow_mut() = compressed_map;
        *self.original_params.borrow_mut() = total_original_params;
        *self.compressed_params.borrow_mut() = total_compressed_params;

        Ok(CompressionReport {
            original_params: total_original_params,
            compressed_params: total_compressed_params,
            compression_ratio: overall_ratio,
            layers: layer_reports,
        })
    }

    /// Get the achieved compression ratio
    pub fn compression_ratio(&self) -> f64 {
        let compressed = *self.compressed_params.borrow();
        if compressed == 0 {
            return 1.0;
        }
        let original = *self.original_params.borrow();
        original as f64 / compressed as f64
    }

    /// Get the number of original parameters
    pub fn original_params(&self) -> usize {
        *self.original_params.borrow()
    }

    /// Get the number of compressed parameters
    pub fn compressed_params(&self) -> usize {
        *self.compressed_params.borrow()
    }

    /// Get compressed tensors
    pub fn compressed_tensors(&self) -> std::cell::Ref<HashMap<String, TensorRing>> {
        self.compressed_tensors.borrow()
    }

    /// Select optimal rank for a tensor based on config
    fn select_rank(&self, shape: &[usize]) -> usize {
        // Simple heuristic: use config rank or adapt based on dimensions
        let min_dim = shape.iter().min().copied().unwrap_or(1024);
        
        let base_rank = self.config.target_ranks.first().copied().unwrap_or(64);
        
        // Clamp to min/max bounds
        base_rank
            .max(self.config.min_rank)
            .min(self.config.max_rank)
            .min(min_dim / 2)
    }

    /// Compress a weight tensor and store the result
    fn compress_weight(
        &self,
        name: &str,
        tensor: &DenseTensor,
    ) -> Result<TensorRing, crate::tensor::TensorError> {
        use crate::tensor::decomposition::tensor_ring::compress_tensor_ring;

        let rank = self.select_rank(tensor.shape());
        let ring = compress_tensor_ring(tensor, rank)?;

        // Update parameter counts
        let original = tensor.shape().iter().product::<usize>();
        let compressed = ring
            .cores
            .iter()
            .map(|c| c.shape().iter().product::<usize>())
            .sum::<usize>();

        *self.original_params.borrow_mut() += original;
        *self.compressed_params.borrow_mut() += compressed;

        // Store compressed tensor
        self.compressed_tensors.borrow_mut().insert(name.to_string(), ring.clone());

        Ok(ring)
    }
}

impl Default for TensorRingCompressor {
    fn default() -> Self {
        Self::new(CompressionConfig::new())
    }
}

/// Adaptive rank selection based on singular value decay
///
/// # Arguments
///
/// * `tensor` - Weight tensor to analyze
/// * `energy_threshold` - Fraction of energy to preserve (e.g., 0.99)
///
/// # Returns
///
/// Recommended rank for compression
pub fn adaptive_rank_selection(
    tensor: &DenseTensor,
    energy_threshold: f64,
) -> Result<usize, crate::tensor::TensorError> {
    use crate::tensor::decomposition::svd_decompose;

    let shape = tensor.shape();
    let min_dim = shape.iter().min().copied().unwrap_or(1);
    
    // Compute SVD
    let (_, s, _) = svd_decompose(tensor, Some(min_dim))?;
    
    // Calculate cumulative energy
    let s_data = s.data();
    let total_energy: f64 = s_data.iter().map(|x| x * x).sum();
    let threshold = total_energy * energy_threshold;
    
    let mut cumulative_energy = 0.0;
    for (i, &sigma) in s_data.iter().enumerate() {
        cumulative_energy += sigma * sigma;
        if cumulative_energy >= threshold {
            return Ok(i + 1);
        }
    }
    
    Ok(min_dim)
}

/// Mixed precision compression strategy
///
/// Compresses different layers with different ranks based on importance.
///
/// # Arguments
///
/// * `tensors` - Map of layer names to weight tensors
/// * `base_rank` - Base compression rank
/// * `importance_map` - Optional importance scores for each layer
///
/// # Returns
///
/// Map of layer names to TensorRing decompositions
pub fn mixed_precision_compress(
    tensors: &HashMap<String, DenseTensor>,
    base_rank: usize,
    importance_map: Option<&HashMap<String, f64>>,
) -> Result<HashMap<String, TensorRing>, crate::tensor::TensorError> {
    use crate::tensor::decomposition::tensor_ring::compress_tensor_ring;

    let mut results = HashMap::new();
    
    for (name, tensor) in tensors {
        // Adjust rank based on importance
        let importance = importance_map
            .and_then(|m| m.get(name))
            .copied()
            .unwrap_or(1.0);
        
        // Higher importance → higher rank
        let rank = (base_rank as f64 * importance).ceil() as usize;
        
        let ring = compress_tensor_ring(tensor, rank)?;
        results.insert(name.clone(), ring);
    }
    
    Ok(results)
}

/// Compression report for a single layer
#[derive(Debug, Clone)]
pub struct LayerCompressionReport {
    /// Layer name
    pub layer_name: String,
    /// Original parameter count
    pub original_params: usize,
    /// Compressed parameter count
    pub compressed_params: usize,
    /// Compression ratio (original / compressed)
    pub compression_ratio: f64,
    /// TR ranks used
    pub ranks: Vec<usize>,
}

/// Overall compression report
#[derive(Debug, Clone)]
pub struct CompressionReport {
    /// Total original parameters
    pub original_params: usize,
    /// Total compressed parameters
    pub compressed_params: usize,
    /// Overall compression ratio
    pub compression_ratio: f64,
    /// Per-layer reports
    pub layers: Vec<LayerCompressionReport>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::traits::TensorOps;

    #[test]
    fn test_compression_config() {
        let config = CompressionConfig::new()
            .with_target_ranks(vec![32, 64])
            .with_layers(vec!["qkv".to_string(), "mlp".to_string()])
            .with_min_rank(16)
            .with_max_rank(128);

        assert!(config.matches_layer("model.layers.0.qkv.weight"));
        assert!(config.matches_layer("model.layers.0.mlp.gate_proj"));
        assert!(!config.matches_layer("model.norm.weight"));
    }

    #[test]
    fn test_tensor_ring_compressor() {
        // Use a smaller rank to achieve actual compression
        // For 64x64 matrix, we need rank < sqrt(64*64) = 64 for compression
        // TR params = 2 * rank * 64 * rank = 128 * rank^2
        // Original = 4096
        // For compression: 128 * rank^2 < 4096 => rank^2 < 32 => rank < 6
        let config = CompressionConfig::new()
            .with_target_ranks(vec![4])
            .with_min_rank(2)
            .with_max_rank(8);
        let compressor = TensorRingCompressor::new(config);

        let tensor = DenseTensor::from_vec(
            vec![1.0; 64 * 64],
            vec![64, 64],
        );

        let ring = compressor.decompose(&tensor).unwrap();
        
        eprintln!("Original shape: {:?}", ring.original_shape);
        eprintln!("Ranks: {:?}", ring.ranks);
        eprintln!("Core shapes: {:?}", ring.cores.iter().map(|c| c.shape()).collect::<Vec<_>>());
        eprintln!("Compression ratio: {}", ring.compression_ratio());
        
        assert!(ring.compression_ratio() > 1.0, "Compression ratio should be > 1.0, got {}", ring.compression_ratio());
    }

    #[test]
    fn test_adaptive_rank_selection() {
        // Create a low-rank tensor (rank 5)
        let u = DenseTensor::from_vec(
            (0..100 * 5).map(|i| (i % 10) as f64 / 10.0).collect(),
            vec![100, 5],
        );
        let v = DenseTensor::from_vec(
            (0..5 * 50).map(|i| (i % 7) as f64 / 10.0).collect(),
            vec![5, 50],
        );
        let tensor = u.matmul(&v);

        let rank = adaptive_rank_selection(&tensor, 0.99).unwrap();
        assert!(rank <= 10); // Should detect low intrinsic rank
    }
}
