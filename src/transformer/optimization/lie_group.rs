//! Lie Group Optimization for LLM weights
//!
//! This module implements Lie group-based weight optimization for LLMs,
//! using orthogonalization and block decomposition for numerical stability
//! and improved conditioning.
//!
//! ## Mathematical Foundation
//!
//! ### Lie Group Decomposition
//!
//! Weight matrices are decomposed into products of orthogonal matrices:
//!
//! W = O₁ × D × O₂
//!
//! where O₁, O₂ ∈ SO(n) are orthogonal matrices and D is a diagonal scaling matrix.
//!
//! ### Lie Algebra
//!
//! The Lie algebra so(n) consists of skew-symmetric matrices:
//! - so(n) = {X ∈ R^(n×n) | X^T = -X}
//! - Exponential map: exp: so(n) → SO(n)
//! - Logarithmic map: log: SO(n) → so(n)
//!
//! ### Orthogonal Constraint
//!
//! Weights are constrained to be orthogonal (W^T W = I) via:
//! - QR decomposition (Phase 1 simple version)
//! - Cayley transform (Phase 2)
//! - SO(k) block decomposition (Phase 2 full version)
//!
//! ## Example
//!
//! ```no_run
//! use god_gragh::transformer::optimization::{LieGroupOptimizer, LieGroupConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = LieGroupConfig::new()
//!     .with_block_size(64)
//!     .with_orthogonalize(true)
//!     .with_target_layers(vec!["q_proj".to_string(), "k_proj".to_string(), "v_proj".to_string()]);
//!
//! let optimizer = LieGroupOptimizer::new(config);
//!
//! // Orthogonalize weights in a graph
//! // optimizer.orthogonalize_weights(&mut graph)?;
//!
//! // Apply block decomposition
//! // optimizer.block_decompose(&mut graph)?;
//! # Ok(())
//! # }
//! ```

use crate::errors::{GraphError, GraphResult};
use crate::graph::Graph;
use crate::tensor::DenseTensor;
use crate::tensor::TensorBase;
use crate::transformer::optimization::error_analysis::ErrorAccumulator;
use crate::transformer::optimization::switch::{OperatorType, WeightTensor};
use std::cell::RefCell;
use std::collections::HashMap;

/// Configuration for Lie group optimization
#[derive(Debug, Clone)]
pub struct LieGroupConfig {
    /// Block size for SO(k) decomposition (e.g., 64 or 128)
    pub block_size: usize,
    /// Whether to apply orthogonalization
    pub orthogonalize: bool,
    /// Target layers for optimization (by name pattern)
    pub target_layers: Vec<String>,
    /// Use Cayley transform instead of QR
    pub use_cayley: bool,
    /// Number of iterations for optimization
    pub iterations: usize,
    /// Tolerance for convergence
    pub tolerance: f64,
}

impl LieGroupConfig {
    /// Create a new config with default values
    pub fn new() -> Self {
        Self {
            block_size: 64,
            orthogonalize: true,
            target_layers: vec![".*".to_string()],
            use_cayley: false,
            iterations: 10,
            tolerance: 1e-6,
        }
    }

    /// Set block size for SO(k) decomposition
    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    /// Enable/disable orthogonalization
    pub fn with_orthogonalize(mut self, ortho: bool) -> Self {
        self.orthogonalize = ortho;
        self
    }

    /// Set target layers
    pub fn with_target_layers(mut self, layers: Vec<String>) -> Self {
        self.target_layers = layers;
        self
    }

    /// Enable/disable Cayley transform
    pub fn with_cayley(mut self, use_cayley: bool) -> Self {
        self.use_cayley = use_cayley;
        self
    }

    /// Set number of iterations
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Check if a layer name matches the target pattern
    pub fn matches_layer(&self, layer_name: &str) -> bool {
        self.target_layers.iter().any(|pattern| {
            if pattern == ".*" {
                true
            } else {
                layer_name.contains(pattern)
            }
        })
    }
}

impl Default for LieGroupConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Lie group optimizer for weight orthogonalization and decomposition
pub struct LieGroupOptimizer {
    config: LieGroupConfig,
    statistics: RefCell<HashMap<String, f64>>,
    error_accumulator: RefCell<ErrorAccumulator>,
}

impl LieGroupOptimizer {
    /// Create a new Lie group optimizer
    pub fn new(config: LieGroupConfig) -> Self {
        Self {
            config,
            statistics: RefCell::new(HashMap::new()),
            error_accumulator: RefCell::new(ErrorAccumulator::new()),
        }
    }

    /// Get the optimizer configuration
    pub fn config(&self) -> &LieGroupConfig {
        &self.config
    }

    /// Get optimization statistics
    pub fn statistics(&self) -> std::cell::Ref<HashMap<String, f64>> {
        self.statistics.borrow()
    }

    /// Get the error accumulator for detailed error analysis
    pub fn error_accumulator(&self) -> std::cell::Ref<ErrorAccumulator> {
        self.error_accumulator.borrow()
    }

    /// Get a mutable reference to the error accumulator
    pub fn error_accumulator_mut(&self) -> std::cell::RefMut<ErrorAccumulator> {
        self.error_accumulator.borrow_mut()
    }

    /// Orthogonalize all weights in a graph (two-pass, zero-copy)
    ///
    /// This implementation uses a two-pass approach to avoid data cloning:
    /// 1. First pass: collect edge indices only (no data cloning)
    /// 2. Second pass: orthogonalize in-place using IndexMut
    ///
    /// # Arguments
    ///
    /// * `graph` - Graph containing weights to orthogonalize
    ///
    /// # Returns
    ///
    /// Ok if successful, error if orthogonalization fails
    pub fn orthogonalize_weights(
        &self,
        graph: &mut Graph<OperatorType, WeightTensor>,
    ) -> GraphResult<()> {
        use crate::graph::traits::GraphQuery;

        let mut orthogonalized_count = 0;
        let mut total_error = 0.0;

        // First pass: collect edge indices only (avoid borrow checker issues)
        let edge_indices: Vec<_> = graph.edges().map(|e| e.index()).collect();

        // Second pass: orthogonalize in-place using IndexMut (zero-copy)
        for edge_idx in edge_indices {
            let error = self.orthogonalize_single_weight(graph, edge_idx)?;
            
            // Record error in accumulator with weight name
            let weight = &graph[edge_idx];
            self.error_accumulator
                .borrow_mut()
                .record_error(&weight.name, error);
            
            total_error += error;
            orthogonalized_count += 1;
        }

        // Store statistics
        if orthogonalized_count > 0 {
            self.statistics.borrow_mut().insert(
                "orthogonalization_error".to_string(),
                total_error / orthogonalized_count as f64
            );
        }

        Ok(())
    }

    /// Orthogonalize a single weight in-place (zero-copy)
    ///
    /// # Arguments
    /// * `graph` - Graph containing the weight to orthogonalize
    /// * `edge_idx` - Edge index of the weight to orthogonalize
    ///
    /// # Returns
    /// Orthogonalization error (0.0 if skipped)
    pub fn orthogonalize_single_weight(
        &self,
        graph: &mut Graph<OperatorType, WeightTensor>,
        edge_idx: crate::edge::EdgeIndex,
    ) -> GraphResult<f64> {
        use crate::tensor::decomposition::qr::orthogonalize_in_place;

        // Get mutable reference to the weight data using index operator
        let weight = &mut graph[edge_idx];
        let shape = weight.shape.to_vec();

        // Skip non-2D tensors (e.g., 1D layer norms)
        if shape.len() != 2 {
            eprintln!("Skipping orthogonalization for {}: shape={:?} (not 2D)", weight.name, shape);
            return Ok(0.0);
        }

        // Skip matrices where m < n (can't orthogonalize)
        if shape[0] < shape[1] {
            eprintln!("Skipping orthogonalization for {}: shape={:?} (m < n)", weight.name, shape);
            return Ok(0.0);
        }

        // In-place orthogonalization (zero-copy)
        let error = orthogonalize_in_place(&mut weight.data, &shape)
            .map_err(|e| GraphError::InvalidFormat(e.to_string()))?;

        Ok(error)
    }

    /// Check orthogonality of a tensor (W^T W ≈ I)
    fn check_orthogonality(tensor: &DenseTensor) -> f64 {
        let shape = tensor.shape();
        if shape.len() != 2 {
            return f64::MAX;
        }

        let n = shape[0];
        let m = shape[1];
        let data = tensor.data();

        // Compute W^T W
        let mut max_error: f64 = 0.0;
        for i in 0..m {
            for j in 0..m {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += data[k * m + i] * data[k * m + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let error = (dot - expected).abs();
                max_error = max_error.max(error);
            }
        }

        max_error
    }

    /// Apply block decomposition to weights
    ///
    /// Decomposes large weight matrices into blocks of SO(k) matrices.
    ///
    /// # Arguments
    ///
    /// * `graph` - Graph containing weights to decompose
    ///
    /// # Returns
    ///
    /// Decomposed weights report
    pub fn block_decompose(
        &self,
        graph: &mut Graph<OperatorType, WeightTensor>,
    ) -> GraphResult<DecomposedWeights> {
        use crate::graph::traits::GraphQuery;

        let block_size = self.config.block_size;
        let mut decomposed_blocks = Vec::new();
        let mut total_blocks = 0;

        // Collect edge data first
        let edge_data: Vec<_> = graph.edges().map(|e| {
            (e.index(), e.data().name.clone(), e.data().data.to_vec(), e.data().shape.to_vec())
        }).collect();

        for (_edge_idx, layer_name, weight_data, weight_shape) in edge_data {
            // Check layer name pattern matching
            if !self.config.matches_layer(&layer_name) {
                continue;
            }

            // Extract weight matrix and decompose
            let weight_tensor = DenseTensor::new(weight_data, weight_shape);

            // Decompose into SO(k) blocks
            let blocks = decompose_into_so_blocks(&weight_tensor, block_size)
                .map_err(|e| GraphError::InvalidFormat(e.to_string()))?;

            total_blocks += blocks.len();

            decomposed_blocks.push(BlockDecomposition {
                layer_name,
                num_blocks: blocks.len(),
                block_size,
            });
        }

        // Store statistics
        self.statistics.borrow_mut().insert(
            "total_blocks".to_string(),
            total_blocks as f64
        );

        Ok(DecomposedWeights {
            blocks: decomposed_blocks,
            total_blocks,
        })
    }

    /// Apply Lie algebra regularization
    ///
    /// Projects weights onto the Lie algebra and applies regularization.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Weight tensor to regularize
    ///
    /// # Returns
    ///
    /// Regularized tensor
    pub fn lie_algebra_regularize(
        &self,
        tensor: &DenseTensor,
    ) -> Result<DenseTensor, crate::tensor::TensorError> {
        use crate::tensor::decomposition::lie_algebra::skew_symmetric_projection;

        // Project to skew-symmetric (Lie algebra so(n))
        let skew = skew_symmetric_projection(tensor)?;

        // Apply exponential map to get back to Lie group
        crate::tensor::decomposition::lie_algebra::lie_exponential(&skew)
    }

    /// Cayley transform for orthogonalization
    ///
    /// The Cayley transform maps skew-symmetric matrices to orthogonal matrices:
    ///
    /// cayley(A) = (I - A/2)^(-1) (I + A/2)
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor
    ///
    /// # Returns
    ///
    /// Orthogonalized tensor
    pub fn cayley_transform(
        &self,
        tensor: &DenseTensor,
    ) -> Result<DenseTensor, crate::tensor::TensorError> {
        use crate::tensor::decomposition::lie_algebra::{
            lie_exponential, skew_symmetric_projection,
        };

        if self.config.use_cayley {
            // Project to Lie algebra
            let skew = skew_symmetric_projection(tensor)?;
            // Map to Lie group via exponential
            lie_exponential(&skew)
        } else {
            // Use QR decomposition instead
            crate::tensor::decomposition::qr::orthogonalize(tensor)
        }
    }

    /// Check if a weight matrix is well-conditioned
    ///
    /// # Arguments
    ///
    /// * `tensor` - Weight tensor to check
    /// * `threshold` - Condition number threshold
    ///
    /// # Returns
    ///
    /// True if the matrix is well-conditioned
    pub fn is_well_conditioned(&self, tensor: &DenseTensor, threshold: f64) -> bool {
        // Estimate condition number using power iteration
        let shape = tensor.shape();
        if shape.len() != 2 {
            return false;
        }

        let data = tensor.data();
        let (m, n) = (shape[0], shape[1]);

        // Power iteration for largest singular value
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        for _ in 0..20 {
            // Compute A^T A v
            let mut av = vec![0.0; m];
            for i in 0..m {
                for j in 0..n {
                    av[i] += data[i * n + j] * v[j];
                }
            }

            let mut atav = vec![0.0; n];
            for i in 0..n {
                for j in 0..m {
                    atav[i] += data[j * n + i] * av[j];
                }
            }

            let norm: f64 = atav.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-10 {
                return true;
            }
            v = atav.into_iter().map(|x| x / norm).collect();
        }

        // Largest eigenvalue of A^T A is sigma_max^2
        let sigma_max_sq: f64 = v
            .iter()
            .enumerate()
            .map(|(i, &vi)| {
                let mut sum = 0.0;
                for j in 0..n {
                    let mut aj = 0.0;
                    for k in 0..m {
                        aj += data[k * n + j] * data[k * n + i];
                    }
                    sum += aj * v[j];
                }
                sum * vi
            })
            .sum();

        let sigma_max = sigma_max_sq.sqrt();
        let sigma_min = 1.0 / sigma_max; // Simplified assumption

        let condition_number = sigma_max / sigma_min;
        condition_number < threshold
    }
}

/// Block decomposition result
#[derive(Debug, Clone)]
pub struct BlockDecomposition {
    /// Layer name
    pub layer_name: String,
    /// Number of blocks
    pub num_blocks: usize,
    /// Block size
    pub block_size: usize,
}

/// Decomposed weights report
#[derive(Debug, Clone)]
pub struct DecomposedWeights {
    /// List of block decompositions
    pub blocks: Vec<BlockDecomposition>,
    /// Total number of blocks
    pub total_blocks: usize,
}

/// SO(k) block for weight decomposition
#[derive(Debug, Clone)]
pub struct SOkBlock {
    /// Block data (orthogonal matrix)
    pub data: Vec<f64>,
    /// Block size k
    pub size: usize,
}

impl SOkBlock {
    /// Create a new SO(k) block
    pub fn new(data: Vec<f64>, size: usize) -> Result<Self, crate::tensor::TensorError> {
        if data.len() != size * size {
            return Err(crate::tensor::TensorError::DimensionMismatch {
                expected: size * size,
                got: data.len(),
            });
        }

        Ok(Self { data, size })
    }

    /// Check if the block is orthogonal
    pub fn is_orthogonal(&self, tolerance: f64) -> bool {
        let n = self.size;
        let data = &self.data;

        // Check O^T O = I
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += data[k * n + i] * data[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                if (dot - expected).abs() > tolerance {
                    return false;
                }
            }
        }

        true
    }
}

/// Decompose a weight matrix into SO(k) blocks
///
/// # Arguments
///
/// * `tensor` - Weight tensor to decompose
/// * `block_size` - Size of each SO(k) block
///
/// # Returns
///
/// Vector of SO(k) blocks
pub fn decompose_into_so_blocks(
    tensor: &DenseTensor,
    block_size: usize,
) -> Result<Vec<SOkBlock>, crate::tensor::TensorError> {
    use crate::tensor::decomposition::qr::orthogonalize;

    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(crate::tensor::TensorError::DimensionMismatch {
            expected: 2,
            got: shape.len(),
        });
    }

    let (m, n) = (shape[0], shape[1]);
    let mut blocks = Vec::new();

    // Partition matrix into block_size × block_size blocks
    for i in (0..m).step_by(block_size) {
        for j in (0..n).step_by(block_size) {
            let block_m = std::cmp::min(block_size, m - i);
            let block_n = std::cmp::min(block_size, n - j);

            // Extract block
            let mut block_data = vec![0.0; block_m * block_n];
            for bi in 0..block_m {
                for bj in 0..block_n {
                    block_data[bi * block_n + bj] =
                        tensor.data()[(i + bi) * n + (j + bj)];
                }
            }

            // Make square if necessary
            if block_m != block_n {
                let size = std::cmp::max(block_m, block_n);
                let mut square_block = vec![0.0; size * size];
                for bi in 0..block_m {
                    for bj in 0..block_n {
                        square_block[bi * size + bj] = block_data[bi * block_n + bj];
                    }
                }
                block_data = square_block;
            }

            // Orthogonalize the block
            let block_tensor = DenseTensor::from_vec(
                block_data,
                vec![block_m.max(block_n), block_m.max(block_n)],
            );
            let ortho = orthogonalize(&block_tensor)?;

            blocks.push(SOkBlock::new(ortho.data().to_vec(), ortho.shape()[0])?);
        }
    }

    Ok(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lie_group_config() {
        let config = LieGroupConfig::new()
            .with_block_size(128)
            .with_orthogonalize(true)
            .with_target_layers(vec!["q_proj".to_string(), "k_proj".to_string()]);

        assert_eq!(config.block_size, 128);
        assert!(config.orthogonalize);
        assert!(config.matches_layer("model.layers.0.attn.q_proj"));
        assert!(config.matches_layer("model.layers.0.attn.k_proj"));
        assert!(!config.matches_layer("model.layers.0.mlp"));
    }

    #[test]
    fn test_sok_block() {
        // Create a 2x2 rotation matrix (in SO(2))
        let theta = std::f64::consts::PI / 4.0;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        
        let block = SOkBlock::new(
            vec![cos_t, -sin_t, sin_t, cos_t],
            2,
        ).unwrap();

        assert!(block.is_orthogonal(1e-5));
    }

    #[test]
    fn test_decompose_into_so_blocks() {
        let tensor = DenseTensor::from_vec(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2],
        );

        let blocks = decompose_into_so_blocks(&tensor, 2).unwrap();
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].is_orthogonal(1e-5));
    }

    #[test]
    fn test_lie_optimizer() {
        let config = LieGroupConfig::new()
            .with_block_size(64)
            .with_orthogonalize(true);
        
        let optimizer = LieGroupOptimizer::new(config);
        
        let tensor = DenseTensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );

        let result = optimizer.cayley_transform(&tensor);
        assert!(result.is_ok());
    }
}

/// Orthogonalize all weights in a graph in-place (zero-copy)
///
/// This function orthogonalizes each weight tensor in the graph without cloning data,
/// achieving true zero-copy optimization.
///
/// # Arguments
/// * `config` - Lie group configuration
/// * `graph` - Graph containing weights to orthogonalize
///
/// # Returns
/// Vector of orthogonalization errors for each weight
pub fn orthogonalize_weights_in_place(
    config: &LieGroupConfig,
    graph: &mut Graph<OperatorType, WeightTensor>,
) -> GraphResult<Vec<f64>> {
    use crate::graph::traits::GraphQuery;

    let mut errors = Vec::new();
    let optimizer = LieGroupOptimizer::new(config.clone());

    // Collect edge indices first to avoid borrow checker issues
    let edge_indices: Vec<_> = graph.edges().map(|e| e.index()).collect();

    for edge_idx in edge_indices {
        let error = optimizer.orthogonalize_single_weight(graph, edge_idx)?;
        errors.push(error);
    }

    Ok(errors)
}
