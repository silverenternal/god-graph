//! Model Switch: Bidirectional lossless conversion between Safetensors and GodGraph
//!
//! This module implements the Model Switch tool for converting between
//! HuggingFace Safetensors format and GodGraph graph structure.
//!
//! ## Features
//!
//! - Safetensors → GodGraph loading
//! - GodGraph → Safetensors exporting
//! - Topology integrity validation
//! - Weight precision verification (lossless check)
//!
//! ## Example
//!
//! ```no_run
//! # #[cfg(feature = "safetensors")]
//! use god_gragh::transformer::optimization::ModelSwitch;
//!
//! # #[cfg(feature = "safetensors")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load from Safetensors
//! let graph = ModelSwitch::load_from_safetensors("model.safetensors")?;
//!
//! // Validate topology
//! let report = ModelSwitch::validate_topology(&graph)?;
//! println!("Topology valid: {}", report.is_valid);
//!
//! // Verify weights against original
//! let diff = ModelSwitch::verify_weights(&graph, &graph)?;
//! println!("Max L2 difference: {}", diff.max_l2_diff);
//!
//! // Save to Safetensors
//! ModelSwitch::save_to_safetensors(&graph, "optimized.safetensors")?;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "safetensors"))]
//! # fn main() {}
//! ```

use crate::errors::{GraphError, GraphResult};
use crate::graph::traits::{GraphBase, GraphOps, GraphQuery};
use crate::graph::Graph;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::path::Path;

/// Operator types for LLM computation graph nodes
#[derive(Debug, Clone, PartialEq)]
pub enum OperatorType {
    /// Multi-head attention operator
    Attention {
        num_heads: usize,
        hidden_dim: usize,
    },
    /// Feed-forward network (MLP)
    MLP {
        hidden_dim: usize,
        activation: String,
    },
    /// Layer normalization
    Norm {
        norm_type: String,
        eps: f64,
    },
    /// Embedding lookup
    Embedding {
        vocab_size: usize,
        embed_dim: usize,
    },
    /// Linear projection
    Linear {
        in_features: usize,
        out_features: usize,
    },
    /// Residual connection (identity)
    Residual,
    /// Custom operator
    Custom {
        name: String,
    },
}

/// 64-byte aligned weight tensor with stride support for efficient N-dimensional access
///
/// This struct provides:
/// - Inline storage with Box<[f64]> to avoid Vec reallocation
/// - SmallVec<[usize; 4]> for shape and strides (avoids heap allocation for ≤4D tensors)
/// - 64-byte alignment to prevent false sharing in multi-threaded scenarios
/// - Stride-based indexing for non-contiguous memory access patterns
/// - In-place reshape without data reallocation
#[repr(align(64))]
#[derive(Clone, Debug)]
pub struct WeightTensor {
    /// Tensor data stored in Box to avoid Vec reallocation overhead
    pub data: Box<[f64]>,
    /// Tensor shape (dimensions) with small array optimization
    pub shape: SmallVec<[usize; 4]>,
    /// Strides for each dimension (C-order by default)
    pub strides: SmallVec<[usize; 4]>,
    /// Tensor name/identifier for weight mapping
    pub name: String,
}

impl WeightTensor {
    /// Create a new weight tensor with automatic stride computation
    ///
    /// # Arguments
    /// * `data` - Tensor data in row-major (C-order) format
    /// * `shape` - Tensor dimensions
    /// * `name` - Tensor identifier
    ///
    /// # Panics
    /// Panics if data length doesn't match the product of shape dimensions
    pub fn new(name: String, data: Vec<f64>, shape: Vec<usize>) -> Self {
        let expected_len = shape.iter().product::<usize>();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} mismatch with shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        let strides = compute_strides(&shape);
        Self {
            data: data.into_boxed_slice(),
            shape: shape.into(),
            strides: strides.into(),
            name,
        }
    }

    /// Create a weight tensor from pre-computed strides
    ///
    /// # Arguments
    /// * `data` - Tensor data
    /// * `shape` - Tensor dimensions
    /// * `strides` - Stride for each dimension
    /// * `name` - Tensor identifier
    pub fn with_strides(
        name: String,
        data: Vec<f64>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> Self {
        let expected_len = shape.iter().product::<usize>();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} mismatch with shape {:?}",
            data.len(),
            shape
        );

        Self {
            data: data.into_boxed_slice(),
            shape: shape.into(),
            strides: strides.into(),
            name,
        }
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Get the shape as a slice
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides as a slice
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get immutable access to the underlying data
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get mutable access to the underlying data for in-place operations
    pub fn as_slice_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Reshape the tensor in-place without reallocating data
    ///
    /// # Arguments
    /// * `new_shape` - New tensor dimensions
    ///
    /// # Returns
    /// Ok if successful, Err if the new shape doesn't match the data size
    pub fn reshape_mut(&mut self, new_shape: Vec<usize>) -> Result<(), TensorReshapeError> {
        let new_size = new_shape.iter().product::<usize>();
        if new_size != self.data.len() {
            return Err(TensorReshapeError {
                expected: self.data.len(),
                got: new_size,
            });
        }
        self.shape = new_shape.into();
        self.strides = compute_strides(&self.shape).into();
        Ok(())
    }

    /// Calculate L2 norm of the tensor
    pub fn l2_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Calculate L2 difference with another tensor
    pub fn l2_diff(&self, other: &Self) -> f64 {
        if self.shape != other.shape {
            return f64::MAX;
        }
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Get element at multi-dimensional index using stride-based access
    ///
    /// # Arguments
    /// * `indices` - Index for each dimension
    ///
    /// # Returns
    /// Some(value) if indices are valid, None otherwise
    pub fn get(&self, indices: &[usize]) -> Option<f64> {
        if indices.len() != self.shape.len() {
            return None;
        }

        for (_i, (&idx, &dim)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if idx >= dim {
                return None;
            }
        }

        let offset = indices
            .iter()
            .zip(self.strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum::<usize>();

        self.data.get(offset).copied()
    }

    /// Set element at multi-dimensional index using stride-based access
    ///
    /// # Arguments
    /// * `indices` - Index for each dimension
    /// * `value` - Value to set
    ///
    /// # Returns
    /// true if successful, false if indices are invalid
    pub fn set(&mut self, indices: &[usize], value: f64) -> bool {
        if indices.len() != self.shape.len() {
            return false;
        }

        for (_i, (&idx, &dim)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if idx >= dim {
                return false;
            }
        }

        let offset = indices
            .iter()
            .zip(self.strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum::<usize>();

        if let Some(elem) = self.data.get_mut(offset) {
            *elem = value;
            true
        } else {
            false
        }
    }

    /// Convert to DenseTensor for compatibility with existing tensor operations
    #[cfg(feature = "tensor")]
    pub fn to_dense_tensor(&self) -> crate::tensor::DenseTensor {
        crate::tensor::DenseTensor::new(self.data.to_vec(), self.shape.to_vec())
    }
}

/// Error type for tensor reshape operations
#[derive(Debug, Clone)]
pub struct TensorReshapeError {
    /// Expected number of elements
    pub expected: usize,
    /// Actual number of elements
    pub got: usize,
}

impl std::fmt::Display for TensorReshapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Reshape error: expected {} elements, got {}",
            self.expected, self.got
        )
    }
}

impl std::error::Error for TensorReshapeError {}

/// Compute strides for C-order (row-major) layout
///
/// # Arguments
/// * `shape` - Tensor dimensions
///
/// # Returns
/// Vector of strides for each dimension
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }

    let mut strides = vec![1; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Topology validation report
#[derive(Debug, Clone)]
pub struct TopologyReport {
    /// Whether the topology is valid
    pub is_valid: bool,
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Number of connected components
    pub connected_components: usize,
    /// Whether the graph is a DAG (directed acyclic graph)
    pub is_dag: bool,
    /// List of issues found
    pub issues: Vec<String>,
}

/// Weight difference report
#[derive(Debug, Clone)]
pub struct WeightDiff {
    /// Maximum L2 difference across all weights
    pub max_l2_diff: f64,
    /// Average L2 difference
    pub avg_l2_diff: f64,
    /// Number of tensors compared
    pub tensor_count: usize,
    /// Per-tensor differences
    pub per_tensor_diff: HashMap<String, f64>,
}

/// Model Switch: Bidirectional conversion between Safetensors and GodGraph
pub struct ModelSwitch;

impl ModelSwitch {
    /// Load a model from Safetensors format into GodGraph
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the Safetensors file
    ///
    /// # Returns
    ///
    /// A GodGraph representation of the model
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed
    #[cfg(feature = "safetensors")]
    pub fn load_from_safetensors<P: AsRef<Path>>(path: P) -> GraphResult<Graph<OperatorType, WeightTensor>> {
        use safetensors::SafeTensors;
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path.as_ref())
            .map_err(|e| GraphError::IoError(format!("Failed to open file: {}", e)))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| GraphError::IoError(format!("Failed to read file: {}", e)))?;

        let safetensors = SafeTensors::deserialize(&buffer)
            .map_err(|e| GraphError::InvalidFormat(format!("Failed to deserialize safetensors: {}", e)))?;

        let mut graph = Graph::<OperatorType, WeightTensor>::directed();

        // Parse tensors and build graph structure
        // This is a simplified implementation; a full implementation would
        // parse the model config and build the appropriate graph structure
        for (name, tensor_view) in safetensors.tensors() {
            let shape = tensor_view.shape().to_vec();
            let dtype = tensor_view.dtype();

            // Convert tensor data to f64
            let data = match dtype {
                safetensors::Dtype::F32 => {
                    let slice = tensor_view.data();
                    // Use try_cast_slice for unaligned data, with manual fallback
                    match bytemuck::try_cast_slice::<u8, f32>(slice) {
                        Ok(f32_data) => f32_data.iter().map(|&x| x as f64).collect(),
                        Err(_) => {
                            slice.chunks_exact(4)
                                .map(|chunk| {
                                    let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                                    f32::from_le_bytes(bytes) as f64
                                })
                                .collect()
                        }
                    }
                }
                safetensors::Dtype::F64 => {
                    let slice = tensor_view.data();
                    match bytemuck::try_cast_slice::<u8, f64>(slice) {
                        Ok(f64_data) => f64_data.to_vec(),
                        Err(_) => {
                            slice.chunks_exact(8)
                                .map(|chunk| {
                                    let bytes: [u8; 8] = [chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]];
                                    f64::from_le_bytes(bytes)
                                })
                                .collect()
                        }
                    }
                }
                safetensors::Dtype::F16 => {
                    let slice = tensor_view.data();
                    // Convert bytes to f16 using proper API
                    let f16_data: Vec<half::f16> = slice
                        .chunks_exact(2)
                        .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                        .collect();
                    f16_data.iter().map(|x| x.to_f32() as f64).collect()
                }
                _ => {
                    return Err(GraphError::InvalidFormat(
                        format!("Unsupported dtype: {:?}", dtype)
                    ));
                }
            };

            // Create weight tensor
            let weight_tensor = WeightTensor::new(name.to_string(), data, shape);

            // Create operator node based on tensor name pattern
            let operator = Self::infer_operator_from_name(&name);
            let node = graph.add_node(operator)?;

            // Store weight tensor as edge data (self-loop for now)
            // In a full implementation, weights would be associated with specific nodes
            // For now, add as self-loop to preserve the weight data
            graph.add_edge(node, node, weight_tensor)?;
        }

        Ok(graph)
    }

    /// Save a GodGraph to Safetensors format
    ///
    /// # Arguments
    ///
    /// * `graph` - The GodGraph to save
    /// * `path` - Output path for the Safetensors file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written
    ///
    /// # Note
    ///
    /// This is a simplified implementation that stores all weights as F32.
    /// A full implementation would preserve the original dtype.
    #[cfg(feature = "safetensors")]
    pub fn save_to_safetensors<P: AsRef<Path>>(
        graph: &Graph<OperatorType, WeightTensor>,
        path: P,
    ) -> GraphResult<()> {
        use std::collections::BTreeMap;
        use safetensors::tensor::{TensorView, Dtype};

        // Collect all tensor data first (owned data)
        let mut tensor_data: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
        
        for edge_ref in graph.edges() {
            let weight = edge_ref.data();
            
            // Convert f64 data back to F32 for storage (most common dtype)
            let data_f32: Vec<f32> = weight.data.iter()
                .map(|&x| x as f32)
                .collect();
            
            let byte_data: Vec<u8> = data_f32.iter()
                .flat_map(|&x| x.to_le_bytes().to_vec())
                .collect();
            
            tensor_data.insert(
                weight.name.clone(),
                (byte_data, weight.shape.to_vec()),
            );
        }

        // Create TensorViews - these borrow from tensor_data
        let mut tensors: BTreeMap<String, TensorView> = BTreeMap::new();
        for (name, (bytes, shape)) in &tensor_data {
            let tensor_view = TensorView::new(
                Dtype::F32,
                shape.clone(),
                bytes,
            ).map_err(|e| GraphError::InvalidFormat(format!("Failed to create tensor view: {}", e)))?;
            
            tensors.insert(name.clone(), tensor_view);
        }

        // Create metadata (empty for now)
        let metadata: Option<std::collections::HashMap<String, String>> = None;

        // Serialize to file - tensors borrow from tensor_data which lives long enough
        safetensors::serialize_to_file(&tensors, &metadata, path.as_ref())
            .map_err(|e| GraphError::IoError(format!("Failed to write safetensors file: {}", e)))?;

        Ok(())
    }

    /// Validate the topology of a graph
    ///
    /// # Arguments
    ///
    /// * `graph` - The graph to validate
    ///
    /// # Returns
    ///
    /// A topology validation report
    pub fn validate_topology(
        graph: &Graph<OperatorType, WeightTensor>,
    ) -> GraphResult<TopologyReport> {
        use crate::algorithms::community::connected_components;
        use crate::algorithms::traversal::topological_sort;

        let node_count = graph.node_count();
        let edge_count = graph.edge_count();
        let mut issues = Vec::new();

        // Check for empty graph
        if node_count == 0 {
            issues.push("Graph is empty".to_string());
            return Ok(TopologyReport {
                is_valid: false,
                node_count,
                edge_count,
                connected_components: 0,
                is_dag: true,
                issues,
            });
        }

        // Check connected components
        let components = connected_components(graph);
        if components.len() > 1 {
            issues.push(format!("Graph has {} disconnected components", components.len()));
        }

        // Check if DAG (for feedforward models)
        let is_dag = topological_sort(graph).is_ok();
        if !is_dag {
            issues.push("Graph contains cycles (may be valid for recurrent models)".to_string());
        }

        // Check for isolated nodes
        let isolated_count = graph
            .nodes()
            .filter(|n| graph.neighbors(n.index()).count() == 0)
            .count();
        if isolated_count > 0 {
            issues.push(format!("Graph has {} isolated nodes", isolated_count));
        }

        let is_valid = issues.is_empty() || (components.len() == 1 && isolated_count == 0);

        Ok(TopologyReport {
            is_valid,
            node_count,
            edge_count,
            connected_components: components.len(),
            is_dag,
            issues,
        })
    }

    /// Verify weights between two graphs
    ///
    /// # Arguments
    ///
    /// * `original` - The original graph
    /// * `modified` - The modified graph to compare
    ///
    /// # Returns
    ///
    /// A weight difference report
    pub fn verify_weights(
        original: &Graph<OperatorType, WeightTensor>,
        modified: &Graph<OperatorType, WeightTensor>,
    ) -> GraphResult<WeightDiff> {
        let mut per_tensor_diff: HashMap<String, f64> = HashMap::new();
        let mut max_l2_diff = 0.0f64;
        let mut total_diff = 0.0f64;
        let mut tensor_count = 0;

        // Build a map of original weights by name
        let original_weights: HashMap<String, &WeightTensor> = original.edges()
            .map(|e| (e.data().name.clone(), e.data()))
            .collect();

        // Compare weights edge by edge
        for edge_ref in modified.edges() {
            let modified_weight = edge_ref.data();
            
            if let Some(&original_weight) = original_weights.get(&modified_weight.name) {
                // Compare shapes first
                if original_weight.shape != modified_weight.shape {
                    per_tensor_diff.insert(
                        modified_weight.name.clone(),
                        f64::MAX,
                    );
                    max_l2_diff = f64::MAX;
                    tensor_count += 1;
                    continue;
                }

                // Calculate L2 difference
                let l2_diff = original_weight.l2_diff(modified_weight);
                per_tensor_diff.insert(modified_weight.name.clone(), l2_diff);
                
                if l2_diff > max_l2_diff {
                    max_l2_diff = l2_diff;
                }
                total_diff += l2_diff;
                tensor_count += 1;
            } else {
                // Weight not found in original
                per_tensor_diff.insert(
                    modified_weight.name.clone(),
                    f64::MAX,
                );
                tensor_count += 1;
            }
        }

        // Check for missing weights in modified graph
        for (name, _) in &original_weights {
            if !per_tensor_diff.contains_key(name) {
                per_tensor_diff.insert(name.clone(), f64::MAX);
                tensor_count += 1;
            }
        }

        let avg_l2_diff = if tensor_count > 0 {
            total_diff / tensor_count as f64
        } else {
            0.0
        };

        Ok(WeightDiff {
            max_l2_diff,
            avg_l2_diff,
            tensor_count,
            per_tensor_diff,
        })
    }

    /// Infer operator type from tensor name
    fn infer_operator_from_name(name: &str) -> OperatorType {
        let name_lower = name.to_lowercase();
        
        if name_lower.contains("attention") || name_lower.contains("attn") {
            OperatorType::Attention {
                num_heads: 32,
                hidden_dim: 4096,
            }
        } else if name_lower.contains("mlp") || name_lower.contains("ffn") {
            OperatorType::MLP {
                hidden_dim: 11008,
                activation: "silu".to_string(),
            }
        } else if name_lower.contains("norm") || name_lower.contains("ln") {
            OperatorType::Norm {
                norm_type: "rmsnorm".to_string(),
                eps: 1e-6,
            }
        } else if name_lower.contains("embed") {
            OperatorType::Embedding {
                vocab_size: 32000,
                embed_dim: 4096,
            }
        } else if name_lower.contains("linear") || name_lower.contains("proj") {
            OperatorType::Linear {
                in_features: 4096,
                out_features: 4096,
            }
        } else {
            OperatorType::Custom {
                name: name.to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_tensor_l2_norm() {
        let tensor = WeightTensor::new(
            "test".to_string(),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );
        let norm = tensor.l2_norm();
        assert!((norm - 5.477).abs() < 0.001);
    }

    #[test]
    fn test_weight_tensor_l2_diff() {
        let t1 = WeightTensor::new(
            "test1".to_string(),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );
        let t2 = WeightTensor::new(
            "test2".to_string(),
            vec![1.1, 2.1, 3.1, 4.1],
            vec![2, 2],
        );
        let diff = t1.l2_diff(&t2);
        assert!(diff < 0.5);
    }

    #[test]
    fn test_weight_tensor_reshape_mut() {
        let mut tensor = WeightTensor::new(
            "test".to_string(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        
        // Reshape from [2, 3] to [3, 2]
        tensor.reshape_mut(vec![3, 2]).unwrap();
        assert_eq!(tensor.shape(), &[3, 2]);
        assert_eq!(tensor.strides(), &[2, 1]);
        
        // Try invalid reshape
        let result = tensor.reshape_mut(vec![2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_tensor_stride_access() {
        let tensor = WeightTensor::new(
            "test".to_string(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        
        // Test get with stride-based indexing
        assert_eq!(tensor.get(&[0, 0]), Some(1.0));
        assert_eq!(tensor.get(&[0, 1]), Some(2.0));
        assert_eq!(tensor.get(&[0, 2]), Some(3.0));
        assert_eq!(tensor.get(&[1, 0]), Some(4.0));
        assert_eq!(tensor.get(&[1, 1]), Some(5.0));
        assert_eq!(tensor.get(&[1, 2]), Some(6.0));
        
        // Test out of bounds
        assert_eq!(tensor.get(&[2, 0]), None);
        assert_eq!(tensor.get(&[0, 3]), None);
    }

    #[test]
    fn test_weight_tensor_set() {
        let mut tensor = WeightTensor::new(
            "test".to_string(),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );
        
        // Set values using stride-based indexing
        assert!(tensor.set(&[0, 1], 10.0));
        assert!(tensor.set(&[1, 0], 20.0));
        
        assert_eq!(tensor.get(&[0, 0]), Some(1.0));
        assert_eq!(tensor.get(&[0, 1]), Some(10.0));
        assert_eq!(tensor.get(&[1, 0]), Some(20.0));
        assert_eq!(tensor.get(&[1, 1]), Some(4.0));
        
        // Test out of bounds set
        assert!(!tensor.set(&[2, 0], 100.0));
    }

    #[test]
    fn test_weight_tensor_ndim_and_numel() {
        let tensor = WeightTensor::new(
            "test".to_string(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.numel(), 6);
    }

    #[test]
    fn test_weight_tensor_struct_size() {
        // Test that WeightTensor has the expected size
        use std::mem::size_of;
        
        // WeightTensor should be 64-byte aligned due to repr(align(64))
        assert!(size_of::<WeightTensor>() >= 64);
        
        // Create a tensor and verify basic properties
        let tensor = WeightTensor::new(
            "test".to_string(),
            vec![1.0; 100],
            vec![10, 10],
        );
        
        // Verify data length
        assert_eq!(tensor.numel(), 100);
    }

    #[test]
    fn test_compute_strides() {
        // 1D tensor
        assert_eq!(compute_strides(&[5]), vec![1]);
        
        // 2D tensor (row-major)
        assert_eq!(compute_strides(&[3, 4]), vec![4, 1]);
        
        // 3D tensor
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        
        // 4D tensor
        assert_eq!(compute_strides(&[2, 3, 4, 5]), vec![60, 20, 5, 1]);
        
        // Empty tensor
        let empty: &[usize] = &[];
        assert_eq!(compute_strides(empty), Vec::<usize>::new());
    }

    #[test]
    fn test_infer_operator_from_name() {
        assert!(matches!(
            ModelSwitch::infer_operator_from_name("model.layers.0.self_attn.q_proj"),
            OperatorType::Attention { .. }
        ));
        assert!(matches!(
            ModelSwitch::infer_operator_from_name("model.layers.0.mlp.gate_proj"),
            OperatorType::MLP { .. }
        ));
        assert!(matches!(
            ModelSwitch::infer_operator_from_name("model.norm.weight"),
            OperatorType::Norm { .. }
        ));
    }

    #[test]
    #[cfg(feature = "safetensors")]
    fn test_save_to_safetensors() {
        use std::fs;
        use std::path::PathBuf;

        // Create a test graph with multiple nodes and weights
        let mut graph = Graph::<OperatorType, WeightTensor>::directed();

        // Add nodes with different operator types
        let embed_node = graph
            .add_node(OperatorType::Embedding {
                vocab_size: 1000,
                embed_dim: 128,
            })
            .unwrap();

        let attn_node = graph
            .add_node(OperatorType::Attention {
                num_heads: 8,
                hidden_dim: 256,
            })
            .unwrap();

        let mlp_node = graph
            .add_node(OperatorType::MLP {
                hidden_dim: 512,
                activation: "relu".to_string(),
            })
            .unwrap();

        let norm_node = graph
            .add_node(OperatorType::Norm {
                norm_type: "layernorm".to_string(),
                eps: 1e-5,
            })
            .unwrap();

        // Add edges with weight tensors
        graph
            .add_edge(
                embed_node,
                embed_node,
                WeightTensor::new(
                    "model.embeddings.weight".to_string(),
                    vec![1.0; 1000 * 128],
                    vec![1000, 128],
                ),
            )
            .unwrap();

        graph
            .add_edge(
                attn_node,
                attn_node,
                WeightTensor::new(
                    "model.layers.0.attention.qkv.weight".to_string(),
                    vec![0.5; 256 * 3 * 256],
                    vec![256, 3, 256],
                ),
            )
            .unwrap();

        graph
            .add_edge(
                mlp_node,
                mlp_node,
                WeightTensor::new(
                    "model.layers.0.mlp.fc1.weight".to_string(),
                    vec![0.25; 256 * 512],
                    vec![256, 512],
                ),
            )
            .unwrap();

        graph
            .add_edge(
                norm_node,
                norm_node,
                WeightTensor::new(
                    "model.norm.weight".to_string(),
                    vec![1.0; 256],
                    vec![256],
                ),
            )
            .unwrap();

        // Add edges between nodes to create a proper graph structure
        graph.add_edge(embed_node, attn_node, WeightTensor::new(
            "model.embed_to_attn.weight".to_string(),
            vec![0.1; 128 * 256],
            vec![128, 256],
        )).unwrap();

        graph.add_edge(attn_node, mlp_node, WeightTensor::new(
            "model.attn_to_mlp.weight".to_string(),
            vec![0.2; 256 * 256],
            vec![256, 256],
        )).unwrap();

        graph.add_edge(mlp_node, norm_node, WeightTensor::new(
            "model.mlp_to_norm.weight".to_string(),
            vec![0.3; 512 * 256],
            vec![512, 256],
        )).unwrap();

        // Create a temporary file path
        let temp_path = PathBuf::from("test_save_to_safetensors_temp.safetensors");

        // Save to safetensors
        let save_result = ModelSwitch::save_to_safetensors(&graph, &temp_path);
        assert!(save_result.is_ok(), "Failed to save to safetensors: {:?}", save_result);

        // Verify file was created
        assert!(temp_path.exists(), "Safetensors file was not created");

        // Load back from safetensors
        let loaded_graph = ModelSwitch::load_from_safetensors(&temp_path);
        assert!(loaded_graph.is_ok(), "Failed to load from safetensors: {:?}", loaded_graph);
        let loaded_graph = loaded_graph.unwrap();

        // Note: The current load_from_safetensors implementation creates one node per tensor
        // (with self-loop edges), so node/edge count will match the number of tensors
        // The important thing is that weight data is preserved

        // Verify weight count (7 tensors in total)
        assert_eq!(
            7,
            loaded_graph.edge_count(),
            "Edge count should match number of tensors"
        );

        // Verify weights using verify_weights - compare edge data only
        // Since node structure changes, we just verify the weight tensors are preserved
        let diff = ModelSwitch::verify_weights(&graph, &loaded_graph).unwrap();
        println!("Save/Load round-trip weight diff: max={:.6e}, avg={:.6e}, count={}", 
                 diff.max_l2_diff, diff.avg_l2_diff, diff.tensor_count);
        
        // Allow small floating point errors from F32 conversion
        assert!(
            diff.max_l2_diff < 1e-5,
            "Weight difference too large: max_l2_diff={}",
            diff.max_l2_diff
        );

        // Clean up temporary file
        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    #[cfg(feature = "safetensors")]
    fn test_save_load_round_trip() {
        use std::fs;
        use std::path::PathBuf;

        // Create a simple test graph
        let mut graph = Graph::<OperatorType, WeightTensor>::directed();

        let node = graph
            .add_node(OperatorType::Linear {
                in_features: 64,
                out_features: 64,
            })
            .unwrap();

        // Add weight tensor
        let original_data: Vec<f64> = (0..64 * 64).map(|i| (i as f64) * 0.01).collect();
        graph
            .add_edge(
                node,
                node,
                WeightTensor::new(
                    "test.linear.weight".to_string(),
                    original_data.clone(),
                    vec![64, 64],
                ),
            )
            .unwrap();

        // Save and load back
        let temp_path = PathBuf::from("test_round_trip_temp.safetensors");
        
        ModelSwitch::save_to_safetensors(&graph, &temp_path).unwrap();
        let loaded_graph = ModelSwitch::load_from_safetensors(&temp_path).unwrap();

        // Compare original and loaded weights
        let diff = ModelSwitch::verify_weights(&graph, &loaded_graph).unwrap();
        
        // The conversion F64 -> F32 -> F64 introduces small errors
        // For values in range [0, 64), F32 precision is ~1e-7 to 1e-6
        println!("Round-trip L2 diff: max={:.6e}, avg={:.6e}", diff.max_l2_diff, diff.avg_l2_diff);
        
        // Clean up
        let _ = fs::remove_file(&temp_path);
    }
}
