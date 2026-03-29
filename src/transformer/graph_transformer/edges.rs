//! Edge types for graph-structured Transformer
//!
//! This module provides edge types that support tensor message passing
//! between nodes in the computation graph.
//!
//! ## Edge Tensor Passing Semantics
//!
//! Edges in the GraphTransformer carry tensor messages for efficient computation:
//!
//! - **SelfAttention edges**: Carry Q/K/V projection tensors for attention computation
//! - **DataFlow edges**: Carry activation tensors between layers
//! - **Residual edges**: Carry identity passthrough tensors for residual connections
//!
//! ## Example
//!
//! ```rust
//! use god_gragh::transformer::graph_transformer::edges::{GraphEdge, GraphEdgeType, SelfAttentionEdge};
//! use god_gragh::tensor::DenseTensor;
//! use god_gragh::tensor::traits::TensorBase;
//!
//! // Create Q/K/V projection tensors
//! let q_proj = DenseTensor::zeros(vec![1, 64]); // Query projection
//! let k_proj = DenseTensor::zeros(vec![1, 64]); // Key projection
//! let v_proj = DenseTensor::zeros(vec![1, 64]); // Value projection
//!
//! // Create self-attention edge with QKV message
//! let mut sa_edge = GraphEdge::self_attention_with_message(
//!     0, 1, 0.5, 2, 0, q_proj
//! );
//!
//! // Access the message tensor
//! if let Some(msg) = sa_edge.message() {
//!     println!("Message shape: {:?}", msg.shape());
//! }
//! ```

use crate::tensor::traits::TensorBase;
use crate::tensor::DenseTensor;

/// Type of graph edge
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphEdgeType {
    /// Self-attention edge
    SelfAttention,
    /// Data flow edge (residual connections, FFN input/output)
    DataFlow,
    /// Residual connection edge
    Residual,
}

/// Skip connection type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipType {
    /// Pre-normalization (norm before attention/FFN)
    PreNorm,
    /// Post-normalization (norm after attention/FFN)
    PostNorm,
}

/// Self-attention edge data
#[derive(Debug, Clone)]
pub struct SelfAttentionEdge {
    /// Attention weight
    pub weight: f64,
    /// Attention head
    pub head: usize,
    /// Layer number
    pub layer: usize,
    /// Message tensor (Q/K/V projections)
    /// For multi-head attention, this contains concatenated QKV projections
    pub message: Option<DenseTensor>,
    /// Optional separate K (key) projection
    pub key_proj: Option<DenseTensor>,
    /// Optional separate V (value) projection
    pub value_proj: Option<DenseTensor>,
}

impl SelfAttentionEdge {
    /// Create a new self-attention edge
    pub fn new(weight: f64, head: usize, layer: usize) -> Self {
        Self {
            weight,
            head,
            layer,
            message: None,
            key_proj: None,
            value_proj: None,
        }
    }

    /// Create with message tensor (Q projection)
    pub fn with_message(weight: f64, head: usize, layer: usize, message: DenseTensor) -> Self {
        Self {
            weight,
            head,
            layer,
            message: Some(message),
            key_proj: None,
            value_proj: None,
        }
    }

    /// Create with separate Q, K, V projections
    pub fn with_qkv(
        weight: f64,
        head: usize,
        layer: usize,
        q_proj: DenseTensor,
        k_proj: DenseTensor,
        v_proj: DenseTensor,
    ) -> Self {
        Self {
            weight,
            head,
            layer,
            message: Some(q_proj),
            key_proj: Some(k_proj),
            value_proj: Some(v_proj),
        }
    }

    /// Set the message tensor (Q projection)
    pub fn set_message(&mut self, message: DenseTensor) {
        self.message = Some(message);
    }

    /// Get the message tensor (Q projection)
    pub fn message(&self) -> Option<&DenseTensor> {
        self.message.as_ref()
    }

    /// Set the key projection
    pub fn set_key_proj(&mut self, key: DenseTensor) {
        self.key_proj = Some(key);
    }

    /// Get the key projection
    pub fn key_proj(&self) -> Option<&DenseTensor> {
        self.key_proj.as_ref()
    }

    /// Set the value projection
    pub fn set_value_proj(&mut self, value: DenseTensor) {
        self.value_proj = Some(value);
    }

    /// Get the value projection
    pub fn value_proj(&self) -> Option<&DenseTensor> {
        self.value_proj.as_ref()
    }

    /// Get all QKV projections if available
    pub fn get_qkv(
        &self,
    ) -> (
        Option<&DenseTensor>,
        Option<&DenseTensor>,
        Option<&DenseTensor>,
    ) {
        (
            self.message.as_ref(),
            self.key_proj.as_ref(),
            self.value_proj.as_ref(),
        )
    }

    /// Check if this edge has complete QKV projections
    pub fn has_qkv(&self) -> bool {
        self.message.is_some() && self.key_proj.is_some() && self.value_proj.is_some()
    }

    /// Compute attention score using Q and K projections
    /// score = Q @ K^T / sqrt(d_k)
    pub fn compute_attention_score(&self, d_k: f64) -> Option<f64> {
        if let (Some(q), Some(k)) = (&self.message, &self.key_proj) {
            if q.shape() == k.shape() && q.ndim() == 2 {
                // Simple dot-product attention
                let q_data = q.data();
                let k_data = k.data();

                let dot_product: f64 = q_data
                    .iter()
                    .zip(k_data.iter())
                    .map(|(&q_val, &k_val)| q_val * k_val)
                    .sum();

                Some(dot_product / d_k.sqrt())
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// Data flow edge data
#[derive(Debug, Clone)]
pub struct DataFlowEdge {
    /// Operation type
    pub operation: DataFlowOp,
    /// Layer number
    pub layer: usize,
    /// Message tensor being transferred
    pub message: Option<DenseTensor>,
}

/// Data flow operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFlowOp {
    /// Input to attention
    InputToAttention,
    /// Attention to output
    AttentionToOutput,
    /// Input to FFN
    InputToFFN,
    /// FFN to output
    FFNToOutput,
    /// Layer output to next layer
    LayerToLayer,
}

impl DataFlowEdge {
    /// Create a new data flow edge
    pub fn new(operation: DataFlowOp, layer: usize) -> Self {
        Self {
            operation,
            layer,
            message: None,
        }
    }

    /// Create with message tensor
    pub fn with_message(operation: DataFlowOp, layer: usize, message: DenseTensor) -> Self {
        Self {
            operation,
            layer,
            message: Some(message),
        }
    }

    /// Set the message tensor
    pub fn set_message(&mut self, message: DenseTensor) {
        self.message = Some(message);
    }

    /// Get the message tensor
    pub fn message(&self) -> Option<&DenseTensor> {
        self.message.as_ref()
    }
}

/// Residual connection edge data
#[derive(Debug, Clone)]
pub struct ResidualEdge {
    /// Layer number
    pub layer: usize,
    /// Skip type
    pub skip_type: SkipType,
    /// Residual tensor (identity passthrough)
    pub residual: Option<DenseTensor>,
}

impl ResidualEdge {
    /// Create a new residual edge
    pub fn new(layer: usize, skip_type: SkipType) -> Self {
        Self {
            layer,
            skip_type,
            residual: None,
        }
    }

    /// Create with residual tensor
    pub fn with_residual(layer: usize, skip_type: SkipType, residual: DenseTensor) -> Self {
        Self {
            layer,
            skip_type,
            residual: Some(residual),
        }
    }

    /// Set the residual tensor
    pub fn set_residual(&mut self, residual: DenseTensor) {
        self.residual = Some(residual);
    }

    /// Get the residual tensor
    pub fn residual(&self) -> Option<&DenseTensor> {
        self.residual.as_ref()
    }
}

/// Graph edge wrapper
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Edge type
    pub edge_type: GraphEdgeType,
    /// Source node ID
    pub source: usize,
    /// Target node ID
    pub target: usize,
    /// Optional self-attention data
    pub self_attention: Option<SelfAttentionEdge>,
    /// Optional data flow data
    pub data_flow: Option<DataFlowEdge>,
    /// Optional residual data
    pub residual: Option<ResidualEdge>,
}

impl GraphEdge {
    /// Create a self-attention edge
    pub fn self_attention(
        source: usize,
        target: usize,
        weight: f64,
        head: usize,
        layer: usize,
    ) -> Self {
        Self {
            edge_type: GraphEdgeType::SelfAttention,
            source,
            target,
            self_attention: Some(SelfAttentionEdge::new(weight, head, layer)),
            data_flow: None,
            residual: None,
        }
    }

    /// Create a data flow edge
    pub fn data_flow(source: usize, target: usize, operation: DataFlowOp, layer: usize) -> Self {
        Self {
            edge_type: GraphEdgeType::DataFlow,
            source,
            target,
            self_attention: None,
            data_flow: Some(DataFlowEdge::new(operation, layer)),
            residual: None,
        }
    }

    /// Create a residual edge
    pub fn residual(source: usize, target: usize, layer: usize, skip_type: SkipType) -> Self {
        Self {
            edge_type: GraphEdgeType::Residual,
            source,
            target,
            self_attention: None,
            data_flow: None,
            residual: Some(ResidualEdge::new(layer, skip_type)),
        }
    }

    /// Get the self-attention data if applicable
    pub fn get_self_attention(&self) -> Option<&SelfAttentionEdge> {
        self.self_attention.as_ref()
    }

    /// Get the data flow info if applicable
    pub fn get_data_flow(&self) -> Option<&DataFlowEdge> {
        self.data_flow.as_ref()
    }

    /// Get the residual info if applicable
    pub fn get_residual(&self) -> Option<&ResidualEdge> {
        self.residual.as_ref()
    }

    /// Get the layer number
    pub fn layer(&self) -> usize {
        if let Some(sa) = &self.self_attention {
            sa.layer
        } else if let Some(df) = &self.data_flow {
            df.layer
        } else if let Some(res) = &self.residual {
            res.layer
        } else {
            0
        }
    }

    /// Create a self-attention edge with message tensor
    pub fn self_attention_with_message(
        source: usize,
        target: usize,
        weight: f64,
        head: usize,
        layer: usize,
        message: DenseTensor,
    ) -> Self {
        Self {
            edge_type: GraphEdgeType::SelfAttention,
            source,
            target,
            self_attention: Some(SelfAttentionEdge::with_message(
                weight, head, layer, message,
            )),
            data_flow: None,
            residual: None,
        }
    }

    /// Create a data flow edge with message tensor
    pub fn data_flow_with_message(
        source: usize,
        target: usize,
        operation: DataFlowOp,
        layer: usize,
        message: DenseTensor,
    ) -> Self {
        Self {
            edge_type: GraphEdgeType::DataFlow,
            source,
            target,
            self_attention: None,
            data_flow: Some(DataFlowEdge::with_message(operation, layer, message)),
            residual: None,
        }
    }

    /// Create a residual edge with tensor
    pub fn residual_with_tensor(
        source: usize,
        target: usize,
        layer: usize,
        skip_type: SkipType,
        residual: DenseTensor,
    ) -> Self {
        Self {
            edge_type: GraphEdgeType::Residual,
            source,
            target,
            self_attention: None,
            data_flow: None,
            residual: Some(ResidualEdge::with_residual(layer, skip_type, residual)),
        }
    }

    /// Get the message tensor from this edge (if any)
    pub fn message(&self) -> Option<&DenseTensor> {
        match self.edge_type {
            GraphEdgeType::SelfAttention => self
                .self_attention
                .as_ref()
                .and_then(|sa| sa.message.as_ref()),
            GraphEdgeType::DataFlow => self.data_flow.as_ref().and_then(|df| df.message.as_ref()),
            GraphEdgeType::Residual => self.residual.as_ref().and_then(|r| r.residual.as_ref()),
        }
    }

    /// Set the message tensor on this edge
    pub fn set_message(&mut self, message: DenseTensor) -> bool {
        match self.edge_type {
            GraphEdgeType::SelfAttention => {
                if let Some(ref mut sa) = self.self_attention {
                    sa.set_message(message);
                    true
                } else {
                    false
                }
            }
            GraphEdgeType::DataFlow => {
                if let Some(ref mut df) = self.data_flow {
                    df.set_message(message);
                    true
                } else {
                    false
                }
            }
            GraphEdgeType::Residual => {
                if let Some(ref mut r) = self.residual {
                    r.set_residual(message);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Create a self-attention edge with separate Q, K, V projections
    pub fn self_attention_with_qkv(
        source: usize,
        target: usize,
        weight: f64,
        head: usize,
        layer: usize,
        q_proj: DenseTensor,
        k_proj: DenseTensor,
        v_proj: DenseTensor,
    ) -> Self {
        Self {
            edge_type: GraphEdgeType::SelfAttention,
            source,
            target,
            self_attention: Some(SelfAttentionEdge::with_qkv(
                weight, head, layer, q_proj, k_proj, v_proj,
            )),
            data_flow: None,
            residual: None,
        }
    }

    /// Get Q/K/V projections if available (SelfAttention edges only)
    pub fn get_qkv(
        &self,
    ) -> (
        Option<&DenseTensor>,
        Option<&DenseTensor>,
        Option<&DenseTensor>,
    ) {
        if let Some(sa) = &self.self_attention {
            sa.get_qkv()
        } else {
            (None, None, None)
        }
    }

    /// Check if this edge has complete QKV projections
    pub fn has_qkv(&self) -> bool {
        self.self_attention
            .as_ref()
            .is_some_and(|sa| sa.has_qkv())
    }

    /// Get the key projection (SelfAttention edges only)
    pub fn key_proj(&self) -> Option<&DenseTensor> {
        self.self_attention.as_ref().and_then(|sa| sa.key_proj())
    }

    /// Get the value projection (SelfAttention edges only)
    pub fn value_proj(&self) -> Option<&DenseTensor> {
        self.self_attention.as_ref().and_then(|sa| sa.value_proj())
    }

    /// Compute attention score using Q and K projections (SelfAttention edges only)
    pub fn compute_attention_score(&self, d_k: f64) -> Option<f64> {
        self.self_attention
            .as_ref()
            .and_then(|sa| sa.compute_attention_score(d_k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_attention_edge() {
        let edge = GraphEdge::self_attention(0, 1, 0.8, 2, 5);

        assert_eq!(edge.edge_type, GraphEdgeType::SelfAttention);
        assert_eq!(edge.source, 0);
        assert_eq!(edge.target, 1);

        let sa = edge.get_self_attention().unwrap();
        assert_eq!(sa.weight, 0.8);
        assert_eq!(sa.head, 2);
        assert_eq!(sa.layer, 5);
    }

    #[test]
    fn test_data_flow_edge() {
        let edge = GraphEdge::data_flow(10, 20, DataFlowOp::InputToAttention, 3);

        assert_eq!(edge.edge_type, GraphEdgeType::DataFlow);
        assert_eq!(edge.source, 10);
        assert_eq!(edge.target, 20);

        let df = edge.get_data_flow().unwrap();
        assert_eq!(df.operation, DataFlowOp::InputToAttention);
        assert_eq!(df.layer, 3);
    }

    #[test]
    fn test_residual_edge() {
        let edge = GraphEdge::residual(5, 15, 7, SkipType::PreNorm);

        assert_eq!(edge.edge_type, GraphEdgeType::Residual);
        assert_eq!(edge.source, 5);
        assert_eq!(edge.target, 15);

        let res = edge.get_residual().unwrap();
        assert_eq!(res.layer, 7);
        assert!(matches!(res.skip_type, SkipType::PreNorm));
    }

    #[test]
    fn test_edge_layer() {
        let sa_edge = GraphEdge::self_attention(0, 1, 0.5, 1, 10);
        assert_eq!(sa_edge.layer(), 10);

        let df_edge = GraphEdge::data_flow(0, 1, DataFlowOp::LayerToLayer, 5);
        assert_eq!(df_edge.layer(), 5);

        let res_edge = GraphEdge::residual(0, 1, 3, SkipType::PostNorm);
        assert_eq!(res_edge.layer(), 3);
    }

    #[test]
    fn test_tensor_message_passing() {
        use crate::tensor::traits::TensorBase;
        use crate::tensor::DenseTensor;

        // Create a message tensor
        let message = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        // Test self-attention edge with message
        let mut sa_edge = GraphEdge::self_attention_with_message(0, 1, 0.8, 2, 5, message.clone());
        assert!(sa_edge.message().is_some());
        assert_eq!(sa_edge.message().unwrap().shape(), &[2, 2]);

        // Test data flow edge with message
        let df_edge = GraphEdge::data_flow_with_message(
            10,
            20,
            DataFlowOp::InputToAttention,
            3,
            message.clone(),
        );
        assert!(df_edge.message().is_some());

        // Test residual edge with tensor
        let res_edge =
            GraphEdge::residual_with_tensor(5, 15, 7, SkipType::PreNorm, message.clone());
        assert!(res_edge.message().is_some());

        // Test set_message on existing edge
        let new_message = DenseTensor::from_vec(vec![5.0, 6.0], vec![2]);
        sa_edge.set_message(new_message.clone());
        assert!(sa_edge.message().is_some());
    }
}
