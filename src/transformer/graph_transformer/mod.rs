//! Graph-structured Transformer core module
//!
//! This module leverages god-gragh's graph structure to explicitly represent
//! the Transformer computation graph, enabling:
//! - Dynamic attention pruning (skip weak attention edges)
//! - Visualization of computation flow
//! - Dynamic graph modification (experimental)

pub mod nodes;
pub mod edges;
pub mod execution;

pub use nodes::{GraphNode, GraphNodeType, TokenEmbeddingNode, HiddenStateNode, AttentionOutputNode, FFNOutputNode};
pub use edges::{GraphEdge, GraphEdgeType, SelfAttentionEdge, DataFlowEdge, ResidualEdge, SkipType};
pub use execution::{GraphExecutor, GraphTransformer};
