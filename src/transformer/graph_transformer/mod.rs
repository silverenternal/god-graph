//! Graph-structured Transformer core module
//!
//! This module leverages god-gragh's graph structure to explicitly represent
//! the Transformer computation graph, enabling:
//! - Dynamic attention pruning (skip weak attention edges)
//! - Visualization of computation flow
//! - Dynamic graph modification (experimental)

pub mod edges;
pub mod execution;
pub mod nodes;

pub use edges::{
    DataFlowEdge, GraphEdge, GraphEdgeType, ResidualEdge, SelfAttentionEdge, SkipType,
};
pub use execution::{GraphExecutor, GraphTransformer};
pub use nodes::{
    AttentionOutputNode, FFNOutputNode, GraphNode, GraphNodeType, HiddenStateNode,
    TokenEmbeddingNode,
};
