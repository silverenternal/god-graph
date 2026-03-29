//! Tensor 模块：为图神经网络和高性能计算提供张量支持
//!
//! 本模块实现了 God-Graph 的 tensor 基础设施，包括：
//! - Dense tensor（密集张量）：基于 ndarray 的 N 维数组
//! - Sparse tensor（稀疏张量）：COO, CSR, BSR 格式
//! - Tensor 操作：矩阵乘法、转置、归约等
//! - 多后端支持：NdArray, Dfdx (GPU), Candle
//! - 内存池优化：减少分配开销
//! - 梯度检查点：降低反向传播内存占用
//!
//! ## 特性
//!
//! - **后端抽象**：通过 trait 系统支持多种 backend（ndarray, dfdx, candle）
//! - **稀疏格式**：COO（坐标格式）、CSR（压缩稀疏行）、BSR（块稀疏行）
//! - **SIMD 优化**：使用 wide crate 实现 SIMD 向量化
//! - **内存对齐**：64 字节缓存行对齐，避免 false sharing
//! - **内存池**：可复用的张量分配，适用于迭代算法
//!
//! ## 示例
//!
//! ```
//! # #[cfg(feature = "tensor")]
//! # {
//! use god_gragh::tensor::{DenseTensor, TensorBase};
//!
//! // 创建 2x3 密集张量
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let tensor = DenseTensor::from_vec(data, vec![2, 3]);
//!
//! assert_eq!(tensor.shape(), &[2, 3]);
//! assert_eq!(tensor.ndim(), 2);
//! # }
//! ```

#[cfg(feature = "tensor")]
pub mod traits;

#[cfg(feature = "tensor")]
pub mod dense;

#[cfg(feature = "tensor-sparse")]
pub mod sparse;

#[cfg(feature = "tensor")]
pub mod ops;

#[cfg(feature = "tensor")]
pub mod error;

#[cfg(feature = "tensor")]
pub mod types;

#[cfg(feature = "tensor")]
pub mod backend;

#[cfg(feature = "tensor-pool")]
pub mod pool;

#[cfg(feature = "tensor-gnn")]
pub mod gnn;

#[cfg(feature = "tensor")]
pub mod graph_tensor;

#[cfg(feature = "tensor")]
pub mod differentiable;

#[cfg(feature = "tensor")]
pub mod decomposition;

#[cfg(feature = "tensor")]
pub mod unified_graph;

// 重新导出核心类型
#[cfg(feature = "tensor")]
pub use traits::{COOView, DType, Device, SparseTensorOps, TensorBase, TensorOps};

#[cfg(feature = "tensor")]
pub use dense::DenseTensor;

#[cfg(feature = "tensor-sparse")]
pub use sparse::{COOTensor, CSRTensor, SparseTensor};

#[cfg(feature = "tensor")]
pub use error::TensorError;

#[cfg(feature = "tensor")]
pub use types::{EdgeFeatures, NodeFeatures, TensorEdge, TensorNode};

#[cfg(feature = "tensor-sparse")]
pub use types::AdjacencyMatrix;

#[cfg(feature = "tensor")]
pub use types::DegreeMatrix;

#[cfg(feature = "tensor")]
pub use backend::{NdArrayStorage, TensorStorage, UnifiedStorage};

#[cfg(feature = "tensor-pool")]
pub use pool::{ArenaStats, ArenaTensor, PoolConfig, PoolStats, PooledTensor, TensorArena, TensorPool};

#[cfg(feature = "tensor-autograd")]
pub use pool::GradientCheckpoint;

#[cfg(feature = "tensor-gnn")]
pub use gnn::{
    Aggregator, GATConv, GCNConv, GraphSAGE, IdentityMessage, LinearMessage, MaxAggregator,
    MeanAggregator, MessageFunction, MessagePassingLayer, SumAggregator, UpdateFunction,
};

#[cfg(feature = "tensor")]
pub use graph_tensor::{
    GraphFeatureExtractor, GraphReconstructor,
};

#[cfg(feature = "tensor-sparse")]
pub use graph_tensor::{
    GraphAdjacencyMatrix, GraphBatch, GraphTensorExt,
};

#[cfg(feature = "tensor")]
pub use differentiable::{
    DifferentiableEdge, DifferentiableGraph, DifferentiableNode, EdgeEditOp, EdgeEditPolicy,
    EditOperation, GradientConfig, GradientRecorder, GraphTransformer, GumbelSoftmaxSampler,
    NodeEditOp, StructureEdit, ThresholdEditPolicy,
};

#[cfg(feature = "tensor")]
pub use unified_graph::{EdgeData, NodeData, UnifiedConfig, UnifiedGraph};

#[cfg(feature = "tensor")]
pub use decomposition::{
    lie_algebra::{lie_exponential, lie_logarithm, skew_symmetric_projection},
    qr::{orthogonalize, qr_decompose},
    svd::{low_rank_approx, svd_decompose},
    tensor_ring::{compress_tensor_ring, tensor_ring_decompose, TensorRing},
};
