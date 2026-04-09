//! Tensor 感知的节点和边类型
//!
//! 扩展现有的节点/边系统以原生支持 tensor 数据
//! 用于图神经网络（GNN）和其他机器学习应用

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

use crate::edge::EdgeIndex;
use crate::node::NodeIndex;
use crate::tensor::dense::DenseTensor;
use crate::tensor::traits::TensorBase;

#[cfg(feature = "tensor")]
use crate::tensor::sparse::SparseTensor;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Tensor 节点：带有 tensor 数据的节点包装器
///
/// 提供零成本抽象，与现有 NodeIndex 兼容
/// 支持任意实现 TensorBase trait 的 tensor 类型
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TensorNode<T: TensorBase> {
    /// 节点索引
    index: NodeIndex,
    /// Tensor 数据
    data: T,
    /// 类型标记
    _marker: PhantomData<T>,
}

impl<T: TensorBase> TensorNode<T> {
    /// 创建新的 TensorNode
    pub fn new(index: NodeIndex, data: T) -> Self {
        Self {
            index,
            data,
            _marker: PhantomData,
        }
    }

    /// 获取节点索引
    pub fn index(&self) -> NodeIndex {
        self.index
    }

    /// 获取 tensor 数据引用
    pub fn data(&self) -> &T {
        &self.data
    }

    /// 获取 tensor 数据可变引用
    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }

    /// 获取 tensor 的形状
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// 设置新的 tensor 数据
    pub fn set_data(&mut self, data: T) {
        self.data = data;
    }

    /// 消耗 self 并返回内部数据
    pub fn into_data(self) -> T {
        self.data
    }
}

impl<T: TensorBase> fmt::Debug for TensorNode<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorNode")
            .field("index", &self.index)
            .field("shape", &self.data.shape())
            .field("dtype", &self.data.dtype())
            .finish()
    }
}

impl<T: TensorBase> PartialEq for TensorNode<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T: TensorBase> Eq for TensorNode<T> {}

impl<T: TensorBase> Hash for TensorNode<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

/// Tensor 边：带有 tensor 数据的边包装器
///
/// 用于存储边特征（注意力权重、关系类型等）
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TensorEdge<E: TensorBase> {
    /// 边索引
    index: EdgeIndex,
    /// Tensor 数据
    data: E,
    /// 源节点索引
    source: NodeIndex,
    /// 目标节点索引
    target: NodeIndex,
}

impl<E: TensorBase> TensorEdge<E> {
    /// 创建新的 TensorEdge
    pub fn new(index: EdgeIndex, data: E, source: NodeIndex, target: NodeIndex) -> Self {
        Self {
            index,
            data,
            source,
            target,
        }
    }

    /// 获取边索引
    pub fn index(&self) -> EdgeIndex {
        self.index
    }

    /// 获取 tensor 数据引用
    pub fn data(&self) -> &E {
        &self.data
    }

    /// 获取 tensor 数据可变引用
    pub fn data_mut(&mut self) -> &mut E {
        &mut self.data
    }

    /// 获取源节点索引
    pub fn source(&self) -> NodeIndex {
        self.source
    }

    /// 获取目标节点索引
    pub fn target(&self) -> NodeIndex {
        self.target
    }

    /// 获取端点对
    pub fn endpoints(&self) -> (NodeIndex, NodeIndex) {
        (self.source, self.target)
    }

    /// 获取 tensor 的形状
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// 设置新的 tensor 数据
    pub fn set_data(&mut self, data: E) {
        self.data = data;
    }
}

impl<E: TensorBase> fmt::Debug for TensorEdge<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorEdge")
            .field("index", &self.index)
            .field(
                "endpoints",
                &format!("({:?}, {:?})", self.source, self.target),
            )
            .field("shape", &self.data.shape())
            .field("dtype", &self.data.dtype())
            .finish()
    }
}

impl<E: TensorBase> PartialEq for TensorEdge<E> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<E: TensorBase> Eq for TensorEdge<E> {}

impl<E: TensorBase> Hash for TensorEdge<E> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

/// 节点特征张量：用于存储节点的特征向量/矩阵
///
/// 这是 TensorNode 的便捷类型别名，使用 DenseTensor 作为默认后端
pub type NodeFeatures = TensorNode<DenseTensor>;

/// 边特征张量：用于存储边的特征（如注意力权重）
///
/// 这是 TensorEdge 的便捷类型别名，使用 DenseTensor 作为默认后端
pub type EdgeFeatures = TensorEdge<DenseTensor>;

/// 节点嵌入：用于图神经网络的节点表示
///
/// 通常是低维稠密向量
pub type NodeEmbedding = TensorNode<DenseTensor>;

/// 图神经网络中的隐藏状态
///
/// 用于存储 GNN 层的中间激活值
pub type HiddenState = DenseTensor;

/// 批量节点特征：用于 mini-batch 处理
///
/// 形状为 [batch_size, num_features] 或 [batch_size, num_nodes, num_features]
pub struct BatchedNodeFeatures<T: TensorBase> {
    /// 批量中的图索引
    pub graph_indices: Vec<usize>,
    /// 批量中的节点索引
    pub node_indices: Vec<NodeIndex>,
    /// 批量特征张量
    pub features: T,
}

impl<T: TensorBase> BatchedNodeFeatures<T> {
    /// 创建新的批量节点特征
    pub fn new(graph_indices: Vec<usize>, node_indices: Vec<NodeIndex>, features: T) -> Self {
        Self {
            graph_indices,
            node_indices,
            features,
        }
    }

    /// 获取批量大小
    pub fn batch_size(&self) -> usize {
        self.graph_indices.len()
    }

    /// 获取特征张量
    pub fn features(&self) -> &T {
        &self.features
    }

    /// 获取指定样本的特征
    pub fn get_sample(&self, sample_idx: usize) -> Option<&T> {
        if sample_idx < self.graph_indices.len() {
            Some(&self.features)
        } else {
            None
        }
    }
}

/// 图神经网络消息：在消息传递过程中使用
///
/// 包含源节点特征、边特征和目标节点特征
pub struct GNMessage<T: TensorBase> {
    /// 源节点特征
    pub source_features: T,
    /// 边特征（如果有）
    pub edge_features: Option<T>,
    /// 目标节点特征
    pub target_features: T,
}

impl<T: TensorBase> GNMessage<T> {
    /// 创建新的 GNN 消息
    pub fn new(source_features: T, edge_features: Option<T>, target_features: T) -> Self {
        Self {
            source_features,
            edge_features,
            target_features,
        }
    }

    /// 获取源节点特征
    pub fn source(&self) -> &T {
        &self.source_features
    }

    /// 获取边特征
    pub fn edge(&self) -> Option<&T> {
        self.edge_features.as_ref()
    }

    /// 获取目标节点特征
    pub fn target(&self) -> &T {
        &self.target_features
    }
}

/// 邻接矩阵表示：用于图神经网络计算
///
/// 使用稀疏张量格式存储图的邻接矩阵
#[cfg(feature = "tensor")]
pub struct AdjacencyMatrix {
    /// 邻接张量（稀疏格式）
    pub tensor: SparseTensor,
    /// 节点数量
    pub num_nodes: usize,
}

#[cfg(feature = "tensor")]
impl AdjacencyMatrix {
    /// 从边列表创建邻接矩阵
    pub fn from_edges(edges: &[(usize, usize, f64)], num_nodes: usize) -> Self {
        let tensor = SparseTensor::from_edges(edges, [num_nodes, num_nodes]);
        Self { tensor, num_nodes }
    }

    /// 获取非零元素数量
    pub fn nnz(&self) -> usize {
        self.tensor.nnz()
    }

    /// 转换为稀疏张量
    pub fn to_sparse(&self) -> SparseTensor {
        self.tensor.clone()
    }

    /// 转换为密集张量
    pub fn to_dense(&self) -> DenseTensor {
        self.tensor.to_dense()
    }
}

/// 度矩阵：对角矩阵，对角线元素为节点的度
pub struct DegreeMatrix {
    /// 度向量
    pub degrees: DenseTensor,
    /// 节点数量
    pub num_nodes: usize,
}

#[cfg(feature = "tensor")]
impl DegreeMatrix {
    /// 从邻接矩阵计算度矩阵
    pub fn from_adjacency(adj: &AdjacencyMatrix) -> Self {
        let degrees = vec![0.0; adj.num_nodes];
        let mut degrees_tensor = DenseTensor::new(degrees, vec![adj.num_nodes]);

        // 计算每个节点的度
        let coo = adj.tensor.to_coo();
        for &row in coo.row_indices() {
            let current = degrees_tensor.get(&[row]).unwrap();
            degrees_tensor.set(&[row], current + 1.0).unwrap();
        }

        Self {
            degrees: degrees_tensor,
            num_nodes: adj.num_nodes,
        }
    }

    /// 获取度向量
    pub fn degrees(&self) -> &DenseTensor {
        &self.degrees
    }

    /// 计算 D^(-1/2)（用于图卷积的归一化）
    pub fn inverse_sqrt(&self, epsilon: f64) -> DenseTensor {
        let shape = self.degrees.shape().to_vec();
        let inv_sqrt: Vec<f64> = self.degrees.data()
            .iter()
            .map(|&d| if d > epsilon { 1.0 / d.sqrt() } else { 0.0 })
            .collect();
        DenseTensor::new(inv_sqrt, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_node_creation() {
        let index = NodeIndex::new(0, 1);
        let data = DenseTensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let node = TensorNode::new(index, data.clone());

        assert_eq!(node.index(), index);
        assert_eq!(node.data(), &data);
        assert_eq!(node.shape(), &[3]);
    }

    #[test]
    fn test_tensor_edge_creation() {
        let index = EdgeIndex::new(0, 1);
        let source = NodeIndex::new(0, 1);
        let target = NodeIndex::new(1, 1);
        let data = DenseTensor::scalar(0.5);

        let edge = TensorEdge::new(index, data.clone(), source, target);

        assert_eq!(edge.index(), index);
        assert_eq!(edge.source(), source);
        assert_eq!(edge.target(), target);
        assert_eq!(edge.endpoints(), (source, target));
    }

    #[test]
    #[cfg(feature = "tensor")]
    fn test_adjacency_matrix() {
        let edges = vec![(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)];
        let adj = AdjacencyMatrix::from_edges(&edges, 3);

        assert_eq!(adj.num_nodes, 3);
        assert_eq!(adj.nnz(), 3);

        let dense = adj.to_dense();
        assert_eq!(dense.shape(), &[3, 3]);
        assert_eq!(dense.get(&[0, 1]).unwrap(), 1.0);
        assert_eq!(dense.get(&[0, 2]).unwrap(), 1.0);
    }

    #[test]
    #[cfg(feature = "tensor")]
    fn test_degree_matrix() {
        let edges = vec![(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)];
        let adj = AdjacencyMatrix::from_edges(&edges, 3);
        let degree = DegreeMatrix::from_adjacency(&adj);

        assert_eq!(degree.num_nodes, 3);
        // 节点 0 的度为 2 (出边：0->1, 0->2)
        // 节点 1 的度为 1 (出边：1->2)
        // 节点 2 的度为 0 (无出边)
        assert!((degree.degrees().get(&[0]).unwrap() - 2.0).abs() < 1e-10);
        assert!((degree.degrees().get(&[1]).unwrap() - 1.0).abs() < 1e-10);
        assert!((degree.degrees().get(&[2]).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gnn_message() {
        let src = DenseTensor::new(vec![1.0, 2.0], vec![2]);
        let edge = DenseTensor::scalar(0.5);
        let dst = DenseTensor::new(vec![3.0, 4.0], vec![2]);

        let msg = GNMessage::new(src.clone(), Some(edge.clone()), dst.clone());

        assert_eq!(msg.source().data(), &[1.0, 2.0]);
        assert_eq!(msg.edge().unwrap().data(), &[0.5]);
        assert_eq!(msg.target().data(), &[3.0, 4.0]);
    }
}
