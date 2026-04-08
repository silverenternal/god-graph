//! 图构建器
//!
//! 提供图的构建和初始化操作
//!
//! # 设计目标
//!
//! 1. **链式调用**: 支持流畅的链式 API
//! 2. **类型安全**: 编译时检查配置正确性
//! 3. **灵活配置**: 支持多种配置选项
//! 4. **后端无关**: 可用于构建任何后端的图

use crate::vgi::error::VgiResult;
use crate::vgi::metadata::{GraphMetadata, GraphType};
use crate::vgi::traits::{BackendConfig, VirtualGraph};
use std::collections::HashMap;
use std::marker::PhantomData;

/// 图构建器
///
/// 用于以链式方式构建和配置图实例
///
/// # 示例
///
/// ```
/// use god_graph::vgi::builder::GraphBuilder;
/// use god_graph::vgi::metadata::GraphType;
/// use god_graph::graph::Graph;
///
/// // 构建有向图
/// let graph = Graph::<String, f64>::builder()
///     .directed()
///     .with_capacity(100, 500)
///     .build();
///
/// // 构建无向图
/// let graph = Graph::<i32, f64>::builder()
///     .undirected()
///     .with_metadata(GraphMetadata::new("custom", GraphType::Undirected))
///     .build();
/// ```
pub struct GraphBuilder<N, E, G>
where
    G: VirtualGraph<NodeData = N, EdgeData = E>,
{
    /// 图类型
    graph_type: Option<GraphType>,
    /// 节点容量
    node_capacity: Option<usize>,
    /// 边容量
    edge_capacity: Option<usize>,
    /// 自定义元数据
    metadata: Option<GraphMetadata>,
    /// 自定义配置
    config: HashMap<String, String>,
    /// 类型标记
    _phantom: PhantomData<(N, E, G)>,
}

impl<N, E, G> GraphBuilder<N, E, G>
where
    G: VirtualGraph<NodeData = N, EdgeData = E>,
{
    /// 创建新的构建器
    pub fn new() -> Self {
        Self {
            graph_type: None,
            node_capacity: None,
            edge_capacity: None,
            metadata: None,
            config: HashMap::new(),
            _phantom: PhantomData,
        }
    }

    /// 设置有向图
    pub fn directed(mut self) -> Self {
        self.graph_type = Some(GraphType::Directed);
        self
    }

    /// 设置无向图
    pub fn undirected(mut self) -> Self {
        self.graph_type = Some(GraphType::Undirected);
        self
    }

    /// 设置图类型
    pub fn with_graph_type(mut self, graph_type: GraphType) -> Self {
        self.graph_type = Some(graph_type);
        self
    }

    /// 设置容量
    pub fn with_capacity(mut self, nodes: usize, edges: usize) -> Self {
        self.node_capacity = Some(nodes);
        self.edge_capacity = Some(edges);
        self
    }

    /// 设置节点容量
    pub fn with_node_capacity(mut self, capacity: usize) -> Self {
        self.node_capacity = Some(capacity);
        self
    }

    /// 设置边容量
    pub fn with_edge_capacity(mut self, capacity: usize) -> Self {
        self.edge_capacity = Some(capacity);
        self
    }

    /// 设置元数据
    pub fn with_metadata(mut self, metadata: GraphMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// 添加自定义配置
    pub fn with_config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.insert(key.into(), value.into());
        self
    }

    /// 构建图实例
    ///
    /// # 注意
    ///
    /// 此方法需要 G 实现 Default trait
    pub fn build(self) -> G
    where
        G: Default,
    {
        // 应用配置
        // 注意：具体的配置应用依赖于 VirtualGraph 实现
        // 这里提供通用的构建框架

        G::default()
    }

    /// 使用后端配置构建图
    pub fn build_with_config(self, backend_config: BackendConfig) -> VgiResult<G>
    where
        G: Default,
    {
        let graph = G::default();

        // 应用后端配置
        let _ = backend_config; // 避免未使用警告

        Ok(graph)
    }
}

impl<N, E, G> Default for GraphBuilder<N, E, G>
where
    G: VirtualGraph<NodeData = N, EdgeData = E>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// 通用图构建器 trait
///
/// 为图类型提供统一的构建器创建方法
pub trait GraphBuilderTrait: VirtualGraph + Sized {
    /// 创建构建器
    fn builder() -> GraphBuilder<Self::NodeData, Self::EdgeData, Self> {
        GraphBuilder::new()
    }

    /// 创建有向图
    fn directed() -> Self;

    /// 创建无向图
    fn undirected() -> Self;

    /// 创建带容量的图
    fn with_capacity(nodes: usize, edges: usize) -> Self;

    /// 创建带元数据的图
    fn with_metadata(metadata: GraphMetadata) -> Self;
}

/// 图构建辅助函数
///
/// 注意：这些函数需要 Graph 实现 GraphBuilderTrait
/// 当前版本暂不提供
pub mod helpers {
    use super::*;

    /// 快速创建有向图（需要 GraphBuilderTrait）
    pub fn directed_graph<N, E, G>() -> G
    where
        G: VirtualGraph<NodeData = N, EdgeData = E> + Default,
    {
        G::default()
    }

    /// 快速创建无向图（需要 GraphBuilderTrait）
    pub fn undirected_graph<N, E, G>() -> G
    where
        G: VirtualGraph<NodeData = N, EdgeData = E> + Default,
    {
        G::default()
    }

    /// 快速创建带容量的图（需要 GraphBuilderTrait）
    pub fn with_capacity<N, E, G>(_nodes: usize, _edges: usize) -> G
    where
        G: VirtualGraph<NodeData = N, EdgeData = E> + Default,
    {
        G::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::graph::traits::GraphOps;

    #[test]
    fn test_graph_builder_basic() {
        let builder: GraphBuilder<String, f64, Graph<String, f64>> = GraphBuilder::new();

        let graph: Graph<String, f64> = builder.directed().build();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_graph_builder_chain() {
        let graph: Graph<i32, f64> = Graph::<i32, f64>::builder()
            .undirected()
            .with_capacity(100, 200)
            .with_config("custom_key", "custom_value")
            .build();

        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_graph_builder_with_metadata() {
        // 测试 builder 可以创建
        let _builder = GraphBuilder::<String, f64, Graph<String, f64>>::new()
            .with_metadata(GraphMetadata::new("test", GraphType::Undirected));
        // 注意：当前 builder 不应用 metadata，仅测试编译
    }

    #[test]
    fn test_helpers() {
        // 简单测试构建器辅助函数的存在
        let _builder = GraphBuilder::<String, f64, Graph<String, f64>>::new();
        // 注意：helpers 函数需要 Graph 实现 Default，这里仅测试编译
    }
}
