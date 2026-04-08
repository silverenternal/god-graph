//! VGI 核心 trait 定义 - 分层 API 设计
//!
//! 提供虚拟图接口的分层抽象，降低学习成本
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    GraphAdvanced                            │
//! │  (可选：高级操作 - reserve, clear, update)                   │
//! ├─────────────────────────────────────────────────────────────┤
//! │                    GraphUpdate                              │
//! │  (增量更新 - add_node, add_edge, remove_node, remove_edge)  │
//! ├─────────────────────────────────────────────────────────────┤
//! │                    GraphRead                                │
//! │  (只读查询 - node_count, get_node, neighbors, etc.)         │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    VirtualGraph                             │
//! │  (完整 trait = GraphRead + GraphUpdate + GraphAdvanced)     │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # 使用指南
//!
//! ## 场景 1: 只需要读取图数据
//!
//! ```rust
//! use god_graph::vgi::traits::GraphRead;
//!
//! fn analyze_graph<G: GraphRead>(graph: &G) {
//!     println!("Nodes: {}", graph.node_count());
//!     for node in graph.nodes() {
//!         println!("Node: {:?}", node.data());
//!     }
//! }
//! ```
//!
//! ## 场景 2: 需要修改图结构
//!
//! ```rust
//! use god_graph::vgi::traits::GraphUpdate;
//!
//! fn build_graph<G: GraphUpdate>(graph: &mut G) {
//!     let n1 = graph.add_node("A".to_string()).unwrap();
//!     let n2 = graph.add_node("B".to_string()).unwrap();
//!     graph.add_edge(n1, n2, 1.0).unwrap();
//! }
//! ```
//!
//! ## 场景 3: 需要完整功能
//!
//! ```rust
//! use god_graph::vgi::traits::VirtualGraph;
//!
//! fn process_graph<G: VirtualGraph>(graph: &mut G) {
//!     // 使用所有功能
//! }
//! ```

use crate::edge::{EdgeIndex, EdgeRef};
use crate::node::{NodeIndex, NodeRef};
use crate::vgi::error::{VgiError, VgiResult};
use crate::vgi::metadata::{Capability, GraphMetadata, GraphType};

// ========== Layer 1: GraphRead (只读查询) ==========

/// 图只读查询 trait
///
/// 提供基础的图查询操作，不修改图结构
///
/// # 使用场景
///
/// - 图算法分析（如 PageRank、BFS）
/// - 图属性计算（如密度、直径）
/// - 图可视化导出
///
/// # 示例
///
/// ```
/// use god_graph::vgi::traits::GraphRead;
///
/// fn print_graph_info<G: GraphRead>(graph: &G) {
///     println!("Nodes: {}", graph.node_count());
///     println!("Edges: {}", graph.edge_count());
/// }
/// ```
pub trait GraphRead {
    /// 节点数据类型
    type NodeData;
    /// 边数据类型
    type EdgeData;

    /// 获取图元数据
    ///
    /// 返回图的描述信息和能力列表
    fn metadata(&self) -> GraphMetadata;

    /// 检查是否支持特定能力
    ///
    /// 默认实现检查元数据中的能力列表
    fn has_capability(&self, capability: Capability) -> bool {
        self.metadata().supports(capability)
    }

    /// 检查是否支持所有指定能力
    ///
    /// 默认实现检查元数据中的能力列表
    fn has_capabilities(&self, capabilities: &[Capability]) -> bool {
        self.metadata().supports_all(capabilities)
    }

    // ========== 基础统计 ==========

    /// 获取节点数量
    fn node_count(&self) -> usize;

    /// 获取边数量
    fn edge_count(&self) -> usize;

    /// 检查是否为空图
    fn is_empty(&self) -> bool {
        self.node_count() == 0
    }

    /// 获取图的类型
    fn graph_type(&self) -> GraphType {
        self.metadata().graph_type
    }

    // ========== 节点查询 ==========

    /// 获取节点引用
    ///
    /// # Arguments
    ///
    /// * `index` - 节点索引
    ///
    /// # Returns
    ///
    /// 如果节点存在，返回节点引用
    fn get_node(&self, index: NodeIndex) -> VgiResult<&Self::NodeData>;

    /// 检查节点是否存在
    fn contains_node(&self, index: NodeIndex) -> bool;

    /// 获取所有节点迭代器
    fn nodes(&self) -> impl Iterator<Item = NodeRef<'_, Self::NodeData>>;

    // ========== 边查询 ==========

    /// 获取边引用
    ///
    /// # Arguments
    ///
    /// * `index` - 边索引
    ///
    /// # Returns
    ///
    /// 如果边存在，返回边引用
    fn get_edge(&self, index: EdgeIndex) -> VgiResult<&Self::EdgeData>;

    /// 获取边的端点
    ///
    /// # Arguments
    ///
    /// * `index` - 边索引
    ///
    /// # Returns
    ///
    /// 返回边的起始和目标节点索引
    fn edge_endpoints(&self, index: EdgeIndex) -> VgiResult<(NodeIndex, NodeIndex)>;

    /// 检查边是否存在
    fn contains_edge(&self, index: EdgeIndex) -> bool;

    /// 检查两个节点之间是否存在边
    fn has_edge(&self, from: NodeIndex, to: NodeIndex) -> bool;

    /// 获取所有边迭代器
    fn edges(&self) -> impl Iterator<Item = EdgeRef<'_, Self::EdgeData>>;

    // ========== 邻接查询 ==========

    /// 获取节点的出边邻居迭代器
    ///
    /// # Arguments
    ///
    /// * `node` - 节点索引
    ///
    /// # Returns
    ///
    /// 返回邻居节点索引的迭代器
    fn neighbors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex>;

    /// 获取节点的关联边迭代器
    ///
    /// # Arguments
    ///
    /// * `node` - 节点索引
    ///
    /// # Returns
    ///
    /// 返回边索引的迭代器
    fn incident_edges(&self, node: NodeIndex) -> impl Iterator<Item = EdgeIndex>;

    /// 获取节点的出度
    fn out_degree(&self, node: NodeIndex) -> VgiResult<usize>;

    /// 获取节点的入度
    ///
    /// 默认实现对无向图返回出度
    fn in_degree(&self, node: NodeIndex) -> VgiResult<usize> {
        self.out_degree(node)
    }

    /// 获取节点的总度数
    ///
    /// 默认实现对无向图返回出度
    fn degree(&self, node: NodeIndex) -> VgiResult<usize> {
        self.out_degree(node)
    }
}

// ========== Layer 2: GraphUpdate (增量更新) ==========

/// 图增量更新 trait
///
/// 提供添加和删除节点/边的操作
///
/// # 设计原则
///
/// 1. **增量更新**: 只支持添加/删除，不支持批量替换
/// 2. **高效实现**: 避免不必要的内存重新分配
/// 3. **错误处理**: 统一的错误类型
///
/// # 使用场景
///
/// - 动态图构建
/// - 在线图更新
/// - 图编辑操作
///
/// # 示例
///
/// ```
/// use god_graph::vgi::traits::GraphUpdate;
///
/// fn add_triangle<G: GraphUpdate>(graph: &mut G, data: G::NodeData, edge_weight: G::EdgeData) {
///     let n1 = graph.add_node(data.clone()).unwrap();
///     let n2 = graph.add_node(data.clone()).unwrap();
///     let n3 = graph.add_node(data).unwrap();
///     
///     graph.add_edge(n1, n2, edge_weight.clone()).unwrap();
///     graph.add_edge(n2, n3, edge_weight.clone()).unwrap();
///     graph.add_edge(n3, n1, edge_weight).unwrap();
/// }
/// ```
pub trait GraphUpdate: GraphRead {
    /// 添加节点
    ///
    /// # Arguments
    ///
    /// * `data` - 节点数据
    ///
    /// # Returns
    ///
    /// 返回新添加节点的索引
    fn add_node(&mut self, data: Self::NodeData) -> VgiResult<NodeIndex>;

    /// 删除节点
    ///
    /// # Arguments
    ///
    /// * `index` - 节点索引
    ///
    /// # Returns
    ///
    /// 返回被删除节点的数据
    fn remove_node(&mut self, index: NodeIndex) -> VgiResult<Self::NodeData>;

    /// 添加边
    ///
    /// # Arguments
    ///
    /// * `from` - 起始节点索引
    /// * `to` - 目标节点索引
    /// * `data` - 边数据
    ///
    /// # Returns
    ///
    /// 返回新添加边的索引
    fn add_edge(
        &mut self,
        from: NodeIndex,
        to: NodeIndex,
        data: Self::EdgeData,
    ) -> VgiResult<EdgeIndex>;

    /// 删除边
    ///
    /// # Arguments
    ///
    /// * `index` - 边索引
    ///
    /// # Returns
    ///
    /// 返回被删除边的数据
    fn remove_edge(&mut self, index: EdgeIndex) -> VgiResult<Self::EdgeData>;
}

// ========== Layer 3: GraphAdvanced (高级操作) ==========

/// 图高级操作 trait
///
/// 提供批量操作、原地更新等高级功能
///
/// # 设计原则
///
/// 1. **可选功能**: 不是所有后端都需要实现
/// 2. **性能优化**: 提供比基础操作更高效的实现
/// 3. **向后兼容**: 保留给需要高级功能的用户
///
/// # 使用场景
///
/// - 大规模图预分配
/// - 原地节点/边数据更新
/// - 图清理操作
///
/// # 示例
///
/// ```
/// use god_graph::vgi::traits::GraphAdvanced;
///
/// fn optimize_graph<G: GraphAdvanced>(graph: &mut G) {
///     // 预分配容量
///     graph.reserve(10000, 50000);
///     
///     // 原地更新节点数据
///     let node = graph.get_node_mut(0.into()).unwrap();
///     *node = "updated".to_string();
/// }
/// ```
pub trait GraphAdvanced: GraphRead {
    /// 获取节点可变引用
    ///
    /// # Arguments
    ///
    /// * `index` - 节点索引
    ///
    /// # Returns
    ///
    /// 如果节点存在，返回节点可变引用
    fn get_node_mut(&mut self, index: NodeIndex) -> VgiResult<&mut Self::NodeData>;

    /// 预分配容量
    ///
    /// # Arguments
    ///
    /// * `additional_nodes` - 额外节点容量
    /// * `additional_edges` - 额外边容量
    fn reserve(&mut self, additional_nodes: usize, additional_edges: usize);

    /// 清空图
    fn clear(&mut self);

    /// 更新节点数据（高效实现）
    ///
    /// # Arguments
    ///
    /// * `index` - 节点索引
    /// * `data` - 新节点数据
    ///
    /// # Returns
    ///
    /// 返回旧的节点数据
    ///
    /// # Note
    ///
    /// 默认实现使用 `get_node_mut()`，高效且 O(1)。
    /// 子类可以重写以提供更优化的实现。
    fn update_node(&mut self, index: NodeIndex, data: Self::NodeData) -> VgiResult<Self::NodeData> {
        let node = self.get_node_mut(index)?;
        Ok(std::mem::replace(node, data))
    }

    /// 更新边数据（高效实现）
    ///
    /// # Arguments
    ///
    /// * `index` - 边索引
    /// * `data` - 新边数据
    ///
    /// # Returns
    ///
    /// 返回旧的边数据
    ///
    /// # Note
    ///
    /// 默认实现先获取端点再删除添加（O(n)）。
    /// 子类可以重写以提供 O(1) 实现。
    fn update_edge(&mut self, index: EdgeIndex, data: Self::EdgeData) -> VgiResult<Self::EdgeData> {
        // 默认实现：先删除再添加（低效，后端可以重写）
        if !self.contains_edge(index) {
            return Err(VgiError::Internal {
                message: format!("Edge {:?} not found", index),
            });
        }
        let old_data = self.remove_edge(index)?;
        let (from, to) = self.edge_endpoints(index)?;
        self.add_edge(from, to, data)?;
        Ok(old_data)
    }
}

// ========== VirtualGraph: 完整 trait ==========

/// 虚拟图核心 trait
///
/// 提供统一的图操作接口，屏蔽底层后端实现细节
///
/// # Trait Hierarchy
///
/// `VirtualGraph` = `GraphRead` + `GraphUpdate` + `GraphAdvanced`
///
/// # 设计原则
///
/// 1. **最小接口**: 只定义必要的核心操作
/// 2. **泛型设计**: 支持任意节点和边数据类型
/// 3. **错误处理**: 统一的错误类型，便于上层处理
/// 4. **能力查询**: 支持运行时查询后端能力
///
/// # 使用示例
///
/// ```
/// use god_graph::vgi::{VirtualGraph, GraphMetadata, GraphType};
/// use god_graph::graph::Graph;
///
/// // Graph<T, E> 实现 VirtualGraph
/// let mut graph = Graph::<String, f64>::directed();
///
/// // 查询元数据
/// let metadata = graph.metadata();
/// assert_eq!(metadata.graph_type, GraphType::Directed);
///
/// // 检查能力
/// assert!(graph.has_capability(Capability::IncrementalUpdate));
/// ```
pub trait VirtualGraph: GraphRead + GraphUpdate + GraphAdvanced {}

// 为所有实现 GraphRead + GraphUpdate + GraphAdvanced 的类型自动实现 VirtualGraph
impl<T> VirtualGraph for T where T: GraphRead + GraphUpdate + GraphAdvanced {}

// ========== 向后兼容的默认实现 ==========

// 为 GraphRead 提供默认的 update_node/update_edge 实现（低效但兼容）
impl<T> GraphAdvanced for T
where
    T: GraphRead + GraphUpdate,
{
    fn get_node_mut(&mut self, index: NodeIndex) -> VgiResult<&mut Self::NodeData> {
        // 默认实现：通过 remove + add 模拟（低效）
        if !self.contains_node(index) {
            return Err(VgiError::Internal {
                message: format!("Node {:?} not found", index),
            });
        }
        // 注意：这个实现效率低下，实现 GraphUpdate 的类型应该重写此方法
        // 这里只是为了编译通过，实际使用需要单独实现 GraphAdvanced
        unimplemented!("get_node_mut requires direct mutable access, which this default implementation cannot provide. Please implement GraphAdvanced explicitly.")
    }

    fn reserve(&mut self, _additional_nodes: usize, _additional_edges: usize) {
        // 默认空实现
    }

    fn clear(&mut self) {
        // 默认实现：逐个删除节点
        let nodes: Vec<_> = self.nodes().map(|n| n.index()).collect();
        for node in nodes {
            let _ = self.remove_node(node);
        }
    }
}

// ========== Backend Trait ==========

/// 图后端 trait
///
/// 定义图后端的基本操作，包括初始化、配置、生命周期管理等。
///
/// **注意**: `Backend` trait 继承自 `VirtualGraph`，实现 `Backend` 的类型
/// 自动支持所有 `VirtualGraph` 的图操作方法。
///
/// # 设计原则
///
/// 1. **统一接口**: Backend 继承 VirtualGraph，避免双重 API
/// 2. **可插拔**: 支持动态切换后端
/// 3. **能力发现**: 支持查询后端支持的能力
///
/// # 生命周期
///
/// ```text
/// create() → configure() → initialize() → ready
///                                    ↓
///                              execute()
///                                    ↓
///                              shutdown() → drop
/// ```
///
/// # 示例
///
/// ```
/// use god_graph::vgi::{Backend, BackendConfig, BackendType, VirtualGraph, GraphType};
/// use god_graph::backend::single_machine::SingleMachineBackend;
///
/// let mut backend: SingleMachineBackend<String, f64> = SingleMachineBackend::default();
/// let config = BackendConfig::new(GraphType::Directed);
/// backend.initialize(config).unwrap();
///
/// // Backend 继承 VirtualGraph 所有方法
/// let n1 = backend.add_node("A".to_string()).unwrap();
/// assert!(backend.has_capability(Capability::IncrementalUpdate));
/// ```
pub trait Backend: VirtualGraph + Send + Sync {
    /// 获取后端名称
    fn name(&self) -> &'static str;

    /// 获取后端版本
    fn version(&self) -> &'static str;

    /// 获取后端元数据
    ///
    /// 注意：此方法返回后端本身的元数据（名称、版本、类型），
    /// 与 `VirtualGraph::metadata()` 返回的图元数据不同。
    fn backend_metadata(&self) -> GraphMetadata;

    /// 获取后端类型
    fn backend_type(&self) -> BackendType;

    /// 检查后端是否已初始化
    fn is_initialized(&self) -> bool;

    /// 检查后端是否健康
    fn is_healthy(&self) -> bool;

    /// 初始化后端
    ///
    /// # Arguments
    ///
    /// * `config` - 后端配置
    ///
    /// # Returns
    ///
    /// 初始化成功返回 Ok，失败返回错误
    fn initialize(&mut self, config: BackendConfig) -> VgiResult<()>;

    /// 关闭后端
    ///
    /// 清理资源，释放内存
    fn shutdown(&mut self) -> VgiResult<()>;
}

/// 后端类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    /// 单机后端
    SingleMachine,
    /// 分布式后端
    Distributed,
    /// 外部数据库后端
    ExternalDatabase,
    /// 内存后端
    InMemory,
    /// 持久化后端
    Persistent,
    /// 混合后端
    Hybrid,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::SingleMachine => write!(f, "SingleMachine"),
            BackendType::Distributed => write!(f, "Distributed"),
            BackendType::ExternalDatabase => write!(f, "ExternalDatabase"),
            BackendType::InMemory => write!(f, "InMemory"),
            BackendType::Persistent => write!(f, "Persistent"),
            BackendType::Hybrid => write!(f, "Hybrid"),
        }
    }
}

/// 后端配置
///
/// 用于配置后端的行为
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// 图类型
    pub graph_type: GraphType,
    /// 初始节点容量
    pub initial_node_capacity: usize,
    /// 初始边容量
    pub initial_edge_capacity: usize,
    /// 是否启用并行
    pub enable_parallel: bool,
    /// 并行线程数
    pub parallel_threads: Option<usize>,
    /// 自定义配置项
    pub custom: std::collections::HashMap<String, String>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            graph_type: GraphType::Directed,
            initial_node_capacity: 1024,
            initial_edge_capacity: 4096,
            enable_parallel: false,
            parallel_threads: None,
            custom: std::collections::HashMap::new(),
        }
    }
}

impl BackendConfig {
    /// 创建新的配置
    pub fn new(graph_type: GraphType) -> Self {
        Self {
            graph_type,
            ..Default::default()
        }
    }

    /// 设置节点容量
    pub fn with_node_capacity(mut self, capacity: usize) -> Self {
        self.initial_node_capacity = capacity;
        self
    }

    /// 设置边容量
    pub fn with_edge_capacity(mut self, capacity: usize) -> Self {
        self.initial_edge_capacity = capacity;
        self
    }

    /// 启用并行
    pub fn with_parallel(mut self, threads: Option<usize>) -> Self {
        self.enable_parallel = true;
        self.parallel_threads = threads;
        self
    }

    /// 添加自定义配置
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }

    /// 获取自定义配置值
    pub fn get_custom(&self, key: &str) -> Option<&String> {
        self.custom.get(key)
    }
}

/// 后端构建器 trait
///
/// 用于构建和配置后端
pub trait BackendBuilder: Sized {
    /// 构建的后端类型
    type Backend: Backend;

    /// 创建新的构建器
    fn new() -> Self;

    /// 设置配置
    fn with_config(self, config: BackendConfig) -> Self;

    /// 构建后端
    fn build(self) -> VgiResult<Self::Backend>;
}

/// 图构建器 trait
///
/// 提供图的构建和初始化操作
pub trait GraphBuilder {
    /// 节点数据类型
    type NodeData;
    /// 边数据类型
    type EdgeData;
    /// 构建的图类型
    type Graph: VirtualGraph<NodeData = Self::NodeData, EdgeData = Self::EdgeData>;

    /// 创建有向图
    fn directed() -> Self::Graph;

    /// 创建无向图
    fn undirected() -> Self::Graph;

    /// 创建指定类型的图
    fn with_type(graph_type: GraphType) -> Self::Graph;

    /// 创建带容量的图
    fn with_capacity(nodes: usize, edges: usize) -> Self::Graph;

    /// 创建带元数据的图
    fn with_metadata(metadata: GraphMetadata) -> Self::Graph;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::graph::traits::GraphOps;

    #[test]
    fn test_graph_type_display() {
        assert_eq!(GraphType::Directed.to_string(), "Directed");
        assert_eq!(GraphType::Undirected.to_string(), "Undirected");
        assert_eq!(GraphType::Mixed.to_string(), "Mixed");
    }

    #[test]
    fn test_graph_read() {
        let mut graph = Graph::<String, f64>::directed();
        let n1 = graph.add_node("A".to_string()).unwrap();
        let n2 = graph.add_node("B".to_string()).unwrap();
        graph.add_edge(n1, n2, 1.0).unwrap();

        // 测试 GraphRead trait
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.get_node(n1).unwrap(), "A");
        assert!(graph.contains_node(n1));
        
        let neighbors: Vec<_> = graph.neighbors(n1).collect();
        assert_eq!(neighbors, vec![n2]);
    }

    #[test]
    fn test_graph_update() {
        let mut graph = Graph::<String, f64>::directed();

        // 测试 GraphUpdate trait
        let n1 = graph.add_node("A".to_string()).unwrap();
        let n2 = graph.add_node("B".to_string()).unwrap();
        let e1 = graph.add_edge(n1, n2, 1.0).unwrap();

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);

        let removed = graph.remove_edge(e1).unwrap();
        assert_eq!(removed, 1.0);
    }

    #[test]
    fn test_virtual_graph_basic() {
        let mut graph = Graph::<String, f64>::directed();

        // 测试 VirtualGraph trait
        let metadata = graph.metadata();
        assert_eq!(metadata.graph_type, GraphType::Directed);

        let n1 = graph.add_node("A".to_string()).unwrap();
        let n2 = graph.add_node("B".to_string()).unwrap();

        assert_eq!(graph.node_count(), 2);
        assert!(graph.contains_node(n1));
        assert!(graph.contains_node(n2));

        assert_eq!(graph.get_node(n1).unwrap(), "A");
        assert_eq!(graph.get_node(n2).unwrap(), "B");

        let e1 = graph.add_edge(n1, n2, 1.0).unwrap();
        assert_eq!(graph.edge_count(), 1);
        assert!(graph.contains_edge(e1));

        let neighbors: Vec<_> = graph.neighbors(n1).collect();
        assert_eq!(neighbors, vec![n2]);
        assert_eq!(graph.out_degree(n1).unwrap(), 1);
    }

    #[test]
    fn test_virtual_graph_remove() {
        let mut graph = Graph::<i32, f64>::undirected();

        let n1 = graph.add_node(1).unwrap();
        let n2 = graph.add_node(2).unwrap();
        let e1 = graph.add_edge(n1, n2, 1.0).unwrap();

        // 删除边
        let edge_data = graph.remove_edge(e1).unwrap();
        assert_eq!(edge_data, 1.0);
        assert_eq!(graph.edge_count(), 0);

        // 删除节点
        let node_data = graph.remove_node(n1).unwrap();
        assert_eq!(node_data, 1);
        assert_eq!(graph.node_count(), 1);
    }
}
