//! 单机后端实现
//!
//! 基于现有 Graph<T, E> 的单机后端实现

use crate::backend::traits::{BackendBuilder, BackendConfig, BackendType};
use crate::edge::EdgeIndex;
use crate::graph::Graph;
use crate::node::NodeIndex;
use crate::vgi::error::{VgiError, VgiResult};
use crate::vgi::metadata::{Capability, GraphMetadata, GraphType};
use crate::vgi::traits::{Backend, VirtualGraph};

/// 单机后端
///
/// 基于 God-Graph 的 Graph 结构实现的单机后端
pub struct SingleMachineBackend<NodeData, EdgeData> {
    /// 内部图实例
    graph: Graph<NodeData, EdgeData>,
    /// 后端配置
    config: BackendConfig,
    /// 是否已初始化
    initialized: bool,
    /// 后端元数据
    metadata: GraphMetadata,
}

impl<NodeData, EdgeData> SingleMachineBackend<NodeData, EdgeData> {
    /// 获取内部图实例的引用
    pub fn inner(&self) -> &Graph<NodeData, EdgeData> {
        &self.graph
    }

    /// 获取内部图实例的可变引用
    pub fn inner_mut(&mut self) -> &mut Graph<NodeData, EdgeData> {
        &mut self.graph
    }

    /// 解构为内部图
    pub fn into_inner(self) -> Graph<NodeData, EdgeData> {
        self.graph
    }
}

impl<NodeData, EdgeData> Default for SingleMachineBackend<NodeData, EdgeData>
where
    NodeData: Clone + Send + Sync,
    EdgeData: Clone + Send + Sync,
{
    fn default() -> Self {
        Self {
            graph: Graph::directed(),
            config: BackendConfig::default(),
            initialized: false,
            metadata: GraphMetadata::new("single_machine", GraphType::Directed),
        }
    }
}

impl<NodeData, EdgeData> Backend for SingleMachineBackend<NodeData, EdgeData>
where
    NodeData: Clone + Send + Sync,
    EdgeData: Clone + Send + Sync,
{
    fn name(&self) -> &'static str {
        "single_machine"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn backend_metadata(&self) -> GraphMetadata {
        self.metadata.clone()
    }

    fn backend_type(&self) -> BackendType {
        BackendType::SingleMachine
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn is_healthy(&self) -> bool {
        self.initialized
    }

    fn initialize(&mut self, config: BackendConfig) -> VgiResult<()> {
        if self.initialized {
            return Err(VgiError::Internal {
                message: "Backend already initialized".to_string(),
            });
        }

        self.config = config.clone();
        self.graph =
            Graph::with_capacity(config.initial_node_capacity, config.initial_edge_capacity);

        // 构建元数据
        let mut metadata = GraphMetadata::new("single_machine", config.graph_type)
            .with_name("Single Machine Backend")
            .with_capability(Capability::IncrementalUpdate)
            .with_capability(Capability::DynamicMode)
            .with_capability(Capability::StaticMode)
            .with_capability(Capability::WeightedEdges)
            .with_capability(Capability::SelfLoops)
            .with_capability(Capability::NodeAttributes)
            .with_capability(Capability::EdgeAttributes);

        // 如果启用了并行，添加并行能力
        if config.enable_parallel {
            metadata = metadata.with_capability(Capability::Parallel);
        }

        self.metadata = metadata;
        self.initialized = true;

        Ok(())
    }

    fn shutdown(&mut self) -> VgiResult<()> {
        self.graph.clear();
        self.initialized = false;
        Ok(())
    }
}

impl<NodeData, EdgeData> VirtualGraph for SingleMachineBackend<NodeData, EdgeData>
where
    NodeData: Clone + Send + Sync,
    EdgeData: Clone + Send + Sync,
{
    type NodeData = NodeData;
    type EdgeData = EdgeData;

    fn metadata(&self) -> GraphMetadata {
        self.metadata.clone()
    }

    fn has_capability(&self, capability: Capability) -> bool {
        // 单机后端支持的能力
        matches!(
            capability,
            Capability::Parallel
                | Capability::IncrementalUpdate
                | Capability::DynamicMode
                | Capability::StaticMode
                | Capability::WeightedEdges
                | Capability::SelfLoops
                | Capability::NodeAttributes
                | Capability::EdgeAttributes
        )
    }

    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    fn add_node(&mut self, data: Self::NodeData) -> VgiResult<NodeIndex> {
        self.graph.add_node(data).map_err(|e| VgiError::Internal {
            message: e.to_string(),
        })
    }

    fn get_node(&self, index: NodeIndex) -> VgiResult<&Self::NodeData> {
        self.graph.get_node(index).map_err(|e| VgiError::Internal {
            message: e.to_string(),
        })
    }

    fn get_node_mut(&mut self, index: NodeIndex) -> VgiResult<&mut Self::NodeData> {
        self.graph
            .get_node_mut(index)
            .map_err(|e| VgiError::Internal {
                message: e.to_string(),
            })
    }

    fn remove_node(&mut self, index: NodeIndex) -> VgiResult<Self::NodeData> {
        self.graph
            .remove_node(index)
            .map_err(|e| VgiError::Internal {
                message: e.to_string(),
            })
    }

    fn contains_node(&self, index: NodeIndex) -> bool {
        self.graph.contains_node(index)
    }

    fn nodes(&self) -> impl Iterator<Item = crate::node::NodeRef<'_, Self::NodeData>> {
        self.graph.nodes()
    }

    fn add_edge(
        &mut self,
        from: NodeIndex,
        to: NodeIndex,
        data: Self::EdgeData,
    ) -> VgiResult<EdgeIndex> {
        self.graph
            .add_edge(from, to, data)
            .map_err(|e| VgiError::Internal {
                message: e.to_string(),
            })
    }

    fn get_edge(&self, index: EdgeIndex) -> VgiResult<&Self::EdgeData> {
        self.graph.get_edge(index).map_err(|e| VgiError::Internal {
            message: e.to_string(),
        })
    }

    fn edge_endpoints(&self, index: EdgeIndex) -> VgiResult<(NodeIndex, NodeIndex)> {
        self.graph
            .edge_endpoints(index)
            .map_err(|e| VgiError::Internal {
                message: e.to_string(),
            })
    }

    fn remove_edge(&mut self, index: EdgeIndex) -> VgiResult<Self::EdgeData> {
        self.graph
            .remove_edge(index)
            .map_err(|e| VgiError::Internal {
                message: e.to_string(),
            })
    }

    fn contains_edge(&self, index: EdgeIndex) -> bool {
        self.graph.contains_edge(index)
    }

    fn has_edge(&self, from: NodeIndex, to: NodeIndex) -> bool {
        self.graph.has_edge(from, to)
    }

    fn edges(&self) -> impl Iterator<Item = crate::edge::EdgeRef<'_, Self::EdgeData>> {
        self.graph.edges()
    }

    fn neighbors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> {
        self.graph.neighbors(node)
    }

    fn incident_edges(&self, node: NodeIndex) -> impl Iterator<Item = EdgeIndex> {
        self.graph.incident_edges(node)
    }

    fn out_degree(&self, node: NodeIndex) -> VgiResult<usize> {
        self.graph.out_degree(node).map_err(|e| VgiError::Internal {
            message: e.to_string(),
        })
    }

    fn reserve(&mut self, additional_nodes: usize, additional_edges: usize) {
        self.graph.reserve(additional_nodes, additional_edges);
    }

    fn clear(&mut self) {
        self.graph.clear();
    }

    fn update_node(&mut self, index: NodeIndex, data: Self::NodeData) -> VgiResult<Self::NodeData> {
        self.graph
            .update_node(index, data)
            .map_err(|e| VgiError::Internal {
                message: e.to_string(),
            })
    }

    fn update_edge(&mut self, index: EdgeIndex, data: Self::EdgeData) -> VgiResult<Self::EdgeData> {
        self.graph
            .update_edge(index, data)
            .map_err(|e| VgiError::Internal {
                message: e.to_string(),
            })
    }
}

/// 单机后端构建器
pub struct SingleMachineBuilder<NodeData, EdgeData> {
    config: BackendConfig,
    _phantom: std::marker::PhantomData<(NodeData, EdgeData)>,
}

impl<NodeData, EdgeData> Default for SingleMachineBuilder<NodeData, EdgeData> {
    fn default() -> Self {
        Self {
            config: BackendConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<NodeData, EdgeData> BackendBuilder for SingleMachineBuilder<NodeData, EdgeData>
where
    NodeData: Clone + Send + Sync,
    EdgeData: Clone + Send + Sync,
{
    type Backend = SingleMachineBackend<NodeData, EdgeData>;

    fn new() -> Self {
        Self::default()
    }

    fn with_config(mut self, config: BackendConfig) -> Self {
        self.config = config;
        self
    }

    fn build(self) -> VgiResult<Self::Backend> {
        let mut backend = SingleMachineBackend {
            graph: Graph::with_capacity(
                self.config.initial_node_capacity,
                self.config.initial_edge_capacity,
            ),
            config: self.config.clone(),
            initialized: true,
            metadata: GraphMetadata::new("single_machine", self.config.graph_type)
                .with_name("Single Machine Backend")
                .with_capabilities(vec![
                    Capability::IncrementalUpdate,
                    Capability::DynamicMode,
                    Capability::StaticMode,
                    Capability::WeightedEdges,
                    Capability::SelfLoops,
                    Capability::NodeAttributes,
                    Capability::EdgeAttributes,
                ]),
        };

        if self.config.enable_parallel {
            backend.metadata = backend.metadata.with_capability(Capability::Parallel);
        }

        Ok(backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vgi::VirtualGraph;

    #[test]
    fn test_single_machine_backend() {
        let config = BackendConfig::new(GraphType::Directed)
            .with_node_capacity(100)
            .with_edge_capacity(500);

        let mut backend: SingleMachineBackend<String, f64> = SingleMachineBackend::default();
        backend.initialize(config).unwrap();

        assert!(backend.is_initialized());
        assert_eq!(backend.backend_type(), BackendType::SingleMachine);
        assert!(backend.has_capability(Capability::IncrementalUpdate));
        assert!(!backend.has_capability(Capability::Distributed));

        // 测试 VirtualGraph 操作
        let n1 = backend.add_node("A".to_string()).unwrap();
        let n2 = backend.add_node("B".to_string()).unwrap();
        let e1 = backend.add_edge(n1, n2, 1.0).unwrap();

        assert_eq!(backend.node_count(), 2);
        assert_eq!(backend.edge_count(), 1);
        assert_eq!(backend.get_node(n1).unwrap(), "A");
        assert_eq!(backend.edge_endpoints(e1).unwrap(), (n1, n2));

        backend.shutdown().unwrap();
        assert!(!backend.is_initialized());
    }

    #[test]
    fn test_single_machine_builder() {
        let config = BackendConfig::new(GraphType::Undirected).with_parallel(Some(4));

        let backend: SingleMachineBackend<i32, f64> = SingleMachineBuilder::new()
            .with_config(config)
            .build()
            .unwrap();

        assert!(backend.is_initialized());
        assert!(backend.has_capability(Capability::Parallel));
        assert_eq!(backend.graph_type(), GraphType::Undirected);
    }
}
