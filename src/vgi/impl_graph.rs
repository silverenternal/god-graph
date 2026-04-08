//! Graph<T, E> 实现 VGI 分层 trait
//!
//! 将现有的 Graph 结构集成到 VGI 分层体系中

use crate::edge::{EdgeIndex, EdgeRef};
use crate::graph::Graph;
use crate::node::{NodeIndex, NodeRef};
use crate::vgi::error::{VgiError, VgiResult};
use crate::vgi::metadata::{Capability, GraphMetadata, GraphType};
use crate::vgi::traits::{GraphAdvanced, GraphRead, GraphUpdate, VirtualGraph};

// ========== GraphRead 实现 ==========

impl<T, E> GraphRead for Graph<T, E>
where
    T: Clone,
    E: Clone,
{
    type NodeData = T;
    type EdgeData = E;

    fn metadata(&self) -> GraphMetadata {
        GraphMetadata::new("graph", GraphType::Directed)
            .with_node_count(self.node_count())
            .with_edge_count(self.edge_count())
            .with_capabilities(vec![
                Capability::IncrementalUpdate,
                Capability::DynamicMode,
                Capability::StaticMode,
                Capability::WeightedEdges,
                Capability::SelfLoops,
                Capability::NodeAttributes,
                Capability::EdgeAttributes,
            ])
    }

    fn has_capability(&self, capability: Capability) -> bool {
        matches!(
            capability,
            Capability::IncrementalUpdate
                | Capability::DynamicMode
                | Capability::StaticMode
                | Capability::WeightedEdges
                | Capability::SelfLoops
                | Capability::NodeAttributes
                | Capability::EdgeAttributes
        )
    }

    fn node_count(&self) -> usize {
        <Graph<T, E> as crate::graph::traits::GraphBase>::node_count(self)
    }

    fn edge_count(&self) -> usize {
        <Graph<T, E> as crate::graph::traits::GraphBase>::edge_count(self)
    }

    fn get_node(&self, index: NodeIndex) -> VgiResult<&Self::NodeData> {
        <Graph<T, E> as crate::graph::traits::GraphQuery>::get_node(self, index).ok_or_else(|| {
            VgiError::Internal {
                message: format!("Node {:?} not found", index),
            }
        })
    }

    fn contains_node(&self, index: NodeIndex) -> bool {
        <Graph<T, E> as crate::graph::traits::GraphBase>::contains_node(self, index)
    }

    fn nodes(&self) -> impl Iterator<Item = NodeRef<'_, Self::NodeData>> {
        <Graph<T, E> as crate::graph::traits::GraphQuery>::nodes(self)
    }

    fn get_edge(&self, index: EdgeIndex) -> VgiResult<&Self::EdgeData> {
        <Graph<T, E> as crate::graph::traits::GraphQuery>::get_edge(self, index).ok_or_else(|| {
            VgiError::Internal {
                message: format!("Edge {:?} not found", index),
            }
        })
    }

    fn edge_endpoints(&self, index: EdgeIndex) -> VgiResult<(NodeIndex, NodeIndex)> {
        <Graph<T, E> as crate::graph::traits::GraphQuery>::edge_endpoints(self, index).ok_or_else(
            || VgiError::Internal {
                message: format!("Edge {:?} not found", index),
            },
        )
    }

    fn contains_edge(&self, index: EdgeIndex) -> bool {
        <Graph<T, E> as crate::graph::traits::GraphBase>::contains_edge(self, index)
    }

    fn has_edge(&self, from: NodeIndex, to: NodeIndex) -> bool {
        <Graph<T, E> as crate::graph::traits::GraphQuery>::has_edge(self, from, to)
    }

    fn edges(&self) -> impl Iterator<Item = EdgeRef<'_, Self::EdgeData>> {
        <Graph<T, E> as crate::graph::traits::GraphQuery>::edges(self)
    }

    fn neighbors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> {
        <Graph<T, E> as crate::graph::traits::GraphQuery>::neighbors(self, node)
    }

    fn incident_edges(&self, node: NodeIndex) -> impl Iterator<Item = EdgeIndex> {
        <Graph<T, E> as crate::graph::traits::GraphQuery>::incident_edges(self, node)
    }

    fn out_degree(&self, node: NodeIndex) -> VgiResult<usize> {
        Ok(<Graph<T, E> as crate::graph::traits::GraphQuery>::out_degree(self, node))
    }
}

// ========== GraphUpdate 实现 ==========

impl<T, E> GraphUpdate for Graph<T, E>
where
    T: Clone,
    E: Clone,
{
    fn add_node(&mut self, data: Self::NodeData) -> VgiResult<NodeIndex> {
        <Graph<T, E> as crate::graph::traits::GraphOps>::add_node(self, data).map_err(|e| {
            VgiError::Internal {
                message: e.to_string(),
            }
        })
    }

    fn remove_node(&mut self, index: NodeIndex) -> VgiResult<Self::NodeData> {
        <Graph<T, E> as crate::graph::traits::GraphOps>::remove_node(self, index).map_err(|e| {
            VgiError::Internal {
                message: e.to_string(),
            }
        })
    }

    fn add_edge(
        &mut self,
        from: NodeIndex,
        to: NodeIndex,
        data: Self::EdgeData,
    ) -> VgiResult<EdgeIndex> {
        <Graph<T, E> as crate::graph::traits::GraphOps>::add_edge(self, from, to, data).map_err(
            |e| VgiError::Internal {
                message: e.to_string(),
            },
        )
    }

    fn remove_edge(&mut self, index: EdgeIndex) -> VgiResult<Self::EdgeData> {
        <Graph<T, E> as crate::graph::traits::GraphOps>::remove_edge(self, index).map_err(|e| {
            VgiError::Internal {
                message: e.to_string(),
            }
        })
    }
}

// ========== GraphAdvanced 实现 ==========

impl<T, E> GraphAdvanced for Graph<T, E>
where
    T: Clone,
    E: Clone,
{
    fn get_node_mut(&mut self, index: NodeIndex) -> VgiResult<&mut Self::NodeData> {
        if !self.contains_node(index) {
            return Err(VgiError::Internal {
                message: format!("Node {:?} not found", index),
            });
        }
        Ok(&mut self[index])
    }

    fn reserve(&mut self, additional_nodes: usize, additional_edges: usize) {
        <Graph<T, E> as crate::graph::traits::GraphOps>::reserve(
            self,
            additional_nodes,
            additional_edges,
        );
    }

    fn clear(&mut self) {
        <Graph<T, E> as crate::graph::traits::GraphOps>::clear(self);
    }

    fn update_node(&mut self, index: NodeIndex, data: Self::NodeData) -> VgiResult<Self::NodeData> {
        <Graph<T, E> as crate::graph::traits::GraphOps>::update_node(self, index, data).map_err(
            |e| VgiError::Internal {
                message: e.to_string(),
            },
        )
    }

    fn update_edge(&mut self, index: EdgeIndex, data: Self::EdgeData) -> VgiResult<Self::EdgeData> {
        <Graph<T, E> as crate::graph::traits::GraphOps>::update_edge(self, index, data).map_err(
            |e| VgiError::Internal {
                message: e.to_string(),
            },
        )
    }
}

// ========== VirtualGraph 自动实现 ==========
// Graph<T, E> 已经实现了 GraphRead + GraphUpdate + GraphAdvanced
// 因此自动实现 VirtualGraph

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_as_graph_read() {
        fn read_only_operation<G: GraphRead>(graph: &G) -> usize {
            graph.node_count()
        }

        let mut graph = Graph::<String, f64>::directed();
        graph.add_node("A".to_string()).unwrap();
        graph.add_node("B".to_string()).unwrap();

        assert_eq!(read_only_operation(&graph), 2);
        assert_eq!(graph.get_node(0.into()).unwrap(), "A");
    }

    #[test]
    fn test_graph_as_graph_update() {
        fn build_graph<G: GraphUpdate>(graph: &mut G) {
            let n1 = graph.add_node("A".to_string()).unwrap();
            let n2 = graph.add_node("B".to_string()).unwrap();
            graph.add_edge(n1, n2, 1.0).unwrap();
        }

        let mut graph = Graph::<String, f64>::directed();
        build_graph(&mut graph);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_graph_as_graph_advanced() {
        fn advanced_operation<G: GraphAdvanced>(graph: &mut G) {
            graph.reserve(100, 200);
            
            let n1 = graph.add_node("A".to_string()).unwrap();
            let node = graph.get_node_mut(n1).unwrap();
            *node = "Updated".to_string();
        }

        let mut graph = Graph::<String, f64>::directed();
        advanced_operation(&mut graph);

        assert_eq!(graph.get_node(0.into()).unwrap(), "Updated");
    }

    #[test]
    fn test_graph_as_virtual_graph() {
        let mut graph = Graph::<String, f64>::directed();

        // 测试 VirtualGraph trait（完整功能）
        let metadata = graph.metadata();
        assert_eq!(metadata.graph_type, GraphType::Directed);
        assert!(graph.has_capability(Capability::IncrementalUpdate));

        // 测试基本操作
        let n1 = graph.add_node("Node1".to_string()).unwrap();
        let n2 = graph.add_node("Node2".to_string()).unwrap();

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.get_node(n1).unwrap(), "Node1");

        // 测试边操作
        let e1 = graph.add_edge(n1, n2, 1.5).unwrap();
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.get_edge(e1).unwrap(), &1.5);

        // 测试邻居
        let neighbors: Vec<_> = graph.neighbors(n1).collect();
        assert_eq!(neighbors, vec![n2]);

        // 测试度数
        assert_eq!(graph.out_degree(n1).unwrap(), 1);

        // 测试更新
        let old = graph.update_node(n1, "Updated".to_string()).unwrap();
        assert_eq!(old, "Node1");
        assert_eq!(graph.get_node(n1).unwrap(), "Updated");

        // 测试删除
        let removed = graph.remove_node(n2).unwrap();
        assert_eq!(removed, "Node2");
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_graph_metadata() {
        let mut graph = Graph::<i32, f64>::undirected();

        for i in 0..10 {
            graph.add_node(i).unwrap();
        }

        let metadata = graph.metadata();
        assert_eq!(metadata.node_count, Some(10));
        assert_eq!(metadata.edge_count, Some(0));
    }

    #[test]
    fn test_graph_update_node_efficient() {
        let mut graph = Graph::<String, f64>::directed();
        let n1 = graph.add_node("A".to_string()).unwrap();

        // 使用高效的 update_node（GraphAdvanced）
        let old = graph.update_node(n1, "B".to_string()).unwrap();
        assert_eq!(old, "A");
        assert_eq!(graph.get_node(n1).unwrap(), "B");
    }
}
